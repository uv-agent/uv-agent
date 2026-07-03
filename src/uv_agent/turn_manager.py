from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from uv_agent.ids import new_id
from uv_agent.plugins.context import UserInput

if TYPE_CHECKING:
    from uv_agent.agent.engine import AgentEngine

TurnConflict = Literal["queue", "reject", "interrupt", "guide"]

logger = logging.getLogger(__name__)


class TurnConflictError(RuntimeError):
    """Raised when a turn request cannot be accepted under its conflict policy."""


@dataclass
class TurnHandle:
    """Observable handle for a submitted turn request.

    A handle represents a supervisor-level request. Requests that have not yet
    started may be merged into a later guide/interrupt takeover request; in that
    case no ThreadStore turn is created for the merged request, keeping model
    history aware only of the final merged user message.
    """

    request_id: str
    user_inputs: list[UserInput]
    thread_id: str | None
    level: str | None = None
    conflict: TurnConflict = "queue"
    status: Literal["queued", "running", "completed", "failed", "interrupted", "cancelled", "merged"] = "queued"
    turn_id: str | None = None
    final_text: str = ""
    error: BaseException | None = None
    merged_into: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    guide_event: asyncio.Event = field(default_factory=asyncio.Event)
    _queue: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)
    _done: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def user_text(self) -> str:
        return self.user_inputs[0].text if self.user_inputs else ""

    @property
    def image_paths(self) -> list[str | Path]:
        paths: list[str | Path] = []
        for item in self.user_inputs:
            paths.extend(item.image_paths)
        return paths

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def wait(self) -> "TurnHandle":
        await self._done.wait()
        return self

    def _emit(self, event: dict[str, Any]) -> None:
        self._queue.put_nowait(event)

    def _finish(self) -> None:
        if not self._done.is_set():
            self._queue.put_nowait(None)
            self._done.set()


@dataclass
class _ThreadState:
    active: TurnHandle | None = None
    queue: deque[TurnHandle] = field(default_factory=deque)
    takeover: TurnHandle | None = None
    takeover_mode: Literal["guide", "interrupt"] | None = None
    worker: asyncio.Task[None] | None = None


class TurnManager:
    """In-memory supervisor for agent turns.

    The manager serializes turns per thread, applies conflict policies, and
    limits global model concurrency. It deliberately does not persist queued
    requests; durable work should be represented by Scheduler or Workflow state.
    """

    def __init__(self, engine: "AgentEngine", *, max_concurrent_turns: int = 4) -> None:
        self.engine = engine
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrent_turns or 1)))
        self._lock = asyncio.Lock()
        self._threads: dict[str, _ThreadState] = {}
        self._detached_tasks: set[asyncio.Task[None]] = set()

    async def submit_turn(
        self,
        *,
        user_text: str,
        thread_id: str | None = None,
        level: str | None = None,
        image_paths: list[str | Path] | None = None,
        conflict: TurnConflict = "queue",
    ) -> TurnHandle:
        if conflict not in {"queue", "reject", "interrupt", "guide"}:
            raise ValueError(f"Unsupported turn conflict policy: {conflict!r}")
        handle = TurnHandle(
            request_id=new_id("req"),
            user_inputs=[UserInput(text=str(user_text), image_paths=tuple(image_paths or []))],
            thread_id=thread_id,
            level=level,
            conflict=conflict,
        )
        if thread_id is None:
            task = asyncio.create_task(self._run_detached(handle), name=f"uv-agent-turn-{handle.request_id}")
            self._detached_tasks.add(task)
            task.add_done_callback(self._detached_tasks.discard)
            logger.info("Turn submitted detached request_id=%s level=%s images=%d", handle.request_id, level, len(handle.image_paths))
            return handle

        async with self._lock:
            state = self._threads.setdefault(thread_id, _ThreadState())
            busy = state.active is not None or bool(state.queue) or state.takeover is not None
            if not busy:
                state.active = handle
                self._ensure_worker_locked(thread_id, state)
                logger.info("Turn submitted thread_id=%s request_id=%s conflict=%s status=active", thread_id, handle.request_id, conflict)
                return handle
            if conflict == "reject":
                logger.warning("Turn rejected thread_id=%s request_id=%s conflict=reject", thread_id, handle.request_id)
                raise TurnConflictError(f"Thread {thread_id} already has pending or running work")
            if conflict == "queue" and state.takeover is None:
                state.queue.append(handle)
                self._ensure_worker_locked(thread_id, state)
                logger.info(
                    "Turn queued thread_id=%s request_id=%s queue_depth=%d",
                    thread_id,
                    handle.request_id,
                    len(state.queue),
                )
                return handle
            if conflict == "queue":
                self._merge_into_takeover_locked(state, handle)
                logger.info("Turn merged into takeover thread_id=%s request_id=%s", thread_id, handle.request_id)
                return handle
            self._absorb_queue_into_takeover_locked(state, incoming=handle, mode=conflict)  # guide/interrupt
            if state.active is not None:
                if state.takeover_mode == "interrupt":
                    state.active.cancel_event.set()
                elif state.takeover_mode == "guide":
                    state.active.guide_event.set()
            self._ensure_worker_locked(thread_id, state)
            logger.info("Turn takeover submitted thread_id=%s request_id=%s mode=%s", thread_id, handle.request_id, conflict)
            return handle

    async def aclose(self) -> None:
        async with self._lock:
            active = [state.active for state in self._threads.values() if state.active is not None]
            workers = [state.worker for state in self._threads.values() if state.worker is not None]
            detached = list(self._detached_tasks)
        for handle in active:
            handle.cancel_event.set()
        for task in [*workers, *detached]:
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*[task for task in [*workers, *detached] if task is not None], return_exceptions=True)

    def _ensure_worker_locked(self, thread_id: str, state: _ThreadState) -> None:
        if state.worker is None or state.worker.done():
            state.worker = asyncio.create_task(self._thread_worker(thread_id), name=f"uv-agent-thread-worker-{thread_id}")

    def _absorb_queue_into_takeover_locked(
        self,
        state: _ThreadState,
        *,
        incoming: TurnHandle,
        mode: Literal["guide", "interrupt"],
    ) -> None:
        if state.takeover is None:
            state.takeover = incoming
            state.takeover_mode = mode
            pending = list(state.queue)
            state.queue.clear()
            if pending:
                merged_inputs: list[UserInput] = []
                for item in pending:
                    merged_inputs.extend(item.user_inputs)
                    self._mark_merged(item, state.takeover)
                merged_inputs.extend(incoming.user_inputs)
                state.takeover.user_inputs = merged_inputs
            return
        self._merge_into_takeover_locked(state, incoming)
        if mode == "interrupt":
            state.takeover_mode = "interrupt"
            state.takeover.conflict = "interrupt"

    def _merge_into_takeover_locked(self, state: _ThreadState, incoming: TurnHandle) -> None:
        takeover = state.takeover
        if takeover is None:
            raise RuntimeError("takeover buffer is not initialized")
        if incoming is not takeover:
            takeover.user_inputs.extend(incoming.user_inputs)
            self._mark_merged(incoming, takeover)

    @staticmethod
    def _mark_merged(handle: TurnHandle, target: TurnHandle) -> None:
        handle.status = "merged"
        handle.merged_into = target.request_id
        handle._emit({"type": "turn.merged", "request_id": handle.request_id, "merged_into": target.request_id})
        handle._finish()

    async def _thread_worker(self, thread_id: str) -> None:
        while True:
            async with self._lock:
                state = self._threads.setdefault(thread_id, _ThreadState())
                if state.active is not None and state.active.status == "queued":
                    handle = state.active
                elif state.active is not None:
                    return
                else:
                    handle = state.takeover if state.takeover is not None else (state.queue.popleft() if state.queue else None)
                    if handle is None:
                        state.worker = None
                        return
                    if handle is state.takeover:
                        state.takeover = None
                        state.takeover_mode = None
                    state.active = handle
            try:
                await self._run_handle(handle)
            finally:
                async with self._lock:
                    current = self._threads.setdefault(thread_id, _ThreadState())
                    if current.active is handle:
                        current.active = None
                    if current.queue or current.takeover is not None:
                        self._ensure_worker_locked(thread_id, current)
                    else:
                        current.worker = None
                        return

    async def _run_detached(self, handle: TurnHandle) -> None:
        await self._run_handle(handle)

    async def _run_handle(self, handle: TurnHandle) -> None:
        started = asyncio.get_running_loop().time()
        handle.status = "running"
        logger.debug("Turn handle running request_id=%s thread_id=%s conflict=%s", handle.request_id, handle.thread_id, handle.conflict)
        try:
            async with self._semaphore:
                async for event in self.engine.run_turn(
                    user_text=handle.user_text,
                    user_inputs=handle.user_inputs,
                    thread_id=handle.thread_id,
                    level=handle.level,
                    cancel_event=handle.cancel_event,
                    guide_event=handle.guide_event,
                ):
                    if event.get("type") == "turn.started":
                        handle.thread_id = str(event.get("thread_id") or handle.thread_id or "") or None
                        handle.turn_id = str(event.get("turn_id") or "") or None
                    elif event.get("type") == "turn.completed":
                        handle.status = "completed"
                        handle.final_text = str(event.get("final_text") or "")
                    elif event.get("type") == "turn.interrupted":
                        handle.status = "interrupted"
                    elif event.get("type") == "turn.error":
                        handle.status = "failed"
                    handle._emit(event)
        except asyncio.CancelledError:
            handle.status = "cancelled"
            handle._emit({"type": "turn.cancelled", "request_id": handle.request_id, "thread_id": handle.thread_id})
            raise
        except BaseException as exc:
            if exc.__class__.__name__ == "TurnInterrupted":
                handle.status = "interrupted"
                handle._emit(
                    {
                        "type": "turn.interrupted",
                        "request_id": handle.request_id,
                        "thread_id": handle.thread_id,
                        "turn_id": handle.turn_id,
                        "reason": "user_interrupt",
                    }
                )
            else:
                handle.status = "failed"
                handle.error = exc
                handle._emit(
                    {
                        "type": "turn.error",
                        "request_id": handle.request_id,
                        "thread_id": handle.thread_id,
                        "turn_id": handle.turn_id,
                        "error_type": exc.__class__.__name__,
                        "message": str(exc) or repr(exc),
                    }
                )
        finally:
            logger.info(
                "Turn handle finished request_id=%s thread_id=%s turn_id=%s status=%s duration_ms=%.1f",
                handle.request_id,
                handle.thread_id,
                handle.turn_id,
                handle.status,
                (asyncio.get_running_loop().time() - started) * 1000,
            )
            handle._finish()
