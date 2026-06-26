from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from uv_agent.state_db import SQLITE_BUSY_TIMEOUT_MS, SQLITE_TIMEOUT_SECONDS

from .events import EventBus, raise_if_reentrant_submit
from .helpers import RuntimeHelperRegistry, RuntimeHelperSpec, payload_from_call


@dataclass
class SubmittedTurn:
    thread_id: str
    turn_id: str
    _queue: asyncio.Queue[dict[str, Any] | None]

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event


@dataclass(frozen=True)
class TurnPrepareRequest:
    """Read-only information passed to pre-user context hooks."""

    thread_id: str
    turn_id: str
    user_text: str
    level: str | None = None
    is_new_thread: bool = False
    is_first_turn: bool = False
    created_at: str | None = None
    last_turn_completed_at: str | None = None
    last_assistant_completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnContextBlock:
    """A model-visible context block to insert before the current user message."""

    text: str
    placement: Literal["before_user"] = "before_user"
    dedupe_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    plugin: str = ""


class PluginThreadAPI:
    """Narrow thread mapping API exposed to plugins."""

    def __init__(self, *, plugin: str, thread_store) -> None:
        self._plugin = plugin
        self._thread_store = thread_store

    def get_external_thread(self, *, source: str, external_id: str) -> str | None:
        return self._thread_store.get_external_thread(
            owner_plugin=self._plugin,
            source=source,
            external_id=external_id,
        )

    def get_or_create_external_thread(
        self,
        *,
        source: str,
        external_id: str,
        title: str = "New thread",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._thread_store.get_or_create_external_thread(
            owner_plugin=self._plugin,
            source=source,
            external_id=external_id,
            title=title,
            metadata=metadata,
        )

    def metadata(self, thread_id: str) -> dict[str, Any]:
        return self._thread_store.thread_metadata(thread_id)

    def update_metadata(self, thread_id: str, metadata: dict[str, Any]) -> None:
        self._thread_store.update_thread_metadata(thread_id, updates=dict(metadata))

    def recent_events(self, thread_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        events, _ = self._thread_store.read_recent_events(thread_id, limit=limit)
        return events


class PluginContext:
    """Capabilities exposed to one uv-agent plugin."""

    def __init__(
        self,
        *,
        name: str,
        project_root: Path,
        user_state_dir: Path,
        data_dir: Path,
        log_dir: Path,
        config: dict[str, Any],
        events: EventBus,
        logger,
        helper_registry: RuntimeHelperRegistry,
        submitter: Callable[..., Any],
        task_factory: Callable[[str, Coroutine[Any, Any, Any], str | None], asyncio.Task[Any]],
        thread_store,
    ) -> None:
        self.name = name
        self.project_root = project_root
        self.user_state_dir = user_state_dir
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.config = config
        self.events = events
        self.logger = logger
        self._helper_registry = helper_registry
        self._submitter = submitter
        self._task_factory = task_factory
        self.threads = PluginThreadAPI(plugin=name, thread_store=thread_store)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def create_task(self, coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task[Any]:
        return self._task_factory(self.name, coro, name)

    def register_handler(
        self,
        name: str,
        fn: Callable[[dict[str, Any]], Any],
        *,
        doc: str,
        schema: dict[str, Any],
        timeout_s: float | None = None,
    ) -> RuntimeHelperSpec:
        return self._helper_registry.register_handler(
            plugin=self.name,
            name=name,
            fn=fn,
            doc=doc,
            schema=schema,
            timeout_s=timeout_s,
        )

    async def call_handler(self, name: str, payload: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return await self._helper_registry.call(name, payload_from_call([payload] if payload is not None else [], kwargs))

    def open_db(self) -> sqlite3.Connection:
        path = self.data_dir / "data.sqlite3"
        connection = sqlite3.connect(path, timeout=SQLITE_TIMEOUT_SECONDS)
        connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    async def submit_turn(
        self,
        *,
        text: str,
        thread_id: str | None = None,
        level: str | None = None,
        image_paths: list[Path] | None = None,
    ) -> SubmittedTurn:
        raise_if_reentrant_submit()
        return await self._submitter(
            text=text,
            thread_id=thread_id,
            level=level,
            image_paths=image_paths,
        )
