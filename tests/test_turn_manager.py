from __future__ import annotations

import asyncio

import pytest

from uv_agent.turn_manager import TurnConflictError, TurnManager


class FakeEngine:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.release: dict[str, asyncio.Event] = {}
        self.calls: list[dict[str, object]] = []

    async def run_turn(self, *, user_text, user_inputs=None, thread_id=None, level=None, image_paths=None, cancel_event=None, guide_event=None):
        thread_id = thread_id or "thr_new"
        turn_id = f"turn_{len(self.calls) + 1}"
        texts = [item.text for item in user_inputs] if user_inputs is not None else [user_text]
        images = [path for item in (user_inputs or []) for path in item.image_paths]
        if user_inputs is None:
            images = list(image_paths or [])
        self.calls.append({"user_text": user_text, "user_texts": texts, "thread_id": thread_id, "image_paths": images})
        key = "|".join(texts)
        self.started.append(key)
        self.release[key] = asyncio.Event()
        yield {"type": "turn.started", "thread_id": thread_id, "turn_id": turn_id}
        while not self.release[key].is_set():
            if cancel_event is not None and cancel_event.is_set():
                yield {"type": "turn.interrupted", "thread_id": thread_id, "turn_id": turn_id, "reason": "user_interrupt"}
                return
            await asyncio.sleep(0.01)
        yield {"type": "turn.completed", "thread_id": thread_id, "turn_id": turn_id, "final_text": f"done {key}"}


async def wait_until(predicate, *, timeout=1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met")


@pytest.mark.asyncio
async def test_turn_manager_queues_same_thread_turns() -> None:
    engine = FakeEngine()
    manager = TurnManager(engine, max_concurrent_turns=2)

    first = await manager.submit_turn(user_text="first", thread_id="thr")
    second = await manager.submit_turn(user_text="second", thread_id="thr")

    await wait_until(lambda: engine.started == ["first"])
    engine.release["first"].set()
    await first.wait()
    await wait_until(lambda: engine.started == ["first", "second"])
    engine.release["second"].set()
    await second.wait()

    assert first.status == "completed"
    assert second.status == "completed"


@pytest.mark.asyncio
async def test_turn_manager_rejects_conflicting_turn() -> None:
    engine = FakeEngine()
    manager = TurnManager(engine)

    await manager.submit_turn(user_text="first", thread_id="thr")
    await wait_until(lambda: engine.started == ["first"])

    with pytest.raises(TurnConflictError):
        await manager.submit_turn(user_text="second", thread_id="thr", conflict="reject")

    engine.release["first"].set()


@pytest.mark.asyncio
async def test_turn_manager_takeover_absorbs_queued_messages_in_order() -> None:
    engine = FakeEngine()
    manager = TurnManager(engine)

    running = await manager.submit_turn(user_text="running", thread_id="thr")
    queued_a = await manager.submit_turn(user_text="queued a", thread_id="thr")
    queued_b = await manager.submit_turn(user_text="queued b", thread_id="thr")
    takeover = await manager.submit_turn(user_text="guide c", thread_id="thr", conflict="guide", image_paths=["c.png"])
    merged_queue = await manager.submit_turn(user_text="queue d", thread_id="thr")
    interrupt = await manager.submit_turn(user_text="interrupt e", thread_id="thr", conflict="interrupt", image_paths=["e.png"])

    assert queued_a.status == "merged"
    assert queued_b.status == "merged"
    assert merged_queue.status == "merged"
    assert interrupt.status == "merged"
    assert queued_a.merged_into == takeover.request_id
    assert [item.text for item in takeover.user_inputs] == ["queued a", "queued b", "guide c", "queue d", "interrupt e"]
    assert takeover.image_paths == ["c.png", "e.png"]

    await wait_until(lambda: engine.started and engine.started[0] == "running")
    # interrupt upgrades the takeover and cancels the active turn immediately.
    await running.wait()
    await wait_until(lambda: len(engine.started) == 2)
    assert engine.started[1] == "queued a|queued b|guide c|queue d|interrupt e"
    assert engine.calls[-1]["user_texts"] == ["queued a", "queued b", "guide c", "queue d", "interrupt e"]
    engine.release[engine.started[1]].set()
    await takeover.wait()
    assert takeover.status == "completed"
