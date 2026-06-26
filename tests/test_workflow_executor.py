from __future__ import annotations

import asyncio
import json

import pytest

import uv_agent_runtime as rt
from uv_agent.session import ThreadStore
from uv_agent.state_db import connect_state_db
from uv_agent.workflow_executor import WorkflowExecutor


class FakeHandle:
    def __init__(self, *, turn_id: str = "turn_fake", status: str = "completed", final_text: str = "node done") -> None:
        self.turn_id = turn_id
        self.status = status
        self.final_text = final_text
        self.error = None

    async def wait(self):
        return self


class FakeTurnManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def submit_turn(self, **kwargs):
        self.calls.append(kwargs)
        return FakeHandle()


@pytest.mark.asyncio
async def test_workflow_executor_runs_ready_node_through_turn_manager(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("UV_AGENT_RUNTIME_STATE_DIR", str(tmp_path))
    wf = rt.workflow.start("host workflow", default_model_level="deep")
    node = wf.agent("Do host work", key="do")

    store = ThreadStore(tmp_path)
    turns = FakeTurnManager()
    executor = WorkflowExecutor(tmp_path, turns, store, lease_seconds=30)

    await executor.run_once()
    # Node execution is spawned in a task; allow it to finish.
    for _ in range(20):
        row = wf._node_by_id(node.node_id)
        if row["status"] == "completed":
            break
        await asyncio.sleep(0.01)

    row = wf._node_by_id(node.node_id)
    assert row["status"] == "completed"
    assert row["thread_id"].startswith("thr_")
    assert row["executor_id"] == executor.executor_id
    assert row["lease_until"] is None
    assert json.loads(row["result_json"])["turn_id"] == "turn_fake"
    assert turns.calls == [
        {"user_text": "Do host work", "thread_id": row["thread_id"], "level": "deep", "conflict": "queue"}
    ]
    assert store.thread_metadata(row["thread_id"])["kind"] == "workflow_node"


@pytest.mark.asyncio
async def test_workflow_executor_reaches_checkpoint_after_completed_dependency(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("UV_AGENT_RUNTIME_STATE_DIR", str(tmp_path))
    wf = rt.workflow.start("checkpoint workflow")
    node = wf.agent("Do work", key="do")
    checkpoint = wf.checkpoint(key="review", after=node, reason="Review")

    store = ThreadStore(tmp_path)
    executor = WorkflowExecutor(tmp_path, FakeTurnManager(), store)
    await executor.run_once()
    for _ in range(20):
        if wf._node_by_id(node.node_id)["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    await executor.run_once()

    result = wf.wait(timeout_s=0.1)
    assert result.status == "checkpoint"
    assert result.checkpoint and result.checkpoint["checkpoint_id"] == checkpoint.checkpoint_id


def test_workflow_executor_cleanup_only_expires_stale_leases(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("UV_AGENT_RUNTIME_STATE_DIR", str(tmp_path))
    wf = rt.workflow.start("stale workflow")
    stale = wf.agent("stale", key="stale")
    fresh = wf.agent("fresh", key="fresh")
    with connect_state_db(tmp_path) as db:
        db.execute(
            "UPDATE workflow_nodes SET status = 'running', executor_id = 'old', lease_until = '2000-01-01T00:00:00Z' WHERE node_id = ?",
            (stale.node_id,),
        )
        db.execute(
            "UPDATE workflow_nodes SET status = 'running', executor_id = 'other', lease_until = '2999-01-01T00:00:00Z' WHERE node_id = ?",
            (fresh.node_id,),
        )

    executor = WorkflowExecutor(tmp_path, FakeTurnManager(), ThreadStore(tmp_path))
    assert executor.cleanup_stale_running() == 1

    assert wf._node_by_id(stale.node_id)["status"] == "failed"
    assert wf._node_by_id(fresh.node_id)["status"] == "running"
