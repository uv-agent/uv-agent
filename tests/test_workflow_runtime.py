from __future__ import annotations

from pathlib import Path
from typing import Any

from uv_agent.builtin.workflow import service as workflow


def test_workflow_wait_polls_host_executor_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UV_AGENT_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("UV_AGENT_RUNTIME_THREAD_ID", "thr_parent")
    monkeypatch.setenv("UV_AGENT_RUNTIME_TURN_ID", "turn_parent")
    monkeypatch.setenv("UV_AGENT_RUNTIME_RUN_ID", "run_parent")

    # workflow.wait only observes persisted state; there is no subprocess-based
    # workflow-node execution hook left in the runtime helper.

    wf = workflow.start(objective="demo", default_model_level="deep")
    node = wf.agent("Investigate the demo", key="investigate")
    checkpoint = wf.checkpoint(key="after_investigation", after=node, reason="Review direction", options=["continue", "abort"])

    assert wf.wait(timeout_s=0.01).status == "timeout"

    now = workflow._utc_now_iso()
    with workflow._connect(tmp_path) as db:
        db.execute(
            """
            UPDATE workflow_nodes
            SET status = 'completed', completed_at = ?, thread_id = 'thr_node', result_summary = ?,
                result_json = ?, error_json = '{}'
            WHERE node_id = ?
            """,
            (now, "final node output\n", workflow._json_dumps({"thread_id": "thr_node", "status": "completed"}), node.node_id),
        )
        db.execute(
            "UPDATE workflow_nodes SET status = 'completed', completed_at = ? WHERE node_id = ?",
            (now, checkpoint.node_id),
        )
        db.execute("UPDATE workflow_checkpoints SET status = 'unresolved' WHERE checkpoint_id = ?", (checkpoint.checkpoint_id,))
        db.execute(
            "UPDATE workflows SET status = 'checkpoint', current_checkpoint_id = ?, updated_at = ? WHERE workflow_id = ?",
            (checkpoint.checkpoint_id, now, wf.workflow_id),
        )

    result = wf.wait(timeout_s=0.5)
    assert result.status == "checkpoint"
    assert "after_investigation" in result.summary()
    assert wf.inspect("investigate") == "final node output\n"
    assert wf.graph()["nodes"][0]["prompt"] == "Investigate the demo"
    assert "result" not in wf.graph()["nodes"][0]
    assert wf.graph(include_results=True)["nodes"][0]["result"]["thread_id"] == "thr_node"

    wf.continue_checkpoint("after_investigation", resolution={"action": "continue"})
    completed = wf.wait(timeout_s=0.5)

    assert completed.status == "completed"
    assert completed.summary() == "final node output\n"


def test_workflow_can_modify_pending_graph(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UV_AGENT_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("UV_AGENT_RUNTIME_THREAD_ID", "thr_parent")

    wf = workflow.start(objective="modify graph")
    first = wf.agent("old prompt", key="first")
    second = wf.agent("second", key="second", after=first)

    wf.update_node("second", prompt="new prompt", model_level="deep")
    wf.remove_dependency("second", "first")
    graph = wf.graph()

    node = next(item for item in graph["nodes"] if item["node_id"] == second.node_id)
    assert node["prompt"] == "new prompt"
    assert node["model_level"] == "deep"
    assert node["dependencies"] == []


def test_workflow_is_blocked_inside_workflow_node(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UV_AGENT_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("UV_AGENT_RUNTIME_THREAD_KIND", "workflow_node")

    try:
        workflow.start("nested")
    except RuntimeError as exc:
        assert "main Agent" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure output
        raise AssertionError("workflow.start should be blocked inside workflow nodes")
