from __future__ import annotations

from pathlib import Path
from typing import Any

from uv_agent_runtime import workflow
from uv_agent_runtime.textops import CommandTextResult


def test_workflow_wait_reaches_checkpoint_and_resumes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UV_AGENT_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("UV_AGENT_RUNTIME_THREAD_ID", "thr_parent")
    monkeypatch.setenv("UV_AGENT_RUNTIME_TURN_ID", "turn_parent")
    monkeypatch.setenv("UV_AGENT_RUNTIME_RUN_ID", "run_parent")
    calls: list[list[str]] = []

    def fake_run_process_text(args: list[str], **kwargs: Any) -> CommandTextResult:
        calls.append(args)
        return CommandTextResult(args=args, returncode=0, stdout="final node output\n", stderr="[workflow-node-thread] thr_node\n")

    monkeypatch.setattr(workflow, "run_process_text", fake_run_process_text)

    wf = workflow.start(objective="demo", default_model_level="deep")
    node = wf.agent("Investigate the demo", key="investigate")
    wf.checkpoint(key="after_investigation", after=node, reason="Review direction", options=["continue", "abort"])

    result = wf.wait()

    assert result.status == "checkpoint"
    assert "after_investigation" in result.summary()
    assert wf.inspect("investigate") == "final node output\n"
    assert "workflow-node" in calls[0]
    assert "ask" not in calls[0]
    assert "--level" in calls[0]
    assert "deep" in calls[0]
    assert wf.graph()["nodes"][0]["prompt"] == "Investigate the demo"
    assert "result" not in wf.graph()["nodes"][0]
    assert wf.graph(include_results=True)["nodes"][0]["result"]["stdout"] == "final node output\n"

    wf.continue_checkpoint("after_investigation", resolution={"action": "continue"})
    completed = wf.wait()

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
