from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import service


HANDLE_METHODS = {
    "agent",
    "agent_many",
    "review",
    "checkpoint",
    "continue_checkpoint",
    "branch",
    "cancel",
    "complete",
    "wait",
    "snapshot",
    "graph",
    "describe_graph",
    "inspect",
    "nodes",
    "update_node",
    "remove_node",
    "replace_node",
    "add_dependency",
    "remove_dependency",
    "update_checkpoint",
    "apply_graph_patch",
}

NODE_METHODS = {"wait", "result", "inspect"}
GROUP_METHODS = {"wait", "completed", "failed"}


def runtime_functions(data_dir: Path) -> dict[str, Any]:
    base_dir = Path(data_dir)

    def start(
        objective: str,
        *,
        key: str | None = None,
        default_model_level: str | None = None,
        metadata: dict[str, Any] | None = None,
        state_dir: str | Path | None = None,
        context: Any = None,
    ) -> dict[str, Any]:
        wf = service.start(
            objective,
            key=key,
            default_model_level=default_model_level,
            metadata=metadata,
            state_dir=_base(base_dir, state_dir),
            parent_thread_id=getattr(context, "thread_id", None),
            parent_turn_id=getattr(context, "turn_id", None),
            parent_run_id=getattr(context, "run_id", None),
        )
        return _encode(wf)

    def resume(workflow_id: str, *, state_dir: str | Path | None = None) -> dict[str, Any]:
        return _encode(service.resume(workflow_id, state_dir=_base(base_dir, state_dir)))

    def list_workflows(
        status: str | None = None,
        limit: int = 20,
        *,
        state_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        return service.list(status=status, limit=limit, state_dir=_base(base_dir, state_dir))

    def agent(
        prompt: str,
        *,
        model_level: str | None = None,
        timeout_s: float | None = None,
        state_dir: str | Path | None = None,
        context: Any = None,
    ) -> dict[str, Any]:
        wf = service.start(
            service._objective_from_prompt(prompt),
            default_model_level=model_level,
            state_dir=_base(base_dir, state_dir),
            parent_thread_id=getattr(context, "thread_id", None),
            parent_turn_id=getattr(context, "turn_id", None),
            parent_run_id=getattr(context, "run_id", None),
        )
        return _encode(wf.agent(prompt, model_level=model_level, timeout_s=timeout_s))

    def active_snapshots(
        *,
        parent_thread_id: str | None = None,
        state_dir: str | Path | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return service.active_snapshots(parent_thread_id=parent_thread_id, state_dir=_base(base_dir, state_dir), limit=limit)

    def handle_call(
        workflow_id: str,
        method: str,
        *,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        state_dir: str | Path | None = None,
    ) -> Any:
        wf = service.resume(workflow_id, state_dir=_base(base_dir, state_dir))
        if method == "set_concurrency":
            wf._update_workflow_state({"concurrency": max(1, int((kwargs or {}).get("concurrency") or 1))})
            return {}
        if method not in HANDLE_METHODS:
            raise LookupError(f"Unsupported workflow method: {method}")
        return _encode(getattr(wf, method)(*(args or []), **dict(kwargs or {})))

    def node_call(
        workflow_id: str,
        node_id: str,
        method: str,
        *,
        key: str | None = None,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        state_dir: str | Path | None = None,
    ) -> Any:
        if method not in NODE_METHODS:
            raise LookupError(f"Unsupported workflow node method: {method}")
        node = service.NodeHandle(workflow_id, node_id, key, _base(base_dir, state_dir))
        return _encode(getattr(node, method)(*(args or []), **dict(kwargs or {})))

    def group_call(
        workflow_id: str,
        node_ids: list[str],
        method: str,
        *,
        key: str | None = None,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        state_dir: str | Path | None = None,
    ) -> Any:
        if method not in GROUP_METHODS:
            raise LookupError(f"Unsupported workflow node group method: {method}")
        group = service.NodeGroupHandle(workflow_id, list(node_ids), key, _base(base_dir, state_dir))
        return _encode(getattr(group, method)(*(args or []), **dict(kwargs or {})))

    return {
        "start": start,
        "resume": resume,
        "list": list_workflows,
        "agent": agent,
        "active_snapshots": active_snapshots,
        "handle_call": handle_call,
        "node_call": node_call,
        "group_call": group_call,
    }


def _base(default: Path, state_dir: str | Path | None) -> Path:
    return Path(state_dir).resolve() if state_dir is not None else default


def _encode(value: Any) -> Any:
    if isinstance(value, service.WorkflowHandle):
        return {"type": "workflow", "workflow_id": value.workflow_id}
    if isinstance(value, service.NodeHandle):
        return {"type": "node", "workflow_id": value.workflow_id, "node_id": value.node_id, "key": value.key}
    if isinstance(value, service.NodeGroupHandle):
        return {"type": "node_group", "workflow_id": value.workflow_id, "node_ids": list(value.node_ids), "key": value.key}
    if isinstance(value, service.CheckpointHandle):
        return {
            "type": "checkpoint",
            "workflow_id": value.workflow_id,
            "checkpoint_id": value.checkpoint_id,
            "node_id": value.node_id,
            "key": value.key,
        }
    if isinstance(value, (service.WorkflowWaitResult, service.NodeResult)):
        return asdict(value)
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    return value
