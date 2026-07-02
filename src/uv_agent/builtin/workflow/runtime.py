from __future__ import annotations

import builtins
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from uv_agent_runtime import transport


@dataclass(frozen=True)
class WorkflowWaitResult:
    workflow_id: str
    status: Literal["completed", "checkpoint", "failed", "timeout", "cancelled", "interrupted"]
    snapshot: dict[str, Any]
    checkpoint: dict[str, Any] | None = None
    final: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def summary(self) -> str:
        if self.status == "completed":
            if self.final and isinstance(self.final.get("output"), str):
                return self.final["output"]
            return "Workflow completed."
        if self.status == "checkpoint" and self.checkpoint:
            return _checkpoint_summary(self.workflow_id, self.checkpoint, self.snapshot)
        if self.status == "timeout":
            return _status_summary("timed out", self.workflow_id, self.snapshot)
        if self.status == "cancelled":
            return _status_summary("was cancelled", self.workflow_id, self.snapshot)
        if self.status == "failed":
            return _status_summary("failed", self.workflow_id, self.snapshot, self.error)
        return _status_summary(self.status, self.workflow_id, self.snapshot, self.error)


@dataclass(frozen=True)
class NodeResult:
    workflow_id: str
    node_id: str
    status: str
    output: str = ""
    error: dict[str, Any] | None = None
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.output

    def raise_for_error(self) -> "NodeResult":
        if self.status != "completed":
            raise RuntimeError(f"workflow node {self.node_id} did not complete: {self.error or self.status}")
        return self


@dataclass(frozen=True)
class NodeHandle:
    workflow_id: str
    node_id: str
    key: str | None = None
    _state_dir: str | Path | None = None

    def wait(self, *, timeout_s: float | None = None) -> NodeResult:
        return _node_result(_node_call(self, "wait", kwargs={"timeout_s": timeout_s}))

    def result(self) -> NodeResult | None:
        result = _node_call(self, "result")
        return _node_result(result) if isinstance(result, dict) else None

    def inspect(self) -> str | dict[str, Any]:
        return _node_call(self, "inspect")


@dataclass(frozen=True)
class NodeGroupHandle:
    workflow_id: str
    node_ids: list[str]
    key: str | None = None
    _state_dir: str | Path | None = None

    def wait(self, *, timeout_s: float | None = None) -> list[NodeResult]:
        return [_node_result(item) for item in _group_call(self, "wait", kwargs={"timeout_s": timeout_s})]

    def completed(self) -> list[NodeResult]:
        return [_node_result(item) for item in _group_call(self, "completed")]

    def failed(self) -> list[NodeResult]:
        return [_node_result(item) for item in _group_call(self, "failed")]


@dataclass(frozen=True)
class CheckpointHandle:
    workflow_id: str
    checkpoint_id: str
    node_id: str
    key: str
    _state_dir: str | Path | None = None


class WorkflowHandle:
    def __init__(self, workflow_id: str, *, state_dir: str | Path | None = None) -> None:
        self.workflow_id = workflow_id
        self._state_dir = state_dir

    def agent(
        self,
        prompt: str,
        *,
        key: str | None = None,
        after: Any = None,
        model_level: str | None = None,
        timeout_s: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> NodeHandle:
        return _node_handle(
            self._call(
                "agent",
                prompt,
                key=key,
                after=_dependency_ref(after),
                model_level=model_level,
                timeout_s=timeout_s,
                metadata=dict(metadata or {}),
            ),
            self._state_dir,
        )

    def agent_many(
        self,
        items: Iterable[Any],
        *,
        key: str | None = None,
        prompt: str | Callable[[Any], str] | None = None,
        concurrency: int | None = None,
        after: Any = None,
        model_level: str | None = None,
    ) -> NodeGroupHandle:
        item_list = builtins.list(items)
        if callable(prompt):
            if concurrency is not None:
                self._call("set_concurrency", concurrency=concurrency)
            nodes = [
                self.agent(
                    str(prompt(item)),
                    key=f"{key}.{index}" if key else None,
                    after=after,
                    model_level=model_level,
                    metadata={"group_key": key, "item": item, "index": index},
                )
                for index, item in enumerate(item_list, start=1)
            ]
            return NodeGroupHandle(self.workflow_id, [node.node_id for node in nodes], key, self._state_dir)
        return _node_group_handle(
            self._call(
                "agent_many",
                item_list,
                key=key,
                prompt=prompt,
                concurrency=concurrency,
                after=_dependency_ref(after),
                model_level=model_level,
            ),
            self._state_dir,
        )

    def review(
        self,
        *,
        key: str | None = None,
        checkpoint: str | None = None,
        prompt: str,
        model_level: str | None = None,
        after: Any = None,
    ) -> NodeHandle:
        return _node_handle(
            self._call(
                "review",
                key=key,
                checkpoint=checkpoint,
                prompt=prompt,
                model_level=model_level,
                after=_dependency_ref(after),
            ),
            self._state_dir,
        )

    def checkpoint(
        self,
        *,
        key: str,
        reason: str,
        after: Any = None,
        options: Sequence[str] | None = None,
        recommended_action: str | None = None,
    ) -> CheckpointHandle:
        return _checkpoint_handle(
            self._call(
                "checkpoint",
                key=key,
                reason=reason,
                after=_dependency_ref(after),
                options=builtins.list(options or []),
                recommended_action=recommended_action,
            ),
            self._state_dir,
        )

    def continue_checkpoint(self, checkpoint: str, *, resolution: Mapping[str, Any] | None = None) -> None:
        self._call("continue_checkpoint", checkpoint, resolution=dict(resolution or {}))

    def branch(
        self,
        *,
        key: str,
        from_checkpoint: str | None = None,
        tasks: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> NodeGroupHandle:
        return _node_group_handle(
            self._call(
                "branch",
                key=key,
                from_checkpoint=from_checkpoint,
                tasks=builtins.list(tasks or []),
                metadata=dict(metadata or {}),
            ),
            self._state_dir,
        )

    def cancel(self, reason: str | None = None) -> None:
        self._call("cancel", reason)

    def complete(self, result: Any = None) -> None:
        self._call("complete", result)

    def wait(self, *, timeout_s: float | None = None, until: str = "next_yield") -> WorkflowWaitResult:
        return _wait_result(self._call("wait", timeout_s=timeout_s, until=until))

    def snapshot(self) -> dict[str, Any]:
        return self._call("snapshot")

    def graph(self, *, include_results: bool = False) -> dict[str, Any]:
        return self._call("graph", include_results=include_results)

    def describe_graph(self) -> str:
        return str(self._call("describe_graph"))

    def inspect(self, node: str) -> str | dict[str, Any]:
        return self._call("inspect", node)

    def nodes(self, *, status: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        return self._call("nodes", status=status, kind=kind)

    def update_node(self, node: str, **patch: Any) -> NodeHandle:
        if "dependencies" in patch:
            patch = {**patch, "dependencies": _dependency_ref(patch["dependencies"])}
        return _node_handle(self._call("update_node", node, **patch), self._state_dir)

    def remove_node(self, node: str, *, cascade: bool = False) -> None:
        self._call("remove_node", node, cascade=cascade)

    def replace_node(self, node: str, *, kind: str | None = None, prompt: str | None = None, dependencies: Any = None, **patch: Any) -> NodeHandle:
        return _node_handle(
            self._call("replace_node", node, kind=kind, prompt=prompt, dependencies=_dependency_ref(dependencies), **patch),
            self._state_dir,
        )

    def add_dependency(self, node: str, depends_on: str) -> None:
        self._call("add_dependency", node, depends_on)

    def remove_dependency(self, node: str, depends_on: str) -> None:
        self._call("remove_dependency", node, depends_on)

    def update_checkpoint(self, checkpoint: str, **patch: Any) -> CheckpointHandle:
        return _checkpoint_handle(self._call("update_checkpoint", checkpoint, **patch), self._state_dir)

    def apply_graph_patch(self, patch: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("apply_graph_patch", dict(patch))

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        call_kwargs: dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "method": method,
            "args": builtins.list(args),
            "kwargs": kwargs,
        }
        if self._state_dir is not None:
            call_kwargs["state_dir"] = str(self._state_dir)
        return transport.call_host("workflow.handle_call", **call_kwargs)


def start(
    objective: str,
    *,
    key: str | None = None,
    default_model_level: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    state_dir: str | Path | None = None,
) -> WorkflowHandle:
    _assert_main_thread()
    return _workflow_handle(
        transport.call_host(
            "workflow.start",
            objective,
            key=key,
            default_model_level=default_model_level,
            metadata=dict(metadata or {}),
            state_dir=str(state_dir) if state_dir is not None else None,
        ),
        state_dir,
    )


def resume(workflow_id: str, *, state_dir: str | Path | None = None) -> WorkflowHandle:
    return _workflow_handle(
        transport.call_host("workflow.resume", workflow_id=workflow_id, state_dir=str(state_dir) if state_dir is not None else None),
        state_dir,
    )


def list(status: str | None = None, limit: int = 20, *, state_dir: str | Path | None = None) -> list[dict[str, Any]]:  # noqa: A001
    return transport.call_host("workflow.list", status=status, limit=limit, state_dir=str(state_dir) if state_dir is not None else None)


def agent(prompt: str, *, model_level: str | None = None, timeout_s: float | None = None) -> NodeHandle:
    return _node_handle(transport.call_host("workflow.agent", prompt=prompt, model_level=model_level, timeout_s=timeout_s), None)


def active_snapshots(
    *,
    parent_thread_id: str | None = None,
    state_dir: str | Path | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    return transport.call_host(
        "workflow.active_snapshots",
        parent_thread_id=parent_thread_id,
        state_dir=str(state_dir) if state_dir is not None else None,
        limit=limit,
    )


def _node_call(handle: NodeHandle, method: str, *, args: list[Any] | None = None, kwargs: dict[str, Any] | None = None) -> Any:
    call_kwargs: dict[str, Any] = {
        "workflow_id": handle.workflow_id,
        "node_id": handle.node_id,
        "key": handle.key,
        "method": method,
        "args": builtins.list(args or []),
        "kwargs": dict(kwargs or {}),
    }
    if handle._state_dir is not None:
        call_kwargs["state_dir"] = str(handle._state_dir)
    return transport.call_host("workflow.node_call", **call_kwargs)


def _group_call(handle: NodeGroupHandle, method: str, *, args: list[Any] | None = None, kwargs: dict[str, Any] | None = None) -> Any:
    call_kwargs: dict[str, Any] = {
        "workflow_id": handle.workflow_id,
        "node_ids": builtins.list(handle.node_ids),
        "key": handle.key,
        "method": method,
        "args": builtins.list(args or []),
        "kwargs": dict(kwargs or {}),
    }
    if handle._state_dir is not None:
        call_kwargs["state_dir"] = str(handle._state_dir)
    return transport.call_host("workflow.group_call", **call_kwargs)


def _dependency_ref(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, NodeHandle):
        return value.node_id
    if isinstance(value, NodeGroupHandle):
        return builtins.list(value.node_ids)
    if isinstance(value, CheckpointHandle):
        return value.node_id
    if isinstance(value, (list, tuple, set)):
        return [_dependency_ref(item) for item in value]
    return value


def _workflow_handle(payload: dict[str, Any], state_dir: str | Path | None) -> WorkflowHandle:
    return WorkflowHandle(str(payload["workflow_id"]), state_dir=state_dir)


def _node_handle(payload: dict[str, Any], state_dir: str | Path | None) -> NodeHandle:
    return NodeHandle(str(payload["workflow_id"]), str(payload["node_id"]), payload.get("key"), state_dir)


def _node_group_handle(payload: dict[str, Any], state_dir: str | Path | None) -> NodeGroupHandle:
    return NodeGroupHandle(str(payload["workflow_id"]), [str(item) for item in payload.get("node_ids") or []], payload.get("key"), state_dir)


def _checkpoint_handle(payload: dict[str, Any], state_dir: str | Path | None) -> CheckpointHandle:
    return CheckpointHandle(
        str(payload["workflow_id"]),
        str(payload["checkpoint_id"]),
        str(payload["node_id"]),
        str(payload["key"]),
        state_dir,
    )


def _wait_result(payload: dict[str, Any]) -> WorkflowWaitResult:
    return WorkflowWaitResult(
        workflow_id=str(payload["workflow_id"]),
        status=payload["status"],
        snapshot=dict(payload.get("snapshot") or {}),
        checkpoint=payload.get("checkpoint"),
        final=payload.get("final"),
        error=payload.get("error"),
    )


def _node_result(payload: dict[str, Any]) -> NodeResult:
    return NodeResult(
        workflow_id=str(payload["workflow_id"]),
        node_id=str(payload["node_id"]),
        status=str(payload["status"]),
        output=str(payload.get("output") or ""),
        error=payload.get("error"),
        result=dict(payload.get("result") or {}),
    )


def _assert_main_thread() -> None:
    if os.environ.get("UV_AGENT_RUNTIME_THREAD_KIND") == "workflow_node":
        raise RuntimeError("workflow is available only to the main Agent thread; workflow node agents must complete their assigned task directly")


def _checkpoint_summary(workflow_id: str, checkpoint: Mapping[str, Any], snapshot: Mapping[str, Any]) -> str:
    progress = snapshot.get("progress") if isinstance(snapshot.get("progress"), Mapping) else {}
    progress_text = ", ".join(f"{key}={value}" for key, value in sorted(progress.items())) or "no nodes"
    lines = [
        f"Workflow {workflow_id} reached checkpoint: {checkpoint.get('key') or checkpoint.get('checkpoint_id')}",
        f"Reason: {checkpoint.get('reason') or ''}",
    ]
    if checkpoint.get("recommended_action"):
        lines.append(f"Recommended: {checkpoint['recommended_action']}")
    lines.append(f"Progress: {progress_text}")
    options = checkpoint.get("options") or []
    if options:
        lines.append("")
        lines.append("Options:")
        for option in options:
            lines.append(f"- {option}")
    lines.append("")
    lines.append("Resume examples:")
    lines.append(f"- wf = workflow.resume(\"{workflow_id}\")")
    lines.append(f"- wf.continue_checkpoint(\"{checkpoint.get('key') or checkpoint.get('checkpoint_id')}\")")
    lines.append("- wf.inspect(\"node_key_or_id\")")
    lines.append("- wf.describe_graph()")
    return "\n".join(lines)


def _status_summary(verb: str, workflow_id: str, snapshot: Mapping[str, Any], error: Mapping[str, Any] | None = None) -> str:
    progress = snapshot.get("progress") if isinstance(snapshot.get("progress"), Mapping) else {}
    progress_text = ", ".join(f"{key}={value}" for key, value in sorted(progress.items())) or "no nodes"
    lines = [f"Workflow {workflow_id} {verb}.", f"Progress: {progress_text}"]
    if error:
        lines.append(f"Error: {json.dumps(dict(error), ensure_ascii=False, separators=(',', ':'))}")
    lines.append(f"Resume: wf = workflow.resume(\"{workflow_id}\")")
    return "\n".join(lines)
