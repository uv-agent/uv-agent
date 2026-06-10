from __future__ import annotations

import builtins
import concurrent.futures
import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, Literal

from .events import emit_event
from .textops import CommandTextResult, run_process_text

DB_FILENAME = "uv-agent.sqlite3"
SQLITE_BUSY_TIMEOUT_MS = 30_000
WORKFLOW_ACTIVE_STATUSES = {"running", "checkpoint", "failed"}
WORKFLOW_TERMINAL_STATUSES = {"completed", "cancelled"}
NODE_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass(frozen=True)
class WorkflowWaitResult:
    workflow_id: str
    status: Literal["completed", "checkpoint", "failed", "timeout", "cancelled", "interrupted"]
    snapshot: dict[str, Any]
    checkpoint: dict[str, Any] | None = None
    final: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def summary(self) -> str:
        """Return compact model-facing text for the current wait boundary."""

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
    _state_dir: Path | None = None

    def wait(self, *, timeout_s: float | None = None) -> NodeResult:
        wf = resume(self.workflow_id, state_dir=self._state_dir)
        result = wf.wait(timeout_s=timeout_s, until="completed")
        if result.status not in {"completed", "checkpoint"}:
            row = wf._node_by_id(self.node_id)
            return _node_result_from_row(row)
        row = wf._node_by_id(self.node_id)
        return _node_result_from_row(row)

    def result(self) -> NodeResult | None:
        wf = resume(self.workflow_id, state_dir=self._state_dir)
        row = wf._node_by_id(self.node_id)
        if row["status"] not in NODE_TERMINAL_STATUSES:
            return None
        return _node_result_from_row(row)

    def inspect(self) -> str | dict[str, Any]:
        return resume(self.workflow_id, state_dir=self._state_dir).inspect(self.node_id)


@dataclass(frozen=True)
class NodeGroupHandle:
    workflow_id: str
    node_ids: list[str]
    key: str | None = None
    _state_dir: Path | None = None

    def wait(self, *, timeout_s: float | None = None) -> list[NodeResult]:
        wf = resume(self.workflow_id, state_dir=self._state_dir)
        wf.wait(timeout_s=timeout_s, until="completed")
        return [wf._node_result(node_id) for node_id in self.node_ids]

    def completed(self) -> list[NodeResult]:
        wf = resume(self.workflow_id, state_dir=self._state_dir)
        return [wf._node_result(node_id) for node_id in self.node_ids if wf._node_by_id(node_id)["status"] == "completed"]

    def failed(self) -> list[NodeResult]:
        wf = resume(self.workflow_id, state_dir=self._state_dir)
        return [wf._node_result(node_id) for node_id in self.node_ids if wf._node_by_id(node_id)["status"] == "failed"]


@dataclass(frozen=True)
class CheckpointHandle:
    workflow_id: str
    checkpoint_id: str
    node_id: str
    key: str
    _state_dir: Path | None = None


class WorkflowHandle:
    def __init__(self, workflow_id: str, *, state_dir: str | Path | None = None) -> None:
        self.workflow_id = workflow_id
        self._state_dir = _state_dir(state_dir)

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
        return self._create_node(
            kind="agent",
            prompt=prompt,
            key=key,
            after=after,
            model_level=model_level,
            timeout_s=timeout_s,
            metadata=dict(metadata or {}),
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
        if concurrency is not None:
            self._update_workflow_state({"concurrency": max(1, int(concurrency))})
        nodes: list[NodeHandle] = []
        for index, item in enumerate(items, start=1):
            if callable(prompt):
                node_prompt = str(prompt(item))
            elif isinstance(prompt, str):
                node_prompt = prompt.format(item=item, index=index)
            else:
                node_prompt = str(item)
            node_key = f"{key}.{index}" if key else None
            nodes.append(
                self.agent(
                    node_prompt,
                    key=node_key,
                    after=after,
                    model_level=model_level,
                    metadata={"group_key": key, "item": item, "index": index},
                )
            )
        return NodeGroupHandle(
            workflow_id=self.workflow_id,
            node_ids=[node.node_id for node in nodes],
            key=key,
            _state_dir=self._state_dir,
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
        dependencies = after if after is not None else checkpoint
        return self._create_node(
            kind="review",
            prompt=prompt,
            key=key,
            after=dependencies,
            model_level=model_level,
            timeout_s=None,
            metadata={"checkpoint": checkpoint},
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
        node = self._create_node(
            kind="checkpoint",
            prompt=reason,
            key=key,
            after=after,
            model_level=None,
            timeout_s=None,
            metadata={},
        )
        checkpoint_id = _new_id("wfc")
        now = _utc_now_iso()
        with _connect(self._state_dir) as db:
            db.execute(
                """
                INSERT INTO workflow_checkpoints(
                    checkpoint_id, workflow_id, node_id, key, status, reason,
                    options_json, recommended_action, created_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    self.workflow_id,
                    node.node_id,
                    key,
                    reason,
                    _json_dumps(builtins.list(options or [])),
                    recommended_action,
                    now,
                ),
            )
        self._write_event("workflow.checkpoint.created", node_id=node.node_id, checkpoint_id=checkpoint_id, key=key)
        return CheckpointHandle(self.workflow_id, checkpoint_id, node.node_id, key, self._state_dir)

    def continue_checkpoint(self, checkpoint: str, *, resolution: Mapping[str, Any] | None = None) -> None:
        row = self._checkpoint_by_id_or_key(checkpoint)
        now = _utc_now_iso()
        with _connect(self._state_dir) as db:
            db.execute(
                """
                UPDATE workflow_checkpoints
                SET status = 'resolved', resolution_json = ?, resolved_at = ?
                WHERE checkpoint_id = ?
                """,
                (_json_dumps(dict(resolution or {})), now, row["checkpoint_id"]),
            )
            db.execute(
                """
                UPDATE workflows
                SET status = 'running', current_checkpoint_id = NULL, updated_at = ?
                WHERE workflow_id = ?
                """,
                (now, self.workflow_id),
            )
        self._write_event("workflow.checkpoint.resolved", node_id=row["node_id"], checkpoint_id=row["checkpoint_id"], resolution=dict(resolution or {}))

    def branch(self, *, key: str, from_checkpoint: str | None = None, tasks: Iterable[str] | None = None, metadata: Mapping[str, Any] | None = None) -> NodeGroupHandle:
        del metadata
        after = from_checkpoint
        return self.agent_many(builtins.list(tasks or []), key=key, after=after)

    def cancel(self, reason: str | None = None) -> None:
        now = _utc_now_iso()
        state = self._workflow_state()
        state["cancel_reason"] = reason or ""
        with _connect(self._state_dir) as db:
            db.execute(
                "UPDATE workflows SET status = 'cancelled', state_json = ?, updated_at = ? WHERE workflow_id = ?",
                (_json_dumps(state), now, self.workflow_id),
            )
            db.execute(
                "UPDATE workflow_nodes SET status = 'cancelled', completed_at = ? WHERE workflow_id = ? AND status IN ('pending', 'running')",
                (now, self.workflow_id),
            )
        self._write_event("workflow.cancelled", reason=reason or "")

    def complete(self, result: Any = None) -> None:
        now = _utc_now_iso()
        state = self._workflow_state()
        state["manual_result"] = result
        with _connect(self._state_dir) as db:
            db.execute(
                "UPDATE workflows SET status = 'completed', state_json = ?, updated_at = ? WHERE workflow_id = ?",
                (_json_dumps(state), now, self.workflow_id),
            )
        self._write_event("workflow.completed", result=result)

    def wait(self, *, timeout_s: float | None = None, until: str = "next_yield") -> WorkflowWaitResult:
        """Run ready nodes until completion, failure, timeout, or checkpoint."""

        start = monotonic()
        while True:
            workflow = self._workflow_row()
            status = str(workflow["status"])
            snapshot = self.snapshot()
            if status == "completed":
                return WorkflowWaitResult(self.workflow_id, "completed", snapshot, final=self._final_payload())
            if status == "cancelled":
                return WorkflowWaitResult(self.workflow_id, "cancelled", snapshot)
            unresolved = self._unresolved_checkpoint()
            if unresolved is not None:
                return WorkflowWaitResult(self.workflow_id, "checkpoint", snapshot, checkpoint=_checkpoint_payload(unresolved))
            if _expired(start, timeout_s):
                return WorkflowWaitResult(self.workflow_id, "timeout", snapshot)

            ready_checkpoints = self._ready_nodes(kind="checkpoint")
            if ready_checkpoints:
                checkpoint = self._reach_checkpoint(ready_checkpoints[0])
                snapshot = self.snapshot()
                return WorkflowWaitResult(self.workflow_id, "checkpoint", snapshot, checkpoint=_checkpoint_payload(checkpoint))

            ready_agents = self._ready_nodes(kind=None, executable_only=True)
            if ready_agents:
                self._run_ready_nodes(ready_agents, start=start, timeout_s=timeout_s)
                if _expired(start, timeout_s):
                    return WorkflowWaitResult(self.workflow_id, "timeout", self.snapshot())
                continue

            nodes = self._nodes()
            pending = [node for node in nodes if node["status"] == "pending"]
            failed = [node for node in nodes if node["status"] == "failed"]
            running = [node for node in nodes if node["status"] == "running"]
            if running:
                # A previous wait was interrupted after marking nodes running. The
                # subprocesses are owned by that old run_python process, so the
                # safest resumable state is to return control to the main Agent for a decision.
                error = {"running_nodes": [row["node_id"] for row in running]}
                self._set_workflow_status("failed", error=error)
                return WorkflowWaitResult(self.workflow_id, "failed", self.snapshot(), error=error)
            if failed and (until == "next_yield" or not pending):
                error = {"failed_nodes": [row["node_id"] for row in failed]}
                self._set_workflow_status("failed", error=error)
                return WorkflowWaitResult(self.workflow_id, "failed", self.snapshot(), error=error)
            if not pending:
                self._set_workflow_status("completed")
                return WorkflowWaitResult(self.workflow_id, "completed", self.snapshot(), final=self._final_payload())
            # Pending nodes exist but none are runnable, usually because a dependency
            # failed or was cancelled. Return a failed boundary instead of spinning.
            error = {"blocked_nodes": [row["node_id"] for row in pending]}
            self._set_workflow_status("failed", error=error)
            return WorkflowWaitResult(self.workflow_id, "failed", self.snapshot(), error=error)

    def snapshot(self) -> dict[str, Any]:
        workflow = dict(self._workflow_row())
        nodes = [dict(row) for row in self._nodes()]
        checkpoints = [dict(row) for row in self._checkpoints()]
        counts: dict[str, int] = {}
        for node in nodes:
            counts[str(node["status"])] = counts.get(str(node["status"]), 0) + 1
        return {
            "workflow_id": self.workflow_id,
            "objective": workflow.get("objective"),
            "status": workflow.get("status"),
            "default_model_level": workflow.get("default_model_level"),
            "current_checkpoint_id": workflow.get("current_checkpoint_id"),
            "created_at": workflow.get("created_at"),
            "updated_at": workflow.get("updated_at"),
            "progress": counts,
            "nodes": [_node_summary(row) for row in nodes],
            "checkpoints": [_checkpoint_summary_dict(row) for row in checkpoints],
        }

    def graph(self, *, include_results: bool = False) -> dict[str, Any]:
        workflow = self._workflow_row()
        graph_nodes = []
        for node in self._nodes():
            item = _node_summary(node, include_prompt=True)
            item["dependencies"] = _json_loads(node["dependencies_json"], default=[])
            if include_results:
                item["result"] = _json_loads(node["result_json"], default={})
                item["error"] = _json_loads(node["error_json"], default={})
            graph_nodes.append(item)
        return {
            "workflow_id": self.workflow_id,
            "objective": workflow["objective"],
            "status": workflow["status"],
            "default_model_level": workflow["default_model_level"],
            "state": _json_loads(workflow["state_json"], default={}),
            "metadata": _json_loads(workflow["metadata_json"], default={}),
            "nodes": graph_nodes,
            "checkpoints": [_checkpoint_summary_dict(row) for row in self._checkpoints()],
            "created_at": workflow["created_at"],
            "updated_at": workflow["updated_at"],
        }

    def describe_graph(self) -> str:
        graph = self.graph(include_results=False)
        lines = [
            f"Workflow {self.workflow_id}: {graph['objective']}",
            f"Status: {graph['status']}",
        ]
        if graph.get("default_model_level"):
            lines.append(f"Default model level: {graph['default_model_level']}")
        lines.append("Nodes:")
        for node in graph["nodes"]:
            deps = ", ".join(node.get("dependencies") or []) or "none"
            key = f" key={node['key']}" if node.get("key") else ""
            model = f" model={node['model_level']}" if node.get("model_level") else ""
            lines.append(f"- {node['node_id']}{key} [{node['kind']}/{node['status']}]{model} deps={deps}")
            prompt = str(node.get("prompt") or "").strip()
            if prompt:
                lines.append(f"  prompt: {prompt}")
        if graph["checkpoints"]:
            lines.append("Checkpoints:")
            for checkpoint in graph["checkpoints"]:
                lines.append(
                    f"- {checkpoint['checkpoint_id']} key={checkpoint['key']} status={checkpoint['status']} reason={checkpoint['reason']}"
                )
        return "\n".join(lines)

    def inspect(self, node: str) -> str | dict[str, Any]:
        row = self._node_by_id_or_key(node)
        if row["kind"] in {"agent", "review"}:
            result = _json_loads(row["result_json"], default={})
            output = str(result.get("stdout") or row["result_summary"] or "")
            if output or row["status"] == "completed":
                return output
        if row["kind"] == "checkpoint":
            checkpoint = self._checkpoint_for_node(row["node_id"])
            return _checkpoint_payload(checkpoint) if checkpoint else _node_summary(row)
        return _node_summary(row)

    def nodes(self, *, status: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        return [
            _node_summary(row, include_prompt=True)
            for row in self._nodes()
            if (status is None or row["status"] == status) and (kind is None or row["kind"] == kind)
        ]

    def update_node(self, node: str, **patch: Any) -> NodeHandle:
        row = self._node_by_id_or_key(node)
        if row["status"] != "pending":
            raise RuntimeError("Only pending workflow nodes can be updated in place; use replace_node for completed nodes")
        allowed = {"key", "prompt", "model_level", "dependencies", "metadata", "timeout_s"}
        unknown = set(patch) - allowed
        if unknown:
            raise ValueError(f"Unsupported node patch keys: {sorted(unknown)}")
        updates: dict[str, Any] = {}
        if "key" in patch:
            updates["key"] = patch["key"]
        if "prompt" in patch:
            updates["prompt"] = patch["prompt"]
        if "model_level" in patch:
            updates["model_level"] = patch["model_level"]
        metadata = _json_loads(row["metadata_json"], default={})
        if "timeout_s" in patch:
            metadata["timeout_s"] = patch["timeout_s"]
        if "metadata" in patch:
            metadata.update(dict(patch["metadata"] or {}))
        updates["metadata_json"] = _json_dumps(metadata)
        if "dependencies" in patch:
            updates["dependencies_json"] = _json_dumps(self._resolve_dependencies(patch["dependencies"]))
        if updates:
            assignments = ", ".join(f"{column} = ?" for column in updates)
            with _connect(self._state_dir) as db:
                db.execute(
                    f"UPDATE workflow_nodes SET {assignments} WHERE node_id = ?",
                    (*updates.values(), row["node_id"]),
                )
        self._touch()
        self._write_event("workflow.node.updated", node_id=row["node_id"], patch={key: value for key, value in patch.items() if key != "metadata"})
        return NodeHandle(self.workflow_id, row["node_id"], updates.get("key", row["key"]), self._state_dir)

    def remove_node(self, node: str, *, cascade: bool = False) -> None:
        row = self._node_by_id_or_key(node)
        to_cancel = {row["node_id"]}
        if cascade:
            changed = True
            while changed:
                changed = False
                for candidate in self._nodes():
                    deps = set(_json_loads(candidate["dependencies_json"], default=[]))
                    if deps & to_cancel and candidate["node_id"] not in to_cancel:
                        to_cancel.add(candidate["node_id"])
                        changed = True
        with _connect(self._state_dir) as db:
            db.execute(
                f"UPDATE workflow_nodes SET status = 'cancelled', completed_at = ? WHERE workflow_id = ? AND node_id IN ({_placeholders(to_cancel)}) AND status = 'pending'",
                (_utc_now_iso(), self.workflow_id, *sorted(to_cancel)),
            )
        self._touch()
        self._write_event("workflow.node.removed", node_id=row["node_id"], cascade=cascade, affected=sorted(to_cancel))

    def replace_node(self, node: str, *, kind: str | None = None, prompt: str | None = None, dependencies: Any = None, **patch: Any) -> NodeHandle:
        row = self._node_by_id_or_key(node)
        if row["status"] == "pending":
            return self.update_node(
                row["node_id"],
                prompt=prompt if prompt is not None else row["prompt"],
                dependencies=dependencies if dependencies is not None else _json_loads(row["dependencies_json"], default=[]),
                **patch,
            )
        replacement_key = str(patch.pop("key", "") or f"{row['key'] or row['node_id']}.replacement")
        new_node = self._create_node(
            kind=kind or row["kind"],
            prompt=prompt if prompt is not None else str(row["prompt"] or ""),
            key=replacement_key,
            after=dependencies if dependencies is not None else _json_loads(row["dependencies_json"], default=[]),
            model_level=patch.pop("model_level", row["model_level"]),
            timeout_s=patch.pop("timeout_s", _json_loads(row["metadata_json"], default={}).get("timeout_s")),
            metadata={"replaces": row["node_id"], **dict(patch.pop("metadata", {}) or {})},
        )
        self._write_event("workflow.node.replaced", node_id=row["node_id"], replacement_node_id=new_node.node_id)
        return new_node

    def add_dependency(self, node: str, depends_on: str) -> None:
        row = self._node_by_id_or_key(node)
        deps = self._resolve_dependencies(_json_loads(row["dependencies_json"], default=[]))
        dep_ids = self._resolve_dependencies(depends_on)
        for dep in dep_ids:
            if dep not in deps:
                deps.append(dep)
        self.update_node(row["node_id"], dependencies=deps)
        self._write_event("workflow.dependency.added", node_id=row["node_id"], depends_on=dep_ids)

    def remove_dependency(self, node: str, depends_on: str) -> None:
        row = self._node_by_id_or_key(node)
        remove = set(self._resolve_dependencies(depends_on))
        deps = [dep for dep in _json_loads(row["dependencies_json"], default=[]) if dep not in remove]
        self.update_node(row["node_id"], dependencies=deps)
        self._write_event("workflow.dependency.removed", node_id=row["node_id"], depends_on=sorted(remove))

    def update_checkpoint(self, checkpoint: str, **patch: Any) -> CheckpointHandle:
        row = self._checkpoint_by_id_or_key(checkpoint)
        allowed = {"key", "reason", "options", "recommended_action"}
        unknown = set(patch) - allowed
        if unknown:
            raise ValueError(f"Unsupported checkpoint patch keys: {sorted(unknown)}")
        updates: dict[str, Any] = {}
        if "key" in patch:
            updates["key"] = patch["key"]
            self.update_node(row["node_id"], key=patch["key"])
        if "reason" in patch:
            updates["reason"] = patch["reason"]
            self.update_node(row["node_id"], prompt=patch["reason"])
        if "options" in patch:
            updates["options_json"] = _json_dumps(builtins.list(patch["options"] or []))
        if "recommended_action" in patch:
            updates["recommended_action"] = patch["recommended_action"]
        if updates:
            assignments = ", ".join(f"{column} = ?" for column in updates)
            with _connect(self._state_dir) as db:
                db.execute(f"UPDATE workflow_checkpoints SET {assignments} WHERE checkpoint_id = ?", (*updates.values(), row["checkpoint_id"]))
        self._touch()
        self._write_event("workflow.checkpoint.updated", node_id=row["node_id"], checkpoint_id=row["checkpoint_id"], patch=patch)
        return CheckpointHandle(self.workflow_id, row["checkpoint_id"], row["node_id"], updates.get("key", row["key"]), self._state_dir)

    def apply_graph_patch(self, patch: Mapping[str, Any]) -> dict[str, Any]:
        for op in patch.get("operations", []) if isinstance(patch.get("operations"), list) else []:
            if not isinstance(op, Mapping):
                continue
            action = op.get("op") or op.get("action")
            if action == "update_node":
                self.update_node(str(op["node"]), **dict(op.get("patch") or {}))
            elif action == "remove_node":
                self.remove_node(str(op["node"]), cascade=bool(op.get("cascade")))
            elif action == "add_dependency":
                self.add_dependency(str(op["node"]), str(op["depends_on"]))
            elif action == "remove_dependency":
                self.remove_dependency(str(op["node"]), str(op["depends_on"]))
            elif action == "update_checkpoint":
                self.update_checkpoint(str(op["checkpoint"]), **dict(op.get("patch") or {}))
            else:
                raise ValueError(f"Unsupported graph patch operation: {action}")
        self._write_event("workflow.graph.patched", patch=dict(patch))
        return self.graph()

    def _create_node(
        self,
        *,
        kind: str,
        prompt: str,
        key: str | None,
        after: Any,
        model_level: str | None,
        timeout_s: float | None,
        metadata: dict[str, Any],
    ) -> NodeHandle:
        node_id = _new_id("wfn")
        now = _utc_now_iso()
        if timeout_s is not None:
            metadata = {**metadata, "timeout_s": timeout_s}
        deps = self._resolve_dependencies(after)
        with _connect(self._state_dir) as db:
            db.execute(
                """
                INSERT INTO workflow_nodes(
                    node_id, workflow_id, key, kind, status, dependencies_json,
                    prompt, model_level, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (node_id, self.workflow_id, key, kind, _json_dumps(deps), prompt, model_level, _json_dumps(metadata), now),
            )
            db.execute("UPDATE workflows SET updated_at = ?, status = 'running' WHERE workflow_id = ?", (now, self.workflow_id))
        self._write_event("workflow.node.created", node_id=node_id, kind=kind, key=key, dependencies=deps)
        return NodeHandle(self.workflow_id, node_id, key, self._state_dir)

    def _resolve_dependencies(self, after: Any) -> list[str]:
        if after is None:
            return []
        if isinstance(after, NodeHandle):
            return [after.node_id]
        if isinstance(after, NodeGroupHandle):
            return builtins.list(after.node_ids)
        if isinstance(after, CheckpointHandle):
            return [after.node_id]
        if isinstance(after, str):
            return [self._node_by_id_or_key(after)["node_id"]]
        if isinstance(after, Iterable) and not isinstance(after, (bytes, bytearray, Mapping)):
            deps: list[str] = []
            for item in after:
                for dep in self._resolve_dependencies(item):
                    if dep not in deps:
                        deps.append(dep)
            return deps
        raise TypeError(f"Unsupported dependency reference: {after!r}")

    def _workflow_row(self) -> sqlite3.Row:
        with _connect(self._state_dir) as db:
            row = db.execute("SELECT * FROM workflows WHERE workflow_id = ?", (self.workflow_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"Missing workflow {self.workflow_id}")
        return row

    def _workflow_state(self) -> dict[str, Any]:
        return _json_loads(self._workflow_row()["state_json"], default={})

    def _update_workflow_state(self, patch: Mapping[str, Any]) -> None:
        state = self._workflow_state()
        state.update(dict(patch))
        now = _utc_now_iso()
        with _connect(self._state_dir) as db:
            db.execute("UPDATE workflows SET state_json = ?, updated_at = ? WHERE workflow_id = ?", (_json_dumps(state), now, self.workflow_id))

    def _set_workflow_status(self, status: str, *, error: Mapping[str, Any] | None = None) -> None:
        state = self._workflow_state()
        if error:
            state["error"] = dict(error)
        now = _utc_now_iso()
        with _connect(self._state_dir) as db:
            db.execute("UPDATE workflows SET status = ?, state_json = ?, updated_at = ? WHERE workflow_id = ?", (status, _json_dumps(state), now, self.workflow_id))

    def _touch(self) -> None:
        with _connect(self._state_dir) as db:
            db.execute("UPDATE workflows SET updated_at = ? WHERE workflow_id = ?", (_utc_now_iso(), self.workflow_id))

    def _nodes(self) -> list[sqlite3.Row]:
        with _connect(self._state_dir) as db:
            return db.execute(
                "SELECT * FROM workflow_nodes WHERE workflow_id = ? ORDER BY created_at ASC, rowid ASC",
                (self.workflow_id,),
            ).fetchall()

    def _checkpoints(self) -> list[sqlite3.Row]:
        with _connect(self._state_dir) as db:
            return db.execute(
                "SELECT * FROM workflow_checkpoints WHERE workflow_id = ? ORDER BY created_at ASC, rowid ASC",
                (self.workflow_id,),
            ).fetchall()

    def _node_by_id(self, node_id: str) -> sqlite3.Row:
        with _connect(self._state_dir) as db:
            row = db.execute("SELECT * FROM workflow_nodes WHERE workflow_id = ? AND node_id = ?", (self.workflow_id, node_id)).fetchone()
        if row is None:
            raise FileNotFoundError(f"Missing workflow node {node_id}")
        return row

    def _node_by_id_or_key(self, value: str) -> sqlite3.Row:
        with _connect(self._state_dir) as db:
            row = db.execute(
                """
                SELECT * FROM workflow_nodes
                WHERE workflow_id = ? AND (node_id = ? OR key = ?)
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (self.workflow_id, value, value),
            ).fetchone()
            if row is None:
                checkpoint = db.execute(
                    "SELECT node_id FROM workflow_checkpoints WHERE workflow_id = ? AND (checkpoint_id = ? OR key = ?) LIMIT 1",
                    (self.workflow_id, value, value),
                ).fetchone()
                if checkpoint is not None:
                    row = db.execute("SELECT * FROM workflow_nodes WHERE node_id = ?", (checkpoint["node_id"],)).fetchone()
        if row is None:
            raise FileNotFoundError(f"Missing workflow node/key {value}")
        return row

    def _checkpoint_by_id_or_key(self, value: str) -> sqlite3.Row:
        with _connect(self._state_dir) as db:
            row = db.execute(
                """
                SELECT * FROM workflow_checkpoints
                WHERE workflow_id = ? AND (checkpoint_id = ? OR key = ?)
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (self.workflow_id, value, value),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"Missing workflow checkpoint {value}")
        return row

    def _checkpoint_for_node(self, node_id: str) -> sqlite3.Row | None:
        with _connect(self._state_dir) as db:
            return db.execute("SELECT * FROM workflow_checkpoints WHERE node_id = ?", (node_id,)).fetchone()

    def _unresolved_checkpoint(self) -> sqlite3.Row | None:
        with _connect(self._state_dir) as db:
            return db.execute(
                """
                SELECT * FROM workflow_checkpoints
                WHERE workflow_id = ? AND status = 'unresolved'
                ORDER BY created_at ASC, rowid ASC
                LIMIT 1
                """,
                (self.workflow_id,),
            ).fetchone()

    def _ready_nodes(self, *, kind: str | None = None, executable_only: bool = False) -> list[sqlite3.Row]:
        nodes = self._nodes()
        completed = {row["node_id"] for row in nodes if row["status"] == "completed"}
        ready: list[sqlite3.Row] = []
        for row in nodes:
            if row["status"] != "pending":
                continue
            if kind is not None and row["kind"] != kind:
                continue
            if executable_only and row["kind"] not in {"agent", "review"}:
                continue
            deps = _json_loads(row["dependencies_json"], default=[])
            if all(dep in completed for dep in deps):
                ready.append(row)
        return ready

    def _run_ready_nodes(self, nodes: list[sqlite3.Row], *, start: float, timeout_s: float | None) -> None:
        state = self._workflow_state()
        concurrency = max(1, int(state.get("concurrency") or min(4, len(nodes)) or 1))
        batch = nodes[:concurrency]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [executor.submit(self._execute_node, dict(row), start, timeout_s) for row in batch]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def _execute_node(self, node: dict[str, Any], start: float, workflow_timeout_s: float | None) -> None:
        node_id = str(node["node_id"])
        now = _utc_now_iso()
        with _connect(self._state_dir) as db:
            db.execute(
                "UPDATE workflow_nodes SET status = 'running', started_at = ? WHERE node_id = ? AND status = 'pending'",
                (now, node_id),
            )
        emit_event("workflow.node.started", workflow_id=self.workflow_id, node_id=node_id, key=node.get("key"), node_kind=node.get("kind"))
        metadata = _json_loads(node.get("metadata_json"), default={})
        timeout_s = metadata.get("timeout_s")
        remaining = _remaining(start, workflow_timeout_s)
        if timeout_s is not None and remaining is not None:
            timeout_s = min(float(timeout_s), remaining)
        elif remaining is not None:
            timeout_s = remaining
        prompt = str(node.get("prompt") or "")
        level = str(node.get("model_level") or self._workflow_row()["default_model_level"] or "") or None
        result = _run_workflow_node(
            prompt,
            workflow_id=self.workflow_id,
            node_id=node_id,
            level=level,
            timeout_s=timeout_s,
            state_dir=self._state_dir,
        )
        completed_at = _utc_now_iso()
        thread_id = _extract_workflow_node_thread_id(result.stderr)
        result_json = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "thread_id": thread_id,
            "model_level": level,
        }
        status = "completed" if result.returncode == 0 and not result.timed_out else "failed"
        error = {} if status == "completed" else {"returncode": result.returncode, "stderr": result.stderr, "timed_out": result.timed_out}
        with _connect(self._state_dir) as db:
            db.execute(
                """
                UPDATE workflow_nodes
                SET status = ?, completed_at = ?, thread_id = ?, result_summary = ?, result_json = ?, error_json = ?
                WHERE node_id = ?
                """,
                (status, completed_at, thread_id, result.stdout, _json_dumps(result_json), _json_dumps(error), node_id),
            )
            db.execute("UPDATE workflows SET updated_at = ? WHERE workflow_id = ?", (completed_at, self.workflow_id))
        emit_event(
            "workflow.node.completed" if status == "completed" else "workflow.node.failed",
            workflow_id=self.workflow_id,
            node_id=node_id,
            key=node.get("key"),
            returncode=result.returncode,
            timed_out=result.timed_out,
            thread_id=thread_id,
        )
        self._write_event(
            "workflow.node.completed" if status == "completed" else "workflow.node.failed",
            node_id=node_id,
            returncode=result.returncode,
            timed_out=result.timed_out,
            thread_id=thread_id,
        )

    def _reach_checkpoint(self, node: sqlite3.Row) -> sqlite3.Row:
        checkpoint = self._checkpoint_for_node(node["node_id"])
        if checkpoint is None:
            raise RuntimeError(f"Checkpoint node {node['node_id']} has no checkpoint row")
        snapshot = self.snapshot()
        now = _utc_now_iso()
        with _connect(self._state_dir) as db:
            db.execute(
                "UPDATE workflow_nodes SET status = 'completed', completed_at = ? WHERE node_id = ?",
                (now, node["node_id"]),
            )
            db.execute(
                """
                UPDATE workflow_checkpoints
                SET status = 'unresolved', snapshot_json = ?
                WHERE checkpoint_id = ?
                """,
                (_json_dumps(snapshot), checkpoint["checkpoint_id"]),
            )
            db.execute(
                """
                UPDATE workflows
                SET status = 'checkpoint', current_checkpoint_id = ?, updated_at = ?
                WHERE workflow_id = ?
                """,
                (checkpoint["checkpoint_id"], now, self.workflow_id),
            )
        emit_event("workflow.checkpoint.reached", workflow_id=self.workflow_id, checkpoint_id=checkpoint["checkpoint_id"], key=checkpoint["key"])
        self._write_event("workflow.checkpoint.reached", node_id=node["node_id"], checkpoint_id=checkpoint["checkpoint_id"], key=checkpoint["key"])
        return self._checkpoint_by_id_or_key(checkpoint["checkpoint_id"])

    def _final_payload(self) -> dict[str, Any]:
        state = self._workflow_state()
        if "manual_result" in state:
            return {"output": state["manual_result"], "source": "manual"}
        agent_nodes = [row for row in self._nodes() if row["kind"] in {"agent", "review"} and row["status"] == "completed"]
        if not agent_nodes:
            return {"output": "", "source": "none"}
        dependents: set[str] = set()
        for row in self._nodes():
            dependents.update(_json_loads(row["dependencies_json"], default=[]))
        terminal = [row for row in agent_nodes if row["node_id"] not in dependents]
        final = (terminal or agent_nodes)[-1]
        result = _json_loads(final["result_json"], default={})
        return {"node_id": final["node_id"], "key": final["key"], "output": str(result.get("stdout") or final["result_summary"] or "")}

    def _node_result(self, node_id: str) -> NodeResult:
        return _node_result_from_row(self._node_by_id(node_id))

    def _write_event(self, event_type: str, *, node_id: str | None = None, **payload: Any) -> None:
        with _connect(self._state_dir) as db:
            db.execute(
                "INSERT INTO workflow_events(workflow_id, node_id, type, created_at, payload_json) VALUES (?, ?, ?, ?, ?)",
                (self.workflow_id, node_id, event_type, _utc_now_iso(), _json_dumps({"type": event_type, **payload})),
            )


def start(
    objective: str,
    *,
    key: str | None = None,
    default_model_level: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    state_dir: str | Path | None = None,
) -> WorkflowHandle:
    _assert_main_thread()
    workflow_id = key if key and key.startswith("wf_") else _new_id("wf")
    now = _utc_now_iso()
    base = _state_dir(state_dir)
    metadata_dict = dict(metadata or {})
    if key:
        metadata_dict.setdefault("key", key)
    with _connect(base) as db:
        db.execute(
            """
            INSERT INTO workflows(
                workflow_id, parent_thread_id, parent_turn_id, parent_run_id,
                objective, status, default_model_level, state_json, metadata_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'running', ?, '{}', ?, ?, ?)
            """,
            (
                workflow_id,
                os.environ.get("UV_AGENT_RUNTIME_THREAD_ID"),
                os.environ.get("UV_AGENT_RUNTIME_TURN_ID"),
                os.environ.get("UV_AGENT_RUNTIME_RUN_ID"),
                objective,
                default_model_level,
                _json_dumps(metadata_dict),
                now,
                now,
            ),
        )
        db.execute(
            "INSERT INTO workflow_events(workflow_id, type, created_at, payload_json) VALUES (?, 'workflow.created', ?, ?)",
            (workflow_id, now, _json_dumps({"type": "workflow.created", "objective": objective, "key": key})),
        )
    emit_event("workflow.started", workflow_id=workflow_id, objective=objective)
    return WorkflowHandle(workflow_id, state_dir=base)


def resume(workflow_id: str, *, state_dir: str | Path | None = None) -> WorkflowHandle:
    handle = WorkflowHandle(workflow_id, state_dir=state_dir)
    handle._workflow_row()
    return handle


def list(status: str | None = None, limit: int = 20, *, state_dir: str | Path | None = None) -> list[dict[str, Any]]:  # noqa: A001 - public API mirrors built-in name by design.
    base = _state_dir(state_dir)
    with _connect(base) as db:
        if status is None:
            rows = db.execute("SELECT * FROM workflows ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = db.execute("SELECT * FROM workflows WHERE status = ? ORDER BY updated_at DESC LIMIT ?", (status, limit)).fetchall()
    return [_workflow_summary(row) for row in rows]


def agent(prompt: str, *, model_level: str | None = None, timeout_s: float | None = None) -> NodeHandle:
    wf = start(objective=_objective_from_prompt(prompt), default_model_level=model_level)
    return wf.agent(prompt, model_level=model_level, timeout_s=timeout_s)


def active_snapshots(
    *,
    parent_thread_id: str | None = None,
    state_dir: str | Path | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    base = _state_dir(state_dir)
    with _connect(base) as db:
        clauses = ["status IN ('running', 'checkpoint', 'failed')"]
        params: list[Any] = []
        if parent_thread_id is not None:
            clauses.append("parent_thread_id = ?")
            params.append(parent_thread_id)
        rows = db.execute(
            f"SELECT workflow_id FROM workflows WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [resume(row["workflow_id"], state_dir=base).snapshot() for row in rows]


def _run_workflow_node(
    prompt: str,
    *,
    workflow_id: str,
    node_id: str,
    level: str | None,
    timeout_s: float | None,
    state_dir: Path,
) -> CommandTextResult:
    args = [os.environ.get("UV_BIN") or "uv", "run", "uv-agent"]
    if level:
        args.extend(["--level", level])
    args.extend(
        [
            "--thread-kind",
            "workflow_node",
            "--workflow-id",
            workflow_id,
            "--node-id",
            node_id,
        ]
    )
    parent_thread_id = os.environ.get("UV_AGENT_RUNTIME_THREAD_ID")
    parent_turn_id = os.environ.get("UV_AGENT_RUNTIME_TURN_ID")
    parent_run_id = os.environ.get("UV_AGENT_RUNTIME_RUN_ID")
    if parent_thread_id:
        args.extend(["--parent-thread", parent_thread_id])
    if parent_turn_id:
        args.extend(["--parent-turn", parent_turn_id])
    if parent_run_id:
        args.extend(["--parent-run", parent_run_id])
    args.extend(["--project-state-dir", str(state_dir), "--no-stream", "workflow-node", prompt])
    return run_process_text(args, cwd=os.environ.get("UV_AGENT_RUNTIME_PROJECT_ROOT") or None, timeout_s=timeout_s)


def _connect(base: Path) -> sqlite3.Connection:
    base.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(base / DB_FILENAME, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    _ensure_workflow_schema(connection)
    return connection


def _ensure_workflow_schema(connection: sqlite3.Connection) -> None:
    with connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS workflows (
              workflow_id TEXT PRIMARY KEY,
              parent_thread_id TEXT,
              parent_turn_id TEXT,
              parent_run_id TEXT,
              objective TEXT NOT NULL,
              status TEXT NOT NULL,
              default_model_level TEXT,
              current_checkpoint_id TEXT,
              state_json TEXT NOT NULL DEFAULT '{}',
              metadata_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workflow_nodes (
              node_id TEXT PRIMARY KEY,
              workflow_id TEXT NOT NULL,
              key TEXT,
              kind TEXT NOT NULL,
              status TEXT NOT NULL,
              dependencies_json TEXT NOT NULL DEFAULT '[]',
              prompt TEXT,
              model_level TEXT,
              thread_id TEXT,
              run_id TEXT,
              result_summary TEXT,
              result_json TEXT NOT NULL DEFAULT '{}',
              error_json TEXT NOT NULL DEFAULT '{}',
              metadata_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              started_at TEXT,
              completed_at TEXT,
              FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS workflow_checkpoints (
              checkpoint_id TEXT PRIMARY KEY,
              workflow_id TEXT NOT NULL,
              node_id TEXT NOT NULL,
              key TEXT NOT NULL,
              status TEXT NOT NULL,
              reason TEXT NOT NULL,
              options_json TEXT NOT NULL DEFAULT '[]',
              recommended_action TEXT,
              snapshot_json TEXT NOT NULL DEFAULT '{}',
              resolution_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              resolved_at TEXT,
              FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
              FOREIGN KEY(node_id) REFERENCES workflow_nodes(node_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS workflow_events (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              workflow_id TEXT NOT NULL,
              node_id TEXT,
              type TEXT NOT NULL,
              created_at TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_workflow_nodes_workflow_status ON workflow_nodes(workflow_id, status);
            CREATE INDEX IF NOT EXISTS idx_workflow_nodes_workflow_key ON workflow_nodes(workflow_id, key);
            CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_workflow_status ON workflow_checkpoints(workflow_id, status);
            CREATE INDEX IF NOT EXISTS idx_workflow_events_workflow_id ON workflow_events(workflow_id, event_id);
            CREATE INDEX IF NOT EXISTS idx_workflows_parent_thread ON workflows(parent_thread_id, status);
            """
        )


def _state_dir(state_dir: str | Path | None) -> Path:
    if state_dir is not None:
        return Path(state_dir).resolve()
    env = os.environ.get("UV_AGENT_RUNTIME_STATE_DIR") or os.environ.get("UV_AGENT_RUNTIME_PROJECT_STATE_DIR")
    if not env:
        raise RuntimeError("UV_AGENT_RUNTIME_STATE_DIR is not set; pass state_dir explicitly")
    return Path(env).resolve()


def _assert_main_thread() -> None:
    if os.environ.get("UV_AGENT_RUNTIME_THREAD_KIND") in {"workflow_node", "subagent"}:
        raise RuntimeError("workflow is available only to the main Agent thread; workflow node agents must complete their assigned task directly")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def _placeholders(values: Iterable[Any]) -> str:
    return ", ".join("?" for _ in values)


def _expired(start: float, timeout_s: float | None) -> bool:
    return timeout_s is not None and monotonic() - start >= timeout_s


def _remaining(start: float, timeout_s: float | None) -> float | None:
    if timeout_s is None:
        return None
    return max(0.01, timeout_s - (monotonic() - start))


def _extract_workflow_node_thread_id(stderr: str) -> str | None:
    match = re.search(r"^\[workflow-node-thread\]\s+(\S+)\s*$", stderr, flags=re.MULTILINE)
    return match.group(1) if match else None


def _objective_from_prompt(prompt: str) -> str:
    first = prompt.strip().splitlines()[0] if prompt.strip() else "Workflow task"
    return first[:120]


def _node_summary(row: sqlite3.Row | Mapping[str, Any], *, include_prompt: bool = False) -> dict[str, Any]:
    data = dict(row)
    summary = {
        "node_id": data.get("node_id"),
        "key": data.get("key"),
        "kind": data.get("kind"),
        "status": data.get("status"),
        "model_level": data.get("model_level"),
        "thread_id": data.get("thread_id"),
        "created_at": data.get("created_at"),
        "started_at": data.get("started_at"),
        "completed_at": data.get("completed_at"),
    }
    if include_prompt:
        summary["prompt"] = data.get("prompt")
    if data.get("error_json"):
        error = _json_loads(data.get("error_json"), default={})
        if error:
            summary["error"] = error
    return {key: value for key, value in summary.items() if value not in (None, "")}


def _checkpoint_summary_dict(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    data = dict(row)
    return {
        "checkpoint_id": data.get("checkpoint_id"),
        "node_id": data.get("node_id"),
        "key": data.get("key"),
        "status": data.get("status"),
        "reason": data.get("reason"),
        "options": _json_loads(data.get("options_json"), default=[]),
        "recommended_action": data.get("recommended_action"),
        "created_at": data.get("created_at"),
        "resolved_at": data.get("resolved_at"),
    }


def _checkpoint_payload(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    payload = _checkpoint_summary_dict(row)
    payload["snapshot"] = _json_loads(dict(row).get("snapshot_json"), default={})
    return payload


def _node_result_from_row(row: sqlite3.Row) -> NodeResult:
    result = _json_loads(row["result_json"], default={})
    error = _json_loads(row["error_json"], default={})
    return NodeResult(
        workflow_id=row["workflow_id"],
        node_id=row["node_id"],
        status=row["status"],
        output=str(result.get("stdout") or row["result_summary"] or ""),
        error=error or None,
        result=result,
    )


def _workflow_summary(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "workflow_id": row["workflow_id"],
        "objective": row["objective"],
        "status": row["status"],
        "default_model_level": row["default_model_level"],
        "current_checkpoint_id": row["current_checkpoint_id"],
        "parent_thread_id": row["parent_thread_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


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
        lines.append(f"Error: {_json_dumps(dict(error))}")
    lines.append(f"Resume: wf = workflow.resume(\"{workflow_id}\")")
    return "\n".join(lines)
