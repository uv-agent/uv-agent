from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .service import connect_workflow_db

ACTIVE_WORKFLOW_STATUSES = {"running", "checkpoint", "failed"}
ACTIVE_WORKFLOWS_SECTION_TITLE = "## 活跃工作流"
ACTIVE_WORKFLOW_NO_NODES = "无节点"
ACTIVE_WORKFLOW_STATUS_LINE_TEMPLATE = "- `{workflow_id}` 状态={status} 目标={objective}"
ACTIVE_WORKFLOW_PROGRESS_LINE_TEMPLATE = "  - 进度：{progress}"
ACTIVE_WORKFLOW_CHECKPOINT_LINE_TEMPLATE = "  - 当前 checkpoint：{checkpoint}（{reason}）"
ACTIVE_WORKFLOW_NO_REASON_RECORDED = "未记录原因"
ACTIVE_WORKFLOW_COMPLETED_INSPECTABLE_LINE_TEMPLATE = "  - 已完成且可 inspect 的节点：{refs}"
ACTIVE_WORKFLOW_RESUME_LINE_TEMPLATE = '  - 恢复：import uv_agent_runtime as rt; wf = rt.workflow.resume("{workflow_id}")'
WORKFLOW_HELPER_SIGNATURE = """WorkflowWaitResult = {workflow_id: str, status: Literal["completed", "checkpoint", "failed", "timeout", "cancelled", "interrupted"], snapshot: dict[str, Any], checkpoint: dict[str, Any] | None, final: dict[str, Any] | None, error: dict[str, Any] | None}
NodeResult = {workflow_id: str, node_id: str, status: str, output: str, error: dict[str, Any] | None, result: dict[str, Any]}
class NodeHandle:
    workflow_id: str
    node_id: str
    key: str | None
    def wait(self, *, timeout_s: float | None = None) -> NodeResult: ...
    def result(self) -> NodeResult | None: ...
    def inspect(self) -> str | dict[str, Any]: ...
class NodeGroupHandle:
    workflow_id: str
    node_ids: list[str]
    key: str | None
    def wait(self, *, timeout_s: float | None = None) -> list[NodeResult]: ...
    def completed(self) -> list[NodeResult]: ...
    def failed(self) -> list[NodeResult]: ...
class CheckpointHandle:
    workflow_id: str
    checkpoint_id: str
    node_id: str
    key: str
class WorkflowHandle:
    workflow_id: str
    def agent(self, prompt: str, *, key: str | None = None, after: Any = None, model_level: str | None = None, timeout_s: float | None = None, metadata: Mapping[str, Any] | None = None) -> NodeHandle: ...
    def agent_many(self, items: Iterable[Any], *, key: str | None = None, prompt: str | Callable[[Any], str] | None = None, concurrency: int | None = None, after: Any = None, model_level: str | None = None) -> NodeGroupHandle: ...
    def review(self, *, key: str | None = None, checkpoint: str | None = None, prompt: str, model_level: str | None = None, after: Any = None) -> NodeHandle: ...
    def checkpoint(self, *, key: str, reason: str, after: Any = None, options: Sequence[str] | None = None, recommended_action: str | None = None) -> CheckpointHandle: ...
    def continue_checkpoint(self, checkpoint: str, *, resolution: Mapping[str, Any] | None = None) -> None: ...
    def wait(self, *, timeout_s: float | None = None, until: str = "next_yield") -> WorkflowWaitResult: ...
    def snapshot(self) -> dict[str, Any]: ...
    def graph(self, *, include_results: bool = False) -> dict[str, Any]: ...
    def describe_graph(self) -> str: ...
    def inspect(self, node: str) -> str | dict[str, Any]: ...
    def nodes(self, *, status: str | None = None, kind: str | None = None) -> list[dict[str, Any]]: ...
    def update_node(self, node: str, **patch: Any) -> NodeHandle: ...
    def replace_node(self, node: str, *, kind: str | None = None, prompt: str | None = None, dependencies: Any = None, **patch: Any) -> NodeHandle: ...
    def remove_node(self, node: str, *, cascade: bool = False) -> None: ...
    def complete(self, result: Any = None) -> None: ...
    def cancel(self, reason: str | None = None) -> None: ...
rt.workflow.start(objective: str, *, key: str | None = None, default_model_level: str | None = None, metadata: Mapping[str, Any] | None = None, state_dir: str | Path | None = None) -> WorkflowHandle
rt.workflow.resume(workflow_id: str, *, state_dir: str | Path | None = None) -> WorkflowHandle
rt.workflow.list(status: str | None = None, limit: int = 20, *, state_dir: str | Path | None = None) -> list[dict[str, Any]]
rt.workflow.agent(prompt: str, *, model_level: str | None = None, timeout_s: float | None = None) -> NodeHandle
rt.workflow.active_snapshots(*, parent_thread_id: str | None = None, state_dir: str | Path | None = None, limit: int = 20) -> list[dict[str, Any]]"""

WORKFLOW_CONTEXT_EXAMPLES: tuple[dict[str, str], ...] = (
    {
        "name": "create_investigation_graph",
        "description": "创建调查任务图并等待第一个 checkpoint。",
        "code": (
            'import uv_agent_runtime as rt\n'
            'wf = rt.workflow.start(objective="调查并规划插件系统")\n'
            'wf.agent("## 目标和任务\\n调查现状并列出重构边界。", key="investigate")\n'
            'wf.checkpoint(key="after_investigation", reason="检查调查结果", after="investigate")\n'
            'wf.wait()\n'
        ),
    },
    {
        "name": "inspect_first_checkpoint_and_extend_graph",
        "description": "检查第一个 checkpoint，继续添加实现和验证节点。",
        "code": (
            'import uv_agent_runtime as rt\n'
            'wf = rt.workflow.resume("wf_example")\n'
            'investigation = wf.inspect("investigate")\n'
            'wf.agent(f"## 要求和说明\\n基于调查结果实现第一批修改。\\n{investigation}", key="implement.first")\n'
            'wf.agent("运行聚焦测试并总结失败。", key="verify.first", after="implement.first")\n'
            'wf.continue_checkpoint("after_investigation")\n'
            'wf.wait()\n'
        ),
    },
    {
        "name": "inspect_review_checkpoint_and_finalize",
        "description": "检查验证结果，收尾并做最终验证。",
        "code": (
            'import uv_agent_runtime as rt\n'
            'wf = rt.workflow.resume("wf_example")\n'
            'verification = wf.inspect("verify.first")\n'
            'wf.agent(f"修复剩余问题并准备最终验证。\\n{verification}", key="fix.remaining")\n'
            'wf.agent("运行最终测试并给出结论。", key="verify.final", after="fix.remaining")\n'
            'wf.wait()\n'
        ),
    },
)


def render_workflow_context() -> dict[str, object]:
    """Return stable main-Agent-only workflow guidance as structured XML body."""

    return {
        "purpose": "Workflow 仅供主 Agent 使用，用于把独立或长时间运行的工作组织成持久任务图。",
        "instructions": [
            "Workflow 操作默认立即返回；需要等待时显式调用 wait()、join() 或 result()。",
            "wait() 会运行到完成、失败、超时、中断或 checkpoint。",
            "checkpoint 会把控制权交还给主 Agent，以便检查结果并调整方向。",
            "使用 graph() 或 describe_graph() 查看任务图；使用 inspect(node) 查看节点最终输出。",
            "节点 prompt 应自包含目标、范围、约束、期望输出，以及是否允许编辑。",
        ],
        "model_level_policy": [
            "在节点上传入 model_level，在 rt.workflow.start() 上传入 default_model_level，或省略以使用配置默认值。",
        ],
        "state_policy": [
            "当前 workflow 状态不会在此块中更新；用 wait()、snapshot()、graph()、inspect() 或 list() 查看。",
            '活跃 workflow 快照会通过压缩摘要的 "## 活跃工作流" 章节恢复。',
        ],
        "helper": {
            "import": "import uv_agent_runtime as rt",
            "namespace": "rt.workflow",
            "signature": WORKFLOW_HELPER_SIGNATURE,
        },
        "examples": [dict(item) for item in WORKFLOW_CONTEXT_EXAMPLES],
    }


def active_workflow_snapshots(
    data_dir: Path,
    *,
    parent_thread_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return compact active workflow snapshots for compaction handoff."""

    with connect_workflow_db(data_dir) as db:
        rows = db.execute(
            """
            SELECT *
            FROM workflows
            WHERE parent_thread_id = ? AND status IN ('running', 'checkpoint', 'failed')
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (parent_thread_id, limit),
        ).fetchall()
        return [_workflow_snapshot(db, row) for row in rows]


def active_workflows_compaction_section(data_dir: Path, *, parent_thread_id: str) -> str:
    """Render the deterministic ``## Active workflows`` compaction section."""

    snapshots = active_workflow_snapshots(data_dir, parent_thread_id=parent_thread_id)
    if not snapshots:
        return ""
    lines = [ACTIVE_WORKFLOWS_SECTION_TITLE, ""]
    for snapshot in snapshots:
        progress = snapshot.get("progress") if isinstance(snapshot.get("progress"), dict) else {}
        progress_text = ", ".join(f"{key}={progress[key]}" for key in sorted(progress)) or ACTIVE_WORKFLOW_NO_NODES
        lines.append(
            ACTIVE_WORKFLOW_STATUS_LINE_TEMPLATE.format(
                workflow_id=snapshot["workflow_id"],
                status=snapshot["status"],
                objective=snapshot["objective"],
            )
        )
        lines.append(ACTIVE_WORKFLOW_PROGRESS_LINE_TEMPLATE.format(progress=progress_text))
        checkpoint = snapshot.get("current_checkpoint")
        if isinstance(checkpoint, dict) and checkpoint:
            lines.append(
                ACTIVE_WORKFLOW_CHECKPOINT_LINE_TEMPLATE.format(
                    checkpoint=checkpoint.get("key") or checkpoint.get("checkpoint_id"),
                    reason=checkpoint.get("reason") or ACTIVE_WORKFLOW_NO_REASON_RECORDED,
                )
            )
        inspectable = [node for node in snapshot.get("nodes", []) if isinstance(node, dict) and node.get("status") == "completed"]
        if inspectable:
            refs = ", ".join(str(node.get("key") or node.get("node_id")) for node in inspectable[:8])
            lines.append(ACTIVE_WORKFLOW_COMPLETED_INSPECTABLE_LINE_TEMPLATE.format(refs=refs))
        lines.append(ACTIVE_WORKFLOW_RESUME_LINE_TEMPLATE.format(workflow_id=snapshot["workflow_id"]))
    return "\n".join(lines).rstrip()


def _workflow_snapshot(db: sqlite3.Connection, workflow: sqlite3.Row) -> dict[str, Any]:
    workflow_id = str(workflow["workflow_id"])
    nodes = db.execute(
        """
        SELECT node_id, key, kind, status, model_level, thread_id, created_at, started_at, completed_at, error_json
        FROM workflow_nodes
        WHERE workflow_id = ?
        ORDER BY created_at ASC, rowid ASC
        """,
        (workflow_id,),
    ).fetchall()
    checkpoints = db.execute(
        """
        SELECT checkpoint_id, node_id, key, status, reason, options_json, recommended_action, created_at, resolved_at
        FROM workflow_checkpoints
        WHERE workflow_id = ?
        ORDER BY created_at ASC, rowid ASC
        """,
        (workflow_id,),
    ).fetchall()
    counts: dict[str, int] = {}
    node_summaries = []
    for node in nodes:
        status = str(node["status"])
        counts[status] = counts.get(status, 0) + 1
        item = {
            "node_id": node["node_id"],
            "key": node["key"],
            "kind": node["kind"],
            "status": status,
            "model_level": node["model_level"],
            "thread_id": node["thread_id"],
            "started_at": node["started_at"],
            "completed_at": node["completed_at"],
        }
        error = _json_loads(node["error_json"], default={})
        if error:
            item["error"] = error
        node_summaries.append({key: value for key, value in item.items() if value not in (None, "")})
    checkpoint_summaries = [_checkpoint_payload(row) for row in checkpoints]
    current_checkpoint = None
    for checkpoint in checkpoint_summaries:
        if checkpoint.get("status") == "unresolved":
            current_checkpoint = checkpoint
            break
    return {
        "workflow_id": workflow_id,
        "objective": workflow["objective"],
        "status": workflow["status"],
        "default_model_level": workflow["default_model_level"],
        "current_checkpoint_id": workflow["current_checkpoint_id"],
        "progress": counts,
        "current_checkpoint": current_checkpoint,
        "nodes": node_summaries,
        "checkpoints": checkpoint_summaries,
        "created_at": workflow["created_at"],
        "updated_at": workflow["updated_at"],
    }


def _checkpoint_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "checkpoint_id": row["checkpoint_id"],
        "node_id": row["node_id"],
        "key": row["key"],
        "status": row["status"],
        "reason": row["reason"],
        "options": _json_loads(row["options_json"], default=[]),
        "recommended_action": row["recommended_action"],
        "created_at": row["created_at"],
        "resolved_at": row["resolved_at"],
    }


def _json_loads(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default
