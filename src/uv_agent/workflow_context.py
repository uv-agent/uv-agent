from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from uv_agent.state_db import connect_state_db

ACTIVE_WORKFLOW_STATUSES = {"running", "checkpoint", "failed"}

WORKFLOW_CONTEXT_TEXT = """<workflow_context scope="main_agent" status="current">
  <purpose>
    Workflow is available to the main Agent only. Use it to build, wait on,
    inspect, and adjust persistent task graphs for independent or long-running work.
  </purpose>

  <rules>
    <rule>Workflow replaces ask.</rule>
    <rule>Workflow operations return immediately unless wait(), join(), or result() is called explicitly.</rule>
    <rule>wait() runs until completion, failure, timeout, interruption, or checkpoint.</rule>
    <rule>checkpoint returns control to the main Agent for direction adjustment.</rule>
    <rule>Use graph() or describe_graph() to review task graph settings.</rule>
    <rule>Use inspect(node) to view a node's final model output.</rule>
    <rule>Use graph modification APIs to adjust pending tasks after checkpoints.</rule>
  </rules>

  <model_level_policy>
    <rule>Pass model_level on a node, default_model_level on workflow.start(), or omit both to use the configured workflow/global default.</rule>
    <rule>If model_levels contains workflow_default, it is the configured default for workflow nodes.</rule>
  </model_level_policy>

  <state_policy>
    <rule>Current workflow state is not updated in this block.</rule>
    <rule>Use wait(), snapshot(), graph(), inspect(), or list() for current workflow state.</rule>
    <rule>Active workflow snapshots are restored through the compaction summary section named "## Active workflows".</rule>
  </state_policy>

  <node_prompting>
    <rule>Workflow node agents do not receive this workflow_context block.</rule>
    <rule>Write node prompts as normal natural-language task details.</rule>
    <rule>Make node prompts self-contained: include goal, scope, constraints, expected output, and whether edits are allowed.</rule>
  </node_prompting>

  <examples>
    <example name="create_graph_and_wait_to_checkpoint">
      <description>Create a workflow, add agent nodes and a checkpoint, then wait until the next yield point.</description>
      <code>
from uv_agent_runtime import workflow

wf = workflow.start(objective="Implement workflow support")
scan = wf.agent_many(
    ["Investigate runner", "Investigate context", "Investigate TUI"],
    key="investigation",
    prompt=lambda item: f"{item}. Return findings and risks; do not edit files.",
    concurrency=3,
)
wf.checkpoint(
    key="after_investigation",
    after=scan,
    reason="Let the main Agent review direction before implementation.",
    options=["continue", "review", "branch", "takeover", "abort"],
)
result = wf.wait()
print(result.summary())
      </code>
    </example>

    <example name="resume_and_continue">
      <description>Resume a workflow after accepting the checkpoint direction.</description>
      <code>
from uv_agent_runtime import workflow

wf = workflow.resume("wf_123")
wf.continue_checkpoint("after_investigation")
result = wf.wait()
print(result.summary())
      </code>
    </example>
  </examples>
</workflow_context>"""


def render_workflow_context() -> str:
    """Return stable main-Agent-only workflow guidance."""

    return WORKFLOW_CONTEXT_TEXT


def active_workflow_snapshots(
    data_dir: Path,
    *,
    parent_thread_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return compact active workflow snapshots for compaction handoff."""

    try:
        with connect_state_db(data_dir) as db:
            if not _table_exists(db, "workflows"):
                return []
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
    except sqlite3.Error:
        return []


def active_workflows_compaction_section(data_dir: Path, *, parent_thread_id: str) -> str:
    """Render the deterministic ``## Active workflows`` compaction section."""

    snapshots = active_workflow_snapshots(data_dir, parent_thread_id=parent_thread_id)
    if not snapshots:
        return ""
    lines = ["## Active workflows", ""]
    for snapshot in snapshots:
        progress = snapshot.get("progress") if isinstance(snapshot.get("progress"), dict) else {}
        progress_text = ", ".join(f"{key}={progress[key]}" for key in sorted(progress)) or "no nodes"
        lines.append(f"- `{snapshot['workflow_id']}` status={snapshot['status']} objective={snapshot['objective']}")
        lines.append(f"  - Progress: {progress_text}")
        checkpoint = snapshot.get("current_checkpoint")
        if isinstance(checkpoint, dict) and checkpoint:
            lines.append(
                "  - Current checkpoint: "
                f"{checkpoint.get('key') or checkpoint.get('checkpoint_id')} "
                f"({checkpoint.get('reason') or 'no reason recorded'})"
            )
        inspectable = [node for node in snapshot.get("nodes", []) if isinstance(node, dict) and node.get("status") == "completed"]
        if inspectable:
            refs = ", ".join(str(node.get("key") or node.get("node_id")) for node in inspectable[:8])
            lines.append(f"  - Completed inspectable nodes: {refs}")
        lines.append(
            "  - Resume: "
            f"from uv_agent_runtime import workflow; wf = workflow.resume(\"{snapshot['workflow_id']}\")"
        )
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


def _table_exists(db: sqlite3.Connection, name: str) -> bool:
    row = db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone()
    return row is not None


def _json_loads(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default
