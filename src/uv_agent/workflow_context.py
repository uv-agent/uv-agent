from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from uv_agent.prompts import (
    ACTIVE_WORKFLOW_CHECKPOINT_LINE_TEMPLATE,
    ACTIVE_WORKFLOW_COMPLETED_INSPECTABLE_LINE_TEMPLATE,
    ACTIVE_WORKFLOW_NO_NODES,
    ACTIVE_WORKFLOW_NO_REASON_RECORDED,
    ACTIVE_WORKFLOW_PROGRESS_LINE_TEMPLATE,
    ACTIVE_WORKFLOW_RESUME_LINE_TEMPLATE,
    ACTIVE_WORKFLOW_STATUS_LINE_TEMPLATE,
    ACTIVE_WORKFLOWS_SECTION_TITLE,
    WORKFLOW_CONTEXT_TEXT,
)
from uv_agent.state_db import connect_state_db

ACTIVE_WORKFLOW_STATUSES = {"running", "checkpoint", "failed"}



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
