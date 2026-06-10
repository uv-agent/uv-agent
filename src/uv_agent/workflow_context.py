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
<example name="long_task_control_flow">
<description>Coordinate investigation, implementation, review, checkpoints, graph adjustment, and final verification for a long task.</description>
<code>
from uv_agent_runtime import workflow

wf = workflow.start(
    objective="Ship a multi-area refactor with investigation, implementation, review, and verification",
    default_model_level="deepseek-pro",
)
runtime = wf.agent(
    '''Goal: investigate the runner/runtime side of the refactor.
Scope: inspect src/uv_agent_runtime, src/uv_agent/runner, and tests that cover managed scripts.
Do not edit files. Identify exact files/functions, risks, migration constraints, and focused tests.
Return: a concise implementation plan with blockers, edge cases, and confidence level.''',
    key="investigate.runtime",
)
context = wf.agent(
    '''Goal: investigate prompt, context, and compaction effects for the refactor.
Scope: inspect context builders, compaction filters, and tests; do not edit files.
Return: required context changes, cache-stability risks, and exact tests that should prove the behavior.
Call out any wording that would pollute model context or create stale instructions.''',
    key="investigate.context",
)
ui = wf.agent(
    '''Goal: investigate how the TUI should surface progress for this long task.
Scope: inspect tui2 rendering plus legacy formatting helpers; do not edit files.
Return: the compact transcript events to show, what should stay hidden, and tests to update.
Keep the recommendation compatible with a single transcript and bottom composer.''',
    key="investigate.ui",
)
wf.checkpoint(
    key="after_investigation",
    after=[runtime, context, ui],
    reason="Review investigation outputs before choosing the implementation path.",
    options=["continue", "revise graph", "branch alternative", "take over", "cancel"],
    recommended_action="Inspect investigation nodes, adjust pending graph if needed, then continue.",
)
first = wf.wait(timeout_s=1800)
print(first.summary())

if first.status == "checkpoint":
    print(wf.describe_graph())
    print(wf.inspect("investigate.runtime"))
    print(wf.inspect("investigate.context"))
    print(wf.inspect("investigate.ui"))
    wf.continue_checkpoint(
        "after_investigation",
        resolution={"decision": "implement runtime and context first, then add UI display"},
    )
    implementation = wf.agent_many(
        [
            {
                "area": "runtime",
                "prompt": '''Goal: implement the runtime/store portion of the approved plan.
Allowed edits: source and focused tests only.
Use the investigation findings reviewed by the main Agent as constraints.
Return: changed files, rationale, verification command, and remaining risks.''',
            },
            {
                "area": "context",
                "prompt": '''Goal: implement prompt/context/compaction changes for the approved plan.
Allowed edits: source and focused tests only.
Preserve prompt-cache stability and avoid adding stale state to stable context.
Return: changed files, rationale, verification command, and remaining risks.''',
            },
            {
                "area": "ui",
                "prompt": '''Goal: implement compact TUI progress display for the approved plan.
Allowed edits: tui2/formatting code and focused tests only.
Do not duplicate model protocol or runner logic inside the UI.
Return: changed files, display behavior, verification command, and remaining risks.''',
            },
        ],
        key="implement",
        prompt=lambda item: item["prompt"],
        concurrency=2,
        after=[runtime, context, ui],
    )
    review = wf.review(
        key="review.integration",
        after=implementation,
        prompt='''Goal: review the integrated implementation.
Check correctness, context stability, migration compatibility, UI compactness, and test coverage.
Do not edit files. Return approve, request changes, or propose a replacement node with exact reasons.''',
    )
    wf.checkpoint(
        key="before_final_verification",
        after=review,
        reason="Review the implementation and decide whether to patch, branch, or verify.",
        options=["continue", "replace node", "branch alternative", "take over", "cancel"],
        recommended_action="Inspect review.integration, modify pending graph if needed, then continue.",
    )
    second = wf.wait(timeout_s=3600)
    print(second.summary())
    if second.status == "checkpoint":
        print(wf.inspect("review.integration"))
        wf.continue_checkpoint(
            "before_final_verification",
            resolution={"decision": "run final verification and summarize remaining risk"},
        )
        verify = wf.agent(
            '''Goal: run final verification for the integrated changes.
Scope: run focused tests first, then the broader suite if focused tests pass.
Allowed edits: only minimal fixes for verification failures; report unexpected design drift before large rewrites.
Return: commands run, pass/fail summary, residual risks, and whether the main Agent should commit.''',
            key="verify.final",
            after="before_final_verification",
        )
        final = wf.wait(timeout_s=3600, until="completed")
        print(final.summary())
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
