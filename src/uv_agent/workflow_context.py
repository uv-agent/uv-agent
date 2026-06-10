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
<example name="create_investigation_graph">
<description>Create a suitable task graph for a long task, then wait until the first checkpoint.</description>
<code>
from uv_agent_runtime import workflow

wf = workflow.start(
    objective="Design and prepare a plugin system for uv-agent",
    default_model_level="deepseek-pro",
)
architecture = wf.agent(
    '''Design the plugin system architecture for uv-agent.

## Objective and task
- Read src/uv_agent/, src/uv_agent_runtime/, and AGENTS.md to understand the host/runtime split.
- Compare plugin mechanisms that fit a Python coding agent with a single run_python action surface.
- Recommend the two most suitable architecture options for this repository.

## Requirements and notes
- Do not edit files; this is an investigation node only.
- Cover compatibility with skills, MCP discovery, runtime helpers, and project configuration.
- Return trade-offs, risks, required dependencies, and source locations that constrain the design.''',
    key="investigate.architecture",
)
hooks = wf.agent(
    '''Map plugin hook points across uv-agent.

## Objective and task
- Inspect host, model client, session store, runner, context, and TUI modules.
- List hook points where plugins could observe, modify, or extend behavior.
- Include expected input/output contracts for each hook.

## Requirements and notes
- Do not edit files; this is an investigation node only.
- Cite code with file:line references so the main Agent can jump directly to the relevant implementation.
- Separate safe read-only hooks from hooks that can change execution behavior.''',
    key="investigate.hooks",
)
runtime = wf.agent(
    '''Assess runtime and packaging constraints for uv-agent plugins.

## Objective and task
- Inspect managed script execution, uv_agent_runtime exports, helper tracking, and project state storage.
- Identify constraints for plugins that expose runtime helpers or interact with managed scripts.
- Propose how plugin metadata should be discovered and persisted.

## Requirements and notes
- Do not edit files; this is an investigation node only.
- Pay special attention to the run_python boundary, environment isolation, and prompt-cache stability.
- Return risks, test targets, and any decisions that must be made by the main Agent.''',
    key="investigate.runtime",
)
wf.checkpoint(
    key="after_investigation",
    after=[architecture, hooks, runtime],
    reason="Review the investigation outputs before choosing the implementation graph.",
    options=["continue", "revise graph", "branch alternative", "take over", "cancel"],
    recommended_action="Inspect the investigation nodes, then decide whether to continue, modify, or branch the graph.",
)
result = wf.wait(timeout_s=1800)
print(result.summary())
</code>
</example>
<example name="inspect_first_checkpoint_and_extend_graph">
<description>Resume at the first checkpoint, inspect completed nodes, then add the next task graph segment.</description>
<code>
from uv_agent_runtime import workflow

wf = workflow.resume("wf_123")
print(wf.describe_graph())
print(wf.inspect("investigate.architecture"))
print(wf.inspect("investigate.hooks"))
print(wf.inspect("investigate.runtime"))

# After inspecting the checkpoint, record the main-Agent decision and add the next segment.
wf.continue_checkpoint(
    "after_investigation",
    resolution={
        "decision": "continue with implementation and review nodes",
        "reason": "the investigation outputs agree on a small host-side plugin manager plus explicit runtime helper registration",
    },
)
host_impl = wf.agent(
    '''Implement the host-side plugin manager for uv-agent.

## Objective and task
- Implement the approved plugin discovery and lifecycle design in the host application.
- Add focused tests for configuration loading, plugin registration, and failure isolation.
- Keep the implementation consistent with the investigation outputs inspected by the main Agent.

## Requirements and notes
- Edit only source and tests needed for the host-side plugin manager.
- Do not change TUI rendering or runtime helper exports in this node.
- Return changed files, important design choices, verification commands, and remaining risks.''',
    key="implement.host",
    after=["investigate.architecture", "investigate.hooks", "investigate.runtime"],
)
runtime_impl = wf.agent(
    '''Implement runtime-helper integration for approved plugins.

## Objective and task
- Add the minimal runtime-side integration needed for plugin-provided helpers.
- Preserve the managed run_python boundary and avoid relying on repository checkout import paths.
- Add focused tests for helper discovery, helper context rendering, and helper-call tracking.

## Requirements and notes
- Edit only runtime/helper integration code and focused tests.
- Do not introduce network calls or plugin execution outside the managed Python boundary.
- Return changed files, verification commands, compatibility risks, and any required follow-up.''',
    key="implement.runtime",
    after=["investigate.architecture", "investigate.runtime"],
)
review = wf.review(
    key="review.integration",
    after=[host_impl, runtime_impl],
    prompt='''Review the plugin implementation before final verification.

## Objective and task
- Check whether the host and runtime changes match the approved graph and investigation constraints.
- Look for context pollution, unsafe plugin execution, migration gaps, and missing tests.
- Decide whether the main Agent should verify, adjust the graph, or take over.

## Requirements and notes
- Do not edit files; this is a review node only.
- Return exactly one recommendation: approve, request changes, or change the graph.
- If you request graph changes, name the node to update, replace, or add.''',
)
wf.checkpoint(
    key="before_final_verification",
    after=review,
    reason="Review implementation results before final verification.",
    options=["continue", "replace node", "branch alternative", "take over", "cancel"],
    recommended_action="Inspect review.integration, then decide whether to verify or adjust the graph.",
)
result = wf.wait(timeout_s=3600)
print(result.summary())
</code>
</example>
<example name="inspect_review_checkpoint_and_finalize">
<description>Inspect a later checkpoint, optionally adjust pending work, then add final verification.</description>
<code>
from uv_agent_runtime import workflow

wf = workflow.resume("wf_123")
print(wf.describe_graph())
print(wf.inspect("review.integration"))

# If the review asks for changes, modify the graph before continuing instead of adding verification.
# For example, add a corrective node and another checkpoint, or replace a completed node with a revised prompt.

wf.continue_checkpoint(
    "before_final_verification",
    resolution={
        "decision": "run final verification",
        "reason": "review.integration approved the implementation with no blocking changes",
    },
)
verify = wf.agent(
    '''Verify the plugin-system changes for uv-agent.

## Objective and task
- Run the focused plugin tests first, then the broader test suite if focused tests pass.
- Investigate failures and apply only minimal fixes needed to make verification meaningful.
- Summarize whether the main Agent should commit, revise the graph, or take over manually.

## Requirements and notes
- Keep edits minimal and directly tied to verification failures.
- Report every command run and whether it passed, failed, or timed out.
- Return final status, residual risks, and recommended next action for the main Agent.''',
    key="verify.final",
    after="before_final_verification",
)
result = wf.wait(timeout_s=3600, until="completed")
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
