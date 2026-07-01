from __future__ import annotations

from typing import Any

from uv_agent.plugins import PluginManifest, SetupPlugin
from uv_agent.workflow_context import render_workflow_context
from uv_agent.state_db import connect_state_db


MANIFEST = PluginManifest(
    id="builtin.workflow",
    version="0.1.0",
    display_name="Workflow",
    description="Persistent workflow task graph runtime namespace and scheduled prompt action.",
    builtin=True,
    priority=300,
    capabilities=("runtime_namespace", "action", "context"),
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


def setup(context) -> None:
    context.context.epoch.publish(tag="workflow_context", body=render_workflow_context(), attrs={"scope": "main_agent"})
    context.runtime.register_namespace(
        "workflow",
        doc="Persistent workflow task graph helpers.",
        transport="local_module",
        module="uv_agent_runtime.workflow",
    )
    context.actions.register(
        "workflow.prompt",
        _scheduled_prompt,
        doc="Create a workflow from a scheduled prompt action.",
        schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "objective": {"type": "string"},
                "thread_id": {"type": ["string", "null"]},
                "model_level": {"type": ["string", "null"]},
                "timeout_s": {"type": ["number", "null"]},
            },
            "required": ["prompt"],
        },
    )


def _scheduled_prompt(payload: dict[str, Any], context=None) -> dict[str, Any]:
    import uv_agent_runtime.workflow as workflow

    if context is None:
        raise RuntimeError("workflow.prompt requires scheduler action context")
    schedule = dict(getattr(context, "schedule", {}) or {})
    thread_id = payload.get("thread_id") or context.schedule_thread()
    objective = str(
        payload.get("objective")
        or schedule.get("name")
        or f"Scheduled prompt {schedule.get('schedule_id') or ''}".strip()
        or "Scheduled prompt"
    )
    model_level = payload.get("model_level")
    timeout_s = payload.get("timeout_s")
    wf = workflow.start(
        objective,
        default_model_level=model_level if isinstance(model_level, str) and model_level else None,
        state_dir=context.data_dir,
    )
    if thread_id:
        with connect_state_db(context.data_dir) as db:
            db.execute("UPDATE workflows SET parent_thread_id = ? WHERE workflow_id = ?", (thread_id, wf.workflow_id))
    wf.agent(
        str(payload.get("prompt") or ""),
        model_level=model_level if isinstance(model_level, str) and model_level else None,
        timeout_s=float(timeout_s) if isinstance(timeout_s, (int, float)) else None,
    )
    return {"workflow_id": wf.workflow_id, "thread_id": thread_id}
