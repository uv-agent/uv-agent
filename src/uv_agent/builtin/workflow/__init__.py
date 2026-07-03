from __future__ import annotations

from typing import Any

from uv_agent.plugins import PluginManifest, SetupPlugin
from .context import active_workflows_compaction_section, render_workflow_context
from .executor import WorkflowExecutor
from .rpc import runtime_functions
from . import service


_EXECUTORS: dict[int, WorkflowExecutor] = {}

MANIFEST = PluginManifest(
    id="builtin.workflow",
    version="0.1.0",
    display_name={"zh": "工作流", "en": "Workflow"},
    description={"zh": "持久化工作流任务图、节点执行器和计划任务 prompt action。", "en": "Persistent workflow task graphs, node execution, and scheduled prompt actions."},
    builtin=True,
    default_enabled=False,
    deprecated=True,
    deprecation_message=(
        "builtin.workflow is deprecated. Enable it only for legacy persistent workflow graphs; "
        "use builtin.subagent and action_id='subagent.prompt' for new scheduled or delegated agent work."
    ),
    priority=300,
    capabilities=("runtime_namespace", "action", "context"),
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup, stop=stop)


def setup(context) -> None:
    _publish_workflow_context(context)
    context.epoch.on_refresh(lambda thread_id=None: _publish_workflow_context(context))
    context.compaction.summary_section(
        lambda thread_id: active_workflows_compaction_section(context.storage.project_data_dir, parent_thread_id=thread_id)
    )
    if context.threads is not None and context.can_submit_turn:
        executor = WorkflowExecutor(context.storage.project_data_dir, context.submit_turn, context.threads)
        _EXECUTORS[id(context)] = executor
        executor.start()
    context.runtime.register_namespace(
        "workflow",
        doc="持久化工作流任务图 helper，用于创建、等待、检查和恢复长期任务图。",
        module="uv_agent.builtin.workflow.runtime",
        functions=runtime_functions(context.storage.project_data_dir),
    )
    context.actions.register(
        "workflow.prompt",
        _scheduled_prompt,
        doc="从计划任务 action 创建一个 workflow 并启动 prompt 节点。",
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


def _publish_workflow_context(context) -> None:
    context.epoch.publish(tag="workflow_context", body=render_workflow_context(), attrs={"scope": "main_agent"})


async def stop(context) -> None:
    executor = _EXECUTORS.pop(id(context), None)
    if executor is not None:
        await executor.stop()


def _scheduled_prompt(payload: dict[str, Any], context=None) -> dict[str, Any]:
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
    wf = service.start(
        objective,
        default_model_level=model_level if isinstance(model_level, str) and model_level else None,
        state_dir=context.data_dir,
    )
    if thread_id:
        with service.connect_workflow_db(context.data_dir) as db:
            db.execute("UPDATE workflows SET parent_thread_id = ? WHERE workflow_id = ?", (thread_id, wf.workflow_id))
    wf.agent(
        str(payload.get("prompt") or ""),
        model_level=model_level if isinstance(model_level, str) and model_level else None,
        timeout_s=float(timeout_s) if isinstance(timeout_s, (int, float)) else None,
    )
    return {"workflow_id": wf.workflow_id, "thread_id": thread_id}
