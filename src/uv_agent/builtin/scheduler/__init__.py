from __future__ import annotations

from uv_agent.plugins import PluginManifest, SetupPlugin
from .service import SchedulerService, scheduler_config_from_plugin_config


_SERVICES: dict[int, SchedulerService] = {}

SCHEDULER_HELPER_SIGNATURE = """ScheduleKind = Literal["once", "interval", "cron"]
MisfirePolicy = Literal["skip", "run_once", "catch_up"]
OverlapPolicy = Literal["skip", "allow", "queue", "replace"]
Schedule = {schedule_id: str, name: str | None, description: str | None, kind: ScheduleKind, enabled: bool, action: dict[str, Any], timing: dict[str, Any], timezone: str | None, next_run_at: str | None, misfire_policy: MisfirePolicy, overlap_policy: OverlapPolicy, metadata: dict[str, Any], created_at: str, updated_at: str}
rt.scheduler.create(*, action_id: str | None = None, action: str | None = None, payload: dict[str, Any] | None = None, kind: ScheduleKind, name: str | None = None, description: str | None = None, enabled: bool = True, at: str | datetime | None = None, every: dict[str, int] | timedelta | None = None, cron: str | None = None, timezone: str | None = None, misfire_policy: MisfirePolicy | None = None, overlap_policy: OverlapPolicy | None = None, metadata: dict[str, Any] | None = None, allow_missing: bool = False) -> Schedule
rt.scheduler.update(schedule_id: str, **changes: Any) -> Schedule
rt.scheduler.list(**filters: Any) -> list[Schedule]
rt.scheduler.delete(schedule_id: str) -> dict[str, Any]
rt.scheduler.run_now(schedule_id: str) -> dict[str, Any]"""


MANIFEST = PluginManifest(
    id="builtin.scheduler",
    version="0.1.0",
    display_name={"zh": "计划任务", "en": "Scheduler"},
    description={"zh": "持久化计划任务管理和调度执行。", "en": "Persistent schedule management and execution."},
    builtin=True,
    priority=400,
    capabilities=("runtime_namespace", "context"),
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup, stop=stop)


def setup(context) -> None:
    service = SchedulerService(
        context.storage.project_data_dir,
        scheduler_config_from_plugin_config(context.config),
        action_resolver=context.actions.resolve,
        action_caller=context.actions.call,
        threads=context.threads,
        submitter=context.submit_turn if context.can_submit_turn else None,
    )
    _SERVICES[id(context)] = service
    context.runtime.register_namespace(
        "scheduler",
        doc="管理持久化计划任务。可创建、更新、列出、删除任务，或立即运行指定任务。",
        functions={
            "create": service.create,
            "update": lambda schedule_id, **changes: service.update(str(schedule_id), **changes),
            "list": service.list,
            "delete": lambda schedule_id: service.delete(str(schedule_id)),
            "run_now": lambda schedule_id: service.run_now(str(schedule_id)),
        },
        docs={
            "create": "创建一个持久化计划任务。",
            "update": "更新指定计划任务。",
            "list": "列出计划任务。",
            "delete": "删除指定计划任务。",
            "run_now": "立即运行指定计划任务。",
        },
        schemas={name: {"type": "object"} for name in ("create", "update", "list", "delete", "run_now")},
    )
    _publish_scheduler_context(context)
    context.epoch.on_refresh(lambda thread_id=None: _publish_scheduler_context(context))
    service.start()


def _publish_scheduler_context(context) -> None:
    context.epoch.publish(
        tag="scheduler_helpers",
        body={
            "instructions": [
                "使用 rt.scheduler.* helpers 管理持久化计划任务。",
                "计划任务只调用 action registry；不要使用旧的 helper 或 prompt 字段。",
                '使用 action_id="subagent.prompt" 可定时向线程提交 prompt；payload 中传 prompt/level/timeout_s。',
                "使用 once 任务时传 at；使用 interval 任务时传 every；使用 cron 任务时传 cron。",
            ],
            "helper": {
                "import": "import uv_agent_runtime as rt",
                "namespace": "rt.scheduler",
                "signature": SCHEDULER_HELPER_SIGNATURE,
            },
        },
    )


async def stop(context) -> None:
    service = _SERVICES.pop(id(context), None)
    if service is not None:
        await service.stop()
