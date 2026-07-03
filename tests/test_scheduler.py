from __future__ import annotations

import asyncio

import pytest

from uv_agent.plugins.registry import ActionRegistry
from uv_agent.builtin.scheduler.service import SchedulerConfig, SchedulerService
from uv_agent.state_db import connect_state_db


class Actions:
    def __init__(self) -> None:
        self.calls = []
        self.registry = ActionRegistry()
        self.registry.register(
            plugin="demo-plugin",
            action_id="demo.run",
            handler=self._run,
            schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        )

    def resolve(self, action_id: str):
        return self.registry.get(action_id)

    async def call(self, action_id: str, payload=None, *, context=None):
        self.calls.append((action_id, dict(payload or {}), context))
        return await self.registry.call(action_id, payload or {}, context=context)

    async def _run(self, payload, context=None):
        return {
            "called": "demo.run",
            "payload": dict(payload),
            "schedule_id": context.schedule["schedule_id"] if context else None,
        }


def test_scheduler_create_update_list_delete_action(tmp_path):
    actions = Actions()
    service = SchedulerService(tmp_path, SchedulerConfig(), actions.resolve, actions.call)

    schedule = service.create(kind="interval", every={"minutes": 5}, action_id="demo.run", payload={"x": 1}, name="demo job")

    assert schedule["schedule_id"].startswith("sch_")
    assert schedule["action"] == {"type": "action.call", "action_id": "demo.run", "payload": {"x": 1}}
    assert schedule["timing"]["every_seconds"] == 300
    assert service.list()[0]["schedule_id"] == schedule["schedule_id"]

    updated = service.update(schedule["schedule_id"], enabled=False, payload={"x": 2})
    assert updated["enabled"] is False
    assert updated["action"]["action_id"] == "demo.run"
    assert updated["action"]["payload"] == {"x": 2}

    assert service.delete(schedule["schedule_id"]) == {"deleted": True, "schedule_id": schedule["schedule_id"]}
    assert service.list() == []


def test_scheduler_validates_action_shape(tmp_path):
    actions = Actions()
    service = SchedulerService(tmp_path, SchedulerConfig(), actions.resolve, actions.call)

    with pytest.raises(ValueError):
        service.create(kind="interval", every={"minutes": 5}, helper="x", prompt="y")
    with pytest.raises(LookupError):
        service.create(kind="interval", every={"minutes": 5}, action_id="missing")
    with pytest.raises(ValueError):
        service.create(kind="interval", every={"minutes": 5}, payload={})
    schedule = service.create(kind="interval", every={"minutes": 5}, action_id="missing", allow_missing=True)
    assert schedule["action"]["action_id"] == "missing"


@pytest.mark.asyncio
async def test_scheduler_run_now_records_history(tmp_path):
    actions = Actions()
    service = SchedulerService(tmp_path, SchedulerConfig(), actions.resolve, actions.call)
    schedule = service.create(kind="interval", every={"minutes": 5}, action_id="demo.run", payload={"x": 1})

    result = await service.run_now(schedule["schedule_id"])

    assert result["status"] == "completed"
    assert result["result"]["payload"] == {"x": 1}
    assert result["result"]["schedule_id"] == schedule["schedule_id"]
    assert actions.calls[0][0] == "demo.run"
    assert actions.calls[0][2].run_id == result["run_id"]
    with connect_state_db(tmp_path) as db:
        row = db.execute("SELECT * FROM schedule_runs WHERE run_id = ?", (result["run_id"],)).fetchone()
    assert row is not None
    assert row["status"] == "completed"
    assert row["schedule_snapshot_json"]


@pytest.mark.asyncio
async def test_scheduler_run_due_once_advances_interval_and_runs_action(tmp_path):
    actions = Actions()
    service = SchedulerService(tmp_path, SchedulerConfig(), actions.resolve, actions.call)
    schedule = service.create(kind="interval", every={"seconds": 60}, action_id="demo.run", payload={"x": 3})
    with connect_state_db(tmp_path) as db:
        db.execute("UPDATE schedules SET next_run_at = '2000-01-01T00:00:00+00:00' WHERE schedule_id = ?", (schedule["schedule_id"],))

    started = await service.run_due_once()
    for _ in range(20):
        with connect_state_db(tmp_path) as db:
            row = db.execute("SELECT status FROM schedule_runs WHERE schedule_id = ?", (schedule["schedule_id"],)).fetchone()
        if row is not None and row["status"] == "completed":
            break
        await asyncio.sleep(0.01)

    assert started[0]["schedule_id"] == schedule["schedule_id"]
    assert [(call[0], call[1]) for call in actions.calls] == [("demo.run", {"x": 3})]
    with connect_state_db(tmp_path) as db:
        schedule_row = db.execute("SELECT next_run_at FROM schedules WHERE schedule_id = ?", (schedule["schedule_id"],)).fetchone()
    assert schedule_row["next_run_at"] != "2000-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_scheduler_subagent_prompt_action_uses_stable_schedule_thread(tmp_path):
    from uv_agent.builtin.subagent import setup as setup_subagent
    from uv_agent.plugins import PluginManifest
    from uv_agent.plugins.context import PluginContext
    from uv_agent.plugins.events import EventBus
    from uv_agent.plugins.i18n import PluginI18nRegistry
    from uv_agent.plugins.registry import CommandRegistry, RuntimeNamespaceRegistry, UiRegistry
    from uv_agent.plugins.storage import PluginStorage
    from uv_agent.session import ThreadStore

    class Submitted:
        def __init__(self, thread_id: str | None) -> None:
            self.thread_id = thread_id
            self.turn_id = "turn_scheduled"
            self.status = "completed"
            self.final_text = "scheduled done"
            self.error = None

        async def wait(self):
            return self

    submitted_calls = []

    async def submitter(**kwargs):
        submitted_calls.append(dict(kwargs))
        return Submitted(kwargs.get("thread_id"))

    actions = ActionRegistry()
    thread_store = ThreadStore(tmp_path)
    plugin_context = PluginContext(
        manifest=PluginManifest("builtin.subagent", "0", "Subagent", "test"),
        project_root=tmp_path,
        user_state_dir=tmp_path / "user",
        config={},
        events=EventBus(),
        logger=__import__("logging").getLogger("test"),
        runtime_registry=RuntimeNamespaceRegistry(),
        action_registry=actions,
        command_registry=CommandRegistry(),
        ui_registry=UiRegistry(),
        i18n_registry=PluginI18nRegistry(),
        context_broker=__import__("uv_agent.plugins.context", fromlist=["PluginContextBroker"]).PluginContextBroker(),
        storage=PluginStorage("builtin.subagent", tmp_path, tmp_path / "user"),
        submitter=None,
        task_factory=lambda plugin, coro, name=None: asyncio.create_task(coro),
        compaction_section_providers=[],
        epoch_context_refreshers=[],
        thread_store=thread_store,
    )
    setup_subagent(plugin_context)
    scheduler_context = PluginContext(
        manifest=PluginManifest("builtin.scheduler", "0", "Scheduler", "test"),
        project_root=tmp_path,
        user_state_dir=tmp_path / "user",
        config={},
        events=EventBus(),
        logger=__import__("logging").getLogger("test"),
        runtime_registry=RuntimeNamespaceRegistry(),
        action_registry=actions,
        command_registry=CommandRegistry(),
        ui_registry=UiRegistry(),
        i18n_registry=PluginI18nRegistry(),
        context_broker=__import__("uv_agent.plugins.context", fromlist=["PluginContextBroker"]).PluginContextBroker(),
        storage=PluginStorage("builtin.scheduler", tmp_path, tmp_path / "user"),
        submitter=None,
        task_factory=lambda plugin, coro, name=None: asyncio.create_task(coro),
        compaction_section_providers=[],
        epoch_context_refreshers=[],
        thread_store=thread_store,
    )
    service = SchedulerService(
        tmp_path,
        SchedulerConfig(),
        actions.get,
        actions.call,
        threads=scheduler_context.threads,
        submitter=submitter,
    )
    schedule = service.create(
        kind="interval",
        every={"minutes": 5},
        action_id="subagent.prompt",
        payload={"prompt": "Do scheduled work", "level": "large"},
        name="subagent job",
    )

    result = await service.run_now(schedule["schedule_id"])

    assert result["status"] == "completed"
    assert "workflow_id" not in result
    assert result["result"]["status"] == "completed"
    assert result["result"]["final_text"] == "scheduled done"
    assert result["result"]["thread_id"].startswith("thr_")
    assert submitted_calls == [
        {
            "text": "Do scheduled work",
            "thread_id": result["result"]["thread_id"],
            "level": "large",
            "conflict": "queue",
        }
    ]
    metadata = thread_store.thread_metadata(result["result"]["thread_id"])
    assert metadata["owner_plugin"] == "builtin.scheduler"
    assert metadata["external_source"] == "scheduler"
    assert metadata["external_id"] == schedule["schedule_id"]
    with connect_state_db(tmp_path) as db:
        row = db.execute("SELECT workflow_id FROM schedule_runs WHERE run_id = ?", (result["run_id"],)).fetchone()
    assert row["workflow_id"] is None
