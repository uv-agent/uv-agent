from __future__ import annotations

import pytest

from uv_agent.config import SchedulerConfig
from uv_agent.scheduler import SchedulerService
from uv_agent.state_db import connect_state_db
import asyncio


class Helpers:
    def __init__(self) -> None:
        self.calls = []

    def resolve(self, name: str):
        return {"found": name == "demo", "name": name}

    async def call(self, name: str, args=None, kwargs=None):
        self.calls.append((name, dict(kwargs or {})))
        return {"called": name, "payload": dict(kwargs or {})}


def test_scheduler_create_update_list_delete_helper(tmp_path):
    helpers = Helpers()
    service = SchedulerService(tmp_path, SchedulerConfig(), helpers.resolve, helpers.call)

    schedule = service.create(kind="interval", every={"minutes": 5}, helper="demo", payload={"x": 1}, name="demo job")

    assert schedule["schedule_id"].startswith("sch_")
    assert schedule["action"] == {"type": "helper.call", "helper": "demo", "payload": {"x": 1}}
    assert schedule["timing"]["every_seconds"] == 300
    assert service.list()[0]["schedule_id"] == schedule["schedule_id"]

    updated = service.update(schedule["schedule_id"], enabled=False, payload={"x": 2}, helper="demo")
    assert updated["enabled"] is False
    assert updated["action"]["payload"] == {"x": 2}

    assert service.delete(schedule["schedule_id"]) == {"deleted": True, "schedule_id": schedule["schedule_id"]}
    assert service.list() == []


def test_scheduler_validates_action_shape(tmp_path):
    service = SchedulerService(tmp_path, SchedulerConfig(), lambda name: {"found": False}, None)

    with pytest.raises(ValueError):
        service.create(kind="interval", every={"minutes": 5}, helper="x", prompt="y")
    with pytest.raises(LookupError):
        service.create(kind="interval", every={"minutes": 5}, helper="missing")
    with pytest.raises(ValueError):
        service.create(kind="interval", every={"minutes": 5}, helper="missing", allow_missing=True, conflict="guide")


@pytest.mark.asyncio
async def test_scheduler_run_now_records_history(tmp_path):
    helpers = Helpers()
    service = SchedulerService(tmp_path, SchedulerConfig(), helpers.resolve, helpers.call)
    schedule = service.create(kind="interval", every={"minutes": 5}, helper="demo", payload={"x": 1})

    result = await service.run_now(schedule["schedule_id"])

    assert result["status"] == "completed"
    assert result["result"]["payload"] == {"x": 1}
    with connect_state_db(tmp_path) as db:
        row = db.execute("SELECT * FROM schedule_runs WHERE run_id = ?", (result["run_id"],)).fetchone()
    assert row is not None
    assert row["status"] == "completed"
    assert row["schedule_snapshot_json"]


@pytest.mark.asyncio
async def test_scheduler_run_due_once_advances_interval_and_runs_helper(tmp_path):
    helpers = Helpers()
    service = SchedulerService(tmp_path, SchedulerConfig(), helpers.resolve, helpers.call)
    schedule = service.create(kind="interval", every={"seconds": 60}, helper="demo", payload={"x": 3})
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
    assert helpers.calls == [("demo", {"x": 3})]
    with connect_state_db(tmp_path) as db:
        schedule_row = db.execute("SELECT next_run_at FROM schedules WHERE schedule_id = ?", (schedule["schedule_id"],)).fetchone()
    assert schedule_row["next_run_at"] != "2000-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_scheduler_prompt_action_creates_workflow(tmp_path):
    service = SchedulerService(tmp_path, SchedulerConfig(), lambda name: {"found": False}, None)
    schedule = service.create(kind="interval", every={"minutes": 5}, prompt="Do scheduled work", objective="Scheduled objective")

    result = await service.run_now(schedule["schedule_id"])

    assert result["status"] == "completed"
    assert result["workflow_id"].startswith("wf_")
    with connect_state_db(tmp_path) as db:
        workflow = db.execute("SELECT * FROM workflows WHERE workflow_id = ?", (result["workflow_id"],)).fetchone()
        node = db.execute("SELECT * FROM workflow_nodes WHERE workflow_id = ?", (result["workflow_id"],)).fetchone()
    assert workflow["objective"] == "Scheduled objective"
    assert node["prompt"] == "Do scheduled work"
