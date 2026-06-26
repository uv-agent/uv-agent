from __future__ import annotations

import pytest

from uv_agent.config import SchedulerConfig
from uv_agent.scheduler import SchedulerService
from uv_agent.state_db import connect_state_db


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
