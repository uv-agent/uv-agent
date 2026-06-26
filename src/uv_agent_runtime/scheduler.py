from __future__ import annotations

from datetime import datetime, timedelta
import os
from typing import Any

from . import transport


def create(**kwargs: Any) -> dict[str, Any]:
    return transport.call_host("scheduler.create", **_normalize(kwargs))


def update(schedule_id: str, **changes: Any) -> dict[str, Any]:
    return transport.call_host("scheduler.update", schedule_id=schedule_id, **_normalize(changes))


def list(**filters: Any) -> list[dict[str, Any]]:
    return transport.call_host("scheduler.list", **filters)


def delete(schedule_id: str) -> dict[str, Any]:
    return transport.call_host("scheduler.delete", schedule_id=schedule_id)


def run_now(schedule_id: str) -> dict[str, Any]:
    return transport.call_host("scheduler.run_now", schedule_id=schedule_id)


def _normalize(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    every = result.get("every")
    if isinstance(every, timedelta):
        result["every"] = {"seconds": every.total_seconds()}
    at = result.get("at")
    if isinstance(at, datetime):
        result["at"] = at.isoformat()
    if result.get("prompt") and not result.get("thread_id"):
        thread_id = os.environ.get("UV_AGENT_RUNTIME_THREAD_ID")
        if thread_id:
            result["thread_id"] = thread_id
    return result
