from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from uv_agent.ids import new_id
from uv_agent.state_db import connect_state_db
from uv_agent.time import utc_now_iso

MISFIRE_POLICIES = {"skip", "run_once", "catch_up"}
OVERLAP_POLICIES = {"skip", "allow", "queue", "replace"}
SCHEDULE_KINDS = {"once", "interval", "cron"}


@dataclass(frozen=True)
class SchedulerService:
    data_dir: Path
    config: Any
    helper_resolver: Any
    helper_caller: Any

    def create(self, **params: Any) -> dict[str, Any]:
        now = utc_now_iso()
        kind = str(params.get("kind") or "").strip()
        if kind not in SCHEDULE_KINDS:
            raise ValueError("kind must be one of: once, interval, cron")
        action = self._action(params)
        timing, next_run_at = self._timing(kind, params)
        schedule_id = new_id("sch")
        row = {
            "schedule_id": schedule_id,
            "name": params.get("name"),
            "description": params.get("description"),
            "kind": kind,
            "enabled": 1 if params.get("enabled", True) else 0,
            "action_json": _dumps(action),
            "timing_json": _dumps(timing),
            "timezone": params.get("timezone"),
            "next_run_at": next_run_at,
            "misfire_policy": _policy(params.get("misfire_policy"), getattr(self.config, "default_misfire_policy", "skip"), MISFIRE_POLICIES),
            "overlap_policy": _policy(params.get("overlap_policy"), getattr(self.config, "default_overlap_policy", "skip"), OVERLAP_POLICIES),
            "owner_type": "agent",
            "owner_name": None,
            "metadata_json": _dumps(params.get("metadata") or {}),
            "created_at": now,
            "updated_at": now,
        }
        with connect_state_db(self.data_dir) as db:
            db.execute(
                """
                INSERT INTO schedules(schedule_id, name, description, kind, enabled, action_json, timing_json,
                  timezone, next_run_at, misfire_policy, overlap_policy, owner_type, owner_name, metadata_json,
                  created_at, updated_at)
                VALUES (:schedule_id, :name, :description, :kind, :enabled, :action_json, :timing_json,
                  :timezone, :next_run_at, :misfire_policy, :overlap_policy, :owner_type, :owner_name,
                  :metadata_json, :created_at, :updated_at)
                """,
                row,
            )
        return self._public_row(row)

    def update(self, schedule_id: str, **changes: Any) -> dict[str, Any]:
        with connect_state_db(self.data_dir) as db:
            row = db.execute("SELECT * FROM schedules WHERE schedule_id = ?", (schedule_id,)).fetchone()
            if row is None:
                raise LookupError(f"Unknown schedule: {schedule_id}")
            data = dict(row)
            params = {**self._public_row(data), **changes}
            if "helper" in changes or "prompt" in changes:
                data["action_json"] = _dumps(self._action(params))
            if any(key in changes for key in ("kind", "at", "every", "cron", "timezone")):
                timing, next_run_at = self._timing(str(params.get("kind") or data["kind"]), params)
                data["kind"] = str(params.get("kind") or data["kind"])
                data["timing_json"] = _dumps(timing)
                data["next_run_at"] = next_run_at
                data["timezone"] = params.get("timezone")
            for key in ("name", "description"):
                if key in changes:
                    data[key] = changes[key]
            if "enabled" in changes:
                data["enabled"] = 1 if changes["enabled"] else 0
            for key, allowed in (("misfire_policy", MISFIRE_POLICIES), ("overlap_policy", OVERLAP_POLICIES)):
                if key in changes:
                    data[key] = _policy(changes[key], data[key], allowed)
            if "metadata" in changes:
                data["metadata_json"] = _dumps(changes.get("metadata") or {})
            data["updated_at"] = utc_now_iso()
            db.execute(
                """
                UPDATE schedules SET name=:name, description=:description, kind=:kind, enabled=:enabled,
                  action_json=:action_json, timing_json=:timing_json, timezone=:timezone, next_run_at=:next_run_at,
                  misfire_policy=:misfire_policy, overlap_policy=:overlap_policy, metadata_json=:metadata_json,
                  updated_at=:updated_at
                WHERE schedule_id=:schedule_id
                """,
                data,
            )
            return self._public_row(data)

    def list(self, *, enabled: bool | None = None, limit: int = 100) -> list[dict[str, Any]]:
        clauses = []
        args: list[Any] = []
        if enabled is not None:
            clauses.append("enabled = ?")
            args.append(1 if enabled else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with connect_state_db(self.data_dir) as db:
            rows = db.execute(f"SELECT * FROM schedules {where} ORDER BY updated_at DESC LIMIT ?", (*args, int(limit))).fetchall()
        return [self._public_row(dict(row)) for row in rows]

    def delete(self, schedule_id: str) -> dict[str, Any]:
        with connect_state_db(self.data_dir) as db:
            row = db.execute("SELECT * FROM schedules WHERE schedule_id = ?", (schedule_id,)).fetchone()
            if row is None:
                raise LookupError(f"Unknown schedule: {schedule_id}")
            db.execute("DELETE FROM schedules WHERE schedule_id = ?", (schedule_id,))
        return {"deleted": True, "schedule_id": schedule_id}

    async def run_now(self, schedule_id: str) -> dict[str, Any]:
        with connect_state_db(self.data_dir) as db:
            row = db.execute("SELECT * FROM schedules WHERE schedule_id = ?", (schedule_id,)).fetchone()
            if row is None:
                raise LookupError(f"Unknown schedule: {schedule_id}")
            schedule = dict(row)
        return await self._run_schedule(schedule, due_at=utc_now_iso())

    async def _run_schedule(self, schedule: dict[str, Any], *, due_at: str | None) -> dict[str, Any]:
        run_id = new_id("sr")
        action = _loads(schedule["action_json"], {})
        snapshot = self._public_row(schedule)
        started = utc_now_iso()
        with connect_state_db(self.data_dir) as db:
            db.execute(
                """
                INSERT INTO schedule_runs(run_id, schedule_id, status, action_json, schedule_snapshot_json, due_at, started_at)
                VALUES (?, ?, 'running', ?, ?, ?, ?)
                """,
                (run_id, schedule.get("schedule_id"), _dumps(action), _dumps(snapshot), due_at, started),
            )
        try:
            if action.get("type") == "helper.call":
                result = await self.helper_caller(action["helper"], kwargs=dict(action.get("payload") or {}))
                status = "completed"
                payload = {"result": result}
                error = {}
            else:
                # Prompt actions are wired to WorkflowExecutor in a later phase; keep them durable and explicit now.
                status = "blocked"
                payload = {}
                error = {"type": "WorkflowExecutorUnavailable", "message": "prompt schedules require the host workflow executor"}
        except Exception as exc:
            status = "failed"
            payload = {}
            error = {"type": exc.__class__.__name__, "message": str(exc) or repr(exc)}
        completed = utc_now_iso()
        with connect_state_db(self.data_dir) as db:
            db.execute(
                "UPDATE schedule_runs SET status = ?, result_json = ?, error_json = ?, completed_at = ? WHERE run_id = ?",
                (status, _dumps(payload), _dumps(error), completed, run_id),
            )
        return {"run_id": run_id, "schedule_id": schedule.get("schedule_id"), "status": status, **payload, "error": error}

    def _action(self, params: dict[str, Any]) -> dict[str, Any]:
        has_helper = bool(params.get("helper"))
        has_prompt = bool(params.get("prompt"))
        if has_helper == has_prompt:
            raise ValueError("Exactly one of helper or prompt is required")
        if has_helper:
            if params.get("conflict") is not None:
                raise ValueError("conflict is only valid for prompt actions")
            helper = str(params["helper"])
            if not params.get("allow_missing") and not self.helper_resolver(helper).get("found"):
                raise LookupError(f"Unknown helper: {helper}")
            payload = params.get("payload") or {}
            _jsonable(payload, "payload")
            return {"type": "helper.call", "helper": helper, "payload": payload}
        return {
            "type": "prompt",
            "prompt": str(params["prompt"]),
            "thread_id": params.get("thread_id"),
            "objective": params.get("objective"),
            "model_level": params.get("model_level"),
            "timeout_s": params.get("timeout_s"),
            "conflict": params.get("conflict") or "queue",
        }

    def _timing(self, kind: str, params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        if kind == "once":
            at = params.get("at")
            if at is None:
                raise ValueError("once schedules require at")
            dt = _parse_datetime(at, params.get("timezone"))
            return {"at": dt.isoformat()}, dt.astimezone(UTC).isoformat()
        if kind == "interval":
            seconds = _interval_seconds(params.get("every"))
            return {"every_seconds": seconds}, (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()
        cron = params.get("cron")
        if not isinstance(cron, str) or not cron.strip():
            raise ValueError("cron schedules require cron")
        return {"cron": cron.strip()}, None

    def _public_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "schedule_id": row.get("schedule_id"),
            "name": row.get("name"),
            "description": row.get("description"),
            "kind": row.get("kind"),
            "enabled": bool(row.get("enabled")),
            "action": _loads(row.get("action_json"), {}),
            "timing": _loads(row.get("timing_json"), {}),
            "timezone": row.get("timezone"),
            "next_run_at": row.get("next_run_at"),
            "misfire_policy": row.get("misfire_policy"),
            "overlap_policy": row.get("overlap_policy"),
            "metadata": _loads(row.get("metadata_json"), {}),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }


def _policy(value: Any, default: str, allowed: set[str]) -> str:
    result = str(value or default)
    if result not in allowed:
        raise ValueError(f"Unsupported policy: {result}")
    return result


def _parse_datetime(value: Any, timezone: Any = None) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError("at must be a datetime or ISO datetime string")
    if dt.tzinfo is None:
        tz = ZoneInfo(str(timezone)) if timezone and str(timezone) != "local" else datetime.now().astimezone().tzinfo
        dt = dt.replace(tzinfo=tz)
    return dt


def _interval_seconds(value: Any) -> int:
    if isinstance(value, timedelta):
        seconds = int(value.total_seconds())
    elif isinstance(value, dict):
        seconds = int(timedelta(**{str(k): v for k, v in value.items()}).total_seconds())
    else:
        raise TypeError("every must be a dict of timedelta arguments or datetime.timedelta")
    if seconds <= 0:
        raise ValueError("interval must be positive")
    return seconds


def _jsonable(value: Any, label: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-serializable") from exc


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default
