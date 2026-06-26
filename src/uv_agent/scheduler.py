from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
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
RUNNING_RUN_STATUSES = {"running"}


@dataclass
class SchedulerService:
    data_dir: Path
    config: Any
    helper_resolver: Any
    helper_caller: Any
    thread_store: Any | None = None
    poll_interval_s: float = 1.0
    workflow_starter: Any | None = None
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _stop: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _running_tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(max(1, int(getattr(self.config, "max_concurrent_jobs", 8) or 8)))

    def start(self) -> None:
        if self._task is None or self._task.done():
            self.prune_history()
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self._run_loop(), name="uv-agent-scheduler")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
        for task in list(self._running_tasks):
            task.cancel()
        await asyncio.gather(*([self._task] if self._task is not None else []), *self._running_tasks, return_exceptions=True)
        self._running_tasks.clear()

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_due_once()
                self.prune_history()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval_s)
            except asyncio.TimeoutError:
                pass

    def create(self, **params: Any) -> dict[str, Any]:
        now = utc_now_iso()
        kind = str(params.get("kind") or "").strip()
        if kind not in SCHEDULE_KINDS:
            raise ValueError("kind must be one of: once, interval, cron")
        action = self._action(params)
        timing, next_run_at = self._timing(kind, params, base=datetime.now(UTC))
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
            old_action = _loads(data["action_json"], {})
            if "helper" in changes or "prompt" in changes or "payload" in changes:
                if "helper" not in params and old_action.get("type") == "helper.call":
                    params["helper"] = old_action.get("helper")
                if "prompt" not in params and old_action.get("type") == "prompt":
                    params["prompt"] = old_action.get("prompt")
                data["action_json"] = _dumps(self._action(params))
            if any(key in changes for key in ("kind", "at", "every", "cron", "timezone")):
                timing, next_run_at = self._timing(str(params.get("kind") or data["kind"]), params, base=datetime.now(UTC))
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
        schedule = self._schedule(schedule_id)
        return await self._run_schedule(schedule, due_at=utc_now_iso(), manual=True)

    async def run_due_once(self) -> list[dict[str, Any]]:
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        with connect_state_db(self.data_dir) as db:
            rows = db.execute(
                """
                SELECT * FROM schedules
                WHERE enabled = 1 AND next_run_at IS NOT NULL AND next_run_at <= ?
                ORDER BY next_run_at ASC
                LIMIT ?
                """,
                (now, max(1, int(getattr(self.config, "max_concurrent_jobs", 8) or 8))),
            ).fetchall()
        started: list[dict[str, Any]] = []
        for row in rows:
            schedule = dict(row)
            if not self._handle_overlap(schedule):
                self._advance_schedule(schedule, base=now_dt)
                continue
            self._advance_schedule(schedule, base=now_dt)
            task = asyncio.create_task(self._run_with_semaphore(schedule, due_at=schedule.get("next_run_at")), name=f"uv-agent-schedule-{schedule['schedule_id']}")
            self._running_tasks.add(task)
            task.add_done_callback(self._running_tasks.discard)
            started.append(self._public_row(schedule))
        return started

    def prune_history(self) -> int:
        days = max(1, int(getattr(self.config, "run_history_retention_days", 7) or 7))
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with connect_state_db(self.data_dir) as db:
            return db.execute("DELETE FROM schedule_runs WHERE started_at < ?", (cutoff,)).rowcount

    async def _run_with_semaphore(self, schedule: dict[str, Any], *, due_at: str | None) -> None:
        async with self._semaphore:
            await self._run_schedule(schedule, due_at=due_at)

    def _schedule(self, schedule_id: str) -> dict[str, Any]:
        with connect_state_db(self.data_dir) as db:
            row = db.execute("SELECT * FROM schedules WHERE schedule_id = ?", (schedule_id,)).fetchone()
            if row is None:
                raise LookupError(f"Unknown schedule: {schedule_id}")
            return dict(row)

    def _handle_overlap(self, schedule: dict[str, Any]) -> bool:
        policy = str(schedule.get("overlap_policy") or "skip")
        with connect_state_db(self.data_dir) as db:
            running = db.execute(
                "SELECT run_id FROM schedule_runs WHERE schedule_id = ? AND status = 'running' ORDER BY started_at DESC",
                (schedule["schedule_id"],),
            ).fetchall()
            if not running:
                return True
            if policy == "allow":
                return True
            if policy == "replace":
                db.executemany(
                    "UPDATE schedule_runs SET status = 'cancelled', completed_at = ?, error_json = ? WHERE run_id = ?",
                    [(utc_now_iso(), _dumps({"type": "Replaced", "message": "Replaced by a newer schedule run."}), row["run_id"]) for row in running],
                )
                return True
        return False

    def _advance_schedule(self, schedule: dict[str, Any], *, base: datetime) -> None:
        kind = str(schedule.get("kind"))
        timing = _loads(schedule.get("timing_json"), {})
        next_run_at: str | None
        enabled = int(schedule.get("enabled") or 0)
        if kind == "once":
            next_run_at = None
            enabled = 0
        elif kind == "interval":
            seconds = int(timing.get("every_seconds") or 0)
            next_run_at = (base + timedelta(seconds=max(1, seconds))).isoformat()
        elif kind == "cron":
            next_run_at = _next_cron(timing.get("cron"), base, schedule.get("timezone"))
        else:
            next_run_at = None
        with connect_state_db(self.data_dir) as db:
            db.execute(
                "UPDATE schedules SET next_run_at = ?, enabled = ?, updated_at = ? WHERE schedule_id = ?",
                (next_run_at, enabled, utc_now_iso(), schedule["schedule_id"]),
            )

    async def _run_schedule(self, schedule: dict[str, Any], *, due_at: str | None, manual: bool = False) -> dict[str, Any]:
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
                workflow_id = None
                error = {}
            else:
                workflow_id = self._start_prompt_workflow(schedule, action)
                status = "completed"
                payload = {"workflow_id": workflow_id}
                error = {}
        except Exception as exc:
            status = "failed"
            payload = {}
            workflow_id = None
            error = {"type": exc.__class__.__name__, "message": str(exc) or repr(exc)}
        completed = utc_now_iso()
        with connect_state_db(self.data_dir) as db:
            db.execute(
                "UPDATE schedule_runs SET status = ?, result_json = ?, error_json = ?, workflow_id = ?, completed_at = ? WHERE run_id = ?",
                (status, _dumps(payload), _dumps(error), workflow_id, completed, run_id),
            )
        return {"run_id": run_id, "schedule_id": schedule.get("schedule_id"), "status": status, **payload, "error": error}

    def _start_prompt_workflow(self, schedule: dict[str, Any], action: dict[str, Any]) -> str:
        import uv_agent_runtime.workflow as workflow

        if self.workflow_starter is not None:
            return str(self.workflow_starter(schedule, action))
        thread_id = action.get("thread_id") or self._schedule_thread(schedule)
        wf = workflow.start(
            str(action.get("objective") or schedule.get("name") or f"Scheduled prompt {schedule['schedule_id']}"),
            default_model_level=action.get("model_level"),
            state_dir=self.data_dir,
        )
        # Parent linkage is persisted on the workflow row; runtime.start cannot
        # receive it directly, so update the row in the same host transaction.
        with connect_state_db(self.data_dir) as db:
            db.execute("UPDATE workflows SET parent_thread_id = ? WHERE workflow_id = ?", (thread_id, wf.workflow_id))
        wf.agent(str(action.get("prompt") or ""), model_level=action.get("model_level"), timeout_s=action.get("timeout_s"))
        return wf.workflow_id

    def _schedule_thread(self, schedule: dict[str, Any]) -> str | None:
        if self.thread_store is None:
            return None
        return self.thread_store.get_or_create_external_thread(
            owner_plugin="scheduler",
            source="scheduler",
            external_id=str(schedule["schedule_id"]),
            title=str(schedule.get("name") or f"Schedule {schedule['schedule_id']}"),
            metadata={"schedule_id": schedule["schedule_id"]},
        )

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

    def _timing(self, kind: str, params: dict[str, Any], *, base: datetime) -> tuple[dict[str, Any], str | None]:
        if kind == "once":
            at = params.get("at")
            if at is None:
                raise ValueError("once schedules require at")
            dt = _parse_datetime(at, params.get("timezone"))
            return {"at": dt.isoformat()}, dt.astimezone(UTC).isoformat()
        if kind == "interval":
            seconds = _interval_seconds(params.get("every"))
            return {"every_seconds": seconds}, (base + timedelta(seconds=seconds)).isoformat()
        cron = params.get("cron")
        if not isinstance(cron, str) or not cron.strip():
            raise ValueError("cron schedules require cron")
        return {"cron": cron.strip()}, _next_cron(cron.strip(), base, params.get("timezone"))

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


def _next_cron(expr: Any, base: datetime, timezone: Any = None) -> str | None:
    parts = str(expr or "").split()
    if len(parts) != 5:
        raise ValueError("cron must have five fields: minute hour day month weekday")
    tz = ZoneInfo(str(timezone)) if timezone and str(timezone) != "local" else UTC
    current = base.astimezone(tz).replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(366 * 24 * 60):
        if _cron_matches(parts, current):
            return current.astimezone(UTC).isoformat()
        current += timedelta(minutes=1)
    raise ValueError("cron expression has no next run within one year")


def _cron_matches(parts: list[str], dt: datetime) -> bool:
    values = [dt.minute, dt.hour, dt.day, dt.month, (dt.weekday() + 1) % 7]
    limits = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    return all(_cron_field_matches(part, value, low, high) for part, value, (low, high) in zip(parts, values, limits, strict=True))


def _cron_field_matches(part: str, value: int, low: int, high: int) -> bool:
    for item in part.split(','):
        item = item.strip()
        if item == '*':
            return True
        if item.startswith('*/'):
            step = int(item[2:])
            if step > 0 and (value - low) % step == 0:
                return True
        elif item.isdigit() and low <= int(item) <= high and value == int(item):
            return True
    return False


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
