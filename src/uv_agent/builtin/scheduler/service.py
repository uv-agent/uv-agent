from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any
from zoneinfo import ZoneInfo

from uv_agent.ids import new_id
from uv_agent.state_db import connect_state_db
from uv_agent.time import utc_now_iso

MISFIRE_POLICIES = {"skip", "run_once", "catch_up"}
OVERLAP_POLICIES = {"skip", "allow", "queue", "replace"}
SCHEDULE_KINDS = {"once", "interval", "cron"}
RUNNING_RUN_STATUSES = {"running"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchedulerConfig:
    max_concurrent_jobs: int = 8
    run_history_retention_days: int = 7
    default_misfire_policy: str = "skip"
    default_overlap_policy: str = "skip"


def scheduler_config_from_plugin_config(config: Mapping[str, Any] | None) -> SchedulerConfig:
    data = dict(config or {})
    return SchedulerConfig(
        max_concurrent_jobs=max(1, int(data.get("max_concurrent_jobs", 8) or 8)),
        run_history_retention_days=max(1, int(data.get("run_history_retention_days", 7) or 7)),
        default_misfire_policy=str(data.get("default_misfire_policy", "skip") or "skip"),
        default_overlap_policy=str(data.get("default_overlap_policy", "skip") or "skip"),
    )


@dataclass(frozen=True)
class SchedulerActionContext:
    """Context passed to plugin action handlers for scheduled executions.

    Scheduler owns timing, persistence and overlap policy; plugins own the action
    semantics.  The small context keeps prompt/workflow behavior out of scheduler
    while still letting actions create a stable external thread for their runs.
    """

    data_dir: Path
    schedule: dict[str, Any]
    run_id: str
    due_at: str | None
    manual: bool = False
    threads: Any | None = None
    submitter: Callable[..., Any] | None = None

    def schedule_thread(self) -> str | None:
        if self.threads is None:
            return None
        schedule_id = str(self.schedule.get("schedule_id") or "")
        if not schedule_id:
            return None
        return self.threads.get_or_create_external_thread(
            source="scheduler",
            external_id=schedule_id,
            title=str(self.schedule.get("name") or f"Schedule {schedule_id}"),
            metadata={"schedule_id": schedule_id},
        )

    async def submit_turn(
        self,
        *,
        text: str,
        thread_id: str | None = None,
        level: str | None = None,
        image_paths: list[Any] | None = None,
        conflict: str = "queue",
    ) -> Any:
        if self.submitter is None:
            raise RuntimeError("Scheduled actions cannot submit turns in this host context")
        kwargs = {"text": text, "thread_id": thread_id, "level": level, "conflict": conflict}
        if image_paths is not None:
            kwargs["image_paths"] = image_paths
        result = self.submitter(**kwargs)
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            result = await result
        return result


@dataclass
class SchedulerService:
    data_dir: Path
    config: Any
    action_resolver: Callable[[str], Any]
    action_caller: Callable[..., Any]
    threads: Any | None = None
    submitter: Callable[..., Any] | None = None
    poll_interval_s: float = 1.0
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _stop: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _running_tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(max(1, int(getattr(self.config, "max_concurrent_jobs", 8) or 8)))
        with self._connect():
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = connect_state_db(self.data_dir)
        try:
            _ensure_scheduler_schema(connection)
        except Exception:
            connection.close()
            raise
        return connection

    def start(self) -> None:
        if self._task is None or self._task.done():
            self.prune_history()
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self._run_loop(), name="uv-agent-scheduler")
            logger.info(
                "Scheduler started data_dir=%s max_concurrent_jobs=%s poll_interval_s=%s",
                self.data_dir,
                getattr(self.config, "max_concurrent_jobs", 8),
                self.poll_interval_s,
            )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
        for task in list(self._running_tasks):
            task.cancel()
        await asyncio.gather(*([self._task] if self._task is not None else []), *self._running_tasks, return_exceptions=True)
        self._running_tasks.clear()
        logger.info("Scheduler stopped")

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_due_once()
                self.prune_history()
            except Exception:
                logger.exception("Scheduler loop failed")
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
        with self._connect() as db:
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
        logger.info("Schedule created schedule_id=%s kind=%s enabled=%s next_run_at=%s", schedule_id, kind, bool(row["enabled"]), next_run_at)
        return self._public_row(row)

    def update(self, schedule_id: str, **changes: Any) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM schedules WHERE schedule_id = ?", (schedule_id,)).fetchone()
            if row is None:
                raise LookupError(f"Unknown schedule: {schedule_id}")
            data = dict(row)
            params = {**self._public_row(data), **changes}
            old_action = _loads(data["action_json"], {})
            if any(key in changes for key in ("action", "action_id", "payload")):
                if "action" not in changes and "action_id" not in changes and old_action.get("type") == "action.call":
                    params["action_id"] = old_action.get("action_id")
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
            logger.info("Schedule updated schedule_id=%s enabled=%s next_run_at=%s", schedule_id, bool(data["enabled"]), data.get("next_run_at"))
            return self._public_row(data)

    def list(self, *, enabled: bool | None = None, limit: int = 100) -> list[dict[str, Any]]:
        clauses = []
        args: list[Any] = []
        if enabled is not None:
            clauses.append("enabled = ?")
            args.append(1 if enabled else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as db:
            rows = db.execute(f"SELECT * FROM schedules {where} ORDER BY updated_at DESC LIMIT ?", (*args, int(limit))).fetchall()
        return [self._public_row(dict(row)) for row in rows]

    def delete(self, schedule_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM schedules WHERE schedule_id = ?", (schedule_id,)).fetchone()
            if row is None:
                raise LookupError(f"Unknown schedule: {schedule_id}")
            db.execute("DELETE FROM schedules WHERE schedule_id = ?", (schedule_id,))
        logger.info("Schedule deleted schedule_id=%s", schedule_id)
        return {"deleted": True, "schedule_id": schedule_id}

    async def run_now(self, schedule_id: str) -> dict[str, Any]:
        schedule = self._schedule(schedule_id)
        return await self._run_schedule(schedule, due_at=utc_now_iso(), manual=True)

    async def run_due_once(self) -> list[dict[str, Any]]:
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        with self._connect() as db:
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
                logger.info("Schedule run skipped for overlap schedule_id=%s policy=%s", schedule.get("schedule_id"), schedule.get("overlap_policy"))
                self._advance_schedule(schedule, base=now_dt)
                continue
            self._advance_schedule(schedule, base=now_dt)
            task = asyncio.create_task(self._run_with_semaphore(schedule, due_at=schedule.get("next_run_at")), name=f"uv-agent-schedule-{schedule['schedule_id']}")
            self._running_tasks.add(task)
            task.add_done_callback(self._running_tasks.discard)
            started.append(self._public_row(schedule))
        if started:
            logger.info("Scheduler started due jobs count=%d", len(started))
        return started

    def prune_history(self) -> int:
        days = max(1, int(getattr(self.config, "run_history_retention_days", 7) or 7))
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with self._connect() as db:
            deleted = db.execute("DELETE FROM schedule_runs WHERE started_at < ?", (cutoff,)).rowcount
        if deleted:
            logger.debug("Scheduler pruned history deleted=%d cutoff=%s", deleted, cutoff)
        return deleted

    async def _run_with_semaphore(self, schedule: dict[str, Any], *, due_at: str | None) -> None:
        async with self._semaphore:
            await self._run_schedule(schedule, due_at=due_at)

    def _schedule(self, schedule_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM schedules WHERE schedule_id = ?", (schedule_id,)).fetchone()
            if row is None:
                raise LookupError(f"Unknown schedule: {schedule_id}")
            return dict(row)

    def _handle_overlap(self, schedule: dict[str, Any]) -> bool:
        policy = str(schedule.get("overlap_policy") or "skip")
        with self._connect() as db:
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
        with self._connect() as db:
            db.execute(
                "UPDATE schedules SET next_run_at = ?, enabled = ?, updated_at = ? WHERE schedule_id = ?",
                (next_run_at, enabled, utc_now_iso(), schedule["schedule_id"]),
            )

    async def _run_schedule(self, schedule: dict[str, Any], *, due_at: str | None, manual: bool = False) -> dict[str, Any]:
        run_id = new_id("sr")
        action = _loads(schedule["action_json"], {})
        snapshot = self._public_row(schedule)
        started = utc_now_iso()
        logger.info(
            "Schedule run started schedule_id=%s run_id=%s manual=%s due_at=%s",
            schedule.get("schedule_id"),
            run_id,
            manual,
            due_at,
        )
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO schedule_runs(run_id, schedule_id, status, action_json, schedule_snapshot_json, due_at, started_at)
                VALUES (?, ?, 'running', ?, ?, ?, ?)
                """,
                (run_id, schedule.get("schedule_id"), _dumps(action), _dumps(snapshot), due_at, started),
            )
        try:
            result = await self._call_action(schedule, action, run_id=run_id, due_at=due_at, manual=manual)
            status = "completed"
            payload = {"result": result}
            workflow_id = str(result.get("workflow_id")) if isinstance(result, dict) and result.get("workflow_id") else None
            error = {}
        except Exception as exc:
            status = "failed"
            payload = {}
            workflow_id = None
            error = {"type": exc.__class__.__name__, "message": str(exc) or repr(exc)}
            logger.warning(
                "Schedule run failed schedule_id=%s run_id=%s error_type=%s",
                schedule.get("schedule_id"),
                run_id,
                exc.__class__.__name__,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
        completed = utc_now_iso()
        with self._connect() as db:
            db.execute(
                "UPDATE schedule_runs SET status = ?, result_json = ?, error_json = ?, workflow_id = ?, completed_at = ? WHERE run_id = ?",
                (status, _dumps(payload), _dumps(error), workflow_id, completed, run_id),
            )
        response = {"run_id": run_id, "schedule_id": schedule.get("schedule_id"), "status": status, **payload, "error": error}
        if workflow_id:
            response["workflow_id"] = workflow_id
        logger.info(
            "Schedule run completed schedule_id=%s run_id=%s status=%s workflow_id=%s",
            schedule.get("schedule_id"),
            run_id,
            status,
            workflow_id,
        )
        return response

    async def _call_action(
        self,
        schedule: dict[str, Any],
        action: dict[str, Any],
        *,
        run_id: str,
        due_at: str | None,
        manual: bool,
    ) -> Any:
        if action.get("type") != "action.call":
            raise ValueError("Schedule action must be an action.call record")
        action_id = str(action.get("action_id") or "")
        if not action_id:
            raise ValueError("Schedule action is missing action_id")
        context = SchedulerActionContext(
            data_dir=self.data_dir,
            schedule=self._public_row(schedule),
            run_id=run_id,
            due_at=due_at,
            manual=manual,
            threads=self.threads,
            submitter=self.submitter,
        )
        result = self.action_caller(action_id, dict(action.get("payload") or {}), context=context)
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            result = await result
        _jsonable(result, "action result")
        return result

    def _action(self, params: dict[str, Any]) -> dict[str, Any]:
        if any(params.get(key) is not None for key in ("helper", "prompt")):
            raise ValueError("Scheduler actions use action_id and payload; helper/prompt fields are not supported")
        action_id = params.get("action_id", params.get("action"))
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("action_id is required")
        action_id = action_id.strip()
        if not params.get("allow_missing") and not _action_exists(self.action_resolver(action_id)):
            raise LookupError(f"Unknown action: {action_id}")
        payload = params.get("payload") or {}
        if not isinstance(payload, dict):
            raise TypeError("payload must be a JSON object")
        _jsonable(payload, "payload")
        return {"type": "action.call", "action_id": action_id, "payload": payload}

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


def _action_exists(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("found", value))
    return value is not None and value is not False


def _ensure_scheduler_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schedules (
          schedule_id TEXT PRIMARY KEY,
          name TEXT,
          description TEXT,
          kind TEXT NOT NULL,
          enabled INTEGER NOT NULL DEFAULT 1,
          action_json TEXT NOT NULL,
          timing_json TEXT NOT NULL,
          timezone TEXT,
          next_run_at TEXT,
          misfire_policy TEXT NOT NULL DEFAULT 'skip',
          overlap_policy TEXT NOT NULL DEFAULT 'skip',
          owner_type TEXT NOT NULL DEFAULT 'agent',
          owner_name TEXT,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_schedules_enabled_next_run
          ON schedules(enabled, next_run_at);

        CREATE TABLE IF NOT EXISTS schedule_runs (
          run_id TEXT PRIMARY KEY,
          schedule_id TEXT,
          status TEXT NOT NULL,
          action_json TEXT NOT NULL,
          schedule_snapshot_json TEXT NOT NULL,
          result_json TEXT NOT NULL DEFAULT '{}',
          error_json TEXT NOT NULL DEFAULT '{}',
          workflow_id TEXT,
          due_at TEXT,
          started_at TEXT NOT NULL,
          completed_at TEXT,
          FOREIGN KEY(schedule_id) REFERENCES schedules(schedule_id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_schedule_runs_schedule_started
          ON schedule_runs(schedule_id, started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_schedule_runs_started
          ON schedule_runs(started_at DESC);
        """
    )
    connection.commit()


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
