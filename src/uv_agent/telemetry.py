from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from uv_agent.billing import decimal_or_none, decimal_to_string, normalize_currency
from uv_agent.state_db import SQLITE_BUSY_TIMEOUT_MS, SQLITE_TIMEOUT_SECONDS
from uv_agent.time import utc_now_iso

DB_FILENAME = "telemetry.sqlite3"
SCHEMA_VERSION = 1

logger = logging.getLogger("uv_agent.telemetry")

_MODEL_CALL_INSERT = """
INSERT INTO model_calls(
    thread_id, turn_id, level, source, model_name, remote_model,
    input_tokens, cached_input_tokens, output_tokens, reasoning_tokens,
    billing_amount, billing_currency, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_RUN_STAT_INSERT = """
INSERT OR REPLACE INTO run_stats(
    run_id, thread_id, turn_id, started_at, completed_at, duration_ms,
    returncode, timed_out, interrupted, truncated,
    helper_count, helper_duration_ms, helper_errors, top_helpers_json,
    stdout_bytes, stderr_bytes, event_count
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class _SqliteBatcher:
    """Buffers model-call and run-stat rows and flushes them in bulk.

    Flushing happens when the combined pending count reaches ``max_size``,
    when ``max_age_ms`` has elapsed since the last flush, or when ``flush()``
    is called explicitly (e.g. on turn completion or shutdown).
    """

    def __init__(
        self,
        connect: Callable[[], sqlite3.Connection],
        *,
        max_size: int = 32,
        max_age_ms: float = 100.0,
    ) -> None:
        self._connect = connect
        self._max_size = max(1, max_size)
        self._max_age_seconds = max(0.0, max_age_ms / 1000.0)
        self._lock = threading.RLock()
        self._model_calls: list[tuple[Any, ...]] = []
        self._run_stats: list[tuple[Any, ...]] = []
        self._last_flush = 0.0

    def add_model_call(self, params: tuple[Any, ...]) -> None:
        with self._lock:
            self._model_calls.append(params)
            self._maybe_flush_locked()

    def add_run_stat(self, params: tuple[Any, ...]) -> None:
        with self._lock:
            self._run_stats.append(params)
            self._maybe_flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        self.flush()

    def _maybe_flush_locked(self) -> None:
        total = len(self._model_calls) + len(self._run_stats)
        if total == 0:
            return
        now = time.monotonic()
        if self._last_flush == 0:
            self._last_flush = now
            if self._max_age_seconds == 0:
                self._flush_locked()
            return
        if total >= self._max_size or (now - self._last_flush) >= self._max_age_seconds:
            self._flush_locked()

    def _flush_locked(self) -> None:
        model_calls = self._model_calls
        run_stats = self._run_stats
        self._model_calls = []
        self._run_stats = []
        self._last_flush = time.monotonic()
        if not model_calls and not run_stats:
            return
        try:
            logger.debug("Telemetry batch flush model_calls=%d run_stats=%d", len(model_calls), len(run_stats))
            with self._connect() as db:
                if model_calls:
                    db.executemany(_MODEL_CALL_INSERT, model_calls)
                if run_stats:
                    db.executemany(_RUN_STAT_INSERT, run_stats)
            logger.debug("Telemetry batch flush completed model_calls=%d run_stats=%d", len(model_calls), len(run_stats))
        except Exception:
            # Telemetry is best-effort; never let a flush failure propagate.
            logger.exception("Telemetry batch flush failed")


class TelemetryStore:
    """SQLite-backed store for aggregated operational telemetry.

    Telemetry is kept in a separate database under ``<data_dir>/log`` so it can
    be queried, pruned, or archived without affecting the main conversation
    state.  The store consumes host events and maintains lightweight in-memory
    per-turn aggregates that are flushed when a turn ends.

    Model-call and run-stat inserts are buffered and flushed in bulk to keep
    per-event overhead minimal.  Turn-stat writes are flushed immediately when
    a turn ends so that per-turn summaries are durable right away.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        batch_max_size: int = 32,
        batch_max_age_ms: float = 100.0,
    ) -> None:
        self.data_dir = data_dir.resolve()
        self.db_path = self.data_dir / "log" / DB_FILENAME
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._turn_aggregates: dict[str, dict[str, Any]] = {}
        self._batcher = _SqliteBatcher(
            self._connect,
            max_size=batch_max_size,
            max_age_ms=batch_max_age_ms,
        )
        self._ensure_schema()
        logger.debug("Telemetry store initialized db_path=%s", self.db_path)

    def db_path_for_data_dir(self) -> Path:
        return self.db_path

    def close(self) -> None:
        """Flush any pending telemetry and release resources."""

        self._batcher.close()

    def flush(self) -> None:
        """Flush pending model-call and run-stat batches immediately."""

        self._batcher.flush()

    def _ensure_schema(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_calls (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  thread_id TEXT,
                  turn_id TEXT,
                  level TEXT,
                  source TEXT NOT NULL,
                  model_name TEXT,
                  remote_model TEXT,
                  input_tokens INTEGER NOT NULL DEFAULT 0,
                  cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                  output_tokens INTEGER NOT NULL DEFAULT 0,
                  reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                  billing_amount TEXT,
                  billing_currency TEXT,
                  created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_model_calls_thread_turn
                  ON model_calls(thread_id, turn_id);

                CREATE INDEX IF NOT EXISTS idx_model_calls_created_at
                  ON model_calls(created_at);

                CREATE TABLE IF NOT EXISTS run_stats (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT UNIQUE NOT NULL,
                  thread_id TEXT,
                  turn_id TEXT,
                  started_at TEXT,
                  completed_at TEXT,
                  duration_ms REAL,
                  returncode INTEGER,
                  timed_out INTEGER NOT NULL DEFAULT 0,
                  interrupted INTEGER NOT NULL DEFAULT 0,
                  truncated INTEGER NOT NULL DEFAULT 0,
                  helper_count INTEGER NOT NULL DEFAULT 0,
                  helper_duration_ms REAL,
                  helper_errors INTEGER NOT NULL DEFAULT 0,
                  top_helpers_json TEXT NOT NULL DEFAULT '{}',
                  stdout_bytes INTEGER NOT NULL DEFAULT 0,
                  stderr_bytes INTEGER NOT NULL DEFAULT 0,
                  event_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_run_stats_thread_turn
                  ON run_stats(thread_id, turn_id);

                CREATE INDEX IF NOT EXISTS idx_run_stats_completed_at
                  ON run_stats(completed_at);

                CREATE TABLE IF NOT EXISTS turn_stats (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  turn_id TEXT NOT NULL,
                  thread_id TEXT,
                  started_at TEXT,
                  completed_at TEXT,
                  duration_ms REAL,
                  status TEXT NOT NULL DEFAULT 'unknown',
                  level TEXT,
                  model_name TEXT,
                  model_calls INTEGER NOT NULL DEFAULT 0,
                  input_tokens INTEGER NOT NULL DEFAULT 0,
                  output_tokens INTEGER NOT NULL DEFAULT 0,
                  total_tokens INTEGER NOT NULL DEFAULT 0,
                  reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                  billing_amount TEXT,
                  billing_currency TEXT,
                  run_python_calls INTEGER NOT NULL DEFAULT 0,
                  run_python_total_duration_ms REAL,
                  run_python_errors INTEGER NOT NULL DEFAULT 0,
                  helper_calls_total INTEGER NOT NULL DEFAULT 0,
                  helper_unique_count INTEGER NOT NULL DEFAULT 0,
                  top_helpers_json TEXT NOT NULL DEFAULT '{}',
                  compactions INTEGER NOT NULL DEFAULT 0
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_turn_stats_turn_id
                  ON turn_stats(turn_id);

                CREATE INDEX IF NOT EXISTS idx_turn_stats_thread_completed
                  ON turn_stats(thread_id, completed_at);
                """
            )
            db.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(
            self.db_path,
            timeout=SQLITE_TIMEOUT_SECONDS,
            check_same_thread=False,
        )
        db.row_factory = sqlite3.Row
        db.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        return db

    def record_model_call(self, event: dict[str, Any]) -> None:
        """Record one billed model call from an agent.model_call_billed event."""

        billing = event.get("billing") or {}
        usage = event.get("usage") or {}
        amount = decimal_or_none(billing.get("amount"))
        currency = normalize_currency(str(billing.get("currency") or "USD"))
        self._batcher.add_model_call(
            (
                event.get("thread_id"),
                event.get("turn_id"),
                event.get("level"),
                billing.get("source") or event.get("source"),
                billing.get("model"),
                billing.get("remote_model"),
                int(billing.get("input_tokens") or 0),
                int(billing.get("cached_input_tokens") or 0),
                int(billing.get("output_tokens") or 0),
                int(billing.get("reasoning_tokens") or 0),
                decimal_to_string(amount) if amount is not None else None,
                currency,
                billing.get("created_at") or utc_now_iso(),
            )
        )
        logger.debug(
            "Telemetry model call recorded thread_id=%s turn_id=%s level=%s source=%s model=%s",
            event.get("thread_id"),
            event.get("turn_id"),
            event.get("level"),
            billing.get("source") or event.get("source"),
            billing.get("model"),
        )
        self._update_turn_aggregate(
            event.get("thread_id"),
            event.get("turn_id"),
            model_call=billing,
            usage=usage,
        )

    def record_run_completed(self, event: dict[str, Any]) -> None:
        """Record run-level statistics from a runner.run_completed event."""

        run_id = event.get("run_id")
        helper_calls = event.get("helper_calls") or []
        summary = _summarize_helper_calls(helper_calls)
        started_at = event.get("started_at")
        completed_at = event.get("completed_at")
        duration_ms = _duration_ms(started_at, completed_at)

        self._batcher.add_run_stat(
            (
                run_id,
                event.get("thread_id"),
                event.get("turn_id"),
                started_at,
                completed_at,
                duration_ms,
                event.get("returncode"),
                int(bool(event.get("timed_out"))),
                int(bool(event.get("interrupted"))),
                int(bool(event.get("truncated"))),
                summary["count"],
                summary["total_duration_ms"],
                summary["errors"],
                json.dumps(summary["top_helpers"], sort_keys=True, separators=(",", ":")),
                int(event.get("stdout_bytes") or 0),
                int(event.get("stderr_bytes") or 0),
                int(event.get("event_count") or 0),
            )
        )
        logger.debug(
            "Telemetry run stat recorded run_id=%s thread_id=%s turn_id=%s returncode=%s duration_ms=%s",
            run_id,
            event.get("thread_id"),
            event.get("turn_id"),
            event.get("returncode"),
            duration_ms,
        )
        self._update_turn_aggregate(
            event.get("thread_id"),
            event.get("turn_id"),
            run_summary={
                "duration_ms": duration_ms,
                "returncode": event.get("returncode"),
                "timed_out": bool(event.get("timed_out")),
                "interrupted": bool(event.get("interrupted")),
                "helper_calls": helper_calls,
            },
        )

    def record_thread_event(self, event: dict[str, Any]) -> None:
        """Update turn aggregates from a thread.event_stored event."""

        wrapper = event.get("event") or {}
        event_type = str(wrapper.get("type") or "")
        thread_id = event.get("thread_id") or wrapper.get("thread_id")
        turn_id = wrapper.get("turn_id")

        if event_type == "turn.started":
            self._ensure_turn_aggregate(thread_id, turn_id, wrapper.get("created_at"))
            return

        if event_type in {"turn.completed", "turn.error", "turn.interrupted"}:
            # Flush pending model/run rows before writing the turn summary so
            # that consumers querying after a turn ends see complete data.
            self._batcher.flush()
            self._flush_turn_aggregate(
                thread_id,
                turn_id,
                status=event_type.replace("turn.", ""),
                completed_at=wrapper.get("created_at"),
            )
            return

        if event_type == "item.compaction":
            self._update_turn_aggregate(thread_id, turn_id, compaction=True)
            return

    def on_event(self, event: dict[str, Any]) -> None:
        """Dispatch a host event to the appropriate telemetry recorder."""

        event_type = str(event.get("type") or "")
        if event_type == "agent.model_call_billed":
            self.record_model_call(event)
        elif event_type == "runner.run_completed":
            self.record_run_completed(event)
        elif event_type == "thread.event_stored":
            self.record_thread_event(event)

    def _ensure_turn_aggregate(
        self,
        thread_id: str | None,
        turn_id: str | None,
        started_at: str | None,
    ) -> None:
        if not turn_id:
            return
        with self._lock:
            agg = self._turn_aggregates.get(turn_id)
            if agg is None:
                self._turn_aggregates[turn_id] = {
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "started_at": started_at,
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "reasoning_tokens": 0,
                    "billing_amounts": defaultdict(lambda: Decimal("0")),
                    "run_python_calls": 0,
                    "run_python_total_duration_ms": 0.0,
                    "run_python_errors": 0,
                    "helper_calls_total": 0,
                    "helper_unique_counts": defaultdict(int),
                    "compactions": 0,
                }

    def _update_turn_aggregate(
        self,
        thread_id: str | None,
        turn_id: str | None,
        *,
        model_call: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        run_summary: dict[str, Any] | None = None,
        compaction: bool = False,
    ) -> None:
        if not turn_id:
            return
        with self._lock:
            agg = self._turn_aggregates.get(turn_id)
            if agg is None:
                self._ensure_turn_aggregate(thread_id, turn_id, None)
                agg = self._turn_aggregates[turn_id]

            if model_call:
                agg["model_calls"] += 1
                agg["input_tokens"] += int(model_call.get("input_tokens") or 0)
                agg["output_tokens"] += int(model_call.get("output_tokens") or 0)
                agg["total_tokens"] += int(model_call.get("input_tokens") or 0) + int(
                    model_call.get("cached_input_tokens") or 0
                ) + int(model_call.get("output_tokens") or 0)
                agg["reasoning_tokens"] += int(model_call.get("reasoning_tokens") or 0)
                currency = normalize_currency(str(model_call.get("currency") or "USD"))
                amount = decimal_or_none(model_call.get("amount"))
                if amount is not None:
                    agg["billing_amounts"][currency] += amount

            if run_summary:
                agg["run_python_calls"] += 1
                duration = run_summary.get("duration_ms")
                if isinstance(duration, (int, float)):
                    agg["run_python_total_duration_ms"] += float(duration)
                if run_summary.get("returncode") not in {0, None} or run_summary.get("timed_out") or run_summary.get("interrupted"):
                    agg["run_python_errors"] += 1
                for call in run_summary.get("helper_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    count = int(call.get("count") or 1)
                    agg["helper_calls_total"] += count
                    name = str(call.get("name") or "")
                    if name:
                        agg["helper_unique_counts"][name] += count

            if compaction:
                agg["compactions"] += 1

    def _flush_turn_aggregate(
        self,
        thread_id: str | None,
        turn_id: str | None,
        *,
        status: str,
        completed_at: str | None,
    ) -> None:
        if not turn_id:
            return
        with self._lock:
            agg = self._turn_aggregates.pop(turn_id, None)
        if agg is None:
            return

        # Pick the primary currency with the largest total amount.
        currency, amount = _primary_currency_amount(dict(agg["billing_amounts"]))
        top_helpers = dict(
            sorted(
                agg["helper_unique_counts"].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        )

        started_at = agg.get("started_at")
        duration_ms = _duration_ms(started_at, completed_at)

        with self._connect() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO turn_stats(
                    turn_id, thread_id, started_at, completed_at, duration_ms, status,
                    model_calls, input_tokens, output_tokens, total_tokens, reasoning_tokens,
                    billing_amount, billing_currency,
                    run_python_calls, run_python_total_duration_ms, run_python_errors,
                    helper_calls_total, helper_unique_count, top_helpers_json, compactions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    thread_id or agg.get("thread_id"),
                    started_at,
                    completed_at,
                    duration_ms,
                    status,
                    agg["model_calls"],
                    agg["input_tokens"],
                    agg["output_tokens"],
                    agg["total_tokens"],
                    agg["reasoning_tokens"],
                    decimal_to_string(amount) if amount is not None else None,
                    currency,
                    agg["run_python_calls"],
                    round(agg["run_python_total_duration_ms"], 3),
                    agg["run_python_errors"],
                    agg["helper_calls_total"],
                    len(agg["helper_unique_counts"]),
                    json.dumps(top_helpers, sort_keys=True, separators=(",", ":")),
                    agg["compactions"],
                ),
            )
        logger.debug(
            "Telemetry turn aggregate flushed thread_id=%s turn_id=%s status=%s model_calls=%d run_python_calls=%d compactions=%d",
            thread_id or agg.get("thread_id"),
            turn_id,
            status,
            agg["model_calls"],
            agg["run_python_calls"],
            agg["compactions"],
        )

    def query_turn_stats(self, turn_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM turn_stats WHERE turn_id = ?", (turn_id,)).fetchone()
        if row is None:
            return None
        return dict(row)


def _summarize_helper_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    total_duration = 0.0
    errors = 0
    total_count = 0
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "")
        count = int(call.get("count") or 1)
        total_count += count
        if name:
            counts[name] += count
        duration = call.get("total_duration_ms") or call.get("duration_ms")
        if isinstance(duration, (int, float)):
            total_duration += float(duration)
        outcomes = call.get("outcomes")
        if isinstance(outcomes, dict):
            errors += int(outcomes.get("error") or 0)
        elif str(call.get("outcome") or "") == "error":
            errors += count
    top = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20])
    return {
        "count": total_count,
        "total_duration_ms": round(total_duration, 3),
        "errors": errors,
        "top_helpers": top,
    }


def _primary_currency_amount(amounts: dict[str, Decimal]) -> tuple[str, Decimal | None]:
    if not amounts:
        return ("USD", None)
    currency = max(amounts.items(), key=lambda item: item[1])[0]
    return (currency, amounts[currency])


def _duration_ms(started_at: str | None, completed_at: str | None) -> float | None:
    if not started_at or not completed_at:
        return None
    try:
        from datetime import datetime

        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(completed_at)
        return max(0.0, (end - start).total_seconds() * 1000)
    except Exception:
        return None
