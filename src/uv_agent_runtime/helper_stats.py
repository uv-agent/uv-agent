from __future__ import annotations

import atexit
import functools
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

DB_FILENAME = "helper-stats.sqlite3"
SCHEMA_VERSION = 1
SQLITE_TIMEOUT_SECONDS = 5.0
T = TypeVar("T", bound=Callable[..., Any])

_thread_local = threading.local()
_process_lock = threading.RLock()
_process_connection: sqlite3.Connection | None = None
_process_connection_path: Path | None = None
_exit_registered = False


def helper_stats_db_path() -> Path:
    """Return the SQLite path used for runtime-helper usage statistics.

    The database intentionally lives under ``<project-state>/log`` instead of the
    main conversation database. Helper statistics are operational telemetry and
    can be inspected or deleted independently without risking session history.
    """

    override = os.environ.get("UV_AGENT_RUNTIME_HELPER_STATS_DB")
    if override:
        return Path(override).expanduser().resolve()
    state_dir = os.environ.get("UV_AGENT_RUNTIME_STATE_DIR")
    if not state_dir:
        raise RuntimeError("UV_AGENT_RUNTIME_STATE_DIR is not set; helper statistics are unavailable")
    return Path(state_dir).expanduser().resolve() / "log" / DB_FILENAME


def log_helper_call(
    name: str,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    called_at_unix: float | None = None,
    duration_ms: float | None = None,
    outcome: str = "ok",
    error_type: str | None = None,
) -> None:
    """Best-effort append of one runtime-helper call to the local stats DB.

    Logging must never change helper behavior. It also avoids storing argument
    values so commands, environment mappings, prompts, and paths that may contain
    sensitive material are not copied into telemetry.
    """

    if not name or getattr(_thread_local, "logging", False):
        return
    _thread_local.logging = True
    try:
        _record_local(
            _call_payload(
                name,
                args,
                kwargs or {},
                called_at_unix=called_at_unix,
                duration_ms=duration_ms,
                outcome=outcome,
                error_type=error_type,
            )
        )
    except Exception:
        return
    finally:
        _thread_local.logging = False


def tracked_helper(func: T, *, name: str | None = None) -> T:
    """Decorate a public runtime helper so calls are included in usage stats."""

    helper_name = name or getattr(func, "__name__", "helper")

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started_unix = time.time()
        started_perf = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            log_helper_call(
                helper_name,
                args,
                kwargs,
                called_at_unix=started_unix,
                duration_ms=_elapsed_ms(started_perf),
                outcome="error",
                error_type=exc.__class__.__name__,
            )
            raise
        log_helper_call(
            helper_name,
            args,
            kwargs,
            called_at_unix=started_unix,
            duration_ms=_elapsed_ms(started_perf),
            outcome="ok",
            error_type=None,
        )
        return result

    return cast(T, wrapper)


def _elapsed_ms(started_perf: float) -> float:
    return max(0.0, (time.perf_counter() - started_perf) * 1000.0)


def _call_payload(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    called_at_unix: float | None,
    duration_ms: float | None,
    outcome: str,
    error_type: str | None,
) -> dict[str, Any]:
    return {
        "helper": name,
        "called_at_unix": time.time() if called_at_unix is None else called_at_unix,
        "run_id": os.environ.get("UV_AGENT_RUNTIME_RUN_ID"),
        "thread_id": os.environ.get("UV_AGENT_RUNTIME_THREAD_ID"),
        "turn_id": os.environ.get("UV_AGENT_RUNTIME_TURN_ID"),
        "cwd": str(Path.cwd()),
        "pid": os.getpid(),
        "positional_count": len(args),
        "keyword_names_json": json.dumps(sorted(str(key) for key in kwargs), separators=(",", ":")),
        "argument_types_json": json.dumps(_argument_types(args, kwargs), sort_keys=True, separators=(",", ":")),
        "duration_ms": duration_ms,
        "outcome": outcome,
        "error_type": error_type,
    }


def _argument_types(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "args": [_type_name(value) for value in args],
        "kwargs": {str(key): _type_name(value) for key, value in kwargs.items()},
    }


def _type_name(value: Any) -> str:
    if isinstance(value, Path):
        return "Path"
    return type(value).__name__


def _record_local(payload: dict[str, Any]) -> None:
    db_path = helper_stats_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _process_lock:
        db = _connection(db_path)
        db.execute(
            """
            INSERT INTO helper_calls(
                helper, called_at_unix, run_id, thread_id, turn_id, cwd, pid,
                positional_count, keyword_names_json, argument_types_json,
                duration_ms, outcome, error_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("helper"),
                payload.get("called_at_unix"),
                payload.get("run_id"),
                payload.get("thread_id"),
                payload.get("turn_id"),
                payload.get("cwd"),
                payload.get("pid"),
                payload.get("positional_count"),
                payload.get("keyword_names_json"),
                payload.get("argument_types_json"),
                payload.get("duration_ms"),
                payload.get("outcome"),
                payload.get("error_type"),
            ),
        )
        db.commit()


def _connection(db_path: Path) -> sqlite3.Connection:
    global _exit_registered, _process_connection, _process_connection_path
    if _process_connection is not None and _process_connection_path != db_path:
        _process_connection.close()
        _process_connection = None
        _process_connection_path = None
    if _process_connection is None:
        _process_connection = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT_SECONDS, check_same_thread=False)
        _process_connection_path = db_path
        _process_connection.execute("PRAGMA busy_timeout=5000")
        _process_connection.execute("PRAGMA journal_mode=WAL")
        _process_connection.execute("PRAGMA synchronous=NORMAL")
        _ensure_schema(_process_connection)
        if not _exit_registered:
            atexit.register(_close_connection)
            _exit_registered = True
    return _process_connection


def _ensure_schema(db: sqlite3.Connection) -> None:
    with db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS helper_calls (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              helper TEXT NOT NULL,
              called_at_unix REAL NOT NULL,
              run_id TEXT,
              thread_id TEXT,
              turn_id TEXT,
              cwd TEXT,
              pid INTEGER,
              positional_count INTEGER NOT NULL DEFAULT 0,
              keyword_names_json TEXT NOT NULL DEFAULT '[]',
              argument_types_json TEXT NOT NULL DEFAULT '{}',
              duration_ms REAL,
              outcome TEXT NOT NULL DEFAULT 'ok',
              error_type TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_helper_calls_helper_time
              ON helper_calls(helper, called_at_unix DESC);

            CREATE INDEX IF NOT EXISTS idx_helper_calls_run_id
              ON helper_calls(run_id);

            CREATE INDEX IF NOT EXISTS idx_helper_calls_thread_turn
              ON helper_calls(thread_id, turn_id);
            """
        )
        _ensure_column(db, "duration_ms", "REAL")
        _ensure_column(db, "outcome", "TEXT NOT NULL DEFAULT 'ok'")
        _ensure_column(db, "error_type", "TEXT")
        db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))


def _ensure_column(db: sqlite3.Connection, name: str, definition: str) -> None:
    columns = {str(row[1]) for row in db.execute("PRAGMA table_info(helper_calls)").fetchall()}
    if name not in columns:
        db.execute(f"ALTER TABLE helper_calls ADD COLUMN {name} {definition}")


def _close_connection() -> None:
    global _process_connection, _process_connection_path
    with _process_lock:
        if _process_connection is None:
            return
        _process_connection.close()
        _process_connection = None
        _process_connection_path = None
