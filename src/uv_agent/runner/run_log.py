from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Protocol

from uv_agent.state_db import connect_state_db


class EventWriter(Protocol):
    """Minimal writer interface shared by runner output and RPC sessions."""

    def write(self, event: dict[str, Any]) -> dict[str, Any]: ...


class RunEventWriter:
    """SQLite-backed event writer for one managed Python run."""

    def __init__(self, data_dir: Path, run_id: str) -> None:
        self.data_dir = data_dir.resolve()
        self.run_id = run_id
        self._lock = threading.RLock()

    def write(self, event: dict[str, Any]) -> dict[str, Any]:
        event_type = str(event.get("type") or "")
        created_at = str(event.get("created_at") or "")
        stored_without_id = {**event, "run_id": event.get("run_id") or self.run_id}
        with self._lock, connect_state_db(self.data_dir) as db:
            cursor = db.execute(
                """
                INSERT INTO run_events(run_id, type, created_at, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    event_type,
                    created_at,
                    _json_dumps(stored_without_id),
                ),
            )
            event_id = int(cursor.lastrowid)
            stored = {**stored_without_id, "_event_id": event_id}
            db.execute(
                "UPDATE run_events SET payload_json = ? WHERE event_id = ?",
                (_json_dumps(stored), event_id),
            )
        return stored


class RunLogStore:
    """Store managed Python run records and events in the project SQLite DB.

    The name is kept to minimize call-site churn while the backing storage is no
    longer JSONL. Script files are still exported for process execution and
    debugging, but the run code stored in SQLite is the source of truth.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        scripts_dir: Path | None = None,
        max_run_logs: int = 200,
    ) -> None:
        self.data_dir = data_dir.resolve()
        self.scripts_dir = (scripts_dir or self.data_dir / "runner" / "scripts").resolve()
        self.max_run_logs = max(1, max_run_logs)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        with connect_state_db(self.data_dir):
            pass

    def script_path(self, run_id: str) -> Path:
        return self.scripts_dir / f"{run_id}.py"

    def writer(self, run_id: str) -> RunEventWriter:
        return RunEventWriter(self.data_dir, run_id)

    def create_run_record(
        self,
        *,
        run_id: str,
        code: str,
        script_args: list[str],
        cwd: Path,
        timeout_s: float | None,
        started_at: str,
        thread_id: str | None,
        turn_id: str | None,
        script_path: Path | None,
    ) -> Path | None:
        script_path = script_path or self.script_path(run_id)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(code, encoding="utf-8")
        with connect_state_db(self.data_dir) as db:
            db.execute(
                """
                INSERT INTO runs(
                    run_id, thread_id, turn_id, cwd, code, script_args_json, timeout_s,
                    started_at, script_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    thread_id,
                    turn_id,
                    str(cwd),
                    code,
                    _json_dumps(script_args),
                    timeout_s,
                    started_at,
                    str(script_path) if script_path is not None else None,
                ),
            )
        return script_path

    def complete_run(
        self,
        *,
        run_id: str,
        completed_at: str,
        returncode: int | None,
        timed_out: bool,
        interrupted: bool,
        truncated: bool,
        stdout: str,
        stderr: str,
        structured_events: list[dict[str, Any]],
    ) -> None:
        with connect_state_db(self.data_dir) as db:
            db.execute(
                """
                UPDATE runs
                SET completed_at = ?, returncode = ?, timed_out = ?, interrupted = ?,
                    truncated = ?, stdout = ?, stderr = ?, structured_events_json = ?
                WHERE run_id = ?
                """,
                (
                    completed_at,
                    returncode,
                    int(timed_out),
                    int(interrupted),
                    int(truncated),
                    stdout,
                    stderr,
                    _json_dumps(structured_events),
                    run_id,
                ),
            )

    def read_events(self, run_id: str) -> list[dict[str, Any]]:
        with connect_state_db(self.data_dir) as db:
            rows = db.execute(
                """
                SELECT event_id, payload_json
                FROM run_events
                WHERE run_id = ?
                ORDER BY event_id ASC
                """,
                (run_id,),
            ).fetchall()
        return [_event_from_row(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with connect_state_db(self.data_dir) as db:
            row = db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["script_args"] = _json_loads(data.pop("script_args_json"), default=[])
        data["structured_events"] = _json_loads(data.pop("structured_events_json"), default=[])
        for key in ("timed_out", "interrupted", "truncated"):
            data[key] = bool(data[key])
        return data

    def prune(self) -> None:
        with connect_state_db(self.data_dir) as db:
            rows = db.execute(
                """
                SELECT run_id, script_path
                FROM runs
                WHERE completed_at IS NOT NULL
                ORDER BY completed_at DESC, started_at DESC, rowid DESC
                """
            ).fetchall()
            stale = rows[self.max_run_logs :]
            if stale:
                run_ids = [row["run_id"] for row in stale]
                db.execute(
                    f"DELETE FROM runs WHERE run_id IN ({', '.join('?' for _ in run_ids)})",
                    run_ids,
                )
        for row in stale:
            script_path = row["script_path"]
            if script_path:
                Path(script_path).unlink(missing_ok=True)


def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    event = _json_loads(row["payload_json"], default={})
    if not isinstance(event, dict):
        event = {}
    event["_event_id"] = int(row["event_id"])
    return event


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default
