from __future__ import annotations

import sqlite3
from pathlib import Path

from uv_agent.time import utc_now_iso

DB_FILENAME = "uv-agent.sqlite3"
SCHEMA_VERSION = 1


class StateDbError(RuntimeError):
    """Raised when the project state database cannot be used safely."""


def state_db_path(data_dir: Path) -> Path:
    """Return the canonical SQLite database path for a project state directory."""

    return Path(data_dir).resolve() / DB_FILENAME


def connect_state_db(data_dir: Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open and initialize the project state database.

    Connections are intentionally short-lived in most callers. SQLite PRAGMAs are
    per-connection, so every opener applies the durability/concurrency settings
    required by uv-agent before returning the handle.
    """

    db_path = state_db_path(data_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    _ensure_schema(connection)
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create the v1 schema or reject databases from a different future schema."""

    # WAL is persistent for the database file, but executing it during init keeps
    # freshly-created project state in the desired mode before any concurrent run
    # writers appear.
    connection.execute("PRAGMA journal_mode=WAL")
    with connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS threads (
              thread_id TEXT PRIMARY KEY,
              kind TEXT NOT NULL DEFAULT 'thread',
              title TEXT NOT NULL DEFAULT 'New thread',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,

              parent_thread_id TEXT,
              parent_turn_id TEXT,
              parent_run_id TEXT,

              active_level TEXT,
              active_model TEXT,
              latest_cwd TEXT,

              turn_count INTEGER NOT NULL DEFAULT 0,
              interrupted_turn_count INTEGER NOT NULL DEFAULT 0,
              user_message_count INTEGER NOT NULL DEFAULT 0,

              last_text TEXT NOT NULL DEFAULT '',
              last_event_id INTEGER,
              latest_compaction_event_id INTEGER,
              latest_usage_tokens INTEGER,

              latest_model_switch_warning_json TEXT,
              latest_compaction_json TEXT,
              billing_currency TEXT,
              billing_total TEXT,
              billing_totals_json TEXT,

              metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_threads_kind_updated_at
              ON threads(kind, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_threads_parent_thread_id_updated_at
              ON threads(parent_thread_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS thread_events (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              thread_id TEXT NOT NULL,
              turn_id TEXT,
              type TEXT NOT NULL,
              created_at TEXT NOT NULL,
              payload_json TEXT NOT NULL,

              FOREIGN KEY(thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_thread_events_thread_id_event_id
              ON thread_events(thread_id, event_id);

            CREATE INDEX IF NOT EXISTS idx_thread_events_thread_type_event_id
              ON thread_events(thread_id, type, event_id);

            CREATE INDEX IF NOT EXISTS idx_thread_events_turn_id_event_id
              ON thread_events(turn_id, event_id);

            CREATE TABLE IF NOT EXISTS thread_locks (
              thread_id TEXT PRIMARY KEY,
              owner_id TEXT NOT NULL,
              token TEXT NOT NULL,
              pid INTEGER,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              thread_id TEXT,
              turn_id TEXT,
              cwd TEXT,
              code TEXT NOT NULL,
              script_args_json TEXT NOT NULL DEFAULT '[]',
              timeout_s REAL,
              started_at TEXT NOT NULL,
              completed_at TEXT,

              returncode INTEGER,
              timed_out INTEGER NOT NULL DEFAULT 0,
              interrupted INTEGER NOT NULL DEFAULT 0,
              truncated INTEGER NOT NULL DEFAULT 0,

              stdout TEXT NOT NULL DEFAULT '',
              stderr TEXT NOT NULL DEFAULT '',
              structured_events_json TEXT NOT NULL DEFAULT '[]',

              script_path TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_runs_thread_turn
              ON runs(thread_id, turn_id);

            CREATE INDEX IF NOT EXISTS idx_runs_started_at
              ON runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS run_events (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              type TEXT NOT NULL,
              created_at TEXT NOT NULL,
              payload_json TEXT NOT NULL,

              FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_run_events_run_id_event_id
              ON run_events(run_id, event_id);
            """
        )
        version = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if version is None:
            connection.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('created_at', ?)",
                (utc_now_iso(),),
            )
            return
        if version["value"] != str(SCHEMA_VERSION):
            raise StateDbError(
                f"Unsupported state database schema version {version['value']}; "
                f"expected {SCHEMA_VERSION}"
            )
