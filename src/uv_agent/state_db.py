from __future__ import annotations

import sqlite3
from pathlib import Path

from uv_agent.time import utc_now_iso

DB_FILENAME = "uv-agent.sqlite3"
SCHEMA_VERSION = 2
SQLITE_TIMEOUT_SECONDS = 30.0
SQLITE_BUSY_TIMEOUT_MS = int(SQLITE_TIMEOUT_SECONDS * 1000)


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
    connection = sqlite3.connect(
        db_path,
        timeout=SQLITE_TIMEOUT_SECONDS,
        check_same_thread=check_same_thread,
    )
    connection.row_factory = sqlite3.Row
    # ``timeout`` only affects locks encountered while opening the connection.
    # PRAGMA busy_timeout is per-connection and covers later statements, which
    # matters when multiple ask subprocesses append to the same project DB.
    connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(connection)
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create or migrate the project state schema to the current version."""

    _ensure_wal(connection)
    existing_version = _read_schema_version(connection)
    if existing_version == str(SCHEMA_VERSION):
        return
    if existing_version is None:
        _create_schema(connection)
        return
    if existing_version == "1":
        _migrate_v1_to_v2(connection)
        return
    raise StateDbError(
        f"Unsupported state database schema version {existing_version}; "
        f"expected {SCHEMA_VERSION}"
    )


def _create_schema(connection: sqlite3.Connection) -> None:
    """Create the full schema for a new project state database."""

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
        _create_workflow_schema(connection)
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        connection.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('created_at', ?)",
            (utc_now_iso(),),
        )


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    """Add workflow persistence tables to an existing v1 project database."""

    with connection:
        _create_workflow_schema(connection)
        connection.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION),),
        )


def _create_workflow_schema(connection: sqlite3.Connection) -> None:
    """Create workflow task-graph tables and indexes."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS workflows (
          workflow_id TEXT PRIMARY KEY,
          parent_thread_id TEXT,
          parent_turn_id TEXT,
          parent_run_id TEXT,
          objective TEXT NOT NULL,
          status TEXT NOT NULL,
          default_model_level TEXT,
          current_checkpoint_id TEXT,
          state_json TEXT NOT NULL DEFAULT '{}',
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS workflow_nodes (
          node_id TEXT PRIMARY KEY,
          workflow_id TEXT NOT NULL,
          key TEXT,
          kind TEXT NOT NULL,
          status TEXT NOT NULL,
          dependencies_json TEXT NOT NULL DEFAULT '[]',
          prompt TEXT,
          model_level TEXT,
          thread_id TEXT,
          run_id TEXT,
          result_summary TEXT,
          result_json TEXT NOT NULL DEFAULT '{}',
          error_json TEXT NOT NULL DEFAULT '{}',
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL,
          started_at TEXT,
          completed_at TEXT,
          FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS workflow_checkpoints (
          checkpoint_id TEXT PRIMARY KEY,
          workflow_id TEXT NOT NULL,
          node_id TEXT NOT NULL,
          key TEXT NOT NULL,
          status TEXT NOT NULL,
          reason TEXT NOT NULL,
          options_json TEXT NOT NULL DEFAULT '[]',
          recommended_action TEXT,
          snapshot_json TEXT NOT NULL DEFAULT '{}',
          resolution_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL,
          resolved_at TEXT,
          FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
          FOREIGN KEY(node_id) REFERENCES workflow_nodes(node_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS workflow_events (
          event_id INTEGER PRIMARY KEY AUTOINCREMENT,
          workflow_id TEXT NOT NULL,
          node_id TEXT,
          type TEXT NOT NULL,
          created_at TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_workflow_nodes_workflow_status
          ON workflow_nodes(workflow_id, status);
        CREATE INDEX IF NOT EXISTS idx_workflow_nodes_workflow_key
          ON workflow_nodes(workflow_id, key);
        CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_workflow_status
          ON workflow_checkpoints(workflow_id, status);
        CREATE INDEX IF NOT EXISTS idx_workflow_events_workflow_id
          ON workflow_events(workflow_id, event_id);
        CREATE INDEX IF NOT EXISTS idx_workflows_parent_thread
          ON workflows(parent_thread_id, status);
        """
    )

def _ensure_wal(connection: sqlite3.Connection) -> None:
    """Switch to WAL only when needed so normal opens stay read-mostly."""

    journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
    current = str(journal_mode[0] if journal_mode else "").lower()
    if current != "wal":
        connection.execute("PRAGMA journal_mode=WAL")
    # NORMAL is the usual durability/concurrency trade-off for SQLite WAL and
    # avoids extra fsync pressure when many short-lived processes append events.
    connection.execute("PRAGMA synchronous=NORMAL")


def _read_schema_version(connection: sqlite3.Connection) -> str | None:
    """Return the stored schema version, or None before the schema exists."""

    try:
        row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return None
        raise
    return str(row["value"]) if row is not None else None
