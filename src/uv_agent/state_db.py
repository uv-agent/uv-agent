from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from uv_agent.time import utc_now_iso

DB_FILENAME = "uv-agent.sqlite3"
SCHEMA_VERSION = 7
SQLITE_TIMEOUT_SECONDS = 30.0
SQLITE_BUSY_TIMEOUT_MS = int(SQLITE_TIMEOUT_SECONDS * 1000)

logger = logging.getLogger(__name__)


class StateDbConnection(sqlite3.Connection):
    """SQLite connection that closes when used as a context manager."""

    _close_on_context_exit: bool

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            if getattr(self, "_close_on_context_exit", True):
                self.close()


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
        factory=StateDbConnection,
    )
    connection._close_on_context_exit = False
    connection.row_factory = sqlite3.Row
    # ``timeout`` only affects locks encountered while opening the connection.
    # PRAGMA busy_timeout is per-connection and covers later statements, which
    # matters when multiple host and runtime processes append to the same project DB.
    connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        _ensure_schema(connection)
    except Exception:
        connection.close()
        raise
    connection._close_on_context_exit = True
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create or migrate the project state schema to the current version."""

    _ensure_wal(connection)
    existing_version = _read_schema_version(connection)
    if existing_version == str(SCHEMA_VERSION):
        return
    if existing_version is None:
        logger.info("Creating state database schema version=%s", SCHEMA_VERSION)
        _create_schema(connection)
        return

    logger.info("Migrating state database schema from version=%s to version=%s", existing_version, SCHEMA_VERSION)
    migrations = {
        "1": _migrate_v1_to_v2,
        "2": _migrate_v2_to_v3,
        "3": _migrate_v3_to_v4,
        "4": _migrate_v4_to_v5,
        "5": _migrate_v5_to_v6,
        "6": _migrate_v6_to_v7,
    }
    version = existing_version
    while version != str(SCHEMA_VERSION):
        migrate = migrations.get(str(version))
        if migrate is None:
            raise StateDbError(
                f"Unsupported state database schema version {version}; "
                f"expected {SCHEMA_VERSION}"
            )
        migrate(connection)
        version = _read_schema_version(connection)


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
              helper_calls_json TEXT,

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
        _create_host_lease_schema(connection)
        _create_plugin_storage_schema(connection)
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        connection.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('created_at', ?)",
            (utc_now_iso(),),
        )


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    """Advance old project databases past the former plugin-table schema version."""

    logger.info("Migrating state database schema version 1 -> 2")
    with connection:
        _create_host_lease_schema(connection)
        connection.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            ("2",),
        )


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    """Add runtime helper-call summaries to persisted run records."""

    logger.info("Migrating state database schema version 2 -> 3")
    with connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(runs)")}
        if "helper_calls_json" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN helper_calls_json TEXT")
        connection.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            ("3",),
        )


def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
    """Advance old project databases past a former plugin-table schema version."""

    logger.info("Migrating state database schema version 3 -> 4")
    with connection:
        connection.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            ("4",),
        )


def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
    """Advance old project databases past a former plugin-table lease version."""

    logger.info("Migrating state database schema version 4 -> 5")
    with connection:
        _create_host_lease_schema(connection)
        connection.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            ("5",),
        )


def _migrate_v5_to_v6(connection: sqlite3.Connection) -> None:
    """Add host daemon lease table."""

    logger.info("Migrating state database schema version 5 -> 6")
    with connection:
        _create_host_lease_schema(connection)
        connection.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            ("6",),
        )


def _migrate_v6_to_v7(connection: sqlite3.Connection) -> None:
    """Add core-managed plugin storage tables."""

    logger.info("Migrating state database schema version 6 -> 7")
    with connection:
        _create_plugin_storage_schema(connection)
        connection.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            ("7",),
        )


def _create_host_lease_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS host_leases (
          name TEXT PRIMARY KEY,
          owner_id TEXT NOT NULL,
          pid INTEGER,
          heartbeat_at TEXT NOT NULL,
          metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )


def _create_plugin_storage_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS plugin_kv (
          plugin_id TEXT NOT NULL,
          scope TEXT NOT NULL,
          scope_id TEXT NOT NULL DEFAULT '',
          key TEXT NOT NULL,
          value_json TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(plugin_id, scope, scope_id, key)
        );
        CREATE INDEX IF NOT EXISTS idx_plugin_kv_scope_key
          ON plugin_kv(plugin_id, scope, scope_id, key);

        CREATE TABLE IF NOT EXISTS plugin_documents (
          plugin_id TEXT NOT NULL,
          scope TEXT NOT NULL,
          scope_id TEXT NOT NULL DEFAULT '',
          collection TEXT NOT NULL,
          doc_id TEXT NOT NULL,
          body_json TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(plugin_id, scope, scope_id, collection, doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_plugin_documents_collection_updated
          ON plugin_documents(plugin_id, scope, scope_id, collection, updated_at DESC, doc_id);

        CREATE TABLE IF NOT EXISTS plugin_document_indexes (
          plugin_id TEXT NOT NULL,
          scope TEXT NOT NULL,
          scope_id TEXT NOT NULL DEFAULT '',
          collection TEXT NOT NULL,
          field TEXT NOT NULL,
          value TEXT NOT NULL,
          doc_id TEXT NOT NULL,
          PRIMARY KEY(plugin_id, scope, scope_id, collection, field, value, doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_plugin_document_indexes_lookup
          ON plugin_document_indexes(plugin_id, scope, scope_id, collection, field, value);
        """
    )


def _ensure_wal(connection: sqlite3.Connection) -> None:
    """Switch to WAL only when needed so normal opens stay read-mostly."""

    journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
    current = str(journal_mode[0] if journal_mode else "").lower()
    if current != "wal":
        logger.info("Switching state database journal_mode to WAL current=%s", current or "unknown")
        connection.execute("PRAGMA journal_mode=WAL")
    # NORMAL is the usual durability/concurrency trade-off for SQLite WAL and
    # avoids extra fsync pressure when many short-lived processes append events.
    connection.execute("PRAGMA synchronous=NORMAL")


def checkpoint_state_db(data_dir: Path, *, mode: str = "PASSIVE") -> None:
    """Run a WAL checkpoint on the project state database.

    PASSIVE checkpoints as much as possible without blocking readers or writers.
    TRUNCATE resets the WAL file after checkpointing and may block briefly.
    """

    db_path = state_db_path(data_dir)
    if not db_path.exists():
        return
    connection = sqlite3.connect(
        db_path,
        timeout=SQLITE_TIMEOUT_SECONDS,
        check_same_thread=True,
    )
    try:
        logger.debug("Running state database checkpoint path=%s mode=%s", db_path, mode)
        connection.execute(f"PRAGMA wal_checkpoint({mode})")
    finally:
        connection.close()


def _read_schema_version(connection: sqlite3.Connection) -> str | None:
    """Return the stored schema version, or None before the schema exists."""

    try:
        row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return None
        raise
    return str(row["value"]) if row is not None else None
