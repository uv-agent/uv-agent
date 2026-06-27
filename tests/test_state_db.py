from __future__ import annotations

import sqlite3
from pathlib import Path

from uv_agent.state_db import SCHEMA_VERSION, SQLITE_BUSY_TIMEOUT_MS, connect_state_db, state_db_path


def test_state_db_initializes_schema_and_pragmas(tmp_path: Path) -> None:
    db_path = state_db_path(tmp_path)

    with connect_state_db(tmp_path) as db:
        version = db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        busy_timeout = db.execute("PRAGMA busy_timeout").fetchone()[0]
        foreign_keys = db.execute("PRAGMA foreign_keys").fetchone()[0]
        journal_mode = db.execute("PRAGMA journal_mode").fetchone()[0]
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        workflow_node_columns = {row["name"] for row in db.execute("PRAGMA table_info(workflow_nodes)")}

    assert db_path.exists()
    assert version["value"] == str(SCHEMA_VERSION)
    assert busy_timeout == SQLITE_BUSY_TIMEOUT_MS
    assert foreign_keys == 1
    assert journal_mode.lower() == "wal"
    assert {
        "meta",
        "threads",
        "thread_events",
        "thread_locks",
        "runs",
        "run_events",
        "workflows",
        "workflow_nodes",
        "workflow_checkpoints",
        "workflow_events",
    } <= tables
    assert {"executor_id", "lease_until"} <= workflow_node_columns


def test_state_db_initialization_is_idempotent(tmp_path: Path) -> None:
    with connect_state_db(tmp_path) as db:
        db.execute("INSERT INTO meta(key, value) VALUES ('custom', 'kept')")

    with connect_state_db(tmp_path) as db:
        row = db.execute("SELECT value FROM meta WHERE key = 'custom'").fetchone()

    assert row["value"] == "kept"



def test_state_db_migrates_v1_database_to_workflows(tmp_path: Path) -> None:
    db_path = state_db_path(tmp_path)
    with connect_state_db(tmp_path) as db:
        db.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
        db.execute("DROP TABLE workflow_events")
        db.execute("DROP TABLE workflow_checkpoints")
        db.execute("DROP TABLE workflow_nodes")
        db.execute("DROP TABLE workflows")

    with connect_state_db(tmp_path) as db:
        version = db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        workflow_table = db.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'workflows'").fetchone()

    assert db_path.exists()
    assert version["value"] == str(SCHEMA_VERSION)
    assert workflow_table is not None


def test_state_db_migrates_v2_database_to_helper_calls(tmp_path: Path) -> None:
    db_path = state_db_path(tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta(key, value) VALUES ('schema_version', '2');
            CREATE TABLE runs (
              run_id TEXT PRIMARY KEY,
              code TEXT NOT NULL,
              script_args_json TEXT NOT NULL DEFAULT '[]',
              structured_events_json TEXT NOT NULL DEFAULT '[]'
            );
            """
        )

    with connect_state_db(tmp_path) as db:
        version = db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        columns = {row["name"] for row in db.execute("PRAGMA table_info(runs)").fetchall()}

    assert version["value"] == str(SCHEMA_VERSION)
    assert "helper_calls_json" in columns
