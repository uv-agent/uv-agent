from __future__ import annotations

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
