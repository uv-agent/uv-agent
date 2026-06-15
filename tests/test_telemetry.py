from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from uv_agent.telemetry import TelemetryStore, _duration_ms, _summarize_helper_calls


def test_telemetry_store_records_model_call(tmp_path: Path) -> None:
    telemetry = TelemetryStore(tmp_path)

    telemetry.on_event(
        {
            "type": "agent.model_call_billed",
            "thread_id": "thr_1",
            "turn_id": "turn_1",
            "level": "medium",
            "source": "model_response",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "billing": {
                "source": "model_response",
                "amount": "0.0015",
                "currency": "USD",
                "model": "default",
                "remote_model": "gpt-4",
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "output_tokens": 5,
                "reasoning_tokens": 0,
            },
        }
    )

    with telemetry._connect() as db:
        rows = db.execute("SELECT * FROM model_calls").fetchall()

    assert len(rows) == 1
    assert rows[0]["thread_id"] == "thr_1"
    assert rows[0]["turn_id"] == "turn_1"
    assert rows[0]["input_tokens"] == 10
    assert rows[0]["output_tokens"] == 5
    assert rows[0]["billing_amount"] == "0.0015"
    assert rows[0]["billing_currency"] == "USD"


def test_telemetry_store_records_run_completed(tmp_path: Path) -> None:
    telemetry = TelemetryStore(tmp_path)

    telemetry.on_event(
        {
            "type": "thread.event_stored",
            "thread_id": "thr_1",
            "event": {
                "type": "turn.started",
                "turn_id": "turn_1",
                "created_at": "2026-06-15T10:00:00+00:00",
            },
        }
    )
    telemetry.on_event(
        {
            "type": "runner.run_completed",
            "run_id": "run_1",
            "thread_id": "thr_1",
            "turn_id": "turn_1",
            "started_at": "2026-06-15T10:00:00+00:00",
            "completed_at": "2026-06-15T10:00:01+00:00",
            "returncode": 0,
            "timed_out": False,
            "interrupted": False,
            "truncated": False,
            "stdout_bytes": 12,
            "stderr_bytes": 0,
            "event_count": 3,
            "helper_calls": [
                {"name": "read_file", "count": 2, "total_duration_ms": 30.0},
                {"name": "search_text", "count": 1, "outcomes": {"error": 1}},
            ],
        }
    )
    telemetry.on_event(
        {
            "type": "thread.event_stored",
            "thread_id": "thr_1",
            "event": {
                "type": "turn.completed",
                "turn_id": "turn_1",
                "created_at": "2026-06-15T10:00:02+00:00",
            },
        }
    )

    row = telemetry.query_turn_stats("turn_1")
    assert row is not None
    assert row["run_python_calls"] == 1
    assert row["helper_calls_total"] == 3
    assert row["run_python_total_duration_ms"] == 1000.0
    assert row["run_python_errors"] == 0

    with telemetry._connect() as db:
        run_rows = db.execute("SELECT * FROM run_stats WHERE run_id = ?", ("run_1",)).fetchall()

    assert len(run_rows) == 1
    assert run_rows[0]["helper_count"] == 3
    assert run_rows[0]["helper_errors"] == 1
    assert run_rows[0]["stdout_bytes"] == 12


def test_telemetry_store_aggregates_turn(tmp_path: Path) -> None:
    telemetry = TelemetryStore(tmp_path)

    telemetry.on_event(
        {
            "type": "thread.event_stored",
            "thread_id": "thr_1",
            "event": {
                "type": "turn.started",
                "turn_id": "turn_1",
                "created_at": "2026-06-15T10:00:00+00:00",
            },
        }
    )
    telemetry.on_event(
        {
            "type": "agent.model_call_billed",
            "thread_id": "thr_1",
            "turn_id": "turn_1",
            "level": "medium",
            "source": "model_response",
            "usage": {},
            "billing": {
                "amount": "0.001",
                "currency": "USD",
                "input_tokens": 5,
                "cached_input_tokens": 0,
                "output_tokens": 3,
                "reasoning_tokens": 1,
            },
        }
    )
    telemetry.on_event(
        {
            "type": "thread.event_stored",
            "thread_id": "thr_1",
            "event": {
                "type": "turn.completed",
                "turn_id": "turn_1",
                "created_at": "2026-06-15T10:00:02+00:00",
            },
        }
    )

    row = telemetry.query_turn_stats("turn_1")
    assert row is not None
    assert row["status"] == "completed"
    assert row["model_calls"] == 1
    assert row["input_tokens"] == 5
    assert row["output_tokens"] == 3
    assert row["total_tokens"] == 8
    assert row["reasoning_tokens"] == 1
    assert row["billing_amount"] == "0.001"
    assert row["billing_currency"] == "USD"
    assert row["duration_ms"] == 2000.0


def test_telemetry_store_turn_error_status(tmp_path: Path) -> None:
    telemetry = TelemetryStore(tmp_path)

    telemetry.on_event(
        {
            "type": "thread.event_stored",
            "thread_id": "thr_1",
            "event": {
                "type": "turn.started",
                "turn_id": "turn_1",
                "created_at": "2026-06-15T10:00:00+00:00",
            },
        }
    )
    telemetry.on_event(
        {
            "type": "thread.event_stored",
            "thread_id": "thr_1",
            "event": {
                "type": "turn.error",
                "turn_id": "turn_1",
                "created_at": "2026-06-15T10:00:01+00:00",
            },
        }
    )

    row = telemetry.query_turn_stats("turn_1")
    assert row is not None
    assert row["status"] == "error"


def test_summarize_helper_calls_counts_and_errors() -> None:
    calls = [
        {"name": "read_file", "count": 2, "total_duration_ms": 30.0},
        {"name": "search_text", "count": 1, "outcomes": {"error": 1}},
        {"name": "write_file", "outcome": "error"},
    ]
    summary = _summarize_helper_calls(calls)
    assert summary["count"] == 4
    assert summary["total_duration_ms"] == 30.0
    assert summary["errors"] == 2
    assert summary["top_helpers"]["read_file"] == 2


def test_duration_ms_parses_iso() -> None:
    assert _duration_ms(
        "2026-06-15T10:00:00+00:00",
        "2026-06-15T10:00:01.500+00:00",
    ) == 1500.0
    assert _duration_ms(None, "2026-06-15T10:00:01+00:00") is None
