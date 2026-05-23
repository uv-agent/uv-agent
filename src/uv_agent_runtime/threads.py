from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

DB_FILENAME = "uv-agent.sqlite3"


def thread_digest(
    thread_id: str,
    *,
    state_dir: str | Path | None = None,
    kind: str | None = None,
    since_last_compaction: bool = True,
    include_tools: bool = False,
) -> dict[str, Any]:
    """Return a compact human/assistant digest for one stored thread."""
    base = _state_dir(state_dir)
    with _connect(base) as db:
        metadata = _read_metadata(db, thread_id, kind=kind)
        if since_last_compaction:
            events, compaction = _read_after_latest_compaction(db, thread_id, metadata)
        else:
            events = _read_events(db, thread_id)
            compaction = None
    return {
        "thread_id": thread_id,
        "title": metadata.get("title") or "New thread",
        "created_at": metadata.get("created_at"),
        "updated_at": metadata.get("updated_at"),
        "last_text": metadata.get("last_text") or "",
        "turn_count": int(metadata.get("turn_count") or 0),
        "interrupted_turn_count": int(metadata.get("interrupted_turn_count") or 0),
        "latest_compaction": _compaction_summary(compaction or metadata.get("latest_compaction")),
        "items": _digest_items(events, include_tools=include_tools),
    }


def list_thread_digests(
    *,
    state_dir: str | Path | None = None,
    limit: int = 10,
    kind: str = "thread",
    parent_thread_id: str | None = None,
    since_last_compaction: bool = True,
    include_tools: bool = False,
) -> list[dict[str, Any]]:
    """Return compact digests for recent stored threads."""
    base = _state_dir(state_dir)
    with _connect(base) as db:
        clauses = ["kind = ?"]
        params: list[Any] = [kind]
        if parent_thread_id is not None:
            clauses.append("parent_thread_id = ?")
            params.append(parent_thread_id)
        rows = db.execute(
            f"SELECT thread_id FROM threads WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [
        thread_digest(
            str(row["thread_id"]),
            state_dir=base,
            kind=kind,
            since_last_compaction=since_last_compaction,
            include_tools=include_tools,
        )
        for row in rows
    ]


def _state_dir(state_dir: str | Path | None) -> Path:
    if state_dir is not None:
        return Path(state_dir).resolve()
    env = os.environ.get("UV_AGENT_RUNTIME_STATE_DIR")
    if not env:
        raise RuntimeError("UV_AGENT_RUNTIME_STATE_DIR is not set; pass state_dir explicitly")
    return Path(env).resolve()


def _connect(base: Path) -> sqlite3.Connection:
    path = base / DB_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing uv-agent state database: {path}")
    # Runtime scripts only introspect conversation state. Opening in read-only
    # URI mode prevents helper bugs from mutating host-owned project state.
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _read_metadata(db: sqlite3.Connection, thread_id: str, *, kind: str | None) -> dict[str, Any]:
    if kind is None:
        row = db.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
    else:
        row = db.execute("SELECT * FROM threads WHERE thread_id = ? AND kind = ?", (thread_id, kind)).fetchone()
    if row is None:
        raise FileNotFoundError(f"Missing thread metadata for {thread_id}")
    return _metadata_from_row(row)


def _read_events(
    db: sqlite3.Connection,
    thread_id: str,
    *,
    event_id_gte: int | None = None,
) -> list[dict[str, Any]]:
    clauses = ["thread_id = ?"]
    params: list[Any] = [thread_id]
    if event_id_gte is not None:
        clauses.append("event_id >= ?")
        params.append(event_id_gte)
    rows = db.execute(
        f"""
        SELECT event_id, payload_json
        FROM thread_events
        WHERE {' AND '.join(clauses)}
        ORDER BY event_id ASC
        """,
        params,
    ).fetchall()
    return [_event_from_row(row) for row in rows]


def _read_after_latest_compaction(
    db: sqlite3.Connection,
    thread_id: str,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    compaction_event_id = _int_or_none(metadata.get("latest_compaction_event_id"))
    if compaction_event_id is None:
        return _read_events(db, thread_id), None
    events = _read_events(db, thread_id, event_id_gte=compaction_event_id)
    if not events:
        return [], None
    return events[1:], events[0]


def _metadata_from_row(row: sqlite3.Row) -> dict[str, Any]:
    metadata = _json_loads(row["metadata_json"], default={})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.update(
        {
            "thread_id": row["thread_id"],
            "kind": row["kind"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "turn_count": int(row["turn_count"] or 0),
            "interrupted_turn_count": int(row["interrupted_turn_count"] or 0),
            "last_text": row["last_text"] or "",
        }
    )
    for key in ("parent_thread_id", "latest_compaction_event_id"):
        if row[key] is not None:
            metadata[key] = row[key]
    metadata["latest_compaction"] = _json_loads(row["latest_compaction_json"], default=None)
    return {key: value for key, value in metadata.items() if value is not None}


def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    event = _json_loads(row["payload_json"], default={})
    if not isinstance(event, dict):
        event = {}
    event["_event_id"] = int(row["event_id"])
    return event


def _digest_items(events: list[dict[str, Any]], *, include_tools: bool) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    completed_assistant_turns: set[str] = set()

    for event in events:
        event_type = event.get("type")
        turn_id = str(event.get("turn_id") or "")
        if event_type == "item.user":
            text = _item_text(event.get("item") or {})
            if text:
                items.append({"role": "user", "text": text})
        elif event_type in {"item.assistant", "item.assistant_partial"}:
            if turn_id in completed_assistant_turns:
                continue
            completed_assistant_turns.add(turn_id)
            text = str(event.get("text") or "")
            if text:
                items.append({"role": "assistant", "text": text})
        elif event_type == "item.model_response":
            if turn_id in completed_assistant_turns:
                continue
            text = _model_response_text(event.get("output") or [])
            if text:
                completed_assistant_turns.add(turn_id)
                items.append({"role": "assistant", "text": text})
        elif event_type == "item.compaction":
            text = str(event.get("text") or "")
            items.append({"role": "summary", "text": text})
        elif event_type == "turn.interrupted":
            items.append({"role": "system", "text": f"turn interrupted: {event.get('reason') or 'user_interrupt'}"})
        elif event_type == "turn.error":
            items.append(
                {
                    "role": "system",
                    "text": f"turn error: {event.get('message') or event.get('error_type') or 'unknown error'}",
                }
            )
        elif include_tools and event_type in {"item.runner_result", "item.tool_output"}:
            items.append({"role": "tool", "text": _tool_event_text(event)})
    return items


def _item_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for content in item.get("content") or []:
        if content.get("type") in {"input_text", "output_text", "text", "refusal"}:
            parts.append(str(content.get("text") or ""))
    return "\n".join(parts)


def _model_response_text(output: list[dict[str, Any]]) -> str:
    return "\n".join(
        text
        for item in output
        if item.get("type") == "message"
        for text in [_item_text(item)]
        if text
    )


def _tool_event_text(event: dict[str, Any]) -> str:
    event_type = event.get("type")
    if event_type == "item.runner_result":
        result = event.get("result") or {}
        return f"run_python rc={result.get('returncode')} run={result.get('run_id') or ''}".strip()
    return event_type or "tool"


def _compaction_summary(compaction: Any) -> dict[str, Any] | None:
    if not isinstance(compaction, dict):
        return None
    return {
        "created_at": compaction.get("created_at"),
        "turn_id": compaction.get("turn_id"),
        "text": compaction.get("text") or "",
    }


def _json_loads(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None
