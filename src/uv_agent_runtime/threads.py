from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

DB_FILENAME = "uv-agent.sqlite3"
SQLITE_BUSY_TIMEOUT_MS = 30_000
REQUIRED_RUN_COLUMNS = {
    "run_id",
    "code",
    "script_args_json",
    "structured_events_json",
}

EpochSelector = Literal["latest", "all"] | int | Sequence[int | str]


class ThreadCompactionSummary(TypedDict):
    """Compact summary for one compaction boundary."""

    created_at: str | None
    turn_id: str | None
    text: str


class ThreadDigestItem(TypedDict):
    """Legacy compact conversation item used by thread_digest/list_thread_digests."""

    role: str
    text: str


class ThreadDigest(TypedDict):
    """Compact legacy digest for a stored thread."""

    thread_id: str
    title: str
    created_at: str | None
    updated_at: str | None
    last_text: str
    turn_count: int
    interrupted_turn_count: int
    latest_compaction: ThreadCompactionSummary | None
    items: list[ThreadDigestItem]


class BoundedText(TypedDict):
    """Text plus explicit truncation metadata."""

    text: str
    chars: int
    truncated: bool
    limit: int


class ThreadCompaction(TypedDict):
    """Compaction event closing an epoch."""

    id: str
    event_id: int
    turn_id: str | None
    created_at: str | None
    text: str


class ThreadEpoch(TypedDict):
    """Chronological conversation segment bounded by compaction events."""

    id: str
    index: int
    start_event_id: int
    end_event_id: int
    compaction: ThreadCompaction | None


class ConversationMessage(TypedDict):
    """Visible user/assistant message in a thread view."""

    id: str
    event_id: int
    role: Literal["user", "assistant"]
    text: str
    chars: int
    truncated: bool


class ProcessRef(TypedDict):
    """Lightweight reference to non-conversation process details."""

    id: str
    kind: str
    event_ref: str
    event_id: int
    turn_id: str
    status: str
    summary: str
    related_ids: NotRequired[list[str]]
    helper_names: NotRequired[list[str]]


class ThreadTurn(TypedDict):
    """One turn in a thread_view response."""

    id: str
    turn_id: str
    epoch_id: str
    status: str
    user_messages: list[ConversationMessage]
    assistant_messages: list[ConversationMessage]
    process_refs: list[ProcessRef]


class ThreadView(TypedDict):
    """Conversation-first view of a stored thread."""

    thread_id: str
    kind: str
    title: str
    created_at: str | None
    updated_at: str | None
    selected_epochs: list[str]
    epochs: list[ThreadEpoch]
    turns: list[ThreadTurn]
    truncated: bool


class HelperCall(TypedDict, total=False):
    """Runtime helper-call summary or static fallback extracted from a run script."""

    name: str
    args: str
    line: int | None
    source: str
    count: int
    outcomes: dict[str, int]
    total_duration_ms: float
    keyword_names: list[str]
    positional_counts: list[int]
    error_types: list[str]


class RunEventDetail(TypedDict):
    """Bounded run_events entry returned from thread_detail."""

    id: str
    event_id: int
    type: str
    summary: str
    raw_event: NotRequired[dict[str, Any]]


class ProcessDetail(TypedDict):
    """Detailed information for one process id, run id, event id, or turn id."""

    id: str
    kind: str
    status: str
    summary: str
    thread_id: str | None
    turn_id: str | None
    event_id: int | None
    event_ref: str | None
    run_id: NotRequired[str]
    returncode: NotRequired[int | None]
    timed_out: NotRequired[bool]
    interrupted: NotRequired[bool]
    code: NotRequired[BoundedText]
    stdout: NotRequired[BoundedText]
    stderr: NotRequired[BoundedText]
    helper_calls: NotRequired[list[HelperCall]]
    structured_events: NotRequired[list[dict[str, Any]]]
    structured_events_truncated: NotRequired[bool]
    events: NotRequired[list[RunEventDetail]]
    events_truncated: NotRequired[bool]
    output: NotRequired[BoundedText]
    related_ids: NotRequired[list[str]]
    raw_event: NotRequired[dict[str, Any]]


class ThreadDetailResult(TypedDict):
    """Batch detail result for ids and/or turn ids."""

    thread_id: str | None
    requested_ids: list[str]
    requested_turn_ids: list[str]
    details: list[ProcessDetail]
    missing: list[str]
    truncated: bool


# Event types that form the visible conversation. Everything else that is useful
# for debugging is exposed as a ProcessRef and can be expanded via thread_detail.
_CONVERSATION_EVENT_TYPES = {
    "item.user",
    "item.assistant",
    "item.assistant_partial",
    "item.model_response",
}
_PROCESS_REF_EVENT_TYPES = {
    "item.runner_result",
    "item.tool_call",
    "item.tool_output",
    "item.image_attachment",
    "item.reasoning_partial",
    "thread.token_estimation_warning",
    "thread.model_switch_warning",
    "turn.stream_retry",
    "turn.interrupted",
    "turn.error",
    "turn.retry",
}


def thread_view(
    thread_id: str,
    *,
    state_dir: str | Path | None = None,
    kind: str | None = None,
    epoch: EpochSelector = "latest",
    max_turns: int | None = None,
    max_text_chars: int = 12_000,
    max_item_chars: int = 4_000,
    max_process_refs: int = 500,
) -> ThreadView:
    """Return a conversation-first view of a stored thread.

    The view intentionally includes only user inputs and assistant/model outputs
    as text. Tool calls, run_python results, warnings, interruptions, and errors
    are returned as lightweight ids in ``process_refs``; pass those ids or a turn
    id to :func:`thread_detail` to inspect the process details.
    """

    base = _state_dir(state_dir)
    with _connect(base) as db:
        metadata = _read_metadata(db, thread_id, kind=kind)
        events = _read_events(db, thread_id)

    epochs, event_epoch_ids = _build_epochs(events)
    selected_indices = _select_epoch_indices(epoch, len(epochs))
    selected_epoch_ids = {epochs[index]["id"] for index in selected_indices}
    selected_events = [event for event in events if event_epoch_ids.get(_event_id(event)) in selected_epoch_ids]

    turns, truncated = _build_turns(
        selected_events,
        event_epoch_ids=event_epoch_ids,
        max_turns=max_turns,
        max_text_chars=max_text_chars,
        max_item_chars=max_item_chars,
        max_process_refs=max_process_refs,
    )

    return {
        "thread_id": thread_id,
        "kind": str(metadata.get("kind") or "thread"),
        "title": str(metadata.get("title") or "New thread"),
        "created_at": cast(str | None, metadata.get("created_at")),
        "updated_at": cast(str | None, metadata.get("updated_at")),
        "selected_epochs": [epochs[index]["id"] for index in selected_indices],
        "epochs": epochs,
        "turns": turns,
        "truncated": truncated,
    }


def thread_detail(
    *,
    state_dir: str | Path | None = None,
    thread_id: str | None = None,
    ids: str | Sequence[str] | None = None,
    turn_ids: str | Sequence[str] | None = None,
    max_code_chars: int = 4_000,
    max_output_chars: int = 4_000,
    max_events: int = 100,
    include_raw_events: bool = False,
) -> ThreadDetailResult:
    """Return details for process ids and/or all process events in turn ids.

    ``ids`` should normally use the prefixed ids returned by ``thread_view``:
    ``run:<run_id>`` for managed run_python executions, ``event:<event_id>`` for
    thread events, and ``turn:<turn_id>`` for one turn. ``turn_ids`` is a
    convenience for expanding one or more turns and requires ``thread_id``.
    """

    requested_ids = _normalize_string_sequence(ids)
    requested_turn_ids = _normalize_string_sequence(turn_ids)
    for ref in requested_ids:
        kind, value = _split_ref(ref)
        if kind == "turn" and value not in requested_turn_ids:
            requested_turn_ids.append(value)

    if not requested_ids and not requested_turn_ids:
        raise ValueError("thread_detail requires ids and/or turn_ids")
    if requested_turn_ids and not thread_id:
        raise ValueError("thread_detail requires thread_id when turn_ids or turn:<id> refs are used")

    base = _state_dir(state_dir)
    details: list[ProcessDetail] = []
    missing: list[str] = []
    seen_detail_ids: set[str] = set()

    with _connect(base) as db:
        for ref in requested_ids:
            ref_kind, value = _split_ref(ref)
            if ref_kind == "turn":
                # Expanded below through requested_turn_ids.
                continue
            detail = _detail_for_ref(
                db,
                ref_kind,
                value,
                thread_id=thread_id,
                max_code_chars=max_code_chars,
                max_output_chars=max_output_chars,
                max_events=max_events,
                include_raw_events=include_raw_events,
            )
            if detail is None:
                missing.append(ref)
                continue
            if detail["id"] not in seen_detail_ids:
                seen_detail_ids.add(detail["id"])
                details.append(detail)

        for turn_id in requested_turn_ids:
            assert thread_id is not None  # validated above
            events = _read_turn_events(db, thread_id, turn_id)
            if not events:
                missing.append(f"turn:{turn_id}")
                continue
            process_events = [event for event in events if _is_process_ref_event(event)]
            if max_events >= 0 and len(process_events) > max_events:
                process_events = process_events[:max_events]
            for event in process_events:
                detail = _detail_for_event(
                    db,
                    event,
                    thread_id=thread_id,
                    max_code_chars=max_code_chars,
                    max_output_chars=max_output_chars,
                    max_events=max_events,
                    include_raw_events=include_raw_events,
                )
                if detail["id"] not in seen_detail_ids:
                    seen_detail_ids.add(detail["id"])
                    details.append(detail)

    return {
        "thread_id": thread_id,
        "requested_ids": requested_ids,
        "requested_turn_ids": requested_turn_ids,
        "details": details,
        "missing": missing,
        "truncated": any(_process_detail_truncated(detail) for detail in details),
    }


def thread_digest(
    thread_id: str,
    *,
    state_dir: str | Path | None = None,
    kind: str | None = None,
    since_last_compaction: bool = True,
    include_tools: bool = False,
) -> ThreadDigest:
    """Return a compact legacy human/assistant digest for one stored thread."""

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
        "title": str(metadata.get("title") or "New thread"),
        "created_at": cast(str | None, metadata.get("created_at")),
        "updated_at": cast(str | None, metadata.get("updated_at")),
        "last_text": str(metadata.get("last_text") or ""),
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
) -> list[ThreadDigest]:
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
    connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    _ensure_readable_schema(connection)
    return connection


def _ensure_readable_schema(connection: sqlite3.Connection) -> None:
    row = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'runs'").fetchone()
    if row is None:
        return
    columns = {str(item["name"]) for item in connection.execute("PRAGMA table_info(runs)").fetchall()}
    missing = REQUIRED_RUN_COLUMNS - columns
    if missing:
        raise RuntimeError(
            "uv-agent state database is missing required run columns: "
            + ", ".join(sorted(missing))
        )


def _read_metadata(db: sqlite3.Connection, thread_id: str, *, kind: str | None) -> dict[str, Any]:
    if kind is None:
        row = db.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
    else:
        row = db.execute("SELECT * FROM threads WHERE thread_id = ? AND kind = ?", (thread_id, kind)).fetchone()
    if row is None:
        raise FileNotFoundError(f"Missing thread metadata for {thread_id}")
    return _metadata_from_row(row)


def _read_run_events(db: sqlite3.Connection, run_id: str, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rows = db.execute(
        """
        SELECT event_id, type, payload_json
        FROM run_events
        WHERE run_id = ?
        ORDER BY event_id ASC
        LIMIT ?
        """,
        (run_id, limit),
    ).fetchall()
    return [_event_from_row(row) for row in rows]


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
        SELECT event_id, type, payload_json
        FROM thread_events
        WHERE {' AND '.join(clauses)}
        ORDER BY event_id ASC
        """,
        params,
    ).fetchall()
    return [_event_from_row(row) for row in rows]


def _read_thread_event_by_id(
    db: sqlite3.Connection,
    event_id: int,
    *,
    thread_id: str | None,
) -> dict[str, Any] | None:
    clauses = ["event_id = ?"]
    params: list[Any] = [event_id]
    if thread_id is not None:
        clauses.append("thread_id = ?")
        params.append(thread_id)
    row = db.execute(
        f"SELECT event_id, type, payload_json FROM thread_events WHERE {' AND '.join(clauses)} LIMIT 1",
        params,
    ).fetchone()
    return _event_from_row(row) if row is not None else None


def _read_turn_events(db: sqlite3.Connection, thread_id: str, turn_id: str) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT event_id, type, payload_json
        FROM thread_events
        WHERE thread_id = ? AND turn_id = ?
        ORDER BY event_id ASC
        """,
        (thread_id, turn_id),
    ).fetchall()
    return [_event_from_row(row) for row in rows]


def _read_run(db: sqlite3.Connection, run_id: str, *, thread_id: str | None = None) -> dict[str, Any] | None:
    clauses = ["run_id = ?"]
    params: list[Any] = [run_id]
    if thread_id is not None:
        clauses.append("thread_id = ?")
        params.append(thread_id)
    row = db.execute(f"SELECT * FROM runs WHERE {' AND '.join(clauses)}", params).fetchone()
    return _run_from_row(row) if row is not None else None


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


def _run_from_row(row: sqlite3.Row) -> dict[str, Any]:
    run = dict(row)
    run["script_args"] = _json_loads(run.pop("script_args_json"), default=[])
    run["structured_events"] = _json_loads(run.pop("structured_events_json"), default=[])
    run["helper_calls"] = _json_loads(run.pop("helper_calls_json", None), default=None)
    for key in ("timed_out", "interrupted", "truncated"):
        run[key] = bool(run[key])
    return run


def _build_epochs(events: list[dict[str, Any]]) -> tuple[list[ThreadEpoch], dict[int, str]]:
    if not events:
        epoch: ThreadEpoch = {
            "id": "epoch:0",
            "index": 0,
            "start_event_id": 0,
            "end_event_id": 0,
            "compaction": None,
        }
        return [epoch], {}

    epochs: list[ThreadEpoch] = []
    event_epoch_ids: dict[int, str] = {}
    segment_start = _event_id(events[0])
    epoch_index = 0

    for event in events:
        event_id = _event_id(event)
        epoch_id = f"epoch:{epoch_index}"
        event_epoch_ids[event_id] = epoch_id
        if event.get("type") == "item.compaction":
            compaction: ThreadCompaction = {
                "id": f"event:{event_id}",
                "event_id": event_id,
                "turn_id": cast(str | None, event.get("turn_id")),
                "created_at": cast(str | None, event.get("created_at")),
                "text": str(event.get("text") or ""),
            }
            epochs.append(
                {
                    "id": epoch_id,
                    "index": epoch_index,
                    "start_event_id": segment_start,
                    "end_event_id": event_id,
                    "compaction": compaction,
                }
            )
            epoch_index += 1
            segment_start = event_id + 1

    last_event_id = _event_id(events[-1])
    if not epochs or segment_start <= last_event_id:
        epochs.append(
            {
                "id": f"epoch:{epoch_index}",
                "index": epoch_index,
                "start_event_id": segment_start,
                "end_event_id": last_event_id,
                "compaction": None,
            }
        )
        for event in events:
            event_id = _event_id(event)
            if event_id >= segment_start:
                event_epoch_ids[event_id] = f"epoch:{epoch_index}"

    return epochs, event_epoch_ids


def _select_epoch_indices(selector: EpochSelector, epoch_count: int) -> list[int]:
    if epoch_count <= 0:
        return []
    if selector == "latest":
        return [epoch_count - 1]
    if selector == "all":
        return list(range(epoch_count))
    raw_values: list[int | str]
    if isinstance(selector, int):
        raw_values = [selector]
    elif isinstance(selector, str):
        raw_values = [selector]
    else:
        raw_values = list(selector)
    indices: list[int] = []
    for raw in raw_values:
        if isinstance(raw, str) and raw.startswith("epoch:"):
            raw = raw.split(":", 1)[1]
        try:
            index = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid epoch selector {raw!r}; use 'latest', 'all', an int, or a list of ints") from exc
        if index < 0:
            index = epoch_count + index
        if index < 0 or index >= epoch_count:
            raise IndexError(f"Epoch index {index} is out of range for {epoch_count} epochs")
        if index not in indices:
            indices.append(index)
    return indices


def _build_turns(
    events: list[dict[str, Any]],
    *,
    event_epoch_ids: dict[int, str],
    max_turns: int | None,
    max_text_chars: int,
    max_item_chars: int,
    max_process_refs: int,
) -> tuple[list[ThreadTurn], bool]:
    order: list[str] = []
    turns: dict[str, ThreadTurn] = {}
    assistant_priorities: dict[str, int] = {}
    truncated = False
    process_ref_count = 0

    def ensure_turn(turn_id: str, event: dict[str, Any]) -> ThreadTurn:
        if turn_id not in turns:
            order.append(turn_id)
            epoch_id = event_epoch_ids.get(_event_id(event), "epoch:0")
            turns[turn_id] = {
                "id": f"turn:{turn_id}",
                "turn_id": turn_id,
                "epoch_id": epoch_id,
                "status": "unknown",
                "user_messages": [],
                "assistant_messages": [],
                "process_refs": [],
            }
        return turns[turn_id]

    for event in events:
        event_type = str(event.get("type") or "")
        if event_type not in _CONVERSATION_EVENT_TYPES and not _is_process_ref_event(event) and event_type != "turn.completed":
            continue
        turn_id = str(event.get("turn_id") or f"event:{_event_id(event)}")
        turn = ensure_turn(turn_id, event)
        if event_type == "turn.completed":
            turn["status"] = "completed"
            continue
        if event_type == "turn.interrupted":
            turn["status"] = "interrupted"
        elif event_type == "turn.error":
            turn["status"] = "error"

        if event_type == "item.user":
            text = _item_text(event.get("item") or {})
            if text:
                turn["user_messages"].append(_conversation_message(event, role="user", text=text, max_item_chars=max_item_chars))
        elif event_type in {"item.assistant", "item.assistant_partial"}:
            text = str(event.get("text") or "")
            if text:
                priority = 1 if event_type == "item.assistant_partial" else 2
                if priority >= assistant_priorities.get(turn_id, 0):
                    assistant_priorities[turn_id] = priority
                    turn["assistant_messages"] = [
                        _conversation_message(event, role="assistant", text=text, max_item_chars=max_item_chars)
                    ]
        elif event_type == "item.model_response":
            text = _model_response_text(event.get("output") or [])
            if text and 2 >= assistant_priorities.get(turn_id, 0):
                assistant_priorities[turn_id] = 2
                turn["assistant_messages"] = [
                    _conversation_message(event, role="assistant", text=text, max_item_chars=max_item_chars)
                ]
        elif _is_process_ref_event(event):
            ref = _process_ref_for_event(event)
            if ref is not None:
                if max_process_refs < 0 or process_ref_count < max_process_refs:
                    turn["process_refs"].append(ref)
                    process_ref_count += 1
                else:
                    truncated = True

    selected_turns = [turns[turn_id] for turn_id in order]
    if max_turns is not None and max_turns >= 0 and len(selected_turns) > max_turns:
        selected_turns = selected_turns[-max_turns:]
        truncated = True

    text_budget = max_text_chars
    for turn in selected_turns:
        for key in ("user_messages", "assistant_messages"):
            kept: list[ConversationMessage] = []
            for message in turn[key]:
                if text_budget == 0:
                    truncated = True
                    continue
                if text_budget > 0 and len(message["text"]) > text_budget:
                    original_chars = message["chars"]
                    text = message["text"][:text_budget]
                    kept.append(
                        {
                            **message,
                            "text": text,
                            "chars": original_chars,
                            "truncated": True,
                        }
                    )
                    text_budget = 0
                    truncated = True
                else:
                    kept.append(message)
                    if text_budget > 0:
                        text_budget -= len(message["text"])
            turn[key] = kept
    return selected_turns, truncated


def _conversation_message(
    event: dict[str, Any],
    *,
    role: Literal["user", "assistant"],
    text: str,
    max_item_chars: int,
) -> ConversationMessage:
    bounded = _bounded_head(text, max_item_chars)
    return {
        "id": f"event:{_event_id(event)}",
        "event_id": _event_id(event),
        "role": role,
        "text": bounded["text"],
        "chars": bounded["chars"],
        "truncated": bounded["truncated"],
    }


def _is_process_ref_event(event: dict[str, Any]) -> bool:
    return str(event.get("type") or "") in _PROCESS_REF_EVENT_TYPES


def _process_ref_for_event(event: dict[str, Any]) -> ProcessRef | None:
    event_type = str(event.get("type") or "")
    event_id = _event_id(event)
    event_ref = f"event:{event_id}"
    turn_id = str(event.get("turn_id") or "")
    kind = _process_kind(event)
    status = _event_status(event)
    summary = _event_summary(event)
    ref_id = event_ref
    related_ids: list[str] = []
    helper_names: list[str] = []

    if event_type == "item.runner_result":
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        run_id = str(result.get("run_id") or "")
        if run_id:
            ref_id = f"run:{run_id}"
            related_ids.append(event_ref)
        helper_calls = result.get("helper_calls") if isinstance(result, dict) else None
        if isinstance(helper_calls, list):
            helper_names = [str(call.get("name") or "helper") for call in helper_calls if isinstance(call, dict)]
    elif event_type == "item.tool_output":
        parsed = _tool_output_json(event)
        if isinstance(parsed, dict) and parsed.get("run_id"):
            related_ids.append(f"run:{parsed.get('run_id')}")

    ref: ProcessRef = {
        "id": ref_id,
        "kind": kind,
        "event_ref": event_ref,
        "event_id": event_id,
        "turn_id": turn_id,
        "status": status,
        "summary": summary,
    }
    if related_ids:
        ref["related_ids"] = related_ids
    if helper_names:
        ref["helper_names"] = helper_names
    return ref


def _detail_for_ref(
    db: sqlite3.Connection,
    ref_kind: str,
    value: str,
    *,
    thread_id: str | None,
    max_code_chars: int,
    max_output_chars: int,
    max_events: int,
    include_raw_events: bool,
) -> ProcessDetail | None:
    if ref_kind == "run":
        return _run_detail(
            db,
            value,
            thread_id=thread_id,
            event=None,
            max_code_chars=max_code_chars,
            max_output_chars=max_output_chars,
            max_events=max_events,
            include_raw_events=include_raw_events,
        )
    if ref_kind == "event":
        try:
            event_id = int(value)
        except ValueError:
            return None
        event = _read_thread_event_by_id(db, event_id, thread_id=thread_id)
        if event is None:
            return None
        return _detail_for_event(
            db,
            event,
            thread_id=thread_id,
            max_code_chars=max_code_chars,
            max_output_chars=max_output_chars,
            max_events=max_events,
            include_raw_events=include_raw_events,
        )
    # Treat unknown prefixed or bare ids as run ids first, then numeric event ids.
    run = _run_detail(
        db,
        value,
        thread_id=thread_id,
        event=None,
        max_code_chars=max_code_chars,
        max_output_chars=max_output_chars,
        max_events=max_events,
        include_raw_events=include_raw_events,
    )
    if run is not None:
        return run
    if value.isdigit():
        event = _read_thread_event_by_id(db, int(value), thread_id=thread_id)
        if event is not None:
            return _detail_for_event(
                db,
                event,
                thread_id=thread_id,
                max_code_chars=max_code_chars,
                max_output_chars=max_output_chars,
                max_events=max_events,
                include_raw_events=include_raw_events,
            )
    return None


def _detail_for_event(
    db: sqlite3.Connection,
    event: dict[str, Any],
    *,
    thread_id: str | None,
    max_code_chars: int,
    max_output_chars: int,
    max_events: int,
    include_raw_events: bool,
) -> ProcessDetail:
    event_type = str(event.get("type") or "")
    if event_type == "item.runner_result":
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        run_id = str(result.get("run_id") or "")
        if run_id:
            detail = _run_detail(
                db,
                run_id,
                thread_id=thread_id,
                event=event,
                max_code_chars=max_code_chars,
                max_output_chars=max_output_chars,
                max_events=max_events,
                include_raw_events=include_raw_events,
            )
            if detail is not None:
                return detail

    event_id = _event_id(event)
    detail: ProcessDetail = {
        "id": f"event:{event_id}",
        "kind": _process_kind(event),
        "status": _event_status(event),
        "summary": _event_summary(event),
        "thread_id": thread_id,
        "turn_id": cast(str | None, event.get("turn_id")),
        "event_id": event_id,
        "event_ref": f"event:{event_id}",
    }
    if event_type == "item.tool_output":
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        output = item.get("output") if isinstance(item, dict) else None
        if isinstance(output, str):
            detail["output"] = _bounded_tail(output, max_output_chars)
            parsed = _json_loads(output, default=None)
            if isinstance(parsed, dict) and parsed.get("run_id"):
                detail["related_ids"] = [f"run:{parsed.get('run_id')}"]
    if include_raw_events:
        detail["raw_event"] = event
    return detail


def _run_detail(
    db: sqlite3.Connection,
    run_id: str,
    *,
    thread_id: str | None,
    event: dict[str, Any] | None,
    max_code_chars: int,
    max_output_chars: int,
    max_events: int,
    include_raw_events: bool,
) -> ProcessDetail | None:
    run = _read_run(db, run_id, thread_id=thread_id)
    if run is None:
        return None

    event_id = _event_id(event) if event is not None else None
    helper_calls = _helper_calls_from_event(event) or _helper_calls_from_run_and_code(run)
    structured_events = run.get("structured_events") if isinstance(run.get("structured_events"), list) else []
    structured_events_truncated = max_events >= 0 and len(structured_events) > max_events
    if max_events >= 0:
        structured_events = structured_events[:max_events]
    run_events = _read_run_events(db, run_id, limit=max_events if max_events >= 0 else 10_000)
    events_truncated = False
    if max_events >= 0:
        # If we filled the whole limit, cheaply check whether there are more.
        count_row = db.execute("SELECT COUNT(*) AS count FROM run_events WHERE run_id = ?", (run_id,)).fetchone()
        events_truncated = count_row is not None and int(count_row["count"] or 0) > max_events

    detail: ProcessDetail = {
        "id": f"run:{run_id}",
        "kind": "run_python",
        "status": _run_status(run),
        "summary": _run_summary(run_id, run, helper_calls),
        "thread_id": cast(str | None, run.get("thread_id")),
        "turn_id": cast(str | None, run.get("turn_id")),
        "event_id": event_id,
        "event_ref": f"event:{event_id}" if event_id is not None else None,
        "run_id": run_id,
        "returncode": cast(int | None, run.get("returncode")),
        "timed_out": bool(run.get("timed_out")),
        "interrupted": bool(run.get("interrupted")),
        "code": _bounded_head(str(run.get("code") or ""), max_code_chars),
        "stdout": _bounded_tail(str(run.get("stdout") or ""), max_output_chars),
        "stderr": _bounded_tail(str(run.get("stderr") or ""), max_output_chars),
        "helper_calls": cast(list[HelperCall], helper_calls),
        "structured_events": cast(list[dict[str, Any]], structured_events),
        "structured_events_truncated": structured_events_truncated,
        "events": [_run_event_detail(run_event, include_raw_events=include_raw_events) for run_event in run_events],
        "events_truncated": events_truncated,
    }
    if include_raw_events and event is not None:
        detail["raw_event"] = event
    return detail


def _run_event_detail(event: dict[str, Any], *, include_raw_events: bool) -> RunEventDetail:
    event_id = _event_id(event)
    detail: RunEventDetail = {
        "id": f"run_event:{event_id}",
        "event_id": event_id,
        "type": str(event.get("type") or ""),
        "summary": _short_text(_event_human_text(event) or str(event.get("type") or "run_event"), 240),
    }
    if include_raw_events:
        detail["raw_event"] = event
    return detail


def _process_kind(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "")
    return {
        "item.runner_result": "run_python",
        "item.tool_call": "tool_call",
        "item.tool_output": "tool_output",
        "item.image_attachment": "image_attachment",
        "item.reasoning_partial": "reasoning",
        "thread.token_estimation_warning": "warning",
        "thread.model_switch_warning": "warning",
        "turn.stream_retry": "retry",
        "turn.interrupted": "interrupted",
        "turn.error": "error",
        "turn.retry": "retry",
    }.get(event_type, event_type or "event")


def _event_status(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "")
    if event_type == "turn.error":
        return "error"
    if event_type == "turn.interrupted":
        return "interrupted"
    if event_type == "item.runner_result":
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        if result.get("timed_out"):
            return "timed_out"
        if result.get("interrupted"):
            return "interrupted"
        rc = result.get("returncode")
        if rc is None:
            return "unknown"
        return "ok" if rc == 0 else "error"
    if event_type == "item.tool_output":
        parsed = _tool_output_json(event)
        if isinstance(parsed, dict):
            rc = parsed.get("returncode")
            if rc is not None:
                return "ok" if rc == 0 else "error"
    return "ok"


def _run_status(run: dict[str, Any]) -> str:
    if run.get("timed_out"):
        return "timed_out"
    if run.get("interrupted"):
        return "interrupted"
    rc = run.get("returncode")
    if rc is None:
        return "pending"
    return "ok" if rc == 0 else "error"


def _event_summary(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "")
    if event_type == "item.runner_result":
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        text = f"run_python rc={result.get('returncode')} run={result.get('run_id') or ''}".strip()
        helper_calls = result.get("helper_calls") if isinstance(result, dict) else None
        if isinstance(helper_calls, list) and helper_calls:
            names = [str(call.get("name") or "helper") for call in helper_calls if isinstance(call, dict)]
            text += " helpers=" + ", ".join(names[:5])
            if len(names) > 5:
                text += f", +{len(names) - 5} more"
        return _short_text(text, 240)
    if event_type == "item.tool_output":
        parsed = _tool_output_json(event)
        if isinstance(parsed, dict):
            parts = ["tool_output"]
            if parsed.get("run_id"):
                parts.append(f"run={parsed.get('run_id')}")
            if parsed.get("returncode") is not None:
                parts.append(f"rc={parsed.get('returncode')}")
            return " ".join(parts)
    if event_type == "turn.error":
        return _short_text(str(event.get("message") or event.get("error_type") or "turn error"), 240)
    if event_type == "turn.interrupted":
        return f"turn interrupted: {event.get('reason') or 'user_interrupt'}"
    return _short_text(_event_human_text(event) or event_type or "event", 240)


def _event_human_text(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "")
    if event_type == "item.user":
        return _item_text(event.get("item") or {})
    if event_type == "item.model_response":
        return _model_response_text(event.get("output") or [])
    if event_type in {"item.assistant", "item.assistant_partial", "item.compaction"}:
        return str(event.get("text") or "")
    if event_type == "turn.error":
        return str(event.get("message") or event.get("error_type") or "")
    if event_type == "turn.interrupted":
        return str(event.get("reason") or "")
    if event_type == "thread.model_switch_warning":
        return str(event.get("message") or "")
    return ""


def _run_summary(run_id: str, run: dict[str, Any], helper_calls: list[HelperCall]) -> str:
    text = f"run_python rc={run.get('returncode')} run={run_id}".strip()
    if helper_calls:
        names = [str(call.get("name") or "helper") for call in helper_calls]
        text += " helpers=" + ", ".join(names[:5])
        if len(names) > 5:
            text += f", +{len(names) - 5} more"
    return _short_text(text, 240)


def _tool_output_json(event: dict[str, Any]) -> Any:
    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    output = item.get("output") if isinstance(item, dict) else None
    if not isinstance(output, str) or not output:
        return None
    return _json_loads(output, default=None)


def _helper_calls_from_run_and_code(run: dict[str, Any]) -> list[HelperCall]:
    helper_calls = run.get("helper_calls")
    if isinstance(helper_calls, list):
        return [cast(HelperCall, call) for call in helper_calls if isinstance(call, dict)]
    return _extract_helper_calls(str(run.get("code") or ""))


def _helper_calls_from_run(run: dict[str, Any]) -> list[HelperCall]:
    helper_calls = run.get("helper_calls")
    if not isinstance(helper_calls, list):
        return []
    return [cast(HelperCall, call) for call in helper_calls if isinstance(call, dict)]


def _helper_calls_from_event(event: dict[str, Any] | None) -> list[HelperCall]:
    if event is None:
        return []
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    helper_calls = result.get("helper_calls") if isinstance(result, dict) else None
    if not isinstance(helper_calls, list):
        return []
    return [cast(HelperCall, call) for call in helper_calls if isinstance(call, dict)]


def _process_detail_truncated(detail: ProcessDetail) -> bool:
    for key in ("code", "stdout", "stderr", "output"):
        value = detail.get(key)
        if isinstance(value, dict) and value.get("truncated"):
            return True
    return bool(detail.get("structured_events_truncated") or detail.get("events_truncated"))


def _normalize_string_sequence(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _split_ref(ref: str) -> tuple[str, str]:
    if ":" in ref:
        prefix, value = ref.split(":", 1)
        return prefix, value
    if ref.isdigit():
        return "event", ref
    if ref.startswith("run_"):
        return "run", ref
    if ref.startswith("turn_"):
        return "turn", ref
    return "unknown", ref


def _digest_items(events: list[dict[str, Any]], *, include_tools: bool) -> list[ThreadDigestItem]:
    items: list[ThreadDigestItem] = []
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
            if text:
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
        text = f"run_python rc={result.get('returncode')} run={result.get('run_id') or ''}".strip()
        helper_calls = result.get("helper_calls") if isinstance(result, dict) else None
        if isinstance(helper_calls, list) and helper_calls:
            text += " helpers=" + ", ".join(_format_helper_call(call) for call in helper_calls[:5])
            if len(helper_calls) > 5:
                text += f", +{len(helper_calls) - 5} more"
        return text
    if event_type == "item.tool_output":
        item = event.get("item") or {}
        output = item.get("output") if isinstance(item, dict) else None
        if isinstance(output, str) and output:
            parsed = _json_loads(output, default=None)
            if isinstance(parsed, dict):
                run_id = parsed.get("run_id")
                rc = parsed.get("returncode")
                stdout = _short_text(str(parsed.get("stdout") or ""), 160)
                stderr = _short_text(str(parsed.get("stderr") or ""), 160)
                parts = ["tool_output"]
                if run_id:
                    parts.append(f"run={run_id}")
                if rc is not None:
                    parts.append(f"rc={rc}")
                if stdout:
                    parts.append(f"stdout={stdout!r}")
                if stderr:
                    parts.append(f"stderr={stderr!r}")
                return " ".join(parts)
            return f"tool_output {_short_text(output, 240)!r}"
    return event_type or "tool"


def _extract_helper_calls(code: str) -> list[HelperCall]:
    try:
        from uv_agent.helper_calls import extract_runtime_helper_calls
    except Exception:
        return []
    try:
        return cast(list[HelperCall], extract_runtime_helper_calls(code))
    except Exception:
        return []


def _format_helper_call(call: Any) -> str:
    if not isinstance(call, dict):
        return "helper()"
    name = str(call.get("name") or "helper")
    args = str(call.get("args") or "")
    text = f"{name}({args})" if args else f"{name}()"
    count = _positive_int(call.get("count")) or 1
    if count > 1:
        text = f"{text} x{count}"
    return _short_text(text, 160)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _short_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _bounded_head(text: str, max_chars: int) -> BoundedText:
    if max_chars <= 0:
        return {"text": "", "chars": len(text), "truncated": bool(text), "limit": max_chars}
    if len(text) <= max_chars:
        return {"text": text, "chars": len(text), "truncated": False, "limit": max_chars}
    return {
        "text": text[:max_chars] + f"\n…<truncated {len(text) - max_chars} chars>",
        "chars": len(text),
        "truncated": True,
        "limit": max_chars,
    }


def _bounded_tail(text: str, max_chars: int) -> BoundedText:
    if max_chars <= 0:
        return {"text": "", "chars": len(text), "truncated": bool(text), "limit": max_chars}
    if len(text) <= max_chars:
        return {"text": text, "chars": len(text), "truncated": False, "limit": max_chars}
    return {
        "text": f"…<truncated {len(text) - max_chars} chars>\n" + text[-max_chars:],
        "chars": len(text),
        "truncated": True,
        "limit": max_chars,
    }


def _truncate_head(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n…<truncated {len(text) - max_chars} chars>", True


def _truncate_tail(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    return f"…<truncated {len(text) - max_chars} chars>\n" + text[-max_chars:], True


def _compaction_summary(compaction: Any) -> ThreadCompactionSummary | None:
    if not isinstance(compaction, dict):
        return None
    return {
        "created_at": cast(str | None, compaction.get("created_at")),
        "turn_id": cast(str | None, compaction.get("turn_id")),
        "text": str(compaction.get("text") or ""),
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


def _event_id(event: dict[str, Any] | None) -> int:
    if not isinstance(event, dict):
        return 0
    value = event.get("_event_id")
    return value if isinstance(value, int) else 0
