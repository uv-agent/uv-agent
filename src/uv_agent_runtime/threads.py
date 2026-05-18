from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


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
    metadata = _read_metadata(base, thread_id, kind=kind)
    if since_last_compaction:
        events, compaction = _read_after_latest_compaction(
            _thread_path(base, thread_id, kind=metadata.get("kind") or kind),
            _int_or_none(metadata.get("latest_compaction_offset")),
        )
    else:
        events = _read_jsonl(_thread_path(base, thread_id, kind=metadata.get("kind") or kind))
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
    metadata_dir = base / ("subthread_metadata" if kind == "subagent" else "thread_metadata")
    summaries: list[dict[str, Any]] = []
    for path in metadata_dir.glob("*.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        if parent_thread_id is not None and metadata.get("parent_thread_id") != parent_thread_id:
            continue
        summaries.append(metadata)
    summaries.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return [
        thread_digest(
            str(summary["thread_id"]),
            state_dir=base,
            kind=kind,
            since_last_compaction=since_last_compaction,
            include_tools=include_tools,
        )
        for summary in summaries[:limit]
    ]


def _state_dir(state_dir: str | Path | None) -> Path:
    if state_dir is not None:
        return Path(state_dir).resolve()
    env = os.environ.get("UV_AGENT_RUNTIME_STATE_DIR")
    if not env:
        raise RuntimeError("UV_AGENT_RUNTIME_STATE_DIR is not set; pass state_dir explicitly")
    return Path(env).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def _read_after_latest_compaction(
    path: Path,
    compaction_offset: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if compaction_offset is None:
        return _read_jsonl(path), None
    events: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        handle.seek(compaction_offset)
        for line in handle:
            if line.strip():
                events.append(json.loads(line.decode("utf-8")))
    if not events:
        return [], None
    return events[1:], events[0]


def _metadata_path(base: Path, thread_id: str, *, kind: str | None) -> Path:
    if kind == "subagent":
        return base / "subthread_metadata" / f"{thread_id}.json"
    if kind == "thread":
        return base / "thread_metadata" / f"{thread_id}.json"
    thread_path = base / "thread_metadata" / f"{thread_id}.json"
    if thread_path.exists():
        return thread_path
    subthread_path = base / "subthread_metadata" / f"{thread_id}.json"
    if subthread_path.exists():
        return subthread_path
    return thread_path


def _read_metadata(base: Path, thread_id: str, *, kind: str | None) -> dict[str, Any]:
    path = _metadata_path(base, thread_id, kind=kind)
    if not path.exists():
        raise FileNotFoundError(f"Missing thread metadata for {thread_id}: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid thread metadata: {path}")
    return metadata


def _thread_path(base: Path, thread_id: str, *, kind: str | None) -> Path:
    if kind == "subagent":
        return base / "subthreads" / f"{thread_id}.jsonl"
    if kind == "thread":
        return base / "threads" / f"{thread_id}.jsonl"
    thread_path = base / "threads" / f"{thread_id}.jsonl"
    if thread_path.exists():
        return thread_path
    subthread_path = base / "subthreads" / f"{thread_id}.jsonl"
    if subthread_path.exists():
        return subthread_path
    return thread_path


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
        elif event_type == "turn.interrupted":
            items.append({"role": "system", "text": f"turn interrupted: {event.get('reason') or 'user_interrupt'}"})
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


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None
