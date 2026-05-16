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
    events = _read_jsonl(_thread_path(base, thread_id, kind=kind))
    created = next((event for event in events if event.get("type") == "thread.created"), {})
    start_index = _latest_compaction_index(events) if since_last_compaction else -1
    compaction = events[start_index] if start_index >= 0 else None
    return {
        "thread_id": thread_id,
        "title": created.get("title") or "New thread",
        "created_at": created.get("created_at"),
        "updated_at": events[-1].get("created_at") if events else None,
        "last_text": _latest_thread_text(events),
        "turn_count": sum(1 for event in events if event.get("type") == "turn.completed"),
        "interrupted_turn_count": sum(1 for event in events if event.get("type") == "turn.interrupted"),
        "latest_compaction": None
        if compaction is None
        else {
            "created_at": compaction.get("created_at"),
            "turn_id": compaction.get("turn_id"),
            "text": compaction.get("text") or "",
        },
        "items": _digest_items(events[start_index + 1 :], include_tools=include_tools),
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
    summaries: list[dict[str, Any]] = []
    directory = base / ("subthreads" if kind == "subagent" else "threads")
    for path in directory.glob("*.jsonl"):
        events = _read_jsonl(path)
        if not events:
            continue
        created = next((event for event in events if event.get("type") == "thread.created"), {})
        if parent_thread_id is not None and created.get("parent_thread_id") != parent_thread_id:
            continue
        summaries.append(
            {
                "thread_id": path.stem,
                "updated_at": events[-1].get("created_at") or "",
            }
        )
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
    env = os.environ.get("UV_AGENT_STATE_DIR")
    if not env:
        raise RuntimeError("UV_AGENT_STATE_DIR is not set; pass state_dir explicitly")
    return Path(env).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _latest_compaction_index(events: list[dict[str, Any]]) -> int:
    for index in range(len(events) - 1, -1, -1):
        if events[index].get("type") == "item.compaction":
            return index
    return -1


def _latest_thread_text(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "item.assistant":
            return str(event.get("text") or "")
        if event.get("type") == "item.user":
            return _item_text(event.get("item") or {})
    return ""


def _digest_items(events: list[dict[str, Any]], *, include_tools: bool) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    assistant_parts: list[str] = []
    assistant_delta_turn_id: str | None = None
    completed_assistant_turns: set[str] = set()

    def flush_assistant() -> None:
        nonlocal assistant_delta_turn_id
        if not assistant_parts:
            return
        text = "".join(assistant_parts).strip()
        assistant_parts.clear()
        turn_id = assistant_delta_turn_id
        assistant_delta_turn_id = None
        if turn_id and turn_id in completed_assistant_turns:
            return
        if text:
            items.append({"role": "assistant", "text": text})

    for event in events:
        event_type = event.get("type")
        turn_id = str(event.get("turn_id") or "")
        if event_type == "item.user":
            flush_assistant()
            text = _item_text(event.get("item") or {})
            if text:
                items.append({"role": "user", "text": text})
        elif event_type == "item.assistant_delta":
            if assistant_delta_turn_id and assistant_delta_turn_id != turn_id:
                flush_assistant()
            assistant_delta_turn_id = turn_id
            assistant_parts.append(str(event.get("text") or ""))
        elif event_type == "item.assistant":
            assistant_parts.clear()
            assistant_delta_turn_id = None
            if turn_id in completed_assistant_turns:
                continue
            completed_assistant_turns.add(turn_id)
            text = str(event.get("text") or "")
            if text:
                items.append({"role": "assistant", "text": text})
        elif event_type == "item.model_response":
            assistant_parts.clear()
            assistant_delta_turn_id = None
            if turn_id in completed_assistant_turns:
                continue
            text = _model_response_text(event.get("output") or [])
            if text:
                completed_assistant_turns.add(turn_id)
                items.append({"role": "assistant", "text": text})
        elif event_type == "turn.interrupted":
            flush_assistant()
            items.append({"role": "system", "text": f"turn interrupted: {event.get('reason') or 'user_interrupt'}"})
        elif include_tools and event_type in {"item.tool_call", "item.runner_result", "item.tool_output"}:
            flush_assistant()
            items.append({"role": "tool", "text": _tool_event_text(event)})
    flush_assistant()
    return items


def _item_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for content in item.get("content") or []:
        if content.get("type") in {"input_text", "output_text", "text"}:
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
    if event_type == "item.tool_call":
        item = event.get("item") or {}
        return f"{item.get('name') or 'tool'} called"
    if event_type == "item.runner_result":
        result = event.get("result") or {}
        return f"run_python rc={result.get('returncode')} run={result.get('run_id') or ''}".strip()
    return event_type or "tool"
