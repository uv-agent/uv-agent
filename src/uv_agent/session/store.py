from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uv_agent.context import usage_token_count
from uv_agent.ids import new_id
from uv_agent.jsonl import (
    JsonlWriter,
    read_jsonl,
    read_jsonl_after_latest_compaction,
    read_jsonl_tail,
)
from uv_agent.time import utc_now_iso


VISIBLE_HISTORY_EVENT_TYPES = {
    "item.user",
    "item.model_response",
    "item.assistant",
    "item.assistant_partial",
    "item.tool_call",
    "item.runner_result",
    "item.image_attachment",
    "item.reasoning_delta",
    "item.reasoning_partial",
    "item.compaction",
    "turn.interrupted",
}


@dataclass(frozen=True)
class ThreadSnapshot:
    events_after_compaction: list[dict[str, Any]]
    latest_compaction: dict[str, Any] | None
    metadata: dict[str, Any]


class ThreadStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.threads_dir = data_dir / "threads"
        self.subthreads_dir = data_dir / "subthreads"
        self.thread_metadata_dir = data_dir / "thread_metadata"
        self.subthread_metadata_dir = data_dir / "subthread_metadata"
        self.threads_dir.mkdir(parents=True, exist_ok=True)
        self.subthreads_dir.mkdir(parents=True, exist_ok=True)
        self.thread_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.subthread_metadata_dir.mkdir(parents=True, exist_ok=True)

    def create_thread(
        self,
        title: str = "New thread",
        *,
        kind: str = "thread",
        parent_thread_id: str | None = None,
        parent_turn_id: str | None = None,
        parent_run_id: str | None = None,
        parent_script_id: str | None = None,
    ) -> str:
        thread_id = new_id("thr")
        created = {
            "type": "thread.created",
            "created_at": utc_now_iso(),
            "thread_id": thread_id,
            "title": title,
            "kind": kind,
        }
        if parent_thread_id:
            created["parent_thread_id"] = parent_thread_id
        if parent_turn_id:
            created["parent_turn_id"] = parent_turn_id
        if parent_run_id:
            created["parent_run_id"] = parent_run_id
        if parent_script_id:
            created["parent_script_id"] = parent_script_id
        self._write_event(thread_id, created, kind=kind)
        return thread_id

    def writer(self, thread_id: str, *, kind: str | None = None) -> JsonlWriter:
        return JsonlWriter(self.path(thread_id, kind=kind))

    def path(self, thread_id: str, *, kind: str | None = None) -> Path:
        if kind == "subagent":
            return self.subthreads_dir / f"{thread_id}.jsonl"
        if kind == "thread":
            return self.threads_dir / f"{thread_id}.jsonl"
        thread_path = self.threads_dir / f"{thread_id}.jsonl"
        if thread_path.exists():
            return thread_path
        subthread_path = self.subthreads_dir / f"{thread_id}.jsonl"
        if subthread_path.exists():
            return subthread_path
        return self.threads_dir / f"{thread_id}.jsonl"

    def metadata_path(self, thread_id: str, *, kind: str | None = None) -> Path:
        if kind == "subagent":
            return self.subthread_metadata_dir / f"{thread_id}.json"
        if kind == "thread":
            return self.thread_metadata_dir / f"{thread_id}.json"
        thread_path = self.thread_metadata_dir / f"{thread_id}.json"
        if thread_path.exists():
            return thread_path
        subthread_path = self.subthread_metadata_dir / f"{thread_id}.json"
        if subthread_path.exists():
            return subthread_path
        return self.thread_metadata_dir / f"{thread_id}.json"

    def append(self, thread_id: str, event_type: str, **data: Any) -> dict[str, Any]:
        event = {
            "type": event_type,
            "created_at": utc_now_iso(),
            "thread_id": thread_id,
            **data,
        }
        return self._write_event(thread_id, event)

    def update_title(self, thread_id: str, title: str, *, source: str = "manual") -> None:
        self.append(thread_id, "thread.title_updated", title=title, source=source)

    def read(self, thread_id: str) -> list[dict[str, Any]]:
        return read_jsonl(self.path(thread_id))

    def snapshot(self, thread_id: str) -> ThreadSnapshot:
        metadata = self._read_metadata(thread_id)
        events, compaction = self.read_after_latest_compaction(thread_id, metadata=metadata)
        return ThreadSnapshot(events_after_compaction=events, latest_compaction=compaction, metadata=metadata)

    def read_after_latest_compaction(
        self,
        thread_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        metadata = metadata or self._read_metadata(thread_id)
        return read_jsonl_after_latest_compaction(
            self.path(thread_id, kind=metadata.get("kind")),
            _int_or_none(metadata.get("latest_compaction_offset")),
        )

    def read_recent_events(
        self,
        thread_id: str,
        *,
        limit: int,
        before_offset: int | None = None,
        event_types: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        metadata = self._read_metadata(thread_id)
        return read_jsonl_tail(
            self.path(thread_id, kind=metadata.get("kind")),
            limit=limit,
            before_offset=before_offset,
            event_types=event_types,
        )

    def latest_event(
        self,
        thread_id: str,
        event_type: str,
    ) -> dict[str, Any] | None:
        events, _ = self.read_recent_events(thread_id, limit=1, event_types={event_type})
        return events[0] if events else None

    def list_threads(self) -> list[dict[str, Any]]:
        return self._list_from_metadata_dir(self.thread_metadata_dir)

    def list_subthreads(self, parent_thread_id: str | None = None) -> list[dict[str, Any]]:
        subthreads = self._list_from_metadata_dir(self.subthread_metadata_dir)
        if parent_thread_id is not None:
            subthreads = [
                thread
                for thread in subthreads
                if thread.get("parent_thread_id") == parent_thread_id
            ]
        return subthreads

    def thread_digest(
        self,
        thread_id: str,
        *,
        since_last_compaction: bool = True,
        include_tools: bool = False,
    ) -> dict[str, Any]:
        metadata = self._read_metadata(thread_id)
        if since_last_compaction:
            events, compaction = self.read_after_latest_compaction(thread_id, metadata=metadata)
        else:
            events = self.read(thread_id)
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
            "items": digest_items(events, include_tools=include_tools),
        }

    def list_thread_digests(
        self,
        *,
        limit: int = 10,
        since_last_compaction: bool = True,
        include_tools: bool = False,
    ) -> list[dict[str, Any]]:
        return [
            self.thread_digest(
                str(thread["thread_id"]),
                since_last_compaction=since_last_compaction,
                include_tools=include_tools,
            )
            for thread in self.list_threads()[:limit]
        ]

    def _write_event(self, thread_id: str, event: dict[str, Any], *, kind: str | None = None) -> dict[str, Any]:
        resolved_kind = kind or self._kind_for_thread(thread_id)
        stored = self.writer(thread_id, kind=resolved_kind).write(event)
        event_kind = resolved_kind or stored.get("kind")
        self._update_metadata(thread_id, stored, kind=str(event_kind or "thread"))
        return stored

    def _kind_for_thread(self, thread_id: str) -> str:
        if (self.subthreads_dir / f"{thread_id}.jsonl").exists():
            return "subagent"
        return "thread"

    def _read_metadata(self, thread_id: str, *, kind: str | None = None) -> dict[str, Any]:
        path = self.metadata_path(thread_id, kind=kind)
        if not path.exists():
            raise FileNotFoundError(f"Missing thread metadata for {thread_id}: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid thread metadata: {path}")
        return data

    def _write_metadata(self, thread_id: str, metadata: dict[str, Any], *, kind: str | None = None) -> None:
        path = self.metadata_path(thread_id, kind=kind or str(metadata.get("kind") or "thread"))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)

    def _update_metadata(self, thread_id: str, event: dict[str, Any], *, kind: str) -> None:
        path = self.metadata_path(thread_id, kind=kind)
        if path.exists():
            metadata = json.loads(path.read_text(encoding="utf-8"))
        else:
            metadata = {
                "thread_id": thread_id,
                "kind": kind,
                "title": "New thread",
                "created_at": event.get("created_at"),
                "updated_at": event.get("created_at"),
                "turn_count": 0,
                "interrupted_turn_count": 0,
                "user_message_count": 0,
                "last_text": "",
            }
        _apply_metadata_event(metadata, event)
        self._write_metadata(thread_id, metadata, kind=kind)

    def _list_from_metadata_dir(self, directory: Path) -> list[dict[str, Any]]:
        threads: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                threads.append(data)
        return sorted(threads, key=lambda item: str(item.get("updated_at") or ""), reverse=True)


def latest_thread_title(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "thread.title_updated":
            title = str(event.get("title") or "").strip()
            if title:
                return title
    created = next((event for event in events if event.get("type") == "thread.created"), {})
    return str(created.get("title") or "").strip()


def latest_thread_text(events: list[dict[str, Any]]) -> str:
    """Return the most recent human-readable message snippet."""
    for event in reversed(events):
        text = event_human_text(event)
        if text:
            return text
    return ""


def latest_compaction_index(events: list[dict[str, Any]]) -> int:
    for index in range(len(events) - 1, -1, -1):
        if events[index].get("type") == "item.compaction":
            return index
    return -1


def digest_items(events: list[dict[str, Any]], *, include_tools: bool = False) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    completed_assistant_turns: set[str] = set()

    for event in events:
        event_type = event.get("type")
        turn_id = str(event.get("turn_id") or "")
        if event_type == "item.user":
            text = item_text(event.get("item") or {})
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
            text = model_response_text(event.get("output") or [])
            if text:
                completed_assistant_turns.add(turn_id)
                items.append({"role": "assistant", "text": text})
        elif event_type == "item.compaction":
            text = str(event.get("text") or "")
            items.append({"role": "summary", "text": text})
        elif event_type == "turn.interrupted":
            items.append(
                {
                    "role": "system",
                    "text": f"turn interrupted: {event.get('reason') or 'user_interrupt'}",
                }
            )
        elif include_tools and event_type in {"item.tool_call", "item.runner_result", "item.tool_output"}:
            items.append({"role": "tool", "text": tool_event_text(event)})
    return items


def model_response_text(output: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in output:
        if item.get("type") == "message":
            parts.append(item_text(item))
    return "\n".join(part for part in parts if part)


def tool_event_text(event: dict[str, Any]) -> str:
    event_type = event.get("type")
    if event_type == "item.tool_call":
        item = event.get("item") or {}
        return f"{item.get('name') or 'tool'} called"
    if event_type == "item.runner_result":
        result = event.get("result") or {}
        return (
            f"run_python rc={result.get('returncode')} "
            f"run={result.get('run_id') or ''}".strip()
        )
    return event_type or "tool"


def item_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for content in item.get("content") or []:
        if content.get("type") in {"input_text", "output_text", "text"}:
            parts.append(str(content.get("text") or ""))
    return "\n".join(parts)


def event_human_text(event: dict[str, Any]) -> str:
    event_type = event.get("type")
    if event_type in {"item.assistant", "item.assistant_partial"}:
        return str(event.get("text") or "")
    if event_type == "item.model_response":
        return model_response_text(event.get("output") or [])
    if event_type == "item.user":
        return item_text(event.get("item") or {})
    return ""


def _apply_metadata_event(metadata: dict[str, Any], event: dict[str, Any]) -> None:
    event_type = event.get("type")
    created_at = event.get("created_at")
    if created_at:
        metadata["updated_at"] = created_at
    metadata["last_event_offset"] = event.get("_jsonl_offset")

    if event_type == "thread.created":
        metadata.update(
            {
                "thread_id": event.get("thread_id"),
                "title": event.get("title") or "New thread",
                "created_at": event.get("created_at"),
                "kind": event.get("kind") or metadata.get("kind") or "thread",
            }
        )
        for key in ("parent_thread_id", "parent_turn_id", "parent_run_id", "parent_script_id"):
            if event.get(key):
                metadata[key] = event[key]
        return

    if event_type == "thread.title_updated":
        title = str(event.get("title") or "").strip()
        if title:
            metadata["title"] = title
            metadata["title_updated_at"] = event.get("created_at")
        return

    if event_type == "thread.cwd_updated":
        cwd = str(event.get("cwd") or "").strip()
        if cwd:
            metadata["latest_cwd"] = cwd
        return

    if event_type == "item.user":
        metadata["user_message_count"] = int(metadata.get("user_message_count") or 0) + 1

    if event_type == "turn.completed":
        metadata["turn_count"] = int(metadata.get("turn_count") or 0) + 1
    elif event_type == "turn.interrupted":
        metadata["interrupted_turn_count"] = int(metadata.get("interrupted_turn_count") or 0) + 1
    elif event_type == "item.compaction":
        metadata["latest_compaction_offset"] = event.get("_jsonl_offset")
        metadata["latest_compaction"] = {
            "created_at": event.get("created_at"),
            "turn_id": event.get("turn_id"),
            "text": event.get("text") or "",
            "_jsonl_offset": event.get("_jsonl_offset"),
        }

    text = event_human_text(event)
    if text:
        metadata["last_text"] = text

    if event_type == "item.model_response":
        usage = usage_token_count(event.get("usage") or {})
        if usage is not None:
            metadata["latest_usage_tokens"] = usage
    elif event_type == "item.compaction":
        usage = usage_token_count(event.get("usage") or {})
        if usage is not None:
            metadata["latest_usage_tokens"] = usage


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
