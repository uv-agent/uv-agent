from __future__ import annotations

from pathlib import Path
from typing import Any

from uv_agent.ids import new_id
from uv_agent.jsonl import JsonlWriter, read_jsonl
from uv_agent.time import utc_now_iso


class ThreadStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.threads_dir = data_dir / "threads"
        self.threads_dir.mkdir(parents=True, exist_ok=True)

    def create_thread(self, title: str = "New thread") -> str:
        thread_id = new_id("thr")
        self.writer(thread_id).write(
            {
                "type": "thread.created",
                "created_at": utc_now_iso(),
                "thread_id": thread_id,
                "title": title,
            }
        )
        return thread_id

    def writer(self, thread_id: str) -> JsonlWriter:
        return JsonlWriter(self.path(thread_id))

    def path(self, thread_id: str) -> Path:
        return self.threads_dir / f"{thread_id}.jsonl"

    def append(self, thread_id: str, event_type: str, **data: Any) -> None:
        self.writer(thread_id).write(
            {
                "type": event_type,
                "created_at": utc_now_iso(),
                "thread_id": thread_id,
                **data,
            }
        )

    def read(self, thread_id: str) -> list[dict[str, Any]]:
        return read_jsonl(self.path(thread_id))

    def list_threads(self) -> list[dict[str, Any]]:
        threads: list[dict[str, Any]] = []
        for path in sorted(self.threads_dir.glob("*.jsonl")):
            events = read_jsonl(path)
            created = next((event for event in events if event.get("type") == "thread.created"), None)
            if created:
                summary = dict(created)
                summary["updated_at"] = events[-1].get("created_at", created.get("created_at"))
                summary["turn_count"] = sum(1 for event in events if event.get("type") == "turn.completed")
                summary["last_text"] = latest_thread_text(events)
                threads.append(summary)
        return sorted(threads, key=lambda item: str(item.get("updated_at") or ""), reverse=True)


def latest_thread_text(events: list[dict[str, Any]]) -> str:
    """Return the most recent human-readable message snippet."""
    for event in reversed(events):
        if event.get("type") == "item.assistant":
            return str(event.get("text") or "")
        if event.get("type") == "item.user":
            return item_text(event.get("item") or {})
    return ""


def item_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for content in item.get("content") or []:
        if content.get("type") in {"input_text", "output_text", "text"}:
            parts.append(str(content.get("text") or ""))
    return "\n".join(parts)
