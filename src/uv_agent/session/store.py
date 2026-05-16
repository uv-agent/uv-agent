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
        self.subthreads_dir = data_dir / "subthreads"
        self.threads_dir.mkdir(parents=True, exist_ok=True)
        self.subthreads_dir.mkdir(parents=True, exist_ok=True)

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
        self.writer(thread_id, kind=kind).write(created)
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
        return self._list_from_dir(self.threads_dir)

    def list_subthreads(self, parent_thread_id: str | None = None) -> list[dict[str, Any]]:
        subthreads = self._list_from_dir(self.subthreads_dir)
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
        events = self.read(thread_id)
        created = next((event for event in events if event.get("type") == "thread.created"), {})
        start_index = latest_compaction_index(events) if since_last_compaction else -1
        compaction = events[start_index] if start_index >= 0 else None
        items = digest_items(events[start_index + 1 :], include_tools=include_tools)
        return {
            "thread_id": thread_id,
            "title": created.get("title") or "New thread",
            "created_at": created.get("created_at"),
            "updated_at": events[-1].get("created_at") if events else None,
            "last_text": latest_thread_text(events),
            "turn_count": sum(1 for event in events if event.get("type") == "turn.completed"),
            "interrupted_turn_count": sum(1 for event in events if event.get("type") == "turn.interrupted"),
            "latest_compaction": None
            if compaction is None
            else {
                "created_at": compaction.get("created_at"),
                "turn_id": compaction.get("turn_id"),
                "text": compaction.get("text") or "",
            },
            "items": items,
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

    def _list_from_dir(self, directory: Path) -> list[dict[str, Any]]:
        threads: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.jsonl")):
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


def latest_compaction_index(events: list[dict[str, Any]]) -> int:
    for index in range(len(events) - 1, -1, -1):
        if events[index].get("type") == "item.compaction":
            return index
    return -1


def digest_items(events: list[dict[str, Any]], *, include_tools: bool = False) -> list[dict[str, Any]]:
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
            text = item_text(event.get("item") or {})
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
            text = model_response_text(event.get("output") or [])
            if text:
                completed_assistant_turns.add(turn_id)
                items.append({"role": "assistant", "text": text})
        elif event_type == "item.compaction":
            flush_assistant()
            text = str(event.get("text") or "")
            items.append({"role": "summary", "text": text})
        elif event_type == "turn.interrupted":
            flush_assistant()
            items.append(
                {
                    "role": "system",
                    "text": f"turn interrupted: {event.get('reason') or 'user_interrupt'}",
                }
            )
        elif include_tools and event_type in {"item.tool_call", "item.runner_result", "item.tool_output"}:
            flush_assistant()
            items.append({"role": "tool", "text": tool_event_text(event)})
    flush_assistant()
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
