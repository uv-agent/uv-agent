from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uv_agent.atomic import atomic_replace
from uv_agent.billing import decimal_or_none, decimal_or_zero, decimal_to_string, normalize_currency
from uv_agent.context import usage_token_count
from uv_agent.ids import new_id
from uv_agent.jsonl import (
    JsonlWriter,
    has_jsonl_event_before,
    latest_jsonl_event_before,
    read_jsonl,
    read_jsonl_after_latest_compaction,
    read_jsonl_range,
    read_jsonl_tail,
)
from uv_agent.time import utc_now_iso


VISIBLE_HISTORY_EVENT_TYPES = {
    "item.user",
    "item.model_response",
    "item.assistant",
    "item.assistant_partial",
    "item.runner_result",
    "item.image_attachment",
    "item.reasoning_delta",
    "item.reasoning_partial",
    "item.compaction",
    "thread.token_estimation_warning",
    "thread.model_switch_warning",
    "turn.stream_retry",
    "turn.interrupted",
    "turn.error",
    "turn.retry",
}


@dataclass(frozen=True)
class ThreadSnapshot:
    events_after_compaction: list[dict[str, Any]]
    latest_compaction: dict[str, Any] | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ThreadHistorySegment:
    events: list[dict[str, Any]]
    start_offset: int
    end_offset: int
    has_more: bool


class ThreadLockedError(RuntimeError):
    def __init__(self, thread_id: str, lock_path: Path, owner: dict[str, Any] | None = None) -> None:
        owner_text = ""
        if owner:
            pid = owner.get("pid")
            created_at = owner.get("created_at")
            owner_text = f" by pid {pid}" if pid else ""
            if created_at:
                owner_text += f" since {created_at}"
        super().__init__(f"Thread {thread_id} is locked{owner_text}: {lock_path}")
        self.thread_id = thread_id
        self.lock_path = lock_path
        self.owner = owner or {}


class ThreadStore:
    def __init__(
        self,
        data_dir: Path,
        *,
        threads_dir: Path | None = None,
        subthreads_dir: Path | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.threads_dir = threads_dir or data_dir / "threads"
        self.subthreads_dir = subthreads_dir or data_dir / "subthreads"
        self._lock_owner_id = new_id("owner")
        self._held_thread_locks: dict[str, str] = {}
        self._held_thread_lock_depth: dict[str, int] = {}
        self._history_segment_cache: dict[tuple[Any, ...], ThreadHistorySegment] = {}
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
            return self.subthreads_dir / f"{thread_id}.json"
        if kind == "thread":
            return self.threads_dir / f"{thread_id}.json"
        thread_path = self.threads_dir / f"{thread_id}.json"
        if thread_path.exists():
            return thread_path
        subthread_path = self.subthreads_dir / f"{thread_id}.json"
        if subthread_path.exists():
            return subthread_path
        return self.threads_dir / f"{thread_id}.json"

    def lock_path(self, thread_id: str, *, kind: str | None = None) -> Path:
        return self.path(thread_id, kind=kind).with_suffix(".lock")

    @contextmanager
    def lock_thread(self, thread_id: str, *, kind: str | None = None) -> Iterator[None]:
        resolved_kind = kind or self._kind_for_thread(thread_id)
        token = self._held_thread_locks.get(thread_id)
        if token is not None:
            self._held_thread_lock_depth[thread_id] = self._held_thread_lock_depth.get(thread_id, 1) + 1
            try:
                yield
            finally:
                self._release_thread_lock(thread_id, token=token, kind=resolved_kind)
            return

        token = new_id("lock")
        self._acquire_thread_lock(thread_id, token=token, kind=resolved_kind)
        try:
            yield
        finally:
            self._release_thread_lock(thread_id, token=token, kind=resolved_kind)

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

    def read_events(
        self,
        thread_id: str,
        *,
        event_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        metadata = self._read_metadata(thread_id)
        events = read_jsonl(self.path(thread_id, kind=metadata.get("kind")))
        if event_types is None:
            return events
        return [event for event in events if event.get("type") in event_types]

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

    def read_history_segment(
        self,
        thread_id: str,
        *,
        before_offset: int | None = None,
        event_types: set[str] | None = None,
    ) -> ThreadHistorySegment:
        metadata = self._read_metadata(thread_id)
        kind = str(metadata.get("kind") or "thread")
        path = self.path(thread_id, kind=kind)
        file_size = path.stat().st_size if path.exists() else 0
        event_type_key = tuple(sorted(event_types)) if event_types is not None else None
        cache_key = (thread_id, kind, before_offset, event_type_key, file_size)
        cached = self._history_segment_cache.get(cache_key)
        if cached is not None:
            return ThreadHistorySegment(
                events=list(cached.events),
                start_offset=cached.start_offset,
                end_offset=cached.end_offset,
                has_more=cached.has_more,
            )

        if before_offset is None:
            start_offset = _int_or_none(metadata.get("latest_compaction_offset")) or 0
            end_offset = file_size
        else:
            end_offset = max(0, min(before_offset, file_size))
            compaction = latest_jsonl_event_before(
                path,
                before_offset=end_offset,
                event_type="item.compaction",
            )
            start_offset = _event_offset(compaction) if compaction is not None else 0
        start_offset = max(0, min(start_offset, end_offset))
        events = read_jsonl_range(path, start_offset=start_offset, end_offset=end_offset)
        if event_types is not None:
            events = [event for event in events if event.get("type") in event_types]
        has_more = start_offset > 0 and has_jsonl_event_before(
            path,
            before_offset=start_offset,
            event_types=event_types,
        )
        segment = ThreadHistorySegment(
            events=events,
            start_offset=start_offset,
            end_offset=end_offset,
            has_more=has_more,
        )
        self._history_segment_cache[cache_key] = segment
        return ThreadHistorySegment(
            events=list(segment.events),
            start_offset=segment.start_offset,
            end_offset=segment.end_offset,
            has_more=segment.has_more,
        )

    def latest_event(
        self,
        thread_id: str,
        event_type: str,
    ) -> dict[str, Any] | None:
        events, _ = self.read_recent_events(thread_id, limit=1, event_types={event_type})
        return events[0] if events else None

    def list_threads(self) -> list[dict[str, Any]]:
        return self._list_from_metadata_dir(self.threads_dir)

    def list_subthreads(self, parent_thread_id: str | None = None) -> list[dict[str, Any]]:
        subthreads = self._list_from_metadata_dir(self.subthreads_dir)
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
            "active_level": metadata.get("active_level"),
            "active_model": metadata.get("active_model"),
            "latest_model_switch_warning": metadata.get("latest_model_switch_warning"),
            "billing_total": metadata.get("billing_total"),
            "billing_currency": metadata.get("billing_currency"),
            "billing_totals": metadata.get("billing_totals"),
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
        self._assert_thread_write_allowed(thread_id, kind=resolved_kind)
        stored = self.writer(thread_id, kind=resolved_kind).write(event)
        event_kind = resolved_kind or stored.get("kind")
        self._update_metadata(thread_id, stored, kind=str(event_kind or "thread"))
        self._history_segment_cache.clear()
        return stored

    def _kind_for_thread(self, thread_id: str) -> str:
        if (self.subthreads_dir / f"{thread_id}.jsonl").exists():
            return "subagent"
        return "thread"

    def _acquire_thread_lock(self, thread_id: str, *, token: str, kind: str) -> None:
        path = self.lock_path(thread_id, kind=kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "thread_id": thread_id,
            "kind": kind,
            "owner_id": self._lock_owner_id,
            "token": token,
            "pid": os.getpid(),
            "created_at": utc_now_iso(),
        }
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except FileExistsError as exc:
            raise ThreadLockedError(thread_id, path, self._read_lock_owner(path)) from exc
        self._held_thread_locks[thread_id] = token
        self._held_thread_lock_depth[thread_id] = 1

    def _release_thread_lock(self, thread_id: str, *, token: str, kind: str) -> None:
        depth = self._held_thread_lock_depth.get(thread_id, 0)
        if depth > 1:
            self._held_thread_lock_depth[thread_id] = depth - 1
            return
        self._held_thread_lock_depth.pop(thread_id, None)
        self._held_thread_locks.pop(thread_id, None)
        path = self.lock_path(thread_id, kind=kind)
        owner = self._read_lock_owner(path)
        if owner.get("owner_id") != self._lock_owner_id or owner.get("token") != token:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _assert_thread_write_allowed(self, thread_id: str, *, kind: str) -> None:
        path = self.lock_path(thread_id, kind=kind)
        if not path.exists():
            return
        owner = self._read_lock_owner(path)
        token = self._held_thread_locks.get(thread_id)
        if owner.get("owner_id") == self._lock_owner_id and owner.get("token") == token:
            return
        raise ThreadLockedError(thread_id, path, owner)

    @staticmethod
    def _read_lock_owner(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

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
        atomic_replace(tmp_path, path)

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


def _event_offset(event: dict[str, Any]) -> int:
    return _int_or_none(event.get("_jsonl_offset")) or 0


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
        elif event_type == "turn.error":
            items.append(
                {
                    "role": "system",
                    "text": f"turn error: {event.get('message') or event.get('error_type') or 'unknown error'}",
                }
            )
        elif include_tools and event_type in {"item.runner_result", "item.tool_output"}:
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
        if content.get("type") in {"input_text", "output_text", "text", "refusal"}:
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
        for key in ("parent_thread_id", "parent_turn_id", "parent_run_id"):
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

    if event_type == "thread.level_updated":
        level = str(event.get("level") or "").strip()
        model = str(event.get("model") or "").strip()
        if level:
            metadata["active_level"] = level
        if model:
            metadata["active_model"] = model
        if event.get("previous_level"):
            metadata["previous_level"] = event.get("previous_level")
        if event.get("previous_model"):
            metadata["previous_model"] = event.get("previous_model")
        return

    if event_type == "thread.model_switch_warning":
        metadata["latest_model_switch_warning"] = {
            "created_at": event.get("created_at"),
            "from_level": event.get("from_level"),
            "to_level": event.get("to_level"),
            "from_model": event.get("from_model"),
            "to_model": event.get("to_model"),
            "message": event.get("message") or "",
            "_jsonl_offset": event.get("_jsonl_offset"),
        }
        return

    if event_type == "item.user":
        metadata["user_message_count"] = int(metadata.get("user_message_count") or 0) + 1

    if event_type == "turn.completed":
        metadata["turn_count"] = int(metadata.get("turn_count") or 0) + 1
    elif event_type in {"turn.interrupted", "turn.error"}:
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
    elif event_type == "thread.billing_accumulated":
        _apply_billing_event(metadata, event)


def _apply_billing_event(metadata: dict[str, Any], event: dict[str, Any]) -> None:
    amount = decimal_or_none(event.get("amount"))
    if amount is None:
        return
    currency = normalize_currency(str(event.get("currency") or metadata.get("billing_currency") or "USD"))
    totals = metadata.get("billing_totals")
    if not isinstance(totals, dict):
        totals = {}
    current = decimal_or_zero(totals.get(currency))
    total = current + amount
    totals[currency] = decimal_to_string(total)
    metadata["billing_totals"] = totals
    # Preserve the historical single-total fields for simple consumers while
    # keeping per-currency totals above so a config currency change is additive
    # instead of silently mixing USD and CNY.
    metadata["billing_total"] = decimal_to_string(total)
    metadata["billing_currency"] = currency


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
