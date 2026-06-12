from __future__ import annotations

import contextvars
import json
import os
import sqlite3
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uv_agent.billing import decimal_or_none, decimal_or_zero, decimal_to_string, normalize_currency
from uv_agent.context import usage_token_count
from uv_agent.ids import new_id
from uv_agent.state_db import connect_state_db, state_db_path
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

_METADATA_COLUMNS = {
    "thread_id",
    "kind",
    "title",
    "created_at",
    "updated_at",
    "parent_thread_id",
    "parent_turn_id",
    "parent_run_id",
    "active_level",
    "active_model",
    "latest_cwd",
    "turn_count",
    "interrupted_turn_count",
    "user_message_count",
    "last_text",
    "last_event_id",
    "latest_compaction_event_id",
    "latest_usage_tokens",
    "latest_model_switch_warning",
    "latest_compaction",
    "billing_currency",
    "billing_total",
    "billing_totals",
}
_THREAD_UPDATE_COLUMNS = {
    "kind",
    "title",
    "created_at",
    "updated_at",
    "parent_thread_id",
    "parent_turn_id",
    "parent_run_id",
    "active_level",
    "active_model",
    "latest_cwd",
    "turn_count",
    "interrupted_turn_count",
    "user_message_count",
    "last_text",
    "last_event_id",
    "latest_compaction_event_id",
    "latest_usage_tokens",
    "latest_model_switch_warning_json",
    "latest_compaction_json",
    "billing_currency",
    "billing_total",
    "billing_totals_json",
    "metadata_json",
}
_THREAD_LOCK_CONTEXT: contextvars.ContextVar[dict[tuple[str, str], tuple[str, int]]] = contextvars.ContextVar(
    "uv_agent_thread_lock_context",
    default={},
)
HISTORY_SEGMENT_CACHE_MAX_ENTRIES = 32


@dataclass(frozen=True)
class ThreadSnapshot:
    events_after_compaction: list[dict[str, Any]]
    latest_compaction: dict[str, Any] | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ThreadHistorySegment:
    events: list[dict[str, Any]]
    start_event_id: int
    end_event_id: int
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
        self.data_dir = data_dir.resolve()
        # threads_dir/subthreads_dir are accepted for older construction sites,
        # but SQLite is now the only source of truth for thread state.
        self.threads_dir = threads_dir or self.data_dir / "threads"
        self.subthreads_dir = subthreads_dir or self.data_dir / "subthreads"
        self.db_path = state_db_path(self.data_dir)
        self._lock_owner_id = new_id("owner")
        self._held_thread_locks: dict[str, str] = {}
        self._history_segment_cache: OrderedDict[tuple[Any, ...], ThreadHistorySegment] = OrderedDict()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._remove_empty_legacy_thread_dirs()
        with self._connect():
            pass

    def _remove_empty_legacy_thread_dirs(self) -> None:
        """Remove obsolete empty JSONL directories left by pre-SQLite stores."""

        for path in (self.threads_dir, self.subthreads_dir):
            try:
                path.rmdir()
            except FileNotFoundError:
                pass
            except OSError:
                # Preserve non-empty directories so legacy JSONL history is not
                # discarded implicitly during startup.
                pass

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

    def lock_path(self, thread_id: str, *, kind: str | None = None) -> Path:
        """Return a descriptive pseudo-path for SQLite-backed lock errors."""

        return self.db_path

    @contextmanager
    def lock_thread(self, thread_id: str, *, kind: str | None = None) -> Iterator[None]:
        resolved_kind = kind or self._kind_for_thread(thread_id)
        entry = self._context_lock_entry(thread_id)
        if entry is not None:
            token, depth = entry
            if self._held_thread_locks.get(thread_id) == token:
                reset_token = self._set_context_lock_depth(thread_id, token=token, depth=depth + 1)
                try:
                    yield
                finally:
                    _THREAD_LOCK_CONTEXT.reset(reset_token)
                return

        token = new_id("lock")
        self._acquire_thread_lock(thread_id, token=token, kind=resolved_kind)
        reset_token = self._set_context_lock_depth(thread_id, token=token, depth=1)
        try:
            yield
        finally:
            _THREAD_LOCK_CONTEXT.reset(reset_token)
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
        with self._connect() as db:
            return self._read_events_db(db, thread_id)

    def read_events(
        self,
        thread_id: str,
        *,
        event_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        with self._connect() as db:
            return self._read_events_db(db, thread_id, event_types=event_types)

    def thread_metadata(self, thread_id: str, *, kind: str | None = None) -> dict[str, Any]:
        """Return thread metadata without loading the event suffix."""

        return self._read_metadata(thread_id, kind=kind)

    def update_thread_metadata(
        self,
        thread_id: str,
        *,
        updates: dict[str, Any] | None = None,
        remover: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Merge extra metadata into an existing thread row.

        ``updates`` is shallow-merged into the persisted extra metadata blob.
        For more complex edits callers may pass ``remover``, which receives the
        current extra metadata dict and can mutate it in place.
        """

        with self._connect() as db:
            metadata = self._metadata_for_update(
                db,
                thread_id,
                kind=self._kind_for_thread(thread_id),
                event={"created_at": utc_now_iso()},
            )
            extra = {key: value for key, value in metadata.items() if key not in _METADATA_COLUMNS}
            if updates:
                extra.update(updates)
            if remover is not None:
                remover(extra)
            for key in list(metadata):
                if key not in _METADATA_COLUMNS:
                    del metadata[key]
            metadata.update(extra)
            self._upsert_metadata(db, metadata)

    def snapshot(self, thread_id: str) -> ThreadSnapshot:
        metadata = self._read_metadata(thread_id)
        events, compaction = self.read_after_latest_compaction(thread_id, metadata=metadata)
        return ThreadSnapshot(events_after_compaction=events, latest_compaction=compaction, metadata=metadata)

    def read_after_latest_compaction(
        self,
        thread_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        event_types: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        metadata = metadata or self._read_metadata(thread_id)
        compaction_event_id = _int_or_none(metadata.get("latest_compaction_event_id"))
        with self._connect() as db:
            if compaction_event_id is None:
                return self._read_events_db(db, thread_id, event_types=event_types), None
            compaction_row = db.execute(
                """
                SELECT event_id, payload_json
                FROM thread_events
                WHERE thread_id = ? AND event_id = ?
                LIMIT 1
                """,
                (thread_id, compaction_event_id),
            ).fetchone()
            events = self._read_events_db(
                db,
                thread_id,
                event_types=event_types,
                event_id_gt=compaction_event_id,
            )
        compaction = _event_from_row(compaction_row) if compaction_row is not None else None
        return events, compaction

    def read_recent_events(
        self,
        thread_id: str,
        *,
        limit: int,
        before_event_id: int | None = None,
        event_types: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        if limit <= 0:
            return [], False
        with self._connect() as db:
            clauses = ["thread_id = ?"]
            params: list[Any] = [thread_id]
            if before_event_id is not None:
                clauses.append("event_id < ?")
                params.append(before_event_id)
            if event_types:
                clauses.append(f"type IN ({_placeholders(event_types)})")
                params.extend(sorted(event_types))
            rows = db.execute(
                f"""
                SELECT event_id, payload_json
                FROM thread_events
                WHERE {' AND '.join(clauses)}
                ORDER BY event_id DESC
                LIMIT ?
                """,
                (*params, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        selected = rows[:limit]
        events = [_event_from_row(row) for row in reversed(selected)]
        return events, has_more

    def read_history_segment(
        self,
        thread_id: str,
        *,
        before_event_id: int | None = None,
        event_types: set[str] | None = None,
    ) -> ThreadHistorySegment:
        metadata = self._read_metadata(thread_id)
        kind = str(metadata.get("kind") or "thread")
        event_type_key = tuple(sorted(event_types)) if event_types is not None else None
        last_event_id = int(metadata.get("last_event_id") or 0)
        cache_key = (thread_id, kind, before_event_id, event_type_key, last_event_id)
        cached = self._history_segment_cache.get(cache_key)
        if cached is not None:
            self._history_segment_cache.move_to_end(cache_key)
            return ThreadHistorySegment(
                events=list(cached.events),
                start_event_id=cached.start_event_id,
                end_event_id=cached.end_event_id,
                has_more=cached.has_more,
            )

        with self._connect() as db:
            if before_event_id is None:
                start_event_id = _int_or_none(metadata.get("latest_compaction_event_id")) or 0
                end_event_id = last_event_id + 1
            else:
                end_event_id = max(0, before_event_id)
                compaction = db.execute(
                    """
                    SELECT event_id, payload_json
                    FROM thread_events
                    WHERE thread_id = ? AND type = 'item.compaction' AND event_id < ?
                    ORDER BY event_id DESC
                    LIMIT 1
                    """,
                    (thread_id, end_event_id),
                ).fetchone()
                start_event_id = int(compaction["event_id"]) if compaction is not None else 0

            rows, has_more = self._history_rows_between(
                db,
                thread_id,
                start_event_id=start_event_id,
                end_event_id=end_event_id,
                event_types=event_types,
            )
        events = [_event_from_row(row) for row in rows]
        segment = ThreadHistorySegment(
            events=events,
            start_event_id=start_event_id,
            end_event_id=end_event_id,
            has_more=has_more,
        )
        self._history_segment_cache[cache_key] = segment
        self._history_segment_cache.move_to_end(cache_key)
        while len(self._history_segment_cache) > HISTORY_SEGMENT_CACHE_MAX_ENTRIES:
            self._history_segment_cache.popitem(last=False)
        return ThreadHistorySegment(
            events=list(segment.events),
            start_event_id=segment.start_event_id,
            end_event_id=segment.end_event_id,
            has_more=segment.has_more,
        )

    def latest_event(self, thread_id: str, event_type: str) -> dict[str, Any] | None:
        events, _ = self.read_recent_events(thread_id, limit=1, event_types={event_type})
        return events[0] if events else None

    def latest_event_after_latest_compaction(
        self,
        thread_id: str,
        *,
        event_types: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        metadata = metadata or self._read_metadata(thread_id)
        compaction_event_id = _int_or_none(metadata.get("latest_compaction_event_id")) or 0
        with self._connect() as db:
            clauses = ["thread_id = ?", "event_id > ?"]
            params: list[Any] = [thread_id, compaction_event_id]
            if event_types:
                clauses.append(f"type IN ({_placeholders(event_types)})")
                params.extend(sorted(event_types))
            row = db.execute(
                f"""
                SELECT event_id, payload_json
                FROM thread_events
                WHERE {' AND '.join(clauses)}
                ORDER BY event_id DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        return _event_from_row(row) if row is not None else None

    def has_event_after_latest_compaction(
        self,
        thread_id: str,
        *,
        event_types: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        metadata = metadata or self._read_metadata(thread_id)
        compaction_event_id = _int_or_none(metadata.get("latest_compaction_event_id")) or 0
        with self._connect() as db:
            clauses = ["thread_id = ?", "event_id > ?"]
            params: list[Any] = [thread_id, compaction_event_id]
            if event_types:
                clauses.append(f"type IN ({_placeholders(event_types)})")
                params.extend(sorted(event_types))
            row = db.execute(
                f"SELECT 1 FROM thread_events WHERE {' AND '.join(clauses)} LIMIT 1",
                params,
            ).fetchone()
        return row is not None

    def list_threads(self) -> list[dict[str, Any]]:
        return self._list_threads(kind="thread")

    def list_subthreads(self, parent_thread_id: str | None = None) -> list[dict[str, Any]]:
        return self._list_threads(kind="subagent", parent_thread_id=parent_thread_id)

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
        digest = {
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
            "goal_mode": metadata.get("goal_mode"),
            "items": digest_items(events, include_tools=include_tools),
        }
        for key in (
            "latest_cwd",
            "worktree_status",
            "worktree_branch",
            "worktree_path",
            "worktree_base_ref",
            "worktree_origin_root",
            "worktree_head",
            "worktree_created_at",
            "worktree_merge_prompted_at",
            "worktree_deleted_at",
            "worktree_deleted_head",
            "worktree_deleted_status",
            "agent_view_joined",
            "agent_view_joined_at",
            "agent_view_source",
            "agent_view_deleted",
            "agent_view_deleted_at",
        ):
            if metadata.get(key) is not None:
                digest[key] = metadata.get(key)
        return digest

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

    def _connect(self) -> sqlite3.Connection:
        return connect_state_db(self.data_dir)

    def _write_event(self, thread_id: str, event: dict[str, Any], *, kind: str | None = None) -> dict[str, Any]:
        resolved_kind = kind or self._kind_for_thread(thread_id)
        self._assert_thread_write_allowed(thread_id, kind=resolved_kind)
        with self._connect() as db:
            metadata = self._metadata_for_update(db, thread_id, kind=resolved_kind, event=event)
            # Thread rows must exist before thread_events can satisfy the foreign
            # key; the final metadata update below records the assigned event_id.
            self._upsert_metadata(db, metadata)
            cursor = db.execute(
                """
                INSERT INTO thread_events(thread_id, turn_id, type, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    event.get("turn_id"),
                    event.get("type"),
                    event.get("created_at") or utc_now_iso(),
                    _json_dumps(event),
                ),
            )
            event_id = int(cursor.lastrowid)
            stored = {**event, "_event_id": event_id}
            # Persist the full event including its assigned event_id so payloads
            # remain self-contained for debugging and runtime helper reads.
            db.execute(
                "UPDATE thread_events SET payload_json = ? WHERE event_id = ?",
                (_json_dumps(stored), event_id),
            )
            _apply_metadata_event(metadata, stored)
            self._upsert_metadata(db, metadata)
        self._history_segment_cache.clear()
        return stored

    def _kind_for_thread(self, thread_id: str) -> str:
        with self._connect() as db:
            row = db.execute("SELECT kind FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        if row is not None:
            return str(row["kind"] or "thread")
        return "thread"

    def _acquire_thread_lock(self, thread_id: str, *, token: str, kind: str) -> None:
        payload = {
            "thread_id": thread_id,
            "kind": kind,
            "owner_id": self._lock_owner_id,
            "token": token,
            "pid": os.getpid(),
            "created_at": utc_now_iso(),
        }
        try:
            with self._connect() as db:
                db.execute(
                    """
                    INSERT INTO thread_locks(thread_id, owner_id, token, pid, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        payload["owner_id"],
                        payload["token"],
                        payload["pid"],
                        payload["created_at"],
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ThreadLockedError(thread_id, self.lock_path(thread_id, kind=kind), self._read_lock_owner(thread_id)) from exc
        self._held_thread_locks[thread_id] = token

    def _release_thread_lock(self, thread_id: str, *, token: str, kind: str) -> None:
        self._held_thread_locks.pop(thread_id, None)
        with self._connect() as db:
            db.execute(
                "DELETE FROM thread_locks WHERE thread_id = ? AND owner_id = ? AND token = ?",
                (thread_id, self._lock_owner_id, token),
            )

    def _assert_thread_write_allowed(self, thread_id: str, *, kind: str) -> None:
        owner = self._read_lock_owner(thread_id)
        if not owner:
            return
        token = self._context_lock_token(thread_id)
        if owner.get("owner_id") == self._lock_owner_id and owner.get("token") == token:
            return
        raise ThreadLockedError(thread_id, self.lock_path(thread_id, kind=kind), owner)

    def _context_lock_key(self, thread_id: str) -> tuple[str, str]:
        return (self._lock_owner_id, thread_id)

    def _context_lock_entry(self, thread_id: str) -> tuple[str, int] | None:
        return _THREAD_LOCK_CONTEXT.get().get(self._context_lock_key(thread_id))

    def _context_lock_token(self, thread_id: str) -> str | None:
        entry = self._context_lock_entry(thread_id)
        if entry is None:
            return None
        token = entry[0]
        return token if self._held_thread_locks.get(thread_id) == token else None

    def _set_context_lock_depth(
        self,
        thread_id: str,
        *,
        token: str,
        depth: int,
    ) -> contextvars.Token[dict[tuple[str, str], tuple[str, int]]]:
        current = dict(_THREAD_LOCK_CONTEXT.get())
        current[self._context_lock_key(thread_id)] = (token, depth)
        return _THREAD_LOCK_CONTEXT.set(current)

    def _read_lock_owner(self, thread_id: str | Path) -> dict[str, Any]:
        # Accept Path for compatibility with tests or older callers that passed
        # lock_path objects to the private helper.
        if isinstance(thread_id, Path):
            return {}
        with self._connect() as db:
            row = db.execute(
                "SELECT thread_id, owner_id, token, pid, created_at FROM thread_locks WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return dict(row) if row is not None else {}

    def _read_metadata(self, thread_id: str, *, kind: str | None = None) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"Missing thread metadata for {thread_id}: {self.db_path}")
        return _metadata_from_row(row)

    def _metadata_for_update(
        self,
        db: sqlite3.Connection,
        thread_id: str,
        *,
        kind: str,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        row = db.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        if row is not None:
            return _metadata_from_row(row)
        return {
            "thread_id": thread_id,
            "kind": kind,
            "title": "New thread",
            "created_at": event.get("created_at") or utc_now_iso(),
            "updated_at": event.get("created_at") or utc_now_iso(),
            "turn_count": 0,
            "interrupted_turn_count": 0,
            "user_message_count": 0,
            "last_text": "",
        }

    def _upsert_metadata(self, db: sqlite3.Connection, metadata: dict[str, Any]) -> None:
        row = _metadata_to_row(metadata)
        columns = ["thread_id", *_THREAD_UPDATE_COLUMNS]
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{column}=excluded.{column}" for column in _THREAD_UPDATE_COLUMNS)
        db.execute(
            f"""
            INSERT INTO threads({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(thread_id) DO UPDATE SET {updates}
            """,
            tuple(row.get(column) for column in columns),
        )

    def _read_events_db(
        self,
        db: sqlite3.Connection,
        thread_id: str,
        *,
        event_types: set[str] | None = None,
        event_id_gte: int | None = None,
        event_id_gt: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["thread_id = ?"]
        params: list[Any] = [thread_id]
        if event_id_gte is not None:
            clauses.append("event_id >= ?")
            params.append(event_id_gte)
        if event_id_gt is not None:
            clauses.append("event_id > ?")
            params.append(event_id_gt)
        if event_types:
            clauses.append(f"type IN ({_placeholders(event_types)})")
            params.extend(sorted(event_types))
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

    def _history_rows_between(
        self,
        db: sqlite3.Connection,
        thread_id: str,
        *,
        start_event_id: int,
        end_event_id: int,
        event_types: set[str] | None,
    ) -> tuple[Sequence[sqlite3.Row], bool]:
        clauses = ["thread_id = ?", "event_id >= ?", "event_id < ?"]
        params: list[Any] = [thread_id, start_event_id, end_event_id]
        if event_types:
            clauses.append(f"type IN ({_placeholders(event_types)})")
            params.extend(sorted(event_types))
        query_tail = f"WHERE {' AND '.join(clauses)}"
        rows = db.execute(
            f"""
            SELECT event_id, payload_json
            FROM thread_events
            {query_tail}
            ORDER BY event_id ASC
            """,
            params,
        ).fetchall()
        has_more = False
        if start_event_id > 0:
            before_clauses = ["thread_id = ?", "event_id < ?"]
            before_params: list[Any] = [thread_id, start_event_id]
            if event_types:
                before_clauses.append(f"type IN ({_placeholders(event_types)})")
                before_params.extend(sorted(event_types))
            has_more = db.execute(
                f"SELECT 1 FROM thread_events WHERE {' AND '.join(before_clauses)} LIMIT 1",
                before_params,
            ).fetchone() is not None
        return rows, has_more

    def _list_threads(self, *, kind: str, parent_thread_id: str | None = None) -> list[dict[str, Any]]:
        clauses = ["kind = ?"]
        params: list[Any] = [kind]
        if parent_thread_id is not None:
            clauses.append("parent_thread_id = ?")
            params.append(parent_thread_id)
        with self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM threads WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC",
                params,
            ).fetchall()
        return [_metadata_from_row(row) for row in rows]


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


def _event_id(event: dict[str, Any]) -> int:
    return _int_or_none(event.get("_event_id")) or 0


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
            if text:
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
    metadata["last_event_id"] = event.get("_event_id")

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

    if event_type == "thread.worktree_created":
        for key in (
            "worktree_status",
            "worktree_branch",
            "worktree_path",
            "worktree_base_ref",
            "worktree_origin_root",
            "worktree_head",
            "worktree_created_at",
        ):
            value = str(event.get(key) or "").strip()
            if value:
                metadata[key] = value
        metadata["worktree_status"] = metadata.get("worktree_status") or "active"
        return

    if event_type == "thread.worktree_merge_prompted":
        metadata["worktree_merge_prompted_at"] = event.get("created_at")
        return

    if event_type == "thread.worktree_deleted":
        metadata["worktree_status"] = "deleted"
        for key in (
            "worktree_deleted_at",
            "worktree_deleted_head",
        ):
            value = str(event.get(key) or "").strip()
            if value:
                metadata[key] = value
        status = str(event.get("worktree_deleted_status") or "")
        if status:
            metadata["worktree_deleted_status"] = status
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
            "_event_id": event.get("_event_id"),
        }
        return

    if event_type == "thread.agent_view_joined":
        metadata["agent_view_joined"] = True
        metadata["agent_view_joined_at"] = event.get("created_at")
        source = str(event.get("source") or "").strip()
        if source:
            metadata["agent_view_source"] = source
        # Re-adding a thread from inside the thread should make a previous
        # Agent View hide reversible without affecting ordinary thread history.
        metadata.pop("agent_view_deleted", None)
        metadata.pop("agent_view_deleted_at", None)
        return

    if event_type == "thread.agent_view_deleted":
        metadata["agent_view_deleted"] = True
        metadata["agent_view_deleted_at"] = event.get("created_at")
        return

    if event_type == "thread.goal_mode_updated":
        enabled = bool(event.get("enabled"))
        previous = metadata.get("goal_mode")
        if not isinstance(previous, dict):
            previous = {}
        metadata["goal_mode"] = {
            "enabled": enabled,
            "status": "enabled" if enabled else "disabled",
            "updated_at": event.get("created_at"),
            "objective": event.get("objective") or previous.get("objective") or "",
            "files": event.get("files") or previous.get("files") or {},
            "_event_id": event.get("_event_id"),
        }
        return

    if event_type == "thread.goal_files_reset":
        current = metadata.get("goal_mode")
        if not isinstance(current, dict):
            current = {"enabled": False, "status": "disabled"}
        current = dict(current)
        current["enabled"] = False
        current["status"] = "disabled"
        current["objective"] = event.get("objective") or ""
        current["files"] = event.get("files") or current.get("files") or {}
        current["reset_at"] = event.get("created_at")
        current["updated_at"] = event.get("created_at")
        current["_reset_event_id"] = event.get("_event_id")
        metadata["goal_mode"] = current
        return

    if event_type == "item.user":
        metadata["user_message_count"] = int(metadata.get("user_message_count") or 0) + 1

    if event_type == "turn.completed":
        metadata["turn_count"] = int(metadata.get("turn_count") or 0) + 1
    elif event_type in {"turn.interrupted", "turn.error"}:
        metadata["interrupted_turn_count"] = int(metadata.get("interrupted_turn_count") or 0) + 1
    elif event_type == "item.compaction":
        metadata["latest_compaction_event_id"] = event.get("_event_id")
        metadata["latest_compaction"] = {
            "created_at": event.get("created_at"),
            "turn_id": event.get("turn_id"),
            "text": event.get("text") or "",
            "_event_id": event.get("_event_id"),
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
            "user_message_count": int(row["user_message_count"] or 0),
            "last_text": row["last_text"] or "",
        }
    )
    for key in (
        "parent_thread_id",
        "parent_turn_id",
        "parent_run_id",
        "active_level",
        "active_model",
        "latest_cwd",
        "last_event_id",
        "latest_compaction_event_id",
        "latest_usage_tokens",
        "billing_currency",
        "billing_total",
    ):
        if row[key] is not None:
            metadata[key] = row[key]
    metadata["latest_model_switch_warning"] = _json_loads(row["latest_model_switch_warning_json"], default=None)
    metadata["latest_compaction"] = _json_loads(row["latest_compaction_json"], default=None)
    metadata["billing_totals"] = _json_loads(row["billing_totals_json"], default=None)
    return {key: value for key, value in metadata.items() if value is not None}


def _metadata_to_row(metadata: dict[str, Any]) -> dict[str, Any]:
    extra = {key: value for key, value in metadata.items() if key not in _METADATA_COLUMNS}
    return {
        "thread_id": metadata["thread_id"],
        "kind": metadata.get("kind") or "thread",
        "title": metadata.get("title") or "New thread",
        "created_at": metadata.get("created_at") or utc_now_iso(),
        "updated_at": metadata.get("updated_at") or metadata.get("created_at") or utc_now_iso(),
        "parent_thread_id": metadata.get("parent_thread_id"),
        "parent_turn_id": metadata.get("parent_turn_id"),
        "parent_run_id": metadata.get("parent_run_id"),
        "active_level": metadata.get("active_level"),
        "active_model": metadata.get("active_model"),
        "latest_cwd": metadata.get("latest_cwd"),
        "turn_count": int(metadata.get("turn_count") or 0),
        "interrupted_turn_count": int(metadata.get("interrupted_turn_count") or 0),
        "user_message_count": int(metadata.get("user_message_count") or 0),
        "last_text": metadata.get("last_text") or "",
        "last_event_id": metadata.get("last_event_id"),
        "latest_compaction_event_id": metadata.get("latest_compaction_event_id"),
        "latest_usage_tokens": metadata.get("latest_usage_tokens"),
        "latest_model_switch_warning_json": _json_dumps(metadata.get("latest_model_switch_warning"))
        if metadata.get("latest_model_switch_warning") is not None
        else None,
        "latest_compaction_json": _json_dumps(metadata.get("latest_compaction"))
        if metadata.get("latest_compaction") is not None
        else None,
        "billing_currency": metadata.get("billing_currency"),
        "billing_total": metadata.get("billing_total"),
        "billing_totals_json": _json_dumps(metadata.get("billing_totals"))
        if metadata.get("billing_totals") is not None
        else None,
        "metadata_json": _json_dumps(extra),
    }


def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    event = _json_loads(row["payload_json"], default={})
    if not isinstance(event, dict):
        event = {}
    event["_event_id"] = int(row["event_id"])
    return event


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def _placeholders(values: Sequence[Any] | set[Any]) -> str:
    return ", ".join("?" for _ in values)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None
