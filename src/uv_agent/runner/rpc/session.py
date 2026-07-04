from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from typing import TYPE_CHECKING

from uv_agent.blobs import BlobStore
from uv_agent.runner.run_log import EventWriter
from uv_agent.time import utc_now_iso
from uv_agent_runtime.events import RUNTIME_EVENT_EVENT_ID_KEY
from uv_agent_runtime.ui import UI_MESSAGE_FORMAT, UI_MESSAGE_KIND

if TYPE_CHECKING:
    from uv_agent.host_events import HostEventBus


@dataclass(frozen=True)
class RunContext:
    """Immutable context passed to host methods called by a runtime script."""

    run_id: str
    thread_id: str | None
    turn_id: str | None
    cwd: Path


class RunSession:
    """Per-run state addressed by a short-lived bearer token."""

    def __init__(
        self,
        *,
        token: str,
        run_id: str,
        thread_id: str | None,
        turn_id: str | None,
        cwd: Path,
        on_structured_event: Callable[[dict[str, Any]], None],
        writer: EventWriter,
        on_helper_calls: Callable[[list[dict[str, Any]]], None] | None = None,
        host_events: "HostEventBus | None" = None,
        blob_store: BlobStore | None = None,
    ) -> None:
        self.token = token
        self._host_events = host_events
        self.context = RunContext(
            run_id=run_id,
            thread_id=thread_id,
            turn_id=turn_id,
            cwd=cwd,
        )
        self._on_structured_event = on_structured_event
        self._on_helper_calls = on_helper_calls or (lambda _calls: None)
        self._writer = writer
        self._blob_store = blob_store
        self._temporary_blob_ids: set[str] = set()
        self._lock = threading.RLock()
        self.closed = False

    @property
    def run_id(self) -> str:
        return self.context.run_id

    def close(self) -> None:
        with self._lock:
            self.closed = True
            temporary_blob_ids = sorted(self._temporary_blob_ids)
            self._temporary_blob_ids.clear()
        if self._blob_store is not None and temporary_blob_ids:
            try:
                self._blob_store.gc_unreferenced(blob_ids=temporary_blob_ids)
            except Exception:
                pass

    def note_temporary_blob(self, blob_id: str) -> None:
        with self._lock:
            if self.closed:
                raise RuntimeError("Run session is closed")
            self._temporary_blob_ids.add(str(blob_id))

    def emit_event(self, event: dict[str, Any]) -> None:
        """Append a structured runtime event and persist it in the run log."""

        event_copy = dict(event)
        with self._lock:
            if self.closed:
                raise RuntimeError("Run session is closed")
            created_at = utc_now_iso()
            self._on_structured_event(event_copy)
            self._writer.write(
                {
                    "type": "run.event",
                    "created_at": created_at,
                    "run_id": self.run_id,
                    "event": event_copy,
                }
            )
            ui_event = _runtime_ui_message_event(event_copy, context=self.context, created_at=created_at)
            if ui_event is not None:
                self._publish_host_event(ui_event)

    def record_helper_calls(self, calls: list[Any]) -> None:
        """Record sanitized helper-call summaries delivered by the runtime."""

        normalized = [call for item in calls for call in [_normalize_helper_call(item)] if call is not None]
        with self._lock:
            if self.closed:
                raise RuntimeError("Run session is closed")
            self._on_helper_calls(normalized)
            self._publish_host_event(
                {
                    "type": "runtime.helper_calls",
                    "run_id": self.run_id,
                    "thread_id": self.context.thread_id,
                    "turn_id": self.context.turn_id,
                    "cwd": str(self.context.cwd),
                    "calls": normalized,
                }
            )


    def _publish_host_event(self, event: dict[str, Any]) -> None:
        """Best-effort publish a host event; never raise."""

        if self._host_events is None:
            return
        try:
            self._host_events.publish(event)
        except Exception:
            return


def _normalize_helper_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    name = str(value.get("name") or value.get("helper") or "").strip()
    if not name:
        return None
    count = _positive_int(value.get("count")) or 1
    call: dict[str, Any] = {
        "name": _short_text(name, 120),
        "args": "",
        "source": "runtime",
        "count": count,
    }
    outcomes = value.get("outcomes")
    if isinstance(outcomes, dict):
        normalized_outcomes = {
            str(key): int(amount)
            for key, amount in outcomes.items()
            if isinstance(amount, int) and amount > 0
        }
        if normalized_outcomes:
            call["outcomes"] = normalized_outcomes
    elif isinstance(value.get("outcome"), str):
        call["outcomes"] = {str(value["outcome"]): count}
    duration = _float_or_none(value.get("total_duration_ms"))
    if duration is None:
        duration = _float_or_none(value.get("duration_ms"))
    if duration is not None:
        call["total_duration_ms"] = round(max(0.0, duration), 3)
    keyword_names = _string_list(value.get("keyword_names"), max_items=64, max_item_chars=120)
    if keyword_names:
        call["keyword_names"] = sorted(set(keyword_names))
    positional_counts = _int_list(value.get("positional_counts"), max_items=16)
    positional_count = _positive_int(value.get("positional_count"))
    if positional_count is not None:
        positional_counts.append(positional_count)
    if positional_counts:
        call["positional_counts"] = sorted(set(positional_counts))
    argument_types = value.get("argument_types")
    if isinstance(argument_types, dict):
        call["argument_types"] = argument_types
    error_types = _string_list(value.get("error_types"), max_items=32, max_item_chars=120)
    error_type = value.get("error_type")
    if isinstance(error_type, str) and error_type:
        error_types.append(_short_text(error_type, 120))
    if error_types:
        call["error_types"] = sorted(set(error_types))
    return call


def _runtime_ui_message_event(
    event: dict[str, Any],
    *,
    context: RunContext,
    created_at: str,
) -> dict[str, Any] | None:
    if event.get("kind") != UI_MESSAGE_KIND:
        return None
    message = str(event.get("message") or "")
    if not message.strip():
        return None
    event_id = str(event.get(RUNTIME_EVENT_EVENT_ID_KEY) or "")
    return {
        "type": "runtime.ui.message",
        "scope": "ui",
        "created_at": created_at,
        "run_id": context.run_id,
        "thread_id": context.thread_id,
        "turn_id": context.turn_id,
        "cwd": str(context.cwd),
        "event_id": event_id,
        "message": message,
        "format": str(event.get("format") or UI_MESSAGE_FORMAT),
        "event": dict(event),
    }


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any, *, max_items: int, max_item_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if len(result) >= max_items:
            break
        text = _short_text(str(item), max_item_chars)
        if text:
            result.append(text)
    return result


def _int_list(value: Any, *, max_items: int) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        if len(result) >= max_items:
            break
        parsed = _positive_int(item)
        if parsed is not None:
            result.append(parsed)
    return result


def _short_text(value: str, max_chars: int) -> str:
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
