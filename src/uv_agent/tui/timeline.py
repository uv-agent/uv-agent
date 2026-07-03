from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from uv_agent.tui.events import runtime_ui_messages_from_payload
from uv_agent.tui.formatting import parse_tool_payload, tool_call_helper_payload

TimelineItemKind = Literal[
    "user",
    "assistant",
    "reasoning",
    "tool_call",
    "tool_result",
    "image",
    "compaction",
    "warning",
    "error",
    "stream_retry",
    "queued",
    "ui_message",
]


def _event_id(event: dict[str, Any] | None) -> int | None:
    if not event:
        return None
    value = event.get("_event_id")
    return value if isinstance(value, int) else None


def _turn_id(event: dict[str, Any] | None) -> str:
    if not event:
        return ""
    return str(event.get("turn_id") or "").strip()


def _item_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for content in item.get("content") or []:
        if isinstance(content, dict) and content.get("type") in {
            "input_text",
            "output_text",
            "text",
            "refusal",
        }:
            parts.append(str(content.get("text") or ""))
    return "\n".join(part for part in parts if part)


def _message_item_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for content in item.get("content") or []:
        if isinstance(content, dict) and content.get("type") in {"output_text", "text", "refusal"}:
            parts.append(str(content.get("text") or ""))
    return "".join(parts)


def _response_text(output: list[dict[str, Any]]) -> str:
    return "".join(
        _message_item_text(item)
        for item in output
        if isinstance(item, dict) and item.get("type") == "message"
    )


def _content_text(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value or "")


def _call_id(call: dict[str, Any]) -> str:
    return str(call.get("call_id") or "").strip()


def _call_index(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _tool_call_field(tool_call: Any, name: str, default: Any = None) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get(name, default)
    return getattr(tool_call, name, default)


def _tool_delta_call(delta: Any) -> dict[str, Any]:
    return {
        "call_id": _tool_call_field(delta, "call_id", "") or "",
        "name": str(_tool_call_field(delta, "name", None) or "python"),
        "arguments": _tool_call_field(delta, "arguments", "")
        or _tool_call_field(delta, "arguments_delta", ""),
    }


def _tool_payload_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    output = event.get("output")
    return parse_tool_payload(output if isinstance(output, dict) else {})


def _tool_result_id(payload: dict[str, Any] | None, call_id: str, turn_id: str) -> str:
    run_id = str((payload or {}).get("run_id") or "").strip()
    if run_id:
        return f"tool_result:{run_id}"
    if call_id:
        return f"tool_result:{call_id}"
    return f"tool_result:{turn_id}:unknown"


def _image_id(attachment: dict[str, Any], event: dict[str, Any]) -> str:
    for key in ("attachment_id", "stored_path", "source_path"):
        value = str(attachment.get(key) or "").strip()
        if value:
            return f"image:{value}"
    event_id = _event_id(event)
    if event_id is not None:
        return f"image:event:{event_id}"
    return f"image:{_turn_id(event)}:{len(json.dumps(attachment, sort_keys=True, default=str))}"


@dataclass
class TimelineItem:
    """A stable, renderable transcript item independent of UI widgets."""

    id: str
    kind: TimelineItemKind
    content: dict[str, Any] = field(default_factory=dict)
    turn_id: str | None = None
    event_id: int | None = None
    process_group: str | None = None

    @property
    def is_process(self) -> bool:
        return self.process_group is not None


@dataclass
class ProcessGroupState:
    id: str
    item_ids: list[str] = field(default_factory=list)
    anchor_item_id: str | None = None
    collapsed: bool = False
    started_at: str | None = None
    completed_at: str | None = None


@dataclass
class TurnAccumulator:
    """Mutable live-turn state used to merge streaming events into TimelineItems."""

    turn_id: str
    started_at: str | None = None
    completed_at: str | None = None
    assistant_item_id: str | None = None
    reasoning_item_id: str | None = None
    reasoning_item_index: int = 0
    assistant_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    tool_delta_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    tool_call_ids_by_index: dict[int, str] = field(default_factory=dict)
    pending_stream_retries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def assistant_buffer(self) -> str:
        return "".join(self.assistant_parts)

    @assistant_buffer.setter
    def assistant_buffer(self, value: str) -> None:
        self.assistant_parts = [value] if value else []

    @property
    def reasoning_buffer(self) -> str:
        return "".join(self.reasoning_parts)

    @reasoning_buffer.setter
    def reasoning_buffer(self, value: str) -> None:
        self.reasoning_parts = [value] if value else []

    def append_assistant_part(self, text: str) -> None:
        self.assistant_parts.append(text)

    def append_reasoning_part(self, text: str) -> None:
        if self.reasoning_parts:
            self.reasoning_parts.append(text)
            return
        stripped = text.lstrip()
        if stripped:
            self.reasoning_parts.append(stripped)


@dataclass
class ThreadViewState:
    """Per-thread UI affordances that survive switching away from a thread."""

    scroll_y: float = 0.0
    follow_tail: bool = True
    fold_collapsed: dict[str, bool] = field(default_factory=dict)
    composer_draft: str = ""
    focused_item_id: str | None = None


class ThreadTimelineState:
    """Data source for one thread transcript.

    Persisted history and live stream events both update this structure. The TUI
    renderer can then rebuild widgets from stable item ids without replaying UI
    mutations or retaining widget references in run state.
    """

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self.items: list[TimelineItem] = []
        self.items_by_id: dict[str, TimelineItem] = {}
        self.process_groups: dict[str, ProcessGroupState] = {}
        self.loaded_start_event_id: int | None = None
        self.loaded_end_event_id: int | None = None
        self.has_older = False
        self.history_loaded = False
        self.active_turns: dict[str, TurnAccumulator] = {}
        self.changed_item_ids: set[str] = set()
        self.changed_process_group_ids: set[str] = set()

    def clear(self) -> None:
        self.items.clear()
        self.items_by_id.clear()
        self.process_groups.clear()
        self.loaded_start_event_id = None
        self.loaded_end_event_id = None
        self.has_older = False
        self.history_loaded = False
        self.active_turns.clear()
        self.changed_item_ids.clear()
        self.changed_process_group_ids.clear()

    def consume_changes(self) -> tuple[set[str], set[str]]:
        """Return and clear item/group ids changed since the last render pass."""

        item_ids = set(self.changed_item_ids)
        group_ids = set(self.changed_process_group_ids)
        self.changed_item_ids.clear()
        self.changed_process_group_ids.clear()
        return item_ids, group_ids

    def load_history_segment(
        self,
        events: list[dict[str, Any]],
        *,
        start_event_id: int | None,
        end_event_id: int | None,
        has_older: bool,
    ) -> None:
        self.clear()
        self.loaded_start_event_id = start_event_id
        self.loaded_end_event_id = end_event_id
        self.has_older = has_older
        self.history_loaded = True
        self._apply_history_events(events, prepend=False)
        self.changed_item_ids.clear()
        self.changed_process_group_ids.clear()

    def merge_history_segment(
        self,
        events: list[dict[str, Any]],
        *,
        start_event_id: int | None,
        end_event_id: int | None,
        has_older: bool,
    ) -> None:
        """Load persisted history before already-buffered live items.

        Thread switching can discover a timeline after live events have already
        arrived for a background turn. In that case history is a prefix, not a
        replacement; clearing here would drop the in-memory stream snapshot that
        exists precisely to make re-entry cheap and deterministic.
        """

        live_items = list(self.items)
        live_groups = {
            group_id: ProcessGroupState(
                id=group.id,
                item_ids=list(group.item_ids),
                anchor_item_id=group.anchor_item_id,
                collapsed=group.collapsed,
                started_at=group.started_at,
                completed_at=group.completed_at,
            )
            for group_id, group in self.process_groups.items()
        }
        active_turns = dict(self.active_turns)
        self.items = []
        self.items_by_id = {}
        self.process_groups = {}
        self.loaded_start_event_id = start_event_id
        self.loaded_end_event_id = end_event_id
        self.has_older = has_older
        self.history_loaded = True
        self._apply_history_events(events, prepend=False)
        history_items = list(self.items)
        history_groups = {
            group_id: ProcessGroupState(
                id=group.id,
                item_ids=list(group.item_ids),
                anchor_item_id=group.anchor_item_id,
                collapsed=group.collapsed,
                started_at=group.started_at,
                completed_at=group.completed_at,
            )
            for group_id, group in self.process_groups.items()
        }
        self.items = []
        self.items_by_id = {}
        self.process_groups = history_groups
        for group_id, group in live_groups.items():
            if group_id not in self.process_groups:
                self.process_groups[group_id] = group
        history_signatures = {self._merge_signature(item) for item in history_items}
        for item in history_items:
            self.items_by_id[item.id] = item
            self.items.append(item)
        for item in live_items:
            if self._merge_signature(item) in history_signatures:
                self._remove_live_item_from_groups(item)
                continue
            self.items_by_id[item.id] = item
            self.items.append(item)
        self._rebuild_process_groups()
        self.active_turns = active_turns
        self.changed_item_ids.clear()
        self.changed_process_group_ids.clear()

    @staticmethod
    def _merge_signature(item: TimelineItem) -> tuple[Any, ...]:
        if item.kind == "tool_result":
            payload = item.content.get("payload") if isinstance(item.content, dict) else {}
            if isinstance(payload, dict):
                run_id = str(payload.get("run_id") or "").strip()
                if run_id:
                    return (item.kind, "run_id", run_id)
        if item.kind == "tool_call":
            call = item.content.get("call") if isinstance(item.content, dict) else {}
            if isinstance(call, dict):
                call_id = _call_id(call)
                if call_id:
                    return (item.kind, "call_id", call_id)
        if item.kind in {"user", "assistant", "reasoning"}:
            text = _content_text(item.content.get("text")) if isinstance(item.content, dict) else ""
            if text:
                return (item.kind, "text", text)
        return (item.kind, item.turn_id, json.dumps(item.content, sort_keys=True, default=str))

    def _remove_live_item_from_groups(self, item: TimelineItem) -> None:
        if not item.process_group:
            return
        group = self.process_groups.get(item.process_group)
        if group is not None and item.id in group.item_ids:
            group.item_ids.remove(item.id)

    def _rebuild_process_groups(self) -> None:
        previous = self.process_groups
        rebuilt: dict[str, ProcessGroupState] = {}
        for item in self.items:
            if not item.process_group:
                continue
            old_group = previous.get(item.process_group)
            group = rebuilt.get(item.process_group)
            if group is None:
                group = ProcessGroupState(
                    id=item.process_group,
                    anchor_item_id=old_group.anchor_item_id if old_group is not None else None,
                    collapsed=old_group.collapsed if old_group is not None else False,
                    started_at=old_group.started_at if old_group is not None else None,
                    completed_at=old_group.completed_at if old_group is not None else None,
                )
                rebuilt[item.process_group] = group
            if item.id not in group.item_ids:
                group.item_ids.append(item.id)
        for group_id, old_group in previous.items():
            if group_id not in rebuilt and old_group.item_ids:
                rebuilt[group_id] = old_group
        self.process_groups = rebuilt

    def prepend_history_segment(
        self,
        events: list[dict[str, Any]],
        *,
        start_event_id: int | None,
        has_older: bool,
    ) -> None:
        before_items = list(self.items)
        before_groups = {
            group_id: ProcessGroupState(
                id=group.id,
                item_ids=list(group.item_ids),
                anchor_item_id=group.anchor_item_id,
                collapsed=group.collapsed,
                started_at=group.started_at,
                completed_at=group.completed_at,
            )
            for group_id, group in self.process_groups.items()
        }
        self.items = []
        self.items_by_id = {}
        self.process_groups = {}
        self._apply_history_events(events, prepend=False)
        prefix_items = list(self.items)
        prefix_groups = {
            group_id: ProcessGroupState(
                id=group.id,
                item_ids=list(group.item_ids),
                anchor_item_id=group.anchor_item_id,
                collapsed=group.collapsed,
                started_at=group.started_at,
                completed_at=group.completed_at,
            )
            for group_id, group in self.process_groups.items()
        }
        self.items = []
        self.items_by_id = {}
        self.process_groups = prefix_groups
        self.process_groups.update(before_groups)
        for item in [*prefix_items, *before_items]:
            self.items_by_id[item.id] = item
            self.items.append(item)
        self.loaded_start_event_id = start_event_id
        self.has_older = has_older
        self.changed_item_ids.clear()
        self.changed_process_group_ids.clear()

    def apply_live_event(self, event_type: str, event: dict[str, Any]) -> None:
        turn_id = _turn_id(event)
        if not turn_id:
            turn_id = "manual" if "manual" in self.active_turns else "turn:unknown"
        acc = self._turn(turn_id)
        self._update_turn_timestamps(acc, event_type, event)

        if event_type == "assistant.delta":
            self._apply_assistant_delta(acc, str(event.get("text") or ""))
        elif event_type == "assistant.reasoning_delta":
            self._apply_reasoning_delta(acc, str(event.get("text") or ""))
        elif event_type == "assistant.reasoning_completed":
            self._finalize_reasoning(acc, str(event.get("text") or ""))
        elif event_type == "assistant.reasoning_absent":
            self._remove_live_reasoning(acc)
        elif event_type == "assistant.response_with_tools":
            text = str(event.get("assistant_text") or acc.assistant_buffer or "")
            if text:
                if acc.assistant_item_id and acc.assistant_item_id in self.items_by_id:
                    self._finalize_assistant_item(acc, text)
                    item = self.items_by_id[acc.assistant_item_id]
                    if item.process_group != acc.turn_id:
                        self._upsert_assistant_item(acc, text, process=True)
                else:
                    self._upsert_assistant_item(acc, text, process=True)
            acc.assistant_buffer = ""
            acc.assistant_item_id = None
        elif event_type == "assistant.final_response_started":
            text = str(event.get("assistant_text") or acc.assistant_buffer or "")
            if text:
                if acc.assistant_item_id and acc.assistant_item_id in self.items_by_id:
                    self._finalize_assistant_item(acc, text)
                else:
                    self._upsert_assistant_item(acc, text, process=False)
            acc.assistant_buffer = ""
            acc.assistant_item_id = None
            self._collapse_process_group(acc.turn_id)
        elif event_type == "tool.delta":
            self._apply_tool_delta(acc, event)
        elif event_type == "tool.started":
            if acc.reasoning_item_id:
                item = self.items_by_id.get(acc.reasoning_item_id)
                if item is not None and item.process_group:
                    group = self.process_groups.get(item.process_group)
                    if group is not None:
                        group.collapsed = True
                        self.changed_process_group_ids.add(group.id)
                else:
                    self._remove_live_reasoning(acc)
            self._apply_tool_started(acc, event)
        elif event_type == "tool.partial":
            self._apply_tool_partial(acc, event)
        elif event_type == "tool.output":
            self._apply_tool_output(acc, event)
        elif event_type == "turn.stream_retry":
            self._apply_stream_retry(event)
        elif event_type == "model.stream_retry":
            acc.pending_stream_retries.append(dict(event))
        elif event_type == "thread.token_estimation_warning":
            self._apply_warning(event, kind="token_estimation")
        elif event_type == "thread.plugin_epoch_context_warning":
            self._apply_warning(event, kind="plugin_epoch_context")
        elif event_type == "compaction.completed":
            self._finalize_turn(acc)
            self._apply_compaction(event)
            self._reset_process_group_after_compaction(acc.turn_id, self._compaction_item_id(event))
        elif event_type == "image.attachment":
            attachment = event.get("attachment") if isinstance(event.get("attachment"), dict) else {}
            self._append_or_update(TimelineItem(
                id=_image_id(attachment, event),
                kind="image",
                content={"attachment": attachment},
                turn_id=turn_id,
                event_id=_event_id(event),
            ))
        elif event_type == "turn.interrupted":
            self._finalize_turn(acc)
            self._collapse_process_group(acc.turn_id)
            self.active_turns.pop(acc.turn_id, None)
        elif event_type == "turn.error":
            self._finalize_turn(acc)
            self._collapse_process_group(acc.turn_id)
            self._apply_error(event)
            self.active_turns.pop(acc.turn_id, None)
        elif event_type == "turn.completed":
            final_text = str(event.get("final_text") or "")
            if final_text and not any(
                existing.kind == "assistant" and existing.turn_id == acc.turn_id
                for existing in self.items
            ):
                self._upsert_assistant_item(acc, final_text, process=False)
            self._finalize_turn(acc)
            self._collapse_process_group(acc.turn_id)
            self.active_turns.pop(acc.turn_id, None)

    def flush_pending_stream_retries(self, turn_id: str) -> None:
        acc = self.active_turns.get(turn_id)
        if acc is None:
            return
        retries = list(acc.pending_stream_retries)
        acc.pending_stream_retries.clear()
        for retry in retries:
            self._apply_stream_retry(retry)

    def _apply_history_events(self, events: list[dict[str, Any]], *, prepend: bool) -> None:
        index = 0
        while index < len(events):
            event = events[index]
            turn_id = _turn_id(event)
            if not turn_id:
                self._apply_single_history_event(event)
                index += 1
                continue
            turn_events: list[dict[str, Any]] = []
            while index < len(events) and _turn_id(events[index]) == turn_id:
                turn_events.append(events[index])
                index += 1
            self._apply_history_turn(turn_id, turn_events)

    def _apply_history_turn(self, turn_id: str, events: list[dict[str, Any]]) -> None:
        group = self._group(turn_id)
        group.started_at = self._history_started_at(events)
        group.completed_at = self._history_completed_at(events)
        show_stream_retries = self._history_turn_ended_with_error(events)
        seen_user = False
        calls_by_id: dict[str, dict[str, Any]] = {}
        for event in events:
            event_type = event.get("type")
            if event_type == "item.user":
                if seen_user:
                    continue
                seen_user = True
            elif event_type == "item.model_response":
                for output_item in event.get("output") or []:
                    if not isinstance(output_item, dict) or output_item.get("type") != "function_call":
                        continue
                    call_id = _call_id(output_item)
                    if call_id:
                        calls_by_id[call_id] = dict(output_item)
            elif event_type == "item.runner_result" and not event.get("call"):
                call_id = str(event.get("call_id") or "")
                if call_id in calls_by_id:
                    event = {**event, "call": calls_by_id[call_id]}
            if event_type == "turn.stream_retry" and not show_stream_retries:
                continue
            for item in self._history_items_for_event(event):
                self._append_or_update(item)
        if group.item_ids:
            group.collapsed = True

    def _apply_single_history_event(self, event: dict[str, Any]) -> None:
        for item in self._history_items_for_event(event):
            self._append_or_update(item)

    def _history_items_for_event(self, event: dict[str, Any]) -> list[TimelineItem]:
        event_type = event.get("type")
        turn_id = _turn_id(event) or None
        event_id = _event_id(event)
        items: list[TimelineItem] = []
        if event_type == "item.user":
            text = _item_text(event.get("item") or {})
            if text:
                items.append(TimelineItem(
                    id=f"user:{turn_id or event_id or len(self.items)}",
                    kind="user",
                    content={"text": text},
                    turn_id=turn_id,
                    event_id=event_id,
                ))
        elif event_type == "item.model_response":
            output = [item for item in event.get("output") or [] if isinstance(item, dict)]
            reasoning_text = str(event.get("reasoning_text") or "")
            response_id = str(event.get("response_id") or event_id or len(self.items))
            has_tool_call = any(item.get("type") == "function_call" for item in output)
            if reasoning_text.strip():
                items.append(TimelineItem(
                    id=f"reasoning:{response_id}",
                    kind="reasoning",
                    content={"text": reasoning_text},
                    turn_id=turn_id,
                    event_id=event_id,
                    process_group=turn_id if turn_id else None,
                ))
            for idx, output_item in enumerate(output):
                item_type = output_item.get("type")
                if item_type == "message":
                    text = _message_item_text(output_item)
                    if text:
                        items.append(TimelineItem(
                            id=f"assistant:{response_id}:{idx}",
                            kind="assistant",
                            content={"text": text},
                            turn_id=turn_id,
                            event_id=event_id,
                            process_group=turn_id if has_tool_call and turn_id else None,
                        ))
                elif item_type == "function_call":
                    call_id = _call_id(output_item) or f"{response_id}:{idx}"
                    items.append(TimelineItem(
                        id=f"tool_call:{call_id}",
                        kind="tool_call",
                        content={"call": dict(output_item), "status": "called"},
                        turn_id=turn_id,
                        event_id=event_id,
                        process_group=turn_id if turn_id else None,
                    ))
        elif event_type == "item.runner_result":
            result = event.get("result") if isinstance(event.get("result"), dict) else {}
            call_id = str(event.get("call_id") or "")
            call = dict(event.get("call") or {}) if isinstance(event.get("call"), dict) else {}
            if call and "helper_calls" not in result:
                result = {**result, "helper_calls": tool_call_helper_payload(call)}
            for ui_message in runtime_ui_messages_from_payload(result):
                items.append(TimelineItem(
                    id=f"ui_message:{ui_message['id']}",
                    kind="ui_message",
                    content={"text": str(ui_message.get("message") or "")},
                    turn_id=turn_id,
                    event_id=event_id,
                ))
            items.append(TimelineItem(
                id=_tool_result_id(result, call_id, turn_id or ""),
                kind="tool_result",
                content={"payload": dict(result), "call": dict(call) if call else None},
                turn_id=turn_id,
                event_id=event_id,
                process_group=turn_id if turn_id else None,
            ))
        elif event_type == "item.image_attachment":
            attachment = event.get("attachment") if isinstance(event.get("attachment"), dict) else {}
            items.append(TimelineItem(
                id=_image_id(attachment, event),
                kind="image",
                content={"attachment": attachment},
                turn_id=turn_id,
                event_id=event_id,
            ))
        elif event_type in {"item.reasoning_delta", "item.reasoning_partial"}:
            text = str(event.get("text") or "")
            if text.strip():
                items.append(TimelineItem(
                    id=f"reasoning:{turn_id or 'unknown'}:{event_id or len(self.items)}",
                    kind="reasoning",
                    content={"text": text},
                    turn_id=turn_id,
                    event_id=event_id,
                    process_group=turn_id if turn_id else None,
                ))
        elif event_type == "turn.error":
            items.append(self._error_item(event))
        elif event_type == "turn.stream_retry":
            items.append(self._stream_retry_item(event))
        elif event_type == "item.compaction":
            item = self._compaction_item(event)
            if item is not None:
                items.append(item)
        elif event_type == "thread.token_estimation_warning":
            items.append(self._warning_item(event, kind="token_estimation"))
        elif event_type == "thread.model_switch_warning":
            items.append(self._warning_item(event, kind="model_switch"))
        elif event_type == "thread.plugin_epoch_context_warning":
            items.append(self._warning_item(event, kind="plugin_epoch_context"))
        return items

    def _turn(self, turn_id: str) -> TurnAccumulator:
        acc = self.active_turns.get(turn_id)
        if acc is None:
            acc = TurnAccumulator(turn_id=turn_id)
            self.active_turns[turn_id] = acc
        return acc

    def _group(self, turn_id: str) -> ProcessGroupState:
        group = self.process_groups.get(turn_id)
        if group is None:
            group = ProcessGroupState(id=turn_id)
            self.process_groups[turn_id] = group
        return group

    def _append_or_update(self, item: TimelineItem) -> TimelineItem:
        existing = self.items_by_id.get(item.id)
        if existing is None:
            self.items_by_id[item.id] = item
            self.items.append(item)
            self.changed_item_ids.add(item.id)
            if item.process_group:
                self._track_process_item(item.process_group, item.id)
            return item
        existing.kind = item.kind
        existing.content = item.content
        existing.turn_id = item.turn_id
        existing.event_id = item.event_id
        old_group = existing.process_group
        existing.process_group = item.process_group
        if old_group and old_group != item.process_group:
            group = self.process_groups.get(old_group)
            if group is not None and item.id in group.item_ids:
                group.item_ids.remove(item.id)
            self.changed_process_group_ids.add(old_group)
        if item.process_group:
            self._track_process_item(item.process_group, item.id)
        self.changed_item_ids.add(item.id)
        return existing

    def _remove_item(self, item_id: str) -> None:
        item = self.items_by_id.pop(item_id, None)
        if item is None:
            return
        self.items = [existing for existing in self.items if existing.id != item_id]
        if item.process_group:
            group = self.process_groups.get(item.process_group)
            if group is not None and item_id in group.item_ids:
                group.item_ids.remove(item_id)
            self.changed_process_group_ids.add(item.process_group)
        self.changed_item_ids.add(item_id)

    def _track_process_item(self, group_id: str, item_id: str) -> None:
        group = self._group(group_id)
        if item_id not in group.item_ids:
            group.item_ids.insert(self._process_group_insert_index(group_id, item_id), item_id)
            self.changed_process_group_ids.add(group_id)

    def _process_group_insert_index(self, group_id: str, item_id: str) -> int:
        """Return the group position matching the transcript item order."""

        item_order = {item.id: index for index, item in enumerate(self.items)}
        item_position = item_order.get(item_id)
        if item_position is None:
            return len(self._group(group_id).item_ids)
        group = self._group(group_id)
        for index, existing_id in enumerate(group.item_ids):
            existing_position = item_order.get(existing_id)
            if existing_position is not None and item_position < existing_position:
                return index
        return len(group.item_ids)

    def _update_turn_timestamps(self, acc: TurnAccumulator, event_type: str, event: dict[str, Any]) -> None:
        started = str(event.get("turn_started_at") or event.get("started_at") or "").strip()
        if started:
            acc.started_at = started
            self._group(acc.turn_id).started_at = started
        if event_type in {"turn.completed", "turn.interrupted", "turn.error"}:
            completed = str(event.get("completed_at") or event.get("created_at") or "").strip()
            if completed:
                acc.completed_at = completed
                self._group(acc.turn_id).completed_at = completed

    def ensure_user_item(self, turn_id: str, text: str, *, created_at: str | None = None) -> None:
        if not text:
            return
        self._append_or_update(TimelineItem(
            id=f"user:{turn_id}",
            kind="user",
            content={"text": text},
            turn_id=turn_id,
        ))
        if created_at:
            self._group(turn_id).started_at = created_at

    def promote_user_turn(self, old_turn_id: str | None, new_turn_id: str) -> None:
        old_turn_id = str(old_turn_id or "").strip()
        new_turn_id = str(new_turn_id or "").strip()
        if not old_turn_id or not new_turn_id or old_turn_id == new_turn_id:
            return
        old_item_id = f"user:{old_turn_id}"
        item = self.items_by_id.pop(old_item_id, None)
        if item is None:
            return
        item.id = f"user:{new_turn_id}"
        item.turn_id = new_turn_id
        self.items_by_id[item.id] = item
        old_group = self.process_groups.pop(old_turn_id, None)
        if old_group is not None:
            group = self._group(new_turn_id)
            group.started_at = group.started_at or old_group.started_at
            group.completed_at = group.completed_at or old_group.completed_at
            group.collapsed = group.collapsed or old_group.collapsed
            group.anchor_item_id = group.anchor_item_id or old_group.anchor_item_id
        self.changed_item_ids.add(item.id)

    def add_queued_turn(self, queue_id: str, prompt: str, image_paths: list[Any]) -> None:
        self._append_or_update(TimelineItem(
            id=f"queued:{queue_id}",
            kind="queued",
            content={"prompt": prompt, "image_paths": list(image_paths)},
        ))

    def remove_queued_turn(self, queue_id: str) -> None:
        self._remove_item(f"queued:{queue_id}")

    def seed_assistant_delta(self, text: str, *, turn_id: str = "manual") -> None:
        acc = self._turn(turn_id)
        self._apply_assistant_delta(acc, text)

    def seed_reasoning_delta(self, text: str, *, turn_id: str = "manual") -> None:
        acc = self._turn(turn_id)
        self._apply_reasoning_delta(acc, text)

    def seed_tool_output(self, event: dict[str, Any], *, turn_id: str = "manual") -> None:
        event = {**event, "turn_id": event.get("turn_id") or turn_id}
        acc = self._turn(str(event.get("turn_id") or turn_id))
        index = _call_index(event.get("tool_call_index", 0))
        call_raw = event.get("call")
        call = dict(call_raw) if isinstance(call_raw, dict) else acc.tool_delta_calls.get(index, {})
        call_id = _call_id(call)
        if call_id and (call.get("arguments") or call.get("name") not in {None, "", "python"}):
            self._apply_tool_output(acc, event)
            return
        payload = _tool_payload_from_event(event)
        if payload is None:
            payload = {"returncode": None, "stdout": "", "stderr": "", "events": []}
        payload = dict(payload)
        if call and "helper_calls" not in payload:
            payload["helper_calls"] = tool_call_helper_payload(call)
        payload.pop("partial", None)
        payload.pop("partial_reason", None)
        payload.pop("call_id", None)
        self._append_or_update(TimelineItem(
            id=_tool_result_id(payload, call_id, acc.turn_id),
            kind="tool_result",
            content={"payload": payload},
            turn_id=acc.turn_id,
            process_group=acc.turn_id,
        ))

    def seed_tool_started(self, event: dict[str, Any], *, turn_id: str = "manual") -> None:
        event = {**event, "turn_id": event.get("turn_id") or turn_id}
        acc = self._turn(str(event.get("turn_id") or turn_id))
        self._apply_tool_started(acc, event)

    def seed_tool_partial(self, event: dict[str, Any], *, turn_id: str = "manual") -> None:
        event = {**event, "turn_id": event.get("turn_id") or turn_id}
        acc = self._turn(str(event.get("turn_id") or turn_id))
        self._apply_tool_partial(acc, event)

    def _apply_assistant_delta(self, acc: TurnAccumulator, text: str) -> None:
        if not text:
            return
        acc.append_assistant_part(text)
        text_parts = acc.assistant_parts if len(acc.assistant_parts) > 1 else acc.assistant_buffer
        self._upsert_assistant_item(acc, text_parts, process=False)

    def _upsert_assistant_item(self, acc: TurnAccumulator, text: str | list[str], *, process: bool) -> None:
        if acc.assistant_item_id is None or acc.assistant_item_id not in self.items_by_id:
            suffix = len([item for item in self.items if item.turn_id == acc.turn_id and item.kind == "assistant"])
            acc.assistant_item_id = f"assistant:live:{acc.turn_id}:{suffix}"
        self._append_or_update(TimelineItem(
            id=acc.assistant_item_id,
            kind="assistant",
            content={"text": text, "partial": acc.turn_id != "manual"},
            turn_id=acc.turn_id,
            process_group=acc.turn_id if process else None,
        ))

    def _apply_reasoning_delta(self, acc: TurnAccumulator, text: str) -> None:
        if not text:
            return
        acc.append_reasoning_part(text)
        reasoning: str | list[str] = acc.reasoning_parts if len(acc.reasoning_parts) > 1 else acc.reasoning_buffer
        if acc.reasoning_item_id is None or acc.reasoning_item_id not in self.items_by_id:
            acc.reasoning_item_id = self._next_reasoning_item_id(acc)
        self._append_or_update(TimelineItem(
            id=acc.reasoning_item_id,
            kind="reasoning",
            content={"text": reasoning, "partial": True},
            turn_id=acc.turn_id,
            # Treat live reasoning as part of the turn process from its first
            # visible chunk.  Otherwise the fold bar is mounted only after the
            # reasoning is finalized or a tool starts, which makes it appear
            # late and then jump above the already-rendered thinking row.
            process_group=acc.turn_id if acc.turn_id != "manual" else None,
        ))

    def _finalize_reasoning(self, acc: TurnAccumulator, text: str) -> None:
        stripped = (text or acc.reasoning_buffer).strip()
        if not stripped:
            self._remove_live_reasoning(acc)
            return
        item_id = acc.reasoning_item_id or self._next_reasoning_item_id(acc)
        acc.reasoning_item_id = item_id
        self._append_or_update(TimelineItem(
            id=item_id,
            kind="reasoning",
            content={"text": stripped, "partial": False},
            turn_id=acc.turn_id,
            process_group=acc.turn_id,
        ))
        acc.reasoning_buffer = ""
        acc.reasoning_item_id = None

    def _next_reasoning_item_id(self, acc: TurnAccumulator) -> str:
        """Return a fresh live reasoning id for one model round.

        A single user turn may contain several model responses separated by tool
        calls. Completed reasoning remains in the process fold, so a later live
        reasoning delta must not reuse and overwrite the previous cell.
        """

        while True:
            item_id = f"reasoning:live:{acc.turn_id}:{acc.reasoning_item_index}"
            acc.reasoning_item_index += 1
            if item_id not in self.items_by_id:
                return item_id

    def _remove_live_reasoning(self, acc: TurnAccumulator) -> None:
        if acc.reasoning_item_id:
            self._remove_item(acc.reasoning_item_id)
        acc.reasoning_item_id = None
        acc.reasoning_buffer = ""

    def _apply_tool_delta(self, acc: TurnAccumulator, event: dict[str, Any]) -> None:
        index = _call_index(_tool_call_field(event.get("tool_call"), "index", event.get("tool_call_index", 0)))
        call = _tool_delta_call(event.get("tool_call"))
        acc.tool_delta_calls[index] = call
        self._upsert_tool_call_item(acc, index, call, status="running")

    def _apply_tool_started(self, acc: TurnAccumulator, event: dict[str, Any]) -> None:
        index = _call_index(event.get("tool_call_index", 0))
        call_raw = event.get("call")
        call = dict(call_raw) if isinstance(call_raw, dict) else acc.tool_delta_calls.get(index, {})
        if not call:
            call = {"name": "python", "arguments": ""}
        acc.tool_delta_calls[index] = call
        self._upsert_tool_call_item(acc, index, call, status="running")

    def _upsert_tool_call_item(self, acc: TurnAccumulator, index: int, call: dict[str, Any], *, status: str) -> str:
        call_id = _call_id(call)
        item_id = f"tool_call:{call_id}" if call_id else f"tool_call:{acc.turn_id}:{index}"
        acc.tool_call_ids_by_index[index] = item_id
        self._append_or_update(TimelineItem(
            id=item_id,
            kind="tool_call",
            content={"call": dict(call), "status": status},
            turn_id=acc.turn_id,
            process_group=acc.turn_id,
        ))
        return item_id

    def _apply_tool_partial(self, acc: TurnAccumulator, event: dict[str, Any]) -> None:
        payload = _tool_payload_from_event(event)
        if payload is None:
            return
        call_raw = event.get("call")
        call = dict(call_raw) if isinstance(call_raw, dict) else {}
        call_id = _call_id(call) or str(payload.get("call_id") or "")
        item_id = _tool_result_id(payload, call_id, acc.turn_id)
        self._append_or_update(TimelineItem(
            id=item_id,
            kind="tool_result",
            content={"payload": dict(payload)},
            turn_id=acc.turn_id,
            process_group=acc.turn_id,
        ))

    def _apply_tool_output(self, acc: TurnAccumulator, event: dict[str, Any]) -> None:
        index = _call_index(event.get("tool_call_index", 0))
        call_raw = event.get("call")
        call = dict(call_raw) if isinstance(call_raw, dict) else acc.tool_delta_calls.get(index, {})
        call_id = _call_id(call)
        if call:
            self._upsert_tool_call_item(acc, index, {**call, "_status_label": call.get("_status_label") or "called"}, status="called")
        payload = _tool_payload_from_event(event)
        if payload is None:
            payload = {"returncode": None, "stdout": "", "stderr": "", "events": []}
        payload = dict(payload)
        if call and "helper_calls" not in payload:
            payload["helper_calls"] = tool_call_helper_payload(call)
        payload.pop("partial", None)
        payload.pop("partial_reason", None)
        payload.pop("call_id", None)
        self._append_or_update(TimelineItem(
            id=_tool_result_id(payload, call_id, acc.turn_id),
            kind="tool_result",
            content={"payload": payload},
            turn_id=acc.turn_id,
            process_group=acc.turn_id,
        ))
        acc.tool_delta_calls.pop(index, None)

    def _finalize_turn(self, acc: TurnAccumulator) -> None:
        if acc.reasoning_buffer.strip():
            self._finalize_reasoning(acc, acc.reasoning_buffer)
        else:
            self._remove_live_reasoning(acc)
        if acc.assistant_buffer and acc.assistant_item_id is None:
            self._upsert_assistant_item(acc, acc.assistant_buffer, process=False)
        acc.assistant_buffer = ""
        acc.assistant_item_id = None

    def _finalize_assistant_item(self, acc: TurnAccumulator, text: str) -> None:
        item_id = acc.assistant_item_id
        if item_id is None or item_id not in self.items_by_id:
            return
        item = self.items_by_id[item_id]
        if not item.content.get("partial") and item.content.get("text") == text:
            return
        self._append_or_update(TimelineItem(
            id=item.id,
            kind="assistant",
            content={"text": text},
            turn_id=acc.turn_id,
            process_group=item.process_group,
            event_id=item.event_id,
        ))

    def _collapse_process_group(self, turn_id: str) -> None:
        group = self.process_groups.get(turn_id)
        if group is not None and group.item_ids:
            group.collapsed = True
            self.changed_process_group_ids.add(turn_id)

    def _reset_process_group_after_compaction(self, turn_id: str, compaction_item_id: str) -> None:
        group = self.process_groups.get(turn_id)
        if group is not None:
            group.anchor_item_id = compaction_item_id
            group.collapsed = True
            self.changed_process_group_ids.add(turn_id)

    def _apply_warning(self, event: dict[str, Any], *, kind: str) -> None:
        self._append_or_update(self._warning_item(event, kind=kind))

    def _warning_item(self, event: dict[str, Any], *, kind: str) -> TimelineItem:
        event_id = _event_id(event)
        turn_id = _turn_id(event) or None
        return TimelineItem(
            id=f"warning:{kind}:{event_id or turn_id or len(self.items)}",
            kind="warning",
            content={"warning_kind": kind, "event": dict(event)},
            turn_id=turn_id,
            event_id=event_id,
        )

    def _apply_stream_retry(self, event: dict[str, Any]) -> None:
        self._append_or_update(self._stream_retry_item(event))

    def _stream_retry_item(self, event: dict[str, Any]) -> TimelineItem:
        event_id = _event_id(event)
        turn_id = _turn_id(event) or None
        attempt = event.get("attempt") or "?"
        return TimelineItem(
            id=f"stream_retry:{turn_id or 'unknown'}:{event_id or attempt}",
            kind="stream_retry",
            content={"event": dict(event)},
            turn_id=turn_id,
            event_id=event_id,
        )

    def _apply_error(self, event: dict[str, Any]) -> None:
        self._append_or_update(self._error_item(event))

    def _error_item(self, event: dict[str, Any]) -> TimelineItem:
        event_id = _event_id(event)
        turn_id = _turn_id(event) or None
        return TimelineItem(
            id=f"error:{turn_id or event_id or len(self.items)}",
            kind="error",
            content={"event": dict(event)},
            turn_id=turn_id,
            event_id=event_id,
        )

    def _apply_compaction(self, event: dict[str, Any]) -> None:
        item = self._compaction_item(event)
        if item is not None:
            self._append_or_update(item)

    def _compaction_item_id(self, event: dict[str, Any]) -> str:
        event_id = _event_id(event)
        turn_id = _turn_id(event)
        text = str(event.get("text") or "")
        return f"compaction:{event_id or turn_id or abs(hash(text))}"

    def _compaction_item(self, event: dict[str, Any]) -> TimelineItem | None:
        text = str(event.get("text") or "").strip()
        if not text:
            return None
        return TimelineItem(
            id=self._compaction_item_id(event),
            kind="compaction",
            content={"event": dict(event), "text": text},
            turn_id=_turn_id(event) or None,
            event_id=_event_id(event),
        )

    @staticmethod
    def _history_turn_ended_with_error(events: list[dict[str, Any]]) -> bool:
        for event in reversed(events):
            if event.get("type") in {"turn.completed", "turn.interrupted", "turn.error"}:
                return event.get("type") == "turn.error"
        return False

    @staticmethod
    def _history_started_at(events: list[dict[str, Any]]) -> str | None:
        for event in events:
            if event.get("type") in {"turn.started", "item.user"}:
                value = str(event.get("created_at") or "").strip()
                if value:
                    return value
        return None

    @staticmethod
    def _history_completed_at(events: list[dict[str, Any]]) -> str | None:
        completed = ""
        for event in events:
            if event.get("type") in {"turn.completed", "turn.interrupted", "turn.error"}:
                completed = str(event.get("created_at") or "").strip()
        return completed or None
