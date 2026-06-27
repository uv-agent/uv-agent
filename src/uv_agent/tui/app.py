from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Literal, TypeAlias, cast

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.geometry import Offset
from textual.screen import Screen
from textual.reactive import reactive
from textual.selection import Selection
from textual.widget import Widget
from textual.widgets import Button, Static, TextArea
from textual.worker import Worker

from uv_agent.atomic import atomic_replace
from uv_agent.billing import billing_total_from_metadata, format_billing_total
from uv_agent.config import ConfigError
from uv_agent.environment import application_version, detect_user_language, host_environment_line
from uv_agent.errors import (
    error_renderable,
    format_error,
    is_retryable_provider_error,
)
from uv_agent.i18n import command_description, tr
from uv_agent.ids import new_id
from uv_agent.notifications import play_completion_sound
from uv_agent.paths import project_state_dir, project_tui_clipboard_dir, uv_agent_home
from uv_agent.session.store import VISIBLE_HISTORY_EVENT_TYPES
from uv_agent.thread_titles import DEFAULT_THREAD_TITLES
from uv_agent.time import utc_now_iso
from uv_agent.tui.config_panels import ConfigPanelMixin
from uv_agent.tui.formatting import (
    format_elapsed,
    format_tokens,
    join_lines,
    parse_tool_payload,
    plain,
    RenderablePart,
    renderable_plain,
    short_thread,
    tool_call_detail_highlight_markup,
    tool_call_preview_line,
    tool_call_summary_markup,
    tool_detail_markup,
    tool_timeline_markup,
)
from uv_agent.tui.image_support import ImageSupportMixin
from uv_agent.tui.mentions import MentionMixin
from uv_agent.tui.panels import (
    FullscreenPanel,
    ImagePreviewPanel,
    PendingImagePreviewPanel,
    PendingSendQueuePanel,
    ToolDetailsPanel,
    WorktreeBranchPanel,
)
from uv_agent.tui.state import (
    CommandSpec,
    MentionScanCache,
    PendingImage,
    PickerItem,
    QueuedTurn,
    ThreadActivityState,
    ThreadRunState,
    TopNotification,
)
from uv_agent.tui.timeline import (
    ThreadTimelineState,
    ThreadViewState,
    TimelineItem,
)
from uv_agent.tui.styles import MAIN_APP_CSS
from uv_agent.tui.widgets import (
    ComposerTextArea,
    EmptyState,
    ExpandableTranscriptCell,
    FoldedProcessCell,
    ImageAttachmentCell,
    LoadOlderHistoryCell,
    RetryTurnButton,
    TranscriptCell,
    TranscriptScroll,
)
from uv_agent.tui.window_title import sanitized_window_title, write_window_title
from uv_agent.worktree import (
    CommandResult,
    WorktreeError,
    cleanup_worktree,
    create_worktree,
    validate_worktree_branch_name,
)


COMPOSER_COLLAPSED_HEIGHT = 5
COMPOSER_EXPANDED_HEIGHT = 8
COMPOSER_AUTO_EXPAND_LINES = 3
QUIT_KEY_DEBOUNCE_SECONDS = 0.08
MAX_COMPOSER_HISTORY = 50
COMPOSER_HISTORY_FILENAME = "composer_history.json"
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
GOAL_MODE_STYLE = "bold #ff5a36"
GOAL_FILE_PREVIEW_MAX_CHARS = 100_000
STREAM_RENDER_INTERVAL_SECONDS = 0.05
STREAM_STATUS_INTERVAL_SECONDS = 0.25
MountBefore: TypeAlias = int | str | Widget | None
NotificationSeverity: TypeAlias = Literal["information", "warning", "error"]


__all__ = [
    "EmptyState",
    "ExpandableTranscriptCell",
    "FoldedProcessCell",
    "FullscreenPanel",
    "ImageAttachmentCell",
    "ImagePreviewPanel",
    "PendingImage",
    "PendingImagePreviewPanel",
    "PendingSendQueuePanel",
    "RetryTurnButton",
    "ToolDetailsPanel",
    "TranscriptCell",
    "TranscriptScroll",
    "UvAgentApp",
    "WorktreeBranchPanel",
]


def create_engine(project_root: Path | None = None, *, data_dir: Path | None = None):
    """Create the agent engine without importing it at TUI module import time.

    Importing ``uv_agent.app_factory`` pulls in the full agent/model stack. This
    wrapper keeps the existing test monkeypatch seam
    (``uv_agent.tui.app.create_engine``) while allowing the Textual app module
    itself to load with fewer provider/MCP dependencies before first paint.
    """

    from uv_agent.app_factory import create_engine as _create_engine

    return _create_engine(project_root, data_dir=data_dir)


def _markdown(text: str):
    """Return a Rich Markdown renderable, importing Markdown on demand.

    Markdown rendering is only needed once assistant/history text exists. The
    import pulls markdown-it/Pygments through Rich, so deferring it keeps the
    first empty composer screen lighter without changing transcript rendering.
    The leading underscore avoids colliding with common ``markdown`` symbols
    other modules might re-export here in the future.
    """

    from rich.markdown import Markdown

    return Markdown(text)


def save_clipboard_image(target_dir: Path):
    """Save a clipboard image while preserving the historical patch seam.

    Tests and external embedders monkeypatch ``uv_agent.tui.app.save_clipboard_image``.
    Keeping this lightweight wrapper avoids importing Pillow at TUI startup while
    preserving that module-level hook.
    """

    from uv_agent.clipboard import save_clipboard_image as _save_clipboard_image

    return _save_clipboard_image(target_dir)


COMMAND_SPECS = [
    ("/clear", "/clear"),
    ("/status", "/status"),
    ("/level", "/level"),
    ("/threads", "/threads"),
    ("/goal", "/goal"),
    ("/config", "/config"),
    ("/cancel", "/cancel"),
    ("/quit", "/quit"),
    ("/help", "/help"),
]





def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _elapsed_between(started_at: str | None, ended_at: str | None = None) -> float | None:
    started = _parse_iso_datetime(started_at)
    if started is None:
        return None
    ended = _parse_iso_datetime(ended_at) or datetime.now(UTC)
    return max(0.0, (ended - started).total_seconds())


def _timeline_text(content: object) -> str:
    """Return timeline text while allowing live items to store text chunks."""

    if isinstance(content, list):
        return "".join(str(part) for part in content)
    return str(content or "")


def composer_history_path() -> Path:
    return uv_agent_home() / COMPOSER_HISTORY_FILENAME


def load_composer_history() -> list[str]:
    path = composer_history_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(raw_items, list):
        return []
    items: list[str] = []
    for value in raw_items:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        if items and items[-1] == text:
            continue
        items.append(text)
    return items[-MAX_COMPOSER_HISTORY:]


def save_composer_history(items: list[str]) -> None:
    path = composer_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {"items": items[-MAX_COMPOSER_HISTORY:]}
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    atomic_replace(tmp_path, path)


def _event_id(event: dict[str, Any] | None) -> int | None:
    if not event:
        return None
    value = event.get("_event_id")
    return value if isinstance(value, int) else None


class TranscriptScreen(Screen[None]):
    """Default screen with tighter transcript selection behavior."""

    def _forward_event(self, event: events.Event) -> None:
        if isinstance(event, events.MouseDown) and not event.is_forwarded:
            release_overlay_capture = getattr(
                self.app,
                "_release_stale_overlay_mouse_capture",
                None,
            )
            if callable(release_overlay_capture) and release_overlay_capture(event):
                return
        super()._forward_event(event)

    def on_paste(self, event: events.Paste) -> None:
        """Forward unhandled pastes to the app-level composer fallback."""
        handler = getattr(self.app, "_handle_unfocused_composer_paste", None)
        if callable(handler):
            handler(event)

    def _watch__select_state(self, select_state: Any) -> None:
        super()._watch__select_state(select_state)
        self._tighten_transcript_selection()

    def _tighten_transcript_selection(self) -> None:
        select_state = self._select_state
        if select_state is None or select_state.end is None:
            return
        start_widget = select_state.start.content_widget
        end_widget = select_state.end.content_widget
        if not isinstance(start_widget, TranscriptCell) or start_widget is not end_widget:
            return
        start_offset = select_state.start.content_offset
        end_offset = select_state.end.content_offset
        if start_offset is None or end_offset is None:
            return
        if self.selections.get(start_widget) is None:
            return
        self._set_selection(start_widget, start_offset, end_offset)

    def _set_selection(self, widget: TranscriptCell, start: Offset, end: Offset) -> None:
        """Replace Textual's selection map for one transcript cell.

        Textual exposes ``selections`` as a reactive descriptor. Assigning via a
        tiny wrapper keeps the descriptor write in one place and makes the
        narrowed value shape obvious to static checkers.
        """

        self.selections = {
            widget: Selection.from_offsets(start, self._inclusive_selection_end(start, end))
        }

    def _inclusive_selection_end(self, start: Offset, end: Offset) -> Offset:
        if end.transpose < start.transpose:
            return end
        return end + (1, 0)


class UvAgentApp(MentionMixin, ConfigPanelMixin, ImageSupportMixin, App[None]):
    ENABLE_COMMAND_PALETTE = False
    CLICK_CHAIN_TIME_THRESHOLD = 0.25

    CSS = MAIN_APP_CSS

    BINDINGS = [
        Binding("ctrl+enter", "submit_composer", "Send", priority=True),
        Binding("ctrl+j", "submit_composer", "Send", priority=True),
        Binding("ctrl+g", "toggle_visible_process_folds", "Process", priority=True),
        Binding("tab", "toggle_composer_height", "Height", priority=True),
        Binding("ctrl+s", "toggle_status_panel", "Status", priority=True),
        Binding("ctrl+o", "open_command_palette", "Commands", priority=True),
        Binding("ctrl+d", "toggle_tool_details", "Details", priority=True),
        Binding("f2", "attach_clipboard_image", "Attach image", priority=True),
        Binding("f3", "preview_images", "Images", priority=True),
        Binding("ctrl+c", "interrupt_turn", "Interrupt", priority=True, show=False),
        Binding("enter", "focus_composer", "Focus composer", priority=True, show=False),
        Binding("f1", "help", "Help", priority=True),
        Binding("escape", "clear_input", "Clear"),
    ]

    busy = reactive(False)

    def get_default_screen(self) -> Screen:
        return TranscriptScreen(id="_default")

    def watch_busy(self, old: bool, new: bool) -> None:
        # Track when work begins so the status footer can render a live elapsed
        # timer alongside the spinner, codex-style. When the turn ends we must
        # also re-render the footer immediately; otherwise the busy branch (with
        # spinner + "Working") keeps showing until the next unrelated refresh.
        if new and not old:
            self._busy_started_at = monotonic()
        elif old and not new:
            self._busy_started_at = None
            if self.is_mounted:
                self._refresh_status()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "interrupt_turn":
            # Let focused text inputs keep the platform-standard Ctrl+C copy
            # binding only when there is an actual selection to copy. A focused
            # but unselected composer must still use Ctrl+C for interrupt/quit.
            focused = self.screen.focused if self.is_mounted else None
            return not (isinstance(focused, TextArea) and bool(focused.selected_text))
        if action == "toggle_tool_details":
            return self.is_mounted and self.screen is self.default_screen
        if action == "toggle_visible_process_folds":
            return self.is_mounted and self.screen is self.default_screen
        if action in {"attach_clipboard_image", "preview_images"}:
            return self.is_mounted and self.screen is self.default_screen
        if action == "focus_composer":
            if not self.is_mounted or self.screen is not self.default_screen:
                return False
            try:
                composer = self.query_one("#composer", TextArea)
            except NoMatches:
                return False
            return self.screen.focused is not composer
        return True

    def __init__(self, *, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.engine = create_engine(project_root)
        self.language = detect_user_language(self.engine.config.ui.language)
        self.thread_id: str | None = None
        self.level: str | None = None
        self._assistant_buffer = ""
        self._assistant_cell: TranscriptCell | None = None
        self._tool_cells: dict[str, TranscriptCell] = {}
        self._tool_started_calls: dict[str, dict[str, Any]] = {}
        self._tool_partial_payloads: dict[str, dict[str, Any]] = {}
        self._tool_delta_cells: dict[int, TranscriptCell] = {}
        self._last_status = tr(self.language, "idle")
        self._spinner_index = 0
        self._busy_started_at: float | None = None
        self._turn_started_at: str | None = None
        self._turn_completed_at: str | None = None
        self._last_tool_payload: dict[str, object] | None = None
        self._composer_history: list[str] = load_composer_history()
        self._composer_history_index: int | None = None
        self._composer_history_draft = ""
        self._quit_armed = False
        self._last_quit_request_at = 0.0
        self._transcript_has_content = False
        self._reasoning_cell: TranscriptCell | None = None
        self._reasoning_buffer = ""
        self._process_cells: list[TranscriptCell] = []
        self._process_fold_cell: FoldedProcessCell | None = None
        self._process_collapsed = False
        self._process_anchor_cell: TranscriptCell | None = None
        self._last_composer_text = ""
        self._interrupt_armed = False
        self._last_interrupt_request_at = 0.0
        self._current_worker: Worker[None] | None = None
        self._current_cancel_event: asyncio.Event | None = None
        self._thread_runs: dict[str, ThreadRunState] = {}
        self._thread_timelines: dict[str, ThreadTimelineState] = {}
        self._thread_view_states: dict[str, ThreadViewState] = {}
        self._timeline_cells: dict[str, Widget] = {}
        self._timeline_item_ids: dict[Widget, str] = {}
        self._history_before_event_id: int | None = None
        self._history_has_more = False
        self._history_more_cell: LoadOlderHistoryCell | None = None
        self._selection_copy_timer: Any | None = None
        self._pending_selection_copy = ""
        self._last_auto_copied_selection = ""
        self._composer_height_override: str | None = None
        self._composer_expanded = False
        self._pending_images_by_thread: dict[str | None, list[PendingImage]] = {}
        self._tool_delta_calls: dict[int, dict[str, Any]] = {}
        self._mention_file_cache = MentionScanCache()
        self._mention_thread_cache = MentionScanCache()
        self._mention_file_cache_dirty = False
        self._mention_file_watcher_worker: Worker[None] | None = None
        self._mention_file_watcher_stop = threading.Event()
        self._window_title_thread_title = ""
        self._last_window_title = ""
        self._thread_activity: dict[str, ThreadActivityState] = {}
        self._top_notifications: list[TopNotification] = []
        self._top_notification_unread = 0
        self._interaction_mode = "normal"
        # Enabling Goal should not create thread records or goal files until the
        # user actually sends. ``None`` represents the unsaved draft thread.
        self._pending_goal_enable_threads: set[str | None] = set()
        self._stream_render_timer: Any | None = None
        self._stream_status_timer: Any | None = None
        self._stream_render_due = False
        self._stream_status_due = False
        self._last_stream_render_at = 0.0
        self._last_stream_status_at = 0.0
        self._status_level_name = self.level or self.engine.config.runtime.default_level
        self._status_compact_context = "0%"
        self._status_thread_label = short_thread(self.thread_id)
        self._status_billing_label = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="main-column"):
            with Horizontal(id="top-bar"):
                yield Static("", id="top-bar-elapsed")
                yield Static("", id="top-bar-mode")
                yield Static("", id="top-bar-worktree", classes="hidden")
                yield Static("", id="top-bar-spacer")
                yield Static("", id="top-bar-active")
                yield Static("", id="top-bar-completed")
                yield Static("", id="top-bar-notifications")
            with TranscriptScroll(id="transcript"):
                yield EmptyState()
            yield Static("", id="pending-turns-btn", classes="hidden")
            yield Static("", id="pending-images-btn", classes="hidden")
            yield Static(
                plain(f"↓ {tr(self.language, 'back_to_bottom')}"),
                id="scroll-to-bottom-btn",
                classes="hidden",
            )
            with Vertical(id="bottom-pane"):
                with Vertical(id="composer-shell"):
                    yield ComposerTextArea(
                        "",
                        placeholder=tr(self.language, "placeholder"),
                        id="composer",
                        compact=True,
                        soft_wrap=True,
                        show_line_numbers=False,
                    )
                    yield Static("", id="composer-footer")

    def on_mount(self) -> None:
        workflow_executor = getattr(self.engine, "workflow_executor", None)
        if workflow_executor is not None:
            workflow_executor.start()
        self.query_one(EmptyState).tick()
        self._refresh_status(self._text("idle"))
        self.set_interval(0.16, self._tick, name="spinner")
        self.query_one("#composer", TextArea).focus()
        transcript = self.query_one("#transcript", TranscriptScroll)
        self.watch(transcript, "near_bottom", self._on_near_bottom_changed)
        self._refresh_pending_turns()
        self._refresh_pending_images()
        self._refresh_composer_overlay()
        self._refresh_top_bar()

    async def on_unmount(self) -> None:
        self._mention_file_watcher_stop.set()
        if self._mention_file_watcher_worker is not None:
            self._mention_file_watcher_worker.cancel()
        await self.engine.aclose()
        if self._stream_render_timer is not None:
            self._stream_render_timer.stop()
            self._stream_render_timer = None
        if self._stream_status_timer is not None:
            self._stream_status_timer.stop()
            self._stream_status_timer = None

    def _on_near_bottom_changed(self, near: bool) -> None:
        self._refresh_composer_overlay()

    def on_click(self, event: events.Click) -> None:
        if self._handle_bottom_overlay_pointer_event(event):
            return
        widget = getattr(event, "widget", None)
        if widget is not None and widget.id == "top-bar-active":
            event.stop()
            self._open_session_threads_panel("active")
        elif widget is not None and widget.id == "top-bar-completed":
            event.stop()
            self._open_session_threads_panel("completed")
        elif widget is not None and widget.id == "top-bar-worktree":
            event.stop()
            self._open_worktree_panel()
        elif widget is not None and widget.id == "top-bar-notifications":
            event.stop()
            self._open_notifications_panel()
        elif widget is not None and widget.id == "pending-turns-btn":
            event.stop()
            self._open_pending_send_queue()
        elif widget is not None and widget.id == "pending-images-btn":
            event.stop()
            self._open_pending_image_preview()
        elif widget is not None and widget.id == "scroll-to-bottom-btn":
            event.stop()
            self._scroll_transcript_to_bottom_from_overlay()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._handle_bottom_overlay_pointer_event(event)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        self._handle_bottom_overlay_pointer_event(event)

    def _handle_bottom_overlay_pointer_event(self, event: events.MouseEvent) -> bool:
        """Handle overlay controls by screen coordinates before stale capture wins.

        Textual routes mouse events to ``app.mouse_captured`` before hit-testing the
        screen. If the scrollbar capture is left behind after a drag/focus edge
        case, clicks on the visually top-most composer overlay are delivered to
        the scrollbar and stopped there. The overlay is screen-positioned, so a
        small coordinate hit test lets these controls recover from stale capture
        without changing normal transcript mouse handling.
        """

        if self.screen is not self.default_screen:
            return False
        button = self._overlay_button_at_offset(event.screen_offset)
        if button is None:
            return False
        event.stop()
        if isinstance(event, events.MouseDown):
            self._release_stale_overlay_mouse_capture(event)
            return True
        if isinstance(event, events.MouseUp):
            self._release_stale_overlay_mouse_capture(event)
            return True
        if isinstance(event, events.Click):
            self._release_stale_overlay_mouse_capture(event)
            if button.id == "pending-turns-btn":
                self._open_pending_send_queue()
            elif button.id == "pending-images-btn":
                self._open_pending_image_preview()
            elif button.id == "scroll-to-bottom-btn":
                self._scroll_transcript_to_bottom_from_overlay()
            return True
        return False

    def _release_stale_overlay_mouse_capture(self, event: events.MouseEvent) -> bool:
        if self.mouse_captured is None:
            return False
        if self._overlay_button_at_offset(event.screen_offset) is None:
            return False
        self.capture_mouse(None)
        event.stop()
        return True

    def _overlay_button_at_offset(self, offset: Offset) -> Static | None:
        for selector in (
            "#pending-turns-btn",
            "#pending-images-btn",
            "#scroll-to-bottom-btn",
        ):
            try:
                button = self.query_one(selector, Static)
            except NoMatches:
                continue
            if button.has_class("hidden"):
                continue
            if offset in button.region:
                return button
        return None

    def _scroll_transcript_to_bottom_from_overlay(self) -> None:
        try:
            transcript = self.query_one("#transcript", TranscriptScroll)
        except NoMatches:
            return
        transcript.engage_follow_tail(force=True)
        self._refresh_composer_overlay()

    def on_resize(self) -> None:
        self._refresh_status()
        self.call_after_refresh(self._resize_composer)
        self.call_after_refresh(self._refresh_composer_overlay)

    def _maximum_composer_height(self) -> int:
        try:
            transcript = self.query_one("#transcript", TranscriptScroll)
        except NoMatches:
            reserved = 1
        else:
            min_height = transcript.styles.min_height
            reserved = min_height.value if min_height is not None and min_height.value is not None else 0
        return max(COMPOSER_COLLAPSED_HEIGHT, self.size.height - int(reserved) - 1)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "composer":
            return
        previous = self._last_composer_text
        current = event.text_area.text
        if self._composer_history_index is not None and current != self._composer_history_text():
            self._reset_composer_history_navigation()
        self._last_composer_text = current
        self._resize_composer()
        self._refresh_status_from_cache()
        if current == "/" and previous == "":
            event.text_area.load_text("")
            self._last_composer_text = ""
            self._resize_composer()
            self._open_command_palette()
        elif current in {"?", "？"} and previous == "":
            event.text_area.load_text("")
            self._last_composer_text = ""
            self._resize_composer()
            self._open_help_panel()
        else:
            self._maybe_open_mention_picker(event.text_area, previous=previous, current=current)

    def on_text_area_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        if event.text_area.id != "composer":
            return
        selected_text = event.text_area.selected_text
        if not selected_text:
            self._cancel_selection_copy()
            self._last_auto_copied_selection = ""
            return
        self._schedule_selection_copy(selected_text)

    def on_text_selected(self, event: events.TextSelected) -> None:
        selected_text = self.screen.get_selected_text()
        if not selected_text:
            self._cancel_selection_copy()
            self._last_auto_copied_selection = ""
            return
        self._schedule_selection_copy(selected_text, source="screen")

    def _handle_unfocused_composer_paste(self, event: events.Paste) -> None:
        """Recover bracketed pastes that arrive while terminal focus is cleared.

        Windows Terminal shows a native confirmation dialog for large pastes. That
        dialog emits FocusOut, so Textual temporarily removes widget focus before
        the bracketed paste payload is delivered. With no focused widget, Textual
        forwards the Paste event to the screen and the composer never sees it.
        If the composer was the widget blurred by AppBlur, treat the paste as
        intended for the composer and insert it exactly like TextArea's paste
        handler would.
        """
        if self.screen is not self.default_screen:
            return
        try:
            composer = self.query_one("#composer", TextArea)
        except NoMatches:
            return

        # Avoid duplicating normal paste handling. When the composer is still
        # focused, Textual routes the Paste event there first; it bubbles up to
        # the app only after TextArea has already inserted the text.
        if self.screen.focused is composer:
            return
        if self.screen.focused is not None:
            return
        if self._last_focused_on_app_blur is not composer:
            return

        event.stop()

        # A Paste event is terminal input, just like a key press, but Textual's
        # App.on_event only treats Key/MouseDown as focus-restoring input. Mark
        # the app focused now so bindings/styles recover even if FocusIn arrives
        # after the paste payload.
        if not self.app_focus:
            self.app_focus = True

        previous = self._last_composer_text
        result = composer.replace(event.text, *composer.selection, maintain_selection_offset=False)
        composer.move_cursor(result.end_location)

        # TextArea.Changed is not delivered for this fallback path because the
        # paste is handled by the screen while no widget has focus. Apply the
        # same composer bookkeeping that normally happens in
        # on_text_area_changed(), otherwise large multi-line pastes remain in a
        # collapsed composer until the next edit.
        current = composer.text
        if self._composer_history_index is not None and current != self._composer_history_text():
            self._reset_composer_history_navigation()
        self._last_composer_text = current
        self._resize_composer()
        self._refresh_status_from_cache()
        self._maybe_open_mention_picker(composer, previous=previous, current=current)
        composer.focus()

    def _tick(self) -> None:
        if not self._transcript_has_content:
            try:
                self.query_one(EmptyState).tick()
            except NoMatches:
                pass
        if self.busy or self._any_thread_running():
            self._spinner_index += 1
        if self.busy:
            self._refresh_busy_status()
        else:
            self._apply_window_title()

    def _text(self, key: str) -> str:
        return tr(self.language, key)

    def _timeline_for_thread(self, thread_id: str | None = None) -> ThreadTimelineState | None:
        resolved = thread_id or self.thread_id or "__draft__"
        timeline = self._thread_timelines.get(resolved)
        if timeline is None:
            timeline = ThreadTimelineState(resolved)
            self._thread_timelines[resolved] = timeline
        return timeline

    def _timeline_for_active(self) -> ThreadTimelineState | None:
        return self._timeline_for_thread(self.thread_id)

    def _save_active_thread_view_state(self) -> None:
        if not self.thread_id or not self._default_screen_mounted():
            return
        try:
            transcript = self.query_one("#transcript", TranscriptScroll)
            composer = self.query_one("#composer", TextArea)
        except NoMatches:
            return
        fold_state: dict[str, bool] = {}
        for child in transcript.children:
            if isinstance(child, FoldedProcessCell):
                group_id = getattr(child, "timeline_group_id", "")
                if isinstance(group_id, str) and group_id:
                    fold_state[group_id] = child.collapsed
        focused_item_id = None
        focused = self.screen.focused
        if isinstance(focused, Widget):
            value = self._timeline_item_ids.get(focused)
            focused_item_id = value if isinstance(value, str) else None
        self._thread_view_states[self.thread_id] = ThreadViewState(
            scroll_y=transcript.scroll_y,
            follow_tail=transcript.follow_tail,
            fold_collapsed=fold_state,
            composer_draft=composer.text,
            focused_item_id=focused_item_id,
        )

    def _restore_thread_view_state(self, thread_id: str) -> None:
        view = self._thread_view_states.get(thread_id)
        try:
            transcript = self.query_one("#transcript", TranscriptScroll)
            composer = self.query_one("#composer", TextArea)
        except NoMatches:
            return
        if view is None:
            composer.load_text("")
            self._last_composer_text = ""
            self._resize_composer()
            self._scroll_end()
            return
        composer.load_text(view.composer_draft)
        self._last_composer_text = view.composer_draft
        self._resize_composer()
        transcript.follow_tail = view.follow_tail

        def _restore() -> None:
            if view.follow_tail:
                transcript.programmatic_scroll_end(force=True)
            else:
                transcript.scroll_y = transcript.validate_scroll_y(view.scroll_y)
                transcript._recompute_near_bottom()
                # Setting scroll_y may synchronously recompute near-bottom and
                # re-enable follow mode. Restored per-thread view state is more
                # specific than that generic watcher, so let the saved value win.
                transcript.follow_tail = False
            focused = self._timeline_cells.get(view.focused_item_id or "")
            if isinstance(focused, Widget) and focused.is_mounted:
                focused.focus()

        transcript.call_after_refresh(_restore)

    def _active_fold_state(self) -> dict[str, bool]:
        if self.thread_id and self.thread_id in self._thread_view_states:
            return dict(self._thread_view_states[self.thread_id].fold_collapsed)
        return {}

    def _sync_transcript_from_timeline(self, *, restore_view: bool = False) -> None:
        timeline = self._timeline_for_active()
        if timeline is None:
            return
        try:
            transcript = self.query_one("#transcript", TranscriptScroll)
        except NoMatches:
            return

        # Full rebuilds are still the clearest path for history loads, thread
        # switches, and structural changes such as inserting/removing old items.
        # The hot live path below handles append/update-only streams without
        # tearing down the whole widget tree on every token/tool partial.
        if not restore_view and self._can_patch_timeline_incrementally(timeline):
            self._patch_transcript_from_timeline(timeline, transcript)
            return

        self._rebuild_transcript_from_timeline(
            timeline,
            transcript,
            restore_view=restore_view,
        )

    def _timeline_widget_is_live(self, widget: Widget | None) -> bool:
        """Return whether a timeline widget can be patched in place.

        Textual mounts children asynchronously: immediately after ``mount()`` a
        widget can already have a parent, yet ``is_mounted`` remains false until
        the next message-pump pass. Live stream events can arrive faster than
        that, so treating parented widgets as absent makes the hot path rebuild
        the whole transcript repeatedly.
        """

        return isinstance(widget, Widget) and (
            widget.is_mounted
            or (widget.parent is not None and not bool(getattr(widget, "_closing", False)))
        )

    def _can_patch_timeline_incrementally(self, timeline: ThreadTimelineState) -> bool:
        if self._history_more_cell is not None:
            return False
        item_ids = self._timeline_cell_item_ids()
        if len(timeline.items) < len(item_ids):
            return False
        # This predicate runs on the live streaming path. Building a full list of
        # timeline ids for every token made each delta O(history) before the
        # actual patcher even ran. Compare only the mounted prefix instead.
        for index, item_id in enumerate(item_ids):
            if timeline.items[index].id != item_id:
                return False
        if len(set(item_ids)) != len(item_ids):
            return False
        for item_id in item_ids:
            cell = self._timeline_cells.get(item_id)
            if not self._timeline_widget_is_live(cell):
                return False
        return True

    def _patch_transcript_from_timeline(
        self,
        timeline: ThreadTimelineState,
        transcript: TranscriptScroll,
    ) -> None:
        follow_tail = transcript.follow_tail
        previous_scroll_y = transcript.scroll_y
        changed_item_ids, changed_groups = timeline.consume_changes()
        force_item_update = bool(changed_item_ids)
        if not changed_item_ids and not changed_groups:
            changed_item_ids = {item.id for item in timeline.items}

        items_by_id = {item.id: item for item in timeline.items}
        stale_ids = [item_id for item_id in self._timeline_cell_item_ids() if item_id not in items_by_id]
        if stale_ids:
            self._rebuild_transcript_from_timeline(timeline, transcript)
            return

        for item_id in changed_item_ids:
            item = items_by_id.get(item_id)
            if item is None:
                continue
            old_group = self._sync_timeline_item_widget(item, force_update=force_item_update)
            if item.process_group:
                changed_groups.add(item.process_group)
            if old_group and old_group != item.process_group:
                changed_groups.add(old_group)

        for group_id in changed_groups:
            self._refresh_timeline_process_fold(timeline, group_id)

        if timeline.items:
            self._mark_transcript_content()
        else:
            self._reset_transcript(show_empty=True)
            return
        self._history_before_event_id = timeline.loaded_start_event_id
        self._history_has_more = timeline.has_older
        self._history_more_cell = None
        transcript.follow_tail = follow_tail
        if follow_tail:
            self._scroll_end()
        else:
            transcript.scroll_y = transcript.validate_scroll_y(previous_scroll_y)
            transcript._recompute_near_bottom()

    def _rebuild_transcript_from_timeline(
        self,
        timeline: ThreadTimelineState,
        transcript: TranscriptScroll,
        *,
        restore_view: bool = False,
    ) -> None:
        follow_tail = transcript.follow_tail
        previous_scroll_y = transcript.scroll_y
        transcript.query("*").remove()
        self._timeline_cells.clear()
        self._timeline_item_ids.clear()
        self._assistant_cell = None
        self._assistant_buffer = ""
        self._reasoning_cell = None
        self._reasoning_buffer = ""
        self._process_cells = []
        self._process_fold_cell = None
        self._process_collapsed = False
        self._process_anchor_cell = None
        self._transcript_has_content = False
        self._history_before_event_id = timeline.loaded_start_event_id
        self._history_has_more = timeline.has_older
        self._history_more_cell = None
        if timeline.has_older:
            marker = LoadOlderHistoryCell(has_more=True, classes="event")
            transcript.mount(marker)
            self._history_more_cell = marker
        group_order: list[str] = []
        for item in timeline.items:
            if item.process_group and item.process_group not in group_order:
                group_order.append(item.process_group)
            self._mount_timeline_item(item)
        # Saved view state is a one-shot restore concern.  During normal live
        # updates the timeline group is the source of truth; reapplying an older
        # ThreadViewState on every sync makes a user-toggled fold snap back on
        # the next stream event or queued-turn refresh.
        fold_state = self._active_fold_state() if restore_view else {}
        for group_id in group_order:
            group = timeline.process_groups.get(group_id)
            if group is None or not group.item_ids:
                continue
            if restore_view:
                group.collapsed = fold_state.get(group.id, group.collapsed)
            self._mount_timeline_process_fold(timeline, group_id)
        timeline.consume_changes()
        if timeline.items:
            self._mark_transcript_content()
        else:
            self._reset_transcript(show_empty=True)
            return
        if restore_view and self.thread_id:
            self._restore_thread_view_state(self.thread_id)
            return
        transcript.follow_tail = follow_tail
        if follow_tail:
            self._scroll_end()
        else:
            transcript.scroll_y = transcript.validate_scroll_y(previous_scroll_y)
            transcript._recompute_near_bottom()

    def _timeline_cell_item_ids(self) -> list[str]:
        return [
            item_id
            for item_id in self._timeline_cells
            if not item_id.startswith("process_fold:")
        ]

    def _timeline_item_signature(self, item: TimelineItem) -> tuple[Any, ...]:
        """Return a cheap widget freshness key for timeline updates.

        The live streaming path calls this often. Avoid serializing ever-growing
        assistant text, tool arguments, or partial stdout on every flush; length
        plus a tail slice is enough to detect the append-only updates we render.
        Stable events still fall back to full JSON so history/final content keeps
        exact change detection.
        """

        if item.kind in {"assistant", "reasoning"} and isinstance(item.content, dict):
            text = item.content.get("text")
            if isinstance(text, list):
                return (
                    item.kind,
                    item.process_group,
                    bool(item.content.get("partial")),
                    len(text),
                    text[-1] if text else "",
                )
        if item.kind == "tool_call" and isinstance(item.content, dict):
            call = item.content.get("call") if isinstance(item.content.get("call"), dict) else {}
            status = str(item.content.get("status") or "")
            if status != "called" and isinstance(call, dict):
                arguments = str(call.get("arguments") or "")
                return (
                    item.kind,
                    item.process_group,
                    status,
                    str(call.get("call_id") or ""),
                    str(call.get("name") or ""),
                    len(arguments),
                    arguments[-120:],
                )
        if item.kind == "tool_result" and isinstance(item.content, dict):
            payload = item.content.get("payload") if isinstance(item.content.get("payload"), dict) else {}
            if isinstance(payload, dict) and payload.get("partial"):
                stdout = str(payload.get("stdout") or "")
                stderr = str(payload.get("stderr") or "")
                events = payload.get("events")
                if not isinstance(events, list):
                    events = []
                return (
                    item.kind,
                    item.process_group,
                    "partial",
                    str(payload.get("run_id") or ""),
                    str(payload.get("call_id") or ""),
                    payload.get("returncode"),
                    len(stdout),
                    stdout[-120:],
                    len(stderr),
                    stderr[-120:],
                    len(events),
                    repr(events[-1])[:240] if events else "",
                )
        return (
            item.kind,
            item.process_group,
            json.dumps(item.content, sort_keys=True, default=str),
        )

    def _set_timeline_widget_metadata(self, widget: Widget, item: TimelineItem) -> None:
        self._timeline_cells[item.id] = widget
        self._timeline_item_ids[widget] = item.id
        setattr(widget, "timeline_process_group", item.process_group)
        setattr(widget, "timeline_signature", self._timeline_item_signature(item))

    def _rebind_timeline_widget_id(self, old_item_id: str, item: TimelineItem) -> None:
        """Move mounted widget metadata after a timeline item id changes."""

        if old_item_id == item.id or old_item_id not in self._timeline_cells:
            return
        widget = self._timeline_cells[old_item_id]
        replaced = self._timeline_cells.get(item.id)
        if isinstance(replaced, Widget) and replaced is not widget:
            self._timeline_item_ids.pop(replaced, None)
        rebound: dict[str, Widget] = {}
        for key, value in self._timeline_cells.items():
            if key == old_item_id:
                rebound[item.id] = value
            elif key != item.id:
                rebound[key] = value
        self._timeline_cells.clear()
        self._timeline_cells.update(rebound)
        self._timeline_item_ids[widget] = item.id
        setattr(widget, "timeline_process_group", item.process_group)
        setattr(widget, "timeline_signature", self._timeline_item_signature(item))

    def _sync_timeline_item_widget(self, item: TimelineItem, *, force_update: bool = False) -> str | None:
        existing = self._timeline_cells.get(item.id)
        old_group = getattr(existing, "timeline_process_group", None) if isinstance(existing, Widget) else None
        signature = None if force_update else self._timeline_item_signature(item)
        if self._timeline_widget_is_live(existing):
            if not force_update and getattr(existing, "timeline_signature", None) == signature:
                setattr(existing, "timeline_process_group", item.process_group)
                return old_group if isinstance(old_group, str) else None
            if self._update_timeline_item_widget(existing, item):
                self._set_timeline_widget_metadata(existing, item)
                return old_group if isinstance(old_group, str) else None
            replacement = self._mount_timeline_item(item, before=existing)
            self._timeline_item_ids.pop(existing, None)
            existing.remove()
            if replacement is None:
                self._timeline_cells.pop(item.id, None)
            return old_group if isinstance(old_group, str) else None
        if isinstance(existing, Widget):
            self._timeline_item_ids.pop(existing, None)
        self._timeline_cells.pop(item.id, None)
        self._mount_timeline_item(item)
        return old_group if isinstance(old_group, str) else None

    def _update_timeline_item_widget(self, widget: Widget, item: TimelineItem) -> bool:
        if item.kind == "user":
            if not isinstance(widget, TranscriptCell) or isinstance(widget, (ExpandableTranscriptCell, FoldedProcessCell)):
                return False
            label = "你" if self.language.is_chinese else "you"
            text = _timeline_text(item.content.get("text"))
            widget.update(join_lines([Text(f"› {label}", style="bold #7dd3fc"), plain(text)]))
            return True
        if item.kind == "assistant":
            if not isinstance(widget, TranscriptCell) or isinstance(widget, (ExpandableTranscriptCell, FoldedProcessCell)):
                return False
            text = _timeline_text(item.content.get("text"))
            if item.content.get("partial"):
                widget.update(plain(text, style="#d7e0ea"), copy_text=text)
                return True
            widget.update(_markdown(text), copy_text=text)
            return True
        if item.kind == "reasoning":
            text = _timeline_text(item.content.get("text"))
            if item.content.get("partial"):
                if not isinstance(widget, TranscriptCell) or isinstance(widget, (ExpandableTranscriptCell, FoldedProcessCell)):
                    return False
                first = text.strip().splitlines()[0] if text.strip() else ""
                if len(first) > 120:
                    first = first[:117].rstrip() + "..."
                widget.update(Text.assemble(
                    (self._text("thinking"), "dim italic"),
                    "  ",
                    (first, "italic #a3b1c2"),
                ))
                return True
            if not isinstance(widget, ExpandableTranscriptCell):
                return False
            stripped = text.strip()
            if not stripped:
                return False
            summary, details = self._reasoning_markup(stripped)
            widget.set_details(summary, details)
            return True
        if item.kind == "tool_call":
            if not isinstance(widget, ExpandableTranscriptCell):
                return False
            call = dict(item.content.get("call") or {})
            status = str(item.content.get("status") or "called")
            status_label = str(
                call.get("_status_label")
                or (self._text("python_called") if status == "called" else self._text("python_running"))
            )
            if status != "called":
                # The streamed tool-call arguments may update every few bytes.
                # Re-highlighting the growing details pane on each delta makes
                # live JSON/function-call output increasingly expensive. Keep
                # the compact row fresh and defer the full highlighted body to
                # tool.started/tool.output, where the call is stable.
                widget.update(tool_call_summary_markup({**call, "_status_label": status_label}))
                widget.update_copy_text(renderable_plain(tool_call_detail_highlight_markup(call)))
                if widget.has_class("event"):
                    widget.remove_class("event")
                if not widget.has_class("tool_pending"):
                    widget.add_class("tool_pending")
                return True
            widget.set_details(
                tool_call_summary_markup({**call, "_status_label": status_label}),
                tool_call_detail_highlight_markup(call),
            )
            if status == "called":
                if widget.has_class("tool_pending"):
                    widget.remove_class("tool_pending")
                if not widget.has_class("event"):
                    widget.add_class("event")
            else:
                if widget.has_class("event"):
                    widget.remove_class("event")
                if not widget.has_class("tool_pending"):
                    widget.add_class("tool_pending")
            return True
        if item.kind == "tool_result":
            if not isinstance(widget, ExpandableTranscriptCell):
                return False
            payload = dict(item.content.get("payload") or {})
            widget.set_details(tool_timeline_markup(payload), tool_detail_markup(payload))
            widget.tool_payload = payload
            self._last_tool_payload = payload
            return True
        if item.kind == "image":
            if not isinstance(widget, ImageAttachmentCell):
                return False
            widget.attachment = dict(item.content.get("attachment") or {})
            widget._refresh_content()
            return True
        if item.kind == "compaction":
            if not isinstance(widget, ExpandableTranscriptCell):
                return False
            summary_text = _timeline_text(item.content.get("text")).strip()
            if not summary_text:
                return False
            summary = Text.assemble(
                (self._text("compacted"), "dim"),
                " · ",
                (self._text("compacted_summary_hint"), "cyan dim"),
            )
            widget.detail_title = "compaction_summary"
            widget.detail_hint = "compaction_summary_hint"
            widget.set_details(summary, plain(summary_text))
            self._process_anchor_cell = widget
            return True
        if item.kind == "warning":
            if not isinstance(widget, TranscriptCell) or isinstance(widget, FoldedProcessCell):
                return False
            event = dict(item.content.get("event") or {})
            if item.content.get("warning_kind") == "model_switch":
                message = str(event.get("message") or self._text("model_switch_warning"))
                from_level = str(event.get("from_level") or "")
                to_level = str(event.get("to_level") or "")
                content = Text(message, style="yellow")
                if from_level or to_level:
                    content.append("\n")
                    content.append(f"{from_level or '?'} -> {to_level or '?'}", style="dim")
                widget.update(content, copy_text=message)
            else:
                message = str(event.get("message") or self._text("token_estimation_warning"))
                widget.update(Text.assemble(("⚠ ", "yellow"), (message, "yellow")), copy_text=message)
            return True
        if item.kind == "error":
            if not isinstance(widget, TranscriptCell) or isinstance(widget, FoldedProcessCell):
                return False
            event = dict(item.content.get("event") or {})
            error_type = str(event.get("error_type") or "Turn error")
            message = str(event.get("message") or "The turn stopped before producing a final response.")
            retryable = self._is_retryable_error_event(event)
            hint = self._text("retry_network_error_hint") if retryable else self._text("thread_stopped_after_error")
            widget.update(
                join_lines([Text.assemble((error_type, "bold red"), " ", message), plain(hint, style="dim")]),
                copy_text=f"{error_type}: {message}\n{hint}",
            )
            return True
        if item.kind == "stream_retry":
            if not isinstance(widget, TranscriptCell) or isinstance(widget, FoldedProcessCell):
                return False
            event = dict(item.content.get("event") or {})
            attempt = event.get("attempt")
            max_attempts = event.get("max_attempts")
            delay_s = event.get("delay_s")
            error_type = str(event.get("error_type") or "stream")
            try:
                if delay_s is None:
                    raise TypeError("delay_s is missing")
                delay_text = f"{float(delay_s):.1f}s"
            except (TypeError, ValueError):
                delay_text = "?s"
            widget.update(plain(
                f"⟳ stream empty, retrying {attempt or '?'}/{max_attempts or '?'} "
                f"in {delay_text} ({error_type})",
                style="dim",
            ))
            return True
        if item.kind == "queued":
            if not isinstance(widget, TranscriptCell) or isinstance(widget, FoldedProcessCell):
                return False
            widget.update(self._queued_turn_markup(
                str(item.content.get("prompt") or ""),
                list(item.content.get("image_paths") or []),
            ))
            return True
        return False

    def _timeline_process_cells(self, timeline: ThreadTimelineState, group_id: str) -> list[TranscriptCell]:
        group = timeline.process_groups.get(group_id)
        if group is None or not group.item_ids:
            return []
        process_cells: list[TranscriptCell] = []
        seen: set[Widget] = set()
        for item_id in group.item_ids:
            cell = self._timeline_cells.get(item_id)
            if not isinstance(cell, TranscriptCell) or isinstance(cell, FoldedProcessCell):
                continue
            if cell in seen:
                continue
            seen.add(cell)
            process_cells.append(cell)
        return process_cells

    def _mount_timeline_process_fold(self, timeline: ThreadTimelineState, group_id: str) -> FoldedProcessCell | None:
        group = timeline.process_groups.get(group_id)
        process_cells = self._timeline_process_cells(timeline, group_id)
        if group is None or not process_cells:
            return None
        elapsed_label = self._process_elapsed_label(started_at=group.started_at, completed_at=group.completed_at)
        fold = self._append_process_fold_cell(
            process_cells,
            collapsed=group.collapsed,
            elapsed_label=elapsed_label,
            timeline_group_id=group_id,
        )
        self._timeline_cells[f"process_fold:{group_id}"] = fold
        self._timeline_item_ids[fold] = f"process_fold:{group_id}"
        return fold

    def _refresh_timeline_process_fold(self, timeline: ThreadTimelineState, group_id: str) -> None:
        group = timeline.process_groups.get(group_id)
        key = f"process_fold:{group_id}"
        existing = self._timeline_cells.get(key)
        process_cells = self._timeline_process_cells(timeline, group_id)
        if group is None or not process_cells:
            if isinstance(existing, Widget):
                existing.remove()
                self._timeline_item_ids.pop(existing, None)
            self._timeline_cells.pop(key, None)
            return
        elapsed_label = self._process_elapsed_label(started_at=group.started_at, completed_at=group.completed_at)
        if isinstance(existing, FoldedProcessCell) and self._timeline_widget_is_live(existing):
            existing.set_cells(process_cells)
            existing.set_elapsed_label(elapsed_label)
            first_cell = process_cells[0]
            if self._timeline_widget_is_live(first_cell):
                children = list(first_cell.parent.children) if first_cell.parent is not None else []
                fold_is_after_first_cell = (
                    existing in children
                    and first_cell in children
                    and children.index(existing) > children.index(first_cell)
                )
                if fold_is_after_first_cell:
                    first_cell.parent.move_child(existing, before=first_cell)
            if existing.collapsed != group.collapsed:
                existing.set_collapsed(group.collapsed, notify=False)
            return
        self._mount_timeline_process_fold(timeline, group_id)

    def _mount_timeline_item(self, item: TimelineItem, *, before: MountBefore = None) -> Widget | None:
        cell: Widget | None = None
        if item.kind == "user":
            cell = self._append_user(str(item.content.get("text") or ""), before=before)
        elif item.kind == "assistant":
            text = _timeline_text(item.content.get("text"))
            content = plain(text, style="#d7e0ea") if item.content.get("partial") else _markdown(text)
            cell = self._append_cell(content, "assistant", before=before, copy_text=text)
        elif item.kind == "reasoning":
            text = _timeline_text(item.content.get("text"))
            if item.content.get("partial"):
                first = text.strip().splitlines()[0] if text.strip() else ""
                if len(first) > 120:
                    first = first[:117].rstrip() + "..."
                cell = self._append_cell(
                    Text.assemble(
                        (self._text("thinking"), "dim italic"),
                        "  ",
                        (first, "italic #a3b1c2"),
                    ),
                    "reasoning",
                    before=before,
                )
            else:
                cell = self._append_reasoning_history(text, before=before)
        elif item.kind == "tool_call":
            call = dict(item.content.get("call") or {})
            status = str(item.content.get("status") or "called")
            status_label = self._text("python_called") if status == "called" else self._text("python_running")
            cell = self._append_tool_call_history({**call, "_status_label": status_label}, before=before)
            if status != "called" and isinstance(cell, TranscriptCell):
                cell.remove_class("event")
                cell.add_class("tool_pending")
        elif item.kind == "tool_result":
            payload = dict(item.content.get("payload") or {})
            cell = self._append_expandable_cell(tool_timeline_markup(payload), tool_detail_markup(payload), "event", before=before)
            cell.tool_payload = payload
        elif item.kind == "image":
            cell = self._append_image_attachment_cell(dict(item.content.get("attachment") or {}), before=before)
        elif item.kind == "compaction":
            cell = self._append_compaction_cell(dict(item.content.get("event") or {}), before=before)
            if isinstance(cell, TranscriptCell):
                self._process_anchor_cell = cell
        elif item.kind == "warning":
            event = dict(item.content.get("event") or {})
            if item.content.get("warning_kind") == "model_switch":
                cell = self._append_model_switch_warning_cell(event, before=before)
            else:
                cell = self._append_token_estimation_warning(event, before=before, flash=False)
        elif item.kind == "error":
            cell = self._append_turn_error(dict(item.content.get("event") or {}), before=before)
        elif item.kind == "stream_retry":
            cell = self._append_stream_retry(dict(item.content.get("event") or {}), before=before)
        elif item.kind == "queued":
            cell = self._append_cell(
                self._queued_turn_markup(
                    str(item.content.get("prompt") or ""),
                    list(item.content.get("image_paths") or []),
                ),
                "event",
                before=before,
            )
        if cell is not None:
            self._set_timeline_widget_metadata(cell, item)
        return cell

    def _commands(self) -> list[CommandSpec]:
        return [
            CommandSpec(name, usage, command_description(self.language, name))
            for name, usage in COMMAND_SPECS
        ]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not isinstance(event.button, RetryTurnButton):
            return
        event.stop()
        self._retry_thread(event.button.retry_thread_id or self.thread_id)

    def action_submit_composer(self) -> None:
        composer = self.query_one("#composer", TextArea)
        prompt = composer.text.strip()
        pending_images_thread_id = self.thread_id
        pending_images = list(self._pending_images)
        if not prompt and not pending_images:
            self._flash(self._text("write_first"))
            return
        if self._thread_has_blocking_error(self.thread_id):
            self._flash(self._text("thread_blocked_after_error"), severity="error")
            return
        try:
            transcript = self.query_one("#transcript", TranscriptScroll)
        except NoMatches:
            transcript = None
        if transcript is not None and transcript.near_bottom:
            transcript.follow_tail = True
        if prompt:
            self._remember_composer_input(prompt)
        self._reset_composer_history_navigation()
        composer.load_text("")
        self._last_composer_text = ""
        self._composer_height_override = None
        self._resize_composer()
        if not prompt:
            prompt = self._text("image_only_prompt")
        if "\n" not in prompt and self._handle_command(prompt):
            return
        thread_id = self._ensure_active_thread()
        image_paths = [image.path for image in pending_images]
        self._clear_pending_images_for_thread(pending_images_thread_id)
        self._refresh_pending_images()
        level = self._current_level_for_thread(thread_id)
        self._persist_thread_level(thread_id, level)
        active_run = self._active_run_state()
        if active_run is not None:
            run_state = active_run
            run_state.queue.append(QueuedTurn(prompt=prompt, level=level, image_paths=image_paths))
            timeline = self._timeline_for_thread(run_state.thread_id)
            if timeline is not None:
                timeline.add_queued_turn(run_state.queue[-1].queue_id, prompt, image_paths)
            if self._is_active_thread(run_state.thread_id):
                self._sync_transcript_from_timeline()
            self._refresh_pending_turns()
            self._refresh_status()
            return
        self._start_turn(prompt, level=level, image_paths=image_paths)

    def _start_turn(
        self,
        prompt: str,
        *,
        level: str | None = None,
        image_paths: list[Path] | None = None,
    ) -> None:
        thread_id = self._ensure_active_thread()
        level = level or self._current_level_for_thread(thread_id)
        self._persist_thread_level(thread_id, level)
        self._start_background_turn(thread_id, prompt, level=level, image_paths=image_paths)

    def _start_background_turn(
        self,
        thread_id: str,
        prompt: str,
        *,
        level: str | None,
        image_paths: list[Path] | None = None,
        queue: list[QueuedTurn] | None = None,
    ) -> None:
        cancel_event = asyncio.Event()
        started_at = utc_now_iso()
        run_state = ThreadRunState(
            thread_id=thread_id,
            worker=None,
            cancel_event=cancel_event,
            queue=list(queue or []),
            status=self._text("sending"),
            started_at=started_at,
        )
        self._thread_runs[thread_id] = run_state
        self._mark_thread_active(thread_id)
        timeline = self._timeline_for_thread(thread_id)
        pending_turn_id = f"pending:{started_at or thread_id}"
        if timeline is not None:
            timeline.ensure_user_item(pending_turn_id, prompt, created_at=started_at)
            run_state.pending_user_turn_id = pending_turn_id
        if self._is_active_thread(thread_id):
            self.query_one("#composer-shell", Vertical).add_class("busy")
            self._reset_live_view_state()
            self._interrupt_armed = False
            self._turn_started_at = started_at
            self._turn_completed_at = None
            self._sync_transcript_from_timeline()
            for image_path in image_paths or []:
                self._append_cell(
                    Text.assemble(
                        (self._text("image_pending_sent"), "dim"),
                        " ",
                        (Path(image_path).name, "cyan"),
                    ),
                    "event",
                )
            self._refresh_status(self._text("sending"))
        worker = self.run_worker(
            self._run_turn(prompt, thread_id, level=level, image_paths=list(image_paths or [])),
            exclusive=False,
            thread=False,
        )
        run_state.worker = worker
        if self._is_active_thread(thread_id):
            self._current_worker = worker
            self._current_cancel_event = cancel_event
        self._refresh_active_run_state()

    def _start_retry_turn(self, thread_id: str) -> None:
        cancel_event = asyncio.Event()
        started_at = utc_now_iso()
        run_state = ThreadRunState(
            thread_id=thread_id,
            worker=None,
            cancel_event=cancel_event,
            queue=[],
            status=self._text("sending"),
            started_at=started_at,
        )
        self._thread_runs[thread_id] = run_state
        self._mark_thread_active(thread_id)
        if self._is_active_thread(thread_id):
            self.query_one("#composer-shell", Vertical).add_class("busy")
            self._reset_live_view_state()
            self._interrupt_armed = False
            self._turn_started_at = started_at
            self._turn_completed_at = None
            self._sync_transcript_from_timeline()
            self._refresh_status(self._text("sending"))
        worker = self.run_worker(
            self._run_turn(
                "",
                thread_id,
                level=self._current_level_for_thread(thread_id),
                image_paths=[],
                retry=True,
            ),
            exclusive=False,
            thread=False,
        )
        run_state.worker = worker
        if self._is_active_thread(thread_id):
            self._current_worker = worker
            self._current_cancel_event = cancel_event
        self._refresh_active_run_state()

    async def _run_turn(
        self,
        prompt: str,
        thread_id: str,
        *,
        level: str | None,
        image_paths: list[Path],
        retry: bool = False,
    ) -> None:
        run_state = self._thread_runs[thread_id]
        try:
            if not retry:
                self._materialize_pending_goal_enable(thread_id)
            turn_kwargs: dict[str, Any] = {
                "thread_id": thread_id,
                "level": level,
                "cancel_event": run_state.cancel_event,
            }
            if not retry:
                turn_kwargs["user_text"] = prompt
            if image_paths and not retry:
                turn_kwargs["image_paths"] = image_paths
            turn_stream = self.engine.retry_turn(**turn_kwargs) if retry else self.engine.run_turn(**turn_kwargs)
            async for item in turn_stream:
                await self._handle_engine_stream_item(item, thread_id, run_state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = format_error(exc)
            retryable = is_retryable_provider_error(exc)
            item = {
                "type": "turn.error",
                "thread_id": thread_id,
                "turn_id": run_state.turn_id,
                "created_at": utc_now_iso(),
                "completed_at": utc_now_iso(),
                "error_type": exc.__class__.__name__,
                "message": str(exc) or repr(exc),
                "retryable": retryable,
            }
            self._mark_run_error_state(run_state, item)
            self._record_live_thread_event(run_state, "turn.error", item)
            run_state.status = self._text("error")
            if self._is_active_thread(thread_id):
                self._flush_pending_stream_retries(run_state)
                self._append_turn_error(item, display_content=error_renderable(error))
                self._refresh_status(self._text("error"))
        finally:
            if self._is_active_thread(thread_id):
                self._flush_stream_updates()
            next_turn = run_state.queue.pop(0) if run_state.queue else None
            if next_turn is not None:
                remaining_queue = list(run_state.queue)
                if self._is_active_thread(thread_id):
                    self.thread_id = thread_id
                self._start_background_turn(
                    thread_id,
                    next_turn.prompt,
                    level=next_turn.level,
                    image_paths=next_turn.image_paths,
                    queue=remaining_queue,
                )
                if self._default_screen_mounted():
                    self._refresh_pending_turns()
            else:
                keep_state = run_state.retryable_error or run_state.terminal_error
                self._mark_thread_inactive(thread_id, completed=not keep_state)
                if keep_state:
                    run_state.worker = None
                    run_state.cancel_event = asyncio.Event()
                    if self._is_active_thread(thread_id):
                        self._sync_transcript_from_timeline()
                else:
                    self._thread_runs.pop(thread_id, None)
                if self._is_active_thread(thread_id) and self._default_screen_mounted():
                    self._current_worker = None
                    self._current_cancel_event = None
                    self._interrupt_armed = False
                    if not keep_state and self._last_status != self._text("error"):
                        self._refresh_status(self._text("idle"))
                    self.query_one("#composer", TextArea).focus()
                if self._default_screen_mounted():
                    self._refresh_active_run_state()
                    self._refresh_pending_turns()

    async def _handle_engine_stream_item(
        self,
        item: dict[str, Any],
        default_thread_id: str,
        run_state: ThreadRunState,
    ) -> None:
        item_thread_id = str(item.get("thread_id") or default_thread_id)
        event_type = item["type"]
        if event_type in {
            "image.attachment",
            "assistant.delta",
            "assistant.reasoning_delta",
            "tool.delta",
            "tool.started",
            "tool.output",
            "tool.partial",
            "model.stream_retry",
            "thread.token_estimation_warning",
            "compaction.started",
            "compaction.completed",
        }:
            await self._handle_thread_event(item_thread_id, event_type, item, run_state)
            return
        if event_type == "model.response":
            await self._handle_model_response_item(item_thread_id, item, run_state)
            billing_charge = item.get("billing_charge")
            if isinstance(billing_charge, dict):
                self._handle_billing_updated_item(item_thread_id, billing_charge, run_state)
            return
        if event_type == "thread.title":
            if self._is_active_thread(item_thread_id):
                self._refresh_status(self._text("idle"))
            return
        if event_type == "turn.completed":
            self._handle_turn_completed_item(item_thread_id, item, run_state)
            return
        if event_type == "turn.interrupted":
            self._handle_turn_interrupted_item(item_thread_id, item, run_state)
            return
        if event_type == "turn.error":
            self._handle_turn_error_item(item_thread_id, item, run_state)

    async def _handle_model_response_item(
        self,
        item_thread_id: str,
        item: dict[str, Any],
        run_state: ThreadRunState,
    ) -> None:
        response = item.get("response")
        reasoning_text = str(
            item.get("reasoning_text")
            or getattr(response, "reasoning_text", "")
            or ""
        )
        output = list(getattr(response, "output", []) or [])
        has_tool_call = any(isinstance(entry, dict) and entry.get("type") == "function_call" for entry in output)
        timeline = self._timeline_for_thread(item_thread_id)
        if timeline is not None:
            if reasoning_text:
                timeline.apply_live_event(
                    "assistant.reasoning_completed",
                    {
                        "type": "assistant.reasoning_completed",
                        "thread_id": item_thread_id,
                        "turn_id": item.get("turn_id"),
                        "turn_started_at": item.get("turn_started_at"),
                        "text": reasoning_text,
                    },
                )
            else:
                timeline.apply_live_event(
                    "assistant.reasoning_absent",
                    {
                        "type": "assistant.reasoning_absent",
                        "thread_id": item_thread_id,
                        "turn_id": item.get("turn_id"),
                        "turn_started_at": item.get("turn_started_at"),
                    },
                )
            turn_id = str(item.get("turn_id") or "").strip()
            acc = timeline._turn(turn_id) if turn_id else None
            assistant_text = acc.assistant_buffer if acc is not None else ""
            if has_tool_call:
                timeline.apply_live_event(
                    "assistant.response_with_tools",
                    {
                        "type": "assistant.response_with_tools",
                        "thread_id": item_thread_id,
                        "turn_id": item.get("turn_id"),
                        "turn_started_at": item.get("turn_started_at"),
                        "assistant_text": assistant_text,
                    },
                )
            else:
                if "assistant_text" not in item:
                    item["assistant_text"] = assistant_text
                timeline.apply_live_event(
                    "assistant.final_response_started",
                    {
                        "type": "assistant.final_response_started",
                        "thread_id": item_thread_id,
                        "turn_id": item.get("turn_id"),
                        "turn_started_at": item.get("turn_started_at"),
                        "assistant_text": item.get("assistant_text"),
                    },
                )
            if self._is_active_thread(item_thread_id):
                self._flush_stream_updates()
                self._sync_transcript_from_timeline()
        run_state.status = self._text("reading")
        if self._is_active_thread(item_thread_id):
            self._refresh_status(self._text("reading"))

    def _handle_turn_completed_item(
        self,
        item_thread_id: str,
        item: dict[str, Any],
        run_state: ThreadRunState,
    ) -> None:
        self._update_turn_timestamps(item, run_state)
        timeline = self._timeline_for_thread(item_thread_id)
        if timeline is not None:
            timeline.apply_live_event("turn.completed", item)
        text = item["final_text"] or ""
        was_active_thread = self._is_active_thread(item_thread_id)
        run_state.status = self._text("idle")
        self._notify_turn_completed(item_thread_id, text, active_thread=was_active_thread)
        if was_active_thread:
            self._flush_stream_updates()
            self._sync_transcript_from_timeline()
            self._refresh_status(self._text("idle"))

    def _handle_billing_updated_item(
        self,
        item_thread_id: str,
        item: dict[str, Any],
        run_state: ThreadRunState,
    ) -> None:
        if self._is_active_thread(item_thread_id):
            self._refresh_status()


    def _handle_turn_interrupted_item(
        self,
        item_thread_id: str,
        item: dict[str, Any],
        run_state: ThreadRunState,
    ) -> None:
        self._update_turn_timestamps(item, run_state)
        timeline = self._timeline_for_thread(item_thread_id)
        if timeline is not None:
            timeline.apply_live_event("turn.interrupted", item)
        run_state.status = self._text("interrupted")
        if self._is_active_thread(item_thread_id):
            self._flush_stream_updates()
            self._sync_transcript_from_timeline()
            self._refresh_status(self._text("interrupted"))

    def _handle_turn_error_item(
        self,
        item_thread_id: str,
        item: dict[str, Any],
        run_state: ThreadRunState,
    ) -> None:
        self._update_turn_timestamps(item, run_state)
        self._mark_run_error_state(run_state, item)
        timeline = self._timeline_for_thread(item_thread_id)
        if timeline is not None:
            timeline.flush_pending_stream_retries(str(item.get("turn_id") or run_state.turn_id or ""))
            timeline.apply_live_event("turn.error", item)
        if self._is_active_thread(item_thread_id):
            self._flush_stream_updates()
            self._sync_transcript_from_timeline()
            self._refresh_status(self._text("error"))

    def action_clear_input(self) -> None:
        composer = self.query_one("#composer", TextArea)
        if composer.text:
            self._reset_composer_history_navigation()
            composer.load_text("")
            self._last_composer_text = ""
            self._composer_height_override = None
            self._resize_composer()
            return

    def _remember_composer_input(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self._composer_history and self._composer_history[-1] == text:
            return
        self._composer_history.append(text)
        overflow = len(self._composer_history) - MAX_COMPOSER_HISTORY
        if overflow > 0:
            del self._composer_history[:overflow]
        try:
            save_composer_history(self._composer_history)
        except OSError as exc:
            self._flash(str(exc), severity="error")

    def _handle_composer_history_key(self, composer: TextArea, key: str) -> bool:
        if key == "up":
            if not self._composer_history:
                return False
            if self._composer_history_index is None:
                if composer.text:
                    return False
                self._composer_history_draft = composer.text
                self._composer_history_index = len(self._composer_history) - 1
            else:
                self._composer_history_index = max(0, self._composer_history_index - 1)
            self._load_composer_history_text(composer, self._composer_history_text())
            return True

        if key == "down" and self._composer_history_index is not None:
            if self._composer_history_index >= len(self._composer_history) - 1:
                draft = self._composer_history_draft
                self._reset_composer_history_navigation()
                self._load_composer_history_text(composer, draft)
            else:
                self._composer_history_index += 1
                self._load_composer_history_text(composer, self._composer_history_text())
            return True

        return False

    def _composer_history_text(self) -> str:
        if self._composer_history_index is None:
            return self._composer_history_draft
        return self._composer_history[self._composer_history_index]

    def _load_composer_history_text(self, composer: TextArea, text: str) -> None:
        composer.load_text(text)
        composer.cursor_location = composer.document.end
        self._last_composer_text = text
        self._resize_composer()
        self._refresh_status()

    def _reset_composer_history_navigation(self) -> None:
        self._composer_history_index = None
        self._composer_history_draft = ""

    def _active_run_state(self) -> ThreadRunState | None:
        if self.thread_id is None:
            return None
        run_state = self._thread_runs.get(self.thread_id)
        if run_state is not None and run_state.worker is None:
            return None
        return run_state

    @property
    def _pending_images(self) -> list[PendingImage]:
        """Composer images for the currently visible thread.

        The composer itself is shared UI, but pasted images should belong to the
        thread draft the user is looking at. Keeping the storage keyed by
        thread id prevents an image pasted in one thread from following the user
        into another thread's next send.
        """

        return self._pending_images_for_thread(self.thread_id)

    def _pending_images_for_thread(self, thread_id: str | None) -> list[PendingImage]:
        return self._pending_images_by_thread.setdefault(thread_id, [])

    def _clear_pending_images_for_thread(self, thread_id: str | None) -> None:
        self._pending_images_by_thread.pop(thread_id, None)

    def _active_queue_length(self) -> int:
        return len(self._queued_turns_for_thread(self.thread_id))

    def _queued_turns_for_thread(self, thread_id: str | None) -> list[QueuedTurn]:
        run_state = self._thread_state(thread_id)
        if run_state is None or run_state.worker is None:
            return []
        return list(run_state.queue)

    def _queued_turn_location(self, thread_id: str | None, queue_id: str) -> tuple[ThreadRunState, int] | None:
        run_state = self._thread_state(thread_id)
        if run_state is None or run_state.worker is None:
            return None
        for index, queued_turn in enumerate(run_state.queue):
            if queued_turn.queue_id == queue_id:
                return run_state, index
        return None

    def _update_queued_turn_prompt(self, thread_id: str | None, queue_id: str, prompt: str) -> str:
        location = self._queued_turn_location(thread_id, queue_id)
        if location is None:
            self._flash(self._text("pending_turn_started"), severity="warning")
            return "missing"
        run_state, index = location
        queued_turn = run_state.queue[index]
        normalized = prompt.strip()
        if not normalized:
            if queued_turn.image_paths:
                normalized = self._text("image_only_prompt")
            else:
                self._flash(self._text("write_first"), severity="warning")
                return "empty"
        run_state.queue[index] = replace(queued_turn, prompt=normalized)
        timeline = self._timeline_for_thread(run_state.thread_id)
        if timeline is not None:
            timeline.add_queued_turn(queued_turn.queue_id, normalized, queued_turn.image_paths)
        if self._is_active_thread(run_state.thread_id):
            self._sync_transcript_from_timeline()
        self._refresh_pending_turns()
        self._refresh_status()
        return "updated"

    def _delete_queued_turn(self, thread_id: str | None, queue_id: str) -> str:
        location = self._queued_turn_location(thread_id, queue_id)
        if location is None:
            self._flash(self._text("pending_turn_started"), severity="warning")
            return "missing"
        run_state, index = location
        queued_turn = run_state.queue[index]
        del run_state.queue[index]
        timeline = self._timeline_for_thread(run_state.thread_id)
        if timeline is not None:
            timeline.remove_queued_turn(queued_turn.queue_id)
        if self._is_active_thread(run_state.thread_id):
            self._sync_transcript_from_timeline()
        self._refresh_pending_turns()
        self._refresh_status()
        return "deleted"

    def _move_queued_turn(self, thread_id: str | None, queue_id: str, step: int) -> str:
        location = self._queued_turn_location(thread_id, queue_id)
        if location is None:
            self._flash(self._text("pending_turn_started"), severity="warning")
            return "missing"
        run_state, index = location
        target = max(0, min(len(run_state.queue) - 1, index + step))
        if target == index:
            return "unchanged"
        queued_turn = run_state.queue.pop(index)
        run_state.queue.insert(target, queued_turn)
        self._refresh_pending_turns()
        self._refresh_status()
        return "moved"

    def _ensure_active_thread(self) -> str:
        if self.thread_id is None:
            self.thread_id = self.engine.thread_store.create_thread()
            self._move_draft_goal_enable_to_thread(self.thread_id)
        return self.thread_id

    def _move_draft_goal_enable_to_thread(self, thread_id: str) -> None:
        if None not in self._pending_goal_enable_threads:
            return
        self._pending_goal_enable_threads.discard(None)
        self._pending_goal_enable_threads.add(thread_id)

    def _set_pending_goal_enable(self, thread_id: str | None, enabled: bool) -> None:
        if enabled:
            self._pending_goal_enable_threads.add(thread_id)
        else:
            self._pending_goal_enable_threads.discard(thread_id)

    def _goal_enable_pending(self, thread_id: str | None) -> bool:
        return thread_id in self._pending_goal_enable_threads

    def _materialize_pending_goal_enable(self, thread_id: str) -> None:
        if thread_id not in self._pending_goal_enable_threads:
            return
        state = self.engine.goal_state(thread_id)
        if state is None or not state.enabled:
            self.engine.enable_goal_mode(thread_id)
        self._pending_goal_enable_threads.discard(thread_id)
        if self.is_mounted:
            self._refresh_status()

    def _current_level_for_thread(self, thread_id: str | None = None) -> str | None:
        if self.level:
            return self.level
        if thread_id:
            metadata_level = self._thread_metadata_level(thread_id)
            if metadata_level:
                return metadata_level
        return self.engine.config.runtime.default_level

    def _thread_metadata(self, thread_id: str) -> dict[str, Any]:
        try:
            return self.engine.thread_store.thread_digest(thread_id)
        except (OSError, ValueError, FileNotFoundError):
            return {}

    def _thread_metadata_level(self, thread_id: str) -> str | None:
        level = str(self._thread_metadata(thread_id).get("active_level") or "").strip()
        return level or None

    def _level_model_name(self, level: str | None) -> str:
        try:
            return self.engine.config.level(level).model
        except ConfigError:
            return ""

    def _level_model_signature(self, level: str | None) -> tuple[str, str, str] | None:
        try:
            model = self.engine.config.model_for_level(level)
        except ConfigError:
            return None
        return (model.provider, model.api, model.model)

    def _levels_use_same_model(self, left: str | None, right: str | None) -> bool:
        left_signature = self._level_model_signature(left)
        right_signature = self._level_model_signature(right)
        return bool(left_signature and right_signature and left_signature == right_signature)

    def _persist_thread_level(self, thread_id: str, level: str | None) -> None:
        if not thread_id or not level:
            return
        metadata = self._thread_metadata(thread_id)
        if metadata.get("active_level") == level and metadata.get("active_model") == self._level_model_name(level):
            return
        self.engine.thread_store.append(
            thread_id,
            "thread.level_updated",
            level=level,
            model=self._level_model_name(level),
            previous_level=metadata.get("active_level"),
            previous_model=metadata.get("active_model"),
        )

    def _append_model_switch_warning(
        self,
        thread_id: str,
        *,
        from_level: str,
        to_level: str,
    ) -> None:
        message = self._text("model_switch_warning")
        event = self.engine.thread_store.append(
            thread_id,
            "thread.model_switch_warning",
            from_level=from_level,
            to_level=to_level,
            from_model=self._level_model_name(from_level),
            to_model=self._level_model_name(to_level),
            message=message,
        )
        if self._is_active_thread(thread_id):
            self._append_model_switch_warning_cell(event)

    def _append_model_switch_warning_cell(
        self,
        event: dict[str, Any],
        *,
        before: MountBefore = None,
    ) -> TranscriptCell:
        message = str(event.get("message") or self._text("model_switch_warning"))
        from_level = str(event.get("from_level") or "")
        to_level = str(event.get("to_level") or "")
        content = Text(message, style="yellow")
        if from_level or to_level:
            content.append("\n")
            content.append(f"{from_level or '?'} -> {to_level or '?'}", style="dim")
        return self._append_cell(
            content,
            "event",
            before=before,
            copy_text=message,
        )

    def _append_token_estimation_warning(
        self,
        event: dict[str, Any],
        *,
        before: MountBefore = None,
        flash: bool = True,
    ) -> TranscriptCell:
        """Show the user when compaction had to rely on token estimation."""

        message = str(event.get("message") or self._text("token_estimation_warning"))
        content = Text.assemble(("⚠ ", "yellow"), (message, "yellow"))
        if flash:
            self._flash(message, severity="warning")
        return self._append_cell(content, "event", before=before, copy_text=message)

    def _default_screen_mounted(self) -> bool:
        try:
            self.query_one("#composer", TextArea)
        except NoMatches:
            return False
        return True

    def _reset_live_render_state(self) -> None:
        self._assistant_buffer = ""
        self._assistant_cell = None
        self._reasoning_cell = None
        self._reasoning_buffer = ""
        self._tool_cells.clear()
        self._tool_started_calls.clear()
        self._tool_partial_payloads.clear()
        self._tool_delta_cells.clear()
        self._tool_delta_calls.clear()
        self._process_cells = []
        self._process_fold_cell = None
        self._process_collapsed = False
        self._process_anchor_cell = None
        self._timeline_cells.clear()
        self._timeline_item_ids.clear()

    def _reset_live_view_state(self) -> None:
        self._reset_live_render_state()
        self._turn_started_at = None
        self._turn_completed_at = None

    def _background_run_states(self) -> list[ThreadRunState]:
        return [
            run_state
            for thread_id, run_state in self._thread_runs.items()
            if not self._is_active_thread(thread_id)
        ]

    def _activity_state_for_thread(self, thread_id: str) -> ThreadActivityState:
        state = self._thread_activity.get(thread_id)
        if state is None:
            state = ThreadActivityState(thread_id=thread_id)
            self._thread_activity[thread_id] = state
        return state

    def _mark_thread_active(self, thread_id: str, *, now: float | None = None) -> None:
        activity = self._activity_state_for_thread(thread_id)
        if activity.active_started_monotonic is None:
            activity.active_started_monotonic = monotonic() if now is None else now
        # A rerun/resume moves the thread out of the completed bucket until this
        # latest piece of work finishes.
        activity.completed = False
        if self.is_mounted:
            self._refresh_top_bar()

    def _mark_thread_inactive(
        self,
        thread_id: str,
        *,
        completed: bool,
        now: float | None = None,
    ) -> None:
        activity = self._activity_state_for_thread(thread_id)
        ended_at = monotonic() if now is None else now
        if activity.active_started_monotonic is not None:
            activity.total_elapsed_s += max(0.0, ended_at - activity.active_started_monotonic)
            activity.active_started_monotonic = None
        activity.completed = completed
        if self.is_mounted:
            self._refresh_top_bar()

    def _thread_elapsed_seconds(self, thread_id: str | None) -> float:
        if thread_id is None:
            return 0.0
        activity = self._thread_activity.get(thread_id)
        if activity is None:
            return 0.0
        total = activity.total_elapsed_s
        if activity.active_started_monotonic is not None:
            total += max(0.0, monotonic() - activity.active_started_monotonic)
        return total

    def _active_activity_thread_ids(self) -> list[str]:
        return sorted(
            thread_id
            for thread_id, activity in self._thread_activity.items()
            if activity.active
        )

    def _completed_activity_thread_ids(self) -> list[str]:
        return sorted(
            thread_id
            for thread_id, activity in self._thread_activity.items()
            if activity.completed and not activity.active
        )

    def _run_state_for_thread(self, thread_id: str | None) -> ThreadRunState:
        if thread_id is None:
            thread_id = self.engine.thread_store.create_thread()
            self.thread_id = thread_id
            self._move_draft_goal_enable_to_thread(thread_id)
        run_state = self._thread_runs.get(thread_id)
        if run_state is None:
            run_state = ThreadRunState(
                thread_id=thread_id,
                worker=None,
                cancel_event=asyncio.Event(),
                queue=[],
                status=self._text("idle"),
            )
            self._thread_runs[thread_id] = run_state
        return run_state

    def _thread_state(self, thread_id: str | None) -> ThreadRunState | None:
        if thread_id is None:
            return None
        return self._thread_runs.get(thread_id)

    def _thread_has_blocking_error(self, thread_id: str | None) -> bool:
        run_state = self._thread_state(thread_id)
        if run_state is not None and run_state.terminal_error:
            return True
        if run_state is not None and run_state.retryable_error:
            return False
        event = self._latest_thread_event(thread_id, "turn.error")
        return bool(event and not self._is_retryable_error_event(event))

    def _is_active_thread(self, thread_id: str | None) -> bool:
        return bool(thread_id and self.thread_id == thread_id)

    def _refresh_active_run_state(self) -> None:
        run_state = self._thread_state(self.thread_id)
        self.busy = run_state is not None and run_state.worker is not None
        shell = self.query_one("#composer-shell", Vertical)
        if run_state is None or run_state.worker is None:
            shell.remove_class("busy")
            self._current_worker = None
            self._current_cancel_event = None
            self._turn_started_at = None
            self._turn_completed_at = None
            return
        shell.add_class("busy")
        self._current_worker = run_state.worker
        self._current_cancel_event = run_state.cancel_event
        self._turn_started_at = run_state.started_at
        self._turn_completed_at = run_state.completed_at

    def _sync_run_state_from_active(self, run_state: ThreadRunState) -> None:
        return

    def _sync_active_from_run_state(self, run_state: ThreadRunState) -> None:
        return

    def _update_turn_timestamps(self, item: dict[str, Any], run_state: ThreadRunState) -> None:
        turn_id = str(item.get("turn_id") or "").strip()
        if turn_id:
            run_state.turn_id = turn_id
            timeline = self._timeline_for_thread(run_state.thread_id)
            if timeline is not None:
                old_pending_turn_id = run_state.pending_user_turn_id
                timeline.promote_user_turn(old_pending_turn_id, turn_id)
                if old_pending_turn_id and self._is_active_thread(run_state.thread_id):
                    promoted = timeline.items_by_id.get(f"user:{turn_id}")
                    if promoted is not None:
                        self._rebind_timeline_widget_id(f"user:{old_pending_turn_id}", promoted)
            run_state.pending_user_turn_id = None
        started_at = str(item.get("turn_started_at") or item.get("started_at") or "").strip()
        if started_at:
            run_state.started_at = started_at
            if self._is_active_thread(run_state.thread_id):
                self._turn_started_at = started_at
        if item.get("type") in {"turn.completed", "turn.interrupted", "turn.error"}:
            completed_at = str(item.get("completed_at") or item.get("created_at") or utc_now_iso()).strip()
            run_state.completed_at = completed_at
            if self._is_active_thread(run_state.thread_id):
                self._turn_completed_at = completed_at
                self._refresh_process_fold_elapsed()

    def _record_live_thread_event(
        self,
        run_state: ThreadRunState,
        event_type: str,
        item: dict[str, Any],
    ) -> None:
        timeline = self._timeline_for_thread(run_state.thread_id)
        if timeline is None:
            return
        timeline.apply_live_event(event_type, item)

    def _defer_stream_retry_event(self, run_state: ThreadRunState, item: dict[str, Any]) -> None:
        timeline = self._timeline_for_thread(run_state.thread_id)
        if timeline is None:
            return
        turn_id = str(item.get("turn_id") or run_state.turn_id or "").strip()
        if not turn_id:
            return
        timeline._turn(turn_id).pending_stream_retries.append(dict(item))

    def _flush_pending_stream_retries(self, run_state: ThreadRunState) -> None:
        timeline = self._timeline_for_thread(run_state.thread_id)
        if timeline is None or not run_state.turn_id:
            return
        timeline.flush_pending_stream_retries(run_state.turn_id)
        if self._is_active_thread(run_state.thread_id):
            self._sync_transcript_from_timeline()

    def _live_event_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return dict(item)

    def _apply_thread_event_to_active(self, event_type: str, item: dict[str, Any]) -> None:
        timeline = self._timeline_for_active()
        if timeline is None:
            return
        timeline.apply_live_event(event_type, item)
        self._sync_transcript_from_timeline()

    def _stream_event_needs_throttle(self, event_type: str) -> bool:
        return event_type in {"assistant.delta", "assistant.reasoning_delta", "tool.delta", "tool.partial"}

    def _schedule_stream_render(self) -> None:
        if not self.is_mounted:
            return
        self._stream_render_due = True
        if self._stream_render_timer is not None:
            return
        delay = max(0.001, STREAM_RENDER_INTERVAL_SECONDS - (monotonic() - self._last_stream_render_at))
        self._stream_render_timer = self.set_timer(delay, self._flush_stream_render, name="stream-render")

    def _flush_stream_render(self) -> None:
        self._stream_render_timer = None
        if not self._stream_render_due:
            return
        if not self._default_screen_mounted():
            self._stream_render_due = False
            return
        self._stream_render_due = False
        self._last_stream_render_at = monotonic()
        self._sync_transcript_from_timeline()

    def _schedule_stream_status_refresh(self) -> None:
        if not self.is_mounted:
            return
        self._stream_status_due = True
        if self._stream_status_timer is not None:
            return
        delay = max(0.001, STREAM_STATUS_INTERVAL_SECONDS - (monotonic() - self._last_stream_status_at))
        self._stream_status_timer = self.set_timer(delay, self._flush_stream_status_refresh, name="stream-status")

    def _flush_stream_status_refresh(self) -> None:
        self._stream_status_timer = None
        if not self._stream_status_due:
            return
        if not self._default_screen_mounted():
            self._stream_status_due = False
            return
        self._stream_status_due = False
        self._last_stream_status_at = monotonic()
        if self.busy:
            self._refresh_busy_status()
        else:
            self._refresh_status()

    def _flush_stream_updates(self) -> None:
        if self._stream_render_timer is not None:
            self._stream_render_timer.stop()
            self._stream_render_timer = None
        if self._stream_status_timer is not None:
            self._stream_status_timer.stop()
            self._stream_status_timer = None
        if not self._default_screen_mounted():
            self._stream_render_due = False
            self._stream_status_due = False
            return
        if self._stream_render_due:
            self._stream_render_due = False
            self._last_stream_render_at = monotonic()
            self._sync_transcript_from_timeline()
        if self._stream_status_due:
            self._stream_status_due = False
            self._last_stream_status_at = monotonic()
            if self.busy:
                self._refresh_busy_status()
            else:
                self._refresh_status()

    def _mark_run_error_state(self, run_state: ThreadRunState, event: dict[str, Any]) -> None:
        retryable = self._is_retryable_error_event(event)
        run_state.retryable_error = retryable
        run_state.terminal_error = not retryable
        run_state.status = self._text("error")

    def _is_retryable_error_event(self, event: dict[str, Any]) -> bool:
        if "retryable" in event:
            return bool(event.get("retryable"))
        error_type = str(event.get("error_type") or "")
        message = str(event.get("message") or "").lower()
        if error_type in {"TimeoutException", "RequestError", "ConnectError", "ReadError", "NetworkError"}:
            return True
        status = self._http_status_from_error(event)
        return status == 429 or (status is not None and 500 <= status < 600) or "timeout" in message

    def _http_status_from_error(self, event: dict[str, Any]) -> int | None:
        value = event.get("status_code")
        if isinstance(value, int):
            return value
        text = " ".join(str(event.get(key) or "") for key in ("error_type", "message", "title"))
        match = re.search(r"\b(?:HTTP|status)\s*(\d{3})\b", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _latest_thread_event(self, thread_id: str | None, event_type: str) -> dict[str, Any] | None:
        if thread_id is None:
            return None
        try:
            return self.engine.thread_store.latest_event(thread_id, event_type)
        except (OSError, ValueError, FileNotFoundError):
            return None

    def _retry_thread(self, thread_id: str | None) -> None:
        if not thread_id:
            return
        run_state = self._thread_state(thread_id)
        if run_state is not None and run_state.worker is not None:
            self._flash(self._text("working"), severity="warning")
            return
        if self.thread_id != thread_id:
            self._resume_thread(thread_id)
        if run_state is not None:
            self._thread_runs.pop(thread_id, None)
        self._remove_retry_buttons(thread_id)
        timeline = self._timeline_for_thread(thread_id)
        if timeline is not None:
            for item in list(timeline.items):
                if item.kind == "error":
                    timeline._remove_item(item.id)
        self._start_retry_turn(thread_id)

    def _remove_retry_buttons(self, thread_id: str) -> None:
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except NoMatches:
            return
        for child in list(transcript.children):
            if isinstance(child, RetryTurnButton) and child.retry_thread_id == thread_id:
                child.remove()

    async def _handle_thread_event(
        self,
        thread_id: str,
        event_type: str,
        item: dict[str, Any],
        run_state: ThreadRunState,
    ) -> None:
        self._update_turn_timestamps(item, run_state)
        if event_type == "assistant.reasoning_absent":
            self._reasoning_cell = None
            self._reasoning_buffer = ""
        if event_type == "model.stream_retry":
            self._defer_stream_retry_event(run_state, item)
            run_state.status = self._text("working")
            if self._is_active_thread(thread_id):
                self._refresh_status(self._text("working"))
            return
        timeline = self._timeline_for_thread(thread_id)
        if timeline is not None:
            timeline.apply_live_event(event_type, item)
        if event_type == "assistant.delta":
            run_state.status = self._text("writing_answer")
        elif event_type == "assistant.reasoning_delta":
            run_state.status = self._text("thinking_status")
        elif event_type in {"tool.delta"}:
            run_state.status = self._text("writing_script")
        elif event_type in {"tool.started", "tool.partial"}:
            run_state.status = self._text("running_python")
        elif event_type in {"tool.output", "compaction.completed", "thread.token_estimation_warning"}:
            run_state.status = self._text("working")
        elif event_type == "compaction.started":
            run_state.status = self._text("compacting")
        elif event_type == "turn.interrupted":
            run_state.status = self._text("interrupted")
        elif event_type == "turn.error":
            self._mark_run_error_state(run_state, item)
        if self._is_active_thread(thread_id):
            if self._stream_event_needs_throttle(event_type):
                self._last_status = run_state.status
                self._schedule_stream_render()
                self._schedule_stream_status_refresh()
                return
            self._flush_stream_updates()
            self._sync_transcript_from_timeline()
            if event_type == "assistant.delta":
                self._refresh_status(self._text("writing_answer"))
            elif event_type == "assistant.reasoning_delta":
                self._refresh_status(self._text("thinking_status"))
            elif event_type == "tool.delta":
                self._refresh_status(self._text("writing_script"))
            elif event_type in {"tool.started", "tool.partial"}:
                self._refresh_status(self._text("running_python"))
            elif event_type == "tool.output":
                self._refresh_status(self._text("working"))
            elif event_type == "compaction.started":
                self._refresh_status(self._text("compacting"))
            elif event_type == "compaction.completed":
                self._refresh_status(self._text("working"))

    def action_request_quit(self) -> None:
        now = monotonic()
        if now - self._last_quit_request_at < QUIT_KEY_DEBOUNCE_SECONDS:
            return
        self._last_quit_request_at = now
        if self._quit_armed:
            self.exit()
            return
        draft = self.query_one("#composer", TextArea).text.strip()
        suffix = self._text("draft_lost") if draft else ""
        self._quit_armed = True
        self._flash(f"{self._text('quit_again')}{suffix}", severity="warning")
        self.set_timer(2.0, self._clear_quit_arm)

    def _quit_from_command(self) -> None:
        self.exit()

    def action_interrupt_turn(self) -> None:
        if not self.busy:
            self.action_request_quit()
            return
        now = monotonic()
        if self._interrupt_armed and now - self._last_interrupt_request_at <= 2.0:
            self._interrupt_armed = False
            if self._current_cancel_event is not None:
                self._current_cancel_event.set()
            self._flash(self._text("interrupted"), severity="warning")
            return
        self._interrupt_armed = True
        self._last_interrupt_request_at = now
        self._flash(self._text("interrupt_again"), severity="warning")
        self.set_timer(2.0, self._clear_interrupt_arm)

    def _interrupt_from_command(self) -> None:
        if self._current_cancel_event is None:
            self._flash(self._text("no_running_turn"))
            return
        self._interrupt_armed = False
        self._current_cancel_event.set()
        self._flash(self._text("interrupted"), severity="warning")

    def action_toggle_status_panel(self) -> None:
        self._open_status_panel()

    def action_open_command_palette(self) -> None:
        panel = self._active_fullscreen_panel()
        if panel is not None:
            panel.close_navigation()
            return
        self._open_command_palette()

    def action_toggle_composer_height(self) -> None:
        self._composer_height_override = "collapsed" if self._composer_expanded else "expanded"
        self._resize_composer()

    def action_focus_composer(self) -> None:
        composer = self.query_one("#composer", TextArea)
        if self.screen.focused is composer:
            return
        composer.focus()

    def action_toggle_tool_details(self) -> None:
        focused = self.screen.focused
        if isinstance(focused, ExpandableTranscriptCell):
            self._open_tool_details_panel(focused)
            return
        cell = self._latest_expandable_cell()
        if cell is None:
            self._flash(self._text("no_details"))
            return
        self._open_tool_details_panel(cell)

    def action_toggle_visible_process_folds(self) -> None:
        folds = self._visible_process_fold_cells()
        if not folds:
            return
        fold = folds[-1]
        fold.set_collapsed(not fold.collapsed)

    def action_attach_clipboard_image(self) -> None:
        try:
            model = self.engine.config.model_for_level(self.level)
        except ConfigError as exc:
            self._flash(str(exc), severity="error")
            return
        if model.supports_images is False:
            self._flash(self._text("image_model_disabled"), severity="error")
            return
        from uv_agent.clipboard import ClipboardImageError

        try:
            image = save_clipboard_image(project_tui_clipboard_dir(self.project_root))
        except ClipboardImageError as exc:
            self._flash(str(exc), severity="warning")
            return
        pending = PendingImage(path=image.path, width=image.width, height=image.height)
        self._pending_images.append(pending)
        self._refresh_pending_images()
        self._flash(
            f"{self._text('image_queued')} {image.width}x{image.height}",
        )

    def action_preview_images(self) -> None:
        attachments = self._thread_image_attachments()
        if not attachments:
            self._flash(self._text("no_images"))
            return
        self.push_screen(ImagePreviewPanel(attachments, len(attachments) - 1))

    def _expandable_cells(self) -> list[ExpandableTranscriptCell]:
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except NoMatches:
            return []
        return [
            child
            for child in transcript.children
            if isinstance(child, ExpandableTranscriptCell)
            and not child.has_class("process_fold_hidden")
        ]

    def _visible_process_fold_cells(self) -> list[FoldedProcessCell]:
        try:
            transcript = self.query_one("#transcript", TranscriptScroll)
        except NoMatches:
            return []
        viewport_top = transcript.scroll_y
        viewport_bottom = viewport_top + transcript.scrollable_content_region.height
        folds: list[FoldedProcessCell] = []
        for child in transcript.children:
            if not isinstance(child, FoldedProcessCell):
                continue
            region = child.virtual_region
            # Fold bar itself is in the viewport.
            if region.y < viewport_bottom and region.y + region.height > viewport_top:
                folds.append(child)
            # Fold bar is scrolled off but the expanded cells it wraps may
            # still be visible; treat the fold as reachable when any of its
            # cells overlaps the viewport.
            elif not child.collapsed:
                for cell in child.cells:
                    cr = cell.virtual_region
                    if cr.y < viewport_bottom and cr.y + cr.height > viewport_top:
                        folds.append(child)
                        break
        return folds

    def _latest_expandable_cell(self) -> ExpandableTranscriptCell | None:
        cells = self._expandable_cells()
        return cells[-1] if cells else None

    def _focus_relative_expandable_cell(self, current: ExpandableTranscriptCell, step: int) -> None:
        next_cell = self._relative_expandable_cell(current, step)
        next_cell.focus()
        next_cell.scroll_visible(animate=False)

    def _relative_expandable_cell(self, current: ExpandableTranscriptCell, step: int) -> ExpandableTranscriptCell:
        cells = self._expandable_cells()
        if not cells:
            return current
        try:
            index = cells.index(current)
        except ValueError:
            index = len(cells) - 1
        return cells[(index + step) % len(cells)]

    def _image_cells(self) -> list[ImageAttachmentCell]:
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except NoMatches:
            return []
        return [
            child
            for child in transcript.children
            if isinstance(child, ImageAttachmentCell)
        ]

    def _focus_relative_image_cell(self, current: ImageAttachmentCell, step: int) -> None:
        next_cell = self._relative_image_cell(current, step)
        next_cell.focus()
        next_cell.scroll_visible(animate=False)

    def _relative_image_cell(self, current: ImageAttachmentCell, step: int) -> ImageAttachmentCell:
        cells = self._image_cells()
        if not cells:
            return current
        try:
            index = cells.index(current)
        except ValueError:
            index = len(cells) - 1
        return cells[(index + step) % len(cells)]

    def _open_image_preview_for_cell(self, cell: ImageAttachmentCell) -> None:
        attachments = self._thread_image_attachments()
        if not attachments:
            attachments = [cell.attachment]
        index = next(
            (
                idx
                for idx, attachment in enumerate(attachments)
                if attachment.get("attachment_id") == cell.attachment.get("attachment_id")
            ),
            len(attachments) - 1,
        )
        self.push_screen(ImagePreviewPanel(attachments, index))

    def _open_tool_details_panel(self, cell: ExpandableTranscriptCell) -> None:
        self.push_screen(ToolDetailsPanel(cell))

    def _open_pending_send_queue(self) -> None:
        if not self._queued_turns_for_thread(self.thread_id):
            self._flash(self._text("no_pending_turns"))
            return
        self.push_screen(PendingSendQueuePanel(self.thread_id))

    def _thread_image_attachments(self) -> list[dict[str, Any]]:
        if not self.thread_id:
            return []
        attachments: list[dict[str, Any]] = []
        events = self.engine.thread_store.read_events(
            self.thread_id,
            event_types={"item.image_attachment"},
        )
        for event in events:
            attachment = event.get("attachment")
            if isinstance(attachment, dict):
                attachments.append(attachment)
        return attachments

    def action_help(self) -> None:
        self._open_help_panel()

    def _clear_quit_arm(self) -> None:
        self._quit_armed = False
        self._refresh_status()

    def _clear_interrupt_arm(self) -> None:
        self._interrupt_armed = False
        self._refresh_status()

    def _schedule_selection_copy(self, text: str, *, source: str = "composer") -> None:
        if text == self._last_auto_copied_selection:
            return
        self._cancel_selection_copy()
        self._pending_selection_copy = text
        self._selection_copy_timer = self.set_timer(
            1.0,
            lambda: self._copy_pending_selection(source),
            name="selection-copy",
        )

    def _cancel_selection_copy(self) -> None:
        if self._selection_copy_timer is not None:
            self._selection_copy_timer.stop()
            self._selection_copy_timer = None
        self._pending_selection_copy = ""

    def _copy_pending_selection(self, source: str) -> None:
        self._selection_copy_timer = None
        text = self._pending_selection_copy
        self._pending_selection_copy = ""
        if not text:
            return
        if source == "composer":
            try:
                composer = self.query_one("#composer", TextArea)
            except NoMatches:
                return
            if composer.selected_text != text:
                return
        else:
            selected_text = self.screen.get_selected_text()
            if not selected_text:
                return
            text = selected_text
        self.copy_to_clipboard(text)
        self._last_auto_copied_selection = text
        self.notify(self._text("copied"), timeout=1.5)

    def _queued_turn_markup(self, prompt: str, image_paths: list[Path]) -> Text:
        content = Text(self._text("queued"), style="dim")
        content.append("\n")
        content.append(prompt)
        if image_paths:
            content.append("\n")
            content.append(f"+{len(image_paths)} {self._text('images')}", style="dim")
        return content

    def _handle_command(self, prompt: str) -> bool:
        command, _, rest = prompt.partition(" ")
        if command == "/clear":
            old_thread_id = self.thread_id
            self._save_active_thread_view_state()
            self._close_active_panel()
            self._set_pending_goal_enable(old_thread_id, False)
            self._set_pending_goal_enable(None, False)
            self.thread_id = None
            self.level = None
            self._reset_live_view_state()
            self._clear_pending_images_for_thread(old_thread_id)
            self._reset_transcript()
            self._refresh_pending_images()
            self._refresh_active_run_state()
            self._refresh_status(self._text("idle"))
            return True
        if command == "/quit":
            self._quit_from_command()
            return True
        if command == "/cancel":
            self._interrupt_from_command()
            return True
        if command == "/threads":
            self._open_threads_panel()
            return True
        if command == "/status":
            self._open_status_panel()
            return True
        if command == "/goal":
            self._open_goal_panel()
            return True
        if command == "/config":
            self._open_config_panel()
            return True
        if command == "/models":
            self._open_models_panel()
            return True
        if command == "/mcp":
            self._open_mcp_panel()
            return True
        if command == "/skills":
            self._open_skills_panel()
            return True
        if command == "/level":
            self._open_current_level_panel()
            return True
        if command in {"/help", "?"}:
            self._open_help_panel()
            return True
        if command.startswith("/"):
            self._flash(f"{self._text('unknown_command')}: {command}", severity="error")
            self._open_help_panel()
            return True
        return False

    async def action_quit(self) -> None:
        # Textual ships Ctrl+Q -> quit by default; this app reserves key-based quit for Ctrl+C.
        return

    def _open_help_panel(self) -> None:
        lines = [
            Text(self._text("keyboard_shortcuts"), style="bold"),
            Text.assemble("- ", ("Ctrl+Enter / Ctrl+J", "cyan"), " ", (self._text("help_send"), "dim")),
            Text.assemble("- ", ("Enter", "cyan"), " ", (self._text("help_newline"), "dim")),
            Text.assemble("- ", ("Ctrl+O / /", "cyan"), " ", (self._text("help_commands"), "dim")),
            Text.assemble("- ", ("F1 / ?", "cyan"), " ", (self._text("help_help"), "dim")),
            Text.assemble("- ", ("Ctrl+S", "cyan"), " ", (self._text("help_status"), "dim")),
            Text.assemble("- ", ("Ctrl+D", "cyan"), " ", (self._text("help_details"), "dim")),
            Text.assemble("- ", ("F2", "cyan"), " ", (self._text("help_attach_image"), "dim")),
            Text.assemble("- ", ("F3", "cyan"), " ", (self._text("help_preview_images"), "dim")),
            Text.assemble("- ", ("Tab", "cyan"), " ", (self._text("help_height"), "dim")),
            Text.assemble("- ", ("Ctrl+C", "cyan"), " ", (self._text("help_interrupt_quit"), "dim")),
            Text(),
            Text(self._text("mentions"), style="bold"),
            Text.assemble("- ", ("@", "cyan"), " ", (self._text("help_mention_files"), "dim")),
            Text.assemble("- ", ("@@", "cyan"), " ", (self._text("help_mention_threads"), "dim")),
            Text(),
            Text.assemble((self._text("commands"), "bold"), " ", ("(Tab/Enter, Esc)", "dim")),
        ]
        for spec in self._commands():
            lines.append(
                Text.assemble((f"{spec.usage:<18}", "cyan"), " ", (spec.description, "dim"))
            )
        self._open_panel(join_lines(lines), "help", self._text("help"))

    def _append_help(self) -> None:
        lines = [Text.assemble((self._text("commands"), "bold"), " ", ("(Ctrl+O, F1, Esc)", "dim"))]
        for spec in self._commands():
            lines.append(
                Text.assemble((f"{spec.usage:<18}", "cyan"), " ", (spec.description, "dim"))
            )
        self._append_cell(join_lines(lines), "event")

    def _append_user(self, text: str, *, before: MountBefore = None) -> TranscriptCell:
        label = "你" if self.language.is_chinese else "you"
        # Codex-style "› " prefix keeps user turns easy to spot.
        return self._append_cell(
            join_lines([Text(f"› {label}", style="bold #7dd3fc"), plain(text)]),
            "user",
            before=before,
        )

    def _append_assistant_text(self, text: str) -> None:
        timeline = self._timeline_for_active()
        if timeline is not None:
            timeline.seed_assistant_delta(text)
            self._sync_transcript_from_timeline()
            acc = timeline.active_turns.get("manual")
            cell = self._timeline_cells.get(acc.assistant_item_id or "") if acc is not None else None
            self._assistant_cell = cell if isinstance(cell, TranscriptCell) else None
            self._assistant_buffer = acc.assistant_buffer if acc is not None else self._assistant_buffer + text
            return
        self._assistant_buffer += text
        if self._assistant_cell is None:
            self._mark_transcript_content()
            self._assistant_cell = TranscriptCell(classes="assistant")
            self.query_one("#transcript", VerticalScroll).mount(self._assistant_cell)
        self._assistant_cell.update(_markdown(self._assistant_buffer), copy_text=self._assistant_buffer)
        self._scroll_end()

    async def _append_assistant_delta(self, text: str) -> None:
        self._append_assistant_text(text)

    def _seal_assistant_round(self) -> None:
        self._assistant_buffer = ""
        self._assistant_cell = None

    def _append_reasoning_delta(self, text: str) -> None:
        if not text:
            return
        timeline = self._timeline_for_active()
        if timeline is not None:
            timeline.seed_reasoning_delta(text)
            self._sync_transcript_from_timeline()
            acc = timeline.active_turns.get("manual")
            cell = self._timeline_cells.get(acc.reasoning_item_id or "") if acc is not None else None
            self._reasoning_cell = cell if isinstance(cell, TranscriptCell) else None
            self._reasoning_buffer = acc.reasoning_buffer if acc is not None else self._append_reasoning_text(self._reasoning_buffer, text)
            return
        self._reasoning_buffer = self._append_reasoning_text(self._reasoning_buffer, text)
        display_text = self._reasoning_buffer.strip()
        if not display_text:
            return
        first = display_text.splitlines()[0]
        if len(first) > 120:
            first = first[:117].rstrip() + "..."
        content = Text.assemble(
            (self._text("thinking"), "dim italic"),
            "  ",
            (first, "italic #a3b1c2"),
        )
        if self._reasoning_cell is None:
            self._reasoning_cell = self._append_cell(content, "reasoning")
        else:
            self._reasoning_cell.update(content)
            self._scroll_end()

    def _append_reasoning_text(self, existing: str, delta: str) -> str:
        if not existing:
            return delta.lstrip()
        return existing + delta

    def _reasoning_markup(self, text: str) -> tuple[Text, Text]:
        stripped = text.strip()
        first = stripped.splitlines()[0]
        if len(first) > 120:
            first = first[:117].rstrip() + "..."
        summary = Text.assemble(
            (self._text("thinking"), "dim italic"),
            "  ",
            (first, "italic #a3b1c2"),
        )
        return summary, plain(stripped)

    def _finalize_reasoning(self, text: str) -> None:
        stripped = text.strip()
        if not stripped and self._reasoning_buffer.strip():
            stripped = self._reasoning_buffer.strip()
        if not stripped:
            self._clear_pending_reasoning()
            return
        summary, details = self._reasoning_markup(stripped)
        if isinstance(self._reasoning_cell, ExpandableTranscriptCell):
            self._reasoning_cell.set_details(summary, details)
            cell = self._reasoning_cell
        elif self._reasoning_cell is not None:
            cell = self._replace_with_reasoning_cell(self._reasoning_cell, summary, details)
        else:
            cell = self._append_reasoning_cell(summary, details)
        self._track_process_cell(cell)
        self._reasoning_cell = None
        self._reasoning_buffer = ""

    def _clear_pending_reasoning(self) -> None:
        if self._reasoning_cell is not None and not isinstance(self._reasoning_cell, ExpandableTranscriptCell):
            self._reasoning_cell.remove()
        self._reasoning_cell = None
        self._reasoning_buffer = ""

    def _append_reasoning_history(
        self,
        text: str,
        *,
        before: MountBefore = None,
    ) -> ExpandableTranscriptCell | None:
        stripped = text.strip()
        if not stripped:
            return None
        summary, details = self._reasoning_markup(stripped)
        return self._append_reasoning_cell(summary, details, before=before)

    def _track_process_cell(self, cell: TranscriptCell | None) -> None:
        if cell is None or isinstance(cell, FoldedProcessCell):
            return
        if cell not in self._process_cells:
            self._process_cells.append(cell)
        if self._process_fold_cell is not None:
            self._process_fold_cell.set_cells(self._process_cells)

    def _process_fold_toggled(self, cell: FoldedProcessCell, collapsed: bool) -> None:
        if cell is self._process_fold_cell:
            self._process_collapsed = collapsed
        group_id = getattr(cell, "timeline_group_id", "")
        timeline = self._timeline_for_active()
        if isinstance(group_id, str) and group_id and timeline is not None:
            group = timeline.process_groups.get(group_id)
            if group is not None:
                group.collapsed = collapsed
            if self.thread_id and self.thread_id in self._thread_view_states:
                # Keep saved per-thread UI state in step with manual toggles so
                # a later restore or full transcript sync cannot resurrect an
                # older fold value.
                self._thread_view_states[self.thread_id].fold_collapsed[group_id] = collapsed

    def _replace_process_cell(self, old_cell: TranscriptCell, new_cell: TranscriptCell) -> None:
        self._process_cells = [
            new_cell if cell is old_cell else cell
            for cell in self._process_cells
        ]
        if self._process_fold_cell is not None:
            self._process_fold_cell.set_cells(self._process_cells)

    def _collapse_process_cells(self) -> None:
        if self._process_collapsed or not self._process_cells:
            return
        self._append_process_fold_cell(
            self._process_cells,
            collapsed=True,
            after=self._process_anchor_cell,
        )
        self._process_collapsed = True
        self._refresh_process_fold_elapsed()

    def _append_compaction_completed(self, item: dict[str, Any]) -> None:
        """Render a completed compaction as a hard visual boundary.

        A compaction checkpoint replaces the previous model context. Mirroring
        that in the transcript keeps old tool/reasoning cells folded before the
        checkpoint while allowing subsequent tool output to start a fresh process
        group below the summary cell.
        """

        self._finalize_turn_render()
        cell = self._append_compaction_cell(item)
        self._process_cells = []
        self._process_fold_cell = None
        self._process_collapsed = False
        self._process_anchor_cell = cell

    def _append_compaction_cell(
        self,
        event: dict[str, Any],
        *,
        before: MountBefore = None,
    ) -> ExpandableTranscriptCell | None:
        """Append an expandable checkpoint cell containing the compaction text.

        Older checkpoints may have been written with an empty summary when a
        provider unexpectedly answered a compaction request with a tool call.
        Hiding those avoids showing a misleading "conversation compacted" block
        whose details contain no actual summary.
        """

        summary_text = str(event.get("text") or "").strip()
        if not summary_text:
            return None
        summary = Text.assemble(
            (self._text("compacted"), "dim"),
            " · ",
            (self._text("compacted_summary_hint"), "cyan dim"),
        )
        details = plain(summary_text)
        return self._append_expandable_cell(
            summary,
            details,
            "event",
            before=before,
            detail_title="compaction_summary",
            detail_hint="compaction_summary_hint",
        )

    def _finalize_turn_render(self) -> None:
        """Bring the live transcript to the same end-of-turn state as re-entry.

        Mirrors `_mount_history_turn_events`: any pending reasoning is cleared,
        the streaming assistant cell is sealed, and every process cell of the
        turn is folded under a single FoldedProcessCell with the turn's elapsed
        label. Safe to call multiple times in a single turn (e.g. once via
        `assistant.final_response_started` and again on `turn.completed`).
        """
        self._clear_pending_reasoning()
        self._seal_assistant_round()
        self._collapse_process_cells()
        self._refresh_process_fold_elapsed()

    def _track_current_assistant_cell_as_process(self) -> None:
        if self._assistant_cell is not None:
            self._track_process_cell(self._assistant_cell)

    def _append_process_fold_cell(
        self,
        cells: list[TranscriptCell],
        *,
        collapsed: bool = True,
        before: MountBefore = None,
        after: TranscriptCell | None = None,
        elapsed_label: str | None = None,
        timeline_group_id: str | None = None,
    ) -> FoldedProcessCell:
        self._mark_transcript_content()
        insert_before = before
        if insert_before is None and after is not None:
            insert_before = self._cell_after(after)
        if insert_before is None:
            insert_before = cells[0] if cells else None
        cell = FoldedProcessCell(
            cells,
            collapsed=collapsed,
            elapsed_label=elapsed_label if elapsed_label is not None else self._process_elapsed_label(),
            classes="process_fold",
        )
        if timeline_group_id:
            setattr(cell, "timeline_group_id", timeline_group_id)
        self.query_one("#transcript", VerticalScroll).mount(cell, before=insert_before)
        self._process_fold_cell = cell
        self._scroll_end()
        return cell

    def _process_elapsed_label(self, *, started_at: str | None = None, completed_at: str | None = None) -> str:
        seconds = _elapsed_between(started_at or self._turn_started_at, completed_at or self._turn_completed_at)
        return format_elapsed(seconds) if seconds is not None else ""

    def _refresh_process_fold_elapsed(self) -> None:
        if self._process_fold_cell is not None:
            self._process_fold_cell.set_elapsed_label(self._process_elapsed_label())

    def _cell_after(self, cell: TranscriptCell) -> MountBefore:
        try:
            children = list(self.query_one("#transcript", VerticalScroll).children)
            index = children.index(cell)
        except (NoMatches, ValueError):
            return None
        next_child = children[index + 1] if index + 1 < len(children) else None
        return next_child if isinstance(next_child, Widget) else None

    def _append_tool_partial(self, item: dict[str, Any]) -> None:
        timeline = self._timeline_for_active()
        if timeline is not None:
            timeline.seed_tool_partial(item)
            self._sync_transcript_from_timeline()
            return
        """Refresh the latest run_python result cell while the process is running."""

        payload = parse_tool_payload(item.get("output", {}))
        if payload is None:
            return
        payload = dict(payload)
        call = item.get("call") if isinstance(item.get("call"), dict) else None
        call_id = str((call or {}).get("call_id") or payload.get("call_id") or "")
        if call_id:
            self._tool_partial_payloads[call_id] = payload
        self._last_tool_payload = payload
        markup = tool_timeline_markup(payload)
        details = tool_detail_markup(payload)
        pending_cell = self._tool_cells.get(call_id) if call_id else None
        if pending_cell is not None:
            self._track_process_cell(pending_cell)
        cell = self._partial_tool_result_cell(item)
        if cell is None:
            cell = self._append_expandable_cell(markup, details, "event")
            self._track_process_cell(cell)
        else:
            cell.set_details(markup, details)
        cell.tool_payload = payload
        self._refresh_status(self._text("running_python"))

    def _partial_tool_result_cell(self, item: dict[str, Any]) -> ExpandableTranscriptCell | None:
        """Return an existing partial-result cell for this tool call, if any."""

        call_raw = item.get("call")
        call = cast(dict[str, Any], call_raw) if isinstance(call_raw, dict) else {}
        call_id = str(call.get("call_id") or "")
        if call_id:
            for cell in reversed(self._process_cells):
                if isinstance(cell, ExpandableTranscriptCell):
                    payload = cell.tool_payload if isinstance(cell.tool_payload, dict) else {}
                    if payload.get("partial") and payload.get("call_id") == call_id:
                        return cell
        payload = parse_tool_payload(item.get("output", {})) or {}
        run_id = str(payload.get("run_id") or "")
        if run_id:
            for cell in reversed(self._process_cells):
                if isinstance(cell, ExpandableTranscriptCell):
                    existing = cell.tool_payload if isinstance(cell.tool_payload, dict) else {}
                    if existing.get("partial") and existing.get("run_id") == run_id:
                        return cell
        return None

    def _append_tool_output(self, item: dict[str, Any]) -> None:
        timeline = self._timeline_for_active()
        if timeline is not None:
            timeline.seed_tool_output(item)
            self._sync_transcript_from_timeline()
            return
        payload = parse_tool_payload(item.get("output", {}))
        call = item.get("call") if isinstance(item.get("call"), dict) else None
        delta_index = item.get("tool_call_index")
        if (call is None or not tool_call_preview_line(call)) and isinstance(delta_index, int):
            call = self._tool_delta_calls.pop(delta_index, None) or call
        call_id = str((call or {}).get("call_id") or item.get("call", {}).get("call_id") or "")
        if call_id:
            self._tool_started_calls.pop(call_id, None)
            self._tool_partial_payloads.pop(call_id, None)
        pending_cell = self._tool_cells.pop(call_id, None) if call_id else None
        if pending_cell is None and isinstance(delta_index, int):
            pending_cell = self._tool_delta_cells.pop(delta_index, None)
            self._tool_delta_calls.pop(delta_index, None)
        elif isinstance(delta_index, int):
            self._tool_delta_cells.pop(delta_index, None)
            self._tool_delta_calls.pop(delta_index, None)
        if pending_cell is not None and call is not None:
            markup = self._tool_pending_markup(
                str(call.get("name") or "python"),
                "",
                call={**call, "_status_label": self._text("python_called")},
            )
            details = tool_call_detail_highlight_markup(call)
            if isinstance(pending_cell, ExpandableTranscriptCell):
                pending_cell.set_details(markup, details)
                self._mark_tool_cell_completed(pending_cell)
            elif tool_call_preview_line(call):
                new_cell = self._replace_with_expandable_cell(pending_cell, markup, details, "event")
                self._replace_process_cell(pending_cell, new_cell)
            else:
                pending_cell.update(markup)
                self._mark_tool_cell_completed(pending_cell)
        partial_lookup_item = {**item, "call": call} if call is not None else item
        partial_cell = self._partial_tool_result_cell(partial_lookup_item) if payload is not None else None
        if payload is None:
            markup = plain(f"{self._text('python')} {self._text('python_completed')}", style="dim")
            cell = self._append_cell(markup, "event")
            self._track_process_cell(cell)
            return

        payload = dict(payload)
        payload.pop("partial", None)
        payload.pop("partial_reason", None)
        payload.pop("call_id", None)
        self._last_tool_payload = payload
        markup = tool_timeline_markup(payload)
        details = tool_detail_markup(payload)
        if partial_cell is not None:
            # A partial result is only a live preview. Reuse that transcript cell
            # for the completed payload so the timeline never shows a stale
            # "still running" entry next to the final result.
            partial_cell.set_details(markup, details)
            partial_cell.tool_payload = payload
            self._refresh_status(self._text("working"))
            return
        cell = self._append_expandable_cell(markup, details, "event")
        cell.tool_payload = payload
        self._track_process_cell(cell)
        self._refresh_status(self._text("working"))

    def _mark_tool_cell_completed(self, cell: TranscriptCell) -> None:
        """Match re-entry rendering: completed tool calls use the 'event' class.

        While streaming the call is marked 'tool_pending'; once `tool.output`
        arrives the cell must look exactly like its history-rendered twin
        (see `_append_tool_call_history`).
        """
        if cell.has_class("tool_pending"):
            cell.remove_class("tool_pending")
        if not cell.has_class("event"):
            cell.add_class("event")

    def _append_tool_started(self, item: dict[str, Any]) -> None:
        timeline = self._timeline_for_active()
        if timeline is not None:
            timeline.seed_tool_started(item)
            self._sync_transcript_from_timeline()
            return
        call = item.get("call") or {}
        delta_index = item.get("tool_call_index")
        if isinstance(delta_index, int):
            self._tool_delta_calls[delta_index] = dict(call)
        call_id = str(call.get("call_id") or "")
        if call_id:
            self._tool_started_calls[call_id] = dict(call)
        name = str(call.get("name") or "python")
        detail = self._tool_call_preview(call)
        markup = self._tool_pending_markup(name, detail, call=call)
        cell = (
            self._tool_delta_cells.pop(delta_index, None)
            if isinstance(delta_index, int)
            else None
        )
        if cell is None and len(self._tool_delta_cells) == 1:
            _, cell = self._tool_delta_cells.popitem()
        details = tool_call_detail_highlight_markup(call)
        if cell is None:
            if tool_call_preview_line(call):
                cell = self._append_expandable_cell(markup, details, "tool_pending")
            else:
                cell = self._append_cell(markup, "tool_pending")
        elif isinstance(cell, ExpandableTranscriptCell):
            cell.set_details(markup, details)
        else:
            old_cell = cell
            cell = self._replace_with_expandable_cell(cell, markup, details, "tool_pending")
            self._replace_process_cell(old_cell, cell)
        if call_id:
            self._tool_cells[call_id] = cell
        self._track_process_cell(cell)
        self._refresh_status(self._text("running_python"))

    def _append_tool_pending_call(self, index: int, call: dict[str, Any]) -> None:
        name = str(call.get("name") or "python")
        detail = self._tool_call_preview(call)
        markup = self._tool_pending_markup(name, detail, call=call)
        if tool_call_preview_line(call):
            cell = self._append_expandable_cell(
                markup,
                tool_call_detail_highlight_markup(call),
                "tool_pending",
            )
        else:
            cell = self._append_cell(markup, "tool_pending")
        self._tool_delta_cells[index] = cell
        call_id = str(call.get("call_id") or "")
        if call_id:
            self._tool_cells[call_id] = cell
        self._track_process_cell(cell)

    def _append_tool_delta(self, item: dict[str, Any]) -> None:
        delta = item.get("tool_call")
        index = int(self._tool_call_field(delta, "index", 0) or 0)
        name = str(self._tool_call_field(delta, "name", None) or "python")
        call = {
            "call_id": self._tool_call_field(delta, "call_id", "") or "",
            "name": name,
            "arguments": self._tool_call_field(delta, "arguments", "")
            or self._tool_call_field(delta, "arguments_delta", ""),
        }
        self._tool_delta_calls[index] = call
        detail = self._tool_call_preview(call)
        markup = self._tool_pending_markup(name, detail, call=call)
        cell = self._tool_delta_cells.get(index)
        if cell is None:
            if tool_call_preview_line(call):
                cell = self._append_expandable_cell(
                    markup,
                    tool_call_detail_highlight_markup(call),
                    "tool_pending",
                )
            else:
                cell = self._append_cell(markup, "tool_pending")
            self._tool_delta_cells[index] = cell
        elif isinstance(cell, ExpandableTranscriptCell):
            cell.set_details(markup, tool_call_detail_highlight_markup(call))
        elif tool_call_preview_line(call):
            old_cell = cell
            cell = self._replace_with_expandable_cell(
                cell,
                markup,
                tool_call_detail_highlight_markup(call),
                "tool_pending",
            )
            self._replace_process_cell(old_cell, cell)
            self._tool_delta_cells[index] = cell
        else:
            cell.update(markup)
        self._track_process_cell(cell)
        self._refresh_status(self._text("writing_script"))

    def _tool_call_field(self, tool_call: Any, name: str, default: Any = None) -> Any:
        if isinstance(tool_call, dict):
            return tool_call.get(name, default)
        return getattr(tool_call, name, default)

    def _tool_pending_markup(
        self,
        name: str,
        detail: str,
        *,
        call: dict[str, Any] | None = None,
    ) -> Text:
        if call is not None and tool_call_preview_line(call):
            return tool_call_summary_markup(
                {**call, "_status_label": call.get("_status_label") or self._text("python_running")}
            )
        status = str((call or {}).get("_status_label") or self._text("python_running"))
        content = Text.assemble(
            ("⠿", "#7dd3fc"),
            " ",
            (name, "bold"),
            " ",
            (status, "dim"),
        )
        if detail:
            content.append(detail, style="dim")
        return content

    def _append_image_attachment_cell(
        self,
        attachment: dict[str, Any],
        *,
        before: MountBefore = None,
    ) -> ImageAttachmentCell:
        self._mark_transcript_content()
        cell = ImageAttachmentCell(attachment, classes="event")
        self.query_one("#transcript", VerticalScroll).mount(cell, before=before)
        self._scroll_end()
        return cell

    def _append_turn_error(
        self,
        event: dict[str, Any],
        *,
        before: MountBefore = None,
        display_content: object | None = None,
    ) -> TranscriptCell:
        error_type = str(event.get("error_type") or "Turn error")
        message = str(event.get("message") or "The turn stopped before producing a final response.")
        retryable = self._is_retryable_error_event(event)
        hint = self._text("retry_network_error_hint") if retryable else self._text("thread_stopped_after_error")
        content = cast(RenderablePart, display_content) if display_content is not None else Text.assemble((error_type, "bold red"), " ", message)
        cell = self._append_cell(
            join_lines([content, plain(hint, style="dim")]),
            "error",
            before=before,
            copy_text=f"{error_type}: {message}\n{hint}",
        )
        if retryable:
            self._append_retry_button(event, before=before)
        return cell

    def _append_stream_retry(
        self,
        event: dict[str, Any],
        *,
        before: MountBefore = None,
    ) -> TranscriptCell:
        attempt = event.get("attempt")
        max_attempts = event.get("max_attempts")
        delay_s = event.get("delay_s")
        error_type = str(event.get("error_type") or "stream")
        try:
            if delay_s is None:
                raise TypeError("delay_s is missing")
            delay_text = f"{float(delay_s):.1f}s"
        except (TypeError, ValueError):
            delay_text = "?s"
        return self._append_cell(
            plain(
                f"⟳ stream empty, retrying {attempt or '?'}/{max_attempts or '?'} "
                f"in {delay_text} ({error_type})",
                style="dim",
            ),
            "event",
            before=before,
        )

    def _append_retry_button(self, event: dict[str, Any], *, before: MountBefore = None) -> RetryTurnButton:
        self._mark_transcript_content()
        button = RetryTurnButton(self._text("retry"), thread_id=str(event.get("thread_id") or self.thread_id or ""))
        self.query_one("#transcript", VerticalScroll).mount(button, before=before)
        self._scroll_end()
        return button

    def _append_tool_call_history(self, item: dict[str, Any], *, before: MountBefore = None) -> ExpandableTranscriptCell:
        return self._append_expandable_cell(
            tool_call_summary_markup({**item, "_status_label": self._text("python_called")}),
            tool_call_detail_highlight_markup(item),
            "event",
            before=before,
        )

    def _tool_call_preview(self, call: dict[str, Any]) -> str:
        first = tool_call_preview_line(call, max_chars=72)
        if not first:
            return ""
        return f"\n{first}"

    def _append_cell(
        self,
        content: object,
        classes: str,
        *,
        before: MountBefore = None,
        copy_text: str | None = None,
    ) -> TranscriptCell:
        self._mark_transcript_content()
        cell = TranscriptCell(content, classes=classes, copy_text=copy_text)
        self.query_one("#transcript", VerticalScroll).mount(cell, before=before)
        self._scroll_end()
        return cell

    def _append_expandable_cell(
        self,
        summary: object,
        details: object,
        classes: str,
        *,
        before: MountBefore = None,
        detail_title: str = "tool_details",
        detail_hint: str = "tool_details_hint",
    ) -> ExpandableTranscriptCell:
        self._mark_transcript_content()
        cell = ExpandableTranscriptCell(
            summary,
            details,
            detail_title=detail_title,
            detail_hint=detail_hint,
            classes=classes,
        )
        self.query_one("#transcript", VerticalScroll).mount(cell, before=before)
        self._scroll_end()
        return cell

    def _append_reasoning_cell(
        self,
        summary: object,
        details: object,
        *,
        before: MountBefore = None,
    ) -> ExpandableTranscriptCell:
        self._mark_transcript_content()
        cell = ExpandableTranscriptCell(
            summary,
            details,
            detail_title="reasoning_details",
            detail_hint="reasoning_details_hint",
            classes="reasoning",
        )
        self.query_one("#transcript", VerticalScroll).mount(cell, before=before)
        self._scroll_end()
        return cell

    def _replace_with_expandable_cell(
        self,
        old_cell: TranscriptCell,
        summary: object,
        details: object,
        classes: str,
    ) -> ExpandableTranscriptCell:
        cell = ExpandableTranscriptCell(summary, details, classes=classes)
        self.query_one("#transcript", VerticalScroll).mount(cell, before=old_cell)
        old_cell.remove()
        return cell

    def _replace_with_reasoning_cell(
        self,
        old_cell: TranscriptCell,
        summary: object,
        details: object,
    ) -> ExpandableTranscriptCell:
        cell = ExpandableTranscriptCell(
            summary,
            details,
            detail_title="reasoning_details",
            detail_hint="reasoning_details_hint",
            classes="reasoning",
        )
        self.query_one("#transcript", VerticalScroll).mount(cell, before=old_cell)
        old_cell.remove()
        return cell

    def _mark_transcript_content(self) -> None:
        self._transcript_has_content = True
        try:
            self.query_one(EmptyState).add_class("hidden")
        except NoMatches:
            pass

    def _reset_transcript(self, *, show_empty: bool = True) -> None:
        transcript = self.query_one("#transcript", TranscriptScroll)
        transcript.query("*").remove()
        self._transcript_has_content = False
        self._history_before_event_id = None
        self._history_has_more = False
        self._history_more_cell = None
        # Brand new transcript: re-engage auto-follow so the first incoming
        # delta isn't stranded above the fold.
        transcript.follow_tail = True
        if show_empty:
            empty_state = EmptyState()
            transcript.mount(empty_state)
            self.call_after_refresh(empty_state.tick)

    def _open_threads_panel(self) -> None:
        threads = self.engine.thread_store.list_threads()
        if not threads:
            self._open_fullscreen_panel(
                self._text("threads"),
                plain(self._text("no_threads"), style="dim"),
            )
            return
        items = []
        for thread in threads:
            thread_id = str(thread.get("thread_id") or "")
            title = str(thread.get("title") or self._text("new_thread"))
            updated = str(thread.get("updated_at") or "")
            last_text = str(thread.get("last_text") or "").replace("\n", " ")
            if len(last_text) > 120:
                last_text = last_text[:117].rstrip() + "..."
            marker = f"{self._text('current')} " if thread_id == self.thread_id else ""
            state = self._thread_runs.get(thread_id)
            running = f" · {self._text('working')}" if state is not None and state.worker is not None else ""
            items.append(
                PickerItem(
                    id=thread_id,
                    title=f"{marker}{title}",
                    description=last_text or self._text("no_messages"),
                    meta=(
                        f"{short_thread(thread_id)} · {thread.get('turn_count', 0)} "
                        f"{self._text('turns')} · {updated}{running}"
                    ),
                )
            )
        self._open_picker(
            self._text("threads"),
            items,
            self._resume_thread,
            subtitle=self._text("thread_search_hint"),
        )

    def _open_status_panel(self) -> None:
        self._open_panel(self._status_panel_markup(), "status", self._text("status"))

    def _open_goal_panel(self, *, replace_current: bool = False) -> None:
        thread_id = self.thread_id
        if thread_id is not None:
            self.level = self._thread_metadata_level(thread_id) or self.level
        state = self.engine.goal_state(thread_id) if thread_id is not None else None
        persisted_enabled = bool(state and state.enabled)
        pending_enabled = not persisted_enabled and self._goal_enable_pending(thread_id)
        enabled = pending_enabled or persisted_enabled
        status = self._text("goal_enabled") if enabled else self._text("goal_disabled")
        files_hint = self._text("goal_files_pending") if thread_id is None else self._text("goal_files_hint")
        items = [
            PickerItem(
                id="enable",
                title=self._text("goal_enable"),
                description=self._text("current") if enabled else "",
                meta=self._text("goal_enable_hint"),
            ),
            PickerItem(
                id="disable",
                title=self._text("goal_disable"),
                description=self._text("current") if not enabled else "",
                meta=self._goal_disable_meta(thread_id, enabled),
            ),
            PickerItem(
                id="files",
                title=self._text("goal_files"),
                description=status,
                meta=files_hint,
            ),
            PickerItem(
                id="reset",
                title=self._text("goal_reset"),
                description="" if not enabled else self._text("goal_reset_disabled_active"),
                meta=self._text("goal_reset_hint"),
            ),
        ]
        self._open_picker(
            self._text("goal"),
            items,
            self._choose_goal_item,
            subtitle=self._text("goal_panel_hint"),
            navigate=True,
            replace_current=replace_current,
        )

    def _goal_disable_meta(self, thread_id: str | None, enabled: bool) -> str:
        if not enabled:
            return self._text("goal_disable_hint")
        if thread_id is None or self._goal_enable_pending(thread_id):
            return self._text("goal_disable_hint")
        if not self._goal_can_disable(thread_id):
            return self._text("goal_disable_requires_completed")
        return self._text("goal_disable_hint")

    def _choose_goal_item(self, item_id: str) -> None:
        thread_id = self.thread_id
        if thread_id is not None:
            self.level = self._thread_metadata_level(thread_id) or self.level
        state = self.engine.goal_state(thread_id) if thread_id is not None else None
        persisted_enabled = bool(state and state.enabled)
        pending_enabled = not persisted_enabled and self._goal_enable_pending(thread_id)
        enabled = pending_enabled or persisted_enabled
        if item_id == "enable":
            if persisted_enabled:
                self._flash(self._text("goal_enabled_flash"))
                self._refresh_status()
                self._open_goal_panel(replace_current=True)
                return
            self._set_pending_goal_enable(thread_id, True)
            self._flash(self._text("goal_enabled_flash"))
            self._refresh_status()
            self._open_goal_panel(replace_current=True)
            return
        if item_id == "disable":
            if pending_enabled:
                self._set_pending_goal_enable(thread_id, False)
                self._flash(self._text("goal_disabled_flash"))
                self._refresh_status()
                self._open_goal_panel(replace_current=True)
                return
            if not persisted_enabled:
                self._flash(self._text("goal_already_disabled"))
                self._open_goal_panel(replace_current=True)
                return
            if thread_id is None:
                self._flash(self._text("goal_already_disabled"))
                self._open_goal_panel(replace_current=True)
                return
            if not self._goal_can_disable(thread_id):
                self._flash(self._text("goal_disable_requires_completed"), severity="warning")
                self._open_goal_panel(replace_current=True)
                return
            self.engine.disable_goal_mode(thread_id)
            self.level = self._thread_metadata_level(thread_id) or self.level
            self._flash(self._text("goal_disabled_flash"))
            self._refresh_status()
            self._open_goal_panel(replace_current=True)
            return
        if item_id == "files":
            if thread_id is None:
                self._flash(self._text("goal_files_pending"), severity="warning")
                self._open_goal_panel(replace_current=True)
                return
            self._open_goal_files_panel(thread_id)
            return
        if item_id == "reset":
            if enabled:
                self._flash(self._text("goal_reset_disabled_active"), severity="warning")
                self._open_goal_panel(replace_current=True)
                return
            if thread_id is None:
                self._flash(self._text("goal_files_pending"), severity="warning")
                self._open_goal_panel(replace_current=True)
                return
            self.engine.reset_goal_files(thread_id)
            self.level = self._thread_metadata_level(thread_id) or self.level
            self._flash(self._text("goal_reset_flash"))
            self._open_goal_panel(replace_current=True)

    def _goal_can_disable(self, thread_id: str) -> bool:
        run_state = self._thread_state(thread_id)
        if run_state is not None and (run_state.worker is not None or run_state.queue):
            return False
        return self._thread_has_final_reply(thread_id)

    def _thread_has_final_reply(self, thread_id: str) -> bool:
        events, _ = self.engine.thread_store.read_recent_events(
            thread_id,
            limit=1,
            event_types={"turn.completed", "turn.interrupted", "turn.error"},
        )
        return bool(events and events[-1].get("type") == "turn.completed")

    def _open_goal_files_panel(self, thread_id: str) -> None:
        state = self.engine.goal_state(thread_id)
        if state is None:
            self._open_panel(plain(self._text("goal_files_pending"), style="dim"), "goal", self._text("goal_files"))
            return
        status = self._text("goal_enabled") if state.enabled or self._goal_enable_pending(thread_id) else self._text("goal_disabled")
        sections: list[object] = [
            Text.assemble((self._text("goal"), "bold"), " ", (status, "cyan")),
            Text(),
            Text.assemble("- state: ", str(state.paths.state)),
            Text.assemble("- checklist: ", str(state.paths.checklist)),
            Text.assemble("- document: ", str(state.paths.notes)),
        ]
        if state.objective:
            sections.extend([Text(), Text.assemble("- objective: ", state.objective)])
        sections.extend(
            [
                Text(),
                *self._goal_file_preview("goal.json", state.paths.state, kind="json"),
                Text(),
                *self._goal_file_preview("checklist.md", state.paths.checklist, kind="markdown"),
                Text(),
                *self._goal_file_preview("notes.md", state.paths.notes, kind="markdown"),
            ]
        )
        self._open_panel(join_lines(sections), "goal", self._text("goal_files"))

    def _goal_file_preview(self, label: str, path: Path, *, kind: Literal["json", "markdown"]) -> list[object]:
        """Render one durable goal file if it already exists.

        The goal files panel is a read-only inspection surface. Missing files are
        reported inline instead of being created as a side effect of opening the
        panel; creation/reset remains owned by the explicit goal-mode actions.
        """

        heading = Text.assemble((label, "bold cyan"), " ", (str(path), "dim"))
        if not path.is_file():
            return [heading, plain(self._text("goal_file_missing"), style="dim")]
        try:
            content, truncated = self._read_goal_file_preview(path)
        except OSError as exc:
            return [heading, plain(f"{self._text('goal_file_read_error')}: {exc}", style="red")]
        if not content:
            body: object = plain(self._text("goal_file_empty"), style="dim")
        elif kind == "json":
            body = plain(self._format_goal_json_preview(content))
        else:
            body = _markdown(content)
        parts: list[object] = [heading, body]
        if truncated:
            parts.append(
                plain(
                    self._text("goal_file_truncated").format(limit=GOAL_FILE_PREVIEW_MAX_CHARS),
                    style="dim",
                )
            )
        return parts

    def _read_goal_file_preview(self, path: Path) -> tuple[str, bool]:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            content = handle.read(GOAL_FILE_PREVIEW_MAX_CHARS + 1)
        if len(content) <= GOAL_FILE_PREVIEW_MAX_CHARS:
            return content, False
        return content[:GOAL_FILE_PREVIEW_MAX_CHARS].rstrip() + "\n", True

    @staticmethod
    def _format_goal_json_preview(content: str) -> str:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return content
        return json.dumps(parsed, ensure_ascii=False, indent=2)

    def _session_thread_ids_for_panel(self, kind: str) -> list[str]:
        active_ids = self._active_activity_thread_ids()
        completed_ids = self._completed_activity_thread_ids()
        if kind == "active":
            return active_ids
        if kind == "completed":
            return completed_ids
        return active_ids + completed_ids

    def _open_session_threads_panel(self, kind: str) -> None:
        thread_ids = self._session_thread_ids_for_panel(kind)
        if not thread_ids:
            key = {
                "active": "no_active_threads",
                "completed": "no_completed_threads",
            }.get(kind, "no_session_threads")
            self._open_panel(plain(self._text(key), style="dim"), "threads", self._text("threads"))
            return
        title_key = {
            "active": "active_threads_title",
            "completed": "completed_threads_title",
        }.get(kind, "session_threads_title")
        items = [self._session_thread_picker_item(thread_id) for thread_id in thread_ids]
        self._open_picker(
            self._text(title_key),
            items,
            self._resume_thread,
            subtitle=self._text("thread_search_hint"),
        )

    def _session_thread_picker_item(self, thread_id: str) -> PickerItem:
        metadata = self._thread_metadata(thread_id)
        title = str(metadata.get("title") or self._text("new_thread")).strip()
        state = self._thread_runs.get(thread_id)
        status = state.status if state is not None and state.worker is not None else self._text("python_completed")
        activity = self._thread_activity.get(thread_id)
        elapsed = format_elapsed(self._thread_elapsed_seconds(thread_id)) or "0s"
        marker = f"{self._text('current')} " if thread_id == self.thread_id else ""
        turn_count = int(metadata.get("turn_count") or 0)
        return PickerItem(
            id=thread_id,
            title=f"{marker}{title}",
            description=str(metadata.get("last_text") or self._text("no_messages")).replace("\n", " ")[:120],
            meta=(
                f"{short_thread(thread_id)} · {status} · {self._text('elapsed')} {elapsed} · "
                f"{turn_count} {self._text('turns')}"
                if activity is not None
                else short_thread(thread_id)
            ),
        )

    def _open_notifications_panel(self) -> None:
        self._top_notification_unread = 0
        self._top_notifications = [
            replace(notification, read=True) for notification in self._top_notifications
        ]
        if not self._top_notifications:
            items = [
                PickerItem(
                    id="",
                    title=self._text("no_notifications"),
                    meta=self._text("notifications_hint"),
                )
            ]
        else:
            items = [self._notification_picker_item(notification) for notification in reversed(self._top_notifications[-100:])]
        self._open_picker(
            self._text("notifications"),
            items,
            self._choose_notification,
            subtitle=self._text("notifications_hint"),
        )
        self._refresh_top_bar()

    def _notification_picker_item(self, notification: TopNotification) -> PickerItem:
        title = notification.title if notification.read else f"● {notification.title}"
        meta_parts = [notification.created_at]
        if notification.thread_id:
            meta_parts.append(short_thread(notification.thread_id))
        if notification.severity != "information":
            meta_parts.append(notification.severity)
        return PickerItem(
            id=notification.id,
            title=title,
            description=notification.message,
            meta=" · ".join(meta_parts),
        )

    def _choose_notification(self, notification_id: str) -> None:
        notification = next(
            (item for item in self._top_notifications if item.id == notification_id),
            None,
        )
        if notification is None or not notification.thread_id:
            return
        self._resume_thread(notification.thread_id)

    def _add_notification(
        self,
        title: str,
        message: str = "",
        *,
        thread_id: str | None = None,
        severity: str = "information",
    ) -> None:
        self._top_notifications.append(
            TopNotification(
                id=new_id("ntf"),
                title=title,
                message=message,
                created_at=utc_now_iso(),
                thread_id=thread_id,
                severity=severity,
            )
        )
        self._top_notification_unread += 1
        if self.is_mounted:
            self._refresh_top_bar()

    def _status_panel_markup(self) -> Text:
        self.engine.refresh_config()
        level_name = self.level or self.engine.config.runtime.default_level
        rules = self.engine.project_rule_context()
        billing_line = self._thread_billing_status_line()
        try:
            model = self.engine.config.model_for_level(self.level)
            provider = self.engine.config.provider_for_model(model)
            stats = self.engine.context_stats(self.thread_id, self.level)
            model_line = f"{model.name} -> {model.model}"
            provider_line = f"{provider.name} / {model.api}"
            context_line = (
                f"{stats.percent}% "
                f"({format_tokens(stats.used_tokens)} / {format_tokens(stats.context_window_tokens)}, "
                f"{stats.source})"
            )
            compress_line = (
                f"{'on' if self.engine.config.runtime.compression.enabled else 'off'} · "
                f"trigger {format_tokens(stats.threshold_tokens)} · "
                f"headroom {format_tokens(stats.headroom_tokens)}"
            )
        except ConfigError as exc:
            model_line = Text("not configured", style="red")
            provider_line = str(exc)
            context_line = "-"
            compress_line = "-"
        rules_line = f"{len(rules.rules)} {self._text('status_rules_loaded')}"
        if rules.truncated:
            rules_line += f" · {self._text('truncated')}"
        if rules.omitted_files:
            rules_line += f" · {rules.omitted_files} {self._text('status_rules_omitted')}"
        lines: list[Text] = [
            Text.assemble("- state: ", (self._last_status, "cyan")),
            Text.assemble("- version: ", (application_version(), "cyan")),
            Text.assemble("- goal: ", self._goal_status_line()),
            Text.assemble("- level: ", (level_name, "cyan")),
            Text.assemble("- model: ", model_line),
            Text.assemble("- provider/api: ", provider_line),
            Text.assemble("- context: ", context_line),
        ]
        if billing_line:
            lines.append(Text.assemble("- billing: ", billing_line))
        lines.extend(
            [
                Text.assemble("- compaction: ", compress_line),
                Text.assemble("- rules: ", rules_line),
                Text.assemble("- thread: ", short_thread(self.thread_id)),
                Text.assemble("- queued: ", str(self._active_queue_length())),
                Text.assemble("- user state: ", str(uv_agent_home())),
                Text.assemble("- project state: ", str(project_state_dir(self.project_root))),
                Text.assemble("- host: ", host_environment_line()),
                Text.assemble("- language: ", self.language.name),
            ]
        )
        background_runs = self._background_run_states()
        if background_runs:
            lines.append(
                Text.assemble("- background: ", (f"{len(background_runs)} {self._text('active_threads')}", "cyan"))
            )
            for run_state in background_runs[:6]:
                lines.append(
                    Text.assemble("  - ", short_thread(run_state.thread_id), ": ", run_state.status)
                )
            if len(background_runs) > 6:
                lines.append(Text(f"  - ... {len(background_runs) - 6} more"))
        if rules.rules:
            lines.append(Text())
            lines.append(Text(self._text("rules"), style="bold"))
            for rule in rules.rules[:6]:
                suffix = f" [{self._text('truncated')}]" if rule.truncated else ""
                lines.append(Text(f"- {rule.scope}: {rule.path}{suffix}"))
            if len(rules.rules) > 6:
                lines.append(Text(f"- ... {len(rules.rules) - 6} more"))
        return join_lines(lines)  # type: ignore[return-value]

    def _goal_status_line(self) -> Text:
        if self._goal_mode_enabled():
            return Text(self._text("goal_enabled"), style="cyan")
        return Text(self._text("goal_disabled"), style="dim")

    def _thread_billing_status_line(self) -> str:
        if not self.engine.config.pricing.models:
            return ""
        label = self._thread_billing_label(decimals=6)
        return label if label else "-"

    def _active_worktree_metadata(self) -> dict[str, Any] | None:
        """Return active worktree metadata for the current thread, if any."""

        if not self.thread_id:
            return None
        metadata = self._thread_metadata(self.thread_id)
        if str(metadata.get("worktree_status") or "").strip() != "active":
            return None
        if not metadata.get("worktree_branch") or not metadata.get("worktree_path"):
            return None
        return metadata

    def _open_worktree_panel(self, *, replace_current: bool = False) -> None:
        metadata = self._active_worktree_metadata()
        if metadata is None:
            items = [
                PickerItem(
                    id="create",
                    title=self._text("worktree_create"),
                    description=self._text("worktree_create_hint"),
                )
            ]
        else:
            branch = str(metadata.get("worktree_branch") or "")
            path = str(metadata.get("worktree_path") or "")
            items = [
                PickerItem(
                    id="info",
                    title=self._text("worktree_current"),
                    description=branch,
                    meta=path,
                ),
                PickerItem(
                    id="merge",
                    title=self._text("worktree_merge"),
                    description=self._text("worktree_merge_hint"),
                    meta=branch,
                ),
                PickerItem(
                    id="delete",
                    title=self._text("worktree_delete"),
                    description=self._text("worktree_delete_hint"),
                    meta=branch,
                ),
            ]
        self._open_picker(
            self._text("worktree_title"),
            items,
            self._choose_worktree_item,
            subtitle=self._text("worktree_panel_hint"),
            navigate=True,
            replace_current=replace_current,
        )

    def _choose_worktree_item(self, item_id: str) -> None:
        if item_id == "create":
            self._open_worktree_create_panel()
            return
        if item_id == "info":
            self._open_worktree_info_panel()
            return
        if item_id == "merge":
            self._append_worktree_merge_prompt()
            self._close_active_panel()
            return
        if item_id == "delete":
            self._open_worktree_delete_panel()

    def _open_worktree_create_panel(self) -> None:
        self._close_active_panel()
        panel = WorktreeBranchPanel(
            title=self._text("worktree_create_title"),
            subtitle=self._text("worktree_branch_hint"),
            placeholder=self._text("worktree_branch_placeholder"),
        )

        def handle(branch: str | None) -> None:
            if branch:
                self._create_worktree_from_input(branch)
            self.query_one("#composer", TextArea).focus()

        self.push_screen(panel, handle)

    def _create_worktree_from_input(self, value: str) -> None:
        branch = value.strip()
        try:
            validate_worktree_branch_name(branch)
            info = create_worktree(self.project_root, branch, run=self._run_worktree_command)
        except WorktreeError as exc:
            self._flash(f"{self._text('worktree_error')}: {exc}", severity="error")
            self._open_worktree_create_panel()
            return
        thread_id = self.engine.thread_store.create_thread(f"Worktree {info.branch}")
        self.engine.thread_store.append(thread_id, "thread.worktree_created", **info.metadata())
        self.engine.thread_store.append(thread_id, "thread.cwd_updated", cwd=str(info.path))
        self._thread_timelines.pop(thread_id, None)
        self._resume_thread(thread_id)
        self._append_cell(
            Text.assemble(
                (self._text("worktree_created"), "dim"),
                " ",
                (info.branch, "cyan"),
                " · ",
                str(info.path),
            ),
            "event",
        )
        self._refresh_status(self._text("worktree_created"))

    def _open_worktree_info_panel(self) -> None:
        metadata = self._active_worktree_metadata()
        if metadata is None:
            self._open_panel(plain(self._text("worktree_none"), style="dim"), "worktree", self._text("worktree_title"))
            return
        lines = [
            Text(self._text("worktree_current"), style="bold cyan"),
            Text.assemble("- branch: ", str(metadata.get("worktree_branch") or "")),
            Text.assemble("- path: ", str(metadata.get("worktree_path") or "")),
            Text.assemble("- base: ", str(metadata.get("worktree_base_ref") or "")),
            Text.assemble("- origin: ", str(metadata.get("worktree_origin_root") or "")),
            Text(),
            Text.assemble("- ", (self._text("worktree_merge"), "cyan"), ": ", self._text("worktree_merge_hint")),
            Text.assemble("- ", (self._text("worktree_delete"), "red"), ": ", self._text("worktree_delete_hint")),
        ]
        self._open_panel(join_lines(lines), "worktree", self._text("worktree_title"))

    def _append_worktree_merge_prompt(self) -> None:
        metadata = self._active_worktree_metadata()
        if metadata is None:
            self._flash(self._text("worktree_none"), severity="warning")
            return
        branch = str(metadata.get("worktree_branch") or "")
        path = str(metadata.get("worktree_path") or "")
        origin = str(metadata.get("worktree_origin_root") or self.project_root)
        prompt = (
            f"请将当前 worktree 分支 `{branch}` 的工作合并回主工作区 `{origin}`。\n"
            f"当前 worktree 路径是 `{path}`。请先检查 worktree 和主工作区的 `git status`、"
            "当前分支、`git worktree list` 和差异。若主工作区有未提交改动、合并会覆盖用户改动、"
            "或者需要处理冲突，请先说明情况并谨慎处理。合并后运行合适的验证。"
            "不要自动删除 worktree 或分支；完成后告诉我可以在 Worktree 面板中点击“删除 worktree 和分支”。"
        )
        composer = self.query_one("#composer", TextArea)
        existing = composer.text.rstrip()
        composer.load_text(f"{existing}\n\n{prompt}" if existing else prompt)
        composer.focus()
        if self.thread_id:
            self.engine.thread_store.append(self.thread_id, "thread.worktree_merge_prompted")
        self._flash(self._text("worktree_prompt_appended"))

    def _open_worktree_delete_panel(self) -> None:
        metadata = self._active_worktree_metadata()
        if metadata is None:
            self._flash(self._text("worktree_none"), severity="warning")
            return
        branch = str(metadata.get("worktree_branch") or "")
        path = str(metadata.get("worktree_path") or "")
        items = [
            PickerItem(
                id="confirm_delete",
                title=self._text("worktree_delete_confirm"),
                description=self._text("worktree_delete_confirm_hint"),
                meta=f"{branch} · {path}",
            )
        ]
        self._open_picker(
            self._text("worktree_delete"),
            items,
            self._delete_worktree_from_panel,
            subtitle=self._text("worktree_delete_confirm_hint"),
            navigate=True,
        )

    def _delete_worktree_from_panel(self, item_id: str) -> None:
        if item_id != "confirm_delete" or not self.thread_id:
            return
        metadata = self._active_worktree_metadata()
        if metadata is None:
            self._flash(self._text("worktree_none"), severity="warning")
            return
        branch = str(metadata.get("worktree_branch") or "")
        path = Path(str(metadata.get("worktree_path") or ""))
        try:
            result = cleanup_worktree(self.project_root, branch, path, run=self._run_worktree_command)
        except WorktreeError as exc:
            self._flash(f"{self._text('worktree_error')}: {exc}", severity="error")
            return
        self.engine.thread_store.append(
            self.thread_id,
            "thread.worktree_deleted",
            worktree_branch=result.branch,
            worktree_path=str(result.path),
            worktree_origin_root=str(result.origin_root),
            worktree_deleted_at=utc_now_iso(),
            worktree_deleted_head=result.head,
            worktree_deleted_status=result.status,
            worktree_removed=result.worktree_removed,
            branch_deleted=result.branch_deleted,
        )
        self.engine.thread_store.append(self.thread_id, "thread.cwd_updated", cwd=str(self.project_root.resolve()))
        rule_states = getattr(self.engine, "_rule_states", None)
        if isinstance(rule_states, dict):
            rule_states.pop(self.thread_id, None)
        self._append_cell(
            Text.assemble((self._text("worktree_deleted"), "dim"), " ", (result.branch, "cyan")),
            "event",
        )
        self._close_active_panel()
        self._refresh_status(self._text("worktree_deleted"))

    def _run_worktree_command(self, args: list[str], *, cwd: Path, timeout_s: float | None = None) -> CommandResult:
        import subprocess

        completed = subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        return CommandResult(
            args=list(args),
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    def _open_command_palette(self, *, query: str = "") -> None:
        self._open_picker(
            self._text("command_palette"),
            self._command_palette_items(),
            self._choose_command,
            subtitle=self._text("command_filter_hint"),
            initial_filter=query,
            navigate=True,
        )

    def _command_palette_items(self) -> list[PickerItem]:
        items = [
            PickerItem(
                id=spec.name,
                title=spec.palette_title,
                description=spec.description,
            )
            for spec in self._commands()
        ]
        items.append(
            PickerItem(
                id="worktree",
                title=self._text("worktree_title"),
                description=self._text("worktree_panel_hint"),
            )
        )
        mention_items: list[PickerItem] = []
        for item in self._mcp_mention_items():
            mention_items.append(
                PickerItem(
                    id=f"mcp:{item.id}",
                    title=f"mcp:{item.title}",
                    description=item.description,
                    meta=item.meta,
                )
            )
        for item in self._skill_mention_items():
            mention_items.append(
                PickerItem(
                    id=f"skill:{item.id}",
                    title=f"skill:{item.title}",
                    description=item.description,
                    meta=item.meta,
                )
            )
        if mention_items:
            items.append(PickerItem(id="", title=""))
            items.extend(mention_items)
        return items

    def _choose_command(self, command: str) -> None:
        if command == "worktree":
            self._open_worktree_panel()
            return
        if command.startswith("mcp:"):
            self._choose_mcp_mention(command.removeprefix("mcp:"))
            self._close_active_panel()
            return
        if command.startswith("skill:"):
            self._choose_skill_mention(command.removeprefix("skill:"))
            self._close_active_panel()
            return
        spec = next((item for item in self._commands() if item.name == command), None)
        if spec is None:
            return
        self._handle_command(command)

    def _cell_text(self, cell: TranscriptCell) -> str:
        parts = [str(cell.copy_text or "")]
        if isinstance(cell, ExpandableTranscriptCell):
            parts.append(renderable_plain(cell.details) or "")
        try:
            parts.append(str(cell.render()))
        except Exception:
            pass
        return "\n".join(parts)

    def _resume_thread(self, thread_id: str) -> None:
        if not thread_id:
            return
        self._save_active_thread_view_state()
        run_state = self._thread_runs.get(thread_id)
        self.thread_id = thread_id
        self.level = self._thread_metadata_level(thread_id)
        self._reset_live_view_state()
        self._render_thread_history(thread_id, sync=False)
        self._refresh_active_run_state()
        self._refresh_pending_images()
        self._refresh_pending_turns()
        self._refresh_status(run_state.status if run_state is not None and run_state.worker is None else self._text("resumed"))
        self._sync_transcript_from_timeline(restore_view=True)
        self._close_active_panel()

    def _render_thread_history(self, thread_id: str, *, sync: bool = True) -> None:
        timeline = self._timeline_for_thread(thread_id)
        if timeline is None:
            return
        if not timeline.history_loaded:
            segment = self.engine.thread_store.read_history_segment(
                thread_id,
                event_types=VISIBLE_HISTORY_EVENT_TYPES | {"turn.started", "turn.completed"},
            )
            if timeline.items or timeline.process_groups or timeline.active_turns:
                timeline.merge_history_segment(
                    segment.events,
                    start_event_id=segment.start_event_id,
                    end_event_id=segment.end_event_id,
                    has_older=segment.has_more,
                )
            else:
                timeline.load_history_segment(
                    segment.events,
                    start_event_id=segment.start_event_id,
                    end_event_id=segment.end_event_id,
                    has_older=segment.has_more,
                )
        if sync and self._is_active_thread(thread_id):
            self._sync_transcript_from_timeline()

    def _load_older_thread_history(self) -> None:
        if not self.thread_id or self._history_before_event_id is None:
            return
        timeline = self._timeline_for_active()
        if timeline is None:
            return
        segment = self.engine.thread_store.read_history_segment(
            self.thread_id,
            before_event_id=self._history_before_event_id,
            event_types=VISIBLE_HISTORY_EVENT_TYPES | {"turn.started", "turn.completed"},
        )
        timeline.prepend_history_segment(
            segment.events,
            start_event_id=segment.start_event_id,
            has_older=segment.has_more,
        )
        self._sync_transcript_from_timeline()

    def _history_turn_elapsed_label(self, events: list[dict[str, Any]]) -> str:
        started_at = ""
        completed_at = ""
        for event in events:
            event_type = event.get("type")
            if not started_at and event_type in {"turn.started", "item.user"}:
                started_at = str(event.get("created_at") or "")
            if event_type in {"turn.completed", "turn.interrupted", "turn.error"}:
                completed_at = str(event.get("created_at") or "")
        seconds = _elapsed_between(started_at, completed_at)
        return format_elapsed(seconds) if seconds is not None else ""

    def _append_user_from_history(self, item: dict[str, Any], *, before: MountBefore = None) -> TranscriptCell | None:
        self._reasoning_cell = None
        self._reasoning_buffer = ""
        parts = []
        for content in item.get("content") or []:
            if content.get("type") in {"input_text", "text"}:
                parts.append(str(content.get("text") or ""))
        if parts:
            return self._append_user("\n".join(parts), before=before)
        return None

    def _model_response_text(self, output: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in output:
            if item.get("type") != "message":
                continue
            text = self._message_item_text(item)
            if text:
                parts.append(text)
        return "".join(parts)

    def _message_item_text(self, item: dict[str, Any]) -> str:
        parts: list[str] = []
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text", "refusal"}:
                parts.append(str(content.get("text") or ""))
        return "".join(parts)

    def _open_fullscreen_panel(self, title: str, content: object, *, subtitle: str = "") -> None:
        self.push_screen(FullscreenPanel(title=title, body=content, subtitle=subtitle))

    def _active_fullscreen_panel(self) -> FullscreenPanel | None:
        if self.screen_stack and isinstance(self.screen_stack[-1], FullscreenPanel):
            return self.screen_stack[-1]
        return None

    def _open_picker(
        self,
        title: str,
        items: list[PickerItem],
        callback: Callable[[str], None],
        *,
        subtitle: str = "",
        initial_filter: str = "",
        mention_kind: str | None = None,
        mention_items: Callable[[str], tuple[str, list[PickerItem], str]] | None = None,
        navigate: bool = False,
        close_on_select: bool = False,
        replace_current: bool = False,
    ) -> None:
        panel = self._active_fullscreen_panel()
        if panel is not None and replace_current:
            panel.replace_picker(
                title=title,
                items=items,
                callback=callback,
                subtitle=subtitle,
                initial_filter=initial_filter,
                close_on_select=close_on_select,
            )
            return
        if panel is not None and panel.can_navigate:
            panel.navigate_picker(
                title=title,
                items=items,
                callback=callback,
                subtitle=subtitle,
                initial_filter=initial_filter,
                close_on_select=close_on_select,
            )
            return
        panel = FullscreenPanel(
            title=title,
            items=items,
            subtitle=subtitle,
            initial_filter=initial_filter,
            mention_kind=mention_kind,
            mention_items=mention_items,
            select_callback=callback if navigate else None,
            close_on_select=close_on_select,
            navigation_enabled=navigate,
        )

        def handle(result: str | None) -> None:
            if result:
                if mention_kind is not None:
                    kind = panel._selected_mention_kind
                    if kind == "thread":
                        self._choose_thread_mention(result)
                    elif kind == "file":
                        self._choose_file_mention(result)
                    else:
                        callback(result)
                else:
                    callback(result)
            self.query_one("#composer", TextArea).focus()

        self.push_screen(panel, handle)

    def _open_panel(self, content: object, name: str | None = None, title: str | None = None) -> None:
        panel_title = title or (name.title() if name else self._text("panel"))
        panel = self._active_fullscreen_panel()
        if panel is not None and panel.can_navigate:
            panel.navigate_panel(title=panel_title, body=content, subtitle=self._text("panel_closes"))
            self._refresh_status()
            return
        self._open_fullscreen_panel(panel_title, content, subtitle=self._text("panel_closes"))
        self._refresh_status()

    def _resize_composer(self) -> None:
        try:
            composer = self.query_one("#composer", TextArea)
        except NoMatches:
            return
        line_count = self._composer_visual_line_count(composer)
        if line_count < COMPOSER_AUTO_EXPAND_LINES and self._composer_height_override == "collapsed":
            self._composer_height_override = None
        if self._composer_height_override == "expanded":
            expanded = True
        elif self._composer_height_override == "collapsed":
            expanded = False
        else:
            expanded = line_count >= COMPOSER_AUTO_EXPAND_LINES
        self._composer_expanded = expanded
        height = self._expanded_composer_height() if expanded else min(
            COMPOSER_COLLAPSED_HEIGHT,
            self._maximum_composer_height(),
        )
        composer.styles.height = height
        if self.is_mounted:
            # Textual's TextArea.edit() triggers scroll_cursor_visible() before
            # virtual_size is refreshed, so when the composer is at its max
            # height a freshly inserted newline leaves the cursor off-screen
            # (max_scroll_y is clamped against the stale virtual_size). Defer a
            # second scroll_cursor_visible() until after the layout pass so the
            # textarea can scroll using the up-to-date geometry.
            self.call_after_refresh(self._refresh_composer_overlay)
            self.call_after_refresh(composer.scroll_cursor_visible)

    def _composer_visual_line_count(self, composer: TextArea) -> int:
        if composer.soft_wrap:
            composer.wrapped_document.wrap(composer.wrap_width, tab_width=composer.indent_width)
            return max(1, composer.wrapped_document.height)
        return max(1, composer.document.line_count)

    def _expanded_composer_height(self) -> int:
        return min(COMPOSER_EXPANDED_HEIGHT, self._maximum_composer_height())

    def _flash(self, message: str, *, severity: NotificationSeverity = "information") -> None:
        self.notify(message, severity=severity, timeout=2.0)
        self._last_status = message
        self._refresh_status()

    def _notify_turn_completed(self, thread_id: str, final_text: str, *, active_thread: bool) -> None:
        config = self.engine.config.ui.completion_notification
        if not config.enabled:
            return
        if config.terminal and not active_thread:
            self._notify_background_thread_completed(thread_id)
        if config.bell:
            self.bell()
            play_completion_sound()

    def _notify_background_thread_completed(self, thread_id: str) -> None:
        digest = self.engine.thread_store.thread_digest(thread_id)
        title = str(digest.get("title") or self._text("new_thread")).strip()
        if len(title) > 48:
            title = title[:45].rstrip() + "..."
        self._add_notification(
            self._text("background_thread_completed"),
            f"{title or self._text('new_thread')} · {short_thread(thread_id)}",
            thread_id=thread_id,
        )

    def _refresh_top_bar(self) -> None:
        if not self.is_mounted:
            return
        active_ids = self._active_activity_thread_ids()
        completed_ids = self._completed_activity_thread_ids()
        elapsed = format_elapsed(self._thread_elapsed_seconds(self.thread_id)) or "0s"
        mode = self._current_mode_label()
        mode_style = GOAL_MODE_STYLE if self._goal_mode_enabled() else "cyan"
        unread = self._top_notification_unread
        notification_count = len(self._top_notifications)
        try:
            elapsed_widget = self.query_one("#top-bar-elapsed", Static)
            mode_widget = self.query_one("#top-bar-mode", Static)
            worktree_widget = self.query_one("#top-bar-worktree", Static)
            active_widget = self.query_one("#top-bar-active", Static)
            completed_widget = self.query_one("#top-bar-completed", Static)
            notification_widget = self.query_one("#top-bar-notifications", Static)
        except NoMatches:
            return
        elapsed_widget.update(
            Text.assemble((self._text("current_thread_elapsed"), "dim"), " ", (elapsed, "cyan"))
        )
        mode_widget.update(Text.assemble((self._text("mode"), "dim"), " ", (mode, mode_style)))
        if self._active_worktree_metadata():
            worktree_widget.remove_class("hidden")
            worktree_widget.update(Text(self._text("worktree"), style="bold cyan"))
        else:
            worktree_widget.add_class("hidden")
            worktree_widget.update("")
        active_widget.update(
            Text.assemble(
                (self._text("thread_activity_active"), "dim"),
                " ",
                (str(len(active_ids)), "cyan" if active_ids else "dim"),
            )
        )
        completed_widget.update(
            Text.assemble(
                (self._text("thread_activity_completed"), "dim"),
                " ",
                (str(len(completed_ids)), "cyan" if completed_ids else "dim"),
            )
        )
        if unread:
            notification_widget.update(
                Text.assemble(
                    (self._text("notifications_short"), "cyan"),
                    " ",
                    (str(unread), "bold cyan"),
                    (f"/{notification_count}", "dim"),
                )
            )
        else:
            notification_widget.update(
                Text.assemble((self._text("notifications_short"), "dim"), " ", (str(notification_count), "dim"))
            )

    def _refresh_top_bar_elapsed(self) -> None:
        """Refresh only the top-bar fields that visibly change during ticks."""

        if not self.is_mounted:
            return
        elapsed = format_elapsed(self._thread_elapsed_seconds(self.thread_id)) or "0s"
        try:
            elapsed_widget = self.query_one("#top-bar-elapsed", Static)
        except NoMatches:
            return
        elapsed_widget.update(
            Text.assemble((self._text("current_thread_elapsed"), "dim"), " ", (elapsed, "cyan"))
        )

    def _refresh_status(self, state: str | None = None) -> None:
        if state is not None:
            self._last_status = state
        self.engine.refresh_config()
        self.language = detect_user_language(self.engine.config.ui.language)
        billing_label = self._thread_billing_label(decimals=4)
        try:
            self.query_one("#composer", TextArea).placeholder = self._text("placeholder")
        except NoMatches:
            pass
        level_name = self.level or self.engine.config.runtime.default_level
        try:
            stats = self.engine.context_stats(self.thread_id, self.level)
            compact_context = f"{stats.percent}%"
        except ConfigError:
            compact_context = "?"
        self._status_level_name = level_name
        self._status_compact_context = compact_context
        self._status_thread_label = short_thread(self.thread_id)
        self._status_billing_label = billing_label
        state_text = self._last_status
        if self.busy and state_text == self._text("idle"):
            state_text = self._text("working")
        footer = self._status_footer(state_text=state_text)
        self._refresh_window_title()
        self._refresh_top_bar()
        try:
            self.query_one("#composer-footer", Static).update(footer)
        except NoMatches:
            return
        self._refresh_pending_images()
        self._refresh_pending_turns()

    def _status_footer(self, *, state_text: str) -> Text:
        """Render the footer from cached status metadata.

        Busy spinner ticks happen several times per second; keeping this pure and
        cache-backed avoids re-reading config, context stats, and thread metadata
        just to advance the elapsed timer by one frame.
        """

        level_name = self._status_level_name
        compact_context = self._status_compact_context
        thread_label = self._status_thread_label
        billing_label = self._status_billing_label
        if self.busy:
            spinner = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)] + " "
            elapsed_suffix = ""
            elapsed_seconds = _elapsed_between(self._turn_started_at)
            if elapsed_seconds is None and self._busy_started_at is not None:
                elapsed_seconds = monotonic() - self._busy_started_at
            if elapsed_seconds is not None:
                elapsed_suffix = f" ({format_elapsed(elapsed_seconds)})"
            footer = Text.assemble(
                (f"{spinner}{state_text}", "cyan"),
                (elapsed_suffix, "dim"),
                " ",
                (f"{level_name} · {compact_context} · {thread_label}", "dim"),
            )
        else:
            footer = Text(f"{level_name} · {compact_context} · {thread_label}", style="dim")
        if billing_label:
            footer.append(" · ", style="dim")
            footer.append(billing_label, style="dim")
        return footer

    def _refresh_busy_status(self) -> None:
        """Advance spinner/elapsed UI without the heavyweight status refresh."""

        state_text = self._last_status
        if state_text == self._text("idle"):
            state_text = self._text("working")
        self._apply_window_title()
        self._refresh_top_bar_elapsed()
        try:
            self.query_one("#composer-footer", Static).update(self._status_footer(state_text=state_text))
        except NoMatches:
            return

    def _refresh_status_from_cache(self) -> None:
        """Refresh footer/top-bar chrome without re-reading config or history."""

        state_text = self._last_status
        if self.busy and state_text == self._text("idle"):
            state_text = self._text("working")
        self._apply_window_title()
        self._refresh_top_bar_elapsed() if self.busy else self._refresh_top_bar()
        try:
            self.query_one("#composer-footer", Static).update(self._status_footer(state_text=state_text))
        except NoMatches:
            return
        self._refresh_pending_images()
        self._refresh_pending_turns()

    def _current_mode_label(self) -> str:
        if self._goal_mode_enabled():
            return self._text("goal")
        return self._text("mode_normal") if self._interaction_mode == "normal" else self._interaction_mode

    def _goal_mode_enabled(self) -> bool:
        if self._goal_enable_pending(self.thread_id):
            return True
        if not self.thread_id:
            return False
        goal_state = self.engine.goal_state(self.thread_id)
        return goal_state is not None and goal_state.enabled

    def _thread_billing_label(self, *, decimals: int) -> str:
        """Return the current thread's formatted cost, or empty when disabled."""

        if self.thread_id is None or not self.engine.config.pricing.models:
            return ""
        metadata = self._thread_metadata(self.thread_id)
        total = billing_total_from_metadata(
            metadata,
            preferred_currency=self.engine.config.pricing.currency,
        )
        if total is None:
            return ""
        amount, currency = total
        return format_billing_total(amount, currency, decimals=decimals)

    def _any_thread_running(self) -> bool:
        return any(
            run_state.worker is not None
            for run_state in self._thread_runs.values()
        )

    def _current_thread_title(self) -> str:
        fallback = self._text("new_thread")
        if self.thread_id is None:
            return fallback
        try:
            digest = self.engine.thread_store.thread_digest(self.thread_id)
        except Exception:
            return fallback
        title = str(digest.get("title") or "").strip()
        # Until the auto-titler renames the thread, the store returns its own
        # localized default ("New thread" / "新会话"). Treat those placeholders
        # as "no real title yet" so the window title keeps showing the user's
        # currently-selected language instead of flipping between locales.
        if not title or title in DEFAULT_THREAD_TITLES:
            return fallback
        return title

    def _refresh_window_title(self) -> None:
        self._window_title_thread_title = self._current_thread_title()
        self._apply_window_title()

    def _apply_window_title(self) -> None:
        title = self._window_title_thread_title or self._text("new_thread")
        if self.busy or self._any_thread_running():
            spinner = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
            title = f"{spinner} {title}"
        title = sanitized_window_title(title)
        if title == self._last_window_title:
            return
        self._last_window_title = title
        # Rich's Console.set_window_title goes through Textual's render pipeline
        # and never reaches the real terminal while the App is running. Write
        # the OSC sequence directly to the original stdout (which Textual leaves
        # untouched at the Python level) so Windows Terminal / xterm-style hosts
        # actually update their title bar.
        write_window_title(title)

    def _scroll_end(self) -> None:
        transcript = self.query_one("#transcript", TranscriptScroll)
        if not transcript.follow_tail:
            # User dragged the scrollbar / pressed PgUp; don't yank them back to
            # the bottom on every streaming SSE delta. They re-engage follow by
            # pressing the "↓ bottom" button above the composer.
            return
        transcript.programmatic_scroll_end()
