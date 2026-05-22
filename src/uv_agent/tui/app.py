from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from rich.markup import escape
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.geometry import Offset
from textual.screen import Screen
from textual.reactive import reactive
from textual.selection import Selection
from textual.widgets import Button, Static, TextArea
from textual.worker import Worker

from uv_agent.atomic import atomic_replace
from uv_agent.billing import billing_total_from_metadata, format_billing_total
from uv_agent.config import ConfigError
from uv_agent.environment import application_version, detect_user_language, host_environment_line
from uv_agent.errors import (
    error_markup,
    escape_markup as escape_error_markup,
    format_error,
    is_retryable_provider_error,
)
from uv_agent.i18n import command_description, tr
from uv_agent.notifications import play_completion_sound
from uv_agent.paths import project_state_dir, project_tui_clipboard_dir, uv_agent_home
from uv_agent.session.store import VISIBLE_HISTORY_EVENT_TYPES
from uv_agent.thread_titles import DEFAULT_THREAD_TITLES
from uv_agent.time import utc_now_iso
from uv_agent.tui.config_panels import ConfigPanelMixin
from uv_agent.tui.formatting import (
    format_elapsed,
    format_tokens,
    parse_tool_payload,
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
    ToolDetailsPanel,
)
from uv_agent.tui.state import (
    CommandSpec,
    MentionScanCache,
    PendingImage,
    PickerItem,
    QueuedTurn,
    ThreadRunState,
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


COMPOSER_COLLAPSED_HEIGHT = 5
COMPOSER_EXPANDED_HEIGHT = 8
COMPOSER_AUTO_EXPAND_LINES = 3
QUIT_KEY_DEBOUNCE_SECONDS = 0.08
MAX_COMPOSER_HISTORY = 50
COMPOSER_HISTORY_FILENAME = "composer_history.json"
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


__all__ = [
    "EmptyState",
    "ExpandableTranscriptCell",
    "FoldedProcessCell",
    "FullscreenPanel",
    "ImageAttachmentCell",
    "ImagePreviewPanel",
    "PendingImage",
    "PendingImagePreviewPanel",
    "RetryTurnButton",
    "ToolDetailsPanel",
    "TranscriptCell",
    "TranscriptScroll",
    "UvAgentApp",
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
    ("/config", "/config"),
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


def _event_offset(event: dict[str, Any] | None) -> int | None:
    if not event:
        return None
    value = event.get("_jsonl_offset")
    return value if isinstance(value, int) else None


class TranscriptScreen(Screen[None]):
    """Default screen with tighter transcript selection behavior."""

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
        self.selections = {
            start_widget: Selection.from_offsets(
                start_offset,
                self._inclusive_selection_end(start_offset, end_offset),
            )
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
        self._history_before_offset: int | None = None
        self._history_has_more = False
        self._history_more_cell: LoadOlderHistoryCell | None = None
        self._selection_copy_timer: Any | None = None
        self._pending_selection_copy = ""
        self._last_auto_copied_selection = ""
        self._composer_height_override: str | None = None
        self._composer_expanded = False
        self._pending_images: list[PendingImage] = []
        self._tool_delta_calls: dict[int, dict[str, Any]] = {}
        self._mention_file_cache = MentionScanCache()
        self._mention_thread_cache = MentionScanCache()
        self._mention_file_cache_dirty = False
        self._mention_file_watcher_worker: Worker[None] | None = None
        self._mention_file_watcher_stop = threading.Event()
        self._window_title_thread_title = ""
        self._last_window_title = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="main-column"):
            with TranscriptScroll(id="transcript"):
                yield EmptyState()
            yield Static("", id="pending-images-btn", classes="hidden")
            yield Static(
                f"↓ {tr(self.language, 'back_to_bottom')}",
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
        self.query_one(EmptyState).tick()
        self._refresh_status(self._text("idle"))
        self.set_interval(0.16, self._tick, name="spinner")
        self.query_one("#composer", TextArea).focus()
        transcript = self.query_one("#transcript", TranscriptScroll)
        self.watch(transcript, "near_bottom", self._on_near_bottom_changed)
        self._refresh_pending_images()
        self._refresh_composer_overlay()

    def on_unmount(self) -> None:
        self._mention_file_watcher_stop.set()
        if self._mention_file_watcher_worker is not None:
            self._mention_file_watcher_worker.cancel()

    def _on_near_bottom_changed(self, near: bool) -> None:
        self._refresh_composer_overlay()

    def on_click(self, event: events.Click) -> None:
        widget = getattr(event, "widget", None)
        if widget is not None and widget.id == "pending-images-btn":
            event.stop()
            self._open_pending_image_preview()
        elif widget is not None and widget.id == "scroll-to-bottom-btn":
            event.stop()
            try:
                transcript = self.query_one("#transcript", TranscriptScroll)
            except NoMatches:
                return
            transcript.engage_follow_tail()
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
            reserved = transcript.styles.min_height.value or 0
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
        self._refresh_status()
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
        self._refresh_status()
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
            self._refresh_status()
        else:
            self._apply_window_title()

    def _text(self, key: str) -> str:
        return tr(self.language, key)

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
        image_paths = [image.path for image in pending_images]
        self._pending_images.clear()
        self._refresh_pending_images()
        thread_id = self._ensure_active_thread()
        level = self._current_level_for_thread(thread_id)
        self._persist_thread_level(thread_id, level)
        active_run = self._active_run_state()
        if active_run is not None:
            run_state = active_run
            run_state.queue.append(QueuedTurn(prompt=prompt, level=level, image_paths=image_paths))
            if self._is_active_thread(run_state.thread_id):
                self._append_cell(
                    self._queued_turn_markup(prompt, image_paths),
                    "event",
                )
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
        if self._is_active_thread(thread_id):
            self.query_one("#composer-shell", Vertical).add_class("busy")
            self._reset_live_view_state()
            self._interrupt_armed = False
            self._process_anchor_cell = self._append_user(prompt)
            self._turn_started_at = started_at
            self._turn_completed_at = None
            for image_path in image_paths or []:
                self._append_cell(
                    f"[dim]{escape(self._text('image_pending_sent'))}[/dim] "
                    f"[cyan]{escape(Path(image_path).name)}[/cyan]",
                    "event",
                )
            self._reasoning_cell = self._append_cell(
                f"[dim]{escape(self._text('thinking'))}...[/dim]",
                "event",
            )
            self._sync_run_state_from_active(run_state)
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
        if self._is_active_thread(thread_id):
            self.query_one("#composer-shell", Vertical).add_class("busy")
            self._reset_live_view_state()
            self._interrupt_armed = False
            self._turn_started_at = started_at
            self._turn_completed_at = None
            self._reasoning_cell = self._append_cell(
                f"[dim]{escape(self._text('thinking'))}...[/dim]",
                "event",
            )
            self._sync_run_state_from_active(run_state)
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
                self._append_turn_error(item, display_markup=error_markup(error))
                self._refresh_status(self._text("error"))
        finally:
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
            else:
                keep_state = run_state.retryable_error or run_state.terminal_error
                if keep_state:
                    run_state.worker = None
                    run_state.cancel_event = asyncio.Event()
                    if self._is_active_thread(thread_id):
                        self._sync_run_state_from_active(run_state)
                    run_state.detach_widgets()
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
            "model.stream_retry",
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
        if reasoning_text:
            await self._handle_thread_event(
                item_thread_id,
                "assistant.reasoning_completed",
                {
                    "type": "assistant.reasoning_completed",
                    "thread_id": item_thread_id,
                    "turn_id": item.get("turn_id"),
                    "turn_started_at": item.get("turn_started_at"),
                    "text": reasoning_text,
                },
                run_state,
            )
        else:
            await self._handle_thread_event(
                item_thread_id,
                "assistant.reasoning_absent",
                {
                    "type": "assistant.reasoning_absent",
                    "thread_id": item_thread_id,
                    "turn_id": item.get("turn_id"),
                    "turn_started_at": item.get("turn_started_at"),
                },
                run_state,
            )
        output = list(getattr(response, "output", []) or [])
        has_tool_call = any(entry.get("type") == "function_call" for entry in output)
        if has_tool_call:
            await self._handle_thread_event(
                item_thread_id,
                "assistant.response_with_tools",
                {
                    "type": "assistant.response_with_tools",
                    "thread_id": item_thread_id,
                    "turn_id": item.get("turn_id"),
                    "turn_started_at": item.get("turn_started_at"),
                    "assistant_text": run_state.assistant_buffer,
                },
                run_state,
            )
        else:
            if "assistant_text" not in item:
                item["assistant_text"] = run_state.assistant_buffer
            await self._handle_thread_event(
                item_thread_id,
                "assistant.final_response_started",
                {
                    "type": "assistant.final_response_started",
                    "thread_id": item_thread_id,
                    "turn_id": item.get("turn_id"),
                    "turn_started_at": item.get("turn_started_at"),
                    "assistant_text": item.get("assistant_text"),
                },
                run_state,
            )
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
        text = item["final_text"] or run_state.assistant_buffer
        was_active_thread = self._is_active_thread(item_thread_id)
        run_state.pending_stream_retries.clear()
        if text and was_active_thread and self._assistant_cell is None:
            self._append_assistant_text(text)
        if was_active_thread:
            # Match re-entry behavior: every turn end folds its process cells,
            # not just turns that emitted assistant.final_response_started.
            self._finalize_turn_render()
            self._sync_run_state_from_active(run_state)
        elif run_state.process_cells:
            run_state.process_collapsed = True
        run_state.status = self._text("idle")
        self._notify_turn_completed(item_thread_id, text, active_thread=was_active_thread)
        if was_active_thread:
            self._refresh_status(self._text("idle"))

    def _handle_billing_updated_item(
        self,
        item_thread_id: str,
        item: dict[str, Any],
        run_state: ThreadRunState,
    ) -> None:
        self._record_live_thread_event(run_state, "thread.billing_accumulated", item)
        if self._is_active_thread(item_thread_id):
            self._refresh_status()


    def _handle_turn_interrupted_item(
        self,
        item_thread_id: str,
        item: dict[str, Any],
        run_state: ThreadRunState,
    ) -> None:
        self._update_turn_timestamps(item, run_state)
        self._record_live_thread_event(run_state, "turn.interrupted", item)
        run_state.status = self._text("interrupted")
        run_state.pending_stream_retries.clear()
        if self._is_active_thread(item_thread_id):
            self._apply_thread_event_to_active("turn.interrupted", item)
            self._refresh_status(self._text("interrupted"))

    def _handle_turn_error_item(
        self,
        item_thread_id: str,
        item: dict[str, Any],
        run_state: ThreadRunState,
    ) -> None:
        self._update_turn_timestamps(item, run_state)
        self._mark_run_error_state(run_state, item)
        self._record_live_thread_event(run_state, "turn.error", item)
        if self._is_active_thread(item_thread_id):
            self._flush_pending_stream_retries(run_state)
            self._apply_thread_event_to_active("turn.error", item)
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

    def _active_queue_length(self) -> int:
        run_state = self._active_run_state()
        return len(run_state.queue) if run_state is not None else 0

    def _ensure_active_thread(self) -> str:
        if self.thread_id is None:
            self.thread_id = self.engine.thread_store.create_thread()
        return self.thread_id

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
        before: object | None = None,
    ) -> TranscriptCell:
        message = str(event.get("message") or self._text("model_switch_warning"))
        from_level = str(event.get("from_level") or "")
        to_level = str(event.get("to_level") or "")
        suffix = ""
        if from_level or to_level:
            suffix = f"\n[dim]{escape(from_level or '?')} -> {escape(to_level or '?')}[/dim]"
        return self._append_cell(
            f"[yellow]{escape(message)}[/yellow]{suffix}",
            "event",
            before=before,
            copy_text=message,
        )

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
        self._tool_delta_cells.clear()
        self._tool_delta_calls.clear()
        self._process_cells = []
        self._process_fold_cell = None
        self._process_collapsed = False
        self._process_anchor_cell = None

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

    def _run_state_for_thread(self, thread_id: str | None) -> ThreadRunState:
        if thread_id is None:
            thread_id = self.engine.thread_store.create_thread()
            self.thread_id = thread_id
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
        run_state.assistant_buffer = self._assistant_buffer
        run_state.assistant_cell = self._assistant_cell
        run_state.reasoning_buffer = self._reasoning_buffer
        run_state.reasoning_cell = self._reasoning_cell
        run_state.tool_cells = dict(self._tool_cells)
        run_state.tool_delta_cells = dict(self._tool_delta_cells)
        run_state.tool_delta_calls = dict(self._tool_delta_calls)
        run_state.process_cells = list(self._process_cells)
        run_state.process_fold_cell = self._process_fold_cell
        run_state.process_collapsed = self._process_collapsed
        run_state.process_anchor_cell = self._process_anchor_cell
        run_state.started_at = self._turn_started_at
        run_state.completed_at = self._turn_completed_at

    def _sync_active_from_run_state(self, run_state: ThreadRunState) -> None:
        self._assistant_buffer = run_state.assistant_buffer
        self._assistant_cell = run_state.assistant_cell
        self._reasoning_buffer = run_state.reasoning_buffer
        self._reasoning_cell = run_state.reasoning_cell
        self._tool_cells = dict(run_state.tool_cells)
        self._tool_delta_cells = dict(run_state.tool_delta_cells)
        self._tool_delta_calls = dict(run_state.tool_delta_calls)
        self._process_cells = list(run_state.process_cells)
        self._process_fold_cell = (
            run_state.process_fold_cell
            if isinstance(run_state.process_fold_cell, FoldedProcessCell)
            else None
        )
        self._process_collapsed = run_state.process_collapsed
        self._process_anchor_cell = run_state.process_anchor_cell
        self._turn_started_at = run_state.started_at
        self._turn_completed_at = run_state.completed_at

    def _update_turn_timestamps(self, item: dict[str, Any], run_state: ThreadRunState) -> None:
        turn_id = str(item.get("turn_id") or "").strip()
        if turn_id:
            run_state.turn_id = turn_id
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
        if event_type in {"image.attachment", "turn.completed", "thread.title"}:
            return
        if event_type == "assistant.response_with_tools":
            item.setdefault("assistant_text", run_state.assistant_buffer)
        turn_id = str(item.get("turn_id") or "").strip()
        if turn_id:
            if run_state.turn_id and run_state.turn_id != turn_id:
                run_state.live_events.clear()
            run_state.turn_id = turn_id
        run_state.live_events.append({"event_type": event_type, "item": self._live_event_item(item)})

    def _defer_stream_retry_event(self, run_state: ThreadRunState, item: dict[str, Any]) -> None:
        retry_item = self._live_event_item(item)
        turn_id = str(retry_item.get("turn_id") or "").strip()
        if turn_id and run_state.pending_stream_retries:
            last_turn_id = str(run_state.pending_stream_retries[-1].get("turn_id") or "").strip()
            if last_turn_id and last_turn_id != turn_id:
                run_state.pending_stream_retries.clear()
        run_state.pending_stream_retries.append(retry_item)

    def _flush_pending_stream_retries(self, run_state: ThreadRunState) -> None:
        if not self._is_active_thread(run_state.thread_id):
            return
        retries = list(run_state.pending_stream_retries)
        run_state.pending_stream_retries.clear()
        for retry_event in retries:
            self._append_stream_retry(retry_event)

    def _live_event_item(self, item: dict[str, Any]) -> dict[str, Any]:
        copy = dict(item)
        tool_call = copy.get("tool_call")
        if tool_call is not None:
            copy["tool_call"] = asdict(tool_call) if is_dataclass(tool_call) else dict(tool_call)
        response = copy.get("response")
        if response is not None:
            copy.pop("response", None)
        return copy

    def _apply_thread_event_to_active(self, event_type: str, item: dict[str, Any]) -> None:
        if event_type == "assistant.delta":
            self._refresh_status(self._text("writing_answer"))
            self._append_assistant_text(str(item.get("text") or ""))
        elif event_type == "assistant.reasoning_delta":
            self._refresh_status(self._text("thinking_status"))
            self._append_reasoning_delta(str(item.get("text") or ""))
        elif event_type == "assistant.reasoning_completed":
            self._finalize_reasoning(str(item.get("text") or ""))
        elif event_type == "assistant.reasoning_absent":
            self._clear_pending_reasoning()
        elif event_type == "image.attachment":
            attachment = item.get("attachment") or {}
            self._append_image_attachment_cell(attachment)
        elif event_type == "tool.delta":
            self._append_tool_delta(item)
        elif event_type == "assistant.response_with_tools":
            item.setdefault("assistant_text", self._assistant_buffer)
            self._track_current_assistant_cell_as_process()
            self._seal_assistant_round()
        elif event_type == "assistant.final_response_started":
            # The current assistant cell here is the final user-visible reply
            # for this turn (its deltas have already streamed). Only collapse
            # prior process cells; do NOT classify the final reply as process.
            self._collapse_process_cells()
        elif event_type == "tool.started":
            self._clear_pending_reasoning()
            self._seal_assistant_round()
            self._append_tool_started(item)
        elif event_type == "tool.output":
            self._append_tool_output(item)
        elif event_type == "turn.stream_retry":
            self._append_stream_retry(item)
        elif event_type == "model.stream_retry":
            pass
        elif event_type == "compaction.completed":
            self._append_cell(f"[dim]{escape(self._text('compacted'))}[/dim]", "event")
        elif event_type == "thread.billing_accumulated":
            self._refresh_status()
        elif event_type == "turn.interrupted":
            # Re-entry path renders no "interrupted" marker cell; mirror it
            # by only folding the turn's process cells. See _finalize_turn_render.
            self._finalize_turn_render()
        elif event_type == "turn.error":
            run_state = self._thread_state(str(item.get("thread_id") or self.thread_id or ""))
            if run_state is not None:
                self._mark_run_error_state(run_state, item)
                self._flush_pending_stream_retries(run_state)
            self._finalize_turn_render()
            self._append_turn_error(item)

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
        except (OSError, ValueError):
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
        self._record_live_thread_event(run_state, event_type, item)
        if event_type == "model.stream_retry":
            self._defer_stream_retry_event(run_state, item)
            run_state.status = self._text("working")
            if self._is_active_thread(thread_id):
                self._refresh_status(self._text("working"))
            return
        if not self._is_active_thread(thread_id):
            if event_type == "assistant.delta":
                run_state.assistant_buffer += str(item.get("text") or "")
                run_state.status = self._text("writing_answer")
            elif event_type == "assistant.reasoning_delta":
                delta_text = str(item.get("text") or "")
                if delta_text:
                    run_state.reasoning_buffer = self._append_reasoning_text(run_state.reasoning_buffer, delta_text)
                run_state.status = self._text("thinking_status")
            elif event_type == "assistant.reasoning_completed":
                run_state.reasoning_buffer = ""
                run_state.reasoning_cell = None
            elif event_type == "assistant.reasoning_absent":
                run_state.reasoning_buffer = ""
                run_state.reasoning_cell = None
            elif event_type == "assistant.response_with_tools":
                run_state.assistant_buffer = ""
                run_state.assistant_cell = None
            elif event_type == "tool.delta":
                delta = item.get("tool_call")
                index = int(self._tool_call_field(delta, "index", 0) or 0)
                run_state.tool_delta_calls[index] = {
                    "call_id": self._tool_call_field(delta, "call_id", "") or "",
                    "name": str(self._tool_call_field(delta, "name", None) or "python"),
                    "arguments": self._tool_call_field(delta, "arguments", "")
                    or self._tool_call_field(delta, "arguments_delta", ""),
                }
                run_state.status = self._text("writing_script")
            elif event_type == "tool.started":
                delta_index = item.get("tool_call_index")
                if isinstance(delta_index, int):
                    call = item.get("call") if isinstance(item.get("call"), dict) else {}
                    run_state.tool_delta_calls[delta_index] = dict(call)
                run_state.reasoning_buffer = ""
                run_state.reasoning_cell = None
                run_state.status = self._text("running_python")
            elif event_type == "tool.output":
                delta_index = item.get("tool_call_index")
                if isinstance(delta_index, int):
                    run_state.tool_delta_calls.pop(delta_index, None)
                run_state.status = self._text("working")
            elif event_type == "assistant.final_response_started" and run_state.process_cells:
                run_state.process_collapsed = True
            elif event_type == "compaction.completed":
                run_state.status = self._text("working")
            elif event_type == "turn.interrupted":
                run_state.status = self._text("interrupted")
                if run_state.process_cells:
                    run_state.process_collapsed = True
            elif event_type == "turn.error":
                run_state.status = self._text("error")
            return
        self._sync_active_from_run_state(run_state)
        self._apply_thread_event_to_active(event_type, item)
        self._sync_run_state_from_active(run_state)

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

    def _queued_turn_markup(self, prompt: str, image_paths: list[Path]) -> str:
        suffix = ""
        if image_paths:
            suffix = "\n" + f"[dim]+{len(image_paths)} {escape(self._text('images'))}[/dim]"
        return f"[dim]{escape(self._text('queued'))}[/dim]\n{escape(prompt)}{suffix}"

    def _handle_command(self, prompt: str) -> bool:
        command, _, rest = prompt.partition(" ")
        if command == "/clear":
            active_run = self._active_run_state()
            if active_run is not None:
                self._sync_run_state_from_active(active_run)
                active_run.detach_widgets()
            self._close_active_panel()
            self.thread_id = None
            self.level = None
            self._reset_live_view_state()
            self._pending_images.clear()
            self._reset_transcript()
            self._refresh_pending_images()
            self._refresh_active_run_state()
            self._refresh_status(self._text("idle"))
            return True
        if command == "/quit":
            self._quit_from_command()
            return True
        if command == "/threads":
            self._open_threads_panel()
            return True
        if command == "/status":
            self._open_status_panel()
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
            f"[bold]{escape(self._text('keyboard_shortcuts'))}[/bold]",
            f"- [cyan]Ctrl+Enter / Ctrl+J[/cyan] [dim]{escape(self._text('help_send'))}[/dim]",
            f"- [cyan]Enter[/cyan] [dim]{escape(self._text('help_newline'))}[/dim]",
            f"- [cyan]Ctrl+O / /[/cyan] [dim]{escape(self._text('help_commands'))}[/dim]",
            f"- [cyan]F1 / ?[/cyan] [dim]{escape(self._text('help_help'))}[/dim]",
            f"- [cyan]Ctrl+S[/cyan] [dim]{escape(self._text('help_status'))}[/dim]",
            f"- [cyan]Ctrl+D[/cyan] [dim]{escape(self._text('help_details'))}[/dim]",
            f"- [cyan]F2[/cyan] [dim]{escape(self._text('help_attach_image'))}[/dim]",
            f"- [cyan]F3[/cyan] [dim]{escape(self._text('help_preview_images'))}[/dim]",
            f"- [cyan]Tab[/cyan] [dim]{escape(self._text('help_height'))}[/dim]",
            f"- [cyan]Ctrl+C[/cyan] [dim]{escape(self._text('help_interrupt_quit'))}[/dim]",
            "",
            f"[bold]{escape(self._text('mentions'))}[/bold]",
            f"- [cyan]@[/cyan] [dim]{escape(self._text('help_mention_files'))}[/dim]",
            f"- [cyan]@@[/cyan] [dim]{escape(self._text('help_mention_threads'))}[/dim]",
            "",
            f"[bold]{escape(self._text('commands'))}[/bold] [dim](Tab/Enter, Esc)[/dim]",
        ]
        for spec in self._commands():
            lines.append(
                f"[cyan]{escape(spec.usage):<18}[/cyan] [dim]{escape(spec.description)}[/dim]"
            )
        self._open_panel("\n".join(lines), "help", self._text("help"))

    def _append_help(self) -> None:
        lines = [f"[bold]{escape(self._text('commands'))}[/bold] [dim](Ctrl+O, F1, Esc)[/dim]"]
        for spec in self._commands():
            lines.append(
                f"[cyan]{escape(spec.usage):<18}[/cyan] [dim]{escape(spec.description)}[/dim]"
            )
        self._append_cell("\n".join(lines), "event")

    def _append_user(self, text: str, *, before: object | None = None) -> TranscriptCell:
        label = "你" if self.language.is_chinese else "you"
        # Codex-style "› " prefix keeps user turns easy to spot.
        return self._append_cell(
            f"[bold #7dd3fc]› {label}[/bold #7dd3fc]\n{escape(text)}",
            "user",
            before=before,
        )

    def _append_assistant_text(self, text: str) -> None:
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
        self._reasoning_buffer = self._append_reasoning_text(self._reasoning_buffer, text)
        display_text = self._reasoning_buffer.strip()
        if not display_text:
            return
        first = display_text.splitlines()[0]
        if len(first) > 120:
            first = first[:117].rstrip() + "..."
        markup = (
            f"[dim italic]{escape(self._text('thinking'))}[/dim italic]  "
            f"[italic #a3b1c2]{escape(first)}[/italic #a3b1c2]"
        )
        if self._reasoning_cell is None:
            self._reasoning_cell = self._append_cell(markup, "reasoning")
        else:
            self._reasoning_cell.update(markup)
            self._scroll_end()

    def _append_reasoning_text(self, existing: str, delta: str) -> str:
        if not existing:
            return delta.lstrip()
        return existing + delta

    def _reasoning_markup(self, text: str) -> tuple[str, str]:
        stripped = text.strip()
        first = stripped.splitlines()[0]
        if len(first) > 120:
            first = first[:117].rstrip() + "..."
        summary = (
            f"[dim italic]{escape(self._text('thinking'))}[/dim italic]  "
            f"[italic #a3b1c2]{escape(first)}[/italic #a3b1c2]"
        )
        return summary, escape(stripped)

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
        before: object | None = None,
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
            run_state = self._active_run_state()
            if run_state is not None:
                run_state.process_collapsed = collapsed

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
        before: object | None = None,
        after: TranscriptCell | None = None,
        elapsed_label: str | None = None,
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
            markup=True,
        )
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

    def _cell_after(self, cell: TranscriptCell) -> object | None:
        try:
            children = list(self.query_one("#transcript", VerticalScroll).children)
            index = children.index(cell)
        except (NoMatches, ValueError):
            return None
        return children[index + 1] if index + 1 < len(children) else None

    def _append_tool_output(self, item: dict[str, Any]) -> None:
        payload = parse_tool_payload(item.get("output", {}))
        call = item.get("call") if isinstance(item.get("call"), dict) else None
        delta_index = item.get("tool_call_index")
        if (call is None or not tool_call_preview_line(call)) and isinstance(delta_index, int):
            call = self._tool_delta_calls.pop(delta_index, None) or call
        call_id = str((call or {}).get("call_id") or item.get("call", {}).get("call_id") or "")
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
        if payload is None:
            markup = f"[dim]{escape(self._text('python'))} {escape(self._text('python_completed'))}[/dim]"
            cell = self._append_cell(markup, "event")
            self._track_process_cell(cell)
            return

        self._last_tool_payload = payload
        markup = tool_timeline_markup(payload)
        details = tool_detail_markup(payload)
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
        call = item.get("call") or {}
        delta_index = item.get("tool_call_index")
        if isinstance(delta_index, int):
            self._tool_delta_calls[delta_index] = dict(call)
        call_id = str(call.get("call_id") or "")
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
    ) -> str:
        if call is not None and tool_call_preview_line(call):
            return tool_call_summary_markup(
                {**call, "_status_label": call.get("_status_label") or self._text("python_running")}
            )
        status = str((call or {}).get("_status_label") or self._text("python_running"))
        return (
            f"[#7dd3fc]⠿[/#7dd3fc] [bold]{escape(name)}[/bold] "
            f"[dim]{escape(status)}[/dim]{detail}"
        )

    def _append_image_attachment_cell(
        self,
        attachment: dict[str, Any],
        *,
        before: object | None = None,
    ) -> ImageAttachmentCell:
        self._mark_transcript_content()
        cell = ImageAttachmentCell(attachment, classes="event", markup=True)
        self.query_one("#transcript", VerticalScroll).mount(cell, before=before)
        self._scroll_end()
        return cell

    def _prepend_history_cells(
        self,
        events: list[dict[str, Any]],
        *,
        has_more: bool,
        start_offset: int | None = None,
    ) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        insert_before: object | None = transcript.children[0] if transcript.children else None
        if self._history_more_cell is not None:
            children = list(transcript.children)
            try:
                marker_index = children.index(self._history_more_cell)
            except ValueError:
                marker_index = -1
            insert_before = children[marker_index + 1] if marker_index >= 0 and marker_index + 1 < len(children) else None
            self._history_more_cell.remove()
            self._history_more_cell = None
        self._history_has_more = has_more
        if start_offset is not None:
            self._history_before_offset = start_offset
        elif events:
            self._history_before_offset = _event_offset(events[0])
        if has_more or events:
            marker = LoadOlderHistoryCell(has_more=has_more, classes="event", markup=True)
            transcript.mount(marker, before=insert_before)
            self._history_more_cell = marker
        self._mount_history_events(events, before=insert_before)
        self._mark_transcript_content()

    def _mount_history_events(
        self,
        events: list[dict[str, Any]],
        *,
        before: object | None = None,
    ) -> None:
        index = 0
        while index < len(events):
            event = events[index]
            turn_id = str(event.get("turn_id") or "")
            if not turn_id:
                self._mount_history_event(event, before=before)
                index += 1
                continue

            turn_events: list[dict[str, Any]] = []
            while index < len(events):
                next_event = events[index]
                if str(next_event.get("turn_id") or "") == turn_id:
                    turn_events.append(next_event)
                    index += 1
                    continue
                break
            self._mount_history_turn_events(turn_events, before=before)

    def _mount_history_turn_events(
        self,
        events: list[dict[str, Any]],
        *,
        before: object | None = None,
    ) -> None:
        process_cells: list[TranscriptCell] = []
        anchor_cell: TranscriptCell | None = None
        seen_user = False
        show_stream_retries = self._history_turn_ended_with_error(events)
        for event in events:
            if event.get("type") == "item.user":
                if seen_user:
                    continue
                seen_user = True
            if event.get("type") == "turn.stream_retry" and not show_stream_retries:
                continue
            for cell in self._mount_history_event(event, before=before) or []:
                if event.get("type") == "item.user":
                    anchor_cell = cell
                if self._history_cell_is_process(event, cell):
                    process_cells.append(cell)
        if process_cells:
            self._append_process_fold_cell(
                process_cells,
                collapsed=True,
                before=before,
                after=anchor_cell,
                elapsed_label=self._history_turn_elapsed_label(events),
            )

    def _history_turn_ended_with_error(self, events: list[dict[str, Any]]) -> bool:
        for event in reversed(events):
            if event.get("type") in {"turn.completed", "turn.interrupted", "turn.error"}:
                return event.get("type") == "turn.error"
        return False

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

    def _history_cell_is_process(self, event: dict[str, Any], cell: TranscriptCell) -> bool:
        event_type = event.get("type")
        if event_type in {"item.runner_result", "item.reasoning_delta", "item.reasoning_partial"}:
            return True
        if event_type != "item.model_response":
            return False
        has_tool_call = any(
            isinstance(item, dict) and item.get("type") == "function_call"
            for item in event.get("output") or []
        )
        if has_tool_call:
            return True
        if cell.has_class("assistant"):
            return False
        if str(event.get("reasoning_text") or "").strip():
            return True
        return False

    def _mount_history_event(
        self,
        event: dict[str, Any],
        *,
        before: object | None = None,
    ) -> list[TranscriptCell]:
        mounted: list[TranscriptCell] = []
        event_type = event.get("type")
        if event_type == "item.user":
            cell = self._append_user_from_history(event.get("item") or {}, before=before)
            if cell is not None:
                mounted.append(cell)
        elif event_type == "item.model_response":
            reasoning_cell = self._append_reasoning_history(str(event.get("reasoning_text") or ""), before=before)
            if reasoning_cell is not None:
                mounted.append(reasoning_cell)
            for item in event.get("output") or []:
                item_type = item.get("type")
                if item_type == "message":
                    text = self._message_item_text(item)
                    if text:
                        mounted.append(self._append_cell(_markdown(text), "assistant", before=before, copy_text=text))
                elif item_type == "function_call":
                    mounted.append(self._append_tool_call_history(item, before=before))
        elif event_type == "item.runner_result":
            result = event.get("result") or {}
            self._last_tool_payload = result
            replayed_cell = self._append_expandable_cell(
                tool_timeline_markup(result),
                tool_detail_markup(result),
                "event",
                before=before,
            )
            replayed_cell.tool_payload = result
            mounted.append(replayed_cell)
        elif event_type == "item.image_attachment":
            attachment = event.get("attachment") or {}
            mounted.append(self._append_image_attachment_cell(attachment, before=before))
        elif event_type in {"item.reasoning_delta", "item.reasoning_partial"}:
            cell = self._append_reasoning_history(str(event.get("text") or ""), before=before)
            if cell is not None:
                mounted.append(cell)
        elif event_type == "turn.error":
            run_state = self._run_state_for_history_error(event)
            if run_state is not None:
                self._mark_run_error_state(run_state, event)
            mounted.append(self._append_turn_error(event, before=before))
        elif event_type == "turn.stream_retry":
            mounted.append(self._append_stream_retry(event, before=before))
        elif event_type == "item.compaction":
            mounted.append(self._append_cell(f"[dim]{escape(self._text('compacted'))}[/dim]", "event", before=before))
        elif event_type == "thread.model_switch_warning":
            mounted.append(self._append_model_switch_warning_cell(event, before=before))
        return mounted

    def _run_state_for_history_error(self, event: dict[str, Any]) -> ThreadRunState | None:
        thread_id = str(event.get("thread_id") or self.thread_id or "")
        if not thread_id:
            return None
        run_state = self._thread_runs.get(thread_id)
        if run_state is None:
            run_state = ThreadRunState(
                thread_id=thread_id,
                worker=None,
                cancel_event=asyncio.Event(),
                queue=[],
                status=self._text("error"),
            )
            self._thread_runs[thread_id] = run_state
        return run_state

    def _append_turn_error(
        self,
        event: dict[str, Any],
        *,
        before: object | None = None,
        display_markup: str | None = None,
    ) -> TranscriptCell:
        error_type = str(event.get("error_type") or "Turn error")
        message = str(event.get("message") or "The turn stopped before producing a final response.")
        retryable = self._is_retryable_error_event(event)
        hint = self._text("retry_network_error_hint") if retryable else self._text("thread_stopped_after_error")
        content = display_markup or f"[bold red]{escape_error_markup(error_type)}[/bold red] {escape_error_markup(message)}"
        cell = self._append_cell(
            f"{content}\n[dim]{escape_error_markup(hint)}[/dim]",
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
        before: object | None = None,
    ) -> TranscriptCell:
        attempt = event.get("attempt")
        max_attempts = event.get("max_attempts")
        delay_s = event.get("delay_s")
        error_type = str(event.get("error_type") or "stream")
        try:
            delay_text = f"{float(delay_s):.1f}s"
        except (TypeError, ValueError):
            delay_text = "?s"
        markup = (
            "[dim]⟳ stream empty, retrying "
            f"{escape(str(attempt or '?'))}/{escape(str(max_attempts or '?'))} "
            f"in {escape(delay_text)} ({escape(error_type)})[/dim]"
        )
        return self._append_cell(markup, "event", before=before)

    def _append_retry_button(self, event: dict[str, Any], *, before: object | None = None) -> RetryTurnButton:
        self._mark_transcript_content()
        button = RetryTurnButton(self._text("retry"), thread_id=str(event.get("thread_id") or self.thread_id or ""))
        self.query_one("#transcript", VerticalScroll).mount(button, before=before)
        self._scroll_end()
        return button

    def _append_tool_call_history(self, item: dict[str, Any], *, before: object | None = None) -> ExpandableTranscriptCell:
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
        return f"\n[dim]{escape(first)}[/dim]"

    def _append_cell(
        self,
        content: object,
        classes: str,
        *,
        before: object | None = None,
        copy_text: str | None = None,
    ) -> TranscriptCell:
        self._mark_transcript_content()
        cell = TranscriptCell(content, classes=classes, markup=True, copy_text=copy_text)
        self.query_one("#transcript", VerticalScroll).mount(cell, before=before)
        self._scroll_end()
        return cell

    def _append_expandable_cell(
        self,
        summary: str,
        details: str,
        classes: str,
        *,
        before: object | None = None,
    ) -> ExpandableTranscriptCell:
        self._mark_transcript_content()
        cell = ExpandableTranscriptCell(summary, details, classes=classes, markup=True)
        self.query_one("#transcript", VerticalScroll).mount(cell, before=before)
        self._scroll_end()
        return cell

    def _append_reasoning_cell(
        self,
        summary: str,
        details: str,
        *,
        before: object | None = None,
    ) -> ExpandableTranscriptCell:
        self._mark_transcript_content()
        cell = ExpandableTranscriptCell(
            summary,
            details,
            detail_title="reasoning_details",
            detail_hint="reasoning_details_hint",
            classes="reasoning",
            markup=True,
        )
        self.query_one("#transcript", VerticalScroll).mount(cell, before=before)
        self._scroll_end()
        return cell

    def _replace_with_expandable_cell(
        self,
        old_cell: TranscriptCell,
        summary: str,
        details: str,
        classes: str,
    ) -> ExpandableTranscriptCell:
        cell = ExpandableTranscriptCell(summary, details, classes=classes, markup=True)
        self.query_one("#transcript", VerticalScroll).mount(cell, before=old_cell)
        old_cell.remove()
        return cell

    def _replace_with_reasoning_cell(
        self,
        old_cell: TranscriptCell,
        summary: str,
        details: str,
    ) -> ExpandableTranscriptCell:
        cell = ExpandableTranscriptCell(
            summary,
            details,
            detail_title="reasoning_details",
            detail_hint="reasoning_details_hint",
            classes="reasoning",
            markup=True,
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
        self._history_before_offset = None
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
                f"[dim]{escape(self._text('no_threads'))}[/dim]",
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

    def _status_panel_markup(self) -> str:
        self.engine.refresh_config()
        level_name = self.level or self.engine.config.runtime.default_level
        rules = self.engine.project_rule_context()
        billing_line = self._thread_billing_status_line()
        try:
            model = self.engine.config.model_for_level(self.level)
            provider = self.engine.config.provider_for_model(model)
            stats = self.engine.context_stats(self.thread_id, self.level)
            model_line = f"{escape(model.name)} -> {escape(model.model)}"
            provider_line = f"{escape(provider.name)} / {escape(model.api)}"
            context_line = (
                f"{stats.percent}% "
                f"({format_tokens(stats.used_tokens)} / {format_tokens(stats.context_window_tokens)}, "
                f"{escape(stats.source)})"
            )
            compress_line = (
                f"{'on' if self.engine.config.runtime.compression.enabled else 'off'} · "
                f"trigger {format_tokens(stats.threshold_tokens)} · "
                f"headroom {format_tokens(stats.headroom_tokens)}"
            )
        except ConfigError as exc:
            model_line = "[red]not configured[/red]"
            provider_line = escape(str(exc))
            context_line = "-"
            compress_line = "-"
        rules_line = f"{len(rules.rules)} {self._text('status_rules_loaded')}"
        if rules.truncated:
            rules_line += f" · {self._text('truncated')}"
        if rules.omitted_files:
            rules_line += f" · {rules.omitted_files} {self._text('status_rules_omitted')}"
        lines = [
            f"- state: [cyan]{escape(self._last_status)}[/cyan]",
            f"- version: [cyan]{escape(application_version())}[/cyan]",
            f"- level: [cyan]{escape(level_name)}[/cyan]",
            f"- model: {model_line}",
            f"- provider/api: {provider_line}",
            f"- context: {context_line}",
        ]
        if billing_line:
            lines.append(f"- billing: {billing_line}")
        lines.extend(
            [
                f"- compaction: {compress_line}",
                f"- rules: {escape(rules_line)}",
                f"- thread: {escape(short_thread(self.thread_id))}",
                f"- queued: {self._active_queue_length()}",
                f"- user state: {escape(str(uv_agent_home()))}",
                f"- project state: {escape(str(project_state_dir(self.project_root)))}",
                f"- host: {escape(host_environment_line())}",
                f"- language: {escape(self.language.name)}",
            ]
        )
        background_runs = self._background_run_states()
        if background_runs:
            lines.append(
                f"- background: [cyan]{len(background_runs)} {escape(self._text('active_threads'))}[/cyan]"
            )
            for run_state in background_runs[:6]:
                queue = f" · q{len(run_state.queue)}" if run_state.queue else ""
                lines.append(
                    f"  - {escape(short_thread(run_state.thread_id))}: "
                    f"{escape(run_state.status)}{queue}"
                )
            if len(background_runs) > 6:
                lines.append(f"  - ... {len(background_runs) - 6} more")
        if rules.rules:
            lines.append("")
            lines.append(f"[bold]{escape(self._text('rules'))}[/bold]")
            for rule in rules.rules[:6]:
                suffix = f" [{escape(self._text('truncated'))}]" if rule.truncated else ""
                lines.append(f"- {escape(rule.scope)}: {escape(str(rule.path))}{suffix}")
            if len(rules.rules) > 6:
                lines.append(f"- ... {len(rules.rules) - 6} more")
        return "\n".join(lines)

    def _thread_billing_status_line(self) -> str:
        if not self.engine.config.pricing.models:
            return ""
        label = self._thread_billing_label(decimals=6)
        return escape(label) if label else "-"

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

    def _resume_thread(self, thread_id: str) -> None:
        if not thread_id:
            return
        active_run = self._active_run_state()
        if active_run is not None:
            self._sync_run_state_from_active(active_run)
            active_run.detach_widgets()
        run_state = self._thread_runs.get(thread_id)
        if run_state is not None and run_state is not active_run:
            run_state.detach_widgets()
        self.thread_id = thread_id
        self.level = self._thread_metadata_level(thread_id)
        self._reset_live_view_state()
        self._reset_transcript(show_empty=False)
        self._render_thread_history(thread_id)
        if run_state is not None:
            if run_state.worker is not None:
                self._replay_running_live_events(run_state)
                if run_state.status != self._text("idle"):
                    self._append_cell(f"[dim]{escape(run_state.status)}...[/dim]", "event")
            elif run_state.retryable_error or run_state.terminal_error:
                self._turn_started_at = run_state.started_at
                self._turn_completed_at = run_state.completed_at
        if not self._transcript_has_content:
            self._reset_transcript()
        self._refresh_active_run_state()
        self._refresh_status(run_state.status if run_state is not None and run_state.worker is None else self._text("resumed"))
        self._close_active_panel()

    def _render_thread_history(self, thread_id: str) -> None:
        segment = self.engine.thread_store.read_history_segment(
            thread_id,
            event_types=VISIBLE_HISTORY_EVENT_TYPES | {"turn.started", "turn.completed"},
        )
        self._history_has_more = segment.has_more
        self._history_before_offset = segment.start_offset
        if segment.has_more:
            transcript = self.query_one("#transcript", VerticalScroll)
            marker = LoadOlderHistoryCell(has_more=True, classes="event", markup=True)
            transcript.mount(marker)
            self._history_more_cell = marker
        self._mount_history_events(segment.events)

    def _replay_running_live_events(self, run_state: ThreadRunState) -> None:
        was_collapsed = run_state.process_collapsed
        self._reset_live_render_state()
        self._turn_started_at = run_state.started_at
        self._turn_completed_at = run_state.completed_at
        if not run_state.live_events:
            self._replay_running_buffers(run_state)
            self._sync_run_state_from_active(run_state)
            return

        events = self._events_after_history(run_state.live_events)
        for live_event in events:
            self._apply_thread_event_to_active(
                str(live_event.get("event_type") or ""),
                live_event.get("item") if isinstance(live_event.get("item"), dict) else {},
            )
        if was_collapsed and self._process_cells and self._process_fold_cell is None:
            self._collapse_process_cells()
        self._sync_run_state_from_active(run_state)

    def _replay_running_buffers(self, run_state: ThreadRunState) -> None:
        was_collapsed = run_state.process_collapsed
        if run_state.reasoning_buffer:
            self._append_reasoning_delta(run_state.reasoning_buffer)
        if run_state.assistant_buffer:
            self._append_assistant_text(run_state.assistant_buffer)
        for index, call in sorted(run_state.tool_delta_calls.items()):
            self._tool_delta_calls[index] = dict(call)
            self._append_tool_pending_call(index, call)
        if was_collapsed and self._process_cells and self._process_fold_cell is None:
            self._collapse_process_cells()

    def _events_after_history(self, live_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        transcript = self.query_one("#transcript", VerticalScroll)
        existing_texts = [
            self._cell_text(child)
            for child in transcript.children
            if isinstance(child, TranscriptCell)
        ]
        start = 0
        for index, live_event in enumerate(live_events):
            event_type = str(live_event.get("event_type") or "")
            item = live_event.get("item") if isinstance(live_event.get("item"), dict) else {}
            if event_type == "tool.output":
                output = item.get("output") if isinstance(item, dict) else {}
                payload = parse_tool_payload(output if isinstance(output, dict) else {})
                run_id = str((payload or {}).get("run_id") or "")
                if run_id and any(run_id in text for text in existing_texts):
                    start = index + 1
            elif event_type == "assistant.response_with_tools":
                text = str(item.get("assistant_text") or "")
                if text and text in existing_texts:
                    start = index + 1
            elif event_type == "assistant.final_response_started":
                text = str(item.get("assistant_text") or "")
                if text and text in existing_texts:
                    start = index + 1
        return live_events[start:]

    def _cell_text(self, cell: TranscriptCell) -> str:
        parts = [str(cell.copy_text or "")]
        if isinstance(cell, ExpandableTranscriptCell):
            parts.append(cell.details)
        try:
            parts.append(str(cell.render()))
        except Exception:
            pass
        return "\n".join(parts)

    def _load_older_thread_history(self) -> None:
        if not self.thread_id or self._history_before_offset is None:
            return
        segment = self.engine.thread_store.read_history_segment(
            self.thread_id,
            before_offset=self._history_before_offset,
            event_types=VISIBLE_HISTORY_EVENT_TYPES,
        )
        self._prepend_history_cells(
            segment.events,
            has_more=segment.has_more,
            start_offset=segment.start_offset,
        )

    def _append_user_from_history(self, item: dict[str, Any], *, before: object | None = None) -> TranscriptCell | None:
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

    def _open_fullscreen_panel(self, title: str, markup: str, *, subtitle: str = "") -> None:
        self.push_screen(FullscreenPanel(title=title, body=markup, subtitle=subtitle))

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

    def _open_panel(self, markup: str, name: str | None = None, title: str | None = None) -> None:
        panel_title = title or (name.title() if name else self._text("panel"))
        panel = self._active_fullscreen_panel()
        if panel is not None and panel.can_navigate:
            panel.navigate_panel(title=panel_title, body=markup, subtitle=self._text("panel_closes"))
            self._refresh_status()
            return
        self._open_fullscreen_panel(panel_title, markup, subtitle=self._text("panel_closes"))
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

    def _flash(self, message: str, *, severity: str = "information") -> None:
        self.notify(message, severity=severity, timeout=2.0)
        self._last_status = message
        self._refresh_status()

    def _notify_turn_completed(self, thread_id: str, final_text: str, *, active_thread: bool) -> None:
        config = self.engine.config.ui.completion_notification
        if not config.enabled:
            return
        if config.terminal and not active_thread:
            self._append_turn_completion_event(thread_id)
        if config.bell:
            self.bell()
            play_completion_sound()

    def _append_turn_completion_event(self, thread_id: str) -> None:
        digest = self.engine.thread_store.thread_digest(thread_id)
        title = str(digest.get("title") or self._text("new_thread")).strip()
        if len(title) > 48:
            title = title[:45].rstrip() + "..."
        markup = (
            f"[dim]{escape(self._text('background_thread_completed'))}[/dim] "
            f"[cyan]{escape(title or self._text('new_thread'))}[/cyan] "
            f"[dim]{escape(short_thread(thread_id))}[/dim]"
        )
        self._append_cell(markup, "event")

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
        state_text = self._last_status
        if self.busy and state_text == self._text("idle"):
            state_text = self._text("working")
        queue_length = self._active_queue_length()
        queued = f" · q{queue_length}" if queue_length else ""
        spinner = ""
        elapsed_suffix = ""
        if self.busy:
            spinner = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)] + " "
            elapsed_seconds = _elapsed_between(self._turn_started_at)
            if elapsed_seconds is None and self._busy_started_at is not None:
                elapsed_seconds = monotonic() - self._busy_started_at
            if elapsed_seconds is not None:
                elapsed = format_elapsed(elapsed_seconds)
                elapsed_suffix = f" [dim]({escape(elapsed)})[/dim]"

        if self.busy:
            footer = (
                f"[cyan]{spinner}{escape(state_text)}[/cyan]{elapsed_suffix} "
                f"[dim]{escape(level_name)} · {escape(compact_context)} · "
                f"{escape(short_thread(self.thread_id))}{queued}[/dim]"
            )
        else:
            footer = (
                f"[dim]{escape(level_name)} · {escape(compact_context)} · "
                f"{escape(short_thread(self.thread_id))}{queued}[/dim]"
            )
        if billing_label:
            footer += f" [dim]·[/dim] [dim]{escape(billing_label)}[/dim]"
        background_count = len(self._background_run_states())
        if background_count:
            footer += (
                f" [dim]·[/dim] [cyan]{background_count} "
                f"{escape(self._text('background_active'))}[/cyan]"
            )
        self._refresh_window_title()
        try:
            self.query_one("#composer-footer", Static).update(footer)
        except NoMatches:
            return
        self._refresh_pending_images()

    def _thread_billing_label(self, *, decimals: int) -> str:
        """Return the current thread's formatted cost, or empty when disabled."""

        if self.thread_id is None or not self.engine.config.pricing.models:
            return ""
        run_state = self._thread_runs.get(self.thread_id)
        if run_state is not None:
            for event in reversed(run_state.live_events):
                if event.get("event_type") != "thread.billing_accumulated":
                    continue
                item = event.get("item") if isinstance(event.get("item"), dict) else {}
                amount = item.get("total")
                currency = item.get("total_currency") or item.get("currency")
                if amount is not None and currency:
                    return format_billing_total(amount, str(currency), decimals=decimals)
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
