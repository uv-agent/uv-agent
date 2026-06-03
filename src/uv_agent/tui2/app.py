from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import exp
from pathlib import Path
from time import monotonic
from typing import Any

from uv_agent.atomic import atomic_replace
from uv_agent.environment import detect_user_language
from uv_agent.helper_calls import extract_runtime_helper_calls
from uv_agent.ids import new_id
from uv_agent.i18n import tr
from uv_agent.mcp_config import discover_mcp_servers
from uv_agent.notifications import play_terminal_buzzer
from uv_agent.paths import project_tui_clipboard_dir, uv_agent_home
from uv_agent.session import ThreadLockedError
from uv_agent.session.store import VISIBLE_HISTORY_EVENT_TYPES
from uv_agent.skills import discover_skills
from uv_agent.thread_titles import DEFAULT_THREAD_TITLES
from uv_agent.tui.formatting import format_elapsed, short_block, short_thread
from uv_agent.billing import billing_total_from_metadata, format_billing_total
from uv_agent.tui.timeline import ThreadTimelineState, TimelineItem
from uv_agent.tui.window_title import sanitized_window_title, write_window_title
from uv_agent.tui2.events import (
    AGENT_VIEW_STATUS_ORDER,
    AgentViewRow,
    CommandSuggestion,
    PendingTurn,
    TranscriptCell,
    Tui2State,
    tool_payload_from_event,
)
from uv_agent.tui2.renderer import Renderer
from uv_agent.tui2.streaming import (
    BREATH_CHARS_PER_PHASE,
    DEFAULT_STREAM_CHARS_PER_SECOND,
    StreamRateEstimator,
    ThreadTokenRatio,
    model_response_visible_units,
    tool_call_name,
    tool_call_stream_key,
    tool_delta_visible_text,
    usage_output_tokens,
)
from uv_agent.tui2.terminal import PASTE_PREFIX, Terminal, TerminalKeyReader
from uv_agent.time import utc_now_iso
from uv_agent.worktree import (
    CommandResult,
    WorktreeError,
    cleanup_worktree,
    create_worktree,
    validate_worktree_branch_name,
)

CODE_FILE_SUFFIXES = {
    ".cfg",
    ".css",
    ".csv",
    ".env",
    ".gd",
    ".go",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".lock",
    ".md",
    ".mjs",
    ".py",
    ".rs",
    ".scss",
    ".toml",
    ".tsx",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
IGNORED_MENTION_DIRS = {
    ".code-search",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-agent",
    ".venv",
    "__pycache__",
    "__pypackages__",
    "build",
    "coverage",
    "dist",
    "env",
    "htmlcov",
    "node_modules",
    "out",
    "site-packages",
    "target",
    "tmp",
    "vendor",
    "venv",
}
MAX_MENTION_ITEMS = 300


def create_engine(project_root: Path | None = None, *, data_dir: Path | None = None):
    """Create the shared uv-agent engine lazily for the raw ANSI TUI."""

    from uv_agent.app_factory import create_engine as _create_engine

    return _create_engine(project_root, data_dir=data_dir)


def save_clipboard_image(target_dir: Path):
    """Save a clipboard image while keeping tui2 startup image-library free."""

    from uv_agent.clipboard import save_clipboard_image as _save_clipboard_image

    return _save_clipboard_image(target_dir)


HELP_TEXT = (
    "Commands:\n"
    "  /help              show this help\n"
    "  /agents            open Agent View dashboard (normal/input/help modes)\n"
    "  /bg                add the current thread to Agent View\n"
    "  /clear             clear view and start a new thread\n"
    "  /threads           choose a thread to resume\n"
    "  /skills            list skills and insert @skill mentions\n"
    "  /mcp               list MCP servers and insert @mcp mentions\n"
    "  /image             attach clipboard image as [Image #N]\n"
    "  /level <name>      switch model level\n"
    "  /title <text>      rename the current thread\n"
    "  /goal <op>         enable | disable | reset | status\n"
    "  /cancel            interrupt the running turn\n"
    "  /quit              exit the TUI\n"
    "\n"
    "Keys: Enter send/select · Ctrl+Enter newline · / command palette · @ mentions · ↑/↓ history\n"
    "      Ctrl+A Agent View · Ctrl+E line end · Ctrl+K cut line · Ctrl+W del word · Ctrl+U clear · Ctrl+C quit/interrupt\n"
    "Agent View: normal mode uses j/k/Enter/Space/c/d/D; i starts a task, m chooses its model, r replies, ? opens help."
)


# Values ending in a space accept more input; the command palette keeps them in
# the composer instead of submitting immediately.
TOP_LEVEL_COMMANDS: tuple[CommandSuggestion, ...] = (
    CommandSuggestion("/help", "show help"),
    CommandSuggestion("/agents", "open Agent View dashboard"),
    CommandSuggestion("/bg", "add current thread to Agent View"),
    CommandSuggestion("/clear", "clear view and start a new thread"),
    CommandSuggestion("/threads", "choose a thread to resume"),
    CommandSuggestion("/status", "show model/context/thread status"),
    CommandSuggestion("/skills", "list skills and insert @skill mentions"),
    CommandSuggestion("/mcp", "list MCP servers and insert @mcp mentions"),
    CommandSuggestion("/image", "attach clipboard image as [Image #N]"),
    CommandSuggestion("/level ", "switch model level"),
    CommandSuggestion("/model ", "alias for /level"),
    CommandSuggestion("/title ", "rename current thread"),
    CommandSuggestion("/goal ", "goal-mode subcommands"),
    CommandSuggestion("/cancel", "interrupt the running turn"),
    CommandSuggestion("/quit", "exit the TUI"),
)

GOAL_COMMANDS: tuple[CommandSuggestion, ...] = (
    CommandSuggestion("/goal enable", "enable goal mode with optional objective"),
    CommandSuggestion("/goal disable", "disable goal mode"),
    CommandSuggestion("/goal reset", "reset goal files with optional objective"),
    CommandSuggestion("/goal status", "show goal state"),
)

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
IMAGE_TOKEN_RE = re.compile(r"\[Image #(\d+)\]")
IMAGE_ONLY_TOKEN_RE = re.compile(r"(?:\s*\[Image #\d+\]\s*)+")
MAX_COMPOSER_HISTORY = 50
COMPOSER_HISTORY_FILENAME = "composer_history.json"
CTRL_C_CONFIRMATION_S = 3.0
UNBRACKETED_PASTE_ENTER_S = 0.08
COMPACTION_SUMMARY_PREVIEW_LINES = 4
COMPACTION_SUMMARY_PREVIEW_CHARS = 800
TOKEN_RATE_DISPLAY_UPDATE_INTERVAL_S = 0.5
TOKEN_RATE_DISPLAY_TAU_S = 3.0
TOKEN_RATE_DISPLAY_HIDE_BELOW = 0.05
_AGENT_VIEW_STATUS_RANK = {status: index for index, status in enumerate(AGENT_VIEW_STATUS_ORDER)}
_RUN_TERMINAL_STATUSES = {"completed", "failed", "interrupted"}
TUI2_FLUSHED_CELLS_MAX = 200
_RETAINED_TOOL_PAYLOAD_KEYS = frozenset(
    {
        "run_id",
        "returncode",
        "timed_out",
        "interrupted",
        "truncated",
        "partial",
        "partial_reason",
        "helper_calls",
    }
)
_RETAINED_TOOL_CALL_KEYS = frozenset({"name", "call_id", "_status_label"})


@dataclass
class ThreadRunState:
    """In-process execution state for one thread in tui2."""

    thread_id: str
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    pending_turns: list[PendingTurn] = field(default_factory=list)
    started_at: float | None = None
    status_message: str = "running"
    last_error: str | None = None
    terminal_status: str = "working"
    assistant_cell: TranscriptCell | None = None
    reasoning_cell: TranscriptCell | None = None
    reasoning_flushed_for_current_response: bool = False
    tool_cells: dict[str, TranscriptCell] = field(default_factory=dict)
    rate_estimator: StreamRateEstimator = field(default_factory=StreamRateEstimator)
    token_ratio: ThreadTokenRatio = field(default_factory=ThreadTokenRatio)
    observed_tool_call_name_keys: set[str] = field(default_factory=set)
    assistant_display_queue: str = ""
    assistant_display_credit: float = 0.0
    last_animation_tick_at: float | None = None
    displayed_token_rate: float | None = None
    last_token_rate_display_update_at: float | None = None
    token_rate_frozen: bool = False
    frozen_token_rate: float | None = None
    token_rate_held: bool = False
    held_token_rate: float | None = None
    pending_finish_assistant_text: str | None = None
    pending_finish_after_drain: bool = False
    engine_finished: bool = False
    completion_notification_pending: bool = False

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def display_pending(self) -> bool:
        return bool(self.assistant_display_queue or self.pending_finish_after_drain)

    def reset_for_turn(self) -> None:
        self.rate_estimator = StreamRateEstimator()
        self.observed_tool_call_name_keys.clear()
        self.assistant_display_queue = ""
        self.assistant_display_credit = 0.0
        self.last_animation_tick_at = None
        self.displayed_token_rate = None
        self.last_token_rate_display_update_at = None
        self.token_rate_frozen = False
        self.frozen_token_rate = None
        self.token_rate_held = False
        self.held_token_rate = None
        self.pending_finish_assistant_text = None
        self.pending_finish_after_drain = False
        self.engine_finished = False
        self.completion_notification_pending = False
        # Token ratio is thread-level and intentionally survives turns.


def _compaction_summary_preview(text: str) -> str:
    """Keep compaction checkpoints compact in tui2 scrollback.

    The full summary remains persisted in the thread store for future model
    context; tui2 only needs a small visual checkpoint so long automatic
    compactions do not flood the terminal.
    """

    return short_block(
        text.strip(),
        max_lines=COMPACTION_SUMMARY_PREVIEW_LINES,
        max_chars=COMPACTION_SUMMARY_PREVIEW_CHARS,
    )


def _compaction_event_text(label: str, summary: object) -> str:
    preview = _compaction_summary_preview(str(summary or ""))
    return label + (f"\n{preview}" if preview else "")


def _retained_mapping(value: dict[str, Any] | None, keys: frozenset[str]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    retained: dict[str, Any] = {}
    for key in keys:
        if key not in value:
            continue
        item = value[key]
        if key == "helper_calls" and isinstance(item, list):
            retained[key] = [dict(helper) if isinstance(helper, dict) else helper for helper in item]
        else:
            retained[key] = item
    return retained or None


def _retained_flushed_cell(cell: TranscriptCell) -> TranscriptCell:
    """Return the lightweight copy kept after a cell is in terminal scrollback."""

    call = _retained_mapping(cell.call, _RETAINED_TOOL_CALL_KEYS) if cell.kind == "tool" else None
    payload = _retained_mapping(cell.payload, _RETAINED_TOOL_PAYLOAD_KEYS) if cell.kind == "tool" else None
    return TranscriptCell(
        cell.kind,
        text=cell.text,
        title=cell.title,
        status=cell.status,
        call=call,
        payload=payload,
        created_at=cell.created_at,
        finished_at=cell.finished_at,
        chars_streamed=cell.chars_streamed,
        animation_phase=cell.animation_phase,
    )


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


class AnsiUvAgentApp:
    """Terminal-native ANSI interface for :class:`uv_agent.agent.AgentEngine`."""

    def __init__(self, project_root: Path | None = None, *, data_dir: Path | None = None) -> None:
        self.project_root = (project_root or Path.cwd()).resolve()
        self.engine = create_engine(self.project_root, data_dir=data_dir)
        ui_config = getattr(self.engine.config, "ui", None)
        self.language = detect_user_language(getattr(ui_config, "language", None))
        self.state = Tui2State(
            level=self.engine.config.runtime.default_level,
            project_path=str(self.project_root),
            language=self.language,
        )
        self.state.agent_view.dispatch_level = self.state.level
        self.renderer = Renderer()
        self._thread_runs: dict[str, ThreadRunState] = {}
        self._thread_token_ratios: dict[str, ThreadTokenRatio] = {}
        self._assistant_cell: TranscriptCell | None = None
        self._reasoning_cell: TranscriptCell | None = None
        self._reasoning_flushed_for_current_response = False
        self._tool_cells: dict[str, TranscriptCell] = {}
        self._history: list[str] = load_composer_history()
        self._history_cursor: int | None = None
        self._draft: str = ""
        self._tab_state: dict[str, Any] | None = None
        self._quit_armed = False
        self._quit_armed_until: float | None = None
        self._quit_confirmation_status: str | None = None
        self._interrupt_armed = False
        self._interrupt_armed_until: float | None = None
        self._interrupt_confirmation_status: str | None = None
        self._spinner_index = 0
        self._window_title_thread_title = ""
        self._last_window_title = ""
        self._ticker_task: asyncio.Task[None] | None = None
        self._last_plain_input_at: float | None = None
        self._skip_next_lf_after_plain_cr = False
        self._picker_mode: str = "command"
        self._mention_start: int | None = None
        self._mention_query: str = ""
        # Match the Textual TUI's lazy Goal behavior: enabling Goal for the
        # unsaved draft thread should update the UI immediately, but must not
        # create thread records or goal files until the first message is sent.
        self._pending_goal_enable = False
        self._pending_goal_objective = ""
        self._image_sequence_next = 1
        self._image_paths_by_number: dict[int, Path] = {}
        self._image_status_token: str | None = None
        self._image_status_message: str | None = None
        self._agent_view_local_threads: set[str] = set()
        self._agent_view_join_pending_persist: set[str] = set()

    @property
    def _running_task(self) -> asyncio.Task[None] | None:
        """Compatibility view of the attached thread's running task."""

        thread_id = self.state.thread_id
        if not thread_id:
            return None
        run_state = self._thread_runs.get(thread_id)
        return run_state.task if run_state is not None else None

    @_running_task.setter
    def _running_task(self, task: asyncio.Task[None] | None) -> None:
        """Compatibility setter used by older tests to simulate a busy turn."""

        thread_id = self.state.thread_id or "__draft__"
        if task is None:
            self._thread_runs.pop(thread_id, None)
            return
        run_state = self._thread_runs.setdefault(thread_id, ThreadRunState(thread_id=thread_id))
        run_state.task = task
        if self.state.thread_id is None:
            self.state.thread_id = thread_id

    @property
    def cancel_event(self) -> asyncio.Event | None:
        """Compatibility view of the attached thread's cancel event."""

        thread_id = self.state.thread_id
        if not thread_id:
            return None
        run_state = self._thread_runs.get(thread_id)
        return run_state.cancel_event if run_state is not None else None

    @cancel_event.setter
    def cancel_event(self, event: asyncio.Event | None) -> None:
        thread_id = self.state.thread_id or "__draft__"
        if event is None:
            run_state = self._thread_runs.get(thread_id)
            if run_state is not None and not run_state.running:
                self._thread_runs.pop(thread_id, None)
            return
        run_state = self._thread_runs.setdefault(thread_id, ThreadRunState(thread_id=thread_id))
        run_state.cancel_event = event
        if self.state.thread_id is None:
            self.state.thread_id = thread_id

    def _run_state(self, thread_id: str | None = None) -> ThreadRunState | None:
        resolved = thread_id or self.state.thread_id
        if not resolved:
            return None
        return self._thread_runs.get(resolved)

    def _run_state_for_event(self, event: dict[str, Any]) -> ThreadRunState | None:
        event_thread_id = str(event.get("thread_id") or self.state.thread_id or "")
        return self._thread_runs.get(event_thread_id) if event_thread_id else None

    def _token_ratio_for_thread(self, thread_id: str | None) -> ThreadTokenRatio:
        """Return the thread-level visible-character/output-token ratio state."""

        key = thread_id or self.state.thread_id or "__draft__"
        ratio = self._thread_token_ratios.get(key)
        if ratio is None:
            ratio = self._load_thread_token_ratio(thread_id) if thread_id else ThreadTokenRatio()
            self._thread_token_ratios[key] = ratio
        return ratio

    def _load_thread_token_ratio(self, thread_id: str | None) -> ThreadTokenRatio:
        if not thread_id:
            return ThreadTokenRatio()
        try:
            events = self.engine.thread_store.read_events(thread_id, event_types={"item.model_response"})
        except Exception:
            return ThreadTokenRatio()
        ratio = ThreadTokenRatio()
        for event in events:
            reasoning_text = str(event.get("reasoning_text") or "")
            response_units = model_response_visible_units(
                event.get("output") if isinstance(event.get("output"), list) else [],
                reasoning_text=reasoning_text,
            )
            ratio.observe_response(
                visible_units=response_units,
                output_tokens=usage_output_tokens(
                    event.get("usage") if isinstance(event.get("usage"), dict) else {},
                    reasoning_visible=bool(reasoning_text),
                ),
            )
        return ratio

    def _current_char_rate(self, run_state: ThreadRunState | None = None, *, now: float | None = None) -> float | None:
        run_state = run_state or self._run_state()
        if run_state is None:
            return None
        return run_state.rate_estimator.current_cps(now=monotonic() if now is None else now)

    def _current_visible_unit_rate(
        self,
        run_state: ThreadRunState | None = None,
        *,
        now: float | None = None,
    ) -> float | None:
        run_state = run_state or self._run_state()
        if run_state is None:
            return None
        return run_state.rate_estimator.current_ups(now=monotonic() if now is None else now)

    def _current_token_rate(self, run_state: ThreadRunState | None = None, *, now: float | None = None) -> float | None:
        run_state = run_state or self._run_state()
        if run_state is None:
            return None
        return run_state.token_ratio.token_rate(self._current_visible_unit_rate(run_state, now=now))

    def _freeze_token_rate(self, run_state: ThreadRunState | None = None, *, now: float | None = None) -> None:
        run_state = run_state or self._run_state()
        if run_state is None or run_state.token_rate_frozen:
            return
        run_state.token_rate_held = False
        run_state.held_token_rate = None
        now = monotonic() if now is None else now
        frozen = run_state.displayed_token_rate
        if frozen is None:
            frozen = self._display_token_rate(run_state, now=now)
        run_state.frozen_token_rate = frozen
        run_state.token_rate_frozen = frozen is not None

    def _hold_token_rate(self, run_state: ThreadRunState | None = None) -> None:
        """Keep the last shown token rate without marking it as frozen."""

        run_state = run_state or self._run_state()
        if run_state is None or run_state.token_rate_held:
            return
        held = run_state.displayed_token_rate
        run_state.held_token_rate = held
        run_state.token_rate_held = True
        run_state.token_rate_frozen = False
        run_state.frozen_token_rate = None

    def _resume_token_rate(self, run_state: ThreadRunState | None = None, *, now: float | None = None) -> None:
        run_state = run_state or self._run_state()
        if run_state is None or not (run_state.token_rate_frozen or run_state.token_rate_held):
            return
        now = monotonic() if now is None else now
        resume_rate = (
            run_state.frozen_token_rate if run_state.token_rate_frozen else run_state.held_token_rate
        )
        if resume_rate is not None:
            run_state.displayed_token_rate = resume_rate
            run_state.last_token_rate_display_update_at = now
        run_state.token_rate_frozen = False
        run_state.frozen_token_rate = None
        run_state.token_rate_held = False
        run_state.held_token_rate = None

    def _display_token_rate(self, run_state: ThreadRunState, *, now: float | None = None) -> float | None:
        """Return the row-1 token rate with display-only smoothing applied."""

        if run_state.token_rate_held:
            return run_state.held_token_rate
        if run_state.token_rate_frozen:
            return run_state.frozen_token_rate
        now = monotonic() if now is None else now
        instant = self._current_token_rate(run_state, now=now)
        if instant is None or instant < TOKEN_RATE_DISPLAY_HIDE_BELOW:
            # Clearing on the display cadence avoids a one-frame flicker without
            # keeping a stale speed visible through long model-output pauses.
            last_update = run_state.last_token_rate_display_update_at
            if (
                run_state.displayed_token_rate is not None
                and last_update is not None
                and now - last_update < TOKEN_RATE_DISPLAY_UPDATE_INTERVAL_S
            ):
                return run_state.displayed_token_rate
            run_state.displayed_token_rate = None
            run_state.last_token_rate_display_update_at = None
            return None

        previous = run_state.displayed_token_rate
        last_update = run_state.last_token_rate_display_update_at
        if previous is None or last_update is None:
            run_state.displayed_token_rate = instant
            run_state.last_token_rate_display_update_at = now
            return instant
        dt = max(0.0, now - last_update)
        if dt < TOKEN_RATE_DISPLAY_UPDATE_INTERVAL_S:
            return previous
        alpha = 1.0 - exp(-dt / TOKEN_RATE_DISPLAY_TAU_S)
        displayed = previous + alpha * (instant - previous)
        run_state.displayed_token_rate = displayed
        run_state.last_token_rate_display_update_at = now
        return displayed if displayed >= TOKEN_RATE_DISPLAY_HIDE_BELOW else None

    def _interruptible_run_state(self) -> ThreadRunState | None:
        """Return the attached run only when Ctrl+C should interrupt it.

        A completed asyncio task remains ``not done()`` while its own
        ``finally`` block is executing, so stale run states can otherwise keep a
        cancel event around after the turn has ended.  Likewise, a turn that has
        already received an interrupt request may keep unwinding for a while.
        Ctrl+C must ignore both cases and fall through to the normal quit
        confirmation instead of repeatedly "interrupting" a non-interruptible
        state.
        """

        run_state = self._run_state()
        if run_state is None:
            return None
        if self._run_state_is_active(run_state):
            return run_state
        # Some tests and embedders simulate a busy turn by installing only a
        # cancel event.  Keep that path working, but do not treat an already
        # completed task as interruptible.
        if (
            run_state.task is None
            and not run_state.cancel_event.is_set()
            and run_state.terminal_status not in _RUN_TERMINAL_STATUSES
            and (self.state.busy or self._interrupt_armed)
        ):
            return run_state
        return None

    @staticmethod
    def _run_state_is_active(run_state: ThreadRunState) -> bool:
        return (
            run_state.running
            and not run_state.cancel_event.is_set()
            and run_state.terminal_status not in _RUN_TERMINAL_STATUSES
        )

    def _inactive_run_status_message(self, run_state: ThreadRunState) -> str:
        if run_state.terminal_status == "interrupted":
            return self._text("interrupted")
        if run_state.terminal_status == "failed" and run_state.last_error:
            return run_state.last_error
        return "ready"

    def _run_state_busy_for_ui(self, run_state: ThreadRunState) -> bool:
        return self._run_state_is_active(run_state) or run_state.display_pending

    def _active_confirmation_status(self) -> str | None:
        if self._quit_armed and self._quit_confirmation_status:
            return self._quit_confirmation_status
        if self._interrupt_armed and self._interrupt_confirmation_status:
            return self._interrupt_confirmation_status
        return None

    def _is_attached_thread(self, thread_id: str | None) -> bool:
        return bool(thread_id and thread_id == self.state.thread_id)

    def _sync_attached_run_state(self, run_state: ThreadRunState | None = None) -> None:
        run_state = run_state or self._run_state()
        if run_state is None:
            self.state.busy = False
            self.state.pending_turns.clear()
            self.state.turn_elapsed_s = None
            self.state.turn_token_rate = None
            self.state.turn_token_rate_frozen = False
            return
        active = self._run_state_is_active(run_state)
        ui_busy = self._run_state_busy_for_ui(run_state)
        self.state.busy = ui_busy
        self.state.pending_turns = run_state.pending_turns
        confirmation_status = self._active_confirmation_status()
        self.state.status_message = (
            confirmation_status
            if confirmation_status is not None
            else run_state.status_message
            if ui_busy
            else self._inactive_run_status_message(run_state)
        )
        self.state.last_error = run_state.last_error
        self._assistant_cell = run_state.assistant_cell
        self._reasoning_cell = run_state.reasoning_cell
        self._reasoning_flushed_for_current_response = run_state.reasoning_flushed_for_current_response
        self._tool_cells = run_state.tool_cells
        now = monotonic()
        self.state.turn_token_rate = self._display_token_rate(run_state, now=now) if ui_busy else None
        self.state.turn_token_rate_frozen = bool(
            ui_busy and run_state.token_rate_frozen and self.state.turn_token_rate is not None
        )
        if ui_busy and run_state.started_at is not None:
            self.state.turn_elapsed_s = now - run_state.started_at
        else:
            self.state.turn_elapsed_s = None

    def _detach_live_run_state(self) -> None:
        run_state = self._run_state()
        if run_state is not None:
            self._capture_attached_run_state(run_state)
        self._assistant_cell = None
        self._reasoning_cell = None
        self._reasoning_flushed_for_current_response = False
        self._tool_cells = {}
        self.state.live.clear()

    def _capture_attached_run_state(self, run_state: ThreadRunState | None = None) -> None:
        run_state = run_state or self._run_state()
        if run_state is None:
            return
        run_state.assistant_cell = self._assistant_cell
        run_state.reasoning_cell = self._reasoning_cell
        run_state.reasoning_flushed_for_current_response = self._reasoning_flushed_for_current_response
        run_state.tool_cells = self._tool_cells
        run_state.status_message = self.state.status_message
        run_state.last_error = self.state.last_error

    def run(self) -> None:
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        with Terminal() as terminal:
            self._ticker_task = asyncio.create_task(self._ticker())
            self._refresh_window_title()
            self._safe_repaint()
            try:
                with TerminalKeyReader(terminal) as keys:
                    while True:
                        try:
                            key = await keys.read_key()
                        except KeyboardInterrupt:
                            # Fallback for platforms/embeddings where SIGINT is
                            # still raised into the event loop.  The dedicated
                            # reader means this no longer leaks executor workers.
                            key = "\x03"
                        if not await self.handle_key(key):
                            break
                running_states = [run_state for run_state in self._thread_runs.values() if run_state.running]
                for run_state in running_states:
                    run_state.cancel_event.set()
                await asyncio.gather(
                    *(run_state.task for run_state in running_states if run_state.task is not None),
                    return_exceptions=True,
                )
            finally:
                if self._ticker_task is not None:
                    self._ticker_task.cancel()
                    await asyncio.gather(self._ticker_task, return_exceptions=True)
                    self._ticker_task = None
                self._apply_window_title()
                self.renderer.close()

    async def _ticker(self) -> None:
        # 12Hz tick: matches the breath animation's target frequency (12
        # phase changes per second at 100 chars/sec of streamed output).
        tick_interval = 1.0 / 12.0
        while True:
            await asyncio.sleep(tick_interval)
            confirmation_expired = self._expire_quit_confirmation()
            confirmation_expired = self._expire_interrupt_confirmation() or confirmation_expired
            animation_updated = self._advance_streaming_display()
            if self.state.mode == "agent_view":
                self._spinner_index += 1
                self.renderer.spinner_frame = self._spinner_index
                self._refresh_agent_view_rows()
                self._safe_repaint()
                continue
            if self.state.busy or animation_updated:
                self._spinner_index += 1
                self.renderer.spinner_frame = self._spinner_index
                self._apply_window_title()
                self._safe_repaint()
            else:
                self._apply_window_title()
                if confirmation_expired:
                    self._safe_repaint()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    async def handle_key(self, key: str) -> bool:
        if self.state.mode == "agent_view":
            return await self._handle_agent_view_key(key)

        if key == "\x03":  # Ctrl+C
            if self._interruptible_run_state() is not None:
                self._expire_interrupt_confirmation()
                if self._interrupt_armed:
                    self._interrupt_running_turn()
                    self._safe_repaint()
                    return True
                self._clear_quit_confirmation()
                self._arm_interrupt_confirmation(self._text("interrupt_again"))
                self._safe_repaint()
                return True
            self._expire_quit_confirmation()
            if self._quit_armed:
                return False
            self._clear_interrupt_confirmation()
            self._arm_quit_confirmation(self._text("quit_again"))
            self._safe_repaint()
            return True

        if key.startswith(PASTE_PREFIX):
            self._insert_pasted_text(key[len(PASTE_PREFIX) :])
            return True

        if key:
            self._clear_quit_confirmation()
            self._clear_interrupt_confirmation()
        if key != "\t":
            self._tab_state = None
        if key == "\x0c":  # Ctrl+L: force a full redraw of the live region.
            self.renderer._has_frame = False  # type: ignore[attr-defined]
            self._safe_repaint()
            return True
        if key == "\r":
            if self._enter_looks_like_unbracketed_paste():
                self._insert_composer_text("\n")
                self._reset_history()
                self._mark_plain_input()
                self._skip_next_lf_after_plain_cr = True
                self._after_composer_changed()
                self._safe_repaint()
                return True
            self._skip_next_lf_after_plain_cr = False
            if self.state.command_palette_open:
                completed = self._accept_command_palette_selection()
                if self._picker_mode != "command":
                    self._safe_repaint()
                    return True
                if completed and self.state.composer.endswith(" "):
                    self._safe_repaint()
                    return True
            return await self.submit()
        if key in {"\n", "\x0a", "<C-ENTER>"}:  # Ctrl+Enter/Ctrl+J inserts a newline.
            if key != "<C-ENTER>" and self._skip_next_lf_after_plain_cr:
                self._mark_plain_input()
                self._safe_repaint()
                return True
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._insert_composer_text("\n")
            self._after_composer_changed()
        elif key == "\x1b":
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._close_command_palette()
        if key == "\t":
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._handle_tab()
        elif key == "\x01":  # Ctrl+A: line start in a draft, Agent View when the composer is empty.
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            if self.state.composer:
                self._move_composer_to_line_start()
            else:
                self._open_agent_view()
        elif key == "\x05":  # Ctrl+E: move to the end of the current logical line.
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._move_composer_to_line_end()
        elif key == "\x0b":  # Ctrl+K: delete from the cursor to the current line end.
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._delete_composer_to_line_end()
            self._after_composer_changed()
        elif key == "\x02":  # Ctrl+B: readline-style cursor-left shortcut.
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._move_composer_cursor(-1)
        elif key == "\x06":  # Ctrl+F: readline-style cursor-right shortcut.
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._move_composer_cursor(1)
        elif key == "\x04":  # Ctrl+D: delete the character under the cursor.
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._delete_composer_after_cursor()
            self._after_composer_changed()
        elif key in {"\x7f", "\b"}:
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._delete_composer_before_cursor()
            self._after_composer_changed()
        elif key == "\x17":  # Ctrl+W
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._delete_composer_word_before_cursor()
            self._after_composer_changed()
        elif key == "\x15":  # Ctrl+U
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._set_composer_text("", cursor=0)
            self._reset_history()
            self._after_composer_changed()
        elif key in {"<H>", "<UP>"}:  # Windows/POSIX up arrow
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            if self.state.command_palette_open:
                self._move_command_palette(-1)
            else:
                if self._history_cursor is not None:
                    self._history_prev()
                elif not self._move_composer_vertical(-1):
                    self._history_prev()
        elif key in {"<P>", "<DOWN>"}:  # Windows/POSIX down arrow
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            if self.state.command_palette_open:
                self._move_command_palette(1)
            else:
                if self._history_cursor is not None:
                    self._history_next()
                elif not self._move_composer_vertical(1):
                    self._history_next()
        elif key in {"<K>", "<LEFT>"}:  # Windows/POSIX left arrow
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._move_composer_cursor(-1)
        elif key in {"<M>", "<RIGHT>"}:  # Windows/POSIX right arrow
            self._skip_next_lf_after_plain_cr = False
            self._last_plain_input_at = None
            self._move_composer_cursor(1)
        elif self._is_text_input_key(key):
            self._skip_next_lf_after_plain_cr = False
            self._insert_composer_text(key)
            self._reset_history()
            self._mark_plain_input()
            self._after_composer_changed()
        self._safe_repaint()
        return True

    @staticmethod
    def _is_text_input_key(key: str) -> bool:
        """Return True for user text keys, excluding wrapped control tokens."""
        return bool(key) and key >= " " and (key == "<" or not key.startswith("<"))

    @staticmethod
    def _delete_word(text: str) -> str:
        stripped = text.rstrip()
        if not stripped:
            return ""
        idx = max(stripped.rfind(" "), stripped.rfind("\n"))
        return stripped[: idx + 1] if idx >= 0 else ""

    async def _handle_agent_view_key(self, key: str) -> bool:
        view = self.state.agent_view
        confirmation = view.pending_confirmation
        if confirmation:
            if key in {"y", "Y"}:
                action, _, thread_id = confirmation.partition(":")
                view.pending_confirmation = None
                if action == "delete_worktree":
                    asyncio.create_task(self._delete_agent_view_worktree(thread_id))
                elif action in {"hide_thread", "delete_thread"}:
                    self._hide_agent_view_thread(thread_id)
                self._safe_repaint()
                return True
            if key in {"n", "N", "\x1b", "\x03"}:
                view.pending_confirmation = None
                view.status_message = self._text("agent_view_delete_cancelled")
                self._safe_repaint()
                return True

        if view.interaction_mode == "help":
            if key in {"?", "h", "H", "\x1b"}:
                view.interaction_mode = "normal"
                view.status_message = self._text("agent_view_normal_hint")
            elif key == "\x03":
                self._close_agent_view()
            self._safe_repaint()
            return True

        if view.interaction_mode == "model":
            return await self._handle_agent_view_model_key(key)

        if view.interaction_mode == "input":
            return await self._handle_agent_view_input_key(key)
        return await self._handle_agent_view_normal_key(key)

    async def _handle_agent_view_normal_key(self, key: str) -> bool:
        view = self.state.agent_view
        if key == "\x03":
            selected = view.selected_row()
            if selected is not None and self._cancel_agent_view_thread(selected.thread_id):
                view.status_message = self._fmt("agent_view_cancelled", thread=short_thread(selected.thread_id))
            else:
                self._close_agent_view()
            self._safe_repaint()
            return True
        if key in {"\x1b", "\x01"}:  # Esc or Ctrl+A toggles back to transcript mode.
            self._close_agent_view()
            self._safe_repaint()
            return True
        if key in {"?", "h", "H"}:
            view.interaction_mode = "help"
            view.status_message = self._text("agent_view_help_status")
        elif key in {"i", "a", "A"}:
            self._enter_agent_view_input_mode(target="dispatch")
        elif key in {"m", "M"}:
            self._open_agent_view_model_picker()
        elif key == "r":
            selected = view.selected_row()
            if selected is None:
                view.status_message = self._text("agent_view_select_reply")
            else:
                self._enter_agent_view_input_mode(target="reply", thread_id=selected.thread_id)
        elif key == "\r":
            selected = view.selected_row()
            if selected is not None:
                self._resume_thread(selected.thread_id)
                self.state.mode = "transcript"
        elif key in {"<H>", "<UP>", "k"}:
            self._move_agent_view_selection(-1)
        elif key in {"<P>", "<DOWN>", "j"}:
            self._move_agent_view_selection(1)
        elif key in {"<I>", "<PAGEUP>"}:
            self._move_agent_view_selection(-5)
        elif key in {"<Q>", "<PAGEDOWN>"}:
            self._move_agent_view_selection(5)
        elif key == " ":
            view.peek_expanded = not view.peek_expanded
        elif key == "c":
            selected = view.selected_row()
            if selected is not None and self._cancel_agent_view_thread(selected.thread_id):
                view.status_message = self._fmt("agent_view_cancelled", thread=short_thread(selected.thread_id))
            else:
                view.status_message = self._text("no_running_turn")
        elif key == "d":
            self._confirm_agent_view_delete(include_worktree=False)
        elif key == "D":
            self._confirm_agent_view_delete(include_worktree=True)
        self._safe_repaint()
        return True

    async def _handle_agent_view_model_key(self, key: str) -> bool:
        view = self.state.agent_view
        if key in {"\x03", "\x1b", "m", "M"}:
            view.interaction_mode = "normal"
            view.status_message = self._text("agent_view_normal_hint")
        elif key in {"<H>", "<UP>", "k"}:
            self._move_agent_view_model_selection(-1)
        elif key in {"<P>", "<DOWN>", "j"}:
            self._move_agent_view_model_selection(1)
        elif key in {"<I>", "<PAGEUP>"}:
            self._move_agent_view_model_selection(-5)
        elif key in {"<Q>", "<PAGEDOWN>"}:
            self._move_agent_view_model_selection(5)
        elif key == "\r":
            self._accept_agent_view_model_selection()
        self._safe_repaint()
        return True

    async def _handle_agent_view_input_key(self, key: str) -> bool:
        if key == "\x03":  # Ctrl+C cancels editing in Agent View input mode.
            self._leave_agent_view_input_mode(clear=True)
            self._safe_repaint()
            return True
        if key == "\x1b":
            self._leave_agent_view_input_mode(clear=False)
            self._safe_repaint()
            return True
        if key.startswith(PASTE_PREFIX):
            self._insert_agent_view_text(key[len(PASTE_PREFIX) :])
            self._safe_repaint()
            return True
        if key == "\r":
            await self._submit_agent_view_input()
            self._safe_repaint()
            return True
        if key in {"\n", "\x0a", "<C-ENTER>"}:
            self._insert_agent_view_text("\n")
        elif key == "\x01":
            self._move_agent_view_to_line_start()
        elif key == "\x05":
            self._move_agent_view_to_line_end()
        elif key == "\x0b":
            self._delete_agent_view_to_line_end()
        elif key in {"\x7f", "\b"}:
            self._delete_agent_view_before_cursor()
        elif key == "\x04":
            self._delete_agent_view_after_cursor()
        elif key == "\x17":
            self._delete_agent_view_word_before_cursor()
        elif key == "\x15":
            self._set_agent_view_text("", cursor=0)
        elif key == "\x02" or key in {"<K>", "<LEFT>"}:
            self._move_agent_view_cursor(-1)
        elif key == "\x06" or key in {"<M>", "<RIGHT>"}:
            self._move_agent_view_cursor(1)
        elif self._is_text_input_key(key):
            self._insert_agent_view_text(key)
        self._safe_repaint()
        return True

    async def _submit_agent_view_input(self) -> None:
        view = self.state.agent_view
        text = view.composer.strip()
        if not text:
            view.status_message = self._text("agent_view_input_status")
            return
        target = view.input_target
        target_thread_id = view.input_target_thread_id
        self._set_agent_view_text("", cursor=0)
        self._leave_agent_view_input_mode(clear=False)
        if target == "reply":
            self._reply_to_agent_view_thread(target_thread_id, text)
            return
        asyncio.create_task(self._dispatch_agent_view_prompt(text))
        view.status_message = self._text("agent_view_dispatching")

    def _enter_agent_view_input_mode(self, *, target: str, thread_id: str | None = None) -> None:
        view = self.state.agent_view
        view.interaction_mode = "input"
        view.input_target = "reply" if target == "reply" else "dispatch"
        view.input_target_thread_id = thread_id
        view.status_message = self._text("agent_view_reply_status" if view.input_target == "reply" else "agent_view_input_status")
        view.composer_cursor = self._agent_view_cursor()

    def _leave_agent_view_input_mode(self, *, clear: bool) -> None:
        view = self.state.agent_view
        if clear:
            self._set_agent_view_text("", cursor=0)
        view.interaction_mode = "normal"
        view.input_target = "dispatch"
        view.input_target_thread_id = None
        view.status_message = self._text("agent_view_normal_hint")

    def _open_agent_view_model_picker(self) -> None:
        view = self.state.agent_view
        view.model_options = list(self._agent_view_model_options())
        current = self._agent_view_dispatch_level()
        view.model_selected = 0
        for index, option in enumerate(view.model_options):
            if option.id == current:
                view.model_selected = index
                break
        view.interaction_mode = "model"
        view.status_message = self._text("agent_view_model_status")

    def _agent_view_model_options(self) -> tuple[CommandSuggestion, ...]:
        levels = sorted(getattr(self.engine.config, "levels", {}).keys())
        return tuple(
            CommandSuggestion(name, self._level_model_name(name), id=name, kind="model")
            for name in levels
        )

    def _agent_view_dispatch_level(self) -> str:
        view = self.state.agent_view
        source_level = view.dispatch_level if view.dispatch_level_explicit else self.state.level
        level = source_level or self.engine.config.runtime.default_level
        if level in getattr(self.engine.config, "levels", {}):
            return level
        return self.engine.config.runtime.default_level

    def _move_agent_view_model_selection(self, delta: int) -> None:
        view = self.state.agent_view
        if not view.model_options:
            view.model_selected = 0
            return
        view.model_selected = max(0, min(len(view.model_options) - 1, view.model_selected + delta))

    def _accept_agent_view_model_selection(self) -> None:
        view = self.state.agent_view
        if not view.model_options:
            view.status_message = self._text("agent_view_no_models")
            return
        option = view.model_options[max(0, min(view.model_selected, len(view.model_options) - 1))]
        level = option.id or option.value
        view.dispatch_level = level
        view.dispatch_level_explicit = True
        view.interaction_mode = "normal"
        view.status_message = self._fmt("agent_view_model_set", level=level)

    def _open_agent_view(self) -> None:
        self._close_command_palette()
        self._refresh_agent_view_rows()
        self.state.mode = "agent_view"
        self.state.agent_view.interaction_mode = "normal"
        self.state.agent_view.input_target = "dispatch"
        self.state.agent_view.input_target_thread_id = None
        self.state.agent_view.dispatch_level = self._agent_view_dispatch_level()
        if self.state.agent_view.selected >= len(self.state.agent_view.rows):
            self.state.agent_view.selected = max(0, len(self.state.agent_view.rows) - 1)
        self.state.agent_view.status_message = self._text("agent_view_open_status")

    def _background_current_thread(self) -> None:
        self._close_command_palette()
        joined = self._join_current_thread_to_agent_view()
        self._refresh_agent_view_rows()
        if self.state.thread_id:
            for index, row in enumerate(self.state.agent_view.rows):
                if row.thread_id == self.state.thread_id:
                    self.state.agent_view.selected = index
                    break
        self.state.mode = "agent_view"
        self.state.agent_view.interaction_mode = "normal"
        self.state.agent_view.input_target = "dispatch"
        self.state.agent_view.input_target_thread_id = None
        self.state.agent_view.dispatch_level = self._agent_view_dispatch_level()
        if joined is True:
            self.state.agent_view.status_message = self._fmt(
                "agent_view_bg_added",
                thread=short_thread(self.state.thread_id),
            )
        elif joined is False:
            self.state.agent_view.status_message = self._fmt(
                "agent_view_bg_present",
                thread=short_thread(self.state.thread_id),
            )
        else:
            self.state.agent_view.status_message = self._text("agent_view_open_status")

    def _join_current_thread_to_agent_view(self) -> bool | None:
        thread_id = self.state.thread_id
        if not thread_id:
            return None
        metadata = self._thread_metadata(thread_id)
        if not metadata.get("agent_view_deleted") and (
            metadata.get("agent_view_joined") or thread_id in self._agent_view_local_threads
        ):
            return False
        self._agent_view_local_threads.add(thread_id)
        try:
            self.engine.thread_store.append(thread_id, "thread.agent_view_joined", source="bg_command")
        except ThreadLockedError:
            self._agent_view_join_pending_persist.add(thread_id)
        return True

    def _persist_pending_agent_view_join(self, thread_id: str) -> None:
        if thread_id not in self._agent_view_join_pending_persist:
            return
        metadata = self._thread_metadata(thread_id)
        if metadata.get("agent_view_joined"):
            self._agent_view_join_pending_persist.discard(thread_id)
            return
        try:
            self.engine.thread_store.append(thread_id, "thread.agent_view_joined", source="bg_command")
        except ThreadLockedError:
            return
        self._agent_view_join_pending_persist.discard(thread_id)

    def _close_agent_view(self) -> None:
        self.state.mode = "transcript"
        self.state.agent_view.pending_confirmation = None
        self.state.agent_view.interaction_mode = "normal"

    def _move_agent_view_selection(self, delta: int) -> None:
        rows = self.state.agent_view.rows
        if not rows:
            self.state.agent_view.selected = 0
            return
        self.state.agent_view.selected = max(0, min(len(rows) - 1, self.state.agent_view.selected + delta))

    def _ordered_agent_view_rows(self) -> list[AgentViewRow]:
        indexed_threads = {row.thread_id: index for index, row in enumerate(self.state.agent_view.rows)}
        rows: list[AgentViewRow] = []
        seen: set[str] = set()
        for thread in self.engine.thread_store.list_threads()[:100]:
            thread_id = str(thread.get("thread_id") or "")
            if not thread_id or thread.get("agent_view_deleted"):
                continue
            if not self._is_agent_view_thread(thread):
                continue
            rows.append(self._agent_view_row_for_thread(thread_id, thread))
            seen.add(thread_id)

        for thread_id in sorted(self._agent_view_local_threads.difference(seen)):
            try:
                thread = self._thread_metadata(thread_id)
            except Exception:
                thread = {"thread_id": thread_id, "title": "New thread"}
            if thread.get("agent_view_deleted") and thread_id not in self._agent_view_join_pending_persist:
                continue
            rows.append(self._agent_view_row_for_thread(thread_id, thread))

        def sort_key(row: AgentViewRow) -> tuple[int, int]:
            return (
                _AGENT_VIEW_STATUS_RANK.get(row.status, len(_AGENT_VIEW_STATUS_RANK)),
                indexed_threads.get(row.thread_id, len(indexed_threads)),
            )

        return sorted(rows, key=sort_key)

    @staticmethod
    def _is_agent_view_thread(thread: dict[str, Any]) -> bool:
        if thread.get("agent_view_joined"):
            return True
        return bool(thread.get("worktree_branch") or thread.get("worktree_path"))

    def _agent_view_cursor(self) -> int:
        cursor = self.state.agent_view.composer_cursor
        text = self.state.agent_view.composer
        if cursor is None:
            return len(text)
        return max(0, min(cursor, len(text)))

    def _set_agent_view_text(self, text: str, *, cursor: int | None = None) -> None:
        self.state.agent_view.composer = text
        self.state.agent_view.composer_cursor = len(text) if cursor is None else max(0, min(cursor, len(text)))

    def _insert_agent_view_text(self, text: str) -> None:
        cursor = self._agent_view_cursor()
        value = self.state.agent_view.composer
        self._set_agent_view_text(value[:cursor] + text + value[cursor:], cursor=cursor + len(text))

    def _delete_agent_view_before_cursor(self) -> None:
        cursor = self._agent_view_cursor()
        if cursor <= 0:
            return
        value = self.state.agent_view.composer
        self._set_agent_view_text(value[: cursor - 1] + value[cursor:], cursor=cursor - 1)

    def _delete_agent_view_after_cursor(self) -> None:
        cursor = self._agent_view_cursor()
        value = self.state.agent_view.composer
        if cursor >= len(value):
            return
        self._set_agent_view_text(value[:cursor] + value[cursor + 1 :], cursor=cursor)

    def _delete_agent_view_word_before_cursor(self) -> None:
        cursor = self._agent_view_cursor()
        value = self.state.agent_view.composer
        before = self._delete_word(value[:cursor])
        self._set_agent_view_text(before + value[cursor:], cursor=len(before))

    def _delete_agent_view_to_line_end(self) -> None:
        cursor = self._agent_view_cursor()
        value = self.state.agent_view.composer
        end = value.find("\n", cursor)
        end = len(value) if end < 0 else end
        self._set_agent_view_text(value[:cursor] + value[end:], cursor=cursor)

    def _move_agent_view_cursor(self, delta: int) -> None:
        self.state.agent_view.composer_cursor = max(
            0,
            min(self._agent_view_cursor() + delta, len(self.state.agent_view.composer)),
        )

    def _move_agent_view_to_line_start(self) -> None:
        cursor = self._agent_view_cursor()
        self.state.agent_view.composer_cursor = self.state.agent_view.composer.rfind("\n", 0, cursor) + 1

    def _move_agent_view_to_line_end(self) -> None:
        cursor = self._agent_view_cursor()
        end = self.state.agent_view.composer.find("\n", cursor)
        self.state.agent_view.composer_cursor = len(self.state.agent_view.composer) if end < 0 else end

    def _refresh_agent_view_rows(self) -> None:
        previous = self.state.agent_view.selected_row()
        previous_id = previous.thread_id if previous is not None else None
        rows = self._ordered_agent_view_rows()
        self.state.agent_view.rows = rows
        if previous_id:
            for index, row in enumerate(rows):
                if row.thread_id == previous_id:
                    self.state.agent_view.selected = index
                    break
            else:
                self.state.agent_view.selected = min(self.state.agent_view.selected, max(0, len(rows) - 1))
        else:
            self.state.agent_view.selected = min(self.state.agent_view.selected, max(0, len(rows) - 1))

    def _agent_view_row_for_thread(self, thread_id: str, thread: dict[str, Any]) -> AgentViewRow:
        status = self._agent_view_thread_status(thread_id, thread)
        run_state = self._thread_runs.get(thread_id)
        return AgentViewRow(
            thread_id=thread_id,
            title=str(thread.get("title") or "New thread"),
            status=status,
            summary=str(thread.get("last_text") or ""),
            updated_at=str(thread.get("updated_at") or ""),
            worktree_branch=str(thread.get("worktree_branch") or ""),
            worktree_path=str(thread.get("worktree_path") or ""),
            elapsed_seconds=monotonic() - run_state.started_at if run_state and run_state.started_at else 0.0,
            queued_turns=len(run_state.pending_turns) if run_state is not None else 0,
        )

    def _agent_view_thread_status(self, thread_id: str, thread: dict[str, Any]) -> str:
        run_state = self._thread_runs.get(thread_id)
        if run_state is not None and run_state.terminal_status == "dispatching":
            return "dispatching"
        if run_state is not None and run_state.display_pending:
            return "working"
        if run_state is not None and run_state.terminal_status in _RUN_TERMINAL_STATUSES:
            return run_state.terminal_status
        if run_state is not None and self._run_state_busy_for_ui(run_state):
            return "working"
        if run_state is not None and run_state.pending_turns:
            return "queued"
        event_type = self._latest_thread_terminal_event_type(thread_id)
        if event_type == "turn.error":
            return "failed"
        if event_type == "turn.interrupted":
            return "interrupted"
        return "completed"

    def _latest_thread_terminal_event_type(self, thread_id: str) -> str:
        read_recent = getattr(self.engine.thread_store, "read_recent_events", None)
        if not callable(read_recent):
            return ""
        try:
            events, _has_more = read_recent(
                thread_id,
                limit=1,
                event_types={"turn.completed", "turn.error", "turn.interrupted"},
            )
        except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError):
            return ""
        return str(events[0].get("type") or "") if events else ""

    def _reply_to_agent_view_thread(self, thread_id: str | None, text: str) -> None:
        view = self.state.agent_view
        if not thread_id:
            view.status_message = self._text("agent_view_select_reply")
            return
        if not text:
            view.status_message = self._text("agent_view_type_reply")
            return
        run_state = self._thread_runs.get(thread_id)
        if run_state is not None and self._run_state_busy_for_ui(run_state):
            run_state.pending_turns.append(PendingTurn(text, []))
            view.status_message = self._fmt("agent_view_reply_queued", thread=short_thread(thread_id))
            self._refresh_agent_view_rows()
            return
        asyncio.create_task(self._start_turn_for_thread(thread_id, text, image_paths=[]))
        view.status_message = self._fmt("agent_view_reply_sent", thread=short_thread(thread_id))

    def _reply_to_agent_view_selection(self) -> None:
        selected = self.state.agent_view.selected_row()
        text = self.state.agent_view.composer.strip()
        self._set_agent_view_text("", cursor=0)
        self._reply_to_agent_view_thread(selected.thread_id if selected else None, text)

    def _cancel_agent_view_thread(self, thread_id: str) -> bool:
        run_state = self._thread_runs.get(thread_id)
        if run_state is None or not run_state.running:
            return False
        run_state.cancel_event.set()
        run_state.terminal_status = "interrupted"
        run_state.status_message = self._text("interrupted")
        self._refresh_agent_view_rows()
        return True

    def _confirm_agent_view_delete(self, *, include_worktree: bool) -> None:
        selected = self.state.agent_view.selected_row()
        if selected is None:
            self.state.agent_view.status_message = self._text("agent_view_delete_select")
            return
        if include_worktree and not (selected.worktree_branch and selected.worktree_path):
            self.state.agent_view.status_message = self._text("agent_view_no_worktree")
            return
        action = "delete_worktree" if include_worktree else "hide_thread"
        target = selected.worktree_branch if include_worktree else short_thread(selected.thread_id)
        self.state.agent_view.pending_confirmation = f"{action}:{selected.thread_id}"
        if include_worktree:
            self.state.agent_view.status_message = self._fmt("agent_view_delete_worktree_status", target=target)
        else:
            self.state.agent_view.status_message = self._fmt("agent_view_hide_thread_status", target=target)

    def _hide_agent_view_thread(self, thread_id: str) -> None:
        self._cancel_agent_view_thread(thread_id)
        try:
            self.engine.thread_store.append(thread_id, "thread.agent_view_deleted")
        except ThreadLockedError:
            self.state.agent_view.status_message = self._fmt(
                "agent_view_delete_locked",
                thread=short_thread(thread_id),
            )
            return
        self._thread_runs.pop(thread_id, None)
        if thread_id == self.state.thread_id:
            self._clear_to_new_thread()
        self.state.agent_view.status_message = self._fmt("agent_view_hidden", thread=short_thread(thread_id))
        self._refresh_agent_view_rows()

    async def _delete_agent_view_worktree(self, thread_id: str) -> None:
        self._cancel_agent_view_thread(thread_id)
        metadata = self._thread_metadata(thread_id)
        branch = str(metadata.get("worktree_branch") or "")
        path_text = str(metadata.get("worktree_path") or "")
        if not branch or not path_text:
            self.state.agent_view.status_message = self._text("agent_view_no_worktree")
            self._safe_repaint()
            return
        try:
            result = await asyncio.to_thread(
                cleanup_worktree,
                self.project_root,
                branch,
                Path(path_text),
                run=self._run_worktree_command,
            )
        except (WorktreeError, OSError, subprocess.SubprocessError) as exc:
            self.state.agent_view.status_message = self._fmt("agent_view_worktree_delete_failed", error=exc)
            self._safe_repaint()
            return
        try:
            self.engine.thread_store.append(
                thread_id,
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
            self.engine.thread_store.append(thread_id, "thread.cwd_updated", cwd=str(self.project_root.resolve()))
        except ThreadLockedError:
            self.state.agent_view.status_message = self._fmt(
                "agent_view_delete_locked",
                thread=short_thread(thread_id),
            )
            self._safe_repaint()
            return
        rule_states = getattr(self.engine, "_rule_states", None)
        if isinstance(rule_states, dict):
            rule_states.pop(thread_id, None)
        self.state.agent_view.status_message = self._fmt("agent_view_worktree_deleted", branch=result.branch)
        self._refresh_agent_view_rows()
        self._safe_repaint()

    async def _dispatch_agent_view_prompt(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            return
        thread_id = self.engine.thread_store.create_thread(self._agent_view_thread_title(prompt))
        run_state = self._thread_runs.setdefault(thread_id, ThreadRunState(thread_id=thread_id))
        self.engine.thread_store.append(thread_id, "thread.agent_view_joined", source="agent_view_dispatch")
        run_state.terminal_status = "dispatching"
        run_state.status_message = "dispatching worktree"
        self._refresh_agent_view_rows()
        self._safe_repaint()
        try:
            level = self._agent_view_dispatch_level()
            branch = await self._agent_view_branch_name(thread_id, prompt, level=level)
            info = await asyncio.to_thread(
                create_worktree,
                self.project_root,
                branch,
                run=self._run_worktree_command,
            )
            self.engine.thread_store.append(thread_id, "thread.worktree_created", **info.metadata())
            self.engine.thread_store.append(thread_id, "thread.cwd_updated", cwd=str(info.path))
            self._persist_thread_level(thread_id, level)
            run_state.terminal_status = "working"
            run_state.status_message = "running"
            await self._start_turn_for_thread(thread_id, prompt, image_paths=[])
            self.state.agent_view.status_message = self._fmt(
                "agent_view_dispatched",
                thread=short_thread(thread_id),
                branch=info.branch,
            )
        except Exception as exc:
            run_state.terminal_status = "failed"
            run_state.last_error = str(exc) or repr(exc)
            self.state.agent_view.status_message = self._fmt("agent_view_dispatch_failed", error=run_state.last_error)
        finally:
            self._refresh_agent_view_rows()
            self._safe_repaint()

    def _agent_view_thread_title(self, prompt: str) -> str:
        first = " ".join(prompt.strip().split())
        if not first:
            return "Agent task"
        return first[:77].rstrip() + "..." if len(first) > 80 else first

    async def _agent_view_branch_name(self, thread_id: str, prompt: str, *, level: str | None = None) -> str:
        short_id = thread_id.replace("thr_", "")[:8] or new_id("agent").replace("agent_", "")[:8]
        slug: str | None = None
        generate = getattr(self.engine, "generate_branch_slug", None)
        if callable(generate):
            try:
                slug = await generate(thread_id, prompt, level=level or self._agent_view_dispatch_level())
            except Exception:
                slug = None
        branch = f"agent-{slug}-{short_id}" if slug else f"agent-{short_id}"
        return validate_worktree_branch_name(branch)

    def _run_worktree_command(self, args: list[str], *, cwd: Path, timeout_s: float | None = None) -> CommandResult:
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

    def _composer_cursor(self) -> int:
        cursor = self.state.composer_cursor
        if cursor is None:
            return len(self.state.composer)
        return max(0, min(cursor, len(self.state.composer)))

    def _set_composer_text(self, text: str, *, cursor: int | None = None) -> None:
        self.state.composer = text
        self.state.composer_cursor = len(text) if cursor is None else max(0, min(cursor, len(text)))

    def _insert_composer_text(self, text: str) -> None:
        cursor = self._composer_cursor()
        value = self.state.composer
        self._set_composer_text(value[:cursor] + text + value[cursor:], cursor=cursor + len(text))

    def _insert_pasted_text(self, text: str) -> None:
        if not text:
            return
        self._clear_quit_confirmation()
        self._last_plain_input_at = None
        self._skip_next_lf_after_plain_cr = False
        self._insert_composer_text(text)
        self._reset_history()
        self._after_composer_changed()
        self._safe_repaint()

    def _attach_clipboard_image_to_composer(self) -> None:
        try:
            model = self.engine.config.model_for_level(self.state.level)
        except Exception as exc:
            self._flush(TranscriptCell("error", text=str(exc)))
            self._safe_repaint()
            return
        if getattr(model, "supports_images", None) is False:
            self._flush(TranscriptCell("error", text=self._text("image_model_disabled")))
            self._safe_repaint()
            return
        from uv_agent.clipboard import ClipboardImageError

        try:
            image = save_clipboard_image(project_tui_clipboard_dir(self.project_root))
        except ClipboardImageError as exc:
            self._flush(TranscriptCell("error", text=str(exc)))
            self._safe_repaint()
            return

        number = self._image_sequence_next
        self._image_sequence_next += 1
        self._image_paths_by_number[number] = Path(image.path)
        self.state.image_token_numbers.add(number)
        token = f"[Image #{number}]"
        if self.state.composer:
            cursor = self._composer_cursor()
            before = self.state.composer[:cursor]
            after = self.state.composer[cursor:]
            insert = token
            if before and not before.endswith((" ", "\n")):
                insert = " " + insert
            if after and not after.startswith((" ", "\n")):
                insert += " "
        else:
            insert = token
        self._insert_composer_text(insert)
        self._reset_history()
        self._after_composer_changed()
        message = f"{self._text('image_queued')} {token} · {image.width}x{image.height}"
        self._image_status_token = token
        self._image_status_message = message
        self.state.status_message = message
        self._safe_repaint()

    def _mark_plain_input(self) -> None:
        self._last_plain_input_at = monotonic()

    def _enter_looks_like_unbracketed_paste(self) -> bool:
        """Protect against terminals that paste newlines as plain Enter keys."""

        if self._last_plain_input_at is None:
            return False
        return monotonic() - self._last_plain_input_at <= UNBRACKETED_PASTE_ENTER_S

    def _arm_quit_confirmation(self, message: str) -> None:
        self._quit_armed = True
        self._quit_armed_until = monotonic() + CTRL_C_CONFIRMATION_S
        self._quit_confirmation_status = message
        self.state.status_message = message

    def _arm_interrupt_confirmation(self, message: str) -> None:
        self._interrupt_armed = True
        self._interrupt_armed_until = monotonic() + CTRL_C_CONFIRMATION_S
        self._interrupt_confirmation_status = message
        self.state.status_message = message

    def _clear_quit_confirmation(self) -> bool:
        was_armed = self._quit_armed
        message = self._quit_confirmation_status
        self._quit_armed = False
        self._quit_armed_until = None
        self._quit_confirmation_status = None
        if message and self.state.status_message == message:
            self.state.status_message = "running" if self.state.busy else "ready"
        return was_armed

    def _expire_quit_confirmation(self) -> bool:
        if not self._quit_armed:
            return False
        if self._quit_armed_until is None or monotonic() < self._quit_armed_until:
            return False
        return self._clear_quit_confirmation()

    def _clear_interrupt_confirmation(self) -> bool:
        was_armed = self._interrupt_armed
        message = self._interrupt_confirmation_status
        self._interrupt_armed = False
        self._interrupt_armed_until = None
        self._interrupt_confirmation_status = None
        if message and self.state.status_message == message:
            self.state.status_message = "running" if self.state.busy else "ready"
        return was_armed

    def _expire_interrupt_confirmation(self) -> bool:
        if not self._interrupt_armed:
            return False
        if self._interrupt_armed_until is None or monotonic() < self._interrupt_armed_until:
            return False
        return self._clear_interrupt_confirmation()

    def _interrupt_running_turn(self) -> bool:
        run_state = self._interruptible_run_state()
        if run_state is None:
            return False
        run_state.cancel_event.set()
        self._clear_interrupt_confirmation()
        self._clear_quit_confirmation()
        run_state.status_message = self._text("interrupted")
        run_state.terminal_status = "interrupted"
        self.state.status_message = self._text("interrupted")
        return True

    def _delete_composer_before_cursor(self) -> None:
        cursor = self._composer_cursor()
        if cursor <= 0:
            return
        value = self.state.composer
        token_span = self._image_token_delete_span_before_cursor(value, cursor)
        if token_span is not None:
            start, end = token_span
            self._set_composer_text(value[:start] + value[end:], cursor=start)
            return
        self._set_composer_text(value[: cursor - 1] + value[cursor:], cursor=cursor - 1)

    def _delete_composer_after_cursor(self) -> None:
        cursor = self._composer_cursor()
        value = self.state.composer
        if cursor >= len(value):
            return
        token_span = self._image_token_delete_span_after_cursor(value, cursor)
        if token_span is not None:
            start, end = token_span
            self._set_composer_text(value[:start] + value[end:], cursor=start)
            return
        self._set_composer_text(value[:cursor] + value[cursor + 1 :], cursor=cursor)

    @staticmethod
    def _image_token_delete_span_before_cursor(text: str, cursor: int) -> tuple[int, int] | None:
        for match in IMAGE_TOKEN_RE.finditer(text):
            if match.start() < cursor <= match.end():
                return AnsiUvAgentApp._expand_image_token_delete_span(text, *match.span())
        space_start = cursor
        while space_start > 0 and text[space_start - 1] in {" ", "\t"}:
            space_start -= 1
        if space_start == cursor:
            return None
        for match in IMAGE_TOKEN_RE.finditer(text):
            if match.end() == space_start:
                return match.start(), cursor
        return None

    @staticmethod
    def _image_token_delete_span_after_cursor(text: str, cursor: int) -> tuple[int, int] | None:
        for match in IMAGE_TOKEN_RE.finditer(text):
            if match.start() <= cursor < match.end():
                return AnsiUvAgentApp._expand_image_token_delete_span(text, *match.span())
        space_end = cursor
        while space_end < len(text) and text[space_end] in {" ", "\t"}:
            space_end += 1
        if space_end == cursor:
            return None
        for match in IMAGE_TOKEN_RE.finditer(text):
            if match.start() == space_end:
                return cursor, match.end()
        return None

    @staticmethod
    def _expand_image_token_delete_span(text: str, start: int, end: int) -> tuple[int, int]:
        """Remove one separator space with an image token when it is obvious."""

        before_is_space = start > 0 and text[start - 1] in {" ", "\t"}
        after_is_space = end < len(text) and text[end] in {" ", "\t"}
        if before_is_space and after_is_space:
            # Keep the word separator before the token and remove the token's
            # trailing separator, producing "a b" from "a [Image #1] b".
            return start, end + 1
        if before_is_space and end == len(text):
            return start - 1, end
        if after_is_space and start == 0:
            return start, end + 1
        return start, end

    def _delete_composer_word_before_cursor(self) -> None:
        cursor = self._composer_cursor()
        before = self.state.composer[:cursor]
        after = self.state.composer[cursor:]
        replacement = self._delete_word(before)
        self._set_composer_text(replacement + after, cursor=len(replacement))

    def _move_composer_cursor(self, delta: int) -> None:
        self.state.composer_cursor = max(0, min(self._composer_cursor() + delta, len(self.state.composer)))

    def _move_composer_to_line_start(self) -> None:
        start, _end = self._composer_line_bounds()
        self.state.composer_cursor = start

    def _move_composer_to_line_end(self) -> None:
        _start, end = self._composer_line_bounds()
        self.state.composer_cursor = end

    def _delete_composer_to_line_end(self) -> None:
        cursor = self._composer_cursor()
        value = self.state.composer
        if cursor >= len(value):
            return
        _start, end = self._composer_line_bounds()
        # Match common terminal editor behaviour: Ctrl+K removes text up to the
        # line break first, and removes the line break itself when already at EOL.
        delete_end = end + 1 if cursor == end and end < len(value) else end
        if delete_end == cursor:
            return
        self._set_composer_text(value[:cursor] + value[delete_end:], cursor=cursor)

    def _composer_line_bounds(self) -> tuple[int, int]:
        cursor = self._composer_cursor()
        value = self.state.composer
        start = value.rfind("\n", 0, cursor) + 1
        end = value.find("\n", cursor)
        if end < 0:
            end = len(value)
        return start, end

    def _move_composer_vertical(self, delta: int) -> bool:
        value = self.state.composer
        if not value:
            return False
        cursor = self._composer_cursor()
        line_start, line_end = self._composer_line_bounds()
        column = cursor - line_start
        if delta < 0:
            if line_start == 0:
                self.state.composer_cursor = line_start
                return True
            prev_end = line_start - 1
            prev_start = value.rfind("\n", 0, prev_end) + 1
            self.state.composer_cursor = prev_start + min(column, prev_end - prev_start)
            return True
        if line_end >= len(value):
            self.state.composer_cursor = line_end
            return True
        next_start = line_end + 1
        next_end = value.find("\n", next_start)
        if next_end < 0:
            next_end = len(value)
        self.state.composer_cursor = next_start + min(column, next_end - next_start)
        return True

    def _reset_history(self) -> None:
        self._history_cursor = None
        self._draft = ""

    def _remember_composer_input(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self._history and self._history[-1] == text:
            return
        self._history.append(text)
        overflow = len(self._history) - MAX_COMPOSER_HISTORY
        if overflow > 0:
            del self._history[:overflow]
        try:
            save_composer_history(self._history)
        except OSError as exc:
            self._flush(TranscriptCell("error", text=str(exc)))

    def _after_composer_changed(self) -> None:
        self._sync_image_status_with_composer()
        if self.state.composer.startswith("/") and "\n" not in self.state.composer:
            self._refresh_command_palette()
            return
        mention = self._mention_query_at_cursor()
        if mention is not None:
            start, query = mention
            self._open_mention_palette(query, start)
            return
        self._close_command_palette()

    def _sync_image_status_with_composer(self) -> None:
        self._release_images_missing_from_composer()
        if not self._image_status_token or not self._image_status_message:
            return
        if self.state.status_message != self._image_status_message:
            return
        if self._image_status_token in self.state.composer:
            return
        self._clear_image_status_tracking()
        self.state.status_message = "ready"

    def _release_images_missing_from_composer(self) -> None:
        if not self._image_paths_by_number:
            return
        present_numbers = self._image_numbers_in_text(self.state.composer)
        stale_numbers = set(self._image_paths_by_number).difference(present_numbers)
        if stale_numbers:
            self._release_image_numbers(stale_numbers)

    @staticmethod
    def _image_numbers_in_text(text: str) -> set[int]:
        numbers: set[int] = set()
        for match in IMAGE_TOKEN_RE.finditer(text):
            numbers.add(int(match.group(1)))
        return numbers

    def _release_image_numbers(self, numbers: set[int]) -> None:
        for number in numbers:
            self._image_paths_by_number.pop(number, None)
            self.state.image_token_numbers.discard(number)

    def _clear_image_status_tracking(self) -> None:
        self._image_status_token = None
        self._image_status_message = None

    def _history_prev(self) -> None:
        if not self._history:
            return
        if self._history_cursor is None:
            # Match the Textual TUI: once the composer has user text, Up/Down
            # belong to editing/navigation rather than replacing that draft with
            # persisted history.  If the user edits a recalled entry we reset the
            # cursor before calling this method, so navigation also stops there.
            if self.state.composer:
                return
            self._draft = ""
            self._history_cursor = len(self._history) - 1
        else:
            self._history_cursor = max(0, self._history_cursor - 1)
        self._set_composer_text(self._history[self._history_cursor])

    def _history_next(self) -> None:
        if self._history_cursor is None:
            return
        if self._history_cursor >= len(self._history) - 1:
            self._history_cursor = None
            self._set_composer_text(self._draft)
            return
        self._history_cursor += 1
        self._set_composer_text(self._history[self._history_cursor])

    # ------------------------------------------------------------------
    # Tab completion
    # ------------------------------------------------------------------

    def _handle_tab(self) -> None:
        text = self.state.composer
        if text.startswith("@") or self._mention_query_at_cursor() is not None:
            mention = self._mention_query_at_cursor()
            if mention is None:
                mention = (0, text)
            start, query = mention
            matches = self._mention_suggestions(query)
            if not matches:
                return
            cycling = (
                self._tab_state is not None
                and self._tab_state.get("kind") == "mention"
                and query == self._tab_state.get("query")
            )
            if not cycling:
                self._tab_state = {"kind": "mention", "matches": matches, "index": -1, "query": query, "start": start}
            assert self._tab_state is not None
            self._tab_state["index"] = (self._tab_state["index"] + 1) % len(self._tab_state["matches"])
            item = self._tab_state["matches"][self._tab_state["index"]]
            self._replace_composer_range(start, self._composer_cursor(), item.value)
            self._close_command_palette()
            return
        if not text.startswith("/"):
            return
        cycling = self._tab_state is not None and text == self._tab_state.get("last")
        if not cycling:
            matches = self._command_suggestions(text)
            if not matches:
                return
            self._tab_state = {"matches": matches, "index": -1, "last": ""}
        assert self._tab_state is not None
        self._tab_state["index"] = (self._tab_state["index"] + 1) % len(self._tab_state["matches"])
        item = self._tab_state["matches"][self._tab_state["index"]]
        self._tab_state["last"] = item.value
        self._set_composer_text(item.value)
        if item.value.endswith(" "):
            self._refresh_command_palette()
        else:
            self._close_command_palette()

    def _command_suggestions(self, text: str) -> list[CommandSuggestion]:
        query = text.lower()
        if query.startswith("/goal ") or query == "/goal":
            pool = GOAL_COMMANDS if query != "/goal" else TOP_LEVEL_COMMANDS
        elif query.startswith("/level ") or query == "/level":
            pool = self._level_command_suggestions("/level") if query != "/level" else TOP_LEVEL_COMMANDS
        elif query.startswith("/model ") or query == "/model":
            pool = self._level_command_suggestions("/model") if query != "/model" else TOP_LEVEL_COMMANDS
        else:
            pool = TOP_LEVEL_COMMANDS
        return [item for item in pool if item.value.lower().startswith(query)][:12]

    def _level_command_suggestions(self, command: str) -> tuple[CommandSuggestion, ...]:
        levels = sorted(getattr(self.engine.config, "levels", {}).keys())
        return tuple(CommandSuggestion(f"{command} {name}", "model level") for name in levels)

    def _refresh_command_palette(self) -> None:
        self._picker_mode = "command"
        self._mention_start = None
        self._mention_query = ""
        self.state.command_palette_open = True
        self.state.command_palette_items = self._command_suggestions(self.state.composer)
        if self.state.command_palette_items:
            self.state.command_palette_index = min(
                self.state.command_palette_index,
                len(self.state.command_palette_items) - 1,
            )
        else:
            self.state.command_palette_index = 0

    def _open_picker(self, mode: str, items: list[CommandSuggestion]) -> None:
        self._picker_mode = mode
        self.state.command_palette_open = True
        self.state.command_palette_items = items
        self.state.command_palette_index = 0

    def _close_command_palette(self) -> None:
        self.state.command_palette_open = False
        self.state.command_palette_items = []
        self.state.command_palette_index = 0
        self._picker_mode = "command"
        self._mention_start = None
        self._mention_query = ""

    def _move_command_palette(self, delta: int) -> None:
        if not self.state.command_palette_items:
            return
        count = len(self.state.command_palette_items)
        self.state.command_palette_index = (self.state.command_palette_index + delta) % count

    def _accept_command_palette_selection(self) -> bool:
        if not self.state.command_palette_items:
            return False
        item = self.state.command_palette_items[self.state.command_palette_index]
        mode = self._picker_mode
        if mode == "thread":
            self._close_command_palette()
            self._resume_thread(item.id or item.value)
            return True
        if mode in {"skill", "mcp", "mention"}:
            self._accept_mention_item(item)
            self._close_command_palette()
            return True
        self._set_composer_text(item.value)
        if item.value.endswith(" "):
            self._refresh_command_palette()
        else:
            self._close_command_palette()
        return True

    def _replace_composer_range(self, start: int, end: int, replacement: str) -> None:
        value = self.state.composer
        start = max(0, min(start, len(value)))
        end = max(start, min(end, len(value)))
        self._set_composer_text(value[:start] + replacement + value[end:], cursor=start + len(replacement))

    def _accept_mention_item(self, item: CommandSuggestion) -> None:
        token = item.value
        if item.kind == "thread-mention" and token.startswith("@@"):
            token = "@thread:" + token[2:]
        if not token.endswith(" "):
            token += " "
        if self._mention_start is not None:
            self._replace_composer_range(self._mention_start, self._composer_cursor(), token)
            return
        if self.state.composer and not self.state.composer.endswith((" ", "\n")):
            token = " " + token
        self._insert_composer_text(token)

    def _mention_query_at_cursor(self) -> tuple[int, str] | None:
        cursor = self._composer_cursor()
        prefix = self.state.composer[:cursor]
        line_start = max(prefix.rfind("\n") + 1, 0)
        token_start = max(prefix.rfind(" ") + 1, line_start)
        token = prefix[token_start:]
        if token == "@" or (token.startswith("@") and not token.startswith("@@") and ":" not in token):
            return token_start, token
        if token == "@@" or token.startswith("@@"):
            return token_start, token
        return None

    def _open_mention_palette(self, query: str, start: int) -> None:
        items = self._mention_suggestions(query)
        self._mention_start = start
        self._mention_query = query
        self._open_picker("mention", items)

    def _mention_suggestions(self, query: str) -> list[CommandSuggestion]:
        query = query or "@"
        if query.startswith("@@"):
            return self._thread_mention_suggestions(query[2:].lower())
        if query.startswith("@"):
            return self._file_mention_suggestions(query[1:].lower())
        return []

    def _file_mention_suggestions(self, needle: str = "") -> list[CommandSuggestion]:
        items: list[CommandSuggestion] = []
        root = self.project_root.resolve()
        stack = [root]
        directories_seen = 0
        while stack and len(items) < MAX_MENTION_ITEMS:
            directory = stack.pop(0)
            directories_seen += 1
            if directories_seen > 20_000:
                break
            try:
                entries = sorted(
                    directory.iterdir(),
                    key=lambda path: (not path.is_dir(), path.name.casefold()),
                )
            except OSError:
                continue
            for path in entries:
                if len(items) >= MAX_MENTION_ITEMS:
                    break
                try:
                    relative = path.relative_to(root)
                except ValueError:
                    continue
                parts = relative.parts
                if any(part in IGNORED_MENTION_DIRS for part in parts[:-1]):
                    continue
                title = relative.as_posix()
                if path.is_dir():
                    if path.name.startswith(".") or path.name in IGNORED_MENTION_DIRS:
                        continue
                    stack.append(path)
                    value = f"@{title}/"
                    if needle and needle not in title.lower():
                        continue
                    items.append(CommandSuggestion(value, "directory", id=title + "/", kind="file-mention"))
                    continue
                if not path.is_file() or path.suffix.lower() not in CODE_FILE_SUFFIXES:
                    continue
                if needle and needle not in title.lower():
                    continue
                items.append(CommandSuggestion(f"@{title}", "file", id=title, kind="file-mention"))
        return items[:50]

    def _thread_mention_suggestions(self, needle: str = "") -> list[CommandSuggestion]:
        items: list[CommandSuggestion] = []
        for thread in self.engine.thread_store.list_threads()[:50]:
            thread_id = str(thread.get("thread_id") or "")
            if not thread_id:
                continue
            title = str(thread.get("title") or "New thread")
            last_text = str(thread.get("last_text") or "").replace("\n", " ")
            haystack = f"{thread_id} {title} {last_text}".lower()
            if needle and needle not in haystack:
                continue
            description = last_text[:100] if last_text else f"{thread.get('turn_count', 0)} turns"
            items.append(
                CommandSuggestion(
                    f"@@{thread_id}",
                    description,
                    id=thread_id,
                    kind="thread-mention",
                    meta=short_thread(thread_id),
                )
            )
        return items[:20]

    def _skill_mention_suggestions(self, needle: str = "") -> list[CommandSuggestion]:
        items: list[CommandSuggestion] = []
        for skill in discover_skills(self.project_root):
            haystack = f"{skill.name} {skill.description} {skill.scope}".lower()
            if needle and needle not in haystack:
                continue
            items.append(
                CommandSuggestion(
                    f"@skill:{skill.name}",
                    skill.description,
                    id=skill.name,
                    kind="skill-mention",
                    meta=f"{skill.scope} · {skill.path}",
                )
            )
        return items[:30]

    def _mcp_mention_suggestions(self, needle: str = "") -> list[CommandSuggestion]:
        items: list[CommandSuggestion] = []
        for server in discover_mcp_servers(self.project_root):
            haystack = f"{server.name} {server.description} {server.scope} {server.transport}".lower()
            if needle and needle not in haystack:
                continue
            endpoint = f" · {server.endpoint}" if server.endpoint else ""
            items.append(
                CommandSuggestion(
                    f"@mcp:{server.name}",
                    server.description,
                    id=server.name,
                    kind="mcp-mention",
                    meta=f"{server.scope} · {server.transport}{endpoint}",
                )
            )
        return items[:30]

    def _open_thread_picker(self) -> None:
        threads = self.engine.thread_store.list_threads()
        if not threads:
            self._flush(TranscriptCell("event", text="(no previous threads)"))
            return
        items: list[CommandSuggestion] = []
        for thread in threads[:50]:
            thread_id = str(thread.get("thread_id") or "")
            title = str(thread.get("title") or "New thread")
            updated = str(thread.get("updated_at") or "")
            last_text = str(thread.get("last_text") or "").replace("\n", " ")
            description = last_text[:120] if last_text else "no messages"
            marker = "current · " if thread_id == self.state.thread_id else ""
            items.append(
                CommandSuggestion(
                    f"{marker}{title}",
                    description,
                    id=thread_id,
                    kind="thread",
                    meta=f"{short_thread(thread_id)} · {thread.get('turn_count', 0)} turns · {updated}",
                )
            )
        self._open_picker("thread", items)

    def _open_skill_picker(self) -> None:
        items = self._skill_mention_suggestions("")
        if not items:
            self._flush(TranscriptCell("event", text="(no skills discovered)"))
            return
        self._open_picker("skill", items)

    def _open_mcp_picker(self) -> None:
        items = self._mcp_mention_suggestions("")
        if not items:
            self._flush(TranscriptCell("event", text="(no MCP servers declared)"))
            return
        self._open_picker("mcp", items)

    # ------------------------------------------------------------------
    # Submit & commands
    # ------------------------------------------------------------------

    async def submit(self) -> bool:
        text = self.state.composer.strip()
        if not text:
            self._safe_repaint()
            return True
        self._remember_composer_input(text)
        self._reset_history()
        self._close_command_palette()
        if text.startswith("/"):
            if text.partition(" ")[0] == "/image":
                self._set_composer_text("", cursor=0)
                self._handle_command(text)
                self._safe_repaint()
                return True
            should_continue = self._handle_command(text)
            self._set_composer_text("", cursor=0)
            self._safe_repaint()
            return should_continue
        prompt, image_paths, image_numbers = self._message_payload_from_composer(text)
        run_state = self._run_state()
        if run_state is not None and self._run_state_busy_for_ui(run_state):
            run_state.pending_turns.append(PendingTurn(prompt, image_paths))
            self._sync_attached_run_state(run_state)
            self._release_image_numbers(image_numbers)
            self._clear_image_status_tracking()
            self._set_composer_text("", cursor=0)
            self.state.status_message = self._text("queued")
            self._safe_repaint()
            return True
        self._set_composer_text("", cursor=0)
        await self._start_turn(prompt, image_paths=image_paths)
        self._release_image_numbers(image_numbers)
        self._clear_image_status_tracking()
        return True

    def _message_payload_from_composer(self, text: str) -> tuple[str, list[Path], set[int]]:
        image_paths: list[Path] = []
        seen: set[int] = set()
        for match in IMAGE_TOKEN_RE.finditer(text):
            number = int(match.group(1))
            path = self._image_paths_by_number.get(number)
            if path is None or number in seen:
                continue
            seen.add(number)
            image_paths.append(path)
        if image_paths and IMAGE_ONLY_TOKEN_RE.fullmatch(text):
            return self._text("image_only_prompt"), image_paths, seen
        return text, image_paths, seen

    def _handle_command(self, text: str) -> bool:
        command, _, arg = text.partition(" ")
        arg = arg.strip()
        if command == "/help":
            self._flush(TranscriptCell("event", text=HELP_TEXT))
        elif command == "/agents":
            self._open_agent_view()
        elif command == "/bg":
            self._background_current_thread()
        elif command in {"/level", "/model"} and arg:
            self.state.level = arg
            if self.state.thread_id and not self.state.busy:
                self._persist_thread_level(self.state.thread_id, self.state.level)
            self._flush(TranscriptCell("event", text=f"model level set to {self.state.level}"))
        elif command == "/threads":
            self._open_thread_picker()
        elif command == "/status":
            self._show_status()
        elif command == "/skills":
            self._open_skill_picker()
        elif command == "/mcp":
            self._open_mcp_picker()
        elif command == "/image":
            self._attach_clipboard_image_to_composer()
        elif command == "/title" and arg:
            self.state.title = arg
            self._refresh_window_title()
            self._flush(TranscriptCell("event", text=f"title set to {arg}"))
        elif command == "/cancel":
            if not self._interrupt_running_turn():
                self._flush(TranscriptCell("event", text=self._text("no_running_turn")))
        elif command in {"/clear", "/new"}:
            # /new is kept as an unadvertised compatibility alias; the palette
            # exposes only /clear, matching the Textual TUI reset behavior.
            self._clear_to_new_thread()
        elif command == "/goal":
            self._handle_goal(arg)
        elif command == "/quit":
            return False
        else:
            self._flush(TranscriptCell("error", text=f"unknown command: {command}  (try /help)"))
        return True

    def _clear_to_new_thread(self) -> None:
        old_thread_id = self.state.thread_id
        self._finish_live_cells(force=True)
        old_run_state = self._run_state(old_thread_id)
        if old_run_state is not None:
            old_run_state.pending_turns.clear()
        self.state.thread_id = None
        self.state.title = "New thread"
        self.state.goal_enabled = False
        self.state.goal_objective = ""
        self._pending_goal_enable = False
        self._pending_goal_objective = ""
        self._image_paths_by_number.clear()
        self.state.image_token_numbers.clear()
        self._clear_image_status_tracking()
        self.state.level = self.engine.config.runtime.default_level
        self.state.flushed.clear()
        self.state.live.clear()
        self.state.pending_turns = []
        self.state.last_error = None
        self._tool_cells.clear()
        self._assistant_cell = None
        self._reasoning_cell = None
        self._reasoning_flushed_for_current_response = False
        self._refresh_window_title()
        if hasattr(self.renderer, "clear_screen"):
            self.renderer.clear_screen()
        else:
            self.renderer.output.write("\x1b[2J\x1b[3J\x1b[H")
            self.renderer.output.flush()
        self.state.status_message = self._text("new_thread")
        if old_thread_id:
            self._flush(TranscriptCell("event", text=f"cleared view · new thread (was {short_thread(old_thread_id)})"))
        else:
            self._flush(TranscriptCell("event", text="cleared view · new thread"))

    def _status_text(self) -> str:
        level_name = self.state.level or self.engine.config.runtime.default_level
        lines = [f"level: {level_name}"]
        try:
            model = self.engine.config.model_for_level(self.state.level)
            lines.append(f"model: {model.name} -> {model.model}")
        except Exception as exc:
            lines.append(f"model: not configured ({exc})")
        try:
            stats = self.engine.context_stats(self.state.thread_id, self.state.level)
            lines.append(
                "context: "
                f"{stats.percent}% ({stats.used_tokens} / {stats.context_window_tokens}, {stats.source})"
            )
            lines.append(f"compression: trigger {stats.threshold_tokens} · headroom {stats.headroom_tokens}")
        except Exception as exc:
            lines.append(f"context: unavailable ({exc})")
        lines.append(f"thread: {short_thread(self.state.thread_id)}")
        if self.state.title:
            lines.append(f"title: {self.state.title}")
        if self.state.project_path:
            lines.append(f"project: {self.state.project_path}")
        # Cumulative billing total and wall-clock time for the current thread.
        if self.state.thread_id:
            try:
                metadata = self._thread_metadata(self.state.thread_id)
                if self.engine.config.pricing.models:
                    total = billing_total_from_metadata(
                        metadata,
                        preferred_currency=self.engine.config.pricing.currency,
                    )
                    if total is not None:
                        amount, currency = total
                        lines.append(f"cost: {format_billing_total(amount, currency, decimals=4)}")
                created_at = metadata.get("created_at")
                if created_at:
                    created_dt = datetime.fromisoformat(str(created_at))
                    elapsed_s = (datetime.now(UTC) - created_dt).total_seconds()
                    lines.append(f"elapsed: {format_elapsed(elapsed_s)}")
            except Exception:
                pass
        return "\n".join(lines)

    def _show_status(self) -> None:
        self._refresh_context_percent()
        self._flush(TranscriptCell("event", text=self._status_text()))

    def _thread_metadata(self, thread_id: str) -> dict[str, Any]:
        listed: dict[str, Any] = {}
        for item in self.engine.thread_store.list_threads():
            if str(item.get("thread_id") or "") == thread_id:
                listed = dict(item)
                break
        try:
            read_metadata = getattr(self.engine.thread_store, "thread_metadata")
            metadata = dict(read_metadata(thread_id))
        except Exception:
            try:
                metadata = dict(self.engine.thread_store.snapshot(thread_id).metadata)
            except Exception:
                metadata = {}
        return {**listed, **metadata}

    def _thread_metadata_level(self, thread_id: str) -> str | None:
        level = str(self._thread_metadata(thread_id).get("active_level") or "").strip()
        return level or None

    def _level_model_name(self, level: str | None) -> str:
        try:
            config_level = getattr(self.engine.config, "level", None)
            if callable(config_level):
                model_name = getattr(config_level(level), "model", "")
                if model_name:
                    return str(model_name)
            return str(getattr(self.engine.config.model_for_level(level), "name", "") or "")
        except Exception:
            return ""

    def _persist_thread_level(self, thread_id: str, level: str | None) -> None:
        if not thread_id or not level:
            return
        append = getattr(self.engine.thread_store, "append", None)
        if not callable(append):
            return
        metadata = self._thread_metadata(thread_id)
        model = self._level_model_name(level)
        if metadata.get("active_level") == level and metadata.get("active_model") == model:
            return
        append(
            thread_id,
            "thread.level_updated",
            level=level,
            model=model,
            previous_level=metadata.get("active_level"),
            previous_model=metadata.get("active_model"),
        )

    def _current_level_for_thread(self, thread_id: str | None = None) -> str | None:
        if thread_id and thread_id != self.state.thread_id:
            metadata_level = self._thread_metadata_level(thread_id)
            if metadata_level:
                return metadata_level
        return self.state.level or self.engine.config.runtime.default_level

    def _ensure_active_thread(self) -> str:
        if self.state.thread_id is None:
            self.state.thread_id = self.engine.thread_store.create_thread("New thread")
            self._refresh_window_title()
        return self.state.thread_id

    def _materialize_pending_goal_enable(self, thread_id: str) -> None:
        if not self._pending_goal_enable:
            return
        state = self.engine.enable_goal_mode(thread_id, objective=self._pending_goal_objective)
        self.state.goal_enabled = True
        self.state.goal_objective = state.objective or self._pending_goal_objective
        self._pending_goal_enable = False
        self._pending_goal_objective = ""

    def _resume_thread(self, thread_id: str) -> None:
        if not thread_id:
            return
        if thread_id != self.state.thread_id:
            self._detach_live_run_state()
        else:
            self._finish_live_cells(force=True)
        metadata = self._thread_metadata(thread_id)
        self.state.thread_id = thread_id
        self.state.title = str(metadata.get("title") or "New thread")
        self.state.level = self._thread_metadata_level(thread_id) or self.engine.config.runtime.default_level
        self._token_ratio_for_thread(thread_id)
        goal = metadata.get("goal_mode") if isinstance(metadata.get("goal_mode"), dict) else {}
        self.state.goal_enabled = bool(goal.get("enabled")) if isinstance(goal, dict) else False
        self.state.goal_objective = str(goal.get("objective") or "") if isinstance(goal, dict) else ""
        self._pending_goal_enable = False
        self._pending_goal_objective = ""
        self.state.flushed.clear()
        self.state.live.clear()
        run_state = self._run_state(thread_id)
        if run_state is not None and self._run_state_busy_for_ui(run_state):
            self._assistant_cell = run_state.assistant_cell
            self._reasoning_cell = run_state.reasoning_cell
            self._reasoning_flushed_for_current_response = run_state.reasoning_flushed_for_current_response
            self._tool_cells = run_state.tool_cells
            self.state.live = [
                cell
                for cell in [self._reasoning_cell, self._assistant_cell, *self._tool_cells.values()]
                if cell is not None and not cell.done
            ]
        else:
            self._tool_cells.clear()
            self._assistant_cell = None
            self._reasoning_cell = None
            self._reasoning_flushed_for_current_response = False
        self.state.last_error = None
        self._refresh_window_title()
        cells = self._history_cells_for_thread(thread_id)
        self.state.flushed.extend(_retained_flushed_cell(cell) for cell in cells)
        self._trim_flushed_cells()
        if hasattr(self.renderer, "clear_screen"):
            self.renderer.clear_screen()
        else:
            self.renderer._has_frame = False  # type: ignore[attr-defined]
            self.renderer.output.write("[2J[H")
            self.renderer.output.flush()
        flushed_rows = self.renderer.flushed_cell_rows(cells) if hasattr(self.renderer, "flushed_cell_rows") else 0
        self.renderer.flush_cells(cells)
        self.state.status_message = f"{self._text('resumed')} {short_thread(thread_id)}"
        self._sync_attached_run_state(run_state)
        if run_state is None or not run_state.running:
            self.state.status_message = f"{self._text('resumed')} {short_thread(thread_id)}"
        if hasattr(self.renderer, "pad_live_region_to_bottom"):
            self.renderer.pad_live_region_to_bottom(self.state, preceding_rows=flushed_rows)
        self._safe_repaint()

    def _history_cells_for_thread(self, thread_id: str) -> list[TranscriptCell]:
        segment = self.engine.thread_store.read_history_segment(
            thread_id,
            event_types=VISIBLE_HISTORY_EVENT_TYPES | {"turn.started", "turn.completed"},
        )
        timeline = ThreadTimelineState(thread_id)
        timeline.load_history_segment(
            segment.events,
            start_event_id=segment.start_event_id,
            end_event_id=segment.end_event_id,
            has_older=segment.has_more,
        )
        cells = [cell for item in timeline.items if (cell := self._timeline_item_cell(item)) is not None]
        latest = self._thread_metadata(thread_id).get("latest_compaction")
        if segment.has_more and isinstance(latest, dict):
            cells.insert(
                0,
                TranscriptCell(
                    "event",
                    text=_compaction_event_text("history since last compaction", latest.get("text")),
                ),
            )
        return cells

    def _timeline_item_cell(self, item: TimelineItem) -> TranscriptCell | None:
        content = item.content or {}
        if item.kind in {"user", "assistant", "reasoning"}:
            text_value = content.get("text")
            if isinstance(text_value, list):
                text = "".join(str(part) for part in text_value)
            else:
                text = str(text_value or "")
            return TranscriptCell(item.kind, text=text) if text else None
        if item.kind == "tool_result":
            payload = content.get("payload") if isinstance(content.get("payload"), dict) else {}
            return TranscriptCell("tool", payload=dict(payload))
        if item.kind == "tool_call":
            call = content.get("call") if isinstance(content.get("call"), dict) else {}
            status = str(content.get("status") or "done")
            return TranscriptCell("tool", status="running" if status == "running" else "done", call=dict(call))
        if item.kind == "image":
            attachment = content.get("attachment") if isinstance(content.get("attachment"), dict) else {}
            path = attachment.get("path") or attachment.get("original_path") or attachment.get("source_path") or "image"
            return TranscriptCell("image", text=f"image attached: {path}")
        if item.kind == "compaction":
            return TranscriptCell("event", text=_compaction_event_text("conversation compacted", content.get("text")))
        if item.kind == "warning":
            event = content.get("event") if isinstance(content.get("event"), dict) else {}
            message = str(event.get("message") or event.get("text") or "warning")
            return TranscriptCell("event", text=message)
        if item.kind == "stream_retry":
            event = content.get("event") if isinstance(content.get("event"), dict) else {}
            attempt = event.get("attempt") or "?"
            message = str(event.get("message") or "model stream retry")
            return TranscriptCell("event", text=f"retry {attempt}: {message}")
        if item.kind == "error":
            event = content.get("event") if isinstance(content.get("event"), dict) else {}
            message = str(event.get("message") or event.get("error") or "turn error")
            return TranscriptCell("error", text=message)
        return None

    def _handle_goal(self, arg: str) -> None:
        sub = arg.split(None, 1)
        op = (sub[0] if sub else "status").lower()
        rest = sub[1] if len(sub) > 1 else ""
        if op not in {"enable", "disable", "reset", "status"}:
            self._flush(
                TranscriptCell("error", text="usage: /goal enable [objective] | disable | reset | status")
            )
            return
        if not self.state.thread_id:
            if op == "enable":
                self._pending_goal_enable = True
                self._pending_goal_objective = rest.strip()
                self.state.goal_enabled = True
                self.state.goal_objective = self._pending_goal_objective
                obj = self._pending_goal_objective or "—"
                self._flush(TranscriptCell("event", text=f"goal mode enabled for next message · objective: {obj}"))
                return
            if op == "disable":
                self._pending_goal_enable = False
                self._pending_goal_objective = ""
                self.state.goal_enabled = False
                self.state.goal_objective = ""
                self._flush(TranscriptCell("event", text="goal mode disabled"))
                return
            if op == "status":
                if self._pending_goal_enable:
                    obj = self._pending_goal_objective or "—"
                    self._flush(
                        TranscriptCell("event", text=f"goal mode: enabled (pending first message)\nobjective: {obj}")
                    )
                else:
                    self._flush(TranscriptCell("event", text="goal mode: disabled (no active thread)"))
                return
            self._flush(TranscriptCell("error", text="/goal reset requires an active thread — send a message first"))
            return
        try:
            if op == "enable":
                state = self.engine.enable_goal_mode(self.state.thread_id, objective=rest)
                self.state.goal_enabled = True
                self.state.goal_objective = state.objective or ""
                obj = state.objective or "—"
                self._flush(TranscriptCell("event", text=f"goal mode enabled · objective: {obj}"))
            elif op == "disable":
                self.engine.disable_goal_mode(self.state.thread_id)
                self.state.goal_enabled = False
                self._flush(TranscriptCell("event", text="goal mode disabled"))
            elif op == "reset":
                state = self.engine.reset_goal_files(self.state.thread_id, objective=rest)
                self.state.goal_objective = state.objective or ""
                self._flush(TranscriptCell("event", text="goal files reset"))
            else:  # status
                state = self.engine.goal_state(self.state.thread_id)
                if state is None:
                    self._flush(TranscriptCell("event", text="goal mode: disabled (no state yet)"))
                else:
                    status = "enabled" if state.status == "enabled" else "disabled"
                    obj = state.objective or "—"
                    self._flush(TranscriptCell("event", text=f"goal mode: {status}\nobjective: {obj}"))
        except Exception as exc:
            self._flush(TranscriptCell("error", text=f"/goal {op} failed: {exc}"))

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    async def _start_turn(self, text: str, *, image_paths: list[Path] | None = None) -> None:
        thread_id = self._ensure_active_thread()
        level = self._current_level_for_thread(thread_id)
        self.state.level = level
        self._persist_thread_level(thread_id, level)
        self._flush(TranscriptCell("user", text=text))
        run_state = self._thread_runs.setdefault(thread_id, ThreadRunState(thread_id=thread_id))
        run_state.reset_for_turn()
        run_state.token_ratio = self._token_ratio_for_thread(thread_id)
        run_state.cancel_event = asyncio.Event()
        run_state.started_at = monotonic()
        run_state.status_message = "running"
        run_state.last_error = None
        run_state.terminal_status = "working"
        run_state.assistant_cell = None
        run_state.reasoning_cell = None
        run_state.reasoning_flushed_for_current_response = False
        run_state.tool_cells = {}
        self._clear_quit_confirmation()
        self._clear_interrupt_confirmation()
        self.state.busy = True
        self.state.status_message = "running"
        self.state.last_error = None
        self.state.turn_token_rate = None
        self.state.turn_token_rate_frozen = False
        self._reasoning_flushed_for_current_response = False
        self._assistant_cell = None
        self._reasoning_cell = None
        self._tool_cells = run_state.tool_cells
        self._apply_window_title()
        run_state.task = asyncio.create_task(
            self._run_turn(thread_id, text, image_paths=list(image_paths or []))
        )
        self._safe_repaint()

    async def _run_turn(self, thread_id: str, text: str, *, image_paths: list[Path]) -> None:
        run_state = self._thread_runs[thread_id]
        try:
            if self._is_attached_thread(thread_id):
                self._sync_attached_run_state(run_state)
            self._materialize_pending_goal_enable(thread_id)
            async for event in self.engine.run_turn(
                user_text=text,
                thread_id=thread_id,
                level=self._current_level_for_thread(thread_id),
                image_paths=image_paths,
                cancel_event=run_state.cancel_event,
            ):
                if self._is_attached_thread(thread_id):
                    self._handle_event(event)
                    self._capture_attached_run_state(run_state)
                    self._safe_repaint()
                else:
                    self._handle_background_event(run_state, event)
        except Exception as exc:
            run_state.last_error = str(exc) or repr(exc)
            run_state.terminal_status = "failed"
            run_state.engine_finished = False
            run_state.assistant_display_queue = ""
            run_state.pending_finish_after_drain = False
            run_state.completion_notification_pending = False
            if self._is_attached_thread(thread_id):
                self.state.last_error = run_state.last_error
                self._flush(TranscriptCell("error", text=run_state.last_error))
        finally:
            if self._is_attached_thread(thread_id):
                self._sync_attached_run_state(run_state)
                self._finish_live_cells()
                self._capture_attached_run_state(run_state)
            self._persist_pending_agent_view_join(thread_id)
            if run_state.display_pending:
                run_state.engine_finished = True
                run_state.status_message = self._text("writing_answer")
                if self._is_attached_thread(thread_id):
                    self._sync_attached_run_state(run_state)
                    self._apply_window_title()
                    self._safe_repaint()
                return
            self._complete_finished_run_display(run_state)
            if self._is_attached_thread(thread_id):
                self._safe_repaint()

    async def _start_turn_for_thread(
        self,
        thread_id: str,
        text: str,
        *,
        image_paths: list[Path] | None = None,
    ) -> None:
        if self._is_attached_thread(thread_id):
            await self._start_turn(text, image_paths=image_paths)
            return
        run_state = self._thread_runs.setdefault(thread_id, ThreadRunState(thread_id=thread_id))
        run_state.reset_for_turn()
        run_state.token_ratio = self._token_ratio_for_thread(thread_id)
        run_state.cancel_event = asyncio.Event()
        run_state.started_at = monotonic()
        run_state.status_message = "running"
        run_state.last_error = None
        run_state.terminal_status = "working"
        run_state.assistant_cell = None
        run_state.reasoning_cell = None
        run_state.reasoning_flushed_for_current_response = False
        run_state.tool_cells = {}
        run_state.task = asyncio.create_task(self._run_turn(thread_id, text, image_paths=list(image_paths or [])))

    def _run_state_for_cell(self, cell: TranscriptCell | None = None) -> ThreadRunState | None:
        attached = self._run_state()
        if cell is None:
            return attached
        for run_state in self._thread_runs.values():
            if (
                cell is run_state.assistant_cell
                or cell is run_state.reasoning_cell
                or cell in run_state.tool_cells.values()
            ):
                return run_state
        return attached

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        previous_thread_id = self.state.thread_id
        event_thread_id = event.get("thread_id")
        if event_thread_id:
            self.state.thread_id = str(event_thread_id)
            if self.state.thread_id != previous_thread_id:
                self._refresh_window_title()
        run_state = self._run_state_for_event(event)
        if run_state is not None:
            if event_type == "turn.started":
                run_state.terminal_status = "working"
                run_state.status_message = "running"
            elif event_type == "turn.error":
                run_state.terminal_status = "failed"
                run_state.last_error = str(event.get("message") or "turn error")
                run_state.assistant_display_queue = ""
                run_state.pending_finish_after_drain = False
                run_state.engine_finished = False
                run_state.completion_notification_pending = False
            elif event_type == "turn.interrupted":
                run_state.terminal_status = "interrupted"
                run_state.status_message = self._text("interrupted")
                run_state.assistant_display_queue = ""
                run_state.pending_finish_after_drain = False
                run_state.engine_finished = False
                run_state.completion_notification_pending = False
            elif event_type == "turn.completed":
                run_state.terminal_status = "completed"
                run_state.status_message = "ready"
        if event_type == "thread.title":
            self.state.title = str(event.get("title") or self.state.title)
            self._refresh_window_title()
        elif event_type == "thread.goal_mode_updated":
            self.state.goal_enabled = bool(event.get("enabled"))
            self.state.goal_objective = str(event.get("objective") or self.state.goal_objective)
        elif event_type == "assistant.delta":
            self.state.status_message = self._text("writing_answer")
            # As soon as the assistant starts writing, flush any in-flight
            # reasoning cell into scrollback so its breath animation stops
            # immediately instead of waiting for a later reasoning_completed
            # or model.response event (which may arrive much later).
            if self._reasoning_cell is not None:
                self._finish_reasoning_cell("")
            self._append_assistant(str(event.get("text") or ""))
        elif event_type == "assistant.reasoning_delta":
            self.state.status_message = self._text("thinking_status")
            self._append_reasoning(str(event.get("text") or ""))
        elif event_type == "assistant.reasoning_completed":
            self.state.status_message = self._text("reading")
            self._finish_reasoning_cell(str(event.get("text") or ""))
        elif event_type == "assistant.reasoning_absent":
            self.state.status_message = self._text("reading")
            self._drop_reasoning_cell()
        elif event_type in {"assistant.final_response_started", "assistant.response_with_tools"}:
            self.state.status_message = self._text("writing_answer")
            self._finish_reasoning_cell("")
            assistant_text = str(event.get("assistant_text") or "")
            if assistant_text:
                self._finish_assistant_cell(assistant_text)
        elif event_type == "model.response":
            self.state.status_message = self._text("reading")
            self._handle_model_response_event(event)
        elif event_type == "tool.delta":
            self.state.status_message = self._text("writing_script")
            self._observe_tool_delta(event, run_state=run_state)
        elif event_type == "tool.started":
            self.state.status_message = self._text("running_python")
            self._freeze_token_rate(run_state)
            self._finish_text_cells(force=True)
            call = event.get("call") if isinstance(event.get("call"), dict) else {}
            key = str(call.get("call_id") or event.get("tool_call_index") or len(self._tool_cells))
            cell = TranscriptCell("tool", status="running", call=call)
            self._tool_cells[key] = cell
            self.state.live.append(cell)
        elif event_type == "tool.partial":
            self.state.status_message = self._text("running_python")
            self._update_tool(event, running=True)
        elif event_type == "tool.output":
            self.state.status_message = self._text("working")
            self._update_tool(event, running=False)
            # A completed tool output is followed by a fresh model response, so
            # provider-only reasoning in that next response must still render.
            self._reasoning_flushed_for_current_response = False
        elif event_type == "image.attachment":
            attachment = event.get("attachment") or {}
            path = attachment.get("path") or attachment.get("original_path") or "image"
            self._flush(TranscriptCell("image", text=f"image attached: {path}"))
        elif event_type == "compaction.started":
            self.state.status_message = self._text("compacting")
            self._flush(TranscriptCell("event", text="compaction started"))
        elif event_type == "compaction.completed":
            self.state.status_message = self._text("working")
            self._flush(TranscriptCell("event", text=_compaction_event_text("conversation compacted", event.get("text"))))
        elif event_type == "turn.error":
            self.state.last_error = str(event.get("message") or "turn error")
            self._flush(TranscriptCell("error", text=self.state.last_error))
        elif event_type == "turn.interrupted":
            self._flush(TranscriptCell("event", text="turn interrupted"))
        elif event_type == "turn.completed":
            self._finish_live_cells()
            if run_state is not None and run_state.display_pending:
                run_state.completion_notification_pending = True
            else:
                self._notify_turn_completed()
            self._refresh_window_title()

    def _handle_background_event(self, run_state: ThreadRunState, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "thread.title":
            self._refresh_agent_view_rows()
        elif event_type == "turn.error":
            run_state.last_error = str(event.get("message") or "turn error")
            run_state.terminal_status = "failed"
            run_state.assistant_display_queue = ""
            run_state.pending_finish_after_drain = False
            run_state.engine_finished = False
            run_state.completion_notification_pending = False
        elif event_type == "turn.interrupted":
            run_state.terminal_status = "interrupted"
            run_state.assistant_display_queue = ""
            run_state.pending_finish_after_drain = False
            run_state.engine_finished = False
            run_state.completion_notification_pending = False
        elif event_type == "turn.completed":
            run_state.terminal_status = "completed"
            self._notify_turn_completed()

    def _notify_turn_completed(self) -> None:
        notification_config = getattr(getattr(self.engine.config, "ui", None), "completion_notification", None)
        if notification_config is not None:
            if not getattr(notification_config, "enabled", True):
                return
            if not getattr(notification_config, "bell", True):
                return
        play_terminal_buzzer()

    def _current_thread_title(self) -> str:
        fallback = self.state.title if self.state.title not in DEFAULT_THREAD_TITLES else self._text("new_thread")
        if not self.state.thread_id:
            return fallback
        metadata_available = False
        try:
            read_metadata = getattr(self.engine.thread_store, "thread_metadata")
            metadata = dict(read_metadata(self.state.thread_id))
            metadata_available = True
        except Exception:
            metadata = {}
        title = str(metadata.get("title") or "").strip()
        if title and title not in DEFAULT_THREAD_TITLES:
            return title
        if metadata_available:
            return fallback
        try:
            digest = self.engine.thread_store.thread_digest(self.state.thread_id)
        except Exception:
            return fallback
        title = str(digest.get("title") or "").strip()
        if not title or title in DEFAULT_THREAD_TITLES:
            return fallback
        return title

    def _window_title_waiting_for_generated_title(self) -> bool:
        if not self.state.thread_id:
            return False
        title = self._window_title_thread_title.strip()
        return not title or title == self._text("new_thread") or title in DEFAULT_THREAD_TITLES

    def _refresh_window_title(self) -> None:
        self._window_title_thread_title = self._current_thread_title()
        self._apply_window_title()

    def _apply_window_title(self) -> None:
        if self.state.busy and self._window_title_waiting_for_generated_title():
            # Title generation writes thread metadata from a side task before the
            # engine emits its final thread.title event. While the visible title
            # is still a placeholder, re-read it on spinner ticks so the
            # terminal title can switch as soon as metadata is available.
            self._window_title_thread_title = self._current_thread_title()
        title = self._window_title_thread_title or self._current_thread_title()
        if self.state.busy:
            spinner = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
            title = f"{spinner} {title}"
        title = sanitized_window_title(title)
        if title == self._last_window_title:
            return
        self._last_window_title = title
        write_window_title(title)

    def _text(self, key: str) -> str:
        return tr(self.language, key)

    def _fmt(self, key: str, **values: object) -> str:
        return self._text(key).format(**values)

    # ------------------------------------------------------------------
    # Cell bookkeeping
    # ------------------------------------------------------------------

    def _observe_stream_text(self, text: str, *, run_state: ThreadRunState | None = None) -> None:
        if not text:
            return
        run_state = run_state or self._run_state()
        if run_state is not None:
            now = monotonic()
            self._resume_token_rate(run_state, now=now)
            run_state.rate_estimator.observe(text, now=now)
            if run_state.last_animation_tick_at is None:
                run_state.last_animation_tick_at = now

    def _observe_tool_delta(self, event: dict[str, Any], *, run_state: ThreadRunState | None = None) -> None:
        run_state = run_state or self._run_state()
        tool_call = event.get("tool_call")
        text = tool_delta_visible_text(tool_call)
        if run_state is not None:
            key = tool_call_stream_key(tool_call, fallback=event.get("tool_call_index") or "0")
            name = tool_call_name(tool_call)
            if name and key not in run_state.observed_tool_call_name_keys:
                run_state.observed_tool_call_name_keys.add(key)
                text = name + text
        self._observe_stream_text(text, run_state=run_state)

    def _advance_streaming_display(self) -> bool:
        """Advance throughput-driven animation and queued assistant text.

        Provider chunks can be uneven even when the model's average throughput is
        steady.  This tick converts the current sliding-window estimate into a
        fractional animation phase and a character display budget so the live UI
        is paced by average speed instead of chunk boundaries.
        """

        run_state = self._run_state()
        if run_state is None:
            return False
        now = monotonic()
        previous = run_state.last_animation_tick_at
        run_state.last_animation_tick_at = now
        if previous is None:
            return bool(run_state.assistant_display_queue)
        dt = max(0.0, min(now - previous, 0.5))
        if dt <= 0.0:
            return False

        changed = False
        cps = run_state.rate_estimator.current_cps(now=now)
        if cps is None and run_state.rate_estimator.first_output_at is not None:
            cps = DEFAULT_STREAM_CHARS_PER_SECOND
        if cps and cps > 0:
            phase_delta = cps * dt / BREATH_CHARS_PER_PHASE
            for cell in (run_state.reasoning_cell, run_state.assistant_cell):
                if cell is not None and cell.status == "streaming":
                    cell.animation_phase = (cell.animation_phase or 0.0) + phase_delta
                    changed = True

        queue = run_state.assistant_display_queue
        if queue:
            display_cps = run_state.rate_estimator.display_cps(now=now, backlog_chars=len(queue))
            run_state.assistant_display_credit += max(0.0, display_cps * dt)
            count = min(len(queue), int(run_state.assistant_display_credit))
            if count > 0:
                chunk = queue[:count]
                run_state.assistant_display_queue = queue[count:]
                run_state.assistant_display_credit -= count
                self._append_assistant_now(chunk)
                changed = True

        if not run_state.assistant_display_queue and run_state.pending_finish_after_drain:
            run_state.pending_finish_after_drain = False
            self._finish_assistant_cell_now(run_state.pending_finish_assistant_text or "")
            run_state.pending_finish_assistant_text = None
            changed = True

        if run_state.engine_finished and not run_state.display_pending:
            self._complete_finished_run_display(run_state)
            changed = True
        return changed

    def _complete_finished_run_display(self, run_state: ThreadRunState) -> None:
        """Finish a turn whose engine task ended while text was still draining."""

        should_notify = run_state.completion_notification_pending
        run_state.completion_notification_pending = False
        run_state.engine_finished = False
        run_state.started_at = None
        run_state.status_message = "ready"
        if self._is_attached_thread(run_state.thread_id):
            self.state.busy = False
            self.state.status_message = "ready"
            self.state.turn_elapsed_s = None
            self.state.turn_token_rate = None
            self.state.turn_token_rate_frozen = False
            self._apply_window_title()
        if should_notify:
            self._notify_turn_completed()
        if run_state.pending_turns:
            next_turn = run_state.pending_turns.pop(0)
            asyncio.create_task(
                self._start_turn_for_thread(
                    run_state.thread_id,
                    next_turn.text,
                    image_paths=next_turn.image_paths,
                )
            )
        elif not run_state.running:
            self._thread_runs.pop(run_state.thread_id, None)

    def _ensure_assistant_cell(self) -> TranscriptCell:
        if self._assistant_cell is None or self._assistant_cell.done:
            self._assistant_cell = TranscriptCell("assistant", status="streaming", animation_phase=0.0)
            self.state.live.append(self._assistant_cell)
        run_state = self._run_state()
        if run_state is not None:
            run_state.assistant_cell = self._assistant_cell
        return self._assistant_cell

    def _append_assistant(self, text: str) -> None:
        if not text:
            return
        run_state = self._run_state()
        if run_state is None:
            self._append_assistant_now(text)
            return
        self._observe_stream_text(text)
        self._ensure_assistant_cell()
        run_state.assistant_display_queue += text

    def _append_assistant_now(self, text: str) -> None:
        if not text:
            return
        cell = self._ensure_assistant_cell()
        cell.text += text
        cell.chars_streamed += len(text)
        run_state = self._run_state()
        if run_state is not None:
            run_state.assistant_cell = cell

    def _append_reasoning(self, text: str) -> None:
        if not text:
            return
        self._observe_stream_text(text)
        if self._reasoning_cell is None or self._reasoning_cell.done:
            self._reasoning_cell = TranscriptCell("reasoning", status="streaming", animation_phase=0.0)
            self.state.live.append(self._reasoning_cell)
            self._reasoning_flushed_for_current_response = False
        self._reasoning_cell.text += text
        self._reasoning_cell.chars_streamed += len(text)
        run_state = self._run_state()
        if run_state is not None:
            run_state.reasoning_cell = self._reasoning_cell
            run_state.reasoning_flushed_for_current_response = self._reasoning_flushed_for_current_response

    def _update_tool(self, event: dict[str, Any], *, running: bool) -> None:
        call = event.get("call") if isinstance(event.get("call"), dict) else {}
        key = str(call.get("call_id") or event.get("tool_call_index") or "0")
        cell = self._tool_cells.get(key)
        if cell is None:
            cell = TranscriptCell("tool", status="running", call=call)
            self._tool_cells[key] = cell
            self.state.live.append(cell)
            run_state = self._run_state()
            if run_state is not None:
                run_state.tool_cells = self._tool_cells
        elif call and not cell.call:
            cell.call = call
        payload = tool_payload_from_event(event)
        if payload is not None:
            if "helper_calls" not in payload:
                payload["helper_calls"] = extract_runtime_helper_calls(self._tool_call_code(cell.call))
            cell.payload = payload
        if not running:
            cell.status = "done"
            cell.finished_at = monotonic()
            self._flush(cell)
            self._tool_cells.pop(key, None)

    @staticmethod
    def _tool_call_code(call: dict[str, Any] | None) -> str:
        if not isinstance(call, dict):
            return ""
        try:
            import json

            args = json.loads(str(call.get("arguments") or "{}"))
        except Exception:
            return ""
        return str(args.get("code") or "")

    def _finish_reasoning_cell(self, text: str = "") -> None:
        cell = self._reasoning_cell
        if cell is None:
            if text.strip() and not self._reasoning_flushed_for_current_response:
                self._flush(TranscriptCell("reasoning", text=text.strip()))
                self._reasoning_flushed_for_current_response = True
            return
        if text.strip():
            cell.text = text.strip()
        if not cell.text.strip():
            self._drop_reasoning_cell()
            return
        if not cell.done:
            cell.status = "done"
            cell.finished_at = monotonic()
        self._flush(cell)
        self._reasoning_flushed_for_current_response = True
        self._reasoning_cell = None

    def _drop_reasoning_cell(self) -> None:
        cell = self._reasoning_cell
        if cell is not None and cell in self.state.live:
            self.state.live.remove(cell)
        self._reasoning_cell = None

    def _finish_assistant_cell(self, text: str = "", *, force: bool = False) -> None:
        run_state = self._run_state()
        if run_state is not None and run_state.assistant_display_queue and force:
            queued = run_state.assistant_display_queue
            run_state.assistant_display_queue = ""
            run_state.assistant_display_credit = 0.0
            if not text:
                self._append_assistant_now(queued)
        if run_state is not None and run_state.assistant_display_queue:
            if text:
                displayed = self._assistant_cell.text if self._assistant_cell is not None else ""
                queued = run_state.assistant_display_queue
                # ``text`` is the canonical final answer from the response.  Keep
                # draining whatever has not yet reached the terminal instead of
                # replacing the live cell with a large completed chunk.
                if text.startswith(displayed):
                    remainder = text[len(displayed):]
                    if remainder != queued:
                        run_state.assistant_display_queue = remainder
                    run_state.pending_finish_assistant_text = text
                else:
                    run_state.pending_finish_assistant_text = text
            run_state.pending_finish_after_drain = True
            return
        self._finish_assistant_cell_now(text)

    def _finish_assistant_cell_now(self, text: str = "") -> None:
        run_state = self._run_state()
        if run_state is not None:
            run_state.pending_finish_after_drain = False
            run_state.pending_finish_assistant_text = None
        cell = self._assistant_cell
        if cell is None:
            if text:
                self._flush(TranscriptCell("assistant", text=text))
            if run_state is not None:
                run_state.assistant_cell = None
            return
        if text:
            cell.text = text
        if not cell.done:
            cell.status = "done"
            cell.finished_at = monotonic()
        self._flush(cell)
        self._assistant_cell = None
        if run_state is not None:
            run_state.assistant_cell = None

    def _handle_model_response_event(self, event: dict[str, Any]) -> None:
        run_state = self._run_state_for_event(event)
        reasoning_text = str(event.get("reasoning_text") or "")
        if reasoning_text:
            self._finish_reasoning_cell(reasoning_text)
        else:
            self._drop_reasoning_cell()
        response = event.get("response")
        output = list(getattr(response, "output", []) or [])
        output_tokens = usage_output_tokens(
            getattr(response, "usage", {}) if response is not None else {},
            reasoning_visible=bool(reasoning_text),
        )
        if output_tokens:
            ratio = self._token_ratio_for_thread(str(event.get("thread_id") or self.state.thread_id or ""))
            ratio.observe_response(
                visible_units=model_response_visible_units(output, reasoning_text=reasoning_text),
                output_tokens=output_tokens,
            )
            if run_state is not None:
                run_state.token_ratio = ratio
        has_tool_call = any(isinstance(item, dict) and item.get("type") == "function_call" for item in output)
        if has_tool_call:
            self._freeze_token_rate(run_state)
        else:
            if run_state is not None and run_state.display_pending:
                self._hold_token_rate(run_state)
            else:
                self._freeze_token_rate(run_state)
            self._finish_assistant_cell()

    def _finish_text_cells(self, *, force: bool = False) -> None:
        self._finish_reasoning_cell("")
        self._finish_assistant_cell("", force=force)

    def _finish_live_cells(self, *, force: bool = False) -> None:
        self._finish_text_cells(force=force)
        for key, cell in list(self._tool_cells.items()):
            if not cell.done:
                cell.status = "done"
                cell.finished_at = monotonic()
                self._flush(cell)
            self._tool_cells.pop(key, None)

    def _remember_flushed_cell(self, cell: TranscriptCell) -> None:
        self.state.flushed.append(_retained_flushed_cell(cell))
        self._trim_flushed_cells()

    def _trim_flushed_cells(self) -> None:
        overflow = len(self.state.flushed) - TUI2_FLUSHED_CELLS_MAX
        if overflow > 0:
            del self.state.flushed[:overflow]

    def _flush(self, cell: TranscriptCell) -> None:
        if cell in self.state.live:
            self.state.live.remove(cell)
        cell.status = "done" if cell.status in {"running", "streaming"} else cell.status
        cell.finished_at = cell.finished_at or monotonic()
        self.renderer.flush_cell(cell)
        self._remember_flushed_cell(cell)

    # ------------------------------------------------------------------
    # Render guard
    # ------------------------------------------------------------------

    def _refresh_context_percent(self) -> None:
        try:
            stats = self.engine.context_stats(self.state.thread_id, self.state.level)
        except Exception:
            self.state.context_percent = None
            return
        self.state.context_percent = stats.percent

    def _safe_repaint(self) -> None:
        """Repaint without letting UI bugs unwind the engine async generator."""

        self._sync_attached_run_state()
        self._refresh_context_percent()
        try:
            self.renderer.repaint(self.state)
        except Exception as exc:
            self.state.last_error = f"tui2 render error: {exc}"
            self.state.status_message = "render error"
