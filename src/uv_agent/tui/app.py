from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from rich.markdown import Markdown
from rich.markup import escape, render as render_markup
from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.reactive import reactive
from textual.selection import Selection
from textual.strip import Strip
from textual.widgets import Input, OptionList, Static, TextArea
from textual.worker import Worker
from textual.widgets._option_list import Option

from uv_agent.app_factory import create_engine
from uv_agent.config import ConfigError, config_sources, load_raw_config, redact_config
from uv_agent.environment import detect_user_language, host_environment_line
from uv_agent.errors import error_markup, format_error
from uv_agent.i18n import command_description, tr
from uv_agent.mcp_config import discover_mcp_servers
from uv_agent.paths import project_state_dir, uv_agent_home
from uv_agent.skills import discover_skills
from uv_agent.tui.formatting import (
    format_tokens,
    parse_tool_payload,
    short_thread,
    tool_result_markup,
    tool_timeline_markup,
)


COMPOSER_COLLAPSED_HEIGHT = 5
COMPOSER_BOTTOM_RESERVED_ROWS = 2


@dataclass(frozen=True)
class PickerItem:
    id: str
    title: str
    description: str = ""
    meta: str = ""


class PickerOptionList(OptionList):
    ALLOW_SELECT = True


class FullscreenPanel(ModalScreen[str | None]):
    """Scrollable full-screen panel/picker."""

    CSS = """
    FullscreenPanel {
        align: center middle;
        background: #05070acc;
    }

    #panel-shell {
        width: 92%;
        height: 88%;
        max-width: 120;
        border: round #3a4a60;
        background: #0c1118;
        padding: 1 2;
    }

    #panel-header {
        height: 1;
        color: #dce7f3;
        text-style: bold;
    }

    #panel-subtitle {
        height: 1;
        color: #8fa2b8;
    }

    #panel-filter {
        height: 3;
        margin: 1 0 0 0;
        border: tall #263649;
        background: #0f1721;
        color: #e9eef5;
    }

    #panel-content {
        height: 1fr;
        margin: 1 0 0 0;
        border: tall #1f2b3a;
        background: #0a0f15;
        padding: 0 1;
    }

    #panel-body {
        height: 1fr;
        margin: 1 0 0 0;
        border: tall #1f2b3a;
        background: #0a0f15;
        padding: 1 1;
    }

    #panel-footer {
        height: 1;
        margin: 1 0 0 0;
        color: #7b8796;
    }

    OptionList {
        height: 1fr;
        border: none;
        background: #0a0f15;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_panel", "Close", priority=True, show=False),
        Binding("up", "cursor_up", "Up", priority=True, show=False),
        Binding("down", "cursor_down", "Down", priority=True, show=False),
        Binding("pageup", "page_up", "Page up", priority=True, show=False),
        Binding("pagedown", "page_down", "Page down", priority=True, show=False),
        Binding("enter", "select_or_close", "Select", priority=True, show=False),
    ]

    def __init__(
        self,
        *,
        title: str,
        body: str = "",
        items: list[PickerItem] | None = None,
        subtitle: str = "",
        initial_filter: str = "",
        mention_kind: str | None = None,
        mention_items: Callable[[str], tuple[str, list[PickerItem], str]] | None = None,
    ) -> None:
        super().__init__()
        self.panel_title = title
        self.body = body
        self.items = items or []
        self.subtitle = subtitle
        self.initial_filter = initial_filter.strip()
        self.mention_kind = mention_kind
        self.mention_items = mention_items
        self._selected_mention_kind: str | None = None
        self._filtered = list(self.items)
        self._option_ids: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="panel-shell"):
            yield Static(self.panel_title, id="panel-header")
            yield Static(self.subtitle, id="panel-subtitle")
            if self.items or self.mention_kind:
                yield Input(placeholder=getattr(self.app, "_text", lambda key: key)("filter"), id="panel-filter")
                yield PickerOptionList(id="panel-content", compact=False)
            else:
                yield VerticalScroll(Static(self.body, markup=True), id="panel-body")
            yield Static(getattr(self.app, "_text", lambda key: key)("panel_footer"), id="panel-footer")

    def on_mount(self) -> None:
        if self.items or self.mention_kind:
            filter_input = self.query_one("#panel-filter", Input)
            if self.initial_filter:
                filter_input.value = self.initial_filter
                self._apply_filter(self.initial_filter)
            else:
                self._refresh_options()
            self.query_one("#panel-content", OptionList).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "panel-filter":
            return
        self._apply_filter(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "panel-filter":
            return
        event.stop()
        self.action_select_or_close()

    def on_key(self, event: events.Key) -> None:
        actions = {
            "up": self.action_cursor_up,
            "down": self.action_cursor_down,
            "pageup": self.action_page_up,
            "page_up": self.action_page_up,
            "pagedown": self.action_page_down,
            "page_down": self.action_page_down,
            "enter": self.action_select_or_close,
        }
        action = actions.get(event.key)
        if action is not None:
            event.stop()
            try:
                action()
            except SkipAction:
                pass
            return
        if not self.items and not self.mention_kind:
            return
        filter_input = self.query_one("#panel-filter", Input)
        if self.mention_kind == "file" and (
            event.character == "@" or event.key in {"@", "at", "commercial_at"}
        ):
            event.stop()
            self._switch_mention_kind("thread", filter_value="@")
            return
        if event.key == "backspace":
            event.stop()
            if self.mention_kind == "thread" and filter_input.value == "@":
                self._switch_mention_kind("file", filter_value="")
                return
            filter_input.value = filter_input.value[:-1]
            self._apply_filter(filter_input.value)
            return
        if event.key in {"ctrl+u", "ctrl+w"}:
            event.stop()
            filter_input.value = ""
            self._apply_filter("")
            return
        if event.character and not event.key.startswith("ctrl+"):
            event.stop()
            filter_input.value += event.character
            self._apply_filter(filter_input.value)

    def _apply_filter(self, value: str) -> None:
        query = value.casefold().strip()
        if self.mention_kind == "thread" and query.startswith("@"):
            query = query[1:].strip()
        if not query:
            self._filtered = list(self.items)
        else:
            prefix_matches = [
                item for item in self.items if item.title.casefold().lstrip("/").startswith(query.lstrip("/"))
            ]
            contains_matches = [
                item
                for item in self.items
                if item not in prefix_matches
                and query in (item.title + " " + item.description + " " + item.meta).casefold()
            ]
            self._filtered = prefix_matches + contains_matches
        self._refresh_options()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id:
            self.dismiss(self._option_ids.get(event.option_id, event.option_id))

    def action_dismiss_panel(self) -> None:
        self.dismiss(None)

    def action_cursor_up(self) -> None:
        if self.items:
            self.query_one("#panel-content", OptionList).action_cursor_up()
            return
        self.query_one("#panel-body", VerticalScroll).action_scroll_up()

    def action_cursor_down(self) -> None:
        if self.items:
            self.query_one("#panel-content", OptionList).action_cursor_down()
            return
        self.query_one("#panel-body", VerticalScroll).action_scroll_down()

    def action_page_up(self) -> None:
        if self.items:
            self.query_one("#panel-content", OptionList).action_page_up()
            return
        self.query_one("#panel-body", VerticalScroll).action_page_up()

    def action_page_down(self) -> None:
        if self.items:
            self.query_one("#panel-content", OptionList).action_page_down()
            return
        self.query_one("#panel-body", VerticalScroll).action_page_down()

    def action_select_or_close(self) -> None:
        if not self.items:
            self.dismiss(None)
            return
        option_list = self.query_one("#panel-content", OptionList)
        highlighted = option_list.highlighted
        if highlighted is None or highlighted < 0 or highlighted >= option_list.option_count:
            return
        option = option_list.get_option_at_index(highlighted)
        if option.id:
            self._selected_mention_kind = self.mention_kind
            self.dismiss(self._option_ids.get(option.id, option.id))

    def _refresh_options(self) -> None:
        self._option_ids = {}
        options = []
        for index, item in enumerate(self._filtered):
            option_id = f"item_{index}"
            self._option_ids[option_id] = item.id
            options.append(
                Option(
                    f"[bold cyan]{escape(item.title)}[/bold cyan]"
                    + (f"\n[dim]{escape(item.description)}[/dim]" if item.description else "")
                    + (f"\n[dim]{escape(item.meta)}[/dim]" if item.meta else ""),
                    id=option_id,
                )
            )
        if not options:
            text = getattr(self.app, "_text", lambda key: key)
            if self.mention_kind == "file":
                label = text("no_mention_files")
            elif self.mention_kind == "thread":
                label = text("no_threads")
            else:
                label = text("no_matches")
            options = [Option(f"[dim]{escape(label)}[/dim]", id="")]
        option_list = self.query_one("#panel-content", OptionList)
        previous = option_list.highlighted
        option_list.set_options(options)
        if options:
            option_list.highlighted = min(previous if previous is not None else 0, len(options) - 1)

    def _switch_mention_kind(self, kind: str, *, filter_value: str) -> None:
        if self.mention_items is None:
            return
        title, items, subtitle = self.mention_items(kind)
        self.mention_kind = kind
        self.panel_title = title
        self.items = items
        self.subtitle = subtitle
        self.query_one("#panel-header", Static).update(title)
        self.query_one("#panel-subtitle", Static).update(subtitle)
        filter_input = self.query_one("#panel-filter", Input)
        filter_input.value = filter_value
        self._apply_filter(filter_value)


COMMAND_SPECS = [
    ("/new", "/new [title]"),
    ("/threads", "/threads"),
    ("/status", "/status"),
    ("/context", "/context"),
    ("/rules", "/rules"),
    ("/config", "/config"),
    ("/models", "/models"),
    ("/level", "/level [name]"),
    ("/mcp", "/mcp"),
    ("/skills", "/skills"),
    ("/skill", "/skill [name]"),
    ("/scripts", "/scripts"),
    ("/runs", "/runs"),
    ("/panel", "/panel"),
    ("/clear", "/clear"),
    ("/quit", "/quit"),
    ("/help", "/help"),
]


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
    "node_modules",
}
MAX_MENTION_ITEMS = 300


@dataclass(frozen=True)
class CommandSpec:
    name: str
    usage: str
    description: str


class EmptyState(Static):
    """Animated empty transcript state."""

    FRAMES = ["·  ", "·· ", "···", " ··", "  ·", "   "]

    DEFAULT_CSS = """
    EmptyState {
        width: 100%;
        height: 100%;
        content-align: center middle;
        color: #7f91a8;
    }

    EmptyState.hidden {
        display: none;
    }
    """

    def __init__(self, *, id: str) -> None:
        super().__init__("", id=id)
        self.frame = 0

    def tick(self) -> None:
        frame = self.FRAMES[self.frame % len(self.FRAMES)]
        self.frame += 1
        text = getattr(self.app, "_text", lambda key: key)
        self.update(
            f"[bold #dce7f3]{escape(text('ready_title'))}[/bold #dce7f3] [dim]{escape(frame)}[/dim]\n"
            f"[dim]{escape(text('ready_hint'))}[/dim]"
        )


class ComposerTextArea(TextArea):
    """Composer text area with Ctrl+C reserved for app-level interrupt/quit."""

    def action_copy(self) -> None:
        super().action_copy()
        app = self.app
        notify = getattr(app, "_text", lambda k: k)
        app.notify(notify("copied"), timeout=1.5)

    BINDINGS = [
        binding
        for binding in TextArea.BINDINGS
        if not {"ctrl+c", "super+c"}.intersection(
            key.strip() for key in binding.key.split(",")
        )
    ]


class TranscriptCell(Static):
    """Small transcript block used by the Textual chat timeline."""

    SELECTION_STYLE = Style(color="#061018", bgcolor="#7dd3fc")

    DEFAULT_CSS = """
    TranscriptCell {
        width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
    }

    TranscriptCell.user {
        background: #111a24;
        color: #dce7f3;
        border-left: solid #2e9ad8;
    }

    TranscriptCell.assistant {
        background: #0f151d;
        color: #e5e7eb;
        border-left: solid #516071;
    }

    TranscriptCell.event {
        background: #0e141b;
        color: #aeb7c4;
        border-left: solid #2b3542;
    }

    TranscriptCell.error {
        background: #241316;
        color: #ffb4b4;
        border-left: solid #e26363;
    }
    """

    def __init__(self, content: object = "", *, copy_text: str | None = None, **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self.copy_text: str | None = copy_text if copy_text is not None else self._plain_copy_text(content)
        self._rendered_copy_lines: dict[int, str] = {}

    def update(self, content: object = "", *, layout: bool = True, copy_text: str | None = None) -> None:
        self.copy_text = copy_text if copy_text is not None else self._plain_copy_text(content)
        self._rendered_copy_lines.clear()
        super().update(content, layout=layout)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        text = self._current_copy_text()
        if text is not None:
            return selection.extract(text), "\n"
        return super().get_selection(selection)

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        rendered_text = strip.text.rstrip()
        if rendered_text:
            self._rendered_copy_lines[y] = rendered_text

        offset_x = 0
        segments = []
        for segment in strip:
            if segment.control:
                segments.append(segment)
                continue
            text = segment.text
            style = segment.style
            if text and (style is None or style._meta is None or "offset" not in style.meta):
                style = (style or Style()) + Style(meta={"offset": (offset_x, y)})
            segments.append(Segment(text, style, segment.control))
            offset_x += len(text)
        return self._highlight_selection(Strip(segments, strip.cell_length), y)

    def _highlight_selection(self, strip: Strip, y: int) -> Strip:
        selection = self.text_selection
        if selection is None:
            return strip
        span = selection.get_span(y)
        if span is None:
            return strip
        start, end = span
        if end == -1:
            end = strip.cell_length
        start = max(0, min(start, strip.cell_length))
        end = max(start, min(end, strip.cell_length))
        if start == end:
            return strip
        before = strip.crop(0, start)
        selected = self._apply_selection_style(strip.crop(start, end))
        after = strip.crop(end, strip.cell_length)
        return Strip.join([before, selected, after])

    def _apply_selection_style(self, strip: Strip) -> Strip:
        segments = []
        for text, style, control in strip:
            if control:
                segments.append(Segment(text, style, control))
            else:
                segments.append(Segment(text, (style or Style()) + self.SELECTION_STYLE))
        return Strip(segments, strip.cell_length)

    def _current_copy_text(self) -> str | None:
        if self._rendered_copy_lines:
            return "\n".join(
                self._rendered_copy_lines.get(y, "")
                for y in range(max(self._rendered_copy_lines) + 1)
            )
        return self.copy_text

    def _plain_copy_text(self, content: object) -> str | None:
        if isinstance(content, str):
            try:
                return str(render_markup(content))
            except Exception:
                return content
        return None


class UvAgentApp(App[None]):
    CSS = """
    Screen {
        layout: horizontal;
        background: #0b0f14;
        color: #d8dee9;
    }

    Screen > .screen--selection {
        background: #7dd3fc;
        color: #061018;
    }

    ToastRack {
        dock: top;
        align-horizontal: right;
    }

    #main-column {
        width: 1fr;
        min-width: 0;
        height: 100%;
        background: #0b0f14;
    }

    #transcript {
        height: 1fr;
        min-height: 6;
        padding: 1 2 0 1;
        background: #0b0f14;
    }

    #bottom-pane {
        height: auto;
        max-height: 50%;
        padding: 0 1 0 1;
        background: #0b0f14;
    }

    #composer-shell {
        height: auto;
        background: #0b0f14;
    }

    #composer-shell.busy {
        background: #0b0f14;
    }

    #composer-meta {
        display: none;
    }

    #composer {
        width: 1fr;
        height: 5;
        min-height: 5;
        margin: 0;
        border: round #2a3646;
        padding: 0 1;
        background: #0b0f14;
        color: #edf2f7;
    }

    #composer:focus {
        border: round #3f9bc9;
    }

    #composer-footer {
        height: 1;
        color: #516071;
        padding: 0 1;
        background: #0b0f14;
    }
    """

    BINDINGS = [
        Binding("ctrl+enter", "submit_composer", "Send", priority=True),
        Binding("ctrl+j", "submit_composer", "Send", priority=True),
        Binding("tab", "toggle_composer_height", "Height", priority=True),
        Binding("ctrl+s", "toggle_status_panel", "Status", priority=True),
        Binding("ctrl+o", "open_threads", "Threads", priority=True),
        Binding("ctrl+p", "open_command_palette", "Commands", priority=True),
        Binding("ctrl+c", "interrupt_turn", "Interrupt", priority=True, show=False),
        Binding("enter", "focus_composer", "Focus composer", priority=True, show=False),
        Binding("f1", "help", "Help", priority=True),
        Binding("escape", "clear_input", "Clear"),
    ]

    busy = reactive(False)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
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
        self._queue: list[str] = []
        self._last_status = tr(self.language, "idle")
        self._spinner_index = 0
        self._last_tool_payload: dict[str, object] | None = None
        self._quit_armed = False
        self._last_quit_request_at = 0.0
        self._transcript_has_content = False
        self._reasoning_cell: TranscriptCell | None = None
        self._reasoning_buffer = ""
        self._last_composer_text = ""
        self._interrupt_armed = False
        self._last_interrupt_request_at = 0.0
        self._current_worker: Worker[None] | None = None
        self._current_cancel_event: asyncio.Event | None = None
        self._selection_copy_timer: Any | None = None
        self._pending_selection_copy = ""
        self._last_auto_copied_selection = ""
        self._composer_height_override: str | None = None
        self._composer_expanded = False

    def compose(self) -> ComposeResult:
        with Vertical(id="main-column"):
            with VerticalScroll(id="transcript"):
                yield EmptyState(id="empty-state")
            with Vertical(id="bottom-pane"):
                with Vertical(id="composer-shell"):
                    yield Static("", id="composer-meta")
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
        self.query_one("#empty-state", EmptyState).tick()
        self._refresh_status(self._text("idle"))
        self.set_interval(0.16, self._tick, name="spinner")
        self.query_one("#composer", TextArea).focus()

    def on_resize(self) -> None:
        self._refresh_status()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "composer":
            return
        previous = self._last_composer_text
        current = event.text_area.text
        self._last_composer_text = current
        self._resize_composer(current)
        self._refresh_status()
        if current == "/" and previous == "":
            event.text_area.load_text("")
            self._last_composer_text = ""
            self._resize_composer("")
            self._open_command_palette()
        elif current in {"?", "？"} and previous == "":
            event.text_area.load_text("")
            self._last_composer_text = ""
            self._resize_composer("")
            self._open_help_panel()
        else:
            self._maybe_open_mention_picker(event.text_area)

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

    def _tick(self) -> None:
        if not self._transcript_has_content:
            try:
                self.query_one("#empty-state", EmptyState).tick()
            except NoMatches:
                pass
        if self.busy:
            self._refresh_status()

    def _text(self, key: str) -> str:
        return tr(self.language, key)

    def _commands(self) -> list[CommandSpec]:
        return [
            CommandSpec(name, usage, command_description(self.language, name))
            for name, usage in COMMAND_SPECS
        ]

    def action_submit_composer(self) -> None:
        composer = self.query_one("#composer", TextArea)
        prompt = composer.text.strip()
        if not prompt:
            self._flash(self._text("write_first"))
            return
        composer.load_text("")
        self._last_composer_text = ""
        self._composer_height_override = None
        self._resize_composer("")
        if "\n" not in prompt and self._handle_command(prompt):
            return
        if self.busy:
            self._queue.append(prompt)
            self._append_cell(f"[dim]{escape(self._text('queued'))}[/dim]\n{escape(prompt)}", "event")
            self._refresh_status()
            return
        self._start_turn(prompt)

    def _start_turn(self, prompt: str) -> None:
        self.busy = True
        self.query_one("#composer-shell", Vertical).add_class("busy")
        self._assistant_buffer = ""
        self._assistant_cell = None
        self._reasoning_cell = None
        self._reasoning_buffer = ""
        self._interrupt_armed = False
        self._current_cancel_event = asyncio.Event()
        self._append_user(prompt)
        self._reasoning_cell = self._append_cell(
            f"[dim]{escape(self._text('thinking'))}...[/dim]",
            "event",
        )
        self._refresh_status(self._text("working"))
        self._current_worker = self.run_worker(self._run_turn(prompt), exclusive=True, thread=False)

    async def _run_turn(self, prompt: str) -> None:
        try:
            async for item in self.engine.run_turn(
                user_text=prompt,
                thread_id=self.thread_id,
                level=self.level,
                cancel_event=self._current_cancel_event,
            ):
                self.thread_id = item.get("thread_id", self.thread_id)
                event_type = item["type"]
                if event_type == "assistant.delta":
                    await self._append_assistant_delta(item["text"])
                elif event_type == "assistant.reasoning_delta":
                    self._append_reasoning_delta(item["text"])
                elif event_type == "tool.delta":
                    self._refresh_status(self._text("running_python"))
                elif event_type == "model.response":
                    self._refresh_status(self._text("reading"))
                elif event_type == "tool.started":
                    self._append_tool_started(item)
                elif event_type == "tool.output":
                    self._append_tool_output(item)
                elif event_type == "compaction.completed":
                    self._append_cell(f"[dim]{escape(self._text('compacted'))}[/dim]", "event")
                elif event_type == "turn.completed":
                    text = item["final_text"] or self._assistant_buffer
                    if text and self._assistant_cell is None:
                        await self._append_assistant_delta(text)
                    self._refresh_status(self._text("idle"))
                elif event_type == "turn.interrupted":
                    self._append_cell(f"[dim]{escape(self._text('interrupted'))}[/dim]", "event")
                    self._refresh_status(self._text("interrupted"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._append_cell(error_markup(format_error(exc)), "error")
            self._refresh_status(self._text("error"))
        finally:
            self.busy = False
            self._current_worker = None
            self._current_cancel_event = None
            self._interrupt_armed = False
            self.query_one("#composer-shell", Vertical).remove_class("busy")
            if self._last_status != self._text("error"):
                self._refresh_status(self._text("idle"))
            self.query_one("#composer", TextArea).focus()
            if self._queue:
                next_prompt = self._queue.pop(0)
                self._start_turn(next_prompt)

    def action_clear_input(self) -> None:
        composer = self.query_one("#composer", TextArea)
        if composer.text:
            composer.load_text("")
            self._last_composer_text = ""
            self._composer_height_override = None
            self._resize_composer("")
            return

    def action_request_quit(self) -> None:
        now = monotonic()
        if now - self._last_quit_request_at < 0.35:
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

    def action_open_threads(self) -> None:
        self._open_threads_panel()

    def action_open_command_palette(self) -> None:
        self._open_command_palette()

    def action_toggle_composer_height(self) -> None:
        self._composer_height_override = "collapsed" if self._composer_expanded else "expanded"
        self._resize_composer(self.query_one("#composer", TextArea).text)

    def action_focus_composer(self) -> None:
        composer = self.query_one("#composer", TextArea)
        if self.screen.focused is composer:
            return
        composer.focus()

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

    def _handle_command(self, prompt: str) -> bool:
        command, _, rest = prompt.partition(" ")
        if command == "/clear":
            self.thread_id = None
            self._assistant_buffer = ""
            self._assistant_cell = None
            self._tool_cells.clear()
            self._queue.clear()
            self._reset_transcript()
            self._refresh_status(self._text("idle"))
            return True
        if command == "/quit":
            self.action_request_quit()
            return True
        if command == "/new":
            title = rest.strip() or self._text("new_thread")
            self.thread_id = self.engine.thread_store.create_thread(title)
            self._append_cell(
                f"[dim]{escape(self._text('new_thread'))}[/dim] [cyan]{escape(short_thread(self.thread_id))}[/cyan]",
                "event",
            )
            self._refresh_status(self._text("idle"))
            return True
        if command == "/threads":
            self._open_threads_panel()
            return True
        if command == "/status":
            self._open_status_panel()
            return True
        if command == "/context":
            self._open_context_panel()
            return True
        if command == "/rules":
            self._open_rules_panel()
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
        if command == "/skill":
            self._append_skill(rest.strip())
            return True
        if command == "/scripts":
            self._open_scripts_panel()
            return True
        if command == "/level":
            self._handle_level_command(rest.strip())
            return True
        if command == "/runs":
            self._open_runs_panel()
            return True
        if command == "/panel":
            self._flash(self._text("panel_close_hint"))
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
        lines = [f"[bold]{escape(self._text('commands'))}[/bold] [dim](Tab/Enter, Esc)[/dim]"]
        for spec in self._commands():
            lines.append(
                f"[cyan]{escape(spec.usage):<18}[/cyan] [dim]{escape(spec.description)}[/dim]"
            )
        self._open_panel("\n".join(lines), "help", self._text("help"))

    def _append_help(self) -> None:
        lines = [f"[bold]{escape(self._text('commands'))}[/bold] [dim](Ctrl+S, Esc)[/dim]"]
        for spec in self._commands():
            lines.append(
                f"[cyan]{escape(spec.usage):<18}[/cyan] [dim]{escape(spec.description)}[/dim]"
            )
        self._append_cell("\n".join(lines), "event")

    def _append_user(self, text: str) -> None:
        label = "你" if self.language.is_chinese else "you"
        self._append_cell(f"[bold #7dd3fc]{label}[/bold #7dd3fc]\n{escape(text)}", "user")

    async def _append_assistant_delta(self, text: str) -> None:
        self._assistant_buffer += text
        if self._assistant_cell is None:
            self._mark_transcript_content()
            self._assistant_cell = TranscriptCell(classes="assistant")
            self.query_one("#transcript", VerticalScroll).mount(self._assistant_cell)
        self._assistant_cell.update(Markdown(self._assistant_buffer), copy_text=self._assistant_buffer)
        self._scroll_end()

    def _append_reasoning_delta(self, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            return
        self._reasoning_buffer = (self._reasoning_buffer + " " + stripped).strip()
        first = self._reasoning_buffer.splitlines()[0]
        if len(first) > 120:
            first = first[:117].rstrip() + "..."
        markup = f"[dim]{escape(self._text('thinking'))}[/dim] [italic]{escape(first)}[/italic]"
        if self._reasoning_cell is None:
            self._reasoning_cell = self._append_cell(markup, "event")
        else:
            self._reasoning_cell.update(markup)
            self._scroll_end()

    def _append_tool_output(self, item: dict[str, Any]) -> None:
        payload = parse_tool_payload(item.get("output", {}))
        if payload is None:
            self._append_cell(
                f"[dim]{escape(self._text('python'))} {escape(self._text('python_completed'))}[/dim]",
                "event",
            )
            return

        self._last_tool_payload = payload
        markup = tool_timeline_markup(payload)
        cell = self._tool_cells.pop(str(item.get("call", {}).get("call_id") or ""), None)
        if cell is None:
            self._append_cell(markup, "event")
        else:
            cell.update(markup)
            self._scroll_end()
        self._refresh_status(self._text("working"))

    def _append_tool_started(self, item: dict[str, Any]) -> None:
        call = item.get("call") or {}
        call_id = str(call.get("call_id") or "")
        name = str(call.get("name") or "python")
        detail = self._tool_call_preview(call)
        cell = self._append_cell(
            f"[cyan]{escape(name)}[/cyan] [dim]{escape(self._text('python_running'))}[/dim]{detail}",
            "event",
        )
        if call_id:
            self._tool_cells[call_id] = cell
        self._refresh_status(self._text("running_python"))

    def _tool_call_preview(self, call: dict[str, Any]) -> str:
        raw_args = call.get("arguments") or ""
        try:
            import json

            args = json.loads(raw_args)
        except Exception:
            return ""
        code = str(args.get("code") or "").strip()
        if not code:
            return ""
        first = next((line.strip() for line in code.splitlines() if line.strip()), "")
        if len(first) > 72:
            first = first[:69].rstrip() + "..."
        return f"\n[dim]{escape(first)}[/dim]"

    def _append_cell(self, content: object, classes: str) -> TranscriptCell:
        self._mark_transcript_content()
        cell = TranscriptCell(content, classes=classes, markup=True)
        self.query_one("#transcript", VerticalScroll).mount(cell)
        self._scroll_end()
        return cell

    def _mark_transcript_content(self) -> None:
        self._transcript_has_content = True
        try:
            self.query_one("#empty-state", EmptyState).add_class("hidden")
        except NoMatches:
            pass

    def _reset_transcript(self, *, show_empty: bool = True) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.query("*").remove()
        self._transcript_has_content = False
        if show_empty:
            empty_state = EmptyState(id="empty-state")
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
            items.append(
                PickerItem(
                    id=thread_id,
                    title=f"{marker}{title}",
                    description=last_text or self._text("no_messages"),
                    meta=(
                        f"{short_thread(thread_id)} · {thread.get('turn_count', 0)} "
                        f"{self._text('turns')} · {updated}"
                    ),
                )
            )
        self._open_picker(
            self._text("threads"),
            items,
            self._resume_thread,
            subtitle=self._text("thread_search_hint"),
        )

    def _maybe_open_mention_picker(self, composer: TextArea) -> None:
        trigger = self._mention_trigger_at_cursor(composer)
        if trigger is None:
            return
        kind = "thread" if trigger == "@@" else "file"
        self._open_mention_picker(kind)

    def _mention_trigger_at_cursor(self, composer: TextArea) -> str | None:
        row, column = composer.cursor_location
        lines = composer.text.split("\n")
        if row >= len(lines):
            return None
        prefix = lines[row][:column]
        if prefix.endswith("@@"):
            return "@@"
        if prefix.endswith("@"):
            return "@"
        return None

    def _open_mention_picker(self, kind: str) -> None:
        try:
            composer = self.query_one("#composer", TextArea)
        except NoMatches:
            return
        expected_trigger = "@@" if kind == "thread" else "@"
        if self._mention_trigger_at_cursor(composer) != expected_trigger:
            return
        if kind == "thread":
            self._open_thread_mention_picker()
            return
        self._open_file_mention_picker()

    def _open_file_mention_picker(self) -> None:
        title, items, subtitle = self._mention_picker_items("file")
        self._open_picker(
            title,
            items,
            self._choose_file_mention,
            subtitle=subtitle,
            mention_kind="file",
            mention_items=self._mention_picker_items,
        )

    def _open_thread_mention_picker(self) -> None:
        title, items, subtitle = self._mention_picker_items("thread")
        self._open_picker(
            title,
            items,
            self._choose_thread_mention,
            subtitle=subtitle,
            mention_kind="thread",
            mention_items=self._mention_picker_items,
            initial_filter="@",
        )

    def _mention_picker_items(self, kind: str) -> tuple[str, list[PickerItem], str]:
        if kind == "thread":
            return (
                self._text("mention_threads"),
                self._thread_mention_items(),
                self._text("mention_threads_hint"),
            )
        return (
            self._text("mention_files"),
            self._file_mention_items(),
            self._text("mention_files_hint"),
        )

    def _thread_mention_items(self) -> list[PickerItem]:
        threads = self.engine.thread_store.list_threads()
        items = []
        for thread in threads:
            thread_id = str(thread.get("thread_id") or "")
            title = str(thread.get("title") or self._text("new_thread"))
            last_text = str(thread.get("last_text") or "").replace("\n", " ")
            if len(last_text) > 120:
                last_text = last_text[:117].rstrip() + "..."
            marker = f"{self._text('current')} " if thread_id == self.thread_id else ""
            items.append(
                PickerItem(
                    id=thread_id,
                    title=f"{marker}{title}",
                    description=last_text or self._text("no_messages"),
                    meta=f"{short_thread(thread_id)} · {thread.get('turn_count', 0)} {self._text('turns')}",
                )
            )
        return items

    def _file_mention_items(self) -> list[PickerItem]:
        root = self.project_root.resolve()
        items: list[PickerItem] = []
        stack = [root]
        while stack and len(items) < MAX_MENTION_ITEMS:
            directory = stack.pop()
            try:
                children = sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.casefold()))
            except OSError:
                continue
            for path in children:
                if len(items) >= MAX_MENTION_ITEMS:
                    break
                if path.is_dir():
                    if path.name not in IGNORED_MENTION_DIRS and not path.name.startswith("."):
                        stack.append(path)
                    continue
                if not path.is_file() or path.suffix.lower() not in CODE_FILE_SUFFIXES:
                    continue
                try:
                    relative = path.relative_to(root)
                except ValueError:
                    continue
                mention = relative.as_posix()
                items.append(
                    PickerItem(
                        id=mention,
                        title=mention,
                        description=self._text("mention_file_description"),
                    )
                )
        return items

    def _choose_file_mention(self, path: str) -> None:
        self._insert_mention(f"@{path}", "@")

    def _choose_thread_mention(self, thread_id: str) -> None:
        self._insert_mention(f"@thread:{thread_id}", ("@@", "@"))

    def _insert_mention(self, mention: str, triggers: str | tuple[str, ...]) -> None:
        composer = self.query_one("#composer", TextArea)
        row, column = composer.cursor_location
        lines = composer.text.split("\n")
        replacement = mention + " "
        trigger_options = (triggers,) if isinstance(triggers, str) else triggers
        if row < len(lines) and mention.startswith("@thread:") and lines[row][:column].endswith("@"):
            trigger_options = ("@",)
        matched_trigger = next(
            (
                trigger
                for trigger in sorted(trigger_options, key=len, reverse=True)
                if row < len(lines) and lines[row][:column].endswith(trigger)
            ),
            "",
        )
        if matched_trigger:
            composer.replace(
                replacement,
                (row, column - len(matched_trigger)),
                (row, column),
                maintain_selection_offset=False,
            )
        else:
            end_trigger = next(
                (
                    trigger
                    for trigger in sorted(trigger_options, key=len, reverse=True)
                    if composer.text.endswith(trigger)
                ),
                "",
            )
            if end_trigger:
                composer.load_text(composer.text[: -len(end_trigger)] + replacement)
                composer.cursor_location = composer.document.end
            else:
                composer.insert(replacement)
        self._last_composer_text = composer.text
        self._resize_composer(composer.text)
        composer.focus()

    def _open_status_panel(self) -> None:
        self._open_panel(self._status_panel_markup(), "status", self._text("status"))

    def _status_panel_markup(self) -> str:
        self.engine.refresh_config()
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
                f"trigger {format_tokens(stats.threshold_tokens)} · "
                f"target {format_tokens(stats.target_tokens)} · "
                f"headroom {format_tokens(stats.headroom_tokens)}"
            )
        except ConfigError as exc:
            model_line = "[red]not configured[/red]"
            provider_line = escape(str(exc))
            context_line = "-"
            compress_line = "-"
        level_name = self.level or self.engine.config.runtime.default_level
        lines = [
            f"- state: [cyan]{escape(self._last_status)}[/cyan]",
            f"- level: [cyan]{escape(level_name)}[/cyan]",
            f"- model: {model_line}",
            f"- provider/api: {provider_line}",
            f"- context: {context_line}",
            f"- compaction: {compress_line}",
            f"- thread: {escape(short_thread(self.thread_id))}",
            f"- queued: {len(self._queue)}",
            f"- user state: {escape(str(uv_agent_home()))}",
            f"- project state: {escape(str(project_state_dir(self.project_root)))}",
            f"- host: {escape(host_environment_line())}",
            f"- language: {escape(self.language.name)}",
        ]
        return "\n".join(lines)

    def _open_context_panel(self) -> None:
        self._open_panel(self._context_panel_markup(), "context", self._text("context"))

    def _context_panel_markup(self) -> str:
        self.engine.refresh_config()
        rules = self.engine.project_rule_context()
        try:
            stats = self.engine.context_stats(self.thread_id, self.level)
            lines = [
                f"[bold]token budget[/bold] [dim]{escape(stats.source)}[/dim]",
                f"- used: [cyan]{format_tokens(stats.used_tokens)}[/cyan] / {format_tokens(stats.context_window_tokens)} ({stats.percent}%)",
                f"- headroom: {format_tokens(stats.headroom_tokens)}",
                f"- compression trigger: {format_tokens(stats.threshold_tokens)}",
                f"- compression target: {format_tokens(stats.target_tokens)}",
            ]
        except ConfigError as exc:
            lines = ["[bold]token budget[/bold]", f"[red]{escape(str(exc))}[/red]"]
        lines.extend(
            [
                "",
                "[bold]workspace rules[/bold]",
                f"- loaded files: {len(rules.rules)}",
                f"- capped: {'yes' if rules.truncated else 'no'}",
            ]
        )
        if rules.omitted_files:
            lines.append(f"- omitted files: {rules.omitted_files}")
        for rule in rules.rules:
            suffix = " [yellow]truncated[/yellow]" if rule.truncated else ""
            lines.append(f"  [cyan]{escape(rule.scope)}[/cyan] {escape(str(rule.path))}{suffix}")
        if not rules.rules:
            lines.append("[dim]no AGENTS.md files discovered[/dim]")
        return "\n".join(lines)

    def _open_rules_panel(self) -> None:
        rules = self.engine.project_rule_context()
        if not rules.rules:
            self._open_panel(
                f"[dim]{escape(self._text('no_agents'))}[/dim]",
                "rules",
                self._text("rules"),
            )
            return
        lines: list[str] = []
        for rule in rules.rules:
            title = f"[cyan]{escape(rule.scope)}[/cyan] {escape(str(rule.path))}"
            if rule.truncated:
                title += " [yellow]truncated[/yellow]"
            preview = rule.text.strip()
            if len(preview) > 2400:
                preview = preview[:2400].rstrip() + "\n..."
            lines.append(title + "\n" + escape(preview))
        if rules.truncated and rules.omitted_files:
            lines.append(f"[yellow]{rules.omitted_files} file(s) omitted by context cap[/yellow]")
        self._open_panel("\n\n".join(lines), "rules", self._text("rules"))

    def _open_config_panel(self) -> None:
        self.engine.refresh_config()
        sources = config_sources(self.project_root)
        lines = ["[bold]sources[/bold]"]
        for source in sources:
            exists = "yes" if source["exists"] else "no"
            lines.append(
                f"- {escape(source['scope'])}: {escape(source['path'])} [dim]exists={exists}[/dim]"
            )
        level_name = self.level or self.engine.config.runtime.default_level
        lines.append("\n[bold]active[/bold]")
        try:
            model = self.engine.config.model_for_level(self.level)
            provider = self.engine.config.provider_for_model(model)
            lines.extend(
                [
                    f"- level: [cyan]{escape(level_name)}[/cyan]",
                    f"- model: {escape(model.name)} -> {escape(model.model)}",
                    f"- provider: {escape(provider.name)}",
                    f"- api: {escape(model.api)}",
                    f"- context window: {format_tokens(model.context_window_tokens)}",
                ]
            )
        except ConfigError as exc:
            lines.append(f"[red]{escape(str(exc))}[/red]")
        redacted = redact_config(load_raw_config(self.project_root))
        preview = json.dumps(redacted, ensure_ascii=False, indent=2)
        if len(preview) > 2200:
            preview = preview[:2200].rstrip() + "\n..."
        lines.extend(["\n[bold]redacted merged config[/bold]", escape(preview)])
        self._open_panel("\n".join(lines), "config", self._text("config"))

    def _open_models_panel(self) -> None:
        self.engine.refresh_config()
        lines = ["[bold]levels[/bold] [dim](/level name selects)[/dim]"]
        for name, level in self.engine.config.levels.items():
            marker = "*" if name == (self.level or self.engine.config.runtime.default_level) else "-"
            lines.append(f"{marker} [cyan]{escape(name)}[/cyan] -> {escape(level.model)}")
        lines.append("\n[bold]models[/bold]")
        for name, model in self.engine.config.models.items():
            lines.append(
                f"- {escape(name)}: {escape(model.model)} [dim]{escape(model.api)} · {format_tokens(model.context_window_tokens)}[/dim]"
            )
        self._open_panel("\n".join(lines), "models", self._text("models"))

    def _handle_level_command(self, name: str) -> None:
        if not name:
            self._open_models_panel()
            return
        if name not in self.engine.config.levels:
            self._append_cell(f"[red]{escape(self._text('unknown_level'))}[/red] {escape(name)}", "error")
            return
        self.level = name
        self._append_cell(f"[dim]{escape(self._text('level'))}[/dim] [cyan]{escape(name)}[/cyan]", "event")
        self._refresh_status()

    def _open_runs_panel(self) -> None:
        if not self._last_tool_payload:
            self._open_panel(
                f"[dim]{escape(self._text('no_runs'))}[/dim]",
                "runs",
                self._text("runs"),
            )
            return
        self._open_panel(
            tool_result_markup(self._last_tool_payload),
            "runs",
            self._text("last_run"),
        )

    def _open_scripts_panel(self) -> None:
        scripts = self.engine.runner.store.list_scripts(limit=32)
        if not scripts:
            self._open_panel(f"[dim]{escape(self._text('no_scripts'))}[/dim]", "scripts", self._text("scripts"))
            return
        lines = [
            f"[bold]{escape(self._text('scripts_header'))}[/bold] "
            f"[dim]({escape(self._text('scripts_limit'))})[/dim]"
        ]
        for script in scripts:
            lines.append(
                f"- [cyan]{escape(str(script.get('script_id') or ''))}[/cyan] "
                f"[dim]runs={script.get('run_count', 0)} · {escape(str(script.get('last_used_at') or ''))}[/dim]\n"
                f"  {escape(str(script.get('summary') or ''))}"
            )
        self._open_panel("\n".join(lines), "scripts", self._text("scripts"))

    def _open_mcp_panel(self) -> None:
        self.engine.refresh_config()
        servers = discover_mcp_servers(self.project_root)
        if not servers:
            self._open_panel(
                f"[dim]{escape(self._text('no_mcp'))}[/dim]",
                "mcp",
                self._text("mcp"),
            )
            return
        lines = [f"[dim]{escape(self._text('mcp_declarations'))}[/dim]"]
        for server in servers:
            command = f" [dim]{escape(server.command)}[/dim]" if server.command else ""
            lines.append(
                f"- [cyan]{escape(server.name)}[/cyan] ({escape(server.scope)}) {escape(server.description)}{command}"
            )
        self._open_panel("\n".join(lines), "mcp", self._text("mcp"))

    def _open_skills_panel(self) -> None:
        self.engine.refresh_config()
        skills = discover_skills(self.project_root)
        if not skills:
            self._open_panel(
                f"[dim]{escape(self._text('no_skills'))}[/dim]",
                "skills",
                self._text("skills"),
            )
            return
        lines = [f"[dim]{escape(self._text('skill_hint'))}[/dim]"]
        for skill in skills:
            lines.append(
                f"- [cyan]{escape(skill.name)}[/cyan] ({escape(skill.scope)}) {escape(skill.description)}"
            )
        self._open_panel("\n".join(lines), "skills", self._text("skills"))

    def _append_skill(self, name: str) -> None:
        if not name:
            self._open_skills_panel()
            return
        skills = {skill.name: skill for skill in discover_skills(self.project_root)}
        skill = skills.get(name)
        if skill is None:
            self._append_cell(f"[red]{escape(self._text('unknown_skill'))}[/red] {escape(name)}", "error")
            return
        text = skill.path.read_text(encoding="utf-8")
        preview = "\n".join(text.splitlines()[:18])
        if len(text.splitlines()) > 18:
            preview += "\n..."
        self._append_cell(
            f"[bold]skill {escape(skill.name)}[/bold] [dim]{escape(str(skill.path))}[/dim]\n"
            + escape(preview),
            "event",
        )

    def _open_command_palette(self, *, query: str = "") -> None:
        items = [
            PickerItem(
                id=spec.name,
                title=spec.usage,
                description=spec.description,
            )
            for spec in self._commands()
        ]
        self._open_picker(
            self._text("command_palette"),
            items,
            self._choose_command,
            subtitle=self._text("command_filter_hint"),
            initial_filter=query,
        )

    def _choose_command(self, command: str) -> None:
        spec = next((item for item in self._commands() if item.name == command), None)
        if spec is None:
            return
        if "[" in spec.usage:
            replacement = command + " "
            composer = self.query_one("#composer", TextArea)
            composer.load_text(replacement)
            self._last_composer_text = replacement
            self._resize_composer(replacement)
            composer.focus()
            return
        self._handle_command(command)

    def _resume_thread(self, thread_id: str) -> None:
        if not thread_id:
            return
        self.thread_id = thread_id
        self._assistant_buffer = ""
        self._assistant_cell = None
        self._reasoning_cell = None
        self._reasoning_buffer = ""
        self._tool_cells.clear()
        self._reset_transcript(show_empty=False)
        self._render_thread_history(thread_id)
        if not self._transcript_has_content:
            self._reset_transcript()
        self._refresh_status(self._text("resumed"))

    def _render_thread_history(self, thread_id: str) -> None:
        for event in self.engine.thread_store.read(thread_id):
            event_type = event.get("type")
            if event_type == "item.user":
                self._append_user_from_history(event.get("item") or {})
            elif event_type == "item.model_response":
                text = self._model_response_text(event.get("output") or [])
                if text:
                    self._append_cell(Markdown(text), "assistant")
            elif event_type == "item.tool_call":
                item = event.get("item") or {}
                name = str(item.get("name") or "python")
                self._append_cell(f"[cyan]{escape(name)}[/cyan] [dim]{escape(self._text('python_called'))}[/dim]", "event")
            elif event_type == "item.runner_result":
                result = event.get("result") or {}
                self._last_tool_payload = result
                self._append_cell(tool_timeline_markup(result), "event")
            elif event_type == "item.image_attachment":
                attachment = event.get("attachment") or {}
                self._append_cell(
                    f"[dim]{escape(self._text('image_attached'))}[/dim] "
                    f"[cyan]{escape(str(attachment.get('stored_path') or ''))}[/cyan]",
                    "event",
                )
            elif event_type == "item.reasoning_delta":
                self._append_reasoning_delta(str(event.get("text") or ""))
            elif event_type == "item.compaction":
                self._append_cell(f"[dim]{escape(self._text('compacted'))}[/dim]", "event")

    def _append_user_from_history(self, item: dict[str, Any]) -> None:
        self._reasoning_cell = None
        self._reasoning_buffer = ""
        parts = []
        for content in item.get("content") or []:
            if content.get("type") in {"input_text", "text"}:
                parts.append(str(content.get("text") or ""))
        if parts:
            self._append_user("\n".join(parts))

    def _model_response_text(self, output: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in output:
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"}:
                    parts.append(str(content.get("text") or ""))
        return "".join(parts)

    def _open_fullscreen_panel(self, title: str, markup: str, *, subtitle: str = "") -> None:
        self.push_screen(FullscreenPanel(title=title, body=markup, subtitle=subtitle))

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
    ) -> None:
        panel = FullscreenPanel(
            title=title,
            items=items,
            subtitle=subtitle,
            initial_filter=initial_filter,
            mention_kind=mention_kind,
            mention_items=mention_items,
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
        self._open_fullscreen_panel(panel_title, markup, subtitle=self._text("panel_closes"))
        self._refresh_status()

    def _resize_composer(self, text: str) -> None:
        line_count = max(1, text.count("\n") + 1)
        if line_count <= 4 and self._composer_height_override == "collapsed":
            self._composer_height_override = None
        if self._composer_height_override == "expanded":
            expanded = True
        elif self._composer_height_override == "collapsed":
            expanded = False
        else:
            expanded = line_count > 4
        self._composer_expanded = expanded
        height = self._expanded_composer_height() if expanded else COMPOSER_COLLAPSED_HEIGHT
        self.query_one("#composer", TextArea).styles.height = height

    def _expanded_composer_height(self) -> int:
        return max(
            COMPOSER_COLLAPSED_HEIGHT,
            (self.size.height // 2) - COMPOSER_BOTTOM_RESERVED_ROWS,
        )

    def _flash(self, message: str, *, severity: str = "information") -> None:
        self.notify(message, severity=severity, timeout=2.0)
        self._last_status = message
        self._refresh_status()

    def _refresh_status(self, state: str | None = None) -> None:
        if state is not None:
            self._last_status = state
        self.engine.refresh_config()
        self.language = detect_user_language(self.engine.config.ui.language)
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
        queued = f" · q{len(self._queue)}" if self._queue else ""
        spinner = ""
        if self.busy:
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            spinner = frames[self._spinner_index % len(frames)] + " "
            self._spinner_index += 1

        if self.busy:
            footer = (
                f"[cyan]{spinner}{escape(state_text)}[/cyan] "
                f"[dim]{escape(level_name)} · {escape(compact_context)} · "
                f"{escape(short_thread(self.thread_id))}{queued}[/dim]"
            )
        else:
            footer = (
                f"[dim]{escape(level_name)} · {escape(compact_context)} · "
                f"{escape(short_thread(self.thread_id))}{queued}[/dim]"
            )
        self.query_one("#composer-meta", Static).update("")
        self.query_one("#composer-footer", Static).update(footer)

    def _scroll_end(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        self.call_after_refresh(transcript.scroll_end, animate=False)
