from __future__ import annotations

import asyncio
import json
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from rich.markdown import Markdown
from rich.markup import escape, render as render_markup
from rich.segment import Segment
from rich.style import Style
from rich.cells import cell_len
from PIL import Image, UnidentifiedImageError
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
from watchfiles import Change, watch

from uv_agent.app_factory import create_engine
from uv_agent.clipboard import ClipboardImageError, save_clipboard_image
from uv_agent.config import (
    ConfigError,
    config_sources,
    editable_config_path,
    load_config,
    load_raw_config,
    redact_config,
)
from uv_agent.environment import application_version, detect_user_language, host_environment_line
from uv_agent.errors import error_markup, format_error
from uv_agent.i18n import command_description, tr
from uv_agent.mcp_config import discover_mcp_servers
from uv_agent.paths import project_state_dir, uv_agent_home
from uv_agent.session.store import VISIBLE_HISTORY_EVENT_TYPES
from uv_agent.skills import discover_skills
from uv_agent.tui.formatting import (
    format_elapsed,
    format_tokens,
    parse_tool_payload,
    short_thread,
    tool_call_detail_highlight_markup,
    tool_call_detail_markup,
    tool_call_preview_line,
    tool_call_summary_markup,
    tool_detail_markup,
    tool_result_markup,
    tool_timeline_markup,
)


COMPOSER_COLLAPSED_HEIGHT = 5
COMPOSER_BOTTOM_RESERVED_ROWS = 2
QUIT_KEY_DEBOUNCE_SECONDS = 0.08
MAX_COMPOSER_HISTORY = 50
COMPOSER_HISTORY_FILENAME = "composer_history.json"


@dataclass(frozen=True)
class PickerItem:
    id: str
    title: str
    description: str = ""
    meta: str = ""


@dataclass
class PanelPage:
    title: str
    body: str = ""
    items: list[PickerItem] | None = None
    subtitle: str = ""
    filter_value: str = ""
    highlighted: int | None = None
    mention_kind: str | None = None
    mention_items: Callable[[str], tuple[str, list[PickerItem], str]] | None = None
    select_callback: Callable[[str], None] | None = None
    close_on_select: bool = False


@dataclass
class MentionScanCache:
    items: list[PickerItem] = field(default_factory=list)
    complete: bool = False
    generation: int = 0
    worker: Worker[None] | None = None


class PickerOptionList(OptionList):
    ALLOW_SELECT = True


class FullscreenPanel(ModalScreen[str | None]):
    """Scrollable full-screen panel/picker."""

    CSS = """
    FullscreenPanel,
    ToolDetailsPanel,
    ImagePreviewPanel {
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
        scrollbar-size-vertical: 1;
        scrollbar-background: #0a0f15;
        scrollbar-background-hover: #0a0f15;
        scrollbar-background-active: #0a0f15;
        scrollbar-color: #2b3542;
        scrollbar-color-hover: #3a4a60;
        scrollbar-color-active: #7dd3fc;
        scrollbar-corner-color: #0a0f15;
    }

    #panel-body {
        height: 1fr;
        margin: 1 0 0 0;
        border: tall #1f2b3a;
        background: #0a0f15;
        padding: 1 1;
        scrollbar-size-vertical: 1;
        scrollbar-background: #0a0f15;
        scrollbar-background-hover: #0a0f15;
        scrollbar-background-active: #0a0f15;
        scrollbar-color: #2b3542;
        scrollbar-color-hover: #3a4a60;
        scrollbar-color-active: #7dd3fc;
        scrollbar-corner-color: #0a0f15;
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
        scrollbar-size-vertical: 1;
        scrollbar-background: #0a0f15;
        scrollbar-background-hover: #0a0f15;
        scrollbar-background-active: #0a0f15;
        scrollbar-color: #2b3542;
        scrollbar-color-hover: #3a4a60;
        scrollbar-color-active: #7dd3fc;
        scrollbar-corner-color: #0a0f15;
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
        select_callback: Callable[[str], None] | None = None,
        close_on_select: bool = False,
        navigation_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.panel_title = title
        self.body = body
        self.picker_mode = items is not None or mention_kind is not None
        self.items = items or []
        self.subtitle = subtitle
        self.initial_filter = initial_filter.strip()
        self.mention_kind = mention_kind
        self.mention_items = mention_items
        self._selected_mention_kind: str | None = None
        self._filtered = list(self.items)
        self._option_ids: dict[str, str] = {}
        self._select_callback = select_callback
        self._close_on_select = close_on_select
        self.can_navigate = navigation_enabled
        self._page_stack: list[PanelPage] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="panel-shell"):
            yield Static(self.panel_title, id="panel-header")
            yield Static(self.subtitle, id="panel-subtitle")
            yield Input(placeholder=getattr(self.app, "_text", lambda key: key)("filter"), id="panel-filter")
            yield PickerOptionList(id="panel-content", compact=False)
            yield VerticalScroll(Static(self.body, markup=True, id="panel-body-content"), id="panel-body")
            yield Static(getattr(self.app, "_text", lambda key: key)("panel_footer"), id="panel-footer")

    def on_mount(self) -> None:
        self._render_page(filter_value=self.initial_filter)

    def on_click(self, event: events.Click) -> None:
        try:
            shell = self.query_one("#panel-shell", Vertical)
        except NoMatches:
            return
        screen_x = event.screen_x if event.screen_x is not None else event.x
        screen_y = event.screen_y if event.screen_y is not None else event.y
        if not shell.region.contains(screen_x, screen_y):
            event.stop()
            self.dismiss(None)

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
        if not self.picker_mode:
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
            self._select_value(self._option_ids.get(event.option_id, event.option_id))

    def action_dismiss_panel(self) -> None:
        if self._page_stack:
            self._restore_previous_page()
            return
        self.dismiss(None)

    def action_cursor_up(self) -> None:
        if self.picker_mode:
            self.query_one("#panel-content", OptionList).action_cursor_up()
            return
        self.query_one("#panel-body", VerticalScroll).action_scroll_up()

    def action_cursor_down(self) -> None:
        if self.picker_mode:
            self.query_one("#panel-content", OptionList).action_cursor_down()
            return
        self.query_one("#panel-body", VerticalScroll).action_scroll_down()

    def action_page_up(self) -> None:
        if self.picker_mode:
            self.query_one("#panel-content", OptionList).action_page_up()
            return
        self.query_one("#panel-body", VerticalScroll).action_page_up()

    def action_page_down(self) -> None:
        if self.picker_mode:
            self.query_one("#panel-content", OptionList).action_page_down()
            return
        self.query_one("#panel-body", VerticalScroll).action_page_down()

    def action_select_or_close(self) -> None:
        if not self.picker_mode:
            self.dismiss(None)
            return
        if not self._filtered:
            return
        option_list = self.query_one("#panel-content", OptionList)
        highlighted = option_list.highlighted
        if highlighted is None or highlighted < 0 or highlighted >= option_list.option_count:
            return
        option = option_list.get_option_at_index(highlighted)
        if option.id and option.id in self._option_ids:
            self._selected_mention_kind = self.mention_kind
            self._select_value(self._option_ids[option.id])

    def navigate_picker(
        self,
        *,
        title: str,
        items: list[PickerItem],
        callback: Callable[[str], None],
        subtitle: str = "",
        initial_filter: str = "",
        close_on_select: bool = False,
    ) -> None:
        self._page_stack.append(self._snapshot_page())
        self._load_page(
            PanelPage(
                title=title,
                items=items,
                subtitle=subtitle,
                filter_value=initial_filter,
                select_callback=callback,
                close_on_select=close_on_select,
            )
        )

    def navigate_panel(self, *, title: str, body: str, subtitle: str = "") -> None:
        self._page_stack.append(self._snapshot_page())
        self._load_page(PanelPage(title=title, body=body, subtitle=subtitle))

    def close_navigation(self) -> None:
        self._page_stack.clear()
        self.dismiss(None)

    def _select_value(self, value: str) -> None:
        if self._select_callback is not None:
            close_on_select = self._close_on_select
            self._select_callback(value)
            if close_on_select:
                self.close_navigation()
            return
        self.dismiss(value)

    def _snapshot_page(self) -> PanelPage:
        filter_value = ""
        highlighted = None
        if self.picker_mode:
            try:
                filter_value = self.query_one("#panel-filter", Input).value
                highlighted = self.query_one("#panel-content", OptionList).highlighted
            except NoMatches:
                pass
        return PanelPage(
            title=self.panel_title,
            body=self.body,
            items=list(self.items) if self.picker_mode else None,
            subtitle=self.subtitle,
            filter_value=filter_value,
            highlighted=highlighted,
            mention_kind=self.mention_kind,
            mention_items=self.mention_items,
            select_callback=self._select_callback,
            close_on_select=self._close_on_select,
        )

    def _restore_previous_page(self) -> None:
        self._load_page(self._page_stack.pop())

    def _load_page(self, page: PanelPage) -> None:
        self.panel_title = page.title
        self.body = page.body
        self.picker_mode = page.items is not None or page.mention_kind is not None
        self.items = page.items or []
        self.subtitle = page.subtitle
        self.mention_kind = page.mention_kind
        self.mention_items = page.mention_items
        self._select_callback = page.select_callback
        self._close_on_select = page.close_on_select
        self._filtered = list(self.items)
        self._render_page(filter_value=page.filter_value, highlighted=page.highlighted)

    def _render_page(self, *, filter_value: str = "", highlighted: int | None = None) -> None:
        self.query_one("#panel-header", Static).update(self.panel_title)
        self.query_one("#panel-subtitle", Static).update(self.subtitle)
        filter_input = self.query_one("#panel-filter", Input)
        option_list = self.query_one("#panel-content", OptionList)
        body = self.query_one("#panel-body", VerticalScroll)
        body_content = self.query_one("#panel-body-content", Static)
        filter_input.display = self.picker_mode
        option_list.display = self.picker_mode
        body.display = not self.picker_mode
        if self.picker_mode:
            filter_input.value = filter_value
            self._apply_filter(filter_value)
            if highlighted is not None and option_list.option_count:
                option_list.highlighted = min(highlighted, option_list.option_count - 1)
            option_list.focus()
            return
        body_content.update(self.body)
        body.focus()

    def _refresh_options(self) -> None:
        self._option_ids = {}
        options = []
        for index, item in enumerate(self._filtered):
            option_id = f"item_{index}"
            disabled = not item.id
            if not disabled:
                self._option_ids[option_id] = item.id
            options.append(
                Option(
                    f"[bold cyan]{escape(item.title)}[/bold cyan]"
                    + (f"\n[dim]{escape(item.description)}[/dim]" if item.description else "")
                    + (f"\n[dim]{escape(item.meta)}[/dim]" if item.meta else ""),
                    id=option_id,
                    disabled=disabled,
                )
            )
        if not options:
            text = getattr(self.app, "_text", lambda key: key)
            if self.mention_kind == "file":
                label = text("no_mention_files")
            elif self.mention_kind == "thread":
                label = text("no_threads")
            elif self.mention_kind == "mcp":
                label = text("no_mcp")
            elif self.mention_kind == "skill":
                label = text("no_skills")
            else:
                label = text("no_matches")
            options = [Option(f"[dim]{escape(label)}[/dim]", id="")]
        option_list = self.query_one("#panel-content", OptionList)
        previous = option_list.highlighted
        option_list.set_options(options)
        if options:
            option_list.highlighted = min(previous if previous is not None else 0, len(options) - 1)

    def update_picker_items(self, items: list[PickerItem], *, subtitle: str | None = None) -> None:
        if not self.picker_mode:
            return
        self.items = list(items)
        if subtitle is not None:
            self.subtitle = subtitle
            try:
                self.query_one("#panel-subtitle", Static).update(subtitle)
            except NoMatches:
                return
        try:
            filter_value = self.query_one("#panel-filter", Input).value
        except NoMatches:
            return
        self._apply_filter(filter_value)

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
        handler = getattr(self.app, "_start_mention_scan", None)
        if callable(handler):
            handler(kind)


COMMAND_SPECS = [
    ("/threads", "/threads"),
    ("/status", "/status"),
    ("/config", "/config"),
    ("/models", "/models"),
    ("/level", "/level"),
    ("/mcp", "/mcp"),
    ("/skills", "/skills"),
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
MENTION_SCAN_BATCH_SIZE = 50
MENTION_SCAN_DIRECTORY_LIMIT = 20000
MENTION_SCAN_FILE_LIMIT = 100000
MENTION_WATCH_DEBOUNCE_MS = 1000
MENTION_WATCH_POLL_DELAY_MS = 2000


def image_attachment_markup(attachment: dict[str, Any], *, label: str = "image attached") -> str:
    path = Path(str(attachment.get("stored_path") or ""))
    name = path.name or str(path)
    size = int(attachment.get("size_bytes") or 0)
    size_label = f" · {format_tokens(size)}B" if size else ""
    return (
        f"[dim]{escape(label)}[/dim] "
        f"[cyan]{escape(name)}[/cyan]"
        f"[dim]{escape(size_label)}[/dim]\n"
        "[dim][preview][/dim]"
    )


def image_ascii_preview(path: Path, *, width: int = 64, height: int = 36) -> str:
    try:
        with Image.open(path) as image:
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            image = image.convert("RGB")
            rows = []
            for y in range(0, image.height, 2):
                segments = []
                for x in range(image.width):
                    top_red, top_green, top_blue = image.getpixel((x, y))
                    if y + 1 < image.height:
                        bottom_red, bottom_green, bottom_blue = image.getpixel((x, y + 1))
                    else:
                        bottom_red, bottom_green, bottom_blue = top_red, top_green, top_blue
                    segments.append(
                        f"[#{top_red:02x}{top_green:02x}{top_blue:02x} "
                        f"on #{bottom_red:02x}{bottom_green:02x}{bottom_blue:02x}]▀[/]"
                    )
                rows.append("".join(segments))
    except (OSError, UnidentifiedImageError, ValueError):
        return ""
    return "\n".join(rows)


@dataclass(frozen=True)
class CommandSpec:
    name: str
    usage: str
    description: str

    @property
    def palette_title(self) -> str:
        return self.name.removeprefix("/")


@dataclass(frozen=True)
class PendingImage:
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class QueuedTurn:
    prompt: str
    image_paths: list[Path] = field(default_factory=list)


@dataclass
class ThreadRunState:
    thread_id: str
    worker: Worker[None] | None
    cancel_event: asyncio.Event
    queue: list[QueuedTurn]
    status: str
    assistant_buffer: str = ""
    assistant_cell: TranscriptCell | None = None
    reasoning_buffer: str = ""
    reasoning_cell: TranscriptCell | None = None
    tool_cells: dict[str, TranscriptCell] = field(default_factory=dict)
    tool_delta_cells: dict[int, TranscriptCell] = field(default_factory=dict)
    tool_delta_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    process_cells: list[TranscriptCell] = field(default_factory=list)
    process_fold_cell: TranscriptCell | None = None
    process_collapsed: bool = False
    process_anchor_cell: TranscriptCell | None = None


class TranscriptScroll(VerticalScroll):
    """VerticalScroll that auto-follows tail until the user intervenes.

    The streaming SSE renderer used to call `scroll_end` on every delta, which
    fought the user when they were dragging the scrollbar to read history. As
    soon as the user moves the scroll position themselves we drop the
    `follow_tail` flag. Returning to the bottom or submitting from the bottom
    resumes auto-follow; the bottom button still explicitly resumes it too.
    """

    follow_tail = reactive(True)
    # Independent of follow_tail: True whenever the viewport is at (or within
    # a small slack from) the bottom, regardless of how it got there. Drives
    # the "back to bottom" button visibility so the button hides as soon as
    # the user is already at the bottom, even if they got there by scrolling
    # manually rather than via the button.
    near_bottom = reactive(True)

    _BOTTOM_THRESHOLD = 2

    def programmatic_scroll_end(self) -> None:
        # Defer the actual scroll to after the next refresh so any pending
        # mount/update has had a chance to recompute virtual_size; otherwise
        # `scroll_end` reads a stale `max_scroll_y` and only crawls along
        # one row at a time during streaming.
        def _do() -> None:
            self.scroll_end(animate=False, immediate=True)
            self._recompute_near_bottom()

        self.call_after_refresh(_do)

    def engage_follow_tail(self) -> None:
        self.follow_tail = True
        self.programmatic_scroll_end()

    def _disengage_follow_tail(self) -> None:
        if self.follow_tail:
            self.follow_tail = False

    def _disengage_follow_tail_if_scrolled(self, old_scroll_y: float) -> None:
        if self.scroll_y != old_scroll_y:
            self._disengage_follow_tail()

    def _disengage_follow_tail_for_target(self, target_y: float) -> None:
        if self.validate_scroll_y(target_y) != self.scroll_y:
            self._disengage_follow_tail()

    def _recompute_near_bottom(self, *, restore_follow: bool = False) -> None:
        # When there's nothing to scroll, the bottom is trivially "right here"
        # so the button stays hidden.
        if self.max_scroll_y <= 0:
            self.near_bottom = True
            if restore_follow:
                self.follow_tail = True
            return
        near_bottom = (self.max_scroll_y - self.scroll_y) <= self._BOTTOM_THRESHOLD
        self.near_bottom = near_bottom
        if near_bottom and restore_follow:
            self.follow_tail = True

    def watch_scroll_y(self, old: float, new: float) -> None:
        super().watch_scroll_y(old, new)
        self._recompute_near_bottom(restore_follow=True)

    def watch_virtual_size(self, old: Any, new: Any) -> None:
        # Content height changed (new cells appended, expand/collapse, etc.)
        # so the distance-from-bottom may have changed even though scroll_y
        # did not.
        self._recompute_near_bottom()

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        old_scroll_y = self.scroll_y
        super()._on_mouse_scroll_up(event)
        self._disengage_follow_tail_if_scrolled(old_scroll_y)

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        old_scroll_y = self.scroll_y
        super()._on_mouse_scroll_down(event)
        self._disengage_follow_tail_if_scrolled(old_scroll_y)

    def _on_scroll_up(self, event: Any) -> None:
        self._disengage_follow_tail_for_target(
            self.scroll_y - self.scrollable_content_region.height
        )
        super()._on_scroll_up(event)

    def _on_scroll_down(self, event: Any) -> None:
        self._disengage_follow_tail_for_target(
            self.scroll_y + self.scrollable_content_region.height
        )
        super()._on_scroll_down(event)

    def _on_scroll_to(self, message: Any) -> None:
        y = getattr(message, "y", None)
        if y is not None and y != self.scroll_y:
            self._disengage_follow_tail()
        super()._on_scroll_to(message)

    def action_scroll_up(self) -> None:
        self._disengage_follow_tail_for_target(self.scroll_target_y - 1)
        super().action_scroll_up()

    def action_scroll_down(self) -> None:
        self._disengage_follow_tail_for_target(self.scroll_target_y + 1)
        super().action_scroll_down()

    def action_page_up(self) -> None:
        self._disengage_follow_tail_for_target(
            self.scroll_y - self.scrollable_content_region.height
        )
        super().action_page_up()

    def action_page_down(self) -> None:
        self._disengage_follow_tail_for_target(
            self.scroll_y + self.scrollable_content_region.height
        )
        super().action_page_down()

    def action_scroll_home(self) -> None:
        self._disengage_follow_tail_for_target(0)
        super().action_scroll_home()

    def action_scroll_end(self) -> None:
        super().action_scroll_end()
        self.follow_tail = True


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

    def __init__(self, *, id: str | None = None) -> None:
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

    def action_cursor_up(self, select: bool = False) -> None:
        if not select:
            handler = getattr(self.app, "_handle_composer_history_key", None)
            if callable(handler) and handler(self, "up"):
                return
        super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
        if not select:
            handler = getattr(self.app, "_handle_composer_history_key", None)
            if callable(handler) and handler(self, "down"):
                return
        super().action_cursor_down(select)

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
    tmp_path.replace(path)


def _event_offset(event: dict[str, Any] | None) -> int | None:
    if not event:
        return None
    value = event.get("_jsonl_offset")
    return value if isinstance(value, int) else None


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
    }

    TranscriptCell.assistant {
        background: #101a17;
        color: #e5e7eb;
    }

    TranscriptCell.event {
        background: #0e141b;
        color: #aeb7c4;
    }

    TranscriptCell.reasoning {
        background: #0c1219;
        color: #9aa6b6;
        text-style: italic;
    }

    TranscriptCell.process_fold {
        background: #0c1219;
        color: #aeb7c4;
    }

    TranscriptCell.process_fold_hidden {
        display: none;
    }

    TranscriptCell.tool_pending {
        background: #0e141b;
        color: #c8d2e0;
    }

    TranscriptCell.error {
        background: #241316;
        color: #ffb4b4;
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
        return self._with_content_offsets(self._highlight_selection(strip, y), y)

    def _with_content_offsets(self, strip: Strip, y: int) -> Strip:
        offset_x = 0
        segments = []
        for segment in strip:
            if segment.control:
                segments.append(segment)
                continue
            text = segment.text
            style = segment.style
            if text:
                style = (style or Style()) + Style(meta={"offset": (offset_x, y)})
            segments.append(Segment(text, style, segment.control))
            offset_x += len(text)
        return Strip(segments, strip.cell_length)

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
        line_text = strip.text
        start = self._character_offset_to_cell(line_text, start)
        end = self._character_offset_to_cell(line_text, end)
        start = max(0, min(start, strip.cell_length))
        end = max(start, min(end, strip.cell_length))
        if start == end:
            return strip
        before = strip.crop(0, start)
        selected = self._apply_selection_style(strip.crop(start, end))
        after = strip.crop(end, strip.cell_length)
        return Strip.join([before, selected, after])

    def _character_offset_to_cell(self, text: str, offset: int) -> int:
        offset = max(0, min(offset, len(text)))
        return cell_len(text[:offset])

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


class ExpandableTranscriptCell(TranscriptCell, can_focus=True):
    """Transcript cell that opens hidden details in a panel."""

    def __init__(
        self,
        summary: str,
        details: str,
        detail_title: str = "tool_details",
        detail_hint: str = "tool_details_hint",
        **kwargs: Any,
    ) -> None:
        self.summary = summary
        self.details = details
        self.detail_title = detail_title
        self.detail_hint = detail_hint
        super().__init__(self._content(), **kwargs)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.open_details()

    def on_key(self, event: events.Key) -> None:
        app = self.app
        if event.key in {"enter", "space"}:
            event.stop()
            self.open_details()
        elif event.key == "j" and hasattr(app, "_focus_relative_expandable_cell"):
            event.stop()
            app._focus_relative_expandable_cell(self, 1)
        elif event.key == "k" and hasattr(app, "_focus_relative_expandable_cell"):
            event.stop()
            app._focus_relative_expandable_cell(self, -1)
        elif event.key == "escape" and hasattr(app, "action_focus_composer"):
            event.stop()
            app.action_focus_composer()

    def set_details(self, summary: str, details: str) -> None:
        self.summary = summary
        self.details = details
        self.update(self._content())

    def open_details(self) -> None:
        app = self.app
        if hasattr(app, "_open_tool_details_panel"):
            app._open_tool_details_panel(self)

    def _content(self) -> str:
        lines = self.summary.splitlines() or [""]
        lines[0] = f"{lines[0]} [dim][details][/dim]"
        return "\n".join(lines)


class FoldedProcessCell(TranscriptCell, can_focus=True):
    """A transcript-level fold that reveals the original in-between cells."""

    def __init__(
        self,
        cells: list[TranscriptCell],
        *,
        collapsed: bool = True,
        **kwargs: Any,
    ) -> None:
        self.cells = list(cells)
        self.collapsed = collapsed
        super().__init__("", **kwargs)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.toggle()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape" and hasattr(self.app, "action_focus_composer"):
            event.stop()
            self.app.action_focus_composer()

    def set_cells(self, cells: list[TranscriptCell]) -> None:
        self.cells = list(cells)
        self._apply_visibility()
        self._refresh()

    def set_collapsed(self, collapsed: bool) -> None:
        self.collapsed = collapsed
        self._apply_visibility()
        self._refresh()
        try:
            self.app._process_fold_toggled(self, collapsed)
        except Exception:
            pass

    def toggle(self) -> None:
        self.set_collapsed(not self.collapsed)

    def on_mount(self) -> None:
        self._apply_visibility()
        self._refresh()

    def _apply_visibility(self) -> None:
        for cell in self.cells:
            try:
                if self.collapsed:
                    cell.add_class("process_fold_hidden")
                else:
                    cell.remove_class("process_fold_hidden")
            except Exception:
                continue

    def _refresh(self) -> None:
        try:
            text = getattr(self.app, "_text", lambda key: key)
        except Exception:
            text = lambda key: key
        count = len(self.cells)
        key = "process_fold_collapsed" if self.collapsed else "process_fold_expanded"
        state = text(key)
        step_label = text("process_fold_step" if count == 1 else "process_fold_steps")
        hint = text("process_fold_expand_hint" if self.collapsed else "process_fold_collapse_hint")
        self.update(
            f"[dim]{escape(state)} · {count} {escape(step_label)}[/dim] "
            f"[dim]{escape(hint)}[/dim]"
        )


class ImageAttachmentCell(TranscriptCell, can_focus=True):
    """Transcript cell that opens image attachments in the preview panel."""

    def __init__(self, attachment: dict[str, Any], **kwargs: Any) -> None:
        self.attachment = attachment
        super().__init__("", **kwargs)

    def on_mount(self) -> None:
        self._refresh_content()

    def _refresh_content(self) -> None:
        text = getattr(self.app, "_text", lambda key: key)
        self.update(image_attachment_markup(self.attachment, label=text("image_attached")))

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.open_preview()

    def on_key(self, event: events.Key) -> None:
        app = self.app
        if event.key in {"enter", "space"}:
            event.stop()
            self.open_preview()
        elif event.key == "j" and hasattr(app, "_focus_relative_image_cell"):
            event.stop()
            app._focus_relative_image_cell(self, 1)
        elif event.key == "k" and hasattr(app, "_focus_relative_image_cell"):
            event.stop()
            app._focus_relative_image_cell(self, -1)
        elif event.key == "escape" and hasattr(app, "action_focus_composer"):
            event.stop()
            app.action_focus_composer()

    def open_preview(self) -> None:
        app = self.app
        if hasattr(app, "_open_image_preview_for_cell"):
            app._open_image_preview_for_cell(self)


class LoadOlderHistoryCell(TranscriptCell, can_focus=True):
    """Transcript cell that pages in older events for the active thread."""

    def __init__(self, *, has_more: bool, **kwargs: Any) -> None:
        self.has_more = has_more
        super().__init__("", **kwargs)

    def on_mount(self) -> None:
        self._refresh_content()

    def _refresh_content(self) -> None:
        text = getattr(self.app, "_text", lambda key: key)
        label = text("load_older_history") if self.has_more else text("history_start")
        self.update(f"[dim]{escape(label)}[/dim]")

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.load_more()

    def on_key(self, event: events.Key) -> None:
        if event.key in {"enter", "space"}:
            event.stop()
            self.load_more()

    def load_more(self) -> None:
        if self.has_more and hasattr(self.app, "_load_older_thread_history"):
            self.app._load_older_thread_history()


class ToolDetailsPanel(FullscreenPanel):
    """Full-screen tool detail panel with j/k navigation between tool results."""

    BINDINGS = [
        Binding("j", "next_detail", "Next", priority=True, show=False),
        Binding("k", "previous_detail", "Previous", priority=True, show=False),
        Binding("ctrl+d", "dismiss_panel", "Close", priority=True, show=False),
        *FullscreenPanel.BINDINGS,
    ]

    def __init__(self, cell: ExpandableTranscriptCell) -> None:
        self.current_cell = cell
        super().__init__(title="", body=cell.details, subtitle="")

    def on_mount(self) -> None:
        self._refresh_current()
        try:
            self.query_one("#panel-body", VerticalScroll).focus()
        except NoMatches:
            pass

    def on_key(self, event: events.Key) -> None:
        if event.key == "j":
            event.stop()
            self.action_next_detail()
            return
        if event.key == "k":
            event.stop()
            self.action_previous_detail()
            return
        if event.key == "ctrl+d":
            event.stop()
            self.action_dismiss_panel()
            return
        super().on_key(event)

    def action_next_detail(self) -> None:
        self._move(1)

    def action_previous_detail(self) -> None:
        self._move(-1)

    def _move(self, step: int) -> None:
        app = self.app
        if not hasattr(app, "_relative_expandable_cell"):
            return
        self.current_cell = app._relative_expandable_cell(self.current_cell, step)
        self._refresh_current()

    def _refresh_current(self) -> None:
        text = getattr(self.app, "_text", lambda key: key)
        self.panel_title = text(self.current_cell.detail_title)
        self.subtitle = text(self.current_cell.detail_hint)
        self.body = self.current_cell.details
        try:
            self.query_one("#panel-header", Static).update(self.panel_title)
            self.query_one("#panel-subtitle", Static).update(self.subtitle)
            self.query_one("#panel-body-content", Static).update(self.body)
            self.query_one("#panel-body", VerticalScroll).scroll_to(y=0, animate=False)
        except NoMatches:
            pass


class ImagePreviewPanel(FullscreenPanel):
    """Full-screen image attachment panel with j/k navigation."""

    BINDINGS = [
        Binding("j", "next_image", "Next", priority=True, show=False),
        Binding("k", "previous_image", "Previous", priority=True, show=False),
        Binding("right", "next_image", "Next", priority=True, show=False),
        Binding("down", "next_image", "Next", priority=True, show=False),
        Binding("left", "previous_image", "Previous", priority=True, show=False),
        Binding("up", "previous_image", "Previous", priority=True, show=False),
        Binding("f3", "dismiss_panel", "Close", priority=True, show=False),
        *FullscreenPanel.BINDINGS,
    ]

    def __init__(self, attachments: list[dict[str, Any]], index: int = 0) -> None:
        self.attachments = attachments
        self.index = max(0, min(index, len(attachments) - 1)) if attachments else 0
        super().__init__(title="", body="", subtitle="")

    def on_mount(self) -> None:
        self._refresh_current()
        try:
            self.query_one("#panel-body", VerticalScroll).focus()
        except NoMatches:
            pass

    def on_key(self, event: events.Key) -> None:
        if event.key in {"j", "right", "down"}:
            event.stop()
            self.action_next_image()
            return
        if event.key in {"k", "left", "up"}:
            event.stop()
            self.action_previous_image()
            return
        if event.key == "f3":
            event.stop()
            self.action_dismiss_panel()
            return
        super().on_key(event)

    def action_next_image(self) -> None:
        self._move(1)

    def action_previous_image(self) -> None:
        self._move(-1)

    def _move(self, step: int) -> None:
        if not self.attachments:
            return
        self.index = (self.index + step) % len(self.attachments)
        self._refresh_current()

    def _refresh_current(self) -> None:
        text = getattr(self.app, "_text", lambda key: key)
        self.panel_title = text("image_preview")
        self.subtitle = text("image_preview_hint")
        self.body = self._attachment_markup()
        try:
            self.query_one("#panel-header", Static).update(self.panel_title)
            self.query_one("#panel-subtitle", Static).update(self.subtitle)
            self.query_one("#panel-body-content", Static).update(self.body)
            self.query_one("#panel-body", VerticalScroll).scroll_to(y=0, animate=False)
        except NoMatches:
            pass

    def _attachment_markup(self) -> str:
        text = getattr(self.app, "_text", lambda key: key)
        if not self.attachments:
            return f"[dim]{escape(text('no_images'))}[/dim]"
        attachment = self.attachments[self.index]
        path = Path(str(attachment.get("stored_path") or ""))
        source = str(attachment.get("source_path") or "")
        note = str(attachment.get("note") or "").strip()
        size = int(attachment.get("size_bytes") or 0)
        lines = [
            f"[bold]{self.index + 1}/{len(self.attachments)}[/bold] "
            f"[cyan]{escape(path.name or str(path))}[/cyan]",
            "",
            f"- {escape(text('image_path'))}: [cyan]{escape(str(path))}[/cyan]",
            f"- {escape(text('image_mime'))}: {escape(str(attachment.get('mime_type') or ''))}",
            f"- {escape(text('image_size'))}: {format_tokens(size)}B",
        ]
        if source:
            lines.append(f"- {escape(text('image_source'))}: {escape(source)}")
        if note:
            lines.append(f"- {escape(text('image_note'))}: {escape(note)}")
        preview = image_ascii_preview(path)
        if preview:
            lines.extend(["", preview])
        lines.append("")
        lines.append(f"[dim]{escape(text('image_open_hint'))}[/dim]")
        return "\n".join(lines)


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
        scrollbar-size-vertical: 1;
        scrollbar-background: #0b0f14;
        scrollbar-background-hover: #0b0f14;
        scrollbar-background-active: #0b0f14;
        scrollbar-color: #2b3542;
        scrollbar-color-hover: #3a4a60;
        scrollbar-color-active: #7dd3fc;
        scrollbar-corner-color: #0b0f14;
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
        height: auto;
        color: #8fa2b8;
        padding: 0 1;
        background: #0b0f14;
    }

    #scroll-to-bottom-bar {
        height: auto;
        align: right top;
        background: #0b0f14;
        padding: 0 1;
    }

    #scroll-to-bottom-btn {
        width: auto;
        height: 1;
        color: #7dd3fc;
        background: #1a2330;
        padding: 0 2;
        text-style: bold;
    }

    #scroll-to-bottom-btn:hover {
        background: #2b3542;
        color: #ffffff;
    }

    #scroll-to-bottom-bar.hidden {
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
        Binding("ctrl+g", "toggle_visible_process_folds", "Process", priority=True),
        Binding("tab", "toggle_composer_height", "Height", priority=True),
        Binding("ctrl+s", "toggle_status_panel", "Status", priority=True),
        Binding("ctrl+o", "open_threads", "Threads", priority=True),
        Binding("ctrl+p", "open_command_palette", "Commands", priority=True),
        Binding("ctrl+d", "toggle_tool_details", "Details", priority=True),
        Binding("f2", "attach_clipboard_image", "Attach image", priority=True),
        Binding("f3", "preview_images", "Images", priority=True),
        Binding("ctrl+c", "interrupt_turn", "Interrupt", priority=True, show=False),
        Binding("enter", "focus_composer", "Focus composer", priority=True, show=False),
        Binding("f1", "help", "Help", priority=True),
        Binding("escape", "clear_input", "Clear"),
    ]

    busy = reactive(False)

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

    def compose(self) -> ComposeResult:
        with Vertical(id="main-column"):
            with TranscriptScroll(id="transcript"):
                yield EmptyState()
            with Vertical(id="bottom-pane"):
                with Vertical(id="scroll-to-bottom-bar", classes="hidden"):
                    yield Static(
                        f"↓ {tr(self.language, 'back_to_bottom')}",
                        id="scroll-to-bottom-btn",
                    )
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
        self.query_one(EmptyState).tick()
        self._refresh_status(self._text("idle"))
        self.set_interval(0.16, self._tick, name="spinner")
        self.query_one("#composer", TextArea).focus()
        transcript = self.query_one("#transcript", TranscriptScroll)
        self.watch(transcript, "near_bottom", self._on_near_bottom_changed)

    def on_unmount(self) -> None:
        self._mention_file_watcher_stop.set()
        if self._mention_file_watcher_worker is not None:
            self._mention_file_watcher_worker.cancel()

    def _on_near_bottom_changed(self, near: bool) -> None:
        try:
            bar = self.query_one("#scroll-to-bottom-bar", Vertical)
        except NoMatches:
            return
        if near:
            bar.add_class("hidden")
        else:
            bar.remove_class("hidden")

    def on_click(self, event: events.Click) -> None:
        widget = getattr(event, "widget", None)
        if widget is not None and widget.id == "scroll-to-bottom-btn":
            event.stop()
            try:
                transcript = self.query_one("#transcript", TranscriptScroll)
            except NoMatches:
                return
            transcript.engage_follow_tail()

    def on_resize(self) -> None:
        self._refresh_status()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "composer":
            return
        previous = self._last_composer_text
        current = event.text_area.text
        if self._composer_history_index is not None and current != self._composer_history_text():
            self._reset_composer_history_navigation()
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

    def _tick(self) -> None:
        if not self._transcript_has_content:
            try:
                self.query_one(EmptyState).tick()
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
        pending_images = list(self._pending_images)
        if not prompt and not pending_images:
            self._flash(self._text("write_first"))
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
        self._resize_composer("")
        if not prompt:
            prompt = self._text("image_only_prompt")
        if "\n" not in prompt and self._handle_command(prompt):
            return
        image_paths = [image.path for image in pending_images]
        self._pending_images.clear()
        self._refresh_pending_images()
        active_run = self._active_run_state()
        if active_run is not None:
            run_state = active_run
            run_state.queue.append(QueuedTurn(prompt=prompt, image_paths=image_paths))
            if self._is_active_thread(run_state.thread_id):
                self._append_cell(
                    self._queued_turn_markup(prompt, image_paths),
                    "event",
                )
            self._refresh_status()
            return
        self._start_turn(prompt, image_paths=image_paths)

    def _start_turn(self, prompt: str, *, image_paths: list[Path] | None = None) -> None:
        if self.thread_id is None:
            self.thread_id = self.engine.thread_store.create_thread()
        self._start_background_turn(self.thread_id, prompt, image_paths=image_paths)

    def _start_background_turn(
        self,
        thread_id: str,
        prompt: str,
        *,
        image_paths: list[Path] | None = None,
        queue: list[QueuedTurn] | None = None,
    ) -> None:
        cancel_event = asyncio.Event()
        run_state = ThreadRunState(
            thread_id=thread_id,
            worker=None,
            cancel_event=cancel_event,
            queue=list(queue or []),
            status=self._text("working"),
        )
        self._thread_runs[thread_id] = run_state
        if self._is_active_thread(thread_id):
            self.query_one("#composer-shell", Vertical).add_class("busy")
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
            self._interrupt_armed = False
            self._process_anchor_cell = self._append_user(prompt)
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
            self._refresh_status(self._text("working"))
        worker = self.run_worker(
            self._run_turn(prompt, thread_id, image_paths=list(image_paths or [])),
            exclusive=False,
            thread=False,
        )
        run_state.worker = worker
        if self._is_active_thread(thread_id):
            self._current_worker = worker
            self._current_cancel_event = cancel_event
        self._refresh_active_run_state()

    async def _run_turn(self, prompt: str, thread_id: str, *, image_paths: list[Path]) -> None:
        run_state = self._thread_runs[thread_id]
        try:
            turn_kwargs: dict[str, Any] = {
                "user_text": prompt,
                "thread_id": thread_id,
                "level": self.level,
                "cancel_event": run_state.cancel_event,
            }
            if image_paths:
                turn_kwargs["image_paths"] = image_paths
            async for item in self.engine.run_turn(**turn_kwargs):
                item_thread_id = str(item.get("thread_id") or thread_id)
                event_type = item["type"]
                if event_type == "image.attachment":
                    await self._handle_thread_event(item_thread_id, "image.attachment", item, run_state)
                elif event_type == "assistant.delta":
                    await self._handle_thread_event(item_thread_id, "assistant.delta", item, run_state)
                elif event_type == "assistant.reasoning_delta":
                    await self._handle_thread_event(item_thread_id, "assistant.reasoning_delta", item, run_state)
                elif event_type == "tool.delta":
                    await self._handle_thread_event(item_thread_id, "tool.delta", item, run_state)
                elif event_type == "model.response":
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
                            {"text": reasoning_text},
                            run_state,
                        )
                    else:
                        await self._handle_thread_event(
                            item_thread_id,
                            "assistant.reasoning_absent",
                            {},
                            run_state,
                        )
                    output = list(getattr(response, "output", []) or [])
                    has_tool_call = any(entry.get("type") == "function_call" for entry in output)
                    if has_tool_call:
                        await self._handle_thread_event(
                            item_thread_id,
                            "assistant.response_with_tools",
                            {},
                            run_state,
                        )
                    else:
                        await self._handle_thread_event(
                            item_thread_id,
                            "assistant.final_response_started",
                            {},
                            run_state,
                        )
                    run_state.status = self._text("reading")
                    if self._is_active_thread(item_thread_id):
                        self._refresh_status(self._text("reading"))
                elif event_type == "thread.title":
                    if self._is_active_thread(item_thread_id):
                        self._refresh_status(self._text("idle"))
                elif event_type == "tool.started":
                    await self._handle_thread_event(item_thread_id, "tool.started", item, run_state)
                elif event_type == "tool.output":
                    await self._handle_thread_event(item_thread_id, "tool.output", item, run_state)
                elif event_type == "compaction.completed":
                    if self._is_active_thread(item_thread_id):
                        self._append_cell(f"[dim]{escape(self._text('compacted'))}[/dim]", "event")
                elif event_type == "turn.completed":
                    text = item["final_text"] or run_state.assistant_buffer
                    if text and self._is_active_thread(item_thread_id) and self._assistant_cell is None:
                        await self._append_assistant_delta(text)
                        self._sync_run_state_from_active(run_state)
                    run_state.status = self._text("idle")
                    if self._is_active_thread(item_thread_id):
                        self._refresh_status(self._text("idle"))
                elif event_type == "turn.interrupted":
                    run_state.status = self._text("interrupted")
                    if self._is_active_thread(item_thread_id):
                        self._append_cell(f"[dim]{escape(self._text('interrupted'))}[/dim]", "event")
                        self._refresh_status(self._text("interrupted"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            run_state.status = self._text("error")
            if self._is_active_thread(thread_id):
                self._append_cell(error_markup(format_error(exc)), "error")
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
                    image_paths=next_turn.image_paths,
                    queue=remaining_queue,
                )
            else:
                self._thread_runs.pop(thread_id, None)
                if self._is_active_thread(thread_id):
                    self._current_worker = None
                    self._current_cancel_event = None
                    self._interrupt_armed = False
                    if self._last_status != self._text("error"):
                        self._refresh_status(self._text("idle"))
                    self.query_one("#composer", TextArea).focus()
                self._refresh_active_run_state()

    def action_clear_input(self) -> None:
        composer = self.query_one("#composer", TextArea)
        if composer.text:
            self._reset_composer_history_navigation()
            composer.load_text("")
            self._last_composer_text = ""
            self._composer_height_override = None
            self._resize_composer("")
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
        self._resize_composer(text)
        self._refresh_status()

    def _reset_composer_history_navigation(self) -> None:
        self._composer_history_index = None
        self._composer_history_draft = ""

    def _active_run_state(self) -> ThreadRunState | None:
        if self.thread_id is None:
            return None
        return self._thread_runs.get(self.thread_id)

    def _active_queue_length(self) -> int:
        run_state = self._active_run_state()
        return len(run_state.queue) if run_state is not None else 0

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

    def _is_active_thread(self, thread_id: str | None) -> bool:
        return bool(thread_id and self.thread_id == thread_id)

    def _refresh_active_run_state(self) -> None:
        run_state = self._active_run_state()
        self.busy = run_state is not None
        shell = self.query_one("#composer-shell", Vertical)
        if run_state is None:
            shell.remove_class("busy")
            self._current_worker = None
            self._current_cancel_event = None
            return
        shell.add_class("busy")
        self._current_worker = run_state.worker
        self._current_cancel_event = run_state.cancel_event

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

    async def _handle_thread_event(
        self,
        thread_id: str,
        event_type: str,
        item: dict[str, Any],
        run_state: ThreadRunState,
    ) -> None:
        if not self._is_active_thread(thread_id):
            if event_type == "assistant.delta":
                run_state.assistant_buffer += str(item.get("text") or "")
            elif event_type == "assistant.reasoning_delta":
                stripped = str(item.get("text") or "").strip()
                if stripped:
                    run_state.reasoning_buffer = (run_state.reasoning_buffer + " " + stripped).strip()
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
                run_state.status = self._text("running_python")
            elif event_type == "tool.started":
                run_state.reasoning_buffer = ""
                run_state.reasoning_cell = None
            elif event_type == "assistant.final_response_started" and run_state.process_cells:
                run_state.process_collapsed = True
            return
        self._sync_active_from_run_state(run_state)
        if event_type == "assistant.delta":
            await self._append_assistant_delta(str(item.get("text") or ""))
        elif event_type == "assistant.reasoning_delta":
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
            self._track_current_assistant_cell_as_process()
            self._seal_assistant_round()
        elif event_type == "assistant.final_response_started":
            self._collapse_process_cells()
        elif event_type == "tool.started":
            self._clear_pending_reasoning()
            self._seal_assistant_round()
            self._append_tool_started(item)
        elif event_type == "tool.output":
            self._append_tool_output(item)
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
        try:
            image = save_clipboard_image(project_state_dir(self.project_root) / "clipboard")
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
            if region.y < viewport_bottom and region.y + region.height > viewport_top:
                folds.append(child)
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

    def _refresh_pending_images(self) -> None:
        try:
            meta = self.query_one("#composer-meta", Static)
        except NoMatches:
            return
        if not self._pending_images:
            meta.update("")
            return
        labels = [
            f"{image.path.name} {image.width}x{image.height}"
            for image in self._pending_images
        ]
        meta.update(
            f"[dim]{escape(self._text('pending_images'))}: "
            f"{escape(', '.join(labels))}[/dim]"
        )

    def _queued_turn_markup(self, prompt: str, image_paths: list[Path]) -> str:
        suffix = ""
        if image_paths:
            suffix = "\n" + f"[dim]+{len(image_paths)} {escape(self._text('images'))}[/dim]"
        return f"[dim]{escape(self._text('queued'))}[/dim]\n{escape(prompt)}{suffix}"

    def _handle_command(self, prompt: str) -> bool:
        command, _, rest = prompt.partition(" ")
        if command == "/clear":
            self._close_active_panel()
            self.thread_id = None
            self._assistant_buffer = ""
            self._assistant_cell = None
            self._tool_cells.clear()
            self._tool_delta_cells.clear()
            self._pending_images.clear()
            active_run = self._active_run_state()
            if active_run is not None:
                active_run.queue.clear()
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
            f"- [cyan]Ctrl+P / /[/cyan] [dim]{escape(self._text('help_commands'))}[/dim]",
            f"- [cyan]F1 / ?[/cyan] [dim]{escape(self._text('help_help'))}[/dim]",
            f"- [cyan]Ctrl+O[/cyan] [dim]{escape(self._text('help_threads'))}[/dim]",
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
        lines = [f"[bold]{escape(self._text('commands'))}[/bold] [dim](Ctrl+P, F1, Esc)[/dim]"]
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

    async def _append_assistant_delta(self, text: str) -> None:
        self._assistant_buffer += text
        if self._assistant_cell is None:
            self._mark_transcript_content()
            self._assistant_cell = TranscriptCell(classes="assistant")
            self.query_one("#transcript", VerticalScroll).mount(self._assistant_cell)
        self._assistant_cell.update(Markdown(self._assistant_buffer), copy_text=self._assistant_buffer)
        self._scroll_end()

    def _seal_assistant_round(self) -> None:
        self._assistant_buffer = ""
        self._assistant_cell = None

    def _append_reasoning_delta(self, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            return
        self._reasoning_buffer = (self._reasoning_buffer + " " + stripped).strip()
        first = self._reasoning_buffer.splitlines()[0]
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
        if not stripped:
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
            classes="process_fold",
            markup=True,
        )
        self.query_one("#transcript", VerticalScroll).mount(cell, before=insert_before)
        self._process_fold_cell = cell
        self._scroll_end()
        return cell

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
        if pending_cell is not None and call is not None:
            markup = self._tool_pending_markup(
                str(call.get("name") or "python"),
                "",
                call={**call, "_status_label": self._text("python_called")},
            )
            details = tool_call_detail_highlight_markup(call)
            if isinstance(pending_cell, ExpandableTranscriptCell):
                pending_cell.set_details(markup, details)
            elif tool_call_preview_line(call):
                new_cell = self._replace_with_expandable_cell(pending_cell, markup, details, "tool_pending")
                self._replace_process_cell(pending_cell, new_cell)
            else:
                pending_cell.update(markup)
        if payload is None:
            markup = f"[dim]{escape(self._text('python'))} {escape(self._text('python_completed'))}[/dim]"
            cell = self._append_cell(markup, "event")
            self._track_process_cell(cell)
            return

        self._last_tool_payload = payload
        markup = tool_timeline_markup(payload)
        details = tool_detail_markup(payload)
        cell = self._append_expandable_cell(markup, details, "event")
        self._track_process_cell(cell)
        self._refresh_status(self._text("working"))

    def _append_tool_started(self, item: dict[str, Any]) -> None:
        call = item.get("call") or {}
        delta_index = item.get("tool_call_index")
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

    def _append_tool_delta(self, item: dict[str, Any]) -> None:
        delta = item.get("tool_call")
        index = int(getattr(delta, "index", 0))
        name = str(getattr(delta, "name", None) or "python")
        call = {
            "call_id": getattr(delta, "call_id", "") or "",
            "name": name,
            "arguments": getattr(delta, "arguments", "") or getattr(delta, "arguments_delta", ""),
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
        self._refresh_status(self._text("running_python"))

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
        for event in events:
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
            )

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
                        mounted.append(self._append_cell(Markdown(text), "assistant", before=before, copy_text=text))
                elif item_type == "function_call":
                    mounted.append(self._append_tool_call_history(item, before=before))
        elif event_type == "item.runner_result":
            result = event.get("result") or {}
            self._last_tool_payload = result
            mounted.append(
                self._append_expandable_cell(
                    tool_timeline_markup(result),
                    tool_detail_markup(result),
                    "event",
                    before=before,
                )
            )
        elif event_type == "item.image_attachment":
            attachment = event.get("attachment") or {}
            mounted.append(self._append_image_attachment_cell(attachment, before=before))
        elif event_type in {"item.reasoning_delta", "item.reasoning_partial"}:
            cell = self._append_reasoning_history(str(event.get("text") or ""), before=before)
            if cell is not None:
                mounted.append(cell)
        elif event_type == "item.compaction":
            mounted.append(self._append_cell(f"[dim]{escape(self._text('compacted'))}[/dim]", "event", before=before))
        return mounted

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
            running = f" · {self._text('working')}" if thread_id in self._thread_runs else ""
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

    def _maybe_open_mention_picker(self, composer: TextArea, *, previous: str, current: str) -> None:
        if len(current) <= len(previous):
            return
        trigger = self._mention_trigger_at_cursor(composer)
        if trigger is None:
            return
        inserted = current[len(previous) :]
        if not inserted or not trigger.endswith(inserted):
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
        expected_triggers = {"thread": ("@@",), "file": ("@",)}.get(kind, ("@",))
        if self._mention_trigger_at_cursor(composer) not in expected_triggers:
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
        self._start_file_mention_scan()

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
        self._start_thread_mention_scan()

    def _mention_picker_items(self, kind: str) -> tuple[str, list[PickerItem], str]:
        if kind == "thread":
            return (
                self._text("mention_threads"),
                self._mention_thread_cache.items,
                self._mention_cache_subtitle("thread"),
            )
        return (
            self._text("mention_files"),
            self._mention_file_cache.items,
            self._mention_cache_subtitle("file"),
        )

    def _mention_cache_subtitle(self, kind: str) -> str:
        if kind == "thread":
            hint = self._text("mention_threads_hint")
            cache = self._mention_thread_cache
        else:
            hint = self._text("mention_files_hint")
            cache = self._mention_file_cache
        if cache.worker is not None and not cache.complete:
            return f"{hint} · {self._text('mention_scanning')}"
        if cache.complete:
            return f"{hint} · {self._text('mention_cached')}"
        return hint

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
        return list(self._iter_file_mention_items(self.project_root.resolve(), generation=None))

    def _start_file_mention_scan(self) -> None:
        cache = self._mention_file_cache
        if cache.complete and not self._mention_file_cache_dirty:
            self._refresh_active_mention_panel("file", cache.generation)
            return
        if cache.worker is not None and not cache.complete:
            if not cache.worker.is_finished:
                return
            cache.worker = None
        cache.generation += 1
        cache.complete = False
        self._mention_file_cache_dirty = False
        generation = cache.generation
        cache.worker = self.run_worker(
            lambda: self._scan_file_mentions_worker(generation),
            name="mention-files",
            group="mention-files",
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )
        self._refresh_active_mention_panel("file", generation)

    def _scan_file_mentions_worker(self, generation: int) -> None:
        root = self.project_root.resolve()
        items: list[PickerItem] = []
        pending: list[PickerItem] = []
        try:
            for item in self._iter_file_mention_items(root, generation=generation):
                if generation != self._mention_file_cache.generation:
                    return
                items.append(item)
                pending.append(item)
                if len(pending) >= MENTION_SCAN_BATCH_SIZE:
                    batch = list(items)
                    self.call_from_thread(self._apply_file_mention_scan_update, generation, batch, False)
                    pending.clear()
        finally:
            self.call_from_thread(self._apply_file_mention_scan_update, generation, items, True)

    def _apply_file_mention_scan_update(self, generation: int, items: list[PickerItem], complete: bool) -> None:
        cache = self._mention_file_cache
        if generation != cache.generation:
            return
        cache.items = list(items)
        cache.complete = complete
        if complete:
            cache.worker = None
            self._start_file_mention_watcher()
        self._refresh_active_mention_panel("file", generation)

    def _start_file_mention_watcher(self) -> None:
        worker = self._mention_file_watcher_worker
        if worker is not None and not worker.is_finished:
            return
        self._mention_file_watcher_stop.clear()
        self._mention_file_watcher_worker = self.run_worker(
            self._watch_file_mentions_worker,
            name="mention-file-watch",
            group="mention-file-watch",
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )

    def _watch_file_mentions_worker(self) -> None:
        root = self.project_root.resolve()
        for changes in watch(
            root,
            watch_filter=self._mention_watch_filter,
            debounce=MENTION_WATCH_DEBOUNCE_MS,
            step=MENTION_WATCH_POLL_DELAY_MS,
            recursive=True,
            ignore_permission_denied=True,
            stop_event=self._mention_file_watcher_stop,
        ):
            if not changes:
                continue
            self.call_from_thread(self._mark_file_mention_cache_dirty)

    def _mention_watch_filter(self, change: Change, path: str) -> bool:
        try:
            relative_parts = Path(path).resolve().relative_to(self.project_root.resolve()).parts
        except (OSError, ValueError):
            return False
        for part in relative_parts[:-1]:
            if part.startswith(".") or part in IGNORED_MENTION_DIRS:
                return False
        name = relative_parts[-1] if relative_parts else ""
        if name.startswith("."):
            return True
        suffix = Path(name).suffix.lower()
        return not suffix or suffix in CODE_FILE_SUFFIXES

    def _mark_file_mention_cache_dirty(self) -> None:
        cache = self._mention_file_cache
        if cache.worker is not None and not cache.worker.is_finished:
            return
        self._mention_file_cache_dirty = True

    def _start_thread_mention_scan(self) -> None:
        cache = self._mention_thread_cache
        if cache.complete:
            self._refresh_active_mention_panel("thread", cache.generation)
            return
        if cache.worker is not None and not cache.complete:
            if not cache.worker.is_finished:
                return
            cache.worker = None
        cache.generation += 1
        cache.complete = False
        generation = cache.generation
        cache.worker = self.run_worker(
            lambda: self._scan_thread_mentions_worker(generation),
            name="mention-threads",
            group="mention-threads",
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )
        self._refresh_active_mention_panel("thread", generation)

    def _start_mention_scan(self, kind: str) -> None:
        if kind == "thread":
            self._start_thread_mention_scan()
        elif kind == "file":
            self._start_file_mention_scan()

    def _scan_thread_mentions_worker(self, generation: int) -> None:
        items: list[PickerItem] = []
        try:
            items = self._thread_mention_items()
        finally:
            if generation != self._mention_thread_cache.generation:
                return
            self.call_from_thread(self._apply_thread_mention_scan_update, generation, items)

    def _apply_thread_mention_scan_update(self, generation: int, items: list[PickerItem]) -> None:
        cache = self._mention_thread_cache
        if generation != cache.generation:
            return
        cache.items = list(items)
        cache.complete = True
        cache.worker = None
        self._refresh_active_mention_panel("thread", generation)

    def _refresh_active_mention_panel(self, kind: str, generation: int) -> None:
        cache = self._mention_thread_cache if kind == "thread" else self._mention_file_cache
        if generation != cache.generation:
            return
        panel = self._active_fullscreen_panel()
        if panel is None or panel.mention_kind != kind:
            return
        panel.update_picker_items(cache.items, subtitle=self._mention_cache_subtitle(kind))

    def _iter_file_mention_items(self, root: Path, *, generation: int | None) -> Any:
        items: list[PickerItem] = []
        directories_seen = 0
        files_seen = 0
        stack = deque([root])
        while stack and len(items) < MAX_MENTION_ITEMS:
            directory = stack.popleft()
            if generation is not None and generation != self._mention_file_cache.generation:
                return
            if directories_seen >= MENTION_SCAN_DIRECTORY_LIMIT or files_seen >= MENTION_SCAN_FILE_LIMIT:
                if generation is None:
                    return
                yield PickerItem(
                    id="",
                    title=self._text("mention_scan_truncated"),
                    description=self._text("mention_scan_truncated_description"),
                )
                return
            try:
                with os.scandir(directory) as entries:
                    children = sorted(entries, key=lambda item: (not item.is_dir(follow_symlinks=False), item.name.casefold()))
            except OSError:
                continue
            directories_seen += 1
            for entry in children:
                if generation is not None and generation != self._mention_file_cache.generation:
                    return
                if len(items) >= MAX_MENTION_ITEMS:
                    break
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if is_dir:
                    path = Path(entry.path)
                    try:
                        relative = path.relative_to(root)
                    except ValueError:
                        continue
                    mention = relative.as_posix().rstrip("/") + "/"
                    item = PickerItem(
                        id=mention,
                        title=mention,
                        description=(
                            self._text("mention_dot_dir_skipped")
                            if entry.name.startswith(".")
                            else self._text("mention_directory_description")
                        ),
                    )
                    items.append(item)
                    yield item
                    if not entry.name.startswith(".") and entry.name not in IGNORED_MENTION_DIRS:
                        stack.append(path)
                    continue
                try:
                    is_file = entry.is_file(follow_symlinks=False)
                except OSError:
                    continue
                files_seen += 1
                if not is_file:
                    continue
                path = Path(entry.path)
                if path.suffix.lower() not in CODE_FILE_SUFFIXES:
                    continue
                try:
                    relative = path.relative_to(root)
                except ValueError:
                    continue
                mention = relative.as_posix()
                item = PickerItem(
                    id=mention,
                    title=mention,
                    description=self._text("mention_file_description"),
                )
                items.append(item)
                yield item
        return items

    def _mcp_mention_items(self) -> list[PickerItem]:
        items: list[PickerItem] = []
        for server in discover_mcp_servers(self.project_root):
            items.append(
                PickerItem(
                    id=server.name,
                    title=server.name,
                    description=server.description,
                    meta=f"{server.scope}" + (f" · {server.command}" if server.command else ""),
                )
            )
        return items

    def _skill_mention_items(self) -> list[PickerItem]:
        items: list[PickerItem] = []
        for skill in discover_skills(self.project_root):
            items.append(
                PickerItem(
                    id=skill.name,
                    title=skill.name,
                    description=skill.description,
                    meta=f"{skill.scope} · {skill.path}",
                )
            )
        return items

    def _choose_file_mention(self, path: str) -> None:
        self._insert_mention(f"@{path}", "@")

    def _choose_thread_mention(self, thread_id: str) -> None:
        self._insert_mention(f"@thread:{thread_id}", ("@@", "@"))

    def _choose_mcp_mention(self, name: str) -> None:
        self._insert_mention(f"@mcp:{name}", "")

    def _choose_skill_mention(self, name: str) -> None:
        self._insert_mention(f"@skill:{name}", "")

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
        level_name = self.level or self.engine.config.runtime.default_level
        rules = self.engine.project_rule_context()
        scripts = self.engine.runner.store.list_scripts(limit=5)
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
        script_line = (
            f"{len(scripts)} {self._text('status_scripts_saved')}"
            if scripts
            else self._text("no_scripts")
        )
        lines = [
            f"- state: [cyan]{escape(self._last_status)}[/cyan]",
            f"- version: [cyan]{escape(application_version())}[/cyan]",
            f"- level: [cyan]{escape(level_name)}[/cyan]",
            f"- model: {model_line}",
            f"- provider/api: {provider_line}",
            f"- context: {context_line}",
            f"- compaction: {compress_line}",
            f"- rules: {escape(rules_line)}",
            f"- scripts: {escape(script_line)}",
            f"- thread: {escape(short_thread(self.thread_id))}",
            f"- queued: {self._active_queue_length()}",
            f"- user state: {escape(str(uv_agent_home()))}",
            f"- project state: {escape(str(project_state_dir(self.project_root)))}",
            f"- host: {escape(host_environment_line())}",
            f"- language: {escape(self.language.name)}",
        ]
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
        if scripts:
            lines.append("")
            lines.append(f"[bold]{escape(self._text('scripts'))}[/bold]")
            for script in scripts:
                summary = str(script.get("summary") or "")
                if len(summary) > 96:
                    summary = summary[:93].rstrip() + "..."
                lines.append(
                    f"- {escape(str(script.get('script_id') or ''))}: {escape(summary)}"
                )
        return "\n".join(lines)

    def _open_config_panel(self) -> None:
        self.engine.refresh_config()
        default_level = self.engine.config.runtime.default_level
        items = [
            PickerItem(
                id="default_level",
                title=self._text("config_default_level"),
                description=default_level,
                meta=self._text("config_default_level_hint"),
            ),
            PickerItem(
                id="language",
                title=self._text("config_language"),
                description=self.engine.config.ui.language,
                meta=self._text("config_language_hint"),
            ),
            PickerItem(
                id="compression",
                title=self._text("config_compression"),
                description="on" if self.engine.config.runtime.compression.enabled else "off",
                meta=self._text("config_compression_hint"),
            ),
            PickerItem(
                id="sources",
                title=self._text("config_sources"),
                description=str(editable_config_path(self.project_root)),
                meta=self._text("config_sources_hint"),
            ),
            PickerItem(
                id="raw",
                title=self._text("config_raw"),
                description=self._text("config_raw_hint"),
            ),
        ]
        subtitle = (
            self._text("config_hint")
            + " · "
            + self._text("config_models_readonly_hint")
        )
        self._open_picker(
            self._text("config"),
            items,
            self._choose_config_item,
            subtitle=subtitle,
        )

    def _choose_config_item(self, item_id: str) -> None:
        if item_id == "default_level":
            self._open_default_level_panel()
        elif item_id == "language":
            self._open_language_panel()
        elif item_id == "compression":
            self._toggle_compression()
        elif item_id == "sources":
            self._open_config_sources_panel()
        elif item_id == "raw":
            self._open_config_raw_panel()

    def _close_active_panel(self) -> None:
        panel = self._active_fullscreen_panel()
        if panel is not None:
            panel.close_navigation()

    def _open_default_level_panel(self) -> None:
        items = []
        current = self.engine.config.runtime.default_level
        for name, level in self.engine.config.levels.items():
            marker = self._text("current") if name == current else ""
            items.append(
                PickerItem(
                    id=name,
                    title=name,
                    description=level.model,
                    meta=marker,
                )
            )
        self._open_picker(
            self._text("config_default_level"),
            items,
            self._set_default_level,
            subtitle=self._text("config_write_hint"),
        )

    def _open_current_level_panel(self) -> None:
        items = []
        current = self.level or self.engine.config.runtime.default_level
        for name, level in self.engine.config.levels.items():
            marker = self._text("current") if name == current else ""
            items.append(
                PickerItem(
                    id=name,
                    title=name,
                    description=level.model,
                    meta=marker,
                )
            )
        self._open_picker(
            self._text("config_current_level"),
            items,
            self._set_current_level,
            subtitle=self._text("config_session_hint"),
        )

    def _open_language_panel(self) -> None:
        current = self.engine.config.ui.language
        items = [
            PickerItem(id=value, title=label, description=self._text("current") if value == current else "")
            for value, label in (("auto", "auto"), ("en", "English"), ("zh-CN", "中文"))
        ]
        self._open_picker(
            self._text("config_language"),
            items,
            self._set_language,
            subtitle=self._text("config_write_hint"),
        )

    def _toggle_compression(self) -> None:
        current = self.engine.config.runtime.compression.enabled
        self._write_user_config_patch({"runtime": {"compression": {"enabled": not current}}})
        self._flash(
            f"{self._text('config_compression')}: {'on' if not current else 'off'}",
        )
        self._open_config_panel()

    def _open_config_sources_panel(self) -> None:
        sources = config_sources(self.project_root)
        lines = ["[bold]sources[/bold]"]
        for source in sources:
            exists = "yes" if source["exists"] else "no"
            lines.append(
                f"- {escape(source['scope'])}: {escape(source['path'])} [dim]exists={exists}[/dim]"
            )
        lines.append(f"\n[bold]editable[/bold]\n{escape(str(editable_config_path(self.project_root)))}")
        self._open_panel("\n".join(lines), "config", self._text("config_sources"))

    def _open_config_raw_panel(self) -> None:
        redacted = redact_config(load_raw_config(self.project_root))
        preview = json.dumps(redacted, ensure_ascii=False, indent=2)
        if len(preview) > 3200:
            preview = preview[:3200].rstrip() + "\n..."
        self._open_panel(escape(preview), "config", self._text("config_raw"))

    def _set_default_level(self, name: str) -> None:
        if name not in self.engine.config.levels:
            self._flash(f"{self._text('unknown_level')}: {name}", severity="error")
            return
        self._write_user_config_patch({"runtime": {"default_level": name}})
        self._flash(f"{self._text('config_default_level')}: {name}")
        if self.level is None:
            self._refresh_status()
        self._close_active_panel()

    def _set_current_level(self, name: str) -> None:
        self._handle_level_command(name)
        self._close_active_panel()

    def _set_language(self, value: str) -> None:
        self._write_user_config_patch({"ui": {"language": value}})
        self._flash(f"{self._text('config_language')}: {value}")
        self._close_active_panel()

    def _write_user_config_patch(self, patch: dict[str, Any]) -> None:
        path = editable_config_path(self.project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raw = {}
        else:
            raw = {}
        updated = self._config_deep_merge(raw, patch)
        path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.engine.config = load_config(self.project_root)
        self.engine.runner.config = self.engine.config.runner
        if hasattr(self.engine.model_client, "reload_config"):
            self.engine.model_client.reload_config(self.engine.config)  # type: ignore[attr-defined]
        self.language = detect_user_language(self.engine.config.ui.language)
        self._refresh_status()

    def _config_deep_merge(self, base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in patch.items():
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = self._config_deep_merge(current, value)
            else:
                merged[key] = value
        return merged

    def _open_models_panel(self) -> None:
        """Read-only models picker. Editing models lives in config.json."""
        self.engine.refresh_config()
        items: list[PickerItem] = []
        # Show which level each configured model is referenced by so users can
        # cross-reference without having to open config.json first.
        levels_by_model: dict[str, list[str]] = {}
        for level_name, level in self.engine.config.levels.items():
            levels_by_model.setdefault(level.model, []).append(level_name)
        for name, model in self.engine.config.models.items():
            level_refs = ", ".join(levels_by_model.get(name, [])) or "-"
            items.append(
                PickerItem(
                    id=name,
                    title=name,
                    description=f"{model.model}  ·  {model.api}",
                    meta=(
                        f"{self._text('models_provider')}: {model.provider}  ·  "
                        f"{self._text('models_context_window')}: "
                        f"{format_tokens(model.context_window_tokens)}  ·  "
                        f"{self._text('level')}: {level_refs}"
                    ),
                )
            )
        if not items:
            items.append(
                PickerItem(
                    id="",
                    title=self._text("none"),
                    description=self._text("models_edit_hint"),
                    meta=str(editable_config_path(self.project_root)),
                )
            )
        subtitle = (
            self._text("models_hint")
            + "  ·  "
            + self._text("models_edit_hint")
            + " "
            + str(editable_config_path(self.project_root))
        )
        self._open_picker(
            self._text("models"),
            items,
            self._open_model_detail_panel,
            subtitle=subtitle,
        )

    def _open_model_detail_panel(self, name: str) -> None:
        if not name:
            return
        model = self.engine.config.models.get(name)
        if model is None:
            self._flash(f"{self._text('models')}: {name}", severity="error")
            return
        try:
            provider = self.engine.config.provider_for_model(model)
        except ConfigError as exc:
            self._flash(str(exc), severity="error")
            return
        lines = [
            f"[bold cyan]{escape(name)}[/bold cyan]",
            f"- {self._text('models_provider')}: {escape(provider.name)}",
            f"- model: {escape(model.model)}",
            f"- {self._text('models_api')}: {escape(model.api)}",
            f"- {self._text('models_context_window')}: "
            f"{format_tokens(model.context_window_tokens)}",
        ]
        lines.append("")
        lines.append(
            f"[dim]{escape(self._text('models_edit_hint'))} "
            f"{escape(str(editable_config_path(self.project_root)))}[/dim]"
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

    def _open_mcp_panel(self) -> None:
        self.engine.refresh_config()
        panel = self._active_fullscreen_panel()
        if panel is not None and panel.can_navigate:
            panel.close_navigation()
            self.call_after_refresh(self._open_mcp_panel)
            return
        self._open_picker(
            self._text("mcp"),
            self._mcp_mention_items(),
            self._choose_mcp_mention,
            subtitle=self._text("mention_mcp_hint"),
        )

    def _open_skills_panel(self) -> None:
        self.engine.refresh_config()
        panel = self._active_fullscreen_panel()
        if panel is not None and panel.can_navigate:
            panel.close_navigation()
            self.call_after_refresh(self._open_skills_panel)
            return
        self._open_picker(
            self._text("skills"),
            self._skill_mention_items(),
            self._choose_skill_mention,
            subtitle=self._text("mention_skills_hint"),
        )

    def _noop_select(self, _value: str) -> None:
        """Callback used by inspect-only pickers."""
        return

    def _open_command_palette(self, *, query: str = "") -> None:
        items = [
            PickerItem(
                id=spec.name,
                title=spec.palette_title,
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
            navigate=True,
        )

    def _choose_command(self, command: str) -> None:
        spec = next((item for item in self._commands() if item.name == command), None)
        if spec is None:
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
        self._tool_delta_cells.clear()
        self._tool_delta_calls.clear()
        self._process_cells = []
        self._process_fold_cell = None
        self._process_collapsed = False
        self._process_anchor_cell = None
        self._reset_transcript(show_empty=False)
        self._render_thread_history(thread_id)
        run_state = self._thread_runs.get(thread_id)
        if run_state is not None:
            if run_state.worker is not None:
                if run_state.process_cells:
                    self._process_cells = list(run_state.process_cells)
                    self._process_collapsed = run_state.process_collapsed
                    if run_state.process_collapsed and run_state.process_fold_cell is None:
                        self._process_fold_cell = self._append_process_fold_cell(
                            self._process_cells,
                            collapsed=True,
                            after=run_state.process_anchor_cell,
                        )
                        run_state.process_fold_cell = self._process_fold_cell
                    elif isinstance(run_state.process_fold_cell, FoldedProcessCell):
                        self._process_fold_cell = run_state.process_fold_cell
                        self._process_fold_cell.set_cells(self._process_cells)
                if run_state.reasoning_buffer:
                    first = run_state.reasoning_buffer.splitlines()[0]
                    if len(first) > 120:
                        first = first[:117].rstrip() + "..."
                    if run_state.reasoning_cell is None:
                        self._reasoning_cell = self._append_cell(
                            f"[dim]{escape(self._text('thinking'))}[/dim] [italic]{escape(first)}[/italic]",
                            "event",
                        )
                        run_state.reasoning_cell = self._reasoning_cell
                if run_state.assistant_buffer:
                    self._assistant_buffer = run_state.assistant_buffer
                    self._assistant_cell = self._append_cell(
                        Markdown(run_state.assistant_buffer),
                        "assistant",
                    )
                    self._assistant_cell.copy_text = run_state.assistant_buffer
                    run_state.assistant_cell = self._assistant_cell
                self._append_cell(f"[dim]{escape(run_state.status)}...[/dim]", "event")
                self._sync_run_state_from_active(run_state)
        if not self._transcript_has_content:
            self._reset_transcript()
        self._refresh_active_run_state()
        self._refresh_status(self._text("resumed"))
        self._close_active_panel()

    def _render_thread_history(self, thread_id: str) -> None:
        segment = self.engine.thread_store.read_history_segment(
            thread_id,
            event_types=VISIBLE_HISTORY_EVENT_TYPES,
        )
        self._history_has_more = segment.has_more
        self._history_before_offset = segment.start_offset
        if segment.has_more:
            transcript = self.query_one("#transcript", VerticalScroll)
            marker = LoadOlderHistoryCell(has_more=True, classes="event", markup=True)
            transcript.mount(marker)
            self._history_more_cell = marker
        self._mount_history_events(segment.events)

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
    ) -> None:
        panel = self._active_fullscreen_panel()
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
        queue_length = self._active_queue_length()
        queued = f" · q{queue_length}" if queue_length else ""
        spinner = ""
        elapsed_suffix = ""
        if self.busy:
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            spinner = frames[self._spinner_index % len(frames)] + " "
            self._spinner_index += 1
            if self._busy_started_at is not None:
                elapsed = format_elapsed(monotonic() - self._busy_started_at)
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
        background_count = len(self._background_run_states())
        if background_count:
            footer += (
                f" [dim]·[/dim] [cyan]{background_count} "
                f"{escape(self._text('background_active'))}[/cyan]"
            )
        self.query_one("#composer-footer", Static).update(footer)
        self._refresh_pending_images()

    def _scroll_end(self) -> None:
        transcript = self.query_one("#transcript", TranscriptScroll)
        if not transcript.follow_tail:
            # User dragged the scrollbar / pressed PgUp; don't yank them back to
            # the bottom on every streaming SSE delta. They re-engage follow by
            # pressing the "↓ bottom" button above the composer.
            return
        transcript.programmatic_scroll_end()
