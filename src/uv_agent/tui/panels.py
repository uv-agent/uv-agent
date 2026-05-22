from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from rich.rule import Rule
from rich.text import Text
from textual import events
from textual.actions import SkipAction
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static, TextArea
from textual.widgets._option_list import Option

from uv_agent.tui.formatting import format_tokens, join_lines, plain, renderable_plain, tool_detail_markup
from uv_agent.tui.state import PanelPage, PendingImage, PickerItem, QueuedTurn
from uv_agent.tui.styles import FULLSCREEN_PANEL_CSS
from uv_agent.tui.widgets import ExpandableTranscriptCell


class PickerOptionList(OptionList):
    ALLOW_SELECT = True


class FullscreenPanel(ModalScreen[str | None]):
    """Scrollable full-screen panel/picker."""

    CSS = FULLSCREEN_PANEL_CSS

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
        body: object = "",
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
            # Title bar: header on the left, subtitle (status / hints) right-
            # aligned on the same row to free a vertical line of chrome.
            with Horizontal(id="panel-titlebar"):
                yield Static(self.panel_title, id="panel-header")
                yield Static(self.subtitle, id="panel-subtitle")
            yield Input(placeholder=getattr(self.app, "_text", lambda key: key)("filter"), id="panel-filter")
            yield PickerOptionList(id="panel-content", compact=False)
            yield VerticalScroll(Static(self.body, markup=False, id="panel-body-content"), id="panel-body")
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

    def replace_picker(
        self,
        *,
        title: str,
        items: list[PickerItem],
        callback: Callable[[str], None],
        subtitle: str = "",
        initial_filter: str = "",
        close_on_select: bool = False,
    ) -> None:
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

    def navigate_panel(self, *, title: str, body: object, subtitle: str = "") -> None:
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
            if disabled and not item.title and not item.description and not item.meta:
                options.append(Option(Rule(characters="─", style="dim"), id=option_id, disabled=True))
                continue
            options.append(
                Option(
                    join_lines(
                        part
                        for part in [
                            plain(item.title, style="bold cyan"),
                            plain(item.description, style="dim") if item.description else None,
                            plain(item.meta, style="dim") if item.meta else None,
                        ]
                        if part is not None
                    ),
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
            options = [Option(plain(label, style="dim"), id="")]
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

class ToolDetailsPanel(FullscreenPanel):
    """Full-screen tool detail panel with j/k navigation between tool results."""

    BINDINGS = [
        Binding("j", "next_detail", "Next", priority=True, show=False),
        Binding("k", "previous_detail", "Previous", priority=True, show=False),
        Binding("e", "toggle_events", "Toggle events", priority=True, show=False),
        Binding("ctrl+d", "dismiss_panel", "Close", priority=True, show=False),
        *FullscreenPanel.BINDINGS,
    ]

    def __init__(self, cell: ExpandableTranscriptCell) -> None:
        self.current_cell = cell
        # events fold state, persisted across j/k navigation within the panel
        self.events_collapsed = False
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
        if event.key == "e":
            event.stop()
            self.action_toggle_events()
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

    def action_toggle_events(self) -> None:
        if self.current_cell.tool_payload is None:
            return
        self.events_collapsed = not self.events_collapsed
        self._refresh_current()

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
        payload = self.current_cell.tool_payload
        if payload is not None:
            self.body = tool_detail_markup(payload, events_collapsed=self.events_collapsed)
        else:
            self.body = self.current_cell.details
        try:
            self.query_one("#panel-header", Static).update(self.panel_title)
            self.query_one("#panel-subtitle", Static).update(self.subtitle)
            self.query_one("#panel-body-content", Static).update(self.body)
            self.query_one("#panel-body", VerticalScroll).scroll_to(y=0, animate=False)
        except NoMatches:
            pass


def _update_static_if_changed(widget: Static, content: object) -> None:
    """Call ``Static.update`` only when the rendered content actually changes.

    ``Static.update`` always triggers a repaint, even if the new content is
    identical to the previous one. For widgets sharing the screen with a
    :class:`TerminalImage`, every redundant repaint becomes a visible image
    flicker on terminals using sixel / TGP encoding. The previous value is
    The previous plain value is cached as an attribute on the widget instance
    (``Static`` keeps its internal content under a name-mangled attribute, so
    storing our own copy is the simplest stable comparison).
    """

    cache_key = renderable_plain(content) or repr(content)
    if getattr(widget, "_uv_last_content", None) == cache_key:
        return
    widget._uv_last_content = cache_key  # type: ignore[attr-defined]
    widget.update(content)


_TERMINAL_IMAGE_CLASS: type[Any] | None = None
_STABLE_TERMINAL_IMAGE_CLASS: type[Any] | None = None


def _terminal_image_class() -> type[Any]:
    """Return textual-image's widget class, importing it only for previews."""

    global _TERMINAL_IMAGE_CLASS
    if _TERMINAL_IMAGE_CLASS is None:
        from textual_image.widget import Image as TerminalImage

        _TERMINAL_IMAGE_CLASS = TerminalImage
    return _TERMINAL_IMAGE_CLASS


def _stable_terminal_image_class() -> type[Any]:
    """Build the cached image widget class lazily.

    ``textual_image`` is optional until a user opens an image preview. Creating
    this subclass on demand preserves the existing flicker fix without importing
    the image backend during ordinary text-only startup.
    """

    global _STABLE_TERMINAL_IMAGE_CLASS
    if _STABLE_TERMINAL_IMAGE_CLASS is not None:
        return _STABLE_TERMINAL_IMAGE_CLASS
    TerminalImage = _terminal_image_class()

    class StableTerminalImage(TerminalImage, Renderable=TerminalImage._Renderable):  # type: ignore[misc, valid-type]
        """A :class:`TerminalImage` that caches its renderable across renders.

        The upstream widget rebuilds the renderable (and so re-encodes /
        retransmits the image bytes) on every :py:meth:`render` call. Textual
        invokes :py:meth:`render` for many unrelated reasons (focus changes,
        layout invalidation, scroll updates), and the repeated re-encode is what
        users perceive as the image preview "flickering" when navigating with
        j/k or scrolling. We memoize the renderable keyed by ``(image identity,
        styled size)`` and only recreate it when one of those actually changes.
        The ``image`` setter already clears ``self._renderable``, so swapping
        images naturally invalidates the cache without extra wiring.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._stable_size: tuple[Any, Any] | None = None

        def render(self) -> Any:
            if not self._image:
                self._stable_size = None
                return ""
            size = self._get_styled_size()
            if self._renderable is not None and self._stable_size == size:
                return self._renderable
            if self._renderable is not None:
                self._renderable.cleanup()
            self._renderable = self._Renderable(self._image, *size)
            self._stable_size = size
            return self._renderable

    _STABLE_TERMINAL_IMAGE_CLASS = StableTerminalImage
    return StableTerminalImage




class PendingSendQueuePanel(FullscreenPanel):
    """Edit the active thread's queued sends before they start running."""

    BINDINGS = [
        Binding("escape", "dismiss_panel", "Close", priority=True, show=False),
        Binding("ctrl+enter", "save_current", "Save", priority=True, show=False),
        Binding("ctrl+j", "save_current", "Save", priority=True, show=False),
    ]

    def __init__(self, thread_id: str | None) -> None:
        self.thread_id = thread_id
        self.queue_items: list[QueuedTurn] = []
        self._option_ids: dict[str, str] = {}
        self._selected_queue_id: str | None = None
        super().__init__(title="", body="", subtitle="")

    def compose(self) -> ComposeResult:
        with Vertical(id="panel-shell"):
            with Horizontal(id="panel-titlebar"):
                yield Static(self.panel_title, id="panel-header")
                yield Static(self.subtitle, id="panel-subtitle")
            with Horizontal(id="pending-turn-actions"):
                yield Button("", id="pending-turn-save", variant="primary", compact=True)
                yield Button("", id="pending-turn-delete", variant="error", compact=True)
                yield Button("", id="pending-turn-up", compact=True)
                yield Button("", id="pending-turn-down", compact=True)
            with Horizontal(id="pending-turn-layout"):
                yield OptionList(id="pending-turn-list", compact=False)
                yield TextArea(
                    "",
                    id="pending-turn-editor",
                    compact=True,
                    soft_wrap=True,
                    show_line_numbers=False,
                )
            yield Static("", id="pending-turn-meta")
            yield Static("", id="panel-footer")

    def _render_page(self, *, filter_value: str = "", highlighted: int | None = None) -> None:
        # ``FullscreenPanel.on_mount`` calls ``_render_page`` for every
        # subclass. This panel has its own layout instead of the generic picker
        # widgets, so rendering must refresh our custom list/editor controls.
        self._refresh_all()

    def on_mount(self) -> None:
        self._refresh_all()
        try:
            self.query_one("#pending-turn-list", OptionList).focus()
        except NoMatches:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "pending-turn-save": self.action_save_current,
            "pending-turn-delete": self.action_delete_current,
            "pending-turn-up": self.action_move_current_up,
            "pending-turn-down": self.action_move_current_down,
        }
        action = actions.get(event.button.id or "")
        if action is None:
            return
        event.stop()
        action()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "pending-turn-list":
            return
        self._select_option(event.option_id)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "pending-turn-list":
            return
        event.stop()
        self._select_option(event.option_id)
        try:
            self.query_one("#pending-turn-editor", TextArea).focus()
        except NoMatches:
            pass

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.action_dismiss_panel()
            return
        if event.key in {"ctrl+enter", "ctrl+j"}:
            event.stop()
            self.action_save_current()
            return
        # Let the editor keep normal composer-like editing behavior: Enter
        # inserts a newline and Delete removes a character. Queue-level delete
        # and reordering remain available from the buttons beside the list.
        if isinstance(self.focused, TextArea):
            if event.key in {"enter", "delete", "ctrl+up", "ctrl+down", "ctrl+left", "ctrl+right"}:
                return
        if event.key == "delete":
            event.stop()
            self.action_delete_current()
            return
        if event.key in {"ctrl+up", "ctrl+left"}:
            event.stop()
            self.action_move_current_up()
            return
        if event.key in {"ctrl+down", "ctrl+right"}:
            event.stop()
            self.action_move_current_down()
            return
        if event.key == "enter":
            event.stop()
            try:
                self.query_one("#pending-turn-editor", TextArea).focus()
            except NoMatches:
                pass
            return
        if event.key in {"up", "down", "pageup", "page_up", "pagedown", "page_down"}:
            event.stop()
            option_list = self.query_one("#pending-turn-list", OptionList)
            if event.key == "up":
                option_list.action_cursor_up()
            elif event.key == "down":
                option_list.action_cursor_down()
            elif event.key in {"pageup", "page_up"}:
                option_list.action_page_up()
            else:
                option_list.action_page_down()
            return

    def action_save_current(self) -> None:
        queue_id = self._selected_queue_id
        if queue_id is None:
            self._flash("pending_turn_missing")
            return
        prompt = self.query_one("#pending-turn-editor", TextArea).text
        result = self._app_call("_update_queued_turn_prompt", self.thread_id, queue_id, prompt)
        if result == "updated":
            self._flash("pending_turn_saved")
        self._refresh_all(preferred_id=queue_id)

    def action_delete_current(self) -> None:
        queue_id = self._selected_queue_id
        if queue_id is None:
            self._flash("pending_turn_missing")
            return
        result = self._app_call("_delete_queued_turn", self.thread_id, queue_id)
        if result == "deleted":
            self._flash("pending_turn_deleted")
        self._refresh_all()

    def action_move_current_up(self) -> None:
        self._move_current(-1)

    def action_move_current_down(self) -> None:
        self._move_current(1)

    def _move_current(self, step: int) -> None:
        queue_id = self._selected_queue_id
        if queue_id is None:
            self._flash("pending_turn_missing")
            return
        result = self._app_call("_move_queued_turn", self.thread_id, queue_id, step)
        if result == "moved":
            self._flash("pending_turn_moved")
        self._refresh_all(preferred_id=queue_id)

    def _refresh_all(self, preferred_id: str | None = None) -> None:
        text = getattr(self.app, "_text", lambda key: key)
        self.panel_title = text("pending_turns_panel")
        self.subtitle = text("pending_turns_hint")
        self.queue_items = list(self._app_call("_queued_turns_for_thread", self.thread_id) or [])
        if preferred_id and any(item.queue_id == preferred_id for item in self.queue_items):
            self._selected_queue_id = preferred_id
        elif self._selected_queue_id not in {item.queue_id for item in self.queue_items}:
            self._selected_queue_id = self.queue_items[0].queue_id if self.queue_items else None
        try:
            with self.app.batch_update():
                _update_static_if_changed(self.query_one("#panel-header", Static), self.panel_title)
                _update_static_if_changed(self.query_one("#panel-subtitle", Static), self.subtitle)
                _update_static_if_changed(self.query_one("#panel-footer", Static), text("pending_turns_footer"))
                self._refresh_buttons()
                self._refresh_options()
                self._refresh_editor()
        except NoMatches:
            pass

    def _refresh_buttons(self) -> None:
        text = getattr(self.app, "_text", lambda key: key)
        labels = {
            "#pending-turn-save": text("save"),
            "#pending-turn-delete": text("delete"),
            "#pending-turn-up": text("move_up"),
            "#pending-turn-down": text("move_down"),
        }
        has_selection = self._selected_queue_id is not None
        for selector, label in labels.items():
            button = self.query_one(selector, Button)
            if str(button.label) != label:
                button.label = label
            button.disabled = not has_selection

    def _refresh_options(self) -> None:
        text = getattr(self.app, "_text", lambda key: key)
        option_list = self.query_one("#pending-turn-list", OptionList)
        self._option_ids = {}
        options: list[Option] = []
        for index, item in enumerate(self.queue_items):
            option_id = f"queued_{index}"
            self._option_ids[option_id] = item.queue_id
            options.append(Option(self._option_markup(index, item), id=option_id))
        if not options:
            options.append(Option(plain(text("no_pending_turns"), style="dim"), id="", disabled=True))
        option_list.set_options(options)
        selected_index = self._selected_index()
        if selected_index is not None and option_list.option_count:
            option_list.highlighted = selected_index

    def _refresh_editor(self) -> None:
        text = getattr(self.app, "_text", lambda key: key)
        editor = self.query_one("#pending-turn-editor", TextArea)
        meta = self.query_one("#pending-turn-meta", Static)
        item = self._selected_item()
        if item is None:
            editor.load_text("")
            editor.disabled = True
            _update_static_if_changed(meta, plain(text("no_pending_turns"), style="dim"))
            return
        if editor.text != item.prompt:
            editor.load_text(item.prompt)
            editor.cursor_location = editor.document.end
        editor.disabled = False
        _update_static_if_changed(meta, self._item_meta(item))

    def _select_option(self, option_id: str | None) -> None:
        queue_id = self._option_ids.get(option_id or "")
        if queue_id is None:
            return
        self._selected_queue_id = queue_id
        self._refresh_editor()
        self._refresh_buttons()

    def _selected_index(self) -> int | None:
        for index, item in enumerate(self.queue_items):
            if item.queue_id == self._selected_queue_id:
                return index
        return None

    def _selected_item(self) -> QueuedTurn | None:
        index = self._selected_index()
        return self.queue_items[index] if index is not None else None

    def _option_markup(self, index: int, item: QueuedTurn) -> Text:
        text = getattr(self.app, "_text", lambda key: key)
        preview = " ".join(item.prompt.split()) or text("empty")
        if len(preview) > 80:
            preview = preview[:77].rstrip() + "..."
        title = Text.assemble((f"{index + 1}. ", "bold cyan"), preview)
        meta_parts = [f"{len(item.image_paths)} {text('images')}" if item.image_paths else f"0 {text('images')}"]
        if item.level:
            meta_parts.append(f"{text('level')}: {item.level}")
        meta = plain(" · ".join(meta_parts), style="dim")
        return join_lines([title, meta])  # type: ignore[return-value]

    def _item_meta(self, item: QueuedTurn) -> Text:
        text = getattr(self.app, "_text", lambda key: key)
        index = self._selected_index()
        prefix = f"{index + 1}/{len(self.queue_items)}" if index is not None else "-"
        parts = [prefix, f"{len(item.image_paths)} {text('images')}"]
        if item.level:
            parts.append(f"{text('level')}: {item.level}")
        return plain(" · ".join(parts), style="dim")

    def _app_call(self, name: str, *args: object) -> object:
        handler = getattr(self.app, name, None)
        if callable(handler):
            return handler(*args)
        return None

    def _flash(self, key: str) -> None:
        flash = getattr(self.app, "_flash", None)
        text = getattr(self.app, "_text", lambda value: value)
        if callable(flash):
            flash(text(key))

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

    def compose(self) -> ComposeResult:
        # The meta line lives INSIDE the scroll container so the whole panel
        # interior (meta + image) scrolls as a single unit when the image is
        # taller than the viewport, instead of giving the image its own
        # standalone scrollbar.
        with Vertical(id="panel-shell"):
            with Horizontal(id="panel-titlebar"):
                yield Static(self.panel_title, id="panel-header")
                yield Static(self.subtitle, id="panel-subtitle")
            with VerticalScroll(id="image-preview-scroll"):
                yield Static("", markup=False, id="image-preview-meta")
                yield _stable_terminal_image_class()(id="image-preview")
            yield Static(getattr(self.app, "_text", lambda key: key)("panel_footer"), id="panel-footer")

    def _render_page(self, *, filter_value: str = "", highlighted: int | None = None) -> None:
        self._refresh_current()

    def on_mount(self) -> None:
        self._refresh_current()
        try:
            self.query_one("#image-preview-scroll", VerticalScroll).focus()
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

    def action_cursor_up(self) -> None:
        self.query_one("#image-preview-scroll", VerticalScroll).action_scroll_up()

    def action_cursor_down(self) -> None:
        self.query_one("#image-preview-scroll", VerticalScroll).action_scroll_down()

    def action_page_up(self) -> None:
        self.query_one("#image-preview-scroll", VerticalScroll).action_page_up()

    def action_page_down(self) -> None:
        self.query_one("#image-preview-scroll", VerticalScroll).action_page_down()

    def _move(self, step: int) -> None:
        if not self.attachments:
            return
        self.index = (self.index + step) % len(self.attachments)
        self._refresh_current()

    def _refresh_current(self) -> None:
        text = getattr(self.app, "_text", lambda key: key)
        self.panel_title = text("image_preview")
        self.subtitle = text("image_preview_hint")
        attachment = self._current_attachment()
        path = Path(str(attachment.get("stored_path") or "")) if attachment else None
        try:
            # Batch all updates so Textual emits one composite paint instead of
            # several, and guard each ``Static.update`` so unchanged labels do
            # not force a redundant repaint that the terminal would render as a
            # flicker over the image.
            with self.app.batch_update():
                _update_static_if_changed(self.query_one("#panel-header", Static), self.panel_title)
                _update_static_if_changed(self.query_one("#panel-subtitle", Static), self.subtitle)
                _update_static_if_changed(
                    self.query_one("#image-preview-meta", Static), self._attachment_markup()
                )
                image_widget = self.query_one("#image-preview", _terminal_image_class())
                new_image = path if path and path.exists() else None
                # The upstream setter unconditionally calls ``refresh(layout=True)``
                # (and on Sixel, ``refresh(recompose=True)``), so we skip the
                # assignment whenever the path has not changed.
                if image_widget.image != new_image:
                    image_widget.image = new_image
                self.query_one("#image-preview-scroll", VerticalScroll).scroll_to(y=0, animate=False)
        except NoMatches:
            pass

    def _current_attachment(self) -> dict[str, Any] | None:
        if not self.attachments:
            return None
        return self.attachments[self.index]

    def _attachment_markup(self) -> Text:
        text = getattr(self.app, "_text", lambda key: key)
        attachment = self._current_attachment()
        if attachment is None:
            return plain(text("no_images"), style="dim")
        path = Path(str(attachment.get("stored_path") or ""))
        source = str(attachment.get("source_path") or "")
        note = str(attachment.get("note") or "").strip()
        size = int(attachment.get("size_bytes") or 0)
        # Display name prefers the user-supplied source filename so tests and
        # users see the file as they know it; the stored path appears dimmed at
        # the end for traceability.
        display_name = Path(source).name if source else (path.name or str(path))
        mime = str(attachment.get("mime_type") or "").strip()
        parts = [
            Text(f"{self.index + 1}/{len(self.attachments)}", style="bold"),
            plain(display_name, style="cyan"),
        ]
        if mime:
            parts.append(plain(mime))
        if size:
            parts.append(plain(f"{format_tokens(size)}B"))
        parts.append(plain(str(path), style="dim"))
        line = Text(" · ").join(parts)
        if note:
            line.append("  ")
            line.append(f"{text('image_note')}: {note}", style="dim")
        return line


class PendingImagePreviewPanel(ImagePreviewPanel):
    """Full-screen preview panel for images queued in the composer."""

    BINDINGS = [
        Binding("delete", "delete_current_image", "Delete", priority=True, show=False),
        Binding("backspace", "delete_current_image", "Delete", priority=True, show=False),
        *ImagePreviewPanel.BINDINGS,
    ]

    def __init__(self, pending_images: list[PendingImage], index: int = 0) -> None:
        self.pending_images = pending_images
        super().__init__([image.to_attachment() for image in pending_images], index)

    def compose(self) -> ComposeResult:
        # Delete button stays outside the scroll so it remains reachable even
        # when a tall image scrolls; meta moves into the scroll alongside the
        # image so the panel interior scrolls as one unit.
        with Vertical(id="panel-shell"):
            with Horizontal(id="panel-titlebar"):
                yield Static(self.panel_title, id="panel-header")
                yield Static(self.subtitle, id="panel-subtitle")
            yield Button("", variant="error", id="pending-image-delete", compact=True)
            with VerticalScroll(id="image-preview-scroll"):
                yield Static("", markup=False, id="image-preview-meta")
                yield _stable_terminal_image_class()(id="image-preview")
            yield Static(getattr(self.app, "_text", lambda key: key)("panel_footer"), id="panel-footer")

    def _refresh_current(self) -> None:
        super()._refresh_current()
        text = getattr(self.app, "_text", lambda key: key)
        self.panel_title = text("pending_image_preview")
        self.subtitle = text("pending_image_preview_hint")
        try:
            with self.app.batch_update():
                _update_static_if_changed(self.query_one("#panel-header", Static), self.panel_title)
                _update_static_if_changed(self.query_one("#panel-subtitle", Static), self.subtitle)
                delete_button = self.query_one("#pending-image-delete", Button)
                new_label = text("delete_pending_image")
                if str(delete_button.label) != new_label:
                    delete_button.label = new_label
        except NoMatches:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "pending-image-delete":
            return
        event.stop()
        self.action_delete_current_image()

    def action_delete_current_image(self) -> None:
        if not self.pending_images:
            self.dismiss(None)
            return
        deleted = self.pending_images.pop(self.index)
        app = self.app
        delete = getattr(app, "_delete_pending_image", None)
        if callable(delete):
            delete(deleted)
        if not self.pending_images:
            self.dismiss(None)
            return
        self.index = min(self.index, len(self.pending_images) - 1)
        self.attachments = [image.to_attachment() for image in self.pending_images]
        self._refresh_current()

    def _attachment_markup(self) -> Text:
        attachment = self._current_attachment()
        if attachment is None:
            text = getattr(self.app, "_text", lambda key: key)
            return plain(text("no_pending_images"), style="dim")
        path = Path(str(attachment.get("stored_path") or ""))
        size = int(attachment.get("size_bytes") or 0)
        width = int(attachment.get("width") or 0)
        height = int(attachment.get("height") or 0)
        parts = [
            Text(f"{self.index + 1}/{len(self.attachments)}", style="bold"),
            plain(path.name or str(path), style="cyan"),
        ]
        if width and height:
            parts.append(plain(f"{width}×{height}"))
        if size:
            parts.append(plain(f"{format_tokens(size)}B"))
        parts.append(plain(str(path), style="dim"))
        return Text(" · ").join(parts)
