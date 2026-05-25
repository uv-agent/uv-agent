from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.selection import Selection
from textual.strip import Strip
from textual.widgets import Button, Static, TextArea

from uv_agent.tui import theme
from uv_agent.tui.formatting import format_tokens, join_lines, plain, renderable_plain
from uv_agent.tui.styles import EMPTY_STATE_CSS, TRANSCRIPT_CELL_CSS


def image_attachment_markup(attachment: dict[str, Any], *, label: str = "image attached") -> Text:
    path = Path(str(attachment.get("stored_path") or ""))
    name = path.name or str(path)
    size = int(attachment.get("size_bytes") or 0)
    size_label = f" · {format_tokens(size)}B" if size else ""
    first = Text()
    first.append(str(label), style="dim")
    first.append(" ")
    first.append(name, style="cyan")
    first.append(size_label, style="dim")
    return join_lines([first, plain("[preview]", style="dim")])  # type: ignore[return-value]

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
    _scroll_pending = False

    def programmatic_scroll_end(self, *, force: bool = False) -> None:
        # Defer the actual scroll to after the next refresh so any pending
        # mount/update has had a chance to recompute virtual_size; otherwise
        # `scroll_end` reads a stale `max_scroll_y` and only crawls along
        # one row at a time during streaming. Coalesce repeated calls within
        # the same refresh cycle so streaming deltas don't pile up callbacks.
        # A user clicking the bottom affordance is different from an automatic
        # stream follow: it must enqueue a fresh callback after any in-flight
        # user scroll callbacks, otherwise the click may be lost behind stale
        # work from a previous refresh.
        if self._scroll_pending and not force:
            return
        self._scroll_pending = True

        def _do() -> None:
            self._scroll_pending = False
            self.scroll_end(animate=False, immediate=True)
            self._recompute_near_bottom()

        self.call_after_refresh(_do)

    def engage_follow_tail(self, *, force: bool = False) -> None:
        self.follow_tail = True
        self.programmatic_scroll_end(force=force)

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
            self._refresh_overlay()
            return
        near_bottom = (self.max_scroll_y - self.scroll_y) <= self._BOTTOM_THRESHOLD
        self.near_bottom = near_bottom
        if near_bottom and restore_follow:
            self.follow_tail = True
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        refresh = getattr(self.app, "_refresh_composer_overlay", None)
        if callable(refresh):
            refresh()

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
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

    DEFAULT_CSS = EMPTY_STATE_CSS

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self.frame = 0

    def tick(self) -> None:
        frame = self.FRAMES[self.frame % len(self.FRAMES)]
        self.frame += 1
        text = getattr(self.app, "_text", lambda key: key)
        self.update(
            join_lines(
                [
                    Text.assemble(
                        (text("ready_title"), "bold #dce7f3"),
                        " ",
                        (frame, "dim"),
                    ),
                    plain(text("ready_hint"), style="dim"),
                ]
            )
        )


class ComposerTextArea(TextArea):
    """Composer text area with app-specific history and copy feedback."""

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

class TranscriptCell(Static):
    """Small transcript block used by the Textual chat timeline."""

    SELECTION_STYLE = Style(color=theme.SELECTION_FG, bgcolor=theme.SELECTION_BG)

    DEFAULT_CSS = TRANSCRIPT_CELL_CSS

    def __init__(self, content: object = "", *, copy_text: str | None = None, **kwargs: Any) -> None:
        # Transcript content is passed as renderables. Parsing string markup here
        # would reintroduce the bracket-escaping bug this refactor removes.
        kwargs.setdefault("markup", False)
        super().__init__(content, **kwargs)
        self.copy_text: str | None = copy_text if copy_text is not None else self._plain_copy_text(content)
        self._rendered_copy_lines: dict[int, str] = {}

    def update(self, content: object = "", *, layout: bool = True, copy_text: str | None = None) -> None:
        self.copy_text = copy_text if copy_text is not None else self._plain_copy_text(content)
        self._rendered_copy_lines.clear()
        super().update(content, layout=layout)

    def update_copy_text(self, copy_text: str | None) -> None:
        """Refresh copy/selection text without invalidating the rendered cell."""

        self.copy_text = copy_text
        self._rendered_copy_lines.clear()

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
        return renderable_plain(content)


class RetryTurnButton(Button):
    """Retry affordance for transient provider/network turn failures."""

    def __init__(self, label: str, *, thread_id: str | None = None) -> None:
        super().__init__(label, variant="primary", compact=True, classes="retry_turn")
        self.retry_thread_id = thread_id


class ExpandableTranscriptCell(TranscriptCell, can_focus=True):
    """Transcript cell that opens hidden details in a panel."""

    def __init__(
        self,
        summary: object,
        details: object,
        detail_title: str = "tool_details",
        detail_hint: str = "tool_details_hint",
        **kwargs: Any,
    ) -> None:
        self.summary = summary
        self.details = details
        self.detail_title = detail_title
        self.detail_hint = detail_hint
        # Optional runner payload, attached for tool-result cells so the
        # details panel can re-render the body with stdout folded/unfolded.
        self.tool_payload: dict[str, Any] | None = None
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

    def set_details(self, summary: object, details: object) -> None:
        self.summary = summary
        self.details = details
        self.update(self._content())

    def open_details(self) -> None:
        app = self.app
        if hasattr(app, "_open_tool_details_panel"):
            app._open_tool_details_panel(self)

    def _content(self) -> object:
        return self.summary


class FoldedProcessCell(TranscriptCell, can_focus=True):
    """A transcript-level fold that reveals the original in-between cells."""

    def __init__(
        self,
        cells: list[TranscriptCell],
        *,
        collapsed: bool = True,
        elapsed_label: str = "",
        **kwargs: Any,
    ) -> None:
        self.cells = list(cells)
        self.collapsed = collapsed
        self.elapsed_label = elapsed_label
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

    def set_elapsed_label(self, elapsed_label: str) -> None:
        self.elapsed_label = elapsed_label
        self._refresh()

    def set_collapsed(self, collapsed: bool, *, notify: bool = True) -> None:
        self.collapsed = collapsed
        self._apply_visibility()
        self._refresh()
        if not notify:
            return
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
        def fallback_text(key: str) -> str:
            return key

        try:
            text = getattr(self.app, "_text", fallback_text)
        except Exception:
            text = fallback_text
        count = len(self.cells)
        key = "process_fold_collapsed" if self.collapsed else "process_fold_expanded"
        state = text(key)
        step_label = text("process_fold_step" if count == 1 else "process_fold_steps")
        hint = text("process_fold_expand_hint" if self.collapsed else "process_fold_collapse_hint")
        elapsed = f" · {self.elapsed_label}" if self.elapsed_label else ""
        self.update(
            Text.assemble(
                (f"{state} · {count} {step_label}{elapsed}", "dim"),
                " ",
                (hint, "dim"),
            )
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
        self.update(plain(label, style="dim"))

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
