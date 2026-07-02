from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TextIO

from uv_agent.tui.ansi import terminal_size, truncate_visible
from uv_agent.tui.components import _needs_gap_between_cells, render_cell, render_live_with_cursor
from uv_agent.tui.events import TranscriptCell, TuiState


class Renderer:
    """Append-only transcript renderer with a repaintable live tail.

    tui deliberately uses the normal terminal buffer so completed transcript
    cells remain available in the user's real scrollback.  The renderer therefore
    treats completed cells as append-only: once a transcript row has been written
    with normal CRLF flow, later repaints must never clear over it.

    Only the *live tail* (status rows, in-flight cells, composer, pickers/pagers)
    is repaintable.  Short transcripts use a floating relative frame so startup
    and resume do not manufacture blank scrollback; after the transcript naturally
    fills the viewport, the frame switches to absolute row erases inside known
    bottom rows.
    """

    _SYNC_ON = "\x1b[?2026h"
    _SYNC_OFF = "\x1b[?2026l"
    # CSI ?7 toggles DECAWM (auto-wrap mode).  With auto-wrap disabled, writing
    # past the last column overwrites that column instead of scrolling the
    # viewport, so a cell-width misestimate cannot desynchronise frame math.
    _AUTOWRAP_OFF = "\x1b[?7l"
    _AUTOWRAP_ON = "\x1b[?7h"
    _EL = "\x1b[2K"  # erase entire line

    def __init__(self, output: TextIO | None = None) -> None:
        self.output = output or sys.stdout
        self.width = self._paint_width(terminal_size()[0])
        self._frame_top_row = 0  # 1-indexed absolute viewport row; 0 = no frame
        self._frame_rows = 0
        self._frame_cursor_row = 0
        self._frame_cursor_col = 0
        self._has_frame = False
        self._frame_anchored = False
        # Approximate visible transcript rows emitted since the last explicit
        # clear.  Short transcripts keep the live frame in a floating relative
        # mode; once content naturally fills the viewport, absolute CUP/EL repaint
        # becomes safe because the live frame is pinned to known bottom rows.
        self._transcript_rows = 0
        # Explicit clears and natural scrolling establish an absolute row anchor.
        # A fresh process starts unanchored so first paint does not manufacture
        # blank scrollback just to discover the bottom of the terminal.
        self._anchor_known = False
        self.spinner_frame = 0
        self._last_flushed_kind: str | None = None
        # Row count for the latest completed transcript cell, excluding the
        # separator gap before it.  Growth of ephemeral chrome such as command
        # palettes may scroll older history, but should not consume the visible
        # tail of the most recent answer.
        self._last_flushed_rows = 0

    @property
    def live_height(self) -> int:
        return self._frame_rows if self._has_frame else 0

    @property
    def cursor_row(self) -> int:
        return self._frame_cursor_row

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_gap_between(last_kind: str | None, current_kind: str) -> bool:
        # Share the rule with the live renderer so flushed scrollback and live
        # show identical spacing.
        return _needs_gap_between_cells(last_kind, current_kind)

    def _write(self, text: str) -> None:
        self.output.write(text)
        self.output.flush()

    @staticmethod
    def _cup(row: int, col: int = 1) -> str:
        return f"\x1b[{row};{col}H"

    @staticmethod
    def _up(rows: int) -> str:
        return f"\x1b[{rows}A" if rows > 0 else ""

    @staticmethod
    def _hpa(col: int) -> str:
        return f"\x1b[{col}G"

    @staticmethod
    def _paint_width(columns: int) -> int:
        """Return a render width that never writes into the last terminal column."""

        # Some Windows ConPTY/terminal combinations advance to the next row as
        # soon as a printable character lands in the rightmost column.  Keeping
        # one column empty avoids that terminal-dependent wrap edge while
        # changing the visual layout by only a single cell.
        return max(1, columns - 1)

    def _live_top(self, rows: int, height: int) -> int:
        """Absolute 1-indexed top row for a live frame of ``height`` rows."""

        return max(1, min(self._transcript_rows + 1, rows - height + 1))

    def _cap_transcript_rows_above_live(self, rows: int, height: int) -> None:
        """Keep ``_transcript_rows`` as the visible rows above the live frame.

        A large picker can make a previously floating live frame scroll the
        viewport for the first time.  Rows scrolled off by that live frame cannot
        reappear when the picker closes, so later smaller frames must anchor
        immediately after the *remaining* visible transcript rows rather than the
        pre-scroll count.  Otherwise the renderer paints the composer too low and
        leaves a large blank band between history and the input box.
        """

        max_visible = max(0, rows - height)
        if self._transcript_rows > max_visible:
            self._transcript_rows = max_visible

    def _transcript_rows_to_preserve_on_growth(self, state: TuiState) -> int:
        """Return visible transcript rows that live-frame growth must keep.

        The last assistant cell is what the user is usually reading when they
        open the slash palette or start another prompt.  Preserving only one row
        is enough for most live growth, but multi-row assistant tails need their
        whole visible cell to avoid the palette looking like it swallowed part of
        the answer.  Other cell kinds can still scroll normally so large history
        blocks do not permanently starve transient UI chrome.
        """

        if self._transcript_rows <= 0:
            return 0
        preserve = max(1, self._last_flushed_rows)
        if state.command_palette_open and self._last_flushed_kind == "assistant":
            return min(self._transcript_rows, preserve)
        # Non-palette growth (for example, the next user prompt) may need more
        # room for live status, but should still keep at least the final row.
        return min(self._transcript_rows, 1)

    def _rerender_live_with_height(
        self,
        state: TuiState,
        cols: int,
        rows: int,
        height: int,
    ) -> tuple[list[str], int, int]:
        """Render live content for an already-decided viewport height.

        ``render_live_with_cursor`` tries to respect ``max_height`` but the
        composer has a minimum shape.  Apply a final defensive clip so repainting
        never writes beyond the rows the renderer decided are safe to touch.
        """

        height = max(1, height)
        lines, cursor_row, cursor_col = self._render_live(state, cols, rows, max_height=height)
        if len(lines) <= height:
            return lines, cursor_row, cursor_col

        dropped = len(lines) - height
        if lines and lines[0].strip():
            keep_tail = max(0, height - 1)
            clipped = [lines[0], *lines[-keep_tail:]] if keep_tail else [lines[0]]
            dropped_after_head = len(lines) - len(clipped)
            if cursor_row > 0:
                cursor_row = max(0, cursor_row - dropped_after_head)
            cursor_row = min(cursor_row, len(clipped) - 1)
            return clipped, cursor_row, cursor_col

        clipped = lines[dropped:]
        cursor_row = max(0, min(cursor_row - dropped, len(clipped) - 1))
        return clipped, cursor_row, cursor_col

    def _clear_rows(self, buf: list[str], top: int, bottom: int, rows: int) -> None:
        for row in range(max(1, top), min(rows, bottom) + 1):
            buf.append(self._cup(row, 1) + self._EL)

    def _clear_floating_frame(self, buf: list[str]) -> None:
        """Erase the current unanchored frame using only rows we just painted.

        A short transcript has no trustworthy absolute viewport row: the app may
        have started anywhere in the user's terminal.  In that phase we use
        relative movement, but only within the tracked live frame and never with
        ``ESC[J`` clear-to-end, so completed transcript rows above the frame stay
        append-only.
        """

        if not self._has_frame:
            return
        buf.append("\r" + self._up(self._frame_cursor_row))
        for index in range(self._frame_rows):
            buf.append(self._EL)
            if index < self._frame_rows - 1:
                buf.append("\r\n")
        if self._frame_rows > 1:
            buf.append("\r" + self._up(self._frame_rows - 1))
        else:
            buf.append("\r")

    def _reset_frame(self) -> None:
        self._has_frame = False
        self._frame_anchored = False
        self._frame_top_row = 0
        self._frame_rows = 0
        self._frame_cursor_row = 0
        self._frame_cursor_col = 0

    def _frame_is_anchored_after_append(self, rows: int, height: int) -> bool:
        if self._anchor_known:
            self._cap_transcript_rows_above_live(rows, height)
            return True
        if self._transcript_rows + height >= rows:
            self._anchor_known = True
            self._cap_transcript_rows_above_live(rows, height)
            return True
        return False

    def _render_live(
        self,
        state: TuiState,
        cols: int,
        rows: int,
        *,
        max_height: int | None = None,
    ) -> tuple[list[str], int, int]:
        live_max_height = max(3, max_height if max_height is not None else rows - 1)
        return render_live_with_cursor(
            state,
            self.width,
            self.spinner_frame,
            max_height=live_max_height,
            preceding_kind=self._last_flushed_kind,
            has_preceding_transcript=self._last_flushed_kind is not None,
        )

    def _record_live_frame(
        self,
        *,
        rows: int,
        cols: int,
        height: int,
        cursor_row: int,
        cursor_col: int,
        anchored: bool,
    ) -> None:
        del cols  # Kept in the signature to mirror cursor placement callers.
        top = self._live_top(rows, height) if anchored else max(1, self._transcript_rows + 1)
        self._has_frame = True
        self._frame_anchored = anchored
        self._frame_top_row = top
        self._frame_rows = height
        self._frame_cursor_row = cursor_row
        self._frame_cursor_col = cursor_col

    def _place_cursor_in_frame(
        self,
        buf: list[str],
        *,
        rows: int,
        cols: int,
        height: int,
        cursor_row: int,
        cursor_col: int,
        anchored: bool,
    ) -> None:
        final_col = max(1, min(cols, cursor_col + 1))
        if anchored:
            final_row = max(1, min(rows, self._frame_top_row + cursor_row))
            buf.append(self._cup(final_row, final_col))
            return

        last_row = max(0, height - 1)
        target_row = max(0, min(last_row, cursor_row))
        if target_row < last_row:
            buf.append(self._up(last_row - target_row))
        buf.append("\r" + self._hpa(final_col))

    def _append_live_frame_after_transcript(
        self,
        buf: list[str],
        *,
        lines: list[str],
        cursor_row: int,
        cursor_col: int,
        cols: int,
        rows: int,
    ) -> None:
        """Append a fresh live frame with normal flow and track it.

        This method intentionally does not reserve bottom rows.  If the transcript
        is still shorter than the viewport, the frame remains floating and future
        repaints use constrained relative erases.  Once transcript + frame reaches
        the viewport bottom naturally, the frame becomes safe for absolute CUP/EL
        repainting.
        """

        if not lines:
            self._reset_frame()
            return
        height = len(lines)
        anchored = self._frame_is_anchored_after_append(rows, height)
        buf.append("\r\n".join(lines))
        self._record_live_frame(
            rows=rows,
            cols=cols,
            height=height,
            cursor_row=cursor_row,
            cursor_col=cursor_col,
            anchored=anchored,
        )
        self._place_cursor_in_frame(
            buf,
            rows=rows,
            cols=cols,
            height=height,
            cursor_row=cursor_row,
            cursor_col=cursor_col,
            anchored=anchored,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def paint_live_after_transcript(self, state: TuiState) -> None:
        """Draw the first repaintable live frame after appended transcript rows."""

        if self._has_frame:
            # The caller is not actually positioned after fresh transcript rows;
            # use the normal repaint path so we only touch the existing live tail.
            self.repaint(state)
            return
        cols, rows = terminal_size()
        self.width = self._paint_width(cols)
        lines, cursor_row, cursor_col = self._render_live(state, cols, rows)
        buf = [self._SYNC_ON + self._AUTOWRAP_OFF]
        self._append_live_frame_after_transcript(
            buf,
            lines=lines,
            cursor_row=cursor_row,
            cursor_col=cursor_col,
            cols=cols,
            rows=rows,
        )
        buf.append(self._AUTOWRAP_ON + self._SYNC_OFF)
        self._write("".join(buf))

    def flush_cell(self, cell: TranscriptCell, live_state: TuiState | None = None) -> None:
        """Append a completed cell into scrollback and optionally redraw live tail."""

        cols, rows = terminal_size()
        self.width = self._paint_width(cols)
        cell_lines = [
            truncate_visible(line, self.width)
            for line in render_cell(cell, self.width, spinner_frame=self.spinner_frame)
        ]
        if not cell_lines:
            return

        # Insert a blank row around turn boundaries (user -> middle,
        # middle -> assistant/user) but pack middle-process cells together.
        gap = self._last_flushed_kind is not None and self._needs_gap_between(
            self._last_flushed_kind, cell.kind
        )
        hard_lines = ([""] + cell_lines) if gap else cell_lines

        live_lines: list[str] = []
        live_cursor_row = 0
        live_cursor_col = 0
        if live_state is not None:
            live_lines, live_cursor_row, live_cursor_col = self._render_live(live_state, cols, rows)

        buf = [self._SYNC_ON + self._AUTOWRAP_OFF]
        # A completed cell replaces the old live tail.  Clearing only tracked
        # live rows preserves completed transcript rows above it.
        if self._has_frame:
            if self._frame_anchored:
                self._clear_rows(buf, self._frame_top_row, self._frame_top_row + self._frame_rows - 1, rows)
                buf.append(self._cup(max(1, min(self._frame_top_row, rows)), 1))
            else:
                self._clear_floating_frame(buf)
        buf.append("\r\n".join(hard_lines) + "\r\n")
        self._transcript_rows += len(hard_lines)
        if self._transcript_rows >= rows:
            self._anchor_known = True
        self._last_flushed_kind = cell.kind
        self._last_flushed_rows = len(cell_lines)
        self._reset_frame()

        if live_lines:
            self._append_live_frame_after_transcript(
                buf,
                lines=live_lines,
                cursor_row=live_cursor_row,
                cursor_col=live_cursor_col,
                cols=cols,
                rows=rows,
            )

        buf.append(self._AUTOWRAP_ON + self._SYNC_OFF)
        self._write("".join(buf))

    def flush_cells(self, cells: Iterable[TranscriptCell], live_state: TuiState | None = None) -> None:
        """Append a batch of transcript cells, then draw one live frame.

        History re-entry uses this path.  Drawing the live frame once after the
        whole batch prevents the composer from overwriting the tail of the final
        historical assistant message.
        """

        for cell in cells:
            self.flush_cell(cell)
        if live_state is not None:
            self.paint_live_after_transcript(live_state)

    def flushed_cell_rows(self, cells: Iterable[TranscriptCell]) -> int:
        """Return how many terminal rows ``flush_cells`` will advance."""

        self.width = self._paint_width(terminal_size()[0])
        total = 0
        last_kind: str | None = None
        for cell in cells:
            lines = render_cell(cell, self.width, spinner_frame=self.spinner_frame)
            if not lines:
                continue
            if last_kind is not None and self._needs_gap_between(last_kind, cell.kind):
                total += 1
            total += len(lines) + 1
            last_kind = cell.kind
        return total

    def clear_screen(self, *, rule: str | None = None) -> None:
        """Clear the visible terminal and optionally leave a top separator."""

        self.width = self._paint_width(terminal_size()[0])
        buf = [self._SYNC_ON + self._AUTOWRAP_OFF]
        # CSI 2J clears the visible viewport; CSI 3J also discards scrollback on
        # terminals that support it.  Only explicit destructive UI resets should
        # call this method.
        buf.append("\x1b[2J\x1b[3J\x1b[H")
        self._transcript_rows = 0
        self._anchor_known = True
        if rule:
            buf.append(truncate_visible(rule, self.width) + "\r\n\r\n")
            self._transcript_rows = 2
        buf.append(self._AUTOWRAP_ON + self._SYNC_OFF)
        self._write("".join(buf))
        self._reset_frame()
        self._last_flushed_kind = None
        self._last_flushed_rows = 0

    def repaint(self, state: TuiState) -> None:
        cols, rows = terminal_size()
        self.width = self._paint_width(cols)
        # The app owns spinner timing. Repaint frequency can spike with streaming
        # deltas, so advancing frames here would make animation speed depend on
        # token throughput instead of wall-clock time.
        lines, cursor_row, cursor_col = self._render_live(state, cols, rows)

        if not self._has_frame:
            # No frame means we have no rows we are allowed to erase.  Append the
            # first live frame in normal flow instead of clearing an absolute
            # bottom slice that might still contain transcript text.
            buf = [self._SYNC_ON + self._AUTOWRAP_OFF]
            self._append_live_frame_after_transcript(
                buf,
                lines=lines,
                cursor_row=cursor_row,
                cursor_col=cursor_col,
                cols=cols,
                rows=rows,
            )
            buf.append(self._AUTOWRAP_ON + self._SYNC_OFF)
            self._write("".join(buf))
            return

        buf = [self._SYNC_ON + self._AUTOWRAP_OFF]
        if not lines:
            if self._frame_anchored:
                self._clear_rows(buf, self._frame_top_row, self._frame_top_row + self._frame_rows - 1, rows)
            else:
                self._clear_floating_frame(buf)
            buf.append(self._AUTOWRAP_ON + self._SYNC_OFF)
            self._write("".join(buf))
            self._reset_frame()
            return

        height = len(lines)
        if not self._frame_anchored:
            if state.command_palette_open:
                preserve_rows = self._transcript_rows_to_preserve_on_growth(state)
                available_height = max(1, rows - preserve_rows)
                if height > available_height:
                    lines, cursor_row, cursor_col = self._rerender_live_with_height(
                        state,
                        cols,
                        rows,
                        available_height,
                    )
                    height = len(lines)
            self._clear_floating_frame(buf)
            buf.append("\r\n".join(lines))
            anchored = self._frame_is_anchored_after_append(rows, height)
            self._record_live_frame(
                rows=rows,
                cols=cols,
                height=height,
                cursor_row=cursor_row,
                cursor_col=cursor_col,
                anchored=anchored,
            )
            self._place_cursor_in_frame(
                buf,
                rows=rows,
                cols=cols,
                height=height,
                cursor_row=cursor_row,
                cursor_col=cursor_col,
                anchored=anchored,
            )
            buf.append(self._AUTOWRAP_ON + self._SYNC_OFF)
            self._write("".join(buf))
            return

        new_top = self._live_top(rows, height)
        old_top = self._frame_top_row
        old_bottom = old_top + self._frame_rows - 1

        if new_top < old_top and old_bottom >= rows:
            # Scroll before erasing the old frame, but never scroll away the
            # visible tail of the latest completed transcript cell.  Command
            # palettes are transient chrome; if there is not enough room after
            # preserving that tail, shrink the palette instead of consuming the
            # answer the user is reading.
            desired_delta = old_top - new_top
            preserve_rows = self._transcript_rows_to_preserve_on_growth(state)
            max_scroll = max(0, self._transcript_rows - preserve_rows)
            delta = min(desired_delta, max_scroll)
            if delta > 0:
                buf.append(self._cup(rows, 1) + "\n" * delta)
                self._transcript_rows = max(0, self._transcript_rows - delta)
                old_top = max(1, old_top - delta)
                old_bottom = max(0, old_bottom - delta)
            if delta < desired_delta:
                new_top = old_top
                available_height = max(1, rows - new_top + 1)
                lines, cursor_row, cursor_col = self._rerender_live_with_height(
                    state,
                    cols,
                    rows,
                    available_height,
                )
                height = len(lines)
        self._cap_transcript_rows_above_live(rows, height)

        clear_top = min(new_top, old_top)
        clear_bottom = max(new_top + height - 1, old_bottom)
        self._clear_rows(buf, clear_top, clear_bottom, rows)
        buf.append(self._cup(new_top, 1))
        buf.append("\r\n".join(lines))

        self._record_live_frame(
            rows=rows,
            cols=cols,
            height=height,
            cursor_row=cursor_row,
            cursor_col=cursor_col,
            anchored=True,
        )
        self._place_cursor_in_frame(
            buf,
            rows=rows,
            cols=cols,
            height=height,
            cursor_row=cursor_row,
            cursor_col=cursor_col,
            anchored=True,
        )
        buf.append(self._AUTOWRAP_ON + self._SYNC_OFF)
        self._write("".join(buf))

    def close(self) -> None:
        _, rows = terminal_size()
        buf = [self._SYNC_ON + self._AUTOWRAP_OFF]
        if self._has_frame:
            if self._frame_anchored:
                frame_top = max(1, min(self._frame_top_row, rows))
                self._clear_rows(buf, frame_top, frame_top + self._frame_rows - 1, rows)
                # Reuse the cleared live frame for the shell prompt.  Leaving the
                # cursor on the bottom cleared row makes the erased status,
                # picker, and composer area look like a block of blank output
                # after the final transcript cell.
                buf.append(self._cup(frame_top, 1))
            else:
                self._clear_floating_frame(buf)
        # Always restore DECAWM so the shell the user returns to does not inherit
        # a nowrap terminal state.
        buf.append(self._AUTOWRAP_ON + "\x1b[?2026l\x1b[0m")
        self._write("".join(buf))
        self._reset_frame()
        self._last_flushed_kind = None
        self._last_flushed_rows = 0
