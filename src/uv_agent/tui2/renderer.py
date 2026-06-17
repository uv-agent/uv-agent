from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TextIO

from uv_agent.tui2.ansi import terminal_size, truncate_visible
from uv_agent.tui2.components import _needs_gap_between_cells, render_cell, render_live_with_cursor
from uv_agent.tui2.events import TranscriptCell, Tui2State


class Renderer:
    """Append-only transcript renderer with a repaintable live tail.

    tui2 deliberately uses the normal terminal buffer so completed transcript
    cells remain available in the user's real scrollback.  The renderer therefore
    treats completed cells as append-only: once a transcript row has been written
    with normal CRLF flow, later repaints must never clear over it.

    Only the *live tail* (status rows, in-flight cells, composer, pickers/pagers)
    is repaintable.  The first live frame after any transcript append is also
    written with normal flow, which naturally scrolls preceding transcript rows
    out of the viewport before we start tracking the frame.  Subsequent repaints
    may use absolute row erases inside that tracked frame.
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
        # Approximate physical transcript rows emitted since the last explicit
        # clear.  Once this exceeds the viewport the live frame pins to bottom;
        # while it is small the frame floats immediately after the transcript.
        self._transcript_rows = 0
        # A freshly-started process does not know the terminal cursor's absolute
        # row.  Before the first absolute repaint, force a harmless bottom scroll
        # so subsequent CUP/EL operations target only rows we own.
        self._anchor_known = False
        self.spinner_frame = 0
        self._last_flushed_kind: str | None = None

    # ------------------------------------------------------------------
    # Compat shims for existing call sites/tests.
    # ------------------------------------------------------------------

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

    def _clear_rows(self, buf: list[str], top: int, bottom: int, rows: int) -> None:
        for row in range(max(1, top), min(rows, bottom) + 1):
            buf.append(self._cup(row, 1) + self._EL)

    def _reset_frame(self) -> None:
        self._has_frame = False
        self._frame_top_row = 0
        self._frame_rows = 0
        self._frame_cursor_row = 0
        self._frame_cursor_col = 0

    def _reserve_bottom_anchor(self, buf: list[str], rows: int, height: int) -> bool:
        """Reserve a known bottom frame without erasing existing transcript.

        Absolute CUP/EL repainting is only safe after we know where the live
        frame lives in the viewport.  On process start, the shell could have left
        the cursor on any row.  Moving down to the bottom margin and emitting one
        CRLF per live row scrolls existing content into scrollback instead of
        overwriting it; the following frame is then painted into those bottom
        rows at known absolute coordinates.
        """

        if self._anchor_known:
            return False
        buf.append(f"\r\x1b[{max(1, rows)}B" + "\r\n" * max(1, height))
        self._transcript_rows = max(self._transcript_rows, rows)
        self._anchor_known = True
        return True

    def _render_live(
        self,
        state: Tui2State,
        cols: int,
        rows: int,
    ) -> tuple[list[str], int, int]:
        max_height = max(3, rows - 1)
        return render_live_with_cursor(
            state,
            self.width,
            self.spinner_frame,
            max_height=max_height,
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
    ) -> tuple[int, int]:
        top = self._live_top(rows, height)
        self._has_frame = True
        self._frame_top_row = top
        self._frame_rows = height
        self._frame_cursor_row = cursor_row
        self._frame_cursor_col = cursor_col
        final_row = max(1, min(rows, top + cursor_row))
        final_col = max(1, min(cols, cursor_col + 1))
        return final_row, final_col

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

        This method intentionally does not clear any rows first.  It is used only
        when no live frame is currently active, typically immediately after
        transcript rows have been appended.  Normal CRLF flow is the critical
        part: if the transcript already fills the viewport, writing the live frame
        scrolls transcript rows into real scrollback instead of overwriting them.
        """

        if not lines:
            self._reset_frame()
            return
        if self._reserve_bottom_anchor(buf, rows, len(lines)):
            buf.append(self._cup(self._live_top(rows, len(lines)), 1))
        buf.append("\r\n".join(lines))
        final_row, final_col = self._record_live_frame(
            rows=rows,
            cols=cols,
            height=len(lines),
            cursor_row=cursor_row,
            cursor_col=cursor_col,
        )
        buf.append(self._cup(final_row, final_col))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def paint_live_after_transcript(self, state: Tui2State) -> None:
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

    def flush_cell(self, cell: TranscriptCell, live_state: Tui2State | None = None) -> None:
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
            self._clear_rows(buf, self._frame_top_row, self._frame_top_row + self._frame_rows - 1, rows)
            buf.append(self._cup(max(1, min(self._frame_top_row, rows)), 1))
        buf.append("\r\n".join(hard_lines) + "\r\n")
        self._transcript_rows += len(hard_lines)
        if self._transcript_rows >= rows:
            self._anchor_known = True
        self._last_flushed_kind = cell.kind
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

    def flush_cells(self, cells: Iterable[TranscriptCell], live_state: Tui2State | None = None) -> None:
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

    def pad_live_region_to_bottom(self, state: Tui2State, *, preceding_rows: int = 0) -> None:
        """Backward-compatible one-shot padding helper for older callers.

        New history rendering prefers ``flush_cells(..., live_state=state)`` so
        the live frame is written with normal flow after the transcript.  This
        helper remains for tests/embedders that still want the old padding
        behaviour after an explicit clear.
        """

        cols, rows = terminal_size()
        self.width = self._paint_width(cols)
        max_height = max(3, rows - 1)
        live_lines, _, _ = render_live_with_cursor(
            state,
            self.width,
            self.spinner_frame,
            max_height=max_height,
            preceding_kind=self._last_flushed_kind,
            has_preceding_transcript=self._last_flushed_kind is not None,
        )
        target_top = max(0, max_height - len(live_lines))
        pad_rows = max(0, target_top - max(0, preceding_rows))
        if pad_rows:
            self._write("\r\n" * pad_rows)
            self._transcript_rows += pad_rows
            if self._transcript_rows >= rows:
                self._anchor_known = True

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

    def repaint(self, state: Tui2State) -> None:
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
            self._clear_rows(buf, self._frame_top_row, self._frame_top_row + self._frame_rows - 1, rows)
            buf.append(self._AUTOWRAP_ON + self._SYNC_OFF)
            self._write("".join(buf))
            self._reset_frame()
            return

        height = len(lines)
        new_top = self._live_top(rows, height)
        old_top = self._frame_top_row
        old_bottom = old_top + self._frame_rows - 1

        # When the live region grows upward while already pinned to the bottom,
        # scroll the viewport up so the transcript row it now covers is pushed
        # into real scrollback instead of being painted over.
        if new_top < old_top and old_bottom >= rows:
            self._clear_rows(buf, old_top, old_bottom, rows)
            delta = old_top - new_top
            buf.append(self._cup(rows, 1) + "\n" * delta)
            old_top = new_top
            old_bottom = 0

        clear_top = min(new_top, old_top)
        clear_bottom = max(new_top + height - 1, old_bottom)
        self._clear_rows(buf, clear_top, clear_bottom, rows)
        buf.append(self._cup(new_top, 1))
        buf.append("\r\n".join(lines))

        final_row, final_col = self._record_live_frame(
            rows=rows,
            cols=cols,
            height=height,
            cursor_row=cursor_row,
            cursor_col=cursor_col,
        )
        buf.append(self._cup(final_row, final_col))
        buf.append(self._AUTOWRAP_ON + self._SYNC_OFF)
        self._write("".join(buf))

    def close(self) -> None:
        _, rows = terminal_size()
        buf = [self._SYNC_ON + self._AUTOWRAP_OFF]
        if self._has_frame:
            self._clear_rows(buf, self._frame_top_row, self._frame_top_row + self._frame_rows - 1, rows)
        # Always restore DECAWM so the shell the user returns to does not inherit
        # a nowrap terminal state.
        buf.append(self._AUTOWRAP_ON + "\x1b[?2026l\x1b[0m")
        self._write("".join(buf))
        self._reset_frame()
        self._last_flushed_kind = None
