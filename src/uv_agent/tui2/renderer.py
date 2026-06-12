from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TextIO

from uv_agent.tui2.ansi import terminal_size, truncate_visible
from uv_agent.tui2.components import render_cell, render_live_with_cursor
from uv_agent.tui2.events import TranscriptCell, Tui2State


class Renderer:
    """Atomic full-repaint renderer for the tui2 live region.

    The original diff renderer tried to update only changed rows, but the
    cursor/row bookkeeping became unreliable as soon as the live region
    pushed against the bottom of the viewport — terminal scrolling shifted
    physical rows out from under us, so subsequent moves landed in the
    wrong place and we ended up rewriting old content twice.

    This renderer trades that diff for a much simpler model: every frame is
    a full repaint wrapped in CSI 2026 synchronized output (which removes
    flicker on terminals that honour it).  It tracks the live frame's height
    and cursor row, then erases from the tracked top with ``\\r\\x1b[NA\\x1b[J``.
    """

    def __init__(self, output: TextIO | None = None) -> None:
        self.output = output or sys.stdout
        self.width = self._paint_width(terminal_size()[0])
        self._frame_cursor_row = 0
        self._frame_rows = 0
        self._has_frame = False
        self.spinner_frame = 0
        self._last_flushed_kind: str | None = None

    # ------------------------------------------------------------------
    # Compat shims for existing call sites/tests.
    # ------------------------------------------------------------------

    @property
    def live_height(self) -> int:
        return 0 if not self._has_frame else self._frame_rows

    @property
    def cursor_row(self) -> int:
        return self._frame_cursor_row

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_gap_between(last_kind: str | None, current_kind: str) -> bool:
        # A blank row is inserted after every flushed cell so the transcript
        # spacing is stable regardless of cell kind.
        return last_kind is not None

    def _write(self, text: str) -> None:
        self.output.write(text)
        self.output.flush()

    # CSI ?7 toggles DECAWM (auto-wrap mode). With auto-wrap disabled,
    # writing past the last terminal column simply overwrites that column
    # instead of scrolling the viewport, so the frame-erase math in
    # ``_erase_frame`` stays correct even when our cell-width estimate is
    # off (e.g. Braille/Symbol glyphs that some terminals render at 2 cells
    # while ``unicodedata.east_asian_width`` reports them as narrow).
    # The renderer already emits explicit ``\r\n`` between rows, so it
    # never relied on implicit wrapping to advance to the next line.
    _AUTOWRAP_OFF = "\x1b[?7l"
    _AUTOWRAP_ON = "\x1b[?7h"

    @staticmethod
    def _up(rows: int) -> str:
        return f"\x1b[{rows}A" if rows > 0 else ""

    @staticmethod
    def _paint_width(columns: int) -> int:
        """Return a render width that never writes into the last terminal column.

        Some Windows ConPTY/terminal combinations advance to the next row as
        soon as a printable character lands in the rightmost column. The live
        renderer erases previous frames by moving a tracked number of rows up;
        an unexpected auto-wrap row makes that math land below the old frame,
        leaving repeated ``run_python · running`` rules in scrollback. Keeping
        one column empty avoids the terminal-dependent wrap edge while changing
        the visual layout by only a single cell.
        """

        return max(1, columns - 1)

    def _erase_frame(self) -> None:
        """Move to the top of the previous frame and clear to end of screen.

        We end every frame with the cursor at row ``_frame_cursor_row`` of
        that frame, so moving up by exactly that amount lands on row 0.
        ``\\x1b[J`` then wipes everything below.
        """

        if not self._has_frame:
            self._write("\r")
            self._frame_cursor_row = 0
            self._frame_rows = 0
            return
        self._write("\r" + self._up(self._frame_cursor_row) + "\x1b[J")
        self._has_frame = False
        self._frame_cursor_row = 0
        self._frame_rows = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def flush_cell(self, cell: TranscriptCell) -> None:
        """Print a completed cell into the terminal's normal scrollback."""

        self.width = self._paint_width(terminal_size()[0])
        lines = [
            truncate_visible(line, self.width)
            for line in render_cell(cell, self.width, spinner_frame=self.spinner_frame)
        ]
        if not lines:
            return
        self._write("\x1b[?2026h" + self._AUTOWRAP_OFF)
        self._erase_frame()
        # Insert a blank row around turn boundaries (user -> middle,
        # middle -> assistant/user) but pack middle-process cells together.
        if self._last_flushed_kind is not None and self._needs_gap_between(
            self._last_flushed_kind, cell.kind
        ):
            self._write("\r\n")
        self._write("\r\n".join(lines) + "\r\n")
        self._write(self._AUTOWRAP_ON + "\x1b[?2026l")
        self._last_flushed_kind = cell.kind

    def flush_cells(self, cells: Iterable[TranscriptCell]) -> None:
        for cell in cells:
            self.flush_cell(cell)

    def flushed_cell_rows(self, cells: Iterable[TranscriptCell]) -> int:
        """Return how many terminal rows ``flush_cells`` will advance.

        A blank row is counted after every flushed cell so spacing stays stable
        across cell kinds.  The app uses this after loading thread history so the
        first live composer repaint can be padded down to the bottom of a
        mostly-empty viewport.
        """

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
        """Insert blank rows so the next repaint starts near the viewport bottom.

        This is intentionally a one-shot helper for history loads after a clear:
        no live frame is active, and ``preceding_rows`` describes transcript rows
        printed since the clear.  If the history already fills the viewport, no
        padding is emitted and normal terminal scrolling keeps the latest rows in
        view.
        """

        cols, rows = terminal_size()
        self.width = self._paint_width(cols)
        max_height = max(3, rows - 1)
        live_lines, _, _ = render_live_with_cursor(
            state, self.width, self.spinner_frame, max_height=max_height
        )
        target_top = max(0, max_height - len(live_lines))
        pad_rows = max(0, target_top - max(0, preceding_rows))
        if pad_rows:
            self._write("\r\n" * pad_rows)

    def clear_screen(self, *, rule: str | None = None) -> None:
        """Clear the visible terminal and optionally leave a top separator.

        tui2 uses the normal screen buffer, so clearing must also discard our
        tracked live frame; otherwise the next repaint would try to erase rows
        that no longer exist in the same place.
        """

        self.width = self._paint_width(terminal_size()[0])
        self._write("\x1b[?2026h" + self._AUTOWRAP_OFF)
        self._erase_frame()
        # CSI 2J clears the visible viewport; CSI 3J also discards scrollback
        # on terminals that support it.  /clear is an explicit destructive UI
        # reset, so matching the user's expectation of a real terminal clear is
        # preferable to leaving old transcript lines in scrollback.
        self._write("\x1b[2J\x1b[3J\x1b[H")
        if rule:
            self._write(truncate_visible(rule, self.width) + "\r\n\r\n")
        self._write(self._AUTOWRAP_ON + "\x1b[?2026l")
        self._has_frame = False
        self._frame_cursor_row = 0
        self._frame_rows = 0
        self._last_flushed_kind = None

    def repaint(self, state: Tui2State) -> None:
        cols, rows = terminal_size()
        self.width = self._paint_width(cols)
        # The app owns spinner timing. Repaint frequency can spike with streaming
        # deltas, so advancing frames here would make animation speed depend on
        # token throughput instead of wall-clock time.
        # Reserve one row for the terminal prompt/cursor below us so a tight
        # viewport doesn't force the live region against the very last row.
        max_height = max(3, rows - 1)
        lines, cursor_row, cursor_col = render_live_with_cursor(
            state, self.width, self.spinner_frame, max_height=max_height
        )

        self._write("\x1b[?2026h" + self._AUTOWRAP_OFF)
        self._erase_frame()
        if not lines:
            self._write(self._AUTOWRAP_ON + "\x1b[?2026l")
            return
        # ``\r\n`` between rows guarantees a column reset; ``\n`` alone is
        # only "move down one row" in raw mode, which would produce a
        # staircase of indented lines on POSIX terminals.
        self._write("\r\n".join(lines))
        last_row = len(lines) - 1
        if cursor_row < last_row:
            self._write(self._up(last_row - cursor_row))
        self._write(f"\r\x1b[{cursor_col + 1}G")
        self._write(self._AUTOWRAP_ON + "\x1b[?2026l")
        self._has_frame = True
        self._frame_cursor_row = cursor_row
        self._frame_rows = len(lines)

    def close(self) -> None:
        self._write("\x1b[?2026h" + self._AUTOWRAP_OFF)
        self._erase_frame()
        # Always restore DECAWM so the shell the user returns to does not
        # inherit a nowrap terminal state.
        self._write(self._AUTOWRAP_ON + "\x1b[?2026l\x1b[0m")
        self._last_flushed_kind = None
