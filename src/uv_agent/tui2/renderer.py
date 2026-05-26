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
    flicker on terminals that honour it).  We only track the cursor row
    relative to the *top* of the previously emitted frame; that single
    number plus ``\\r\\x1b[NA\\x1b[J`` is enough to erase the old frame
    regardless of whether the terminal scrolled while drawing it.
    """

    def __init__(self, output: TextIO | None = None) -> None:
        self.output = output or sys.stdout
        self.width = terminal_size()[0]
        self._frame_cursor_row = 0
        self._has_frame = False
        self.spinner_frame = 0

    # ------------------------------------------------------------------
    # Compat shims for existing call sites/tests.
    # ------------------------------------------------------------------

    @property
    def live_height(self) -> int:
        return 0 if not self._has_frame else self._frame_cursor_row + 1

    @property
    def cursor_row(self) -> int:
        return self._frame_cursor_row

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _write(self, text: str) -> None:
        self.output.write(text)
        self.output.flush()

    @staticmethod
    def _up(rows: int) -> str:
        return f"\x1b[{rows}A" if rows > 0 else ""

    def _erase_frame(self) -> None:
        """Move to the top of the previous frame and clear to end of screen.

        We end every frame with the cursor at row ``_frame_cursor_row`` of
        that frame, so moving up by exactly that amount lands on row 0.
        ``\\x1b[J`` then wipes everything below.  This is safe even after the
        terminal scrolls: the cursor follows the scroll, and so does the
        frame, so the offset stays correct.
        """

        if not self._has_frame:
            self._write("\r")
            return
        self._write("\r" + self._up(self._frame_cursor_row) + "\x1b[J")
        self._has_frame = False
        self._frame_cursor_row = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def flush_cell(self, cell: TranscriptCell) -> None:
        """Print a completed cell into the terminal's normal scrollback."""

        self.width = terminal_size()[0]
        lines = [
            truncate_visible(line, self.width)
            for line in render_cell(cell, self.width, spinner_frame=self.spinner_frame)
        ]
        if not lines:
            return
        self._write("\x1b[?2026h")
        self._erase_frame()
        # Trailing blank row gives a visual gap between flushed cells; this
        # replaces the per-cell horizontal rule that previously cluttered the
        # transcript.
        self._write("\r\n".join(lines) + "\r\n\r\n")
        self._write("\x1b[?2026l")

    def flush_cells(self, cells: Iterable[TranscriptCell]) -> None:
        for cell in cells:
            self.flush_cell(cell)

    def clear_screen(self, *, rule: str | None = None) -> None:
        """Clear the visible terminal and optionally leave a top separator.

        tui2 uses the normal screen buffer, so clearing must also discard our
        tracked live frame; otherwise the next repaint would try to erase rows
        that no longer exist in the same place.
        """

        self.width = terminal_size()[0]
        self._write("\x1b[?2026h")
        self._erase_frame()
        self._write("\x1b[2J\x1b[H")
        if rule:
            self._write(truncate_visible(rule, self.width) + "\r\n\r\n")
        self._write("\x1b[?2026l")
        self._has_frame = False
        self._frame_cursor_row = 0

    def repaint(self, state: Tui2State) -> None:
        cols, rows = terminal_size()
        self.width = cols
        # The app owns spinner timing. Repaint frequency can spike with streaming
        # deltas, so advancing frames here would make animation speed depend on
        # token throughput instead of wall-clock time.
        # Reserve one row for the terminal prompt/cursor below us so a tight
        # viewport doesn't force the live region against the very last row.
        max_height = max(3, rows - 1)
        lines, cursor_row, cursor_col = render_live_with_cursor(
            state, self.width, self.spinner_frame, max_height=max_height
        )

        self._write("\x1b[?2026h")
        self._erase_frame()
        if not lines:
            self._write("\x1b[?2026l")
            return
        # ``\r\n`` between rows guarantees a column reset; ``\n`` alone is
        # only "move down one row" in raw mode, which would produce a
        # staircase of indented lines on POSIX terminals.
        self._write("\r\n".join(lines))
        last_row = len(lines) - 1
        if cursor_row < last_row:
            self._write(self._up(last_row - cursor_row))
        self._write(f"\r\x1b[{cursor_col + 1}G")
        self._write("\x1b[?2026l")
        self._has_frame = True
        self._frame_cursor_row = cursor_row

    def close(self) -> None:
        self._write("\x1b[?2026h")
        self._erase_frame()
        self._write("\x1b[?2026l\x1b[0m")
