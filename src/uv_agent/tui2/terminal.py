from __future__ import annotations

import os
import sys
from contextlib import AbstractContextManager
from time import monotonic, sleep
from typing import TextIO


PASTE_PREFIX = "\x00paste\x00"


class Terminal(AbstractContextManager["Terminal"]):
    """Small raw terminal wrapper for tui2.

    The class uses the normal screen buffer on purpose. It only enables raw-ish
    input and bracketed paste while the app is active; transcript lines written
    before the live region remain in the host terminal scrollback.
    """

    def __init__(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self._old_termios = None
        self._old_input_mode: int | None = None
        self._old_output_mode: int | None = None
        self._windows = os.name == "nt"

    def __enter__(self) -> "Terminal":
        if self._windows:
            self._enable_windows_vt()
        else:
            import termios
            import tty

            fd = self.stdin.fileno()
            self._old_termios = termios.tcgetattr(fd)
            tty.setraw(fd)
        self.write("\x1b[?2004h")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.write("\x1b[?2004l\x1b[0m\n")
        if not self._windows and self._old_termios is not None:
            import termios

            termios.tcsetattr(self.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
        if self._windows:
            self._restore_windows_vt()

    def write(self, data: str) -> None:
        self.stdout.write(data)
        self.stdout.flush()

    def read_key(self) -> str:
        ch = self._read_char()
        if self._windows:
            # Ctrl+C is often delivered as KeyboardInterrupt by getwch rather
            # than as the literal ETX byte. Normalize both forms for app logic.
            if ch in {"\x00", "\xe0"}:
                code = self._read_char()
                if code == "\n":
                    return "<C-ENTER>"
                if code == "M":
                    return "<RIGHT>"
                if code == "K":
                    return "<LEFT>"
                if code == "H":
                    return "<UP>"
                if code == "P":
                    return "<DOWN>"
                return "<" + code + ">"
        if ch != "\x1b":
            return ch
        # Terminals report arrows, modified Enter and bracketed paste as CSI
        # sequences.  Bracketed paste must be returned as one key so pasted
        # newlines don't look like Enter presses to the app.
        second = self._read_char_after_escape()
        if second != "[":
            return "\x1b"
        return self._read_csi_key()

    def _read_char(self) -> str:
        if self._windows:
            import msvcrt

            return msvcrt.getwch()
        return self.stdin.read(1)

    def _read_char_after_escape(self) -> str | None:
        """Read the next escape-sequence byte without making bare Esc sticky.

        Interactive terminals may send a lone Esc key.  If no additional byte is
        ready shortly after Esc, treat it as a standalone key instead of blocking
        forever.  In tests and other in-memory streams, ``fileno()`` is usually
        unavailable, so a normal read is non-blocking and deterministic.
        """

        if self._windows:
            import msvcrt

            if msvcrt.kbhit():
                return self._read_char()
            # Tests may replace getwch with an in-memory iterator while kbhit()
            # still reflects the real console, so fall back to a direct read in
            # that case.  In normal interactive use, keep the timeout path so a
            # lone Esc key remains responsive.
            if getattr(msvcrt.getwch, "__module__", "msvcrt") != "msvcrt":
                return self._read_char()
            deadline = monotonic() + 0.03
            while monotonic() < deadline:
                if msvcrt.kbhit():
                    return self._read_char()
                sleep(0.001)
            return None

        try:
            fd = self.stdin.fileno()
        except (AttributeError, OSError):
            return self._read_char() or None
        import select

        ready, _, _ = select.select([fd], [], [], 0.03)
        if not ready:
            return None
        return self._read_char()

    def _read_csi_key(self) -> str:
        sequence = "\x1b["
        while len(sequence) < 32:
            ch = self._read_char()
            if not ch:
                break
            sequence += ch
            if ch == "~" or ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
                break

        if sequence == "\x1b[A":
            return "<UP>"
        if sequence == "\x1b[B":
            return "<DOWN>"
        if sequence == "\x1b[C":
            return "<RIGHT>"
        if sequence == "\x1b[D":
            return "<LEFT>"
        if sequence == "\x1b[27;5;13~":
            return "<C-ENTER>"
        if sequence == "\x1b[200~":
            return PASTE_PREFIX + self._read_bracketed_paste()
        return "\x1b"

    def _read_bracketed_paste(self) -> str:
        terminator = "\x1b[201~"
        chars: list[str] = []
        while True:
            ch = self._read_char()
            if not ch:
                break
            chars.append(ch)
            if len(chars) >= len(terminator) and "".join(chars[-len(terminator) :]) == terminator:
                del chars[-len(terminator) :]
                break
        return self._normalize_paste_text("".join(chars))

    @staticmethod
    def _normalize_paste_text(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _enable_windows_vt(self) -> None:
        """Enable VT input/output when running under a Windows console.

        Bracketed paste only helps if the console is allowed to deliver VT input
        sequences such as ``ESC [ 200 ~``.  If the handle calls fail (for
        example under redirected stdio), fall back to the classic no-op trick
        that enables ANSI output on older consoles.
        """

        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            mode = wintypes.DWORD()
            input_handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
            if kernel32.GetConsoleMode(input_handle, ctypes.byref(mode)):
                self._old_input_mode = int(mode.value)
                kernel32.SetConsoleMode(input_handle, mode.value | 0x0200)  # ENABLE_VIRTUAL_TERMINAL_INPUT

            output_handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            if kernel32.GetConsoleMode(output_handle, ctypes.byref(mode)):
                self._old_output_mode = int(mode.value)
                kernel32.SetConsoleMode(output_handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            os.system("")  # best-effort ANSI output support on classic consoles

    def _restore_windows_vt(self) -> None:
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            if self._old_input_mode is not None:
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), self._old_input_mode)
            if self._old_output_mode is not None:
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), self._old_output_mode)
        except Exception:
            return
