from __future__ import annotations

import os
import sys
from contextlib import AbstractContextManager
from typing import TextIO


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
        self._windows = os.name == "nt"

    def __enter__(self) -> "Terminal":
        if self._windows:
            os.system("")  # enable VT processing on classic Windows consoles
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

    def write(self, data: str) -> None:
        self.stdout.write(data)
        self.stdout.flush()

    def read_key(self) -> str:
        if self._windows:
            import msvcrt

            ch = msvcrt.getwch()
            # Ctrl+C is often delivered as KeyboardInterrupt by getwch rather
            # than as the literal ETX byte. Normalize both forms for app logic.
            if ch in {"\x00", "\xe0"}:
                code = msvcrt.getwch()
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
            return ch
        ch = self.stdin.read(1)
        if ch != "\x1b":
            return ch
        # POSIX terminals report arrows and modified Enter as CSI sequences in
        # raw mode.  Read only the short sequences we need so Escape remains a
        # harmless no-op for the app.
        second = self.stdin.read(1)
        if second != "[":
            return "\x1b"
        third = self.stdin.read(1)
        if third == "A":
            return "<UP>"
        if third == "B":
            return "<DOWN>"
        if third == "C":
            return "<RIGHT>"
        if third == "D":
            return "<LEFT>"
        if third == "2" and self.stdin.read(1) == "7" and self.stdin.read(1) == ";":
            # xterm-style Ctrl+Enter: ESC [ 27 ; 5 ; 13 ~
            modifier = self.stdin.read(1)
            semi = self.stdin.read(1)
            key = self.stdin.read(2)
            if modifier == "5" and semi == ";" and key == "13" and self.stdin.read(1) == "~":
                return "<C-ENTER>"
        return "\x1b"
