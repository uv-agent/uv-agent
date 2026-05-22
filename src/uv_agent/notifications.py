from __future__ import annotations

import os
import sys
from typing import Protocol, runtime_checkable


@runtime_checkable
class _BellStream(Protocol):
    """Small stream protocol needed for terminal-bell output."""

    def write(self, text: str) -> object: ...

    def flush(self) -> object: ...


def play_completion_sound() -> bool:
    """Play a completion sound, falling back to the terminal bell."""
    if os.name == "nt" and _play_windows_completion_sound():
        return True
    return ring_terminal_bell()


def _play_windows_completion_sound() -> bool:
    try:
        import winsound

        winsound.PlaySound("SystemNotification", winsound.SND_ALIAS | winsound.SND_ASYNC)
    except (ImportError, RuntimeError):
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except (ImportError, RuntimeError):
            return False
    return True


def ring_terminal_bell() -> bool:
    """Write BEL to an interactive terminal when one is available."""
    if _write_bell(sys.stderr):
        return True
    if sys.stdout is not sys.stderr and _write_bell(sys.stdout):
        return True
    return _write_bell_to_tty()


def _write_bell(stream: object) -> bool:
    if not isinstance(stream, _BellStream):
        return False
    try:
        isatty = getattr(stream, "isatty", None)
        if callable(isatty) and not isatty():
            return False
        stream.write("\a")
        stream.flush()
    except OSError:
        return False
    return True


def _write_bell_to_tty() -> bool:
    try:
        with open("/dev/tty", "w", encoding="utf-8") as tty:
            tty.write("\a")
            tty.flush()
    except OSError:
        return False
    return True
