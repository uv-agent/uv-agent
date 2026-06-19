from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


_MACOS_SOUND_DIR = Path("/System/Library/Sounds")
_MACOS_COMPLETION_SOUNDS = ("Glass.aiff", "Ping.aiff", "Hero.aiff")
_MACOS_BUZZER_SOUNDS = ("Tink.aiff", "Glass.aiff", "Ping.aiff")


@runtime_checkable
class _BellStream(Protocol):
    """Small stream protocol needed for terminal-bell output."""

    def write(self, text: str) -> object: ...

    def flush(self) -> object: ...


def play_completion_sound() -> bool:
    """Play a completion sound, falling back to the terminal bell."""
    if os.name == "nt":
        if _play_windows_completion_sound():
            return True
    elif sys.platform == "darwin" and _play_macos_completion_sound():
        return True
    return ring_terminal_bell()


def play_terminal_buzzer() -> bool:
    """Play a short buzzer-like cue for terminal-native interfaces."""
    if os.name == "nt":
        if _play_windows_terminal_buzzer():
            return True
    elif sys.platform == "darwin" and _play_macos_terminal_buzzer():
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


def _play_windows_terminal_buzzer() -> bool:
    try:
        import winsound

        winsound.Beep(880, 80)
    except (ImportError, RuntimeError, ValueError):
        return False
    return True


def _play_macos_completion_sound() -> bool:
    return _play_macos_system_sound(_MACOS_COMPLETION_SOUNDS)


def _play_macos_terminal_buzzer() -> bool:
    return _play_macos_system_sound(_MACOS_BUZZER_SOUNDS)


def _play_macos_system_sound(sound_names: Sequence[str]) -> bool:
    """Play a macOS alert sound without relying on terminal BEL settings."""

    afplay = _macos_executable("afplay")
    if afplay is not None:
        for sound_name in sound_names:
            sound_path = _MACOS_SOUND_DIR / sound_name
            if sound_path.is_file() and _spawn_detached([afplay, str(sound_path)]):
                return True

    osascript = _macos_executable("osascript")
    if osascript is not None:
        return _spawn_detached([osascript, "-e", "beep 1"])
    return False


def _macos_executable(name: str) -> str | None:
    executable = shutil.which(name)
    if executable:
        return executable
    fallback = Path("/usr/bin") / name
    if fallback.exists():
        return str(fallback)
    return None


def _spawn_detached(args: Sequence[str]) -> bool:
    try:
        subprocess.Popen(
            list(args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, ValueError):
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
