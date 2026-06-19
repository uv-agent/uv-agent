from __future__ import annotations

from uv_agent import notifications


class FakeStream:
    def __init__(self, *, tty: bool) -> None:
        self.tty = tty
        self.text = ""
        self.flushed = False

    def isatty(self) -> bool:
        return self.tty

    def write(self, text: str) -> None:
        self.text += text

    def flush(self) -> None:
        self.flushed = True


def test_ring_terminal_bell_writes_to_stderr_tty(monkeypatch) -> None:
    stderr = FakeStream(tty=True)
    stdout = FakeStream(tty=True)
    monkeypatch.setattr(notifications.sys, "stderr", stderr)
    monkeypatch.setattr(notifications.sys, "stdout", stdout)

    assert notifications.ring_terminal_bell() is True
    assert stderr.text == "\a"
    assert stderr.flushed is True
    assert stdout.text == ""


def test_ring_terminal_bell_falls_back_to_stdout_tty(monkeypatch) -> None:
    stderr = FakeStream(tty=False)
    stdout = FakeStream(tty=True)
    monkeypatch.setattr(notifications.sys, "stderr", stderr)
    monkeypatch.setattr(notifications.sys, "stdout", stdout)
    monkeypatch.setattr(notifications, "_write_bell_to_tty", lambda: False)

    assert notifications.ring_terminal_bell() is True
    assert stderr.text == ""
    assert stdout.text == "\a"


def test_play_completion_sound_uses_terminal_bell_off_windows(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(notifications.os, "name", "posix")
    monkeypatch.setattr(notifications.sys, "platform", "linux")
    monkeypatch.setattr(notifications, "_play_windows_completion_sound", lambda: calls.append("win") or True)
    monkeypatch.setattr(notifications, "_play_macos_completion_sound", lambda: calls.append("mac") or True)
    monkeypatch.setattr(notifications, "ring_terminal_bell", lambda: calls.append("bell") or True)

    assert notifications.play_completion_sound() is True
    assert calls == ["bell"]


def test_play_completion_sound_uses_macos_system_sound(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(notifications.os, "name", "posix")
    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications, "_play_macos_completion_sound", lambda: calls.append("mac") or True)
    monkeypatch.setattr(notifications, "ring_terminal_bell", lambda: calls.append("bell") or True)

    assert notifications.play_completion_sound() is True
    assert calls == ["mac"]


def test_play_terminal_buzzer_uses_windows_beep(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(notifications.os, "name", "nt")
    monkeypatch.setattr(notifications, "_play_windows_terminal_buzzer", lambda: calls.append("win") or True)
    monkeypatch.setattr(notifications, "ring_terminal_bell", lambda: calls.append("bell") or True)

    assert notifications.play_terminal_buzzer() is True
    assert calls == ["win"]


def test_play_terminal_buzzer_falls_back_to_terminal_bell(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(notifications.os, "name", "nt")
    monkeypatch.setattr(notifications, "_play_windows_terminal_buzzer", lambda: calls.append("win") or False)
    monkeypatch.setattr(notifications, "ring_terminal_bell", lambda: calls.append("bell") or True)

    assert notifications.play_terminal_buzzer() is True
    assert calls == ["win", "bell"]


def test_play_terminal_buzzer_uses_macos_system_sound(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(notifications.os, "name", "posix")
    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications, "_play_macos_terminal_buzzer", lambda: calls.append("mac") or True)
    monkeypatch.setattr(notifications, "ring_terminal_bell", lambda: calls.append("bell") or True)

    assert notifications.play_terminal_buzzer() is True
    assert calls == ["mac"]


def test_play_terminal_buzzer_falls_back_to_bell_when_macos_sound_fails(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(notifications.os, "name", "posix")
    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications, "_play_macos_terminal_buzzer", lambda: calls.append("mac") or False)
    monkeypatch.setattr(notifications, "ring_terminal_bell", lambda: calls.append("bell") or True)

    assert notifications.play_terminal_buzzer() is True
    assert calls == ["mac", "bell"]


def test_macos_system_sound_prefers_afplay(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(notifications, "_MACOS_SOUND_DIR", tmp_path)
    (tmp_path / "Tink.aiff").write_bytes(b"sound")
    monkeypatch.setattr(
        notifications,
        "_macos_executable",
        lambda name: f"/usr/bin/{name}" if name == "afplay" else None,
    )
    monkeypatch.setattr(notifications, "_spawn_detached", lambda args: calls.append(list(args)) or True)

    assert notifications._play_macos_system_sound(("Tink.aiff",)) is True
    assert calls == [["/usr/bin/afplay", str(tmp_path / "Tink.aiff")]]


def test_macos_system_sound_falls_back_to_osascript(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(notifications, "_MACOS_SOUND_DIR", tmp_path)
    monkeypatch.setattr(
        notifications,
        "_macos_executable",
        lambda name: "/usr/bin/osascript" if name == "osascript" else None,
    )
    monkeypatch.setattr(notifications, "_spawn_detached", lambda args: calls.append(list(args)) or True)

    assert notifications._play_macos_system_sound(("Missing.aiff",)) is True
    assert calls == [["/usr/bin/osascript", "-e", "beep 1"]]

