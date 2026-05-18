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
    monkeypatch.setattr(notifications, "_play_windows_completion_sound", lambda: calls.append("win") or True)
    monkeypatch.setattr(notifications, "ring_terminal_bell", lambda: calls.append("bell") or True)

    assert notifications.play_completion_sound() is True
    assert calls == ["bell"]

