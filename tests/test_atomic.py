from __future__ import annotations

import os
from pathlib import Path

import pytest

from uv_agent import atomic


def _write(path: Path, text: str = "data") -> None:
    path.write_text(text, encoding="utf-8")


def test_atomic_replace_success(tmp_path: Path) -> None:
    src = tmp_path / "src.json.tmp"
    dest = tmp_path / "src.json"
    _write(src, "new")
    _write(dest, "old")

    atomic.atomic_replace(src, dest)

    assert not src.exists()
    assert dest.read_text(encoding="utf-8") == "new"


def test_atomic_replace_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = tmp_path / "src.json.tmp"
    dest = tmp_path / "src.json"
    _write(src, "new")
    _write(dest, "old")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(a: str, b: str) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Access is denied", str(b))
        real_replace(a, b)

    monkeypatch.setattr(atomic.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic.time, "sleep", lambda _s: None)

    atomic.atomic_replace(src, dest)

    assert calls["n"] == 3
    assert dest.read_text(encoding="utf-8") == "new"
    assert not src.exists()


def test_atomic_replace_clears_readonly_before_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = tmp_path / "src.json.tmp"
    dest = tmp_path / "src.json"
    _write(src, "new")
    _write(dest, "old")

    cleared: list[Path] = []

    def fake_clear_readonly(path: Path) -> None:
        cleared.append(Path(path))

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(a: str, b: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(5, "Access is denied", str(b))
        real_replace(a, b)

    monkeypatch.setattr(atomic, "_clear_readonly", fake_clear_readonly)
    monkeypatch.setattr(atomic.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic.time, "sleep", lambda _s: None)

    atomic.atomic_replace(src, dest)

    assert cleared == [dest]


def test_atomic_replace_gives_up_after_max_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = tmp_path / "src.json.tmp"
    dest = tmp_path / "src.json"
    _write(src, "new")
    _write(dest, "old")

    calls = {"n": 0}

    def always_denied(a: str, b: str) -> None:
        calls["n"] += 1
        raise PermissionError(5, "Access is denied", str(b))

    monkeypatch.setattr(atomic.os, "replace", always_denied)
    monkeypatch.setattr(atomic.time, "sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        atomic.atomic_replace(src, dest)

    # Initial attempt + every retry delay must be exercised before giving up.
    assert calls["n"] == 1 + len(atomic._RETRY_DELAYS_SECONDS)
