from __future__ import annotations

from pathlib import Path

from uv_agent_runtime import (
    check_command,
    list_files,
    read_json,
    read_text,
    run_command,
    write_json,
    write_text,
)


def test_runtime_file_helpers(tmp_path: Path) -> None:
    write_text(tmp_path / "a.txt", "hello")
    write_json(tmp_path / "nested" / "data.json", {"ok": True})

    assert read_text(tmp_path / "a.txt") == "hello"
    assert read_json(tmp_path / "nested" / "data.json") == {"ok": True}
    assert "a.txt" in list_files(tmp_path, pattern="*.txt")


def test_runtime_command_helpers() -> None:
    result = run_command(["python", "-c", "print('ok')"])

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"
    assert check_command(["python", "-c", "print('checked')"]).stdout.strip() == "checked"
