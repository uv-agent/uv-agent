from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class PythonRunRequest:
    code: str
    uv_args: list[str] = field(default_factory=list)
    script_args: list[str] = field(default_factory=list)
    cwd: Path | None = None
    timeout_s: float | None = None
    thread_id: str | None = None
    turn_id: str | None = None


@dataclass(frozen=True)
class RerunRequest:
    script_id: str | None = None
    run_id: str | None = None
    mode: Literal["rerun", "replay"] = "rerun"
    uv_args: list[str] | None = None
    script_args: list[str] | None = None
    cwd: Path | None = None
    timeout_s: float | None = None
    thread_id: str | None = None
    turn_id: str | None = None


@dataclass(frozen=True)
class RunnerEvent:
    type: str
    data: dict[str, Any]


@dataclass(frozen=True)
class PythonRunResult:
    script_id: str
    run_id: str
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool
    run_log_path: Path
    script_path: Path
    final_script_path: Path
    events: list[dict[str, Any]] = field(default_factory=list)
