from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncio


@dataclass(frozen=True)
class PythonRunRequest:
    code: str
    script_args: list[str] = field(default_factory=list)
    cwd: Path | None = None
    timeout_s: float | None = None
    thread_id: str | None = None
    thread_kind: str | None = None
    turn_id: str | None = None
    cancel_event: asyncio.Event | None = None


@dataclass(frozen=True)
class RunnerEvent:
    type: str
    data: dict[str, Any]


@dataclass(frozen=True)
class PythonRunResult:
    run_id: str
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    interrupted: bool
    truncated: bool
    run_log_path: Path
    script_path: Path
    events: list[dict[str, Any]] = field(default_factory=list)
