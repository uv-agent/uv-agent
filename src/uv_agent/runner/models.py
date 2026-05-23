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
    script_path: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-serializable payload exposed as a tool result.

        The runner is used by both the engine's normal blocking path and the
        streaming path that can surface partial output while a process is still
        running. Keeping the shape in one place avoids subtle differences in
        what the model, TUI, and persisted history see for completed runs.
        """

        return {
            "run_id": self.run_id,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "interrupted": self.interrupted,
            "truncated": self.truncated,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "events": self.events,
        }
