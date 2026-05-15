from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


def run_command(
    args: list[str],
    *,
    cwd: str | None = None,
    timeout_s: float | None = None,
) -> CommandResult:
    """Run a subprocess from a temporary script and capture its text output."""
    completed = subprocess.run(
        args,
        cwd=cwd,
        timeout=timeout_s,
        text=True,
        capture_output=True,
        check=False,
    )
    return CommandResult(
        args=list(args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
