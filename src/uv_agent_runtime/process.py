from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Mapping


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
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
) -> CommandResult:
    """Run a subprocess from a temporary script and capture its text output."""
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=None if env is None else dict(env),
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


def check_command(
    args: list[str],
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
) -> CommandResult:
    """Run a command and raise RuntimeError when it exits non-zero."""
    result = run_command(args, cwd=cwd, env=env, timeout_s=timeout_s)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {args!r}\n{result.stderr}"
        )
    return result
