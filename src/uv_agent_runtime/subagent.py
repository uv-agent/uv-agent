from __future__ import annotations

from dataclasses import dataclass

from .process import run_command


@dataclass(frozen=True)
class SubagentResult:
    prompt: str
    level: str | None
    returncode: int
    stdout: str
    stderr: str

    @property
    def text(self) -> str:
        """Return stdout trimmed for direct use in follow-up reasoning."""
        return self.stdout.strip()


def ask(
    prompt: str,
    *,
    level: str | None = None,
    cwd: str | None = None,
    timeout_s: float | None = 300,
) -> SubagentResult:
    """Ask a nested uv-agent via the CLI and return its captured output.

    This intentionally uses a subprocess from inside the Python runner, so the
    outer agent still has exactly one external action surface: run_python.
    """
    args = ["uv", "run", "uv-agent"]
    if level:
        args.extend(["--level", level])
    args.extend(["ask", prompt])
    result = run_command(args, cwd=cwd, timeout_s=timeout_s)
    return SubagentResult(
        prompt=prompt,
        level=level,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
