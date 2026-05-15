from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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

    def raise_for_error(self) -> "SubagentResult":
        """Raise RuntimeError if the subagent exited non-zero."""
        if self.returncode != 0:
            raise RuntimeError(
                f"subagent failed with exit {self.returncode}: {self.stderr or self.stdout}"
            )
        return self


def ask(
    prompt: str,
    *,
    level: str | None = None,
    model_level: str | None = None,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    executable: list[str] | None = None,
    timeout_s: float | None = 300,
    check: bool = False,
) -> SubagentResult:
    """Ask a nested uv-agent via the CLI and return its captured output.

    This intentionally uses a subprocess from inside the Python runner, so the
    outer agent still has exactly one external action surface: run_python.
    """
    selected_level = model_level or level
    args = list(executable or ["uv", "run", "uv-agent"])
    if selected_level:
        args.extend(["--level", selected_level])
    args.extend(["ask", prompt])
    result = run_command(args, cwd=cwd, env=env, timeout_s=timeout_s)
    subagent_result = SubagentResult(
        prompt=prompt,
        level=selected_level,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    if check:
        subagent_result.raise_for_error()
    return subagent_result
