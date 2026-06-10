from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from uv_agent.prompts import SUBAGENT_LEGACY_UNAVAILABLE

# Backward-compatible constant names for older imports.
ASK_REPLACED_MESSAGE = SUBAGENT_LEGACY_UNAVAILABLE
NESTED_ASK_BLOCKED_MESSAGE = SUBAGENT_LEGACY_UNAVAILABLE


@dataclass(frozen=True)
class SubagentResult:
    level: str | None
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    thread_id: str | None = None

    @property
    def text(self) -> str:
        return self.stdout.strip()

    def raise_for_error(self) -> "SubagentResult":
        if self.returncode != 0:
            raise RuntimeError(self.stderr or self.stdout or ASK_REPLACED_MESSAGE)
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
    retain: bool = True,
) -> SubagentResult:
    """Compatibility shim for removed ask helper."""

    del prompt, level, model_level, cwd, env, executable, timeout_s, check, retain
    raise RuntimeError(ASK_REPLACED_MESSAGE)


def _extract_subagent_thread_id(stderr: str) -> str | None:
    del stderr
    return None
