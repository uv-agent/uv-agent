from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from typing import Mapping

from .events import emit_event
from .process import run_command


@dataclass(frozen=True)
class SubagentResult:
    prompt: str
    level: str | None
    returncode: int
    stdout: str
    stderr: str
    thread_id: str | None = None

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
    retain: bool = True,
) -> SubagentResult:
    """Ask a nested uv-agent via the CLI and return its captured output.

    This intentionally uses a subprocess from inside the Python runner, so the
    outer agent still has exactly one external action surface: run_python.
    """
    selected_level = model_level or level
    args = list(executable or ["uv", "run", "uv-agent"])
    if selected_level:
        args.extend(["--level", selected_level])
    parent_thread_id = os.environ.get("UV_AGENT_THREAD_ID")
    parent_turn_id = os.environ.get("UV_AGENT_TURN_ID")
    parent_run_id = os.environ.get("UV_AGENT_RUN_ID")
    parent_script_id = os.environ.get("UV_AGENT_SCRIPT_ID")
    state_dir = os.environ.get("UV_AGENT_STATE_DIR")
    using_default_executable = executable is None
    if retain and using_default_executable:
        args.extend(["--thread-kind", "subagent"])
        if parent_thread_id:
            args.extend(["--parent-thread", parent_thread_id])
        if parent_turn_id:
            args.extend(["--parent-turn", parent_turn_id])
        if parent_run_id:
            args.extend(["--parent-run", parent_run_id])
        if parent_script_id:
            args.extend(["--parent-script", parent_script_id])
    args.extend(["ask", prompt])
    emit_event(
        "subagent.started",
        prompt=prompt,
        level=selected_level,
        parent_thread_id=parent_thread_id,
        parent_turn_id=parent_turn_id,
    )
    subagent_env = dict(env) if env is not None else os.environ.copy()
    if retain and state_dir:
        subagent_env["UV_AGENT_PROJECT_STATE_DIR"] = state_dir
        result = run_command(args, cwd=cwd, env=subagent_env, timeout_s=timeout_s)
    else:
        with tempfile.TemporaryDirectory(prefix="uv-agent-subagent-") as temporary_state_dir:
            subagent_env["UV_AGENT_PROJECT_STATE_DIR"] = temporary_state_dir
            result = run_command(args, cwd=cwd, env=subagent_env, timeout_s=timeout_s)
    thread_id = _extract_subagent_thread_id(result.stderr)
    emit_event(
        "subagent.completed",
        prompt=prompt,
        level=selected_level,
        returncode=result.returncode,
        thread_id=thread_id,
        summary=result.stdout.strip()[:500],
    )
    subagent_result = SubagentResult(
        prompt=prompt,
        level=selected_level,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        thread_id=thread_id,
    )
    if check:
        subagent_result.raise_for_error()
    return subagent_result


def _extract_subagent_thread_id(stderr: str) -> str | None:
    match = re.search(r"^\[subagent-thread\]\s+(\S+)\s*$", stderr, flags=re.MULTILINE)
    return match.group(1) if match else None
