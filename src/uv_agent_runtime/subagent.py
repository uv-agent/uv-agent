from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from typing import Mapping

from .events import emit_event
from .textops import run_process_text

NESTED_ASK_BLOCKED_MESSAGE = (
    "ask is unavailable inside a subagent thread. This project only permits one "
    "level of ask delegation so nested subagents do not grow unbounded or require "
    "tree-shaped UI tracking. Complete the task yourself with the current context "
    "and available runtime helpers instead of delegating again."
)


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
    if os.environ.get("UV_AGENT_RUNTIME_THREAD_KIND") == "subagent":
        emit_event("subagent.blocked", reason="nested_ask_disabled")
        result = SubagentResult(
            level=model_level or level,
            returncode=2,
            stdout="",
            stderr=NESTED_ASK_BLOCKED_MESSAGE,
            timed_out=False,
            thread_id=None,
        )
        if check:
            result.raise_for_error()
        return result
    selected_level = model_level or level
    args = list(executable or ["uv", "run", "uv-agent"])
    if selected_level:
        args.extend(["--level", selected_level])
    parent_thread_id = os.environ.get("UV_AGENT_RUNTIME_THREAD_ID")
    parent_turn_id = os.environ.get("UV_AGENT_RUNTIME_TURN_ID")
    parent_run_id = os.environ.get("UV_AGENT_RUNTIME_RUN_ID")
    state_dir = os.environ.get("UV_AGENT_RUNTIME_STATE_DIR")
    using_default_executable = executable is None
    if retain and using_default_executable:
        args.extend(["--thread-kind", "subagent"])
        if parent_thread_id:
            args.extend(["--parent-thread", parent_thread_id])
        if parent_turn_id:
            args.extend(["--parent-turn", parent_turn_id])
        if parent_run_id:
            args.extend(["--parent-run", parent_run_id])
    if using_default_executable:
        # ``ask`` is consumed programmatically by the parent agent. Capturing
        # only the completed answer avoids mistaking streamed pre-tool narration
        # for a final subagent result when a timeout or provider error happens.
        args.append("--no-stream")
    emit_event(
        "subagent.started",
        level=selected_level,
        parent_thread_id=parent_thread_id,
        parent_turn_id=parent_turn_id,
    )
    subagent_env = dict(env) if env is not None else os.environ.copy()
    if retain and state_dir:
        if using_default_executable:
            args.extend(["--project-state-dir", state_dir])
        else:
            subagent_env["UV_AGENT_RUNTIME_PROJECT_STATE_DIR"] = state_dir
        result = run_process_text(args + ["ask", prompt], cwd=cwd, env=subagent_env, timeout_s=timeout_s)
    else:
        with tempfile.TemporaryDirectory(prefix="uv-agent-subagent-") as temporary_state_dir:
            if using_default_executable:
                args.extend(["--project-state-dir", temporary_state_dir])
            else:
                subagent_env["UV_AGENT_RUNTIME_PROJECT_STATE_DIR"] = temporary_state_dir
            result = run_process_text(args + ["ask", prompt], cwd=cwd, env=subagent_env, timeout_s=timeout_s)
    thread_id = _extract_subagent_thread_id(result.stderr)
    emit_event(
        "subagent.completed",
        level=selected_level,
        returncode=result.returncode,
        timed_out=result.timed_out,
        thread_id=thread_id,
        summary=result.stdout.strip()[:500],
    )
    subagent_result = SubagentResult(
        level=selected_level,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=result.timed_out,
        thread_id=thread_id,
    )
    if check:
        subagent_result.raise_for_error()
    return subagent_result


def _extract_subagent_thread_id(stderr: str) -> str | None:
    match = re.search(r"^\[subagent-thread\]\s+(\S+)\s*$", stderr, flags=re.MULTILINE)
    return match.group(1) if match else None
