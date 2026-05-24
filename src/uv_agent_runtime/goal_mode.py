from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class RuntimeGoalPaths:
    """Goal-mode file locations for the current run_python thread."""

    directory: Path
    state: Path
    checklist: Path
    notes: Path


def goal_paths() -> RuntimeGoalPaths:
    """Return the durable goal files for this run_python thread.

    The helper relies on the thread/project-state environment supplied by the
    managed runner. It intentionally does not create files; goal mode itself owns
    lifecycle and reset behavior.
    """

    state_dir = os.environ.get("UV_AGENT_RUNTIME_STATE_DIR")
    thread_id = os.environ.get("UV_AGENT_RUNTIME_THREAD_ID")
    if not state_dir or not thread_id:
        raise RuntimeError("goal_paths requires a run_python execution attached to a uv-agent thread")
    directory = Path(state_dir).resolve() / "goals" / thread_id
    return RuntimeGoalPaths(
        directory=directory,
        state=directory / "goal.json",
        checklist=directory / "checklist.md",
        notes=directory / "notes.md",
    )
