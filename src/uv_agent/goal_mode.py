from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape as xml_escape
from pathlib import Path
from typing import Any, Literal

from uv_agent.time import utc_now_iso

GoalModeStatus = Literal["enabled", "disabled"]

GOAL_STATE_FILENAME = "goal.json"
GOAL_CHECKLIST_FILENAME = "checklist.md"
GOAL_NOTES_FILENAME = "notes.md"


def _xml_text(value: object) -> str:
    return xml_escape(str(value), quote=False)


@dataclass(frozen=True)
class GoalPaths:
    """Filesystem locations backing the goal state for one thread."""

    directory: Path
    state: Path
    checklist: Path
    notes: Path


@dataclass(frozen=True)
class GoalState:
    """Current persisted goal-mode state for one thread."""

    enabled: bool
    status: str
    paths: GoalPaths
    objective: str = ""
    created_at: str = ""
    updated_at: str = ""


def goal_paths(project_state_dir: Path, thread_id: str) -> GoalPaths:
    """Return the stable internal goal file locations for ``thread_id``."""

    directory = Path(project_state_dir).resolve() / "goals" / thread_id
    return GoalPaths(
        directory=directory,
        state=directory / GOAL_STATE_FILENAME,
        checklist=directory / GOAL_CHECKLIST_FILENAME,
        notes=directory / GOAL_NOTES_FILENAME,
    )


def ensure_goal_files(
    project_state_dir: Path,
    thread_id: str,
    *,
    objective: str = "",
    reset: bool = False,
) -> GoalState:
    """Create or reset the durable files used by goal mode for one thread.

    ``reset=False`` preserves any existing user/model-maintained content. This is
    important because toggling goal mode should not erase the long-lived work
    memory the mode is designed to protect.
    """

    paths = goal_paths(project_state_dir, thread_id)
    paths.directory.mkdir(parents=True, exist_ok=True)
    now = utc_now_iso()
    existing = _read_goal_json(paths.state)
    resolved_objective = objective.strip() or str(existing.get("objective") or "")
    created_at = str(existing.get("created_at") or now)
    state_payload: dict[str, Any] = {
        "thread_id": thread_id,
        "objective": resolved_objective,
        "created_at": now if reset else created_at,
        "updated_at": now,
        "files": {
            "checklist": str(paths.checklist),
            "notes": str(paths.notes),
        },
    }
    _write_json(paths.state, state_payload)
    if reset or not paths.checklist.exists():
        paths.checklist.write_text(_checklist_template(resolved_objective), encoding="utf-8", newline="\n")
    if reset or not paths.notes.exists():
        paths.notes.write_text(_notes_template(resolved_objective), encoding="utf-8", newline="\n")
    return GoalState(
        enabled=False,
        status="disabled",
        paths=paths,
        objective=resolved_objective,
        created_at=str(state_payload["created_at"]),
        updated_at=now,
    )


def read_goal_state(project_state_dir: Path, thread_id: str, *, enabled: bool) -> GoalState:
    """Read goal metadata and paths without creating files."""

    paths = goal_paths(project_state_dir, thread_id)
    data = _read_goal_json(paths.state)
    objective = str(data.get("objective") or "") if isinstance(data, dict) else ""
    created_at = str(data.get("created_at") or "") if isinstance(data, dict) else ""
    updated_at = str(data.get("updated_at") or "") if isinstance(data, dict) else ""
    return GoalState(
        enabled=enabled,
        status="enabled" if enabled else "disabled",
        paths=paths,
        objective=objective,
        created_at=created_at,
        updated_at=updated_at,
    )


def render_goal_mode_notice(state: GoalState, *, status: GoalModeStatus) -> str:
    """Render the stable model-visible notice for a goal mode transition/epoch."""

    if status == "disabled":
        return "\n".join(
            [
                '<goal_mode status="disabled">',
                "Goal mode is now disabled for this thread.",
                "",
                "<files>",
                f"<state>{_xml_text(state.paths.state)}</state>",
                f"<checklist>{_xml_text(state.paths.checklist)}</checklist>",
                f"<document>{_xml_text(state.paths.notes)}</document>",
                "</files>",
                "",
                "<rule>The existing goal files are preserved, but they are no longer active durable memory unless goal mode is enabled again.</rule>",
                "</goal_mode>",
            ]
        )

    lines = [
        '<goal_mode status="enabled">',
        "Goal mode is active for this thread.",
    ]
    if state.objective.strip():
        lines.extend(["", f"<objective>{_xml_text(state.objective.strip())}</objective>"])
    lines.extend(
        [
            "",
            "<files>",
            f"<state>{_xml_text(state.paths.state)}</state>",
            f"<checklist>{_xml_text(state.paths.checklist)}</checklist>",
            f"<document>{_xml_text(state.paths.notes)}</document>",
            "</files>",
            "",
            "<rules>",
            "<rule>Use these files as durable external memory for this thread goal.</rule>",
            "<rule>Maintain checklist.md for acceptance criteria, tasks, progress, blockers, and the next step.</rule>",
            "<rule>Maintain notes.md for decisions, investigation notes, constraints, and handoff context.</rule>",
            "<rule>Read or update the files with run_python when goal progress changes or when resuming from unclear context.</rule>",
            "<rule>Do not paste full goal files into chat unless the user asks or it is necessary.</rule>",
            "<rule>During compaction or resume, prefer these files over conversation memory for goal progress.</rule>",
            "</rules>",
            "</goal_mode>",
        ]
    )
    return "\n".join(lines)


def _read_goal_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _checklist_template(objective: str) -> str:
    objective_line = objective.strip() or "Describe the goal here."
    return f"""# Goal Checklist

Objective: {objective_line}

## Acceptance Criteria

- [ ] Define what complete means for this goal.

## Tasks

- [ ] Capture the first concrete task.

## Current Next Step

- Decide the next action.

## Blockers

- None recorded.
"""


def _notes_template(objective: str) -> str:
    objective_line = objective.strip() or "Describe the goal here."
    return f"""# Goal Notes

Objective: {objective_line}

## Decisions

- None recorded.

## Investigation Notes

- None recorded.

## Handoff Context

- Keep this section updated with concise context needed after compaction or resume.
"""
