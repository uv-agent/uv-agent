from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape as xml_escape
from pathlib import Path
from typing import Any, Literal

from uv_agent.time import utc_now_iso

from uv_agent.prompts import (
    GOAL_MODE_ACTIVE,
    GOAL_MODE_ACTIVE_RULES,
    GOAL_MODE_CHECKLIST_FILE_TEMPLATE,
    GOAL_MODE_CHECKLIST_TEMPLATE,
    GOAL_MODE_CLOSE,
    GOAL_MODE_DISABLED,
    GOAL_MODE_DISABLED_OPEN,
    GOAL_MODE_DISABLED_RULES,
    GOAL_MODE_ENABLED_OPEN,
    GOAL_MODE_FIELD_CHECKLIST,
    GOAL_MODE_FIELD_DOCUMENT,
    GOAL_MODE_FIELD_OBJECTIVE,
    GOAL_MODE_FIELD_STATE,
    GOAL_MODE_FILES_CLOSE,
    GOAL_MODE_FILES_OPEN,
    GOAL_MODE_NOTES_FILE_TEMPLATE,
    GOAL_MODE_NOTES_HINT,
    GOAL_MODE_RULES_CLOSE,
    GOAL_MODE_RULES_OPEN,
    XML_ELEMENT_TEMPLATE,
)

GoalModeStatus = Literal["enabled", "disabled"]

GOAL_STATE_FILENAME = "goal.json"
GOAL_CHECKLIST_FILENAME = "checklist.md"
GOAL_NOTES_FILENAME = "notes.md"


def _xml_text(value: object) -> str:
    return xml_escape(str(value), quote=False)


def _xml_element(tag: str, value: object) -> str:
    return XML_ELEMENT_TEMPLATE.format(tag=tag, value=_xml_text(value))


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
                GOAL_MODE_DISABLED_OPEN,
                GOAL_MODE_DISABLED,
                "",
                GOAL_MODE_FILES_OPEN,
                _xml_element(GOAL_MODE_FIELD_STATE, state.paths.state),
                _xml_element(GOAL_MODE_FIELD_CHECKLIST, state.paths.checklist),
                _xml_element(GOAL_MODE_FIELD_DOCUMENT, state.paths.notes),
                GOAL_MODE_FILES_CLOSE,
                "",
                GOAL_MODE_DISABLED_RULES,
                GOAL_MODE_CLOSE,
            ]
        )

    lines = [
        GOAL_MODE_ENABLED_OPEN,
        GOAL_MODE_ACTIVE,
    ]
    if state.objective.strip():
        lines.extend(["", _xml_element(GOAL_MODE_FIELD_OBJECTIVE, state.objective.strip())])
    lines.extend(
        [
            "",
            GOAL_MODE_FILES_OPEN,
            _xml_element(GOAL_MODE_FIELD_STATE, state.paths.state),
            _xml_element(GOAL_MODE_FIELD_CHECKLIST, state.paths.checklist),
            _xml_element(GOAL_MODE_FIELD_DOCUMENT, state.paths.notes),
            GOAL_MODE_FILES_CLOSE,
            "",
            GOAL_MODE_RULES_OPEN,
            GOAL_MODE_ACTIVE_RULES,
            GOAL_MODE_RULES_CLOSE,
            GOAL_MODE_CLOSE,
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
    objective_line = objective.strip() or GOAL_MODE_CHECKLIST_TEMPLATE
    return GOAL_MODE_CHECKLIST_FILE_TEMPLATE.format(objective=objective_line)

def _notes_template(objective: str) -> str:
    objective_line = objective.strip() or GOAL_MODE_CHECKLIST_TEMPLATE
    return GOAL_MODE_NOTES_FILE_TEMPLATE.format(
        objective=objective_line,
        handoff_hint=GOAL_MODE_NOTES_HINT,
    )
