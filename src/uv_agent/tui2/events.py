from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from uv_agent.environment import UserLanguage, normalize_language
from uv_agent.tui.formatting import parse_tool_payload, short_thread


def _default_language() -> UserLanguage:
    return normalize_language("en")

CellKind = Literal["user", "assistant", "reasoning", "tool", "event", "error", "image"]
Tui2Mode = Literal["transcript", "agent_view"]
AgentViewInteractionMode = Literal["normal", "input", "help", "model"]
AgentViewInputTarget = Literal["dispatch", "reply"]
AgentViewRowStatus = Literal[
    "dispatching",
    "working",
    "queued",
    "completed",
    "failed",
    "interrupted",
]
AGENT_VIEW_STATUS_ORDER: tuple[AgentViewRowStatus, ...] = (
    "dispatching",
    "working",
    "queued",
    "failed",
    "interrupted",
    "completed",
)


@dataclass(frozen=True)
class CommandSuggestion:
    """One command-palette or picker completion row.

    ``value`` is the text shown in the palette.  ``id``/``kind`` are optional
    action metadata used by tui2's lightweight thread, skill, MCP, and mention
    pickers; keeping them here avoids a second nearly-identical row model.
    """

    value: str
    description: str = ""
    id: str = ""
    kind: str = "command"
    meta: str = ""


@dataclass
class TranscriptCell:
    """A transcript unit that can be flushed into terminal scrollback."""

    kind: CellKind
    text: str = ""
    title: str = ""
    status: str = "done"
    call: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    created_at: float = field(default_factory=monotonic)
    finished_at: float | None = None
    # Cumulative characters streamed for breath animation phasing.
    chars_streamed: int = 0
    # Fractional animation phase driven by a per-turn throughput estimator.  The
    # legacy integer ``chars_streamed`` is still retained for older callers and
    # finished scrollback metadata.
    animation_phase: float | None = None

    @property
    def done(self) -> bool:
        return self.status not in {"running", "streaming"}

    @property
    def elapsed_s(self) -> float:
        return max(0.0, (self.finished_at or monotonic()) - self.created_at)


@dataclass
class PendingTurn:
    """A user message waiting behind the active tui2 turn."""

    text: str
    image_paths: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class AgentViewRow:
    """One session row in the Agent View dashboard."""

    thread_id: str
    title: str
    status: AgentViewRowStatus
    summary: str = ""
    updated_at: str = ""
    worktree_branch: str = ""
    worktree_path: str = ""
    elapsed_seconds: float = 0.0
    queued_turns: int = 0


@dataclass
class AgentViewState:
    """UI-only state for the Agent View full-screen mode."""

    rows: list[AgentViewRow] = field(default_factory=list)
    selected: int = 0
    peek_expanded: bool = True
    dispatch_level: str | None = None
    dispatch_level_explicit: bool = False
    model_options: list[CommandSuggestion] = field(default_factory=list)
    model_selected: int = 0
    composer: str = ""
    composer_cursor: int | None = None
    status_message: str = "Enter dispatches a background worktree task"
    pending_confirmation: str | None = None
    interaction_mode: AgentViewInteractionMode = "normal"
    input_target: AgentViewInputTarget = "dispatch"
    input_target_thread_id: str | None = None

    def selected_row(self) -> AgentViewRow | None:
        if not self.rows:
            return None
        self.selected = max(0, min(self.selected, len(self.rows) - 1))
        return self.rows[self.selected]


@dataclass
class Tui2State:
    """Mutable render state shared by the app and renderer."""

    thread_id: str | None = None
    level: str | None = None
    title: str = "New thread"
    composer: str = ""
    composer_cursor: int | None = None
    pending_turns: list[PendingTurn] = field(default_factory=list)
    flushed: list[TranscriptCell] = field(default_factory=list)
    live: list[TranscriptCell] = field(default_factory=list)
    last_error: str | None = None
    busy: bool = False
    status_message: str = "ready"
    project_path: str = ""
    context_percent: int | None = None
    goal_enabled: bool = False
    goal_objective: str = ""
    image_token_numbers: set[int] = field(default_factory=set)
    turn_elapsed_s: float | None = None
    turn_token_rate: float | None = None
    command_palette_open: bool = False
    command_palette_items: list[CommandSuggestion] = field(default_factory=list)
    command_palette_index: int = 0
    language: UserLanguage = field(default_factory=_default_language)
    mode: Tui2Mode = "transcript"
    agent_view: AgentViewState = field(default_factory=AgentViewState)

    def status_label(self) -> str:
        parts = [f"thread {short_thread(self.thread_id)}"]
        if self.level:
            parts.append(self.level)
        if self.pending_turns:
            parts.append(f"{len(self.pending_turns)} queued")
        if self.last_error:
            parts.append("error")
        parts.append(self.status_message)
        return " · ".join(parts)


def event_user_text(event: dict[str, Any]) -> str:
    item = event.get("item") or {}
    content = item.get("content") or []
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in {"input_text", "text"}:
            parts.append(str(part.get("text") or ""))
    return "\n".join(part for part in parts if part)


def tool_payload_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    output = event.get("output")
    if isinstance(output, dict):
        parsed = parse_tool_payload(output)
        if parsed is not None:
            return parsed
        raw = output.get("output")
        if isinstance(raw, str):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def tool_title(call: dict[str, Any] | None) -> str:
    if not call:
        return "python"
    return str(call.get("name") or "python")
