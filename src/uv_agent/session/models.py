from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SessionEvent:
    type: str
    data: dict[str, Any]


@dataclass
class TurnState:
    turn_id: str
    status: Literal["running", "completed", "failed", "interrupted"] = "running"
    items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ThreadState:
    thread_id: str
    title: str = "New thread"
    turns: list[TurnState] = field(default_factory=list)
    model_input: list[dict[str, Any]] = field(default_factory=list)
