from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from uv_agent.ids import new_id

from textual.worker import Worker


@dataclass(frozen=True)
class PickerItem:
    id: str
    title: str
    description: str = ""
    meta: str = ""


@dataclass
class PanelPage:
    title: str
    body: object = ""
    items: list[PickerItem] | None = None
    subtitle: str = ""
    filter_value: str = ""
    highlighted: int | None = None
    mention_kind: str | None = None
    mention_items: Callable[[str], tuple[str, list[PickerItem], str]] | None = None
    select_callback: Callable[[str], None] | None = None
    close_on_select: bool = False


@dataclass
class MentionScanCache:
    items: list[PickerItem] = field(default_factory=list)
    complete: bool = False
    generation: int = 0
    worker: Worker[None] | None = None


@dataclass(frozen=True)
class CommandSpec:
    name: str
    usage: str
    description: str

    @property
    def palette_title(self) -> str:
        return self.name.removeprefix("/")


@dataclass(frozen=True)
class PendingImage:
    path: Path
    width: int
    height: int

    def to_attachment(self) -> dict[str, Any]:
        size = self.path.stat().st_size if self.path.exists() else 0
        return {
            "stored_path": str(self.path),
            "source_path": str(self.path),
            "mime_type": "image/png",
            "size_bytes": size,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class TopNotification:
    """In-memory notification-center entry for non-transcript UI events."""

    id: str
    title: str
    message: str
    created_at: str
    thread_id: str | None = None
    severity: str = "information"
    read: bool = False


@dataclass
class ThreadActivityState:
    """Session-local activity accounting for the top status bar.

    This deliberately tracks only the current TUI process lifetime. Persisted
    thread history remains the source of truth for transcript content, while the
    top bar answers "what has happened since I opened the app?".
    """

    thread_id: str
    total_elapsed_s: float = 0.0
    active_started_monotonic: float | None = None
    completed: bool = False

    @property
    def active(self) -> bool:
        return self.active_started_monotonic is not None


@dataclass(frozen=True)
class QueuedTurn:
    """A user send that is waiting behind the active turn.

    ``queue_id`` is generated when the item is enqueued so list-panel edits can
    safely target the same item even if other queued sends are reordered or
    deleted. It is intentionally UI-local and is not persisted.
    """

    prompt: str
    level: str | None = None
    image_paths: list[Path] = field(default_factory=list)
    queue_id: str = field(default_factory=lambda: new_id("queue"))


@dataclass
class ThreadRunState:
    thread_id: str
    worker: Worker[None] | None
    cancel_event: asyncio.Event
    queue: list[QueuedTurn]
    status: str
    turn_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    pending_user_turn_id: str | None = None
    retryable_error: bool = False
    terminal_error: bool = False
