from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from uv_agent.ids import new_id

if TYPE_CHECKING:
    from textual.worker import Worker

    from uv_agent.tui.widgets import TranscriptCell


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
    retryable_error: bool = False
    terminal_error: bool = False
    live_events: list[dict[str, Any]] = field(default_factory=list)
    pending_stream_retries: list[dict[str, Any]] = field(default_factory=list)
    assistant_buffer: str = ""
    assistant_cell: TranscriptCell | None = None
    reasoning_buffer: str = ""
    reasoning_cell: TranscriptCell | None = None
    tool_cells: dict[str, TranscriptCell] = field(default_factory=dict)
    tool_delta_cells: dict[int, TranscriptCell] = field(default_factory=dict)
    tool_delta_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    process_cells: list[TranscriptCell] = field(default_factory=list)
    process_fold_cell: TranscriptCell | None = None
    process_collapsed: bool = False
    process_anchor_cell: TranscriptCell | None = None

    def detach_widgets(self) -> None:
        self.assistant_cell = None
        self.reasoning_cell = None
        self.tool_cells.clear()
        self.tool_delta_cells.clear()
        self.process_cells = []
        self.process_fold_cell = None
        self.process_anchor_cell = None
