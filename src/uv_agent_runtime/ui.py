from __future__ import annotations

from .events import emit_event

UI_MESSAGE_KIND = "ui.message"
UI_MESSAGE_FORMAT = "markdown"


def message(markdown: str) -> dict[str, object]:
    """Emit a Markdown message intended for user-facing UI surfaces."""

    return emit_event(UI_MESSAGE_KIND, message=str(markdown), format=UI_MESSAGE_FORMAT)
