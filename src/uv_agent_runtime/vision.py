from __future__ import annotations

from pathlib import Path
from typing import Any

from .events import emit_event
from .files import resolve_workspace_path


def look_at(path: str | Path, *, note: str = "") -> dict[str, Any]:
    """Attach an image to the uv-agent conversation context.

    The runner records this structured event and the host agent appends the image
    to future model input. Use it when visual inspection is needed; then ask the
    model to reason about the image in the next response.
    """
    resolved = resolve_workspace_path(path)
    return emit_event("look_at", path=str(resolved), note=note)
