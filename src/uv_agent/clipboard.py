from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import ImageGrab

from uv_agent.ids import new_id


class ClipboardImageError(RuntimeError):
    """Raised when the clipboard does not contain a supported bitmap image."""


@dataclass(frozen=True)
class ClipboardImage:
    path: Path
    width: int
    height: int


def save_clipboard_image(target_dir: Path) -> ClipboardImage:
    """Save the current clipboard image as a PNG under target_dir."""
    grabbed = ImageGrab.grabclipboard()
    if grabbed is None:
        raise ClipboardImageError("Clipboard does not contain an image")
    if isinstance(grabbed, list):
        raise ClipboardImageError("Clipboard contains files, not image data")
    if not hasattr(grabbed, "save") or not hasattr(grabbed, "size"):
        raise ClipboardImageError("Clipboard image format is not supported")

    image: Any = grabbed
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{new_id('clip')}.png"
    if getattr(image, "mode", "") not in {"RGB", "RGBA"}:
        image = image.convert("RGBA")
    image.save(path, format="PNG")
    width, height = image.size
    return ClipboardImage(path=path, width=int(width), height=int(height))
