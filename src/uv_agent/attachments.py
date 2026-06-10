from __future__ import annotations

import base64
import hashlib
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uv_agent.ids import new_id
from uv_agent.prompts import IMAGE_ATTACHMENT_NOTE_TEMPLATE, IMAGE_ATTACHMENT_TEXT_TEMPLATE


SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}


@dataclass(frozen=True)
class ImageAttachment:
    """A persisted image attachment that can be replayed into model context."""

    attachment_id: str
    source_path: Path
    stored_path: Path
    mime_type: str
    sha256: str
    size_bytes: int
    note: str = ""

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "attachment_id": self.attachment_id,
            "source_path": str(self.source_path),
            "stored_path": str(self.stored_path),
            "mime_type": self.mime_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "note": self.note,
        }


class AttachmentStore:
    """Store binary context attachments outside SQLite history."""

    def __init__(self, attachments_dir: Path) -> None:
        self.attachments_dir = attachments_dir
        self.attachments_dir.mkdir(parents=True, exist_ok=True)

    def register_image(
        self,
        path: str | Path,
        *,
        cwd: Path,
        thread_id: str,
        note: str = "",
    ) -> ImageAttachment:
        source = Path(path)
        if not source.is_absolute():
            source = cwd / source
        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError(f"Image does not exist: {source}")
        if not source.is_file():
            raise ValueError(f"Image path is not a file: {source}")

        mime_type = mimetypes.guess_type(str(source))[0] or "application/octet-stream"
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError(f"Unsupported image type: {mime_type}")

        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        suffix = source.suffix.lower() or _suffix_for_mime(mime_type)
        attachment_id = new_id("img")
        target_dir = self.attachments_dir / thread_id
        target_dir.mkdir(parents=True, exist_ok=True)
        stored_path = target_dir / f"{attachment_id}-{digest[:12]}{suffix}"
        if not stored_path.exists():
            shutil.copyfile(source, stored_path)

        return ImageAttachment(
            attachment_id=attachment_id,
            source_path=source,
            stored_path=stored_path,
            mime_type=mime_type,
            sha256=digest,
            size_bytes=len(content),
            note=note,
        )


def image_message_item(attachment: dict[str, Any]) -> dict[str, Any]:
    """Build a Responses-style user item containing an image attachment."""
    note = str(attachment.get("note") or "").strip()
    path = Path(str(attachment["stored_path"]))
    mime_type = str(attachment["mime_type"])
    data_url = image_data_url(path, mime_type)
    text = IMAGE_ATTACHMENT_TEXT_TEMPLATE.format(
        attachment_id=attachment.get("attachment_id"),
        filename=path.name,
    )
    if note:
        text += "\n" + IMAGE_ATTACHMENT_NOTE_TEMPLATE.format(note=note)
    return {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": data_url},
        ],
    }


def image_data_url(path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _suffix_for_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/png":
        return ".png"
    if mime_type == "image/gif":
        return ".gif"
    if mime_type == "image/webp":
        return ".webp"
    return ".bin"
