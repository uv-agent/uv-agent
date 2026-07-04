from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uv_agent.blobs import BlobRecord, BlobStore
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
    """A persisted image attachment backed by the project blob store."""

    attachment_id: str
    blob: BlobRecord
    mime_type: str
    filename: str
    source_path: Path | None = None
    source_uri: str = ""
    note: str = ""

    def to_event_payload(self) -> dict[str, Any]:
        payload = {
            "attachment_id": self.attachment_id,
            "blob_id": self.blob.blob_id,
            "blob_path": str(self.blob.path),
            "mime_type": self.mime_type,
            "sha256": self.blob.sha256,
            "size_bytes": self.blob.size_bytes,
            "filename": self.filename,
            "source_uri": self.source_uri,
            "note": self.note,
        }
        if self.source_path is not None:
            payload["source_path"] = str(self.source_path)
        return payload


class AttachmentStore:
    """Store image attachments as references to project blobs."""

    def __init__(self, blobs: BlobStore) -> None:
        self.blobs = blobs

    def register_image(
        self,
        path: str | Path,
        *,
        cwd: Path,
        thread_id: str,
        note: str = "",
        mime_type: str | None = None,
        owner_id: str | None = None,
    ) -> ImageAttachment:
        source = Path(path)
        if not source.is_absolute():
            source = cwd / source
        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError(f"Image does not exist: {source}")
        if not source.is_file():
            raise ValueError(f"Image path is not a file: {source}")
        resolved_mime = mime_type or mimetypes.guess_type(str(source))[0] or "application/octet-stream"
        return self.register_image_bytes(
            source.read_bytes(),
            thread_id=thread_id,
            note=note,
            mime_type=resolved_mime,
            filename=source.name,
            source_path=source,
            owner_id=owner_id,
        )

    def register_image_bytes(
        self,
        data: bytes,
        *,
        thread_id: str,
        mime_type: str,
        note: str = "",
        filename: str = "",
        source_path: Path | None = None,
        source_uri: str = "",
        owner_id: str | None = None,
    ) -> ImageAttachment:
        resolved_mime = str(mime_type or "application/octet-stream")
        if resolved_mime not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError(f"Unsupported image type: {resolved_mime}")
        blob = self.blobs.put_bytes(data)
        attachment_id = owner_id or new_id("img")
        attachment = ImageAttachment(
            attachment_id=attachment_id,
            blob=blob,
            mime_type=resolved_mime,
            filename=filename or f"{attachment_id}{_suffix_for_mime(resolved_mime)}",
            source_path=source_path,
            source_uri=source_uri,
            note=note,
        )
        self.blobs.add_ref(
            blob.blob_id,
            thread_id=thread_id,
            owner_type="image_attachment",
            owner_id=attachment.attachment_id,
            mime_type=resolved_mime,
            filename=attachment.filename,
            source_uri=source_uri,
            note=note,
        )
        return attachment

    def register_image_blob(
        self,
        blob_id: str,
        *,
        thread_id: str,
        mime_type: str,
        note: str = "",
        filename: str = "",
        source_uri: str = "",
        owner_id: str | None = None,
    ) -> ImageAttachment:
        resolved_mime = str(mime_type or "application/octet-stream")
        if resolved_mime not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError(f"Unsupported image type: {resolved_mime}")
        info = self.blobs.info(blob_id)
        attachment_id = owner_id or new_id("img")
        blob = BlobRecord(
            blob_id=str(info["blob_id"]),
            sha256=str(info["sha256"]),
            size_bytes=int(info["size_bytes"]),
            path=Path(str(info["stored_path"])),
            created_at=str(info["created_at"]),
        )
        attachment = ImageAttachment(
            attachment_id=attachment_id,
            blob=blob,
            mime_type=resolved_mime,
            filename=filename or f"{attachment_id}{_suffix_for_mime(resolved_mime)}",
            source_uri=source_uri,
            note=note,
        )
        self.blobs.add_ref(
            blob.blob_id,
            thread_id=thread_id,
            owner_type="image_attachment",
            owner_id=attachment.attachment_id,
            mime_type=resolved_mime,
            filename=attachment.filename,
            source_uri=source_uri,
            note=note,
        )
        return attachment


def image_message_item(attachment: dict[str, Any]) -> dict[str, Any]:
    """Build a Responses-style user item containing an image attachment."""

    note = str(attachment.get("note") or "").strip()
    path = Path(str(attachment["blob_path"]))
    mime_type = str(attachment["mime_type"])
    data_url = image_data_url(path, mime_type)
    text = IMAGE_ATTACHMENT_TEXT_TEMPLATE.format(
        attachment_id=attachment.get("attachment_id"),
        filename=attachment.get("filename") or path.name,
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
