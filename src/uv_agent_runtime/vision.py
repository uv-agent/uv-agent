from __future__ import annotations

from pathlib import Path
from typing import Any

from .events import emit_event
from .files import resolve_workspace_path
from .resources import Resource
from .transport import upload_blob


def look_at(target: str | Path | bytes | Resource, *, note: str = "", mime_type: str | None = None, filename: str | None = None) -> dict[str, Any]:
    """Attach an image to the uv-agent conversation context.

    The runner records this structured event and the host agent appends the image
    to future model input. Use it when visual inspection is needed; then ask the
    model to reason about the image in the next response.
    """
    if isinstance(target, Resource):
        if target.kind == "text":
            raise ValueError("Text resources cannot be attached with look_at")
        resolved_mime = mime_type or target.mime_type
        if target.kind == "path":
            return emit_event("look_at", path=str(target.path()), mime_type=resolved_mime, filename=filename or target.metadata.get("filename") or "", note=note)
        blob_id = target.metadata.get("blob_id")
        if blob_id:
            return emit_event("look_at", blob_id=str(blob_id), mime_type=resolved_mime, filename=filename or str(target.metadata.get("filename") or ""), source_uri=target.uri, note=note)
        blob = upload_blob(target.bytes(), mime_type=resolved_mime, filename=filename or str(target.metadata.get("filename") or "resource.bin"))
        return emit_event("look_at", blob_id=blob["blob_id"], mime_type=resolved_mime, filename=blob.get("filename") or filename or "", source_uri=target.uri, note=note)
    if isinstance(target, bytes):
        if not mime_type:
            raise ValueError("mime_type is required when attaching image bytes")
        blob = upload_blob(target, mime_type=mime_type, filename=filename or "image")
        return emit_event("look_at", blob_id=blob["blob_id"], mime_type=mime_type, filename=blob.get("filename") or filename or "", note=note)
    resolved = resolve_workspace_path(target)
    return emit_event("look_at", path=str(resolved), mime_type=mime_type or "", filename=filename or "", note=note)
