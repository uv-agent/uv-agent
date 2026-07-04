from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from . import textops
from .transport import blob_info, download_blob, upload_blob, call_host

ResourceKind = Literal["text", "bytes", "path"]
_URI_RE = re.compile(r"^[a-z][a-z0-9+.-]*://")


@dataclass
class Resource:
    uri: str
    kind: ResourceKind
    mime_type: str = "application/octet-stream"
    metadata: dict[str, Any] = field(default_factory=dict)
    _text: str | None = None
    _data: bytes | None = None
    _path: Path | None = None
    _blob_id: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Resource":
        metadata = dict(payload.get("metadata") or {})
        blob = payload.get("blob")
        blob_id = None
        if isinstance(blob, dict):
            blob_id = str(blob.get("blob_id") or "") or None
            metadata.setdefault("blob_id", blob_id)
            for key in ("size_bytes", "filename", "sha256", "path"):
                if key in blob:
                    metadata.setdefault(key, blob[key])
        path = payload.get("path")
        return cls(
            uri=str(payload["uri"]),
            kind=str(payload["kind"]),  # type: ignore[arg-type]
            mime_type=str(payload.get("mime_type") or "application/octet-stream"),
            metadata=metadata,
            _text=payload.get("text") if isinstance(payload.get("text"), str) else None,
            _path=Path(path) if isinstance(path, str) and path else None,
            _blob_id=blob_id,
        )

    @property
    def data(self) -> bytes | None:
        if self.kind != "bytes":
            return None
        return self.bytes()

    def text(self, *, encoding: str = "utf-8") -> str:
        if self.kind == "text":
            return self._text or ""
        return self.bytes().decode(encoding, errors="replace")

    def bytes(self) -> bytes:
        if self.kind == "text":
            raise ValueError(f"Resource {self.uri} is text-only; use .text()")
        if self._data is not None:
            return self._data
        if self._path is not None:
            self._data = self._path.read_bytes()
            return self._data
        if self._blob_id:
            self._data = download_blob(self._blob_id)
            return self._data
        return b""

    def path(self) -> Path:
        if self.kind == "path" and self._path is not None:
            return self._path
        if self.kind == "bytes":
            if self._path is not None:
                return self._path
            if self._blob_id:
                info = blob_info(self._blob_id)
                path = Path(str(info["path"]))
                self._path = path
                return path
            blob = upload_blob(self.bytes(), mime_type=self.mime_type, filename=str(self.metadata.get("filename") or "resource.bin"))
            self._blob_id = str(blob["blob_id"])
            self.metadata.setdefault("blob_id", self._blob_id)
            self._path = Path(str(blob["path"]))
            return self._path
        raise ValueError(f"Resource {self.uri} is text-only; use .text()")

    def read(
        self,
        *,
        lines: tuple[int, int] | None = None,
        head: int | None = None,
        tail: int | None = None,
        around: str | None = None,
        context: int = 20,
        encoding: str = "utf-8",
    ) -> textops.FileView:
        if self.kind == "path":
            return textops.read_file(self.path(), lines=lines, head=head, tail=tail, around=around, context=context, encoding=encoding)
        text = self.text(encoding=encoding)
        selected, start, end, truncated = _select_text(text, lines=lines, head=head, tail=tail, around=around, context=context)
        return textops.FileView(
            path=self.uri,
            exists=True,
            text=selected,
            line_count=len(text.splitlines()),
            start_line=start,
            end_line=end,
            truncated=truncated,
            encoding=encoding,
            newline="lf",
            final_newline=text.endswith("\n"),
            bom=False,
            size=len(text.encode(encoding, errors="replace")),
            kind="file",
        )


def get(target: str | Path, *, max_bytes: int | None = None):
    from .facade import File

    if is_resource_uri(target):
        payload = call_host("resource.get", str(target), max_bytes=max_bytes)
        if not isinstance(payload, dict):
            raise RuntimeError("resource.get returned an invalid payload")
        return Resource.from_payload(payload)
    return File(target)


def is_resource_uri(value: object) -> bool:
    return isinstance(value, str) and bool(_URI_RE.match(value))


class _BlobNamespace:
    def info(self, blob_id: str) -> dict[str, Any]:
        return blob_info(blob_id)

    def path(self, blob_id: str) -> Path:
        info = blob_info(blob_id)
        return Path(str(info["path"]))


blob = _BlobNamespace()


def _select_text(
    text: str,
    *,
    lines: tuple[int, int] | None,
    head: int | None,
    tail: int | None,
    around: str | None,
    context: int,
) -> tuple[str, int, int, bool]:
    all_lines = text.splitlines()
    count = len(all_lines)
    if lines is not None:
        start, end = max(1, int(lines[0])), min(count, int(lines[1]))
    elif head is not None:
        start, end = 1, min(count, max(0, int(head)))
    elif tail is not None:
        amount = max(0, int(tail))
        start, end = max(1, count - amount + 1), count
    elif around:
        index = next((idx for idx, line in enumerate(all_lines, start=1) if around in line), 1)
        start, end = max(1, index - context), min(count, index + context)
    else:
        start, end = (1, count)
    selected = "\n".join(all_lines[start - 1 : end]) if count else ""
    truncated = start > 1 or end < count
    return selected, start, end, truncated
