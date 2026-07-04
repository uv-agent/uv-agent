from __future__ import annotations

import hashlib
import mimetypes
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uv_agent.state_db import connect_state_db
from uv_agent.time import utc_now_iso

BLOB_ID_PREFIX = "blob:sha256:"
MAX_BLOB_BYTES = 500 * 1024 * 1024


@dataclass(frozen=True)
class BlobRecord:
    blob_id: str
    sha256: str
    size_bytes: int
    path: Path
    created_at: str

    def to_ref(
        self,
        *,
        mime_type: str = "application/octet-stream",
        filename: str = "",
    ) -> dict[str, Any]:
        return {
            "blob_id": self.blob_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "path": str(self.path),
            "mime_type": mime_type or "application/octet-stream",
            "filename": filename or "",
        }


class BlobStore:
    """Content-addressed project blob storage backed by the state database."""

    def __init__(self, data_dir: Path, *, blobs_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.blobs_dir = (blobs_dir or self.data_dir / "blobs").resolve()
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        with self._connect():
            pass

    def put_bytes(self, data: bytes, *, max_bytes: int = MAX_BLOB_BYTES) -> BlobRecord:
        if not isinstance(data, bytes):
            raise TypeError("blob data must be bytes")
        if len(data) > max_bytes:
            raise ValueError(f"Blob is {len(data)} bytes, above max_bytes={max_bytes}")
        digest = hashlib.sha256(data).hexdigest()
        blob_id = f"{BLOB_ID_PREFIX}{digest}"
        path = self._path_for_sha256(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return self._upsert_record(blob_id=blob_id, sha256=digest, size_bytes=len(data), path=path)

    def put_path(self, path: str | Path, *, max_bytes: int = MAX_BLOB_BYTES) -> BlobRecord:
        source = Path(path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Blob source does not exist: {source}")
        if not source.is_file():
            raise ValueError(f"Blob source is not a file: {source}")
        size = source.stat().st_size
        if size > max_bytes:
            raise ValueError(f"Blob source is {size} bytes, above max_bytes={max_bytes}: {source}")
        return self.put_bytes(source.read_bytes(), max_bytes=max_bytes)

    def info(self, blob_id: str) -> dict[str, Any]:
        row = self._record_row(blob_id)
        if row is None:
            raise FileNotFoundError(f"Unknown blob: {blob_id}")
        return dict(row)

    def path(self, blob_id: str) -> Path:
        row = self._record_row(blob_id)
        if row is None:
            raise FileNotFoundError(f"Unknown blob: {blob_id}")
        path = Path(str(row["stored_path"]))
        if not path.exists():
            raise FileNotFoundError(f"Blob file is missing for {blob_id}: {path}")
        return path

    def read_bytes(self, blob_id: str) -> bytes:
        return self.path(blob_id).read_bytes()

    def add_ref(
        self,
        blob_id: str,
        *,
        thread_id: str,
        owner_type: str,
        owner_id: str,
        mime_type: str = "application/octet-stream",
        filename: str = "",
        source_uri: str = "",
        note: str = "",
    ) -> None:
        self.info(blob_id)
        with self._connect() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO blob_refs(
                  blob_id, thread_id, owner_type, owner_id, mime_type,
                  filename, source_uri, note, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    blob_id,
                    str(thread_id or ""),
                    str(owner_type or "ref"),
                    str(owner_id or blob_id),
                    str(mime_type or "application/octet-stream"),
                    str(filename or ""),
                    str(source_uri or ""),
                    str(note or ""),
                    utc_now_iso(),
                ),
            )

    def remove_thread_refs(self, thread_id: str) -> int:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM blob_refs WHERE thread_id = ?", (thread_id,))
            return int(cursor.rowcount or 0)

    def gc_unreferenced(self, *, blob_ids: list[str] | None = None) -> dict[str, int]:
        params: list[Any] = []
        filter_sql = ""
        if blob_ids:
            filter_sql = f"AND blob_id IN ({','.join('?' for _ in blob_ids)})"
            params.extend(blob_ids)
        with self._connect() as db:
            rows = db.execute(
                f"""
                SELECT blob_id, stored_path, size_bytes
                FROM blobs
                WHERE NOT EXISTS (
                  SELECT 1 FROM blob_refs WHERE blob_refs.blob_id = blobs.blob_id
                )
                {filter_sql}
                """,
                params,
            ).fetchall()
            deleted = 0
            freed = 0
            for row in rows:
                path = Path(str(row["stored_path"]))
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    continue
                db.execute("DELETE FROM blobs WHERE blob_id = ?", (row["blob_id"],))
                deleted += 1
                freed += int(row["size_bytes"] or 0)
        return {"deleted_blobs": deleted, "freed_bytes": freed}

    def _upsert_record(self, *, blob_id: str, sha256: str, size_bytes: int, path: Path) -> BlobRecord:
        created_at = utc_now_iso()
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO blobs(blob_id, sha256, size_bytes, stored_path, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(blob_id) DO NOTHING
                """,
                (blob_id, sha256, size_bytes, str(path), created_at),
            )
            row = self._record_row(blob_id, db=db)
        if row is None:
            raise RuntimeError(f"Failed to store blob {blob_id}")
        return BlobRecord(
            blob_id=str(row["blob_id"]),
            sha256=str(row["sha256"]),
            size_bytes=int(row["size_bytes"]),
            path=Path(str(row["stored_path"])),
            created_at=str(row["created_at"]),
        )

    def _record_row(self, blob_id: str, *, db: sqlite3.Connection | None = None) -> sqlite3.Row | None:
        if not str(blob_id).startswith(BLOB_ID_PREFIX):
            raise ValueError(f"Invalid blob id: {blob_id!r}")
        if db is not None:
            return db.execute("SELECT * FROM blobs WHERE blob_id = ?", (blob_id,)).fetchone()
        with self._connect() as connection:
            return connection.execute("SELECT * FROM blobs WHERE blob_id = ?", (blob_id,)).fetchone()

    def _path_for_sha256(self, digest: str) -> Path:
        return self.blobs_dir / "sha256" / digest[:2] / digest

    def _connect(self) -> sqlite3.Connection:
        return connect_state_db(self.data_dir)


def guess_mime_type(name: str | Path, *, default: str = "application/octet-stream") -> str:
    return mimetypes.guess_type(str(name))[0] or default
