from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> dict[str, Any]:
        with self.path.open("ab") as handle:
            offset = handle.tell()
            stored = {**event, "_jsonl_offset": offset}
            line = json.dumps(stored, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            handle.write(line + b"\n")
        return stored


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def read_jsonl_from_offset(path: Path, offset: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        for line in handle:
            if line.strip():
                events.append(json.loads(line.decode("utf-8")))
    return events


def read_jsonl_after_latest_compaction(
    path: Path,
    compaction_offset: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return the latest compaction event and the events after it."""
    if compaction_offset is None:
        return read_jsonl(path), None
    events = read_jsonl_from_offset(path, compaction_offset)
    if not events:
        return [], None
    latest_compaction = events[0]
    events_after = events[1:]
    return events_after, latest_compaction


def iter_jsonl_reverse(
    path: Path,
    *,
    before_offset: int | None = None,
    chunk_size: int = 64 * 1024,
) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("rb") as handle:
        end = handle.seek(0, os.SEEK_END)
        position = min(end, before_offset) if before_offset is not None else end
        buffer = b""
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            buffer = chunk + buffer
            lines = buffer.split(b"\n")
            buffer = lines[0]
            for line in reversed(lines[1:]):
                if line.strip():
                    yield json.loads(line.decode("utf-8"))
        if buffer.strip():
            yield json.loads(buffer.decode("utf-8"))


def read_jsonl_tail(
    path: Path,
    *,
    limit: int,
    before_offset: int | None = None,
    event_types: set[str] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    selected: list[dict[str, Any]] = []
    has_more = False
    if limit <= 0:
        return [], False
    for event in iter_jsonl_reverse(path, before_offset=before_offset):
        if event_types is not None and event.get("type") not in event_types:
            continue
        if len(selected) >= limit:
            has_more = True
            break
        selected.append(event)
    selected.reverse()
    return selected, has_more


def append_jsonl(path: Path, events: Iterable[dict[str, Any]]) -> None:
    writer = JsonlWriter(path)
    for event in events:
        writer.write(event)
