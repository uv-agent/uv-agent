from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(event, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_jsonl_after_latest_compaction(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return events after the latest compaction without retaining older parsed events."""
    if not path.exists():
        return [], None
    events_after: list[dict[str, Any]] = []
    latest_compaction: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") == "item.compaction":
                latest_compaction = event
                events_after = []
            else:
                events_after.append(event)
    return events_after, latest_compaction


def append_jsonl(path: Path, events: Iterable[dict[str, Any]]) -> None:
    writer = JsonlWriter(path)
    for event in events:
        writer.write(event)
