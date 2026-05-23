from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uv_agent.jsonl import JsonlWriter
from uv_agent.time import utc_now_iso


@dataclass(frozen=True)
class RunContext:
    """Immutable context passed to host methods called by a runtime script."""

    run_id: str
    thread_id: str | None
    turn_id: str | None
    cwd: Path


class RunSession:
    """Per-run state addressed by a short-lived bearer token."""

    def __init__(
        self,
        *,
        token: str,
        run_id: str,
        thread_id: str | None,
        turn_id: str | None,
        cwd: Path,
        structured_events: list[dict[str, Any]],
        writer: JsonlWriter,
    ) -> None:
        self.token = token
        self.context = RunContext(
            run_id=run_id,
            thread_id=thread_id,
            turn_id=turn_id,
            cwd=cwd,
        )
        self._structured_events = structured_events
        self._writer = writer
        self._lock = threading.RLock()
        self.closed = False

    @property
    def run_id(self) -> str:
        return self.context.run_id

    def close(self) -> None:
        with self._lock:
            self.closed = True

    def emit_event(self, event: dict[str, Any]) -> None:
        """Append a structured runtime event and persist it in the run log."""

        event_copy = dict(event)
        with self._lock:
            if self.closed:
                raise RuntimeError("Run session is closed")
            self._structured_events.append(event_copy)
            self._writer.write(
                {
                    "type": "run.event",
                    "created_at": utc_now_iso(),
                    "run_id": self.run_id,
                    "event": event_copy,
                }
            )
