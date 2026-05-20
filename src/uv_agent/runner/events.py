from __future__ import annotations

import json
from dataclasses import dataclass

RUNTIME_EVENT_EVENT_ID_KEY = "_uv_agent_event_id"
RUNTIME_EVENT_RUN_ID_KEY = "_uv_agent_run_id"


@dataclass
class StructuredEventLineParser:
    """Incrementally parse JSON runtime events without buffering ordinary long lines."""

    structured_events: list[dict]
    run_id: str | None = None
    _candidate: list[str] | None = None
    _line_disqualified: bool = False

    def feed(self, text: str) -> None:
        start = 0
        while True:
            newline_index = text.find("\n", start)
            if newline_index == -1:
                self._feed_fragment(text[start:])
                return
            self._feed_fragment(text[start : newline_index + 1])
            self.finish()
            start = newline_index + 1

    def finish(self) -> None:
        if self._candidate is not None:
            parsed = parse_structured_event("".join(self._candidate), run_id=self.run_id)
            if parsed is not None:
                self.structured_events.append(parsed)
        self._candidate = None
        self._line_disqualified = False

    def _feed_fragment(self, text: str) -> None:
        if not text:
            return
        if self._candidate is not None:
            self._candidate.append(text)
            return
        if self._line_disqualified:
            return
        stripped = text.lstrip()
        if stripped.startswith("{"):
            self._candidate = [stripped]
        elif stripped:
            self._line_disqualified = True


def parse_structured_event(text: str, *, run_id: str | None = None) -> dict | None:
    """Parse one uv_agent_runtime.emit_event JSON line if present."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        value = json.loads(stripped)
    except Exception:
        return None
    if not isinstance(value, dict) or "kind" not in value:
        return None
    event_id = value.get(RUNTIME_EVENT_EVENT_ID_KEY)
    if not isinstance(event_id, str) or not event_id:
        return None
    event_run_id = value.get(RUNTIME_EVENT_RUN_ID_KEY)
    if not isinstance(event_run_id, str) or not event_run_id:
        return None
    if run_id is not None and event_run_id != run_id:
        return None
    return value
