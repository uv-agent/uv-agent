from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any
from uuid import uuid4

RUNTIME_EVENT_EVENT_ID_KEY = "_uv_agent_event_id"
RUNTIME_EVENT_RUN_ID_KEY = "_uv_agent_run_id"
_EVENT_WRITE_LOCK = threading.Lock()


def emit_event(kind: str, **payload: Any) -> dict[str, Any]:
    """Emit a structured event on stdout for the runner or user to inspect."""
    event = {"kind": kind, **payload}
    event[RUNTIME_EVENT_EVENT_ID_KEY] = f"evt_{uuid4().hex}"
    run_id = os.environ.get("UV_AGENT_RUNTIME_RUN_ID")
    if run_id:
        event[RUNTIME_EVENT_RUN_ID_KEY] = run_id
    _write_event_line(json.dumps(event, ensure_ascii=False))
    return event


def _write_event_line(line: str) -> None:
    text = line + "\n"
    with _EVENT_WRITE_LOCK:
        try:
            fd = sys.stdout.fileno()
        except (AttributeError, OSError):
            sys.stdout.write(text)
            sys.stdout.flush()
            return
        sys.stdout.flush()
        os.write(fd, text.encode("utf-8"))
        sys.stdout.flush()


def emit_progress(message: str, **payload: Any) -> dict[str, Any]:
    """Emit a progress event from a temporary script."""
    return emit_event("progress", message=message, **payload)


def emit_result(**payload: Any) -> dict[str, Any]:
    """Emit a final structured result event from a temporary script."""
    return emit_event("result", **payload)
