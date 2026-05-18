from __future__ import annotations

import json
import os
import sys
from typing import Any

RUNTIME_EVENT_RUN_ID_KEY = "_uv_agent_run_id"


def emit_event(kind: str, **payload: Any) -> None:
    """Emit a structured event on stdout for the runner or user to inspect."""
    event = {"kind": kind, **payload}
    run_id = os.environ.get("UV_AGENT_RUN_ID")
    if run_id:
        event[RUNTIME_EVENT_RUN_ID_KEY] = run_id
    print(json.dumps(event, ensure_ascii=False), flush=True)


def emit_progress(message: str, **payload: Any) -> None:
    """Emit a progress event from a temporary script."""
    emit_event("progress", message=message, **payload)


def emit_result(**payload: Any) -> None:
    """Emit a final structured result event from a temporary script."""
    emit_event("result", **payload)
