from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from .transport import emit_event_rpc

RUNTIME_EVENT_EVENT_ID_KEY = "_uv_agent_event_id"
RUNTIME_EVENT_RUN_ID_KEY = "_uv_agent_run_id"


def emit_event(kind: str, **payload: Any) -> dict[str, Any]:
    """Emit a structured event to the host over the runtime RPC channel."""

    event = {"kind": kind, **payload}
    event[RUNTIME_EVENT_EVENT_ID_KEY] = f"evt_{uuid4().hex}"
    run_id = os.environ.get("UV_AGENT_RUNTIME_RUN_ID")
    if run_id:
        event[RUNTIME_EVENT_RUN_ID_KEY] = run_id
    emit_event_rpc(event)
    return event


def emit_progress(message: str, **payload: Any) -> dict[str, Any]:
    """Emit a progress event from a temporary script."""

    return emit_event("progress", message=message, **payload)


def emit_result(**payload: Any) -> dict[str, Any]:
    """Emit a final structured result event from a temporary script."""

    return emit_event("result", **payload)


# Facade aliases used by `import uv_agent_runtime as rt` when the submodule is imported explicitly.
emit = emit_event
progress = emit_progress
result = emit_result

def look_at(path, *, note=""):
    from .vision import look_at as _look_at

    return _look_at(path, note=note)
