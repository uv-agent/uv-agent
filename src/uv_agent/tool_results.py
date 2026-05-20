from __future__ import annotations

import json
from typing import Any

RUNTIME_EVENT_EVENT_ID_KEY = "_uv_agent_event_id"
RUNTIME_EVENT_RUN_ID_KEY = "_uv_agent_run_id"


def function_output(call: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call.get("call_id"),
        "output": json.dumps(output, ensure_ascii=False),
    }


def model_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the run payload that is safe and useful to feed back to the model."""
    visible = {
        "script_id": payload.get("script_id"),
        "run_id": payload.get("run_id"),
        "returncode": payload.get("returncode"),
        "timed_out": payload.get("timed_out"),
        "interrupted": payload.get("interrupted"),
        "truncated": payload.get("truncated"),
        "stdout": strip_structured_event_lines(
            str(payload.get("stdout") or ""),
            run_id=str(payload.get("run_id") or ""),
        ),
        "stderr": payload.get("stderr") or "",
    }
    if payload.get("rules_loaded"):
        visible["rules_loaded"] = payload["rules_loaded"]
    return visible


def strip_structured_event_lines(text: str, *, run_id: str | None = None) -> str:
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if _is_structured_event_line(line, run_id=run_id):
            continue
        lines.append(line)
    return "".join(lines)


def _is_structured_event_line(line: str, *, run_id: str | None = None) -> bool:
    stripped = line.strip()
    if not stripped.startswith("{"):
        return False
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    if not isinstance(value, dict) or "kind" not in value:
        return False
    event_id = value.get(RUNTIME_EVENT_EVENT_ID_KEY)
    if not isinstance(event_id, str) or not event_id:
        return False
    event_run_id = value.get(RUNTIME_EVENT_RUN_ID_KEY)
    if not isinstance(event_run_id, str) or not event_run_id:
        return False
    return not run_id or event_run_id == run_id
