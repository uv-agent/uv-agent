from __future__ import annotations

import json
from typing import Any

from rich.markup import escape


def parse_tool_payload(output_item: dict[str, Any]) -> dict[str, Any] | None:
    """Decode a run_python tool output item into the runner payload."""
    raw = output_item.get("output")
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def short_block(value: str, *, max_lines: int = 8, max_chars: int = 1800) -> str:
    """Return a terminal-friendly preview of stdout or stderr."""
    value = value.strip()
    if not value:
        return ""
    lines = value.splitlines()
    clipped = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        clipped += f"\n... {len(lines) - max_lines} more lines"
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars].rstrip() + "\n..."
    return clipped


def short_thread(thread_id: str | None) -> str:
    """Render a compact thread id for the status line."""
    if not thread_id:
        return "new"
    return thread_id[-8:]


def tool_result_markup(payload: dict[str, Any]) -> str:
    """Render a Python runner result as a compact transcript block."""
    returncode = payload.get("returncode")
    timed_out = bool(payload.get("timed_out"))
    truncated = bool(payload.get("truncated"))
    script_id = str(payload.get("script_id") or "-")
    run_id = str(payload.get("run_id") or "-")
    status = "timeout" if timed_out else f"exit {returncode}"
    color = "green" if returncode == 0 and not timed_out else "red"

    lines = [
        f"[{color}]python[/{color}] [dim]{escape(script_id)} · {escape(run_id)} ·[/dim] [{color}]{status}[/{color}]"
    ]
    stdout = short_block(str(payload.get("stdout") or ""))
    stderr = short_block(str(payload.get("stderr") or ""))
    if stdout:
        lines.append("[dim]stdout[/dim]\n" + escape(stdout))
    if stderr:
        label = "stderr" if returncode == 0 and not timed_out else "[red]stderr[/red]"
        lines.append(f"{label}\n" + escape(stderr))
    if truncated:
        lines.append("[dim]output truncated[/dim]")
    return "\n".join(lines)


def json_markup(value: object) -> str:
    """Render JSON with escaped markup for transcript display."""
    return escape(json.dumps(value, ensure_ascii=False, indent=2))
