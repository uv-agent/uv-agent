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


def format_tokens(value: int | None) -> str:
    """Format token counts for compact TUI status surfaces."""
    if value is None:
        return "-"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.0f}K"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


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


def tool_timeline_markup(payload: dict[str, Any]) -> str:
    """Render a one-cell tool timeline item with structured events."""
    returncode = payload.get("returncode")
    timed_out = bool(payload.get("timed_out"))
    color = "green" if returncode == 0 and not timed_out else "red"
    status = "timeout" if timed_out else f"exit {returncode}"
    script_id = str(payload.get("script_id") or "-")
    run_id = str(payload.get("run_id") or "-")
    lines = [
        f"[{color}]python[/{color}] [dim]{escape(script_id)} · {escape(run_id)}[/dim] [{color}]{status}[/{color}]"
    ]
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    for event in events[:5]:
        if not isinstance(event, dict):
            continue
        lines.append(structured_event_markup(event))
    if len(events) > 5:
        lines.append(f"[dim]... {len(events) - 5} more events[/dim]")
    stderr = short_block(str(payload.get("stderr") or ""), max_lines=3, max_chars=600)
    if stderr and returncode != 0:
        lines.append("[red]stderr[/red]\n" + escape(stderr))
    elif stderr:
        lines.append("[dim]stderr[/dim]\n" + escape(stderr))
    if payload.get("stdout"):
        lines.append("[dim]stdout hidden in details[/dim]")
    if payload.get("truncated"):
        lines.append("[dim]output truncated[/dim]")
    return "\n".join(lines)


def tool_detail_markup(payload: dict[str, Any]) -> str:
    """Render complete hidden details for an expandable tool cell."""
    lines = [
        "[dim]details[/dim]",
        f"script_id: {escape(str(payload.get('script_id') or '-'))}",
        f"run_id: {escape(str(payload.get('run_id') or '-'))}",
    ]
    run_log_path = str(payload.get("run_log_path") or "")
    if run_log_path:
        lines.append(f"run_log_path: {escape(run_log_path)}")
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    if events:
        lines.append("[dim]events[/dim]")
        lines.append(escape(json.dumps(events, ensure_ascii=False, indent=2)))
    stdout = str(payload.get("stdout") or "").strip()
    stderr = str(payload.get("stderr") or "").strip()
    if stdout:
        lines.append("[dim]stdout[/dim]")
        lines.append(escape(stdout))
    if stderr:
        lines.append("[dim]stderr[/dim]")
        lines.append(escape(stderr))
    return "\n".join(lines)


def structured_event_markup(event: dict[str, Any]) -> str:
    """Render one uv_agent_runtime structured event for compact timelines."""
    kind = str(event.get("kind") or "event")
    if kind == "progress":
        message = str(event.get("message") or "")
        return f"[dim]↳ progress[/dim] {escape(message)}"
    if kind == "result":
        return f"[dim]↳ result[/dim] {escape(json.dumps(event, ensure_ascii=False))}"
    if kind == "look_at":
        return f"[dim]↳ look_at[/dim] {escape(str(event.get('path') or ''))}"
    if kind == "subagent.started":
        prompt = str(event.get("prompt") or "").splitlines()[0]
        if len(prompt) > 90:
            prompt = prompt[:87].rstrip() + "..."
        return f"[magenta]↳ subagent[/magenta] [dim]started[/dim] {escape(prompt)}"
    if kind == "subagent.completed":
        thread_id = str(event.get("thread_id") or "")
        summary = str(event.get("summary") or "").splitlines()[0]
        if len(summary) > 90:
            summary = summary[:87].rstrip() + "..."
        detail = f" {escape(short_thread(thread_id))}" if thread_id else ""
        return f"[magenta]↳ subagent[/magenta] [dim]completed{detail}[/dim] {escape(summary)}"
    return f"[dim]↳ {escape(kind)}[/dim] {escape(json.dumps(event, ensure_ascii=False))}"


def json_markup(value: object) -> str:
    """Render JSON with escaped markup for transcript display."""
    return escape(json.dumps(value, ensure_ascii=False, indent=2))
