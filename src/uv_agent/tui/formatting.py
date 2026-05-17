from __future__ import annotations

import json
from typing import Any


# Glyphs shared by transcript cells. Sigils (not emoji) render uniformly across
# terminals; chosen to match codex/gemini/opencode conventions.
GLYPH_USER = "›"
GLYPH_ASSISTANT = "✦"
GLYPH_REASONING = "·"
GLYPH_EVENT = "•"
GLYPH_OK = "✓"
GLYPH_ERR = "✗"
GLYPH_RUNNING = "⠿"
GLYPH_NESTED = "└─"


def escape(value: Any) -> str:
    """Escape text for Textual markup surfaces."""
    return str(value).replace("[", "\\[").replace("]", "\\]")


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


def tool_call_args(call: dict[str, Any]) -> dict[str, Any]:
    """Decode a run_python function call's JSON arguments."""
    raw_args = call.get("arguments") or ""
    if not isinstance(raw_args, str) or not raw_args:
        return {}
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


def tool_call_code(call: dict[str, Any] | None) -> str:
    """Return the full Python source embedded in a run_python call."""
    if not isinstance(call, dict):
        return ""
    return str(tool_call_args(call).get("code") or "").strip()


def tool_call_preview_line(call: dict[str, Any] | None, *, max_chars: int = 90) -> str:
    """Return the first meaningful Python source line for compact summaries."""
    code = tool_call_code(call)
    if not code:
        return ""
    first = next((line.strip() for line in code.splitlines() if line.strip()), "")
    if len(first) > max_chars:
        first = first[: max_chars - 3].rstrip() + "..."
    return first


def tool_call_summary_markup(call: dict[str, Any]) -> str:
    """Render a run_python call before a result is available."""
    name = str(call.get("name") or "python")
    running = str(call.get("_status_label") or "running")
    preview = tool_call_preview_line(call)
    lines = [
        f"[#7dd3fc]{GLYPH_RUNNING}[/#7dd3fc] [bold]{escape(name)}[/bold] [dim]{escape(running)}[/dim]"
    ]
    if preview:
        lines.append(f"  [dim]{GLYPH_NESTED} script[/dim] {escape(preview)}")
    return "\n".join(lines)


def tool_call_detail_markup(call: dict[str, Any]) -> str:
    """Render hidden details for a run_python call, including full source."""
    args = tool_call_args(call)
    lines = [
        "[dim]call[/dim]",
        f"name: {escape(str(call.get('name') or 'python'))}",
    ]
    call_id = str(call.get("call_id") or "")
    if call_id:
        lines.append(f"call_id: {escape(call_id)}")
    code = str(args.get("code") or "").strip()
    if code:
        lines.append("[dim]script[/dim]")
        lines.append(escape(code))
    if args:
        remainder = {key: value for key, value in args.items() if key != "code"}
        if remainder:
            lines.append("[dim]arguments[/dim]")
            lines.append(escape(json.dumps(remainder, ensure_ascii=False, indent=2)))
    else:
        raw_args = str(call.get("arguments") or "").strip()
        if raw_args:
            lines.append("[dim]arguments[/dim]")
            lines.append(escape(raw_args))
    return "\n".join(lines)


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


def format_elapsed(seconds: float | int | None) -> str:
    """Compact elapsed-time formatting like codex (`12s`, `1m 02s`, `1h 02m`)."""
    if seconds is None:
        return ""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _tool_status_glyph(returncode: Any, timed_out: bool) -> tuple[str, str]:
    """Return (glyph, color) for a finished tool call."""
    if timed_out:
        return GLYPH_ERR, "yellow"
    if returncode == 0:
        return GLYPH_OK, "green"
    return GLYPH_ERR, "red"


def _payload_elapsed(payload: dict[str, Any]) -> str:
    """Pull an elapsed-time label from a runner payload if present."""
    for key in ("duration_s", "elapsed_s", "elapsed", "duration"):
        if key in payload:
            try:
                return format_elapsed(float(payload[key]))
            except (TypeError, ValueError):
                continue
    started = payload.get("started_at")
    ended = payload.get("ended_at")
    try:
        if started is not None and ended is not None:
            return format_elapsed(float(ended) - float(started))
    except (TypeError, ValueError):
        pass
    return ""


def tool_result_markup(payload: dict[str, Any]) -> str:
    """Render a Python runner result as a compact transcript block."""
    returncode = payload.get("returncode")
    timed_out = bool(payload.get("timed_out"))
    truncated = bool(payload.get("truncated"))
    script_id = str(payload.get("script_id") or "-")
    run_id = str(payload.get("run_id") or "-")
    glyph, color = _tool_status_glyph(returncode, timed_out)
    status = "timeout" if timed_out else f"exit {returncode}"
    elapsed = _payload_elapsed(payload)
    elapsed_suffix = f" [dim]· {escape(elapsed)}[/dim]" if elapsed else ""

    header = (
        f"[{color}]{glyph}[/{color}] [bold]python[/bold] "
        f"[dim]{escape(script_id)} · {escape(run_id)} ·[/dim] "
        f"[{color}]{status}[/{color}]{elapsed_suffix}"
    )
    lines = [header]
    stdout = short_block(str(payload.get("stdout") or ""))
    stderr = short_block(str(payload.get("stderr") or ""))
    if stdout:
        lines.append("[dim]stdout[/dim]\n" + escape(stdout))
    if stderr:
        label = "[dim]stderr[/dim]" if returncode == 0 and not timed_out else "[red]stderr[/red]"
        lines.append(f"{label}\n" + escape(stderr))
    if truncated:
        lines.append("[dim]output truncated[/dim]")
    return "\n".join(lines)


def tool_timeline_markup(payload: dict[str, Any]) -> str:
    """Render a one-cell tool timeline item with structured events."""
    returncode = payload.get("returncode")
    timed_out = bool(payload.get("timed_out"))
    glyph, color = _tool_status_glyph(returncode, timed_out)
    status = "timeout" if timed_out else f"exit {returncode}"
    script_id = str(payload.get("script_id") or "-")
    run_id = str(payload.get("run_id") or "-")
    elapsed = _payload_elapsed(payload)
    elapsed_suffix = f" [dim]· {escape(elapsed)}[/dim]" if elapsed else ""
    lines = [
        f"[{color}]{glyph}[/{color}] [bold]python[/bold] "
        f"[dim]{escape(script_id)} · {escape(run_id)}[/dim] "
        f"[{color}]{status}[/{color}]{elapsed_suffix}"
    ]
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    for event in events[:5]:
        if not isinstance(event, dict):
            continue
        lines.append("  " + structured_event_markup(event))
    if len(events) > 5:
        lines.append(f"  [dim]… +{len(events) - 5} more events[/dim]")
    stderr = short_block(str(payload.get("stderr") or ""), max_lines=3, max_chars=600)
    stdout = short_block(str(payload.get("stdout") or ""), max_lines=3, max_chars=600)
    if stderr and returncode != 0:
        lines.append("  [red]" + GLYPH_NESTED + " stderr[/red]\n  " + escape(stderr).replace("\n", "\n  "))
    elif stderr:
        lines.append("  [dim]" + GLYPH_NESTED + " stderr[/dim]\n  " + escape(stderr).replace("\n", "\n  "))
    if stdout:
        lines.append("  [dim]" + GLYPH_NESTED + " stdout[/dim]\n  " + escape(stdout).replace("\n", "\n  "))
    if payload.get("truncated"):
        lines.append("  [dim]output truncated[/dim]")
    return "\n".join(lines)


def tool_detail_markup(payload: dict[str, Any]) -> str:
    """Render complete hidden details for an expandable tool cell."""
    lines = [
        "[dim]details[/dim]",
        f"script_id: {escape(str(payload.get('script_id') or '-'))}",
        f"run_id: {escape(str(payload.get('run_id') or '-'))}",
    ]
    elapsed = _payload_elapsed(payload)
    if elapsed:
        lines.append(f"elapsed: {escape(elapsed)}")
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
    arrow = f"[dim]{GLYPH_NESTED}[/dim]"
    if kind == "progress":
        message = str(event.get("message") or "")
        return f"{arrow} [cyan]progress[/cyan] {escape(message)}"
    if kind == "result":
        return f"{arrow} [cyan]result[/cyan] [dim]{escape(json.dumps(event, ensure_ascii=False))}[/dim]"
    if kind == "look_at":
        return f"{arrow} [cyan]look_at[/cyan] {escape(str(event.get('path') or ''))}"
    if kind == "subagent.started":
        prompt = str(event.get("prompt") or "").splitlines()[0]
        if len(prompt) > 90:
            prompt = prompt[:87].rstrip() + "..."
        return f"{arrow} [magenta]subagent[/magenta] [dim]started[/dim] {escape(prompt)}"
    if kind == "subagent.completed":
        thread_id = str(event.get("thread_id") or "")
        summary = str(event.get("summary") or "").splitlines()[0]
        if len(summary) > 90:
            summary = summary[:87].rstrip() + "..."
        detail = f" {escape(short_thread(thread_id))}" if thread_id else ""
        return f"{arrow} [magenta]subagent[/magenta] [dim]completed{detail}[/dim] {escape(summary)}"
    return f"{arrow} [dim]{escape(kind)}[/dim] [dim]{escape(json.dumps(event, ensure_ascii=False))}[/dim]"


def json_markup(value: object) -> str:
    """Render JSON with escaped markup for transcript display."""
    return escape(json.dumps(value, ensure_ascii=False, indent=2))
