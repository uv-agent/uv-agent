from __future__ import annotations

import json
from functools import lru_cache
from typing import Any


# Runtime event sentinel keys. These mirror values in
# ``uv_agent_runtime.events`` and ``uv_agent.agent`` so the TUI can detect and
# hide structured-event JSON lines that the runner interleaves into stdout.
RUNTIME_EVENT_EVENT_ID_KEY = "_uv_agent_event_id"
RUNTIME_EVENT_RUN_ID_KEY = "_uv_agent_run_id"


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


def tool_call_detail_highlight_markup(call: dict[str, Any]) -> str:
    """Render hidden details with Python syntax highlighting for script source."""
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
        lines.append(python_syntax_markup(code))
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


# Mapping from Pygments token types to Textual markup styles. The list is
# ordered most-specific first; ``token in parent`` walks the token hierarchy
# so e.g. ``Token.Literal.String.Single in Token.Literal.String`` matches.
# Colors intentionally extend the previous tokenize-based palette so existing
# tests around keyword / string highlight continue to hold.
@lru_cache(maxsize=1)
def _pygments_helpers() -> tuple[Any, Any, tuple[tuple[Any, str], ...]]:
    """Return Pygments lexer/style helpers, importing Pygments on first use."""

    from pygments import lex as pyg_lex
    from pygments.lexers.python import PythonLexer
    from pygments.token import Token

    style_map: tuple[tuple[Any, str], ...] = (
        (Token.Comment, "#94a3b8 italic"),
        (Token.Keyword.Constant, "bold #f472b6"),
        (Token.Operator.Word, "bold #7dd3fc"),
        (Token.Keyword, "bold #7dd3fc"),
        (Token.Name.Builtin.Pseudo, "italic #a78bfa"),
        (Token.Name.Builtin, "#a78bfa"),
        (Token.Name.Function, "#facc15"),
        (Token.Name.Class, "bold #facc15"),
        (Token.Name.Decorator, "#fde68a"),
        (Token.Name.Exception, "bold #f87171"),
        (Token.Literal.String.Doc, "#94a3b8 italic"),
        (Token.Literal.String.Escape, "#fb923c"),
        (Token.Literal.String.Interpol, "#fb923c"),
        (Token.Literal.String, "#fbbf24"),
        (Token.Literal.Number, "#fb923c"),
        (Token.Operator, "#94a3b8"),
    )
    # Reuse a single lexer instance. ``stripnl``/``ensurenl`` off so trailing
    # whitespace round-trips exactly.
    return pyg_lex, PythonLexer(stripnl=False, ensurenl=False), style_map


def _pyg_style(token_type: Any) -> str:
    _, _, style_map = _pygments_helpers()
    for parent, style in style_map:
        if token_type in parent:
            return style
    return ""


def python_syntax_markup(code: str) -> str:
    """Return Textual markup with full Python syntax highlighting.

    Uses Pygments' ``PythonLexer`` so the script panel covers keywords,
    builtins, numbers, operators, decorators, class/function names, string
    escapes, f-string interpolation, comments and docstrings. Contiguous
    tokens sharing the same style are coalesced into one markup span to keep
    the output compact and stable.
    """
    if not code:
        return ""
    pieces: list[str] = []
    pending_style = ""
    pending_text = ""

    def flush() -> None:
        nonlocal pending_text
        if not pending_text:
            return
        escaped = escape(pending_text)
        if pending_style:
            pieces.append(f"[{pending_style}]{escaped}[/{pending_style}]")
        else:
            pieces.append(escaped)
        pending_text = ""

    try:
        pyg_lex, py_lexer, _ = _pygments_helpers()
        for token_type, token_text in pyg_lex(code, py_lexer):
            if not token_text:
                continue
            style = _pyg_style(token_type)
            if style != pending_style:
                flush()
                pending_style = style
            pending_text += token_text
        flush()
    except Exception:
        return escape(code)
    return "".join(pieces)


def strip_runtime_event_lines(text: str, *, run_id: str | None = None) -> str:
    """Drop structured runtime-event JSON lines from interleaved stdout."""
    if not text:
        return ""
    kept: list[str] = []
    for line in text.splitlines(keepends=True):
        if _is_runtime_event_line(line, run_id=run_id):
            continue
        kept.append(line)
    return "".join(kept)


def _is_runtime_event_line(line: str, *, run_id: str | None = None) -> bool:
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
    run_id_str = str(payload.get("run_id") or "")
    raw_stdout = strip_runtime_event_lines(
        str(payload.get("stdout") or ""), run_id=run_id_str
    )
    stdout = short_block(raw_stdout)
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
    stdout_raw = strip_runtime_event_lines(
        str(payload.get("stdout") or ""), run_id=run_id
    )
    stdout = short_block(stdout_raw, max_lines=3, max_chars=600)
    if stderr and returncode != 0:
        lines.append("  [red]" + GLYPH_NESTED + " stderr[/red]\n  " + escape(stderr).replace("\n", "\n  "))
    elif stderr:
        lines.append("  [dim]" + GLYPH_NESTED + " stderr[/dim]\n  " + escape(stderr).replace("\n", "\n  "))
    if stdout:
        lines.append("  [dim]" + GLYPH_NESTED + " stdout[/dim]\n  " + escape(stdout).replace("\n", "\n  "))
    if payload.get("truncated"):
        lines.append("  [dim]output truncated[/dim]")
    return "\n".join(lines)


def tool_detail_markup(
    payload: dict[str, Any], *, events_collapsed: bool = False
) -> str:
    """Render complete hidden details for an expandable tool cell.

    Structured runtime events are stripped from the displayed stdout (they are
    already surfaced individually in the events section). Events are rendered
    one per line in a friendly format so backslash escape characters from
    ``json.dumps`` are not shown. The events section can be folded via
    ``events_collapsed=True``.
    """
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
    valid_events = [event for event in events if isinstance(event, dict)]
    if valid_events:
        if events_collapsed:
            lines.append(
                "[dim]events (collapsed · "
                f"{len(valid_events)} events · press e to expand)[/dim]"
            )
        else:
            lines.append("[dim]events (press e to collapse)[/dim]")
            for event in valid_events:
                lines.append(structured_event_markup(event))
    run_id = str(payload.get("run_id") or "")
    stdout = strip_runtime_event_lines(
        str(payload.get("stdout") or ""), run_id=run_id
    ).strip()
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
        return f"{arrow} [magenta]subagent[/magenta] [dim]started[/dim]"
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
