from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from rich.console import Group, RenderableType
from rich.text import Text

from uv_agent.helper_calls import extract_runtime_helper_calls, format_helper_call


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


RenderablePart = str | Text | RenderableType


def markup(value: str) -> Text:
    """Create styled text from trusted, static Rich markup snippets only."""
    return Text.from_markup(value)


def plain(value: Any = "", *, style: str | None = None) -> Text:
    """Create Rich text from external/plain data without markup parsing."""
    return Text(str(value), style=style or "")


def line(*parts: Any, style: str | None = None) -> Text:
    """Compose one ``Text`` line from trusted markup snippets and plain data.

    ``str`` parts are treated as literal text. Use ``markup(...)`` at the call
    site for trusted labels/glyphs that intentionally contain Rich markup.
    """
    text = Text(style=style or "")
    for part in parts:
        if part is None:
            continue
        if isinstance(part, Text):
            text.append_text(part)
        else:
            text.append(str(part))
    return text


def join_lines(parts: Iterable[RenderablePart]) -> Text | Group:
    """Join renderables with newlines, returning ``Text`` when possible.

    Transcript cells mostly need styled text. ``Group`` remains available for
    future non-Text renderables while keeping plain-text extraction explicit.
    """
    items = list(parts)
    if all(isinstance(item, (str, Text)) for item in items):
        result = Text()
        for index, item in enumerate(items):
            if index:
                result.append("\n")
            if isinstance(item, Text):
                result.append_text(item)
            else:
                result.append(str(item))
        return result
    grouped: list[RenderableType] = []
    for index, item in enumerate(items):
        if index:
            grouped.append(Text("\n"))
        grouped.append(item if not isinstance(item, str) else Text(item))
    return Group(*grouped)


def renderable_plain(renderable: object) -> str | None:
    """Best-effort plain text for copy/search/test helpers.

    Rich renderables do not expose a universal plain-text protocol. The TUI uses
    ``Text`` for composed external data, so this helper deliberately handles the
    small set of renderables we create instead of trying to render via a console.
    """
    if renderable is None:
        return ""
    if isinstance(renderable, str):
        return renderable
    if isinstance(renderable, Text):
        return renderable.plain
    if isinstance(renderable, Group):
        rendered = getattr(renderable, "renderables", ())
        plain_parts: list[str] = []
        for part in rendered:
            value = renderable_plain(part)
            if value is None:
                return None
            plain_parts.append(value)
        return "".join(plain_parts)
    plain_attr = getattr(renderable, "plain", None)
    if isinstance(plain_attr, str):
        return plain_attr
    return None


def indent_text(text: str, prefix: str) -> str:
    """Indent every displayed line with ``prefix`` while preserving content."""
    return prefix + text.replace("\n", "\n" + prefix)


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


def tool_call_summary_markup(call: dict[str, Any]) -> Text:
    """Render a run_python call before a result is available."""
    name = str(call.get("name") or "python")
    running = str(call.get("_status_label") or "running")
    preview = tool_call_preview_line(call)
    lines = [
        line(
            markup(f"[#7dd3fc]{GLYPH_RUNNING}[/#7dd3fc] "),
            plain(name, style="bold"),
            " ",
            plain(running, style="dim"),
        )
    ]
    if preview:
        lines.append(line(markup(f"  [dim]{GLYPH_NESTED} script[/dim] "), preview))
    return join_lines(lines)  # type: ignore[return-value]


def tool_call_detail_markup(call: dict[str, Any]) -> Text:
    """Render hidden details for a run_python call without dumping full source."""

    args = tool_call_args(call)
    lines: list[Text] = [markup("[dim]call[/dim]"), line("name: ", str(call.get("name") or "python"))]
    call_id = str(call.get("call_id") or "")
    if call_id:
        lines.append(line("call_id: ", call_id))
    code = str(args.get("code") or "").strip()
    if code:
        lines.append(markup("[dim]helpers[/dim]"))
        helper_calls = extract_runtime_helper_calls(code)
        if helper_calls:
            lines.extend(plain(format_helper_call(helper)) for helper in helper_calls)
        else:
            lines.append(plain("(no uv_agent_runtime helpers)", style="dim"))
    if args:
        remainder = {key: value for key, value in args.items() if key != "code"}
        if remainder:
            lines.append(markup("[dim]arguments[/dim]"))
            lines.append(json_markup(remainder))
    else:
        raw_args = str(call.get("arguments") or "").strip()
        if raw_args:
            lines.append(markup("[dim]arguments[/dim]"))
            lines.append(plain(raw_args))
    return join_lines(lines)  # type: ignore[return-value]


def tool_call_detail_highlight_markup(call: dict[str, Any]) -> Text:
    """Render hidden details for a run_python call without dumping full source."""

    return tool_call_detail_markup(call)


def tool_call_helper_payload(call: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract helper-call details from a run_python call for result payloads."""

    return extract_runtime_helper_calls(tool_call_code(call))


# Mapping from Pygments token types to Rich styles. The list is ordered
# most-specific first; ``token in parent`` walks the token hierarchy so e.g.
# ``Token.Literal.String.Single in Token.Literal.String`` matches.
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


def python_syntax_markup(code: str) -> Text:
    """Return styled Rich text with full Python syntax highlighting."""
    if not code:
        return Text()
    result = Text()
    pending_style = ""
    pending_text = ""

    def flush() -> None:
        nonlocal pending_text
        if not pending_text:
            return
        result.append(pending_text, style=pending_style or None)
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
        return plain(code)
    return result


def strip_runtime_event_lines(text: str, *, run_id: str | None = None) -> str:
    """Drop structured runtime-event JSON lines from interleaved stdout."""
    if not text:
        return ""
    kept: list[str] = []
    for line_value in text.splitlines(keepends=True):
        if _is_runtime_event_line(line_value, run_id=run_id):
            continue
        kept.append(line_value)
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


def _tool_status_glyph(returncode: Any, timed_out: bool, *, partial: bool = False) -> tuple[str, str]:
    """Return (glyph, color) for a tool call status."""

    if partial:
        return GLYPH_RUNNING, "#7dd3fc"
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


def tool_result_markup(payload: dict[str, Any]) -> Text:
    """Render a Python runner result as a compact transcript block."""
    returncode = payload.get("returncode")
    timed_out = bool(payload.get("timed_out"))
    truncated = bool(payload.get("truncated"))
    run_id = str(payload.get("run_id") or "-")
    partial = bool(payload.get("partial"))
    glyph, color = _tool_status_glyph(returncode, timed_out, partial=partial)
    status = "running" if partial else "timeout" if timed_out else f"exit {returncode}"
    elapsed = _payload_elapsed(payload)

    header = line(
        markup(f"[{color}]{glyph}[/{color}] [bold]python[/bold] "),
        plain(f"{run_id} ·", style="dim"),
        " ",
        plain(status, style=color),
    )
    if elapsed:
        header.append(" ")
        header.append_text(plain(f"· {elapsed}", style="dim"))
    lines: list[Text] = [header]
    run_id_str = str(payload.get("run_id") or "")
    raw_stdout = strip_runtime_event_lines(str(payload.get("stdout") or ""), run_id=run_id_str)
    stdout = short_block(raw_stdout)
    stderr = short_block(str(payload.get("stderr") or ""))
    if stdout:
        lines.append(join_lines([markup("[dim]stdout[/dim]"), plain(stdout)]))  # type: ignore[arg-type]
    if stderr:
        label = markup("[dim]stderr[/dim]") if returncode == 0 and not timed_out else markup("[red]stderr[/red]")
        lines.append(join_lines([label, plain(stderr)]))  # type: ignore[arg-type]
    if partial:
        lines.append(markup("[dim]still running; output is partial[/dim]"))
    if truncated:
        lines.append(markup("[dim]output truncated[/dim]"))
    return join_lines(lines)  # type: ignore[return-value]


def tool_timeline_markup(payload: dict[str, Any]) -> Text:
    """Render a one-cell tool timeline item with structured events."""
    returncode = payload.get("returncode")
    timed_out = bool(payload.get("timed_out"))
    partial = bool(payload.get("partial"))
    glyph, color = _tool_status_glyph(returncode, timed_out, partial=partial)
    status = "running" if partial else "timeout" if timed_out else f"exit {returncode}"
    run_id = str(payload.get("run_id") or "-")
    elapsed = _payload_elapsed(payload)
    header = line(
        markup(f"[{color}]{glyph}[/{color}] [bold]python[/bold] "),
        plain(run_id, style="dim"),
        " ",
        plain(status, style=color),
    )
    if elapsed:
        header.append(" ")
        header.append_text(plain(f"· {elapsed}", style="dim"))
    lines: list[Text] = [header]
    events = payload.get("events")
    if not isinstance(events, list):
        events = []
    for event in events[:5]:
        if not isinstance(event, dict):
            continue
        event_line = Text("  ")
        event_line.append_text(structured_event_markup(event))
        lines.append(event_line)
    if len(events) > 5:
        lines.append(plain(f"  … +{len(events) - 5} more events", style="dim"))
    stderr = short_block(str(payload.get("stderr") or ""), max_lines=3, max_chars=600)
    stdout_raw = strip_runtime_event_lines(str(payload.get("stdout") or ""), run_id=run_id)
    stdout = short_block(stdout_raw, max_lines=3, max_chars=600)
    if stderr and returncode != 0:
        lines.append(join_lines([markup(f"  [red]{GLYPH_NESTED} stderr[/red]"), plain(indent_text(stderr, "  "))]))  # type: ignore[arg-type]
    elif stderr:
        lines.append(join_lines([markup(f"  [dim]{GLYPH_NESTED} stderr[/dim]"), plain(indent_text(stderr, "  "))]))  # type: ignore[arg-type]
    if stdout:
        lines.append(join_lines([markup(f"  [dim]{GLYPH_NESTED} stdout[/dim]"), plain(indent_text(stdout, "  "))]))  # type: ignore[arg-type]
    if payload.get("partial"):
        lines.append(markup("  [dim]still running; output is partial[/dim]"))
    if payload.get("truncated"):
        lines.append(markup("  [dim]output truncated[/dim]"))
    return join_lines(lines)  # type: ignore[return-value]


def tool_detail_markup(payload: dict[str, Any], *, events_collapsed: bool = False) -> Text:
    """Render hidden tool details as helper calls only."""

    lines: list[Text] = [markup("[dim]helpers[/dim]")]
    helpers = payload.get("helper_calls")
    valid_helpers = [helper for helper in helpers if isinstance(helper, dict)] if isinstance(helpers, list) else []
    if valid_helpers:
        lines.extend(plain(format_helper_call(helper)) for helper in valid_helpers)
    else:
        lines.append(plain("(no uv_agent_runtime helpers)", style="dim"))
    return join_lines(lines)  # type: ignore[return-value]


def structured_event_markup(event: dict[str, Any]) -> Text:
    """Render one uv_agent_runtime structured event for compact timelines."""
    kind = str(event.get("kind") or "event")
    prefix = markup(f"[dim]{GLYPH_NESTED}[/dim] ")
    if kind == "progress":
        return line(prefix, markup("[cyan]progress[/cyan] "), str(event.get("message") or ""))
    if kind == "result":
        return line(prefix, markup("[cyan]result[/cyan] "), plain(json.dumps(event, ensure_ascii=False), style="dim"))
    if kind == "look_at":
        return line(prefix, markup("[cyan]look_at[/cyan] "), str(event.get("path") or ""))
    if kind == "workflow.started":
        workflow_id = _workflow_event_ref(event, key_name="key", id_name="workflow_id")
        objective = _event_first_line(event.get("objective"), max_chars=80)
        return line(
            prefix,
            markup("[magenta]workflow[/magenta] "),
            plain(f"{workflow_id} started", style="dim"),
            plain(f" {objective}" if objective else ""),
        )
    if kind == "workflow.node.started":
        node = _workflow_event_ref(event, key_name="key", id_name="node_id")
        node_kind = str(event.get("node_kind") or "node")
        return line(
            prefix,
            markup("[magenta]workflow[/magenta] "),
            plain(f"node {node} started", style="dim"),
            plain(f" {node_kind}" if node_kind and node_kind != "node" else "", style="dim"),
        )
    if kind in {"workflow.node.completed", "workflow.node.failed"}:
        node = _workflow_event_ref(event, key_name="key", id_name="node_id")
        status = "completed" if kind == "workflow.node.completed" else "failed"
        detail = _workflow_node_result_detail(event)
        return line(
            prefix,
            markup("[magenta]workflow[/magenta] "),
            plain(f"node {node} {status}", style="dim"),
            plain(f" {detail}" if detail else "", style="dim"),
        )
    if kind == "workflow.checkpoint.reached":
        checkpoint = _workflow_event_ref(event, key_name="key", id_name="checkpoint_id")
        return line(prefix, markup("[magenta]workflow[/magenta] "), plain(f"checkpoint {checkpoint} reached", style="dim"))
    return line(prefix, plain(kind, style="dim"), " ", plain(json.dumps(event, ensure_ascii=False), style="dim"))


def _workflow_event_ref(event: dict[str, Any], *, key_name: str, id_name: str) -> str:
    """Return the stable human label for a workflow event entity."""

    key = str(event.get(key_name) or "").strip()
    if key:
        return key
    identifier = str(event.get(id_name) or "").strip()
    return short_thread(identifier) if identifier else "?"


def _workflow_node_result_detail(event: dict[str, Any]) -> str:
    details: list[str] = []
    thread_id = str(event.get("thread_id") or "")
    if thread_id:
        details.append(f"thread {short_thread(thread_id)}")
    if bool(event.get("timed_out")):
        details.append("timeout")
    else:
        returncode = event.get("returncode")
        if returncode not in (None, 0):
            details.append(f"exit {returncode}")
    return " · ".join(details)


def _event_first_line(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip().splitlines()[0] if str(value or "").strip() else ""
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def json_markup(value: object) -> Text:
    """Render JSON as plain Rich text for transcript display."""
    return plain(json.dumps(value, ensure_ascii=False, indent=2))
