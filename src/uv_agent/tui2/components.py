from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from uv_agent.environment import UserLanguage, normalize_language
from uv_agent.helper_calls import extract_runtime_helper_calls, format_helper_call
from uv_agent.i18n import tr
from uv_agent.tui.formatting import format_elapsed, short_block, short_thread
from uv_agent.tui2.ansi import strip_ansi, truncate_visible, visible_len, wrap_plain
from uv_agent.tui2.events import CommandSuggestion, TranscriptCell, Tui2State, tool_title
from uv_agent.tui2.theme import AnsiTheme, DEFAULT_THEME, sgr


def _resolve_language(value: UserLanguage | str | None) -> UserLanguage:
    """Allow callers to pass either a ``UserLanguage`` or a short code/string.

    Tests and ad hoc callers find ``language="zh"`` more ergonomic than building
    a ``UserLanguage`` object; the renderer/state still own a real object.
    """

    if isinstance(value, UserLanguage):
        return value
    return normalize_language(value or "en")


# Breath animation calibration: target 12 phase changes per second while text
# is streaming at 100 characters/second.  Working backwards gives one phase
# change every 100/12 ≈ 8.33 characters of streamed content.  When the cell
# is idle (no new chars) the animation naturally pauses, which is desirable.
_BREATH_CHARS_PER_PHASE = max(1, round(100 / 12))


def _breath_frame(cell: TranscriptCell) -> int:
    """Return the breath animation phase index for *cell*.

    Driven by the cell's cumulative streamed character count rather than the
    wall-clock spinner so the animation speed reflects actual model throughput.
    """

    return cell.chars_streamed // _BREATH_CHARS_PER_PHASE


# ---------------------------------------------------------------------------
# Message cells
# ---------------------------------------------------------------------------


def _prefix_lines(prefix: str, text: str, width: int) -> list[str]:
    body_width = max(1, width - visible_len(prefix))
    wrapped: list[str] = []
    for para in text.split("\n"):
        wrapped.extend(wrap_plain(para, body_width) or [""])
    if not wrapped:
        wrapped = [""]
    indent = " " * visible_len(prefix)
    return [prefix + wrapped[0], *(indent + line for line in wrapped[1:])]


def render_markdown(text: str, width: int) -> list[str]:
    if not text.strip():
        return [""]
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        color_system="256",
        width=max(20, width),
        legacy_windows=False,
    )
    try:
        console.print(Markdown(text))
    except Exception:
        return wrap_plain(text, width)
    rendered = stream.getvalue().rstrip("\n")
    return rendered.splitlines() or [""]


# Reserved width for the reasoning suffix.  Three characters covers the longest
# animation frame ("...") so padding the suffix to this width keeps the body
# column-stable and removes the per-frame jitter the user observed.
_REASONING_SUFFIX_WIDTH = 3
_BREATHING_DOTS = ("·", "•", "●", "•")
_ASSISTANT_PREFIX_STYLES = ("assistant", "accent", "success", "warning", "accent")


def render_reasoning_cell(
    cell: TranscriptCell,
    width: int,
    theme: AnsiTheme = DEFAULT_THEME,
    *,
    spinner_frame: int = 0,
) -> list[str]:
    """Compact one-line reasoning view with a non-spinner breathing dot.

    The rotating activity spinner is reserved for the terminal title and status
    row.  Reasoning still gets a tiny first-column pulse so the user can see it
    is live without duplicating the global spinner animation.
    """

    # The breath phase is driven by streamed-character throughput so the dot
    # paces itself with model output rather than wall-clock spinner ticks.
    breath_frame = _breath_frame(cell)
    dot = _BREATHING_DOTS[breath_frame % len(_BREATHING_DOTS)] if cell.status == "streaming" else "·"
    dot_style = theme.reasoning if dot != "●" else theme.reasoning + ";1"
    prefix = sgr(dot_style, dot + " ")
    compact_text = " ".join(cell.text.strip().split())
    plain_width = max(1, width - 2)
    if visible_len(compact_text) <= plain_width:
        return [prefix + sgr(theme.dim, compact_text)]
    suffix = "…".ljust(_REASONING_SUFFIX_WIDTH)
    body_width = max(
        1,
        width - 2 - _REASONING_SUFFIX_WIDTH - 1,  # prefix + " " + suffix
    )
    body = truncate_visible(compact_text, body_width, suffix="")
    body_pad = " " * max(0, body_width - visible_len(body))
    return [prefix + sgr(theme.dim, body + body_pad) + " " + sgr(theme.dim, suffix)]


def render_message_cell(
    cell: TranscriptCell,
    width: int,
    theme: AnsiTheme = DEFAULT_THEME,
    *,
    spinner_frame: int = 0,
) -> list[str]:
    if cell.kind == "user":
        return _prefix_lines(sgr(theme.user, "› "), cell.text, width)
    if cell.kind == "reasoning":
        return render_reasoning_cell(cell, width, theme)
    if cell.kind == "error":
        return _prefix_lines(sgr(theme.error, "✗ "), cell.text or cell.title, width)
    if cell.kind == "event":
        return _prefix_lines(sgr(theme.muted, "◆ "), cell.text or cell.title, width)
    if cell.kind == "image":
        return _prefix_lines(sgr(theme.accent, "▧ "), cell.text or cell.title, width)
    # Mirror the reasoning breath: the assistant glyph colour cycles based on
    # streamed-character throughput so its tempo matches the model.
    style_idx = _breath_frame(cell) if cell.status == "streaming" else 0
    style_name = _ASSISTANT_PREFIX_STYLES[style_idx % len(_ASSISTANT_PREFIX_STYLES)]
    prefix = sgr(getattr(theme, style_name), "✦ ")
    md_lines = render_markdown(cell.text, max(20, width - 2))
    if not md_lines:
        md_lines = [""]
    return [prefix + md_lines[0], *("  " + line for line in md_lines[1:])]


# ---------------------------------------------------------------------------
# Tool cells (lighter than the original double-border card)
# ---------------------------------------------------------------------------


_TOOL_STDOUT_MAX_LINES = 5
_TOOL_STDERR_MAX_LINES = 3
_TOOL_HELPER_MAX_LINES = 6


def _rule(label: str, width: int, style: str, theme: AnsiTheme) -> str:
    head = "── "
    label_plain = strip_ansi(label).strip()
    if label_plain:
        text = head + label
        text_width = visible_len(text)
        fill = max(1, width - text_width - 1)
        # Use a single colour for both the leading "── " and trailing fill so
        # the rule reads as one continuous separator; the prior split between
        # ``border_faint`` and ``style`` looked like a colour seam.
        return sgr(style, head) + label + " " + sgr(style, "─" * fill)
    return sgr(style, "─" * max(1, width))


def _thin_rule(width: int, theme: AnsiTheme = DEFAULT_THEME) -> str:
    """Subtle separator used between live transcript and input chrome."""

    if width <= 8:
        return sgr(theme.border_accent, "─" * max(1, width))
    left = "─" * max(1, width // 3)
    right = "─" * max(1, width - len(left))
    return sgr(theme.border_accent, left) + sgr(theme.border_faint, right)


def _indented(text: str, width: int, style: str) -> str:
    inner = max(1, width - 2)
    clipped = truncate_visible(text.expandtabs(4), inner)
    return "  " + sgr(style, clipped)


def render_tool_cell(cell: TranscriptCell, width: int, theme: AnsiTheme = DEFAULT_THEME) -> list[str]:
    payload = cell.payload or {}
    running = cell.status == "running"
    returncode = payload.get("returncode")
    timed_out = bool(payload.get("timed_out"))
    errored = timed_out or (returncode not in (None, 0) and not running)
    if running:
        glyph = "⠿"
        status = f"running · {format_elapsed(cell.elapsed_s)}"
        border_style = theme.border_accent
        glyph_style = theme.border_accent
    elif errored:
        glyph = "✗"
        status = "timeout" if timed_out else f"exit {returncode}"
        border_style = theme.error
        glyph_style = theme.error
    else:
        glyph = "✓"
        status = f"exit {returncode}" if returncode is not None else "done"
        border_style = theme.border
        glyph_style = theme.success
    run_id = str(payload.get("run_id") or (cell.call or {}).get("call_id") or "")
    suffix = f" · {run_id[-12:]}" if run_id else ""
    title = (
        sgr(glyph_style, glyph)
        + " "
        + sgr(theme.tool_title, tool_title(cell.call))
        + " "
        + sgr(theme.muted, "· " + status + suffix)
    )
    lines = [_rule(title, width, border_style, theme)]
    code = ""
    if cell.call:
        import json

        try:
            args = json.loads(str(cell.call.get("arguments") or "{}"))
            code = str(args.get("code") or "").strip()
        except Exception:
            code = ""
    payload_helpers = payload.get("helper_calls")
    helper_calls = [helper for helper in payload_helpers if isinstance(helper, dict)] if isinstance(payload_helpers, list) else []
    if not helper_calls and code:
        helper_calls = extract_runtime_helper_calls(code)
    if helper_calls:
        for helper in helper_calls[:_TOOL_HELPER_MAX_LINES]:
            lines.append(_indented(format_helper_call(helper), width, theme.muted))
        if len(helper_calls) > _TOOL_HELPER_MAX_LINES:
            lines.append(_indented(f"… more helpers +{len(helper_calls) - _TOOL_HELPER_MAX_LINES} calls", width, theme.muted))
    elif code:
        lines.append(_indented("(no uv_agent_runtime helpers)", width, theme.muted))
    stdout = short_block(str(payload.get("stdout") or ""), max_lines=_TOOL_STDOUT_MAX_LINES, max_chars=1000)
    stderr = short_block(str(payload.get("stderr") or ""), max_lines=_TOOL_STDERR_MAX_LINES, max_chars=700)
    if stdout or stderr:
        if helper_calls or code:
            lines.append(sgr(theme.border, "  ─"))
        for block, style in ((stdout, theme.tool_output), (stderr, theme.error)):
            if not block:
                continue
            for raw in block.splitlines():
                lines.append(_indented(raw, width, style))
    elif running:
        lines.append(_indented("waiting for run_python output…", width, theme.muted))
    return lines


def render_cell(
    cell: TranscriptCell,
    width: int,
    theme: AnsiTheme = DEFAULT_THEME,
    *,
    spinner_frame: int = 0,
) -> list[str]:
    """Render one cell.  No leading horizontal rules; cells are separated by
    blank lines when flushed into scrollback, which keeps the transcript clean
    and lets the terminal background show through.
    """

    if cell.kind == "tool":
        return render_tool_cell(cell, width, theme)
    if cell.kind == "reasoning":
        return render_reasoning_cell(cell, width, theme, spinner_frame=spinner_frame)
    return render_message_cell(cell, width, theme, spinner_frame=spinner_frame)


# ---------------------------------------------------------------------------
# Status lines: a verbose two-row context strip above the composer.
# ---------------------------------------------------------------------------


def render_status_lines(
    state: Tui2State,
    width: int,
    spinner_frame: int,
    theme: AnsiTheme = DEFAULT_THEME,
) -> list[str]:
    """Up to two muted status rows.

    Row 1 — activity: spinner + elapsed, queued count, last error.
    Row 2 — context: thread title, model level, goal badge, project path.

    Empty rows are dropped so a fresh idle session collapses to nothing.
    """

    lang = _resolve_language(state.language)
    busy_fallback = tr(lang, "working")
    queued_label = tr(lang, "queued")

    activity: list[str] = []
    if state.busy:
        frame = theme.spinner_frames[spinner_frame % len(theme.spinner_frames)]
        elapsed = format_elapsed(state.turn_elapsed_s) if state.turn_elapsed_s is not None else ""
        status = state.status_message if state.status_message not in {"", "ready", "running"} else busy_fallback
        text = f"{frame} {status}" + (f" · {elapsed}" if elapsed else "")
        activity.append(sgr(theme.accent, text))
    if state.pending_turns:
        activity.append(sgr(theme.warning, f"↕ {len(state.pending_turns)} {queued_label}"))
    if not state.busy and state.status_message and state.status_message not in {"ready", "running", ""}:
        activity.append(sgr(theme.muted, state.status_message))
    if state.last_error and not state.busy:
        activity.append(sgr(theme.error, f"✗ {state.last_error}"))

    context: list[str] = []
    if state.goal_enabled:
        label = "⊕ " + tr(lang, "goal").lower()
        if state.goal_objective:
            obj = state.goal_objective.splitlines()[0]
            label = f"{label}: {obj[:40]}"
        context.append(sgr(theme.success, label))
    # Thread title is intentionally omitted: the terminal title already shows
    # it, and the /status command surfaces the full metadata.  The bottom row
    # only needs model + project context to stay scannable.
    if state.level:
        level = state.level
        if state.context_percent is not None:
            level = f"{level} · {state.context_percent}%"
        context.append(sgr(theme.muted, level))
    if state.project_path:
        context.append(sgr(theme.muted, _shorten_path(state.project_path, max_len=48)))

    lines: list[str] = []
    if activity:
        lines.append(truncate_visible(sgr(theme.muted, "◆ ") + "  ".join(activity), width))
    if context:
        lines.append(truncate_visible(sgr(theme.muted, "◇ ") + " · ".join(context), width))
    return lines


def _shorten_path(path: str, *, max_len: int = 48, home: Path | None = None) -> str:
    """Shorten paths for status rows, preferring a stable ``~`` prefix."""

    if not path:
        return path
    display = path
    try:
        home_path = (home or Path.home()).resolve()
        target = Path(path).expanduser().resolve()
        if target == home_path:
            display = "~"
        elif target.is_relative_to(home_path):
            separator = "\\" if "\\" in path else "/"
            display = "~" + separator + str(target.relative_to(home_path))
    except (OSError, RuntimeError, ValueError):
        try:
            home_text = str((home or Path.home()).resolve())
            if path == home_text or path.startswith(home_text + "\\") or path.startswith(home_text + "/"):
                display = "~" + path[len(home_text) :]
        except (OSError, RuntimeError, ValueError):
            display = path
    if len(display) <= max_len:
        return display
    return "…" + display[-(max_len - 1) :]


def render_command_palette(
    items: list[CommandSuggestion],
    selected: int,
    width: int,
    theme: AnsiTheme = DEFAULT_THEME,
    *,
    max_rows: int = 8,
) -> list[str]:
    """Render the slash-command palette above the composer."""

    inner = max(8, width - 4)
    window_size = max(1, max_rows)
    if items:
        selected = max(0, min(selected, len(items) - 1))
        start = min(max(0, selected - window_size + 1), max(0, len(items) - window_size))
    else:
        selected = 0
        start = 0
    visible = items[start : start + window_size]
    hidden_before = start
    hidden_after = max(0, len(items) - (start + len(visible)))
    rows = [sgr(theme.command_palette_border, "╭" + "─" * (width - 2) + "╮")]
    if not visible:
        text = sgr(theme.muted, "No matching commands")
        pad = " " * max(0, inner - visible_len(text))
        rows.append(sgr(theme.command_palette_border, "│ ") + text + pad + sgr(theme.command_palette_border, " │"))
    else:
        for index, item in enumerate(visible):
            absolute_index = start + index
            active = absolute_index == selected
            marker = sgr(theme.accent, "› ") if active else "  "
            value_style = theme.command_palette_selected if active else theme.command_palette
            desc_parts = []
            if item.description:
                desc_parts.append(item.description)
            if item.meta:
                desc_parts.append(item.meta)
            desc = f" — {' · '.join(desc_parts)}" if desc_parts else ""
            text = marker + sgr(value_style, item.value) + sgr(theme.muted, desc)
            pad = " " * max(0, inner - visible_len(text))
            rows.append(
                sgr(theme.command_palette_border, "│ ")
                + truncate_visible(text, inner)
                + pad
                + sgr(theme.command_palette_border, " │")
            )
        if hidden_before or hidden_after:
            parts: list[str] = []
            if hidden_before:
                parts.append(f"↑ {hidden_before}")
            if hidden_after:
                parts.append(f"↓ {hidden_after}")
            more = sgr(theme.muted, "… " + " · ".join(parts))
            pad = " " * max(0, inner - visible_len(more))
            rows.append(sgr(theme.command_palette_border, "│ ") + more + pad + sgr(theme.command_palette_border, " │"))
    rows.append(sgr(theme.command_palette_border, "╰" + "─" * (width - 2) + "╯"))
    return rows


# ---------------------------------------------------------------------------
# Composer: rounded-corner box, single-row when empty, grows with input.
# ---------------------------------------------------------------------------


def render_composer_with_cursor(
    text: str,
    width: int,
    theme: AnsiTheme = DEFAULT_THEME,
    *,
    max_input_rows: int = 8,
    cursor_index: int | None = None,
    language: UserLanguage | str | None = "en",
) -> tuple[list[str], int, int]:
    """Render the boxed composer and the cursor position inside it.

    Layout (width = W):
        ╭─── … ─╮      ← row 0
        │ › ... │      ← row 1 (first input row)
        │   ... │      ← additional rows for continuation lines
        ╰─── … ─╯      ← bottom border
    """

    inner = max(8, width - 4)  # space between "│ " and " │"
    body_inner = inner - 2  # leave room for "› " or "  "
    cursor_index = len(text) if cursor_index is None else max(0, min(cursor_index, len(text)))

    if not text:
        hint = sgr(theme.muted, tr(_resolve_language(language), "placeholder"))
        body = sgr(theme.accent, "› ") + hint
        body_plain_len = visible_len(body)
        pad = " " * max(0, inner - body_plain_len)
        input_row = sgr(theme.border, "│ ") + truncate_visible(body, inner) + pad + sgr(theme.border, " │")
        rows = [
            sgr(theme.border, "╭" + "─" * (width - 2) + "╮"),
            input_row,
            sgr(theme.border, "╰" + "─" * (width - 2) + "╯"),
        ]
        cursor_col = 2 + 2  # "│ " + "› "
        return rows, 1, cursor_col

    wrapped: list[str] = []
    cursor_wrapped_row = 0
    cursor_wrapped_col = 0
    consumed = 0
    # ``splitlines()`` drops the final empty field for trailing newlines, which
    # made Ctrl+Enter look like it did nothing until the user typed another
    # character.  Split on the literal separator so a trailing newline reserves
    # and paints the next blank input row immediately.
    for raw in text.split("\n"):
        segment_start = consumed
        segment_end = segment_start + len(raw)
        parts = wrap_plain(raw, body_inner) or [""]
        if segment_start <= cursor_index <= segment_end:
            rel = cursor_index - segment_start
            remaining = rel
            local_row = 0
            local_col = 0
            for part_index, part in enumerate(parts):
                part_len = len(part)
                if remaining <= part_len or part_index == len(parts) - 1:
                    local_row = part_index
                    local_col = remaining
                    break
                remaining -= part_len
            cursor_wrapped_row = len(wrapped) + local_row
            cursor_wrapped_col = visible_len(parts[local_row][:local_col])
        wrapped.extend(parts)
        consumed = segment_end + 1
    if not wrapped:
        wrapped = [""]
    visible_rows = max(1, min(max_input_rows, len(wrapped)))
    # Keep the head of multi-line input visible while the composer still has
    # room.  The previous cursor-following window replaced the first visible row
    # with an "earlier lines hidden" marker as soon as the cursor moved beyond
    # the small box, which made the start of a pasted prompt disappear.  Only
    # tail the cursor once we need to scroll beyond the initial visible window.
    clipped_start = 0 if cursor_wrapped_row < visible_rows else cursor_wrapped_row - visible_rows + 1
    clipped = wrapped[clipped_start : clipped_start + visible_rows]
    hidden_before = clipped_start
    hidden_after = max(0, len(wrapped) - (clipped_start + len(clipped)))

    input_rows: list[str] = []
    for idx, body in enumerate(clipped):
        absolute_row = clipped_start + idx
        prefix = sgr(theme.accent, "› ") if absolute_row == 0 else "  "
        line = prefix + body
        pad = " " * max(0, inner - visible_len(line))
        input_rows.append(sgr(theme.border, "│ ") + line + pad + sgr(theme.border, " │"))

    rows = [_composer_top_border(width, theme, hidden_before=hidden_before, hidden_after=hidden_after)]
    rows.extend(input_rows)
    rows.append(sgr(theme.border, "╰" + "─" * (width - 2) + "╯"))

    cursor_row = 1 + max(0, min(cursor_wrapped_row - clipped_start, len(clipped) - 1))
    cursor_col = 2 + 2 + cursor_wrapped_col  # "│ " + line prefix + cursor text
    return rows, cursor_row, cursor_col


def _composer_top_border(
    width: int,
    theme: AnsiTheme,
    *,
    hidden_before: int = 0,
    hidden_after: int = 0,
) -> str:
    """Render the top composer border, including clipping status if needed.

    The hidden-line status lives in the border rather than in an input row so it
    never hides the first visible line of a multi-line paste.
    """

    inner_width = max(0, width - 2)
    if hidden_before <= 0 and hidden_after <= 0:
        return sgr(theme.border, "╭" + "─" * inner_width + "╮")

    if hidden_before > 0 and hidden_after > 0:
        label = f"… {hidden_before} earlier · {hidden_after} later lines hidden"
    elif hidden_before > 0:
        label = f"… {hidden_before} earlier lines hidden"
    else:
        label = f"… {hidden_after} later lines hidden"
    content = truncate_visible(f"─ {label} ", inner_width)
    content += "─" * max(0, inner_width - visible_len(content))
    return sgr(theme.border, "╭" + content + "╮")


def render_composer(text: str, width: int, theme: AnsiTheme = DEFAULT_THEME) -> list[str]:
    return render_composer_with_cursor(text, width, theme)[0]


def render_live_with_cursor(
    state: Tui2State,
    width: int,
    spinner_frame: int = 0,
    theme: AnsiTheme = DEFAULT_THEME,
    *,
    max_height: int | None = None,
) -> tuple[list[str], int, int]:
    """Render in-flight cells, optional status lines, and the boxed composer.

    When ``max_height`` is set, the live region is bounded to that many
    rows.  Cell lines are dropped from the front (replaced with a ``…
    N earlier lines hidden`` marker) so the painted area never exceeds the
    viewport — which is the only way to avoid the streaming-assistant
    "duplicate scrollback" artefact, since once the live region scrolls
    out from under us our erase math can no longer reach the top.
    """

    cell_lines: list[str] = []
    for cell in state.live:
        cell_lines.extend(render_cell(cell, width, theme, spinner_frame=spinner_frame))
    status_lines = render_status_lines(state, width, spinner_frame, theme)
    composer_lines, row, col = render_composer_with_cursor(
        state.composer,
        width,
        theme,
        cursor_index=state.composer_cursor,
        language=state.language,
    )
    palette_lines = (
        render_command_palette(
            state.command_palette_items,
            state.command_palette_index,
            width,
            theme,
        )
        if state.command_palette_open
        else []
    )

    if max_height is not None and cell_lines:
        gaps = sum(1 for group in (status_lines, palette_lines) if group)
        reserved = len(composer_lines) + len(status_lines) + len(palette_lines) + gaps
        available = max(1, max_height - reserved)
        if len(cell_lines) > available:
            dropped = len(cell_lines) - available + 1
            marker = sgr(theme.muted, f"…  {dropped} earlier lines hidden in live; full text on completion")
            cell_lines = [marker] + cell_lines[dropped:]

    lines: list[str] = list(cell_lines)
    if status_lines:
        if lines:
            lines.append("")
        lines.extend(status_lines)
    if palette_lines:
        if lines:
            lines.append("")
        lines.extend(palette_lines)
    cursor_row = len(lines) + row
    lines.extend(composer_lines)
    bounded = [truncate_visible(line, width) for line in lines]
    return bounded, cursor_row, col


def render_live(state: Tui2State, width: int, spinner_frame: int = 0) -> list[str]:
    return render_live_with_cursor(state, width, spinner_frame)[0]
