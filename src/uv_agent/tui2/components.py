from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from uv_agent.environment import UserLanguage, normalize_language
from uv_agent.helper_calls import (
    extract_import_anchor_chains,
    format_import_anchor_chains,
)
from uv_agent.i18n import tr
from uv_agent.tui.formatting import (
    format_elapsed,
    renderable_plain,
    short_thread,
)
from uv_agent.tui2.ansi import strip_ansi, truncate_visible, visible_len, wrap_plain
from uv_agent.tui2.events import (
    AGENT_VIEW_STATUS_ORDER,
    AgentViewRow,
    CommandSuggestion,
    TranscriptCell,
    Tui2State,
    tool_title,
)
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
# change every 100/12 ≈ 8.33 characters of streamed content.
_BREATH_CHARS_PER_PHASE = max(1, round(100 / 12))


def _breath_frame(cell: TranscriptCell) -> int:
    """Return the breath animation phase index for *cell*.

    Live cells normally carry a fractional phase integrated from the current
    turn's sliding-window throughput estimate.  ``chars_streamed`` remains as a
    compatibility fallback for tests and restored scrollback cells.
    """

    if cell.animation_phase is not None:
        return int(cell.animation_phase)
    return cell.chars_streamed // _BREATH_CHARS_PER_PHASE


def _format_token_rate(value: float) -> str:
    if value >= 100:
        return f"{value:.0f} tok/s"
    if value >= 10:
        return f"{value:.1f} tok/s"
    return f"{value:.2f} tok/s"


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


def _live_tool_rule(label: str, width: int, style: str, theme: AnsiTheme) -> str:
    """Return a non-full-width rule for frequently repainted live tool rows.

    Running tool headers repaint on every spinner tick.  A full-width rule is
    fragile on Windows terminals when font fallback renders Braille/box/ambiguous
    glyphs wider than our Python cell-width estimate: the physical row can wrap
    by one line, so the renderer's later erase lands below the leaked header.
    Keep generous right-side slack while the tool is live; the completed result
    still uses the full-width rule when it is flushed once into scrollback.
    """

    head = "── "
    label_width = max(1, width - visible_len(head) - 8)
    return sgr(style, head) + truncate_visible(label, label_width)


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
    """Compact two-line tool cell: status + imported call chains.

    Full source, stdout/stderr, and events are intentionally omitted here;
    use the ``/show <run_id>`` pager to inspect them.
    """

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
    lines = [
        _live_tool_rule(title, width, border_style, theme)
        if running
        else _rule(title, width, border_style, theme)
    ]
    chains = _tool_cell_import_chains(cell)
    if chains:
        compact = " · ".join(format_import_anchor_chains(chains))
        lines.append(_indented(compact, width, theme.muted))
    return lines


def _tool_cell_import_chains(cell: TranscriptCell) -> list[str]:
    """Extract imported-name anchored call chains from a tool cell."""

    payload = cell.payload or {}
    payload_helpers = payload.get("helper_calls")
    if isinstance(payload_helpers, list) and payload_helpers:
        # Legacy payloads may store helper_calls as {"name": ..., "args": ...}.
        # We only need the names and cannot reconstruct chains from these,
        # so treat each as a single-chain item.
        return [str(h.get("name") or "helper") for h in payload_helpers if isinstance(h, dict)]
    code = ""
    if cell.call:
        try:
            args = json.loads(str(cell.call.get("arguments") or "{}"))
            code = str(args.get("code") or "").strip()
        except Exception:
            code = ""
    if code:
        return extract_import_anchor_chains(code)
    return []


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

GOAL_OBJECTIVE_STATUS_MAX_CELLS = 24
IMAGE_TOKEN_RE = re.compile(r"\[Image #(\d+)\]")


def _goal_objective_status_text(objective: str) -> str:
    """Return a compact one-line Goal objective for the activity row."""

    text = " ".join(objective.split())
    if not text:
        return ""
    return truncate_visible(text, GOAL_OBJECTIVE_STATUS_MAX_CELLS)


def _style_composer_image_tokens(text: str, token_numbers: set[int], theme: AnsiTheme) -> str:
    if not token_numbers or "[Image #" not in text:
        return text
    parts: list[str] = []
    last = 0
    for match in IMAGE_TOKEN_RE.finditer(text):
        number = int(match.group(1))
        parts.append(text[last : match.start()])
        token = match.group(0)
        parts.append(sgr(theme.image_token, token) if number in token_numbers else token)
        last = match.end()
    parts.append(text[last:])
    return "".join(parts)


def render_status_lines(
    state: Tui2State,
    width: int,
    spinner_frame: int,
    theme: AnsiTheme = DEFAULT_THEME,
) -> list[str]:
    """Up to two muted status rows.

    Row 1 — activity: spinner + elapsed, queued count, last error.
    Row 2 — context: thread title, model level, Goal badge, project path.

    Empty rows are dropped so a fresh idle session collapses to nothing.
    """

    lang = _resolve_language(state.language)
    busy_fallback = tr(lang, "working")
    queued_label = tr(lang, "queued")

    activity: list[str] = []
    if state.busy:
        frame = theme.spinner_frames[spinner_frame % len(theme.spinner_frames)]
        elapsed = format_elapsed(state.turn_elapsed_s) if state.turn_elapsed_s is not None else ""
        token_rate = _format_token_rate(state.turn_token_rate) if state.turn_token_rate is not None else ""
        status = state.status_message if state.status_message not in {"", "ready", "running"} else busy_fallback
        rendered_text = sgr(theme.accent, f"{frame} {status}")
        if elapsed:
            rendered_text += sgr(theme.accent, f" · {elapsed}")
        if token_rate:
            token_style = theme.muted if state.turn_token_rate_frozen else theme.accent
            rendered_text += sgr(theme.accent, " · ") + sgr(token_style, token_rate)
        objective = _goal_objective_status_text(state.goal_objective) if state.goal_enabled else ""
        if objective:
            rendered_text += sgr(theme.muted, " · ") + sgr(theme.goal, objective)
        activity.append(rendered_text)
    if state.pending_turns:
        activity.append(sgr(theme.warning, f"↕ {len(state.pending_turns)} {queued_label}"))
    if not state.busy and state.status_message and state.status_message not in {"ready", "running", ""}:
        activity.append(sgr(theme.muted, state.status_message))
    if state.last_error and not state.busy:
        activity.append(sgr(theme.error, f"✗ {state.last_error}"))

    context: list[str] = []
    # Thread title is intentionally omitted: the terminal title already shows
    # it, and the /status command surfaces the full metadata.  The bottom row
    # only needs model + project context to stay scannable.
    if state.level:
        context.append(sgr(theme.muted, state.level))
        if state.goal_enabled:
            context.append(sgr(theme.goal, "Goal"))
        if state.context_percent is not None:
            context.append(sgr(theme.muted, f"{state.context_percent}%"))
    elif state.goal_enabled:
        context.append(sgr(theme.goal, "Goal"))
    if state.project_path:
        context.append(sgr(theme.muted, _shorten_path(state.project_path, max_len=48)))

    lines: list[str] = []
    if activity:
        lines.append(truncate_visible(sgr(theme.muted, "◆ ") + "  ".join(activity), width))
    if context:
        # Style separators explicitly instead of relying on whichever segment
        # precedes them.  Otherwise a row like "level · 31% · path" gives the
        # first dot the level colour and leaves the second dot unstyled.
        lines.append(truncate_visible(sgr(theme.muted, "◇ ") + sgr(theme.muted, " · ").join(context), width))
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
# Agent View dashboard
# ---------------------------------------------------------------------------


_AGENT_STATUS_GLYPHS = {
    "dispatching": "…",
    "working": "⠿",
    "queued": "↕",
    "completed": "✓",
    "failed": "✗",
    "interrupted": "■",
}


def render_agent_view_with_cursor(
    state: Tui2State,
    width: int,
    spinner_frame: int = 0,
    theme: AnsiTheme = DEFAULT_THEME,
    *,
    max_height: int | None = None,
) -> tuple[list[str], int, int]:
    """Render the full-screen Agent View dashboard and input row."""

    width = max(20, width)
    max_rows = max_height if max_height is None else max(1, max_height)
    view = state.agent_view
    lang = _resolve_language(state.language)
    if view.interaction_mode == "help":
        return _render_agent_help_view(state, width, theme, max_rows)
    if view.interaction_mode == "model":
        return _render_agent_model_view(state, width, theme, max_rows)

    header = _agent_header_line(width, theme, lang=lang, mode=view.interaction_mode)

    if max_rows is not None and max_rows <= 4:
        return _render_compact_agent_view(state, width, theme, max_rows)

    # Render the bottom chrome first so the session list can consume only the
    # remaining rows.  The previous fixed ``- 6`` estimate missed the optional
    # hidden-row marker, peek/status rows, and multi-line composer, which could
    # push the panel past the viewport and make the terminal scroll away rows
    # that the renderer later tried to erase.
    composer_max_rows = 3
    if max_rows is not None:
        composer_max_rows = max(1, min(3, max_rows - 5))
    composer_lines, composer_cursor_row, cursor_col = _render_agent_composer_with_cursor(
        state,
        width,
        theme,
        max_input_rows=composer_max_rows,
    )
    peek_lines = _agent_peek_lines(state, width, theme)
    if max_rows is not None:
        # Prefer keeping the status/confirmation line when vertical space is
        # tight; the detailed peek can be restored with a taller terminal.
        while len(peek_lines) > 1 and _agent_chrome_height(peek_lines, composer_lines) > max_rows:
            peek_lines = peek_lines[:1]
        if _agent_chrome_height(peek_lines, composer_lines) > max_rows:
            peek_lines = []

    body_budget = max(0, (max_rows or 30) - _agent_chrome_height(peek_lines, composer_lines))
    body_lines = _agent_body_lines(state, width, spinner_frame, theme, max_lines=body_budget)

    rows: list[str] = [header]
    rows.extend(body_lines)
    rows.append(_agent_separator(width, theme))
    rows.extend(peek_lines)
    rows.append(_agent_separator(width, theme))
    cursor_row = len(rows) + composer_cursor_row
    rows.extend(composer_lines)
    rows.append(sgr(theme.border_accent, "╰" + "─" * (width - 2) + "╯"))

    if max_rows is not None and len(rows) > max_rows:
        rows, cursor_row = _trim_agent_rows_to_height(rows, cursor_row, max_rows)

    return [truncate_visible(line, width) for line in rows], min(cursor_row, len(rows) - 1), cursor_col


def _agent_header_line(
    width: int,
    theme: AnsiTheme,
    *,
    lang: UserLanguage | None = None,
    mode: str | None = None,
) -> str:
    lang = lang or normalize_language("en")
    suffix = ""
    if mode:
        suffix = f" · {_agent_mode_label(mode, lang)}"
    title = f"─ {tr(lang, 'agent_view_title')}{suffix} "
    fill = "─" * max(0, width - visible_len("╭" + title + "╮"))
    return sgr(theme.border_accent, "╭" + title + fill + "╮")


def _render_compact_agent_view(
    state: Tui2State,
    width: int,
    theme: AnsiTheme,
    max_rows: int,
) -> tuple[list[str], int, int]:
    """Fallback for very short terminals where the full dashboard cannot fit."""

    lang = _resolve_language(state.language)
    composer_lines, composer_cursor_row, cursor_col = _render_agent_composer_with_cursor(
        state,
        width,
        theme,
        max_input_rows=1,
    )
    if max_rows <= 1:
        return [truncate_visible(composer_lines[0], width)], 0, cursor_col
    if max_rows == 2:
        lines = [_agent_header_line(width, theme, lang=lang, mode=state.agent_view.interaction_mode), composer_lines[0]]
        return [truncate_visible(line, width) for line in lines], 1, cursor_col
    lines = [
        _agent_header_line(width, theme, lang=lang, mode=state.agent_view.interaction_mode),
        composer_lines[min(composer_cursor_row, len(composer_lines) - 1)],
        sgr(theme.border_accent, "╰" + "─" * (width - 2) + "╯"),
    ]
    return [truncate_visible(line, width) for line in lines[:max_rows]], 1, cursor_col


def _render_agent_help_view(
    state: Tui2State,
    width: int,
    theme: AnsiTheme,
    max_rows: int | None,
) -> tuple[list[str], int, int]:
    lang = _resolve_language(state.language)
    budget = max_rows or 30
    content_budget = max(0, budget - 2)
    content = _agent_help_lines(lang, width, theme)
    if len(content) > content_budget:
        hidden = len(content) - content_budget + 1
        content = content[: max(0, content_budget - 1)] + [
            _agent_box_line(sgr(theme.muted, tr(lang, "agent_view_help_more").format(count=hidden)), width, theme)
        ]
    rows = [_agent_header_line(width, theme, lang=lang, mode="help"), *content]
    if budget > 1:
        rows.append(sgr(theme.border_accent, "╰" + "─" * (width - 2) + "╯"))
    rows = rows[:budget]
    return [truncate_visible(line, width) for line in rows], min(max(0, len(rows) - 1), len(rows) - 1), 0


def _render_agent_model_view(
    state: Tui2State,
    width: int,
    theme: AnsiTheme,
    max_rows: int | None,
) -> tuple[list[str], int, int]:
    """Render the Agent View model picker as a bounded full-screen panel."""

    lang = _resolve_language(state.language)
    view = state.agent_view
    budget = max_rows or 30
    content_budget = max(0, budget - 2)
    rows: list[str] = [_agent_header_line(width, theme, lang=lang, mode="model")]
    options = view.model_options
    if not options:
        rows.append(_agent_box_line(sgr(theme.muted, tr(lang, "agent_view_no_models")), width, theme))
    else:
        selected = max(0, min(view.model_selected, len(options) - 1))
        max_option_rows = max(1, content_budget - 2)
        start = min(max(0, selected - max_option_rows // 2), max(0, len(options) - max_option_rows))
        visible = options[start : start + max_option_rows]
        hidden_before = start
        hidden_after = max(0, len(options) - (start + len(visible)))
        rows.append(_agent_box_line(sgr(theme.muted, tr(lang, "agent_view_model_hint")), width, theme))
        for offset, option in enumerate(visible):
            absolute = start + offset
            active = absolute == selected
            marker = sgr(theme.accent, "▸") if active else " "
            value_style = theme.accent if option.id == view.dispatch_level else theme.command_palette
            model = f" — {option.description}" if option.description else ""
            current = f" · {tr(lang, 'current')}" if option.id == view.dispatch_level else ""
            line = f"{marker} " + sgr(value_style, option.value) + sgr(theme.muted, model + current)
            rows.append(_agent_box_line(line, width, theme))
        if hidden_before or hidden_after:
            rows.append(_agent_hidden_marker(width, theme, hidden_before=hidden_before, hidden_after=hidden_after))
    if budget > 1:
        rows.append(sgr(theme.border_accent, "╰" + "─" * (width - 2) + "╯"))
    rows = rows[:budget]
    return [truncate_visible(line, width) for line in rows], min(max(0, len(rows) - 1), len(rows) - 1), 0


def _agent_help_lines(lang: UserLanguage, width: int, theme: AnsiTheme) -> list[str]:
    rows: list[str] = []
    for line in tr(lang, "agent_view_help_body").split("\n"):
        if not line:
            rows.append(_agent_box_line("", width, theme))
            continue
        rows.append(_agent_box_line(line, width, theme))
    return rows


def _agent_mode_label(mode: str, lang: UserLanguage) -> str:
    if mode == "input":
        return tr(lang, "agent_view_mode_input")
    if mode == "help":
        return tr(lang, "agent_view_mode_help")
    if mode == "model":
        return tr(lang, "agent_view_mode_model")
    return tr(lang, "agent_view_mode_normal")


def _agent_chrome_height(peek_lines: list[str], composer_lines: list[str]) -> int:
    # Header, body/peek separator, peek, composer separator, composer, footer.
    return 1 + 1 + len(peek_lines) + 1 + len(composer_lines) + 1


def _agent_body_lines(
    state: Tui2State,
    width: int,
    spinner_frame: int,
    theme: AnsiTheme,
    *,
    max_lines: int,
) -> list[str]:
    view = state.agent_view
    grouped: dict[str, list[tuple[int, AgentViewRow]]] = {status: [] for status in AGENT_VIEW_STATUS_ORDER}
    for index, row in enumerate(view.rows):
        grouped.setdefault(row.status, []).append((index, row))

    if not view.rows:
        lang = _resolve_language(state.language)
        empty = [
            _agent_box_line(sgr(theme.muted, tr(lang, "agent_view_no_sessions")), width, theme),
            _agent_box_line(sgr(theme.dim, tr(lang, "agent_view_empty_hint")), width, theme),
        ]
        return empty[:max(0, max_lines)]

    if max_lines <= 0:
        return []

    items: list[str] = []
    selected_item_index = 0
    lang = _resolve_language(state.language)
    for status in AGENT_VIEW_STATUS_ORDER:
        entries = grouped.get(status) or []
        if not entries:
            continue
        label = f"{tr(lang, f'agent_view_status_{status}')} ({len(entries)})"
        items.append(_agent_box_line(sgr(theme.muted, label), width, theme))
        for absolute, row in entries:
            if absolute == view.selected:
                selected_item_index = len(items)
            items.append(_agent_row_line(row, absolute == view.selected, width, spinner_frame, theme))

    if len(items) <= max_lines:
        return items
    if max_lines == 1:
        return [_agent_hidden_marker(width, theme, hidden_before=0, hidden_after=len(items))]

    window = max_lines - 1
    start = min(max(0, selected_item_index - window // 2), max(0, len(items) - window))
    visible = items[start : start + window]
    hidden_before = start
    hidden_after = max(0, len(items) - (start + len(visible)))
    marker = _agent_hidden_marker(width, theme, hidden_before=hidden_before, hidden_after=hidden_after)
    if hidden_before and not hidden_after:
        return [marker] + visible
    return visible + [marker]


def _agent_hidden_marker(width: int, theme: AnsiTheme, *, hidden_before: int, hidden_after: int) -> str:
    parts: list[str] = []
    if hidden_before:
        parts.append(f"↑ {hidden_before}")
    if hidden_after:
        parts.append(f"↓ {hidden_after}")
    detail = " · ".join(parts) if parts else "0"
    return _agent_box_line(sgr(theme.muted, f"… {detail} rows hidden"), width, theme)


def _trim_agent_rows_to_height(rows: list[str], cursor_row: int, max_rows: int) -> tuple[list[str], int]:
    """Last-resort guard: never let Agent View exceed the renderer budget."""

    if len(rows) <= max_rows:
        return rows, cursor_row
    keep_from = max(0, len(rows) - max_rows)
    return rows[keep_from:], max(0, cursor_row - keep_from)


def render_agent_view(
    state: Tui2State,
    width: int,
    spinner_frame: int = 0,
) -> list[str]:
    return render_agent_view_with_cursor(state, width, spinner_frame)[0]


def _agent_box_line(content: str, width: int, theme: AnsiTheme) -> str:
    inner = max(1, width - 4)
    clipped = truncate_visible(content, inner)
    pad = " " * max(0, inner - visible_len(clipped))
    return sgr(theme.border_accent, "│ ") + clipped + pad + sgr(theme.border_accent, " │")


def _agent_separator(width: int, theme: AnsiTheme) -> str:
    return (
        sgr(theme.border_accent, "│")
        + sgr(theme.border_faint, "─" * max(1, width - 2))
        + sgr(theme.border_accent, "│")
    )


def _agent_row_line(
    row: AgentViewRow,
    selected: bool,
    width: int,
    spinner_frame: int,
    theme: AnsiTheme,
) -> str:
    marker = sgr(theme.accent, "▸") if selected else " "
    glyph = _AGENT_STATUS_GLYPHS.get(row.status, "·")
    if row.status in {"working", "dispatching"}:
        glyph = theme.spinner_frames[spinner_frame % len(theme.spinner_frames)]
    glyph_style = {
        "completed": theme.success,
        "failed": theme.error,
        "interrupted": theme.warning,
        "queued": theme.warning,
    }.get(row.status, theme.accent)
    short_id = short_thread(row.thread_id)
    title = row.title or "New thread"
    elapsed = format_elapsed(row.elapsed_seconds) if row.elapsed_seconds else ""
    queue = f"q{row.queued_turns}" if row.queued_turns else ""
    branch = row.worktree_branch or ""
    suffix_parts = [part for part in (elapsed, queue, branch) if part]
    suffix = " · ".join(suffix_parts)
    prefix = f"{marker} {short_id} "
    right = f" {glyph}"
    if suffix:
        right += f" {suffix}"
    available = max(1, width - 4 - visible_len(prefix) - visible_len(right) - 1)
    title_text = truncate_visible(title.replace("\n", " "), available)
    line = prefix + title_text + " " + sgr(glyph_style, right.strip())
    return _agent_box_line(line, width, theme)


def _agent_confirmation_lines(state: Tui2State, width: int, theme: AnsiTheme) -> list[str]:
    view = state.agent_view
    confirmation = view.pending_confirmation or ""
    action, _, thread_id = confirmation.partition(":")
    lang = _resolve_language(state.language)
    selected = view.selected_row()
    target = short_thread(thread_id)
    if selected is not None and selected.thread_id == thread_id:
        target = selected.worktree_branch if action == "delete_worktree" else short_thread(selected.thread_id)
    if action == "delete_worktree":
        return [
            _agent_box_line(
                sgr(theme.error, tr(lang, "agent_view_delete_worktree_status").format(target=target)),
                width,
                theme,
            ),
            _agent_box_line(sgr(theme.warning, tr(lang, "agent_view_delete_worktree_detail")), width, theme),
        ]
    if action in {"hide_thread", "delete_thread"}:
        return [
            _agent_box_line(
                sgr(theme.warning, tr(lang, "agent_view_hide_thread_status").format(target=target)),
                width,
                theme,
            ),
            _agent_box_line(sgr(theme.muted, tr(lang, "agent_view_delete_thread_detail")), width, theme),
        ]
    status = view.status_message or confirmation
    return [_agent_box_line(sgr(theme.warning, status), width, theme)]


def _agent_peek_lines(state: Tui2State, width: int, theme: AnsiTheme) -> list[str]:
    view = state.agent_view
    lang = _resolve_language(state.language)
    selected = view.selected_row()
    lines: list[str] = []
    if view.pending_confirmation:
        return _agent_confirmation_lines(state, width, theme)
    if view.status_message:
        lines.append(_agent_box_line(sgr(theme.muted, view.status_message), width, theme))
    if view.dispatch_level:
        model_label = tr(lang, "agent_view_dispatch_model").format(level=view.dispatch_level)
        lines.append(_agent_box_line(sgr(theme.muted, model_label), width, theme))
    if selected is None:
        return lines or [_agent_box_line("", width, theme)]
    if not view.peek_expanded:
        lines.append(_agent_box_line(sgr(theme.dim, tr(lang, "agent_view_peek_collapsed")), width, theme))
        return lines
    summary = selected.summary.strip().replace("\n", " ")
    if not summary:
        summary = selected.worktree_path or tr(lang, "agent_view_no_transcript")
    label = sgr(theme.muted, tr(lang, "agent_view_peek_prefix")) + truncate_visible(summary, max(1, width - 10))
    lines.append(_agent_box_line(label, width, theme))
    return lines


def _render_agent_composer_with_cursor(
    state: Tui2State,
    width: int,
    theme: AnsiTheme,
    *,
    max_input_rows: int = 3,
) -> tuple[list[str], int, int]:
    view = state.agent_view
    lang = _resolve_language(state.language)
    inner = max(8, width - 4)
    body_width = max(1, inner - 2)
    text = view.composer
    cursor = len(text) if view.composer_cursor is None else max(0, min(view.composer_cursor, len(text)))
    if not text:
        hint_key = "agent_view_reply_placeholder" if view.input_target == "reply" else "agent_view_dispatch_placeholder"
        hint = sgr(theme.muted, tr(lang, hint_key))
        body = sgr(theme.accent, "> ") + hint
        pad = " " * max(0, inner - visible_len(body))
        return [_agent_box_line(body + pad, width, theme)], 0, 4

    wrapped: list[str] = []
    cursor_row = 0
    cursor_col = 0
    consumed = 0
    for raw in text.split("\n"):
        start = consumed
        end = start + len(raw)
        parts = wrap_plain(raw, body_width) or [""]
        if start <= cursor <= end:
            remaining = cursor - start
            for part_index, part in enumerate(parts):
                if remaining <= len(part) or part_index == len(parts) - 1:
                    cursor_row = len(wrapped) + part_index
                    cursor_col = visible_len(part[:remaining])
                    break
                remaining -= len(part)
        wrapped.extend(parts)
        consumed = end + 1
    visible_count = max(1, min(max_input_rows, len(wrapped)))
    hidden = 0 if cursor_row < visible_count else cursor_row - visible_count + 1
    visible_rows = wrapped[hidden : hidden + visible_count]
    rendered: list[str] = []
    for idx, body in enumerate(visible_rows):
        absolute = hidden + idx
        prefix = sgr(theme.accent, "> ") if absolute == 0 else "  "
        line = prefix + body
        rendered.append(_agent_box_line(line, width, theme))
    return rendered, max(0, cursor_row - hidden), 4 + cursor_col


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
    image_token_numbers: set[int] | None = None,
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
    image_token_numbers = image_token_numbers or set()

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
        body = _style_composer_image_tokens(body, image_token_numbers, theme)
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


def render_pager_with_cursor(
    state: Tui2State,
    width: int,
    theme: AnsiTheme = DEFAULT_THEME,
    *,
    max_height: int,
) -> tuple[list[str], int, int]:
    """Render a read-only pager for /show <run_id> output.

    The pager occupies the full live region.  The top border and footer are
    fixed; the middle content scrolls with ``state.pager_scroll``.
    """

    width = max(20, width)
    max_height = max(5, max_height)
    # Borders + title row + footer row + at least one content row.
    content_height = max(1, max_height - 3)

    lines: list[str] = []
    title = state.pager_title or "run detail"
    header_inner = max(0, width - 2)
    header_text = truncate_visible(f"─ {title} ", header_inner)
    header_fill = "─" * max(0, header_inner - visible_len(header_text))
    lines.append(sgr(theme.border_accent, "╭" + header_text + header_fill + "╮"))

    total = max(len(state.pager_lines), state.pager_total_lines)
    scroll = max(0, min(state.pager_scroll, max(0, total - content_height)))
    visible_lines = state.pager_lines[scroll : scroll + content_height]
    # Pad content to keep the footer in a predictable place.
    while len(visible_lines) < content_height:
        visible_lines.append("")

    inner = max(0, width - 4)  # space between "│ " and " │"
    for raw in visible_lines:
        # Preserve ANSI in raw when it fits; truncate_visible strips ANSI only
        # when the line overflows.
        text = truncate_visible(raw, inner)
        pad = " " * max(0, inner - visible_len(text))
        lines.append(sgr(theme.border_accent, "│ ") + text + pad + sgr(theme.border_accent, " │"))

    footer_text = f"{scroll + len(visible_lines)}/{total} · ↑↓/PgUpPgDn scroll · Ctrl+C=code · Ctrl+O=output · q=close"
    footer_plain = truncate_visible(footer_text, inner)
    footer_pad = " " * max(0, inner - visible_len(footer_plain))
    lines.append(
        sgr(theme.border_accent, "│ ")
        + sgr(theme.muted, footer_plain)
        + footer_pad
        + sgr(theme.border_accent, " │")
    )
    lines.append(sgr(theme.border_accent, "╰" + "─" * (width - 2) + "╯"))

    # Cursor is placed on the footer; callers that paint this frame will hide
    # the real cursor inside the border characters.
    cursor_row = len(lines) - 2
    cursor_col = 2
    return [truncate_visible(line, width) for line in lines], cursor_row, cursor_col




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

    if state.pager_open:
        return render_pager_with_cursor(
            state,
            width,
            theme,
            max_height=max_height or 30,
        )

    if state.mode == "agent_view":
        return render_agent_view_with_cursor(
            state,
            width,
            spinner_frame,
            theme,
            max_height=max_height,
        )

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
        image_token_numbers=state.image_token_numbers,
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
