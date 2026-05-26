from __future__ import annotations

import re
import shutil
import unicodedata
from collections.abc import Iterable

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)")


def terminal_size(default: tuple[int, int] = (100, 30)) -> tuple[int, int]:
    """Return ``(columns, rows)`` with a deterministic fallback for tests."""

    size = shutil.get_terminal_size(default)
    return max(20, size.columns), max(10, size.lines)


def strip_ansi(text: str) -> str:
    return OSC_RE.sub("", ANSI_RE.sub("", text))


def char_width(char: str) -> int:
    """Return a practical terminal cell width for one Unicode character."""

    if not char:
        return 0
    codepoint = ord(char)
    if codepoint == 0:
        return 0
    if codepoint < 32 or 0x7F <= codepoint < 0xA0:
        return 0
    category = unicodedata.category(char)
    if category in {"Mn", "Me", "Cf"}:
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def display_width(text: str) -> int:
    """Return terminal cell width for plain text."""

    return sum(char_width(char) for char in text)


def visible_len(text: str) -> int:
    """Best-effort terminal cell width after removing ANSI escapes."""

    return display_width(strip_ansi(text))


def truncate_visible(text: str, width: int, suffix: str = "…") -> str:
    """Truncate a plain/ANSI line by terminal cell width.

    ANSI styling is intentionally dropped when truncation is required; this keeps
    the function simple and prevents partial escape sequences. Non-truncated
    lines are returned unchanged.
    """

    if width <= 0:
        return ""
    plain = strip_ansi(text)
    if display_width(plain) <= width:
        return text
    suffix_width = display_width(suffix)
    keep_width = max(0, width - suffix_width)
    cells = 0
    chars: list[str] = []
    for char in plain:
        next_width = cells + char_width(char)
        if next_width > keep_width:
            break
        chars.append(char)
        cells = next_width
    return "".join(chars).rstrip() + suffix


def wrap_plain(text: str, width: int, *, subsequent_indent: str = "") -> list[str]:
    """Wrap plain text by terminal cell width while preserving blank lines.

    ``textwrap.wrap`` counts Python code points, which means full-width CJK
    input can overflow the composer and get clipped by the renderer's final
    safety truncation.  Counting display cells here keeps the produced rows
    within the same width model used by ``visible_len`` and cursor placement.
    """

    width = max(1, width)
    lines: list[str] = []
    indent_width = display_width(subsequent_indent)
    # A too-wide indent would otherwise leave no room for content and could
    # loop forever.  This path is not used by tui2 today, but keeping it safe
    # makes the helper predictable for future callers.
    safe_indent = subsequent_indent if indent_width < width else ""
    safe_indent_width = indent_width if safe_indent else 0
    for raw_line in text.splitlines() or [""]:
        if not raw_line:
            lines.append("")
            continue
        current: list[str] = []
        current_width = 0
        for char in raw_line:
            char_cells = char_width(char)
            if current and current_width + char_cells > width:
                lines.append("".join(current))
                current = [safe_indent] if safe_indent else []
                current_width = safe_indent_width
            current.append(char)
            current_width += char_cells
        lines.append("".join(current))
    return lines


def pad_right(text: str, width: int) -> str:
    """Pad a line to visible *width* without adding background colour."""

    return text + " " * max(0, width - visible_len(text))


def clamp_lines(lines: Iterable[str], max_lines: int, *, more_label: str = "…") -> list[str]:
    materialized = list(lines)
    if max_lines <= 0 or len(materialized) <= max_lines:
        return materialized
    omitted = len(materialized) - max_lines + 1
    return [*materialized[: max_lines - 1], f"{more_label} +{omitted} lines"]
