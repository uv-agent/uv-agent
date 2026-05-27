from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnsiTheme:
    """Semantic ANSI palette used by tui2 components.

    Values are SGR fragments without the leading escape wrapper.  The raw ANSI
    TUI deliberately avoids background colours in normal transcript output so
    transparent terminals and native selection keep working well.
    """

    reset: str = "0"
    bold: str = "1"
    dim: str = "2"
    italic: str = "3"
    accent: str = "38;5;117"
    muted: str = "38;5;245"
    user: str = "38;5;80"
    assistant: str = "38;5;121"
    reasoning: str = "38;5;141"
    success: str = "38;5;114"
    warning: str = "38;5;221"
    goal: str = "1;38;5;202"
    image_token: str = "1;38;5;117"
    error: str = "38;5;203"
    border: str = "38;5;240"
    border_faint: str = "38;5;237"
    border_accent: str = "38;5;75"
    tool_title: str = "38;5;153"
    tool_output: str = "38;5;252"
    status: str = "38;5;250"
    command_palette: str = "38;5;252"
    command_palette_selected: str = "1;38;5;117"
    command_palette_border: str = "38;5;60"
    spinner_frames: tuple[str, ...] = field(
        default=("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    )


DEFAULT_THEME = AnsiTheme()


def sgr(code: str, text: str) -> str:
    """Wrap *text* in a single SGR code."""

    if not text:
        return text
    return f"[{code}m{text}[0m"


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text* for width and tests."""

    import re

    return re.sub(r"\[[0-?]*[ -/]*[@-~]", "", text)
