"""Centralized color tokens for the TUI.

A small palette is intentional: every surface, border, and accent in the
application should resolve to one of these constants. This keeps the visual
language consistent and makes whole-theme changes a single-file edit.
"""

from __future__ import annotations

# Surfaces (three-step hierarchy).
BG_CANVAS = "#0b0f14"     # main background, transcript, composer
BG_SURFACE = "#121821"    # raised surfaces, modal shell, option list rows
BG_OVERLAY = "#1a2230"    # popovers, hover, sticky buttons

# Foreground text.
TEXT = "#d8dee9"
TEXT_STRONG = "#dce7f3"
TEXT_MUTED = "#7b8796"

# Borders and accents.
BORDER = "#2a3646"
BORDER_FOCUS = "#3f9bc9"
ACCENT = "#7dd3fc"
ACCENT_DIM = "#3a4a60"

# Modal scrim (rgba so the underlying transcript shows through).
SCRIM = "rgba(5, 7, 10, 0.78)"

# Role colors for the TranscriptCell left "stripe".
ROLE_USER = "#7dd3fc"        # cyan
ROLE_ASSISTANT = "#a7f3d0"   # mint
ROLE_REASONING = "#c4b5fd"   # violet
ROLE_TOOL = "#fbbf24"        # amber
ROLE_EVENT = "#64748b"       # slate
ROLE_ERROR = "#f87171"       # red
ROLE_PROCESS = "#94a3b8"     # gray

# Error fill (the only role that keeps a background tint).
BG_ERROR = "#241316"
TEXT_ERROR = "#ffb4b4"

# Selection highlight (used by TranscriptCell.SELECTION_STYLE).
SELECTION_BG = ACCENT
SELECTION_FG = "#061018"
