"""Pre-formatted Textual CSS strings for the TUI.

All visual rules live here so individual widget classes can stay focused on
behavior. Color values come from :mod:`uv_agent.tui.theme`; never hard-code
hex values in this module.

Design rules applied across panels:
  * Exactly one border per modal — the outer ``#panel-shell``. Inner
    containers (filter input, option list, body scroll, image scroll) draw no
    border and rely on a single blank-line gap for separation.
  * Vertical chrome is kept to a single header row plus one optional subtitle
    row; closing hints live in the dedicated ``#panel-footer`` row only.
  * Scroll containers expose both axes (``overflow-x: auto``) so long lines
    can scroll horizontally instead of being silently clipped.
"""

from __future__ import annotations

from . import theme as t


# ---------------------------------------------------------------------------
# Full-screen panels (pickers, body panels, tool details, image previews).
# ---------------------------------------------------------------------------

FULLSCREEN_PANEL_CSS = f"""
/* Opaque scrim: an rgba scrim forces the terminal to re-composite the
 * transcript underneath on every panel paint, which on sixel/TGP terminals
 * can show up as image flicker. A solid color avoids the per-frame blend. */
FullscreenPanel,
ToolDetailsPanel,
ImagePreviewPanel,
PendingImagePreviewPanel {{
    align: center middle;
    background: {t.BG_CANVAS};
}}

#panel-shell {{
    width: 95%;
    height: 92%;
    max-width: 140;
    border: round {t.BORDER};
    background: {t.BG_SURFACE};
    padding: 0 1;
}}

/* Title bar: a single 1-row horizontal that combines the bold title (left,
 * shrink-to-fit) and the muted subtitle/status hint (right, fills remaining
 * width and right-aligns). Saves one row of vertical chrome per panel. */
#panel-titlebar {{
    height: 1;
    padding: 0 1;
}}

#panel-header {{
    width: auto;
    height: 1;
    color: {t.TEXT_STRONG};
    text-style: bold;
    padding: 0;
}}

#panel-subtitle {{
    width: 1fr;
    height: 1;
    color: {t.TEXT_MUTED};
    padding: 0 0 0 2;
    content-align: right middle;
    text-align: right;
}}

#panel-filter {{
    height: 1;
    margin: 1 0 0 0;
    border: none;
    background: {t.BG_CANVAS};
    color: {t.TEXT};
    padding: 0 1;
}}

#panel-content {{
    height: 1fr;
    margin: 1 0 0 0;
    border: none;
    background: {t.BG_SURFACE};
    padding: 0;
    overflow-x: auto;
    scrollbar-size: 1 1;
    scrollbar-background: {t.BG_SURFACE};
    scrollbar-background-hover: {t.BG_SURFACE};
    scrollbar-background-active: {t.BG_SURFACE};
    scrollbar-color: {t.ACCENT_DIM};
    scrollbar-color-hover: {t.BORDER};
    scrollbar-color-active: {t.ACCENT};
    scrollbar-corner-color: {t.BG_SURFACE};
}}

#panel-body {{
    height: 1fr;
    margin: 1 0 0 0;
    border: none;
    background: {t.BG_SURFACE};
    padding: 0 1;
    overflow-x: auto;
    scrollbar-size: 1 1;
    scrollbar-background: {t.BG_SURFACE};
    scrollbar-background-hover: {t.BG_SURFACE};
    scrollbar-background-active: {t.BG_SURFACE};
    scrollbar-color: {t.ACCENT_DIM};
    scrollbar-color-hover: {t.BORDER};
    scrollbar-color-active: {t.ACCENT};
    scrollbar-corner-color: {t.BG_SURFACE};
}}

/* Lives inside ``#image-preview-scroll``: a single status line that scrolls
 * with the image. Keeping the height fixed at 1 prevents reflow-induced
 * flicker when navigating between attachments. */
#image-preview-meta {{
    height: 1;
    margin: 0 0 1 0;
    border: none;
    background: {t.BG_CANVAS};
    padding: 0 1;
    color: {t.TEXT_MUTED};
}}

/* Horizontal overflow is disabled: the image is sized to fit the container
 * width, so a horizontal scrollbar would only ever appear/disappear due to
 * layout jitter. Only the vertical axis scrolls when a tall image overflows. */
#image-preview-scroll {{
    height: 1fr;
    margin: 1 0 0 0;
    border: none;
    background: {t.BG_CANVAS};
    padding: 0;
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-size: 1 1;
    scrollbar-background: {t.BG_CANVAS};
    scrollbar-background-hover: {t.BG_CANVAS};
    scrollbar-background-active: {t.BG_CANVAS};
    scrollbar-color: {t.ACCENT_DIM};
    scrollbar-color-hover: {t.BORDER};
    scrollbar-color-active: {t.ACCENT};
    scrollbar-corner-color: {t.BG_CANVAS};
}}

/* The image uses ``width: 1fr; height: auto`` so textual-image scales the
 * image to fill the available width while keeping its aspect ratio; when the
 * resulting natural height exceeds the container, ``#image-preview-scroll``
 * shows a vertical scrollbar instead of squashing the image. */
#image-preview {{
    width: 1fr;
    height: auto;
}}

#pending-image-delete {{
    width: auto;
    height: 1;
    margin: 0 0 0 1;
}}

#panel-footer {{
    height: 1;
    margin: 1 0 0 0;
    color: {t.TEXT_MUTED};
    padding: 0 1;
}}

OptionList {{
    height: 1fr;
    border: none;
    background: {t.BG_SURFACE};
    padding: 0 1;
    scrollbar-size: 1 1;
    scrollbar-background: {t.BG_SURFACE};
    scrollbar-background-hover: {t.BG_SURFACE};
    scrollbar-background-active: {t.BG_SURFACE};
    scrollbar-color: {t.ACCENT_DIM};
    scrollbar-color-hover: {t.BORDER};
    scrollbar-color-active: {t.ACCENT};
    scrollbar-corner-color: {t.BG_SURFACE};
}}
"""


# ---------------------------------------------------------------------------
# Empty-state placeholder shown when the transcript is empty.
# ---------------------------------------------------------------------------

EMPTY_STATE_CSS = f"""
EmptyState {{
    width: 100%;
    height: 100%;
    content-align: center middle;
    color: {t.TEXT_MUTED};
}}

EmptyState.hidden {{
    display: none;
}}
"""


# ---------------------------------------------------------------------------
# Transcript cells. Role is signaled by a one-cell left "stripe" drawn with
# ``border-left: outer``; the cell background stays flat so the conversation
# reads as a single canvas instead of a checkerboard of dark fills.
#
# Mouse offset preservation: ``border-left: outer`` adds 1 cell of leading
# chrome, so we drop the original 1 cell of left padding (``padding: 0 1`` ->
# ``padding: 0 1 0 0``). Total leading chrome stays at 1 cell, keeping mouse
# offsets identical to the previous layout (mouse-drag tests assert against
# fixed column offsets relative to the cell origin).
# ---------------------------------------------------------------------------

TRANSCRIPT_CELL_CSS = f"""
TranscriptCell {{
    width: 100%;
    margin: 0 0 1 0;
    padding: 0 1 0 0;
    border-left: outer {t.BG_CANVAS};
    background: {t.BG_CANVAS};
    color: {t.TEXT};
}}

TranscriptCell.user {{
    border-left: outer {t.ROLE_USER};
    color: {t.TEXT_STRONG};
}}

TranscriptCell.assistant {{
    border-left: outer {t.ROLE_ASSISTANT};
    color: {t.TEXT_STRONG};
}}

TranscriptCell.event {{
    border-left: outer {t.ROLE_EVENT};
    color: {t.TEXT_MUTED};
}}

TranscriptCell.reasoning {{
    border-left: outer {t.ROLE_REASONING};
    color: {t.ROLE_REASONING};
    text-style: italic;
}}

TranscriptCell.process_fold {{
    border-left: outer {t.ROLE_PROCESS};
    color: {t.TEXT_MUTED};
}}

TranscriptCell.process_fold_hidden {{
    display: none;
}}

TranscriptCell.tool_pending {{
    border-left: outer {t.ROLE_TOOL};
    color: {t.ROLE_TOOL};
}}

TranscriptCell.error {{
    border-left: outer {t.ROLE_ERROR};
    background: {t.BG_ERROR};
    color: {t.TEXT_ERROR};
}}
"""


# ---------------------------------------------------------------------------
# Main application chrome: transcript scroll, composer, overlay buttons.
# ---------------------------------------------------------------------------

MAIN_APP_CSS = f"""
Screen {{
    layout: horizontal;
    background: {t.BG_CANVAS};
    color: {t.TEXT};
}}

Screen > .screen--selection {{
    background: {t.ACCENT};
    color: #061018;
}}

ToastRack {{
    dock: top;
    align-horizontal: right;
}}

#main-column {{
    width: 1fr;
    min-width: 0;
    height: 100%;
    background: {t.BG_CANVAS};
}}

#transcript {{
    height: 1fr;
    min-height: 6;
    /* No right padding so the vertical scrollbar can sit flush against the
     * screen's right column. Right-side breathing room for transcript text is
     * provided by ``TranscriptCell``'s right padding instead. */
    padding: 1 0 0 1;
    background: {t.BG_CANVAS};
    overflow-x: auto;
    scrollbar-size: 1 1;
    scrollbar-background: {t.BG_CANVAS};
    scrollbar-background-hover: {t.BG_CANVAS};
    scrollbar-background-active: {t.BG_CANVAS};
    scrollbar-color: {t.ACCENT_DIM};
    scrollbar-color-hover: {t.BORDER};
    scrollbar-color-active: {t.ACCENT};
    scrollbar-corner-color: {t.BG_CANVAS};
}}

#bottom-pane {{
    height: auto;
    max-height: 9;
    padding: 0 1 0 1;
    background: {t.BG_CANVAS};
}}

#composer-shell {{
    height: auto;
    background: {t.BG_CANVAS};
}}

#composer-shell.busy {{
    background: {t.BG_CANVAS};
}}

#pending-images-btn,
#scroll-to-bottom-btn {{
    position: absolute;
    overlay: screen;
    layer: overlay;
    width: auto;
    height: 1;
    color: {t.ACCENT};
    background: {t.BG_OVERLAY};
    padding: 0 2;
    text-style: bold;
}}

#pending-images-btn {{
    color: #c4b5fd;
}}

#pending-images-btn.hidden,
#scroll-to-bottom-btn.hidden {{
    display: none;
}}

#pending-images-btn:hover,
#scroll-to-bottom-btn:hover {{
    background: {t.BORDER};
    color: #ffffff;
}}

#composer {{
    width: 1fr;
    height: 5;
    min-height: 5;
    margin: 0;
    border: round {t.BORDER};
    padding: 0 1;
    background: {t.BG_CANVAS};
    color: {t.TEXT_STRONG};
}}

#composer:focus {{
    border: round {t.BORDER_FOCUS};
}}

#composer-footer {{
    height: 1;
    color: {t.TEXT_MUTED};
    padding: 0 1;
    background: {t.BG_CANVAS};
}}
"""
