# Rich transcript renderables refactor plan

## Background

The TUI transcript and several related panels currently build Rich/Textual markup
strings by concatenating static labels with external data, then pass the result to
`Static(markup=True)`. External data has to be escaped before concatenation. That
escape layer leaks into rendered text for common bracket-heavy output such as
`[100%]`, `list[int]`, JSON arrays, and Windows paths, producing visible
backslashes in the transcript.

The old plan in `docs/pending-richify-transcript.md` identified the right root
cause: user, model, tool, filesystem, and JSON data should not travel through a
markup parser at all. This document is the working checklist for completing that
refactor in one pass.

## Goal

Render transcript and panel content from Rich/Textual objects (`Text`, Markdown,
`Rule`, and simple renderable groups) instead of markup strings whenever external
data is involved. Keep existing colors, glyphs, folding, details navigation,
selection, and copy behavior.

## Scope

- Transcript cells, expandable cells, folded process cells, image attachment
  cells, and load-history cells.
- Tool call summaries/details, tool result timelines/details, structured runtime
  events, JSON previews, and Python syntax highlighting.
- Full-screen panels, picker options, tool details, config/model panels, image
  preview metadata, error messages, queued/background events, and other TUI text
  that combines labels with external data.
- Tests that asserted raw markup strings should move to rendered/plain-text
  assertions, with style checks only where style is the behavior under test.

## Non-goals

- No redesign of the transcript UI.
- No new custom markup language or placeholder protocol.
- No broad cleanup outside Rich/Textual rendering paths.
- No dependency changes.

## Implementation checklist

- [x] Create a feature branch for the refactor.
- [x] Add small formatting helpers for composing `Text` from static styled labels
      and plain external data.
- [x] Convert formatting helpers from `*_markup` string construction to Rich
      renderables while preserving public names where it keeps the diff smaller.
- [x] Update transcript widgets to accept renderables without `markup=True` and
      keep plain copy text explicit or derivable.
- [x] Update app transcript append/replace paths and all TUI event builders.
- [x] Update full-screen panels, picker options, tool details, config panels, and
      image metadata to use renderables.
- [x] Replace error markup helpers with renderable helpers.
- [x] Update tests for renderable/plain-text assertions and add bracket
      regression coverage.
- [ ] Run focused TUI tests, then the full test suite.
      - Focused TUI suite: `uv run pytest tests/test_tui.py -q` passes after renderable migration.
- [ ] Commit the refactor in focused steps with clear English messages.

