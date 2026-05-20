# TUI Refactor Checklist

Goal: improve TUI maintainability while preserving behavior.

Constraints:
- Keep changes scoped to TUI structure and direct imports.
- Prefer moving existing code over rewriting behavior.
- Avoid new abstractions unless they remove clear coupling or duplication.
- Keep tests passing after each meaningful step.

Plan:
- [x] Move shared TUI dataclasses/state helpers out of `app.py`.
- [x] Move reusable widgets out of `app.py`.
- [x] Move fullscreen panel classes out of `app.py`.
- [x] Isolate mention scanning/picker support from the app class.
- [x] Isolate config panel/write helpers from the app class.
- [x] Isolate image attachment/preview app helpers where it reduces app state.
- [x] Split turn event handling enough to reduce the `_run_turn` event chain.
- [x] Run the full test suite.
- [x] Commit the refactor in small, reviewable steps.
