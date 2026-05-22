# uv-agent Project Rules

This repository builds `uv-agent`, an experimental coding agent with a Textual TUI. This file records the repository rules an agent should follow while editing the project.

## Project Shape

- `src/uv_agent/`: host application, configuration, model clients, session store, Python runner, project rules, skills/MCP discovery, and TUI.
- `src/uv_agent_runtime/`: helper package installed into the project shared `scriptenv` uv project so scripts can access file helpers, dependency helpers, subprocess helpers, structured events, image attachment, subagent launch helpers, and MCP clients.
- `tests/`: pytest coverage for runner, runtime, model clients, project rules, sessions, config, and Textual UI behavior.

## Hard Boundaries

- The agent has exactly one external action surface: `run_python`.
- `run_python` executes Python through the managed runner. Do not add direct shell, filesystem, browser, network, or MCP model tools.
- Python scripts may call `subprocess`; that capability must stay inside the Python runner boundary.
- Managed scripts run through `uv run --project <scriptenv> --directory <active-cwd> python <run_id>.py`; the uv project environment and active working directory are separate concepts.
- Managed scripts add third-party dependencies to the shared `run_python` uv environment from inside Python, typically with `add_dependency("package-name")`. Do not add a separate dependency argument to the tool API.
- `uv_agent_runtime` must work as a package dependency for managed scripts; scripts must not rely on the repository checkout, current `.venv`, or implicit import paths.
- MCP and skills are progressively disclosed context. MCP calls happen through Python runtime helpers, not direct model tool calls.

## Prompt And Context

- Keep the stable system prompt concise and structured with explicit XML-style sections and closing tags.
- Include stable host metadata and detected user language in the system prompt.
- Keep AGENTS rules, runtime environment details, skills summaries, and MCP declarations out of the stable prompt. Append them as dynamic workspace context only when first seen, changed, removed, or after compaction.
- Runtime context must include the `run_python` environment directory, its `pyproject.toml`, and direct dependencies from that file only; do not expose transitive dependencies from `uv.lock` as context.
- Within an epoch, keep context update prefix text stable and keep the context section order stable. After compaction starts a new epoch, resend the updated system context for that epoch with the same stable prefix and sequence.
- If dynamic context is removed, the next update must explicitly tell the agent not to rely on older appended context.
- Compression must use the latest system prompt, model config, runner config, and dynamic workspace context after a thread resumes.

## TUI Rules

- TUI uses Textual and should remain a Codex-style single transcript with a bottom composer.
- Composer is multi-line: Enter inserts a newline; Ctrl+Enter or Ctrl+J sends.
- Typing `/` from an empty composer opens the full-screen command picker. Editing or deleting an existing slash command must not reopen it.
- Full-screen pickers must support keyboard and mouse: type to filter, arrows/PageUp/PageDown to move, Enter to select, Esc to close.
- TUI displays model reasoning, tool starts/results, compaction, image attachment, and errors as compact transcript events.
- TUI should not implement model protocol, runner execution, JSONL persistence, compression, or configuration rules directly; consume events from `AgentEngine` and formatting helpers instead.

## Config And State

- User config lives at `~/.uv-agent/config.json`; project overrides may live in `.uv-agent/config.json`.
- Project runtime state lives under user-level `~/.uv-agent/projects/<project-id>/` by default.
- `.uv-agent/` in this repository is ignored local state. Do not commit debug screenshots, local config, scripts, runs, or thread state.
- Never commit API keys, tokens, provider secrets, or redacted copies that still reveal secret material.

## Development

- Use `uv` for project commands, especially `uv run pytest`.
- Follow normal Python `src/` layout conventions. Keep library code importable without starting the TUI.
- Use typed dataclasses or structured dictionaries for persisted and cross-module data shapes when practical.
- Prefer focused tests with each behavior change. Update existing tests when changing prompt structure, runner semantics, context management, or TUI keyboard behavior.
- Keep comments sparse and useful. Use comments for non-obvious compatibility, caching, or protocol decisions; do not narrate ordinary control flow.

## Verification

- Run `uv run pytest` before committing meaningful behavior changes.
- For TUI interaction changes, add or update Textual `run_test` coverage. Manual screenshots can be exported with `App.export_screenshot()` into `.uv-agent/screenshots/`.
- Before committing, check `git status --short` and ensure no secrets or ignored local artifacts are staged.
- Once `uv run pytest` passes for a meaningful change, proactively create a focused git commit with a clear message describing the change; do not wait to be asked. Do not push unless the user explicitly requests it.
