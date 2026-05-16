# Runtime Helper Surface Proposal

## Goal

Reduce accidental misuse of runtime helpers by exposing fewer, higher-value APIs in the system prompt.

## Proposal

Keep the prompt focused on helpers that provide environment-specific capabilities:

- `apply_patch`
- `run_command` / `check_command`
- `look_at`
- `ask`
- `saved_scripts`
- `thread_digest` / `list_thread_digests`
- MCP helpers
- structured event helpers such as `emit_event`, `emit_progress`, `emit_result`

De-emphasize or remove generic filesystem helpers from the prompt, especially APIs that are easy to confuse with richer utility libraries, such as `list_files`.

For ordinary file traversal and inspection, instruct agents to prefer Python standard library APIs, for example `pathlib`, `os`, and `json`, unless an exact helper signature is known.

## Suggested prompt wording

> Use Python standard library for ordinary filesystem traversal and data handling. Runtime helpers are primarily for managed environment features. Do not assume helper signatures; if a helper signature is not explicitly documented, inspect its implementation or use the standard library.

## Rationale

This keeps important managed capabilities visible while reducing mistakes caused by guessing undocumented helper parameters.
