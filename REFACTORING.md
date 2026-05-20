# Core Refactoring Checklist

Goal: improve core maintainability without changing user-visible behavior.

Principles:
- Keep changes behavior-preserving unless a checklist item explicitly says otherwise.
- Preserve model context prefix order and item sequence.
- Prefer narrow modules with clear ownership over generic service layers.
- Keep the public imports used by tests and downstream code stable.
- Run tests after each meaningful step.

## Scope

- `src/uv_agent/agent.py`
- `src/uv_agent/model_client.py`
- `src/uv_agent/runner/runner.py`
- Closely related tests and docs only when needed to preserve existing behavior.

## Checklist

- [x] Extract stable prompt and runtime-context rendering from `agent.py`.
- [x] Extract compaction helpers from `agent.py`.
- [x] Extract tool result projection/filtering from `agent.py`.
- [x] Reduce duplicated turn/retry model-loop code in `agent.py`.
- [x] Split model provider protocol code out of `model_client.py`.
- [x] Replace hand-built Anthropic HTTP calls with the official SDK.
- [x] Remove the root `model_client.py` compatibility module.
- [x] Split runner stream/event parsing helpers out of `runner.py`.
- [x] Run full test suite and confirm behavior parity.
- [x] Group split agent and model modules under package directories.

## Non-goals

- No protocol or event-format redesign.
- No TUI behavior changes.
- No new model context sections or reordered context sections.
- No broad naming/style churn outside the listed files.
