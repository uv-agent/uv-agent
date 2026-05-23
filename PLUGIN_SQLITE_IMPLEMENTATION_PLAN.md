# Plugin System And SQLite State Implementation Plan

Planning references (ignored by git on purpose):

- `docs/pending-plugin-system-proposal.md`
- `docs/pending-sqlite-state-store.md`

Execution order:

1. Finish the plugin system first.
2. Then migrate project state from JSONL/JSON files to SQLite.
3. Remove this tracked checklist after both implementations are complete.

## Plugin system checklist

- [ ] Add plugin config and pluggy dependency.
- [ ] Add plugin manager, context, event bus, first-load registry, logging, and private SQLite storage helpers.
- [ ] Extend runtime RPC and runtime lazy exports for dynamic plugin helper resolve/call with `*args`/`**kwargs`.
- [ ] Wire plugin background lifecycle into AgentEngine/TUI/CLI and publish plugin/turn/tool events.
- [ ] Add plugin runtime helper context disclosure.
- [ ] Add plugin tests and docs.

## SQLite state store checklist

- [ ] Add project SQLite schema/connection infrastructure.
- [ ] Migrate ThreadStore events, metadata, history pagination, and locks to SQLite.
- [ ] Migrate runner run records/events to SQLite and update PythonRunResult shape.
- [ ] Migrate runtime thread introspection helpers to SQLite.
- [ ] Update TUI/agent/session/runner tests and docs for event_id-based storage.
- [ ] Run full test suite, commit final state, then delete this checklist.
