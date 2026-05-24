# Goal Mode Implementation Checklist

## Decisions

- Goal mode is a per-thread persistent mode, not a per-turn dynamic summary.
- Enabling goal mode creates or preserves three internal files under project state.
- Disabling goal mode preserves files and only changes thread mode state.
- The model-visible goal block is emitted once per context epoch while enabled, plus once after an explicit disable transition.
- If goal mode is disabled, a new compaction epoch must not re-emit the enabled block.
- The block must keep a stable prefix/order and should only include mode rules and file locations, not file contents.
- Compression mechanics remain unchanged; goal files complement compaction.

## Implementation Tasks

- [x] Add a goal-state module for file paths, templates, state reads/writes, and notice rendering.
- [x] Persist goal mode changes in thread events/metadata.
- [x] Inject goal mode notices before user messages with once-per-epoch behavior.
- [x] Treat `<goal_mode>` as pre-user context so compaction does not retain it as normal history.
- [x] Add TUI command/panel entry for Goal.
- [x] Add Goal panel actions: enable, disable, view files, reset files.
- [x] Gate reset to disabled state only.
- [x] Gate disable to completed/idle threads only.
- [x] Update i18n and command descriptions.
- [x] Add tests for goal state, context notices, compaction epoch behavior, and TUI panel actions.
- [x] Run focused tests, then full test suite if practical.
