# Pending Send Queue Refactor Plan

## Background

`docs/pending-send-queue.md` defines the expected behavior for sending while a thread is already running. The current TUI already has an in-memory `ThreadRunState.queue` and starts the next queued turn after the current one finishes, but that support is mostly hidden in the footer/transcript. Composer image attachments are also tracked globally, which can make pasted images follow the user across thread switches.

## Goal

Complete the pending-send queue UX with the smallest durable change set:

- Treat each send as a queue item containing prompt text plus the images attached at send time.
- Show an explicit composer overlay button for the active thread's queued messages.
- Provide a queue list panel that can inspect, edit, delete, and reorder queued items before they start sending.
- Keep composer-pending images scoped per thread so switching threads cannot accidentally move images into another thread's next send.
- Keep queued turns in memory only; no persistence across app restarts.

## Constraints

- Preserve the existing Textual TUI structure: transcript, composer, and full-screen modal panels.
- Reuse the current `ThreadRunState.queue` execution path instead of adding a second scheduler.
- Do not add model, runner, JSONL, or persistence changes.
- Prefer focused state helpers and tests over broad rewrites.
- Remove the old footer `qN` queue hint after the explicit overlay button lands.

## Checklist

- [x] Create this plan file at the repository root.
- [ ] Add stable queue item identity for edit/delete/reorder operations.
- [ ] Add active-thread queue helpers in the TUI app.
- [ ] Add the pending queued-message overlay button and keep layout compatible with the existing image and back-to-bottom buttons.
- [ ] Scope composer pending images by thread.
- [ ] Add a pending-send list panel with edit, delete, move up, and move down controls.
- [ ] Wire panel callbacks to queue mutation helpers with out-of-queue safeguards.
- [ ] Update i18n text for English and Chinese.
- [ ] Update focused TUI tests for queue overlay, edit/delete/reorder, out-of-queue saves, and per-thread images.
- [ ] Run targeted tests, then the full test suite if feasible.
- [ ] Commit logical steps on the feature branch.
