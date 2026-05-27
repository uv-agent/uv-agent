# TUI And Slash Commands

`uv-agent` opens a Textual TUI by default. The interface is a single transcript
with a bottom composer, compact status footer, and full-screen panels for focused
views such as threads, config, model levels, and MCP declarations.

```powershell
uvx uv-agent@latest
```

From a local checkout:

```powershell
uv run uv-agent
```

## Composer

The composer is multi-line.

| Key | Action |
| --- | --- |
| `Enter` | Insert a newline when composer is focused, or focus the composer. |
| `Ctrl+Enter` or `Ctrl+J` | Send the current composer text. |
| `Esc` | Clear the composer or close the active panel. |
| `Tab` | Toggle composer height. |
| `/` from an empty composer | Open the command palette. |
| `@` | Open file mention search. |
| `@@` | Open thread mention search. |
| `@mcp:` followed by text | Insert an MCP server mention from configured declarations. |
| `@skill:` followed by text | Insert a skill mention from discovered skills. |

Editing text that already starts with `/` does not reopen the command palette.
Use `Ctrl+O` when you want to open the command palette explicitly.

## Global Shortcuts

| Key | Action |
| --- | --- |
| `F1` | Open help. |
| `Ctrl+O` | Open command palette. |
| `Ctrl+A` from an empty composer | Open Agent View. |
| `Ctrl+S` | Open runtime status. |
| `Ctrl+G` | Collapse or expand the most recent process fold. |
| `Ctrl+D` | Toggle Python/tool detail display. |
| `F2` | Attach an image from the clipboard. |
| `F3` | Preview pending images. |
| `Ctrl+C` | Interrupt a running turn; press twice while idle to quit. |

Full-screen panels support keyboard and mouse interaction. Type to filter when a
filter box is focused, use arrows or PageUp/PageDown to move, Enter to select,
and Esc to close.

## Slash Commands

| Command | Action |
| --- | --- |
| `/threads` | Open recent threads and resume history. |
| `/status` | Show runtime status, model level, context usage, config paths, and rules. |
| `/config` | Edit user-facing settings and inspect redacted config. |
| `/models` | Show configured models. This panel is read-only. |
| `/level [name]` | Switch the active model level for the current thread. |
| `/mcp` | Show MCP declarations and insert MCP mentions. |
| `/skills` | Show discovered skills and insert skill mentions. |
| `/agents` | Open Agent View for background worktree sessions. |
| `/clear` | Clear the current transcript view and active thread selection. |
| `/quit` | Quit after confirmation. |
| `/help` or `?` | Show local commands and shortcuts. |

## Agent View

Agent View (`/agents`, or `Ctrl+A` from an empty composer) is a full-screen
dashboard for running multiple background agent sessions. New tasks are launched
in isolated Git worktrees with generated `agent-*` branch names. Existing rows
can be attached back into the main transcript when you want to inspect or
continue them directly.

Ordinary threads are not listed automatically. To add the current normal thread
to Agent View, run `/agents` from inside that thread. Tasks dispatched from Agent
View are added to the panel automatically.

Agent View deliberately uses two lightweight modes so navigation keys do not
fight with text editing:

| Mode | Keys |
| --- | --- |
| Normal | `j/k` or arrows move; `PageUp/PageDown` jump; `Enter` attaches the selected session; `Space` toggles peek; `m` chooses the model level for new tasks; `c` cancels the selected running session; `d` hides a row; `D` deletes its worktree and branch after confirmation; `?` opens Agent View help; `Esc` or `Ctrl+A` returns to the transcript. |
| Input | Entered with `i` for a new background task or `r` to reply to the selected session. `Enter` sends, `Ctrl+Enter`/`Ctrl+J` inserts a newline, and `Esc` returns to normal mode. |
| Model | Entered with `m` from normal mode. `j/k` or arrows move, `Enter` selects the level for future Agent View tasks, and `Esc` or `m` returns to normal mode. |

## Config Panel

`/config` can write only the settings that are safe to edit from the TUI:

- `runtime.default_level`
- `runtime.compression.enabled`
- `ui.language`
- `ui.completion_notification.enabled`

Provider definitions, model definitions, and level-to-model mappings are edited
in JSON config files. Use `/models` to inspect configured models and `/level` to
switch the current thread level. Switching between levels backed by different
models is allowed, but the transcript records a warning because context
conversion between models is best effort.

Language choices are:

- `auto`
- `en`
- `zh-CN`

See [configuration](configuration.md) for config file locations and schema.

Completion notifications can add a short transcript event for background
threads and play a completion sound. Windows uses the system notification sound
in the Textual TUI; the terminal-native `tui2` uses a short buzzer-like terminal
cue. Other platforms use the terminal bell. The active thread does not add an
extra terminal event when it finishes.

## Transcript Events

The transcript renders model output and runtime activity as compact timeline
items:

- user and assistant messages
- configured provider reasoning fields, shown as expandable reasoning details
- Python run start/result events
- stdout/stderr summaries and truncation markers
- structured runtime events such as `progress`, `result`, `look_at`,
  `subagent.started`, and `subagent.completed`
- compaction and image attachment notices
- readable error cards with hints when available

Detailed Python run output remains available from run/tool details and status
views instead of flooding the transcript by default.

## Images

`F2` attaches an image from the clipboard when image clipboard access is
available. `F3` previews pending images before sending. Images attached from the
TUI are sent with the next user turn, and models with `supports_images: false`
reject image input.

Managed Python scripts can also attach image context with
`uv_agent_runtime.look_at`; see [runtime](runtime.md).
