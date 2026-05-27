# uv-agent

<img align="right" src="docs/t2.png" alt="uv-agent tui2 screenshot" width="300">

[简体中文](README.zh-CN.md)

`uv-agent` is a Windows-first coding agent with an ANSI-first terminal TUI.
It is designed
around one external action surface: `run_python`. The model writes managed Python
scripts, uv-agent runs them through `uv run`, and the runtime exposes focused
helpers for editing files, running commands, searching code, using MCP, launching
subagents, and attaching images.

That single boundary keeps coding-agent work easier to inspect, replay, interrupt,
and compact during long sessions. The design also makes it easy to port to any
Python and uv environment. The project is still experimental, so public APIs,
config fields, and runtime behavior may change.

## Why uv-agent?

- **Windows-first coding UI.** A terminal-native transcript with a multi-line
  composer, command palette, model/tool timeline, file and thread mentions, image
  attachments, and English or Chinese UI.
- **One action boundary.** No direct shell, filesystem, browser, or MCP model
  tools; external work flows through managed Python runs and one persistent event
  stream.
- **Long-task friendly.** Checkpoint compaction, workspace rules, skills, MCP
  declarations, Goal state, and Worktree state are replayed as structured context
  when needed.
- **Practical coding workflows.** `/goal` adds lightweight per-thread
  checklist/notes for longer tasks; Worktree mode creates an isolated Git branch
  worktree for task-focused changes.
- **Extensible runtime.** Plugins can add `uv_agent_runtime` helpers, subscribe
  to agent events, and submit turns from external systems without adding extra
  model-visible tools.

## Quick Start

Install the required tools:

- **uv** — https://docs.astral.sh/uv/getting-started/installation/
- **ripgrep** — https://github.com/BurntSushi/ripgrep#installation
- **Git** — needed for normal coding workflows and Worktree mode.

Run the latest published package:

```powershell
uvx uv-agent@latest
```

Run from a local checkout:

```powershell
uv run uv-agent
```

Ask a single prompt without opening the TUI:

```powershell
uvx uv-agent@latest ask "Reply with exactly: ok"
```

Resume an existing thread:

```powershell
uvx uv-agent@latest ask --thread thr_xxx "Continue from here"
```

## Model Configuration

uv-agent does not ship a real provider configuration. Configure at least one
provider, model, and level before making model calls.

Config is loaded from `~/.uv-agent/config.json`, then optional project overrides
from `.uv-agent/config.json`. The project-local `.uv-agent/` directory is ignored
by git. Prefer environment variables or ignored local config for API keys.

Supported model API formats:

| `api` value | Format |
| --- | --- |
| `"responses"` | OpenAI Responses API |
| `"chat_completions"` | OpenAI Chat Completions API |
| `"anthropic_messages"` | Anthropic Messages API |

<details>
<summary>Full configuration example</summary>

```json
{
  "providers": {
    "deepseek": {
      "base_url": "https://api.deepseek.com",
      "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      "chat_completions": {
        "path": "/chat/completions"
      },
      "message_passthrough": {
        "assistant": [
          "reasoning_content"
        ]
      },
      "reasoning_display": {
        "assistant_message_fields": [
          "reasoning_content"
        ],
        "stream_delta_fields": [
          "reasoning_content"
        ]
      }
    },
    "minimax": {
      "base_url": "https://api.minimaxi.com",
      "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      "chat_completions": {
        "path": "/v1/chat/completions"
      },
      "anthropic_messages": {
        "path": "/anthropic/v1/messages"
      }
    }
  },
  "models": {
    "deepseek-v4-flash": {
      "provider": "deepseek",
      "model": "deepseek-v4-flash",
      "api": "chat_completions",
      "supports_images": false,
      "context_window_tokens": 1000000,
      "params": {
        "reasoning_effort": "high"
      }
    },
    "deepseek-v4-pro": {
      "provider": "deepseek",
      "model": "deepseek-v4-pro",
      "api": "chat_completions",
      "supports_images": false,
      "context_window_tokens": 1000000,
      "params": {
        "reasoning_effort": "max"
      }
    },
    "MiniMax-M2.7": {
      "provider": "minimax",
      "model": "MiniMax-M2.7-highspeed",
      "api": "anthropic_messages",
      "supports_images": false,
      "context_window_tokens": 204800
    }
  },
  "levels": {
    "deepseek-flash": {
      "model": "deepseek-v4-flash"
    },
    "deepseek-pro": {
      "model": "deepseek-v4-pro"
    },
    "MiniMax-M2.7": {
      "model": "MiniMax-M2.7"
    }
  },
  "runtime": {
    "default_level": "deepseek-flash",
    "ask_default_level": "deepseek-flash",
    "store_provider_response": false,
    "max_agent_rounds": 1000,
    "compression": {
      "enabled": true,
      "model_level": "deepseek-flash",
      "trigger_ratio": 0.9
    },
    "title_generation": {
      "enabled": true,
      "model_level": "deepseek-flash"
    }
  },
  "runner": {
    "default_timeout_s": 7200,
    "max_output_bytes": 1000000,
    "scriptenv_index_url": null
  },
  "pricing": {
    "currency": "RMB",
    "unit": "1M_tokens",
    "models": {
      "deepseek-v4-flash": {
        "input": 1,
        "output": 2,
        "cached_input": 0.02
      },
      "deepseek-v4-pro": {
        "input": 3,
        "output": 6,
        "cached_input": 0.025
      }
    }
  },
  "ui": {
    "completion_notification": {
      "enabled": true
    }
  },
  "plugins": {
    "disabled": [],
    "config": {}
  }
}

```

</details>

Use `/config` in the TUI to switch user-facing settings such as default level,
language, completion notification, and automatic compression. See
[configuration](docs/configuration.md) for every supported option and
[config.example.json](docs/config.example.json) for a detailed standalone example.

## Everyday Workflow

- Type normally and press `Enter` to send. Use `Ctrl+Enter` or `Ctrl+J` when you
  want to insert a newline in the composer.
- Type `/` from an empty composer to open the tui2 command palette; continue
  typing to filter commands.
- Use `@` for file mentions, `@@` for thread mentions, and `/threads` to resume
  past work.
- Use `/level <name>` (or `/model <name>`) to switch model level; the selected
  level is remembered per thread.
- Use `/goal enable [objective]` for durable checklist/notes. It can be enabled
  before the first message and will initialize when the thread starts.
- Use `/agents` to open Agent View for background worktree tasks. It has a small
  vim-like split: normal mode navigates sessions (`j/k`, `Enter`, `Space`, `c`,
  `d`, `D`, `?`), while input mode (`i` for a new task, `r` for a reply) edits
  the bottom composer and sends with `Enter` (`Ctrl+Enter` inserts a newline).
- Use `/status`, `/mcp`, and `/skills` to inspect runtime state and available
  capabilities.
- To use the original Textual-only panels such as `/config`, `/models`, Worktree
  management, or clipboard image shortcuts, start the old UI with `uv-agent tui`.

See [TUI and slash commands](docs/tui.md) for the full command and shortcut list.

## TUI Interfaces

uv-agent ships two interactive interfaces:

- **tui2** — the default (`uv-agent` or `uv-agent tui2`). A lightweight ANSI TUI
  that renders directly in the terminal, with compact status rows, command and
  mention palettes, streaming model/tool events, Goal mode, Worktree mode, and
  image attachments.
- **Textual TUI** — the original widget-based interface (`uv-agent tui`). It keeps
  the richer Textual layout and remains available when you prefer that UI. Its
  screenshot is linked here: [docs/t1.png](docs/t1.png).

## Plugins

Plugins are normal Python packages discovered through the `uv_agent.plugins`
entry point group. They run in the uv-agent host process and can register runtime
helpers, subscribe to events, or submit turns from external systems.

For a one-off run with an extra plugin package, add it beside the app launched by
`uvx`:

```powershell
uvx --with your-uv-agent-plugin uv-agent@latest
```

Disable installed plugins with `plugins.disabled` in config. See
[Plugin system](docs/plugins.md) for the plugin API, event bus, helper
registration, and examples.

## Runtime And Context

Every model-visible turn is built from a stable system prompt plus replayable
pre-user context items. The system prompt stays compact and rarely changes;
project and runtime details are delivered as structured messages immediately
before the user turn.

- **Managed runtime.** `run_python` is the only external action surface. Managed
  scripts run in the project-shared uv environment at
  `~/.uv-agent/projects/<project-id>/runner/scriptenv/` and import
  `uv_agent_runtime` helpers for file edits, search, subprocesses, dependency
  installation, subagents, image context, MCP clients, plugin helpers, and more.
  The script uv environment and the active working directory are separate; the
  cwd can move with `enter_dir` or Worktree mode.
- **Incremental runtime context.** Runtime environment, model levels, helper
  lists, direct script-environment dependencies, skills, MCP servers, and plugin
  helpers are split into fingerprinted context parts. uv-agent sends only
  changed parts inside `<context_update ...>` envelopes and explicitly marks
  removed skills or MCP servers so the model does not rely on stale capabilities.
- **Workspace and thread context.** Workspace rules are disclosed progressively:
  the model first receives a rule index, then full AGENTS.md content when it
  enters a relevant directory. Active cwd changes, image attachments, Worktree
  notices, tool results, run logs, and thread metadata are persisted in the same
  event stream and replayed when reconstructing a turn.
- **Goal mode durable memory.** `/goal` adds a per-thread memory layer under
  `~/.uv-agent/projects/<project-id>/goals/<thread-id>/` with `goal.json`,
  `checklist.md`, and `notes.md`. When Goal mode is active, uv-agent replays a
  `<goal_mode>` notice containing those paths and maintenance rules. The model
  uses `checklist.md` for acceptance criteria, progress, blockers, and next
  steps, and `notes.md` for decisions, investigation notes, constraints, and
  handoff context. The `goal_paths()` runtime helper lets managed scripts find
  those files without hard-coding paths.
- **Compaction and resume.** Checkpoint compaction summarizes the conversation
  while excluding reloadable runtime context, workspace rules, Goal notices, and
  Worktree notices from retained history. A new epoch replays current structured
  context before retained history, and Goal files remain the preferred source for
  long-running task progress after compaction or resume.

Thread state, run logs, shared script dependencies, attachments, Goal files, and
other project runtime data live under `~/.uv-agent/projects/<project-id>/`.

## Documentation

- [Configuration](docs/configuration.md)
- [Full config example](docs/config.example.json)
- [TUI and slash commands](docs/tui.md)
- [Runtime and managed scripts](docs/runtime.md)
- [Plugin system](docs/plugins.md)

## Development

uv-agent is developed in a self-bootstrapping style: the project is routinely
read, edited, tested, and refined with uv-agent itself.

```powershell
uv run pytest
```

Local debug state, screenshots, config, scripts, runs, and thread data belong in
`.uv-agent/` and should stay out of git.

## License

MIT. See [LICENSE](LICENSE).
