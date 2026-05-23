# uv-agent

[简体中文](README.zh-CN.md)

`uv-agent` is a Windows-first coding agent with a Textual TUI. It is designed
to feel at home on Windows, where many coding agents stumble over PowerShell
quoting, shell semantics, or Unix-first assumptions. Its only external action
surface is `run_python`: the model submits Python scripts to a managed `uv run`
runner, and those scripts do the actual work instead of relying on fragile
shell snippets. Around this `run_python` boundary, uv-agent's context layer
applies Harness Engineering ideas: checkpoint compaction, stable incremental
updates, protocol-safe interruption handling, and epoch replay keep the model's
view coherent during long-running work. See
[Context Management](#context-management) for details.

Public APIs, config fields, and runtime behavior may still change as the
project evolves.

## Prerequisites

Install the following tools:

- **uv** — https://docs.astral.sh/uv/getting-started/installation/
  Python package and project manager used to run the agent.
- **ripgrep** — https://github.com/BurntSushi/ripgrep#installation
  Used for fast file-content searches inside the workspace.

## Install And Run

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

## Configuration

User config lives at `~/.uv-agent/config.json`. A project can override it with
`.uv-agent/config.json`; that project-local directory is ignored by git. Keep
API keys in environment variables or ignored local config.

> **API compatibility**  
> This project supports three API formats — set `api` on your model config:
> 
> | `api` value | Format | Status |
> |---|---|---|
> | `"chat_completions"` | OpenAI Chat Completions API | ✅ Supported |
> | `"responses"` | OpenAI Responses API | ✅ Supported |
> | `"anthropic_messages"` | Anthropic Messages API | ✅ Supported |
> 
> Issues and PRs are welcome for any format!Example configuration:

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
    "max_output_bytes": 1000000
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
  }
}

```

Use `/config` in the TUI to switch the default level, language, and automatic
compression. Set `ui.language` to `zh-CN` for a Chinese UI. Completion
notifications can be configured under `ui.completion_notification`. Non-Windows
platforms use the terminal bell for completion sound.


See [configuration](docs/configuration.md) for all supported options and
[config.example.json](docs/config.example.json) for a detailed example.

## Documentation

- [Configuration](docs/configuration.md)
- [Full config example](docs/config.example.json)
- [TUI and slash commands](docs/tui.md)
- [Runtime and managed scripts](docs/runtime.md)

## Core Ideas

- The agent has exactly one external action surface: `run_python`.
- Managed scripts run in a project-shared uv environment; scripts add
  third-party dependencies to that environment with `add_dependency`.
- The distributed package includes both `uv_agent` and `uv_agent_runtime`; managed
  scripts import helpers from `uv_agent_runtime`.
- Workspace rules, skills, and MCP declarations are progressively disclosed as
  context. MCP calls happen from Python runtime helpers, not direct model tools.
- Thread state, run logs, the shared script environment, and attachments live under
  `~/.uv-agent/projects/<project-id>/`.

## Context Management

uv-agent's context management is one part of its Harness Engineering approach: it brings the agent's inputs, actions, state, and exception handling into an explicit engineering protocol so long-running work remains traceable, recoverable, and maintainable. Two mechanisms anchor the design: **checkpoint compaction** creates durable continuation points, and the single **`run_python` execution surface** makes every external action flow through the same event stream.

- **Incremental, fingerprinted updates.** Runtime environment, model levels, helpers, skills, and MCP declarations are split into context parts. Only changed dynamic parts are re-sent inside `<context_update ...>` messages; unchanged parts remain current within the epoch, and removed skills or MCP servers are explicitly tombstoned.
- **Stable prefix and ordering.** The system prompt stays stable. Dynamic context is appended as pre-user messages with a fixed update prefix and stable section order, which keeps long conversations from drifting as context grows or changes.
- **Protocol-safe sequence completion.** Because `run_python` is the only external action surface, tool calls, runner results, working-directory updates, rule loads, attachments, and dependency state all flow through one persistent event stream. If a turn is interrupted, unfinished tool calls receive explicit synthetic outputs and bridge messages; partial model streams and provider or tool errors are recorded instead of being treated as successful completions.
- **Epoch replay after compaction.** A compaction checkpoint stores a continuation summary plus retained recent conversation while excluding reloadable runtime and rule context. The next epoch re-emits the current runtime context and workspace rules before retained history; mid-turn compaction uses the same ordering before the assistant continues after tool results.

Together, these mechanisms keep the model's view coherent across workspace changes, runtime changes, interruptions, errors, and long-running sessions.

## Development

uv-agent is developed in a self-bootstrapping style: the project is routinely read, edited, tested, and refined with uv-agent itself.

```powershell
uv run pytest
```

Local debug state, screenshots, config, scripts, runs, and thread data belong in
`.uv-agent/` and should stay out of git.

## License

MIT. See [LICENSE](LICENSE).
