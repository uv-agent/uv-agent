# uv-agent

[简体中文](README.zh-CN.md)

`uv-agent` is a portable coding agent with a Textual TUI. Its only external
action surface is `run_python`: the model submits Python scripts to a managed
`uv run` runner, and those scripts do the actual work. This single-tool design
keeps agent behavior consistent across platforms—any OS with Python and uv
behaves the same way.

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
> | `"chat_completions"` | OpenAI Chat Completions API | ✅ Primary (fully supported) |
> | `"responses"` | OpenAI Responses API | ⚠️ Experimental — may have bugs |
> | `"anthropic_messages"` | Anthropic Messages API | ⚠️ Experimental — may have bugs |
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
    }
  },
  "levels": {
    "deepseek-flash": {
      "model": "deepseek-v4-flash"
    },
    "deepseek-pro": {
      "model": "deepseek-v4-pro"
    }
  },
  "runtime": {
    "default_level": "deepseek-flash",
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
- Managed scripts declare third-party dependencies with PEP 723 inline metadata.
- The distributed package includes both `uv_agent` and `uv_agent_runtime`; managed
  scripts import helpers from `uv_agent_runtime`.
- Workspace rules, skills, and MCP declarations are progressively disclosed as
  context. MCP calls happen from Python runtime helpers, not direct model tools.
- Thread state, run logs, saved scripts, and attachments live under
  `~/.uv-agent/projects/<project-id>/`.

## Development

```powershell
uv run pytest
```

Local debug state, screenshots, config, scripts, runs, and thread data belong in
`.uv-agent/` and should stay out of git.

## License

MIT. See [LICENSE](LICENSE).
