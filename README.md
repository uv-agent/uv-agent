# uv-agent

[简体中文](README.zh-CN.md)

`uv-agent` is a portable coding agent with a Textual TUI. Its only external
action surface is `run_python`: the model submits Python scripts to a managed
`uv run` runner, and those scripts do the actual work. This single-tool design
keeps agent behavior consistent across platforms—any OS with Python and uv
behaves the same way.

Public APIs, config fields, and runtime behavior may still change as the
project evolves.

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

Minimal shape:

```json
{
  "providers": {
    "main": {
      "base_url": "https://api.example.com/v1",
      "api_key_env": "UV_AGENT_API_KEY",
      "responses": { "path": "/responses" }
    }
  },
  "models": {
    "main": {
      "provider": "main",
      "model": "your-model-name",
      "api": "responses",
      "context_window_tokens": 128000
    }
  },
  "levels": {
    "medium": { "model": "main" }
  },
  "runtime": {
    "default_level": "medium"
  },
  "ui": {
    "language": "auto"
  }
}
```

Use `/config` in the TUI to switch the default level, language, and automatic
compression. Set `ui.language` to `zh-CN` for a Chinese UI.

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
