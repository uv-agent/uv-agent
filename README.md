# uv-agent

Experimental coding agent with a Textual TUI. The agent has one external action surface: it submits Python scripts to a managed `uv run` runner.

## Run

```powershell
uv run uv-agent
```

Single prompt smoke test:

```powershell
uv run uv-agent ask "Reply with exactly: ok"
```

## Local Config

User-level provider config lives at `~/.uv-agent/config.json` by default. A project can override it with `.uv-agent/config.json`; that project-local directory is ignored by git.

The config supports providers, models, levels, runtime options, and runner options. Secrets such as `api_key` must stay in ignored local config or environment variables.

Minimal shape:

```json
{
  "providers": {
    "local": {
      "base_url": "https://api.example.com",
      "api_key_env": "UV_AGENT_API_KEY",
      "responses": { "path": "/responses" },
      "chat_completions": { "path": "/chat/completions" }
    }
  },
  "models": {
    "main": {
      "provider": "local",
      "model": "gpt-5.5",
      "api": "responses",
      "context_window_tokens": 200000
    },
    "chat": {
      "provider": "local",
      "model": "gpt-5.4-mini",
      "api": "chat_completions",
      "context_window_tokens": 200000
    }
  },
  "levels": {
    "medium": { "model": "main" },
    "chat": { "model": "chat" }
  }
}
```

Supported model APIs:

- `responses`
- `chat_completions`
- `anthropic_messages`

Both are streamed with SSE when the provider supports streaming.

## Runtime Scripts

Agent scripts declare dependencies with PEP 723 inline metadata. The runner records managed script artifacts, run JSONL, and thread JSONL under `~/.uv-agent/projects/<project-id>/`, injects the configured runtime dependency into inline metadata when needed, and supports rerunning saved scripts.

`uv_agent_runtime` exposes convenience helpers for text/JSON files, subprocesses, structured events, nested `uv-agent ask` calls, and MCP stdio servers declared in `.agents/mcp.json`.

## TUI

The default TUI follows a Codex-style shape: a single transcript, inline Python runner events, and a bottom composer/status line.

- `Esc`: clear the current input when idle
- `Ctrl+C`: quit
- `?` + Enter or `/help`: show local commands
- `/new [title]`, `/threads`, and `/clear`: light thread controls
- `/config`: show config sources and redacted merged config
- `/models` and `/level [name]`: inspect and switch configured model levels
- `/mcp`: show MCP declarations from `.agents/mcp.json`
- `/skills` and `/skill [name]`: inspect `.agents/skills` entries
- `/runs`: show the latest Python run summary from this TUI session
- `/panel`: close the temporary panel
- Python runs appear inline with script/run ids, exit status, stdout/stderr summaries, and truncation markers
