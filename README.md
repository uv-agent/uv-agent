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

Local provider config can live at `.uv-agent/config.json`. That directory is ignored by git.

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

Both are streamed with SSE when the provider supports streaming.

## Runtime Scripts

Agent scripts declare dependencies with PEP 723 inline metadata. The runner records managed script artifacts and run JSONL under `.uv-agent/`, injects the configured runtime dependency into inline metadata when needed, and supports rerunning saved scripts.

## TUI

The default TUI follows a Codex-style shape: a single transcript, inline Python runner events, and a bottom composer/status line.

- `Esc`: clear the current input when idle
- `Ctrl+C`: quit
- `?` + Enter or `/help`: show local commands
- `/new [title]`, `/threads`, and `/clear`: light thread controls
- Python runs appear inline with script/run ids, exit status, stdout/stderr summaries, and truncation markers
