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

Resume a thread without opening the TUI:

```powershell
uv run uv-agent ask --thread thr_xxx "Continue from here"
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
      "context_window_tokens": 258000
    },
    "chat": {
      "provider": "local",
      "model": "gpt-5.4-mini",
      "api": "chat_completions",
      "context_window_tokens": 258000
    }
  },
  "levels": {
    "medium": { "model": "main" },
    "chat": { "model": "chat" }
  },
  "ui": {
    "language": "auto"
  },
  "runner": {
    "max_saved_scripts": 32
  }
}
```

Supported model APIs:

- `responses`
- `chat_completions`
- `anthropic_messages`

Streaming uses SSE when the provider supports it.

Compression can be tuned under `runtime.compression` with `trigger_ratio`,
`target_ratio`, and `min_tokens`. The TUI reads provider `usage` when available
and falls back to a local estimate for the context meter.

## Runtime Scripts

Agent scripts declare dependencies with PEP 723 inline metadata. The runner records managed script artifacts, run JSONL, and thread JSONL under `~/.uv-agent/projects/<project-id>/`, injects the configured runtime dependency into inline metadata when needed, and supports rerunning saved scripts.

Each project keeps the 32 most recently used managed scripts by default
(`runner.max_saved_scripts`). Run logs are still kept separately. Scripts can
inspect recent saved scripts with `uv_agent_runtime.saved_scripts()`.

`uv_agent_runtime` exposes convenience helpers for text/JSON files, subprocesses, structured events, nested `uv-agent ask` calls by model level, image context, saved script summaries, and MCP stdio servers declared in `.agents/mcp.json`.

Image context is added from a script with `look_at`:

```python
from uv_agent_runtime import look_at

look_at("screenshot.png", note="inspect the failed layout")
```

The host copies the image into the project state attachments directory and
appends it to later model input. Large image bytes are not stored directly in
thread JSONL.

Saved scripts can be rerun by passing `script_id` or `run_id` to `run_python`;
`rerun_mode="replay"` inherits a previous run's arguments when a `run_id` is
available.

Nested agents are launched from Python with level names rather than concrete
models:

```python
from uv_agent_runtime import ask

result = ask("inspect the test failure", level="small", check=True)
print(result.text)
```

MCP remains a Python-triggered runtime capability, not a direct model tool:

```python
from uv_agent_runtime import connect_named

with connect_named("filesystem") as client:
    client.initialize()
    print(client.list_tools())
```

## Workspace Rules

`AGENTS.md` context is loaded from `~/.agents/AGENTS.md` and from the current
git root down to the startup directory, including `AGENTS.*.md` variants.
Rules, discovered skills, and MCP declarations are kept out of the stable system
prompt. The engine appends a compact workspace-context update only when that
context is first seen, changed, removed, or when a thread continues after
compaction. Removal updates explicitly tell the agent not to rely on older
appended rule/capability context unless it appears again. Context update events
are stored in thread JSONL for change tracking, but they are not reconstructed as
ordinary conversation items or included in compression input.

## TUI

The default TUI follows a Codex-style shape: a single transcript, inline Python runner events, full-screen focus panels, and a bottom composer with a plain metadata line, bordered input box, and plain hint line.

- `Enter`: insert a newline in the composer
- `Ctrl+Enter` or `Ctrl+J`: send the composer text
- `Esc`: close command suggestions, clear the composer, or close the open panel
- `Ctrl+S` or `/status`: open detailed runtime status
- `Ctrl+O` or `/threads`: open a searchable thread picker and resume history
- `Ctrl+P`: open a full-screen command palette
- `Ctrl+Q`, `Ctrl+C`, or `/quit`: quit after a second confirmation
- `?` + `Ctrl+Enter` or `/help`: show local commands
- `/context`: show context budget and loaded workspace rules
- `/rules`: inspect loaded `AGENTS.md` files
- `/new [title]`, `/threads`, and `/clear`: light thread controls
- `/config`: show config sources and redacted merged config
- `/models` and `/level [name]`: inspect and switch configured model levels
- `/mcp`: show MCP declarations from `.agents/mcp.json`
- `/skills` and `/skill [name]`: inspect `.agents/skills` entries
- `/runs`: show the latest Python run summary from this TUI session
- `/scripts`: show recent managed script summaries
- `/panel`: reminder that panels close with `Esc`
- Python runs appear inline with script/run ids, exit status, stdout/stderr summaries, and truncation markers
- Reasoning/tool/runtime events appear as compact timeline entries; detailed run output remains available in `/runs`.
- Typing `/` opens live command suggestions; `Tab`, `Enter`, and arrow keys work inside the picker.
- Temporary panels open as full-screen overlays with search/scroll/select behavior.

For manual UI checks, Textual's `App.export_screenshot()` works in headless
tests and writes SVG snapshots. Local debug captures belong under `.uv-agent/`
so they stay out of git.
