# Runtime And Managed Scripts

`uv-agent` gives the model one external action surface: `run_python`. The model
does not receive direct shell, filesystem, browser, network, or MCP tools.
Instead, it submits Python scripts to a managed runner. Those scripts may use
Python libraries, subprocesses, and the `uv_agent_runtime` helper package.

## Managed Script Lifecycle

Each project has one shared script venv:

```text
~/.uv-agent/projects/<project-id>/
  runner/
    scriptenv/.venv/
    runs/
      <run_id>.py
      <run_id>.jsonl
  threads/
    <thread_id>.jsonl
    <thread_id>.json
  subthreads/
    <thread_id>.jsonl
    <thread_id>.json
```

The runner creates the venv lazily, installs the `uv-agent` runtime package into
it, writes each `run_python` call to `runner/runs/<run_id>.py`, and executes that
file with the venv Python. Run JSONL records include the generated `run_id`, cwd,
timeout, script args, stdout/stderr stream events, structured runtime events,
exit status, truncation state, and script path.

The number of retained run log pairs is controlled by `runner.max_run_logs`,
defaulting to 200. The runner prunes `<run_id>.py` and `<run_id>.jsonl` together.

## Dependencies

To install a third-party package, run `uv pip install` against the current
interpreter:

```python
import sys

from uv_agent_runtime import run_process_text

run_process_text(
    ["uv", "pip", "install", "--python", sys.executable, "-q", "requests"],
    check=True,
)

import requests
```

For dependency installation, leave the active directory unchanged and call
`run_process_text` without `cwd`. The target environment is chosen by
`--python sys.executable`; the current directory is not part of the install
semantics. Installed packages persist in the project script venv for later runs.

## Runtime Helpers

Import helpers from `uv_agent_runtime` inside managed scripts:

```python
from uv_agent_runtime import read_text, run_process_text, emit_result

content = read_text("README.md")
result = run_process_text(["python3", "--version"], check=True)
emit_result(readme_bytes=len(content.encode()), python=result.stdout.strip())
```

Available helper groups:

| Helper | Purpose |
| --- | --- |
| `read_text`, `write_text`, `read_json`, `write_json`, `list_files`, `resolve_workspace_path` | Workspace-relative file helpers. |
| `run_process_text` | Argv-list subprocess helper with explicit text decoding, env/env_patch support, timeouts, and optional `check=True`. |
| `apply_patch` | Codex-style `*** Begin Patch` file edit helper. |
| `enter_dir` | Change the active working directory and trigger directory rule loading. |
| `emit_event`, `emit_progress`, `emit_result` | Structured events rendered by the host; each returns the emitted event dict. |
| `look_at` | Attach image context to the conversation and return the emitted event dict. |
| `thread_digest`, `list_thread_digests` | Read compact thread summaries from project state. |
| `ask` | Launch a nested `uv-agent ask` subagent. |
| `connect_stdio`, `connect_url`, `connect_declared`, `connect_named`, `list_declared_servers` | MCP helpers backed by the official SDK. |

## Structured Events

Runtime events are printed as JSON lines on stdout. The runner recognizes and
stores them, and the TUI renders common event kinds compactly.

```python
from uv_agent_runtime import emit_progress, emit_result

progress = emit_progress("reading files", count=12)
result = emit_result(status="ok")
print(result["status"])
```

Use structured events for machine-readable progress or results. Each event gets
an `_uv_agent_event_id`, and events emitted from managed runs also carry
`_uv_agent_run_id`. Runtime helpers write each JSON event line atomically within
the Python process so threaded scripts do not interleave event text. Regular
stdout and stderr are still captured.

## Directory Rules And Cwd

Managed scripts start in the thread's active cwd, which defaults to the
workspace root. The `run_python` tool does not expose a `cwd` parameter. Use
`enter_dir(path)` when switching work to a subdirectory:

```python
from uv_agent_runtime import enter_dir

enter_dir("src")
```

`enter_dir` behaves like `os.chdir(path)` for the running script and persists
that directory as the default cwd for later runs in the same thread. The host may
load `AGENTS.md` / `AGENTS.*.md` files in the entered directory and returns newly
loaded rule text in the same Python tool result.

## Image Context

Use `look_at` when a script produces or finds an image that the model should
inspect in later context:

```python
from uv_agent_runtime import look_at

image = look_at("screenshot.png", note="inspect the failed layout")
print(image["path"])
```

The host copies image metadata into project state and appends the image to later
model input. Large image bytes are not embedded directly in thread JSONL.

## Nested Agents

Managed scripts can launch a nested `uv-agent ask` through `ask`:

```python
from uv_agent_runtime import ask

result = ask("Inspect the failing test and summarize the likely cause.", check=True)
print(result.text)
```

Nested agents run through a subprocess from inside the Python runner, preserving
the single external action surface. When project state is available, retained
subagents are stored under `subthreads/` and linked to the parent thread, turn,
and run ids.

## MCP From Runtime

MCP is available from managed Python scripts, not as direct model tools.

```python
from uv_agent_runtime import connect_named

with connect_named("filesystem") as client:
    client.initialize()
    print(client.list_tools())
```

`connect_named` searches user/project MCP declarations such as
`.agents/mcp.json`. Declarations may use stdio, Streamable HTTP, or SSE.

## Environment

The runner sets useful environment variables for managed scripts:

| Variable | Meaning |
| --- | --- |
| `UV_AGENT_RUNTIME_PROJECT_ROOT` | Project workspace root. |
| `UV_AGENT_RUNTIME_STATE_DIR` | Project state directory for runs, threads, and attachments. |
| `UV_AGENT_RUNTIME_THREAD_ID` | Current parent thread id when available. |
| `UV_AGENT_RUNTIME_THREAD_KIND` | Current thread kind when available. |
| `UV_AGENT_RUNTIME_TURN_ID` | Current turn id when available. |
| `UV_AGENT_RUNTIME_RUN_ID` | Current run id. |
| `UV_AGENT_VENV_DIR` | Project shared script venv directory. |
| `UV_AGENT_SCRIPT_DIR` | Directory containing run scripts and logs. |
| `UV_BIN` | Resolved `uv` executable used by the runner. |
| `PYTHONIOENCODING` | Forced to `utf-8` for child output. |
| `PYTHONUTF8` | Forced to `1` for child Python UTF-8 behavior. |

Scripts should still use explicit encodings for file I/O when practical.
