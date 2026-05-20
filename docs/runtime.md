# Runtime And Managed Scripts

`uv-agent` gives the model one external action surface: `run_python`. The model
does not receive direct shell, filesystem, browser, network, or MCP tools.
Instead, it submits Python scripts to a managed runner. Those scripts may use
Python libraries, `subprocess`, and the `uv_agent_runtime` helper package.

## Managed Script Lifecycle

Each `run_python` call is saved as a managed script artifact:

```text
~/.uv-agent/projects/<project-id>/
  scripts/
    <script_id>/
      script.original.py
      script.py
      metadata.json
  runs/
    <run_id>.jsonl
  threads/
    <thread_id>.jsonl
  subthreads/
    <thread_id>.jsonl
```

The original script is stored separately from the final executable script. The
runner injects the configured runtime dependency into the final script when
needed so the script can import `uv_agent_runtime`.

Run JSONL records include the generated `run_id`, `script_id`, cwd, timeout,
`uv` argv, script args, stdout/stderr stream events, structured runtime events,
exit status, truncation state, and script paths.

The number of saved script artifacts is controlled by
`runner.max_saved_scripts`, defaulting to 32. Run logs are stored separately.

## Dependency Metadata

Managed scripts declare third-party dependencies with PEP 723 inline metadata:

```python
# /// script
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///

from uv_agent_runtime import emit_result

emit_result(ok=True)
```

The tool API does not expose a separate dependency argument. Dependencies belong
inside the script. The runner may rewrite the final script to add the configured
runtime dependency, but ordinary script dependencies stay in inline metadata.

The published `uv-agent` package includes both `uv_agent` and
`uv_agent_runtime`. For local source development, point the runtime dependency at
the checkout:

```json
{
  "runner": {
    "runtime_dependency": "uv-agent @ file:///C:/path/to/uv-agent"
  }
}
```

See [configuration](configuration.md) for runner options.

## Rerun Behavior

Saved scripts can be rerun by `script_id` or `run_id`.

- Rerunning by `script_id` uses the saved final script content.
- Rerunning by `run_id` first resolves the script used by that run.
- `rerun` mode uses fresh args, cwd, and timeout unless they are supplied.
- `replay` mode inherits the previous run's `uv_args`, `script_args`, cwd, and
  timeout when a `run_id` is available.

Reruns do not roll back filesystem side effects from the original run. They are
best treated as repeat executions of a saved script artifact, not as a full
time-travel snapshot.

Scripts can inspect recent saved script summaries:

```python
from uv_agent_runtime import saved_scripts

for script in saved_scripts(limit=5):
    print(script["script_id"], script["summary"])
```

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
| `saved_scripts` | Inspect recent managed scripts. |
| `thread_digest`, `list_thread_digests` | Read compact thread summaries from project state. |
| `ask` | Launch a nested `uv-agent ask` subagent. |
| `connect_stdio`, `connect_declared`, `connect_named`, `list_declared_servers` | MCP stdio helpers. |

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

The host filters internal structured-event lines out of the tool output that is
fed back to the model. UI/log payloads may keep richer event details, but the
model-visible payload does not include runtime events. Print any event fields
that are needed for later reasoning.

## Directory Rules And Cwd

Managed scripts start in the thread's active cwd, which defaults to the
workspace root. Use `enter_dir(path)` when switching work to a subdirectory:

```python
from uv_agent_runtime import enter_dir

enter_dir("src")
```

`enter_dir` behaves like `os.chdir(path)` for the running script and persists
that directory as the default cwd for later runs in the same thread. The host may
load `AGENTS.md` / `AGENTS.*.md` files in the entered directory and returns newly
loaded rule text in the same Python tool result. Rule files are de-duplicated
within the current context epoch; after compaction, dynamic context is rebuilt,
rules are not summarized, a fresh local rule index is generated from the active
cwd, and active-directory rules are loaded again on demand.

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

Use `level` or `model_level` only when intentionally selecting a configured
level:

```python
result = ask("Review this patch for regressions.", level="large", check=True)
```

Nested agents run through a subprocess from inside the Python runner, preserving
the single external action surface. When project state is available, retained
subagents are stored under `subthreads/` and linked to the parent thread, turn,
run, and script ids.

## MCP From Runtime

MCP is available from managed Python scripts, not as direct model tools.

```python
from uv_agent_runtime import connect_named

with connect_named("filesystem") as client:
    client.initialize()
    print(client.list_tools())
```

`connect_named` searches user/project MCP declarations such as
`.agents/mcp.json`. `connect_stdio` can connect to an explicit stdio server
command when a script needs a one-off server.

## Environment

The runner sets useful environment variables for managed scripts:

| Variable | Meaning |
| --- | --- |
| `UV_AGENT_RUNTIME_STATE_DIR` | Project state directory for scripts, runs, threads, and attachments. |
| `UV_AGENT_RUNTIME_THREAD_ID` | Current parent thread id when available. |
| `UV_AGENT_RUNTIME_TURN_ID` | Current turn id when available. |
| `UV_AGENT_RUNTIME_RUN_ID` | Current run id. |
| `UV_AGENT_RUNTIME_SCRIPT_ID` | Current script id. |
| `PYTHONIOENCODING` | Forced to `utf-8` for child output. |
| `PYTHONUTF8` | Forced to `1` for child Python UTF-8 behavior. |

Scripts should still use explicit encodings for file I/O when practical.
