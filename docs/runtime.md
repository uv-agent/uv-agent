# Runtime And Managed Scripts

`uv-agent` gives the model one external action surface: `run_python`. The model
does not receive direct shell, filesystem, browser, network, or MCP tools.
Instead, it submits Python scripts to a managed runner. Those scripts may use
Python libraries, subprocesses, and the `uv_agent_runtime` helper package.

## Managed Script Lifecycle

Each project has one shared script venv:

```text
~/.uv-agent/projects/<project-id>/
  uv-agent.sqlite3
  runner/
    scriptenv/
      pyproject.toml
      uv.lock
      .venv/
    scripts/
      <run_id>.py          # debug/export copy; SQLite is the source of truth
  attachments/
```

The runner creates the `scriptenv` uv project lazily with `uv init`, adds
`uv-agent` with `uv add` (editable from the current checkout during source-tree
development), stores each `run_python` call and its events in
`uv-agent.sqlite3`, exports a debug script to `runner/scripts/<run_id>.py`, and
executes it with
`uv run --project <scriptenv> --directory <active-cwd> python <run_id>.py`. Run
records include the generated `run_id`, cwd, timeout, script args,
stdout/stderr stream events, runtime RPC events, exit status, truncation state,
and script path.

The number of retained run records is controlled by `runner.max_run_logs`,
defaulting to 200. Pruning deletes old rows from SQLite and removes matching
exported debug scripts.

## Dependencies

To add a third-party package to the shared run environment, use
`add_dependency`:

```python
from uv_agent_runtime import add_dependency

add_dependency("requests", check=True)

import requests
```

Call `add_dependency` before importing the package in the current script. It is
not a way to upgrade or replace a package that has already been imported in that
same Python process.

The runtime context shows the `run_python` environment directory and its
`pyproject.toml`. That directory is the uv project used by `run_python`, not the
workspace or active cwd. Direct dependencies from that `pyproject.toml` are shown
in runtime context; transitive dependencies from `uv.lock` are not. Installed
packages persist in the project script environment for later runs.

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
| **File I/O** | |
| `read_text`, `write_text` | Read or write workspace-relative text files. |
| `read_text_lossless`, `write_text_lossless` | Read/write text while preserving encoding, BOM, newline style, and final newline. |
| `read_json`, `write_json` | Read or write workspace-relative JSON files. |
| `list_files`, `resolve_workspace_path` | Enumerate workspace files or resolve relative paths. |
| `path_info` | Inspect a resolved path (existence, kind, size, relative-to-base check). |
| **Text editing** | |
| `replace_text` | Perform small targeted text replacements in an existing file. |
| `apply_patch` | Apply edits via the `*** Begin Patch` envelope format. |
| `apply_patch_any` | Apply edits by auto-detecting patch format (envelope or unified diff). |
| `convert_patch` | Convert between the envelope format and unified diff format. |
| `make_unified_diff` | Generate a unified diff from before/after text. |
| **File snapshots & transactions** | |
| `snapshot_files`, `restore_snapshot` | Capture file state and restore it later (manual savepoints). |
| `workspace_transaction` | Multi-file edit transaction with automatic rollback on error. |
| **Search & discovery** | |
| `search_text` | Grep-like content search via ripgrep across the workspace. |
| `find_files` | Enumerate workspace files via ripgrep (faster than manual walking). |
| `find_symbols` | Locate function/class/method/etc. definitions via tree-sitter. |
| `query_code` | Run a custom tree-sitter S-expression query over the workspace. |
| `supported_symbol_languages` | List languages with built-in tree-sitter symbol support. |
| `clear_codequery_cache` | Drop the tree-sitter capture cache (scoped to a root path). |
| **Text utilities** | |
| `compare_text`, `normalize_text` | Compare two texts or normalize line endings/whitespace. |
| **Managed environment** | |
| `add_dependency`, `add_dependencies` | Add third-party packages to the shared `run_python` uv project. |
| `run_python_env_dir` | Return the shared script venv directory path. |
| `enter_dir` | Change the active working directory and trigger directory rule loading. |
| **Subprocess** | |
| `run_process_text` | Run external commands with explicit text decoding, env support, timeouts, and optional `check=True`. |
| **Structured events** | |
| `emit_event`, `emit_progress`, `emit_result` | Emit structured JSON events rendered by the host. |
| `look_at` | Attach image context to the conversation. |
| **Thread introspection** | |
| `thread_digest`, `list_thread_digests` | Read compact conversation summaries from project state. |
| **Nested agents** | |
| `ask` | Launch a nested `uv-agent ask` subagent. |
| **MCP** | |
| `connect_named` | Connect to an MCP server declared in user/project `.agents/mcp.json`. |
| `connect_declared` | Connect to a server declared in a specific `.agents/mcp.json` file. |
| `connect_url` | Connect to an MCP server via HTTP/SSE URL. |
| `connect_stdio` | Connect to an MCP server via a local stdio command. |
| `list_declared_servers` | List MCP servers from user and project declarations. |

## Structured Events And Host Calls

Managed scripts send structured runtime events to the host over a local
JSON-RPC-over-HTTP channel. The channel is internal to the runner: stdout and
stderr remain pure user output, so printing JSON never creates a structured
event.

```python
from uv_agent_runtime import emit_progress, emit_result

progress = emit_progress("reading files", count=12)
result = emit_result(status="ok")
print(result["status"])
```

Use structured events for machine-readable progress or results. Each event gets
an `_uv_agent_event_id`, and events emitted from managed runs also carry
`_uv_agent_run_id`. The runner stores delivered events in the run result and as
`run.event` rows in `uv-agent.sqlite3`. Regular stdout and stderr are captured
only as `run.stdout` / `run.stderr` run events.

The same channel supports explicit host calls for registered helpers:

```python
from uv_agent_runtime import call_host

result = call_host("helper_name", value=1)
```

`call_host` raises if the helper is missing or the host returns an error.

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
model input. Large image bytes are not embedded directly in the thread event
payload.

## Nested Agents

Managed scripts can launch a nested `uv-agent ask` through `ask`:

```python
from uv_agent_runtime import ask

result = ask("Inspect the failing test and summarize the likely cause.", check=True)
print(result.text)
```

Nested agents run through a subprocess from inside the Python runner, preserving
the single external action surface. When project state is available, retained
subagents are stored in SQLite with `kind='subagent'` and linked to the parent
thread, turn, and run ids.

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
| `UV_AGENT_RPC_URL` | Loopback runtime RPC endpoint for structured events and host calls. |
| `UV_AGENT_RPC_TOKEN` | Per-run bearer token for the runtime RPC endpoint. |
| `UV_AGENT_SCRIPTENV_DIR` | Project shared `scriptenv` uv project directory. |
| `UV_AGENT_SCRIPT_DIR` | Directory containing exported debug run scripts. |
| `UV_BIN` | Resolved `uv` executable used by the runner. |
| `PYTHONIOENCODING` | Forced to `utf-8` for child output. |
| `PYTHONUTF8` | Forced to `1` for child Python UTF-8 behavior. |

Scripts should still use explicit encodings for file I/O when practical.
