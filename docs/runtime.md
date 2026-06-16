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
exported debug scripts. Set `runner.scriptenv_index_url` to make the managed
`scriptenv` pyproject use a specific uv default package index, for example a
regional PyPI mirror.

## Dependencies

To add a third-party package to the shared run environment, use
`rt.deps.add`:

```python
import uv_agent_runtime as rt

rt.deps.add("requests", check=True)

import requests
```

Call `rt.deps.add` before importing the package in the current script. It is
not a way to upgrade or replace a package that has already been imported in that
same Python process.

The runtime context shows the `run_python` environment directory and its
`pyproject.toml`. That directory is the uv project used by `run_python`, not the
workspace or active cwd. Direct dependencies from that `pyproject.toml` are shown
in runtime context; transitive dependencies from `uv.lock` are not. Installed
packages persist in the project script environment for later runs.

## Runtime Helpers

Import the runtime facade inside managed scripts:

```python
import uv_agent_runtime as rt

content = rt.file("README.md").text()
result = rt.run("python3", "--version", check=True)
rt.events.result(readme_bytes=len(content.encode()), python=result.stdout.strip())
```

Available helper groups:

| Helper | Purpose |
| --- | --- |
| **File I/O** | |
| `rt.file(path).read()` / `.text()` / `.json()` | Read text views, raw text, or JSON. |
| `rt.file(path).write()` / `.write_json()` | Write full text or JSON while preserving file metadata when requested. |
| `rt.file(path).info()` / `rt.path(path)` | Inspect a resolved path (existence, kind, size, relative-to-base check). |
| **Text editing** | |
| `rt.file(path).replace()` | Perform small targeted text replacements in an existing file. |
| `rt.file(path).edit()` / `.insert_before()` / `.insert_after()` / `.delete_lines()` | Apply anchored line edits while preserving newline/BOM/final-newline metadata. |
| `rt.patch()` / `rt.apply_patch()` / `rt.dry_run_patch()` | Apply or validate patch-envelope / unified-diff edits. |
| `rt.convert_patch()` / `rt.diff()` | Convert patch formats or generate unified diffs. |
| **File snapshots & transactions** | |
| `rt.snapshot()` / `rt.restore()` / `rt.transaction()` | Capture file state and restore it later, manually or as a rollback transaction. |
| **Search & discovery** | |
| `rt.search()` | Grep-like content search via ripgrep across the workspace. |
| `rt.files()` | Enumerate workspace files via ripgrep (faster than manual walking). |
| `rt.symbols()` | Locate function/class/method/etc. definitions via tree-sitter. |
| `rt.query()` | Run a custom tree-sitter S-expression query over the workspace. |
| **Text utilities** | |
| `rt.compare()` / `rt.normalize()` | Compare two texts or normalize line endings/final newline. |
| **Managed environment** | |
| `rt.deps.add()` | Add third-party packages to the shared `run_python` uv project. |
| `rt.deps.env_dir()` | Return the shared script venv directory path. |
| `rt.cd()` / `rt.pwd()` | Change or inspect the active working directory; `rt.cd()` triggers directory rule loading. |
| **Subprocess** | |
| `rt.run()` | Run external commands with explicit text decoding, env support, timeouts, and optional `check=True`. |
| **Structured events** | |
| `rt.events.emit()` / `.progress()` / `.result()` | Emit structured JSON events rendered by the host. |
| `rt.look_at()` / `rt.events.look_at()` | Attach image context to the conversation. |
| **Thread introspection** | |
| `rt.threads.list()` / `.view()` / `.detail()` | Find stored threads, view conversation-only epochs, and expand process/run details by id or turn id. |
| **Nested model work** | |
| `rt.workflow.start()` / `.resume()` / `.agent()` | Build persistent workflow graphs for nested or long-running model tasks. |
| **MCP** | |
| `rt.mcp.list()` | List MCP servers from user and project declarations. |
| `rt.mcp.connect()` / `.connect_declared()` | Connect to declared MCP servers. |
| `rt.mcp.connect_url()` / `.connect_stdio()` | Connect to MCP servers via HTTP/SSE URL or local stdio command. |

## Structured Events And Host Calls

Managed scripts send structured runtime events to the host over a local
JSON-RPC-over-HTTP channel. The channel is internal to the runner: stdout and
stderr remain pure user output, so printing JSON never creates a structured
event.

```python
import uv_agent_runtime as rt

progress = rt.events.progress("reading files", count=12)
result = rt.events.result(status="ok")
print(result["status"])
```

Use structured events for machine-readable progress or results. Each event gets
an `_uv_agent_event_id`, and events emitted from managed runs also carry
`_uv_agent_run_id`. The runner stores delivered events in the run result and as
`run.event` rows in `uv-agent.sqlite3`. Regular stdout and stderr are captured
only as `run.stdout` / `run.stderr` run events.

The same channel supports explicit host calls for registered helpers:

```python
import uv_agent_runtime as rt

result = rt.helper_name(value=1)
```

Dynamic `rt.<helper>` calls raise if the helper is missing or the host returns an error.

## Directory Rules And Cwd

Managed scripts start in the thread's active cwd, which defaults to the
workspace root. The `run_python` tool does not expose a `cwd` parameter. Use
`rt.cd(path)` when switching work to a subdirectory:

```python
import uv_agent_runtime as rt

rt.cd("src")
```

`rt.cd` behaves like `os.chdir(path)` for the running script and persists
that directory as the default cwd for later runs in the same thread. The host may
load `AGENTS.md` / `AGENTS.*.md` files in the entered directory and returns newly
loaded rule text in the same Python tool result.

## Image Context

Use `rt.look_at` when a script produces or finds an image that the model should
inspect in later context:

```python
import uv_agent_runtime as rt

image = rt.look_at("screenshot.png", note="inspect the failed layout")
print(image["path"])
```

The host copies image metadata into project state and appends the image to later
model input. Large image bytes are not embedded directly in the thread event
payload.

## Nested Agents

Managed scripts can launch nested model work through `rt.workflow`:

```python
import uv_agent_runtime as rt

wf = rt.workflow.start("Inspect the failing test")
node = wf.agent("Inspect the failing test and summarize the likely cause.")
result = node.wait()
print(result.text())
```

Workflow nodes run through managed uv-agent subprocesses from inside the Python runner, preserving
the single external action surface. When project state is available, retained
subagents are stored in SQLite with `kind='subagent'` and linked to the parent
thread, turn, and run ids.

## MCP From Runtime

MCP is available from managed Python scripts, not as direct model tools.

```python
import uv_agent_runtime as rt

with rt.mcp.connect("filesystem") as client:
    client.initialize()
    print(client.list_tools())
```

`rt.mcp.connect()` searches user/project MCP declarations such as
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
