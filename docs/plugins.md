# Plugin System

`uv-agent` plugins are installed Python packages that extend the host process while
preserving the agent's core boundary: the model still has exactly one external
action surface, `run_python`. A plugin can add new helpers that managed scripts
import from `uv_agent_runtime`, subscribe to agent events, and submit turns from
external systems such as chat bots or webhooks.

The plugin API is intentionally small and may evolve with the project. Plugins
run in the uv-agent host process, so install only plugins you trust.

## What Plugins Can Do

A plugin can:

- register runtime helpers that become available inside `run_python` scripts;
- subscribe to the agent event stream for notifications, relays, audit logs, or
  integrations;
- submit a user turn programmatically and consume that turn's event stream;
- keep private data in a per-plugin SQLite database;
- write diagnostic logs to a per-plugin log file.

Plugins cannot intercept, rewrite, or veto model input/output in this first
version. They also do not add direct model tools; model-visible actions continue
to flow through `run_python`.

## Discovery And Enablement

uv-agent discovers plugins through the Python entry point group
`uv_agent.plugins` in the host environment. A plugin distribution declares an
entry point in its `pyproject.toml`:

```toml
[project.entry-points."uv_agent.plugins"]
my-plugin = "my_plugin"
```

The entry point should resolve to a module or object containing pluggy hook
implementations. Discovered plugins are enabled by default and are started in the
background when an agent turn begins. Disable an installed plugin with config:

```json
{
  "plugins": {
    "disabled": ["my-plugin"],
    "config": {
      "another-plugin": {
        "option": "value"
      }
    }
  }
}
```

Config can live in user config (`~/.uv-agent/config.json`) or project config
(`.uv-agent/config.json`) and follows the normal config merge rules. The
`plugins.config.<entry-point-name>` object is passed to that plugin as
`context.config`.

uv-agent records first-seen plugin names in:

```text
~/.uv-agent/plugins/registry.sqlite3
```

A plugin's private data and logs live under:

```text
~/.uv-agent/plugins/<plugin-name>/
  data.sqlite3
  logs/plugin.log
```

## Minimal Plugin

```python
# my_plugin.py
from __future__ import annotations

import pluggy

hookimpl = pluggy.HookimplMarker("uv_agent")


@hookimpl
async def uv_agent_start(context):
    """Register capabilities when uv-agent starts the plugin."""

    def shout(text: str) -> dict[str, str]:
        return {"text": text.upper()}

    context.register_runtime_helper(
        "shout",
        shout,
        doc="Return the provided text in uppercase. Usage: shout(text)",
    )


@hookimpl
async def uv_agent_stop(context):
    """Release plugin resources before uv-agent exits."""

    context.logger.info("stopped")
```

Package metadata:

```toml
[project]
name = "my-uv-agent-plugin"
version = "0.1.0"
dependencies = ["pluggy>=1.6.0"]

[project.entry-points."uv_agent.plugins"]
my-plugin = "my_plugin"
```

After installing the package into the environment used to run uv-agent, managed
scripts can use the helper:

```python
from uv_agent_runtime import shout

print(shout("done"))
```

When a helper is registered, uv-agent adds it to the dynamic runtime context as a
`<plugin_runtime_helpers>` section so the model knows it can import it from
`uv_agent_runtime`.

## Lifecycle Hooks

Plugins use the pluggy project name `uv_agent`.

### `uv_agent_start(context)`

Called when the plugin starts. The implementation may be `async def`; uv-agent
awaits returned hook coroutines. Typical work includes registering helpers,
subscribing to events, starting background services, and initializing private
state.

If startup raises, uv-agent marks the plugin failed, writes the exception to the
plugin log, emits a `plugin.failed` event, and continues running.

### `uv_agent_stop(context)`

Called when the engine shuts down after a plugin has started or failed. Use it to
cancel background tasks, flush logs, or close external connections. Stop failures
are logged and reported as `plugin.failed` events.

## `PluginContext`

Each plugin receives a context object with these attributes:

| Attribute | Description |
| --- | --- |
| `name` | Entry point name of the plugin. |
| `project_root` | Current uv-agent workspace root. |
| `user_state_dir` | User state directory, normally `~/.uv-agent`. |
| `data_dir` | Private plugin directory: `~/.uv-agent/plugins/<plugin-name>/`. |
| `log_dir` | Plugin log directory: `~/.uv-agent/plugins/<plugin-name>/logs/`. |
| `config` | Dict from `plugins.config.<plugin-name>`. |
| `events` | Shared `EventBus` instance. |
| `logger` | Logger writing to the plugin log file. |

### `register_runtime_helper(name, fn, *, doc=None, schema=None)`

Registers a synchronous host function that managed scripts can call through
`uv_agent_runtime`.

```python
def notify(message: str, *, channel: str = "default") -> dict[str, bool]:
    send_to_service(channel, message)
    return {"ok": True}

context.register_runtime_helper(
    "notify",
    notify,
    doc="Send a message to the configured notification service.",
    schema={
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "channel": {"type": "string"}
        },
        "required": ["message"]
    },
)
```

Helper names must be valid Python identifiers. Duplicate helper names are
rejected. The `schema` is metadata for documentation/context; it is not currently
enforced as JSON Schema validation.

Runtime use:

```python
from uv_agent_runtime import notify

notify("build finished", channel="dev")
```

Under the hood, `uv_agent_runtime` resolves unknown imports by asking the host
via the runner RPC transport. Calls are sent back to the host as `call.<helper>`
requests with positional and keyword arguments.

### `open_db()`

Opens the plugin's private SQLite database at `data_dir / "data.sqlite3"` and
enables WAL mode, foreign keys, and the standard uv-agent busy timeout. The host
creates the directory and connection; the plugin owns its schema.

```python
with context.open_db() as db:
    db.execute(
        "CREATE TABLE IF NOT EXISTS seen_messages "
        "(id TEXT PRIMARY KEY, created_at TEXT NOT NULL)"
    )
```

### `submit_turn(...)`

Submits a user turn from plugin code and returns a `SubmittedTurn` handle:

```python
handle = await context.submit_turn(
    text="Summarize the latest webhook payload",
    thread_id=None,
    level=None,
    image_paths=None,
)

print(handle.thread_id, handle.turn_id)
async for event in handle.events():
    await relay_event(event)
```

Arguments:

| Argument | Description |
| --- | --- |
| `text` | User text for the new turn. |
| `thread_id` | Existing thread to continue, or `None` to create a new thread. |
| `level` | Optional model level name. |
| `image_paths` | Optional list of image paths to attach. |

`submit_turn` must not be called directly from inside an event handler's current
call stack. If a plugin wants to start another turn in reaction to an event, it
should schedule a new task, for example `asyncio.create_task(...)`, to avoid
recursive event loops.

## Event Bus

Plugins subscribe with `context.events.subscribe(...)`:

```python
@hookimpl
async def uv_agent_start(context):
    async def on_done(event):
        context.logger.info("turn %s completed", event.get("turn_id"))

    context.events.subscribe(
        "turn.completed",
        on_done,
        logger=context.logger,
    )
```

`subscribe` accepts a single event type or a list of event types. Optional
`thread_id` and `turn_id` filters restrict delivery. The return value is an
`unsubscribe()` function.

Event handlers must be async functions. They are scheduled with
`asyncio.create_task`; slow handlers do not block the main agent stream. Handler
exceptions are caught and written to the plugin logger.

Common public event types include:

| Event type | When it is emitted |
| --- | --- |
| `plugin.discovered` | uv-agent found a plugin entry point. |
| `plugin.first_load` | The plugin was seen for the first time on this machine. |
| `plugin.starting` / `plugin.started` | Plugin lifecycle progress. |
| `plugin.failed` | Plugin start/stop failed. |
| `plugin.stopping` / `plugin.stopped` | Plugin shutdown progress. |
| `turn.started` | A user turn starts. |
| `assistant.delta` | Assistant text streaming delta. |
| `assistant.reasoning_delta` | Reasoning text streaming delta, when available. |
| `tool.delta` | Tool-call argument streaming delta. |
| `tool.started` | A `run_python` call starts. |
| `tool.partial` | Partial runner output before final completion. |
| `tool.output` | Final `run_python` result. |
| `image.attachment` | An image is attached to the turn. |
| `thread.title` | A generated thread title is available. |
| `compaction.started` / `compaction.completed` | Context compaction progress. |
| `turn.completed` | A turn completed normally. |
| `turn.error` | A turn ended with an error. |
| `turn.interrupted` | A turn was interrupted. |

Event payloads are the same public dictionaries consumed by the CLI/TUI stream.
Treat them as versioned project API: prefer checking `event.get("type")` and
optional keys rather than assuming every field is present.

## Security Model And Limitations

- Plugins execute inside the uv-agent host process with the same permissions as
  uv-agent itself. Install only trusted packages.
- The plugin system does not sandbox code, isolate dependencies, or restrict
  network/file access.
- Plugin runtime helpers are synchronous host functions. Wrap async SDKs behind a
  synchronous function if a helper must call them.
- Plugins are discovered at startup/first turn; there is no hot reload.
- Plugins cannot inject TUI commands, alter model messages, or bypass
  `run_python` as the model's action surface.
- Avoid logging secrets. uv-agent stores plugin logs under user state, but plugin
  code is responsible for redacting its own sensitive values.
