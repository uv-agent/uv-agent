# Plugin System

`uv-agent` plugins are trusted Python packages loaded into the host process.
They extend the host without changing the agent boundary: the model still acts
through `run_python`, and plugin capabilities appear as runtime helpers,
commands, actions, UI providers, i18n text, and model-visible context.

The API is intentionally small and breaking changes are allowed while the project
is experimental. Install only plugins you trust.

## Discovery

Plugins are discovered from the Python entry point group `uv_agent.plugins`:

```toml
[project.entry-points."uv_agent.plugins"]
my_plugin = "my_plugin:plugin"
```

The entry point must return a `SetupPlugin`:

```python
from uv_agent.plugins import PluginManifest, SetupPlugin

MANIFEST = PluginManifest(
    id="my.plugin",
    version="0.1.0",
    display_name="My Plugin",
    description="Small example plugin.",
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup, stop=stop)
```

Installed plugins are enabled by default. Disable or configure them in user or
project config:

```json
{
  "plugins": {
    "my.plugin": {
      "enabled": false,
      "config": {
        "option": "value"
      }
    }
  }
}
```

The per-plugin config object is exposed as `context.config`.

## Manifest

`PluginManifest` is a static, data-only declaration consumed before `setup`
runs. Use it for anything the host needs to know before starting the plugin.

```python
from uv_agent.plugins import PluginManifest

MANIFEST = PluginManifest(
    id="my.plugin",
    version="0.1.0",
    display_name={"en": "My Plugin", "zh": "我的插件"},
    description={"en": "Small example plugin.", "zh": "小型示例插件。"},
    builtin=False,
    default_enabled=True,
    priority=100,
    dependencies=(),
    optional_dependencies=(),
    capabilities=("runtime_namespace", "context", "command", "action", "ui", "storage"),
    activation="always",
    config_schema={},
    storage_schema={"collections": {"messages": {"indexes": ["channel"]}}},
)
```

| Field | Purpose |
| --- | --- |
| `id` | Unique plugin id. Use dotted names. |
| `version` | Plugin version string. |
| `display_name` | Short name; string or `{language: text}` map. |
| `description` | Long description; string or localized map. |
| `builtin` | True for built-in plugins. |
| `default_enabled` | Whether the plugin is enabled when first seen. |
| `priority` | Load order hint; lower numbers load first. |
| `dependencies` | Other plugin ids that must start first. |
| `optional_dependencies` | Optional ids used for load-order hints only. |
| `capabilities` | Self-declared capability strings for documentation. |
| `activation` | Host lifecycle policy: `always`, `persistent_only`, or `session_only`. |
| `config_schema` | Optional JSON schema for validating `context.config`. |
| `storage_schema` | Declares document collections and indexed fields. |

`activation` is a coarse host-side startup policy. Use `persistent_only` for
long-lived services such as schedulers or fixed-port servers that should not run
inside a short TUI session host. Use `session_only` for plugins that only make
sense in an interactive session. The default, `always`, preserves compatibility
and lets the plugin inspect `context.host` to choose its own degraded or
ephemeral behavior.

## Lifecycle

`SetupPlugin.setup(context)` is called once when the plugin starts. It may be a
normal function or an async function. Use it to register capabilities,
initialize storage, subscribe to events, and start background tasks.

`SetupPlugin.stop(context)` is optional and may also be async. Use it to dispose
background tasks or external resources. If setup or stop fails, uv-agent logs the
failure and emits plugin lifecycle events without taking down the host.

## Context

`setup(context)` receives a `PluginContext` with these primary surfaces:

| Surface | Purpose |
| --- | --- |
| `context.manifest` | The plugin manifest. |
| `context.host` | Read-only host invocation, lifecycle, and state directory information. |
| `context.project_root` | Active workspace root. |
| `context.user_state_dir` | User state directory, normally `~/.uv-agent`. |
| `context.config` | Plugin config from `plugins.<id>.config`. |
| `context.logger` | Plugin log writer. |
| `context.events` | Event bus subscription API. |
| `context.runtime` | Runtime helper namespace registry. |
| `context.resources` | URI resource prefix registry for read-only plugin resources. |
| `context.actions` | Host action registry. |
| `context.commands` | TUI command registry. |
| `context.ui` | UI provider registry. |
| `context.epoch` | Model-visible epoch context API. |
| `context.turn` | Model-visible turn context API. |
| `context.i18n` | Plugin-owned localization registry. |
| `context.storage` | SQLite-backed KV and document stores. |
| `context.threads` | Narrow thread creation, metadata, and event API, when available. |
| `context.submit_turn` | Programmatic turn submission, when configured. Returns a waitable `SubmittedTurn`. |
| `context.create_task` | Launch a plugin-owned background `asyncio.Task`. |
| `context.compaction` | Add compact handoff sections after compaction. |

`context.logger` writes to `~/.uv-agent/plugins/<plugin-id>/logs/plugin.log`.
Plugin logs use the top-level `logging.max_bytes` and `logging.backup_count`
rotation settings; see [Configuration](configuration.md#logging-options).

`context.host.invocation` is currently `tui` or `daemon`.
`context.host.lifetime` is `session` for the local TUI host and `persistent` for
daemon mode. `context.host.is_persistent` is the preferred check for plugins
that need to decide whether to start long-running resources. Host mode is not
automatically added to model context; publish plugin context only for the
capabilities the model should actually use.

## Minimal Plugin

```python
from uv_agent.plugins import PluginManifest, SetupPlugin

MANIFEST = PluginManifest(
    id="my.plugin",
    version="0.1.0",
    display_name="My Plugin",
    description="Example",
    capabilities=("runtime_namespace", "context"),
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


def setup(context) -> None:
    context.runtime.register_namespace(
        "demo",
        doc="Small demo helpers.",
        functions={
            "shout": lambda text: {"text": str(text).upper()},
        },
        docs={"shout": "Return the input text in uppercase."},
        schemas={"shout": {"type": "object"}},
    )
    context.epoch.publish(
        tag="demo_helpers",
        body={
            "helper": {
                "import": "import uv_agent_runtime as rt",
                "namespace": "rt.demo",
                "signature": "rt.demo.shout(text: str) -> dict[str, str]",
            }
        },
    )
```

Managed scripts can then call:

```python
import uv_agent_runtime as rt

print(rt.demo.shout("done"))
```

## Runtime Helpers

Register helpers with `context.runtime.register_namespace(...)`. Namespaces are
globally unique and appear under `uv_agent_runtime` as `rt.<namespace>`.

```python
def notify(message: str, *, channel: str = "default") -> dict[str, bool]:
    send_to_service(channel, message)
    return {"ok": True}


context.runtime.register_namespace(
    "notify",
    doc="Notification helpers.",
    functions={"send": notify},
    docs={"send": "Send a message."},
    schemas={"send": {"type": "object"}},
    module="my_plugin.runtime",
)
```

If `module` is provided, `uv_agent_runtime` imports that module as the script-side
facade. The facade should expose natural Python functions and call host helpers
through `uv_agent_runtime.transport.call_host(...)`. The host callable receives
natural positional and keyword arguments. If it accepts a `context` keyword, the
runner injects the current run context.

Helper arguments and return values must be JSON-compatible. Every helper
function must be associated with a `{"type": "object"}` schema. You may also pass
`RuntimeFunctionSpec` objects to set per-function `timeout_s`.

### Reserved namespaces

The following runtime namespaces are reserved by core and cannot be registered
by plugins:

```
file, files, search, symbols, query, patch, apply_patch, diff, compare,
snapshot, restore, transaction, run, deps, cd, pwd, path, events, look_at,
threads, get, blob
```

Builtin plugins also register the namespaces `goal`, `mcp`, `scheduler`,
`skills`, `workflow`, and `worktree`. Third-party plugins should avoid all of
these names to prevent conflicts.

## Resources

Plugins can register read-only URI resources without adding a model-visible
runtime namespace:

```python
from uv_agent.plugins import ResourceData

context.resources.register(
    prefix="demo://",
    read=lambda uri: ResourceData(uri=uri, kind="text", text="hello"),
)
```

The core registry only routes by URI prefix; resource contents belong to the
plugin. Readers may return `ResourceData`, `str`, `bytes`, `Path`, or a dict
with exactly one of `text`, `data`, or `path`.

## Actions

Actions are host operations that UI, scheduler, or other plugins can call without
coupling to a builtin implementation.

```python
async def send_action(payload: dict, context=None) -> dict:
    message = str(payload["message"])
    await send_async(message)
    return {"ok": True}


context.actions.register(
    "notify.send",
    send_action,
    doc="Send a notification.",
    schema={
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    },
)
```

Actions receive a single payload object and return serializable data. In-process
plugin callers may pass richer values such as bytes; runtime helper calls still
cross JSON-RPC and must stay JSON-compatible. If the handler accepts `context`,
uv-agent injects the plugin context.

Plugins may also call registered actions through the same surface:

```python
info = context.actions.resolve("notify.send")
if info["found"]:
    await context.actions.call("notify.send", {"message": "done"})
```

Passing `context=...` to `call(...)` overrides the injected target plugin
context. Scheduler uses this to pass a schedule execution context to actions
without importing the target plugin.

Pass `missing="ignore"` to `context.actions.call(...)` when an optional action
should be skipped if its provider plugin is disabled.

## Commands And UI

Plugins can register slash commands and UI picker providers. Commands return
structured UI actions such as transcript messages or composer edits:

```python
from uv_agent.plugins import CommandResult, TranscriptAction


def hello_command(payload: dict, context=None) -> CommandResult:
    name = str(payload.get("arg") or "world")
    return CommandResult((TranscriptAction("event", f"hello {name}"),))


context.commands.register("/hello", hello_command, description="Say hello")
```

The TUI calls command/action registries rather than importing builtin modules.
Builtin plugins use the same surfaces as third-party plugins.

Slash command handlers are invoked from the synchronous TUI path, so they must
be synchronous. Async command handlers will raise `RuntimeError`.

UI picker providers can be synchronous or async and return an iterable of
`PickerItem` values or plain dicts:

```python
context.ui.picker(
    id="my.items",
    title="My Items",
    provider=lambda query="": [{"value": f"item:{i}", "description": f"item {i}"} for i in range(5)],
    trigger="@my",
)
```

## Model Context

Plugins publish model-visible context through `context.epoch` and `context.turn`.
The core broker does not keep semantic document state; it faithfully renders the
plugin contribution to XML and queues it for the appropriate thread or epoch.
State and diff decisions belong to the plugin.

Epoch context is sent after compaction and when the plugin refreshes an epoch:

```python
context.epoch.publish(
    tag="available_tools",
    body={"items": ["alpha", "beta"]},
)

context.epoch.update(
    tag="available_tools",
    body={"items": ["gamma"]},
)

context.epoch.remove(tag="available_tools", reason="Plugin disabled")
```

`context.epoch.on_refresh(callback)` registers a refresh callback. uv-agent calls
it once after setup. If setup already published epoch context for the plugin, the
first refresh output is silently discarded; later refreshes are queued normally.
The callback may accept `thread_id`.

Turn context is used for information tied to a specific turn. Only turn context
registered with `replay_after_compaction=True` is replayed after compaction.
Replay requires a stable `replay_key`; enqueueing another message with the same
key replaces the pending message, and `clear_replay(...)` removes it:

```python
context.turn.enqueue(
    thread_id=thread_id,
    tag="external_notice",
    body={"text": "Webhook payload is available."},
    replay_after_compaction=True,
    replay_key="webhook-latest",
)
```

Plugins can add compact handoff text after compaction with
`context.compaction.summary_section(provider)`. The provider is called with a
keyword argument and should return an empty string when it has nothing to add:

```python
def active_jobs_section(*, thread_id: str) -> str:
    return "## Active jobs\n- job_123 waiting for review"


context.compaction.summary_section(active_jobs_section)
```

## i18n

Plugin text lives with the plugin. Register localized strings during setup:

```python
TEXTS = {
    "hello_status": {"en": "Hello", "zh": "你好"},
}

context.i18n.register(TEXTS)
```

UI code can resolve plugin text through the plugin i18n registry. Do not add
plugin-owned copy to core `i18n.py`.

## Threads

`context.threads` exposes the thread data that plugins are allowed to touch.
Use `create_thread(...)` for plugin-owned child threads, and `record_event(...)`
for plugin-owned thread state changes that should appear in history or notify
other subscribers:

```python
child_id = context.threads.create_thread(
    "Worker",
    kind="plugin_worker",
    parent_thread_id=parent_id,
)
context.threads.record_event(
    child_id,
    "thread.example_updated",
    status="active",
)
```

Plugins can read metadata with `metadata(thread_id)` and update narrow metadata
patches with `update_metadata(thread_id, {...})`. Plugins own the meaning of
their metadata; the host persists and exposes it without interpreting plugin
event types. Do not import or depend on the host `ThreadStore` implementation
directly.

Plugins that need to ask the agent to do work can submit a turn and await it:

```python
submitted = await context.submit_turn(
    text="Summarize the queued payload.",
    thread_id=child_id,
    conflict="queue",
)
await submitted.wait()
if submitted.status == "completed":
    context.logger.info("turn result: %s", submitted.final_text)
```

`submit_turn` is async and cannot be called from inside a plugin event handler;
doing so raises `ReentrantSubmitError`. Use event handlers to react to state
changes and schedule work through `context.create_task` instead.

## Storage

`context.storage` exposes SQLite-backed KV and document stores scoped to the
plugin. Plugins own their schema and migrations.

```python
kv = context.storage.thread_kv(thread_id)
kv.set("last_seen", {"at": "2024-01-01T00:00:00Z"})
value = kv.get("last_seen")

messages = context.storage.thread_collection(thread_id, "messages")
messages.put("msg_1", {"channel": "general", "text": "hello"})
for item in messages.query_index("channel", "general"):
    print(item["body"])
```

Available scopes are `global`, `project`, and `thread`. Declare indexed
collections in `manifest.storage_schema` so the host can create the right indexes:

```python
storage_schema={
    "collections": {
        "messages": {"indexes": ["channel", {"field": "meta.kind"}]},
    }
}
```

Use plugin storage for plugin state. Do not rely on host internals unless the
data is exposed through `context.threads`, actions, or runtime helpers.

## Background Tasks

Use `context.create_task` to start plugin-owned background work that should be
cancelled when the plugin stops:

```python
import asyncio


async def poll():
    while True:
        await asyncio.sleep(60)
        await check_queue()


context.create_task(poll(), name="poll")
```

Tasks started this way are tracked per plugin and cancelled during shutdown.
Uncaught task exceptions mark the plugin status as `warning` and publish a
`plugin.task_failed` event.

## Events

Subscribe to host events with `context.events.subscribe(...)`:

```python
def on_turn_completed(event: dict) -> None:
    context.logger.info("turn completed: %s", event.get("turn_id"))


unsubscribe = context.events.subscribe("turn.completed", on_turn_completed, logger=context.logger)
```

Handlers may be sync or async. Slow or failing handlers should not block the main
agent turn; exceptions are logged.

Plugin lifecycle events published by the plugin manager:

- `plugin.discovered`
- `plugin.first_load`
- `plugin.skipped`
- `plugin.starting`
- `plugin.started`
- `plugin.stopped`
- `plugin.failed`
- `plugin.warning`
- `plugin.task_failed`

Turn and thread events published by the agent and thread store:

- `turn.started`
- `turn.completed`
- `turn.error`
- `turn.interrupted`
- `assistant.delta`
- `tool.started`
- `tool.output`
- `compaction.started`
- `compaction.completed`
- `thread.event_stored`

`thread.event_stored` is emitted by the thread store when any event is persisted;
its payload includes the stored `event` object.

## Security

- Plugins execute in the uv-agent host process with the same permissions as the
  host.
- There is no dependency sandbox or file/network isolation.
- Plugins must not expose secrets in logs, model context, runtime helper returns,
  or action payloads.
- Plugins must not add direct model tools. Model-visible execution continues to
  flow through `run_python`.
