from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from uv_agent.config import PluginConfigBlock, PluginsConfig
from uv_agent.plugins import EventBus, PluginManager, PluginManifest, SetupPlugin
from uv_agent.plugins.context import PluginContextBroker
from uv_agent.plugins.registry import RuntimeFunctionSpec, RuntimeNamespaceRegistry
from uv_agent.plugins.storage import PluginStorage
from uv_agent.plugins.xml import XmlContribution, render_contribution, render_update_envelope


class EntryPoint:
    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self._value = value

    def load(self) -> object:
        return self._value


def _plugin(plugin_id: str, setup, *, priority: int = 100, dependencies: tuple[str, ...] = ()) -> SetupPlugin:
    return SetupPlugin(
        manifest=PluginManifest(
            id=plugin_id,
            version="0.1.0",
            display_name=plugin_id,
            description="Test plugin",
            priority=priority,
            dependencies=dependencies,
        ),
        setup=setup,
    )


def _install_entry_points(monkeypatch: pytest.MonkeyPatch, plugins: list[SetupPlugin]) -> None:
    monkeypatch.setattr(
        "uv_agent.plugins.manager.importlib.metadata.entry_points",
        lambda group: [EntryPoint(plugin.manifest.id, plugin) for plugin in plugins]
        if group == "uv_agent.plugins"
        else [],
    )
    # Most tests want exact entry-point fixtures, not any builtin plugins that
    # may be present in the checkout while the refactor is underway.
    monkeypatch.setattr("uv_agent.plugins.manager._builtin_plugins", lambda: [])


def _manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plugins: list[SetupPlugin], *, config: PluginsConfig | None = None) -> PluginManager:
    _install_entry_points(monkeypatch, plugins)
    return PluginManager(
        config=config or PluginsConfig(),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeNamespaceRegistry(),
        submitter=None,
        thread_store=None,
        user_state_dir=tmp_path / "state",
    )


@pytest.mark.asyncio
async def test_builtin_plugins_publish_context_and_runtime_namespaces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("uv_agent.plugins.manager.importlib.metadata.entry_points", lambda group: [])
    skill_dir = tmp_path / ".agents" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo\nUse this skill for demos.\n", encoding="utf-8")
    agents_dir = tmp_path / ".agents"
    agents_dir.mkdir(exist_ok=True)
    (agents_dir / "mcp.json").write_text(
        '{"servers":{"files":{"command":"python","description":"File helpers"}}}',
        encoding="utf-8",
    )
    manager = PluginManager(
        config=PluginsConfig(),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeNamespaceRegistry(),
        submitter=None,
        thread_store=None,
        user_state_dir=tmp_path / "state",
    )

    await manager.start()

    states = {record.id: record.state for record in manager.records}
    assert states["builtin.skills"] == "started"
    assert states["builtin.mcp"] == "started"
    assert states["builtin.workflow"] == "started"
    assert states["builtin.scheduler"] == "started"
    text = manager.contexts.full_context_text("thr", core_texts=[])
    assert "<agent_available_skills>" in text
    assert "<name>demo</name>" in text
    assert "<agent_available_mcp_servers>" in text
    assert "<name>files</name>" in text
    assert manager.resolve_helper("mcp")["transport"] == "local_module"
    assert manager.resolve_helper("workflow")["transport"] == "local_module"
    assert manager.resolve_helper("scheduler")["transport"] == "local_module"
    assert manager.resolve_action("workflow.prompt")["found"] is True


@pytest.mark.asyncio
async def test_plugin_manager_starts_setup_plugin_and_registers_capabilities(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    async def setup(context) -> None:
        seen.append(context.config)
        context.runtime.register_namespace(
            "demo",
            doc="Demo runtime namespace.",
            functions={"greet": lambda payload: {"hello": payload["name"]}},
            docs={"greet": "Greet somebody."},
            schemas={
                "greet": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
        )
        context.actions.register(
            "demo.echo",
            lambda payload: {"echo": payload["text"]},
            schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        )
        context.commands.register("/demo", lambda payload: {"command": payload.get("value")}, description="Demo command.")
        context.context.epoch.publish(tag="demo_status", body={"state": "ready"})

    manager = _manager(
        tmp_path,
        monkeypatch,
        [_plugin("demo-plugin", setup)],
        config=PluginsConfig(entries={"demo-plugin": PluginConfigBlock(config={"answer": 42})}),
    )

    await manager.start()

    assert seen == [{"answer": 42}]
    assert manager.records[0].state == "started"
    namespace = manager.resolve_helper("demo")
    assert namespace["found"] is True
    assert namespace["kind"] == "namespace"
    assert namespace["functions"][0]["full_name"] == "demo.greet"
    function = manager.resolve_helper("demo.greet")
    assert function["schema"]["required"] == ["name"]
    assert await manager.call_helper("demo.greet", kwargs={"name": "Ada"}) == {"hello": "Ada"}
    with pytest.raises(ValueError):
        await manager.call_helper("demo.greet", kwargs={})
    assert await manager.call_action("demo.echo", {"text": "ok"}) == {"echo": "ok"}
    assert manager.call_command("/demo", {"value": 7}) == {"command": 7}
    assert manager.contexts.full_context_text("thread", core_texts=["<core />"]).endswith(
        "<agent_demo_status>\n<state>ready</state>\n</agent_demo_status>"
    )


@pytest.mark.asyncio
async def test_plugin_config_can_disable_plugin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    started = False

    def setup(_context) -> None:
        nonlocal started
        started = True

    manager = _manager(
        tmp_path,
        monkeypatch,
        [_plugin("disabled-plugin", setup)],
        config=PluginsConfig(entries={"disabled-plugin": PluginConfigBlock(enabled=False)}),
    )

    await manager.start()

    assert started is False
    assert manager.records[0].state == "disabled"


@pytest.mark.asyncio
async def test_plugin_manager_orders_dependencies_and_isolates_setup_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    order: list[str] = []

    def setup_base(context) -> None:
        order.append(context.plugin_id)
        context.runtime.register_namespace("base", functions={"ok": lambda payload: {"ok": True}})

    def setup_child(context) -> None:
        order.append(context.plugin_id)
        context.runtime.register_namespace("base", functions={"boom": lambda payload: {}})

    manager = _manager(
        tmp_path,
        monkeypatch,
        [
            _plugin("child", setup_child, priority=1, dependencies=("base",)),
            _plugin("base", setup_base, priority=50),
        ],
    )

    await manager.start()

    assert order == ["base", "child"]
    states = {record.id: record.state for record in manager.records}
    assert states == {"base": "started", "child": "failed"}
    child = next(record for record in manager.records if record.id == "child")
    assert child.error_type == "ValueError"
    assert "already registered" in child.message


@pytest.mark.asyncio
async def test_plugin_context_tracks_background_task_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events = EventBus()
    published: list[dict[str, object]] = []

    async def on_failed(event):
        published.append(event)

    events.subscribe("plugin.task_failed", on_failed)

    async def setup(context) -> None:
        async def boom() -> None:
            raise RuntimeError("task boom")

        context.create_task(boom(), name="boom-task")

    _install_entry_points(monkeypatch, [_plugin("task-plugin", setup)])
    manager = PluginManager(
        config=PluginsConfig(),
        project_root=tmp_path,
        events=events,
        helper_registry=RuntimeNamespaceRegistry(),
        submitter=None,
        thread_store=None,
        user_state_dir=tmp_path / "state",
    )

    await manager.start()
    for _ in range(20):
        await asyncio.sleep(0.01)
        await events.drain()
        if manager.records[0].state == "warning":
            break

    assert manager.records[0].state == "warning"
    assert published and published[0]["type"] == "plugin.task_failed"


def test_runtime_namespace_registry_validates_names_schemas_and_reserved_names() -> None:
    registry = RuntimeNamespaceRegistry(reserved={"file"})
    with pytest.raises(ValueError):
        registry.register_namespace(plugin="p", namespace="file", functions={"x": lambda payload: {}})
    with pytest.raises(ValueError):
        registry.register_namespace(plugin="p", namespace="bad-name", functions={})
    with pytest.raises(ValueError):
        registry.register_namespace(
            plugin="p",
            namespace="demo",
            functions=(RuntimeFunctionSpec(namespace="demo", name="x", plugin="p", doc="", schema={"type": "string"}, fn=lambda payload: {}),),
        )
    with pytest.raises(ValueError):
        registry.register_namespace(plugin="p", namespace="demo", functions={"x": None})  # type: ignore[arg-type]


def test_plugin_storage_kv_collection_and_indexes(tmp_path: Path) -> None:
    storage = PluginStorage(
        plugin_id="demo",
        project_data_dir=tmp_path / "project",
        global_data_dir=tmp_path / "global",
        indexes={"messages": ("chat.id", "kind")},
    )
    kv = storage.project_kv()
    kv.set("settings/theme", {"name": "dark"})
    assert kv.get("settings/theme") == {"name": "dark"}
    assert kv.update_json("settings/theme", {"font": "mono"}) == {"name": "dark", "font": "mono"}
    assert kv.list_prefix("settings/")[0]["key"] == "settings/theme"

    collection = storage.thread_collection("thread_1", "messages")
    collection.put("doc1", {"chat": {"id": "chat:1"}, "kind": "text", "body": "hello"})
    collection.put("doc2", {"chat": {"id": "chat:2"}, "kind": "image", "body": "photo"})

    assert collection.get("doc1")["body"] == "hello"
    assert [item["doc_id"] for item in collection.query_index("chat.id", "chat:1")] == ["doc1"]
    assert [item["doc_id"] for item in collection.query_index("kind", "image")] == ["doc2"]
    assert collection.delete("doc1") == {"doc_id": "doc1", "deleted": True}
    assert collection.get("doc1") is None


def test_xml_renderer_prefixes_top_level_and_escapes_values() -> None:
    rendered = render_contribution(
        "goal_mode",
        {"objective": "A&B", "rules": ["one", "two"], "skip": None},
        attrs={"status": "enabled", "active": True},
    )

    assert rendered.startswith('<agent_goal_mode active="true" status="enabled">')
    assert "<objective>A&amp;B</objective>" in rendered
    assert "<rules>\n<item>one</item>\n<item>two</item>\n</rules>" in rendered
    assert "skip" not in rendered
    with pytest.raises(ValueError):
        render_contribution("bad tag", {})


def test_context_broker_renders_full_update_turn_and_replay() -> None:
    broker = PluginContextBroker()
    broker.publish(plugin="p", tag="status", body={"state": "ready"})

    full = broker.full_context_text("thread", core_texts=["<agent_core />"])
    assert full == "<agent_core />\n\n<agent_status>\n<state>ready</state>\n</agent_status>"
    assert broker.update_context_text("thread") == ""

    broker.update(plugin="p", tag="status", body={"state": "running"})
    update = broker.update_context_text("thread")
    assert update.startswith("<agent_epoch_context_update>")
    assert '<agent_status operation="update">' in update
    assert "<state>running</state>" in update

    broker.enqueue_turn(
        plugin="p",
        thread_id="thread",
        tag="notice",
        body={"message": "check"},
        replay_after_compaction=True,
        replay_key="notice",
    )
    assert broker.turn_context_text("thread") == "<agent_notice>\n<message>check</message>\n</agent_notice>"
    broker.replay_after_compaction("thread")
    assert broker.turn_context_text("thread") == "<agent_notice>\n<message>check</message>\n</agent_notice>"
    assert broker.turn_context_text("thread") == ""


def test_render_update_envelope_batches_contributions() -> None:
    text = render_update_envelope([
        XmlContribution("skills", {"skill": {"name": "demo"}}, attrs={"operation": "publish"}),
        XmlContribution("mcp", {"reason": "removed"}, attrs={"operation": "remove"}),
    ])

    assert text.startswith("<agent_epoch_context_update>")
    assert '<agent_skills operation="publish">' in text
    assert '<agent_mcp operation="remove">' in text


def test_thread_store_external_thread_mapping(tmp_path: Path) -> None:
    from uv_agent.session import ThreadStore

    store = ThreadStore(tmp_path / "state")
    assert store.get_external_thread(owner_plugin="telegram", source="telegram", external_id="chat:1") is None

    thread_id = store.get_or_create_external_thread(
        owner_plugin="telegram",
        source="telegram",
        external_id="chat:1",
        title="Telegram chat",
        metadata={"custom": "value"},
    )

    assert store.get_external_thread(owner_plugin="telegram", source="telegram", external_id="chat:1") == thread_id
    assert store.get_or_create_external_thread(
        owner_plugin="telegram",
        source="telegram",
        external_id="chat:1",
        title="Other",
    ) == thread_id
    metadata = store.thread_metadata(thread_id)
    assert metadata["owner_type"] == "plugin"
    assert metadata["owner_plugin"] == "telegram"
    assert metadata["external_source"] == "telegram"
    assert metadata["external_id"] == "chat:1"
    assert metadata["custom"] == "value"


@pytest.mark.asyncio
async def test_plugin_event_bus_supports_wildcards() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def on_event(event):
        seen.append(event["type"])

    bus.subscribe(["turn.*", "plugin.started"], on_event)
    bus.publish({"type": "turn.started"})
    bus.publish({"type": "tool.started"})
    bus.publish({"type": "plugin.started"})
    await bus.drain()

    assert seen == ["turn.started", "plugin.started"]
