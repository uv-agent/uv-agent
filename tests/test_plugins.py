from __future__ import annotations

from pathlib import Path
import asyncio

import pluggy
import pytest

from uv_agent.config import PluginsConfig
from uv_agent.plugins import EventBus, PluginManager, TurnContextBlock, TurnPrepareRequest
from uv_agent.plugins.helpers import RuntimeHelperRegistry


def _request() -> TurnPrepareRequest:
    return TurnPrepareRequest(
        thread_id="thr_demo",
        turn_id="turn_demo",
        user_text="hello",
        level=None,
        is_new_thread=True,
        is_first_turn=True,
        created_at="2026-06-22T12:00:00+00:00",
        metadata={},
    )


@pytest.mark.asyncio
async def test_plugin_manager_prepare_turn_collects_pre_user_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hookimpl = pluggy.HookimplMarker("uv_agent")
    seen: list[tuple[str, str]] = []

    class DemoPlugin:
        @hookimpl
        async def uv_agent_prepare_turn(self, context, request):
            seen.append((context.name, request.user_text))
            return [
                TurnContextBlock("current time", dedupe_key="time"),
                {"text": "dict block", "dedupe_key": "dict"},
            ]

    class EntryPoint:
        name = "demo-plugin"

        def load(self):
            return DemoPlugin()

    monkeypatch.setattr(
        "uv_agent.plugins.manager.importlib.metadata.entry_points",
        lambda group: [EntryPoint()] if group == "uv_agent.plugins" else [],
    )
    manager = PluginManager(
        config=PluginsConfig(),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeHelperRegistry(),
        submitter=None,
        user_state_dir=tmp_path / "state",
    )

    await manager.start()
    blocks = await manager.prepare_turn(_request())

    assert seen == [("demo-plugin", "hello")]
    assert [(block.plugin, block.text, block.dedupe_key) for block in blocks] == [
        ("demo-plugin", "current time", "time"),
        ("demo-plugin", "dict block", "dict"),
    ]


@pytest.mark.asyncio
async def test_plugin_manager_prepare_turn_isolates_hook_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hookimpl = pluggy.HookimplMarker("uv_agent")
    published: list[dict[str, object]] = []

    class BadPlugin:
        @hookimpl
        async def uv_agent_prepare_turn(self, context, request):
            raise RuntimeError("boom")

    class EntryPoint:
        name = "bad-plugin"

        def load(self):
            return BadPlugin()

    events = EventBus()

    async def on_failed(event):
        published.append(event)

    events.subscribe("plugin.hook_failed", on_failed)
    monkeypatch.setattr(
        "uv_agent.plugins.manager.importlib.metadata.entry_points",
        lambda group: [EntryPoint()] if group == "uv_agent.plugins" else [],
    )
    manager = PluginManager(
        config=PluginsConfig(),
        project_root=tmp_path,
        events=events,
        helper_registry=RuntimeHelperRegistry(),
        submitter=None,
        user_state_dir=tmp_path / "state",
    )

    await manager.start()
    blocks = await manager.prepare_turn(_request())
    await events.drain()

    assert blocks == []
    assert published
    assert published[0]["type"] == "plugin.hook_failed"
    assert published[0]["plugin"] == "bad-plugin"
    assert published[0]["hook"] == "uv_agent_prepare_turn"


@pytest.mark.asyncio
async def test_plugin_context_registers_and_calls_handler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hookimpl = pluggy.HookimplMarker("uv_agent")

    class HandlerPlugin:
        @hookimpl
        async def uv_agent_start(self, context):
            context.register_handler(
                "demo_handler",
                lambda payload: {"hello": payload["name"]},
                doc="Greet somebody.",
                schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            )

    class EntryPoint:
        name = "handler-plugin"

        def load(self):
            return HandlerPlugin()

    monkeypatch.setattr(
        "uv_agent.plugins.manager.importlib.metadata.entry_points",
        lambda group: [EntryPoint()] if group == "uv_agent.plugins" else [],
    )
    manager = PluginManager(
        config=PluginsConfig(),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeHelperRegistry(),
        submitter=None,
        user_state_dir=tmp_path / "state",
    )

    await manager.start()

    resolved = manager.resolve_helper("demo_handler")
    assert resolved["found"] is True
    assert resolved["schema"]["required"] == ["name"]
    assert await manager.call_helper("demo_handler", kwargs={"name": "Ada"}) == {"hello": "Ada"}
    with pytest.raises(ValueError):
        await manager.call_helper("demo_handler", kwargs={})


def test_handler_registration_requires_doc_and_schema(tmp_path: Path) -> None:
    registry = RuntimeHelperRegistry()
    with pytest.raises(ValueError):
        registry.register_handler(plugin="p", name="h", fn=lambda payload: None, doc="", schema={"type": "object"})
    with pytest.raises(ValueError):
        registry.register_handler(plugin="p", name="h", fn=lambda payload: None, doc="doc", schema={"type": "string"})


@pytest.mark.asyncio
async def test_plugin_context_tracks_background_task_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hookimpl = pluggy.HookimplMarker("uv_agent")
    published: list[dict[str, object]] = []

    class TaskPlugin:
        @hookimpl
        async def uv_agent_start(self, context):
            async def boom():
                raise RuntimeError("task boom")

            context.create_task(boom(), name="boom-task")

    class EntryPoint:
        name = "task-plugin"

        def load(self):
            return TaskPlugin()

    events = EventBus()

    async def on_failed(event):
        published.append(event)

    events.subscribe("plugin.task_failed", on_failed)
    monkeypatch.setattr(
        "uv_agent.plugins.manager.importlib.metadata.entry_points",
        lambda group: [EntryPoint()] if group == "uv_agent.plugins" else [],
    )
    manager = PluginManager(
        config=PluginsConfig(),
        project_root=tmp_path,
        events=events,
        helper_registry=RuntimeHelperRegistry(),
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
