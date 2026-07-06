from __future__ import annotations

import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from uv_agent.blobs import BlobStore
from uv_agent.config import LoggingConfig, PluginConfigBlock, PluginsConfig, parse_config
from uv_agent.session import ThreadStore
from uv_agent.plugins import EventBus, PluginHostInfo, PluginManager, PluginManifest, PluginStatus, ResourceData, SetupPlugin
from uv_agent.plugins.context import PluginContextBroker
from uv_agent.plugins.registry import ActionRegistry, PickerSource, RuntimeFunctionSpec, RuntimeNamespaceRegistry, UiRegistry
from uv_agent.plugins.resources import ResourceRegistry, coerce_resource_data
from uv_agent.plugins.storage import PluginStorage
from uv_agent.plugins.summary import format_plugin_detail_lines, format_plugin_status_counts
from uv_agent.plugins.xml import XmlContribution, render_contribution, render_update_envelope


class EntryPoint:
    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self._value = value

    def load(self) -> object:
        return self._value


def _host_info(tmp_path: Path, *, invocation: str = "tui", lifetime: str = "session") -> PluginHostInfo:
    return PluginHostInfo(
        invocation=invocation,  # type: ignore[arg-type]
        lifetime=lifetime,  # type: ignore[arg-type]
        project_root=tmp_path,
        project_state_dir=tmp_path / "project-state",
        user_state_dir=tmp_path / "user-state",
    )


def _plugin(
    plugin_id: str,
    setup,
    *,
    priority: int = 100,
    dependencies: tuple[str, ...] = (),
    activation: str = "always",
) -> SetupPlugin:
    return SetupPlugin(
        manifest=PluginManifest(
            id=plugin_id,
            version="0.1.0",
            display_name=plugin_id,
            description="Test plugin",
            priority=priority,
            dependencies=dependencies,
            activation=activation,  # type: ignore[arg-type]
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


def _builtin_entry_points() -> list[EntryPoint]:
    from uv_agent.builtin.goal import plugin as goal_plugin
    from uv_agent.builtin.mcp import plugin as mcp_plugin
    from uv_agent.builtin.scheduler import plugin as scheduler_plugin
    from uv_agent.builtin.skills import plugin as skills_plugin
    from uv_agent.builtin.subagent import plugin as subagent_plugin
    from uv_agent.builtin.workflow import plugin as workflow_plugin
    from uv_agent.builtin.worktree import plugin as worktree_plugin

    return [
        EntryPoint("builtin_goal", goal_plugin),
        EntryPoint("builtin_worktree", worktree_plugin),
        EntryPoint("builtin_skills", skills_plugin),
        EntryPoint("builtin_mcp", mcp_plugin),
        EntryPoint("builtin_subagent", subagent_plugin),
        EntryPoint("builtin_workflow", workflow_plugin),
        EntryPoint("builtin_scheduler", scheduler_plugin),
    ]


def _manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plugins: list[SetupPlugin],
    *,
    config: PluginsConfig | None = None,
    logging_config: LoggingConfig | None = None,
    host: PluginHostInfo | None = None,
    blob_store: BlobStore | None = None,
    agent_config=None,
) -> PluginManager:
    _install_entry_points(monkeypatch, plugins)
    return PluginManager(
        config=config or PluginsConfig(),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeNamespaceRegistry(),
        submitter=None,
        thread_store=None,
        blob_store=blob_store,
        logging_config=logging_config,
        user_state_dir=tmp_path / "state",
        host=host,
        agent_config=agent_config,
    )


def _epoch_text(broker: PluginContextBroker, thread_id: str, *, core_texts: list[str] | None = None) -> str:
    parts = [text for text in (core_texts or []) if text]
    parts.extend(item.text for item in broker.consume_epoch(thread_id))
    return "\n\n".join(parts)


@pytest.mark.asyncio
async def test_plugin_agent_api_summarizes_models_and_pickers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def setup(context) -> None:
        seen["context"] = context
        context.ui.picker(
            id="skills",
            title={"zh": "技能", "en": "Skills"},
            trigger="@skill",
            provider=lambda query="": [
                {
                    "id": "skill://user/demo",
                    "value": "@skill://user/demo",
                    "description": "Demo skill",
                    "kind": "skill-mention",
                    "meta": "user",
                }
            ],
        )

    config = parse_config(
        {
            "providers": {"p": {"base_url": "https://example.com", "api_key": "secret"}},
            "models": {"m": {"provider": "p", "model": "remote", "context_window_tokens": 64000, "supports_images": True}},
            "levels": {"fast": {"model": "m"}, "hidden": {"model": "m", "hidden": True}},
            "runtime": {"default_level": "fast"},
        },
        tmp_path,
    )
    manager = _manager(
        tmp_path,
        monkeypatch,
        [_plugin("demo", setup)],
        agent_config=lambda: config,
    )

    await manager.start()

    context = seen["context"]
    model_summary = context.agent.model_levels()
    assert model_summary["available"] is True
    assert model_summary["default_level"] == "fast"
    assert [item["id"] for item in model_summary["levels"]] == ["fast"]
    assert model_summary["levels"][0]["model"] == "remote"
    assert model_summary["levels"][0]["provider_configured"] is True

    picker_summary = context.agent.picker_summary(["skills", "mcp"])
    assert picker_summary["skills"]["available"] is True
    assert picker_summary["skills"]["title"]["zh"] == "技能"
    assert picker_summary["skills"]["items"][0]["value"] == "@skill://user/demo"
    assert picker_summary["mcp"]["available"] is False


@pytest.mark.asyncio
async def test_builtin_plugins_publish_context_and_runtime_namespaces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "uv_agent.plugins.manager.importlib.metadata.entry_points",
        lambda group: _builtin_entry_points() if group == "uv_agent.plugins" else [],
    )
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
        config=PluginsConfig(entries={"builtin.mcp": PluginConfigBlock(enabled=True)}),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeNamespaceRegistry(),
        submitter=None,
        thread_store=None,
        user_state_dir=tmp_path / "state",
        host=_host_info(tmp_path, invocation="daemon", lifetime="persistent"),
    )

    try:
        await manager.start()

        states = {record.id: record.state for record in manager.records}
        assert states["builtin.goal"] == "started"
        assert states["builtin.worktree"] == "started"
        assert states["builtin.skills"] == "started"
        assert states["builtin.mcp"] == "warning"
        assert states["builtin.subagent"] == "started"
        assert states["builtin.workflow"] == "disabled"
        assert states["builtin.scheduler"] == "started"
        text = _epoch_text(manager.contexts, "thr")
        assert "<agent_goal_helpers>" not in text
        assert manager.resolve_helper("goal")["found"] is True
        assert manager.resolve_helper("goal.add_task")["found"] is True
        assert "<agent_available_skills>" in text
        assert "<name>demo</name>" in text
        assert "<agent_available_mcp_servers>" in text
        assert "<name>files</name>" in text
        assert "class McpClient" in text
        assert "rt.mcp.connect_named" in text
        assert "<agent_worktree_helpers>" not in text
        assert "<agent_subagent_helpers" in text
        assert "rt.subagent.run(prompt: str" in text
        assert 'action_id="subagent.prompt"' in text
        assert "<agent_workflow_context" not in text
        assert "<agent_scheduler_helpers>" in text
        assert "rt.scheduler.create(*, action_id: str | None" in text
        assert "<functions>" not in text
        assert manager.resolve_helper("mcp")["module"] == "uv_agent.builtin.mcp.runtime"
        assert manager.resolve_helper("worktree")["module"] == "uv_agent.builtin.worktree.runtime"
        subagent = manager.resolve_helper("subagent")
        assert subagent["module"] is None
        assert {item["name"] for item in subagent["functions"]} == {"run"}
        assert manager.resolve_helper("workflow")["found"] is False
        scheduler = manager.resolve_helper("scheduler")
        assert scheduler["module"] is None
        assert {item["name"] for item in scheduler["functions"]} >= {"create", "update", "list", "delete", "run_now"}
        assert manager.resolve_action("worktree.create")["found"] is True
        assert manager.resolve_action("worktree.cleanup")["found"] is True
        assert manager.resolve_action("subagent.prompt")["found"] is True
        assert manager.resolve_action("workflow.prompt")["found"] is False
        assert manager.text("goal_enabled", "zh-CN") == "已开启"
        assert manager.text("worktree_delete", "en") == "Delete worktree and branch"
        assert manager.text("mention_mcp_hint", "zh-CN") == "搜索后按 Enter 插入 MCP 引用"
        assert manager.text("mention_skills_hint", "en") == "Search and Enter to insert a skill mention"
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_builtin_persistent_plugins_skip_session_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "uv_agent.plugins.manager.importlib.metadata.entry_points",
        lambda group: _builtin_entry_points() if group == "uv_agent.plugins" else [],
    )
    manager = PluginManager(
        config=PluginsConfig(entries={"builtin.workflow": PluginConfigBlock(enabled=True)}),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeNamespaceRegistry(),
        submitter=None,
        thread_store=None,
        user_state_dir=tmp_path / "state",
        host=_host_info(tmp_path, invocation="tui", lifetime="session"),
    )

    try:
        await manager.start()

        statuses = {record.id: record for record in manager.records}
        assert statuses["builtin.scheduler"].state == "skipped"
        assert statuses["builtin.scheduler"].message == "requires persistent host"
        assert statuses["builtin.workflow"].state == "skipped"
        assert statuses["builtin.workflow"].message == "requires persistent host"
        assert manager.resolve_helper("scheduler")["found"] is False
        assert manager.resolve_helper("workflow")["found"] is False
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_builtin_mcp_is_disabled_by_default_and_warns_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from uv_agent.builtin.mcp import plugin as mcp_plugin

    monkeypatch.setattr(
        "uv_agent.plugins.manager.importlib.metadata.entry_points",
        lambda group: [EntryPoint("builtin_mcp", mcp_plugin)] if group == "uv_agent.plugins" else [],
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
    try:
        status = {record.id: record for record in manager.records}["builtin.mcp"]
        assert status.state == "disabled"
        assert status.deprecated is True
    finally:
        await manager.stop()

    warnings: list[dict[str, object]] = []
    enabled_manager = PluginManager(
        config=PluginsConfig(entries={"builtin.mcp": PluginConfigBlock(enabled=True)}),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeNamespaceRegistry(),
        submitter=None,
        thread_store=None,
        user_state_dir=tmp_path / "state-enabled",
    )
    enabled_manager.events.subscribe("plugin.warning", lambda event: warnings.append(event))
    await enabled_manager.start()
    try:
        status = {record.id: record for record in enabled_manager.records}["builtin.mcp"]
        assert status.state == "warning"
        assert status.error_type == "DeprecatedPlugin"
        assert "deprecated" in status.message.lower()
        assert warnings and any(warning.get("deprecated") for warning in warnings)
    finally:
        await enabled_manager.stop()

@pytest.mark.asyncio
async def test_builtin_subagent_runtime_helper_creates_child_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from uv_agent.builtin.subagent import plugin as subagent_plugin

    class Submitted:
        def __init__(self, thread_id: str | None) -> None:
            self.thread_id = thread_id
            self.turn_id = "turn_child"
            self.status = "completed"
            self.final_text = "child done"
            self.error = None

        async def wait(self):
            return self

    submit_calls: list[dict[str, Any]] = []

    async def submitter(**kwargs):
        submit_calls.append(dict(kwargs))
        return Submitted(kwargs.get("thread_id"))

    monkeypatch.setattr(
        "uv_agent.plugins.manager.importlib.metadata.entry_points",
        lambda group: [EntryPoint("builtin_subagent", subagent_plugin)] if group == "uv_agent.plugins" else [],
    )
    store = ThreadStore(tmp_path / "project-state")
    parent_thread_id = store.create_thread("Parent")
    manager = PluginManager(
        config=PluginsConfig(),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeNamespaceRegistry(),
        submitter=submitter,
        thread_store=store,
        user_state_dir=tmp_path / "state",
    )

    await manager.start()
    try:
        result = await manager.call_helper(
            "subagent.run",
            args=["Investigate child task"],
            kwargs={"level": "small", "title": "Child task"},
            context=SimpleNamespace(thread_id=parent_thread_id, turn_id="turn_parent", run_id="run_parent"),
        )

        assert result["status"] == "completed"
        assert result["final_text"] == "child done"
        assert result["thread_id"].startswith("thr_")
        assert submit_calls == [
            {
                "text": "Investigate child task",
                "thread_id": result["thread_id"],
                "level": "small",
                "image_paths": None,
                "attachments": None,
                "conflict": "queue",
            }
        ]
        metadata = store.thread_metadata(result["thread_id"])
        assert metadata["kind"] == "subagent"
        assert metadata["parent_thread_id"] == parent_thread_id
        assert metadata["parent_turn_id"] == "turn_parent"
        assert metadata["parent_run_id"] == "run_parent"
        assert metadata["owner_plugin"] == "builtin.subagent"
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_plugin_context_blobs_and_submit_attachments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Submitted:
        thread_id = "thr"
        turn_id = "turn"
        status = "completed"
        final_text = ""
        error = None

        async def wait(self):
            return self

    submit_calls: list[dict[str, Any]] = []

    async def submitter(**kwargs):
        submit_calls.append(dict(kwargs))
        return Submitted()

    async def setup(context) -> None:
        ref = context.blobs.put_bytes(
            b"hello",
            mime_type="text/plain",
            filename="report.txt",
            max_bytes=1024,
        )
        await context.submit_turn(
            text="read [File report.txt]",
            thread_id="thr",
            attachments=[
                {
                    "kind": "file",
                    "token": "[File report.txt]",
                    "blob_id": ref["blob_id"],
                    "filename": ref["filename"],
                    "mime_type": ref["mime_type"],
                }
            ],
        )

    blob_store = BlobStore(tmp_path / "project-state")
    _install_entry_points(monkeypatch, [_plugin("blob-plugin", setup)])
    manager = PluginManager(
        config=PluginsConfig(),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeNamespaceRegistry(),
        submitter=submitter,
        thread_store=None,
        blob_store=blob_store,
        user_state_dir=tmp_path / "state",
    )

    await manager.start()
    try:
        assert len(submit_calls) == 1
        call = submit_calls[0]
        assert call["text"] == "read [File report.txt]"
        assert call["thread_id"] == "thr"
        assert call["image_paths"] is None
        attachment = call["attachments"][0]
        assert attachment["kind"] == "file"
        assert attachment["token"] == "[File report.txt]"
        assert attachment["filename"] == "report.txt"
        assert blob_store.info(attachment["blob_id"])["size_bytes"] == 5
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_deprecated_plugin_is_disabled_by_default_and_warns_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def setup(context) -> None:
        context.runtime.register_namespace("legacy", functions={"ping": lambda: {"ok": True}})

    legacy = SetupPlugin(
        manifest=PluginManifest(
            id="legacy-plugin",
            version="0.1.0",
            display_name="Legacy",
            description="Deprecated test plugin",
            default_enabled=False,
            deprecated=True,
            deprecation_message="legacy-plugin is deprecated; use demo-plugin instead.",
        ),
        setup=setup,
    )

    manager = _manager(tmp_path, monkeypatch, [legacy])
    await manager.start()
    assert manager.records[0].state == "disabled"
    assert manager.records[0].deprecated is True
    await manager.stop()

    warnings: list[dict[str, Any]] = []
    _install_entry_points(monkeypatch, [legacy])
    enabled = PluginManager(
        config=PluginsConfig(entries={"legacy-plugin": PluginConfigBlock(enabled=True)}),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeNamespaceRegistry(),
        submitter=None,
        thread_store=None,
        user_state_dir=tmp_path / "state-enabled",
    )
    enabled.events.subscribe("plugin.warning", lambda event: warnings.append(event))

    await enabled.start()

    status = {record.id: record for record in enabled.records}["legacy-plugin"]
    assert status.state == "warning"
    assert status.error_type == "DeprecatedPlugin"
    assert status.message == "legacy-plugin is deprecated; use demo-plugin instead."
    assert warnings and warnings[0]["deprecated"] is True
    assert enabled.resolve_helper("legacy")["found"] is True
    await enabled.stop()


@pytest.mark.asyncio
async def test_plugin_manager_starts_setup_plugin_and_registers_capabilities(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    async def setup(context) -> None:
        seen.append(context.config)
        context.runtime.register_namespace(
            "demo",
            doc="Demo runtime namespace.",
            functions={"greet": lambda name: {"hello": name}},
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
        context.actions.register(
            "demo.context",
            lambda payload, context=None: {
                "plugin": context.manifest.id if context is not None else "",
                "answer": context.config.get("answer") if context is not None else None,
            },
        )
        context.commands.register("/demo", lambda payload: {"command": payload.get("value")}, description="Demo command.")
        context.epoch.publish(tag="demo_status", body={"state": "ready"})

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
    assert await manager.call_action("demo.context") == {"plugin": "demo-plugin", "answer": 42}
    explicit_context = SimpleNamespace(manifest=SimpleNamespace(id="caller"), config={"answer": 99})
    assert await manager.call_action("demo.context", context=explicit_context) == {"plugin": "caller", "answer": 99}
    context = manager.context_for("demo-plugin")
    assert context is not None
    assert context.actions.resolve("demo.echo")["found"] is True
    assert await context.actions.call("demo.context") == {"plugin": "demo-plugin", "answer": 42}
    assert manager.call_command("/demo", {"value": 7}) == {"command": 7}
    assert _epoch_text(manager.contexts, "thread", core_texts=["<core />"]).endswith(
        "<agent_demo_status>\n<state>ready</state>\n</agent_demo_status>"
    )


@pytest.mark.asyncio
async def test_plugin_context_exposes_host_info(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[tuple[str, str, bool, Path, Path, Path]] = []

    def setup(context) -> None:
        seen.append(
            (
                context.host.invocation,
                context.host.lifetime,
                context.host.is_persistent,
                context.host.project_root,
                context.host.project_state_dir,
                context.host.user_state_dir,
            )
        )

    host = _host_info(tmp_path, invocation="daemon", lifetime="persistent")
    manager = _manager(tmp_path, monkeypatch, [_plugin("host-aware", setup)], host=host)

    await manager.start()

    assert seen == [
        (
            "daemon",
            "persistent",
            True,
            tmp_path,
            tmp_path / "project-state",
            tmp_path / "user-state",
        )
    ]
    await manager.stop()


@pytest.mark.asyncio
async def test_plugin_activation_skips_persistent_only_in_session_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started = False
    skipped_events: list[dict[str, Any]] = []

    def setup(context) -> None:
        nonlocal started
        started = True

    manager = _manager(
        tmp_path,
        monkeypatch,
        [_plugin("server-plugin", setup, activation="persistent_only")],
        host=_host_info(tmp_path, invocation="tui", lifetime="session"),
    )
    manager.events.subscribe("plugin.skipped", lambda event: skipped_events.append(event))

    await manager.start()

    status = manager.records[0]
    assert status.state == "skipped"
    assert status.message == "requires persistent host"
    assert started is False
    assert skipped_events == [
        {
            "type": "plugin.skipped",
            "plugin": "server-plugin",
            "message": "requires persistent host",
            "activation": "persistent_only",
            "host_lifetime": "session",
        }
    ]
    await manager.stop()


@pytest.mark.asyncio
async def test_plugin_activation_allows_matching_host_lifetime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started: list[str] = []

    def setup(context) -> None:
        started.append(context.host.lifetime)

    manager = _manager(
        tmp_path,
        monkeypatch,
        [_plugin("server-plugin", setup, activation="persistent_only")],
        host=_host_info(tmp_path, invocation="daemon", lifetime="persistent"),
    )

    await manager.start()

    assert started == ["persistent"]
    assert manager.records[0].state == "started"
    await manager.stop()


@pytest.mark.asyncio
async def test_plugin_compaction_provider_errors_become_warnings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def setup(context) -> None:
        def provider(*, thread_id: str) -> str:
            raise TypeError(f"bad provider body for {thread_id}")

        context.compaction.summary_section(provider)

    manager = _manager(tmp_path, monkeypatch, [_plugin("compaction-plugin", setup)])

    await manager.start()
    try:
        assert manager.compaction_sections("thread") == []
        status = manager.records[0]
        assert status.state == "warning"
        assert status.error_type == "TypeError"
        assert "bad provider body" in status.message
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_plugin_epoch_refresh_preserves_type_error_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def setup(context) -> None:
        def refresh(*, thread_id: str | None = None) -> None:
            raise TypeError(f"bad refresh body for {thread_id}")

        context.epoch.on_refresh(refresh)

    manager = _manager(tmp_path, monkeypatch, [_plugin("refresh-plugin", setup)])

    await manager.start()
    try:
        status = manager.records[0]
        assert status.state == "warning"
        assert status.error_type == "TypeError"
        assert "bad refresh body" in status.message
    finally:
        await manager.stop()


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


def test_plugin_status_summary_formats_counts_and_details() -> None:
    records = [
        PluginStatus(id="builtin.goal", state="started", builtin=True),
        PluginStatus(id="remote-control", state="started", first_load=True),
        PluginStatus(id="auth-code", state="failed", message="missing token", error_type="ValueError"),
        PluginStatus(id="server-plugin", state="skipped", message="requires persistent host"),
        PluginStatus(id="legacy-plugin", state="warning", message="deprecated", error_type="DeprecatedPlugin"),
        PluginStatus(id="off-plugin", state="disabled"),
    ]

    assert (
        format_plugin_status_counts(records)
        == "plugins: total=6 started=2 warning=1 failed=1 skipped=1 disabled=1"
    )
    assert format_plugin_detail_lines(records, include_started_external=True, include_first_load_external=True) == [
        "plugin started: remote-control",
        "plugin first load: remote-control",
        "plugin failed: auth-code (ValueError: missing token)",
        "plugin warning: legacy-plugin (DeprecatedPlugin: deprecated)",
        "plugin skipped: server-plugin (requires persistent host)",
        "plugin disabled: off-plugin",
    ]


@pytest.mark.asyncio
async def test_plugin_manager_rejects_non_setup_plugin_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "uv_agent.plugins.manager.importlib.metadata.entry_points",
        lambda group: [EntryPoint("dict-plugin", {"manifest": {}, "setup": lambda _ctx: None})]
        if group == "uv_agent.plugins"
        else [],
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

    assert len(manager.records) == 1
    record = manager.records[0]
    assert record.id == "dict-plugin"
    assert record.state == "failed"
    assert record.error_type == "TypeError"
    assert "SetupPlugin" in record.message


@pytest.mark.asyncio
async def test_plugin_manager_orders_dependencies_and_isolates_setup_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    order: list[str] = []

    def setup_base(context) -> None:
        order.append(context.plugin_id)
        context.runtime.register_namespace("base", functions={"ok": lambda: {"ok": True}})

    def setup_child(context) -> None:
        order.append(context.plugin_id)
        context.runtime.register_namespace("base", functions={"boom": lambda: {}})

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
async def test_plugin_logs_use_rotating_file_handler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def setup(context) -> None:
        handlers = [
            handler
            for handler in context.logger.handlers
            if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename).name == "plugin.log"
        ]
        assert handlers
        assert handlers[0].maxBytes == 400
        assert handlers[0].backupCount == 1
        for index in range(20):
            context.logger.info("plugin log rotation smoke %02d %s", index, "x" * 80)

    app_logger = logging.getLogger("uv_agent")
    old_level = app_logger.level
    app_logger.setLevel(logging.INFO)
    try:
        manager = _manager(
            tmp_path,
            monkeypatch,
            [_plugin("rotating-plugin", setup)],
            logging_config=LoggingConfig(max_bytes=400, backup_count=1),
        )

        await manager.start()
        await manager.stop()
    finally:
        app_logger.setLevel(old_level)

    log_dir = tmp_path / "state" / "plugins" / "rotating-plugin" / "logs"
    assert (log_dir / "plugin.log").exists()
    assert (log_dir / "plugin.log.1").exists()


@pytest.mark.asyncio
async def test_plugin_log_rotation_updates_on_logging_config_reload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = _manager(
        tmp_path,
        monkeypatch,
        [_plugin("reload-log-plugin", lambda context: None)],
        logging_config=LoggingConfig(max_bytes=400, backup_count=1),
    )

    await manager.start()
    try:
        context = manager.context_for("reload-log-plugin")
        assert context is not None

        def rotating_handlers() -> list[RotatingFileHandler]:
            return [
                handler
                for handler in context.logger.handlers
                if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename).name == "plugin.log"
            ]

        before = rotating_handlers()
        assert len(before) == 1
        assert before[0].maxBytes == 400
        assert before[0].backupCount == 1

        manager.reload_logging_config(LoggingConfig(max_bytes=800, backup_count=2))

        after = rotating_handlers()
        assert len(after) == 1
        assert after[0].maxBytes == 800
        assert after[0].backupCount == 2
    finally:
        await manager.stop()


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
        registry.register_namespace(plugin="p", namespace="file", functions={"x": lambda: {}})
    with pytest.raises(ValueError):
        registry.register_namespace(plugin="p", namespace="bad-name", functions={})
    with pytest.raises(ValueError):
        registry.register_namespace(
            plugin="p",
            namespace="demo",
            functions=(RuntimeFunctionSpec(namespace="demo", name="x", plugin="p", doc="", schema={"type": "string"}, fn=lambda: {}),),
        )
    with pytest.raises(ValueError):
        registry.register_namespace(plugin="p", namespace="demo", functions={"x": None})  # type: ignore[arg-type]


def test_resource_registry_routes_longest_prefix_and_validates_payloads() -> None:
    registry = ResourceRegistry()
    registry.register(plugin="root", prefix="skill://", read=lambda uri: ResourceData(uri=uri, kind="text", text="root"))

    async def read_plugin_resource(uri: str, *, max_bytes: int | None = None) -> dict[str, object]:
        assert max_bytes == 10
        return {"data": b"image-bytes", "mime_type": "image/png", "filename": "demo.png"}

    registry.register(plugin="plugin", prefix="skill://plugin/demo/", read=read_plugin_resource)

    assert registry.provider_for("skill://plugin/demo/image.png").plugin == "plugin"
    assert registry.read("skill://project/demo").text == "root"
    payload = coerce_resource_data(registry.read("skill://plugin/demo/image.png", max_bytes=10), uri="skill://plugin/demo/image.png")
    assert payload.kind == "bytes"
    assert payload.data == b"image-bytes"
    assert payload.mime_type == "image/png"

    with pytest.raises(ValueError):
        registry.provider_for("Skill://project/demo")
    with pytest.raises(ValueError):
        ResourceData(uri="skill://bad", kind="text", text="x", data=b"y")
    with pytest.raises(ValueError):
        coerce_resource_data({"text": "x", "data": b"y"}, uri="skill://bad")


@pytest.mark.asyncio
async def test_action_registry_supports_optional_missing_and_rich_payload() -> None:
    registry = ActionRegistry()

    def handler(payload: dict[str, object], *, caller_plugin: str | None = None) -> dict[str, object]:
        data = payload["data"]
        assert isinstance(data, bytes)
        return {"caller": caller_plugin, "size": len(data)}

    registry.register(plugin="target", action_id="demo.bytes", handler=handler)

    assert await registry.call("missing.action", missing="ignore") == {
        "ok": False,
        "missing": True,
        "action_id": "missing.action",
    }
    assert await registry.call("demo.bytes", {"data": b"abc"}, caller_plugin="caller.plugin") == {
        "caller": "caller.plugin",
        "size": 3,
    }


def test_plugin_registry_accepts_localized_text_and_normalizes_picker_items() -> None:
    from uv_agent.plugins.i18n import PluginI18nRegistry, localize_text

    ui = UiRegistry()
    ui.register_picker(
        PickerSource(
            plugin="demo",
            id="demo",
            title={"zh": "演示", "en": "Demo"},
            provider=lambda query="": [
                {"value": "/demo", "description": {"zh": "运行演示", "en": "run demo"}, "meta": {"zh": "插件", "en": "plugin"}}
            ],
        )
    )

    item = ui.picker_items("demo")[0]
    assert localize_text(item.description, "zh-CN") == "运行演示"
    assert localize_text(item.description, "en") == "run demo"
    assert localize_text(item.meta, "zh-CN") == "插件"

    i18n = PluginI18nRegistry()
    i18n.register(plugin="demo", texts={"demo.label": {"zh": "演示", "en": "Demo"}})
    assert i18n.text("demo.label", "zh-CN") == "演示"
    assert i18n.text("demo.label", "en") == "Demo"
    assert i18n.text("missing", "en") == ""
    with pytest.raises(ValueError):
        i18n.register(plugin="other", texts={"demo.label": "Other"})


@pytest.mark.asyncio
async def test_builtin_goal_command_records_thread_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from uv_agent.builtin.goal import plugin as goal_plugin

    monkeypatch.setattr(
        "uv_agent.plugins.manager.importlib.metadata.entry_points",
        lambda group: [EntryPoint("builtin_goal", goal_plugin)] if group == "uv_agent.plugins" else [],
    )
    store = ThreadStore(tmp_path / "project-state")
    thread_id = store.create_thread("Goal")
    manager = PluginManager(
        config=PluginsConfig(),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeNamespaceRegistry(),
        submitter=None,
        thread_store=store,
        user_state_dir=tmp_path / "state",
    )

    try:
        await manager.start()

        manager.call_command("/goal", {"arg": "enable ship it", "thread_id": thread_id})
        metadata = store.thread_digest(thread_id)
        assert metadata["goal_mode"]["enabled"] is True
        assert metadata["goal_mode"]["objective"] == "ship it"

        manager.call_command("/goal", {"arg": "disable", "thread_id": thread_id})
        metadata = store.thread_digest(thread_id)
        assert metadata["goal_mode"]["enabled"] is False

        manager.call_command("/goal", {"arg": "reset next", "thread_id": thread_id})
        metadata = store.thread_digest(thread_id)
        assert metadata["goal_mode"]["enabled"] is False
        assert metadata["goal_mode"]["objective"] == "next"
        events = store.read_events(thread_id, event_types={"thread.goal_mode_updated", "thread.goal_state_reset"})
        assert [event["type"] for event in events] == [
            "thread.goal_mode_updated",
            "thread.goal_mode_updated",
            "thread.goal_state_reset",
        ]
    finally:
        await manager.stop()


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

    full = _epoch_text(broker, "thread", core_texts=["<agent_core />"])
    assert full == "<agent_core />\n\n<agent_status>\n<state>ready</state>\n</agent_status>"
    assert broker.consume_updates("thread") == []

    broker.update(plugin="p", tag="status", body={"state": "running"})
    update = render_update_envelope([item.contribution for item in broker.consume_updates("thread")])
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
    broker.replay_after_compaction("thread")
    assert broker.turn_context_text("thread") == "<agent_notice>\n<message>check</message>\n</agent_notice>"


def test_context_broker_requires_replay_key_and_clears_replay_with_non_replay_update() -> None:
    broker = PluginContextBroker()
    with pytest.raises(ValueError):
        broker.enqueue_turn(
            plugin="p",
            thread_id="thread",
            tag="notice",
            body={"message": "missing key"},
            replay_after_compaction=True,
        )

    broker.enqueue_turn(
        plugin="p",
        thread_id="thread",
        tag="notice",
        body={"message": "replay"},
        replay_after_compaction=True,
        replay_key="notice",
    )
    assert "replay" in broker.turn_context_text("thread")
    broker.enqueue_turn(
        plugin="p",
        thread_id="thread",
        tag="notice",
        body={"message": "single"},
        replay_key="notice",
    )
    assert "single" in broker.turn_context_text("thread")
    broker.replay_after_compaction("thread")
    assert broker.turn_context_text("thread") == ""


def test_context_broker_update_is_send_queue_not_epoch_state() -> None:
    broker = PluginContextBroker()
    broker.publish(plugin="p", tag="status", body={"all": [{"name": "old"}]})
    _epoch_text(broker, "thread")

    broker.update(
        plugin="p",
        tag="status_update",
        body={"skill": [{"name": "new"}]},
    )

    update = render_update_envelope([item.contribution for item in broker.consume_updates("thread")])
    assert '<agent_status_update operation="update">' in update
    assert "<skill>" in update
    full = _epoch_text(broker, "thread")
    assert full == ""


def test_epoch_refresh_registration_can_be_disposed() -> None:
    refreshers = []
    broker = PluginContextBroker()
    from uv_agent.plugins.context import PluginEpochContextAPI

    api = PluginEpochContextAPI(plugin="p", broker=broker, refreshers=refreshers)
    registration = api.on_refresh(lambda thread_id=None: None)

    assert len(refreshers) == 1
    registration.dispose()
    registration.dispose()
    assert refreshers == []


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


def test_plugin_thread_api_creates_threads_and_records_events(tmp_path: Path) -> None:
    from uv_agent.plugins.context import PluginThreadAPI
    from uv_agent.session import ThreadStore

    store = ThreadStore(tmp_path / "state")
    api = PluginThreadAPI(plugin="demo", thread_store=store)

    parent = api.create_thread("Parent")
    child = api.create_thread("Child", kind="plugin_worker", parent_thread_id=parent)
    stored = api.record_event(child, "thread.demo_updated", status="active")
    later = api.record_event(child, "thread.demo_updated", status="later")
    last = api.record_event(child, "thread.demo_updated", status="last")

    assert store.thread_metadata(child)["kind"] == "plugin_worker"
    assert store.thread_metadata(child)["parent_thread_id"] == parent
    assert stored["type"] == "thread.demo_updated"
    after_page = api.event_page(child, after_event_id=stored["_event_id"], limit=1)
    assert [event["status"] for event in after_page["events"]] == ["later"]
    assert after_page["has_more"] is True
    before_page = api.event_page(child, before_event_id=last["_event_id"], limit=1)
    assert [event["status"] for event in before_page["events"]] == ["later"]
    assert before_page["has_more"] is True


@pytest.mark.asyncio
async def test_plugin_context_start_turn_does_not_wait(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    submit_calls: list[dict[str, Any]] = []
    handles: list[Any] = []

    async def submitter(**kwargs):
        submit_calls.append(dict(kwargs))
        return SimpleNamespace(request_id="req_1", thread_id=kwargs.get("thread_id"), status="queued")

    async def setup(context) -> None:
        handles.append(await context.start_turn(text="go", thread_id="thr"))

    _install_entry_points(monkeypatch, [_plugin("starter-plugin", setup)])
    manager = PluginManager(
        config=PluginsConfig(),
        project_root=tmp_path,
        events=EventBus(),
        helper_registry=RuntimeNamespaceRegistry(),
        submitter=submitter,
        thread_store=None,
        user_state_dir=tmp_path / "state",
    )

    await manager.start()
    try:
        assert handles[0].request_id == "req_1"
        assert submit_calls == [
            {
                "text": "go",
                "thread_id": "thr",
                "level": None,
                "image_paths": None,
                "attachments": None,
                "conflict": "queue",
                "wait": False,
            }
        ]
    finally:
        await manager.stop()


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
