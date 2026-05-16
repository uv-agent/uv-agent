from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest

from uv_agent.agent import AgentEngine, PYTHON_TOOL, usage_token_count
from uv_agent.attachments import image_message_item
from uv_agent.config import (
    AppConfig,
    CompressionConfig,
    LevelConfig,
    ModelConfig,
    ProviderConfig,
    RunnerConfig,
    RuntimeConfig,
    load_config,
)
from uv_agent.model_client import FakeModelClient
from uv_agent.runner import PythonRunner
from uv_agent.runner.models import PythonRunRequest
from uv_agent.session import ThreadStore


class BlockingModelClient(FakeModelClient):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()

    async def stream_response(self, **kwargs):
        self.started.set()
        await asyncio.Event().wait()
        yield


def test_agent_exposes_only_python_runner_tool() -> None:
    assert PYTHON_TOOL["name"] == "run_python"
    assert PYTHON_TOOL["type"] == "function"
    assert "script_id" in PYTHON_TOOL["parameters"]["properties"]


@pytest.mark.asyncio
async def test_agent_persists_compaction_item(tmp_path: Path) -> None:
    project_root = Path.cwd()
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                context_window_tokens=1,
                params={},
            )
        },
        levels={
            "medium": LevelConfig(name="medium", model="default", params={}),
            "small": LevelConfig(name="small", model="default", params={}),
        },
        runtime=RuntimeConfig(
            default_level="medium",
            auto_compress=True,
            compression=CompressionConfig(model_level="small", trigger_ratio=0.1, min_tokens=1),
        ),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {project_root.resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    client = FakeModelClient(
        [
            {
                "id": "resp_1",
                "output_text": "final",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "final"}],
                    }
                ],
            },
            {
                "id": "resp_compact",
                "output_text": "summary",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "summary"}],
                    }
                ],
            },
        ]
    )
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / ".uv-agent",
        config=RunnerConfig(
            runtime_dependency=f"uv-agent @ {project_root.resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=runner,
        thread_store=ThreadStore(tmp_path / ".uv-agent"),
        project_root=project_root,
    )
    events = [event async for event in engine.run_turn(user_text="hello")]

    stored = engine.thread_store.read(events[-1]["thread_id"])
    assert any(event["type"] == "item.compaction" for event in stored)


@pytest.mark.asyncio
async def test_agent_persists_interrupted_turn_and_follow_up_continues(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                context_window_tokens=100_000,
                params={},
            )
        },
        levels={"medium": LevelConfig(name="medium", model="default", params={})},
        runtime=RuntimeConfig(auto_compress=False),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    blocking_client = BlockingModelClient()
    engine = AgentEngine(
        config=config,
        model_client=blocking_client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    cancel_event = asyncio.Event()

    async def collect() -> list[dict[str, object]]:
        return [event async for event in engine.run_turn(user_text="stop me", cancel_event=cancel_event)]

    task = asyncio.create_task(collect())
    await blocking_client.started.wait()
    cancel_event.set()
    events = await asyncio.wait_for(task, timeout=5)
    thread_id = str(events[-1]["thread_id"])

    assert events[-1]["type"] == "turn.interrupted"
    assert any(event["type"] == "turn.interrupted" for event in engine.thread_store.read(thread_id))

    follow_client = FakeModelClient(
        [
            {
                "id": "resp_follow",
                "output_text": "after",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "after"}],
                    }
                ],
            }
        ]
    )
    engine.model_client = follow_client
    follow_up = [event async for event in engine.run_turn(user_text="continue", thread_id=thread_id)]

    assert follow_up[-1]["type"] == "turn.completed"
    assert "interrupted" in str(follow_client.requests[0]["input"])


@pytest.mark.asyncio
async def test_agent_runs_python_tool_boundary(tmp_path: Path) -> None:
    project_root = Path.cwd()
    config = load_config(project_root, [])
    config = AppConfig(
        providers=config.providers,
        models=config.models,
        levels=config.levels,
        runtime=RuntimeConfig(auto_compress=False),
        runner=config.runner,
    )
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / ".uv-agent",
        config=RunnerConfig(
            runtime_dependency=f"uv-agent @ {project_root.resolve().as_uri()}",
            runtime_package_name="uv-agent",
            default_timeout_s=30,
        ),
    )
    client = FakeModelClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "run_python",
                        "arguments": "{\"code\":\"print('observed')\\n\"}",
                    }
                ],
            },
            {
                "id": "resp_2",
                "output_text": "done",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
        ]
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=runner,
        thread_store=ThreadStore(tmp_path / ".uv-agent"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="run it")]

    assert events[-1]["type"] == "turn.completed"
    assert events[-1]["final_text"] == "done"
    assert any(event["type"] == "assistant.delta" for event in events)
    assert any(event["type"] == "tool.started" for event in events)
    assert len(client.requests) == 2
    assert client.requests[0]["tools"] == [PYTHON_TOOL]
    tool_output = client.requests[1]["input"][-1]
    assert tool_output["type"] == "function_call_output"
    assert "observed" in tool_output["output"]
    thread_id = events[-1]["thread_id"]
    stored_events = engine.thread_store.read(thread_id)
    assert any(event["type"] == "item.tool_output" for event in stored_events)


@pytest.mark.asyncio
async def test_agent_can_rerun_saved_script_by_id(tmp_path: Path) -> None:
    project_root = Path.cwd()
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                context_window_tokens=100_000,
                params={},
            )
        },
        levels={"medium": LevelConfig(name="medium", model="default", params={})},
        runtime=RuntimeConfig(auto_compress=False),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {project_root.resolve().as_uri()}",
            runtime_package_name="uv-agent",
            default_timeout_s=30,
        ),
    )
    runner = PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner)
    first = await runner.run(PythonRunRequest(code="print('saved')\n", cwd=project_root))
    client = FakeModelClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "run_python",
                        "arguments": json.dumps({"script_id": first.script_id}),
                    }
                ],
            },
            {
                "id": "resp_2",
                "output_text": "done",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
        ]
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=runner,
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="rerun")]

    assert events[-1]["final_text"] == "done"
    assert "saved" in client.requests[1]["input"][-1]["output"]


def test_agent_prompt_keeps_dynamic_capabilities_in_turn_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    project_root = tmp_path / "project"
    skill_dir = project_root / ".agents" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo\nUse this for demo work.\n", encoding="utf-8")
    (project_root / ".agents" / "mcp.json").write_text(
        "{\"servers\":{\"demo\":{\"command\":\"python\",\"description\":\"Demo MCP\"}}}",
        encoding="utf-8",
    )
    config = load_config(project_root, [])
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / "state",
        config=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=runner,
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    prompt = engine.system_instructions()

    assert "run_python" in prompt
    assert "uv_agent_runtime" in prompt
    assert str(project_root) in prompt
    assert prompt.startswith("<uv_agent_system_prompt>")
    assert "</uv_agent_system_prompt>" in prompt
    assert "<environment>" in prompt
    assert "<host>" in prompt
    assert "<user_language>" in prompt
    assert "</runtime_helpers>" in prompt
    assert 'requires-python = ">=3.12"' in prompt
    assert '# dependencies = [' in prompt
    assert "plain Python source without a metadata block" in prompt
    assert "not a temporary-script wrapper" in prompt
    assert "temporary nested uv-agent subprocess" in prompt
    assert "summarizing a thread" in prompt
    assert "Mentions are plain-text hints only" in prompt
    assert "saved_scripts(limit=32)" in prompt
    assert "Rules, skills, and MCP declarations are appended only when first seen" in prompt
    assert "demo (project)" not in prompt

    turn_context = engine._turn_context_text()

    assert "demo (project)" in turn_context
    assert "available_mcp_servers" in turn_context


def test_usage_token_count_supports_provider_shapes() -> None:
    assert usage_token_count({"total_tokens": 42}) == 42
    assert usage_token_count({"input_tokens": 10, "output_tokens": 3}) == 13
    assert usage_token_count({"prompt_tokens": 9, "completion_tokens": 2}) == 11
    assert usage_token_count({}) is None


def test_context_percent_prefers_latest_usage(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                context_window_tokens=100,
                params={},
            )
        },
        levels={"medium": LevelConfig(name="medium", model="default", params={})},
        runtime=RuntimeConfig(default_level="medium", auto_compress=False),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(
            project_root=project_root,
            data_dir=tmp_path / "state",
            config=config.runner,
        ),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread()
    engine.thread_store.append(
        thread_id,
        "item.model_response",
        turn_id="turn_1",
        response_id="resp_1",
        output=[],
        usage={"input_tokens": 23, "output_tokens": 7},
    )

    assert engine.context_percent(thread_id) == 30


@pytest.mark.asyncio
async def test_agent_records_project_rule_context_update(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "AGENTS.md").write_text("Use the local rule.", encoding="utf-8")
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                context_window_tokens=100_000,
                params={},
            )
        },
        levels={"medium": LevelConfig(name="medium", model="default", params={})},
        runtime=RuntimeConfig(default_level="medium", auto_compress=False),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    client = FakeModelClient(
        [
            {
                "id": "resp_1",
                "output_text": "ok",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            }
        ]
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="hello")]

    request_text = str(client.requests[0]["input"])
    assert "Use the local rule." in request_text
    stored_text = str(engine.thread_store.read(events[-1]["thread_id"]))
    assert "Use the local rule." in stored_text
    assert any(event["type"] == "item.context_update" for event in engine.thread_store.read(events[-1]["thread_id"]))


@pytest.mark.asyncio
async def test_compaction_does_not_include_project_rules(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "AGENTS.md").write_text("Never persist this rule.", encoding="utf-8")
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                context_window_tokens=20,
                params={},
            )
        },
        levels={
            "medium": LevelConfig(name="medium", model="default", params={}),
            "small": LevelConfig(name="small", model="default", params={}),
        },
        runtime=RuntimeConfig(
            default_level="medium",
            auto_compress=True,
            compression=CompressionConfig(model_level="small", trigger_ratio=0.1, min_tokens=1),
        ),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    client = FakeModelClient(
        [
            {
                "id": "resp_1",
                "output_text": "ok",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            },
            {
                "id": "resp_compact",
                "output_text": "summary",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "summary"}],
                    }
                ],
            },
        ]
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="hello")]

    assert events[-1]["type"] == "turn.completed"
    assert "Never persist this rule." in str(client.requests[0]["input"])
    assert "Never persist this rule." not in str(client.requests[1]["input"])


@pytest.mark.asyncio
async def test_dynamic_context_only_appends_when_changed(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    rules = project_root / "AGENTS.md"
    rules.write_text("Rule v1.", encoding="utf-8")
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                context_window_tokens=100_000,
                params={},
            )
        },
        levels={"medium": LevelConfig(name="medium", model="default", params={})},
        runtime=RuntimeConfig(default_level="medium", auto_compress=False),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    client = FakeModelClient(
        [
            {
                "id": "resp_1",
                "output_text": "one",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "one"}],
                    }
                ],
            },
            {
                "id": "resp_2",
                "output_text": "two",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "two"}],
                    }
                ],
            },
            {
                "id": "resp_3",
                "output_text": "three",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "three"}],
                    }
                ],
            },
            {
                "id": "resp_4",
                "output_text": "four",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "four"}],
                    }
                ],
            },
        ]
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    first = [event async for event in engine.run_turn(user_text="one")][-1]["thread_id"]
    [event async for event in engine.run_turn(user_text="two", thread_id=first)]
    requests_text = [str(request["input"]) for request in client.requests[:2]]

    assert "Rule v1." in requests_text[0]
    assert "Rule v1." not in requests_text[1]

    rules.write_text("Rule v2.", encoding="utf-8")
    [event async for event in engine.run_turn(user_text="three", thread_id=first)]
    assert "Rule v2." in str(client.requests[2]["input"])

    rules.unlink()
    [event async for event in engine.run_turn(user_text="four", thread_id=first)]
    assert "Do not rely on older appended" in str(client.requests[3]["input"])


@pytest.mark.asyncio
async def test_context_update_reappears_after_compaction(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "AGENTS.md").write_text("After compaction rule.", encoding="utf-8")
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                context_window_tokens=100_000,
                params={},
            )
        },
        levels={"medium": LevelConfig(name="medium", model="default", params={})},
        runtime=RuntimeConfig(default_level="medium", auto_compress=False),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread()
    first = engine._workspace_context_items(thread_id)
    engine.thread_store.append(thread_id, "item.compaction", turn_id="t1", text="summary", usage={})
    second = engine._workspace_context_items(thread_id)

    assert first
    assert second
    assert "After compaction rule." in str(second)


def test_reconstruct_input_starts_after_latest_compaction(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                context_window_tokens=100_000,
                params={},
            )
        },
        levels={"medium": LevelConfig(name="medium", model="default", params={})},
        runtime=RuntimeConfig(auto_compress=False),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread()
    engine.thread_store.append(thread_id, "item.user", turn_id="t1", item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "old"}]})
    engine.thread_store.append(thread_id, "item.compaction", turn_id="t1", text="summary", usage={})
    engine.thread_store.append(thread_id, "item.user", turn_id="t2", item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "new"}]})

    reconstructed = engine._reconstruct_input(thread_id)
    text = str(reconstructed)

    assert "summary" in text
    assert "new" in text
    assert "old" not in text


def test_image_attachment_reconstructs_after_compaction(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                context_window_tokens=100_000,
                params={},
            )
        },
        levels={"medium": LevelConfig(name="medium", model="default", params={})},
        runtime=RuntimeConfig(auto_compress=False),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread()
    attachment = engine.attachments.register_image(image, cwd=project_root, thread_id=thread_id)
    engine.thread_store.append(thread_id, "item.compaction", turn_id="t1", text="summary", usage={})
    engine.thread_store.append(
        thread_id,
        "item.image_attachment",
        turn_id="t1",
        attachment=attachment.to_event_payload(),
    )

    reconstructed = engine._reconstruct_input(thread_id)

    assert any(
        content.get("type") == "input_image"
        for item in reconstructed
        for content in item.get("content", [])
    )
    assert image_message_item(attachment.to_event_payload())["content"][1]["image_url"].startswith(
        "data:image/png;base64,"
    )


def test_refresh_config_updates_engine_and_runner(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    project_root = tmp_path / "project"
    project_root.mkdir()
    user_config = tmp_path / "home" / "config.json"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(
        json.dumps(
            {
                "providers": {"p": {"base_url": "https://example.com"}},
                "models": {
                    "m": {
                        "provider": "p",
                        "model": "fake",
                        "context_window_tokens": 10,
                    }
                },
                "levels": {"medium": {"model": "m"}},
                "runner": {"default_timeout_s": 11},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(project_root)
    client = FakeModelClient([])
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
        config_loader=lambda: load_config(project_root),
    )
    raw = json.loads(user_config.read_text(encoding="utf-8"))
    raw["models"]["m"]["context_window_tokens"] = 20
    raw["runner"]["default_timeout_s"] = 22
    user_config.write_text(json.dumps(raw), encoding="utf-8")

    engine.refresh_config(force=True)

    assert engine.config.model_for_level("medium").context_window_tokens == 20
    assert engine.runner.config.default_timeout_s == 22
