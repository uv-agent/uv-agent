from __future__ import annotations

from pathlib import Path

import pytest

from uv_agent.agent import AgentEngine, PYTHON_TOOL, usage_token_count
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
from uv_agent.session import ThreadStore


def test_agent_exposes_only_python_runner_tool() -> None:
    assert PYTHON_TOOL["name"] == "run_python"
    assert PYTHON_TOOL["type"] == "function"


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


def test_agent_system_prompt_mentions_runtime_and_skills(tmp_path: Path, monkeypatch) -> None:
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
    assert "demo (project)" in prompt
    assert "MCP servers" in prompt
    assert str(project_root) in prompt
    assert "Project-specific AGENTS.md files are appended" in prompt


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
async def test_agent_appends_project_rules_without_persisting_context(tmp_path: Path) -> None:
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
    assert "Use the local rule." not in stored_text


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
