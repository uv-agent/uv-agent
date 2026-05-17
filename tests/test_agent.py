from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any

import pytest

from uv_agent.agent import AgentEngine, PYTHON_TOOL, message_item, message_item_text, usage_token_count
from uv_agent.attachments import image_message_item
from uv_agent.config import (
    AppConfig,
    CompressionConfig,
    LevelConfig,
    ModelConfig,
    ProviderConfig,
    RunnerConfig,
    RuntimeConfig,
    TitleGenerationConfig,
    load_config,
)
from uv_agent.model_client import FakeModelClient, ModelStreamEvent, parse_responses_response
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


class RoutedModelClient(FakeModelClient):
    def __init__(self, *, main: dict[str, Any], title: dict[str, Any] | None = None) -> None:
        super().__init__([])
        self.main = main
        self.title = title

    async def create_response(self, **kwargs):
        request = {
            "input": kwargs.get("input_items", []),
            "level": kwargs.get("level"),
            "tools": kwargs.get("tools") or [],
            "instructions": kwargs.get("instructions"),
            "stream": False,
        }
        instructions = str(kwargs.get("instructions") or "")
        if "Generate a short thread title" in instructions and self.title is not None:
            self.requests.append(request)
            return parse_responses_response(self.title)
        self.requests.append(request)
        return parse_responses_response(self.main)


class ReasoningStreamClient(FakeModelClient):
    async def stream_response(self, **kwargs):
        self.requests.append(
            {
                "input": kwargs.get("input_items", []),
                "level": kwargs.get("level"),
                "tools": kwargs.get("tools") or [],
                "instructions": kwargs.get("instructions"),
                "stream": True,
            }
        )
        yield ModelStreamEvent(type="reasoning_delta", text="provider reasoning")
        yield ModelStreamEvent(
            type="completed",
            response=parse_responses_response(
                {
                    "id": "resp_1",
                    "output_text": "done",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        }
                    ],
                }
            ),
        )


class CompletedOnlyStreamClient(FakeModelClient):
    async def stream_response(self, **kwargs):
        self.requests.append(
            {
                "input": kwargs.get("input_items", []),
                "level": kwargs.get("level"),
                "tools": kwargs.get("tools") or [],
                "instructions": kwargs.get("instructions"),
                "stream": True,
                "previous_response_id": kwargs.get("previous_response_id"),
            }
        )
        yield ModelStreamEvent(type="completed", response=parse_responses_response(self.responses.pop(0)))


def make_test_config(
    project_root: Path,
    *,
    api: str = "responses",
    compression_enabled: bool = False,
    context_window_tokens: int = 100_000,
    compression: CompressionConfig | None = None,
    title_generation: TitleGenerationConfig | None = None,
    default_level: str = "medium",
) -> AppConfig:
    return AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                api=api,
                context_window_tokens=context_window_tokens,
                params={},
            )
        },
        levels={
            "medium": LevelConfig(name="medium", model="default", params={}),
            "small": LevelConfig(name="small", model="default", params={}),
        },
        runtime=RuntimeConfig(
            default_level=default_level,
            compression=compression or CompressionConfig(enabled=compression_enabled),
            title_generation=title_generation or TitleGenerationConfig(enabled=False),
        ),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {project_root.resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )


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
            compression=CompressionConfig(enabled=True, model_level="small", trigger_ratio=0.1, min_tokens=1),
            title_generation=TitleGenerationConfig(enabled=False),
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
    compaction = next(event for event in stored if event["type"] == "item.compaction")
    assert compaction["replacement_input"]
    assert client.requests[1]["level"] == "small"
    assert client.requests[1]["tools"] == [PYTHON_TOOL]
    assert client.requests[0]["input"] == client.requests[1]["input"][: len(client.requests[0]["input"])]
    assert "context_compaction_request" in str(client.requests[1]["input"][-1])
    assert "CONTEXT CHECKPOINT COMPACTION" in str(client.requests[1]["input"][-1])
    assert "Target length" not in str(client.requests[1]["input"][-1])
    assert "Continue from this compacted context" in str(compaction["replacement_input"])


@pytest.mark.asyncio
async def test_agent_compaction_falls_back_to_current_level(tmp_path: Path) -> None:
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
            "fast": LevelConfig(name="fast", model="default", params={}),
            "deep": LevelConfig(name="deep", model="default", params={}),
        },
        runtime=RuntimeConfig(
            default_level="fast",
            compression=CompressionConfig(enabled=True, trigger_ratio=0.1, min_tokens=1),
            title_generation=TitleGenerationConfig(enabled=False),
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
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / ".uv-agent", config=config.runner),
        thread_store=ThreadStore(tmp_path / ".uv-agent"),
        project_root=project_root,
    )

    [event async for event in engine.run_turn(user_text="hello", level="deep")]

    assert client.requests[1]["level"] == "deep"


def test_compaction_replacement_keeps_recent_user_messages_with_budget(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root)
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    old_text = "old " * 30_000
    recent_text = "recent request"
    input_items = [
        message_item("user", old_text),
        message_item("user", "<workspace_rule_index>\nAGENTS.md\n</workspace_rule_index>"),
        message_item("assistant", "assistant output"),
        message_item("user", recent_text),
    ]
    response = parse_responses_response(
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
        }
    )

    replacement = engine._compaction_replacement_input(input_items, response)
    text = str(replacement)

    assert recent_text in text
    assert "workspace_rule_index" not in text
    assert "assistant output" not in text
    assert "[truncated during context compaction]" in text
    assert "The messages above may include several earlier user messages" in text


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
        runtime=RuntimeConfig(compression=CompressionConfig(enabled=False)),
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
    stored_events = engine.thread_store.read(thread_id)
    assert any(event["type"] == "turn.interrupted" for event in stored_events)
    assert not any(event["type"] == "item.assistant_delta" for event in stored_events)

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
async def test_agent_generates_title_for_default_new_thread(tmp_path: Path) -> None:
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
        levels={
            "medium": LevelConfig(name="medium", model="default", params={}),
            "small": LevelConfig(name="small", model="default", params={}),
        },
        runtime=RuntimeConfig(compression=CompressionConfig(enabled=False)),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    client = RoutedModelClient(
        main={
            "id": "resp_1",
            "output_text": "done",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
        },
        title={
            "id": "resp_title",
            "output_text": '"Fix import error in runner"',
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": '"Fix import error in runner"'}],
                }
            ],
        },
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="fix the import error in runner")]
    thread_id = str(events[-1]["thread_id"])

    assert any(event["type"] == "thread.title" for event in events)
    assert engine.thread_store.thread_digest(thread_id)["title"] == "Fix import error in runner"
    title_request = next(
        request for request in client.requests if "Generate a short thread title" in str(request["instructions"])
    )
    assert title_request["level"] is None
    assert "first message" in str(title_request["input"])


@pytest.mark.asyncio
async def test_agent_does_not_replace_manual_thread_title(tmp_path: Path) -> None:
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
        levels={
            "medium": LevelConfig(name="medium", model="default", params={}),
            "small": LevelConfig(name="small", model="default", params={}),
        },
        runtime=RuntimeConfig(
            compression=CompressionConfig(enabled=False),
            title_generation=TitleGenerationConfig(enabled=False),
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
                "output_text": "done",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
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
    thread_id = engine.thread_store.create_thread("Manual title")

    events = [event async for event in engine.run_turn(user_text="please rename nothing", thread_id=thread_id)]

    assert not any(event["type"] == "thread.title" for event in events)
    assert engine.thread_store.thread_digest(thread_id)["title"] == "Manual title"
    assert len(client.requests) == 1


@pytest.mark.asyncio
async def test_agent_uses_configured_title_generation_level(tmp_path: Path) -> None:
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
        levels={
            "medium": LevelConfig(name="medium", model="default", params={}),
            "title": LevelConfig(name="title", model="default", params={}),
        },
        runtime=RuntimeConfig(
            compression=CompressionConfig(enabled=False),
            title_generation=TitleGenerationConfig(model_level="title"),
        ),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    client = RoutedModelClient(
        main={
            "id": "resp_1",
            "output_text": "done",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
        },
        title={
            "id": "resp_title",
            "output_text": "Configured title",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Configured title"}],
                }
            ],
        },
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    [event async for event in engine.run_turn(user_text="use custom title model")]

    title_request = next(
        request for request in client.requests if "Generate a short thread title" in str(request["instructions"])
    )
    assert title_request["level"] == "title"


@pytest.mark.asyncio
async def test_agent_generates_title_only_for_first_user_message(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root, title_generation=TitleGenerationConfig(enabled=True))
    client = RoutedModelClient(
        main={
            "id": "resp_1",
            "output_text": "done",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
        },
        title={
            "id": "resp_title",
            "output_text": "Generated title",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Generated title"}],
                }
            ],
        },
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    first_events = [event async for event in engine.run_turn(user_text="first")]
    thread_id = str(first_events[-1]["thread_id"])
    [event async for event in engine.run_turn(user_text="second", thread_id=thread_id)]

    title_requests = [
        request
        for request in client.requests
        if "Generate a short thread title" in str(request.get("instructions") or "")
    ]
    assert len(title_requests) == 1
    assert engine.thread_store.thread_digest(thread_id)["title"] == "Generated title"


@pytest.mark.asyncio
async def test_agent_title_generation_falls_back_to_current_level(tmp_path: Path) -> None:
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
        levels={
            "fast": LevelConfig(name="fast", model="default", params={}),
            "deep": LevelConfig(name="deep", model="default", params={}),
        },
        runtime=RuntimeConfig(default_level="fast", compression=CompressionConfig(enabled=False)),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    client = RoutedModelClient(
        main={
            "id": "resp_1",
            "output_text": "done",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
        },
        title={
            "id": "resp_title",
            "output_text": "Current level title",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Current level title"}],
                }
            ],
        },
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    [event async for event in engine.run_turn(user_text="use current title level", level="deep")]

    title_request = next(
        request for request in client.requests if "Generate a short thread title" in str(request["instructions"])
    )
    assert title_request["level"] == "deep"


@pytest.mark.asyncio
async def test_agent_attaches_user_turn_images(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    image = tmp_path / "clipboard.png"
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
        runtime=RuntimeConfig(
            compression=CompressionConfig(enabled=False),
            title_generation=TitleGenerationConfig(enabled=False),
        ),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    client = FakeModelClient(
        [
            {
                "id": "resp_image",
                "output_text": "seen",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "seen"}],
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

    events = [event async for event in engine.run_turn(user_text="look", image_paths=[image])]
    thread_id = str(events[-1]["thread_id"])

    assert any(event["type"] == "image.attachment" for event in events)
    assert any(event["type"] == "item.image_attachment" for event in engine.thread_store.read(thread_id))
    assert any(
        content.get("type") == "input_image"
        for item in client.requests[0]["input"]
        for content in item.get("content", [])
    )


@pytest.mark.asyncio
async def test_agent_runs_python_tool_boundary(tmp_path: Path) -> None:
    project_root = Path.cwd()
    config = load_config(project_root, [])
    config = AppConfig(
        providers=config.providers,
        models=config.models,
        levels=config.levels,
        runtime=RuntimeConfig(
            compression=CompressionConfig(enabled=False),
            title_generation=TitleGenerationConfig(enabled=False),
        ),
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
    assert client.requests[1]["previous_response_id"] == "resp_1"
    tool_output = client.requests[1]["input"][-1]
    assert tool_output["type"] == "function_call_output"
    assert "observed" in tool_output["output"]
    thread_id = events[-1]["thread_id"]
    stored_events = engine.thread_store.read(thread_id)
    assert any(event["type"] == "item.tool_output" for event in stored_events)
    assert not any(event["type"] == "item.tool_call" for event in stored_events)
    assert not any(event["type"] == "item.assistant_delta" for event in stored_events)
    assert not any(event["type"] == "item.reasoning_delta" for event in stored_events)


@pytest.mark.asyncio
async def test_agent_displays_and_reconstructs_mixed_text_tool_response(tmp_path: Path) -> None:
    project_root = Path.cwd()
    config = make_test_config(project_root)
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / ".uv-agent",
        config=RunnerConfig(
            runtime_dependency=f"uv-agent @ {project_root.resolve().as_uri()}",
            runtime_package_name="uv-agent",
            default_timeout_s=30,
        ),
    )
    client = CompletedOnlyStreamClient(
        [
            {
                "id": "resp_1",
                "output_text": "I will run Python now.",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "I will run Python now."}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "run_python",
                        "arguments": "{\"code\":\"print('mixed')\\n\"}",
                    },
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
    event_types = [event["type"] for event in events]
    thread_id = events[-1]["thread_id"]
    stored_response = next(
        event for event in engine.thread_store.read(thread_id) if event["type"] == "item.model_response"
    )
    reconstructed = engine._reconstruct_input(thread_id)

    assert events[event_types.index("assistant.delta")]["text"] == "I will run Python now."
    assert event_types.index("assistant.delta") < event_types.index("tool.started")
    assert stored_response["output"][0]["type"] == "message"
    assert stored_response["output"][1]["type"] == "function_call"
    assert not any(event["type"] == "item.tool_call" for event in engine.thread_store.read(thread_id))
    reconstructed_message_index = reconstructed.index(stored_response["output"][0])
    assert reconstructed[reconstructed_message_index + 1] == stored_response["output"][1]
    assert reconstructed[reconstructed_message_index + 2]["type"] == "function_call_output"
    assert "mixed" in reconstructed[reconstructed_message_index + 2]["output"]


@pytest.mark.asyncio
async def test_responses_turn_uses_previous_response_id_for_follow_up(tmp_path: Path) -> None:
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
        runtime=RuntimeConfig(
            compression=CompressionConfig(enabled=False),
            title_generation=TitleGenerationConfig(enabled=False),
        ),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
            default_timeout_s=30,
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
        ]
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    thread_id = [event async for event in engine.run_turn(user_text="one")][-1]["thread_id"]
    [event async for event in engine.run_turn(user_text="two", thread_id=thread_id)]

    assert client.requests[0]["previous_response_id"] is None
    assert client.requests[1]["previous_response_id"] == "resp_1"
    assert "one" not in str(client.requests[1]["input"])
    assert "two" in str(client.requests[1]["input"])


@pytest.mark.asyncio
async def test_agent_filters_internal_events_from_model_tool_output(tmp_path: Path) -> None:
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
        runtime=RuntimeConfig(
            compression=CompressionConfig(enabled=False),
            title_generation=TitleGenerationConfig(enabled=False),
        ),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {project_root.resolve().as_uri()}",
            runtime_package_name="uv-agent",
            default_timeout_s=30,
        ),
    )
    runner = PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner)
    client = FakeModelClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "run_python",
                        "arguments": json.dumps(
                            {
                                "code": (
                                    "from uv_agent_runtime import emit_progress\n"
                                    "print('visible before')\n"
                                    "emit_progress('internal progress')\n"
                                    "print('visible after')\n"
                                )
                            }
                        ),
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

    events = [event async for event in engine.run_turn(user_text="run event")]

    model_payload = json.loads(client.requests[1]["input"][-1]["output"])
    assert model_payload["stdout"].replace("\r\n", "\n") == "visible before\nvisible after\n"
    assert "events" not in model_payload
    assert "run_log_path" not in model_payload

    display_payload = json.loads(
        next(event for event in events if event["type"] == "tool.output")["output"]["output"]
    )
    assert display_payload["events"] == [{"kind": "progress", "message": "internal progress"}]
    assert '"kind": "progress"' in display_payload["stdout"]

    stored = engine.thread_store.read(events[-1]["thread_id"])
    runner_result = next(event["result"] for event in stored if event["type"] == "item.runner_result")
    assert runner_result["events"] == [{"kind": "progress", "message": "internal progress"}]


@pytest.mark.asyncio
async def test_enter_dir_loads_rules_in_tool_result_and_persists_cwd(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    src = project_root / "src"
    src.mkdir()
    (src / "AGENTS.md").write_text("Use src rule.", encoding="utf-8")
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
        runtime=RuntimeConfig(
            default_level="medium",
            compression=CompressionConfig(enabled=False),
            title_generation=TitleGenerationConfig(enabled=False),
        ),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    runner = PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner)
    client = FakeModelClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "run_python",
                        "arguments": json.dumps(
                            {
                                "code": (
                                    "from uv_agent_runtime import enter_dir\n"
                                    "from pathlib import Path\n"
                                    "enter_dir('src')\n"
                                    "print(Path.cwd().name)\n"
                                )
                            }
                        ),
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
            {
                "id": "resp_3",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "run_python",
                        "arguments": json.dumps(
                            {"code": "from pathlib import Path\nprint(Path.cwd().name)\n"}
                        ),
                    }
                ],
            },
            {
                "id": "resp_4",
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

    first_events = [event async for event in engine.run_turn(user_text="enter")]
    thread_id = first_events[-1]["thread_id"]
    first_tool_payload = json.loads(client.requests[1]["input"][-1]["output"])

    assert "rules_loaded" in first_tool_payload
    assert "Use src rule." in str(first_tool_payload["rules_loaded"])
    assert "events" not in first_tool_payload
    assert '"kind": "enter_dir"' not in first_tool_payload["stdout"]

    [event async for event in engine.run_turn(user_text="again", thread_id=thread_id)]
    second_tool_payload = json.loads(client.requests[3]["input"][-1]["output"])

    assert "rules_loaded" not in second_tool_payload
    assert second_tool_payload["stdout"].strip() == "src"


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
        runtime=RuntimeConfig(
            compression=CompressionConfig(enabled=False),
            title_generation=TitleGenerationConfig(enabled=False),
        ),
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
    assert "<model_levels>" in prompt
    assert "<response_style>" in prompt
    assert "reply concisely and with a friendly, approachable tone" in prompt
    assert "<default>medium</default>" in prompt
    assert "<level>small</level>" in prompt
    assert "<level>medium</level>" in prompt
    assert "</runtime_helpers>" in prompt
    assert 'requires-python = ">=3.12"' in prompt
    assert '# dependencies = [' in prompt
    assert "plain Python source without a metadata block" in prompt
    assert "not a temporary-script wrapper" in prompt
    assert "Do not rely on the system to truncate oversized output for you" in prompt
    assert "custom patch envelope" in prompt
    assert "*** Begin Patch" in prompt
    assert "*** Update File: path.txt" in prompt
    assert "nested uv-agent subagent" in prompt
    assert "connect_named(\"files\")" in prompt
    assert "client.call_tool" in prompt
    assert "result = ask(" in prompt
    assert 'level="small"' not in prompt
    assert "pathlib" in prompt
    assert "Mentions are plain-text hints only" in prompt
    assert "saved_scripts(limit=32)" in prompt
    assert "read_text, write_text" not in prompt
    assert "list_files" not in prompt
    assert "run_command/check_command" not in prompt
    assert "emit_event" not in prompt
    assert "Directory rules from AGENTS files are loaded automatically" in prompt
    assert "Skills and MCP declarations are appended only when first seen" in prompt
    assert "enter_dir" in prompt
    assert "demo (project)" not in prompt

    turn_context = engine._turn_context_text()

    assert "demo (project)" in turn_context
    assert "available_mcp_servers" in turn_context


def test_agent_prompt_lists_configured_model_levels_without_fixed_examples(tmp_path: Path) -> None:
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
        levels={
            "fast": LevelConfig(name="fast", model="default", params={}),
            "deep": LevelConfig(name="deep", model="default", params={}),
        },
        runtime=RuntimeConfig(default_level="deep", compression=CompressionConfig(enabled=False)),
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

    prompt = engine.system_instructions()

    assert "<default>deep</default>" in prompt
    assert "<level>fast</level>" in prompt
    assert "<level>deep</level>" in prompt
    assert 'level="small"' not in prompt
    assert 'model_level="large"' not in prompt
    assert "small/medium/large" not in prompt


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
        runtime=RuntimeConfig(
            default_level="medium",
            compression=CompressionConfig(enabled=False),
            title_generation=TitleGenerationConfig(enabled=False),
        ),
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
async def test_agent_loads_project_rules_without_context_update(tmp_path: Path) -> None:
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
        runtime=RuntimeConfig(
            default_level="medium",
            compression=CompressionConfig(enabled=False),
            title_generation=TitleGenerationConfig(enabled=False),
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
    assert "<workspace_rule_index>" in request_text
    assert "<directory_rules_loaded>" in request_text
    stored_text = str(engine.thread_store.read(events[-1]["thread_id"]))
    assert "Use the local rule." in stored_text
    stored = engine.thread_store.read(events[-1]["thread_id"])
    assert any(event["type"] == "item.rules_loaded" for event in stored)
    assert not any(event["type"] == "item.context_update" and "Use the local rule." in str(event) for event in stored)


@pytest.mark.asyncio
async def test_compaction_request_reuses_main_prefix(tmp_path: Path) -> None:
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
            compression=CompressionConfig(enabled=True, model_level="small", trigger_ratio=0.1, min_tokens=1),
            title_generation=TitleGenerationConfig(enabled=False),
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
    assert client.requests[0]["input"] == client.requests[1]["input"][: len(client.requests[0]["input"])]
    assert "context_compaction_request" in str(client.requests[1]["input"][-1])
    assert "CONTEXT CHECKPOINT COMPACTION" in str(client.requests[1]["input"][-1])


@pytest.mark.asyncio
async def test_project_rules_are_deduped_and_not_reloaded_on_file_change(tmp_path: Path) -> None:
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
        runtime=RuntimeConfig(
            default_level="medium",
            compression=CompressionConfig(enabled=False),
            title_generation=TitleGenerationConfig(enabled=False),
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
    assert client.requests[1]["previous_response_id"] == "resp_1"
    assert "Rule v1." not in requests_text[1]

    rules.write_text("Rule v2.", encoding="utf-8")
    [event async for event in engine.run_turn(user_text="three", thread_id=first)]
    assert client.requests[2]["previous_response_id"] == "resp_2"
    assert "Rule v2." not in str(client.requests[2]["input"])

    rules.unlink()
    [event async for event in engine.run_turn(user_text="four", thread_id=first)]
    assert client.requests[3]["previous_response_id"] == "resp_3"
    assert "Do not rely on older appended" not in str(client.requests[3]["input"])


@pytest.mark.asyncio
async def test_project_rules_reappear_after_compaction_epoch(tmp_path: Path) -> None:
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
        runtime=RuntimeConfig(default_level="medium", compression=CompressionConfig(enabled=False)),
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
    first = engine._pre_user_context_items(thread_id)
    engine.thread_store.append(thread_id, "item.compaction", turn_id="t1", text="summary", usage={})
    engine._reset_rule_epoch(thread_id)
    second = engine._pre_user_context_items(thread_id)

    assert first
    assert second
    assert "After compaction rule." in str(second)
    assert "<workspace_rule_index>" in str(second)


@pytest.mark.asyncio
async def test_compaction_epoch_uses_active_cwd_local_index_and_notice(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    child = project_root / "src"
    nested = child / "pkg"
    nested.mkdir(parents=True)
    (project_root / "AGENTS.md").write_text("Root rule.", encoding="utf-8")
    (child / "AGENTS.md").write_text("Child rule.", encoding="utf-8")
    (nested / "AGENTS.md").write_text("Nested rule.", encoding="utf-8")
    config = make_test_config(project_root)
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread()
    state = engine._rule_state(thread_id)
    state.active_cwd = child.resolve()
    engine.thread_store.append(thread_id, "thread.cwd_updated", turn_id="t1", cwd=str(child.resolve()))
    engine.thread_store.append(thread_id, "item.compaction", turn_id="t1", text="summary", usage={})
    engine._reset_rule_epoch(thread_id)

    items = engine._pre_user_context_items(thread_id)
    text = str(items)

    assert "<workspace_rule_index>" in text
    assert "AGENTS.md" in text
    assert "pkg/AGENTS.md" in text
    assert "Root rule." not in text
    assert "Child rule." in text
    assert "active_cwd_notice" in text
    assert "src" in text


@pytest.mark.asyncio
async def test_system_instructions_are_persisted_before_first_model_request(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root, api="chat_completions")
    client = FakeModelClient(
        [
            {
                "id": "chatcmpl_1",
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
                "id": "chatcmpl_2",
                "output_text": "two",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "two"}],
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

    thread_id = str([event async for event in engine.run_turn(user_text="one")][-1]["thread_id"])
    stored = engine.thread_store.read(thread_id)
    system_index = next(
        index for index, event in enumerate(stored) if event["type"] == "item.system_instructions"
    )
    turn_index = next(index for index, event in enumerate(stored) if event["type"] == "turn.started")
    frozen = stored[system_index]["text"]
    assert system_index < turn_index
    assert client.requests[0]["instructions"] == frozen

    engine.config = make_test_config(project_root, api="chat_completions", default_level="small")
    [event async for event in engine.run_turn(user_text="two", thread_id=thread_id)]

    stored_after = engine.thread_store.read(thread_id)
    assert sum(1 for event in stored_after if event["type"] == "item.system_instructions") == 1
    assert client.requests[1]["instructions"] == frozen
    assert "<default>medium</default>" in client.requests[1]["instructions"]
    assert "<default>small</default>" not in client.requests[1]["instructions"]


@pytest.mark.asyncio
async def test_system_instructions_refresh_after_compaction(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root, api="chat_completions")
    client = FakeModelClient(
        [
            {
                "id": "chatcmpl_1",
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
                "id": "chatcmpl_2",
                "output_text": "two",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "two"}],
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

    thread_id = str([event async for event in engine.run_turn(user_text="one")][-1]["thread_id"])
    engine.thread_store.append(thread_id, "item.compaction", turn_id="t1", text="summary", usage={})
    engine.config = make_test_config(project_root, api="chat_completions", default_level="small")
    [event async for event in engine.run_turn(user_text="two", thread_id=thread_id)]

    assert "<default>medium</default>" in client.requests[0]["instructions"]
    assert "<default>small</default>" in client.requests[1]["instructions"]
    stored = engine.thread_store.read(thread_id)
    assert sum(1 for event in stored if event["type"] == "item.system_instructions") == 2


def test_workspace_context_reappears_after_compaction_epoch(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    skill_dir = project_root / ".agents" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo\nUse this for demo work.\n", encoding="utf-8")
    config = make_test_config(project_root)
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

    assert "demo (project)" in str(first)
    assert "demo (project)" in str(second)


def test_workspace_context_is_not_repeated_after_compaction_epoch_update(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    skill_dir = project_root / ".agents" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo\nUse this for demo work.\n", encoding="utf-8")
    config = make_test_config(project_root)
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread()

    engine._workspace_context_items(thread_id)
    engine.thread_store.append(thread_id, "item.compaction", turn_id="t1", text="summary", usage={})
    after_compaction = engine._workspace_context_items(thread_id)
    repeated = engine._workspace_context_items(thread_id)

    assert "demo (project)" in str(after_compaction)
    assert repeated == []


@pytest.mark.asyncio
async def test_compaction_uses_persisted_system_instructions(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        context_window_tokens=20,
        compression=CompressionConfig(enabled=True, model_level="small", trigger_ratio=0.1, min_tokens=1),
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

    thread_id = str([event async for event in engine.run_turn(user_text="hello")][-1]["thread_id"])
    frozen = next(
        event["text"]
        for event in engine.thread_store.read(thread_id)
        if event["type"] == "item.system_instructions"
    )

    assert client.requests[0]["instructions"] == frozen
    assert client.requests[1]["instructions"] == frozen
    assert client.requests[0]["instructions"] == client.requests[1]["instructions"]


@pytest.mark.asyncio
async def test_agent_persists_completed_reasoning_text(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root)
    client = ReasoningStreamClient([])
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    thread_id = str([event async for event in engine.run_turn(user_text="hello")][-1]["thread_id"])
    response = next(
        event for event in engine.thread_store.read(thread_id) if event["type"] == "item.model_response"
    )

    assert response["reasoning_text"] == "provider reasoning"


def test_reconstruct_input_uses_compaction_replacement_input(tmp_path: Path) -> None:
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
        runtime=RuntimeConfig(compression=CompressionConfig(enabled=False)),
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
    replacement = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "kept"}]},
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "<conversation_summary>\nsummary\n</conversation_summary>",
                }
            ],
        },
    ]
    engine.thread_store.append(
        thread_id,
        "item.compaction",
        turn_id="t1",
        text="summary",
        replacement_input=replacement,
        usage={},
    )
    engine.thread_store.append(thread_id, "item.user", turn_id="t2", item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "new"}]})

    reconstructed = engine._reconstruct_input(thread_id)
    text = str(reconstructed)

    assert reconstructed[: len(replacement)] == replacement
    assert "kept" in text
    assert "summary" in text
    assert "new" in text
    assert "old" not in text


def test_context_update_reconstructs_as_stable_prefix(tmp_path: Path) -> None:
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
        runtime=RuntimeConfig(compression=CompressionConfig(enabled=False)),
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
    engine.thread_store.append(
        thread_id,
        "item.context_update",
        turn_id="t1",
        context_fingerprint="fp",
        context_state={"fingerprint": "fp", "parts": {"rules": "rules-fp"}},
        context_kind="workspace",
        removed=[],
        text="stable rules",
    )
    engine.thread_store.append(thread_id, "item.user", turn_id="t1", item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]})

    reconstructed = engine._reconstruct_input(thread_id)

    assert reconstructed[0]["role"] == "user"
    assert "stable rules" in str(reconstructed[0])
    assert "hello" in str(reconstructed[1])


def test_rules_loaded_from_tool_result_is_not_reconstructed_between_tool_call_and_output(tmp_path: Path) -> None:
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
        runtime=RuntimeConfig(compression=CompressionConfig(enabled=False)),
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
    engine.thread_store.append(
        thread_id,
        "item.model_response",
        turn_id="t1",
        output=[
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "run_python",
                "arguments": "{}",
            }
        ],
    )
    engine.thread_store.append(
        thread_id,
        "item.rules_loaded",
        turn_id="t1",
        source="tool_result",
        text="must not become user message",
    )
    engine.thread_store.append(
        thread_id,
        "item.tool_output",
        turn_id="t1",
        item={"type": "function_call_output", "call_id": "call_1", "output": "{}"},
    )

    reconstructed = engine._reconstruct_input(thread_id)

    assert reconstructed[0]["type"] == "function_call"
    assert reconstructed[1]["type"] == "function_call_output"
    assert "must not become user message" not in str(reconstructed)


def test_context_update_is_reanchored_before_next_user_when_reconstructing(tmp_path: Path) -> None:
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
        runtime=RuntimeConfig(compression=CompressionConfig(enabled=False)),
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
    engine.thread_store.append(
        thread_id,
        "item.model_response",
        turn_id="t1",
        output=[
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "run_python",
                "arguments": "{}",
            }
        ],
    )
    engine.thread_store.append(
        thread_id,
        "item.context_update",
        turn_id="t1",
        context_fingerprint="fp",
        context_state={"fingerprint": "fp", "parts": {"skills": "s"}},
        text="capability update",
    )
    engine.thread_store.append(
        thread_id,
        "item.tool_output",
        turn_id="t1",
        item={"type": "function_call_output", "call_id": "call_1", "output": "{}"},
    )
    engine.thread_store.append(
        thread_id,
        "item.user",
        turn_id="t2",
        item=message_item("user", "next"),
    )

    reconstructed = engine._reconstruct_input(thread_id)

    assert reconstructed[0]["type"] == "function_call"
    assert reconstructed[1]["type"] == "function_call_output"
    assert message_item_text(reconstructed[2]) == "capability update"
    assert message_item_text(reconstructed[3]) == "next"


def test_rule_state_restore_uses_local_index_when_active_cwd_is_child(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    child = project_root / "src"
    child.mkdir(parents=True)
    nested = child / "pkg"
    nested.mkdir()
    (child / "AGENTS.md").write_text("child rule", encoding="utf-8")
    (nested / "AGENTS.md").write_text("nested rule", encoding="utf-8")
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
        runtime=RuntimeConfig(compression=CompressionConfig(enabled=False)),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    store = ThreadStore(tmp_path / "state")
    thread_id = store.create_thread()
    store.append(thread_id, "thread.cwd_updated", turn_id="t1", cwd=str(child))

    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=store,
        project_root=project_root,
    )

    items = engine._pre_user_context_items(thread_id)
    text = str(items)

    assert "child rule" in text
    assert "<workspace_rule_index>" in text
    assert "AGENTS.md" in text
    assert "pkg/AGENTS.md" in text
    assert "active_cwd_notice" in text


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
