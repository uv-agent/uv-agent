from __future__ import annotations

import json
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import openai

from uv_agent.agent import (
    AgentEngine,
    PYTHON_TOOL,
    clean_branch_slug,
    message_item,
    message_item_text,
    model_tool_payload,
    tool_attachment_context_items,
    usage_token_count,
)
from uv_agent.agent.compaction import (
    compaction_judge_request_item,
    compaction_response_summary_text,
    retain_item_after_compaction,
    retain_recent_context,
    strip_compaction_judge_history,
)
from uv_agent.prompts import (
    BRANCH_NAME_GENERATION_PROMPT,
    BRANCH_SLUG_INSTRUCTION,
    COMPACTED_CONTEXT_CONTINUATION,
    COMPACTION_TRUNCATION_SUFFIX,
    INTERRUPTED_STREAM_CONTEXT_BRIDGE,
    INTERRUPTED_TOOL_CONTEXT_BRIDGE,
    POST_TOOL_COMPACTION_BRIDGE,
    THREAD_TITLE_INSTRUCTION,
    TOKEN_ESTIMATION_WARNING,
    TOOL_ATTACHMENT_CONTEXT_BRIDGE,
)
from uv_agent.billing import billing_charge_for_usage, billing_token_breakdown, format_billing_total
from uv_agent.config import (
    AppConfig,
    CompressionConfig,
    LevelConfig,
    ModelConfig,
    ModelPricingConfig,
    PricingConfig,
    ProviderConfig,
    RunnerConfig,
    RuntimeConfig,
    StreamRetryConfig,
    BranchNameGenerationConfig,
    TitleGenerationConfig,
    load_config,
)
from uv_agent.errors import EmptyModelStreamError, format_error, is_retryable_provider_error
from uv_agent.mcp_config import McpInstructionsPreview
from uv_agent.model import (
    FakeModelClient,
    ModelStreamEvent,
    anthropic_messages,
    chat_messages,
    parse_responses_response,
)
from uv_agent.runner import PythonRunner
from uv_agent.runner.models import PythonRunRequest, PythonRunResult
from uv_agent.session import ThreadLockedError, ThreadStore


class BlockingModelClient(FakeModelClient):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()

    async def stream_response(self, **kwargs):
        self.started.set()
        await asyncio.Event().wait()
        yield


class PartialStreamClient(FakeModelClient):
    def __init__(self, event: ModelStreamEvent) -> None:
        super().__init__([])
        self.event = event
        self.started = asyncio.Event()
        self.delivered = asyncio.Event()

    async def stream_response(self, **kwargs):
        self.started.set()
        yield self.event
        self.delivered.set()
        await asyncio.Event().wait()


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
        if THREAD_TITLE_INSTRUCTION in instructions and self.title is not None:
            self.requests.append(request)
            return parse_responses_response(self.title)
        self.requests.append(request)
        return parse_responses_response(self.main)


class HangingResponseClient(FakeModelClient):
    async def create_response(self, **kwargs):
        self.requests.append(kwargs)
        await asyncio.Event().wait()


class GateResponseClient(FakeModelClient):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(responses)
        self.request_started = asyncio.Event()
        self.release = asyncio.Event()

    async def create_response(self, **kwargs):
        self.request_started.set()
        await self.release.wait()
        return await super().create_response(**kwargs)


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


class FakeMcpInstructionsProbe:
    def __init__(
        self,
        instructions: dict[tuple[str, str, str], McpInstructionsPreview] | None = None,
    ) -> None:
        self.instructions = instructions or {}
        self.started = False

    def start(self) -> None:
        self.started = True

    def snapshot(self) -> dict[tuple[str, str, str], McpInstructionsPreview]:
        return dict(self.instructions)


class DelayedPluginManager:
    def __init__(self, engine: AgentEngine) -> None:
        self.engine = engine
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.start_count = 0

    def start_background(self) -> asyncio.Task[None]:
        self.start_count += 1
        return asyncio.create_task(self._start())

    async def _start(self) -> None:
        self.started.set()
        await self.release.wait()
        self.engine.runtime_helpers.register(
            plugin="delayed-plugin",
            name="delayed_helper",
            fn=lambda: None,
            doc="Delayed helper.",
        )

    async def stop(self) -> None:
        self.release.set()

    def helper_specs(self):
        return self.engine.runtime_helpers.list()

    def resolve_helper(self, name: str) -> dict[str, Any]:
        return self.engine.runtime_helpers.resolve_payload(name)


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


class FailingStreamClient(FakeModelClient):
    def __init__(self, exc: BaseException) -> None:
        super().__init__([])
        self.exc = exc

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
        if False:
            yield None
        raise self.exc


class EmptyThenSuccessStreamClient(FakeModelClient):
    def __init__(self, failures: int) -> None:
        super().__init__([])
        self.failures = failures

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
        if len(self.requests) <= self.failures:
            if False:
                yield None
            raise EmptyModelStreamError("empty stream")
        yield ModelStreamEvent(type="text_delta", text="done")
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


def openai_connection_error(message: str = "network down") -> openai.APIConnectionError:
    exc = openai.APIConnectionError.__new__(openai.APIConnectionError)
    Exception.__init__(exc, message)
    exc.message = message
    exc.body = None
    return exc


def openai_status_error(
    status_code: int,
    message: str,
    body: object | None,
    *,
    error_cls: type[openai.APIStatusError] = openai.APIStatusError,
) -> openai.APIStatusError:
    # Construct SDK exceptions without calling their network-response-heavy
    # initializers; the error formatter only needs the public attributes real
    # SDK instances expose.
    exc = error_cls.__new__(error_cls)
    Exception.__init__(exc, message)
    exc.message = message
    exc.body = body
    exc.status_code = status_code
    exc.response = SimpleNamespace(
        status_code=status_code,
        reason_phrase=message,
        text=json.dumps(body or {}, ensure_ascii=False),
    )
    return exc


class LookAtRunner:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path

    async def run(self, request: PythonRunRequest) -> PythonRunResult:
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return PythonRunResult(
            run_id="run_look",
            returncode=0,
            stdout="created image\n",
            stderr="",
            timed_out=False,
            interrupted=False,
            truncated=False,
            script_path=self.image_path.parent / "script.py",
            events=[
                {
                    "kind": "look_at",
                    "path": str(self.image_path),
                    "note": "inspect",
                }
            ],
        )

class SimpleRunner:
    def __init__(self, *, interrupted: bool = False) -> None:
        self.requests: list[PythonRunRequest] = []
        self.interrupted = interrupted

    async def run(self, request: PythonRunRequest) -> PythonRunResult:
        self.requests.append(request)
        return PythonRunResult(
            run_id="run_simple",
            returncode=0,
            stdout="simple\n",
            stderr="",
            timed_out=False,
            interrupted=self.interrupted,
            truncated=False,
            script_path=request.cwd / "script.py",
            events=[],
        )


class LargeOutputRunner(SimpleRunner):
    def __init__(self, stdout: str) -> None:
        super().__init__()
        self.stdout = stdout

    async def run(self, request: PythonRunRequest) -> PythonRunResult:
        self.requests.append(request)
        return PythonRunResult(
            run_id="run_large",
            returncode=0,
            stdout=self.stdout,
            stderr="",
            timed_out=False,
            interrupted=False,
            truncated=False,
            script_path=request.cwd / "script.py",
            events=[],
        )


class StreamingRunner(SimpleRunner):
    async def stream_run(self, request: PythonRunRequest):
        self.requests.append(request)
        partial = PythonRunResult(
            run_id="run_stream",
            returncode=None,
            stdout="partial output\n",
            stderr="",
            timed_out=False,
            interrupted=False,
            truncated=False,
            script_path=request.cwd / "script.py",
            events=[],
        )
        yield SimpleNamespace(type="run.partial", data={"result": partial, "reason": "interval"})
        final = PythonRunResult(
            run_id="run_stream",
            returncode=0,
            stdout="partial output\nfinal output\n",
            stderr="",
            timed_out=False,
            interrupted=False,
            truncated=False,
            script_path=request.cwd / "script.py",
            events=[],
        )
        yield SimpleNamespace(type="run.completed", data={"result": final})



def make_test_config(
    project_root: Path,
    *,
    api: str = "responses",
    compression_enabled: bool = False,
    context_window_tokens: int = 100_000,
    compression: CompressionConfig | None = None,
    title_generation: TitleGenerationConfig | None = None,
    branch_name_generation: BranchNameGenerationConfig | None = None,
    default_level: str = "medium",
    stream_retry: StreamRetryConfig | None = None,
    pricing: PricingConfig | None = None,
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
            branch_name_generation=branch_name_generation or BranchNameGenerationConfig(enabled=False),
            stream_retry=stream_retry or StreamRetryConfig(),
        ),
        runner=RunnerConfig(
        ),
        pricing=pricing or PricingConfig(),
    )


def test_agent_exposes_only_python_runner_tool() -> None:
    assert PYTHON_TOOL["name"] == "run_python"
    assert PYTHON_TOOL["type"] == "function"
    assert "全新的 Python 进程" in PYTHON_TOOL["description"]
    assert "活动 cwd" in PYTHON_TOOL["description"]
    assert "Python 原生控制流和 import" in PYTHON_TOOL["description"]
    assert "不是 shell 风格片段" in PYTHON_TOOL["description"]
    assert "完整、独立的 Python 脚本" in PYTHON_TOOL["description"]
    assert "优先使用 runtime helpers" in PYTHON_TOOL["description"]
    assert "普通外部命令" in PYTHON_TOOL["description"] and "run_process_text" in PYTHON_TOOL["description"]
    assert "call subprocesses" not in PYTHON_TOOL["description"]
    assert set(PYTHON_TOOL["parameters"]["properties"]) == {"code", "timeout_s"}
    code_description = PYTHON_TOOL["parameters"]["properties"]["code"]["description"]
    assert "完整、有效的 Python 源码" in code_description
    assert "常规 Python 语法" in code_description
    assert "不要使用 shell 风格伪代码" in code_description
    assert "小型 Python 程序" in code_description
    assert "变量、函数、循环、条件、try/except" in code_description
    assert "runtime helper 调用" in code_description
    assert "script_args" not in PYTHON_TOOL["parameters"]["properties"]
    assert PYTHON_TOOL["parameters"]["required"] == ["code"]


def test_model_tool_payload_filters_only_tagged_runtime_event_lines() -> None:
    payload = {
        "run_id": "run_1",
        "returncode": 0,
        "timed_out": False,
        "interrupted": False,
        "truncated": False,
        "stdout": (
            '{"kind":"user_json"}\n'
            '{"kind":"progress","_uv_agent_event_id":"evt_1","_uv_agent_run_id":"run_1"}\n'
            '{"kind":"progress","_uv_agent_run_id":"run_1"}\n'
            '{"kind":"progress","_uv_agent_run_id":"run_other"}\n'
        ),
        "stderr": "",
        "events": [
            {
                "kind": "progress",
                "message": "working",
                "_uv_agent_event_id": "evt_1",
                "_uv_agent_run_id": "run_1",
            }
        ],
    }

    visible = model_tool_payload(payload)

    assert visible["stdout"] == (
        '{"kind":"user_json"}\n'
        '{"kind":"progress","_uv_agent_run_id":"run_1"}\n'
        '{"kind":"progress","_uv_agent_run_id":"run_other"}\n'
    )
    assert "events" not in visible


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
    assert COMPACTED_CONTEXT_CONTINUATION in str(compaction["replacement_input"])


def test_compaction_summary_falls_back_to_message_text_when_tool_call_is_present() -> None:
    response = parse_responses_response(
        {
            "id": "resp_compact",
            "output_text": "",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "summary before stray tool"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_unwanted",
                    "name": "run_python",
                    "arguments": "{}",
                },
            ],
        }
    )

    assert compaction_response_summary_text(response) == "summary before stray tool"


def test_retain_recent_context_converts_tool_protocol_items_to_messages() -> None:
    retained = retain_recent_context(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "run_python",
                "arguments": '{"code":"print(1)"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "stdout text",
            },
        ],
        K=1_000,
    )

    assert retained
    assert {item["type"] for item in retained} == {"message"}
    retained_text = "\n".join(message_item_text(item) for item in retained)
    assert "<retained_tool_call" in retained_text
    assert "<retained_tool_output" in retained_text
    assert "function_call" not in {item.get("type") for item in retained}


def test_strip_compaction_judge_history_removes_internal_exchange_only() -> None:
    judge_request = compaction_judge_request_item("fresh task")
    judge_response = message_item(
        "assistant",
        '{"remaining_calls_bucket":"60_plus","history_dependency":"low"}',
    )
    real_user = message_item("user", "fresh task")
    real_assistant = message_item("assistant", "real answer")

    filtered = strip_compaction_judge_history(
        [
            message_item("user", "old request"),
            judge_request,
            judge_response,
            real_user,
            real_assistant,
        ]
    )
    filtered_text = "\n".join(message_item_text(item) for item in filtered if item.get("type") == "message")

    assert "<compaction_judge_request>" not in filtered_text
    assert "remaining_calls_bucket" not in filtered_text
    assert [message_item_text(item) for item in filtered if item.get("type") == "message"] == [
        "old request",
        "fresh task",
        "real answer",
    ]


@pytest.mark.asyncio
async def test_cache_aware_judge_replays_completed_judge_before_user_without_counting_as_user(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        compression=CompressionConfig(
            enabled=True,
            cache_aware=True,
            judge_min_context_ratio=0.0,
            min_gain=999_999.0,
        ),
        pricing=PricingConfig(unit="token", models={"fake": ModelPricingConfig(input=1.0, output=1.0, cached_input=0.5)}),
    )
    client = FakeModelClient(
        [
            {
                "id": "resp_judge",
                "output_text": '{"remaining_calls_bucket":"60_plus","history_dependency":"low"}',
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"remaining_calls_bucket":"60_plus","history_dependency":"low"}',
                            }
                        ],
                    }
                ],
            },
            {
                "id": "resp_main",
                "output_text": "answered",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "answered"}],
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

    events = [event async for event in engine.run_turn(user_text="fresh task")]
    thread_id = events[-1]["thread_id"]
    stored = engine.thread_store.read(thread_id)
    main_input = client.requests[1]["input"]
    main_texts = [message_item_text(item) for item in main_input if item.get("type") == "message"]

    assert [event["type"] for event in events if event["type"].startswith("judge.")] == [
        "judge.started",
        "judge.completed",
    ]
    assert [event["type"] for event in stored].count("item.user") == 1
    assert any(event["type"] == "item.judge_request" for event in stored)
    assert any(event["type"] == "item.judge_response" for event in stored)
    assert "<compaction_judge_request>" in "\n".join(main_texts)
    assert '{"remaining_calls_bucket":"60_plus"' in str(main_input)
    assert main_texts[-1] == "fresh task"


@pytest.mark.asyncio
async def test_cache_aware_judge_started_streams_before_judge_response_finishes(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        compression=CompressionConfig(
            enabled=True,
            cache_aware=True,
            judge_min_context_ratio=0.0,
            min_gain=999_999.0,
        ),
        pricing=PricingConfig(unit="token", models={"fake": ModelPricingConfig(input=1.0, output=1.0, cached_input=0.5)}),
    )
    client = GateResponseClient(
        [
            {
                "id": "resp_judge",
                "output_text": '{"remaining_calls_bucket":"60_plus","history_dependency":"low"}',
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"remaining_calls_bucket":"60_plus","history_dependency":"low"}',
                            }
                        ],
                    }
                ],
            },
            {
                "id": "resp_main",
                "output_text": "answered",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "answered"}],
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

    events: list[dict[str, Any]] = []
    turn_events = engine.run_turn(user_text="fresh task")

    events.append(await asyncio.wait_for(anext(turn_events), timeout=1.0))
    events.append(await asyncio.wait_for(anext(turn_events), timeout=1.0))
    assert [event["type"] for event in events] == ["turn.started", "judge.started"]
    assert not client.request_started.is_set()

    next_event = asyncio.create_task(anext(turn_events))
    await asyncio.wait_for(client.request_started.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert not next_event.done()

    client.release.set()
    events.append(await asyncio.wait_for(next_event, timeout=1.0))
    async for event in turn_events:
        events.append(event)

    assert [event["type"] for event in events if event["type"].startswith("judge.")] == [
        "judge.started",
        "judge.completed",
    ]


@pytest.mark.asyncio
async def test_cache_aware_judge_can_compact_below_threshold_and_keeps_current_user_fresh(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        context_window_tokens=10_000,
        compression=CompressionConfig(
            enabled=True,
            cache_aware=True,
            trigger_ratio=0.95,
            min_tokens=1,
            judge_min_context_ratio=0.0,
            min_gain=0.0,
            margin=0.0,
        ),
        pricing=PricingConfig(unit="token", models={"fake": ModelPricingConfig(input=1.0, output=1.0, cached_input=0.5)}),
    )
    client = FakeModelClient(
        [
            {
                "id": "resp_judge",
                "output_text": '{"remaining_calls_bucket":"60_plus","history_dependency":"low"}',
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"remaining_calls_bucket":"60_plus","history_dependency":"low"}',
                            }
                        ],
                    }
                ],
            },
            {
                "id": "resp_compact",
                "output_text": "history summary",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "history summary"}],
                    }
                ],
            },
            {
                "id": "resp_main",
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
    thread_store = ThreadStore(tmp_path / "state")
    thread_id = thread_store.create_thread("Existing")
    thread_store.append(thread_id, "turn.started", turn_id="old")
    thread_store.append(thread_id, "item.user", turn_id="old", item=message_item("user", "old request"))
    thread_store.append(
        thread_id,
        "item.model_response",
        turn_id="old",
        model_api="responses",
        response_id="resp_old",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "old assistant detail " * 1_000}],
            }
        ],
        usage={},
    )
    thread_store.append(thread_id, "turn.completed", turn_id="old", final_text="old assistant detail")
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=thread_store,
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="new work", thread_id=thread_id)]
    stored = engine.thread_store.read(thread_id)
    main_input = client.requests[2]["input"]
    main_texts = [message_item_text(item) for item in main_input if item.get("type") == "message"]

    assert any(event["type"] == "compaction.completed" for event in events)
    assert any(event["type"] == "item.compaction" for event in stored)
    assert client.requests[1]["input"][-1]  # compaction request ran before main response
    assert "<conversation_summary>" in "\n".join(main_texts)
    assert "history summary" in "\n".join(main_texts)
    assert "<compaction_judge_request>" not in "\n".join(main_texts)
    assert main_texts[-1] == "new work"
    assert engine.last_judge_summary()["compacted"] is True



@pytest.mark.asyncio
async def test_cache_aware_judge_skips_when_no_pricing_configured(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        compression=CompressionConfig(
            enabled=True,
            cache_aware=True,
            judge_min_context_ratio=0.0,
        ),
        # Default PricingConfig has no models, so no amount/price is configured.
    )
    client = FakeModelClient(
        [
            {
                "id": "resp_main",
                "output_text": "answered",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "answered"}],
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

    events = [event async for event in engine.run_turn(user_text="fresh task")]
    thread_id = events[-1]["thread_id"]
    stored = engine.thread_store.read(thread_id)

    assert not any(event["type"].startswith("judge.") for event in events)
    assert not any(event["type"].startswith("item.judge") for event in stored)
    assert len(client.requests) == 1
    assert engine.last_judge_summary() == {
        "skipped": True,
        "reason": "no_pricing",
        "total_tokens": engine.last_judge_summary()["total_tokens"],
    }
    assert engine.last_judge_summary()["total_tokens"] > 0


@pytest.mark.asyncio
async def test_compaction_sanitizes_prior_judge_history_without_changing_replay(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        api="chat_completions",
        compression=CompressionConfig(enabled=True, trigger_ratio=0.0, min_tokens=1),
    )
    client = FakeModelClient(
        [
            {
                "id": "resp_main",
                "output_text": "new answer",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "new answer"}],
                    }
                ],
            },
            {
                "id": "resp_compact",
                "output_text": "clean summary",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "clean summary"}],
                    }
                ],
            },
        ]
    )
    thread_store = ThreadStore(tmp_path / "state")
    thread_id = thread_store.create_thread("Existing")
    thread_store.append(thread_id, "turn.started", turn_id="old")
    thread_store.append(thread_id, "item.user", turn_id="old", item=message_item("user", "old request"))
    thread_store.append(
        thread_id,
        "item.judge_request",
        turn_id="old",
        item=compaction_judge_request_item("historical task"),
    )
    thread_store.append(
        thread_id,
        "item.judge_response",
        turn_id="old",
        model_api="responses",
        response_id="resp_judge_old",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"remaining_calls_bucket":"60_plus","history_dependency":"high"}',
                    }
                ],
            }
        ],
        usage={},
    )
    thread_store.append(thread_id, "item.user", turn_id="old", item=message_item("user", "historical task"))
    thread_store.append(
        thread_id,
        "item.model_response",
        turn_id="old",
        model_api="chat_completions",
        response_id="resp_old",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "historical answer"}],
            }
        ],
        usage={},
    )
    thread_store.append(thread_id, "turn.completed", turn_id="old", final_text="historical answer")
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=thread_store,
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="new work", thread_id=thread_id)]
    assert any(event["type"] == "compaction.completed" for event in events)

    replay_text = "\n".join(str(item) for item in client.requests[0]["input"])
    compact_text = "\n".join(str(item) for item in client.requests[1]["input"])
    compaction = next(event for event in engine.thread_store.read(thread_id) if event.get("type") == "item.compaction")
    replacement_text = "\n".join(str(item) for item in compaction.get("replacement_input") or [])

    assert "<compaction_judge_request>" in replay_text
    assert "remaining_calls_bucket" in replay_text
    assert "<compaction_judge_request>" not in compact_text
    assert "remaining_calls_bucket" not in compact_text
    assert "<compaction_judge_request>" not in replacement_text
    assert "remaining_calls_bucket" not in replacement_text

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


@pytest.mark.asyncio
async def test_compaction_trigger_uses_active_level_context_window(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "small-model": ModelConfig(
                name="small-model",
                provider="p",
                model="small-remote",
                context_window_tokens=100,
                params={},
            ),
            "deep-model": ModelConfig(
                name="deep-model",
                provider="p",
                model="deep-remote",
                context_window_tokens=1_000,
                params={},
            ),
        },
        levels={
            "small": LevelConfig(name="small", model="small-model", params={}),
            "deep": LevelConfig(name="deep", model="deep-model", params={}),
        },
        runtime=RuntimeConfig(
            default_level="deep",
            compression=CompressionConfig(enabled=True, model_level="small", trigger_ratio=0.5, min_tokens=1),
            title_generation=TitleGenerationConfig(enabled=False),
        ),
        runner=RunnerConfig(),
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
                "usage": {"total_tokens": 200},
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

    events = [event async for event in engine.run_turn(user_text="hello", level="deep")]

    assert [event["type"] for event in events].count("compaction.started") == 0
    assert len(client.requests) == 1


@pytest.mark.asyncio
async def test_compaction_trigger_prefers_provider_usage(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        context_window_tokens=100,
        compression=CompressionConfig(enabled=True, model_level="small", trigger_ratio=0.5, min_tokens=1),
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
                "usage": {"total_tokens": 75},
            },
            {
                "id": "resp_compact",
                "output_text": "provider summary",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "provider summary"}],
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

    events = [event async for event in engine.run_turn(user_text="tiny")]
    stored = engine.thread_store.read(events[-1]["thread_id"])

    assert any(event["type"] == "item.compaction" for event in stored)
    assert not any(event["type"] == "thread.token_estimation_warning" for event in stored)


@pytest.mark.asyncio
async def test_compaction_warns_when_trigger_uses_estimate(tmp_path: Path) -> None:
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
                "output_text": "estimated summary",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "estimated summary"}],
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
    stored = engine.thread_store.read(events[-1]["thread_id"])
    warning = next(event for event in stored if event["type"] == "thread.token_estimation_warning")

    assert warning["message"] == TOKEN_ESTIMATION_WARNING
    assert any(event["type"] == "item.compaction" for event in stored)
    assert [event["type"] for event in stored].index("thread.token_estimation_warning") < [
        event["type"] for event in stored
    ].index("item.compaction")


@pytest.mark.asyncio
async def test_compaction_uses_provider_usage_even_when_tool_output_followed(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        context_window_tokens=20,
        compression=CompressionConfig(enabled=True, model_level="small", trigger_ratio=0.1, min_tokens=1),
    )
    client = CompletedOnlyStreamClient(
        [
            {
                "id": "resp_tool",
                "output_text": "",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_stale_usage",
                        "name": "run_python",
                        "arguments": json.dumps({"code": "print('hello')"}),
                    }
                ],
                "usage": {"total_tokens": 5},
            },
            {
                "id": "resp_compact",
                "output_text": "summary after stale usage",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "summary after stale usage"}],
                    }
                ],
            },
            {
                "id": "resp_final",
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
        runner=SimpleRunner(),  # type: ignore[arg-type]
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="run a tool")]
    stored = engine.thread_store.read(events[-1]["thread_id"])

    assert not any(event["type"] == "thread.token_estimation_warning" for event in events)
    assert not any(event["type"] == "thread.token_estimation_warning" for event in stored)
    assert any(event["type"] == "item.compaction" for event in stored)


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
    assert COMPACTION_TRUNCATION_SUFFIX.strip() in text
    assert "<retained_history_message" in text
    assert COMPACTED_CONTEXT_CONTINUATION in text
    assert "<compacted_context_continuation>" not in text


@pytest.mark.asyncio
async def test_agent_compacts_after_tool_outputs_before_next_model_request(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        context_window_tokens=20,
        compression=CompressionConfig(enabled=True, model_level="small", trigger_ratio=0.1, min_tokens=1),
    )
    client = CompletedOnlyStreamClient(
        [
            {
                "id": "resp_tool",
                "output_text": "",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_mid_compact",
                        "name": "run_python",
                        "arguments": json.dumps({"code": "print('hello')"}),
                    }
                ],
            },
            {
                "id": "resp_compact",
                "output_text": "summary includes tool result",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "summary includes tool result"}],
                    }
                ],
            },
            {
                "id": "resp_final",
                "output_text": "done after compaction",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done after compaction"}],
                    }
                ],
            },
        ]
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=SimpleRunner(),  # type: ignore[arg-type]
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="run a tool")]
    stored = engine.thread_store.read(events[-1]["thread_id"])

    assert [event["type"] for event in events].count("compaction.started") == 1
    assert [event["type"] for event in events].count("compaction.completed") == 1
    assert [event["type"] for event in events].index("compaction.started") < [
        event["type"] for event in events
    ].index("compaction.completed")
    compaction_event = next(event for event in events if event["type"] == "compaction.completed")
    assert compaction_event["text"] == "summary includes tool result"
    assert events[-1]["type"] == "turn.completed"
    assert events[-1]["final_text"] == "done after compaction"
    assert len(client.requests) == 3
    assert "context_compaction_request" in str(client.requests[1]["input"][-1])
    assert POST_TOOL_COMPACTION_BRIDGE in str(client.requests[1]["input"])
    assert client.requests[2]["previous_response_id"] is None
    assert "conversation_summary" in str(client.requests[2]["input"])
    assert "summary includes tool result" in str(client.requests[2]["input"])
    assert any(event["type"] == "item.assistant" for event in stored)
    assert any(event["type"] == "item.compaction" for event in stored)
    assert [event["type"] for event in stored].index("item.assistant") < [
        event["type"] for event in stored
    ].index("item.compaction")


@pytest.mark.asyncio
async def test_mid_turn_compaction_truncates_last_tool_output_for_compaction_request(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        context_window_tokens=20_000,
        compression=CompressionConfig(enabled=True, model_level="small", trigger_ratio=0.1, min_tokens=1),
    )
    client = CompletedOnlyStreamClient(
        [
            {
                "id": "resp_tool",
                "output_text": "",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_large",
                        "name": "run_python",
                        "arguments": json.dumps({"code": "print('large')"}),
                    }
                ],
            },
            {
                "id": "resp_compact",
                "output_text": "summary of truncated result",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "summary of truncated result"}],
                    }
                ],
            },
            {
                "id": "resp_final",
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
    huge_stdout = "START" + ("x" * 80_000) + "END"
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=LargeOutputRunner(huge_stdout),  # type: ignore[arg-type]
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="run a large tool")]
    compact_request = client.requests[1]["input"]
    compact_tool_output = next(item for item in compact_request if item.get("type") == "function_call_output")
    compact_payload = json.loads(compact_tool_output["output"])
    stored_tool_output = next(
        event["item"]
        for event in engine.thread_store.read(events[-1]["thread_id"])
        if event["type"] == "item.tool_output"
    )

    assert compact_payload["truncated_for_context_compaction"] is True
    assert "truncated for context compaction" in compact_payload["stdout"]
    assert len(compact_payload["stdout"]) < len(huge_stdout)
    assert "START" in compact_payload["stdout"]
    assert "END" in compact_payload["stdout"]
    assert json.loads(stored_tool_output["output"])["stdout"] == huge_stdout
    assert events[-1]["final_text"] == "done"


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
    assert "turn interrupted:" not in str(follow_client.requests[0]["input"])
    assert "continue" in str(follow_client.requests[0]["input"])


@pytest.mark.asyncio
async def test_responses_interrupted_partial_stream_adds_bridge_and_uses_full_replay(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root)
    partial_client = PartialStreamClient(ModelStreamEvent(type="text_delta", text="I will call a tool"))
    engine = AgentEngine(
        config=config,
        model_client=partial_client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    cancel_event = asyncio.Event()

    async def collect() -> list[dict[str, object]]:
        return [event async for event in engine.run_turn(user_text="start", cancel_event=cancel_event)]

    task = asyncio.create_task(collect())
    await partial_client.started.wait()
    await partial_client.delivered.wait()
    cancel_event.set()
    events = await asyncio.wait_for(task, timeout=5)
    thread_id = str(events[-1]["thread_id"])

    follow_client = FakeModelClient(
        [
            {
                "id": "resp_follow",
                "output_text": "continued",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "continued"}],
                    }
                ],
            }
        ]
    )
    engine.model_client = follow_client
    [event async for event in engine.run_turn(user_text="continue", thread_id=thread_id)]

    assert follow_client.requests[0]["previous_response_id"] is None
    request_input = follow_client.requests[0]["input"]
    assert any(
        item.get("role") == "assistant"
        and INTERRUPTED_STREAM_CONTEXT_BRIDGE in message_item_text(item)
        for item in request_input
    )
    assert request_input[-1]["role"] == "user"
    assert "continue" in str(request_input[-1])


@pytest.mark.asyncio
async def test_agent_locks_thread_while_turn_is_running(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root)
    state_dir = tmp_path / "state"
    blocking_client = BlockingModelClient()
    engine = AgentEngine(
        config=config,
        model_client=blocking_client,
        runner=PythonRunner(project_root=project_root, data_dir=state_dir, config=config.runner),
        thread_store=ThreadStore(state_dir),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread("Locked run")
    cancel_event = asyncio.Event()

    async def collect() -> list[dict[str, object]]:
        return [
            event
            async for event in engine.run_turn(
                user_text="hold lock",
                thread_id=thread_id,
                cancel_event=cancel_event,
            )
        ]

    task = asyncio.create_task(collect())
    await blocking_client.started.wait()
    other = ThreadStore(state_dir)

    assert engine.thread_store._read_lock_owner(thread_id)
    with pytest.raises(ThreadLockedError):
        other.append(
            thread_id,
            "item.user",
            turn_id="other",
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "other"}]},
        )

    cancel_event.set()
    events = await asyncio.wait_for(task, timeout=5)

    assert events[-1]["type"] == "turn.interrupted"
    assert not engine.thread_store._read_lock_owner(thread_id)


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
        request for request in client.requests if THREAD_TITLE_INSTRUCTION in str(request["instructions"])
    )
    assert title_request["level"] is None
    assert "用户第一条消息" in str(title_request["input"])


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



def test_clean_branch_slug_normalizes_model_output() -> None:
    assert clean_branch_slug('"Fix Login Redirect!"') == "fix-login-redirect"
    assert clean_branch_slug("feature/foo <> bad") == "feature-foo-bad"
    assert clean_branch_slug("!!!") is None
    assert clean_branch_slug("x" * 40) == "x" * 30


@pytest.mark.asyncio
async def test_agent_generates_branch_slug_with_configured_level_and_billing(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        branch_name_generation=BranchNameGenerationConfig(enabled=True, model_level="small"),
        pricing=PricingConfig(
            models={
                "default": ModelPricingConfig(input=1.0, output=2.0),
            }
        ),
    )
    client = RoutedModelClient(
        main={
            "id": "resp_branch",
            "output_text": "Fix Login Redirect!",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Fix Login Redirect!"}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread("Agent View dispatch")

    slug = await engine.generate_branch_slug(thread_id, "Fix the login redirect bug", level="medium")

    assert slug == "fix-login-redirect"
    request = client.requests[-1]
    assert request["level"] == "small"
    assert BRANCH_SLUG_INSTRUCTION in str(request["instructions"])
    assert BRANCH_NAME_GENERATION_PROMPT in str(request["input"])
    billing = engine.thread_store.latest_event(thread_id, "thread.billing_accumulated")
    assert billing is not None
    assert billing["source"] == "branch_name_generation"


@pytest.mark.asyncio
async def test_agent_branch_slug_generation_can_be_disabled(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        branch_name_generation=BranchNameGenerationConfig(enabled=False),
    )
    client = FakeModelClient([])
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread("Agent View dispatch")

    assert await engine.generate_branch_slug(thread_id, "anything", level="medium") is None
    assert client.requests == []


@pytest.mark.asyncio
async def test_agent_branch_slug_generation_times_out(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        branch_name_generation=BranchNameGenerationConfig(enabled=True, timeout_s=0.01),
    )
    client = HangingResponseClient([])
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread("Agent View dispatch")

    with pytest.raises(asyncio.TimeoutError):
        await engine.generate_branch_slug(thread_id, "Fix the login redirect bug", level="medium")
    assert client.requests


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
        request for request in client.requests if THREAD_TITLE_INSTRUCTION in str(request["instructions"])
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
        if THREAD_TITLE_INSTRUCTION in str(request.get("instructions") or "")
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
        request for request in client.requests if THREAD_TITLE_INSTRUCTION in str(request["instructions"])
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
    config = make_test_config(project_root)
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / ".uv-agent",
        config=RunnerConfig(
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
async def test_agent_yields_partial_tool_output_before_final_result(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root)
    runner = StreamingRunner()
    client = FakeModelClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "run_python",
                        "arguments": json.dumps({"code": "print('x')"}),
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
        runner=runner,  # type: ignore[arg-type]
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="run it")]
    event_types = [event["type"] for event in events]
    partial_payload = json.loads(next(event for event in events if event["type"] == "tool.partial")["output"]["output"])
    final_payload = json.loads(next(event for event in events if event["type"] == "tool.output")["output"]["output"])

    assert event_types.index("tool.started") < event_types.index("tool.partial") < event_types.index("tool.output")
    assert partial_payload["partial"] is True
    assert partial_payload["stdout"] == "partial output\n"
    assert final_payload["stdout"] == "partial output\nfinal output\n"
    model_tool_output = json.loads(client.requests[1]["input"][-1]["output"])
    assert "partial" not in model_tool_output
    assert "partial_reason" not in model_tool_output


@pytest.mark.asyncio
async def test_agent_sends_all_tool_outputs_with_previous_response_id(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root)
    runner = SimpleRunner()
    client = FakeModelClient(
        [
            {
                "id": "resp_multi_tool",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "run_python",
                        "arguments": json.dumps({"code": "print('one')"}),
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "run_python",
                        "arguments": json.dumps({"code": "print('two')"}),
                    },
                ],
            },
            {
                "id": "resp_done",
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
        runner=runner,  # type: ignore[arg-type]
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="run both")]

    assert events[-1]["type"] == "turn.completed"
    assert len(runner.requests) == 2
    assert client.requests[1]["previous_response_id"] == "resp_multi_tool"
    assert [item["call_id"] for item in client.requests[1]["input"]] == ["call_1", "call_2"]
    assert all(item["type"] == "function_call_output" for item in client.requests[1]["input"])


@pytest.mark.asyncio
async def test_agent_persists_model_stream_error(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root)
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
        ]
    )
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / ".uv-agent",
        config=RunnerConfig(
            default_timeout_s=30,
        ),
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=runner,
        thread_store=ThreadStore(tmp_path / ".uv-agent"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="run it")]
    thread_id = events[-1]["thread_id"]
    stored_events = engine.thread_store.read(thread_id)

    assert events[-1]["type"] == "turn.error"
    assert events[-1]["error_type"] == "RuntimeError"
    assert "FakeModelClient has no responses left" in events[-1]["message"]
    assert stored_events[-1]["type"] == "turn.error"
    assert stored_events[-1]["retryable"] is False
    assert not any(event["type"] == "turn.completed" for event in stored_events)


@pytest.mark.asyncio
async def test_cli_workflow_node_exits_nonzero_on_turn_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from uv_agent import cli

    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = AgentEngine(
        config=make_test_config(project_root),
        model_client=FailingStreamClient(RuntimeError("provider exploded")),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=RunnerConfig()),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    monkeypatch.setattr("uv_agent.app_factory.create_engine", lambda *_args, **_kwargs: engine)

    with pytest.raises(SystemExit) as exc_info:
        await cli._workflow_node(
            "try provider",
            None,
            None,
            stream=False,
            workflow_id="wf_test",
            node_id="wfn_test",
            project_state_dir=tmp_path / "state",
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "[RuntimeError] provider exploded" in captured.err


@pytest.mark.asyncio
async def test_agent_marks_provider_network_errors_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root)
    exc = openai_connection_error()
    sleeps: list[float] = []

    async def fake_sleep(delay_s: float) -> None:
        sleeps.append(delay_s)

    monkeypatch.setattr("uv_agent.agent.engine._sleep_stream_retry", fake_sleep)
    engine = AgentEngine(
        config=config,
        model_client=FailingStreamClient(exc),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="try provider")]
    stored_events = engine.thread_store.read(events[-1]["thread_id"])
    retry_events = [event for event in events if event["type"] == "model.stream_retry"]

    assert events[-1]["type"] == "turn.error"
    assert events[-1]["retryable"] is True
    assert stored_events[-1]["retryable"] is True
    assert len(retry_events) == 5
    assert len(sleeps) == 5


@pytest.mark.asyncio
async def test_agent_retries_empty_model_stream_then_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        stream_retry=StreamRetryConfig(max_retries=5, base=1.0, factor=2.0, max=30.0, jitter=0.0),
    )
    client = EmptyThenSuccessStreamClient(failures=2)
    sleeps: list[float] = []

    async def fake_sleep(delay_s: float) -> None:
        sleeps.append(delay_s)

    monkeypatch.setattr("uv_agent.agent.engine._sleep_stream_retry", fake_sleep)
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="try provider")]
    retry_events = [event for event in events if event["type"] == "model.stream_retry"]
    stored_events = engine.thread_store.read(events[-1]["thread_id"])

    assert events[-1]["type"] == "turn.completed"
    assert events[-1]["final_text"] == "done"
    assert len(client.requests) == 3
    assert [event["attempt"] for event in retry_events] == [1, 2]
    assert [event["delay_s"] for event in retry_events] == [1.0, 2.0]
    assert sleeps == [1.0, 2.0]
    assert [event["type"] for event in stored_events].count("turn.stream_retry") == 2
    assert [event["type"] for event in stored_events].count("item.model_response") == 1


@pytest.mark.asyncio
async def test_agent_retry_turn_retries_empty_model_stream_then_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        stream_retry=StreamRetryConfig(max_retries=5, base=1.0, factor=2.0, max=30.0, jitter=0.0),
    )
    thread_store = ThreadStore(tmp_path / "state")
    thread_id = thread_store.create_thread("Retry")
    thread_store.append(thread_id, "turn.started", turn_id="turn_old")
    thread_store.append(thread_id, "item.user", turn_id="turn_old", item=message_item("user", "try provider"))
    thread_store.append(
        thread_id,
        "turn.error",
        turn_id="turn_old",
        error_type="EmptyModelStreamError",
        message="empty stream",
        retryable=True,
    )
    client = EmptyThenSuccessStreamClient(failures=1)
    sleeps: list[float] = []

    async def fake_sleep(delay_s: float) -> None:
        sleeps.append(delay_s)

    monkeypatch.setattr("uv_agent.agent.engine._sleep_stream_retry", fake_sleep)
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=thread_store,
        project_root=project_root,
    )

    events = [event async for event in engine.retry_turn(thread_id=thread_id)]
    retry_events = [event for event in events if event["type"] == "model.stream_retry"]
    stored_events = engine.thread_store.read(thread_id)

    assert events[-1]["type"] == "turn.completed"
    assert events[-1]["final_text"] == "done"
    assert [event["attempt"] for event in retry_events] == [1]
    assert sleeps == [1.0]
    assert [event["type"] for event in stored_events].count("turn.stream_retry") == 1
    assert [event["type"] for event in stored_events].count("item.model_response") == 1


@pytest.mark.asyncio
async def test_agent_empty_model_stream_exhausts_auto_retries_then_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        stream_retry=StreamRetryConfig(max_retries=5, base=1.0, factor=2.0, max=30.0, jitter=0.0),
    )
    client = EmptyThenSuccessStreamClient(failures=6)
    sleeps: list[float] = []

    async def fake_sleep(delay_s: float) -> None:
        sleeps.append(delay_s)

    monkeypatch.setattr("uv_agent.agent.engine._sleep_stream_retry", fake_sleep)
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="try provider")]
    retry_events = [event for event in events if event["type"] == "model.stream_retry"]
    stored_events = engine.thread_store.read(events[-1]["thread_id"])

    assert events[-1]["type"] == "turn.error"
    assert events[-1]["error_type"] == "EmptyModelStreamError"
    assert events[-1]["retryable"] is True
    assert len(client.requests) == 6
    assert len(retry_events) == 5
    assert [event["delay_s"] for event in retry_events] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert stored_events[-1]["type"] == "turn.error"
    assert stored_events[-1]["retryable"] is True
    assert [event["type"] for event in stored_events].count("turn.stream_retry") == 5
    assert not any(event["type"] == "turn.completed" for event in stored_events)


@pytest.mark.asyncio
async def test_agent_stream_retry_sleep_can_be_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        stream_retry=StreamRetryConfig(max_retries=5, base=1.0, factor=2.0, max=30.0, jitter=0.0),
    )
    cancel_event = asyncio.Event()

    async def fake_sleep(delay_s: float) -> None:
        cancel_event.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("uv_agent.agent.engine._sleep_stream_retry", fake_sleep)
    engine = AgentEngine(
        config=config,
        model_client=EmptyThenSuccessStreamClient(failures=5),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [
        event
        async for event in engine.run_turn(
            user_text="try provider",
            cancel_event=cancel_event,
        )
    ]
    stored_events = engine.thread_store.read(events[-1]["thread_id"])

    assert [event["type"] for event in events].count("model.stream_retry") == 1
    assert events[-1]["type"] == "turn.interrupted"
    assert len(engine.model_client.requests) == 1
    assert stored_events[-1]["type"] == "turn.interrupted"


def test_openai_sdk_status_errors_format_and_retry_like_provider_errors() -> None:
    exc = openai_status_error(429, "rate limited", {"error": "rate limited"})

    error = format_error(exc)

    assert error.title == "Provider HTTP 429"
    assert "rate limited" in error.detail
    assert is_retryable_provider_error(exc) is True


def test_openai_sdk_status_error_subclasses_are_retryable_provider_errors() -> None:
    exc = openai_status_error(
        502,
        "bad gateway",
        {"error": "bad gateway"},
        error_cls=openai.InternalServerError,
    )

    error = format_error(exc)

    assert error.title == "Provider HTTP 502"
    assert "bad gateway" in error.detail
    assert is_retryable_provider_error(exc) is True


def test_openai_sdk_connection_errors_are_retryable_provider_errors() -> None:
    exc = openai_connection_error()

    error = format_error(exc)

    assert error.title == "Provider connection error"
    assert is_retryable_provider_error(exc) is True


@pytest.mark.asyncio
async def test_agent_retry_turn_retries_model_request_without_new_user_message(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root)
    first_client = FailingStreamClient(openai_connection_error())
    engine = AgentEngine(
        config=config,
        model_client=first_client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    failed_events = [event async for event in engine.run_turn(user_text="try provider")]
    thread_id = failed_events[-1]["thread_id"]
    failed_input = first_client.requests[0]["input"]

    retry_client = FakeModelClient(
        [
            {
                "id": "resp_retry",
                "output_text": "retried",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "retried"}],
                    }
                ],
            }
        ]
    )
    engine.model_client = retry_client
    retry_events = [event async for event in engine.retry_turn(thread_id=thread_id)]
    stored_events = engine.thread_store.read(thread_id)

    assert retry_events[-1]["type"] == "turn.completed"
    assert [event["type"] for event in stored_events].count("item.user") == 1
    assert retry_client.requests[0]["input"] == failed_input
    assert "try provider" in str(retry_client.requests[0]["input"])


@pytest.mark.asyncio
async def test_agent_retry_turn_resumes_pending_tool_call(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root)
    runner = SimpleRunner()
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient(
            [
                {
                    "id": "resp_final",
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
        ),
        runner=runner,  # type: ignore[arg-type]
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread()
    engine.thread_store.append(thread_id, "turn.started", turn_id="t1")
    user_item = message_item("user", "run tool")
    engine.thread_store.append(thread_id, "item.user", turn_id="t1", item=user_item)
    response_output = [
        {
            "type": "function_call",
            "call_id": "call_retry",
            "name": "run_python",
            "arguments": json.dumps({"code": "print('retry')"}),
        }
    ]
    engine.thread_store.append(
        thread_id,
        "item.model_response",
        turn_id="t1",
        model_api="responses",
        response_id="resp_tool",
        output=response_output,
        usage={},
        reasoning_text="",
    )
    engine.thread_store.append(
        thread_id,
        "turn.error",
        turn_id="t1",
        error_type="ConnectError",
        message="network down",
        retryable=True,
    )

    events = [event async for event in engine.retry_turn(thread_id=thread_id)]

    assert any(event["type"] == "tool.output" for event in events)
    assert events[-1]["type"] == "turn.completed"
    assert len(runner.requests) == 1
    assert len(engine.model_client.requests) == 1  # type: ignore[attr-defined]
    assert engine.model_client.requests[0]["previous_response_id"] == "resp_tool"  # type: ignore[attr-defined]
    request_input = engine.model_client.requests[0]["input"]  # type: ignore[attr-defined]
    assert request_input[0]["type"] == "function_call_output"
    assert request_input[0]["call_id"] == "call_retry"
    assert user_item not in request_input
    assert response_output[0] not in request_input


def test_reconstruct_input_closes_interrupted_pending_tool_call(tmp_path: Path) -> None:
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
    thread_id = engine.thread_store.create_thread()
    engine.thread_store.append(thread_id, "turn.started", turn_id="t1")
    user_item = message_item("user", "run tool")
    engine.thread_store.append(thread_id, "item.user", turn_id="t1", item=user_item)
    response_output = [
        {
            "type": "function_call",
            "call_id": "call_interrupted",
            "name": "run_python",
            "arguments": json.dumps({"code": "print(1)"}),
        }
    ]
    engine.thread_store.append(
        thread_id,
        "item.model_response",
        turn_id="t1",
        output=response_output,
    )
    engine.thread_store.append(thread_id, "turn.interrupted", turn_id="t1", reason="user_interrupt")

    reconstructed = engine._reconstruct_input(thread_id)

    assert reconstructed[:2] == [user_item, *response_output]
    assert [item.get("type") for item in reconstructed[-2:]] == [
        "function_call_output",
        "message",
    ]
    assert reconstructed[-1]["role"] == "assistant"
    assert INTERRUPTED_TOOL_CONTEXT_BRIDGE in message_item_text(reconstructed[-1])

    messages = chat_messages(reconstructed, instructions=None, model=config.model_for_level(None))
    assert [message["role"] for message in messages[-2:]] == ["tool", "assistant"]


def test_model_switch_warning_is_not_reconstructed_as_model_context(tmp_path: Path) -> None:
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
    thread_id = engine.thread_store.create_thread()
    engine.thread_store.append(thread_id, "thread.level_updated", level="medium", model="default")
    engine.thread_store.append(
        thread_id,
        "thread.model_switch_warning",
        from_level="medium",
        to_level="other",
        from_model="default",
        to_model="other",
        message="context conversion is best effort",
    )
    user_item = message_item("user", "continue")
    engine.thread_store.append(thread_id, "item.user", turn_id="t1", item=user_item)

    reconstructed = engine._reconstruct_input(thread_id)

    assert reconstructed == [user_item]
    assert "context conversion is best effort" not in str(reconstructed)


@pytest.mark.asyncio
async def test_tool_look_at_adds_assistant_bridge_before_image_context(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root, api="chat_completions")
    client = FakeModelClient(
        [
            {
                "id": "chat_1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "run_python",
                        "arguments": json.dumps({"code": "from uv_agent_runtime import look_at"}),
                    }
                ],
            },
            {
                "id": "chat_2",
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
    runner = LookAtRunner(tmp_path / "generated.png")
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=runner,  # type: ignore[arg-type]
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    events = [event async for event in engine.run_turn(user_text="make image")]
    follow_up_input = client.requests[1]["input"]

    assert [item["type"] for item in follow_up_input[-3:]] == [
        "function_call_output",
        "message",
        "message",
    ]
    assert follow_up_input[-2]["role"] == "assistant"
    assert TOOL_ATTACHMENT_CONTEXT_BRIDGE in message_item_text(follow_up_input[-2])
    assert follow_up_input[-1]["role"] == "user"
    assert any(content.get("type") == "input_image" for content in follow_up_input[-1]["content"])

    messages = chat_messages(follow_up_input, instructions=None, model=config.model_for_level(None))
    assert messages[-3]["role"] == "tool"
    assert messages[-2]["role"] == "assistant"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"][1]["type"] == "image_url"

    stored_events = engine.thread_store.read(events[-1]["thread_id"])
    tool_index = next(index for index, event in enumerate(stored_events) if event["type"] == "item.tool_output")
    image_index = next(index for index, event in enumerate(stored_events) if event["type"] == "item.image_attachment")
    assert image_index > tool_index
    assert stored_events[image_index]["source"] == "tool"
    reconstructed = engine._reconstruct_input(events[-1]["thread_id"])
    assert [item["type"] for item in reconstructed[-4:-1]] == [
        "function_call_output",
        "message",
        "message",
    ]


@pytest.mark.asyncio
async def test_responses_tool_look_at_resends_full_context_before_resuming_incremental(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root)
    client = FakeModelClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "run_python",
                        "arguments": json.dumps({"code": "from uv_agent_runtime import look_at"}),
                    }
                ],
            },
            {
                "id": "resp_2",
                "output_text": "seen",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "seen"}],
                    }
                ],
            },
            {
                "id": "resp_3",
                "output_text": "next",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "next"}],
                    }
                ],
            },
        ]
    )
    runner = LookAtRunner(tmp_path / "responses.png")
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=runner,  # type: ignore[arg-type]
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    first_events = [event async for event in engine.run_turn(user_text="make image")]
    [event async for event in engine.run_turn(user_text="follow up", thread_id=first_events[-1]["thread_id"])]

    assert client.requests[0]["previous_response_id"] is None
    assert client.requests[1]["previous_response_id"] is None
    assert any(item.get("call_id") == "call_1" for item in client.requests[1]["input"])
    assert any(
        content.get("type") == "input_image"
        for item in client.requests[1]["input"]
        for content in item.get("content", [])
    )
    assert client.requests[2]["previous_response_id"] == "resp_2"
    assert "make image" not in str(client.requests[2]["input"])
    assert "follow up" in str(client.requests[2]["input"])


def test_reconstructs_legacy_tool_image_after_tool_output(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    image = tmp_path / "legacy.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    config = make_test_config(project_root)
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread()
    attachment = engine.attachments.register_image(image, cwd=project_root, thread_id=thread_id)
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
        "item.image_attachment",
        turn_id="t1",
        attachment=attachment.to_event_payload(),
    )
    engine.thread_store.append(
        thread_id,
        "item.tool_output",
        turn_id="t1",
        item={"type": "function_call_output", "call_id": "call_1", "output": "{}"},
    )

    reconstructed = engine._reconstruct_input(thread_id)
    messages = chat_messages(reconstructed, instructions=None, model=config.model_for_level(None))

    assert [message["role"] for message in messages[-3:]] == ["tool", "assistant", "user"]
    assert messages[-1]["content"][1]["type"] == "image_url"


def test_anthropic_tool_image_context_keeps_tool_result_before_bridge(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    image = tmp_path / "anthropic.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    config = make_test_config(project_root, api="anthropic_messages")
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread()
    attachment = engine.attachments.register_image(image, cwd=project_root, thread_id=thread_id)
    items = [
        {
            "type": "function_call",
            "call_id": "toolu_1",
            "name": "run_python",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "toolu_1", "output": "{}"},
        *tool_attachment_context_items([attachment.to_event_payload()]),
    ]

    messages = anthropic_messages(items)

    assert [message["role"] for message in messages] == ["assistant", "user", "assistant", "user"]
    assert messages[1]["content"][0]["type"] == "tool_result"
    assert messages[2]["content"] == TOOL_ATTACHMENT_CONTEXT_BRIDGE
    assert messages[3]["content"][1]["type"] == "image"


@pytest.mark.asyncio
async def test_agent_displays_and_reconstructs_mixed_text_tool_response(tmp_path: Path) -> None:
    project_root = Path.cwd()
    config = make_test_config(project_root)
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / ".uv-agent",
        config=RunnerConfig(
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
    assert display_payload["events"][0]["kind"] == "progress"
    assert display_payload["events"][0]["message"] == "internal progress"
    assert display_payload["events"][0]["_uv_agent_run_id"] == display_payload["run_id"]
    assert display_payload["events"][0]["_uv_agent_event_id"].startswith("evt_")
    assert '"kind": "progress"' not in display_payload["stdout"]

    stored = engine.thread_store.read(events[-1]["thread_id"])
    runner_result = next(event["result"] for event in stored if event["type"] == "item.runner_result")
    assert runner_result["events"][0]["kind"] == "progress"
    assert runner_result["events"][0]["message"] == "internal progress"
    assert runner_result["events"][0]["_uv_agent_run_id"] == runner_result["run_id"]
    assert runner_result["events"][0]["_uv_agent_event_id"].startswith("evt_")


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


def test_agent_prompt_keeps_dynamic_capabilities_in_turn_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    project_root = tmp_path / "project"
    skill_dir = project_root / ".agents" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo\nUse this for demo work.\n", encoding="utf-8")
    mcp_path = project_root / ".agents" / "mcp.json"
    mcp_path.write_text(
        "{\"servers\":{\"demo\":{\"command\":\"python\",\"description\":\"Demo MCP\"}}}",
        encoding="utf-8",
    )
    mcp_probe = FakeMcpInstructionsProbe(
        {
            ("project", "demo", str(mcp_path)): McpInstructionsPreview(
                "Use demo tools carefully.",
                truncated=False,
            )
        }
    )
    config = load_config(project_root, [])
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / "state",
        config=RunnerConfig(
        ),
    )
    runner.scriptenv_dir.mkdir(parents=True)
    (runner.scriptenv_dir / "pyproject.toml").write_text(
        "[project]\nname = \"uv-agent-scriptenv\"\ndependencies = [\"uv-agent>=0.6.2\", \"requests>=2\"]\n",
        encoding="utf-8",
    )
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=runner,
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
        mcp_instructions_probe=mcp_probe,
    )

    prompt = engine.system_instructions()

    assert "run_python" in prompt
    assert "uv_agent_runtime" not in prompt
    assert str(project_root) not in prompt
    assert prompt.startswith("<uv_agent_system_prompt>")
    assert "</uv_agent_system_prompt>" in prompt
    assert "<response_style>" in prompt
    assert "简洁、友好、自然" in prompt
    assert "默认控制回答长度" in prompt
    assert "明确要求详细解释具体内容" in prompt
    assert "当项目规则或用户指令未另行要求时" in prompt
    assert "倾向于编写更充分的代码内文档" in prompt
    assert "改了什么、为什么改、如何验证" in prompt
    assert "Write comments generously" not in prompt
    assert "project-shared uv environment" not in prompt
    assert 'add_dependency("package-name")' not in prompt
    assert "Call add_dependency before importing the package" not in prompt
    assert "already been imported in the current Python process" not in prompt
    assert "run_python environment pyproject.toml" not in prompt
    assert "run_python accepts" not in prompt
    assert "script_args" not in prompt
    assert "thread's active cwd" not in prompt
    assert "PEP 723" not in prompt
    assert "uv pip" not in prompt
    assert "在成熟领域，临时使用可靠的第三方依赖往往比手写实现更安全、更高效" in prompt
    assert "用 unidiff 解析 diffs" in prompt
    assert "用 libcst 进行 Python 源码转换" in prompt
    assert "你唯一的外部动作工具是 run_python" in prompt
    assert "必须由 run_python 调用中的 Python 代码发起" in prompt
    assert "Do not assume shell, filesystem, browser, network, or MCP model tools exist outside Python" not in prompt
    assert "prefer uv_agent_runtime helpers when they fit" not in prompt
    assert "Consult the appended runtime helper guidance for operation-specific details" not in prompt
    assert "raw subprocess" not in prompt
    assert "Use Python standard library modules such as pathlib, os, and json" not in prompt
    assert "use Python standard library modules such as pathlib, os, and json for in-script glue" not in prompt
    assert "especially file and edit helpers for repository-visible text work" not in prompt
    assert "metadata such as newline style, BOM, final newline" not in prompt
    assert "ordinary in-script glue" not in prompt
    assert "prefer runtime file and edit helpers" not in prompt
    assert "When running independent work concurrently inside run_python" not in prompt
    assert "在 Python 中可使用 asyncio、concurrent.futures 和 threading" in prompt
    assert "asyncio, concurrent.futures, threading, and subprocess" not in prompt
    assert "按确定顺序收集结果" in prompt
    assert "Do not guess helper signatures" not in prompt
    assert "系统不会替你截断过大的输出" in prompt
    assert "先过滤、限制或摘要" in prompt
    assert "<run_python_workflow>" in prompt
    assert "搜索、读取、计算、编辑、验证和条件回退都在同一个脚本内用 Python 原生控制流编排" in prompt
    assert "应尽量在同一个脚本内完成" not in prompt
    assert "在脚本内使用常规 Python 语法" in prompt
    assert "借助 Python 强大的特性、runtime helpers 以及其他能力" in prompt
    assert "同时处理多文件、多步骤、可预见的分支或失败" in prompt
    assert "在探索阶段，在单脚本中一次性收集足够信息" in prompt
    assert "Call enter_dir proactively whenever the task clearly belongs" not in prompt
    assert "including paths discovered during execution" not in prompt
    assert "<capability_use>" in prompt
    assert "如果某项能力能减少步骤、节省时间或降低风险，就优先使用" in prompt
    assert "Actively use available capabilities" not in prompt
    assert "Actively use available external capabilities" not in prompt
    assert "runtime helpers、declared skills、declared MCP servers" in prompt
    assert "subprocesses through Python" not in prompt
    assert "Prefer existing helpers and declared external capabilities" not in prompt
    assert "use simple Python for glue code or very small work" not in prompt
    assert "only when it materially helps" not in prompt
    assert "使用 workflow 相关的 runtime helper 函数" in prompt and "独立或长时间运行的模型任务" in prompt
    assert "并发运行相互独立的任务" in prompt
    assert "workflow nodes 或 run_python 内的独立 helper operations" in prompt
    assert "对同一文件的写入保持顺序执行" in prompt
    assert "Plan each run_python call as a complete work unit" not in prompt
    assert "Use one call for a complete work unit" not in prompt
    assert "batch related commands, searches, reads, edits" not in prompt
    assert "shell-style fragments" not in prompt
    assert "shell-style pseudo-code" not in prompt
    assert "not as a wrapper around one helper call" not in prompt
    assert "Treat run_python as a free-form multi-step tool" not in prompt
    assert "do them in one script and return one consolidated result" not in prompt
    assert "genuinely need a user check-in" not in prompt
    assert "Occam's razor" not in prompt
    assert "capability explanations layered" not in prompt
    assert "item.context_update is an internal persistence event" not in prompt
    assert "After compaction, current context updates are re-sent" not in prompt
    assert "Interrupted turns may appear in context" not in prompt
    assert "<runtime_environment>" not in prompt
    assert "<model_levels>" not in prompt
    assert "</runtime_helpers>" not in prompt
    assert "custom patch envelope" not in prompt
    assert "connect_named(\"files\")" not in prompt
    assert "saved_scripts" not in prompt
    assert "Directory rules from AGENTS files are loaded automatically" not in prompt

    turn_context = engine._turn_context_text()

    assert "<runtime_environment>" in turn_context
    assert "<host>" in turn_context
    assert "<user_language>" in turn_context
    assert str(project_root) in turn_context
    assert "<run_python_environment>" in turn_context
    assert "这是 run_python 使用的 uv project 环境" in turn_context
    assert str(runner.scriptenv_dir) in turn_context
    assert str(runner.scriptenv_dir / "pyproject.toml") in turn_context
    assert "uv-agent&gt;=0.6.2" not in turn_context
    assert "<dependency>requests&gt;=2</dependency>" in turn_context
    assert "<model_levels>" in turn_context
    assert "<default>medium</default>" in turn_context
    assert "<level>small</level>" in turn_context
    assert "<level>medium</level>" in turn_context
    assert "</runtime_helpers>" in turn_context
    assert "它们来自 uv_agent_runtime，不是预加载 globals" in turn_context
    assert "read_file" in turn_context
    assert "write_file" in turn_context
    assert "edit_lines" in turn_context
    assert "path_info" not in turn_context
    assert "read_text_lossless" not in turn_context
    assert "write_text_lossless" not in turn_context
    assert "compare_text" not in turn_context
    assert "normalize_text" not in turn_context
    assert "replace_text" in turn_context
    assert "replace_exact" not in turn_context
    assert "make_unified_diff" not in turn_context
    assert "apply_patch_any" not in turn_context
    assert "convert_patch" not in turn_context
    assert "workspace_transaction" not in turn_context
    assert "snapshot_files" not in turn_context
    assert "restore_snapshot" not in turn_context
    assert "goal_paths" not in turn_context
    assert "supported_symbol_languages" not in turn_context
    assert "clear_codequery_cache" not in turn_context
    assert "run_process_text" in turn_context
    assert "add_dependency" in turn_context
    assert "向共享 run_python uv project 添加 direct packages" in turn_context
    assert "在当前脚本 import 该 package 前调用" in turn_context
    assert "该进程中已经 import 的 package" in turn_context
    assert "run_python_env_dir" in turn_context
    assert "context=None" in turn_context
    assert "max_total=None" in turn_context
    assert '<helper name="threads">' in turn_context
    assert "thread_view" in turn_context
    assert "thread_detail" in turn_context
    assert "run_digest" not in turn_context
    assert "<usage_pattern>" in turn_context
    assert "helpers 是工作单元脚本里使用的 Python 函数" in turn_context
    assert "不是独立的工具模式" in turn_context
    assert "不要仅因为下一步要用另一个 helper" in turn_context
    assert "对方向已经明确的后续步骤" in turn_context
    assert "用 Python 编排" in turn_context
    assert "根据 helper 结果分支" in turn_context
    assert "用 Python libraries 解析结构化输出" in turn_context
    assert "收集一份摘要" in turn_context
    assert "把 shell 习惯改成 Python 写法" in turn_context
    assert "用 read_file 代替 cat" in turn_context
    assert "用 search_text/find_files 代替临时 grep/find" in turn_context
    assert "用 run_process_text([...]) 代替 raw subprocess" in turn_context
    assert "skill 文件用 read_file 读取 SKILL.md" in turn_context
    assert "在同一脚本中处理可预见的后续解析或回退逻辑" in turn_context
    assert '<example name="round-1-find">' in turn_context
    assert "查找并理解" in turn_context
    assert "并行搜索多个 pattern、一次读取多个相关文件" in turn_context
    assert "在决定修改前收集上下文" in turn_context
    assert "参考示例" in turn_context
    assert "编辑并验证" in turn_context
    assert "在目标、位置和修改方式已经明确后" in turn_context
    assert "先快速搜索确认目标，再一起应用变更并验证" in turn_context
    assert "不要把已知编辑推迟到下一轮" in turn_context
    assert "from uv_agent_runtime import search_text, find_files, read_file" in turn_context
    assert 'search_text("def handle_login"' in turn_context
    assert "未定义 handle_login" in turn_context
    assert 'search_text("handle_login("' in turn_context
    assert "call_hits" in turn_context
    assert "find_files(globs=" in turn_context
    assert "from uv_agent_runtime import search_text, replace_text, edit_lines, run_process_text" in turn_context
    assert "replace_text(" in turn_context
    assert 'redirect("/old-dashboard")' in turn_context
    assert 'redirect(url_for("dashboard"))' in turn_context
    assert "MAX_LOGIN_ATTEMPTS" in turn_context
    assert "未找到目标" in turn_context
    assert "anchor 不匹配" in turn_context
    assert "edit_lines(" in turn_context
    assert '<example name="anti-pattern-one-helper-per-call">' in turn_context
    assert "不要把一个清晰工作单元拆成多次 run_python" in turn_context
    assert "每次只调用一个 helper" in turn_context
    assert "偷懒式串行" in turn_context
    assert "浪费往返、丢失前一次的返回值" in turn_context
    assert "在一次 run_python 脚本中导入并组合" in turn_context
    assert "已应用变更" in turn_context
    assert '"uv", "run", "pytest"' in turn_context
    assert "def section" not in turn_context
    assert "def bounded" not in turn_context
    assert "def command" not in turn_context
    assert "json.loads" not in turn_context
    assert "<helper_selection>" in turn_context
    assert "列出的 helpers 是普通 Python 函数" in turn_context
    assert "与标准库代码和控制流组合使用" in turn_context
    assert "pathlib、os、json" in turn_context and "做衔接逻辑" in turn_context
    assert "适合时优先使用 helpers" in turn_context
    assert "newline style、BOM、final newline" in turn_context
    assert "按任务选择：" in turn_context
    assert "discovery=find_files/search_text/find_symbols/query_code" in turn_context
    assert "search_text 默认 regex" in turn_context
    assert "精确代码字符串用 literal=True" in turn_context
    assert "rg type aliases 用 file_types" in turn_context
    assert "edit=用 replace_text 替换唯一小段文本，用 edit_lines 处理 anchored ranges/inserts" in turn_context
    assert "完整文件或生成的内容用 write_file" in turn_context
    assert "process=run_process_text for ordinary external commands" not in turn_context
    assert "普通外部命令（包括" in turn_context
    assert "包括 skills 或 docs 中展示的 shell commands" in turn_context
    assert "优先用 run_process_text 而不是 raw subprocess" in turn_context
    assert "自定义进程控制" in turn_context
    assert "thread history=list_thread_digests/thread_view/thread_detail" in turn_context
    assert "数据量较大时，优先提取字段、行范围、head/tail 或生成摘要" in turn_context
    assert "不要猜测 helper signatures" in turn_context
    assert "Search 和 symbol helpers 返回给 file helpers 使用的是绝对路径" in turn_context
    assert "start=end+1 插入" in turn_context
    assert "pattern 默认是 regex，精确字符串传 literal=True" in turn_context
    assert "Prefer the smallest helper that directly matches the task" not in turn_context
    assert "uv-agent patch envelope shown below" not in turn_context
    assert turn_context.count("<description>") >= 15
    assert "<![CDATA[" not in turn_context
    assert '<helper name="replace_text">' in turn_context
    assert '<helper name="mcp">' in turn_context
    assert '<helper name="stdlib">' not in turn_context
    assert '<helper name="inspect_signatures">' not in turn_context
    assert "These helpers do not switch the active TUI thread" not in turn_context
    assert "*** Begin Patch" not in turn_context
    assert "*** Update File: src/app.py" not in turn_context
    assert 'find_files("src", globs=["*.py", "!**/migrations/**"], max_total=30)' not in turn_context
    assert "before.text.replace" not in turn_context
    assert 'connect_named("server-name")' not in turn_context
    assert "client.initialize()" in turn_context
    assert "检查返回的 instructions" in turn_context
    assert '<helper name="workflow"' in turn_context
    assert "构建持久任务图" in turn_context
    assert "replaces " + "ask" not in turn_context
    assert 'level="small"' not in prompt
    assert "pathlib" in turn_context
    assert "这些 mentions 只是纯文本提示" in prompt
    assert "在 run_python 中使用对应 runtime helper 读取并检查它" in prompt
    assert "read_text, write_text" not in prompt
    assert "list_files" not in prompt
    assert "run_command/check_command" not in prompt
    assert "emit_event" not in prompt
    assert "enter_dir" in turn_context
    assert "demo (project)" not in prompt

    assert '<skill name="demo" scope="project"' in turn_context
    assert "available_mcp_servers" in turn_context
    assert '<mcp_server name="demo" scope="project"' in turn_context
    assert "<description>Demo MCP</description>" in turn_context
    assert '<instructions truncated="false">Use demo tools carefully.</instructions>' in turn_context
    assert mcp_probe.started is True
    assert "遇到适合任务的 skill 时" in turn_context
    assert "遇到适合任务的 MCP server 时" in turn_context


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
        ),
    )
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
        mcp_instructions_probe=FakeMcpInstructionsProbe(),
    )

    prompt = engine.system_instructions()
    turn_context = engine._turn_context_text()

    assert "<default>deep</default>" in turn_context
    assert "<level>fast</level>" in turn_context
    assert "<level>deep</level>" in turn_context
    assert 'level="small"' not in prompt
    assert 'model_level="large"' not in prompt
    assert "small/medium/large" not in prompt


def test_usage_token_count_supports_provider_shapes() -> None:
    # OpenAI Responses API: input/output/total all present, total is authoritative.
    assert usage_token_count({"total_tokens": 42, "input_tokens": 30, "output_tokens": 12}) == 42
    # OpenAI Chat Completions: prompt/completion/total all present.
    assert (
        usage_token_count({"total_tokens": 100, "prompt_tokens": 70, "completion_tokens": 30}) == 100
    )
    # Anthropic: no total_tokens; sum non-cache input + output + cache_creation + cache_read.
    assert (
        usage_token_count(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
            }
        )
        == 200
    )
    # Anthropic without cache fields.
    assert usage_token_count({"input_tokens": 10, "output_tokens": 3}) == 13
    # Some providers expose total_token_count instead of total_tokens.
    assert usage_token_count({"total_token_count": 7}) == 7
    # Null/missing direct keys fall through to summing pairs.
    assert usage_token_count({"total_tokens": None, "prompt_tokens": 9, "completion_tokens": 2}) == 11
    assert usage_token_count({}) is None


def test_billing_charge_uses_uncached_cached_and_output_tokens(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        pricing=PricingConfig(
            currency="CNY",
            unit="1M_tokens",
            models={
                "default": ModelPricingConfig(input=2.0, output=8.0, cached_input=0.5),
            },
        ),
    )
    model = config.model_for_level("medium")

    charge = billing_charge_for_usage(
        config,
        model,
        {
            "input_tokens": 1_000,
            "input_tokens_details": {"cached_tokens": 200},
            "output_tokens": 300,
        },
        level="medium",
    )

    assert charge is not None
    assert charge.input_tokens == 800
    assert charge.cached_input_tokens == 200
    assert charge.output_tokens == 300
    assert format_billing_total(charge.amount, charge.currency, decimals=6) == "¥0.004100"


def test_billing_token_breakdown_supports_anthropic_cache_tokens() -> None:
    breakdown = billing_token_breakdown(
        {
            "input_tokens": 100,
            "cache_creation_input_tokens": 40,
            "cache_read_input_tokens": 60,
            "output_tokens": 20,
        }
    )

    assert breakdown.input_tokens == 140
    assert breakdown.cached_input_tokens == 60
    assert breakdown.output_tokens == 20


def test_billing_token_breakdown_extracts_reasoning_tokens() -> None:
    """Reasoning tokens from completion_tokens_details/output_tokens_details."""
    breakdown = billing_token_breakdown(
        {
            "input_tokens": 100,
            "output_tokens": 80,
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
    )
    assert breakdown.output_tokens == 80
    assert breakdown.reasoning_tokens == 30

    # Responses API shape: output_tokens_details.reasoning_tokens
    breakdown2 = billing_token_breakdown(
        {
            "input_tokens": 50,
            "output_tokens": 60,
            "output_tokens_details": {"reasoning_tokens": 40},
        }
    )
    assert breakdown2.output_tokens == 60
    assert breakdown2.reasoning_tokens == 40

    # Fallback to top-level "reasoning_tokens" (some compatible providers)
    breakdown3 = billing_token_breakdown(
        {
            "input_tokens": 50,
            "output_tokens": 60,
            "reasoning_tokens": 20,
        }
    )
    assert breakdown3.output_tokens == 60
    assert breakdown3.reasoning_tokens == 20

    # No reasoning tokens present
    breakdown4 = billing_token_breakdown(
        {
            "input_tokens": 50,
            "output_tokens": 60,
        }
    )
    assert breakdown4.output_tokens == 60
    assert breakdown4.reasoning_tokens == 0

@pytest.mark.asyncio
async def test_agent_accumulates_billing_for_model_response(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        pricing=PricingConfig(
            currency="USD",
            unit="1M_tokens",
            models={
                "default": ModelPricingConfig(input=1.0, output=2.0, cached_input=0.25),
            },
        ),
    )
    client = CompletedOnlyStreamClient(
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
                "usage": {
                    "input_tokens": 1_000,
                    "input_tokens_details": {"cached_tokens": 200},
                    "output_tokens": 500,
                },
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

    events = [event async for event in engine.run_turn(user_text="hi")]
    thread_id = events[-1]["thread_id"]
    digest = engine.thread_store.thread_digest(thread_id)

    assert digest["billing_currency"] == "USD"
    assert digest["billing_total"] == "0.00185"
    assert any(event["type"] == "thread.billing_accumulated" for event in engine.thread_store.read(thread_id))


def test_subagent_billing_rolls_into_parent_thread(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(
        project_root,
        pricing=PricingConfig(
            currency="USD",
            unit="1M_tokens",
            models={"default": ModelPricingConfig(input=1.0, output=2.0, cached_input=0.25)},
        ),
    )
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    parent_id = engine.thread_store.create_thread()
    subthread_id = engine.thread_store.create_thread(
        "Subagent",
        kind="subagent",
        parent_thread_id=parent_id,
        parent_turn_id="turn_parent",
    )
    engine.thread_store.append(
        subthread_id,
        "thread.billing_accumulated",
        amount="0.00042",
        currency="USD",
        source="model_response",
    )

    _rules, visible = engine._process_runner_events(
        [{"kind": "subagent.completed", "thread_id": subthread_id}],
        thread_id=parent_id,
        turn_id="turn_parent",
    )
    digest = engine.thread_store.thread_digest(parent_id)

    assert visible == [{"kind": "subagent.completed", "thread_id": subthread_id}]
    assert digest["billing_total"] == "0.00042"
    event = engine.thread_store.latest_event(parent_id, "thread.billing_accumulated")
    assert event is not None
    assert event["source"] == "subagent"
    assert event["subthread_id"] == subthread_id


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
async def test_agent_sends_project_rule_index_without_rule_contents(tmp_path: Path) -> None:
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
    assert '<workspace_rules path=".">' in request_text
    assert '<rule file="AGENTS.md">' in request_text
    assert "Use the local rule." in request_text
    assert "<workspace_rule_index>" in request_text
    assert "AGENTS.md" in request_text
    stored = engine.thread_store.read(events[-1]["thread_id"])
    assert any(event["type"] == "item.rules_loaded" and event.get("source") == "project" for event in stored)
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
    assert '<workspace_rules path=".">' in str(client.requests[0]["input"])
    assert "Never persist this rule." in str(client.requests[0]["input"])
    assert "<workspace_rule_index>" in str(client.requests[0]["input"])
    assert "AGENTS.md" in str(client.requests[0]["input"])
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

    assert '<workspace_rules path=".">' in requests_text[0]
    assert "AGENTS.md" in requests_text[0]
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
async def test_project_rules_persist_across_engine_restart_without_one_turn_reload(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    rules = project_root / "AGENTS.md"
    rules.write_text("Persistent rule v1.", encoding="utf-8")
    config = make_test_config(project_root, api="chat_completions")
    store = ThreadStore(tmp_path / "state")
    first_client = FakeModelClient(
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
            }
        ]
    )
    first_engine = AgentEngine(
        config=config,
        model_client=first_client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=store,
        project_root=project_root,
    )

    thread_id = [event async for event in first_engine.run_turn(user_text="one")][-1]["thread_id"]
    rules.write_text("Persistent rule v2.", encoding="utf-8")
    second_client = FakeModelClient(
        [
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
            }
        ]
    )
    second_engine = AgentEngine(
        config=config,
        model_client=second_client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=store,
        project_root=project_root,
    )

    [event async for event in second_engine.run_turn(user_text="two", thread_id=thread_id)]

    request_text = str(second_client.requests[0]["input"])
    assert second_client.requests[0]["previous_response_id"] is None
    assert "Persistent rule v1." in request_text
    assert request_text.count("Persistent rule v1.") == 1
    assert "Persistent rule v2." not in request_text


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
    assert '<workspace_rules path=".">' in str(second)
    assert "After compaction rule." in str(second)
    assert "<workspace_rule_index>" in str(second)
    assert "AGENTS.md" in str(second)


@pytest.mark.asyncio
async def test_compaction_epoch_reloads_project_rules_and_active_cwd_rules(tmp_path: Path) -> None:
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
    assert '<workspace_rules path=".">' in text
    assert '<workspace_rules path="src">' in text
    assert '<rule file="AGENTS.md">' in text
    assert "AGENTS.md" in text
    assert "src/AGENTS.md" in text
    assert "pkg/AGENTS.md" in text
    assert "Root rule." in text
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
    assert "<model_levels>" not in client.requests[1]["instructions"]


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

    assert "<default>medium</default>" in str(client.requests[0]["input"])
    assert "<default>small</default>" in str(client.requests[1]["input"])
    stored = engine.thread_store.read(thread_id)
    assert sum(1 for event in stored if event["type"] == "item.system_instructions") == 2


def test_dynamic_runtime_context_reappears_after_compaction_epoch(tmp_path: Path) -> None:
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

    first = engine._runtime_context_items(thread_id)
    engine.thread_store.append(thread_id, "item.compaction", turn_id="t1", text="summary", usage={})
    second = engine._runtime_context_items(thread_id)

    assert '<skill name="demo" scope="project"' in str(first)
    assert '<skill name="demo" scope="project"' in str(second)
    assert "<runtime_environment>" in str(second)
    assert "<model_levels>" in str(second)
    assert "<runtime_helpers>" in str(second)


def test_runtime_context_is_not_repeated_after_compaction_epoch_update(tmp_path: Path) -> None:
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

    engine._runtime_context_items(thread_id)
    engine.thread_store.append(thread_id, "item.compaction", turn_id="t1", text="summary", usage={})
    after_compaction = engine._runtime_context_items(thread_id)
    repeated = engine._runtime_context_items(thread_id)

    assert '<skill name="demo" scope="project"' in str(after_compaction)
    assert repeated == []


def test_runtime_context_skill_change_sends_incremental_section_only(tmp_path: Path) -> None:
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
    thread_id = engine.thread_store.create_thread()

    first = engine._runtime_context_items(thread_id)
    skill_dir = project_root / ".agents" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo\nUse this for demo work.\n", encoding="utf-8")
    second = engine._runtime_context_items(thread_id)

    assert "<runtime_environment>" in str(first)
    assert "<model_levels>" in str(first)
    assert "<runtime_helpers>" in str(first)
    text = str(second)
    assert "changed:" not in text
    assert "fingerprint:" not in text
    assert "<available_skills>" not in text
    assert '<skill name="demo" scope="project"' in text
    assert "<runtime_environment>" not in text
    assert "<model_levels>" not in text
    assert "<runtime_helpers>" not in text


def test_runtime_context_mcp_removal_sends_removal_only(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    agents_dir = project_root / ".agents"
    agents_dir.mkdir(parents=True)
    mcp_path = agents_dir / "mcp.json"
    mcp_path.write_text(
        "{\"servers\":{\"demo\":{\"command\":\"python\",\"description\":\"Demo MCP\"}}}",
        encoding="utf-8",
    )
    config = make_test_config(project_root)
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
        mcp_instructions_probe=FakeMcpInstructionsProbe(),
    )
    thread_id = engine.thread_store.create_thread()

    first = engine._runtime_context_items(thread_id)
    mcp_path.unlink()
    second = engine._runtime_context_items(thread_id)

    assert "<available_mcp_servers>" in str(first)
    text = str(second)
    assert "removed:" not in text
    assert "fingerprint:" not in text
    assert "<context_update_removed id=\"runtime_context\">" in text
    assert '<removed_mcp_server name="demo" scope="project"' in text
    assert "<available_mcp_servers>" not in text
    assert '<mcp_server name="demo"' not in text
    assert "<runtime_environment>" not in text
    assert "<model_levels>" not in text
    assert "<runtime_helpers>" not in text


def test_runtime_context_mcp_instruction_change_sends_single_server_only(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    agents_dir = project_root / ".agents"
    agents_dir.mkdir(parents=True)
    mcp_path = agents_dir / "mcp.json"
    mcp_path.write_text(
        json.dumps(
            {
                "servers": {
                    "first": {"command": "python", "description": "First MCP"},
                    "second": {"command": "python", "description": "Second MCP"},
                }
            }
        ),
        encoding="utf-8",
    )
    config = make_test_config(project_root)
    probe = FakeMcpInstructionsProbe()
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
        mcp_instructions_probe=probe,
    )
    thread_id = engine.thread_store.create_thread()

    first = engine._runtime_context_items(thread_id)
    probe.instructions[("project", "second", str(mcp_path))] = McpInstructionsPreview(
        "Use the second MCP carefully.",
        truncated=False,
    )
    second = engine._runtime_context_items(thread_id)

    assert '<mcp_server name="first"' in str(first)
    assert '<mcp_server name="second"' in str(first)
    text = str(second)
    assert "changed:" not in text
    assert "fingerprint:" not in text
    assert '<mcp_server name="second" scope="project"' in text
    assert '<instructions truncated="false">Use the second MCP carefully.</instructions>' in text
    assert '<mcp_server name="first"' not in text
    assert "<available_mcp_servers>" not in text


def test_runtime_context_restart_preserves_mcp_instructions_until_probe_refresh(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    agents_dir = project_root / ".agents"
    agents_dir.mkdir(parents=True)
    mcp_path = agents_dir / "mcp.json"
    mcp_path.write_text(
        json.dumps(
            {"servers": {"demo": {"command": "python", "description": "Demo MCP"}}}
        ),
        encoding="utf-8",
    )
    config = make_test_config(project_root)
    store = ThreadStore(tmp_path / "state")
    probe = FakeMcpInstructionsProbe(
        {
            ("project", "demo", str(mcp_path)): McpInstructionsPreview(
                "Use persisted MCP instructions.",
                truncated=False,
            )
        }
    )
    first_engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=store,
        project_root=project_root,
        mcp_instructions_probe=probe,
    )
    thread_id = store.create_thread()

    first = first_engine._runtime_context_items(thread_id)
    restarted_engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
        mcp_instructions_probe=FakeMcpInstructionsProbe(),
    )
    second = restarted_engine._runtime_context_items(thread_id)

    assert '<instructions truncated="false">Use persisted MCP instructions.</instructions>' in str(first)
    assert second == []


def test_runtime_context_update_has_stable_order_and_prefix(tmp_path: Path) -> None:
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

    update = engine._turn_context_update(None)

    assert update is not None
    text = update["text"]
    assert text.startswith('<context_update id="runtime_context" status="current">\n')
    assert text.index("<runtime_environment>") < text.index("<model_levels>")
    assert text.index("<model_levels>") < text.index("<runtime_helpers>")
    assert text.index('name="enter_dir"') < text.index('name="workflow"')
    assert text.index('name="workflow"') < text.index('name="add_dependency"')
    assert text.index('name="add_dependency"') < text.index('name="look_at"')
    assert text.index('name="look_at"') < text.index('name="read_file"')
    assert text.index('name="read_file"') < text.index('name="write_file"')
    assert text.index('name="write_file"') < text.index('name="edit_lines"')
    assert text.index('name="edit_lines"') < text.index('name="replace_text"')
    assert text.index('name="replace_text"') < text.index('name="run_process_text"')


def test_plugin_runtime_helpers_context_clarifies_helper_name(tmp_path: Path) -> None:
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
    engine.runtime_helpers.register(
        plugin="demo-plugin",
        name="demo_helper",
        fn=lambda: None,
        doc="Demo helper.",
    )

    update = engine._turn_context_update(None)

    assert update is not None
    text = update["text"]
    assert text.index("<runtime_helpers>") < text.index("<plugin_runtime_helpers>")
    assert (
        "使用 helper 的 name 属性作为 Python 中的 import/callable 名称；"
        "plugin 属性只标识提供方 plugin。"
    ) in text
    assert '<helper name="demo_helper" plugin="demo-plugin">Demo helper.</helper>' in text


@pytest.mark.asyncio
async def test_run_turn_waits_for_plugin_start_before_context_update(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root)
    client = CompletedOnlyStreamClient(
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
    plugins = DelayedPluginManager(engine)
    engine.plugins = plugins  # type: ignore[assignment]

    async def collect_events() -> list[dict[str, Any]]:
        return [event async for event in engine.run_turn(user_text="hello")]

    turn_task = asyncio.create_task(collect_events())
    await asyncio.wait_for(plugins.started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not client.requests

    plugins.release.set()
    events = await asyncio.wait_for(turn_task, timeout=2)

    assert plugins.start_count == 1
    assert events[-1]["type"] == "turn.completed"
    request_text = "\n".join(message_item_text(item) for item in client.requests[0]["input"])
    assert '<helper name="delayed_helper" plugin="delayed-plugin">Delayed helper.</helper>' in request_text



def test_workflow_context_xml_has_compact_prompt_format() -> None:
    from uv_agent.workflow_context import render_workflow_context

    context = render_workflow_context()
    lines = context.splitlines()

    assert lines[0] == '<workflow_context scope="main_agent" status="current">'
    assert lines[-1] == "</workflow_context>"
    assert "  <" not in context
    assert "\n\n<" not in context

    import re

    examples = re.findall(r"<example name=\"([^\"]+)\">", context)
    code_blocks = re.findall(r"<code>\n(.*?)\n</code>", context, flags=re.S)
    assert examples == [
        "create_investigation_graph",
        "inspect_first_checkpoint_and_extend_graph",
        "inspect_review_checkpoint_and_finalize",
    ]
    assert len(code_blocks) == 3
    for code in code_blocks:
        compile(code, "<workflow_context_example>", "exec")


def test_workflow_context_emits_for_main_thread_once_per_epoch(tmp_path: Path) -> None:
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
    thread_id = engine.thread_store.create_thread()

    first = "\n".join(message_item_text(item) for item in engine._pre_user_context_items(thread_id))
    repeated = "\n".join(message_item_text(item) for item in engine._pre_user_context_items(thread_id))

    assert '<workflow_context scope="main_agent" status="current">' in first
    assert 'example name="create_investigation_graph"' in first
    assert 'example name="inspect_first_checkpoint_and_extend_graph"' in first
    assert 'example name="inspect_review_checkpoint_and_finalize"' in first
    assert "## 目标和任务" in first
    assert "## 要求和说明" in first
    assert "wf.continue_checkpoint" in first
    assert "Workflow " + "replaces " + "ask" not in first
    assert "verify.final" in first
    assert '<workflow_context scope="main_agent" status="current">' not in repeated

    engine.thread_store.append(thread_id, "item.compaction", turn_id="t1", text="summary", usage={})
    after_compaction = "\n".join(message_item_text(item) for item in engine._pre_user_context_items(thread_id))
    assert '<workflow_context scope="main_agent" status="current">' in after_compaction


def test_workflow_context_is_not_sent_to_workflow_node_threads(tmp_path: Path) -> None:
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
    parent = engine.thread_store.create_thread()
    node_thread = engine.thread_store.create_thread("Node", kind="workflow_node", parent_thread_id=parent)

    text = "\n".join(message_item_text(item) for item in engine._pre_user_context_items(node_thread))

    assert "<workflow_context" not in text


def test_workflow_context_is_pre_user_context_and_not_retained(tmp_path: Path) -> None:
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
    thread_id = engine.thread_store.create_thread()
    workflow_item = next(
        item for item in engine._pre_user_context_items(thread_id) if "<workflow_context" in message_item_text(item)
    )

    assert engine._is_pre_user_context_item(workflow_item)
    assert retain_item_after_compaction(workflow_item) is False


def test_compaction_summary_appends_active_workflows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from uv_agent_runtime import workflow

    project_root = tmp_path / "project"
    project_root.mkdir()
    state_dir = tmp_path / "state"
    config = make_test_config(project_root)
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=state_dir, config=config.runner),
        thread_store=ThreadStore(state_dir),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread()
    monkeypatch.setenv("UV_AGENT_RUNTIME_THREAD_ID", thread_id)
    wf = workflow.start("Long task", state_dir=state_dir)
    wf.agent("Do the first part", key="first")

    summary = engine._compaction_summary_with_active_workflows(thread_id, "Conversation summary")

    assert summary.startswith("Conversation summary")
    assert "## 活跃工作流" in summary
    assert wf.workflow_id in summary
    assert "workflow.resume" in summary

def test_goal_mode_notice_emits_once_per_epoch_and_after_disable(tmp_path: Path) -> None:
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
    thread_id = engine.thread_store.create_thread()

    enabled_state = engine.enable_goal_mode(thread_id, objective="Ship the goal feature")
    first = engine._pre_user_context_items(thread_id)
    repeated = engine._pre_user_context_items(thread_id)

    first_text = "\n".join(message_item_text(item) for item in first)
    assert '<goal_mode status="enabled">' in first_text
    assert "Ship the goal feature" in first_text
    assert str(enabled_state.paths.checklist) in first_text
    assert '<goal_mode status="enabled">' not in str(repeated)

    engine.thread_store.append(thread_id, "item.compaction", turn_id="t1", text="summary", usage={})
    after_compaction = engine._pre_user_context_items(thread_id)
    assert '<goal_mode status="enabled">' in str(after_compaction)

    engine.disable_goal_mode(thread_id)
    disabled = engine._pre_user_context_items(thread_id)
    repeated_disabled = engine._pre_user_context_items(thread_id)
    assert '<goal_mode status="disabled">' in str(disabled)
    assert '<goal_mode status="disabled">' not in str(repeated_disabled)

    engine.thread_store.append(thread_id, "item.compaction", turn_id="t2", text="summary", usage={})
    assert '<goal_mode' not in str(engine._pre_user_context_items(thread_id))


@pytest.mark.asyncio
async def test_goal_mode_enable_notice_reaches_first_send(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root, title_generation=TitleGenerationConfig(enabled=False))
    model_client = CompletedOnlyStreamClient(
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
        model_client=model_client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread()

    engine.enable_goal_mode(thread_id, objective="lazy goal")
    events = [event async for event in engine.run_turn(user_text="start", thread_id=thread_id)]

    assert any(event.get("type") == "turn.completed" for event in events)
    request_text = "\n".join(message_item_text(item) for item in model_client.requests[0]["input"])
    assert '<goal_mode status="enabled">' in request_text
    assert "lazy goal" in request_text
    assert engine.thread_store.read_events(thread_id, event_types={"item.goal_mode_notice"})


def test_goal_mode_reenable_before_next_turn_emits_enabled_notice(tmp_path: Path) -> None:
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
    thread_id = engine.thread_store.create_thread()
    engine.enable_goal_mode(thread_id)
    assert '<goal_mode status="enabled">' in str(engine._pre_user_context_items(thread_id))

    engine.disable_goal_mode(thread_id)
    engine.reset_goal_files(thread_id, objective="fresh goal")
    engine.enable_goal_mode(thread_id)

    notice = "\n".join(message_item_text(item) for item in engine._pre_user_context_items(thread_id))
    assert '<goal_mode status="enabled">' in notice
    assert '<goal_mode status="disabled">' not in notice
    assert "fresh goal" in notice


def test_goal_mode_notice_is_pre_user_context_and_not_retained(tmp_path: Path) -> None:
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
    thread_id = engine.thread_store.create_thread()
    engine.enable_goal_mode(thread_id)
    goal_item = engine._pre_user_context_items(thread_id)[0]
    engine.thread_store.append(thread_id, "item.user", turn_id="t1", item=message_item("user", "do work"))

    assert engine._is_pre_user_context_item(goal_item)
    assert retain_item_after_compaction(goal_item) is False
    reconstructed = engine._reconstruct_input(thread_id)
    reconstructed_texts = [message_item_text(item) for item in reconstructed]
    assert "<goal_mode" in reconstructed_texts[0]
    assert "do work" in reconstructed_texts


def test_worktree_notice_emits_once_per_epoch_and_after_delete(tmp_path: Path) -> None:
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
    thread_id = engine.thread_store.create_thread("Worktree feature")
    worktree_path = project_root / ".uv-agent" / "worktrees" / "feature"

    engine.thread_store.append(
        thread_id,
        "thread.worktree_created",
        worktree_status="active",
        worktree_branch="feature",
        worktree_path=str(worktree_path),
        worktree_base_ref="HEAD",
        worktree_origin_root=str(project_root),
        worktree_head="abc123",
        worktree_created_at="2026-01-01T00:00:00Z",
    )
    engine.thread_store.append(thread_id, "thread.cwd_updated", cwd=str(worktree_path))
    first = engine._pre_user_context_items(thread_id)
    repeated = engine._pre_user_context_items(thread_id)

    first_text = "\n".join(message_item_text(item) for item in first)
    assert '<worktree status="active">' in first_text
    assert str(worktree_path) in first_text
    assert "origin workspace" in first_text and "不是" in first_text
    assert '<worktree status="active">' not in str(repeated)

    engine.thread_store.append(thread_id, "item.compaction", turn_id="t1", text="summary", usage={})
    after_compaction = engine._pre_user_context_items(thread_id)
    assert '<worktree status="active">' in str(after_compaction)

    engine.thread_store.append(
        thread_id,
        "thread.worktree_deleted",
        worktree_branch="feature",
        worktree_path=str(worktree_path),
        worktree_origin_root=str(project_root),
        worktree_deleted_at="2026-01-02T00:00:00Z",
        worktree_deleted_head="def456",
        worktree_deleted_status=" M file.py",
    )
    engine.thread_store.append(thread_id, "thread.cwd_updated", cwd=str(project_root))
    deleted = engine._pre_user_context_items(thread_id)
    repeated_deleted = engine._pre_user_context_items(thread_id)
    assert '<worktree status="deleted">' in str(deleted)
    assert "def456" in str(deleted)
    assert '<worktree status="deleted">' not in str(repeated_deleted)

    engine.thread_store.append(thread_id, "item.compaction", turn_id="t2", text="summary", usage={})
    assert "<worktree" not in str(engine._pre_user_context_items(thread_id))


@pytest.mark.asyncio
async def test_worktree_notice_reaches_first_send_and_coexists_with_goal_mode(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = make_test_config(project_root, title_generation=TitleGenerationConfig(enabled=False))
    model_client = CompletedOnlyStreamClient(
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
        model_client=model_client,
        runner=PythonRunner(project_root=project_root, data_dir=tmp_path / "state", config=config.runner),
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )
    thread_id = engine.thread_store.create_thread("Goal worktree")
    worktree_path = project_root / ".uv-agent" / "worktrees" / "feature"
    engine.enable_goal_mode(thread_id, objective="ship from worktree")
    engine.thread_store.append(
        thread_id,
        "thread.worktree_created",
        worktree_status="active",
        worktree_branch="feature",
        worktree_path=str(worktree_path),
        worktree_base_ref="HEAD",
        worktree_origin_root=str(project_root),
        worktree_head="abc123",
        worktree_created_at="2026-01-01T00:00:00Z",
    )
    engine.thread_store.append(thread_id, "thread.cwd_updated", cwd=str(worktree_path))

    events = [event async for event in engine.run_turn(user_text="start", thread_id=thread_id)]

    assert any(event.get("type") == "turn.completed" for event in events)
    request_text = "\n".join(message_item_text(item) for item in model_client.requests[0]["input"])
    assert '<goal_mode status="enabled">' in request_text
    assert '<worktree status="active">' in request_text
    assert request_text.index('<goal_mode status="enabled">') < request_text.index('<worktree status="active">')
    assert engine.thread_store.read_events(thread_id, event_types={"item.worktree_notice"})


def test_worktree_notice_is_pre_user_context_and_not_retained(tmp_path: Path) -> None:
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
    thread_id = engine.thread_store.create_thread("Worktree")
    worktree_path = project_root / ".uv-agent" / "worktrees" / "feature"
    engine.thread_store.append(
        thread_id,
        "thread.worktree_created",
        worktree_status="active",
        worktree_branch="feature",
        worktree_path=str(worktree_path),
        worktree_base_ref="HEAD",
        worktree_origin_root=str(project_root),
    )
    engine.thread_store.append(thread_id, "thread.cwd_updated", cwd=str(worktree_path))
    worktree_item = engine._pre_user_context_items(thread_id)[0]
    engine.thread_store.append(thread_id, "item.user", turn_id="t1", item=message_item("user", "do work"))

    assert engine._is_pre_user_context_item(worktree_item)
    assert retain_item_after_compaction(worktree_item) is False
    reconstructed = engine._reconstruct_input(thread_id)
    reconstructed_texts = [message_item_text(item) for item in reconstructed]
    assert any("<worktree" in text for text in reconstructed_texts[:2])
    assert "do work" in reconstructed_texts


def test_goal_mode_reset_requires_disabled_mode_and_preserves_files_on_disable(tmp_path: Path) -> None:
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
    thread_id = engine.thread_store.create_thread()
    state = engine.enable_goal_mode(thread_id)
    state.paths.checklist.write_text("custom checklist", encoding="utf-8")

    with pytest.raises(ValueError):
        engine.reset_goal_files(thread_id)

    engine.disable_goal_mode(thread_id)
    assert state.paths.checklist.read_text(encoding="utf-8") == "custom checklist"
    reset_state = engine.reset_goal_files(thread_id, objective="new objective")
    assert "new objective" in reset_state.paths.checklist.read_text(encoding="utf-8")
    assert engine.goal_state(thread_id) is not None
    assert engine.goal_state(thread_id).enabled is False


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

    assert "<retained_history_message" in message_item_text(reconstructed[0])
    assert "kept" in text
    assert "summary" in text
    assert "new" in text
    assert "old" not in text


def test_reconstruct_input_places_post_compaction_context_before_replacement(tmp_path: Path) -> None:
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
    thread_id = engine.thread_store.create_thread()
    replacement = [
        message_item("user", "kept request"),
        message_item("user", "<conversation_summary>\nsummary\n</conversation_summary>"),
    ]
    engine.thread_store.append(
        thread_id,
        "item.compaction",
        turn_id="t1",
        text="summary",
        replacement_input=replacement,
        usage={},
    )
    engine.thread_store.append(
        thread_id,
        "item.context_update",
        turn_id="t1",
        context_fingerprint="fp",
        context_state={"fingerprint": "fp", "parts": {"runtime": {}}},
        context_kind="runtime",
        removed=[],
        text="<context_update id=\"runtime_context\" status=\"current\">\ncurrent context\n</context_update>",
    )
    engine.thread_store.append(thread_id, "item.user", turn_id="t2", item=message_item("user", "new request"))

    reconstructed = engine._reconstruct_input(thread_id)

    assert message_item_text(reconstructed[0]).startswith("<context_update")
    assert "<retained_history_message" in message_item_text(reconstructed[1])
    assert "kept request" in message_item_text(reconstructed[1])
    assert "<conversation_summary>" in message_item_text(reconstructed[2])
    assert message_item_text(reconstructed[3]) == "new request"


def test_prepare_turn_prelude_inserts_new_context_before_compacted_history(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "AGENTS.md").write_text("Reloaded rule.", encoding="utf-8")
    config = make_test_config(project_root)
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
        "item.compaction",
        turn_id="t1",
        text="summary",
        replacement_input=[
            message_item("user", "kept request"),
            message_item("user", "<conversation_summary>\nsummary\n</conversation_summary>"),
        ],
        usage={},
    )

    prelude = engine._prepare_run_turn_prelude(
        user_text="new request",
        thread_id=thread_id,
        level=None,
        image_paths=None,
        cancel_event=None,
    )

    texts = [message_item_text(item) for item in prelude.input_items if item.get("type") == "message"]
    assert texts[0].startswith("<workspace_rules")
    assert "Reloaded rule." in texts[0]
    retained_index = next(index for index, text in enumerate(texts) if "<retained_history_message" in text)
    summary_index = next(index for index, text in enumerate(texts) if "<conversation_summary>" in text)
    assert "kept request" in texts[retained_index]
    assert texts[retained_index - 1].startswith("<context_update")
    assert retained_index < summary_index
    assert message_item_text(prelude.user_item) == "new request"
    assert "new request" not in texts


@pytest.mark.asyncio
async def test_mid_turn_compaction_readds_epoch_context_before_continuing(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "AGENTS.md").write_text("Mid-turn rule.", encoding="utf-8")
    config = make_test_config(
        project_root,
        context_window_tokens=20,
        compression=CompressionConfig(enabled=True, model_level="small", trigger_ratio=0.1, min_tokens=1),
    )
    client = CompletedOnlyStreamClient(
        [
            {
                "id": "resp_tool",
                "output_text": "",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_mid_compact",
                        "name": "run_python",
                        "arguments": json.dumps({"code": "print('hello')"}),
                    }
                ],
            },
            {
                "id": "resp_compact",
                "output_text": "summary includes tool result",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "summary includes tool result"}],
                    }
                ],
            },
            {
                "id": "resp_final",
                "output_text": "done after compaction",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done after compaction"}],
                    }
                ],
            },
        ]
    )
    engine = AgentEngine(
        config=config,
        model_client=client,
        runner=SimpleRunner(),  # type: ignore[arg-type]
        thread_store=ThreadStore(tmp_path / "state"),
        project_root=project_root,
    )

    [event async for event in engine.run_turn(user_text="run a tool")]

    continued_input = client.requests[2]["input"]
    continued_texts = [message_item_text(item) for item in continued_input if item.get("type") == "message"]
    assert continued_texts[0].startswith("<workspace_rules")
    assert "Mid-turn rule." in continued_texts[0]
    retained_index = next(index for index, text in enumerate(continued_texts) if "<retained_history_message" in text)
    summary_index = next(index for index, text in enumerate(continued_texts) if "<conversation_summary>" in text)
    assert continued_texts[retained_index - 1].startswith("<context_update")
    assert retained_index < summary_index


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
        context_kind="runtime",
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
    assert "src/AGENTS.md" in text
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


def test_context_stats_is_cached_within_ttl(tmp_path: Path) -> None:
    config = AppConfig(
        providers={
            "default": ProviderConfig(name="default", base_url="https://api.example.com/v1")
        },
        models={
            "default": ModelConfig(name="default", provider="default", model="remote")
        },
        levels={"medium": LevelConfig(name="medium", model="default")},
        runtime=RuntimeConfig(),
        runner=RunnerConfig(),
    )
    runner = PythonRunner(
        project_root=tmp_path,
        data_dir=tmp_path / ".uv-agent",
        config=RunnerConfig(),
    )
    engine = AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=runner,
        thread_store=ThreadStore(tmp_path / ".uv-agent"),
        project_root=tmp_path,
    )

    stats1 = engine.context_stats(None, "medium")
    assert (None, "medium") in engine._context_stats_cache
    stats2 = engine.context_stats(None, "medium")

    assert stats1 is stats2
