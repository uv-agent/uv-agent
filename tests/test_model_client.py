from __future__ import annotations

import pytest

from uv_agent.config import (
    AppConfig,
    CompressionConfig,
    EndpointConfig,
    LevelConfig,
    MessagePassthroughConfig,
    ModelConfig,
    ProviderConfig,
    ReasoningDisplayConfig,
    RunnerConfig,
    RuntimeConfig,
)
from uv_agent.model_client import (
    ModelStreamEvent,
    UnifiedModelClient,
    anthropic_image_source,
    anthropic_messages,
    anthropic_payload,
    chat_message_content,
    chat_messages,
    chat_payload,
    parse_anthropic_response,
    parse_chat_response,
    parse_chat_response_for_model,
    parse_sse_event,
    responses_payload,
)


def test_chat_messages_convert_responses_items() -> None:
    messages = chat_messages(
        [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "run_python",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "{\"ok\":true}",
            },
        ],
        "system",
    )

    assert messages[0] == {"role": "system", "content": "system"}
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2]["tool_calls"][0]["function"]["name"] == "run_python"
    assert messages[3]["role"] == "tool"


def test_chat_payload_uses_chat_endpoint_shape() -> None:
    provider = ProviderConfig(
        name="p",
        base_url="https://example.com",
        chat_completions=EndpointConfig(path="/chat", params={"temperature": 0}),
    )
    model = ModelConfig(name="m", provider="p", model="remote", api="chat_completions")

    payload = chat_payload(provider, model, [], [], None, stream=True)

    assert payload["model"] == "remote"
    assert payload["messages"] == []
    assert payload["stream"] is True
    assert payload["temperature"] == 0
    assert "tools" not in payload


def test_chat_messages_replay_configured_message_passthrough_fields() -> None:
    model = ModelConfig(
        name="m",
        provider="p",
        model="remote",
        api="chat_completions",
        message_passthrough=MessagePassthroughConfig(assistant=["reasoning_content"]),
    )

    messages = chat_messages(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello! I am MiMo."}],
                "reasoning_content": "I should introduce myself.",
            }
        ],
        None,
        model,
    )

    assert messages == [
        {
            "role": "assistant",
            "content": "Hello! I am MiMo.",
            "reasoning_content": "I should introduce myself.",
        }
    ]


def test_responses_payload_supports_previous_response_id() -> None:
    provider = ProviderConfig(name="p", base_url="https://example.com")
    model = ModelConfig(name="m", provider="p", model="remote", api="responses")
    input_items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "next"}]}
    ]

    payload = responses_payload(
        provider,
        model,
        input_items,
        [],
        "system",
        stream=True,
        previous_response_id="resp_1",
    )
    input_items[0]["content"][0]["text"] = "mutated"

    assert payload["previous_response_id"] == "resp_1"
    assert payload["input"][0]["content"][0]["text"] == "next"
    assert payload["stream"] is True


def test_parse_chat_response_maps_tool_calls() -> None:
    response = parse_chat_response(
        {
            "id": "chat_1",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "run_python", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ],
        }
    )

    assert response.output[0]["type"] == "function_call"
    assert response.output[0]["call_id"] == "call_1"


def test_parse_chat_response_preserves_passthrough_and_reasoning_fields() -> None:
    model = ModelConfig(
        name="m",
        provider="p",
        model="remote",
        api="chat_completions",
        message_passthrough=MessagePassthroughConfig(assistant=["reasoning_content"]),
        reasoning_display=ReasoningDisplayConfig(assistant_message_fields=["reasoning_content"]),
    )

    response = parse_chat_response_for_model(
        {
            "id": "chat_1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I am MiMo.",
                        "reasoning_content": "I should introduce myself.",
                    }
                }
            ],
        },
        model,
    )

    assert response.output[0]["reasoning_content"] == "I should introduce myself."
    assert response.reasoning_text == "I should introduce myself."


def test_parse_sse_event_handles_done_and_json() -> None:
    assert parse_sse_event(None, ["[DONE]"]) is None
    parsed = parse_sse_event("response.completed", ['{"response":{"id":"x"}}'])
    assert parsed == {"type": "response.completed", "response": {"id": "x"}}


def test_anthropic_payload_uses_messages_shape() -> None:
    provider = ProviderConfig(
        name="p",
        base_url="https://example.com",
        anthropic_messages=EndpointConfig(path="/v1/messages", params={"max_tokens": 99}),
    )
    model = ModelConfig(name="m", provider="p", model="claude", api="anthropic_messages")

    payload = anthropic_payload(provider, model, [], [], "system", stream=True)

    assert payload["model"] == "claude"
    assert payload["system"] == "system"
    assert payload["messages"] == []
    assert payload["stream"] is True
    assert payload["max_tokens"] == 99
    assert "tools" not in payload


def test_anthropic_messages_convert_tool_items() -> None:
    messages = anthropic_messages(
        [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {
                "type": "function_call",
                "call_id": "toolu_1",
                "name": "run_python",
                "arguments": "{\"code\":\"print(1)\"}",
            },
            {"type": "function_call_output", "call_id": "toolu_1", "output": "{\"ok\":true}"},
        ]
    )

    assert messages[0] == {"role": "user", "content": "hi"}
    assert messages[1]["content"][0]["type"] == "tool_use"
    assert messages[2]["content"][0]["type"] == "tool_result"


def test_parse_anthropic_response_maps_tool_use() -> None:
    response = parse_anthropic_response(
        {
            "id": "msg_1",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "run_python",
                    "input": {"code": "print(1)"},
                },
            ],
        }
    )

    assert response.output_text == "hello"
    assert response.output[1]["type"] == "function_call"
    assert response.output[1]["call_id"] == "toolu_1"


def test_image_parts_convert_for_chat_and_anthropic() -> None:
    item = {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ],
    }

    chat = chat_message_content(item)
    anthropic = anthropic_messages([item])[0]["content"]

    assert isinstance(chat, list)
    assert chat[1]["type"] == "image_url"
    assert isinstance(anthropic, list)
    assert anthropic[1]["source"]["media_type"] == "image/png"
    assert anthropic_image_source("https://example.com/image.png") is None


@pytest.mark.asyncio
async def test_stream_chat_accumulates_passthrough_and_configured_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stream_sse(provider, api, payload):
        yield {
            "id": "chat_1",
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "think ",
                    }
                }
            ],
        }
        yield {
            "id": "chat_1",
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "more",
                        "content": "done",
                    }
                }
            ],
        }

    monkeypatch.setattr("uv_agent.model_client.stream_sse", fake_stream_sse)
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "m": ModelConfig(
                name="m",
                provider="p",
                model="remote",
                api="chat_completions",
                message_passthrough=MessagePassthroughConfig(assistant=["reasoning_content"]),
                reasoning_display=ReasoningDisplayConfig(
                    stream_delta_fields=["reasoning_content"],
                    assistant_message_fields=["reasoning_content"],
                ),
            )
        },
        levels={"medium": LevelConfig(name="medium", model="m", params={})},
        runtime=RuntimeConfig(compression=CompressionConfig(enabled=False)),
        runner=RunnerConfig(runtime_dependency="uv-agent==0.1.4"),
    )
    client = UnifiedModelClient(config)

    events = [
        event
        async for event in client.stream_response(
            input_items=[],
            level="medium",
            tools=[],
            instructions=None,
        )
    ]
    reasoning = [event.text for event in events if event.type == "reasoning_delta"]
    completed = next(event for event in events if event.type == "completed")

    assert reasoning == ["think ", "more"]
    assert isinstance(completed, ModelStreamEvent)
    assert completed.response is not None
    assert completed.response.output[0]["reasoning_content"] == "think more"
    assert completed.response.reasoning_text == "think more"


@pytest.mark.asyncio
async def test_stream_chat_can_treat_unknown_text_delta_as_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stream_sse(provider, api, payload):
        yield {"id": "chat_1", "choices": [{"delta": {"vendor_thought": "hidden-ish"}}]}

    monkeypatch.setattr("uv_agent.model_client.stream_sse", fake_stream_sse)
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "m": ModelConfig(
                name="m",
                provider="p",
                model="remote",
                api="chat_completions",
                reasoning_display=ReasoningDisplayConfig(unknown_text_delta_as_reasoning=True),
            )
        },
        levels={"medium": LevelConfig(name="medium", model="m", params={})},
        runtime=RuntimeConfig(compression=CompressionConfig(enabled=False)),
        runner=RunnerConfig(runtime_dependency="uv-agent==0.1.4"),
    )
    client = UnifiedModelClient(config)

    events = [
        event
        async for event in client.stream_response(
            input_items=[],
            level="medium",
            tools=[],
            instructions=None,
        )
    ]

    assert [event.text for event in events if event.type == "reasoning_delta"] == ["hidden-ish"]
    completed = next(event for event in events if event.type == "completed")
    assert completed.response is not None
    assert completed.response.reasoning_text == "hidden-ish"


@pytest.mark.asyncio
async def test_stream_chat_fallback_can_display_passthrough_field_as_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stream_sse(provider, api, payload):
        yield {"id": "chat_1", "choices": [{"delta": {"reasoning_content": "think"}}]}

    monkeypatch.setattr("uv_agent.model_client.stream_sse", fake_stream_sse)
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "m": ModelConfig(
                name="m",
                provider="p",
                model="remote",
                api="chat_completions",
                message_passthrough=MessagePassthroughConfig(assistant=["reasoning_content"]),
                reasoning_display=ReasoningDisplayConfig(unknown_text_delta_as_reasoning=True),
            )
        },
        levels={"medium": LevelConfig(name="medium", model="m", params={})},
        runtime=RuntimeConfig(compression=CompressionConfig(enabled=False)),
        runner=RunnerConfig(runtime_dependency="uv-agent==0.1.4"),
    )
    client = UnifiedModelClient(config)

    events = [
        event
        async for event in client.stream_response(
            input_items=[],
            level="medium",
            tools=[],
            instructions=None,
        )
    ]

    assert [event.text for event in events if event.type == "reasoning_delta"] == ["think"]
    completed = next(event for event in events if event.type == "completed")
    assert completed.response is not None
    assert completed.response.output[0]["reasoning_content"] == "think"
    assert completed.response.reasoning_text == "think"
