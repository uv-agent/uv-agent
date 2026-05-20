from __future__ import annotations

import pytest

from uv_agent.config import (
    EndpointConfig,
    MessagePassthroughConfig,
    ModelConfig,
    ProviderConfig,
    ReasoningDisplayConfig,
)
from uv_agent.model import (
    ModelStreamEvent,
    anthropic_image_source,
    anthropic_sdk_base_url,
    anthropic_messages,
    anthropic_payload,
    chat_create_kwargs,
    chat_message_content,
    chat_messages,
    chat_payload,
    create_anthropic_response,
    create_chat_response,
    create_responses_response,
    endpoint_extra_body,
    openai_client,
    parse_anthropic_message,
    parse_anthropic_response,
    parse_chat_response,
    parse_chat_response_for_model,
    parse_responses_response,
    responses_create_kwargs,
    responses_payload,
    stream_chat_response,
)


class FakeOpenAIStream:
    def __init__(self, events):
        self.events = events

    def __aiter__(self):
        self._iter = iter(self.events)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FakeChatCompletions:
    def __init__(self, events):
        self.events = events
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return FakeOpenAIStream(self.events)
        return self.events[0]


class FakeOpenAIClient:
    def __init__(self, *, chat_events=None, response_events=None):
        self.chat = type("Chat", (), {"completions": FakeChatCompletions(chat_events or [])})()
        self.responses = type("Responses", (), {"create": self._create_response})()
        self.response_events = response_events or []
        self.response_kwargs = None

    async def _create_response(self, **kwargs):
        self.response_kwargs = kwargs
        if kwargs.get("stream"):
            return FakeOpenAIStream(self.response_events)
        return self.response_events[0]


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


def test_chat_messages_keep_assistant_text_and_tool_calls_together() -> None:
    messages = chat_messages(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will inspect the files."}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "run_python",
                "arguments": "{\"code\":\"print('ok')\"}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "{\"ok\":true}"},
        ],
        None,
    )

    assert messages == [
        {
            "role": "assistant",
            "content": "I will inspect the files.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "run_python",
                        "arguments": "{\"code\":\"print('ok')\"}",
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\":true}"},
    ]


def test_chat_messages_keep_assistant_tool_calls_together_across_unknown_items() -> None:
    messages = chat_messages(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will inspect the files."}],
            },
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "run_python",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "{\"ok\":true}"},
        ],
        None,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == "I will inspect the files."
    assert messages[0]["tool_calls"][0]["id"] == "call_1"
    assert messages[1]["role"] == "tool"


def test_responses_refusal_content_is_treated_as_visible_text() -> None:
    response = parse_responses_response(
        {
            "id": "resp_1",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "text": "I cannot help with that."}],
                }
            ],
        }
    )

    assert response.output_text == "I cannot help with that."
    assert chat_message_content(response.output[0]) == "I cannot help with that."


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


def test_openai_client_strips_sdk_owned_endpoint_path_and_preserves_header_only_auth() -> None:
    provider = ProviderConfig(
        name="p",
        base_url="https://api.example.com/v1",
        headers={"api-key": "test-key"},
        responses=EndpointConfig(path="/responses"),
    )

    client = openai_client(provider, "responses", "/responses")

    assert str(client.base_url) == "https://api.example.com/v1/"
    assert client.auth_headers == {}
    assert client.default_headers["api-key"] == "test-key"


def test_responses_create_kwargs_passes_unknown_params_as_extra_body() -> None:
    provider = ProviderConfig(
        name="p",
        base_url="https://api.example.com/v1",
        params={"temperature": 0.2, "vendor_flag": True},
        responses=EndpointConfig(path="/responses", params={"extra_body": {"endpoint_flag": "yes"}}),
    )
    model = ModelConfig(name="m", provider="p", model="remote", params={"custom": "x"})

    kwargs = responses_create_kwargs(
        provider=provider,
        model=model,
        input_items=[],
        tools=[],
        instructions=None,
        previous_response_id=None,
    )

    assert kwargs["temperature"] == 0.2
    assert kwargs["extra_body"] == {
        "vendor_flag": True,
        "endpoint_flag": "yes",
        "custom": "x",
    }
    assert "vendor_flag" not in kwargs
    assert "custom" not in kwargs


@pytest.mark.asyncio
async def test_create_responses_response_uses_sdk_client() -> None:
    provider = ProviderConfig(name="p", base_url="https://api.example.com/v1")
    model = ModelConfig(name="m", provider="p", model="remote", api="responses")
    sdk_client = FakeOpenAIClient(
        response_events=[
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

    response = await create_responses_response(
        provider=provider,
        model=model,
        input_items=[],
        tools=[],
        instructions="system",
        previous_response_id="resp_prev",
        client=sdk_client,
    )

    assert response.output_text == "done"
    assert sdk_client.response_kwargs["model"] == "remote"
    assert sdk_client.response_kwargs["instructions"] == "system"
    assert sdk_client.response_kwargs["previous_response_id"] == "resp_prev"


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


def test_chat_create_kwargs_passes_unknown_params_as_extra_body() -> None:
    provider = ProviderConfig(
        name="p",
        base_url="https://api.example.com/v1",
        params={"temperature": 0.2, "vendor_flag": True},
        chat_completions=EndpointConfig(
            path="/chat/completions",
            params={"extra_body": {"endpoint_flag": "yes"}},
        ),
    )
    model = ModelConfig(name="m", provider="p", model="remote", params={"custom": "x"})

    kwargs = chat_create_kwargs(
        provider=provider,
        model=model,
        input_items=[],
        tools=[],
        instructions=None,
    )

    assert kwargs["temperature"] == 0.2
    assert kwargs["extra_body"] == {
        "vendor_flag": True,
        "endpoint_flag": "yes",
        "custom": "x",
    }
    assert "vendor_flag" not in kwargs
    assert "custom" not in kwargs


@pytest.mark.asyncio
async def test_create_chat_response_uses_sdk_client() -> None:
    provider = ProviderConfig(name="p", base_url="https://api.example.com/v1")
    model = ModelConfig(name="m", provider="p", model="remote", api="chat_completions")
    sdk_client = FakeOpenAIClient(
        chat_events=[
            {
                "id": "chat_1",
                "choices": [{"message": {"content": "done"}}],
                "usage": {"total_tokens": 2},
            }
        ]
    )

    response = await create_chat_response(
        provider=provider,
        model=model,
        input_items=[],
        tools=[],
        instructions="system",
        client=sdk_client,
    )

    assert response.output_text == "done"
    assert sdk_client.chat.completions.kwargs["model"] == "remote"
    assert sdk_client.chat.completions.kwargs["messages"] == [{"role": "system", "content": "system"}]


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


def test_anthropic_sdk_base_url_strips_default_messages_path() -> None:
    provider = ProviderConfig(
        name="p",
        base_url="https://api.anthropic.com",
        anthropic_messages=EndpointConfig(path="/v1/messages"),
    )

    assert anthropic_sdk_base_url(provider) == "https://api.anthropic.com"


def test_anthropic_sdk_base_url_strips_messages_path_when_base_url_has_v1() -> None:
    provider = ProviderConfig(
        name="p",
        base_url="https://api.example.com/v1",
        anthropic_messages=EndpointConfig(path="/messages"),
    )

    assert anthropic_sdk_base_url(provider) == "https://api.example.com"


def test_anthropic_endpoint_extra_body_keeps_unknown_params() -> None:
    provider = ProviderConfig(
        name="p",
        base_url="https://api.anthropic.com",
        params={"temperature": 0.2, "vendor_flag": True, "extra_body": {"manual": "y"}},
        anthropic_messages=EndpointConfig(path="/v1/messages", params={"max_tokens": 99}),
    )
    model = ModelConfig(name="m", provider="p", model="claude", params={"custom": "x"})

    assert endpoint_extra_body(provider, model) == {
        "manual": "y",
        "vendor_flag": True,
        "custom": "x",
    }


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


def test_parse_anthropic_message_maps_sdk_message() -> None:
    class Message:
        def model_dump(self, mode):
            assert mode == "json"
            return {
                "id": "msg_1",
                "usage": {"input_tokens": 1, "output_tokens": 2},
                "content": [{"type": "text", "text": "hello"}],
            }

    response = parse_anthropic_message(Message())

    assert response.id == "msg_1"
    assert response.output_text == "hello"
    assert response.usage == {"input_tokens": 1, "output_tokens": 2}


@pytest.mark.asyncio
async def test_create_anthropic_response_uses_sdk_client() -> None:
    class Messages:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs

            class Message:
                def model_dump(self, mode):
                    return {
                        "id": "msg_1",
                        "usage": {},
                        "content": [{"type": "text", "text": "done"}],
                    }

            return Message()

    class Client:
        def __init__(self) -> None:
            self.messages = Messages()

    provider = ProviderConfig(name="p", base_url="https://api.anthropic.com")
    model = ModelConfig(name="m", provider="p", model="claude", api="anthropic_messages")
    client = Client()

    response = await create_anthropic_response(
        provider=provider,
        model=model,
        input_items=[],
        tools=[],
        instructions="system",
        client=client,
    )

    assert response.output_text == "done"
    assert client.messages.kwargs["model"] == "claude"
    assert client.messages.kwargs["messages"] == []
    assert client.messages.kwargs["system"] == "system"


@pytest.mark.asyncio
async def test_create_anthropic_response_passes_unknown_params_as_extra_body() -> None:
    class Messages:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs

            class Message:
                def model_dump(self, mode):
                    return {"id": "msg_1", "usage": {}, "content": []}

            return Message()

    class Client:
        def __init__(self) -> None:
            self.messages = Messages()

    provider = ProviderConfig(
        name="p",
        base_url="https://api.anthropic.com",
        params={"temperature": 0.2, "vendor_flag": True, "extra_body": {"manual": "y"}},
    )
    model = ModelConfig(name="m", provider="p", model="claude", params={"custom": "x"})
    client = Client()

    await create_anthropic_response(
        provider=provider,
        model=model,
        input_items=[],
        tools=[],
        instructions=None,
        client=client,
    )

    assert client.messages.kwargs["temperature"] == 0.2
    assert client.messages.kwargs["extra_body"] == {"manual": "y", "vendor_flag": True, "custom": "x"}
    assert "vendor_flag" not in client.messages.kwargs
    assert "custom" not in client.messages.kwargs


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
) -> None:
    sdk_client = FakeOpenAIClient(
        chat_events=[
            {
            "id": "chat_1",
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "think ",
                    }
                }
            ],
            },
            {
            "id": "chat_1",
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "more",
                        "content": "done",
                    }
                }
            ],
            },
        ]
    )
    provider = ProviderConfig(name="p", base_url="https://example.com")
    model = ModelConfig(
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

    events = [
        event
        async for event in stream_chat_response(
            provider=provider,
            model=model,
            input_items=[],
            tools=[],
            instructions=None,
            client=sdk_client,
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
async def test_stream_chat_allows_empty_chunks_before_done(
) -> None:
    sdk_client = FakeOpenAIClient(
        chat_events=[
            {"id": "chat_1", "choices": []},
            {"id": "chat_1", "choices": [{"delta": {}}]},
        ]
    )
    provider = ProviderConfig(name="p", base_url="https://example.com")
    model = ModelConfig(name="m", provider="p", model="remote", api="chat_completions")

    events = [
        event
        async for event in stream_chat_response(
            provider=provider,
            model=model,
            input_items=[],
            tools=[],
            instructions=None,
            client=sdk_client,
        )
    ]

    assert len(events) == 1
    assert events[0].type == "completed"
    assert events[0].response is not None
    assert events[0].response.output_text == ""
    assert events[0].response.output == []


@pytest.mark.asyncio
async def test_stream_chat_can_treat_unknown_text_delta_as_reasoning(
) -> None:
    sdk_client = FakeOpenAIClient(
        chat_events=[{"id": "chat_1", "choices": [{"delta": {"vendor_thought": "hidden-ish"}}]}]
    )
    provider = ProviderConfig(name="p", base_url="https://example.com")
    model = ModelConfig(
        name="m",
        provider="p",
        model="remote",
        api="chat_completions",
        reasoning_display=ReasoningDisplayConfig(unknown_text_delta_as_reasoning=True),
    )

    events = [
        event
        async for event in stream_chat_response(
            provider=provider,
            model=model,
            input_items=[],
            tools=[],
            instructions=None,
            client=sdk_client,
        )
    ]

    assert [event.text for event in events if event.type == "reasoning_delta"] == ["hidden-ish"]
    completed = next(event for event in events if event.type == "completed")
    assert completed.response is not None
    assert completed.response.reasoning_text == "hidden-ish"


@pytest.mark.asyncio
async def test_stream_chat_fallback_can_display_passthrough_field_as_reasoning(
) -> None:
    sdk_client = FakeOpenAIClient(
        chat_events=[{"id": "chat_1", "choices": [{"delta": {"reasoning_content": "think"}}]}]
    )
    provider = ProviderConfig(name="p", base_url="https://example.com")
    model = ModelConfig(
        name="m",
        provider="p",
        model="remote",
        api="chat_completions",
        message_passthrough=MessagePassthroughConfig(assistant=["reasoning_content"]),
        reasoning_display=ReasoningDisplayConfig(unknown_text_delta_as_reasoning=True),
    )

    events = [
        event
        async for event in stream_chat_response(
            provider=provider,
            model=model,
            input_items=[],
            tools=[],
            instructions=None,
            client=sdk_client,
        )
    ]

    assert [event.text for event in events if event.type == "reasoning_delta"] == ["think"]
    completed = next(event for event in events if event.type == "completed")
    assert completed.response is not None
    assert completed.response.output[0]["reasoning_content"] == "think"
    assert completed.response.reasoning_text == "think"
