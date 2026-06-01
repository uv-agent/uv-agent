from __future__ import annotations

import openai
import pytest

from uv_agent.config import (
    EndpointConfig,
    MessagePassthroughConfig,
    ModelConfig,
    ProviderConfig,
    ReasoningDisplayConfig,
)
from uv_agent.errors import EmptyModelStreamError
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
    stream_anthropic_response,
    stream_chat_response,
    stream_responses_response,
)
from uv_agent.model.sdk import object_dump


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


def test_object_dump_ignores_non_mapping_iterables() -> None:
    assert object_dump(["not", "pairs"]) == {}


class FakeAnthropicStream:
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


def test_chat_messages_replaces_invalid_passthrough_tool_calls_field() -> None:
    model = ModelConfig(
        name="m",
        provider="p",
        model="remote",
        api="chat_completions",
        message_passthrough=MessagePassthroughConfig(assistant=["tool_calls"]),
    )

    messages = chat_messages(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will run it."}],
                "tool_calls": "provider-specific-text",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "run_python",
                "arguments": "{}",
            },
        ],
        None,
        model,
    )

    assert messages[0]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "run_python", "arguments": "{}"},
        }
    ]


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
    assert "tool_choice" not in payload


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
    assert "tool_choice" not in payload


def test_tool_choice_is_not_added_by_default() -> None:
    tool = {
        "type": "function",
        "name": "run_python",
        "description": "Run Python",
        "parameters": {"type": "object", "properties": {}},
    }
    chat_provider = ProviderConfig(name="p", base_url="https://example.com")
    chat_model = ModelConfig(name="m", provider="p", model="remote", api="chat_completions")
    responses_model = ModelConfig(name="m", provider="p", model="remote", api="responses")

    chat = chat_payload(chat_provider, chat_model, [], [tool], None, stream=False)
    responses = responses_payload(
        chat_provider,
        responses_model,
        [],
        [tool],
        None,
        stream=False,
    )

    assert "tool_choice" not in chat
    assert "tool_choice" not in responses


def test_openai_client_strips_sdk_owned_endpoint_path_and_passes_extra_headers() -> None:
    provider = ProviderConfig(
        name="p",
        base_url="https://api.example.com/v1",
        api_key="test-key",
        headers={"api-key": "test-key"},
        responses=EndpointConfig(path="/responses"),
    )

    client = openai_client(provider, "responses", "/responses")

    assert str(client.base_url) == "https://api.example.com/v1/"
    assert client.auth_headers == {"Authorization": "Bearer test-key"}
    assert client.default_headers["api-key"] == "test-key"


def test_openai_client_uses_sdk_default_missing_credentials_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    provider = ProviderConfig(name="p", base_url="https://api.example.com/v1")

    with pytest.raises(openai.OpenAIError, match="Missing credentials"):
        openai_client(provider, "responses", "/responses")


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


@pytest.mark.asyncio
async def test_stream_responses_allows_empty_events_until_valid_delta() -> None:
    sdk_client = FakeOpenAIClient(
        response_events=[
            {"type": "response.created", "response": {"id": "resp_1"}},
            {"type": "response.output_text.delta", "delta": "done"},
            {"type": "response.completed", "response": {"id": "resp_1", "output": []}},
        ]
    )
    provider = ProviderConfig(name="p", base_url="https://api.example.com/v1")
    model = ModelConfig(name="m", provider="p", model="remote", api="responses")

    events = [
        event
        async for event in stream_responses_response(
            provider=provider,
            model=model,
            input_items=[],
            tools=[],
            instructions=None,
            previous_response_id=None,
            client=sdk_client,
        )
    ]

    assert [event.type for event in events] == ["text_delta", "completed"]
    assert events[0].text == "done"
    completed = events[1]
    assert completed.response is not None
    assert completed.response.output_text == "done"
    assert completed.response.output[0]["content"][0]["text"] == "done"


@pytest.mark.asyncio
async def test_stream_responses_rejects_completed_without_output() -> None:
    sdk_client = FakeOpenAIClient(
        response_events=[
            {"type": "response.created", "response": {"id": "resp_1"}},
            {"type": "response.completed", "response": {"id": "resp_1", "output": []}},
        ]
    )
    provider = ProviderConfig(name="p", base_url="https://api.example.com/v1")
    model = ModelConfig(name="m", provider="p", model="remote", api="responses")

    with pytest.raises(EmptyModelStreamError, match="without returning content"):
        [
            event
            async for event in stream_responses_response(
                provider=provider,
                model=model,
                input_items=[],
                tools=[],
                instructions=None,
                previous_response_id=None,
                client=sdk_client,
            )
        ]


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
    )
    model = ModelConfig(name="m", provider="p", model="claude", api="anthropic_messages")

    payload = anthropic_payload(provider, model, [], [], "system", stream=True)

    assert payload["model"] == "claude"
    assert payload["system"] == "system"
    assert payload["messages"] == []
    assert payload["stream"] is True
    assert "max_tokens" not in payload
    assert "tools" not in payload


def test_anthropic_payload_preserves_configured_max_tokens() -> None:
    provider = ProviderConfig(
        name="p",
        base_url="https://example.com",
        anthropic_messages=EndpointConfig(path="/v1/messages", params={"max_tokens": 99}),
    )
    model = ModelConfig(name="m", provider="p", model="claude", api="anthropic_messages")

    payload = anthropic_payload(provider, model, [], [], None, stream=False)

    assert payload["max_tokens"] == 99


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


def test_anthropic_messages_group_parallel_tool_results_after_tool_uses() -> None:
    messages = anthropic_messages(
        [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {
                "type": "function_call",
                "call_id": "toolu_1",
                "name": "run_python",
                "arguments": "{\"code\":\"print(1)\"}",
            },
            {
                "type": "function_call",
                "call_id": "toolu_2",
                "name": "run_python",
                "arguments": "{\"code\":\"print(2)\"}",
            },
            {"type": "function_call_output", "call_id": "toolu_1", "output": "{\"one\":true}"},
            {"type": "function_call_output", "call_id": "toolu_2", "output": "{\"two\":true}"},
        ]
    )

    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert [block["type"] for block in messages[1]["content"]] == ["tool_use", "tool_use"]
    assert [block["type"] for block in messages[2]["content"]] == ["tool_result", "tool_result"]
    assert [block["tool_use_id"] for block in messages[2]["content"]] == ["toolu_1", "toolu_2"]


def test_parse_anthropic_response_maps_tool_use() -> None:
    response = parse_anthropic_response(
        {
            "id": "msg_1",
            "content": [
                {"type": "thinking", "thinking": "plan", "signature": "sig"},
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
    assert response.reasoning_text == "plan"
    assert response.output[0]["anthropic_content"] == [
        {"type": "thinking", "thinking": "plan", "signature": "sig"},
        {"type": "text", "text": "hello"},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "run_python",
            "input": {"code": "print(1)"},
        },
    ]
    assert response.output[1]["type"] == "function_call"
    assert response.output[1]["call_id"] == "toolu_1"


def test_anthropic_messages_replay_provider_content_without_duplicate_tool_use() -> None:
    messages = anthropic_messages(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
                "anthropic_content": [
                    {"type": "thinking", "thinking": "plan", "signature": "sig"},
                    {"type": "text", "text": "hello"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "run_python",
                        "input": {"code": "print(1)"},
                    },
                ],
            },
            {
                "type": "function_call",
                "call_id": "toolu_1",
                "name": "run_python",
                "arguments": '{"code":"print(1)"}',
            },
        ]
    )

    assert messages == [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan", "signature": "sig"},
                {"type": "text", "text": "hello"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "run_python",
                    "input": {"code": "print(1)"},
                },
            ],
        }
    ]


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


@pytest.mark.asyncio
async def test_stream_anthropic_allows_empty_events_until_valid_delta() -> None:
    class Messages:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs

            class MessageStart:
                type = "message_start"

                class Message:
                    id = "msg_1"
                    usage = {}

                message = Message()

            class EmptyDelta:
                type = "content_block_delta"
                index = 0

                class Delta:
                    type = "text_delta"
                    text = ""

                delta = Delta()

            class TextDelta:
                type = "content_block_delta"
                index = 0

                class Delta:
                    type = "text_delta"
                    text = "done"

                delta = Delta()

            class MessageStop:
                type = "message_stop"

            return FakeAnthropicStream([MessageStart(), EmptyDelta(), TextDelta(), MessageStop()])

    class Client:
        def __init__(self) -> None:
            self.messages = Messages()

    provider = ProviderConfig(name="p", base_url="https://api.anthropic.com")
    model = ModelConfig(name="m", provider="p", model="claude", api="anthropic_messages")
    client = Client()

    events = [
        event
        async for event in stream_anthropic_response(
            provider=provider,
            model=model,
            input_items=[],
            tools=[],
            instructions=None,
            client=client,
        )
    ]

    assert [event.type for event in events] == ["text_delta", "text_delta", "completed"]
    assert events[0].text == ""
    assert events[1].text == "done"
    completed = events[2]
    assert completed.response is not None
    assert completed.response.output_text == "done"


@pytest.mark.asyncio
async def test_stream_anthropic_tool_use_does_not_prefix_empty_start_input() -> None:
    class Messages:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs

            class MessageStart:
                type = "message_start"

                class Message:
                    id = "msg_1"
                    usage = {}

                message = Message()

            class ToolStart:
                type = "content_block_start"
                index = 0

                class ContentBlock:
                    type = "tool_use"
                    id = "toolu_1"
                    name = "run_python"
                    input = {}

                content_block = ContentBlock()

            class ToolInputDelta:
                type = "content_block_delta"
                index = 0

                class Delta:
                    type = "input_json_delta"
                    partial_json = "{\"code\":\"print(1)\"}"

                delta = Delta()

            class MessageStop:
                type = "message_stop"

            return FakeAnthropicStream([MessageStart(), ToolStart(), ToolInputDelta(), MessageStop()])

    class Client:
        def __init__(self) -> None:
            self.messages = Messages()

    provider = ProviderConfig(name="p", base_url="https://api.anthropic.com")
    model = ModelConfig(name="m", provider="p", model="claude", api="anthropic_messages")
    client = Client()

    events = [
        event
        async for event in stream_anthropic_response(
            provider=provider,
            model=model,
            input_items=[],
            tools=[],
            instructions=None,
            client=client,
        )
    ]

    completed = events[-1]
    assert completed.response is not None
    assert completed.response.output == [
        {
            "type": "message",
            "role": "assistant",
            "content": [],
            "anthropic_content": [
                {
                    "type": "tool_use",
                    "name": "run_python",
                    "input": {"code": "print(1)"},
                    "id": "toolu_1",
                }
            ],
        },
        {
            "type": "function_call",
            "call_id": "toolu_1",
            "name": "run_python",
            "arguments": "{\"code\":\"print(1)\"}",
        }
    ]


@pytest.mark.asyncio
async def test_stream_anthropic_preserves_thinking_signature_blocks() -> None:
    class Messages:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs

            class MessageStart:
                type = "message_start"

                class Message:
                    id = "msg_1"
                    usage = {}

                message = Message()

            class ThinkingStart:
                type = "content_block_start"
                index = 0

                class ContentBlock:
                    type = "thinking"
                    thinking = ""

                content_block = ContentBlock()

            class ThinkingDelta:
                type = "content_block_delta"
                index = 0

                class Delta:
                    type = "thinking_delta"
                    thinking = "plan"

                delta = Delta()

            class SignatureDelta:
                type = "content_block_delta"
                index = 0

                class Delta:
                    type = "signature_delta"
                    signature = "sig"

                delta = Delta()

            class TextStart:
                type = "content_block_start"
                index = 1

                class ContentBlock:
                    type = "text"
                    text = ""

                content_block = ContentBlock()

            class TextDelta:
                type = "content_block_delta"
                index = 1

                class Delta:
                    type = "text_delta"
                    text = "hello"

                delta = Delta()

            class MessageStop:
                type = "message_stop"

            return FakeAnthropicStream(
                [MessageStart(), ThinkingStart(), ThinkingDelta(), SignatureDelta(), TextStart(), TextDelta(), MessageStop()]
            )

    class Client:
        def __init__(self) -> None:
            self.messages = Messages()

    provider = ProviderConfig(name="p", base_url="https://api.anthropic.com")
    model = ModelConfig(name="m", provider="p", model="claude", api="anthropic_messages")

    events = [
        event
        async for event in stream_anthropic_response(
            provider=provider,
            model=model,
            input_items=[],
            tools=[],
            instructions=None,
            client=Client(),
        )
    ]

    completed = events[-1]
    assert completed.response is not None
    assert completed.response.output_text == "hello"
    assert completed.response.reasoning_text == "plan"
    assert completed.response.output[0]["anthropic_content"] == [
        {"type": "thinking", "thinking": "plan", "signature": "sig"},
        {"type": "text", "text": "hello"},
    ]


@pytest.mark.asyncio
async def test_stream_anthropic_rejects_message_stop_without_output() -> None:
    class Messages:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs

            class MessageStart:
                type = "message_start"

                class Message:
                    id = "msg_1"
                    usage = {}

                message = Message()

            class MessageStop:
                type = "message_stop"

            return FakeAnthropicStream([MessageStart(), MessageStop()])

    class Client:
        def __init__(self) -> None:
            self.messages = Messages()

    provider = ProviderConfig(name="p", base_url="https://api.anthropic.com")
    model = ModelConfig(name="m", provider="p", model="claude", api="anthropic_messages")
    client = Client()

    with pytest.raises(EmptyModelStreamError, match="without returning content"):
        [
            event
            async for event in stream_anthropic_response(
                provider=provider,
                model=model,
                input_items=[],
                tools=[],
                instructions=None,
                client=client,
            )
        ]


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
async def test_stream_chat_allows_empty_sdk_chunks(
) -> None:
    sdk_client = FakeOpenAIClient(
        chat_events=[
            {"id": "chat_1", "choices": []},
            {"id": "chat_1", "choices": [{"delta": {}}]},
            {"id": "chat_1", "choices": [{"delta": {"content": "done"}}]},
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

    assert [event.type for event in events] == ["text_delta", "completed"]
    assert events[0].text == "done"
    completed = events[1]
    assert completed.response is not None
    assert completed.response.output_text == "done"


@pytest.mark.asyncio
async def test_stream_chat_rejects_stream_with_only_empty_sdk_chunks(
) -> None:
    sdk_client = FakeOpenAIClient(
        chat_events=[
            {"id": "chat_1", "choices": []},
            {"id": "chat_1", "choices": [{"delta": {}}]},
        ]
    )
    provider = ProviderConfig(name="p", base_url="https://example.com")
    model = ModelConfig(name="m", provider="p", model="remote", api="chat_completions")

    with pytest.raises(EmptyModelStreamError, match="without returning content"):
        [
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
