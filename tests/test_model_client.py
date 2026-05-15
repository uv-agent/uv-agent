from __future__ import annotations

from uv_agent.config import EndpointConfig, ModelConfig, ProviderConfig
from uv_agent.model_client import (
    chat_messages,
    chat_payload,
    parse_chat_response,
    parse_sse_event,
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


def test_parse_sse_event_handles_done_and_json() -> None:
    assert parse_sse_event(None, ["[DONE]"]) is None
    parsed = parse_sse_event("response.completed", ['{"response":{"id":"x"}}'])
    assert parsed == {"type": "response.completed", "response": {"id": "x"}}
