from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Callable

from uv_agent.config import ModelConfig, ProviderConfig
from uv_agent.model.content import (
    chat_message_passthrough,
    chat_message_reasoning_text,
    chat_messages,
    chat_output_items,
    chat_tool,
    chat_tool_acc_from_message,
)
from uv_agent.model.http import SSE_DONE, post_json, stream_sse
from uv_agent.model.types import ModelResponse, ModelStreamEvent, ToolCallDelta

CHAT_DELTA_CONTROL_FIELDS = {
    "role",
    "content",
    "tool_calls",
    "function_call",
    "refusal",
}


def chat_payload(
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    *,
    stream: bool,
) -> dict[str, Any]:
    endpoint = provider.endpoint_for_api("chat_completions")
    payload: dict[str, Any] = {
        "model": model.model,
        "messages": chat_messages(input_items, instructions, model),
        "tools": [chat_tool(tool) for tool in tools],
        "tool_choice": "auto" if tools else "none",
        **provider.params,
        **endpoint.params,
        **model.params,
    }
    if not payload["tools"]:
        payload.pop("tools")
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def parse_chat_response(data: dict[str, Any]) -> ModelResponse:
    return parse_chat_response_for_model(data, None)


def parse_chat_response_for_model(data: dict[str, Any], model: ModelConfig | None) -> ModelResponse:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    passthrough = chat_message_passthrough(message, model) if model is not None else {}
    reasoning_text = chat_message_reasoning_text(message, model) if model is not None else ""
    output = chat_output_items(text, chat_tool_acc_from_message(message), passthrough=passthrough)
    return ModelResponse(
        id=data.get("id"),
        output=output,
        output_text=text,
        raw=data,
        usage=data.get("usage") or {},
        reasoning_text=reasoning_text,
    )


async def create_chat_response(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
) -> ModelResponse:
    payload = chat_payload(provider, model, input_items, tools, instructions, stream=False)
    data = await post_json(provider, model.api, payload)
    return parse_chat_response_for_model(data, model)


async def stream_chat_response(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    stream_events: Callable[[ProviderConfig, str, dict[str, Any]], AsyncIterator[dict[str, Any]]] = stream_sse,
) -> AsyncIterator[ModelStreamEvent]:
    payload = chat_payload(provider, model, input_items, tools, instructions, stream=True)
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    passthrough_acc: dict[str, str] = {}
    tool_acc: dict[int, dict[str, Any]] = {}
    response_id: str | None = None
    usage: dict[str, Any] = {}
    done = False
    saw_payload = False
    passthrough_fields = set(model.message_passthrough.fields_for_role("assistant"))
    reasoning_fields = set(model.reasoning_display.stream_delta_fields)
    async for data in stream_events(provider, model.api, payload):
        if data.get("type") == SSE_DONE:
            done = True
            break
        if not data:
            continue
        response_id = data.get("id") or response_id
        if data.get("usage"):
            usage = data["usage"]
        for choice in data.get("choices", []):
            delta = choice.get("delta") or {}
            reasoning_fields_seen: set[str] = set()
            if delta.get("content"):
                text = delta["content"]
                text_parts.append(text)
                saw_payload = True
                yield ModelStreamEvent(type="text_delta", text=text)
            for field in passthrough_fields:
                value = delta.get(field)
                if isinstance(value, str) and value:
                    passthrough_acc[field] = passthrough_acc.get(field, "") + value
                    saw_payload = True
            for field in reasoning_fields:
                value = delta.get(field)
                if isinstance(value, str) and value:
                    reasoning_fields_seen.add(field)
                    reasoning_parts.append(value)
                    saw_payload = True
                    yield ModelStreamEvent(type="reasoning_delta", text=value)
            if model.reasoning_display.unknown_text_delta_as_reasoning:
                for field, value in delta.items():
                    if field in reasoning_fields_seen or field in CHAT_DELTA_CONTROL_FIELDS:
                        continue
                    if isinstance(value, str) and value:
                        reasoning_parts.append(value)
                        saw_payload = True
                        yield ModelStreamEvent(type="reasoning_delta", text=value)
            for tool_call in delta.get("tool_calls") or []:
                saw_payload = True
                index = int(tool_call.get("index", 0))
                existing = tool_acc.setdefault(index, {"arguments": ""})
                if tool_call.get("id"):
                    existing["call_id"] = tool_call["id"]
                function = tool_call.get("function") or {}
                if function.get("name"):
                    existing["name"] = function["name"]
                if function.get("arguments"):
                    existing["arguments"] += function["arguments"]
                yield ModelStreamEvent(
                    type="tool_call_delta",
                    tool_call=ToolCallDelta(
                        index=index,
                        call_id=existing.get("call_id"),
                        name=existing.get("name"),
                        arguments=existing.get("arguments", ""),
                        arguments_delta=function.get("arguments", ""),
                    ),
                )
    if not done and not saw_payload:
        raise RuntimeError("Chat completions stream ended before [DONE] without returning content")

    output_text = "".join(text_parts)
    reasoning_text = "".join(reasoning_parts)
    output = chat_output_items(output_text, tool_acc, passthrough=passthrough_acc)
    yield ModelStreamEvent(
        type="completed",
        response=ModelResponse(
            id=response_id,
            output=output,
            output_text=output_text,
            raw={"id": response_id, "output": output},
            usage=usage,
            reasoning_text=reasoning_text,
        ),
    )
