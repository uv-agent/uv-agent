from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI
from openai.resources.chat.completions import AsyncCompletions

from uv_agent.config import ModelConfig, ProviderConfig
from uv_agent.errors import EmptyModelStreamError
from uv_agent.model.content import (
    chat_message_passthrough,
    chat_message_reasoning_text,
    chat_messages,
    chat_output_items,
    chat_tool,
    chat_tool_acc_from_message,
)
from uv_agent.model.openai_sdk import openai_client
from uv_agent.model.sdk import model_param_sources, object_dump, sdk_kwargs, sdk_param_keys
from uv_agent.model.types import ModelResponse, ModelStreamEvent, ToolCallDelta

CHAT_COMPLETIONS_PATH = "/chat/completions"
CHAT_DELTA_CONTROL_FIELDS = {
    "role",
    "content",
    "tool_calls",
    "function_call",
    "refusal",
}
CHAT_COMPLETIONS_SDK_PARAM_KEYS = sdk_param_keys(AsyncCompletions.create)
EMPTY_CHAT_COMPLETIONS_STREAM_MESSAGE = (
    "Chat completions stream ended without returning content, reasoning, or tool calls"
)


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


def chat_create_kwargs(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
) -> dict[str, Any]:
    payload = chat_payload(provider, model, input_items, tools, instructions, stream=False)
    payload.pop("stream", None)
    payload.pop("stream_options", None)
    return sdk_kwargs(
        payload,
        model_param_sources(provider, model, "chat_completions"),
        CHAT_COMPLETIONS_SDK_PARAM_KEYS,
    )


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
    client: AsyncOpenAI | None = None,
) -> ModelResponse:
    client = client or openai_client(provider, model.api, CHAT_COMPLETIONS_PATH)
    response = await client.chat.completions.create(
        **chat_create_kwargs(
            provider=provider,
            model=model,
            input_items=input_items,
            tools=tools,
            instructions=instructions,
        )
    )
    return parse_chat_response_for_model(object_dump(response), model)


async def stream_chat_response(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    client: AsyncOpenAI | None = None,
) -> AsyncIterator[ModelStreamEvent]:
    client = client or openai_client(provider, model.api, CHAT_COMPLETIONS_PATH)
    payload = chat_payload(provider, model, input_items, tools, instructions, stream=True)
    payload_kwargs = sdk_kwargs(
        payload,
        model_param_sources(provider, model, "chat_completions"),
        CHAT_COMPLETIONS_SDK_PARAM_KEYS,
    )
    stream = await client.chat.completions.create(**payload_kwargs)
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    passthrough_acc: dict[str, str] = {}
    tool_acc: dict[int, dict[str, Any]] = {}
    response_id: str | None = None
    usage: dict[str, Any] = {}
    passthrough_fields = set(model.message_passthrough.fields_for_role("assistant"))
    reasoning_fields = set(model.reasoning_display.stream_delta_fields)
    async for event in stream:
        data = object_dump(event)
        response_id = data.get("id") or response_id
        if data.get("usage"):
            usage = data["usage"]
        for choice in data.get("choices", []):
            delta = choice.get("delta") or {}
            reasoning_fields_seen: set[str] = set()
            if delta.get("content"):
                text = delta["content"]
                text_parts.append(text)
                yield ModelStreamEvent(type="text_delta", text=text)
            for field in passthrough_fields:
                value = delta.get(field)
                if isinstance(value, str) and value:
                    passthrough_acc[field] = passthrough_acc.get(field, "") + value
            for field in reasoning_fields:
                value = delta.get(field)
                if isinstance(value, str) and value:
                    reasoning_fields_seen.add(field)
                    reasoning_parts.append(value)
                    yield ModelStreamEvent(type="reasoning_delta", text=value)
            if model.reasoning_display.unknown_text_delta_as_reasoning:
                for field, value in delta.items():
                    if field in reasoning_fields_seen or field in CHAT_DELTA_CONTROL_FIELDS:
                        continue
                    if isinstance(value, str) and value:
                        reasoning_parts.append(value)
                        yield ModelStreamEvent(type="reasoning_delta", text=value)
            for tool_call in delta.get("tool_calls") or []:
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
    output_text = "".join(text_parts)
    reasoning_text = "".join(reasoning_parts)
    output = chat_output_items(output_text, tool_acc, passthrough=passthrough_acc)
    if not output and not reasoning_text:
        raise EmptyModelStreamError(EMPTY_CHAT_COMPLETIONS_STREAM_MESSAGE)
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
