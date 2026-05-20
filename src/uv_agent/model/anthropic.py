from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Message

from uv_agent.config import ModelConfig, ProviderConfig
from uv_agent.model.content import chat_output_items
from uv_agent.model.types import ModelResponse, ModelStreamEvent


ANTHROPIC_MESSAGES_PATH = "/v1/messages"
ANTHROPIC_SDK_PARAM_KEYS = {
    "cache_control",
    "container",
    "extra_headers",
    "extra_query",
    "inference_geo",
    "max_tokens",
    "messages",
    "metadata",
    "model",
    "output_config",
    "service_tier",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "timeout",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
}


def anthropic_payload(
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    *,
    stream: bool,
) -> dict[str, Any]:
    endpoint = provider.endpoint_for_api("anthropic_messages")
    payload: dict[str, Any] = {
        "model": model.model,
        "messages": anthropic_messages(input_items),
        "tools": [anthropic_tool(tool) for tool in tools],
        "max_tokens": 4096,
        **provider.params,
        **endpoint.params,
        **model.params,
    }
    if instructions:
        payload["system"] = instructions
    if not payload["tools"]:
        payload.pop("tools")
    if stream:
        payload["stream"] = True
    return payload


def parse_anthropic_response(data: dict[str, Any]) -> ModelResponse:
    text_parts: list[str] = []
    tool_acc: dict[int, dict[str, Any]] = {}
    for index, block in enumerate(data.get("content") or []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_acc[index] = {
                "call_id": block.get("id"),
                "name": block.get("name"),
                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
            }
    output_text = "".join(text_parts)
    output = chat_output_items(output_text, tool_acc)
    return ModelResponse(
        id=data.get("id"),
        output=output,
        output_text=output_text,
        raw=data,
        usage=data.get("usage") or {},
    )


def anthropic_messages(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in input_items:
        item_type = item.get("type")
        if item_type == "message":
            role = item.get("role", "user")
            content = anthropic_message_content(item)
            messages.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": content,
                }
            )
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": item.get("call_id"),
                            "name": item.get("name"),
                            "input": json.loads(item.get("arguments") or "{}"),
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": item.get("call_id"),
                            "content": item.get("output", ""),
                        }
                    ],
                }
            )
    return messages


def anthropic_message_content(item: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Return Anthropic content, preserving base64 data URL image parts."""
    parts: list[dict[str, Any]] = []
    has_image = False
    for content in item.get("content") or []:
        content_type = content.get("type")
        if content_type in {"input_text", "output_text", "text", "refusal"}:
            parts.append({"type": "text", "text": content.get("text", "")})
        elif content_type == "input_image":
            image_url = content.get("image_url") or content.get("url")
            source = anthropic_image_source(str(image_url or ""))
            if source:
                has_image = True
                parts.append({"type": "image", "source": source})
    if has_image:
        return parts
    return "\n".join(part.get("text", "") for part in parts)


def anthropic_image_source(data_url: str) -> dict[str, str] | None:
    prefix = "data:"
    marker = ";base64,"
    if not data_url.startswith(prefix) or marker not in data_url:
        return None
    media_type, data = data_url[len(prefix) :].split(marker, 1)
    data = re.sub(r"\s+", "", data)
    return {
        "type": "base64",
        "media_type": media_type,
        "data": data,
    }


def anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
    }


def anthropic_client(provider: ProviderConfig) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key=provider.resolved_api_key(),
        base_url=anthropic_sdk_base_url(provider),
        default_headers=provider.headers or None,
    )


def anthropic_sdk_base_url(provider: ProviderConfig) -> str:
    endpoint_path = provider.endpoint_for_api("anthropic_messages").path
    messages_url = provider.base_url.rstrip("/") + endpoint_path
    if messages_url.endswith(ANTHROPIC_MESSAGES_PATH):
        return messages_url[: -len(ANTHROPIC_MESSAGES_PATH)] or provider.base_url.rstrip("/")
    return provider.base_url


def anthropic_create_kwargs(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
) -> dict[str, Any]:
    payload = anthropic_payload(provider, model, input_items, tools, instructions, stream=False)
    payload.pop("stream", None)
    kwargs = {key: value for key, value in payload.items() if key in ANTHROPIC_SDK_PARAM_KEYS}
    extra_body = endpoint_extra_body(provider, model)
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def endpoint_extra_body(provider: ProviderConfig, model: ModelConfig) -> dict[str, Any] | None:
    endpoint = provider.endpoint_for_api("anthropic_messages")
    extra = {
        key: value
        for source in (provider.params, endpoint.params, model.params)
        for key, value in source.items()
        if key not in ANTHROPIC_SDK_PARAM_KEYS
    }
    return extra or None


async def create_anthropic_response(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    client: AsyncAnthropic | None = None,
) -> ModelResponse:
    client = client or anthropic_client(provider)
    response = await client.messages.create(
        **anthropic_create_kwargs(
            provider=provider,
            model=model,
            input_items=input_items,
            tools=tools,
            instructions=instructions,
        )
    )
    return parse_anthropic_message(response)


async def stream_anthropic_response(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    client: AsyncAnthropic | None = None,
) -> AsyncIterator[ModelStreamEvent]:
    client = client or anthropic_client(provider)
    text_parts: list[str] = []
    tool_acc: dict[int, dict[str, Any]] = {}
    response_id: str | None = None
    usage: dict[str, Any] = {}
    stream = await client.messages.create(
        stream=True,
        **anthropic_create_kwargs(
            provider=provider,
            model=model,
            input_items=input_items,
            tools=tools,
            instructions=instructions,
        ),
    )
    async for event in stream:
        event_type = getattr(event, "type", "")
        message = getattr(event, "message", None)
        if message is not None:
            response_id = getattr(message, "id", None) or response_id
            usage = object_dump(getattr(message, "usage", None)) or usage
        if event_type == "message_start":
            continue
        if event_type == "content_block_delta":
            index = int(getattr(event, "index", 0))
            delta = getattr(event, "delta", None)
            delta_type = getattr(delta, "type", "")
            if delta_type == "text_delta":
                text = getattr(delta, "text", "")
                text_parts.append(text)
                yield ModelStreamEvent(type="text_delta", text=text)
            elif delta_type in {"thinking_delta", "signature_delta"}:
                yield ModelStreamEvent(type="reasoning_delta", text=str(getattr(delta, "thinking", "") or ""))
            elif delta_type == "input_json_delta":
                existing = tool_acc.setdefault(index, {"arguments": ""})
                existing["arguments"] += getattr(delta, "partial_json", "")
        elif event_type == "content_block_start":
            block = getattr(event, "content_block", None)
            if getattr(block, "type", "") == "tool_use":
                index = int(getattr(event, "index", 0))
                tool_acc[index] = {
                    "call_id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "arguments": json.dumps(object_dump(getattr(block, "input", None)) or {}, ensure_ascii=False),
                }
        elif event_type == "message_delta":
            usage = object_dump(getattr(event, "usage", None)) or usage
        elif event_type == "message_stop":
            output = chat_output_items("".join(text_parts), tool_acc)
            yield ModelStreamEvent(
                type="completed",
                response=ModelResponse(
                    id=response_id,
                    output=output,
                    output_text="".join(text_parts),
                    raw={"id": response_id, "output": output},
                    usage=usage,
                ),
            )
            return


def parse_anthropic_message(message: Message) -> ModelResponse:
    return parse_anthropic_response(object_dump(message))


def object_dump(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return dict(value) if hasattr(value, "__iter__") else {}
