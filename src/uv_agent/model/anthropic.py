from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any, Callable

from uv_agent.config import ModelConfig, ProviderConfig
from uv_agent.model.content import chat_output_items
from uv_agent.model.http import post_json, stream_sse
from uv_agent.model.types import ModelResponse, ModelStreamEvent


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


async def create_anthropic_response(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
) -> ModelResponse:
    payload = anthropic_payload(provider, model, input_items, tools, instructions, stream=False)
    data = await post_json(provider, model.api, payload)
    return parse_anthropic_response(data)


async def stream_anthropic_response(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    stream_events: Callable[[ProviderConfig, str, dict[str, Any]], AsyncIterator[dict[str, Any]]] = stream_sse,
) -> AsyncIterator[ModelStreamEvent]:
    payload = anthropic_payload(provider, model, input_items, tools, instructions, stream=True)
    text_parts: list[str] = []
    tool_acc: dict[int, dict[str, Any]] = {}
    response_id: str | None = None
    usage: dict[str, Any] = {}
    async for data in stream_events(provider, model.api, payload):
        event_type = data.get("type", "")
        if data.get("message", {}).get("id"):
            response_id = data["message"]["id"]
        if data.get("message", {}).get("usage"):
            usage = data["message"]["usage"]
        if event_type == "content_block_delta":
            index = int(data.get("index", 0))
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                text_parts.append(text)
                yield ModelStreamEvent(type="text_delta", text=text)
            elif delta.get("type") in {"thinking_delta", "signature_delta"}:
                yield ModelStreamEvent(type="reasoning_delta", text=str(delta.get("thinking") or ""))
            elif delta.get("type") == "input_json_delta":
                existing = tool_acc.setdefault(index, {"arguments": ""})
                existing["arguments"] += delta.get("partial_json", "")
        elif event_type == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                index = int(data.get("index", 0))
                tool_acc[index] = {
                    "call_id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                }
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
