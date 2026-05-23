from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from uv_agent.config import ModelConfig, ProviderConfig
from uv_agent.errors import EmptyModelStreamError
from uv_agent.model.content import chat_output_items
from uv_agent.model.sdk import model_param_sources, object_dump, sdk_base_url, sdk_extra_body, sdk_kwargs, sdk_param_keys
from uv_agent.model.types import ModelResponse, ModelStreamEvent


ANTHROPIC_MESSAGES_PATH = "/v1/messages"
EMPTY_ANTHROPIC_STREAM_MESSAGE = (
    "Anthropic messages stream ended without returning content, reasoning, or tool calls"
)


@lru_cache(maxsize=1)
def anthropic_sdk_param_keys() -> set[str]:
    """Return Anthropic SDK parameter names, importing the SDK lazily."""

    from anthropic.resources.messages import AsyncMessages

    return sdk_param_keys(AsyncMessages.create)


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
    pending_assistant: dict[str, Any] | None = None
    pending_tool_results: list[dict[str, Any]] = []

    def flush_pending_assistant() -> None:
        nonlocal pending_assistant
        if pending_assistant is not None:
            messages.append(pending_assistant)
            pending_assistant = None

    def flush_tool_results() -> None:
        nonlocal pending_tool_results
        if pending_tool_results:
            # Anthropic requires all tool_result blocks for one assistant
            # tool_use batch to be in the immediately following user message.
            # Emitting one user message per result makes the second result no
            # longer directly follow the assistant tool_use message.
            messages.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    def pending_assistant_content() -> list[dict[str, Any]]:
        nonlocal pending_assistant
        if pending_assistant is None:
            pending_assistant = {"role": "assistant", "content": []}
        content = pending_assistant.get("content")
        if isinstance(content, list):
            return content
        blocks = [{"type": "text", "text": str(content)}] if content else []
        pending_assistant["content"] = blocks
        return blocks

    for item in input_items:
        item_type = item.get("type")
        if item_type == "message":
            flush_pending_assistant()
            flush_tool_results()
            role = item.get("role", "user")
            message = {
                "role": "assistant" if role == "assistant" else "user",
                "content": anthropic_message_content(item),
            }
            if role == "assistant":
                # Keep plain assistant text as a string unless a following
                # function_call needs to append tool_use blocks to the same
                # assistant message.
                pending_assistant = message
            else:
                messages.append(message)
        elif item_type == "function_call":
            flush_tool_results()
            pending_assistant_content().append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id"),
                    "name": item.get("name"),
                    "input": json.loads(item.get("arguments") or "{}"),
                }
            )
        elif item_type == "function_call_output":
            flush_pending_assistant()
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": item.get("call_id"),
                    "content": item.get("output", ""),
                }
            )
    flush_pending_assistant()
    flush_tool_results()
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


def anthropic_client(provider: ProviderConfig) -> Any:
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(
        api_key=provider.resolved_api_key(),
        base_url=anthropic_sdk_base_url(provider),
        default_headers=provider.headers or None,
    )


def anthropic_sdk_base_url(provider: ProviderConfig) -> str:
    return sdk_base_url(provider, "anthropic_messages", ANTHROPIC_MESSAGES_PATH)


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
    return sdk_kwargs(
        payload,
        model_param_sources(provider, model, "anthropic_messages"),
        anthropic_sdk_param_keys(),
    )


def endpoint_extra_body(provider: ProviderConfig, model: ModelConfig) -> dict[str, Any] | None:
    return sdk_extra_body(
        model_param_sources(provider, model, "anthropic_messages"),
        anthropic_sdk_param_keys(),
    )


async def create_anthropic_response(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    client: Any | None = None,
) -> ModelResponse:
    sdk_client: Any = anthropic_client(provider) if client is None else client
    response = await sdk_client.messages.create(
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
    client: Any | None = None,
) -> AsyncIterator[ModelStreamEvent]:
    sdk_client: Any = anthropic_client(provider) if client is None else client
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_acc: dict[int, dict[str, Any]] = {}
    response_id: str | None = None
    usage: dict[str, Any] = {}
    stream = await sdk_client.messages.create(
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
                reasoning_text = str(getattr(delta, "thinking", "") or "")
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                yield ModelStreamEvent(type="reasoning_delta", text=reasoning_text)
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
                    "arguments": "",
                    "_input": object_dump(getattr(block, "input", None)),
                }
        elif event_type == "message_delta":
            usage = object_dump(getattr(event, "usage", None)) or usage
        elif event_type == "message_stop":
            output = chat_output_items("".join(text_parts), _finalize_anthropic_stream_tools(tool_acc))
            reasoning_text = "".join(reasoning_parts)
            if not output and not reasoning_text:
                raise EmptyModelStreamError(EMPTY_ANTHROPIC_STREAM_MESSAGE)
            yield ModelStreamEvent(
                type="completed",
                response=ModelResponse(
                    id=response_id,
                    output=output,
                    output_text="".join(text_parts),
                    raw={"id": response_id, "output": output},
                    usage=usage,
                    reasoning_text=reasoning_text,
                ),
            )
            return


def _finalize_anthropic_stream_tools(tool_acc: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    finalized: dict[int, dict[str, Any]] = {}
    for index, call in tool_acc.items():
        arguments = call.get("arguments", "")
        if not arguments and "_input" in call:
            arguments = json.dumps(call.get("_input") or {}, ensure_ascii=False)
        finalized[index] = {
            "call_id": call.get("call_id"),
            "name": call.get("name"),
            "arguments": arguments,
        }
    return finalized


def parse_anthropic_message(message: Any) -> ModelResponse:
    return parse_anthropic_response(object_dump(message))
