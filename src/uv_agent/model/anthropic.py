from __future__ import annotations

import copy
import json
import re
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from uv_agent.config import ModelConfig, ProviderConfig
from uv_agent.errors import EmptyModelStreamError
from uv_agent.model.sdk import model_param_sources, object_dump, sdk_base_url, sdk_extra_body, sdk_kwargs, sdk_param_keys
from uv_agent.model.types import ModelResponse, ModelStreamEvent


ANTHROPIC_MESSAGES_PATH = "/v1/messages"
EMPTY_ANTHROPIC_STREAM_MESSAGE = (
    "Anthropic messages stream ended without returning content, reasoning, or tool calls"
)
ANTHROPIC_CONTENT_KEY = "anthropic_content"
ANTHROPIC_FALLBACK_BLOCK_FIELDS = (
    "type",
    "text",
    "thinking",
    "signature",
    "data",
    "id",
    "name",
    "input",
    "content",
    "tool_use_id",
    "citations",
    "source",
    "title",
    "url",
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
    content = _anthropic_content_blocks(data.get("content") or [])
    output_text = _anthropic_text_from_content(content)
    output = anthropic_output_items(content, _anthropic_tool_acc_from_content(content), output_text=output_text)
    return ModelResponse(
        id=data.get("id"),
        output=output,
        output_text=output_text,
        raw=data,
        usage=data.get("usage") or {},
        reasoning_text=_anthropic_reasoning_from_content(content),
    )


def anthropic_messages(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_assistant: dict[str, Any] | None = None
    pending_assistant_tool_use_ids: set[str] = set()
    pending_tool_results: list[dict[str, Any]] = []

    def flush_pending_assistant() -> None:
        nonlocal pending_assistant, pending_assistant_tool_use_ids
        if pending_assistant is not None:
            messages.append(pending_assistant)
            pending_assistant = None
            pending_assistant_tool_use_ids = set()

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
                pending_assistant_tool_use_ids = _anthropic_tool_use_ids(message.get("content"))
            else:
                messages.append(message)
        elif item_type == "function_call":
            flush_tool_results()
            call_id = str(item.get("call_id") or "")
            existing_input = _anthropic_tool_input_from_content(
                pending_assistant.get("content") if pending_assistant is not None else None,
                call_id,
            )
            if existing_input is not None:
                item_arguments = _compact_json(item.get("arguments") or "{}")
                existing_arguments = json.dumps(existing_input or {}, ensure_ascii=False)
                if _compact_json(existing_arguments) == item_arguments:
                    continue
            elif call_id and call_id in pending_assistant_tool_use_ids:
                continue
            pending_assistant_content().append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id"),
                    "name": item.get("name"),
                    "input": json.loads(item.get("arguments") or "{}"),
                }
            )
            if call_id:
                pending_assistant_tool_use_ids.add(call_id)
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

    if item.get("role") == "assistant":
        anthropic_content = item.get(ANTHROPIC_CONTENT_KEY)
        if isinstance(anthropic_content, list):
            return _anthropic_content_blocks(anthropic_content)

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
    input_json_acc: dict[int, str] = {}
    content_blocks: dict[int, dict[str, Any]] = {}
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
                block = content_blocks.setdefault(index, {"type": "text", "text": ""})
                block["type"] = "text"
                block["text"] = str(block.get("text") or "") + text
                yield ModelStreamEvent(type="text_delta", text=text)
            elif delta_type in {"thinking_delta", "signature_delta"}:
                reasoning_text = str(getattr(delta, "thinking", "") or "")
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                    block = content_blocks.setdefault(index, {"type": "thinking", "thinking": ""})
                    block["type"] = "thinking"
                    block["thinking"] = str(block.get("thinking") or "") + reasoning_text
                signature = str(getattr(delta, "signature", "") or "")
                if signature:
                    block = content_blocks.setdefault(index, {"type": "thinking", "thinking": ""})
                    block["type"] = "thinking"
                    block["signature"] = str(block.get("signature") or "") + signature
                yield ModelStreamEvent(type="reasoning_delta", text=reasoning_text)
            elif delta_type == "input_json_delta":
                partial_json = getattr(delta, "partial_json", "")
                input_json_acc[index] = input_json_acc.get(index, "") + partial_json
                if index in tool_acc:
                    tool_acc[index]["arguments"] += partial_json
            elif delta_type == "citations_delta":
                citation = _anthropic_delta_citation(delta)
                if citation:
                    block = content_blocks.setdefault(index, {"type": "text", "text": ""})
                    block.setdefault("citations", []).append(citation)
        elif event_type == "content_block_start":
            block = getattr(event, "content_block", None)
            index = int(getattr(event, "index", 0))
            block_data = _anthropic_content_block(block)
            if block_data:
                content_blocks[index] = block_data
            if getattr(block, "type", "") == "tool_use":
                tool_acc[index] = {
                    "call_id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "arguments": input_json_acc.get(index, ""),
                    "_input": object_dump(getattr(block, "input", None)),
                }
        elif event_type == "message_delta":
            usage = object_dump(getattr(event, "usage", None)) or usage
        elif event_type == "message_stop":
            content = _finalize_anthropic_stream_content(content_blocks, tool_acc, input_json_acc)
            output_text = _anthropic_text_from_content(content) or "".join(text_parts)
            if not content and output_text:
                content = [{"type": "text", "text": output_text}]
            output = anthropic_output_items(
                content,
                _finalize_anthropic_stream_tools(tool_acc, input_json_acc),
                output_text=output_text,
            )
            reasoning_text = _anthropic_reasoning_from_content(content) or "".join(reasoning_parts)
            if not output and not reasoning_text:
                raise EmptyModelStreamError(EMPTY_ANTHROPIC_STREAM_MESSAGE)
            yield ModelStreamEvent(
                type="completed",
                response=ModelResponse(
                    id=response_id,
                    output=output,
                    output_text=output_text,
                    raw={"id": response_id, "content": content, "output": output},
                    usage=usage,
                    reasoning_text=reasoning_text,
                ),
            )
            return


def anthropic_output_items(
    content: list[dict[str, Any]],
    tool_acc: dict[int, dict[str, Any]],
    *,
    output_text: str | None = None,
) -> list[dict[str, Any]]:
    """Return internal output while preserving the original Anthropic blocks.

    The ``message`` item carries the provider-native ``anthropic_content`` so a
    later Anthropic request can replay the assistant turn exactly, including
    thinking/signature blocks and any provider-added block types.  Separate
    ``function_call`` items are still emitted for the agent loop, TUI, and tool
    execution bookkeeping.
    """

    text = _anthropic_text_from_content(content) if output_text is None else output_text
    output: list[dict[str, Any]] = []
    if content:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": _anthropic_response_message_content(content, text),
                ANTHROPIC_CONTENT_KEY: copy.deepcopy(content),
            }
        )
    elif text:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )

    for index in sorted(tool_acc):
        call = tool_acc[index]
        output.append(
            {
                "type": "function_call",
                "call_id": call.get("call_id"),
                "name": call.get("name"),
                "arguments": call.get("arguments", ""),
            }
        )
    return output


def _finalize_anthropic_stream_content(
    content_blocks: dict[int, dict[str, Any]],
    tool_acc: dict[int, dict[str, Any]],
    input_json_acc: dict[int, str],
) -> list[dict[str, Any]]:
    for index in set(tool_acc) | set(input_json_acc):
        call = tool_acc.get(index) or {}
        block = content_blocks.setdefault(index, {"type": "tool_use"})
        if block.get("type") not in {"tool_use", "server_tool_use"}:
            continue
        if call.get("call_id"):
            block["id"] = call.get("call_id")
        if call.get("name"):
            block["name"] = call.get("name")
        arguments = str(input_json_acc.get(index) or call.get("arguments") or "")
        if arguments:
            try:
                block["input"] = json.loads(arguments)
            except json.JSONDecodeError:
                block["input"] = block.get("input") or call.get("_input") or {}
        elif "input" not in block and "_input" in call:
            block["input"] = call.get("_input") or {}
    return [_strip_none(copy.deepcopy(content_blocks[index])) for index in sorted(content_blocks)]


def _finalize_anthropic_stream_tools(
    tool_acc: dict[int, dict[str, Any]],
    input_json_acc: dict[int, str],
) -> dict[int, dict[str, Any]]:
    finalized: dict[int, dict[str, Any]] = {}
    for index, call in tool_acc.items():
        arguments = input_json_acc.get(index) or call.get("arguments", "")
        if not arguments and "_input" in call:
            arguments = json.dumps(call.get("_input") or {}, ensure_ascii=False)
        finalized[index] = {
            "call_id": call.get("call_id"),
            "name": call.get("name"),
            "arguments": arguments,
        }
    return finalized


def _anthropic_tool_acc_from_content(content: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    tool_acc: dict[int, dict[str, Any]] = {}
    for index, block in enumerate(content):
        if block.get("type") != "tool_use":
            continue
        tool_acc[index] = {
            "call_id": block.get("id"),
            "name": block.get("name"),
            "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
        }
    return tool_acc


def _anthropic_content_blocks(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    blocks: list[dict[str, Any]] = []
    for block in value:
        block_data = _anthropic_content_block(block)
        if block_data:
            blocks.append(block_data)
    return blocks


def _anthropic_content_block(value: object) -> dict[str, Any]:
    dumped: dict[str, Any] = {}
    if isinstance(value, dict):
        dumped = copy.deepcopy(value)
    else:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                raw = model_dump(mode="json", exclude_none=True)
            except TypeError:
                raw = model_dump(mode="json")
            if isinstance(raw, dict):
                dumped = raw
        if not dumped:
            dumped = object_dump(value)
        if not dumped:
            dumped = _anthropic_content_block_from_attrs(value)
    return _strip_none(dumped)


def _anthropic_content_block_from_attrs(value: object) -> dict[str, Any]:
    dumped: dict[str, Any] = {}
    for field in ANTHROPIC_FALLBACK_BLOCK_FIELDS:
        if not hasattr(value, field):
            continue
        field_value = getattr(value, field)
        if field_value is not None:
            dumped[field] = _json_value(field_value)
    return dumped


def _anthropic_delta_citation(delta: object) -> dict[str, Any] | None:
    citation = getattr(delta, "citation", None)
    if citation is None and isinstance(delta, dict):
        citation = delta.get("citation")
    if citation is None:
        return None
    value = _json_value(citation)
    return value if isinstance(value, dict) else None


def _anthropic_response_message_content(content: list[dict[str, Any]], output_text: str) -> list[dict[str, str]]:
    text_blocks = [str(block.get("text") or "") for block in content if block.get("type") == "text"]
    if text_blocks:
        return [{"type": "output_text", "text": text} for text in text_blocks]
    if output_text:
        return [{"type": "output_text", "text": output_text}]
    return []


def _anthropic_text_from_content(content: list[dict[str, Any]]) -> str:
    return "".join(str(block.get("text") or "") for block in content if block.get("type") == "text")


def _anthropic_reasoning_from_content(content: list[dict[str, Any]]) -> str:
    return "".join(str(block.get("thinking") or "") for block in content if block.get("type") == "thinking")


def _anthropic_tool_use_ids(content: object) -> set[str]:
    if not isinstance(content, list):
        return set()
    return {
        str(block.get("id") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
    }


def _anthropic_tool_input_from_content(content: object, call_id: str) -> Any:
    if not isinstance(content, list) or not call_id:
        return None
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and str(block.get("id") or "") == call_id:
            return block.get("input")
    return None


def _compact_json(value: object) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "{}")
        except json.JSONDecodeError:
            return value
    else:
        parsed = value
    return json.dumps(parsed or {}, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json", exclude_none=True)
        except TypeError:
            dumped = model_dump(mode="json")
        return _strip_none(dumped)
    dumped = object_dump(value)
    if dumped:
        return _strip_none(dumped)
    return str(value)


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_strip_none(item) for item in value]
    return value


def parse_anthropic_message(message: Any) -> ModelResponse:
    return parse_anthropic_response(object_dump(message))
