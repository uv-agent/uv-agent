from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from uv_agent.config import ModelConfig, ProviderConfig
from uv_agent.errors import EmptyModelStreamError
from uv_agent.model.content import extract_responses_text
from uv_agent.model.sdk import model_param_sources, object_dump, sdk_kwargs, sdk_param_keys
from uv_agent.model.types import ModelResponse, ModelStreamEvent, ToolCallDelta

RESPONSES_PATH = "/responses"
EMPTY_RESPONSES_STREAM_MESSAGE = (
    "Responses stream ended without returning content, reasoning, or tool calls"
)


@lru_cache(maxsize=1)
def responses_sdk_param_keys() -> set[str]:
    """Return OpenAI Responses SDK parameter names, importing the SDK lazily."""

    from openai.resources.responses import AsyncResponses

    return sdk_param_keys(AsyncResponses.create)


def responses_payload(
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    *,
    stream: bool,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    endpoint = provider.endpoint_for_api("responses")
    payload: dict[str, Any] = {
        "model": model.model,
        "input": copy.deepcopy(input_items),
        "tools": copy.deepcopy(tools),
        **provider.params,
        **endpoint.params,
        **model.params,
    }
    if instructions:
        payload["instructions"] = instructions
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    if stream:
        payload["stream"] = True
    return payload


def responses_create_kwargs(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    previous_response_id: str | None,
) -> dict[str, Any]:
    payload = responses_payload(
        provider,
        model,
        input_items,
        tools,
        instructions,
        stream=False,
        previous_response_id=previous_response_id,
    )
    payload.pop("stream", None)
    return sdk_kwargs(
        payload,
        model_param_sources(provider, model, "responses"),
        responses_sdk_param_keys(),
    )


def parse_responses_response(data: dict[str, Any]) -> ModelResponse:
    output = data.get("output") or []
    output_text = data.get("output_text") or extract_responses_text(output)
    return ModelResponse(
        id=data.get("id"),
        output=output,
        output_text=output_text,
        raw=data,
        usage=data.get("usage") or {},
    )


async def create_responses_response(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    previous_response_id: str | None,
    client: Any | None = None,
) -> ModelResponse:
    from uv_agent.model.openai_sdk import openai_client

    sdk_client: Any = openai_client(provider, model.api, RESPONSES_PATH) if client is None else client
    response = await sdk_client.responses.create(
        **responses_create_kwargs(
            provider=provider,
            model=model,
            input_items=input_items,
            tools=tools,
            instructions=instructions,
            previous_response_id=previous_response_id,
        )
    )
    return parse_responses_response(object_dump(response))


async def stream_responses_response(
    *,
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    previous_response_id: str | None,
    client: Any | None = None,
) -> AsyncIterator[ModelStreamEvent]:
    from uv_agent.model.openai_sdk import openai_client

    sdk_client: Any = openai_client(provider, model.api, RESPONSES_PATH) if client is None else client
    stream = await sdk_client.responses.create(
        stream=True,
        **responses_create_kwargs(
            provider=provider,
            model=model,
            input_items=input_items,
            tools=tools,
            instructions=instructions,
            previous_response_id=previous_response_id,
        ),
    )
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    output: list[dict[str, Any]] = []
    tool_acc: dict[int, dict[str, Any]] = {}
    async for event in stream:
        data = object_dump(event)
        event_type = data.get("type", "")
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            delta = data.get("delta", "")
            text_parts.append(delta)
            yield ModelStreamEvent(type="text_delta", text=delta)
        elif event_type in {
            "response.reasoning_text.delta",
            "response.output_item.reasoning.delta",
            "response.reasoning_summary_text.delta",
        }:
            reasoning_text = str(data.get("delta") or "")
            if reasoning_text:
                reasoning_parts.append(reasoning_text)
            yield ModelStreamEvent(type="reasoning_delta", text=reasoning_text)
        elif event_type == "response.output_item.done":
            item = data.get("item")
            if isinstance(item, dict):
                output.append(item)
                if item.get("type") == "function_call":
                    index = _response_output_index(data, len(tool_acc))
                    existing = tool_acc.setdefault(index, {"arguments": ""})
                    previous_arguments = str(existing.get("arguments") or "")
                    if item.get("call_id"):
                        existing["call_id"] = item.get("call_id")
                    if item.get("name"):
                        existing["name"] = item.get("name")
                    if item.get("arguments"):
                        existing["arguments"] = item.get("arguments")
                    final_arguments = str(existing.get("arguments") or "")
                    if not previous_arguments or final_arguments.startswith(previous_arguments):
                        arguments_delta = final_arguments[len(previous_arguments):]
                    else:
                        arguments_delta = final_arguments
                    if existing.get("name") or arguments_delta:
                        yield ModelStreamEvent(
                            type="tool_call_delta",
                            tool_call=ToolCallDelta(
                                index=index,
                                call_id=existing.get("call_id"),
                                name=existing.get("name"),
                                arguments=final_arguments,
                                arguments_delta=arguments_delta,
                            ),
                        )
        elif event_type in {"response.output_item.added", "response.output_item.created"}:
            item = data.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                index = _response_output_index(data, len(tool_acc))
                existing = tool_acc.setdefault(index, {"arguments": ""})
                if item.get("call_id"):
                    existing["call_id"] = item.get("call_id")
                if item.get("name"):
                    existing["name"] = item.get("name")
                yield ModelStreamEvent(
                    type="tool_call_delta",
                    tool_call=ToolCallDelta(
                        index=index,
                        call_id=existing.get("call_id"),
                        name=existing.get("name"),
                        arguments=existing.get("arguments", ""),
                        arguments_delta="",
                    ),
                )
        elif event_type == "response.function_call_arguments.delta":
            index = _response_output_index(data, len(tool_acc))
            existing = tool_acc.setdefault(index, {"arguments": ""})
            delta = str(data.get("delta") or "")
            if delta:
                existing["arguments"] = str(existing.get("arguments") or "") + delta
            yield ModelStreamEvent(
                type="tool_call_delta",
                tool_call=ToolCallDelta(
                    index=index,
                    call_id=existing.get("call_id"),
                    name=existing.get("name"),
                    arguments=existing.get("arguments", ""),
                    arguments_delta=delta,
                ),
            )
        elif event_type == "response.function_call_arguments.done":
            index = _response_output_index(data, len(tool_acc))
            existing = tool_acc.setdefault(index, {"arguments": ""})
            arguments = data.get("arguments")
            if isinstance(arguments, str):
                existing["arguments"] = arguments
        elif event_type == "response.completed":
            response_data = data.get("response") or {}
            response = parse_responses_response(response_data)
            output_text = response.output_text or "".join(text_parts)
            final_output = response.output
            if not final_output and output:
                final_output = output
            if not final_output and output_text:
                final_output = [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": output_text}],
                    }
                ]
            reasoning_text = "".join(reasoning_parts)
            if not final_output and not reasoning_text:
                raise EmptyModelStreamError(EMPTY_RESPONSES_STREAM_MESSAGE)
            if final_output != response.output or output_text != response.output_text or reasoning_text:
                response = ModelResponse(
                    id=response.id,
                    output=final_output,
                    output_text=output_text,
                    raw=response.raw,
                    usage=response.usage,
                    reasoning_text=reasoning_text,
                )
            yield ModelStreamEvent(type="completed", response=response)
            return


def _response_output_index(data: dict[str, Any], fallback: int = 0) -> int:
    for key in ("output_index", "item_index", "index"):
        value = data.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return fallback
