from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI
from openai.resources.responses import AsyncResponses

from uv_agent.config import ModelConfig, ProviderConfig
from uv_agent.model.content import extract_responses_text
from uv_agent.model.openai_sdk import openai_client
from uv_agent.model.sdk import model_param_sources, object_dump, sdk_kwargs, sdk_param_keys
from uv_agent.model.types import ModelResponse, ModelStreamEvent

RESPONSES_PATH = "/responses"
RESPONSES_SDK_PARAM_KEYS = sdk_param_keys(AsyncResponses.create)


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
        "tool_choice": "auto" if tools else "none",
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
        RESPONSES_SDK_PARAM_KEYS,
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
    client: AsyncOpenAI | None = None,
) -> ModelResponse:
    client = client or openai_client(provider, model.api, RESPONSES_PATH)
    response = await client.responses.create(
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
    client: AsyncOpenAI | None = None,
) -> AsyncIterator[ModelStreamEvent]:
    client = client or openai_client(provider, model.api, RESPONSES_PATH)
    stream = await client.responses.create(
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
    output: list[dict[str, Any]] = []
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
            yield ModelStreamEvent(type="reasoning_delta", text=str(data.get("delta") or ""))
        elif event_type == "response.output_item.done":
            item = data.get("item")
            if isinstance(item, dict):
                output.append(item)
        elif event_type == "response.completed":
            response_data = data.get("response") or {}
            response = parse_responses_response(response_data)
            if not response.output and output:
                response = ModelResponse(
                    id=response.id,
                    output=output,
                    output_text=response.output_text or "".join(text_parts),
                    raw=response.raw,
                    usage=response.usage,
                )
            yield ModelStreamEvent(type="completed", response=response)
            return
