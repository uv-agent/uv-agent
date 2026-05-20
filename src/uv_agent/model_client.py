from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from typing import Any

from uv_agent.config import AppConfig
from uv_agent.model.anthropic import (
    anthropic_image_source,
    anthropic_message_content,
    anthropic_messages,
    anthropic_payload,
    anthropic_tool,
    create_anthropic_response,
    parse_anthropic_response,
    stream_anthropic_response,
)
from uv_agent.model.chat import (
    CHAT_DELTA_CONTROL_FIELDS,
    chat_payload,
    create_chat_response,
    parse_chat_response,
    parse_chat_response_for_model,
    stream_chat_response,
)
from uv_agent.model.content import (
    chat_message_content,
    chat_message_passthrough,
    chat_message_reasoning_text,
    chat_messages,
    chat_output_items,
    chat_tool,
    chat_tool_acc_from_message,
    chat_tool_call_message,
    extract_responses_text,
    message_text,
)
from uv_agent.model.http import (
    SSE_DONE,
    auth_headers,
    decode_json_response,
    endpoint_url,
    parse_sse_event,
    post_json,
    stream_sse,
)
from uv_agent.model.responses import (
    create_responses_response,
    parse_responses_response,
    responses_payload,
    stream_responses_response,
)
from uv_agent.model.types import ModelClient, ModelResponse, ModelStreamEvent, ToolCallDelta


class UnifiedModelClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def reload_config(self, config: AppConfig) -> None:
        self.config = config

    async def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        level: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
    ) -> ModelResponse:
        model = self.config.model_for_level(level)
        provider = self.config.provider_for_model(model)
        if model.api == "anthropic_messages":
            return await create_anthropic_response(
                provider=provider,
                model=model,
                input_items=input_items,
                tools=tools or [],
                instructions=instructions,
            )
        if model.api == "chat_completions":
            return await create_chat_response(
                provider=provider,
                model=model,
                input_items=input_items,
                tools=tools or [],
                instructions=instructions,
            )
        return await create_responses_response(
            provider=provider,
            model=model,
            input_items=input_items,
            tools=tools or [],
            instructions=instructions,
            previous_response_id=previous_response_id,
        )

    async def stream_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        level: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        model = self.config.model_for_level(level)
        provider = self.config.provider_for_model(model)
        if model.api == "anthropic_messages":
            async for event in stream_anthropic_response(
                provider=provider,
                model=model,
                input_items=input_items,
                tools=tools or [],
                instructions=instructions,
                stream_events=stream_sse,
            ):
                yield event
            return
        if model.api == "chat_completions":
            async for event in stream_chat_response(
                provider=provider,
                model=model,
                input_items=input_items,
                tools=tools or [],
                instructions=instructions,
                stream_events=stream_sse,
            ):
                yield event
            return
        async for event in stream_responses_response(
            provider=provider,
            model=model,
            input_items=input_items,
            tools=tools or [],
            instructions=instructions,
            previous_response_id=previous_response_id,
            stream_events=stream_sse,
        ):
            yield event


class FakeModelClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        level: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
    ) -> ModelResponse:
        self.requests.append(
            copy.deepcopy(
                {
                    "input": input_items,
                    "level": level,
                    "tools": tools or [],
                    "instructions": instructions,
                    "stream": False,
                    "previous_response_id": previous_response_id,
                }
            )
        )
        if not self.responses:
            raise RuntimeError("FakeModelClient has no responses left")
        return parse_responses_response(self.responses.pop(0))

    async def stream_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        level: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        response = await self.create_response(
            input_items=input_items,
            level=level,
            tools=tools,
            instructions=instructions,
            previous_response_id=previous_response_id,
        )
        if response.output_text:
            yield ModelStreamEvent(type="text_delta", text=response.output_text)
        yield ModelStreamEvent(type="completed", response=response)
