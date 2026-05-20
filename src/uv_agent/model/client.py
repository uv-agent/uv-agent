from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from typing import Any

from uv_agent.config import AppConfig
from uv_agent.model.anthropic import create_anthropic_response, stream_anthropic_response
from uv_agent.model.chat import create_chat_response, stream_chat_response
from uv_agent.model.responses import (
    create_responses_response,
    parse_responses_response,
    stream_responses_response,
)
from uv_agent.model.types import ModelResponse, ModelStreamEvent


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
