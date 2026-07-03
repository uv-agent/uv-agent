from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator
from typing import Any

from uv_agent.config import AppConfig
from uv_agent.model.types import ModelResponse, ModelStreamEvent


logger = logging.getLogger(__name__)


class UnifiedModelClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def reload_config(self, config: AppConfig) -> None:
        self.config = config
        logger.debug("Model client config reloaded")

    async def aclose(self) -> None:
        """Close all cached provider SDK clients."""

        from uv_agent.model.anthropic import close_all_anthropic_clients
        from uv_agent.model.openai_sdk import close_all_openai_clients

        logger.debug("Closing cached provider clients")
        await close_all_openai_clients()
        await close_all_anthropic_clients()

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
        request_tools = tools or []
        logger.debug(
            "Model request started stream=False level=%s provider=%s api=%s model=%s input_items=%d tools=%d previous_response=%s",
            level,
            provider.name,
            model.api,
            model.model,
            len(input_items),
            len(request_tools),
            bool(previous_response_id),
        )
        try:
            if model.api == "anthropic_messages":
                # Provider SDKs are comparatively expensive to import. Resolve the
                # concrete backend only when a request actually targets it so the
                # TUI can reach first paint without loading every provider.
                from uv_agent.model.anthropic import create_anthropic_response

                response = await create_anthropic_response(
                    provider=provider,
                    model=model,
                    input_items=input_items,
                    tools=request_tools,
                    instructions=instructions,
                )
            elif model.api == "chat_completions":
                from uv_agent.model.chat import create_chat_response

                response = await create_chat_response(
                    provider=provider,
                    model=model,
                    input_items=input_items,
                    tools=request_tools,
                    instructions=instructions,
                )
            else:
                from uv_agent.model.responses import create_responses_response

                response = await create_responses_response(
                    provider=provider,
                    model=model,
                    input_items=input_items,
                    tools=request_tools,
                    instructions=instructions,
                    previous_response_id=previous_response_id,
                )
        except Exception as exc:
            logger.warning(
                "Model request failed stream=False level=%s provider=%s api=%s model=%s error_type=%s",
                level,
                provider.name,
                model.api,
                model.model,
                exc.__class__.__name__,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            raise
        logger.debug(
            "Model request completed stream=False level=%s provider=%s api=%s model=%s response_id=%s output_items=%d",
            level,
            provider.name,
            model.api,
            model.model,
            response.id,
            len(response.output),
        )
        return response

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
        request_tools = tools or []
        event_count = 0
        completed = False
        logger.debug(
            "Model request started stream=True level=%s provider=%s api=%s model=%s input_items=%d tools=%d previous_response=%s",
            level,
            provider.name,
            model.api,
            model.model,
            len(input_items),
            len(request_tools),
            bool(previous_response_id),
        )
        try:
            if model.api == "anthropic_messages":
                from uv_agent.model.anthropic import stream_anthropic_response

                stream = stream_anthropic_response(
                    provider=provider,
                    model=model,
                    input_items=input_items,
                    tools=request_tools,
                    instructions=instructions,
                )
            elif model.api == "chat_completions":
                from uv_agent.model.chat import stream_chat_response

                stream = stream_chat_response(
                    provider=provider,
                    model=model,
                    input_items=input_items,
                    tools=request_tools,
                    instructions=instructions,
                )
            else:
                from uv_agent.model.responses import stream_responses_response

                stream = stream_responses_response(
                    provider=provider,
                    model=model,
                    input_items=input_items,
                    tools=request_tools,
                    instructions=instructions,
                    previous_response_id=previous_response_id,
                )

            async for event in stream:
                event_count += 1
                if event.type == "completed":
                    completed = True
                yield event
        except Exception as exc:
            logger.warning(
                "Model request failed stream=True level=%s provider=%s api=%s model=%s events=%d error_type=%s",
                level,
                provider.name,
                model.api,
                model.model,
                event_count,
                exc.__class__.__name__,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            raise
        logger.debug(
            "Model request completed stream=True level=%s provider=%s api=%s model=%s events=%d completed=%s",
            level,
            provider.name,
            model.api,
            model.model,
            event_count,
            completed,
        )


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
        from uv_agent.model.responses import parse_responses_response

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
