from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ModelResponse:
    id: str | None
    output: list[dict[str, Any]]
    output_text: str
    raw: dict[str, Any]
    usage: dict[str, Any]
    reasoning_text: str = ""


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""
    arguments_delta: str = ""


@dataclass(frozen=True)
class ModelStreamEvent:
    type: Literal["text_delta", "tool_call_delta", "reasoning_delta", "completed"]
    text: str = ""
    tool_call: ToolCallDelta | None = None
    response: ModelResponse | None = None


class ModelClient(Protocol):
    async def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        level: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
    ) -> ModelResponse:
        ...

    def stream_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        level: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Return an async iterator immediately; callers must not await it first.

        Concrete clients implement this as async-generator methods. An ``async
        def`` protocol member would describe a coroutine that resolves to an
        iterator, which is a different call contract and caused cascading type
        errors in clients and tests.
        """

        ...
