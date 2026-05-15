from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from uv_agent.config import AppConfig, ModelConfig, ProviderConfig


@dataclass(frozen=True)
class ModelResponse:
    id: str | None
    output: list[dict[str, Any]]
    output_text: str
    raw: dict[str, Any]
    usage: dict[str, Any]


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


@dataclass(frozen=True)
class ModelStreamEvent:
    type: Literal["text_delta", "tool_call_delta", "completed"]
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
    ) -> ModelResponse:
        ...

    async def stream_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        level: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        ...


class UnifiedModelClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    async def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        level: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
    ) -> ModelResponse:
        model = self.config.model_for_level(level)
        provider = self.config.provider_for_model(model)
        if model.api == "anthropic_messages":
            return await self._create_anthropic(
                provider=provider,
                model=model,
                input_items=input_items,
                tools=tools or [],
                instructions=instructions,
            )
        if model.api == "chat_completions":
            return await self._create_chat(
                provider=provider,
                model=model,
                input_items=input_items,
                tools=tools or [],
                instructions=instructions,
            )
        return await self._create_responses(
            provider=provider,
            model=model,
            input_items=input_items,
            tools=tools or [],
            instructions=instructions,
        )

    async def stream_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        level: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        model = self.config.model_for_level(level)
        provider = self.config.provider_for_model(model)
        if model.api == "anthropic_messages":
            async for event in self._stream_anthropic(
                provider=provider,
                model=model,
                input_items=input_items,
                tools=tools or [],
                instructions=instructions,
            ):
                yield event
            return
        if model.api == "chat_completions":
            async for event in self._stream_chat(
                provider=provider,
                model=model,
                input_items=input_items,
                tools=tools or [],
                instructions=instructions,
            ):
                yield event
            return
        async for event in self._stream_responses(
            provider=provider,
            model=model,
            input_items=input_items,
            tools=tools or [],
            instructions=instructions,
        ):
            yield event

    async def _create_responses(
        self,
        *,
        provider: ProviderConfig,
        model: ModelConfig,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str | None,
    ) -> ModelResponse:
        payload = responses_payload(provider, model, input_items, tools, instructions, stream=False)
        data = await post_json(provider, model.api, payload)
        return parse_responses_response(data)

    async def _create_chat(
        self,
        *,
        provider: ProviderConfig,
        model: ModelConfig,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str | None,
    ) -> ModelResponse:
        payload = chat_payload(provider, model, input_items, tools, instructions, stream=False)
        data = await post_json(provider, model.api, payload)
        return parse_chat_response(data)

    async def _create_anthropic(
        self,
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

    async def _stream_responses(
        self,
        *,
        provider: ProviderConfig,
        model: ModelConfig,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str | None,
    ) -> AsyncIterator[ModelStreamEvent]:
        payload = responses_payload(provider, model, input_items, tools, instructions, stream=True)
        text_parts: list[str] = []
        output: list[dict[str, Any]] = []
        async for data in stream_sse(provider, model.api, payload):
            event_type = data.get("type", "")
            if event_type in {"response.output_text.delta", "response.refusal.delta"}:
                delta = data.get("delta", "")
                text_parts.append(delta)
                yield ModelStreamEvent(type="text_delta", text=delta)
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

    async def _stream_chat(
        self,
        *,
        provider: ProviderConfig,
        model: ModelConfig,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str | None,
    ) -> AsyncIterator[ModelStreamEvent]:
        payload = chat_payload(provider, model, input_items, tools, instructions, stream=True)
        text_parts: list[str] = []
        tool_acc: dict[int, dict[str, Any]] = {}
        response_id: str | None = None
        usage: dict[str, Any] = {}
        async for data in stream_sse(provider, model.api, payload):
            if not data:
                continue
            response_id = data.get("id") or response_id
            if data.get("usage"):
                usage = data["usage"]
            for choice in data.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text = delta["content"]
                    text_parts.append(text)
                    yield ModelStreamEvent(type="text_delta", text=text)
                for tool_call in delta.get("tool_calls") or []:
                    index = int(tool_call.get("index", 0))
                    existing = tool_acc.setdefault(index, {"arguments": ""})
                    if tool_call.get("id"):
                        existing["call_id"] = tool_call["id"]
                    function = tool_call.get("function") or {}
                    if function.get("name"):
                        existing["name"] = function["name"]
                    if function.get("arguments"):
                        existing["arguments"] += function["arguments"]
                    yield ModelStreamEvent(
                        type="tool_call_delta",
                        tool_call=ToolCallDelta(
                            index=index,
                            call_id=existing.get("call_id"),
                            name=existing.get("name"),
                            arguments_delta=function.get("arguments", ""),
                        ),
                    )

        output_text = "".join(text_parts)
        output = chat_output_items(output_text, tool_acc)
        yield ModelStreamEvent(
            type="completed",
            response=ModelResponse(
                id=response_id,
                output=output,
                output_text=output_text,
                raw={"id": response_id, "output": output},
                usage=usage,
            ),
        )

    async def _stream_anthropic(
        self,
        *,
        provider: ProviderConfig,
        model: ModelConfig,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str | None,
    ) -> AsyncIterator[ModelStreamEvent]:
        payload = anthropic_payload(provider, model, input_items, tools, instructions, stream=True)
        text_parts: list[str] = []
        tool_acc: dict[int, dict[str, Any]] = {}
        response_id: str | None = None
        usage: dict[str, Any] = {}
        async for data in stream_sse(provider, model.api, payload):
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


def auth_headers(provider: ProviderConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json", **provider.headers}
    api_key = provider.resolved_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def endpoint_url(provider: ProviderConfig, api: str) -> str:
    endpoint = provider.endpoint_for_api(api)
    return provider.base_url.rstrip("/") + endpoint.path


async def post_json(provider: ProviderConfig, api: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            endpoint_url(provider, api),
            headers=auth_headers(provider),
            json=payload,
        )
        response.raise_for_status()
        return decode_json_response(response, endpoint_url(provider, api))


async def stream_sse(
    provider: ProviderConfig,
    api: str,
    payload: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            endpoint_url(provider, api),
            headers=auth_headers(provider),
            json=payload,
        ) as response:
            response.raise_for_status()
            event_name: str | None = None
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    parsed = parse_sse_event(event_name, data_lines)
                    event_name = None
                    data_lines = []
                    if parsed is not None:
                        yield parsed
                    continue
                if line.startswith("event:"):
                    event_name = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").strip())
            parsed = parse_sse_event(event_name, data_lines)
            if parsed is not None:
                yield parsed


def parse_sse_event(event_name: str | None, data_lines: list[str]) -> dict[str, Any] | None:
    if not data_lines:
        return None
    raw = "\n".join(data_lines)
    if raw == "[DONE]":
        return None
    data = json.loads(raw)
    if event_name and "type" not in data:
        data["type"] = event_name
    return data


def decode_json_response(response: httpx.Response, url: str) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        preview = response.text[:160].replace("\n", " ")
        raise ValueError(
            f"Provider returned non-JSON response from {url} "
            f"(content-type={content_type!r}, preview={preview!r})"
        ) from exc


def responses_payload(
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    *,
    stream: bool,
) -> dict[str, Any]:
    endpoint = provider.endpoint_for_api("responses")
    payload: dict[str, Any] = {
        "model": model.model,
        "input": input_items,
        "tools": tools,
        "tool_choice": "auto" if tools else "none",
        **provider.params,
        **endpoint.params,
        **model.params,
    }
    if instructions:
        payload["instructions"] = instructions
    if stream:
        payload["stream"] = True
    return payload


def chat_payload(
    provider: ProviderConfig,
    model: ModelConfig,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str | None,
    *,
    stream: bool,
) -> dict[str, Any]:
    endpoint = provider.endpoint_for_api("chat_completions")
    payload: dict[str, Any] = {
        "model": model.model,
        "messages": chat_messages(input_items, instructions),
        "tools": [chat_tool(tool) for tool in tools],
        "tool_choice": "auto" if tools else "none",
        **provider.params,
        **endpoint.params,
        **model.params,
    }
    if not payload["tools"]:
        payload.pop("tools")
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


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


def parse_chat_response(data: dict[str, Any]) -> ModelResponse:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    output = chat_output_items(text, chat_tool_acc_from_message(message))
    return ModelResponse(
        id=data.get("id"),
        output=output,
        output_text=text,
        raw=data,
        usage=data.get("usage") or {},
    )


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


def extract_responses_text(output: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


def chat_messages(input_items: list[dict[str, Any]], instructions: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    for item in input_items:
        item_type = item.get("type")
        if item_type == "message":
            role = item.get("role", "user")
            messages.append({"role": role, "content": message_text(item)})
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name"),
                                "arguments": item.get("arguments") or "{}",
                            },
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": item.get("output", ""),
                }
            )
    return messages


def anthropic_messages(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in input_items:
        item_type = item.get("type")
        if item_type == "message":
            role = item.get("role", "user")
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": message_text(item)})
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


def message_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for content in item.get("content") or []:
        if content.get("type") in {"input_text", "output_text", "text"}:
            parts.append(content.get("text", ""))
    return "\n".join(parts)


def chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
    }


def chat_tool_acc_from_message(message: dict[str, Any]) -> dict[int, dict[str, Any]]:
    acc: dict[int, dict[str, Any]] = {}
    for index, tool_call in enumerate(message.get("tool_calls") or []):
        function = tool_call.get("function") or {}
        acc[index] = {
            "call_id": tool_call.get("id"),
            "name": function.get("name"),
            "arguments": function.get("arguments", ""),
        }
    return acc


def chat_output_items(output_text: str, tool_acc: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if output_text:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": output_text}],
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
    ) -> ModelResponse:
        self.requests.append(
            copy.deepcopy(
                {
                    "input": input_items,
                    "level": level,
                    "tools": tools or [],
                    "instructions": instructions,
                    "stream": False,
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
    ) -> AsyncIterator[ModelStreamEvent]:
        response = await self.create_response(
            input_items=input_items,
            level=level,
            tools=tools,
            instructions=instructions,
        )
        if response.output_text:
            yield ModelStreamEvent(type="text_delta", text=response.output_text)
        yield ModelStreamEvent(type="completed", response=response)
