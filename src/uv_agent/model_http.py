from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from uv_agent.config import ProviderConfig

SSE_DONE = "__uv_agent_sse_done__"


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
        return {"type": SSE_DONE}
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
