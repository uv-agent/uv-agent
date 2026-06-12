from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from openai import Timeout

from uv_agent.config import ProviderConfig
from uv_agent.model.sdk import default_headers, sdk_base_url


_openai_client_cache: dict[tuple[Any, ...], AsyncOpenAI] = {}


def _openai_client_key(provider: ProviderConfig, api: str, endpoint_suffix: str) -> tuple[Any, ...]:
    """Hashable key for the cached OpenAI-compatible client."""

    headers = provider.headers or {}
    return (
        provider.name,
        sdk_base_url(provider, api, endpoint_suffix),
        api,
        endpoint_suffix,
        provider.resolved_api_key(),
        frozenset(headers.items()),
        provider.timeout_s,
    )


def openai_client(provider: ProviderConfig, api: str, endpoint_suffix: str) -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client for the provider configuration.

    Clients are reused across requests that share the same provider identity,
    API, endpoint, resolved API key, headers, and timeout. Callers should use
    ``close_all_openai_clients`` to release resources when the process shuts
    down or the configuration changes materially.
    """

    key = _openai_client_key(provider, api, endpoint_suffix)
    cached = _openai_client_cache.get(key)
    if cached is not None:
        return cached
    kwargs = {
        "api_key": provider.resolved_api_key(),
        "base_url": sdk_base_url(provider, api, endpoint_suffix),
        "default_headers": default_headers(provider.headers),
    }
    if provider.timeout_s is not None:
        # Keep connection failures quick while allowing long model generation
        # or streaming gaps from slower upstream providers.
        kwargs["timeout"] = Timeout(provider.timeout_s, connect=5.0)
    client = AsyncOpenAI(**kwargs)
    _openai_client_cache[key] = client
    return client


async def close_all_openai_clients() -> None:
    """Close all cached OpenAI-compatible clients and empty the cache."""

    clients: list[AsyncOpenAI] = []
    clients.extend(_openai_client_cache.values())
    _openai_client_cache.clear()
    for client in clients:
        await client.close()


def _reset_openai_client_cache() -> None:
    """Clear the cache without closing clients. For tests only."""

    _openai_client_cache.clear()
