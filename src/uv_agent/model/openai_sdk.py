from __future__ import annotations

from openai import AsyncOpenAI
from openai import Timeout

from uv_agent.config import ProviderConfig
from uv_agent.model.sdk import sdk_base_url


def openai_client(provider: ProviderConfig, api: str, endpoint_suffix: str) -> AsyncOpenAI:
    kwargs = {
        "api_key": provider.resolved_api_key(),
        "base_url": sdk_base_url(provider, api, endpoint_suffix),
        "default_headers": provider.headers or None,
    }
    if provider.timeout_s is not None:
        # Keep connection failures quick while allowing long model generation
        # or streaming gaps from slower upstream providers.
        kwargs["timeout"] = Timeout(provider.timeout_s, connect=5.0)
    return AsyncOpenAI(**kwargs)
