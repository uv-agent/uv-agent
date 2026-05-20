from __future__ import annotations

from openai import AsyncOpenAI

from uv_agent.config import ProviderConfig
from uv_agent.model.sdk import sdk_base_url


def openai_client(provider: ProviderConfig, api: str, endpoint_suffix: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=provider.resolved_api_key(),
        base_url=sdk_base_url(provider, api, endpoint_suffix),
        default_headers=provider.headers or None,
        _enforce_credentials=False,
    )
