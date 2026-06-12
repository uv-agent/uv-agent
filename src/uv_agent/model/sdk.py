from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from uv_agent import DEFAULT_USER_AGENT
from uv_agent.config import ModelConfig, ProviderConfig


def sdk_param_keys(method: Callable[..., object]) -> set[str]:
    return {key for key in inspect.signature(method).parameters if key != "self"}


def sdk_base_url(provider: ProviderConfig, api: str, endpoint_suffix: str) -> str:
    endpoint_path = provider.endpoint_for_api(api).path
    endpoint_url = provider.base_url.rstrip("/") + endpoint_path
    if endpoint_url.endswith(endpoint_suffix):
        return endpoint_url[: -len(endpoint_suffix)] or provider.base_url.rstrip("/")
    return provider.base_url


def default_headers(provider_headers: dict[str, str] | None) -> dict[str, str] | None:
    """Return default SDK headers merged with provider-configured headers."""

    headers: dict[str, str] = {"User-Agent": DEFAULT_USER_AGENT}
    if provider_headers:
        headers.update(provider_headers)
    return headers


def sdk_kwargs(
    payload: Mapping[str, Any],
    param_sources: Sequence[Mapping[str, Any]],
    allowed_keys: set[str],
) -> dict[str, Any]:
    kwargs = {
        key: value
        for key, value in payload.items()
        if key in allowed_keys and key != "extra_body"
    }
    extra_body = sdk_extra_body(param_sources, allowed_keys)
    if extra_body:
        kwargs["extra_body"] = extra_body
    elif "extra_body" in payload and "extra_body" in allowed_keys:
        kwargs["extra_body"] = payload["extra_body"]
    return kwargs


def sdk_extra_body(
    param_sources: Sequence[Mapping[str, Any]],
    allowed_keys: set[str],
) -> dict[str, Any] | None:
    extra: dict[str, Any] = {}
    for source in param_sources:
        configured_extra = source.get("extra_body")
        if isinstance(configured_extra, dict):
            extra.update(configured_extra)
        for key, value in source.items():
            if key not in allowed_keys:
                extra[key] = value
    return extra or None


def model_param_sources(
    provider: ProviderConfig,
    model: ModelConfig,
    api: str | None = None,
) -> list[Mapping[str, Any]]:
    endpoint = provider.endpoint_for_api(api or model.api)
    return [provider.params, endpoint.params, model.params]


def object_dump(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return value
    if not hasattr(value, "__iter__"):
        return {}
    try:
        return dict(cast(Any, value))
    except (TypeError, ValueError):
        return {}
