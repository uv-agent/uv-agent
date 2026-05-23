from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from typing import Any

from .session import RunContext

HostMethod = Callable[..., Any]


class MethodRegistry:
    """Thread-safe registry for host methods callable from runtime scripts."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._methods: dict[str, HostMethod] = {}

    def register(self, name: str, method: HostMethod) -> None:
        normalized = _normalize_name(name)
        if not callable(method):
            raise TypeError(f"Host method is not callable: {name}")
        with self._lock:
            self._methods[normalized] = method

    def unregister(self, name: str) -> None:
        normalized = _normalize_name(name)
        with self._lock:
            self._methods.pop(normalized, None)

    def get(self, name: str) -> HostMethod | None:
        normalized = _normalize_name(name)
        with self._lock:
            return self._methods.get(normalized)

    def call(self, name: str, params: dict[str, Any], *, context: RunContext) -> Any:
        method = self.get(name)
        if method is None:
            raise KeyError(name)
        call_params = dict(params)
        if "context" not in call_params and _accepts_context(method):
            call_params["context"] = context
        return method(**call_params)


def _accepts_context(method: HostMethod) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return "context" in signature.parameters


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Host method name must be a non-empty string")
    normalized = name.strip()
    if normalized.startswith("call."):
        normalized = normalized.removeprefix("call.")
    if any(part == "" for part in normalized.split(".")):
        raise ValueError(f"Invalid host method name: {name!r}")
    return normalized
