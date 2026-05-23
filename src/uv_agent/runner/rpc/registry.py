from __future__ import annotations

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
        try:
            return method(**params)
        except TypeError as exc:
            # If a host method explicitly asks for context, provide the current
            # immutable run context without making it mandatory for simple helpers.
            if "context" not in params:
                try:
                    return method(context=context, **params)
                except TypeError:
                    pass
            raise exc


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Host method name must be a non-empty string")
    normalized = name.strip()
    if normalized.startswith("call."):
        normalized = normalized.removeprefix("call.")
    if any(part == "" for part in normalized.split(".")):
        raise ValueError(f"Invalid host method name: {name!r}")
    return normalized
