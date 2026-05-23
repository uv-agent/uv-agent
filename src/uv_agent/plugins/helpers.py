from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


_HELPER_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")


@dataclass(frozen=True)
class RuntimeHelperSpec:
    name: str
    plugin: str
    fn: Callable[..., Any]
    doc: str | None = None
    schema: dict[str, Any] | None = None


class RuntimeHelperRegistry:
    """Thread-safe registry of plugin-provided runtime helpers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._helpers: dict[str, RuntimeHelperSpec] = {}

    def register(
        self,
        *,
        plugin: str,
        name: str,
        fn: Callable[..., Any],
        doc: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> RuntimeHelperSpec:
        if not _HELPER_NAME_RE.match(name):
            raise ValueError(f"Invalid runtime helper name: {name!r}")
        if not callable(fn):
            raise TypeError(f"Runtime helper is not callable: {name}")
        spec = RuntimeHelperSpec(name=name, plugin=plugin, fn=fn, doc=doc, schema=schema)
        with self._lock:
            if name in self._helpers:
                raise ValueError(f"Runtime helper already registered: {name}")
            self._helpers[name] = spec
        return spec

    def get(self, name: str) -> RuntimeHelperSpec | None:
        with self._lock:
            return self._helpers.get(name)

    def list(self) -> list[RuntimeHelperSpec]:
        with self._lock:
            return sorted(self._helpers.values(), key=lambda item: item.name)

    def resolve_payload(self, name: str) -> dict[str, Any]:
        spec = self.get(name)
        if spec is None:
            return {"found": False, "name": name}
        return {
            "found": True,
            "name": spec.name,
            "plugin": spec.plugin,
            "doc": spec.doc or "",
            "schema": spec.schema or {},
        }
