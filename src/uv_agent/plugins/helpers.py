from __future__ import annotations

import asyncio
import inspect
import json
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
    fn: Callable[[dict[str, Any]], Any]
    doc: str
    schema: dict[str, Any]
    timeout_s: float | None = None


class RuntimeHelperRegistry:
    """Thread-safe registry of plugin-provided host handlers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._helpers: dict[str, RuntimeHelperSpec] = {}

    def register_handler(
        self,
        *,
        plugin: str,
        name: str,
        fn: Callable[[dict[str, Any]], Any],
        doc: str,
        schema: dict[str, Any],
        timeout_s: float | None = None,
    ) -> RuntimeHelperSpec:
        if not _HELPER_NAME_RE.match(name):
            raise ValueError(f"Invalid handler name: {name!r}")
        if not callable(fn):
            raise TypeError(f"Handler is not callable: {name}")
        doc = str(doc or "").strip()
        if not doc:
            raise ValueError(f"Handler {name!r} requires non-empty doc")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError(f"Handler {name!r} requires an object JSON schema")
        spec = RuntimeHelperSpec(name=name, plugin=plugin, fn=fn, doc=doc, schema=dict(schema), timeout_s=timeout_s)
        with self._lock:
            if name in self._helpers:
                raise ValueError(f"Handler already registered: {name}")
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
            "doc": spec.doc,
            "schema": spec.schema,
        }

    async def call(self, name: str, payload: dict[str, Any], *, timeout_s: float | None = None) -> Any:
        spec = self.get(name)
        if spec is None:
            raise LookupError(f"Unknown handler: {name}")
        validate_payload(payload, spec.schema)
        result = spec.fn(payload)
        effective_timeout = timeout_s if timeout_s is not None else spec.timeout_s
        if inspect.isawaitable(result):
            if effective_timeout is not None:
                result = await asyncio.wait_for(result, timeout=effective_timeout)
            else:
                result = await result
        validate_json_value(result, label=f"handler {name!r} return value")
        return result


def payload_from_call(args: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    if args and kwargs:
        raise TypeError("Handler helpers accept either one payload dict or keyword arguments, not both")
    if len(args) > 1:
        raise TypeError("Handler helpers accept at most one positional payload dict")
    if args:
        payload = args[0]
        if not isinstance(payload, dict):
            raise TypeError("Handler helper payload must be a dict")
        return dict(payload)
    return dict(kwargs)


def validate_json_value(value: Any, *, label: str = "value") -> None:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-serializable") from exc


def validate_payload(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Handler payload must be a dict")
    validate_json_value(payload, label="handler payload")
    required = schema.get("required") or []
    if isinstance(required, list):
        missing = [str(name) for name in required if isinstance(name, str) and name not in payload]
        if missing:
            raise ValueError(f"Handler payload missing required field(s): {', '.join(missing)}")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    for key, spec in properties.items():
        if key not in payload or not isinstance(spec, dict):
            continue
        expected = spec.get("type")
        if expected is None:
            continue
        if not _matches_json_type(payload[key], expected):
            raise TypeError(f"Handler payload field {key!r} does not match schema type {expected!r}")


def _matches_json_type(value: Any, expected: Any) -> bool:
    expected_types = expected if isinstance(expected, list) else [expected]
    for item in expected_types:
        if item == "null" and value is None:
            return True
        if item == "string" and isinstance(value, str):
            return True
        if item == "boolean" and isinstance(value, bool):
            return True
        if item == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if item == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if item == "object" and isinstance(value, dict):
            return True
        if item == "array" and isinstance(value, list):
            return True
    return False
