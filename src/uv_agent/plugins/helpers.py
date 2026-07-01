from __future__ import annotations

import json
from typing import Any


def payload_from_call(args: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    if args and kwargs:
        raise TypeError("Runtime helpers accept either one payload dict or keyword arguments, not both")
    if len(args) > 1:
        raise TypeError("Runtime helpers accept at most one positional payload dict")
    if args:
        payload = args[0]
        if not isinstance(payload, dict):
            raise TypeError("Runtime helper payload must be a dict")
        return dict(payload)
    return dict(kwargs)


def validate_json_value(value: Any, *, label: str = "value") -> None:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-serializable") from exc


def validate_payload(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Runtime helper payload must be a dict")
    validate_json_value(payload, label="runtime helper payload")
    required = schema.get("required") or []
    if isinstance(required, list):
        missing = [str(name) for name in required if isinstance(name, str) and name not in payload]
        if missing:
            raise ValueError(f"Runtime helper payload missing required field(s): {', '.join(missing)}")
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
            raise TypeError(f"Runtime helper payload field {key!r} does not match schema type {expected!r}")


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
