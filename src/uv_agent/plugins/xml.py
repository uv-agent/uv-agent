from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape as xml_escape
from typing import Any

_TAG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class XmlContribution:
    """Structured model-visible XML contribution supplied by a plugin."""

    tag: str
    body: Any
    attrs: dict[str, Any] | None = None
    plugin: str = ""
    operation: str | None = None


def render_contribution(
    tag: str,
    body: Any,
    *,
    attrs: Mapping[str, Any] | None = None,
    operation: str | None = None,
    agent_prefix: bool = True,
) -> str:
    """Render a structured contribution using uv-agent XML rules.

    Plugin authors pass dictionaries/lists/scalars instead of hand-written XML.
    The renderer owns escaping and the top-level ``agent_`` prefix, which keeps
    system/context blocks visually distinct from user-authored XML.
    """

    top_tag = _agent_tag(tag) if agent_prefix else _validate_tag(tag)
    merged_attrs = dict(attrs or {})
    if operation:
        merged_attrs["operation"] = operation
    attr_text = _attrs(merged_attrs)
    if body is None:
        return f"<{top_tag}{attr_text} />"
    inner = _render_value(body, tag_name=None)
    if isinstance(body, (Mapping, list, tuple)) or "\n" in inner:
        return f"<{top_tag}{attr_text}>\n{inner}\n</{top_tag}>"
    return f"<{top_tag}{attr_text}>{inner}</{top_tag}>"


def render_update_envelope(contributions: Sequence[XmlContribution]) -> str:
    inner = "\n\n".join(
        render_contribution(item.tag, item.body, attrs=item.attrs, operation=item.operation or (item.attrs or {}).get("operation"))
        for item in contributions
    )
    return f"<agent_epoch_context_update>\n{inner}\n</agent_epoch_context_update>"


def _agent_tag(tag: str) -> str:
    tag = _validate_tag(tag)
    return tag if tag.startswith("agent_") else f"agent_{tag}"


def _validate_tag(tag: str) -> str:
    value = str(tag or "").strip()
    if not _TAG_RE.match(value):
        raise ValueError(f"Invalid XML tag name: {tag!r}")
    return value


def _attrs(attrs: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(attrs):
        if key == "operation" and attrs.get(key) is None:
            continue
        value = attrs[key]
        if value is None:
            continue
        name = _validate_tag(str(key))
        rendered = _scalar_text(value, attr=True)
        parts.append(f' {name}="{rendered}"')
    return "".join(parts)


def _render_value(value: Any, *, tag_name: str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, child in value.items():
            if child is None:
                continue
            lines.extend(_render_child(str(key), child))
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        child_tag = tag_name or "item"
        return "\n".join(line for item in value for line in _render_child(child_tag, item))
    return _scalar_text(value)


def _render_child(tag: str, value: Any) -> list[str]:
    tag = _validate_tag(tag)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        body = "\n".join(line for item in value for line in _render_child("item", item))
        if body:
            return [f"<{tag}>", body, f"</{tag}>"]
        return [f"<{tag} />"]
    if isinstance(value, Mapping):
        body = _render_value(value, tag_name=None)
        if body:
            return [f"<{tag}>", body, f"</{tag}>"]
        return [f"<{tag} />"]
    text = _scalar_text(value)
    return [f"<{tag}>{text}</{tag}>"]


def _scalar_text(value: Any, *, attr: bool = False) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return xml_escape(text, quote=attr)
