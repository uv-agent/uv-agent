from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape as xml_escape
from pathlib import Path
from typing import Any, Mapping

MCP_NONE_DECLARED = "未声明。"
MCP_DEFAULT_DESCRIPTION = "无描述"
MCP_SERVER_INLINE_TEMPLATE = '<mcp_server {attrs}>{description}</mcp_server>'
MCP_SERVER_OPEN_TEMPLATE = '<mcp_server {attrs}>'
MCP_SERVER_DESCRIPTION_TEMPLATE = '<description>{description}</description>'
MCP_SERVER_INSTRUCTIONS_TEMPLATE = '<instructions truncated="{truncated}">{instructions}</instructions>'
MCP_SERVER_CLOSE = "</mcp_server>"
MCP_OMITTED_TEMPLATE = '<omitted_mcp_servers count="{count}" />'


@dataclass(frozen=True)
class McpServerSummary:
    name: str
    scope: str
    transport: str
    endpoint: str | None
    description: str
    path: Path

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.scope, self.name, str(self.path))


@dataclass(frozen=True)
class McpInstructionsPreview:
    text: str
    truncated: bool = False


def discover_mcp_servers(project_root: Path, *, home: Path | None = None) -> list[McpServerSummary]:
    """Discover MCP server declarations from .agents/mcp.json files."""
    files = [
        ("project", project_root.resolve() / ".agents" / "mcp.json"),
        ("user", (home or Path.home()).resolve() / ".agents" / "mcp.json"),
    ]
    servers: list[McpServerSummary] = []
    for scope, path in files:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_servers = data.get("servers") if isinstance(data, dict) else None
        if not isinstance(raw_servers, dict):
            continue
        for name, value in raw_servers.items():
            if not isinstance(value, dict):
                continue
            servers.append(
                McpServerSummary(
                    name=str(name),
                    scope=scope,
                    transport=transport_summary(value),
                    endpoint=endpoint_summary(value),
                    description=str(value.get("description") or MCP_DEFAULT_DESCRIPTION),
                    path=path,
                )
            )
    return servers


def transport_summary(value: dict[str, Any]) -> str:
    if isinstance(value.get("transport"), str):
        transport = str(value["transport"])
        return "streamable_http" if transport == "http" else transport
    if url_summary(value):
        return "streamable_http"
    return "stdio"


def endpoint_summary(value: dict[str, Any]) -> str | None:
    url = url_summary(value)
    if url:
        return url
    command = value.get("command")
    if isinstance(command, str):
        args = value.get("args")
        if isinstance(args, list) and args:
            return " ".join([command, *map(str, args[:3])])
        return command
    return None


def url_summary(value: dict[str, Any]) -> str | None:
    for key in ("url", "httpUrl", "serverUrl"):
        item = value.get(key)
        if isinstance(item, str) and item:
            return item
    return None


def render_mcp_summary(
    servers: list[McpServerSummary],
    *,
    instructions: Mapping[tuple[str, str, str], McpInstructionsPreview] | None = None,
    limit: int = 10,
) -> str:
    """Render MCP declarations for the system prompt."""
    if not servers:
        return MCP_NONE_DECLARED
    lines = []
    for server in servers[:limit]:
        lines.append(render_mcp_entry(server, instructions.get(server.key) if instructions else None))
    if len(servers) > limit:
        lines.append(MCP_OMITTED_TEMPLATE.format(count=len(servers) - limit))
    return "\n".join(lines)


def render_mcp_entry(
    server: McpServerSummary,
    instructions: McpInstructionsPreview | None = None,
) -> str:
    attrs = (
        f'name="{_xml_attr(server.name)}" scope="{_xml_attr(server.scope)}" '
        f'config="{_xml_attr(server.path)}"'
    )
    if instructions is None:
        return MCP_SERVER_INLINE_TEMPLATE.format(attrs=attrs, description=_xml_text(server.description))
    truncated = "true" if instructions.truncated else "false"
    return "\n".join(
        [
            MCP_SERVER_OPEN_TEMPLATE.format(attrs=attrs),
            MCP_SERVER_DESCRIPTION_TEMPLATE.format(description=_xml_text(server.description)),
            MCP_SERVER_INSTRUCTIONS_TEMPLATE.format(truncated=truncated, instructions=_xml_text(instructions.text)),
            MCP_SERVER_CLOSE,
        ]
    )


def _xml_attr(value: object) -> str:
    return xml_escape(str(value), quote=True)


def _xml_text(value: object) -> str:
    return xml_escape(str(value), quote=False)
