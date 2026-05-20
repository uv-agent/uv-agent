from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class McpServerSummary:
    name: str
    scope: str
    transport: str
    endpoint: str | None
    description: str
    path: Path


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
                    description=str(value.get("description") or "No description"),
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


def render_mcp_summary(servers: list[McpServerSummary], *, limit: int = 10) -> str:
    """Render MCP declarations for the system prompt."""
    if not servers:
        return "None declared."
    lines = []
    for server in servers[:limit]:
        lines.append(
            f"- {server.name} ({server.scope}): {server.description}; config {server.path}"
        )
    if len(servers) > limit:
        lines.append(f"- ... {len(servers) - limit} more MCP servers declared")
    return "\n".join(lines)
