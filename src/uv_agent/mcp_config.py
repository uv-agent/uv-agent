from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class McpServerSummary:
    name: str
    scope: str
    command: str | None
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
                    command=command_summary(value),
                    description=str(value.get("description") or "No description"),
                    path=path,
                )
            )
    return servers


def command_summary(value: dict[str, Any]) -> str | None:
    command = value.get("command")
    if isinstance(command, str):
        args = value.get("args")
        if isinstance(args, list) and args:
            return " ".join([command, *map(str, args[:3])])
        return command
    return None


def render_mcp_summary(servers: list[McpServerSummary], *, limit: int = 10) -> str:
    """Render MCP declarations for the system prompt."""
    if not servers:
        return "None declared."
    lines = []
    for server in servers[:limit]:
        command = f"; command: {server.command}" if server.command else ""
        lines.append(
            f"- {server.name} ({server.scope}): {server.description}{command}; config {server.path}"
        )
    if len(servers) > limit:
        lines.append(f"- ... {len(servers) - limit} more MCP servers declared")
    return "\n".join(lines)
