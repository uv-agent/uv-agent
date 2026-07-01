from __future__ import annotations

from uv_agent.mcp_config import McpInstructionsPreview, discover_mcp_servers
from uv_agent.plugins import CommandResult, OpenPickerAction, PluginManifest, SetupPlugin


MANIFEST = PluginManifest(
    id="builtin.mcp",
    version="0.1.0",
    display_name="MCP",
    description="Discover declared MCP servers and expose the rt.mcp runtime namespace.",
    builtin=True,
    priority=200,
    capabilities=("runtime_namespace", "context", "ui"),
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


def setup(context) -> None:
    context.runtime.register_namespace(
        "mcp",
        doc="MCP discovery and client connection helpers.",
        transport="local_module",
        module="uv_agent_runtime.mcp",
    )
    servers = discover_mcp_servers(context.project_root)
    instructions = _instructions_snapshot(getattr(context, "host", None))
    body: dict[str, object] = {
        "rule": "遇到适合任务的 MCP server 时，通过 rt.mcp helpers 初始化并检查 instructions。",
        "helper": {
            "import": "import uv_agent_runtime as rt",
            "signature": "rt.mcp.list/connect/connect_url/connect_stdio/connect_declared",
        },
        "server": [_server_body(server, instructions.get(server.key)) for server in servers[:10]],
    }
    if len(servers) > 10:
        body["omitted"] = len(servers) - 10
    context.ui.picker(id="mcp", title="MCP servers", provider=lambda query="": _mcp_items(context.project_root, query), trigger="@mcp")
    context.commands.register("/mcp", lambda payload: CommandResult((OpenPickerAction("mcp"),)), description="list MCP servers and insert @mcp mentions")
    context.context.epoch.publish(tag="available_mcp_servers", body=body)


def _mcp_items(project_root, query: str = "") -> list[dict[str, str]]:
    needle = str(query or "").lower()
    items: list[dict[str, str]] = []
    for server in discover_mcp_servers(project_root):
        haystack = f"{server.name} {server.description} {server.scope} {server.transport}".lower()
        if needle and needle not in haystack:
            continue
        endpoint = f" · {server.endpoint}" if server.endpoint else ""
        items.append({
            "value": f"@mcp:{server.name}",
            "description": server.description,
            "id": server.name,
            "kind": "mcp-mention",
            "meta": f"{server.scope} · {server.transport}{endpoint}",
        })
    return items[:30]


def _instructions_snapshot(host) -> dict[tuple[str, str, str], McpInstructionsPreview]:
    probe = getattr(host, "_mcp_instructions_probe", None)
    snapshot = getattr(probe, "snapshot", None)
    if not callable(snapshot):
        return {}
    try:
        return dict(snapshot())
    except Exception:
        return {}


def _server_body(server, preview: McpInstructionsPreview | None) -> dict[str, object]:
    data: dict[str, object] = {
        "name": server.name,
        "scope": server.scope,
        "config": str(server.path),
        "transport": server.transport,
        "description": server.description,
    }
    if server.endpoint:
        data["endpoint"] = server.endpoint
    if preview is not None:
        data["instructions"] = {"text": preview.text, "truncated": preview.truncated}
    return data
