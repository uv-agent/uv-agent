from __future__ import annotations

from .config import McpInstructionsPreview, discover_mcp_servers
from .i18n import TEXTS
from .probe import McpInstructionsProbe
from uv_agent.plugins import CommandResult, OpenPickerAction, PluginManifest, SetupPlugin


MCP_HELPER_SIGNATURE = """McpTransport = Literal["stdio", "streamable_http", "sse"]
McpServerConfig = {name: str, transport: McpTransport, command: str | None, args: list[str], url: str | None, cwd: str | None, env: Mapping[str, str] | None, description: str, path: str | None}
class McpResult:
    value: Any
    raw: Any
class McpClient:
    config: McpServerConfig
    def initialize(self) -> McpResult: ...
    def list_tools(self) -> list[dict[str, Any]]: ...
    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> McpResult: ...
    def close(self) -> None: ...
rt.mcp.list_declared_servers(*, config_paths: list[str | Path] | None = None, cwd: str | Path | None = None) -> list[dict[str, Any]]
rt.mcp.connect_named(name: str, *, config_paths: list[str | Path] | None = None, cwd: str | Path | None = None, timeout_s: float | None = 30) -> McpClient
rt.mcp.connect_declared(name: str, *, config_path: str | Path = ".agents/mcp.json", cwd: str | None = None, timeout_s: float | None = 30) -> McpClient
rt.mcp.connect_url(url: str, *, transport: Literal["streamable_http", "sse"] = "streamable_http", timeout_s: float | None = 30) -> McpClient
rt.mcp.connect_stdio(command: list[str], *, cwd: str | None = None, env: Mapping[str, str] | None = None, timeout_s: float | None = 30) -> McpClient"""


MANIFEST = PluginManifest(
    id="builtin.mcp",
    version="0.1.0",
    display_name={"zh": "MCP", "en": "MCP"},
    description={"zh": "发现声明的 MCP servers，并暴露 rt.mcp runtime namespace。", "en": "Discover declared MCP servers and expose the rt.mcp runtime namespace."},
    builtin=True,
    default_enabled=False,
    deprecated=True,
    deprecation_message="builtin.mcp is deprecated and will be removed in a future uv-agent release; use installable MCP skills instead.",
    priority=200,
    capabilities=("runtime_namespace", "context", "ui"),
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


def setup(context) -> None:
    context.i18n.register(TEXTS)
    probe = McpInstructionsProbe(context.project_root)
    probe.start()
    context.runtime.register_namespace(
        "mcp",
        doc="MCP server 发现和 client 连接 helpers。",
        module="uv_agent.builtin.mcp.runtime",
    )
    context.ui.picker(
        id="mcp",
        title={"zh": "MCP servers", "en": "MCP servers"},
        provider=lambda query="": _mcp_items(context.project_root, query),
        trigger="@mcp",
    )
    context.commands.register(
        "/mcp",
        lambda payload: CommandResult((OpenPickerAction("mcp"),)),
        description={"zh": "列出 MCP servers 并插入 @mcp 引用", "en": "list MCP servers and insert @mcp mentions"},
    )
    _publish_mcp_context(context, probe)
    context.epoch.on_refresh(lambda thread_id=None: _publish_mcp_context(context, probe))


def _publish_mcp_context(context, probe: McpInstructionsProbe) -> None:
    servers = discover_mcp_servers(context.project_root)
    instructions = _instructions_snapshot(probe)
    body: dict[str, object] = {
        "rule": "遇到适合任务的 MCP server 时，通过 rt.mcp helpers 初始化并检查 instructions。",
        "helper": {
            "import": "import uv_agent_runtime as rt",
            "signature": MCP_HELPER_SIGNATURE,
        },
        "server": [_server_body(server, instructions.get(server.key)) for server in servers[:10]],
    }
    if len(servers) > 10:
        body["omitted"] = len(servers) - 10
    context.epoch.publish(tag="available_mcp_servers", body=body)


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


def _instructions_snapshot(probe: McpInstructionsProbe) -> dict[tuple[str, str, str], McpInstructionsPreview]:
    return dict(probe.snapshot())


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
