from __future__ import annotations

import asyncio
import json
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from uv_agent.mcp_config import McpInstructionsPreview, McpServerSummary, discover_mcp_servers

MCP_INSTRUCTIONS_PREVIEW_CHARS = 500
MCP_PROBE_SERVER_TIMEOUT_S = 5.0

McpProbeTransport = Literal["stdio", "streamable_http", "sse"]


@dataclass(frozen=True)
class McpProbeResult:
    key: tuple[str, str, str]
    instructions: McpInstructionsPreview


@dataclass(frozen=True)
class McpProbeServerConfig:
    name: str
    transport: McpProbeTransport
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    cwd: str | None = None
    env: Mapping[str, str] | None = None


class McpInstructionsProbe:
    """Best-effort host-side MCP initialize probe for prompt context."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._lock = threading.Lock()
        self._started = False
        self._thread: threading.Thread | None = None
        self._instructions: dict[tuple[str, str, str], McpInstructionsPreview] = {}

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="mcp-instructions-probe",
                daemon=True,
            )
            self._thread.start()

    def snapshot(self) -> dict[tuple[str, str, str], McpInstructionsPreview]:
        with self._lock:
            return dict(self._instructions)

    def _run(self) -> None:
        try:
            servers = discover_mcp_servers(self.project_root)
        except Exception:
            return
        for server in servers:
            result = _probe_server(server, cwd=self.project_root)
            if result is None:
                continue
            with self._lock:
                self._instructions[result.key] = result.instructions


def _probe_server(server: McpServerSummary, *, cwd: Path) -> McpProbeResult | None:
    try:
        config = _probe_config(server, fallback_cwd=str(cwd))
        init = asyncio.run(_initialize_server(config))
    except Exception:
        return None
    instructions = _extract_instructions(init)
    if not instructions:
        return None
    return McpProbeResult(key=server.key, instructions=_preview_instructions(instructions))


async def _initialize_server(config: McpProbeServerConfig) -> Any:
    # The MCP SDK imports a sizeable stack. The probe already runs in a daemon
    # thread and only needs the SDK when a declared server is actually probed,
    # so defer these imports to keep normal TUI startup lightweight.
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client

    async def run_initialize() -> Any:
        async with AsyncExitStack() as stack:
            if config.transport == "stdio":
                if not config.command:
                    raise ValueError(f"MCP stdio server requires command: {config.name}")
                params = StdioServerParameters(
                    command=config.command,
                    args=list(config.args),
                    env=None if config.env is None else dict(config.env),
                    cwd=config.cwd,
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            elif config.transport == "streamable_http":
                if not config.url:
                    raise ValueError(f"MCP streamable_http server requires url: {config.name}")
                read_stream, write_stream, _get_session_id = await stack.enter_async_context(
                    streamable_http_client(config.url)
                )
            elif config.transport == "sse":
                if not config.url:
                    raise ValueError(f"MCP sse server requires url: {config.name}")
                read_stream, write_stream = await stack.enter_async_context(sse_client(config.url))
            else:
                raise ValueError(f"Unsupported MCP transport: {config.transport}")

            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            result = await session.initialize()
            return _dump_result(result)

    # TODO: Consider a short-lived host-side MCP client pool so repeated user
    # turns can reuse recently initialized servers instead of restarting them.
    return await asyncio.wait_for(run_initialize(), timeout=MCP_PROBE_SERVER_TIMEOUT_S)


def _probe_config(server: McpServerSummary, *, fallback_cwd: str) -> McpProbeServerConfig:
    data = json.loads(server.path.read_text(encoding="utf-8"))
    raw_servers = data.get("servers") if isinstance(data, dict) else None
    if not isinstance(raw_servers, dict):
        raise ValueError("MCP config has no servers object")
    value = raw_servers.get(server.name)
    if not isinstance(value, dict):
        raise ValueError(f"MCP server not declared: {server.name}")

    url = _first_string(value, "url", "httpUrl", "serverUrl")
    transport = _probe_transport(value.get("transport"), has_url=bool(url), server_name=server.name)

    command = value.get("command")
    args = value.get("args")
    env = value.get("env")
    server_cwd = str(value.get("cwd")) if value.get("cwd") else fallback_cwd
    if transport == "stdio":
        if not isinstance(command, str):
            raise ValueError(f"MCP stdio declaration requires command: {server.name}")
        return McpProbeServerConfig(
            name=server.name,
            transport="stdio",
            command=command,
            args=tuple(str(arg) for arg in args) if isinstance(args, list) else (),
            cwd=server_cwd,
            env={str(key): str(value) for key, value in env.items()} if isinstance(env, dict) else None,
        )

    if not url:
        raise ValueError(f"MCP {transport} declaration requires url: {server.name}")
    return McpProbeServerConfig(
        name=server.name,
        transport=transport,
        url=url,
    )


def _probe_transport(value: object, *, has_url: bool, server_name: str) -> McpProbeTransport:
    """Normalize MCP transport aliases into the probe transport literal set."""

    transport = str(value or ("streamable_http" if has_url else "stdio"))
    if transport == "http":
        transport = "streamable_http"
    if transport == "stdio":
        return "stdio"
    if transport == "streamable_http":
        return "streamable_http"
    if transport == "sse":
        return "sse"
    raise ValueError(f"Unsupported MCP transport for {server_name}: {transport}")


def _first_string(value: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item:
            return item
    return None


def _dump_result(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


def _extract_instructions(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    instructions = value.get("instructions")
    return instructions if isinstance(instructions, str) else ""


def _preview_instructions(text: str) -> McpInstructionsPreview:
    stripped = text.strip()
    if len(stripped) <= MCP_INSTRUCTIONS_PREVIEW_CHARS:
        return McpInstructionsPreview(stripped, truncated=False)
    return McpInstructionsPreview(
        stripped[:MCP_INSTRUCTIONS_PREVIEW_CHARS],
        truncated=True,
    )
