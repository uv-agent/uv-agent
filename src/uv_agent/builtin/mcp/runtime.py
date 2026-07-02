from __future__ import annotations

import json
import os
import queue
import threading
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

McpTransport = Literal["stdio", "streamable_http", "sse"]


@dataclass(frozen=True)
class McpResult:
    value: Any
    raw: Any


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    transport: McpTransport
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    description: str = ""
    path: str | None = None


class McpClient:
    """Synchronous wrapper around the official async MCP SDK client."""

    def __init__(
        self,
        config: McpServerConfig,
        *,
        timeout_s: float | None = 30,
    ) -> None:
        self.config = config
        self.timeout_s = timeout_s
        self._requests: queue.Queue[tuple[str, Any, queue.Queue[Any]]] = queue.Queue()
        self._ready: queue.Queue[object] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "McpClient":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_worker, name=f"mcp-{self.config.name}", daemon=False)
        self._thread.start()
        ready = self._ready.get(timeout=self.timeout_s)
        if isinstance(ready, BaseException):
            self._thread.join(timeout=2)
            self._thread = None
            raise ready

    def close(self) -> None:
        if self._thread is None:
            return
        thread = self._thread
        self._thread = None
        response: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._requests.put(("close", None, response))
        try:
            item = response.get(timeout=self.timeout_s)
            if isinstance(item, BaseException):
                raise item
        finally:
            thread.join(timeout=2)

    def initialize(self) -> McpResult:
        return self._call("initialize")

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._call("list_tools")
        value = result.value
        if isinstance(value, dict):
            tools = value.get("tools", [])
            return tools if isinstance(tools, list) else []
        return []

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> McpResult:
        return self._call("call_tool", name, arguments or {})

    def _run_worker(self) -> None:
        try:
            import anyio

            anyio.run(self._worker)
        except BaseException as exc:
            with suppress(queue.Full):
                self._ready.put_nowait(exc)

    async def _worker(self) -> None:
        from anyio import to_thread

        async with AsyncExitStack() as stack:
            if self.config.transport == "stdio":
                if not self.config.command:
                    raise ValueError(f"MCP stdio server requires command: {self.config.name}")
                params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args,
                    env=None if self.config.env is None else dict(self.config.env),
                    cwd=self.config.cwd,
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            elif self.config.transport == "streamable_http":
                if not self.config.url:
                    raise ValueError(f"MCP streamable_http server requires url: {self.config.name}")
                read_stream, write_stream, _get_session_id = await stack.enter_async_context(
                    streamable_http_client(self.config.url)
                )
            elif self.config.transport == "sse":
                if not self.config.url:
                    raise ValueError(f"MCP sse server requires url: {self.config.name}")
                read_stream, write_stream = await stack.enter_async_context(sse_client(self.config.url))
            else:
                raise ValueError(f"Unsupported MCP transport: {self.config.transport}")

            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            self._ready.put(None)
            while True:
                op, payload, response = await to_thread.run_sync(self._requests.get)
                try:
                    if op == "close":
                        response.put(None)
                        return
                    result = await self._dispatch(session, op, payload)
                    response.put(McpResult(value=_dump_result(result), raw=result))
                except BaseException as exc:
                    response.put(exc)

    async def _dispatch(self, session: ClientSession, op: str, payload: Any) -> Any:
        operations: dict[str, Callable[[], Awaitable[Any]]] = {
            "initialize": session.initialize,
            "list_tools": session.list_tools,
            "call_tool": lambda: session.call_tool(payload[0], payload[1]),
        }
        if op not in operations:
            raise RuntimeError(f"Unknown MCP operation: {op}")
        return await operations[op]()

    def _call(self, op: str, *payload: Any) -> McpResult:
        if self._thread is None:
            self.start()
        response: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._requests.put((op, payload, response))
        item = response.get(timeout=self.timeout_s)
        if isinstance(item, BaseException):
            raise item
        return item


def connect_stdio(
    command: list[str],
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = 30,
) -> McpClient:
    """Create an official-SDK stdio MCP client for use in temporary scripts."""
    if not command:
        raise ValueError("MCP stdio command cannot be empty")
    config = McpServerConfig(
        name="stdio",
        transport="stdio",
        command=command[0],
        args=[str(arg) for arg in command[1:]],
        cwd=cwd,
        env=env,
    )
    return McpClient(config, timeout_s=timeout_s)


def connect_url(
    url: str,
    *,
    transport: McpTransport = "streamable_http",
    timeout_s: float | None = 30,
) -> McpClient:
    """Create an official-SDK HTTP/SSE MCP client for use in temporary scripts."""
    if transport not in {"streamable_http", "sse"}:
        raise ValueError(f"URL MCP transport must be streamable_http or sse, got: {transport}")
    return McpClient(
        McpServerConfig(name="url", transport=transport, url=url),
        timeout_s=timeout_s,
    )


def connect_declared(
    name: str,
    *,
    config_path: str | Path = ".agents/mcp.json",
    cwd: str | None = None,
    timeout_s: float | None = 30,
) -> McpClient:
    """Connect to a server declared in an .agents/mcp.json file."""
    config = _declared_server_config(name, Path(config_path), fallback_cwd=cwd)
    return McpClient(config, timeout_s=timeout_s)


def list_declared_servers(
    *,
    config_paths: list[str | Path] | None = None,
    cwd: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List MCP servers declared in user/project .agents/mcp.json files."""
    servers: list[dict[str, Any]] = []
    for scope, path in _candidate_config_paths(config_paths=config_paths, cwd=cwd):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_servers = data.get("servers") if isinstance(data, dict) else None
        if not isinstance(raw_servers, dict):
            continue
        for name, value in raw_servers.items():
            if isinstance(value, dict):
                config = _server_config_from_value(str(name), value, path=path, fallback_cwd=str(cwd) if cwd else None)
                servers.append(
                    {
                        "name": config.name,
                        "scope": scope,
                        "path": str(path),
                        "description": config.description,
                        "transport": config.transport,
                        "command": config.command,
                        "url": config.url,
                    }
                )
    return servers


def connect_named(
    name: str,
    *,
    config_paths: list[str | Path] | None = None,
    cwd: str | Path | None = None,
    timeout_s: float | None = 30,
) -> McpClient:
    """Connect to a named MCP server from project or user declarations."""
    for _scope, path in _candidate_config_paths(config_paths=config_paths, cwd=cwd):
        if not path.exists():
            continue
        try:
            return connect_declared(name, config_path=path, cwd=str(cwd) if cwd else None, timeout_s=timeout_s)
        except KeyError:
            continue
    raise KeyError(f"MCP server not declared: {name}")


def _dump_result(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


def _declared_server_config(name: str, path: Path, *, fallback_cwd: str | None = None) -> McpServerConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    servers = data.get("servers") if isinstance(data, dict) else None
    if not isinstance(servers, dict) or name not in servers:
        raise KeyError(f"MCP server not declared: {name}")
    server = servers[name]
    if not isinstance(server, dict):
        raise ValueError(f"Invalid MCP server declaration: {name}")
    return _server_config_from_value(name, server, path=path, fallback_cwd=fallback_cwd)


def _server_config_from_value(
    name: str,
    value: dict[str, Any],
    *,
    path: Path,
    fallback_cwd: str | None,
) -> McpServerConfig:
    url = _first_string(value, "url", "httpUrl", "serverUrl")
    command = value.get("command")
    transport = str(value.get("transport") or ("streamable_http" if url else "stdio"))
    if transport == "http":
        transport = "streamable_http"
    if transport not in {"stdio", "streamable_http", "sse"}:
        raise ValueError(f"Unsupported MCP transport for {name}: {transport}")

    args = value.get("args")
    env = value.get("env")
    server_cwd = str(value.get("cwd")) if value.get("cwd") else fallback_cwd
    description = str(value.get("description") or "")

    if transport == "stdio":
        if not isinstance(command, str):
            raise ValueError(f"MCP stdio declaration requires command: {name}")
        return McpServerConfig(
            name=name,
            transport="stdio",
            command=command,
            args=[str(arg) for arg in args] if isinstance(args, list) else [],
            cwd=server_cwd,
            env=env if isinstance(env, dict) else None,
            description=description,
            path=str(path),
        )

    if not url:
        raise ValueError(f"MCP {transport} declaration requires url: {name}")
    return McpServerConfig(
        name=name,
        transport=transport,  # type: ignore[arg-type]
        url=url,
        description=description,
        path=str(path),
    )


def _first_string(value: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item:
            return item
    return None


def _candidate_config_paths(
    *,
    config_paths: list[str | Path] | None,
    cwd: str | Path | None,
) -> list[tuple[str, Path]]:
    if config_paths is not None:
        return [(f"custom:{index}", Path(path)) for index, path in enumerate(config_paths)]
    default_root = os.environ.get("UV_AGENT_RUNTIME_PROJECT_ROOT") or Path.cwd()
    root = Path(cwd or default_root).resolve()
    home = Path(os.path.expanduser("~")).resolve()
    return [
        ("project", root / ".agents" / "mcp.json"),
        ("user", home / ".agents" / "mcp.json"),
    ]
