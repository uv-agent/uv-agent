from __future__ import annotations

import itertools
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class McpResult:
    value: Any
    raw: dict[str, Any]


@dataclass
class McpStdioClient:
    command: list[str]
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    timeout_s: float | None = 30
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(1), init=False)
    _process: subprocess.Popen[str] | None = field(default=None, init=False)

    def __enter__(self) -> "McpStdioClient":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def start(self) -> None:
        """Start the MCP stdio server process."""
        if self._process is not None:
            return
        self._process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=None if self.env is None else dict(self.env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    def close(self) -> None:
        """Terminate the MCP server process."""
        if self._process is None:
            return
        process = self._process
        self._process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def initialize(
        self,
        *,
        client_name: str = "uv-agent-runtime",
        client_version: str = "0.1.0",
    ) -> McpResult:
        """Send MCP initialize and initialized messages."""
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": client_version},
            },
        )
        self.notify("notifications/initialized", {})
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tools exposed by the MCP server."""
        result = self.request("tools/list", {})
        value = result.value
        if isinstance(value, dict):
            tools = value.get("tools", [])
            return tools if isinstance(tools, list) else []
        return []

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> McpResult:
        """Call a tool exposed by the MCP server."""
        return self.request("tools/call", {"name": name, "arguments": arguments or {}})

    def request(self, method: str, params: dict[str, Any] | None = None) -> McpResult:
        """Send a JSON-RPC request and wait for the response."""
        process = self._ensure_process()
        request_id = next(self._ids)
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        self._write(payload)
        while True:
            line = process.stdout.readline() if process.stdout else ""
            if line == "":
                raise RuntimeError("MCP server closed stdout before response")
            message = json.loads(line)
            if message.get("id") != request_id:
                continue
            if message.get("error"):
                raise RuntimeError(f"MCP request failed: {message['error']}")
            return McpResult(value=message.get("result"), raw=message)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification."""
        self._ensure_process()
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            self.start()
        if self._process is None:
            raise RuntimeError("MCP process did not start")
        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"MCP server exited with {self._process.returncode}: {stderr}")
        return self._process

    def _write(self, payload: dict[str, Any]) -> None:
        process = self._ensure_process()
        if process.stdin is None:
            raise RuntimeError("MCP server stdin is not available")
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()


def connect_stdio(
    command: list[str],
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = 30,
) -> McpStdioClient:
    """Create an MCP stdio client for use in temporary scripts."""
    return McpStdioClient(command=command, cwd=cwd, env=env, timeout_s=timeout_s)


def connect_declared(
    name: str,
    *,
    config_path: str | Path = ".agents/mcp.json",
    cwd: str | None = None,
    timeout_s: float | None = 30,
) -> McpStdioClient:
    """Connect to a server declared in an .agents/mcp.json file."""
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    servers = data.get("servers") if isinstance(data, dict) else None
    if not isinstance(servers, dict) or name not in servers:
        raise KeyError(f"MCP server not declared: {name}")
    server = servers[name]
    if not isinstance(server, dict):
        raise ValueError(f"Invalid MCP server declaration: {name}")
    command = server.get("command")
    if not isinstance(command, str):
        raise ValueError(f"MCP server declaration requires command: {name}")
    args = server.get("args")
    argv = [command, *[str(arg) for arg in args]] if isinstance(args, list) else [command]
    env = server.get("env")
    server_cwd = str(server.get("cwd")) if server.get("cwd") else cwd
    return connect_stdio(
        argv,
        cwd=server_cwd,
        env=env if isinstance(env, dict) else None,
        timeout_s=timeout_s,
    )
