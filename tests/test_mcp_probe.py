from __future__ import annotations

import json
import sys
from pathlib import Path

from uv_agent.builtin.mcp.config import discover_mcp_servers
from uv_agent.builtin.mcp.probe import (
    MCP_INSTRUCTIONS_PREVIEW_CHARS,
    _probe_server,
)


def test_probe_server_collects_truncated_instructions(tmp_path: Path) -> None:
    server = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    long_instructions = "x" * (MCP_INSTRUCTIONS_PREVIEW_CHARS + 5)
    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "echo": {
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [str(server)],
                        "env": {"UV_AGENT_TEST_MCP_INSTRUCTIONS": long_instructions},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    server_summary = discover_mcp_servers(tmp_path)[0]

    result = _probe_server(server_summary, cwd=tmp_path)

    assert result is not None
    assert result.key == server_summary.key
    assert result.instructions.truncated is True
    assert result.instructions.text == long_instructions[:MCP_INSTRUCTIONS_PREVIEW_CHARS]


def test_probe_server_silently_ignores_failed_server(tmp_path: Path) -> None:
    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "broken": {
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": ["does-not-exist.py"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    server_summary = discover_mcp_servers(tmp_path)[0]

    assert _probe_server(server_summary, cwd=tmp_path) is None
