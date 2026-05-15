from __future__ import annotations

import json
from pathlib import Path

from uv_agent.mcp_config import discover_mcp_servers, render_mcp_summary


def test_discover_mcp_servers_from_agents_dir(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".agents"
    agents_dir.mkdir()
    (agents_dir / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "files": {
                        "command": "python",
                        "args": ["server.py"],
                        "description": "File helpers",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    servers = discover_mcp_servers(tmp_path)

    assert len(servers) == 1
    assert servers[0].name == "files"
    assert servers[0].command == "python server.py"
    assert "files (project)" in render_mcp_summary(servers)
