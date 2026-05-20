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
    assert servers[0].transport == "stdio"
    assert servers[0].endpoint == "python server.py"
    summary = render_mcp_summary(servers)
    assert '<mcp_server name="files" scope="project"' in summary
    assert f'config="{agents_dir / "mcp.json"}"' in summary
    assert ">File helpers</mcp_server>" in summary
    assert "stdio" not in summary
    assert "endpoint:" not in summary


def test_discover_mcp_servers_from_http_declaration(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".agents"
    agents_dir.mkdir()
    (agents_dir / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "web": {
                        "transport": "streamable_http",
                        "url": "http://localhost:3001/mcp",
                        "description": "Web tools",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    servers = discover_mcp_servers(tmp_path)

    assert servers[0].transport == "streamable_http"
    assert servers[0].endpoint == "http://localhost:3001/mcp"
    summary = render_mcp_summary(servers)
    assert '<mcp_server name="web" scope="project"' in summary
    assert f'config="{agents_dir / "mcp.json"}"' in summary
    assert ">Web tools</mcp_server>" in summary
    assert "streamable_http" not in summary
    assert "endpoint:" not in summary


def test_render_mcp_summary_escapes_xml_text(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".agents"
    agents_dir.mkdir()
    (agents_dir / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "web&files": {
                        "command": "python",
                        "description": "Read & write <files>",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    summary = render_mcp_summary(discover_mcp_servers(tmp_path))

    assert '<mcp_server name="web&amp;files"' in summary
    assert ">Read &amp; write &lt;files&gt;</mcp_server>" in summary
