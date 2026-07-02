from __future__ import annotations

from uv_agent.plugins.i18n import LocalizedText


TEXTS: dict[str, LocalizedText] = {
    "mcp": {"en": "MCP", "zh": "MCP"},
    "mention_mcp_hint": {
        "en": "Search and Enter to insert an MCP mention",
        "zh": "搜索后按 Enter 插入 MCP 引用",
    },
    "no_mcp": {
        "en": "no .agents/mcp.json servers declared",
        "zh": "没有声明 .agents/mcp.json server",
    },
}
