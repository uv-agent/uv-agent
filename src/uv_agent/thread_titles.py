from __future__ import annotations


# Thread-title placeholders are needed by both the agent engine and the TUI.
# Keeping them in a tiny module lets the TUI import this constant without
# importing the full agent engine (which in turn pulls model providers and MCP
# support). The values intentionally match the historical engine constant so
# persisted thread metadata and window-title behavior stay unchanged.
DEFAULT_THREAD_TITLES = {"New thread", "new thread", "新会话"}
