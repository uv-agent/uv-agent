from __future__ import annotations

from importlib import import_module
from typing import Any

# Public names are resolved lazily so cheap helpers such as ``enter_dir`` do not
# pay the import cost of heavyweight optional areas like MCP or tree-sitter code
# queries on every run_python startup.  Keeping the exports table explicit makes
# the compatibility surface just as visible as eager imports while avoiding the
# global side effects of importing every submodule up front.
_EXPORTS: dict[str, tuple[str, str]] = {
    "Capture": (".codequery", "Capture"),
    "CommandTextResult": (".textops", "CommandTextResult"),
    "Match": (".codesearch", "Match"),
    "McpResult": (".mcp", "McpResult"),
    "PatchResult": (".patch", "PatchResult"),
    "PathInfo": (".textops", "PathInfo"),
    "ReplacementResult": (".textops", "ReplacementResult"),
    "RipgrepNotFoundError": (".codesearch", "RipgrepNotFoundError"),
    "Snapshot": (".textops", "Snapshot"),
    "Submatch": (".codesearch", "Submatch"),
    "Symbol": (".codequery", "Symbol"),
    "McpClient": (".mcp", "McpClient"),
    "SubagentResult": (".subagent", "SubagentResult"),
    "TextComparison": (".textops", "TextComparison"),
    "TextFile": (".textops", "TextFile"),
    "add_dependencies": (".dependencies", "add_dependencies"),
    "add_dependency": (".dependencies", "add_dependency"),
    "apply_patch": (".patch", "apply_patch"),
    "apply_patch_any": (".textops", "apply_patch_any"),
    "ask": (".subagent", "ask"),
    "clear_codequery_cache": (".codequery", "clear_cache"),
    "compare_text": (".textops", "compare_text"),
    "connect_declared": (".mcp", "connect_declared"),
    "connect_named": (".mcp", "connect_named"),
    "connect_stdio": (".mcp", "connect_stdio"),
    "connect_url": (".mcp", "connect_url"),
    "convert_patch": (".textops", "convert_patch"),
    "emit_event": (".events", "emit_event"),
    "emit_progress": (".events", "emit_progress"),
    "emit_result": (".events", "emit_result"),
    "enter_dir": (".cwd", "enter_dir"),
    "find_files": (".codesearch", "find_files"),
    "find_symbols": (".codequery", "find_symbols"),
    "list_declared_servers": (".mcp", "list_declared_servers"),
    "list_files": (".files", "list_files"),
    "list_thread_digests": (".threads", "list_thread_digests"),
    "look_at": (".vision", "look_at"),
    "make_unified_diff": (".textops", "make_unified_diff"),
    "normalize_text": (".textops", "normalize_text"),
    "path_info": (".textops", "path_info"),
    "query_code": (".codequery", "query_code"),
    "read_json": (".files", "read_json"),
    "read_text": (".files", "read_text"),
    "read_text_lossless": (".textops", "read_text_lossless"),
    "replace_text": (".textops", "replace_text"),
    "resolve_workspace_path": (".files", "resolve_workspace_path"),
    "restore_snapshot": (".textops", "restore_snapshot"),
    "run_process_text": (".textops", "run_process_text"),
    "run_python_env_dir": (".dependencies", "run_python_env_dir"),
    "search_text": (".codesearch", "search_text"),
    "snapshot_files": (".textops", "snapshot_files"),
    "supported_symbol_languages": (".codequery", "supported_symbol_languages"),
    "thread_digest": (".threads", "thread_digest"),
    "workspace_transaction": (".textops", "workspace_transaction"),
    "write_json": (".files", "write_json"),
    "write_text": (".files", "write_text"),
    "write_text_lossless": (".textops", "write_text_lossless"),
}

# Submodules are also exposed lazily for compatibility with code that does
# ``from uv_agent_runtime import codequery`` before reaching into internals in
# tests or one-off scripts.
_SUBMODULES = {
    "codequery",
    "codesearch",
    "cwd",
    "dependencies",
    "events",
    "files",
    "mcp",
    "patch",
    "subagent",
    "textops",
    "threads",
    "vision",
}

__all__ = sorted([*_EXPORTS, *_SUBMODULES])


def __getattr__(name: str) -> Any:
    """Resolve public helpers on first use instead of importing every backend."""

    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in interactive introspection without importing them."""

    return sorted({*globals(), *__all__})
