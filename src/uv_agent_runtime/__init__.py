from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from .errors import MANAGED_RUN_ENV, install_friendly_excepthook

# Public names are resolved lazily so cheap helpers such as ``enter_dir`` do not
# pay the import cost of heavyweight optional areas like MCP or tree-sitter code
# queries on every run_python startup.  Keeping the exports table explicit makes
# the compatibility surface just as visible as eager imports while avoiding the
# global side effects of importing every submodule up front.
_EXPORTS: dict[str, tuple[str, str]] = {
    "Capture": (".codequery", "Capture"),
    "CommandError": (".errors", "CommandError"),
    "CommandTextResult": (".textops", "CommandTextResult"),
    "EditResult": (".textops", "EditResult"),
    "FileSelectionError": (".errors", "FileSelectionError"),
    "FileView": (".textops", "FileView"),
    "FriendlyErrorMixin": (".errors", "FriendlyErrorMixin"),
    "HelperRuntimeError": (".errors", "HelperRuntimeError"),
    "HelperValueError": (".errors", "HelperValueError"),
    "helper_stats_db_path": (".helper_stats", "helper_stats_db_path"),
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
    "call_host": (".transport", "call_host"),
    "resolve_host_helper": (".transport", "resolve_host_helper"),
    "convert_patch": (".textops", "convert_patch"),
    "emit_event": (".events", "emit_event"),
    "emit_progress": (".events", "emit_progress"),
    "emit_result": (".events", "emit_result"),
    "enter_dir": (".cwd", "enter_dir"),
    "edit_lines": (".textops", "edit_lines"),
    "find_files": (".codesearch", "find_files"),
    "find_symbols": (".codequery", "find_symbols"),
    "format_friendly_exception": (".errors", "format_friendly_exception"),
    "goal_paths": (".goal_mode", "goal_paths"),
    "list_declared_servers": (".mcp", "list_declared_servers"),
    "list_files": (".files", "list_files"),
    "list_thread_digests": (".threads", "list_thread_digests"),
    "look_at": (".vision", "look_at"),
    "file_lock": (".lockfile", "file_lock"),
    "make_unified_diff": (".textops", "make_unified_diff"),
    "normalize_text": (".textops", "normalize_text"),
    "path_info": (".textops", "path_info"),
    "query_code": (".codequery", "query_code"),
    "read_json": (".files", "read_json"),
    "read_file": (".textops", "read_file"),
    "read_text": (".files", "read_text"),
    "read_text_lossless": (".textops", "read_text_lossless"),
    "replace_text": (".textops", "replace_text"),
    "resolve_workspace_path": (".files", "resolve_workspace_path"),
    "restore_snapshot": (".textops", "restore_snapshot"),
    "run_process_text": (".textops", "run_process_text"),
    "run_digest": (".threads", "run_digest"),
    "run_python_env_dir": (".dependencies", "run_python_env_dir"),
    "search_text": (".codesearch", "search_text"),
    "snapshot_files": (".textops", "snapshot_files"),
    "supported_symbol_languages": (".codequery", "supported_symbol_languages"),
    "thread_digest": (".threads", "thread_digest"),
    "workspace_transaction": (".textops", "workspace_transaction"),
    "write_file": (".textops", "write_file"),
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
    "errors",
    "goal_mode",
    "helper_stats",
    "lockfile",
    "mcp",
    "patch",
    "subagent",
    "textops",
    "threads",
    "transport",
    "vision",
}

__all__ = sorted([*_EXPORTS, *_SUBMODULES])

if os.environ.get(MANAGED_RUN_ENV):
    install_friendly_excepthook()


def __getattr__(name: str) -> Any:
    """Resolve public helpers on first use instead of importing every backend."""

    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        dynamic = _dynamic_host_helper(name)
        if dynamic is not None:
            globals()[name] = dynamic
            return dynamic
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = _maybe_tracked_helper(name, getattr(module, attribute))
    globals()[name] = value
    return value


_TRACKED_HELPER_EXPORTS: frozenset[str] = frozenset(
    {
        "add_dependencies",
        "add_dependency",
        "apply_patch",
        "apply_patch_any",
        "ask",
        "call_host",
        "clear_codequery_cache",
        "compare_text",
        "connect_declared",
        "connect_named",
        "connect_stdio",
        "connect_url",
        "convert_patch",
        "emit_event",
        "emit_progress",
        "emit_result",
        "enter_dir",
        "edit_lines",
        "file_lock",
        "find_files",
        "find_symbols",
        "goal_paths",
        "list_declared_servers",
        "list_files",
        "list_thread_digests",
        "look_at",
        "make_unified_diff",
        "normalize_text",
        "path_info",
        "query_code",
        "read_file",
        "read_json",
        "read_text",
        "read_text_lossless",
        "replace_text",
        "resolve_workspace_path",
        "restore_snapshot",
        "run_digest",
        "run_process_text",
        "run_python_env_dir",
        "search_text",
        "snapshot_files",
        "supported_symbol_languages",
        "thread_digest",
        "workspace_transaction",
        "write_file",
        "write_json",
        "write_text",
        "write_text_lossless",
    }
)


def _maybe_tracked_helper(name: str, value: Any) -> Any:
    if name not in _TRACKED_HELPER_EXPORTS or not callable(value):
        return value
    helper_stats = import_module(".helper_stats", __name__)
    return helper_stats.tracked_helper(value, name=name)


def _dynamic_host_helper(name: str) -> Any:
    if not name.isidentifier() or name.startswith("_"):
        return None
    transport = import_module(".transport", __name__)
    resolved = transport.resolve_host_helper(name)
    if not resolved.get("found"):
        return None

    def helper(*args: Any, **kwargs: Any) -> Any:
        return transport.call_host(name, *args, **kwargs)

    helper.__name__ = name
    helper.__qualname__ = name
    helper.__doc__ = str(resolved.get("doc") or f"Host-provided runtime helper {name}.")
    helper_stats = import_module(".helper_stats", __name__)
    return helper_stats.tracked_helper(helper, name=name)


def __dir__() -> list[str]:
    """Include lazy exports in interactive introspection without importing them."""

    return sorted({*globals(), *__all__})
