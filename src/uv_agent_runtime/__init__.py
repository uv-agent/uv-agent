from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from .errors import (
    CommandError,
    FileSelectionError,
    FriendlyErrorMixin,
    HelperRuntimeError,
    HelperValueError,
    MANAGED_RUN_ENV,
    install_friendly_excepthook,
)
from .facade import (
    CaptureResults,
    CollectionResult,
    File,
    FileSet,
    SearchResults,
    SelectionError,
    SymbolResults,
    apply_patch,
    cd,
    compare,
    deps,
    diff,
    dry_run_patch,
    events,
    file,
    files,
    look_at,
    mcp,
    normalize,
    patch,
    convert_patch,
    path,
    pwd,
    query,
    restore,
    run,
    search,
    snapshot,
    symbols,
    threads,
    transaction,
)
from .textops import (
    CommandTextResult,
    EditResult,
    FileView,
    PathInfo,
    ReplacementResult,
    Snapshot,
    TextComparison,
    TextFile,
)
from .codesearch import FffSearchNotAvailableError, Match, RipgrepNotFoundError, Submatch
from .codequery import Capture, Symbol
from .lockfile import HeldFileLock
from .mcp import McpClient, McpResult, McpServerConfig
from .patch import PatchResult
from .threads import (
    BoundedText,
    ConversationMessage,
    HelperCall,
    ProcessDetail,
    ProcessRef,
    RunEventDetail,
    ThreadCompaction,
    ThreadCompactionSummary,
    ThreadDetailResult,
    ThreadDigest,
    ThreadDigestItem,
    ThreadEpoch,
    ThreadTurn,
    ThreadView,
)
from . import workflow as workflow
from . import scheduler as scheduler

if os.environ.get(MANAGED_RUN_ENV):
    install_friendly_excepthook()

__all__ = [
    "BoundedText",
    "Capture",
    "CaptureResults",
    "CollectionResult",
    "CommandError",
    "CommandTextResult",
    "ConversationMessage",
    "EditResult",
    "File",
    "FileSelectionError",
    "FileSet",
    "FileView",
    "FriendlyErrorMixin",
    "HeldFileLock",
    "HelperCall",
    "HelperRuntimeError",
    "HelperValueError",
    "Match",
    "McpClient",
    "McpResult",
    "McpServerConfig",
    "PatchResult",
    "PathInfo",
    "ProcessDetail",
    "ProcessRef",
    "ReplacementResult",
    "FffSearchNotAvailableError",
    "RipgrepNotFoundError",
    "RunEventDetail",
    "SearchResults",
    "SelectionError",
    "Snapshot",
    "Submatch",
    "Symbol",
    "SymbolResults",
    "TextComparison",
    "TextFile",
    "ThreadCompaction",
    "ThreadCompactionSummary",
    "ThreadDetailResult",
    "ThreadDigest",
    "ThreadDigestItem",
    "ThreadEpoch",
    "ThreadTurn",
    "ThreadView",
    "apply_patch",
    "cd",
    "compare",
    "deps",
    "diff",
    "dry_run_patch",
    "events",
    "file",
    "files",
    "look_at",
    "mcp",
    "normalize",
    "patch",
    "convert_patch",
    "path",
    "pwd",
    "query",
    "restore",
    "run",
    "search",
    "scheduler",
    "snapshot",
    "symbols",
    "threads",
    "transaction",
    "workflow",
]


def __getattr__(name: str) -> Any:
    """Resolve plugin-provided runtime helpers without keeping old helper exports."""

    dynamic = _dynamic_host_helper(name)
    if dynamic is not None:
        globals()[name] = dynamic
        return dynamic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _HostNamespaceProxy:
    """Lazy proxy for one plugin-provided runtime helper namespace."""

    def __init__(self, namespace: str, resolved: dict[str, Any]) -> None:
        self.__name__ = namespace
        self.__doc__ = str(resolved.get("doc") or f"Host-provided runtime namespace {namespace}.")
        self._namespace = namespace
        self._functions = {
            str(item.get("name")): item
            for item in resolved.get("functions", [])
            if isinstance(item, dict) and item.get("name")
        }

    def __getattr__(self, name: str) -> Any:
        if not name.isidentifier() or name.startswith("_"):
            raise AttributeError(name)
        full_name = f"{self._namespace}.{name}"
        transport = import_module(".transport", __name__)
        resolved = transport.resolve_host_helper(full_name)
        if not resolved.get("found") and name not in self._functions:
            raise AttributeError(name)
        helper = _host_function(full_name, resolved or self._functions.get(name) or {})
        setattr(self, name, helper)
        return helper

    def __dir__(self) -> list[str]:
        return sorted({*self.__dict__, *self._functions})


def _dynamic_host_helper(name: str) -> Any:
    if not name.isidentifier() or name.startswith("_"):
        return None
    transport = import_module(".transport", __name__)
    resolved = transport.resolve_host_helper(name)
    if not resolved.get("found"):
        return None
    kind = resolved.get("kind")
    if kind == "namespace":
        if resolved.get("transport") == "local_module" and resolved.get("module"):
            return import_module(str(resolved["module"]))
        return _HostNamespaceProxy(name, resolved)
    if kind == "function" and resolved.get("transport") == "local_module" and resolved.get("module"):
        module = import_module(str(resolved["module"]))
        return getattr(module, str(resolved.get("function") or name.rpartition(".")[2]))
    return _host_function(str(resolved.get("name") or name), resolved)


def _host_function(full_name: str, resolved: dict[str, Any]) -> Any:
    transport = import_module(".transport", __name__)

    def helper(*args: Any, **kwargs: Any) -> Any:
        return transport.call_host(full_name, *args, **kwargs)

    helper.__name__ = full_name.rpartition(".")[2] or full_name
    helper.__qualname__ = full_name
    helper.__doc__ = str(resolved.get("doc") or f"Host-provided runtime helper {full_name}.")
    helper_tracking = import_module(".helper_tracking", __name__)
    return helper_tracking.tracked_helper(helper, name=full_name)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
