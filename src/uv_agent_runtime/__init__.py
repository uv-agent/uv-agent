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
    goals,
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
from .goal_mode import RuntimeGoalPaths
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
    "goals",
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
    helper_tracking = import_module(".helper_tracking", __name__)
    return helper_tracking.tracked_helper(helper, name=name)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
