from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from types import TracebackType
from typing import Any

MANAGED_RUN_ENV = "UV_AGENT_RUNTIME_RUN_ID"
FULL_TRACEBACK_ENV = "UV_AGENT_RUNTIME_FULL_TRACEBACK"
_MAX_FIELD_CHARS = 500
_MAX_PREVIEW_CHARS = 4000


class FriendlyErrorMixin:
    """Mixin for helper errors that can be rendered without a Python traceback."""

    helper: str
    problem: str
    hints: tuple[str, ...]
    details: dict[str, Any]
    preview_title: str | None
    preview: str | None

    def _init_friendly(
        self,
        *,
        helper: str,
        problem: str,
        hints: tuple[str, ...] | list[str] = (),
        details: dict[str, Any] | None = None,
        preview_title: str | None = None,
        preview: str | None = None,
    ) -> None:
        self.helper = helper
        self.problem = problem
        self.hints = tuple(hint for hint in hints if hint)
        self.details = dict(details or {})
        self.preview_title = preview_title
        self.preview = _bound_text(preview, limit=_MAX_PREVIEW_CHARS) if preview else None

    def __str__(self) -> str:
        """Return a useful message even when scripts catch and print the error."""

        return format_friendly_exception(self, include_header=False, include_traceback_hint=False)


class HelperValueError(FriendlyErrorMixin, ValueError):
    """A ValueError from a runtime helper with structured recovery guidance."""

    def __init__(
        self,
        *,
        helper: str,
        problem: str,
        hints: tuple[str, ...] | list[str] = (),
        details: dict[str, Any] | None = None,
        preview_title: str | None = None,
        preview: str | None = None,
    ) -> None:
        self._init_friendly(
            helper=helper,
            problem=problem,
            hints=hints,
            details=details,
            preview_title=preview_title,
            preview=preview,
        )
        ValueError.__init__(self, problem)


class HelperRuntimeError(FriendlyErrorMixin, RuntimeError):
    """A RuntimeError from a runtime helper with structured recovery guidance."""

    def __init__(
        self,
        *,
        helper: str,
        problem: str,
        hints: tuple[str, ...] | list[str] = (),
        details: dict[str, Any] | None = None,
        preview_title: str | None = None,
        preview: str | None = None,
    ) -> None:
        self._init_friendly(
            helper=helper,
            problem=problem,
            hints=hints,
            details=details,
            preview_title=preview_title,
            preview=preview,
        )
        RuntimeError.__init__(self, problem)


class FileSelectionError(HelperValueError):
    """Raised when a text helper cannot satisfy a line-based selection strictly."""

    def __init__(self, *, partial_view: Any | None = None, **kwargs: Any) -> None:
        self.partial_view = partial_view
        super().__init__(**kwargs)


class CommandError(HelperRuntimeError):
    """Raised by CommandTextResult.raise_for_error with bounded command output."""


_original_excepthook: Any | None = None
_installed = False


def install_friendly_excepthook() -> None:
    """Install an excepthook that suppresses tracebacks for friendly helper errors."""

    global _installed, _original_excepthook
    if _installed:
        return
    _installed = True
    _original_excepthook = sys.excepthook

    def _hook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        if isinstance(exc, FriendlyErrorMixin) and not os.environ.get(FULL_TRACEBACK_ENV):
            print(format_friendly_exception(exc, tb=tb), file=sys.stderr)
            return
        assert _original_excepthook is not None
        _original_excepthook(exc_type, exc, tb)

    sys.excepthook = _hook


def format_friendly_exception(
    exc: FriendlyErrorMixin,
    *,
    tb: TracebackType | None = None,
    include_header: bool = True,
    include_traceback_hint: bool = True,
) -> str:
    """Format a helper error for model-readable stderr output."""

    lines: list[str] = []
    if include_header:
        lines.append(f"uv_agent_runtime helper error: {exc.helper}")
        callsite = _callsite_from_traceback(tb)
        if callsite:
            lines.extend(["", *callsite])
        lines.extend(["", "Problem:", f"  {exc.problem}"])
    else:
        lines.append(f"{exc.helper}: {exc.problem}")

    if exc.details:
        lines.extend(["", "Details:"])
        for key, value in exc.details.items():
            lines.append(f"  {key}: {_format_detail(value)}")

    if exc.preview:
        lines.extend(["", "Preview:"])
        prefix = "  "
        if exc.preview_title:
            lines.append(f"  {exc.preview_title.rstrip(':')}:")
            prefix = "    "
        lines.extend(f"{prefix}{line}" if line else "" for line in exc.preview.splitlines())

    if exc.hints:
        lines.extend(["", "Hints:"])
        lines.extend(f"  - {hint}" for hint in exc.hints)

    if include_header and include_traceback_hint:
        lines.extend(["", f"Set {FULL_TRACEBACK_ENV}=1 to show the full Python traceback."])
    return "\n".join(lines)


def _callsite_from_traceback(tb: TracebackType | None) -> list[str] | None:
    if tb is None:
        return None
    frames = traceback.extract_tb(tb)
    if not frames:
        return None
    candidates = [frame for frame in frames if not _is_runtime_frame(frame.filename)]
    frame = candidates[-1] if candidates else frames[-1]
    location = f"At script line {frame.lineno}"
    filename = _short_filename(frame.filename)
    if filename:
        location += f" ({filename})"
    location += ":"
    result = [location]
    if frame.line:
        result.append(f"  {frame.line.strip()}")
    return result


def _is_runtime_frame(filename: str) -> bool:
    parts = {part.lower() for part in Path(filename).parts}
    return "uv_agent_runtime" in parts


def _short_filename(filename: str) -> str:
    try:
        path = Path(filename)
        return path.name or filename
    except (OSError, ValueError):
        return filename


def _format_detail(value: Any) -> str:
    return _bound_text(repr(value), limit=_MAX_FIELD_CHARS) or ""


def _bound_text(text: str | None, *, limit: int) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 24)] + "\n...<preview truncated>"


__all__ = [
    "CommandError",
    "FileSelectionError",
    "FriendlyErrorMixin",
    "FULL_TRACEBACK_ENV",
    "HelperRuntimeError",
    "HelperValueError",
    "MANAGED_RUN_ENV",
    "format_friendly_exception",
    "install_friendly_excepthook",
]
