from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from uv_agent.config import ConfigError


@dataclass(frozen=True)
class DisplayError:
    title: str
    message: str
    hint: str = ""
    detail: str = ""
    severity: str = "error"


def format_error(exc: BaseException) -> DisplayError:
    """Turn internal/provider failures into compact user-facing diagnostics."""
    if isinstance(exc, ConfigError):
        return DisplayError(
            title="Configuration error",
            message=str(exc),
            hint="Open /config and check provider, model, level, and credentials.",
        )
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        preview = response.text[:800].replace("\n", " ").strip()
        return DisplayError(
            title=f"Provider HTTP {response.status_code}",
            message=response.reason_phrase or "Provider request failed",
            hint="Check the configured endpoint, API key, model name, and API format.",
            detail=preview,
        )
    if isinstance(exc, httpx.TimeoutException):
        return DisplayError(
            title="Provider timeout",
            message=str(exc) or "Provider request timed out",
            hint="Try again, lower the level, or increase provider timeout later.",
        )
    if isinstance(exc, httpx.RequestError):
        return DisplayError(
            title="Provider connection error",
            message=str(exc),
            hint="Check network connectivity and provider base_url.",
        )
    if isinstance(exc, TimeoutError):
        return DisplayError(
            title="Operation timed out",
            message=str(exc) or "The operation timed out.",
        )
    return DisplayError(title=exc.__class__.__name__, message=str(exc) or repr(exc))


def error_markup(error: DisplayError) -> str:
    lines = [f"[bold red]{error.title}[/bold red] {escape_markup(error.message)}"]
    if error.hint:
        lines.append(f"[dim]{escape_markup(error.hint)}[/dim]")
    if error.detail:
        lines.append(escape_markup(error.detail))
    return "\n".join(lines)


def escape_markup(value: Any) -> str:
    text = str(value)
    return text.replace("[", "\\[").replace("]", "\\]")
