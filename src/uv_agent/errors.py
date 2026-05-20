from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic
import openai

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
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        response = exc.response
        preview = str(getattr(exc, "body", "") or response.text)[:800].replace("\n", " ").strip()
        return DisplayError(
            title=f"Provider HTTP {response.status_code}",
            message=response.reason_phrase or "Provider request failed",
            hint="Check the configured endpoint, API key, model name, and API format.",
            detail=preview,
        )
    if isinstance(exc, (openai.APITimeoutError, anthropic.APITimeoutError)):
        return DisplayError(
            title="Provider timeout",
            message=str(exc) or "Provider request timed out",
            hint="Try again, lower the level, or increase provider timeout later.",
        )
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
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


def is_retryable_provider_error(exc: BaseException) -> bool:
    """Return True for transient provider/network failures that can be retried."""
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    if isinstance(
        exc,
        (
            openai.APITimeoutError,
            openai.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
        ),
    ):
        return True
    return False


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
