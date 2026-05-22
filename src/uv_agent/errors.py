from __future__ import annotations

from dataclasses import dataclass
from rich.text import Text

from uv_agent.config import ConfigError


class EmptyModelStreamError(RuntimeError):
    """Raised when a provider stream ends before any usable model output arrives."""


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
    provider_kind = _provider_error_kind(exc)
    if provider_kind == "status":
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", getattr(exc, "status_code", "?"))
        reason = getattr(response, "reason_phrase", "") or "Provider request failed"
        response_text = getattr(response, "text", "")
        preview = str(getattr(exc, "body", "") or response_text)[:800].replace("\n", " ").strip()
        return DisplayError(
            title=f"Provider HTTP {status_code}",
            message=reason,
            hint="Check the configured endpoint, API key, model name, and API format.",
            detail=preview,
        )
    if provider_kind == "timeout":
        return DisplayError(
            title="Provider timeout",
            message=str(exc) or "Provider request timed out",
            hint="Try again, lower the level, or increase provider timeout later.",
        )
    if provider_kind == "connection":
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
    if isinstance(exc, EmptyModelStreamError):
        return True
    provider_kind = _provider_error_kind(exc)
    if provider_kind == "status":
        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int):
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600)
    if provider_kind in {"timeout", "connection"}:
        return True
    return False


def _provider_error_kind(exc: BaseException) -> str:
    """Classify OpenAI/Anthropic errors without importing provider SDKs.

    The concrete SDK exception classes live in large packages. Importing those
    packages merely to format an error made TUI startup pay provider import cost
    even before any model request. Real SDK errors carry distinctive class names,
    module paths, and base classes; walking the MRO keeps subclasses such as
    ``openai.InternalServerError`` equivalent to the old ``isinstance`` checks
    without loading the SDK during startup.
    """

    for cls in exc.__class__.__mro__:
        module = cls.__module__.lower()
        if not (module == "openai" or module.startswith("openai.") or module.startswith("anthropic")):
            continue
        name = cls.__name__
        if name == "APITimeoutError":
            return "timeout"
        if name == "APIConnectionError":
            return "connection"
        if name == "APIStatusError":
            return "status"
    return ""


def error_renderable(error: DisplayError) -> Text:
    """Build a styled error block without parsing external text as markup."""
    first = Text()
    first.append(error.title, style="bold red")
    first.append(" ")
    first.append(error.message)
    lines = [first]
    if error.hint:
        lines.append(Text(error.hint, style="dim"))
    if error.detail:
        lines.append(Text(error.detail))
    result = Text()
    for index, line in enumerate(lines):
        if index:
            result.append("\n")
        result.append_text(line)
    return result
