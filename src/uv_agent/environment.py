from __future__ import annotations

import locale
import os
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any


@dataclass(frozen=True)
class UserLanguage:
    code: str
    name: str
    ui: str

    @property
    def is_chinese(self) -> bool:
        lowered = (self.code + " " + self.name).lower()
        return "zh" in lowered or "chinese" in lowered or "中文" in lowered


def detect_user_language(configured: str | None = None) -> UserLanguage:
    """Resolve the user's language from config/env/locale."""
    raw = (configured or "").strip()
    if not raw or raw.lower() == "auto":
        raw = language_from_environment()
    return normalize_language(raw)


def language_from_environment() -> str:
    for key in ("UV_AGENT_LANGUAGE", "UV_AGENT_LOCALE", "LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        loc = locale.getlocale()[0]
    except Exception:
        loc = None
    return loc or "en"


def normalize_language(value: str) -> UserLanguage:
    lowered = value.strip().replace("_", "-").lower()
    if lowered.startswith("zh") or "chinese" in lowered or "中文" in lowered:
        return UserLanguage(code=value or "zh", name="Chinese", ui="zh")
    if lowered.startswith("ja") or "japanese" in lowered:
        return UserLanguage(code=value or "ja", name="Japanese", ui="en")
    if lowered.startswith("ko") or "korean" in lowered:
        return UserLanguage(code=value or "ko", name="Korean", ui="en")
    return UserLanguage(code=value or "en", name="English", ui="en")


def host_environment() -> dict[str, Any]:
    """Return concise host facts that are stable for the current process."""
    shell = os.environ.get("ComSpec") if os.name == "nt" else os.environ.get("SHELL")
    terminal = (
        "Windows Terminal"
        if os.environ.get("WT_SESSION")
        else os.environ.get("TERM_PROGRAM") or os.environ.get("TERM") or "unknown"
    )
    return {
        "os": platform.system() or sys.platform,
        "os_release": platform.release(),
        "platform": sys.platform,
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "path_separator": os.sep,
        "shell": shell or "unknown",
        "terminal": terminal,
    }


def host_environment_line(metadata: dict[str, Any] | None = None) -> str:
    data = metadata or host_environment()
    return (
        f"{data['os']} {data['os_release']} ({data['platform']}, {data['architecture']}), "
        f"Python {data['python']}, shell={data['shell']}, path_separator={data['path_separator']!r}"
    )


def application_version(package_name: str = "uv-agent") -> str:
    """Return the installed application version for status/debug surfaces."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"
