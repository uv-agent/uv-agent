from __future__ import annotations

import hashlib
from html import escape as xml_escape
from pathlib import Path
from typing import Any

from uv_agent.config import AppConfig
from uv_agent.environment import UserLanguage, host_environment_line
from uv_agent.prompts import RUNTIME_HELPERS_CONTEXT


def runtime_environment_context(
    *,
    project_root: Path,
    user_state: Path,
    project_state: Path,
    host_environment: dict[str, Any],
    user_language: UserLanguage,
) -> str:
    return "\n".join(
        [
            "<runtime_environment>",
            f"<workspace>{xml_text(project_root)}</workspace>",
            f"<user_state>{xml_text(user_state)}</user_state>",
            f"<project_state>{xml_text(project_state)}</project_state>",
            f"<host>{xml_text(host_environment_line(host_environment))}</host>",
            f"<user_language>{xml_text(user_language.name)}</user_language>",
            "<persistence>Persisted scripts, runs, and threads live under the project state directory.</persistence>",
            "</runtime_environment>",
        ]
    )


def model_levels_context(config: AppConfig) -> str:
    lines = [
        "<model_levels>",
        f"<default>{xml_text(config.runtime.default_level)}</default>",
        "<available>",
    ]
    for name in config.levels:
        lines.append(f"<level>{xml_text(name)}</level>")
    lines.extend(
        [
            "</available>",
            "<rule>level and model_level values are configuration-defined; use only an available name, or omit them to use the default.</rule>",
            "</model_levels>",
        ]
    )
    return "\n".join(lines)


def runtime_helpers_context() -> str:
    return RUNTIME_HELPERS_CONTEXT


def context_fingerprint(text: str) -> str:
    """Stable fingerprint for dynamic per-turn context."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def xml_text(value: object) -> str:
    return xml_escape(str(value), quote=False)
