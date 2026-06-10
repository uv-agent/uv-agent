from __future__ import annotations

import hashlib
import re
from html import escape as xml_escape
from pathlib import Path
from typing import Any

from uv_agent.config import AppConfig
from uv_agent.environment import UserLanguage, host_environment_line
from uv_agent.agent.prompts import RUNTIME_HELPERS_CONTEXT


def runtime_environment_context(
    *,
    project_root: Path,
    user_state: Path,
    project_state: Path,
    scriptenv_dir: Path,
    scriptenv_dependencies: list[str],
    host_environment: dict[str, Any],
    user_language: UserLanguage,
) -> str:
    dependencies = "\n".join(
        f"<dependency>{xml_text(dependency)}</dependency>"
        for dependency in scriptenv_dependencies
        if not _is_uv_agent_dependency(dependency)
    )
    if not dependencies:
        dependencies = '<dependency_list empty="true" />'
    return "\n".join(
        [
            "<runtime_environment>",
            f"<workspace>{xml_text(project_root)}</workspace>",
            f"<user_state>{xml_text(user_state)}</user_state>",
            f"<project_state>{xml_text(project_state)}</project_state>",
            "<run_python_environment>",
            f"<directory>{xml_text(scriptenv_dir)}</directory>",
            f"<pyproject>{xml_text(scriptenv_dir / 'pyproject.toml')}</pyproject>",
            "<rule>This is the uv project environment used by run_python; it is not the workspace or active cwd.</rule>",
            "<direct_dependencies>",
            dependencies,
            "</direct_dependencies>",
            "</run_python_environment>",
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
    ]
    workflow_default_level = config.runtime.workflow_default_level
    if workflow_default_level:
        lines.append(f"<workflow_default>{xml_text(workflow_default_level)}</workflow_default>")
    lines.append("<available>")
    for name in config.levels:
        lines.append(f"<level>{xml_text(name)}</level>")
    lines.append("</available>")
    lines.append(
        "<rule>level and model_level values are configuration-defined; use only an available name, or omit them to use the default.</rule>"
    )
    lines.append("</model_levels>")
    return "\n".join(lines)


def runtime_helpers_context() -> str:
    return RUNTIME_HELPERS_CONTEXT


def context_fingerprint(text: str) -> str:
    """Stable fingerprint for dynamic per-turn context."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def xml_text(value: object) -> str:
    return xml_escape(str(value), quote=False)


def _is_uv_agent_dependency(dependency: str) -> bool:
    match = re.match(r"^\s*([A-Za-z0-9_.-]+)", dependency)
    if match is None:
        return False
    normalized = re.sub(r"[-_.]+", "-", match.group(1)).lower()
    return normalized == "uv-agent"
