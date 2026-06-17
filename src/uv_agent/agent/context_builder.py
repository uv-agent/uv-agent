from __future__ import annotations

import hashlib
import re
from html import escape as xml_escape
from pathlib import Path
from typing import Any

from uv_agent.config import AppConfig
from uv_agent.environment import UserLanguage, host_environment_line
from uv_agent.prompts import (
    MODEL_LEVELS_LEVEL_TEMPLATE,
    MODEL_LEVELS_RULE,
    MODEL_LEVELS_TEMPLATE,
    MODEL_LEVELS_WORKFLOW_DEFAULT_TEMPLATE,
    RUNTIME_ENVIRONMENT_DEPENDENCIES_EMPTY,
    RUNTIME_ENVIRONMENT_DEPENDENCY_TEMPLATE,
    RUNTIME_ENVIRONMENT_PERSISTENCE,
    RUNTIME_ENVIRONMENT_TEMPLATE,
    RUNTIME_ENVIRONMENT_UV_PROJECT_RULE,
    RUNTIME_HELPERS_CONTEXT,
)


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
        RUNTIME_ENVIRONMENT_DEPENDENCY_TEMPLATE.format(dependency=xml_text(dependency))
        for dependency in scriptenv_dependencies
        if not _is_uv_agent_dependency(dependency)
    )
    if not dependencies:
        dependencies = RUNTIME_ENVIRONMENT_DEPENDENCIES_EMPTY
    return RUNTIME_ENVIRONMENT_TEMPLATE.format(
        workspace=xml_text(project_root),
        user_state=xml_text(user_state),
        project_state=xml_text(project_state),
        scriptenv_dir=xml_text(scriptenv_dir),
        scriptenv_pyproject=xml_text(scriptenv_dir / "pyproject.toml"),
        uv_project_rule=RUNTIME_ENVIRONMENT_UV_PROJECT_RULE,
        dependencies=dependencies,
        host=xml_text(host_environment_line(host_environment)),
        user_language=xml_text(user_language.name),
        persistence=RUNTIME_ENVIRONMENT_PERSISTENCE,
    )


def model_levels_context(config: AppConfig) -> str:
    workflow_default = ""
    workflow_default_level = config.runtime.workflow_default_level
    if workflow_default_level:
        workflow_default = MODEL_LEVELS_WORKFLOW_DEFAULT_TEMPLATE.format(
            workflow_default=xml_text(workflow_default_level)
        )
    levels = "\n".join(
        MODEL_LEVELS_LEVEL_TEMPLATE.format(level=xml_text(name)) for name in config.public_levels()
    )
    return MODEL_LEVELS_TEMPLATE.format(
        default=xml_text(config.runtime.default_level),
        workflow_default=workflow_default,
        levels=levels,
        rule=MODEL_LEVELS_RULE,
    )


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
