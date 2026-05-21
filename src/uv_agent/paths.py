from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


def uv_agent_home() -> Path:
    """Return the user-level uv-agent home directory."""
    override = os.environ.get("UV_AGENT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".uv-agent").resolve()


def project_local_dir(project_root: Path) -> Path:
    """Return the project-local uv-agent directory used for overrides only."""
    return project_root.resolve() / ".uv-agent"


def user_config_path() -> Path:
    """Return the default user-level config path."""
    return uv_agent_home() / "config.json"


def project_config_path(project_root: Path) -> Path:
    """Return the project-local config override path."""
    return project_local_dir(project_root) / "config.json"


def project_id(project_root: Path) -> str:
    """Build a stable readable id for a workspace path."""
    root = project_root.resolve()
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", root.name).strip("-_.").lower()
    slug = slug or "workspace"
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def project_state_dir(project_root: Path) -> Path:
    """Return the user-level state directory for one workspace."""
    return uv_agent_home() / "projects" / project_id(project_root)


def project_threads_dir(project_root: Path) -> Path:
    return project_state_dir(project_root) / "threads"


def project_subthreads_dir(project_root: Path) -> Path:
    return project_state_dir(project_root) / "subthreads"


def project_attachments_dir(project_root: Path) -> Path:
    return project_state_dir(project_root) / "attachments"


def project_runner_dir(project_root: Path) -> Path:
    return project_state_dir(project_root) / "runner"


def project_run_logs_dir(project_root: Path) -> Path:
    return project_runner_dir(project_root) / "runs"


def project_scriptenv_dir(project_root: Path) -> Path:
    return project_runner_dir(project_root) / "scriptenv"


def project_tui_clipboard_dir(project_root: Path) -> Path:
    return project_state_dir(project_root) / "tui" / "clipboard"
