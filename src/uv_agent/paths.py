from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


PROJECT_LOCAL_GITIGNORE = "*\n"


def uv_agent_home() -> Path:
    """Return the user-level uv-agent home directory."""
    override = os.environ.get("UV_AGENT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".uv-agent").resolve()


def default_daemon_workspace_dir() -> Path:
    """Return the default persistent workspace used by daemon mode."""

    return uv_agent_home() / "workspace"


def project_local_dir(project_root: Path) -> Path:
    """Return the project-local uv-agent directory used for overrides only."""
    return project_root.resolve() / ".uv-agent"


def ensure_project_local_dir(project_root: Path) -> Path:
    """Create and protect the project-local uv-agent directory.

    The project-local ``.uv-agent`` tree stores machine-local config and, for
    worktrees, entire alternate checkouts. Some repositories do not ignore that
    directory themselves, so every writer that materializes it should also place
    a local ``.gitignore`` that hides all contents from the parent repository.
    Existing ignore files are left untouched so users can keep stricter local
    policies if they have already customized the directory.
    """

    directory = project_local_dir(project_root)
    directory.mkdir(parents=True, exist_ok=True)
    gitignore = directory / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(PROJECT_LOCAL_GITIGNORE, encoding="utf-8")
    return directory


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


def project_blobs_dir(project_root: Path) -> Path:
    return project_state_dir(project_root) / "blobs"


def project_runner_dir(project_root: Path) -> Path:
    return project_state_dir(project_root) / "runner"


def project_run_scripts_dir(project_root: Path) -> Path:
    return project_runner_dir(project_root) / "scripts"


def project_scriptenv_dir(project_root: Path) -> Path:
    return project_runner_dir(project_root) / "scriptenv"


def project_tui_clipboard_dir(project_root: Path) -> Path:
    return project_state_dir(project_root) / "tui" / "clipboard"
