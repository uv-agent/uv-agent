from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import tomllib
from pathlib import Path


_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
_READY_LOCK = threading.Lock()
_READY_DIRS: set[Path] = set()


def uv_binary() -> str:
    return shutil.which("uv") or "uv"


def ensure_venv(scriptenv_dir: Path) -> Path:
    resolved = scriptenv_dir.resolve()
    python = _venv_python(resolved / ".venv")
    if resolved in _READY_DIRS and python.exists():
        return python
    with _READY_LOCK:
        if resolved in _READY_DIRS and python.exists():
            return python
        ensure_project(resolved)
        _ensure_runtime_package(resolved, python)
        _READY_DIRS.add(resolved)
        return python


def direct_dependencies(scriptenv_dir: Path) -> list[str]:
    pyproject = scriptenv_dir / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    dependencies = data.get("project", {}).get("dependencies", [])
    if not isinstance(dependencies, list):
        return []
    return [dependency for dependency in dependencies if isinstance(dependency, str)]


def ensure_project(scriptenv_dir: Path) -> None:
    venv_dir = scriptenv_dir / ".venv"
    pyproject = scriptenv_dir / "pyproject.toml"
    if not pyproject.exists():
        scriptenv_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                uv_binary(),
                "init",
                "-q",
                "--bare",
                "--name",
                "uv-agent-scriptenv",
                "--no-pin-python",
                str(scriptenv_dir),
            ],
            env=_uv_env(),
            check=True,
        )
    if not venv_dir.exists() or not _declares_dependency(pyproject, "uv-agent"):
        _add_runtime_package(scriptenv_dir)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_runtime_package(scriptenv_dir: Path, python: Path) -> None:
    probe = subprocess.run(
        [str(python), "-c", "import uv_agent_runtime"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        return
    _add_runtime_package(scriptenv_dir)


def _add_runtime_package(scriptenv_dir: Path) -> None:
    subprocess.run(
        [
            uv_binary(),
            "add",
            "--project",
            str(scriptenv_dir),
            "-q",
            "uv-agent",
        ],
        env=_uv_env(),
        check=True,
    )


def _declares_dependency(pyproject: Path, name: str) -> bool:
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    dependencies = data.get("project", {}).get("dependencies", [])
    if not isinstance(dependencies, list):
        return False
    normalized = _normalize_name(name)
    for dependency in dependencies:
        if not isinstance(dependency, str):
            continue
        match = _REQ_NAME_RE.match(dependency)
        if match and _normalize_name(match.group(1)) == normalized:
            return True
    return False


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _uv_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    return env
