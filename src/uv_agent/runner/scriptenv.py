from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def uv_binary() -> str:
    return shutil.which("uv") or "uv"


def ensure_venv(scriptenv_dir: Path) -> Path:
    venv_dir = scriptenv_dir / ".venv"
    python = _venv_python(venv_dir)
    if not python.exists():
        scriptenv_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([uv_binary(), "venv", str(venv_dir)], check=True)
    _ensure_runtime_package(python)
    return python


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_runtime_package(python: Path) -> None:
    probe = subprocess.run(
        [str(python), "-c", "import uv_agent_runtime"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        return
    subprocess.run(
        [
            uv_binary(),
            "pip",
            "install",
            "--python",
            str(python),
            "-q",
            _runtime_package_spec(),
        ],
        check=True,
    )


def _runtime_package_spec() -> str:
    root = Path(__file__).resolve().parents[3]
    if (root / "pyproject.toml").exists() and (root / "src" / "uv_agent_runtime").exists():
        return str(root)
    return "uv-agent"
