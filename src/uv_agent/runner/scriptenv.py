from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import tomllib
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

import tomlkit
from tomlkit.exceptions import TOMLKitError


_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
_READY_LOCK = threading.Lock()
_READY_DIRS: set[Path] = set()
_HOST_PACKAGE = "uv-agent"

# Default Python version for the shared scriptenv. uv will download a matching
# interpreter if none is available on the host.
DEFAULT_PYTHON_VERSION = "3.12"


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
        _ensure_runtime_version(resolved, python)
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
                "--python",
                DEFAULT_PYTHON_VERSION,
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
    args = [uv_binary(), "add", "--project", str(scriptenv_dir), "-q"]
    checkout = _editable_checkout_root()
    if checkout is not None:
        args.extend(["--editable", str(checkout)])
    else:
        args.append("uv-agent")
    subprocess.run(args, env=_uv_env(), check=True)


def _ensure_runtime_version(scriptenv_dir: Path, python: Path) -> None:
    target = _host_runtime_version()
    if not target:
        return
    installed = _installed_runtime_version(python)
    if not installed or installed == target:
        return
    pyproject = scriptenv_dir / "pyproject.toml"
    try:
        original = pyproject.read_text(encoding="utf-8")
    except OSError:
        return
    if not _pin_dependency(pyproject, _HOST_PACKAGE, target):
        return
    result = subprocess.run(
        [
            uv_binary(),
            "sync",
            "--project",
            str(scriptenv_dir),
            "-q",
        ],
        env=_uv_env(),
    )
    if result.returncode != 0:
        # Restore the previous pyproject so the scriptenv is not left in a
        # broken state (e.g. host pins a version not yet on the index).
        pyproject.write_text(original, encoding="utf-8")


def _pin_dependency(pyproject: Path, package: str, version: str) -> bool:
    try:
        document = tomlkit.parse(pyproject.read_text(encoding="utf-8"))
    except (OSError, TOMLKitError):
        return False
    project = document.get("project")
    if project is None:
        return False
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list):
        return False
    normalized = _normalize_name(package)
    pinned = f"{package}=={version}"
    changed = False
    for index, entry in enumerate(dependencies):
        if not isinstance(entry, str):
            continue
        match = _REQ_NAME_RE.match(entry)
        if not match or _normalize_name(match.group(1)) != normalized:
            continue
        if entry == pinned:
            continue
        dependencies[index] = pinned
        changed = True
    if not changed:
        return False
    pyproject.write_text(tomlkit.dumps(document), encoding="utf-8")
    return True


def _host_runtime_version() -> str | None:
    try:
        return _pkg_version(_HOST_PACKAGE)
    except PackageNotFoundError:
        return None


def _installed_runtime_version(python: Path) -> str | None:
    code = (
        "from importlib.metadata import PackageNotFoundError, version\n"
        "try:\n"
        f"    print(version({_HOST_PACKAGE!r}))\n"
        "except PackageNotFoundError:\n"
        "    pass\n"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    stdout = result.stdout or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "replace")
    return stdout.strip() or None


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



def _editable_checkout_root() -> Path | None:
    """Return the repository root when running from a source checkout."""

    root = Path(__file__).resolve().parents[3]
    if (root / "pyproject.toml").exists() and (root / "src" / "uv_agent_runtime").exists():
        return root
    return None


def _uv_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    return env
