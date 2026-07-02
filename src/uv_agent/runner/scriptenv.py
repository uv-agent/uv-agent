from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import tomllib
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Sequence

import tomlkit
from tomlkit.exceptions import TOMLKitError

from uv_agent_runtime.lockfile import file_lock


_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
_READY_LOCK = threading.Lock()
_READY_DIRS: set[tuple[Path, str | None]] = set()
_HOST_PACKAGE = "uv-agent"

# Default Python version for the shared scriptenv. uv will download a matching
# interpreter if none is available on the host.
DEFAULT_PYTHON_VERSION = "3.12"


def uv_binary() -> str:
    return shutil.which("uv") or "uv"


def ensure_venv(scriptenv_dir: Path, *, index_url: str | None = None) -> Path:
    resolved = scriptenv_dir.resolve()
    requested_index_url = _normalize_index_url(index_url)
    ready_key = (resolved, requested_index_url)
    python = _venv_python(resolved / ".venv")
    if ready_key in _READY_DIRS and python.exists():
        return python
    with _READY_LOCK:
        if ready_key in _READY_DIRS and python.exists():
            return python
        # Multiple uv-agent processes can share one project scriptenv. Serialize
        # uv init/add/sync so concurrent runner subprocesses do not corrupt
        # pyproject.toml or uv.lock while the environment is being prepared.
        with _scriptenv_lock(resolved):
            ensure_project(resolved, index_url=requested_index_url)
            _ensure_runtime_package(resolved, python)
            _ensure_runtime_version(resolved, python)
            _ensure_lock_current(resolved)
            _READY_DIRS.add(ready_key)
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


def ensure_project(scriptenv_dir: Path, *, index_url: str | None = None) -> None:
    venv_dir = scriptenv_dir / ".venv"
    pyproject = scriptenv_dir / "pyproject.toml"
    if not pyproject.exists():
        scriptenv_dir.mkdir(parents=True, exist_ok=True)
        _run_uv(
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
            check=True,
        )
    if index_url:
        _ensure_default_index(pyproject, index_url)
    checkout = _editable_checkout_root()
    needs_runtime = not venv_dir.exists() or not _declares_dependency(pyproject, "uv-agent")
    needs_checkout_source = checkout is not None and not _declares_editable_checkout(pyproject, checkout)
    if needs_runtime or needs_checkout_source:
        _add_runtime_package(scriptenv_dir)
    if checkout is None:
        original = _read_optional_text(pyproject)
        original_lock = _read_optional_bytes(scriptenv_dir / "uv.lock")
        if original is not None and _remove_dependency_source(pyproject, _HOST_PACKAGE):
            _lock_and_sync_project(scriptenv_dir, original_pyproject=original, original_lock=original_lock)


def _normalize_index_url(index_url: str | None) -> str | None:
    if not index_url:
        return None
    normalized = index_url.strip()
    return normalized or None


def _scriptenv_lock(scriptenv_dir: Path):
    return file_lock(scriptenv_dir / ".uv-agent-scriptenv.lock", timeout_s=300.0)


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
    _run_uv(args, check=True)


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
    original_lock = _read_optional_bytes(scriptenv_dir / "uv.lock")
    if not _pin_dependency(pyproject, _HOST_PACKAGE, target):
        return
    _lock_and_sync_project(scriptenv_dir, original_pyproject=original, original_lock=original_lock)


def _ensure_lock_current(scriptenv_dir: Path) -> None:
    pyproject = scriptenv_dir / "pyproject.toml"
    if not pyproject.exists():
        return
    result = _run_uv(
        [
            uv_binary(),
            "lock",
            "--project",
            str(scriptenv_dir),
            "--check",
            "-q",
        ]
    )
    if result.returncode == 0:
        return
    _lock_and_sync_project(
        scriptenv_dir,
        original_pyproject=_read_optional_text(pyproject),
        original_lock=_read_optional_bytes(scriptenv_dir / "uv.lock"),
    )


def _lock_and_sync_project(
    scriptenv_dir: Path,
    *,
    original_pyproject: str | None,
    original_lock: bytes | None,
) -> bool:
    lock = scriptenv_dir / "uv.lock"
    lock_result = _run_uv([uv_binary(), "lock", "--project", str(scriptenv_dir), "-q"])
    if lock_result.returncode != 0:
        _restore_project_files(scriptenv_dir / "pyproject.toml", original_pyproject, lock, original_lock)
        return False
    sync_result = _run_uv(
        [
            uv_binary(),
            "sync",
            "--project",
            str(scriptenv_dir),
            "--locked",
            "-q",
        ]
    )
    if sync_result.returncode != 0:
        # Restore the previous project metadata so the scriptenv is not left in
        # a broken state (e.g. the host pins a version not yet on the index).
        _restore_project_files(scriptenv_dir / "pyproject.toml", original_pyproject, lock, original_lock)
        return False
    return True


def _ensure_default_index(pyproject: Path, index_url: str) -> bool:
    """Ensure the managed scriptenv pyproject uses the configured default index."""

    normalized_url = index_url.strip()
    if not normalized_url:
        return False
    try:
        original = pyproject.read_text(encoding="utf-8")
        document = tomlkit.parse(original)
    except (OSError, TOMLKitError):
        return False

    tool = document.get("tool")
    if not isinstance(tool, dict):
        tool = tomlkit.table()
        document["tool"] = tool
    uv = tool.get("uv")
    if not isinstance(uv, dict):
        uv = tomlkit.table()
        tool["uv"] = uv
    existing_indexes = uv.get("index")
    valid_indexes = [item for item in existing_indexes if isinstance(item, dict)] if isinstance(existing_indexes, list) else []
    target = next((item for item in valid_indexes if item.get("url") == normalized_url), None)
    if target is None:
        target = next((item for item in valid_indexes if item.get("default") is True), None)

    indexes = tomlkit.aot()
    for item in valid_indexes:
        index = tomlkit.table()
        for key, value in item.items():
            index[key] = value
        if item is target:
            index["url"] = normalized_url
            index["default"] = True
        elif index.get("default") is True:
            index["default"] = False
        indexes.append(index)
    if target is None:
        index = tomlkit.table()
        index["url"] = normalized_url
        index["default"] = True
        indexes.append(index)
    uv["index"] = indexes

    updated = tomlkit.dumps(document)
    if updated == original:
        return False
    pyproject.write_text(updated, encoding="utf-8")
    return True


def _remove_dependency_source(pyproject: Path, package: str) -> bool:
    try:
        document = tomlkit.parse(pyproject.read_text(encoding="utf-8"))
    except (OSError, TOMLKitError):
        return False
    tool = document.get("tool")
    uv = tool.get("uv") if isinstance(tool, dict) else None
    sources = uv.get("sources") if isinstance(uv, dict) else None
    if not isinstance(sources, dict) or package not in sources:
        return False
    del sources[package]
    pyproject.write_text(tomlkit.dumps(document), encoding="utf-8")
    return True


def _read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _read_optional_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _restore_project_files(
    pyproject: Path,
    pyproject_text: str | None,
    lock: Path,
    lock_bytes: bytes | None,
) -> None:
    if pyproject_text is not None:
        pyproject.write_text(pyproject_text, encoding="utf-8")
    if lock_bytes is not None:
        lock.write_bytes(lock_bytes)
    else:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _run_uv(args: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        env=_uv_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=check,
    )


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
            encoding="utf-8",
            errors="replace",
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


def _declares_editable_checkout(pyproject: Path, checkout: Path) -> bool:
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    if not isinstance(sources, dict):
        return False
    source = sources.get(_HOST_PACKAGE)
    if not isinstance(source, dict) or source.get("editable") is not True:
        return False
    source_path = source.get("path")
    if not isinstance(source_path, str) or not source_path:
        return False
    try:
        return (pyproject.parent / source_path).resolve() == checkout.resolve()
    except OSError:
        return False


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _editable_checkout_root() -> Path | None:
    """Return the repository root when running from a source checkout."""

    # This path is only used when uv-agent is bootstrapping its own managed
    # script environment from a source checkout during development. Installed
    # releases do not have the repository-shaped parent and fall back to the
    # published ``uv-agent`` package dependency.
    root = Path(__file__).resolve().parents[3]
    if (root / "pyproject.toml").exists() and (root / "src" / "uv_agent_runtime").exists():
        return root
    return None


def _uv_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    return env
