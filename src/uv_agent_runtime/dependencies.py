from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from .textops import CommandTextResult, run_process_text


def run_python_env_dir() -> Path:
    """Return the uv project directory backing the current run_python environment."""

    raw = os.environ.get("UV_AGENT_SCRIPTENV_DIR")
    if not raw:
        raise RuntimeError("UV_AGENT_SCRIPTENV_DIR is not set")
    return Path(raw).resolve()


def add_dependency(
    *packages: str,
    editable: bool = False,
    optional: str | None = None,
    dev: bool = False,
    group: str | None = None,
    timeout_s: float | None = None,
    check: bool = True,
) -> CommandTextResult:
    """Add direct dependencies to the run_python uv project with uv add."""

    if not packages:
        raise ValueError("add_dependency requires at least one package")
    args: list[str] = [_uv_binary(), "add", "--project", str(run_python_env_dir())]
    if editable:
        args.append("--editable")
    if optional:
        args.extend(["--optional", optional])
    if dev:
        args.append("--dev")
    if group:
        args.extend(["--group", group])
    args.extend(packages)
    return run_process_text(args, timeout_s=timeout_s, check=check)


def add_dependencies(
    packages: Sequence[str],
    *,
    editable: bool = False,
    optional: str | None = None,
    dev: bool = False,
    group: str | None = None,
    timeout_s: float | None = None,
    check: bool = True,
) -> CommandTextResult:
    """Add one or more direct dependencies to the run_python uv project."""

    return add_dependency(
        *list(packages),
        editable=editable,
        optional=optional,
        dev=dev,
        group=group,
        timeout_s=timeout_s,
        check=check,
    )


def _uv_binary() -> str:
    return os.environ.get("UV_BIN") or "uv"
