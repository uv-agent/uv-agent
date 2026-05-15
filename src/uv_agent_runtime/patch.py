from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PatchResult:
    returncode: int
    stdout: str
    stderr: str
    changed_files: list[str]


def apply_patch(patch: str, *, cwd: str | Path | None = None, check: bool = True) -> PatchResult:
    """Apply a unified diff with git apply and report changed paths."""
    workdir = Path(cwd).resolve() if cwd is not None else Path.cwd()
    before = _git_changed_files(workdir)
    patch_paths = _patch_changed_files(patch)
    completed = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        input=patch,
        text=True,
        capture_output=True,
        cwd=workdir,
        check=False,
    )
    after = _git_changed_files(workdir)
    result = PatchResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        changed_files=sorted(patch_paths or (after - before if completed.returncode == 0 else after)),
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"patch failed with exit {result.returncode}:\n{result.stderr}")
    return result


def _patch_changed_files(patch: str) -> set[str]:
    files: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path != "/dev/null":
                files.add(_strip_diff_prefix(path))
        elif line.startswith("--- "):
            path = line[4:].strip()
            if path != "/dev/null":
                files.add(_strip_diff_prefix(path))
    return files


def _strip_diff_prefix(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _git_changed_files(cwd: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        text=True,
        capture_output=True,
        cwd=cwd,
        check=False,
    )
    if completed.returncode != 0:
        return set()
    files: set[str] = set()
    for line in completed.stdout.splitlines():
        if len(line) > 3:
            files.add(line[3:].strip())
    return files
