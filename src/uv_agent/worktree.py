from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from uv_agent.paths import ensure_project_local_dir
from uv_agent.time import utc_now_iso


class WorktreeError(RuntimeError):
    """Raised when a Git worktree operation cannot be completed safely."""


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class WorktreeInfo:
    branch: str
    path: Path
    base_ref: str
    origin_root: Path
    head: str = ""
    status: str = ""
    created_at: str = ""

    def metadata(self) -> dict[str, str]:
        return {
            "worktree_status": "active",
            "worktree_branch": self.branch,
            "worktree_path": str(self.path),
            "worktree_base_ref": self.base_ref,
            "worktree_origin_root": str(self.origin_root),
            "worktree_head": self.head,
            "worktree_created_at": self.created_at,
        }


@dataclass(frozen=True)
class WorktreeCleanupResult:
    branch: str
    path: Path
    origin_root: Path
    head: str
    status: str
    worktree_removed: bool
    branch_deleted: bool
    worktree_remove_stdout: str = ""
    worktree_remove_stderr: str = ""
    branch_delete_stdout: str = ""
    branch_delete_stderr: str = ""


class CommandRunner(Protocol):
    def __call__(self, args: list[str], *, cwd: Path, timeout_s: float | None = None) -> CommandResult: ...


_INVALID_WINDOWS_PATH_CHARS = set('<>:"|?*')
_BRANCH_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def validate_worktree_branch_name(branch: str) -> str:
    """Return a stripped branch name or raise ``WorktreeError``.

    Git allows branch namespaces such as ``feature/foo``, but the first TUI
    implementation maps the branch directly to one directory under
    ``.uv-agent/worktrees``. Rejecting separators keeps creation and cleanup
    simple and avoids surprising nested paths on Windows and POSIX.
    """

    name = branch.strip()
    if not name:
        raise WorktreeError("Branch name is required")
    if name in {".", ".."}:
        raise WorktreeError("Branch name cannot be . or ..")
    if "/" in name or "\\" in name:
        raise WorktreeError("Branch name cannot contain path separators")
    if _BRANCH_CONTROL_RE.search(name):
        raise WorktreeError("Branch name cannot contain control characters")
    if any(char in _INVALID_WINDOWS_PATH_CHARS for char in name):
        raise WorktreeError("Branch name contains characters that cannot be used as a directory name")
    if name.endswith(".") or name.endswith(".lock"):
        raise WorktreeError("Branch name cannot end with . or .lock")
    if ".." in name or "@{" in name:
        raise WorktreeError("Branch name is not a valid Git branch name")
    return name


def worktree_base_dir(project_root: Path) -> Path:
    """Return the project-local directory that contains uv-agent worktrees."""

    return ensure_project_local_dir(project_root) / "worktrees"


def worktree_path_for_branch(project_root: Path, branch: str) -> Path:
    """Return the direct child worktree path for a validated branch name."""

    name = validate_worktree_branch_name(branch)
    base = worktree_base_dir(project_root).resolve()
    path = (base / name).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:  # pragma: no cover - defensive after separator validation.
        raise WorktreeError("Branch name escapes the worktree directory") from exc
    if path.parent != base:
        raise WorktreeError("Branch name must map to a single worktree directory")
    return path


def create_worktree(
    project_root: Path,
    branch: str,
    *,
    run: CommandRunner,
    base_ref: str = "HEAD",
) -> WorktreeInfo:
    """Create a new branch/worktree from ``base_ref`` under ``.uv-agent``."""

    root = _git_root(project_root, run=run)
    name = validate_worktree_branch_name(branch)
    _validate_git_branch_name(root, name, run=run)
    path = worktree_path_for_branch(root, name)
    if path.exists():
        raise WorktreeError(f"Worktree path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        ["git", "worktree", "add", "-b", name, str(path), base_ref],
        cwd=root,
        timeout_s=60,
    )
    if result.returncode != 0:
        raise WorktreeError(_command_error("git worktree add", result))
    head = _git_output(["git", "rev-parse", "HEAD"], cwd=path, run=run, required=False)
    status = _git_output(["git", "status", "--short"], cwd=path, run=run, required=False)
    return WorktreeInfo(
        branch=name,
        path=path,
        base_ref=base_ref,
        origin_root=root,
        head=head,
        status=status,
        created_at=utc_now_iso(),
    )


def cleanup_worktree(
    project_root: Path,
    branch: str,
    path: Path,
    *,
    run: CommandRunner,
) -> WorktreeCleanupResult:
    """Remove a uv-agent worktree and delete its local branch."""

    root = _git_root(project_root, run=run)
    name = validate_worktree_branch_name(branch)
    resolved_path = Path(path).resolve()
    base = worktree_base_dir(root).resolve()
    try:
        resolved_path.relative_to(base)
    except ValueError as exc:
        raise WorktreeError("Refusing to remove a worktree outside .uv-agent/worktrees") from exc
    if resolved_path.parent != base:
        raise WorktreeError("Refusing to remove nested worktree paths")
    head = _git_output(["git", "rev-parse", "HEAD"], cwd=resolved_path, run=run, required=False) if resolved_path.exists() else ""
    status = (
        _git_output(["git", "status", "--short"], cwd=resolved_path, run=run, required=False)
        if resolved_path.exists()
        else ""
    )
    remove = run(
        ["git", "worktree", "remove", "--force", str(resolved_path)],
        cwd=root,
        timeout_s=60,
    )
    if remove.returncode != 0:
        raise WorktreeError(_command_error("git worktree remove", remove))
    delete = run(["git", "branch", "-D", name], cwd=root, timeout_s=60)
    if delete.returncode != 0:
        raise WorktreeError(_command_error("git branch -D", delete))
    return WorktreeCleanupResult(
        branch=name,
        path=resolved_path,
        origin_root=root,
        head=head,
        status=status,
        worktree_removed=True,
        branch_deleted=True,
        worktree_remove_stdout=remove.stdout,
        worktree_remove_stderr=remove.stderr,
        branch_delete_stdout=delete.stdout,
        branch_delete_stderr=delete.stderr,
    )


def _git_root(project_root: Path, *, run: CommandRunner) -> Path:
    result = run(["git", "rev-parse", "--show-toplevel"], cwd=project_root, timeout_s=15)
    if result.returncode != 0:
        raise WorktreeError(_command_error("git rev-parse --show-toplevel", result))
    root = result.stdout.strip()
    if not root:
        raise WorktreeError("Could not determine Git repository root")
    return Path(root).resolve()


def _validate_git_branch_name(root: Path, branch: str, *, run: CommandRunner) -> None:
    result = run(["git", "check-ref-format", "--branch", branch], cwd=root, timeout_s=15)
    if result.returncode != 0:
        raise WorktreeError(_command_error("git check-ref-format", result))


def _git_output(args: list[str], *, cwd: Path, run: CommandRunner, required: bool) -> str:
    result = run(args, cwd=cwd, timeout_s=15)
    if result.returncode != 0:
        if required:
            raise WorktreeError(_command_error(" ".join(args), result))
        return ""
    return result.stdout.strip()


def _command_error(label: str, result: CommandResult) -> str:
    message = (result.stderr or result.stdout).strip()
    if not message:
        message = f"exit {result.returncode}"
    return f"{label} failed: {message}"
