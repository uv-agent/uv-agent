from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from uv_agent.builtin.worktree.operations import (
    CommandResult,
    WorktreeError,
    cleanup_worktree,
    create_worktree,
    validate_worktree_branch_name,
    worktree_path_for_branch,
)


def test_validate_worktree_branch_name_rejects_path_like_names() -> None:
    assert validate_worktree_branch_name("feature-1") == "feature-1"
    for name in ["feature/foo", r"feature\foo", "", "bad:name", "bad..name", "bad.lock"]:
        with pytest.raises(WorktreeError):
            validate_worktree_branch_name(name)


def test_worktree_path_for_branch_stays_single_child(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()

    path = worktree_path_for_branch(project, "feature")

    assert path == project / ".uv-agent" / "worktrees" / "feature"
    assert (project / ".uv-agent" / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_create_and_cleanup_worktree_with_git(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")

    info = create_worktree(repo, "feature-test", run=_run)

    assert info.branch == "feature-test"
    assert info.path == repo / ".uv-agent" / "worktrees" / "feature-test"
    assert (info.path / "README.md").exists()
    assert _git(repo, "branch", "--list", "feature-test").stdout.strip()

    result = cleanup_worktree(repo, "feature-test", info.path, run=_run)

    assert result.worktree_removed is True
    assert result.branch_deleted is True
    assert not info.path.exists()
    assert not _git(repo, "branch", "--list", "feature-test").stdout.strip()


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def _run(args: list[str], *, cwd: Path, timeout_s: float | None = None) -> CommandResult:
    completed = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    return CommandResult(args=list(args), returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
