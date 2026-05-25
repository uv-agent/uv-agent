from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from uv_agent.worktree import (
    CommandResult,
    WorktreeError,
    cleanup_worktree,
    create_worktree,
    render_worktree_notice,
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


def test_render_worktree_notice_describes_active_and_deleted_context(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    worktree = project / ".uv-agent" / "worktrees" / "feature"
    metadata = {
        "worktree_status": "active",
        "worktree_branch": "feature",
        "worktree_path": str(worktree),
        "worktree_base_ref": "HEAD",
        "worktree_origin_root": str(project),
        "worktree_head": "abc123",
        "worktree_created_at": "2026-01-01T00:00:00Z",
        "latest_cwd": str(worktree),
    }

    active = render_worktree_notice(metadata, status="active")
    deleted = render_worktree_notice(
        {
            **metadata,
            "worktree_status": "deleted",
            "latest_cwd": str(project),
            "worktree_deleted_at": "2026-01-02T00:00:00Z",
            "worktree_deleted_head": "def456",
            "worktree_deleted_status": " M file.py",
        },
        status="deleted",
    )

    assert '<worktree status="active">' in active
    assert "Worktree mode is active" in active
    assert str(worktree) in active
    assert "not in the origin workspace" in active
    assert '<worktree status="deleted">' in deleted
    assert "do not rely on the deleted path" in deleted
    assert "def456" in deleted
    assert " M file.py" in deleted


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
