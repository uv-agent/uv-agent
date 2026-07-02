from __future__ import annotations

from typing import Any

from uv_agent_runtime import transport


def current(*, thread_id: str | None = None) -> dict[str, Any]:
    """Return worktree metadata for the current or specified thread."""

    return transport.call_host("worktree.current", thread_id=thread_id)


def validate_branch(branch: str) -> str:
    """Validate a candidate worktree branch name and return the normalized value."""

    result = transport.call_host("worktree.validate_branch", branch=branch)
    return str(result.get("branch") or branch)


def create(
    branch: str,
    *,
    base_ref: str = "HEAD",
    thread_id: str | None = None,
    project_root: str | None = None,
) -> dict[str, Any]:
    """Create a project-local Git worktree and attach it to the thread when available."""

    return transport.call_host(
        "worktree.create",
        branch=branch,
        base_ref=base_ref,
        thread_id=thread_id,
        project_root=project_root,
    )


def cleanup(
    branch: str | None = None,
    *,
    path: str | None = None,
    thread_id: str | None = None,
    project_root: str | None = None,
) -> dict[str, Any]:
    """Remove a project-local Git worktree and detach it from the thread when available."""

    return transport.call_host(
        "worktree.cleanup",
        branch=branch,
        path=path,
        thread_id=thread_id,
        project_root=project_root,
    )
