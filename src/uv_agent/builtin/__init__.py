from __future__ import annotations

from collections.abc import Iterable

from uv_agent.plugins import SetupPlugin

from .goal import plugin as goal_plugin
from .mcp import plugin as mcp_plugin
from .scheduler import plugin as scheduler_plugin
from .skills import plugin as skills_plugin
from .workflow import plugin as workflow_plugin
from .worktree import plugin as worktree_plugin


def builtin_plugins() -> Iterable[SetupPlugin]:
    """Return builtin plugins in their stable startup order."""

    return [
        goal_plugin(),
        worktree_plugin(),
        skills_plugin(),
        mcp_plugin(),
        workflow_plugin(),
        scheduler_plugin(),
    ]
