from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_builtin_plugins_live_in_package_directories() -> None:
    for plugin in ("goal", "skills", "mcp", "workflow", "scheduler", "worktree"):
        assert (PROJECT_ROOT / "src" / "uv_agent" / "builtin" / plugin / "__init__.py").is_file()
        assert not (PROJECT_ROOT / "src" / "uv_agent" / "builtin" / f"{plugin}.py").exists()

    for module in (
        "goal_mode",
        "skills",
        "mcp_config",
        "mcp_probe",
        "scheduler",
        "workflow_context",
        "workflow_executor",
        "worktree",
    ):
        assert not (PROJECT_ROOT / "src" / "uv_agent" / f"{module}.py").exists()


def test_core_i18n_excludes_builtin_plugin_texts() -> None:
    core_i18n = _source("src/uv_agent/i18n.py")

    for key in (
        "goal_panel_hint",
        "goal_enabled_flash",
        "worktree_create",
        "worktree_delete_confirm_hint",
        "mention_mcp_hint",
        "mention_skills_hint",
        "no_mcp",
        "no_skills",
    ):
        assert f'"{key}"' not in core_i18n


def test_engine_no_longer_owns_builtin_plugin_services() -> None:
    engine = _source("src/uv_agent/agent/engine.py")

    for needle in (
        "SchedulerService",
        "WorkflowExecutor",
        "McpInstructionsProbe",
        "_mcp_instructions_probe",
        "scheduler_config_from_plugin_config",
        "register_method(\"scheduler.",
        "enable_goal_mode",
        "disable_goal_mode",
        "goal_state",
        "host=self",
    ):
        assert needle not in engine

    plugin_context = _source("src/uv_agent/plugins/context.py")
    assert "self.host" not in plugin_context


def test_app_entrypoints_do_not_start_scheduler_or_workflow_services_directly() -> None:
    combined = "\n".join(
        [
            _source("src/uv_agent/daemon.py"),
            _source("src/uv_agent/tui/app.py"),
        ]
    )

    for needle in (
        ".scheduler.start",
        ".workflow_executor.start",
        "engine.scheduler",
        "engine.workflow_executor",
    ):
        assert needle not in combined


def test_stable_prompt_excludes_builtin_plugin_domain_templates() -> None:
    prompts = _source("src/uv_agent/prompts.py")

    for needle in (
        "SKILLS_HEADER",
        "MCP_SERVERS_HEADER",
        "PLUGIN_HELPERS_HEADER",
        "GOAL_MODE_ACTIVE",
        "WORKTREE_MODE_ACTIVE",
        "WORKFLOW_CONTEXT_TEXT",
        "SKILL_DEFAULT_DESCRIPTION",
        "SKILLS_NONE_DISCOVERED",
        "MCP_DEFAULT_DESCRIPTION",
        "rt.workflow",
        "rt.mcp",
    ):
        assert needle not in prompts
