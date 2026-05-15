from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import TextArea

from uv_agent.agent import AgentEngine
from uv_agent.config import (
    AppConfig,
    LevelConfig,
    ModelConfig,
    ProviderConfig,
    RunnerConfig,
    RuntimeConfig,
)
from uv_agent.model_client import FakeModelClient
from uv_agent.runner import PythonRunner
from uv_agent.session import ThreadStore
from uv_agent.tui.app import CommandSuggestions, UvAgentApp


def fake_engine(project_root: Path, state_dir: Path) -> AgentEngine:
    config = AppConfig(
        providers={"p": ProviderConfig(name="p", base_url="https://example.com")},
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                context_window_tokens=258_000,
                params={},
            )
        },
        levels={"medium": LevelConfig(name="medium", model="default", params={})},
        runtime=RuntimeConfig(default_level="medium", auto_compress=False),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
    )
    return AgentEngine(
        config=config,
        model_client=FakeModelClient([]),
        runner=PythonRunner(project_root=project_root, data_dir=state_dir, config=config.runner),
        thread_store=ThreadStore(state_dir),
        project_root=project_root,
    )


@pytest.mark.asyncio
async def test_tui_command_palette_completes_without_blocking_newlines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(
        "uv_agent.tui.app.create_engine",
        lambda root: fake_engine(root, tmp_path / "state"),
    )
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.press("/")
        await pilot.press("s")
        suggestions = app.query_one("#command-suggestions", CommandSuggestions)
        assert not suggestions.has_class("hidden")
        await pilot.press("enter")
        composer = app.query_one("#composer", TextArea)
        assert composer.text == "/status"
        assert suggestions.has_class("hidden")
        await pilot.press("end")
        await pilot.press("enter")
        assert composer.text == "/status\n"


@pytest.mark.asyncio
async def test_tui_status_panel_uses_side_drawer_on_wide_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(
        "uv_agent.tui.app.create_engine",
        lambda root: fake_engine(root, tmp_path / "state"),
    )
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(140, 32)) as pilot:
        await pilot.press("ctrl+s")
        assert app.query_one("#drawer").has_class("hidden")
        assert not app.query_one("#side-drawer").has_class("hidden")
        assert "258K" in str(app.query_one("#drawer-body").content)
