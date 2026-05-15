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
from uv_agent.tui.app import CommandSuggestions, FullscreenPanel, UvAgentApp


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
async def test_tui_status_panel_opens_fullscreen_overlay(
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
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert "258K" in panel.body


@pytest.mark.asyncio
async def test_tui_thread_picker_resumes_and_renders_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    state = tmp_path / "state"
    engine = fake_engine(project_root, state)
    thread_id = engine.thread_store.create_thread("Saved work")
    engine.thread_store.append(
        thread_id,
        "item.user",
        turn_id="turn_1",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
    )
    engine.thread_store.append(
        thread_id,
        "item.model_response",
        turn_id="turn_1",
        response_id="resp_1",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi there"}],
            }
        ],
        usage={},
    )
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("ctrl+o")
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        panel.dismiss(thread_id)
        await pilot.pause()

        assert app.thread_id == thread_id
        assert app._transcript_has_content is True
