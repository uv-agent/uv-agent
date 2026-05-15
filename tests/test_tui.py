from __future__ import annotations

from pathlib import Path
import asyncio

import pytest
from textual import events
from textual.widgets import TextArea

from uv_agent.agent import AgentEngine
from uv_agent.config import (
    AppConfig,
    LevelConfig,
    ModelConfig,
    ProviderConfig,
    RunnerConfig,
    RuntimeConfig,
    UiConfig,
)
from uv_agent.model_client import FakeModelClient
from uv_agent.runner import PythonRunner
from uv_agent.session import ThreadStore
from uv_agent.tui.app import FullscreenPanel, UvAgentApp


class BlockingEngine(AgentEngine):
    def __init__(self, engine: AgentEngine) -> None:
        self.__dict__.update(engine.__dict__)
        self.started = asyncio.Event()

    async def run_turn(self, *, user_text: str, thread_id: str | None = None, level: str | None = None, cancel_event: asyncio.Event | None = None):
        thread_id = thread_id or self.thread_store.create_thread("New thread")
        turn_id = "turn_blocking"
        self.thread_store.append(thread_id, "turn.started", turn_id=turn_id)
        self.thread_store.append(
            thread_id,
            "item.user",
            turn_id=turn_id,
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_text}]},
        )
        yield {"type": "assistant.delta", "thread_id": thread_id, "turn_id": turn_id, "text": "working"}
        self.started.set()
        while cancel_event is not None and not cancel_event.is_set():
            await asyncio.sleep(0.01)
        self.thread_store.append(thread_id, "turn.interrupted", turn_id=turn_id, reason="user_interrupt")
        yield {"type": "turn.interrupted", "thread_id": thread_id, "turn_id": turn_id, "reason": "user_interrupt"}


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
        ui=UiConfig(language="en"),
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
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        await pilot.press("s")
        await pilot.press("enter")
        await pilot.pause()
        status_panel = app.screen_stack[-1]
        assert isinstance(status_panel, FullscreenPanel)
        assert "258K" in status_panel.body


@pytest.mark.asyncio
async def test_tui_slash_picker_does_not_open_when_deleting_existing_command(
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
        composer = app.query_one("#composer", TextArea)
        composer.insert("/new")
        await pilot.pause()
        await pilot.press("backspace")
        await pilot.press("backspace")
        await pilot.press("backspace")
        await pilot.pause()

        assert not isinstance(app.screen_stack[-1], FullscreenPanel)
        assert composer.text == "/"


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
async def test_tui_command_picker_supports_keyboard_selection(
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
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert isinstance(app.screen_stack[-1], FullscreenPanel)

        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        threads_panel = app.screen_stack[-1]
        assert isinstance(threads_panel, FullscreenPanel)
        assert threads_panel.panel_title == app._text("threads")


@pytest.mark.asyncio
async def test_tui_command_picker_prefills_argument_commands(
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
        await pilot.press("ctrl+p")
        await pilot.press("l")
        await pilot.press("enter")
        await pilot.pause()

        composer = app.query_one("#composer", TextArea)
        assert composer.text == "/level "


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
        "item.reasoning_delta",
        turn_id="turn_1",
        text="checking files",
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
        await pilot.press("enter")
        await pilot.pause()

        assert app.thread_id == thread_id
        assert app._transcript_has_content is True
        assert app._reasoning_buffer == "checking files"
        assert not app.query("#empty-state")


@pytest.mark.asyncio
async def test_tui_enter_refocuses_composer_when_transcript_has_focus(
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
        composer = app.query_one("#composer", TextArea)
        app.screen.set_focus(None)
        assert app.screen.focused is None

        await pilot.press("enter")
        await pilot.pause()

        assert app.screen.focused is composer


@pytest.mark.asyncio
async def test_tui_enter_keeps_newline_when_composer_has_focus(
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
        composer = app.query_one("#composer", TextArea)
        composer.insert("line one")

        await pilot.press("enter")
        await pilot.pause()

        assert composer.text == "line one\n"


@pytest.mark.asyncio
async def test_tui_uses_chinese_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = fake_engine(project_root, tmp_path / "state")
    engine.config = AppConfig(
        providers=engine.config.providers,
        models=engine.config.models,
        levels=engine.config.levels,
        runtime=engine.config.runtime,
        runner=engine.config.runner,
        ui=UiConfig(language="zh-CN"),
    )
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)):
        footer = app.query_one("#composer-footer")
        placeholder = app.query_one("#composer", TextArea).placeholder
        assert "medium" in str(footer.content)
        assert "0%" in str(footer.content)
        assert "输入" in placeholder


@pytest.mark.asyncio
async def test_tui_updates_placeholder_after_language_config_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = fake_engine(project_root, tmp_path / "state")
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)):
        composer = app.query_one("#composer", TextArea)
        assert "Ask" in composer.placeholder

        engine.config = AppConfig(
            providers=engine.config.providers,
            models=engine.config.models,
            levels=engine.config.levels,
            runtime=engine.config.runtime,
            runner=engine.config.runner,
            ui=UiConfig(language="zh-CN"),
        )
        app._refresh_status()

        assert "输入" in composer.placeholder


@pytest.mark.asyncio
async def test_tui_selection_auto_copies_after_delay(
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

    async with app.run_test(size=(90, 24), notifications=True) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.insert("copy me")
        composer.select_all()

        await pilot.pause(1.1)

        assert app._clipboard == "copy me"
        assert any(str(toast.render()) == "Copied" for toast in app.screen.query("Toast"))


@pytest.mark.asyncio
async def test_tui_transcript_selection_auto_copies_after_delay(
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

    async with app.run_test(size=(90, 24), notifications=True) as pilot:
        cell = app._append_cell("[bold]agent reply[/bold]", "assistant")
        cell.text_select_all()
        app.screen.post_message(events.TextSelected())

        await pilot.pause(1.1)

        assert app._clipboard == "agent reply"
        assert any(str(toast.render()) == "Copied" for toast in app.screen.query("Toast"))


@pytest.mark.asyncio
async def test_tui_markdown_assistant_selection_auto_copies_plain_text(
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

    async with app.run_test(size=(90, 24), notifications=True) as pilot:
        await app._append_assistant_delta("**agent reply**")
        assert app._assistant_cell is not None
        app._assistant_cell.text_select_all()
        app.screen.post_message(events.TextSelected())

        await pilot.pause(1.1)

        assert app._clipboard == "agent reply"
        assert any(str(toast.render()) == "Copied" for toast in app.screen.query("Toast"))


@pytest.mark.asyncio
async def test_tui_mouse_drag_selects_and_copies_assistant_reply(
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

    async with app.run_test(size=(90, 24), notifications=True) as pilot:
        await app._append_assistant_delta("agent reply")
        assert app._assistant_cell is not None
        await pilot.pause(0.2)

        await pilot.mouse_down(app._assistant_cell, offset=(2, 0))
        await pilot.hover(app._assistant_cell, offset=(10, 0))
        await pilot.mouse_up(app._assistant_cell, offset=(10, 0))
        await pilot.pause(1.1)

        assert app.screen.get_selected_text() == "agent rep"
        assert app._clipboard == "agent rep"
        assert any(str(toast.render()) == "Copied" for toast in app.screen.query("Toast"))


@pytest.mark.asyncio
async def test_tui_ctrl_c_twice_interrupts_busy_turn_without_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = BlockingEngine(fake_engine(project_root, tmp_path / "state"))
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24), notifications=True) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.insert("long work")
        await pilot.press("ctrl+enter")
        await engine.started.wait()

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.busy is True
        assert any(str(toast.render()) == app._text("interrupt_again") for toast in app.screen.query("Toast"))

        await pilot.press("ctrl+c")
        await pilot.pause(0.2)

        assert app.busy is False
        assert any(event["type"] == "turn.interrupted" for event in engine.thread_store.read(app.thread_id))


@pytest.mark.asyncio
async def test_tui_ctrl_c_arms_quit_when_idle(
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

    async with app.run_test(size=(90, 24), notifications=True) as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app._quit_armed is True
        assert any(str(toast.render()) == app._text("quit_again") for toast in app.screen.query("Toast"))


@pytest.mark.asyncio
async def test_tui_ctrl_q_is_not_a_quit_binding(
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

    async with app.run_test(size=(90, 24), notifications=True) as pilot:
        await pilot.press("ctrl+q")
        await pilot.pause()

        assert app._quit_armed is False
        assert not list(app.screen.query("Toast"))
