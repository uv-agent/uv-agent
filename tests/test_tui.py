from __future__ import annotations

from pathlib import Path
import asyncio

import pytest
from textual import events
from textual.widgets import Static, TextArea

from uv_agent.clipboard import ClipboardImage
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
from uv_agent.model_client import FakeModelClient, ToolCallDelta, parse_responses_response
from uv_agent.runner import PythonRunner
from uv_agent.session import ThreadStore
from uv_agent.tui.app import (
    EmptyState,
    ExpandableTranscriptCell,
    FullscreenPanel,
    ImageAttachmentCell,
    ImagePreviewPanel,
    TranscriptScroll,
    ToolDetailsPanel,
    UvAgentApp,
)


@pytest.fixture(autouse=True)
def isolate_uv_agent_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))


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


class ReleasableEngine(AgentEngine):
    def __init__(self, engine: AgentEngine) -> None:
        self.__dict__.update(engine.__dict__)
        self.started: dict[str, asyncio.Event] = {}
        self.release: dict[str, asyncio.Event] = {}

    async def run_turn(self, *, user_text: str, thread_id: str | None = None, level: str | None = None, cancel_event: asyncio.Event | None = None):
        thread_id = thread_id or self.thread_store.create_thread("New thread")
        turn_id = f"turn_{user_text.replace(' ', '_')}"
        self.thread_store.append(thread_id, "turn.started", turn_id=turn_id)
        self.thread_store.append(
            thread_id,
            "item.user",
            turn_id=turn_id,
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_text}]},
        )
        self.started.setdefault(thread_id, asyncio.Event()).set()
        release = self.release.setdefault(thread_id, asyncio.Event())
        while not release.is_set():
            if cancel_event is not None and cancel_event.is_set():
                self.thread_store.append(thread_id, "turn.interrupted", turn_id=turn_id, reason="user_interrupt")
                yield {"type": "turn.interrupted", "thread_id": thread_id, "turn_id": turn_id, "reason": "user_interrupt"}
                return
            await asyncio.sleep(0.01)
        text = f"done {user_text}"
        self.thread_store.append(thread_id, "item.assistant", turn_id=turn_id, text=text)
        self.thread_store.append(thread_id, "turn.completed", turn_id=turn_id, final_text=text)
        yield {"type": "turn.completed", "thread_id": thread_id, "turn_id": turn_id, "final_text": text}


class ImageCaptureEngine(AgentEngine):
    def __init__(self, engine: AgentEngine) -> None:
        self.__dict__.update(engine.__dict__)
        self.image_paths: list[Path] = []

    async def run_turn(
        self,
        *,
        user_text: str,
        thread_id: str | None = None,
        level: str | None = None,
        image_paths: list[Path] | None = None,
        cancel_event: asyncio.Event | None = None,
    ):
        self.image_paths = list(image_paths or [])
        thread_id = thread_id or self.thread_store.create_thread("New thread")
        turn_id = "turn_image"
        self.thread_store.append(thread_id, "turn.started", turn_id=turn_id)
        self.thread_store.append(
            thread_id,
            "item.user",
            turn_id=turn_id,
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_text}]},
        )
        for image_path in image_paths or []:
            attachment = self.attachments.register_image(
                image_path,
                cwd=self.project_root,
                thread_id=thread_id,
                note="pasted from clipboard",
            )
            payload = attachment.to_event_payload()
            self.thread_store.append(thread_id, "item.image_attachment", turn_id=turn_id, attachment=payload)
            yield {"type": "image.attachment", "thread_id": thread_id, "turn_id": turn_id, "attachment": payload}
        text = "done"
        self.thread_store.append(thread_id, "item.assistant", turn_id=turn_id, text=text)
        self.thread_store.append(thread_id, "turn.completed", turn_id=turn_id, final_text=text)
        yield {"type": "turn.completed", "thread_id": thread_id, "turn_id": turn_id, "final_text": text}


class RoutedModelClient(FakeModelClient):
    def __init__(self, *, main: dict[str, object], title: dict[str, object]) -> None:
        super().__init__([])
        self.main = main
        self.title = title

    async def create_response(self, **kwargs):
        self.requests.append(
            {
                "input": kwargs.get("input_items", []),
                "level": kwargs.get("level"),
                "tools": kwargs.get("tools") or [],
                "instructions": kwargs.get("instructions"),
                "stream": False,
            }
        )
        if "Generate a short thread title" in str(kwargs.get("instructions") or ""):
            return parse_responses_response(self.title)
        return parse_responses_response(self.main)


def fake_engine(project_root: Path, state_dir: Path) -> AgentEngine:
    config = AppConfig(
        providers={
            "p": ProviderConfig(
                name="p",
                base_url="https://example.com",
            )
        },
        models={
            "default": ModelConfig(
                name="default",
                provider="p",
                model="fake",
                context_window_tokens=258_000,
                params={},
            )
        },
        levels={
            "small": LevelConfig(name="small", model="default", params={}),
            "medium": LevelConfig(name="medium", model="default", params={}),
            "large": LevelConfig(name="large", model="default", params={}),
        },
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


def response(text: str, response_id: str = "resp") -> dict[str, object]:
    return {
        "id": response_id,
        "output_text": text,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


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
        assert status_panel.panel_title == app._text("status")
        assert status_panel.picker_mode is False
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
async def test_tui_empty_new_thread_gets_generated_title_from_first_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = fake_engine(project_root, tmp_path / "state")
    engine.model_client = RoutedModelClient(
        main=response("done", "resp_1"),
        title=response("Investigate startup crash", "resp_title"),
    )
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.insert("/new")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        thread_id = app.thread_id
        assert thread_id is not None

        composer.insert("investigate the startup crash")
        await pilot.press("ctrl+enter")
        await pilot.pause(0.2)

        assert engine.thread_store.thread_digest(thread_id)["title"] == "Investigate startup crash"


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
        assert panel.panel_title == app._text("status")
        assert panel.picker_mode is False
        assert "258K" in panel.body


@pytest.mark.asyncio
async def test_tui_short_text_panel_ignores_unavailable_scroll_actions(
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
        panel = FullscreenPanel(title="Short", body="short")
        app.push_screen(panel)
        await pilot.pause()

        await pilot.press("down", "pagedown", "up", "pageup")
        await pilot.pause()

        assert app.screen_stack[-1] is panel


@pytest.mark.asyncio
async def test_tui_scroll_y_changes_do_not_disable_auto_follow(
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

    async with app.run_test(size=(80, 12)) as pilot:
        transcript = app.query_one("#transcript", TranscriptScroll)
        for index in range(30):
            app._append_cell(f"line {index}\nextra text", "event")
        await pilot.pause(0.2)

        assert transcript.max_scroll_y > 0
        transcript.follow_tail = True
        transcript.scroll_y = max(0, transcript.scroll_y - 3)
        await pilot.pause()

        assert transcript.follow_tail is True


@pytest.mark.asyncio
async def test_tui_user_scroll_disables_auto_follow(
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

    async with app.run_test(size=(80, 12)) as pilot:
        transcript = app.query_one("#transcript", TranscriptScroll)
        for index in range(30):
            app._append_cell(f"line {index}\nextra text", "event")
        await pilot.pause(0.2)
        transcript.engage_follow_tail()
        await pilot.pause(0.2)

        assert transcript.follow_tail is True
        transcript.action_page_up()
        await pilot.pause()

        assert transcript.follow_tail is False
        assert not app.query_one("#scroll-to-bottom-bar").has_class("hidden")


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
async def test_tui_command_picker_opens_level_panel(
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
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("config_current_level")
        assert composer.text == ""

        await pilot.press("enter")
        await pilot.pause()

        assert app.level == "small"


@pytest.mark.asyncio
async def test_tui_level_panel_uses_configured_levels_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = fake_engine(project_root, tmp_path / "state")
    engine.config = AppConfig(
        providers=engine.config.providers,
        models=engine.config.models,
        levels={
            "small": LevelConfig(name="small", model="default", params={}),
            "medium": LevelConfig(name="medium", model="default", params={}),
        },
        runtime=engine.config.runtime,
        runner=engine.config.runner,
        ui=engine.config.ui,
    )
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        app._open_current_level_panel()
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert [item.id for item in panel.items] == ["small", "medium"]


@pytest.mark.asyncio
async def test_tui_command_palette_hides_run_and_skill_name_commands(
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

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        titles = [item.title for item in panel.items]
        assert "/runs" not in titles
        assert "/skill [name]" not in titles
        assert "/context" not in titles
        assert "/rules" not in titles
        assert "/scripts" not in titles
        assert "/panel" not in titles
        assert "level" in titles
        assert "skills" in titles
        assert all(not title.startswith("/") for title in titles)


@pytest.mark.asyncio
async def test_tui_command_palette_executes_new_without_custom_title(
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
        await pilot.press("enter")
        await pilot.pause()

        assert app.thread_id is not None
        assert app.engine.thread_store.thread_digest(app.thread_id)["title"] == app._text("new_thread")
        composer = app.query_one("#composer", TextArea)
        assert composer.text == ""


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
        "item.reasoning_partial",
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
        assert not app.query(EmptyState)


@pytest.mark.asyncio
async def test_tui_renders_tool_delta_before_tool_started(
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

    async with app.run_test(size=(120, 30)) as pilot:
        run_state = app._run_state_for_thread("thread_1")
        app.thread_id = "thread_1"

        await app._handle_thread_event(
            "thread_1",
            "tool.delta",
            {
                "tool_call": ToolCallDelta(
                    index=0,
                    call_id="call_1",
                    name="run_python",
                    arguments='{"code":"print(1)"}',
                )
            },
            run_state,
        )
        await pilot.pause()

        pending = app.query(".tool_pending").nodes
        assert len(pending) == 1
        assert "print(1)" in str(pending[0].render())

        await app._handle_thread_event(
            "thread_1",
            "tool.started",
            {
                "call": {
                    "call_id": "call_1",
                    "name": "run_python",
                    "arguments": '{"code":"print(1)"}',
                },
                "tool_call_index": 0,
            },
            run_state,
        )
        await pilot.pause()

        assert len(app.query(".tool_pending").nodes) == 1


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
async def test_tui_composer_up_down_browses_recent_inputs(
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
        app._remember_composer_input("first prompt")
        app._remember_composer_input("second prompt")

        await pilot.press("up")
        await pilot.pause()
        assert composer.text == "second prompt"
        assert composer.cursor_location == composer.document.end

        await pilot.press("up")
        await pilot.pause()
        assert composer.text == "first prompt"

        await pilot.press("down")
        await pilot.pause()
        assert composer.text == "second prompt"

        await pilot.press("down")
        await pilot.pause()
        assert composer.text == ""


@pytest.mark.asyncio
async def test_tui_composer_up_keeps_normal_editing_when_text_exists(
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
        app._remember_composer_input("previous prompt")
        composer.insert("draft")

        await pilot.press("up")
        await pilot.pause()

        assert composer.text == "draft"
        assert app._composer_history_index is None


@pytest.mark.asyncio
async def test_tui_composer_history_is_bounded_and_skips_consecutive_duplicates(
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
        for index in range(55):
            app._remember_composer_input(f"prompt {index}")
        app._remember_composer_input("prompt 54")

        assert len(app._composer_history) == 50
        assert app._composer_history[0] == "prompt 5"
        assert app._composer_history[-1] == "prompt 54"

        await pilot.press("up")
        await pilot.pause()

        assert composer.text == "prompt 54"


@pytest.mark.asyncio
async def test_tui_composer_history_persists_globally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    state_dir = tmp_path / "state"
    monkeypatch.setattr(
        "uv_agent.tui.app.create_engine",
        lambda root: fake_engine(root, state_dir),
    )
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)):
        app._remember_composer_input("persisted prompt")

    restarted = UvAgentApp(project_root=project_root)
    async with restarted.run_test(size=(90, 24)) as pilot:
        composer = restarted.query_one("#composer", TextArea)

        await pilot.press("up")
        await pilot.pause()

        assert composer.text == "persisted prompt"


@pytest.mark.asyncio
async def test_tui_composer_expands_and_collapses_with_tab_without_button(
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

    async with app.run_test(size=(90, 30)) as pilot:
        composer = app.query_one("#composer", TextArea)
        footer = app.query_one("#composer-footer")

        assert composer.styles.height.value == 5
        assert not app.query("#composer-toggle")
        assert str(footer.content)

        composer.insert("1\n2\n3\n4\n5")
        await pilot.pause()

        assert app._composer_expanded is True
        assert composer.styles.height.value == 13
        assert str(footer.content)

        await pilot.press("tab")
        await pilot.pause()

        assert app._composer_expanded is False
        assert composer.styles.height.value == 5
        assert str(footer.content)


@pytest.mark.asyncio
async def test_tui_at_mention_inserts_file_reference_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    source_dir = project_root / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "example.py").write_text("print('hi')\n", encoding="utf-8")
    monkeypatch.setattr(
        "uv_agent.tui.app.create_engine",
        lambda root: fake_engine(root, tmp_path / "state"),
    )
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.insert("@")
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("mention_files")

        await pilot.press("enter")
        await pilot.pause()

        assert composer.text == "@src/example.py "
        assert app.thread_id is None
        assert app._transcript_has_content is False


@pytest.mark.asyncio
async def test_tui_mention_switches_from_files_to_threads_with_at(
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
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.insert("@")
        await pilot.pause()

        file_panel = app.screen_stack[-1]
        assert isinstance(file_panel, FullscreenPanel)
        assert file_panel.panel_title == app._text("mention_files")

        await pilot.press("@")
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("mention_threads")

        await pilot.press("enter")
        await pilot.pause()

        assert composer.text == f"@thread:{thread_id} "
        assert app.thread_id is None
        assert app._transcript_has_content is False


@pytest.mark.asyncio
async def test_tui_mention_backspace_returns_from_threads_to_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    source_dir = project_root / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "example.py").write_text("print('hi')\n", encoding="utf-8")
    state = tmp_path / "state"
    engine = fake_engine(project_root, state)
    engine.thread_store.create_thread("Saved work")
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.insert("@")
        await pilot.pause()
        await pilot.press("@")
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("mention_threads")

        await pilot.press("backspace")
        await pilot.pause()

        assert panel.panel_title == app._text("mention_files")
        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "@src/example.py "


@pytest.mark.asyncio
async def test_tui_mention_picker_does_not_reopen_when_backspacing_to_trigger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    source_dir = project_root / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "example.py").write_text("print('hi')\n", encoding="utf-8")
    monkeypatch.setattr(
        "uv_agent.tui.app.create_engine",
        lambda root: fake_engine(root, tmp_path / "state"),
    )
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.insert("@a")
        await pilot.pause()

        assert not isinstance(app.screen_stack[-1], FullscreenPanel)

        await pilot.press("backspace")
        await pilot.pause()

        assert composer.text == "@"
        assert not isinstance(app.screen_stack[-1], FullscreenPanel)


@pytest.mark.asyncio
async def test_tui_mcp_and_skill_mentions_are_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`@mcp:` / `@skill:` triggers were removed; use /mcp and /skills instead.

    Typing those literal strings must leave the composer alone (no popup) so the
    user can still type the same prefix as plain text without losing focus.
    """
    project_root = tmp_path / "project"
    agents_dir = project_root / ".agents"
    skill_dir = agents_dir / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo\nUse this for demo work.\n", encoding="utf-8")
    (agents_dir / "mcp.json").write_text(
        __import__("json").dumps(
            {
                "servers": {
                    "files": {
                        "description": "File helpers",
                        "command": "python",
                        "args": ["server.py"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "uv_agent.tui.app.create_engine",
        lambda root: fake_engine(root, tmp_path / "state"),
    )
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        composer = app.query_one("#composer", TextArea)
        baseline_screens = len(app.screen_stack)

        composer.insert("@mcp:")
        await pilot.pause()
        assert len(app.screen_stack) == baseline_screens
        assert composer.text == "@mcp:"

        composer.insert("@skill：")
        await pilot.pause()
        assert len(app.screen_stack) == baseline_screens
        assert composer.text == "@mcp:@skill："
        composer.load_text("")

        # The /mcp and /skills slash commands still use the same underlying
        # data, but selecting an item inserts a mention instead of only
        # inspecting it.
        app._open_mcp_panel()
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("mcp")
        await pilot.press("f")
        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "@mcp:files "
        composer.load_text("")

        app._open_skills_panel()
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("skills")
        await pilot.press("d")
        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "@skill:demo "


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
async def test_tui_config_panel_sets_default_level_and_writes_editable_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_path = project_root / ".uv-agent" / "config.json"
    monkeypatch.setattr(
        "uv_agent.tui.app.create_engine",
        lambda root: fake_engine(root, tmp_path / "state"),
    )
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        app._open_config_panel()
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("config")

        await pilot.press("enter")
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("config_default_level")

        await pilot.press("enter")
        await pilot.pause()

        data = __import__("json").loads(config_path.read_text(encoding="utf-8"))
        assert data["runtime"]["default_level"] == "small"


@pytest.mark.asyncio
async def test_tui_models_panel_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/models` must list configured models without offering edits.

    Model and level definitions belong in `config.json`; the TUI may only let
    the user switch the active level at runtime via `/level`.
    """
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_path = project_root / ".uv-agent" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        __import__("json").dumps(
            {
                "providers": {"p": {"base_url": "https://example.com"}},
                "models": {"default": {"provider": "p", "model": "fake"}},
                "levels": {
                    "small": {"model": "default"},
                    "medium": {"model": "default"},
                    "large": {"model": "default"},
                },
                "runtime": {"default_level": "medium", "auto_compress": False},
            }
        ),
        encoding="utf-8",
    )
    engine = fake_engine(project_root, tmp_path / "state")
    engine.config_loader = None
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        app._open_models_panel()
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("models")
        # The picker must list every configured model and nothing else;
        # no "level ..." rows that would imply runtime level switching here.
        ids = [item.id for item in panel.items]
        assert ids == ["default"]
        # Subtitle must steer the user to the config file for any edits.
        assert app._text("models_edit_hint") in panel.subtitle


@pytest.mark.asyncio
async def test_tui_config_panel_omits_model_editing_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/config` must not offer model editing rows."""
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(
        "uv_agent.tui.app.create_engine",
        lambda root: fake_engine(root, tmp_path / "state"),
    )
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        app._open_config_panel()
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        ids = [item.id for item in panel.items]
        assert "default_level" in ids
        assert "level_models" not in ids
        assert "current_level" not in ids
        # Helpers used by the removed flows must also be gone.
        assert not hasattr(app, "_open_reasoning_level_panel")
        assert not hasattr(app, "_open_level_model_panel")
        assert not hasattr(app, "_set_level_model_from_choice")
        assert not hasattr(app, "_set_level_reasoning_from_choice")


@pytest.mark.asyncio
async def test_tui_status_summarizes_context_rules_and_scripts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "AGENTS.md").write_text("Use local rules.", encoding="utf-8")
    engine = fake_engine(project_root, tmp_path / "state")
    engine.runner.store.create_script(
        original_code="print('hi')",
        final_code="print('hi')",
        thread_id=None,
        turn_id=None,
    )
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        app._open_status_panel()
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("status")
        assert panel.picker_mode is False
        assert "258K" in panel.body
        assert "AGENTS.md" in panel.body
        assert "Use local rules" not in panel.body
        assert "print('hi')" in panel.body


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
async def test_tui_markdown_assistant_selection_is_visibly_highlighted(
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
        await app._append_assistant_delta("agent reply")
        assert app._assistant_cell is not None

        await pilot.pause(0.2)
        await pilot.mouse_down(app._assistant_cell, offset=(1, 0))
        await pilot.hover(app._assistant_cell, offset=(9, 0))
        await pilot.mouse_up(app._assistant_cell, offset=(9, 0))
        await pilot.pause()

        strip = app._assistant_cell.render_line(0)
        highlighted = [
            segment
            for segment in strip
            if segment.style is not None
            and segment.style.bgcolor is not None
            and segment.style.bgcolor.triplet is not None
            and segment.style.bgcolor.triplet.red == 125
            and segment.style.bgcolor.triplet.green == 211
            and segment.style.bgcolor.triplet.blue == 252
        ]

        assert highlighted


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

        await pilot.mouse_down(app._assistant_cell, offset=(1, 0))
        await pilot.hover(app._assistant_cell, offset=(9, 0))
        await pilot.mouse_up(app._assistant_cell, offset=(9, 0))
        await pilot.pause(1.1)

        assert app.screen.get_selected_text() == "agent rep"
        assert app._clipboard == "agent rep"
        assert any(str(toast.render()) == "Copied" for toast in app.screen.query("Toast"))


@pytest.mark.asyncio
async def test_tui_mouse_drag_selection_tracks_after_intermediate_render(
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
        await app._append_assistant_delta("agent reply")
        assert app._assistant_cell is not None
        await pilot.pause(0.2)

        await pilot.mouse_down(app._assistant_cell, offset=(1, 0))
        await pilot.hover(app._assistant_cell, offset=(4, 0))
        await pilot.pause()
        app._assistant_cell.render_line(0)
        await pilot.hover(app._assistant_cell, offset=(9, 0))
        await pilot.pause()

        assert app.screen.get_selected_text() == "agent rep"


@pytest.mark.asyncio
async def test_tui_wide_character_selection_highlight_preserves_edge_text(
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
        await app._append_assistant_delta("你好世界abc")
        assert app._assistant_cell is not None
        await pilot.pause(0.2)

        await pilot.mouse_down(app._assistant_cell, offset=(1, 0))
        await pilot.hover(app._assistant_cell, offset=(8, 0))
        await pilot.pause()

        strip = app._assistant_cell.render_line(0)
        highlighted_text = "".join(
            segment.text
            for segment in strip
            if segment.style is not None
            and segment.style.bgcolor is not None
            and segment.style.bgcolor.triplet is not None
            and segment.style.bgcolor.triplet.red == 125
            and segment.style.bgcolor.triplet.green == 211
            and segment.style.bgcolor.triplet.blue == 252
        )

        assert app.screen.get_selected_text() == "你好世界a"
        assert strip.text.rstrip() == "你好世界abc"
        assert highlighted_text == "你好世界a"


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
async def test_tui_new_thread_while_current_thread_runs_keeps_old_thread_backgrounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = ReleasableEngine(fake_engine(project_root, tmp_path / "state"))
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.insert("old work")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        old_thread = app.thread_id
        assert old_thread is not None
        await engine.started[old_thread].wait()

        composer.insert("/new second")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        new_thread = app.thread_id

        assert new_thread != old_thread
        assert old_thread in app._thread_runs
        assert app.busy is False

        composer.insert("new work")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert new_thread in app._thread_runs

        engine.release[old_thread].set()
        engine.release[new_thread].set()
        await pilot.pause(0.2)

        assert old_thread not in app._thread_runs
        assert new_thread not in app._thread_runs


@pytest.mark.asyncio
async def test_tui_tool_result_details_expand_on_click(
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
    payload = {
        "script_id": "scr_1",
        "run_id": "run_1",
        "returncode": 0,
        "stdout": "visible one\nvisible two\nvisible three\nhidden tail",
        "stderr": "",
        "events": [{"kind": "subagent.completed", "thread_id": "thr_child", "summary": "child done"}],
        "run_log_path": str(tmp_path / "run.jsonl"),
    }

    async with app.run_test(size=(90, 24)) as pilot:
        app._append_tool_output(
            {
                "call": {"call_id": "call_1"},
                "output": {"output": __import__("json").dumps(payload)},
            }
        )
        await pilot.pause()
        cell = app.query_one(ExpandableTranscriptCell)

        rendered = str(cell.render())
        assert "visible one" in rendered
        assert "visible three" in rendered
        assert "hidden tail" not in rendered

        await pilot.click(cell)
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, ToolDetailsPanel)
        shell = panel.query_one("#panel-shell")
        assert shell.region.x == (app.size.width - shell.region.width) // 2
        assert shell.region.y == (app.size.height - shell.region.height) // 2
        assert "hidden tail" in str(panel.query_one("#panel-body-content", Static).render())

        await pilot.click(panel, offset=(0, 0))
        await pilot.pause()

        assert app.screen is app.default_screen


@pytest.mark.asyncio
async def test_tui_tool_result_details_escape_literal_brackets(
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
    payload = {
        "script_id": "scr_1",
        "run_id": "run_1",
        "returncode": 0,
        "stdout": (
            "assert '# dependencies = [' in prompt\n"
            "+    assert 'level=\"small\"' not in prompt\n"
        ),
        "stderr": "",
        "events": [],
        "run_log_path": str(tmp_path / "run.jsonl"),
    }

    async with app.run_test(size=(90, 24)) as pilot:
        app._append_tool_output(
            {
                "call": {"call_id": "call_1"},
                "output": {"output": __import__("json").dumps(payload)},
            }
        )
        await pilot.pause()
        cell = app.query_one(ExpandableTranscriptCell)

        await pilot.click(cell)
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, ToolDetailsPanel)
        rendered = str(panel.query_one("#panel-body-content", Static).render())
        assert "# dependencies = [" in rendered
        assert "level=\"small\"" in rendered


@pytest.mark.asyncio
async def test_tui_tool_result_details_support_keyboard_navigation(
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
        for index in range(2):
            app._append_tool_output(
                {
                    "call": {"call_id": f"call_{index}"},
                    "output": {
                        "output": __import__("json").dumps(
                            {
                                "script_id": f"scr_{index}",
                                "run_id": f"run_{index}",
                                "returncode": 0,
                                "stdout": f"preview {index}\nfull tail {index}",
                                "stderr": "",
                                "events": [],
                                "run_log_path": str(tmp_path / f"run-{index}.jsonl"),
                            }
                        )
                    },
                }
            )
        await pilot.pause()
        first, second = app.query(ExpandableTranscriptCell).nodes

        await pilot.press("ctrl+d")
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, ToolDetailsPanel)
        assert panel.current_cell is second
        assert "full tail 1" in str(panel.query_one("#panel-body-content", Static).render())

        await pilot.press("k")
        await pilot.pause()

        assert panel.current_cell is first
        assert "full tail 0" in str(panel.query_one("#panel-body-content", Static).render())

        await pilot.press("j")
        await pilot.pause()

        assert panel.current_cell is second
        assert "full tail 1" in str(panel.query_one("#panel-body-content", Static).render())

        await pilot.press("ctrl+d")
        await pilot.pause()

        assert app.screen is app.default_screen


@pytest.mark.asyncio
async def test_tui_f2_attaches_clipboard_image_and_sends_with_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    image = tmp_path / "clip.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    engine = ImageCaptureEngine(fake_engine(project_root, tmp_path / "state"))
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    monkeypatch.setattr(
        "uv_agent.tui.app.save_clipboard_image",
        lambda target_dir: ClipboardImage(path=image, width=20, height=10),
    )
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24), notifications=True) as pilot:
        await pilot.press("f2")
        await pilot.pause()

        assert len(app._pending_images) == 1
        assert "clip.png" in str(app.query_one("#composer-meta", Static).render())

        composer = app.query_one("#composer", TextArea)
        composer.insert("inspect")
        await pilot.press("ctrl+enter")
        await pilot.pause(0.2)

        assert app._pending_images == []
        assert engine.image_paths == [image]
        assert app.query_one(ImageAttachmentCell)


@pytest.mark.asyncio
async def test_tui_image_preview_panel_switches_sent_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = fake_engine(project_root, tmp_path / "state")
    thread_id = engine.thread_store.create_thread("images")
    for index in range(2):
        image = tmp_path / f"image-{index}.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        attachment = engine.attachments.register_image(image, cwd=project_root, thread_id=thread_id)
        engine.thread_store.append(
            thread_id,
            "item.image_attachment",
            turn_id=f"turn_{index}",
            attachment=attachment.to_event_payload(),
        )
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        app._resume_thread(thread_id)
        await pilot.pause()
        cells = app.query(ImageAttachmentCell).nodes
        assert len(cells) == 2

        await pilot.press("f3")
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, ImagePreviewPanel)
        assert panel.index == 1
        assert "image-1" in str(panel.query_one("#panel-body-content", Static).render())

        await pilot.press("k")
        await pilot.pause()

        assert panel.index == 0
        assert "image-0" in str(panel.query_one("#panel-body-content", Static).render())

        await pilot.press("f3")
        await pilot.pause()

        assert app.screen is app.default_screen


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
async def test_tui_ctrl_c_fast_second_press_quits_when_idle(
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
        await pilot.pause(0.12)
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app._exit is True


@pytest.mark.asyncio
async def test_tui_quit_command_exits_without_confirmation(
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
        composer.insert("/quit")
        await pilot.press("ctrl+enter")
        await pilot.pause()

        assert app._quit_armed is False
        assert app._exit is True


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
