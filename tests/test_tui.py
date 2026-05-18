from __future__ import annotations

from pathlib import Path
import asyncio
import threading

from PIL import Image as PILImage
import pytest
from textual import events
from textual.widgets import OptionList, Static, TextArea
from textual_image.widget import Image as TerminalImage

from uv_agent.clipboard import ClipboardImage
from uv_agent.agent import AgentEngine
from uv_agent.config import (
    AppConfig,
    CompressionConfig,
    LevelConfig,
    ModelConfig,
    ProviderConfig,
    RunnerConfig,
    RuntimeConfig,
    CompletionNotificationConfig,
    UiConfig,
)
from uv_agent.model_client import FakeModelClient, ModelResponse, ToolCallDelta, parse_responses_response
from uv_agent.runner import PythonRunner
from uv_agent.session import ThreadStore
from uv_agent.tui.formatting import short_thread
from uv_agent.tui.app import (
    EmptyState,
    ExpandableTranscriptCell,
    FoldedProcessCell,
    FullscreenPanel,
    ImageAttachmentCell,
    ImagePreviewPanel,
    TranscriptCell,
    TranscriptScroll,
    ToolDetailsPanel,
    UvAgentApp,
)


@pytest.fixture(autouse=True)
def isolate_uv_agent_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))


def write_png(path: Path, *, color: tuple[int, int, int] = (32, 128, 224)) -> None:
    image = PILImage.new("RGB", (8, 6), color)
    image.save(path, format="PNG")


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


class StableRoundEngine(AgentEngine):
    def __init__(self, engine: AgentEngine) -> None:
        self.__dict__.update(engine.__dict__)

    async def run_turn(
        self,
        *,
        user_text: str,
        thread_id: str | None = None,
        level: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ):
        thread_id = thread_id or self.thread_store.create_thread("Stable round")
        turn_id = "turn_stable"
        output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will inspect first."}],
                "reasoning_content": "provider reasoning",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "run_python",
                "arguments": '{"code":"print(1)"}',
            },
        ]
        self.thread_store.append(thread_id, "turn.started", turn_id=turn_id)
        self.thread_store.append(
            thread_id,
            "item.user",
            turn_id=turn_id,
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_text}]},
        )
        yield {"type": "assistant.reasoning_delta", "thread_id": thread_id, "turn_id": turn_id, "text": "provider "}
        yield {"type": "assistant.reasoning_delta", "thread_id": thread_id, "turn_id": turn_id, "text": "reasoning"}
        yield {
            "type": "assistant.delta",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "text": "I will inspect first.",
        }
        response = ModelResponse(
            id="resp_1",
            output=output,
            output_text="I will inspect first.",
            raw={"id": "resp_1", "output": output},
            usage={},
            reasoning_text="provider reasoning",
        )
        self.thread_store.append(
            thread_id,
            "item.model_response",
            turn_id=turn_id,
            response_id=response.id,
            output=response.output,
            usage=response.usage,
            reasoning_text=response.reasoning_text,
        )
        yield {"type": "model.response", "thread_id": thread_id, "turn_id": turn_id, "response": response}
        yield {
            "type": "tool.started",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "call": output[1],
            "tool_call_index": 0,
        }
        payload = {
            "script_id": "scr_1",
            "run_id": "run_1",
            "returncode": 0,
            "timed_out": False,
            "interrupted": False,
            "truncated": False,
            "stdout": "ok\n",
            "stderr": "",
            "events": [],
            "run_log_path": "",
        }
        self.thread_store.append(
            thread_id,
            "item.runner_result",
            turn_id=turn_id,
            call_id="call_1",
            result=payload,
        )
        tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "{}"}
        self.thread_store.append(thread_id, "item.tool_output", turn_id=turn_id, item=tool_output)
        yield {
            "type": "tool.output",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "call": output[1],
            "tool_call_index": 0,
            "output": {"type": "function_call_output", "call_id": "call_1", "output": __import__("json").dumps(payload)},
        }
        yield {"type": "assistant.delta", "thread_id": thread_id, "turn_id": turn_id, "text": "Done."}
        final_response = ModelResponse(
            id="resp_2",
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done."}],
                }
            ],
            output_text="Done.",
            raw={"id": "resp_2"},
            usage={},
        )
        self.thread_store.append(
            thread_id,
            "item.model_response",
            turn_id=turn_id,
            response_id=final_response.id,
            output=final_response.output,
            usage={},
            reasoning_text="",
        )
        yield {"type": "model.response", "thread_id": thread_id, "turn_id": turn_id, "response": final_response}
        self.thread_store.append(thread_id, "turn.completed", turn_id=turn_id, final_text="Done.")
        yield {"type": "turn.completed", "thread_id": thread_id, "turn_id": turn_id, "final_text": "Done."}


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
        runtime=RuntimeConfig(default_level="medium", compression=CompressionConfig(enabled=False)),
        runner=RunnerConfig(
            runtime_dependency=f"uv-agent @ {Path.cwd().resolve().as_uri()}",
            runtime_package_name="uv-agent",
        ),
        ui=UiConfig(
            language="en",
            completion_notification=CompletionNotificationConfig(enabled=False),
        ),
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
        composer.insert("/clear")
        await pilot.pause()
        for _ in range(len("clear")):
            await pilot.press("backspace")
            await pilot.pause()

        assert not isinstance(app.screen_stack[-1], FullscreenPanel)
        assert composer.text == "/"


@pytest.mark.asyncio
async def test_tui_clear_keeps_next_prompt_title_generation(
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
        composer.insert("/clear")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert app.thread_id is None

        composer.insert("investigate the startup crash")
        await pilot.press("ctrl+enter")
        await pilot.pause(0.2)

        thread_id = app.thread_id
        assert thread_id is not None
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
async def test_tui_scrolling_back_to_bottom_reengages_auto_follow(
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
        transcript.action_page_up()
        await pilot.pause()

        assert transcript.follow_tail is False
        assert transcript.near_bottom is False

        transcript.scroll_to(y=transcript.max_scroll_y, animate=False)
        await pilot.pause()

        assert transcript.near_bottom is True
        assert transcript.follow_tail is True


@pytest.mark.asyncio
async def test_tui_submit_from_bottom_reengages_auto_follow(
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
        transcript.follow_tail = False
        transcript.scroll_y = transcript.max_scroll_y
        await pilot.pause()

        assert transcript.near_bottom is True
        assert transcript.follow_tail is False

        composer = app.query_one("#composer", TextArea)
        composer.insert("continue")
        await pilot.press("ctrl+j")
        await pilot.pause(0.2)

        assert transcript.follow_tail is True
        assert transcript.scroll_y == transcript.max_scroll_y


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
async def test_tui_command_picker_escape_returns_from_level_to_commands(
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

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("config_current_level")

        await pilot.press("escape")
        await pilot.pause()

        assert app.screen_stack[-1] is panel
        assert panel.panel_title == app._text("command_palette")

        await pilot.press("escape")
        await pilot.pause()

        assert not isinstance(app.screen_stack[-1], FullscreenPanel)


@pytest.mark.asyncio
async def test_tui_command_picker_escape_returns_through_config_pages(
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
        await pilot.press("c")
        await pilot.press("enter")
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("config")

        await pilot.press("enter")
        await pilot.pause()

        assert app.screen_stack[-1] is panel
        assert panel.panel_title == app._text("config_default_level")

        await pilot.press("escape")
        await pilot.pause()

        assert app.screen_stack[-1] is panel
        assert panel.panel_title == app._text("config")

        await pilot.press("escape")
        await pilot.pause()

        assert app.screen_stack[-1] is panel
        assert panel.panel_title == app._text("command_palette")


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
async def test_tui_command_palette_starts_with_threads_command(
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

        assert app.thread_id is None
        assert isinstance(app.screen_stack[-1], FullscreenPanel)
        assert app.screen_stack[-1].panel_title == app._text("threads")
        composer = app.query_one("#composer", TextArea)
        assert composer.text == ""


@pytest.mark.asyncio
async def test_tui_command_palette_thread_selection_closes_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = fake_engine(project_root, tmp_path / "state")
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
        await pilot.press("ctrl+p")
        await pilot.press("enter")
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("threads")

        await pilot.press("enter")
        await pilot.pause()

        assert app.thread_id == thread_id
        assert not isinstance(app.screen_stack[-1], FullscreenPanel)


@pytest.mark.asyncio
async def test_tui_command_palette_clear_closes_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = fake_engine(project_root, tmp_path / "state")
    thread_id = engine.thread_store.create_thread("Current work")
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        app.thread_id = thread_id
        await pilot.press("ctrl+p")
        await pilot.press("c", "l")
        await pilot.press("enter")
        await pilot.pause()

        assert app.thread_id is None
        assert not isinstance(app.screen_stack[-1], FullscreenPanel)


@pytest.mark.asyncio
async def test_tui_command_palette_skills_opens_leaf_picker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    skill_dir = project_root / ".agents" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\ndescription: demo skill\n---\n", encoding="utf-8")
    monkeypatch.setattr(
        "uv_agent.tui.app.create_engine",
        lambda root: fake_engine(root, tmp_path / "state"),
    )
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.press("ctrl+p")
        await pilot.press("s", "k")
        await pilot.press("enter")
        await pilot.pause(0.2)

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("skills")

        await pilot.press("enter")
        await pilot.pause()

        composer = app.query_one("#composer", TextArea)
        assert composer.text == "@skill:demo "
        assert not isinstance(app.screen_stack[-1], FullscreenPanel)


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
        reasoning_text="checking files",
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
        reasoning_cells = [
            child
            for child in app.query_one("#transcript", TranscriptScroll).children
            if isinstance(child, ExpandableTranscriptCell) and child.detail_title == "reasoning_details"
        ]
        assert reasoning_cells
        assert "checking files" in reasoning_cells[0].details
        assert not app.query(EmptyState)


@pytest.mark.asyncio
async def test_tui_thread_resume_renders_mixed_text_tool_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    state = tmp_path / "state"
    engine = fake_engine(project_root, state)
    thread_id = engine.thread_store.create_thread("Mixed response")
    engine.thread_store.append(
        thread_id,
        "item.user",
        turn_id="turn_1",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "inspect"}]},
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
                "content": [{"type": "output_text", "text": "I will inspect first."}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "run_python",
                "arguments": "{\"code\":\"print('ok')\"}",
            },
        ],
        reasoning_text="thinking first",
        usage={},
    )
    engine.thread_store.append(
        thread_id,
        "item.runner_result",
        turn_id="turn_1",
        call_id="call_1",
        result={
            "script_id": "scr_1",
            "run_id": "run_1",
            "returncode": 0,
            "timed_out": False,
            "interrupted": False,
            "truncated": False,
            "stdout": "ok\n",
            "stderr": "",
            "events": [],
            "run_log_path": "",
        },
    )
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(120, 30)) as pilot:
        app._resume_thread(thread_id)
        await pilot.pause()

        children = list(app.query_one("#transcript", TranscriptScroll).children)
        fold_index = next(
            index
            for index, child in enumerate(children)
            if isinstance(child, FoldedProcessCell)
        )
        assistant_index = next(
            index
            for index, child in enumerate(children)
            if isinstance(child, TranscriptCell) and child.has_class("assistant")
        )
        result_index = next(
            index
            for index, child in enumerate(children)
            if (
                isinstance(child, ExpandableTranscriptCell)
                and "run_1" in child.details
                and index > assistant_index
            )
        )
        fold_cell = children[fold_index]
        assert isinstance(fold_cell, FoldedProcessCell)
        assert fold_cell.collapsed is True
        assert len(fold_cell.cells) == 4
        user_index = next(
            index
            for index, child in enumerate(children)
            if isinstance(child, TranscriptCell) and child.has_class("user")
        )
        assert fold_index == user_index + 1
        assert any(
            isinstance(cell, ExpandableTranscriptCell)
            and cell.detail_title == "reasoning_details"
            and "thinking first" in cell.details
            for cell in fold_cell.cells
        )
        assert any(
            isinstance(cell, ExpandableTranscriptCell)
            and "print(" in cell.details
            and "'ok'" in cell.details
            for cell in fold_cell.cells
        )
        assert any(
            isinstance(cell, TranscriptCell)
            and cell.has_class("assistant")
            and cell.copy_text == "I will inspect first."
            for cell in fold_cell.cells
        )
        result_cell = children[result_index]
        assert isinstance(result_cell, ExpandableTranscriptCell)
        assert "print('ok')" not in result_cell.details
        assert fold_index < assistant_index < result_index
        assert children[assistant_index].copy_text == "I will inspect first."


@pytest.mark.asyncio
async def test_tui_live_tool_call_and_result_are_separate_cells(
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
        "script_id": "scr_live",
        "run_id": "run_live",
        "returncode": 0,
        "timed_out": False,
        "interrupted": False,
        "truncated": False,
        "stdout": "ok\n",
        "stderr": "",
        "events": [],
        "run_log_path": "",
    }

    async with app.run_test(size=(120, 30)) as pilot:
        app._append_tool_started(
            {
                "call": {
                    "call_id": "call_live",
                    "name": "run_python",
                    "arguments": '{"code":"print(42)\\nprint(43)"}',
                },
                "tool_call_index": 0,
            }
        )
        app._append_tool_output(
            {
                "call": {
                    "call_id": "call_live",
                    "name": "run_python",
                    "arguments": '{"code":"print(42)\\nprint(43)"}',
                },
                "output": {"output": __import__("json").dumps(payload)},
            }
        )
        await pilot.pause()

        call_cell, result_cell = app.query(ExpandableTranscriptCell).nodes
        assert "print(42)" in str(call_cell.render())
        assert "print(43)" in call_cell.details
        assert "run_live" in result_cell.details
        assert "ok" in result_cell.details
        assert "print(42)" not in str(result_cell.render())
        assert "print(43)" not in result_cell.details


@pytest.mark.asyncio
async def test_tui_live_multiple_tool_calls_keep_call_result_boundaries(
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
        for index in range(2):
            call = {
                "call_id": f"call_{index}",
                "name": "run_python",
                "arguments": f'{{"code":"print({index})"}}',
            }
            payload = {
                "script_id": f"scr_{index}",
                "run_id": f"run_{index}",
                "returncode": 0,
                "timed_out": False,
                "interrupted": False,
                "truncated": False,
                "stdout": f"out {index}\n",
                "stderr": "",
                "events": [],
                "run_log_path": "",
            }
            app._append_tool_started({"call": call, "tool_call_index": index})
            app._append_tool_output(
                {
                    "call": call,
                    "tool_call_index": index,
                    "output": {"output": __import__("json").dumps(payload)},
                }
            )
        await pilot.pause()

        cells = app.query(ExpandableTranscriptCell).nodes
        assert len(cells) == 4
        assert "print(0)" in cells[0].details
        assert "run_0" in cells[1].details
        assert "print(0)" not in cells[1].details
        assert "print(1)" in cells[2].details
        assert "run_1" in cells[3].details
        assert "print(1)" not in cells[3].details


@pytest.mark.asyncio
async def test_tui_live_rounds_do_not_duplicate_reasoning_or_merge_final_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = StableRoundEngine(fake_engine(project_root, tmp_path / "state"))
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(120, 30)) as pilot:
        app._start_turn("go")
        await pilot.pause()
        await pilot.pause()

        transcript = app.query_one("#transcript", TranscriptScroll)
        children = list(transcript.children)
        reasoning_cells = [
            child
            for child in children
            if isinstance(child, ExpandableTranscriptCell) and child.detail_title == "reasoning_details"
        ]
        fold_cells = [
            child
            for child in children
            if isinstance(child, FoldedProcessCell)
        ]
        assistant_cells = [
            child
            for child in children
            if isinstance(child, TranscriptCell) and child.has_class("assistant")
        ]
        assert len(reasoning_cells) == 1
        assert len(fold_cells) == 1
        assert fold_cells[0].collapsed is True
        assert reasoning_cells[0].details.count("provider reasoning") == 1
        assert reasoning_cells[0].has_class("process_fold_hidden")
        assert assistant_cells[0].has_class("process_fold_hidden")
        assert any(
            isinstance(cell, ExpandableTranscriptCell)
            and "print(1)" in str(cell.render())
            and cell.has_class("process_fold_hidden")
            for cell in fold_cells[0].cells
        )
        assert [cell.copy_text for cell in assistant_cells] == ["I will inspect first.", "Done."]
        user_cell = next(child for child in children if isinstance(child, TranscriptCell) and child.has_class("user"))
        assert children.index(fold_cells[0]) == children.index(user_cell) + 1
        assert children.index(fold_cells[0]) < children.index(assistant_cells[0])
        assert children.index(fold_cells[0]) < children.index(assistant_cells[1])

        stored = engine.thread_store.read(str(app.thread_id))
        assert [event["type"] for event in stored].count("item.model_response") == 2
        assert not any(event["type"] == "item.tool_call" for event in stored)


@pytest.mark.asyncio
async def test_tui_process_fold_expands_original_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = StableRoundEngine(fake_engine(project_root, tmp_path / "state"))
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(120, 30)) as pilot:
        app._start_turn("go")
        await pilot.pause()
        await pilot.pause()

        fold_cell = next(
            child
            for child in app.query_one("#transcript", TranscriptScroll).children
            if isinstance(child, FoldedProcessCell)
        )
        assert all(cell.has_class("process_fold_hidden") for cell in fold_cell.cells)

        await pilot.click(fold_cell)
        await pilot.pause()

        assert fold_cell.collapsed is False
        assert all(not cell.has_class("process_fold_hidden") for cell in fold_cell.cells)
        assert any(
            isinstance(cell, ExpandableTranscriptCell)
            and cell.detail_title == "reasoning_details"
            and "provider reasoning" in cell.details
            for cell in fold_cell.cells
        )
        assert any(
            isinstance(cell, ExpandableTranscriptCell)
            and "print(1)" in str(cell.render())
            for cell in fold_cell.cells
        )
        assert any(
            isinstance(cell, ExpandableTranscriptCell)
            and "run_1" in cell.details
            for cell in fold_cell.cells
        )
        assert any(
            isinstance(cell, TranscriptCell)
            and cell.has_class("assistant")
            and cell.copy_text == "I will inspect first."
            for cell in fold_cell.cells
        )

        await pilot.press("ctrl+g")
        await pilot.pause()

        assert fold_cell.collapsed is True
        assert all(cell.has_class("process_fold_hidden") for cell in fold_cell.cells)

        await pilot.press("ctrl+g")
        await pilot.pause()

        assert fold_cell.collapsed is False
        assert all(not cell.has_class("process_fold_hidden") for cell in fold_cell.cells)


@pytest.mark.asyncio
async def test_tui_ctrl_g_toggles_lowest_visible_process_fold_only(
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
        top_process = [
            app._append_cell("[dim]top reasoning[/dim]", "reasoning"),
            app._append_cell("[dim]top tool[/dim]", "event"),
        ]
        top_fold = app._append_process_fold_cell(top_process, collapsed=True)
        bottom_process = [
            app._append_cell("[dim]bottom reasoning[/dim]", "reasoning"),
            app._append_cell("[dim]bottom tool[/dim]", "event"),
        ]
        bottom_fold = app._append_process_fold_cell(bottom_process, collapsed=True)
        await pilot.pause()

        assert top_fold in app._visible_process_fold_cells()
        assert bottom_fold in app._visible_process_fold_cells()

        await pilot.press("ctrl+g")
        await pilot.pause()

        assert top_fold.collapsed is True
        assert all(cell.has_class("process_fold_hidden") for cell in top_process)
        assert bottom_fold.collapsed is False
        assert all(not cell.has_class("process_fold_hidden") for cell in bottom_process)

        await pilot.press("ctrl+g")
        await pilot.pause()

        assert top_fold.collapsed is True
        assert all(cell.has_class("process_fold_hidden") for cell in top_process)
        assert bottom_fold.collapsed is True
        assert all(cell.has_class("process_fold_hidden") for cell in bottom_process)


@pytest.mark.asyncio
async def test_tui_clears_initial_thinking_when_response_has_no_reasoning(
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
        app._reasoning_cell = app._append_cell("[dim]thinking...[/dim]", "event")
        app._sync_run_state_from_active(run_state)

        await app._handle_thread_event(
            "thread_1",
            "assistant.reasoning_absent",
            {},
            run_state,
        )
        await pilot.pause()

        assert app._reasoning_cell is None
        assert "thinking" not in "\n".join(str(child.render()) for child in app.query_one("#transcript", TranscriptScroll).children)


@pytest.mark.asyncio
async def test_tui_thread_resume_pages_older_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    state = tmp_path / "state"
    engine = fake_engine(project_root, state)
    thread_id = engine.thread_store.create_thread("Long compacted work")
    engine.thread_store.append(
        thread_id,
        "item.user",
        turn_id="turn_1",
        item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "old segment"}],
        },
    )
    engine.thread_store.append(thread_id, "item.compaction", turn_id="turn_1", text="summary one")
    engine.thread_store.append(
        thread_id,
        "item.user",
        turn_id="turn_2",
        item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "middle segment"}],
        },
    )
    engine.thread_store.append(thread_id, "item.compaction", turn_id="turn_2", text="summary two")
    engine.thread_store.append(
        thread_id,
        "item.user",
        turn_id="turn_3",
        item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "new segment"}],
        },
    )
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(120, 30)) as pilot:
        app._resume_thread(thread_id)
        await pilot.pause()

        transcript_text = "\n".join(
            str(getattr(child, "copy_text", None) or "")
            for child in app.query_one("#transcript", TranscriptScroll).children
        )
        assert "old segment" not in transcript_text
        assert "middle segment" not in transcript_text
        assert "new segment" in transcript_text
        assert app._history_has_more is True

        app._load_older_thread_history()
        await pilot.pause()

        transcript_text = "\n".join(
            str(getattr(child, "copy_text", None) or "")
            for child in app.query_one("#transcript", TranscriptScroll).children
        )
        assert "old segment" not in transcript_text
        assert "middle segment" in transcript_text
        assert "new segment" in transcript_text
        assert app._history_has_more is True

        app._load_older_thread_history()
        await pilot.pause()

        transcript_text = "\n".join(
            str(getattr(child, "copy_text", None) or "")
            for child in app.query_one("#transcript", TranscriptScroll).children
        )
        assert "old segment" in transcript_text
        assert app._history_has_more is False


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
        cell = app.query_one(ExpandableTranscriptCell)
        assert "print(1)" in cell.details


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

        composer.insert("1\n2\n3")
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
async def test_tui_composer_expands_on_three_visual_lines(
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

    async with app.run_test(size=(40, 30)) as pilot:
        composer = app.query_one("#composer", TextArea)

        composer.insert("x" * (composer.wrap_width * 2 + 1))
        await pilot.pause()

        assert app._composer_visual_line_count(composer) == 3
        assert app._composer_expanded is True
        assert composer.styles.height.value == 13


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

        for _ in range(20):
            await pilot.pause()
            if panel.items:
                break
        await pilot.press("e")
        await pilot.press("x")
        await pilot.press("enter")
        await pilot.pause()

        assert composer.text == "@src/example.py "
        assert app.thread_id is None
        assert app._transcript_has_content is False


@pytest.mark.asyncio
async def test_tui_file_mention_scan_skips_dot_directories_with_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    source_dir = project_root / "src"
    dot_dir = project_root / ".hidden"
    dotted_dir = project_root / "version.1"
    source_dir.mkdir(parents=True)
    dot_dir.mkdir(parents=True)
    dotted_dir.mkdir(parents=True)
    (source_dir / "example.py").write_text("print('hi')\n", encoding="utf-8")
    (dot_dir / "secret.py").write_text("print('secret')\n", encoding="utf-8")
    (dotted_dir / "inside.py").write_text("print('inside')\n", encoding="utf-8")
    monkeypatch.setattr(
        "uv_agent.tui.app.create_engine",
        lambda root: fake_engine(root, tmp_path / "state"),
    )
    app = UvAgentApp(project_root=project_root)

    items = app._file_mention_items()

    assert any(item.title == ".hidden/" and item.id == ".hidden/" for item in items)
    assert any(item.title == "version.1/" and item.id == "version.1/" for item in items)
    assert any(item.title == "version.1/inside.py" for item in items)
    assert any(item.title == "src/" and item.id == "src/" for item in items)
    assert any(item.title == "src/example.py" for item in items)
    assert not any(item.title == ".hidden/secret.py" for item in items)

    async with app.run_test(size=(90, 24)) as pilot:
        app._open_picker(
            app._text("mention_files"),
            items,
            app._choose_file_mention,
            mention_kind="file",
            mention_items=app._mention_picker_items,
        )
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        option_list = panel.query_one("#panel-content", OptionList)
        first_option = option_list.get_option_at_index(0)
        assert first_option.disabled is False
        await pilot.press("enter")
        await pilot.pause()

        composer = app.query_one("#composer", TextArea)
        assert composer.text == "@.hidden/ "


@pytest.mark.asyncio
async def test_tui_file_mention_opens_before_background_scan_completes(
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
    scan_started = threading.Event()
    release_scan = threading.Event()
    app = UvAgentApp(project_root=project_root)
    original_iter = app._iter_file_mention_items

    def slow_iter(root: Path, *, generation: int | None):
        scan_started.set()
        release_scan.wait(2)
        yield from original_iter(root, generation=generation)

    app._iter_file_mention_items = slow_iter  # type: ignore[method-assign]

    async with app.run_test(size=(90, 24)) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.insert("@")
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("mention_files")
        assert app._text("mention_scanning") in panel.subtitle
        assert panel.items == []
        assert scan_started.wait(1)

        release_scan.set()
        for _ in range(20):
            await pilot.pause()
            if any(item.title == "src/example.py" for item in panel.items):
                break

        assert any(item.title == "src/example.py" for item in panel.items)
        assert app._text("mention_cached") in panel.subtitle


@pytest.mark.asyncio
async def test_tui_file_mention_dirty_cache_rescans_on_next_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    source_dir = project_root / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "old.py").write_text("print('old')\n", encoding="utf-8")
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
        for _ in range(20):
            await pilot.pause()
            if app._mention_file_cache.complete:
                break
        assert any(item.title == "src/old.py" for item in panel.items)

        await pilot.press("escape")
        await pilot.pause()
        composer.load_text("")
        await pilot.pause()
        (source_dir / "new.py").write_text("print('new')\n", encoding="utf-8")
        app._mark_file_mention_cache_dirty()

        composer.insert("@")
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        for _ in range(20):
            await pilot.pause()
            if any(item.title == "src/new.py" for item in panel.items):
                break

        assert any(item.title == "src/new.py" for item in panel.items)


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
        for _ in range(20):
            await pilot.pause()
            if file_panel.items:
                break

        await pilot.press("@")
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("mention_threads")
        for _ in range(20):
            await pilot.pause()
            if panel.items:
                break

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
        for _ in range(20):
            await pilot.pause()
            panel = app.screen_stack[-1]
            if isinstance(panel, FullscreenPanel) and panel.items:
                break
        await pilot.press("@")
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("mention_threads")

        await pilot.press("backspace")
        await pilot.pause()

        assert panel.panel_title == app._text("mention_files")
        await pilot.press("e")
        await pilot.press("x")
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
        ui=UiConfig(
            language="zh-CN",
            completion_notification=CompletionNotificationConfig(enabled=False),
        ),
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
            ui=UiConfig(
                language="zh-CN",
                completion_notification=CompletionNotificationConfig(enabled=False),
            ),
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
async def test_tui_config_panel_toggles_completion_notification(
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

    async with app.run_test(size=(90, 24), notifications=True) as pilot:
        app._open_config_panel()
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        ids = [item.id for item in panel.items]
        panel.query_one("#panel-content", OptionList).highlighted = ids.index(
            "completion_notification"
        )

        await pilot.press("enter")
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("config")
        ids = [item.id for item in panel.items]
        item = panel.items[ids.index("completion_notification")]
        assert item.description == "on"

        data = __import__("json").loads(config_path.read_text(encoding="utf-8"))
        assert data["ui"]["completion_notification"]["enabled"] is True


@pytest.mark.asyncio
async def test_tui_config_toggle_refreshes_without_escape_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(
        "uv_agent.tui.app.create_engine",
        lambda root: fake_engine(root, tmp_path / "state"),
    )
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24), notifications=True) as pilot:
        command_palette_title = app._text("command_palette")
        await pilot.press("ctrl+p")
        await pilot.press("c")
        await pilot.press("enter")
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("config")

        for expected in ("on", "off"):
            ids = [item.id for item in panel.items]
            panel.query_one("#panel-content", OptionList).highlighted = ids.index(
                "completion_notification"
            )

            await pilot.press("enter")
            await pilot.pause()

            assert app.screen_stack[-1] is panel
            assert panel.panel_title == app._text("config")
            ids = [item.id for item in panel.items]
            item = panel.items[ids.index("completion_notification")]
            assert item.description == expected

        await pilot.press("escape")
        await pilot.pause()

        assert app.screen_stack[-1] is panel
        assert panel.panel_title == command_palette_title


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
                "runtime": {"default_level": "medium", "compression": {"enabled": False}},
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
    monkeypatch.setattr("uv_agent.tui.app.application_version", lambda: "9.8.7")
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        app._open_status_panel()
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert panel.panel_title == app._text("status")
        assert panel.picker_mode is False
        assert "- version: [cyan]9.8.7[/cyan]" in panel.body
        assert "258K" in panel.body
        assert "AGENTS.md" in panel.body
        assert "Use local rules" not in panel.body
        assert "print('hi')" in panel.body


@pytest.mark.asyncio
async def test_tui_turn_completion_notification_uses_configured_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = ImageCaptureEngine(fake_engine(project_root, tmp_path / "state"))
    engine.config = AppConfig(
        providers=engine.config.providers,
        models=engine.config.models,
        levels=engine.config.levels,
        runtime=engine.config.runtime,
        runner=engine.config.runner,
        ui=UiConfig(
            language="en",
            completion_notification=CompletionNotificationConfig(
                enabled=True,
                terminal=True,
                bell=True,
            ),
        ),
    )
    bell_count = 0

    def fake_bell(app: UvAgentApp) -> None:
        nonlocal bell_count
        bell_count += 1

    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    monkeypatch.setattr("uv_agent.tui.app.play_completion_sound", lambda: True)
    monkeypatch.setattr(UvAgentApp, "bell", fake_bell)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24), notifications=True) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.insert("inspect")
        await pilot.press("ctrl+enter")
        await pilot.pause(0.2)

        assert not list(app.screen.query("Toast"))
        assert bell_count == 1


@pytest.mark.asyncio
async def test_tui_terminal_completion_event_only_for_background_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = ReleasableEngine(fake_engine(project_root, tmp_path / "state"))
    engine.config = AppConfig(
        providers=engine.config.providers,
        models=engine.config.models,
        levels=engine.config.levels,
        runtime=engine.config.runtime,
        runner=engine.config.runner,
        ui=UiConfig(
            language="en",
            completion_notification=CompletionNotificationConfig(
                enabled=True,
                terminal=True,
                bell=False,
            ),
        ),
    )
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24), notifications=True) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.insert("old work")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        old_thread = app.thread_id
        assert old_thread is not None
        await engine.started[old_thread].wait()

        composer.insert("/clear")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert app.thread_id is None

        engine.release[old_thread].set()
        await pilot.pause(0.2)

        transcript_text = "\n".join(
            str(child.render())
            for child in app.query_one("#transcript", TranscriptScroll).children
            if isinstance(child, TranscriptCell)
        )
        assert app._text("background_thread_completed") in transcript_text
        assert short_thread(old_thread) in transcript_text
        assert "done old work" not in transcript_text

        composer.insert("current work")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        current_thread = app.thread_id
        assert current_thread is not None
        await engine.started[current_thread].wait()

        before_count = transcript_text.count(app._text("background_thread_completed"))
        engine.release[current_thread].set()
        await pilot.pause(0.2)

        after_text = "\n".join(
            str(child.render())
            for child in app.query_one("#transcript", TranscriptScroll).children
            if isinstance(child, TranscriptCell)
        )
        assert after_text.count(app._text("background_thread_completed")) == before_count
        assert not list(app.screen.query("Toast"))


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
async def test_tui_mouse_drag_on_first_line_does_not_select_next_line(
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
        cell = app._append_cell(
            "first line\nsecond line",
            "event",
            copy_text="first line\nsecond line",
        )
        await pilot.pause(0.2)

        await pilot.mouse_down(cell, offset=(0, 0))
        await pilot.hover(cell, offset=(40, 0))
        await pilot.pause()

        assert app.screen.get_selected_text() == "first line"


@pytest.mark.asyncio
async def test_tui_click_chain_threshold_is_stricter_than_textual_default(
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
        await pilot.mouse_up(app._assistant_cell, offset=(1, 0))
        await pilot.pause(0.35)
        await pilot.mouse_down(app._assistant_cell, offset=(1, 0))
        await pilot.mouse_up(app._assistant_cell, offset=(1, 0))
        await pilot.pause()

        assert app._chained_clicks == 1
        assert app.screen.get_selected_text() is None


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
async def test_tui_clear_while_current_thread_runs_keeps_old_thread_backgrounded(
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

        composer.insert("/clear")
        await pilot.press("ctrl+enter")
        await pilot.pause()

        assert app.thread_id is None
        assert old_thread in app._thread_runs
        assert app.busy is False
        footer = app.query_one("#composer-footer", Static)
        assert app._text("background_active") in str(footer.content)

        app._open_status_panel()
        await pilot.pause()
        panel = app.screen_stack[-1]
        assert isinstance(panel, FullscreenPanel)
        assert app._text("active_threads") in panel.body
        assert old_thread[-8:] in panel.body
        await pilot.press("escape")
        await pilot.pause()

        composer.insert("new work")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        new_thread = app.thread_id
        assert new_thread is not None
        assert new_thread != old_thread
        assert new_thread in app._thread_runs

        engine.release[old_thread].set()
        engine.release[new_thread].set()
        await pilot.pause(0.2)

        assert old_thread not in app._thread_runs
        assert new_thread not in app._thread_runs


@pytest.mark.asyncio
async def test_tui_resume_running_thread_rebinds_live_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    engine = fake_engine(project_root, tmp_path / "state")
    monkeypatch.setattr("uv_agent.tui.app.create_engine", lambda root: engine)
    app = UvAgentApp(project_root=project_root)

    async with app.run_test(size=(90, 24)) as pilot:
        thread_id = engine.thread_store.create_thread("Running")
        app.thread_id = thread_id
        run_state = app._run_state_for_thread(thread_id)
        run_state.worker = object()  # type: ignore[assignment]
        app._append_user("inspect")
        await app._handle_thread_event(
            thread_id,
            "assistant.delta",
            {"text": "I will run "},
            run_state,
        )
        await app._handle_thread_event(
            thread_id,
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

        app._handle_command("/clear")
        await pilot.pause()
        assert app.thread_id is None
        assert run_state.assistant_cell is None
        assert run_state.tool_delta_cells == {}

        app._resume_thread(thread_id)
        await pilot.pause()
        assert app.thread_id == thread_id
        assert app.query(".assistant").nodes
        assert app.query(".tool_pending").nodes
        assert "print(1)" in str(app.query(".tool_pending").nodes[-1].render())

        await app._handle_thread_event(
            thread_id,
            "assistant.delta",
            {"text": "more"},
            run_state,
        )
        await app._handle_thread_event(
            thread_id,
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

        assistant = app.query(".assistant").nodes[-1]
        assert isinstance(assistant, TranscriptCell)
        assert assistant.copy_text == "I will run more"
        assert len(app.query(".tool_pending").nodes) == 1
        cell = app.query_one(ExpandableTranscriptCell)
        assert "print(1)" in cell.details


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
async def test_tui_tool_call_details_highlight_python_source(
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
    code = "for item in ['alpha', 'beta']:\n    print(item)\n"
    call = {
        "call_id": "call_1",
        "name": "run_python",
        "arguments": __import__("json").dumps({"code": code}),
    }

    async with app.run_test(size=(90, 24)) as pilot:
        app._append_tool_started({"call": call})
        await pilot.pause()
        cell = app.query_one(ExpandableTranscriptCell)

        await pilot.click(cell)
        await pilot.pause()

        panel = app.screen_stack[-1]
        assert isinstance(panel, ToolDetailsPanel)
        content = panel.query_one("#panel-body-content", Static)
        rendered = str(content.render())
        assert "for item in ['alpha', 'beta'" in rendered
        assert "print(item)" in rendered
        assert "[bold #7dd3fc]for[/bold #7dd3fc]" in panel.body
        assert "[#fbbf24]'alpha'[/#fbbf24]" in panel.body


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
    write_png(image)
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
        write_png(image, color=(index * 120, 80, 200))
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
        assert "image-1" in str(panel.query_one("#image-preview-meta", Static).render())
        assert Path(panel.query_one("#image-preview", TerminalImage).image or "").name.startswith("img_")

        await pilot.press("k")
        await pilot.pause()

        assert panel.index == 0
        assert "image-0" in str(panel.query_one("#image-preview-meta", Static).render())
        assert Path(panel.query_one("#image-preview", TerminalImage).image or "").name.startswith("img_")

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
