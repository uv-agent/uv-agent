from __future__ import annotations

import ctypes
import json
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from rich.markdown import Markdown
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.reactive import reactive
from textual.widgets import Input, OptionList, Static, TextArea
from textual.widgets._option_list import Option

from uv_agent.app_factory import create_engine
from uv_agent.config import ConfigError, config_sources, load_raw_config, redact_config
from uv_agent.mcp_config import discover_mcp_servers
from uv_agent.paths import project_state_dir, uv_agent_home
from uv_agent.skills import discover_skills
from uv_agent.tui.formatting import format_tokens, parse_tool_payload, short_thread, tool_result_markup


@dataclass(frozen=True)
class PickerItem:
    id: str
    title: str
    description: str = ""
    meta: str = ""


class FullscreenPanel(ModalScreen[str | None]):
    """Scrollable full-screen panel/picker."""

    CSS = """
    FullscreenPanel {
        align: center middle;
        background: #05070acc;
    }

    #panel-shell {
        width: 92%;
        height: 88%;
        max-width: 120;
        border: round #3a4a60;
        background: #0c1118;
        padding: 1 2;
    }

    #panel-header {
        height: 1;
        color: #dce7f3;
        text-style: bold;
    }

    #panel-subtitle {
        height: 1;
        color: #8fa2b8;
    }

    #panel-filter {
        height: 3;
        margin: 1 0 0 0;
        border: tall #263649;
        background: #0f1721;
        color: #e9eef5;
    }

    #panel-content {
        height: 1fr;
        margin: 1 0 0 0;
        border: tall #1f2b3a;
        background: #0a0f15;
        padding: 0 1;
    }

    #panel-body {
        height: 1fr;
        margin: 1 0 0 0;
        border: tall #1f2b3a;
        background: #0a0f15;
        padding: 1 1;
    }

    #panel-footer {
        height: 1;
        margin: 1 0 0 0;
        color: #7b8796;
    }

    OptionList {
        height: 1fr;
        border: none;
        background: #0a0f15;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_panel", "Close", priority=True),
        Binding("ctrl+c", "dismiss_panel", "Close", priority=True),
    ]

    def __init__(
        self,
        *,
        title: str,
        body: str = "",
        items: list[PickerItem] | None = None,
        subtitle: str = "",
    ) -> None:
        super().__init__()
        self.panel_title = title
        self.body = body
        self.items = items or []
        self.subtitle = subtitle
        self._filtered = list(self.items)

    def compose(self) -> ComposeResult:
        with Vertical(id="panel-shell"):
            yield Static(self.panel_title, id="panel-header")
            yield Static(self.subtitle, id="panel-subtitle")
            if self.items:
                yield Input(placeholder="Filter...", id="panel-filter")
                yield OptionList(id="panel-content")
            else:
                yield VerticalScroll(Static(self.body, markup=True), id="panel-body")
            yield Static("Esc close · arrows scroll/select · Enter open", id="panel-footer")

    def on_mount(self) -> None:
        if self.items:
            self._refresh_options()
            self.query_one("#panel-filter", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "panel-filter":
            return
        query = event.value.casefold().strip()
        if not query:
            self._filtered = list(self.items)
        else:
            self._filtered = [
                item
                for item in self.items
                if query in (item.title + " " + item.description + " " + item.meta).casefold()
            ]
        self._refresh_options()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id:
            self.dismiss(event.option_id)

    def action_dismiss_panel(self) -> None:
        self.dismiss(None)

    def _refresh_options(self) -> None:
        options = [
            Option(
                f"[bold cyan]{escape(item.title)}[/bold cyan]"
                + (f"\n[dim]{escape(item.description)}[/dim]" if item.description else "")
                + (f"\n[dim]{escape(item.meta)}[/dim]" if item.meta else ""),
                id=item.id,
            )
            for item in self._filtered
        ]
        if not options:
            options = [Option("[dim]No matches[/dim]", id="")]
        self.query_one("#panel-content", OptionList).set_options(options)


@dataclass(frozen=True)
class CommandSpec:
    name: str
    usage: str
    description: str


COMMANDS = [
    CommandSpec("/new", "/new [title]", "start a named thread"),
    CommandSpec("/threads", "/threads", "show recent threads"),
    CommandSpec("/status", "/status", "open runtime status"),
    CommandSpec("/context", "/context", "show token budget and AGENTS rules"),
    CommandSpec("/rules", "/rules", "show loaded AGENTS instructions"),
    CommandSpec("/config", "/config", "show config sources"),
    CommandSpec("/models", "/models", "show levels and models"),
    CommandSpec("/level", "/level [name]", "switch model level"),
    CommandSpec("/mcp", "/mcp", "show MCP declarations"),
    CommandSpec("/skills", "/skills", "show discovered skills"),
    CommandSpec("/skill", "/skill [name]", "preview a skill file"),
    CommandSpec("/runs", "/runs", "show latest Python run"),
    CommandSpec("/panel", "/panel", "close the open panel"),
    CommandSpec("/clear", "/clear", "clear transcript and queue"),
    CommandSpec("/quit", "/quit", "quit with confirmation"),
    CommandSpec("/help", "/help", "show all commands"),
]


class EmptyState(Static):
    """Animated empty transcript state."""

    FRAMES = ["·  ", "·· ", "···", " ··", "  ·", "   "]

    DEFAULT_CSS = """
    EmptyState {
        width: 100%;
        height: 100%;
        content-align: center middle;
        color: #7f91a8;
    }

    EmptyState.hidden {
        display: none;
    }
    """

    def __init__(self, *, id: str) -> None:
        super().__init__("", id=id)
        self.frame = 0

    def tick(self) -> None:
        frame = self.FRAMES[self.frame % len(self.FRAMES)]
        self.frame += 1
        self.update(
            f"[bold #dce7f3]Ready[/bold #dce7f3]\n"
            f"[dim]Python runner only {escape(frame)}[/dim]\n"
            "[dim]Type / for commands or Ctrl+O for threads[/dim]"
        )


class CommandSuggestions(Static):
    """Command picker shown above the composer."""

    DEFAULT_CSS = """
    CommandSuggestions {
        height: auto;
        max-height: 9;
        border: tall #34465d;
        padding: 1 1;
        margin: 0 0 1 0;
        background: #101923;
        color: #d6e2ef;
    }

    CommandSuggestions.hidden {
        display: none;
    }
    """

    def __init__(self, *, id: str) -> None:
        super().__init__("", id=id, classes="hidden")
        self.matches: list[CommandSpec] = []
        self.selected_index = 0

    def set_matches(self, matches: list[CommandSpec]) -> None:
        self.matches = matches
        self.selected_index = 0
        self._refresh_options()
        self.remove_class("hidden")

    def clear(self) -> None:
        self.matches = []
        self.selected_index = 0
        self.update("")
        self.add_class("hidden")

    def move(self, delta: int) -> None:
        if not self.matches:
            return
        self.selected_index = (self.selected_index + delta) % len(self.matches)
        self._refresh_options()

    def choose(self) -> str | None:
        if not self.matches:
            return None
        return self.matches[self.selected_index].name

    def _refresh_options(self) -> None:
        lines = ["[bold]commands[/bold] [dim]Tab/Enter select · Esc close[/dim]"]
        visible_start = max(0, min(self.selected_index - 6, max(0, len(self.matches) - 8)))
        visible = self.matches[visible_start : visible_start + 8]
        if visible_start:
            lines.append(f"[dim]... {visible_start} above[/dim]")
        for offset, spec in enumerate(visible):
            index = visible_start + offset
            active = index == self.selected_index
            prefix = "[reverse]›[/reverse]" if active else " "
            command_style = "bold cyan" if active else "cyan"
            desc_style = "white" if active else "dim"
            lines.append(
                f"{prefix} [{command_style}]{escape(spec.usage):<18}[/{command_style}] "
                f"[{desc_style}]{escape(spec.description)}[/{desc_style}]"
            )
        remaining = len(self.matches) - visible_start - len(visible)
        if remaining > 0:
            lines.append(f"[dim]... {remaining} below[/dim]")
        self.update("\n".join(lines))


class TranscriptCell(Static):
    """Small transcript block used by the Textual chat timeline."""

    DEFAULT_CSS = """
    TranscriptCell {
        width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
    }

    TranscriptCell.user {
        background: #111a24;
        color: #dce7f3;
        border-left: solid #2e9ad8;
    }

    TranscriptCell.assistant {
        background: #0f151d;
        color: #e5e7eb;
        border-left: solid #516071;
    }

    TranscriptCell.event {
        background: #0e141b;
        color: #aeb7c4;
        border-left: solid #2b3542;
    }

    TranscriptCell.error {
        background: #241316;
        color: #ffb4b4;
        border-left: solid #e26363;
    }
    """


class UvAgentApp(App[None]):
    CSS = """
    Screen {
        layout: horizontal;
        background: #0b0f14;
        color: #d8dee9;
    }

    #main-column {
        width: 1fr;
        min-width: 0;
        height: 100%;
        background: #0b0f14;
    }

    #transcript {
        height: 1fr;
        min-height: 6;
        padding: 1 2 0 1;
        background: #0b0f14;
    }

    #bottom-pane {
        height: auto;
        max-height: 18;
        padding: 0 2 1 1;
        background: #0b0f14;
    }

    #composer-shell {
        height: auto;
        border: tall #2a3646;
        background: #0f1721;
        padding: 0 1;
    }

    #composer-shell.busy {
        border: round #3d516b;
    }

    #composer-shell.command-mode {
        border: tall #28708f;
    }

    #composer-row {
        height: auto;
        min-height: 3;
    }

    #composer-left {
        width: 10;
        min-width: 7;
        height: 3;
        content-align: center middle;
        color: #8fa2b8;
        border-right: tall #263649;
    }

    #composer {
        width: 1fr;
        height: 3;
        min-height: 3;
        max-height: 8;
        border: none;
        padding: 0 1;
        background: #0f1721;
        color: #edf2f7;
    }

    #composer:focus {
        border: none;
    }

    #composer-right {
        width: 14;
        min-width: 10;
        height: 3;
        content-align: center middle;
        color: #8fa2b8;
        border-left: tall #263649;
    }
    """

    BINDINGS = [
        Binding("ctrl+enter", "submit_composer", "Send", priority=True),
        Binding("ctrl+j", "submit_composer", "Send", priority=True),
        Binding("tab", "complete_command", "Complete", priority=True),
        Binding("up", "command_up", "Command up", priority=True),
        Binding("down", "command_down", "Command down", priority=True),
        Binding("enter", "command_accept", "Command accept", priority=True),
        Binding("ctrl+s", "toggle_status_panel", "Status", priority=True),
        Binding("ctrl+o", "open_threads", "Threads", priority=True),
        Binding("ctrl+p", "open_command_palette", "Commands", priority=True),
        Binding("ctrl+q", "request_quit", "Quit", priority=True),
        Binding("ctrl+c", "request_quit", "Quit", priority=True),
        Binding("f1", "help", "Help", priority=True),
        Binding("escape", "clear_input", "Clear"),
    ]

    busy = reactive(False)

    def __init__(self, *, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.engine = create_engine(project_root)
        self.thread_id: str | None = None
        self.level: str | None = None
        self._assistant_buffer = ""
        self._assistant_cell: TranscriptCell | None = None
        self._tool_cells: dict[str, TranscriptCell] = {}
        self._queue: list[str] = []
        self._last_status = "Idle"
        self._spinner_index = 0
        self._last_tool_payload: dict[str, object] | None = None
        self._quit_armed = False
        self._last_quit_request_at = 0.0
        self._previous_sigint_handler: Any = None
        self._windows_ctrl_handler: Any = None
        self._transcript_has_content = False

    def compose(self) -> ComposeResult:
        with Vertical(id="main-column"):
            with VerticalScroll(id="transcript"):
                yield EmptyState(id="empty-state")
            with Vertical(id="bottom-pane"):
                yield CommandSuggestions(id="command-suggestions")
                with Vertical(id="composer-shell"):
                    with Horizontal(id="composer-row"):
                        yield Static("idle", id="composer-left")
                        yield TextArea(
                            "",
                            placeholder="Ask, edit, or type / for commands",
                            id="composer",
                            compact=True,
                            soft_wrap=True,
                            show_line_numbers=False,
                        )
                        yield Static("ctx 0%", id="composer-right")

    def on_mount(self) -> None:
        self.query_one("#empty-state", EmptyState).tick()
        self._refresh_status("Idle")
        self.set_interval(0.16, self._tick, name="spinner")
        self.query_one("#composer", TextArea).focus()
        self._install_sigint_guard()

    def on_unmount(self) -> None:
        if self._previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._previous_sigint_handler)
        self._uninstall_windows_ctrl_guard()

    def on_resize(self) -> None:
        self._refresh_status()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "composer":
            return
        self._resize_composer(event.text_area.text)
        self._update_command_suggestions(event.text_area.text)
        self._refresh_status()

    def _tick(self) -> None:
        if not self._transcript_has_content:
            try:
                self.query_one("#empty-state", EmptyState).tick()
            except NoMatches:
                pass
        if self.busy:
            self._refresh_status()

    def action_submit_composer(self) -> None:
        composer = self.query_one("#composer", TextArea)
        prompt = composer.text.strip()
        if not prompt:
            self._flash("Write something first")
            return
        composer.load_text("")
        self._hide_command_suggestions()
        self._resize_composer("")
        if "\n" not in prompt and self._handle_command(prompt):
            return
        if self.busy:
            self._queue.append(prompt)
            self._append_cell(f"[dim]queued[/dim]\n{escape(prompt)}", "event")
            self._refresh_status()
            return
        self._start_turn(prompt)

    def _start_turn(self, prompt: str) -> None:
        self.busy = True
        self.query_one("#composer-shell", Vertical).add_class("busy")
        self._assistant_buffer = ""
        self._assistant_cell = None
        self._append_user(prompt)
        self._refresh_status("Working")
        self.run_worker(self._run_turn(prompt), exclusive=True, thread=False)

    async def _run_turn(self, prompt: str) -> None:
        try:
            async for item in self.engine.run_turn(
                user_text=prompt,
                thread_id=self.thread_id,
                level=self.level,
            ):
                self.thread_id = item.get("thread_id", self.thread_id)
                event_type = item["type"]
                if event_type == "assistant.delta":
                    await self._append_assistant_delta(item["text"])
                elif event_type == "model.response":
                    self._refresh_status("Reading")
                elif event_type == "tool.started":
                    self._append_tool_started(item)
                elif event_type == "tool.output":
                    self._append_tool_output(item)
                elif event_type == "turn.completed":
                    text = item["final_text"] or self._assistant_buffer
                    if text and self._assistant_cell is None:
                        await self._append_assistant_delta(text)
                    self._refresh_status("Idle")
        except Exception as exc:
            self._append_cell(f"[bold red]Error[/bold red] {escape(repr(exc))}", "error")
            self._refresh_status("Error")
        finally:
            self.busy = False
            self.query_one("#composer-shell", Vertical).remove_class("busy")
            if self._last_status != "Error":
                self._refresh_status("Idle")
            self.query_one("#composer", TextArea).focus()
            if self._queue:
                next_prompt = self._queue.pop(0)
                self._start_turn(next_prompt)

    def action_clear_input(self) -> None:
        if self._command_panel_visible():
            self._hide_command_suggestions()
            return
        composer = self.query_one("#composer", TextArea)
        if composer.text:
            composer.load_text("")
            self._resize_composer("")
            return

    def action_request_quit(self) -> None:
        now = monotonic()
        if now - self._last_quit_request_at < 0.35:
            return
        self._last_quit_request_at = now
        if self._quit_armed:
            self.exit()
            return
        draft = self.query_one("#composer", TextArea).text.strip()
        suffix = " · draft will be lost" if draft else ""
        self._quit_armed = True
        self._flash(f"Press Ctrl+Q or Ctrl+C again to quit{suffix}", severity="warning")
        self.set_timer(2.0, self._clear_quit_arm)

    def action_toggle_status_panel(self) -> None:
        self._open_status_panel()

    def action_open_threads(self) -> None:
        self._open_threads_panel()

    def action_open_command_palette(self) -> None:
        self._open_command_palette()

    def action_complete_command(self) -> None:
        panel = self.query_one("#command-suggestions", CommandSuggestions)
        if self._command_panel_visible():
            command = panel.choose()
            if command is not None:
                self._apply_command_completion(command)
            return
        composer = self.query_one("#composer", TextArea)
        replacement = self._command_completion(composer.text)
        if replacement is None:
            return
        composer.load_text(replacement)
        self._resize_composer(replacement)
        self._update_command_suggestions(replacement)

    def action_command_up(self) -> None:
        if self._command_panel_visible():
            self.query_one("#command-suggestions", CommandSuggestions).move(-1)
            return
        self.query_one("#composer", TextArea).action_cursor_up()

    def action_command_down(self) -> None:
        if self._command_panel_visible():
            self.query_one("#command-suggestions", CommandSuggestions).move(1)
            return
        self.query_one("#composer", TextArea).action_cursor_down()

    def action_command_accept(self) -> None:
        if self._command_panel_visible():
            command = self.query_one("#command-suggestions", CommandSuggestions).choose()
            if command is not None:
                self._apply_command_completion(command)
            return
        self.query_one("#composer", TextArea).insert("\n")

    def action_help(self) -> None:
        self._open_help_panel()

    def _clear_quit_arm(self) -> None:
        self._quit_armed = False
        self._refresh_status()

    def _install_sigint_guard(self) -> None:
        self._previous_sigint_handler = signal.getsignal(signal.SIGINT)

        def handle_sigint(signum: int, frame: object) -> None:
            self.call_from_thread(self.action_request_quit)

        signal.signal(signal.SIGINT, handle_sigint)
        self._install_windows_ctrl_guard()

    def _install_windows_ctrl_guard(self) -> None:
        if sys.platform != "win32":
            return
        ctrl_c_event = 0
        handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

        def handle_ctrl(control_type: int) -> bool:
            if control_type != ctrl_c_event:
                return False
            self.call_from_thread(self.action_request_quit)
            return True

        self._windows_ctrl_handler = handler_type(handle_ctrl)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(self._windows_ctrl_handler, True)

    def _uninstall_windows_ctrl_guard(self) -> None:
        if sys.platform != "win32" or self._windows_ctrl_handler is None:
            return
        ctypes.windll.kernel32.SetConsoleCtrlHandler(self._windows_ctrl_handler, False)
        self._windows_ctrl_handler = None

    def _handle_command(self, prompt: str) -> bool:
        command, _, rest = prompt.partition(" ")
        if command == "/clear":
            self.thread_id = None
            self._assistant_buffer = ""
            self._assistant_cell = None
            self._tool_cells.clear()
            self._queue.clear()
            self._reset_transcript()
            self._refresh_status("Idle")
            return True
        if command == "/quit":
            self.action_request_quit()
            return True
        if command == "/new":
            title = rest.strip() or "New thread"
            self.thread_id = self.engine.thread_store.create_thread(title)
            self._append_cell(
                f"[dim]new thread[/dim] [cyan]{escape(short_thread(self.thread_id))}[/cyan]",
                "event",
            )
            self._refresh_status("Idle")
            return True
        if command == "/threads":
            self._open_threads_panel()
            return True
        if command == "/status":
            self._open_status_panel()
            return True
        if command == "/context":
            self._open_context_panel()
            return True
        if command == "/rules":
            self._open_rules_panel()
            return True
        if command == "/config":
            self._open_config_panel()
            return True
        if command == "/models":
            self._open_models_panel()
            return True
        if command == "/mcp":
            self._open_mcp_panel()
            return True
        if command == "/skills":
            self._open_skills_panel()
            return True
        if command == "/skill":
            self._append_skill(rest.strip())
            return True
        if command == "/level":
            self._handle_level_command(rest.strip())
            return True
        if command == "/runs":
            self._open_runs_panel()
            return True
        if command == "/panel":
            self._flash("Panels now close with Esc")
            return True
        if command in {"/help", "?"}:
            self._open_help_panel()
            return True
        if command.startswith("/"):
            self._flash(f"Unknown command: {command}", severity="error")
            self._open_help_panel()
            return True
        return False

    async def action_quit(self) -> None:
        self.action_request_quit()

    def _open_help_panel(self) -> None:
        lines = ["[bold]commands[/bold] [dim](Tab/Enter selects suggestions, Esc closes)[/dim]"]
        for spec in COMMANDS:
            lines.append(
                f"[cyan]{escape(spec.usage):<18}[/cyan] [dim]{escape(spec.description)}[/dim]"
            )
        self._open_panel("\n".join(lines), "help", "Help")

    def _append_help(self) -> None:
        lines = ["[bold]commands[/bold] [dim](Ctrl+S opens status, Esc closes panels)[/dim]"]
        for spec in COMMANDS:
            lines.append(
                f"[cyan]{escape(spec.usage):<18}[/cyan] [dim]{escape(spec.description)}[/dim]"
            )
        self._append_cell("\n".join(lines), "event")

    def _append_user(self, text: str) -> None:
        self._append_cell(f"[bold #7dd3fc]you[/bold #7dd3fc]\n{escape(text)}", "user")

    async def _append_assistant_delta(self, text: str) -> None:
        self._assistant_buffer += text
        if self._assistant_cell is None:
            self._assistant_cell = TranscriptCell(classes="assistant")
            self.query_one("#transcript", VerticalScroll).mount(self._assistant_cell)
        self._assistant_cell.update(Markdown(self._assistant_buffer))
        self._scroll_end()

    def _append_tool_output(self, item: dict[str, Any]) -> None:
        payload = parse_tool_payload(item.get("output", {}))
        if payload is None:
            self._append_cell("[dim]python completed[/dim]", "event")
            return

        self._last_tool_payload = payload
        markup = tool_result_markup(payload)
        cell = self._tool_cells.pop(str(item.get("call", {}).get("call_id") or ""), None)
        if cell is None:
            self._append_cell(markup, "event")
        else:
            cell.update(markup)
            self._scroll_end()
        self._refresh_status("Working")

    def _append_tool_started(self, item: dict[str, Any]) -> None:
        call = item.get("call") or {}
        call_id = str(call.get("call_id") or "")
        name = str(call.get("name") or "python")
        detail = self._tool_call_preview(call)
        cell = self._append_cell(
            f"[cyan]{escape(name)}[/cyan] [dim]running[/dim]{detail}",
            "event",
        )
        if call_id:
            self._tool_cells[call_id] = cell
        self._refresh_status("Running python")

    def _tool_call_preview(self, call: dict[str, Any]) -> str:
        raw_args = call.get("arguments") or ""
        try:
            import json

            args = json.loads(raw_args)
        except Exception:
            return ""
        code = str(args.get("code") or "").strip()
        if not code:
            return ""
        first = next((line.strip() for line in code.splitlines() if line.strip()), "")
        if len(first) > 72:
            first = first[:69].rstrip() + "..."
        return f"\n[dim]{escape(first)}[/dim]"

    def _append_cell(self, content: object, classes: str) -> TranscriptCell:
        self._mark_transcript_content()
        cell = TranscriptCell(content, classes=classes, markup=True)
        self.query_one("#transcript", VerticalScroll).mount(cell)
        self._scroll_end()
        return cell

    def _mark_transcript_content(self) -> None:
        if self._transcript_has_content:
            return
        self._transcript_has_content = True
        self.query_one("#empty-state", EmptyState).add_class("hidden")

    def _reset_transcript(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.query("*").remove()
        self.call_after_refresh(transcript.mount, EmptyState(id="empty-state"))
        self._transcript_has_content = False
        self.call_after_refresh(lambda: self.query_one("#empty-state", EmptyState).tick())

    def _open_threads_panel(self) -> None:
        threads = self.engine.thread_store.list_threads()
        if not threads:
            self._open_fullscreen_panel("Threads", "[dim]No saved threads[/dim]")
            return
        items = []
        for thread in threads:
            thread_id = str(thread.get("thread_id") or "")
            title = str(thread.get("title") or "New thread")
            updated = str(thread.get("updated_at") or "")
            last_text = str(thread.get("last_text") or "").replace("\n", " ")
            if len(last_text) > 120:
                last_text = last_text[:117].rstrip() + "..."
            marker = "current " if thread_id == self.thread_id else ""
            items.append(
                PickerItem(
                    id=thread_id,
                    title=f"{marker}{title}",
                    description=last_text or "No messages yet",
                    meta=f"{short_thread(thread_id)} · {thread.get('turn_count', 0)} turns · {updated}",
                )
            )
        self._open_picker("Threads", items, self._resume_thread, subtitle="Search and Enter to resume")

    def _open_status_panel(self) -> None:
        self._open_panel(self._status_panel_markup(), "status", "Status")

    def _status_panel_markup(self) -> str:
        self.engine.refresh_config()
        try:
            model = self.engine.config.model_for_level(self.level)
            provider = self.engine.config.provider_for_model(model)
            stats = self.engine.context_stats(self.thread_id, self.level)
            model_line = f"{escape(model.name)} -> {escape(model.model)}"
            provider_line = f"{escape(provider.name)} / {escape(model.api)}"
            context_line = (
                f"{stats.percent}% "
                f"({format_tokens(stats.used_tokens)} / {format_tokens(stats.context_window_tokens)}, "
                f"{escape(stats.source)})"
            )
            compress_line = (
                f"trigger {format_tokens(stats.threshold_tokens)} · "
                f"target {format_tokens(stats.target_tokens)} · "
                f"headroom {format_tokens(stats.headroom_tokens)}"
            )
        except ConfigError as exc:
            model_line = "[red]not configured[/red]"
            provider_line = escape(str(exc))
            context_line = "-"
            compress_line = "-"
        level_name = self.level or self.engine.config.runtime.default_level
        lines = [
            f"- state: [cyan]{escape(self._last_status)}[/cyan]",
            f"- level: [cyan]{escape(level_name)}[/cyan]",
            f"- model: {model_line}",
            f"- provider/api: {provider_line}",
            f"- context: {context_line}",
            f"- compaction: {compress_line}",
            f"- thread: {escape(short_thread(self.thread_id))}",
            f"- queued: {len(self._queue)}",
            f"- user state: {escape(str(uv_agent_home()))}",
            f"- project state: {escape(str(project_state_dir(self.project_root)))}",
        ]
        return "\n".join(lines)

    def _open_context_panel(self) -> None:
        self._open_panel(self._context_panel_markup(), "context", "Context")

    def _context_panel_markup(self) -> str:
        self.engine.refresh_config()
        rules = self.engine.project_rule_context()
        try:
            stats = self.engine.context_stats(self.thread_id, self.level)
            lines = [
                f"[bold]token budget[/bold] [dim]{escape(stats.source)}[/dim]",
                f"- used: [cyan]{format_tokens(stats.used_tokens)}[/cyan] / {format_tokens(stats.context_window_tokens)} ({stats.percent}%)",
                f"- headroom: {format_tokens(stats.headroom_tokens)}",
                f"- compression trigger: {format_tokens(stats.threshold_tokens)}",
                f"- compression target: {format_tokens(stats.target_tokens)}",
            ]
        except ConfigError as exc:
            lines = ["[bold]token budget[/bold]", f"[red]{escape(str(exc))}[/red]"]
        lines.extend(
            [
                "",
                "[bold]workspace rules[/bold]",
                f"- loaded files: {len(rules.rules)}",
                f"- capped: {'yes' if rules.truncated else 'no'}",
            ]
        )
        if rules.omitted_files:
            lines.append(f"- omitted files: {rules.omitted_files}")
        for rule in rules.rules:
            suffix = " [yellow]truncated[/yellow]" if rule.truncated else ""
            lines.append(f"  [cyan]{escape(rule.scope)}[/cyan] {escape(str(rule.path))}{suffix}")
        if not rules.rules:
            lines.append("[dim]no AGENTS.md files discovered[/dim]")
        return "\n".join(lines)

    def _open_rules_panel(self) -> None:
        rules = self.engine.project_rule_context()
        if not rules.rules:
            self._open_panel("[dim]no AGENTS.md files discovered[/dim]", "rules", "Rules")
            return
        lines: list[str] = []
        for rule in rules.rules:
            title = f"[cyan]{escape(rule.scope)}[/cyan] {escape(str(rule.path))}"
            if rule.truncated:
                title += " [yellow]truncated[/yellow]"
            preview = rule.text.strip()
            if len(preview) > 2400:
                preview = preview[:2400].rstrip() + "\n..."
            lines.append(title + "\n" + escape(preview))
        if rules.truncated and rules.omitted_files:
            lines.append(f"[yellow]{rules.omitted_files} file(s) omitted by context cap[/yellow]")
        self._open_panel("\n\n".join(lines), "rules", "Rules")

    def _open_config_panel(self) -> None:
        self.engine.refresh_config()
        sources = config_sources(self.project_root)
        lines = ["[bold]sources[/bold]"]
        for source in sources:
            exists = "yes" if source["exists"] else "no"
            lines.append(
                f"- {escape(source['scope'])}: {escape(source['path'])} [dim]exists={exists}[/dim]"
            )
        level_name = self.level or self.engine.config.runtime.default_level
        lines.append("\n[bold]active[/bold]")
        try:
            model = self.engine.config.model_for_level(self.level)
            provider = self.engine.config.provider_for_model(model)
            lines.extend(
                [
                    f"- level: [cyan]{escape(level_name)}[/cyan]",
                    f"- model: {escape(model.name)} -> {escape(model.model)}",
                    f"- provider: {escape(provider.name)}",
                    f"- api: {escape(model.api)}",
                    f"- context window: {format_tokens(model.context_window_tokens)}",
                ]
            )
        except ConfigError as exc:
            lines.append(f"[red]{escape(str(exc))}[/red]")
        redacted = redact_config(load_raw_config(self.project_root))
        preview = json.dumps(redacted, ensure_ascii=False, indent=2)
        if len(preview) > 2200:
            preview = preview[:2200].rstrip() + "\n..."
        lines.extend(["\n[bold]redacted merged config[/bold]", escape(preview)])
        self._open_panel("\n".join(lines), "config", "Config")

    def _open_models_panel(self) -> None:
        self.engine.refresh_config()
        lines = ["[bold]levels[/bold] [dim](/level name selects)[/dim]"]
        for name, level in self.engine.config.levels.items():
            marker = "*" if name == (self.level or self.engine.config.runtime.default_level) else "-"
            lines.append(f"{marker} [cyan]{escape(name)}[/cyan] -> {escape(level.model)}")
        lines.append("\n[bold]models[/bold]")
        for name, model in self.engine.config.models.items():
            lines.append(
                f"- {escape(name)}: {escape(model.model)} [dim]{escape(model.api)} · {format_tokens(model.context_window_tokens)}[/dim]"
            )
        self._open_panel("\n".join(lines), "models", "Models")

    def _handle_level_command(self, name: str) -> None:
        if not name:
            self._open_models_panel()
            return
        if name not in self.engine.config.levels:
            self._append_cell(f"[red]unknown level[/red] {escape(name)}", "error")
            return
        self.level = name
        self._append_cell(f"[dim]level[/dim] [cyan]{escape(name)}[/cyan]", "event")
        self._refresh_status()

    def _open_runs_panel(self) -> None:
        if not self._last_tool_payload:
            self._open_panel("[dim]no Python runs in this TUI session[/dim]", "runs", "Runs")
            return
        self._open_panel(
            tool_result_markup(self._last_tool_payload),
            "runs",
            "Last Run",
        )

    def _open_mcp_panel(self) -> None:
        self.engine.refresh_config()
        servers = discover_mcp_servers(self.project_root)
        if not servers:
            self._open_panel("[dim]no .agents/mcp.json servers declared[/dim]", "mcp", "MCP")
            return
        lines = ["[dim]declarations only[/dim]"]
        for server in servers:
            command = f" [dim]{escape(server.command)}[/dim]" if server.command else ""
            lines.append(
                f"- [cyan]{escape(server.name)}[/cyan] ({escape(server.scope)}) {escape(server.description)}{command}"
            )
        self._open_panel("\n".join(lines), "mcp", "MCP")

    def _open_skills_panel(self) -> None:
        self.engine.refresh_config()
        skills = discover_skills(self.project_root)
        if not skills:
            self._open_panel("[dim]no .agents/skills entries discovered[/dim]", "skills", "Skills")
            return
        lines = ["[dim]/skill name previews a skill file[/dim]"]
        for skill in skills:
            lines.append(
                f"- [cyan]{escape(skill.name)}[/cyan] ({escape(skill.scope)}) {escape(skill.description)}"
            )
        self._open_panel("\n".join(lines), "skills", "Skills")

    def _append_skill(self, name: str) -> None:
        if not name:
            self._open_skills_panel()
            return
        skills = {skill.name: skill for skill in discover_skills(self.project_root)}
        skill = skills.get(name)
        if skill is None:
            self._append_cell(f"[red]unknown skill[/red] {escape(name)}", "error")
            return
        text = skill.path.read_text(encoding="utf-8")
        preview = "\n".join(text.splitlines()[:18])
        if len(text.splitlines()) > 18:
            preview += "\n..."
        self._append_cell(
            f"[bold]skill {escape(skill.name)}[/bold] [dim]{escape(str(skill.path))}[/dim]\n"
            + escape(preview),
            "event",
        )

    def _open_command_palette(self) -> None:
        items = [
            PickerItem(
                id=spec.name,
                title=spec.usage,
                description=spec.description,
            )
            for spec in COMMANDS
        ]
        self._open_picker("Commands", items, self._choose_command, subtitle="Type to filter commands")

    def _choose_command(self, command: str) -> None:
        spec = next((item for item in COMMANDS if item.name == command), None)
        if spec is None:
            return
        if "[" in spec.usage:
            self._apply_command_completion(command)
            return
        self._handle_command(command)

    def _resume_thread(self, thread_id: str) -> None:
        if not thread_id:
            return
        self.thread_id = thread_id
        self._assistant_buffer = ""
        self._assistant_cell = None
        self._tool_cells.clear()
        self._reset_transcript()
        self._render_thread_history(thread_id)
        self._refresh_status("Resumed")

    def _render_thread_history(self, thread_id: str) -> None:
        for event in self.engine.thread_store.read(thread_id):
            event_type = event.get("type")
            if event_type == "item.user":
                self._append_user_from_history(event.get("item") or {})
            elif event_type == "item.model_response":
                text = self._model_response_text(event.get("output") or [])
                if text:
                    self._append_cell(Markdown(text), "assistant")
            elif event_type == "item.tool_call":
                item = event.get("item") or {}
                name = str(item.get("name") or "python")
                self._append_cell(f"[cyan]{escape(name)}[/cyan] [dim]called[/dim]", "event")
            elif event_type == "item.runner_result":
                result = event.get("result") or {}
                self._last_tool_payload = result
                self._append_cell(tool_result_markup(result), "event")
            elif event_type == "item.image_attachment":
                attachment = event.get("attachment") or {}
                self._append_cell(
                    f"[dim]image attached[/dim] [cyan]{escape(str(attachment.get('stored_path') or ''))}[/cyan]",
                    "event",
                )

    def _append_user_from_history(self, item: dict[str, Any]) -> None:
        parts = []
        for content in item.get("content") or []:
            if content.get("type") in {"input_text", "text"}:
                parts.append(str(content.get("text") or ""))
        if parts:
            self._append_user("\n".join(parts))

    def _model_response_text(self, output: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in output:
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"}:
                    parts.append(str(content.get("text") or ""))
        return "".join(parts)

    def _open_fullscreen_panel(self, title: str, markup: str, *, subtitle: str = "") -> None:
        self.push_screen(FullscreenPanel(title=title, body=markup, subtitle=subtitle))

    def _open_picker(
        self,
        title: str,
        items: list[PickerItem],
        callback: Callable[[str], None],
        *,
        subtitle: str = "",
    ) -> None:
        def handle(result: str | None) -> None:
            if result:
                callback(result)
            self.query_one("#composer", TextArea).focus()

        self.push_screen(
            FullscreenPanel(title=title, items=items, subtitle=subtitle),
            handle,
        )

    def _open_panel(self, markup: str, name: str | None = None, title: str | None = None) -> None:
        panel_title = title or (name.title() if name else "Panel")
        self._open_fullscreen_panel(panel_title, markup, subtitle="Esc closes")
        self._refresh_status()

    def _update_command_suggestions(self, text: str) -> None:
        stripped = text.strip()
        if not stripped.startswith("/") or "\n" in stripped or " " in stripped:
            self._hide_command_suggestions()
            return
        token = stripped.split(" ", 1)[0]
        exact = next((spec for spec in COMMANDS if spec.name == token), None)
        if exact is not None and "[" not in exact.usage:
            self._hide_command_suggestions()
            return
        matches = [spec for spec in COMMANDS if spec.name.startswith(token)]
        if not matches and len(token) > 2:
            query = token.removeprefix("/").lower()
            matches = [spec for spec in COMMANDS if query in spec.description.lower()]
        if not matches:
            matches = [CommandSpec(token, token, "unknown command")]
        self.query_one("#command-suggestions", CommandSuggestions).set_matches(matches)
        self.query_one("#composer-shell", Vertical).add_class("command-mode")

    def _command_completion(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped.startswith("/") or "\n" in stripped or " " in stripped:
            return None
        matches = [spec for spec in COMMANDS if spec.name.startswith(stripped)]
        if not matches:
            return None
        command = matches[0].name
        needs_arg = "[" in matches[0].usage
        return command + (" " if needs_arg else "")

    def _apply_command_completion(self, command: str) -> None:
        spec = next((item for item in COMMANDS if item.name == command), None)
        needs_arg = spec is not None and "[" in spec.usage
        replacement = command + (" " if needs_arg else "")
        composer = self.query_one("#composer", TextArea)
        composer.load_text(replacement)
        self._resize_composer(replacement)
        self._hide_command_suggestions()
        composer.focus()

    def _hide_command_suggestions(self) -> None:
        self.query_one("#command-suggestions", CommandSuggestions).clear()
        self.query_one("#composer-shell", Vertical).remove_class("command-mode")

    def _command_panel_visible(self) -> bool:
        return not self.query_one("#command-suggestions", CommandSuggestions).has_class("hidden")

    def _resize_composer(self, text: str) -> None:
        line_count = max(1, text.count("\n") + 1)
        height = min(8, max(3, line_count + 1))
        self.query_one("#composer", TextArea).styles.height = height

    def _flash(self, message: str, *, severity: str = "information") -> None:
        self.notify(message, severity=severity, timeout=2.0)
        self._last_status = message
        self._refresh_status()

    def _refresh_status(self, state: str | None = None) -> None:
        if state is not None:
            self._last_status = state
        self.engine.refresh_config()
        level_name = self.level or self.engine.config.runtime.default_level
        try:
            stats = self.engine.context_stats(self.thread_id, self.level)
            context = f"{stats.percent}% {format_tokens(stats.used_tokens)}/{format_tokens(stats.context_window_tokens)}"
        except ConfigError:
            context = "config?"
        state_text = self._last_status
        if self.busy and state_text == "Idle":
            state_text = "Working"
        queued = f" · q {len(self._queue)}" if self._queue else ""
        spinner = ""
        if self.busy:
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            spinner = frames[self._spinner_index % len(frames)] + " "
            self._spinner_index += 1

        left = f"[cyan]{spinner}{escape(state_text)}[/cyan]"
        compact_context = context.split(" ", 1)[0]
        right = f"[dim]{escape(level_name)}\nctx {escape(compact_context)}{queued}[/dim]"
        self.query_one("#composer-left", Static).update(left)
        self.query_one("#composer-right", Static).update(right)

    def _scroll_end(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        self.call_after_refresh(transcript.scroll_end, animate=False)
