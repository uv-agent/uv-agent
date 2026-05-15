from __future__ import annotations

import ctypes
import json
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from rich.markdown import Markdown
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Static, TextArea

from uv_agent.app_factory import create_engine
from uv_agent.config import ConfigError, config_sources, load_raw_config, redact_config
from uv_agent.mcp_config import discover_mcp_servers
from uv_agent.paths import project_state_dir, uv_agent_home
from uv_agent.skills import discover_skills
from uv_agent.tui.formatting import format_tokens, parse_tool_payload, short_thread, tool_result_markup


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
        for index, spec in enumerate(self.matches[:8]):
            active = index == self.selected_index
            prefix = "[reverse]›[/reverse]" if active else " "
            command_style = "bold cyan" if active else "cyan"
            desc_style = "white" if active else "dim"
            lines.append(
                f"{prefix} [{command_style}]{escape(spec.usage):<18}[/{command_style}] "
                f"[{desc_style}]{escape(spec.description)}[/{desc_style}]"
            )
        if len(self.matches) > 8:
            lines.append(f"[dim]... {len(self.matches) - 8} more[/dim]")
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

    #side-drawer {
        width: 38;
        min-width: 28;
        max-width: 46;
        height: 100%;
        border-left: tall #263241;
        background: #0d141c;
        padding: 1 1;
        color: #d8dee9;
    }

    #side-drawer.hidden {
        display: none;
    }

    #drawer-title {
        height: 1;
        color: #7dd3fc;
        text-style: bold;
    }

    #drawer-body {
        height: 1fr;
        overflow-y: auto;
        padding: 1 0 0 0;
    }

    #bottom-pane {
        height: auto;
        max-height: 18;
        padding: 0 2 1 1;
        background: #0b0f14;
    }

    #drawer {
        height: auto;
        max-height: 13;
        border: tall #263241;
        padding: 1 1;
        margin: 0 0 1 0;
        background: #0d141c;
        color: #d8dee9;
        overflow-y: auto;
    }

    #drawer.hidden {
        display: none;
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

    #run-status {
        height: 1;
        color: #9fb0c3;
    }

    #composer-row {
        height: auto;
    }

    #prompt-marker {
        width: 3;
        color: #7dd3fc;
        text-style: bold;
    }

    #composer {
        height: 3;
        min-height: 3;
        max-height: 8;
        border: none;
        padding: 0;
        background: #0f1721;
        color: #edf2f7;
    }

    #composer:focus {
        border: none;
    }

    #hint-line {
        height: 1;
        color: #7b8796;
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
        self._panel_name: str | None = None
        self._panel_markup = ""
        self._panel_title = ""
        self._previous_sigint_handler: Any = None
        self._windows_ctrl_handler: Any = None

    def compose(self) -> ComposeResult:
        with Vertical(id="main-column"):
            yield VerticalScroll(id="transcript")
            with Vertical(id="bottom-pane"):
                yield Static(id="drawer", classes="hidden")
                yield CommandSuggestions(id="command-suggestions")
                with Vertical(id="composer-shell"):
                    yield Static(id="run-status")
                    with Horizontal(id="composer-row"):
                        yield Static("›", id="prompt-marker")
                        yield TextArea(
                            "",
                            placeholder="Ask, edit, or type / for commands",
                            id="composer",
                            compact=True,
                            soft_wrap=True,
                            show_line_numbers=False,
                        )
                    yield Static(id="hint-line")
        with Vertical(id="side-drawer", classes="hidden"):
            yield Static(id="drawer-title")
            yield Static(id="drawer-body")

    def on_mount(self) -> None:
        self._append_cell(
            "[bold #7dd3fc]uv-agent[/bold #7dd3fc] [dim]ready · Python runner only[/dim]",
            "event",
        )
        self._refresh_status("Idle")
        self.set_interval(0.12, self._tick, name="spinner")
        self.query_one("#composer", TextArea).focus()
        self._install_sigint_guard()

    def on_unmount(self) -> None:
        if self._previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._previous_sigint_handler)
        self._uninstall_windows_ctrl_guard()

    def on_resize(self) -> None:
        if self._panel_name:
            self._render_panel()
        self._refresh_status()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "composer":
            return
        self._resize_composer(event.text_area.text)
        self._update_command_suggestions(event.text_area.text)
        self._refresh_status()

    def _tick(self) -> None:
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
        if self._panel_name:
            self._close_panel()

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
        if self._panel_name == "status":
            self._close_panel()
            return
        self._open_status_panel()

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
            self.query_one("#transcript", VerticalScroll).remove_children()
            self._append_cell(
                "[bold #7dd3fc]uv-agent[/bold #7dd3fc] [dim]ready · Python runner only[/dim]",
                "event",
            )
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
            self._close_panel()
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

    def _append_cell(self, content: str, classes: str) -> TranscriptCell:
        cell = TranscriptCell(content, classes=classes, markup=True)
        self.query_one("#transcript", VerticalScroll).mount(cell)
        self._scroll_end()
        return cell

    def _open_threads_panel(self) -> None:
        threads = self.engine.thread_store.list_threads()[-10:]
        if not threads:
            body = "[dim]no saved threads[/dim]"
        else:
            lines = []
            for thread in threads:
                marker = "*" if thread.get("thread_id") == self.thread_id else "-"
                title = str(thread.get("title") or "New thread")
                thread_id = short_thread(str(thread.get("thread_id") or ""))
                lines.append(f"{marker} [cyan]{escape(thread_id)}[/cyan] [dim]{escape(title)}[/dim]")
            body = "\n".join(lines)
        self._open_panel(body, "threads", "Threads")

    def _open_status_panel(self) -> None:
        self._open_panel(self._status_panel_markup(), "status", "Status")

    def _status_panel_markup(self) -> str:
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

    def _open_panel(self, markup: str, name: str | None = None, title: str | None = None) -> None:
        self._panel_name = name
        self._panel_markup = markup
        self._panel_title = title or (name.title() if name else "Panel")
        self._render_panel()
        self._refresh_status()

    def _close_panel(self) -> None:
        drawer = self.query_one("#drawer", Static)
        drawer.update("")
        drawer.add_class("hidden")
        side = self.query_one("#side-drawer", Vertical)
        side.add_class("hidden")
        self.query_one("#drawer-title", Static).update("")
        self.query_one("#drawer-body", Static).update("")
        self._panel_name = None
        self._panel_markup = ""
        self._panel_title = ""
        self._refresh_status()

    def _render_panel(self) -> None:
        if not self._panel_name:
            return
        title = escape(self._panel_title)
        if self._use_side_drawer():
            self.query_one("#drawer", Static).add_class("hidden")
            self.query_one("#drawer", Static).update("")
            self.query_one("#drawer-title", Static).update(
                f"[bold]{title}[/bold] [dim]Esc closes[/dim]"
            )
            self.query_one("#drawer-body", Static).update(self._panel_markup)
            self.query_one("#side-drawer", Vertical).remove_class("hidden")
            return
        self.query_one("#side-drawer", Vertical).add_class("hidden")
        self.query_one("#drawer-title", Static).update("")
        self.query_one("#drawer-body", Static).update("")
        drawer = self.query_one("#drawer", Static)
        drawer.update(f"[bold]{title}[/bold] [dim]Esc closes[/dim]\n{self._panel_markup}")
        drawer.remove_class("hidden")

    def _use_side_drawer(self) -> bool:
        return self.size.width >= 110 and self.size.height >= 20

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
        matches = matches[:8]
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
        color = "red" if severity == "error" else "yellow" if severity == "warning" else "cyan"
        self._last_status = message
        self.query_one("#run-status", Static).update(f"[{color}]{escape(message)}[/{color}]")

    def _refresh_status(self, state: str | None = None) -> None:
        if state is not None:
            self._last_status = state
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

        status = (
            f"[cyan]{spinner}{escape(state_text)}[/cyan] "
            f"[dim]· {escape(level_name)} · ctx {escape(context)}{queued}[/dim]"
        )
        if self.size.width < 58:
            hint = "[dim]Ctrl+Enter · /[/dim]"
        else:
            hint = "[dim]Enter newline · Ctrl+Enter send · / commands · Ctrl+S details · Ctrl+Q quit[/dim]"
        self.query_one("#run-status", Static).update(status)
        self.query_one("#hint-line", Static).update(hint)
        if self._panel_name == "status":
            self._panel_markup = self._status_panel_markup()
            self._render_panel()
        elif self._panel_name == "context":
            self._panel_markup = self._context_panel_markup()
            self._render_panel()

    def _scroll_end(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        self.call_after_refresh(transcript.scroll_end, animate=False)
