from __future__ import annotations

from pathlib import Path

from rich.markdown import Markdown
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Input, Static

from uv_agent.app_factory import create_engine
from uv_agent.config import config_sources
from uv_agent.mcp_config import discover_mcp_servers
from uv_agent.skills import discover_skills
from uv_agent.tui.formatting import parse_tool_payload, short_thread, tool_result_markup


class TranscriptCell(Static):
    """Small transcript block used by the Textual chat timeline."""

    DEFAULT_CSS = """
    TranscriptCell {
        width: 100%;
        margin: 0 0 1 0;
    }

    TranscriptCell.user {
        color: $text;
    }

    TranscriptCell.assistant {
        color: $text;
    }

    TranscriptCell.event {
        color: $text-muted;
    }

    TranscriptCell.error {
        color: $error;
    }
    """


class UvAgentApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #transcript {
        height: 1fr;
        min-height: 6;
        padding: 1 2 0 2;
        background: $surface;
    }

    #bottom-pane {
        height: auto;
        max-height: 16;
        padding: 0 2 1 2;
        background: $surface;
    }

    #drawer {
        height: auto;
        max-height: 10;
        border-top: tall $secondary;
        padding: 1 0 0 0;
        color: $text;
    }

    #drawer.hidden {
        display: none;
    }

    #run-status {
        height: 1;
        color: $text-muted;
    }

    #composer-row {
        height: 1;
    }

    #prompt-marker {
        width: 2;
        color: $primary;
        text-style: bold;
    }

    #input {
        height: 1;
        border: none;
        padding: 0;
        background: $surface;
    }

    #input:focus {
        border: none;
    }

    #hint-line {
        height: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("escape", "clear_input", "Clear"),
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

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="transcript")
        with Vertical(id="bottom-pane"):
            yield Static(id="drawer", classes="hidden")
            yield Static(id="run-status")
            with Horizontal(id="composer-row"):
                yield Static("›", id="prompt-marker")
                yield Input(
                    placeholder="Ask uv-agent to do anything",
                    id="input",
                    compact=True,
                )
            yield Static(id="hint-line")

    def on_mount(self) -> None:
        self._append_cell(
            "[bold magenta]uv-agent[/bold magenta] [dim]ready · python runner only[/dim]",
            "event",
        )
        self._refresh_status("Idle")
        self.set_interval(0.12, self._tick, name="spinner")
        self.query_one("#input", Input).focus()

    def on_resize(self) -> None:
        self._refresh_status()

    def _tick(self) -> None:
        if self.busy:
            self._refresh_status()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        event.input.value = ""
        if not prompt:
            return
        if self._handle_command(prompt):
            return
        if self.busy:
            self._queue.append(prompt)
            self._append_cell(f"[dim]queued[/dim] {escape(prompt)}", "event")
            self._refresh_status()
            return
        self._start_turn(prompt)

    def _start_turn(self, prompt: str) -> None:
        self.busy = True
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
                    self._refresh_status("Reading model response")
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
            if self._last_status != "Error":
                self._refresh_status("Idle")
            self.query_one("#input", Input).focus()
            if self._queue:
                next_prompt = self._queue.pop(0)
                self._start_turn(next_prompt)

    def action_clear_input(self) -> None:
        if not self.busy:
            self.query_one("#input", Input).value = ""

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
                "[bold magenta]uv-agent[/bold magenta] [dim]ready · python runner only[/dim]",
                "event",
            )
            self._refresh_status("Idle")
            return True
        if command == "/new":
            title = rest.strip() or "New thread"
            self.thread_id = self.engine.thread_store.create_thread(title)
            self._append_cell(
                f"[dim]new thread[/dim] {escape(short_thread(self.thread_id))}",
                "event",
            )
            self._refresh_status("Idle")
            return True
        if command == "/threads":
            self._open_threads_panel()
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
            self._append_help()
            return True
        return False

    def _append_help(self) -> None:
        self._append_cell(
            "[bold]commands[/bold]\n"
            "/new \\[title]\n"
            "/threads\n"
            "/config\n"
            "/models\n"
            "/mcp\n"
            "/skills\n"
            "/skill \\[name]\n"
            "/level \\[name]\n"
            "/runs\n"
            "/panel\n"
            "/clear\n"
            "/help",
            "event",
        )

    def _append_user(self, text: str) -> None:
        self._append_cell(f"[bold cyan]>[/bold cyan] {escape(text)}", "user")

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
        cell = self._append_cell(
            f"[cyan]{escape(name)}[/cyan] [dim]running[/dim]",
            "event",
        )
        if call_id:
            self._tool_cells[call_id] = cell
        self._refresh_status("Running python")

    def _append_cell(self, content: str, classes: str) -> TranscriptCell:
        cell = TranscriptCell(content, classes=classes, markup=True)
        self.query_one("#transcript", VerticalScroll).mount(cell)
        self._scroll_end()
        return cell

    def _append_threads(self) -> None:
        threads = self.engine.thread_store.list_threads()[-8:]
        if not threads:
            self._append_cell("[dim]no saved threads[/dim]", "event")
            return
        lines = ["[bold]threads[/bold]"]
        for thread in threads:
            marker = "*" if thread.get("thread_id") == self.thread_id else "-"
            title = str(thread.get("title") or "New thread")
            thread_id = short_thread(str(thread.get("thread_id") or ""))
            lines.append(f"{marker} {escape(thread_id)} [dim]{escape(title)}[/dim]")
        self._append_cell("\n".join(lines), "event")

    def _open_threads_panel(self) -> None:
        threads = self.engine.thread_store.list_threads()[-10:]
        if not threads:
            body = "[bold]threads[/bold]\n[dim]no saved threads[/dim]"
        else:
            lines = ["[bold]threads[/bold] [dim](/panel closes)[/dim]"]
            for thread in threads:
                marker = "*" if thread.get("thread_id") == self.thread_id else "-"
                title = str(thread.get("title") or "New thread")
                thread_id = short_thread(str(thread.get("thread_id") or ""))
                lines.append(f"{marker} [cyan]{escape(thread_id)}[/cyan] [dim]{escape(title)}[/dim]")
            body = "\n".join(lines)
        self._open_panel(body)

    def _open_config_panel(self) -> None:
        sources = config_sources(self.project_root)
        lines = ["[bold]config[/bold] [dim](/panel closes)[/dim]", "[dim]sources[/dim]"]
        for source in sources:
            exists = "yes" if source["exists"] else "no"
            lines.append(
                f"- {escape(source['scope'])}: {escape(source['path'])} [dim]exists={exists}[/dim]"
            )
        model = self.engine.config.model_for_level(self.level)
        provider = self.engine.config.provider_for_model(model)
        level_name = self.level or self.engine.config.runtime.default_level
        lines.extend(
            [
                "[dim]active[/dim]",
                f"- level: [cyan]{escape(level_name)}[/cyan]",
                f"- model: {escape(model.name)} -> {escape(model.model)}",
                f"- provider: {escape(provider.name)}",
                f"- api: {escape(model.api)}",
                f"- context: {model.context_window_tokens}",
            ]
        )
        self._open_panel("\n".join(lines))

    def _open_models_panel(self) -> None:
        lines = ["[bold]models[/bold] [dim](/level name selects, /panel closes)[/dim]"]
        lines.append("[dim]levels[/dim]")
        for name, level in self.engine.config.levels.items():
            marker = "*" if name == (self.level or self.engine.config.runtime.default_level) else "-"
            lines.append(f"{marker} [cyan]{escape(name)}[/cyan] -> {escape(level.model)}")
        lines.append("[dim]models[/dim]")
        for name, model in self.engine.config.models.items():
            lines.append(
                f"- {escape(name)}: {escape(model.model)} [dim]{escape(model.api)}[/dim]"
            )
        self._open_panel("\n".join(lines))

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
            self._open_panel("[bold]runs[/bold]\n[dim]no Python runs in this TUI session[/dim]")
            return
        self._open_panel(
            "[bold]last run[/bold] [dim](/panel closes)[/dim]\n"
            + tool_result_markup(self._last_tool_payload)
        )

    def _open_mcp_panel(self) -> None:
        servers = discover_mcp_servers(self.project_root)
        if not servers:
            self._open_panel("[bold]mcp[/bold]\n[dim]no .agents/mcp.json servers declared[/dim]")
            return
        lines = ["[bold]mcp[/bold] [dim](declarations only, /panel closes)[/dim]"]
        for server in servers:
            command = f" [dim]{escape(server.command)}[/dim]" if server.command else ""
            lines.append(
                f"- [cyan]{escape(server.name)}[/cyan] ({escape(server.scope)}) {escape(server.description)}{command}"
            )
        self._open_panel("\n".join(lines))

    def _open_skills_panel(self) -> None:
        skills = discover_skills(self.project_root)
        if not skills:
            self._open_panel("[bold]skills[/bold]\n[dim]no .agents/skills entries discovered[/dim]")
            return
        lines = ["[bold]skills[/bold] [dim](/skill name previews, /panel closes)[/dim]"]
        for skill in skills:
            lines.append(
                f"- [cyan]{escape(skill.name)}[/cyan] ({escape(skill.scope)}) {escape(skill.description)}"
            )
        self._open_panel("\n".join(lines))

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

    def _open_panel(self, markup: str) -> None:
        drawer = self.query_one("#drawer", Static)
        drawer.update(markup)
        drawer.remove_class("hidden")
        self._refresh_status()

    def _close_panel(self) -> None:
        drawer = self.query_one("#drawer", Static)
        drawer.update("")
        drawer.add_class("hidden")
        self._refresh_status()

    def _refresh_status(self, state: str | None = None) -> None:
        if state is not None:
            self._last_status = state
        level_name = self.level or self.engine.config.runtime.default_level
        model = self.engine.config.model_for_level(self.level)
        model_name = model.model
        api = model.api.replace("_", "-")
        thread = short_thread(self.thread_id)
        context = self.engine.context_percent(self.thread_id, self.level)
        state_text = self._last_status
        if self.busy and state_text == "Idle":
            state_text = "Working"
        queued = f" · queued {len(self._queue)}" if self._queue else ""
        spinner = ""
        if self.busy:
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            spinner = frames[self._spinner_index % len(frames)] + " "
            self._spinner_index += 1
        if self.size.width < 72:
            status = (
                f"[cyan]{spinner}{escape(state_text)}[/cyan] "
                f"[dim]· {escape(level_name)} · {escape(model_name)} · {context}% · {escape(thread)}{queued}[/dim]"
            )
            hint = "[dim]/help · /config · /models · /skills · /mcp · Ctrl+C quit[/dim]"
        else:
            status = (
                f"[cyan]{spinner}{escape(state_text)}[/cyan] "
                f"[dim]· {escape(level_name)} · {escape(model_name)} · {api} · context {context}% · thread {escape(thread)}{queued}[/dim]"
            )
            hint = "[dim]/help[/dim] [dim]·[/dim] [dim]/config[/dim] [dim]·[/dim] [dim]/models[/dim] [dim]·[/dim] [dim]/skills[/dim] [dim]·[/dim] [dim]/mcp[/dim] [dim]·[/dim] [dim]/threads[/dim] [dim]·[/dim] [dim]/runs[/dim] [dim]·[/dim] [dim]Ctrl+C quit[/dim]"
        self.query_one("#run-status", Static).update(status)
        self.query_one("#hint-line", Static).update(hint)

    def _scroll_end(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        self.call_after_refresh(transcript.scroll_end, animate=False)
