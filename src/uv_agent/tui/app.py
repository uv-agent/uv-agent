from __future__ import annotations

from pathlib import Path

from rich.markdown import Markdown
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Input, Static

from uv_agent.app_factory import create_engine
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
        max-height: 5;
        padding: 0 2 1 2;
        background: $surface;
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
        self._assistant_buffer = ""
        self._assistant_cell: TranscriptCell | None = None
        self._tool_cells: dict[str, TranscriptCell] = {}
        self._queue: list[str] = []
        self._last_status = "Idle"

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="transcript")
        with Vertical(id="bottom-pane"):
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
        self.query_one("#input", Input).focus()

    def on_resize(self) -> None:
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
            async for item in self.engine.run_turn(user_text=prompt, thread_id=self.thread_id):
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
            self._append_threads()
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

    def _refresh_status(self, state: str | None = None) -> None:
        if state is not None:
            self._last_status = state
        model = self.engine.config.model_for_level(None)
        model_name = model.model
        api = model.api.replace("_", "-")
        thread = short_thread(self.thread_id)
        context = self.engine.context_percent(self.thread_id)
        state_text = self._last_status
        if self.busy and state_text == "Idle":
            state_text = "Working"
        queued = f" · queued {len(self._queue)}" if self._queue else ""
        if self.size.width < 72:
            status = (
                f"[cyan]{escape(state_text)}[/cyan] "
                f"[dim]· {escape(model_name)} · {context}% · {escape(thread)}{queued}[/dim]"
            )
            hint = "[dim]/help · Ctrl+C quit[/dim]"
        else:
            status = (
                f"[cyan]{escape(state_text)}[/cyan] "
                f"[dim]· {escape(model_name)} · {api} · context {context}% · thread {escape(thread)}{queued}[/dim]"
            )
            hint = "[dim]/help[/dim] [dim]·[/dim] [dim]Ctrl+C quit[/dim] [dim]·[/dim] [dim]Esc clear[/dim]"
        self.query_one("#run-status", Static).update(status)
        self.query_one("#hint-line", Static).update(hint)

    def _scroll_end(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        self.call_after_refresh(transcript.scroll_end, animate=False)
