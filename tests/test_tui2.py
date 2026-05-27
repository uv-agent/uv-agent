from __future__ import annotations

import asyncio
import json
import io
from pathlib import Path
from types import SimpleNamespace

from uv_agent.tui2.ansi import strip_ansi, visible_len
from uv_agent.tui2.components import (
    render_cell,
    render_composer_with_cursor,
    render_live_with_cursor,
    render_markdown,
    render_status_lines,
    render_tool_cell,
)
import uv_agent.tui2.app as tui2_app

from uv_agent.tui2.app import TOP_LEVEL_COMMANDS, load_composer_history, save_composer_history
from uv_agent.tui2.events import CommandSuggestion, TranscriptCell, Tui2State
from uv_agent.tui2.renderer import Renderer
from uv_agent.tui2.terminal import PASTE_PREFIX, Terminal
from uv_agent.tui2.theme import DEFAULT_THEME, sgr


# ---------------------------------------------------------------------------
# Cell rendering: no per-cell horizontal rules
# ---------------------------------------------------------------------------


def test_user_message_has_no_leading_separator() -> None:
    lines = render_cell(TranscriptCell("user", text="hello"), 40)

    plain = [strip_ansi(line) for line in lines]
    assert plain == ["› hello"]
    assert "\x1b[48;" not in "\n".join(lines)


def test_assistant_message_has_no_leading_separator() -> None:
    lines = render_cell(TranscriptCell("assistant", text="hi"), 40)

    assert lines
    assert set(strip_ansi(lines[0]).rstrip()) != {"─"}


def test_assistant_prefix_color_changes_with_streamed_chars() -> None:
    # The prefix colour cycles with cumulative streamed characters, calibrated
    # to ~12 phase changes per 100 chars.  Spinner frame is independent.
    early = TranscriptCell("assistant", text="hi", status="streaming", chars_streamed=0)
    later = TranscriptCell("assistant", text="hi", status="streaming", chars_streamed=50)

    first = render_cell(early, 40, spinner_frame=0)[0]
    second = render_cell(later, 40, spinner_frame=0)[0]

    assert strip_ansi(first) == strip_ansi(second)
    assert first != second


def test_reasoning_cell_is_single_line_and_flattens_newlines() -> None:
    cell = TranscriptCell("reasoning", text="first line\nsecond line", status="streaming")
    lines = render_cell(cell, 80, spinner_frame=0)

    assert len(lines) == 1
    assert "first line second line" in strip_ansi(lines[0])


def test_reasoning_cell_uses_breathing_dot_not_spinner_frames() -> None:
    # The breath phase advances with streamed characters (target 12 phases
    # per 100 chars), not with the global spinner frame.
    early = TranscriptCell("reasoning", text="thinking", status="streaming", chars_streamed=0)
    later = TranscriptCell("reasoning", text="thinking", status="streaming", chars_streamed=16)

    first = strip_ansi(render_cell(early, 50, spinner_frame=0)[0])
    second = strip_ansi(render_cell(later, 50, spinner_frame=10)[0])

    assert "⠋" not in first and "⠙" not in second
    assert first.startswith("· ")
    assert second.startswith("● ")


def test_reasoning_animation_keeps_total_width_constant() -> None:
    # Width must stay constant across both spinner frames and char growth.
    widths: set[int] = set()
    for chars in range(0, 24, 4):
        cell = TranscriptCell("reasoning", text="thinking", status="streaming", chars_streamed=chars)
        widths.add(visible_len(render_cell(cell, 60, spinner_frame=chars)[0]))
    done = TranscriptCell("reasoning", text="thinking", status="done")
    widths.add(visible_len(render_cell(done, 60, spinner_frame=0)[0]))

    assert len(widths) == 1, f"animation must not jitter, got widths {widths}"


def test_markdown_renderer_accepts_256_color() -> None:
    lines = render_markdown("**hello**", 40)

    assert any("hello" in strip_ansi(line) for line in lines)


# ---------------------------------------------------------------------------
# Tool cell: light rule + indented output
# ---------------------------------------------------------------------------


def test_tool_cell_uses_rule_and_indented_output() -> None:
    call = {
        "name": "run_python",
        "call_id": "call_123",
        "arguments": '{"code":"from uv_agent_runtime import run_process_text\\nrun_process_text([\\"git\\", \\"status\\"])"}',
    }
    payload = {"returncode": 0, "run_id": "run_abcdef", "stdout": "one\ntwo"}
    lines = render_tool_cell(TranscriptCell("tool", call=call, payload=payload), 60)
    plain_lines = [strip_ansi(line) for line in lines]
    plain = "\n".join(plain_lines)

    assert plain_lines[0].startswith("── ")
    assert "✓" in plain_lines[0] and "run_python" in plain_lines[0]
    assert "print(1)" not in plain_lines[0]
    assert "run_process_text" in plain
    assert "from uv_agent_runtime" not in plain
    assert "┌" not in plain and "└" not in plain and "│" not in plain
    assert any(line.startswith("  one") for line in plain_lines)
    assert any(line.startswith("  two") for line in plain_lines)


def test_tool_cell_compresses_script_and_output_lines() -> None:
    call = {
        "name": "run_python",
        "call_id": "call_123",
        "arguments": '{"code":"from uv_agent_runtime import path_info\\npath_info(\\"0\\")\\npath_info(\\"1\\")\\npath_info(\\"2\\")\\npath_info(\\"3\\")\\npath_info(\\"4\\")\\npath_info(\\"5\\")\\npath_info(\\"6\\")"}',
    }
    payload = {"returncode": 0, "stdout": "\n".join(f"out{i}" for i in range(8))}
    lines = render_tool_cell(TranscriptCell("tool", call=call, payload=payload), 80)
    plain_lines = [strip_ansi(line) for line in lines]

    assert [line.strip() for line in plain_lines if line.strip().startswith("path_info")] == [
        'path_info("0")',
        'path_info("1")',
        'path_info("2")',
        'path_info("3")',
        'path_info("4")',
        'path_info("5")',
    ]
    assert any("… more helpers +1 calls" in line for line in plain_lines)
    assert [line.strip() for line in plain_lines if line.strip().startswith("out")] == [
        "out0",
        "out1",
        "out2",
        "out3",
        "out4",
    ]
    assert any("... 3 more lines" in line for line in plain_lines)


def test_tool_cell_uses_payload_helper_calls_without_source() -> None:
    call = {"name": "run_python", "call_id": "call_123", "arguments": '{"code":"print(1)"}'}
    payload = {
        "returncode": 0,
        "helper_calls": [{"name": "replace_text", "args": '"a.txt", "old", "new"'}],
    }

    plain = "\n".join(strip_ansi(line) for line in render_tool_cell(TranscriptCell("tool", call=call, payload=payload), 80))

    assert 'replace_text("a.txt", "old", "new")' in plain
    assert "print(1)" not in plain


def test_running_tool_cell_uses_spinner_glyph() -> None:
    call = {"name": "run_python", "call_id": "x", "arguments": "{}"}
    cell = TranscriptCell("tool", status="running", call=call)
    lines = render_tool_cell(cell, 60)

    header = strip_ansi(lines[0])
    assert "⠿" in header
    assert "running" in header


# ---------------------------------------------------------------------------
# Composer: rounded-corner box with inline hint when empty
# ---------------------------------------------------------------------------


def test_empty_composer_is_boxed_with_inline_hint() -> None:
    lines, row, col = render_composer_with_cursor("", 60)

    assert len(lines) == 3
    top, middle, bottom = (strip_ansi(line) for line in lines)
    assert top.startswith("╭") and top.endswith("╮")
    assert bottom.startswith("╰") and bottom.endswith("╯")
    assert "│" in middle and "› " in middle and "Ask" in middle
    assert row == 1  # input row inside the box
    assert col == 4  # "│ " + "› "


def test_empty_composer_uses_chinese_placeholder_when_language_zh() -> None:
    lines, _, _ = render_composer_with_cursor("", 60, language="zh")

    middle = strip_ansi(lines[1])
    assert "输入" in middle


def test_short_composer_grows_one_input_row_per_line() -> None:
    short_lines, short_row, short_col = render_composer_with_cursor("abc", 50)
    multi_lines, multi_row, _ = render_composer_with_cursor("a\nb", 50)

    assert len(short_lines) == 3  # top + 1 input row + bottom
    assert short_row == 1
    assert short_col == 4 + len("abc")
    assert strip_ansi(short_lines[1]).startswith("│ › abc")

    assert len(multi_lines) == 4  # top + 2 input rows + bottom
    assert multi_row == 2
    assert strip_ansi(multi_lines[1]).startswith("│ › a")
    assert strip_ansi(multi_lines[2]).startswith("│   b")


def test_composer_cursor_can_render_before_end() -> None:
    lines, row, col = render_composer_with_cursor("abc", 50, cursor_index=1)

    assert strip_ansi(lines[1]).startswith("│ › abc")
    assert row == 1
    assert col == 5  # "│ " + "› " + one visible cell


def test_composer_trailing_newline_shows_blank_second_input_row() -> None:
    lines, row, col = render_composer_with_cursor("hello\n", 50)

    plain = [strip_ansi(line) for line in lines]
    assert len(lines) == 4  # top + "hello" + blank continuation + bottom
    assert plain[1].startswith("│ › hello")
    assert plain[2].startswith("│   ")
    assert row == 2
    assert col == 4


def test_long_composer_caps_visible_rows_and_summarizes_hidden() -> None:
    text = "\n".join(str(i) for i in range(20))
    lines, row, col = render_composer_with_cursor(text, 50, max_input_rows=6)

    assert len(lines) == 8  # top + 6 input rows + bottom
    assert row == 7 - 1  # last input row (above bottom border)
    assert col >= 4
    assert any("earlier lines hidden" in strip_ansi(line) for line in lines)
    assert all(visible_len(line) <= 50 for line in lines)


def test_long_composer_hidden_marker_does_not_replace_first_visible_input_row() -> None:
    text = "\n".join(f"line {i}" for i in range(12))
    lines, _, _ = render_composer_with_cursor(text, 60, max_input_rows=6)
    plain = [strip_ansi(line) for line in lines]

    assert "earlier lines hidden" in plain[0]
    assert plain[1].startswith("│   line 6")
    assert all("earlier lines hidden" not in line for line in plain[1:-1])


def test_multiline_composer_keeps_first_lines_visible_until_window_fills() -> None:
    text = "\n".join(f"line {i}" for i in range(6))
    lines, row, _ = render_composer_with_cursor(text, 60, max_input_rows=6)
    plain = [strip_ansi(line) for line in lines]

    assert plain[1].startswith("│ › line 0")
    assert plain[6].startswith("│   line 5")
    assert not any("hidden" in line for line in plain)
    assert row == 6


def test_cjk_input_width_and_cursor_position() -> None:
    assert visible_len("你好") == 4
    lines, row, col = render_composer_with_cursor("你好", 50)

    assert row == 1
    assert col == 4 + visible_len("你好")
    assert all(visible_len(line) <= 50 for line in lines)


def test_long_cjk_composer_soft_wraps_without_ellipsis() -> None:
    text = "这是一个测试" * 6
    lines, row, col = render_live_with_cursor(Tui2State(composer=text), 30)

    plain = [strip_ansi(line) for line in lines]
    assert len(lines) > 3
    assert not any("…" in line for line in plain)
    assert all(visible_len(line) <= 30 for line in lines)
    assert row == len(lines) - 2
    assert 4 <= col <= 30


# ---------------------------------------------------------------------------
# Status lines: two-row context strip above the composer
# ---------------------------------------------------------------------------


def test_idle_session_has_no_status_lines() -> None:
    state = Tui2State()
    assert render_status_lines(state, 80, 0) == []


def test_busy_state_renders_activity_line() -> None:
    state = Tui2State(busy=True, turn_elapsed_s=12.0)
    lines = render_status_lines(state, 80, 0)

    assert len(lines) == 1
    assert "Working" in strip_ansi(lines[0])
    assert "0:12" in strip_ansi(lines[0]) or "12" in strip_ansi(lines[0])


def test_busy_state_uses_status_message_as_primary_label() -> None:
    state = Tui2State(busy=True, status_message="回复中", turn_elapsed_s=13.0)

    plain = strip_ansi(render_status_lines(state, 80, 1)[0])

    assert "回复中" in plain
    assert "working" not in plain


def test_context_row_includes_goal_model_project_but_not_thread_title() -> None:
    # The thread title is shown in the terminal title and the /status output;
    # the bottom status row keeps only goal + model + project so the row stays
    # scannable.
    state = Tui2State(
        thread_id="t-abc",
        title="my thread",
        level="gpt-5-codex",
        project_path="/home/user/proj",
        goal_enabled=True,
        goal_objective="ship it",
    )
    lines = render_status_lines(state, 120, 0)
    plain = "\n".join(strip_ansi(line) for line in lines)

    assert "⊕ goal" in plain
    assert "my thread" not in plain
    assert "t-abc" not in plain
    assert "gpt-5-codex" in plain
    assert "/home/user/proj" in plain


def test_context_row_shrinks_home_path(monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: Path("C:/Users/me"))
    state = Tui2State(project_path="C:/Users/me/work/proj", level="test")
    lines = render_status_lines(state, 120, 0)
    plain = "\n".join(strip_ansi(line) for line in lines)

    assert "~/" in plain
    assert "C:/Users/me" not in plain


def test_context_row_styles_all_separators_consistently() -> None:
    state = Tui2State(
        level="gpt-5.5-xhigh",
        context_percent=31,
        project_path=r"C:\Users\me\Desktop\Project\AI\uv-agent",
    )

    line = render_status_lines(state, 120, 0)[0]

    assert strip_ansi(line).count(" · ") == 2
    assert line.count(sgr(DEFAULT_THEME.muted, " · ")) == 2


def test_tall_render_area_caps_to_viewport() -> None:
    # Use a user cell so we get one rendered row per input line — Markdown
    # rendering of an assistant cell collapses many rows into one paragraph.
    state = Tui2State(composer="")
    big = TranscriptCell("user", text="\n".join(f"row {i}" for i in range(80)))
    state.live.append(big)

    uncapped, _, _ = render_live_with_cursor(state, 80, 0)
    assert len(uncapped) > 50

    capped, cursor_row, _ = render_live_with_cursor(state, 80, 0, max_height=10)
    assert len(capped) <= 10
    assert any("earlier lines hidden" in strip_ansi(line) for line in capped)
    assert cursor_row < len(capped)


def test_busy_render_layout_has_status_and_box() -> None:
    state = Tui2State(busy=True, turn_elapsed_s=3.0, composer="hi")
    lines, cursor_row, _ = render_live_with_cursor(state, 60, spinner_frame=1)

    plain = [strip_ansi(line) for line in lines]
    assert any("Working" in line for line in plain)
    assert any(line.startswith("╭") for line in plain)
    assert any(line.startswith("╰") for line in plain)
    assert cursor_row == len(lines) - 2  # cursor is in the input row of the box


# ---------------------------------------------------------------------------
# Renderer: robust full-repaint
# ---------------------------------------------------------------------------


def test_first_repaint_uses_sync_output() -> None:
    output = io.StringIO()
    Renderer(output=output).repaint(Tui2State(composer=""))
    rendered = output.getvalue()

    assert "\x1b[?2026h" in rendered and "\x1b[?2026l" in rendered
    assert "\x1b[J" not in rendered  # no previous frame to erase


def test_second_repaint_erases_using_tracked_cursor_row() -> None:
    output = io.StringIO()
    renderer = Renderer(output=output)

    renderer.repaint(Tui2State(composer=""))
    expected = renderer.cursor_row
    assert expected >= 1, "boxed composer puts the cursor below the top border"
    output.seek(0)
    output.truncate(0)

    renderer.repaint(Tui2State(composer="a"))
    rendered = output.getvalue()

    assert rendered.startswith(f"\x1b[?2026h\r\x1b[{expected}A\x1b[J")
    assert "\r\n" in rendered  # CR+LF separator avoids POSIX staircase


def test_grown_paint_area_remembers_cursor_for_next_erase() -> None:
    output = io.StringIO()
    renderer = Renderer(output=output)

    state = Tui2State(composer="line1\nline2\nline3", busy=True, turn_elapsed_s=1.0)
    renderer.repaint(state)
    expected_row = renderer.cursor_row
    assert expected_row > 1
    output.seek(0)
    output.truncate(0)

    renderer.repaint(state)
    erase = output.getvalue()
    assert f"\x1b[{expected_row}A" in erase
    assert "\x1b[J" in erase


def test_flush_cell_separates_cells_with_blank_line() -> None:
    output = io.StringIO()
    Renderer(output=output).flush_cell(TranscriptCell("user", text="hi"))
    rendered = output.getvalue()

    # Each flush ends with two CRLFs → one blank visual row between cells.
    assert rendered.rstrip("\x1b[?2026l").endswith("\r\n\r\n")


def test_flush_cell_only_uses_crlf_separators() -> None:
    output = io.StringIO()
    Renderer(output=output).flush_cell(TranscriptCell("user", text="a\nb"))
    plain = strip_ansi(output.getvalue()).replace("\x1b[?2026h", "").replace("\x1b[?2026l", "")
    bare_lf = [i for i, ch in enumerate(plain) if ch == "\n" and (i == 0 or plain[i - 1] != "\r")]
    assert bare_lf == []


def test_renderer_reserves_last_column_to_avoid_terminal_autowrap(monkeypatch) -> None:
    monkeypatch.setattr("uv_agent.tui2.renderer.terminal_size", lambda default=(100, 30): (40, 10))
    output = io.StringIO()
    renderer = Renderer(output=output)

    renderer.repaint(Tui2State(composer="hello"))

    assert renderer.width == 39
    assert all(
        visible_len(line) <= 39
        for line in strip_ansi(output.getvalue()).splitlines()
        if line
    )

    output.seek(0)
    output.truncate(0)

    renderer.flush_cell(TranscriptCell("tool", payload={"returncode": 0, "run_id": "run_" + "x" * 24}))

    assert renderer.width == 39
    assert all(
        visible_len(line) <= 39
        for line in strip_ansi(output.getvalue()).splitlines()
        if line
    )


def test_idempotent_repaint_wraps_in_sync_output() -> None:
    output = io.StringIO()
    renderer = Renderer(output=output)
    renderer.repaint(Tui2State(composer=""))
    output.seek(0)
    output.truncate(0)

    renderer.repaint(Tui2State(composer=""))
    rendered = output.getvalue()
    assert "\x1b[?2026h" in rendered and "\x1b[?2026l" in rendered


# ---------------------------------------------------------------------------
# Key handling: Ctrl combos, history, Tab completion
# ---------------------------------------------------------------------------


class _DummyEngine:
    def __init__(self) -> None:
        self.turns: list[dict[str, object]] = []
        self.goal_updates: list[dict[str, object]] = []
        self.goal_states: dict[str, SimpleNamespace] = {}

    class config:
        class runtime:
            default_level = "test"

        ui = SimpleNamespace(completion_notification=SimpleNamespace(enabled=True, bell=True))

        levels = {"alpha": object(), "test": object()}

        @staticmethod
        def level(level):
            class Level:
                model = f"{level or 'test'}-model"

            return Level()

        @staticmethod
        def model_for_level(level):
            class Model:
                name = f"{level or 'test'}-model"
                model = f"{level or 'test'}-provider-model"
                context_window_tokens = 1000

            return Model()

    config = config()

    @staticmethod
    def context_stats(thread_id, level):
        class Stats:
            percent = 42
            used_tokens = 420
            context_window_tokens = 1000
            source = "estimate"
            threshold_tokens = 800
            headroom_tokens = 580

        return Stats()

    class thread_store:
        threads: list[dict] = []
        events: list[dict] = []
        snapshots: dict[str, dict] = {}

        @classmethod
        def create_thread(cls, title="New thread"):
            thread_id = f"thr_{len(cls.threads) + 1}"
            cls.threads.append({"thread_id": thread_id, "title": title})
            return thread_id

        @classmethod
        def append(cls, thread_id, event_type, **data):
            event = {"type": event_type, "thread_id": thread_id, **data}
            cls.events.append(event)
            if event_type == "thread.level_updated":
                for thread in cls.threads:
                    if thread.get("thread_id") == thread_id:
                        thread["active_level"] = data.get("level")
                        thread["active_model"] = data.get("model")
                        break
            return event

        @classmethod
        def thread_digest(cls, thread_id):
            for thread in cls.threads:
                if thread.get("thread_id") == thread_id:
                    return {"title": "Stored title", **thread}
            return {"title": "Stored title"}

        @classmethod
        def list_threads(cls):
            return list(cls.threads)

        @classmethod
        def read_history_segment(cls, *args, **kwargs):
            from uv_agent.session.store import ThreadHistorySegment

            return ThreadHistorySegment(events=[], start_event_id=0, end_event_id=0, has_more=False)

        @classmethod
        def snapshot(cls, thread_id):
            class Snapshot:
                metadata = dict(
                    cls.snapshots.get(thread_id)
                    or next(
                        (dict(thread) for thread in cls.threads if thread.get("thread_id") == thread_id),
                        {},
                    )
                )

            return Snapshot()

    async def run_turn(self, *, user_text, thread_id=None, level=None, cancel_event=None):
        thread_id = thread_id or self.thread_store.create_thread("New thread")
        self.turns.append({"user_text": user_text, "thread_id": thread_id, "level": level})
        yield {"type": "turn.started", "thread_id": thread_id, "turn_id": f"turn_{len(self.turns)}"}
        yield {"type": "turn.completed", "thread_id": thread_id, "turn_id": f"turn_{len(self.turns)}"}

    def enable_goal_mode(self, thread_id, *, objective=""):
        state = SimpleNamespace(enabled=True, status="enabled", objective=objective)
        self.goal_states[thread_id] = state
        self.goal_updates.append(
            {
                "op": "enable",
                "thread_id": thread_id,
                "objective": objective,
                "turns_started": len(self.turns),
            }
        )
        for thread in self.thread_store.threads:
            if thread.get("thread_id") == thread_id:
                thread["goal_mode"] = {"enabled": True, "objective": objective}
                break
        return state

    def disable_goal_mode(self, thread_id):
        previous = self.goal_states.get(thread_id)
        objective = str(getattr(previous, "objective", "")) if previous is not None else ""
        state = SimpleNamespace(enabled=False, status="disabled", objective=objective)
        self.goal_states[thread_id] = state
        self.goal_updates.append({"op": "disable", "thread_id": thread_id, "objective": objective})
        for thread in self.thread_store.threads:
            if thread.get("thread_id") == thread_id:
                thread["goal_mode"] = {"enabled": False, "objective": objective}
                break
        return state

    def reset_goal_files(self, thread_id, *, objective=""):
        state = SimpleNamespace(enabled=False, status="disabled", objective=objective)
        self.goal_states[thread_id] = state
        self.goal_updates.append({"op": "reset", "thread_id": thread_id, "objective": objective})
        return state

    def goal_state(self, thread_id):
        if not thread_id:
            return None
        return self.goal_states.get(thread_id)


class _DummyRenderer:
    def __init__(self) -> None:
        self.output = io.StringIO()
        self._has_frame = False
        self.width = 80
        self.flushed: list[TranscriptCell] = []

    def repaint(self, state) -> None:
        pass

    def flush_cell(self, cell) -> None:
        self.flushed.append(cell)

    def flush_cells(self, cells) -> None:
        for cell in cells:
            self.flush_cell(cell)

    def clear_screen(self, *, rule=None) -> None:
        if rule:
            self.output.write(rule + "\n")


def _make_app(monkeypatch):
    from uv_agent.tui2.app import AnsiUvAgentApp

    engine = _DummyEngine()
    engine.thread_store.threads = []
    engine.thread_store.events = []
    engine.thread_store.snapshots = {}
    monkeypatch.setattr("uv_agent.tui2.app.create_engine", lambda *a, **k: engine)
    app = AnsiUvAgentApp()
    app.renderer = _DummyRenderer()
    return app


def test_ctrl_c_requires_second_press_to_exit_when_idle(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    assert asyncio.run(app.handle_key("\x03")) is True
    # The status message is localised, so check that quit was armed instead
    # of pinning a particular language's phrase.
    assert app._quit_armed
    assert app.state.status_message  # localised hint is present
    assert asyncio.run(app.handle_key("\x03")) is False


def test_ctrl_c_quit_confirmation_expires_after_timeout(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    times = iter([10.0, 10.0 + tui2_app.CTRL_C_CONFIRMATION_S + 0.01])
    monkeypatch.setattr(tui2_app, "monotonic", lambda: next(times))

    assert asyncio.run(app.handle_key("\x03")) is True
    assert app._quit_armed
    assert app._expire_quit_confirmation() is True

    assert not app._quit_armed
    assert app.state.status_message == "ready"


def test_second_ctrl_c_after_confirmation_timeout_rearms_instead_of_exiting(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    now = 20.0

    def fake_monotonic() -> float:
        return now

    monkeypatch.setattr(tui2_app, "monotonic", fake_monotonic)

    assert asyncio.run(app.handle_key("\x03")) is True
    now += tui2_app.CTRL_C_CONFIRMATION_S + 0.01

    assert asyncio.run(app.handle_key("\x03")) is True

    assert app._quit_armed


def test_ctrl_c_preserves_composer_while_arming_quit(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "abc"
    assert asyncio.run(app.handle_key("\x03")) is True
    assert app.state.composer == "abc"


def test_regular_key_cancels_ctrl_c_quit_confirmation(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    asyncio.run(app.handle_key("\x03"))
    asyncio.run(app.handle_key("a"))

    assert asyncio.run(app.handle_key("\x03")) is True


def test_ctrl_u_clears_composer(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "hello"
    assert asyncio.run(app.handle_key("\x15")) is True
    assert app.state.composer == ""


def test_ctrl_w_deletes_last_word(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "hello world"
    assert asyncio.run(app.handle_key("\x17")) is True
    assert app.state.composer == "hello "


def test_ctrl_a_and_ctrl_e_move_to_logical_line_edges(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "ab\ncd\nef"
    app.state.composer_cursor = len("ab\nc")

    assert asyncio.run(app.handle_key("\x01")) is True
    assert app.state.composer_cursor == len("ab\n")

    assert asyncio.run(app.handle_key("\x05")) is True
    assert app.state.composer_cursor == len("ab\ncd")


def test_ctrl_k_deletes_to_line_end_then_line_break(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "ab\ncd\nef"
    app.state.composer_cursor = len("ab\nc")

    assert asyncio.run(app.handle_key("\x0b")) is True
    assert app.state.composer == "ab\nc\nef"
    assert app.state.composer_cursor == len("ab\nc")

    assert asyncio.run(app.handle_key("\x0b")) is True
    assert app.state.composer == "ab\ncef"
    assert app.state.composer_cursor == len("ab\nc")


def test_left_right_arrows_move_composer_cursor(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "abc"
    asyncio.run(app.handle_key("<LEFT>"))
    asyncio.run(app.handle_key("X"))

    assert app.state.composer == "abXc"
    assert app.state.composer_cursor == 3
    asyncio.run(app.handle_key("<RIGHT>"))
    assert app.state.composer_cursor == 4


def test_backspace_uses_composer_cursor(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "abc"
    app.state.composer_cursor = 1
    asyncio.run(app.handle_key("\b"))

    assert app.state.composer == "bc"
    assert app.state.composer_cursor == 0


def test_composer_history_persists_across_app_instances(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("uv_agent.tui2.app.uv_agent_home", lambda: tmp_path)
    first = _make_app(monkeypatch)
    first._remember_composer_input("hello")

    second = _make_app(monkeypatch)

    assert second._history == ["hello"]


def test_composer_history_save_load_uses_original_tui_format(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("uv_agent.tui2.app.uv_agent_home", lambda: tmp_path)

    save_composer_history(["", "one", "one", "two"])
    raw = json.loads((tmp_path / "composer_history.json").read_text(encoding="utf-8"))

    assert raw == {"items": ["", "one", "one", "two"]}
    assert load_composer_history() == ["one", "two"]


def test_history_arrow_keys_navigate_submissions(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._history = ["first", "second"]
    asyncio.run(app.handle_key("<H>"))
    assert app.state.composer == "second"
    asyncio.run(app.handle_key("<H>"))
    assert app.state.composer == "first"
    asyncio.run(app.handle_key("<P>"))
    assert app.state.composer == "second"


def test_posix_arrow_keys_navigate_submissions(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._history = ["first", "second"]
    asyncio.run(app.handle_key("<UP>"))
    assert app.state.composer == "second"
    asyncio.run(app.handle_key("<DOWN>"))
    assert app.state.composer == ""


def test_up_down_arrows_move_to_edges_at_composer_boundaries(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    app.state.composer = "abc"
    app.state.composer_cursor = 1
    asyncio.run(app.handle_key("<UP>"))
    assert app.state.composer_cursor == 0
    app.state.composer_cursor = 1
    asyncio.run(app.handle_key("<DOWN>"))
    assert app.state.composer_cursor == len("abc")

    app.state.composer = "abc\ndef"
    app.state.composer_cursor = len("ab")
    asyncio.run(app.handle_key("<UP>"))
    assert app.state.composer_cursor == 0
    app.state.composer_cursor = len("abc\nde")
    asyncio.run(app.handle_key("<DOWN>"))
    assert app.state.composer_cursor == len("abc\ndef")


def test_history_arrow_ignores_non_empty_composer(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._history = ["first", "second"]
    app.state.composer = "draft"

    asyncio.run(app.handle_key("<UP>"))

    assert app.state.composer == "draft"
    assert app._history_cursor is None


def test_editing_recalled_history_disables_further_navigation(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._history = ["first", "second"]
    asyncio.run(app.handle_key("<UP>"))
    asyncio.run(app.handle_key("!"))

    asyncio.run(app.handle_key("<UP>"))

    assert app.state.composer == "second!"
    assert app._history_cursor is None


def test_slash_opens_command_palette(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    asyncio.run(app.handle_key("/"))

    assert app.state.command_palette_open
    assert any(item.value == "/help" for item in app.state.command_palette_items)


def test_command_palette_lists_clear_not_new() -> None:
    values = [item.value for item in TOP_LEVEL_COMMANDS]

    assert "/clear" in values
    assert "/new" not in values


def test_command_palette_lists_status_command() -> None:
    values = [item.value for item in TOP_LEVEL_COMMANDS]

    assert "/status" in values


def test_status_command_flushes_context_summary(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    app._handle_command("/status")

    last = app.state.flushed[-1]
    assert last.kind == "event"
    assert "context:" in last.text
    assert "42%" in last.text


def test_streamed_reasoning_is_not_flushed_again_on_model_response(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    app._handle_event({"type": "assistant.reasoning_delta", "text": "plan"})
    app._handle_event({"type": "assistant.delta", "text": "answer"})
    app._handle_event({
        "type": "model.response",
        "reasoning_text": "plan",
        "response": SimpleNamespace(output=[]),
    })

    reasoning_cells = [cell for cell in app.state.flushed if cell.kind == "reasoning"]
    assert [cell.text for cell in reasoning_cells] == ["plan"]


def test_provider_only_reasoning_still_flushes_on_model_response(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    app._handle_event({
        "type": "model.response",
        "reasoning_text": "plan",
        "response": SimpleNamespace(output=[]),
    })

    reasoning_cells = [cell for cell in app.state.flushed if cell.kind == "reasoning"]
    assert [cell.text for cell in reasoning_cells] == ["plan"]


def test_tool_output_allows_next_response_reasoning_to_flush(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    app._handle_event({"type": "assistant.reasoning_delta", "text": "tool plan"})
    app._handle_event({
        "type": "model.response",
        "reasoning_text": "tool plan",
        "response": SimpleNamespace(output=[{"type": "function_call", "call_id": "call_1"}]),
    })
    app._handle_event({
        "type": "tool.output",
        "call": {"call_id": "call_1"},
        "output": {"output": json.dumps({"returncode": 0, "stdout": "ok"})},
    })
    app._handle_event({
        "type": "model.response",
        "reasoning_text": "final plan",
        "response": SimpleNamespace(output=[]),
    })

    reasoning_cells = [cell for cell in app.state.flushed if cell.kind == "reasoning"]
    assert [cell.text for cell in reasoning_cells] == ["tool plan", "final plan"]


def test_turn_completed_plays_terminal_buzzer(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr("uv_agent.tui2.app.play_terminal_buzzer", lambda: calls.append("buzzer") or True)

    app._handle_event({"type": "turn.completed"})

    assert calls == ["buzzer"]


def test_turn_completed_respects_buzzer_config(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.config.ui = SimpleNamespace(completion_notification=SimpleNamespace(enabled=True, bell=False))
    calls: list[str] = []
    monkeypatch.setattr("uv_agent.tui2.app.play_terminal_buzzer", lambda: calls.append("buzzer") or True)

    app._handle_event({"type": "turn.completed"})

    assert calls == []


def test_at_opens_file_mention_palette(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    asyncio.run(app.handle_key("@"))

    values = [item.value for item in app.state.command_palette_items]
    assert app.state.command_palette_open
    assert any(value.startswith("@src/") for value in values)
    assert "@thread:" not in values
    assert "@skill:" not in values
    assert "@mcp:" not in values


def test_double_at_completes_threads_and_inserts_thread_mention(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [
        {"thread_id": "thr_1", "title": "Alpha", "last_text": "hello", "turn_count": 2},
    ]
    app.state.composer = "see @@alp"
    app.state.composer_cursor = len(app.state.composer)
    app._after_composer_changed()
    assert app.state.command_palette_open

    app._accept_command_palette_selection()

    assert app.state.composer == "see @thread:thr_1 "


def test_threads_command_opens_interactive_picker(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [{"thread_id": "thr_1", "title": "Alpha", "last_text": "hello"}]

    app._handle_command("/threads")

    assert app.state.command_palette_open
    assert app._picker_mode == "thread"
    assert app.state.command_palette_items[0].id == "thr_1"


def test_command_palette_supports_goal_subcommands(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "/goal "
    app._refresh_command_palette()

    values = [item.value for item in app.state.command_palette_items]
    assert "/goal enable" in values
    assert "/goal status" in values


def test_command_palette_supports_level_names(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "/level "
    app._refresh_command_palette()

    values = [item.value for item in app.state.command_palette_items]
    assert "/level alpha" in values
    assert "/level test" in values


def test_start_turn_persists_selected_level_for_new_thread(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    app._handle_command("/level alpha")

    async def run_turn() -> None:
        await app._start_turn("first")
        assert app._running_task is not None
        await app._running_task

    asyncio.run(run_turn())

    assert app.state.thread_id == "thr_1"
    assert app.engine.turns[-1]["level"] == "alpha"
    assert app.engine.thread_store.threads[0]["active_level"] == "alpha"
    assert app.engine.thread_store.threads[0]["active_model"] == "alpha-model"


def test_resuming_thread_restores_persisted_level(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.level = "test"
    app.engine.thread_store.threads = [
        {
            "thread_id": "thr_alpha",
            "title": "Alpha work",
            "active_level": "alpha",
            "active_model": "alpha-model",
        },
        {
            "thread_id": "thr_test",
            "title": "Test work",
            "active_level": "test",
            "active_model": "test-model",
        },
    ]

    app._resume_thread("thr_alpha")

    assert app.state.level == "alpha"

    async def run_turn() -> None:
        await app._start_turn("continue")
        assert app._running_task is not None
        await app._running_task

    asyncio.run(run_turn())

    assert app.engine.turns[-1]["thread_id"] == "thr_alpha"
    assert app.engine.turns[-1]["level"] == "alpha"


def test_resume_thread_prefers_snapshot_level_over_picker_listing(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.level = "test"
    app.engine.thread_store.threads = [
        {
            "thread_id": "thr_alpha",
            "title": "Alpha work",
            "active_level": "test",
            "active_model": "test-model",
        }
    ]
    app.engine.thread_store.snapshots = {
        "thr_alpha": {
            "thread_id": "thr_alpha",
            "title": "Alpha work",
            "active_level": "alpha",
            "active_model": "alpha-model",
        }
    }

    app._resume_thread("thr_alpha")

    assert app.state.level == "alpha"


def test_tab_completes_unique_command_prefix(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "/quit"
    asyncio.run(app.handle_key("\t"))
    assert app.state.composer == "/quit"


def test_quit_command_exits_without_confirmation(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "/quit"

    assert asyncio.run(app.handle_key("\r")) is False

    assert app.state.composer == ""
    assert not app._quit_armed


def test_tab_cycles_through_matching_commands(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "/c"

    asyncio.run(app.handle_key("\t"))
    first = app.state.composer
    asyncio.run(app.handle_key("\t"))
    second = app.state.composer

    assert first.startswith("/c")
    assert second.startswith("/c")
    assert first != second  # cycled to next match


def test_ctrl_enter_inserts_newline(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "hello"
    asyncio.run(app.handle_key("<C-ENTER>"))

    assert app.state.composer == "hello\n"


def test_bracketed_paste_inserts_multiline_text_without_submitting(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "prefix "

    asyncio.run(app.handle_key(PASTE_PREFIX + "one\ntwo"))

    assert app.state.composer == "prefix one\ntwo"
    assert app.engine.turns == []


def test_plain_enter_shortly_after_typing_is_treated_as_paste_newline(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    asyncio.run(app.handle_key("a"))
    asyncio.run(app.handle_key("\r"))
    asyncio.run(app.handle_key("b"))

    assert app.state.composer == "a\nb"
    assert app.engine.turns == []


def test_plain_crlf_shortly_after_typing_inserts_one_paste_newline(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    asyncio.run(app.handle_key("a"))
    asyncio.run(app.handle_key("\r"))
    asyncio.run(app.handle_key("\n"))
    asyncio.run(app.handle_key("b"))

    assert app.state.composer == "a\nb"
    assert app.engine.turns == []


def test_plain_enter_after_idle_still_submits(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "hello"
    app._last_plain_input_at = None

    async def run() -> None:
        assert await app.handle_key("\r") is True
        assert app._running_task is not None
        await app._running_task

    asyncio.run(run())

    assert app.engine.turns[-1]["user_text"] == "hello"


def test_terminal_reads_bracketed_paste_as_single_key() -> None:
    terminal = Terminal(stdin=io.StringIO("\x1b[200~one\r\ntwo\x1b[201~"))
    terminal._windows = False

    assert terminal.read_key() == PASTE_PREFIX + "one\ntwo"


def test_windows_terminal_reads_vt_paste_before_enter(monkeypatch) -> None:
    terminal = Terminal()
    terminal._windows = True
    chars = iter("\x1b[200~one\r\ntwo\x1b[201~\r")
    monkeypatch.setattr("msvcrt.getwch", lambda: next(chars))

    assert terminal.read_key() == PASTE_PREFIX + "one\ntwo"
    assert terminal.read_key() == "\r"


def test_windows_terminal_coalesces_unbracketed_paste(monkeypatch) -> None:
    terminal = Terminal()
    terminal._windows = True
    chars = iter("one\r\ntwo")
    remaining = [True] * len("one\r\ntwo")

    def fake_getwch() -> str:
        remaining.pop(0)
        return next(chars)

    monkeypatch.setattr("msvcrt.getwch", fake_getwch)
    monkeypatch.setattr("msvcrt.kbhit", lambda: bool(remaining))

    assert terminal.read_key() == PASTE_PREFIX + "one\ntwo"


def test_unbracketed_paste_fallback_does_not_swallow_stringio_input() -> None:
    terminal = Terminal(stdin=io.StringIO("ab"))
    terminal._windows = False

    assert terminal.read_key() == "a"
    assert terminal.read_key() == "b"


def test_command_palette_render_shows_selection() -> None:
    state = Tui2State(
        composer="/",
        command_palette_open=True,
        command_palette_items=[CommandSuggestion("/help", "show help")],
    )
    lines, _, _ = render_live_with_cursor(state, 60, 0)
    plain = "\n".join(strip_ansi(line) for line in lines)

    assert "/help" in plain
    assert "show help" in plain


def test_command_palette_scrolls_to_selected_item() -> None:
    state = Tui2State(
        composer="/",
        command_palette_open=True,
        command_palette_index=9,
        command_palette_items=[CommandSuggestion(f"/cmd{i}", f"command {i}") for i in range(12)],
    )
    lines, _, _ = render_live_with_cursor(state, 80, 0)
    plain = "\n".join(strip_ansi(line) for line in lines)

    assert "/cmd9" in plain
    assert "/cmd0" not in plain
    assert "↑" in plain


def test_live_region_does_not_insert_colored_status_separator() -> None:
    state = Tui2State(busy=True, turn_elapsed_s=1.0, composer="hi")
    lines, _, _ = render_live_with_cursor(state, 60, 0)
    plain_lines = [strip_ansi(line) for line in lines]

    assert not any(set(line) == {"─"} for line in plain_lines if line)


def test_tab_with_no_slash_does_nothing(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "hello"
    asyncio.run(app.handle_key("\t"))
    assert app.state.composer == "hello"


def test_window_title_uses_thread_title_and_busy_spinner(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    written: list[str] = []
    monkeypatch.setattr("uv_agent.tui2.app.write_window_title", written.append)
    app.state.thread_id = "T-test"

    app._refresh_window_title()
    assert written[-1] == "Stored title"

    app.state.busy = True
    app._spinner_index = 1
    app._apply_window_title()

    assert written[-1] == "⠙ Stored title"


def test_window_title_refreshes_when_turn_assigns_thread_id(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    written: list[str] = []
    monkeypatch.setattr("uv_agent.tui2.app.write_window_title", written.append)

    app._refresh_window_title()
    app._handle_event({"type": "turn.started", "thread_id": "T-test"})

    assert written == [app._text("new_thread"), "Stored title"]


def test_window_title_polls_pending_generated_title_while_busy(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    written: list[str] = []
    titles = iter(["New thread", "Generated title"])
    monkeypatch.setattr("uv_agent.tui2.app.write_window_title", written.append)
    monkeypatch.setattr(
        app.engine.thread_store,
        "thread_digest",
        lambda thread_id: {"title": next(titles)},
    )
    app.state.thread_id = "T-test"

    app._refresh_window_title()
    app.state.busy = True
    app._apply_window_title()

    assert written == [app._text("new_thread"), "⠋ Generated title"]


def test_window_title_is_sanitized_and_deduplicated(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    written: list[str] = []
    monkeypatch.setattr("uv_agent.tui2.app.write_window_title", written.append)
    app.state.title = "bad\x00title"

    app._refresh_window_title()
    app._refresh_window_title()

    assert written == ["badtitle"]


def test_goal_enable_without_thread_is_pending_until_first_send(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._handle_command("/goal enable build something")

    last = app.state.flushed[-1]
    assert last.kind == "event"
    assert "enabled for next message" in last.text
    assert app.state.thread_id is None
    assert app.state.goal_enabled
    assert app.state.goal_objective == "build something"
    assert app.engine.thread_store.threads == []
    assert app.engine.goal_updates == []


def test_goal_enable_palette_selection_submits_without_extra_enter(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "/goal "
    app._refresh_command_palette()
    values = [item.value for item in app.state.command_palette_items]
    app.state.command_palette_index = values.index("/goal enable")

    assert asyncio.run(app.handle_key("\r")) is True

    assert app.state.composer == ""
    assert not app.state.command_palette_open
    assert app.state.flushed[-1].kind == "event"
    assert "enabled for next message" in app.state.flushed[-1].text


def test_pending_goal_enable_materializes_before_first_turn(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._handle_command("/goal enable build something")

    async def run_turn() -> None:
        await app._start_turn("first")
        assert app._running_task is not None
        await app._running_task

    asyncio.run(run_turn())

    assert app.state.thread_id == "thr_1"
    assert app.engine.goal_updates == [
        {
            "op": "enable",
            "thread_id": "thr_1",
            "objective": "build something",
            "turns_started": 0,
        }
    ]
    assert app.engine.turns[-1]["thread_id"] == "thr_1"
    assert app.state.goal_enabled
    assert app.state.goal_objective == "build something"


def test_goal_disable_clears_pending_draft_goal(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    app._handle_command("/goal enable build something")
    app._handle_command("/goal disable")

    assert app.state.thread_id is None
    assert not app.state.goal_enabled
    assert app.state.goal_objective == ""
    assert app.engine.goal_updates == []
    assert app.state.flushed[-1].text == "goal mode disabled"


def test_goal_command_with_invalid_op_shows_usage(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "T-test"
    app._handle_command("/goal bogus")

    last = app.state.flushed[-1]
    assert last.kind == "error"
    assert "usage" in last.text
