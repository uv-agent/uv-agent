from __future__ import annotations

import asyncio
import json
import io
import math
import pytest
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from uv_agent.tui.formatting import RUNTIME_EVENT_EVENT_ID_KEY, RUNTIME_EVENT_RUN_ID_KEY, renderable_plain, structured_event_markup
from uv_agent.tui2.ansi import strip_ansi, visible_len
from uv_agent.session import ThreadLockedError
from uv_agent.tui2.components import (
    render_agent_view,
    render_agent_view_with_cursor,
    render_cell,
    render_composer_with_cursor,
    render_live_with_cursor,
    render_markdown,
    render_status_lines,
    render_tool_cell,
)
import uv_agent.tui2.app as tui2_app

from uv_agent.tui2.app import TOP_LEVEL_COMMANDS, ThreadRunState, load_composer_history, save_composer_history
from uv_agent.tui2.events import AgentViewRow, CommandSuggestion, TranscriptCell, Tui2State
from uv_agent.tui2.app import _retained_flushed_cell, TUI2_RETAINED_FLUSHED_TEXT_CHARS
from uv_agent.tui2.renderer import Renderer
from uv_agent.tui2.terminal import PASTE_PREFIX, Terminal, TerminalKeyReader
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


def test_tool_cell_has_no_rule_and_tree_indented_chains() -> None:
    call = {
        "name": "run_python",
        "call_id": "call_123",
        "arguments": '{"code":"from uv_agent_runtime import run_process_text\\nrun_process_text([])"}',
    }
    payload = {"returncode": 0, "run_id": "run_abcdef", "stdout": "one\ntwo"}
    lines = render_tool_cell(TranscriptCell("tool", call=call, payload=payload), 60)
    plain_lines = [strip_ansi(line) for line in lines]
    plain = "\n".join(plain_lines)

    # No horizontal rule on the title line.
    assert not plain_lines[0].startswith("── ")
    assert "✓" in plain_lines[0] and "run_python" in plain_lines[0]
    assert "run_process_text([])" not in plain
    # Success omits exit code; run id is shortened to the last 6 characters.
    assert "exit 0" not in plain
    assert "abcdef" in plain
    assert "run_abcdef" not in plain
    assert "run_process_text" in plain
    assert "from uv_agent_runtime" not in plain
    # stdout/stderr are no longer inlined; use /show <run_id> for full output.
    assert "one" not in plain and "two" not in plain
    # At width 60 the chain fits on the title line.
    assert len(plain_lines) == 1


def test_tool_cell_compresses_script_and_output_lines() -> None:
    call = {
        "name": "run_python",
        "call_id": "call_123",
        "arguments": '{"code":"from uv_agent_runtime import path_info\\npath_info(\\"0\\")\\npath_info(\\"1\\")\\npath_info(\\"2\\")\\npath_info(\\"3\\")\\npath_info(\\"4\\")\\npath_info(\\"5\\")\\npath_info(\\"6\\")"}',
    }
    payload = {"returncode": 0, "stdout": "\n".join(f"out{i}" for i in range(8))}
    lines = render_tool_cell(TranscriptCell("tool", call=call, payload=payload), 80)
    plain_lines = [strip_ansi(line) for line in lines]

    # Only the compact imported-name chain is shown; stdout is omitted.
    assert any("path_info x7" in line for line in plain_lines)
    assert not any(line.strip().startswith("out") for line in plain_lines)
    assert "exit 0" not in "\n".join(plain_lines)
    # At width 80 the chain fits on the title line.
    assert len(plain_lines) == 1


def test_tool_cell_uses_payload_helper_calls_without_source() -> None:
    call = {"name": "run_python", "call_id": "call_123", "arguments": '{"code":"print(1)"}'}
    payload = {
        "returncode": 0,
        "helper_calls": [{"name": "replace_text", "args": '"a.txt", "old", "new"'}],
    }

    plain = "\n".join(strip_ansi(line) for line in render_tool_cell(TranscriptCell("tool", call=call, payload=payload), 80))

    assert "replace_text" in plain
    assert "print(1)" not in plain
    assert "exit 0" not in plain
    assert len([line for line in plain.splitlines() if line.strip()]) == 1


def test_workflow_structured_event_markup_is_compact() -> None:
    started = renderable_plain(
        structured_event_markup(
            {"kind": "workflow.node.started", "key": "investigate", "node_kind": "agent"}
        )
    )
    completed = renderable_plain(
        structured_event_markup(
            {
                "kind": "workflow.node.completed",
                "key": "investigate",
                "thread_id": "thr_12345678",
                "returncode": 0,
            }
        )
    )
    failed = renderable_plain(
        structured_event_markup(
            {"kind": "workflow.node.failed", "node_id": "wfn_deadbeef", "returncode": 2}
        )
    )
    checkpoint = renderable_plain(
        structured_event_markup(
            {"kind": "workflow.checkpoint.reached", "key": "after_investigation"}
        )
    )

    assert started == "└─ workflow node investigate started agent"
    assert completed == "└─ workflow node investigate completed thread 12345678"
    assert failed == "└─ workflow node deadbeef failed exit 2"
    assert checkpoint == "└─ workflow checkpoint after_investigation reached"


def test_tool_cell_omits_events_and_stdout_in_compact_view() -> None:
    event = {
        "kind": "workflow.node.completed",
        "key": "investigate",
        "thread_id": "thr_node12345678",
        "returncode": 0,
        RUNTIME_EVENT_EVENT_ID_KEY: "evt_1",
        RUNTIME_EVENT_RUN_ID_KEY: "run_1",
    }
    payload = {
        "run_id": "run_1",
        "returncode": 0,
        "stdout": "visible output\n" + json.dumps(event) + "\n",
        "events": [event],
    }

    plain = "\n".join(
        strip_ansi(line)
        for line in render_tool_cell(TranscriptCell("tool", payload=payload), 100)
    )

    # Events and stdout are no longer inlined in the compact cell.
    assert "workflow node investigate completed thread 12345678" not in plain
    assert "visible output" not in plain
    assert RUNTIME_EVENT_EVENT_ID_KEY not in plain
    assert RUNTIME_EVENT_RUN_ID_KEY not in plain
    assert "run_1" in plain


def test_tui2_compaction_event_shows_preview_only(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    summary = "\n".join(f"summary line {i}" for i in range(8))

    app._handle_event({"type": "compaction.completed", "text": summary})

    last = app.state.flushed[-1]
    assert last.kind == "event"
    assert "conversation compacted" in last.text
    assert "summary line 0" in last.text
    assert "summary line 3" in last.text
    assert "summary line 4" not in last.text
    assert "... 4 more lines" in last.text


def test_tui2_history_compaction_cell_shows_preview_only(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    summary = "\n".join(f"history line {i}" for i in range(6))
    item = tui2_app.TimelineItem(
        id="compaction:1",
        kind="compaction",
        content={"text": summary},
    )

    cell = app._timeline_item_cell(item)

    assert cell is not None
    assert cell.kind == "event"
    assert "conversation compacted" in cell.text
    assert "history line 3" in cell.text
    assert "history line 4" not in cell.text
    assert "... 2 more lines" in cell.text


def test_running_tool_cell_uses_spinner_glyph() -> None:
    call = {"name": "run_python", "call_id": "x", "arguments": "{}"}
    cell = TranscriptCell("tool", status="running", call=call)
    lines = render_tool_cell(cell, 60)

    header = strip_ansi(lines[0])
    assert "⠿" in header
    assert "running" in header


def test_running_tool_cell_has_no_rule_and_constant_height() -> None:
    call = {
        "name": "run_python",
        "call_id": "call_" + "x" * 24,
        "arguments": '{"code":"from uv_agent_runtime import path_info\\npath_info(\\".\\")"}',
    }
    running = render_tool_cell(TranscriptCell("tool", status="running", call=call), 80)
    completed = render_tool_cell(TranscriptCell("tool", call=call, payload={"returncode": 0}), 80)

    # Running and completed tool cells stay one line; chain is omitted while
    # running and inlined on completion when it fits.
    assert len(running) == 1
    assert len(completed) == 1
    assert not strip_ansi(running[0]).startswith("── ")
    assert not strip_ansi(completed[0]).startswith("── ")
    assert "⠿" in strip_ansi(running[0])
    assert "running" in strip_ansi(running[0])
    assert "exit 0" not in strip_ansi(completed[0])


def test_running_tool_cell_height_is_constant_across_payload_growth() -> None:
    """Live tool cells must stay at a constant one-line height.

    A growing live frame can push its top row out of the viewport on
    terminals that don't honour DECAWM, leaving leaked
    ``── ⠿ run_python · running…`` rows in scrollback.  The compact cell
    keeps only the header and the static imported-call chain, so payload
    growth never changes the rendered height.
    """

    call = {
        "name": "run_python",
        "call_id": "x",
        "arguments": '{"code":"from uv_agent_runtime import path_info\\npath_info(\\".\\")"}',
    }
    empty = render_tool_cell(TranscriptCell("tool", status="running", call=call), 80)
    with_stdout = render_tool_cell(
        TranscriptCell(
            "tool",
            status="running",
            call=call,
            payload={"stdout": "line1\nline2\nline3"},
        ),
        80,
    )
    with_both = render_tool_cell(
        TranscriptCell(
            "tool",
            status="running",
            call=call,
            payload={"stdout": "a\nb\nc", "stderr": "warn1\nwarn2"},
        ),
        80,
    )

    assert len(empty) == len(with_stdout) == len(with_both) == 1
    for lines in (empty, with_stdout, with_both):
        plain = "\n".join(strip_ansi(line) for line in lines)
        assert "line1" not in plain and "warn1" not in plain
        assert "waiting for run_python output" not in plain
        assert "path_info" in plain  # call chain still shown for context

    completed = render_tool_cell(
        TranscriptCell(
            "tool",
            call=call,
            payload={"returncode": 0, "stdout": "done-line", "stderr": "done-warn"},
        ),
        80,
    )
    completed_plain = "\n".join(strip_ansi(line) for line in completed)
    assert "done-line" not in completed_plain
    assert "done-warn" not in completed_plain
    assert "path_info" in completed_plain

def test_tool_cell_wraps_chain_when_narrow() -> None:
    call = {
        "name": "run_python",
        "call_id": "call_123",
        "arguments": '{"code":"from uv_agent_runtime import path_info\\npath_info(\\".\\")\\npath_info(\\".\\")"}',
    }
    payload = {"returncode": 0}
    lines = render_tool_cell(TranscriptCell("tool", call=call, payload=payload), 20)
    plain = [strip_ansi(line) for line in lines]

    assert len(lines) == 2
    assert "✓" in plain[0]
    assert "└─" in plain[1]
    assert "path_info" in plain[1]


def test_tool_cell_wraps_long_chain_across_all_helpers() -> None:
    call = {"name": "run_python", "call_id": "call_123", "arguments": '{"code":"print(1)"}'}
    payload = {
        "returncode": 0,
        "helper_calls": [{"name": f"helper_{index}_name", "args": ""} for index in range(12)],
    }

    lines = render_tool_cell(TranscriptCell("tool", call=call, payload=payload), 32)
    plain = [strip_ansi(line) for line in lines]
    body = "\n".join(plain)

    assert len(plain) > 2
    assert "helper_0_name" in body
    assert "helper_11_name" in body
    assert "…" not in body
    assert plain[1].startswith("  └─ ")
    assert all(line.startswith("     ") for line in plain[2:])
    assert all(visible_len(line) <= 32 for line in plain)


def test_running_tool_cell_wraps_long_chain_with_elapsed_status() -> None:
    call = {"name": "run_python", "call_id": "call_123", "arguments": '{"code":"print(1)"}'}
    payload = {
        "helper_calls": [{"name": f"live_helper_{index}", "args": ""} for index in range(10)],
    }
    cell = TranscriptCell(
        "tool",
        status="running",
        call=call,
        payload=payload,
        created_at=tui2_app.monotonic() - 12.0,
    )

    lines = render_tool_cell(cell, 36)
    plain = [strip_ansi(line) for line in lines]
    body = "\n".join(plain)

    assert "running" in plain[0]
    assert "12" in plain[0]
    assert len(plain) > 2
    assert "live_helper_9" in body
    assert "…" not in body
    assert all(visible_len(line) <= 36 for line in plain)


def test_live_region_keeps_blank_separator_between_cells_and_composer() -> None:
    state = Tui2State(composer="hi")
    state.live.append(TranscriptCell("reasoning", text="thinking"))
    lines, _, _ = render_live_with_cursor(state, 60, 0)
    plain = [strip_ansi(line) for line in lines]

    assert plain[0].startswith("·")
    assert plain[1].strip() == ""
    assert plain[-1].startswith("╰")

def test_live_region_separates_middle_from_assistant_before_composer() -> None:
    state = Tui2State(composer="hi")
    state.live.append(TranscriptCell("reasoning", text="thinking"))
    state.live.append(TranscriptCell("tool", call={"name": "run_python"}, payload={"returncode": 0}))
    state.live.append(TranscriptCell("assistant", text="done"))
    lines, _, _ = render_live_with_cursor(state, 60, 0)
    plain = [strip_ansi(line) for line in lines]

    # Reasoning and tool are compact; tool and assistant are separated; exactly
    # one blank row separates the cell block from the composer.
    assert plain[0].startswith("·")
    assert plain[1].startswith("✓")
    assert plain[2].strip() == ""
    assert plain[3].startswith("✦")
    assert plain[4].strip() == ""
    assert plain[-1].startswith("╰")






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


def test_composer_highlights_active_image_tokens_only() -> None:
    text = "look [Image #1] and [Image #99]"
    lines, _, _ = render_composer_with_cursor(text, 80, image_token_numbers={1})
    rendered = "\n".join(lines)
    plain = strip_ansi(rendered)

    assert "[Image #1]" in plain
    assert "[Image #99]" in plain
    assert sgr(DEFAULT_THEME.image_token, "[Image #1]") in rendered
    assert sgr(DEFAULT_THEME.image_token, "[Image #99]") not in rendered


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


def test_busy_state_renders_token_rate_after_elapsed() -> None:
    state = Tui2State(busy=True, turn_elapsed_s=12.0, turn_token_rate=18.4)

    plain = strip_ansi(render_status_lines(state, 80, 0)[0])

    assert "12s · 18.4 tok/s" in plain


def test_busy_state_renders_frozen_token_rate_muted() -> None:
    state = Tui2State(
        busy=True,
        turn_elapsed_s=12.0,
        turn_token_rate=18.4,
        turn_token_rate_frozen=True,
    )

    rendered = render_status_lines(state, 80, 0)[0]

    assert "12s · 18.4 tok/s" in strip_ansi(rendered)
    assert sgr(DEFAULT_THEME.muted, "18.4 tok/s") in rendered


def test_busy_state_uses_status_message_as_primary_label() -> None:
    state = Tui2State(busy=True, status_message="回复中", turn_elapsed_s=13.0)

    plain = strip_ansi(render_status_lines(state, 80, 1)[0])

    assert "回复中" in plain
    assert "working" not in plain


def test_busy_goal_state_renders_truncated_objective_after_elapsed() -> None:
    state = Tui2State(
        busy=True,
        turn_elapsed_s=12.0,
        goal_enabled=True,
        goal_objective="推进中文目标状态刷新并观察工具完成后的变化",
    )

    line = render_status_lines(state, 120, 0)[0]
    plain = strip_ansi(line)

    assert "Working · 12s · 推进中文目标状态刷新并…" in plain
    assert "推进中文目标状态刷新并观察" not in plain
    assert sgr(DEFAULT_THEME.goal, "推进中文目标状态刷新并…") in line


def test_context_row_includes_goal_model_project_but_not_thread_title() -> None:
    # The thread title is shown in the terminal title and the /status output;
    # the bottom status row keeps only model + Goal + project so the row stays
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

    assert "gpt-5-codex · Goal" in plain
    assert "⊕" not in plain
    assert "ship it" not in plain
    assert "my thread" not in plain
    assert "t-abc" not in plain
    assert "gpt-5-codex" in plain
    assert "/home/user/proj" in plain


def test_context_row_styles_goal_as_orange_red_badge() -> None:
    state = Tui2State(level="test", goal_enabled=True)

    line = render_status_lines(state, 80, 0)[0]

    assert sgr(DEFAULT_THEME.goal, "Goal") in line


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


def test_live_region_keeps_two_status_rows_together_below_cell_gap() -> None:
    state = Tui2State(
        busy=True,
        turn_elapsed_s=3.0,
        composer="hi",
        level="alpha",
        project_path="/home/user/project",
    )
    state.live.append(TranscriptCell("assistant", text="done"))

    lines, _, _ = render_live_with_cursor(state, 80, spinner_frame=1)
    plain = [strip_ansi(line) for line in lines]

    assistant_idx = next(i for i, line in enumerate(plain) if line.startswith("✦ done"))
    activity_idx = next(i for i, line in enumerate(plain) if "Working" in line)
    context_idx = next(i for i, line in enumerate(plain) if line.startswith("◇ "))

    # One blank row separates transcript cells from the status strip...
    assert activity_idx - assistant_idx == 2
    assert plain[assistant_idx + 1].strip() == ""
    # ...but the status strip itself stays contiguous: no row1/row2 gap.
    assert context_idx == activity_idx + 1
    assert plain[context_idx + 1].startswith("╭")


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

    assert rendered.startswith(f"\x1b[?2026h\x1b[?7l\r\x1b[{expected}A\x1b[J")
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
def test_flush_cell_separates_user_from_middle_process_and_turns() -> None:
    output = io.StringIO()
    renderer = Renderer(output=output)
    renderer.flush_cell(TranscriptCell("user", text="hi"))
    renderer.flush_cell(TranscriptCell("reasoning", text="thinking"))
    renderer.flush_cell(TranscriptCell("tool", call={"name": "run_python"}, payload={"returncode": 0}))
    renderer.flush_cell(TranscriptCell("assistant", text="done"))
    renderer.flush_cell(TranscriptCell("user", text="next"))
    rendered = output.getvalue()

    body = rendered[: -len("\x1b[?7h\x1b[?2026l")]
    lines = [strip_ansi(line).replace("\r", "").rstrip() for line in body.split("\r\n")]
    non_empty = [line for line in lines if line.strip()]

    assert non_empty == ["› hi", "· thinking", "✓ run_python", "✦ done", "› next"]
    indices = [lines.index(row) for row in non_empty]
    # User message is separated from the following reasoning/tool chain.
    assert indices[1] - indices[0] == 2
    # Reasoning and tool are compact within the turn.
    assert indices[2] - indices[1] == 1
    # Tool chain and assistant final output are separated.
    assert indices[3] - indices[2] == 2
    # A blank row separates the assistant from the next user turn.
    assert indices[4] - indices[3] == 2


def test_flush_cell_only_uses_crlf_separators() -> None:
    output = io.StringIO()
    Renderer(output=output).flush_cell(TranscriptCell("user", text="a\nb"))
    plain = strip_ansi(output.getvalue()).replace("\x1b[?2026h", "").replace("\x1b[?2026l", "")
    bare_lf = [i for i, ch in enumerate(plain) if ch == "\n" and (i == 0 or plain[i - 1] != "\r")]
    assert bare_lf == []


def test_renderer_clear_screen_clears_scrollback_and_omits_rule_by_default() -> None:
    output = io.StringIO()
    renderer = Renderer(output=output)

    renderer.clear_screen()
    rendered = output.getvalue()

    assert "\x1b[2J\x1b[3J\x1b[H" in rendered
    assert "────────" not in strip_ansi(rendered)


def test_renderer_can_pad_live_region_to_viewport_bottom(monkeypatch) -> None:
    monkeypatch.setattr("uv_agent.tui2.renderer.terminal_size", lambda default=(100, 30): (80, 12))
    output = io.StringIO()
    renderer = Renderer(output=output)

    renderer.pad_live_region_to_bottom(Tui2State(composer=""), preceding_rows=0)

    # A 12-row terminal reserves one row; the empty composer is 3 rows, so the
    # next repaint can start after 8 blank rows and place the composer at bottom.
    assert output.getvalue() == "\r\n" * 8


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


def test_renderer_caps_agent_view_to_terminal_height(monkeypatch) -> None:
    monkeypatch.setattr("uv_agent.tui2.renderer.terminal_size", lambda default=(100, 30): (80, 10))
    state = Tui2State(mode="agent_view")
    state.agent_view.rows = [
        AgentViewRow(thread_id=f"thr_{index}", title=f"Task {index}", status="working")
        for index in range(30)
    ]
    output = io.StringIO()
    renderer = Renderer(output=output)

    renderer.repaint(state)

    # The renderer reserves one terminal row, so the Agent View frame must not
    # write more than 9 physical rows into a 10-row terminal.
    painted = strip_ansi(output.getvalue()).replace("\x1b[?2026h", "").replace("\x1b[?2026l", "")
    assert len([line for line in painted.splitlines() if line]) <= 9


def test_idempotent_repaint_wraps_in_sync_output() -> None:
    output = io.StringIO()
    renderer = Renderer(output=output)
    renderer.repaint(Tui2State(composer=""))
    output.seek(0)
    output.truncate(0)

    renderer.repaint(Tui2State(composer=""))
    rendered = output.getvalue()
    assert "\x1b[?2026h" in rendered and "\x1b[?2026l" in rendered


def test_renderer_disables_autowrap_during_paint() -> None:
    """DECAWM off/on must wrap every frame so terminals never auto-scroll
    when a cell-width estimate is off (e.g. Braille glyphs rendered as 2
    cells on Windows ConPTY). Without this, the ``_frame_cursor_row``
    erase math drifts and leaks stale ``run_python · running`` rules into
    scrollback."""

    output = io.StringIO()
    renderer = Renderer(output=output)

    renderer.repaint(Tui2State(composer="hi"))
    rendered = output.getvalue()
    assert "\x1b[?7l" in rendered
    assert "\x1b[?7h" in rendered
    # Autowrap must be re-enabled before the sync-output region closes so
    # any nested terminal behaviour outside the frame is unaffected.
    assert rendered.rfind("\x1b[?7h") < rendered.rfind("\x1b[?2026l")

    output.seek(0)
    output.truncate(0)
    renderer.flush_cell(TranscriptCell("user", text="hello"))
    flushed = output.getvalue()
    assert "\x1b[?7l" in flushed and "\x1b[?7h" in flushed

    output.seek(0)
    output.truncate(0)
    renderer.close()
    closed = output.getvalue()
    # close() must leave the shell with autowrap re-enabled.
    assert closed.endswith("\x1b[?2026l\x1b[0m")
    assert "\x1b[?7h" in closed


# ---------------------------------------------------------------------------
# Key handling: Ctrl combos, history, Tab completion
# ---------------------------------------------------------------------------


class _DummyEngine:
    def __init__(self) -> None:
        self.turns: list[dict[str, object]] = []
        self.goal_updates: list[dict[str, object]] = []
        self.goal_states: dict[str, SimpleNamespace] = {}
        self.branch_slug = "test-task"
        self.branch_slug_requests: list[dict[str, object]] = []

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
            for thread in cls.threads:
                if thread.get("thread_id") != thread_id:
                    continue
                thread["latest_event_type"] = event_type
                if event_type == "thread.level_updated":
                    thread["active_level"] = data.get("level")
                    thread["active_model"] = data.get("model")
                elif event_type == "thread.agent_view_joined":
                    thread["agent_view_joined"] = True
                    thread["agent_view_source"] = data.get("source")
                    thread.pop("agent_view_deleted", None)
                elif event_type == "thread.worktree_created":
                    thread.update(data)
                elif event_type == "thread.cwd_updated":
                    thread["latest_cwd"] = data.get("cwd")
                elif event_type in {"turn.completed", "turn.error", "turn.interrupted"}:
                    thread["terminal_event_type"] = event_type
                elif event_type == "thread.agent_view_deleted":
                    thread["agent_view_deleted"] = True
                break
            return event

        @classmethod
        def read_recent_events(cls, thread_id, *, limit=1, event_types=None):
            matches = [
                event
                for event in reversed(cls.events)
                if event.get("thread_id") == thread_id
                and (event_types is None or event.get("type") in event_types)
            ]
            return matches[:limit], len(matches) > limit

        @classmethod
        def read_events(cls, thread_id, *, event_types=None):
            return [
                event
                for event in cls.events
                if event.get("thread_id") == thread_id
                and (event_types is None or event.get("type") in event_types)
            ]

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

    async def run_turn(self, *, user_text, thread_id=None, level=None, image_paths=None, cancel_event=None):
        thread_id = thread_id or self.thread_store.create_thread("New thread")
        self.turns.append(
            {
                "user_text": user_text,
                "thread_id": thread_id,
                "level": level,
                "image_paths": list(image_paths or []),
            }
        )
        yield {"type": "turn.started", "thread_id": thread_id, "turn_id": f"turn_{len(self.turns)}"}
        yield {"type": "turn.completed", "thread_id": thread_id, "turn_id": f"turn_{len(self.turns)}"}

    async def generate_branch_slug(self, thread_id, user_text, *, level=None):
        self.branch_slug_requests.append({"thread_id": thread_id, "user_text": user_text, "level": level})
        return self.branch_slug

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
        self.clear_calls: list[str | None] = []
        self.repaint_status_messages: list[str] = []

    def repaint(self, state) -> None:
        self.repaint_status_messages.append(state.status_message)

    def flush_cell(self, cell) -> None:
        self.flushed.append(cell)

    def flush_cells(self, cells) -> None:
        for cell in cells:
            self.flush_cell(cell)

    def clear_screen(self, *, rule=None) -> None:
        self.clear_calls.append(rule)
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


def test_ctrl_c_requires_second_press_to_interrupt_when_busy(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.cancel_event = asyncio.Event()
    app.state.busy = True

    assert asyncio.run(app.handle_key("\x03")) is True
    assert app._interrupt_armed
    assert app.cancel_event is not None and not app.cancel_event.is_set()

    assert asyncio.run(app.handle_key("\x03")) is True
    assert not app._interrupt_armed
    assert app.cancel_event is not None and app.cancel_event.is_set()


def test_ctrl_c_ignores_completed_run_state_after_agent_view_resume(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [{"thread_id": "thr_done", "title": "Done", "agent_view_joined": True}]

    async def run_background_turn() -> None:
        await app._start_turn_for_thread("thr_done", "background")
        task = app._thread_runs["thr_done"].task
        assert task is not None
        await task

    asyncio.run(run_background_turn())
    run_state = app._thread_runs["thr_done"]
    assert not run_state.running
    assert run_state.cancel_event is not None

    app._open_agent_view()
    asyncio.run(app.handle_key("\r"))

    assert app.state.mode == "transcript"
    assert app.state.thread_id == "thr_done"
    assert app.cancel_event is not None

    assert asyncio.run(app.handle_key("\x03")) is True
    assert app._quit_armed
    assert not app._interrupt_armed
    assert asyncio.run(app.handle_key("\x03")) is False


def test_judge_events_update_visible_tui2_status(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "thr_judge"
    app._thread_runs["thr_judge"] = tui2_app.ThreadRunState(thread_id="thr_judge")
    app._thread_runs["thr_judge"].started_at = 1.0

    class PendingTask:
        def done(self):
            return False

    app._thread_runs["thr_judge"].task = PendingTask()  # type: ignore[assignment]

    app._handle_event({"type": "turn.started", "thread_id": "thr_judge", "turn_id": "turn_1"})
    app._handle_event({"type": "judge.started", "thread_id": "thr_judge", "turn_id": "turn_1"})
    app._safe_repaint()

    assert app.state.status_message == app._text("judging")
    assert app.renderer.repaint_status_messages[-1] == app._text("judging")

    app._handle_event({"type": "judge.completed", "thread_id": "thr_judge", "turn_id": "turn_1"})
    app._safe_repaint()

    assert app.state.status_message == app._text("working")
    assert app.renderer.repaint_status_messages[-1] == app._text("working")


def test_ctrl_c_quits_after_current_thread_completed_but_task_is_unwinding(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "thr_current"
    run_state = tui2_app.ThreadRunState(thread_id="thr_current")

    class UnwindingTask:
        def done(self):
            return False

    run_state.task = UnwindingTask()  # type: ignore[assignment]
    run_state.terminal_status = "completed"
    app._thread_runs["thr_current"] = run_state
    app.state.busy = True

    assert asyncio.run(app.handle_key("\x03")) is True
    assert app._quit_armed
    assert not app._interrupt_armed
    assert not run_state.cancel_event.is_set()
    assert app.state.busy is False
    assert app.state.status_message == app._text("quit_again")
    assert asyncio.run(app.handle_key("\x03")) is False


def test_ctrl_c_quits_after_interrupt_requested_for_long_running_turn(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "thr_current"
    run_state = tui2_app.ThreadRunState(thread_id="thr_current")

    class RunningTask:
        def done(self):
            return False

    run_state.task = RunningTask()  # type: ignore[assignment]
    app._thread_runs["thr_current"] = run_state
    app.state.busy = True

    assert asyncio.run(app.handle_key("\x03")) is True
    assert app._interrupt_armed

    assert asyncio.run(app.handle_key("\x03")) is True
    assert run_state.cancel_event.is_set()
    assert run_state.terminal_status == "interrupted"

    assert asyncio.run(app.handle_key("\x03")) is True
    assert app._quit_armed
    assert not app._interrupt_armed
    assert app.state.busy is False
    assert app.state.status_message == app._text("quit_again")
    assert asyncio.run(app.handle_key("\x03")) is False


def test_cancel_command_interrupts_without_confirmation(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.cancel_event = asyncio.Event()
    app.state.busy = True

    assert app._handle_command("/cancel") is True

    assert not app._interrupt_armed
    assert app.cancel_event.is_set()
    assert app.state.status_message == app._text("interrupted")


def test_clear_command_uses_plain_clear_screen_without_separator(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.flushed = [TranscriptCell("user", text="old")]
    app.state.live = [TranscriptCell("assistant", text="streaming", status="streaming")]
    app.state.thread_id = "thr_old"

    assert app._handle_command("/clear") is True

    assert app.renderer.clear_calls == [None]
    assert "─" not in app.renderer.output.getvalue()
    assert app.state.flushed[-1].kind == "event"
    assert "cleared view" in app.state.flushed[-1].text


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


def test_left_angle_bracket_inserts_as_text(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    asyncio.run(app.handle_key("<"))

    assert app.state.composer == "<"
    assert app.state.composer_cursor == 1


def test_left_angle_bracket_in_agent_view_composer(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._open_agent_view()
    app._enter_agent_view_input_mode(target="dispatch")

    asyncio.run(app.handle_key("<"))

    assert app.state.agent_view.composer == "<"
    assert app.state.agent_view.composer_cursor == 1


def test_wrapped_key_tokens_are_not_inserted_as_text(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "abc"

    asyncio.run(app.handle_key("<V>"))

    assert app.state.composer == "abc"
    assert app.state.composer_cursor is None


def test_image_token_deletes_as_single_unit_with_backspace(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "look [Image #1] now"
    app.state.composer_cursor = len("look [Image #1]")

    asyncio.run(app.handle_key("\b"))

    assert app.state.composer == "look now"
    assert app.state.composer_cursor == len("look ")


def test_image_token_deletes_as_single_unit_with_delete(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "look [Image #1] now"
    app.state.composer_cursor = len("look ")

    asyncio.run(app.handle_key("\x04"))

    assert app.state.composer == "look now"
    assert app.state.composer_cursor == len("look ")


def test_image_token_deletion_trims_edge_separator(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "prefix [Image #1]"
    app.state.composer_cursor = len(app.state.composer)

    asyncio.run(app.handle_key("\b"))

    assert app.state.composer == "prefix"
    assert app.state.composer_cursor == len("prefix")


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


def test_command_palette_lists_bg_command() -> None:
    values = [item.value for item in TOP_LEVEL_COMMANDS]

    assert "/bg" in values


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


def test_assistant_delta_queues_smooth_display(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "T-test"
    app._thread_runs["T-test"] = ThreadRunState(thread_id="T-test")

    app._handle_event({"type": "assistant.delta", "text": "chunk"})

    assert app._assistant_cell is not None
    assert app._assistant_cell.text == ""
    assert app._thread_runs["T-test"].assistant_display_queue == "chunk"


def test_streaming_display_tick_drains_queued_assistant_text(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "T-test"
    run_state = ThreadRunState(thread_id="T-test")
    app._thread_runs["T-test"] = run_state
    app._handle_event({"type": "assistant.delta", "text": "chunk"})
    run_state.last_animation_tick_at = tui2_app.monotonic() - 1.0

    app._advance_streaming_display()

    assert app._assistant_cell is not None
    assert app._assistant_cell.text == "chunk"
    assert run_state.assistant_display_queue == ""


def test_model_response_usage_updates_thread_token_rate(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "T-test"
    run_state = ThreadRunState(thread_id="T-test")
    app._thread_runs["T-test"] = run_state
    now = tui2_app.monotonic()
    run_state.rate_estimator.observe("ab", now=now - 1.0)
    run_state.rate_estimator.observe("cd", now=now)
    response = SimpleNamespace(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "abcd"}]}],
        usage={"output_tokens": 2},
    )

    app._handle_event({"type": "model.response", "thread_id": "T-test", "response": response})

    assert run_state.token_ratio.visible_units == 1
    assert run_state.token_ratio.output_tokens == 2
    assert app._current_token_rate(run_state) is not None


def test_token_rate_display_smoothing_throttles_row1_updates(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    run_state = ThreadRunState(thread_id="T-test")
    instant_token_rate = 10.0

    def current_token_rate(_run_state: ThreadRunState, *, now: float | None = None) -> float:
        return instant_token_rate

    monkeypatch.setattr(app, "_current_token_rate", current_token_rate)

    assert app._display_token_rate(run_state, now=0.0) == 10.0

    instant_token_rate = 50.0
    assert app._display_token_rate(run_state, now=0.2) == 10.0

    displayed = app._display_token_rate(run_state, now=0.5)
    expected = 10.0 + (50.0 - 10.0) * (1.0 - math.exp(-0.5 / tui2_app.TOKEN_RATE_DISPLAY_TAU_S))
    assert displayed is not None
    assert math.isclose(displayed, expected)
    assert 10.0 < displayed < 50.0


def test_repaint_sync_restores_activity_row_for_active_run_state(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "T-test"
    run_state = ThreadRunState(thread_id="T-test")
    run_state.started_at = tui2_app.monotonic() - 3.0

    class RunningTask:
        def done(self):
            return False

    run_state.task = RunningTask()  # type: ignore[assignment]
    app._thread_runs["T-test"] = run_state
    app.state.busy = False
    app.state.status_message = "ready"

    app._sync_attached_run_state_for_repaint()

    assert app.state.busy
    plain = "\n".join(strip_ansi(line) for line in render_status_lines(app.state, 80, 0))
    assert app._text("working") in plain
    assert "3s" in plain


def test_token_rate_freezes_for_tool_execution_and_resumes_on_stream(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "T-test"
    run_state = ThreadRunState(thread_id="T-test")
    run_state.displayed_token_rate = 12.0
    run_state.last_token_rate_display_update_at = 0.0

    class RunningTask:
        def done(self):
            return False

    run_state.task = RunningTask()  # type: ignore[assignment]
    app._thread_runs["T-test"] = run_state

    app._handle_event({"type": "tool.started", "thread_id": "T-test", "call": {"call_id": "call_1"}})

    assert run_state.token_rate_frozen
    assert run_state.frozen_token_rate == 12.0
    assert app._display_token_rate(run_state, now=10.0) == 12.0

    app._sync_attached_run_state(run_state)
    assert app.state.turn_token_rate == 12.0
    assert app.state.turn_token_rate_frozen

    app._handle_event({"type": "assistant.delta", "thread_id": "T-test", "text": "next"})

    assert not run_state.token_rate_frozen
    app._sync_attached_run_state(run_state)
    assert not app.state.turn_token_rate_frozen


def test_final_model_response_holds_token_rate_while_text_drains(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "T-test"
    run_state = ThreadRunState(thread_id="T-test")
    run_state.displayed_token_rate = 12.0
    run_state.last_token_rate_display_update_at = 0.0
    app._thread_runs["T-test"] = run_state
    response = SimpleNamespace(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "chunk"}]}],
        usage={"output_tokens": 2},
    )

    app._handle_event({"type": "assistant.delta", "thread_id": "T-test", "text": "chunk"})
    app._handle_event({"type": "model.response", "thread_id": "T-test", "response": response})

    assert run_state.token_rate_held
    assert run_state.held_token_rate == 12.0
    assert not run_state.token_rate_frozen
    assert run_state.display_pending

    def fail_current_token_rate(_run_state: ThreadRunState, *, now: float | None = None) -> float:
        raise AssertionError("held token rate should not be recalculated")

    monkeypatch.setattr(app, "_current_token_rate", fail_current_token_rate)
    assert app._display_token_rate(run_state, now=10.0) == 12.0
    app._sync_attached_run_state(run_state)
    assert app.state.turn_token_rate == 12.0
    assert not app.state.turn_token_rate_frozen


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


def test_flushed_tool_cells_retain_only_lightweight_payload(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    call = {"name": "run_python", "call_id": "call_1", "arguments": '{"code":"print(1)"}'}
    stdout = "x" * 5000

    app._handle_event({"type": "tool.started", "call": call})
    app._handle_event({
        "type": "tool.output",
        "call": call,
        "output": {
            "output": json.dumps(
                {
                    "run_id": "run_123",
                    "returncode": 0,
                    "stdout": stdout,
                    "stderr": "err",
                    "events": [{"big": stdout}],
                    "helper_calls": [{"name": "run_process_text", "args": "[]"}],
                }
            )
        },
    })

    rendered = app.renderer.flushed[-1]
    retained = app.state.flushed[-1]

    assert rendered.payload["stdout"] == stdout
    assert retained.payload == {
        "run_id": "run_123",
        "returncode": 0,
        "helper_calls": [{"name": "run_process_text", "args": "[]"}],
    }
    assert retained.call == {"name": "run_python", "call_id": "call_1"}


def test_flushed_cells_are_bounded(monkeypatch) -> None:
    from uv_agent.tui2.app import TUI2_FLUSHED_CELLS_MAX

    app = _make_app(monkeypatch)

    for index in range(TUI2_FLUSHED_CELLS_MAX + 3):
        app._flush(TranscriptCell("event", text=str(index)))

    assert len(app.state.flushed) == TUI2_FLUSHED_CELLS_MAX
    assert app.state.flushed[0].text == "3"
    assert app.state.flushed[-1].text == str(TUI2_FLUSHED_CELLS_MAX + 2)


def test_turn_completed_plays_terminal_buzzer(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr("uv_agent.tui2.app.play_terminal_buzzer", lambda: calls.append("buzzer") or True)

    app._handle_event({"type": "turn.completed"})

    assert calls == ["buzzer"]


def test_turn_completed_delays_buzzer_until_streaming_display_drains(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "T-test"
    run_state = ThreadRunState(thread_id="T-test")
    app._thread_runs["T-test"] = run_state
    calls: list[str] = []
    monkeypatch.setattr("uv_agent.tui2.app.play_terminal_buzzer", lambda: calls.append("buzzer") or True)

    app._handle_event({"type": "assistant.delta", "thread_id": "T-test", "text": "x"})
    app._handle_event({"type": "turn.completed", "thread_id": "T-test"})

    assert calls == []
    assert run_state.completion_notification_pending

    run_state.engine_finished = True
    run_state.assistant_display_credit = 1.0
    run_state.last_animation_tick_at = tui2_app.monotonic() - 1.0
    app._advance_streaming_display()

    assert calls == ["buzzer"]
    assert not run_state.completion_notification_pending


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



def test_agent_view_renderer_groups_rows_and_shows_peek() -> None:
    state = Tui2State(mode="agent_view")
    state.agent_view.rows = [
        AgentViewRow(
            thread_id="thr_working",
            title="Fix login redirect",
            status="working",
            summary="running tests",
            worktree_branch="agent-fix-login-abc12345",
            elapsed_seconds=12,
        ),
        AgentViewRow(
            thread_id="thr_done",
            title="Bump deps",
            status="completed",
            summary="updated lockfile",
        ),
    ]

    plain = "\n".join(strip_ansi(line) for line in render_agent_view(state, 88, 0))

    assert "Agent View" in plain
    assert "WORKING (1)" in plain
    assert "COMPLETED (1)" in plain
    assert "Fix login redirect" in plain
    assert "agent-fix-login-abc12345" in plain
    assert "peek:" in plain
    assert "running tests" in plain


def test_agent_view_renderer_shows_help_mode() -> None:
    state = Tui2State(mode="agent_view")
    state.agent_view.interaction_mode = "help"

    plain = "\n".join(strip_ansi(line) for line in render_agent_view(state, 88, 0))

    assert "HELP" in plain
    assert "Normal mode keys" in plain
    assert "Input mode keys" in plain


def test_agent_view_renderer_uses_chinese_labels() -> None:
    from uv_agent.environment import normalize_language

    state = Tui2State(mode="agent_view", language=normalize_language("zh-CN"))
    state.agent_view.rows = [AgentViewRow(thread_id="thr_done", title="完成", status="completed")]

    plain = "\n".join(strip_ansi(line) for line in render_agent_view(state, 88, 0))

    assert "普通" in plain
    assert "已完成 (1)" in plain


def test_agent_view_renderer_shows_dispatch_model_level() -> None:
    state = Tui2State(mode="agent_view")
    state.agent_view.dispatch_level = "alpha"

    plain = "\n".join(strip_ansi(line) for line in render_agent_view(state, 88, 0))

    assert "new task model: alpha" in plain


def test_agent_view_renderer_shows_model_picker() -> None:
    state = Tui2State(mode="agent_view")
    state.agent_view.interaction_mode = "model"
    state.agent_view.dispatch_level = "alpha"
    state.agent_view.model_options = [
        CommandSuggestion("alpha", "alpha-model", id="alpha"),
        CommandSuggestion("test", "test-model", id="test"),
    ]
    state.agent_view.model_selected = 1

    plain = "\n".join(strip_ansi(line) for line in render_agent_view(state, 88, 0))

    assert "MODEL" in plain
    assert "Choose the model level" in plain
    assert "alpha — alpha-model" in plain
    assert "test — test-model" in plain


def test_agent_view_renderer_distinguishes_hide_confirmation() -> None:
    state = Tui2State(mode="agent_view")
    state.agent_view.rows = [
        AgentViewRow(
            thread_id="thr_keep",
            title="Keep workspace",
            status="completed",
            worktree_branch="agent-keep-workspace",
        )
    ]
    state.agent_view.pending_confirmation = "hide_thread:thr_keep"

    plain = "\n".join(strip_ansi(line) for line in render_agent_view(state, 88, 0))

    assert "HIDE from Agent View only" in plain
    assert "Keeps transcript, worktree, and branch" in plain
    assert "delete_thread:thr_keep" not in plain
    assert "delete_worktree:thr_keep" not in plain


def test_agent_view_renderer_distinguishes_worktree_delete_confirmation() -> None:
    state = Tui2State(mode="agent_view")
    state.agent_view.rows = [
        AgentViewRow(
            thread_id="thr_delete",
            title="Delete workspace",
            status="completed",
            worktree_branch="agent-delete-workspace",
            worktree_path="/tmp/agent-delete-workspace",
        )
    ]
    state.agent_view.pending_confirmation = "delete_worktree:thr_delete"

    plain = "\n".join(strip_ansi(line) for line in render_agent_view(state, 88, 0))

    assert "DELETE WORKTREE + branch" in plain
    assert "Destructive: removes the worktree directory and local branch" in plain
    assert "agent-delete-workspace" in plain
    assert "delete_worktree:thr_delete" not in plain


def test_agent_view_renderer_respects_max_height_with_many_rows() -> None:
    state = Tui2State(mode="agent_view")
    state.agent_view.rows = [
        AgentViewRow(
            thread_id=f"thr_{index}",
            title=f"Task {index}",
            status="working" if index % 2 else "completed",
            summary="summary",
            worktree_branch=f"agent-task-{index}",
        )
        for index in range(24)
    ]
    state.agent_view.selected = 15

    lines, cursor_row, _ = render_agent_view_with_cursor(state, 88, 0, max_height=12)
    plain = "\n".join(strip_ansi(line) for line in lines)

    assert len(lines) <= 12
    assert 0 <= cursor_row < len(lines)
    assert "rows hidden" in plain
    assert "Task 15" in plain


def test_agent_view_renderer_accounts_for_multiline_composer_height() -> None:
    state = Tui2State(mode="agent_view")
    state.agent_view.rows = [
        AgentViewRow(thread_id=f"thr_{index}", title=f"Task {index}", status="completed")
        for index in range(12)
    ]
    state.agent_view.composer = "\n".join(f"line {index}" for index in range(6))
    state.agent_view.composer_cursor = len(state.agent_view.composer)

    lines, cursor_row, _ = render_agent_view_with_cursor(state, 60, 0, max_height=10)

    assert len(lines) <= 10
    assert 0 <= cursor_row < len(lines)


def test_agent_view_renderer_has_compact_layout_for_tiny_viewports() -> None:
    state = Tui2State(mode="agent_view")
    state.agent_view.composer = "tiny terminal prompt"

    lines, cursor_row, _ = render_agent_view_with_cursor(state, 40, 0, max_height=3)

    assert len(lines) <= 3
    assert 0 <= cursor_row < len(lines)



def test_ctrl_a_opens_agent_view_from_empty_composer(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    assert asyncio.run(app.handle_key("\x01")) is True

    assert app.state.mode == "agent_view"


def test_agents_command_opens_agent_view(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [
        {"thread_id": "thr_1", "title": "Alpha", "last_text": "hello", "agent_view_joined": True},
    ]

    app._handle_command("/agents")

    assert app.state.mode == "agent_view"
    assert app.state.agent_view.rows[0].thread_id == "thr_1"


def test_agents_command_does_not_join_current_thread(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [{"thread_id": "thr_plain", "title": "Plain"}]
    app.state.thread_id = "thr_plain"

    app._handle_command("/agents")

    assert "agent_view_joined" not in app.engine.thread_store.threads[0]
    assert app.state.mode == "agent_view"
    assert app.state.agent_view.rows == []


def test_agent_view_omits_ordinary_threads_until_joined(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [
        {"thread_id": "thr_plain", "title": "Plain"},
        {"thread_id": "thr_joined", "title": "Joined", "agent_view_joined": True},
        {"thread_id": "thr_worktree", "title": "Worktree", "worktree_branch": "agent-work"},
    ]

    app._open_agent_view()

    assert [row.thread_id for row in app.state.agent_view.rows] == ["thr_joined", "thr_worktree"]


def test_bg_command_joins_current_thread_to_agent_view(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [{"thread_id": "thr_plain", "title": "Plain"}]
    app.state.thread_id = "thr_plain"

    app._handle_command("/bg")

    assert app.engine.thread_store.threads[0]["agent_view_joined"] is True
    assert app.engine.thread_store.threads[0]["agent_view_source"] == "bg_command"
    assert [row.thread_id for row in app.state.agent_view.rows] == ["thr_plain"]


def test_bg_command_selects_running_current_thread(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [
        {"thread_id": "thr_other", "title": "Other", "agent_view_joined": True},
        {"thread_id": "thr_current", "title": "Current"},
    ]
    app.state.thread_id = "thr_current"
    run_state = tui2_app.ThreadRunState(thread_id="thr_current")

    class RunningTask:
        def done(self):
            return False

    run_state.task = RunningTask()  # type: ignore[assignment]
    app._thread_runs["thr_current"] = run_state

    app._handle_command("/bg")

    assert app.state.mode == "agent_view"
    assert run_state.task is not None and not run_state.task.done()
    assert app.state.agent_view.selected_row().thread_id == "thr_current"
    assert app.state.agent_view.selected_row().status == "working"


def test_agent_view_selection_order_is_stable_across_status_changes(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [
        {"thread_id": "thr_completed", "title": "Done", "agent_view_joined": True},
        {"thread_id": "thr_interrupted", "title": "Stopped", "agent_view_joined": True},
    ]
    app.engine.thread_store.events = [
        {"type": "turn.completed", "thread_id": "thr_completed"},
        {"type": "turn.interrupted", "thread_id": "thr_interrupted"},
    ]
    app._open_agent_view()

    assert [row.thread_id for row in app.state.agent_view.rows] == ["thr_interrupted", "thr_completed"]
    app.state.agent_view.selected = 1
    app._refresh_agent_view_rows()

    assert [row.thread_id for row in app.state.agent_view.rows] == ["thr_interrupted", "thr_completed"]
    assert app.state.agent_view.selected_row().thread_id == "thr_completed"


def test_agent_view_navigation_and_attach(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [
        {"thread_id": "thr_1", "title": "Alpha", "agent_view_joined": True},
        {"thread_id": "thr_2", "title": "Beta", "agent_view_joined": True},
    ]
    app._open_agent_view()

    asyncio.run(app.handle_key("j"))
    assert app.state.agent_view.selected == 1
    asyncio.run(app.handle_key(" "))
    assert not app.state.agent_view.peek_expanded
    asyncio.run(app.handle_key("\r"))

    assert app.state.mode == "transcript"
    assert app.state.thread_id == "thr_2"
    assert app.state.title == "Beta"



def test_agent_view_dispatch_creates_worktree_thread_and_runs(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._open_agent_view()
    created = SimpleNamespace(
        branch="agent-test-task-1",
        path=app.project_root / ".uv-agent" / "worktrees" / "agent-test-task-1",
        origin_root=app.project_root,
        metadata=lambda: {
            "worktree_status": "active",
            "worktree_branch": "agent-test-task-1",
            "worktree_path": str(app.project_root / ".uv-agent" / "worktrees" / "agent-test-task-1"),
            "worktree_base_ref": "HEAD",
            "worktree_origin_root": str(app.project_root),
        },
    )
    monkeypatch.setattr(tui2_app, "create_worktree", lambda project_root, branch, *, run: created)

    async def run() -> None:
        await app._dispatch_agent_view_prompt("fix login")
        task = app._thread_runs.get("thr_1").task
        assert task is not None
        await task

    asyncio.run(run())

    assert app.engine.thread_store.threads[0]["title"] == "fix login"
    assert app.engine.thread_store.threads[0]["worktree_branch"] == "agent-test-task-1"
    assert app.engine.thread_store.threads[0]["latest_cwd"] == str(created.path)
    assert app.engine.turns[-1]["thread_id"] == "thr_1"
    assert app.engine.turns[-1]["user_text"] == "fix login"


def test_agent_view_model_picker_sets_dispatch_level(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._open_agent_view()

    asyncio.run(app.handle_key("m"))
    assert app.state.agent_view.interaction_mode == "model"
    assert [item.id for item in app.state.agent_view.model_options] == ["alpha", "test"]

    asyncio.run(app.handle_key("k"))
    asyncio.run(app.handle_key("\r"))

    assert app.state.agent_view.interaction_mode == "normal"
    assert app.state.agent_view.dispatch_level == "alpha"
    assert app.state.level == "test"


def test_agent_view_dispatch_uses_selected_model_level(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._open_agent_view()
    app.state.agent_view.dispatch_level = "alpha"
    app.state.agent_view.dispatch_level_explicit = True
    created = SimpleNamespace(
        branch="agent-test-task-1",
        path=app.project_root / ".uv-agent" / "worktrees" / "agent-test-task-1",
        origin_root=app.project_root,
        metadata=lambda: {
            "worktree_status": "active",
            "worktree_branch": "agent-test-task-1",
            "worktree_path": str(app.project_root / ".uv-agent" / "worktrees" / "agent-test-task-1"),
            "worktree_base_ref": "HEAD",
            "worktree_origin_root": str(app.project_root),
        },
    )
    monkeypatch.setattr(tui2_app, "create_worktree", lambda project_root, branch, *, run: created)

    async def run() -> None:
        await app._dispatch_agent_view_prompt("fix login")
        task = app._thread_runs.get("thr_1").task
        assert task is not None
        await task

    asyncio.run(run())

    assert app.engine.turns[-1]["level"] == "alpha"
    assert app.engine.thread_store.threads[0]["active_level"] == "alpha"
    assert app.engine.branch_slug_requests[-1]["level"] == "alpha"


def test_agent_view_default_dispatch_level_tracks_current_thread(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.level = "alpha"

    app._open_agent_view()

    assert app.state.agent_view.dispatch_level == "alpha"
    assert app._agent_view_dispatch_level() == "alpha"


def test_agent_view_input_mode_dispatches_on_enter(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._open_agent_view()
    created = SimpleNamespace(
        branch="agent-test-task-1",
        path=app.project_root / ".uv-agent" / "worktrees" / "agent-test-task-1",
        origin_root=app.project_root,
        metadata=lambda: {
            "worktree_status": "active",
            "worktree_branch": "agent-test-task-1",
            "worktree_path": str(app.project_root / ".uv-agent" / "worktrees" / "agent-test-task-1"),
            "worktree_base_ref": "HEAD",
            "worktree_origin_root": str(app.project_root),
        },
    )
    monkeypatch.setattr(tui2_app, "create_worktree", lambda project_root, branch, *, run: created)

    async def run() -> None:
        await app.handle_key("i")
        await app.handle_key("f")
        await app.handle_key("i")
        await app.handle_key("x")
        await app.handle_key("\r")
        for _ in range(5):
            await asyncio.sleep(0)
        task = app._thread_runs.get("thr_1").task
        assert task is not None
        await task

    asyncio.run(run())

    assert app.state.agent_view.interaction_mode == "normal"
    assert app.state.agent_view.composer == ""
    assert app.engine.turns[-1]["user_text"] == "fix"


def test_agent_view_branch_name_falls_back_to_thread_id(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.branch_slug = ""

    branch = asyncio.run(app._agent_view_branch_name("thr_abcdef123456", "prompt"))

    assert branch == "agent-abcdef12"


def test_agent_view_branch_name_uses_generated_slug(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.branch_slug = "fix-login"

    branch = asyncio.run(app._agent_view_branch_name("thr_abcdef123456", "prompt", level="alpha"))

    assert branch == "agent-fix-login-abcdef12"
    assert app.engine.branch_slug_requests[-1] == {
        "thread_id": "thr_abcdef123456",
        "user_text": "prompt",
        "level": "alpha",
    }


def test_agent_view_branch_name_waits_for_engine_slug(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def generate_branch_slug(thread_id, user_text, *, level=None):
            started.set()
            await release.wait()
            return "delayed-task"

        app.engine.generate_branch_slug = generate_branch_slug
        task = asyncio.create_task(app._agent_view_branch_name("thr_abcdef123456", "prompt"))
        await started.wait()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        assert await task == "agent-delayed-task-abcdef12"

    asyncio.run(run())


def test_agent_view_reply_queues_for_running_thread(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [{"thread_id": "thr_1", "title": "One", "agent_view_joined": True}]
    app._open_agent_view()
    run_state = tui2_app.ThreadRunState(thread_id="thr_1")

    class RunningTask:
        def done(self):
            return False

    run_state.task = RunningTask()  # type: ignore[assignment]
    app._thread_runs["thr_1"] = run_state
    asyncio.run(app.handle_key("r"))
    assert app.state.agent_view.interaction_mode == "input"
    asyncio.run(app.handle_key("f"))
    asyncio.run(app.handle_key("o"))
    asyncio.run(app.handle_key("l"))
    asyncio.run(app.handle_key("l"))
    asyncio.run(app.handle_key("o"))
    asyncio.run(app.handle_key("w"))
    asyncio.run(app.handle_key(" "))
    asyncio.run(app.handle_key("u"))
    asyncio.run(app.handle_key("p"))
    asyncio.run(app.handle_key("\r"))

    assert [turn.text for turn in run_state.pending_turns] == ["follow up"]
    assert app.state.agent_view.composer == ""
    assert app.state.agent_view.interaction_mode == "normal"


def test_agent_view_input_ctrl_enter_inserts_newline(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._open_agent_view()

    asyncio.run(app.handle_key("i"))
    asyncio.run(app.handle_key("a"))
    asyncio.run(app.handle_key("<C-ENTER>"))
    asyncio.run(app.handle_key("b"))

    assert app.state.agent_view.interaction_mode == "input"
    assert app.state.agent_view.composer == "a\nb"


def test_agent_view_ctrl_c_cancels_selected_running_thread(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [{"thread_id": "thr_1", "title": "One", "agent_view_joined": True}]
    app._open_agent_view()
    run_state = tui2_app.ThreadRunState(thread_id="thr_1")

    class RunningTask:
        def done(self):
            return False

    run_state.task = RunningTask()  # type: ignore[assignment]
    app._thread_runs["thr_1"] = run_state

    asyncio.run(app.handle_key("\x03"))

    assert run_state.cancel_event.is_set()
    assert app.state.mode == "agent_view"


def test_agent_view_delete_hides_thread_after_confirmation(monkeypatch) -> None:
    from uv_agent.environment import normalize_language

    app = _make_app(monkeypatch)
    app.language = app.state.language = normalize_language("en")
    app.engine.thread_store.threads = [{"thread_id": "thr_1", "title": "One", "agent_view_joined": True}]
    app._open_agent_view()

    asyncio.run(app.handle_key("d"))
    assert app.state.agent_view.pending_confirmation == "hide_thread:thr_1"
    assert "HIDE" in app.state.agent_view.status_message
    asyncio.run(app.handle_key("y"))

    assert app.engine.thread_store.threads[0]["agent_view_deleted"] is True
    assert app.state.agent_view.rows == []


def test_agent_view_delete_worktree_requires_worktree_metadata(monkeypatch) -> None:
    from uv_agent.environment import normalize_language

    app = _make_app(monkeypatch)
    app.language = app.state.language = normalize_language("en")
    app.engine.thread_store.threads = [{"thread_id": "thr_1", "title": "One", "agent_view_joined": True}]
    app._open_agent_view()

    asyncio.run(app.handle_key("D"))

    assert app.state.agent_view.pending_confirmation is None
    assert app.state.agent_view.status_message == "selected session has no worktree"


def test_agent_view_delete_locked_thread_reports_status(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [{"thread_id": "thr_1", "title": "One", "agent_view_joined": True}]
    app._open_agent_view()

    def locked_append(thread_id, event_type, **data):
        raise ThreadLockedError(thread_id, app.project_root / "state.sqlite3")

    monkeypatch.setattr(app.engine.thread_store, "append", locked_append)

    asyncio.run(app.handle_key("d"))
    asyncio.run(app.handle_key("y"))

    assert app.state.mode == "agent_view"
    assert app.state.agent_view.rows[0].thread_id == "thr_1"
    assert app.state.agent_view.status_message
    assert "thr_1" in app.state.agent_view.status_message


def test_agent_view_delete_worktree_records_metadata(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    path = app.project_root / ".uv-agent" / "worktrees" / "agent-test-task-1"
    app.engine.thread_store.threads = [
        {
            "thread_id": "thr_1",
            "title": "One",
            "worktree_branch": "agent-test-task-1",
            "worktree_path": str(path),
            "worktree_origin_root": str(app.project_root),
        }
    ]
    app._open_agent_view()
    result = SimpleNamespace(
        branch="agent-test-task-1",
        path=path,
        origin_root=app.project_root,
        head="abc123",
        status=" M file.py",
        worktree_removed=True,
        branch_deleted=True,
    )
    monkeypatch.setattr(tui2_app, "cleanup_worktree", lambda project_root, branch, path, *, run: result)

    async def run() -> None:
        await app.handle_key("D")
        await app.handle_key("y")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(run())

    events = app.engine.thread_store.events
    assert any(event["type"] == "thread.worktree_deleted" for event in events)
    assert app.engine.thread_store.threads[0]["latest_cwd"] == str(app.project_root.resolve())


def test_agent_view_composer_is_separate_from_transcript(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.composer = "transcript draft"
    app._open_agent_view()

    asyncio.run(app.handle_key("i"))
    asyncio.run(app.handle_key("a"))
    asyncio.run(app.handle_key("b"))

    assert app.state.composer == "transcript draft"
    assert app.state.agent_view.composer == "ab"


def test_agent_view_normal_mode_letters_do_not_edit_composer(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._open_agent_view()

    asyncio.run(app.handle_key("x"))

    assert app.state.agent_view.composer == ""


def test_agent_view_input_escape_returns_to_normal_without_cursor_jumps(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app._open_agent_view()

    asyncio.run(app.handle_key("i"))
    asyncio.run(app.handle_key("a"))
    asyncio.run(app.handle_key("b"))
    asyncio.run(app.handle_key("\x1b"))
    asyncio.run(app.handle_key("j"))

    assert app.state.agent_view.interaction_mode == "normal"
    assert app.state.agent_view.composer == "ab"


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

def test_start_turn_keeps_user_cell_in_live_until_flushed(monkeypatch) -> None:
    app = _make_app(monkeypatch)

    async def run_turn() -> None:
        await app._start_turn("hello")
        assert app._user_cell is not None
        assert app._user_cell.kind == "user"
        assert app._user_cell in app.state.live
        assert app._run_state().user_cell is app._user_cell
        assert not app.state.flushed
        # Simulate the model starting to respond; the user cell should be flushed
        # before the reasoning cell so scrollback order matches the live region.
        app.state.live.append(TranscriptCell("reasoning", text="thinking"))
        app._flush(app.state.live[-1])
        assert app._user_cell is None
        assert any(cell.kind == "user" for cell in app.state.flushed)

    asyncio.run(run_turn())



def test_resume_thread_restores_live_user_cell_before_first_response(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.engine.thread_store.threads = [{"thread_id": "thr_active", "title": "Active"}]
    run_state = ThreadRunState(thread_id="thr_active")
    run_state.user_cell = TranscriptCell("user", text="hello")

    class RunningTask:
        def done(self):
            return False

    run_state.task = RunningTask()  # type: ignore[assignment]
    app._thread_runs["thr_active"] = run_state

    app._resume_thread("thr_active")

    assert app._user_cell is run_state.user_cell
    assert app.state.live[:1] == [run_state.user_cell]
    plain_lines = [strip_ansi(line) for line in render_live_with_cursor(app.state, 80, 0)[0]]
    user_idx = next(i for i, line in enumerate(plain_lines) if line.startswith("› hello"))
    activity_idx = next(i for i, line in enumerate(plain_lines) if app._text("working") in line)
    assert activity_idx - user_idx == 2
    assert plain_lines[user_idx + 1].strip() == ""


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


def test_image_command_attaches_clipboard_image_token(monkeypatch, tmp_path) -> None:
    app = _make_app(monkeypatch)
    image = tmp_path / "clip.png"
    image.write_bytes(b"fake-png")
    monkeypatch.setattr(
        "uv_agent.tui2.app.save_clipboard_image",
        lambda target_dir: SimpleNamespace(path=image, width=20, height=10),
    )

    assert app._handle_command("/image") is True

    assert app.state.composer == "[Image #1]"
    assert app._image_paths_by_number == {1: image}
    assert app.state.image_token_numbers == {1}
    assert "[Image #1]" in app.state.status_message


def test_image_status_clears_after_token_deleted(monkeypatch, tmp_path) -> None:
    app = _make_app(monkeypatch)
    image = tmp_path / "clip.png"
    image.write_bytes(b"fake-png")
    monkeypatch.setattr(
        "uv_agent.tui2.app.save_clipboard_image",
        lambda target_dir: SimpleNamespace(path=image, width=20, height=10),
    )
    app._handle_command("/image")
    assert "[Image #1]" in app.state.status_message

    asyncio.run(app.handle_key("\b"))

    assert app.state.composer == ""
    assert app.state.status_message == "ready"
    assert app._image_paths_by_number == {}
    assert app.state.image_token_numbers == set()


def test_removed_image_token_is_not_attached_if_retyped(monkeypatch, tmp_path) -> None:
    app = _make_app(monkeypatch)
    image = tmp_path / "clip.png"
    image.write_bytes(b"fake-png")
    monkeypatch.setattr(
        "uv_agent.tui2.app.save_clipboard_image",
        lambda target_dir: SimpleNamespace(path=image, width=20, height=10),
    )
    app._handle_command("/image")
    asyncio.run(app.handle_key("\b"))
    app.state.composer = "literal [Image #1]"

    async def run() -> None:
        await app.submit()
        assert app._running_task is not None
        await app._running_task

    asyncio.run(run())

    assert app.engine.turns[-1]["user_text"] == "literal [Image #1]"
    assert app.engine.turns[-1]["image_paths"] == []


def test_image_tokens_send_matching_paths_once(monkeypatch, tmp_path) -> None:
    app = _make_app(monkeypatch)
    image = tmp_path / "clip.png"
    image.write_bytes(b"fake-png")
    app._image_paths_by_number[1] = image
    app.state.composer = "look [Image #1] and again [Image #1] [Image #99]"

    async def run() -> None:
        await app.submit()
        assert app._running_task is not None
        await app._running_task

    asyncio.run(run())

    assert app.engine.turns[-1]["user_text"] == "look [Image #1] and again [Image #1] [Image #99]"
    assert app.engine.turns[-1]["image_paths"] == [image]
    assert app._image_paths_by_number == {}
    assert app.state.image_token_numbers == set()


def test_image_only_message_uses_default_prompt(monkeypatch, tmp_path) -> None:
    app = _make_app(monkeypatch)
    image = tmp_path / "clip.png"
    image.write_bytes(b"fake-png")
    app._image_paths_by_number[1] = image
    app.state.composer = "[Image #1]"

    async def run() -> None:
        await app.submit()
        assert app._running_task is not None
        await app._running_task

    asyncio.run(run())

    assert app.engine.turns[-1]["user_text"] == app._text("image_only_prompt")
    assert app.engine.turns[-1]["image_paths"] == [image]


def test_queued_turn_captures_image_paths_at_submit_time(monkeypatch, tmp_path) -> None:
    app = _make_app(monkeypatch)
    image = tmp_path / "clip.png"
    image.write_bytes(b"fake-png")
    app._image_paths_by_number[1] = image
    app.state.composer = "queued [Image #1]"

    async def run() -> None:
        app._running_task = asyncio.create_task(asyncio.sleep(0.01))
        await app.submit()
        await app._running_task

    asyncio.run(run())

    assert len(app.state.pending_turns) == 1
    queued = app.state.pending_turns[0]
    assert queued.text == "queued [Image #1]"
    assert queued.image_paths == [image]
    assert app._image_paths_by_number == {}
    assert app.state.image_token_numbers == set()



def test_thread_run_state_tracks_queues_per_thread(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "thr_attached"
    app.engine.thread_store.threads = [
        {"thread_id": "thr_attached", "title": "Attached"},
        {"thread_id": "thr_bg", "title": "Background"},
    ]
    attached = tui2_app.ThreadRunState(thread_id="thr_attached")
    attached.pending_turns.append(tui2_app.PendingTurn("attached queued"))
    background = tui2_app.ThreadRunState(thread_id="thr_bg")
    background.pending_turns.append(tui2_app.PendingTurn("background queued"))
    app._thread_runs = {"thr_attached": attached, "thr_bg": background}

    app._sync_attached_run_state(attached)

    assert app.state.pending_turns is attached.pending_turns
    rows = {row.thread_id: row for row in [app._agent_view_row_for_thread("thr_attached", app.engine.thread_store.threads[0]), app._agent_view_row_for_thread("thr_bg", app.engine.thread_store.threads[1])]}
    assert rows["thr_attached"].queued_turns == 1
    assert rows["thr_bg"].queued_turns == 1


def test_resuming_thread_preserves_running_state_for_previous_thread(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    app.state.thread_id = "thr_1"
    app.state.live = [TranscriptCell("assistant", text="streaming", status="streaming")]
    run_state = tui2_app.ThreadRunState(thread_id="thr_1")
    app._thread_runs["thr_1"] = run_state
    app.engine.thread_store.threads = [
        {"thread_id": "thr_1", "title": "One"},
        {"thread_id": "thr_2", "title": "Two"},
    ]

    app._resume_thread("thr_2")

    assert app.state.thread_id == "thr_2"
    assert app.state.live == []
    assert "thr_1" in app._thread_runs


def test_terminal_reads_bracketed_paste_as_single_key() -> None:
    terminal = Terminal(stdin=io.StringIO("\x1b[200~one\r\ntwo\x1b[201~"))
    terminal._windows = False

    assert terminal.read_key() == PASTE_PREFIX + "one\ntwo"


def _install_fake_msvcrt(monkeypatch, text: str) -> None:
    """Provide a tiny ``msvcrt`` module so Windows input paths run on CI."""

    chars = list(text)

    def fake_getwch() -> str:
        return chars.pop(0)

    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(getwch=fake_getwch, kbhit=lambda: bool(chars)))


def test_windows_terminal_reads_vt_paste_before_enter(monkeypatch) -> None:
    terminal = Terminal()
    terminal._windows = True
    _install_fake_msvcrt(monkeypatch, "\x1b[200~one\r\ntwo\x1b[201~\r")

    assert terminal.read_key() == PASTE_PREFIX + "one\ntwo"
    assert terminal.read_key() == "\r"


def test_windows_terminal_coalesces_unbracketed_paste(monkeypatch) -> None:
    terminal = Terminal()
    terminal._windows = True
    _install_fake_msvcrt(monkeypatch, "one\r\ntwo")

    assert terminal.read_key() == PASTE_PREFIX + "one\ntwo"


def test_windows_terminal_translates_oem_102_to_angle_bracket(monkeypatch) -> None:
    """The OEM_102 scan code should reach the composer as a real character.

    On ISO/European 102-key keyboards, ``<`` arrives as an extended-key
    sequence (``\xe0`` + scan code 0x56).  Without the layout-aware
    translator the wrapped ``"<V>"`` token would be silently dropped by
    the composer.
    """
    terminal = Terminal()
    terminal._windows = True
    _install_fake_msvcrt(monkeypatch, "\xe0V")
    monkeypatch.setattr(
        Terminal,
        "_translate_extended_scan_code",
        lambda self, code: "<" if code == "V" else None,
    )

    assert terminal.read_key() == "<"


def test_windows_terminal_falls_back_to_token_when_translator_returns_none(monkeypatch) -> None:
    """Navigation-like extended keys keep the ``<...>`` token contract.

    PageUp still arrives as ``"<I>"`` so the higher-level key dispatcher
    can match it; only character-producing extended keys get rewritten.
    """
    terminal = Terminal()
    terminal._windows = True
    _install_fake_msvcrt(monkeypatch, "\xe0I")
    monkeypatch.setattr(
        Terminal,
        "_translate_extended_scan_code",
        lambda self, code: None,
    )

    assert terminal.read_key() == "<I>"


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="ctypes.windll is only available on Windows; verified on windows-latest CI",
)
def test_translate_extended_scan_code_uses_win32_translation(monkeypatch) -> None:
    """The translator must consult MapVirtualKeyW + ToUnicode + modifier state."""
    import ctypes

    class FakeUser32:
        def __init__(self) -> None:
            self.map_args: list[tuple[int, int]] = []
            self.tounicode_args: list[tuple[int, int, int]] = []
            self.async_calls: list[int] = []

        def MapVirtualKeyW(self, scan: int, mode: int) -> int:  # noqa: ARG002
            self.map_args.append((scan, mode))
            return 0xE2  # VK_OEM_102

        def GetAsyncKeyState(self, vk: int) -> int:
            self.async_calls.append(vk)
            return 0x8000 if vk in (0x10, 0xA0) else 0  # shift pressed

        def ToUnicode(self, vk: int, scan: int, state, buf, bufsize: int, flags: int):  # noqa: ARG002
            self.tounicode_args.append((vk, scan, bufsize))
            buf[0] = ord("<")
            return 1

    fake = FakeUser32()
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=fake))

    terminal = Terminal()
    assert terminal._translate_extended_scan_code("V") == "<"
    assert (0x56, 3) in fake.map_args
    assert fake.async_calls  # modifier state was queried
    assert fake.tounicode_args and fake.tounicode_args[0][2] == 8


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="ctypes.windll is only available on Windows; verified on windows-latest CI",
)
def test_translate_extended_scan_code_returns_none_for_unmappable_keys() -> None:
    """Empty input, non-byte scan codes, and unmapped scan codes return None."""
    terminal = Terminal()

    assert terminal._translate_extended_scan_code("") is None
    assert terminal._translate_extended_scan_code("\u0100") is None  # > 0xFF

    import ctypes

    class StubUser32:
        def MapVirtualKeyW(self, scan: int, mode: int) -> int:  # noqa: ARG002
            return 0  # nothing mapped

        def GetAsyncKeyState(self, vk: int) -> int:  # noqa: ARG002
            return 0

        def ToUnicode(self, *args, **kwargs):  # noqa: ARG002
            return 0  # no character

    original_windll = ctypes.windll
    ctypes.windll = SimpleNamespace(user32=StubUser32())  # type: ignore[attr-defined]
    try:
        assert terminal._translate_extended_scan_code("V") is None
    finally:
        ctypes.windll = original_windll  # type: ignore[attr-defined]


def test_unbracketed_paste_fallback_does_not_swallow_stringio_input() -> None:
    terminal = Terminal(stdin=io.StringIO("ab"))
    terminal._windows = False

    assert terminal.read_key() == "a"
    assert terminal.read_key() == "b"


def test_terminal_key_reader_uses_one_thread_for_repeated_keyboardinterrupt(monkeypatch) -> None:
    class InterruptingTerminal:
        _windows = True

        def __init__(self) -> None:
            self.calls = 0

        def read_key(self) -> str:
            self.calls += 1
            if self.calls <= 3:
                raise KeyboardInterrupt
            return "q"

    started_threads: list[threading.Thread] = []
    original_thread = threading.Thread

    class RecordingThread(original_thread):
        def start(self) -> None:
            started_threads.append(self)
            super().start()

    monkeypatch.setattr(threading, "Thread", RecordingThread)
    terminal = InterruptingTerminal()

    async def collect_keys() -> list[str]:
        with TerminalKeyReader(terminal, capture_sigint=False) as reader:
            return [await reader.read_key() for _ in range(4)]

    assert asyncio.run(collect_keys()) == ["\x03", "\x03", "\x03", "q"]
    assert len(started_threads) == 1


def test_terminal_key_reader_turns_sigint_signal_into_ctrl_c(monkeypatch) -> None:
    import signal

    installed = {}
    previous_handler = object()

    monkeypatch.setattr(signal, "getsignal", lambda signum: previous_handler)

    def fake_signal(signum, handler):
        installed[signum] = handler

    monkeypatch.setattr(signal, "signal", fake_signal)

    async def wait_for_sigint_key() -> str:
        terminal = SimpleNamespace(_windows=True, read_key=lambda: "")
        with TerminalKeyReader(terminal) as reader:
            installed[signal.SIGINT](signal.SIGINT, None)
            return await reader.read_key()

    assert asyncio.run(wait_for_sigint_key()) == "\x03"
    assert installed[signal.SIGINT] is previous_handler


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


def test_retained_flushed_cell_truncates_long_text() -> None:
    long_text = "x" * (TUI2_RETAINED_FLUSHED_TEXT_CHARS + 100)
    cell = TranscriptCell("assistant", text=long_text)
    retained = _retained_flushed_cell(cell)

    assert len(retained.text) < len(long_text)
    assert "...[truncated]" in retained.text
    assert retained.text.startswith("x" * (TUI2_RETAINED_FLUSHED_TEXT_CHARS // 2))



def test_tool_cell_shows_import_anchor_chains_with_method_calls() -> None:
    call = {
        "name": "run_python",
        "call_id": "call_abc",
        "arguments": json.dumps(
            {
                "code": (
                    "from pathlib import Path\n"
                    "from uv_agent_runtime import search_text, read_file\n"
                    "import json\n\n"
                    "hits = search_text(\"foo\")\n"
                    "p = Path.home().resolve()\n"
                    "data = json.loads(read_file(p).text)\n"
                )
            }
        ),
    }
    lines = render_tool_cell(TranscriptCell("tool", call=call, payload={"returncode": 0}), 80)
    plain = "\n".join(strip_ansi(line) for line in lines)

    assert "search_text" in plain
    assert "Path.home.resolve" in plain
    assert "json.loads" in plain
    assert "read_file" in plain


def test_pager_renders_content_with_fixed_chrome() -> None:
    state = Tui2State(pager_open=True, pager_title="test run", pager_lines=["line1", "line2", "line3"])
    lines, cursor_row, cursor_col = render_live_with_cursor(state, 80, max_height=8)
    plain = "\n".join(strip_ansi(line) for line in lines)

    assert "test run" in plain
    assert "line1" in plain
    assert "line2" in plain
    assert "line3" in plain
    assert "q=close" in plain
    # Footer is the second-to-last row inside the outer border.
    assert cursor_row == len(lines) - 2


def test_history_cells_merge_tool_call_and_result(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    thread_id = app.engine.thread_store.create_thread("tool merge test")

    def fake_read_history_segment(thread_id, *, event_types=None):
        from uv_agent.session.store import ThreadHistorySegment

        return ThreadHistorySegment(
            events=[
                {
                    "type": "item.model_response",
                    "thread_id": thread_id,
                    "turn_id": "turn_1",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_123",
                            "name": "run_python",
                            "arguments": json.dumps({"code": "print(1)"}),
                        }
                    ],
                },
                {
                    "type": "item.runner_result",
                    "thread_id": thread_id,
                    "turn_id": "turn_1",
                    "call_id": "call_123",
                    "result": {
                        "run_id": "run_abc",
                        "returncode": 0,
                        "stdout": "ok",
                    },
                },
            ],
            start_event_id=0,
            end_event_id=2,
            has_more=False,
        )

    monkeypatch.setattr(app.engine.thread_store, "read_history_segment", fake_read_history_segment)

    cells = app._history_cells_for_thread(thread_id)
    tool_cells = [cell for cell in cells if cell.kind == "tool"]

    assert len(tool_cells) == 1
    assert tool_cells[0].call is not None
    assert tool_cells[0].call.get("call_id") == "call_123"
    assert tool_cells[0].payload is not None
    assert tool_cells[0].payload.get("run_id") == "run_abc"


def test_history_merge_preserves_reasoning_before_tool_calls(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    thread_id = app.engine.thread_store.create_thread("reasoning merge test")

    def fake_read_history_segment(thread_id, *, event_types=None):
        from uv_agent.session.store import ThreadHistorySegment

        return ThreadHistorySegment(
            events=[
                {
                    "type": "item.model_response",
                    "thread_id": thread_id,
                    "turn_id": "turn_1",
                    "reasoning_text": "first thought",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "run_python",
                            "arguments": json.dumps({"code": "print(1)"}),
                        }
                    ],
                },
                {
                    "type": "item.runner_result",
                    "thread_id": thread_id,
                    "turn_id": "turn_1",
                    "call_id": "call_1",
                    "result": {"run_id": "run_1", "returncode": 0},
                },
                {
                    "type": "item.model_response",
                    "thread_id": thread_id,
                    "turn_id": "turn_1",
                    "reasoning_text": "second thought",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_2",
                            "name": "run_python",
                            "arguments": json.dumps({"code": "print(2)"}),
                        }
                    ],
                },
                {
                    "type": "item.runner_result",
                    "thread_id": thread_id,
                    "turn_id": "turn_1",
                    "call_id": "call_2",
                    "result": {"run_id": "run_2", "returncode": 0},
                },
            ],
            start_event_id=0,
            end_event_id=4,
            has_more=False,
        )

    monkeypatch.setattr(app.engine.thread_store, "read_history_segment", fake_read_history_segment)

    cells = app._history_cells_for_thread(thread_id)
    reasoning_cells = [cell for cell in cells if cell.kind == "reasoning"]
    tool_cells = [cell for cell in cells if cell.kind == "tool"]

    assert len(reasoning_cells) == 2
    assert reasoning_cells[0].text == "first thought"
    assert reasoning_cells[1].text == "second thought"
    assert len(tool_cells) == 2


def test_show_command_opens_pager_for_matching_run(monkeypatch) -> None:
    app = _make_app(monkeypatch)
    thread_id = app.engine.thread_store.create_thread("show test")
    app.state.thread_id = thread_id

    def fake_read_events(thread_id, *, event_types=None):
        return [
            {
                "type": "item.runner_result",
                "thread_id": thread_id,
                "call": {
                    "call_id": "call_show",
                    "name": "run_python",
                    "arguments": json.dumps({"code": "print(1)"}),
                },
                "result": {
                    "run_id": "run_showme",
                    "returncode": 0,
                    "stdout": "one",
                    "stderr": "",
                    "events": [],
                },
            }
        ]

    monkeypatch.setattr(app.engine.thread_store, "read_events", fake_read_events)

    app._handle_command("/show run_showme")

    assert app.state.pager_open
    assert app.state.pager_run_id is not None
    plain = "\n".join(strip_ansi(line) for line in app.state.pager_lines)
    assert "print(1)" in plain
    assert "one" in plain
