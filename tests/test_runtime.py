from __future__ import annotations

import os
import json
from pathlib import Path
import sys
import threading

import pytest

from uv_agent_runtime import (
    apply_patch,
    ask,
    check_command,
    connect_declared,
    connect_named,
    connect_stdio,
    emit_event,
    emit_progress,
    emit_result,
    enter_dir,
    list_declared_servers,
    list_files,
    look_at,
    read_json,
    read_text,
    run_command,
    saved_scripts,
    thread_digest,
    write_json,
    write_text,
)
from uv_agent_runtime.subagent import NESTED_ASK_BLOCKED_MESSAGE, _extract_subagent_thread_id
from uv_agent.session import ThreadStore


def test_runtime_file_helpers(tmp_path: Path) -> None:
    write_text(tmp_path / "a.txt", "hello")
    write_json(tmp_path / "nested" / "data.json", {"ok": True})

    assert read_text(tmp_path / "a.txt") == "hello"
    assert read_json(tmp_path / "nested" / "data.json") == {"ok": True}
    assert "a.txt" in list_files(tmp_path, pattern="*.txt")


def test_runtime_apply_patch_helper(tmp_path: Path) -> None:
    write_text(tmp_path / "a.txt", "old\n")
    write_text(tmp_path / "remove.txt", "unused\n")

    result = apply_patch(
        """*** Begin Patch
*** Update File: a.txt
@@
-old
+new
*** Add File: nested/b.txt
+hello
+world
*** Delete File: remove.txt
*** End Patch
""",
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.changed_files == ["a.txt", "nested/b.txt", "remove.txt"]
    assert read_text(tmp_path / "a.txt") == "new\n"
    assert read_text(tmp_path / "nested" / "b.txt") == "hello\nworld\n"
    assert not (tmp_path / "remove.txt").exists()


def test_runtime_apply_patch_helper_moves_file(tmp_path: Path) -> None:
    write_text(tmp_path / "old.txt", "before\n")

    result = apply_patch(
        """*** Begin Patch
*** Update File: old.txt
*** Move to: new.txt
@@ rename and edit
-before
+after
*** End Patch
""",
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.changed_files == ["new.txt", "old.txt"]
    assert not (tmp_path / "old.txt").exists()
    assert read_text(tmp_path / "new.txt") == "after\n"


def test_runtime_apply_patch_helper_returns_failure_without_writing(tmp_path: Path) -> None:
    write_text(tmp_path / "a.txt", "old\n")

    result = apply_patch(
        """*** Begin Patch
*** Update File: a.txt
@@
-missing
+new
*** End Patch
""",
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 1
    assert "hunk context was not found" in result.stderr
    assert result.changed_files == []
    assert read_text(tmp_path / "a.txt") == "old\n"


def test_runtime_apply_patch_helper_preserves_existing_context_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("first\r\nold\r\nlast\r\n", encoding="utf-8", newline="")

    result = apply_patch(
        """*** Begin Patch
*** Update File: a.txt
@@
 first
-old
+new
 last
*** End Patch
""",
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert path.read_bytes() == b"first\r\nnew\r\nlast\r\n"


def test_runtime_apply_patch_helper_preserves_lf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("first\nold\nlast\n", encoding="utf-8", newline="")

    result = apply_patch(
        """*** Begin Patch
*** Update File: a.txt
@@
 first
-old
+new
 last
*** End Patch
""",
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert path.read_bytes() == b"first\nnew\nlast\n"


def test_runtime_apply_patch_helper_explains_bare_blank_hunk_line(tmp_path: Path) -> None:
    write_text(tmp_path / "a.txt", "first\n\nlast\n")

    result = apply_patch(
        """*** Begin Patch
*** Update File: a.txt
@@
 first

 last
*** End Patch
""",
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 1
    assert "blank hunk line without a diff prefix" in result.stderr
    assert "Every hunk line must start with a space" in result.stderr
    assert read_text(tmp_path / "a.txt") == "first\n\nlast\n"


def test_runtime_apply_patch_helper_rejects_paths_outside_workdir(tmp_path: Path) -> None:
    result = apply_patch(
        """*** Begin Patch
*** Add File: ../outside.txt
+nope
*** End Patch
""",
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 1
    assert "escapes the working directory" in result.stderr
    assert not (tmp_path.parent / "outside.txt").exists()


def test_runtime_command_helpers() -> None:
    result = run_command(["python", "-c", "print('ok')"])

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"
    assert check_command(["python", "-c", "print('checked')"]).stdout.strip() == "checked"


def test_runtime_enter_dir_changes_cwd_and_emits_event(tmp_path: Path, capsys) -> None:
    previous = Path.cwd()
    try:
        resolved = enter_dir(tmp_path)
        out = capsys.readouterr().out
        event = json.loads(out)

        assert resolved == tmp_path.resolve()
        assert Path.cwd() == tmp_path.resolve()
        assert event["kind"] == "enter_dir"
        assert event["cwd"] == str(tmp_path.resolve())
        assert event["_uv_agent_event_id"].startswith("evt_")
    finally:
        os.chdir(previous)


def test_runtime_mcp_stdio_client() -> None:
    server = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"

    with connect_stdio([sys.executable, str(server)]) as client:
        init = client.initialize()
        tools = client.list_tools()
        result = client.call_tool("echo", {"text": "hello"})

    assert init.value["serverInfo"]["name"] == "echo"
    assert tools[0]["name"] == "echo"
    assert result.value["content"][0]["text"] == "hello"


def test_runtime_mcp_connect_declared(tmp_path: Path) -> None:
    server = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    config = tmp_path / "mcp.json"
    write_json(
        config,
        {
            "servers": {
                "echo": {
                    "command": sys.executable,
                    "args": [str(server)],
                }
            }
        },
    )

    with connect_declared("echo", config_path=config) as client:
        client.initialize()
        result = client.call_tool("echo", {"text": "declared"})

    assert result.value["content"][0]["text"] == "declared"


def test_runtime_mcp_lists_and_connects_named(tmp_path: Path) -> None:
    server = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    agents = tmp_path / ".agents"
    agents.mkdir()
    write_json(
        agents / "mcp.json",
        {"servers": {"echo": {"command": sys.executable, "args": [str(server)]}}},
    )

    declared = list_declared_servers(cwd=tmp_path)
    with connect_named("echo", cwd=tmp_path) as client:
        client.initialize()
        result = client.call_tool("echo", {"text": "named"})

    assert declared[0]["name"] == "echo"
    assert result.value["content"][0]["text"] == "named"


def test_runtime_subagent_ask_with_custom_executable() -> None:
    result = ask(
        "ignored",
        executable=[sys.executable, "-c", "import sys; print(sys.argv[-1])"],
        check=True,
    )

    assert result.text == "ignored"


def test_runtime_subagent_accepts_model_level_alias() -> None:
    result = ask(
        "ignored",
        model_level="small",
        executable=[sys.executable, "-c", "import sys; print(' '.join(sys.argv[1:]))"],
        check=True,
    )

    assert "--level small ask ignored" in result.text


def test_runtime_subagent_ask_uses_temporary_project_state_without_host_state() -> None:
    code = (
        "import os; from pathlib import Path; "
        "state = Path(os.environ['UV_AGENT_RUNTIME_PROJECT_STATE_DIR']); "
        "(state / 'marker.txt').write_text('temporary', encoding='utf-8'); "
        "print(state)"
    )
    result = ask("ignored", executable=[sys.executable, "-c", code], check=True)

    assert result.text
    assert not Path(result.text).exists()
    assert "UV_AGENT_RUNTIME_PROJECT_STATE_DIR" not in os.environ


def test_runtime_subagent_ask_retains_project_state_when_host_state_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = (
        "import os; "
        "print(os.environ['UV_AGENT_RUNTIME_PROJECT_STATE_DIR']); "
        "import sys; print('[subagent-thread] thr_child', file=sys.stderr)"
    )
    monkeypatch.setenv("UV_AGENT_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("UV_AGENT_RUNTIME_THREAD_ID", "thr_parent")
    monkeypatch.setenv("UV_AGENT_RUNTIME_TURN_ID", "turn_parent")
    result = ask("ignored", executable=[sys.executable, "-c", code], check=True)

    assert result.text == str(tmp_path)
    assert result.thread_id == "thr_child"


def test_runtime_subagent_ask_blocks_nested_subagent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_AGENT_RUNTIME_THREAD_KIND", "subagent")
    monkeypatch.setenv("UV_AGENT_RUNTIME_RUN_ID", "run_child")

    result = ask(
        "delegate again",
        executable=[sys.executable, "-c", "raise SystemExit('should not run')"],
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.thread_id is None
    assert result.stderr == NESTED_ASK_BLOCKED_MESSAGE


def test_runtime_subagent_events_do_not_include_prompt(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_AGENT_RUNTIME_RUN_ID", "run_parent")

    result = ask(
        "secret task text",
        executable=[sys.executable, "-c", "import sys; print('done'); print('[subagent-thread] thr_child', file=sys.stderr)"],
        check=True,
    )
    out = capsys.readouterr().out
    events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]

    assert result.text == "done"
    assert result.thread_id == "thr_child"
    assert [event["kind"] for event in events] == ["subagent.started", "subagent.completed"]
    assert all("prompt" not in event for event in events)


def test_extract_subagent_thread_id_from_stderr() -> None:
    assert _extract_subagent_thread_id("noise\n[subagent-thread] thr_123\n") == "thr_123"


def test_runtime_saved_scripts_reads_state_dir(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "scr_1"
    script.mkdir(parents=True)
    final = script / "script.py"
    final.write_text("# /// script\n# dependencies=[]\n# ///\n\nprint('hello')\n", encoding="utf-8")
    (script / "metadata.json").write_text(
        json.dumps(
            {
                "script_id": "scr_1",
                "created_at": "2026-01-01T00:00:00Z",
                "final_path": str(final),
            }
        ),
        encoding="utf-8",
    )

    summaries = saved_scripts(state_dir=tmp_path)

    assert summaries[0]["script_id"] == "scr_1"
    assert summaries[0]["summary"] == "print('hello')"


def test_runtime_thread_digest_reads_state_dir(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Thread")
    store.append(
        thread_id,
        "item.user",
        turn_id="t1",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
    )
    store.append(thread_id, "item.compaction", turn_id="t1", text="summary")
    store.append(
        thread_id,
        "item.user",
        turn_id="t2",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "after"}]},
    )

    digest = thread_digest(thread_id, state_dir=tmp_path)

    assert digest["latest_compaction"]["text"] == "summary"
    assert digest["items"] == [{"role": "user", "text": "after"}]


def test_runtime_look_at_emits_structured_event(tmp_path: Path, capsys) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    event = look_at(image, note="inspect")
    out = capsys.readouterr().out

    assert event["kind"] == "look_at"
    assert event["path"] == str(image.resolve())
    assert event["note"] == "inspect"
    assert event["_uv_agent_event_id"].startswith("evt_")
    assert '"kind": "look_at"' in out
    assert '"note": "inspect"' in out


def test_runtime_emit_helpers_return_event_dict(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_AGENT_RUNTIME_RUN_ID", "run_events")

    custom = emit_event("custom", value=1)
    progress = emit_progress("working", count=2)
    result = emit_result(ok=True)
    out = capsys.readouterr().out

    assert custom["kind"] == "custom"
    assert custom["value"] == 1
    assert custom["_uv_agent_run_id"] == "run_events"
    assert custom["_uv_agent_event_id"].startswith("evt_")
    assert progress["kind"] == "progress"
    assert progress["message"] == "working"
    assert progress["count"] == 2
    assert progress["_uv_agent_run_id"] == "run_events"
    assert progress["_uv_agent_event_id"].startswith("evt_")
    assert result["kind"] == "result"
    assert result["ok"] is True
    assert result["_uv_agent_run_id"] == "run_events"
    assert result["_uv_agent_event_id"].startswith("evt_")
    assert len(
        {
            custom["_uv_agent_event_id"],
            progress["_uv_agent_event_id"],
            result["_uv_agent_event_id"],
        }
    ) == 3
    assert '"kind": "custom"' in out
    assert '"kind": "progress"' in out
    assert '"kind": "result"' in out


def test_runtime_emit_event_writes_complete_lines_from_threads(
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UV_AGENT_RUNTIME_RUN_ID", "run_threads")

    def emit_many(worker: int) -> None:
        for index in range(25):
            emit_event("threaded", worker=worker, index=index)

    threads = [threading.Thread(target=emit_many, args=(worker,)) for worker in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = capsys.readouterr().out.splitlines()
    events = [json.loads(line) for line in lines]

    assert len(events) == 100
    assert all(event["kind"] == "threaded" for event in events)
    assert all(event["_uv_agent_run_id"] == "run_threads" for event in events)
    event_ids = [event["_uv_agent_event_id"] for event in events]
    assert all(event_id.startswith("evt_") for event_id in event_ids)
    assert len(set(event_ids)) == len(event_ids)
