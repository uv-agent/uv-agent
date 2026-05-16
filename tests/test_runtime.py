from __future__ import annotations

import os
import json
from pathlib import Path
import sys

from uv_agent_runtime import (
    apply_patch,
    ask,
    check_command,
    connect_declared,
    connect_named,
    connect_stdio,
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
from uv_agent_runtime.subagent import _extract_subagent_thread_id


def test_runtime_file_helpers(tmp_path: Path) -> None:
    write_text(tmp_path / "a.txt", "hello")
    write_json(tmp_path / "nested" / "data.json", {"ok": True})

    assert read_text(tmp_path / "a.txt") == "hello"
    assert read_json(tmp_path / "nested" / "data.json") == {"ok": True}
    assert "a.txt" in list_files(tmp_path, pattern="*.txt")


def test_runtime_apply_patch_helper(tmp_path: Path) -> None:
    run_command(["git", "init"], cwd=str(tmp_path))
    write_text(tmp_path / "a.txt", "old\n")

    result = apply_patch(
        """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-old
+new
""",
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "a.txt" in result.changed_files
    assert read_text(tmp_path / "a.txt") == "new\n"


def test_runtime_command_helpers() -> None:
    result = run_command(["python", "-c", "print('ok')"])

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"
    assert check_command(["python", "-c", "print('checked')"]).stdout.strip() == "checked"


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
        "state = Path(os.environ['UV_AGENT_PROJECT_STATE_DIR']); "
        "(state / 'marker.txt').write_text('temporary', encoding='utf-8'); "
        "print(state)"
    )
    result = ask("ignored", executable=[sys.executable, "-c", code], check=True)

    assert result.text
    assert not Path(result.text).exists()
    assert "UV_AGENT_PROJECT_STATE_DIR" not in os.environ


def test_runtime_subagent_ask_retains_project_state_when_host_state_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = (
        "import os; "
        "print(os.environ['UV_AGENT_PROJECT_STATE_DIR']); "
        "import sys; print('[subagent-thread] thr_child', file=sys.stderr)"
    )
    monkeypatch.setenv("UV_AGENT_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("UV_AGENT_THREAD_ID", "thr_parent")
    monkeypatch.setenv("UV_AGENT_TURN_ID", "turn_parent")
    result = ask("ignored", executable=[sys.executable, "-c", code], check=True)

    assert result.text == str(tmp_path)
    assert result.thread_id == "thr_child"


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
    thread_dir = tmp_path / "threads"
    thread_dir.mkdir()
    thread_id = "thr_test"
    events = [
        {"type": "thread.created", "created_at": "1", "thread_id": thread_id, "title": "Thread"},
        {
            "type": "item.user",
            "created_at": "2",
            "thread_id": thread_id,
            "turn_id": "t1",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        },
        {"type": "item.compaction", "created_at": "3", "thread_id": thread_id, "turn_id": "t1", "text": "summary"},
        {
            "type": "item.user",
            "created_at": "4",
            "thread_id": thread_id,
            "turn_id": "t2",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "after"}]},
        },
    ]
    (thread_dir / f"{thread_id}.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    digest = thread_digest(thread_id, state_dir=tmp_path)

    assert digest["latest_compaction"]["text"] == "summary"
    assert digest["items"] == [{"role": "user", "text": "after"}]


def test_runtime_look_at_emits_structured_event(tmp_path: Path, capsys) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    resolved = look_at(image, note="inspect")
    out = capsys.readouterr().out

    assert resolved == image.resolve()
    assert '"kind": "look_at"' in out
    assert '"note": "inspect"' in out
