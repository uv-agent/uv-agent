from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from uv_agent_runtime import (
    add_dependency,
    apply_patch,
    apply_patch_any,
    ask,
    clear_codequery_cache,
    CommandTextResult,
    compare_text,
    connect_declared,
    connect_named,
    connect_stdio,
    connect_url,
    convert_patch,
    emit_event,
    emit_progress,
    emit_result,
    enter_dir,
    edit_lines,
    find_files,
    find_symbols,
    list_declared_servers,
    list_files,
    list_thread_digests,
    look_at,
    make_unified_diff,
    normalize_text,
    path_info,
    query_code,
    read_file,
    read_json,
    read_text,
    read_text_lossless,
    replace_text,
    restore_snapshot,
    run_python_env_dir,
    run_process_text,
    run_digest,
    search_text,
    snapshot_files,
    supported_symbol_languages,
    thread_digest,
    workspace_transaction,
    write_file,
    write_text_lossless,
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


def test_runtime_run_process_text_check_and_result_helpers() -> None:
    result = run_process_text([sys.executable, "-c", "print('ok')"], check=True)

    assert result.returncode == 0
    assert result.ok is True
    assert result.stdout.strip() == "ok"
    assert result.raise_for_error() is result

    with pytest.raises(RuntimeError, match="command failed with exit 3"):
        run_process_text([sys.executable, "-c", "raise SystemExit(3)"], check=True)


def test_runtime_run_process_text_timeout_returns_partial_output() -> None:
    result = run_process_text(
        [
            sys.executable,
            "-c",
            "import sys, time; print('started', flush=True); time.sleep(30)",
        ],
        timeout_s=0.2,
    )

    assert result.timed_out is True
    assert result.returncode != 0
    assert "started" in result.stdout
    with pytest.raises(RuntimeError, match="command timed out"):
        result.raise_for_error()


def test_runtime_run_process_text_accepts_env_and_env_patch() -> None:
    code = "import os; print(os.environ.get('UV_AGENT_TEST_VALUE', 'missing'))"

    result = run_process_text(
        [sys.executable, "-c", code],
        env={},
        env_patch={"UV_AGENT_TEST_VALUE": "patched"},
        check=True,
    )

    assert result.stdout.strip() == "patched"


def test_runtime_run_process_text_resolves_command_from_env_path(
    tmp_path: Path,
) -> None:
    command_name = "uv-agent-runtime-test-command"
    script = tmp_path / (command_name + (".cmd" if os.name == "nt" else ""))
    if os.name == "nt":
        script.write_text("@echo off\r\necho resolved-from-env-path\r\n", encoding="utf-8")
    else:
        script.write_text("#!/bin/sh\necho resolved-from-env-path\n", encoding="utf-8")
        script.chmod(0o755)

    process_env = os.environ.copy()
    process_env["PATH"] = str(tmp_path)
    process_env["PATHEXT"] = ".CMD;.EXE;.BAT;.COM"

    result = run_process_text(
        [command_name],
        env=process_env,
        check=True,
    )

    assert result.stdout.strip() == "resolved-from-env-path"


def test_runtime_run_process_text_env_patch_updates_path_case_insensitively(
    tmp_path: Path,
) -> None:
    command_name = "uv-agent-runtime-test-patched-path"
    script = tmp_path / (command_name + (".cmd" if os.name == "nt" else ""))
    if os.name == "nt":
        script.write_text("@echo off\r\necho resolved-from-env-patch\r\n", encoding="utf-8")
    else:
        script.write_text("#!/bin/sh\necho resolved-from-env-patch\n", encoding="utf-8")
        script.chmod(0o755)

    process_env = os.environ.copy()
    path_key = "Path" if os.name == "nt" else "PATH"
    for key in list(process_env):
        if key.casefold() == "path":
            process_env.pop(key)
    process_env[path_key] = ""

    result = run_process_text(
        [command_name],
        env=process_env,
        env_patch={"PATH": str(tmp_path)},
        check=True,
    )

    assert result.stdout.strip() == "resolved-from-env-patch"


def test_runtime_dependency_helpers_use_run_python_env_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uv_agent_runtime.dependencies as dependencies

    scriptenv = tmp_path / "scriptenv"
    scriptenv.mkdir()
    monkeypatch.setenv("UV_AGENT_SCRIPTENV_DIR", str(scriptenv))
    monkeypatch.setenv("UV_BIN", "uv-test")
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_run_process_text(*args: Any, **kwargs: Any) -> CommandTextResult:
        assert (scriptenv / ".uv-agent-scriptenv.lock").exists()
        calls.append((args, kwargs))
        return CommandTextResult(args=list(args[0]), returncode=0, stdout="", stderr="")

    monkeypatch.setattr(dependencies, "run_process_text", fake_run_process_text)

    assert run_python_env_dir() == scriptenv.resolve()

    result = add_dependency("idna", check=False, timeout_s=1)

    assert result.args[:4] == ["uv-test", "add", "--project", str(scriptenv.resolve())]
    assert result.args[-1] == "idna"
    assert calls[0][1] == {"timeout_s": 1, "check": False}


def test_runtime_lossless_text_helpers_preserve_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"\xef\xbb\xbffirst\r\nsecond\r\n")

    loaded = read_text_lossless(path)

    assert loaded.text == "first\r\nsecond\r\n"
    assert loaded.bom is True
    assert loaded.newline == "crlf"
    assert loaded.final_newline is True

    write_text_lossless(path, "changed\nagain\n", like=loaded)

    assert path.read_bytes() == b"\xef\xbb\xbfchanged\r\nagain\r\n"



def test_runtime_read_file_views_and_write_file_preserve_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"\xef\xbb\xbffirst\r\nsecond\r\nthird\r\n")

    view = read_file(path, around="second", context=1)

    assert view.path == str(path.resolve())
    assert view.exists is True
    assert view.kind == "file"
    assert view.line_count == 3
    assert view.start_line == 1
    assert view.end_line == 3
    assert view.text == "first\r\nsecond\r\nthird\r\n"
    assert view.numbered().splitlines()[1].endswith(": second")
    assert view.newline == "crlf"
    assert view.bom is True

    tail = read_file(path, tail=1)
    assert tail.start_line == 3
    assert tail.truncated is True
    assert tail.text == "third\r\n"

    missing = read_file(tmp_path / "missing.txt")
    assert missing.exists is False
    assert missing.kind == "missing"
    assert missing.path == str((tmp_path / "missing.txt").resolve())

    write_file(path, "changed\nagain\n", like=view)

    assert path.read_bytes() == b"\xef\xbb\xbfchanged\r\nagain\r\n"


def test_runtime_read_file_errors_include_recovery_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("first\nsecond\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line range \(2, 5\).*2 lines"):
        read_file(path, lines=(2, 5))
    with pytest.raises(ValueError, match="around text not found in file with 2 lines"):
        read_file(path, around="missing")



def test_runtime_edit_lines_replaces_inserts_deletes_and_checks_anchors(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    path.write_text(
        "def parse():\n    return 1\n\ndef other():\n    return 2\n",
        encoding="utf-8",
        newline="",
    )

    result = edit_lines(
        path,
        1,
        2,
        "def parse(value):\n    return value",
        expect_first="def parse",
        expect_last="return",
        expect_mode="contains",
    )

    assert result.path == str(path.resolve())
    assert result.changed is True
    assert result.replaced_text == "def parse():\n    return 1"
    assert result.line_count_before == 5
    assert result.line_count_after == 5
    assert result.line_delta == 0
    assert path.read_text(encoding="utf-8") == (
        "def parse(value):\n    return value\n\ndef other():\n    return 2\n"
    )

    inserted = edit_lines(path, 3, 2, "# inserted", expect_mode="exact")
    assert inserted.line_delta == 1
    assert "# inserted" in path.read_text(encoding="utf-8")

    deleted = edit_lines(path, 3, 3, "", expect_first="# inserted")
    assert deleted.line_delta == -1
    assert "# inserted" not in path.read_text(encoding="utf-8")

    eof = edit_lines(path, 6, 5, "# eof")
    assert eof.line_delta == 1
    assert path.read_text(encoding="utf-8").endswith("# eof\n")

    with pytest.raises(ValueError, match="expect_first did not match"):
        edit_lines(path, 1, 1, "def nope():", expect_first="class")
    with pytest.raises(ValueError, match=r"line range \(1, 99\).*6 lines"):
        edit_lines(path, 1, 99, "def nope():")
    with pytest.raises(ValueError, match="valid insert start"):
        edit_lines(path, 99, 98, "# nope")

def test_runtime_write_text_lossless_without_template_preserves_input_text(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"

    write_text_lossless(path, "first\r\nsecond\r\n")

    assert path.read_bytes() == b"first\r\nsecond\r\n"


def test_runtime_compare_and_normalize_text_helpers() -> None:
    comparison = compare_text("a\r\nb\r\n", "a\nb\n", ignore_eol=True)

    assert comparison.kind == "eol"
    assert normalize_text("a\r\nb", eol="crlf", final_newline=True) == "a\r\nb\r\n"


def test_runtime_replace_text_uses_logical_newlines_and_preserves_style(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("first\r\nold\r\n\r\nlast\r\n", encoding="utf-8", newline="")

    result = replace_text(path, "old\n\nlast", "new\n\nlast")

    assert result.replacements == 1
    assert path.read_bytes() == b"first\r\nnew\r\n\r\nlast\r\n"
    with pytest.raises(ValueError, match="File newline='crlf'"):
        replace_text(path, "missing", "nope")
    with pytest.raises(ValueError, match="old text must not be empty"):
        replace_text(path, "", "nope")
    with pytest.raises(ValueError, match="no-op"):
        replace_text(path, "new", "new")


def test_runtime_replace_text_raw_mode_is_newline_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("first\r\nold\r\nlast\r\n", encoding="utf-8", newline="")

    with pytest.raises(ValueError, match="CRLF/LF mismatch"):
        replace_text(path, "old\nlast", "new\nlast", newlines="raw")

    result = replace_text(path, "old\r\nlast", "new\r\nlast", newlines="raw")

    assert result.replacements == 1
    assert path.read_bytes() == b"first\r\nnew\r\nlast\r\n"




def test_runtime_replace_text_result_changed_and_repr_omits_full_text(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    long_text = "old " + ("secret-ish text " * 40) + "tail\n"
    path.write_text(long_text, encoding="utf-8")

    result = replace_text(path, "old", "new")

    assert result.changed is True
    assert result.replacements == 1
    rendered = repr(result)
    assert "TextFile" in rendered
    assert "secret-ish text" not in rendered
    assert "text=" not in rendered


def test_runtime_run_digest_summarizes_code_outputs_and_helper_calls(tmp_path: Path) -> None:
    from uv_agent.runner.run_log import RunLogStore

    store = RunLogStore(tmp_path)
    code = "from uv_agent_runtime import replace_text\nreplace_text('a.txt', 'old', 'new')\n"
    store.create_run_record(
        run_id="run_one",
        code=code,
        script_args=[],
        cwd=tmp_path,
        timeout_s=30,
        started_at="2026-01-01T00:00:00+00:00",
        thread_id="thr_one",
        turn_id="turn_one",
        script_path=None,
    )
    store.complete_run(
        run_id="run_one",
        completed_at="2026-01-01T00:00:01+00:00",
        returncode=0,
        timed_out=False,
        interrupted=False,
        truncated=False,
        stdout="line1\n" + "x" * 80,
        stderr="",
        structured_events=[{"kind": "cwd", "cwd": "."}],
    )

    digest = run_digest("run_one", state_dir=tmp_path, max_code_chars=30, max_output_chars=20)

    assert digest["run_id"] == "run_one"
    assert digest["thread_id"] == "thr_one"
    assert digest["returncode"] == 0
    assert digest["code_truncated"] is True
    assert digest["stdout_truncated"] is True
    assert digest["stdout"].endswith("x" * 20)
    assert digest["helper_calls"] == [
        {"name": "replace_text", "args": "'a.txt', 'old', 'new'", "line": 2}
    ]
    assert digest["structured_events"] == [{"kind": "cwd", "cwd": "."}]


def test_runtime_thread_digest_includes_bounded_tool_details(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Tools")
    store.append(
        thread_id,
        "item.runner_result",
        turn_id="turn_one",
        result={
            "run_id": "run_tool",
            "returncode": 0,
            "helper_calls": [{"name": "replace_text", "args": "'a.txt', 'old', 'new'"}],
        },
    )
    store.append(
        thread_id,
        "item.tool_output",
        turn_id="turn_one",
        item={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"run_id":"run_tool","returncode":0,"stdout":"ok\\n"}',
        },
    )

    digest = thread_digest(thread_id, state_dir=tmp_path, include_tools=True)

    assert digest["items"] == [
        {"role": "tool", "text": "run_python rc=0 run=run_tool helpers=replace_text('a.txt', 'old', 'new')"},
        {"role": "tool", "text": "tool_output run=run_tool rc=0 stdout='ok'"},
    ]


def test_runtime_replace_text_preserves_mixed_newlines(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"first\r\nold\nlast\r")

    result = replace_text(path, "old", "new")

    assert result.before.newline == "mixed"
    assert path.read_bytes() == b"first\r\nnew\nlast\r"


def test_runtime_unified_diff_conversion_and_apply_any(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("old\n", encoding="utf-8")
    diff = make_unified_diff("old\n", "new\n", path="a.txt")

    envelope = convert_patch(diff, from_format="unified", to_format="apply_patch")
    result = apply_patch_any(envelope, cwd=tmp_path)

    assert "*** Update File: a.txt" in envelope
    assert result.returncode == 0
    assert path.read_text(encoding="utf-8") == "new\n"


def test_runtime_apply_patch_any_dry_run_restores_created_files(tmp_path: Path) -> None:
    patch = """*** Begin Patch
*** Add File: nested/new.txt
+created
*** End Patch
"""

    result = apply_patch_any(patch, cwd=tmp_path, dry_run=True)

    assert result.returncode == 0
    assert result.changed_files == ["nested/new.txt"]
    assert not (tmp_path / "nested" / "new.txt").exists()
    assert not (tmp_path / "nested").exists()


def test_runtime_git_unified_diff_conversion_handles_add_delete_and_rename(tmp_path: Path) -> None:
    (tmp_path / "old.txt").write_text("remove\n", encoding="utf-8")
    (tmp_path / "rename_from.txt").write_text("before\n", encoding="utf-8")
    diff = """diff --git a/new.txt b/new.txt
new file mode 100644
index 0000000..3b18e51
--- /dev/null
+++ b/new.txt
@@ -0,0 +1,2 @@
+hello
+world
diff --git a/old.txt b/old.txt
deleted file mode 100644
index 3b18e51..0000000
--- a/old.txt
+++ /dev/null
@@ -1 +0,0 @@
-remove
diff --git a/rename_from.txt b/rename_to.txt
similarity index 50%
rename from rename_from.txt
rename to rename_to.txt
--- a/rename_from.txt
+++ b/rename_to.txt
@@ -1 +1 @@
-before
+after
"""

    envelope = convert_patch(diff, from_format="unified", to_format="apply_patch")
    result = apply_patch_any(envelope, cwd=tmp_path)

    assert "*** Add File: new.txt" in envelope
    assert "*** Delete File: old.txt" in envelope
    assert "*** Move to: rename_to.txt" in envelope
    assert result.returncode == 0
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert not (tmp_path / "old.txt").exists()
    assert not (tmp_path / "rename_from.txt").exists()
    assert (tmp_path / "rename_to.txt").read_text(encoding="utf-8") == "after\n"


def test_runtime_workspace_transaction_restores_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("before\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"):
        with workspace_transaction([path], root=tmp_path):
            path.write_text("after\n", encoding="utf-8")
            raise RuntimeError("boom")

    assert path.read_text(encoding="utf-8") == "before\n"


def test_runtime_snapshot_restore_and_path_info(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("before\n", encoding="utf-8")

    snapshot = snapshot_files([path], root=tmp_path)
    path.write_text("after\n", encoding="utf-8")
    restored = restore_snapshot(snapshot)
    info = path_info(path, base=tmp_path)

    assert restored == ["a.txt"]
    assert path.read_text(encoding="utf-8") == "before\n"
    assert info.kind == "file"
    assert info.is_relative_to_base is True


def test_runtime_run_process_text_decodes_explicitly() -> None:
    result = run_process_text(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write('✓'.encode('utf-8'))"],
        encoding="utf-8",
    )

    assert result.returncode == 0
    assert result.stdout == "✓"


def test_runtime_enter_dir_changes_cwd_and_returns_event(tmp_path: Path, capsys) -> None:
    previous = Path.cwd()
    try:
        resolved = enter_dir(tmp_path)
        captured = capsys.readouterr()

        assert resolved == tmp_path.resolve()
        assert Path.cwd() == tmp_path.resolve()
        assert captured.out == ""
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
    assert result.raw.content[0].text == "hello"


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
        {"servers": {"echo": {"transport": "stdio", "command": sys.executable, "args": [str(server)]}}},
    )

    declared = list_declared_servers(cwd=tmp_path)
    with connect_named("echo", cwd=tmp_path) as client:
        client.initialize()
        result = client.call_tool("echo", {"text": "named"})

    assert declared[0]["name"] == "echo"
    assert declared[0]["transport"] == "stdio"
    assert result.value["content"][0]["text"] == "named"


def test_runtime_mcp_lists_http_declarations(tmp_path: Path) -> None:
    agents = tmp_path / ".agents"
    agents.mkdir()
    write_json(
        agents / "mcp.json",
        {"servers": {"web": {"transport": "streamable_http", "url": "http://localhost:3001/mcp"}}},
    )

    declared = list_declared_servers(cwd=tmp_path)

    assert declared == [
        {
            "name": "web",
            "scope": "project",
            "path": str(agents / "mcp.json"),
            "description": "",
            "transport": "streamable_http",
            "command": None,
            "url": "http://localhost:3001/mcp",
        }
    ]


def test_runtime_mcp_defaults_to_runtime_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    nested = project / "nested"
    agents = project / ".agents"
    nested.mkdir(parents=True)
    agents.mkdir()
    write_json(
        agents / "mcp.json",
        {"servers": {"web": {"transport": "streamable_http", "url": "http://localhost:3001/mcp"}}},
    )
    monkeypatch.setenv("UV_AGENT_RUNTIME_PROJECT_ROOT", str(project))
    monkeypatch.chdir(nested)

    declared = list_declared_servers()

    assert declared[0]["name"] == "web"
    assert declared[0]["path"] == str(agents / "mcp.json")


def test_runtime_mcp_connect_url_rejects_stdio_transport() -> None:
    with pytest.raises(ValueError, match="streamable_http or sse"):
        connect_url("http://localhost:3001/mcp", transport="stdio")


def test_runtime_subagent_ask_with_custom_executable() -> None:
    result = ask(
        "ignored",
        executable=[sys.executable, "-c", "import sys; print(sys.argv[-1])"],
        check=True,
    )

    assert result.text == "ignored"
    assert result.timed_out is False


def test_runtime_subagent_ask_exposes_timeout() -> None:
    result = ask(
        "ignored",
        executable=[sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_s=0.1,
    )

    assert result.timed_out is True
    assert result.returncode != 0
    assert "timed_out=True" in repr(result)


def test_runtime_subagent_accepts_model_level_alias() -> None:
    result = ask(
        "ignored",
        model_level="small",
        executable=[sys.executable, "-c", "import sys; print(' '.join(sys.argv[1:]))"],
        check=True,
    )

    assert "--level small ask ignored" in result.text


def test_runtime_subagent_default_executable_disables_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run_process_text(args: list[str], **kwargs: Any) -> CommandTextResult:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return CommandTextResult(args=args, returncode=0, stdout="final\n", stderr="[subagent-thread] thr_child\n")

    monkeypatch.setattr("uv_agent_runtime.subagent.run_process_text", fake_run_process_text)

    result = ask("inspect", timeout_s=12, retain=False)

    assert result.text == "final"
    assert result.thread_id == "thr_child"
    assert "--no-stream" in captured["args"]
    assert captured["args"].index("--no-stream") < captured["args"].index("ask")
    assert captured["kwargs"]["timeout_s"] == 12


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
    captured = capsys.readouterr()

    assert result.text == "done"
    assert result.thread_id == "thr_child"
    assert captured.out == ""


def test_extract_subagent_thread_id_from_stderr() -> None:
    assert _extract_subagent_thread_id("noise\n[subagent-thread] thr_123\n") == "thr_123"


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

def test_runtime_goal_paths_uses_runner_thread_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uv_agent_runtime import goal_paths

    monkeypatch.setenv("UV_AGENT_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("UV_AGENT_RUNTIME_THREAD_ID", "thr_goal")

    paths = goal_paths()

    assert paths.directory == tmp_path / "goals" / "thr_goal"
    assert paths.state == paths.directory / "goal.json"
    assert paths.checklist == paths.directory / "checklist.md"
    assert paths.notes == paths.directory / "notes.md"


def test_runtime_goal_paths_requires_thread_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from uv_agent_runtime import goal_paths

    monkeypatch.delenv("UV_AGENT_RUNTIME_STATE_DIR", raising=False)
    monkeypatch.delenv("UV_AGENT_RUNTIME_THREAD_ID", raising=False)

    with pytest.raises(RuntimeError, match="goal_paths requires"):
        goal_paths()



def test_runtime_list_thread_digests_filters_subagents(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    parent = store.create_thread("Parent")
    child = store.create_thread(
        "Subagent",
        kind="subagent",
        parent_thread_id=parent,
        parent_turn_id="turn_parent",
        parent_run_id="run_parent",
    )
    store.append(
        child,
        "item.user",
        turn_id="turn_child",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "child work"}]},
    )

    digests = list_thread_digests(state_dir=tmp_path, kind="subagent", parent_thread_id=parent)

    assert [digest["thread_id"] for digest in digests] == [child]
    assert digests[0]["items"] == [{"role": "user", "text": "child work"}]


def test_runtime_thread_helpers_do_not_create_legacy_thread_directories(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    parent = store.create_thread("Parent")
    child = store.create_thread("Subagent", kind="subagent", parent_thread_id=parent)
    store.append(
        child,
        "item.user",
        turn_id="turn_child",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "child work"}]},
    )

    assert not (tmp_path / "threads").exists()
    assert not (tmp_path / "subthreads").exists()

    assert thread_digest(parent, state_dir=tmp_path)["thread_id"] == parent
    assert list_thread_digests(state_dir=tmp_path, kind="subagent", parent_thread_id=parent)[0][
        "thread_id"
    ] == child

    assert not (tmp_path / "threads").exists()
    assert not (tmp_path / "subthreads").exists()


def test_runtime_look_at_returns_structured_event(tmp_path: Path, capsys) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    event = look_at(image, note="inspect")
    captured = capsys.readouterr()

    assert event["kind"] == "look_at"
    assert event["path"] == str(image.resolve())
    assert event["note"] == "inspect"
    assert event["_uv_agent_event_id"].startswith("evt_")
    assert captured.out == ""


def test_runtime_emit_helpers_return_event_dict(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_AGENT_RUNTIME_RUN_ID", "run_events")

    custom = emit_event("custom", value=1)
    progress = emit_progress("working", count=2)
    result = emit_result(ok=True)
    captured = capsys.readouterr()

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
    assert captured.out == ""


def test_runtime_emit_event_is_thread_safe_without_stdout(
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

    captured = capsys.readouterr()

    assert captured.out == ""


# ---- codesearch / codequery -----------------------------------------------

requires_rg = pytest.mark.skipif(
    shutil.which("rg") is None,
    reason="ripgrep (`rg`) not on PATH",
)


@pytest.fixture
def codequery_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("UV_AGENT_HOME", str(home))
    # Reset in-process LRU caches so language/parser/query objects are rebuilt
    # against the freshly created on-disk cache directory.
    from uv_agent_runtime import codequery

    codequery._language.cache_clear()
    codequery._parser.cache_clear()
    codequery._query.cache_clear()
    yield home


def _make_python_workspace(root: Path) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "a.py").write_text(
        "def hello():\n    return 1\n\nclass Foo:\n    def bar(self):\n        return hello()\n",
        encoding="utf-8",
    )
    (root / "src" / "b.py").write_text(
        "def world():\n    return 42\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# project\n", encoding="utf-8")


@requires_rg
def test_codesearch_find_files_and_search_text(tmp_path: Path) -> None:
    _make_python_workspace(tmp_path)

    files = find_files(tmp_path, globs=["*.py"])
    assert sorted(Path(p).relative_to(tmp_path).as_posix() for p in files) == ["src/a.py", "src/b.py"]
    assert all(Path(p).is_absolute() for p in files)

    limited_files = find_files(tmp_path, globs=["*.py"], max_total=1)
    assert len(limited_files) == 1
    assert Path(limited_files[0]).is_absolute()

    hits = search_text("hello", root=tmp_path, file_types=["py"], context=1)
    paths = [h.rel_path.replace("\\", "/") for h in hits]
    lines = sorted((p, h.line) for p, h in zip(paths, hits))
    assert ("src/a.py", 1) in lines
    assert ("src/a.py", 6) in lines
    assert all(Path(h.path).is_absolute() for h in hits)
    assert all(h.text and h.submatches for h in hits)
    assert hits[0].context_after


@requires_rg
def test_codesearch_accepts_file_root(tmp_path: Path) -> None:
    _make_python_workspace(tmp_path)
    target = tmp_path / "src" / "a.py"

    files = find_files(target)
    assert [Path(p).name for p in files] == ["a.py"]
    assert files == [str(target.resolve())]

    hits = search_text("hello", root=target)
    assert hits
    assert {h.rel_path.replace("\\", "/") for h in hits} == {"a.py"}
    assert {h.path for h in hits} == {str(target.resolve())}

    # Searching a file should not pick up unrelated matches in sibling files.
    world_hits = search_text("world", root=target)
    assert world_hits == []


@requires_rg
def test_codesearch_accepts_multiple_roots(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _make_python_workspace(first)
    _make_python_workspace(second)

    files = find_files(roots=[first / "src" / "a.py", second / "src"], globs=["*.py"])
    assert sorted(Path(p).name for p in files) == ["a.py", "a.py", "b.py"]
    assert all(Path(p).is_absolute() for p in files)

    hits = search_text("hello", roots=[first / "src" / "a.py", second / "src"])
    rel_paths = [h.rel_path.replace("\\", "/") for h in hits]
    assert "a.py" in rel_paths
    assert "b.py" not in rel_paths
    assert all(Path(h.path).is_absolute() for h in hits)


@requires_rg
def test_codesearch_multiple_roots_share_max_total(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _make_python_workspace(first)
    _make_python_workspace(second)

    files = find_files(roots=[first, second], globs=["*.py"], max_total=2)
    hits = search_text("def ", roots=[first, second], fixed_string=True, max_total=2)

    assert len(files) == 2
    assert len(hits) == 2


@requires_rg
def test_codesearch_roots_accepts_single_path_string(tmp_path: Path) -> None:
    _make_python_workspace(tmp_path)

    files = find_files(roots=str(tmp_path / "src" / "a.py"))

    assert files == [str((tmp_path / "src" / "a.py").resolve())]


def test_codesearch_root_and_roots_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        search_text("hello", root=tmp_path, roots=[tmp_path])


@requires_rg
def test_codequery_accepts_file_root(
    tmp_path: Path,
    codequery_home: Path,
) -> None:
    _make_python_workspace(tmp_path)
    target = tmp_path / "src" / "a.py"

    symbols = find_symbols(target)
    names = {(s.kind, s.name, s.rel_path.replace("\\", "/")) for s in symbols}
    assert ("function", "hello", "a.py") in names
    assert ("class", "Foo", "a.py") in names
    assert all(Path(s.path).is_absolute() for s in symbols)
    assert all(s.start_line >= 1 and s.end_line >= s.start_line for s in symbols)
    # b.py's `world` must not appear when scoping to a.py.
    assert all(s.name != "world" for s in symbols)

    captures = query_code(
        "(call function: (identifier) @call)",
        language="python",
        root=target,
    )
    assert [c.text for c in captures] == ["hello"]
    assert {c.rel_path.replace("\\", "/") for c in captures} == {"a.py"}
    assert {c.path for c in captures} == {str(target.resolve())}


@requires_rg
def test_codesearch_search_text_fixed_string_and_max_total(tmp_path: Path) -> None:
    _make_python_workspace(tmp_path)
    hits = search_text("def ", root=tmp_path, fixed_string=True, max_total=2)
    assert len(hits) == 2
    for hit in hits:
        assert hit.line >= 1
        assert hit.submatches[0].text == "def "


@requires_rg
def test_codesearch_search_text_accepts_literal_and_case_sensitive_aliases(tmp_path: Path) -> None:
    _make_python_workspace(tmp_path)

    literal_hits = search_text("hello(", root=tmp_path, literal=True, max_total=1)
    case_hits = search_text("HELLO", root=tmp_path, case_sensitive=False, max_total=1)

    assert literal_hits[0].submatches[0].text == "hello("
    assert case_hits[0].submatches[0].text == "hello"


@requires_rg
def test_codesearch_accepts_scalar_filter_arguments(tmp_path: Path) -> None:
    _make_python_workspace(tmp_path)

    files = find_files(tmp_path, globs="*.py", file_types="py", max_total=1)
    hits = search_text("hello", root=tmp_path, globs="*.py", file_types="py", max_total=1)

    assert len(files) == 1
    assert len(hits) == 1


def test_codesearch_file_types_rejects_extension_patterns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="file_types uses ripgrep type aliases"):
        find_files(tmp_path, file_types=".py")


@requires_rg
def test_codesearch_regex_error_suggests_literal_search(tmp_path: Path) -> None:
    _make_python_workspace(tmp_path)

    with pytest.raises(RuntimeError, match="literal=True"):
        search_text("hello(", root=tmp_path)

@requires_rg
def test_codequery_supported_languages_includes_python() -> None:
    langs = supported_symbol_languages()
    assert "python" in langs
    assert "rust" in langs


@requires_rg
def test_codequery_find_symbols_returns_python_definitions(
    tmp_path: Path,
    codequery_home: Path,
) -> None:
    _make_python_workspace(tmp_path)

    symbols = find_symbols(tmp_path)
    names = {(s.kind, s.name, s.rel_path.replace("\\", "/")) for s in symbols}
    assert ("function", "hello", "src/a.py") in names
    assert ("class", "Foo", "src/a.py") in names
    assert ("function", "bar", "src/a.py") in names
    assert ("function", "world", "src/b.py") in names
    hello = next(s for s in symbols if s.name == "hello")
    assert hello.path == str((tmp_path / "src" / "a.py").resolve())
    assert hello.start_line == 1
    assert hello.end_line == 2

    # Cache was populated on disk under the isolated home.
    assert (codequery_home / "cache" / "codequery" / "index.sqlite").exists()


@requires_rg
def test_codequery_find_symbols_filters_by_kind_and_name(
    tmp_path: Path,
    codequery_home: Path,
) -> None:
    _make_python_workspace(tmp_path)
    only_class = find_symbols(tmp_path, kinds=["class"])
    assert [s.name for s in only_class] == ["Foo"]

    named = find_symbols(tmp_path, name_pattern=r"^h")
    assert [s.name for s in named] == ["hello"]

    exact = find_symbols(tmp_path, kind="function", name="world")
    assert [s.name for s in exact] == ["world"]

    contained = find_symbols(tmp_path, language="python", contains="oo")
    assert [s.name for s in contained] == ["Foo"]


    scalar_filters = find_symbols(tmp_path, languages="python", kinds="class")
    assert [s.name for s in scalar_filters] == ["Foo"]

@requires_rg
def test_codequery_query_code_runs_arbitrary_tree_sitter_query(
    tmp_path: Path,
    codequery_home: Path,
) -> None:
    _make_python_workspace(tmp_path)
    captures = query_code(
        "(call function: (identifier) @call)",
        language="python",
        root=tmp_path,
        globs="*.py",
    )
    assert len(captures) == 1
    cap = captures[0]
    assert cap.name == "call"
    assert cap.text == "hello"
    assert cap.rel_path.replace("\\", "/") == "src/a.py"
    assert cap.path == str((tmp_path / "src" / "a.py").resolve())
    assert cap.start_line == 6


@requires_rg
def test_codequery_cache_is_incremental(
    tmp_path: Path,
    codequery_home: Path,
) -> None:
    import sqlite3

    _make_python_workspace(tmp_path)
    find_symbols(tmp_path)
    db = codequery_home / "cache" / "codequery" / "index.sqlite"

    def read_stats() -> dict[str, tuple[int, int]]:
        conn = sqlite3.connect(db)
        try:
            return {
                rel: (mtime, size)
                for rel, mtime, size in conn.execute(
                    "SELECT rel_path, mtime_ns, size FROM files"
                )
            }
        finally:
            conn.close()

    before = read_stats()
    assert before, "cache should contain at least one file row"

    # Re-running without changes must reuse cached rows verbatim.
    find_symbols(tmp_path)
    assert read_stats() == before

    # Mutating one file invalidates only that file's row.
    target = tmp_path / "src" / "b.py"
    target.write_text(
        "def world():\n    return 0\n\ndef extra():\n    return None\n",
        encoding="utf-8",
    )
    symbols = find_symbols(tmp_path)
    names = {s.name for s in symbols if s.path.endswith("b.py")}
    assert {"world", "extra"} <= names

    after = read_stats()
    rel_b = next(p for p in after if p.endswith("b.py"))
    rel_a = next(p for p in after if p.endswith("a.py"))
    assert after[rel_b] != before[rel_b]
    assert after[rel_a] == before[rel_a]

    # Scoped cache refreshes intentionally do not prune unrelated rows; explicit
    # clear_cache remains the cleanup mechanism.  The query result still reflects
    # the current filesystem because missing candidate files are not returned.
    target.unlink()
    symbols_after_delete = find_symbols(tmp_path)
    assert all(not (s.path.endswith("b.py") and s.name == "extra") for s in symbols_after_delete)
    assert rel_b in read_stats()


@requires_rg
def test_codequery_scoped_queries_do_not_prune_other_cached_files(
    tmp_path: Path,
    codequery_home: Path,
) -> None:
    import sqlite3

    _make_python_workspace(tmp_path)
    find_symbols(tmp_path)
    db = codequery_home / "cache" / "codequery" / "index.sqlite"

    target = tmp_path / "src" / "a.py"
    find_symbols(target)

    conn = sqlite3.connect(db)
    try:
        cached_paths = {
            rel.replace("\\", "/")
            for (rel,) in conn.execute("SELECT rel_path FROM files")
        }
    finally:
        conn.close()

    assert "a.py" in cached_paths or "src/a.py" in cached_paths
    assert any(path.endswith("b.py") for path in cached_paths)


@requires_rg
def test_codequery_clear_cache_drops_rows(
    tmp_path: Path,
    codequery_home: Path,
) -> None:
    _make_python_workspace(tmp_path)
    find_symbols(tmp_path)
    removed = clear_codequery_cache(root=tmp_path)
    assert removed > 0
    # Second clear has nothing to remove.
    assert clear_codequery_cache(root=tmp_path) == 0
