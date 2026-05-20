from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from pathlib import Path

import pytest

from uv_agent_runtime import (
    apply_patch,
    apply_patch_any,
    ask,
    clear_codequery_cache,
    compare_text,
    connect_declared,
    connect_named,
    connect_stdio,
    convert_patch,
    emit_event,
    emit_progress,
    emit_result,
    enter_dir,
    find_files,
    find_symbols,
    list_declared_servers,
    list_files,
    look_at,
    make_unified_diff,
    normalize_text,
    path_info,
    query_code,
    read_json,
    read_text,
    read_text_lossless,
    replace_exact,
    restore_snapshot,
    run_process_text,
    saved_scripts,
    search_text,
    snapshot_files,
    supported_symbol_languages,
    thread_digest,
    workspace_transaction,
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


def test_runtime_run_process_text_accepts_env_and_env_patch() -> None:
    code = "import os; print(os.environ.get('UV_AGENT_TEST_VALUE', 'missing'))"

    result = run_process_text(
        [sys.executable, "-c", code],
        env={},
        env_patch={"UV_AGENT_TEST_VALUE": "patched"},
        check=True,
    )

    assert result.stdout.strip() == "patched"


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


def test_runtime_write_text_lossless_without_template_preserves_input_text(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"

    write_text_lossless(path, "first\r\nsecond\r\n")

    assert path.read_bytes() == b"first\r\nsecond\r\n"


def test_runtime_compare_and_normalize_text_helpers() -> None:
    comparison = compare_text("a\r\nb\r\n", "a\nb\n", ignore_eol=True)

    assert comparison.kind == "eol"
    assert normalize_text("a\r\nb", eol="crlf", final_newline=True) == "a\r\nb\r\n"


def test_runtime_replace_exact_reports_context_and_preserves_style(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("first\r\nold\r\nlast\r\n", encoding="utf-8", newline="")

    result = replace_exact(path, "old", "new")

    assert result.replacements == 1
    assert path.read_bytes() == b"first\r\nnew\r\nlast\r\n"
    with pytest.raises(ValueError, match="found 0"):
        replace_exact(path, "missing", "nope")
    with pytest.raises(ValueError, match="old text must not be empty"):
        replace_exact(path, "", "nope")


def test_runtime_replace_exact_preserves_mixed_newlines(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"first\r\nold\nlast\r")

    result = replace_exact(path, "old", "new")

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
    assert sorted(p.replace("\\", "/") for p in files) == ["src/a.py", "src/b.py"]

    hits = search_text("hello", root=tmp_path, file_types=["py"])
    paths = [h.path.replace("\\", "/") for h in hits]
    lines = sorted((p, h.line) for p, h in zip(paths, hits))
    assert ("src/a.py", 1) in lines
    assert ("src/a.py", 6) in lines
    assert all(h.text and h.submatches for h in hits)


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
    names = {(s.kind, s.name, s.path.replace("\\", "/")) for s in symbols}
    assert ("function", "hello", "src/a.py") in names
    assert ("class", "Foo", "src/a.py") in names
    assert ("function", "bar", "src/a.py") in names
    assert ("function", "world", "src/b.py") in names

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
    )
    assert len(captures) == 1
    cap = captures[0]
    assert cap.name == "call"
    assert cap.text == "hello"
    assert cap.path.replace("\\", "/") == "src/a.py"


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

    # Deleting a file prunes the row on the next call.
    target.unlink()
    find_symbols(tmp_path)
    pruned = read_stats()
    assert all(not p.endswith("b.py") for p in pruned)


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
