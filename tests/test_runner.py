from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from uv_agent.config import RunnerConfig
from uv_agent.jsonl import read_jsonl
from uv_agent.runner import PythonRunRequest, PythonRunner
import uv_agent.runner.scriptenv as scriptenv
from uv_agent.runner.runner import parse_structured_event
from uv_agent.runner.scriptenv import direct_dependencies, ensure_venv


def make_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: RunnerConfig | None = None,
) -> PythonRunner:
    monkeypatch.setattr("uv_agent.runner.runner.ensure_venv", lambda _path: Path(sys.executable))
    return PythonRunner(
        project_root=Path.cwd(),
        data_dir=tmp_path / ".uv-agent",
        config=config or RunnerConfig(default_timeout_s=30),
    )


@pytest.mark.asyncio
async def test_runner_executes_script_and_records_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path.cwd()
    runner = make_runner(tmp_path, monkeypatch)

    result = await runner.run(
        PythonRunRequest(
            code="from uv_agent_runtime import emit_event\nemit_event('hello', value=42)\n",
            cwd=project_root,
        )
    )

    assert result.returncode == 0
    assert '"kind": "hello"' in result.stdout
    assert result.script_path.exists()
    events = read_jsonl(result.run_log_path)
    assert events[0]["type"] == "run.started"
    assert events[-1]["type"] == "run.completed"
    assert events[0]["argv"][1:5] == [
        "run",
        "--project",
        str(runner.scriptenv_dir),
        "--directory",
    ]
    assert events[0]["argv"][5] == str(project_root)
    assert events[0]["argv"][6] == "python"
    assert events[0]["script_path"] == str(result.script_path)


@pytest.mark.asyncio
async def test_runner_passes_project_root_without_polluting_parent_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path.cwd()
    runner = make_runner(tmp_path, monkeypatch)

    monkeypatch.delenv("UV_AGENT_RUNTIME_PROJECT_ROOT", raising=False)
    result = await runner.run(
        PythonRunRequest(
            code=(
                "import os\n"
                "print(os.environ['UV_AGENT_RUNTIME_PROJECT_ROOT'])\n"
            ),
            cwd=tmp_path,
        )
    )

    assert result.returncode == 0
    assert result.stdout.strip() == str(project_root.resolve())
    assert "UV_AGENT_RUNTIME_PROJECT_ROOT" not in os.environ


@pytest.mark.asyncio
async def test_runner_truncates_large_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path.cwd()
    runner = make_runner(tmp_path, monkeypatch, config=RunnerConfig(default_timeout_s=30, max_output_bytes=10))
    result = await runner.run(PythonRunRequest(code="print('x' * 100)\n", cwd=project_root))

    assert result.truncated is True
    assert "[uv-agent runner output truncated]" in result.stdout + result.stderr
    assert any(event["type"] == "run.output_truncated" for event in read_jsonl(result.run_log_path))


@pytest.mark.asyncio
async def test_runner_handles_long_single_line_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path.cwd()
    runner = make_runner(tmp_path, monkeypatch, config=RunnerConfig(default_timeout_s=30, max_output_bytes=200_000))

    result = await runner.run(PythonRunRequest(code="print('x' * 70_000)\n", cwd=project_root))

    assert result.returncode == 0
    assert result.truncated is False
    assert result.stdout.rstrip("\r\n") == "x" * 70_000


@pytest.mark.asyncio
async def test_runner_parses_structured_event_across_read_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path.cwd()
    runner = make_runner(tmp_path, monkeypatch, config=RunnerConfig(default_timeout_s=30, max_output_bytes=200_000))

    result = await runner.run(
        PythonRunRequest(
            code=(
                "from uv_agent_runtime import emit_event\n"
                "emit_event('big', value='x' * 70_000)\n"
            ),
            cwd=project_root,
        )
    )

    assert result.returncode == 0
    assert result.events[0]["kind"] == "big"
    assert result.events[0]["value"] == "x" * 70_000


@pytest.mark.asyncio
async def test_runner_does_not_parse_json_in_middle_of_long_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path.cwd()
    runner = make_runner(tmp_path, monkeypatch, config=RunnerConfig(default_timeout_s=30, max_output_bytes=200_000))

    result = await runner.run(
        PythonRunRequest(
            code=(
                "import json\n"
                "import os\n"
                "run_id = os.environ['UV_AGENT_RUNTIME_RUN_ID']\n"
                "event = json.dumps({'kind': 'fake', '_uv_agent_run_id': run_id})\n"
                "print(('x' * 70_000) + event)\n"
            ),
            cwd=project_root,
        )
    )

    assert result.returncode == 0
    assert result.events == []


@pytest.mark.asyncio
async def test_runner_parses_threaded_runtime_events_without_line_interleaving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path.cwd()
    runner = make_runner(tmp_path, monkeypatch)

    result = await runner.run(
        PythonRunRequest(
            code=(
                "import threading\n"
                "from uv_agent_runtime import emit_event\n"
                "def emit_many(worker):\n"
                "    for index in range(25):\n"
                "        emit_event('threaded', worker=worker, index=index)\n"
                "threads = [threading.Thread(target=emit_many, args=(worker,)) for worker in range(4)]\n"
                "for thread in threads:\n"
                "    thread.start()\n"
                "for thread in threads:\n"
                "    thread.join()\n"
            ),
            cwd=project_root,
        )
    )

    event_ids = [event["_uv_agent_event_id"] for event in result.events]
    assert result.returncode == 0
    assert len(result.events) == 100
    assert all(event["kind"] == "threaded" for event in result.events)
    assert all(event["_uv_agent_run_id"] == result.run_id for event in result.events)
    assert len(set(event_ids)) == len(event_ids)


@pytest.mark.asyncio
async def test_runner_interrupts_script_when_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path.cwd()
    runner = make_runner(tmp_path, monkeypatch)
    cancel_event = asyncio.Event()
    task = asyncio.create_task(
        runner.run(
            PythonRunRequest(
                code="import time\nprint('start', flush=True)\ntime.sleep(30)\n",
                cwd=project_root,
                cancel_event=cancel_event,
            )
        )
    )
    await asyncio.sleep(0.5)
    cancel_event.set()

    result = await asyncio.wait_for(task, timeout=10)

    assert result.interrupted is True
    assert result.timed_out is False
    assert any(event.get("interrupted") is True for event in read_jsonl(result.run_log_path))


def test_parse_structured_event_reads_runtime_json_line() -> None:
    assert parse_structured_event(
        '{"kind":"look_at","path":"image.png","_uv_agent_event_id":"evt_1","_uv_agent_run_id":"run_1"}\n',
        run_id="run_1",
    ) == {
        "kind": "look_at",
        "path": "image.png",
        "_uv_agent_event_id": "evt_1",
        "_uv_agent_run_id": "run_1",
    }
    assert (
        parse_structured_event(
            '{"kind":"look_at","path":"image.png","_uv_agent_run_id":"run_1"}\n',
            run_id="run_1",
        )
        is None
    )
    assert parse_structured_event('{"kind":"look_at","path":"image.png"}\n', run_id="run_1") is None
    assert (
        parse_structured_event(
            '{"kind":"look_at","path":"image.png","_uv_agent_event_id":"evt_2","_uv_agent_run_id":"run_other"}\n',
            run_id="run_1",
        )
        is None
    )
    assert parse_structured_event("plain text\n") is None


@pytest.mark.asyncio
async def test_runner_prunes_old_run_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path.cwd()
    runner = make_runner(tmp_path, monkeypatch, config=RunnerConfig(default_timeout_s=30, max_run_logs=2))

    for index in range(3):
        await runner.run(PythonRunRequest(code=f"print({index})\n", cwd=project_root))

    logs = sorted(runner.runs_dir.glob("*.jsonl"))
    scripts = sorted(runner.runs_dir.glob("*.py"))

    assert len(logs) == 2
    assert len(scripts) == 2
    assert {path.stem for path in logs} == {path.stem for path in scripts}


def test_ensure_venv_installs_runtime_package(tmp_path: Path) -> None:
    python = ensure_venv(tmp_path / "scriptenv")

    assert python.exists()
    assert (tmp_path / "scriptenv" / "pyproject.toml").exists()
    assert any(dependency.startswith("uv-agent") for dependency in direct_dependencies(tmp_path / "scriptenv"))
    result = subprocess.run([str(python), "-c", "import uv_agent_runtime"], check=False)
    assert result.returncode == 0


def test_ensure_venv_reuses_ready_environment_without_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scriptenv_dir = tmp_path / "scriptenv"
    python = ensure_venv(scriptenv_dir)

    def fail_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise AssertionError("ready scriptenv should not spawn subprocesses")

    monkeypatch.setattr(scriptenv.subprocess, "run", fail_run)

    assert ensure_venv(scriptenv_dir) == python


def test_ensure_venv_serializes_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scriptenv_dir = tmp_path / "scriptenv"
    monkeypatch.setattr(scriptenv, "_READY_DIRS", set())
    monkeypatch.setattr(scriptenv, "_READY_LOCK", threading.Lock())
    calls: list[list[str]] = []
    python = scriptenv_dir / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(list(args))
        if args[1] == "init":
            (scriptenv_dir / "pyproject.toml").parent.mkdir(parents=True, exist_ok=True)
            (scriptenv_dir / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            return subprocess.CompletedProcess(args, 0)
        if args[1] == "add":
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(args, 0)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(scriptenv.subprocess, "run", fake_run)

    results: list[Path] = []
    threads = [threading.Thread(target=lambda: results.append(ensure_venv(scriptenv_dir))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert set(results) == {python}
    assert [call[1] for call in calls].count("init") == 1
    assert [call[1] for call in calls].count("add") == 1


def test_ensure_venv_pins_pyproject_and_syncs_on_version_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scriptenv_dir = tmp_path / "scriptenv"
    monkeypatch.setattr(scriptenv, "_READY_DIRS", set())
    monkeypatch.setattr(scriptenv, "_READY_LOCK", threading.Lock())
    python = scriptenv_dir / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    pyproject = scriptenv_dir / "pyproject.toml"
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(list(args))
        if args[0] != scriptenv.uv_binary():
            return subprocess.CompletedProcess(args, 0, stdout="0.0.1\n", stderr="")
        if args[1] == "init":
            scriptenv_dir.mkdir(parents=True, exist_ok=True)
            pyproject.write_text(
                '[project]\nname = "x"\ndependencies = [\n    "uv-agent",\n]\n',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args, 0)
        if args[1] == "add":
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(args, 0)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(scriptenv.subprocess, "run", fake_run)
    monkeypatch.setattr(scriptenv, "_host_runtime_version", lambda: "9.9.9")

    ensure_venv(scriptenv_dir)

    assert '"uv-agent==9.9.9"' in pyproject.read_text(encoding="utf-8")
    sync_calls = [call for call in calls if call[0] == scriptenv.uv_binary() and call[1] == "sync"]
    assert sync_calls, "expected uv sync to be invoked after pinning the version"


def test_ensure_venv_skips_sync_when_versions_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scriptenv_dir = tmp_path / "scriptenv"
    monkeypatch.setattr(scriptenv, "_READY_DIRS", set())
    monkeypatch.setattr(scriptenv, "_READY_LOCK", threading.Lock())
    python = scriptenv_dir / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    pyproject = scriptenv_dir / "pyproject.toml"
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(list(args))
        if args[0] != scriptenv.uv_binary():
            return subprocess.CompletedProcess(args, 0, stdout="9.9.9\n", stderr="")
        if args[1] == "init":
            scriptenv_dir.mkdir(parents=True, exist_ok=True)
            pyproject.write_text(
                '[project]\nname = "x"\ndependencies = [\n    "uv-agent",\n]\n',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args, 0)
        if args[1] == "add":
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(args, 0)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(scriptenv.subprocess, "run", fake_run)
    monkeypatch.setattr(scriptenv, "_host_runtime_version", lambda: "9.9.9")

    ensure_venv(scriptenv_dir)

    assert '"uv-agent==' not in pyproject.read_text(encoding="utf-8")
    assert not [call for call in calls if call[0] == scriptenv.uv_binary() and call[1] == "sync"]


def test_ensure_venv_restores_pyproject_when_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scriptenv_dir = tmp_path / "scriptenv"
    monkeypatch.setattr(scriptenv, "_READY_DIRS", set())
    monkeypatch.setattr(scriptenv, "_READY_LOCK", threading.Lock())
    python = scriptenv_dir / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    pyproject = scriptenv_dir / "pyproject.toml"
    original_pyproject = '[project]\nname = "x"\ndependencies = [\n    "uv-agent",\n]\n'

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if args[0] != scriptenv.uv_binary():
            return subprocess.CompletedProcess(args, 0, stdout="0.0.1\n", stderr="")
        if args[1] == "init":
            scriptenv_dir.mkdir(parents=True, exist_ok=True)
            pyproject.write_text(original_pyproject, encoding="utf-8")
            return subprocess.CompletedProcess(args, 0)
        if args[1] == "add":
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(args, 0)
        if args[1] == "sync":
            return subprocess.CompletedProcess(args, 1)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(scriptenv.subprocess, "run", fake_run)
    monkeypatch.setattr(scriptenv, "_host_runtime_version", lambda: "9.9.9")

    ensure_venv(scriptenv_dir)

    assert pyproject.read_text(encoding="utf-8") == original_pyproject
