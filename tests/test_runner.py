from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from uv_agent.config import RunnerConfig
from uv_agent.jsonl import read_jsonl
from uv_agent.runner import PythonRunRequest, PythonRunner
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
