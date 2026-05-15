from __future__ import annotations

from pathlib import Path
import asyncio

import pytest

from uv_agent.config import RunnerConfig
from uv_agent.jsonl import read_jsonl
from uv_agent.runner import PythonRunRequest, PythonRunner, RerunRequest
from uv_agent.runner.runner import parse_structured_event
from uv_agent.runner.store import ScriptStore


@pytest.mark.asyncio
async def test_runner_executes_script_and_records_jsonl(tmp_path: Path) -> None:
    project_root = Path.cwd()
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / ".uv-agent",
        config=RunnerConfig(
            runtime_dependency=f"uv-agent @ {project_root.resolve().as_uri()}",
            runtime_package_name="uv-agent",
            default_timeout_s=30,
        ),
    )

    result = await runner.run(
        PythonRunRequest(
            code="from uv_agent_runtime import emit_event\nemit_event('hello', value=42)\n",
            cwd=project_root,
        )
    )

    assert result.returncode == 0
    assert '"kind": "hello"' in result.stdout
    events = read_jsonl(result.run_log_path)
    assert events[0]["type"] == "run.started"
    assert events[-1]["type"] == "run.completed"
    assert events[0]["argv"][0:4] == ["uv", "run", "--reinstall-package", "uv-agent"]
    assert "--with" not in events[0]["argv"]


@pytest.mark.asyncio
async def test_runner_reruns_by_script_id(tmp_path: Path) -> None:
    project_root = Path.cwd()
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / ".uv-agent",
        config=RunnerConfig(
            runtime_dependency=f"uv-agent @ {project_root.resolve().as_uri()}",
            runtime_package_name="uv-agent",
            default_timeout_s=30,
        ),
    )
    first = await runner.run(PythonRunRequest(code="print('again')\n", cwd=project_root))
    second = await runner.rerun(RerunRequest(script_id=first.script_id, cwd=project_root))

    assert second.script_id == first.script_id
    assert second.run_id != first.run_id
    assert second.stdout == first.stdout


@pytest.mark.asyncio
async def test_runner_truncates_large_output(tmp_path: Path) -> None:
    project_root = Path.cwd()
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / ".uv-agent",
        config=RunnerConfig(
            runtime_dependency=f"uv-agent @ {project_root.resolve().as_uri()}",
            runtime_package_name="uv-agent",
            default_timeout_s=30,
            max_output_bytes=10,
        ),
    )
    result = await runner.run(PythonRunRequest(code="print('x' * 100)\n", cwd=project_root))

    assert result.truncated is True
    assert "[uv-agent runner output truncated]" in result.stdout + result.stderr
    assert any(event["type"] == "run.output_truncated" for event in read_jsonl(result.run_log_path))


@pytest.mark.asyncio
async def test_runner_interrupts_script_when_cancelled(tmp_path: Path) -> None:
    project_root = Path.cwd()
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / ".uv-agent",
        config=RunnerConfig(
            runtime_dependency=f"uv-agent @ {project_root.resolve().as_uri()}",
            runtime_package_name="uv-agent",
            default_timeout_s=30,
        ),
    )
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
    assert parse_structured_event('{"kind":"look_at","path":"image.png"}\n') == {
        "kind": "look_at",
        "path": "image.png",
    }
    assert parse_structured_event("plain text\n") is None


@pytest.mark.asyncio
async def test_runner_prunes_old_scripts(tmp_path: Path) -> None:
    project_root = Path.cwd()
    runner = PythonRunner(
        project_root=project_root,
        data_dir=tmp_path / ".uv-agent",
        config=RunnerConfig(
            runtime_dependency=f"uv-agent @ {project_root.resolve().as_uri()}",
            runtime_package_name="uv-agent",
            default_timeout_s=30,
            max_saved_scripts=2,
        ),
    )

    for index in range(3):
        await runner.run(PythonRunRequest(code=f"print({index})\n", cwd=project_root))

    scripts = runner.store.list_scripts()

    assert len(scripts) == 2
    assert all(script["summary"].startswith("print(") for script in scripts)


def test_script_store_summary_ignores_inline_metadata(tmp_path: Path) -> None:
    store = ScriptStore(tmp_path, max_saved_scripts=32)
    script_id, _, _ = store.create_script(
        original_code="",
        final_code="# /// script\n# dependencies=['x']\n# ///\n\nprint('real')\n",
        thread_id=None,
        turn_id=None,
    )

    summaries = store.list_scripts()

    assert summaries[0]["script_id"] == script_id
    assert summaries[0]["summary"] == "print('real')"
