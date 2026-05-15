from __future__ import annotations

from pathlib import Path

import pytest

from uv_agent.config import RunnerConfig
from uv_agent.jsonl import read_jsonl
from uv_agent.runner import PythonRunRequest, PythonRunner, RerunRequest


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
