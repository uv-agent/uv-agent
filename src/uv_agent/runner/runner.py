from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from uv_agent.config import RunnerConfig
from uv_agent.ids import new_id
from uv_agent.jsonl import JsonlWriter
from uv_agent.runner.metadata import ensure_dependency
from uv_agent.runner.models import PythonRunRequest, PythonRunResult, RerunRequest, RunnerEvent
from uv_agent.runner.store import ScriptStore
from uv_agent.time import utc_now_iso


class PythonRunner:
    def __init__(self, *, project_root: Path, data_dir: Path, config: RunnerConfig) -> None:
        self.project_root = project_root.resolve()
        self.config = config
        self.store = ScriptStore(data_dir)

    async def run(self, request: PythonRunRequest) -> PythonRunResult:
        events: list[RunnerEvent] = []
        async for event in self.stream_run(request):
            events.append(event)
        completed = next(
            (event for event in reversed(events) if event.type == "run.completed"),
            None,
        )
        if completed is None:
            raise RuntimeError("Runner did not emit run.completed")
        return completed.data["result"]

    async def stream_run(self, request: PythonRunRequest) -> AsyncIterator[RunnerEvent]:
        timeout_s = request.timeout_s or self.config.default_timeout_s
        final_code = ensure_dependency(
            request.code,
            self.config.runtime_dependency,
            self.config.runtime_package_name,
        )
        script_id, original_path, final_path = self.store.create_script(
            original_code=request.code,
            final_code=final_code,
            thread_id=request.thread_id,
            turn_id=request.turn_id,
        )
        async for event in self._execute_saved_script(
            script_id=script_id,
            original_path=original_path,
            final_path=final_path,
            uv_args=self._merged_uv_args(request.uv_args),
            script_args=request.script_args,
            cwd=request.cwd,
            timeout_s=timeout_s,
            thread_id=request.thread_id,
            turn_id=request.turn_id,
        ):
            yield event

    async def rerun(self, request: RerunRequest) -> PythonRunResult:
        events: list[RunnerEvent] = []
        async for event in self.stream_rerun(request):
            events.append(event)
        completed = next(
            (event for event in reversed(events) if event.type == "run.completed"),
            None,
        )
        if completed is None:
            raise RuntimeError("Runner did not emit run.completed")
        return completed.data["result"]

    async def stream_rerun(self, request: RerunRequest) -> AsyncIterator[RunnerEvent]:
        if not request.script_id and not request.run_id:
            raise ValueError("RerunRequest requires script_id or run_id")

        inherited: dict[str, Any] = {}
        script_id = request.script_id
        if request.run_id:
            inherited = self.store.find_run(request.run_id)
            script_id = script_id or inherited["script_id"]
        if script_id is None:
            raise ValueError("Unable to resolve script_id for rerun")

        metadata = self.store.get_script(script_id)
        final_path = Path(metadata["final_path"])
        original_path = Path(metadata["original_path"])
        if request.mode == "replay":
            uv_args = request.uv_args if request.uv_args is not None else inherited.get("uv_args", [])
            script_args = (
                request.script_args
                if request.script_args is not None
                else inherited.get("script_args", [])
            )
            cwd = request.cwd or (Path(inherited["cwd"]) if inherited.get("cwd") else None)
            timeout_s = request.timeout_s or inherited.get("timeout_s") or self.config.default_timeout_s
        else:
            uv_args = request.uv_args or []
            script_args = request.script_args or []
            cwd = request.cwd
            timeout_s = request.timeout_s or self.config.default_timeout_s

        async for event in self._execute_saved_script(
            script_id=script_id,
            original_path=original_path,
            final_path=final_path,
            uv_args=self._merged_uv_args(list(uv_args)),
            script_args=list(script_args),
            cwd=cwd,
            timeout_s=float(timeout_s),
            thread_id=request.thread_id,
            turn_id=request.turn_id,
            rerun_of=request.run_id,
        ):
            yield event

    async def _execute_saved_script(
        self,
        *,
        script_id: str,
        original_path: Path,
        final_path: Path,
        uv_args: list[str],
        script_args: list[str],
        cwd: Path | None,
        timeout_s: float,
        thread_id: str | None,
        turn_id: str | None,
        rerun_of: str | None = None,
    ) -> AsyncIterator[RunnerEvent]:
        run_id = new_id("run")
        run_log_path = self.store.create_run_log_path(run_id)
        writer = JsonlWriter(run_log_path)
        run_cwd = (cwd or self.project_root).resolve()
        argv = ["uv", "run", *uv_args, str(final_path), *script_args]
        started = {
            "type": "run.started",
            "created_at": utc_now_iso(),
            "run_id": run_id,
            "script_id": script_id,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "rerun_of": rerun_of,
            "cwd": str(run_cwd),
            "timeout_s": timeout_s,
            "uv_args": uv_args,
            "script_args": script_args,
            "argv": argv,
            "original_script_path": str(original_path),
            "final_script_path": str(final_path),
        }
        writer.write(started)
        yield RunnerEvent("run.started", started)

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        structured_events: list[dict[str, Any]] = []
        byte_count = {"value": 0}
        truncated = {"value": False}
        returncode: int | None = None
        timed_out = False
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(run_cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_task = asyncio.create_task(
                self._pump_stream(
                    "stdout",
                    process.stdout,
                    writer,
                    stdout_parts,
                    structured_events,
                    run_id,
                    script_id,
                    byte_count,
                    truncated,
                )
            )
            stderr_task = asyncio.create_task(
                self._pump_stream(
                    "stderr",
                    process.stderr,
                    writer,
                    stderr_parts,
                    structured_events,
                    run_id,
                    script_id,
                    byte_count,
                    truncated,
                )
            )
            try:
                returncode = await asyncio.wait_for(process.wait(), timeout=timeout_s)
            except TimeoutError:
                timed_out = True
                process.kill()
                returncode = await process.wait()
            await asyncio.gather(stdout_task, stderr_task)
        except Exception as exc:
            failed = {
                "type": "run.failed",
                "created_at": utc_now_iso(),
                "run_id": run_id,
                "script_id": script_id,
                "error": repr(exc),
            }
            writer.write(failed)
            yield RunnerEvent("run.failed", failed)
            raise

        result = PythonRunResult(
            script_id=script_id,
            run_id=run_id,
            returncode=returncode,
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            timed_out=timed_out,
            truncated=truncated["value"],
            run_log_path=run_log_path,
            script_path=original_path,
            final_script_path=final_path,
            events=structured_events,
        )
        completed = {
            "type": "run.completed",
            "created_at": utc_now_iso(),
            "run_id": run_id,
            "script_id": script_id,
            "returncode": returncode,
            "timed_out": timed_out,
            "truncated": truncated["value"],
        }
        writer.write(completed)
        yield RunnerEvent("run.completed", {**completed, "result": result})

    async def _pump_stream(
        self,
        stream_name: str,
        stream: asyncio.StreamReader | None,
        writer: JsonlWriter,
        sink: list[str],
        structured_events: list[dict[str, Any]],
        run_id: str,
        script_id: str,
        byte_count: dict[str, int],
        truncated: dict[str, bool],
    ) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.readline()
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            byte_count["value"] += len(chunk)
            if byte_count["value"] > self.config.max_output_bytes:
                if not truncated["value"]:
                    truncated["value"] = True
                    marker = "\n[uv-agent runner output truncated]\n"
                    sink.append(marker)
                    writer.write(
                        {
                            "type": "run.output_truncated",
                            "created_at": utc_now_iso(),
                            "run_id": run_id,
                            "script_id": script_id,
                            "max_output_bytes": self.config.max_output_bytes,
                        }
                    )
                continue
            sink.append(text)
            if stream_name == "stdout":
                parsed = parse_structured_event(text)
                if parsed is not None:
                    structured_events.append(parsed)
            writer.write(
                {
                    "type": f"run.{stream_name}",
                    "created_at": utc_now_iso(),
                    "run_id": run_id,
                    "script_id": script_id,
                    "text": text,
                }
            )

    def _merged_uv_args(self, uv_args: list[str]) -> list[str]:
        merged = list(self.config.default_uv_args)
        for arg in uv_args:
            if arg not in merged:
                merged.append(arg)
        return merged


def parse_structured_event(text: str) -> dict[str, Any] | None:
    """Parse one uv_agent_runtime.emit_event JSON line if present."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        value = json.loads(stripped)
    except Exception:
        return None
    if not isinstance(value, dict) or "kind" not in value:
        return None
    return value
