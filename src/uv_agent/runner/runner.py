from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from uv_agent.config import RunnerConfig
from uv_agent.ids import new_id
from uv_agent.jsonl import JsonlWriter
from uv_agent.runner.events import parse_structured_event as parse_structured_event
from uv_agent.runner.metadata import ensure_dependency
from uv_agent.runner.models import PythonRunRequest, PythonRunResult, RerunRequest, RunnerEvent
from uv_agent.runner.output import OutputCapture, pump_stream
from uv_agent.runner.process import kill_process_tree, subprocess_group_kwargs, uv_run_argv
from uv_agent.runner.store import ScriptStore
from uv_agent.time import utc_now_iso


class PythonRunner:
    def __init__(self, *, project_root: Path, data_dir: Path, config: RunnerConfig) -> None:
        self.project_root = project_root.resolve()
        self.config = config
        self.store = ScriptStore(data_dir, max_saved_scripts=config.max_saved_scripts)

    @property
    def config(self) -> RunnerConfig:
        return self._config

    @config.setter
    def config(self, value: RunnerConfig) -> None:
        self._config = value
        if hasattr(self, "store"):
            self.store.max_saved_scripts = max(1, value.max_saved_scripts)
            self.store.prune_scripts()

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
            thread_kind=request.thread_kind,
            turn_id=request.turn_id,
            cancel_event=request.cancel_event,
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
            thread_kind=request.thread_kind,
            turn_id=request.turn_id,
            cancel_event=request.cancel_event,
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
        thread_kind: str | None,
        turn_id: str | None,
        cancel_event: asyncio.Event | None,
        rerun_of: str | None = None,
    ) -> AsyncIterator[RunnerEvent]:
        run_id = new_id("run")
        run_log_path = self.store.create_run_log_path(run_id)
        writer = JsonlWriter(run_log_path)
        run_cwd = (cwd or self.project_root).resolve()
        argv = uv_run_argv(uv_args, final_path, script_args)
        env = dict(os.environ)
        # Force UTF-8 for the child Python so non-ASCII stdout/stderr (e.g.
        # Chinese on Windows where the default code page is cp936) round-trips
        # cleanly through our `decode("utf-8")` pump. PYTHONUTF8 also flips the
        # child's `open()` / locale default to UTF-8 so file I/O inside scripts
        # stops depending on the host code page.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["UV_AGENT_RUNTIME_PROJECT_ROOT"] = str(self.project_root)
        env["UV_AGENT_RUNTIME_STATE_DIR"] = str(self.store.data_dir)
        if thread_id:
            env["UV_AGENT_RUNTIME_THREAD_ID"] = thread_id
        if thread_kind:
            env["UV_AGENT_RUNTIME_THREAD_KIND"] = thread_kind
        if turn_id:
            env["UV_AGENT_RUNTIME_TURN_ID"] = turn_id
        env["UV_AGENT_RUNTIME_RUN_ID"] = run_id
        env["UV_AGENT_RUNTIME_SCRIPT_ID"] = script_id
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

        capture = OutputCapture()
        returncode: int | None = None
        timed_out = False
        interrupted = False
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(run_cwd),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **subprocess_group_kwargs(),
            )
            stdout_task = asyncio.create_task(
                pump_stream(
                    stream_name="stdout",
                    stream=process.stdout,
                    writer=writer,
                    sink=capture.stdout_parts,
                    structured_events=capture.structured_events,
                    run_id=run_id,
                    script_id=script_id,
                    max_output_bytes=self.config.max_output_bytes,
                    capture=capture,
                )
            )
            stderr_task = asyncio.create_task(
                pump_stream(
                    stream_name="stderr",
                    stream=process.stderr,
                    writer=writer,
                    sink=capture.stderr_parts,
                    structured_events=capture.structured_events,
                    run_id=run_id,
                    script_id=script_id,
                    max_output_bytes=self.config.max_output_bytes,
                    capture=capture,
                )
            )
            try:
                wait_task = asyncio.create_task(process.wait())
                cancel_task = (
                    asyncio.create_task(cancel_event.wait())
                    if cancel_event is not None
                    else None
                )
                tasks = {wait_task}
                if cancel_task is not None:
                    tasks.add(cancel_task)
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if wait_task in done:
                    returncode = wait_task.result()
                elif cancel_task is not None and cancel_task in done:
                    interrupted = True
                    await kill_process_tree(process)
                    returncode = await wait_task
                else:
                    timed_out = True
                    await kill_process_tree(process)
                    returncode = await wait_task
                for task in tasks:
                    if task is wait_task or task.done():
                        continue
                    task.cancel()
                cancellable = [task for task in tasks if task is not wait_task and not task.done()]
                if cancellable:
                    await asyncio.gather(*cancellable, return_exceptions=True)
            except TimeoutError:
                timed_out = True
                await kill_process_tree(process)
                returncode = await wait_task
            except asyncio.CancelledError:
                await kill_process_tree(process)
                raise
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
            stdout="".join(capture.stdout_parts),
            stderr="".join(capture.stderr_parts),
            timed_out=timed_out,
            interrupted=interrupted,
            truncated=capture.truncated,
            run_log_path=run_log_path,
            script_path=original_path,
            final_script_path=final_path,
            events=capture.structured_events,
        )
        completed = {
            "type": "run.completed",
            "created_at": utc_now_iso(),
            "run_id": run_id,
            "script_id": script_id,
            "returncode": returncode,
            "timed_out": timed_out,
            "interrupted": interrupted,
            "truncated": capture.truncated,
        }
        writer.write(completed)
        yield RunnerEvent("run.completed", {**completed, "result": result})

    def _merged_uv_args(self, uv_args: list[str]) -> list[str]:
        merged = list(self.config.default_uv_args)
        for arg in uv_args:
            if arg not in merged:
                merged.append(arg)
        return merged
