from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from time import monotonic

from uv_agent.config import RunnerConfig
from uv_agent.ids import new_id
from uv_agent.runner.models import PythonRunRequest, PythonRunResult, RunnerEvent
from uv_agent.runner.output import OutputCapture, pump_stream
from uv_agent.runner.process import kill_process_tree, subprocess_group_kwargs
from uv_agent.runner.rpc import RuntimeRPCServer
from uv_agent.runner.run_log import RunLogStore
from uv_agent.runner.scriptenv import ensure_venv, uv_binary
from uv_agent.time import utc_now_iso


def _deadline_expired(started_monotonic: float, timeout_s: float | None) -> bool:
    """Return True once a run-level timeout has elapsed."""

    return timeout_s is not None and monotonic() - started_monotonic >= timeout_s


class PythonRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        data_dir: Path,
        config: RunnerConfig,
        runs_dir: Path | None = None,
        scriptenv_dir: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.data_dir = data_dir.resolve()
        # ``runs_dir`` is retained as an optional debug script export location;
        # run code/events are stored in the project SQLite database.
        self.runs_dir = (runs_dir or self.data_dir / "runner" / "scripts").resolve()
        self.scriptenv_dir = (scriptenv_dir or self.data_dir / "runner" / "scriptenv").resolve()
        self.config = config
        self.run_logs = RunLogStore(self.data_dir, scripts_dir=self.runs_dir, max_run_logs=config.max_run_logs)
        self.rpc_server = RuntimeRPCServer()

    @property
    def config(self) -> RunnerConfig:
        return self._config

    @config.setter
    def config(self, value: RunnerConfig) -> None:
        self._config = value
        if hasattr(self, "run_logs"):
            self.run_logs.max_run_logs = max(1, value.max_run_logs)
            self.run_logs.prune()

    def close(self) -> None:
        """Stop long-lived runner resources such as the runtime RPC server."""

        self.rpc_server.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    async def run(self, request: PythonRunRequest) -> PythonRunResult:
        completed: RunnerEvent | None = None
        async for event in self.stream_run(request):
            if event.type == "run.completed":
                completed = event
        if completed is None:
            raise RuntimeError("Runner did not emit run.completed")
        return completed.data["result"]

    async def stream_run(self, request: PythonRunRequest) -> AsyncIterator[RunnerEvent]:
        run_id = new_id("run")
        timeout_s = request.timeout_s or self.config.default_timeout_s
        await asyncio.to_thread(
            ensure_venv,
            self.scriptenv_dir,
            index_url=self.config.scriptenv_index_url,
        )
        run_cwd = (request.cwd or self.project_root).resolve()
        started_at = utc_now_iso()
        script_path = await asyncio.to_thread(
            self.run_logs.create_run_record,
            run_id=run_id,
            code=request.code,
            script_args=list(request.script_args),
            cwd=run_cwd,
            timeout_s=timeout_s,
            started_at=started_at,
            thread_id=request.thread_id,
            turn_id=request.turn_id,
            script_path=None,
        )
        writer = self.run_logs.writer(run_id)
        argv = [
            uv_binary(),
            "run",
            "--project",
            str(self.scriptenv_dir),
            "--frozen",
            "--directory",
            str(run_cwd),
            "python",
            str(script_path),
            *request.script_args,
        ]
        env = self._run_env(
            run_id=run_id,
            thread_id=request.thread_id,
            thread_kind=request.thread_kind,
            turn_id=request.turn_id,
        )
        started = {
            "type": "run.started",
            "created_at": started_at,
            "run_id": run_id,
            "thread_id": request.thread_id,
            "turn_id": request.turn_id,
            "cwd": str(run_cwd),
            "timeout_s": timeout_s,
            "script_args": request.script_args,
            "argv": argv,
            "script_path": str(script_path),
        }
        writer.write(started)
        yield RunnerEvent("run.started", started)

        capture = OutputCapture()
        rpc_session = self.rpc_server.open_session(
            run_id=run_id,
            thread_id=request.thread_id,
            turn_id=request.turn_id,
            cwd=run_cwd,
            on_structured_event=capture.append_structured_event,
            writer=writer,
        )
        env["UV_AGENT_RPC_URL"] = self.rpc_server.url
        env["UV_AGENT_RPC_TOKEN"] = rpc_session.token
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
                    run_id=run_id,
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
                    run_id=run_id,
                    max_output_bytes=self.config.max_output_bytes,
                    capture=capture,
                )
            )
            started_monotonic = monotonic()
            wait_task = asyncio.create_task(process.wait())
            cancel_task = (
                asyncio.create_task(request.cancel_event.wait())
                if request.cancel_event is not None
                else None
            )
            tasks = {wait_task}
            if cancel_task is not None:
                tasks.add(cancel_task)
            last_partial_signature: tuple[int, int, int, int, bool] | None = None

            def partial_signature() -> tuple[int, int, int, int, bool]:
                """Compactly detect whether captured output changed."""

                return (
                    len(capture.stdout_parts),
                    len(capture.stderr_parts),
                    capture.byte_count,
                    len(capture.structured_events),
                    capture.truncated,
                )

            async def current_result() -> PythonRunResult:
                """Snapshot the output captured so far without stopping the run."""

                return PythonRunResult(
                    run_id=run_id,
                    returncode=returncode,
                    stdout="".join(capture.stdout_parts),
                    stderr="".join(capture.stderr_parts),
                    timed_out=timed_out,
                    interrupted=interrupted,
                    truncated=capture.truncated,
                    script_path=script_path,
                    events=list(capture.structured_events),
                )

            async def emit_partial(*, reason: str, force: bool = False) -> RunnerEvent | None:
                """Create a bounded progress event for consumers while running.

                The output pumps append complete decoded chunks to ``capture`` as
                the child process writes. Yielding a fresh snapshot lets the TUI
                and CLI expose that buffered output before the process exits,
                while the final ``run.completed`` payload remains the source of
                truth for the model. Interval events are suppressed when nothing
                changed so long blocked commands do not accumulate duplicate
                megabyte-sized snapshots in memory.
                """

                nonlocal last_partial_signature
                signature = partial_signature()
                if not force and signature == last_partial_signature:
                    return None
                last_partial_signature = signature
                data = {
                    "type": "run.partial",
                    "created_at": utc_now_iso(),
                    "run_id": run_id,
                    "reason": reason,
                    "result": await current_result(),
                }
                return RunnerEvent("run.partial", data)

            while True:
                interval = min(1.0, timeout_s) if timeout_s is not None else 1.0
                if timeout_s is not None:
                    remaining = timeout_s - (monotonic() - started_monotonic)
                    interval = max(0.0, min(interval, remaining))
                done, _ = await asyncio.wait(
                    tasks,
                    timeout=interval,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if wait_task in done:
                    returncode = wait_task.result()
                    break
                if cancel_task is not None and cancel_task in done:
                    interrupted = True
                    partial_event = await emit_partial(reason="interrupted", force=True)
                    if partial_event is not None:
                        yield partial_event
                    await kill_process_tree(process)
                    returncode = await wait_task
                    break
                timed_out = _deadline_expired(started_monotonic, timeout_s)
                if timed_out:
                    partial_event = await emit_partial(reason="timeout", force=True)
                    if partial_event is not None:
                        yield partial_event
                    await kill_process_tree(process)
                    returncode = await wait_task
                    break
                partial_event = await emit_partial(reason="interval")
                if partial_event is not None:
                    yield partial_event
            for task in tasks:
                if task is not wait_task and not task.done():
                    task.cancel()
            cancellable = [task for task in tasks if task is not wait_task and not task.done()]
            if cancellable:
                await asyncio.gather(*cancellable, return_exceptions=True)
            await asyncio.gather(stdout_task, stderr_task)
        except asyncio.CancelledError:
            if process is not None:
                await kill_process_tree(process)
            raise
        except Exception as exc:
            failed = {
                "type": "run.failed",
                "created_at": utc_now_iso(),
                "run_id": run_id,
                "error": repr(exc),
            }
            writer.write(failed)
            yield RunnerEvent("run.failed", failed)
            raise
        finally:
            rpc_session.close()

        result = PythonRunResult(
            run_id=run_id,
            returncode=returncode,
            stdout="".join(capture.stdout_parts),
            stderr="".join(capture.stderr_parts),
            timed_out=timed_out,
            interrupted=interrupted,
            truncated=capture.truncated,
            script_path=script_path,
            events=capture.structured_events,
        )
        completed_at = utc_now_iso()
        self.run_logs.complete_run(
            run_id=run_id,
            completed_at=completed_at,
            returncode=returncode,
            timed_out=timed_out,
            interrupted=interrupted,
            truncated=capture.truncated,
            stdout=result.stdout,
            stderr=result.stderr,
            structured_events=capture.structured_events,
        )
        completed = {
            "type": "run.completed",
            "created_at": completed_at,
            "run_id": run_id,
            "returncode": returncode,
            "timed_out": timed_out,
            "interrupted": interrupted,
            "truncated": capture.truncated,
        }
        writer.write(completed)
        self.run_logs.prune()
        yield RunnerEvent("run.completed", {**completed, "result": result})

    def _run_env(
        self,
        *,
        run_id: str,
        thread_id: str | None,
        thread_kind: str | None,
        turn_id: str | None,
    ) -> dict[str, str]:
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["UV_AGENT_RUNTIME_PROJECT_ROOT"] = str(self.project_root)
        env["UV_AGENT_RUNTIME_STATE_DIR"] = str(self.data_dir)
        env["UV_AGENT_SCRIPTENV_DIR"] = str(self.scriptenv_dir)
        env["UV_AGENT_SCRIPT_DIR"] = str(self.runs_dir)
        env["UV_BIN"] = uv_binary()
        if thread_id:
            env["UV_AGENT_RUNTIME_THREAD_ID"] = thread_id
        if thread_kind:
            env["UV_AGENT_RUNTIME_THREAD_KIND"] = thread_kind
        if turn_id:
            env["UV_AGENT_RUNTIME_TURN_ID"] = turn_id
        env["UV_AGENT_RUNTIME_RUN_ID"] = run_id
        return env
