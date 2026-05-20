from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from pathlib import Path
from typing import Any


def uv_run_argv(uv_args: list[str], script_path: Path, script_args: list[str]) -> list[str]:
    effective_uv_args = list(uv_args)
    if not has_uv_log_level_arg(effective_uv_args):
        effective_uv_args.insert(0, "--quiet")
    return ["uv", "run", *effective_uv_args, str(script_path), *script_args]


def subprocess_group_kwargs() -> dict[str, Any]:
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


async def kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    elif os.name == "nt":
        try:
            taskkill = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await taskkill.wait()
        except OSError:
            process.kill()
    else:
        process.kill()


def has_uv_log_level_arg(args: list[str]) -> bool:
    for arg in args:
        if arg in {"--quiet", "--verbose"}:
            return True
        if arg.startswith("-") and not arg.startswith("--") and set(arg[1:]) <= {"q", "v"}:
            return True
    return False
