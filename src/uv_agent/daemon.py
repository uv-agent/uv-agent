from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from uv_agent.app_factory import create_engine
from uv_agent.config import load_config
from uv_agent.environment import UserLanguage, detect_user_language
from uv_agent.ids import new_id
from uv_agent.logging_config import active_log_file
from uv_agent.paths import default_daemon_workspace_dir
from uv_agent.state_db import connect_state_db


logger = logging.getLogger(__name__)

DEFAULT_DAEMON_AGENTS_MD_EN = """# uv-agent Daemon Workspace

## Purpose

This is the default persistent workspace for `uv-agent daemon`. Use it for durable notes, plans, references, and generated artifacts that should remain useful across daemon sessions.

## Directory Map

- `AGENTS.md`: Workspace policy, directory map, and maintenance rules.
- `notes/`: User-facing durable notes, TODO lists, reminders, decisions, and references. Use this by default for "remember this" requests unless the user names another path.
- Add entries here when durable top-level directories are created.

## Workspace Maintenance

- Before creating a new top-level file or directory, inspect the existing layout and reuse a suitable location.
- Keep related material together in clearly named subdirectories.
- Prefer `notes/index.md` as the notes map and `notes/inbox.md` for unsorted captures when the notes area grows.
- Prefer short README or index files inside long-lived directories so future agents can understand what belongs there.
- Remove or archive stale outputs when they no longer help future work.

## Temporary Files

- Do not put throwaway scratch files, temporary scripts, build caches, command output dumps, or intermediate downloads in this workspace.
- For temporary work, use the operating system temp directory, the run_python state directory, or another explicit temporary location.
- Clean temporary locations when the task is done.

## Updating These Instructions

- Update this `AGENTS.md` whenever you add a durable top-level directory, establish a new naming convention, or change how this workspace should be organized.
- When you update this file, keep the structure clear: purpose, directory map, notes convention, maintenance rules, and temporary-file policy.
- Do not delete user-authored instructions unless the user asks you to.
"""

DEFAULT_DAEMON_AGENTS_MD_ZH = """# uv-agent Daemon 工作区

## 用途

这是 `uv-agent daemon` 的默认持久工作区。这里用于保存跨 daemon 会话仍然有价值的长期笔记、计划、参考资料和生成产物。

## 目录说明

- `AGENTS.md`：工作区策略、目录说明和维护规则。
- `notes/`：面向用户的长期笔记、待办、提醒、决策和参考资料。除非用户指定其他路径，默认把“记一下”之类的请求放在这里。
- 当创建长期顶层目录时，在这里补充该目录的用途说明。

## 工作区维护

- 创建新的顶层文件或目录前，先检查现有结构，并优先复用合适的位置。
- 将相关资料放在命名清晰的子目录中，避免把文件散落在工作区根目录。
- 当笔记区内容增多时，优先用 `notes/index.md` 做索引，用 `notes/inbox.md` 暂存未分类记录。
- 对长期保留的目录，优先补充简短的 README 或索引文件，说明该目录存放什么、如何维护。
- 对已经没有长期价值的旧输出，及时清理或归档。

## 临时文件

- 不要把一次性草稿、临时脚本、构建缓存、命令输出 dump、中间下载产物放进这个工作区。
- 临时工作应使用操作系统临时目录、run_python 状态目录，或其他明确标记的临时位置。
- 任务完成后清理临时位置，除非用户明确要求保留。

## 更新本说明

- 当你新增长期顶层目录、建立新的命名约定，或改变工作区整理方式时，必须更新这个 `AGENTS.md`。
- 更新时保持结构清晰：用途、目录说明、笔记约定、维护规则、临时文件策略。
- 不要删除用户写下的说明，除非用户明确要求。
"""

DEFAULT_DAEMON_AGENTS_MD = DEFAULT_DAEMON_AGENTS_MD_EN


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def daemon_agents_template(language: UserLanguage | str | None = None) -> str:
    lang = language if isinstance(language, UserLanguage) else detect_user_language(str(language or "auto"))
    return DEFAULT_DAEMON_AGENTS_MD_ZH if lang.is_chinese else DEFAULT_DAEMON_AGENTS_MD_EN


def ensure_daemon_workspace(project_root: Path | None = None, *, language: UserLanguage | str | None = None) -> Path:
    """Prepare the daemon workspace and seed its default instructions."""

    workspace = (project_root or default_daemon_workspace_dir()).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    agents_path = workspace / "AGENTS.md"
    if not agents_path.exists():
        if language is None:
            language = detect_user_language(load_config(workspace).ui.language)
        agents_path.write_text(daemon_agents_template(language), encoding="utf-8")
    return workspace


@dataclass
class DaemonLease:
    data_dir: Path
    name: str = "daemon"
    stale_after_s: float = 30.0
    heartbeat_interval_s: float = 5.0
    owner_id: str = field(default_factory=lambda: new_id("host"))
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _stop: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)

    def acquire(self, *, replace: bool = False) -> None:
        now = _now()
        with connect_state_db(self.data_dir) as db:
            row = db.execute("SELECT * FROM host_leases WHERE name = ?", (self.name,)).fetchone()
            if row is not None and self._fresh(dict(row)):
                pid = int(row["pid"] or 0)
                if not replace:
                    raise RuntimeError(f"uv-agent daemon is already running (pid={pid}, owner={row['owner_id']})")
                logger.info("Replacing existing uv-agent daemon pid=%s owner=%s", pid, row["owner_id"])
                self._terminate_old(pid)
            db.execute(
                """
                INSERT OR REPLACE INTO host_leases(name, owner_id, pid, heartbeat_at, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.name, self.owner_id, os.getpid(), now, json.dumps({"started_at": now}, sort_keys=True)),
            )

    def start_heartbeat(self) -> None:
        if self._task is None or self._task.done():
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self._heartbeat_loop(), name="uv-agent-daemon-heartbeat")

    async def release(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        with connect_state_db(self.data_dir) as db:
            db.execute("DELETE FROM host_leases WHERE name = ? AND owner_id = ?", (self.name, self.owner_id))

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            with connect_state_db(self.data_dir) as db:
                db.execute(
                    "UPDATE host_leases SET heartbeat_at = ?, pid = ? WHERE name = ? AND owner_id = ?",
                    (_now(), os.getpid(), self.name, self.owner_id),
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.heartbeat_interval_s)
            except asyncio.TimeoutError:
                pass

    def _fresh(self, row: dict[str, Any]) -> bool:
        try:
            heartbeat = datetime.fromisoformat(str(row.get("heartbeat_at", "")).replace("Z", "+00:00"))
        except ValueError:
            return False
        if heartbeat < datetime.now(UTC) - timedelta(seconds=self.stale_after_s):
            return False
        pid = int(row.get("pid") or 0)
        return pid <= 0 or _pid_alive(pid)

    def _terminate_old(self, pid: int) -> None:
        if pid <= 0 or pid == os.getpid() or not _pid_alive(pid):
            return
        logger.info("Terminating existing uv-agent daemon pid=%s", pid)
        os.kill(pid, signal.SIGTERM)
        # This path runs before the async service loop starts; use a simple sleep loop.
        import time
        stop_at = time.monotonic() + self.stale_after_s
        while time.monotonic() < stop_at:
            if not _pid_alive(pid):
                return
            time.sleep(0.1)
        raise RuntimeError(f"Timed out waiting for existing daemon pid={pid} to exit")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_alive_windows(pid: int) -> bool:
    """Check process liveness on Windows without delivering a console signal."""

    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process_query_limited_information = 0x1000
    still_active = 259
    kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int

    handle = kernel32.OpenProcess(process_query_limited_information, 0, pid)
    if not handle:
        # ERROR_ACCESS_DENIED means a process exists but cannot be queried; most
        # other failures for this use case mean the PID is gone or invalid.
        return ctypes.get_last_error() == 5
    try:
        exit_code = ctypes.c_uint32()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


async def run_daemon(
    *,
    project_root: Path,
    data_dir: Path | None = None,
    replace: bool = False,
    log_level: str | int | None = None,
) -> None:
    engine = create_engine(project_root, data_dir=data_dir, log_level=log_level)
    lease = DaemonLease(engine.thread_store.data_dir)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    try:
        lease.acquire(replace=replace)
        lease.start_heartbeat()
        await engine.plugins.start()
        log_path = active_log_file()
        started_message = f"uv-agent daemon started pid={os.getpid()} state={engine.thread_store.data_dir}"
        logger.info("%s log=%s", started_message, log_path)
        if log_path is not None:
            started_message = f"{started_message} log={log_path}"
        print(started_message, flush=True)
        await stop.wait()
    finally:
        await engine.aclose()
        await lease.release()
        logger.info("uv-agent daemon stopped")
        print("uv-agent daemon stopped", flush=True)
