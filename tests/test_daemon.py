from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from uv_agent.cli import main
from uv_agent.daemon import (
    DEFAULT_DAEMON_AGENTS_MD,
    DEFAULT_DAEMON_AGENTS_MD_ZH,
    DaemonLease,
    _DaemonStopController,
    _daemon_startup_lines,
    _install_daemon_signal_handlers,
    _pid_alive,
    ensure_daemon_workspace,
    run_daemon,
)
from uv_agent.plugins import PluginStatus
from uv_agent.state_db import connect_state_db


def test_daemon_workspace_defaults_to_user_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UV_AGENT_LANGUAGE", "en")

    workspace = ensure_daemon_workspace()

    assert workspace == (tmp_path / "home" / "workspace").resolve()
    text = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    assert text == DEFAULT_DAEMON_AGENTS_MD
    assert "`notes/`" in text
    assert "`notes/inbox.md`" in text


def test_daemon_workspace_uses_chinese_template_from_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UV_AGENT_LANGUAGE", "en")
    workspace = tmp_path / "workspace"
    project_config = workspace / ".uv-agent" / "config.json"
    project_config.parent.mkdir(parents=True)
    project_config.write_text(json.dumps({"ui": {"language": "zh-CN"}}), encoding="utf-8")

    ensure_daemon_workspace(workspace)

    text = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    assert text == DEFAULT_DAEMON_AGENTS_MD_ZH
    assert "## 目录说明" in text
    assert "`notes/`" in text
    assert "`notes/inbox.md`" in text
    assert "## 更新本说明" in text


def test_daemon_workspace_does_not_overwrite_existing_agents(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    agents = workspace / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text("custom\n", encoding="utf-8")

    assert ensure_daemon_workspace(workspace) == workspace.resolve()
    assert agents.read_text(encoding="utf-8") == "custom\n"


def test_cli_daemon_uses_default_workspace(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    async def fake_run_daemon(**kwargs):
        captured.update(kwargs)

    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UV_AGENT_LANGUAGE", "en")
    monkeypatch.setattr("uv_agent.daemon.run_daemon", fake_run_daemon)
    monkeypatch.setattr(sys, "argv", ["uv-agent", "daemon", "--replace"])

    main()

    workspace = (tmp_path / "home" / "workspace").resolve()
    assert captured["project_root"] == workspace
    assert captured["replace"] is True
    assert (workspace / "AGENTS.md").read_text(encoding="utf-8") == DEFAULT_DAEMON_AGENTS_MD


def test_daemon_startup_lines_include_operational_context(tmp_path: Path) -> None:
    records = [
        PluginStatus(id="builtin.goal", state="started", builtin=True),
        PluginStatus(id="remote-control", state="started", first_load=True),
        PluginStatus(id="auth-code", state="failed", message="missing token", error_type="ValueError"),
    ]

    lines = _daemon_startup_lines(
        project_root=tmp_path / "workspace",
        state_dir=tmp_path / "state",
        log_path=tmp_path / "state" / "log" / "uv-agent.log",
        log_level="INFO",
        plugin_records=records,
        plugin_log_root=tmp_path / "home" / "plugins",
    )

    assert f"workspace={tmp_path / 'workspace'}" in lines
    assert f"state={tmp_path / 'state'}" in lines
    assert f"plugin_logs={tmp_path / 'home' / 'plugins'}" in lines
    assert "plugins: total=3 started=2 warning=0 failed=1 skipped=0 disabled=0" in lines
    assert "plugin started: remote-control" in lines
    assert "plugin first load: remote-control" in lines
    assert "plugin failed: auth-code (ValueError: missing token)" in lines


def test_daemon_stop_controller_requests_stop_once(capsys: pytest.CaptureFixture[str]) -> None:
    stop = asyncio.Event()
    controller = _DaemonStopController(stop)

    controller.request(signal.SIGINT)
    controller.request(signal.SIGTERM)

    assert stop.is_set()
    assert capsys.readouterr().out == "uv-agent daemon stopping signal=SIGINT\n"


def test_daemon_signal_handlers_fallback_to_signal_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = asyncio.Event()
    controller = _DaemonStopController(stop)
    installed: dict[int, Any] = {}
    restored: list[tuple[int, Any]] = []
    previous = object()

    class Loop:
        def add_signal_handler(self, signum, callback, *args):  # noqa: ANN001
            raise NotImplementedError

        def call_soon_threadsafe(self, callback, *args):  # noqa: ANN001
            callback(*args)

    monkeypatch.setattr(signal, "getsignal", lambda signum: previous)

    def fake_signal(signum, handler):  # noqa: ANN001
        installed[int(signum)] = handler
        restored.append((int(signum), handler))

    monkeypatch.setattr(signal, "signal", fake_signal)

    restore = _install_daemon_signal_handlers(controller, Loop())  # type: ignore[arg-type]

    installed[int(signal.SIGINT)](signal.SIGINT, None)

    assert stop.is_set()
    restore()
    assert restored[-1] == (int(signal.SIGINT), previous)


def test_daemon_signal_handlers_use_event_loop_when_available() -> None:
    stop = asyncio.Event()
    controller = _DaemonStopController(stop)
    installed: dict[int, tuple[Any, tuple[Any, ...]]] = {}
    removed: list[int] = []

    class Loop:
        def add_signal_handler(self, signum, callback, *args):  # noqa: ANN001
            installed[int(signum)] = (callback, args)

        def remove_signal_handler(self, signum):  # noqa: ANN001
            removed.append(int(signum))
            return True

    restore = _install_daemon_signal_handlers(controller, Loop())  # type: ignore[arg-type]
    callback, args = installed[int(signal.SIGINT)]

    callback(*args)

    assert stop.is_set()
    restore()
    assert int(signal.SIGINT) in removed


@pytest.mark.asyncio
async def test_run_daemon_handles_keyboard_interrupt_gracefully(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []

    class FakeThreadStore:
        data_dir = tmp_path / "state"

    class FakePlugins:
        records: list[PluginStatus] = []
        user_state_dir = tmp_path / "home"

        async def start(self) -> None:
            raise KeyboardInterrupt

    class FakeEngine:
        project_root = tmp_path / "workspace"
        thread_store = FakeThreadStore()
        plugins = FakePlugins()
        config = type("Config", (), {"logging": type("Logging", (), {"level": "INFO"})()})()

        async def aclose(self) -> None:
            events.append("engine.aclose")

    class FakeLease:
        def __init__(self, data_dir: Path) -> None:
            assert data_dir == tmp_path / "state"

        def acquire(self, *, replace: bool = False) -> None:
            events.append(f"lease.acquire:{replace}")

        def start_heartbeat(self) -> None:
            events.append("lease.start_heartbeat")

        async def release(self) -> None:
            events.append("lease.release")

    monkeypatch.setattr("uv_agent.daemon.create_engine", lambda *args, **kwargs: FakeEngine())
    monkeypatch.setattr("uv_agent.daemon.DaemonLease", FakeLease)

    await run_daemon(project_root=tmp_path / "workspace", replace=True)

    assert events == [
        "lease.acquire:True",
        "lease.start_heartbeat",
        "engine.aclose",
        "lease.release",
    ]
    assert capsys.readouterr().out == (
        "uv-agent daemon stopping signal=SIGINT\n"
        "uv-agent daemon stopped\n"
    )


def test_daemon_lease_rejects_fresh_owner(tmp_path):
    first = DaemonLease(tmp_path, owner_id="one")
    first.acquire()

    with pytest.raises(RuntimeError):
        DaemonLease(tmp_path, owner_id="two").acquire()


def test_daemon_lease_replaces_stale_owner(tmp_path):
    first = DaemonLease(tmp_path, owner_id="one")
    first.acquire()
    stale = (datetime.now(UTC) - timedelta(seconds=120)).isoformat().replace("+00:00", "Z")
    with connect_state_db(tmp_path) as db:
        db.execute("UPDATE host_leases SET heartbeat_at = ?, pid = 0 WHERE name = 'daemon'", (stale,))

    second = DaemonLease(tmp_path, owner_id="two")
    second.acquire()

    with connect_state_db(tmp_path) as db:
        row = db.execute("SELECT * FROM host_leases WHERE name = 'daemon'").fetchone()
    assert row["owner_id"] == "two"


def test_pid_alive_short_circuits_current_process(monkeypatch):
    calls = []
    monkeypatch.setattr("uv_agent.daemon.os.kill", lambda pid, sig: calls.append((pid, sig)))

    assert _pid_alive(os.getpid()) is True
    assert calls == []
