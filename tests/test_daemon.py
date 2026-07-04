from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from uv_agent.cli import main
from uv_agent.daemon import (
    DEFAULT_DAEMON_AGENTS_MD,
    DEFAULT_DAEMON_AGENTS_MD_ZH,
    DaemonLease,
    _pid_alive,
    ensure_daemon_workspace,
)
from uv_agent.state_db import connect_state_db


def test_daemon_workspace_defaults_to_user_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UV_AGENT_LANGUAGE", "en")

    workspace = ensure_daemon_workspace()

    assert workspace == (tmp_path / "home" / "workspace").resolve()
    assert (workspace / "AGENTS.md").read_text(encoding="utf-8") == DEFAULT_DAEMON_AGENTS_MD


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
