from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from uv_agent.daemon import DaemonLease, _pid_alive
from uv_agent.state_db import connect_state_db


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
