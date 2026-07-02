from __future__ import annotations

import pytest

HOST_RUN_ENV_VARS = (
    "UV_AGENT_RUNTIME_STATE_DIR",
    "UV_AGENT_RUNTIME_THREAD_ID",
    "UV_AGENT_RUNTIME_THREAD_KIND",
    "UV_AGENT_RUNTIME_TURN_ID",
    "UV_AGENT_RUNTIME_RUN_ID",
    "UV_AGENT_SCRIPTENV_DIR",
    "UV_AGENT_SCRIPT_DIR",
    "UV_BIN",
    "UV_AGENT_RUNTIME_PROJECT_STATE_DIR",
    "UV_AGENT_RUNTIME_HELPER_STATS_DB",
)


@pytest.fixture(autouse=True)
def isolate_host_run_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent from an outer uv-agent run_python process."""
    for key in HOST_RUN_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
