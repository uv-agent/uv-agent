from __future__ import annotations

import logging
from pathlib import Path

from uv_agent.agent import AgentEngine
from uv_agent.config import load_config
from uv_agent.host_events import HostEventBus
from uv_agent.logging_config import configure_logging
from uv_agent.telemetry import TelemetryStore
from uv_agent.model import UnifiedModelClient
from uv_agent.paths import (
    ensure_project_local_dir,
    project_attachments_dir,
    project_run_scripts_dir,
    project_scriptenv_dir,
    project_state_dir,
)
from uv_agent.runner import PythonRunner
from uv_agent.session import ThreadStore


logger = logging.getLogger(__name__)


def create_engine(
    project_root: Path | None = None,
    *,
    data_dir: Path | None = None,
    log_level: str | int | None = None,
    log_to_console: bool | None = None,
) -> AgentEngine:
    root = (project_root or Path.cwd()).resolve()
    ensure_project_local_dir(root)
    config = load_config(root)
    state_dir = (data_dir or project_state_dir(root)).resolve()
    configure_logging(
        state_dir,
        config.logging,
        level_override=log_level,
        console=log_to_console,
    )
    logger.info(
        "Creating uv-agent engine project_root=%s state_dir=%s log_level=%s",
        root,
        state_dir,
        log_level or config.logging.level,
    )
    host_events = HostEventBus()
    telemetry = TelemetryStore(state_dir)
    host_events.subscribe(telemetry.on_event)
    runner = PythonRunner(
        project_root=root,
        data_dir=state_dir,
        config=config.runner,
        runs_dir=project_run_scripts_dir(root) if data_dir is None else state_dir / "runner" / "scripts",
        scriptenv_dir=project_scriptenv_dir(root) if data_dir is None else state_dir / "runner" / "scriptenv",
        host_events=host_events,
    )
    thread_store = ThreadStore(state_dir, host_events=host_events)
    model_client = UnifiedModelClient(config)
    logger.debug(
        "Engine components initialized runner_scripts=%s scriptenv=%s attachments=%s",
        project_run_scripts_dir(root) if data_dir is None else state_dir / "runner" / "scripts",
        project_scriptenv_dir(root) if data_dir is None else state_dir / "runner" / "scriptenv",
        project_attachments_dir(root) if data_dir is None else state_dir / "attachments",
    )
    return AgentEngine(
        config=config,
        model_client=model_client,
        runner=runner,
        thread_store=thread_store,
        attachments_dir=project_attachments_dir(root) if data_dir is None else state_dir / "attachments",
        project_root=root,
        config_loader=lambda: load_config(root),
        host_events=host_events,
    )
