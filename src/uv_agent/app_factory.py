from __future__ import annotations

from pathlib import Path

from uv_agent.agent import AgentEngine
from uv_agent.config import load_config
from uv_agent.model import UnifiedModelClient
from uv_agent.paths import (
    project_attachments_dir,
    project_run_scripts_dir,
    project_scriptenv_dir,
    project_state_dir,
)
from uv_agent.runner import PythonRunner
from uv_agent.session import ThreadStore


def create_engine(project_root: Path | None = None, *, data_dir: Path | None = None) -> AgentEngine:
    root = (project_root or Path.cwd()).resolve()
    config = load_config(root)
    state_dir = (data_dir or project_state_dir(root)).resolve()
    runner = PythonRunner(
        project_root=root,
        data_dir=state_dir,
        config=config.runner,
        runs_dir=project_run_scripts_dir(root) if data_dir is None else state_dir / "runner" / "scripts",
        scriptenv_dir=project_scriptenv_dir(root) if data_dir is None else state_dir / "runner" / "scriptenv",
    )
    thread_store = ThreadStore(state_dir)
    model_client = UnifiedModelClient(config)
    return AgentEngine(
        config=config,
        model_client=model_client,
        runner=runner,
        thread_store=thread_store,
        attachments_dir=project_attachments_dir(root) if data_dir is None else state_dir / "attachments",
        project_root=root,
        config_loader=lambda: load_config(root),
    )
