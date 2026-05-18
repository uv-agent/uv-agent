from __future__ import annotations

from pathlib import Path

from uv_agent.agent import AgentEngine
from uv_agent.config import load_config
from uv_agent.model_client import UnifiedModelClient
from uv_agent.paths import project_state_dir
from uv_agent.runner import PythonRunner
from uv_agent.session import ThreadStore


def create_engine(project_root: Path | None = None, *, data_dir: Path | None = None) -> AgentEngine:
    root = (project_root or Path.cwd()).resolve()
    config = load_config(root)
    state_dir = (data_dir or project_state_dir(root)).resolve()
    runner = PythonRunner(project_root=root, data_dir=state_dir, config=config.runner)
    thread_store = ThreadStore(state_dir)
    model_client = UnifiedModelClient(config)
    return AgentEngine(
        config=config,
        model_client=model_client,
        runner=runner,
        thread_store=thread_store,
        project_root=root,
        config_loader=lambda: load_config(root),
    )
