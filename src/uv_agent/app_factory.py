from __future__ import annotations

from pathlib import Path

from uv_agent.agent import AgentEngine
from uv_agent.config import load_config
from uv_agent.model_client import UnifiedModelClient
from uv_agent.runner import PythonRunner
from uv_agent.session import ThreadStore


def create_engine(project_root: Path | None = None) -> AgentEngine:
    root = (project_root or Path.cwd()).resolve()
    config = load_config(root)
    data_dir = root / ".uv-agent"
    runner = PythonRunner(project_root=root, data_dir=data_dir, config=config.runner)
    thread_store = ThreadStore(data_dir)
    model_client = UnifiedModelClient(config)
    return AgentEngine(
        config=config,
        model_client=model_client,
        runner=runner,
        thread_store=thread_store,
        project_root=root,
    )
