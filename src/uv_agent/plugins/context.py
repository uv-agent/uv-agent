from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .events import EventBus, raise_if_reentrant_submit
from .helpers import RuntimeHelperRegistry, RuntimeHelperSpec


@dataclass
class SubmittedTurn:
    thread_id: str
    turn_id: str
    _queue: asyncio.Queue[dict[str, Any] | None]

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event


class PluginContext:
    """Capabilities exposed to one uv-agent plugin."""

    def __init__(
        self,
        *,
        name: str,
        project_root: Path,
        user_state_dir: Path,
        data_dir: Path,
        log_dir: Path,
        config: dict[str, Any],
        events: EventBus,
        logger,
        helper_registry: RuntimeHelperRegistry,
        submitter: Callable[..., Any],
    ) -> None:
        self.name = name
        self.project_root = project_root
        self.user_state_dir = user_state_dir
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.config = config
        self.events = events
        self.logger = logger
        self._helper_registry = helper_registry
        self._submitter = submitter
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def register_runtime_helper(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        doc: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> RuntimeHelperSpec:
        return self._helper_registry.register(
            plugin=self.name,
            name=name,
            fn=fn,
            doc=doc,
            schema=schema,
        )

    def open_db(self) -> sqlite3.Connection:
        path = self.data_dir / "data.sqlite3"
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    async def submit_turn(
        self,
        *,
        text: str,
        thread_id: str | None = None,
        level: str | None = None,
        image_paths: list[Path] | None = None,
    ) -> SubmittedTurn:
        raise_if_reentrant_submit()
        return await self._submitter(
            text=text,
            thread_id=thread_id,
            level=level,
            image_paths=image_paths,
        )
