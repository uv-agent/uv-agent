from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pluggy

from uv_agent.config import PluginsConfig
from uv_agent.paths import uv_agent_home
from uv_agent.state_db import SQLITE_BUSY_TIMEOUT_MS, SQLITE_TIMEOUT_SECONDS

from .context import PluginContext
from .events import EventBus
from .helpers import RuntimeHelperRegistry, RuntimeHelperSpec
from . import hookspecs

PLUGIN_ENTRY_POINT_GROUP = "uv_agent.plugins"


@dataclass
class PluginStatus:
    name: str
    state: str = "discovered"
    first_load: bool = False
    message: str = ""
    error_type: str | None = None


@dataclass
class _PluginRecord:
    name: str
    entry_point: importlib.metadata.EntryPoint
    context: PluginContext
    plugin_manager: pluggy.PluginManager
    status: PluginStatus = field(default_factory=lambda: PluginStatus(name=""))


class PluginManager:
    """Discover, start, and stop uv-agent plugins without blocking the TUI."""

    def __init__(
        self,
        *,
        config: PluginsConfig,
        project_root: Path,
        events: EventBus,
        helper_registry: RuntimeHelperRegistry,
        submitter,
        user_state_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root
        self.user_state_dir = user_state_dir or uv_agent_home()
        self.events = events
        self.helpers = helper_registry
        self._submitter = submitter
        self._records: dict[str, _PluginRecord] = {}
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._lock = asyncio.Lock()

    @property
    def records(self) -> list[PluginStatus]:
        return [record.status for record in self._records.values()]

    def helper_specs(self) -> list[RuntimeHelperSpec]:
        return self.helpers.list()

    def resolve_helper(self, name: str) -> dict[str, Any]:
        return self.helpers.resolve_payload(name)

    def start_background(self) -> asyncio.Task[None]:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.start(), name="uv-agent-plugin-start")
        return self._task

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            self._started = True
            self._discover()
            for record in list(self._records.values()):
                await self._start_record(record)

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            try:
                await self._task
            except Exception:
                pass
        for record in reversed(list(self._records.values())):
            await self._stop_record(record)
        await self.events.drain()

    def _discover(self) -> None:
        disabled = set(self.config.disabled)
        for entry_point in importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP):
            name = entry_point.name
            if name in disabled:
                continue
            if name in self._records:
                continue
            first_load = self._mark_first_load(name)
            logger = self._logger_for(name)
            context = PluginContext(
                name=name,
                project_root=self.project_root,
                user_state_dir=self.user_state_dir,
                data_dir=self._plugin_dir(name),
                log_dir=self._plugin_dir(name) / "logs",
                config=dict(self.config.config.get(name, {})),
                events=self.events,
                logger=logger,
                helper_registry=self.helpers,
                submitter=self._submitter,
            )
            manager = pluggy.PluginManager("uv_agent")
            manager.add_hookspecs(hookspecs)
            status = PluginStatus(name=name, state="discovered", first_load=first_load)
            record = _PluginRecord(
                name=name,
                entry_point=entry_point,
                context=context,
                plugin_manager=manager,
                status=status,
            )
            self._records[name] = record
            self._publish({"type": "plugin.discovered", "plugin": name, "first_load": first_load})
            if first_load:
                self._publish({"type": "plugin.first_load", "plugin": name})

    async def _start_record(self, record: _PluginRecord) -> None:
        record.status.state = "starting"
        self._publish({"type": "plugin.starting", "plugin": record.name})
        try:
            plugin = record.entry_point.load()
            record.plugin_manager.register(plugin, name=record.name)
            results = record.plugin_manager.hook.uv_agent_start(context=record.context)
            await _await_hook_results(results)
        except Exception as exc:
            record.status.state = "failed"
            record.status.error_type = exc.__class__.__name__
            record.status.message = str(exc) or repr(exc)
            record.context.logger.exception("Plugin start failed")
            self._publish(
                {
                    "type": "plugin.failed",
                    "plugin": record.name,
                    "error_type": record.status.error_type,
                    "message": record.status.message,
                }
            )
            return
        record.status.state = "started"
        self._publish({"type": "plugin.started", "plugin": record.name})

    async def _stop_record(self, record: _PluginRecord) -> None:
        if record.status.state not in {"started", "failed"}:
            return
        self._publish({"type": "plugin.stopping", "plugin": record.name})
        try:
            results = record.plugin_manager.hook.uv_agent_stop(context=record.context)
            await _await_hook_results(results)
        except Exception as exc:
            record.context.logger.exception("Plugin stop failed")
            record.status.state = "failed"
            record.status.error_type = exc.__class__.__name__
            record.status.message = str(exc) or repr(exc)
            self._publish(
                {
                    "type": "plugin.failed",
                    "plugin": record.name,
                    "error_type": exc.__class__.__name__,
                    "message": str(exc) or repr(exc),
                }
            )
            return
        record.status.state = "stopped"
        self._publish({"type": "plugin.stopped", "plugin": record.name})

    def _publish(self, event: dict[str, Any]) -> None:
        self.events.publish(event)

    def _plugin_dir(self, name: str) -> Path:
        return self.user_state_dir / "plugins" / _safe_plugin_name(name)

    def _registry_db_path(self) -> Path:
        return self.user_state_dir / "plugins" / "registry.sqlite3"

    def _mark_first_load(self, name: str) -> bool:
        path = self._registry_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=SQLITE_TIMEOUT_SECONDS) as db:
            db.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS loaded_plugins (name TEXT PRIMARY KEY, first_seen_at TEXT NOT NULL)")
            row = db.execute("SELECT name FROM loaded_plugins WHERE name = ?", (name,)).fetchone()
            if row is not None:
                return False
            from uv_agent.time import utc_now_iso

            db.execute("INSERT INTO loaded_plugins(name, first_seen_at) VALUES (?, ?)", (name, utc_now_iso()))
            return True

    def _logger_for(self, name: str) -> logging.Logger:
        logger = logging.getLogger(f"uv_agent.plugins.{name}")
        logger.setLevel(logging.INFO)
        log_dir = self._plugin_dir(name) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "plugin.log"
        if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path for handler in logger.handlers):
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(handler)
        return logger


async def _await_hook_results(results: list[Any]) -> None:
    for result in results:
        if result is not None:
            await result


def _safe_plugin_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-_.")
    return safe or "plugin"
