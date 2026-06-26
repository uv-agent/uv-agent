from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import logging
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pluggy

from uv_agent.config import PluginsConfig
from uv_agent.paths import uv_agent_home
from uv_agent.state_db import SQLITE_BUSY_TIMEOUT_MS, SQLITE_TIMEOUT_SECONDS

from .context import PluginContext, TurnContextBlock, TurnPrepareRequest
from .events import EventBus
from .helpers import RuntimeHelperRegistry, RuntimeHelperSpec
from . import hookspecs

PLUGIN_ENTRY_POINT_GROUP = "uv_agent.plugins"
PREPARE_TURN_HOOK_TIMEOUT_S = 2.0
PREPARE_TURN_BLOCK_MAX_CHARS = 8_192
PREPARE_TURN_TOTAL_MAX_CHARS = 32_768
PREPARE_TURN_TRUNCATION_SUFFIX = "\n...[plugin context truncated]"


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
        thread_store=None,
        user_state_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root
        self.user_state_dir = user_state_dir or uv_agent_home()
        self.events = events
        self.helpers = helper_registry
        self._submitter = submitter
        self._thread_store = thread_store
        self._tasks: dict[str, set[asyncio.Task[Any]]] = {}
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

    async def call_helper(self, name: str, args: list[Any] | None = None, kwargs: dict[str, Any] | None = None) -> Any:
        from .helpers import payload_from_call

        return await self.helpers.call(name, payload_from_call(list(args or []), dict(kwargs or {})))

    async def prepare_turn(self, request: TurnPrepareRequest) -> list[TurnContextBlock]:
        """Collect additive pre-user context from started plugins.

        This hook runs on the critical path before the model request is built, so
        each plugin is isolated: failures are logged and surfaced as plugin events
        without blocking other plugins or the current turn.
        """

        blocks: list[TurnContextBlock] = []
        seen_dedupe_keys: set[tuple[str, str]] = set()
        remaining = PREPARE_TURN_TOTAL_MAX_CHARS
        for record in list(self._records.values()):
            if remaining <= 0:
                break
            if record.status.state != "started":
                continue
            try:
                results = record.plugin_manager.hook.uv_agent_prepare_turn(
                    context=record.context,
                    request=request,
                )
                for result in results:
                    resolved = await _resolve_hook_result(result, timeout_s=PREPARE_TURN_HOOK_TIMEOUT_S)
                    for raw_block in _iter_turn_context_blocks(resolved):
                        block = _normalize_turn_context_block(record.name, raw_block)
                        if block is None:
                            continue
                        if block.dedupe_key:
                            dedupe_key = (record.name, block.dedupe_key)
                            if dedupe_key in seen_dedupe_keys:
                                continue
                            seen_dedupe_keys.add(dedupe_key)
                        text = _truncate_context_text(block.text, min(PREPARE_TURN_BLOCK_MAX_CHARS, remaining))
                        if not text:
                            continue
                        blocks.append(
                            TurnContextBlock(
                                text=text,
                                placement=block.placement,
                                dedupe_key=block.dedupe_key,
                                metadata=block.metadata,
                                plugin=record.name,
                            )
                        )
                        remaining -= len(text)
                        if remaining <= 0:
                            break
                    if remaining <= 0:
                        break
            except Exception as exc:
                record.context.logger.exception("Plugin prepare_turn hook failed")
                self._publish(
                    {
                        "type": "plugin.hook_failed",
                        "plugin": record.name,
                        "hook": "uv_agent_prepare_turn",
                        "error_type": exc.__class__.__name__,
                        "message": str(exc) or repr(exc),
                    }
                )
        return blocks

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
                task_factory=self._create_task,
                thread_store=self._thread_store,
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
        if record.status.state != "warning":
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
        await self._cancel_record_tasks(record)
        record.status.state = "stopped"
        self._publish({"type": "plugin.stopped", "plugin": record.name})


    def _create_task(self, plugin: str, coro, name: str | None = None) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name or f"uv-agent-plugin-{plugin}")
        self._tasks.setdefault(plugin, set()).add(task)

        def done(completed: asyncio.Task[Any]) -> None:
            self._tasks.get(plugin, set()).discard(completed)
            if completed.cancelled():
                return
            exc = completed.exception()
            if exc is None:
                return
            record = self._records.get(plugin)
            if record is not None and record.status.state in {"started", "starting"}:
                record.status.state = "warning"
                record.status.error_type = exc.__class__.__name__
                record.status.message = str(exc) or repr(exc)
                record.context.logger.exception("Plugin task failed", exc_info=exc)
            self._publish(
                {
                    "type": "plugin.task_failed",
                    "plugin": plugin,
                    "task": completed.get_name(),
                    "error_type": exc.__class__.__name__,
                    "message": str(exc) or repr(exc),
                }
            )

        task.add_done_callback(done)
        return task

    async def _cancel_record_tasks(self, record: _PluginRecord, *, timeout_s: float = 5.0) -> None:
        tasks = list(self._tasks.get(record.name, set()))
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.wait(tasks, timeout=timeout_s)
        self._tasks.pop(record.name, None)

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
        await _resolve_hook_result(result)


async def _resolve_hook_result(result: Any, *, timeout_s: float | None = None) -> Any:
    if result is None:
        return None
    if inspect.isawaitable(result):
        if timeout_s is None:
            return await result
        return await asyncio.wait_for(result, timeout=timeout_s)
    return result


def _iter_turn_context_blocks(value: Any) -> Iterable[TurnContextBlock]:
    if value is None:
        return
    if isinstance(value, TurnContextBlock):
        yield value
        return
    if isinstance(value, str):
        yield TurnContextBlock(text=value)
        return
    if isinstance(value, dict):
        yield _turn_context_block_from_mapping(value)
        return
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _iter_turn_context_blocks(item)
        return
    raise TypeError(f"Unsupported turn context block result: {type(value).__name__}")


def _turn_context_block_from_mapping(value: dict[str, Any]) -> TurnContextBlock:
    metadata = value.get("metadata")
    return TurnContextBlock(
        text=str(value.get("text") or ""),
        placement=str(value.get("placement") or "before_user"),  # type: ignore[arg-type]
        dedupe_key=None if value.get("dedupe_key") is None else str(value.get("dedupe_key")),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _normalize_turn_context_block(plugin: str, block: TurnContextBlock) -> TurnContextBlock | None:
    placement = str(block.placement or "before_user")
    if placement != "before_user":
        raise ValueError(f"Unsupported turn context placement from {plugin}: {placement!r}")
    text = str(block.text or "").strip()
    if not text:
        return None
    dedupe_key = str(block.dedupe_key).strip() if block.dedupe_key else None
    metadata = dict(block.metadata) if isinstance(block.metadata, dict) else {}
    return TurnContextBlock(
        text=text,
        placement="before_user",
        dedupe_key=dedupe_key or None,
        metadata=metadata,
        plugin=plugin,
    )


def _truncate_context_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(PREPARE_TURN_TRUNCATION_SUFFIX):
        return text[:limit]
    keep = limit - len(PREPARE_TURN_TRUNCATION_SUFFIX)
    return text[:keep].rstrip() + PREPARE_TURN_TRUNCATION_SUFFIX


def _safe_plugin_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-_.")
    return safe or "plugin"
