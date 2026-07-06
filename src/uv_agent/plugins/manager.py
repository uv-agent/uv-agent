from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import logging
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from uv_agent.config import LoggingConfig, PluginsConfig
from uv_agent.paths import uv_agent_home
from uv_agent.state_db import SQLITE_BUSY_TIMEOUT_MS, SQLITE_TIMEOUT_SECONDS

from .api import PluginHostInfo, PluginManifest, PluginStatus, SetupPlugin
from .context import PluginActionAPI, PluginContext, PluginContextBroker, maybe_await
from .events import EventBus
from .i18n import PluginI18nRegistry
from .registry import ActionRegistry, CommandRegistry, RuntimeFunctionSpec, RuntimeNamespaceRegistry, UiRegistry
from .resources import ResourceRegistry
from .storage import PluginStorage, indexes_from_storage_schema

PLUGIN_ENTRY_POINT_GROUP = "uv_agent.plugins"
CORE_COMMANDS = {"/help", "/quit", "/clear", "/cancel", "/status", "/threads", "/show", "/image", "/level", "/model", "/title"}
RESERVED_RUNTIME_NAMESPACES = {
    "file", "files", "search", "symbols", "query", "patch", "apply_patch", "diff", "compare",
    "snapshot", "restore", "transaction", "run", "deps", "cd", "pwd", "path", "events", "look_at", "threads",
    "get", "blob",
}
logger = logging.getLogger(__name__)


@dataclass
class _PluginRecord:
    plugin: SetupPlugin
    context: PluginContext | None = None
    status: PluginStatus = field(default_factory=lambda: PluginStatus(id=""))


class PluginManager:
    """Discover, configure, and start manifest/setup uv-agent plugins."""

    def __init__(
        self,
        *,
        config: PluginsConfig,
        project_root: Path,
        events: EventBus,
        helper_registry: RuntimeNamespaceRegistry,
        submitter,
        thread_store=None,
        blob_store=None,
        logging_config: LoggingConfig | None = None,
        user_state_dir: Path | None = None,
        host: PluginHostInfo | None = None,
        agent_config: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.logging_config = logging_config or LoggingConfig()
        self.project_root = project_root
        self.user_state_dir = user_state_dir or uv_agent_home()
        project_state_dir = thread_store.data_dir if thread_store is not None else self.project_root / ".uv-agent"
        self.host = host or PluginHostInfo(
            invocation="tui",
            lifetime="session",
            project_root=self.project_root,
            project_state_dir=project_state_dir,
            user_state_dir=self.user_state_dir,
        )
        self._agent_config = agent_config
        self.events = events
        self.events.on_handler_error(self._mark_plugin_warning_from_event_logger)
        self.runtime = helper_registry
        self.resources = ResourceRegistry()
        self.actions = ActionRegistry()
        self.commands = CommandRegistry(reserved=CORE_COMMANDS)
        self.ui = UiRegistry()
        self.i18n = PluginI18nRegistry()
        self.contexts = PluginContextBroker()
        self._compaction_section_providers: list[tuple[str, Callable[..., str]]] = []
        self._epoch_context_refreshers: list[tuple[str, Callable[..., Any]]] = []
        self._submitter = submitter
        self._thread_store = thread_store
        self._blob_store = blob_store
        self._tasks: dict[str, set[asyncio.Task[Any]]] = {}
        self._records: dict[str, _PluginRecord] = {}
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._lock = asyncio.Lock()

    @property
    def records(self) -> list[PluginStatus]:
        return [record.status for record in self._records.values()]

    def reload_logging_config(self, config: LoggingConfig) -> None:
        self.logging_config = config
        for plugin_id, record in self._records.items():
            if record.context is not None:
                self._logger_for(plugin_id)

    def context_for(self, plugin_id: str) -> PluginContext | None:
        record = self._records.get(plugin_id)
        return record.context if record is not None else None

    def helper_specs(self) -> list[RuntimeFunctionSpec]:
        return [function for namespace in self.runtime.list_namespaces() for function in namespace.functions]

    def helper_namespaces(self):
        return self.runtime.list_namespaces()

    def resolve_helper(self, name: str) -> dict[str, Any]:
        return self.runtime.resolve_payload(name)

    async def call_helper(
        self,
        name: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        context: Any = None,
    ) -> Any:
        return await self.runtime.call(name, args=list(args or []), kwargs=dict(kwargs or {}), context=context)

    def resolve_action(self, action_id: str) -> dict[str, Any]:
        return PluginActionAPI(plugin="host", registry=self.actions).resolve(action_id)

    async def call_action(
        self,
        action_id: str,
        payload: dict[str, Any] | None = None,
        *,
        context: Any = None,
        missing: str = "error",
    ) -> Any:
        return await PluginActionAPI(
            plugin="host",
            registry=self.actions,
            context_resolver=self.context_for,
        ).call(action_id, payload, context=context, missing=missing)

    def command_suggestions(self):
        return self.commands.list()

    def call_command(self, name: str, payload: dict[str, Any] | None = None) -> Any:
        spec = self.commands.get(name)
        if spec is None:
            raise LookupError(f"Unknown command: {name}")
        data = dict(payload or {})
        kwargs: dict[str, Any] = {"payload": data}
        context = self.context_for(spec.plugin)
        if context is not None and _accepts_context(spec.handler):
            kwargs["context"] = context
        result = spec.handler(**kwargs)
        if inspect.isawaitable(result):
            raise RuntimeError("Async plugin commands cannot be called from the synchronous TUI path")
        return result

    def picker_items(self, picker_id: str, query: str = ""):
        source = self.ui.picker(picker_id)
        if source is None:
            return []
        try:
            return self.ui.picker_items(picker_id, query=query)
        except Exception as exc:
            self._mark_plugin_warning(source.plugin, exc)
            return []

    def text(self, key: str, language=None) -> str:
        return self.i18n.text(key, language)

    def compaction_sections(self, thread_id: str) -> list[str]:
        sections: list[str] = []
        for plugin_id, provider in list(self._compaction_section_providers):
            try:
                section = provider(thread_id=thread_id)
            except Exception as exc:
                self._mark_plugin_warning(plugin_id, exc)
                continue
            text = str(section or "").strip()
            if text:
                sections.append(text)
        return sections

    def refresh_epoch_context(self, thread_id: str, *, discard_plugins: set[str] | None = None) -> None:
        discard_plugins = set(discard_plugins or ())
        for plugin_id, handler in list(self._epoch_context_refreshers):
            self._refresh_epoch_context_handler(
                plugin_id,
                handler,
                thread_id,
                discard=plugin_id in discard_plugins,
            )

    def _refresh_epoch_context_for_plugin(self, plugin_id: str, thread_id: str | None, *, discard: bool = False) -> None:
        for current_plugin_id, handler in list(self._epoch_context_refreshers):
            if current_plugin_id == plugin_id:
                self._refresh_epoch_context_handler(current_plugin_id, handler, thread_id, discard=discard)

    def _refresh_epoch_context_handler(
        self,
        plugin_id: str,
        handler: Callable[..., Any],
        thread_id: str | None,
        *,
        discard: bool = False,
    ) -> None:
        try:
            if discard:
                with self.contexts.suppress_epoch_outputs(plugin_id):
                    result = self._call_epoch_context_refresher(handler, thread_id)
            else:
                result = self._call_epoch_context_refresher(handler, thread_id)
        except Exception as exc:
            self._mark_plugin_warning(plugin_id, exc)
            return
        if inspect.isawaitable(result):
            self._mark_plugin_warning(
                plugin_id,
                RuntimeError("Epoch context refresh handlers must be synchronous"),
            )

    def _call_epoch_context_refresher(self, handler: Callable[..., Any], thread_id: str | None) -> Any:
        signature = inspect.signature(handler)
        parameters = list(signature.parameters.values())
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters) or "thread_id" in signature.parameters:
            return handler(thread_id=thread_id)
        if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
            return handler(thread_id)
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        if positional:
            return handler(thread_id)
        return handler()

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
            load_order = self._load_order()
            logger.info("Plugin manager starting plugins=%d enabled=%d", len(self._records), len(load_order))
            for plugin_id in load_order:
                await self._start_record(self._records[plugin_id])

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            try:
                await self._task
            except Exception:
                logger.exception("Plugin startup task failed during stop")
        logger.info("Plugin manager stopping plugins=%d", len(self._records))
        for record in reversed(list(self._records.values())):
            await self._stop_record(record)
        await self.events.drain()

    def _discover(self) -> None:
        for entry_point in sorted(importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP), key=lambda item: item.name):
            try:
                setup_plugin = _normalize_plugin_object(entry_point.load())
            except Exception as exc:
                plugin_id = str(getattr(entry_point, "name", "entry-point"))
                self._records[plugin_id] = _PluginRecord(
                    plugin=SetupPlugin(
                        manifest=PluginManifest(plugin_id, "0", plugin_id, "invalid plugin"),
                        setup=lambda _ctx: None,
                    ),
                    status=PluginStatus(id=plugin_id, state="failed", message=str(exc) or repr(exc), error_type=exc.__class__.__name__),
                )
                continue
            self._add_discovered(setup_plugin)

    def _add_discovered(self, plugin: SetupPlugin) -> None:
        manifest = plugin.manifest
        existing = self._records.get(manifest.id)
        if existing is not None:
            logger.warning("Duplicate plugin discovered plugin=%s", manifest.id)
            status = PluginStatus(
                id=f"{manifest.id}#duplicate",
                display_name=manifest.display_name,
                state="failed",
                builtin=manifest.builtin,
                message=f"Duplicate plugin id {manifest.id!r}",
                error_type="DuplicatePluginId",
            )
            self._records[status.id] = _PluginRecord(plugin=plugin, status=status)
            return
        first_load = self._mark_first_load(manifest.id)
        enabled = self.config.enabled(manifest.id, default=manifest.default_enabled)
        skip_message = _activation_skip_message(manifest, self.host) if enabled else ""
        state = "skipped" if skip_message else "discovered" if enabled else "disabled"
        status = PluginStatus(
            id=manifest.id,
            display_name=manifest.display_name,
            state=state,
            builtin=manifest.builtin,
            first_load=first_load,
            message=skip_message,
            deprecated=manifest.deprecated,
            deprecation_message=manifest.deprecation_message,
        )
        self._records[manifest.id] = _PluginRecord(plugin=plugin, status=status)
        logger.debug(
            "Plugin discovered plugin=%s enabled=%s builtin=%s first_load=%s",
            manifest.id,
            enabled,
            manifest.builtin,
            first_load,
        )
        self._publish({"type": "plugin.discovered", "plugin": manifest.id, "first_load": first_load, "builtin": manifest.builtin})
        if first_load:
            self._publish({"type": "plugin.first_load", "plugin": manifest.id})
        if skip_message:
            self._publish(
                {
                    "type": "plugin.skipped",
                    "plugin": manifest.id,
                    "message": skip_message,
                    "activation": manifest.activation,
                    "host_lifetime": self.host.lifetime,
                }
            )

    def _load_order(self) -> list[str]:
        pending = [
            plugin_id
            for plugin_id, record in self._records.items()
            if record.status.state not in {"disabled", "skipped"}
        ]
        pending.sort(key=lambda plugin_id: (self._records[plugin_id].plugin.manifest.priority, plugin_id))
        resolved: list[str] = []
        seen: set[str] = set()
        while pending:
            progressed = False
            for plugin_id in list(pending):
                deps = set(self._records[plugin_id].plugin.manifest.dependencies)
                if deps <= seen:
                    resolved.append(plugin_id)
                    seen.add(plugin_id)
                    pending.remove(plugin_id)
                    progressed = True
            if not progressed:
                # Preserve deterministic failure ordering for dependency cycles.
                resolved.extend(pending)
                break
        return resolved

    async def _start_record(self, record: _PluginRecord) -> None:
        manifest = record.plugin.manifest
        if record.status.state in {"disabled", "skipped", "failed"}:
            return
        missing = [
            dep
            for dep in manifest.dependencies
            if self._records.get(dep) is None or self._records[dep].status.state not in {"started", "warning"}
        ]
        if missing:
            record.status.state = "failed"
            record.status.error_type = "MissingDependency"
            record.status.message = f"Missing required plugin dependencies: {', '.join(missing)}"
            logger.warning("Plugin failed missing dependencies plugin=%s missing=%s", manifest.id, ",".join(missing))
            self._publish({"type": "plugin.failed", "plugin": manifest.id, "message": record.status.message})
            return
        record.status.state = "starting"
        self._publish({"type": "plugin.starting", "plugin": manifest.id})
        logger.info("Plugin starting plugin=%s", manifest.id)
        plugin_logger = self._logger_for(manifest.id)
        storage = PluginStorage(
            plugin_id=manifest.id,
            project_data_dir=self._thread_store.data_dir if self._thread_store is not None else self.project_root / ".uv-agent",
            global_data_dir=self.user_state_dir,
            indexes=indexes_from_storage_schema(manifest.storage_schema),
        )
        context = PluginContext(
            manifest=manifest,
            host=self.host,
            project_root=self.project_root,
            user_state_dir=self.user_state_dir,
            config=self.config.plugin_config(manifest.id),
            events=self.events,
            logger=plugin_logger,
            runtime_registry=self.runtime,
            resource_registry=self.resources,
            action_registry=self.actions,
            command_registry=self.commands,
            ui_registry=self.ui,
            i18n_registry=self.i18n,
            context_broker=self.contexts,
            storage=storage,
            blob_store=self._blob_store,
            submitter=self._submitter,
            task_factory=self._create_task,
            compaction_section_providers=self._compaction_section_providers,
            epoch_context_refreshers=self._epoch_context_refreshers,
            thread_store=self._thread_store,
            action_context_resolver=self.context_for,
            agent_config=self._agent_config,
        )
        record.context = context
        try:
            await maybe_await(record.plugin.setup(context))
        except Exception as exc:
            record.status.state = "failed"
            record.status.error_type = exc.__class__.__name__
            record.status.message = str(exc) or repr(exc)
            plugin_logger.exception("Plugin setup failed")
            logger.warning("Plugin setup failed plugin=%s error_type=%s", manifest.id, exc.__class__.__name__)
            self._publish(
                {
                    "type": "plugin.failed",
                    "plugin": manifest.id,
                    "error_type": record.status.error_type,
                    "message": record.status.message,
                }
            )
            return
        self._refresh_epoch_context_for_plugin(
            manifest.id,
            None,
            discard=self.contexts.plugin_has_pending_epoch(manifest.id),
        )
        if manifest.deprecated:
            record.status.state = "warning"
            record.status.error_type = "DeprecatedPlugin"
            record.status.message = _deprecation_message(manifest)
            plugin_logger.warning(record.status.message)
            self._publish(
                {
                    "type": "plugin.warning",
                    "plugin": manifest.id,
                    "error_type": record.status.error_type,
                    "message": record.status.message,
                    "deprecated": True,
                }
            )
        if record.status.state != "warning":
            record.status.state = "started"
        self._publish({"type": "plugin.started", "plugin": manifest.id})
        logger.info("Plugin started plugin=%s state=%s", manifest.id, record.status.state)

    async def _stop_record(self, record: _PluginRecord) -> None:
        if record.status.state not in {"started", "warning", "failed"}:
            return
        manifest = record.plugin.manifest
        self._publish({"type": "plugin.stopping", "plugin": manifest.id})
        logger.info("Plugin stopping plugin=%s state=%s", manifest.id, record.status.state)
        try:
            if record.plugin.stop is not None and record.context is not None:
                await maybe_await(record.plugin.stop(record.context))
        except Exception as exc:
            record.status.state = "failed"
            record.status.error_type = exc.__class__.__name__
            record.status.message = str(exc) or repr(exc)
            logger.warning("Plugin stop failed plugin=%s error_type=%s", manifest.id, exc.__class__.__name__)
            self._publish({"type": "plugin.failed", "plugin": manifest.id, "error_type": exc.__class__.__name__, "message": str(exc) or repr(exc)})
            self._close_logger_for(manifest.id)
            return
        await self._cancel_record_tasks(manifest.id)
        self._close_logger_for(manifest.id)
        record.status.state = "stopped"
        self._publish({"type": "plugin.stopped", "plugin": manifest.id})
        logger.info("Plugin stopped plugin=%s", manifest.id)

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
                if record.context is not None:
                    record.context.logger.error("Plugin task failed", exc_info=exc)
            self._publish({"type": "plugin.task_failed", "plugin": plugin, "task": completed.get_name(), "error_type": exc.__class__.__name__, "message": str(exc) or repr(exc)})

        task.add_done_callback(done)
        return task

    def _mark_plugin_warning(self, plugin_id: str, exc: Exception) -> None:
        record = self._records.get(plugin_id)
        if record is not None and record.status.state in {"started", "starting", "warning"}:
            record.status.state = "warning"
            record.status.error_type = exc.__class__.__name__
            record.status.message = str(exc) or repr(exc)
            if record.context is not None:
                record.context.logger.error("Plugin warning", exc_info=(type(exc), exc, exc.__traceback__))
        self._publish({
            "type": "plugin.warning",
            "plugin": plugin_id,
            "error_type": exc.__class__.__name__,
            "message": str(exc) or repr(exc),
        })

    def _mark_plugin_warning_from_event_logger(self, logger: logging.Logger, exc: BaseException) -> None:
        prefix = "uv_agent.plugins."
        if not logger.name.startswith(prefix):
            return
        plugin_id = logger.name[len(prefix):]
        self._mark_plugin_warning(plugin_id, exc if isinstance(exc, Exception) else RuntimeError(str(exc)))

    async def _cancel_record_tasks(self, plugin: str, *, timeout_s: float = 5.0) -> None:
        tasks = list(self._tasks.get(plugin, set()))
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.wait(tasks, timeout=timeout_s)
        self._tasks.pop(plugin, None)

    def _publish(self, event: dict[str, Any]) -> None:
        self.events.publish(event)

    def _plugin_dir(self, plugin_id: str) -> Path:
        return self.user_state_dir / "plugins" / _safe_plugin_name(plugin_id)

    def _registry_db_path(self) -> Path:
        return self.user_state_dir / "plugins" / "registry.sqlite3"

    def _mark_first_load(self, plugin_id: str) -> bool:
        path = self._registry_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=SQLITE_TIMEOUT_SECONDS)
        try:
            db.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS loaded_plugins (id TEXT PRIMARY KEY, first_seen_at TEXT NOT NULL)")
            row = db.execute("SELECT id FROM loaded_plugins WHERE id = ?", (plugin_id,)).fetchone()
            if row is not None:
                return False
            from uv_agent.time import utc_now_iso

            db.execute("INSERT INTO loaded_plugins(id, first_seen_at) VALUES (?, ?)", (plugin_id, utc_now_iso()))
            db.commit()
            return True
        finally:
            db.close()

    def _logger_for(self, plugin_id: str) -> logging.Logger:
        logger = logging.getLogger(f"uv_agent.plugins.{plugin_id}")
        app_level = logging.getLogger("uv_agent").level or logging.INFO
        logger.setLevel(app_level)
        log_dir = self._plugin_dir(plugin_id) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "plugin.log"
        max_bytes = max(0, int(self.logging_config.max_bytes))
        backup_count = max(0, int(self.logging_config.backup_count))
        has_handler = False
        for existing in list(logger.handlers):
            if not isinstance(existing, logging.FileHandler) or Path(existing.baseFilename) != log_path:
                continue
            if (
                isinstance(existing, RotatingFileHandler)
                and existing.maxBytes == max_bytes
                and existing.backupCount == backup_count
            ):
                has_handler = True
                continue
            logger.removeHandler(existing)
            existing.close()
        if not has_handler:
            handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
                delay=True,
            )
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(handler)
        return logger

    def _close_logger_for(self, plugin_id: str) -> None:
        logger = logging.getLogger(f"uv_agent.plugins.{plugin_id}")
        log_path = self._plugin_dir(plugin_id) / "logs" / "plugin.log"
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
                logger.removeHandler(handler)
                handler.close()

def _normalize_plugin_object(value: Any) -> SetupPlugin:
    if isinstance(value, SetupPlugin):
        return value
    if callable(value):
        plugin = value()
        if isinstance(plugin, SetupPlugin):
            return plugin
        raise TypeError(f"Plugin factory returned {type(plugin).__name__}, expected SetupPlugin")
    raise TypeError(f"Plugin entry point must be SetupPlugin or a factory returning SetupPlugin, got {type(value).__name__}")


def _safe_plugin_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-_.")
    return safe or "plugin"


def _activation_skip_message(manifest: PluginManifest, host: PluginHostInfo) -> str:
    activation = manifest.activation
    if activation == "always":
        return ""
    if activation == "persistent_only":
        if host.is_persistent:
            return ""
        return "requires persistent host"
    if activation == "session_only":
        if not host.is_persistent:
            return ""
        return "requires session host"
    return f"unsupported activation {activation!r}"


def _deprecation_message(manifest: PluginManifest) -> str:
    message = str(manifest.deprecation_message or "").strip()
    if message:
        return message
    return f"Plugin {manifest.id!r} is deprecated and may be removed in a future uv-agent release."


def _accepts_context(fn: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return "context" in signature.parameters
