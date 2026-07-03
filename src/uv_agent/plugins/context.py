from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uv_agent.paths import uv_agent_home

from .api import PluginManifest
from .events import EventBus, raise_if_reentrant_submit
from .i18n import LocalizedText, PluginI18nRegistry
from .registry import (
    ActionRegistry,
    ActionSpec,
    CommandRegistry,
    CommandSpec,
    PickerItem,
    PickerSource,
    RuntimeNamespaceRegistry,
    RuntimeNamespaceSpec,
    UiRegistry,
)
from .storage import PluginStorage
from .xml import XmlContribution, render_contribution


@dataclass
class SubmittedTurn:
    thread_id: str | None
    turn_id: str | None
    _queue: asyncio.Queue[dict[str, Any] | None]
    _waiter: Callable[[], Awaitable[Any]] | None = None
    status: str = "queued"
    final_text: str = ""
    error: BaseException | None = None

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def wait(self) -> "SubmittedTurn":
        if self._waiter is None:
            return self
        result = await self._waiter()
        self.thread_id = str(getattr(result, "thread_id", self.thread_id) or "") or self.thread_id
        self.turn_id = str(getattr(result, "turn_id", self.turn_id) or "") or self.turn_id
        self.status = str(getattr(result, "status", self.status) or self.status)
        self.final_text = str(getattr(result, "final_text", self.final_text) or "")
        self.error = getattr(result, "error", self.error)
        return self


@dataclass(frozen=True)
class UserInput:
    """One external user message that may be coalesced into a supervised turn."""

    text: str
    image_paths: tuple[str | Path, ...] = ()
    request_id: str | None = None


@dataclass(frozen=True)
class RenderedEpochContribution:
    contribution: XmlContribution
    text: str


@dataclass
class PluginRegistration:
    _dispose: Callable[[], None]
    _disposed: bool = False

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._dispose()


class PluginContextBroker:
    """In-memory active context registry owned by the PluginManager.

    ThreadStore persists what was actually sent to the model.  The broker stores
    pending epoch/turn notices so Engine can batch them into the context item
    groups required by the refactor plan.  Epoch context is deliberately a
    send queue, not a state store; plugins own their own refresh state.
    """

    def __init__(self) -> None:
        self._epoch_items: dict[str | None, deque[XmlContribution]] = defaultdict(deque)
        self._updates: dict[str | None, deque[XmlContribution]] = defaultdict(deque)
        self._turn_items: dict[str, deque[XmlContribution]] = defaultdict(deque)
        self._turn_replay_items: dict[tuple[str, str, str], XmlContribution] = {}
        self._replay: dict[tuple[str, str, str], XmlContribution] = {}
        self._suppressed_epoch_plugins: dict[str, int] = defaultdict(int)

    def publish(
        self,
        *,
        plugin: str,
        tag: str,
        body: Any,
        attrs: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> None:
        if self._is_epoch_suppressed(plugin):
            return
        contribution = XmlContribution(tag=tag, body=body, attrs=dict(attrs or {}), plugin=plugin, operation="publish")
        self._epoch_items[thread_id].append(contribution)
        self._updates[thread_id].append(contribution)

    def update(
        self,
        *,
        plugin: str,
        tag: str,
        body: Any,
        attrs: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> None:
        if self._is_epoch_suppressed(plugin):
            return
        contribution = XmlContribution(tag=tag, body=body, attrs=dict(attrs or {}), plugin=plugin, operation="update")
        self._epoch_items[thread_id].append(contribution)
        self._updates[thread_id].append(contribution)

    def remove(
        self,
        *,
        plugin: str,
        tag: str,
        reason: str = "",
        thread_id: str | None = None,
    ) -> None:
        if self._is_epoch_suppressed(plugin):
            return
        body = {"reason": reason} if reason else {}
        contribution = XmlContribution(tag=tag, body=body, attrs={}, plugin=plugin, operation="remove")
        self._epoch_items[thread_id].append(contribution)
        self._updates[thread_id].append(contribution)

    def enqueue_turn(
        self,
        *,
        plugin: str,
        thread_id: str,
        tag: str,
        body: Any,
        attrs: Mapping[str, Any] | None = None,
        replay_after_compaction: bool = False,
        replay_key: str | None = None,
    ) -> None:
        if replay_after_compaction and not replay_key:
            raise ValueError("replay_key is required when replay_after_compaction is true")
        contribution = XmlContribution(tag=tag, body=body, attrs=dict(attrs or {}), plugin=plugin)
        key = (thread_id, plugin, replay_key) if replay_key else None
        if key is not None:
            self._replace_turn_replay_item(key, contribution)
        self._turn_items[thread_id].append(contribution)
        if replay_after_compaction and key is not None:
            self._replay[key] = contribution
        elif key is not None:
            self._replay.pop(key, None)

    def clear_replay(self, *, plugin: str, thread_id: str, replay_key: str) -> None:
        key = (thread_id, plugin, replay_key)
        self._replay.pop(key, None)
        previous = self._turn_replay_items.pop(key, None)
        if previous is not None:
            self._remove_turn_contribution(thread_id, previous)

    def _replace_turn_replay_item(self, key: tuple[str, str, str], contribution: XmlContribution) -> None:
        thread_id, _plugin, _replay_key = key
        previous = self._turn_replay_items.pop(key, None)
        if previous is not None:
            self._remove_turn_contribution(thread_id, previous)
        self._turn_replay_items[key] = contribution

    def _remove_turn_contribution(self, thread_id: str, contribution: XmlContribution) -> None:
        queue = self._turn_items.get(thread_id)
        if not queue:
            return
        self._turn_items[thread_id] = deque(item for item in queue if item != contribution)

    def plugins_with_pending_epoch(self, thread_id: str | None) -> set[str]:
        return {
            contribution.plugin
            for contribution in [*self._epoch_items.get(None, ()), *self._epoch_items.get(thread_id, ())]
            if contribution.plugin
        }

    def plugin_has_pending_epoch(self, plugin: str) -> bool:
        return any(
            contribution.plugin == plugin
            for queue in self._epoch_items.values()
            for contribution in queue
        )

    @contextmanager
    def suppress_epoch_outputs(self, plugin: str):
        self._suppressed_epoch_plugins[plugin] += 1
        try:
            yield
        finally:
            remaining = self._suppressed_epoch_plugins[plugin] - 1
            if remaining > 0:
                self._suppressed_epoch_plugins[plugin] = remaining
            else:
                self._suppressed_epoch_plugins.pop(plugin, None)

    def _is_epoch_suppressed(self, plugin: str) -> bool:
        return self._suppressed_epoch_plugins.get(plugin, 0) > 0

    def consume_epoch(
        self,
        thread_id: str,
        *,
        include_contribution: Callable[[XmlContribution], bool] | None = None,
    ) -> list[RenderedEpochContribution]:
        contributions = [*self._epoch_items.pop(None, ()), *self._epoch_items.pop(thread_id, ())]
        # Full epoch context supersedes any startup publish/update notices queued
        # before the thread had received its first full item.
        self._updates.pop(None, None)
        self._updates.pop(thread_id, None)
        return self._rendered_contributions(contributions, include_contribution=include_contribution, include_operation=False)

    def consume_updates(self, thread_id: str) -> list[RenderedEpochContribution]:
        contributions = [*self._updates.pop(None, ()), *self._updates.pop(thread_id, ())]
        for contribution in contributions:
            self._remove_epoch_contribution(None, contribution)
            self._remove_epoch_contribution(thread_id, contribution)
        return self._rendered_contributions(contributions, include_operation=True)

    def clear_epoch(self, thread_id: str) -> None:
        self._epoch_items.pop(None, None)
        self._epoch_items.pop(thread_id, None)
        self._updates.pop(None, None)
        self._updates.pop(thread_id, None)

    def _remove_epoch_contribution(self, thread_id: str | None, contribution: XmlContribution) -> None:
        queue = self._epoch_items.get(thread_id)
        if not queue:
            return
        removed = False
        kept: deque[XmlContribution] = deque()
        for item in queue:
            if not removed and item is contribution:
                removed = True
                continue
            kept.append(item)
        if kept:
            self._epoch_items[thread_id] = kept
        else:
            self._epoch_items.pop(thread_id, None)

    def _rendered_contributions(
        self,
        contributions: Iterable[XmlContribution],
        *,
        include_contribution: Callable[[XmlContribution], bool] | None = None,
        include_operation: bool = False,
    ) -> list[RenderedEpochContribution]:
        rendered: list[RenderedEpochContribution] = []
        for contribution in contributions:
            if include_contribution is not None and not include_contribution(contribution):
                continue
            text = render_contribution(
                contribution.tag,
                contribution.body,
                attrs=contribution.attrs,
                operation=contribution.operation if include_operation or contribution.operation != "publish" else None,
            )
            rendered.append(RenderedEpochContribution(contribution=contribution, text=text))
        return rendered

    def turn_context_text(self, thread_id: str) -> str:
        contributions = list(self._turn_items.pop(thread_id, ()))
        if not contributions:
            return ""
        return "\n\n".join(
            render_contribution(item.tag, item.body, attrs=item.attrs)
            for item in contributions
        )

    def replay_after_compaction(self, thread_id: str) -> None:
        for key, contribution in list(self._replay.items()):
            stored_thread_id, _plugin, _replay_key = key
            if stored_thread_id == thread_id:
                self._replace_turn_replay_item(key, contribution)
                self._turn_items[thread_id].append(contribution)

class PluginThreadAPI:
    """Narrow thread mapping API exposed to plugins."""

    def __init__(self, *, plugin: str, thread_store) -> None:
        self._plugin = plugin
        self._thread_store = thread_store

    def get_external_thread(self, *, source: str, external_id: str) -> str | None:
        return self._thread_store.get_external_thread(owner_plugin=self._plugin, source=source, external_id=external_id)

    def get_or_create_external_thread(
        self,
        *,
        source: str,
        external_id: str,
        title: str = "New thread",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._thread_store.get_or_create_external_thread(
            owner_plugin=self._plugin,
            source=source,
            external_id=external_id,
            title=title,
            metadata=metadata,
        )

    def create_thread(
        self,
        title: str = "New thread",
        *,
        kind: str = "thread",
        parent_thread_id: str | None = None,
        parent_turn_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> str:
        return self._thread_store.create_thread(
            title,
            kind=kind,
            parent_thread_id=parent_thread_id,
            parent_turn_id=parent_turn_id,
            parent_run_id=parent_run_id,
        )

    def metadata(self, thread_id: str) -> dict[str, Any]:
        return self._thread_store.thread_metadata(thread_id)

    def update_metadata(self, thread_id: str, metadata: dict[str, Any]) -> None:
        self._thread_store.update_thread_metadata(thread_id, updates=dict(metadata))

    def record_event(self, thread_id: str, event_type: str, **data: Any) -> dict[str, Any]:
        return self._thread_store.append(thread_id, event_type, **data)

    def list_threads(self, *, kind: str = "thread") -> list[dict[str, Any]]:
        return self._thread_store.list_threads(kind=kind)

    def recent_events(self, thread_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        events, _ = self._thread_store.read_recent_events(thread_id, limit=limit)
        return events


class PluginRuntimeAPI:
    def __init__(self, *, plugin: str, registry: RuntimeNamespaceRegistry) -> None:
        self._plugin = plugin
        self._registry = registry

    def register_namespace(self, namespace: str, **kwargs: Any) -> RuntimeNamespaceSpec:
        return self._registry.register_namespace(plugin=self._plugin, namespace=namespace, **kwargs)


class PluginActionAPI:
    def __init__(
        self,
        *,
        plugin: str,
        registry: ActionRegistry,
        context_resolver: Callable[[str], Any] | None = None,
    ) -> None:
        self._plugin = plugin
        self._registry = registry
        self._context_resolver = context_resolver or (lambda _plugin: None)

    def register(self, action_id: str, handler: Callable[..., Any], *, doc: str = "", schema: dict[str, Any] | None = None) -> ActionSpec:
        return self._registry.register(plugin=self._plugin, action_id=action_id, handler=handler, doc=doc, schema=schema)

    def resolve(self, action_id: str) -> dict[str, Any]:
        spec = self._registry.get(action_id)
        if spec is None:
            return {"found": False, "action_id": action_id}
        return {
            "found": True,
            "action_id": spec.action_id,
            "plugin": spec.plugin,
            "doc": spec.doc,
            "schema": spec.schema,
        }

    async def call(self, action_id: str, payload: dict[str, Any] | None = None, *, context: Any = None) -> Any:
        spec = self._registry.get(action_id)
        if spec is None:
            raise LookupError(f"Unknown action: {action_id}")
        action_context = context if context is not None else self._context_resolver(spec.plugin)
        return await self._registry.call(action_id, payload or {}, context=action_context)


class PluginCommandAPI:
    def __init__(self, *, plugin: str, registry: CommandRegistry) -> None:
        self._plugin = plugin
        self._registry = registry

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        description: LocalizedText = "",
        aliases: Iterable[str] = (),
    ) -> CommandSpec:
        return self._registry.register(plugin=self._plugin, name=name, handler=handler, description=description, aliases=aliases)


class PluginEpochContextAPI:
    def __init__(self, *, plugin: str, broker: PluginContextBroker, refreshers: list[tuple[str, Callable[..., Any]]]) -> None:
        self._plugin = plugin
        self._broker = broker
        self._refreshers = refreshers

    def publish(self, *, tag: str, body: Any, attrs: Mapping[str, Any] | None = None, thread_id: str | None = None) -> None:
        self._broker.publish(plugin=self._plugin, tag=tag, body=body, attrs=attrs, thread_id=thread_id)

    def update(
        self,
        *,
        tag: str,
        body: Any,
        attrs: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> None:
        self._broker.update(plugin=self._plugin, tag=tag, body=body, attrs=attrs, thread_id=thread_id)

    def remove(self, *, tag: str, reason: str = "", thread_id: str | None = None) -> None:
        self._broker.remove(plugin=self._plugin, tag=tag, reason=reason, thread_id=thread_id)

    def on_refresh(self, handler: Callable[..., Any]) -> PluginRegistration:
        if not callable(handler):
            raise TypeError("Epoch context refresh handler must be callable")
        item = (self._plugin, handler)
        self._refreshers.append(item)

        def dispose() -> None:
            try:
                self._refreshers.remove(item)
            except ValueError:
                pass

        return PluginRegistration(dispose)


class PluginTurnContextAPI:
    def __init__(self, *, plugin: str, broker: PluginContextBroker) -> None:
        self._plugin = plugin
        self._broker = broker

    def enqueue(
        self,
        *,
        thread_id: str,
        tag: str,
        body: Any,
        attrs: Mapping[str, Any] | None = None,
        replay_after_compaction: bool = False,
        replay_key: str | None = None,
    ) -> None:
        self._broker.enqueue_turn(
            plugin=self._plugin,
            thread_id=thread_id,
            tag=tag,
            body=body,
            attrs=attrs,
            replay_after_compaction=replay_after_compaction,
            replay_key=replay_key,
        )

    def clear_replay(self, *, thread_id: str, replay_key: str) -> None:
        self._broker.clear_replay(plugin=self._plugin, thread_id=thread_id, replay_key=replay_key)



class PluginUiAPI:
    def __init__(self, *, plugin: str, registry: UiRegistry) -> None:
        self._plugin = plugin
        self._registry = registry

    def picker(self, *, id: str, title: LocalizedText, provider: Callable[..., Any], trigger: str = "") -> None:
        self._registry.register_picker(
            PickerSource(plugin=self._plugin, id=id, title=title, provider=provider, trigger=trigger)
        )

    def picker_items(self, picker_id: str, query: str = "") -> list[PickerItem]:
        return self._registry.picker_items(picker_id, query=query)


class PluginI18nAPI:
    def __init__(self, *, plugin: str, registry: PluginI18nRegistry) -> None:
        self._plugin = plugin
        self._registry = registry

    def register(self, texts: Mapping[str, LocalizedText]) -> None:
        self._registry.register(plugin=self._plugin, texts=texts)


class PluginCompactionAPI:
    def __init__(self, *, plugin: str, providers: list[tuple[str, Callable[..., str]]]) -> None:
        self._plugin = plugin
        self._providers = providers

    def summary_section(self, provider: Callable[..., str]) -> None:
        if not callable(provider):
            raise TypeError("Compaction summary provider must be callable")
        self._providers.append((self._plugin, provider))


class PluginContext:
    """Capabilities exposed to one uv-agent plugin setup function."""

    def __init__(
        self,
        *,
        manifest: PluginManifest,
        project_root: Path,
        user_state_dir: Path | None,
        config: dict[str, Any],
        events: EventBus,
        logger: logging.Logger,
        runtime_registry: RuntimeNamespaceRegistry,
        action_registry: ActionRegistry,
        command_registry: CommandRegistry,
        ui_registry: UiRegistry,
        i18n_registry: PluginI18nRegistry,
        context_broker: PluginContextBroker,
        storage: PluginStorage,
        submitter: Callable[..., Any] | None,
        task_factory: Callable[[str, Coroutine[Any, Any, Any], str | None], asyncio.Task[Any]],
        compaction_section_providers: list[tuple[str, Callable[..., str]]],
        epoch_context_refreshers: list[tuple[str, Callable[..., Any]]],
        thread_store,
        action_context_resolver: Callable[[str], Any] | None = None,
    ) -> None:
        self.manifest = manifest
        self.plugin_id = manifest.id
        self.name = manifest.id
        self.project_root = project_root
        self.user_state_dir = user_state_dir or uv_agent_home()
        self.config = config
        self.events = events
        self.logger = logger
        self.runtime = PluginRuntimeAPI(plugin=manifest.id, registry=runtime_registry)
        self.actions = PluginActionAPI(plugin=manifest.id, registry=action_registry, context_resolver=action_context_resolver)
        self.commands = PluginCommandAPI(plugin=manifest.id, registry=command_registry)
        self.ui = PluginUiAPI(plugin=manifest.id, registry=ui_registry)
        self.i18n = PluginI18nAPI(plugin=manifest.id, registry=i18n_registry)
        self.compaction = PluginCompactionAPI(plugin=manifest.id, providers=compaction_section_providers)
        self.epoch = PluginEpochContextAPI(plugin=manifest.id, broker=context_broker, refreshers=epoch_context_refreshers)
        self.turn = PluginTurnContextAPI(plugin=manifest.id, broker=context_broker)
        self.storage = storage
        self.threads = PluginThreadAPI(plugin=manifest.id, thread_store=thread_store) if thread_store is not None else None
        self._submitter = submitter
        self._task_factory = task_factory

    def create_task(self, coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task[Any]:
        return self._task_factory(self.plugin_id, coro, name)

    async def submit_turn(
        self,
        *,
        text: str,
        thread_id: str | None = None,
        level: str | None = None,
        image_paths: list[Path] | None = None,
        conflict: str = "queue",
    ) -> SubmittedTurn:
        if self._submitter is None:
            raise RuntimeError("Plugin turn submission is not available")
        raise_if_reentrant_submit()
        return await self._submitter(text=text, thread_id=thread_id, level=level, image_paths=image_paths, conflict=conflict)

    @property
    def can_submit_turn(self) -> bool:
        return self._submitter is not None


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
