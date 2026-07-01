from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from uv_agent.paths import uv_agent_home

from .api import PluginManifest
from .events import EventBus, raise_if_reentrant_submit
from .registry import (
    ActionRegistry,
    ActionSpec,
    CommandRegistry,
    CommandSpec,
    PickerItem,
    PickerSource,
    RuntimeFunctionSpec,
    RuntimeNamespaceRegistry,
    RuntimeNamespaceSpec,
    UiRegistry,
)
from .storage import PluginStorage
from .xml import XmlContribution, render_contribution, render_update_envelope


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


@dataclass(frozen=True)
class UserInput:
    """One external user message that may be coalesced into a supervised turn."""

    text: str
    image_paths: tuple[str | Path, ...] = ()
    request_id: str | None = None


@dataclass(frozen=True)
class TurnPrepareRequest:
    """Read-only information available when building turn context."""

    thread_id: str
    turn_id: str
    user_text: str
    level: str | None = None
    is_new_thread: bool = False
    is_first_turn: bool = False
    created_at: str | None = None
    last_turn_completed_at: str | None = None
    last_assistant_completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnContextBlock:
    """Rendered plugin turn context kept for transitional engine plumbing."""

    text: str
    plugin: str = ""
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpochDocument:
    plugin: str
    tag: str
    body: Any
    attrs: dict[str, Any] = field(default_factory=dict)
    thread_id: str | None = None


class PluginContextBroker:
    """In-memory active context registry owned by the PluginManager.

    ThreadStore persists what was actually sent to the model.  The broker stores
    current plugin documents and pending turn/update notices so Engine can batch
    them into the context item groups required by the refactor plan.
    """

    def __init__(self) -> None:
        self._documents: dict[tuple[str | None, str, str], EpochDocument] = {}
        self._updates: dict[str | None, deque[XmlContribution]] = defaultdict(deque)
        self._turn_items: dict[str, deque[XmlContribution]] = defaultdict(deque)
        self._turn_replay_items: dict[tuple[str, str, str], XmlContribution] = {}
        self._replay: dict[tuple[str, str, str], XmlContribution] = {}

    def publish(
        self,
        *,
        plugin: str,
        tag: str,
        body: Any,
        attrs: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> None:
        key = (thread_id, plugin, tag)
        self._documents[key] = EpochDocument(plugin=plugin, tag=tag, body=body, attrs=dict(attrs or {}), thread_id=thread_id)
        self._updates[thread_id].append(XmlContribution(tag=tag, body=body, attrs={**dict(attrs or {}), "operation": "publish"}))

    def update(
        self,
        *,
        plugin: str,
        tag: str,
        body: Mapping[str, Any],
        attrs: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> None:
        key = (thread_id, plugin, tag)
        previous = self._documents.get(key)
        merged: Any
        if previous is not None and isinstance(previous.body, dict):
            merged = dict(previous.body)
            merged.update(dict(body))
        else:
            merged = dict(body)
        self._documents[key] = EpochDocument(plugin=plugin, tag=tag, body=merged, attrs=dict(attrs or {}), thread_id=thread_id)
        self._updates[thread_id].append(XmlContribution(tag=tag, body=dict(body), attrs={**dict(attrs or {}), "operation": "update"}))

    def remove(
        self,
        *,
        plugin: str,
        tag: str,
        reason: str = "",
        thread_id: str | None = None,
    ) -> None:
        self._documents.pop((thread_id, plugin, tag), None)
        body = {"reason": reason} if reason else {}
        self._updates[thread_id].append(XmlContribution(tag=tag, body=body, attrs={"operation": "remove"}))

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
        contribution = XmlContribution(tag=tag, body=body, attrs=dict(attrs or {}))
        key = (thread_id, plugin, replay_key) if replay_key else None
        if key is not None:
            previous = self._turn_replay_items.pop(key, None)
            if previous is not None:
                self._remove_turn_contribution(thread_id, previous)
            self._turn_replay_items[key] = contribution
        self._turn_items[thread_id].append(contribution)
        if replay_after_compaction and replay_key:
            self._replay[(thread_id, plugin, replay_key)] = contribution

    def clear_replay(self, *, plugin: str, thread_id: str, replay_key: str) -> None:
        key = (thread_id, plugin, replay_key)
        self._replay.pop(key, None)
        previous = self._turn_replay_items.pop(key, None)
        if previous is not None:
            self._remove_turn_contribution(thread_id, previous)

    def _remove_turn_contribution(self, thread_id: str, contribution: XmlContribution) -> None:
        queue = self._turn_items.get(thread_id)
        if not queue:
            return
        self._turn_items[thread_id] = deque(item for item in queue if item != contribution)

    def full_context_text(self, thread_id: str, *, core_texts: Iterable[str], include_document: Callable[[EpochDocument], bool] | None = None) -> str:
        parts = [text for text in core_texts if text]
        documents = sorted(
            (
                document
                for key, document in self._documents.items()
                if (key[0] is None or key[0] == thread_id)
                and (include_document is None or include_document(document))
            ),
            key=lambda item: (item.thread_id is not None, item.plugin, item.tag),
        )
        parts.extend(
            render_contribution(document.tag, document.body, attrs=document.attrs)
            for document in documents
        )
        # Full epoch context supersedes any startup publish/update notices queued
        # before the thread had received its first full item.
        self._updates.pop(None, None)
        self._updates.pop(thread_id, None)
        return "\n\n".join(parts)

    def update_context_text(self, thread_id: str) -> str:
        contributions = [*self._updates.pop(None, ()), *self._updates.pop(thread_id, ())]
        if not contributions:
            return ""
        return render_update_envelope(contributions)

    def turn_context_text(self, thread_id: str) -> str:
        contributions = list(self._turn_items.pop(thread_id, ()))
        if not contributions:
            return ""
        return "\n\n".join(
            render_contribution(item.tag, item.body, attrs=item.attrs)
            for item in contributions
        )

    def replay_after_compaction(self, thread_id: str) -> None:
        for (stored_thread_id, _plugin, _key), contribution in list(self._replay.items()):
            if stored_thread_id == thread_id:
                self._turn_items[thread_id].append(contribution)
                self._replay.pop((stored_thread_id, _plugin, _key), None)


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

    def metadata(self, thread_id: str) -> dict[str, Any]:
        return self._thread_store.thread_metadata(thread_id)

    def update_metadata(self, thread_id: str, metadata: dict[str, Any]) -> None:
        self._thread_store.update_thread_metadata(thread_id, updates=dict(metadata))

    def list_threads(self, *, kind: str = "thread") -> list[dict[str, Any]]:
        return self._thread_store.list_threads() if kind == "thread" else self._thread_store.list_subthreads()

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
    def __init__(self, *, plugin: str, registry: ActionRegistry) -> None:
        self._plugin = plugin
        self._registry = registry

    def register(self, action_id: str, handler: Callable[..., Any], *, doc: str = "", schema: dict[str, Any] | None = None) -> ActionSpec:
        return self._registry.register(plugin=self._plugin, action_id=action_id, handler=handler, doc=doc, schema=schema)


class PluginCommandAPI:
    def __init__(self, *, plugin: str, registry: CommandRegistry) -> None:
        self._plugin = plugin
        self._registry = registry

    def register(self, name: str, handler: Callable[..., Any], *, description: str = "", aliases: Iterable[str] = ()) -> CommandSpec:
        return self._registry.register(plugin=self._plugin, name=name, handler=handler, description=description, aliases=aliases)


class PluginEpochContextAPI:
    def __init__(self, *, plugin: str, broker: PluginContextBroker) -> None:
        self._plugin = plugin
        self._broker = broker

    def publish(self, *, tag: str, body: Any, attrs: Mapping[str, Any] | None = None, thread_id: str | None = None) -> None:
        self._broker.publish(plugin=self._plugin, tag=tag, body=body, attrs=attrs, thread_id=thread_id)

    def update(self, *, tag: str, body: Mapping[str, Any], attrs: Mapping[str, Any] | None = None, thread_id: str | None = None) -> None:
        self._broker.update(plugin=self._plugin, tag=tag, body=body, attrs=attrs, thread_id=thread_id)

    def remove(self, *, tag: str, reason: str = "", thread_id: str | None = None) -> None:
        self._broker.remove(plugin=self._plugin, tag=tag, reason=reason, thread_id=thread_id)


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

    def status_item(self, *, id: str, label: str, value: str = "", priority: int = 100, style: str = "") -> None:
        from .registry import StatusItem

        self._registry.register_status_item(
            StatusItem(plugin=self._plugin, id=id, label=label, value=value, priority=priority, style=style)
        )

    def picker(self, *, id: str, title: str, provider: Callable[..., Any], trigger: str = "") -> None:
        self._registry.register_picker(
            PickerSource(plugin=self._plugin, id=id, title=title, provider=provider, trigger=trigger)
        )

    def picker_items(self, picker_id: str, query: str = "") -> list[PickerItem]:
        return self._registry.picker_items(picker_id, query=query)

class PluginContextAPI:
    def __init__(self, *, plugin: str, broker: PluginContextBroker) -> None:
        self.epoch = PluginEpochContextAPI(plugin=plugin, broker=broker)
        self.turn = PluginTurnContextAPI(plugin=plugin, broker=broker)


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
        context_broker: PluginContextBroker,
        storage: PluginStorage,
        submitter: Callable[..., Any] | None,
        task_factory: Callable[[str, Coroutine[Any, Any, Any], str | None], asyncio.Task[Any]],
        thread_store,
        host: Any = None,
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
        self.actions = PluginActionAPI(plugin=manifest.id, registry=action_registry)
        self.commands = PluginCommandAPI(plugin=manifest.id, registry=command_registry)
        self.ui = PluginUiAPI(plugin=manifest.id, registry=ui_registry)
        self.context = PluginContextAPI(plugin=manifest.id, broker=context_broker)
        self.storage = storage
        self.threads = PluginThreadAPI(plugin=manifest.id, thread_store=thread_store) if thread_store is not None else None
        self.host = host
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
    ) -> SubmittedTurn:
        if self._submitter is None:
            raise RuntimeError("Plugin turn submission is not available")
        raise_if_reentrant_submit()
        return await self._submitter(text=text, thread_id=thread_id, level=level, image_paths=image_paths)


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
