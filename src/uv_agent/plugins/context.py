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

from uv_agent.blobs import BlobStore
from uv_agent.paths import uv_agent_home

from .api import PluginHostInfo, PluginManifest
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
from .resources import ResourceRegistry
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
class TurnAttachment:
    """One structured attachment reference submitted with a user message."""

    kind: str
    token: str
    blob_id: str
    filename: str = ""
    mime_type: str = "application/octet-stream"
    slot: int | None = None


def coerce_turn_attachment(value: TurnAttachment | Mapping[str, Any]) -> TurnAttachment:
    if isinstance(value, TurnAttachment):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported turn attachment value: {value!r}")
    kind = str(value.get("kind") or "").strip()
    token = str(value.get("token") or "")
    blob_id = str(value.get("blob_id") or "").strip()
    if not kind:
        raise ValueError("Attachment kind is required")
    if not token:
        raise ValueError("Attachment token is required")
    if not blob_id:
        raise ValueError("Attachment blob_id is required")
    raw_slot = value.get("slot")
    slot = int(raw_slot) if raw_slot is not None and raw_slot != "" else None
    return TurnAttachment(
        kind=kind,
        token=token,
        blob_id=blob_id,
        filename=str(value.get("filename") or ""),
        mime_type=str(value.get("mime_type") or "application/octet-stream"),
        slot=slot,
    )


@dataclass(frozen=True)
class UserInput:
    """One external user message that may be coalesced into a supervised turn."""

    text: str
    image_paths: tuple[str | Path, ...] = ()
    attachments: tuple[TurnAttachment, ...] = ()
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

    def update_title(self, thread_id: str, title: str, *, source: str = "plugin") -> None:
        self._thread_store.update_title(thread_id, title, source=source)

    def record_event(self, thread_id: str, event_type: str, **data: Any) -> dict[str, Any]:
        return self._thread_store.append(thread_id, event_type, **data)

    def list_threads(self, *, kind: str = "thread") -> list[dict[str, Any]]:
        return self._thread_store.list_threads(kind=kind)

    def recent_events(self, thread_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        events, _ = self._thread_store.read_recent_events(thread_id, limit=limit)
        return events

    def event_page(
        self,
        thread_id: str,
        *,
        after_event_id: int | None = None,
        before_event_id: int | None = None,
        limit: int = 200,
        event_types: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        if after_event_id is not None and before_event_id is not None:
            raise ValueError("Use either after_event_id or before_event_id, not both")
        normalized_types = set(event_types) if event_types is not None else None
        if before_event_id is not None:
            events, has_more = self._thread_store.read_recent_events(
                thread_id,
                limit=limit,
                before_event_id=before_event_id,
                event_types=normalized_types,
            )
            return {"events": events, "has_more": has_more}
        events, has_more = self._thread_store.read_events_after(
            thread_id,
            after_event_id=after_event_id or 0,
            limit=limit,
            event_types=normalized_types,
        )
        return {"events": events, "has_more": has_more}


class PluginRuntimeAPI:
    def __init__(self, *, plugin: str, registry: RuntimeNamespaceRegistry) -> None:
        self._plugin = plugin
        self._registry = registry

    def register_namespace(self, namespace: str, **kwargs: Any) -> RuntimeNamespaceSpec:
        return self._registry.register_namespace(plugin=self._plugin, namespace=namespace, **kwargs)


class PluginResourceAPI:
    def __init__(self, *, plugin: str, registry: ResourceRegistry) -> None:
        self._plugin = plugin
        self._registry = registry

    def register(self, *, prefix: str, read: Callable[..., Any]) -> PluginRegistration:
        provider = self._registry.register(plugin=self._plugin, prefix=prefix, read=read)

        def dispose() -> None:
            self._registry.unregister(provider.prefix)

        return PluginRegistration(dispose)


class PluginBlobAPI:
    """Narrow project blob API exposed to plugins."""

    def __init__(self, *, blob_store: BlobStore | None) -> None:
        self._blob_store = blob_store

    @property
    def available(self) -> bool:
        return self._blob_store is not None

    def put_bytes(
        self,
        data: bytes,
        *,
        mime_type: str = "application/octet-stream",
        filename: str = "",
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        if self._blob_store is None:
            raise RuntimeError("Plugin blob storage is not available")
        blob = self._blob_store.put_bytes(data, **({"max_bytes": max_bytes} if max_bytes is not None else {}))
        return blob.to_ref(mime_type=mime_type, filename=filename)

    def info(self, blob_id: str) -> dict[str, Any]:
        if self._blob_store is None:
            raise RuntimeError("Plugin blob storage is not available")
        return self._blob_store.info(blob_id)


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

    async def call(
        self,
        action_id: str,
        payload: dict[str, Any] | None = None,
        *,
        context: Any = None,
        missing: str = "error",
    ) -> Any:
        spec = self._registry.get(action_id)
        if spec is None:
            if missing == "ignore":
                return {"ok": False, "missing": True, "action_id": action_id}
            raise LookupError(f"Unknown action: {action_id}")
        action_context = context if context is not None else self._context_resolver(spec.plugin)
        return await self._registry.call(
            action_id,
            payload or {},
            context=action_context,
            missing=missing,  # type: ignore[arg-type]
            caller_plugin=self._plugin,
        )


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


class PluginAgentAPI:
    """Read-only host summaries intended for plugin-built user interfaces."""

    def __init__(self, *, config_getter: Callable[[], Any] | None, ui_registry: UiRegistry) -> None:
        self._config_getter = config_getter
        self._ui_registry = ui_registry

    def model_levels(self) -> dict[str, Any]:
        config = self._config() if self._config_getter is not None else None
        if config is None:
            return {"available": False, "default_level": "", "levels": []}
        public_levels = getattr(config, "public_levels", lambda: {})()
        default_level = str(getattr(getattr(config, "runtime", None), "default_level", "") or "")
        levels: list[dict[str, Any]] = []
        for name, level in public_levels.items():
            item: dict[str, Any] = {
                "id": str(name),
                "label": str(name),
                "model_key": str(getattr(level, "model", "") or ""),
                "is_default": str(name) == default_level,
            }
            try:
                model = config.model_for_level(name)
                provider = config.provider_for_model(model)
            except Exception as exc:
                item["status"] = "error"
                item["error_type"] = exc.__class__.__name__
                item["message"] = str(exc) or repr(exc)
            else:
                item.update(
                    {
                        "status": "available",
                        "model": str(getattr(model, "model", "") or ""),
                        "model_name": str(getattr(model, "name", "") or ""),
                        "provider": str(getattr(model, "provider", "") or ""),
                        "api": str(getattr(model, "api", "") or ""),
                        "context_window_tokens": int(getattr(model, "context_window_tokens", 0) or 0),
                        "supports_images": getattr(model, "supports_images", None),
                        "provider_configured": bool(getattr(provider, "resolved_api_key", lambda: None)()),
                    }
                )
            levels.append(item)
        return {
            "available": True,
            "default_level": default_level,
            "levels": levels,
        }

    def picker_summary(self, picker_ids: Iterable[str] | None = None, *, query: str = "", limit: int = 30) -> dict[str, Any]:
        ids = [str(item) for item in (picker_ids or []) if str(item)]
        if not ids:
            ids = [picker.id for picker in self._ui_registry.pickers()]
        pickers = {picker.id: picker for picker in self._ui_registry.pickers()}
        result: dict[str, Any] = {}
        for picker_id in ids:
            picker = pickers.get(picker_id)
            if picker is None:
                result[picker_id] = {"available": False, "items": []}
                continue
            items = self._ui_registry.picker_items(picker_id, query=query)
            result[picker_id] = {
                "available": True,
                "plugin": picker.plugin,
                "id": picker.id,
                "title": _localized_payload(picker.title),
                "trigger": picker.trigger,
                "items": [_picker_item_summary(item) for item in items[: max(0, int(limit))]],
                "total": len(items),
            }
        return result

    def _config(self) -> Any:
        return self._config_getter() if self._config_getter is not None else None


def _picker_item_summary(item: PickerItem | Mapping[str, Any]) -> dict[str, str]:
    if isinstance(item, Mapping):
        return {
            "id": str(item.get("id") or ""),
            "value": str(item.get("value") or ""),
            "description": str(item.get("description") or ""),
            "kind": str(item.get("kind") or ""),
            "meta": str(item.get("meta") or ""),
        }
    return {
        "id": str(getattr(item, "id", "") or ""),
        "value": str(getattr(item, "value", "") or ""),
        "description": str(getattr(item, "description", "") or ""),
        "kind": str(getattr(item, "kind", "") or ""),
        "meta": str(getattr(item, "meta", "") or ""),
    }


def _localized_payload(value: LocalizedText) -> str | dict[str, str]:
    if isinstance(value, Mapping):
        return {str(key): str(text) for key, text in value.items()}
    return str(value or "")


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
        resource_registry: ResourceRegistry,
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
        blob_store: BlobStore | None = None,
        action_context_resolver: Callable[[str], Any] | None = None,
        agent_config: Callable[[], Any] | None = None,
        host: PluginHostInfo | None = None,
    ) -> None:
        self.manifest = manifest
        self.plugin_id = manifest.id
        self.name = manifest.id
        self.project_root = project_root
        self.user_state_dir = user_state_dir or uv_agent_home()
        self.host = host or PluginHostInfo(
            invocation="tui",
            lifetime="session",
            project_root=project_root,
            project_state_dir=storage.project_data_dir,
            user_state_dir=self.user_state_dir,
        )
        self.config = config
        self.events = events
        self.logger = logger
        self.runtime = PluginRuntimeAPI(plugin=manifest.id, registry=runtime_registry)
        self.resources = PluginResourceAPI(plugin=manifest.id, registry=resource_registry)
        self.blobs = PluginBlobAPI(blob_store=blob_store)
        self.actions = PluginActionAPI(plugin=manifest.id, registry=action_registry, context_resolver=action_context_resolver)
        self.commands = PluginCommandAPI(plugin=manifest.id, registry=command_registry)
        self.ui = PluginUiAPI(plugin=manifest.id, registry=ui_registry)
        self.agent = PluginAgentAPI(config_getter=agent_config, ui_registry=ui_registry)
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
        attachments: list[TurnAttachment | Mapping[str, Any]] | None = None,
        conflict: str = "queue",
    ) -> SubmittedTurn:
        if self._submitter is None:
            raise RuntimeError("Plugin turn submission is not available")
        raise_if_reentrant_submit()
        return await self._submitter(
            text=text,
            thread_id=thread_id,
            level=level,
            image_paths=image_paths,
            attachments=attachments,
            conflict=conflict,
        )

    async def start_turn(
        self,
        *,
        text: str,
        thread_id: str | None = None,
        level: str | None = None,
        image_paths: list[Path] | None = None,
        attachments: list[TurnAttachment | Mapping[str, Any]] | None = None,
        conflict: str = "queue",
    ) -> Any:
        if self._submitter is None:
            raise RuntimeError("Plugin turn submission is not available")
        raise_if_reentrant_submit()
        return await self._submitter(
            text=text,
            thread_id=thread_id,
            level=level,
            image_paths=image_paths,
            attachments=attachments,
            conflict=conflict,
            wait=False,
        )

    @property
    def can_submit_turn(self) -> bool:
        return self._submitter is not None


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
