from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("uv_agent.host_events")


@runtime_checkable
class SyncEventHandler(Protocol):
    """Synchronous host event handler; called in registration order."""

    def __call__(self, event: dict[str, Any]) -> None: ...


@runtime_checkable
class AsyncEventHandler(Protocol):
    """Asynchronous host event handler; scheduled without blocking the publisher."""

    def __call__(self, event: dict[str, Any]) -> Awaitable[None]: ...


EventHandler = SyncEventHandler | AsyncEventHandler


class HostEventBus:
    """Lightweight, ordered, synchronous event bus for host-internal subsystems.

    ``HostEventBus`` is intentionally simple: handlers are called directly in the
    publisher thread, preserving event order and avoiding background queues that
    could grow unbounded.  Handlers that need async behavior can schedule their
    own tasks inside a synchronous wrapper.

    The plugin event system is wired in as a subsystem via
    ``register_plugin_bus``.  Events whose type starts with ``plugin.`` or whose
    ``scope`` is ``"plugin"`` are forwarded to the plugin bus so external plugins
    can observe host lifecycle events without host code needing to know about
    plugins.
    """

    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []
        self._lock = threading.RLock()
        self._plugin_bus: Any | None = None

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        """Register an event handler and return an unsubscribe callable."""

        with self._lock:
            self._handlers.append(handler)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._handlers.remove(handler)
                except ValueError:
                    pass

        return unsubscribe

    def register_plugin_bus(self, plugin_bus: Any) -> None:
        """Attach the plugin event system as a subsystem.

        After registration, host events scoped for plugins are forwarded to the
        plugin bus.  The plugin bus itself is not exposed as a generic host
        handler so that plugin lifecycle concerns stay isolated.
        """

        with self._lock:
            self._plugin_bus = plugin_bus

    def close(self) -> None:
        """Notify all handlers that support ``close`` to release resources."""

        with self._lock:
            handlers = list(self._handlers)
        for handler in handlers:
            close_fn = getattr(handler, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    logger.exception("Host event handler close failed")

    def publish(self, event: dict[str, Any]) -> None:
        """Deliver an event to all registered handlers in order.

        Handler errors are logged and swallowed so that one misbehaving consumer
        cannot break the publisher or other consumers.
        """

        if not isinstance(event, dict):
            logger.warning("Ignoring non-dict host event: %r", event)
            return

        with self._lock:
            handlers = list(self._handlers)
            plugin_bus = self._plugin_bus

        for handler in handlers:
            try:
                result = handler(event)
                if result is not None and inspect.isawaitable(result):
                    # Async handlers are fire-and-forget.  Host bus does not
                    # manage an event loop; callers that need awaitable
                    # handlers must arrange their own scheduling.
                    try:
                        import asyncio

                        asyncio.get_running_loop().create_task(result)
                    except RuntimeError:
                        logger.warning(
                            "Async host event handler %r has no running event loop; skipping",
                            handler,
                        )
            except Exception:
                logger.exception("Host event handler failed for %s", event.get("type"))

        self._forward_to_plugin_bus(event, plugin_bus)

    def _forward_to_plugin_bus(self, event: dict[str, Any], plugin_bus: Any | None) -> None:
        if plugin_bus is None:
            return
        event_type = str(event.get("type") or "")
        scope = str(event.get("scope") or "")
        if scope != "plugin" and not event_type.startswith("plugin."):
            return
        try:
            plugin_bus.publish(event)
        except Exception:
            logger.exception("Failed to forward host event to plugin bus: %s", event_type)
