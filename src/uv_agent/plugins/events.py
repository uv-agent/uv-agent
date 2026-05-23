from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .errors import ReentrantSubmitError

EventHandler = Callable[[dict[str, Any]], Awaitable[None]]
_IN_EVENT_HANDLER: contextvars.ContextVar[bool] = contextvars.ContextVar("uv_agent_plugin_event_handler", default=False)


def in_event_handler() -> bool:
    return _IN_EVENT_HANDLER.get()


@dataclass(frozen=True)
class _Subscription:
    kinds: frozenset[str]
    handler: EventHandler
    logger: logging.Logger
    thread_id: str | None = None
    turn_id: str | None = None


class EventBus:
    """In-memory async notification bus for plugins."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscriptions: list[_Subscription] = []
        self._tasks: set[asyncio.Task[None]] = set()

    def subscribe(
        self,
        kinds: list[str] | str,
        handler: EventHandler,
        *,
        logger: logging.Logger | None = None,
        thread_id: str | None = None,
        turn_id: str | None = None,
    ) -> Callable[[], None]:
        if not inspect.iscoroutinefunction(handler):
            raise TypeError("Plugin event handlers must be async functions")
        kind_set = frozenset([kinds] if isinstance(kinds, str) else kinds)
        if not kind_set:
            raise ValueError("At least one event kind is required")
        subscription = _Subscription(
            kinds=kind_set,
            handler=handler,
            logger=logger or logging.getLogger("uv_agent.plugins"),
            thread_id=thread_id,
            turn_id=turn_id,
        )
        with self._lock:
            self._subscriptions.append(subscription)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscriptions.remove(subscription)
                except ValueError:
                    pass

        return unsubscribe

    def publish(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if not event_type:
            return
        with self._lock:
            subscriptions = list(self._subscriptions)
        for subscription in subscriptions:
            if not _matches(subscription, event, event_type):
                continue
            task = asyncio.create_task(self._run_handler(subscription, dict(event)))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def drain(self, *, timeout_s: float = 2.0) -> None:
        with self._lock:
            tasks = list(self._tasks)
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=timeout_s)
        for task in done:
            task.result()
        for task in pending:
            task.cancel()

    async def _run_handler(self, subscription: _Subscription, event: dict[str, Any]) -> None:
        token = _IN_EVENT_HANDLER.set(True)
        try:
            await subscription.handler(event)
        except Exception:
            subscription.logger.exception("Plugin event handler failed for %s", event.get("type"))
        finally:
            _IN_EVENT_HANDLER.reset(token)


def raise_if_reentrant_submit() -> None:
    if in_event_handler():
        raise ReentrantSubmitError("submit_turn cannot be called directly from a plugin event handler")


def _matches(subscription: _Subscription, event: dict[str, Any], event_type: str) -> bool:
    if event_type not in subscription.kinds:
        return False
    if subscription.thread_id is not None and event.get("thread_id") != subscription.thread_id:
        return False
    if subscription.turn_id is not None and event.get("turn_id") != subscription.turn_id:
        return False
    return True
