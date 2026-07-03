from __future__ import annotations

import asyncio
from typing import Any

import pytest

from uv_agent.host_events import HostEventBus


class FakePluginBus:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def test_host_event_bus_calls_sync_handlers_in_order() -> None:
    bus = HostEventBus()
    calls: list[str] = []

    def first(event: dict[str, Any]) -> None:
        calls.append(f"first:{event.get('type')}")

    def second(event: dict[str, Any]) -> None:
        calls.append(f"second:{event.get('type')}")

    bus.subscribe(first)
    bus.subscribe(second)
    bus.publish({"type": "test.event"})

    assert calls == ["first:test.event", "second:test.event"]


def test_host_event_bus_unsubscribe_works() -> None:
    bus = HostEventBus()
    calls: list[str] = []

    def handler(event: dict[str, Any]) -> None:
        calls.append(str(event.get("type")))

    unsub = bus.subscribe(handler)
    bus.publish({"type": "a"})
    unsub()
    bus.publish({"type": "b"})

    assert calls == ["a"]


def test_host_event_bus_handler_errors_do_not_break_others() -> None:
    bus = HostEventBus()
    calls: list[str] = []

    def bad(event: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    def good(event: dict[str, Any]) -> None:
        calls.append(str(event.get("type")))

    bus.subscribe(bad)
    bus.subscribe(good)
    bus.publish({"type": "x"})

    assert calls == ["x"]


def test_host_event_bus_forwards_plugin_events_to_plugin_bus() -> None:
    bus = HostEventBus()
    plugin_bus = FakePluginBus()
    bus.register_plugin_bus(plugin_bus)

    bus.publish({"type": "plugin.started", "plugin": "demo"})
    bus.publish({"type": "host.event", "scope": "host"})
    bus.publish({"type": "plugin.failed", "plugin": "demo"})

    assert len(plugin_bus.events) == 2
    assert plugin_bus.events[0]["type"] == "plugin.started"
    assert plugin_bus.events[1]["type"] == "plugin.failed"


def test_host_event_bus_forwards_scoped_plugin_events() -> None:
    bus = HostEventBus()
    plugin_bus = FakePluginBus()
    bus.register_plugin_bus(plugin_bus)

    bus.publish({"type": "telemetry", "scope": "plugin"})

    assert len(plugin_bus.events) == 1
    assert plugin_bus.events[0]["type"] == "telemetry"


def test_host_event_bus_forwards_ui_events_to_plugin_bus() -> None:
    bus = HostEventBus()
    plugin_bus = FakePluginBus()
    bus.register_plugin_bus(plugin_bus)

    bus.publish({"type": "runtime.ui.message", "scope": "ui", "message": "Open **link**"})
    bus.publish({"type": "runtime.ui.toast", "message": "Shown by type prefix"})

    assert [event["type"] for event in plugin_bus.events] == [
        "runtime.ui.message",
        "runtime.ui.toast",
    ]


def test_host_event_bus_close_calls_handler_close() -> None:
    bus = HostEventBus()
    closed: list[bool] = []

    class CloseableHandler:
        def __call__(self, event: dict[str, Any]) -> None:
            pass

        def close(self) -> None:
            closed.append(True)

    bus.subscribe(CloseableHandler())
    bus.close()

    assert closed == [True]


@pytest.mark.asyncio
async def test_host_event_bus_schedules_async_handlers() -> None:
    bus = HostEventBus()
    calls: list[str] = []

    async def handler(event: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        calls.append(str(event.get("type")))

    bus.subscribe(handler)
    bus.publish({"type": "async"})

    # Allow the scheduled task to run.
    await asyncio.sleep(0.01)

    assert calls == ["async"]
