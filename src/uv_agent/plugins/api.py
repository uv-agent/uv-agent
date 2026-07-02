from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .i18n import LocalizedText

PluginCapability = Literal[
    "runtime_namespace",
    "context",
    "command",
    "action",
    "ui",
    "storage",
]
PluginState = Literal[
    "discovered",
    "disabled",
    "starting",
    "started",
    "warning",
    "failed",
    "stopped",
]


@dataclass(frozen=True)
class PluginManifest:
    """Static plugin declaration consumed before setup runs.

    The manifest is deliberately data-only.  Host code can validate config,
    calculate load order, and render status without importing implementation
    details from other plugins.
    """

    id: str
    version: str
    display_name: LocalizedText
    description: LocalizedText
    builtin: bool = False
    default_enabled: bool = True
    priority: int = 100
    dependencies: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    capabilities: tuple[PluginCapability | str, ...] = ()
    config_schema: dict[str, Any] = field(default_factory=dict)
    storage_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginConfig:
    """Merged per-plugin config.

    ``enabled`` is separate from the arbitrary ``config`` payload so project/user
    layers can toggle a plugin without replacing nested plugin configuration.
    """

    enabled: bool | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginStatus:
    id: str
    display_name: LocalizedText = ""
    state: PluginState = "discovered"
    builtin: bool = False
    first_load: bool = False
    message: str = ""
    error_type: str | None = None


@dataclass(frozen=True)
class SetupPlugin:
    """Plugin object loaded from builtin modules or entry points."""

    manifest: PluginManifest
    setup: Callable[[Any], Any]
    stop: Callable[[Any], Any] | None = None
