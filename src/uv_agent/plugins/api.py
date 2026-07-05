from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .i18n import LocalizedText

PluginActivation = Literal[
    "always",
    "persistent_only",
    "session_only",
]
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
    "skipped",
    "starting",
    "started",
    "warning",
    "failed",
    "stopped",
]
PluginHostInvocation = Literal[
    "tui",
    "daemon",
]
PluginHostLifetime = Literal[
    "session",
    "persistent",
]


@dataclass(frozen=True)
class PluginHostInfo:
    """Read-only host runtime information exposed to plugin setup code."""

    invocation: PluginHostInvocation
    lifetime: PluginHostLifetime
    project_root: Path
    project_state_dir: Path
    user_state_dir: Path

    @property
    def is_persistent(self) -> bool:
        return self.lifetime == "persistent"


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
    activation: PluginActivation = "always"
    config_schema: dict[str, Any] = field(default_factory=dict)
    storage_schema: dict[str, Any] = field(default_factory=dict)
    deprecated: bool = False
    deprecation_message: str = ""


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
    deprecated: bool = False
    deprecation_message: str = ""


@dataclass(frozen=True)
class SetupPlugin:
    """Plugin object loaded from builtin modules or entry points."""

    manifest: PluginManifest
    setup: Callable[[Any], Any]
    stop: Callable[[Any], Any] | None = None
