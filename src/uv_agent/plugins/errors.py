from __future__ import annotations


class PluginError(RuntimeError):
    """Base class for plugin system errors."""


class ReentrantSubmitError(PluginError):
    """Raised when a plugin recursively submits a turn from an event handler."""
