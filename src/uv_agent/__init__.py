"""Core package for the uv-agent experiment."""

import platform

__all__ = ["__version__", "DEFAULT_USER_AGENT"]

__version__ = "0.20.0"

DEFAULT_USER_AGENT = (
    f"uv-agent/{__version__} ({platform.system()}; Python/{platform.python_version()})"
)
