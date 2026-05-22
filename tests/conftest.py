from __future__ import annotations

import pytest


HOST_RUN_ENV_VARS = (
    "UV_AGENT_RUNTIME_STATE_DIR",
    "UV_AGENT_RUNTIME_THREAD_ID",
    "UV_AGENT_RUNTIME_THREAD_KIND",
    "UV_AGENT_RUNTIME_TURN_ID",
    "UV_AGENT_RUNTIME_RUN_ID",
    "UV_AGENT_SCRIPTENV_DIR",
    "UV_AGENT_SCRIPT_DIR",
    "UV_BIN",
    "UV_AGENT_RUNTIME_PROJECT_STATE_DIR",
)


@pytest.fixture(autouse=True)
def isolate_host_run_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent from an outer uv-agent run_python process."""
    for key in HOST_RUN_ENV_VARS:
        monkeypatch.delenv(key, raising=False)



def pytest_configure() -> None:
    """Avoid textual-image terminal probing under pytest capture.

    ``textual_image.renderable`` asks the terminal about Sixel/TGP support at
    import time when stdout looks like a TTY. In some Windows terminals pytest's
    captured stdin is a pseudofile without ``fileno()``, so that probe can fail
    before tests are collected. The TUI tests exercise image-panel behavior, not
    terminal capability detection, so force the safe non-TTY fallback during
    collection.
    """

    import sys

    class _NonTtyStdout:
        """Tiny proxy preserving stdout behavior except for ``isatty()``."""

        def __init__(self, wrapped: object) -> None:
            self._wrapped = wrapped

        def isatty(self) -> bool:
            return False

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

    if sys.__stdout__ is not None and sys.__stdout__.isatty():
        sys.__stdout__ = _NonTtyStdout(sys.__stdout__)  # type: ignore[assignment]
