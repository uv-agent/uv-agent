"""Guards against regressions in TUI/model startup-time imports.

The recent lazy-import work moved provider SDKs (OpenAI, Anthropic, MCP),
image/markdown rendering libraries, and Pygments out of module-level imports
so the Textual TUI can reach first paint without paying their cost. These
tests pin that behavior: importing ``uv_agent.model``, ``uv_agent.tui.app``,
``uv_agent.errors``, or ``uv_agent.thread_titles`` must not pull those heavy
dependencies into ``sys.modules`` as a side effect.

We spawn fresh interpreters because once *any* other test imports e.g.
``openai`` the module stays cached in this process, which would mask the
regression we are trying to detect.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


# Modules that must NOT be loaded as a side effect of importing the targets
# below. Each entry is a top-level package name as it appears in
# ``sys.modules``.
HEAVY_MODULES = (
    "openai",
    "anthropic",
    "mcp",
    "PIL",
    "textual_image",
    "pygments",
    "rich.markdown",
)


def _assert_imports_stay_lazy(import_target: str) -> None:
    """Run ``import <import_target>`` in a fresh interpreter and assert no
    heavy dependency was loaded as a side effect."""
    script = textwrap.dedent(
        f"""
        import sys
        import importlib

        importlib.import_module({import_target!r})

        leaked = sorted(
            name for name in {HEAVY_MODULES!r} if name in sys.modules
        )
        if leaked:
            print("LEAKED:" + ",".join(leaked))
            raise SystemExit(1)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Importing {import_target!r} pulled in heavy modules.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_import_uv_agent_model_stays_lazy() -> None:
    _assert_imports_stay_lazy("uv_agent.model")


def test_import_uv_agent_errors_stays_lazy() -> None:
    _assert_imports_stay_lazy("uv_agent.errors")


def test_import_uv_agent_thread_titles_stays_lazy() -> None:
    _assert_imports_stay_lazy("uv_agent.thread_titles")


def test_import_uv_agent_tui_app_stays_lazy() -> None:
    # The TUI app module is the main beneficiary of the lazy-import work: it
    # must be importable without dragging in provider SDKs, MCP, Pillow,
    # textual_image, Pygments, or Rich's Markdown renderer.
    _assert_imports_stay_lazy("uv_agent.tui.app")
