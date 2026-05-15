from __future__ import annotations

from uv_agent.runner.metadata import ensure_dependency


def test_ensure_dependency_adds_metadata_when_missing() -> None:
    code = "print('hi')\n"
    updated = ensure_dependency(code, "uv-agent @ file:///tmp/uv-agent", "uv-agent")
    assert updated.startswith("# /// script")
    assert "uv-agent @ file:///tmp/uv-agent" in updated
    assert updated.endswith(code)


def test_ensure_dependency_does_not_duplicate_existing_runtime() -> None:
    code = """# /// script
# dependencies = [
#   "uv-agent-runtime>=0.1,<0.2",
# ]
# ///
print("hi")
"""
    updated = ensure_dependency(code, "uv-agent-runtime>=0.1,<0.2", "uv-agent-runtime")
    assert updated.count("uv-agent-runtime") == 1


def test_ensure_dependency_inserts_into_existing_dependencies() -> None:
    code = """# /// script
# dependencies = [
#   "rich",
# ]
# ///
print("hi")
"""
    updated = ensure_dependency(code, "uv-agent-runtime>=0.1,<0.2", "uv-agent-runtime")
    assert "rich" in updated
    assert "uv-agent-runtime>=0.1,<0.2" in updated
