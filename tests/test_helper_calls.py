from __future__ import annotations

from uv_agent.helper_calls import (
    extract_import_anchor_chains,
    format_helper_call,
    format_import_anchor_chains,
    runtime_corrected_helper_calls,
)


def test_extract_import_anchor_chains_finds_runtime_module_alias() -> None:
    code = """import uv_agent_runtime as rt
import json

hits = rt.search("def foo", types=["py"])
data = json.loads(hits[0].view().text)
"""
    chains = extract_import_anchor_chains(code)
    assert chains == ["rt.search", "json.loads"]


def test_extract_import_anchor_chains_follows_method_chains() -> None:
    code = """from pathlib import Path

p = Path.home().resolve()
"""
    chains = extract_import_anchor_chains(code)
    assert "Path.home.resolve" in chains


def test_extract_import_anchor_chains_ignores_unimported_calls() -> None:
    code = """import uv_agent_runtime as rt

foo()
rt.file("x").read()
"""
    chains = extract_import_anchor_chains(code)
    assert chains == ["rt.file.read"]


def test_extract_import_anchor_chains_returns_empty_for_invalid_syntax() -> None:
    assert extract_import_anchor_chains("def broken(") == []


def test_format_import_anchor_chains_orders_by_import_and_xn() -> None:
    chains = ["rt.search", "rt.file.read", "rt.search", "json.loads", "rt.file.read", "rt.file.read"]
    formatted = format_import_anchor_chains(chains)
    assert formatted == ["rt.search x2", "rt.file.read x3", "json.loads"]


def test_runtime_corrected_helper_calls_uses_runtime_report_as_source_of_truth() -> None:
    code = """import uv_agent_runtime as rt
for _ in range(3):
    rt.path('.')
rt.file('x.txt').read()
"""

    calls = runtime_corrected_helper_calls(code, [{"name": "path", "count": 3, "source": "runtime"}])

    assert calls == [{"name": "path", "args": "", "source": "runtime", "count": 3}]
    assert format_helper_call(calls[0]) == "path() x3"


def test_runtime_corrected_helper_calls_distinguishes_empty_runtime_report_from_missing_report() -> None:
    code = "import uv_agent_runtime as rt\nrt.path('.')\n"

    assert runtime_corrected_helper_calls(code, None) == [
        {"name": "path", "args": "'.'", "line": 2}
    ]
    assert runtime_corrected_helper_calls(code, []) == []
