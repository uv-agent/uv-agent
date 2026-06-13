from __future__ import annotations

from uv_agent.helper_calls import (
    extract_import_anchor_chains,
    format_helper_call,
    format_import_anchor_chains,
    runtime_corrected_helper_calls,
)


def test_extract_import_anchor_chains_finds_direct_imports() -> None:
    code = """from uv_agent_runtime import search_text, read_file
import json

hits = search_text("def foo", file_types=["py"])
data = json.loads(read_file(hits[0].path).text)
"""
    chains = extract_import_anchor_chains(code)
    assert chains == ["search_text", "json.loads", "read_file"]


def test_extract_import_anchor_chains_follows_method_chains() -> None:
    code = """from pathlib import Path

p = Path.home().resolve()
"""
    chains = extract_import_anchor_chains(code)
    assert "Path.home.resolve" in chains


def test_extract_import_anchor_chains_ignores_unimported_calls() -> None:
    code = """from uv_agent_runtime import read_file

foo()
read_file("x")
"""
    chains = extract_import_anchor_chains(code)
    assert chains == ["read_file"]


def test_extract_import_anchor_chains_returns_empty_for_invalid_syntax() -> None:
    assert extract_import_anchor_chains("def broken(") == []


def test_format_import_anchor_chains_orders_by_import_and_xn() -> None:
    chains = ["search_text", "read_file", "search_text", "json.loads", "read_file", "read_file"]
    formatted = format_import_anchor_chains(chains)
    assert formatted == ["search_text x2", "read_file x3", "json.loads"]


def test_runtime_corrected_helper_calls_uses_runtime_report_as_source_of_truth() -> None:
    code = """from uv_agent_runtime import path_info, read_file
for _ in range(3):
    path_info('.')
read_file('x.txt')
"""

    calls = runtime_corrected_helper_calls(code, [{"name": "path_info", "count": 3, "source": "runtime"}])

    assert calls == [{"name": "path_info", "args": "", "source": "runtime", "count": 3}]
    assert format_helper_call(calls[0]) == "path_info() x3"


def test_runtime_corrected_helper_calls_distinguishes_empty_runtime_report_from_missing_report() -> None:
    code = "from uv_agent_runtime import path_info\npath_info('.')\n"

    assert runtime_corrected_helper_calls(code, None) == [
        {"name": "path_info", "args": "'.'", "line": 2}
    ]
    assert runtime_corrected_helper_calls(code, []) == []
