from __future__ import annotations

from uv_agent.helper_calls import (
    extract_import_anchor_chains,
    format_import_anchor_chains,
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
