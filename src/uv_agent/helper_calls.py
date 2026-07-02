from __future__ import annotations

import ast
from typing import Any

# Keep this list focused on callable helpers exposed in the stable runtime
# context.  Calls imported directly from ``uv_agent_runtime`` are also accepted
# so plugin-provided helpers can still be displayed even though their names are
# not known to the host ahead of time.
RUNTIME_HELPER_NAMES: frozenset[str] = frozenset(
    {
        "apply_patch",
        "cd",
        "compare",
        "convert_patch",
        "diff",
        "dry_run_patch",
        "file",
        "files",
        "look_at",
        "normalize",
        "patch",
        "path",
        "pwd",
        "query",
        "restore",
        "run",
        "search",
        "snapshot",
        "symbols",
        "transaction",
    }
)


RUNTIME_SUBMODULE_NAMES: frozenset[str] = frozenset(
    {
        "deps",
        "events",
        "goals",
        "mcp",
        "threads",
        "workflow",
    }
)


HelperCall = dict[str, Any]


def extract_runtime_helper_calls(source: str, *, max_calls: int = 80) -> list[HelperCall]:
    """Return runtime-helper calls found in a run_python script.

    This is intentionally static and best-effort.  It recognizes calls through a
    module alias such as ``rt.search(...)`` or namespace calls such as
    ``rt.threads.view(...)``.  Direct imports from ``uv_agent_runtime`` are still
    accepted as plugin-provided helpers, but the built-in runtime context now
    recommends ``import uv_agent_runtime as rt``.  It never evaluates user code.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = _RuntimeHelperCallVisitor(source, max_calls=max_calls)
    visitor.visit(tree)
    return visitor.calls


def format_helper_call(call: HelperCall, *, max_chars: int = 160) -> str:
    """Format one extracted or runtime-recorded helper call for display."""

    name = str(call.get("name") or call.get("helper") or "helper")
    args = str(call.get("args") or "")
    text = f"{name}({args})" if args else f"{name}()"
    count = _positive_int(call.get("count")) or 1
    if count > 1:
        text = f"{text} x{count}"
    return _truncate(text, max_chars)


def runtime_corrected_helper_calls(
    source: str,
    runtime_calls: list[HelperCall] | None,
    *,
    max_helpers: int = 80,
) -> list[HelperCall]:
    """Return helper calls corrected by runtime data while preserving static-only calls."""

    if runtime_calls is None:
        return extract_runtime_helper_calls(source, max_calls=max_helpers)
    return summarize_runtime_helper_calls(runtime_calls, max_helpers=max_helpers)


def summarize_runtime_helper_calls(calls: list[HelperCall], *, max_helpers: int = 80) -> list[HelperCall]:
    """Aggregate sanitized runtime helper-call summaries for result payloads."""

    ordered: dict[str, HelperCall] = {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or call.get("helper") or "").strip()
        if not name:
            continue
        count = _positive_int(call.get("count")) or 1
        entry = ordered.get(name)
        if entry is None:
            if len(ordered) >= max_helpers:
                continue
            entry = {"name": name, "args": str(call.get("args") or ""), "source": "runtime", "count": 0}
            ordered[name] = entry
        entry["count"] = int(entry.get("count") or 0) + count
        if isinstance(call.get("outcomes"), dict) or isinstance(call.get("outcome"), str):
            _merge_outcomes(entry, call, count=count)
        _merge_duration(entry, call)
        _merge_string_list(entry, "keyword_names", call.get("keyword_names"), max_items=64)
        _merge_int_list(entry, "positional_counts", call.get("positional_counts"), max_items=16)
        positional_count = _positive_int(call.get("positional_count"))
        if positional_count is not None:
            _merge_int_list(entry, "positional_counts", [positional_count], max_items=16)
        _merge_string_list(entry, "error_types", call.get("error_types"), max_items=32)
        error_type = call.get("error_type")
        if isinstance(error_type, str) and error_type:
            _merge_string_list(entry, "error_types", [error_type], max_items=32)
        if "argument_types" not in entry and isinstance(call.get("argument_types"), dict):
            entry["argument_types"] = call["argument_types"]
    result = list(ordered.values())
    for entry in result:
        duration = entry.get("total_duration_ms")
        if isinstance(duration, float):
            entry["total_duration_ms"] = round(duration, 3)
        if entry.get("count") == 1:
            # Keep the count explicit for machine consumers; display code treats
            # missing and one-count entries the same way.
            entry["count"] = 1
    return result


class _RuntimeHelperCallVisitor(ast.NodeVisitor):
    def __init__(self, source: str, *, max_calls: int) -> None:
        self.source = source
        self.max_calls = max_calls
        self.calls: list[HelperCall] = []
        self.imported_helpers: dict[str, str] = {}
        self.runtime_module_aliases: set[str] = set()
        self.runtime_submodule_aliases: set[str] = set()
        self.star_imported_runtime = False

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast API
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            if alias.name == "uv_agent_runtime":
                self.runtime_module_aliases.add(local)
            elif alias.name.startswith("uv_agent_runtime."):
                self.runtime_submodule_aliases.add(local)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - ast API
        module = node.module or ""
        if module == "uv_agent_runtime":
            for alias in node.names:
                if alias.name == "*":
                    self.star_imported_runtime = True
                    continue
                local = alias.asname or alias.name
                if alias.name in RUNTIME_SUBMODULE_NAMES:
                    self.runtime_submodule_aliases.add(local)
                else:
                    # Accept unknown direct imports too; plugin helpers are
                    # resolved dynamically by uv_agent_runtime.__getattr__.
                    self.imported_helpers[local] = alias.name
        elif module.startswith("uv_agent_runtime."):
            for alias in node.names:
                if alias.name == "*":
                    self.star_imported_runtime = True
                    continue
                local = alias.asname or alias.name
                self.imported_helpers[local] = alias.name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        recorded = False
        if len(self.calls) < self.max_calls:
            name = self._helper_name(node.func)
            if name:
                self.calls.append(
                    {
                        "name": name,
                        "args": self._format_arguments(node),
                        "line": getattr(node, "lineno", None),
                    }
                )
                recorded = True
        if recorded:
            # The outer call already captures chains such as
            # ``rt.file('x').replace(...)``.  Visiting the callee expression would
            # add a second, less useful ``file('x')`` entry, but arguments may
            # still contain independent helper calls.
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return
        self.generic_visit(node)

    def _helper_name(self, func: ast.expr) -> str | None:
        if isinstance(func, ast.Name):
            if func.id in self.imported_helpers:
                return self.imported_helpers[func.id]
            if self.star_imported_runtime and func.id in RUNTIME_HELPER_NAMES:
                return func.id
            return None
        if isinstance(func, ast.Attribute):
            chain = self._runtime_call_chain(func)
            if chain:
                return chain
            if self._is_runtime_submodule_expr(func.value):
                # For imported runtime submodules we accept unknown attributes as
                # well, because plugin/submodule helpers may not be in the stable
                # top-level helper list.
                return func.attr
        return None

    def _runtime_call_chain(self, expr: ast.expr) -> str | None:
        if isinstance(expr, ast.Attribute):
            prefix = self._runtime_call_chain(expr.value)
            if prefix is not None:
                return f"{prefix}.{expr.attr}"
            if self._is_runtime_module_expr(expr.value):
                if expr.attr in RUNTIME_HELPER_NAMES or expr.attr in RUNTIME_SUBMODULE_NAMES:
                    return expr.attr
            if self._is_runtime_submodule_expr(expr.value):
                submodule = self._runtime_call_chain(expr.value)
                return f"{submodule}.{expr.attr}" if submodule else expr.attr
            return None
        if isinstance(expr, ast.Call):
            return self._runtime_call_chain(expr.func)
        if isinstance(expr, ast.Name):
            if expr.id in self.imported_helpers:
                return self.imported_helpers[expr.id]
            if self.star_imported_runtime and expr.id in RUNTIME_HELPER_NAMES:
                return expr.id
        return None

    def _is_runtime_module_expr(self, expr: ast.expr) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in self.runtime_module_aliases
        return False

    def _is_runtime_submodule_expr(self, expr: ast.expr) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in self.runtime_submodule_aliases
        if isinstance(expr, ast.Attribute):
            return self._is_runtime_module_expr(expr.value) and expr.attr in RUNTIME_SUBMODULE_NAMES
        return False

    def _format_arguments(self, node: ast.Call) -> str:
        parts: list[str] = []
        for arg in node.args:
            prefix = "*" if isinstance(arg, ast.Starred) else ""
            value = arg.value if isinstance(arg, ast.Starred) else arg
            parts.append(prefix + self._expr_source(value))
        for keyword in node.keywords:
            value = self._expr_source(keyword.value)
            if keyword.arg is None:
                parts.append("**" + value)
            else:
                parts.append(f"{keyword.arg}={value}")
        return ", ".join(parts)

    def _expr_source(self, expr: ast.expr) -> str:
        segment = ast.get_source_segment(self.source, expr)
        if segment is None:
            try:
                segment = ast.unparse(expr)
            except Exception:
                segment = "…"
        return _compact(segment)


def _compact(value: str, *, max_chars: int = 120) -> str:
    text = " ".join(value.strip().split())
    return _truncate(text, max_chars)


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_outcomes(entry: HelperCall, call: HelperCall, *, count: int) -> None:
    outcomes = entry.setdefault("outcomes", {})
    if not isinstance(outcomes, dict):
        return
    call_outcomes = call.get("outcomes")
    if isinstance(call_outcomes, dict):
        for key, value in call_outcomes.items():
            amount = _positive_int(value) or 0
            if amount:
                outcomes[str(key)] = int(outcomes.get(str(key)) or 0) + amount
        return
    outcome = call.get("outcome")
    if isinstance(outcome, str) and outcome:
        outcomes[outcome] = int(outcomes.get(outcome) or 0) + count


def _merge_duration(entry: HelperCall, call: HelperCall) -> None:
    duration = _float_or_none(call.get("total_duration_ms"))
    if duration is None:
        duration = _float_or_none(call.get("duration_ms"))
    if duration is None:
        return
    entry["total_duration_ms"] = float(entry.get("total_duration_ms") or 0.0) + max(0.0, duration)


def _merge_string_list(entry: HelperCall, key: str, values: Any, *, max_items: int) -> None:
    if not isinstance(values, list):
        return
    target = entry.setdefault(key, [])
    if not isinstance(target, list):
        return
    seen = {str(item) for item in target}
    for value in values:
        if len(target) >= max_items:
            break
        text = str(value)
        if text and text not in seen:
            target.append(text)
            seen.add(text)


def _merge_int_list(entry: HelperCall, key: str, values: Any, *, max_items: int) -> None:
    if not isinstance(values, list):
        return
    target = entry.setdefault(key, [])
    if not isinstance(target, list):
        return
    seen = {int(item) for item in target if isinstance(item, int)}
    for value in values:
        if len(target) >= max_items:
            break
        parsed = _positive_int(value)
        if parsed is not None and parsed not in seen:
            target.append(parsed)
            seen.add(parsed)


def extract_import_anchor_chains(source: str, *, max_calls: int = 80) -> list[str]:
    """Return call chains anchored by imported names, in import order.

    A chain starts at a name that was imported into the module (e.g.
    ``rt``, ``Path``, or ``json``) and follows attribute access and call
    expressions that use that name directly.  For example:

        from pathlib import Path
        import uv_agent_runtime as rt
        import json

        rt.search("foo")
        Path.home().resolve()
        json.loads(s)

    produces ``["rt.search", "Path.home.resolve", "json.loads"]``.

    This is intentionally best-effort and literal: it does not trace variables
    assigned from imported objects, so ``p = Path.home(); p.resolve()`` is not
    recognised as ``Path.home.resolve``.  It also stops at non-import names,
    so ``foo().bar()`` without an imported anchor is skipped.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = _ImportAnchorChainVisitor(source, max_calls=max_calls)
    visitor.visit(tree)
    return visitor.chains


def format_import_anchor_chains(chains: list[str]) -> list[str]:
    """Deduplicate chains while preserving import order, adding ``xN`` suffixes."""

    counts: dict[str, int] = {}
    ordered: list[str] = []
    for chain in chains:
        counts[chain] = counts.get(chain, 0) + 1
        if counts[chain] == 1:
            ordered.append(chain)
    return [f"{chain} x{counts[chain]}" if counts[chain] > 1 else chain for chain in ordered]


class _ImportAnchorChainVisitor(ast.NodeVisitor):
    def __init__(self, source: str, *, max_calls: int) -> None:
        self.source = source
        self.max_calls = max_calls
        self.chains: list[str] = []
        # imported name -> import order index
        self.imported: dict[str, int] = {}
        self._next_order = 0
        self._star_imported_runtime = False

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast API
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.imported.setdefault(local, self._next_order)
            self._next_order += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - ast API
        module = node.module or ""
        if module == "uv_agent_runtime":
            for alias in node.names:
                if alias.name == "*":
                    self._star_imported_runtime = True
                    continue
                local = alias.asname or alias.name
                self.imported.setdefault(local, self._next_order)
                self._next_order += 1
        elif module:
            for alias in node.names:
                if alias.name == "*":
                    # We do not know the module's exports, so record the
                    # module name as a fallback anchor when it is imported
                    # explicitly elsewhere, but otherwise ignore star imports
                    # from unknown modules.
                    continue
                local = alias.asname or alias.name
                self.imported.setdefault(local, self._next_order)
                self._next_order += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        recorded = False
        if len(self.chains) < self.max_calls:
            chain = self._call_chain(node.func)
            if chain:
                self.chains.append(chain)
                recorded = True
        if recorded:
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return
        self.generic_visit(node)

    def _call_chain(self, func: ast.expr) -> str | None:
        # Unwrap decorators/context-managers such as @workflow.agent(...) if
        # the call is syntactically a call on a callable attribute chain.
        if isinstance(func, ast.Name):
            if func.id in self.imported:
                return func.id
            if self._star_imported_runtime and func.id in RUNTIME_HELPER_NAMES:
                return func.id
            return None
        if isinstance(func, ast.Attribute):
            anchor = self._attribute_anchor(func.value)
            if anchor is not None:
                return f"{anchor}.{func.attr}"
            return None
        return None

    def _attribute_anchor(self, expr: ast.expr) -> str | None:
        """Return the imported-anchored prefix of an attribute/call chain."""

        if isinstance(expr, ast.Call):
            inner = self._call_chain(expr.func)
            if inner is not None:
                return inner
            # The call's func is not itself imported, but its callee might be
            # an attribute chain anchored by an import (e.g. Path.home().x()).
            return self._attribute_anchor(expr.func)
        if isinstance(expr, ast.Attribute):
            prefix = self._attribute_anchor(expr.value)
            if prefix is not None:
                return f"{prefix}.{expr.attr}"
            return None
        if isinstance(expr, ast.Name):
            if expr.id in self.imported:
                return expr.id
            if self._star_imported_runtime and expr.id in RUNTIME_HELPER_NAMES:
                return expr.id
            return None
        return None
