from __future__ import annotations

import ast
from typing import Any

# Keep this list focused on callable helpers exposed in the stable runtime
# context.  Calls imported directly from ``uv_agent_runtime`` are also accepted
# so plugin-provided helpers can still be displayed even though their names are
# not known to the host ahead of time.
RUNTIME_HELPER_NAMES: frozenset[str] = frozenset(
    {
        "add_dependencies",
        "add_dependency",
        "apply_patch",
        "apply_patch_any",
        "clear_codequery_cache",
        "compare_text",
        "connect_declared",
        "connect_named",
        "connect_stdio",
        "connect_url",
        "convert_patch",
        "emit_event",
        "emit_progress",
        "emit_result",
        "enter_dir",
        "edit_lines",
        "find_files",
        "find_symbols",
        "goal_paths",
        "helper_stats_db_path",
        "list_declared_servers",
        "list_files",
        "list_thread_digests",
        "look_at",
        "make_unified_diff",
        "normalize_text",
        "path_info",
        "query_code",
        "read_file",
        "read_json",
        "read_text",
        "read_text_lossless",
        "replace_text",
        "resolve_workspace_path",
        "restore_snapshot",
        "run_process_text",
        "run_python_env_dir",
        "search_text",
        "snapshot_files",
        "supported_symbol_languages",
        "thread_detail",
        "thread_digest",
        "thread_view",
        "workspace_transaction",
        "write_file",
        "write_json",
        "write_text",
        "write_text_lossless",
    }
)

RUNTIME_SUBMODULE_NAMES: frozenset[str] = frozenset(
    {
        "codequery",
        "codesearch",
        "cwd",
        "dependencies",
        "events",
        "files",
        "goal_mode",
        "helper_stats",
        "lockfile",
        "mcp",
        "patch",
        "textops",
        "threads",
        "transport",
        "vision",
        "workflow",
    }
)

HelperCall = dict[str, Any]


def extract_runtime_helper_calls(source: str, *, max_calls: int = 80) -> list[HelperCall]:
    """Return runtime-helper calls found in a run_python script.

    This is intentionally static and best-effort.  It recognizes helpers imported
    from ``uv_agent_runtime`` directly, calls through a module alias such as
    ``rt.run_process_text(...)``, and calls through imported runtime submodules
    such as ``textops.replace_text(...)``.  It never evaluates user code.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = _RuntimeHelperCallVisitor(source, max_calls=max_calls)
    visitor.visit(tree)
    return visitor.calls


def format_helper_call(call: HelperCall, *, max_chars: int = 160) -> str:
    """Format one extracted helper call as ``name(args...)`` for display."""

    name = str(call.get("name") or "helper")
    args = str(call.get("args") or "")
    text = f"{name}({args})" if args else f"{name}()"
    return _truncate(text, max_chars)


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
        self.generic_visit(node)

    def _helper_name(self, func: ast.expr) -> str | None:
        if isinstance(func, ast.Name):
            if func.id in self.imported_helpers:
                return self.imported_helpers[func.id]
            if self.star_imported_runtime and func.id in RUNTIME_HELPER_NAMES:
                return func.id
            return None
        if isinstance(func, ast.Attribute):
            if self._is_runtime_module_expr(func.value):
                return func.attr if func.attr in RUNTIME_HELPER_NAMES else None
            if self._is_runtime_submodule_expr(func.value):
                # For imported runtime submodules we accept unknown attributes as
                # well, because plugin/submodule helpers may not be in the stable
                # top-level helper list.
                return func.attr
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



def extract_import_anchor_chains(source: str, *, max_calls: int = 80) -> list[str]:
    """Return call chains anchored by imported names, in import order.

    A chain starts at a name that was imported into the module (e.g.
    ``search_text``, ``Path``, or ``json``) and follows attribute access and
    call expressions that use that name directly.  For example:

        from pathlib import Path
        from uv_agent_runtime import search_text
        import json

        search_text("foo")
        Path.home().resolve()
        json.loads(s)

    produces ``["search_text", "Path.home.resolve", "json.loads"]``.

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
        if len(self.chains) < self.max_calls:
            chain = self._call_chain(node.func)
            if chain:
                self.chains.append(chain)
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
