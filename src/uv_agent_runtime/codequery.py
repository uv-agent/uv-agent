"""tree-sitter backed code query helpers with an incremental SQLite cache.

Languages are loaded lazily through ``tree_sitter_language_pack``. Parse trees
themselves are not cached (the Python bindings do not provide a stable
serialization); instead, each query's captures are stored per file and reused
whenever the file's ``(mtime_ns, size)`` is unchanged.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Node, Parser, Query, QueryCursor
from tree_sitter_language_pack import (
    LanguageNotFoundError,
    detect_language_from_path,
    get_language,
)

from . import codesearch
from .files import resolve_workspace_path


@dataclass(frozen=True)
class Capture:
    """A single tree-sitter capture located in a file."""

    name: str
    path: str
    language: str
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    text: str


@dataclass(frozen=True)
class Symbol:
    """A code symbol (function, class, method, ...) found in a file."""

    kind: str
    name: str
    path: str
    language: str
    start_row: int
    end_row: int


# Pre-baked symbol queries. Each capture name maps directly to ``Symbol.kind``.
# Keep them focused on identifier nodes so ``text`` is the symbol name and the
# enclosing definition's range is captured separately as ``<kind>.body``.
_SYMBOL_QUERIES: dict[str, str] = {
    "python": """
        (function_definition name: (identifier) @function) @function.body
        (class_definition name: (identifier) @class) @class.body
    """,
    "javascript": """
        (function_declaration name: (identifier) @function) @function.body
        (class_declaration name: (identifier) @class) @class.body
        (method_definition name: (property_identifier) @method) @method.body
        (variable_declarator
          name: (identifier) @function
          value: [(arrow_function) (function_expression)]) @function.body
    """,
    "typescript": """
        (function_declaration name: (identifier) @function) @function.body
        (class_declaration name: (type_identifier) @class) @class.body
        (interface_declaration name: (type_identifier) @interface) @interface.body
        (method_signature name: (property_identifier) @method) @method.body
        (method_definition name: (property_identifier) @method) @method.body
        (variable_declarator
          name: (identifier) @function
          value: [(arrow_function) (function_expression)]) @function.body
    """,
    "tsx": """
        (function_declaration name: (identifier) @function) @function.body
        (class_declaration name: (type_identifier) @class) @class.body
        (interface_declaration name: (type_identifier) @interface) @interface.body
        (method_definition name: (property_identifier) @method) @method.body
        (variable_declarator
          name: (identifier) @function
          value: [(arrow_function) (function_expression)]) @function.body
    """,
    "rust": """
        (function_item name: (identifier) @function) @function.body
        (struct_item name: (type_identifier) @struct) @struct.body
        (enum_item name: (type_identifier) @enum) @enum.body
        (trait_item name: (type_identifier) @trait) @trait.body
        (impl_item type: (type_identifier) @impl) @impl.body
        (mod_item name: (identifier) @module) @module.body
    """,
    "go": """
        (function_declaration name: (identifier) @function) @function.body
        (method_declaration name: (field_identifier) @method) @method.body
        (type_declaration (type_spec name: (type_identifier) @type)) @type.body
    """,
    "java": """
        (class_declaration name: (identifier) @class) @class.body
        (interface_declaration name: (identifier) @interface) @interface.body
        (method_declaration name: (identifier) @method) @method.body
        (constructor_declaration name: (identifier) @method) @method.body
    """,
    "c": """
        (function_definition declarator: (function_declarator declarator: (identifier) @function)) @function.body
        (struct_specifier name: (type_identifier) @struct) @struct.body
    """,
    "cpp": """
        (function_definition declarator: (function_declarator declarator: (identifier) @function)) @function.body
        (class_specifier name: (type_identifier) @class) @class.body
        (struct_specifier name: (type_identifier) @struct) @struct.body
        (namespace_definition name: (namespace_identifier) @module) @module.body
    """,
    "ruby": """
        (method name: (identifier) @method) @method.body
        (singleton_method name: (identifier) @method) @method.body
        (class name: (constant) @class) @class.body
        (module name: (constant) @module) @module.body
    """,
}


_db_lock = threading.Lock()
_SCHEMA_VERSION = 1


def _cache_root() -> Path:
    override = os.environ.get("UV_AGENT_HOME")
    base = Path(override).expanduser() if override else Path.home() / ".uv-agent"
    return (base / "cache" / "codequery").resolve()


def _db_path() -> Path:
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / "index.sqlite"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_db_path(), isolation_level=None, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            root TEXT NOT NULL,
            rel_path TEXT NOT NULL,
            language TEXT NOT NULL,
            mtime_ns INTEGER NOT NULL,
            size INTEGER NOT NULL,
            PRIMARY KEY (root, rel_path)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS captures (
            root TEXT NOT NULL,
            rel_path TEXT NOT NULL,
            query_sha TEXT NOT NULL,
            captures_json TEXT NOT NULL,
            PRIMARY KEY (root, rel_path, query_sha)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS captures_by_query ON captures(root, query_sha)"
    )
    conn.execute(
        f"PRAGMA user_version={_SCHEMA_VERSION}"
    )
    try:
        yield conn
    finally:
        conn.close()


@functools.lru_cache(maxsize=64)
def _language(name: str) -> Language:
    return get_language(name)  # type: ignore[arg-type]


@functools.lru_cache(maxsize=64)
def _parser(name: str) -> Parser:
    return Parser(_language(name))


@functools.lru_cache(maxsize=128)
def _query(name: str, query_text: str) -> Query:
    return Query(_language(name), query_text)


def _query_sha(query_text: str) -> str:
    return hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:16]


def _detect_language(path: str | Path) -> str | None:
    name = detect_language_from_path(str(path))
    return name if isinstance(name, str) and name else None


def _captures_for_file(
    *,
    abs_path: Path,
    language: str,
    query_text: str,
) -> list[dict]:
    data = abs_path.read_bytes()
    parser = _parser(language)
    tree = parser.parse(data)
    query = _query(language, query_text)
    cursor = QueryCursor(query)
    grouped = cursor.captures(tree.root_node)
    out: list[dict] = []
    for name, nodes in grouped.items():
        for node in nodes:
            out.append(_node_to_dict(name, node))
    out.sort(key=lambda c: (c["start_row"], c["start_col"], c["name"]))
    return out


def _node_to_dict(name: str, node: Node) -> dict:
    text_bytes = node.text or b""
    return {
        "name": name,
        "start_row": node.start_point.row,
        "start_col": node.start_point.column,
        "end_row": node.end_point.row,
        "end_col": node.end_point.column,
        "text": text_bytes.decode("utf-8", errors="replace"),
    }


def _candidate_files(
    *,
    root: Path,
    languages: set[str] | None,
    globs: Sequence[str] | None,
    file_types: Sequence[str] | None,
    hidden: bool,
    no_ignore: bool,
) -> list[tuple[str, str]]:
    """Return ``[(rel_path, language), ...]`` for files under root."""
    rels = codesearch.find_files(
        root,
        globs=globs,
        file_types=file_types,
        hidden=hidden,
        no_ignore=no_ignore,
    )
    candidates: list[tuple[str, str]] = []
    for rel in rels:
        lang = _detect_language(rel)
        if lang is None:
            continue
        if languages and lang not in languages:
            continue
        candidates.append((rel, lang))
    return candidates


def _refresh_cache(
    *,
    conn: sqlite3.Connection,
    root_key: str,
    root_path: Path,
    candidates: Sequence[tuple[str, str]],
    query_text: str,
    query_sha: str,
    parse_languages: set[str] | None,
) -> dict[tuple[str, str], list[dict]]:
    """Bring the cache in sync for ``candidates`` and return captures per file."""
    existing_files: dict[str, tuple[str, int, int]] = {
        row[0]: (row[1], row[2], row[3])
        for row in conn.execute(
            "SELECT rel_path, language, mtime_ns, size FROM files WHERE root = ?",
            (root_key,),
        )
    }
    existing_captures: dict[str, str] = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT rel_path, captures_json FROM captures WHERE root = ? AND query_sha = ?",
            (root_key, query_sha),
        )
    }

    candidate_set = {rel for rel, _ in candidates}
    results: dict[tuple[str, str], list[dict]] = {}

    conn.execute("BEGIN")
    try:
        for rel, lang in candidates:
            abs_path = (root_path / rel).resolve()
            try:
                stat = abs_path.stat()
            except FileNotFoundError:
                continue
            mtime_ns, size = stat.st_mtime_ns, stat.st_size
            prev = existing_files.get(rel)
            file_unchanged = (
                prev is not None
                and prev[0] == lang
                and prev[1] == mtime_ns
                and prev[2] == size
            )
            cached_json = existing_captures.get(rel) if file_unchanged else None
            if cached_json is not None:
                results[(rel, lang)] = json.loads(cached_json)
                continue
            if parse_languages is not None and lang not in parse_languages:
                # Language without a pre-baked query; record stat but skip parse.
                conn.execute(
                    "INSERT OR REPLACE INTO files(root, rel_path, language, mtime_ns, size)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (root_key, rel, lang, mtime_ns, size),
                )
                continue
            try:
                captures = _captures_for_file(
                    abs_path=abs_path,
                    language=lang,
                    query_text=query_text,
                )
            except (LanguageNotFoundError, OSError, ValueError):
                continue
            payload = json.dumps(captures, ensure_ascii=False)
            conn.execute(
                "INSERT OR REPLACE INTO files(root, rel_path, language, mtime_ns, size)"
                " VALUES (?, ?, ?, ?, ?)",
                (root_key, rel, lang, mtime_ns, size),
            )
            conn.execute(
                "INSERT OR REPLACE INTO captures(root, rel_path, query_sha, captures_json)"
                " VALUES (?, ?, ?, ?)",
                (root_key, rel, query_sha, payload),
            )
            results[(rel, lang)] = captures

        # Prune entries for files that disappeared from the candidate listing.
        stale_files = set(existing_files) - candidate_set
        for rel in stale_files:
            conn.execute(
                "DELETE FROM files WHERE root = ? AND rel_path = ?",
                (root_key, rel),
            )
            conn.execute(
                "DELETE FROM captures WHERE root = ? AND rel_path = ?",
                (root_key, rel),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return results


def query_code(
    query_text: str,
    *,
    language: str,
    root: str | Path = ".",
    globs: Sequence[str] | None = None,
    file_types: Sequence[str] | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
    max_count: int | None = None,
) -> list[Capture]:
    """Run a tree-sitter query (S-expression text) across files of one language.

    Results are cached per file by ``(mtime_ns, size, query_sha)`` so repeated
    queries only re-parse files that changed since the previous call.
    """
    if not query_text.strip():
        raise ValueError("query_text must be non-empty")
    root_path = resolve_workspace_path(root)
    root_key = str(root_path)
    candidates = _candidate_files(
        root=root_path,
        languages={language},
        globs=globs,
        file_types=file_types,
        hidden=hidden,
        no_ignore=no_ignore,
    )
    sha = _query_sha(query_text)
    with _db_lock, _connect() as conn:
        captures_by_file = _refresh_cache(
            conn=conn,
            root_key=root_key,
            root_path=root_path,
            candidates=candidates,
            query_text=query_text,
            query_sha=sha,
            parse_languages={language},
        )
    out: list[Capture] = []
    for (rel, lang), captures in captures_by_file.items():
        for cap in captures:
            out.append(
                Capture(
                    name=cap["name"],
                    path=rel,
                    language=lang,
                    start_row=cap["start_row"],
                    start_col=cap["start_col"],
                    end_row=cap["end_row"],
                    end_col=cap["end_col"],
                    text=cap["text"],
                )
            )
            if max_count is not None and len(out) >= max_count:
                return out
    out.sort(key=lambda c: (c.path, c.start_row, c.start_col))
    return out


def find_symbols(
    root: str | Path = ".",
    *,
    languages: Sequence[str] | None = None,
    kinds: Sequence[str] | None = None,
    name_pattern: str | None = None,
    max_count: int | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
    globs: Sequence[str] | None = None,
) -> list[Symbol]:
    """Find function/class/method/... definitions across the workspace.

    Uses pre-baked tree-sitter queries per supported language. ``kinds``
    filters by capture kind (``function``, ``class``, ``method``, ...).
    ``name_pattern`` is a regular expression matched against the symbol name.
    """
    import re

    root_path = resolve_workspace_path(root)
    root_key = str(root_path)
    lang_set = set(languages) if languages else set(_SYMBOL_QUERIES)
    available = lang_set & set(_SYMBOL_QUERIES)
    if not available:
        return []
    candidates = _candidate_files(
        root=root_path,
        languages=available,
        globs=globs,
        file_types=None,
        hidden=hidden,
        no_ignore=no_ignore,
    )
    kind_filter = {k for k in kinds} if kinds else None
    pattern = re.compile(name_pattern) if name_pattern else None

    # Group candidates by language so each language runs against its own query.
    by_lang: dict[str, list[tuple[str, str]]] = {}
    for rel, lang in candidates:
        by_lang.setdefault(lang, []).append((rel, lang))

    out: list[Symbol] = []
    with _db_lock, _connect() as conn:
        for lang, group in by_lang.items():
            query_text = _SYMBOL_QUERIES[lang]
            sha = _query_sha(query_text)
            captures_by_file = _refresh_cache(
                conn=conn,
                root_key=root_key,
                root_path=root_path,
                candidates=group,
                query_text=query_text,
                query_sha=sha,
                parse_languages={lang},
            )
            for (rel, file_lang), captures in captures_by_file.items():
                bodies: dict[tuple[int, int], dict] = {}
                names: list[dict] = []
                for cap in captures:
                    if cap["name"].endswith(".body"):
                        bodies[(cap["start_row"], cap["start_col"])] = cap
                    else:
                        names.append(cap)
                for cap in names:
                    if kind_filter and cap["name"] not in kind_filter:
                        continue
                    if pattern and not pattern.search(cap["text"]):
                        continue
                    body = _enclosing_body(bodies, cap)
                    end_row = body["end_row"] if body else cap["end_row"]
                    out.append(
                        Symbol(
                            kind=cap["name"],
                            name=cap["text"],
                            path=rel,
                            language=file_lang,
                            start_row=cap["start_row"],
                            end_row=end_row,
                        )
                    )
                    if max_count is not None and len(out) >= max_count:
                        out.sort(key=lambda s: (s.path, s.start_row))
                        return out
    out.sort(key=lambda s: (s.path, s.start_row))
    return out


def _enclosing_body(
    bodies: dict[tuple[int, int], dict],
    cap: dict,
) -> dict | None:
    """Pick the smallest body capture that fully contains the name capture."""
    best: dict | None = None
    for body in bodies.values():
        if body["name"].split(".")[0] != cap["name"]:
            continue
        if not _contains(body, cap):
            continue
        if best is None or _span(body) < _span(best):
            best = body
    return best


def _contains(outer: dict, inner: dict) -> bool:
    if (outer["start_row"], outer["start_col"]) > (inner["start_row"], inner["start_col"]):
        return False
    if (outer["end_row"], outer["end_col"]) < (inner["end_row"], inner["end_col"]):
        return False
    return True


def _span(cap: dict) -> tuple[int, int]:
    return (cap["end_row"] - cap["start_row"], cap["end_col"] - cap["start_col"])


def supported_symbol_languages() -> list[str]:
    """List languages with a built-in symbol query."""
    return sorted(_SYMBOL_QUERIES)


def clear_cache(*, root: str | Path | None = None) -> int:
    """Drop cached rows. Returns the number of removed rows across both tables."""
    with _db_lock, _connect() as conn:
        if root is None:
            cur = conn.execute("SELECT COUNT(*) FROM files")
            files_count = int(cur.fetchone()[0])
            cur = conn.execute("SELECT COUNT(*) FROM captures")
            captures_count = int(cur.fetchone()[0])
            conn.execute("DELETE FROM files")
            conn.execute("DELETE FROM captures")
            return files_count + captures_count
        root_key = str(resolve_workspace_path(root))
        cur = conn.execute(
            "SELECT COUNT(*) FROM files WHERE root = ?", (root_key,)
        )
        files_count = int(cur.fetchone()[0])
        cur = conn.execute(
            "SELECT COUNT(*) FROM captures WHERE root = ?", (root_key,)
        )
        captures_count = int(cur.fetchone()[0])
        conn.execute("DELETE FROM files WHERE root = ?", (root_key,))
        conn.execute("DELETE FROM captures WHERE root = ?", (root_key,))
        return files_count + captures_count


__all__ = [
    "Capture",
    "Symbol",
    "clear_cache",
    "find_symbols",
    "query_code",
    "supported_symbol_languages",
]
