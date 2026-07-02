"""FFF-backed code search helpers.

The helpers use the native ``fff-search`` Python binding instead of spawning an
external grep binary.  A small in-process finder cache is kept so a single
``run_python`` script can perform several searches against the same root without
paying the indexing cost repeatedly.
"""
from __future__ import annotations

import atexit
import fnmatch
import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .errors import FileSelectionError, HelperRuntimeError, HelperValueError
from .files import resolve_workspace_path

SearchMode = Literal["text", "plain", "literal", "regex", "fuzzy"]


class FffSearchNotAvailableError(HelperRuntimeError):
    """Raised when the native ``fff-search`` binding is unavailable."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            helper="search helpers",
            problem=message or "fff-search Python binding is not available in this run_python environment.",
            hints=(
                "Install project dependencies with `uv sync`, or add the fff-search package to the managed run_python environment.",
                "The search helpers use the `fff` Python module and no longer require an external rg/ripgrep binary.",
            ),
        )

@dataclass(frozen=True)
class Submatch:
    """Byte-range of a single match inside the surrounding line."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Match:
    """A single content-search match line."""

    path: str
    rel_path: str
    line: int
    column: int
    text: str
    submatches: list[Submatch] = field(default_factory=list)
    context_before: list[tuple[int, str]] = field(default_factory=list)
    context_after: list[tuple[int, str]] = field(default_factory=list)

    def file(self):
        import uv_agent_runtime as rt

        return rt.file(self.path)

    def line_range(self, *, context: int = 0) -> tuple[int, int]:
        return (max(1, self.line - context), self.line + context)

    def view(self, *, context: int = 8):
        from .textops import read_file

        try:
            return read_file(self.path, lines=self.line_range(context=context))
        except FileSelectionError as exc:
            if exc.partial_view is not None:
                return exc.partial_view
            raise


_FINDER_CACHE_LIMIT = 8
_FINDER_CACHE: "OrderedDict[str, object]" = OrderedDict()

# Common language/type shortcuts that are convenient for agents.  Unknown
# entries are treated as extension names (``types="proto"`` -> ``*.proto``),
# while extension-looking or glob-looking values are rejected to keep ``types``
# distinct from ``globs``.
_TYPE_GLOBS: dict[str, tuple[str, ...]] = {
    "py": ("*.py", "*.pyi"),
    "python": ("*.py", "*.pyi"),
    "js": ("*.js", "*.jsx", "*.mjs", "*.cjs"),
    "javascript": ("*.js", "*.jsx", "*.mjs", "*.cjs"),
    "ts": ("*.ts", "*.tsx"),
    "typescript": ("*.ts", "*.tsx"),
    "tsx": ("*.tsx",),
    "jsx": ("*.jsx",),
    "rs": ("*.rs",),
    "rust": ("*.rs",),
    "go": ("*.go",),
    "java": ("*.java",),
    "c": ("*.c", "*.h"),
    "cpp": ("*.cc", "*.cpp", "*.cxx", "*.hpp", "*.hh", "*.hxx"),
    "cxx": ("*.cc", "*.cpp", "*.cxx", "*.hpp", "*.hh", "*.hxx"),
    "ruby": ("*.rb",),
    "rb": ("*.rb",),
    "sh": ("*.sh", "*.bash", "*.zsh"),
    "shell": ("*.sh", "*.bash", "*.zsh"),
    "md": ("*.md", "*.markdown"),
    "markdown": ("*.md", "*.markdown"),
    "json": ("*.json",),
    "toml": ("*.toml",),
    "yaml": ("*.yaml", "*.yml"),
    "yml": ("*.yaml", "*.yml"),
    "html": ("*.html", "*.htm"),
    "css": ("*.css",),
    "xml": ("*.xml",),
    "txt": ("*.txt",),
}


def _coerce_str_sequence(value: str | Sequence[str] | None, *, name: str) -> list[str] | None:
    """Accept a scalar string or a sequence for model-friendly list parameters."""

    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        raise HelperValueError(
            helper="search helpers",
            problem=f"{name} must be a string or a sequence of strings, not bytes",
            details={"parameter": name, "received_type": "bytes"},
        )
    try:
        items = list(value)
    except TypeError as exc:
        raise HelperValueError(
            helper="search helpers",
            problem=f"{name} must be a string or a sequence of strings",
            details={"parameter": name, "received_type": type(value).__name__},
            hints=(f"Pass {name}='value' for one item or {name}=['value1', 'value2'] for multiple items.",),
        ) from exc
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise HelperValueError(
                helper="search helpers",
                problem=f"{name}[{index}] must be a string, got {type(item).__name__}",
                details={"parameter": name, "index": index, "received_type": type(item).__name__},
            )
    return items


def _coerce_path_sequence(
    value: str | Path | Sequence[str | Path] | None,
    *,
    name: str,
) -> list[str | Path] | None:
    """Accept either one path-like value or an explicit path sequence."""

    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return [value]
    if isinstance(value, bytes):
        raise HelperValueError(
            helper="search helpers",
            problem=f"{name} must be a path or a sequence of paths, not bytes",
            details={"parameter": name, "received_type": "bytes"},
        )
    try:
        items = list(value)
    except TypeError as exc:
        raise HelperValueError(
            helper="search helpers",
            problem=f"{name} must be a path or a sequence of paths",
            details={"parameter": name, "received_type": type(value).__name__},
            hints=(f"Pass {name}=path for one path or {name}=[path1, path2] for multiple paths.",),
        ) from exc
    for index, item in enumerate(items):
        if not isinstance(item, (str, Path)):
            raise HelperValueError(
                helper="search helpers",
                problem=f"{name}[{index}] must be a path-like value, got {type(item).__name__}",
                details={"parameter": name, "index": index, "received_type": type(item).__name__},
            )
    return items


def _coerce_file_types(value: str | Sequence[str] | None) -> list[str] | None:
    """Normalize model-friendly language/extension aliases."""

    kinds = _coerce_str_sequence(value, name="types")
    if kinds is None:
        return None
    normalized: list[str] = []
    for kind in kinds:
        if not kind:
            raise HelperValueError(
                helper="search helpers",
                problem="types entries must be non-empty aliases such as 'py' or extension names such as 'proto'",
                details={"types": kinds},
            )
        if kind.startswith(".") or any(char in kind for char in "*?[]/\\"):
            raise HelperValueError(
                helper="search helpers",
                problem=(
                    "types uses aliases or extension names such as 'py'/'python'/'rs', "
                    f"not extensions or glob patterns: {kind!r}. Use globs=['*.py'] for path patterns."
                ),
                details={"types": kinds, "invalid_entry": kind},
                hints=("Use types='py' for Python files, or globs=['*.py'] for filename/path patterns.",),
            )
        normalized.append(kind.lower())
    return normalized


def _type_globs(file_types: Sequence[str] | None) -> list[str]:
    globs: list[str] = []
    for kind in file_types or ():
        globs.extend(_TYPE_GLOBS.get(kind, (f"*.{kind}",)))
    return globs


def _normalize_rel_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _normalize_glob(pattern: str) -> str:
    negative = pattern.startswith("!")
    body = pattern[1:] if negative else pattern
    body = body.replace("\\", "/").lstrip("./")
    return f"!{body}" if negative else body


def _glob_variants(pattern: str) -> tuple[str, ...]:
    """Return fnmatch-compatible forms for common recursive glob semantics.

    ``fnmatch`` treats ``**`` as ordinary ``*`` characters, so ``src/**/*.py``
    only matches paths with at least one component below ``src``.  Agents and
    users usually expect shell/gitignore-style ``**/`` to also match zero
    directories.  Keeping this small expansion local preserves the existing
    permissive basename behavior of patterns such as ``*.py``.
    """

    variants = {pattern}
    stack = [pattern]
    while stack:
        current = stack.pop()
        index = current.find("**/")
        while index >= 0:
            collapsed = current[:index] + current[index + 3 :]
            if collapsed not in variants:
                variants.add(collapsed)
                stack.append(collapsed)
            index = current.find("**/", index + 1)
    return tuple(variants)


def _matches_any_glob(rel_path: str, patterns: Sequence[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(rel_path, variant)
        for pattern in patterns
        for variant in _glob_variants(pattern)
    )


def _path_allowed(
    rel_path: str,
    *,
    scope_rel: str | None,
    globs: Sequence[str] | None,
    file_types: Sequence[str] | None,
) -> bool:
    rel = _normalize_rel_path(rel_path)
    if scope_rel is not None and rel != scope_rel:
        return False

    type_patterns = _type_globs(file_types)
    if type_patterns and not _matches_any_glob(rel, type_patterns):
        return False

    normalized_globs = [_normalize_glob(glob) for glob in globs or ()]
    include_globs = [glob for glob in normalized_globs if not glob.startswith("!")]
    exclude_globs = [glob[1:] for glob in normalized_globs if glob.startswith("!")]
    if include_globs and not _matches_any_glob(rel, include_globs):
        return False
    if exclude_globs and _matches_any_glob(rel, exclude_globs):
        return False
    return True


def _abs_path(base: Path, rel_path: str) -> str:
    parts = [part for part in _normalize_rel_path(rel_path).split("/") if part]
    return str((base.joinpath(*parts) if parts else base).resolve())


def _fff_module():
    try:
        import fff  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only in broken envs
        raise FffSearchNotAvailableError() from exc
    return fff


def _finder(base_path: Path, *, refresh: bool = False):
    """Return a cached finder for ``base_path``.

    The cache is intentionally process-local.  Managed scripts run in fresh
    Python processes, but a single script often performs several related searches;
    reusing the native index within that process is where FFF's long-running
    design gives the runtime helpers their biggest win.
    """

    fff = _fff_module()
    key = str(base_path)
    if refresh and key in _FINDER_CACHE:
        old = _FINDER_CACHE.pop(key)
        close = getattr(old, "close", None)
        if callable(close):
            close()
    cached = _FINDER_CACHE.get(key)
    if cached is not None:
        _FINDER_CACHE.move_to_end(key)
        return cached

    try:
        finder = fff.FileFinder(
            base_path,
            watch=False,
            enable_content_indexing=True,
            ai_mode=True,
        )
        if not finder.wait_for_scan_blocking(timeout_ms=120_000):
            finder.close()
            raise HelperRuntimeError(
                helper="search helpers",
                problem=f"fff indexing timed out for {base_path}",
                hints=("Try a narrower root, or pass refresh=True after reducing the file set.",),
            )
    except HelperRuntimeError:
        raise
    except Exception as exc:
        raise HelperRuntimeError(
            helper="search helpers",
            problem=f"fff failed to index {base_path}: {exc}",
        ) from exc

    _FINDER_CACHE[key] = finder
    _FINDER_CACHE.move_to_end(key)
    while len(_FINDER_CACHE) > _FINDER_CACHE_LIMIT:
        _, evicted = _FINDER_CACHE.popitem(last=False)
        close = getattr(evicted, "close", None)
        if callable(close):
            close()
    return finder


def _close_finders() -> None:
    for finder in list(_FINDER_CACHE.values()):
        close = getattr(finder, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    _FINDER_CACHE.clear()


atexit.register(_close_finders)


def _resolve_roots(
    *,
    root: str | Path,
    roots: str | Path | Sequence[str | Path] | None,
) -> list[Path]:
    """Resolve either one root or a multi-root list."""

    if roots is None:
        return [resolve_workspace_path(root)]
    if Path(root) != Path("."):
        raise HelperValueError(
            helper="search helpers",
            problem="root and roots are mutually exclusive",
            hints=("Use root=... for one search root, or leave root='.' and pass roots=[...].",),
        )
    return [resolve_workspace_path(item) for item in _coerce_path_sequence(roots, name="roots") or []]


def _split_root(resolved: Path) -> tuple[Path, str | None]:
    """Return ``(finder_base, optional_single_file_rel)`` for a root."""

    if resolved.is_file():
        return resolved.parent, _normalize_rel_path(resolved.name)
    return resolved, None


def _resolve_context_counts(*, before: int, after: int, context: int | None) -> tuple[int, int]:
    if before < 0 or after < 0:
        raise HelperValueError(helper="search_text", problem="before and after must be >= 0")
    if context is None:
        return before, after
    if context < 0:
        raise HelperValueError(helper="search_text", problem="context must be >= 0")
    if before or after:
        raise HelperValueError(helper="search_text", problem="context is mutually exclusive with before/after")
    return context, context


def _reject_extra_args(extra_args: str | Sequence[str] | None) -> None:
    if extra_args is not None:
        raise HelperValueError(
            helper="search helpers",
            problem="extra_args is not supported by the fff-backed search helpers",
            hints=("Use root/roots, globs, types, mode, context, and limit instead of backend-specific command arguments.",),
        )


def _finder_items(
    finder,
    *,
    query: str,
    page_size: int,
):
    offset = 0
    while True:
        result = finder.search(query, page_index=offset, page_size=page_size)
        items = list(result.items)
        if not items:
            break
        yield from items
        offset += len(items)
        if offset >= int(result.total_matched):
            break


def find_files(
    root: str | Path = ".",
    *,
    roots: str | Path | Sequence[str | Path] | None = None,
    query: str = "",
    globs: str | Sequence[str] | None = None,
    file_types: str | Sequence[str] | None = None,
    types: str | Sequence[str] | None = None,
    max_total: int | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
    extra_args: str | Sequence[str] | None = None,
    refresh: bool = False,
) -> list[str]:
    """List indexed workspace files with optional fuzzy, glob, and type filters.

    ``query`` performs FFF's typo-tolerant filename search. ``globs`` and
    ``types``/``file_types`` are deterministic filters applied to relative paths.
    FFF respects gitignore/ignore files during indexing by default; ``hidden``
    and ``no_ignore`` are accepted by the shared helper signature but do not alter
    the native indexer.
    """

    _reject_extra_args(extra_args)
    if max_total is not None and max_total <= 0:
        return []
    if types is not None and file_types is not None:
        raise HelperValueError(helper="find_files", problem="types and file_types are aliases; pass only one")
    normalized_globs = _coerce_str_sequence(globs, name="globs")
    normalized_file_types = _coerce_file_types(types if types is not None else file_types)
    resolved_roots = _resolve_roots(root=root, roots=roots)

    files: list[str] = []
    remaining = max_total
    for resolved in resolved_roots:
        base_path, scope_rel = _split_root(resolved)
        finder = _finder(base_path, refresh=refresh)
        page_size = max(100, min(1000, remaining or 1000))
        for item in _finder_items(finder, query=query, page_size=page_size):
            rel_path = str(item.relative_path)
            if not _path_allowed(
                rel_path,
                scope_rel=scope_rel,
                globs=normalized_globs,
                file_types=normalized_file_types,
            ):
                continue
            files.append(_abs_path(base_path, rel_path))
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    return files
    return files


def _effective_mode(
    *,
    mode: SearchMode | None,
    fixed_string: bool,
    literal: bool | None,
) -> str:
    if mode is None:
        mode = "text"
    normalized = "text" if mode in {"plain", "literal"} else mode
    if literal is True or fixed_string:
        normalized = "text"
    if normalized not in {"text", "regex", "fuzzy"}:
        raise HelperValueError(
            helper="search_text",
            problem=f"mode must be one of 'text', 'regex', or 'fuzzy', got {mode!r}",
        )
    return normalized



def _prepare_query(
    pattern: str,
    *,
    mode: str,
    ignore_case: bool,
    case_sensitive: bool | None,
    word: bool,
) -> tuple[str, str, bool]:
    """Return ``(query, fff_mode, smart_case)``."""

    force_ignore_case = ignore_case or case_sensitive is False
    force_case_sensitive = case_sensitive is True or not force_ignore_case
    fff_mode = "plain" if mode == "text" else mode
    query = pattern

    if word:
        if mode == "text":
            query = re.escape(query)
        query = rf"\b(?:{query})\b"
        fff_mode = "regex"

    if force_ignore_case:
        if fff_mode == "regex":
            query = f"(?i:{query})"
            return query, fff_mode, False
        # FFF's smart-case mode is case-insensitive only when the query has no
        # uppercase letters. Lowercasing the needle preserves literal semantics
        # while forcing an ignore-case search.
        return query.lower(), fff_mode, True

    return query, fff_mode, not force_case_sensitive


def _submatch_text(line: str, start: int, end: int) -> str:
    return line.encode("utf-8", errors="replace")[start:end].decode("utf-8", errors="replace")



def _literal_submatches(pattern: str, line: str, *, ignore_case: bool) -> list[Submatch]:
    haystack = line.lower() if ignore_case else line
    needle = pattern.lower() if ignore_case else pattern
    out: list[Submatch] = []
    start_index = 0
    while needle:
        found = haystack.find(needle, start_index)
        if found < 0:
            break
        end_index = found + len(needle)
        start = len(line[:found].encode("utf-8", errors="replace"))
        end = len(line[:end_index].encode("utf-8", errors="replace"))
        out.append(Submatch(start=start, end=end, text=line[found:end_index]))
        start_index = max(end_index, found + 1)
    return out


def _match_from_fff(
    base_path: Path,
    item,
    *,
    literal_pattern: str | None,
    ignore_case: bool,
) -> Match:
    rel_path = str(item.relative_path)
    ranges = list(item.match_ranges)
    line_content = str(item.line_content)
    # FFF's regex and fuzzy modes already return backend-accurate byte ranges.
    # Plain mode can return broad ranges for some queries, so reconstruct exact
    # literal spans only when the caller requested text search without word-mode
    # regex expansion.
    submatches = (
        _literal_submatches(literal_pattern, line_content, ignore_case=ignore_case)
        if literal_pattern is not None
        else []
    ) or [
        Submatch(
            start=int(match_range.start),
            end=int(match_range.end),
            text=_submatch_text(line_content, int(match_range.start), int(match_range.end)),
        )
        for match_range in ranges
    ]
    first_col = submatches[0].start + 1 if submatches else int(item.col) + 1
    line_no = int(item.line_number)
    before_lines = list(item.context_before)
    after_lines = list(item.context_after)
    context_before = [
        (line_no - len(before_lines) + index, text)
        for index, text in enumerate(before_lines)
    ]
    context_after = [(line_no + index + 1, text) for index, text in enumerate(after_lines)]
    return Match(
        path=_abs_path(base_path, rel_path),
        rel_path=rel_path,
        line=line_no,
        column=first_col,
        text=str(item.line_content),
        submatches=submatches,
        context_before=context_before,
        context_after=context_after,
    )


def _raise_regex_error(pattern: str, error: str | None) -> None:
    detail = (error or "invalid regex pattern").strip()
    raise HelperRuntimeError(
        helper="search_text",
        problem=f"regex search failed: {detail}",
        details={"pattern": pattern, "mode": "regex"},
        preview_title="regex error",
        preview=detail,
        hints=(
            "rt.search uses plain text by default; omit mode='regex' for exact code strings.",
            "Use mode='regex' only when the pattern is intended to be a regular expression.",
        ),
    )


def search_text(
    pattern: str,
    *,
    root: str | Path = ".",
    roots: str | Path | Sequence[str | Path] | None = None,
    globs: str | Sequence[str] | None = None,
    file_types: str | Sequence[str] | None = None,
    types: str | Sequence[str] | None = None,
    mode: SearchMode | None = None,
    ignore_case: bool = False,
    case_sensitive: bool | None = None,
    fixed_string: bool = False,
    literal: bool | None = None,
    multiline: bool = False,
    word: bool = False,
    before: int = 0,
    after: int = 0,
    context: int | None = None,
    max_count_per_file: int | None = None,
    max_total: int | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
    extra_args: str | Sequence[str] | None = None,
    refresh: bool = False,
) -> list[Match]:
    """Search file contents with FFF and return structured matches.

    The default mode is plain text, which is safer for exact code snippets than
    regular expressions. Use ``mode="regex"`` for regex patterns or
    ``mode="fuzzy"`` for typo-tolerant line search. ``literal`` and
    ``fixed_string`` force plain-text matching.
    """

    _reject_extra_args(extra_args)
    if not pattern:
        raise HelperValueError(helper="search_text", problem="pattern must be non-empty")
    if multiline:
        raise HelperValueError(
            helper="search_text",
            problem="multiline content search is not supported by the fff-backed search helper",
            hints=("Search for a line-local fragment, or run a custom Python scan when matching across line boundaries is required.",),
        )
    if types is not None and file_types is not None:
        raise HelperValueError(helper="search_text", problem="types and file_types are aliases; pass only one")

    before, after = _resolve_context_counts(before=before, after=after, context=context)
    normalized_globs = _coerce_str_sequence(globs, name="globs")
    normalized_file_types = _coerce_file_types(types if types is not None else file_types)
    effective_mode = _effective_mode(mode=mode, fixed_string=fixed_string, literal=literal)
    query, fff_mode, smart_case = _prepare_query(
        pattern,
        mode=effective_mode,
        ignore_case=ignore_case,
        case_sensitive=case_sensitive,
        word=word,
    )
    if max_total is not None and max_total <= 0:
        return []

    matches: list[Match] = []
    remaining = max_total
    resolved_roots = _resolve_roots(root=root, roots=roots)
    for resolved in resolved_roots:
        base_path, scope_rel = _split_root(resolved)
        finder = _finder(base_path, refresh=refresh)
        cursor = None
        has_path_filters = bool(scope_rel or normalized_globs or normalized_file_types)
        while True:
            # Use pages even when the caller asks for no global limit so filtered
            # searches can skip non-matching files without asking FFF for the
            # entire repository in one Python object.  When a small global limit
            # is combined with path filters, request a full page so we do not
            # crawl one disallowed match at a time before reaching an allowed file.
            page_limit = 1000 if remaining is None or has_path_filters else min(1000, remaining)
            try:
                result = finder.grep(
                    query,
                    mode=fff_mode,
                    max_matches_per_file=max_count_per_file or 0,
                    smart_case=smart_case,
                    cursor=cursor,
                    page_limit=page_limit,
                    before_context=before,
                    after_context=after,
                    classify_definitions=True,
                )
            except Exception as exc:
                raise HelperRuntimeError(
                    helper="search_text",
                    problem=f"fff search failed for {base_path}: {exc}",
                    details={"pattern": pattern, "mode": effective_mode},
                ) from exc

            if fff_mode == "regex" and result.regex_fallback_error:
                _raise_regex_error(pattern, result.regex_fallback_error)

            for item in result.items:
                rel_path = str(item.relative_path)
                if not _path_allowed(
                    rel_path,
                    scope_rel=scope_rel,
                    globs=normalized_globs,
                    file_types=normalized_file_types,
                ):
                    continue
                matches.append(
                    _match_from_fff(
                        base_path,
                        item,
                        literal_pattern=pattern if effective_mode == "text" and not word else None,
                        ignore_case=ignore_case or case_sensitive is False,
                    )
                )
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return matches

            if not result.has_more:
                break
            cursor = result.next_cursor()
            if cursor is None:
                break
    return matches
