from __future__ import annotations

import builtins
import os
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from . import codequery, codesearch, dependencies, events as _events, files as _files, mcp as _mcp, patch as _patch
from .errors import HelperValueError
from . import textops, threads as _threads, vision as _vision
from .cwd import enter_dir as _enter_dir
from .helper_tracking import tracked_helper


def _track(name: str):
    def decorator(func):
        return tracked_helper(func, name=name)

    return decorator

T = TypeVar("T")


class SelectionError(ValueError):
    """Raised by ``one()`` when a result collection is empty or ambiguous."""


class CollectionResult(Generic[T]):
    """Small list-like wrapper shared by search, symbol, capture, and file results.

    The wrapper deliberately behaves like a normal Python collection so scripts can
    use ``if results:``, ``len(results)``, slicing, and iteration without learning a
    custom protocol.  The named helpers are for the common agent pattern of
    asserting that a discovery step found either zero/one/many items.
    """

    label = "items"

    def __init__(self, items: Iterable[T] = ()) -> None:
        self._items = builtins.list(items)

    @property
    def ok(self) -> bool:
        return bool(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __getitem__(self, index: int | slice) -> T | builtins.list[T]:
        return self._items[index]

    def first(self) -> T | None:
        return self._items[0] if self._items else None

    def one(self) -> T:
        if len(self._items) != 1:
            raise SelectionError(f"expected exactly one {self.label}, found {len(self._items)}")
        return self._items[0]

    def all(self) -> builtins.list[T]:
        return builtins.list(self._items)

    def summary(self) -> str:
        return f"{len(self._items)} {self.label}"

    def print(self) -> None:
        print(self.summary())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._items!r})"


class SearchResults(CollectionResult[codesearch.Match]):
    label = "matches"

    def summary(self) -> str:
        file_count = len({item.path for item in self._items})
        return f"{len(self._items)} matches across {file_count} files"

    def grouped(self) -> dict[str, builtins.list[codesearch.Match]]:
        grouped: "OrderedDict[str, builtins.list[codesearch.Match]]" = OrderedDict()
        for item in self._items:
            grouped.setdefault(item.path, []).append(item)
        return dict(grouped)

    def views(self, *, context: int = 8, limit: int | None = None) -> builtins.list[textops.FileView]:
        selected = self._items if limit is None else self._items[:limit]
        return [item.view(context=context) for item in selected]


class SymbolResults(CollectionResult[codequery.Symbol]):
    label = "symbols"

    def summary(self) -> str:
        file_count = len({item.path for item in self._items})
        return f"{len(self._items)} symbols across {file_count} files"

    def views(self, *, context: int = 12, limit: int | None = None) -> builtins.list[textops.FileView]:
        selected = self._items if limit is None else self._items[:limit]
        return [item.view(context=context) for item in selected]


class CaptureResults(CollectionResult[codequery.Capture]):
    label = "captures"

    def summary(self) -> str:
        file_count = len({item.path for item in self._items})
        return f"{len(self._items)} captures across {file_count} files"

    def views(self, *, context: int = 8, limit: int | None = None) -> builtins.list[textops.FileView]:
        selected = self._items if limit is None else self._items[:limit]
        return [item.view(context=context) for item in selected]


class FileSet(CollectionResult[str]):
    label = "files"

    def files(self) -> builtins.list["File"]:
        return [File(path) for path in self._items]

    def views(
        self,
        *,
        head: int | None = None,
        tail: int | None = None,
        lines: tuple[int, int] | None = None,
        around: str | None = None,
        context: int = 20,
        limit: int | None = None,
    ) -> builtins.list[textops.FileView]:
        selected = self._items if limit is None else self._items[:limit]
        return [
            File(path).read(head=head, tail=tail, lines=lines, around=around, context=context)
            for path in selected
        ]

    def search(
        self,
        query: str,
        *,
        globs: str | Sequence[str] | None = None,
        types: str | Sequence[str] | None = None,
        mode: codesearch.SearchMode | None = None,
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
        limit: int | None = None,
        hidden: bool = False,
        no_ignore: bool = False,
        extra_args: str | Sequence[str] | None = None,
        refresh: bool = False,
    ) -> SearchResults:
        return search(
            query,
            roots=self._items,
            globs=globs,
            types=types,
            mode=mode,
            ignore_case=ignore_case,
            case_sensitive=case_sensitive,
            fixed_string=fixed_string,
            literal=literal,
            multiline=multiline,
            word=word,
            before=before,
            after=after,
            context=context,
            max_count_per_file=max_count_per_file,
            limit=limit,
            hidden=hidden,
            no_ignore=no_ignore,
            extra_args=extra_args,
            refresh=refresh,
        )

    def symbols(
        self,
        *,
        language: str | None = None,
        languages: str | Sequence[str] | None = None,
        kind: str | None = None,
        kinds: str | Sequence[str] | None = None,
        name: str | None = None,
        name_pattern: str | None = None,
        contains: str | None = None,
        limit: int | None = None,
        hidden: bool = False,
        no_ignore: bool = False,
        globs: str | Sequence[str] | None = None,
    ) -> SymbolResults:
        # tree-sitter symbol search takes one root, not an arbitrary explicit file
        # set.  Reusing the common ancestors would be surprising, so ask callers to
        # pass globs/root through rt.symbols for symbol queries.
        roots = _common_roots(self._items)
        out: builtins.list[codequery.Symbol] = []
        remaining = limit
        for root in roots:
            batch = symbols(
                root=root,
                language=language,
                languages=languages,
                kind=kind,
                kinds=kinds,
                name=name,
                name_pattern=name_pattern,
                contains=contains,
                limit=remaining,
                hidden=hidden,
                no_ignore=no_ignore,
                globs=globs,
            ).all()
            out.extend(batch)
            if limit is not None:
                remaining = limit - len(out)
                if remaining <= 0:
                    break
        return SymbolResults(out)


@dataclass(frozen=True)
class File:
    """Workspace-relative or absolute file handle for read/edit operations."""

    path: str | Path

    @_track("file.read")
    def read(
        self,
        *,
        lines: tuple[int, int] | None = None,
        head: int | None = None,
        tail: int | None = None,
        around: str | None = None,
        context: int = 20,
        encoding: str = "utf-8",
    ) -> textops.FileView:
        return textops.read_file(
            self.path,
            lines=lines,
            head=head,
            tail=tail,
            around=around,
            context=context,
            encoding=encoding,
        )

    @_track("file.text")
    def text(self, *, encoding: str = "utf-8") -> str:
        return _files.read_text(self.path, encoding=encoding)

    @_track("file.json")
    def json(self, *, encoding: str = "utf-8") -> Any:
        return _files.read_json(self.path, encoding=encoding)

    @_track("file.write")
    def write(
        self,
        text: str,
        *,
        like: textops.FileView | textops.TextFile | str | Path | None = None,
        encoding: str | None = None,
        newline: Literal["lf", "crlf", "cr", "none"] | None = None,
        final_newline: bool | None = None,
        bom: bool | None = None,
    ) -> Path:
        return textops.write_file(
            self.path,
            text,
            like=like,
            encoding=encoding,
            newline=newline,
            final_newline=final_newline,
            bom=bom,
        )

    write_text = write

    @_track("file.write_json")
    def write_json(self, value: object, *, encoding: str = "utf-8", indent: int = 2) -> Path:
        return _files.write_json(self.path, value, encoding=encoding, indent=indent)

    @_track("file.replace")
    def replace(
        self,
        old: str,
        new: str,
        *,
        count: int = 1,
        newlines: Literal["logical", "raw"] = "logical",
    ) -> textops.ReplacementResult:
        return textops.replace_text(self.path, old, new, count=count, newlines=newlines)

    @_track("file.edit")
    def edit(
        self,
        start: int,
        end: int,
        new_text: str,
        *,
        expect_first: str | None = None,
        expect_last: str | None = None,
        expect_mode: Literal["startswith", "contains", "exact", "regex"] = "startswith",
        strip_indent: bool = True,
        encoding: str | None = None,
        newline: Literal["preserve", "lf", "crlf", "cr"] = "preserve",
        final_newline: bool | None = None,
        bom: bool | None = None,
    ) -> textops.EditResult:
        return textops.edit_lines(
            self.path,
            start,
            end,
            new_text,
            expect_first=expect_first,
            expect_last=expect_last,
            expect_mode=expect_mode,
            strip_indent=strip_indent,
            encoding=encoding,
            newline=newline,
            final_newline=final_newline,
            bom=bom,
        )

    @_track("file.insert_after")
    def insert_after(
        self,
        line: int,
        text: str,
        *,
        expect_line: str | None = None,
        expect_mode: Literal["startswith", "contains", "exact", "regex"] = "startswith",
        encoding: str | None = None,
    ) -> textops.EditResult:
        if expect_line is not None:
            _check_expected_insert_line(self.path, line, expect_line, expect_mode)
        return textops.edit_lines(
            self.path,
            line + 1,
            line,
            _ensure_insert_text(text),
            encoding=encoding,
        )

    @_track("file.insert_before")
    def insert_before(
        self,
        line: int,
        text: str,
        *,
        expect_line: str | None = None,
        expect_mode: Literal["startswith", "contains", "exact", "regex"] = "startswith",
        encoding: str | None = None,
    ) -> textops.EditResult:
        if expect_line is not None:
            _check_expected_insert_line(self.path, line, expect_line, expect_mode)
        return textops.edit_lines(
            self.path,
            line,
            line - 1,
            _ensure_insert_text(text),
            encoding=encoding,
        )

    @_track("file.delete_lines")
    def delete_lines(
        self,
        start: int,
        end: int,
        *,
        expect_first: str | None = None,
        expect_last: str | None = None,
        expect_mode: Literal["startswith", "contains", "exact", "regex"] = "startswith",
        encoding: str | None = None,
    ) -> textops.EditResult:
        return textops.edit_lines(
            self.path,
            start,
            end,
            "",
            expect_first=expect_first,
            expect_last=expect_last,
            expect_mode=expect_mode,
            encoding=encoding,
        )

    @_track("file.info")
    def info(self, *, base: str | Path | None = None) -> textops.PathInfo:
        return textops.path_info(self.path, base=base)

    @_track("file.compare")
    def compare(
        self,
        other: str | Path,
        *,
        ignore_eol: bool = False,
        ignore_final_newline: bool = False,
    ) -> textops.TextComparison:
        return textops.compare_text(
            self.text(),
            File(other).text() if Path(other).exists() else str(other),
            ignore_eol=ignore_eol,
            ignore_final_newline=ignore_final_newline,
        )

    @_track("file.diff")
    def diff(self, other: str | Path, *, context: int = 3) -> str:
        other_text = File(other).text() if Path(other).exists() else str(other)
        return textops.make_unified_diff(self.text(), other_text, path=str(self.path), context=context)




class _DepsNamespace:
    @_track("deps.add")
    def add(
        self,
        *packages: str,
        editable: bool = False,
        optional: str | None = None,
        dev: bool = False,
        group: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> textops.CommandTextResult:
        return dependencies.add_dependency(
            *packages,
            editable=editable,
            optional=optional,
            dev=dev,
            group=group,
            timeout_s=timeout,
            check=check,
        )

    @_track("deps.env_dir")
    def env_dir(self) -> Path:
        return dependencies.run_python_env_dir()


class _ThreadsNamespace:
    @_track("threads.list")
    def list(
        self,
        *,
        state_dir: str | Path | None = None,
        limit: int = 10,
        kind: str = "thread",
        parent_thread_id: str | None = None,
        since_last_compaction: bool = True,
        include_tools: bool = False,
    ) -> builtins.list[_threads.ThreadDigest]:
        return _threads.list_thread_digests(
            state_dir=state_dir,
            limit=limit,
            kind=kind,
            parent_thread_id=parent_thread_id,
            since_last_compaction=since_last_compaction,
            include_tools=include_tools,
        )

    @_track("threads.digest")
    def digest(
        self,
        thread_id: str,
        *,
        state_dir: str | Path | None = None,
        kind: str | None = None,
        since_last_compaction: bool = True,
        include_tools: bool = False,
    ) -> _threads.ThreadDigest:
        return _threads.thread_digest(
            thread_id,
            state_dir=state_dir,
            kind=kind,
            since_last_compaction=since_last_compaction,
            include_tools=include_tools,
        )

    @_track("threads.view")
    def view(
        self,
        thread_id: str,
        *,
        state_dir: str | Path | None = None,
        kind: str | None = None,
        epoch: _threads.EpochSelector = "latest",
        max_turns: int | None = None,
        max_text_chars: int = 12_000,
        max_item_chars: int = 4_000,
        max_process_refs: int = 500,
    ) -> _threads.ThreadView:
        return _threads.thread_view(
            thread_id,
            state_dir=state_dir,
            kind=kind,
            epoch=epoch,
            max_turns=max_turns,
            max_text_chars=max_text_chars,
            max_item_chars=max_item_chars,
            max_process_refs=max_process_refs,
        )

    @_track("threads.detail")
    def detail(
        self,
        *,
        state_dir: str | Path | None = None,
        thread_id: str | None = None,
        ids: str | Sequence[str] | None = None,
        turn_ids: str | Sequence[str] | None = None,
        max_code_chars: int = 4_000,
        max_output_chars: int = 4_000,
        max_events: int = 100,
        include_raw_events: bool = False,
    ) -> _threads.ThreadDetailResult:
        return _threads.thread_detail(
            state_dir=state_dir,
            thread_id=thread_id,
            ids=ids,
            turn_ids=turn_ids,
            max_code_chars=max_code_chars,
            max_output_chars=max_output_chars,
            max_events=max_events,
            include_raw_events=include_raw_events,
        )


class _McpNamespace:
    @_track("mcp.list")
    def list(
        self,
        *,
        config_paths: builtins.list[str | Path] | None = None,
        cwd: str | Path | None = None,
    ) -> builtins.list[dict[str, Any]]:
        return _mcp.list_declared_servers(config_paths=config_paths, cwd=cwd)

    @_track("mcp.connect")
    def connect(
        self,
        name: str,
        *,
        config_paths: builtins.list[str | Path] | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = 30,
    ) -> _mcp.McpClient:
        return _mcp.connect_named(name, config_paths=config_paths, cwd=cwd, timeout_s=timeout)

    @_track("mcp.connect_url")
    def connect_url(
        self,
        url: str,
        *,
        transport: _mcp.McpTransport = "streamable_http",
        timeout: float | None = 30,
    ) -> _mcp.McpClient:
        return _mcp.connect_url(url, transport=transport, timeout_s=timeout)

    @_track("mcp.connect_stdio")
    def connect_stdio(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = 30,
    ) -> _mcp.McpClient:
        return _mcp.connect_stdio(builtins.list(command), cwd=cwd, env=env, timeout_s=timeout)

    @_track("mcp.connect_declared")
    def connect_declared(
        self,
        name: str,
        *,
        config_path: str | Path = ".agents/mcp.json",
        cwd: str | None = None,
        timeout: float | None = 30,
    ) -> _mcp.McpClient:
        return _mcp.connect_declared(name, config_path=config_path, cwd=cwd, timeout_s=timeout)


class _EventsNamespace:
    @_track("events.emit")
    def emit(self, kind: str, **payload: Any) -> dict[str, Any]:
        return _events.emit_event(kind, **payload)

    @_track("events.progress")
    def progress(self, message: str, **payload: Any) -> dict[str, Any]:
        return _events.emit_progress(message, **payload)

    @_track("events.result")
    def result(self, **payload: Any) -> dict[str, Any]:
        return _events.emit_result(**payload)

    @_track("events.look_at")
    def look_at(self, path: str | Path, *, note: str = "") -> dict[str, Any]:
        return _vision.look_at(path, note=note)


@_track("search")
def search(
    query: str,
    *,
    root: str | Path = ".",
    roots: str | Path | Sequence[str | Path] | None = None,
    globs: str | Sequence[str] | None = None,
    types: str | Sequence[str] | None = None,
    mode: codesearch.SearchMode | None = None,
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
    limit: int | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
    extra_args: str | Sequence[str] | None = None,
    refresh: bool = False,
) -> SearchResults:
    return SearchResults(
        codesearch.search_text(
            query,
            root=root,
            roots=roots,
            globs=globs,
            types=types,
            mode=mode,
            ignore_case=ignore_case,
            case_sensitive=case_sensitive,
            fixed_string=fixed_string,
            literal=literal,
            multiline=multiline,
            word=word,
            before=before,
            after=after,
            context=context,
            max_count_per_file=max_count_per_file,
            max_total=limit,
            hidden=hidden,
            no_ignore=no_ignore,
            extra_args=extra_args,
            refresh=refresh,
        )
    )


@_track("files")
def files(
    root: str | Path = ".",
    *,
    roots: str | Path | Sequence[str | Path] | None = None,
    query: str = "",
    globs: str | Sequence[str] | None = None,
    types: str | Sequence[str] | None = None,
    limit: int | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
    extra_args: str | Sequence[str] | None = None,
    refresh: bool = False,
) -> FileSet:
    return FileSet(
        codesearch.find_files(
            root,
            roots=roots,
            query=query,
            globs=globs,
            types=types,
            max_total=limit,
            hidden=hidden,
            no_ignore=no_ignore,
            extra_args=extra_args,
            refresh=refresh,
        )
    )


@_track("symbols")
def symbols(
    root: str | Path = ".",
    *,
    languages: str | Sequence[str] | None = None,
    language: str | None = None,
    kinds: str | Sequence[str] | None = None,
    kind: str | None = None,
    name_pattern: str | None = None,
    name: str | None = None,
    contains: str | None = None,
    limit: int | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
    globs: str | Sequence[str] | None = None,
) -> SymbolResults:
    return SymbolResults(
        codequery.find_symbols(
            root=root,
            languages=languages,
            language=language,
            kinds=kinds,
            kind=kind,
            name_pattern=name_pattern,
            name=name,
            contains=contains,
            max_count=limit,
            hidden=hidden,
            no_ignore=no_ignore,
            globs=globs,
        )
    )


@_track("query")
def query(
    query_text: str,
    *,
    language: str,
    root: str | Path = ".",
    globs: str | Sequence[str] | None = None,
    types: str | Sequence[str] | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
    limit: int | None = None,
) -> CaptureResults:
    return CaptureResults(
        codequery.query_code(
            query_text,
            language=language,
            root=root,
            globs=globs,
            file_types=types,
            hidden=hidden,
            no_ignore=no_ignore,
            max_count=limit,
        )
    )


@_track("run")
def run(
    *args: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    cwd: str | Path | None = None,
    encoding: str = "utf-8",
    errors: str = "replace",
    env: Mapping[str, str] | None = None,
    env_patch: Mapping[str, str | None] | None = None,
    timeout: float | None = None,
    check: bool = False,
) -> textops.CommandTextResult:
    if len(args) == 1 and isinstance(args[0], (builtins.list, tuple)):
        args = args[0]
    return textops.run_process_text(
        [str(arg) for arg in args],
        cwd=cwd,
        encoding=encoding,
        errors=errors,
        env=env,
        env_patch=env_patch,
        timeout_s=timeout,
        check=check,
    )


def file(path: str | Path) -> File:
    return File(path)


@_track("cd")
def cd(path: str | Path) -> Path:
    return _enter_dir(path)


@_track("pwd")
def pwd() -> Path:
    return Path.cwd()


@_track("look_at")
def look_at(path: str | Path, *, note: str = "") -> dict[str, Any]:
    return _vision.look_at(path, note=note)


@_track("path")
def path(path: str | Path, *, base: str | Path | None = None) -> textops.PathInfo:
    return textops.path_info(path, base=base)


@_track("patch")
def patch(
    patch_text: str,
    *,
    cwd: str | Path | None = None,
    format: Literal["auto", "apply_patch", "unified"] = "auto",
    dry_run: bool = False,
    check: bool = True,
) -> _patch.PatchResult:
    return textops.apply_patch_any(patch_text, cwd=cwd, format=format, dry_run=dry_run, check=check)


@_track("apply_patch")
def apply_patch(patch_text: str, *, cwd: str | Path | None = None, check: bool = True) -> _patch.PatchResult:
    return _patch.apply_patch(patch_text, cwd=cwd, check=check)


@_track("dry_run_patch")
def dry_run_patch(patch_text: str, *, cwd: str | Path | None = None, check: bool = True) -> _patch.PatchResult:
    return _patch.dry_run_patch(patch_text, cwd=cwd, check=check)


@_track("convert_patch")
def convert_patch(
    patch_text: str,
    *,
    from_format: Literal["apply_patch", "unified"],
    to_format: Literal["apply_patch", "unified"],
) -> str:
    return textops.convert_patch(patch_text, from_format=from_format, to_format=to_format)


@_track("diff")
def diff(before: str, after: str, *, path: str | None = None, context: int = 3) -> str:
    return textops.make_unified_diff(before, after, path=path, context=context)


@_track("compare")
def compare(
    left: str,
    right: str,
    *,
    ignore_eol: bool = False,
    ignore_final_newline: bool = False,
) -> textops.TextComparison:
    return textops.compare_text(left, right, ignore_eol=ignore_eol, ignore_final_newline=ignore_final_newline)


@_track("normalize")
def normalize(
    text: str,
    *,
    eol: Literal["lf", "crlf", "cr"] | None = "lf",
    final_newline: bool | None = None,
) -> str:
    return textops.normalize_text(text, eol=eol, final_newline=final_newline)


@_track("snapshot")
def snapshot(paths: Sequence[str | Path] | None = None, *, root: str | Path = ".") -> textops.Snapshot:
    return textops.snapshot_files(paths, root=root)


@_track("restore")
def restore(snapshot: textops.Snapshot) -> builtins.list[str]:
    return textops.restore_snapshot(snapshot)


def transaction(paths: Sequence[str | Path] | None = None, *, root: str | Path = ".") -> Iterator[textops.Snapshot]:
    return textops.workspace_transaction(paths, root=root)


def _check_expected_insert_line(
    path: str | Path,
    line: int,
    expected: str,
    mode: Literal["startswith", "contains", "exact", "regex"],
) -> None:
    view = textops.read_file(path, lines=(line, line))
    actual = view.text.rstrip("\r\n").lstrip()
    target = expected.lstrip()
    if mode == "startswith":
        matched = actual.startswith(target)
    elif mode == "contains":
        matched = target in actual
    elif mode == "exact":
        matched = actual == target
    elif mode == "regex":
        import re

        matched = re.search(expected, actual) is not None
    else:
        raise HelperValueError(helper="file.insert", problem="expect_mode must be 'startswith', 'contains', 'exact', or 'regex'")
    if not matched:
        raise HelperValueError(
            helper="file.insert",
            problem=f"expect_line did not match line {line}",
            details={"path": str(path), "line": line, "expected": expected, "actual": actual, "mode": mode},
        )


def _ensure_insert_text(value: str) -> str:
    if value and not value.endswith(("\n", "\r")):
        return value + "\n"
    return value


def _common_roots(paths: Sequence[str]) -> builtins.list[Path]:
    if not paths:
        return [Path.cwd()]
    resolved = [Path(path).resolve() for path in paths]
    parents = [path if path.is_dir() else path.parent for path in resolved]
    try:
        return [Path(os.path.commonpath([str(parent) for parent in parents]))]
    except ValueError:
        return sorted(set(parents))


deps = _DepsNamespace()
threads = _ThreadsNamespace()
mcp = _McpNamespace()
events = _EventsNamespace()
