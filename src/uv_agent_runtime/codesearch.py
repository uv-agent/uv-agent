"""ripgrep-backed code search helpers.

Requires the `rg` binary (https://github.com/BurntSushi/ripgrep) on PATH.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .errors import HelperRuntimeError, HelperValueError
from .files import resolve_workspace_path


class RipgrepNotFoundError(HelperRuntimeError):
    """Raised when the `rg` binary is not on PATH."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            helper="search helpers",
            problem=message
            or "ripgrep (`rg`) not found on PATH; install it and retry the search helper.",
            hints=(
                "Install ripgrep with winget install BurntSushi.ripgrep.MSVC, brew install ripgrep, or apt-get install ripgrep.",
                "After installing, restart the agent or ensure rg is available on PATH for run_python.",
            ),
        )


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
    """Normalize rg type aliases and reject common extension/glob mixups."""

    kinds = _coerce_str_sequence(value, name="file_types")
    if kinds is None:
        return None
    for kind in kinds:
        if not kind:
            raise HelperValueError(
                helper="search helpers",
                problem="file_types entries must be non-empty ripgrep type aliases such as 'py'",
                details={"file_types": kinds},
            )
        if kind.startswith(".") or any(char in kind for char in "*?[]/\\"):
            raise HelperValueError(
                helper="search helpers",
                problem=(
                    "file_types uses ripgrep type aliases such as 'py', not filename extensions "
                    f"or glob patterns: {kind!r}. Use globs=['*.py'] for extension/path patterns."
                ),
                details={"file_types": kinds, "invalid_entry": kind},
                hints=("Use file_types='py' for ripgrep's Python alias, or globs=['*.py'] for filename patterns.",),
            )
    return kinds


def _raise_ripgrep_error(
    code: int,
    stderr: str,
    *,
    helper: str = "search_text",
    pattern: str | None = None,
    fixed_string: bool = False,
) -> None:
    """Raise a ripgrep failure with hints tuned for common script mistakes."""

    detail = stderr.strip()
    hints: list[str] = []
    if pattern is not None and not fixed_string and "regex parse error" in detail:
        hints.append(
            "search_text patterns are regular expressions by default; use literal=True "
            "or fixed_string=True when searching for exact code text."
        )
    if "unrecognized file type" in detail:
        hints.append(
            "file_types uses ripgrep type aliases such as 'py'; use globs=['*.py'] "
            "for filename extensions or path patterns."
        )
    raise HelperRuntimeError(
        helper=helper,
        problem=f"ripgrep failed (exit {code}): {detail}" if detail else f"ripgrep failed (exit {code})",
        details={"pattern": pattern, "fixed_string": fixed_string},
        preview_title="ripgrep stderr",
        preview=detail or None,
        hints=hints,
    )


@dataclass(frozen=True)
class Submatch:
    """Byte-range of a single match inside the surrounding line."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Match:
    """A single ripgrep match line."""

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

        return read_file(self.path, lines=self.line_range(context=context))


def _rg_binary() -> str:
    binary = shutil.which("rg")
    if binary is None:
        raise RipgrepNotFoundError(
            "ripgrep (`rg`) not found on PATH; install via your system package manager "
            "(winget install BurntSushi.ripgrep.MSVC / brew install ripgrep / "
            "apt-get install ripgrep) and retry."
        )
    return binary


def _build_args(
    *,
    pattern: str | None,
    files_only: bool,
    globs: Sequence[str] | None,
    file_types: Sequence[str] | None,
    ignore_case: bool,
    fixed_string: bool,
    multiline: bool,
    word: bool,
    before: int,
    after: int,
    max_count: int | None,
    hidden: bool,
    no_ignore: bool,
    extra: Sequence[str] | None,
) -> list[str]:
    args: list[str] = [_rg_binary()]
    if files_only:
        args.append("--files")
    else:
        args.extend(["--json"])
        if ignore_case:
            args.append("--ignore-case")
        if fixed_string:
            args.append("--fixed-strings")
        if multiline:
            args.extend(["--multiline", "--multiline-dotall"])
        if word:
            args.append("--word-regexp")
        if before:
            args.extend(["--before-context", str(before)])
        if after:
            args.extend(["--after-context", str(after)])
        if max_count is not None:
            args.extend(["--max-count", str(max_count)])
    if hidden:
        args.append("--hidden")
    if no_ignore:
        args.append("--no-ignore")
    for glob in globs or ():
        args.extend(["--glob", glob])
    for kind in file_types or ():
        args.extend(["--type", kind])
    if extra:
        args.extend(extra)
    if not files_only:
        args.extend(["--", pattern or ""])
    return args


def _run(args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run ripgrep to completion and return decoded output.

    The simple capture path is still best when callers do not request a global
    limit: ripgrep can stream internally without Python sitting in the middle.
    Bounded calls use ``_run_limited`` below so ``max_total`` can stop the child
    process before it scans and emits the rest of a very large repository.
    """

    completed = subprocess.run(
        args,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode, completed.stdout, completed.stderr


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Stop a bounded ripgrep process without leaving it behind.

    ``terminate`` is usually enough, but on busy Windows file scans the process
    may need a final ``kill``.  The helper centralizes that defensive cleanup so
    the streaming readers stay small and deterministic.
    """

    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _run_limited(args: list[str], cwd: Path, *, line_limit: int) -> tuple[int, list[str], str, bool]:
    """Run ripgrep and stop after collecting ``line_limit`` stdout lines.

    ``rg`` has ``--max-count``, but that limit is per file, not global.  The
    runtime helpers expose ``max_total`` as a global cap, so we enforce it by
    streaming stdout and terminating ripgrep once enough relevant lines are in
    hand.  A non-zero return code caused by our own termination is reported via
    ``terminated`` instead of treated as a ripgrep failure.
    """

    if line_limit <= 0:
        return 0, [], "", False
    process = subprocess.Popen(
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: list[str] = []
    terminated = False
    try:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip("\r\n"))
            if len(lines) >= line_limit:
                terminated = True
                process.terminate()
                break
        _remaining_stdout, stderr = process.communicate(timeout=2 if terminated else None)
    except subprocess.TimeoutExpired:
        process.kill()
        _remaining_stdout, stderr = process.communicate()
        terminated = True
    return process.returncode or 0, lines, stderr, terminated


def _run_until_matches(args: list[str], cwd: Path, *, match_limit: int) -> tuple[int, list[str], str, bool]:
    """Run ripgrep JSON output until ``match_limit`` match events are collected."""

    if match_limit <= 0:
        return 0, [], "", False
    process = subprocess.Popen(
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: list[str] = []
    matches = 0
    terminated = False
    try:
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip("\r\n")
            lines.append(stripped)
            if _is_match_event_line(stripped):
                matches += 1
                if matches >= match_limit:
                    terminated = True
                    _terminate_process(process)
                    break
        _remaining_stdout, stderr = process.communicate(timeout=2 if terminated else None)
    except subprocess.TimeoutExpired:
        process.kill()
        _remaining_stdout, stderr = process.communicate()
        terminated = True
    return process.returncode or 0, lines, stderr, terminated


def _is_match_event_line(line: str) -> bool:
    """Fast-path detection for rg JSON match records.

    The line is parsed later by ``_parse_search_matches``; here we only need a
    cheap counter to know when to stop the child process.
    """

    return '"type":"match"' in line or '"type": "match"' in line

def search_text(
    pattern: str,
    *,
    root: str | Path = ".",
    roots: str | Path | Sequence[str | Path] | None = None,
    globs: str | Sequence[str] | None = None,
    file_types: str | Sequence[str] | None = None,
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
) -> list[Match]:
    """Search file contents with ripgrep and return structured matches.

    `pattern` is a regex unless `fixed_string=True`. Paths are returned relative
    to `root` when ripgrep emits them that way (typical for recursive search). If
    `roots` is supplied, each result path is relative to the root that produced
    it. Honors `.gitignore` and skips binary files by default; toggle with
    `hidden` and `no_ignore`. `globs` is a list of include/exclude rg glob
    patterns (prefix with `!` to exclude); `file_types` uses rg's `--type`
    aliases.
    """
    if not pattern:
        raise HelperValueError(helper="search_text", problem="pattern must be non-empty")
    before, after = _resolve_context_counts(before=before, after=after, context=context)
    resolved_roots = _resolve_roots(root=root, roots=roots)
    normalized_globs = _coerce_str_sequence(globs, name="globs")
    normalized_file_types = _coerce_file_types(file_types)
    normalized_extra_args = _coerce_str_sequence(extra_args, name="extra_args")
    if max_total is not None and max_total <= 0:
        return []

    matches: list[Match] = []
    remaining = max_total
    for resolved in resolved_roots:
        root_matches = _search_text_one(
            pattern,
            resolved=resolved,
            globs=normalized_globs,
            file_types=normalized_file_types,
            ignore_case=ignore_case,
            case_sensitive=case_sensitive,
            fixed_string=fixed_string,
            literal=literal,
            multiline=multiline,
            word=word,
            before=before,
            after=after,
            max_count_per_file=max_count_per_file,
            max_total=remaining,
            hidden=hidden,
            no_ignore=no_ignore,
            extra_args=normalized_extra_args,
        )
        matches.extend(root_matches)
        if remaining is not None:
            remaining -= len(root_matches)
            if remaining <= 0:
                break
    return matches


def _search_text_one(
    pattern: str,
    *,
    resolved: Path,
    globs: Sequence[str] | None,
    file_types: Sequence[str] | None,
    ignore_case: bool,
    case_sensitive: bool | None,
    fixed_string: bool,
    literal: bool | None,
    multiline: bool,
    word: bool,
    before: int,
    after: int,
    max_count_per_file: int | None,
    max_total: int | None,
    hidden: bool,
    no_ignore: bool,
    extra_args: Sequence[str] | None,
) -> list[Match]:
    cwd_path, target_paths = _split_root(resolved)
    effective_ignore_case = ignore_case if case_sensitive is None else not case_sensitive
    effective_fixed_string = fixed_string if literal is None else literal
    args = _build_args(
        pattern=pattern,
        files_only=False,
        globs=globs,
        file_types=file_types,
        ignore_case=effective_ignore_case,
        fixed_string=effective_fixed_string,
        multiline=multiline,
        word=word,
        before=before,
        after=after,
        max_count=max_count_per_file,
        hidden=hidden,
        no_ignore=no_ignore,
        extra=extra_args,
    )
    args.extend(target_paths)
    if max_total is None:
        code, stdout, stderr = _run(args, cwd_path)
        # rg exits 1 when no matches; 2+ for real errors.
        if code >= 2:
            _raise_ripgrep_error(code, stderr, helper="search_text", pattern=pattern, fixed_string=effective_fixed_string)
        return _parse_search_matches(stdout.splitlines(), cwd=cwd_path, before=before, after=after, max_total=None)

    if before or after:
        code, stdout, stderr = _run(args, cwd_path)
        if code >= 2:
            _raise_ripgrep_error(code, stderr, helper="search_text", pattern=pattern, fixed_string=effective_fixed_string)
        return _parse_search_matches(stdout.splitlines(), cwd=cwd_path, before=before, after=after, max_total=max_total)

    code, lines, stderr, terminated = _run_until_matches(args, cwd_path, match_limit=max_total)
    if code >= 2 and not terminated:
        _raise_ripgrep_error(code, stderr, helper="search_text", pattern=pattern, fixed_string=effective_fixed_string)
    return _parse_search_matches(lines, cwd=cwd_path, before=before, after=after, max_total=max_total)


def _parse_search_matches(
    lines: Sequence[str],
    *,
    cwd: Path,
    before: int,
    after: int,
    max_total: int | None,
) -> list[Match]:
    """Parse ripgrep JSON lines into Match objects without rerunning rg."""

    matches: list[Match] = []
    for line in lines:
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data", {})
        path_obj = data.get("path", {})
        rel_path = path_obj.get("text") or _decode_bytes_field(path_obj)
        if rel_path is None:
            continue
        lines_field = data.get("lines", {})
        text = lines_field.get("text") or _decode_bytes_field(lines_field) or ""
        text = text.rstrip("\r\n")
        line_no = int(data.get("line_number") or 0)
        submatches: list[Submatch] = []
        first_col = None
        for sub in data.get("submatches") or ():
            start = int(sub.get("start", 0))
            end = int(sub.get("end", start))
            match_field = sub.get("match", {})
            sub_text = match_field.get("text") or _decode_bytes_field(match_field) or ""
            submatches.append(Submatch(start=start, end=end, text=sub_text))
            if first_col is None:
                first_col = start + 1
        matches.append(
            Match(
                path=str((cwd / rel_path).resolve()),
                rel_path=rel_path,
                line=line_no,
                column=first_col or 1,
                text=text,
                submatches=submatches,
                context_before=_context_for_match(
                    lines,
                    cwd=cwd,
                    rel_path=rel_path,
                    line_no=line_no,
                    before=True,
                    context_lines=before,
                ),
                context_after=_context_for_match(
                    lines,
                    cwd=cwd,
                    rel_path=rel_path,
                    line_no=line_no,
                    before=False,
                    context_lines=after,
                ),
            )
        )
        if max_total is not None and len(matches) >= max_total:
            break
    return matches


def _context_for_match(
    events: Sequence[str],
    *,
    cwd: Path,
    rel_path: str,
    line_no: int,
    before: bool,
    context_lines: int,
) -> list[tuple[int, str]]:
    if context_lines <= 0:
        return []
    context: list[tuple[int, str]] = []
    abs_path = (cwd / rel_path).resolve()
    for raw in events:
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "context":
            continue
        data = event.get("data", {})
        event_rel = data.get("path", {}).get("text") or _decode_bytes_field(data.get("path", {}))
        if event_rel is None or (cwd / event_rel).resolve() != abs_path:
            continue
        context_line_no = int(data.get("line_number") or 0)
        if before:
            if not line_no - context_lines <= context_line_no < line_no:
                continue
        elif not line_no < context_line_no <= line_no + context_lines:
            continue
        lines_field = data.get("lines", {})
        text = lines_field.get("text") or _decode_bytes_field(lines_field) or ""
        context.append((context_line_no, text.rstrip("\r\n")))
    context.sort(key=lambda item: item[0])
    return context


def find_files(
    root: str | Path = ".",
    *,
    roots: str | Path | Sequence[str | Path] | None = None,
    globs: str | Sequence[str] | None = None,
    file_types: str | Sequence[str] | None = None,
    max_total: int | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
    extra_args: str | Sequence[str] | None = None,
) -> list[str]:
    """List workspace files via ripgrep, respecting `.gitignore` by default.

    Returns absolute paths so the result can be passed directly to file helpers.
    This is typically far faster than `Path.rglob` on large repositories.
    """
    resolved_roots = _resolve_roots(root=root, roots=roots)
    normalized_globs = _coerce_str_sequence(globs, name="globs")
    normalized_file_types = _coerce_file_types(file_types)
    normalized_extra_args = _coerce_str_sequence(extra_args, name="extra_args")
    if max_total is not None and max_total <= 0:
        return []

    files: list[str] = []
    remaining = max_total
    for resolved in resolved_roots:
        root_files = _find_files_one(
            resolved=resolved,
            globs=normalized_globs,
            file_types=normalized_file_types,
            max_total=remaining,
            hidden=hidden,
            no_ignore=no_ignore,
            extra_args=normalized_extra_args,
        )
        files.extend(root_files)
        if remaining is not None:
            remaining -= len(root_files)
            if remaining <= 0:
                break
    return files


def _find_files_one(
    *,
    resolved: Path,
    globs: Sequence[str] | None,
    file_types: Sequence[str] | None,
    max_total: int | None,
    hidden: bool,
    no_ignore: bool,
    extra_args: Sequence[str] | None,
) -> list[str]:
    cwd_path, target_paths = _split_root(resolved)
    args = _build_args(
        pattern=None,
        files_only=True,
        globs=globs,
        file_types=file_types,
        ignore_case=False,
        fixed_string=False,
        multiline=False,
        word=False,
        before=0,
        after=0,
        max_count=None,
        hidden=hidden,
        no_ignore=no_ignore,
        extra=extra_args,
    )
    args.extend(target_paths)
    if max_total is None:
        code, stdout, stderr = _run(args, cwd_path)
        if code >= 2:
            _raise_ripgrep_error(code, stderr, helper="find_files")
        return [_absolute_result_path(cwd_path, line) for line in stdout.splitlines() if line]
    code, lines, stderr, terminated = _run_limited(args, cwd_path, line_limit=max_total)
    if code >= 2 and not terminated:
        _raise_ripgrep_error(code, stderr, helper="find_files")
    return [_absolute_result_path(cwd_path, line) for line in lines if line][:max_total]


def _resolve_roots(
    *,
    root: str | Path,
    roots: str | Path | Sequence[str | Path] | None,
) -> list[Path]:
    """Resolve either the legacy single root or the newer multi-root list."""

    if roots is None:
        return [resolve_workspace_path(root)]
    if Path(root) != Path("."):
        raise HelperValueError(
            helper="search helpers",
            problem="root and roots are mutually exclusive",
            hints=("Use root=... for one search root, or leave root='.' and pass roots=[...].",),
        )
    return [resolve_workspace_path(item) for item in _coerce_path_sequence(roots, name="roots") or []]


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


def _absolute_result_path(cwd: Path, rel_path: str) -> str:
    return str((cwd / rel_path).resolve())


def _split_root(resolved: Path) -> tuple[Path, list[str]]:
    """Return ``(cwd, positional_paths)`` for ripgrep based on the resolved root.

    Ripgrep requires ``cwd`` to be a directory. When the agent passes a file as
    the search root, scope the search to that single file by using its parent
    as ``cwd`` and appending the file name as a positional path argument.
    """
    if resolved.is_file():
        return resolved.parent, [resolved.name]
    return resolved, []


def _decode_bytes_field(field_obj: dict) -> str | None:
    """rg --json sometimes emits {'bytes': '<base64>'} for non-UTF-8 input."""
    encoded = field_obj.get("bytes")
    if not encoded:
        return None
    import base64

    try:
        return base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception:
        return None
