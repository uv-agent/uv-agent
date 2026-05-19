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

from .files import resolve_workspace_path


class RipgrepNotFoundError(RuntimeError):
    """Raised when the `rg` binary is not on PATH."""


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
    line: int
    column: int
    text: str
    submatches: list[Submatch] = field(default_factory=list)


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


def search_text(
    pattern: str,
    *,
    root: str | Path = ".",
    globs: Sequence[str] | None = None,
    file_types: Sequence[str] | None = None,
    ignore_case: bool = False,
    fixed_string: bool = False,
    multiline: bool = False,
    word: bool = False,
    max_count_per_file: int | None = None,
    max_total: int | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
    extra_args: Sequence[str] | None = None,
) -> list[Match]:
    """Search file contents with ripgrep and return structured matches.

    `pattern` is a regex unless `fixed_string=True`. Paths are returned relative
    to `root` when ripgrep emits them that way (typical for recursive search).
    Honors `.gitignore` and skips binary files by default; toggle with `hidden`
    and `no_ignore`. `globs` is a list of include/exclude rg glob patterns
    (prefix with `!` to exclude); `file_types` uses rg's `--type` aliases.
    """
    if not pattern:
        raise ValueError("pattern must be non-empty")
    resolved = resolve_workspace_path(root)
    args = _build_args(
        pattern=pattern,
        files_only=False,
        globs=globs,
        file_types=file_types,
        ignore_case=ignore_case,
        fixed_string=fixed_string,
        multiline=multiline,
        word=word,
        max_count=max_count_per_file,
        hidden=hidden,
        no_ignore=no_ignore,
        extra=extra_args,
    )
    code, stdout, stderr = _run(args, resolved)
    # rg exits 1 when no matches; 2+ for real errors.
    if code >= 2:
        raise RuntimeError(f"ripgrep failed (exit {code}): {stderr.strip()}")
    matches: list[Match] = []
    for line in stdout.splitlines():
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
        path = path_obj.get("text") or _decode_bytes_field(path_obj)
        if path is None:
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
                path=path,
                line=line_no,
                column=first_col or 1,
                text=text,
                submatches=submatches,
            )
        )
        if max_total is not None and len(matches) >= max_total:
            break
    return matches


def find_files(
    root: str | Path = ".",
    *,
    globs: Sequence[str] | None = None,
    file_types: Sequence[str] | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    """List workspace files via ripgrep, respecting `.gitignore` by default.

    Returns paths relative to `root` using ripgrep's own enumeration, which is
    typically far faster than `Path.rglob` on large repositories.
    """
    resolved = resolve_workspace_path(root)
    args = _build_args(
        pattern=None,
        files_only=True,
        globs=globs,
        file_types=file_types,
        ignore_case=False,
        fixed_string=False,
        multiline=False,
        word=False,
        max_count=None,
        hidden=hidden,
        no_ignore=no_ignore,
        extra=extra_args,
    )
    code, stdout, stderr = _run(args, resolved)
    if code >= 2:
        raise RuntimeError(f"ripgrep failed (exit {code}): {stderr.strip()}")
    return [line for line in stdout.splitlines() if line]


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
