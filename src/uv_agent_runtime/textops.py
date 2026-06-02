from __future__ import annotations

import difflib
import os
import re
import signal
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from unidiff import PatchSet
from unidiff.patch import PatchedFile

from .files import resolve_workspace_path
from .patch import PatchResult, apply_patch, dry_run_patch


@dataclass(frozen=True)
class PathInfo:
    path: str
    exists: bool
    kind: Literal["file", "dir", "missing", "other"]
    size: int | None
    cwd: str
    base: str | None
    is_absolute: bool
    is_relative_to_base: bool | None


@dataclass(frozen=True)
class TextFile:
    path: str
    text: str
    encoding: str
    newline: Literal["lf", "crlf", "cr", "mixed", "none"]
    final_newline: bool
    bom: bool


@dataclass(frozen=True)
class FileView:
    """A text-file view plus metadata useful for follow-up edits."""

    path: str
    exists: bool
    text: str
    line_count: int
    start_line: int
    end_line: int
    truncated: bool
    encoding: str
    newline: Literal["lf", "crlf", "cr", "mixed", "none"]
    final_newline: bool
    bom: bool
    size: int | None
    kind: Literal["file", "dir", "missing", "other"]

    def numbered(self) -> str:
        """Return the selected text with 1-indexed line-number prefixes."""

        if not self.text:
            return ""
        width = max(len(str(self.end_line)), len(str(self.start_line)), 1)
        return "\n".join(
            f"{line_no:>{width}}: {line}"
            for line_no, line in enumerate(self.text.splitlines(), start=self.start_line)
        )


@dataclass(frozen=True)
class TextComparison:
    equal: bool
    kind: Literal["equal", "content", "eol", "final_newline"]
    message: str
    first_difference_line: int | None = None
    left: str | None = None
    right: str | None = None


@dataclass(frozen=True)
class ReplacementResult:
    path: str
    replacements: int
    before: TextFile
    after: TextFile


@dataclass(frozen=True)
class EditResult:
    path: str
    changed: bool
    replaced_text: str
    line_count_before: int
    line_count_after: int
    line_delta: int


@dataclass(frozen=True)
class Snapshot:
    root: str
    files: dict[str, bytes | None]


@dataclass(frozen=True)
class CommandTextResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_for_error(self) -> "CommandTextResult":
        """Raise RuntimeError if the command exited non-zero."""
        if self.timed_out:
            detail = self.stderr or self.stdout
            raise RuntimeError(
                f"command timed out: {self.args!r}"
                + (f"\n{detail}" if detail else "")
            )
        if self.returncode != 0:
            detail = self.stderr or self.stdout
            raise RuntimeError(
                f"command failed with exit {self.returncode}: {self.args!r}"
                + (f"\n{detail}" if detail else "")
            )
        return self


def path_info(path: str | Path, *, base: str | Path | None = None) -> PathInfo:
    """Return resolved path metadata without mutating the filesystem."""

    resolved = resolve_workspace_path(path)
    kind: Literal["file", "dir", "missing", "other"]
    if resolved.is_file():
        kind = "file"
    elif resolved.is_dir():
        kind = "dir"
    elif resolved.exists():
        kind = "other"
    else:
        kind = "missing"
    size = resolved.stat().st_size if kind == "file" else None
    resolved_base = resolve_workspace_path(base) if base is not None else None
    is_relative = None
    if resolved_base is not None:
        try:
            resolved.relative_to(resolved_base)
            is_relative = True
        except ValueError:
            is_relative = False
    return PathInfo(
        path=str(resolved),
        exists=resolved.exists(),
        kind=kind,
        size=size,
        cwd=str(Path.cwd().resolve()),
        base=str(resolved_base) if resolved_base is not None else None,
        is_absolute=Path(path).is_absolute(),
        is_relative_to_base=is_relative,
    )


def read_text_lossless(path: str | Path, *, encoding: str = "utf-8") -> TextFile:
    """Read text while preserving newline, BOM, and final-newline metadata."""

    resolved = resolve_workspace_path(path)
    raw = resolved.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig" if encoding.lower().replace("_", "-") == "utf-8" else encoding)
    return TextFile(
        path=str(resolved),
        text=text,
        encoding=encoding,
        newline=_detect_newline_style(text),
        final_newline=text.endswith(("\n", "\r")),
        bom=bom,
    )


def read_file(
    path: str | Path,
    *,
    lines: tuple[int, int] | None = None,
    head: int | None = None,
    tail: int | None = None,
    around: str | None = None,
    context: int = 20,
    encoding: str = "utf-8",
) -> FileView:
    """Read a file or selected line range and return metadata in one object."""

    selectors = [lines is not None, head is not None, tail is not None, around is not None]
    if sum(selectors) > 1:
        raise ValueError("lines, head, tail, and around are mutually exclusive")
    if context < 0:
        raise ValueError("context must be >= 0")
    resolved = resolve_workspace_path(path)
    kind = _path_kind(resolved)
    size = resolved.stat().st_size if kind == "file" else None
    if kind != "file":
        return FileView(
            path=str(resolved),
            exists=resolved.exists(),
            text="",
            line_count=0,
            start_line=0,
            end_line=0,
            truncated=False,
            encoding=encoding,
            newline="none",
            final_newline=False,
            bom=False,
            size=size,
            kind=kind,
        )

    loaded = read_text_lossless(resolved, encoding=encoding)
    logical_lines = _logical_lines(loaded.text)
    line_count = len(logical_lines)
    start_line, end_line = _selected_line_range(
        logical_lines,
        lines=lines,
        head=head,
        tail=tail,
        around=around,
        context=context,
    )
    selected = _slice_text_by_lines(loaded.text, start_line, end_line)
    return FileView(
        path=loaded.path,
        exists=True,
        text=selected,
        line_count=line_count,
        start_line=start_line,
        end_line=end_line,
        truncated=start_line > 1 or end_line < line_count,
        encoding=loaded.encoding,
        newline=loaded.newline,
        final_newline=loaded.final_newline,
        bom=loaded.bom,
        size=size,
        kind="file",
    )


def write_text_lossless(
    path: str | Path,
    text: str,
    *,
    like: TextFile | str | Path | None = None,
    encoding: str | None = None,
    newline: Literal["lf", "crlf", "cr", "none"] | None = None,
    final_newline: bool | None = None,
    bom: bool | None = None,
    atomic: bool = True,
) -> Path:
    """Write text with explicit or source-derived encoding/newline metadata."""

    resolved = resolve_workspace_path(path)
    template = _coerce_text_file(like) if like is not None else None
    chosen_encoding = encoding or (template.encoding if template else "utf-8")
    chosen_newline = newline or (_single_newline_style(template.newline) if template else None)
    chosen_final_newline = final_newline if final_newline is not None else template.final_newline if template else None
    chosen_bom = template.bom if bom is None and template else bool(bom)
    if bom is not None:
        chosen_bom = bom

    normalized = normalize_text(text, eol=None if chosen_newline == "none" else chosen_newline, final_newline=chosen_final_newline)
    data = normalized.encode(chosen_encoding)
    if chosen_bom and chosen_encoding.lower().replace("_", "-") == "utf-8" and not data.startswith(b"\xef\xbb\xbf"):
        data = b"\xef\xbb\xbf" + data
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        _atomic_write_bytes(resolved, data)
    else:
        resolved.write_bytes(data)
    return resolved


def write_file(
    path: str | Path,
    text: str,
    *,
    like: FileView | TextFile | str | Path | None = None,
    encoding: str | None = None,
    newline: Literal["lf", "crlf", "cr", "none"] | None = None,
    final_newline: bool | None = None,
    bom: bool | None = None,
) -> Path:
    """Write text with source-derived metadata, keeping atomic writes internal."""

    return write_text_lossless(
        path,
        text,
        like=_coerce_write_like(like) if like is not None else None,
        encoding=encoding,
        newline=newline,
        final_newline=final_newline,
        bom=bom,
    )


def compare_text(
    left: str | TextFile,
    right: str | TextFile,
    *,
    ignore_eol: bool = False,
    ignore_final_newline: bool = False,
) -> TextComparison:
    """Compare two text values and classify common newline-only differences."""

    left_text = left.text if isinstance(left, TextFile) else left
    right_text = right.text if isinstance(right, TextFile) else right
    if left_text == right_text:
        return TextComparison(True, "equal", "texts are identical")
    normalized_left = left_text
    normalized_right = right_text
    if ignore_eol:
        normalized_left = normalize_text(normalized_left, eol="lf")
        normalized_right = normalize_text(normalized_right, eol="lf")
    if ignore_final_newline:
        normalized_left = normalized_left.rstrip("\r\n")
        normalized_right = normalized_right.rstrip("\r\n")
    if normalized_left == normalized_right:
        if _strip_final_newline(left_text) == _strip_final_newline(right_text):
            return TextComparison(False, "final_newline", "texts differ only by final newline")
        return TextComparison(False, "eol", "texts differ only by newline representation")
    line, left_line, right_line = _first_line_difference(normalized_left, normalized_right)
    return TextComparison(False, "content", "texts differ by content", line, left_line, right_line)


def normalize_text(
    text: str,
    *,
    eol: Literal["lf", "crlf", "cr"] | None = None,
    final_newline: bool | None = None,
) -> str:
    """Normalize EOL style and final newline policy for generated text."""

    if eol is None:
        normalized = text
    elif eol == "lf":
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    elif eol == "crlf":
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    elif eol == "cr":
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r")
    else:
        raise ValueError(f"unsupported eol style: {eol!r}")
    if final_newline is True and not normalized.endswith(("\n", "\r")):
        normalized += _final_newline_for_text(normalized, eol)
    elif final_newline is False:
        normalized = normalized.rstrip("\r\n")
    return normalized


def replace_text(
    path: str | Path,
    old: str,
    new: str,
    *,
    count: int = 1,
    newlines: Literal["logical", "raw"] = "logical",
) -> ReplacementResult:
    """Replace text in a file while preserving its original text metadata."""

    if not old:
        raise ValueError("old text must not be empty")
    if count < 1:
        raise ValueError("count must be >= 1")
    if newlines not in {"logical", "raw"}:
        raise ValueError("newlines must be 'logical' or 'raw'")
    before = read_text_lossless(path)
    best_found = 0
    for candidate_old, candidate_new in _replacement_candidates(before, old, new, newlines):
        found = before.text.count(candidate_old)
        best_found = max(best_found, found)
        if found < count:
            continue
        after_text = before.text.replace(candidate_old, candidate_new, count)
        written = write_text_lossless(path, after_text, like=before)
        # Avoid a second full-file read after writing.  ``write_text_lossless``
        # derives bytes from this exact normalized text and metadata, so the
        # resulting TextFile can be reconstructed without touching disk again.
        after = TextFile(
            path=str(written),
            text=after_text,
            encoding=before.encoding,
            newline=_detect_newline_style(after_text),
            final_newline=after_text.endswith(("\n", "\r")),
            bom=before.bom,
        )
        return ReplacementResult(path=after.path, replacements=count, before=before, after=after)
    context = _replacement_missing_context(before, old, found=best_found, count=count, newlines=newlines)
    raise ValueError(f"expected at least {count} occurrence(s), found {best_found}.{context}")


def edit_lines(
    path: str | Path,
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
) -> EditResult:
    """Replace, delete, or insert 1-indexed lines with optional anchor checks."""

    if expect_mode not in {"startswith", "contains", "exact", "regex"}:
        raise ValueError("expect_mode must be 'startswith', 'contains', 'exact', or 'regex'")
    if newline not in {"preserve", "lf", "crlf", "cr"}:
        raise ValueError("newline must be 'preserve', 'lf', 'crlf', or 'cr'")
    if start < 1:
        raise ValueError("start must be >= 1")
    if end < 0:
        raise ValueError("end must be >= 0")

    before = read_text_lossless(path, encoding=encoding or "utf-8")
    logical_lines = _logical_lines(before.text)
    line_count_before = len(logical_lines)
    inserting = start == end + 1
    if inserting:
        if start > line_count_before + 1:
            raise ValueError("insert start must be at most line_count + 1")
        replaced_lines: list[str] = []
    else:
        if start > end:
            raise ValueError("start must be <= end unless inserting with start == end + 1")
        if start > line_count_before or end > line_count_before:
            raise ValueError("line range is outside the file")
        replaced_lines = logical_lines[start - 1 : end]

    if expect_first is not None:
        if inserting:
            raise ValueError("expect_first is not valid for insert edits")
        _check_expected_line(
            replaced_lines[0],
            expect_first,
            mode=expect_mode,
            strip_indent=strip_indent,
            label="expect_first",
        )
    if expect_last is not None:
        if inserting:
            raise ValueError("expect_last is not valid for insert edits")
        _check_expected_line(
            replaced_lines[-1],
            expect_last,
            mode=expect_mode,
            strip_indent=strip_indent,
            label="expect_last",
        )

    new_lines = _logical_lines(new_text)
    after_lines = logical_lines[: start - 1] + new_lines + logical_lines[end:]
    after_text = _join_logical_lines(after_lines)
    chosen_newline: Literal["lf", "crlf", "cr", "none"] | None
    chosen_newline = None if newline == "preserve" else newline
    chosen_final_newline = (before.final_newline and bool(after_lines)) if final_newline is None else final_newline
    write_file(
        before.path,
        after_text,
        like=before,
        encoding=encoding,
        newline=chosen_newline,
        final_newline=chosen_final_newline,
        bom=bom,
    )
    after = read_text_lossless(before.path, encoding=encoding or before.encoding)
    line_count_after = len(after_lines)
    replaced_text = _join_logical_lines(replaced_lines)
    return EditResult(
        path=before.path,
        changed=before.text != after.text,
        replaced_text=replaced_text,
        line_count_before=line_count_before,
        line_count_after=line_count_after,
        line_delta=line_count_after - line_count_before,
    )


def make_unified_diff(
    before: str,
    after: str,
    *,
    path: str | None = None,
    context: int = 3,
) -> str:
    """Create a unified diff from two text values."""

    label = path or "text"
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            n=context,
        )
    )


def apply_patch_any(
    patch: str,
    *,
    cwd: str | Path | None = None,
    format: Literal["auto", "apply_patch", "unified"] = "auto",
    dry_run: bool = False,
    check: bool = True,
) -> PatchResult:
    """Apply either uv-agent patch envelopes or simple unified-diff patches."""

    selected = _detect_patch_format(patch) if format == "auto" else format
    if selected == "apply_patch":
        if dry_run:
            return _dry_run_apply_patch(patch, cwd=cwd, check=check)
        return apply_patch(patch, cwd=cwd, check=check)
    if selected == "unified":
        envelope = convert_patch(patch, from_format="unified", to_format="apply_patch")
        if dry_run:
            return _dry_run_apply_patch(envelope, cwd=cwd, check=check)
        return apply_patch(envelope, cwd=cwd, check=check)
    raise ValueError(f"unsupported patch format: {format!r}")


def convert_patch(
    patch: str,
    *,
    from_format: Literal["apply_patch", "unified"],
    to_format: Literal["apply_patch", "unified"],
) -> str:
    """Convert between simple unified diffs and uv-agent patch envelopes."""

    if from_format == to_format:
        return patch
    if from_format == "unified" and to_format == "apply_patch":
        return _unified_to_apply_patch(patch)
    raise ValueError(f"conversion {from_format!r} -> {to_format!r} is not supported")


@contextmanager
def workspace_transaction(paths: Sequence[str | Path] | None = None, *, root: str | Path = ".") -> Iterator[Snapshot]:
    """Snapshot selected files and restore them if the enclosed block fails."""

    snapshot = snapshot_files(paths or ["."], root=root)
    try:
        yield snapshot
    except BaseException:
        restore_snapshot(snapshot)
        raise


def snapshot_files(paths: Sequence[str | Path], *, root: str | Path = ".") -> Snapshot:
    """Capture file bytes for explicit restoration later."""

    resolved_root = resolve_workspace_path(root)
    captured: dict[str, bytes | None] = {}
    for item in paths:
        resolved = resolve_workspace_path(item, cwd=resolved_root)
        targets = sorted(path for path in resolved.rglob("*") if path.is_file()) if resolved.is_dir() else [resolved]
        if not targets and not resolved.exists():
            _record_snapshot_path(captured, resolved_root, resolved, None)
        for target in targets:
            _record_snapshot_path(captured, resolved_root, target, target.read_bytes() if target.exists() else None)
    return Snapshot(root=str(resolved_root), files=captured)


def restore_snapshot(snapshot: Snapshot) -> list[str]:
    """Restore a Snapshot captured by snapshot_files."""

    root = Path(snapshot.root).resolve()
    restored: list[str] = []
    for rel_path, data in snapshot.files.items():
        target = (root / rel_path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"snapshot path escapes root: {rel_path}") from exc
        if data is None:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
                restored.append(rel_path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(target, data)
        restored.append(rel_path)
    return sorted(restored)


def run_process_text(
    args: Sequence[str],
    *,
    cwd: str | Path | None = None,
    encoding: str = "utf-8",
    errors: str = "replace",
    env: Mapping[str, str] | None = None,
    env_patch: Mapping[str, str | None] | None = None,
    timeout_s: float | None = None,
    check: bool = False,
) -> CommandTextResult:
    """Run a command and decode stdout/stderr with explicit encoding policy."""

    process_env = _build_process_env(env, env_patch)
    argv = _resolve_process_args(args, env=process_env)
    process = subprocess.Popen(
        argv,
        cwd=None if cwd is None else str(resolve_workspace_path(cwd)),
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_subprocess_tree_kwargs(),
    )
    timed_out = False
    stdout_bytes: bytes | str | None = b""
    stderr_bytes: bytes | str | None = b""
    try:
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_bytes = _coerce_subprocess_output(exc.output)
        stderr_bytes = _coerce_subprocess_output(exc.stderr)
        # ``subprocess.run(..., timeout=...)`` only kills the direct child.  On
        # Windows in particular that can leave grandchildren (for example
        # ``uv run pytest``'s Python process) alive with stdout/stderr pipe
        # handles open, causing the retry ``communicate()`` to block until the
        # outer run_python timeout.  Kill the whole tree before collecting the
        # buffered output so helper-level timeouts are actually bounded.
        _kill_process_tree(process)
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # If the platform failed to tear down every descendant, still
            # return the output captured by the timeout exception instead of
            # letting the managed run hang indefinitely. Close any pipes we own
            # so the interpreter does not wait on them during cleanup.
            _kill_direct_process(process)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            stdout_bytes = _coerce_subprocess_output(exc.output)
            stderr_bytes = _coerce_subprocess_output(exc.stderr)
    result = CommandTextResult(
        args=argv,
        returncode=process.returncode if process.returncode is not None else -9,
        stdout=_decode_subprocess_output(stdout_bytes, encoding=encoding, errors=errors),
        stderr=_decode_subprocess_output(stderr_bytes, encoding=encoding, errors=errors),
        timed_out=timed_out,
    )
    if check:
        result.raise_for_error()
    return result


def _build_process_env(
    env: Mapping[str, str] | None,
    env_patch: Mapping[str, str | None] | None,
) -> dict[str, str]:
    """Return the exact environment map that should be passed to Popen."""

    process_env = os.environ.copy() if env is None else dict(env)
    for key, value in (env_patch or {}).items():
        if value is None:
            _unset_process_env_value(process_env, key)
        else:
            _set_process_env_value(process_env, key, value)
    return process_env


def _resolve_process_args(args: Sequence[str], *, env: Mapping[str, str]) -> list[str]:
    """Resolve Windows command shims with the same PATH passed to the child."""

    argv = list(args)
    if os.name != "nt" or not argv:
        return argv

    command = argv[0]
    if not command or os.path.dirname(command):
        return argv

    path_value = _get_process_env_value(env, "PATH")
    if path_value is None:
        return argv

    # On Windows, subprocess with shell=False does not reliably use the env= PATH
    # while locating the executable, and cmd.exe's PATHEXT expansion is not in
    # play. Resolve names such as ``npm`` to ``npm.cmd`` up front while keeping
    # shell=False so arguments are not reinterpreted by a shell.
    resolved = shutil.which(command, path=path_value)
    if resolved is not None:
        argv[0] = resolved
    return argv


def _get_process_env_value(env: Mapping[str, str], key: str) -> str | None:
    """Read an environment value using Windows' case-insensitive key rules."""

    if key in env:
        return env[key]
    if os.name == "nt":
        normalized = key.casefold()
        for candidate, value in env.items():
            if candidate.casefold() == normalized:
                return value
    return None


def _set_process_env_value(env: dict[str, str], key: str, value: str) -> None:
    """Set an environment value without leaving duplicate Windows key casings."""

    for candidate in _matching_process_env_keys(env, key):
        env.pop(candidate, None)
    env[key] = value


def _unset_process_env_value(env: dict[str, str], key: str) -> None:
    """Unset an environment value without leaving duplicate Windows key casings."""

    for candidate in _matching_process_env_keys(env, key):
        env.pop(candidate, None)


def _matching_process_env_keys(env: Mapping[str, str], key: str) -> list[str]:
    if os.name != "nt":
        return [key] if key in env else []
    normalized = key.casefold()
    return [candidate for candidate in env if candidate.casefold() == normalized]


def _subprocess_tree_kwargs() -> dict[str, Any]:
    """Start commands in their own process group when the platform supports it."""

    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


def _kill_process_tree(process: subprocess.Popen[Any]) -> None:
    """Best-effort synchronous process-tree termination for run_process_text."""

    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            _kill_direct_process(process)
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            _kill_direct_process(process)
        return
    _kill_direct_process(process)


def _kill_direct_process(process: subprocess.Popen[Any]) -> None:
    """Kill only the direct child, ignoring races with normal process exit."""

    if process.poll() is not None:
        return
    try:
        process.kill()
    except OSError:
        pass


def _coerce_subprocess_output(value: bytes | str | None) -> bytes:
    """Normalize TimeoutExpired partial output to bytes for shared decoding."""

    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode()


def _decode_subprocess_output(value: bytes | str | None, *, encoding: str, errors: str) -> str:
    """Decode captured subprocess output without assuming its exact type."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode(encoding, errors=errors)


def _coerce_text_file(value: TextFile | str | Path) -> TextFile:
    if isinstance(value, TextFile):
        return value
    return read_text_lossless(value)


def _coerce_write_like(value: FileView | TextFile | str | Path) -> TextFile | str | Path:
    if isinstance(value, FileView):
        return TextFile(
            path=value.path,
            text=value.text,
            encoding=value.encoding,
            newline=value.newline,
            final_newline=value.final_newline,
            bom=value.bom,
        )
    return value


def _path_kind(path: Path) -> Literal["file", "dir", "missing", "other"]:
    if path.is_file():
        return "file"
    if path.is_dir():
        return "dir"
    if path.exists():
        return "other"
    return "missing"


def _logical_lines(text: str) -> list[str]:
    """Split text into logical lines without retaining newline separators."""

    return text.splitlines()


def _join_logical_lines(lines: Sequence[str]) -> str:
    return "\n".join(lines)


def _selected_line_range(
    logical_lines: Sequence[str],
    *,
    lines: tuple[int, int] | None,
    head: int | None,
    tail: int | None,
    around: str | None,
    context: int,
) -> tuple[int, int]:
    line_count = len(logical_lines)
    if line_count == 0:
        if lines is not None and lines != (1, 0):
            raise ValueError("line range is outside the file")
        if around is not None:
            raise ValueError(f"around text not found: {around!r}")
        return (0, 0)
    if lines is not None:
        start, end = lines
        if start < 1 or end < start or end > line_count:
            raise ValueError("line range is outside the file")
        return (start, end)
    if head is not None:
        if head < 0:
            raise ValueError("head must be >= 0")
        if head == 0:
            return (0, 0)
        return (1, min(head, line_count))
    if tail is not None:
        if tail < 0:
            raise ValueError("tail must be >= 0")
        if tail == 0:
            return (0, 0)
        start = max(1, line_count - tail + 1)
        return (start, line_count)
    if around is not None:
        if not around:
            raise ValueError("around must be non-empty")
        for index, line in enumerate(logical_lines, start=1):
            if around in line:
                return (max(1, index - context), min(line_count, index + context))
        raise ValueError(f"around text not found: {around!r}")
    return (1, line_count)


def _slice_text_by_lines(text: str, start_line: int, end_line: int) -> str:
    if start_line == 0:
        return ""
    return "".join(text.splitlines(keepends=True)[start_line - 1 : end_line])


def _check_expected_line(
    line: str,
    expected: str,
    *,
    mode: Literal["startswith", "contains", "exact", "regex"],
    strip_indent: bool,
    label: str,
) -> None:
    actual = line.lstrip() if strip_indent else line
    if mode == "startswith":
        matched = actual.startswith(expected)
    elif mode == "contains":
        matched = expected in actual
    elif mode == "exact":
        matched = actual == expected
    elif mode == "regex":
        matched = re.search(expected, actual) is not None
    else:  # Defensive guard for future edits; public validation happens earlier.
        raise ValueError(f"unsupported expect_mode: {mode!r}")
    if not matched:
        raise ValueError(f"{label} did not match: expected {expected!r} with {mode}, got {actual!r}")


def _detect_newline_style(text: str) -> Literal["lf", "crlf", "cr", "mixed", "none"]:
    crlf = text.count("\r\n")
    without_crlf = text.replace("\r\n", "")
    lf = without_crlf.count("\n")
    cr = without_crlf.count("\r")
    styles = sum(1 for count in (crlf, lf, cr) if count)
    if styles == 0:
        return "none"
    if styles > 1:
        return "mixed"
    if crlf:
        return "crlf"
    if cr:
        return "cr"
    return "lf"


def _single_newline_style(
    style: str,
) -> Literal["lf", "crlf", "cr", "none"] | None:
    if style == "lf":
        return "lf"
    if style == "crlf":
        return "crlf"
    if style == "cr":
        return "cr"
    if style == "none":
        return "none"
    return None


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _strip_final_newline(text: str) -> str:
    return text[:-1] if text.endswith(("\n", "\r")) else text


def _final_newline_for_text(text: str, eol: str | None) -> str:
    if eol == "crlf":
        return "\r\n"
    if eol == "cr":
        return "\r"
    if eol == "lf":
        return "\n"
    style = _detect_newline_style(text)
    if style == "crlf":
        return "\r\n"
    if style == "cr":
        return "\r"
    return "\n"


def _first_line_difference(left: str, right: str) -> tuple[int | None, str | None, str | None]:
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    for index, (left_line, right_line) in enumerate(zip(left_lines, right_lines), start=1):
        if left_line != right_line:
            return index, left_line, right_line
    if len(left_lines) != len(right_lines):
        index = min(len(left_lines), len(right_lines)) + 1
        left_line = left_lines[index - 1] if index <= len(left_lines) else None
        right_line = right_lines[index - 1] if index <= len(right_lines) else None
        return index, left_line, right_line
    return None, None, None


def _replacement_candidates(
    before: TextFile,
    old: str,
    new: str,
    newlines: Literal["logical", "raw"],
) -> list[tuple[str, str]]:
    if newlines == "raw":
        return [(old, new)]

    styles: list[Literal["lf", "crlf", "cr"]] = []
    single_style = _single_newline_style(before.newline)
    if single_style == "lf":
        styles.append("lf")
    elif single_style == "crlf":
        styles.append("crlf")
    elif single_style == "cr":
        styles.append("cr")
    elif before.newline == "mixed":
        styles.extend(_newline_styles_in_text(before.text))
    styles.append("lf")

    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for style in styles:
        candidate = (normalize_text(old, eol=style), normalize_text(new, eol=style))
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)
    raw_candidate = (old, new)
    if raw_candidate not in seen:
        candidates.append(raw_candidate)
    return candidates


def _newline_styles_in_text(text: str) -> list[Literal["lf", "crlf", "cr"]]:
    styles: list[Literal["lf", "crlf", "cr"]] = []
    if "\r\n" in text:
        styles.append("crlf")
    without_crlf = text.replace("\r\n", "")
    if "\n" in without_crlf:
        styles.append("lf")
    if "\r" in without_crlf:
        styles.append("cr")
    return styles


def _replacement_missing_context(
    before: TextFile,
    needle: str,
    *,
    found: int,
    count: int,
    newlines: Literal["logical", "raw"],
) -> str:
    parts = [
        f" File newline={before.newline!r}, final_newline={before.final_newline!r}.",
        f" Search text repr={_short_repr(needle)}.",
    ]
    if newlines == "raw" and "\n" in needle and before.newline == "crlf":
        parts.append(" Raw matching is newline-sensitive; this may be a CRLF/LF mismatch.")
    elif newlines == "logical" and found == 0 and "\n" in needle and before.newline == "mixed":
        parts.append(" The file has mixed newlines; inspect the target snippet when matching across lines.")
    if found < count:
        context = _missing_context(before.text, needle)
        if context:
            parts.append(context)
    return "".join(parts)


def _short_repr(text: str, *, limit: int = 160) -> str:
    value = repr(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _missing_context(text: str, needle: str) -> str:
    if not needle:
        return " Empty search text is not allowed."
    first = needle.splitlines()[0] if needle.splitlines() else needle
    for line_no, line in enumerate(text.splitlines(), start=1):
        if first and first in line:
            return f" First needle line appears near line {line_no}: {line[:120]!r}"
    return ""


def _detect_patch_format(patch: str) -> Literal["apply_patch", "unified"]:
    stripped = patch.lstrip()
    if stripped.startswith("*** Begin Patch"):
        return "apply_patch"
    if stripped.startswith("diff --git") or stripped.startswith("--- "):
        return "unified"
    raise ValueError("could not detect patch format")


def _dry_run_apply_patch(patch: str, *, cwd: str | Path | None, check: bool) -> PatchResult:
    """Validate a patch without snapshotting unrelated workspace files."""

    return dry_run_patch(patch, cwd=cwd, check=check)


def _file_paths_under(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    if root.is_file():
        return {root.resolve()}
    return {path.resolve() for path in root.rglob("*") if path.is_file()}


def _remove_empty_parents(path: Path, stop: Path) -> None:
    stop = stop.resolve()
    current = path.resolve()
    while current != stop and current.exists():
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _unified_to_apply_patch(diff: str) -> str:
    files = PatchSet(diff.splitlines(keepends=True))
    if not files:
        raise ValueError("unified diff contains no file headers")
    lines = ["*** Begin Patch"]
    for file_diff in files:
        old_path = _patched_source_path(file_diff)
        new_path = _patched_target_path(file_diff)
        if file_diff.is_binary_file:
            raise ValueError(f"binary diffs are not supported: {new_path or old_path}")
        if file_diff.is_added_file:
            lines.append(f"*** Add File: {new_path}")
            for hunk in file_diff:
                for line in hunk:
                    if line.is_added:
                        lines.append(_apply_patch_line("+", line.value))
            continue
        if file_diff.is_removed_file:
            lines.append(f"*** Delete File: {old_path}")
            continue
        lines.append(f"*** Update File: {old_path}")
        if file_diff.is_rename or new_path != old_path:
            lines.append(f"*** Move to: {new_path}")
        for hunk in file_diff:
            lines.append("@@")
            for line in hunk:
                if line.is_context:
                    lines.append(_apply_patch_line(" ", line.value))
                elif line.is_removed:
                    lines.append(_apply_patch_line("-", line.value))
                elif line.is_added:
                    lines.append(_apply_patch_line("+", line.value))
    lines.append("*** End Patch")
    return "\n".join(lines) + "\n"


def _clean_diff_path(path: str) -> str:
    if path == "/dev/null":
        return path
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _patched_source_path(file_diff: PatchedFile) -> str:
    return _clean_diff_path(file_diff.source_file)


def _patched_target_path(file_diff: PatchedFile) -> str:
    return _clean_diff_path(file_diff.target_file)


def _apply_patch_line(prefix: str, value: str) -> str:
    return prefix + value.rstrip("\r\n")


def _record_snapshot_path(captured: dict[str, bytes | None], root: Path, path: Path, data: bytes | None) -> None:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"snapshot path escapes root: {path}") from exc
    captured[str(rel)] = data
