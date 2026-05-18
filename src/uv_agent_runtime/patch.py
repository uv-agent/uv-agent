from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PatchResult:
    returncode: int
    stdout: str
    stderr: str
    changed_files: list[str]


@dataclass(frozen=True)
class _AddFile:
    path: str
    lines: list[str]


@dataclass(frozen=True)
class _DeleteFile:
    path: str


@dataclass(frozen=True)
class _UpdateFile:
    path: str
    hunks: list[list[str]]
    move_to: str | None = None


_FileOp = _AddFile | _DeleteFile | _UpdateFile
_MISSING = object()


def apply_patch(patch: str, *, cwd: str | Path | None = None, check: bool = True) -> PatchResult:
    """Apply a Codex-style custom patch envelope and report changed paths."""
    workdir = Path(cwd).resolve() if cwd is not None else Path.cwd()
    try:
        ops = _parse_patch(patch)
        changed_files = _changed_files(ops)
        _apply_ops(workdir, ops)
    except Exception as exc:
        result = PatchResult(returncode=1, stdout="", stderr=str(exc), changed_files=[])
        if check:
            raise RuntimeError(f"patch failed with exit 1:\n{result.stderr}") from exc
        return result

    return PatchResult(returncode=0, stdout="", stderr="", changed_files=sorted(changed_files))


def _parse_patch(patch: str) -> list[_FileOp]:
    lines = patch.splitlines()
    if not lines:
        raise ValueError("patch is empty")
    if lines[0] != "*** Begin Patch":
        raise ValueError("patch must start with *** Begin Patch")
    if lines[-1] != "*** End Patch":
        raise ValueError("patch must end with *** End Patch")

    ops: list[_FileOp] = []
    index = 1
    while index < len(lines) - 1:
        line = lines[index]
        if line.startswith("*** Add File: "):
            path = _parse_path(line, "*** Add File: ")
            index += 1
            added: list[str] = []
            while index < len(lines) - 1 and not lines[index].startswith("*** "):
                current = lines[index]
                if not current.startswith("+"):
                    raise ValueError(f"add file {path} contains a non-added line: {current!r}")
                added.append(current[1:])
                index += 1
            ops.append(_AddFile(path=path, lines=added))
        elif line.startswith("*** Delete File: "):
            ops.append(_DeleteFile(path=_parse_path(line, "*** Delete File: ")))
            index += 1
        elif line.startswith("*** Update File: "):
            path = _parse_path(line, "*** Update File: ")
            index += 1
            move_to: str | None = None
            hunks: list[list[str]] = []
            if index < len(lines) - 1 and lines[index].startswith("*** Move to: "):
                move_to = _parse_path(lines[index], "*** Move to: ")
                index += 1
            while index < len(lines) - 1:
                if lines[index] == "*** End of File":
                    index += 1
                    continue
                if lines[index].startswith("*** "):
                    break
                if not lines[index].startswith("@@"):
                    raise ValueError(f"update file {path} expected @@ hunk marker, got: {lines[index]!r}")
                index += 1
                hunk: list[str] = []
                while (
                    index < len(lines) - 1
                    and not lines[index].startswith("@@")
                    and lines[index] != "*** End of File"
                    and not lines[index].startswith("*** ")
                ):
                    current = lines[index]
                    if not current:
                        raise ValueError(
                            f"update file {path} contains a blank hunk line without a diff prefix. "
                            "Every hunk line must start with a space for context, - for removal, "
                            "or + for addition; use a single space line for blank context, "
                            "or + / - by itself to add or remove a blank line."
                        )
                    if current[0] not in " +-":
                        raise ValueError(
                            f"update file {path} contains an invalid hunk line: {current!r}. "
                            "Every hunk line must start with a space for context, - for removal, "
                            "or + for addition."
                        )
                    hunk.append(current)
                    index += 1
                if not hunk:
                    raise ValueError(f"update file {path} contains an empty hunk")
                hunks.append(hunk)
            if not hunks and move_to is None:
                raise ValueError(f"update file {path} has no hunks or move target")
            ops.append(_UpdateFile(path=path, hunks=hunks, move_to=move_to))
        else:
            raise ValueError(f"unexpected patch line: {line!r}")

    if not ops:
        raise ValueError("patch contains no file operations")
    return ops


def _parse_path(line: str, prefix: str) -> str:
    path = line[len(prefix) :].strip()
    if not path:
        raise ValueError(f"{prefix.strip()} requires a path")
    return path


def _changed_files(ops: list[_FileOp]) -> set[str]:
    changed: set[str] = set()
    for op in ops:
        changed.add(op.path)
        if isinstance(op, _UpdateFile) and op.move_to is not None:
            changed.add(op.move_to)
    return changed


def _apply_ops(workdir: Path, ops: list[_FileOp]) -> None:
    pending: dict[Path, str | None] = {}
    for op in ops:
        if isinstance(op, _AddFile):
            path = _resolve_patch_path(workdir, op.path)
            current = _read_pending(path, pending)
            if current is not _MISSING and current is not None:
                raise FileExistsError(f"add file already exists: {op.path}")
            if current is _MISSING and path.exists():
                raise FileExistsError(f"add file already exists: {op.path}")
            pending[path] = _join_lines(op.lines)
        elif isinstance(op, _DeleteFile):
            path = _resolve_patch_path(workdir, op.path)
            current = _read_pending(path, pending)
            if current is None:
                raise FileNotFoundError(f"delete file does not exist: {op.path}")
            if current is _MISSING and not path.exists():
                raise FileNotFoundError(f"delete file does not exist: {op.path}")
            pending[path] = None
        elif isinstance(op, _UpdateFile):
            source = _resolve_patch_path(workdir, op.path)
            text = _read_pending(source, pending)
            if text is None:
                raise FileNotFoundError(f"update file does not exist: {op.path}")
            if text is _MISSING:
                if not source.exists():
                    raise FileNotFoundError(f"update file does not exist: {op.path}")
                text = source.read_text(encoding="utf-8", newline="")
            for hunk in op.hunks:
                text = _apply_hunk(text, hunk, op.path)
            if op.move_to is None:
                pending[source] = text
            else:
                target = _resolve_patch_path(workdir, op.move_to)
                current = _read_pending(target, pending)
                if target != source and current is not _MISSING and current is not None:
                    raise FileExistsError(f"move target already exists: {op.move_to}")
                if target != source and current is _MISSING and target.exists():
                    raise FileExistsError(f"move target already exists: {op.move_to}")
                pending[source] = None
                pending[target] = text

    for path, text in pending.items():
        if text is None:
            if path.exists():
                path.unlink()
                _remove_empty_parents(path.parent, workdir)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="")


def _resolve_patch_path(workdir: Path, path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workdir / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workdir)
    except ValueError as exc:
        raise ValueError(f"patch path escapes the working directory: {path}") from exc
    return resolved


def _read_pending(path: Path, pending: dict[Path, str | None]) -> str | None | object:
    if path in pending:
        return pending[path]
    return _MISSING


def _join_lines(lines: list[str]) -> str:
    return "\n".join(lines) + ("\n" if lines else "")


def _apply_hunk(text: str, hunk: list[str], path: str) -> str:
    original = text.splitlines(keepends=True)
    expected = [line[1:] for line in hunk if line[0] in " -"]
    start = _find_subsequence(original, expected)
    if start is None:
        raise ValueError(f"hunk context was not found in {path}")
    replacement = _replacement_lines(original[start : start + len(expected)], hunk, original)
    return "".join(original[:start] + replacement + original[start + len(expected) :])


def _replacement_lines(matched: list[str], hunk: list[str], original: list[str]) -> list[str]:
    newline = _detect_newline(original)
    replacement: list[str] = []
    matched_index = 0
    for line in hunk:
        kind = line[0]
        value = line[1:]
        if kind == " ":
            replacement.append(matched[matched_index])
            matched_index += 1
        elif kind == "-":
            matched_index += 1
        elif kind == "+":
            replacement.append(value + newline)
    return replacement


def _detect_newline(lines: list[str]) -> str:
    for line in lines:
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
    return "\n"


def _find_subsequence(lines: list[str], target: list[str]) -> int | None:
    if not target:
        return 0
    last_start = len(lines) - len(target)
    for index in range(last_start + 1):
        if [_line_body(line) for line in lines[index : index + len(target)]] == target:
            return index
    return None


def _line_body(line: str) -> str:
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith("\n"):
        return line[:-1]
    return line


def _remove_empty_parents(path: Path, stop: Path) -> None:
    while path != stop and path.exists():
        try:
            path.rmdir()
        except OSError:
            return
        path = path.parent
