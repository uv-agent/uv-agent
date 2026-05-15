from __future__ import annotations

from pathlib import Path


def resolve_workspace_path(path: str | Path, *, cwd: str | Path | None = None) -> Path:
    """Resolve a script-provided path against the current working directory."""
    base = Path(cwd) if cwd is not None else Path.cwd()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def read_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read a text file using an explicit encoding."""
    return resolve_workspace_path(path).read_text(encoding=encoding)


def write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write a text file, creating parent directories as needed."""
    resolved = resolve_workspace_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding=encoding)
    return resolved


def list_files(
    root: str | Path = ".",
    *,
    pattern: str = "*",
    include_dirs: bool = False,
    max_count: int = 1000,
) -> list[str]:
    """List workspace-relative files matching a glob pattern."""
    resolved = resolve_workspace_path(root)
    matches: list[str] = []
    for path in resolved.rglob(pattern):
        if not include_dirs and path.is_dir():
            continue
        matches.append(str(path.relative_to(resolved)))
        if len(matches) >= max_count:
            break
    return matches


def read_json(path: str | Path, *, encoding: str = "utf-8") -> object:
    """Read JSON from a workspace-relative path."""
    import json

    return json.loads(read_text(path, encoding=encoding))


def write_json(
    path: str | Path,
    value: object,
    *,
    encoding: str = "utf-8",
    indent: int = 2,
) -> Path:
    """Write JSON to a workspace-relative path."""
    import json

    return write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=indent) + "\n",
        encoding=encoding,
    )
