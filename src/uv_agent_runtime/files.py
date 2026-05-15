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
