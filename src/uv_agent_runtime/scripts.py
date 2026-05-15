from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def saved_scripts(limit: int = 32, state_dir: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Return recent managed script summaries for this uv-agent project.

    Temporary scripts receive ``UV_AGENT_STATE_DIR`` from the host runner. Passing
    ``state_dir`` is mainly useful for tests or external library use.
    """
    root = Path(state_dir or os.environ.get("UV_AGENT_STATE_DIR") or ".uv-agent")
    scripts_dir = root / "scripts"
    runs_dir = root / "runs"
    run_counts = _run_counts(runs_dir)
    latest_runs = _latest_runs(runs_dir)
    items: list[dict[str, Any]] = []
    for metadata_path in scripts_dir.glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            final_path = Path(metadata["final_path"])
            code = final_path.read_text(encoding="utf-8") if final_path.exists() else ""
        except Exception:
            continue
        script_id = str(metadata.get("script_id") or metadata_path.parent.name)
        items.append(
            {
                "script_id": script_id,
                "created_at": metadata.get("created_at"),
                "last_used_at": latest_runs.get(script_id) or metadata.get("created_at"),
                "thread_id": metadata.get("thread_id"),
                "turn_id": metadata.get("turn_id"),
                "run_count": run_counts.get(script_id, 0),
                "summary": _summary(code),
                "final_path": str(final_path),
            }
        )
    items.sort(key=lambda item: str(item.get("last_used_at") or ""), reverse=True)
    return items[: max(1, int(limit))]


def _run_counts(runs_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in runs_dir.glob("*.jsonl"):
        started = _run_started(path)
        script_id = str(started.get("script_id") or "") if started else ""
        if script_id:
            counts[script_id] = counts.get(script_id, 0) + 1
    return counts


def _latest_runs(runs_dir: Path) -> dict[str, str]:
    latest: dict[str, str] = {}
    for path in runs_dir.glob("*.jsonl"):
        started = _run_started(path)
        script_id = str(started.get("script_id") or "") if started else ""
        created_at = str(started.get("created_at") or "") if started else ""
        if script_id and created_at > latest.get(script_id, ""):
            latest[script_id] = created_at
    return latest


def _run_started(path: Path) -> dict[str, Any] | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") == "run.started":
                return event
    except Exception:
        return None
    return None


def _summary(code: str) -> str:
    in_metadata = False
    for line in code.splitlines():
        stripped = line.strip()
        if stripped == "# /// script":
            in_metadata = True
            continue
        if in_metadata:
            if stripped == "# ///":
                in_metadata = False
            continue
        if not stripped or stripped.startswith("#"):
            continue
        return stripped[:160]
    return "(empty script)"
