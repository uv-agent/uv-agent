from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from uv_agent.ids import new_id
from uv_agent.jsonl import read_jsonl
from uv_agent.time import utc_now_iso


class ScriptStore:
    def __init__(self, data_dir: Path, *, max_saved_scripts: int = 32) -> None:
        self.data_dir = data_dir
        self.max_saved_scripts = max(1, max_saved_scripts)
        self.scripts_dir = data_dir / "scripts"
        self.runs_dir = data_dir / "runs"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def create_script(
        self,
        *,
        original_code: str,
        final_code: str,
        thread_id: str | None,
        turn_id: str | None,
    ) -> tuple[str, Path, Path]:
        script_id = new_id("scr")
        script_dir = self.scripts_dir / script_id
        script_dir.mkdir(parents=True, exist_ok=False)
        original_path = script_dir / "script.original.py"
        final_path = script_dir / "script.py"
        metadata_path = script_dir / "metadata.json"
        original_path.write_text(original_code, encoding="utf-8")
        final_path.write_text(final_code, encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "script_id": script_id,
                    "created_at": utc_now_iso(),
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "original_path": str(original_path),
                    "final_path": str(final_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.prune_scripts()
        return script_id, original_path, final_path

    def get_script(self, script_id: str) -> dict[str, Any]:
        script_dir = self.scripts_dir / script_id
        metadata_path = script_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Unknown script_id: {script_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["code"] = Path(metadata["final_path"]).read_text(encoding="utf-8")
        return metadata

    def create_run_log_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.jsonl"

    def find_run(self, run_id: str) -> dict[str, Any]:
        path = self.runs_dir / f"{run_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Unknown run_id: {run_id}")
        started = next((event for event in read_jsonl(path) if event.get("type") == "run.started"), None)
        if not started:
            raise ValueError(f"Run log does not contain run.started: {run_id}")
        return {"path": path, **started}

    def list_scripts(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        run_counts = self._run_counts()
        latest_runs = self._latest_run_times()
        for metadata_path in self.scripts_dir.glob("*/metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                final_path = Path(metadata["final_path"])
                code = final_path.read_text(encoding="utf-8") if final_path.exists() else ""
            except Exception:
                continue
            script_id = str(metadata.get("script_id") or metadata_path.parent.name)
            summaries.append(
                {
                    "script_id": script_id,
                    "created_at": metadata.get("created_at"),
                    "last_used_at": latest_runs.get(script_id) or metadata.get("created_at"),
                    "thread_id": metadata.get("thread_id"),
                    "turn_id": metadata.get("turn_id"),
                    "run_count": run_counts.get(script_id, 0),
                    "summary": script_summary(code),
                    "final_path": str(final_path),
                    "original_path": metadata.get("original_path"),
                }
            )
        summaries.sort(key=lambda item: str(item.get("last_used_at") or ""), reverse=True)
        return summaries[:limit] if limit is not None else summaries

    def prune_scripts(self) -> None:
        scripts = self.list_scripts()
        for item in scripts[self.max_saved_scripts :]:
            script_id = str(item.get("script_id") or "")
            if not script_id:
                continue
            shutil.rmtree(self.scripts_dir / script_id, ignore_errors=True)

    def _run_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for run_path in self.runs_dir.glob("*.jsonl"):
            started = next((event for event in read_jsonl(run_path) if event.get("type") == "run.started"), None)
            if not started:
                continue
            script_id = str(started.get("script_id") or "")
            if script_id:
                counts[script_id] = counts.get(script_id, 0) + 1
        return counts

    def _latest_run_times(self) -> dict[str, str]:
        latest: dict[str, str] = {}
        for run_path in self.runs_dir.glob("*.jsonl"):
            started = next((event for event in read_jsonl(run_path) if event.get("type") == "run.started"), None)
            if not started:
                continue
            script_id = str(started.get("script_id") or "")
            created_at = str(started.get("created_at") or "")
            if script_id and created_at > latest.get(script_id, ""):
                latest[script_id] = created_at
        return latest


def script_summary(code: str) -> str:
    """Return a compact first-meaningful-line summary for a managed script."""
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
