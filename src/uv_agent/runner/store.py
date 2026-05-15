from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uv_agent.ids import new_id
from uv_agent.jsonl import read_jsonl
from uv_agent.time import utc_now_iso


class ScriptStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
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
