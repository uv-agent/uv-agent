from __future__ import annotations

from pathlib import Path


class RunLogStore:
    def __init__(self, runs_dir: Path, *, max_run_logs: int = 200) -> None:
        self.runs_dir = runs_dir
        self.max_run_logs = max(1, max_run_logs)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def script_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.py"

    def log_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.jsonl"

    def create_run_files(self, run_id: str, code: str) -> tuple[Path, Path]:
        script_path = self.script_path(run_id)
        log_path = self.log_path(run_id)
        script_path.write_text(code, encoding="utf-8")
        return script_path, log_path

    def prune(self) -> None:
        logs = sorted(
            self.runs_dir.glob("*.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for log_path in logs[self.max_run_logs :]:
            run_id = log_path.stem
            self.script_path(run_id).unlink(missing_ok=True)
            log_path.unlink(missing_ok=True)
