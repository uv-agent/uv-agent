from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from collections.abc import Callable, Awaitable

from uv_agent.ids import new_id
from uv_agent.time import utc_now_iso
from .service import connect_workflow_db

NODE_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
logger = logging.getLogger(__name__)


@dataclass
class WorkflowExecutor:
    """Host-side executor for persisted workflow nodes.

    Runtime scripts create and mutate workflow graphs, but node turns must be
    owned by the long-lived host so they can share TurnManager concurrency and
    avoid spawning nested uv-agent subprocesses. SQLite status updates claim work;
    short leases let a later host mark abandoned running nodes stale without
    touching work that another active TUI/daemon is still heartbeating.
    """

    data_dir: Path
    submit_turn: Callable[..., Awaitable[Any]]
    threads: Any
    default_poll_interval_s: float = 0.5
    lease_seconds: float = 30.0
    heartbeat_interval_s: float = 5.0

    def __post_init__(self) -> None:
        self.executor_id = new_id("wfx")
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._node_tasks: set[asyncio.Task[None]] = set()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self.cleanup_stale_running()
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self._run(), name="uv-agent-workflow-executor")
            logger.info(
                "Workflow executor started executor_id=%s data_dir=%s poll_interval_s=%s",
                self.executor_id,
                self.data_dir,
                self.default_poll_interval_s,
            )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
        for task in list(self._node_tasks):
            task.cancel()
        await asyncio.gather(*([self._task] if self._task is not None else []), *self._node_tasks, return_exceptions=True)
        self._node_tasks.clear()
        logger.info("Workflow executor stopped executor_id=%s", self.executor_id)

    def cleanup_stale_running(self) -> int:
        """Fail running nodes whose lease has expired.

        A fresh lease means another host process may still be executing the
        node, so startup cleanup must not blindly fail every running node.
        """

        now = utc_now_iso()
        error = {"type": "WorkflowNodeLeaseExpired", "message": "Workflow node lease expired before completion."}
        with connect_workflow_db(self.data_dir) as db:
            stale = db.execute(
                "SELECT node_id, workflow_id FROM workflow_nodes WHERE status = 'running' AND (lease_until IS NULL OR lease_until < ?)",
                (now,),
            ).fetchall()
            for row in stale:
                db.execute(
                    """
                    UPDATE workflow_nodes
                    SET status = 'failed', completed_at = ?, error_json = ?, executor_id = NULL, lease_until = NULL
                    WHERE node_id = ? AND status = 'running'
                    """,
                    (now, _dumps(error), row["node_id"]),
                )
            workflow_ids = sorted({row["workflow_id"] for row in stale})
            for workflow_id in workflow_ids:
                unresolved = db.execute(
                    "SELECT 1 FROM workflow_checkpoints WHERE workflow_id = ? AND status = 'unresolved' LIMIT 1",
                    (workflow_id,),
                ).fetchone()
                if unresolved is None:
                    db.execute("UPDATE workflows SET status = 'failed', updated_at = ? WHERE workflow_id = ?", (now, workflow_id))
        if stale:
            logger.warning("Workflow executor marked stale running nodes failed count=%d", len(stale))
        return len(stale)

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.cleanup_stale_running()
                await self.run_once()
            except Exception:
                # Background host infrastructure should not crash the UI/daemon;
                # failures are recorded on individual nodes where possible.
                logger.exception("Workflow executor loop failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.default_poll_interval_s)
            except asyncio.TimeoutError:
                pass

    async def run_once(self) -> None:
        for workflow in self._running_workflows():
            if self._has_unresolved_checkpoint(workflow["workflow_id"]):
                continue
            if self._claim_ready_checkpoint(workflow) is not None:
                continue
            for node in self._claim_ready_agent_nodes(workflow):
                logger.info(
                    "Workflow node claimed workflow_id=%s node_id=%s kind=%s executor_id=%s",
                    workflow["workflow_id"],
                    node["node_id"],
                    node["kind"],
                    self.executor_id,
                )
                task = asyncio.create_task(self._execute_node(workflow, node), name=f"uv-agent-workflow-node-{node['node_id']}")
                self._node_tasks.add(task)
                task.add_done_callback(self._node_tasks.discard)
            self._complete_if_done(workflow["workflow_id"])

    def _running_workflows(self) -> list[dict[str, Any]]:
        with connect_workflow_db(self.data_dir) as db:
            rows = db.execute("SELECT * FROM workflows WHERE status = 'running' ORDER BY updated_at ASC").fetchall()
        return [dict(row) for row in rows]

    def _nodes(self, workflow_id: str) -> list[dict[str, Any]]:
        with connect_workflow_db(self.data_dir) as db:
            rows = db.execute("SELECT * FROM workflow_nodes WHERE workflow_id = ? ORDER BY created_at ASC, rowid ASC", (workflow_id,)).fetchall()
        return [dict(row) for row in rows]

    def _ready_nodes(self, workflow_id: str, *, kind: str | None = None, executable_only: bool = False) -> list[dict[str, Any]]:
        nodes = self._nodes(workflow_id)
        completed = {row["node_id"] for row in nodes if row["status"] == "completed"}
        ready: list[dict[str, Any]] = []
        for row in nodes:
            if row["status"] != "pending":
                continue
            if kind is not None and row["kind"] != kind:
                continue
            if executable_only and row["kind"] not in {"agent", "review"}:
                continue
            deps = _loads(row.get("dependencies_json"), [])
            if all(dep in completed for dep in deps):
                ready.append(row)
        return ready

    def _has_unresolved_checkpoint(self, workflow_id: str) -> bool:
        with connect_workflow_db(self.data_dir) as db:
            return db.execute(
                "SELECT 1 FROM workflow_checkpoints WHERE workflow_id = ? AND status = 'unresolved' LIMIT 1",
                (workflow_id,),
            ).fetchone() is not None

    def _claim_ready_checkpoint(self, workflow: dict[str, Any]) -> dict[str, Any] | None:
        ready = self._ready_nodes(workflow["workflow_id"], kind="checkpoint")
        if not ready:
            return None
        node = ready[0]
        now = utc_now_iso()
        with connect_workflow_db(self.data_dir) as db:
            changed = db.execute(
                """
                UPDATE workflow_nodes
                SET status = 'completed', started_at = ?, completed_at = ?, executor_id = ?, lease_until = NULL
                WHERE node_id = ? AND status = 'pending'
                """,
                (now, now, self.executor_id, node["node_id"]),
            ).rowcount
            if not changed:
                return None
            checkpoint = db.execute("SELECT * FROM workflow_checkpoints WHERE node_id = ? LIMIT 1", (node["node_id"],)).fetchone()
            if checkpoint is None:
                return None
            db.execute("UPDATE workflow_checkpoints SET status = 'unresolved' WHERE checkpoint_id = ? AND status = 'pending'", (checkpoint["checkpoint_id"],))
            db.execute(
                "UPDATE workflows SET status = 'checkpoint', current_checkpoint_id = ?, updated_at = ? WHERE workflow_id = ?",
                (checkpoint["checkpoint_id"], now, workflow["workflow_id"]),
            )
        return dict(checkpoint)

    def _claim_ready_agent_nodes(self, workflow: dict[str, Any]) -> list[dict[str, Any]]:
        state = _loads(workflow.get("state_json"), {})
        ready = self._ready_nodes(workflow["workflow_id"], executable_only=True)
        concurrency = max(1, int(state.get("concurrency") or min(4, len(ready)) or 1))
        claimed: list[dict[str, Any]] = []
        now = utc_now_iso()
        lease_until = _lease_deadline(self.lease_seconds)
        with connect_workflow_db(self.data_dir) as db:
            for node in ready[:concurrency]:
                changed = db.execute(
                    """
                    UPDATE workflow_nodes
                    SET status = 'running', started_at = ?, executor_id = ?, lease_until = ?
                    WHERE node_id = ? AND status = 'pending'
                    """,
                    (now, self.executor_id, lease_until, node["node_id"]),
                ).rowcount
                if changed:
                    claimed_node = dict(node)
                    claimed_node["started_at"] = now
                    claimed_node["executor_id"] = self.executor_id
                    claimed.append(claimed_node)
        return claimed

    async def _execute_node(self, workflow: dict[str, Any], node: dict[str, Any]) -> None:
        node_id = str(node["node_id"])
        level = str(node.get("model_level") or workflow.get("default_model_level") or "") or None
        thread_id = str(node.get("thread_id") or "") or self._create_node_thread(workflow, node)
        heartbeat = asyncio.create_task(self._heartbeat(node_id), name=f"uv-agent-workflow-heartbeat-{node_id}")
        logger.info(
            "Workflow node execution started workflow_id=%s node_id=%s thread_id=%s level=%s",
            workflow["workflow_id"],
            node_id,
            thread_id,
            level,
        )
        try:
            submitted = await self.submit_turn(
                text=str(node.get("prompt") or ""),
                thread_id=thread_id,
                level=level,
                conflict="queue",
            )
            await submitted.wait()
            completed = utc_now_iso()
            if submitted.status == "completed":
                status = "completed"
                error = {}
            elif submitted.status == "cancelled":
                status = "cancelled"
                error = {"type": "Cancelled", "message": "Workflow node turn was cancelled."}
            else:
                status = "failed"
                error = {"type": str(submitted.status), "message": str(submitted.error or submitted.status)}
            result = {"thread_id": thread_id, "turn_id": submitted.turn_id, "model_level": level, "status": submitted.status}
            with connect_workflow_db(self.data_dir) as db:
                db.execute(
                    """
                    UPDATE workflow_nodes
                    SET status = ?, completed_at = ?, thread_id = ?, result_summary = ?, result_json = ?, error_json = ?, lease_until = NULL
                    WHERE node_id = ? AND executor_id = ?
                    """,
                    (status, completed, thread_id, submitted.final_text, _dumps(result), _dumps(error), node_id, self.executor_id),
                )
                db.execute("UPDATE workflows SET updated_at = ? WHERE workflow_id = ?", (completed, workflow["workflow_id"]))
            logger.info(
                "Workflow node execution completed workflow_id=%s node_id=%s thread_id=%s status=%s turn_id=%s",
                workflow["workflow_id"],
                node_id,
                thread_id,
                status,
                submitted.turn_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            completed = utc_now_iso()
            with connect_workflow_db(self.data_dir) as db:
                db.execute(
                    """
                    UPDATE workflow_nodes
                    SET status = 'failed', completed_at = ?, thread_id = ?, error_json = ?, lease_until = NULL
                    WHERE node_id = ? AND executor_id = ?
                    """,
                    (completed, thread_id, _dumps({"type": exc.__class__.__name__, "message": str(exc) or repr(exc)}), node_id, self.executor_id),
                )
                db.execute("UPDATE workflows SET updated_at = ? WHERE workflow_id = ?", (completed, workflow["workflow_id"]))
            logger.warning(
                "Workflow node execution failed workflow_id=%s node_id=%s thread_id=%s error_type=%s",
                workflow["workflow_id"],
                node_id,
                thread_id,
                exc.__class__.__name__,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            self._complete_if_done(workflow["workflow_id"])

    def _create_node_thread(self, workflow: dict[str, Any], node: dict[str, Any]) -> str:
        title = _title(str(node.get("prompt") or node["node_id"]))
        thread_id = self.threads.create_thread(
            f"Workflow node: {title}",
            kind="workflow_node",
            parent_thread_id=workflow.get("parent_thread_id"),
            parent_turn_id=workflow.get("parent_turn_id"),
            parent_run_id=workflow.get("parent_run_id"),
        )
        self.threads.record_event(thread_id, "thread.workflow_node_bound", workflow_id=workflow["workflow_id"], node_id=node["node_id"])
        with connect_workflow_db(self.data_dir) as db:
            db.execute("UPDATE workflow_nodes SET thread_id = ? WHERE node_id = ?", (thread_id, node["node_id"]))
        logger.info("Workflow node thread created workflow_id=%s node_id=%s thread_id=%s", workflow["workflow_id"], node["node_id"], thread_id)
        return thread_id

    async def _heartbeat(self, node_id: str) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval_s)
            with connect_workflow_db(self.data_dir) as db:
                db.execute(
                    "UPDATE workflow_nodes SET lease_until = ? WHERE node_id = ? AND executor_id = ? AND status = 'running'",
                    (_lease_deadline(self.lease_seconds), node_id, self.executor_id),
                )

    def _complete_if_done(self, workflow_id: str) -> None:
        nodes = self._nodes(workflow_id)
        if not nodes or any(node["status"] in {"pending", "running"} for node in nodes):
            return
        now = utc_now_iso()
        status = "failed" if any(node["status"] == "failed" for node in nodes) else "completed"
        with connect_workflow_db(self.data_dir) as db:
            db.execute("UPDATE workflows SET status = ?, updated_at = ? WHERE workflow_id = ? AND status = 'running'", (status, now, workflow_id))
        logger.info("Workflow completed workflow_id=%s status=%s", workflow_id, status)


def _lease_deadline(seconds: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _title(prompt: str) -> str:
    first = prompt.strip().splitlines()[0] if prompt.strip() else "task"
    return first[:77].rstrip() + ("..." if len(first) > 80 else "")
