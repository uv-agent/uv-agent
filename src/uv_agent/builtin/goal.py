from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uv_agent.ids import new_id
from uv_agent.plugins import (
    CommandResult,
    OpenPickerAction,
    PluginManifest,
    SetupPlugin,
    SetComposerAction,
    TranscriptAction,
)
from uv_agent.time import utc_now_iso


PLUGIN_ID = "builtin.goal"

MANIFEST = PluginManifest(
    id=PLUGIN_ID,
    version="0.1.0",
    display_name="Goal Mode",
    description="Persistent goal/task tracking for one thread.",
    builtin=True,
    priority=90,
    capabilities=("runtime_namespace", "context", "command", "ui", "storage"),
    storage_schema={"collections": {"tasks": {"indexes": ["status"]}}},
)

GOAL_HELP = "usage: /goal enable [objective] | disable | reset | status"


@dataclass(frozen=True)
class GoalRuntimeContext:
    storage: Any
    context: Any


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


def setup(context) -> None:
    runtime = GoalRuntimeContext(storage=context.storage, context=context.context)
    context.runtime.register_namespace(
        "goal",
        doc="Manage this thread's active goal, tasks, phase, and notes.",
        functions={
            "state": lambda payload: _runtime_thread_call(runtime, payload, _state),
            "set_objective": lambda payload: _runtime_thread_call(runtime, payload, _set_objective),
            "list_tasks": lambda payload: _runtime_thread_call(runtime, payload, _list_tasks),
            "add_task": lambda payload: _runtime_thread_call(runtime, payload, _add_task),
            "update_task": lambda payload: _runtime_thread_call(runtime, payload, _update_task),
            "delete_task": lambda payload: _runtime_thread_call(runtime, payload, _delete_task),
            "get_phase": lambda payload: _runtime_thread_call(runtime, payload, _get_phase),
            "set_phase": lambda payload: _runtime_thread_call(runtime, payload, _set_phase),
            "get_notes": lambda payload: _runtime_thread_call(runtime, payload, _get_notes),
            "set_notes": lambda payload: _runtime_thread_call(runtime, payload, _set_notes),
            "append_note": lambda payload: _runtime_thread_call(runtime, payload, _append_note),
        },
        docs={
            "state": "Return current goal state for the active thread.",
            "set_objective": "Update the active goal objective.",
            "list_tasks": "List goal tasks, optionally filtered by status.",
            "add_task": "Add a goal task.",
            "update_task": "Update a goal task by task_id.",
            "delete_task": "Delete a goal task by task_id.",
            "get_phase": "Return the current goal phase.",
            "set_phase": "Update the current goal phase.",
            "get_notes": "Return goal notes.",
            "set_notes": "Replace goal notes.",
            "append_note": "Append a timestamped note.",
        },
        schemas={name: {"type": "object"} for name in (
            "state", "set_objective", "list_tasks", "add_task", "update_task", "delete_task",
            "get_phase", "set_phase", "get_notes", "set_notes", "append_note",
        )},
    )
    context.commands.register("/goal", _goal_command, description="enable, disable, reset, or show goal mode")
    context.ui.picker(
        id="goal.commands",
        title="Goal commands",
        provider=lambda query="": _goal_picker_items(query),
        trigger="/goal",
    )
    context.context.epoch.publish(
        tag="goal_helpers",
        body={
            "instructions": [
                "Goal mode is managed by the builtin.goal plugin when enabled for a thread.",
                "Use rt.goal.* helpers to manage objective, tasks, phase, and notes; do not edit plugin storage directly.",
                "After meaningful progress, update tasks/phase/notes so future turns and compactions preserve state.",
            ],
            "helper": {
                "import": "import uv_agent_runtime as rt",
                "namespace": "rt.goal",
                "functions": [
                    "state", "set_objective", "list_tasks", "add_task", "update_task", "delete_task",
                    "get_phase", "set_phase", "get_notes", "set_notes", "append_note",
                ],
            },
        },
    )


def _goal_command(payload: dict[str, Any], context=None) -> CommandResult:
    text = str(payload.get("arg") or "").strip()
    parts = text.split(None, 1)
    op = (parts[0] if parts else "status").lower()
    rest = parts[1] if len(parts) > 1 else ""
    thread_id = str(payload.get("thread_id") or "").strip()
    if op not in {"enable", "disable", "reset", "status"}:
        return CommandResult((TranscriptAction("error", GOAL_HELP),))
    if context is None:
        return CommandResult((TranscriptAction("error", "/goal unavailable: builtin.goal is not started"),))
    if not thread_id:
        if op == "enable":
            objective = rest.strip()
            return CommandResult((
                TranscriptAction(
                    "event",
                    f"goal mode enabled for next message · objective: {objective or '—'}",
                    metadata={"goal_pending": True, "goal_enabled": True, "goal_objective": objective},
                ),
                SetComposerAction(""),
            ))
        if op == "disable":
            return CommandResult((
                TranscriptAction(
                    "event",
                    "goal mode disabled",
                    metadata={"goal_pending": False, "goal_enabled": False, "goal_objective": ""},
                ),
                SetComposerAction(""),
            ))
        if op == "status":
            return CommandResult((TranscriptAction("event", "goal mode: disabled (no active thread)"),))
        return CommandResult((TranscriptAction("error", "/goal reset requires an active thread — send a message first"),))
    try:
        if op == "enable":
            state = enable_thread_goal(context, thread_id, objective=rest)
            return CommandResult((TranscriptAction(
                "event",
                f"goal mode enabled · objective: {state['objective'] or '—'}",
                metadata={"goal_pending": False, "goal_enabled": True, "goal_objective": state.get("objective") or ""},
            ),))
        if op == "disable":
            disable_thread_goal(context, thread_id)
            return CommandResult((TranscriptAction(
                "event",
                "goal mode disabled",
                metadata={"goal_pending": False, "goal_enabled": False},
            ),))
        if op == "reset":
            reset_thread_goal(context, thread_id, objective=rest)
            return CommandResult((TranscriptAction(
                "event",
                "goal state reset",
                metadata={"goal_pending": False, "goal_enabled": False, "goal_objective": rest.strip()},
            ),))
        state = _state(context.storage, thread_id, {})
        status = "enabled" if state.get("enabled") else "disabled"
        return CommandResult((TranscriptAction("event", f"goal mode: {status}\nobjective: {state.get('objective') or '—'}"),))
    except Exception as exc:
        return CommandResult((TranscriptAction("error", f"/goal {op} failed: {exc}"),))


def _goal_picker_items(query: str = "") -> list[dict[str, str]]:
    commands = [
        ("/goal enable", "enable goal mode with optional objective"),
        ("/goal disable", "disable goal mode"),
        ("/goal reset", "reset plugin-stored goal state"),
        ("/goal status", "show current goal state"),
    ]
    needle = str(query or "").lower()
    return [
        {"value": value, "description": description, "kind": "plugin-command"}
        for value, description in commands
        if not needle or needle in value.lower() or needle in description.lower()
    ]


def enable_thread_goal(context, thread_id: str, *, objective: str = "") -> dict[str, Any]:
    kv = context.storage.thread_kv(thread_id)
    existing = _state(context.storage, thread_id, {})
    resolved_objective = objective.strip() or str(existing.get("objective") or "")
    now = utc_now_iso()
    kv.set("state", {
        "enabled": True,
        "objective": resolved_objective,
        "phase": existing.get("phase") or {},
        "notes": existing.get("notes") or "",
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
    })
    state = _state(context.storage, thread_id, {})
    _publish_goal_notice(context, thread_id, state)
    return state


def disable_thread_goal(context, thread_id: str) -> dict[str, Any]:
    kv = context.storage.thread_kv(thread_id)
    state = _state(context.storage, thread_id, {})
    state.update({"enabled": False, "updated_at": utc_now_iso()})
    kv.set("state", state)
    context.context.epoch.remove(tag="goal_mode", reason="Goal mode disabled for this thread.", thread_id=thread_id)
    context.context.turn.clear_replay(thread_id=thread_id, replay_key="goal_state")
    return state


def reset_thread_goal(context, thread_id: str, *, objective: str = "") -> dict[str, Any]:
    kv = context.storage.thread_kv(thread_id)
    now = utc_now_iso()
    kv.set("state", {"enabled": False, "objective": objective.strip(), "phase": {}, "notes": "", "created_at": now, "updated_at": now})
    for task in context.storage.thread_collection(thread_id, "tasks").list(limit=500):
        context.storage.thread_collection(thread_id, "tasks").delete(task["doc_id"])
    context.context.epoch.remove(tag="goal_mode", reason="Goal state reset while disabled.", thread_id=thread_id)
    return _state(context.storage, thread_id, {})


def _runtime_thread_call(runtime: GoalRuntimeContext, payload: dict[str, Any], fn):
    thread_id = _thread_id(payload)
    return fn(runtime.storage, thread_id, payload)


def _thread_id(payload: dict[str, Any]) -> str:
    thread_id = str(payload.get("thread_id") or "").strip()
    if thread_id:
        return thread_id
    import os

    thread_id = os.environ.get("UV_AGENT_RUNTIME_THREAD_ID", "").strip()
    if not thread_id:
        raise RuntimeError("rt.goal helpers require an active uv-agent thread")
    return thread_id


def _state(storage, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    raw = storage.thread_kv(thread_id).get("state", {})
    state = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(state.get("enabled")),
        "objective": str(state.get("objective") or ""),
        "phase": state.get("phase") if isinstance(state.get("phase"), dict) else {},
        "notes": str(state.get("notes") or ""),
        "tasks": [item["body"] for item in storage.thread_collection(thread_id, "tasks").list(limit=500)],
        "created_at": str(state.get("created_at") or ""),
        "updated_at": str(state.get("updated_at") or ""),
    }


def _set_objective(storage, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    state = _state(storage, thread_id, payload)
    state["objective"] = str(payload.get("text") or payload.get("objective") or "")
    state["updated_at"] = utc_now_iso()
    storage.thread_kv(thread_id).set("state", {k: v for k, v in state.items() if k != "tasks"})
    return state


def _list_tasks(storage, thread_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    status = payload.get("status")
    collection = storage.thread_collection(thread_id, "tasks")
    if isinstance(status, str) and status:
        return [item["body"] for item in collection.query_index("status", status, limit=500)]
    return [item["body"] for item in collection.list(limit=500)]


def _add_task(storage, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("task text is required")
    now = utc_now_iso()
    task = {
        "id": new_id("goal_task"),
        "text": text,
        "status": str(payload.get("status") or "todo"),
        "priority": payload.get("priority"),
        "created_at": now,
        "updated_at": now,
    }
    storage.thread_collection(thread_id, "tasks").put(task["id"], task)
    return task


def _update_task(storage, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload.get("task_id") or payload.get("id") or "").strip()
    if not task_id:
        raise ValueError("task_id is required")
    collection = storage.thread_collection(thread_id, "tasks")
    task = collection.get(task_id)
    if not isinstance(task, dict):
        raise LookupError(f"Unknown task: {task_id}")
    for key in ("text", "status", "priority"):
        if key in payload:
            task[key] = payload[key]
    task["updated_at"] = utc_now_iso()
    collection.put(task_id, task)
    return task


def _delete_task(storage, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload.get("task_id") or payload.get("id") or "").strip()
    if not task_id:
        raise ValueError("task_id is required")
    return storage.thread_collection(thread_id, "tasks").delete(task_id)


def _get_phase(storage, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _state(storage, thread_id, payload)["phase"]


def _set_phase(storage, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    phase = {"name": str(payload.get("name") or ""), "summary": str(payload.get("summary") or ""), "updated_at": utc_now_iso()}
    state = _state(storage, thread_id, payload)
    state["phase"] = phase
    state["updated_at"] = phase["updated_at"]
    storage.thread_kv(thread_id).set("state", {k: v for k, v in state.items() if k != "tasks"})
    return phase


def _get_notes(storage, thread_id: str, payload: dict[str, Any]) -> str:
    return _state(storage, thread_id, payload)["notes"]


def _set_notes(storage, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    state = _state(storage, thread_id, payload)
    state["notes"] = str(payload.get("text") or payload.get("notes") or "")
    state["updated_at"] = utc_now_iso()
    storage.thread_kv(thread_id).set("state", {k: v for k, v in state.items() if k != "tasks"})
    return {"updated_at": state["updated_at"]}


def _append_note(storage, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    state = _state(storage, thread_id, payload)
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("note text is required")
    addition = f"- {utc_now_iso()}: {text}"
    state["notes"] = (state["notes"].rstrip() + "\n" + addition).strip()
    state["updated_at"] = utc_now_iso()
    storage.thread_kv(thread_id).set("state", {k: v for k, v in state.items() if k != "tasks"})
    return {"updated_at": state["updated_at"]}


def _publish_goal_notice(context, thread_id: str, state: dict[str, Any]) -> None:
    body = {
        "status": "enabled",
        "objective": state.get("objective") or "",
        "instructions": [
            "Use rt.goal.* helpers to keep objective, tasks, phase, and notes current.",
            "Update tasks/phase/notes after meaningful progress or before handing off.",
            "Do not edit plugin storage directly.",
        ],
        "summary": {
            "phase": state.get("phase") or {},
            "task_count": len(state.get("tasks") or []),
            "has_notes": bool(state.get("notes")),
        },
    }
    context.context.epoch.publish(tag="goal_mode", body=body, attrs={"status": "enabled"}, thread_id=thread_id)
    context.context.turn.enqueue(
        thread_id=thread_id,
        tag="goal_mode",
        body=body,
        attrs={"status": "enabled"},
        replay_after_compaction=True,
        replay_key="goal_state",
    )
