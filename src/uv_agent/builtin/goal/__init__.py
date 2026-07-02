from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uv_agent.ids import new_id
from uv_agent.plugins import (
    CommandResult,
    PluginManifest,
    SetupPlugin,
    SetComposerAction,
    TranscriptAction,
)
from uv_agent.time import utc_now_iso
from .i18n import TEXTS


PLUGIN_ID = "builtin.goal"

MANIFEST = PluginManifest(
    id=PLUGIN_ID,
    version="0.1.0",
    display_name={"zh": "目标模式", "en": "Goal Mode"},
    description={"zh": "为单个线程维护持久目标、任务、阶段和笔记。", "en": "Persistent goal/task tracking for one thread."},
    builtin=True,
    priority=90,
    capabilities=("runtime_namespace", "context", "command", "ui", "storage"),
    storage_schema={"collections": {"tasks": {"indexes": ["status"]}}},
)

GOAL_HELP = "用法: /goal enable [objective] | disable | reset | status"
GOAL_HELPER_SIGNATURE = """GoalTask = {id: str, text: str, status: str, priority: Any | None, created_at: str, updated_at: str}
GoalPhase = {name: str, summary: str, updated_at: str}
GoalState = {enabled: bool, objective: str, phase: GoalPhase | dict[str, Any], notes: str, tasks: list[GoalTask], created_at: str, updated_at: str}
rt.goal.state(*, thread_id: str | None = None) -> GoalState
rt.goal.set_objective(*, text: str | None = None, objective: str | None = None, thread_id: str | None = None) -> GoalState
rt.goal.list_tasks(*, status: str | None = None, thread_id: str | None = None) -> list[GoalTask]
rt.goal.add_task(*, text: str, status: str = "todo", priority: Any | None = None, thread_id: str | None = None) -> GoalTask
rt.goal.update_task(*, task_id: str, text: str | None = None, status: str | None = None, priority: Any | None = None, thread_id: str | None = None) -> GoalTask
rt.goal.delete_task(*, task_id: str, thread_id: str | None = None) -> dict[str, Any]
rt.goal.get_phase(*, thread_id: str | None = None) -> GoalPhase | dict[str, Any]
rt.goal.set_phase(*, name: str = "", summary: str = "", thread_id: str | None = None) -> GoalPhase
rt.goal.get_notes(*, thread_id: str | None = None) -> str
rt.goal.set_notes(*, text: str | None = None, notes: str | None = None, thread_id: str | None = None) -> dict[str, str]
rt.goal.append_note(*, text: str, thread_id: str | None = None) -> dict[str, str]"""


@dataclass(frozen=True)
class GoalRuntimeContext:
    storage: Any


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


def setup(context) -> None:
    context.i18n.register(TEXTS)
    runtime = GoalRuntimeContext(storage=context.storage)
    context.runtime.register_namespace(
        "goal",
        doc="管理当前线程的活动目标、任务、阶段和笔记。",
        functions={
            "state": lambda thread_id=None: _state(runtime.storage, _runtime_thread_id(thread_id), {}),
            "set_objective": lambda text=None, objective=None, thread_id=None: _set_objective(
                runtime.storage,
                _runtime_thread_id(thread_id),
                {"text": text, "objective": objective},
            ),
            "list_tasks": lambda status=None, thread_id=None: _list_tasks(
                runtime.storage,
                _runtime_thread_id(thread_id),
                {"status": status},
            ),
            "add_task": lambda text, status="todo", priority=None, thread_id=None: _add_task(
                runtime.storage,
                _runtime_thread_id(thread_id),
                {"text": text, "status": status, "priority": priority},
            ),
            "update_task": lambda task_id, text=None, status=None, priority=None, thread_id=None: _runtime_update_task(
                runtime.storage,
                _runtime_thread_id(thread_id),
                task_id=task_id,
                text=text,
                status=status,
                priority=priority,
            ),
            "delete_task": lambda task_id, thread_id=None: _delete_task(
                runtime.storage,
                _runtime_thread_id(thread_id),
                {"task_id": task_id},
            ),
            "get_phase": lambda thread_id=None: _get_phase(runtime.storage, _runtime_thread_id(thread_id), {}),
            "set_phase": lambda name="", summary="", thread_id=None: _set_phase(
                runtime.storage,
                _runtime_thread_id(thread_id),
                {"name": name, "summary": summary},
            ),
            "get_notes": lambda thread_id=None: _get_notes(runtime.storage, _runtime_thread_id(thread_id), {}),
            "set_notes": lambda text=None, notes=None, thread_id=None: _set_notes(
                runtime.storage,
                _runtime_thread_id(thread_id),
                {"text": text, "notes": notes},
            ),
            "append_note": lambda text, thread_id=None: _append_note(
                runtime.storage,
                _runtime_thread_id(thread_id),
                {"text": text},
            ),
        },
        docs={
            "state": "返回当前线程的 goal 状态。",
            "set_objective": "更新当前目标描述。",
            "list_tasks": "列出 goal 任务，可按 status 过滤。",
            "add_task": "新增一个 goal 任务。",
            "update_task": "按 task_id 更新 goal 任务。",
            "delete_task": "按 task_id 删除 goal 任务。",
            "get_phase": "返回当前 goal 阶段。",
            "set_phase": "更新当前 goal 阶段。",
            "get_notes": "返回 goal 笔记。",
            "set_notes": "替换 goal 笔记。",
            "append_note": "追加带时间戳的 goal 笔记。",
        },
        schemas={name: {"type": "object"} for name in (
            "state", "set_objective", "list_tasks", "add_task", "update_task", "delete_task",
            "get_phase", "set_phase", "get_notes", "set_notes", "append_note",
        )},
    )
    context.commands.register(
        "/goal",
        _goal_command,
        description={"zh": "启用、禁用、重置或查看目标模式", "en": "enable, disable, reset, or show goal mode"},
    )
    context.ui.picker(
        id="goal.commands",
        title={"zh": "目标命令", "en": "Goal commands"},
        provider=lambda query="": _goal_picker_items(query),
        trigger="/goal",
    )
    _publish_goal_helpers(context)
    context.epoch.on_refresh(lambda thread_id=None: _publish_goal_helpers(context))


def _publish_goal_helpers(context) -> None:
    context.epoch.publish(
        tag="goal_helpers",
        body={
            "instructions": [
                "当线程启用 goal mode 时，它由 builtin.goal 插件管理。",
                "使用 rt.goal.* helpers 管理 objective、tasks、phase 和 notes；不要直接编辑插件存储。",
                "有实质进展后，更新 tasks/phase/notes，让后续轮次和压缩后恢复都能保留状态。",
            ],
            "helper": {
                "import": "import uv_agent_runtime as rt",
                "namespace": "rt.goal",
                "signature": GOAL_HELPER_SIGNATURE,
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
        return CommandResult((TranscriptAction("error", "/goal 不可用：builtin.goal 尚未启动"),))
    if not thread_id:
        if op == "enable":
            objective = rest.strip()
            return CommandResult((
                TranscriptAction(
                    "event",
                    f"下一条消息将启用 goal mode · 目标: {objective or '—'}",
                    metadata={"goal_pending": True, "goal_enabled": True, "goal_objective": objective},
                ),
                SetComposerAction(""),
            ))
        if op == "disable":
            return CommandResult((
                TranscriptAction(
                    "event",
                    "goal mode 已禁用",
                    metadata={"goal_pending": False, "goal_enabled": False, "goal_objective": ""},
                ),
                SetComposerAction(""),
            ))
        if op == "status":
            return CommandResult((TranscriptAction("event", "goal mode: 已禁用（没有活动线程）"),))
        return CommandResult((TranscriptAction("error", "/goal reset 需要活动线程，请先发送一条消息"),))
    try:
        if op == "enable":
            state = enable_thread_goal(context, thread_id, objective=rest)
            return CommandResult((TranscriptAction(
                "event",
                f"goal mode 已启用 · 目标: {state['objective'] or '—'}",
                metadata={"goal_pending": False, "goal_enabled": True, "goal_objective": state.get("objective") or ""},
            ),))
        if op == "disable":
            disable_thread_goal(context, thread_id)
            return CommandResult((TranscriptAction(
                "event",
                "goal mode 已禁用",
                metadata={"goal_pending": False, "goal_enabled": False},
            ),))
        if op == "reset":
            reset_thread_goal(context, thread_id, objective=rest)
            return CommandResult((TranscriptAction(
                "event",
                "goal 状态已重置",
                metadata={"goal_pending": False, "goal_enabled": False, "goal_objective": rest.strip()},
            ),))
        state = _state(context.storage, thread_id, {})
        status = "已启用" if state.get("enabled") else "已禁用"
        return CommandResult((TranscriptAction("event", f"goal mode: {status}\n目标: {state.get('objective') or '—'}"),))
    except Exception as exc:
        return CommandResult((TranscriptAction("error", f"/goal {op} 失败: {exc}"),))


def _goal_picker_items(query: str = "") -> list[dict[str, str]]:
    commands = [
        ("/goal enable", {"zh": "启用 goal mode，可附带目标", "en": "enable goal mode with optional objective"}),
        ("/goal disable", {"zh": "禁用 goal mode", "en": "disable goal mode"}),
        ("/goal reset", {"zh": "重置插件存储的 goal 状态", "en": "reset plugin-stored goal state"}),
        ("/goal status", {"zh": "查看当前 goal 状态", "en": "show current goal state"}),
    ]
    needle = str(query or "").lower()
    return [
        {"value": value, "description": description, "kind": "plugin-command"}
        for value, description in commands
        if not needle or needle in value.lower() or any(needle in text.lower() for text in description.values())
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
    _record_goal_mode_event(context, thread_id, state)
    _publish_goal_notice(context, thread_id, state)
    return state


def disable_thread_goal(context, thread_id: str) -> dict[str, Any]:
    kv = context.storage.thread_kv(thread_id)
    state = _state(context.storage, thread_id, {})
    state.update({"enabled": False, "updated_at": utc_now_iso()})
    kv.set("state", state)
    context.epoch.remove(tag="goal_mode", reason="此线程的 goal mode 已禁用。", thread_id=thread_id)
    context.turn.clear_replay(thread_id=thread_id, replay_key="goal_state")
    _record_goal_mode_event(context, thread_id, state)
    return state


def reset_thread_goal(context, thread_id: str, *, objective: str = "") -> dict[str, Any]:
    kv = context.storage.thread_kv(thread_id)
    now = utc_now_iso()
    kv.set("state", {"enabled": False, "objective": objective.strip(), "phase": {}, "notes": "", "created_at": now, "updated_at": now})
    for task in context.storage.thread_collection(thread_id, "tasks").list(limit=500):
        context.storage.thread_collection(thread_id, "tasks").delete(task["doc_id"])
    context.epoch.remove(tag="goal_mode", reason="goal 状态已在禁用状态下重置。", thread_id=thread_id)
    state = _state(context.storage, thread_id, {})
    _record_goal_reset_event(context, thread_id, state)
    return state


def _runtime_thread_id(value: Any = None) -> str:
    thread_id = str(value or "").strip()
    if thread_id:
        return thread_id
    import os

    thread_id = os.environ.get("UV_AGENT_RUNTIME_THREAD_ID", "").strip()
    if not thread_id:
        raise RuntimeError("rt.goal helpers require an active uv-agent thread")
    return thread_id


def _runtime_update_task(
    storage,
    thread_id: str,
    *,
    task_id: str,
    text: Any = None,
    status: Any = None,
    priority: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"task_id": task_id}
    if text is not None:
        payload["text"] = text
    if status is not None:
        payload["status"] = status
    if priority is not None:
        payload["priority"] = priority
    return _update_task(storage, thread_id, payload)


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
            "使用 rt.goal.* helpers 持续维护 objective、tasks、phase 和 notes。",
            "有实质进展或交接前，更新 tasks/phase/notes。",
            "不要直接编辑插件存储。",
        ],
        "summary": {
            "phase": state.get("phase") or {},
            "task_count": len(state.get("tasks") or []),
            "has_notes": bool(state.get("notes")),
        },
    }
    context.turn.enqueue(
        thread_id=thread_id,
        tag="goal_mode",
        body=body,
        attrs={"status": "enabled"},
        replay_after_compaction=True,
        replay_key="goal_state",
    )


def _record_goal_mode_event(context, thread_id: str, state: dict[str, Any]) -> None:
    event = context.threads.record_event(
        thread_id,
        "thread.goal_mode_updated",
        enabled=bool(state.get("enabled")),
        objective=str(state.get("objective") or ""),
    )
    context.threads.update_metadata(thread_id, {
        "goal_mode": {
            "enabled": bool(state.get("enabled")),
            "status": "enabled" if state.get("enabled") else "disabled",
            "updated_at": event.get("created_at"),
            "objective": str(state.get("objective") or ""),
            "_event_id": event.get("_event_id"),
        }
    })


def _record_goal_reset_event(context, thread_id: str, state: dict[str, Any]) -> None:
    event = context.threads.record_event(
        thread_id,
        "thread.goal_state_reset",
        objective=str(state.get("objective") or ""),
    )
    context.threads.update_metadata(thread_id, {
        "goal_mode": {
            "enabled": False,
            "status": "disabled",
            "updated_at": event.get("created_at"),
            "reset_at": event.get("created_at"),
            "objective": str(state.get("objective") or ""),
            "_reset_event_id": event.get("_event_id"),
        }
    })
