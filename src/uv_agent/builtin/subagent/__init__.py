from __future__ import annotations

import asyncio
from typing import Any

from uv_agent.plugins import PluginManifest, SetupPlugin

SUBAGENT_HELPER_SIGNATURE = """SubagentResult = {status: str, thread_id: str | None, turn_id: str | None, final_text: str, error: dict[str, Any] | None, timed_out: bool}
rt.subagent.run(prompt: str, *, level: str | None = None, timeout_s: float | None = None, thread_id: str | None = None, title: str | None = None) -> SubagentResult"""

MANIFEST = PluginManifest(
    id="builtin.subagent",
    version="0.1.0",
    display_name={"zh": "子代理", "en": "Subagent"},
    description={
        "zh": "一次性子代理线程和计划任务 prompt action。",
        "en": "One-shot subagent threads and scheduled prompt actions.",
    },
    builtin=True,
    priority=300,
    capabilities=("runtime_namespace", "action", "context"),
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


def setup(context) -> None:
    plugin_context = context

    async def run(
        prompt: str,
        *,
        level: str | None = None,
        timeout_s: float | None = None,
        thread_id: str | None = None,
        title: str | None = None,
        context=None,
    ) -> dict[str, Any]:
        parent_thread_id = str(getattr(context, "thread_id", "") or "") or None
        _raise_if_nested_subagent(plugin_context, parent_thread_id)
        return await _run_subagent(
            submit_turn=plugin_context.submit_turn,
            threads=plugin_context.threads,
            prompt=prompt,
            level=level,
            timeout_s=timeout_s,
            thread_id=thread_id,
            title=title,
            parent_thread_id=parent_thread_id,
            parent_turn_id=str(getattr(context, "turn_id", "") or "") or None,
            parent_run_id=str(getattr(context, "run_id", "") or "") or None,
            owner_plugin=plugin_context.plugin_id,
        )

    context.runtime.register_namespace(
        "subagent",
        doc="运行一次性子代理线程，并返回子线程 turn 的结果。",
        functions={"run": run},
        docs={
            "run": "创建或复用一个子代理线程，提交 prompt，并等待该 turn 完成或超时。"
        },
        schemas={
            "run": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "level": {"type": ["string", "null"]},
                    "timeout_s": {"type": ["number", "null"]},
                    "thread_id": {"type": ["string", "null"]},
                    "title": {"type": ["string", "null"]},
                },
            }
        },
    )
    context.actions.register(
        "subagent.prompt",
        _scheduled_prompt,
        doc="从计划任务 action 在线程中提交一次 prompt，并返回该 turn 的结果。",
        schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "thread_id": {"type": ["string", "null"]},
                "level": {"type": ["string", "null"]},
                "model_level": {"type": ["string", "null"]},
                "timeout_s": {"type": ["number", "null"]},
            },
            "required": ["prompt"],
        },
    )
    _publish_subagent_context(context)
    context.epoch.on_refresh(lambda thread_id=None: _publish_subagent_context(context))


def _publish_subagent_context(context) -> None:
    context.epoch.publish(
        tag="subagent_helpers",
        attrs={"scope": "main_agent"},
        body={
            "instructions": [
                "使用 rt.subagent.run(...) 启动一次性子代理线程；子代理适合调查、验证或独立处理一段任务。",
                "level 如果用户没有提及，就不用填写；不传时使用 uv-agent 默认 level。",
                '计划任务请使用 action_id="subagent.prompt"，payload 中传 prompt/level/timeout_s。',
                "子代理只返回结果摘要和 thread_id；需要完整对话时再用 rt.threads helpers 查看对应线程。",
            ],
            "helper": {
                "import": "import uv_agent_runtime as rt",
                "namespace": "rt.subagent",
                "signature": SUBAGENT_HELPER_SIGNATURE,
            },
        },
    )


async def _scheduled_prompt(payload: dict[str, Any], context=None) -> dict[str, Any]:
    if context is None:
        raise RuntimeError("subagent.prompt requires scheduler action context")
    prompt = _required_prompt(payload)
    level_value = (
        payload.get("level")
        if payload.get("level") is not None
        else payload.get("model_level")
    )
    level = _normalize_level(level_value)
    thread_id = _normalize_optional_str(payload.get("thread_id"))
    if thread_id is None and hasattr(context, "schedule_thread"):
        thread_id = context.schedule_thread()
    timeout_s = _normalize_timeout(payload.get("timeout_s"))
    submit_turn = getattr(context, "submit_turn", None)
    if not callable(submit_turn):
        raise RuntimeError("scheduler action context cannot submit turns")
    submitted = await submit_turn(
        text=prompt, thread_id=thread_id, level=level, conflict="queue"
    )
    return await _wait_for_submitted(submitted, timeout_s=timeout_s)


async def _run_subagent(
    *,
    submit_turn,
    threads,
    prompt: str,
    level: str | None,
    timeout_s: float | None,
    thread_id: str | None,
    title: str | None,
    parent_thread_id: str | None,
    parent_turn_id: str | None,
    parent_run_id: str | None,
    owner_plugin: str,
) -> dict[str, Any]:
    prompt = _required_prompt({"prompt": prompt})
    level = _normalize_level(level)
    thread_id = _normalize_optional_str(thread_id)
    timeout_s = _normalize_timeout(timeout_s)
    if thread_id is None and threads is not None:
        thread_id = threads.create_thread(
            _thread_title(prompt, title),
            kind="subagent",
            parent_thread_id=parent_thread_id,
            parent_turn_id=parent_turn_id,
            parent_run_id=parent_run_id,
        )
        threads.update_metadata(
            thread_id,
            {
                "owner_type": "plugin",
                "owner_plugin": owner_plugin,
                "subagent": True,
            },
        )
    submitted = await submit_turn(
        text=prompt, thread_id=thread_id, level=level, conflict="queue"
    )
    return await _wait_for_submitted(submitted, timeout_s=timeout_s)


async def _wait_for_submitted(
    submitted: Any, *, timeout_s: float | None
) -> dict[str, Any]:
    timed_out = False
    try:
        if timeout_s is None:
            completed = await submitted.wait()
        else:
            completed = await asyncio.wait_for(submitted.wait(), timeout=timeout_s)
    except TimeoutError:
        completed = submitted
        timed_out = True
    except asyncio.TimeoutError:
        completed = submitted
        timed_out = True
    status = (
        "timeout" if timed_out else str(getattr(completed, "status", "") or "unknown")
    )
    return {
        "status": status,
        "thread_id": _normalize_optional_str(getattr(completed, "thread_id", None)),
        "turn_id": _normalize_optional_str(getattr(completed, "turn_id", None)),
        "final_text": str(getattr(completed, "final_text", "") or ""),
        "error": _error_payload(getattr(completed, "error", None)),
        "timed_out": timed_out,
    }


def _raise_if_nested_subagent(
    plugin_context: Any, parent_thread_id: str | None
) -> None:
    if not parent_thread_id or plugin_context.threads is None:
        return
    try:
        kind = str(
            plugin_context.threads.metadata(parent_thread_id).get("kind") or "thread"
        )
    except Exception:
        return
    if kind != "thread":
        raise RuntimeError(
            "rt.subagent.run is only available from main Agent threads, not inside child agent threads"
        )


def _required_prompt(payload: dict[str, Any]) -> str:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    return prompt


def _normalize_level(value: Any) -> str | None:
    return _normalize_optional_str(value)


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_timeout(value: Any) -> float | None:
    if value is None or value == "":
        return None
    timeout_s = float(value)
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    return timeout_s


def _thread_title(prompt: str, title: str | None) -> str:
    raw = _normalize_optional_str(title) or prompt.splitlines()[0].strip() or "task"
    if len(raw) > 80:
        raw = raw[:77].rstrip() + "..."
    return f"Subagent: {raw}"


def _error_payload(error: Any) -> dict[str, Any] | None:
    if error is None:
        return None
    return {"type": error.__class__.__name__, "message": str(error) or repr(error)}
