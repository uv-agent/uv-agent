from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

from uv_agent.plugins import PluginManifest, SetupPlugin
from uv_agent.time import utc_now_iso
from .operations import (
    CommandResult,
    cleanup_worktree,
    create_worktree,
    validate_worktree_branch_name,
)
from .i18n import TEXTS


MANIFEST = PluginManifest(
    id="builtin.worktree",
    version="0.1.0",
    display_name={"zh": "Worktree 上下文", "en": "Worktree Context"},
    description={"zh": "为线程级 Git worktree 发布模型可见上下文。", "en": "Publishes model-visible context for thread-scoped Git worktrees."},
    builtin=True,
    priority=95,
    capabilities=("runtime_namespace", "context", "action"),
)

WORKTREE_HELPER_SIGNATURE = """WorktreeState = {active: bool, thread_id: str, status: str, branch: str, path: str, origin: str, current_cwd: str, metadata: dict[str, Any]}
rt.worktree.current(*, thread_id: str | None = None) -> WorktreeState
rt.worktree.validate_branch(branch: str) -> str
rt.worktree.create(branch: str, *, base_ref: str = "HEAD", thread_id: str | None = None, project_root: str | None = None) -> dict[str, Any]
rt.worktree.cleanup(branch: str | None = None, *, path: str | None = None, thread_id: str | None = None, project_root: str | None = None) -> dict[str, Any]"""


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


def setup(context) -> None:
    context.i18n.register(TEXTS)
    context.runtime.register_namespace(
        "worktree",
        doc="检查、创建和清理线程级 Git worktree。",
        module="uv_agent.builtin.worktree.runtime",
        functions=_runtime_functions(context),
        docs={
            "current": "返回当前线程或指定线程的 worktree metadata。",
            "validate_branch": "校验候选 worktree 分支名。",
            "create": "创建项目内 Git worktree，并在有 thread_id 时绑定到线程。",
            "cleanup": "清理项目内 Git worktree，并在有 thread_id 时更新线程状态。",
        },
        schemas={name: {"type": "object"} for name in ("current", "validate_branch", "create", "cleanup")},
    )
    context.actions.register(
        "worktree.validate_branch",
        _validate_branch_action,
        doc="Validate a candidate Git worktree branch name.",
        schema={"type": "object", "properties": {"branch": {"type": "string"}}, "required": ["branch"]},
    )
    context.actions.register(
        "worktree.create",
        _create_worktree_action,
        doc="Create a project-local Git worktree and return thread metadata.",
        schema={
            "type": "object",
            "properties": {
                "project_root": {"type": ["string", "null"]},
                "thread_id": {"type": ["string", "null"]},
                "branch": {"type": "string"},
                "base_ref": {"type": ["string", "null"]},
            },
            "required": ["branch"],
        },
    )
    context.actions.register(
        "worktree.cleanup",
        _cleanup_worktree_action,
        doc="Remove a project-local Git worktree and delete its local branch.",
        schema={
            "type": "object",
            "properties": {
                "project_root": {"type": ["string", "null"]},
                "thread_id": {"type": ["string", "null"]},
                "branch": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["branch", "path"],
        },
    )
    if context.threads is not None:
        for metadata in context.threads.list_threads():
            thread_id = str(metadata.get("thread_id") or "").strip()
            if thread_id and str(metadata.get("worktree_status") or "") == "active":
                _publish_worktree_context(context, thread_id, metadata)
    context.epoch.on_refresh(lambda thread_id=None: _refresh_worktree_epoch(context, thread_id))

    def on_thread_event(event: dict[str, Any]) -> None:
        stored = event.get("event") if isinstance(event.get("event"), dict) else {}
        event_type = str(stored.get("type") or "")
        if event_type not in {"thread.worktree_created", "thread.worktree_deleted"}:
            return
        thread_id = str(event.get("thread_id") or stored.get("thread_id") or "").strip()
        if not thread_id or context.threads is None:
            return
        metadata = dict(stored)
        try:
            current = context.threads.metadata(thread_id)
        except FileNotFoundError:
            current = {}
        if event_type == "thread.worktree_deleted":
            metadata = {**current, **metadata, "worktree_status": "deleted"}
        _publish_worktree_context(context, thread_id, metadata, event_type=event_type)

    context.events.subscribe("thread.event_stored", on_thread_event, logger=context.logger)


def _validate_branch_action(payload: dict[str, Any], context=None) -> dict[str, str]:
    del context
    return {"branch": validate_worktree_branch_name(str(payload.get("branch") or ""))}


def _runtime_functions(plugin_context) -> dict[str, Any]:
    return {
        "current": lambda thread_id=None, context=None: _current_worktree(plugin_context, thread_id, context),
        "validate_branch": lambda branch: _validate_branch_action({"branch": branch}),
        "create": lambda branch, base_ref="HEAD", thread_id=None, project_root=None, context=None: _runtime_create_worktree(
            plugin_context,
            branch=branch,
            base_ref=base_ref,
            thread_id=thread_id,
            project_root=project_root,
            run_context=context,
        ),
        "cleanup": lambda branch=None, path=None, thread_id=None, project_root=None, context=None: _runtime_cleanup_worktree(
            plugin_context,
            branch=branch,
            path=path,
            thread_id=thread_id,
            project_root=project_root,
            run_context=context,
        ),
    }


def _current_worktree(plugin_context, thread_id: str | None, run_context=None) -> dict[str, Any]:
    resolved_thread_id = _runtime_thread_id(thread_id, run_context)
    empty = {
        "active": False,
        "thread_id": resolved_thread_id,
        "status": "",
        "branch": "",
        "path": "",
        "origin": "",
        "current_cwd": "",
        "metadata": {},
    }
    if not resolved_thread_id or plugin_context.threads is None:
        return empty
    try:
        metadata = plugin_context.threads.metadata(resolved_thread_id)
    except FileNotFoundError:
        return empty
    status = str(metadata.get("worktree_status") or "").strip()
    path = str(metadata.get("worktree_path") or "").strip()
    origin = str(metadata.get("worktree_origin_root") or "").strip()
    latest_cwd = str(metadata.get("latest_cwd") or "").strip()
    current_cwd = latest_cwd or path
    if status == "deleted":
        current_cwd = latest_cwd if latest_cwd and latest_cwd != path else origin
    return {
        "active": status == "active",
        "thread_id": resolved_thread_id,
        "status": status,
        "branch": str(metadata.get("worktree_branch") or "").strip(),
        "path": path,
        "origin": origin,
        "current_cwd": current_cwd,
        "metadata": dict(metadata),
    }


async def _runtime_create_worktree(
    plugin_context,
    *,
    branch: str,
    base_ref: str = "HEAD",
    thread_id: str | None = None,
    project_root: str | None = None,
    run_context=None,
) -> dict[str, Any]:
    return await _create_worktree_action(
        {
            "project_root": project_root,
            "thread_id": _runtime_thread_id(thread_id, run_context),
            "branch": branch,
            "base_ref": base_ref,
        },
        context=plugin_context,
    )


async def _runtime_cleanup_worktree(
    plugin_context,
    *,
    branch: str | None = None,
    path: str | None = None,
    thread_id: str | None = None,
    project_root: str | None = None,
    run_context=None,
) -> dict[str, Any]:
    resolved_thread_id = _runtime_thread_id(thread_id, run_context)
    if (not branch or not path) and resolved_thread_id and plugin_context.threads is not None:
        try:
            metadata = plugin_context.threads.metadata(resolved_thread_id)
        except FileNotFoundError:
            metadata = {}
        branch = branch or str(metadata.get("worktree_branch") or "").strip()
        path = path or str(metadata.get("worktree_path") or "").strip()
    if not branch or not path:
        raise ValueError("branch and path are required when the current thread has no worktree metadata")
    return await _cleanup_worktree_action(
        {
            "project_root": project_root,
            "thread_id": resolved_thread_id,
            "branch": branch,
            "path": path,
        },
        context=plugin_context,
    )


def _runtime_thread_id(thread_id: str | None, run_context=None) -> str:
    value = str(thread_id or "").strip()
    if value:
        return value
    return str(getattr(run_context, "thread_id", None) or "").strip()


async def _create_worktree_action(payload: dict[str, Any], context=None) -> dict[str, Any]:
    project_root = _project_root(payload, context)
    branch = str(payload.get("branch") or "")
    base_ref = str(payload.get("base_ref") or "HEAD")
    info = await asyncio.to_thread(create_worktree, project_root, branch, run=_run_command, base_ref=base_ref)
    metadata = info.metadata()
    thread_id = str(payload.get("thread_id") or "").strip()
    if thread_id and context is not None and context.threads is not None:
        context.threads.record_event(thread_id, "thread.worktree_created", **metadata)
        context.threads.update_metadata(thread_id, metadata)
        context.threads.record_event(thread_id, "thread.cwd_updated", cwd=str(info.path))
    return {
        "branch": info.branch,
        "path": str(info.path),
        "base_ref": info.base_ref,
        "origin_root": str(info.origin_root),
        "head": info.head,
        "status": info.status,
        "created_at": info.created_at,
        "metadata": metadata,
    }


async def _cleanup_worktree_action(payload: dict[str, Any], context=None) -> dict[str, Any]:
    project_root = _project_root(payload, context)
    result = await asyncio.to_thread(
        cleanup_worktree,
        project_root,
        str(payload.get("branch") or ""),
        Path(str(payload.get("path") or "")),
        run=_run_command,
    )
    thread_id = str(payload.get("thread_id") or "").strip()
    if thread_id and context is not None and context.threads is not None:
        deleted_at = utc_now_iso()
        context.threads.record_event(
            thread_id,
            "thread.worktree_deleted",
            worktree_branch=result.branch,
            worktree_path=str(result.path),
            worktree_origin_root=str(result.origin_root),
            worktree_deleted_at=deleted_at,
            worktree_deleted_head=result.head,
            worktree_deleted_status=result.status,
            worktree_removed=result.worktree_removed,
            branch_deleted=result.branch_deleted,
        )
        context.threads.update_metadata(thread_id, {
            "worktree_status": "deleted",
            "worktree_branch": result.branch,
            "worktree_path": str(result.path),
            "worktree_origin_root": str(result.origin_root),
            "worktree_deleted_at": deleted_at,
            "worktree_deleted_head": result.head,
            "worktree_deleted_status": result.status,
            "worktree_removed": result.worktree_removed,
            "branch_deleted": result.branch_deleted,
        })
        context.threads.record_event(thread_id, "thread.cwd_updated", cwd=str(project_root))
    return {
        "branch": result.branch,
        "path": str(result.path),
        "origin_root": str(result.origin_root),
        "head": result.head,
        "status": result.status,
        "worktree_removed": result.worktree_removed,
        "branch_deleted": result.branch_deleted,
        "worktree_remove_stdout": result.worktree_remove_stdout,
        "worktree_remove_stderr": result.worktree_remove_stderr,
        "branch_delete_stdout": result.branch_delete_stdout,
        "branch_delete_stderr": result.branch_delete_stderr,
    }


def _project_root(payload: dict[str, Any], context) -> Path:
    value = payload.get("project_root")
    if isinstance(value, str) and value.strip():
        return Path(value).resolve()
    if context is not None:
        return Path(context.project_root).resolve()
    return Path.cwd().resolve()


def _run_command(args: list[str], *, cwd: Path, timeout_s: float | None = None) -> CommandResult:
    completed = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    return CommandResult(
        args=list(args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _refresh_worktree_context(context, thread_id: str | None = None) -> None:
    if not thread_id or context.threads is None:
        return
    try:
        metadata = context.threads.metadata(thread_id)
    except FileNotFoundError:
        return
    if str(metadata.get("worktree_status") or "") != "active":
        return
    _publish_worktree_context(context, thread_id, metadata)


def _refresh_worktree_epoch(context, thread_id: str | None = None) -> None:
    _refresh_worktree_context(context, thread_id)


def _publish_worktree_helpers(context, *, thread_id: str) -> None:
    context.epoch.publish(
        tag="worktree_helpers",
        thread_id=thread_id,
        body={
            "instructions": [
                "使用 rt.worktree.current() 检查当前线程是否绑定 worktree。",
                "创建或清理 worktree 时使用 rt.worktree.* helpers；不要直接编辑线程 metadata。",
                "如果当前线程有 active worktree，文件系统、Git、构建和测试默认应在 worktree current_cwd 中执行。",
            ],
            "helper": {
                "import": "import uv_agent_runtime as rt",
                "namespace": "rt.worktree",
                "signature": WORKTREE_HELPER_SIGNATURE,
            },
        },
    )


def _publish_worktree_context(context, thread_id: str, metadata: dict[str, Any], *, event_type: str = "") -> None:
    status = str(metadata.get("worktree_status") or "").strip()
    if status not in {"active", "deleted"}:
        context.epoch.remove(tag="worktree", reason="No active worktree metadata for this thread.", thread_id=thread_id)
        return
    branch = str(metadata.get("worktree_branch") or "").strip()
    path = str(metadata.get("worktree_path") or "").strip()
    origin = str(metadata.get("worktree_origin_root") or "").strip()
    if not branch or not path or not origin:
        return
    body = _worktree_body(metadata, status=status)
    if status == "deleted":
        if event_type == "thread.worktree_deleted":
            context.epoch.update(tag="worktree", body=body, attrs={"status": "deleted"}, thread_id=thread_id)
        context.epoch.remove(tag="worktree_helpers", reason="Worktree mode closed for this thread.", thread_id=thread_id)
        context.epoch.remove(tag="worktree", reason="Worktree mode closed for this thread.", thread_id=thread_id)
        return
    _publish_worktree_helpers(context, thread_id=thread_id)
    context.epoch.publish(tag="worktree", body=body, attrs={"status": "active"}, thread_id=thread_id)


def _worktree_body(metadata: dict[str, Any], *, status: str) -> dict[str, Any]:
    path = _value(metadata, "worktree_path")
    origin = _value(metadata, "worktree_origin_root")
    latest_cwd = _value(metadata, "latest_cwd")
    current_cwd = latest_cwd or path
    if status == "deleted":
        current_cwd = latest_cwd if latest_cwd and latest_cwd != path else origin
    workspace: dict[str, str] = {
        "branch": _value(metadata, "worktree_branch"),
        "path": path,
        "origin": origin,
        "current_cwd": current_cwd,
    }
    for source, target in (
        ("worktree_base_ref", "base_ref"),
        ("worktree_head", "head"),
        ("worktree_created_at", "created_at"),
        ("worktree_deleted_at", "deleted_at"),
        ("worktree_deleted_head", "deleted_head"),
    ):
        value = _value(metadata, source)
        if value:
            workspace[target] = value
    deleted_status = str(metadata.get("worktree_deleted_status") or "")
    if deleted_status:
        workspace["deleted_git_status"] = deleted_status
    if status == "deleted":
        instructions = [
            "此线程之前的 worktree 已移除；不要依赖已删除的路径或分支。",
            "除非用户另有要求，文件系统、Git、构建和测试命令都从 current_cwd 执行。",
            "如果 goal mode 也处于活动状态，继续维护 goal 状态；删除 worktree 不会禁用 goal mode。",
        ]
    else:
        instructions = [
            "此线程有活动 Git worktree；除非用户另有要求，文件系统、Git、构建和测试工作都应在其中进行。",
            "如果命令需要在 worktree 中运行，请在 run_python 早期调用 rt.cd(workspace.path)。",
            "Worktree mode 独立于 goal mode；两者同时活动时，两组指令都要遵循。",
        ]
    return {
        "status": status,
        "summary": "此线程的 Worktree mode 处于活动状态。" if status == "active" else "此线程的 Worktree mode 已关闭。",
        "workspace": workspace,
        "instructions": instructions,
    }


def _value(metadata: dict[str, Any], key: str) -> str:
    return str(metadata.get(key) or "").strip()
