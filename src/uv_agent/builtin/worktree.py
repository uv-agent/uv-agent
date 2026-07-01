from __future__ import annotations

from typing import Any

from uv_agent.plugins import PluginManifest, SetupPlugin


MANIFEST = PluginManifest(
    id="builtin.worktree",
    version="0.1.0",
    display_name="Worktree Context",
    description="Publishes model-visible context for thread-scoped Git worktrees.",
    builtin=True,
    priority=95,
    capabilities=("context",),
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


def setup(context) -> None:
    if context.threads is not None:
        for metadata in context.threads.list_threads():
            thread_id = str(metadata.get("thread_id") or "").strip()
            if thread_id and str(metadata.get("worktree_status") or "") == "active":
                _publish_worktree_context(context, thread_id, metadata)

    def on_thread_event(event: dict[str, Any]) -> None:
        stored = event.get("event") if isinstance(event.get("event"), dict) else {}
        event_type = str(stored.get("type") or "")
        if event_type not in {"thread.worktree_created", "thread.worktree_deleted"}:
            return
        thread_id = str(event.get("thread_id") or stored.get("thread_id") or "").strip()
        if not thread_id or context.threads is None:
            return
        try:
            metadata = context.threads.metadata(thread_id)
        except FileNotFoundError:
            return
        _publish_worktree_context(context, thread_id, metadata, event_type=event_type)

    context.events.subscribe("thread.event_stored", on_thread_event, logger=context.logger)


def _publish_worktree_context(context, thread_id: str, metadata: dict[str, Any], *, event_type: str = "") -> None:
    status = str(metadata.get("worktree_status") or "").strip()
    if status not in {"active", "deleted"}:
        context.context.epoch.remove(tag="worktree", reason="No active worktree metadata for this thread.", thread_id=thread_id)
        return
    branch = str(metadata.get("worktree_branch") or "").strip()
    path = str(metadata.get("worktree_path") or "").strip()
    origin = str(metadata.get("worktree_origin_root") or "").strip()
    if not branch or not path or not origin:
        return
    body = _worktree_body(metadata, status=status)
    if status == "deleted":
        if event_type == "thread.worktree_deleted":
            context.context.epoch.update(tag="worktree", body=body, attrs={"status": "deleted"}, thread_id=thread_id)
            context.context.turn.enqueue(thread_id=thread_id, tag="worktree", body=body, attrs={"status": "deleted"})
        context.context.epoch.remove(tag="worktree", reason="Worktree mode closed for this thread.", thread_id=thread_id)
        return
    context.context.epoch.publish(tag="worktree", body=body, attrs={"status": "active"}, thread_id=thread_id)


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
            "This thread's previous worktree has been removed; do not rely on the deleted path or branch.",
            "Run filesystem, Git, build, and test commands from current_cwd unless the user asks otherwise.",
            "If goal mode is also active, continue preserving goal state; deleting the worktree does not disable goal mode.",
        ]
    else:
        instructions = [
            "This thread has an active Git worktree. Use it for filesystem, Git, build, and test work unless the user asks otherwise.",
            "Call rt.cd(workspace.path) early in run_python if you need commands to run in the worktree.",
            "Worktree mode is independent from goal mode; follow both sets of instructions when both are active.",
        ]
    return {
        "status": status,
        "summary": "Worktree mode active for this thread." if status == "active" else "Worktree mode has been closed for this thread.",
        "workspace": workspace,
        "instructions": instructions,
    }


def _value(metadata: dict[str, Any], key: str) -> str:
    return str(metadata.get(key) or "").strip()
