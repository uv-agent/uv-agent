from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from uv_agent.tui.formatting import parse_tool_payload, short_thread


def main() -> None:
    parser = argparse.ArgumentParser(prog="uv-agent")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["tui", "workflow-node", "daemon"],
        default="tui",
        help="Run the terminal TUI, daemon, or execute a workflow node.",
    )
    parser.add_argument("--level", default=None, help="Model level to use for workflow-node mode.")
    parser.add_argument("--thread", default=None, help="Thread id to resume in workflow-node mode.")
    parser.add_argument(
        "--thread-kind",
        default="thread",
        choices=["thread", "workflow_node"],
        help="Thread storage kind for internal execution modes.",
    )
    parser.add_argument("--parent-thread", default=None, help="Parent thread id for workflow-node mode.")
    parser.add_argument("--parent-turn", default=None, help="Parent turn id for workflow-node mode.")
    parser.add_argument("--parent-run", default=None, help="Parent run id for workflow-node mode.")
    parser.add_argument("--workflow-id", default=None, help="Workflow id for workflow-node mode.")
    parser.add_argument("--node-id", default=None, help="Workflow node id for workflow-node mode.")
    parser.add_argument(
        "--project-state-dir",
        default=None,
        help="Project state directory for workflow-node mode.",
    )
    parser.add_argument("--no-stream", action="store_true", help="Only print the final answer in workflow-node mode.")
    parser.add_argument("--replace", action="store_true", help="Replace an existing daemon for this project state.")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override uv-agent log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text for workflow-node mode.")
    args = parser.parse_args()
    if args.command == "tui":
        from uv_agent.tui.app import UvAgentApp

        UvAgentApp(project_root=Path.cwd(), log_level=args.log_level).run()
        return
    if args.command == "daemon":
        from uv_agent.daemon import run_daemon

        asyncio.run(
            run_daemon(
                project_root=Path.cwd(),
                data_dir=Path(args.project_state_dir) if args.project_state_dir else None,
                replace=args.replace,
                log_level=args.log_level,
            )
        )
        return
    if args.command == "workflow-node":
        prompt = " ".join(args.prompt).strip()
        if not prompt:
            raise SystemExit("workflow-node mode requires a prompt")
        asyncio.run(
            _workflow_node(
                prompt,
                args.level,
                args.thread,
                stream=not args.no_stream,
                parent_thread_id=args.parent_thread,
                parent_turn_id=args.parent_turn,
                parent_run_id=args.parent_run,
                workflow_id=args.workflow_id,
                node_id=args.node_id,
                project_state_dir=Path(args.project_state_dir) if args.project_state_dir else None,
                log_level=args.log_level,
            )
        )
        return
    raise SystemExit(f"Unknown command: {args.command}")


async def _workflow_node(
    prompt: str,
    level: str | None,
    thread_id: str | None,
    *,
    stream: bool,
    parent_thread_id: str | None = None,
    parent_turn_id: str | None = None,
    parent_run_id: str | None = None,
    workflow_id: str | None = None,
    node_id: str | None = None,
    project_state_dir: Path | None = None,
    log_level: str | int | None = None,
) -> None:
    """Execute one workflow node in an isolated thread and print its final answer."""

    from uv_agent.app_factory import create_engine

    engine = create_engine(Path.cwd(), data_dir=project_state_dir, log_level=log_level, log_to_console=False)
    try:
        if level is None:
            workflow_config = engine.config.plugins.plugin_config("builtin.workflow")
            configured_default_level = workflow_config.get("default_level")
            if isinstance(configured_default_level, str) and configured_default_level:
                level = configured_default_level
        if thread_id is None:
            title = prompt.splitlines()[0].strip()
            if len(title) > 80:
                title = title[:77].rstrip() + "..."
            thread_id = engine.thread_store.create_thread(
                f"Workflow node: {title or node_id or 'task'}",
                kind="workflow_node",
                parent_thread_id=parent_thread_id,
                parent_turn_id=parent_turn_id,
                parent_run_id=parent_run_id,
            )
            engine.thread_store.append(
                thread_id,
                "thread.workflow_node_bound",
                workflow_id=workflow_id,
                node_id=node_id,
            )
            print(f"[workflow-node-thread] {thread_id}", file=sys.stderr)
        saw_delta = False
        async for event in engine.run_turn(user_text=prompt, thread_id=thread_id, level=level):
            event_type = event["type"]
            if event_type == "assistant.delta":
                if stream:
                    print(event["text"], end="", flush=True)
                saw_delta = True
            elif event_type == "tool.started" and stream:
                call = event.get("call") or {}
                print(
                    f"\n\n[python] {call.get('name', 'run_python')} started "
                    f"({call.get('call_id') or 'call'})",
                    file=sys.stderr,
                    flush=True,
                )
            elif event_type == "tool.partial" and stream:
                payload = parse_tool_payload(event.get("output") or {}) or {}
                stdout = str(payload.get("stdout") or "").strip()
                stderr = str(payload.get("stderr") or "").strip()
                summary = stdout.splitlines()[-1] if stdout else stderr.splitlines()[-1] if stderr else ""
                if len(summary) > 140:
                    summary = summary[:137].rstrip() + "..."
                status = payload.get("partial_reason") or "running"
                print(
                    f"[python] {status} run={payload.get('run_id')} {summary}",
                    file=sys.stderr,
                    flush=True,
                )
            elif event_type == "tool.output" and stream:
                payload = parse_tool_payload(event.get("output") or {}) or {}
                stdout = str(payload.get("stdout") or "").strip()
                stderr = str(payload.get("stderr") or "").strip()
                summary = stdout.splitlines()[-1] if stdout else stderr.splitlines()[-1] if stderr else ""
                if len(summary) > 140:
                    summary = summary[:137].rstrip() + "..."
                print(
                    f"[python] rc={payload.get('returncode')} run={payload.get('run_id')} {summary}",
                    file=sys.stderr,
                    flush=True,
                )
            elif event_type == "turn.completed":
                if not stream or not saw_delta:
                    print(event["final_text"])
                elif stream:
                    print()
                print(f"[thread] {short_thread(event['thread_id'])}", file=sys.stderr)
            elif event_type == "turn.error":
                message = str(event.get("message") or "turn failed")
                error_type = str(event.get("error_type") or "TurnError")
                print(f"[{error_type}] {message}", file=sys.stderr)
                raise SystemExit(1)
    finally:
        await engine.aclose()
