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
        choices=["tui", "ask"],
        default="tui",
        help="Run the TUI or ask a single question.",
    )
    parser.add_argument("--level", default=None, help="Model level to use for ask mode.")
    parser.add_argument("--thread", default=None, help="Thread id to resume in ask mode.")
    parser.add_argument(
        "--thread-kind",
        default="thread",
        choices=["thread", "subagent"],
        help="Thread storage kind for ask mode.",
    )
    parser.add_argument("--parent-thread", default=None, help="Parent thread id for subagent ask mode.")
    parser.add_argument("--parent-turn", default=None, help="Parent turn id for subagent ask mode.")
    parser.add_argument("--parent-run", default=None, help="Parent run id for subagent ask mode.")
    parser.add_argument(
        "--project-state-dir",
        default=None,
        help="Project state directory for subagent ask mode.",
    )
    parser.add_argument("--no-stream", action="store_true", help="Only print the final answer in ask mode.")
    parser.add_argument("prompt", nargs="*", help="Prompt text for ask mode.")
    args = parser.parse_args()
    if args.command == "ask":
        prompt = " ".join(args.prompt).strip()
        if not prompt:
            raise SystemExit("ask mode requires a prompt")
        asyncio.run(
            _ask(
                prompt,
                args.level,
                args.thread,
                stream=not args.no_stream,
                thread_kind=args.thread_kind,
                parent_thread_id=args.parent_thread,
                parent_turn_id=args.parent_turn,
                parent_run_id=args.parent_run,
                project_state_dir=Path(args.project_state_dir) if args.project_state_dir else None,
            )
        )
        return
    from uv_agent.tui.app import UvAgentApp

    UvAgentApp(project_root=Path.cwd()).run()


async def _ask(
    prompt: str,
    level: str | None,
    thread_id: str | None,
    *,
    stream: bool,
    thread_kind: str = "thread",
    parent_thread_id: str | None = None,
    parent_turn_id: str | None = None,
    parent_run_id: str | None = None,
    project_state_dir: Path | None = None,
) -> None:
    # Ask mode needs the full engine/model stack. Import it here instead of at
    # CLI module import time so ``uv-agent`` can parse and start TUI mode before
    # provider SDKs are loaded.
    from uv_agent.app_factory import create_engine

    engine = create_engine(Path.cwd(), data_dir=project_state_dir)
    try:
        if level is None:
            ask_default = engine.config.runtime.ask_default_level
            if ask_default:
                level = ask_default
        if thread_id is None and thread_kind == "subagent":
            title = prompt.splitlines()[0].strip()
            if len(title) > 80:
                title = title[:77].rstrip() + "..."
            thread_id = engine.thread_store.create_thread(
                f"Subagent: {title or 'task'}",
                kind="subagent",
                parent_thread_id=parent_thread_id,
                parent_turn_id=parent_turn_id,
                parent_run_id=parent_run_id,
            )
            print(f"[subagent-thread] {thread_id}", file=sys.stderr)
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
    finally:
        await engine.aclose()
