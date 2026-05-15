from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from uv_agent.app_factory import create_engine
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
    parser.add_argument("--no-stream", action="store_true", help="Only print the final answer in ask mode.")
    parser.add_argument("prompt", nargs="*", help="Prompt text for ask mode.")
    args = parser.parse_args()
    if args.command == "ask":
        prompt = " ".join(args.prompt).strip()
        if not prompt:
            raise SystemExit("ask mode requires a prompt")
        asyncio.run(_ask(prompt, args.level, args.thread, stream=not args.no_stream))
        return
    from uv_agent.tui.app import UvAgentApp

    UvAgentApp(project_root=Path.cwd()).run()


async def _ask(prompt: str, level: str | None, thread_id: str | None, *, stream: bool) -> None:
    engine = create_engine(Path.cwd())
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
