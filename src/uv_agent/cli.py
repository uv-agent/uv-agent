from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from uv_agent.app_factory import create_engine


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
    parser.add_argument("prompt", nargs="*", help="Prompt text for ask mode.")
    args = parser.parse_args()
    if args.command == "ask":
        prompt = " ".join(args.prompt).strip()
        if not prompt:
            raise SystemExit("ask mode requires a prompt")
        asyncio.run(_ask(prompt, args.level))
        return
    from uv_agent.tui.app import UvAgentApp

    UvAgentApp(project_root=Path.cwd()).run()


async def _ask(prompt: str, level: str | None) -> None:
    engine = create_engine(Path.cwd())
    async for event in engine.run_turn(user_text=prompt, level=level):
        if event["type"] == "turn.completed":
            print(event["final_text"])
