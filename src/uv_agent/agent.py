from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from uv_agent.config import AppConfig
from uv_agent.ids import new_id
from uv_agent.model_client import ModelClient, ModelResponse
from uv_agent.paths import project_state_dir, uv_agent_home
from uv_agent.runner import PythonRunRequest, PythonRunner
from uv_agent.session.store import ThreadStore
from uv_agent.skills import discover_skills, render_skill_summary


PYTHON_TOOL = {
    "type": "function",
    "name": "run_python",
    "description": (
        "Run a Python script through the uv-agent Python runner. Use this as the only "
        "way to inspect files, call subprocesses, access the network, or perform external actions. "
        "Declare third-party dependencies inside the script with PEP 723 inline metadata."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python script source. Include PEP 723 inline metadata when dependencies are needed.",
            },
            "uv_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exceptional extra arguments for uv run, such as --refresh-package.",
                "default": [],
            },
            "script_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments passed to the Python script.",
                "default": [],
            },
            "timeout_s": {
                "type": "number",
                "description": "Execution timeout in seconds.",
                "default": 60,
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    "strict": False,
}


SYSTEM_INSTRUCTIONS_TEMPLATE = """You are uv-agent, an experimental coding agent.

Environment:
- Workspace: {workspace}
- User state: {user_state}
- Project state: {project_state}
- Persisted scripts/runs/threads live under the project state directory.

Rules:
- You have exactly one external action tool: run_python.
- Use Python for file inspection, edits, subprocesses, network access, and verification.
- Do not assume shell/filesystem/browser/network tools exist outside Python.
- Put third-party dependencies in PEP 723 inline metadata. uv_agent_runtime is injected automatically.
- Prefer small inspect-then-change steps, keep scripts deterministic, and run focused verification when behavior changes.
- Never print secrets; summarize sensitive config after redaction.

Runtime helpers available in scripts:
- uv_agent_runtime: read_text, write_text, read_json, write_json, list_files
- run_command/check_command for subprocesses
- emit_event/emit_progress/emit_result for structured output
- ask for a nested uv-agent subagent via subprocess when useful

Skills discovered under .agents/skills:
{skills}
"""


class AgentEngine:
    def __init__(
        self,
        *,
        config: AppConfig,
        model_client: ModelClient,
        runner: PythonRunner,
        thread_store: ThreadStore,
        project_root: Path,
    ) -> None:
        self.config = config
        self.model_client = model_client
        self.runner = runner
        self.thread_store = thread_store
        self.project_root = project_root

    async def run_turn(
        self,
        *,
        user_text: str,
        thread_id: str | None = None,
        level: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        thread_id = thread_id or self.thread_store.create_thread("New thread")
        turn_id = new_id("turn")
        input_items = self._reconstruct_input(thread_id)
        self.thread_store.append(thread_id, "turn.started", turn_id=turn_id)
        user_item = message_item("user", user_text)
        self.thread_store.append(thread_id, "item.user", turn_id=turn_id, item=user_item)
        input_items.append(user_item)

        final_text = ""
        for round_index in range(self.config.runtime.max_agent_rounds):
            response: ModelResponse | None = None
            async for stream_event in self.model_client.stream_response(
                input_items=input_items,
                level=level,
                tools=[PYTHON_TOOL],
                instructions=self.system_instructions(),
            ):
                if stream_event.type == "text_delta" and stream_event.text:
                    self.thread_store.append(
                        thread_id,
                        "item.assistant_delta",
                        turn_id=turn_id,
                        text=stream_event.text,
                    )
                    yield {
                        "type": "assistant.delta",
                        "thread_id": thread_id,
                        "turn_id": turn_id,
                        "text": stream_event.text,
                    }
                elif stream_event.type == "completed":
                    response = stream_event.response
            if response is None:
                raise RuntimeError("Model stream ended without completion")
            self.thread_store.append(
                thread_id,
                "item.model_response",
                turn_id=turn_id,
                response_id=response.id,
                output=response.output,
                usage=response.usage,
            )
            yield {
                "type": "model.response",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "response": response,
            }
            input_items.extend(response.output)

            tool_calls = [item for item in response.output if item.get("type") == "function_call"]
            if not tool_calls:
                final_text = response.output_text
                self.thread_store.append(
                    thread_id,
                    "item.assistant",
                    turn_id=turn_id,
                    text=final_text,
                )
                break

            for call in tool_calls:
                self.thread_store.append(
                    thread_id,
                    "item.tool_call",
                    turn_id=turn_id,
                    item=call,
                )
                yield {
                    "type": "tool.started",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "call": call,
                }
                tool_output = await self._handle_tool_call(call, thread_id, turn_id)
                self.thread_store.append(
                    thread_id,
                    "item.tool_output",
                    turn_id=turn_id,
                    item=tool_output,
                )
                input_items.append(tool_output)
                yield {
                    "type": "tool.output",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "call": call,
                    "output": tool_output,
                }
        else:
            raise RuntimeError("Agent exceeded max_agent_rounds")

        self.thread_store.append(
            thread_id,
            "turn.completed",
            turn_id=turn_id,
            final_text=final_text,
        )
        await self._maybe_compact(thread_id, turn_id, input_items)
        yield {
            "type": "turn.completed",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "final_text": final_text,
        }

    async def _maybe_compact(
        self,
        thread_id: str,
        turn_id: str,
        input_items: list[dict[str, Any]],
    ) -> None:
        if not self.config.runtime.auto_compress:
            return
        default_model = self.config.model_for_level(None)
        approx_tokens = estimate_tokens(input_items)
        trigger_tokens = int(
            default_model.context_window_tokens
            * self.config.runtime.compression.trigger_ratio
        )
        if approx_tokens < trigger_tokens:
            return
        prompt = self.config.runtime.compression.prompt
        compact_input = [
            message_item(
                "user",
                prompt + "\n\nConversation items:\n" + json.dumps(input_items, ensure_ascii=False),
            )
        ]
        response = await self.model_client.create_response(
            input_items=compact_input,
            level=self.config.runtime.compression.model_level,
            tools=[],
            instructions="Create a concise continuation summary for this uv-agent thread.",
        )
        self.thread_store.append(
            thread_id,
            "item.compaction",
            turn_id=turn_id,
            text=response.output_text,
            usage=response.usage,
        )

    async def _handle_tool_call(
        self,
        call: dict[str, Any],
        thread_id: str,
        turn_id: str,
    ) -> dict[str, Any]:
        if call.get("name") != "run_python":
            output = {"error": f"Unsupported tool: {call.get('name')}"}
            return function_output(call, output)
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            output = {"error": f"Invalid tool arguments JSON: {exc}"}
            return function_output(call, output)

        result = await self.runner.run(
            PythonRunRequest(
                code=args["code"],
                uv_args=list(args.get("uv_args") or []),
                script_args=list(args.get("script_args") or []),
                timeout_s=float(args.get("timeout_s") or self.config.runner.default_timeout_s),
                cwd=self.project_root,
                thread_id=thread_id,
                turn_id=turn_id,
            )
        )
        payload = {
            "script_id": result.script_id,
            "run_id": result.run_id,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "run_log_path": str(result.run_log_path),
        }
        self.thread_store.append(
            thread_id,
            "item.runner_result",
            turn_id=turn_id,
            call_id=call.get("call_id"),
            result=payload,
        )
        return function_output(call, payload)

    def _reconstruct_input(self, thread_id: str) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        for event in self.thread_store.read(thread_id):
            if event.get("type") == "item.user":
                input_items.append(event["item"])
            elif event.get("type") == "item.model_response":
                input_items.extend(event.get("output") or [])
            elif event.get("type") == "item.tool_output":
                input_items.append(event["item"])
        return input_items

    def context_percent(self, thread_id: str | None, level: str | None = None) -> int:
        """Return an approximate context-window usage percentage for a thread."""
        if not thread_id:
            return 0
        model = self.config.model_for_level(level)
        used = estimate_tokens(self._reconstruct_input(thread_id))
        return min(100, max(0, round(used * 100 / model.context_window_tokens)))

    def system_instructions(self) -> str:
        """Build concise environment-aware system instructions."""
        skills = discover_skills(self.project_root)
        return SYSTEM_INSTRUCTIONS_TEMPLATE.format(
            workspace=self.project_root,
            user_state=uv_agent_home(),
            project_state=project_state_dir(self.project_root),
            skills=render_skill_summary(skills),
        )

def message_item(role: str, text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }


def function_output(call: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call.get("call_id"),
        "output": json.dumps(output, ensure_ascii=False),
    }


def estimate_tokens(items: list[dict[str, Any]]) -> int:
    # A rough local estimate is enough to decide when to ask the model to compress.
    text = json.dumps(items, ensure_ascii=False)
    return max(1, len(text) // 4)
