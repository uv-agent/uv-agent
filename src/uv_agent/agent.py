from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from html import escape as xml_escape
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from uv_agent.attachments import AttachmentStore, image_message_item
from uv_agent.config import AppConfig
from uv_agent.context import ContextStats, compact_target_tokens, estimate_tokens, usage_token_count
from uv_agent.environment import detect_user_language, host_environment, host_environment_line
from uv_agent.ids import new_id
from uv_agent.mcp_config import discover_mcp_servers, render_mcp_summary
from uv_agent.model_client import ModelClient, ModelResponse
from uv_agent.paths import project_state_dir, uv_agent_home
from uv_agent.project_rules import ProjectRuleContext, load_project_rules
from uv_agent.runner import PythonRunRequest, PythonRunner, RerunRequest
from uv_agent.session.store import ThreadStore, digest_items
from uv_agent.skills import discover_skills, render_skill_summary


class TurnInterrupted(Exception):
    """Raised internally when the active turn is interrupted by the user."""


PYTHON_TOOL = {
    "type": "function",
    "name": "run_python",
    "description": (
        "Run a Python script through the uv-agent Python runner. Use this as the only "
        "way to inspect files, call subprocesses, access the network, or perform external actions. "
        "Declare third-party dependencies inside the script with PEP 723 inline metadata, "
        "or rerun a previously saved script by script_id/run_id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python script source. Include PEP 723 inline metadata when dependencies are needed. Omit only when rerunning by script_id/run_id.",
            },
            "script_id": {
                "type": "string",
                "description": "Previously saved script id to rerun instead of creating new code.",
            },
            "run_id": {
                "type": "string",
                "description": "Previous run id to replay or rerun.",
            },
            "rerun_mode": {
                "type": "string",
                "enum": ["rerun", "replay"],
                "description": "rerun uses fresh args; replay inherits the previous run context when run_id is given.",
                "default": "rerun",
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
        "required": [],
        "additionalProperties": False,
    },
    "strict": False,
}


SYSTEM_INSTRUCTIONS_TEMPLATE = """<uv_agent_system_prompt>
<identity>
You are uv-agent, an experimental coding agent.
</identity>

<environment>
<workspace>{workspace}</workspace>
<user_state>{user_state}</user_state>
<project_state>{project_state}</project_state>
<host>{host_environment}</host>
<user_language>{user_language}</user_language>
<persistence>Persisted scripts, runs, and threads live under the project state directory.</persistence>
</environment>

<tool_boundary>
<rule>You have exactly one external action tool: run_python.</rule>
<rule>Use Python for file inspection, edits, subprocesses, network access, and verification.</rule>
<rule>Do not assume shell, filesystem, browser, or network tools exist outside Python.</rule>
<rule>When dependencies or a Python version constraint are needed, put PEP 723 inline metadata at the top of the script, for example:
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "requests",
# ]
# ///
</rule>
<rule>If no inline metadata is needed, write plain Python source without a metadata block and treat it like normal project code, not a temporary-script wrapper. uv_agent_runtime is injected automatically even if metadata is omitted.</rule>
<rule>Use uv_args only for exceptional uv behavior such as refresh, reinstall, or debug flags.</rule>
<rule>Prefer small inspect-then-change steps, then run focused verification when behavior changes.</rule>
<rule>Never print secrets; summarize sensitive config after redaction.</rule>
</tool_boundary>

<runtime_helpers>
<helper>uv_agent_runtime: read_text, write_text, read_json, write_json, list_files</helper>
<helper>apply_patch(patch_text) applies unified diffs for focused file edits</helper>
<helper>run_command/check_command for subprocesses</helper>
<helper>emit_event/emit_progress/emit_result for structured output</helper>
<helper>look_at(path, note="") attaches an image to future model context</helper>
<helper>rerun saved scripts by passing script_id or run_id to run_python</helper>
<helper>ask(prompt, level="small"|"medium"|"large") can invoke a temporary nested uv-agent subprocess when useful</helper>
<helper>Use subagents for tedious or parallelizable work such as summarizing a thread, searching candidate files, auditing repetitive patterns, or gathering focused evidence.</helper>
<helper>saved_scripts(limit=32) returns recent managed scripts with ids, timestamps, first lines, and run counts</helper>
<helper>thread_digest(thread_id) and list_thread_digests() return compact cross-thread summaries</helper>
<helper>MCP helpers connect to declared stdio MCP servers; call MCP through Python, not as model tools</helper>
</runtime_helpers>

<mentions>
<rule>User text may include @file or @thread references. Mentions are plain-text hints only; they do not attach or load content automatically.</rule>
<rule>When a mentioned file or thread matters, use run_python helpers such as read_text, list_files, thread_digest, or list_thread_digests to inspect it yourself.</rule>
</mentions>

<dynamic_workspace_context>
<rule>Rules, skills, and MCP declarations are appended only when first seen, changed, removed, or after compaction.</rule>
<rule>A removal notice means older appended rule/capability context must not be used unless it appears again.</rule>
<rule>Interrupted turns may appear in context as summaries of partial work; do not assume unfinished tool calls completed.</rule>
</dynamic_workspace_context>
</uv_agent_system_prompt>
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
        config_loader: Callable[[], AppConfig] | None = None,
    ) -> None:
        self.config = config
        self.model_client = model_client
        self.runner = runner
        self.thread_store = thread_store
        self.project_root = project_root
        self.attachments = AttachmentStore(thread_store.data_dir)
        self._last_config_refresh_at = 0.0
        self._config_loader = config_loader
        self._host_environment = host_environment()

    def refresh_config(self, *, force: bool = False) -> None:
        """Reload user/project config for long-running sessions."""
        if self._config_loader is None:
            return
        now = monotonic()
        if not force and now - self._last_config_refresh_at < 1.0:
            return
        self.config = self._config_loader()
        if hasattr(self.model_client, "reload_config"):
            self.model_client.reload_config(self.config)  # type: ignore[attr-defined]
        self.runner.config = self.config.runner
        self._last_config_refresh_at = now

    async def run_turn(
        self,
        *,
        user_text: str,
        thread_id: str | None = None,
        level: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        self.refresh_config(force=True)
        thread_id = thread_id or self.thread_store.create_thread("New thread")
        turn_id = new_id("turn")
        conversation_items = self._reconstruct_input(thread_id)
        input_items = list(conversation_items)
        context_items = self._workspace_context_items(thread_id)
        self.thread_store.append(thread_id, "turn.started", turn_id=turn_id)
        user_item = message_item("user", user_text)
        self.thread_store.append(thread_id, "item.user", turn_id=turn_id, item=user_item)
        input_items.extend(context_items)
        input_items.append(user_item)
        conversation_items.append(user_item)

        final_text = ""
        try:
            for round_index in range(self.config.runtime.max_agent_rounds):
                self._raise_if_cancelled(cancel_event)
                response: ModelResponse | None = None
                async for stream_event in self._stream_response_until_cancelled(
                    input_items=input_items,
                    level=level,
                    cancel_event=cancel_event,
                ):
                    self._raise_if_cancelled(cancel_event)
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
                    elif stream_event.type == "reasoning_delta" and stream_event.text:
                        self.thread_store.append(
                            thread_id,
                            "item.reasoning_delta",
                            turn_id=turn_id,
                            text=stream_event.text,
                        )
                        yield {
                            "type": "assistant.reasoning_delta",
                            "thread_id": thread_id,
                            "turn_id": turn_id,
                            "text": stream_event.text,
                        }
                    elif stream_event.type == "tool_call_delta" and stream_event.tool_call:
                        yield {
                            "type": "tool.delta",
                            "thread_id": thread_id,
                            "turn_id": turn_id,
                            "tool_call": stream_event.tool_call,
                        }
                    elif stream_event.type == "completed":
                        response = stream_event.response
                self._raise_if_cancelled(cancel_event)
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
                conversation_items.extend(response.output)

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
                    self._raise_if_cancelled(cancel_event)
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
                    tool_output, attachments = await self._handle_tool_call(
                        call,
                        thread_id,
                        turn_id,
                        cancel_event=cancel_event,
                    )
                    self.thread_store.append(
                        thread_id,
                        "item.tool_output",
                        turn_id=turn_id,
                        item=tool_output,
                    )
                    input_items.append(tool_output)
                    conversation_items.append(tool_output)
                    for attachment in attachments:
                        image_item = image_message_item(attachment)
                        input_items.append(image_item)
                        conversation_items.append(image_item)
                    yield {
                        "type": "tool.output",
                        "thread_id": thread_id,
                        "turn_id": turn_id,
                        "call": call,
                        "output": tool_output,
                    }
            else:
                raise RuntimeError("Agent exceeded max_agent_rounds")
        except (asyncio.CancelledError, TurnInterrupted):
            self.thread_store.append(
                thread_id,
                "turn.interrupted",
                turn_id=turn_id,
                reason="user_interrupt",
            )
            yield {
                "type": "turn.interrupted",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "reason": "user_interrupt",
            }
            return

        self.thread_store.append(
            thread_id,
            "turn.completed",
            turn_id=turn_id,
            final_text=final_text,
        )
        compacted = await self._maybe_compact(thread_id, turn_id, conversation_items)
        if compacted:
            yield {
                "type": "compaction.completed",
                "thread_id": thread_id,
                "turn_id": turn_id,
            }
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
    ) -> bool:
        if not self.config.runtime.auto_compress:
            return False
        default_model = self.config.model_for_level(None)
        approx_tokens = estimate_tokens(input_items)
        if approx_tokens < self.config.runtime.compression.min_tokens:
            return False
        trigger_tokens = int(
            default_model.context_window_tokens
            * self.config.runtime.compression.trigger_ratio
        )
        if approx_tokens < trigger_tokens:
            return False
        target_tokens = compact_target_tokens(
            default_model.context_window_tokens,
            target_ratio=self.config.runtime.compression.target_ratio,
        )
        prompt = self.config.runtime.compression.prompt
        compact_input = [
            message_item(
                "user",
                prompt
                + f"\n\nTarget length: about {target_tokens} tokens."
                + "\n\nConversation items:\n"
                + json.dumps(input_items, ensure_ascii=False),
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
        return True

    async def _handle_tool_call(
        self,
        call: dict[str, Any],
        thread_id: str,
        turn_id: str,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if call.get("name") != "run_python":
            output = {"error": f"Unsupported tool: {call.get('name')}"}
            return function_output(call, output), []
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            output = {"error": f"Invalid tool arguments JSON: {exc}"}
            return function_output(call, output), []

        if args.get("script_id") or args.get("run_id"):
            result = await self.runner.rerun(
                RerunRequest(
                    script_id=args.get("script_id"),
                    run_id=args.get("run_id"),
                    mode="replay" if args.get("rerun_mode") == "replay" else "rerun",
                    uv_args=list(args.get("uv_args") or []),
                    script_args=list(args.get("script_args") or []),
                    timeout_s=float(args.get("timeout_s") or self.config.runner.default_timeout_s),
                    cwd=self.project_root,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    cancel_event=cancel_event,
                )
            )
        else:
            code = args.get("code")
            if not isinstance(code, str) or not code.strip():
                output = {"error": "run_python requires code, script_id, or run_id"}
                return function_output(call, output), []
            result = await self.runner.run(
                PythonRunRequest(
                    code=code,
                    uv_args=list(args.get("uv_args") or []),
                    script_args=list(args.get("script_args") or []),
                    timeout_s=float(args.get("timeout_s") or self.config.runner.default_timeout_s),
                    cwd=self.project_root,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    cancel_event=cancel_event,
                )
            )
        if result.interrupted:
            raise TurnInterrupted()
        payload = {
            "script_id": result.script_id,
            "run_id": result.run_id,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "interrupted": result.interrupted,
            "truncated": result.truncated,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "events": result.events,
            "run_log_path": str(result.run_log_path),
        }
        attachments = self._register_look_at_events(
            result.events,
            thread_id=thread_id,
            turn_id=turn_id,
            cwd=self.project_root,
        )
        if attachments:
            payload["attachments"] = attachments
        self.thread_store.append(
            thread_id,
            "item.runner_result",
            turn_id=turn_id,
            call_id=call.get("call_id"),
            result=payload,
        )
        return function_output(call, payload), attachments

    @staticmethod
    def _raise_if_cancelled(cancel_event: asyncio.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise TurnInterrupted()

    async def _stream_response_until_cancelled(
        self,
        *,
        input_items: list[dict[str, Any]],
        level: str | None,
        cancel_event: asyncio.Event | None,
    ) -> AsyncIterator[Any]:
        stream = self.model_client.stream_response(
            input_items=input_items,
            level=level,
            tools=[PYTHON_TOOL],
            instructions=self.system_instructions(),
        )
        iterator = stream.__aiter__()
        cancel_task: asyncio.Task[bool] | None = None
        if cancel_event is not None:
            cancel_task = asyncio.create_task(cancel_event.wait())
        try:
            while True:
                next_task = asyncio.create_task(iterator.__anext__())
                tasks: set[asyncio.Task[Any]] = {next_task}
                if cancel_task is not None:
                    tasks.add(cancel_task)
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if cancel_task is not None and cancel_task in done:
                    next_task.cancel()
                    await asyncio.gather(next_task, return_exceptions=True)
                    raise TurnInterrupted()
                if next_task in done:
                    if cancel_event is not None and cancel_event.is_set():
                        raise TurnInterrupted()
                    yield next_task.result()
                for task in pending:
                    if task is not cancel_task:
                        task.cancel()
        except StopAsyncIteration:
            return
        finally:
            if cancel_task is not None:
                cancel_task.cancel()
                await asyncio.gather(cancel_task, return_exceptions=True)
            aclose = getattr(iterator, "aclose", None)
            if callable(aclose):
                await aclose()

    def _register_look_at_events(
        self,
        events: list[dict[str, Any]],
        *,
        thread_id: str,
        turn_id: str,
        cwd: Path,
    ) -> list[dict[str, Any]]:
        attachments: list[dict[str, Any]] = []
        for event in events:
            if event.get("kind") != "look_at":
                continue
            path = event.get("path")
            if not isinstance(path, str) or not path:
                continue
            attachment = self.attachments.register_image(
                path,
                cwd=cwd,
                thread_id=thread_id,
                note=str(event.get("note") or ""),
            )
            payload = attachment.to_event_payload()
            self.thread_store.append(
                thread_id,
                "item.image_attachment",
                turn_id=turn_id,
                attachment=payload,
            )
            attachments.append(payload)
        return attachments

    def _reconstruct_input(self, thread_id: str) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        events = self.thread_store.read(thread_id)
        last_compaction_index = -1
        for index, event in enumerate(events):
            if event.get("type") == "item.compaction":
                last_compaction_index = index
        if last_compaction_index >= 0:
            summary = str(events[last_compaction_index].get("text") or "").strip()
            if summary:
                input_items.append(
                    message_item(
                        "user",
                        "<conversation_summary>\n"
                        + summary
                        + "\n</conversation_summary>\nContinue from this compacted context.",
                    )
                )
        for event in events[: last_compaction_index + 1]:
            if event.get("type") == "item.image_attachment":
                input_items.append(image_message_item(event["attachment"]))
        for event in events[last_compaction_index + 1 :]:
            if event.get("type") == "item.user":
                input_items.append(event["item"])
            elif event.get("type") == "item.model_response":
                input_items.extend(event.get("output") or [])
            elif event.get("type") == "item.tool_output":
                input_items.append(event["item"])
            elif event.get("type") == "item.image_attachment":
                input_items.append(image_message_item(event["attachment"]))
            elif event.get("type") == "turn.interrupted":
                interrupted = self._interrupted_turn_context(events, str(event.get("turn_id") or ""))
                if interrupted:
                    input_items.append(message_item("user", interrupted))
        return input_items

    def _interrupted_turn_context(self, events: list[dict[str, Any]], turn_id: str) -> str:
        turn_events = [
            event
            for event in events
            if event.get("turn_id") == turn_id
            and event.get("type") not in {"item.tool_call", "item.tool_output"}
        ]
        items = digest_items(turn_events, include_tools=False)
        if not items:
            return ""
        lines = [
            "<interrupted_turn>",
            "The previous turn was interrupted by the user. Use this as context only; do not assume unfinished tool calls completed.",
        ]
        for item in items:
            role = item.get("role")
            text = str(item.get("text") or "").strip()
            if text:
                lines.append(f"{role}: {text}")
        lines.append("</interrupted_turn>")
        return "\n".join(lines)

    def _workspace_context_items(self, thread_id: str | None = None) -> list[dict[str, Any]]:
        update = self._turn_context_update(thread_id)
        if update is None:
            return []
        if thread_id:
            self.thread_store.append(
                thread_id,
                "item.context_update",
                context_fingerprint=update["fingerprint"],
                context_state=update["state"],
                context_kind="workspace",
                removed=update["removed"],
                text=update["text"],
            )
        return [message_item("user", update["text"])]

    def _turn_context_update(self, thread_id: str | None) -> dict[str, Any] | None:
        parts = self._turn_context_parts()
        rendered = "\n\n".join(parts.values())
        fingerprint = context_fingerprint(rendered)
        state = {key: context_fingerprint(value) for key, value in parts.items()}
        previous = self._latest_context_state(thread_id) if thread_id else None
        previous_fingerprint = previous.get("fingerprint") if previous else None
        if previous_fingerprint == fingerprint:
            return None
        previous_parts = previous.get("parts", {}) if previous else {}
        removed = [key for key in previous_parts if key not in state]
        changed = [key for key in state if previous_parts.get(key) != state[key]]
        if not rendered:
            if previous_fingerprint is None:
                return None
            return {
                "fingerprint": fingerprint,
                "state": {"fingerprint": fingerprint, "parts": state},
                "removed": removed or sorted(previous_parts),
                "text": (
                    "<workspace_context_update>\n"
                    "Previously available workspace rules, skills, or MCP declarations are no longer present. "
                    "Do not rely on older appended capability/rule context unless it appears again.\n"
                    "</workspace_context_update>"
                ),
            }
        if removed:
            removed_text = (
                "\n\n<workspace_context_removed>\n"
                f"Removed context kinds: {', '.join(removed)}. "
                "Do not rely on older appended content for these kinds unless it appears again.\n"
                "</workspace_context_removed>"
            )
        else:
            removed_text = ""
        prefix = (
            "<workspace_context_update>\n"
            "The following workspace rules/capabilities are current. This update replaces any older appended "
            "workspace rules, skills, or MCP declarations in this thread.\n"
            f"fingerprint: {fingerprint}\n"
            + (f"removed: {', '.join(removed)}\n" if removed else "")
            + (f"changed: {', '.join(changed)}\n" if changed else "")
            + "</workspace_context_update>"
        )
        return {
            "fingerprint": fingerprint,
            "state": {"fingerprint": fingerprint, "parts": state},
            "removed": removed,
            "text": prefix + removed_text + "\n\n" + rendered,
        }

    def _latest_context_state(self, thread_id: str | None) -> dict[str, Any] | None:
        if not thread_id:
            return None
        after_compaction = False
        for event in reversed(self.thread_store.read(thread_id)):
            if event.get("type") == "item.context_update":
                if after_compaction:
                    return None
                state = event.get("context_state")
                if isinstance(state, dict):
                    return state
                return {"fingerprint": str(event.get("context_fingerprint") or ""), "parts": {}}
            if event.get("type") == "item.compaction":
                after_compaction = True
        return None

    def _turn_context_text(self) -> str:
        return "\n\n".join(self._turn_context_parts().values())

    def _turn_context_parts(self) -> dict[str, str]:
        sections: dict[str, str] = {}
        rendered_rules = self.project_rule_context().render()
        if rendered_rules:
            sections["rules"] = rendered_rules

        skills = render_skill_summary(discover_skills(self.project_root))
        if skills != "None discovered.":
            sections["skills"] = (
                "<available_skills>\n"
                "Read the listed SKILL.md with Python only when relevant.\n"
                f"{skills}\n"
                "</available_skills>"
            )

        mcp_servers = render_mcp_summary(discover_mcp_servers(self.project_root))
        if mcp_servers != "None declared.":
            sections["mcp"] = (
                "<available_mcp_servers>\n"
                "Use uv_agent_runtime MCP helpers from Python to inspect or call these servers.\n"
                f"{mcp_servers}\n"
                "</available_mcp_servers>"
            )
        return sections

    def project_rule_context(self) -> ProjectRuleContext:
        """Load AGENTS.md context for the active workspace."""
        return load_project_rules(self.project_root)

    def context_percent(self, thread_id: str | None, level: str | None = None) -> int:
        """Return a context-window usage percentage for a thread."""
        return self.context_stats(thread_id, level).percent

    def context_stats(self, thread_id: str | None, level: str | None = None) -> ContextStats:
        """Return detailed context-window statistics for a thread."""
        model = self.config.model_for_level(level)
        trigger_tokens = int(
            model.context_window_tokens * self.config.runtime.compression.trigger_ratio
        )
        target_tokens = compact_target_tokens(
            model.context_window_tokens,
            target_ratio=self.config.runtime.compression.target_ratio,
        )
        if not thread_id:
            return ContextStats(
                used_tokens=0,
                context_window_tokens=model.context_window_tokens,
                percent=0,
                threshold_tokens=trigger_tokens,
                target_tokens=target_tokens,
                headroom_tokens=model.context_window_tokens,
                source="empty",
            )
        used = self._latest_usage_tokens(thread_id)
        source = "provider"
        if used is None:
            update = self._turn_context_update(thread_id)
            context_items = [message_item("user", update["text"])] if update else []
            used = estimate_tokens(self._reconstruct_input(thread_id) + context_items)
            source = "estimate"
        percent = min(100, max(0, round(used * 100 / model.context_window_tokens)))
        return ContextStats(
            used_tokens=used,
            context_window_tokens=model.context_window_tokens,
            percent=percent,
            threshold_tokens=trigger_tokens,
            target_tokens=target_tokens,
            headroom_tokens=max(0, model.context_window_tokens - used),
            source=source,
        )

    def _latest_usage_tokens(self, thread_id: str) -> int | None:
        """Return the latest provider-reported token usage when available."""
        for event in reversed(self.thread_store.read(thread_id)):
            if event.get("type") not in {"item.model_response", "item.compaction"}:
                continue
            used = usage_token_count(event.get("usage") or {})
            if used is not None:
                return used
        return None

    def system_instructions(self) -> str:
        """Build concise environment-aware system instructions."""
        return SYSTEM_INSTRUCTIONS_TEMPLATE.format(
            workspace=xml_text(self.project_root),
            user_state=xml_text(uv_agent_home()),
            project_state=xml_text(project_state_dir(self.project_root)),
            host_environment=xml_text(host_environment_line(self._host_environment)),
            user_language=xml_text(detect_user_language(self.config.ui.language).name),
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


def context_fingerprint(text: str) -> str:
    """Stable fingerprint for dynamic per-turn context."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def xml_text(value: object) -> str:
    return xml_escape(str(value), quote=False)
