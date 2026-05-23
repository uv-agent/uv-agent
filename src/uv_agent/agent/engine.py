from __future__ import annotations

import asyncio
import copy
import inspect
import importlib
import json
import random
import re
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, field
from html import escape as xml_escape
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from uv_agent.attachments import AttachmentStore, image_message_item
from uv_agent.agent.compaction import (
    compaction_summary_item,
    compaction_replacement_input,
    compaction_trigger_item,
    normalize_compaction_replacement_input,
    retained_user_messages_after_compaction,
    retain_item_after_compaction,
)
from uv_agent.billing import billing_charge_for_usage, billing_total_from_metadata, decimal_to_string
from uv_agent.config import AppConfig
from uv_agent.agent.context_builder import (
    context_fingerprint,
    model_levels_context,
    runtime_environment_context,
    runtime_helpers_context,
    xml_text,
)
from uv_agent.context import ContextStats, estimate_tokens, usage_token_count
from uv_agent.environment import detect_user_language, host_environment
from uv_agent.errors import EmptyModelStreamError, is_retryable_provider_error
from uv_agent.ids import new_id
from uv_agent.agent.messages import assistant_output_item, message_item, message_item_text
from uv_agent.mcp_config import McpInstructionsPreview, McpServerSummary, discover_mcp_servers, render_mcp_entry
from uv_agent.mcp_probe import McpInstructionsProbe
from uv_agent.model.types import ModelClient, ModelResponse
from uv_agent.paths import uv_agent_home
from uv_agent.agent.prompts import (
    INTERRUPTED_STREAM_CONTEXT_BRIDGE,
    INTERRUPTED_TOOL_CONTEXT_BRIDGE,
    POST_TOOL_COMPACTION_BRIDGE,
    PYTHON_TOOL,
    SYSTEM_INSTRUCTIONS_TEMPLATE,
    TITLE_GENERATION_PROMPT,
    TOOL_ATTACHMENT_CONTEXT_BRIDGE,
)
from uv_agent.project_rules import (
    ProjectRuleContext,
    discover_workspace_rule_index,
    load_directory_rules,
    load_project_rules,
)
from uv_agent.runner import PythonRunRequest, PythonRunner, RunnerEvent
from uv_agent.runner.scriptenv import direct_dependencies
from uv_agent.session.store import ThreadSnapshot, ThreadStore
from uv_agent.skills import SkillSummary, discover_skills, render_skill_entry
from uv_agent.thread_titles import DEFAULT_THREAD_TITLES
from uv_agent.agent.tool_results import function_output, model_tool_payload


async def _await_next_stream_event(awaitable: Awaitable[Any]) -> Any:
    """Bridge async-iterator ``__anext__`` awaitables into coroutine tasks.

    ``asyncio.create_task`` accepts coroutine objects at runtime, but async
    generators return an ``async_generator_asend`` awaitable from ``__anext__``.
    Wrapping the awaitable keeps cancellation behavior explicit and gives static
    checkers the concrete coroutine shape they expect.
    """

    return await awaitable


async def _sleep_stream_retry(delay_s: float) -> None:
    await asyncio.sleep(delay_s)


class TurnInterrupted(Exception):
    """Raised internally when the active turn is interrupted by the user."""



async def _ensure_async_runner_events(events: Any) -> AsyncIterator[RunnerEvent]:
    """Iterate over either an async runner stream or a small synchronous list.

    Tests and third-party integrations sometimes provide a minimal runner with
    only ``run()``. The production runner exposes ``stream_run()`` so partial
    output can be rendered while a process is still running.
    """

    if hasattr(events, "__aiter__"):
        async for event in events:
            yield event
        return
    for event in events:
        yield event

def _context_item_id(key: tuple[str, str, str]) -> str:
    scope, name, path = key
    return f"{scope}:{name}:{context_fingerprint(path)}"


def _context_state_parts(state: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not state:
        return {}
    raw_parts = state.get("parts")
    if not isinstance(raw_parts, dict):
        return {}
    parts: dict[str, dict[str, Any]] = {}
    for key, value in raw_parts.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, dict):
            fingerprint = value.get("fingerprint")
            if isinstance(fingerprint, str):
                parts[key] = {
                    "fingerprint": fingerprint,
                    "kind": str(value.get("kind") or ""),
                    "dynamic": bool(value.get("dynamic")),
                    "metadata": value.get("metadata") if isinstance(value.get("metadata"), dict) else {},
                }
        elif isinstance(value, str):
            parts[key] = {
                "fingerprint": value,
                "kind": key,
                "dynamic": key in {"skills", "mcp"},
                "metadata": {},
            }
    return parts


def _removed_context_text(removed: list[str], previous_parts: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for item_id in removed:
        metadata = previous_parts.get(item_id, {}).get("metadata")
        if not isinstance(metadata, dict):
            continue
        kind = str(metadata.get("kind") or "")
        name = metadata.get("name")
        scope = metadata.get("scope")
        path = metadata.get("path") or metadata.get("config")
        if kind == "skill" and name and scope:
            lines.append(
                f'\n<removed_skill name="{_xml_attr(name)}" scope="{_xml_attr(scope)}" path="{_xml_attr(path or "")}" />'
            )
        elif kind == "mcp" and name and scope:
            lines.append(
                f'\n<removed_mcp_server name="{_xml_attr(name)}" scope="{_xml_attr(scope)}" config="{_xml_attr(path or "")}" />'
            )
    return "".join(lines)


def _xml_attr(value: object) -> str:
    return xml_escape(str(value), quote=True)


def _mcp_preview_from_metadata(value: object) -> McpInstructionsPreview | None:
    if not isinstance(value, dict):
        return None
    text = value.get("text")
    if not isinstance(text, str) or not text:
        return None
    return McpInstructionsPreview(text=text, truncated=bool(value.get("truncated")))


def _mcp_preview_metadata(preview: McpInstructionsPreview | None) -> dict[str, Any] | None:
    if preview is None:
        return None
    return {"text": preview.text, "truncated": preview.truncated}


@dataclass
class TurnInputState:
    input_items: list[dict[str, Any]]
    previous_response_id: str | None = None
    use_previous_response_id: bool = False
    pending_items: list[dict[str, Any]] = field(default_factory=list)

    def request_input_items(self) -> list[dict[str, Any]]:
        if self.use_previous_response_id and self.previous_response_id:
            return copy.deepcopy(self.pending_items)
        return copy.deepcopy(self.input_items)

    def request_previous_response_id(self) -> str | None:
        if self.use_previous_response_id and self.previous_response_id:
            return self.previous_response_id
        return None


@dataclass
class RetryState:
    input_items: list[dict[str, Any]]
    previous_response_id: str | None = None
    use_previous_response_id: bool = False
    pending_items: list[dict[str, Any]] = field(default_factory=list)
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def request_input_items(self) -> list[dict[str, Any]]:
        if self.use_previous_response_id and self.previous_response_id:
            return copy.deepcopy(self.pending_items)
        return copy.deepcopy(self.input_items)

    def request_previous_response_id(self) -> str | None:
        if self.use_previous_response_id and self.previous_response_id:
            return self.previous_response_id
        return None


@dataclass
class RuleRuntimeState:
    active_cwd: Path
    loaded_rule_paths: set[Path] = field(default_factory=set)
    index_emitted: bool = False
    cwd_notice_cwd: Path | None = None


@dataclass
class StreamResponseState:
    assistant_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    saw_stream_output: bool = False
    response: ModelResponse | None = None

    @property
    def partial_text(self) -> str:
        return "".join(self.assistant_parts).strip()

    @property
    def partial_reasoning_text(self) -> str:
        return "".join(self.reasoning_parts).strip()

    def reset(self) -> None:
        self.assistant_parts.clear()
        self.reasoning_parts.clear()
        self.response = None

    def require_response(self) -> ModelResponse:
        if self.response is None:
            raise RuntimeError("Model stream ended without completion")
        return self.response


@dataclass(frozen=True)
class ToolCallTurnResult:
    tool_output: dict[str, Any]
    attachments: list[dict[str, Any]]
    started_event: dict[str, Any]
    output_event: dict[str, Any]


@dataclass(frozen=True)
class CompactionResult:
    """State produced by one context compaction pass.

    The engine may compact both after a completed turn and mid-turn after a
    batch of tool outputs. Mid-turn callers need the freshly prepared input to
    continue without relying on a provider-side previous_response_id that still
    points at the uncompressed history.
    """

    replacement_input: list[dict[str, Any]]
    text: str
    truncated_last_tool_output: bool = False


@dataclass(frozen=True)
class CompactionDecision:
    """Outcome of checking whether a thread should be compacted now."""

    result: CompactionResult | None = None
    token_warning_event: dict[str, Any] | None = None


@dataclass(frozen=True)
class TokenCountResult:
    """Token count plus provenance used for context-window decisions."""

    tokens: int
    source: str


@dataclass(frozen=True)
class RunTurnPrelude:
    thread_id: str
    turn_id: str
    system_instructions: str
    should_generate_title: bool
    turn_input: TurnInputState
    input_items: list[dict[str, Any]]
    request_input_items: list[dict[str, Any]]
    turn_started_event: dict[str, Any]
    image_events: list[dict[str, Any]]


@dataclass(frozen=True)
class ContextPart:
    id: str
    kind: str
    text: str
    dynamic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentEngine:
    def __init__(
        self,
        *,
        config: AppConfig,
        model_client: ModelClient,
        runner: PythonRunner,
        thread_store: ThreadStore,
        attachments_dir: Path | None = None,
        project_root: Path,
        config_loader: Callable[[], AppConfig] | None = None,
        mcp_instructions_probe: McpInstructionsProbe | None = None,
    ) -> None:
        self.config = config
        self.model_client = model_client
        self.runner = runner
        self.thread_store = thread_store
        self.project_root = project_root
        self.attachments = AttachmentStore(attachments_dir or thread_store.data_dir / "attachments")
        self._last_config_refresh_at = 0.0
        self._config_loader = config_loader
        self._host_environment = host_environment()
        self._rule_states: dict[str, RuleRuntimeState] = {}
        self._mcp_instructions_probe = mcp_instructions_probe or McpInstructionsProbe(self.project_root)
        self._mcp_instructions_probe.start()

    def close(self) -> None:
        """Release long-lived host resources owned by the engine."""

        close = getattr(self.runner, "close", None)
        if callable(close):
            close()

    async def aclose(self) -> None:
        close = getattr(self.runner, "aclose", None)
        if callable(close):
            await close()
            return
        await asyncio.to_thread(self.close)

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
        image_paths: list[str | Path] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        thread_id = thread_id or await asyncio.to_thread(self.thread_store.create_thread, "New thread")
        with self.thread_store.lock_thread(thread_id):
            prelude = await asyncio.to_thread(
                self._prepare_run_turn_prelude,
                user_text=user_text,
                thread_id=thread_id,
                level=level,
                image_paths=image_paths,
                cancel_event=cancel_event,
            )
            turn_id = prelude.turn_id
            system_instructions = prelude.system_instructions
            turn_input = prelude.turn_input
            input_items = prelude.input_items
            request_input_items = prelude.request_input_items
            turn_started_event = prelude.turn_started_event
            title_task = self._start_title_generation_task(
                thread_id,
                user_text,
                should_generate=prelude.should_generate_title,
                level=level,
            )
            for event in prelude.image_events:
                yield event

            final_text = ""
            compacted_this_turn = False
            stream_state = StreamResponseState()
            try:
                for round_index in range(self.config.runtime.max_agent_rounds):
                    self._raise_if_cancelled(cancel_event)
                    async for event in self._stream_model_response_with_retries(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        turn_started_at=turn_started_event.get("created_at"),
                        input_items=copy.deepcopy(request_input_items),
                        level=level,
                        instructions=system_instructions,
                        previous_response_id=turn_input.request_previous_response_id(),
                        stream_state=stream_state,
                        cancel_event=cancel_event,
                    ):
                        yield event
                    response = stream_state.require_response()
                    input_items.extend(response.output)
                    turn_input.previous_response_id = response.id
                    turn_input.use_previous_response_id = bool(
                        response.id and self._level_uses_responses_api(level)
                    )
                    turn_input.pending_items.clear()
                    request_input_items = turn_input.request_input_items()
                    stream_state.reset()

                    tool_calls = [item for item in response.output if item.get("type") == "function_call"]
                    if not tool_calls:
                        final_text = response.output_text
                        break

                    round_attachments: list[dict[str, Any]] = []
                    for call_index, call in enumerate(tool_calls):
                        async for tool_event in self._stream_tool_call_for_turn(
                            call=call,
                            call_index=call_index,
                            thread_id=thread_id,
                            turn_id=turn_id,
                            turn_started_at=turn_started_event.get("created_at"),
                            cancel_event=cancel_event,
                        ):
                            public_event = {key: value for key, value in tool_event.items() if key != "_result"}
                            yield public_event
                            if tool_event.get("type") != "tool.output":
                                continue
                            result = self._tool_result_from_event(tool_event, public_event)
                            self.thread_store.append(
                                thread_id,
                                "item.tool_output",
                                turn_id=turn_id,
                                item=result.tool_output,
                            )
                            input_items.append(result.tool_output)
                            request_input_items.append(result.tool_output)
                            turn_input.pending_items.append(result.tool_output)
                            round_attachments.extend(result.attachments)
                    if round_attachments:
                        for attachment in round_attachments:
                            self.thread_store.append(
                                thread_id,
                                "item.image_attachment",
                                turn_id=turn_id,
                                source="tool",
                                attachment=attachment,
                            )
                        attachment_items = tool_attachment_context_items(round_attachments)
                        input_items.extend(attachment_items)
                        turn_input.pending_items.extend(copy.deepcopy(attachment_items))
                        turn_input.use_previous_response_id = False
                        request_input_items = turn_input.request_input_items()
                    if self._will_compact_after_tool_results(
                        thread_id,
                        input_items,
                        level=level,
                        instructions=system_instructions,
                    ):
                        yield self._compaction_started_event(thread_id, turn_id)
                    mid_turn_compaction = await self._maybe_compact_after_tool_results(
                        thread_id,
                        turn_id,
                        input_items,
                        level=level,
                        instructions=system_instructions,
                    )
                    if mid_turn_compaction.token_warning_event is not None:
                        yield self._public_event(mid_turn_compaction.token_warning_event)
                    if mid_turn_compaction.result is not None:
                        compacted_this_turn = True
                        yield self._compaction_completed_event(
                            thread_id,
                            turn_id,
                            mid_turn_compaction.result,
                        )
                        input_items = self._input_after_compaction(thread_id, mid_turn_compaction.result)
                        turn_input.input_items = input_items
                        turn_input.previous_response_id = None
                        turn_input.use_previous_response_id = False
                        turn_input.pending_items.clear()
                        request_input_items = turn_input.request_input_items()
                else:
                    raise RuntimeError("Agent exceeded max_agent_rounds")
            except (asyncio.CancelledError, TurnInterrupted):
                partial_text = stream_state.partial_text
                if partial_text:
                    self.thread_store.append(
                        thread_id,
                        "item.assistant_partial",
                        turn_id=turn_id,
                        text=partial_text,
                    )
                reasoning_text = stream_state.partial_reasoning_text
                if reasoning_text:
                    self.thread_store.append(
                        thread_id,
                        "item.reasoning_partial",
                        turn_id=turn_id,
                        text=reasoning_text,
                    )
                self.thread_store.append(
                    thread_id,
                    "turn.interrupted",
                    turn_id=turn_id,
                    reason="user_interrupt",
                    partial_stream=stream_state.saw_stream_output,
                )
                yield {
                    "type": "turn.interrupted",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "reason": "user_interrupt",
                    "partial_stream": stream_state.saw_stream_output,
                }
                if title_task is not None:
                    title_task.cancel()
                return
            except Exception as exc:
                error_event = self.thread_store.append(
                    thread_id,
                    "turn.error",
                    turn_id=turn_id,
                    error_type=exc.__class__.__name__,
                    message=str(exc) or repr(exc),
                    retryable=is_retryable_provider_error(exc),
                )
                if title_task is not None:
                    title_task.cancel()
                yield {
                    "type": "turn.error",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_event.get("created_at"),
                    "created_at": error_event.get("created_at"),
                    "completed_at": error_event.get("created_at"),
                    "error_type": exc.__class__.__name__,
                    "message": str(exc) or repr(exc),
                    "retryable": is_retryable_provider_error(exc),
                }
                return

            turn_completed_event = self.thread_store.append(
                thread_id,
                "turn.completed",
                turn_id=turn_id,
                final_text=final_text,
            )
            compacted = CompactionDecision()
            if not compacted_this_turn:
                if self._will_compact(thread_id, input_items, level=level, instructions=system_instructions):
                    yield self._compaction_started_event(thread_id, turn_id)
                compacted = await self._maybe_compact(
                    thread_id,
                    turn_id,
                    input_items,
                    level=level,
                    instructions=system_instructions,
                )
            if compacted.token_warning_event is not None:
                yield self._public_event(compacted.token_warning_event)
            if compacted.result is not None:
                yield self._compaction_completed_event(thread_id, turn_id, compacted.result)
            generated_title = await self._finish_title_generation(title_task)
            if generated_title:
                yield {
                    "type": "thread.title",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "title": generated_title,
                }
            yield {
                "type": "turn.completed",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "turn_started_at": turn_started_event.get("created_at"),
                "created_at": turn_completed_event.get("created_at"),
                "completed_at": turn_completed_event.get("created_at"),
                "final_text": final_text,
            }

    def _prepare_run_turn_prelude(
        self,
        *,
        user_text: str,
        thread_id: str | None,
        level: str | None,
        image_paths: list[str | Path] | None,
        cancel_event: asyncio.Event | None,
    ) -> RunTurnPrelude:
        self._raise_if_cancelled(cancel_event)
        self.refresh_config(force=True)
        if thread_id is None:
            raise ValueError("thread_id is required after run_turn creates the thread")
        system_instructions = self._system_instructions_for_turn(thread_id)
        turn_id = new_id("turn")
        should_generate_title = self._should_generate_title(thread_id)
        turn_input = self._prepare_turn_input(thread_id, level=level)
        input_items = turn_input.input_items
        request_input_items = turn_input.request_input_items()
        pre_user_items = self._pre_user_context_items(thread_id)
        turn_started_event = self.thread_store.append(thread_id, "turn.started", turn_id=turn_id)
        user_item = message_item("user", user_text)
        self.thread_store.append(thread_id, "item.user", turn_id=turn_id, item=user_item)

        # ``_reconstruct_input`` already places persisted post-compaction
        # context ahead of the compacted history. If this turn emits additional
        # epoch context (for example the first rules/runtime update after the
        # checkpoint), insert it at the same front-of-epoch anchor.
        if pre_user_items and self._has_compaction(thread_id):
            self._insert_pre_user_context_before_history(input_items, pre_user_items)
            if turn_input.request_previous_response_id() is None:
                self._insert_pre_user_context_before_history(request_input_items, pre_user_items)
            else:
                request_input_items.extend(pre_user_items)
        else:
            input_items.extend(pre_user_items)
            request_input_items.extend(pre_user_items)
        turn_input.pending_items.extend(pre_user_items)
        input_items.append(user_item)
        request_input_items.append(user_item)
        turn_input.pending_items.append(user_item)

        image_events: list[dict[str, Any]] = []
        for image_path in image_paths or []:
            attachment = self.attachments.register_image(
                image_path,
                cwd=self.project_root,
                thread_id=thread_id,
                note="pasted from clipboard",
            )
            payload = attachment.to_event_payload()
            self.thread_store.append(
                thread_id,
                "item.image_attachment",
                turn_id=turn_id,
                attachment=payload,
            )
            image_item = image_message_item(payload)
            input_items.append(image_item)
            request_input_items.append(image_item)
            turn_input.pending_items.append(image_item)
            image_events.append(
                {
                    "type": "image.attachment",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "attachment": payload,
                }
            )

        self._warm_model_backend_for_level(level)
        if should_generate_title:
            title_level = self.config.runtime.title_generation.model_level or level
            self._warm_model_backend_for_level(title_level)

        return RunTurnPrelude(
            thread_id=thread_id,
            turn_id=turn_id,
            system_instructions=system_instructions,
            should_generate_title=should_generate_title,
            turn_input=turn_input,
            input_items=input_items,
            request_input_items=request_input_items,
            turn_started_event=turn_started_event,
            image_events=image_events,
        )

    def _warm_model_backend_for_level(self, level: str | None) -> None:
        if not self._model_client_uses_builtin_lazy_provider_imports():
            return
        try:
            api = self.config.model_for_level(level).api
        except Exception:
            return
        try:
            if api == "anthropic_messages":
                module = importlib.import_module("uv_agent.model.anthropic")
                module.anthropic_sdk_param_keys()
                return
            if api == "chat_completions":
                module = importlib.import_module("uv_agent.model.chat")
                importlib.import_module("uv_agent.model.openai_sdk")
                module.chat_completions_sdk_param_keys()
                return
            module = importlib.import_module("uv_agent.model.responses")
            importlib.import_module("uv_agent.model.openai_sdk")
            module.responses_sdk_param_keys()
        except Exception:
            return

    def _model_client_uses_builtin_lazy_provider_imports(self) -> bool:
        client_type = type(self.model_client)
        return (
            client_type.__module__ == "uv_agent.model.client"
            and client_type.__name__ == "UnifiedModelClient"
        )

    async def retry_turn(
        self,
        *,
        thread_id: str,
        level: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        self.refresh_config(force=True)
        with self.thread_store.lock_thread(thread_id):
            retry_state = self._prepare_retry_input(thread_id, level=level)
            system_instructions = self._system_instructions_for_turn(thread_id)
            turn_id = new_id("turn")
            turn_started_event = self.thread_store.append(thread_id, "turn.started", turn_id=turn_id, retry=True)
            self.thread_store.append(thread_id, "turn.retry", turn_id=turn_id)
            final_text = ""
            stream_state = StreamResponseState()
            try:
                if retry_state.pending_tool_calls:
                    round_attachments: list[dict[str, Any]] = []
                    for call_index, call in enumerate(retry_state.pending_tool_calls):
                        async for tool_event in self._stream_tool_call_for_turn(
                            call=call,
                            call_index=call_index,
                            thread_id=thread_id,
                            turn_id=turn_id,
                            turn_started_at=turn_started_event.get("created_at"),
                            cancel_event=cancel_event,
                        ):
                            public_event = {key: value for key, value in tool_event.items() if key != "_result"}
                            yield public_event
                            if tool_event.get("type") != "tool.output":
                                continue
                            result = self._tool_result_from_event(tool_event, public_event)
                            self.thread_store.append(
                                thread_id,
                                "item.tool_output",
                                turn_id=turn_id,
                                item=result.tool_output,
                            )
                            retry_state.input_items.append(result.tool_output)
                            retry_state.pending_items.append(result.tool_output)
                            round_attachments.extend(result.attachments)
                    if round_attachments:
                        for attachment in round_attachments:
                            self.thread_store.append(
                                thread_id,
                                "item.image_attachment",
                                turn_id=turn_id,
                                source="tool",
                                attachment=attachment,
                            )
                        retry_state.input_items.extend(tool_attachment_context_items(round_attachments))
                    if self._will_compact_after_tool_results(
                        thread_id,
                        retry_state.input_items,
                        level=level,
                        instructions=system_instructions,
                    ):
                        yield self._compaction_started_event(thread_id, turn_id)
                    mid_turn_compaction = await self._maybe_compact_after_tool_results(
                        thread_id,
                        turn_id,
                        retry_state.input_items,
                        level=level,
                        instructions=system_instructions,
                    )
                    if mid_turn_compaction.token_warning_event is not None:
                        yield self._public_event(mid_turn_compaction.token_warning_event)
                    if mid_turn_compaction.result is not None:
                        yield self._compaction_completed_event(
                            thread_id,
                            turn_id,
                            mid_turn_compaction.result,
                        )
                        retry_state.input_items = self._input_after_compaction(thread_id, mid_turn_compaction.result)
                        retry_state.previous_response_id = None
                        retry_state.use_previous_response_id = False
                        retry_state.pending_items.clear()
                        retry_state.pending_tool_calls.clear()

                for _ in range(self.config.runtime.max_agent_rounds):
                    self._raise_if_cancelled(cancel_event)
                    async for event in self._stream_model_response_with_retries(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        turn_started_at=turn_started_event.get("created_at"),
                        input_items=retry_state.request_input_items(),
                        level=level,
                        instructions=system_instructions,
                        previous_response_id=retry_state.request_previous_response_id(),
                        stream_state=stream_state,
                        cancel_event=cancel_event,
                    ):
                        yield event
                    response = stream_state.require_response()
                    retry_state.input_items.extend(response.output)
                    retry_state.previous_response_id = response.id
                    retry_state.use_previous_response_id = bool(response.id and self._level_uses_responses_api(level))
                    retry_state.pending_items.clear()
                    stream_state.reset()
                    tool_calls = [item for item in response.output if item.get("type") == "function_call"]
                    if not tool_calls:
                        final_text = response.output_text
                        break
                    retry_state.pending_tool_calls = tool_calls
                    round_attachments = []
                    for call_index, call in enumerate(tool_calls):
                        async for tool_event in self._stream_tool_call_for_turn(
                            call=call,
                            call_index=call_index,
                            thread_id=thread_id,
                            turn_id=turn_id,
                            turn_started_at=turn_started_event.get("created_at"),
                            cancel_event=cancel_event,
                        ):
                            public_event = {key: value for key, value in tool_event.items() if key != "_result"}
                            yield public_event
                            if tool_event.get("type") != "tool.output":
                                continue
                            result = self._tool_result_from_event(tool_event, public_event)
                            self.thread_store.append(
                                thread_id,
                                "item.tool_output",
                                turn_id=turn_id,
                                item=result.tool_output,
                            )
                            retry_state.input_items.append(result.tool_output)
                            retry_state.pending_items.append(result.tool_output)
                            round_attachments.extend(result.attachments)
                    if round_attachments:
                        for attachment in round_attachments:
                            self.thread_store.append(
                                thread_id,
                                "item.image_attachment",
                                turn_id=turn_id,
                                source="tool",
                                attachment=attachment,
                            )
                        retry_state.input_items.extend(tool_attachment_context_items(round_attachments))
                        retry_state.pending_items.extend(tool_attachment_context_items(round_attachments))
                        retry_state.use_previous_response_id = False
                    if self._will_compact_after_tool_results(
                        thread_id,
                        retry_state.input_items,
                        level=level,
                        instructions=system_instructions,
                    ):
                        yield self._compaction_started_event(thread_id, turn_id)
                    mid_turn_compaction = await self._maybe_compact_after_tool_results(
                        thread_id,
                        turn_id,
                        retry_state.input_items,
                        level=level,
                        instructions=system_instructions,
                    )
                    if mid_turn_compaction.token_warning_event is not None:
                        yield self._public_event(mid_turn_compaction.token_warning_event)
                    if mid_turn_compaction.result is not None:
                        yield self._compaction_completed_event(
                            thread_id,
                            turn_id,
                            mid_turn_compaction.result,
                        )
                        retry_state.input_items = self._input_after_compaction(thread_id, mid_turn_compaction.result)
                        retry_state.previous_response_id = None
                        retry_state.use_previous_response_id = False
                        retry_state.pending_items.clear()
                        retry_state.pending_tool_calls.clear()
                else:
                    raise RuntimeError("Agent exceeded max_agent_rounds")
            except (asyncio.CancelledError, TurnInterrupted):
                partial_text = stream_state.partial_text
                if partial_text:
                    self.thread_store.append(thread_id, "item.assistant_partial", turn_id=turn_id, text=partial_text)
                reasoning_text = stream_state.partial_reasoning_text
                if reasoning_text:
                    self.thread_store.append(thread_id, "item.reasoning_partial", turn_id=turn_id, text=reasoning_text)
                self.thread_store.append(
                    thread_id,
                    "turn.interrupted",
                    turn_id=turn_id,
                    reason="user_interrupt",
                    partial_stream=stream_state.saw_stream_output,
                )
                yield {
                    "type": "turn.interrupted",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "reason": "user_interrupt",
                    "partial_stream": stream_state.saw_stream_output,
                }
                return
            except Exception as exc:
                error_event = self.thread_store.append(
                    thread_id,
                    "turn.error",
                    turn_id=turn_id,
                    error_type=exc.__class__.__name__,
                    message=str(exc) or repr(exc),
                    retryable=is_retryable_provider_error(exc),
                )
                yield {
                    "type": "turn.error",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_event.get("created_at"),
                    "created_at": error_event.get("created_at"),
                    "completed_at": error_event.get("created_at"),
                    "error_type": exc.__class__.__name__,
                    "message": str(exc) or repr(exc),
                    "retryable": is_retryable_provider_error(exc),
                }
                return

            completed_event = self.thread_store.append(thread_id, "turn.completed", turn_id=turn_id, final_text=final_text)
            yield {
                "type": "turn.completed",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "turn_started_at": turn_started_event.get("created_at"),
                "created_at": completed_event.get("created_at"),
                "completed_at": completed_event.get("created_at"),
                "final_text": final_text,
            }
            return

    def _should_generate_title(self, thread_id: str) -> bool:
        if not self.config.runtime.title_generation.enabled:
            return False
        metadata = self.thread_store.snapshot(thread_id).metadata
        if int(metadata.get("user_message_count") or 0) > 0:
            return False
        return self._thread_title_is_pending(metadata)

    def _thread_title_is_pending(self, metadata: dict[str, Any]) -> bool:
        if metadata.get("title_updated_at"):
            return False
        return is_default_thread_title(str(metadata.get("title") or ""))

    def _start_title_generation_task(
        self,
        thread_id: str,
        user_text: str,
        *,
        should_generate: bool,
        level: str | None,
    ) -> asyncio.Task[str | None] | None:
        if not should_generate:
            return None
        return asyncio.create_task(self._maybe_generate_title(thread_id, user_text, level=level))

    async def _finish_title_generation(
        self,
        title_task: asyncio.Task[str | None] | None,
    ) -> str | None:
        if title_task is None:
            return None
        try:
            return await title_task
        except asyncio.CancelledError:
            return None

    async def _maybe_generate_title(
        self,
        thread_id: str,
        user_text: str,
        *,
        level: str | None,
    ) -> str | None:
        try:
            title = await self._generate_thread_title(thread_id, user_text, level=level)
        except Exception:
            return None
        if not title or not self._thread_title_is_pending(self.thread_store.snapshot(thread_id).metadata):
            return None
        self.thread_store.update_title(thread_id, title, source="generated")
        return title

    async def _generate_thread_title(self, thread_id: str, user_text: str, *, level: str | None) -> str | None:
        title_level = self.config.runtime.title_generation.model_level or level
        response = await self.model_client.create_response(
            input_items=[
                message_item(
                    "user",
                    TITLE_GENERATION_PROMPT
                    + "\n\nUser message:\n"
                    + user_text.strip(),
                )
            ],
            level=title_level,
            tools=[],
            instructions="Generate a short thread title. Return only the title.",
        )
        self._record_billing_charge(
            thread_id,
            None,
            response.usage,
            level=title_level,
            source="title_generation",
        )
        return clean_thread_title(response.output_text)

    async def _maybe_compact(
        self,
        thread_id: str,
        turn_id: str,
        input_items: list[dict[str, Any]],
        *,
        level: str | None,
        instructions: str,
    ) -> CompactionDecision:
        return await self._compact_if_needed(
            thread_id,
            turn_id,
            input_items,
            level=level,
            instructions=instructions,
        )

    async def _maybe_compact_after_tool_results(
        self,
        thread_id: str,
        turn_id: str,
        input_items: list[dict[str, Any]],
        *,
        level: str | None,
        instructions: str,
    ) -> CompactionDecision:
        if not self._has_tool_output(input_items):
            return CompactionDecision()
        bridged_input = copy.deepcopy(input_items)
        bridge_item = assistant_output_item(POST_TOOL_COMPACTION_BRIDGE)
        bridged_input.append(bridge_item)
        result = await self._compact_if_needed(
            thread_id,
            turn_id,
            bridged_input,
            level=level,
            instructions=instructions,
            allow_last_tool_output_truncation=True,
            pre_compaction_event={
                "type": "item.assistant",
                "turn_id": turn_id,
                "text": POST_TOOL_COMPACTION_BRIDGE,
            },
        )
        if result.result is None:
            return result
        # Keep the in-memory turn state aligned with the event persisted just
        # before the compaction checkpoint.
        input_items.append(bridge_item)
        return result

    def _will_compact(
        self,
        thread_id: str,
        input_items: list[dict[str, Any]],
        *,
        level: str | None,
        instructions: str,
    ) -> bool:
        """Cheaply predict whether the next compaction check will call a model."""

        return self._compaction_should_run(
            thread_id,
            input_items,
            level=level,
            instructions=instructions,
        )

    def _will_compact_after_tool_results(
        self,
        thread_id: str,
        input_items: list[dict[str, Any]],
        *,
        level: str | None,
        instructions: str,
    ) -> bool:
        """Predict mid-turn compaction without mutating the turn input list."""

        if not self._has_tool_output(input_items):
            return False
        bridged_input = copy.deepcopy(input_items)
        bridged_input.append(assistant_output_item(POST_TOOL_COMPACTION_BRIDGE))
        return self._compaction_should_run(
            thread_id,
            bridged_input,
            level=level,
            instructions=instructions,
        )

    def _compaction_should_run(
        self,
        thread_id: str,
        input_items: list[dict[str, Any]],
        *,
        level: str | None,
        instructions: str,
    ) -> bool:
        """Return whether compaction thresholds currently require a checkpoint."""

        if not self.config.runtime.compression.enabled:
            return False
        compact_level = self.config.runtime.compression.model_level or level
        model = self.config.model_for_level(compact_level)
        token_count = self._compaction_token_count(thread_id, input_items, instructions=instructions)
        trigger_tokens = int(model.context_window_tokens * self.config.runtime.compression.trigger_ratio)
        return (
            token_count.tokens >= self.config.runtime.compression.min_tokens
            and token_count.tokens >= trigger_tokens
        )

    async def _compact_if_needed(
        self,
        thread_id: str,
        turn_id: str,
        input_items: list[dict[str, Any]],
        *,
        level: str | None,
        instructions: str,
        allow_last_tool_output_truncation: bool = False,
        pre_compaction_event: dict[str, Any] | None = None,
    ) -> CompactionDecision:
        if not self.config.runtime.compression.enabled:
            return CompactionDecision()
        compact_level = self.config.runtime.compression.model_level or level
        model = self.config.model_for_level(compact_level)
        token_count = self._compaction_token_count(thread_id, input_items, instructions=instructions)
        trigger_tokens = int(model.context_window_tokens * self.config.runtime.compression.trigger_ratio)
        token_warning_event = None
        if token_count.source == "estimate" and token_count.tokens >= self.config.runtime.compression.min_tokens:
            token_warning_event = self.thread_store.append(
                thread_id,
                "thread.token_estimation_warning",
                turn_id=turn_id,
                message=(
                    "Provider token usage is unavailable; context compaction is "
                    "using a local estimate and may fail calls or compact too late."
                ),
                used_tokens=token_count.tokens,
                threshold_tokens=trigger_tokens,
                context_window_tokens=model.context_window_tokens,
            )
        if token_count.tokens < self.config.runtime.compression.min_tokens:
            return CompactionDecision(token_warning_event=token_warning_event)
        if token_count.tokens < trigger_tokens:
            return CompactionDecision(token_warning_event=token_warning_event)
        if pre_compaction_event is not None:
            event_type = str(pre_compaction_event.get("type") or "")
            payload = {key: value for key, value in pre_compaction_event.items() if key != "type"}
            self.thread_store.append(thread_id, event_type, **payload)
        compact_input = copy.deepcopy(input_items)
        compact_input.append(self._compaction_trigger_item())
        truncated_last_tool_output = False
        if allow_last_tool_output_truncation:
            compact_input, truncated_last_tool_output = self._fit_compaction_input_by_truncating_last_tool_output(
                compact_input,
                context_window_tokens=model.context_window_tokens,
            )
        response = await self.model_client.create_response(
            input_items=compact_input,
            level=compact_level,
            tools=[PYTHON_TOOL],
            instructions=instructions,
        )
        replacement_input = self._compaction_replacement_input(input_items, response)
        context_state = self._latest_context_state(thread_id)
        self.thread_store.append(
            thread_id,
            "item.compaction",
            turn_id=turn_id,
            text=response.output_text,
            output=response.output,
            replacement_input=replacement_input,
            context_state=context_state,
            usage=response.usage,
        )
        self._record_billing_charge(
            thread_id,
            turn_id,
            response.usage,
            level=compact_level,
            source="compaction",
        )
        self._reset_rule_epoch(thread_id)
        return CompactionDecision(
            result=CompactionResult(
                replacement_input=replacement_input,
                text=response.output_text,
                truncated_last_tool_output=truncated_last_tool_output,
            ),
            token_warning_event=token_warning_event,
        )

    @staticmethod
    def _compaction_started_event(thread_id: str, turn_id: str) -> dict[str, Any]:
        """Public stream item emitted before a potentially slow compaction call."""

        return {"type": "compaction.started", "thread_id": thread_id, "turn_id": turn_id}

    @staticmethod
    def _compaction_completed_event(
        thread_id: str,
        turn_id: str,
        result: CompactionResult,
    ) -> dict[str, Any]:
        """Public stream item emitted after compaction persists a checkpoint."""

        return {
            "type": "compaction.completed",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "text": result.text,
            "truncated_last_tool_output": result.truncated_last_tool_output,
        }

    def _compaction_token_count(
        self,
        thread_id: str,
        input_items: list[dict[str, Any]],
        *,
        instructions: str,
    ) -> TokenCountResult:
        """Return the authoritative token count used for compaction triggers.

        Provider usage is the source of truth because it reflects the exact
        server-side tokenizer and hidden request framing. We always trust the
        most recent provider-reported usage when any exists in the open epoch,
        even if extra events (tool outputs, user turns, context updates) were
        appended afterwards: the very next model call will refresh the count.
        The local estimate is only used when no provider usage exists at all
        (e.g. first turn before any model call, or providers that omit usage),
        and that case is the only one that surfaces the estimation warning.
        """

        provider_tokens = self._latest_compaction_provider_tokens(thread_id)
        if provider_tokens is not None:
            return TokenCountResult(provider_tokens, "provider")

        return TokenCountResult(
            self._estimate_compaction_tokens(input_items, instructions=instructions),
            "estimate",
        )

    def _latest_compaction_provider_tokens(self, thread_id: str) -> int | None:
        """Return the latest provider-reported usage tokens for the open epoch."""

        snap = self.thread_store.snapshot(thread_id)
        for event in reversed(snap.events_after_compaction):
            if event.get("type") not in {"item.model_response", "item.compaction"}:
                continue
            used = usage_token_count(event.get("usage") or {})
            if used is not None:
                return used

        compaction = snap.latest_compaction
        if compaction is not None:
            used = usage_token_count(compaction.get("usage") or {})
            if used is not None:
                return used
        return None

    def _estimate_compaction_tokens(
        self,
        input_items: list[dict[str, Any]],
        *,
        instructions: str,
    ) -> int:
        """Estimate the full request size when provider usage is unavailable."""

        estimated_items = list(input_items)
        if instructions:
            estimated_items = [message_item("system", instructions), *estimated_items]
        return estimate_tokens(estimated_items)

    @staticmethod
    def _public_event(event: dict[str, Any]) -> dict[str, Any]:
        """Return a streamed event payload without JSONL bookkeeping fields."""

        return {key: value for key, value in event.items() if not key.startswith("_")}

    @staticmethod
    def _has_tool_output(input_items: list[dict[str, Any]]) -> bool:
        return any(item.get("type") == "function_call_output" for item in input_items)

    def _fit_compaction_input_by_truncating_last_tool_output(
        self,
        input_items: list[dict[str, Any]],
        *,
        context_window_tokens: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Shrink only the last tool output when a compaction request is too large.

        Tool output is the common mid-turn source of sudden context growth. The
        first emergency mechanism intentionally stays conservative: keep the
        full historical record on disk, but shorten the model-facing copy used
        for summarization. If old conversation alone is too large, this function
        leaves that harder case visible for a later, broader compaction policy.
        """

        budget = max(1, context_window_tokens - 5_000)
        current_tokens = estimate_tokens(input_items)
        if current_tokens <= budget:
            return input_items, False

        tool_index = self._last_tool_output_index(input_items)
        if tool_index is None:
            return input_items, False

        tool_item = input_items[tool_index]
        raw_output = tool_item.get("output")
        if not isinstance(raw_output, str) or not raw_output:
            return input_items, False

        # Token estimation is currently character based. Convert the excess into
        # a conservative character reduction and retry a few times because JSON
        # escaping and marker metadata make the final size non-linear.
        output_budget_chars = max(0, len(raw_output) - ((current_tokens - budget) * 4))
        truncated_output = raw_output
        for _ in range(6):
            truncated_output = truncate_tool_output_for_compaction(raw_output, output_budget_chars)
            candidate = copy.deepcopy(input_items)
            candidate[tool_index] = {**copy.deepcopy(tool_item), "output": truncated_output}
            if estimate_tokens(candidate) <= budget:
                return candidate, truncated_output != raw_output
            output_budget_chars = max(0, int(output_budget_chars * 0.75) - 1_024)

        candidate = copy.deepcopy(input_items)
        candidate[tool_index] = {
            **copy.deepcopy(tool_item),
            "output": truncate_tool_output_for_compaction(raw_output, 0),
        }
        return candidate, candidate[tool_index].get("output") != raw_output

    @staticmethod
    def _last_tool_output_index(input_items: list[dict[str, Any]]) -> int | None:
        for index in range(len(input_items) - 1, -1, -1):
            if input_items[index].get("type") == "function_call_output":
                return index
        return None

    @staticmethod
    def _tool_result_from_event(
        event: dict[str, Any],
        public_event: dict[str, Any],
    ) -> ToolCallTurnResult:
        """Return the internal tool result attached to a streamed tool.output."""

        result = event.get("_result")
        if isinstance(result, ToolCallTurnResult):
            return result
        return ToolCallTurnResult(
            tool_output=event["output"],
            attachments=[],
            started_event=public_event,
            output_event=public_event,
        )

    def _compaction_trigger_item(self) -> dict[str, Any]:
        return compaction_trigger_item()

    def _compaction_replacement_input(
        self,
        input_items: list[dict[str, Any]],
        response: ModelResponse,
    ) -> list[dict[str, Any]]:
        return compaction_replacement_input(input_items, response)

    def _retained_user_messages_after_compaction(self, input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return retained_user_messages_after_compaction(input_items)

    @staticmethod
    def _retain_item_after_compaction(item: dict[str, Any]) -> bool:
        return retain_item_after_compaction(item)

    def _input_after_compaction(
        self,
        thread_id: str,
        result: CompactionResult,
    ) -> list[dict[str, Any]]:
        """Build post-compaction model input for continuing in the same turn.

        Persisted compaction events intentionally store only the replacement
        history. For an immediate mid-turn continuation we must also prepend the
        freshly re-emitted epoch context, mirroring the ordering used when a
        later turn reconstructs from disk.
        """

        return self._pre_user_context_items(thread_id) + copy.deepcopy(result.replacement_input)

    @staticmethod
    def _compaction_replacement_items(compaction: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the model-input replacement for a persisted compaction event."""

        replacement_input = compaction.get("replacement_input")
        if isinstance(replacement_input, list):
            return normalize_compaction_replacement_input(replacement_input)
        summary = str(compaction.get("text") or "").strip()
        if not summary:
            return []
        return [compaction_summary_item(summary)]

    @staticmethod
    def _insert_pre_user_context_before_history(
        input_items: list[dict[str, Any]],
        pre_user_items: list[dict[str, Any]],
    ) -> None:
        """Insert current epoch context before compacted retained history.

        After a compaction checkpoint, reconstructed input begins with the
        retained messages and summary. The next turn may emit fresh dynamic
        context before adding the new user message; those environment messages
        belong ahead of the compacted history rather than at the tail.
        """

        if not pre_user_items:
            return
        insert_at = 0
        while insert_at < len(input_items) and AgentEngine._is_pre_user_context_item(input_items[insert_at]):
            insert_at += 1
        input_items[insert_at:insert_at] = pre_user_items

    @staticmethod
    def _is_pre_user_context_item(item: dict[str, Any]) -> bool:
        if item.get("type") != "message" or item.get("role") != "user":
            return False
        text = message_item_text(item)
        return (
            "<runtime_environment>" in text
            or "<model_levels>" in text
            or "<runtime_helpers>" in text
            or "<workspace_rules" in text
            or "<workspace_rule_index>" in text
            or "<active_cwd_notice>" in text
            or "<available_skills>" in text
            or "<available_mcp_servers>" in text
            or "<context_update" in text
        )

    @staticmethod
    def _is_replayable_input_event(event: dict[str, Any]) -> bool:
        """Return whether an event contributes ordinary conversation input."""

        return event.get("type") in {
            "item.user",
            "item.assistant",
            "item.model_response",
            "item.tool_output",
            "item.image_attachment",
            "turn.interrupted",
        }

    async def _handle_tool_call(
        self,
        call: dict[str, Any],
        thread_id: str,
        turn_id: str,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """Execute a tool call and return the final model/display payloads."""

        final: ToolCallTurnResult | None = None
        async for event in self._stream_tool_call(
            call,
            thread_id,
            turn_id,
            turn_started_at=None,
            tool_call_index=0,
            cancel_event=cancel_event,
        ):
            if event.get("type") == "tool.output":
                final = event["_result"]
        if final is None:
            raise RuntimeError("Tool call did not produce a final output")
        return final.tool_output, final.attachments, final.output_event["output"]

    async def _stream_tool_call(
        self,
        call: dict[str, Any],
        thread_id: str,
        turn_id: str,
        *,
        turn_started_at: object,
        tool_call_index: int,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run a tool call and yield partial/final UI events as they become available."""

        def output_event(tool_output: dict[str, Any], result: ToolCallTurnResult) -> dict[str, Any]:
            """Build a public tool.output event with the private result attached."""

            return {
                "type": "tool.output",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "turn_started_at": turn_started_at,
                "call": call,
                "tool_call_index": tool_call_index,
                "output": tool_output,
                "_result": result,
            }

        if call.get("name") != "run_python":
            output = {"error": f"Unsupported tool: {call.get('name')}"}
            tool_output = function_output(call, output)
            result = ToolCallTurnResult(
                tool_output=tool_output,
                attachments=[],
                started_event={},
                output_event={"output": tool_output},
            )
            yield output_event(tool_output, result)
            return
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            output = {"error": f"Invalid tool arguments JSON: {exc}"}
            tool_output = function_output(call, output)
            result = ToolCallTurnResult(
                tool_output=tool_output,
                attachments=[],
                started_event={},
                output_event={"output": tool_output},
            )
            yield output_event(tool_output, result)
            return

        thread_kind = str(self.thread_store.snapshot(thread_id).metadata.get("kind") or "thread")
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            output = {"error": "run_python requires code"}
            tool_output = function_output(call, output)
            result = ToolCallTurnResult(
                tool_output=tool_output,
                attachments=[],
                started_event={},
                output_event={"output": tool_output},
            )
            yield output_event(tool_output, result)
            return

        request = PythonRunRequest(
            code=code,
            script_args=list(args.get("script_args") or []),
            timeout_s=float(args.get("timeout_s") or self.config.runner.default_timeout_s),
            cwd=self._active_cwd(thread_id),
            thread_id=thread_id,
            thread_kind=thread_kind,
            turn_id=turn_id,
            cancel_event=cancel_event,
        )
        stream_run = getattr(self.runner, "stream_run", None)
        if stream_run is None:
            result = await self.runner.run(request)
            runner_events = [RunnerEvent("run.completed", {"result": result, "returncode": result.returncode})]
        else:
            runner_events = stream_run(request)
        async for runner_event in _ensure_async_runner_events(runner_events):
            if runner_event.type == "run.partial":
                partial_payload = runner_event.data["result"].to_payload()
                partial_payload["partial"] = True
                partial_payload["partial_reason"] = runner_event.data.get("reason")
                partial_payload["call_id"] = call.get("call_id")
                visible_partial_events = [
                    event
                    for event in partial_payload.get("events", [])
                    if isinstance(event, dict) and event.get("kind") != "enter_dir"
                ]
                partial_payload["events"] = visible_partial_events
                yield {
                    "type": "tool.partial",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_at,
                    "call": call,
                    "tool_call_index": tool_call_index,
                    "output": function_output(call, partial_payload),
                }
                continue
            if runner_event.type != "run.completed":
                continue
            result = runner_event.data["result"]
            if result.interrupted:
                raise TurnInterrupted()
            rule_events, visible_events = self._process_runner_events(
                result.events,
                thread_id=thread_id,
                turn_id=turn_id,
            )
            payload = result.to_payload()
            payload["events"] = visible_events
            if rule_events:
                payload["rules_loaded"] = rule_events
            attachments = self._register_look_at_events(
                visible_events,
                thread_id=thread_id,
                turn_id=turn_id,
                cwd=self._active_cwd(thread_id),
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
            tool_output = function_output(call, model_tool_payload(payload))
            display_output = function_output(call, payload)
            result_payload = ToolCallTurnResult(
                tool_output=tool_output,
                attachments=attachments,
                started_event={},
                output_event={"output": display_output},
            )
            yield output_event(display_output, result_payload)
            return
        raise RuntimeError("Runner did not emit run.completed")

    async def _execute_tool_call_for_turn(
        self,
        *,
        call: dict[str, Any],
        call_index: int,
        thread_id: str,
        turn_id: str,
        turn_started_at: object,
        cancel_event: asyncio.Event | None,
    ) -> ToolCallTurnResult:
        """Backward-compatible helper that waits for the final tool result."""

        final: ToolCallTurnResult | None = None
        async for event in self._stream_tool_call_for_turn(
            call=call,
            call_index=call_index,
            thread_id=thread_id,
            turn_id=turn_id,
            turn_started_at=turn_started_at,
            cancel_event=cancel_event,
        ):
            if event.get("type") == "tool.output":
                final = event["_result"]
        if final is None:
            raise RuntimeError("Tool call did not produce a final output")
        return final

    async def _stream_tool_call_for_turn(
        self,
        *,
        call: dict[str, Any],
        call_index: int,
        thread_id: str,
        turn_id: str,
        turn_started_at: object,
        cancel_event: asyncio.Event | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield tool.started, zero or more tool.partial events, then tool.output."""

        self._raise_if_cancelled(cancel_event)
        started_event = {
            "type": "tool.started",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "turn_started_at": turn_started_at,
            "call": call,
            "tool_call_index": call_index,
        }
        yield started_event
        async for event in self._stream_tool_call(
            call,
            thread_id,
            turn_id,
            turn_started_at=turn_started_at,
            tool_call_index=call_index,
            cancel_event=cancel_event,
        ):
            if event.get("type") == "tool.output":
                result_value = event.get("_result")
                if not isinstance(result_value, ToolCallTurnResult):
                    result_value = ToolCallTurnResult(
                        tool_output=event["output"],
                        attachments=[],
                        started_event=started_event,
                        output_event={key: value for key, value in event.items() if key != "_result"},
                    )
                result = ToolCallTurnResult(
                    tool_output=result_value.tool_output,
                    attachments=result_value.attachments,
                    started_event=started_event,
                    output_event={key: value for key, value in event.items() if key != "_result"},
                )
                event["_result"] = result
            yield event

    @staticmethod
    def _raise_if_cancelled(cancel_event: asyncio.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise TurnInterrupted()

    async def _stream_and_persist_model_response(
        self,
        *,
        thread_id: str,
        turn_id: str,
        turn_started_at: object,
        input_items: list[dict[str, Any]],
        level: str | None,
        instructions: str,
        previous_response_id: str | None,
        stream_state: StreamResponseState,
        cancel_event: asyncio.Event | None,
    ) -> AsyncIterator[dict[str, Any]]:
        stream_state.response = None
        async for stream_event in self._stream_response_until_cancelled(
            input_items=input_items,
            level=level,
            instructions=instructions,
            cancel_event=cancel_event,
            previous_response_id=previous_response_id,
        ):
            self._raise_if_cancelled(cancel_event)
            if stream_event.type == "text_delta" and stream_event.text:
                stream_state.saw_stream_output = True
                stream_state.assistant_parts.append(stream_event.text)
                yield {
                    "type": "assistant.delta",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_at,
                    "text": stream_event.text,
                }
            elif stream_event.type == "reasoning_delta" and stream_event.text:
                stream_state.saw_stream_output = True
                stream_state.reasoning_parts.append(stream_event.text)
                yield {
                    "type": "assistant.reasoning_delta",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_at,
                    "text": stream_event.text,
                }
            elif stream_event.type == "tool_call_delta" and stream_event.tool_call:
                stream_state.saw_stream_output = True
                yield {
                    "type": "tool.delta",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_at,
                    "tool_call": stream_event.tool_call,
                }
            elif stream_event.type == "completed":
                stream_state.response = stream_event.response
        self._raise_if_cancelled(cancel_event)
        response = stream_state.require_response()
        completed_text_delta = completion_text_delta(
            response.output_text,
            "".join(stream_state.assistant_parts),
        )
        if completed_text_delta:
            stream_state.assistant_parts.append(completed_text_delta)
            yield {
                "type": "assistant.delta",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "turn_started_at": turn_started_at,
                "text": completed_text_delta,
            }
        reasoning_text = response.reasoning_text or stream_state.partial_reasoning_text
        self.thread_store.append(
            thread_id,
            "item.model_response",
            turn_id=turn_id,
            model_api=self._model_api_for_level(level),
            response_id=response.id,
            output=response.output,
            usage=response.usage,
            reasoning_text=reasoning_text,
        )
        billing_charge = self._record_billing_charge(
            thread_id,
            turn_id,
            response.usage,
            level=level,
            source="model_response",
        )
        yield {
            "type": "model.response",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "turn_started_at": turn_started_at,
            "response": response,
            "reasoning_text": reasoning_text,
            "billing_charge": billing_charge,
        }

    def _record_billing_charge(
        self,
        thread_id: str,
        turn_id: str | None,
        usage: dict[str, Any],
        *,
        level: str | None,
        source: str,
    ) -> dict[str, Any] | None:
        """Persist one incremental model-call charge when pricing is configured."""

        try:
            model = self.config.model_for_level(level)
        except Exception:
            return None
        charge = billing_charge_for_usage(self.config, model, usage or {}, level=level)
        if charge is None:
            return None
        payload = charge.to_event_payload(source=source, turn_id=turn_id)
        self.thread_store.append(
            thread_id,
            "thread.billing_accumulated",
            **payload,
        )
        total = billing_total_from_metadata(
            self.thread_store.snapshot(thread_id).metadata,
            preferred_currency=charge.currency,
        )
        if total is not None:
            payload["total"] = decimal_to_string(total[0])
            payload["total_currency"] = total[1]
        return payload

    async def _stream_model_response_with_retries(
        self,
        *,
        thread_id: str,
        turn_id: str,
        turn_started_at: object,
        input_items: list[dict[str, Any]],
        level: str | None,
        instructions: str,
        previous_response_id: str | None,
        stream_state: StreamResponseState,
        cancel_event: asyncio.Event | None,
    ) -> AsyncIterator[dict[str, Any]]:
        retry = self.config.runtime.stream_retry
        for retry_index in range(retry.max_retries + 1):
            self._raise_if_cancelled(cancel_event)
            try:
                async for event in self._stream_and_persist_model_response(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    turn_started_at=turn_started_at,
                    input_items=copy.deepcopy(input_items),
                    level=level,
                    instructions=instructions,
                    previous_response_id=previous_response_id,
                    stream_state=stream_state,
                    cancel_event=cancel_event,
                ):
                    yield event
                return
            except Exception as exc:
                if not self._should_retry_model_stream_error(exc) or retry_index >= retry.max_retries:
                    raise
                stream_state.reset()
                attempt = retry_index + 1
                delay_s = self._stream_retry_delay(attempt)
                retry_event = self.thread_store.append(
                    thread_id,
                    "turn.stream_retry",
                    turn_id=turn_id,
                    attempt=attempt,
                    max_attempts=retry.max_retries,
                    delay_s=delay_s,
                    error_type=exc.__class__.__name__,
                    message=str(exc) or repr(exc),
                )
                yield {
                    "type": "model.stream_retry",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_at,
                    "created_at": retry_event.get("created_at"),
                    "attempt": attempt,
                    "max_attempts": retry.max_retries,
                    "delay_s": delay_s,
                    "error_type": exc.__class__.__name__,
                    "message": str(exc) or repr(exc),
                }
                await self._sleep_stream_retry(delay_s, cancel_event=cancel_event)

    @staticmethod
    def _should_retry_model_stream_error(exc: BaseException) -> bool:
        return isinstance(exc, EmptyModelStreamError) or is_retryable_provider_error(exc)

    def _stream_retry_delay(self, attempt: int) -> float:
        retry = self.config.runtime.stream_retry
        delay = retry.base * (retry.factor ** (attempt - 1))
        delay = min(delay, retry.max)
        if retry.jitter:
            jitter = max(0.0, retry.jitter)
            delay *= random.uniform(1.0 - jitter, 1.0 + jitter)
        return max(0.0, delay)

    async def _sleep_stream_retry(
        self,
        delay_s: float,
        *,
        cancel_event: asyncio.Event | None,
    ) -> None:
        if delay_s <= 0:
            self._raise_if_cancelled(cancel_event)
            return
        if cancel_event is None:
            await _sleep_stream_retry(delay_s)
            return
        sleep_task = asyncio.create_task(_sleep_stream_retry(delay_s))
        cancel_task = asyncio.create_task(cancel_event.wait())
        try:
            done, pending = await asyncio.wait(
                {sleep_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done:
                raise TurnInterrupted()
            if sleep_task in done:
                sleep_task.result()
        finally:
            for task in (sleep_task, cancel_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sleep_task, cancel_task, return_exceptions=True)
        self._raise_if_cancelled(cancel_event)

    async def _stream_response_until_cancelled(
        self,
        *,
        input_items: list[dict[str, Any]],
        level: str | None,
        instructions: str,
        cancel_event: asyncio.Event | None,
        previous_response_id: str | None = None,
    ) -> AsyncIterator[Any]:
        stream = self.model_client.stream_response(
            input_items=input_items,
            level=level,
            tools=[PYTHON_TOOL],
            instructions=instructions,
            previous_response_id=previous_response_id,
        )
        iterator = stream.__aiter__()
        cancel_task: asyncio.Task[bool] | None = None
        if cancel_event is not None:
            cancel_task = asyncio.create_task(cancel_event.wait())
        try:
            while True:
                next_task = asyncio.create_task(_await_next_stream_event(iterator.__anext__()))
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
                close_result = aclose()
                if inspect.isawaitable(close_result):
                    await _await_next_stream_event(close_result)

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
            attachments.append(attachment.to_event_payload())
        return attachments

    def _process_runner_events(
        self,
        events: list[dict[str, Any]],
        *,
        thread_id: str,
        turn_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        visible_events: list[dict[str, Any]] = []
        rules_loaded: list[dict[str, Any]] = []
        for event in events:
            if event.get("kind") == "enter_dir":
                entered = self._handle_enter_dir_event(event, thread_id=thread_id, turn_id=turn_id)
                if entered is not None:
                    visible_events.append({"kind": "cwd", "cwd": self._relative_to_project(entered)})
                    rules_loaded.extend(
                        self._load_unseen_rules_for_dir(thread_id, entered, source="tool_result")
                    )
                continue
            if event.get("kind") == "subagent.completed":
                self._record_subagent_billing_from_event(event, thread_id=thread_id, turn_id=turn_id)
            visible_events.append(event)
        return rules_loaded, visible_events

    def _record_subagent_billing_from_event(
        self,
        event: dict[str, Any],
        *,
        thread_id: str,
        turn_id: str,
    ) -> None:
        """Roll retained ask/subagent model costs into the parent thread total."""

        subthread_id = str(event.get("thread_id") or "")
        if not subthread_id or subthread_id == thread_id:
            return
        try:
            metadata = self.thread_store.snapshot(subthread_id).metadata
        except (OSError, ValueError, FileNotFoundError):
            return
        total = billing_total_from_metadata(
            metadata,
            preferred_currency=self.config.pricing.currency,
        )
        if total is None or total[0] == 0:
            return
        amount, currency = total
        self.thread_store.append(
            thread_id,
            "thread.billing_accumulated",
            turn_id=turn_id,
            source="subagent",
            subthread_id=subthread_id,
            amount=decimal_to_string(amount),
            currency=currency,
        )

    def _handle_enter_dir_event(
        self,
        event: dict[str, Any],
        *,
        thread_id: str,
        turn_id: str,
    ) -> Path | None:
        raw_cwd = event.get("cwd")
        if not isinstance(raw_cwd, str) or not raw_cwd:
            return None
        try:
            cwd = Path(raw_cwd).resolve()
        except OSError:
            return None
        if not self._is_within_project(cwd):
            return None
        state = self._rule_state(thread_id)
        state.active_cwd = cwd
        self.thread_store.append(
            thread_id,
            "thread.cwd_updated",
            turn_id=turn_id,
            cwd=str(cwd),
        )
        return cwd

    def _prepare_turn_input(self, thread_id: str, *, level: str | None) -> TurnInputState:
        snapshot = self.thread_store.snapshot(thread_id)
        input_items = self._reconstruct_input(thread_id, snapshot=snapshot)
        if not self._level_uses_responses_api(level):
            return TurnInputState(input_items=input_items)

        resume = self._latest_responses_resume(thread_id, snapshot=snapshot)
        if resume is None:
            return TurnInputState(input_items=input_items)

        _, previous_response_id, pending_items = resume
        return TurnInputState(
            input_items=input_items,
            previous_response_id=previous_response_id,
            use_previous_response_id=True,
            pending_items=pending_items,
        )

    def _prepare_retry_input(self, thread_id: str, *, level: str | None) -> RetryState:
        snapshot = self.thread_store.snapshot(thread_id)
        pending_tool_calls = self._pending_tool_calls_after_latest_response(snapshot.events_after_compaction)
        if pending_tool_calls:
            resume = self._latest_responses_resume(thread_id, snapshot=snapshot)
            previous_response_id = resume[1] if resume is not None else None
            return RetryState(
                input_items=self._reconstruct_input(thread_id, snapshot=snapshot),
                previous_response_id=previous_response_id,
                use_previous_response_id=bool(previous_response_id and self._level_uses_responses_api(level)),
                pending_items=[],
                pending_tool_calls=pending_tool_calls,
            )
        turn_input = self._prepare_turn_input(thread_id, level=level)
        return RetryState(
            input_items=turn_input.request_input_items(),
            previous_response_id=turn_input.previous_response_id,
            use_previous_response_id=turn_input.use_previous_response_id,
            pending_items=turn_input.pending_items,
        )

    def _pending_tool_calls_after_latest_response(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        latest_index = -1
        latest_output: list[dict[str, Any]] = []
        for index, event in enumerate(events):
            if event.get("type") == "item.model_response":
                latest_index = index
                latest_output = list(event.get("output") or [])
        if latest_index < 0:
            return []
        tool_calls = [copy.deepcopy(item) for item in latest_output if item.get("type") == "function_call"]
        if not tool_calls:
            return []
        completed_call_ids = {
            str((event.get("item") or {}).get("call_id") or "")
            for event in events[latest_index + 1 :]
            if event.get("type") == "item.tool_output"
        }
        return [
            call
            for call in tool_calls
            if str(call.get("call_id") or "") not in completed_call_ids
        ]

    def _system_instructions_for_turn(self, thread_id: str) -> str:
        snapshot = self.thread_store.snapshot(thread_id)
        existing = self._thread_system_instructions(thread_id, snapshot=snapshot)
        if existing is not None and not self._needs_system_instruction_refresh(thread_id, snapshot=snapshot):
            return existing
        instructions = self.system_instructions()
        self.thread_store.append(
            thread_id,
            "item.system_instructions",
            text=instructions,
            fingerprint=context_fingerprint(instructions),
            after_compaction=self._has_compaction(thread_id, snapshot=snapshot),
        )
        return instructions

    def _ensure_system_instructions(self, thread_id: str) -> str:
        return self._system_instructions_for_turn(thread_id)

    def _thread_system_instructions(self, thread_id: str, *, snapshot: ThreadSnapshot | None = None) -> str | None:
        events = (snapshot or self.thread_store.snapshot(thread_id)).events_after_compaction
        for event in reversed(events):
            if event.get("type") != "item.system_instructions":
                continue
            text = event.get("text")
            if isinstance(text, str) and text:
                return text
        return None

    def _needs_system_instruction_refresh(self, thread_id: str, *, snapshot: ThreadSnapshot | None = None) -> bool:
        snap = snapshot or self.thread_store.snapshot(thread_id)
        events = snap.events_after_compaction
        compaction = snap.latest_compaction
        if compaction is None:
            return False
        return not any(event.get("type") == "item.system_instructions" for event in events)

    def _level_uses_responses_api(self, level: str | None) -> bool:
        return self._model_api_for_level(level) == "responses"

    def _model_api_for_level(self, level: str | None) -> str | None:
        try:
            return self.config.model_for_level(level).api
        except Exception:
            return None

    def _latest_responses_resume(
        self,
        thread_id: str,
        *,
        snapshot: ThreadSnapshot | None = None,
    ) -> tuple[int, str, list[dict[str, Any]]] | None:
        events = (snapshot or self.thread_store.snapshot(thread_id)).events_after_compaction
        for index in range(len(events) - 1, -1, -1):
            event = events[index]
            if event.get("type") != "item.model_response":
                continue
            if event.get("model_api") not in {"responses", None}:
                continue
            response_id = str(event.get("response_id") or "")
            if not response_id:
                continue
            if event.get("model_api") is None and not response_id.startswith("resp"):
                continue
            expected_tool_outputs = sum(
                1
                for item in event.get("output") or []
                if item.get("type") == "function_call"
            )
            pending_items = self._input_items_after_event(
                events[index + 1 :],
                expected_tool_outputs=expected_tool_outputs,
            )
            if any(self._item_is_assistant_bridge(item) for item in pending_items):
                return None
            return index, response_id, pending_items
        return None

    def _item_is_assistant_bridge(self, item: dict[str, Any]) -> bool:
        return item.get("type") == "message" and item.get("role") == "assistant" and message_item_text(item) in {
            TOOL_ATTACHMENT_CONTEXT_BRIDGE,
            INTERRUPTED_TOOL_CONTEXT_BRIDGE,
            INTERRUPTED_STREAM_CONTEXT_BRIDGE,
        }

    def _input_items_after_event(
        self,
        events: list[dict[str, Any]],
        *,
        expected_tool_outputs: int = 0,
    ) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        pending_pre_user: list[dict[str, Any]] = []
        pending_tool_attachments: list[dict[str, Any]] = []
        pending_legacy_tool_attachments: list[dict[str, Any]] = []

        def flush_tool_attachments() -> None:
            nonlocal pending_tool_attachments
            if pending_tool_attachments:
                input_items.extend(tool_attachment_context_items(pending_tool_attachments))
                pending_tool_attachments = []

        for event in events:
            pre_user_item = self._pre_user_event_item(event)
            if pre_user_item is not None:
                flush_tool_attachments()
                pending_pre_user.append(pre_user_item)
            elif event.get("type") == "item.user":
                flush_tool_attachments()
                input_items.extend(pending_pre_user)
                pending_pre_user.clear()
                input_items.append(copy.deepcopy(event["item"]))
            elif event.get("type") == "item.assistant":
                flush_tool_attachments()
                text = str(event.get("text") or "")
                if text:
                    input_items.append(assistant_output_item(text))
            elif event.get("type") == "item.model_response":
                flush_tool_attachments()
                output = event.get("output") or []
                expected_tool_outputs += sum(1 for item in output if item.get("type") == "function_call")
                input_items.extend(copy.deepcopy(output))
            elif event.get("type") == "item.tool_output":
                flush_tool_attachments()
                input_items.append(copy.deepcopy(event["item"]))
                if expected_tool_outputs > 0:
                    expected_tool_outputs -= 1
                if pending_legacy_tool_attachments:
                    pending_tool_attachments.extend(pending_legacy_tool_attachments)
                    pending_legacy_tool_attachments = []
            elif event.get("type") == "item.image_attachment":
                if event.get("source") == "tool":
                    pending_tool_attachments.append(copy.deepcopy(event["attachment"]))
                elif expected_tool_outputs > 0:
                    pending_legacy_tool_attachments.append(copy.deepcopy(event["attachment"]))
                else:
                    flush_tool_attachments()
                    input_items.append(image_message_item(event["attachment"]))
            elif event.get("type") == "turn.interrupted":
                flush_tool_attachments()
                interrupted_items = self._interrupted_tool_completion_items(events, str(event.get("turn_id") or ""))
                for item in interrupted_items:
                    input_items.append(item)
                    if expected_tool_outputs > 0:
                        expected_tool_outputs -= 1
                if interrupted_items:
                    input_items.append(assistant_output_item(INTERRUPTED_TOOL_CONTEXT_BRIDGE))
                elif self._interrupted_partial_stream_needs_bridge(events, str(event.get("turn_id") or "")):
                    input_items.append(assistant_output_item(INTERRUPTED_STREAM_CONTEXT_BRIDGE))
        flush_tool_attachments()
        return input_items

    def _reconstruct_input(
        self,
        thread_id: str,
        *,
        snapshot: ThreadSnapshot | None = None,
    ) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        pending_pre_user: list[dict[str, Any]] = []
        pending_tool_attachments: list[dict[str, Any]] = []
        pending_legacy_tool_attachments: list[dict[str, Any]] = []
        expected_tool_outputs = 0

        def flush_tool_attachments() -> None:
            nonlocal pending_tool_attachments
            if pending_tool_attachments:
                input_items.extend(tool_attachment_context_items(pending_tool_attachments))
                pending_tool_attachments = []

        snap = snapshot or self.thread_store.snapshot(thread_id)
        events = snap.events_after_compaction
        compaction = snap.latest_compaction
        if compaction is not None:
            # A compaction checkpoint starts a new context epoch. Its re-emitted
            # rules/runtime updates are environment context, so place the
            # leading post-compaction pre-user block immediately after system
            # instructions and before the retained history/summary.
            pre_user_events: list[dict[str, Any]] = []
            events_after_context: list[dict[str, Any]] = []
            reached_replayable_history = False
            for event in events:
                pre_user_item = self._pre_user_event_item(event)
                if pre_user_item is not None and not reached_replayable_history:
                    pre_user_events.append(pre_user_item)
                    continue
                if not reached_replayable_history and not self._is_replayable_input_event(event):
                    continue
                reached_replayable_history = True
                events_after_context.append(event)
            input_items.extend(pre_user_events)
            input_items.extend(self._compaction_replacement_items(compaction))
            events = events_after_context
        for event in events:
            pre_user_item = self._pre_user_event_item(event)
            if pre_user_item is not None:
                flush_tool_attachments()
                pending_pre_user.append(pre_user_item)
            elif event.get("type") == "item.user":
                flush_tool_attachments()
                input_items.extend(pending_pre_user)
                pending_pre_user.clear()
                input_items.append(event["item"])
            elif event.get("type") == "item.assistant":
                flush_tool_attachments()
                text = str(event.get("text") or "")
                if text:
                    input_items.append(assistant_output_item(text))
            elif event.get("type") == "item.model_response":
                flush_tool_attachments()
                output = event.get("output") or []
                expected_tool_outputs += sum(1 for item in output if item.get("type") == "function_call")
                input_items.extend(output)
            elif event.get("type") == "item.tool_output":
                flush_tool_attachments()
                input_items.append(event["item"])
                if expected_tool_outputs > 0:
                    expected_tool_outputs -= 1
                if pending_legacy_tool_attachments:
                    pending_tool_attachments.extend(pending_legacy_tool_attachments)
                    pending_legacy_tool_attachments = []
            elif event.get("type") == "item.image_attachment":
                if event.get("source") == "tool":
                    pending_tool_attachments.append(event["attachment"])
                elif expected_tool_outputs > 0:
                    pending_legacy_tool_attachments.append(event["attachment"])
                else:
                    flush_tool_attachments()
                    input_items.append(image_message_item(event["attachment"]))
            elif event.get("type") == "turn.interrupted":
                flush_tool_attachments()
                interrupted_items = self._interrupted_tool_completion_items(events, str(event.get("turn_id") or ""))
                for item in interrupted_items:
                    input_items.append(item)
                    if expected_tool_outputs > 0:
                        expected_tool_outputs -= 1
                if interrupted_items:
                    input_items.append(assistant_output_item(INTERRUPTED_TOOL_CONTEXT_BRIDGE))
                elif self._interrupted_partial_stream_needs_bridge(events, str(event.get("turn_id") or "")):
                    input_items.append(assistant_output_item(INTERRUPTED_STREAM_CONTEXT_BRIDGE))
        flush_tool_attachments()
        return input_items

    def _pre_user_event_item(self, event: dict[str, Any]) -> dict[str, Any] | None:
        event_type = event.get("type")
        if event_type == "item.context_update":
            text = str(event.get("text") or "")
        elif event_type == "item.rules_loaded" and event.get("source") in {
            "project",
            "active_cwd",
        }:
            text = str(event.get("text") or "")
        elif event_type == "item.rule_index":
            text = str(event.get("text") or "")
        elif event_type == "item.cwd_notice":
            text = str(event.get("text") or "")
        else:
            return None
        if not text:
            return None
        return message_item("user", text)

    def _interrupted_tool_completion_items(self, events: list[dict[str, Any]], turn_id: str) -> list[dict[str, Any]]:
        response_output: list[dict[str, Any]] = []
        completed_call_ids: set[str] = set()
        for event in events:
            if event.get("turn_id") != turn_id:
                continue
            if event.get("type") == "item.model_response":
                response_output = list(event.get("output") or [])
            elif event.get("type") == "item.tool_output":
                completed_call_ids.add(str((event.get("item") or {}).get("call_id") or ""))
        items = []
        for call in response_output:
            if call.get("type") != "function_call":
                continue
            call_id = str(call.get("call_id") or "")
            if call_id in completed_call_ids:
                continue
            items.append(
                function_output(
                    call,
                    {
                        "error": (
                            "Tool call did not complete because the user interrupted this turn. "
                            "Do not assume the tool ran successfully."
                        )
                    },
                )
            )
        return items

    def _interrupted_partial_stream_needs_bridge(self, events: list[dict[str, Any]], turn_id: str) -> bool:
        saw_partial = False
        saw_model_response = False
        for event in events:
            if event.get("turn_id") != turn_id:
                continue
            if event.get("type") == "turn.interrupted" and event.get("partial_stream"):
                return True
            if event.get("type") in {"item.assistant_partial", "item.reasoning_partial"}:
                saw_partial = True
            elif event.get("type") == "item.model_response":
                saw_model_response = True
        return saw_partial and not saw_model_response

    def _runtime_context_items(self, thread_id: str | None = None) -> list[dict[str, Any]]:
        update = self._turn_context_update(thread_id)
        if update is None:
            return []
        if thread_id:
            self.thread_store.append(
                thread_id,
                "item.context_update",
                context_fingerprint=update["fingerprint"],
                context_state=update["state"],
                context_kind="runtime",
                removed=update["removed"],
                text=update["text"],
            )
        return [message_item("user", update["text"])]

    def _pre_user_context_items(self, thread_id: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for text in self._rule_context_texts(thread_id):
            items.append(message_item("user", text))
        items.extend(self._runtime_context_items(thread_id))
        return items

    def _rule_context_texts(self, thread_id: str) -> list[str]:
        state = self._rule_state(thread_id)
        texts: list[str] = []
        if not state.index_emitted:
            project_rules = self._project_rule_reload_text(thread_id, state)
            if project_rules:
                texts.append(project_rules)
            index_root = self.project_root
            index = discover_workspace_rule_index(index_root)
            rendered = index.render(label="project directory")
            if rendered:
                texts.append(rendered)
            state.index_emitted = True
            self.thread_store.append(
                thread_id,
                "item.rule_index",
                text=rendered,
                root=str(index_root.resolve()),
                max_depth=index.max_depth,
                max_entries=index.max_entries,
                truncated=index.truncated_entries or index.depth_limited,
                paths=[str(path) for path in index.paths],
            )
            if state.active_cwd.resolve() != self.project_root.resolve():
                texts.extend(
                    message["text"]
                    for message in self._load_unseen_rules_for_dir(
                        thread_id,
                        state.active_cwd,
                        source="active_cwd",
                    )
                    if message.get("text")
                )
        cwd_notice = self._active_cwd_notice(thread_id)
        if cwd_notice:
            texts.append(cwd_notice)
        return texts

    def _project_rule_reload_text(self, thread_id: str, state: RuleRuntimeState) -> str:
        context = load_project_rules(self.project_root)
        new_rules = [rule for rule in context.rules if rule.path not in state.loaded_rule_paths]
        if not new_rules:
            return ""
        for rule in new_rules:
            state.loaded_rule_paths.add(rule.path)
        filtered = ProjectRuleContext(
            rules=new_rules,
            truncated=context.truncated,
            omitted_files=context.omitted_files,
        )
        text = filtered.render(root=self.project_root, context_path=".")
        # Persist root-level rules just like enter_dir-loaded rules so a resumed
        # engine can rebuild the epoch state instead of re-reading changed files.
        self.thread_store.append(
            thread_id,
            "item.rules_loaded",
            cwd=str(self.project_root.resolve()),
            paths=[str(rule.path) for rule in new_rules],
            text=text,
            source="project",
        )
        return text

    def _active_cwd_notice(self, thread_id: str) -> str:
        state = self._rule_state(thread_id)
        active_cwd = state.active_cwd.resolve()
        initial_cwd = self.project_root.resolve()
        if active_cwd == initial_cwd:
            state.cwd_notice_cwd = None
            return ""
        if state.cwd_notice_cwd == active_cwd:
            return ""
        text = (
            "<active_cwd_notice>\n"
            f"The active working directory for run_python is now {xml_text(self._relative_to_project(active_cwd))}. "
            f"The thread opened at {xml_text(self._relative_to_project(initial_cwd))}. "
            "Relative paths and automatic directory rules follow the active working directory.\n"
            "</active_cwd_notice>"
        )
        state.cwd_notice_cwd = active_cwd
        self.thread_store.append(
            thread_id,
            "item.cwd_notice",
            cwd=str(active_cwd),
            initial_cwd=str(initial_cwd),
            text=text,
        )
        return text

    def _load_unseen_rules_for_dir(
        self,
        thread_id: str,
        directory: Path,
        *,
        source: str,
    ) -> list[dict[str, Any]]:
        state = self._rule_state(thread_id)
        context = load_directory_rules(directory, root=self.project_root)
        new_rules = [rule for rule in context.rules if rule.path not in state.loaded_rule_paths]
        if not new_rules:
            return []
        for rule in new_rules:
            state.loaded_rule_paths.add(rule.path)
        filtered = ProjectRuleContext(
            rules=new_rules,
            truncated=context.truncated,
            omitted_files=context.omitted_files,
        )
        text = filtered.render(
            root=self.project_root,
            base_path=directory,
            context_path=self._relative_to_project(directory),
        )
        event = {
            "kind": "rules_loaded",
            "cwd": self._relative_to_project(directory),
            "paths": [self._relative_to_project(rule.path) for rule in new_rules],
            "text": text,
        }
        self.thread_store.append(
            thread_id,
            "item.rules_loaded",
            cwd=str(directory.resolve()),
            paths=[str(rule.path) for rule in new_rules],
            text=text,
            source=source,
        )
        return [event]

    def _rule_state(self, thread_id: str) -> RuleRuntimeState:
        state = self._rule_states.get(thread_id)
        if state is not None:
            return state
        active_cwd = self._latest_active_cwd(thread_id)
        state = RuleRuntimeState(
            active_cwd=active_cwd,
            loaded_rule_paths=self._loaded_rule_paths_in_epoch(thread_id),
            index_emitted=self._rule_index_already_emitted(thread_id, active_cwd),
            cwd_notice_cwd=self._latest_cwd_notice_in_epoch(thread_id),
        )
        self._rule_states[thread_id] = state
        return state

    def _latest_active_cwd(self, thread_id: str) -> Path:
        raw_cwd = self.thread_store.snapshot(thread_id).metadata.get("latest_cwd")
        if isinstance(raw_cwd, str) and raw_cwd:
            try:
                cwd = Path(raw_cwd).resolve()
            except OSError:
                return self.project_root.resolve()
            if self._is_within_project(cwd):
                return cwd
        return self.project_root.resolve()

    def _active_cwd(self, thread_id: str) -> Path:
        return self._rule_state(thread_id).active_cwd

    def _reset_rule_epoch(self, thread_id: str) -> None:
        state = self._rule_state(thread_id)
        state.loaded_rule_paths.clear()
        state.index_emitted = False
        state.cwd_notice_cwd = None

    def _loaded_rule_paths_in_epoch(self, thread_id: str) -> set[Path]:
        events = self.thread_store.snapshot(thread_id).events_after_compaction
        loaded: set[Path] = set()
        for event in events:
            if event.get("type") != "item.rules_loaded":
                continue
            paths = event.get("paths")
            if not isinstance(paths, list):
                continue
            for raw_path in paths:
                if not isinstance(raw_path, str) or not raw_path:
                    continue
                try:
                    loaded.add(Path(raw_path).resolve())
                except OSError:
                    continue
        return loaded

    def _rule_index_already_emitted(self, thread_id: str, active_cwd: Path) -> bool:
        del active_cwd
        events = self.thread_store.snapshot(thread_id).events_after_compaction
        for event in events:
            if event.get("type") == "item.rule_index":
                return True
        return False

    def _latest_cwd_notice_in_epoch(self, thread_id: str) -> Path | None:
        events = self.thread_store.snapshot(thread_id).events_after_compaction
        for event in reversed(events):
            if event.get("type") != "item.cwd_notice":
                continue
            raw_cwd = event.get("cwd")
            if not isinstance(raw_cwd, str) or not raw_cwd:
                continue
            try:
                cwd = Path(raw_cwd).resolve()
            except OSError:
                continue
            if self._is_within_project(cwd):
                return cwd
        return None

    def _has_compaction(self, thread_id: str, *, snapshot: ThreadSnapshot | None = None) -> bool:
        return (snapshot or self.thread_store.snapshot(thread_id)).latest_compaction is not None

    def _is_within_project(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.project_root.resolve())
        except ValueError:
            return False
        return True

    def _relative_to_project(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self.project_root.resolve())
        except ValueError:
            return str(resolved)
        return "." if not relative.parts else relative.as_posix()

    def _turn_context_update(self, thread_id: str | None) -> dict[str, Any] | None:
        previous = self._latest_context_state(thread_id) if thread_id else None
        previous_parts = _context_state_parts(previous)
        parts = self._turn_context_parts(previous_parts=previous_parts)
        full_rendered = "\n\n".join(part.text for part in parts)
        fingerprint = context_fingerprint(full_rendered)
        state_parts = {
            part.id: {
                "fingerprint": context_fingerprint(part.text),
                "kind": part.kind,
                "dynamic": part.dynamic,
                "metadata": part.metadata,
            }
            for part in parts
        }
        previous_fingerprint = previous.get("fingerprint") if previous else None
        initial = previous_fingerprint is None
        if initial:
            removed = [key for key in previous_parts if key not in state_parts]
            rendered_parts = parts
        else:
            current_kinds = {part.kind for part in parts}
            previous_dynamic = {
                key
                for key, value in previous_parts.items()
                if value.get("dynamic")
                or key in {"skills", "mcp"}
                or (value.get("kind") in {"skills", "mcp"} and value.get("kind") not in current_kinds)
            }
            current_dynamic = {part.id for part in parts if part.dynamic}
            changed = [
                part.id
                for part in parts
                if part.dynamic
                and previous_parts.get(part.id, {}).get("fingerprint")
                != state_parts[part.id]["fingerprint"]
            ]
            removed = [
                key
                for key in previous_dynamic
                if key not in current_dynamic
            ]
            if not changed and not removed:
                return None
            changed_set = set(changed)
            rendered_parts = [part for part in parts if part.id in changed_set]
        rendered = "\n\n".join(part.text for part in rendered_parts)
        if not full_rendered:
            if previous_fingerprint is None:
                return None
            return {
                "fingerprint": fingerprint,
                "state": {"fingerprint": fingerprint, "parts": state_parts},
                "removed": removed or sorted(previous_parts),
                "text": (
                    "<context_update id=\"runtime_context\" status=\"removed\">\n"
                    "Previously available runtime context is no longer present. "
                    "Do not rely on older runtime context unless it appears again.\n"
                    "</context_update>"
                ),
            }
        if removed:
            removed_text = (
                "\n\n<context_update_removed id=\"runtime_context\">\n"
                "Some previously available runtime context is no longer present. "
                "Do not rely on older appended content for removed skills or MCP servers unless they appear again.\n"
                + _removed_context_text(removed, previous_parts)
                + "\n"
                "</context_update_removed>"
            )
        else:
            removed_text = ""
        prefix = (
            "<context_update id=\"runtime_context\" status=\"current\">\n"
            "The following runtime context is current. It updates only the listed content; prior runtime context remains current within this epoch unless explicitly removed.\n"
            + "</context_update>"
        )
        text = prefix + removed_text + ("\n\n" + rendered if rendered else "")
        return {
            "fingerprint": fingerprint,
            "state": {"fingerprint": fingerprint, "parts": state_parts},
            "removed": removed,
            "text": text,
        }

    def _latest_context_state(self, thread_id: str | None) -> dict[str, Any] | None:
        if not thread_id:
            return None
        snap = self.thread_store.snapshot(thread_id)
        events = snap.events_after_compaction
        for event in reversed(events):
            if event.get("type") == "item.context_update":
                state = event.get("context_state")
                if isinstance(state, dict):
                    return state
                return {"fingerprint": str(event.get("context_fingerprint") or ""), "parts": {}}
        return None

    def _turn_context_text(self) -> str:
        return "\n\n".join(part.text for part in self._turn_context_parts())

    def _turn_context_parts(
        self,
        *,
        previous_parts: dict[str, dict[str, Any]] | None = None,
    ) -> list[ContextPart]:
        previous_parts = previous_parts or {}
        parts: list[ContextPart] = [
            ContextPart("runtime_environment", "runtime_environment", self._runtime_environment_context()),
            ContextPart("model_levels", "model_levels", self._model_levels_context()),
            ContextPart("runtime_helpers", "runtime_helpers", self._runtime_helpers_context()),
        ]
        skills = discover_skills(self.project_root)
        if skills:
            parts.extend(self._skill_context_parts(skills))

        mcp_servers = discover_mcp_servers(self.project_root)
        if mcp_servers:
            parts.extend(self._mcp_context_parts(mcp_servers, previous_parts=previous_parts))
        return parts

    def _skill_context_parts(self, skills: list[SkillSummary]) -> list[ContextPart]:
        parts = [
            ContextPart(
                "skills/header",
                "skills",
                (
                    "<available_skills>\n"
                    "Use these skills when one matches the task; read the listed SKILL.md with Python before applying it."
                ),
            )
        ]
        for skill in skills[:10]:
            parts.append(
                ContextPart(
                    f"skills/{_context_item_id(skill.key)}",
                    "skills",
                    render_skill_entry(skill),
                    dynamic=True,
                    metadata={
                        "kind": "skill",
                        "name": skill.name,
                        "scope": skill.scope,
                        "path": str(skill.path),
                    },
                )
            )
        if len(skills) > 10:
            parts.append(
                ContextPart(
                    "skills/omitted",
                    "skills",
                    f'<omitted_skills count="{len(skills) - 10}" />',
                    dynamic=True,
                )
            )
        parts.append(ContextPart("skills/footer", "skills", "</available_skills>"))
        return parts

    def _mcp_context_parts(
        self,
        servers: list[McpServerSummary],
        *,
        previous_parts: dict[str, dict[str, Any]],
    ) -> list[ContextPart]:
        instructions = self._mcp_instructions_probe.snapshot()
        parts = [
            ContextPart(
                "mcp/header",
                "mcp",
                (
                    "<available_mcp_servers>\n"
                    "Use these MCP servers when they fit the task; inspect and call them through uv_agent_runtime MCP helpers from Python."
                ),
            )
        ]
        for server in servers[:10]:
            part_id = f"mcp/{_context_item_id(server.key)}"
            previous_metadata = previous_parts.get(part_id, {}).get("metadata")
            previous_instructions = _mcp_preview_from_metadata(
                previous_metadata.get("instructions")
                if isinstance(previous_metadata, dict)
                else None
            )
            preview = instructions.get(server.key) or (
                previous_instructions
                if previous_instructions is not None
                else None
            )
            parts.append(
                ContextPart(
                    part_id,
                    "mcp",
                    render_mcp_entry(server, preview),
                    dynamic=True,
                    metadata={
                        "kind": "mcp",
                        "name": server.name,
                        "scope": server.scope,
                        "config": str(server.path),
                        "instructions": _mcp_preview_metadata(preview),
                    },
                )
            )
        if len(servers) > 10:
            parts.append(
                ContextPart(
                    "mcp/omitted",
                    "mcp",
                    f'<omitted_mcp_servers count="{len(servers) - 10}" />',
                    dynamic=True,
                )
            )
        parts.append(ContextPart("mcp/footer", "mcp", "</available_mcp_servers>"))
        return parts

    def project_rule_context(self) -> ProjectRuleContext:
        """Load AGENTS.md context for status/debug display."""
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
        if not thread_id:
            return ContextStats(
                used_tokens=0,
                context_window_tokens=model.context_window_tokens,
                percent=0,
                threshold_tokens=trigger_tokens,
                headroom_tokens=model.context_window_tokens,
                source="empty",
            )
        try:
            snapshot = self.thread_store.snapshot(thread_id)
        except FileNotFoundError:
            return ContextStats(
                used_tokens=0,
                context_window_tokens=model.context_window_tokens,
                percent=0,
                threshold_tokens=trigger_tokens,
                headroom_tokens=model.context_window_tokens,
                source="empty",
            )
        used = self._latest_usage_tokens(thread_id, snapshot=snapshot)
        source = "provider"
        if used is None:
            update = self._turn_context_update(thread_id)
            context_items = [message_item("user", update["text"])] if update else []
            used = estimate_tokens(self._reconstruct_input(thread_id, snapshot=snapshot) + context_items)
            source = "estimate"
        percent = min(100, max(0, round(used * 100 / model.context_window_tokens)))
        return ContextStats(
            used_tokens=used,
            context_window_tokens=model.context_window_tokens,
            percent=percent,
            threshold_tokens=trigger_tokens,
            headroom_tokens=max(0, model.context_window_tokens - used),
            source=source,
        )

    def _latest_usage_tokens(self, thread_id: str, *, snapshot: ThreadSnapshot | None = None) -> int | None:
        """Return the latest provider-reported token usage when available."""
        snap = snapshot or self.thread_store.snapshot(thread_id)
        metadata_usage = snap.metadata.get("latest_usage_tokens")
        if isinstance(metadata_usage, int):
            return metadata_usage
        events = snap.events_after_compaction
        compaction = snap.latest_compaction
        for event in reversed(events):
            if event.get("type") != "item.model_response":
                continue
            used = usage_token_count(event.get("usage") or {})
            if used is not None:
                return used
        if compaction is not None:
            used = usage_token_count(compaction.get("usage") or {})
            if used is not None:
                return used
        return None

    def system_instructions(self) -> str:
        """Build concise environment-aware system instructions."""
        return SYSTEM_INSTRUCTIONS_TEMPLATE

    def _runtime_environment_context(self) -> str:
        scriptenv_dir = getattr(self.runner, "scriptenv_dir", self.thread_store.data_dir / "runner" / "scriptenv")
        return runtime_environment_context(
            project_root=self.project_root,
            user_state=uv_agent_home(),
            project_state=self.thread_store.data_dir,
            scriptenv_dir=scriptenv_dir,
            scriptenv_dependencies=direct_dependencies(scriptenv_dir),
            host_environment=self._host_environment,
            user_language=detect_user_language(self.config.ui.language),
        )

    def _model_levels_context(self) -> str:
        return model_levels_context(self.config)

    def _runtime_helpers_context(self) -> str:
        return runtime_helpers_context()

def tool_attachment_context_items(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a neutral assistant bridge followed by tool-produced image context."""
    if not attachments:
        return []
    items = [assistant_output_item(TOOL_ATTACHMENT_CONTEXT_BRIDGE)]
    items.extend(image_message_item(attachment) for attachment in attachments)
    return items


def truncate_tool_output_for_compaction(raw_output: str, max_chars: int) -> str:
    """Return a context-sized tool output while preserving JSON when possible.

    `function_call_output.output` is usually a JSON string produced by
    `function_output()`. Keeping that string parseable gives the compaction
    model useful metadata (return code, run id, truncation flag) instead of a
    broken fragment. Non-JSON outputs are rare but still get a head/tail clip.
    """

    max_chars = max(0, max_chars)
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        return _head_tail_truncate_text(raw_output, max_chars, marker="[tool output truncated for context compaction]")
    if not isinstance(payload, dict):
        return _head_tail_truncate_text(raw_output, max_chars, marker="[tool output truncated for context compaction]")
    if max_chars <= 0:
        return json.dumps(
            {
                "truncated_for_context_compaction": True,
                "truncation_note": (
                    "Tool output was omitted to fit the context compaction request."
                ),
                "original_json_length": len(raw_output),
            },
            ensure_ascii=False,
        )

    truncated = copy.deepcopy(payload)
    truncated["truncated_for_context_compaction"] = True
    truncated["truncation_note"] = (
        "Tool output was shortened to fit the context compaction request. "
        "Only a head/tail excerpt of large text fields may be present."
    )
    original_lengths = {
        key: len(value)
        for key, value in payload.items()
        if key in {"stdout", "stderr", "output"} and isinstance(value, str)
    }
    if original_lengths:
        truncated["original_text_lengths"] = original_lengths

    large_keys = [key for key in ("stdout", "stderr", "output") if isinstance(truncated.get(key), str)]
    if not large_keys:
        return _head_tail_truncate_text(raw_output, max_chars, marker="[tool output truncated for context compaction]")

    for key in large_keys:
        truncated[key] = str(truncated.get(key) or "")

    for _ in range(8):
        candidate = json.dumps(truncated, ensure_ascii=False)
        if len(candidate) <= max_chars:
            return candidate
        oversized = max(1, len(candidate) - max_chars)
        current_lengths = {key: len(str(truncated.get(key) or "")) for key in large_keys}
        total_text = sum(current_lengths.values())
        if total_text <= 0:
            break
        for key in large_keys:
            current = str(truncated.get(key) or "")
            if not current:
                continue
            reduction = max(1, int(oversized * (len(current) / total_text)) + 256)
            target = max(0, len(current) - reduction)
            truncated[key] = _head_tail_truncate_text(
                current,
                target,
                marker="[truncated for context compaction]",
            )
    return json.dumps(truncated, ensure_ascii=False)


def _head_tail_truncate_text(text: str, max_chars: int, *, marker: str) -> str:
    """Keep both ends of text because diagnostics often finish with the error."""

    max_chars = max(0, max_chars)
    if len(text) <= max_chars:
        return text
    if max_chars == 0:
        return ""
    marker_text = f"\n...{marker}...\n"
    if max_chars <= len(marker_text):
        return marker_text[:max_chars]
    keep = max_chars - len(marker_text)
    head = keep // 2
    tail = keep - head
    return text[:head].rstrip() + marker_text + text[len(text) - tail :].lstrip()


def is_default_thread_title(title: str) -> bool:
    return title.strip() in DEFAULT_THREAD_TITLES


def clean_thread_title(text: str) -> str | None:
    title = text.strip().splitlines()[0].strip()
    title = title.strip(" \t\r\n\"'`“”‘’")
    title = re.sub(r"^[#*\-\d\.\)\s]+", "", title).strip()
    title = re.sub(r"\s+", " ", title)
    if not title:
        return None
    if len(title) > 80:
        title = title[:77].rstrip() + "..."
    return title


def completion_text_delta(output_text: str, emitted_text: str) -> str:
    if not output_text:
        return ""
    if not emitted_text:
        return output_text
    if output_text.startswith(emitted_text):
        return output_text[len(emitted_text) :]
    return ""
