from __future__ import annotations

import asyncio
import copy
import inspect
import importlib
import json
import random
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, field
from html import escape as xml_escape
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from uv_agent.attachments import AttachmentStore, image_message_item
from uv_agent.agent.compaction import (
    DEPENDENCY_PARAMS,
    K_CANDIDATE_PCTS,
    N_BUCKET_MAP,
    S_MAX,
    S_MIN,
    compaction_judge_request_item,
    compaction_response_summary_text,
    compaction_summary_item,
    compaction_replacement_input,
    compaction_trigger_item,
    compute_net_gain,
    estimate_compact_cost,
    normalize_compaction_replacement_input,
    parse_judge_response,
    retain_item_after_compaction,
    retained_user_messages_after_compaction,
    strip_compaction_judge_history,
)
from uv_agent.billing import (
    billing_charge_for_usage,
    billing_total_from_metadata,
    decimal_to_string,
    pricing_for_model,
    unit_divisor,
)
from uv_agent.config import AppConfig
from uv_agent.agent.context_builder import (
    context_fingerprint,
    model_levels_context,
    runtime_environment_context,
    runtime_helpers_context,
    xml_text,
)
from uv_agent.context import ContextStats, estimate_tokens, usage_token_count
from uv_agent.state_db import checkpoint_state_db
from uv_agent.environment import detect_user_language, host_environment
from uv_agent.host_events import HostEventBus
from uv_agent.errors import EmptyModelStreamError, is_retryable_provider_error
from uv_agent.goal_mode import (
    GoalState,
    ensure_goal_files,
    read_goal_state,
    render_goal_mode_notice,
)
from uv_agent.ids import new_id
from uv_agent.agent.messages import assistant_output_item, message_item, message_item_text
from uv_agent.mcp_config import McpInstructionsPreview, McpServerSummary, discover_mcp_servers, render_mcp_entry
from uv_agent.mcp_probe import McpInstructionsProbe
from uv_agent.model.types import ModelClient, ModelResponse
from uv_agent.paths import uv_agent_home
from uv_agent.plugins import EventBus, PluginManager, SubmittedTurn, TurnContextBlock, TurnPrepareRequest
from uv_agent.plugins.helpers import RuntimeHelperRegistry
from uv_agent.turn_manager import TurnManager
from uv_agent.scheduler import SchedulerService
from uv_agent.prompts import (
    BRANCH_NAME_GENERATION_PROMPT,
    INTERRUPTED_STREAM_CONTEXT_BRIDGE,
    INTERRUPTED_TOOL_CONTEXT_BRIDGE,
    POST_TOOL_COMPACTION_BRIDGE,
    PYTHON_TOOL,
    SYSTEM_INSTRUCTIONS_TEMPLATE,
    TITLE_GENERATION_PROMPT,
    TOOL_ATTACHMENT_CONTEXT_BRIDGE,
    BRANCH_SLUG_INSTRUCTION,
    THREAD_TITLE_INSTRUCTION,
    PRE_TURN_JUDGE_ERROR_STDERR,
    TOKEN_ESTIMATION_WARNING,
    COMPACTION_TOOL_ERROR_STDERR,
    COMPACTION_CONTINUE_WITHOUT_CURRENT_USER,
    INTERRUPTED_TOOL_ERROR,
    GUIDED_INPUT_CONTEXT_BRIDGE,
    ACTIVE_CWD_NOTICE_TEMPLATE,
    CONTEXT_REMOVED_ALL,
    CONTEXT_REMOVED_SOME_PREFIX,
    CONTEXT_REMOVED_SOME_SUFFIX,
    CONTEXT_UPDATE_CURRENT_PREFIX,
    CONTEXT_UPDATE_CURRENT_SUFFIX,
    SKILLS_HEADER,
    MCP_SERVERS_HEADER,
    PLUGIN_HELPERS_HEADER,
    TOOL_OUTPUT_TRUNCATED_MARKER,
    TOOL_OUTPUT_OMITTED_NOTE,
    AVAILABLE_MCP_SERVERS_FOOTER,
    AVAILABLE_SKILLS_FOOTER,
    GOAL_MODE_ENABLED_STATUS_FRAGMENT,
    MCP_OMITTED_TEMPLATE,
    PLUGIN_HELPER_ENTRY_TEMPLATE,
    PLUGIN_CONTEXT_TAG,
    PLUGIN_HELPERS_FOOTER,
    PRE_USER_CONTEXT_MARKERS,
    REMOVED_MCP_SERVER_TEMPLATE,
    REMOVED_SKILL_TEMPLATE,
    SKILLS_OMITTED_TEMPLATE,
    TOOL_OUTPUT_SHORTENED_NOTE,
    WORKTREE_ACTIVE_STATUS_FRAGMENT,
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
from uv_agent.helper_calls import extract_runtime_helper_calls, runtime_corrected_helper_calls
from uv_agent.worktree import render_worktree_notice
from uv_agent.workflow_context import active_workflows_compaction_section, render_workflow_context


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
                REMOVED_SKILL_TEMPLATE.format(
                    name=_xml_attr(name),
                    scope=_xml_attr(scope),
                    path=_xml_attr(path or ""),
                )
            )
        elif kind == "mcp" and name and scope:
            lines.append(
                REMOVED_MCP_SERVER_TEMPLATE.format(
                    name=_xml_attr(name),
                    scope=_xml_attr(scope),
                    config=_xml_attr(path or ""),
                )
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
        # Shallow copy is sufficient: the engine treats persisted items as
        # immutable, and the streaming path makes its own deep copy before
        # handing the list to provider SDKs for retries.
        if self.use_previous_response_id and self.previous_response_id:
            return list(self.pending_items)
        return list(self.input_items)

    def request_previous_response_id(self) -> str | None:
        if self.use_previous_response_id and self.previous_response_id:
            return self.previous_response_id
        return None

    def note_tool_output(self, item: dict[str, Any]) -> None:
        """Add a tool result to both full history and incremental request input."""

        self.input_items.append(item)
        self.pending_items.append(item)

    def note_tool_attachments(self, attachments: list[dict[str, Any]]) -> None:
        if not attachments:
            return
        attachment_items = tool_attachment_context_items(attachments)
        self.input_items.extend(attachment_items)
        self.pending_items.extend(copy.deepcopy(attachment_items))
        self.use_previous_response_id = False


@dataclass
class RetryState:
    input_items: list[dict[str, Any]]
    previous_response_id: str | None = None
    use_previous_response_id: bool = False
    pending_items: list[dict[str, Any]] = field(default_factory=list)
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def request_input_items(self) -> list[dict[str, Any]]:
        # Shallow copy is sufficient: the engine treats persisted items as
        # immutable, and the streaming path makes its own deep copy before
        # handing the list to provider SDKs for retries.
        if self.use_previous_response_id and self.previous_response_id:
            return list(self.pending_items)
        return list(self.input_items)

    def request_previous_response_id(self) -> str | None:
        if self.use_previous_response_id and self.previous_response_id:
            return self.previous_response_id
        return None

    def note_tool_output(self, item: dict[str, Any]) -> None:
        """Add a tool result to both full history and incremental request input."""

        self.input_items.append(item)
        self.pending_items.append(item)

    def note_tool_attachments(self, attachments: list[dict[str, Any]]) -> None:
        if not attachments:
            return
        attachment_items = tool_attachment_context_items(attachments)
        self.input_items.extend(attachment_items)
        self.pending_items.extend(copy.deepcopy(attachment_items))
        self.use_previous_response_id = False


@dataclass
class RuleRuntimeState:
    active_cwd: Path
    loaded_rule_paths: set[Path] = field(default_factory=set)
    index_emitted: bool = False
    cwd_notice_cwd: Path | None = None


@dataclass
class StreamResponseState:
    # Single-string buffers avoid holding the streaming answer twice: once in
    # the provider layer and once as a list of deltas in the engine.
    assistant_text: str = ""
    reasoning_text: str = ""
    saw_stream_output: bool = False
    response: ModelResponse | None = None

    @property
    def partial_text(self) -> str:
        return self.assistant_text.strip()

    @property
    def partial_reasoning_text(self) -> str:
        return self.reasoning_text.strip()

    def reset(self) -> None:
        self.assistant_text = ""
        self.reasoning_text = ""
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


@dataclass
class JudgeRunState:
    """Mutable result side channel for a streamed pre-turn judge.

    Async generators cannot return a final value, but ``run_turn`` needs to know
    whether cache-aware pre-turn compaction already ran so it can skip the
    regular post-turn compaction pass.
    """

    compacted: bool = False


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
    prepare_request: TurnPrepareRequest
    user_item: dict[str, Any]
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
        host_events: HostEventBus | None = None,
    ) -> None:
        self.config = config
        self.model_client = model_client
        self.runner = runner
        self.thread_store = thread_store
        self.host_events = host_events or HostEventBus()
        self.project_root = project_root
        self.attachments = AttachmentStore(attachments_dir or thread_store.data_dir / "attachments")
        self._last_config_refresh_at = 0.0
        self._config_loader = config_loader
        self._host_environment = host_environment()
        self._rule_states: OrderedDict[str, RuleRuntimeState] = OrderedDict()
        self._rule_states_max_size: int = 64
        self._mcp_instructions_probe = mcp_instructions_probe or McpInstructionsProbe(self.project_root)
        self._mcp_instructions_probe.start()
        self.events = EventBus()
        self.host_events.register_plugin_bus(self.events)
        self.runtime_helpers = RuntimeHelperRegistry()
        self.plugins = PluginManager(
            config=self.config.plugins,
            project_root=self.project_root,
            events=self.events,
            helper_registry=self.runtime_helpers,
            submitter=self._plugin_submit_turn,
            thread_store=self.thread_store,
        )
        rpc_server = getattr(self.runner, "rpc_server", None)
        if rpc_server is not None:
            rpc_server.register_method("helper.resolve", self.plugins.resolve_helper)
            rpc_server.register_method("helper.call", self.plugins.call_helper)
            rpc_server.register_method("scheduler.create", self.scheduler.create)
            rpc_server.register_method("scheduler.update", self.scheduler.update)
            rpc_server.register_method("scheduler.list", self.scheduler.list)
            rpc_server.register_method("scheduler.delete", self.scheduler.delete)
            rpc_server.register_method("scheduler.run_now", self.scheduler.run_now)
        self._plugins_started = False
        self._plugins_start_task: asyncio.Task[None] | None = None
        self._last_judge: dict[str, Any] | None = None
        self._context_stats_ttl_s: float = 1.0
        self._context_stats_cache: dict[tuple[str | None, str | None], tuple[float, ContextStats]] = {}
        self._turns_since_db_checkpoint: int = 0
        self._db_checkpoint_interval: int = 50
        self.turn_manager = TurnManager(
            self,
            max_concurrent_turns=getattr(self.config.runtime, "max_concurrent_turns", 4),
        )
        self.scheduler = SchedulerService(
            self.thread_store.data_dir,
            self.config.scheduler,
            helper_resolver=self.plugins.resolve_helper,
            helper_caller=self.plugins.call_helper,
        )

    def _publish_host_event(self, event: dict[str, Any]) -> None:
        """Best-effort publish a host event; never raise."""

        try:
            self.host_events.publish(event)
        except Exception:
            return

    def close(self) -> None:
        """Release long-lived host resources owned by the engine."""

        close = getattr(self.runner, "close", None)
        if callable(close):
            close()

    async def aclose(self) -> None:
        await self.turn_manager.aclose()
        await self.plugins.stop()
        model_close = getattr(self.model_client, "aclose", None)
        if callable(model_close):
            await model_close()
        try:
            await asyncio.to_thread(checkpoint_state_db, self.thread_store.data_dir, mode="PASSIVE")
        except Exception:
            # Checkpointing is best-effort; do not let cleanup failures mask
            # the real shutdown path.
            pass
        close = getattr(self.runner, "aclose", None)
        if callable(close):
            await close()
        else:
            await asyncio.to_thread(self.close)
        self.host_events.close()

    def start_plugins_background(self) -> asyncio.Task[None]:
        self._plugins_started = True
        self._plugins_start_task = self.plugins.start_background()
        return self._plugins_start_task

    async def _ensure_plugins_started(self) -> None:
        if not self._plugins_started or self._plugins_start_task is None:
            self.start_plugins_background()
        task = self._plugins_start_task
        if task is None or task is asyncio.current_task():
            return
        try:
            await task
        except Exception:
            # Individual plugin startup errors are recorded by PluginManager;
            # unexpected manager-level failures should not block an agent turn.
            return

    async def submit_turn(
        self,
        *,
        user_text: str,
        thread_id: str | None = None,
        level: str | None = None,
        image_paths: list[str | Path] | None = None,
        conflict: str = "queue",
    ):
        return await self.turn_manager.submit_turn(
            user_text=user_text,
            thread_id=thread_id,
            level=level,
            image_paths=image_paths,
            conflict=conflict,  # type: ignore[arg-type]
        )

    async def _plugin_submit_turn(
        self,
        *,
        text: str,
        thread_id: str | None = None,
        level: str | None = None,
        image_paths: list[Path] | None = None,
    ) -> SubmittedTurn:
        handle = await self.turn_manager.submit_turn(
            user_text=text,
            thread_id=thread_id,
            level=level,
            image_paths=image_paths,
            conflict="queue",
        )
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        started: asyncio.Future[tuple[str, str]] = asyncio.get_running_loop().create_future()

        async def forward_events() -> None:
            try:
                async for event in handle.events():
                    if event.get("type") == "turn.started" and not started.done():
                        started.set_result((str(event.get("thread_id") or ""), str(event.get("turn_id") or "")))
                    await queue.put(event)
                if not started.done():
                    started.set_result((handle.thread_id or thread_id or "", handle.turn_id or ""))
            except Exception as exc:
                if not started.done():
                    started.set_exception(exc)
                await queue.put({"type": "turn.error", "message": str(exc) or repr(exc), "error_type": exc.__class__.__name__})
            finally:
                await queue.put(None)

        asyncio.create_task(forward_events(), name="uv-agent-plugin-submit-turn-forward")
        submitted_thread_id, submitted_turn_id = await started
        return SubmittedTurn(thread_id=submitted_thread_id, turn_id=submitted_turn_id, _queue=queue)

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

    def _record_judge(self, data: dict[str, Any]) -> None:
        """Store the most recent judge calculation for /status display."""
        self._last_judge = data

    def last_judge_summary(self) -> dict[str, Any] | None:
        """Return the most recent cache-aware judge calculation, if any."""
        return self._last_judge

    async def run_turn(
        self,
        *,
        user_text: str,
        thread_id: str | None = None,
        level: str | None = None,
        image_paths: list[str | Path] | None = None,
        cancel_event: asyncio.Event | None = None,
        guide_event: asyncio.Event | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        await self._ensure_plugins_started()
        is_new_thread = thread_id is None
        thread_id = thread_id or await asyncio.to_thread(self.thread_store.create_thread, "New thread")
        with self.thread_store.lock_thread(thread_id):
            prelude = await asyncio.to_thread(
                self._prepare_run_turn_prelude,
                user_text=user_text,
                thread_id=thread_id,
                level=level,
                image_paths=image_paths,
                cancel_event=cancel_event,
                is_new_thread=is_new_thread,
            )
            turn_id = prelude.turn_id
            system_instructions = prelude.system_instructions
            turn_input = prelude.turn_input
            input_items = prelude.input_items
            request_input_items = prelude.request_input_items
            turn_started_event = prelude.turn_started_event
            title_task: asyncio.Task[str | None] | None = None
            yield self._publish_event({
                "type": "turn.started",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "turn_started_at": turn_started_event.get("created_at"),
            })

            # ---- cache-aware compaction judge ----
            # The prelude intentionally leaves the current user message out of
            # the replay input.  The optional judge/compaction operates on prior
            # history only; the real user task is appended afterwards so it stays
            # a fresh instruction rather than retained-history text.
            compacted_this_turn = False
            if (
                self.config.runtime.compression.cache_aware
                and self.config.runtime.compression.enabled
            ):
                judge_state = JudgeRunState()
                async for event in self._stream_judge_and_compact(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    input_items=input_items,
                    turn_input=turn_input,
                    user_item=prelude.user_item,
                    image_events=prelude.image_events,
                    system_instructions=system_instructions,
                    level=level,
                    cancel_event=cancel_event,
                    judge_state=judge_state,
                ):
                    yield event
                compacted_this_turn = judge_state.compacted

            plugin_context_blocks = await self.plugins.prepare_turn(prelude.prepare_request)
            self._append_plugin_context_items(
                thread_id=thread_id,
                turn_id=turn_id,
                turn_input=turn_input,
                blocks=plugin_context_blocks,
            )
            self._append_current_user_items(
                thread_id=thread_id,
                turn_id=turn_id,
                turn_input=turn_input,
                user_item=prelude.user_item,
                image_events=prelude.image_events,
            )
            request_input_items = turn_input.request_input_items()
            for event in prelude.image_events:
                yield self._publish_event(event)
            title_task = self._start_title_generation_task(
                thread_id,
                user_text,
                should_generate=prelude.should_generate_title,
                level=level,
            )

            final_text = ""
            stream_state = StreamResponseState()
            try:
                for round_index in range(self.config.runtime.max_agent_rounds):
                    self._raise_if_cancelled(cancel_event)
                    async for event in self._stream_model_response_with_retries(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        turn_started_at=turn_started_event.get("created_at"),
                        input_items=request_input_items,
                        level=level,
                        instructions=system_instructions,
                        previous_response_id=turn_input.request_previous_response_id(),
                        stream_state=stream_state,
                        cancel_event=cancel_event,
                    ):
                        yield self._publish_event(event)
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
                            yield self._publish_event(public_event)
                            if tool_event.get("type") != "tool.output":
                                continue
                            result = self._tool_result_from_event(tool_event, public_event)
                            self.thread_store.append(
                                thread_id,
                                "item.tool_output",
                                turn_id=turn_id,
                                item=result.tool_output,
                            )
                            turn_input.note_tool_output(result.tool_output)
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
                        turn_input.note_tool_attachments(round_attachments)
                    request_input_items = turn_input.request_input_items()
                    if guide_event is not None and guide_event.is_set():
                        self.thread_store.append(
                            thread_id,
                            "item.assistant",
                            turn_id=turn_id,
                            text=GUIDED_INPUT_CONTEXT_BRIDGE,
                        )
                        interrupted_event = self.thread_store.append(
                            thread_id,
                            "turn.interrupted",
                            turn_id=turn_id,
                            reason="guided_input",
                            partial_stream=stream_state.saw_stream_output,
                        )
                        yield self._publish_event({
                            "type": "turn.interrupted",
                            "thread_id": thread_id,
                            "turn_id": turn_id,
                            "reason": "guided_input",
                            "partial_stream": stream_state.saw_stream_output,
                            "created_at": interrupted_event.get("created_at"),
                            "completed_at": interrupted_event.get("created_at"),
                        })
                        if title_task is not None:
                            title_task.cancel()
                        return
                    if self._will_compact_after_tool_results(
                        thread_id,
                        input_items,
                        level=level,
                        instructions=system_instructions,
                    ):
                        yield self._publish_event(self._compaction_started_event(thread_id, turn_id))
                    mid_turn_compaction = await self._maybe_compact_after_tool_results(
                        thread_id,
                        turn_id,
                        input_items,
                        level=level,
                        instructions=system_instructions,
                    )
                    if mid_turn_compaction.token_warning_event is not None:
                        yield self._publish_event(self._public_event(mid_turn_compaction.token_warning_event))
                    if mid_turn_compaction.result is not None:
                        compacted_this_turn = True
                        yield self._publish_event(self._compaction_completed_event(
                            thread_id,
                            turn_id,
                            mid_turn_compaction.result,
                        ))
                        input_items = self._input_after_compaction(
                            thread_id,
                            mid_turn_compaction.result,
                            continue_without_current_user=True,
                        )
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
                yield self._publish_event({
                    "type": "turn.interrupted",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "reason": "user_interrupt",
                    "partial_stream": stream_state.saw_stream_output,
                })
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
                yield self._publish_event({
                    "type": "turn.error",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_event.get("created_at"),
                    "created_at": error_event.get("created_at"),
                    "completed_at": error_event.get("created_at"),
                    "error_type": exc.__class__.__name__,
                    "message": str(exc) or repr(exc),
                    "retryable": is_retryable_provider_error(exc),
                })
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
                    yield self._publish_event(self._compaction_started_event(thread_id, turn_id))
                compacted = await self._maybe_compact(
                    thread_id,
                    turn_id,
                    input_items,
                    level=level,
                    instructions=system_instructions,
                )
            if compacted.token_warning_event is not None:
                yield self._publish_event(self._public_event(compacted.token_warning_event))
            if compacted.result is not None:
                yield self._publish_event(self._compaction_completed_event(thread_id, turn_id, compacted.result))
            generated_title = await self._finish_title_generation(title_task)
            if generated_title:
                yield self._publish_event({
                    "type": "thread.title",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "title": generated_title,
                })
            yield self._publish_event({
                "type": "turn.completed",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "turn_started_at": turn_started_event.get("created_at"),
                "created_at": turn_completed_event.get("created_at"),
                "completed_at": turn_completed_event.get("created_at"),
                "final_text": final_text,
            })
            self._turns_since_db_checkpoint += 1
            if self._turns_since_db_checkpoint >= self._db_checkpoint_interval:
                self._turns_since_db_checkpoint = 0
                try:
                    await asyncio.to_thread(checkpoint_state_db, self.thread_store.data_dir, mode="PASSIVE")
                except Exception:
                    # Best-effort WAL checkpoint; failures should not break turns.
                    pass

    def _prepare_run_turn_prelude(
        self,
        *,
        user_text: str,
        thread_id: str | None,
        level: str | None,
        image_paths: list[str | Path] | None,
        cancel_event: asyncio.Event | None,
        is_new_thread: bool = False,
    ) -> RunTurnPrelude:
        self._raise_if_cancelled(cancel_event)
        self.refresh_config(force=True)
        if thread_id is None:
            raise ValueError("thread_id is required after run_turn creates the thread")
        system_instructions = self._system_instructions_for_turn(thread_id)
        turn_id = new_id("turn")
        should_generate_title = self._should_generate_title(thread_id)
        metadata = dict(self.thread_store.thread_metadata(thread_id))
        last_turn_completed_at = self._latest_event_created_at(
            thread_id,
            event_types={"turn.completed"},
        )
        last_assistant_completed_at = self._latest_event_created_at(
            thread_id,
            event_types={"item.model_response", "item.assistant"},
        )
        turn_input = self._prepare_turn_input(thread_id, level=level)
        input_items = turn_input.input_items
        request_input_items = turn_input.request_input_items()
        pre_user_items = self._pre_user_context_items(thread_id)
        turn_started_event = self.thread_store.append(thread_id, "turn.started", turn_id=turn_id)
        prepare_request = TurnPrepareRequest(
            thread_id=thread_id,
            turn_id=turn_id,
            user_text=user_text,
            level=level,
            is_new_thread=is_new_thread,
            is_first_turn=int(metadata.get("user_message_count") or 0) == 0,
            created_at=str(turn_started_event.get("created_at") or "") or None,
            last_turn_completed_at=last_turn_completed_at,
            last_assistant_completed_at=last_assistant_completed_at,
            metadata=metadata,
        )
        user_item = message_item("user", user_text)

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
        image_events: list[dict[str, Any]] = []
        for image_path in image_paths or []:
            attachment = self.attachments.register_image(
                image_path,
                cwd=self.project_root,
                thread_id=thread_id,
                note="pasted from clipboard",
            )
            payload = attachment.to_event_payload()
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
            prepare_request=prepare_request,
            user_item=user_item,
            image_events=image_events,
        )

    def _latest_event_created_at(self, thread_id: str, *, event_types: set[str]) -> str | None:
        events, _ = self.thread_store.read_recent_events(thread_id, limit=1, event_types=event_types)
        if not events:
            return None
        created_at = events[-1].get("created_at")
        return str(created_at) if created_at else None

    def _append_plugin_context_items(
        self,
        *,
        thread_id: str,
        turn_id: str,
        turn_input: TurnInputState,
        blocks: list[TurnContextBlock],
    ) -> None:
        for block in blocks:
            text = self._plugin_context_text(block)
            if not text:
                continue
            self.thread_store.append(
                thread_id,
                "item.plugin_context",
                turn_id=turn_id,
                plugin=block.plugin,
                placement=block.placement,
                dedupe_key=block.dedupe_key,
                metadata=block.metadata,
                text=text,
            )
            item = message_item("user", text)
            turn_input.input_items.append(item)
            turn_input.pending_items.append(copy.deepcopy(item))

    @staticmethod
    def _plugin_context_text(block: TurnContextBlock) -> str:
        body = str(block.text or "").strip()
        if not body:
            return ""
        plugin = block.plugin or "unknown"
        attrs = [f'plugin="{_xml_attr(plugin)}"']
        if block.dedupe_key:
            attrs.append(f'dedupe_key="{_xml_attr(block.dedupe_key)}"')
        return f"<{PLUGIN_CONTEXT_TAG} {' '.join(attrs)}>\n{body}\n</{PLUGIN_CONTEXT_TAG}>"

    def _append_current_user_items(
        self,
        *,
        thread_id: str,
        turn_id: str,
        turn_input: TurnInputState,
        user_item: dict[str, Any],
        image_events: list[dict[str, Any]],
    ) -> None:
        """Persist and append the real user task after optional pre-turn work."""

        self.thread_store.append(thread_id, "item.user", turn_id=turn_id, item=user_item)
        turn_input.input_items.append(user_item)
        turn_input.pending_items.append(copy.deepcopy(user_item))
        for event in image_events:
            attachment = event.get("attachment") if isinstance(event, dict) else None
            if not isinstance(attachment, dict):
                continue
            self.thread_store.append(
                thread_id,
                "item.image_attachment",
                turn_id=turn_id,
                attachment=attachment,
            )
            image_item = image_message_item(attachment)
            turn_input.input_items.append(image_item)
            turn_input.pending_items.append(copy.deepcopy(image_item))

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
        await self._ensure_plugins_started()
        self.refresh_config(force=True)
        with self.thread_store.lock_thread(thread_id):
            retry_state = self._prepare_retry_input(thread_id, level=level)
            system_instructions = self._system_instructions_for_turn(thread_id)
            turn_id = new_id("turn")
            turn_started_event = self.thread_store.append(thread_id, "turn.started", turn_id=turn_id, retry=True)
            self.thread_store.append(thread_id, "turn.retry", turn_id=turn_id)
            yield self._publish_event({
                "type": "turn.started",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "turn_started_at": turn_started_event.get("created_at"),
                "retry": True,
            })
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
                            yield self._publish_event(public_event)
                            if tool_event.get("type") != "tool.output":
                                continue
                            result = self._tool_result_from_event(tool_event, public_event)
                            self.thread_store.append(
                                thread_id,
                                "item.tool_output",
                                turn_id=turn_id,
                                item=result.tool_output,
                            )
                            retry_state.note_tool_output(result.tool_output)
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
                        retry_state.note_tool_attachments(round_attachments)
                    if self._will_compact_after_tool_results(
                        thread_id,
                        retry_state.input_items,
                        level=level,
                        instructions=system_instructions,
                    ):
                        yield self._publish_event(self._compaction_started_event(thread_id, turn_id))
                    mid_turn_compaction = await self._maybe_compact_after_tool_results(
                        thread_id,
                        turn_id,
                        retry_state.input_items,
                        level=level,
                        instructions=system_instructions,
                    )
                    if mid_turn_compaction.token_warning_event is not None:
                        yield self._publish_event(self._public_event(mid_turn_compaction.token_warning_event))
                    if mid_turn_compaction.result is not None:
                        yield self._publish_event(self._compaction_completed_event(
                            thread_id,
                            turn_id,
                            mid_turn_compaction.result,
                        ))
                        retry_state.input_items = self._input_after_compaction(
                            thread_id,
                            mid_turn_compaction.result,
                            continue_without_current_user=True,
                        )
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
                        yield self._publish_event(event)
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
                            yield self._publish_event(public_event)
                            if tool_event.get("type") != "tool.output":
                                continue
                            result = self._tool_result_from_event(tool_event, public_event)
                            self.thread_store.append(
                                thread_id,
                                "item.tool_output",
                                turn_id=turn_id,
                                item=result.tool_output,
                            )
                            retry_state.note_tool_output(result.tool_output)
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
                        retry_state.note_tool_attachments(round_attachments)
                    if self._will_compact_after_tool_results(
                        thread_id,
                        retry_state.input_items,
                        level=level,
                        instructions=system_instructions,
                    ):
                        yield self._publish_event(self._compaction_started_event(thread_id, turn_id))
                    mid_turn_compaction = await self._maybe_compact_after_tool_results(
                        thread_id,
                        turn_id,
                        retry_state.input_items,
                        level=level,
                        instructions=system_instructions,
                    )
                    if mid_turn_compaction.token_warning_event is not None:
                        yield self._publish_event(self._public_event(mid_turn_compaction.token_warning_event))
                    if mid_turn_compaction.result is not None:
                        yield self._publish_event(self._compaction_completed_event(
                            thread_id,
                            turn_id,
                            mid_turn_compaction.result,
                        ))
                        retry_state.input_items = self._input_after_compaction(
                            thread_id,
                            mid_turn_compaction.result,
                            continue_without_current_user=True,
                        )
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
                yield self._publish_event({
                    "type": "turn.interrupted",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "reason": "user_interrupt",
                    "partial_stream": stream_state.saw_stream_output,
                })
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
                yield self._publish_event({
                    "type": "turn.error",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_event.get("created_at"),
                    "created_at": error_event.get("created_at"),
                    "completed_at": error_event.get("created_at"),
                    "error_type": exc.__class__.__name__,
                    "message": str(exc) or repr(exc),
                    "retryable": is_retryable_provider_error(exc),
                })
                return

            completed_event = self.thread_store.append(thread_id, "turn.completed", turn_id=turn_id, final_text=final_text)
            yield self._publish_event({
                "type": "turn.completed",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "turn_started_at": turn_started_event.get("created_at"),
                "created_at": completed_event.get("created_at"),
                "completed_at": completed_event.get("created_at"),
                "final_text": final_text,
            })
            return

    def _should_generate_title(self, thread_id: str) -> bool:
        if not self.config.runtime.title_generation.enabled:
            return False
        metadata = self.thread_store.thread_metadata(thread_id)
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
        if not title or not self._thread_title_is_pending(self.thread_store.thread_metadata(thread_id)):
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
            instructions=THREAD_TITLE_INSTRUCTION,
        )
        self._record_billing_charge(
            thread_id,
            None,
            response.usage,
            level=title_level,
            source="title_generation",
        )
        return clean_thread_title(response.output_text)

    async def generate_branch_slug(
        self,
        thread_id: str,
        user_text: str,
        *,
        level: str | None = None,
    ) -> str | None:
        """Generate a safe semantic branch slug for Agent View worktrees.

        This mirrors title generation but is explicitly optional: callers should
        always be prepared to fall back to an id-only branch when model output is
        unavailable or cleans down to an empty string.
        """

        if not self.config.runtime.branch_name_generation.enabled:
            return None
        branch_level = self.config.runtime.branch_name_generation.model_level or level
        response = await asyncio.wait_for(
            self.model_client.create_response(
                input_items=[
                    message_item(
                        "user",
                        BRANCH_NAME_GENERATION_PROMPT
                        + "\n\nUser message:\n"
                        + user_text.strip(),
                    )
                ],
                level=branch_level,
                tools=[],
                instructions=BRANCH_SLUG_INSTRUCTION,
            ),
            timeout=max(0.1, self.config.runtime.branch_name_generation.timeout_s),
        )
        self._record_billing_charge(
            thread_id,
            None,
            response.usage,
            level=branch_level,
            source="branch_name_generation",
        )
        return clean_branch_slug(response.output_text)

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
        # Mid-turn threshold-triggered compaction: retain 25% of the context
        # window verbatim as a safety net against context overflow.
        compact_model = self.config.model_for_level(
            self.config.runtime.compression.model_level or level
        )
        mid_retain_K = int(0.25 * compact_model.context_window_tokens)
        result = await self._compact_if_needed(
            thread_id,
            turn_id,
            bridged_input,
            level=level,
            instructions=instructions,
            allow_last_tool_output_truncation=True,
            retain_K=mid_retain_K,
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

    async def _stream_judge_and_compact(
        self,
        *,
        thread_id: str,
        turn_id: str,
        input_items: list[dict[str, Any]],
        turn_input: TurnInputState,
        user_item: dict[str, Any],
        image_events: list[dict[str, Any]],
        system_instructions: str,
        level: str | None,
        cancel_event: asyncio.Event | None,
        judge_state: JudgeRunState,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream cache-aware judge lifecycle events while optionally compacting.

        The TUI can only show ``judge.started`` if the event reaches ``run_turn``
        before the judge model request blocks.  Keep the lifecycle events in this
        async generator instead of buffering them until the full judge/compaction
        operation has already completed.
        """

        def _done(compacted: bool = False) -> None:
            judge_state.compacted = compacted

        compact_level = self.config.runtime.compression.model_level or level
        compact_model = self.config.model_for_level(compact_level)
        judge_level = self.config.runtime.compression.judge_model_level or compact_level
        ctx = compact_model.context_window_tokens

        image_items = [
            image_message_item(event["attachment"])
            for event in image_events
            if isinstance(event.get("attachment"), dict)
        ]
        # Cheap gate: include the incoming user payload in the estimate without
        # adding it to the historical input that may be compacted.
        total_tokens = estimate_tokens(
            input_items + [user_item, *image_items, message_item("system", system_instructions)]
        )
        if total_tokens < int(ctx * self.config.runtime.compression.judge_min_context_ratio):
            self._record_judge({
                "skipped": True,
                "reason": "below_threshold",
                "total_tokens": total_tokens,
            })
            _done()
            return

        # Skip the judge round entirely when no price/amount is configured.
        # Without pricing the NetGain calculation cannot produce a positive
        # savings, so running the judge would only waste a model call.
        pricing_level = compact_level or level or self.config.runtime.default_level
        model_pricing = pricing_for_model(self.config, compact_model, level=pricing_level)
        if model_pricing is None:
            self._record_judge({
                "skipped": True,
                "reason": "no_pricing",
                "total_tokens": total_tokens,
            })
            _done()
            return

        judge_req_item = compaction_judge_request_item(message_item_text(user_item))
        judge_input = copy.deepcopy(input_items)
        judge_input.append(copy.deepcopy(judge_req_item))

        yield self._publish_event({
            "type": "judge.started",
            "thread_id": thread_id,
            "turn_id": turn_id,
        })

        judge_responses: list[tuple[ModelResponse, list[dict[str, Any]]]] = []
        response: ModelResponse | None = None
        try:
            self._raise_if_cancelled(cancel_event)
            response = await self.model_client.create_response(
                input_items=judge_input,
                level=judge_level,
                tools=[PYTHON_TOOL],
                instructions=system_instructions,
            )
            # Guard against stray tool calls during judge. The synthetic outputs
            # are persisted and replayed as part of the completed judge exchange
            # when no compaction checkpoint supersedes it.
            for _attempt in range(2):
                self._raise_if_cancelled(cancel_event)
                tool_calls = [item for item in response.output if item.get("type") == "function_call"]
                if not tool_calls:
                    break
                judge_input.extend(response.output)
                tool_outputs: list[dict[str, Any]] = []
                for call in tool_calls:
                    synth = function_output(call, {
                        "returncode": 1,
                        "run_id": "(judge-guard)",
                        "timed_out": False,
                        "interrupted": False,
                        "truncated": False,
                        "stdout": "",
                        "stderr": PRE_TURN_JUDGE_ERROR_STDERR,
                    })
                    judge_input.append(synth)
                    tool_outputs.append(synth)
                judge_responses.append((response, tool_outputs))
                response = await self.model_client.create_response(
                    input_items=judge_input,
                    level=judge_level,
                    tools=[PYTHON_TOOL],
                    instructions=system_instructions,
                )
            judge_responses.append((response, []))
        except (asyncio.CancelledError, TurnInterrupted):
            raise
        except Exception as exc:
            self._record_judge({
                "skipped": True,
                "reason": "judge_error",
                "error_type": exc.__class__.__name__,
            })
            yield self._publish_event({
                "type": "judge.completed",
                "thread_id": thread_id,
                "turn_id": turn_id,
            })
            _done()
            return

        self._persist_judge_interaction(
            thread_id=thread_id,
            turn_id=turn_id,
            judge_level=judge_level,
            judge_req_item=judge_req_item,
            judge_responses=judge_responses,
        )
        judge_history_items = self._judge_history_items(judge_req_item, judge_responses)
        yield self._publish_event({
            "type": "judge.completed",
            "thread_id": thread_id,
            "turn_id": turn_id,
        })

        assert response is not None, "judge response must exist after successful judge call"
        judge = parse_judge_response(response.output_text)
        if judge is None:
            self._append_judge_history_to_input(turn_input, input_items, judge_history_items)
            self._record_judge({"skipped": True, "reason": "parse_failed"})
            _done()
            return

        dependency = str(judge.get("history_dependency") or "")
        if dependency == "exact" or dependency not in DEPENDENCY_PARAMS:
            self._append_judge_history_to_input(turn_input, input_items, judge_history_items)
            self._record_judge({
                "skipped": True,
                "reason": "dependency",
                "dependency": dependency,
            })
            _done()
            return

        N = N_BUCKET_MAP.get(str(judge.get("remaining_calls_bucket") or ""))
        if N is None:
            self._append_judge_history_to_input(turn_input, input_items, judge_history_items)
            self._record_judge({
                "skipped": True,
                "reason": "unknown_bucket",
                "bucket": str(judge.get("remaining_calls_bucket") or ""),
            })
            _done()
            return

        S_ratio, K_min_pct = DEPENDENCY_PARAMS[dependency]

        pricing_level = compact_level or level or self.config.runtime.default_level
        model_pricing = pricing_for_model(self.config, compact_model, level=pricing_level)
        if model_pricing is not None:
            divisor = float(unit_divisor(model_pricing.unit or self.config.pricing.unit))
            P_write = (model_pricing.input or 0.0) / divisor
            P_read = (model_pricing.cached_input or 0.0) / divisor
            # When cache reads are priced at zero the NetGain formula cannot
            # generate positive savings (save = replaced*N*0 = 0). Use a small
            # fraction of P_write as a conservative floor so cache-aware
            # compaction can still fire for these models.
            if P_read == 0.0:
                P_read = P_write * 0.01
            P_summary_out = (model_pricing.output or 0.0) / divisor
        else:
            P_write = P_read = P_summary_out = 0.0
        P_summary_in = P_write

        # Enumerate K candidates (percentage of context window).
        K_min = max(500, int(K_min_pct * ctx))
        K_candidates: list[int] = []
        for pct in K_CANDIDATE_PCTS:
            k = max(K_min, int(pct * ctx))
            if k not in K_candidates:
                K_candidates.append(k)
        K_candidates.sort()

        # D/U are based on prior replayable history only. Dynamic epoch context
        # is re-emitted after compaction and should not make Path A summarize an
        # otherwise empty thread.
        history_items = [
            item for item in strip_compaction_judge_history(input_items)
            if not self._is_pre_user_context_item(item)
        ]
        D = estimate_tokens(history_items)
        U = estimate_tokens(retained_user_messages_after_compaction(history_items))

        best_gain = 0.0
        best_K: int | None = None
        best_S: int | None = None

        for K in K_candidates:
            S = max(S_MIN, min(S_MAX, int((D - U) * S_ratio)))
            compact_cost = estimate_compact_cost(
                D=D,
                S=S,
                P_summary_in=P_summary_in,
                P_summary_out=P_summary_out,
            )
            gain = compute_net_gain(
                D=D, U=U, K=K, S=S, N=N,
                P_read=P_read, P_write=P_write,
                compact_cost=compact_cost,
            )
            if gain > best_gain:
                best_gain = gain
                best_K = K
                best_S = S

        if best_K is None:
            self._append_judge_history_to_input(turn_input, input_items, judge_history_items)
            self._record_judge({
                "skipped": True,
                "reason": "no_valid_K",
                "D": D,
                "U": U,
                "N": N,
                "dependency": dependency,
            })
            _done()
            return

        threshold = max(
            self.config.runtime.compression.min_gain,
            estimate_compact_cost(
                D=D, S=best_S or S_MIN,
                P_summary_in=P_summary_in, P_summary_out=P_summary_out,
            ) * self.config.runtime.compression.margin,
        )

        if best_gain <= threshold:
            self._append_judge_history_to_input(turn_input, input_items, judge_history_items)
            self._record_judge({
                "triggered": False,
                "dependency": dependency,
                "N": N,
                "D": D,
                "U": U,
                "best_K": best_K,
                "best_S": best_S,
                "net_gain": best_gain,
                "threshold": threshold,
            })
            _done()
            return

        yield self._publish_event(self._compaction_started_event(thread_id, turn_id))
        compact_result = await self._compact_if_needed(
            thread_id,
            turn_id,
            input_items,
            level=level,
            instructions=system_instructions,
            retain_K=best_K,
            force=True,
        )
        if compact_result.token_warning_event is not None:
            yield self._publish_event(self._public_event(compact_result.token_warning_event))
        if compact_result.result is not None:
            input_items[:] = self._input_after_compaction(thread_id, compact_result.result)
            turn_input.input_items = input_items
            turn_input.previous_response_id = None
            turn_input.use_previous_response_id = False
            turn_input.pending_items.clear()
            yield self._publish_event(self._compaction_completed_event(
                thread_id, turn_id, compact_result.result,
            ))
            self._record_judge({
                "triggered": True,
                "compacted": True,
                "dependency": dependency,
                "N": N,
                "D": D,
                "U": U,
                "best_K": best_K,
                "best_S": best_S,
                "net_gain": best_gain,
                "threshold": threshold,
            })
            _done(compacted=True)
            return

        # Should be rare with force=True, but if compaction is skipped we must
        # replay the already-persisted judge exchange in the main request.
        self._append_judge_history_to_input(turn_input, input_items, judge_history_items)
        self._record_judge({
            "triggered": True,
            "compacted": False,
            "dependency": dependency,
            "N": N,
            "D": D,
            "U": U,
            "best_K": best_K,
            "best_S": best_S,
            "net_gain": best_gain,
            "threshold": threshold,
        })
        _done()
        return

    def _persist_judge_interaction(
        self,
        *,
        thread_id: str,
        turn_id: str,
        judge_level: str | None,
        judge_req_item: dict[str, Any],
        judge_responses: list[tuple[ModelResponse, list[dict[str, Any]]]],
    ) -> None:
        """Persist a completed judge exchange without counting it as user chat."""

        self.thread_store.append(thread_id, "item.judge_request", turn_id=turn_id, item=judge_req_item)
        for resp, tool_outputs in judge_responses:
            self.thread_store.append(
                thread_id,
                "item.judge_response",
                turn_id=turn_id,
                model_api=self._model_api_for_level(judge_level),
                response_id=resp.id,
                output=resp.output,
                usage=resp.usage,
                reasoning_text=resp.reasoning_text,
            )
            for tool_output in tool_outputs:
                self.thread_store.append(
                    thread_id,
                    "item.judge_tool_output",
                    turn_id=turn_id,
                    item=tool_output,
                )
            self._record_billing_charge(
                thread_id, turn_id, resp.usage, level=judge_level, source="judge",
            )

    @staticmethod
    def _judge_history_items(
        judge_req_item: dict[str, Any],
        judge_responses: list[tuple[ModelResponse, list[dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        """Return model-input items representing the completed judge exchange."""

        items = [copy.deepcopy(judge_req_item)]
        for resp, tool_outputs in judge_responses:
            items.extend(copy.deepcopy(resp.output))
            items.extend(copy.deepcopy(tool_outputs))
        return items

    @staticmethod
    def _append_judge_history_to_input(
        turn_input: TurnInputState,
        input_items: list[dict[str, Any]],
        judge_history_items: list[dict[str, Any]],
    ) -> None:
        """Replay a persisted judge exchange and force a full next request."""

        input_items.extend(copy.deepcopy(judge_history_items))
        turn_input.input_items = input_items
        turn_input.previous_response_id = None
        turn_input.use_previous_response_id = False
        turn_input.pending_items.clear()

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
        active_model = self.config.model_for_level(level)
        token_count = self._compaction_token_count(thread_id, input_items, instructions=instructions)
        trigger_tokens = int(active_model.context_window_tokens * self.config.runtime.compression.trigger_ratio)
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
        retain_K: int = 0,
        force: bool = False,
    ) -> CompactionDecision:
        if not self.config.runtime.compression.enabled:
            return CompactionDecision()
        active_model = self.config.model_for_level(level)
        compact_level = self.config.runtime.compression.model_level or level
        compact_model = self.config.model_for_level(compact_level)
        token_count = self._compaction_token_count(thread_id, input_items, instructions=instructions)
        trigger_tokens = int(active_model.context_window_tokens * self.config.runtime.compression.trigger_ratio)
        token_warning_event = None
        if token_count.source == "estimate" and token_count.tokens >= self.config.runtime.compression.min_tokens:
            token_warning_event = self.thread_store.append(
                thread_id,
                "thread.token_estimation_warning",
                turn_id=turn_id,
                message=(
                    TOKEN_ESTIMATION_WARNING
                ),
                used_tokens=token_count.tokens,
                threshold_tokens=trigger_tokens,
                context_window_tokens=active_model.context_window_tokens,
            )
        if not force:
            if token_count.tokens < self.config.runtime.compression.min_tokens:
                return CompactionDecision(token_warning_event=token_warning_event)
            if token_count.tokens < trigger_tokens:
                return CompactionDecision(token_warning_event=token_warning_event)
        if pre_compaction_event is not None:
            event_type = str(pre_compaction_event.get("type") or "")
            payload = {key: value for key, value in pre_compaction_event.items() if key != "type"}
            self.thread_store.append(thread_id, event_type, **payload)
        # Preserve ordinary replay (and provider cache prefixes), but keep
        # internal judge prompts/JSON out of compaction summaries and retained
        # post-compaction history.
        compaction_source_items = strip_compaction_judge_history(input_items)
        compact_input = copy.deepcopy(compaction_source_items)
        compact_input.append(self._compaction_trigger_item())
        truncated_last_tool_output = False
        if allow_last_tool_output_truncation:
            compact_input, truncated_last_tool_output = self._fit_compaction_input_by_truncating_last_tool_output(
                compact_input,
                context_window_tokens=compact_model.context_window_tokens,
            )
        # Expose run_python so the request structure matches normal calls and
        # preserves provider-side prompt caching. When a provider ignores the
        # compaction prompt and emits a function_call anyway, retry with a
        # synthetic error output until the model returns a summary.
        response: ModelResponse | None = None
        for _compaction_attempt in range(3):
            response = await self.model_client.create_response(
                input_items=compact_input,
                level=compact_level,
                tools=[PYTHON_TOOL],
                instructions=instructions,
            )
            tool_calls = [item for item in response.output if item.get("type") == "function_call"]
            if not tool_calls:
                break
            compact_input.extend(response.output)
            for call in tool_calls:
                compact_input.append(function_output(call, {
                    "returncode": 1,
                    "run_id": "(compaction-guard)",
                    "timed_out": False,
                    "interrupted": False,
                    "truncated": False,
                    "stdout": "",
                    "stderr": (
                        COMPACTION_TOOL_ERROR_STDERR
                    ),
                }))
        assert response is not None, "compaction retry loop must produce a response"  # type: ignore[unreachable]
        summary_text = self._compaction_summary_with_active_workflows(
            thread_id,
            compaction_response_summary_text(response),
        )
        replacement_input = self._compaction_replacement_input(compaction_source_items, response, K=retain_K)
        context_state = self._latest_context_state(thread_id)
        self.thread_store.append(
            thread_id,
            "item.compaction",
            turn_id=turn_id,
            text=summary_text,
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
                text=summary_text,
                truncated_last_tool_output=truncated_last_tool_output,
            ),
            token_warning_event=token_warning_event,
        )


    def _compaction_summary_with_active_workflows(self, thread_id: str, summary_text: str) -> str:
        if not self._is_main_agent_thread(thread_id):
            return summary_text
        section = active_workflows_compaction_section(
            self.thread_store.data_dir,
            parent_thread_id=thread_id,
        )
        if not section:
            return summary_text
        base = summary_text.rstrip()
        return f"{base}\n\n{section}" if base else section

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

        events, compaction = self.thread_store.read_after_latest_compaction(
            thread_id,
            event_types={"item.model_response", "item.compaction"},
        )
        for event in reversed(events):
            if event.get("type") not in {"item.model_response", "item.compaction"}:
                continue
            used = usage_token_count(event.get("usage") or {})
            if used is not None:
                return used

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
        """Return a streamed event payload without private storage fields."""

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
        *,
        K: int = 0,
    ) -> list[dict[str, Any]]:
        return compaction_replacement_input(input_items, response, K=K)

    def _retained_user_messages_after_compaction(self, input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return retained_user_messages_after_compaction(input_items)

    @staticmethod
    def _retain_item_after_compaction(item: dict[str, Any]) -> bool:
        return retain_item_after_compaction(item)

    def _input_after_compaction(
        self,
        thread_id: str,
        result: CompactionResult,
        *,
        continue_without_current_user: bool = False,
    ) -> list[dict[str, Any]]:
        """Build post-compaction model input for continuing in the same turn.

        Persisted compaction events intentionally store only the replacement
        history. For an immediate mid-turn continuation we must also prepend the
        freshly re-emitted epoch context, mirroring the ordering used when a
        later turn reconstructs from disk.
        """

        items = self._pre_user_context_items(thread_id) + copy.deepcopy(result.replacement_input)
        if continue_without_current_user:
            items.append(message_item("user", COMPACTION_CONTINUE_WITHOUT_CURRENT_USER))
        return items

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
        return any(marker in text for marker in PRE_USER_CONTEXT_MARKERS)

    @staticmethod
    def _is_replayable_input_event(event: dict[str, Any]) -> bool:
        """Return whether an event contributes ordinary conversation input."""

        return event.get("type") in {
            "item.user",
            "item.plugin_context",
            "item.assistant",
            "item.model_response",
            "item.tool_output",
            "item.image_attachment",
            "item.judge_request",
            "item.judge_response",
            "item.judge_tool_output",
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

        thread_kind = str(self.thread_store.thread_metadata(thread_id).get("kind") or "thread")
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
                if "helper_calls" not in partial_payload:
                    partial_payload["helper_calls"] = extract_runtime_helper_calls(code)
                visible_partial_events = [
                    event
                    for event in partial_payload.get("events", [])
                    if isinstance(event, dict) and event.get("kind") != "enter_dir"
                ]
                partial_payload["events"] = visible_partial_events
                yield self._publish_event({
                    "type": "tool.partial",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_at,
                    "call": call,
                    "tool_call_index": tool_call_index,
                    "output": function_output(call, partial_payload),
                })
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
            runtime_helper_calls = payload.get("helper_calls")
            payload["helper_calls"] = runtime_corrected_helper_calls(
                code,
                runtime_helper_calls if isinstance(runtime_helper_calls, list) else None,
            )
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
        yield self._publish_event(started_event)
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
            yield self._publish_event(event)

    def _publish_event(self, event: dict[str, Any]) -> dict[str, Any]:
        self.events.publish(event)
        return event

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
                stream_state.assistant_text += stream_event.text
                yield self._publish_event({
                    "type": "assistant.delta",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_at,
                    "text": stream_event.text,
                })
            elif stream_event.type == "reasoning_delta" and stream_event.text:
                stream_state.saw_stream_output = True
                stream_state.reasoning_text += stream_event.text
                yield self._publish_event({
                    "type": "assistant.reasoning_delta",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_at,
                    "text": stream_event.text,
                })
            elif stream_event.type == "tool_call_delta" and stream_event.tool_call:
                stream_state.saw_stream_output = True
                yield self._publish_event({
                    "type": "tool.delta",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_started_at": turn_started_at,
                    "tool_call": stream_event.tool_call,
                })
            elif stream_event.type == "completed":
                stream_state.response = stream_event.response
        self._raise_if_cancelled(cancel_event)
        response = stream_state.require_response()
        completed_text_delta = completion_text_delta(
            response.output_text,
            stream_state.assistant_text,
        )
        if completed_text_delta:
            stream_state.assistant_text += completed_text_delta
            yield self._publish_event({
                "type": "assistant.delta",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "turn_started_at": turn_started_at,
                "text": completed_text_delta,
            })
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
        yield self._publish_event({
            "type": "model.response",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "turn_started_at": turn_started_at,
            "response": response,
            "reasoning_text": reasoning_text,
            "billing_charge": billing_charge,
        })

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
        self._publish_host_event(
            {
                "type": "agent.model_call_billed",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "level": level,
                "source": source,
                "usage": usage,
                "billing": payload,
            }
        )
        total = billing_total_from_metadata(
            self.thread_store.thread_metadata(thread_id),
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
                yield self._publish_event({
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
                })
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
        """Roll nested model-call costs into the parent thread total."""

        subthread_id = str(event.get("thread_id") or "")
        if not subthread_id or subthread_id == thread_id:
            return
        try:
            metadata = self.thread_store.thread_metadata(subthread_id)
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
            elif event.get("type") == "item.judge_request":
                flush_tool_attachments()
                input_items.extend(pending_pre_user)
                pending_pre_user.clear()
                input_items.append(copy.deepcopy(event["item"]))
            elif event.get("type") == "item.judge_response":
                flush_tool_attachments()
                input_items.extend(copy.deepcopy(event.get("output") or []))
            elif event.get("type") == "item.judge_tool_output":
                flush_tool_attachments()
                input_items.append(copy.deepcopy(event["item"]))
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
                if event.get("type") == "item.plugin_context":
                    # Plugin context belongs to the turn that requested it.  Unlike
                    # regenerated epoch context (rules/runtime/workflow), keep it
                    # with the following user message after a compaction checkpoint.
                    reached_replayable_history = True
                    events_after_context.append(event)
                    continue
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
            elif event.get("type") == "item.judge_request":
                flush_tool_attachments()
                input_items.extend(pending_pre_user)
                pending_pre_user.clear()
                input_items.append(copy.deepcopy(event["item"]))
            elif event.get("type") == "item.judge_response":
                flush_tool_attachments()
                input_items.extend(copy.deepcopy(event.get("output") or []))
            elif event.get("type") == "item.judge_tool_output":
                flush_tool_attachments()
                input_items.append(copy.deepcopy(event["item"]))
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
        elif event_type == "item.plugin_context":
            text = str(event.get("text") or "")
        elif event_type == "item.goal_mode_notice":
            text = str(event.get("text") or "")
        elif event_type == "item.worktree_notice":
            text = str(event.get("text") or "")
        elif event_type == "item.workflow_context":
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
                            INTERRUPTED_TOOL_ERROR
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
        items.extend(self._workflow_context_items(thread_id))
        items.extend(self._goal_context_items(thread_id))
        items.extend(self._worktree_context_items(thread_id))
        return items


    def _workflow_context_items(self, thread_id: str) -> list[dict[str, Any]]:
        if not self._is_main_agent_thread(thread_id):
            return []
        if self.thread_store.has_event_after_latest_compaction(
            thread_id,
            event_types={"item.workflow_context"},
        ):
            return []
        text = render_workflow_context()
        self.thread_store.append(thread_id, "item.workflow_context", text=text)
        return [message_item("user", text)]

    def _is_main_agent_thread(self, thread_id: str) -> bool:
        try:
            kind = str(self.thread_store.thread_metadata(thread_id).get("kind") or "thread")
        except FileNotFoundError:
            return False
        return kind == "thread"

    def enable_goal_mode(self, thread_id: str, *, objective: str = "") -> GoalState:
        """Enable per-thread goal mode and preserve any existing goal files."""

        state = ensure_goal_files(self.thread_store.data_dir, thread_id, objective=objective)
        event = self.thread_store.append(
            thread_id,
            "thread.goal_mode_updated",
            enabled=True,
            objective=state.objective,
            files=self._goal_files_payload(state),
        )
        return GoalState(
            enabled=True,
            status="enabled",
            paths=state.paths,
            objective=state.objective,
            created_at=state.created_at,
            updated_at=str(event.get("created_at") or state.updated_at),
        )

    def disable_goal_mode(self, thread_id: str) -> GoalState:
        """Disable per-thread goal mode without modifying the durable files."""

        previous = self.goal_state(thread_id)
        if previous is None:
            previous = read_goal_state(self.thread_store.data_dir, thread_id, enabled=False)
        event = self.thread_store.append(
            thread_id,
            "thread.goal_mode_updated",
            enabled=False,
            objective=previous.objective,
            files=self._goal_files_payload(previous),
        )
        return GoalState(
            enabled=False,
            status="disabled",
            paths=previous.paths,
            objective=previous.objective,
            created_at=previous.created_at,
            updated_at=str(event.get("created_at") or previous.updated_at),
        )

    def reset_goal_files(self, thread_id: str, *, objective: str = "") -> GoalState:
        """Reset the durable goal files while leaving goal mode disabled."""

        current = self.goal_state(thread_id)
        if current is not None and current.enabled:
            raise ValueError("goal files can only be reset while goal mode is disabled")
        state = ensure_goal_files(self.thread_store.data_dir, thread_id, objective=objective, reset=True)
        event = self.thread_store.append(
            thread_id,
            "thread.goal_files_reset",
            objective=state.objective,
            files=self._goal_files_payload(state),
        )
        return GoalState(
            enabled=False,
            status="disabled",
            paths=state.paths,
            objective=state.objective,
            created_at=state.created_at,
            updated_at=str(event.get("created_at") or state.updated_at),
        )

    def goal_state(self, thread_id: str | None) -> GoalState | None:
        """Return the current goal-mode state for a thread, if one is active."""

        if not thread_id:
            return None
        try:
            metadata = self.thread_store.thread_metadata(thread_id)
        except FileNotFoundError:
            return None
        raw_goal = metadata.get("goal_mode")
        enabled = isinstance(raw_goal, dict) and bool(raw_goal.get("enabled"))
        return read_goal_state(self.thread_store.data_dir, thread_id, enabled=enabled)

    def _goal_context_items(self, thread_id: str) -> list[dict[str, Any]]:
        notice = self._goal_mode_notice_text(thread_id)
        if not notice:
            return []
        status = "enabled" if GOAL_MODE_ENABLED_STATUS_FRAGMENT in notice else "disabled"
        self.thread_store.append(
            thread_id,
            "item.goal_mode_notice",
            text=notice,
            status=status,
        )
        return [message_item("user", notice)]

    def _goal_mode_notice_text(self, thread_id: str) -> str:
        state = self.goal_state(thread_id)
        if state is None:
            return ""
        previous_notice = self._latest_goal_notice_status(thread_id)
        if state.enabled:
            if previous_notice in {"enabled", "pending_disabled"}:
                return ""
            return render_goal_mode_notice(state, status="enabled")
        if previous_notice == "pending_disabled":
            return render_goal_mode_notice(state, status="disabled")
        return ""

    def _latest_goal_notice_status(self, thread_id: str) -> str | None:
        events, _ = self.thread_store.read_after_latest_compaction(
            thread_id,
            event_types={"item.goal_mode_notice", "thread.goal_mode_updated", "thread.goal_files_reset"},
        )
        status: str | None = None
        for event in events:
            event_type = event.get("type")
            if event_type == "item.goal_mode_notice":
                status = str(event.get("status") or "") or None
            elif event_type == "thread.goal_mode_updated":
                enabled = bool(event.get("enabled"))
                if enabled:
                    status = "pending_enabled"
                else:
                    status = "pending_disabled"
            elif event_type == "thread.goal_files_reset":
                # Reset is only allowed while disabled; it does not require a
                # model-visible notice unless the mode is enabled afterwards.
                status = None
        return status

    @staticmethod
    def _goal_files_payload(state: GoalState) -> dict[str, str]:
        return {
            "state": str(state.paths.state),
            "checklist": str(state.paths.checklist),
            "notes": str(state.paths.notes),
        }

    def _worktree_context_items(self, thread_id: str) -> list[dict[str, Any]]:
        notice = self._worktree_notice_text(thread_id)
        if not notice:
            return []
        status = "active" if WORKTREE_ACTIVE_STATUS_FRAGMENT in notice else "deleted"
        self.thread_store.append(
            thread_id,
            "item.worktree_notice",
            text=notice,
            status=status,
        )
        return [message_item("user", notice)]

    def _worktree_notice_text(self, thread_id: str) -> str:
        state = self._worktree_state(thread_id)
        if state is None:
            return ""
        previous_notice = self._latest_worktree_notice_status(thread_id)
        status = str(state.get("worktree_status") or "").strip()
        if status == "active":
            if previous_notice == "active":
                return ""
            return render_worktree_notice(state, status="active")
        if status == "deleted" and previous_notice == "pending_deleted":
            return render_worktree_notice(state, status="deleted")
        return ""

    def _worktree_state(self, thread_id: str) -> dict[str, Any] | None:
        try:
            metadata = self.thread_store.thread_metadata(thread_id)
        except FileNotFoundError:
            return None
        status = str(metadata.get("worktree_status") or "").strip()
        if status not in {"active", "deleted"}:
            return None
        branch = str(metadata.get("worktree_branch") or "").strip()
        path = str(metadata.get("worktree_path") or "").strip()
        if not branch or not path:
            return None
        return dict(metadata)

    def _latest_worktree_notice_status(self, thread_id: str) -> str | None:
        events, _ = self.thread_store.read_after_latest_compaction(
            thread_id,
            event_types={"item.worktree_notice", "thread.worktree_created", "thread.worktree_deleted"},
        )
        status: str | None = None
        for event in events:
            event_type = event.get("type")
            if event_type == "item.worktree_notice":
                status = str(event.get("status") or "") or None
            elif event_type == "thread.worktree_created":
                status = "pending_active"
            elif event_type == "thread.worktree_deleted":
                status = "pending_deleted"
        return status

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
        text = ACTIVE_CWD_NOTICE_TEMPLATE.format(
            active_cwd_rel=xml_text(self._relative_to_project(active_cwd)),
            initial_cwd_rel=xml_text(self._relative_to_project(initial_cwd)),
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
            self._rule_states.move_to_end(thread_id)
            return state
        active_cwd = self._latest_active_cwd(thread_id)
        state = RuleRuntimeState(
            active_cwd=active_cwd,
            loaded_rule_paths=self._loaded_rule_paths_in_epoch(thread_id),
            index_emitted=self._rule_index_already_emitted(thread_id, active_cwd),
            cwd_notice_cwd=self._latest_cwd_notice_in_epoch(thread_id),
        )
        self._rule_states[thread_id] = state
        while len(self._rule_states) > self._rule_states_max_size:
            self._rule_states.popitem(last=False)
        return state

    def _latest_active_cwd(self, thread_id: str) -> Path:
        raw_cwd = self.thread_store.thread_metadata(thread_id).get("latest_cwd")
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
        events, _ = self.thread_store.read_after_latest_compaction(
            thread_id,
            event_types={"item.rules_loaded"},
        )
        loaded: set[Path] = set()
        for event in events:
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
        return self.thread_store.has_event_after_latest_compaction(
            thread_id,
            event_types={"item.rule_index"},
        )

    def _latest_cwd_notice_in_epoch(self, thread_id: str) -> Path | None:
        event = self.thread_store.latest_event_after_latest_compaction(
            thread_id,
            event_types={"item.cwd_notice"},
        )
        if event is None:
            return None
        raw_cwd = event.get("cwd")
        if not isinstance(raw_cwd, str) or not raw_cwd:
            return None
        try:
            cwd = Path(raw_cwd).resolve()
        except OSError:
            return None
        return cwd if self._is_within_project(cwd) else None

    def _has_compaction(self, thread_id: str, *, snapshot: ThreadSnapshot | None = None) -> bool:
        if snapshot is not None:
            return snapshot.latest_compaction is not None
        metadata = self.thread_store.thread_metadata(thread_id)
        return metadata.get("latest_compaction") is not None or metadata.get("latest_compaction_event_id") is not None

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
                "text": CONTEXT_REMOVED_ALL,
            }
        if initial:
            text = rendered
        else:
            body_parts = [CONTEXT_UPDATE_CURRENT_PREFIX]
            if removed:
                body_parts.append(
                    CONTEXT_REMOVED_SOME_PREFIX
                    + _removed_context_text(removed, previous_parts)
                    + CONTEXT_REMOVED_SOME_SUFFIX
                )
            if rendered:
                body_parts.append(rendered)
            text = "\n\n".join(body_parts) + CONTEXT_UPDATE_CURRENT_SUFFIX
        return {
            "fingerprint": fingerprint,
            "state": {"fingerprint": fingerprint, "parts": state_parts},
            "removed": removed,
            "text": text,
        }

    def _latest_context_state(self, thread_id: str | None) -> dict[str, Any] | None:
        if not thread_id:
            return None
        try:
            event = self.thread_store.latest_event_after_latest_compaction(
                thread_id,
                event_types={"item.context_update"},
            )
        except FileNotFoundError:
            return None
        if event is None:
            return None
        state = event.get("context_state")
        if isinstance(state, dict):
            return state
        return {"fingerprint": str(event.get("context_fingerprint") or ""), "parts": {}}

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
        plugin_part = self._plugin_runtime_helpers_context()
        if plugin_part:
            parts.append(ContextPart("plugin_runtime_helpers", "plugin_runtime_helpers", plugin_part, dynamic=True))
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
                    SKILLS_HEADER
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
                    SKILLS_OMITTED_TEMPLATE.format(count=len(skills) - 10),
                    dynamic=True,
                )
            )
        parts.append(ContextPart("skills/footer", "skills", AVAILABLE_SKILLS_FOOTER))
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
                    MCP_SERVERS_HEADER
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
                    MCP_OMITTED_TEMPLATE.format(count=len(servers) - 10),
                    dynamic=True,
                )
            )
        parts.append(ContextPart("mcp/footer", "mcp", AVAILABLE_MCP_SERVERS_FOOTER))
        return parts

    def project_rule_context(self) -> ProjectRuleContext:
        """Load AGENTS.md context for status/debug display."""
        return load_project_rules(self.project_root)

    def context_percent(self, thread_id: str | None, level: str | None = None) -> int:
        """Return a context-window usage percentage for a thread."""
        return self.context_stats(thread_id, level).percent

    def _empty_context_stats(self, level: str | None) -> ContextStats:
        """Return zero-usage stats for a missing thread."""

        model = self.config.model_for_level(level)
        trigger_tokens = int(
            model.context_window_tokens * self.config.runtime.compression.trigger_ratio
        )
        return ContextStats(
            used_tokens=0,
            context_window_tokens=model.context_window_tokens,
            percent=0,
            threshold_tokens=trigger_tokens,
            headroom_tokens=model.context_window_tokens,
            source="empty",
        )

    def context_stats(self, thread_id: str | None, level: str | None = None) -> ContextStats:
        """Return detailed context-window statistics for a thread."""

        from time import monotonic as _monotonic

        cache_key = (thread_id, level)
        cached = self._context_stats_cache.get(cache_key)
        if cached is not None:
            cached_at, stats = cached
            if _monotonic() - cached_at < self._context_stats_ttl_s:
                return stats

        model = self.config.model_for_level(level)
        trigger_tokens = int(
            model.context_window_tokens * self.config.runtime.compression.trigger_ratio
        )
        if not thread_id:
            stats = ContextStats(
                used_tokens=0,
                context_window_tokens=model.context_window_tokens,
                percent=0,
                threshold_tokens=trigger_tokens,
                headroom_tokens=model.context_window_tokens,
                source="empty",
            )
            self._context_stats_cache[cache_key] = (_monotonic(), stats)
            return stats
        try:
            metadata = self.thread_store.thread_metadata(thread_id)
        except FileNotFoundError:
            stats = ContextStats(
                used_tokens=0,
                context_window_tokens=model.context_window_tokens,
                percent=0,
                threshold_tokens=trigger_tokens,
                headroom_tokens=model.context_window_tokens,
                source="empty",
            )
            self._context_stats_cache[cache_key] = (_monotonic(), stats)
            return stats
        used = metadata.get("latest_usage_tokens") if isinstance(metadata.get("latest_usage_tokens"), int) else None
        source = "provider"
        if used is None:
            snapshot = self.thread_store.snapshot(thread_id)
            used = self._latest_usage_tokens(thread_id, snapshot=snapshot)
            if used is None:
                update = self._turn_context_update(thread_id)
                context_items = [message_item("user", update["text"])] if update else []
                used = estimate_tokens(self._reconstruct_input(thread_id, snapshot=snapshot) + context_items)
                source = "estimate"
        percent = min(100, max(0, round(used * 100 / model.context_window_tokens)))
        stats = ContextStats(
            used_tokens=used,
            context_window_tokens=model.context_window_tokens,
            percent=percent,
            threshold_tokens=trigger_tokens,
            headroom_tokens=max(0, model.context_window_tokens - used),
            source=source,
        )
        self._context_stats_cache[cache_key] = (_monotonic(), stats)
        return stats

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

    def _plugin_runtime_helpers_context(self) -> str:
        helpers = self.plugins.helper_specs()
        if not helpers:
            return ""
        lines = [
            PLUGIN_HELPERS_HEADER,
        ]
        # TODO: When plugins can load or unload after startup, split this into
        # per-plugin context parts so runtime context updates can mention only
        # the changed plugin. Startup-time loading means the current block-level
        # refresh is enough for now.
        for helper in helpers:
            lines.append(
                PLUGIN_HELPER_ENTRY_TEMPLATE.format(
                    name=xml_text(helper.name),
                    plugin=xml_text(helper.plugin),
                    signature=xml_text(_handler_signature(helper.name, helper.schema)),
                    doc=xml_text(helper.doc),
                )
            )
        lines.append(PLUGIN_HELPERS_FOOTER)
        return "\n".join(lines)

def _handler_signature(name: str, schema: dict[str, Any]) -> str:
    properties = schema.get("properties") if isinstance(schema, dict) else None
    required = set(schema.get("required") or []) if isinstance(schema, dict) else set()
    if not isinstance(properties, dict) or not properties:
        return f"rt.{name}(payload: dict) -> Any"
    parts = []
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            typ = "Any"
            default = None
        else:
            typ = _json_schema_type_name(prop_schema.get("type"), prop_schema.get("enum"))
            default = prop_schema.get("default")
        if prop_name in required:
            parts.append(f"{prop_name}: {typ}")
        elif "default" in (prop_schema if isinstance(prop_schema, dict) else {}):
            parts.append(f"{prop_name}: {typ} = {default!r}")
        else:
            parts.append(f"{prop_name}: {typ} | None = None")
    return f"rt.{name}({', '.join(parts)}) -> Any"


def _json_schema_type_name(value: Any, enum: Any = None) -> str:
    if isinstance(enum, list) and enum and all(isinstance(item, str) for item in enum[:8]):
        return "Literal[" + ", ".join(repr(item) for item in enum[:8]) + "]"
    values = value if isinstance(value, list) else [value]
    mapped = []
    for item in values:
        mapped.append({
            "string": "str",
            "boolean": "bool",
            "integer": "int",
            "number": "float",
            "object": "dict",
            "array": "list",
            "null": "None",
        }.get(str(item), "Any"))
    result = " | ".join(dict.fromkeys(mapped))
    return result or "Any"


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
        return _head_tail_truncate_text(raw_output, max_chars, marker=TOOL_OUTPUT_TRUNCATED_MARKER)
    if not isinstance(payload, dict):
        return _head_tail_truncate_text(raw_output, max_chars, marker=TOOL_OUTPUT_TRUNCATED_MARKER)
    if max_chars <= 0:
        return json.dumps(
            {
                "truncated_for_context_compaction": True,
                "truncation_note": TOOL_OUTPUT_OMITTED_NOTE,
                "original_json_length": len(raw_output),
            },
            ensure_ascii=False,
        )

    truncated = copy.deepcopy(payload)
    truncated["truncated_for_context_compaction"] = True
    truncated["truncation_note"] = TOOL_OUTPUT_SHORTENED_NOTE
    original_lengths = {
        key: len(value)
        for key, value in payload.items()
        if key in {"stdout", "stderr", "output"} and isinstance(value, str)
    }
    if original_lengths:
        truncated["original_text_lengths"] = original_lengths

    large_keys = [key for key in ("stdout", "stderr", "output") if isinstance(truncated.get(key), str)]
    if not large_keys:
        return _head_tail_truncate_text(raw_output, max_chars, marker=TOOL_OUTPUT_TRUNCATED_MARKER)

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


def clean_branch_slug(text: str, *, max_length: int = 30) -> str | None:
    """Return a flat ASCII branch slug or ``None`` if nothing usable remains."""

    slug = text.strip().splitlines()[0].strip().lower()
    slug = slug.strip(" \t\r\n\"'`“”‘’")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        return None
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug or None


def completion_text_delta(output_text: str, emitted_text: str) -> str:
    if not output_text:
        return ""
    if not emitted_text:
        return output_text
    if output_text.startswith(emitted_text):
        return output_text[len(emitted_text) :]
    return ""
