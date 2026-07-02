from __future__ import annotations

import copy
import json as _json
import re as _re
from dataclasses import dataclass
from typing import Any

from uv_agent.context import estimate_tokens
from uv_agent.agent.context_builder import xml_text
from uv_agent.agent.messages import message_item, message_item_text
from uv_agent.prompts import (
    COMPACTED_CONTEXT_CONTINUATION,
    COMPACTION_CONTINUATION_TEMPLATE,
    COMPACTION_HANDOFF_OPEN,
    COMPACTION_HANDOFF_TEMPLATE,
    COMPACTION_JUDGE_REQUEST,
    COMPACTION_NO_SUMMARY_FALLBACK,
    COMPACTION_RETURN_ONLY_INSTRUCTION,
    COMPACTION_SUMMARIZATION_PROMPT,
    COMPACTION_TRUNCATION_SUFFIX,
    CONTEXT_COMPACTION_REQUEST_TEMPLATE,
    CONTEXT_SCAFFOLD_MARKERS,
    CONVERSATION_SUMMARY_CLOSE,
    CONVERSATION_SUMMARY_OPEN,
    CONVERSATION_SUMMARY_TEMPLATE,
    POST_TOOL_COMPACTION_BRIDGE,
    RETAINED_HISTORY_EMPTY_TEMPLATE,
    RETAINED_HISTORY_MARKER,
    RETAINED_HISTORY_MESSAGE_ENTRY_TEMPLATE,
    RETAINED_HISTORY_TEMPLATE,
    RETAINED_HISTORY_TOOL_CALL_ENTRY_TEMPLATE,
    RETAINED_HISTORY_TOOL_FALLBACK_NAME,
    RETAINED_HISTORY_TOOL_OUTPUT_ENTRY_TEMPLATE,
    UPCOMING_USER_TASK_TEMPLATE,
)
from uv_agent.model.types import ModelResponse

# ---------------------------------------------------------------------------
# Cache-aware NetGain compaction judge
# ---------------------------------------------------------------------------


N_BUCKET_MAP: dict[str, int] = {
    "0_10": 5,
    "10_30": 10,
    "30_60": 30,
    "60_plus": 60,
}

# history_dependency -> (S_ratio, K_min_pct)
DEPENDENCY_PARAMS: dict[str, tuple[float, float]] = {
    "low":    (0.04, 0.02),
    "medium": (0.08, 0.04),
    "high":   (0.15, 0.10),
    # "exact" means skip compaction entirely
}

K_CANDIDATE_PCTS = [0.02, 0.05, 0.10, 0.15, 0.25]

S_MIN = 500
S_MAX = 8000


@dataclass(frozen=True)
class RetainedHistoryEntry:
    """One inert entry inside the model-visible <agent_retained_history> block."""

    order: int
    identity: tuple[Any, ...]
    text: str


def compaction_judge_request_item(upcoming_user_text: str | None = None) -> dict[str, Any]:
    """Return the user-role message that asks the model for a compaction judge JSON.

    The judge runs before the real user task is appended to the main turn.  A
    bounded preview lets the judge estimate task complexity without turning the
    actual user message into part of the historical context being compacted.
    """

    text = COMPACTION_JUDGE_REQUEST
    if upcoming_user_text:
        preview = truncate_text_to_estimated_tokens(upcoming_user_text.strip(), 2_000)
        text += "\n" + UPCOMING_USER_TASK_TEMPLATE.format(task=xml_text(preview))
    return message_item("user", text)


def parse_judge_response(text: str) -> dict[str, Any] | None:
    """Extract a compaction judge JSON object from model output text.

    Returns None when the model did not produce a valid judge block.
    """
    if not text:
        return None
    # Look for a JSON object on its own line (possibly surrounded by other text).
    for match in _re.finditer(r'\{[^}]+\}', text):
        try:
            obj = _json.loads(match.group())
        except (_json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "history_dependency" in obj:
            return obj
    return None


def compute_net_gain(
    *,
    D: int,          # compressible tokens being replaced
    U: int,          # retained old user-message tokens (existing logic)
    K: int,          # retained recent context tokens
    S: int,          # estimated summary tokens
    N: int,          # projected remaining calls
    P_read: float,   # cache read price per token
    P_write: float,  # write / uncached input price per token
    compact_cost: float,  # estimated cost of the summary-generation call
) -> float:
    """Return the estimated net gain of compacting now with parameters K, S.

    Simplified MVP: D_c = D-U, D_n = 0, K_c = K+U.
    """
    replaced = max(0, D - U)
    retained = K + U
    save = replaced * N * P_read
    summary_cost = S * (P_write + (N - 1) * P_read)
    cache_rebuild = retained * (P_write - P_read)
    return save - summary_cost - cache_rebuild - compact_cost


def estimate_compact_cost(
    *,
    D: int,
    S: int,
    instruction_tokens: int = 500,
    P_summary_in: float = 0.0,
    P_summary_out: float = 0.0,
) -> float:
    """Estimate the cost of the summary-generation model call in USD."""
    input_tokens = D + instruction_tokens
    return input_tokens * P_summary_in + S * P_summary_out


# ---------------------------------------------------------------------------
# Retained history selection and rendering
# ---------------------------------------------------------------------------


def retain_recent_context(
    input_items: list[dict[str, Any]],
    K: int,
) -> list[dict[str, Any]]:
    """Return recent context as inert retained-history message items.

    Cache-aware compaction stores recent user/assistant/tool protocol history in
    a single textual <agent_retained_history> block.  This helper keeps its historical
    test-facing shape (a list of message items) while using the canonical entry
    tags that are embedded in that block.
    """

    return [message_item("user", entry.text) for entry in _retained_recent_history_entries(input_items, K)]


def _retained_recent_history_entries(
    input_items: list[dict[str, Any]],
    K: int,
) -> list[RetainedHistoryEntry]:
    if K <= 0:
        return []
    selected: list[RetainedHistoryEntry] = []
    remaining = K
    for index in range(len(input_items) - 1, -1, -1):
        entry = _recent_context_candidate_entry(input_items[index], index)
        if entry is None:
            continue
        tokens = _entry_tokens(entry)
        if tokens <= remaining:
            selected.append(entry)
            remaining -= tokens
            if remaining <= 0:
                break
            continue
        if remaining > 0:
            truncated = _recent_context_candidate_entry(input_items[index], index, max_tokens=remaining)
            if truncated is not None and truncated.text:
                selected.append(truncated)
        break
    selected.reverse()
    return selected


def _retained_user_history_entries(input_items: list[dict[str, Any]]) -> list[RetainedHistoryEntry]:
    """Return entries selected by the retained user-message budget."""

    selected: list[RetainedHistoryEntry] = []
    remaining = COMPACTION_USER_MESSAGE_MAX_TOKENS
    for index in range(len(input_items) - 1, -1, -1):
        item = input_items[index]
        if not retain_item_after_compaction(item):
            continue
        entry = _retained_history_entry_from_item(item, index)
        if entry is None:
            continue
        tokens = _entry_tokens(entry)
        if tokens <= remaining:
            selected.append(entry)
            remaining -= tokens
            if remaining <= 0:
                break
            continue
        if remaining > 0:
            truncated = _retained_history_entry_from_item(item, index, max_tokens=remaining)
            if truncated is not None:
                selected.append(truncated)
        break
    selected.reverse()
    return selected


def _recent_context_candidate_entry(
    item: dict[str, Any],
    order: int,
    *,
    max_tokens: int | None = None,
) -> RetainedHistoryEntry | None:
    """Return an inert retained-history entry for a replay item."""

    typ = item.get("type")
    if typ == "message":
        role = item.get("role")
        if role not in ("user", "assistant"):
            return None
        text = message_item_text(item)
        # Skip system/context scaffolding that is re-emitted each epoch, and skip
        # judge prompts so they do not become long-lived user-looking history.
        if any(marker in text for marker in CONTEXT_SCAFFOLD_MARKERS):
            return None
        return _retained_history_entry_from_item(item, order, max_tokens=max_tokens)
    if typ in {"function_call", "function_call_output"}:
        return _retained_history_entry_from_item(item, order, max_tokens=max_tokens)
    return None


def _retained_history_entry_from_item(
    item: dict[str, Any],
    order: int,
    *,
    max_tokens: int | None = None,
    identity_prefix: str = "item",
) -> RetainedHistoryEntry | None:
    typ = item.get("type")
    if typ == "message":
        role = str(item.get("role") or "user")
        if role not in {"user", "assistant"}:
            return None
        body = message_item_text(item)
        if max_tokens is not None:
            body = truncate_text_to_estimated_tokens(body, max_tokens)
        if not body:
            return None
        return RetainedHistoryEntry(
            order=order,
            identity=(identity_prefix, order, "message", role),
            text=RETAINED_HISTORY_MESSAGE_ENTRY_TEMPLATE.format(
                role=xml_text(role),
                text=xml_text(body),
            ),
        )
    if typ == "function_call":
        call_id = str(item.get("call_id") or "")
        name = str(item.get("name") or RETAINED_HISTORY_TOOL_FALLBACK_NAME)
        arguments = str(item.get("arguments") or "")
        if max_tokens is not None:
            arguments = truncate_text_to_estimated_tokens(arguments, max_tokens)
        if not (call_id or name or arguments):
            return None
        return RetainedHistoryEntry(
            order=order,
            identity=(identity_prefix, order, "tool_call", call_id, name),
            text=RETAINED_HISTORY_TOOL_CALL_ENTRY_TEMPLATE.format(
                name=xml_text(name),
                call_id=xml_text(call_id),
                arguments=xml_text(arguments),
            ),
        )
    if typ == "function_call_output":
        call_id = str(item.get("call_id") or "")
        output = str(item.get("output") or "")
        if max_tokens is not None:
            output = truncate_text_to_estimated_tokens(output, max_tokens)
        if not (call_id or output):
            return None
        return RetainedHistoryEntry(
            order=order,
            identity=(identity_prefix, order, "tool_output", call_id),
            text=RETAINED_HISTORY_TOOL_OUTPUT_ENTRY_TEMPLATE.format(
                call_id=xml_text(call_id),
                output=xml_text(output),
            ),
        )
    return None


def _entry_tokens(entry: RetainedHistoryEntry) -> int:
    return estimate_tokens([message_item("user", entry.text)])


def _merge_retained_history_entries(entries: list[RetainedHistoryEntry]) -> list[RetainedHistoryEntry]:
    """Deduplicate retained user history and cache-aware recent context.

    Both selectors may choose the same underlying item. Keep the longest render
    for that item and then restore chronological order inside the single
    <agent_retained_history> block.
    """

    by_identity: dict[tuple[Any, ...], RetainedHistoryEntry] = {}
    for entry in entries:
        current = by_identity.get(entry.identity)
        if current is None or len(entry.text) > len(current.text):
            by_identity[entry.identity] = entry
    return sorted(by_identity.values(), key=lambda entry: entry.order)


def _render_retained_history(entries: list[RetainedHistoryEntry]) -> str:
    merged = _merge_retained_history_entries(entries)
    if not merged:
        return RETAINED_HISTORY_EMPTY_TEMPLATE
    history = "\n\n".join(entry.text for entry in merged if entry.text)
    if not history:
        return RETAINED_HISTORY_EMPTY_TEMPLATE
    return RETAINED_HISTORY_TEMPLATE.format(history=history)


# ---------------------------------------------------------------------------
# Judge-history filtering
# ---------------------------------------------------------------------------


def strip_compaction_judge_history(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compaction input without internal judge exchanges.

    Ordinary replay intentionally keeps judge traffic so providers with long
    prompt-cache lifetimes can reuse the same prefix. Compaction is different:
    the model response becomes durable summary/retained history, so internal
    judge prompts and JSON answers must not be summarized into the next epoch.
    """

    filtered: list[dict[str, Any]] = []
    skipping_judge_exchange = False
    for item in input_items:
        if _is_compaction_judge_request_item(item):
            skipping_judge_exchange = True
            continue
        if skipping_judge_exchange:
            # Persisted judge items are inserted immediately before the real user
            # item. Drop all model/tool protocol artifacts until that user item,
            # then resume normal history so the actual task remains available.
            if _is_non_judge_user_message(item):
                skipping_judge_exchange = False
            else:
                continue
        if _is_compaction_judge_response_item(item):
            continue
        filtered.append(copy.deepcopy(item))
    return filtered


def _is_compaction_judge_request_item(item: dict[str, Any]) -> bool:
    if item.get("type") != "message" or item.get("role") != "user":
        return False
    return message_item_text(item).lstrip().startswith(COMPACTION_JUDGE_REQUEST.strip())


def _is_non_judge_user_message(item: dict[str, Any]) -> bool:
    return (
        item.get("type") == "message"
        and item.get("role") == "user"
        and not _is_compaction_judge_request_item(item)
    )


def _is_compaction_judge_response_item(item: dict[str, Any]) -> bool:
    return (
        item.get("type") == "message"
        and item.get("role") == "assistant"
        and parse_judge_response(message_item_text(item)) is not None
    )


# ---------------------------------------------------------------------------
# Compaction trigger, replacement, and retention
# ---------------------------------------------------------------------------

COMPACTION_USER_MESSAGE_MAX_TOKENS = 20_000
TEXT_CONTENT_TYPES = {"input_text", "output_text", "text", "refusal"}


def compaction_trigger_item() -> dict[str, Any]:
    return message_item(
        "user",
        CONTEXT_COMPACTION_REQUEST_TEMPLATE.format(
            prompt=COMPACTION_SUMMARIZATION_PROMPT,
            return_only_instruction=COMPACTION_RETURN_ONLY_INSTRUCTION,
        ),
    )


def compaction_replacement_input(
    input_items: list[dict[str, Any]],
    response: ModelResponse,
    *,
    K: int = 0,
) -> list[dict[str, Any]]:
    """Build the replacement for pre-compaction history.

    The replacement is a single inert <agent_compaction_handoff> user message.  The
    retained-user selector and the cache-aware recent-context selector are merged
    into one <agent_retained_history> block so user, assistant, tool-call, and
    tool-output history share one tag contract.
    """

    entries = _retained_user_history_entries(input_items)
    if K > 0:
        entries.extend(_retained_recent_history_entries(input_items, K))
    summary = compaction_response_summary_text(response).strip() or COMPACTION_NO_SUMMARY_FALLBACK
    return [compaction_handoff_item(summary, entries=entries)]


def compaction_response_summary_text(response: ModelResponse) -> str:
    """Return the user-visible summary from a compaction model response.

    Some Responses-compatible providers may emit tool calls when tools are
    present. In that case ``output_text`` can be empty even though earlier
    message items may contain useful summary text. Treat the message text as the
    compaction result and ignore function_call items so the checkpoint does not
    become an empty "conversation compacted" block.
    """

    if response.output_text.strip():
        return response.output_text
    parts: list[str] = []
    for item in response.output:
        if item.get("type") != "message":
            continue
        text = message_item_text(item)
        if text:
            parts.append(text)
    return "\n".join(parts)


def compaction_summary_item(summary: str) -> dict[str, Any]:
    """Return a handoff item containing a summary and an empty history block."""

    return compaction_handoff_item(summary, entries=[])


def compaction_handoff_item(
    summary: str,
    *,
    entries: list[RetainedHistoryEntry] | None = None,
    retained_history: str | None = None,
) -> dict[str, Any]:
    """Return the model-visible compaction handoff item."""

    if retained_history is None:
        retained_history = _render_retained_history(entries or [])
    conversation_summary = CONVERSATION_SUMMARY_TEMPLATE.format(summary=summary)
    continuation = COMPACTION_CONTINUATION_TEMPLATE.format(
        continuation=COMPACTED_CONTEXT_CONTINUATION
    )
    return message_item(
        "user",
        COMPACTION_HANDOFF_TEMPLATE.format(
            retained_history=retained_history,
            conversation_summary=conversation_summary,
            continuation=continuation,
        ),
    )


def normalize_compaction_replacement_input(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return stored replacement input in the current handoff shape."""

    source = strip_compaction_judge_history(items)
    for item in source:
        if COMPACTION_HANDOFF_OPEN in message_item_text(item):
            return [copy.deepcopy(item)]
    if not source:
        return []
    summary, retained_items = _legacy_replacement_parts(source)
    entries = [
        entry
        for index, item in enumerate(retained_items)
        if (entry := _recent_context_candidate_entry(item, index)) is not None
    ]
    return [compaction_handoff_item(summary or COMPACTION_NO_SUMMARY_FALLBACK, entries=entries)]


def _legacy_replacement_parts(items: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    summaries: list[str] = []
    retained: list[dict[str, Any]] = []
    for item in items:
        text = message_item_text(item)
        summary = _legacy_conversation_summary_text(text)
        if summary is not None:
            if summary:
                summaries.append(summary)
            continue
        if any(marker in text for marker in CONTEXT_SCAFFOLD_MARKERS):
            continue
        retained.append(copy.deepcopy(item))
    return "\n\n".join(summaries).strip(), retained


def _legacy_conversation_summary_text(text: str) -> str | None:
    start = text.find(CONVERSATION_SUMMARY_OPEN)
    if start < 0:
        return None
    body_start = start + len(CONVERSATION_SUMMARY_OPEN)
    end = text.find(CONVERSATION_SUMMARY_CLOSE, body_start)
    if end < 0:
        return text[body_start:].strip()
    return text[body_start:end].strip()


def retained_user_messages_after_compaction(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = COMPACTION_USER_MESSAGE_MAX_TOKENS
    for item in reversed(input_items):
        if not retain_item_after_compaction(item):
            continue
        tokens = estimate_tokens([item])
        if tokens <= remaining:
            selected.append(copy.deepcopy(item))
            remaining -= tokens
            if remaining <= 0:
                break
            continue
        text = message_item_text(item)
        if remaining > 0 and text:
            selected.append(message_item("user", truncate_text_to_estimated_tokens(text, remaining)))
        break
    selected.reverse()
    return selected


def retain_item_after_compaction(item: dict[str, Any]) -> bool:
    if item.get("type") != "message" or item.get("role") not in {"user", "assistant"}:
        return False
    text = message_item_text(item)
    if RETAINED_HISTORY_MARKER in text:
        return False
    if item.get("role") == "assistant":
        return POST_TOOL_COMPACTION_BRIDGE in text
    return not any(marker in text for marker in CONTEXT_SCAFFOLD_MARKERS)


def truncate_text_to_estimated_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    suffix = COMPACTION_TRUNCATION_SUFFIX
    keep = max(0, max_chars - len(suffix))
    return text[:keep].rstrip() + suffix
