from __future__ import annotations

import copy
import json as _json
import re as _re
from typing import Any

from uv_agent.context import estimate_tokens
from uv_agent.agent.context_builder import xml_text
from uv_agent.agent.messages import message_item, message_item_text
from uv_agent.prompts import (
    COMPACTED_CONTEXT_CONTINUATION,
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
    RETAINED_HISTORY_EMPTY_MESSAGE_TEMPLATE,
    RETAINED_HISTORY_MARKER,
    RETAINED_HISTORY_MESSAGE_TEMPLATE,
    RETAINED_TOOL_CALL_TEMPLATE,
    RETAINED_TOOL_FALLBACK_NAME,
    RETAINED_TOOL_OUTPUT_TEMPLATE,
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


def retain_recent_context(
    input_items: list[dict[str, Any]],
    K: int,
) -> list[dict[str, Any]]:
    """Return the most recent replay-safe context items totalling up to K tokens.

    Provider tool-call protocol items are converted to ordinary retained-history
    messages.  Replaying raw ``function_call`` / ``function_call_output`` items
    after a compaction can create an invalid or dangling tool-call sequence, but
    the textual artefact is still useful as recent context for the model.
    """
    selected: list[dict[str, Any]] = []
    remaining = K
    for item in reversed(input_items):
        candidate = _recent_context_candidate_item(item)
        if candidate is None:
            continue
        tokens = estimate_tokens([candidate])
        if tokens <= remaining:
            selected.append(candidate)
            remaining -= tokens
        else:
            text = message_item_text(candidate)
            if remaining > 0 and text:
                truncated = truncate_text_to_estimated_tokens(text, remaining)
                selected.append(message_item(str(candidate.get("role") or "user"), truncated))
            break
    selected.reverse()
    return selected


def _recent_context_candidate_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Return a replay-safe item suitable for verbatim recent-history retention."""

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
        return copy.deepcopy(item)
    if typ in {"function_call", "function_call_output"}:
        text = _tool_artifact_history_text(item)
        if text:
            return message_item("user", text)
    return None


def _tool_artifact_history_text(item: dict[str, Any]) -> str:
    """Render a tool protocol item as inert retained-history text."""

    typ = item.get("type")
    call_id = xml_text(str(item.get("call_id") or ""))
    if typ == "function_call":
        name = xml_text(str(item.get("name") or RETAINED_TOOL_FALLBACK_NAME))
        arguments = xml_text(str(item.get("arguments") or ""))
        return RETAINED_TOOL_CALL_TEMPLATE.format(
            name=name,
            call_id=call_id,
            arguments=arguments,
        )
    if typ == "function_call_output":
        output = xml_text(str(item.get("output") or ""))
        return RETAINED_TOOL_OUTPUT_TEMPLATE.format(call_id=call_id, output=output)
    return ""


# ---------------------------------------------------------------------------
# Compaction trigger, replacement, and retention (existing)
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

    When *K* > 0 the replacement contains three parts:
      1. retained old user messages (XML-wrapped, existing logic)
      2. retained recent context (K tokens of verbatim tail)
      3. compaction summary

    When *K* == 0 the behaviour matches the pre-cache-aware code path (used
    by threshold-triggered mid-turn compaction).
    """
    retained_users = retained_user_messages_after_compaction(input_items)
    replacement = retained_history_items(retained_users)
    if K > 0:
        recent = retain_recent_context(input_items, K)
        # Don't double-count items already pulled into retained_users.
        replacement.extend(_retained_history_item(item) for item in recent)
    summary = compaction_response_summary_text(response).strip() or COMPACTION_NO_SUMMARY_FALLBACK
    replacement.append(compaction_summary_item(summary))
    return replacement


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
    """Return the model-visible summary item used to resume after compaction."""

    return message_item(
        "user",
        CONVERSATION_SUMMARY_TEMPLATE.format(
            summary=summary,
            continuation=COMPACTED_CONTEXT_CONTINUATION,
        ),
    )


def retained_history_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap retained pre-compaction messages so future compactions can identify them.

    The retained messages are historical context, not fresh user instructions. We
    keep the original message role and non-text parts (for example images) but
    wrap text parts in an XML-ish envelope to make that boundary explicit to the
    model and easy to filter out during later compactions.
    """

    return [_retained_history_item(item) for item in items]


def normalize_compaction_replacement_input(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return stored replacement input using the current retained-history shape.

    Older thread files may contain replacement inputs written before retained
    history had its own XML envelope. Normalizing on read lets resumed threads
    get the clearer prompt contract without changing the persisted event format.
    """

    normalized: list[dict[str, Any]] = []
    saw_summary = False
    for item in items:
        text = message_item_text(item)
        if not saw_summary and CONVERSATION_SUMMARY_OPEN in text:
            normalized.append(compaction_summary_item(_conversation_summary_text(text) or text))
            saw_summary = True
            continue
        if not saw_summary and _should_wrap_retained_history_item(item):
            normalized.append(_retained_history_item(item))
        else:
            normalized.append(copy.deepcopy(item))
    return normalized


def _should_wrap_retained_history_item(item: dict[str, Any]) -> bool:
    text = message_item_text(item)
    return (
        item.get("type") == "message"
        and item.get("role") in {"user", "assistant"}
        and RETAINED_HISTORY_MARKER not in text
    )


def _conversation_summary_text(text: str) -> str:
    start_tag = CONVERSATION_SUMMARY_OPEN
    end_tag = CONVERSATION_SUMMARY_CLOSE
    start = text.find(start_tag)
    if start < 0:
        return ""
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end < 0:
        return text[start:].strip()
    return text[start:end].strip()


def _retained_history_item(item: dict[str, Any]) -> dict[str, Any]:
    wrapped = copy.deepcopy(item)
    role = str(wrapped.get("role") or "user")
    saw_text = False
    for content in wrapped.get("content") or []:
        if content.get("type") not in TEXT_CONTENT_TYPES:
            continue
        saw_text = True
        text = str(content.get("text") or "")
        content["text"] = RETAINED_HISTORY_MESSAGE_TEMPLATE.format(
            role=xml_text(role),
            text=xml_text(text),
        )
    if not saw_text:
        # Rare, but keeps even text-free retained items visibly inside the
        # retained-history envelope instead of silently passing through as a new
        # user/assistant message.
        content_type = "output_text" if role == "assistant" else "input_text"
        wrapped.setdefault("content", []).insert(
            0,
            {
                "type": content_type,
                "text": RETAINED_HISTORY_EMPTY_MESSAGE_TEMPLATE.format(role=xml_text(role)),
            },
        )
    return wrapped


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
