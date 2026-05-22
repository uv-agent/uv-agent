from __future__ import annotations

import copy
from typing import Any

from uv_agent.context import estimate_tokens
from uv_agent.agent.messages import message_item, message_item_text
from uv_agent.agent.prompts import COMPACTED_CONTEXT_CONTINUATION, COMPACTION_SUMMARIZATION_PROMPT
from uv_agent.model.types import ModelResponse

COMPACTION_USER_MESSAGE_MAX_TOKENS = 20_000


def compaction_trigger_item() -> dict[str, Any]:
    return message_item(
        "user",
        "<context_compaction_request>\n"
        + COMPACTION_SUMMARIZATION_PROMPT
        + "\n\n"
        + "Return only the continuation summary. Preserve user intent, decisions, "
        + "file changes, tool results, and unresolved tasks. Do not restate AGENTS "
        + "directory rules; they are reloaded automatically when needed.\n"
        + "</context_compaction_request>",
    )


def compaction_replacement_input(
    input_items: list[dict[str, Any]],
    response: ModelResponse,
) -> list[dict[str, Any]]:
    replacement = retained_user_messages_after_compaction(input_items)
    summary = response.output_text.strip() or "(no summary available)"
    replacement.append(
        message_item(
            "user",
            "<conversation_summary>\n"
            + summary
            + "\n</conversation_summary>\n"
            + COMPACTED_CONTEXT_CONTINUATION,
        )
    )
    return replacement


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
    if item.get("role") == "assistant":
        return "Context is being compacted before the assistant continues" in text
    return not (
        "<runtime_environment>" in text
        or "<model_levels>" in text
        or "<runtime_helpers>" in text
        or "<workspace_rules" in text
        or "<workspace_rule_index>" in text
        or "<active_cwd_notice>" in text
        or "<conversation_summary>" in text
        or "<available_skills>" in text
        or "<available_mcp_servers>" in text
        or "<context_update" in text
    )


def truncate_text_to_estimated_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    suffix = "\n[truncated during context compaction]"
    keep = max(0, max_chars - len(suffix))
    return text[:keep].rstrip() + suffix
