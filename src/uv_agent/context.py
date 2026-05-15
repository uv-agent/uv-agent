from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextStats:
    """Token-window statistics for one reconstructed thread prompt."""

    used_tokens: int
    context_window_tokens: int
    percent: int
    threshold_tokens: int
    target_tokens: int
    headroom_tokens: int
    source: str


def estimate_tokens(items: list[dict[str, Any]]) -> int:
    """Estimate token count for model input items with a cheap local fallback."""
    text = json.dumps(items, ensure_ascii=False)
    return max(1, len(text) // 4)


def usage_token_count(usage: dict[str, Any]) -> int | None:
    """Extract a comparable token count from Responses, Chat, or Anthropic usage."""
    direct_keys = ("total_tokens", "total_token_count")
    for key in direct_keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    pairs = [
        ("input_tokens", "output_tokens"),
        ("prompt_tokens", "completion_tokens"),
        ("cache_creation_input_tokens", "cache_read_input_tokens"),
    ]
    total = 0
    found = False
    for left, right in pairs:
        for key in (left, right):
            value = usage.get(key)
            if isinstance(value, int):
                total += value
                found = True
    return total if found else None


def compact_target_tokens(context_window_tokens: int, *, target_ratio: float = 0.3) -> int:
    """Return the desired approximate size after compaction."""
    return max(1, int(context_window_tokens * target_ratio))
