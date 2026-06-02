from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from uv_agent.billing import billing_token_breakdown


STREAM_RATE_WINDOW_S = 3.0
DEFAULT_STREAM_CHARS_PER_SECOND = 100.0
BREATH_CHARS_PER_PHASE = max(1, round(DEFAULT_STREAM_CHARS_PER_SECOND / 12))


@dataclass
class StreamRateEstimator:
    """Estimate model-output character throughput over a sliding time window.

    Timing starts when the first visible model-output text arrives.  The first
    chunk can be arbitrarily large depending on provider buffering, so callers
    may still choose a display fallback while the estimator only has one point.
    """

    window_s: float = STREAM_RATE_WINDOW_S
    first_output_at: float | None = None
    total_chars: int = 0
    _samples: deque[tuple[float, int]] = field(default_factory=deque)

    def observe(self, text: str, *, now: float) -> None:
        """Record newly received model-output text at ``now``."""

        count = len(text)
        if count <= 0:
            return
        if self.first_output_at is None:
            self.first_output_at = now
        self.total_chars += count
        self._samples.append((now, self.total_chars))
        self._trim(now)

    def current_cps(self, *, now: float) -> float | None:
        """Return the current window-average character rate, if started."""

        if self.first_output_at is None:
            return None
        self._trim(now)
        if len(self._samples) < 2:
            elapsed = max(0.0, now - self.first_output_at)
            if self._samples and now - self._samples[-1][0] >= self.window_s:
                return 0.0
            if elapsed <= 0.0:
                return None
            return min(DEFAULT_STREAM_CHARS_PER_SECOND, self.total_chars / elapsed)
        start = max(self.first_output_at, now - self.window_s)
        baseline_chars = 0 if start <= self.first_output_at else self._chars_at_or_before(start)
        elapsed = max(0.0, now - start)
        if elapsed <= 0.0:
            return None
        return max(0.0, (self.total_chars - baseline_chars) / elapsed)

    def display_cps(self, *, now: float, backlog_chars: int) -> float:
        """Return a smooth display speed for queued final-answer text.

        The base pace follows the model throughput estimate.  If a provider sent
        a large chunk, the backlog term drains it over a short horizon rather
        than making the user wait for a strict real-time replay.
        """

        base = self.current_cps(now=now) or DEFAULT_STREAM_CHARS_PER_SECOND
        if backlog_chars <= 0:
            return base
        # A one-ish-second drain horizon keeps large provider chunks from
        # sitting in the queue, while the 12Hz ticker still spreads them over
        # several repaints instead of one abrupt terminal update.
        backlog_cps = backlog_chars / 1.2
        return max(base, backlog_cps)

    def _chars_at_or_before(self, when: float) -> int:
        chars = 0
        for sample_time, sample_chars in self._samples:
            if sample_time > when:
                break
            chars = sample_chars
        return chars

    def _trim(self, now: float) -> None:
        # Keep one sample older than the window as a cumulative baseline.  Without
        # it, the rate would overstate throughput just after the oldest in-window
        # sample is discarded.
        boundary = now - self.window_s
        while len(self._samples) > 1 and self._samples[1][0] <= boundary:
            self._samples.popleft()


@dataclass
class ThreadTokenRatio:
    """Per-thread visible-output character/token ratio."""

    chars: int = 0
    output_tokens: int = 0

    @property
    def available(self) -> bool:
        return self.chars > 0 and self.output_tokens > 0

    @property
    def chars_per_token(self) -> float | None:
        if not self.available:
            return None
        return self.chars / self.output_tokens

    def observe_response(self, *, visible_chars: int, output_tokens: int) -> None:
        if visible_chars <= 0 or output_tokens <= 0:
            return
        self.chars += visible_chars
        self.output_tokens += output_tokens

    def token_rate(self, char_rate: float | None) -> float | None:
        chars_per_token = self.chars_per_token
        if char_rate is None or char_rate <= 0.0 or chars_per_token is None or chars_per_token <= 0:
            return None
        return max(0.0, char_rate / chars_per_token)


def usage_output_tokens(usage: dict[str, Any] | None) -> int:
    """Return provider-reported output tokens from common usage shapes."""

    if not isinstance(usage, dict):
        return 0
    return billing_token_breakdown(usage).output_tokens


def model_response_visible_chars(
    output: list[dict[str, Any]] | None,
    *,
    reasoning_text: str = "",
) -> int:
    """Count visible text available from a model response.

    This intentionally ignores hidden reasoning token counts that providers may
    include in usage.  It counts the text uv-agent can actually observe: final
    message text/refusals, streamed/persisted reasoning text, and tool-call names
    or arguments.
    """

    parts: list[str] = []
    if reasoning_text:
        parts.append(reasoning_text)
    for item in output or []:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type == "message":
            parts.extend(_content_text_parts(item.get("content")))
        elif item_type == "function_call":
            name = item.get("name")
            arguments = item.get("arguments")
            if isinstance(name, str) and name:
                parts.append(name)
            if isinstance(arguments, str) and arguments:
                parts.append(arguments)
        else:
            for key in ("text", "output_text", "name", "arguments"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    parts.append(value)
            parts.extend(_content_text_parts(item.get("content")))
            parts.extend(_content_text_parts(item.get("summary")))
    return sum(len(part) for part in parts)


def tool_delta_visible_text(tool_call: object) -> str:
    """Return the newly streamed textual portion of a tool-call delta."""

    value: object = None
    if isinstance(tool_call, dict):
        value = tool_call.get("arguments_delta")
    else:
        value = getattr(tool_call, "arguments_delta", None)
    return value if isinstance(value, str) else ""


def tool_call_name(tool_call: object) -> str:
    """Return a tool-call function name from a delta-like object, if present."""

    if isinstance(tool_call, dict):
        value = tool_call.get("name")
    else:
        value = getattr(tool_call, "name", None)
    return value if isinstance(value, str) else ""


def tool_call_stream_key(tool_call: object, *, fallback: object = "0") -> str:
    """Return a stable-ish key for one streamed tool call."""

    if isinstance(tool_call, dict):
        value = tool_call.get("index") if tool_call.get("index") is not None else tool_call.get("call_id")
    else:
        index = getattr(tool_call, "index", None)
        value = index if index is not None else getattr(tool_call, "call_id", None)
    return str(value if value is not None else fallback)


def _content_text_parts(content: object) -> list[str]:
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        content_type = part.get("type")
        if content_type in {"input_text", "output_text", "text", "refusal"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return parts
