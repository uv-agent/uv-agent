from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from unicodedata import category
from typing import Any

from uv_agent.billing import billing_token_breakdown


STREAM_RATE_WINDOW_S = 3.0
DEFAULT_STREAM_CHARS_PER_SECOND = 100.0
DEFAULT_STREAM_VISIBLE_UNITS_PER_SECOND = DEFAULT_STREAM_CHARS_PER_SECOND
BREATH_CHARS_PER_PHASE = max(1, round(DEFAULT_STREAM_CHARS_PER_SECOND / 12))


def visible_units(text: str) -> int:
    """Count visible text units used for token-rate estimates.

    The unit is intentionally tokenizer-adjacent rather than linguistically
    precise: CJK characters count one-by-one, contiguous word characters count
    as one unit, and visible punctuation/symbols each count as one unit.
    """

    count, _in_word = _count_visible_units(text, starts_in_word=False)
    return count


def _count_visible_units(text: str, *, starts_in_word: bool) -> tuple[int, bool]:
    count = 0
    in_word = starts_in_word
    for char in text:
        if char.isspace():
            in_word = False
            continue
        if _is_cjk_visible_unit(char):
            count += 1
            in_word = False
            continue
        if char == "_" or char.isalnum():
            if not in_word:
                count += 1
            in_word = True
            continue
        kind = category(char)[0]
        if kind in {"C", "M"}:
            continue
        count += 1
        in_word = False
    return count, in_word


def _is_cjk_visible_unit(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2EBEF
        or 0x3040 <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0xAC00 <= code <= 0xD7AF
        or 0x1100 <= code <= 0x11FF
    )


@dataclass
class StreamRateEstimator:
    """Estimate model-output throughput over a sliding time window.

    Timing starts when the first visible model-output text arrives.  The first
    chunk can be arbitrarily large depending on provider buffering, so callers
    may still choose a display fallback while the estimator only has one point.
    Character throughput drives animation/display pacing; visible-unit
    throughput drives the displayed token-rate estimate.
    """

    window_s: float = STREAM_RATE_WINDOW_S
    first_output_at: float | None = None
    total_chars: int = 0
    total_units: int = 0
    _samples: deque[tuple[float, int, int]] = field(default_factory=deque)
    _in_visible_word_unit: bool = False

    def observe(self, text: str, *, now: float) -> None:
        """Record newly received model-output text at ``now``."""

        count = len(text)
        if count <= 0:
            return
        unit_count, self._in_visible_word_unit = _count_visible_units(
            text,
            starts_in_word=self._in_visible_word_unit,
        )
        if self.first_output_at is None:
            self.first_output_at = now
        self.total_chars += count
        self.total_units += unit_count
        self._samples.append((now, self.total_chars, self.total_units))
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
        baseline_chars = 0 if start <= self.first_output_at else self._values_at_or_before(start)[0]
        elapsed = max(0.0, now - start)
        if elapsed <= 0.0:
            return None
        return max(0.0, (self.total_chars - baseline_chars) / elapsed)

    def current_ups(self, *, now: float) -> float | None:
        """Return the current window-average visible-unit rate, if started."""

        if self.first_output_at is None:
            return None
        self._trim(now)
        if len(self._samples) < 2:
            elapsed = max(0.0, now - self.first_output_at)
            if self._samples and now - self._samples[-1][0] >= self.window_s:
                return 0.0
            if elapsed <= 0.0:
                return None
            return min(DEFAULT_STREAM_VISIBLE_UNITS_PER_SECOND, self.total_units / elapsed)
        start = max(self.first_output_at, now - self.window_s)
        baseline_units = 0 if start <= self.first_output_at else self._values_at_or_before(start)[1]
        elapsed = max(0.0, now - start)
        if elapsed <= 0.0:
            return None
        return max(0.0, (self.total_units - baseline_units) / elapsed)

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

    def _values_at_or_before(self, when: float) -> tuple[int, int]:
        chars = 0
        units = 0
        for sample_time, sample_chars, sample_units in self._samples:
            if sample_time > when:
                break
            chars = sample_chars
            units = sample_units
        return chars, units

    def _trim(self, now: float) -> None:
        # Keep one sample older than the window as a cumulative baseline.  Without
        # it, the rate would overstate throughput just after the oldest in-window
        # sample is discarded.
        boundary = now - self.window_s
        while len(self._samples) > 1 and self._samples[1][0] <= boundary:
            self._samples.popleft()


@dataclass
class ThreadTokenRatio:
    """Per-thread visible-output unit/token ratio."""

    visible_units: int = 0
    output_tokens: int = 0

    @property
    def available(self) -> bool:
        return self.visible_units > 0 and self.output_tokens > 0

    @property
    def visible_units_per_token(self) -> float | None:
        if not self.available:
            return None
        return self.visible_units / self.output_tokens

    def observe_response(
        self,
        *,
        visible_units: int | None = None,
        output_tokens: int = 0,
        visible_chars: int | None = None,
    ) -> None:
        units = visible_units if visible_units is not None else visible_chars or 0
        if units <= 0 or output_tokens <= 0:
            return
        self.visible_units += units
        self.output_tokens += output_tokens

    def token_rate(self, unit_rate: float | None) -> float | None:
        units_per_token = self.visible_units_per_token
        if unit_rate is None or unit_rate <= 0.0 or units_per_token is None or units_per_token <= 0:
            return None
        return max(0.0, unit_rate / units_per_token)

    def to_metadata(self) -> dict[str, int]:
        """Serialize accumulated counts for persistence."""

        return {
            "visible_units": self.visible_units,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_metadata(cls, data: dict[str, Any]) -> ThreadTokenRatio:
        """Restore accumulated counts from persisted metadata."""

        return cls(
            visible_units=int(data.get("visible_units") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
        )


def usage_output_tokens(
    usage: dict[str, Any] | None,
    *,
    reasoning_visible: bool = False,
) -> int:
    """Return provider-reported visible output tokens from common usage shapes.

    When ``reasoning_visible`` is False, the provider's ``reasoning_tokens``
    are subtracted so the token-rate ratio is not inflated by tokens the user
    cannot see.  When the reasoning text was streamed/persisted and already
    counted in ``visible_units``, pass ``reasoning_visible=True`` so the
    denominator matches the numerator.
    """

    if not isinstance(usage, dict):
        return 0
    breakdown = billing_token_breakdown(usage)
    if reasoning_visible:
        return max(0, breakdown.output_tokens)
    return max(0, breakdown.output_tokens - breakdown.reasoning_tokens)


def model_response_visible_units(
    output: list[dict[str, Any]] | None,
    *,
    reasoning_text: str = "",
) -> int:
    """Count token-rate visible units available from a model response.

    This intentionally ignores hidden reasoning token counts that providers may
    include in usage.  It counts the text uv-agent can actually observe: final
    message text/refusals, streamed/persisted reasoning text, and tool-call names
    or arguments.
    """

    return sum(visible_units(part) for part in _model_response_visible_text_parts(output, reasoning_text=reasoning_text))


def model_response_visible_chars(
    output: list[dict[str, Any]] | None,
    *,
    reasoning_text: str = "",
) -> int:
    """Count raw visible characters from a model response.

    The displayed token-rate estimate uses :func:`model_response_visible_units`;
    this helper is retained for callers that still need raw character volume.
    """

    return sum(len(part) for part in _model_response_visible_text_parts(output, reasoning_text=reasoning_text))


def _model_response_visible_text_parts(
    output: list[dict[str, Any]] | None,
    *,
    reasoning_text: str = "",
) -> list[str]:
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
    return parts


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
