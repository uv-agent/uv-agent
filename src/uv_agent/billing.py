from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from uv_agent.config import AppConfig, ModelConfig, ModelPricingConfig


BILLING_UNIT_DIVISORS: dict[str, Decimal] = {
    "token": Decimal(1),
    "tokens": Decimal(1),
    "per_token": Decimal(1),
    "1k_tokens": Decimal(1_000),
    "k_tokens": Decimal(1_000),
    "thousand_tokens": Decimal(1_000),
    "per_1k_tokens": Decimal(1_000),
    "1m_tokens": Decimal(1_000_000),
    "m_tokens": Decimal(1_000_000),
    "million_tokens": Decimal(1_000_000),
    "per_1m_tokens": Decimal(1_000_000),
    "per_million_tokens": Decimal(1_000_000),
}

CURRENCY_SYMBOLS = {
    "USD": "$",
    "$": "$",
    "CNY": "¥",
    "RMB": "¥",
    "CNH": "¥",
    "¥": "¥",
    "￥": "¥",
}


@dataclass(frozen=True)
class BillingTokenBreakdown:
    """Provider usage normalized into billable token buckets.

    ``output_tokens`` is the provider-reported total including reasoning tokens,
    while ``reasoning_tokens`` captures the portion that came from hidden
    reasoning / thinking (not visible in the model output text).  Callers that
    need a visible‑output token estimate can compute
    ``output_tokens - reasoning_tokens``.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class BillingCharge:
    """A computed incremental billing charge for one model call."""

    amount: Decimal
    currency: str
    model_name: str
    remote_model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    input_rate: Decimal
    cached_input_rate: Decimal
    output_rate: Decimal
    unit: str

    def to_event_payload(self, *, source: str, turn_id: str | None = None) -> dict[str, Any]:
        """Return JSON-friendly event fields for ThreadStore persistence."""

        payload: dict[str, Any] = {
            "source": source,
            "amount": decimal_to_string(self.amount),
            "currency": self.currency,
            "model": self.model_name,
            "remote_model": self.remote_model,
            "unit": self.unit,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "input_rate": decimal_to_string(self.input_rate),
            "cached_input_rate": decimal_to_string(self.cached_input_rate),
            "output_rate": decimal_to_string(self.output_rate),
        }
        if turn_id:
            payload["turn_id"] = turn_id
        return payload


def pricing_for_model(
    config: AppConfig,
    model: ModelConfig,
    *,
    level: str | None = None,
) -> ModelPricingConfig | None:
    """Return the configured price entry for a resolved model.

    Users usually key pricing by uv-agent's local model name, but supporting the
    remote provider model id (and finally the level name) makes hand-written
    configs more forgiving without changing the public model/level semantics.
    """

    if not config.pricing.models:
        return None
    keys = [model.name, model.model]
    if level:
        keys.append(level)
    for key in keys:
        price = config.pricing.models.get(key)
        if price is not None:
            return price
    return None


def billing_charge_for_usage(
    config: AppConfig,
    model: ModelConfig,
    usage: dict[str, Any],
    *,
    level: str | None = None,
) -> BillingCharge | None:
    """Compute the incremental charge for one provider usage payload.

    Returns ``None`` when pricing is not configured for the model or the provider
    returned no billable token counts. Prices are interpreted per the configured
    unit, which defaults to the vendor-standard one million tokens.
    """

    price = pricing_for_model(config, model, level=level)
    if price is None:
        return None
    breakdown = billing_token_breakdown(usage)
    if breakdown.input_tokens <= 0 and breakdown.cached_input_tokens <= 0 and breakdown.output_tokens <= 0:
        return None
    unit = price.unit or config.pricing.unit
    divisor = unit_divisor(unit)
    input_rate = decimal_or_zero(price.input)
    cached_rate = decimal_or_zero(price.cached_input)
    output_rate = decimal_or_zero(price.output)
    amount = (
        Decimal(breakdown.input_tokens) * input_rate
        + Decimal(breakdown.cached_input_tokens) * cached_rate
        + Decimal(breakdown.output_tokens) * output_rate
    ) / divisor
    return BillingCharge(
        amount=amount,
        currency=config.pricing.currency,
        model_name=model.name,
        remote_model=model.model,
        input_tokens=breakdown.input_tokens,
        cached_input_tokens=breakdown.cached_input_tokens,
        output_tokens=breakdown.output_tokens,
        input_rate=input_rate,
        cached_input_rate=cached_rate,
        output_rate=output_rate,
        unit=unit,
    )


def billing_token_breakdown(usage: dict[str, Any]) -> BillingTokenBreakdown:
    """Normalize common OpenAI-compatible and Anthropic usage shapes.

    OpenAI-style ``prompt_tokens``/``input_tokens`` usually include cached input
    tokens and expose the cached portion under ``*_tokens_details.cached_tokens``.
    Anthropic exposes cache reads separately, so those are billed as cached input
    while ``input_tokens`` and cache-creation tokens remain normal input.
    """

    output_tokens = first_int(usage, "output_tokens", "completion_tokens") or 0

    detail_cached = first_nested_int(
        usage,
        ("input_tokens_details", "cached_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
    )
    direct_cached = first_int(
        usage,
        "cached_input_tokens",
        "cache_read_input_tokens",
        "prompt_cache_hit_tokens",
    )
    cached_tokens = max(0, detail_cached if detail_cached is not None else direct_cached or 0)

    explicit_miss = first_int(usage, "prompt_cache_miss_tokens")
    cache_creation = first_int(usage, "cache_creation_input_tokens") or 0
    input_total = first_int(usage, "input_tokens", "prompt_tokens")

    if explicit_miss is not None:
        input_tokens = max(0, explicit_miss + cache_creation)
    elif "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        # Anthropic usage keeps cache read/create counts outside ordinary input.
        input_tokens = max(0, (input_total or 0) + cache_creation)
    elif input_total is not None:
        input_tokens = max(0, input_total - cached_tokens)
    else:
        input_tokens = max(0, cache_creation)

    # Extract hidden reasoning/thinking token counts that are included in
    # output_tokens but not visible in the model response text.
    reasoning = (
        first_nested_int(
            usage,
            ("output_tokens_details", "reasoning_tokens"),
            ("completion_tokens_details", "reasoning_tokens"),
        )
        or first_int(usage, "reasoning_tokens")
        or 0
    )

    return BillingTokenBreakdown(
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=max(0, output_tokens),
        reasoning_tokens=max(0, reasoning),
    )


def billing_enabled(config: AppConfig) -> bool:
    """Return whether any model pricing is configured."""

    return bool(config.pricing.models)


def billing_total_from_metadata(
    metadata: dict[str, Any],
    *,
    preferred_currency: str | None = None,
) -> tuple[Decimal, str] | None:
    """Extract a persisted thread total, preferring the active config currency.

    Metadata keeps a per-currency map so a later currency change does not destroy
    the old total. UI surfaces still show a single amount, so they prefer the
    current pricing currency when available and otherwise fall back to the legacy
    single ``billing_total`` fields.
    """

    totals = metadata.get("billing_totals")
    if isinstance(totals, dict):
        currency_keys: list[str] = []
        if preferred_currency:
            currency_keys.append(normalize_currency(preferred_currency))
        currency_keys.extend(str(key) for key in totals.keys())
        seen: set[str] = set()
        for currency in currency_keys:
            normalized = normalize_currency(currency)
            if normalized in seen:
                continue
            seen.add(normalized)
            if normalized not in totals:
                continue
            total = decimal_or_none(totals.get(normalized))
            if total is not None:
                return total, normalized
    total = decimal_or_none(metadata.get("billing_total"))
    currency = str(metadata.get("billing_currency") or preferred_currency or "").strip()
    if total is None or not currency:
        return None
    return total, normalize_currency(currency)


def format_billing_total(
    amount: Decimal | float | int | str,
    currency: str,
    *,
    decimals: int,
) -> str:
    """Format a thread billing total as ``$0.0000`` / ``¥0.0000``."""

    decimal_amount = decimal_or_zero(amount)
    quant = Decimal(1).scaleb(-max(0, decimals))
    rounded = decimal_amount.quantize(quant, rounding=ROUND_HALF_UP)
    return f"{currency_symbol(currency)}{rounded:.{max(0, decimals)}f}"


def currency_symbol(currency: str) -> str:
    """Return the short display symbol for configured billing currency."""

    normalized = normalize_currency(currency)
    return CURRENCY_SYMBOLS.get(normalized, f"{normalized} ")


def normalize_currency(currency: str) -> str:
    """Normalize user-entered currency names without rejecting custom values."""

    value = str(currency or "").strip()
    if value in {"¥", "￥", "$"}:
        return value
    return value.upper() or "USD"


def unit_divisor(unit: str) -> Decimal:
    """Return the token divisor for a pricing unit string."""

    key = str(unit or "1M_tokens").strip().lower().replace("-", "_").replace(" ", "_")
    return BILLING_UNIT_DIVISORS.get(key, Decimal(1_000_000))


def first_int(value: dict[str, Any], *keys: str) -> int | None:
    """Return the first integer-like value for any top-level key."""

    for key in keys:
        parsed = int_or_none(value.get(key))
        if parsed is not None:
            return parsed
    return None


def first_nested_int(value: dict[str, Any], *paths: tuple[str, str]) -> int | None:
    """Return the first integer-like value for a two-level usage detail path."""

    for outer, inner in paths:
        nested = value.get(outer)
        if not isinstance(nested, dict):
            continue
        parsed = int_or_none(nested.get(inner))
        if parsed is not None:
            return parsed
    return None


def int_or_none(value: Any) -> int | None:
    """Parse provider token counts while rejecting booleans and negatives."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def decimal_or_zero(value: Any) -> Decimal:
    """Parse JSON numeric/string money fields to Decimal, defaulting to zero."""

    parsed = decimal_or_none(value)
    return parsed if parsed is not None else Decimal(0)


def decimal_or_none(value: Any) -> Decimal | None:
    """Parse a JSON value to Decimal without accepting booleans or empties."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite():
        return None
    return parsed


def decimal_to_string(value: Decimal) -> str:
    """Serialize Decimal values compactly for JSON metadata."""

    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return format(normalized, "f")
    return format(normalized, "f")
