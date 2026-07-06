from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .api import PluginStatus

_MAIN_STATES = ("started", "warning", "failed", "skipped", "disabled")
_DETAIL_LIMIT = 6
_MESSAGE_LIMIT = 160


def format_plugin_status_counts(records: Iterable[PluginStatus]) -> str:
    """Return a compact operational summary for plugin startup/status output."""

    items = tuple(records)
    counts = Counter(str(record.state or "unknown") for record in items)
    parts = [f"total={len(items)}"]
    parts.extend(f"{state}={counts.get(state, 0)}" for state in _MAIN_STATES)
    parts.extend(
        f"{state}={counts[state]}"
        for state in sorted(counts)
        if state not in _MAIN_STATES and counts[state]
    )
    return "plugins: " + " ".join(parts)


def format_plugin_detail_lines(
    records: Iterable[PluginStatus],
    *,
    include_started_external: bool = False,
    include_first_load_external: bool = False,
    limit: int = _DETAIL_LIMIT,
) -> list[str]:
    """Return detail lines for notable plugin states.

    Builtin plugins are summarized by count unless they failed, warn, or were
    skipped. Third-party plugins are named because they are usually why the host
    was launched with plugin-related flags.
    """

    items = tuple(records)
    lines: list[str] = []
    if include_started_external:
        started = [record.id for record in items if record.state == "started" and not record.builtin]
        if started:
            lines.append(_format_names("plugin started", started, limit=limit))
    if include_first_load_external:
        first_load = [record.id for record in items if record.first_load and not record.builtin]
        if first_load:
            lines.append(_format_names("plugin first load", first_load, limit=limit))
    failed = [record for record in items if record.state == "failed"]
    if failed:
        lines.append(_format_records("plugin failed", failed, limit=limit))
    warnings = [record for record in items if record.state == "warning"]
    if warnings:
        lines.append(_format_records("plugin warning", warnings, limit=limit))
    skipped = [record for record in items if record.state == "skipped"]
    if skipped:
        lines.append(_format_records("plugin skipped", skipped, limit=limit))
    disabled_external = [record.id for record in items if record.state == "disabled" and not record.builtin]
    if disabled_external:
        lines.append(_format_names("plugin disabled", disabled_external, limit=limit))
    return lines


def _format_names(prefix: str, names: list[str], *, limit: int) -> str:
    shown = names[: max(1, limit)]
    suffix = f" (+{len(names) - len(shown)} more)" if len(shown) < len(names) else ""
    return f"{prefix}: {', '.join(shown)}{suffix}"


def _format_records(prefix: str, records: list[PluginStatus], *, limit: int) -> str:
    shown = records[: max(1, limit)]
    suffix = f" (+{len(records) - len(shown)} more)" if len(shown) < len(records) else ""
    return f"{prefix}: {'; '.join(_record_detail(record) for record in shown)}{suffix}"


def _record_detail(record: PluginStatus) -> str:
    message = _truncate(str(record.message or "").strip())
    error_type = str(record.error_type or "").strip()
    if error_type and message:
        return f"{record.id} ({error_type}: {message})"
    if error_type:
        return f"{record.id} ({error_type})"
    if message:
        return f"{record.id} ({message})"
    return record.id


def _truncate(message: str) -> str:
    if len(message) <= _MESSAGE_LIMIT:
        return message
    return message[: _MESSAGE_LIMIT - 3].rstrip() + "..."
