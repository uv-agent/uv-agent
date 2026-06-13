from __future__ import annotations

import atexit
import functools
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, TypeVar, cast

T = TypeVar("T", bound=Callable[..., Any])

_thread_local = threading.local()
_summary_lock = threading.RLock()
_helper_summaries: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_flush_registered = False
_flushed = False


def tracked_helper(func: T, *, name: str | None = None) -> T:
    """Decorate a public runtime helper so host/UI metadata sees real calls.

    The wrapper records only metadata that is safe to persist or display: helper
    name, counts, keyword names, argument *types*, duration, and outcome. It does
    not store argument values, command strings, environment mappings, file
    contents, prompts, or paths passed as arguments.
    """

    helper_name = name or getattr(func, "__name__", "helper")
    _ensure_flush_registered()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started_unix = time.time()
        started_perf = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            record_helper_call(
                helper_name,
                args,
                kwargs,
                called_at_unix=started_unix,
                duration_ms=_elapsed_ms(started_perf),
                outcome="error",
                error_type=exc.__class__.__name__,
            )
            raise
        record_helper_call(
            helper_name,
            args,
            kwargs,
            called_at_unix=started_unix,
            duration_ms=_elapsed_ms(started_perf),
            outcome="ok",
            error_type=None,
        )
        return result

    return cast(T, wrapper)


def record_helper_call(
    name: str,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    called_at_unix: float | None = None,
    duration_ms: float | None = None,
    outcome: str = "ok",
    error_type: str | None = None,
) -> None:
    """Publish one helper-call observation to all local subscribers.

    Host-facing data is summarized in-process and flushed once at interpreter
    exit, so loops do not pay an HTTP round-trip per helper call. The historical
    helper_stats SQLite sink remains independent and receives the same sanitized
    payload without becoming the source queried by the host.
    """

    if not name or getattr(_thread_local, "recording", False):
        return
    _thread_local.recording = True
    try:
        payload = helper_call_payload(
            name,
            args,
            kwargs or {},
            called_at_unix=called_at_unix,
            duration_ms=duration_ms,
            outcome=outcome,
            error_type=error_type,
        )
        _record_for_host(payload)
        _record_for_stats(payload)
    except Exception:
        # Helper tracking must never change helper behavior.
        return
    finally:
        _thread_local.recording = False


def helper_call_payload(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    called_at_unix: float | None,
    duration_ms: float | None,
    outcome: str,
    error_type: str | None,
) -> dict[str, Any]:
    """Return the sanitized metadata shape shared by tracking subscribers."""

    return {
        "helper": str(name),
        "called_at_unix": time.time() if called_at_unix is None else called_at_unix,
        "run_id": os.environ.get("UV_AGENT_RUNTIME_RUN_ID"),
        "thread_id": os.environ.get("UV_AGENT_RUNTIME_THREAD_ID"),
        "turn_id": os.environ.get("UV_AGENT_RUNTIME_TURN_ID"),
        "cwd": str(Path.cwd()),
        "pid": os.getpid(),
        "positional_count": len(args),
        "keyword_names": sorted(str(key) for key in kwargs),
        "argument_types": _argument_types(args, kwargs),
        "duration_ms": duration_ms,
        "outcome": str(outcome or "ok"),
        "error_type": error_type,
    }


def helper_call_summaries(*, reset: bool = False) -> list[dict[str, Any]]:
    """Return host-facing helper-call summaries accumulated in this process."""

    with _summary_lock:
        summaries = [_public_summary(entry) for entry in _helper_summaries.values()]
        if reset:
            _helper_summaries.clear()
        return summaries


def flush_host_helper_calls() -> None:
    """Best-effort one-shot flush of runtime helper summaries to the host."""

    global _flushed
    with _summary_lock:
        if _flushed:
            return
        _flushed = True
        summaries = [_public_summary(entry) for entry in _helper_summaries.values()]
    if not summaries:
        return
    try:
        from .transport import emit_helper_calls_rpc

        emit_helper_calls_rpc(summaries)
    except Exception:
        return


def _record_for_host(payload: dict[str, Any]) -> None:
    _ensure_flush_registered()
    helper = str(payload.get("helper") or "").strip()
    if not helper:
        return
    with _summary_lock:
        entry = _helper_summaries.get(helper)
        if entry is None:
            entry = {
                "helper": helper,
                "run_id": payload.get("run_id"),
                "thread_id": payload.get("thread_id"),
                "turn_id": payload.get("turn_id"),
                "count": 0,
                "outcomes": {},
                "total_duration_ms": 0.0,
                "first_called_at_unix": payload.get("called_at_unix"),
                "last_called_at_unix": payload.get("called_at_unix"),
                "positional_counts": [],
                "keyword_names": [],
                "argument_types": payload.get("argument_types") if isinstance(payload.get("argument_types"), dict) else {},
                "error_types": [],
            }
            _helper_summaries[helper] = entry
        entry["count"] = int(entry.get("count") or 0) + 1
        outcome = str(payload.get("outcome") or "ok")
        outcomes = entry.setdefault("outcomes", {})
        if isinstance(outcomes, dict):
            outcomes[outcome] = int(outcomes.get(outcome) or 0) + 1
        duration_ms = _float_or_none(payload.get("duration_ms"))
        if duration_ms is not None:
            entry["total_duration_ms"] = float(entry.get("total_duration_ms") or 0.0) + duration_ms
        called_at = _float_or_none(payload.get("called_at_unix"))
        if called_at is not None:
            if entry.get("first_called_at_unix") is None:
                entry["first_called_at_unix"] = called_at
            entry["last_called_at_unix"] = called_at
        positional_count = _int_or_none(payload.get("positional_count"))
        if positional_count is not None:
            _append_unique(entry.setdefault("positional_counts", []), positional_count, max_items=16)
        for keyword in payload.get("keyword_names") if isinstance(payload.get("keyword_names"), list) else []:
            _append_unique(entry.setdefault("keyword_names", []), str(keyword), max_items=64)
        error_type = payload.get("error_type")
        if error_type:
            _append_unique(entry.setdefault("error_types", []), str(error_type), max_items=32)


def _record_for_stats(payload: dict[str, Any]) -> None:
    try:
        helper_stats = import_module(".helper_stats", __package__ or "uv_agent_runtime")
        recorder = getattr(helper_stats, "record_helper_call_payload", None)
        if callable(recorder):
            recorder(payload)
    except Exception:
        return


def _ensure_flush_registered() -> None:
    global _flush_registered
    if _flush_registered:
        return
    with _summary_lock:
        if _flush_registered:
            return
        atexit.register(flush_host_helper_calls)
        _flush_registered = True


def _public_summary(entry: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "helper": str(entry.get("helper") or "helper"),
        "count": max(0, int(entry.get("count") or 0)),
    }
    for key in ("run_id", "thread_id", "turn_id"):
        value = entry.get(key)
        if value:
            result[key] = value
    outcomes = entry.get("outcomes")
    if isinstance(outcomes, dict) and outcomes:
        result["outcomes"] = {str(key): int(value or 0) for key, value in outcomes.items()}
    duration_ms = _float_or_none(entry.get("total_duration_ms"))
    if duration_ms is not None:
        result["total_duration_ms"] = round(duration_ms, 3)
    for key in ("first_called_at_unix", "last_called_at_unix"):
        value = _float_or_none(entry.get(key))
        if value is not None:
            result[key] = value
    positional_counts = entry.get("positional_counts")
    if isinstance(positional_counts, list) and positional_counts:
        result["positional_counts"] = sorted({int(value) for value in positional_counts if isinstance(value, int)})
    keyword_names = entry.get("keyword_names")
    if isinstance(keyword_names, list) and keyword_names:
        result["keyword_names"] = sorted({str(value) for value in keyword_names})
    argument_types = entry.get("argument_types")
    if isinstance(argument_types, dict) and argument_types:
        result["argument_types"] = argument_types
    error_types = entry.get("error_types")
    if isinstance(error_types, list) and error_types:
        result["error_types"] = sorted({str(value) for value in error_types})
    return result


def _elapsed_ms(started_perf: float) -> float:
    return max(0.0, (time.perf_counter() - started_perf) * 1000.0)


def _argument_types(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "args": [_type_name(value) for value in args],
        "kwargs": {str(key): _type_name(value) for key, value in kwargs.items()},
    }


def _type_name(value: Any) -> str:
    if isinstance(value, Path):
        return "Path"
    return type(value).__name__


def _append_unique(target: list[Any], value: Any, *, max_items: int) -> None:
    if value in target or len(target) >= max_items:
        return
    target.append(value)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
