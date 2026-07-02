from __future__ import annotations

from uv_agent.tui.streaming import (
    StreamRateEstimator,
    ThreadTokenRatio,
    model_response_visible_units,
    tool_call_name,
    tool_delta_visible_text,
    usage_output_tokens,
    visible_units,
)


def test_stream_rate_estimator_uses_three_second_sliding_window() -> None:
    estimator = StreamRateEstimator(window_s=3.0)

    estimator.observe("a" * 10, now=0.0)
    estimator.observe("b" * 30, now=2.0)
    estimator.observe("c" * 30, now=4.0)

    assert estimator.current_cps(now=4.0) == 20.0


def test_stream_rate_estimator_tracks_visible_unit_rate_across_chunks() -> None:
    estimator = StreamRateEstimator(window_s=3.0)

    estimator.observe("hel", now=0.0)
    estimator.observe("lo, 世界", now=1.0)

    assert estimator.total_units == 4
    assert estimator.current_ups(now=1.0) == 4.0


def test_thread_token_ratio_converts_visible_unit_rate_after_usage() -> None:
    ratio = ThreadTokenRatio()

    assert ratio.token_rate(80.0) is None

    ratio.observe_response(visible_units=40, output_tokens=10)

    assert ratio.token_rate(80.0) == 20.0


def test_visible_units_count_words_cjk_and_punctuation() -> None:
    assert visible_units('hello, 世界! {"a": 1}') == 12


def test_usage_output_tokens_subtracts_hidden_reasoning_only() -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 100,
        "output_tokens_details": {"reasoning_tokens": 40},
    }

    # Hidden CoT (no visible reasoning text): subtract reasoning tokens so the
    # denominator matches the visible-units numerator.
    assert usage_output_tokens(usage) == 60
    assert usage_output_tokens(usage, reasoning_visible=False) == 60

    # Visible thinking (reasoning text already counted in visible_units):
    # keep the provider total so we do not under-count tok/s.
    assert usage_output_tokens(usage, reasoning_visible=True) == 100


def test_model_response_visible_units_counts_text_reasoning_and_tool_calls() -> None:
    output = [
        {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
        {"type": "function_call", "name": "run_python", "arguments": '{"code":"print(1)"}'},
    ]

    assert model_response_visible_units(output, reasoning_text="think") == 15


def test_tool_delta_visible_text_splits_name_and_argument_delta() -> None:
    tool_call = {"name": "run_python", "arguments_delta": '{"code":'}

    assert tool_call_name(tool_call) == "run_python"
    assert tool_delta_visible_text(tool_call) == '{"code":'


def test_thread_token_ratio_metadata_round_trip() -> None:
    ratio = ThreadTokenRatio()
    ratio.observe_response(visible_units=40, output_tokens=10)

    metadata = ratio.to_metadata()
    restored = ThreadTokenRatio.from_metadata(metadata)

    assert restored.visible_units == 40
    assert restored.output_tokens == 10
    assert restored.token_rate(80.0) == 20.0
