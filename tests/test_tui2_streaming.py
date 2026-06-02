from __future__ import annotations

from uv_agent.tui2.streaming import (
    StreamRateEstimator,
    ThreadTokenRatio,
    model_response_visible_chars,
    tool_call_name,
    tool_delta_visible_text,
)


def test_stream_rate_estimator_uses_three_second_sliding_window() -> None:
    estimator = StreamRateEstimator(window_s=3.0)

    estimator.observe("a" * 10, now=0.0)
    estimator.observe("b" * 30, now=2.0)
    estimator.observe("c" * 30, now=4.0)

    assert estimator.current_cps(now=4.0) == 20.0


def test_thread_token_ratio_converts_char_rate_after_usage() -> None:
    ratio = ThreadTokenRatio()

    assert ratio.token_rate(80.0) is None

    ratio.observe_response(visible_chars=40, output_tokens=10)

    assert ratio.token_rate(80.0) == 20.0


def test_model_response_visible_chars_counts_text_reasoning_and_tool_calls() -> None:
    output = [
        {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
        {"type": "function_call", "name": "run_python", "arguments": '{"code":"print(1)"}'},
    ]

    assert model_response_visible_chars(output, reasoning_text="think") == len(
        "think" + "answer" + "run_python" + '{"code":"print(1)"}'
    )


def test_tool_delta_visible_text_splits_name_and_argument_delta() -> None:
    tool_call = {"name": "run_python", "arguments_delta": '{"code":'}

    assert tool_call_name(tool_call) == "run_python"
    assert tool_delta_visible_text(tool_call) == '{"code":'
