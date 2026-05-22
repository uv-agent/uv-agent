from __future__ import annotations

import json
from pathlib import Path

from uv_agent.config import config_paths, editable_config_path, load_config, redact_config
from uv_agent.paths import project_config_path, project_state_dir, user_config_path


def test_load_config_merges_project_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "p": {
                        "base_url": "https://example.com",
                        "api_key": "secret",
                        "responses": {"path": "/v1/responses"},
                        "chat_completions": {"path": "/v1/chat/completions"},
                        "message_passthrough": {
                            "assistant": ["reasoning_content"],
                            "user": ["vendor_user"],
                        },
                        "reasoning_display": {
                            "assistant_message_fields": ["provider_reasoning"],
                            "stream_delta_fields": ["reasoning_content"],
                            "unknown_text_delta_as_reasoning": True,
                        },
                    }
                },
                "models": {
                    "m": {
                        "provider": "p",
                        "model": "remote",
                        "api": "chat_completions",
                        "supports_images": False,
                        "params": {"reasoning": {"effort": "low"}},
                        "message_passthrough": {"assistant": ["model_reasoning"]},
                        "reasoning_display": {
                            "assistant_message_fields": ["model_reasoning"],
                        },
                    }
                },
                "pricing": {
                    "currency": "CNY",
                    "unit": "1M_tokens",
                    "models": {
                        "m": {
                            "input": 2.0,
                            "output": 8.0,
                            "cached_input": 0.5,
                        }
                    },
                },
                "levels": {
                    "medium": {
                        "model": "m",
                        "params": {"reasoning": {"effort": "high"}},
                    }
                },
                "runtime": {
                    "title_generation": {"model_level": "quick"},
                    "stream_retry": {
                        "max_retries": 3,
                        "base": 0.5,
                        "factor": 3,
                        "max": 10,
                        "jitter": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(tmp_path, [config_path])

    model = config.model_for_level("medium")
    provider = config.provider_for_model(model)
    assert provider.name == "p"
    assert provider.resolved_api_key() == "secret"
    assert provider.endpoint_for_api("chat_completions").path == "/v1/chat/completions"
    assert model.api == "chat_completions"
    assert model.supports_images is False
    assert model.params["reasoning"]["effort"] == "high"
    assert model.message_passthrough.assistant == ["model_reasoning"]
    assert model.message_passthrough.user == ["vendor_user"]
    assert provider.message_passthrough.assistant == ["reasoning_content"]
    assert model.reasoning_display.assistant_message_fields == ["model_reasoning"]
    assert model.reasoning_display.stream_delta_fields == ["reasoning_content"]
    assert model.reasoning_display.unknown_text_delta_as_reasoning is True
    assert config.runtime.title_generation.enabled is True
    assert config.runtime.title_generation.model_level == "quick"
    assert config.runtime.compression.enabled is True
    assert config.runtime.stream_retry.max_retries == 3
    assert config.runtime.stream_retry.base == 0.5
    assert config.runtime.stream_retry.factor == 3
    assert config.runtime.stream_retry.max == 10
    assert config.runtime.stream_retry.jitter == 0
    assert config.pricing.currency == "CNY"
    assert config.pricing.unit == "1M_tokens"
    assert config.pricing.models["m"].input == 2.0
    assert config.pricing.models["m"].output == 8.0
    assert config.pricing.models["m"].cached_input == 0.5
    assert config.runner.default_timeout_s == 7200
    assert config.runner.max_run_logs == 200


def test_runner_max_run_logs_can_be_configured(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "runner": {
                    "max_run_logs": 12,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(tmp_path, [config_path])

    assert config.runner.max_run_logs == 12


def test_legacy_reasoning_option_fields_are_ignored(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "p": {
                        "base_url": "https://example.com",
                        "reasoning_options": [
                            {"name": "low", "params": {"reasoning": {"effort": "low"}}},
                        ],
                    }
                },
                "models": {"m": {"provider": "p", "model": "remote"}},
                "levels": {"medium": {"model": "m", "reasoning": "low", "params": {"temperature": 0}}},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(tmp_path, [config_path])

    assert config.model_for_level("medium").params == {"temperature": 0}


def test_model_inherits_provider_passthrough_and_reasoning_display(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "p": {
                        "base_url": "https://example.com",
                        "message_passthrough": {"assistant": ["reasoning_content"]},
                        "reasoning_display": {
                            "stream_delta_fields": ["reasoning_content"],
                            "unknown_text_delta_as_reasoning": True,
                        },
                    }
                },
                "models": {"m": {"provider": "p", "model": "remote"}},
                "levels": {"medium": {"model": "m"}},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(tmp_path, [config_path])
    model = config.model_for_level("medium")

    assert model.message_passthrough.assistant == ["reasoning_content"]
    assert model.reasoning_display.stream_delta_fields == ["reasoning_content"]
    assert model.reasoning_display.unknown_text_delta_as_reasoning is True


def test_configured_levels_replace_default_level_template(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {"p": {"base_url": "https://example.com"}},
                "models": {"m": {"provider": "p", "model": "remote"}},
                "levels": {
                    "small": {"model": "m"},
                    "medium": {"model": "m"},
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(tmp_path, [config_path])

    assert list(config.levels) == ["small", "medium"]


def test_default_title_and_compression_levels_do_not_assume_small(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {"p": {"base_url": "https://example.com"}},
                "models": {"m": {"provider": "p", "model": "remote"}},
                "levels": {
                    "fast": {"model": "m"},
                    "deep": {"model": "m"},
                },
                "runtime": {"default_level": "fast"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(tmp_path, [config_path])

    assert list(config.levels) == ["fast", "deep"]
    assert config.runtime.default_level == "fast"
    assert config.runtime.title_generation.model_level is None
    assert config.runtime.compression.model_level is None
    assert config.runtime.compression.enabled is True


def test_compression_and_title_prompt_are_not_config_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "runtime": {
                    "compression": {"enabled": False, "prompt": "custom", "target_ratio": 0.1},
                    "title_generation": {"prompt": "custom"},
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        load_config(tmp_path, [config_path])
    except TypeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected prompt/target_ratio config to be rejected")

    assert "prompt" in message or "target_ratio" in message


def test_redact_config_masks_sensitive_values() -> None:
    redacted = redact_config({"api_key": "secret", "nested": {"token": "x", "ok": 1}})
    assert redacted["api_key"] == "***REDACTED***"
    assert redacted["nested"]["token"] == "***REDACTED***"
    assert redacted["nested"]["ok"] == 1


def test_default_paths_are_user_level(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    project_root = tmp_path / "workspace"
    project_root.mkdir()

    paths = config_paths(project_root)

    assert paths[0] == user_config_path()
    assert paths[1] == project_config_path(project_root)
    assert project_state_dir(project_root).is_relative_to(tmp_path / "home" / "projects")


def test_project_state_dir_ignores_runtime_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UV_AGENT_PROJECT_STATE_DIR", str(tmp_path / "old-runtime-state"))
    project_root = tmp_path / "workspace"
    project_root.mkdir()

    assert project_state_dir(project_root).is_relative_to(tmp_path / "home" / "projects")
    assert project_state_dir(project_root) != (tmp_path / "old-runtime-state").resolve()


def test_ui_language_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"ui": {"language": "zh-CN"}}), encoding="utf-8")

    config = load_config(tmp_path, [config_path])

    assert config.ui.language == "zh-CN"


def test_ui_completion_notification_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "ui": {
                    "completion_notification": {
                        "enabled": True,
                        "terminal": False,
                        "bell": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(tmp_path, [config_path])

    assert config.ui.completion_notification.enabled is True
    assert config.ui.completion_notification.terminal is False
    assert config.ui.completion_notification.bell is True


def test_editable_config_prefers_existing_user_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    project_root = tmp_path / "workspace"
    project_root.mkdir()
    user_config_path().parent.mkdir(parents=True)
    user_config_path().write_text("{}", encoding="utf-8")

    assert editable_config_path(project_root) == user_config_path()


def test_editable_config_falls_back_to_project_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UV_AGENT_HOME", str(tmp_path / "home"))
    project_root = tmp_path / "workspace"
    project_root.mkdir()

    assert editable_config_path(project_root) == project_config_path(project_root)
