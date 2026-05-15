from __future__ import annotations

import json
from pathlib import Path

from uv_agent.config import load_config, redact_config


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
                    }
                },
                "models": {
                    "m": {
                        "provider": "p",
                        "model": "remote",
                        "api": "chat_completions",
                        "params": {"reasoning": {"effort": "low"}},
                    }
                },
                "levels": {
                    "medium": {
                        "model": "m",
                        "params": {"reasoning": {"effort": "high"}},
                    }
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
    assert model.params["reasoning"]["effort"] == "high"
    assert "uv-agent @ file:///" in config.runner.runtime_dependency


def test_redact_config_masks_sensitive_values() -> None:
    redacted = redact_config({"api_key": "secret", "nested": {"token": "x", "ok": 1}})
    assert redacted["api_key"] == "***REDACTED***"
    assert redacted["nested"]["token"] == "***REDACTED***"
    assert redacted["nested"]["ok"] == 1
