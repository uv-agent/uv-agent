from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from uv_agent.paths import project_config_path, project_local_dir, user_config_path


SENSITIVE_KEYS = {"api_key", "authorization", "token", "secret", "password"}


class ConfigError(ValueError):
    """Raised when a configuration file is missing required data."""


@dataclass(frozen=True)
class EndpointConfig:
    path: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    responses: EndpointConfig = field(default_factory=lambda: EndpointConfig(path="/responses"))
    chat_completions: EndpointConfig = field(
        default_factory=lambda: EndpointConfig(path="/chat/completions")
    )
    anthropic_messages: EndpointConfig = field(
        default_factory=lambda: EndpointConfig(path="/v1/messages")
    )

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    def endpoint_for_api(self, api: str) -> EndpointConfig:
        if api == "responses":
            return self.responses
        if api == "chat_completions":
            return self.chat_completions
        if api == "anthropic_messages":
            return self.anthropic_messages
        raise ConfigError(f"Unsupported provider API: {api}")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model: str
    api: str = "responses"
    context_window_tokens: int = 128_000
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LevelConfig:
    name: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompressionConfig:
    model_level: str = "small"
    prompt: str = (
        "Summarize the conversation context for future continuation. Keep user intent, "
        "decisions, file changes, tool results, and unresolved tasks. Be concise but complete."
    )
    trigger_ratio: float = 0.7


@dataclass(frozen=True)
class RuntimeConfig:
    default_level: str = "medium"
    auto_compress: bool = True
    store_provider_response: bool = False
    max_agent_rounds: int = 100
    compression: CompressionConfig = field(default_factory=CompressionConfig)


@dataclass(frozen=True)
class RunnerConfig:
    runtime_dependency: str
    runtime_package_name: str = "uv-agent"
    default_timeout_s: float = 60.0
    max_output_bytes: int = 1_000_000


@dataclass(frozen=True)
class AppConfig:
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    levels: dict[str, LevelConfig]
    runtime: RuntimeConfig
    runner: RunnerConfig

    def level(self, name: str | None = None) -> LevelConfig:
        level_name = name or self.runtime.default_level
        try:
            return self.levels[level_name]
        except KeyError as exc:
            raise ConfigError(f"Unknown level: {level_name}") from exc

    def model_for_level(self, name: str | None = None) -> ModelConfig:
        level = self.level(name)
        try:
            model = self.models[level.model]
        except KeyError as exc:
            raise ConfigError(f"Unknown model for level {level.name}: {level.model}") from exc
        merged_params = deep_merge(model.params, level.params)
        return ModelConfig(
            name=model.name,
            provider=model.provider,
            model=model.model,
            api=model.api,
            context_window_tokens=model.context_window_tokens,
            params=merged_params,
        )

    def provider_for_model(self, model: ModelConfig) -> ProviderConfig:
        try:
            return self.providers[model.provider]
        except KeyError as exc:
            raise ConfigError(f"Unknown provider for model {model.name}: {model.provider}") from exc


def default_config(project_root: Path) -> dict[str, Any]:
    runtime_dependency = f"uv-agent @ {project_root.resolve().as_uri()}"
    return {
        "providers": {},
        "models": {},
        "levels": {
            "small": {"model": "default"},
            "medium": {"model": "default"},
            "large": {"model": "default"},
        },
        "runtime": {
            "default_level": "medium",
            "auto_compress": True,
            "store_provider_response": False,
            "max_agent_rounds": 100,
            "compression": {
                "model_level": "small",
                "prompt": CompressionConfig().prompt,
                "trigger_ratio": 0.7,
            },
        },
        "ui": {
            "stream": True,
        },
        "runner": {
            "runtime_dependency": runtime_dependency,
            "runtime_package_name": "uv-agent",
            "default_timeout_s": 60,
            "max_output_bytes": 1_000_000,
        },
    }


def config_paths(project_root: Path) -> list[Path]:
    paths: list[Path] = []
    env_config = os.environ.get("UV_AGENT_CONFIG")
    paths.append(user_config_path())
    paths.append(project_config_path(project_root))
    if env_config:
        paths.append(Path(env_config))
    return paths


def config_sources(project_root: Path) -> list[dict[str, Any]]:
    """Return config layers with existence metadata for UI/debug views."""
    return [
        {"scope": "user", "path": str(user_config_path()), "exists": user_config_path().exists()},
        {
            "scope": "project",
            "path": str(project_config_path(project_root)),
            "exists": project_config_path(project_root).exists(),
        },
        {
            "scope": "project_data",
            "path": str(project_local_dir(project_root)),
            "exists": project_local_dir(project_root).exists(),
        },
    ]


def load_config(project_root: Path | None = None, paths: list[Path] | None = None) -> AppConfig:
    root = (project_root or Path.cwd()).resolve()
    raw = load_raw_config(root, paths)
    return parse_config(raw, root)


def load_raw_config(project_root: Path | None = None, paths: list[Path] | None = None) -> dict[str, Any]:
    """Load merged config as raw dictionaries for redacted UI/debug display."""
    root = (project_root or Path.cwd()).resolve()
    raw = default_config(root)
    for path in paths if paths is not None else config_paths(root):
        if path.exists():
            raw = deep_merge(raw, json.loads(path.read_text(encoding="utf-8")))
    return raw


def parse_config(raw: dict[str, Any], project_root: Path) -> AppConfig:
    providers = {}
    for name, value in raw.get("providers", {}).items():
        provider_value = dict(value)
        provider_value.pop("name", None)
        legacy_endpoint = provider_value.pop("endpoint", None)
        legacy_api = provider_value.pop("api_format", None)
        responses_raw = provider_value.pop("responses", None)
        chat_raw = provider_value.pop("chat_completions", None)
        anthropic_raw = provider_value.pop("anthropic_messages", None)
        if responses_raw is None:
            responses_raw = {"path": legacy_endpoint or "/responses"}
        if chat_raw is None:
            chat_raw = {"path": "/chat/completions"}
        if anthropic_raw is None:
            anthropic_raw = {"path": "/v1/messages"}
        provider_value["responses"] = EndpointConfig(**responses_raw)
        provider_value["chat_completions"] = EndpointConfig(**chat_raw)
        provider_value["anthropic_messages"] = EndpointConfig(**anthropic_raw)
        if legacy_api and legacy_api != "responses":
            # Older experimental configs used provider-level api_format. Models now own API choice.
            provider_value.setdefault("params", {})
        providers[name] = ProviderConfig(name=name, **provider_value)
    models = {}
    for name, value in raw.get("models", {}).items():
        model_value = dict(value)
        model_value.setdefault("api", model_value.pop("api_format", "responses"))
        models[name] = ModelConfig(name=name, **model_value)
    levels = {
        name: LevelConfig(name=name, **value)
        for name, value in raw.get("levels", {}).items()
    }
    runtime_raw = raw.get("runtime", {})
    compression = CompressionConfig(**runtime_raw.get("compression", {}))
    runtime = RuntimeConfig(
        default_level=runtime_raw.get("default_level", "medium"),
        auto_compress=runtime_raw.get("auto_compress", True),
        store_provider_response=runtime_raw.get("store_provider_response", False),
        max_agent_rounds=runtime_raw.get("max_agent_rounds", 100),
        compression=compression,
    )
    runner_raw = raw.get("runner", {})
    runner = RunnerConfig(
        runtime_dependency=runner_raw.get(
            "runtime_dependency",
            f"uv-agent @ {project_root.resolve().as_uri()}",
        ),
        runtime_package_name=runner_raw.get("runtime_package_name", "uv-agent"),
        default_timeout_s=float(runner_raw.get("default_timeout_s", 60)),
        max_output_bytes=int(runner_raw.get("max_output_bytes", 1_000_000)),
    )
    return AppConfig(
        providers=providers,
        models=models,
        levels=levels,
        runtime=runtime,
        runner=runner,
    )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def redact_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***REDACTED***" if key.lower() in SENSITIVE_KEYS else redact_config(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_config(item) for item in value]
    return value
