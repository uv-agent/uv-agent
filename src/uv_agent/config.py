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
    message_passthrough: MessagePassthroughConfig = field(default_factory=lambda: MessagePassthroughConfig())
    reasoning_display: ReasoningDisplayConfig = field(default_factory=lambda: ReasoningDisplayConfig())
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
class MessagePassthroughConfig:
    assistant: list[str] = field(default_factory=list)
    user: list[str] = field(default_factory=list)
    system: list[str] = field(default_factory=list)
    tool: list[str] = field(default_factory=list)

    def fields_for_role(self, role: str) -> list[str]:
        return list(getattr(self, role, []))


@dataclass(frozen=True)
class ReasoningDisplayConfig:
    assistant_message_fields: list[str] = field(default_factory=list)
    stream_delta_fields: list[str] = field(default_factory=list)
    unknown_text_delta_as_reasoning: bool = False


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model: str
    api: str = "responses"
    context_window_tokens: int = 128_000
    supports_images: bool | None = None
    params: dict[str, Any] = field(default_factory=dict)
    message_passthrough: MessagePassthroughConfig = field(default_factory=MessagePassthroughConfig)
    reasoning_display: ReasoningDisplayConfig = field(default_factory=ReasoningDisplayConfig)


@dataclass(frozen=True)
class ModelPricingConfig:
    input: float = 0.0
    output: float = 0.0
    cached_input: float = 0.0
    unit: str | None = None


@dataclass(frozen=True)
class PricingConfig:
    currency: str = "USD"
    unit: str = "1M_tokens"
    models: dict[str, ModelPricingConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class LevelConfig:
    name: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompressionConfig:
    enabled: bool = True
    model_level: str | None = None
    trigger_ratio: float = 0.7
    min_tokens: int = 5_000


@dataclass(frozen=True)
class TitleGenerationConfig:
    enabled: bool = True
    model_level: str | None = None


@dataclass(frozen=True)
class StreamRetryConfig:
    max_retries: int = 5
    base: float = 1.0
    factor: float = 2.0
    max: float = 30.0
    jitter: float = 0.2


@dataclass(frozen=True)
class RuntimeConfig:
    default_level: str = "medium"
    ask_default_level: str | None = None
    store_provider_response: bool = False
    max_agent_rounds: int = 100
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    title_generation: TitleGenerationConfig = field(default_factory=TitleGenerationConfig)
    stream_retry: StreamRetryConfig = field(default_factory=StreamRetryConfig)


@dataclass(frozen=True)
class CompletionNotificationConfig:
    enabled: bool = True
    terminal: bool = True
    bell: bool = True


@dataclass(frozen=True)
class UiConfig:
    language: str = "auto"
    completion_notification: CompletionNotificationConfig = field(
        default_factory=CompletionNotificationConfig
    )


@dataclass(frozen=True)
class RunnerConfig:
    default_timeout_s: float = 7200.0
    max_output_bytes: int = 1_000_000
    max_run_logs: int = 200


@dataclass(frozen=True)
class PluginsConfig:
    disabled: list[str] = field(default_factory=list)
    config: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class AppConfig:
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    levels: dict[str, LevelConfig]
    runtime: RuntimeConfig
    runner: RunnerConfig
    ui: UiConfig = field(default_factory=UiConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)

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
            supports_images=model.supports_images,
            params=merged_params,
            message_passthrough=model.message_passthrough,
            reasoning_display=model.reasoning_display,
        )

    def provider_for_model(self, model: ModelConfig) -> ProviderConfig:
        try:
            return self.providers[model.provider]
        except KeyError as exc:
            raise ConfigError(f"Unknown provider for model {model.name}: {model.provider}") from exc

def default_config(project_root: Path) -> dict[str, Any]:
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
            "store_provider_response": False,
            "max_agent_rounds": 100,
            "compression": {
                "enabled": True,
                "trigger_ratio": 0.7,
                "min_tokens": 5_000,
            },
            "title_generation": {
                "enabled": True,
            },
            "stream_retry": {
                "max_retries": 5,
                "base": 1.0,
                "factor": 2.0,
                "max": 30.0,
                "jitter": 0.2,
            },
        },
        "ui": {
            "stream": True,
            "language": "auto",
            "completion_notification": {
                "enabled": True,
                "terminal": True,
                "bell": True,
            },
        },
        "runner": {
            "default_timeout_s": 7200,
            "max_output_bytes": 1_000_000,
            "max_run_logs": 200,
        },
        "plugins": {
            "disabled": [],
            "config": {},
        },
        "pricing": {
            "currency": "USD",
            "unit": "1M_tokens",
            "models": {},
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
    default_raw = default_config(root)
    raw = copy.deepcopy(default_raw)
    for path in paths if paths is not None else config_paths(root):
        if path.exists():
            raw = merge_config_layer(raw, json.loads(path.read_text(encoding="utf-8")), default_raw)
    return raw


def parse_config(raw: dict[str, Any], project_root: Path) -> AppConfig:
    providers: dict[str, ProviderConfig] = {}
    providers_raw = _object_dict(raw.get("providers", {}))
    for name, value in providers_raw.items():
        if not isinstance(value, dict):
            continue
        provider_value = dict(value)
        provider_value.pop("name", None)
        legacy_endpoint = provider_value.pop("endpoint", None)
        legacy_api = provider_value.pop("api_format", None)
        responses_raw = provider_value.pop("responses", None)
        chat_raw = provider_value.pop("chat_completions", None)
        anthropic_raw = provider_value.pop("anthropic_messages", None)
        provider_value.pop("reasoning_options", None)
        provider_value["message_passthrough"] = parse_message_passthrough(
            provider_value.pop("message_passthrough", {})
        )
        provider_value["reasoning_display"] = parse_reasoning_display(
            provider_value.pop("reasoning_display", {})
        )
        provider_value["responses"] = parse_endpoint_config(
            responses_raw,
            default_path=str(legacy_endpoint or "/responses"),
        )
        provider_value["chat_completions"] = parse_endpoint_config(
            chat_raw,
            default_path="/chat/completions",
        )
        provider_value["anthropic_messages"] = parse_endpoint_config(
            anthropic_raw,
            default_path="/v1/messages",
        )
        if legacy_api and legacy_api != "responses":
            # Older experimental configs used provider-level api_format. Models now own API choice.
            provider_value.setdefault("params", {})
        providers[name] = ProviderConfig(name=name, **provider_value)
    models: dict[str, ModelConfig] = {}
    models_raw = _object_dict(raw.get("models", {}))
    for name, value in models_raw.items():
        if not isinstance(value, dict):
            continue
        model_value = dict(value)
        model_value.setdefault("api", model_value.pop("api_format", "responses"))
        model_value.pop("reasoning_options", None)
        provider_defaults = providers.get(str(model_value.get("provider") or ""))
        provider_message_passthrough = (
            provider_defaults.message_passthrough
            if provider_defaults is not None
            else MessagePassthroughConfig()
        )
        provider_reasoning_display = (
            provider_defaults.reasoning_display
            if provider_defaults is not None
            else ReasoningDisplayConfig()
        )
        model_message_passthrough = model_value.pop("message_passthrough", None)
        model_reasoning_display = model_value.pop("reasoning_display", None)
        model_value["message_passthrough"] = merge_message_passthrough(
            provider_message_passthrough,
            model_message_passthrough,
        )
        model_value["reasoning_display"] = merge_reasoning_display(
            provider_reasoning_display,
            model_reasoning_display,
        )
        models[name] = ModelConfig(name=name, **model_value)
    levels: dict[str, LevelConfig] = {}
    levels_raw = _object_dict(raw.get("levels", {}))
    for name, value in levels_raw.items():
        if not isinstance(value, dict):
            continue
        level_value = dict(value)
        level_value.pop("reasoning", None)
        levels[name] = LevelConfig(name=name, **level_value)
    runtime_raw = _object_dict(raw.get("runtime", {}))
    compression = CompressionConfig(**_object_dict(runtime_raw.get("compression", {})))
    title_generation = TitleGenerationConfig(**_object_dict(runtime_raw.get("title_generation", {})))
    stream_retry = StreamRetryConfig(**_object_dict(runtime_raw.get("stream_retry", {})))
    default_level = runtime_raw.get("default_level", "medium")
    if default_level not in levels and levels:
        default_level = next(iter(levels))
    ask_default_level_raw = runtime_raw.get("ask_default_level")
    ask_default_level: str | None
    if isinstance(ask_default_level_raw, str) and ask_default_level_raw:
        ask_default_level = ask_default_level_raw if ask_default_level_raw in levels else None
    else:
        ask_default_level = None
    runtime = RuntimeConfig(
        default_level=default_level,
        ask_default_level=ask_default_level,
        store_provider_response=runtime_raw.get("store_provider_response", False),
        max_agent_rounds=runtime_raw.get("max_agent_rounds", 100),
        compression=compression,
        title_generation=title_generation,
        stream_retry=stream_retry,
    )
    runner_raw = _object_dict(raw.get("runner", {}))
    runner = RunnerConfig(
        default_timeout_s=float(runner_raw.get("default_timeout_s", 7200)),
        max_output_bytes=int(runner_raw.get("max_output_bytes", 1_000_000)),
        max_run_logs=int(runner_raw.get("max_run_logs", 200)),
    )
    ui_raw = _object_dict(raw.get("ui", {}))
    plugins_raw = _object_dict(raw.get("plugins", {}))
    disabled_raw = plugins_raw.get("disabled", [])
    disabled = [str(item) for item in disabled_raw if isinstance(item, str)] if isinstance(disabled_raw, list) else []
    plugin_config = {
        str(name): dict(value)
        for name, value in _object_dict(plugins_raw.get("config", {})).items()
        if isinstance(value, dict)
    }
    plugins = PluginsConfig(disabled=disabled, config=plugin_config)
    pricing = parse_pricing(raw.get("pricing", {}))
    return AppConfig(
        providers=providers,
        models=models,
        levels=levels,
        runtime=runtime,
        runner=runner,
        ui=UiConfig(
            language=str(ui_raw.get("language", "auto")),
            completion_notification=parse_completion_notification(
                ui_raw.get("completion_notification", {})
            ),
        ),
        plugins=plugins,
        pricing=pricing,
    )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _object_dict(value: object) -> dict[str, Any]:
    """Return a shallow dict only for JSON object values.

    Config files are user-controlled JSON. Narrowing at module boundaries keeps
    parser code from accidentally calling ``dict(...)`` on strings/lists, which
    would either crash with confusing errors or be interpreted as iterable pairs.
    """

    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def parse_endpoint_config(value: object, *, default_path: str) -> EndpointConfig:
    """Parse one provider endpoint block with a stable fallback path."""

    if value is None:
        return EndpointConfig(path=default_path)
    if isinstance(value, str):
        # Accept a compact string form as a forgiving shorthand for path-only
        # endpoints. Older configs already used provider-level endpoint strings.
        return EndpointConfig(path=value or default_path)
    data = _object_dict(value)
    unexpected = sorted(set(data) - {"path", "params"})
    if unexpected:
        raise TypeError(f"unknown endpoint config fields: {', '.join(unexpected)}")
    path = data.get("path", default_path)
    params = data.get("params", {})
    return EndpointConfig(
        path=str(path or default_path),
        params=_object_dict(params),
    )


def parse_message_passthrough(value: object) -> MessagePassthroughConfig:
    if isinstance(value, list):
        return MessagePassthroughConfig(assistant=[str(field) for field in value])
    if not isinstance(value, dict):
        return MessagePassthroughConfig()
    data = _object_dict(value)
    return MessagePassthroughConfig(
        assistant=string_list(data.get("assistant")),
        user=string_list(data.get("user")),
        system=string_list(data.get("system")),
        tool=string_list(data.get("tool")),
    )


def parse_reasoning_display(value: object) -> ReasoningDisplayConfig:
    if not isinstance(value, dict):
        return ReasoningDisplayConfig()
    data = _object_dict(value)
    return ReasoningDisplayConfig(
        assistant_message_fields=string_list(data.get("assistant_message_fields")),
        stream_delta_fields=string_list(data.get("stream_delta_fields")),
        unknown_text_delta_as_reasoning=bool(data.get("unknown_text_delta_as_reasoning", False)),
    )


def parse_completion_notification(value: object) -> CompletionNotificationConfig:
    if isinstance(value, bool):
        return CompletionNotificationConfig(enabled=value)
    if not isinstance(value, dict):
        return CompletionNotificationConfig()
    data = _object_dict(value)
    terminal = data.get("terminal", data.get("toast", True))
    return CompletionNotificationConfig(
        enabled=bool(data.get("enabled", True)),
        terminal=bool(terminal),
        bell=bool(data.get("bell", True)),
    )


def parse_pricing(value: object) -> PricingConfig:
    if not isinstance(value, dict):
        return PricingConfig()
    data = _object_dict(value)
    models: dict[str, ModelPricingConfig] = {}
    raw_models = data.get("models", {})
    if isinstance(raw_models, dict):
        for name, raw_price in raw_models.items():
            if not isinstance(name, str) or not isinstance(raw_price, dict):
                continue
            price_value = _object_dict(raw_price)
            # Model entries inherit the top-level unit unless they explicitly
            # override it. The resolved unit is handled in billing.py so the raw
            # config still reflects exactly what the user wrote.
            unit = price_value.get("unit")
            models[name] = ModelPricingConfig(
                input=float(price_value.get("input", 0.0) or 0.0),
                output=float(price_value.get("output", 0.0) or 0.0),
                cached_input=float(price_value.get("cached_input", 0.0) or 0.0),
                unit=str(unit) if unit is not None else None,
            )
    return PricingConfig(
        currency=str(data.get("currency", "USD") or "USD"),
        unit=str(data.get("unit", "1M_tokens") or "1M_tokens"),
        models=models,
    )


def merge_message_passthrough(
    base: MessagePassthroughConfig,
    override: object,
) -> MessagePassthroughConfig:
    if override is None:
        return base
    if isinstance(override, list):
        return MessagePassthroughConfig(assistant=[str(field) for field in override])
    if not isinstance(override, dict):
        return base
    data = _object_dict(override)
    return MessagePassthroughConfig(
        assistant=string_list(data.get("assistant")) if "assistant" in data else base.assistant,
        user=string_list(data.get("user")) if "user" in data else base.user,
        system=string_list(data.get("system")) if "system" in data else base.system,
        tool=string_list(data.get("tool")) if "tool" in data else base.tool,
    )


def merge_reasoning_display(
    base: ReasoningDisplayConfig,
    override: object,
) -> ReasoningDisplayConfig:
    if override is None or not isinstance(override, dict):
        return base
    data = _object_dict(override)
    return ReasoningDisplayConfig(
        assistant_message_fields=string_list(data.get("assistant_message_fields"))
        if "assistant_message_fields" in data
        else base.assistant_message_fields,
        stream_delta_fields=string_list(data.get("stream_delta_fields"))
        if "stream_delta_fields" in data
        else base.stream_delta_fields,
        unknown_text_delta_as_reasoning=bool(data.get("unknown_text_delta_as_reasoning"))
        if "unknown_text_delta_as_reasoning" in data
        else base.unknown_text_delta_as_reasoning,
    )


def string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def merge_config_layer(
    base: dict[str, Any],
    override: dict[str, Any],
    default_raw: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key == "levels"
            and isinstance(value, dict)
            and result.get(key) == default_raw.get(key)
        ):
            result[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config_layer(result[key], value, default_raw.get(key, {}))
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


def editable_config_path(project_root: Path) -> Path:
    """Return the config file the TUI should edit for user-facing settings."""
    user_path = user_config_path()
    if user_path.exists():
        return user_path
    return project_config_path(project_root)
