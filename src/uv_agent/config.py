from __future__ import annotations

import copy
import json
import os
from importlib.metadata import PackageNotFoundError, version
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from uv_agent.paths import project_config_path, project_local_dir, user_config_path


SENSITIVE_KEYS = {"api_key", "authorization", "token", "secret", "password"}
DEFAULT_PACKAGE_NAME = "uv-agent"


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
class RuntimeConfig:
    default_level: str = "medium"
    store_provider_response: bool = False
    max_agent_rounds: int = 100
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    title_generation: TitleGenerationConfig = field(default_factory=TitleGenerationConfig)


@dataclass(frozen=True)
class UiConfig:
    language: str = "auto"


@dataclass(frozen=True)
class RunnerConfig:
    runtime_dependency: str
    runtime_package_name: str = "uv-agent"
    default_uv_args: list[str] = field(default_factory=list)
    default_timeout_s: float = 7200.0
    max_output_bytes: int = 1_000_000
    max_saved_scripts: int = 32

    def __post_init__(self) -> None:
        if not self.default_uv_args:
            object.__setattr__(
                self,
                "default_uv_args",
                default_runtime_uv_args(self.runtime_dependency, self.runtime_package_name),
            )


@dataclass(frozen=True)
class AppConfig:
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    levels: dict[str, LevelConfig]
    runtime: RuntimeConfig
    runner: RunnerConfig
    ui: UiConfig = field(default_factory=UiConfig)

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
    runtime_dependency = default_runtime_dependency(DEFAULT_PACKAGE_NAME)
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
        },
        "ui": {
            "stream": True,
            "language": "auto",
        },
        "runner": {
            "runtime_dependency": runtime_dependency,
            "runtime_package_name": DEFAULT_PACKAGE_NAME,
            "default_timeout_s": 7200,
            "max_output_bytes": 1_000_000,
            "max_saved_scripts": 32,
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
    providers = {}
    for name, value in raw.get("providers", {}).items():
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
    levels = {}
    for name, value in raw.get("levels", {}).items():
        level_value = dict(value)
        level_value.pop("reasoning", None)
        levels[name] = LevelConfig(name=name, **level_value)
    runtime_raw = raw.get("runtime", {})
    compression = CompressionConfig(**dict(runtime_raw.get("compression", {})))
    title_generation = TitleGenerationConfig(**dict(runtime_raw.get("title_generation", {})))
    default_level = runtime_raw.get("default_level", "medium")
    if default_level not in levels and levels:
        default_level = next(iter(levels))
    runtime = RuntimeConfig(
        default_level=default_level,
        store_provider_response=runtime_raw.get("store_provider_response", False),
        max_agent_rounds=runtime_raw.get("max_agent_rounds", 100),
        compression=compression,
        title_generation=title_generation,
    )
    runner_raw = raw.get("runner", {})
    runner_package_name = runner_raw.get("runtime_package_name", DEFAULT_PACKAGE_NAME)
    runner_dependency = runner_raw.get(
        "runtime_dependency",
        default_runtime_dependency(runner_package_name),
    )
    runner_uv_args = (
        list(runner_raw["default_uv_args"])
        if "default_uv_args" in runner_raw
        else default_runtime_uv_args(runner_dependency, runner_package_name)
    )
    runner = RunnerConfig(
        runtime_dependency=runner_dependency,
        runtime_package_name=runner_package_name,
        default_uv_args=runner_uv_args,
        default_timeout_s=float(runner_raw.get("default_timeout_s", 7200)),
        max_output_bytes=int(runner_raw.get("max_output_bytes", 1_000_000)),
        max_saved_scripts=int(runner_raw.get("max_saved_scripts", 32)),
    )
    ui_raw = raw.get("ui", {})
    return AppConfig(
        providers=providers,
        models=models,
        levels=levels,
        runtime=runtime,
        runner=runner,
        ui=UiConfig(language=str(ui_raw.get("language", "auto"))),
    )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_message_passthrough(value: object) -> MessagePassthroughConfig:
    if isinstance(value, list):
        return MessagePassthroughConfig(assistant=[str(field) for field in value])
    if not isinstance(value, dict):
        return MessagePassthroughConfig()
    return MessagePassthroughConfig(
        assistant=string_list(value.get("assistant")),
        user=string_list(value.get("user")),
        system=string_list(value.get("system")),
        tool=string_list(value.get("tool")),
    )


def parse_reasoning_display(value: object) -> ReasoningDisplayConfig:
    if not isinstance(value, dict):
        return ReasoningDisplayConfig()
    return ReasoningDisplayConfig(
        assistant_message_fields=string_list(value.get("assistant_message_fields")),
        stream_delta_fields=string_list(value.get("stream_delta_fields")),
        unknown_text_delta_as_reasoning=bool(value.get("unknown_text_delta_as_reasoning", False)),
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
    return MessagePassthroughConfig(
        assistant=string_list(override["assistant"]) if "assistant" in override else base.assistant,
        user=string_list(override["user"]) if "user" in override else base.user,
        system=string_list(override["system"]) if "system" in override else base.system,
        tool=string_list(override["tool"]) if "tool" in override else base.tool,
    )


def merge_reasoning_display(
    base: ReasoningDisplayConfig,
    override: object,
) -> ReasoningDisplayConfig:
    if override is None or not isinstance(override, dict):
        return base
    return ReasoningDisplayConfig(
        assistant_message_fields=string_list(override["assistant_message_fields"])
        if "assistant_message_fields" in override
        else base.assistant_message_fields,
        stream_delta_fields=string_list(override["stream_delta_fields"])
        if "stream_delta_fields" in override
        else base.stream_delta_fields,
        unknown_text_delta_as_reasoning=bool(override["unknown_text_delta_as_reasoning"])
        if "unknown_text_delta_as_reasoning" in override
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


def default_runtime_uv_args(runtime_dependency: str, package_name: str) -> list[str]:
    """Return uv args that keep local file dependencies fresh during development."""
    if " @ file:" not in runtime_dependency:
        return []
    return ["--reinstall-package", package_name]


def default_runtime_dependency(package_name: str = DEFAULT_PACKAGE_NAME) -> str:
    """Return the installable package spec injected into managed scripts."""
    try:
        package_version = version(package_name)
    except PackageNotFoundError:
        return package_name
    return f"{package_name}=={package_version}"
