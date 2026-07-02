from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from uv_agent.environment import UserLanguage, normalize_language

LocalizedText: TypeAlias = str | Mapping[str, str]


@dataclass(frozen=True)
class I18nTextSpec:
    plugin: str
    key: str
    text: LocalizedText


class PluginI18nRegistry:
    """Registry for plugin-owned UI text."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._texts: dict[str, I18nTextSpec] = {}

    def register(self, *, plugin: str, texts: Mapping[str, LocalizedText]) -> None:
        normalized: dict[str, I18nTextSpec] = {}
        for key, text in texts.items():
            name = str(key).strip()
            if not name:
                raise ValueError("Plugin i18n text key cannot be empty")
            normalized[name] = I18nTextSpec(plugin=plugin, key=name, text=text)
        with self._lock:
            for key, spec in normalized.items():
                existing = self._texts.get(key)
                if existing is not None and existing.plugin != plugin:
                    raise ValueError(
                        f"Plugin i18n text key {key!r} already registered by {existing.plugin}"
                    )
                self._texts[key] = spec

    def text(self, key: str, language: UserLanguage | str | None = None) -> str:
        with self._lock:
            spec = self._texts.get(str(key))
        if spec is None:
            return ""
        return localize_text(spec.text, language)

    def list(self) -> list[I18nTextSpec]:
        with self._lock:
            return sorted(self._texts.values(), key=lambda item: item.key)


def localize_text(value: LocalizedText, language: UserLanguage | str | None = None) -> str:
    """Resolve plugin-provided text for the active UI language."""

    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return str(value or "")
    lang = language if isinstance(language, UserLanguage) else normalize_language(str(language or "en"))
    preferred = ("zh", "zh-CN", "zh_CN", "zh-Hans") if lang.is_chinese else ("en", "en-US", "en_US")
    for key in preferred:
        text = value.get(key)
        if isinstance(text, str) and text:
            return text
    fallback_order = ("en", "zh") if not lang.is_chinese else ("zh", "en")
    for key in fallback_order:
        text = value.get(key)
        if isinstance(text, str) and text:
            return text
    for text in value.values():
        if isinstance(text, str) and text:
            return text
    return ""
