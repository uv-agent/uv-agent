from __future__ import annotations

import json
from typing import Any

from rich.text import Text

from uv_agent.config import (
    ConfigError,
    config_sources,
    editable_config_path,
    load_config,
    load_raw_config,
    redact_config,
)
from uv_agent.environment import detect_user_language
from uv_agent.tui.formatting import format_tokens, join_lines, plain
from uv_agent.tui.state import PickerItem


class ConfigPanelMixin:
    def _open_config_panel(self, *, replace_current: bool = False) -> None:
        self.engine.refresh_config()
        default_level = self.engine.config.runtime.default_level
        items = [
            PickerItem(
                id="default_level",
                title=self._text("config_default_level"),
                description=default_level,
                meta=self._text("config_default_level_hint"),
            ),
            PickerItem(
                id="language",
                title=self._text("config_language"),
                description=self.engine.config.ui.language,
                meta=self._text("config_language_hint"),
            ),
            PickerItem(
                id="completion_notification",
                title=self._text("config_completion_notification"),
                description=(
                    "on" if self.engine.config.ui.completion_notification.enabled else "off"
                ),
                meta=self._text("config_completion_notification_hint"),
            ),
            PickerItem(
                id="compression",
                title=self._text("config_compression"),
                description="on" if self.engine.config.runtime.compression.enabled else "off",
                meta=self._text("config_compression_hint"),
            ),
            PickerItem(
                id="models",
                title=self._text("models"),
                description=self._text("models_hint"),
                meta=self._text("config_models_readonly_hint"),
            ),
            PickerItem(
                id="sources",
                title=self._text("config_sources"),
                description=str(editable_config_path(self.project_root)),
                meta=self._text("config_sources_hint"),
            ),
            PickerItem(
                id="raw",
                title=self._text("config_raw"),
                description=self._text("config_raw_hint"),
            ),
        ]
        subtitle = (
            self._text("config_hint")
            + " · "
            + self._text("config_models_readonly_hint")
        )
        self._open_picker(
            self._text("config"),
            items,
            self._choose_config_item,
            subtitle=subtitle,
            replace_current=replace_current,
        )

    def _choose_config_item(self, item_id: str) -> None:
        if item_id == "default_level":
            self._open_default_level_panel()
        elif item_id == "language":
            self._open_language_panel()
        elif item_id == "completion_notification":
            self._toggle_completion_notification()
        elif item_id == "compression":
            self._toggle_compression()
        elif item_id == "models":
            self._open_models_panel()
        elif item_id == "sources":
            self._open_config_sources_panel()
        elif item_id == "raw":
            self._open_config_raw_panel()

    def _close_active_panel(self) -> None:
        panel = self._active_fullscreen_panel()
        if panel is not None:
            panel.close_navigation()

    def _open_default_level_panel(self) -> None:
        items = []
        current = self.engine.config.runtime.default_level
        for name, level in self.engine.config.levels.items():
            marker = self._text("current") if name == current else ""
            items.append(
                PickerItem(
                    id=name,
                    title=name,
                    description=level.model,
                    meta=marker,
                )
            )
        self._open_picker(
            self._text("config_default_level"),
            items,
            self._set_default_level,
            subtitle=self._text("config_write_hint"),
        )

    def _open_current_level_panel(self) -> None:
        items = []
        current = self.level or self.engine.config.runtime.default_level
        for name, level in self.engine.config.levels.items():
            marker = self._text("current") if name == current else ""
            items.append(
                PickerItem(
                    id=name,
                    title=name,
                    description=level.model,
                    meta=marker,
                )
            )
        self._open_picker(
            self._text("config_current_level"),
            items,
            self._set_current_level,
            subtitle=self._text("config_session_hint"),
        )

    def _open_language_panel(self) -> None:
        current = self.engine.config.ui.language
        items = [
            PickerItem(id=value, title=label, description=self._text("current") if value == current else "")
            for value, label in (("auto", "auto"), ("en", "English"), ("zh-CN", "中文"))
        ]
        self._open_picker(
            self._text("config_language"),
            items,
            self._set_language,
            subtitle=self._text("config_write_hint"),
        )

    def _toggle_compression(self) -> None:
        current = self.engine.config.runtime.compression.enabled
        self._write_user_config_patch({"runtime": {"compression": {"enabled": not current}}})
        self._flash(
            f"{self._text('config_compression')}: {'on' if not current else 'off'}",
        )
        self._open_config_panel(replace_current=True)

    def _toggle_completion_notification(self) -> None:
        current = self.engine.config.ui.completion_notification.enabled
        self._write_user_config_patch({"ui": {"completion_notification": {"enabled": not current}}})
        self._flash(
            f"{self._text('config_completion_notification')}: {'on' if not current else 'off'}",
        )
        self._open_config_panel(replace_current=True)

    def _open_config_sources_panel(self) -> None:
        sources = config_sources(self.project_root)
        lines = [Text("sources", style="bold")]
        for source in sources:
            exists = "yes" if source["exists"] else "no"
            lines.append(
                Text.assemble(f"- {source['scope']}: {source['path']} ", (f"exists={exists}", "dim"))
            )
        lines.extend([Text(), Text("editable", style="bold"), plain(str(editable_config_path(self.project_root)))])
        self._open_panel(join_lines(lines), "config", self._text("config_sources"))

    def _open_config_raw_panel(self) -> None:
        redacted = redact_config(load_raw_config(self.project_root))
        preview = json.dumps(redacted, ensure_ascii=False, indent=2)
        if len(preview) > 3200:
            preview = preview[:3200].rstrip() + "\n..."
        self._open_panel(plain(preview), "config", self._text("config_raw"))

    def _set_default_level(self, name: str) -> None:
        if name not in self.engine.config.levels:
            self._flash(f"{self._text('unknown_level')}: {name}", severity="error")
            return
        self._write_user_config_patch({"runtime": {"default_level": name}})
        self._flash(f"{self._text('config_default_level')}: {name}")
        if self.level is None:
            self._refresh_status()
        self._close_active_panel()

    def _set_current_level(self, name: str) -> None:
        self._handle_level_command(name)
        self._close_active_panel()

    def _set_language(self, value: str) -> None:
        self._write_user_config_patch({"ui": {"language": value}})
        self._flash(f"{self._text('config_language')}: {value}")
        self._close_active_panel()

    def _write_user_config_patch(self, patch: dict[str, Any]) -> None:
        path = editable_config_path(self.project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raw = {}
        else:
            raw = {}
        updated = self._config_deep_merge(raw, patch)
        path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.engine.config = load_config(self.project_root)
        self.engine.runner.config = self.engine.config.runner
        if hasattr(self.engine.model_client, "reload_config"):
            self.engine.model_client.reload_config(self.engine.config)  # type: ignore[attr-defined]
        self.language = detect_user_language(self.engine.config.ui.language)
        self._refresh_status()

    def _config_deep_merge(self, base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in patch.items():
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = self._config_deep_merge(current, value)
            else:
                merged[key] = value
        return merged

    def _open_models_panel(self) -> None:
        """Read-only models picker. Editing models lives in config.json."""
        self.engine.refresh_config()
        items: list[PickerItem] = []
        # Show which level each configured model is referenced by so users can
        # cross-reference without having to open config.json first.
        levels_by_model: dict[str, list[str]] = {}
        for level_name, level in self.engine.config.levels.items():
            levels_by_model.setdefault(level.model, []).append(level_name)
        for name, model in self.engine.config.models.items():
            level_refs = ", ".join(levels_by_model.get(name, [])) or "-"
            items.append(
                PickerItem(
                    id=name,
                    title=name,
                    description=f"{model.model}  ·  {model.api}",
                    meta=(
                        f"{self._text('models_provider')}: {model.provider}  ·  "
                        f"{self._text('models_context_window')}: "
                        f"{format_tokens(model.context_window_tokens)}  ·  "
                        f"{self._text('level')}: {level_refs}"
                    ),
                )
            )
        if not items:
            items.append(
                PickerItem(
                    id="",
                    title=self._text("none"),
                    description=self._text("models_edit_hint"),
                    meta=str(editable_config_path(self.project_root)),
                )
            )
        subtitle = (
            self._text("models_hint")
            + "  ·  "
            + self._text("models_edit_hint")
            + " "
            + str(editable_config_path(self.project_root))
        )
        self._open_picker(
            self._text("models"),
            items,
            self._open_model_detail_panel,
            subtitle=subtitle,
        )

    def _open_model_detail_panel(self, name: str) -> None:
        if not name:
            return
        model = self.engine.config.models.get(name)
        if model is None:
            self._flash(f"{self._text('models')}: {name}", severity="error")
            return
        try:
            provider = self.engine.config.provider_for_model(model)
        except ConfigError as exc:
            self._flash(str(exc), severity="error")
            return
        lines = [
            Text(name, style="bold cyan"),
            Text(f"- {self._text('models_provider')}: {provider.name}"),
            Text(f"- model: {model.model}"),
            Text(f"- {self._text('models_api')}: {model.api}"),
            Text(
                f"- {self._text('models_context_window')}: "
                f"{format_tokens(model.context_window_tokens)}"
            ),
        ]
        lines.append(Text())
        lines.append(
            Text(
                f"{self._text('models_edit_hint')} "
                f"{editable_config_path(self.project_root)}",
                style="dim",
            )
        )
        self._open_panel(join_lines(lines), "models", self._text("models"))

    def _handle_level_command(self, name: str) -> None:
        if not name:
            self._open_models_panel()
            return
        if name not in self.engine.config.levels:
            self._append_cell(Text.assemble((self._text("unknown_level"), "red"), " ", name), "error")
            return
        previous_level = self._current_level_for_thread(self.thread_id)
        if (
            self.thread_id is not None
            and previous_level
            and previous_level != name
            and not self._levels_use_same_model(previous_level, name)
        ):
            self._append_model_switch_warning(
                self.thread_id,
                from_level=previous_level,
                to_level=name,
            )
        self.level = name
        if self.thread_id is not None:
            self._persist_thread_level(self.thread_id, name)
        self._append_cell(Text.assemble((self._text("level"), "dim"), " ", (name, "cyan")), "event")
        self._refresh_status()

    def _open_mcp_panel(self) -> None:
        self.engine.refresh_config()
        panel = self._active_fullscreen_panel()
        if panel is not None and panel.can_navigate:
            panel.close_navigation()
            self.call_after_refresh(self._open_mcp_panel)
            return
        self._open_picker(
            self._text("mcp"),
            self._mcp_mention_items(),
            self._choose_mcp_mention,
            subtitle=self._text("mention_mcp_hint"),
        )

    def _open_skills_panel(self) -> None:
        self.engine.refresh_config()
        panel = self._active_fullscreen_panel()
        if panel is not None and panel.can_navigate:
            panel.close_navigation()
            self.call_after_refresh(self._open_skills_panel)
            return
        self._open_picker(
            self._text("skills"),
            self._skill_mention_items(),
            self._choose_skill_mention,
            subtitle=self._text("mention_skills_hint"),
        )

    def _noop_select(self, _value: str) -> None:
        """Callback used by inspect-only pickers."""
        return
