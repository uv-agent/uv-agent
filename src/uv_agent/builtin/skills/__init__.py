from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from uv_agent.blobs import guess_mime_type
from uv_agent.plugins import CommandResult, OpenPickerAction, PluginManifest, ResourceData, SetupPlugin
from .discovery import SkillSummary, discover_skills, extract_description_text, skill_uri
from .i18n import TEXTS


MANIFEST = PluginManifest(
    id="builtin.skills",
    version="0.1.0",
    display_name={"zh": "技能", "en": "Skills"},
    description={"zh": "发现已安装的 agent skills，并作为 epoch context 发布。", "en": "Discover installed agent skills and publish them as epoch context."},
    builtin=True,
    priority=100,
    capabilities=("context", "ui", "action"),
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


@dataclass
class RegisteredSkill:
    provider: str
    name: str
    skill_md: str
    description: str
    resources: dict[str, ResourceData] = field(default_factory=dict)

    @property
    def uri(self) -> str:
        return skill_uri("plugin", self.provider, self.name)

    def summary(self) -> SkillSummary:
        return SkillSummary(name=self.name, uri=self.uri, description=self.description)


@dataclass
class SkillsSnapshot:
    items_by_key: dict[tuple[str, str], dict[str, str]]


@dataclass
class SkillsState:
    snapshot: SkillsSnapshot | None = None
    plugin_skills: dict[tuple[str, str], RegisteredSkill] = field(default_factory=dict)


def setup(context) -> None:
    state = SkillsState()
    context.i18n.register(TEXTS)
    if hasattr(context, "resources"):
        context.resources.register(prefix="skill://", read=lambda uri, **_kwargs: _read_skill_resource(context, state, uri))
    if hasattr(context, "actions"):
        context.actions.register(
            "skills.register",
            lambda payload, context=None, caller_plugin=None: _register_skill(state, payload, caller_plugin=caller_plugin),
            doc="Register a plugin-provided skill.",
        )
        context.actions.register(
            "skills.unregister",
            lambda payload, context=None, caller_plugin=None: _unregister_skill(state, payload, caller_plugin=caller_plugin),
            doc="Unregister plugin-provided skills.",
        )
    _publish_skills_epoch(context, state)
    context.ui.picker(
        id="skills",
        title={"zh": "技能", "en": "Skills"},
        provider=lambda query="": _skill_items(context, state, query),
        trigger="@skill",
    )
    context.commands.register(
        "/skills",
        lambda payload: CommandResult((OpenPickerAction("skills"),)),
        description={"zh": "列出 skills 并插入 @skill 引用", "en": "list skills and insert @skill mentions"},
    )
    context.epoch.on_refresh(lambda thread_id=None: _publish_skills_epoch(context, state))
    context.events.subscribe(
        "thread.event_stored",
        lambda event: _sync_skills_on_turn_started(context, state, event),
        logger=context.logger,
    )


def _register_skill(state: SkillsState, payload: dict[str, Any], *, caller_plugin: str | None) -> dict[str, Any]:
    provider = str(caller_plugin or "").strip()
    if not provider or provider == "host":
        raise ValueError("skills.register must be called by a plugin context")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("skills.register requires name")
    skill_md = payload.get("skill_md")
    if not isinstance(skill_md, str) or not skill_md.strip():
        raise ValueError("skills.register requires skill_md text")
    description = str(payload.get("description") or "").strip() or extract_description_text(skill_md)
    resources = _normalize_resources(payload.get("resources") or {})
    registered = RegisteredSkill(
        provider=provider,
        name=name,
        skill_md=skill_md,
        description=description,
        resources=resources,
    )
    state.plugin_skills[(provider, name)] = registered
    return {"ok": True, "uri": registered.uri}


def _unregister_skill(state: SkillsState, payload: dict[str, Any], *, caller_plugin: str | None) -> dict[str, Any]:
    provider = str(caller_plugin or "").strip()
    if not provider or provider == "host":
        raise ValueError("skills.unregister must be called by a plugin context")
    removed = 0
    if payload.get("all") is True:
        for key in [key for key in state.plugin_skills if key[0] == provider]:
            state.plugin_skills.pop(key, None)
            removed += 1
        return {"ok": True, "removed": removed}
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("skills.unregister requires name or all=True")
    if state.plugin_skills.pop((provider, name), None) is not None:
        removed = 1
    return {"ok": True, "removed": removed}


def _normalize_resources(value: Any) -> dict[str, ResourceData]:
    if not isinstance(value, dict):
        raise TypeError("resources must be a mapping")
    resources: dict[str, ResourceData] = {}
    for raw_key, raw_value in value.items():
        key = _normalize_resource_key(str(raw_key))
        if key == "SKILL.md":
            raise ValueError("resources must not include SKILL.md; pass skill_md instead")
        resources[key] = _resource_data_for_key(key, raw_value)
    return resources


def _resource_data_for_key(key: str, value: Any) -> ResourceData:
    if isinstance(value, ResourceData):
        return value
    mime_type = _mime_for_resource(key, text=isinstance(value, str))
    if isinstance(value, str):
        return ResourceData(kind="text", text=value, mime_type=mime_type, filename=Path(key).name)
    if isinstance(value, bytes):
        return ResourceData(kind="bytes", data=value, mime_type=mime_type, filename=Path(key).name)
    if isinstance(value, Path):
        return ResourceData(kind="path", path=value, mime_type=guess_mime_type(key), filename=Path(key).name)
    if isinstance(value, dict):
        content = value.get("content")
        mime = str(value.get("mime_type") or mime_type)
        filename = str(value.get("filename") or Path(key).name)
        if isinstance(content, str):
            return ResourceData(kind="text", text=content, mime_type=mime, filename=filename)
        if isinstance(content, bytes):
            return ResourceData(kind="bytes", data=content, mime_type=mime, filename=filename)
        if isinstance(content, Path):
            return ResourceData(kind="path", path=content, mime_type=mime, filename=filename)
    raise TypeError(f"Unsupported resource value for {key}: {type(value).__name__}")


def _publish_skills_epoch(context, state: SkillsState) -> None:
    snapshot = _skills_snapshot(_all_skills(context, state))
    state.snapshot = snapshot
    context.epoch.publish(tag="available_skills", body=_skills_context_body(snapshot.items_by_key.values()))


def _sync_skills_context(context, state: SkillsState) -> None:
    previous = state.snapshot
    current = _skills_snapshot(_all_skills(context, state))
    if not isinstance(previous, SkillsSnapshot):
        state.snapshot = current
        return
    previous_keys = set(previous.items_by_key)
    current_keys = set(current.items_by_key)
    changed = [
        current.items_by_key[key]
        for key in sorted((current_keys - previous_keys) | {
            key for key in current_keys & previous_keys
            if current.items_by_key[key] != previous.items_by_key[key]
        })
    ]
    unavailable = [
        previous.items_by_key[key]
        for key in sorted(previous_keys - current_keys)
    ]
    if not changed and not unavailable:
        return
    state.snapshot = current
    body: dict[str, object] = {
        "rule": "skills context changed; treat this block as an incremental correction for the current epoch.",
    }
    if changed:
        body["skill"] = changed
    if unavailable:
        body["unavailable_skill"] = unavailable
    context.epoch.update(tag="available_skills", body=body)


def _sync_skills_on_turn_started(context, state: SkillsState, event: dict[str, object]) -> None:
    stored = event.get("event")
    if not isinstance(stored, dict) or stored.get("type") != "turn.started":
        return
    _sync_skills_context(context, state)


def _all_skills(context, state: SkillsState) -> list[SkillSummary]:
    return [*discover_skills(context.project_root), *(item.summary() for item in state.plugin_skills.values())]


def _skills_snapshot(skills: list[SkillSummary]) -> SkillsSnapshot:
    items = [_skill_body(skill) for skill in skills]
    return SkillsSnapshot(items_by_key={_skill_key(item): item for item in items})


def _skills_context_body(items) -> dict[str, object]:
    return {
        "rule": "遇到适合任务的 skill 或用户提到 @skill://... 时，用 rt.get(\"skill://...\") 读取它。",
        "skill": list(items),
    }


def _skill_body(skill: SkillSummary) -> dict[str, str]:
    return {
        "name": skill.name,
        "uri": skill.uri,
        "description": skill.description,
    }


def _skill_key(item: dict[str, str]) -> tuple[str, str]:
    return (item["uri"], item["name"])


def _skill_items(context, state: SkillsState, query: str = "") -> list[dict[str, str]]:
    _sync_skills_context(context, state)
    needle = str(query or "").lower()
    items: list[dict[str, str]] = []
    for skill in _all_skills(context, state):
        haystack = f"{skill.name} {skill.description} {skill.uri}".lower()
        if needle and needle not in haystack:
            continue
        items.append({
            "value": f"@{skill.uri}",
            "description": skill.description,
            "id": skill.uri,
            "kind": "skill-mention",
            "meta": skill.uri,
        })
    return items


def _read_skill_resource(context, state: SkillsState, uri: str) -> ResourceData:
    scope, segments = _parse_skill_uri(uri)
    if scope == "project" or scope == "user":
        if not segments:
            raise FileNotFoundError(f"Missing skill name in URI: {uri}")
        name = segments[0]
        relative = _resource_relative_path(segments[1:])
        for skill in discover_skills(context.project_root):
            if skill.uri != skill_uri(scope, name) or skill.path is None:
                continue
            root = skill.path.parent
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise FileNotFoundError(f"Resource escapes skill root: {uri}")
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"Skill resource not found: {uri}")
            return ResourceData(uri=uri, kind="path", path=path, mime_type=_mime_for_resource(path.name), filename=path.name)
        raise FileNotFoundError(f"Skill not found: {uri}")
    if scope == "plugin":
        if len(segments) < 2:
            raise FileNotFoundError(f"Missing plugin skill name in URI: {uri}")
        provider, name = segments[0], segments[1]
        skill = state.plugin_skills.get((provider, name))
        if skill is None:
            raise FileNotFoundError(f"Plugin skill not found: {uri}")
        relative = _resource_relative_path(segments[2:])
        if relative == Path("SKILL.md"):
            return ResourceData(uri=uri, kind="text", text=skill.skill_md, mime_type="text/markdown; charset=utf-8", filename="SKILL.md")
        key = relative.as_posix()
        resource = skill.resources.get(key)
        if resource is None:
            raise FileNotFoundError(f"Plugin skill resource not found: {uri}")
        return ResourceData(
            uri=uri,
            kind=resource.kind,
            text=resource.text,
            data=resource.data,
            path=resource.path,
            mime_type=resource.mime_type,
            filename=resource.filename,
            metadata=dict(resource.metadata),
        )
    raise FileNotFoundError(f"Unknown skill URI scope: {uri}")


def _parse_skill_uri(uri: str) -> tuple[str, list[str]]:
    prefix = "skill://"
    if not uri.startswith(prefix):
        raise FileNotFoundError(f"Not a skill URI: {uri}")
    rest = uri[len(prefix):]
    raw_parts = rest.split("/")
    if not raw_parts or any(part == "" for part in raw_parts):
        raise FileNotFoundError(f"Missing skill URI scope: {uri}")
    parts = [unquote(part) for part in raw_parts]
    scope, *segments = parts
    for segment in segments:
        if not segment or segment in {".", ".."} or "/" in segment or "\\" in segment:
            raise FileNotFoundError(f"Invalid skill URI segment: {uri}")
    return scope, segments


def _resource_relative_path(segments: list[str]) -> Path:
    if not segments:
        return Path("SKILL.md")
    if any(segment in {"", ".", ".."} or "/" in segment or "\\" in segment for segment in segments):
        raise FileNotFoundError("Invalid skill resource path")
    return Path(*segments)


def _normalize_resource_key(value: str) -> str:
    path = value.replace("\\", "/").strip("/")
    parts = [part for part in path.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid skill resource path: {value!r}")
    return "/".join(parts)


def _mime_for_resource(name: str | Path, *, text: bool | None = None) -> str:
    suffix = Path(name).suffix.lower()
    if text is True:
        if suffix == ".md":
            return "text/markdown; charset=utf-8"
        return "text/plain; charset=utf-8"
    guessed = guess_mime_type(name)
    if guessed == "application/octet-stream" and suffix == ".md":
        return "text/markdown; charset=utf-8"
    return guessed
