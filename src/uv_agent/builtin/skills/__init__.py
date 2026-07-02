from __future__ import annotations

from dataclasses import dataclass

from uv_agent.plugins import CommandResult, OpenPickerAction, PluginManifest, SetupPlugin
from .discovery import SkillSummary, discover_skills
from .i18n import TEXTS


MANIFEST = PluginManifest(
    id="builtin.skills",
    version="0.1.0",
    display_name={"zh": "技能", "en": "Skills"},
    description={"zh": "发现已安装的 agent skills，并作为 epoch context 发布。", "en": "Discover installed agent skills and publish them as epoch context."},
    builtin=True,
    priority=100,
    capabilities=("context", "ui"),
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


@dataclass
class SkillsSnapshot:
    items_by_key: dict[tuple[str, str, str], dict[str, str]]


@dataclass
class SkillsState:
    snapshot: SkillsSnapshot | None = None


def setup(context) -> None:
    state = SkillsState()
    context.i18n.register(TEXTS)
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


def _publish_skills_epoch(context, state: SkillsState) -> None:
    snapshot = _skills_snapshot(discover_skills(context.project_root))
    state.snapshot = snapshot
    context.epoch.publish(tag="available_skills", body=_skills_context_body(snapshot.items_by_key.values()))


def _sync_skills_context(context, state: SkillsState) -> None:
    previous = state.snapshot
    current = _skills_snapshot(discover_skills(context.project_root))
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


def _skills_snapshot(skills: list[SkillSummary]) -> SkillsSnapshot:
    items = [_skill_body(skill) for skill in skills]
    return SkillsSnapshot(
        items_by_key={_skill_key(item): item for item in items},
    )


def _skills_context_body(items) -> dict[str, object]:
    return {
        "rule": "遇到适合任务的 skill 时，先用 Python 读取它的 SKILL.md。",
        "skill": list(items),
    }


def _skill_body(skill: SkillSummary) -> dict[str, str]:
    return {
        "name": skill.name,
        "scope": skill.scope,
        "path": str(skill.path),
        "description": skill.description,
    }


def _skill_key(item: dict[str, str]) -> tuple[str, str, str]:
    return (item["scope"], item["name"], item["path"])


def _skill_items(context, state: SkillsState, query: str = "") -> list[dict[str, str]]:
    _sync_skills_context(context, state)
    needle = str(query or "").lower()
    items: list[dict[str, str]] = []
    for skill in discover_skills(context.project_root):
        haystack = f"{skill.name} {skill.description} {skill.scope}".lower()
        if needle and needle not in haystack:
            continue
        items.append({
            "value": f"@skill:{skill.name}",
            "description": skill.description,
            "id": skill.name,
            "kind": "skill-mention",
            "meta": f"{skill.scope} · {skill.path}",
        })
    return items
