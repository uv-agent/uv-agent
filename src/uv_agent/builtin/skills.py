from __future__ import annotations

from uv_agent.plugins import CommandResult, OpenPickerAction, PluginManifest, SetupPlugin
from uv_agent.skills import discover_skills


MANIFEST = PluginManifest(
    id="builtin.skills",
    version="0.1.0",
    display_name="Skills",
    description="Discover installed agent skills and publish them as epoch context.",
    builtin=True,
    priority=100,
    capabilities=("context", "ui"),
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


def setup(context) -> None:
    skills = discover_skills(context.project_root)
    body: dict[str, object] = {
        "rule": "遇到适合任务的 skill 时，先用 Python 读取它的 SKILL.md。",
        "skill": [
            {
                "name": skill.name,
                "scope": skill.scope,
                "path": str(skill.path),
                "description": skill.description,
            }
            for skill in skills[:10]
        ],
    }
    if len(skills) > 10:
        body["omitted"] = len(skills) - 10
    context.ui.picker(id="skills", title="Skills", provider=lambda query="": _skill_items(context.project_root, query), trigger="@skill")
    context.commands.register("/skills", lambda payload: CommandResult((OpenPickerAction("skills"),)), description="list skills and insert @skill mentions")
    context.context.epoch.publish(tag="available_skills", body=body)


def _skill_items(project_root, query: str = "") -> list[dict[str, str]]:
    needle = str(query or "").lower()
    items: list[dict[str, str]] = []
    for skill in discover_skills(project_root):
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
    return items[:30]
