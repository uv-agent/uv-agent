from __future__ import annotations

import logging
from pathlib import Path

import pytest

from uv_agent.builtin.skills import setup as setup_skills
from uv_agent.builtin.skills.api import register_packaged_skill
from uv_agent.builtin.skills.discovery import discover_skills, render_skill_summary
from uv_agent.plugins.registry import ActionRegistry
from uv_agent.plugins.resources import ResourceRegistry


def test_discover_project_skills(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".agents" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo\nUse this skill for demos.\n", encoding="utf-8")

    skills = discover_skills(tmp_path, home=tmp_path / "home")

    assert len(skills) == 1
    assert skills[0].name == "demo"
    assert skills[0].description == "Use this skill for demos."
    summary = render_skill_summary(skills)
    assert '<skill name="demo" uri="skill://project/demo"' in summary
    assert ">Use this skill for demos.</skill>" in summary


def test_skill_description_reads_yaml_frontmatter(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".agents" / "skills" / "fm"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: fm\n"
        'description: Lightning-fast helper.\n'
        "---\n"
        "\n"
        "# fm\n"
        "Body paragraph follows.\n",
        encoding="utf-8",
    )

    skills = discover_skills(tmp_path, home=tmp_path / "home")

    assert len(skills) == 1
    assert skills[0].description == "Lightning-fast helper."


def test_skill_description_supports_block_scalar(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".agents" / "skills" / "block"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: block\n"
        "description: |\n"
        "  First line of description.\n"
        "  Second line gets joined.\n"
        "---\n"
        "# block\n",
        encoding="utf-8",
    )

    skills = discover_skills(tmp_path, home=tmp_path / "home")

    assert skills[0].description == "First line of description. Second line gets joined."


def test_render_skill_summary_escapes_xml_text(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".agents" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "description: Research & compare <tools>\n"
        "---\n",
        encoding="utf-8",
    )

    summary = render_skill_summary(discover_skills(tmp_path, home=tmp_path / "home"))

    assert ">Research &amp; compare &lt;tools&gt;</skill>" in summary


def test_render_skill_summary_does_not_omit_large_skill_sets(tmp_path: Path) -> None:
    skill_root = tmp_path / ".agents" / "skills"
    for index in range(12):
        skill_dir = skill_root / f"skill{index:02d}"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"skill {index}\n", encoding="utf-8")

    summary = render_skill_summary(discover_skills(tmp_path, home=tmp_path / "home"))

    assert '<skill name="skill11" uri="skill://project/skill11"' in summary
    assert "omitted" not in summary


def test_render_skill_summary_empty() -> None:
    assert render_skill_summary([]) == "未发现。"


def test_skills_refresh_publishes_full_and_sync_sends_incremental_update(tmp_path: Path) -> None:
    skill_root = tmp_path / ".agents" / "skills"
    first = skill_root / "first"
    first.mkdir(parents=True)
    (first / "SKILL.md").write_text("---\ndescription: first skill\n---\n", encoding="utf-8")

    class Registry:
        def __init__(self) -> None:
            self.provider = None

        def register(self, *args, **kwargs):
            return None

        def picker(self, *args, **kwargs):
            self.provider = kwargs.get("provider")
            return None

    class Epoch:
        def __init__(self) -> None:
            self.body = None
            self.updates = []
            self.refresher = None

        def publish(self, *, tag, body, attrs=None, thread_id=None):
            self.tag = tag
            self.body = body

        def update(self, *, tag, body, attrs=None, thread_id=None):
            self.updates.append((tag, body))

        def on_refresh(self, handler):
            self.refresher = handler

    class Events:
        def __init__(self) -> None:
            self.handler = None

        def subscribe(self, kinds, handler, *, logger=None, thread_id=None, turn_id=None):
            self.handler = handler
            return lambda: None

    class Context:
        def __init__(self) -> None:
            self.project_root = tmp_path
            self.i18n = Registry()
            self.ui = Registry()
            self.commands = Registry()
            self.epoch = Epoch()
            self.events = Events()
            self.logger = logging.getLogger("test.skills")

    context = Context()
    setup_skills(context)
    second = skill_root / "second"
    second.mkdir()
    (second / "SKILL.md").write_text("---\ndescription: second skill\n---\n", encoding="utf-8")

    context.epoch.refresher()

    assert context.epoch.tag == "available_skills"
    project_skills = [item for item in context.epoch.body["skill"] if item["uri"].startswith("skill://project/")]
    assert len(project_skills) == 2
    third = skill_root / "third"
    third.mkdir()
    (third / "SKILL.md").write_text("---\ndescription: third skill\n---\n", encoding="utf-8")

    assert context.events.handler is not None
    context.events.handler({"type": "thread.event_stored", "event": {"type": "turn.started"}})

    tag, body = context.epoch.updates[-1]
    assert tag == "available_skills"
    assert set(body) == {"rule", "skill"}
    assert body["skill"] == [
        {
            "name": "third",
            "uri": "skill://project/third",
            "description": "third skill",
        }
    ]


@pytest.mark.asyncio
async def test_plugin_skills_register_unregister_and_read_resources_by_uri(tmp_path: Path) -> None:
    class Registry:
        def register(self, *args, **kwargs):
            return None

        def picker(self, *args, **kwargs):
            return None

    class Epoch:
        def publish(self, *args, **kwargs):
            return None

        def on_refresh(self, handler):
            self.refresher = handler

    class Events:
        def subscribe(self, *args, **kwargs):
            return lambda: None

    class Resources:
        def __init__(self) -> None:
            self.registry = ResourceRegistry()

        def register(self, *, prefix, read):
            return self.registry.register(plugin="builtin.skills", prefix=prefix, read=read)

    class Actions:
        def __init__(self) -> None:
            self.registry = ActionRegistry()

        def register(self, action_id, handler, *, doc="", schema=None):
            return self.registry.register(plugin="builtin.skills", action_id=action_id, handler=handler, doc=doc, schema=schema)

    class Context:
        def __init__(self) -> None:
            self.project_root = tmp_path
            self.i18n = Registry()
            self.ui = Registry()
            self.commands = Registry()
            self.epoch = Epoch()
            self.events = Events()
            self.resources = Resources()
            self.actions = Actions()
            self.logger = logging.getLogger("test.skills")

    context = Context()
    setup_skills(context)

    result = await context.actions.registry.call(
        "skills.register",
        {
            "name": "demo",
            "skill_md": "---\ndescription: plugin skill\n---\n# Demo\n",
            "resources": {
                "docs/guide.md": "hello guide",
                "image.bin": b"\x00\xff",
            },
        },
        caller_plugin="plugin.pack",
    )

    assert result == {"ok": True, "uri": "skill://plugin/plugin.pack/demo"}
    skill = context.resources.registry.read("skill://plugin/plugin.pack/demo")
    assert skill.text.startswith("---\ndescription: plugin skill")
    guide = context.resources.registry.read("skill://plugin/plugin.pack/demo/docs/guide.md")
    assert guide.text == "hello guide"
    image = context.resources.registry.read("skill://plugin/plugin.pack/demo/image.bin")
    assert image.data == b"\x00\xff"

    await context.actions.registry.call("skills.unregister", {"name": "demo"}, caller_plugin="plugin.pack")
    with pytest.raises(FileNotFoundError):
        context.resources.registry.read("skill://plugin/plugin.pack/demo")


@pytest.mark.asyncio
async def test_register_packaged_skill_reads_all_packaged_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = tmp_path / "demo_pkg"
    skill_dir = package_dir / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text("# Demo\nUse packaged skill.\n", encoding="utf-8")
    (skill_dir / "guide.txt").write_text("hello", encoding="utf-8")
    (skill_dir / "asset.bin").write_bytes(b"\x00\xff")
    monkeypatch.syspath_prepend(str(tmp_path))

    class Actions:
        def __init__(self) -> None:
            self.calls = []

        async def call(self, action_id, payload, *, missing):
            self.calls.append((action_id, payload, missing))
            return {"ok": True, "uri": "skill://plugin/demo/demo"}

    class Context:
        def __init__(self) -> None:
            self.actions = Actions()

    context = Context()
    result = await register_packaged_skill(context, package="demo_pkg", root="skills/demo", name="demo")

    assert result["ok"] is True
    action_id, payload, missing = context.actions.calls[0]
    assert action_id == "skills.register"
    assert missing == "ignore"
    assert payload["name"] == "demo"
    assert payload["skill_md"] == "# Demo\nUse packaged skill.\n"
    assert payload["resources"] == {
        "guide.txt": "hello",
        "asset.bin": b"\x00\xff",
    }
