from __future__ import annotations

from pathlib import Path

from uv_agent.skills import discover_skills, render_skill_summary


def test_discover_project_skills(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".agents" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo\nUse this skill for demos.\n", encoding="utf-8")

    skills = discover_skills(tmp_path, home=tmp_path / "home")

    assert len(skills) == 1
    assert skills[0].name == "demo"
    assert skills[0].description == "Use this skill for demos."
    summary = render_skill_summary(skills)
    assert '<skill name="demo" scope="project"' in summary
    assert f'path="{skill_dir / "SKILL.md"}"' in summary
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
