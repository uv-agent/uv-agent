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
    assert "demo (project)" in render_skill_summary(skills)
