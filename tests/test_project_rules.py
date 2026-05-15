from __future__ import annotations

from pathlib import Path

from uv_agent.project_rules import discover_rule_files, load_project_rules


def test_discover_rule_files_loads_user_and_project_chain(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "repo"
    nested = project / "src" / "pkg"
    nested.mkdir(parents=True)
    (project / ".git").mkdir()
    user_rule = home / ".agents" / "AGENTS.md"
    user_rule.parent.mkdir(parents=True)
    user_rule.write_text("user", encoding="utf-8")
    root_rule = project / "AGENTS.md"
    root_rule.write_text("root", encoding="utf-8")
    py_rule = project / "AGENTS.python.md"
    py_rule.write_text("python", encoding="utf-8")
    nested_rule = nested / "AGENTS.md"
    nested_rule.write_text("nested", encoding="utf-8")

    files = discover_rule_files(nested, home=home)

    assert files == [
        ("user", user_rule),
        ("project", root_rule),
        ("project", py_rule),
        ("project", nested_rule),
    ]


def test_load_project_rules_caps_total_context(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    (project / "AGENTS.md").write_text("a" * 50, encoding="utf-8")
    (project / "AGENTS.extra.md").write_text("b" * 50, encoding="utf-8")

    context = load_project_rules(project, max_chars_per_file=30, max_total_chars=40)

    assert context.truncated
    assert len(context.rules) == 2
    rendered = context.render()
    assert "<workspace_rules>" in rendered
    assert "...[truncated]" in rendered
