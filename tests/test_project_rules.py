from __future__ import annotations

from pathlib import Path

from uv_agent.project_rules import (
    discover_rule_files,
    discover_workspace_rule_index,
    load_directory_rules,
    load_project_rules,
)


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
    rendered = context.render(context_path=".")
    assert '<workspace_rules path="." truncated="true">' in rendered
    assert '<rule file="' in rendered
    assert 'truncated="true"' in rendered
    assert "...[truncated]" in rendered


def test_workspace_rule_index_is_bounded_and_reports_limits(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    deep = project / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (project / "AGENTS.md").write_text("root", encoding="utf-8")
    (project / "a" / "AGENTS.md").write_text("a", encoding="utf-8")
    (project / "a" / "b" / "AGENTS.md").write_text("b", encoding="utf-8")
    (deep / "AGENTS.md").write_text("deep", encoding="utf-8")

    index = discover_workspace_rule_index(project, max_depth=1, max_entries=2)
    rendered = index.render()

    assert [path.name for path in index.paths] == ["AGENTS.md", "AGENTS.md"]
    assert index.depth_limited is True
    assert "<workspace_rule_index>" in rendered
    assert "active workspace" in rendered
    assert "scan_depth: 1" in rendered
    assert "max_entries: 2" in rendered
    assert "truncated: true" in rendered
    assert "depth_limit_reached" in rendered


def test_workspace_rule_index_can_render_active_working_directory_label(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    (project / "AGENTS.md").write_text("root", encoding="utf-8")

    rendered = discover_workspace_rule_index(project).render(label="working directory")

    assert "active working directory" in rendered
    assert "Use enter_dir" in rendered


def test_load_directory_rules_only_loads_current_directory(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    child = project / "child"
    child.mkdir(parents=True)
    (project / "AGENTS.md").write_text("root", encoding="utf-8")
    (child / "AGENTS.md").write_text("child", encoding="utf-8")

    context = load_directory_rules(project, root=project)

    rendered = context.render(root=project)
    assert "<workspace_rules>" in rendered
    assert '<rule file="AGENTS.md">' in rendered
    assert "root" in rendered
    assert "child" not in rendered
