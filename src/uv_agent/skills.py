from __future__ import annotations

from dataclasses import dataclass
from html import escape as xml_escape
from pathlib import Path

from uv_agent.prompts import SKILLS_NONE_DISCOVERED

@dataclass(frozen=True)
class SkillSummary:
    name: str
    path: Path
    description: str
    scope: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.scope, self.name, str(self.path))


def discover_skills(project_root: Path, *, home: Path | None = None) -> list[SkillSummary]:
    """Discover project and user skills stored under .agents/skills."""
    roots = [
        ("project", project_root.resolve() / ".agents" / "skills"),
        ("user", (home or Path.home()).resolve() / ".agents" / "skills"),
    ]
    skills: list[SkillSummary] = []
    seen: set[Path] = set()
    for scope, root in roots:
        if not root.exists():
            continue
        for skill_file in sorted(root.glob("*/SKILL.md")):
            resolved = skill_file.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            skills.append(
                SkillSummary(
                    name=skill_file.parent.name,
                    path=resolved,
                    description=extract_description(skill_file),
                    scope=scope,
                )
            )
    return skills


def extract_description(path: Path) -> str:
    """Extract a compact description from a SKILL.md file.

    SKILL.md may carry a YAML frontmatter block (Anthropic agent skills style)
    whose ``description`` field is the authoritative summary; if present, it
    wins. Otherwise we fall back to the first non-heading prose line so legacy
    SKILL.md files without frontmatter still surface a useful description.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        # Walk to the closing fence to scope the frontmatter parse.
        end = next(
            (idx for idx in range(1, len(lines)) if lines[idx].strip() == "---"),
            None,
        )
        if end is not None:
            description = _read_frontmatter_description(lines[1:end])
            if description:
                return description[:300]
            body_lines = lines[end + 1 :]
        else:
            body_lines = lines[1:]
    else:
        body_lines = lines
    for line in body_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped[:180]
    return "No description"


def _read_frontmatter_description(lines: list[str]) -> str:
    """Return the ``description`` field from a YAML-like frontmatter block.

    We deliberately implement a tiny ad-hoc reader instead of pulling in PyYAML:
    skill frontmatter is plain ``key: value`` plus optional multi-line block
    scalars (``description: |`` / ``>``), which is trivial to parse and keeps
    the project dependency-free.
    """
    iterator = iter(enumerate(lines))
    for idx, raw in iterator:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition(":")
        if not sep or key.strip().lower() != "description":
            continue
        value = value.strip()
        if value in {"|", ">", "|-", ">-", "|+", ">+"}:
            # Block scalar: collect subsequent indented lines.
            block: list[str] = []
            for _, follow in iterator:
                if follow and not follow[:1].isspace():
                    break
                block.append(follow.strip())
            joined = " ".join(part for part in block if part).strip()
            return joined
        # Plain scalar; trim surrounding quotes if any.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value
    return ""


def render_skill_summary(skills: list[SkillSummary], *, limit: int = 10) -> str:
    """Render discovered skills for the system prompt."""
    if not skills:
        return SKILLS_NONE_DISCOVERED
    lines = []
    for skill in skills[:limit]:
        lines.append(render_skill_entry(skill))
    if len(skills) > limit:
        lines.append(f'<omitted_skills count="{len(skills) - limit}" />')
    return "\n".join(lines)


def render_skill_entry(skill: SkillSummary) -> str:
    return (
        f'<skill name="{_xml_attr(skill.name)}" scope="{_xml_attr(skill.scope)}" '
        f'path="{_xml_attr(skill.path)}">{_xml_text(skill.description)}</skill>'
    )


def _xml_attr(value: object) -> str:
    return xml_escape(str(value), quote=True)


def _xml_text(value: object) -> str:
    return xml_escape(str(value), quote=False)
