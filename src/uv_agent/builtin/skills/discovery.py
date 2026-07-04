from __future__ import annotations

from dataclasses import dataclass
from html import escape as xml_escape
from pathlib import Path
from urllib.parse import quote

SKILL_DEFAULT_DESCRIPTION = "无描述"
SKILL_ENTRY_TEMPLATE = '<skill name="{name}" uri="{uri}">{description}</skill>'
SKILLS_NONE_DISCOVERED = "未发现。"


@dataclass(frozen=True)
class SkillSummary:
    name: str
    uri: str
    description: str
    path: Path | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.uri, self.name)


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
            name = skill_file.parent.name
            skills.append(
                SkillSummary(
                    name=name,
                    uri=skill_uri(scope, name),
                    path=resolved,
                    description=extract_description(skill_file),
                )
            )
    return skills


def skill_uri(scope: str, *segments: str) -> str:
    encoded = "/".join(quote(str(segment), safe="") for segment in segments if str(segment))
    return f"skill://{scope}/{encoded}" if encoded else f"skill://{scope}"


def extract_description(path: Path) -> str:
    return extract_description_text(path.read_text(encoding="utf-8"))


def extract_description_text(text: str) -> str:
    """Extract a compact description from SKILL.md text."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
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
    return SKILL_DEFAULT_DESCRIPTION


def _read_frontmatter_description(lines: list[str]) -> str:
    iterator = iter(enumerate(lines))
    for _idx, raw in iterator:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition(":")
        if not sep or key.strip().lower() != "description":
            continue
        value = value.strip()
        if value in {"|", ">", "|-", ">-", "|+", ">+"}:
            block: list[str] = []
            for _, follow in iterator:
                if follow and not follow[:1].isspace():
                    break
                block.append(follow.strip())
            return " ".join(part for part in block if part).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value
    return ""


def render_skill_summary(skills: list[SkillSummary]) -> str:
    if not skills:
        return SKILLS_NONE_DISCOVERED
    return "\n".join(render_skill_entry(skill) for skill in skills)


def render_skill_entry(skill: SkillSummary) -> str:
    return SKILL_ENTRY_TEMPLATE.format(
        name=_xml_attr(skill.name),
        uri=_xml_attr(skill.uri),
        description=_xml_text(skill.description),
    )


def _xml_attr(value: object) -> str:
    return xml_escape(str(value), quote=True)


def _xml_text(value: object) -> str:
    return xml_escape(str(value), quote=False)
