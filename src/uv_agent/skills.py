from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillSummary:
    name: str
    path: Path
    description: str
    scope: str


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
    """Extract a compact description from a SKILL.md file."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped[:180]
    return "No description"


def render_skill_summary(skills: list[SkillSummary], *, limit: int = 10) -> str:
    """Render discovered skills for the system prompt."""
    if not skills:
        return "None discovered."
    lines = []
    for skill in skills[:limit]:
        lines.append(
            f"- {skill.name} ({skill.scope}): {skill.description}; read {skill.path}"
        )
    if len(skills) > limit:
        lines.append(f"- ... {len(skills) - limit} more skills available")
    return "\n".join(lines)
