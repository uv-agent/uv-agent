from __future__ import annotations

from importlib.resources import files
from pathlib import PurePosixPath
from typing import Any


async def register_packaged_skill(
    context: Any,
    *,
    package: str,
    root: str,
    name: str,
    description: str | None = None,
) -> dict[str, Any]:
    """Register a skill directory packaged inside a plugin distribution."""

    root_ref = files(package).joinpath(root)
    skill_ref = root_ref.joinpath("SKILL.md")
    if not skill_ref.is_file():
        raise FileNotFoundError(f"Packaged skill is missing SKILL.md: {package}:{root}")
    skill_md = skill_ref.read_text(encoding="utf-8")
    resources: dict[str, str | bytes] = {}

    def walk(ref, prefix: PurePosixPath = PurePosixPath("")) -> None:
        for child in ref.iterdir():
            child_rel = prefix / child.name
            if child.is_dir():
                walk(child, child_rel)
                continue
            if not child.is_file() or child_rel.as_posix() == "SKILL.md":
                continue
            data = child.read_bytes()
            try:
                resources[child_rel.as_posix()] = data.decode("utf-8")
            except UnicodeDecodeError:
                resources[child_rel.as_posix()] = data

    walk(root_ref)
    payload: dict[str, Any] = {
        "name": name,
        "skill_md": skill_md,
        "resources": resources,
    }
    if description:
        payload["description"] = description
    return await context.actions.call("skills.register", payload, missing="ignore")
