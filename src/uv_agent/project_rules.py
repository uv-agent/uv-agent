from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RULE_FILE_NAMES = ("AGENTS.md",)
RULE_FILE_GLOB = "AGENTS.*.md"
DEFAULT_MAX_CHARS_PER_FILE = 12_000
DEFAULT_MAX_TOTAL_CHARS = 36_000


@dataclass(frozen=True)
class ProjectRule:
    """A single loaded instruction file for the active workspace."""

    path: Path
    scope: str
    text: str
    truncated: bool = False


@dataclass(frozen=True)
class ProjectRuleContext:
    """Loaded project-rule context plus discovery metadata for UI/debug views."""

    rules: list[ProjectRule]
    truncated: bool = False
    omitted_files: int = 0

    @property
    def paths(self) -> list[Path]:
        return [rule.path for rule in self.rules]

    def render(self) -> str:
        """Render loaded rules as a compact context block for model input."""
        if not self.rules:
            return ""
        lines = [
            "<workspace_rules>",
            "The following instruction files were loaded for the current workspace. Follow them when relevant; newer user messages still define the immediate task.",
        ]
        for rule in self.rules:
            rel = str(rule.path)
            suffix = " (truncated)" if rule.truncated else ""
            lines.extend(
                [
                    f"\n## {rule.scope}: {rel}{suffix}",
                    rule.text.strip(),
                ]
            )
        if self.truncated:
            lines.append(
                f"\nNote: rule context was capped; {self.omitted_files} file(s) were omitted."
            )
        lines.append("</workspace_rules>")
        return "\n".join(lines)


def load_project_rules(
    project_root: Path,
    *,
    home: Path | None = None,
    max_chars_per_file: int = DEFAULT_MAX_CHARS_PER_FILE,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> ProjectRuleContext:
    """Discover and load AGENTS instruction files for a workspace.

    Discovery loads user rules from ``~/.agents/AGENTS.md`` first, then project
    rules from the git root down to ``project_root``. The rendered output is
    intended to be appended to a request, not baked into the stable system
    prompt.
    """
    root = project_root.resolve()
    total = 0
    omitted = 0
    truncated_context = False
    rules: list[ProjectRule] = []
    seen: set[Path] = set()

    for scope, path in discover_rule_files(root, home=home):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if total >= max_total_chars:
            omitted += 1
            truncated_context = True
            continue
        try:
            text = resolved.read_text(encoding="utf-8")
        except OSError:
            continue
        remaining = max_total_chars - total
        limit = min(max_chars_per_file, remaining)
        truncated_file = len(text) > limit
        clipped = text[:limit]
        total += len(clipped)
        if truncated_file:
            clipped = clipped.rstrip() + "\n...[truncated]"
            truncated_context = True
        rules.append(ProjectRule(path=resolved, scope=scope, text=clipped, truncated=truncated_file))

    return ProjectRuleContext(rules=rules, truncated=truncated_context, omitted_files=omitted)


def discover_rule_files(project_root: Path, *, home: Path | None = None) -> list[tuple[str, Path]]:
    """Return rule files in precedence order from broadest to most local."""
    root = project_root.resolve()
    files: list[tuple[str, Path]] = []

    user_rule = (home or Path.home()).resolve() / ".agents" / "AGENTS.md"
    if user_rule.exists():
        files.append(("user", user_rule))

    git_root = find_git_root(root) or root
    for directory in path_chain(git_root, root):
        files.extend(("project", path) for path in rule_files_in_dir(directory))
    return files


def find_git_root(start: Path) -> Path | None:
    """Find the nearest ancestor containing a .git entry."""
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for directory in [current, *current.parents]:
        if (directory / ".git").exists():
            return directory
    return None


def path_chain(root: Path, leaf: Path) -> list[Path]:
    """Return directories from root to leaf, inclusive when leaf is below root."""
    root = root.resolve()
    leaf = leaf.resolve()
    if leaf.is_file():
        leaf = leaf.parent
    try:
        relative = leaf.relative_to(root)
    except ValueError:
        return [leaf]
    directories = [root]
    current = root
    for part in relative.parts:
        current = current / part
        directories.append(current)
    return directories


def rule_files_in_dir(directory: Path) -> list[Path]:
    """Return AGENTS files in one directory with the base file first."""
    files: list[Path] = []
    for name in RULE_FILE_NAMES:
        path = directory / name
        if path.exists():
            files.append(path)
    extension_files = sorted(
        path for path in directory.glob(RULE_FILE_GLOB) if path.name not in RULE_FILE_NAMES
    )
    files.extend(extension_files)
    return files
