from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from uv_agent.prompts import (
    PROJECT_RULE_CONTEXT_PATH_ATTR_TEMPLATE,
    PROJECT_RULE_ENTRY_CLOSE,
    PROJECT_RULE_ENTRY_OPEN_TEMPLATE,
    PROJECT_RULE_FILE_ATTR_TEMPLATE,
    PROJECT_RULE_INDEX_CLOSE,
    PROJECT_RULE_INDEX_DEPTH_LIMIT_REACHED,
    PROJECT_RULE_INDEX_ENTRY_LIMIT_REACHED,
    PROJECT_RULE_INDEX_HEADER,
    PROJECT_RULE_INDEX_MAX_ENTRIES_TEMPLATE,
    PROJECT_RULE_INDEX_OPEN,
    PROJECT_RULE_INDEX_SCAN_DEPTH_TEMPLATE,
    PROJECT_RULE_INDEX_TRUNCATED_TEMPLATE,
    PROJECT_RULE_OMITTED_FILES_ATTR_TEMPLATE,
    PROJECT_RULE_TRUNCATED_ATTR,
    PROJECT_RULE_TRUNCATED_SUFFIX,
    PROJECT_RULES_CLOSE_TEMPLATE,
    PROJECT_RULES_DEFAULT_TAG,
    PROJECT_RULES_LOADED_HEADER,
    PROJECT_RULES_OPEN_TEMPLATE,
)

logger = logging.getLogger(__name__)


RULE_FILE_NAMES = ("AGENTS.md",)
RULE_FILE_GLOB = "AGENTS.*.md"
DEFAULT_MAX_CHARS_PER_FILE = 12_000
DEFAULT_MAX_TOTAL_CHARS = 36_000
DEFAULT_RULE_INDEX_MAX_DEPTH = 2
DEFAULT_RULE_INDEX_MAX_ENTRIES = 50
SKIPPED_RULE_INDEX_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".uv-agent",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


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

    def render(
        self,
        *,
        root: Path | None = None,
        base_path: Path | None = None,
        context_path: str | None = None,
        heading: str = PROJECT_RULES_DEFAULT_TAG,
    ) -> str:
        """Render loaded rules as a compact context block for model input."""
        if not self.rules:
            return ""
        attrs = []
        if context_path is not None:
            attrs.append(PROJECT_RULE_CONTEXT_PATH_ATTR_TEMPLATE.format(path=xml_attr(context_path)))
        if self.truncated:
            attrs.append(PROJECT_RULE_TRUNCATED_ATTR)
        if self.omitted_files:
            attrs.append(PROJECT_RULE_OMITTED_FILES_ATTR_TEMPLATE.format(count=self.omitted_files))
        open_tag = PROJECT_RULES_OPEN_TEMPLATE.format(heading=heading, attrs=(" " + " ".join(attrs)) if attrs else "")
        lines = [
            open_tag,
            PROJECT_RULES_LOADED_HEADER,
        ]
        for rule in self.rules:
            rel = display_path(rule.path, root=base_path or root)
            rule_attrs = [PROJECT_RULE_FILE_ATTR_TEMPLATE.format(file=xml_attr(rel))]
            if rule.truncated:
                rule_attrs.append(PROJECT_RULE_TRUNCATED_ATTR)
            lines.extend(
                [
                    PROJECT_RULE_ENTRY_OPEN_TEMPLATE.format(attrs=" ".join(rule_attrs)),
                    rule.text.strip(),
                    PROJECT_RULE_ENTRY_CLOSE,
                ]
            )
        lines.append(PROJECT_RULES_CLOSE_TEMPLATE.format(heading=heading))
        return "\n".join(lines)


@dataclass(frozen=True)
class WorkspaceRuleIndex:
    """A lightweight listing of rule files without their contents."""

    root: Path
    paths: list[Path]
    max_depth: int
    max_entries: int
    truncated_entries: bool = False
    depth_limited: bool = False

    def render(self, *, label: str = "workspace") -> str:
        if not self.paths:
            return ""
        lines = [
            PROJECT_RULE_INDEX_OPEN,
            PROJECT_RULE_INDEX_HEADER.format(label=label),
        ]
        for path in self.paths:
            lines.append(f"- {display_path(path, root=self.root)}")
        truncated = self.truncated_entries or self.depth_limited
        lines.extend(
            [
                "",
                PROJECT_RULE_INDEX_SCAN_DEPTH_TEMPLATE.format(depth=self.max_depth),
                PROJECT_RULE_INDEX_MAX_ENTRIES_TEMPLATE.format(max_entries=self.max_entries),
                PROJECT_RULE_INDEX_TRUNCATED_TEMPLATE.format(truncated=str(truncated).lower()),
            ]
        )
        if self.depth_limited:
            lines.append(PROJECT_RULE_INDEX_DEPTH_LIMIT_REACHED)
        if self.truncated_entries:
            lines.append(PROJECT_RULE_INDEX_ENTRY_LIMIT_REACHED)
        lines.append(PROJECT_RULE_INDEX_CLOSE)
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
        except OSError as exc:
            logger.debug("Project rule read failed path=%s error=%s", resolved, exc)
            continue
        remaining = max_total_chars - total
        limit = min(max_chars_per_file, remaining)
        truncated_file = len(text) > limit
        clipped = text[:limit]
        total += len(clipped)
        if truncated_file:
            clipped = clipped.rstrip() + PROJECT_RULE_TRUNCATED_SUFFIX
            truncated_context = True
            logger.debug("Project rule truncated path=%s limit=%d", resolved, limit)
        rules.append(ProjectRule(path=resolved, scope=scope, text=clipped, truncated=truncated_file))

    logger.debug(
        "Project rules loaded root=%s count=%d truncated=%s omitted=%d",
        root,
        len(rules),
        truncated_context,
        omitted,
    )
    return ProjectRuleContext(rules=rules, truncated=truncated_context, omitted_files=omitted)


def load_directory_rules(
    directory: Path,
    *,
    root: Path | None = None,
    max_chars_per_file: int = DEFAULT_MAX_CHARS_PER_FILE,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> ProjectRuleContext:
    """Load only rule files located directly in one directory."""
    resolved_dir = directory.resolve()
    total = 0
    omitted = 0
    truncated_context = False
    rules: list[ProjectRule] = []

    for path in rule_files_in_dir(resolved_dir):
        if total >= max_total_chars:
            omitted += 1
            truncated_context = True
            continue
        try:
            text = path.resolve().read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug("Directory rule read failed path=%s error=%s", path, exc)
            continue
        remaining = max_total_chars - total
        limit = min(max_chars_per_file, remaining)
        truncated_file = len(text) > limit
        clipped = text[:limit]
        total += len(clipped)
        if truncated_file:
            clipped = clipped.rstrip() + PROJECT_RULE_TRUNCATED_SUFFIX
            truncated_context = True
            logger.debug("Directory rule truncated path=%s limit=%d", path, limit)
        rules.append(
            ProjectRule(
                path=path.resolve(),
                scope=display_path(resolved_dir, root=root),
                text=clipped,
                truncated=truncated_file,
            )
        )

    logger.debug(
        "Directory rules loaded directory=%s count=%d truncated=%s omitted=%d",
        resolved_dir,
        len(rules),
        truncated_context,
        omitted,
    )
    return ProjectRuleContext(rules=rules, truncated=truncated_context, omitted_files=omitted)


def discover_workspace_rule_index(
    workspace_root: Path,
    *,
    max_depth: int = DEFAULT_RULE_INDEX_MAX_DEPTH,
    max_entries: int = DEFAULT_RULE_INDEX_MAX_ENTRIES,
) -> WorkspaceRuleIndex:
    """Return a bounded recursive index of rule files below the workspace."""
    root = workspace_root.resolve()
    paths: list[Path] = []
    truncated_entries = False
    depth_limited = False
    queue: deque[tuple[Path, int]] = deque([(root, 0)])

    while queue:
        directory, depth = queue.popleft()
        for path in rule_files_in_dir(directory):
            if len(paths) >= max_entries:
                truncated_entries = True
                break
            paths.append(path.resolve())
        if truncated_entries:
            break
        try:
            children = sorted(
                child
                for child in directory.iterdir()
                if child.is_dir() and not should_skip_rule_index_dir(child)
            )
        except OSError:
            logger.debug("Workspace rule index directory read failed path=%s", directory)
            continue
        if depth >= max_depth:
            if children:
                depth_limited = True
            continue
        for child in children:
            queue.append((child, depth + 1))

    index = WorkspaceRuleIndex(
        root=root,
        paths=paths,
        max_depth=max_depth,
        max_entries=max_entries,
        truncated_entries=truncated_entries,
        depth_limited=depth_limited,
    )
    logger.debug(
        "Workspace rule index discovered root=%s count=%d truncated=%s depth_limited=%s",
        root,
        len(paths),
        truncated_entries,
        depth_limited,
    )
    return index


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


def should_skip_rule_index_dir(path: Path) -> bool:
    name = path.name
    if name in SKIPPED_RULE_INDEX_DIRS:
        return True
    return name.startswith(".") and name not in {".agents"}


def display_path(path: Path, *, root: Path | None = None) -> str:
    resolved = path.resolve()
    if root is not None:
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            pass
        else:
            return "." if not relative.parts else relative.as_posix()
    return str(resolved)


def xml_attr(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
