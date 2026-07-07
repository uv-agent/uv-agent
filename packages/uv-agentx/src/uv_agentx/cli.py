from __future__ import annotations

import argparse
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TextIO

from uv_agentx import __version__

PYPI_SIMPLE_BASE = "https://pypi.org/simple"
DEFAULT_PYPI_TIMEOUT_S = 3.0
PACKAGE_NAME_RE = re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)(.*)$")
DIRECT_PREFIXES = (
    "http://",
    "https://",
    "git+",
    "hg+",
    "svn+",
    "bzr+",
    "file:",
)
VERSION_TOKENS = ("===", "==", "~=", "!=", ">=", "<=", ">", "<")
INTERRUPTED_RETURN_CODE = 128 + int(signal.SIGINT)


class UserError(Exception):
    """A user-facing launcher error."""


@dataclass(frozen=True)
class PluginRequirement:
    requirement: str
    refresh_package: str | None = None


class PypiProjectResolver:
    def __init__(
        self,
        *,
        timeout_s: float = DEFAULT_PYPI_TIMEOUT_S,
        stderr: TextIO = sys.stderr,
        enabled: bool = True,
    ) -> None:
        self.timeout_s = timeout_s
        self.stderr = stderr
        self.enabled = enabled
        self._cache: dict[str, bool | None] = {}
        self._warned = False

    def exists(self, project_name: str) -> bool | None:
        normalized = normalize_project_name(project_name)
        if not self.enabled:
            return None
        cached = self._cache.get(normalized)
        if normalized in self._cache:
            return cached
        url = f"{PYPI_SIMPLE_BASE}/{normalized}/"
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/vnd.pypi.simple.v1+json, text/html"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                exists = 200 <= response.status < 300
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                exists = False
            else:
                exists = None
                self._warn_once(f"PyPI lookup for {normalized!r} returned HTTP {exc.code}; letting uv resolve it")
        except OSError as exc:
            exists = None
            self._warn_once(f"PyPI lookup failed ({exc}); letting uv resolve plugin names")
        self._cache[normalized] = exists
        return exists

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        print(f"uv-agentx: warning: {message}", file=self.stderr)


def normalize_project_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def split_latest_marker(spec: str) -> tuple[str, bool]:
    stripped = spec.strip()
    if stripped.lower().endswith("@latest"):
        without = stripped[: -len("@latest")].strip()
        if not without:
            raise UserError("@latest must follow a plugin package name")
        return without, True
    return stripped, False


def is_direct_or_path(spec: str) -> bool:
    stripped = spec.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    if lowered.startswith(DIRECT_PREFIXES):
        return True
    if " @ " in stripped:
        return True
    if stripped.startswith((".", "/", "\\", "~")):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", stripped):
        return True
    return False


def parse_requirement_name(spec: str) -> tuple[str, str] | None:
    if is_direct_or_path(spec):
        return None
    match = PACKAGE_NAME_RE.match(spec.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def replace_requirement_name(spec: str, new_name: str) -> str:
    parsed = parse_requirement_name(spec)
    if parsed is None:
        return spec.strip()
    _, rest = parsed
    return f"{new_name}{rest}"


def is_unpinned_requirement(spec: str) -> bool:
    if is_direct_or_path(spec):
        return False
    parsed = parse_requirement_name(spec)
    if parsed is None:
        return False
    _, rest = parsed
    stripped = rest.strip()
    if not stripped:
        return True
    if re.fullmatch(r"\[[^\]]+\]", stripped):
        return True
    return not any(token in stripped for token in VERSION_TOKENS)


def resolve_raw_plugin(spec: str, *, latest: bool) -> PluginRequirement:
    requirement, latest_marker = split_latest_marker(spec)
    parsed = parse_requirement_name(requirement)
    if latest_marker and parsed is None:
        raise UserError(f"@latest is only supported for package-like plugin requirements: {spec!r}")
    refresh = latest_marker or (latest and is_unpinned_requirement(requirement))
    refresh_package = normalize_project_name(parsed[0]) if refresh and parsed is not None else None
    return PluginRequirement(requirement=requirement, refresh_package=refresh_package)


def resolve_plugin(
    spec: str,
    *,
    latest: bool,
    resolver: PypiProjectResolver,
    stderr: TextIO,
) -> PluginRequirement:
    requirement, latest_marker = split_latest_marker(spec)
    parsed = parse_requirement_name(requirement)
    if parsed is None:
        return resolve_raw_plugin(spec, latest=latest)

    original_name, _ = parsed
    original_project = normalize_project_name(original_name)
    refresh = latest_marker or (latest and is_unpinned_requirement(requirement))

    if original_project.startswith("uv-agent-"):
        selected_project = original_project
        selected_requirement = replace_requirement_name(requirement, selected_project)
        exists = resolver.exists(selected_project)
        if exists is False:
            raise UserError(f"plugin package {selected_project!r} was not found on PyPI")
    else:
        prefixed_project = f"uv-agent-{original_project}"
        prefixed_exists = resolver.exists(prefixed_project)
        if prefixed_exists is True:
            selected_project = prefixed_project
            selected_requirement = replace_requirement_name(requirement, selected_project)
        elif prefixed_exists is False:
            original_exists = resolver.exists(original_project)
            if original_exists is True:
                selected_project = original_project
                selected_requirement = replace_requirement_name(requirement, selected_project)
                print(
                    f"uv-agentx: {prefixed_project} was not found on PyPI; retrying as {original_project}",
                    file=stderr,
                )
            elif original_exists is False:
                raise UserError(
                    "plugin package was not found on PyPI: "
                    f"tried {prefixed_project!r} and {original_project!r}"
                )
            else:
                selected_project = prefixed_project
                selected_requirement = replace_requirement_name(requirement, selected_project)
        else:
            selected_project = prefixed_project
            selected_requirement = replace_requirement_name(requirement, selected_project)

    return PluginRequirement(
        requirement=selected_requirement,
        refresh_package=selected_project if refresh else None,
    )


def build_command(
    *,
    uv_executable: str,
    plugins: list[str],
    raw_plugins: list[str],
    latest: bool,
    agent_args: list[str],
    resolver: PypiProjectResolver,
    stderr: TextIO = sys.stderr,
) -> list[str]:
    requirements: list[PluginRequirement] = []
    for spec in plugins:
        requirements.append(resolve_plugin(spec, latest=latest, resolver=resolver, stderr=stderr))
    for spec in raw_plugins:
        requirements.append(resolve_raw_plugin(spec, latest=latest))

    command = [uv_executable, "tool", "run"]
    for requirement in requirements:
        command.extend(["--with", requirement.requirement])
    for requirement in requirements:
        if requirement.refresh_package is not None:
            command.extend(["--refresh-package", requirement.refresh_package])
    command.append("uv-agent@latest" if latest else "uv-agent")
    command.extend(agent_args)
    return command


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def split_argv(argv: list[str]) -> tuple[list[str], list[str], bool]:
    if "--" not in argv:
        return argv, [], False
    index = argv.index("--")
    return argv[:index], argv[index + 1 :], True


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    launcher_argv, explicit_agent_args, has_separator = split_argv(argv)
    parser = argparse.ArgumentParser(
        prog="uv-agentx",
        description="Launch uv-agent through uv tool run with a concise plugin syntax.",
    )
    parser.add_argument("-p", "--plugin", action="append", default=[], help="uv-agent plugin short name or requirement.")
    parser.add_argument(
        "--raw-plugin",
        action="append",
        default=[],
        help="Plugin requirement to pass through without uv-agent- name expansion.",
    )
    parser.add_argument("--latest", action="store_true", help="Run uv-agent@latest and refresh unpinned plugins.")
    parser.add_argument("--dry-run", action="store_true", help="Print the uv command without executing it.")
    parser.add_argument("--version", action="version", version=f"uv-agentx {__version__}")
    if has_separator:
        return parser.parse_args(launcher_argv), explicit_agent_args
    args, agent_args = parser.parse_known_args(launcher_argv)
    return args, agent_args


def offline_requested() -> bool:
    return os.environ.get("UV_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def run_child_process(command: list[str]) -> int:
    try:
        process = subprocess.Popen(command)
    except KeyboardInterrupt:
        return INTERRUPTED_RETURN_CODE
    try:
        return int(process.wait())
    except KeyboardInterrupt:
        try:
            return int(process.wait())
        except KeyboardInterrupt:
            return INTERRUPTED_RETURN_CODE


def run(argv: list[str], *, stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr) -> int:
    try:
        args, agent_args = parse_args(argv)
        uv_executable = shutil.which("uv")
        if uv_executable is None:
            print("uv-agentx: uv executable was not found on PATH", file=stderr)
            return 127
        resolver = PypiProjectResolver(stderr=stderr, enabled=not offline_requested())
        command = build_command(
            uv_executable=uv_executable,
            plugins=args.plugin,
            raw_plugins=args.raw_plugin,
            latest=args.latest,
            agent_args=agent_args,
            resolver=resolver,
            stderr=stderr,
        )
        if args.dry_run:
            print(format_command(command), file=stdout)
            return 0
        return run_child_process(command)
    except UserError as exc:
        print(f"uv-agentx: {exc}", file=stderr)
        return 2


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))
