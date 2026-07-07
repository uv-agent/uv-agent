from __future__ import annotations

import io

import pytest

from uv_agentx.cli import PypiProjectResolver, UserError, build_command, parse_args, resolve_plugin, resolve_raw_plugin, run


class FakeResolver(PypiProjectResolver):
    def __init__(self, projects: set[str], unknown: set[str] | None = None) -> None:
        super().__init__(enabled=False)
        self.projects = projects
        self.unknown = unknown or set()
        self.queries: list[str] = []

    def exists(self, project_name: str) -> bool | None:
        self.queries.append(project_name)
        if project_name in self.unknown:
            return None
        return project_name in self.projects


def test_plugin_short_name_prefers_uv_agent_package() -> None:
    stderr = io.StringIO()
    resolver = FakeResolver({"uv-agent-auth-code"})

    requirement = resolve_plugin("auth-code", latest=False, resolver=resolver, stderr=stderr)

    assert requirement.requirement == "uv-agent-auth-code"
    assert requirement.refresh_package is None
    assert resolver.queries == ["uv-agent-auth-code"]
    assert stderr.getvalue() == ""


def test_plugin_short_name_falls_back_to_original_with_log() -> None:
    stderr = io.StringIO()
    resolver = FakeResolver({"auth-code"})

    requirement = resolve_plugin("auth-code", latest=False, resolver=resolver, stderr=stderr)

    assert requirement.requirement == "auth-code"
    assert "retrying as auth-code" in stderr.getvalue()
    assert resolver.queries == ["uv-agent-auth-code", "auth-code"]


def test_plugin_short_name_errors_when_no_candidate_exists() -> None:
    with pytest.raises(UserError, match="tried 'uv-agent-auth-code' and 'auth-code'"):
        resolve_plugin("auth-code", latest=False, resolver=FakeResolver(set()), stderr=io.StringIO())


def test_network_unknown_lets_uv_resolve_prefixed_name() -> None:
    requirement = resolve_plugin(
        "auth-code",
        latest=False,
        resolver=FakeResolver(set(), unknown={"uv-agent-auth-code"}),
        stderr=io.StringIO(),
    )

    assert requirement.requirement == "uv-agent-auth-code"


def test_latest_refreshes_unpinned_plugin_and_agent() -> None:
    command = build_command(
        uv_executable="uv",
        plugins=["auth-code"],
        raw_plugins=[],
        latest=True,
        agent_args=["daemon", "--replace"],
        resolver=FakeResolver({"uv-agent-auth-code"}),
        stderr=io.StringIO(),
    )

    assert command == [
        "uv",
        "tool",
        "run",
        "--with",
        "uv-agent-auth-code",
        "--refresh-package",
        "uv-agent-auth-code",
        "uv-agent@latest",
        "daemon",
        "--replace",
    ]


def test_plugin_level_latest_does_not_force_agent_latest() -> None:
    command = build_command(
        uv_executable="uv",
        plugins=["auth-code@latest"],
        raw_plugins=[],
        latest=False,
        agent_args=[],
        resolver=FakeResolver({"uv-agent-auth-code"}),
        stderr=io.StringIO(),
    )

    assert command == [
        "uv",
        "tool",
        "run",
        "--with",
        "uv-agent-auth-code",
        "--refresh-package",
        "uv-agent-auth-code",
        "uv-agent",
    ]


def test_pinned_plugin_keeps_constraint_and_does_not_refresh() -> None:
    command = build_command(
        uv_executable="uv",
        plugins=["auth-code==1.2.0"],
        raw_plugins=[],
        latest=True,
        agent_args=[],
        resolver=FakeResolver({"uv-agent-auth-code"}),
        stderr=io.StringIO(),
    )

    assert command == ["uv", "tool", "run", "--with", "uv-agent-auth-code==1.2.0", "uv-agent@latest"]


def test_raw_plugin_skips_name_expansion() -> None:
    requirement = resolve_raw_plugin("auth-code@latest", latest=False)

    assert requirement.requirement == "auth-code"
    assert requirement.refresh_package == "auth-code"


def test_parse_separator_sends_remaining_args_to_uv_agent() -> None:
    args, agent_args = parse_args(["--latest", "-p", "auth-code", "--", "daemon", "--replace"])

    assert args.latest is True
    assert args.plugin == ["auth-code"]
    assert agent_args == ["daemon", "--replace"]


def test_parse_without_separator_passes_unknown_args_to_uv_agent() -> None:
    args, agent_args = parse_args(["-p", "auth-code", "--log-level", "DEBUG"])

    assert args.plugin == ["auth-code"]
    assert agent_args == ["--log-level", "DEBUG"]


def test_run_waits_for_child_after_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.wait_calls = 0

        def wait(self) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise KeyboardInterrupt
            return 0

    commands: list[list[str]] = []

    def fake_popen(command: list[str]) -> FakeProcess:
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr("uv_agentx.cli.shutil.which", lambda _name: "uv")
    monkeypatch.setattr("uv_agentx.cli.subprocess.Popen", fake_popen)

    assert run(["daemon"], stderr=io.StringIO()) == 0
    assert commands == [["uv", "tool", "run", "uv-agent", "daemon"]]


def test_run_returns_interrupt_code_after_repeated_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        def wait(self) -> int:
            raise KeyboardInterrupt

    monkeypatch.setattr("uv_agentx.cli.shutil.which", lambda _name: "uv")
    monkeypatch.setattr("uv_agentx.cli.subprocess.Popen", lambda _command: FakeProcess())

    assert run(["daemon"], stderr=io.StringIO()) == 130
