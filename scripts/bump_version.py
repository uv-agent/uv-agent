from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
PACKAGE_INIT = PROJECT_ROOT / "src" / "uv_agent" / "__init__.py"
UV_LOCK = PROJECT_ROOT / "uv.lock"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?$")


def project_version() -> str:
    """Return the version declared in pyproject.toml."""
    with PYPROJECT.open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def replace_one(text: str, pattern: str, replacement: str, path: Path) -> str:
    """Replace exactly one regex match so stale or duplicated versions fail loudly."""
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Expected exactly one version match in {path}")
    return updated


def read_text(path: Path) -> str:
    """Read text without newline translation so edited files keep their style."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return file.read()


def write_text_preserving_newlines(path: Path, before: str, after: str) -> None:
    if after == before:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(after)


def update_pyproject(version: str) -> None:
    text = read_text(PYPROJECT)
    updated = replace_one(
        text,
        r'^(version = ")([^"]+)(")(\r?)$',
        rf"\g<1>{version}\g<3>\g<4>",
        PYPROJECT,
    )
    write_text_preserving_newlines(PYPROJECT, text, updated)


def update_package_init(version: str) -> None:
    text = read_text(PACKAGE_INIT)
    updated = replace_one(
        text,
        r'^(__version__ = ")([^"]+)(")(\r?)$',
        rf"\g<1>{version}\g<3>\g<4>",
        PACKAGE_INIT,
    )
    write_text_preserving_newlines(PACKAGE_INIT, text, updated)


def run_uv_lock() -> None:
    subprocess.run(["uv", "lock"], cwd=PROJECT_ROOT, check=True)


def assert_versions_match() -> None:
    pyproject = project_version()

    namespace: dict[str, str] = {}
    exec(read_text(PACKAGE_INIT), namespace)
    package = namespace.get("__version__")
    if package != pyproject:
        raise SystemExit(
            f"{PACKAGE_INIT.relative_to(PROJECT_ROOT)} has __version__={package!r}, "
            f"but pyproject.toml has version={pyproject!r}"
        )

    lock_text = read_text(UV_LOCK)
    lock_pattern = re.compile(
        r'\[\[package\]\]\r?\nname = "uv-agent"\r?\nversion = "([^"]+)"',
        re.MULTILINE,
    )
    lock_versions = lock_pattern.findall(lock_text)
    if lock_versions != [pyproject]:
        raise SystemExit(
            f"uv.lock has uv-agent versions {lock_versions!r}, "
            f"but pyproject.toml has version={pyproject!r}. Run `uv lock`."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Update the uv-agent release version in pyproject.toml, "
            "src/uv_agent/__init__.py, and uv.lock."
        )
    )
    parser.add_argument(
        "version",
        nargs="?",
        help="Version to write. Omit with --check to only verify existing files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that all version files already match.",
    )
    args = parser.parse_args(argv)

    if args.check:
        if args.version is not None:
            parser.error("--check does not accept a version argument")
        assert_versions_match()
        return 0

    if args.version is None:
        parser.error("version is required unless --check is used")
    if not VERSION_RE.fullmatch(args.version):
        parser.error(
            "version must be a PEP 440-style release such as 1.2.3, "
            "1.2.3rc1, 1.2.3.post1, or 1.2.3.dev1"
        )

    update_pyproject(args.version)
    update_package_init(args.version)
    run_uv_lock()
    assert_versions_match()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
