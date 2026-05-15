from __future__ import annotations

import re


_METADATA_RE = re.compile(
    r"\A(?P<prefix>(?:#![^\n]*\n)?)# /// script\n(?P<body>.*?)# ///\n?",
    re.DOTALL,
)
_DEPENDENCIES_RE = re.compile(r"dependencies\s*=\s*\[(?P<body>.*?)\]", re.DOTALL)
_STRING_RE = re.compile(r"""["'](?P<value>[^"']+)["']""")


def ensure_dependency(code: str, dependency: str, package_name: str) -> str:
    """Return code whose PEP 723 script metadata includes dependency."""
    match = _METADATA_RE.match(code)
    if not match:
        return _metadata_block([dependency]) + code

    prefix = match.group("prefix")
    body = match.group("body")
    rest = code[match.end() :]
    dependencies_match = _DEPENDENCIES_RE.search(body)
    if not dependencies_match:
        new_body = body.rstrip() + "\n# dependencies = [\n#   " + repr(dependency) + ",\n# ]\n"
        return f"{prefix}# /// script\n{new_body}# ///\n{rest}"

    dependencies_body = dependencies_match.group("body")
    deps = _STRING_RE.findall(dependencies_body)
    if any(_dependency_name(dep) == package_name for dep in deps):
        return code

    insertion = f"#   {dependency!r},\n"
    insert_at = dependencies_match.start("body")
    new_body = body[:insert_at] + "\n" + insertion + body[insert_at:]
    return f"{prefix}# /// script\n{new_body}# ///\n{rest}"


def _metadata_block(dependencies: list[str]) -> str:
    lines = ["# /// script", "# dependencies = ["]
    lines.extend(f"#   {dep!r}," for dep in dependencies)
    lines.extend(["# ]", "# ///", ""])
    return "\n".join(lines)


def _dependency_name(requirement: str) -> str:
    requirement = requirement.strip()
    for marker in (" @ ", "==", ">=", "<=", "~=", "!=", ">", "<", "["):
        if marker in requirement:
            return requirement.split(marker, 1)[0].strip().lower().replace("_", "-")
    return requirement.strip().lower().replace("_", "-")
