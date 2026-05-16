from __future__ import annotations

import ast
import json
import re
import tomllib


_METADATA_RE = re.compile(
    r"\A(?P<prefix>(?:#![^\n]*\n)?)# /// script\n(?P<body>.*?)# ///\n?",
    re.DOTALL,
)
_DEPENDENCIES_RE = re.compile(r"dependencies\s*=\s*\[(?P<body>.*?)\]", re.DOTALL)
_METADATA_COMMENT_RE = re.compile(r"^# ?(?P<text>.*)$", re.MULTILINE)


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

    deps = _parse_dependencies(body)
    if deps is None:
        deps = _parse_dependency_strings(dependencies_match.group("body"))
    if any(_dependency_name(dep) == package_name for dep in deps):
        return code

    new_body = _add_dependency_to_existing_dependencies(body, dependency, deps)
    return f"{prefix}# /// script\n{new_body}# ///\n{rest}"


def _metadata_block(dependencies: list[str]) -> str:
    lines = ["# /// script", "# dependencies = ["]
    lines.extend(f"#   {_toml_string(dep)}," for dep in dependencies)
    lines.extend(["# ]", "# ///", ""])
    return "\n".join(lines)


def _parse_dependencies(metadata_body: str) -> list[str] | None:
    toml_text = _metadata_comment_replacement(metadata_body)
    try:
        parsed = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError:
        return None
    dependencies = parsed.get("dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        return None
    return dependencies


def _metadata_comment_replacement(metadata_body: str) -> str:
    return _METADATA_COMMENT_RE.sub(lambda match: match.group("text"), metadata_body)


def _parse_dependency_strings(dependencies_body: str) -> list[str]:
    quoted = re.finditer(r"""(?P<quote>["'])(?:\\.|(?! (?P=quote) ).)*(?P=quote)""", dependencies_body, re.VERBOSE | re.DOTALL)
    values: list[str] = []
    for match in quoted:
        literal = match.group(0)
        try:
            parsed = ast.literal_eval(literal)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, str):
            values.append(parsed)
    return values


def _add_dependency_to_existing_dependencies(
    metadata_body: str,
    dependency: str,
    existing_dependencies: list[str],
) -> str:
    span = _dependencies_line_span(metadata_body)
    if span is None:
        dependencies_match = _DEPENDENCIES_RE.search(metadata_body)
        if dependencies_match is None:
            return metadata_body
        insert_at = dependencies_match.start("body")
        return metadata_body[:insert_at] + "\n#   " + _toml_string(dependency) + ",\n" + metadata_body[insert_at:]

    lines = metadata_body.splitlines(keepends=True)
    start_line, end_line = span
    if _can_insert_after_opening_line(lines[start_line], start_line, end_line):
        insertion = f"#   {_toml_string(dependency)},\n"
        return "".join(lines[: start_line + 1] + [insertion] + lines[start_line + 1 :])

    replacement = _dependency_assignment_lines([dependency, *existing_dependencies])
    return "".join(lines[:start_line] + replacement + lines[end_line + 1 :])


def _dependency_assignment_lines(dependencies: list[str]) -> list[str]:
    return [
        "# dependencies = [\n",
        *(f"#   {_toml_string(dependency)},\n" for dependency in dependencies),
        "# ]\n",
    ]


def _dependencies_line_span(metadata_body: str) -> tuple[int, int] | None:
    lines = metadata_body.splitlines(keepends=True)
    for index, line in enumerate(lines):
        content = _metadata_line_content(line)
        if re.match(r"^\s*dependencies\s*=", content):
            end = _find_array_end_line(lines, index)
            if end is not None:
                return index, end
            return None
    return None


def _can_insert_after_opening_line(line: str, start_line: int, end_line: int) -> bool:
    if start_line == end_line:
        return False
    content = _metadata_line_content(line)
    bracket_at = content.find("[")
    if bracket_at < 0:
        return False
    return _strip_toml_comment(content[bracket_at + 1 :]).strip() == ""


def _find_array_end_line(lines: list[str], start_line: int) -> int | None:
    depth = 0
    saw_array = False
    quote: str | None = None
    triple_quote = False
    escaped = False
    for line_index in range(start_line, len(lines)):
        content = _metadata_line_content(lines[line_index])
        char_index = 0
        while char_index < len(content):
            char = content[char_index]
            if quote is not None:
                if quote == '"' and not triple_quote and escaped:
                    escaped = False
                elif quote == '"' and not triple_quote and char == "\\":
                    escaped = True
                elif triple_quote and content.startswith(quote * 3, char_index):
                    quote = None
                    triple_quote = False
                    char_index += 2
                elif not triple_quote and char == quote:
                    quote = None
                char_index += 1
                continue

            if char == "#":
                break
            if char in {'"', "'"}:
                if content.startswith(char * 3, char_index):
                    quote = char
                    triple_quote = True
                    char_index += 3
                    continue
                quote = char
                triple_quote = False
                char_index += 1
                continue
            if char == "[":
                depth += 1
                saw_array = True
            elif char == "]" and saw_array:
                depth -= 1
                if depth == 0:
                    return line_index
            char_index += 1
    return None


def _metadata_line_content(line: str) -> str:
    content = line.rstrip("\r\n")
    if content.startswith("# "):
        return content[2:]
    if content.startswith("#"):
        return content[1:]
    return content


def _strip_toml_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if quote is not None:
            if quote == '"' and escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == "#":
            return value[:index]
    return value


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _dependency_name(requirement: str) -> str:
    requirement = requirement.strip()
    for marker in (" @ ", "==", ">=", "<=", "~=", "!=", ">", "<", "["):
        if marker in requirement:
            return requirement.split(marker, 1)[0].strip().lower().replace("_", "-")
    return requirement.strip().lower().replace("_", "-")
