from __future__ import annotations

import asyncio
import inspect
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .helpers import validate_json_value, validate_payload

_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")
_DOTTED_NAME_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
RuntimeTransport = Literal["rpc", "local_module"]
ActionHandler = Callable[..., Any]
CommandHandler = Callable[..., Any]



@dataclass(frozen=True)
class CommandResult:
    """UI-neutral result returned by plugin slash commands."""

    actions: tuple[Any, ...] = ()


@dataclass(frozen=True)
class TranscriptAction:
    kind: Literal["event", "error"]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SetComposerAction:
    text: str


@dataclass(frozen=True)
class OpenPickerAction:
    picker_id: str
    query: str = ""


@dataclass(frozen=True)
class PickerItem:
    value: str
    description: str = ""
    id: str = ""
    kind: str = "mention"
    meta: str = ""

@dataclass(frozen=True)
class RuntimeFunctionSpec:
    namespace: str
    name: str
    plugin: str
    doc: str
    schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    fn: Callable[..., Any] | None = None
    timeout_s: float | None = None

    @property
    def full_name(self) -> str:
        return f"{self.namespace}.{self.name}"


@dataclass(frozen=True)
class RuntimeNamespaceSpec:
    namespace: str
    plugin: str
    doc: str = ""
    transport: RuntimeTransport = "rpc"
    module: str | None = None
    functions: tuple[RuntimeFunctionSpec, ...] = ()


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    plugin: str
    handler: ActionHandler
    doc: str = ""
    schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})


@dataclass(frozen=True)
class CommandSpec:
    name: str
    plugin: str
    handler: CommandHandler
    description: str = ""
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class StatusItem:
    plugin: str
    id: str
    label: str
    value: str = ""
    priority: int = 100
    style: str = ""


@dataclass(frozen=True)
class Panel:
    plugin: str
    id: str
    title: str
    kind: Literal["text", "list", "table", "todo", "progress"]
    body: Any
    priority: int = 100


@dataclass(frozen=True)
class PickerSource:
    plugin: str
    id: str
    title: str
    provider: Callable[..., Any]
    trigger: str = ""


@dataclass(frozen=True)
class TranscriptEventSpec:
    plugin: str
    type: str
    renderer: Callable[..., Any]


class RuntimeNamespaceRegistry:
    """Thread-safe registry for runtime helper namespaces.

    The runtime package asks the host to resolve either a namespace (``goal``) or
    a concrete function (``goal.state``).  The registry keeps the public namespace
    globally unique while allowing each namespace to contain several functions.
    """

    def __init__(self, *, reserved: Iterable[str] = ()) -> None:
        self._lock = threading.RLock()
        self._reserved = set(reserved)
        self._namespaces: dict[str, RuntimeNamespaceSpec] = {}

    def register_namespace(
        self,
        *,
        plugin: str,
        namespace: str,
        doc: str = "",
        functions: Mapping[str, Callable[..., Any] | RuntimeFunctionSpec] | Iterable[RuntimeFunctionSpec] = (),
        schemas: Mapping[str, dict[str, Any]] | None = None,
        docs: Mapping[str, str] | None = None,
        transport: RuntimeTransport = "rpc",
        module: str | None = None,
    ) -> RuntimeNamespaceSpec:
        namespace = _validate_name(namespace, label="runtime namespace")
        docs = dict(docs or {})
        schemas = dict(schemas or {})
        specs: list[RuntimeFunctionSpec] = []
        if isinstance(functions, Mapping):
            for name, fn_or_spec in functions.items():
                if isinstance(fn_or_spec, RuntimeFunctionSpec):
                    spec = fn_or_spec
                else:
                    spec = RuntimeFunctionSpec(
                        namespace=namespace,
                        name=_validate_name(str(name), label="runtime function"),
                        plugin=plugin,
                        fn=fn_or_spec,
                        doc=str(docs.get(str(name)) or "").strip(),
                        schema=dict(schemas.get(str(name)) or {"type": "object"}),
                    )
                specs.append(spec)
        else:
            specs.extend(functions)
        normalized = tuple(
            RuntimeFunctionSpec(
                namespace=namespace,
                name=_validate_name(spec.name, label="runtime function"),
                plugin=plugin,
                fn=spec.fn,
                doc=str(spec.doc or docs.get(spec.name) or "").strip(),
                schema=dict(spec.schema or schemas.get(spec.name) or {"type": "object"}),
                timeout_s=spec.timeout_s,
            )
            for spec in specs
        )
        for spec in normalized:
            if spec.schema.get("type") != "object":
                raise ValueError(f"Runtime helper {spec.full_name!r} requires an object JSON schema")
            if transport == "rpc" and spec.fn is None:
                raise ValueError(f"RPC runtime helper {spec.full_name!r} requires a callable")
        entry = RuntimeNamespaceSpec(
            namespace=namespace,
            plugin=plugin,
            doc=str(doc or "").strip(),
            transport=transport,
            module=module,
            functions=normalized,
        )
        if namespace in self._reserved:
            raise ValueError(f"Runtime namespace {namespace!r} is reserved by core")
        with self._lock:
            existing = self._namespaces.get(namespace)
            if existing is not None:
                raise ValueError(
                    f"Runtime namespace {namespace!r} already registered by {existing.plugin}"
                )
            self._namespaces[namespace] = entry
        return entry

    def list_namespaces(self) -> list[RuntimeNamespaceSpec]:
        with self._lock:
            return sorted(self._namespaces.values(), key=lambda item: item.namespace)

    def namespace(self, name: str) -> RuntimeNamespaceSpec | None:
        with self._lock:
            return self._namespaces.get(name)

    def function(self, full_name: str) -> RuntimeFunctionSpec | None:
        if "." not in full_name:
            return None
        namespace, _, function = full_name.partition(".")
        entry = self.namespace(namespace)
        if entry is None:
            return None
        for spec in entry.functions:
            if spec.name == function:
                return spec
        return None

    def resolve_payload(self, name: str) -> dict[str, Any]:
        if not isinstance(name, str) or not name:
            return {"found": False, "name": str(name)}
        if "." not in name:
            namespace = self.namespace(name)
            if namespace is None:
                return {"found": False, "name": name}
            return {
                "found": True,
                "kind": "namespace",
                "name": namespace.namespace,
                "plugin": namespace.plugin,
                "doc": namespace.doc,
                "transport": namespace.transport,
                "module": namespace.module,
                "functions": [
                    {
                        "name": fn.name,
                        "full_name": fn.full_name,
                        "doc": fn.doc,
                        "schema": fn.schema,
                    }
                    for fn in namespace.functions
                ],
            }
        function = self.function(name)
        if function is None:
            return {"found": False, "name": name}
        namespace = self.namespace(function.namespace)
        return {
            "found": True,
            "kind": "function",
            "name": function.full_name,
            "namespace": function.namespace,
            "function": function.name,
            "plugin": function.plugin,
            "doc": function.doc,
            "schema": function.schema,
            "transport": namespace.transport if namespace else "rpc",
            "module": namespace.module if namespace else None,
        }

    async def call(self, full_name: str, payload: dict[str, Any], *, context: Any = None) -> Any:
        spec = self.function(full_name)
        if spec is None or spec.fn is None:
            raise LookupError(f"Unknown runtime helper: {full_name}")
        validate_payload(payload, spec.schema)
        kwargs: dict[str, Any] = {"payload": payload}
        if _accepts_context(spec.fn):
            kwargs["context"] = context
        result = spec.fn(**kwargs)
        if inspect.isawaitable(result):
            if spec.timeout_s is not None:
                result = await asyncio.wait_for(result, timeout=spec.timeout_s)
            else:
                result = await result
        validate_json_value(result, label=f"runtime helper {full_name!r} return value")
        return result


class ActionRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._actions: dict[str, ActionSpec] = {}

    def register(self, *, plugin: str, action_id: str, handler: ActionHandler, doc: str = "", schema: dict[str, Any] | None = None) -> ActionSpec:
        action_id = _validate_dotted(action_id, label="action id")
        if not callable(handler):
            raise TypeError(f"Action handler is not callable: {action_id}")
        spec = ActionSpec(action_id=action_id, plugin=plugin, handler=handler, doc=doc, schema=dict(schema or {"type": "object"}))
        with self._lock:
            if action_id in self._actions:
                raise ValueError(f"Action already registered: {action_id}")
            self._actions[action_id] = spec
        return spec

    def get(self, action_id: str) -> ActionSpec | None:
        with self._lock:
            return self._actions.get(action_id)

    def list(self) -> list[ActionSpec]:
        with self._lock:
            return sorted(self._actions.values(), key=lambda item: item.action_id)

    async def call(self, action_id: str, payload: dict[str, Any] | None = None, *, context: Any = None) -> Any:
        spec = self.get(action_id)
        if spec is None:
            raise LookupError(f"Unknown action: {action_id}")
        data = dict(payload or {})
        validate_payload(data, spec.schema)
        kwargs: dict[str, Any] = {"payload": data}
        if _accepts_context(spec.handler):
            kwargs["context"] = context
        result = spec.handler(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        validate_json_value(result, label=f"action {action_id!r} return value")
        return result


class CommandRegistry:
    def __init__(self, *, reserved: Iterable[str] = ()) -> None:
        self._lock = threading.RLock()
        self._reserved = {name if name.startswith("/") else f"/{name}" for name in reserved}
        self._commands: dict[str, CommandSpec] = {}

    def register(self, *, plugin: str, name: str, handler: CommandHandler, description: str = "", aliases: Iterable[str] = ()) -> CommandSpec:
        command = _normalize_command(name)
        normalized_aliases = tuple(_normalize_command(alias) for alias in aliases)
        if command in self._reserved:
            raise ValueError(f"Command {command!r} is reserved by core")
        if not callable(handler):
            raise TypeError(f"Command handler is not callable: {command}")
        spec = CommandSpec(name=command, plugin=plugin, handler=handler, description=description, aliases=normalized_aliases)
        with self._lock:
            for key in (command, *normalized_aliases):
                if key in self._commands:
                    raise ValueError(f"Command already registered: {key}")
            self._commands[command] = spec
            for alias in normalized_aliases:
                self._commands[alias] = spec
        return spec

    def get(self, name: str) -> CommandSpec | None:
        with self._lock:
            return self._commands.get(_normalize_command(name))

    def list(self) -> list[CommandSpec]:
        with self._lock:
            unique = {id(spec): spec for spec in self._commands.values()}
        return sorted(unique.values(), key=lambda item: item.name)


class UiRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._status: list[StatusItem] = []
        self._panels: list[Panel] = []
        self._pickers: dict[str, PickerSource] = {}
        self._transcript: dict[str, TranscriptEventSpec] = {}

    def register_status_item(self, item: StatusItem) -> None:
        with self._lock:
            self._status.append(item)

    def status_items(self) -> list[StatusItem]:
        with self._lock:
            return sorted(self._status, key=lambda item: (item.priority, item.plugin, item.id))

    def register_panel(self, panel: Panel) -> None:
        with self._lock:
            self._panels.append(panel)

    def panels(self) -> list[Panel]:
        with self._lock:
            return sorted(self._panels, key=lambda item: (item.priority, item.plugin, item.id))

    def register_picker(self, picker: PickerSource) -> None:
        with self._lock:
            if picker.id in self._pickers:
                raise ValueError(f"Picker source already registered: {picker.id}")
            self._pickers[picker.id] = picker

    def picker(self, picker_id: str) -> PickerSource | None:
        with self._lock:
            return self._pickers.get(picker_id)

    def pickers(self) -> list[PickerSource]:
        with self._lock:
            return sorted(self._pickers.values(), key=lambda item: (item.plugin, item.id))

    def picker_items(self, picker_id: str, query: str = "") -> list[PickerItem]:
        source = self.picker(picker_id)
        if source is None:
            return []
        result = source.provider(query=query)
        return _normalize_picker_items(result)

    def register_transcript_event(self, spec: TranscriptEventSpec) -> None:
        with self._lock:
            if spec.type in self._transcript:
                raise ValueError(f"Transcript event already registered: {spec.type}")
            self._transcript[spec.type] = spec


def _validate_name(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not _NAME_RE.match(text):
        raise ValueError(f"Invalid {label}: {value!r}")
    return text


def _validate_dotted(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not _DOTTED_NAME_RE.match(text):
        raise ValueError(f"Invalid {label}: {value!r}")
    return text


def _normalize_command(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Command name must not be empty")
    if not text.startswith("/"):
        text = "/" + text
    parts = text.split("/")
    if parts[0] != "" or any(part == "" for part in parts[1:]):
        raise ValueError(f"Invalid command name: {value!r}")
    return text


def _accepts_context(fn: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return "context" in signature.parameters
