from __future__ import annotations

import asyncio
import inspect
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ResourceKind = Literal["text", "bytes", "path"]
ResourceReader = Callable[..., Any]
_URI_RE = re.compile(r"^[a-z][a-z0-9+.-]*://")


class ResourceNotFoundError(FileNotFoundError):
    pass


class UnknownResourceProviderError(LookupError):
    pass


@dataclass(frozen=True)
class ResourceData:
    uri: str = ""
    kind: ResourceKind = "text"
    text: str | None = None
    data: bytes | None = None
    path: Path | None = None
    mime_type: str = "text/plain; charset=utf-8"
    filename: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"text", "bytes", "path"}:
            raise ValueError(f"Invalid resource kind: {self.kind!r}")
        present = {
            "text": self.text is not None,
            "bytes": self.data is not None,
            "path": self.path is not None,
        }
        if sum(1 for enabled in present.values() if enabled) != 1:
            raise ValueError("ResourceData requires exactly one of text, data, or path")
        if not present[self.kind]:
            raise ValueError(f"ResourceData kind={self.kind!r} must provide the matching payload")
        if self.kind == "bytes" and not isinstance(self.data, bytes):
            raise TypeError("ResourceData data must be bytes")
        if self.kind == "path" and not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ResourceProvider:
    plugin: str
    prefix: str
    read: ResourceReader


class ResourceRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._providers: dict[str, ResourceProvider] = {}

    def register(self, *, plugin: str, prefix: str, read: ResourceReader) -> ResourceProvider:
        normalized = _normalize_prefix(prefix)
        if not callable(read):
            raise TypeError(f"Resource reader is not callable for {prefix!r}")
        provider = ResourceProvider(plugin=plugin, prefix=normalized, read=read)
        with self._lock:
            if normalized in self._providers:
                existing = self._providers[normalized]
                raise ValueError(f"Resource prefix {normalized!r} already registered by {existing.plugin}")
            self._providers[normalized] = provider
        return provider

    def unregister(self, prefix: str) -> None:
        normalized = _normalize_prefix(prefix)
        with self._lock:
            self._providers.pop(normalized, None)

    def provider_for(self, uri: str) -> ResourceProvider:
        if not is_resource_uri(uri):
            raise ValueError(f"Not a resource URI: {uri!r}")
        with self._lock:
            matches = [provider for prefix, provider in self._providers.items() if uri.startswith(prefix)]
        if not matches:
            raise UnknownResourceProviderError(f"No resource provider registered for URI: {uri}")
        return max(matches, key=lambda provider: len(provider.prefix))

    def read(self, uri: str, *, max_bytes: int | None = None, context: Any = None) -> Any:
        provider = self.provider_for(uri)
        kwargs: dict[str, Any] = {"uri": uri}
        if _accepts(provider.read, "max_bytes"):
            kwargs["max_bytes"] = max_bytes
        if _accepts(provider.read, "context"):
            kwargs["context"] = context
        result = provider.read(**kwargs)
        if inspect.isawaitable(result):
            return _run_awaitable(result)
        return result


def is_resource_uri(value: object) -> bool:
    return isinstance(value, str) and bool(_URI_RE.match(value))


def coerce_resource_data(value: Any, *, uri: str, mime_type: str | None = None, filename: str = "") -> ResourceData:
    if isinstance(value, ResourceData):
        return ResourceData(
            uri=value.uri or uri,
            kind=value.kind,
            text=value.text,
            data=value.data,
            path=value.path,
            mime_type=value.mime_type or mime_type or _default_mime_for_kind(value.kind),
            filename=value.filename or filename,
            metadata=dict(value.metadata),
        )
    if isinstance(value, str):
        return ResourceData(uri=uri, kind="text", text=value, mime_type=mime_type or "text/plain; charset=utf-8", filename=filename)
    if isinstance(value, bytes):
        return ResourceData(uri=uri, kind="bytes", data=value, mime_type=mime_type or "application/octet-stream", filename=filename)
    if isinstance(value, Path):
        return ResourceData(uri=uri, kind="path", path=value, mime_type=mime_type or "application/octet-stream", filename=filename or value.name)
    if isinstance(value, dict):
        return _resource_data_from_dict(value, uri=uri, default_mime_type=mime_type, default_filename=filename)
    raise TypeError(f"Unsupported resource result for {uri}: {type(value).__name__}")


def _resource_data_from_dict(value: dict[str, Any], *, uri: str, default_mime_type: str | None, default_filename: str) -> ResourceData:
    kind = str(value.get("kind") or "").strip()
    text = value.get("text")
    data = value.get("data")
    path = value.get("path")
    present = {
        "text": text is not None,
        "bytes": data is not None,
        "path": path is not None,
    }
    if sum(1 for enabled in present.values() if enabled) != 1:
        raise ValueError(f"Resource result for {uri} must set exactly one of text, data, or path")
    if not kind:
        if text is not None:
            kind = "text"
        elif data is not None:
            kind = "bytes"
        elif path is not None:
            kind = "path"
    if kind not in {"text", "bytes", "path"}:
        raise ValueError(f"Invalid resource kind for {uri}: {kind!r}")
    if not present[kind]:
        raise ValueError(f"Resource result for {uri} has kind={kind!r} but no matching payload")
    if kind == "text" and not isinstance(text, str):
        raise TypeError(f"Resource text for {uri} must be str")
    if kind == "bytes" and not isinstance(data, bytes):
        raise TypeError(f"Resource data for {uri} must be bytes")
    metadata = value.get("metadata")
    return ResourceData(
        uri=str(value.get("uri") or uri),
        kind=kind,  # type: ignore[arg-type]
        text=text if kind == "text" else None,
        data=data if kind == "bytes" else None,
        path=Path(path) if kind == "path" and path is not None else None,
        mime_type=str(value.get("mime_type") or default_mime_type or _default_mime_for_kind(kind)),
        filename=str(value.get("filename") or default_filename or ""),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _normalize_prefix(prefix: str) -> str:
    value = str(prefix or "").strip()
    if not is_resource_uri(value):
        raise ValueError(f"Resource prefix must start with a URI scheme: {prefix!r}")
    return value


def _accepts(fn: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()) or name in signature.parameters


def _run_awaitable(value: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError("Async resource readers cannot be resolved while an event loop is already running")


def _default_mime_for_kind(kind: str) -> str:
    if kind == "text":
        return "text/plain; charset=utf-8"
    return "application/octet-stream"
