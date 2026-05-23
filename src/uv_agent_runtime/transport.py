from __future__ import annotations

import http.client
import json
import os
import threading
from itertools import count
from typing import Any
from urllib.parse import urlparse

RPC_URL_ENV = "UV_AGENT_RPC_URL"
RPC_TOKEN_ENV = "UV_AGENT_RPC_TOKEN"
JSONRPC_VERSION = "2.0"
DEFAULT_TIMEOUT_S = 5.0

_request_counter = count(1)
_counter_lock = threading.Lock()


class HostCallError(RuntimeError):
    """Raised when the runtime cannot complete a host RPC call."""


def emit_event_rpc(event: dict[str, Any]) -> None:
    """Best-effort JSON-RPC notification for runtime structured events."""

    if not _rpc_configured():
        return
    try:
        _post_jsonrpc({"jsonrpc": JSONRPC_VERSION, "method": "event.emit", "params": event}, expect_response=False)
    except Exception:
        # Structured events are auxiliary metadata; failing to deliver one should
        # not contaminate stdout or fail an otherwise useful user script.
        return


def call_host(name: str, **kwargs: Any) -> Any:
    """Call a host-registered helper from inside a managed runtime script."""

    if not _rpc_configured():
        raise RuntimeError(f"Host RPC is not configured; cannot call {name!r}")
    request_id = _next_request_id()
    response = _post_jsonrpc(
        {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": f"call.{name}",
            "params": kwargs,
        },
        expect_response=True,
    )
    if not isinstance(response, dict):
        raise HostCallError("Host RPC returned an invalid response")
    error = response.get("error")
    if error is not None:
        if not isinstance(error, dict):
            raise HostCallError("Host RPC returned an invalid error response")
        code = error.get("code")
        message = str(error.get("message") or "Host RPC error")
        if code == -32601:
            raise LookupError(message)
        raise RuntimeError(message)
    return response.get("result")


def _next_request_id() -> str:
    with _counter_lock:
        return str(next(_request_counter))


def _rpc_configured() -> bool:
    return bool(os.environ.get(RPC_URL_ENV) and os.environ.get(RPC_TOKEN_ENV))


def _post_jsonrpc(payload: dict[str, Any], *, expect_response: bool) -> Any:
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            return _post_jsonrpc_once(payload, expect_response=expect_response)
        except (OSError, TimeoutError, http.client.HTTPException, HostCallError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise HostCallError("Host RPC failed")


def _post_jsonrpc_once(payload: dict[str, Any], *, expect_response: bool) -> Any:
    url = os.environ.get(RPC_URL_ENV)
    token = os.environ.get(RPC_TOKEN_ENV)
    if not url or not token:
        raise HostCallError("Host RPC is not configured")
    parsed = urlparse(url)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise HostCallError(f"Unsupported host RPC URL: {url!r}")
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=DEFAULT_TIMEOUT_S)
    try:
        connection.request(
            "POST",
            "/rpc",
            body=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        response_body = response.read()
        if not expect_response and response.status == 204:
            return None
        if response.status == 401:
            raise HostCallError("Host RPC authorization failed")
        if response.status >= 400:
            raise HostCallError(f"Host RPC HTTP {response.status}")
        if not response_body:
            if expect_response:
                raise HostCallError("Host RPC returned an empty response")
            return None
        try:
            return json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HostCallError(f"Host RPC returned invalid JSON: {exc}") from exc
    finally:
        connection.close()
