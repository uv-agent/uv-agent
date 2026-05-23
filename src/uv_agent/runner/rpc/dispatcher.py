from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from typing import Any

from uv_agent_runtime.events import RUNTIME_EVENT_RUN_ID_KEY

from .registry import MethodRegistry
from .session import RunSession

JSONRPC_VERSION = "2.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
BUSINESS_ERROR = -32000
SESSION_CLOSED = -32003


@dataclass(frozen=True)
class DispatchResult:
    status: int
    body: dict[str, Any] | None = None


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str, *, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class JsonRpcDispatcher:
    """Small JSON-RPC 2.0 dispatcher for runtime-to-host calls."""

    def __init__(self, methods: MethodRegistry) -> None:
        self._methods = methods

    def dispatch_bytes(self, body: bytes, *, session: RunSession) -> DispatchResult:
        try:
            payload = json.loads(body.decode("utf-8"))
        except UnicodeDecodeError as exc:
            return DispatchResult(200, _error_response(None, PARSE_ERROR, f"Invalid UTF-8: {exc}"))
        except json.JSONDecodeError as exc:
            return DispatchResult(200, _error_response(None, PARSE_ERROR, f"Parse error: {exc}"))
        return self.dispatch(payload, session=session)

    def dispatch(self, payload: Any, *, session: RunSession) -> DispatchResult:
        if isinstance(payload, list):
            return DispatchResult(200, _error_response(None, INVALID_REQUEST, "Batch requests are not implemented"))
        if not isinstance(payload, dict):
            return DispatchResult(200, _error_response(None, INVALID_REQUEST, "Invalid request"))

        request_id = payload.get("id")
        is_notification = "id" not in payload
        try:
            result = self._handle(payload, session=session)
        except JsonRpcError as exc:
            if is_notification:
                return DispatchResult(204)
            return DispatchResult(200, _error_response(request_id, exc.code, exc.message, data=exc.data))
        except Exception as exc:
            if is_notification:
                return DispatchResult(204)
            return DispatchResult(
                200,
                _error_response(
                    request_id,
                    INTERNAL_ERROR,
                    "Internal error",
                    data={"type": exc.__class__.__name__},
                ),
            )

        if is_notification:
            return DispatchResult(204)
        return DispatchResult(200, {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result})

    def _handle(self, payload: dict[str, Any], *, session: RunSession) -> Any:
        if payload.get("jsonrpc") != JSONRPC_VERSION:
            raise JsonRpcError(INVALID_REQUEST, "Invalid JSON-RPC version")
        method = payload.get("method")
        if not isinstance(method, str) or not method:
            raise JsonRpcError(INVALID_REQUEST, "Missing method")
        params = payload.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise JsonRpcError(INVALID_PARAMS, "Params must be an object")
        if session.closed:
            raise JsonRpcError(SESSION_CLOSED, "Run session is closed")

        if method == "event.emit":
            return self._emit_event(params, session=session)
        if method == "helper.resolve":
            return self._resolve_helper(params)
        if method.startswith("call."):
            return self._call_host(method.removeprefix("call."), params, session=session)
        raise JsonRpcError(METHOD_NOT_FOUND, "Method not found", data={"method": method})

    def _emit_event(self, params: dict[str, Any], *, session: RunSession) -> dict[str, bool]:
        kind = params.get("kind")
        if not isinstance(kind, str) or not kind:
            raise JsonRpcError(INVALID_PARAMS, "Runtime event requires a non-empty kind")
        event_run_id = params.get(RUNTIME_EVENT_RUN_ID_KEY)
        if event_run_id != session.run_id:
            raise JsonRpcError(
                INVALID_PARAMS,
                "Runtime event run id does not match session",
                data={"run_id": session.run_id},
            )
        try:
            session.emit_event(params)
        except RuntimeError as exc:
            raise JsonRpcError(SESSION_CLOSED, str(exc)) from exc
        return {"ok": True}

    def _resolve_helper(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise JsonRpcError(INVALID_PARAMS, "helper.resolve requires a helper name")
        resolver = self._methods.get("helper.resolve")
        if resolver is None:
            return {"found": False, "name": name}
        try:
            result = resolver(name=name)
        except Exception as exc:
            raise JsonRpcError(BUSINESS_ERROR, f"{exc.__class__.__name__}: {exc}") from exc
        return result if isinstance(result, dict) else {"found": False, "name": name}

    def _call_host(self, name: str, params: dict[str, Any], *, session: RunSession) -> Any:
        try:
            return self._methods.call(name, params, context=session.context)
        except KeyError as exc:
            raise JsonRpcError(METHOD_NOT_FOUND, "Method not found", data={"method": f"call.{name}"}) from exc
        except Exception as exc:
            raise JsonRpcError(
                BUSINESS_ERROR,
                f"{exc.__class__.__name__}: {exc}",
                data={
                    "type": exc.__class__.__name__,
                    "traceback": "".join(traceback.format_exception(exc)),
                },
            ) from exc


def _error_response(
    request_id: Any,
    code: int,
    message: str,
    *,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}
