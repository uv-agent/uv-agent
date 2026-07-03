from __future__ import annotations

import json
import logging
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from uv_agent.runner.run_log import EventWriter

from typing import TYPE_CHECKING

from .auth import bearer_token, is_loopback_address
from .dispatcher import DispatchResult, JsonRpcDispatcher
from .registry import HostMethod, MethodRegistry
from .session import RunSession

if TYPE_CHECKING:
    from uv_agent.host_events import HostEventBus

DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024

logger = logging.getLogger(__name__)


class _RuntimeRPCHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        # Runtime RPC is an internal transport.  Request-level failures are
        # converted to JSON-RPC errors where possible; worker tracebacks would
        # leak implementation details into the TUI/terminal.
        return


class RuntimeRPCSessionHandle:
    """Handle returned to the runner for closing a per-run RPC session."""

    def __init__(self, server: "RuntimeRPCServer", token: str) -> None:
        self._server = server
        self.token = token
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server.close_session(self.token)

    def __enter__(self) -> "RuntimeRPCSessionHandle":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class RuntimeRPCServer:
    """Long-lived lightweight HTTP server used by managed runtime scripts."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        host_events: "HostEventBus | None" = None,
    ) -> None:
        self.host = host
        self.max_body_bytes = max_body_bytes
        self._host_events = host_events
        self.methods = MethodRegistry()
        self._dispatcher = JsonRpcDispatcher(self.methods)
        self._lock = threading.RLock()
        self._sessions: dict[str, RunSession] = {}
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None

    @property
    def url(self) -> str:
        self.start()
        assert self._url is not None
        return self._url

    def start(self) -> str:
        with self._lock:
            if self._httpd is not None and self._url is not None:
                return self._url

            handler = self._handler_class()
            httpd = _RuntimeRPCHTTPServer((self.host, 0), handler)
            httpd.daemon_threads = True
            self._httpd = httpd
            host, port = httpd.server_address[:2]
            self._url = f"http://{host}:{port}"
            self._thread = threading.Thread(
                target=httpd.serve_forever,
                name="uv-agent-runtime-rpc",
                daemon=True,
            )
            self._thread.start()
            logger.info("Runtime RPC server started url=%s", self._url)
            return self._url

    def stop(self) -> None:
        with self._lock:
            httpd = self._httpd
            thread = self._thread
            self._httpd = None
            self._thread = None
            self._url = None
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if httpd is not None or sessions:
            logger.info("Runtime RPC server stopped closed_sessions=%d", len(sessions))

    close = stop

    def register_method(self, name: str, method: HostMethod) -> None:
        self.methods.register(name, method)

    def unregister_method(self, name: str) -> None:
        self.methods.unregister(name)

    def open_session(
        self,
        *,
        run_id: str,
        thread_id: str | None,
        turn_id: str | None,
        cwd: Path,
        writer: EventWriter,
        structured_events: list[dict[str, Any]] | None = None,
        on_structured_event: Callable[[dict[str, Any]], None] | None = None,
        on_helper_calls: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> RuntimeRPCSessionHandle:
        self.start()
        token = secrets.token_urlsafe(32)
        if on_structured_event is not None:
            callback = on_structured_event
        else:
            events = structured_events if structured_events is not None else []

            def callback(event: dict[str, Any]) -> None:
                events.append(event)

        session = RunSession(
            token=token,
            run_id=run_id,
            thread_id=thread_id,
            turn_id=turn_id,
            cwd=cwd,
            on_structured_event=callback,
            on_helper_calls=on_helper_calls,
            writer=writer,
            host_events=self._host_events,
        )
        with self._lock:
            self._sessions[token] = session
        logger.debug("Runtime RPC session opened run_id=%s thread_id=%s turn_id=%s cwd=%s", run_id, thread_id, turn_id, cwd)
        return RuntimeRPCSessionHandle(self, token)

    def close_session(self, token: str) -> None:
        with self._lock:
            session = self._sessions.pop(token, None)
        if session is not None:
            session.close()
            logger.debug("Runtime RPC session closed run_id=%s", session.run_id)

    def session_for_token(self, token: str | None) -> RunSession | None:
        if not token:
            return None
        with self._lock:
            return self._sessions.get(token)

    def dispatch(self, body: bytes, *, session: RunSession) -> DispatchResult:
        return self._dispatcher.dispatch_bytes(body, session=session)

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        rpc_server = self

        class RuntimeRPCRequestHandler(BaseHTTPRequestHandler):
            server_version = "UvAgentRuntimeRPC/1"

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
                if self.path == "/healthz":
                    self._send_bytes(HTTPStatus.OK, b"ok\n", content_type="text/plain; charset=utf-8")
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                try:
                    self._do_POST()
                except Exception as exc:
                    logger.warning("Runtime RPC request failed error_type=%s", exc.__class__.__name__)
                    self._send_internal_error(exc)

            def _do_POST(self) -> None:
                if self.path != "/rpc":
                    self._discard_request_body()
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                if not is_loopback_address(self.client_address[0]):
                    self._discard_request_body()
                    self.send_error(HTTPStatus.UNAUTHORIZED)
                    return
                session = rpc_server.session_for_token(bearer_token(self.headers.get("Authorization")))
                if session is None:
                    self._discard_request_body()
                    self.send_error(HTTPStatus.UNAUTHORIZED)
                    return
                content_length = self.headers.get("Content-Length")
                try:
                    length = int(content_length or "0")
                except ValueError:
                    self._discard_request_body()
                    self.send_error(HTTPStatus.LENGTH_REQUIRED)
                    return
                if length < 0:
                    self._discard_request_body()
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                if length > rpc_server.max_body_bytes:
                    self._discard_request_body(max_bytes=0)
                    self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                    return
                content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
                if content_type and content_type != "application/json":
                    self._discard_request_body(max_bytes=length)
                    self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                    return
                body = self.rfile.read(length)
                result = rpc_server.dispatch(body, session=session)
                if result.body is None:
                    self.send_response(result.status)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                response = json.dumps(result.body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self._send_bytes(result.status, response, content_type="application/json; charset=utf-8")

            def log_message(self, _format: str, *_args: object) -> None:
                # Runtime RPC is an internal transport; request logs would add
                # noise and risk exposing method names from user scripts.
                return

            def _discard_request_body(self, *, max_bytes: int | None = None) -> None:
                """Best-effort drain before early POST errors.

                Windows can reset a loopback HTTP connection if the server sends
                an error response and closes while the client is still writing
                the request body. Draining bounded bodies first keeps auth/path
                failures deterministic without changing the RPC surface.
                """

                try:
                    remaining = int(self.headers.get("Content-Length") or "0")
                except ValueError:
                    return
                if max_bytes is None:
                    max_bytes = rpc_server.max_body_bytes
                remaining = max(0, min(remaining, max_bytes))
                while remaining:
                    chunk = self.rfile.read(min(remaining, 64 * 1024))
                    if not chunk:
                        return
                    remaining -= len(chunk)

            def _send_bytes(self, status: int, body: bytes, *, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _send_internal_error(self, exc: Exception) -> None:
                body = json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32603,
                            "message": "Internal error",
                            "data": {"type": exc.__class__.__name__},
                        },
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                try:
                    self._send_bytes(HTTPStatus.INTERNAL_SERVER_ERROR, body, content_type="application/json; charset=utf-8")
                except Exception:
                    return

        return RuntimeRPCRequestHandler
