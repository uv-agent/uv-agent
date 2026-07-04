from __future__ import annotations

import http.client
import json
from pathlib import Path
from urllib.parse import quote

from uv_agent.blobs import BLOB_ID_PREFIX, BlobStore
from uv_agent.host_events import HostEventBus
from uv_agent.runner.run_log import RunLogStore
from uv_agent.runner.rpc import RuntimeRPCServer


def test_runtime_rpc_server_handles_notification_and_call(tmp_path: Path) -> None:
    server = RuntimeRPCServer()
    events: list[dict] = []
    store = _run_store(tmp_path, "run_rpc")
    writer = store.writer("run_rpc")
    server.register_method("echo", lambda text: {"text": text})
    server.register_method("run_id", lambda context: context.run_id)
    try:
        handle = server.open_session(
            run_id="run_rpc",
            thread_id="thread_1",
            turn_id="turn_1",
            cwd=tmp_path,
            structured_events=events,
            writer=writer,
        )
        try:
            status, body = _post(
                server.url,
                handle.token,
                {
                    "jsonrpc": "2.0",
                    "method": "event.emit",
                    "params": {
                        "kind": "progress",
                        "message": "working",
                        "_uv_agent_event_id": "evt_1",
                        "_uv_agent_run_id": "run_rpc",
                    },
                },
            )
            assert status == 204
            assert body == b""
            assert events == [
                {
                    "kind": "progress",
                    "message": "working",
                    "_uv_agent_event_id": "evt_1",
                    "_uv_agent_run_id": "run_rpc",
                }
            ]
            assert store.read_events("run_rpc")[0]["type"] == "run.event"

            status, body = _post(
                server.url,
                handle.token,
                {"jsonrpc": "2.0", "id": "1", "method": "call.echo", "params": {"text": "hi"}},
            )
            assert status == 200
            assert json.loads(body) == {"jsonrpc": "2.0", "id": "1", "result": {"text": "hi"}}

            status, body = _post(
                server.url,
                handle.token,
                {"jsonrpc": "2.0", "id": "2", "method": "call.run_id", "params": {}},
            )
            assert status == 200
            assert json.loads(body) == {"jsonrpc": "2.0", "id": "2", "result": "run_rpc"}
        finally:
            handle.close()
    finally:
        server.close()


def test_runtime_rpc_server_accepts_helper_call_summaries(tmp_path: Path) -> None:
    server = RuntimeRPCServer()
    calls: list[dict] = []
    store = _run_store(tmp_path, "run_rpc")
    writer = store.writer("run_rpc")
    try:
        handle = server.open_session(
            run_id="run_rpc",
            thread_id="thread_1",
            turn_id="turn_1",
            cwd=tmp_path,
            on_helper_calls=calls.extend,
            writer=writer,
        )
        try:
            status, body = _post(
                server.url,
                handle.token,
                {
                    "jsonrpc": "2.0",
                    "method": "helper.calls",
                    "params": {
                        "run_id": "run_rpc",
                        "calls": [
                            {
                                "helper": "path",
                                "count": 3,
                                "outcomes": {"ok": 3},
                                "keyword_names": ["root"],
                                "argument_types": {"args": ["str"], "kwargs": {}},
                            }
                        ],
                    },
                },
            )

            assert status == 204
            assert body == b""
            assert calls == [
                {
                    "name": "path",
                    "args": "",
                    "source": "runtime",
                    "count": 3,
                    "outcomes": {"ok": 3},
                    "keyword_names": ["root"],
                    "argument_types": {"args": ["str"], "kwargs": {}},
                }
            ]
        finally:
            handle.close()
    finally:
        server.close()


def test_runtime_rpc_server_publishes_ui_message_host_events(tmp_path: Path) -> None:
    host_events = HostEventBus()
    published: list[dict] = []
    host_events.subscribe(lambda event: published.append(event))
    server = RuntimeRPCServer(host_events=host_events)
    events: list[dict] = []
    store = _run_store(tmp_path, "run_rpc")
    writer = store.writer("run_rpc")
    try:
        handle = server.open_session(
            run_id="run_rpc",
            thread_id="thread_1",
            turn_id="turn_1",
            cwd=tmp_path,
            structured_events=events,
            writer=writer,
        )
        try:
            status, body = _post(
                server.url,
                handle.token,
                {
                    "jsonrpc": "2.0",
                    "method": "event.emit",
                    "params": {
                        "kind": "ui.message",
                        "message": "**Authorize** at https://example.test",
                        "format": "markdown",
                        "_uv_agent_event_id": "evt_ui",
                        "_uv_agent_run_id": "run_rpc",
                    },
                },
            )

            assert status == 204
            assert body == b""
            assert events[0]["kind"] == "ui.message"
            assert len(published) == 1
            assert published[0]["type"] == "runtime.ui.message"
            assert published[0]["scope"] == "ui"
            assert published[0]["message"] == "**Authorize** at https://example.test"
            assert published[0]["format"] == "markdown"
            assert published[0]["run_id"] == "run_rpc"
            assert published[0]["thread_id"] == "thread_1"
            assert published[0]["turn_id"] == "turn_1"
            assert published[0]["event_id"] == "evt_ui"
        finally:
            handle.close()
    finally:
        server.close()


def test_runtime_rpc_server_rejects_unknown_or_closed_token(tmp_path: Path) -> None:
    server = RuntimeRPCServer()
    store = _run_store(tmp_path, "run_rpc")
    writer = store.writer("run_rpc")
    try:
        handle = server.open_session(
            run_id="run_rpc",
            thread_id=None,
            turn_id=None,
            cwd=tmp_path,
            structured_events=[],
            writer=writer,
        )
        url = server.url
        handle.close()
        status, _body = _post(
            url,
            handle.token,
            {"jsonrpc": "2.0", "id": "1", "method": "call.missing", "params": {}},
        )
        assert status == 401
    finally:
        server.close()


def test_runtime_rpc_server_maps_method_errors(tmp_path: Path) -> None:
    server = RuntimeRPCServer()
    store = _run_store(tmp_path, "run_rpc")
    writer = store.writer("run_rpc")
    server.register_method("boom", lambda: (_ for _ in ()).throw(ValueError("bad")))
    try:
        handle = server.open_session(
            run_id="run_rpc",
            thread_id=None,
            turn_id=None,
            cwd=tmp_path,
            structured_events=[],
            writer=writer,
        )
        try:
            status, body = _post(
                server.url,
                handle.token,
                {"jsonrpc": "2.0", "id": "1", "method": "call.unknown", "params": {}},
            )
            assert status == 200
            missing = json.loads(body)
            assert missing["error"]["code"] == -32601

            status, body = _post(
                server.url,
                handle.token,
                {"jsonrpc": "2.0", "id": "2", "method": "call.boom", "params": {}},
            )
            assert status == 200
            failed = json.loads(body)
            assert failed["error"]["code"] == -32000
            assert failed["error"]["data"]["type"] == "ValueError"
            assert "traceback" not in failed["error"]["data"]
        finally:
            handle.close()
    finally:
        server.close()


def test_runtime_rpc_server_handles_unexpected_post_errors_without_traceback(tmp_path: Path, capsys) -> None:
    server = RuntimeRPCServer()
    store = _run_store(tmp_path, "run_rpc")
    writer = store.writer("run_rpc")
    try:
        handle = server.open_session(
            run_id="run_rpc",
            thread_id=None,
            turn_id=None,
            cwd=tmp_path,
            structured_events=[],
            writer=writer,
        )
        try:
            def boom(_body, *, session):
                raise RuntimeError("dispatch boom")

            server.dispatch = boom  # type: ignore[method-assign]
            status, body = _post(
                server.url,
                handle.token,
                {"jsonrpc": "2.0", "id": "1", "method": "call.anything", "params": {}},
            )
            payload = json.loads(body)
            captured = capsys.readouterr()
            assert status == 500
            assert payload["error"]["code"] == -32603
            assert payload["error"]["data"]["type"] == "RuntimeError"
            assert "Traceback" not in captured.err
        finally:
            handle.close()
    finally:
        server.close()


def test_runtime_rpc_server_supports_dynamic_namespace_proxy(tmp_path: Path, monkeypatch) -> None:
    import uv_agent_runtime as rt

    server = RuntimeRPCServer()
    store = _run_store(tmp_path, "run_rpc")
    writer = store.writer("run_rpc")
    server.register_method(
        "helper.resolve",
        lambda name: {
            "demo": {
                "found": True,
                "kind": "namespace",
                "name": "demo",
                "plugin": "demo-plugin",
                "doc": "Demo namespace.",
                "functions": [{"name": "greet", "full_name": "demo.greet", "doc": "Greet.", "schema": {"type": "object"}}],
            },
            "demo.greet": {
                "found": True,
                "kind": "function",
                "name": "demo.greet",
                "namespace": "demo",
                "function": "greet",
                "plugin": "demo-plugin",
                "doc": "Greet.",
                "schema": {"type": "object"},
            },
        }.get(name, {"found": False, "name": name}),
    )
    server.register_method(
        "helper.call",
        lambda name, args=None, kwargs=None, context=None: {"name": name, "payload": dict(kwargs or {})},
    )
    try:
        handle = server.open_session(
            run_id="run_rpc",
            thread_id=None,
            turn_id=None,
            cwd=tmp_path,
            structured_events=[],
            writer=writer,
        )
        try:
            monkeypatch.setenv("UV_AGENT_RPC_URL", server.url)
            monkeypatch.setenv("UV_AGENT_RPC_TOKEN", handle.token)
            rt.__dict__.pop("demo", None)

            assert rt.demo.greet(name="Ada") == {"name": "demo.greet", "payload": {"name": "Ada"}}
        finally:
            handle.close()
            rt.__dict__.pop("demo", None)
    finally:
        server.close()


def test_runtime_rpc_server_supports_local_module_namespace(tmp_path: Path, monkeypatch) -> None:
    import uv_agent_runtime as rt

    server = RuntimeRPCServer()
    store = _run_store(tmp_path, "run_rpc")
    writer = store.writer("run_rpc")
    server.register_method(
        "helper.resolve",
        lambda name: {
            "found": True,
            "kind": "namespace",
            "name": "json",
            "plugin": "demo-plugin",
            "doc": "JSON module.",
            "module": "json",
            "functions": [],
        }
        if name == "json"
        else {"found": False, "name": name},
    )
    try:
        handle = server.open_session(
            run_id="run_rpc",
            thread_id=None,
            turn_id=None,
            cwd=tmp_path,
            structured_events=[],
            writer=writer,
        )
        try:
            monkeypatch.setenv("UV_AGENT_RPC_URL", server.url)
            monkeypatch.setenv("UV_AGENT_RPC_TOKEN", handle.token)
            rt.__dict__.pop("json", None)

            assert rt.json.loads('{"ok": true}') == {"ok": True}
        finally:
            handle.close()
            rt.__dict__.pop("json", None)
    finally:
        server.close()


def test_runtime_get_resolves_resource_uri_over_rpc(tmp_path: Path, monkeypatch) -> None:
    import uv_agent_runtime as rt

    server = RuntimeRPCServer()
    store = _run_store(tmp_path, "run_rpc")
    writer = store.writer("run_rpc")
    server.register_method(
        "resource.get",
        lambda target, max_bytes=None, context=None: {
            "uri": target,
            "kind": "text",
            "mime_type": "text/plain; charset=utf-8",
            "text": f"hello {max_bytes}",
            "metadata": {"source": "test"},
        },
    )
    try:
        handle = server.open_session(
            run_id="run_rpc",
            thread_id=None,
            turn_id=None,
            cwd=tmp_path,
            structured_events=[],
            writer=writer,
        )
        try:
            monkeypatch.setenv("UV_AGENT_RPC_URL", server.url)
            monkeypatch.setenv("UV_AGENT_RPC_TOKEN", handle.token)

            resource = rt.get("skill://project/demo", max_bytes=123)
            assert isinstance(resource, rt.Resource)
            assert resource.uri == "skill://project/demo"
            assert resource.text() == "hello 123"
            assert resource.metadata["source"] == "test"
            assert isinstance(rt.get(tmp_path / "local.txt"), rt.File)
        finally:
            handle.close()
    finally:
        server.close()


def test_runtime_rpc_server_uploads_and_serves_blobs(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path)
    server = RuntimeRPCServer(blob_store=blob_store, max_blob_bytes=16)
    store = _run_store(tmp_path, "run_rpc")
    writer = store.writer("run_rpc")
    try:
        handle = server.open_session(
            run_id="run_rpc",
            thread_id="thread_1",
            turn_id="turn_1",
            cwd=tmp_path,
            structured_events=[],
            writer=writer,
        )
        try:
            status, body = _post_blob(
                server.url,
                handle.token,
                b"abc",
                headers={"X-Uv-Agent-Mime-Type": "image/png", "X-Uv-Agent-Filename": "demo.png"},
            )
            assert status == 200
            payload = json.loads(body)
            assert payload["blob_id"].startswith(BLOB_ID_PREFIX)
            assert payload["mime_type"] == "image/png"
            assert payload["filename"] == "demo.png"
            assert Path(payload["path"]).read_bytes() == b"abc"

            encoded = quote(payload["blob_id"], safe="")
            status, body = _get(server.url, handle.token, f"/blob/{encoded}/info")
            assert status == 200
            info = json.loads(body)
            assert info["blob_id"] == payload["blob_id"]
            assert info["size_bytes"] == 3

            status, body = _get(server.url, handle.token, f"/blob/{encoded}")
            assert status == 200
            assert body == b"abc"

            status, _body = _post_blob(server.url, handle.token, b"x" * 17)
            assert status == 413
        finally:
            handle.close()
    finally:
        server.close()


def _run_store(tmp_path: Path, run_id: str) -> RunLogStore:
    store = RunLogStore(tmp_path)
    store.create_run_record(
        run_id=run_id,
        code="",
        script_args=[],
        cwd=tmp_path,
        timeout_s=None,
        started_at="test",
        thread_id=None,
        turn_id=None,
        script_path=None,
    )
    return store


def _post(url: str, token: str, payload: dict) -> tuple[int, bytes]:
    host, port_text = url.removeprefix("http://").split(":", 1)
    connection = http.client.HTTPConnection(host, int(port_text), timeout=5)
    try:
        body = json.dumps(payload).encode("utf-8")
        connection.request(
            "POST",
            "/rpc",
            body=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _post_blob(url: str, token: str, body: bytes, *, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    host, port_text = url.removeprefix("http://").split(":", 1)
    connection = http.client.HTTPConnection(host, int(port_text), timeout=5)
    try:
        merged_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
            **(headers or {}),
        }
        connection.request("POST", "/blob", body=body, headers=merged_headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _get(url: str, token: str, path: str) -> tuple[int, bytes]:
    host, port_text = url.removeprefix("http://").split(":", 1)
    connection = http.client.HTTPConnection(host, int(port_text), timeout=5)
    try:
        connection.request("GET", path, headers={"Authorization": f"Bearer {token}"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()
