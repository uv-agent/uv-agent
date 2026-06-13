from __future__ import annotations

import http.client
import json
from pathlib import Path

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
                                "helper": "path_info",
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
                    "name": "path_info",
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
