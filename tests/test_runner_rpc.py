from __future__ import annotations

import http.client
import json
from pathlib import Path

from uv_agent.jsonl import JsonlWriter, read_jsonl
from uv_agent.runner.rpc import RuntimeRPCServer


def test_runtime_rpc_server_handles_notification_and_call(tmp_path: Path) -> None:
    server = RuntimeRPCServer()
    events: list[dict] = []
    writer = JsonlWriter(tmp_path / "run.jsonl")
    server.register_method("echo", lambda text: {"text": text})
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
            assert read_jsonl(tmp_path / "run.jsonl")[0]["type"] == "run.event"

            status, body = _post(
                server.url,
                handle.token,
                {"jsonrpc": "2.0", "id": "1", "method": "call.echo", "params": {"text": "hi"}},
            )
            assert status == 200
            assert json.loads(body) == {"jsonrpc": "2.0", "id": "1", "result": {"text": "hi"}}
        finally:
            handle.close()
    finally:
        server.close()


def test_runtime_rpc_server_rejects_unknown_or_closed_token(tmp_path: Path) -> None:
    server = RuntimeRPCServer()
    writer = JsonlWriter(tmp_path / "run.jsonl")
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
    writer = JsonlWriter(tmp_path / "run.jsonl")
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
