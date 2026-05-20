from __future__ import annotations

import json
import os
import sys


TOOLS = [
    {
        "name": "echo",
        "description": "Echo text",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
    }
]


def respond(message: dict[str, object], result: object) -> None:
    print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)


for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        respond(
            message,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "echo", "version": "0.1"},
                **(
                    {"instructions": os.environ["UV_AGENT_TEST_MCP_INSTRUCTIONS"]}
                    if os.environ.get("UV_AGENT_TEST_MCP_INSTRUCTIONS")
                    else {}
                ),
            },
        )
    elif method == "tools/list":
        respond(message, {"tools": TOOLS})
    elif method == "tools/call":
        params = message.get("params") or {}
        arguments = params.get("arguments") or {}
        respond(
            message,
            {
                "content": [
                    {"type": "text", "text": str(arguments.get("text", ""))},
                ]
            },
        )
    elif "id" in message:
        print(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32601, "message": "method not found"},
                }
            ),
            flush=True,
        )
