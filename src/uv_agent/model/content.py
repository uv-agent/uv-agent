from __future__ import annotations

import copy
from typing import Any

from uv_agent.config import ModelConfig


def extract_responses_text(output: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "refusal"}:
                parts.append(content.get("text", ""))
    return "".join(parts)


def chat_messages(
    input_items: list[dict[str, Any]],
    instructions: str | None,
    model: ModelConfig | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    pending_assistant: dict[str, Any] | None = None

    def flush_pending_assistant() -> None:
        nonlocal pending_assistant
        if pending_assistant is not None:
            messages.append(pending_assistant)
            pending_assistant = None

    for item in input_items:
        item_type = item.get("type")
        if item_type == "message":
            flush_pending_assistant()
            role = item.get("role", "user")
            message = {"role": role, "content": chat_message_content(item)}
            if model is not None:
                for field in model.message_passthrough.fields_for_role(str(role)):
                    if field in item:
                        message[field] = copy.deepcopy(item[field])
            if role == "assistant":
                pending_assistant = message
            else:
                messages.append(message)
        elif item_type == "function_call":
            if pending_assistant is None:
                pending_assistant = {"role": "assistant", "content": ""}
            pending_assistant.setdefault("tool_calls", []).append(chat_tool_call_message(item))
        elif item_type == "function_call_output":
            flush_pending_assistant()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": item.get("output", ""),
                }
            )
    flush_pending_assistant()
    return messages


def chat_tool_call_message(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("call_id"),
        "type": "function",
        "function": {
            "name": item.get("name"),
            "arguments": item.get("arguments") or "{}",
        },
    }


def message_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for content in item.get("content") or []:
        if content.get("type") in {"input_text", "output_text", "text", "refusal"}:
            parts.append(content.get("text", ""))
    return "\n".join(parts)


def chat_message_content(item: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Return Chat Completions content, preserving image parts when present."""
    parts: list[dict[str, Any]] = []
    has_image = False
    for content in item.get("content") or []:
        content_type = content.get("type")
        if content_type in {"input_text", "output_text", "text", "refusal"}:
            parts.append({"type": "text", "text": content.get("text", "")})
        elif content_type == "input_image":
            image_url = content.get("image_url") or content.get("url")
            if image_url:
                has_image = True
                parts.append({"type": "image_url", "image_url": {"url": image_url}})
    if has_image:
        return parts
    return "\n".join(part.get("text", "") for part in parts)


def chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def chat_tool_acc_from_message(message: dict[str, Any]) -> dict[int, dict[str, Any]]:
    acc: dict[int, dict[str, Any]] = {}
    for index, tool_call in enumerate(message.get("tool_calls") or []):
        function = tool_call.get("function") or {}
        acc[index] = {
            "call_id": tool_call.get("id"),
            "name": function.get("name"),
            "arguments": function.get("arguments", ""),
        }
    return acc


def chat_message_passthrough(message: dict[str, Any], model: ModelConfig | None) -> dict[str, Any]:
    if model is None:
        return {}
    return {
        field: copy.deepcopy(message[field])
        for field in model.message_passthrough.fields_for_role("assistant")
        if field in message
    }


def chat_message_reasoning_text(message: dict[str, Any], model: ModelConfig | None) -> str:
    if model is None:
        return ""
    parts = [
        str(message[field])
        for field in model.reasoning_display.assistant_message_fields
        if isinstance(message.get(field), str) and message.get(field)
    ]
    return "".join(parts)


def chat_output_items(
    output_text: str,
    tool_acc: dict[int, dict[str, Any]],
    *,
    passthrough: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    passthrough = passthrough or {}
    if output_text or passthrough:
        message = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": output_text}],
        }
        message.update(copy.deepcopy(passthrough))
        output.append(message)
    for index in sorted(tool_acc):
        call = tool_acc[index]
        output.append(
            {
                "type": "function_call",
                "call_id": call.get("call_id"),
                "name": call.get("name"),
                "arguments": call.get("arguments", ""),
            }
        )
    return output
