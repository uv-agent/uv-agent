from __future__ import annotations

from typing import TYPE_CHECKING, Any


# Public names are kept for backwards compatibility, but provider-specific
# modules are imported only when a caller actually asks for those symbols. This
# is especially important for TUI startup: importing ``uv_agent.model`` used to
# import OpenAI, Anthropic, and large generated type modules even before the
# first screen could be drawn.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CHAT_DELTA_CONTROL_FIELDS": ("uv_agent.model.chat", "CHAT_DELTA_CONTROL_FIELDS"),
    "FakeModelClient": ("uv_agent.model.client", "FakeModelClient"),
    "UnifiedModelClient": ("uv_agent.model.client", "UnifiedModelClient"),
    "ModelClient": ("uv_agent.model.types", "ModelClient"),
    "ModelResponse": ("uv_agent.model.types", "ModelResponse"),
    "ModelStreamEvent": ("uv_agent.model.types", "ModelStreamEvent"),
    "ToolCallDelta": ("uv_agent.model.types", "ToolCallDelta"),
    "anthropic_image_source": ("uv_agent.model.anthropic", "anthropic_image_source"),
    "anthropic_sdk_base_url": ("uv_agent.model.anthropic", "anthropic_sdk_base_url"),
    "anthropic_message_content": ("uv_agent.model.anthropic", "anthropic_message_content"),
    "anthropic_messages": ("uv_agent.model.anthropic", "anthropic_messages"),
    "anthropic_payload": ("uv_agent.model.anthropic", "anthropic_payload"),
    "anthropic_tool": ("uv_agent.model.anthropic", "anthropic_tool"),
    "chat_message_content": ("uv_agent.model.content", "chat_message_content"),
    "chat_message_passthrough": ("uv_agent.model.content", "chat_message_passthrough"),
    "chat_message_reasoning_text": ("uv_agent.model.content", "chat_message_reasoning_text"),
    "chat_messages": ("uv_agent.model.content", "chat_messages"),
    "chat_output_items": ("uv_agent.model.content", "chat_output_items"),
    "chat_create_kwargs": ("uv_agent.model.chat", "chat_create_kwargs"),
    "chat_payload": ("uv_agent.model.chat", "chat_payload"),
    "chat_tool": ("uv_agent.model.content", "chat_tool"),
    "chat_tool_acc_from_message": ("uv_agent.model.content", "chat_tool_acc_from_message"),
    "chat_tool_call_message": ("uv_agent.model.content", "chat_tool_call_message"),
    "create_anthropic_response": ("uv_agent.model.anthropic", "create_anthropic_response"),
    "create_chat_response": ("uv_agent.model.chat", "create_chat_response"),
    "endpoint_extra_body": ("uv_agent.model.anthropic", "endpoint_extra_body"),
    "extract_responses_text": ("uv_agent.model.content", "extract_responses_text"),
    "message_text": ("uv_agent.model.content", "message_text"),
    "openai_client": ("uv_agent.model.openai_sdk", "openai_client"),
    "parse_anthropic_response": ("uv_agent.model.anthropic", "parse_anthropic_response"),
    "parse_anthropic_message": ("uv_agent.model.anthropic", "parse_anthropic_message"),
    "parse_chat_response": ("uv_agent.model.chat", "parse_chat_response"),
    "parse_chat_response_for_model": ("uv_agent.model.chat", "parse_chat_response_for_model"),
    "parse_responses_response": ("uv_agent.model.responses", "parse_responses_response"),
    "create_responses_response": ("uv_agent.model.responses", "create_responses_response"),
    "responses_create_kwargs": ("uv_agent.model.responses", "responses_create_kwargs"),
    "responses_payload": ("uv_agent.model.responses", "responses_payload"),
    "stream_anthropic_response": ("uv_agent.model.anthropic", "stream_anthropic_response"),
    "stream_chat_response": ("uv_agent.model.chat", "stream_chat_response"),
    "stream_responses_response": ("uv_agent.model.responses", "stream_responses_response"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import public model helpers on demand.

    ``from uv_agent.model import UnifiedModelClient`` still works exactly like
    before, but unrelated provider SDKs stay unloaded until their APIs are used.
    The resolved attribute is cached in ``globals()`` so repeated access is as
    cheap as a normal module global lookup.
    """

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


if TYPE_CHECKING:
    # These imports are for static analyzers only; runtime uses __getattr__.
    # The ``X as X`` re-export form marks each name as an intentional public
    # re-export so the unused-import check (F401) stays satisfied.
    from uv_agent.model.anthropic import (
        anthropic_image_source as anthropic_image_source,
        anthropic_sdk_base_url as anthropic_sdk_base_url,
        anthropic_message_content as anthropic_message_content,
        anthropic_messages as anthropic_messages,
        anthropic_payload as anthropic_payload,
        anthropic_tool as anthropic_tool,
        create_anthropic_response as create_anthropic_response,
        endpoint_extra_body as endpoint_extra_body,
        parse_anthropic_message as parse_anthropic_message,
        parse_anthropic_response as parse_anthropic_response,
        stream_anthropic_response as stream_anthropic_response,
    )
    from uv_agent.model.chat import (
        CHAT_DELTA_CONTROL_FIELDS as CHAT_DELTA_CONTROL_FIELDS,
        chat_create_kwargs as chat_create_kwargs,
        chat_payload as chat_payload,
        create_chat_response as create_chat_response,
        parse_chat_response as parse_chat_response,
        parse_chat_response_for_model as parse_chat_response_for_model,
        stream_chat_response as stream_chat_response,
    )
    from uv_agent.model.client import (
        FakeModelClient as FakeModelClient,
        UnifiedModelClient as UnifiedModelClient,
    )
    from uv_agent.model.content import (
        chat_message_content as chat_message_content,
        chat_message_passthrough as chat_message_passthrough,
        chat_message_reasoning_text as chat_message_reasoning_text,
        chat_messages as chat_messages,
        chat_output_items as chat_output_items,
        chat_tool as chat_tool,
        chat_tool_acc_from_message as chat_tool_acc_from_message,
        chat_tool_call_message as chat_tool_call_message,
        extract_responses_text as extract_responses_text,
        message_text as message_text,
    )
    from uv_agent.model.openai_sdk import openai_client as openai_client
    from uv_agent.model.responses import (
        create_responses_response as create_responses_response,
        parse_responses_response as parse_responses_response,
        responses_create_kwargs as responses_create_kwargs,
        responses_payload as responses_payload,
        stream_responses_response as stream_responses_response,
    )
    from uv_agent.model.types import (
        ModelClient as ModelClient,
        ModelResponse as ModelResponse,
        ModelStreamEvent as ModelStreamEvent,
        ToolCallDelta as ToolCallDelta,
    )
