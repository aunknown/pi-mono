"""
LLM client abstraction for pi-agent-core.

Provides a StreamFn protocol and a default httpx-based implementation
that speaks the Anthropic Messages API (SSE streaming).

For other providers, pass a custom `stream_fn` to Agent.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Callable, Optional, Protocol

import httpx

from .event_stream import EventStream
from .types import (
    AssistantMessage,
    ContentBlock,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    AgentContext,
    AgentTool,
)


# ---------------------------------------------------------------------------
# Stream event types (mirrors pi-ai AssistantMessageEvent)
# ---------------------------------------------------------------------------

class StartEvent:
    type = "start"
    def __init__(self, partial: AssistantMessage):
        self.partial = partial

class TextStartEvent:
    type = "text_start"
    def __init__(self, content_index: int, partial: AssistantMessage):
        self.content_index = content_index
        self.partial = partial

class TextDeltaEvent:
    type = "text_delta"
    def __init__(self, content_index: int, delta: str, partial: AssistantMessage):
        self.content_index = content_index
        self.delta = delta
        self.partial = partial

class TextEndEvent:
    type = "text_end"
    def __init__(self, content_index: int, content: str, partial: AssistantMessage):
        self.content_index = content_index
        self.content = content
        self.partial = partial

class ThinkingStartEvent:
    type = "thinking_start"
    def __init__(self, content_index: int, partial: AssistantMessage):
        self.content_index = content_index
        self.partial = partial

class ThinkingDeltaEvent:
    type = "thinking_delta"
    def __init__(self, content_index: int, delta: str, partial: AssistantMessage):
        self.content_index = content_index
        self.delta = delta
        self.partial = partial

class ThinkingEndEvent:
    type = "thinking_end"
    def __init__(self, content_index: int, content: str, partial: AssistantMessage):
        self.content_index = content_index
        self.content = content
        self.partial = partial

class ToolCallStartEvent:
    type = "toolcall_start"
    def __init__(self, content_index: int, partial: AssistantMessage):
        self.content_index = content_index
        self.partial = partial

class ToolCallDeltaEvent:
    type = "toolcall_delta"
    def __init__(self, content_index: int, delta: str, partial: AssistantMessage):
        self.content_index = content_index
        self.delta = delta
        self.partial = partial

class ToolCallEndEvent:
    type = "toolcall_end"
    def __init__(self, content_index: int, tool_call: ToolCall, partial: AssistantMessage):
        self.content_index = content_index
        self.tool_call = tool_call
        self.partial = partial

class DoneEvent:
    type = "done"
    def __init__(self, message: AssistantMessage):
        self.message = message

class ErrorEvent:
    type = "error"
    def __init__(self, error: AssistantMessage):
        self.error = error


AssistantMessageEvent = (
    StartEvent | TextStartEvent | TextDeltaEvent | TextEndEvent |
    ThinkingStartEvent | ThinkingDeltaEvent | ThinkingEndEvent |
    ToolCallStartEvent | ToolCallDeltaEvent | ToolCallEndEvent |
    DoneEvent | ErrorEvent
)


# ---------------------------------------------------------------------------
# StreamFn protocol
# ---------------------------------------------------------------------------

class StreamOptions:
    """Options passed to the stream function."""
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com",
        model_id: str = "claude-3-5-sonnet-20241022",
        provider: str = "anthropic",
        reasoning: Optional[str] = None,  # ThinkingLevel != "off"
        temperature: Optional[float] = None,
        max_tokens: int = 8192,
        signal: Optional[Any] = None,     # asyncio.Event for cancellation
        session_id: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self.provider = provider
        self.reasoning = reasoning
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.signal = signal
        self.session_id = session_id


StreamFn = Callable[
    [AgentContext, StreamOptions],
    "EventStream[AssistantMessageEvent, AssistantMessage]",
]


# ---------------------------------------------------------------------------
# Anthropic streaming implementation
# ---------------------------------------------------------------------------

def _build_anthropic_messages(context: AgentContext) -> list[dict]:
    """Convert AgentContext messages to Anthropic API format."""
    result = []
    for msg in context.messages:
        if hasattr(msg, "role"):
            role = msg.role
        else:
            continue

        if role == "user":
            content_parts = []
            if isinstance(msg, UserMessage):
                for block in msg.content:
                    if isinstance(block, TextContent):
                        content_parts.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageContent):
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.mime_type,
                                "data": block.data,
                            },
                        })
            else:
                # Fallback: treat as dict
                content_parts = msg.get("content", []) if isinstance(msg, dict) else []
            result.append({"role": "user", "content": content_parts})

        elif role == "assistant":
            if isinstance(msg, AssistantMessage):
                content_parts = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        content_parts.append({"type": "text", "text": block.text})
                    elif isinstance(block, ThinkingContent):
                        content_parts.append({
                            "type": "thinking",
                            "thinking": block.thinking,
                        })
                    elif isinstance(block, ToolCall):
                        content_parts.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.arguments,
                        })
                result.append({"role": "assistant", "content": content_parts})

        elif role == "toolResult":
            if isinstance(msg, ToolResultMessage):
                content_parts = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        content_parts.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageContent):
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.mime_type,
                                "data": block.data,
                            },
                        })
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": content_parts,
                        "is_error": msg.is_error,
                    }],
                })

    return result


def _build_anthropic_tools(tools: list[AgentTool]) -> list[dict]:
    """Convert AgentTool list to Anthropic API tool format."""
    result = []
    for tool in tools:
        result.append({
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        })
    return result


def stream_anthropic(
    context: AgentContext,
    options: StreamOptions,
) -> "EventStream[AssistantMessageEvent, AssistantMessage]":
    """
    Stream assistant response from Anthropic API.

    Returns an EventStream that yields AssistantMessageEvents and
    resolves to the final AssistantMessage.
    """
    from .event_stream import EventStream

    stream: EventStream[AssistantMessageEvent, AssistantMessage] = EventStream(
        is_done=lambda e: e.type in ("done", "error"),
        get_result=lambda e: (
            e.message if e.type == "done" else
            e.error if e.type == "error" else
            AssistantMessage(content=[], stop_reason="error")
        ),
    )

    import asyncio
    asyncio.ensure_future(_run_anthropic_stream(context, options, stream))
    return stream


async def _run_anthropic_stream(
    context: AgentContext,
    options: StreamOptions,
    stream: "EventStream[AssistantMessageEvent, AssistantMessage]",
) -> None:
    """Run the Anthropic streaming request and push events to the stream."""
    partial = AssistantMessage(
        content=[],
        api="messages",
        provider="anthropic",
        model=options.model_id,
        stop_reason="stop",
    )

    try:
        messages = _build_anthropic_messages(context)
        tools_payload = _build_anthropic_tools(context.tools) if context.tools else []

        request_body: dict = {
            "model": options.model_id,
            "max_tokens": options.max_tokens,
            "system": context.system_prompt,
            "messages": messages,
            "stream": True,
        }
        if tools_payload:
            request_body["tools"] = tools_payload
        if options.temperature is not None:
            request_body["temperature"] = options.temperature
        if options.reasoning and options.reasoning != "off":
            # Map thinking level to budget tokens
            budget_map = {
                "minimal": 1024,
                "low": 4096,
                "medium": 8192,
                "high": 16384,
                "xhigh": 32768,
            }
            budget = budget_map.get(options.reasoning, 4096)
            request_body["thinking"] = {"type": "enabled", "budget_tokens": budget}

        headers = {
            "x-api-key": options.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{options.base_url}/v1/messages",
                headers=headers,
                json=request_body,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise RuntimeError(
                        f"Anthropic API error {response.status_code}: {body.decode()}"
                    )

                stream.push(StartEvent(partial))

                content_index_map: dict[int, int] = {}  # API index -> content list index
                partial_json: dict[int, str] = {}       # content_index -> partial JSON string

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("type", "")

                    if event_type == "content_block_start":
                        idx = data["index"]
                        block = data["content_block"]
                        list_idx = len(partial.content)
                        content_index_map[idx] = list_idx

                        if block["type"] == "text":
                            partial.content.append(TextContent(text=""))
                            stream.push(TextStartEvent(list_idx, partial))
                        elif block["type"] == "thinking":
                            partial.content.append(ThinkingContent(thinking=""))
                            stream.push(ThinkingStartEvent(list_idx, partial))
                        elif block["type"] == "tool_use":
                            partial.content.append(ToolCall(
                                id=block["id"],
                                name=block["name"],
                                arguments={},
                            ))
                            partial_json[list_idx] = ""
                            stream.push(ToolCallStartEvent(list_idx, partial))

                    elif event_type == "content_block_delta":
                        idx = data["index"]
                        list_idx = content_index_map.get(idx, 0)
                        delta = data["delta"]

                        if delta["type"] == "text_delta":
                            d = delta["text"]
                            block = partial.content[list_idx]
                            if isinstance(block, TextContent):
                                block.text += d
                            stream.push(TextDeltaEvent(list_idx, d, partial))
                        elif delta["type"] == "thinking_delta":
                            d = delta["thinking"]
                            block = partial.content[list_idx]
                            if isinstance(block, ThinkingContent):
                                block.thinking += d
                            stream.push(ThinkingDeltaEvent(list_idx, d, partial))
                        elif delta["type"] == "input_json_delta":
                            d = delta["partial_json"]
                            partial_json[list_idx] = partial_json.get(list_idx, "") + d
                            # Try to parse partial JSON
                            try:
                                block = partial.content[list_idx]
                                if isinstance(block, ToolCall):
                                    block.arguments = json.loads(partial_json[list_idx])
                            except json.JSONDecodeError:
                                pass
                            stream.push(ToolCallDeltaEvent(list_idx, d, partial))

                    elif event_type == "content_block_stop":
                        idx = data["index"]
                        list_idx = content_index_map.get(idx, 0)
                        block = partial.content[list_idx]

                        if isinstance(block, TextContent):
                            stream.push(TextEndEvent(list_idx, block.text, partial))
                        elif isinstance(block, ThinkingContent):
                            stream.push(ThinkingEndEvent(list_idx, block.thinking, partial))
                        elif isinstance(block, ToolCall):
                            # Finalize JSON
                            try:
                                block.arguments = json.loads(partial_json.get(list_idx, "{}"))
                            except json.JSONDecodeError:
                                block.arguments = {}
                            stream.push(ToolCallEndEvent(list_idx, block, partial))

                    elif event_type == "message_delta":
                        delta = data.get("delta", {})
                        stop_reason = delta.get("stop_reason")
                        if stop_reason == "tool_use":
                            partial.stop_reason = "toolUse"
                        elif stop_reason == "end_turn":
                            partial.stop_reason = "stop"
                        elif stop_reason == "max_tokens":
                            partial.stop_reason = "length"

                        usage = data.get("usage", {})
                        if usage:
                            partial.usage.output = usage.get("output_tokens", 0)

                    elif event_type == "message_start":
                        msg_data = data.get("message", {})
                        usage = msg_data.get("usage", {})
                        if usage:
                            partial.usage.input = usage.get("input_tokens", 0)
                            partial.usage.cache_read = usage.get("cache_read_input_tokens", 0)
                            partial.usage.cache_write = usage.get("cache_creation_input_tokens", 0)

                    elif event_type == "message_stop":
                        pass  # Final event, handled by message_delta

                stream.push(DoneEvent(partial))
                stream.end(partial)

    except Exception as e:
        partial.stop_reason = "error"
        partial.error_message = str(e)
        stream.push(ErrorEvent(partial))
        stream.end(partial)
