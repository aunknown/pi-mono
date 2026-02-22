"""
Proxy stream function for apps that route LLM calls through a server.

Mirrors proxy.ts from @mariozechner/pi-agent-core.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

import httpx

from .event_stream import EventStream
from .llm_client import (
    AssistantMessageEvent,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    StreamOptions,
)
from .types import (
    AgentContext,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
)


class ProxyStreamOptions(StreamOptions):
    """Options for the proxy stream function."""

    def __init__(
        self,
        auth_token: str,
        proxy_url: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.auth_token = auth_token
        self.proxy_url = proxy_url


def stream_proxy(
    context: AgentContext,
    options: ProxyStreamOptions,
) -> EventStream[AssistantMessageEvent, AssistantMessage]:
    """
    Stream function that proxies through a server instead of calling LLM providers directly.

    The server manages auth and proxies requests to LLM providers.
    Use this as the stream_fn option when creating an Agent that needs to go through a proxy.

    Example::

        from pi_agent_core.proxy import stream_proxy, ProxyStreamOptions

        def my_stream_fn(context, opts):
            return stream_proxy(context, ProxyStreamOptions(
                auth_token="my-token",
                proxy_url="https://genai.example.com",
                model_id=opts.model_id,
                provider=opts.provider,
            ))

        agent = Agent(AgentOptions(stream_fn=my_stream_fn))
    """
    stream: EventStream[AssistantMessageEvent, AssistantMessage] = EventStream(
        is_done=lambda e: e.type in ("done", "error"),
        get_result=lambda e: (
            e.message if e.type == "done" else
            e.error if e.type == "error" else
            AssistantMessage(content=[], stop_reason="error")
        ),
    )

    asyncio.create_task(_run_proxy_stream(context, options, stream))
    return stream


async def _run_proxy_stream(
    context: AgentContext,
    options: ProxyStreamOptions,
    stream: EventStream[AssistantMessageEvent, AssistantMessage],
) -> None:
    """Run the proxy streaming request."""
    partial = AssistantMessage(
        content=[],
        api="messages",
        provider=options.provider,
        model=options.model_id,
        stop_reason="stop",
    )

    try:
        # Serialize context for the proxy
        context_data = {
            "system_prompt": context.system_prompt,
            "messages": [_serialize_message(m) for m in context.messages],
            "tools": [_serialize_tool(t) for t in context.tools] if context.tools else [],
        }

        request_body = {
            "model": {"id": options.model_id, "provider": options.provider},
            "context": context_data,
            "options": {
                "reasoning": options.reasoning,
                "temperature": options.temperature,
                "max_tokens": options.max_tokens,
            },
        }

        headers = {
            "Authorization": f"Bearer {options.auth_token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{options.proxy_url}/api/stream",
                headers=headers,
                json=request_body,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    try:
                        error_data = json.loads(body)
                        error_msg = f"Proxy error: {error_data.get('error', body.decode())}"
                    except Exception:
                        error_msg = f"Proxy error: {response.status_code} {body.decode()}"
                    raise RuntimeError(error_msg)

                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    lines = buffer.split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if not data_str:
                            continue

                        try:
                            proxy_event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event = _process_proxy_event(proxy_event, partial)
                        if event:
                            stream.push(event)

        # Ensure a DoneEvent is pushed if the proxy didn't send one
        if partial.stop_reason != "error":
            stream.push(DoneEvent(partial))
        stream.end(partial)

    except Exception as e:
        error_msg = str(e)
        partial.stop_reason = "error"
        partial.error_message = error_msg
        stream.push(ErrorEvent(partial))
        stream.end(partial)


def _process_proxy_event(
    proxy_event: dict,
    partial: AssistantMessage,
) -> Optional[AssistantMessageEvent]:
    """Process a proxy event and update the partial message."""
    event_type = proxy_event.get("type", "")

    if event_type == "start":
        return StartEvent(partial)

    elif event_type == "text_start":
        idx = proxy_event["contentIndex"]
        # Extend content list if needed
        while len(partial.content) <= idx:
            partial.content.append(TextContent(text=""))
        partial.content[idx] = TextContent(text="")
        return TextStartEvent(idx, partial)

    elif event_type == "text_delta":
        idx = proxy_event["contentIndex"]
        if idx >= len(partial.content):
            return None
        delta = proxy_event["delta"]
        block = partial.content[idx]
        if isinstance(block, TextContent):
            block.text += delta
        return TextDeltaEvent(idx, delta, partial)

    elif event_type == "text_end":
        idx = proxy_event["contentIndex"]
        if idx >= len(partial.content):
            return None
        block = partial.content[idx]
        if isinstance(block, TextContent):
            return TextEndEvent(idx, block.text, partial)

    elif event_type == "thinking_start":
        idx = proxy_event["contentIndex"]
        while len(partial.content) <= idx:
            partial.content.append(ThinkingContent(thinking=""))
        partial.content[idx] = ThinkingContent(thinking="")
        return ThinkingStartEvent(idx, partial)

    elif event_type == "thinking_delta":
        idx = proxy_event["contentIndex"]
        if idx >= len(partial.content):
            return None
        delta = proxy_event["delta"]
        block = partial.content[idx]
        if isinstance(block, ThinkingContent):
            block.thinking += delta
        return ThinkingDeltaEvent(idx, delta, partial)

    elif event_type == "thinking_end":
        idx = proxy_event["contentIndex"]
        if idx >= len(partial.content):
            return None
        block = partial.content[idx]
        if isinstance(block, ThinkingContent):
            return ThinkingEndEvent(idx, block.thinking, partial)

    elif event_type == "toolcall_start":
        idx = proxy_event["contentIndex"]
        while len(partial.content) <= idx:
            partial.content.append(ToolCall(id="", name="", arguments={}))
        partial.content[idx] = ToolCall(
            id=proxy_event["id"],
            name=proxy_event["toolName"],
            arguments={},
        )
        return ToolCallStartEvent(idx, partial)

    elif event_type == "toolcall_delta":
        idx = proxy_event["contentIndex"]
        if idx >= len(partial.content):
            return None
        delta = proxy_event["delta"]
        block = partial.content[idx]
        if isinstance(block, ToolCall):
            # Try to parse accumulated JSON
            try:
                if not hasattr(block, "_partial_json"):
                    block._partial_json = ""
                block._partial_json += delta
                block.arguments = json.loads(block._partial_json)
            except json.JSONDecodeError:
                pass
        return ToolCallDeltaEvent(idx, delta, partial)

    elif event_type == "toolcall_end":
        idx = proxy_event["contentIndex"]
        if idx >= len(partial.content):
            return None
        block = partial.content[idx]
        if isinstance(block, ToolCall):
            return ToolCallEndEvent(idx, block, partial)

    elif event_type == "done":
        reason = proxy_event.get("reason", "stop")
        stop_reason_map = {"stop": "stop", "length": "length", "toolUse": "toolUse"}
        partial.stop_reason = stop_reason_map.get(reason, "stop")
        usage_data = proxy_event.get("usage", {})
        if usage_data:
            partial.usage = Usage(
                input=usage_data.get("input", 0),
                output=usage_data.get("output", 0),
                cache_read=usage_data.get("cacheRead", 0),
                cache_write=usage_data.get("cacheWrite", 0),
                total_tokens=usage_data.get("totalTokens", 0),
            )
        return DoneEvent(partial)

    elif event_type == "error":
        reason = proxy_event.get("reason", "error")
        partial.stop_reason = reason
        partial.error_message = proxy_event.get("errorMessage")
        return ErrorEvent(partial)

    return None


def _serialize_message(msg: Any) -> dict:
    """Serialize an AgentMessage to dict for the proxy."""
    role = getattr(msg, "role", None)
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    return msg if isinstance(msg, dict) else {}


def _serialize_tool(tool: Any) -> dict:
    """Serialize an AgentTool to dict for the proxy."""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
