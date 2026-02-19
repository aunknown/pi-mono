"""
Agent loop implementation.

Mirrors agent-loop.ts from @mariozechner/pi-agent-core.

The agent loop:
1. Takes initial prompt messages
2. Calls the LLM (via stream_fn)
3. Executes any tool calls in parallel
4. Checks for steering/follow-up messages
5. Loops until done
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Optional

from .event_stream import EventStream
from .types import (
    AgentContext,
    AgentEvent,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AgentEndEvent,
    AgentStartEvent,
    AssistantMessage,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    TextContent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResultMessage,
    TurnEndEvent,
    TurnStartEvent,
    UserMessage,
)
from .llm_client import (
    AssistantMessageEvent,
    StreamFn,
    StreamOptions,
    stream_anthropic,
)


def _create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    """Create the main agent event stream."""
    return EventStream(
        is_done=lambda e: e.type == "agent_end",
        get_result=lambda e: e.messages if e.type == "agent_end" else [],
    )


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    options: "AgentLoopOptions",
    signal: Optional[asyncio.Event] = None,
    stream_fn: Optional[StreamFn] = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """
    Start an agent loop with new prompt messages.

    Returns an EventStream that yields AgentEvents and resolves
    to the list of new messages added during this run.
    """
    stream = _create_agent_stream()

    async def run():
        new_messages: list[AgentMessage] = list(prompts)
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=list(context.messages) + list(prompts),
            tools=list(context.tools),
        )

        stream.push(AgentStartEvent())
        stream.push(TurnStartEvent())
        for prompt in prompts:
            stream.push(MessageStartEvent(message=prompt))
            stream.push(MessageEndEvent(message=prompt))

        await _run_loop(current_context, new_messages, options, signal, stream, stream_fn)

    asyncio.ensure_future(run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    options: "AgentLoopOptions",
    signal: Optional[asyncio.Event] = None,
    stream_fn: Optional[StreamFn] = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """
    Continue an agent loop from existing context.

    The last message in context must be a user/toolResult message.
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    last_msg = context.messages[-1]
    last_role = getattr(last_msg, "role", None)
    if last_role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = _create_agent_stream()

    async def run():
        new_messages: list[AgentMessage] = []
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=list(context.messages),
            tools=list(context.tools),
        )

        stream.push(AgentStartEvent())
        stream.push(TurnStartEvent())

        await _run_loop(current_context, new_messages, options, signal, stream, stream_fn)

    asyncio.ensure_future(run())
    return stream


class AgentLoopOptions:
    """Configuration for the agent loop."""

    def __init__(
        self,
        stream_options: StreamOptions,
        convert_to_llm: Optional[Callable[[list[AgentMessage]], list[AgentMessage]]] = None,
        transform_context: Optional[Callable[
            [list[AgentMessage], Optional[asyncio.Event]],
            "asyncio.Coroutine[Any, Any, list[AgentMessage]]"
        ]] = None,
        get_api_key: Optional[Callable[[str], "asyncio.Coroutine[Any, Any, Optional[str]]"]] = None,
        get_steering_messages: Optional[Callable[
            [],
            "asyncio.Coroutine[Any, Any, list[AgentMessage]]"
        ]] = None,
        get_follow_up_messages: Optional[Callable[
            [],
            "asyncio.Coroutine[Any, Any, list[AgentMessage]]"
        ]] = None,
    ):
        self.stream_options = stream_options
        self.convert_to_llm = convert_to_llm or _default_convert_to_llm
        self.transform_context = transform_context
        self.get_api_key = get_api_key
        self.get_steering_messages = get_steering_messages
        self.get_follow_up_messages = get_follow_up_messages


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[AgentMessage]:
    """Default: keep only LLM-compatible messages."""
    result = []
    for m in messages:
        role = getattr(m, "role", None)
        if role in ("user", "assistant", "toolResult"):
            result.append(m)
    return result


async def _run_loop(
    context: AgentContext,
    new_messages: list[AgentMessage],
    options: AgentLoopOptions,
    signal: Optional[asyncio.Event],
    stream: EventStream[AgentEvent, list[AgentMessage]],
    stream_fn: Optional[StreamFn],
) -> None:
    """Main loop logic shared by agent_loop and agent_loop_continue."""
    first_turn = True
    pending_messages: list[AgentMessage] = []
    if options.get_steering_messages:
        pending_messages = await options.get_steering_messages()

    # Outer loop: continues for follow-up messages
    while True:
        has_more_tool_calls = True
        steering_after_tools: Optional[list[AgentMessage]] = None

        # Inner loop: process tool calls and steering messages
        while has_more_tool_calls or pending_messages:
            if not first_turn:
                stream.push(TurnStartEvent())
            else:
                first_turn = False

            # Inject pending messages before next LLM call
            if pending_messages:
                for msg in pending_messages:
                    stream.push(MessageStartEvent(message=msg))
                    stream.push(MessageEndEvent(message=msg))
                    context.messages.append(msg)
                    new_messages.append(msg)
                pending_messages = []

            # Stream assistant response
            message = await _stream_assistant_response(
                context, options, signal, stream, stream_fn
            )
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                stream.push(TurnEndEvent(message=message, tool_results=[]))
                stream.push(AgentEndEvent(messages=new_messages))
                stream.end(new_messages)
                return

            # Check for tool calls
            tool_calls = [b for b in message.content if isinstance(b, ToolCall)]
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                execution = await _execute_tool_calls(
                    context.tools, message, signal, stream,
                    options.get_steering_messages,
                )
                tool_results = execution["tool_results"]
                steering_after_tools = execution.get("steering_messages")

                for result in tool_results:
                    context.messages.append(result)
                    new_messages.append(result)

            stream.push(TurnEndEvent(message=message, tool_results=tool_results))

            # Get steering messages after turn
            if steering_after_tools:
                pending_messages = steering_after_tools
                steering_after_tools = None
            elif options.get_steering_messages:
                pending_messages = await options.get_steering_messages()

        # Agent would stop - check for follow-up messages
        if options.get_follow_up_messages:
            follow_up = await options.get_follow_up_messages()
            if follow_up:
                pending_messages = follow_up
                continue

        break

    stream.push(AgentEndEvent(messages=new_messages))
    stream.end(new_messages)


async def _stream_assistant_response(
    context: AgentContext,
    options: AgentLoopOptions,
    signal: Optional[asyncio.Event],
    stream: EventStream[AgentEvent, list[AgentMessage]],
    stream_fn: Optional[StreamFn],
) -> AssistantMessage:
    """Stream an assistant response from the LLM."""
    # Apply context transform
    messages = context.messages
    if options.transform_context:
        messages = await options.transform_context(messages, signal)

    # Convert to LLM-compatible messages
    llm_messages = options.convert_to_llm(messages)

    # Build context for LLM
    llm_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=context.tools,
    )

    # Resolve API key
    stream_opts = options.stream_options
    if options.get_api_key:
        resolved_key = await options.get_api_key(stream_opts.provider)
        if resolved_key:
            # Create a copy of options with resolved key
            import dataclasses
            stream_opts = StreamOptions(
                api_key=resolved_key,
                base_url=stream_opts.base_url,
                model_id=stream_opts.model_id,
                provider=stream_opts.provider,
                reasoning=stream_opts.reasoning,
                temperature=stream_opts.temperature,
                max_tokens=stream_opts.max_tokens,
                signal=stream_opts.signal,
                session_id=stream_opts.session_id,
            )

    fn = stream_fn or stream_anthropic
    response = fn(llm_context, stream_opts)

    partial_message: Optional[AssistantMessage] = None
    added_partial = False

    async for event in response:
        etype = event.type

        if etype == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            stream.push(MessageStartEvent(message=AssistantMessage(**partial_message.model_dump())))

        elif etype in (
            "text_start", "text_delta", "text_end",
            "thinking_start", "thinking_delta", "thinking_end",
            "toolcall_start", "toolcall_delta", "toolcall_end",
        ):
            if partial_message is not None:
                partial_message = event.partial
                if added_partial:
                    context.messages[-1] = partial_message
                stream.push(MessageUpdateEvent(
                    message=AssistantMessage(**partial_message.model_dump()),
                    assistant_message_event=event,
                ))

        elif etype in ("done", "error"):
            final_message = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                stream.push(MessageStartEvent(message=final_message))
            stream.push(MessageEndEvent(message=final_message))
            return final_message

    return await response.result()


async def _execute_tool_calls(
    tools: list[AgentTool],
    assistant_message: AssistantMessage,
    signal: Optional[asyncio.Event],
    stream: EventStream[AgentEvent, list[AgentMessage]],
    get_steering_messages: Optional[Callable],
) -> dict:
    """Execute tool calls from an assistant message."""
    tool_calls = [b for b in assistant_message.content if isinstance(b, ToolCall)]
    results: list[ToolResultMessage] = []
    steering_messages: Optional[list[AgentMessage]] = None

    for i, tool_call in enumerate(tool_calls):
        tool = next((t for t in tools if t.name == tool_call.name), None)

        stream.push(ToolExecutionStartEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
        ))

        result: AgentToolResult
        is_error = False

        try:
            if not tool:
                raise ValueError(f"Tool {tool_call.name} not found")

            # Validate and execute tool
            def make_update_callback(tc_id, tc_name, tc_args):
                def callback(partial_result: AgentToolResult):
                    stream.push(ToolExecutionUpdateEvent(
                        tool_call_id=tc_id,
                        tool_name=tc_name,
                        args=tc_args,
                        partial_result=partial_result,
                    ))
                return callback

            update_cb = make_update_callback(
                tool_call.id, tool_call.name, tool_call.arguments
            )

            if asyncio.iscoroutinefunction(tool.execute):
                result = await tool.execute(
                    tool_call.id,
                    tool_call.arguments,
                    signal,
                    update_cb,
                )
            else:
                result = tool.execute(
                    tool_call.id,
                    tool_call.arguments,
                    signal,
                    update_cb,
                )

        except Exception as e:
            result = AgentToolResult(
                content=[TextContent(text=str(e))],
                details={},
            )
            is_error = True

        stream.push(ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
            is_error=is_error,
        ))

        tool_result_msg = ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=result.content,
            details=result.details,
            is_error=is_error,
        )
        results.append(tool_result_msg)
        stream.push(MessageStartEvent(message=tool_result_msg))
        stream.push(MessageEndEvent(message=tool_result_msg))

        # Check for steering messages after each tool
        if get_steering_messages:
            steering = await get_steering_messages()
            if steering:
                steering_messages = steering
                # Skip remaining tool calls
                remaining = tool_calls[i + 1:]
                for skipped in remaining:
                    results.append(_skip_tool_call(skipped, stream))
                break

    return {"tool_results": results, "steering_messages": steering_messages}


def _skip_tool_call(
    tool_call: ToolCall,
    stream: EventStream[AgentEvent, list[AgentMessage]],
) -> ToolResultMessage:
    """Create a skipped tool result for a tool call that was not executed."""
    result = AgentToolResult(
        content=[TextContent(text="Skipped due to queued user message.")],
        details={},
    )

    stream.push(ToolExecutionStartEvent(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        args=tool_call.arguments,
    ))
    stream.push(ToolExecutionEndEvent(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        result=result,
        is_error=True,
    ))

    tool_result_msg = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details={},
        is_error=True,
    )
    stream.push(MessageStartEvent(message=tool_result_msg))
    stream.push(MessageEndEvent(message=tool_result_msg))

    return tool_result_msg
