"""Tests for pi_agent_core module."""
import pytest
from pi_agent_core.types import (
    TextContent,
    ImageContent,
    ToolCall,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    AgentTool,
    AgentToolResult,
    AgentState,
)
from pi_agent_core.event_stream import EventStream


def test_text_content():
    tc = TextContent(text="hello")
    assert tc.type == "text"
    assert tc.text == "hello"


def test_user_message():
    msg = UserMessage(content=[TextContent(text="hello")])
    assert msg.role == "user"
    assert len(msg.content) == 1


def test_assistant_message():
    msg = AssistantMessage(content=[TextContent(text="hi")])
    assert msg.role == "assistant"
    assert msg.stop_reason == "stop"


def test_tool_result_message():
    msg = ToolResultMessage(
        tool_call_id="abc",
        tool_name="bash",
        content=[TextContent(text="output")],
    )
    assert msg.role == "toolResult"
    assert msg.is_error is False


def test_agent_state():
    state = AgentState()
    assert state.is_streaming is False
    assert state.messages == []
    assert state.tools == []


def test_event_stream_basic():
    """Test EventStream push/iterate."""
    import asyncio

    async def run():
        events = []

        class DoneEvent:
            type = "done"
            def __init__(self, val):
                self.val = val

        stream: EventStream[DoneEvent, int] = EventStream(
            is_done=lambda e: e.type == "done",
            get_result=lambda e: e.val if e.type == "done" else 0,
        )

        # Push a done event
        stream.push(DoneEvent(42))
        stream.end(42)

        async for event in stream:
            events.append(event)

        result = await stream.result()
        assert result == 42
        assert len(events) == 1

    asyncio.run(run())

