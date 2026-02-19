"""
Core types for pi-agent-core.

Mirrors the TypeScript types.ts from @mariozechner/pi-agent-core.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Content block types
# ---------------------------------------------------------------------------

class TextContent(BaseModel):
    """Plain text content block."""
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Base64-encoded image content block."""
    type: Literal["image"] = "image"
    data: str          # base64 encoded
    mime_type: str     # e.g. "image/png"


class ThinkingContent(BaseModel):
    """Reasoning/thinking content block (for models that support it)."""
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: Optional[str] = None


class ToolCall(BaseModel):
    """Tool call content block inside an assistant message."""
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


ContentBlock = Union[TextContent, ImageContent, ThinkingContent, ToolCall]


# ---------------------------------------------------------------------------
# Stop reasons
# ---------------------------------------------------------------------------

StopReason = Literal["stop", "length", "toolUse", "aborted", "error"]

# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class Usage(BaseModel):
    """Token usage statistics."""
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: dict[str, float] = Field(default_factory=lambda: {
        "input": 0.0,
        "output": 0.0,
        "cache_read": 0.0,
        "cache_write": 0.0,
        "total": 0.0,
    })


class UserMessage(BaseModel):
    """Message from the user."""
    role: Literal["user"] = "user"
    content: list[Union[TextContent, ImageContent]]
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class AssistantMessage(BaseModel):
    """Message from the LLM assistant."""
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    api: str = ""
    provider: str = ""
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    error_message: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class ToolResultMessage(BaseModel):
    """Result of a tool call execution."""
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[Union[TextContent, ImageContent]]
    details: Any = None
    is_error: bool = False
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


# Union of all message types
Message = Union[UserMessage, AssistantMessage, ToolResultMessage]

# AgentMessage is Message or any custom message type
AgentMessage = Any  # Extended by apps via custom types


# ---------------------------------------------------------------------------
# Thinking level
# ---------------------------------------------------------------------------

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]


# ---------------------------------------------------------------------------
# Tool types
# ---------------------------------------------------------------------------

class AgentToolResult(BaseModel):
    """Result returned by a tool execution."""
    content: list[Union[TextContent, ImageContent]]
    details: Any = None


AgentToolUpdateCallback = Callable[["AgentToolResult"], None]


class AgentTool(BaseModel):
    """A tool that the agent can use."""
    name: str
    label: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    # execute is a callable, not part of the Pydantic schema
    execute: Any = Field(exclude=True, default=None)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Agent context
# ---------------------------------------------------------------------------

class AgentContext(BaseModel):
    """Context passed to the agent loop for each LLM call."""
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    """Full agent state."""
    system_prompt: str = ""
    model: Optional[str] = None          # model identifier string e.g. "anthropic/claude-3-5-sonnet"
    provider: str = ""
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    is_streaming: bool = False
    stream_message: Optional[AgentMessage] = None
    pending_tool_calls: set[str] = Field(default_factory=set)
    error: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Agent events
# ---------------------------------------------------------------------------

class AgentStartEvent(BaseModel):
    type: Literal["agent_start"] = "agent_start"

class AgentEndEvent(BaseModel):
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage]

    model_config = {"arbitrary_types_allowed": True}

class TurnStartEvent(BaseModel):
    type: Literal["turn_start"] = "turn_start"

class TurnEndEvent(BaseModel):
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage
    tool_results: list[ToolResultMessage]

    model_config = {"arbitrary_types_allowed": True}

class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: AgentMessage

    model_config = {"arbitrary_types_allowed": True}

class MessageUpdateEvent(BaseModel):
    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    assistant_message_event: Any = None

    model_config = {"arbitrary_types_allowed": True}

class MessageEndEvent(BaseModel):
    type: Literal["message_end"] = "message_end"
    message: AgentMessage

    model_config = {"arbitrary_types_allowed": True}

class ToolExecutionStartEvent(BaseModel):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: Any

class ToolExecutionUpdateEvent(BaseModel):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any

class ToolExecutionEndEvent(BaseModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool


AgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
]
