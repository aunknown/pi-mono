"""
pi_agent_core - Python implementation of @mariozechner/pi-agent-core

Core agent loop, event streaming, and LLM client abstractions.
"""

from .agent import Agent, AgentOptions
from .agent_loop import AgentLoopOptions, agent_loop, agent_loop_continue
from .event_stream import EventStream
from .llm_client import StreamOptions, stream_anthropic
from .proxy import ProxyStreamOptions, stream_proxy
from .types import (
    AgentContext,
    AgentEvent,
    AgentMessage,
    AgentState,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    ContentBlock,
    ImageContent,
    TextContent,
    ThinkingContent,
    ThinkingLevel,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)

__all__ = [
    # Agent
    "Agent",
    "AgentOptions",
    # Agent loop
    "AgentLoopOptions",
    "agent_loop",
    "agent_loop_continue",
    # Streaming
    "EventStream",
    "StreamOptions",
    "stream_anthropic",
    # Proxy
    "ProxyStreamOptions",
    "stream_proxy",
    # Types
    "AgentContext",
    "AgentEvent",
    "AgentMessage",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "AssistantMessage",
    "ContentBlock",
    "ImageContent",
    "TextContent",
    "ThinkingContent",
    "ThinkingLevel",
    "ToolCall",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
]
