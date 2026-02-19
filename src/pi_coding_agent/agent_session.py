"""
AgentSession - Core abstraction for agent lifecycle and session management.

Mirrors agent-session.ts from pi-coding-agent (core subset).

This class is shared between all run modes. It encapsulates:
- Agent state access
- Event subscription with automatic session persistence
- Model management
- Session switching

Modes use this class and add their own I/O layer on top.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pi_agent_core.agent import Agent, AgentOptions
from pi_agent_core.llm_client import StreamOptions
from pi_agent_core.types import (
    AgentEvent,
    AgentMessage,
    AgentTool,
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingLevel,
    ToolResultMessage,
    UserMessage,
)

from .session_manager import SessionManager, SessionContext
from .tools import create_coding_tools, create_all_tools


# ---------------------------------------------------------------------------
# Session event types
# ---------------------------------------------------------------------------

@dataclass
class AutoCompactionStartEvent:
    type: str = "auto_compaction_start"
    reason: str = "threshold"


@dataclass
class AutoCompactionEndEvent:
    type: str = "auto_compaction_end"
    aborted: bool = False
    error_message: Optional[str] = None


@dataclass
class AutoRetryStartEvent:
    type: str = "auto_retry_start"
    attempt: int = 0
    max_attempts: int = 3
    delay_ms: int = 1000
    error_message: str = ""


@dataclass
class AutoRetryEndEvent:
    type: str = "auto_retry_end"
    success: bool = False
    attempt: int = 0
    final_error: Optional[str] = None


AgentSessionEvent = Any  # AgentEvent | Auto*Event


# ---------------------------------------------------------------------------
# AgentSession
# ---------------------------------------------------------------------------

@dataclass
class AgentSessionConfig:
    """Configuration for an AgentSession."""
    cwd: str
    stream_options: Optional[StreamOptions] = None
    system_prompt: str = ""
    initial_tools: Optional[list[str]] = None  # Tool names to enable
    sessions_dir: Optional[str] = None
    session_file: Optional[str] = None
    session_id: Optional[str] = None


class AgentSession:
    """
    High-level session abstraction that combines Agent + SessionManager.

    Handles:
    - Session persistence (messages saved on message_end)
    - Model/thinking level management
    - Tool management
    - Event routing
    """

    def __init__(self, config: AgentSessionConfig) -> None:
        self._cwd = config.cwd

        # Create agent
        stream_opts = config.stream_options or StreamOptions()
        self._agent = Agent(AgentOptions(stream_options=stream_opts))
        self._agent.set_system_prompt(config.system_prompt)

        # Create session manager
        self._session_manager = SessionManager(
            sessions_dir=config.sessions_dir,
            session_id=config.session_id,
            session_file=config.session_file,
        )

        # Set up tools
        all_tools = create_all_tools(self._cwd)
        tool_names = config.initial_tools or ["read", "bash", "edit", "write"]
        active_tools = [all_tools[name] for name in tool_names if name in all_tools]
        self._agent.set_tools(active_tools)
        self._tool_registry = all_tools

        # Event listeners
        self._event_listeners: list[Callable[[AgentSessionEvent], None]] = []

        # Subscribe to agent events for session persistence
        self._unsubscribe_agent = self._agent.subscribe(self._handle_agent_event)

        # Queue management
        self._steering_messages: list[str] = []
        self._follow_up_messages: list[str] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def is_streaming(self) -> bool:
        return self._agent.state.is_streaming

    @property
    def messages(self) -> list[AgentMessage]:
        return self._agent.state.messages

    @property
    def session_id(self) -> str:
        return self._session_manager.get_session_id()

    @property
    def session_file(self) -> Optional[str]:
        return self._session_manager.get_session_file()

    @property
    def thinking_level(self) -> str:
        return self._agent.state.thinking_level

    @property
    def pending_message_count(self) -> int:
        return len(self._steering_messages) + len(self._follow_up_messages)

    # -------------------------------------------------------------------------
    # Event subscription
    # -------------------------------------------------------------------------

    def subscribe(self, listener: Callable[[AgentSessionEvent], None]) -> Callable[[], None]:
        """Subscribe to session events. Returns an unsubscribe function."""
        self._event_listeners.append(listener)

        def unsubscribe():
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)

        return unsubscribe

    def _emit(self, event: AgentSessionEvent) -> None:
        """Emit an event to all listeners."""
        for listener in list(self._event_listeners):
            listener(event)

    def _handle_agent_event(self, event: AgentEvent) -> None:
        """Handle agent events: persist messages and forward to listeners."""
        etype = event.type

        # Persist messages on message_end
        if etype == "message_end":
            msg = event.message
            role = getattr(msg, "role", None)
            if role in ("user", "assistant", "toolResult"):
                self._session_manager.append_message(msg)

            # Remove from steering/follow-up display queues
            if role == "user":
                text = self._get_message_text(msg)
                if text in self._steering_messages:
                    self._steering_messages.remove(text)
                elif text in self._follow_up_messages:
                    self._follow_up_messages.remove(text)

        # Forward to external listeners
        self._emit(event)

    def _get_message_text(self, message: AgentMessage) -> str:
        """Extract text from a user message."""
        if not isinstance(message, UserMessage):
            return ""
        texts = [b.text for b in message.content if isinstance(b, TextContent)]
        return "".join(texts)

    def dispose(self) -> None:
        """Clean up the session."""
        if self._unsubscribe_agent:
            self._unsubscribe_agent()
            self._unsubscribe_agent = None
        self._event_listeners.clear()

    # -------------------------------------------------------------------------
    # Prompting
    # -------------------------------------------------------------------------

    async def prompt(
        self,
        text: str,
        images: Optional[list[ImageContent]] = None,
        streaming_behavior: Optional[str] = None,
    ) -> None:
        """
        Send a prompt to the agent.

        Args:
            text: The prompt text
            images: Optional image attachments
            streaming_behavior: When streaming, "steer" or "followUp" to queue
        """
        if self.is_streaming:
            if not streaming_behavior:
                raise RuntimeError(
                    "Agent is already processing. "
                    "Specify streaming_behavior ('steer' or 'followUp') to queue."
                )
            if streaming_behavior == "followUp":
                await self._queue_follow_up(text, images)
            else:
                await self._queue_steer(text, images)
            return

        content: list = [TextContent(text=text)]
        if images:
            content.extend(images)

        messages = [UserMessage(content=content)]
        await self._agent.prompt(messages)

    async def steer(
        self,
        text: str,
        images: Optional[list[ImageContent]] = None,
    ) -> None:
        """Queue a steering message to interrupt the agent mid-run."""
        await self._queue_steer(text, images)

    async def follow_up(
        self,
        text: str,
        images: Optional[list[ImageContent]] = None,
    ) -> None:
        """Queue a follow-up message for after the agent finishes."""
        await self._queue_follow_up(text, images)

    async def _queue_steer(
        self,
        text: str,
        images: Optional[list[ImageContent]] = None,
    ) -> None:
        self._steering_messages.append(text)
        content: list = [TextContent(text=text)]
        if images:
            content.extend(images)
        self._agent.steer(UserMessage(content=content))

    async def _queue_follow_up(
        self,
        text: str,
        images: Optional[list[ImageContent]] = None,
    ) -> None:
        self._follow_up_messages.append(text)
        content: list = [TextContent(text=text)]
        if images:
            content.extend(images)
        self._agent.follow_up(UserMessage(content=content))

    def get_steering_messages(self) -> list[str]:
        """Get pending steering messages (read-only)."""
        return list(self._steering_messages)

    def get_follow_up_messages(self) -> list[str]:
        """Get pending follow-up messages (read-only)."""
        return list(self._follow_up_messages)

    def clear_queue(self) -> dict:
        """Clear all queued messages and return them."""
        steering = list(self._steering_messages)
        follow_up = list(self._follow_up_messages)
        self._steering_messages.clear()
        self._follow_up_messages.clear()
        self._agent.clear_all_queues()
        return {"steering": steering, "follow_up": follow_up}

    # -------------------------------------------------------------------------
    # Abort / wait
    # -------------------------------------------------------------------------

    def abort(self) -> None:
        """Abort the current operation."""
        self._agent.abort()

    async def wait_for_idle(self) -> None:
        """Wait until the agent is no longer streaming."""
        await self._agent.wait_for_idle()

    async def abort_and_wait(self) -> None:
        """Abort and wait for the agent to become idle."""
        self.abort()
        await self.wait_for_idle()

    # -------------------------------------------------------------------------
    # Tool management
    # -------------------------------------------------------------------------

    def get_active_tool_names(self) -> list[str]:
        """Get names of currently active tools."""
        return [t.name for t in self._agent.state.tools]

    def set_active_tools_by_name(self, tool_names: list[str]) -> None:
        """Set active tools by name."""
        tools = []
        for name in tool_names:
            tool = self._tool_registry.get(name)
            if tool:
                tools.append(tool)
        self._agent.set_tools(tools)

    def get_all_tools(self) -> list[AgentTool]:
        """Get all available tools."""
        return list(self._tool_registry.values())

    # -------------------------------------------------------------------------
    # Thinking level
    # -------------------------------------------------------------------------

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        """Set the thinking level."""
        self._agent.set_thinking_level(level)
        self._session_manager.append_thinking_level_change(level)

    def cycle_thinking_level(self) -> ThinkingLevel:
        """Cycle to the next thinking level."""
        levels: list[ThinkingLevel] = ["off", "minimal", "low", "medium", "high"]
        current = self._agent.state.thinking_level
        current_idx = levels.index(current) if current in levels else 0
        next_level = levels[(current_idx + 1) % len(levels)]
        self.set_thinking_level(next_level)
        return next_level

    # -------------------------------------------------------------------------
    # Session management
    # -------------------------------------------------------------------------

    async def new_session(self) -> None:
        """Start a new session."""
        await self.abort_and_wait()
        self._agent.reset()
        self._session_manager.new_session()
        self._agent.session_id = self._session_manager.get_session_id()
        self._steering_messages.clear()
        self._follow_up_messages.clear()

    def load_session_from_file(self, file_path: str) -> bool:
        """
        Load a session from a JSONL file.

        Returns True if successful.
        """
        if not self._session_manager.load_session(file_path):
            return False

        # Restore agent state from session
        context = self._session_manager.build_session_context()
        self._agent.replace_messages(context.messages)
        self._agent.session_id = self._session_manager.get_session_id()

        return True

    # -------------------------------------------------------------------------
    # Session statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get session statistics."""
        messages = self._agent.state.messages
        user_msgs = sum(1 for m in messages if getattr(m, "role", None) == "user")
        assistant_msgs = sum(1 for m in messages if getattr(m, "role", None) == "assistant")

        from pi_agent_core.types import AssistantMessage as AM
        tool_results = sum(1 for m in messages if getattr(m, "role", None) == "toolResult")

        total_input = 0
        total_output = 0
        total_cost = 0.0
        tool_calls = 0

        for m in messages:
            if isinstance(m, AM) and m.usage:
                total_input += m.usage.input
                total_output += m.usage.output
                total_cost += m.usage.cost.get("total", 0.0)
            if isinstance(m, AM):
                from pi_agent_core.types import ToolCall
                tool_calls += sum(
                    1 for b in m.content if isinstance(b, ToolCall)
                )

        return {
            "session_id": self.session_id,
            "session_file": self.session_file,
            "user_messages": user_msgs,
            "assistant_messages": assistant_msgs,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "total_messages": len(messages),
            "tokens": {
                "input": total_input,
                "output": total_output,
            },
            "cost": total_cost,
        }
