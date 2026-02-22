"""
Agent class - high-level API for the agent loop.

Mirrors agent.ts from @mariozechner/pi-agent-core.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .agent_loop import (
    AgentLoopOptions,
    agent_loop,
    agent_loop_continue,
)
from .llm_client import StreamFn, StreamOptions, stream_anthropic
from .types import (
    AgentContext,
    AgentEvent,
    AgentMessage,
    AgentState,
    AgentTool,
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingLevel,
    UserMessage,
    AgentEndEvent,
)


@dataclass
class AgentOptions:
    """Options for creating an Agent."""

    initial_state: Optional[dict] = None

    # Converts AgentMessage[] to LLM-compatible Message[] before each LLM call.
    convert_to_llm: Optional[Callable[[list[AgentMessage]], list[AgentMessage]]] = None

    # Optional transform applied to context before convert_to_llm.
    transform_context: Optional[Callable] = None

    # Steering mode: "all" or "one-at-a-time"
    steering_mode: str = "one-at-a-time"

    # Follow-up mode: "all" or "one-at-a-time"
    follow_up_mode: str = "one-at-a-time"

    # Custom stream function (for proxy backends, etc.)
    stream_fn: Optional[StreamFn] = None

    # Optional session identifier forwarded to LLM providers
    session_id: Optional[str] = None

    # Resolves an API key dynamically for each LLM call
    get_api_key: Optional[Callable[[str], Any]] = None

    # Stream options (model, api_key, base_url, etc.)
    stream_options: Optional[StreamOptions] = None


class Agent:
    """
    High-level agent that manages state and the agent loop.

    Usage::

        agent = Agent()
        agent.set_system_prompt("You are a helpful assistant.")
        agent.set_stream_options(StreamOptions(
            api_key="sk-ant-...",
            model_id="claude-3-5-sonnet-20241022",
        ))

        unsubscribe = agent.subscribe(lambda event: print(event.type))
        await agent.prompt("Hello, world!")
        unsubscribe()
    """

    def __init__(self, opts: Optional[AgentOptions] = None) -> None:
        opts = opts or AgentOptions()

        self._state = AgentState()
        if opts.initial_state:
            for k, v in opts.initial_state.items():
                if hasattr(self._state, k):
                    setattr(self._state, k, v)

        self._listeners: list[Callable[[AgentEvent], None]] = []
        self._abort_event: Optional[asyncio.Event] = None
        self._convert_to_llm = opts.convert_to_llm or _default_convert_to_llm
        self._transform_context = opts.transform_context
        self._steering_mode = opts.steering_mode
        self._follow_up_mode = opts.follow_up_mode
        self._stream_fn: StreamFn = opts.stream_fn or stream_anthropic
        self._session_id = opts.session_id
        self._get_api_key = opts.get_api_key
        self._stream_options: StreamOptions = opts.stream_options or StreamOptions()

        # Message queues for steering and follow-up
        self._steering_queue: list[AgentMessage] = []
        self._follow_up_queue: list[AgentMessage] = []

        # Running state
        self._running_task: Optional[asyncio.Task] = None
        self._idle_event: Optional[asyncio.Event] = None

    # -------------------------------------------------------------------------
    # Public properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        self._session_id = value

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self._state.system_prompt = prompt

    def set_stream_options(self, opts: StreamOptions) -> None:
        """Set LLM stream options (model, api_key, etc.)."""
        self._stream_options = opts

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        """Set the thinking/reasoning level."""
        self._state.thinking_level = level

    def set_steering_mode(self, mode: str) -> None:
        self._steering_mode = mode

    def get_steering_mode(self) -> str:
        return self._steering_mode

    def set_follow_up_mode(self, mode: str) -> None:
        self._follow_up_mode = mode

    def get_follow_up_mode(self) -> str:
        return self._follow_up_mode

    def set_tools(self, tools: list[AgentTool]) -> None:
        """Set the available tools."""
        self._state.tools = tools

    def replace_messages(self, messages: list[AgentMessage]) -> None:
        """Replace all messages in the conversation."""
        self._state.messages = list(messages)

    def append_message(self, message: AgentMessage) -> None:
        """Append a single message to the conversation."""
        self._state.messages = self._state.messages + [message]

    def clear_messages(self) -> None:
        """Clear all messages."""
        self._state.messages = []

    # -------------------------------------------------------------------------
    # Event subscription
    # -------------------------------------------------------------------------

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Subscribe to agent events. Returns an unsubscribe function."""
        self._listeners.append(fn)
        def _unsubscribe():
            if fn in self._listeners:
                self._listeners.remove(fn)
        return _unsubscribe

    def _emit(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            try:
                result = listener(event)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Message queuing (steering and follow-up)
    # -------------------------------------------------------------------------

    def steer(self, message: AgentMessage) -> None:
        """Queue a steering message to interrupt the agent mid-run."""
        self._steering_queue.append(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a follow-up message for after the agent finishes."""
        self._follow_up_queue.append(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue = []

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue = []

    def clear_all_queues(self) -> None:
        self._steering_queue = []
        self._follow_up_queue = []

    def has_queued_messages(self) -> bool:
        return bool(self._steering_queue or self._follow_up_queue)

    def _dequeue_steering(self) -> list[AgentMessage]:
        if self._steering_mode == "one-at-a-time":
            if self._steering_queue:
                msg = self._steering_queue.pop(0)
                return [msg]
            return []
        msgs = list(self._steering_queue)
        self._steering_queue = []
        return msgs

    def _dequeue_follow_up(self) -> list[AgentMessage]:
        if self._follow_up_mode == "one-at-a-time":
            if self._follow_up_queue:
                msg = self._follow_up_queue.pop(0)
                return [msg]
            return []
        msgs = list(self._follow_up_queue)
        self._follow_up_queue = []
        return msgs

    # -------------------------------------------------------------------------
    # Prompt / continue
    # -------------------------------------------------------------------------

    async def prompt(
        self,
        input: "str | AgentMessage | list[AgentMessage]",
        images: Optional[list[ImageContent]] = None,
    ) -> None:
        """Send a prompt to the agent and wait for it to complete."""
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing a prompt. "
                "Use steer() or follow_up() to queue messages, or wait for completion."
            )

        msgs: list[AgentMessage]
        if isinstance(input, list):
            msgs = input
        elif isinstance(input, str):
            content: list = [TextContent(text=input)]
            if images:
                content.extend(images)
            msgs = [UserMessage(content=content)]
        else:
            msgs = [input]

        await self._run_loop(msgs)

    async def continue_from_context(self) -> None:
        """Continue from the current context (used for retries)."""
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing.")

        messages = self._state.messages
        if not messages:
            raise ValueError("No messages to continue from")

        last_role = getattr(messages[-1], "role", None)
        if last_role == "assistant":
            queued_steering = self._dequeue_steering()
            if queued_steering:
                await self._run_loop(queued_steering, skip_initial_steering_poll=True)
                return
            queued_follow_up = self._dequeue_follow_up()
            if queued_follow_up:
                await self._run_loop(queued_follow_up)
                return
            raise ValueError("Cannot continue from message role: assistant")

        await self._run_loop(None)

    def abort(self) -> None:
        """Abort the current operation."""
        if self._abort_event:
            self._abort_event.set()

    def _get_idle_event(self) -> asyncio.Event:
        """Lazily create the idle event on the running event loop."""
        if self._idle_event is None:
            self._idle_event = asyncio.Event()
            self._idle_event.set()
        return self._idle_event

    async def wait_for_idle(self) -> None:
        """Wait until the agent is no longer streaming."""
        await self._get_idle_event().wait()

    def reset(self) -> None:
        """Reset agent to initial state."""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._state.error = None
        self._steering_queue = []
        self._follow_up_queue = []

    # -------------------------------------------------------------------------
    # Internal run loop
    # -------------------------------------------------------------------------

    async def _run_loop(
        self,
        messages: Optional[list[AgentMessage]],
        skip_initial_steering_poll: bool = False,
    ) -> None:
        """Execute the agent loop."""
        self._abort_event = asyncio.Event()
        self._get_idle_event().clear()
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None

        reasoning = None if self._state.thinking_level == "off" else self._state.thinking_level

        # Build stream options with reasoning
        stream_opts = StreamOptions(
            api_key=self._stream_options.api_key,
            base_url=self._stream_options.base_url,
            model_id=self._stream_options.model_id,
            provider=self._stream_options.provider,
            reasoning=reasoning,
            temperature=self._stream_options.temperature,
            max_tokens=self._stream_options.max_tokens,
            session_id=self._session_id,
        )

        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

        _skip_poll = skip_initial_steering_poll

        async def get_steering() -> list[AgentMessage]:
            nonlocal _skip_poll
            if _skip_poll:
                _skip_poll = False
                return []
            return self._dequeue_steering()

        async def get_follow_up() -> list[AgentMessage]:
            return self._dequeue_follow_up()

        loop_options = AgentLoopOptions(
            stream_options=stream_opts,
            convert_to_llm=self._convert_to_llm,
            transform_context=self._transform_context,
            get_api_key=self._get_api_key,
            get_steering_messages=get_steering,
            get_follow_up_messages=get_follow_up,
        )

        partial: Optional[AgentMessage] = None

        try:
            if messages is not None:
                event_stream = agent_loop(
                    messages, context, loop_options,
                    self._abort_event, self._stream_fn
                )
            else:
                event_stream = agent_loop_continue(
                    context, loop_options,
                    self._abort_event, self._stream_fn
                )

            async for event in event_stream:
                # Update internal state
                etype = event.type

                if etype == "message_start":
                    partial = event.message
                    self._state.stream_message = event.message
                elif etype == "message_update":
                    partial = event.message
                    self._state.stream_message = event.message
                elif etype == "message_end":
                    partial = None
                    self._state.stream_message = None
                    self.append_message(event.message)
                elif etype == "tool_execution_start":
                    s = set(self._state.pending_tool_calls)
                    s.add(event.tool_call_id)
                    self._state.pending_tool_calls = s
                elif etype == "tool_execution_end":
                    s = set(self._state.pending_tool_calls)
                    s.discard(event.tool_call_id)
                    self._state.pending_tool_calls = s
                elif etype == "turn_end":
                    msg = event.message
                    if getattr(msg, "role", None) == "assistant":
                        err = getattr(msg, "error_message", None)
                        if err:
                            self._state.error = err
                elif etype == "agent_end":
                    self._state.is_streaming = False
                    self._state.stream_message = None

                # Forward to listeners
                self._emit(event)

            # Handle any remaining partial message
            if partial and getattr(partial, "role", None) == "assistant":
                if isinstance(partial, AssistantMessage):
                    non_empty = any(
                        (isinstance(b, TextContent) and b.text.strip()) or
                        (hasattr(b, "thinking") and b.thinking.strip()) or
                        (hasattr(b, "name") and b.name.strip())
                        for b in partial.content
                    )
                    if non_empty:
                        self.append_message(partial)

        except Exception as err:
            error_msg = AssistantMessage(
                content=[TextContent(text="")],
                api=self._stream_options.provider,
                provider=self._stream_options.provider,
                model=self._stream_options.model_id,
                stop_reason="aborted" if (self._abort_event and self._abort_event.is_set()) else "error",
                error_message=str(err),
            )
            self.append_message(error_msg)
            self._state.error = str(err)
            self._emit(AgentEndEvent(messages=[error_msg]))

        finally:
            self._state.is_streaming = False
            self._state.stream_message = None
            self._state.pending_tool_calls = set()
            self._abort_event = None
            self._get_idle_event().set()


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[AgentMessage]:
    """Default: keep only LLM-compatible messages.

    Note: This duplicates the same function in agent_loop.py for backward
    compatibility; both implementations must stay in sync.
    """
    return [
        m for m in messages
        if getattr(m, "role", None) in ("user", "assistant", "toolResult")
    ]
