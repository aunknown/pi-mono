"""
Event bus for decoupled communication within the coding agent.

Mirrors event-bus.ts from pi-coding-agent.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable


class EventBus:
    """
    Simple pub/sub event bus.

    Usage::

        bus = EventBus()

        unsubscribe = bus.on("my_event", lambda data: print(data))
        bus.emit("my_event", {"foo": "bar"})
        unsubscribe()
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = {}

    def emit(self, channel: str, data: Any = None) -> None:
        """Emit an event to all subscribers of a channel."""
        for handler in list(self._handlers.get(channel, [])):
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception as e:
                print(f"Event handler error ({channel}): {e}")

    def on(self, channel: str, handler: Callable) -> Callable[[], None]:
        """
        Subscribe to a channel.

        Returns a callable that unsubscribes when called.
        """
        if channel not in self._handlers:
            self._handlers[channel] = []
        self._handlers[channel].append(handler)

        def unsubscribe():
            handlers = self._handlers.get(channel, [])
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    def clear(self) -> None:
        """Remove all event handlers."""
        self._handlers.clear()


def create_event_bus() -> EventBus:
    """Create a new EventBus instance."""
    return EventBus()
