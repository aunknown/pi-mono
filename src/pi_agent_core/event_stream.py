"""
EventStream - async iterable stream with a final result.

Mirrors the TypeScript EventStream<TEvent, TResult> from @mariozechner/pi-ai.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable, Generic, TypeVar

TEvent = TypeVar("TEvent")
TResult = TypeVar("TResult")


class EventStream(Generic[TEvent, TResult]):
    """
    Async iterable stream of events with a final result value.

    Usage::

        stream = EventStream(
            is_done=lambda e: e.type == "agent_end",
            get_result=lambda e: e.messages if e.type == "agent_end" else None,
        )

        # Producer side:
        stream.push(some_event)
        stream.end(final_result)

        # Consumer side:
        async for event in stream:
            ...

        result = await stream.result()
    """

    def __init__(
        self,
        is_done: Callable[[TEvent], bool],
        get_result: Callable[[TEvent], TResult],
    ) -> None:
        self._is_done = is_done
        self._get_result = get_result
        self._queue: asyncio.Queue[TEvent | _Sentinel] = asyncio.Queue()
        self._result_future: asyncio.Future[TResult] = asyncio.get_event_loop().create_future()
        self._ended = False

    def push(self, event: TEvent) -> None:
        """Push an event onto the stream."""
        if self._ended:
            return
        self._queue.put_nowait(event)
        if self._is_done(event):
            result = self._get_result(event)
            if not self._result_future.done():
                self._result_future.set_result(result)

    def end(self, result: TResult | None = None) -> None:
        """Signal end of stream."""
        if self._ended:
            return
        self._ended = True
        self._queue.put_nowait(_SENTINEL)
        if not self._result_future.done():
            if result is not None:
                self._result_future.set_result(result)
            else:
                self._result_future.set_exception(
                    RuntimeError("Stream ended without a result")
                )

    def end_with_error(self, error: Exception) -> None:
        """Signal end of stream with an error."""
        if self._ended:
            return
        self._ended = True
        self._queue.put_nowait(_SENTINEL)
        if not self._result_future.done():
            self._result_future.set_exception(error)

    async def result(self) -> TResult:
        """Await the final result of the stream."""
        return await self._result_future

    def __aiter__(self) -> AsyncIterator[TEvent]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[TEvent]:
        while True:
            item = await self._queue.get()
            if isinstance(item, _Sentinel):
                break
            yield item


class _Sentinel:
    """Sentinel value to signal end of stream."""
    pass


_SENTINEL = _Sentinel()
