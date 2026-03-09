"""Lightweight in-process event bus for the live runtime."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

EventHandler = Callable[["RuntimeEvent"], Optional[Awaitable[None]]]


@dataclass
class RuntimeEvent:
    event_type: str
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class InMemoryEventBus:
    """Bounded, best-effort dispatcher for single-node runtime events."""

    def __init__(self, max_queue_size: int = 10000) -> None:
        self._queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue(maxsize=max_queue_size)
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._running = False
        self._dropped = 0

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        self._running = False
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
            self._dispatcher_task = None

    def publish_nowait(self, event_type: str, payload: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(RuntimeEvent(event_type=event_type, payload=payload))
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning("Event bus queue full, dropping event: %s", event_type)

    async def _dispatch_loop(self) -> None:
        while self._running:
            event = await self._queue.get()
            try:
                handlers = list(self._handlers.get(event.event_type, []))
                handlers += list(self._handlers.get("*", []))
                for handler in handlers:
                    try:
                        result = handler(event)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:
                        logger.debug("Event handler error for %s: %s", event.event_type, exc)
            finally:
                self._queue.task_done()

    def summary(self) -> Dict[str, Any]:
        return {
            "queue_depth": self._queue.qsize(),
            "dropped_events": self._dropped,
            "subscriptions": {k: len(v) for k, v in self._handlers.items()},
        }
