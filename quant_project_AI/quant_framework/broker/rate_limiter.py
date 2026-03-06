"""O(1) rate limiter for exchange API."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque, Tuple


class RateLimiter:
    def __init__(self, max_weight: int = 1200, window_seconds: float = 60.0) -> None:
        self._max_weight = max_weight
        self._window = window_seconds
        self._used: Deque[Tuple[float, int]] = deque()
        self._running_total = 0

    async def acquire(self, weight: int = 1) -> None:
        while self._current_usage() + weight > self._max_weight:
            await asyncio.sleep(0.05)
        self._used.append((time.time(), weight))
        self._running_total += weight

    def _current_usage(self) -> int:
        now = time.time()
        while self._used and now - self._used[0][0] > self._window:
            _, w = self._used.popleft()
            self._running_total -= w
        return self._running_total
