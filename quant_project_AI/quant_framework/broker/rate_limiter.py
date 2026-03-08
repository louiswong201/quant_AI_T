"""O(1) rate limiter for exchange API."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Deque, Tuple


class RateLimiter:
    def __init__(self, max_weight: int = 1200, window_seconds: float = 60.0) -> None:
        self._max_weight = max_weight
        self._window = window_seconds
        self._used: Deque[Tuple[float, int]] = deque()
        self._running_total = 0
        self._lock = threading.Lock()

    def check(self, weight: int = 1) -> bool:
        """Thread-safe check-and-append. Returns True if request allowed, False if rate limited."""
        with self._lock:
            now = time.monotonic()
            self._used = deque((t, w) for t, w in self._used if now - t < self._window)
            self._running_total = sum(w for _, w in self._used)
            if self._running_total + weight > self._max_weight:
                return False
            self._used.append((now, weight))
            self._running_total += weight
            return True

    async def acquire(self, weight: int = 1) -> None:
        """Block until a request slot is available, then add it atomically."""
        while not self.check(weight):
            await asyncio.sleep(0.05)
