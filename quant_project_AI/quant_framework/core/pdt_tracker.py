from __future__ import annotations

from collections import deque
from datetime import date, datetime, timedelta


class PDTTracker:
    """Pattern Day Trader: equity < 25k and 3+ day trades in 5 trading days -> violation."""

    def __init__(self, equity_threshold: float = 25_000) -> None:
        self._threshold = equity_threshold
        self._day_trades: deque[tuple[str, date]] = deque()

    def record_round_trip(self, symbol: str, dt: datetime) -> None:
        self._day_trades.append((symbol, dt.date()))
        self._cleanup_old(dt.date())

    def day_trade_count(self, current_date: date) -> int:
        self._cleanup_old(current_date)
        return len(self._day_trades)

    def would_violate(self, equity: float, current_date: date) -> bool:
        if equity >= self._threshold:
            return False
        return self.day_trade_count(current_date) >= 3

    def _cleanup_old(self, current_date: date) -> None:
        cutoff = current_date - timedelta(days=7)
        while self._day_trades and self._day_trades[0][1] < cutoff:
            self._day_trades.popleft()
