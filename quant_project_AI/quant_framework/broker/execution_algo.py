"""Execution algorithms: TWAP, LimitChase, Iceberg."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ExecutionAlgo(ABC):
    @abstractmethod
    async def execute(
        self,
        submit_fn: Callable[[Dict[str, Any]], Any],
        symbol: str,
        side: str,
        total_qty: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        pass


class TWAP(ExecutionAlgo):
    """Split order into K slices, execute every T seconds."""

    def __init__(self, slices: int = 5, interval_seconds: float = 60.0) -> None:
        self._slices = max(1, slices)
        self._interval = max(1.0, interval_seconds)

    async def execute(
        self,
        submit_fn: Callable[[Dict[str, Any]], Any],
        symbol: str,
        side: str,
        total_qty: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        qty_per_slice = total_qty / self._slices
        filled = 0.0
        for i in range(self._slices):
            signal = {
                "symbol": symbol,
                "action": side.lower(),
                "shares": qty_per_slice,
                **kwargs,
            }
            result = await submit_fn(signal)
            if result.get("status") == "filled":
                filled += float(result.get("filled_shares", 0))
            elif result.get("status") in ("error", "rejected"):
                return result
            if i < self._slices - 1:
                await asyncio.sleep(self._interval)
        return {"status": "filled", "filled_shares": filled}


class LimitChase(ExecutionAlgo):
    """Start with limit, chase price if unfilled."""

    def __init__(
        self,
        chase_ticks: int = 1,
        retry_interval: float = 5.0,
        max_retries: int = 10,
    ) -> None:
        self._chase_ticks = chase_ticks
        self._retry_interval = retry_interval
        self._max_retries = max_retries

    async def execute(
        self,
        submit_fn: Callable[[Dict[str, Any]], Any],
        symbol: str,
        side: str,
        total_qty: float,
        price: float,
        tick_size: float = 0.01,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        max_iterations = min(self._max_retries, 1000)
        current_price = price
        for _ in range(max_iterations):
            signal = {
                "symbol": symbol,
                "action": side.lower(),
                "shares": total_qty,
                "price": current_price,
                "order_type": "limit",
                **kwargs,
            }
            result = await submit_fn(signal)
            if result.get("status") == "filled":
                return result
            if result.get("status") in ("error", "rejected"):
                return result
            delta = self._chase_ticks * tick_size
            if side.upper() == "BUY":
                current_price += delta
            else:
                current_price -= delta
            await asyncio.sleep(self._retry_interval)
        return {"status": "error", "message": "LimitChase max retries exceeded"}


class Iceberg(ExecutionAlgo):
    """Show 10% of real quantity, auto-refill on fill."""

    def __init__(
        self,
        display_pct: float = 0.1,
        refill_delay: float = 1.0,
    ) -> None:
        self._display_pct = max(0.01, min(1.0, display_pct))
        self._refill_delay = refill_delay

    async def execute(
        self,
        submit_fn: Callable[[Dict[str, Any]], Any],
        symbol: str,
        side: str,
        total_qty: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        display_qty = max(
            round(total_qty * self._display_pct, 8),
            total_qty / 100,
        )
        remaining = total_qty
        filled_total = 0.0
        max_iterations = 1000
        iterations = 0
        while remaining > 1e-8:
            iterations += 1
            if iterations > max_iterations:
                logger.warning("Iceberg: max iterations (%d) exceeded, breaking", max_iterations)
                break
            qty = min(display_qty, remaining)
            signal = {
                "symbol": symbol,
                "action": side.lower(),
                "shares": qty,
                **kwargs,
            }
            result = await submit_fn(signal)
            if result.get("status") == "filled":
                filled = float(result.get("filled_shares", 0))
                filled_total += filled
                remaining -= filled
            elif result.get("status") in ("error", "rejected"):
                return {**result, "filled_shares": filled_total}
            await asyncio.sleep(self._refill_delay)
        return {"status": "filled", "filled_shares": filled_total}
