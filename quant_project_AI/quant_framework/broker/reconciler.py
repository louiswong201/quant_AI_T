"""Periodic position reconciliation: exchange vs internal state."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, TYPE_CHECKING, Union

from .base import Broker

if TYPE_CHECKING:
    from ..live.trade_journal import TradeJournal

logger = logging.getLogger(__name__)


class PositionReconciler:
    def __init__(
        self,
        broker: Broker,
        journal: Optional["TradeJournal"] = None,
        alert_manager: Optional[Callable[..., Union[Any, Awaitable[Any]]]] = None,
        interval_seconds: float = 60.0,
    ) -> None:
        self._broker = broker
        self._journal = journal
        self._alert = alert_manager
        self._interval = interval_seconds

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            await self.reconcile()

    async def reconcile(self) -> None:
        exchange_positions = await self._broker.sync_positions()
        internal_positions = self._broker.get_positions()
        await self._broker.sync_balance()
        discrepancies = []
        all_symbols = set(exchange_positions) | set(internal_positions)
        for sym in all_symbols:
            ex_qty = exchange_positions.get(sym, 0.0)
            in_qty = internal_positions.get(sym, 0.0)
            if abs(ex_qty - in_qty) > 1e-8:
                discrepancies.append({
                    "symbol": sym,
                    "exchange": ex_qty,
                    "internal": in_qty,
                    "diff": ex_qty - in_qty,
                })
        if discrepancies:
            logger.warning("Position discrepancies: %s", discrepancies)
            if self._alert:
                if hasattr(self._alert, "send"):
                    result = self._alert.send("WARNING", "Position mismatch", str(discrepancies))
                    if asyncio.iscoroutine(result):
                        await result
                else:
                    result = self._alert("WARNING", "Position mismatch", str(discrepancies))
                    if asyncio.iscoroutine(result):
                        await result
            if hasattr(self._broker, "_positions"):
                self._broker._positions = dict(exchange_positions)
