"""Interactive Brokers broker (ib_insync)."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any, Dict, Optional, Union

from ..core.asset_types import AssetClass
from ..core.market_hours import MarketCalendar
from ..core.pdt_tracker import PDTTracker
from .base import Broker

logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder
except ImportError:
    IB = Stock = MarketOrder = LimitOrder = None


class IBKRBroker(Broker):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ) -> None:
        if IB is None:
            raise ImportError("ib_insync required: pip install ib_insync")
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib: Optional[IB] = None
        self._positions: Dict[str, float] = {}
        self._cash: float = 0.0
        self._pdt_tracker = PDTTracker()
        self._calendar = MarketCalendar()

    async def connect(self) -> None:
        self._ib = IB()
        await self._ib.connectAsync(self._host, self._port, self._client_id)
        await self.sync_positions()
        await self.sync_balance()

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    async def submit_order_async(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        if self._ib is None:
            raise RuntimeError("IBKR broker not connected. Call connect() first.")
        symbol = str(signal.get("symbol", ""))
        action = str(signal.get("action", "")).lower()
        shares = int(float(signal.get("shares", 0)))
        if not symbol or shares <= 0:
            return {"status": "rejected", "message": "invalid symbol/shares"}
        equity = self._cash
        for item in (self._ib.accountSummary() if self._ib else []):
            if item.tag == "NetLiquidation":
                equity = float(item.value)
                break
        if action == "sell" and self._pdt_tracker.would_violate(equity, date.today()):
            return {"status": "rejected", "message": "PDT rule violation"}
        if not self._calendar.is_tradable(AssetClass.US_EQUITY, datetime.now()):
            return {"status": "rejected", "message": "market closed"}
        contract = Stock(symbol, "SMART", "USD")
        order = MarketOrder("BUY" if action == "buy" else "SELL", shares)
        trade = self._ib.placeOrder(contract, order)
        fill_event = asyncio.Event()

        def on_done(t):
            if t.isDone():
                fill_event.set()

        trade.statusEvent += on_done
        try:
            await asyncio.wait_for(fill_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            trade.statusEvent -= on_done
            self._ib.cancelOrder(trade.order)
            return {"status": "timeout", "message": "order timeout 30s"}
        trade.statusEvent -= on_done
        status = str(trade.orderStatus.status).lower()
        if status == "filled":
            commission = sum(f.commission for f in trade.fills) if trade.fills else 0.0
            return {
                "status": "filled",
                "order_id": str(trade.order.orderId),
                "fill_price": float(trade.orderStatus.avgFillPrice or 0),
                "filled_shares": float(trade.orderStatus.filled or 0),
                "commission": commission,
            }
        return {"status": status, "message": trade.orderStatus.whyHeld or ""}

    def submit_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.submit_order_async(signal))

    async def cancel_order_async(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        if self._ib is None:
            raise RuntimeError("IBKR broker not connected. Call connect() first.")
        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == order_id:
                self._ib.cancelOrder(trade.order)
                return {"status": "cancelled", "order_id": order_id}
        return {"status": "error", "message": "order not found"}

    async def sync_positions(self) -> Dict[str, Union[int, float]]:
        if not self._ib:
            return dict(self._positions)
        positions = self._ib.positions()
        self._positions = {}
        for p in positions:
            sym = p.contract.symbol
            self._positions[sym] = float(p.position)
        return dict(self._positions)

    async def sync_balance(self) -> Dict[str, Any]:
        if not self._ib:
            return {"cash": self._cash}
        account = self._ib.accountSummary()
        for item in account:
            if item.tag == "TotalCashValue":
                self._cash = float(item.value)
                break
        return {"cash": self._cash}

    def get_positions(self) -> Dict[str, Union[int, float]]:
        return dict(self._positions)

    def get_cash(self) -> float:
        return self._cash
