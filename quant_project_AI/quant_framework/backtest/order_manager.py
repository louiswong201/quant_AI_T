"""
Order lifecycle manager: submit, cancel, expire, track.

Extracted from BacktestEngine to isolate order state management.
This module owns the pending queue and the full order audit trail.

Design decisions:
  - Orders are mutable dataclass instances (status transitions in-place)
    rather than immutable copies, because the pending queue is small (<100)
    and copy overhead would dominate for no safety benefit in a single-threaded
    backtest loop.
  - The order_id -> index mapping uses a dict for O(1) lookup.
    For a live system this would be replaced by an order book data structure.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .protocols import Order, OrderSide, OrderStatus, OrderType


class DefaultOrderManager:
    """Default order manager for backtesting.

    Thread-safety: NOT thread-safe. Backtest runs in a single thread.
    For live trading, wrap with a lock or use a concurrent order manager.
    """

    def __init__(self) -> None:
        self._pending: List[Order] = []
        self._all_orders: List[Order] = []
        self._index: Dict[str, int] = {}
        self._seq: int = 0

    def next_order_id(self) -> str:
        """Generate a unique order ID."""
        oid = f"ord_{self._seq}"
        self._seq += 1
        return oid

    def submit(self, order: Order) -> str:
        """Accept a new order into the system."""
        if not order.order_id:
            order.order_id = self.next_order_id()
        order.status = OrderStatus.SUBMITTED
        self._index[order.order_id] = len(self._all_orders)
        self._all_orders.append(order)
        self._pending.append(order)
        return order.order_id

    def cancel(self, order_id: str) -> bool:
        """Cancel a pending order by ID. Returns True if found and cancelled."""
        new_pending: List[Order] = []
        found = False
        for o in self._pending:
            if o.order_id == order_id:
                o.status = OrderStatus.CANCELLED
                found = True
            else:
                new_pending.append(o)
        self._pending = new_pending
        return found

    def expire_stale(self, current_bar: int, current_date: pd.Timestamp) -> List[Order]:
        """Expire orders that exceeded their time-in-force."""
        expired: List[Order] = []
        remaining: List[Order] = []
        for o in self._pending:
            if o.tif_bars is not None and (current_bar - o.submitted_bar) >= o.tif_bars:
                o.status = OrderStatus.EXPIRED
                expired.append(o)
            else:
                remaining.append(o)
        self._pending = remaining
        return expired

    def remove_filled(self, filled_ids: List[str]) -> None:
        """Remove filled orders from pending queue."""
        id_set = set(filled_ids)
        self._pending = [o for o in self._pending if o.order_id not in id_set]

    def create_order(
        self,
        side: OrderSide,
        symbol: str,
        shares: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        bar_index: int = 0,
        date: Optional[pd.Timestamp] = None,
        tif_bars: Optional[int] = None,
        order_id: Optional[str] = None,
    ) -> Order:
        """Create and submit an order in one step."""
        order = Order(
            order_id=order_id or self.next_order_id(),
            side=side,
            symbol=symbol,
            shares=shares,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            submitted_bar=bar_index,
            submitted_date=date,
            tif_bars=tif_bars,
        )
        self.submit(order)
        return order

    @property
    def pending(self) -> List[Order]:
        return self._pending

    @property
    def all_orders(self) -> List[Order]:
        return self._all_orders

    def get_order(self, order_id: str) -> Optional[Order]:
        """O(1) order lookup by ID."""
        idx = self._index.get(order_id)
        if idx is not None and idx < len(self._all_orders):
            return self._all_orders[idx]
        return None
