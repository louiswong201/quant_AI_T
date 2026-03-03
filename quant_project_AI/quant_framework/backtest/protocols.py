"""
Backtest engine component protocols (interfaces).

Defines the contracts for pluggable components: fill simulation, order management,
and portfolio tracking. Using Protocol (structural subtyping) allows swapping
implementations between backtest and live trading without inheritance coupling.

Why Protocol over ABC:
  - Structural subtyping: any class matching the method signatures is valid,
    no forced inheritance hierarchy.
  - Enables duck-typing with static type checker support (mypy).
  - Backtest FillSimulator and live Exchange adapter can share the same interface
    without a common base class, keeping the dependency graph clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

class OrderSide(Enum):
    BUY = auto()
    SELL = auto()
    CANCEL = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()


class OrderStatus(Enum):
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order ticket — supports both integer and fractional share quantities."""

    order_id: str
    side: OrderSide
    symbol: str
    shares: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    submitted_bar: int = 0
    submitted_date: Optional[pd.Timestamp] = None
    tif_bars: Optional[int] = None
    status: OrderStatus = OrderStatus.SUBMITTED
    filled_shares: float = 0.0
    fill_price: float = 0.0
    filled_bar: Optional[int] = None
    filled_date: Optional[pd.Timestamp] = None


@dataclass
class Fill:
    """Result of executing an order (or partial fill)."""

    order: Order
    exec_price: float
    exec_shares: float
    commission: float
    bar_index: int
    date: pd.Timestamp


@dataclass
class BarData:
    """Pre-extracted numpy arrays for a single symbol's OHLCV.

    Avoids repeated DataFrame column lookups inside the hot loop.
    All arrays are contiguous float64 for Numba/SIMD compatibility.
    """

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    rolling_vol: np.ndarray


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class IFillSimulator(Protocol):
    """Simulates order execution: slippage, market impact, partial fills.

    Backtest uses CostModelFillSimulator; live trading can implement
    ExchangeFillAdapter with the same interface.
    """

    def try_fill_pending(
        self,
        pending: List[Order],
        symbol: str,
        bar: BarData,
        bar_index: int,
        date: pd.Timestamp,
    ) -> Tuple[List[Fill], List[Order]]:
        """Attempt to fill pending orders against current bar OHLCV.

        Returns:
            fills: successfully executed fills this bar
            remaining: orders that could not be filled (stay pending)
        """
        ...

    def execute_market(
        self,
        order: Order,
        ref_price: float,
        bar: BarData,
        bar_index: int,
        date: pd.Timestamp,
    ) -> Optional[Fill]:
        """Execute a market order immediately at ref_price (with slippage/impact)."""
        ...


@runtime_checkable
class IOrderManager(Protocol):
    """Manages the order lifecycle: submit, cancel, expire, track status."""

    def submit(self, order: Order) -> str:
        """Accept a new order, return its order_id."""
        ...

    def cancel(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if found and cancelled."""
        ...

    def expire_stale(self, current_bar: int, current_date: Any = None) -> List[Order]:
        """Expire orders past their time-in-force. Returns expired orders."""
        ...

    @property
    def pending(self) -> List[Order]:
        """Currently active (unfilled) orders."""
        ...

    @property
    def all_orders(self) -> List[Order]:
        """Full order history."""
        ...


@runtime_checkable
class IPortfolioTracker(Protocol):
    """Tracks positions (long+short, int or float), cash, equity curve, and trade history."""

    def apply_fill(self, fill: Fill) -> None:
        """Update positions and cash after a fill."""
        ...

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        """Revalue portfolio at current prices. Returns portfolio value."""
        ...

    @property
    def cash(self) -> float:
        ...

    @property
    def positions(self) -> Dict[str, float]:
        ...

    @property
    def trades(self) -> List[Dict[str, Any]]:
        ...

    @property
    def portfolio_value(self) -> float:
        ...

    @property
    def is_liquidated(self) -> bool:
        ...
