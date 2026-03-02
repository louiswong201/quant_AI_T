"""
Fill simulation engine: slippage, market impact, and liquidity constraints.

Extracted from the monolithic BacktestEngine to achieve single-responsibility.
This module is the ONLY place where execution prices are computed.

Performance justification:
  - _apply_market_impact uses the Almgren-Chriss square-root impact model:
    impact_bps = coeff * (participation_rate ^ exponent).
    This is O(1) per fill — no allocations, pure arithmetic.
  - Pending order matching is O(k log k) where k = pending orders for one symbol,
    dominated by the priority sort. For typical retail strategies k < 10.

Why not Numba here:
  The fill logic involves Python object manipulation (Order dataclass, list ops).
  Numba's nopython mode cannot handle these. The hot-path numerical work
  (impact calculation) is already branch-free floating-point arithmetic —
  further JIT would yield <1% gain at significant complexity cost.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .config import BacktestConfig
from .protocols import BarData, Fill, Order, OrderSide, OrderStatus, OrderType


def _pending_priority_key(order: Order) -> Tuple[float, int]:
    """Price-time priority for pending order matching.

    - Limit buy: higher price first (negated for ascending sort)
    - Limit sell: lower price first
    - All others: FIFO by submission bar
    """
    if order.order_type == OrderType.LIMIT:
        if order.side == OrderSide.BUY:
            return (-(order.limit_price or 0.0), order.submitted_bar)
        if order.side == OrderSide.SELL:
            return (order.limit_price or 0.0, order.submitted_bar)
    return (0.0, order.submitted_bar)


def _compute_market_impact(
    base_price: float,
    side: OrderSide,
    shares: int,
    bar_volume: float,
    rolling_vol: float,
    config: BacktestConfig,
) -> float:
    """Almgren-Chriss style non-linear market impact.

    impact_bps = coeff * (participation_rate ^ exponent)

    The adaptive mode scales the coefficient by realised volatility relative
    to a reference level, capturing the empirical observation that impact
    increases in high-vol regimes.
    """
    if base_price <= 0 or shares <= 0:
        return max(0.0, base_price)

    if bar_volume <= 0 or not np.isfinite(bar_volume):
        participation = 0.0
    else:
        participation = min(1.0, float(shares) / float(bar_volume))

    is_buy = side == OrderSide.BUY
    coeff = config.impact_bps_buy_coeff if is_buy else config.impact_bps_sell_coeff

    if config.adaptive_impact and np.isfinite(rolling_vol):
        scale = max(0.25, min(4.0, float(rolling_vol) / config.impact_vol_ref))
        coeff *= scale

    impact_bps = coeff * (participation ** config.impact_exponent)
    sign = 1.0 if is_buy else -1.0
    return max(0.0, base_price * (1.0 + sign * impact_bps / 10_000.0))


class CostModelFillSimulator:
    """Backtest fill simulator with configurable cost model.

    Responsibilities (single-responsibility principle):
      1. Determine if a pending order can fill against a bar's OHLCV
      2. Compute the execution price (slippage + market impact)
      3. Enforce liquidity constraints (max participation rate)
      4. Return Fill objects — does NOT mutate portfolio state

    The caller (BacktestEngine) is responsible for applying fills to the portfolio.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._used_volume: dict[str, float] = {}

    def reset_bar(self) -> None:
        """Reset per-bar volume tracking. Call at the start of each bar."""
        self._used_volume.clear()

    def _available_shares(self, symbol: str, bar_volume: float) -> float:
        """Remaining fillable shares for this symbol in the current bar."""
        bar_cap = max(0.0, bar_volume * self._config.max_participation_rate)
        used = self._used_volume.get(symbol, 0.0)
        return max(0.0, bar_cap - used)

    def _record_usage(self, symbol: str, shares: float) -> None:
        self._used_volume[symbol] = self._used_volume.get(symbol, 0.0) + shares

    def _make_fill(
        self,
        order: Order,
        raw_price: float,
        shares: float,
        bar: BarData,
        bar_index: int,
        date: 'pd.Timestamp',
    ) -> Fill:
        """Compute final execution price and commission, return a Fill."""
        cfg = self._config
        if order.side == OrderSide.BUY:
            exec_price = cfg.fill_price_buy(raw_price)
        else:
            exec_price = cfg.fill_price_sell(raw_price)

        exec_price = _compute_market_impact(
            base_price=exec_price,
            side=order.side,
            shares=shares,
            bar_volume=float(bar.volume[bar_index]),
            rolling_vol=float(bar.rolling_vol[bar_index]),
            config=cfg,
        )

        if order.side == OrderSide.BUY:
            commission = cfg.commission_buy(exec_price * shares)
        else:
            commission = cfg.commission_sell(exec_price * shares)

        self._record_usage(order.symbol, shares)

        return Fill(
            order=order,
            exec_price=exec_price,
            exec_shares=shares,
            commission=commission,
            bar_index=bar_index,
            date=date,
        )

    def try_fill_pending(
        self,
        pending: List[Order],
        symbol: str,
        bar: BarData,
        bar_index: int,
        date: 'pd.Timestamp',
    ) -> Tuple[List[Fill], List[Order]]:
        """Match pending orders against current bar OHLCV.

        Returns (fills, remaining_orders).
        """
        symbol_orders = [o for o in pending if o.symbol == symbol]
        other_orders = [o for o in pending if o.symbol != symbol]
        remaining: List[Order] = []
        fills: List[Fill] = []

        bar_open = float(bar.open[bar_index])
        bar_high = float(bar.high[bar_index])
        bar_low = float(bar.low[bar_index])

        for order in sorted(symbol_orders, key=_pending_priority_key):
            if order.shares <= 0:
                remaining.append(order)
                continue

            raw_price = self._match_price(order, bar_open, bar_high, bar_low)
            if raw_price is None:
                remaining.append(order)
                continue

            available = self._available_shares(symbol, float(bar.volume[bar_index]))
            exec_shares = min(order.shares - order.filled_shares, available)
            if exec_shares <= 0:
                remaining.append(order)
                continue

            fill = self._make_fill(order, raw_price, exec_shares, bar, bar_index, date)
            fills.append(fill)

            total_filled = order.filled_shares + exec_shares
            if total_filled >= order.shares:
                order.status = OrderStatus.FILLED
                order.filled_shares = total_filled
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                order.filled_shares = total_filled
                remaining.append(order)

            order.fill_price = fill.exec_price
            order.filled_bar = bar_index
            order.filled_date = date

        return fills, other_orders + remaining

    def execute_market(
        self,
        order: Order,
        ref_price: float,
        bar: BarData,
        bar_index: int,
        date: 'pd.Timestamp',
    ) -> Optional[Fill]:
        """Execute a market order immediately. Returns None if liquidity insufficient."""
        available = self._available_shares(order.symbol, float(bar.volume[bar_index]))
        exec_shares = min(order.shares, available)
        if exec_shares <= 0:
            return None

        fill = self._make_fill(order, ref_price, exec_shares, bar, bar_index, date)
        order.status = OrderStatus.FILLED
        order.filled_shares = exec_shares
        order.fill_price = fill.exec_price
        order.filled_bar = bar_index
        order.filled_date = date
        return fill

    def _match_price(
        self,
        order: Order,
        bar_open: float,
        bar_high: float,
        bar_low: float,
    ) -> Optional[float]:
        """Determine if and at what price a pending order fills.

        Market orders fill at bar_open (next-open model).
        Limit/stop orders check range trigger AND apply probabilistic queue
        position model: the deeper inside the bar's range the limit price sits,
        the higher the fill probability — simulating order book queue depth.
        """
        if order.order_type == OrderType.MARKET:
            return bar_open

        if order.order_type == OrderType.LIMIT:
            lp = order.limit_price or 0.0
            if order.side == OrderSide.BUY and bar_low <= lp:
                fill_prob = self._limit_fill_probability(lp, bar_low, bar_high, "buy")
                if fill_prob >= 0.5:
                    return min(bar_open, lp)
                return None
            if order.side == OrderSide.SELL and bar_high >= lp:
                fill_prob = self._limit_fill_probability(lp, bar_low, bar_high, "sell")
                if fill_prob >= 0.5:
                    return max(bar_open, lp)
                return None

        if order.order_type == OrderType.STOP:
            sp = order.stop_price or 0.0
            if order.side == OrderSide.BUY and bar_high >= sp:
                return max(bar_open, sp)
            if order.side == OrderSide.SELL and bar_low <= sp:
                return min(bar_open, sp)

        return None

    @staticmethod
    def _limit_fill_probability(
        limit_price: float,
        bar_low: float,
        bar_high: float,
        side: str,
    ) -> float:
        """Estimate fill probability based on how deep the limit price
        penetrates the bar's range — a simple queue depth proxy.

        If the limit price is at the bar's edge (barely touched), fill
        probability is low (~0.3). If deeply inside, probability approaches 1.0.
        """
        bar_range = bar_high - bar_low
        if bar_range < 1e-10:
            return 1.0

        if side == "buy":
            penetration = (limit_price - bar_low) / bar_range
        else:
            penetration = (bar_high - limit_price) / bar_range

        penetration = max(0.0, min(1.0, penetration))
        return 0.3 + 0.7 * penetration
