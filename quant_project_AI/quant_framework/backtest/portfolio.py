"""
Portfolio tracker: positions, cash, equity curve, trade journal.

Supports:
  - Long AND short positions (negative shares = short)
  - Fractional shares (float positions when config allows)
  - Leverage-aware margin accounting
  - Per-bar funding/borrow cost deduction
  - Stop-loss / take-profit / trailing-stop enforcement
  - Liquidation checks

The tracker is the SINGLE source of truth for cash and positions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .protocols import Fill, OrderSide

PositionQty = Union[int, float]


class PortfolioTracker:
    """Tracks portfolio state through the backtest lifecycle.

    Owns: cash, positions (long+short), equity curve arrays, trade journal.
    Handles: funding costs, stop-loss, liquidation.
    Does NOT own: order management or fill simulation.
    """

    def __init__(
        self,
        initial_capital: float,
        n_bars: int,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: Dict[str, PositionQty] = {}
        self._entry_prices: Dict[str, float] = {}
        self._high_water: Dict[str, float] = {}
        self._portfolio_value = initial_capital

        self._portfolio_values = np.empty(n_bars, dtype=np.float64)
        self._cumulative_returns = np.empty(n_bars, dtype=np.float64)
        self._daily_returns = np.empty(n_bars, dtype=np.float64)
        self._cash_arr = np.empty(n_bars, dtype=np.float64)
        self._dates_arr: List[Any] = [None] * n_bars

        self._trades: List[Dict[str, Any]] = []

        self._config = config or BacktestConfig()
        self._liquidated = False
        self._total_funding_paid = 0.0
        self._total_borrow_paid = 0.0

    # -----------------------------------------------------------------
    # Fill execution
    # -----------------------------------------------------------------

    def apply_fill(self, fill: Fill) -> None:
        """Update positions and cash after a fill. Supports long + short."""
        sym = fill.order.symbol
        shares = fill.exec_shares if self._config.allow_fractional_shares else int(fill.exec_shares)

        if fill.order.side == OrderSide.BUY:
            cost = fill.exec_price * shares + fill.commission
            self._cash -= cost
            old_pos = self._positions.get(sym, 0)
            new_pos = old_pos + shares

            if old_pos <= 0 and new_pos > 0:
                self._entry_prices[sym] = fill.exec_price
                self._high_water[sym] = fill.exec_price
            elif old_pos > 0 and new_pos > 0:
                old_entry = self._entry_prices.get(sym, fill.exec_price)
                self._entry_prices[sym] = (
                    (old_entry * old_pos + fill.exec_price * shares) / new_pos
                )
        else:
            revenue = fill.exec_price * shares - fill.commission
            self._cash += revenue
            old_pos = self._positions.get(sym, 0)
            new_pos = old_pos - shares

            if old_pos >= 0 and new_pos < 0:
                self._entry_prices[sym] = fill.exec_price
                self._high_water[sym] = fill.exec_price
            elif old_pos < 0 and new_pos < 0:
                old_entry = self._entry_prices.get(sym, fill.exec_price)
                self._entry_prices[sym] = (
                    (old_entry * abs(old_pos) + fill.exec_price * shares) / abs(new_pos)
                )

        if new_pos == 0:
            self._positions.pop(sym, None)
            self._entry_prices.pop(sym, None)
            self._high_water.pop(sym, None)
        else:
            self._positions[sym] = new_pos

        self._trades.append({
            "date": fill.date,
            "action": "buy" if fill.order.side == OrderSide.BUY else "sell",
            "symbol": sym,
            "price": fill.exec_price,
            "shares": shares,
            "commission": fill.commission,
            "order_id": fill.order.order_id,
            "order_type": fill.order.order_type.name.lower(),
        })

    # -----------------------------------------------------------------
    # Mark-to-market
    # -----------------------------------------------------------------

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        """Revalue portfolio at current market prices.

        Long: value = qty * px
        Short: cash already includes sale proceeds (entry * |qty|), so the
        short liability is |qty| * px. Net contribution = qty * px (qty < 0).
        Total portfolio = cash + sum(qty * px).
        """
        positions_value = 0.0
        for sym, qty in self._positions.items():
            px = prices.get(sym, 0.0)
            positions_value += qty * px

        self._portfolio_value = self._cash + positions_value
        return self._portfolio_value

    # -----------------------------------------------------------------
    # Per-bar funding, borrow, stop-loss, liquidation
    # -----------------------------------------------------------------

    def process_bar_costs(
        self,
        prices: Dict[str, float],
        bar_index: int,
        date: pd.Timestamp,
    ) -> List[Dict[str, Any]]:
        """Deduct daily funding + borrow costs, check stops and liquidation.

        Returns a list of auto-generated close signals (stop-loss, take-profit,
        trailing-stop, or liquidation) that the engine should execute.
        """
        cfg = self._config
        close_signals: List[Dict[str, Any]] = []

        if self._liquidated:
            return close_signals

        for sym, qty in list(self._positions.items()):
            px = prices.get(sym, 0.0)
            if px <= 0:
                continue

            entry = self._entry_prices.get(sym, px)
            lev = cfg.leverage

            if qty > 0:
                pnl_pct = ((px / entry) - 1.0) * lev if entry > 0 else 0.0
            else:
                pnl_pct = ((entry / px) - 1.0) * lev if px > 0 else 0.0

            # -- Funding cost (leveraged long or short) --
            if qty != 0 and cfg.daily_funding_rate > 0:
                deployed = abs(qty) * entry * cfg.position_fraction
                funding = deployed * cfg.daily_funding_cost()
                self._cash -= funding
                self._total_funding_paid += funding

            # -- Borrow cost (short only) --
            if qty < 0:
                daily_borrow = cfg.borrow_rate_for(sym)
                if daily_borrow > 0:
                    borrow_cost = abs(qty) * px * daily_borrow
                    self._cash -= borrow_cost
                    self._total_borrow_paid += borrow_cost

            # -- Trailing stop --
            if cfg.trailing_stop_pct is not None and qty != 0:
                if qty > 0:
                    hw = self._high_water.get(sym, px)
                    if px > hw:
                        self._high_water[sym] = px
                        hw = px
                    drawdown_from_hw = (hw - px) / hw if hw > 0 else 0.0
                    if drawdown_from_hw >= cfg.trailing_stop_pct:
                        close_signals.append({
                            "action": "sell", "symbol": sym,
                            "shares": abs(qty), "reason": "trailing_stop",
                        })
                        continue
                else:
                    lw = self._high_water.get(sym, px)
                    if px < lw:
                        self._high_water[sym] = px
                        lw = px
                    adverse_move = (px - lw) / lw if lw > 0 else 0.0
                    if adverse_move >= cfg.trailing_stop_pct:
                        close_signals.append({
                            "action": "buy", "symbol": sym,
                            "shares": abs(qty), "reason": "trailing_stop",
                        })
                        continue

            # -- Stop-loss --
            if cfg.stop_loss_pct is not None and pnl_pct <= -cfg.stop_loss_pct:
                action = "sell" if qty > 0 else "buy"
                close_signals.append({
                    "action": action, "symbol": sym,
                    "shares": abs(qty), "reason": "stop_loss",
                })
                continue

            # -- Take-profit --
            if cfg.take_profit_pct is not None and pnl_pct >= cfg.take_profit_pct:
                action = "sell" if qty > 0 else "buy"
                close_signals.append({
                    "action": action, "symbol": sym,
                    "shares": abs(qty), "reason": "take_profit",
                })
                continue

            # -- Liquidation check --
            if lev > 1.0 and pnl_pct <= -cfg.liquidation_threshold_pct:
                action = "sell" if qty > 0 else "buy"
                close_signals.append({
                    "action": action, "symbol": sym,
                    "shares": abs(qty), "reason": "liquidation",
                })
                self._liquidated = True
                continue

        return close_signals

    # -----------------------------------------------------------------
    # Bar recording
    # -----------------------------------------------------------------

    def record_bar(self, bar_index: int, date: pd.Timestamp) -> None:
        pv = self._portfolio_value
        self._portfolio_values[bar_index] = pv
        ic = self._initial_capital
        self._cumulative_returns[bar_index] = (pv - ic) / ic

        if bar_index == 0:
            self._daily_returns[bar_index] = (pv - ic) / ic
        else:
            prev = self._portfolio_values[bar_index - 1]
            self._daily_returns[bar_index] = (pv - prev) / prev if prev > 0 else 0.0

        self._cash_arr[bar_index] = self._cash
        self._dates_arr[bar_index] = date

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------

    def can_afford(self, price: float, shares: PositionQty, commission: float) -> bool:
        return self._cash >= (price * abs(shares) + commission)

    def position_size(self, symbol: str) -> PositionQty:
        return self._positions.get(symbol, 0)

    def to_results_dict(self) -> Dict[str, Any]:
        results_df = pd.DataFrame({
            "date": self._dates_arr,
            "portfolio_value": self._portfolio_values,
            "cash": self._cash_arr,
            "daily_return": self._daily_returns,
            "cumulative_return": self._cumulative_returns,
        })
        return {
            "results": results_df,
            "trades": pd.DataFrame(self._trades) if self._trades else pd.DataFrame(),
            "daily_returns": self._daily_returns,
            "cumulative_returns": self._cumulative_returns,
            "portfolio_values": self._portfolio_values,
            "final_value": self._portfolio_value,
            "initial_capital": self._initial_capital,
            "total_funding_paid": self._total_funding_paid,
            "total_borrow_paid": self._total_borrow_paid,
            "liquidated": self._liquidated,
        }

    @property
    def cash(self) -> float:
        return self._cash

    @cash.setter
    def cash(self, value: float) -> None:
        self._cash = value

    @property
    def positions(self) -> Dict[str, PositionQty]:
        return self._positions

    @property
    def trades(self) -> List[Dict[str, Any]]:
        return self._trades

    @property
    def portfolio_value(self) -> float:
        return self._portfolio_value

    @property
    def is_liquidated(self) -> bool:
        return self._liquidated
