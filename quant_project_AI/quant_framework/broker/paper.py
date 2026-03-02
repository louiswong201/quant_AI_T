"""Paper trading broker — supports long, short, and fractional positions.

Simulates fills in memory with configurable costs. Consistent with
BacktestConfig so backtest and paper trading use identical cost models.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

from .base import Broker
if TYPE_CHECKING:
    from ..backtest.config import BacktestConfig

PositionQty = Union[int, float]


class PaperBroker(Broker):
    """Paper broker with short selling, fractional shares, and margin support.

    Key changes from legacy:
      - Sells beyond current position create short positions (negative qty)
      - Fractional shares supported when allow_fractional=True
      - Short MTM: cash += 2*entry*qty - qty*current_price
      - Entry prices tracked for accurate short P&L
    """

    def __init__(
        self,
        initial_cash: float = 0.0,
        initial_positions: Optional[Dict[str, PositionQty]] = None,
        fill_price_callback: Optional[Callable[[Dict[str, Any], str], float]] = None,
        commission_pct_buy: float = 0.0,
        commission_pct_sell: float = 0.0,
        slippage_bps_buy: float = 0.0,
        slippage_bps_sell: float = 0.0,
        allow_short: bool = False,
        allow_fractional: bool = False,
    ):
        self._initial_cash_stored = float(initial_cash)
        self._cash = float(initial_cash)
        self._positions: Dict[str, PositionQty] = dict(initial_positions) if initial_positions else {}
        self._entry_prices: Dict[str, float] = {}
        self._fill_price_cb = fill_price_callback
        self._commission_pct_buy = float(commission_pct_buy)
        self._commission_pct_sell = float(commission_pct_sell)
        self._slippage_bps_buy = float(slippage_bps_buy)
        self._slippage_bps_sell = float(slippage_bps_sell)
        self._allow_short = allow_short
        self._allow_fractional = allow_fractional
        for v in (
            self._commission_pct_buy,
            self._commission_pct_sell,
            self._slippage_bps_buy,
            self._slippage_bps_sell,
        ):
            if v < 0:
                raise ValueError("commission/slippage must be >= 0")
        self._orders: List[Dict[str, Any]] = []

    @classmethod
    def from_backtest_config(
        cls,
        config: "BacktestConfig",
        *,
        initial_cash: float = 0.0,
        initial_positions: Optional[Dict[str, PositionQty]] = None,
        fill_price_callback: Optional[Callable[[Dict[str, Any], str], float]] = None,
    ) -> "PaperBroker":
        """Build from BacktestConfig for cost consistency between backtest and paper."""
        return cls(
            initial_cash=initial_cash,
            initial_positions=initial_positions,
            fill_price_callback=fill_price_callback,
            commission_pct_buy=config.commission_pct_buy,
            commission_pct_sell=config.commission_pct_sell,
            slippage_bps_buy=config.slippage_bps_buy,
            slippage_bps_sell=config.slippage_bps_sell,
            allow_short=config.allow_short,
            allow_fractional=config.allow_fractional_shares,
        )

    def submit_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        action = (signal.get("action") or "").lower()
        symbol = signal.get("symbol") or ""
        raw_shares = signal.get("shares") or 0
        shares: PositionQty = float(raw_shares) if self._allow_fractional else int(raw_shares)
        if action not in ("buy", "sell") or not symbol or shares <= 0:
            return {"status": "rejected", "message": "invalid signal"}

        price = self._resolve_price(signal, action)
        if price <= 0:
            return {"status": "rejected", "message": "no fill price"}

        if action == "buy":
            exec_price = price * (1.0 + self._slippage_bps_buy / 10000.0)
            notional = exec_price * shares
            commission = notional * self._commission_pct_buy
            total_cost = notional + commission
            if self._cash < total_cost:
                return {"status": "rejected", "message": "insufficient cash"}

            self._cash -= total_cost
            old_pos = self._positions.get(symbol, 0)
            new_pos = old_pos + shares

            if old_pos < 0 and new_pos > 0:
                self._entry_prices[symbol] = exec_price
            elif old_pos < 0 and new_pos == 0:
                self._entry_prices.pop(symbol, None)
            elif old_pos <= 0 and new_pos > 0:
                self._entry_prices[symbol] = exec_price
            elif old_pos > 0:
                old_entry = self._entry_prices.get(symbol, exec_price)
                self._entry_prices[symbol] = (old_entry * old_pos + exec_price * shares) / new_pos

            self._update_position(symbol, new_pos)

        else:
            pos = self._positions.get(symbol, 0)
            if not self._allow_short:
                shares = min(shares, max(0, pos))
                if shares <= 0:
                    return {"status": "rejected", "message": "insufficient position (short disabled)"}

            exec_price = price * (1.0 - self._slippage_bps_sell / 10000.0)
            notional = exec_price * shares
            commission = notional * self._commission_pct_sell
            self._cash += notional - commission

            old_pos = pos
            new_pos = old_pos - shares

            if old_pos > 0 and new_pos <= 0:
                self._entry_prices.pop(symbol, None)
                if new_pos < 0:
                    self._entry_prices[symbol] = exec_price
            elif old_pos <= 0 and new_pos < 0:
                old_entry = self._entry_prices.get(symbol, exec_price)
                self._entry_prices[symbol] = (
                    (old_entry * abs(old_pos) + exec_price * shares) / abs(new_pos)
                )

            self._update_position(symbol, new_pos)

        self._orders.append({
            **copy.deepcopy(signal),
            "side": action,
            "fill_price": exec_price,
            "filled_shares": shares,
            "commission": commission,
            "status": "filled",
        })
        return {
            "status": "filled",
            "fill_price": exec_price,
            "filled_shares": shares,
            "commission": commission,
        }

    def _resolve_price(self, signal: Dict[str, Any], action: str) -> float:
        if self._fill_price_cb:
            return self._fill_price_cb(signal, action)
        return float(signal.get("price") or signal.get("fill_price") or 0)

    def _update_position(self, symbol: str, qty: PositionQty) -> None:
        if abs(qty) < 1e-12:
            self._positions.pop(symbol, None)
            self._entry_prices.pop(symbol, None)
        else:
            self._positions[symbol] = qty

    def get_positions(self) -> Dict[str, PositionQty]:
        return dict(self._positions)

    def get_cash(self) -> float:
        return self._cash

    def get_orders(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._orders)

    def get_trade_history(self) -> List[Dict[str, Any]]:
        return [o for o in self.get_orders() if o.get("status") == "filled"]

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """MTM: cash + sum(qty * px). Short qty is negative, so shorts
        subtract their current liability from portfolio value."""
        value = self._cash
        for sym, qty in self._positions.items():
            px = prices.get(sym, 0.0)
            value += qty * px
        return value

    @property
    def initial_cash(self) -> float:
        return self._initial_cash_stored
