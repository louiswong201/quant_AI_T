from __future__ import annotations

import math
from abc import ABC, abstractmethod
import numpy as np


class MarginModel(ABC):
    @abstractmethod
    def initial_margin(self, notional: float, leverage: float = 1.0) -> float:
        ...

    @abstractmethod
    def maintenance_margin(self, notional: float) -> float:
        ...


class CryptoSpotMargin(MarginModel):
    """100% margin, no liquidation."""

    def initial_margin(self, notional: float, leverage: float = 1.0) -> float:
        return notional

    def maintenance_margin(self, notional: float) -> float:
        return notional


class CryptoFuturesMargin(MarginModel):
    """Binance tiered maintenance rates, isolated/cross margin."""

    TIERS: list[tuple[float, float, float]] = [
        (50_000, 0.004, 0),
        (250_000, 0.005, 50),
        (1_000_000, 0.01, 1_050),
        (10_000_000, 0.025, 16_050),
        (20_000_000, 0.05, 266_050),
        (50_000_000, 0.10, 1_266_050),
        (100_000_000, 0.125, 2_516_050),
        (200_000_000, 0.15, 5_016_050),
        (math.inf, 0.25, 25_016_050),
    ]

    _TIER_BOUNDS = np.array([50_000, 250_000, 1_000_000, 10_000_000,
                              20_000_000, 50_000_000, 100_000_000, 200_000_000])
    _TIER_RATES = np.array([0.004, 0.005, 0.01, 0.025, 0.05, 0.10, 0.125, 0.15, 0.25])
    _TIER_AMOUNTS = np.array([0, 50, 1_050, 16_050, 266_050, 1_266_050,
                              2_516_050, 5_016_050, 25_016_050])

    def initial_margin(self, notional: float, leverage: float = 1.0) -> float:
        leverage = max(leverage, 1.0)
        return notional / leverage

    def maintenance_margin(self, notional: float) -> float:
        idx = np.searchsorted(self._TIER_BOUNDS, notional)
        return notional * float(self._TIER_RATES[idx]) - float(self._TIER_AMOUNTS[idx])

    def liquidation_price_long(self, entry: float, leverage: float, maint_rate: float) -> float:
        leverage = max(leverage, 1.0)
        return entry * (1 - 1 / leverage + maint_rate)

    def liquidation_price_short(self, entry: float, leverage: float, maint_rate: float) -> float:
        leverage = max(leverage, 1.0)
        return entry * (1 + 1 / leverage - maint_rate)

    def maintenance_rate_for_notional(self, notional: float) -> float:
        idx = np.searchsorted(self._TIER_BOUNDS, notional)
        return float(self._TIER_RATES[idx])

    def available_balance_cross(
        self,
        total_balance: float,
        all_positions: dict[str, float],
        prices: dict[str, float],
        entry_prices: dict[str, float] | None = None,
    ) -> float:
        total_maint = sum(
            self.maintenance_margin(abs(qty) * prices.get(sym, 0))
            for sym, qty in all_positions.items()
        )
        entries = entry_prices or {}
        unrealized = sum(
            self._unrealized_pnl(sym, qty, prices.get(sym, 0), entries.get(sym, 0))
            for sym, qty in all_positions.items()
        )
        return total_balance + unrealized - total_maint

    def _unrealized_pnl(
        self, sym: str, qty: float, price: float, entry: float
    ) -> float:
        if price <= 0 or qty == 0:
            return 0.0
        if qty > 0:
            return qty * (price - entry)
        return abs(qty) * (entry - price)


class RegTMargin(MarginModel):
    """US equity Reg T: 50% initial, 25% maintenance, PDT threshold 25000."""

    INITIAL_RATE = 0.50
    MAINTENANCE_RATE = 0.25
    PDT_EQUITY_THRESHOLD = 25_000

    def initial_margin(self, notional: float, leverage: float = 2.0) -> float:
        return notional * self.INITIAL_RATE

    def maintenance_margin(self, market_value: float) -> float:
        return market_value * self.MAINTENANCE_RATE

    def check_margin_call(self, equity: float, market_value: float) -> bool:
        return equity < market_value * self.MAINTENANCE_RATE

    def margin_call_deadline_days(self) -> int:
        return 2
