"""
Backtest configuration: costs, execution, leverage, shorting, stop-loss, funding.

Unified config that brings the Numba-kernel-level realism (leverage, margin,
stop-loss, funding/borrow costs) into the framework engine, eliminating the
dual-backtest divergence between BacktestEngine and examples/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

_BARS_PER_YEAR_CRYPTO = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040,
    "1h": 8_760, "4h": 2_190, "1d": 365,
}
_BARS_PER_YEAR_STOCK = {
    "1m": 98_280, "5m": 19_656, "15m": 6_552,
    "1h": 1_764, "4h": 441, "1d": 252,
}
_BARS_PER_DAY_CRYPTO = {
    "1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1,
}
_BARS_PER_DAY_STOCK = {
    "1m": 390, "5m": 78, "15m": 26, "1h": 7, "4h": 2, "1d": 1,
}
_VALID_INTERVALS = ("1m", "5m", "15m", "1h", "4h", "1d")


@dataclass(frozen=True)
class BacktestConfig:
    """
    Unified backtest configuration — costs, execution, leverage, risk controls.

    Supports arbitrary bar intervals via ``interval``, ``bars_per_year``, and
    ``bars_per_day`` which control annualization and per-bar cost scaling.
    """

    # -- Asset class (V5) ------------------------------------------------
    asset_class: str = "crypto_perp"
    margin_model_type: str = "default"  # "default"|"crypto_futures"|"reg_t"|"spot"

    # -- Market fill mode ------------------------------------------------
    market_fill_mode: str = "next_open"

    # -- Commission: maker/taker distinction (V5) -------------------------
    commission_pct_maker: float = 0.0002
    commission_pct_taker: float = 0.0004
    maker_ratio: float = 0.3

    # -- Commission: US equity IBKR fixed (V5) ----------------------------
    commission_per_share: float = 0.005
    commission_min: float = 1.0
    exchange_fee_per_share: float = 0.003

    # -- Margin interest (V5) --------------------------------------------
    margin_interest_rate: float = 0.0583

    # -- Commission (proportion, 0.001 = 0.1%) ---------------------------
    commission_pct_buy: float = 0.001
    commission_pct_sell: float = 0.001
    commission_fixed_buy: float = 0.0
    commission_fixed_sell: float = 0.0

    # -- Slippage (bps or fixed) -----------------------------------------
    slippage_bps_buy: float = 0.0
    slippage_bps_sell: float = 0.0
    slippage_fixed_buy: float = 0.0
    slippage_fixed_sell: float = 0.0

    # -- Liquidity / market impact ---------------------------------------
    max_participation_rate: float = 1.0
    impact_bps_buy_coeff: float = 0.0
    impact_bps_sell_coeff: float = 0.0
    impact_exponent: float = 1.0
    adaptive_impact: bool = False
    impact_vol_window: int = 20
    impact_vol_ref: float = 0.02

    # -- Leverage --------------------------------------------------------
    leverage: float = 1.0
    allow_fractional_shares: bool = False

    # -- Shorting --------------------------------------------------------
    allow_short: bool = False

    # -- Stop-loss / take-profit -----------------------------------------
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    stop_loss_slippage_pct: float = 0.005

    # -- Funding / borrow costs ------------------------------------------
    daily_funding_rate: float = 0.0
    funding_leverage_scaling: bool = True
    short_borrow_rate_annual: float = 0.0
    borrow_rates_by_symbol: Optional[Dict[str, float]] = None

    # -- Margin / liquidation --------------------------------------------
    initial_margin_pct: float = 1.0
    maintenance_margin_pct: float = 0.0
    liquidation_threshold_pct: float = 0.80

    # -- Position sizing -------------------------------------------------
    position_fraction: float = 1.0

    # -- Bar interval / annualization ------------------------------------
    interval: str = "1d"
    bars_per_year: float = 252.0
    bars_per_day: float = 1.0

    # -- Execution reporting ---------------------------------------------
    auto_export_execution_report: bool = False
    execution_report_path: str = "docs/execution_divergence_report.md"

    def __post_init__(self) -> None:
        if self.market_fill_mode not in ("next_open", "current_close"):
            raise ValueError(
                f"market_fill_mode must be 'next_open' or 'current_close', "
                f"got '{self.market_fill_mode}'"
            )
        _NON_NEG = (
            "commission_pct_buy", "commission_pct_sell",
            "commission_fixed_buy", "commission_fixed_sell",
            "slippage_bps_buy", "slippage_bps_sell",
            "slippage_fixed_buy", "slippage_fixed_sell",
            "daily_funding_rate", "short_borrow_rate_annual",
            "stop_loss_slippage_pct", "initial_margin_pct",
            "maintenance_margin_pct", "liquidation_threshold_pct",
        )
        for name in _NON_NEG:
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")

        if self.leverage < 1.0:
            raise ValueError(f"leverage must be >= 1.0, got {self.leverage}")
        if not (0.0 < self.max_participation_rate <= 1.0):
            raise ValueError(
                f"max_participation_rate must be in (0, 1], got {self.max_participation_rate}"
            )
        if self.impact_exponent <= 0:
            raise ValueError(f"impact_exponent must be > 0, got {self.impact_exponent}")
        if self.impact_vol_window <= 1:
            raise ValueError(f"impact_vol_window must be > 1, got {self.impact_vol_window}")
        if self.impact_vol_ref <= 0:
            raise ValueError(f"impact_vol_ref must be > 0, got {self.impact_vol_ref}")
        if self.stop_loss_pct is not None and not (0.0 < self.stop_loss_pct <= 1.0):
            raise ValueError(f"stop_loss_pct must be in (0, 1], got {self.stop_loss_pct}")
        if self.take_profit_pct is not None and self.take_profit_pct <= 0:
            raise ValueError(f"take_profit_pct must be > 0, got {self.take_profit_pct}")
        if self.trailing_stop_pct is not None and not (0.0 < self.trailing_stop_pct <= 1.0):
            raise ValueError(f"trailing_stop_pct must be in (0, 1], got {self.trailing_stop_pct}")
        if not (0.0 < self.position_fraction <= 1.0):
            raise ValueError(f"position_fraction must be in (0, 1], got {self.position_fraction}")

    # -- Dynamic cost helpers --------------------------------------------

    def effective_slippage_bps(self, side: str) -> float:
        """Leverage-scaled slippage: base * sqrt(leverage)."""
        base = self.slippage_bps_buy if side == "buy" else self.slippage_bps_sell
        return base * math.sqrt(self.leverage)

    def daily_funding_cost(self) -> float:
        """Per-day funding cost rate, optionally scaled by leverage."""
        rate = self.daily_funding_rate
        if self.funding_leverage_scaling and self.leverage > 1.0:
            rate *= self.leverage * (1.0 + 0.02 * self.leverage)
        return rate

    def borrow_rate_for(self, symbol: str) -> float:
        """Per-bar borrow cost for short positions."""
        if self.borrow_rates_by_symbol and symbol in self.borrow_rates_by_symbol:
            annual = self.borrow_rates_by_symbol[symbol]
        else:
            annual = self.short_borrow_rate_annual
        return annual / self.bars_per_year

    # -- Factory methods -------------------------------------------------

    @classmethod
    def from_legacy_rate(cls, commission_rate: float) -> BacktestConfig:
        """Backward-compatible: symmetric proportional commission only."""
        return cls(
            commission_pct_buy=commission_rate,
            commission_pct_sell=commission_rate,
        )

    @classmethod
    def conservative(cls) -> BacktestConfig:
        """Conservative preset: slightly above typical live costs."""
        return cls(
            commission_pct_buy=0.0015,
            commission_pct_sell=0.0015,
            slippage_bps_buy=5.0,
            slippage_bps_sell=5.0,
        )

    @classmethod
    def crypto(
        cls,
        leverage: float = 1.0,
        stop_loss_pct: Optional[float] = None,
        interval: str = "1d",
    ) -> BacktestConfig:
        """Binance-style crypto preset. Supports any interval (1m/5m/15m/1h/4h/1d)."""
        bpy = _BARS_PER_YEAR_CRYPTO.get(interval, 365)
        bpd = _BARS_PER_DAY_CRYPTO.get(interval, 1)
        return cls(
            commission_pct_buy=0.0004,
            commission_pct_sell=0.0004,
            slippage_bps_buy=3.0,
            slippage_bps_sell=3.0,
            leverage=leverage,
            allow_short=True,
            allow_fractional_shares=True,
            daily_funding_rate=0.0003,
            funding_leverage_scaling=True,
            stop_loss_pct=stop_loss_pct,
            liquidation_threshold_pct=0.80,
            interval=interval,
            bars_per_year=float(bpy),
            bars_per_day=float(bpd),
        )

    @classmethod
    def stock_ibkr(
        cls,
        leverage: float = 1.0,
        stop_loss_pct: Optional[float] = None,
        interval: str = "1d",
    ) -> BacktestConfig:
        """IBKR-style stock preset. Supports any interval (1m/5m/15m/1h/4h/1d)."""
        bpy = _BARS_PER_YEAR_STOCK.get(interval, 252)
        bpd = _BARS_PER_DAY_STOCK.get(interval, 1)
        return cls(
            commission_pct_buy=0.0005,
            commission_pct_sell=0.0005,
            slippage_bps_buy=5.0,
            slippage_bps_sell=5.0,
            leverage=leverage,
            allow_short=True,
            short_borrow_rate_annual=0.03,
            stop_loss_pct=stop_loss_pct,
            initial_margin_pct=0.5 if leverage <= 2.0 else 1.0 / leverage,
            interval=interval,
            bars_per_year=float(bpy),
            bars_per_day=float(bpd),
        )

    # -- Price helpers ---------------------------------------------------

    def fill_price_buy(self, ref_price: float) -> float:
        """Execution price for buy (including leverage-scaled slippage)."""
        bps = self.effective_slippage_bps("buy")
        px = ref_price * (1.0 + bps / 10000.0) + self.slippage_fixed_buy
        return max(0.0, px)

    def fill_price_sell(self, ref_price: float) -> float:
        """Execution price for sell (including leverage-scaled slippage)."""
        bps = self.effective_slippage_bps("sell")
        px = ref_price * (1.0 - bps / 10000.0) - self.slippage_fixed_sell
        return max(0.0, px)

    def commission_buy(self, notional: float) -> float:
        return self.commission_fixed_buy + notional * self.commission_pct_buy

    def commission_sell(self, notional: float) -> float:
        return self.commission_fixed_sell + notional * self.commission_pct_sell
