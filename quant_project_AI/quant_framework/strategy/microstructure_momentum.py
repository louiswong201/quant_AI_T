"""Microstructure Momentum (MSM) Strategy.

Theoretical foundation:
  Traditional momentum (Jegadeesh & Titman 1993) measures past *returns*.
  But returns are a *lagging* indicator — by the time momentum appears in
  price, informed traders have already positioned.  MSM captures momentum
  at its source: **order flow**.

  Primary signal — OFI Rate-of-Change:
    Instead of looking at OFI level, we measure whether buying pressure is
    *accelerating* or *decelerating*.  Acceleration precedes price moves
    because it reflects informed order splitting (Bouchaud et al. 2018,
    "Trades, Quotes and Prices").

  Confirmation filter — VPIN:
    Volume-Synchronized Probability of Informed Trading (Easley, Lopez de
    Prado & O'Hara 2012).  High VPIN = informed traders are active =
    OFI direction is reliable.  Low VPIN = noise-dominated = sit out.

  Volatility gate — Yang-Zhang vol:
    Entry only when vol is in a "Goldilocks zone" (vol_min..vol_max).
    Too-low vol = no opportunity; too-high vol = tail risk / flash crash.

  Adaptive exit — Vol-of-Vol trailing stop:
    Stop distance = ATR * (base_mult - vov_tightener * normalized_vov).
    When vol-of-vol spikes → regime uncertainty → tighter stop → faster
    exit.  When vol-of-vol is calm → let profits run.  This is a direct
    application of Heston (1993) stochastic volatility: vol itself is
    volatile, so static stops are sub-optimal.

References:
  - Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers"
  - Easley, Lopez de Prado & O'Hara (2012), "Flow Toxicity and Liquidity"
  - Bouchaud, Bonart, Donier & Gould (2018), "Trades, Quotes and Prices"
  - Heston (1993), "A Closed-Form Solution for Options with Stochastic
    Volatility"
  - Yang & Zhang (2000), "Drift Independent Volatility Estimation"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from ..alpha.order_flow import _ofi_numba, _vpin_numba, _trade_imbalance_numba
from ..alpha.volatility import _yang_zhang_vol_numba, _vol_of_vol_numba
from .base_strategy import BaseStrategy


# ── Numba kernels ────────────────────────────────────────────────────

def _atr_kernel_impl(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int,
) -> np.ndarray:
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
    atr = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return atr
    s = 0.0
    for i in range(period):
        s += tr[i]
    atr[period - 1] = s / period
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _ofi_roc_impl(ofi: np.ndarray, roc_period: int) -> np.ndarray:
    """Rate-of-change of OFI — measures flow *acceleration*."""
    n = len(ofi)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(roc_period, n):
        prev = ofi[i - roc_period]
        if np.isnan(prev) or np.isnan(ofi[i]):
            continue
        denom = abs(prev) + 1e-10
        out[i] = (ofi[i] - prev) / denom
    return out


if NUMBA_AVAILABLE:
    _atr_kernel = njit(cache=True, fastmath=True)(_atr_kernel_impl)
    _ofi_roc_kernel = njit(cache=True, fastmath=True)(_ofi_roc_impl)
else:
    _atr_kernel = _atr_kernel_impl
    _ofi_roc_kernel = _ofi_roc_impl


# ── Strategy ─────────────────────────────────────────────────────────

class MicrostructureMomentum(BaseStrategy):
    """Microstructure Momentum — order flow acceleration with VPIN filter,
    volatility gate, and vol-of-vol adaptive trailing stop.

    Parameters:
        ofi_window:       OFI smoothing window (default 20)
        ofi_roc_period:   Bars to measure OFI rate-of-change (default 5)
        vpin_buckets:     Number of volume buckets for VPIN (default 50)
        yz_window:        Yang-Zhang vol window (default 20)
        vov_vol_window:   Vol window for vol-of-vol calculation (default 20)
        vov_vov_window:   Outer window for vol-of-vol (default 20)
        atr_period:       ATR period for trailing stop (default 14)
        entry_ofi_roc:    OFI RoC threshold for entry (default 0.3)
        vpin_threshold:   Minimum VPIN for informed flow confirmation (default 0.3)
        vol_min:          Minimum annualized vol to trade (default 0.10)
        vol_max:          Maximum annualized vol to trade (default 0.80)
        atr_base_mult:    Base ATR multiplier for stop (default 3.0)
        vov_tightener:    How much vol-of-vol tightens the stop (default 2.0)
        max_risk_pct:     Portfolio risk cap per trade (default 0.02)
    """

    def __init__(
        self,
        name: str = "MicrostructureMomentum",
        initial_capital: float = 1_000_000,
        ofi_window: int = 20,
        ofi_roc_period: int = 5,
        vpin_buckets: int = 50,
        yz_window: int = 20,
        vov_vol_window: int = 20,
        vov_vov_window: int = 20,
        atr_period: int = 14,
        entry_ofi_roc: float = 0.3,
        vpin_threshold: float = 0.3,
        vol_min: float = 0.10,
        vol_max: float = 0.80,
        atr_base_mult: float = 3.0,
        vov_tightener: float = 2.0,
        max_risk_pct: float = 0.02,
        rag_provider: Any = None,
    ) -> None:
        super().__init__(name, initial_capital, rag_provider=rag_provider)
        self.ofi_window = ofi_window
        self.ofi_roc_period = ofi_roc_period
        self.vpin_buckets = vpin_buckets
        self.yz_window = yz_window
        self.vov_vol_window = vov_vol_window
        self.vov_vov_window = vov_vov_window
        self.atr_period = atr_period
        self.entry_ofi_roc = entry_ofi_roc
        self.vpin_threshold = vpin_threshold
        self.vol_min = vol_min
        self.vol_max = vol_max
        self.atr_base_mult = atr_base_mult
        self.vov_tightener = vov_tightener
        self.max_risk_pct = max_risk_pct
        self._min_lookback = max(
            ofi_window + ofi_roc_period,
            vov_vol_window + vov_vov_window,
            yz_window,
            atr_period,
        ) + 5
        self._trailing_stop: Optional[float] = None
        self._last_vpin: float = 0.0

    @property
    def fast_columns(self) -> Tuple[str, ...]:
        return ("open", "high", "low", "close", "volume")

    # ── Core computation ─────────────────────────────────────────

    def _compute_signal(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        i: int,
    ) -> Tuple[float, float, float, float, float]:
        """Compute all signals at bar i.

        Returns (ofi_roc, vpin_val, yz_vol_ann, vov_val, atr_val).
        """
        n = i + 1
        c = close[:n]
        h = high[:n]
        l = low[:n]
        o = open_[:n]
        v = volume[:n]

        # 1. OFI and its rate-of-change (flow acceleration)
        ofi = _ofi_numba(h, l, c, v, self.ofi_window)
        ofi_roc = _ofi_roc_kernel(ofi, self.ofi_roc_period)
        ofi_roc_val = ofi_roc[i] if not np.isnan(ofi_roc[i]) else 0.0

        # 2. VPIN — informed trading probability
        vpin = _vpin_numba(c, v, self.vpin_buckets)
        vpin_val = np.nan
        for j in range(i, max(-1, i - 10), -1):
            if not np.isnan(vpin[j]):
                vpin_val = vpin[j]
                break
        if np.isnan(vpin_val):
            vpin_val = self._last_vpin
        else:
            self._last_vpin = vpin_val

        # 3. Yang-Zhang vol for Goldilocks gate
        yz = _yang_zhang_vol_numba(o, h, l, c, self.yz_window)
        yz_val = yz[i] if not np.isnan(yz[i]) else 0.02
        yz_ann = yz_val * np.sqrt(252)

        # 4. Vol-of-Vol for adaptive stop
        vov = _vol_of_vol_numba(c, self.vov_vol_window, self.vov_vov_window)
        vov_val = vov[i] if not np.isnan(vov[i]) else 0.0

        # 5. ATR for stop calculation
        atr = _atr_kernel(h, l, c, self.atr_period)
        atr_val = atr[i] if not np.isnan(atr[i]) else 0.0

        return ofi_roc_val, vpin_val, yz_ann, vov_val, atr_val

    def _adaptive_stop_distance(self, atr_val: float, vov_val: float) -> float:
        """Compute trailing stop distance, tightened by vol-of-vol.

        High vov → regime uncertainty → tighter stop (faster exit).
        Low vov → calm regime → wider stop (let profits run).
        """
        vov_norm = min(vov_val * 100.0, 1.0)
        mult = max(1.0, self.atr_base_mult - self.vov_tightener * vov_norm)
        return atr_val * mult

    def _inverse_vol_size(self, price: float, yz_ann: float) -> int:
        """Inverse-vol position sizing capped at 95% of portfolio."""
        if yz_ann < 0.01:
            yz_ann = 0.01
        target = self.portfolio_value * self.max_risk_pct / yz_ann
        target = min(target, self.portfolio_value * 0.95)
        return max(0, int(target / price))

    # ── on_bar_fast (primary hot path) ───────────────────────────

    def on_bar_fast(
        self,
        data_arrays: Dict[str, Any],
        i: int,
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[Union[Dict, List[Dict]]]:
        close = data_arrays.get("close")
        high = data_arrays.get("high")
        low = data_arrays.get("low")
        open_ = data_arrays.get("open")
        volume = data_arrays.get("volume")
        symbol = data_arrays.get("symbol", "STOCK")

        if close is None or high is None or low is None or open_ is None or volume is None:
            return None
        if i < self._min_lookback:
            return {"action": "hold"}

        price = float(close[i])
        holdings = self.positions.get(symbol, 0)

        ofi_roc, vpin_val, yz_ann, vov_val, atr_val = self._compute_signal(
            open_, high, low, close, volume, i,
        )

        # Adaptive trailing stop management
        if holdings > 0 and self._trailing_stop is not None:
            if price <= self._trailing_stop:
                self._trailing_stop = None
                return {"action": "sell", "symbol": symbol, "shares": holdings}
            stop_dist = self._adaptive_stop_distance(atr_val, vov_val)
            new_stop = price - stop_dist
            if new_stop > self._trailing_stop:
                self._trailing_stop = new_stop

        # Entry conditions:
        #   1. OFI accelerating (buying pressure increasing)
        #   2. VPIN confirms informed flow
        #   3. Vol in Goldilocks zone
        if holdings == 0:
            vol_in_zone = self.vol_min <= yz_ann <= self.vol_max
            ofi_strong = ofi_roc > self.entry_ofi_roc
            informed_flow = vpin_val >= self.vpin_threshold

            if ofi_strong and informed_flow and vol_in_zone:
                shares = self._inverse_vol_size(price, yz_ann)
                if shares > 0 and self.can_buy(symbol, price, shares):
                    stop_dist = self._adaptive_stop_distance(atr_val, vov_val)
                    self._trailing_stop = price - stop_dist
                    return {"action": "buy", "symbol": symbol, "shares": shares}

        # Exit on OFI reversal (flow decelerating sharply)
        if holdings > 0 and ofi_roc < -self.entry_ofi_roc:
            self._trailing_stop = None
            return {"action": "sell", "symbol": symbol, "shares": holdings}

        return {"action": "hold"}

    # ── on_bar (DataFrame fallback) ──────────────────────────────

    def on_bar(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Union[Dict, List[Dict]]:
        if isinstance(data, dict):
            symbol = next(iter(data))
            df = data[symbol]
        else:
            df = data
            symbol = df.attrs.get("symbol", "STOCK") if hasattr(df, "attrs") else "STOCK"

        if len(df) < self._min_lookback:
            return {"action": "hold"}

        arrays: Dict[str, Any] = {
            "open": df["open"].values.astype(np.float64) if "open" in df.columns else df["close"].values.astype(np.float64),
            "high": df["high"].values.astype(np.float64) if "high" in df.columns else df["close"].values.astype(np.float64),
            "low": df["low"].values.astype(np.float64) if "low" in df.columns else df["close"].values.astype(np.float64),
            "close": df["close"].values.astype(np.float64),
            "volume": df["volume"].values.astype(np.float64) if "volume" in df.columns else np.ones(len(df), dtype=np.float64),
            "symbol": symbol,
        }
        return self.on_bar_fast(arrays, len(df) - 1, current_date, current_prices)
