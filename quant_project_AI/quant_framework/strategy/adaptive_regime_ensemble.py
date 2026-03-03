"""Adaptive Regime Ensemble (ARE) Strategy.

Theoretical foundation:
  Markets alternate between *trending* and *mean-reverting* regimes.
  A single-regime strategy hemorrhages returns when the regime flips.
  ARE detects the current regime via Volatility Ratio (fast_vol / slow_vol)
  and smoothly blends two orthogonal sub-signals:

  1. Trend signal    — KAMA direction confirmed by Order Flow Imbalance (OFI).
     KAMA already adapts its speed to trend strength (Kaufman 1995).
     OFI adds microstructure confirmation: is the price move backed by
     genuine directional volume? (Cont, Kukanov & Stoikov 2014).

  2. Reversion signal — Z-score of price vs. KAMA, cross-checked against
     Trade Imbalance divergence.  When price overshoots KAMA (high |z-score|)
     and volume flow disagrees with the move, mean-reversion is likely.

  Regime blending uses a sigmoid on (vol_ratio - 1.0) for a smooth,
  differentiable transition — no hard regime switches that cause whipsaws.

  Position sizing uses inverse Yang-Zhang volatility (risk parity principle,
  Bridgewater All Weather).  Higher vol → smaller position → equal risk
  contribution across time.

References:
  - Hamilton (1989), "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle"
  - Ang & Bekaert (2002), "Regime Switches in Interest Rates"
  - Cont, Kukanov & Stoikov (2014), "The Price Impact of Order Book Events"
  - Kaufman (1995), "Smarter Trading"
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

from ..alpha.order_flow import _ofi_numba, _trade_imbalance_numba
from ..alpha.volatility import _vol_ratio_numba, _yang_zhang_vol_numba
from .base_strategy import BaseStrategy


# ── Numba kernels ────────────────────────────────────────────────────

def _kama_kernel_impl(
    close: np.ndarray, er_period: int, fast_c: float, slow_c: float,
) -> np.ndarray:
    n = len(close)
    kama = np.full(n, np.nan, dtype=np.float64)
    if n < er_period:
        return kama
    kama[er_period - 1] = close[er_period - 1]
    for i in range(er_period, n):
        direction = abs(close[i] - close[i - er_period])
        volatility = 0.0
        for j in range(1, er_period + 1):
            volatility += abs(close[i - j + 1] - close[i - j])
        er = direction / volatility if volatility > 0 else 0.0
        sc = (er * (fast_c - slow_c) + slow_c) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    return kama


def _zscore_kernel_impl(close: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score: (x - mean) / std."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    s = 0.0
    s2 = 0.0
    for i in range(window):
        s += close[i]
        s2 += close[i] * close[i]
    mean = s / window
    var = s2 / window - mean * mean
    if var > 0:
        out[window - 1] = (close[window - 1] - mean) / np.sqrt(var)
    for i in range(window, n):
        old = close[i - window]
        new = close[i]
        s += new - old
        s2 += new * new - old * old
        mean = s / window
        var = s2 / window - mean * mean
        if var > 1e-20:
            out[i] = (new - mean) / np.sqrt(var)
        else:
            out[i] = 0.0
    return out


def _atr_kernel_impl(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int,
) -> np.ndarray:
    """Wilder ATR — O(n) single-pass."""
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


if NUMBA_AVAILABLE:
    _kama_kernel = njit(cache=True, fastmath=True)(_kama_kernel_impl)
    _zscore_kernel = njit(cache=True, fastmath=True)(_zscore_kernel_impl)
    _atr_kernel = njit(cache=True, fastmath=True)(_atr_kernel_impl)
else:
    _kama_kernel = _kama_kernel_impl
    _zscore_kernel = _zscore_kernel_impl
    _atr_kernel = _atr_kernel_impl


# ── Strategy ─────────────────────────────────────────────────────────

class AdaptiveRegimeEnsemble(BaseStrategy):
    """Adaptive Regime Ensemble — blends trend and mean-reversion signals
    with regime-aware weighting and inverse-volatility position sizing.

    Parameters:
        er_period:        KAMA efficiency ratio lookback (default 10)
        fast_period:      KAMA fast constant (default 2)
        slow_period:      KAMA slow constant (default 30)
        ofi_window:       OFI rolling window (default 20)
        ti_window:        Trade Imbalance window (default 20)
        vol_fast:         Fast vol window for regime detection (default 10)
        vol_slow:         Slow vol window for regime detection (default 60)
        yz_window:        Yang-Zhang vol window for position sizing (default 20)
        zscore_window:    Z-score lookback (default 20)
        atr_period:       ATR period for trailing stop (default 14)
        atr_stop_mult:    ATR trailing stop multiplier (default 2.5)
        entry_threshold:  Composite score threshold for entry (default 0.3)
        exit_threshold:   Composite score threshold for exit (default 0.15)
        sigmoid_scale:    Controls regime transition steepness (default 5.0)
        max_risk_pct:     Maximum portfolio risk per trade (default 0.02)
    """

    def __init__(
        self,
        name: str = "AdaptiveRegimeEnsemble",
        initial_capital: float = 1_000_000,
        er_period: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
        ofi_window: int = 20,
        ti_window: int = 20,
        vol_fast: int = 10,
        vol_slow: int = 60,
        yz_window: int = 20,
        zscore_window: int = 20,
        atr_period: int = 14,
        atr_stop_mult: float = 2.5,
        entry_threshold: float = 0.3,
        exit_threshold: float = 0.15,
        sigmoid_scale: float = 5.0,
        max_risk_pct: float = 0.02,
        rag_provider: Any = None,
    ) -> None:
        super().__init__(name, initial_capital, rag_provider=rag_provider)
        self.er_period = er_period
        self.fast_c = 2.0 / (fast_period + 1.0)
        self.slow_c = 2.0 / (slow_period + 1.0)
        self.ofi_window = ofi_window
        self.ti_window = ti_window
        self.vol_fast = vol_fast
        self.vol_slow = vol_slow
        self.yz_window = yz_window
        self.zscore_window = zscore_window
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.sigmoid_scale = sigmoid_scale
        self.max_risk_pct = max_risk_pct
        self._min_lookback = max(vol_slow, yz_window, er_period, zscore_window) + 5
        self._trailing_stop: Dict[str, Optional[float]] = {}

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
    ) -> Tuple[float, float, float]:
        """Compute (composite_score, yang_zhang_vol, atr) at bar i.

        Returns (score, yz_vol, atr_val) where score is in [-1, +1].
        """
        n = i + 1
        c = close[:n]
        h = high[:n]
        l = low[:n]
        o = open_[:n]
        v = volume[:n]

        # 1. Regime detection via vol ratio
        vol_ratio = _vol_ratio_numba(c, self.vol_fast, self.vol_slow)
        vr = vol_ratio[i] if not np.isnan(vol_ratio[i]) else 1.0
        regime_weight = 1.0 / (1.0 + np.exp(-self.sigmoid_scale * (vr - 1.0)))

        # 2. KAMA for trend direction
        kama = _kama_kernel(c, self.er_period, self.fast_c, self.slow_c)
        if i < 1 or np.isnan(kama[i]) or np.isnan(kama[i - 1]):
            return 0.0, np.nan, np.nan
        kama_dir = 1.0 if kama[i] > kama[i - 1] else (-1.0 if kama[i] < kama[i - 1] else 0.0)

        # 3. OFI for microstructure confirmation
        ofi = _ofi_numba(h, l, c, v, self.ofi_window)
        ofi_val = ofi[i] if not np.isnan(ofi[i]) else 0.0
        ofi_std = np.nanstd(ofi[max(0, i - self.vol_slow):i + 1])
        ofi_norm = ofi_val / (ofi_std + 1e-10)
        ofi_norm = max(-3.0, min(3.0, ofi_norm))

        trend_score = kama_dir * abs(ofi_norm) / 3.0

        # 4. Z-score for mean-reversion
        zs = _zscore_kernel(c, self.zscore_window)
        z_val = zs[i] if not np.isnan(zs[i]) else 0.0

        # 5. Trade imbalance divergence
        ti = _trade_imbalance_numba(c, v, self.ti_window)
        ti_val = ti[i] if not np.isnan(ti[i]) else 0.0
        price_dir = 1.0 if c[i] > c[max(0, i - 1)] else -1.0
        divergence = 1.0 if (price_dir > 0 and ti_val < -0.1) or (price_dir < 0 and ti_val > 0.1) else 0.5

        reversion_score = -z_val / 3.0 * divergence
        reversion_score = max(-1.0, min(1.0, reversion_score))

        # 6. Composite blend
        composite = regime_weight * trend_score + (1.0 - regime_weight) * reversion_score

        # 7. Yang-Zhang vol for position sizing
        yz = _yang_zhang_vol_numba(o, h, l, c, self.yz_window)
        yz_val = yz[i] if not np.isnan(yz[i]) else 0.02

        # 8. ATR for trailing stop
        atr = _atr_kernel(h, l, c, self.atr_period)
        atr_val = atr[i] if not np.isnan(atr[i]) else 0.0

        return composite, yz_val, atr_val

    def _inverse_vol_size(self, price: float, yz_vol: float) -> int:
        """Inverse-volatility position sizing (risk parity principle).

        target_dollar = portfolio * max_risk_pct / annualized_vol
        Capped at 95% of portfolio to prevent excessive leverage.
        """
        ann_vol = yz_vol * np.sqrt(252)
        if ann_vol < 0.01:
            ann_vol = 0.01
        target_dollar = self.portfolio_value * self.max_risk_pct / ann_vol
        target_dollar = min(target_dollar, self.portfolio_value * 0.95)
        shares = int(target_dollar / price)
        return max(0, shares)

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

        score, yz_vol, atr_val = self._compute_signal(open_, high, low, close, volume, i)

        if np.isnan(yz_vol) or np.isnan(atr_val):
            return {"action": "hold"}

        # Trailing stop check
        if holdings > 0 and self._trailing_stop.get(symbol) is not None:
            if price <= self._trailing_stop[symbol]:
                self._trailing_stop[symbol] = None
                return {"action": "sell", "symbol": symbol, "shares": holdings}
            new_stop = price - self.atr_stop_mult * atr_val
            if new_stop > self._trailing_stop.get(symbol, 0.0):
                self._trailing_stop[symbol] = new_stop

        # Entry
        if holdings == 0 and score > self.entry_threshold:
            shares = self._inverse_vol_size(price, yz_vol)
            if shares > 0 and self.can_buy(symbol, price, shares):
                self._trailing_stop[symbol] = price - self.atr_stop_mult * atr_val
                return {"action": "buy", "symbol": symbol, "shares": shares}

        # Signal-based exit
        if holdings > 0 and score < -self.exit_threshold:
            self._trailing_stop[symbol] = None
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
