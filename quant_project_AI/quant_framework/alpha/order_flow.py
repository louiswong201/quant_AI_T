"""
Order flow / microstructure features — Numba-compiled.

These features capture information that traditional OHLCV indicators miss:
the *asymmetry* between buying and selling pressure within each bar.

Theoretical backing:
  - OFI (Order Flow Imbalance): Measures the directional pressure from
    quote updates. Empirically shown to predict short-term returns
    (Cont, Kukanov & Stoikov, 2014). When OFI is positive, there is
    net buying pressure; when negative, net selling pressure.
  - VPIN (Volume-Synchronized PIN): Estimates the probability of informed
    trading using volume buckets. High VPIN → likely informed flow →
    wider spreads → potential volatility spike. (Easley, Lopez de Prado
    & O'Hara, 2012).
  - Trade Imbalance: Simple ratio of (buy_vol - sell_vol) / total_vol,
    approximated from price direction when tick-level data is unavailable.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):  # type: ignore[no-untyped-def]
            return func
        return decorator


@njit(cache=True, fastmath=True)
def _ofi_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """Order Flow Imbalance using bar-level proxy.

    Without tick data, we approximate OFI as:
      bar_ofi = volume * (2*close - high - low) / (high - low + 1e-10)
    Then smooth with a rolling sum over `window` bars.

    This captures the intuition: if close is near the high, buyers dominated;
    if near the low, sellers dominated. Volume amplifies the signal.
    """
    n = len(close)
    raw = np.empty(n, dtype=np.float64)
    for i in range(n):
        hl_range = high[i] - low[i]
        if hl_range < 1e-10:
            raw[i] = 0.0
        else:
            raw[i] = volume[i] * (2.0 * close[i] - high[i] - low[i]) / hl_range

    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    s = 0.0
    for i in range(window):
        s += raw[i]
    out[window - 1] = s
    for i in range(window, n):
        s += raw[i] - raw[i - window]
        out[i] = s
    return out


@njit(cache=True, fastmath=True)
def _vpin_numba(
    close: np.ndarray,
    volume: np.ndarray,
    n_buckets: int,
) -> np.ndarray:
    """Volume-Synchronized Probability of Informed Trading (VPIN).

    Simplified bar-level approximation:
      1. Classify each bar's volume as buy/sell using close vs previous close.
      2. Accumulate into volume buckets of equal size.
      3. VPIN = abs(buy_vol - sell_vol) / total_vol per bucket.

    The result is assigned to the last bar of each bucket.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2 or n_buckets <= 0:
        return out

    total_vol = 0.0
    for i in range(n):
        total_vol += volume[i]
    bucket_size = total_vol / n_buckets if n_buckets > 0 else total_vol

    if bucket_size <= 0:
        return out

    buy_vol = 0.0
    sell_vol = 0.0
    bucket_vol = 0.0

    for i in range(1, n):
        v = volume[i]
        if close[i] >= close[i - 1]:
            buy_vol += v
        else:
            sell_vol += v
        bucket_vol += v

        if bucket_vol >= bucket_size:
            total = buy_vol + sell_vol
            if total > 0:
                out[i] = abs(buy_vol - sell_vol) / total
            else:
                out[i] = 0.0
            buy_vol = 0.0
            sell_vol = 0.0
            bucket_vol = 0.0

    return out


@njit(cache=True, fastmath=True)
def _trade_imbalance_numba(
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """Rolling trade imbalance: (buy_vol - sell_vol) / total_vol.

    Buy/sell classification uses the tick rule:
      close[i] > close[i-1] → buy; close[i] < close[i-1] → sell.
    """
    n = len(close)
    signed = np.empty(n, dtype=np.float64)
    signed[0] = 0.0
    for i in range(1, n):
        if close[i] > close[i - 1]:
            signed[i] = volume[i]
        elif close[i] < close[i - 1]:
            signed[i] = -volume[i]
        else:
            signed[i] = 0.0

    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out

    s_signed = 0.0
    s_total = 0.0
    for i in range(window):
        s_signed += signed[i]
        s_total += volume[i]
    if s_total > 0:
        out[window - 1] = s_signed / s_total
    for i in range(window, n):
        s_signed += signed[i] - signed[i - window]
        s_total += volume[i] - volume[i - window]
        if s_total > 0:
            out[i] = s_signed / s_total
        else:
            out[i] = 0.0
    return out


class OrderFlowFeatures:
    """Order flow / microstructure feature generator."""

    @staticmethod
    def ofi(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Order Flow Imbalance — rolling sum of bar-level directional volume."""
        return _ofi_numba(
            np.ascontiguousarray(high, dtype=np.float64),
            np.ascontiguousarray(low, dtype=np.float64),
            np.ascontiguousarray(close, dtype=np.float64),
            np.ascontiguousarray(volume, dtype=np.float64),
            window,
        )

    @staticmethod
    def vpin(
        close: np.ndarray,
        volume: np.ndarray,
        n_buckets: int = 50,
    ) -> np.ndarray:
        """VPIN — probability of informed trading (volume-bucketed)."""
        return _vpin_numba(
            np.ascontiguousarray(close, dtype=np.float64),
            np.ascontiguousarray(volume, dtype=np.float64),
            n_buckets,
        )

    @staticmethod
    def trade_imbalance(
        close: np.ndarray,
        volume: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Rolling trade imbalance — normalized buy/sell volume difference."""
        return _trade_imbalance_numba(
            np.ascontiguousarray(close, dtype=np.float64),
            np.ascontiguousarray(volume, dtype=np.float64),
            window,
        )
