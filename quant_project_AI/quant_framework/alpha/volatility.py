"""
Volatility surface features — Numba-compiled.

Captures volatility dynamics that simple ATR/Bollinger bands miss.

Theoretical backing:
  - Realised Volatility (close-to-close): Standard estimator but can be
    biased by discrete sampling. We use Yang-Zhang estimator when OHLC
    is available for a more efficient estimate (Yang & Zhang, 2000).
  - Volatility of Volatility (vol-of-vol): Captures the "convexity" of
    the vol process. High vol-of-vol → fat-tailed return distributions
    → increased tail risk. Used in options pricing (Heston model) and
    regime detection.
  - Volatility Ratio: Current vol / longer-term vol. Values > 1 indicate
    a vol expansion regime (breakout conditions); values < 1 indicate
    mean-reversion regime (range-bound conditions).
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
def _yang_zhang_vol_numba(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """Yang-Zhang volatility estimator — most efficient OHLC estimator.

    sigma^2 = sigma_overnight^2 + k * sigma_open^2 + (1-k) * sigma_rs^2
    where sigma_rs is Rogers-Satchell and k is chosen for minimum variance.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window + 1:
        return out

    for i in range(window, n):
        s_oc = 0.0
        s_oc2 = 0.0
        s_co = 0.0
        s_co2 = 0.0
        s_rs = 0.0

        for j in range(i - window + 1, i + 1):
            if j == 0:
                continue
            log_co = np.log(open_[j] / close[j - 1]) if close[j - 1] > 0 else 0.0
            log_oc = np.log(close[j] / open_[j]) if open_[j] > 0 else 0.0
            log_ho = np.log(high[j] / open_[j]) if open_[j] > 0 else 0.0
            log_lo = np.log(low[j] / open_[j]) if open_[j] > 0 else 0.0
            log_hc = np.log(high[j] / close[j]) if close[j] > 0 else 0.0
            log_lc = np.log(low[j] / close[j]) if close[j] > 0 else 0.0

            s_co += log_co
            s_co2 += log_co * log_co
            s_oc += log_oc
            s_oc2 += log_oc * log_oc
            s_rs += log_ho * log_hc + log_lo * log_lc

        w = float(window)
        var_co = s_co2 / w - (s_co / w) ** 2
        var_oc = s_oc2 / w - (s_oc / w) ** 2
        var_rs = s_rs / w

        k = 0.34 / (1.34 + (w + 1.0) / (w - 1.0))
        sigma2 = var_co + k * var_oc + (1.0 - k) * var_rs
        out[i] = np.sqrt(max(0.0, sigma2))

    return out


@njit(cache=True, fastmath=True)
def _vol_of_vol_numba(
    close: np.ndarray,
    vol_window: int,
    vov_window: int,
) -> np.ndarray:
    """Volatility of volatility: rolling std of rolling vol."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < vol_window + vov_window:
        return out

    vol = np.full(n, np.nan, dtype=np.float64)
    ret = np.empty(n, dtype=np.float64)
    ret[0] = 0.0
    for i in range(1, n):
        if close[i - 1] > 0:
            ret[i] = np.log(close[i] / close[i - 1])
        else:
            ret[i] = 0.0

    for i in range(vol_window - 1, n):
        s = 0.0
        s2 = 0.0
        for j in range(i - vol_window + 1, i + 1):
            s += ret[j]
            s2 += ret[j] * ret[j]
        mean = s / vol_window
        var = s2 / vol_window - mean * mean
        vol[i] = np.sqrt(max(0.0, var))

    total_lookback = vol_window + vov_window - 1
    for i in range(total_lookback - 1, n):
        s = 0.0
        s2 = 0.0
        count = 0
        for j in range(i - vov_window + 1, i + 1):
            if not np.isnan(vol[j]):
                s += vol[j]
                s2 += vol[j] * vol[j]
                count += 1
        if count >= 2:
            mean = s / count
            var = s2 / count - mean * mean
            out[i] = np.sqrt(max(0.0, var))
    return out


@njit(cache=True, fastmath=True)
def _vol_ratio_numba(
    close: np.ndarray,
    fast_window: int,
    slow_window: int,
) -> np.ndarray:
    """Volatility ratio: fast_vol / slow_vol.

    >1 → vol expansion (breakout regime)
    <1 → vol contraction (mean-reversion regime)
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < slow_window + 1:
        return out

    ret = np.empty(n, dtype=np.float64)
    ret[0] = 0.0
    for i in range(1, n):
        if close[i - 1] > 0:
            ret[i] = close[i] / close[i - 1] - 1.0
        else:
            ret[i] = 0.0

    for i in range(slow_window - 1, n):
        sf = 0.0
        sf2 = 0.0
        ss = 0.0
        ss2 = 0.0
        for j in range(i - fast_window + 1, i + 1):
            sf += ret[j]
            sf2 += ret[j] * ret[j]
        for j in range(i - slow_window + 1, i + 1):
            ss += ret[j]
            ss2 += ret[j] * ret[j]
        fast_mean = sf / fast_window
        fast_var = sf2 / fast_window - fast_mean * fast_mean
        slow_mean = ss / slow_window
        slow_var = ss2 / slow_window - slow_mean * slow_mean

        fast_vol = np.sqrt(max(0.0, fast_var))
        slow_vol = np.sqrt(max(0.0, slow_var))
        if slow_vol > 1e-20:
            out[i] = fast_vol / slow_vol
    return out


class VolatilityFeatures:
    """Volatility surface feature generator."""

    @staticmethod
    def yang_zhang_vol(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Yang-Zhang estimator — most efficient OHLC volatility measure."""
        return _yang_zhang_vol_numba(
            np.ascontiguousarray(open_, dtype=np.float64),
            np.ascontiguousarray(high, dtype=np.float64),
            np.ascontiguousarray(low, dtype=np.float64),
            np.ascontiguousarray(close, dtype=np.float64),
            window,
        )

    @staticmethod
    def vol_of_vol(
        close: np.ndarray,
        vol_window: int = 20,
        vov_window: int = 20,
    ) -> np.ndarray:
        """Volatility of volatility — measures vol process convexity."""
        return _vol_of_vol_numba(
            np.ascontiguousarray(close, dtype=np.float64),
            vol_window,
            vov_window,
        )

    @staticmethod
    def vol_ratio(
        close: np.ndarray,
        fast_window: int = 10,
        slow_window: int = 60,
    ) -> np.ndarray:
        """Volatility ratio — fast/slow vol for regime detection."""
        return _vol_ratio_numba(
            np.ascontiguousarray(close, dtype=np.float64),
            fast_window,
            slow_window,
        )
