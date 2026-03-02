"""
Cross-asset features — Numba-compiled.

Captures inter-market dynamics that single-asset indicators miss.
These features exploit the empirical observation that assets are interconnected
through common risk factors, arbitrage relationships, and lead-lag effects.

Theoretical backing:
  - Rolling Beta: Measures the systematic risk exposure of an asset relative
    to a benchmark. Time-varying beta captures regime shifts (bear market beta
    expansion, crisis contagion). (Ang & Chen, 2007)
  - Cross-Correlation: Rolling Pearson correlation between two return series.
    Correlation breakdowns or spikes signal regime changes and contagion risk.
  - Lead-Lag Ratio: Measures which asset "leads" the other by comparing the
    cross-correlation at lag +1 vs lag -1. A ratio > 1 means asset A leads B.
    (Lo & MacKinlay, 1990)
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
def _rolling_beta_numba(
    returns_asset: np.ndarray,
    returns_benchmark: np.ndarray,
    window: int,
) -> np.ndarray:
    """Rolling OLS beta: cov(asset, bench) / var(bench)."""
    n = len(returns_asset)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out

    for i in range(window - 1, n):
        sx = 0.0
        sy = 0.0
        sxy = 0.0
        sxx = 0.0
        for j in range(i - window + 1, i + 1):
            x = returns_benchmark[j]
            y = returns_asset[j]
            sx += x
            sy += y
            sxy += x * y
            sxx += x * x
        mean_x = sx / window
        var_x = sxx / window - mean_x * mean_x
        if var_x > 1e-20:
            cov_xy = sxy / window - mean_x * (sy / window)
            out[i] = cov_xy / var_x
    return out


@njit(cache=True, fastmath=True)
def _rolling_correlation_numba(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    window: int,
) -> np.ndarray:
    """Rolling Pearson correlation."""
    n = len(returns_a)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out

    for i in range(window - 1, n):
        sx = 0.0
        sy = 0.0
        sxy = 0.0
        sxx = 0.0
        syy = 0.0
        for j in range(i - window + 1, i + 1):
            x = returns_a[j]
            y = returns_b[j]
            sx += x
            sy += y
            sxy += x * y
            sxx += x * x
            syy += y * y
        mx = sx / window
        my = sy / window
        cov = sxy / window - mx * my
        vx = sxx / window - mx * mx
        vy = syy / window - my * my
        denom = np.sqrt(vx * vy)
        if denom > 1e-20:
            out[i] = cov / denom
    return out


@njit(cache=True, fastmath=True)
def _lead_lag_ratio_numba(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    window: int,
) -> np.ndarray:
    """Lead-lag ratio: corr(A[t], B[t+1]) / corr(A[t+1], B[t]).

    Values > 1 indicate A leads B; values < 1 indicate B leads A.
    """
    n = len(returns_a)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window + 1:
        return out

    for i in range(window, n):
        s_ab_lead = 0.0
        s_ba_lead = 0.0
        s_aa = 0.0
        s_bb = 0.0
        count = 0
        for j in range(i - window, i):
            if j + 1 < n:
                a_lead = returns_a[j] * returns_b[j + 1]
                b_lead = returns_b[j] * returns_a[j + 1]
                s_ab_lead += a_lead
                s_ba_lead += b_lead
                s_aa += returns_a[j] * returns_a[j]
                s_bb += returns_b[j] * returns_b[j]
                count += 1
        if count > 0 and abs(s_ba_lead) > 1e-20:
            out[i] = abs(s_ab_lead / s_ba_lead)
    return out


def _to_returns(close: np.ndarray) -> np.ndarray:
    """Convert price series to simple returns."""
    ret = np.empty(len(close), dtype=np.float64)
    ret[0] = 0.0
    ret[1:] = np.diff(close) / np.where(close[:-1] > 0, close[:-1], 1.0)
    return ret


class CrossAssetFeatures:
    """Cross-asset feature generator for multi-market analysis."""

    @staticmethod
    def rolling_beta(
        close_asset: np.ndarray,
        close_benchmark: np.ndarray,
        window: int = 60,
    ) -> np.ndarray:
        """Rolling beta of asset vs benchmark (e.g., BTC vs ETH)."""
        ret_a = _to_returns(np.ascontiguousarray(close_asset, dtype=np.float64))
        ret_b = _to_returns(np.ascontiguousarray(close_benchmark, dtype=np.float64))
        return _rolling_beta_numba(ret_a, ret_b, window)

    @staticmethod
    def rolling_correlation(
        close_a: np.ndarray,
        close_b: np.ndarray,
        window: int = 30,
    ) -> np.ndarray:
        """Rolling Pearson correlation between two assets' returns."""
        ret_a = _to_returns(np.ascontiguousarray(close_a, dtype=np.float64))
        ret_b = _to_returns(np.ascontiguousarray(close_b, dtype=np.float64))
        return _rolling_correlation_numba(ret_a, ret_b, window)

    @staticmethod
    def lead_lag_ratio(
        close_a: np.ndarray,
        close_b: np.ndarray,
        window: int = 30,
    ) -> np.ndarray:
        """Lead-lag ratio between two assets. >1 means A leads B."""
        ret_a = _to_returns(np.ascontiguousarray(close_a, dtype=np.float64))
        ret_b = _to_returns(np.ascontiguousarray(close_b, dtype=np.float64))
        return _lead_lag_ratio_numba(ret_a, ret_b, window)
