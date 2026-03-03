"""
Unified Numba kernel backtest system — thread-parallel, equity-aware.

All 18 long/short leveraged strategy kernels live here as first-class
framework components.  Each kernel runs signals + fills + PnL in a single
compiled loop with fastmath + Numba-compiled scan.  Thread-parallel
scanning via ThreadPoolExecutor (Numba releases GIL).

Public API
----------
KERNEL_REGISTRY : dict[str, Callable]
    Maps strategy name -> Numba kernel function.
run_kernel(name, params, c, o, h, l, config) -> KernelResult
    Run a single kernel with BacktestConfig translation.
run_kernel_detailed(name, params, c, o, h, l, config) -> DetailedKernelResult
    Run a kernel and return rich results with equity curve and risk metrics.
scan_all_kernels(c, o, h, l, config, *, n_threads=...) -> dict[str, dict]
    Thread-parallel 18-strategy parameter grid scan.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange

from .config import BacktestConfig

# Keep FMA/reassociation/reciprocal but preserve NaN/Inf safety.
# Excludes 'nnan' and 'ninf' which cause prange deadlocks on Windows
# when division-by-zero produces hardware FP exceptions under MSVC.
_SAFE_FASTMATH = {'nsz', 'arcp', 'contract', 'afn', 'reassoc'}

def validate_ohlc(c, o, h, l):
    """Validate OHLC arrays have no zeros, negatives, or NaN values."""
    for name, arr in [("close", c), ("open", o), ("high", h), ("low", l)]:
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN values — clean data before backtesting")
        if np.any(arr <= 0):
            raise ValueError(f"{name} contains zero/negative values — clean data before backtesting")


# =====================================================================
#  Indicator helpers (Numba-compiled)
# =====================================================================

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _ema(arr, span):
    n = len(arr); out = np.empty(n, dtype=np.float64)
    k = 2.0 / (span + 1.0); out[0] = arr[0]
    for i in range(1, n):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _rolling_mean(arr, w):
    n = len(arr); out = np.full(n, np.nan); s = 0.0
    for i in range(n):
        s += arr[i]
        if i >= w:
            s -= arr[i - w]
        if i >= w - 1:
            out[i] = s / w
    return out


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _rolling_std(arr, w):
    n = len(arr); out = np.full(n, np.nan); s = 0.0; s2 = 0.0
    for i in range(n):
        s += arr[i]; s2 += arr[i] * arr[i]
        if i >= w:
            s -= arr[i - w]; s2 -= arr[i - w] * arr[i - w]
        if i >= w - 1:
            m = s / w; out[i] = np.sqrt(max(0.0, s2 / w - m * m))
    return out


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _atr(high, low, close, period):
    n = len(close); tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     max(abs(high[i] - close[i - 1]),
                         abs(low[i] - close[i - 1])))
    out = np.full(n, np.nan); s = 0.0
    for i in range(period):
        s += tr[i]
    out[period - 1] = s / period
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _rsi_wilder(close, period):
    n = len(close); out = np.full(n, np.nan)
    if n < period + 1:
        return out
    gs = 0.0; ls = 0.0
    for i in range(1, period + 1):
        d = close[i] - close[i - 1]
        if d > 0:
            gs += d
        else:
            ls -= d
    ag = gs / period; al = ls / period
    out[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al)
    for i in range(period + 1, n):
        d = close[i] - close[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        ag = (ag * (period - 1) + g) / period
        al = (al * (period - 1) + l) / period
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al)
    return out


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _score(ret, dd, nt):
    return ret / max(1.0, dd) * min(1.0, nt / 20.0)


# =====================================================================
#  Precomputation — all Numba-compiled for maximum speed
# =====================================================================

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def precompute_all_ma(close, max_w=200):
    n = len(close)
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    for i in range(n):
        cs[i + 1] = cs[i] + close[i]
    mas = np.full((max_w + 1, n), np.nan, dtype=np.float64)
    for w in range(2, min(max_w + 1, n + 1)):
        inv_w = 1.0 / w
        for i in range(w - 1, n):
            mas[w, i] = (cs[i + 1] - cs[i - w + 1]) * inv_w
    return mas


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def precompute_all_ema(close, max_s):
    n = len(close)
    emas = np.full((max_s + 1, n), np.nan, dtype=np.float64)
    for s in range(2, max_s + 1):
        k = 2.0 / (s + 1.0)
        emas[s, 0] = close[0]
        for i in range(1, n):
            emas[s, i] = close[i] * k + emas[s, i - 1] * (1.0 - k)
    return emas


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def precompute_all_rsi(close, max_p):
    n = len(close)
    rsi = np.full((max_p + 1, n), np.nan, dtype=np.float64)
    for p in range(2, max_p + 1):
        if n <= p:
            continue
        gs = 0.0; ls = 0.0
        for i in range(1, p + 1):
            d = close[i] - close[i - 1]
            if d > 0:
                gs += d
            else:
                ls -= d
        ag = gs / p; al = ls / p
        rsi[p, p] = 100.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al)
        for i in range(p + 1, n):
            d = close[i] - close[i - 1]
            g = d if d > 0 else 0.0
            l_ = -d if d < 0 else 0.0
            ag = (ag * (p - 1) + g) / p
            al = (al * (p - 1) + l_) / p
            rsi[p, i] = 100.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al)
    return rsi


# =====================================================================
#  Additional precomputation for O(n*p)->O(n) kernel optimization
# =====================================================================

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _rolling_max_1d(arr, w):
    """O(n) rolling max for a single window using block decomposition."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if w < 1 or w > n:
        return out
    prefix = np.empty(n, dtype=np.float64)
    suffix = np.empty(n, dtype=np.float64)
    for bi in range(0, n, w):
        be = min(bi + w, n)
        suffix[bi] = arr[bi]
        for j in range(bi + 1, be):
            suffix[j] = arr[j] if arr[j] > suffix[j - 1] else suffix[j - 1]
        prefix[be - 1] = arr[be - 1]
        for j in range(be - 2, bi - 1, -1):
            prefix[j] = arr[j] if arr[j] > prefix[j + 1] else prefix[j + 1]
    for i in range(w - 1, n):
        p = prefix[i - w + 1]
        s = suffix[i]
        out[i] = p if p > s else s
    return out


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def precompute_rolling_max(arr, max_w):
    """Precompute rolling max for all windows [2..max_w]. O(n) per window."""
    n = len(arr)
    out = np.full((max_w + 1, n), np.nan, dtype=np.float64)
    for w in range(2, max_w + 1):
        rm = _rolling_max_1d(arr, w)
        for i in range(n):
            out[w, i] = rm[i]
    return out


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _rolling_min_1d(arr, w):
    """O(n) rolling min for a single window using block decomposition."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if w < 1 or w > n:
        return out
    prefix = np.empty(n, dtype=np.float64)
    suffix = np.empty(n, dtype=np.float64)
    for bi in range(0, n, w):
        be = min(bi + w, n)
        suffix[bi] = arr[bi]
        for j in range(bi + 1, be):
            suffix[j] = arr[j] if arr[j] < suffix[j - 1] else suffix[j - 1]
        prefix[be - 1] = arr[be - 1]
        for j in range(be - 2, bi - 1, -1):
            prefix[j] = arr[j] if arr[j] < prefix[j + 1] else prefix[j + 1]
    for i in range(w - 1, n):
        p = prefix[i - w + 1]
        s = suffix[i]
        out[i] = p if p < s else s
    return out


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def precompute_rolling_min(arr, max_w):
    """Precompute rolling min for all windows [2..max_w]. O(n) per window."""
    n = len(arr)
    out = np.full((max_w + 1, n), np.nan, dtype=np.float64)
    for w in range(2, max_w + 1):
        rm = _rolling_min_1d(arr, w)
        for i in range(n):
            out[w, i] = rm[i]
    return out


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def precompute_up_prefix(close):
    """Prefix-sum of up-bars for O(1) drift ratio lookup."""
    n = len(close)
    psum = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n):
        psum[i + 1] = psum[i] + (1 if close[i] > close[i - 1] else 0)
    return psum


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def precompute_rolling_vol(close, max_vp):
    """Precompute rolling volatility for all vol_p in [2..max_vp]."""
    n = len(close)
    rets = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        rets[i] = (close[i] / close[i - 1] - 1.0) if close[i - 1] > 0 else 0.0
    vols = np.full((max_vp + 1, n), np.nan, dtype=np.float64)
    for vp in range(2, max_vp + 1):
        s = 0.0; s2 = 0.0
        for i in range(vp):
            s += rets[i]; s2 += rets[i] * rets[i]
        if vp > 0:
            m = s / vp
            vols[vp, vp - 1] = np.sqrt(max(1e-20, s2 / vp - m * m))
        for i in range(vp, n):
            s += rets[i] - rets[i - vp]
            s2 += rets[i] * rets[i] - rets[i - vp] * rets[i - vp]
            m = s / vp
            vols[vp, i] = np.sqrt(max(1e-20, s2 / vp - m * m))
    return vols


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def precompute_all_rolling_std(close, max_w):
    """Precompute rolling std for windows [2..max_w]. Reusable by Bollinger/ZScore."""
    n = len(close)
    stds = np.full((max_w + 1, n), np.nan, dtype=np.float64)
    for w in range(2, max_w + 1):
        s = 0.0; s2 = 0.0
        for i in range(n):
            s += close[i]; s2 += close[i] * close[i]
            if i >= w:
                s -= close[i - w]; s2 -= close[i - w] * close[i - w]
            if i >= w - 1:
                m = s / w
                stds[w, i] = np.sqrt(max(0.0, s2 / w - m * m))
    return stds


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _compute_mama_fama(close):
    """Precompute MAMA/FAMA arrays for MESA strategy."""
    n = len(close)
    mama = np.zeros(n, dtype=np.float64)
    fama = np.zeros(n, dtype=np.float64)
    if n < 40:
        return mama, fama
    smooth = np.zeros(n); det = np.zeros(n); I1 = np.zeros(n); Q1 = np.zeros(n)
    jI = np.zeros(n); jQ = np.zeros(n); I2 = np.zeros(n); Q2 = np.zeros(n)
    Re_ = np.zeros(n); Im_ = np.zeros(n); per = np.zeros(n); sp_ = np.zeros(n)
    ph = np.zeros(n)
    for i in range(min(6, n)):
        mama[i] = close[i]; fama[i] = close[i]; per[i] = 6.0; sp_[i] = 6.0
    for i in range(6, n):
        smooth[i] = (4*close[i]+3*close[i-1]+2*close[i-2]+close[i-3])/10.0
        adj = 0.075*per[i-1]+0.54
        det[i] = (0.0962*smooth[i]+0.5769*smooth[max(0,i-2)]-0.5769*smooth[max(0,i-4)]-0.0962*smooth[max(0,i-6)])*adj
        I1[i] = det[max(0,i-3)]
        Q1[i] = (0.0962*det[i]+0.5769*det[max(0,i-2)]-0.5769*det[max(0,i-4)]-0.0962*det[max(0,i-6)])*adj
        jI[i] = (0.0962*I1[i]+0.5769*I1[max(0,i-2)]-0.5769*I1[max(0,i-4)]-0.0962*I1[max(0,i-6)])*adj
        jQ[i] = (0.0962*Q1[i]+0.5769*Q1[max(0,i-2)]-0.5769*Q1[max(0,i-4)]-0.0962*Q1[max(0,i-6)])*adj
        I2[i] = I1[i]-jQ[i]; Q2[i] = Q1[i]+jI[i]
        I2[i] = 0.2*I2[i]+0.8*I2[i-1]; Q2[i] = 0.2*Q2[i]+0.8*Q2[i-1]
        Re_[i] = I2[i]*I2[i-1]+Q2[i]*Q2[i-1]; Im_[i] = I2[i]*Q2[i-1]-Q2[i]*I2[i-1]
        Re_[i] = 0.2*Re_[i]+0.8*Re_[i-1]; Im_[i] = 0.2*Im_[i]+0.8*Im_[i-1]
        per[i] = (2*np.pi/np.arctan(Im_[i]/Re_[i])) if (Im_[i]!=0 and Re_[i]!=0) else per[i-1]
        if per[i] > 1.5*per[i-1]: per[i] = 1.5*per[i-1]
        if per[i] < 0.67*per[i-1]: per[i] = 0.67*per[i-1]
        if per[i] < 6.0: per[i] = 6.0
        if per[i] > 50.0: per[i] = 50.0
        per[i] = 0.2*per[i]+0.8*per[i-1]; sp_[i] = 0.33*per[i]+0.67*sp_[i-1]
        ph[i] = (np.arctan(Q1[i]/I1[i])*180.0/np.pi) if I1[i]!=0 else ph[i-1]
        dp = ph[i-1]-ph[i]
        if dp < 1.0: dp = 1.0
        alpha = 0.5/dp  # fl=0.5 placeholder, actual alpha set per combo
        if alpha < 0.05: alpha = 0.05
        if alpha > 0.5: alpha = 0.5
        mama[i] = alpha*close[i]+(1-alpha)*mama[i-1]
        fama[i] = 0.5*alpha*mama[i]+(1-0.5*alpha)*fama[i-1]
    return mama, fama


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _compute_kama(close, er_p, fast_sc, slow_sc):
    """Precompute KAMA array for given parameters."""
    n = len(close)
    kama = np.full(n, np.nan, dtype=np.float64)
    if n < er_p + 2:
        return kama
    fc = 2.0/(fast_sc+1.0); sc_v = 2.0/(slow_sc+1.0)
    kama[er_p-1] = close[er_p-1]
    for i in range(er_p, n):
        d = abs(close[i]-close[i-er_p]); v = 0.0
        for j in range(1, er_p+1):
            v += abs(close[i-j+1]-close[i-j])
        er = d/v if v > 0 else 0.0; sc2 = (er*(fc-sc_v)+sc_v)**2
        kama[i] = kama[i-1]+sc2*(close[i]-kama[i-1])
    return kama


# =====================================================================
#  Optimized kernel variants that accept precomputed arrays
# =====================================================================

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_turtle_precomp(c, o, atr_arr, rmax_entry, rmin_entry, rmin_exit, rmax_exit,
                      entry_p, exit_p, atr_stop, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """Turtle with precomputed rolling max/min and ATR. O(n) per combo."""
    n = len(c)
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    start = max(entry_p, exit_p)
    for i in range(start, n):
        pos, ep, tr, tc, liq = _fx_lev(pend, pos, ep, o[i], tr, sb, ss, cm, lev, dc, pfrac); nt += tc; pend = 0
        if liq: pos = 0; ep = 0.0; continue
        pos, ep, tr, tc2 = _sl_exit(pos, ep, tr, c[i], sb, ss, cm, lev, sl, pfrac, sl_slip); nt += tc2
        eh = rmax_entry[i - 1]; el = rmin_entry[i - 1]
        xl = rmin_exit[i - 1]; xh = rmax_exit[i - 1]
        a = atr_arr[i]
        if a != a or eh != eh or el != el: pass
        else:
            if pos == 1 and sb > 0:
                if c[i] < ep / sb - atr_stop * a or c[i] < xl: pend = 2
            elif pos == -1 and ss > 0:
                if c[i] > ep / ss + atr_stop * a or c[i] > xh: pend = 2
            if pos == 0 and pend == 0:
                if c[i] > eh: pend = 1
                elif c[i] < el: pend = -1
        eq = _mtm_lev(pos, tr, c[i], ep, sb, ss, cm, lev, sl, pfrac)
        if eq > pk: pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd: mdd = dd_
    if pos == 1 and ep > 0:
        raw = (c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep = _deploy(tr, pfrac); tr += dep*((raw-1.0)*lev); tr = max(0.01, tr); nt += 1
    elif pos == -1 and ep > 0:
        raw = (ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep = _deploy(tr, pfrac); tr += dep*((raw-1.0)*lev); tr = max(0.01, tr); nt += 1
    return (tr - 1.0) * 100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_drift_precomp(c, o, up_prefix, lookback, drift_thr, hold_p, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """Drift with precomputed up-bar prefix sum. O(n) per combo."""
    n = len(c); pos = 0; ep = 0.0; tr = 1.0; pend = 0; hc = 0; pk = 1.0; mdd = 0.0; nt = 0
    for i in range(lookback, n):
        pos, ep, tr, tc, liq = _fx_lev(pend, pos, ep, o[i], tr, sb, ss, cm, lev, dc, pfrac); nt += tc; pend = 0
        if liq: pos = 0; ep = 0.0; hc = 0; continue
        pos, ep, tr, tc2 = _sl_exit(pos, ep, tr, c[i], sb, ss, cm, lev, sl, pfrac, sl_slip); nt += tc2
        if tc2 > 0: hc = 0
        up = up_prefix[i + 1] - up_prefix[i - lookback + 1]
        ratio = up / lookback if lookback > 0 else 0.5
        if pos != 0:
            hc += 1
            if hc >= hold_p: pend = 2; hc = 0
        if pos == 0 and pend == 0:
            if ratio >= drift_thr: pend = -1; hc = 0
            elif ratio <= 1.0 - drift_thr: pend = 1; hc = 0
        eq = _mtm_lev(pos, tr, c[i], ep, sb, ss, cm, lev, sl, pfrac)
        if eq > pk: pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd: mdd = dd_
    if pos == 1 and ep > 0:
        raw = (c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep = _deploy(tr, pfrac); tr += dep*((raw-1.0)*lev); tr = max(0.01, tr); nt += 1
    elif pos == -1 and ep > 0:
        raw = (ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep = _deploy(tr, pfrac); tr += dep*((raw-1.0)*lev); tr = max(0.01, tr); nt += 1
    return (tr - 1.0) * 100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_ramom_precomp(c, o, mom_p, vol_arr, ez, xz, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """RAMOM with precomputed rolling volatility. O(n) per combo."""
    n = len(c); pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    for i in range(mom_p, n):
        pos, ep, tr, tc, liq = _fx_lev(pend, pos, ep, o[i], tr, sb, ss, cm, lev, dc, pfrac); nt += tc; pend = 0
        if liq: pos = 0; ep = 0.0; continue
        pos, ep, tr, tc2 = _sl_exit(pos, ep, tr, c[i], sb, ss, cm, lev, sl, pfrac, sl_slip); nt += tc2
        prev = c[i - mom_p]
        if prev <= 0: continue
        mom = (c[i] / prev) - 1.0
        vol = vol_arr[i]
        if vol != vol or vol < 1e-20: continue
        z = mom / vol
        if pos == 0:
            if z > ez: pend = 1
            elif z < -ez: pend = -1
        elif pos == 1 and z < xz: pend = 2
        elif pos == -1 and z > -xz: pend = 2
        eq = _mtm_lev(pos, tr, c[i], ep, sb, ss, cm, lev, sl, pfrac)
        if eq > pk: pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd: mdd = dd_
    if pos == 1 and ep > 0:
        raw = (c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep = _deploy(tr, pfrac); tr += dep*((raw-1.0)*lev); tr = max(0.01, tr); nt += 1
    elif pos == -1 and ep > 0:
        raw = (ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep = _deploy(tr, pfrac); tr += dep*((raw-1.0)*lev); tr = max(0.01, tr); nt += 1
    return (tr - 1.0) * 100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_mombreak_precomp(c, o, atr_arr, rh, rl, hp, prox, atr_t, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """MomBreak with precomputed rolling H/L and ATR. O(n) per combo."""
    n = len(c)
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; ts_l = 0.0; ts_s = 1e18; pk = 1.0; mdd = 0.0; nt = 0
    start = hp
    for i in range(start, n):
        pos, ep, tr, tc, liq = _fx_lev(pend, pos, ep, o[i], tr, sb, ss, cm, lev, dc, pfrac); nt += tc
        if pend == 1 and atr_arr[i] == atr_arr[i]: ts_l = o[i] - atr_t * atr_arr[i]
        elif pend == -1 and atr_arr[i] == atr_arr[i]: ts_s = o[i] + atr_t * atr_arr[i]
        pend = 0
        if liq: pos = 0; ep = 0.0; continue
        pos, ep, tr, tc2 = _sl_exit(pos, ep, tr, c[i], sb, ss, cm, lev, sl, pfrac, sl_slip); nt += tc2
        hv = rh[i]; lv = rl[i]; a = atr_arr[i]
        if hv != hv or lv != lv or a != a: pass
        else:
            if pos == 1:
                ns = c[i] - atr_t * a
                if ns > ts_l: ts_l = ns
                if c[i] < ts_l: pend = 2
            elif pos == -1:
                ns = c[i] + atr_t * a
                if ns < ts_s: ts_s = ns
                if c[i] > ts_s: pend = 2
            if pos == 0 and pend == 0:
                if c[i] >= hv * (1.0 - prox): pend = 1
                elif c[i] <= lv * (1.0 + prox): pend = -1
        eq = _mtm_lev(pos, tr, c[i], ep, sb, ss, cm, lev, sl, pfrac)
        if eq > pk: pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd: mdd = dd_
    if pos == 1 and ep > 0:
        raw = (c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep = _deploy(tr, pfrac); tr += dep*((raw-1.0)*lev); tr = max(0.01, tr); nt += 1
    elif pos == -1 and ep > 0:
        raw = (ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep = _deploy(tr, pfrac); tr += dep*((raw-1.0)*lev); tr = max(0.01, tr); nt += 1
    return (tr - 1.0) * 100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_volregime_precomp(c, o, atr_arr, rsi_arr, ma_s_arr, ma_l_arr,
                         vol_thr, rsi_os, rsi_ob, start,
                         sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """VolRegime with precomputed ATR, RSI, and MAs. O(n) per combo."""
    n = len(c)
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; mode = 0; pk = 1.0; mdd = 0.0; nt = 0
    for i in range(start, n):
        pos, ep, tr, tc, liq = _fx_lev(pend, pos, ep, o[i], tr, sb, ss, cm, lev, dc, pfrac); nt += tc; pend = 0
        if liq: pos = 0; ep = 0.0; continue
        pos, ep, tr, tc2 = _sl_exit(pos, ep, tr, c[i], sb, ss, cm, lev, sl, pfrac, sl_slip); nt += tc2
        a = atr_arr[i]
        if a != a or c[i] <= 0: pass
        else:
            hv = a / c[i] > vol_thr
            if hv:
                r = rsi_arr[i]
                if r != r: pass
                else:
                    if pos == 0:
                        if r < rsi_os: pend = 1; mode = 1
                        elif r > rsi_ob: pend = -1; mode = 1
                    elif pos == 1 and mode == 1 and r > 50: pend = 2
                    elif pos == -1 and mode == 1 and r < 50: pend = 2
            else:
                s_ = ma_s_arr[i]; l_ = ma_l_arr[i]; s0 = ma_s_arr[i-1]; l0 = ma_l_arr[i-1]
                if s_ != s_ or l_ != l_ or s0 != s0 or l0 != l0: pass
                else:
                    if pos == 0:
                        if s0 <= l0 and s_ > l_: pend = 1; mode = 0
                        elif s0 >= l0 and s_ < l_: pend = -1; mode = 0
                    elif pos == 1 and mode == 0 and s_ < l_: pend = 2
                    elif pos == -1 and mode == 0 and s_ > l_: pend = 2
            if pos != 0 and pend == 0:
                if (mode == 0 and hv) or (mode == 1 and not hv): pend = 2
        eq = _mtm_lev(pos, tr, c[i], ep, sb, ss, cm, lev, sl, pfrac)
        if eq > pk: pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd: mdd = dd_
    if pos == 1 and ep > 0:
        raw = (c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep = _deploy(tr, pfrac); tr += dep*((raw-1.0)*lev); tr = max(0.01, tr); nt += 1
    elif pos == -1 and ep > 0:
        raw = (ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep = _deploy(tr, pfrac); tr += dep*((raw-1.0)*lev); tr = max(0.01, tr); nt += 1
    return (tr - 1.0) * 100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_multifactor_precomp(c, o, rsi_arr, mom_p, vol_arr, lt, st,
                           sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """MultiFactor with precomputed RSI and rolling vol. O(n) per combo."""
    n = len(c)
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    start = max(mom_p, 2)
    for i in range(start, n):
        pos, ep, tr, tc, liq = _fx_lev(pend, pos, ep, o[i], tr, sb, ss, cm, lev, dc, pfrac); nt += tc; pend = 0
        if liq: pos = 0; ep = 0.0; continue
        pos, ep, tr, tc2 = _sl_exit(pos, ep, tr, c[i], sb, ss, cm, lev, sl, pfrac, sl_slip); nt += tc2
        r = rsi_arr[i]
        if r != r: pass
        else:
            rs = (100.0 - r) / 100.0
            prev_mf = c[i - mom_p]
            mom = (c[i] / prev_mf - 1.0) if prev_mf > 0 else 0.0
            ms = max(-0.5, min(0.5, mom)) + 0.5
            v = vol_arr[i]
            vs = max(0.0, 1.0 - v * 20.0) if v == v else 0.5
            comp = (rs + ms + vs) / 3.0
            if pos == 0:
                if comp > lt: pend = 1
                elif comp < st: pend = -1
            elif pos == 1 and comp < 0.5: pend = 2
            elif pos == -1 and comp > 0.5: pend = 2
        eq = _mtm_lev(pos, tr, c[i], ep, sb, ss, cm, lev, sl, pfrac)
        if eq > pk: pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd: mdd = dd_
    if pos == 1 and ep > 0:
        raw = (c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep = _deploy(tr, pfrac); tr += dep*((raw-1.0)*lev); tr = max(0.01, tr); nt += 1
    elif pos == -1 and ep > 0:
        raw = (ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep = _deploy(tr, pfrac); tr += dep*((raw-1.0)*lev); tr = max(0.01, tr); nt += 1
    return (tr - 1.0) * 100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_macd_precomp(c, o, macd_line, sig_span, sb, ss, cm, lev, dc, stop=0.80, pfrac=1.0, sl_slip=0.0):
    """MACD with precomputed MACD line. Signal EMA fused into trading loop."""
    n = len(c)
    k = 2.0/(sig_span+1.0); k1 = 1.0 - k
    sp = macd_line[0]
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(1, n):
        pos,ep,tr,tc,liq = _fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0
        else:
            pos,ep,tr,tc2 = _sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,stop,pfrac,sl_slip); nt+=tc2
        mp = macd_line[i-1]
        mc = macd_line[i]
        sc = mc*k + sp*k1
        if mp!=mp or mc!=mc: pass
        elif mp<=sp and mc>sc:
            if pos==0: pend=1
            elif pos==-1: pend=3
        elif mp>=sp and mc<sc:
            if pos==0: pend=-1
            elif pos==1: pend=-3
        sp = sc
        eq = _mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,stop,pfrac)
        if eq>pk: pk=eq
        dd_ = (pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_bollinger_precomp(c, o, ma_arr, std_arr, num_std, period,
                         sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """Bollinger with precomputed rolling mean/std. O(n) per combo."""
    n = len(c)
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    for i in range(period, n):
        pos,ep,tr,tc,liq = _fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2 = _sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        m = ma_arr[i]; s = std_arr[i]
        if m!=m or s!=s or s<1e-10: pass
        else:
            u = m+num_std*s; lo = m-num_std*s
            if pos==0:
                if c[i]<lo: pend=1
                elif c[i]>u: pend=-1
            elif pos==1 and c[i]>=m: pend=2
            elif pos==-1 and c[i]<=m: pend=2
        eq = _mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_ = (pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_keltner_precomp(c, o, ema_arr, atr_arr, atr_m, ema_p, atr_p,
                       sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """Keltner with precomputed EMA/ATR. O(n) per combo."""
    n = len(c)
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    start = max(ema_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq = _fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2 = _sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        e = ema_arr[i]; a = atr_arr[i]
        if e!=e or a!=a: pass
        else:
            if pos==0:
                if c[i]>e+atr_m*a: pend=1
                elif c[i]<e-atr_m*a: pend=-1
            elif pos==1 and c[i]<e: pend=2
            elif pos==-1 and c[i]>e: pend=2
        eq = _mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_ = (pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_donchian_precomp(c, o, dh_arr, dl_arr, atr_arr, atr_m, entry_p, atr_p,
                        sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """Donchian with precomputed rolling H/L and ATR. O(n) per combo."""
    n = len(c)
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; ts = 0.0; pk = 1.0; mdd = 0.0; nt = 0
    start = max(entry_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq = _fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc
        if pend==1 and atr_arr[i]==atr_arr[i]: ts=o[i]-atr_m*atr_arr[i]
        elif pend==-1 and atr_arr[i]==atr_arr[i]: ts=o[i]+atr_m*atr_arr[i]
        pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2 = _sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        d1 = dh_arr[i-1]; d2 = dl_arr[i-1]; a = atr_arr[i]
        if d1!=d1 or d2!=d2 or a!=a: pass
        else:
            if pos==1:
                ns = c[i]-atr_m*a
                if ns>ts: ts=ns
                if c[i]<ts: pend=2
            elif pos==-1:
                ns = c[i]+atr_m*a
                if ns<ts: ts=ns
                if c[i]>ts: pend=2
            if pos==0 and pend==0:
                if c[i]>d1: pend=1
                elif c[i]<d2: pend=-1
        eq = _mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_ = (pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_zscore_precomp(c, o, rm_arr, rs_arr, lookback, ez, xz, sz,
                      sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """ZScore with precomputed rolling mean/std. O(n) per combo."""
    n = len(c)
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    for i in range(lookback, n):
        pos,ep,tr,tc,liq = _fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2 = _sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        m = rm_arr[i]; sd = rs_arr[i]
        if sd==0 or sd!=sd: pass
        else:
            z = (c[i]-m)/sd
            if pos==1 and (z>-xz or z>sz): pend=2
            elif pos==-1 and (z<xz or z<-sz): pend=2
            if pos==0 and pend==0:
                if z<-ez: pend=1
                elif z>ez: pend=-1
        eq = _mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_ = (pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_regime_ema_precomp(c, o, atr_arr, ef, es, et, vt, atr_p, se_p, te_p,
                          sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """RegimeEMA with precomputed ATR and EMAs. O(n) per combo."""
    n = len(c)
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    start = max(atr_p, max(se_p, te_p))
    for i in range(start, n):
        pos,ep,tr,tc,liq = _fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2 = _sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        a = atr_arr[i]
        if a!=a or c[i]<=0: pass
        else:
            hv = a/c[i] > vt
            if hv:
                if pos==0:
                    if ef[i]>es[i] and ef[i-1]<=es[i-1]: pend=1
                    elif ef[i]<es[i] and ef[i-1]>=es[i-1]: pend=-1
                elif pos==1 and ef[i]<es[i]: pend=2
                elif pos==-1 and ef[i]>es[i]: pend=2
            else:
                pd_ = (c[i]-et[i])/et[i] if abs(et[i]) > 1e-20 else 0.0
                if pos==0:
                    if pd_<-0.02: pend=1
                    elif pd_>0.02: pend=-1
                elif pos==1 and pd_>0.0: pend=2
                elif pos==-1 and pd_<0.0: pend=2
        eq = _mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_ = (pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_mesa_precomp(c, o, fl, slow_lim, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """MESA kernel — kept as-is since MAMA/FAMA depend on fl parameter."""
    n = len(c)
    if n<40: return 0.0, 0.0, 0
    smooth=np.zeros(n); det=np.zeros(n); I1=np.zeros(n); Q1=np.zeros(n)
    jI=np.zeros(n); jQ=np.zeros(n); I2=np.zeros(n); Q2=np.zeros(n)
    Re_=np.zeros(n); Im_=np.zeros(n); per=np.zeros(n); sp_=np.zeros(n)
    ph=np.zeros(n); mama=np.zeros(n); fama=np.zeros(n)
    for i in range(min(6,n)):
        mama[i]=c[i]; fama[i]=c[i]; per[i]=6.0; sp_[i]=6.0
    for i in range(6, n):
        smooth[i]=(4*c[i]+3*c[i-1]+2*c[i-2]+c[i-3])/10.0
        adj=0.075*per[i-1]+0.54
        det[i]=(0.0962*smooth[i]+0.5769*smooth[max(0,i-2)]-0.5769*smooth[max(0,i-4)]-0.0962*smooth[max(0,i-6)])*adj
        I1[i]=det[max(0,i-3)]
        Q1[i]=(0.0962*det[i]+0.5769*det[max(0,i-2)]-0.5769*det[max(0,i-4)]-0.0962*det[max(0,i-6)])*adj
        jI[i]=(0.0962*I1[i]+0.5769*I1[max(0,i-2)]-0.5769*I1[max(0,i-4)]-0.0962*I1[max(0,i-6)])*adj
        jQ[i]=(0.0962*Q1[i]+0.5769*Q1[max(0,i-2)]-0.5769*Q1[max(0,i-4)]-0.0962*Q1[max(0,i-6)])*adj
        I2[i]=I1[i]-jQ[i]; Q2[i]=Q1[i]+jI[i]
        I2[i]=0.2*I2[i]+0.8*I2[i-1]; Q2[i]=0.2*Q2[i]+0.8*Q2[i-1]
        Re_[i]=I2[i]*I2[i-1]+Q2[i]*Q2[i-1]; Im_[i]=I2[i]*Q2[i-1]-Q2[i]*I2[i-1]
        Re_[i]=0.2*Re_[i]+0.8*Re_[i-1]; Im_[i]=0.2*Im_[i]+0.8*Im_[i-1]
        per[i]=(2*np.pi/np.arctan(Im_[i]/Re_[i])) if (Im_[i]!=0 and Re_[i]!=0) else per[i-1]
        if per[i]>1.5*per[i-1]: per[i]=1.5*per[i-1]
        if per[i]<0.67*per[i-1]: per[i]=0.67*per[i-1]
        if per[i]<6.0: per[i]=6.0
        if per[i]>50.0: per[i]=50.0
        per[i]=0.2*per[i]+0.8*per[i-1]; sp_[i]=0.33*per[i]+0.67*sp_[i-1]
        ph[i]=(np.arctan(Q1[i]/I1[i])*180.0/np.pi) if I1[i]!=0 else ph[i-1]
        dp=ph[i-1]-ph[i]
        if dp<1.0: dp=1.0
        alpha=fl/dp
        if alpha<slow_lim: alpha=slow_lim
        if alpha>fl: alpha=fl
        mama[i]=alpha*c[i]+(1-alpha)*mama[i-1]
        fama[i]=0.5*alpha*mama[i]+(1-0.5*alpha)*fama[i-1]
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(7, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        if pos==0:
            if mama[i]>fama[i] and mama[i-1]<=fama[i-1]: pend=1
            elif mama[i]<fama[i] and mama[i-1]>=fama[i-1]: pend=-1
        elif pos==1 and mama[i]<fama[i] and mama[i-1]>=fama[i-1]: pend=-3
        elif pos==-1 and mama[i]>fama[i] and mama[i-1]<=fama[i-1]: pend=3
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_kama_precomp(c, o, kama_arr, atr_arr, atr_sm,
                    er_p, atr_p, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """KAMA with precomputed KAMA and ATR arrays. O(n) per combo."""
    n = len(c)
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    start = max(er_p+2, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq = _fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2 = _sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        k = kama_arr[i]; kp = kama_arr[i-1]; a = atr_arr[i]
        if k!=k or kp!=kp or a!=a: pass
        else:
            if pos==1:
                if c[i]<ep/sb-atr_sm*a or k<kp: pend=2
            elif pos==-1:
                if c[i]>ep/ss+atr_sm*a or k>kp: pend=2
            if pos==0 and pend==0:
                if c[i]>k and k>kp: pend=1
                elif c[i]<k and k<kp: pend=-1
        eq = _mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_ = (pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


# =====================================================================
#  Common epilogue helper — used by equity kernels
# =====================================================================

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _close_pos(pos, ep, tr, c_last, sb, ss, cm, lev, pfrac):
    """Close any open position at the last bar. Returns (tr, extra_trades)."""
    if ep <= 0:
        return tr, 0
    if pos == 1:
        raw = (c_last * ss * (1 - cm)) / (ep * (1 + cm))
        dep = _deploy(tr, pfrac); tr += dep * ((raw - 1.0) * lev)
        return max(0.01, tr), 1
    if pos == -1:
        raw = (ep * (1 - cm)) / (c_last * sb * (1 + cm))
        dep = _deploy(tr, pfrac); tr += dep * ((raw - 1.0) * lev)
        return max(0.01, tr), 1
    return tr, 0


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _equity_from_fused_positions(fused_pos, c, o, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    """Compute equity from a pre-computed fused position series.

    fused_pos[i] is the target position decided after bar i; execution
    at bar i+1's open — matching the kernel convention.
    """
    n = len(c)
    if n == 0:
        return 0.0, 0.0, 0, np.ones(0, dtype=np.float64), 0
    eq_arr = np.ones(n, dtype=np.float64)
    pos = 0; ep = 0.0; tr = 1.0; pk = 1.0; mdd = 0.0; nt = 0

    for i in range(1, n):
        target = fused_pos[i - 1]

        if target == pos:
            pend = 0
        elif pos == 0:
            pend = int(target)
        elif target == 0:
            pend = 2
        elif target == 1:
            pend = 3
        elif target == -1:
            pend = -3
        else:
            pend = 0

        pos, ep, tr, tc, liq = _fx_lev(pend, pos, ep, o[i], tr, sb, ss, cm, lev, dc, pfrac)
        nt += tc

        if liq:
            pos = 0; ep = 0.0; eq_arr[i] = tr; continue

        pos, ep, tr, tc2 = _sl_exit(pos, ep, tr, c[i], sb, ss, cm, lev, sl, pfrac, sl_slip)
        nt += tc2

        eq = _mtm_lev(pos, tr, c[i], ep, sb, ss, cm, lev, sl, pfrac)
        eq_arr[i] = eq

        if eq > pk:
            pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd:
            mdd = dd_

    fpos = pos
    tr, tc = _close_pos(pos, ep, tr, c[n - 1], sb, ss, cm, lev, pfrac)
    nt += tc
    eq_arr[n - 1] = tr

    return (tr - 1.0) * 100.0, mdd, nt, eq_arr, fpos


# =====================================================================
#  Core fill/exit/MTM helpers with leverage
# =====================================================================

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _deploy(tr, pfrac):
    d = tr * pfrac
    if tr > 1.0 and d > pfrac:
        d = pfrac
    return d


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _fx_lev(pend, pos, ep, oi, tr, sb, ss, cm, lev, daily_cost, pfrac):
    if pos != 0:
        deployed = _deploy(tr, pfrac)
        cost = deployed * daily_cost
        tr -= cost
        if tr < 0.01:
            return 0, 0.0, 0.01, 0, 1
    if pend == 0:
        return pos, ep, tr, 0, 0
    tc = 0; liq = 0
    if abs(pend) >= 2:
        deployed = _deploy(tr, pfrac)
        if pos == 1 and ep > 0:
            raw = (oi * ss * (1.0 - cm)) / (ep * (1.0 + cm))
            pnl = (raw - 1.0) * lev
            tr += deployed * pnl
            if tr < 0.01:
                tr = 0.01
            tc = 1
        elif pos == -1 and ep > 0 and oi > 0:
            raw = (ep * (1.0 - cm)) / (oi * sb * (1.0 + cm))
            pnl = (raw - 1.0) * lev
            tr += deployed * pnl
            if tr < 0.01:
                tr = 0.01
            tc = 1
        pos = 0
    if pend == 1 or pend == 3:
        ep = oi * sb; pos = 1
    elif pend == -1 or pend == -3:
        ep = oi * ss; pos = -1
    return pos, ep, tr, tc, liq


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _sl_exit(pos, ep, tr, ci, sb, ss, cm, lev, sl, pfrac, sl_slip):
    if pos == 0 or ep <= 0:
        return pos, ep, tr, 0
    if pos == 1:
        raw = (ci * ss * (1.0 - cm)) / (ep * (1.0 + cm))
        pnl = (raw - 1.0) * lev
    else:
        denom = ci * sb * (1.0 + cm)
        raw = (ep * (1.0 - cm)) / denom if denom > 0 else 1.0
        pnl = (raw - 1.0) * lev
    if pnl >= -sl:
        return pos, ep, tr, 0
    actual_loss = sl + sl_slip
    deployed = _deploy(tr, pfrac)
    tr -= deployed * actual_loss
    if tr < 0.01:
        tr = 0.01
    return 0, 0.0, tr, 1


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _mtm_lev(pos, tr, ci, ep, sb, ss, cm, lev, sl, pfrac):
    deployed = _deploy(tr, pfrac)
    if pos == 1 and ep > 0:
        raw = (ci * ss * (1 - cm)) / (ep * (1 + cm))
        pnl = (raw - 1.0) * lev
        pnl = max(pnl, -sl)
        return tr + deployed * pnl
    if pos == -1 and ep > 0 and ci > 0:
        raw = (ep * (1 - cm)) / (ci * sb * (1 + cm))
        pnl = (raw - 1.0) * lev
        pnl = max(pnl, -sl)
        return tr + deployed * pnl
    return tr


# =====================================================================
#  18 Long/Short Leveraged Strategy Kernels
# =====================================================================

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_ma_ls(c, o, ma_s, ma_l, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(1, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        s0=ma_s[i-1]; l0=ma_l[i-1]; s1=ma_s[i]; l1=ma_l[i]
        if s0!=s0 or l0!=l0 or s1!=s1 or l1!=l1: pass
        elif s0<=l0 and s1>l1:
            if pos==0: pend=1
            elif pos==-1: pend=3
        elif s0>=l0 and s1<l1:
            if pos==0: pend=-1
            elif pos==1: pend=-3
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_rsi_ls(c, o, rsi, os_v, ob_v, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(1, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        r=rsi[i]
        if r!=r: pass
        elif r<os_v:
            if pos==0: pend=1
            elif pos==-1: pend=3
        elif r>ob_v:
            if pos==0: pend=-1
            elif pos==1: pend=-3
        elif pos==1 and r>50: pend=2
        elif pos==-1 and r<50: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_macd_ls(c, o, ef, es, sig_span, sb, ss, cm, lev, dc, stop=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); ml=np.empty(n); sl=np.empty(n)
    for i in range(n): ml[i]=ef[i]-es[i]
    k=2.0/(sig_span+1.0); sl[0]=ml[0]
    for i in range(1,n): sl[i]=ml[i]*k+sl[i-1]*(1-k)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(1, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,stop,pfrac,sl_slip); nt+=tc2
        mp=ml[i-1]; sp=sl[i-1]; mc=ml[i]; sc=sl[i]
        if mp!=mp or sp!=sp or mc!=mc or sc!=sc: pass
        elif mp<=sp and mc>sc:
            if pos==0: pend=1
            elif pos==-1: pend=3
        elif mp>=sp and mc<sc:
            if pos==0: pend=-1
            elif pos==1: pend=-3
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,stop,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_drift_ls(c, o, lookback, drift_thr, hold_p, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); pos=0; ep=0.0; tr=1.0; pend=0; hc=0; pk=1.0; mdd=0.0; nt=0
    for i in range(lookback, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; hc=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        if tc2>0: hc=0
        up=0
        for j in range(1, lookback+1):
            if c[i-j+1]>c[i-j]: up+=1
        ratio=up/lookback
        if pos!=0:
            hc+=1
            if hc>=hold_p: pend=2; hc=0
        if pos==0 and pend==0:
            if ratio>=drift_thr: pend=-1; hc=0
            elif ratio<=1.0-drift_thr: pend=1; hc=0
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_ramom_ls(c, o, mom_p, vol_p, ez, xz, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(mom_p, vol_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        prev_r=c[i-mom_p]
        if prev_r<=0: continue
        mom=(c[i]/prev_r)-1.0
        s=0.0; s2=0.0
        for j in range(vol_p):
            r=(c[i-j]/c[i-j-1]-1.0) if i-j>0 and c[i-j-1]>0 else 0.0
            s+=r; s2+=r*r
        m=s/vol_p if vol_p>0 else 0.0; vol=np.sqrt(max(1e-20, s2/max(1,vol_p)-m*m))
        z=mom/vol
        if pos==0:
            if z>ez: pend=1
            elif z<-ez: pend=-1
        elif pos==1 and z<xz: pend=2
        elif pos==-1 and z>-xz: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_turtle_ls(c, o, h, l, entry_p, exit_p, atr_p, atr_stop, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); aa=_atr(h,l,c,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(entry_p, exit_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        eh=-1e18; el=1e18
        for j in range(1, entry_p+1):
            if h[i-j]>eh: eh=h[i-j]
            if l[i-j]<el: el=l[i-j]
        xl=1e18; xh=-1e18
        for j in range(1, exit_p+1):
            if l[i-j]<xl: xl=l[i-j]
            if h[i-j]>xh: xh=h[i-j]
        a=aa[i]
        if a!=a: pass
        else:
            if pos==1:
                if c[i]<ep/sb-atr_stop*a or c[i]<xl: pend=2
            elif pos==-1:
                if c[i]>ep/ss+atr_stop*a or c[i]>xh: pend=2
            if pos==0 and pend==0:
                if c[i]>eh: pend=1
                elif c[i]<el: pend=-1
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_bollinger_ls(c, o, period, num_std, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); ma=_rolling_mean(c,period); sd=_rolling_std(c,period)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(period, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        m=ma[i]; s=sd[i]
        if m!=m or s!=s or s<1e-10: pass
        else:
            u=m+num_std*s; lo=m-num_std*s
            if pos==0:
                if c[i]<lo: pend=1
                elif c[i]>u: pend=-1
            elif pos==1 and c[i]>=m: pend=2
            elif pos==-1 and c[i]<=m: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_keltner_ls(c, o, h, l, ema_p, atr_p, atr_m, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); ea=_ema(c,ema_p); aa=_atr(h,l,c,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(ema_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        e=ea[i]; a=aa[i]
        if e!=e or a!=a: pass
        else:
            if pos==0:
                if c[i]>e+atr_m*a: pend=1
                elif c[i]<e-atr_m*a: pend=-1
            elif pos==1 and c[i]<e: pend=2
            elif pos==-1 and c[i]>e: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_multifactor_ls(c, o, rsi_p, mom_p, vol_p, lt, st, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); rsi=_rsi_wilder(c,rsi_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(rsi_p+1, mom_p, vol_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        r=rsi[i]
        if r!=r: pass
        else:
            prev_mf=c[i-mom_p]; mom=(c[i]/prev_mf-1.0) if prev_mf>0 else 0.0
            rs=(100.0-r)/100.0; ms=max(-0.5,min(0.5,mom))+0.5
            s2=0.0
            for j in range(vol_p):
                ret=(c[i-j]/c[i-j-1]-1.0) if i-j>0 and c[i-j-1]>0 else 0.0; s2+=ret*ret
            vs=max(0.0,1.0-np.sqrt(s2/max(1,vol_p))*20.0); comp=(rs+ms+vs)/3.0
            if pos==0:
                if comp>lt: pend=1
                elif comp<st: pend=-1
            elif pos==1 and comp<0.5: pend=2
            elif pos==-1 and comp>0.5: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_volregime_ls(c, o, h, l, atr_p, vol_thr, ma_s, ma_l, rsi_p, rsi_os, rsi_ob, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); aa=_atr(h,l,c,atr_p); ra=_rsi_wilder(c,rsi_p)
    ms_a=_rolling_mean(c,ma_s); ml_a=_rolling_mean(c,ma_l)
    pos=0; ep=0.0; tr=1.0; pend=0; mode=0; pk=1.0; mdd=0.0; nt=0
    start=max(atr_p,rsi_p+1,ma_l)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        a=aa[i]
        if a!=a or c[i]<=0: pass
        else:
            hv=a/c[i]>vol_thr
            if hv:
                r=ra[i]
                if r!=r: pass
                else:
                    if pos==0:
                        if r<rsi_os: pend=1; mode=1
                        elif r>rsi_ob: pend=-1; mode=1
                    elif pos==1 and mode==1 and r>50: pend=2
                    elif pos==-1 and mode==1 and r<50: pend=2
            else:
                s_=ms_a[i]; l_=ml_a[i]; s0=ms_a[i-1]; l0=ml_a[i-1]
                if s_!=s_ or l_!=l_ or s0!=s0 or l0!=l0: pass
                else:
                    if pos==0:
                        if s0<=l0 and s_>l_: pend=1; mode=0
                        elif s0>=l0 and s_<l_: pend=-1; mode=0
                    elif pos==1 and mode==0 and s_<l_: pend=2
                    elif pos==-1 and mode==0 and s_>l_: pend=2
            if pos!=0 and pend==0:
                if (mode==0 and hv) or (mode==1 and not hv): pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_mesa_ls(c, o, fl, slow_lim, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<40: return 0.0, 0.0, 0
    smooth=np.zeros(n); det=np.zeros(n); I1=np.zeros(n); Q1=np.zeros(n)
    jI=np.zeros(n); jQ=np.zeros(n); I2=np.zeros(n); Q2=np.zeros(n)
    Re_=np.zeros(n); Im_=np.zeros(n); per=np.zeros(n); sp_=np.zeros(n)
    ph=np.zeros(n); mama=np.zeros(n); fama=np.zeros(n)
    for i in range(min(6,n)):
        mama[i]=c[i]; fama[i]=c[i]; per[i]=6.0; sp_[i]=6.0
    for i in range(6, n):
        smooth[i]=(4*c[i]+3*c[i-1]+2*c[i-2]+c[i-3])/10.0
        adj=0.075*per[i-1]+0.54
        det[i]=(0.0962*smooth[i]+0.5769*smooth[max(0,i-2)]-0.5769*smooth[max(0,i-4)]-0.0962*smooth[max(0,i-6)])*adj
        I1[i]=det[max(0,i-3)]
        Q1[i]=(0.0962*det[i]+0.5769*det[max(0,i-2)]-0.5769*det[max(0,i-4)]-0.0962*det[max(0,i-6)])*adj
        jI[i]=(0.0962*I1[i]+0.5769*I1[max(0,i-2)]-0.5769*I1[max(0,i-4)]-0.0962*I1[max(0,i-6)])*adj
        jQ[i]=(0.0962*Q1[i]+0.5769*Q1[max(0,i-2)]-0.5769*Q1[max(0,i-4)]-0.0962*Q1[max(0,i-6)])*adj
        I2[i]=I1[i]-jQ[i]; Q2[i]=Q1[i]+jI[i]
        I2[i]=0.2*I2[i]+0.8*I2[i-1]; Q2[i]=0.2*Q2[i]+0.8*Q2[i-1]
        Re_[i]=I2[i]*I2[i-1]+Q2[i]*Q2[i-1]; Im_[i]=I2[i]*Q2[i-1]-Q2[i]*I2[i-1]
        Re_[i]=0.2*Re_[i]+0.8*Re_[i-1]; Im_[i]=0.2*Im_[i]+0.8*Im_[i-1]
        per[i]=(2*np.pi/np.arctan(Im_[i]/Re_[i])) if (Im_[i]!=0 and Re_[i]!=0) else per[i-1]
        if per[i]>1.5*per[i-1]: per[i]=1.5*per[i-1]
        if per[i]<0.67*per[i-1]: per[i]=0.67*per[i-1]
        if per[i]<6.0: per[i]=6.0
        if per[i]>50.0: per[i]=50.0
        per[i]=0.2*per[i]+0.8*per[i-1]; sp_[i]=0.33*per[i]+0.67*sp_[i-1]
        ph[i]=(np.arctan(Q1[i]/I1[i])*180.0/np.pi) if I1[i]!=0 else ph[i-1]
        dp=ph[i-1]-ph[i]
        if dp<1.0: dp=1.0
        alpha=fl/dp
        if alpha<slow_lim: alpha=slow_lim
        if alpha>fl: alpha=fl
        mama[i]=alpha*c[i]+(1-alpha)*mama[i-1]
        fama[i]=0.5*alpha*mama[i]+(1-0.5*alpha)*fama[i-1]
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(7, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        if pos==0:
            if mama[i]>fama[i] and mama[i-1]<=fama[i-1]: pend=1
            elif mama[i]<fama[i] and mama[i-1]>=fama[i-1]: pend=-1
        elif pos==1 and mama[i]<fama[i] and mama[i-1]>=fama[i-1]: pend=-3
        elif pos==-1 and mama[i]>fama[i] and mama[i-1]<=fama[i-1]: pend=3
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_kama_ls(c, o, h, l, er_p, fast_sc, slow_sc, atr_sm, atr_p, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<er_p+2: return 0.0, 0.0, 0
    fc=2.0/(fast_sc+1.0); sc_v=2.0/(slow_sc+1.0)
    kama=np.full(n,np.nan); kama[er_p-1]=c[er_p-1]
    for i in range(er_p, n):
        d=abs(c[i]-c[i-er_p]); v=0.0
        for j in range(1, er_p+1): v+=abs(c[i-j+1]-c[i-j])
        er=d/v if v>0 else 0.0; sc2=(er*(fc-sc_v)+sc_v)**2
        kama[i]=kama[i-1]+sc2*(c[i]-kama[i-1])
    av=_atr(h,l,c,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(er_p+2, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        k=kama[i]; kp=kama[i-1]; a=av[i]
        if k!=k or kp!=kp or a!=a: pass
        else:
            if pos==1:
                if c[i]<ep/sb-atr_sm*a or k<kp: pend=2
            elif pos==-1:
                if c[i]>ep/ss+atr_sm*a or k>kp: pend=2
            if pos==0 and pend==0:
                if c[i]>k and k>kp: pend=1
                elif c[i]<k and k<kp: pend=-1
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_donchian_ls(c, o, h, l, entry_p, atr_p, atr_m, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<entry_p+atr_p: return 0.0, 0.0, 0
    av=_atr(h,l,c,atr_p)
    dh=np.full(n,np.nan); dl=np.full(n,np.nan)
    for i in range(entry_p-1, n):
        mx=h[i]; mn=l[i]
        for j in range(1, entry_p):
            if h[i-j]>mx: mx=h[i-j]
            if l[i-j]<mn: mn=l[i-j]
        dh[i]=mx; dl[i]=mn
    pos=0; ep=0.0; tr=1.0; pend=0; ts=0.0; pk=1.0; mdd=0.0; nt=0
    start=max(entry_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc
        if pend==1 and av[i]==av[i]: ts=o[i]-atr_m*av[i]
        elif pend==-1 and av[i]==av[i]: ts=o[i]+atr_m*av[i]
        pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        d1=dh[i-1]; d2=dl[i-1]; a=av[i]
        if d1!=d1 or d2!=d2 or a!=a: pass
        else:
            if pos==1:
                ns=c[i]-atr_m*a
                if ns>ts: ts=ns
                if c[i]<ts: pend=2
            elif pos==-1:
                ns=c[i]+atr_m*a
                if ns<ts: ts=ns
                if c[i]>ts: pend=2
            if pos==0 and pend==0:
                if c[i]>d1: pend=1
                elif c[i]<d2: pend=-1
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_zscore_ls(c, o, lookback, ez, xz, sz, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<lookback+2: return 0.0, 0.0, 0
    rm=np.full(n,np.nan); rs=np.full(n,np.nan); s=0.0; s2=0.0
    for i in range(n):
        s+=c[i]; s2+=c[i]*c[i]
        if i>=lookback: s-=c[i-lookback]; s2-=c[i-lookback]*c[i-lookback]
        if i>=lookback-1: m=s/lookback; rm[i]=m; rs[i]=np.sqrt(max(0.0,s2/lookback-m*m))
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(lookback, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        m=rm[i]; sd=rs[i]
        if sd==0 or sd!=sd: pass
        else:
            z=(c[i]-m)/sd
            if pos==1 and (z>-xz or z>sz): pend=2
            elif pos==-1 and (z<xz or z<-sz): pend=2
            if pos==0 and pend==0:
                if z<-ez: pend=1
                elif z>ez: pend=-1
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_mombreak_ls(c, o, h, l, hp, prox, atr_p, atr_t, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<max(hp, atr_p)+2: return 0.0, 0.0, 0
    rh=np.full(n,np.nan); rl=np.full(n,np.nan)
    for i in range(hp-1, n):
        mx=h[i]; mn=l[i]
        for j in range(1, hp):
            if h[i-j]>mx: mx=h[i-j]
            if l[i-j]<mn: mn=l[i-j]
        rh[i]=mx; rl[i]=mn
    av=_atr(h,l,c,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; ts_l=0.0; ts_s=1e18; pk=1.0; mdd=0.0; nt=0
    start=max(hp, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc
        if pend==1 and av[i]==av[i]: ts_l=o[i]-atr_t*av[i]
        elif pend==-1 and av[i]==av[i]: ts_s=o[i]+atr_t*av[i]
        pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        hv=rh[i]; lv=rl[i]; a=av[i]
        if hv!=hv or lv!=lv or a!=a: pass
        else:
            if pos==1:
                ns=c[i]-atr_t*a
                if ns>ts_l: ts_l=ns
                if c[i]<ts_l: pend=2
            elif pos==-1:
                ns=c[i]+atr_t*a
                if ns<ts_s: ts_s=ns
                if c[i]>ts_s: pend=2
            if pos==0 and pend==0:
                if c[i]>=hv*(1.0-prox): pend=1
                elif c[i]<=lv*(1.0+prox): pend=-1
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_regime_ema_ls(c, o, h, l, atr_p, vt, fe_p, se_p, te_p, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<max(atr_p, max(se_p, te_p))+2: return 0.0, 0.0, 0
    av=_atr(h,l,c,atr_p)
    fk=2.0/(fe_p+1.0); sk=2.0/(se_p+1.0); tk=2.0/(te_p+1.0)
    ef=np.empty(n); es=np.empty(n); et=np.empty(n)
    ef[0]=c[0]; es[0]=c[0]; et[0]=c[0]
    for i in range(1, n):
        ef[i]=c[i]*fk+ef[i-1]*(1-fk); es[i]=c[i]*sk+es[i-1]*(1-sk); et[i]=c[i]*tk+et[i-1]*(1-tk)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(atr_p, max(se_p, te_p))
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        a=av[i]
        if a!=a or c[i]<=0: pass
        else:
            hv=a/c[i]>vt
            if hv:
                if pos==0:
                    if ef[i]>es[i] and ef[i-1]<=es[i-1]: pend=1
                    elif ef[i]<es[i] and ef[i-1]>=es[i-1]: pend=-1
                elif pos==1 and ef[i]<es[i]: pend=2
                elif pos==-1 and ef[i]>es[i]: pend=2
            else:
                pd_=(c[i]-et[i])/et[i] if abs(et[i])>1e-20 else 0.0
                if pos==0:
                    if pd_<-0.02: pend=1
                    elif pd_>0.02: pend=-1
                elif pos==1 and pd_>0.0: pend=2
                elif pos==-1 and pd_<0.0: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_dualmom_ls(c, o, fast_lb, slow_lb, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); lb=max(fast_lb, slow_lb)
    if n<lb+2: return 0.0, 0.0, 0
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(lb, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        pf=c[i-fast_lb]; ps=c[i-slow_lb]
        fast_ret=(c[i]-pf)/pf if pf>0 else 0.0
        slow_ret=(c[i]-ps)/ps if ps>0 else 0.0
        if pos==0:
            if fast_ret>0 and slow_ret>0: pend=1
            elif fast_ret<0 and slow_ret<0: pend=-1
        elif pos==1:
            if fast_ret<0 or slow_ret<0: pend=2
        elif pos==-1:
            if fast_ret>0 or slow_ret>0: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_consensus_ls(c, o, ma_s_arr, ma_l_arr, rsi_arr, mom_lb,
                    rsi_os, rsi_ob, vote_thr, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<mom_lb+2: return 0.0, 0.0, 0
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(mom_lb, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        v=0; ms=ma_s_arr[i]; ml=ma_l_arr[i]; r=rsi_arr[i]
        if ms!=ms or ml!=ml or r!=r: continue
        if ms>ml: v+=1
        elif ms<ml: v-=1
        if r<rsi_os: v+=1
        elif r>rsi_ob: v-=1
        mom_ret=(c[i]-c[i-mom_lb])/c[i-mom_lb] if c[i-mom_lb]>0 else 0.0
        if mom_ret>0.02: v+=1
        elif mom_ret<-0.02: v-=1
        if pos==0:
            if v>=vote_thr: pend=1
            elif v<=-vote_thr: pend=-1
        elif pos==1 and v<=-1: pend=2
        elif pos==-1 and v>=1: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1 and ep>0:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1 and ep>0:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


# =====================================================================
#  Equity-tracking kernel variants — return (ret, dd, nt, equity)
#  Used by eval_kernel_detailed() for post-scan rich analysis.
#  Each mirrors its original bt_*_ls kernel with added equity recording.
# =====================================================================

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_ma(c, o, ma_s, ma_l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c); pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        s0=ma_s[i-1]; l0=ma_l[i-1]; s1=ma_s[i]; l1=ma_l[i]
        if s0!=s0 or l0!=l0 or s1!=s1 or l1!=l1: pass
        elif s0<=l0 and s1>l1:
            if pos==0: pend=1
            elif pos==-1: pend=3
        elif s0>=l0 and s1<l1:
            if pos==0: pend=-1
            elif pos==1: pend=-3
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_rsi(c, o, rsi, os_v, ob_v, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c); pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        r=rsi[i]
        if r!=r: pass
        elif r<os_v:
            if pos==0: pend=1
            elif pos==-1: pend=3
        elif r>ob_v:
            if pos==0: pend=-1
            elif pos==1: pend=-3
        elif pos==1 and r>50: pend=2
        elif pos==-1 and r<50: pend=2
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_macd(c, o, ef, es, sig_span, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c); ml=np.empty(n); sl_=np.empty(n)
    for i in range(n): ml[i]=ef[i]-es[i]
    k=2.0/(sig_span+1.0); sl_[0]=ml[0]
    for i in range(1,n): sl_[i]=ml[i]*k+sl_[i-1]*(1-k)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        mp=ml[i-1]; sp=sl_[i-1]; mc=ml[i]; sc=sl_[i]
        if mp!=mp or sp!=sp or mc!=mc or sc!=sc: pass
        elif mp<=sp and mc>sc:
            if pos==0: pend=1
            elif pos==-1: pend=3
        elif mp>=sp and mc<sc:
            if pos==0: pend=-1
            elif pos==1: pend=-3
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_drift(c, o, lookback, drift_thr, hold_p, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c); pos=0; ep=0.0; tr=1.0; pend=0; hc=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    for i in range(lookback, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; hc=0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        if tc2>0: hc=0
        up=0
        for j in range(1, lookback+1):
            if c[i-j+1]>c[i-j]: up+=1
        ratio=up/lookback
        if pos!=0:
            hc+=1
            if hc>=hold_p: pend=2; hc=0
        if pos==0 and pend==0:
            if ratio>=drift_thr: pend=-1; hc=0
            elif ratio<=1.0-drift_thr: pend=1; hc=0
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_ramom(c, o, mom_p, vol_p, ez, xz, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c); pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    start=max(mom_p, vol_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        prev_r=c[i-mom_p]
        if prev_r<=0: eq_arr[i]=tr; pos_arr[i]=pos; continue
        mom=(c[i]/prev_r)-1.0
        s=0.0; s2=0.0
        for j in range(vol_p):
            r=(c[i-j]/c[i-j-1]-1.0) if i-j>0 and c[i-j-1]>0 else 0.0
            s+=r; s2+=r*r
        m=s/max(1,vol_p); vol=np.sqrt(max(1e-20, s2/max(1,vol_p)-m*m))
        z=mom/vol
        if pos==0:
            if z>ez: pend=1
            elif z<-ez: pend=-1
        elif pos==1 and z<xz: pend=2
        elif pos==-1 and z>-xz: pend=2
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_turtle(c, o, h, l, entry_p, exit_p, atr_p, atr_stop, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c); aa=_atr(h,l,c,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    start=max(entry_p, exit_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        eh=-1e18; el=1e18
        for j in range(1, entry_p+1):
            if h[i-j]>eh: eh=h[i-j]
            if l[i-j]<el: el=l[i-j]
        xl=1e18; xh=-1e18
        for j in range(1, exit_p+1):
            if l[i-j]<xl: xl=l[i-j]
            if h[i-j]>xh: xh=h[i-j]
        a=aa[i]
        if a!=a: pass
        else:
            if pos==1:
                if c[i]<ep/sb-atr_stop*a or c[i]<xl: pend=2
            elif pos==-1:
                if c[i]>ep/ss+atr_stop*a or c[i]>xh: pend=2
            if pos==0 and pend==0:
                if c[i]>eh: pend=1
                elif c[i]<el: pend=-1
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_bollinger(c, o, period, num_std, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c); ma=_rolling_mean(c,period); sd=_rolling_std(c,period)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    for i in range(period, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        m=ma[i]; s=sd[i]
        if m!=m or s!=s or s<1e-10: pass
        else:
            u=m+num_std*s; lo=m-num_std*s
            if pos==0:
                if c[i]<lo: pend=1
                elif c[i]>u: pend=-1
            elif pos==1 and c[i]>=m: pend=2
            elif pos==-1 and c[i]<=m: pend=2
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_keltner(c, o, h, l, ema_p, atr_p, atr_m, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c); ea=_ema(c,ema_p); aa=_atr(h,l,c,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    start=max(ema_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        e=ea[i]; a=aa[i]
        if e!=e or a!=a: pass
        else:
            if pos==0:
                if c[i]>e+atr_m*a: pend=1
                elif c[i]<e-atr_m*a: pend=-1
            elif pos==1 and c[i]<e: pend=2
            elif pos==-1 and c[i]>e: pend=2
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_multifactor(c, o, rsi_p, mom_p, vol_p, lt, st, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c); rsi=_rsi_wilder(c,rsi_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    start=max(rsi_p+1, mom_p, vol_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        r=rsi[i]
        if r!=r: pass
        else:
            prev_mf=c[i-mom_p]; mom=(c[i]/prev_mf-1.0) if prev_mf>0 else 0.0
            rs=(100.0-r)/100.0; ms=max(-0.5,min(0.5,mom))+0.5
            s2=0.0
            for j in range(vol_p):
                ret=(c[i-j]/c[i-j-1]-1.0) if i-j>0 and c[i-j-1]>0 else 0.0; s2+=ret*ret
            vs=max(0.0,1.0-np.sqrt(s2/vol_p)*20.0); comp=(rs+ms+vs)/3.0
            if pos==0:
                if comp>lt: pend=1
                elif comp<st: pend=-1
            elif pos==1 and comp<0.5: pend=2
            elif pos==-1 and comp>0.5: pend=2
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_volregime(c, o, h, l, atr_p, vol_thr, ma_s, ma_l, rsi_p, rsi_os, rsi_ob,
                  sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c); aa=_atr(h,l,c,atr_p); ra=_rsi_wilder(c,rsi_p)
    ms_a=_rolling_mean(c,ma_s); ml_a=_rolling_mean(c,ma_l)
    pos=0; ep=0.0; tr=1.0; pend=0; mode=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    start=max(atr_p,rsi_p+1,ma_l)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        a=aa[i]
        if a!=a or c[i]<=0: pass
        else:
            hv=a/c[i]>vol_thr
            if hv:
                r=ra[i]
                if r!=r: pass
                else:
                    if pos==0:
                        if r<rsi_os: pend=1; mode=1
                        elif r>rsi_ob: pend=-1; mode=1
                    elif pos==1 and mode==1 and r>50: pend=2
                    elif pos==-1 and mode==1 and r<50: pend=2
            else:
                s_=ms_a[i]; l_=ml_a[i]; s0=ms_a[i-1]; l0=ml_a[i-1]
                if s_!=s_ or l_!=l_ or s0!=s0 or l0!=l0: pass
                else:
                    if pos==0:
                        if s0<=l0 and s_>l_: pend=1; mode=0
                        elif s0>=l0 and s_<l_: pend=-1; mode=0
                    elif pos==1 and mode==0 and s_<l_: pend=2
                    elif pos==-1 and mode==0 and s_>l_: pend=2
            if pos!=0 and pend==0:
                if (mode==0 and hv) or (mode==1 and not hv): pend=2
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_mesa(c, o, fl, slow_lim, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c)
    if n<40: return 0.0, 0.0, 0, np.ones(n, dtype=np.float64), 0, np.zeros(n, dtype=np.int64)
    smooth=np.zeros(n); det=np.zeros(n); I1=np.zeros(n); Q1=np.zeros(n)
    jI=np.zeros(n); jQ=np.zeros(n); I2=np.zeros(n); Q2=np.zeros(n)
    Re_=np.zeros(n); Im_=np.zeros(n); per=np.zeros(n); sp_=np.zeros(n)
    ph=np.zeros(n); mama=np.zeros(n); fama=np.zeros(n)
    for i in range(min(6,n)):
        mama[i]=c[i]; fama[i]=c[i]; per[i]=6.0; sp_[i]=6.0
    for i in range(6, n):
        smooth[i]=(4*c[i]+3*c[i-1]+2*c[i-2]+c[i-3])/10.0
        adj=0.075*per[i-1]+0.54
        det[i]=(0.0962*smooth[i]+0.5769*smooth[max(0,i-2)]-0.5769*smooth[max(0,i-4)]-0.0962*smooth[max(0,i-6)])*adj
        I1[i]=det[max(0,i-3)]
        Q1[i]=(0.0962*det[i]+0.5769*det[max(0,i-2)]-0.5769*det[max(0,i-4)]-0.0962*det[max(0,i-6)])*adj
        jI[i]=(0.0962*I1[i]+0.5769*I1[max(0,i-2)]-0.5769*I1[max(0,i-4)]-0.0962*I1[max(0,i-6)])*adj
        jQ[i]=(0.0962*Q1[i]+0.5769*Q1[max(0,i-2)]-0.5769*Q1[max(0,i-4)]-0.0962*Q1[max(0,i-6)])*adj
        I2[i]=I1[i]-jQ[i]; Q2[i]=Q1[i]+jI[i]
        I2[i]=0.2*I2[i]+0.8*I2[i-1]; Q2[i]=0.2*Q2[i]+0.8*Q2[i-1]
        Re_[i]=I2[i]*I2[i-1]+Q2[i]*Q2[i-1]; Im_[i]=I2[i]*Q2[i-1]-Q2[i]*I2[i-1]
        Re_[i]=0.2*Re_[i]+0.8*Re_[i-1]; Im_[i]=0.2*Im_[i]+0.8*Im_[i-1]
        per[i]=(2*np.pi/np.arctan(Im_[i]/Re_[i])) if (Im_[i]!=0 and Re_[i]!=0) else per[i-1]
        if per[i]>1.5*per[i-1]: per[i]=1.5*per[i-1]
        if per[i]<0.67*per[i-1]: per[i]=0.67*per[i-1]
        if per[i]<6.0: per[i]=6.0
        if per[i]>50.0: per[i]=50.0
        per[i]=0.2*per[i]+0.8*per[i-1]; sp_[i]=0.33*per[i]+0.67*sp_[i-1]
        ph[i]=(np.arctan(Q1[i]/I1[i])*180.0/np.pi) if I1[i]!=0 else ph[i-1]
        dp=ph[i-1]-ph[i]
        if dp<1.0: dp=1.0
        alpha=fl/dp
        if alpha<slow_lim: alpha=slow_lim
        if alpha>fl: alpha=fl
        mama[i]=alpha*c[i]+(1-alpha)*mama[i-1]
        fama[i]=0.5*alpha*mama[i]+(1-0.5*alpha)*fama[i-1]
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    for i in range(7, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        if pos==0:
            if mama[i]>fama[i] and mama[i-1]<=fama[i-1]: pend=1
            elif mama[i]<fama[i] and mama[i-1]>=fama[i-1]: pend=-1
        elif pos==1 and mama[i]<fama[i] and mama[i-1]>=fama[i-1]: pend=-3
        elif pos==-1 and mama[i]>fama[i] and mama[i-1]<=fama[i-1]: pend=3
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_kama(c, o, h, l, er_p, fast_sc, slow_sc, atr_sm, atr_p, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c)
    if n<er_p+2: return 0.0, 0.0, 0, np.ones(n, dtype=np.float64), 0, np.zeros(n, dtype=np.int64)
    fc=2.0/(fast_sc+1.0); sc_v=2.0/(slow_sc+1.0)
    kama=np.full(n,np.nan); kama[er_p-1]=c[er_p-1]
    for i in range(er_p, n):
        d=abs(c[i]-c[i-er_p]); v=0.0
        for j in range(1, er_p+1): v+=abs(c[i-j+1]-c[i-j])
        er=d/v if v>0 else 0.0; sc2=(er*(fc-sc_v)+sc_v)**2
        kama[i]=kama[i-1]+sc2*(c[i]-kama[i-1])
    av=_atr(h,l,c,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    start=max(er_p+2, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        k=kama[i]; kp=kama[i-1]; a=av[i]
        if k!=k or kp!=kp or a!=a: pass
        else:
            if pos==1:
                if c[i]<ep/sb-atr_sm*a or k<kp: pend=2
            elif pos==-1:
                if c[i]>ep/ss+atr_sm*a or k>kp: pend=2
            if pos==0 and pend==0:
                if c[i]>k and k>kp: pend=1
                elif c[i]<k and k<kp: pend=-1
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_donchian(c, o, h, l, entry_p, atr_p, atr_m, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c)
    if n<entry_p+atr_p: return 0.0, 0.0, 0, np.ones(n, dtype=np.float64), 0, np.zeros(n, dtype=np.int64)
    av=_atr(h,l,c,atr_p)
    dh=np.full(n,np.nan); dl=np.full(n,np.nan)
    for i in range(entry_p-1, n):
        mx=h[i]; mn=l[i]
        for j in range(1, entry_p):
            if h[i-j]>mx: mx=h[i-j]
            if l[i-j]<mn: mn=l[i-j]
        dh[i]=mx; dl[i]=mn
    pos=0; ep=0.0; tr=1.0; pend=0; ts=0.0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    start=max(entry_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc
        if pend==1 and av[i]==av[i]: ts=o[i]-atr_m*av[i]
        elif pend==-1 and av[i]==av[i]: ts=o[i]+atr_m*av[i]
        pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        d1=dh[i-1]; d2=dl[i-1]; a=av[i]
        if d1!=d1 or d2!=d2 or a!=a: pass
        else:
            if pos==1:
                ns=c[i]-atr_m*a
                if ns>ts: ts=ns
                if c[i]<ts: pend=2
            elif pos==-1:
                ns=c[i]+atr_m*a
                if ns<ts: ts=ns
                if c[i]>ts: pend=2
            if pos==0 and pend==0:
                if c[i]>d1: pend=1
                elif c[i]<d2: pend=-1
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_zscore(c, o, lookback, ez, xz, sz, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c)
    if n<lookback+2: return 0.0, 0.0, 0, np.ones(n, dtype=np.float64), 0, np.zeros(n, dtype=np.int64)
    rm=np.full(n,np.nan); rs=np.full(n,np.nan); s=0.0; s2=0.0
    for i in range(n):
        s+=c[i]; s2+=c[i]*c[i]
        if i>=lookback: s-=c[i-lookback]; s2-=c[i-lookback]*c[i-lookback]
        if i>=lookback-1: m=s/lookback; rm[i]=m; rs[i]=np.sqrt(max(0.0,s2/lookback-m*m))
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    for i in range(lookback, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        m=rm[i]; sd=rs[i]
        if sd==0 or sd!=sd: pass
        else:
            z=(c[i]-m)/sd
            if pos==1 and (z>-xz or z>sz): pend=2
            elif pos==-1 and (z<xz or z<-sz): pend=2
            if pos==0 and pend==0:
                if z<-ez: pend=1
                elif z>ez: pend=-1
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_mombreak(c, o, h, l, hp, prox, atr_p, atr_t, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c)
    if n<max(hp,atr_p)+2: return 0.0, 0.0, 0, np.ones(n, dtype=np.float64), 0, np.zeros(n, dtype=np.int64)
    rh=np.full(n,np.nan); rl=np.full(n,np.nan)
    for i in range(hp-1, n):
        mx=h[i]; mn=l[i]
        for j in range(1, hp):
            if h[i-j]>mx: mx=h[i-j]
            if l[i-j]<mn: mn=l[i-j]
        rh[i]=mx; rl[i]=mn
    av=_atr(h,l,c,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; ts_l=0.0; ts_s=1e18; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    start=max(hp, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc
        if pend==1 and av[i]==av[i]: ts_l=o[i]-atr_t*av[i]
        elif pend==-1 and av[i]==av[i]: ts_s=o[i]+atr_t*av[i]
        pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        hv=rh[i]; lv=rl[i]; a=av[i]
        if hv!=hv or lv!=lv or a!=a: pass
        else:
            if pos==1:
                ns=c[i]-atr_t*a
                if ns>ts_l: ts_l=ns
                if c[i]<ts_l: pend=2
            elif pos==-1:
                ns=c[i]+atr_t*a
                if ns<ts_s: ts_s=ns
                if c[i]>ts_s: pend=2
            if pos==0 and pend==0:
                if c[i]>=hv*(1.0-prox): pend=1
                elif c[i]<=lv*(1.0+prox): pend=-1
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_regime_ema(c, o, h, l, atr_p, vt, fe_p, se_p, te_p, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c)
    if n<max(atr_p,max(se_p,te_p))+2: return 0.0, 0.0, 0, np.ones(n, dtype=np.float64), 0, np.zeros(n, dtype=np.int64)
    av=_atr(h,l,c,atr_p)
    fk=2.0/(fe_p+1.0); sk=2.0/(se_p+1.0); tk=2.0/(te_p+1.0)
    ef=np.empty(n); es=np.empty(n); et=np.empty(n)
    ef[0]=c[0]; es[0]=c[0]; et[0]=c[0]
    for i in range(1, n):
        ef[i]=c[i]*fk+ef[i-1]*(1-fk); es[i]=c[i]*sk+es[i-1]*(1-sk); et[i]=c[i]*tk+et[i-1]*(1-tk)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    start=max(atr_p, max(se_p, te_p))
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        a=av[i]
        if a!=a or c[i]<=0: pass
        else:
            hv=a/c[i]>vt
            if hv:
                if pos==0:
                    if ef[i]>es[i] and ef[i-1]<=es[i-1]: pend=1
                    elif ef[i]<es[i] and ef[i-1]>=es[i-1]: pend=-1
                elif pos==1 and ef[i]<es[i]: pend=2
                elif pos==-1 and ef[i]>es[i]: pend=2
            else:
                pd_=(c[i]-et[i])/et[i] if abs(et[i])>1e-20 else 0.0
                if pos==0:
                    if pd_<-0.02: pend=1
                    elif pd_>0.02: pend=-1
                elif pos==1 and pd_>0.0: pend=2
                elif pos==-1 and pd_<0.0: pend=2
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_dualmom(c, o, fast_lb, slow_lb, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c); lb=max(fast_lb, slow_lb)
    if n<lb+2: return 0.0, 0.0, 0, np.ones(n, dtype=np.float64), 0, np.zeros(n, dtype=np.int64)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    for i in range(lb, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        pf=c[i-fast_lb]; ps=c[i-slow_lb]
        fast_ret=(c[i]-pf)/pf if pf>0 else 0.0
        slow_ret=(c[i]-ps)/ps if ps>0 else 0.0
        if pos==0:
            if fast_ret>0 and slow_ret>0: pend=1
            elif fast_ret<0 and slow_ret<0: pend=-1
        elif pos==1:
            if fast_ret<0 or slow_ret<0: pend=2
        elif pos==-1:
            if fast_ret>0 or slow_ret>0: pend=2
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_consensus(c, o, ma_s_arr, ma_l_arr, rsi_arr, mom_lb,
                  rsi_os, rsi_ob, vote_thr, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n=len(c)
    if n<mom_lb+2: return 0.0, 0.0, 0, np.ones(n, dtype=np.float64), 0, np.zeros(n, dtype=np.int64)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    eq_arr=np.ones(n, dtype=np.float64); pos_arr=np.zeros(n, dtype=np.int64)
    for i in range(mom_lb, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; eq_arr[i]=tr; pos_arr[i]=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        v=0; ms=ma_s_arr[i]; ml=ma_l_arr[i]; r=rsi_arr[i]
        if ms!=ms or ml!=ml or r!=r: pos_arr[i]=pos; eq_arr[i]=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); continue
        if ms>ml: v+=1
        elif ms<ml: v-=1
        if r<rsi_os: v+=1
        elif r>rsi_ob: v-=1
        mom_ret=(c[i]-c[i-mom_lb])/c[i-mom_lb] if c[i-mom_lb]>0 else 0.0
        if mom_ret>0.02: v+=1
        elif mom_ret<-0.02: v-=1
        if pos==0:
            if v>=vote_thr: pend=1
            elif v<=-vote_thr: pend=-1
        elif pos==1 and v<=-1: pend=2
        elif pos==-1 and v>=1: pend=2
        pos_arr[i]=pos; eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac); eq_arr[i]=eq
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    fpos=pos; tr,tc=_close_pos(pos,ep,tr,c[n-1],sb,ss,cm,lev,pfrac); nt+=tc; eq_arr[n-1]=tr
    return (tr-1.0)*100.0, mdd, nt, eq_arr, fpos, pos_arr


# =====================================================================
#  Registry
# =====================================================================

KERNEL_REGISTRY: Dict[str, Callable] = {
    "MA": bt_ma_ls,
    "RSI": bt_rsi_ls,
    "MACD": bt_macd_ls,
    "Drift": bt_drift_ls,
    "RAMOM": bt_ramom_ls,
    "Turtle": bt_turtle_ls,
    "Bollinger": bt_bollinger_ls,
    "Keltner": bt_keltner_ls,
    "MultiFactor": bt_multifactor_ls,
    "VolRegime": bt_volregime_ls,
    "MESA": bt_mesa_ls,
    "KAMA": bt_kama_ls,
    "Donchian": bt_donchian_ls,
    "ZScore": bt_zscore_ls,
    "MomBreak": bt_mombreak_ls,
    "RegimeEMA": bt_regime_ema_ls,
    "DualMom": bt_dualmom_ls,
    "Consensus": bt_consensus_ls,
}

KERNEL_NAMES: List[str] = list(KERNEL_REGISTRY.keys())

INDICATOR_DEPS: Dict[str, frozenset] = {
    "MA":          frozenset({"mas"}),
    "RSI":         frozenset({"rsis"}),
    "MACD":        frozenset({"emas"}),
    "Drift":       frozenset({"up_prefix"}),
    "RAMOM":       frozenset({"vols"}),
    "Turtle":      frozenset({"rmax_h", "rmin_l", "atr"}),
    "Bollinger":   frozenset({"mas", "stds"}),
    "Keltner":     frozenset({"emas", "atr"}),
    "MultiFactor": frozenset({"rsis", "vols"}),
    "VolRegime":   frozenset({"mas", "rsis", "atr"}),
    "MESA":        frozenset(),
    "KAMA":        frozenset({"atr"}),
    "Donchian":    frozenset({"rmax_h", "rmin_l", "atr"}),
    "ZScore":      frozenset({"mas", "stds"}),
    "MomBreak":    frozenset({"rmax_h", "rmin_l", "atr"}),
    "RegimeEMA":   frozenset({"emas", "atr"}),
    "DualMom":     frozenset(),
    "Consensus":   frozenset({"mas", "rsis"}),
}

POSITION_FRAC: Dict[int, float] = {
    1: 1.0, 2: 0.90, 4: 0.70, 5: 0.60,
    10: 0.40, 20: 0.25, 50: 0.15,
}


# =====================================================================
#  DEFAULT_PARAM_GRIDS — user can override any or all of these
#
#  To add a new strategy:
#    1. Write your @njit kernel: bt_newstrat_ls(c, o, ..., sb, ss, cm,
#       lev, dc, sl, pfrac, sl_slip) -> (ret%, maxDD%, nTrades)
#    2. Add it to KERNEL_REGISTRY
#    3. Add its default param grid to DEFAULT_PARAM_GRIDS
#    4. Add a _scan_<name> function below
#    5. Add @njit _scan_<name>_njit function and dispatch in scan_all_kernels
#    6. Add eval_kernel dispatch case
#    7. (Optional) Write a Python strategy class with kernel_name property
# =====================================================================

DEFAULT_PARAM_GRIDS: Dict[str, List[tuple]] = {
    "MA": [(s, lg) for s in range(5, 100, 3) for lg in range(s + 5, 201, 5)],
    "RSI": [(p, os_v, ob_v) for p in range(5, 101, 5)
            for os_v in range(15, 40, 5) for ob_v in range(60, 90, 5)],
    "MACD": [(f, s, sg) for f in range(4, 50, 3)
             for s in range(f + 4, 120, 5) for sg in range(3, min(s, 50), 4)],
    "Drift": [(lb, dt, hp) for lb in range(10, 120, 10)
              for dt in [0.55, 0.60, 0.65, 0.70] for hp in range(3, 25, 4)],
    "RAMOM": [(mp, vp, ez, xz) for mp in range(5, 100, 10)
              for vp in range(5, 50, 10)
              for ez in [1.0, 1.5, 2.0, 2.5, 3.0] for xz in [0.0, 0.5, 1.0]],
    "Turtle": [(ep, xp, ap, am) for ep in range(10, 80, 10)
               for xp in range(5, 40, 7) for ap in [10, 14, 20]
               for am in [1.5, 2.0, 2.5, 3.0]],
    "Bollinger": [(p, ns) for p in range(10, 120, 5)
                  for ns in [1.0, 1.5, 2.0, 2.5, 3.0]],
    "Keltner": [(ep, ap, am) for ep in range(10, 100, 10)
                for ap in [10, 14, 20] for am in [1.0, 1.5, 2.0, 2.5, 3.0]],
    "MultiFactor": [(rp, mp, vp, lt, st) for rp in [7, 14, 21]
                    for mp in range(10, 60, 10) for vp in range(10, 40, 10)
                    for lt in [0.55, 0.60, 0.65, 0.70]
                    for st in [0.25, 0.30, 0.35, 0.40]],
    "VolRegime": [(ap, vt, ms, ml, ros, rob)
                  for ap in [14, 20] for vt in [0.015, 0.020, 0.025, 0.030]
                  for ms in [5, 10, 20] for ml in [30, 50, 80] if ms < ml
                  for ros in [25, 30] for rob in [70, 75]],
    "MESA": [(fl, sl) for fl in [0.3, 0.5, 0.7] for sl in [0.02, 0.05, 0.10]],
    "KAMA": [(erp, fsc, ssc, asm, ap) for erp in [10, 15, 20] for fsc in [2, 3]
             for ssc in [20, 30, 50] for asm in [1.5, 2.0, 2.5, 3.0] for ap in [14, 20]],
    "Donchian": [(ep, ap, am) for ep in range(10, 80, 10)
                 for ap in [10, 14, 20] for am in [1.5, 2.0, 2.5, 3.0]],
    "ZScore": [(lb, ez, xz, sz) for lb in range(15, 100, 10)
               for ez in [1.5, 2.0, 2.5] for xz in [0.0, 0.5] for sz in [3.0, 4.0]],
    "MomBreak": [(hp, pp, ap, at) for hp in [20, 40, 60, 100, 200]
                 for pp in [0.00, 0.02, 0.05, 0.08]
                 for ap in [10, 14, 20] for at in [1.5, 2.0, 2.5, 3.0]],
    "RegimeEMA": [(ap, vt, fe, se, te) for ap in [14, 20]
                  for vt in [0.015, 0.020, 0.025]
                  for fe in [5, 10, 15] for se in [20, 40, 60] if fe < se
                  for te in [50, 80, 100]],
    "DualMom": [(fl, slo) for fl in [5, 10, 20, 40, 60]
                for slo in [20, 40, 80, 120, 200] if fl < slo],
    "Consensus": [(ms, ml, rp, mom_lb, os_v, ob_v, vt)
                  for ms in [10, 20] for ml in [50, 100, 150] if ms < ml
                  for rp in [14, 21] for mom_lb in [20, 40]
                  for os_v in [25, 30] for ob_v in [70, 75] for vt in [2, 3]],
}


@dataclass
class KernelResult:
    """Result from a single kernel run."""
    ret_pct: float
    max_dd_pct: float
    n_trades: int
    params: Any = None
    score: float = 0.0


def config_to_kernel_costs(config: BacktestConfig) -> Dict[str, float]:
    """Translate BacktestConfig into the flat cost params kernels expect."""
    lev = config.leverage
    base_slip_buy = config.slippage_bps_buy / 10000.0
    base_slip_sell = config.slippage_bps_sell / 10000.0
    slip = max(base_slip_buy, base_slip_sell) * math.sqrt(lev)
    sb = 1.0 + slip
    ss = 1.0 - slip
    cm = max(config.commission_pct_buy, config.commission_pct_sell)

    bpd = getattr(config, "bars_per_day", 1.0)
    dc = config.daily_funding_rate / bpd
    if config.funding_leverage_scaling and lev > 1.0:
        dc *= lev * (1.0 + 0.02 * lev)
    bpy = getattr(config, "bars_per_year", 252.0)
    # NOTE: borrow cost excluded from dc — it should only apply to short
    # positions, but dc is charged uniformly to all positions in _fx_lev.
    # Keeping borrow out prevents long positions from being penalised.
    # TODO: implement direction-specific per-bar costs (dc_long / dc_short).

    sl = config.stop_loss_pct if config.stop_loss_pct else 0.80
    pfrac = config.position_fraction
    if pfrac == 1.0 and lev > 1:
        pfrac = POSITION_FRAC.get(int(lev), max(0.10, 1.0 / math.sqrt(lev)))
    sl_slip = max(base_slip_buy, base_slip_sell) * lev * 0.5

    return {
        "sb": sb, "ss": ss, "cm": cm, "lev": lev,
        "dc": dc, "sl": sl, "pfrac": pfrac, "sl_slip": sl_slip,
        "bars_per_year": bpy,
    }


def run_kernel(
    name: str,
    params: tuple,
    c: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    config: BacktestConfig,
) -> KernelResult:
    """Run a single named kernel with BacktestConfig cost translation."""
    costs = config_to_kernel_costs(config)
    r, d, nt = eval_kernel(
        name, params, c, o, h, l,
        costs["sb"], costs["ss"], costs["cm"], costs["lev"],
        costs["dc"], costs["sl"], costs["pfrac"], costs["sl_slip"],
    )
    sc = _score(r, d, nt)
    return KernelResult(ret_pct=r, max_dd_pct=d, n_trades=nt, params=params, score=sc)


@dataclass
class DetailedKernelResult:
    """Rich result from a detailed kernel run — includes bar-by-bar equity."""
    ret_pct: float
    max_dd_pct: float
    n_trades: int
    params: Any = None
    score: float = 0.0
    equity: Optional[np.ndarray] = field(default=None, repr=False)
    daily_returns: Optional[np.ndarray] = field(default=None, repr=False)
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    final_position: int = 0


def run_kernel_detailed(
    name: str,
    params: tuple,
    c: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    config: BacktestConfig,
) -> DetailedKernelResult:
    """Run a kernel and return rich results with equity curve and risk metrics."""
    costs = config_to_kernel_costs(config)
    r, d, nt, eq, fpos, _ = eval_kernel_detailed(
        name, params, c, o, h, l,
        costs["sb"], costs["ss"], costs["cm"], costs["lev"],
        costs["dc"], costs["sl"], costs["pfrac"], costs["sl_slip"],
    )
    sc = _score(r, d, nt)
    bar_ret = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
    n_ret = len(bar_ret)
    bpy = costs["bars_per_year"]
    sharpe = sortino = calmar = 0.0
    if n_ret > 1:
        mu = np.mean(bar_ret)
        sigma = np.std(bar_ret)
        if sigma > 1e-10:
            sharpe = mu / sigma * np.sqrt(bpy)
        ds = bar_ret[bar_ret < 0]
        if len(ds) > 0:
            down_std = np.sqrt(np.mean(ds * ds))
            if down_std > 1e-10:
                sortino = mu / down_std * np.sqrt(bpy)
        if d > 0.01:
            ann_ret = (eq[-1] / eq[0]) ** (bpy / max(1, n_ret)) - 1.0
            calmar = ann_ret / (d / 100.0)
    return DetailedKernelResult(
        ret_pct=r, max_dd_pct=d, n_trades=nt, params=params, score=sc,
        equity=eq, daily_returns=bar_ret,
        sharpe=sharpe, sortino=sortino, calmar=calmar,
        final_position=fpos,
    )


def eval_kernel_detailed(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """Dispatch to equity-tracking kernel variant.

    Returns (ret, dd, nt, equity, final_pos, pos_arr).
    """
    if params is None:
        n = len(c)
        return (0.0, 0.0, 0, np.ones(n, dtype=np.float64), 0, np.zeros(n, dtype=np.int64))
    p = params
    if name == "MA":
        return _eq_ma(c, o, _rolling_mean(c, int(p[0])), _rolling_mean(c, int(p[1])),
                      sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "RSI":
        return _eq_rsi(c, o, _rsi_wilder(c, int(p[0])), float(p[1]), float(p[2]),
                       sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "MACD":
        return _eq_macd(c, o, _ema(c, int(p[0])), _ema(c, int(p[1])), int(p[2]),
                        sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Drift":
        return _eq_drift(c, o, int(p[0]), float(p[1]), int(p[2]),
                         sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "RAMOM":
        return _eq_ramom(c, o, int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                         sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Turtle":
        return _eq_turtle(c, o, h, l, int(p[0]), int(p[1]), int(p[2]), float(p[3]),
                          sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Bollinger":
        return _eq_bollinger(c, o, int(p[0]), float(p[1]),
                             sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Keltner":
        return _eq_keltner(c, o, h, l, int(p[0]), int(p[1]), float(p[2]),
                           sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "MultiFactor":
        return _eq_multifactor(c, o, int(p[0]), int(p[1]), int(p[2]), float(p[3]), float(p[4]),
                               sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "VolRegime":
        ms_, ml_ = int(max(2, p[2])), int(max(5, p[3]))
        if ms_ >= ml_:
            return (0.0, 0.0, 0, np.ones(len(c), dtype=np.float64), 0, np.zeros(len(c), dtype=np.int64))
        return _eq_volregime(c, o, h, l, int(p[0]), float(p[1]), ms_, ml_, 14,
                             int(p[4]), int(p[5]), sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "MESA":
        return _eq_mesa(c, o, float(p[0]), float(p[1]),
                        sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "KAMA":
        return _eq_kama(c, o, h, l, int(p[0]), int(p[1]), int(p[2]), float(p[3]), int(p[4]),
                        sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Donchian":
        return _eq_donchian(c, o, h, l, int(p[0]), int(p[1]), float(p[2]),
                            sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "ZScore":
        return _eq_zscore(c, o, int(p[0]), float(p[1]), float(p[2]), float(p[3]),
                          sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "MomBreak":
        return _eq_mombreak(c, o, h, l, int(p[0]), float(p[1]), int(p[2]), float(p[3]),
                            sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "RegimeEMA":
        fe, se = int(max(2, p[2])), int(max(5, p[3]))
        if fe >= se:
            return (0.0, 0.0, 0, np.ones(len(c), dtype=np.float64), 0, np.zeros(len(c), dtype=np.int64))
        return _eq_regime_ema(c, o, h, l, int(max(5, p[0])), float(p[1]), fe, se, int(p[4]),
                              sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "DualMom":
        fl_, sl_ = int(max(2, p[0])), int(max(5, p[1]))
        if fl_ >= sl_:
            return (0.0, 0.0, 0, np.ones(len(c), dtype=np.float64), 0, np.zeros(len(c), dtype=np.int64))
        return _eq_dualmom(c, o, fl_, sl_,
                           sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Consensus":
        ms_, ml_ = int(min(200, max(2, p[0]))), int(min(200, max(5, p[1])))
        if ms_ >= ml_:
            return (0.0, 0.0, 0, np.ones(len(c), dtype=np.float64), 0, np.zeros(len(c), dtype=np.int64))
        rp_ = int(min(200, max(2, p[2])))
        return _eq_consensus(c, o, _rolling_mean(c, ms_), _rolling_mean(c, ml_),
                             _rsi_wilder(c, rp_), int(p[3]), float(p[4]), float(p[5]),
                             int(p[6]), sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    return (0.0, 0.0, 0, np.ones(len(c), dtype=np.float64), 0, np.zeros(len(c), dtype=np.int64))


def eval_kernel_position(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """Run equity kernel and return only the final position: +1 (long), -1 (short), 0 (flat)."""
    _, _, _, _, fpos, _ = eval_kernel_detailed(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    return fpos


def eval_kernel_position_array(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """Run equity kernel and return per-bar position array in O(n).

    Returns np.ndarray of int64 with values +1 (long), -1 (short), 0 (flat)
    at each bar.  This is a single-pass O(n) operation — vastly faster than
    the O(n²) eval_kernel_position_series for large bar counts.
    """
    _, _, _, _, _, pos_arr = eval_kernel_detailed(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    return pos_arr


def eval_kernel_position_series(
    name, params, c, o, h, l,
    sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0,
    min_bars: int = 30,
) -> np.ndarray:
    """Return the kernel position at each bar as an int array (+1, -1, 0).

    Uses the O(n) position array extracted directly from the kernel in a
    single pass.  No progressive-window O(n²) loop needed.
    """
    return eval_kernel_position_array(
        name, params, c, o, h, l,
        sb, ss, cm, lev, dc, sl, pfrac, sl_slip,
    )


# =====================================================================
#  Single-kernel dispatch (used by BacktestEngine and robust_scan)
# =====================================================================

def eval_kernel(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """Dispatch to the correct kernel with on-the-fly indicator computation."""
    if params is None:
        return (0.0, 0.0, 0)
    p = params
    if name == "MA":
        return bt_ma_ls(c, o, _rolling_mean(c, int(p[0])), _rolling_mean(c, int(p[1])),
                        sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "RSI":
        return bt_rsi_ls(c, o, _rsi_wilder(c, int(p[0])), float(p[1]), float(p[2]),
                         sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "MACD":
        return bt_macd_ls(c, o, _ema(c, int(p[0])), _ema(c, int(p[1])), int(p[2]),
                          sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Drift":
        return bt_drift_ls(c, o, int(p[0]), float(p[1]), int(p[2]),
                           sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "RAMOM":
        return bt_ramom_ls(c, o, int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                           sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Turtle":
        return bt_turtle_ls(c, o, h, l, int(p[0]), int(p[1]), int(p[2]), float(p[3]),
                            sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Bollinger":
        return bt_bollinger_ls(c, o, int(p[0]), float(p[1]),
                               sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Keltner":
        return bt_keltner_ls(c, o, h, l, int(p[0]), int(p[1]), float(p[2]),
                             sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "MultiFactor":
        return bt_multifactor_ls(c, o, int(p[0]), int(p[1]), int(p[2]), float(p[3]), float(p[4]),
                                 sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "VolRegime":
        ms_, ml_ = int(max(2, p[2])), int(max(5, p[3]))
        if ms_ >= ml_:
            return (0.0, 0.0, 0)
        return bt_volregime_ls(c, o, h, l, int(p[0]), float(p[1]), ms_, ml_, 14,
                               int(p[4]), int(p[5]), sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "MESA":
        return bt_mesa_ls(c, o, float(p[0]), float(p[1]),
                          sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "KAMA":
        return bt_kama_ls(c, o, h, l, int(p[0]), int(p[1]), int(p[2]), float(p[3]), int(p[4]),
                          sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Donchian":
        return bt_donchian_ls(c, o, h, l, int(p[0]), int(p[1]), float(p[2]),
                              sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "ZScore":
        return bt_zscore_ls(c, o, int(p[0]), float(p[1]), float(p[2]), float(p[3]),
                            sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "MomBreak":
        return bt_mombreak_ls(c, o, h, l, int(p[0]), float(p[1]), int(p[2]), float(p[3]),
                              sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "RegimeEMA":
        fe, se = int(max(2, p[2])), int(max(5, p[3]))
        if fe >= se:
            return (0.0, 0.0, 0)
        return bt_regime_ema_ls(c, o, h, l, int(max(5, p[0])), float(p[1]), fe, se, int(p[4]),
                                sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "DualMom":
        fl_, sl_ = int(max(2, p[0])), int(max(5, p[1]))
        if fl_ >= sl_:
            return (0.0, 0.0, 0)
        return bt_dualmom_ls(c, o, fl_, sl_,
                             sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Consensus":
        ms_, ml_ = int(min(200, max(2, p[0]))), int(min(200, max(5, p[1])))
        if ms_ >= ml_:
            return (0.0, 0.0, 0)
        rp_ = int(min(200, max(2, p[2])))
        return bt_consensus_ls(c, o, _rolling_mean(c, ms_), _rolling_mean(c, ml_),
                               _rsi_wilder(c, rp_), int(p[3]), float(p[4]), float(p[5]),
                               int(p[6]), sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    return (0.0, 0.0, 0)


# =====================================================================
#  Numba-compiled scan functions — zero Python-dispatch overhead
# =====================================================================

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_ma_njit(grid, c, o, mas, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        si = int(grid[k,0]); li = int(grid[k,1])
        r,d,nt = bt_ma_ls(c,o,mas[si],mas[li],sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_rsi_njit(grid, c, o, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        r,d,nt = bt_rsi_ls(c,o,rsis[int(grid[k,0])],grid[k,1],grid[k,2],sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _precompute_macd_lines(emas, pairs, n):
    """Precompute MACD lines for unique (fast, slow) EMA pairs."""
    np_ = pairs.shape[0]
    out = np.empty((np_, n), dtype=np.float64)
    for p in range(np_):
        fi = int(pairs[p, 0]); si = int(pairs[p, 1])
        for i in range(n):
            out[p, i] = emas[fi, i] - emas[si, i]
    return out

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_macd_njit(grid, c, o, macd_lines, pair_idx, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        r,d,nt = bt_macd_precomp(c,o,macd_lines[pair_idx[k]],int(grid[k,2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_drift_njit(grid, c, o, up_prefix, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        r,d,nt = bt_drift_precomp(c,o,up_prefix,int(grid[k,0]),grid[k,1],int(grid[k,2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_ramom_njit(grid, c, o, vols, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        r,d,nt = bt_ramom_precomp(c,o,int(grid[k,0]),vols[int(grid[k,1])],grid[k,2],grid[k,3],sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_turtle_njit(grid, c, o, rmax_h, rmin_l, atr_10, atr_14, atr_20,
                      sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        ep_ = int(grid[k,0]); xp_ = int(grid[k,1]); ap_ = int(grid[k,2])
        if ap_ == 10: aa = atr_10
        elif ap_ == 14: aa = atr_14
        else: aa = atr_20
        r,d,nt = bt_turtle_precomp(c,o,aa,rmax_h[ep_],rmin_l[ep_],rmin_l[xp_],rmax_h[xp_],
                                    ep_,xp_,grid[k,3],sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_bollinger_njit(grid, c, o, mas, stds, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        p_ = int(grid[k,0])
        r,d,nt = bt_bollinger_precomp(c,o,mas[p_],stds[p_],grid[k,1],p_,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_keltner_njit(grid, c, o, emas, atr_10, atr_14, atr_20,
                       sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        ep_ = int(grid[k,0]); ap_ = int(grid[k,1])
        if ap_ == 10: aa = atr_10
        elif ap_ == 14: aa = atr_14
        else: aa = atr_20
        r,d,nt = bt_keltner_precomp(c,o,emas[ep_],aa,grid[k,2],ep_,ap_,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_multifactor_njit(grid, c, o, rsis, vols, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        rp_ = int(grid[k,0]); mp_ = int(grid[k,1]); vp_ = int(grid[k,2])
        r,d,nt = bt_multifactor_precomp(c,o,rsis[rp_],mp_,vols[vp_],grid[k,3],grid[k,4],
                                         sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_volregime_njit(grid, c, o, mas, rsis, atr_14, atr_20,
                         sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        ap_ = int(grid[k,0]); ms_ = int(grid[k,2]); ml_ = int(grid[k,3])
        aa = atr_14 if ap_ == 14 else atr_20
        start = max(ap_, 15, ml_)
        r,d,nt = bt_volregime_precomp(c,o,aa,rsis[14],mas[ms_],mas[ml_],
                                       grid[k,1],int(grid[k,4]),int(grid[k,5]),start,
                                       sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_mesa_njit(grid, c, o, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        r,d,nt = bt_mesa_precomp(c,o,grid[k,0],grid[k,1],sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _precompute_all_kama(close, unique_params):
    """Precompute KAMA arrays for unique (erp, fsc, ssc) parameter combos."""
    n = len(close)
    n_unique = unique_params.shape[0]
    out = np.full((n_unique, n), np.nan, dtype=np.float64)
    for k in range(n_unique):
        er_p = int(unique_params[k, 0])
        fast_sc = int(unique_params[k, 1])
        slow_sc = int(unique_params[k, 2])
        if n < er_p + 2:
            continue
        fc = 2.0 / (fast_sc + 1.0)
        sc_v = 2.0 / (slow_sc + 1.0)
        out[k, er_p - 1] = close[er_p - 1]
        for i in range(er_p, n):
            d = abs(close[i] - close[i - er_p])
            v = 0.0
            for j in range(1, er_p + 1):
                v += abs(close[i - j + 1] - close[i - j])
            er = d / v if v > 0 else 0.0
            sc2 = (er * (fc - sc_v) + sc_v) ** 2
            out[k, i] = out[k, i - 1] + sc2 * (close[i] - out[k, i - 1])
    return out

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_kama_njit(grid, c, o, kama_arrs, kama_idx, atr_14, atr_20,
                    sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        erp = int(grid[k,0]); asm = grid[k,3]; ap_ = int(grid[k,4])
        kama_arr = kama_arrs[kama_idx[k]]
        aa = atr_14 if ap_ == 14 else atr_20
        r,d,nt = bt_kama_precomp(c,o,kama_arr,aa,asm,erp,ap_,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_donchian_njit(grid, c, o, rmax_h, rmin_l, atr_10, atr_14, atr_20,
                        sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        ep_ = int(grid[k,0]); ap_ = int(grid[k,1])
        if ap_ == 10: aa = atr_10
        elif ap_ == 14: aa = atr_14
        else: aa = atr_20
        r,d,nt = bt_donchian_precomp(c,o,rmax_h[ep_],rmin_l[ep_],aa,grid[k,2],ep_,ap_,
                                      sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_zscore_njit(grid, c, o, mas, stds, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        lb = int(grid[k,0])
        r,d,nt = bt_zscore_precomp(c,o,mas[lb],stds[lb],lb,grid[k,1],grid[k,2],grid[k,3],
                                    sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_mombreak_njit(grid, c, o, rmax_h, rmin_l, atr_10, atr_14, atr_20,
                        sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        hp_ = int(grid[k,0]); ap_ = int(grid[k,2])
        if ap_ == 10: aa = atr_10
        elif ap_ == 14: aa = atr_14
        else: aa = atr_20
        r,d,nt = bt_mombreak_precomp(c,o,aa,rmax_h[hp_],rmin_l[hp_],hp_,grid[k,1],grid[k,3],
                                      sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_regime_ema_njit(grid, c, o, emas, atr_14, atr_20,
                          sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        ap_ = int(grid[k,0]); fe_ = int(grid[k,2]); se_ = int(grid[k,3]); te_ = int(grid[k,4])
        aa = atr_14 if ap_ == 14 else atr_20
        r,d,nt = bt_regime_ema_precomp(c,o,aa,emas[fe_],emas[se_],emas[te_],grid[k,1],ap_,se_,te_,
                                        sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_dualmom_njit(grid, c, o, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        r,d,nt = bt_dualmom_ls(c,o,int(grid[k,0]),int(grid[k,1]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng

@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_consensus_njit(grid, c, o, mas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng); _d = np.empty(ng); _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        ms_ = int(grid[k,0]); ml_ = int(grid[k,1])
        r,d,nt = bt_consensus_ls(c,o,mas[ms_],mas[ml_],rsis[int(grid[k,2])],int(grid[k,3]),
                                  grid[k,4],grid[k,5],int(grid[k,6]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        _sc[k] = _score(r,d,nt); _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng


# =====================================================================
#  Full parameter grid scan — user-configurable
# =====================================================================

def _grid_to_array(grid: List[tuple]) -> np.ndarray:
    """Convert list of tuples to 2D float64 array for Numba."""
    return np.array(grid, dtype=np.float64)


_GRID_CACHE: Dict[int, np.ndarray] = {}

def _cached_grid(grid: List[tuple]) -> np.ndarray:
    """Cache numpy conversion of grid by id (works for module-level DEFAULT lists)."""
    gid = id(grid)
    if gid not in _GRID_CACHE:
        _GRID_CACHE[gid] = _grid_to_array(grid)
    return _GRID_CACHE[gid]


def scan_all_kernels(
    c: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray,
    config: BacktestConfig,
    *,
    param_grids: Optional[Dict[str, List[tuple]]] = None,
    strategies: Optional[List[str]] = None,
    mas: Optional[np.ndarray] = None,
    emas: Optional[np.ndarray] = None,
    rsis: Optional[np.ndarray] = None,
    n_threads: Optional[int] = None,
    atr10: Optional[np.ndarray] = None,
    atr14: Optional[np.ndarray] = None,
    atr20: Optional[np.ndarray] = None,
    rmax_h: Optional[np.ndarray] = None,
    rmin_l: Optional[np.ndarray] = None,
    stds: Optional[np.ndarray] = None,
    vols: Optional[np.ndarray] = None,
    up_prefix: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """Scan strategies over parameter grids with thread-parallel execution.

    Numba njit functions release the GIL, so multiple strategies scan
    concurrently on separate CPU cores via ThreadPoolExecutor.

    Args:
        c, o, h, l: OHLC numpy arrays (float64).
        config: BacktestConfig with costs/leverage/stop-loss.
        param_grids: Custom parameter grids per strategy.
        strategies: Which strategies to scan. If None, scans all 18.
        mas, emas, rsis: Pre-computed indicator arrays (optional).
        n_threads: Number of parallel threads.  Set to 1 for sequential
                   (default — avoids nested-parallelism with inner prange).
        atr10, atr14, atr20: Pre-computed ATR arrays (optional).
        rmax_h, rmin_l: Pre-computed rolling max/min arrays (optional).
        stds: Pre-computed rolling std arrays (optional).
        vols: Pre-computed rolling vol arrays (optional).
        up_prefix: Pre-computed up-prefix array for Drift (optional).

    Returns:
        Dict mapping strategy name -> {params, score, ret, dd, nt, cnt}.
    """
    validate_ohlc(c, o, h, l)
    costs = config_to_kernel_costs(config)
    sb, ss_, cm = costs["sb"], costs["ss"], costs["cm"]
    lev, dc = costs["lev"], costs["dc"]
    sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

    grids = param_grids if param_grids is not None else DEFAULT_PARAM_GRIDS

    max_period = 200
    for _grid_vals in grids.values():
        for _p in _grid_vals:
            for _v in _p:
                if isinstance(_v, (int, np.integer)) and _v > max_period:
                    max_period = int(_v)
    max_period = min(max_period + 1, len(c) - 1)

    if mas is None:
        mas = precompute_all_ma(c, max_period)
    if emas is None:
        emas = precompute_all_ema(c, max_period)
    if rsis is None:
        rsis = precompute_all_rsi(c, max_period)
    strat_names = strategies or [sn for sn in KERNEL_NAMES if sn in grids]

    # prange inside each _scan_*_njit handles parallelism; outer threading
    # is redundant and causes nested-parallelism issues on some platforms.
    n_workers = n_threads if n_threads is not None else 1
    use_threads = n_workers > 1

    # ── Phase 1: precompute shared arrays based on indicator deps ──
    needed: set = set()
    for _sn in strat_names:
        needed |= INDICATOR_DEPS.get(_sn, frozenset())
    need_atr = "atr" in needed
    need_rmax = "rmax_h" in needed or "rmin_l" in needed
    need_stds = "stds" in needed
    need_vols = "vols" in needed
    need_up = "up_prefix" in needed

    if need_atr:
        if atr10 is None: atr10 = _atr(h, l, c, 10)
        if atr14 is None: atr14 = _atr(h, l, c, 14)
        if atr20 is None: atr20 = _atr(h, l, c, 20)
    if need_rmax:
        if rmax_h is None: rmax_h = precompute_rolling_max(h, max_period)
        if rmin_l is None: rmin_l = precompute_rolling_min(l, max_period)
    if need_stds and stds is None:
        stds = precompute_all_rolling_std(c, max_period)
    if need_vols and vols is None:
        vols = precompute_rolling_vol(c, max_period)
    if need_up and up_prefix is None:
        up_prefix = precompute_up_prefix(c)

    # KAMA precomputation: deduplicate (erp, fsc, ssc) tuples
    kama_arrs = kama_idx = None
    if "KAMA" in strat_names:
        raw_kama = grids.get("KAMA")
        if raw_kama:
            gid_k = id(raw_kama)
            ck_key = gid_k + 2
            cached_k = _GRID_CACHE.get(ck_key)
            if isinstance(cached_k, tuple):
                kama_unique, kama_idx = cached_k
            else:
                ga_k = _cached_grid(raw_kama)
                kparams = {}
                kama_idx = np.empty(ga_k.shape[0], dtype=np.int64)
                for k in range(ga_k.shape[0]):
                    key = (int(ga_k[k, 0]), int(ga_k[k, 1]), int(ga_k[k, 2]))
                    if key not in kparams:
                        kparams[key] = len(kparams)
                    kama_idx[k] = kparams[key]
                kama_unique = np.array(list(kparams.keys()), dtype=np.float64)
                _GRID_CACHE[ck_key] = (kama_unique, kama_idx)
            kama_arrs = _precompute_all_kama(c, kama_unique)

    # MACD pair mapping (cached) + precompute lines
    macd_lines = macd_pair_idx = None
    if "MACD" in strat_names:
        raw_macd = grids.get("MACD")
        if raw_macd:
            gid = id(raw_macd)
            cache_key = gid + 1
            cached = _GRID_CACHE.get(cache_key)
            if isinstance(cached, tuple):
                macd_pairs_arr, macd_pair_idx = cached
            else:
                ga_m = _cached_grid(raw_macd)
                pairs_set = {}
                macd_pair_idx = np.empty(ga_m.shape[0], dtype=np.int64)
                for k in range(ga_m.shape[0]):
                    key = (int(ga_m[k,0]), int(ga_m[k,1]))
                    if key not in pairs_set:
                        pairs_set[key] = len(pairs_set)
                    macd_pair_idx[k] = pairs_set[key]
                macd_pairs_arr = np.array(list(pairs_set.keys()), dtype=np.float64)
                _GRID_CACHE[cache_key] = (macd_pairs_arr, macd_pair_idx)
            macd_lines = _precompute_macd_lines(emas, macd_pairs_arr, len(c))

    # ── Build task list ──
    tasks = []
    for sn in strat_names:
        raw_grid = grids.get(sn)
        if not raw_grid:
            continue
        ga = _cached_grid(raw_grid)
        tasks.append((sn, raw_grid, ga))

    # ── Per-strategy dispatch via closure-bound lambdas (O(1) lookup) ──
    _cost = (sb, ss_, cm, lev, dc, sl, pfrac, sl_slip)
    _scan_dispatch = {
        "MA":          lambda ga: _scan_ma_njit(ga, c, o, mas, *_cost),
        "RSI":         lambda ga: _scan_rsi_njit(ga, c, o, rsis, *_cost),
        "MACD":        lambda ga: _scan_macd_njit(ga, c, o, macd_lines, macd_pair_idx, *_cost),
        "Drift":       lambda ga: _scan_drift_njit(ga, c, o, up_prefix, *_cost),
        "RAMOM":       lambda ga: _scan_ramom_njit(ga, c, o, vols, *_cost),
        "Turtle":      lambda ga: _scan_turtle_njit(ga, c, o, rmax_h, rmin_l, atr10, atr14, atr20, *_cost),
        "Bollinger":   lambda ga: _scan_bollinger_njit(ga, c, o, mas, stds, *_cost),
        "Keltner":     lambda ga: _scan_keltner_njit(ga, c, o, emas, atr10, atr14, atr20, *_cost),
        "MultiFactor": lambda ga: _scan_multifactor_njit(ga, c, o, rsis, vols, *_cost),
        "VolRegime":   lambda ga: _scan_volregime_njit(ga, c, o, mas, rsis, atr14, atr20, *_cost),
        "MESA":        lambda ga: _scan_mesa_njit(ga, c, o, *_cost),
        "KAMA":        lambda ga: _scan_kama_njit(ga, c, o, kama_arrs, kama_idx, atr14, atr20, *_cost),
        "Donchian":    lambda ga: _scan_donchian_njit(ga, c, o, rmax_h, rmin_l, atr10, atr14, atr20, *_cost),
        "ZScore":      lambda ga: _scan_zscore_njit(ga, c, o, mas, stds, *_cost),
        "MomBreak":    lambda ga: _scan_mombreak_njit(ga, c, o, rmax_h, rmin_l, atr10, atr14, atr20, *_cost),
        "RegimeEMA":   lambda ga: _scan_regime_ema_njit(ga, c, o, emas, atr14, atr20, *_cost),
        "DualMom":     lambda ga: _scan_dualmom_njit(ga, c, o, *_cost),
        "Consensus":   lambda ga: _scan_consensus_njit(ga, c, o, mas, rsis, *_cost),
    }
    _null_result = (-1, -1e18, 0.0, 0.0, 0, 0)

    def _run_one(sn, ga):
        try:
            return _scan_dispatch.get(sn, lambda _ga: _null_result)(ga)
        except Exception:
            return _null_result

    # ── Phase 2: scan strategies (threaded or serial) ──
    R: Dict[str, dict] = {}

    if not use_threads or len(tasks) <= 1:
        for sn, raw_grid, ga in tasks:
            bi, bs, br, bd, bn, cnt = _run_one(sn, ga)
            bp = raw_grid[bi] if bi >= 0 else None
            R[sn] = dict(params=bp, score=bs, ret=br, dd=bd, nt=int(bn), cnt=int(cnt))
    else:
        _pool = ThreadPoolExecutor(max_workers=n_workers)
        future_to_meta = {}
        for sn, raw_grid, ga in tasks:
            fut = _pool.submit(_run_one, sn, ga)
            future_to_meta[fut] = (sn, raw_grid)
        for fut in as_completed(future_to_meta):
            sn, raw_grid = future_to_meta[fut]
            bi, bs, br, bd, bn, cnt = fut.result()
            bp = raw_grid[bi] if bi >= 0 else None
            R[sn] = dict(params=bp, score=bs, ret=br, dd=bd, nt=int(bn), cnt=int(cnt))
        _pool.shutdown(wait=False)

    return R
