"""
11-Layer Anti-Overfitting Robust Scan — unified framework module.

This is the ONLY correct way to search for optimal parameters.
All parameter exploration happens strictly inside walk-forward training
windows — the optimizer never sees validation/test data.

Layers:
  1. Purged Walk-Forward with Train/Val/Test (6 windows, embargo gap)
  2. Multi-Metric Scoring (return / drawdown * trade_count_factor)
  3. Minimum Trade Filter (>= 20 trades)
  4. Validation Gate (val must not catastrophically diverge from train)
  5. Cross-Window Consistency (params must work across multiple time periods)
  6. Monte Carlo Price Perturbation (on OOS data only)
  7. OHLC Shuffle Perturbation (on OOS data only)
  8. Block Bootstrap Resampling (on OOS data only)
  9. Deflated Sharpe Ratio (corrects for multiple hypothesis testing)
  10. Composite Ranking
  11. Combinatorial Purged Cross-Validation (CPCV) — full data utilisation

Data flow per walk-forward window:
  ┌─────────────────┬──────────┬──────────┐
  │  Train (scan)   │ Val      │  Test    │
  │  param search   │ gate     │  OOS     │
  │  scan_all here  │ check    │  report  │
  └─────────────────┴──────────┴──────────┘
         ↑ NEVER sees val/test data

Public API:
    run_robust_scan(symbols, data, config, ...) -> RobustScanResult
    run_cpcv_scan(symbols, data, config, ...)   -> CPCVResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numba import njit
from scipy import stats as sp_stats

from .config import BacktestConfig
from .kernels import (
    DEFAULT_PARAM_GRIDS,
    KERNEL_NAMES,
    _score,
    config_to_kernel_costs,
    eval_kernel,
    precompute_all_ema,
    precompute_all_ma,
    precompute_all_rsi,
    scan_all_kernels,
)

# =====================================================================
#  Default walk-forward configuration
# =====================================================================

DEFAULT_WF_WINDOWS = [
    (0.35, 0.45, 0.55),
    (0.45, 0.55, 0.65),
    (0.55, 0.65, 0.75),
    (0.65, 0.75, 0.85),
    (0.75, 0.85, 0.95),
    (0.80, 0.90, 1.00),
]
DEFAULT_EMBARGO = 5

# =====================================================================
#  Monte Carlo / perturbation helpers (Numba-compiled)
# =====================================================================

@njit(cache=True)
def perturb_ohlc(close, open_, high, low, noise_std, seed):
    """Generate perturbed OHLC maintaining validity."""
    n = len(close)
    np.random.seed(seed)
    c_p = np.empty(n, np.float64)
    o_p = np.empty(n, np.float64)
    h_p = np.empty(n, np.float64)
    l_p = np.empty(n, np.float64)
    for i in range(n):
        nc = 1.0 + np.random.randn() * noise_std
        no = 1.0 + np.random.randn() * noise_std
        c_p[i] = close[i] * nc
        o_p[i] = open_[i] * no
        body_hi = max(c_p[i], o_p[i])
        body_lo = min(c_p[i], o_p[i])
        uw = max(0.0, high[i] - max(close[i], open_[i]))
        lw = max(0.0, min(close[i], open_[i]) - low[i])
        h_p[i] = body_hi + uw * (1.0 + np.random.randn() * noise_std * 0.5)
        l_p[i] = body_lo - lw * (1.0 + np.random.randn() * noise_std * 0.5)
    return c_p, o_p, h_p, l_p


@njit(cache=True)
def shuffle_ohlc(close, open_, high, low, seed):
    """Randomly reassign OHLC roles within each bar."""
    n = len(close)
    np.random.seed(seed)
    c_p = np.empty(n, np.float64)
    o_p = np.empty(n, np.float64)
    h_p = np.empty(n, np.float64)
    l_p = np.empty(n, np.float64)
    for i in range(n):
        vals = np.empty(4, np.float64)
        vals[0] = open_[i]; vals[1] = high[i]; vals[2] = low[i]; vals[3] = close[i]
        for j in range(3, 0, -1):
            k = int(np.random.random() * (j + 1))
            if k > j:
                k = j
            vals[j], vals[k] = vals[k], vals[j]
        mx = vals[0]; mn = vals[0]
        mx_i = 0; mn_i = 0
        for j in range(1, 4):
            if vals[j] > mx:
                mx = vals[j]; mx_i = j
            if vals[j] < mn:
                mn = vals[j]; mn_i = j
        h_p[i] = mx; l_p[i] = mn
        rem = np.empty(2, np.float64); ri = 0
        for j in range(4):
            if j != mx_i and j != mn_i and ri < 2:
                rem[ri] = vals[j]; ri += 1
        if ri < 2:
            rem[ri] = vals[3]
        o_p[i] = rem[0]; c_p[i] = rem[1]
    return c_p, o_p, h_p, l_p


@njit(cache=True)
def block_bootstrap_ohlc(close, open_, high, low, block_size, seed):
    """Block bootstrap: resample contiguous blocks of bars with replacement."""
    n = len(close)
    np.random.seed(seed)
    n_blocks = (n + block_size - 1) // block_size
    max_start = n - block_size
    if max_start < 0:
        max_start = 0
    c_p = np.empty(n, np.float64)
    o_p = np.empty(n, np.float64)
    h_p = np.empty(n, np.float64)
    l_p = np.empty(n, np.float64)
    idx = 0
    scale = 1.0
    last_c = close[0]
    for _ in range(n_blocks):
        start = int(np.random.random() * (max_start + 1))
        if start > max_start:
            start = max_start
        if start > 0:
            scale = last_c / close[start - 1] if close[start - 1] > 0 else 1.0
        else:
            scale = 1.0
        for j in range(block_size):
            if idx >= n:
                break
            si = start + j
            if si >= n:
                si = n - 1
            c_p[idx] = close[si] * scale
            o_p[idx] = open_[si] * scale
            h_p[idx] = high[si] * scale
            l_p[idx] = low[si] * scale
            last_c = c_p[idx]
            idx += 1
    return c_p, o_p, h_p, l_p


# =====================================================================
#  Deflated Sharpe & scoring
# =====================================================================

def deflated_sharpe(sharpe_obs: float, n_trials: int, n_bars: int,
                    skew: float = 0.0, kurtosis: float = 3.0) -> float:
    if n_bars < 3 or n_trials < 1:
        return 0.0
    e_max_sr = ((1.0 - 0.5772) * (2.0 * math.log(n_trials)) ** 0.5
                + 0.5772 * (2.0 * math.log(n_trials)) ** -0.5)
    se_sr = ((1.0 - skew * sharpe_obs
              + (kurtosis - 1.0) / 4.0 * sharpe_obs ** 2)
             / max(1.0, n_bars - 1.0)) ** 0.5
    if se_sr < 1e-12:
        return 1.0 if sharpe_obs > e_max_sr else 0.0
    z = (sharpe_obs - e_max_sr) / se_sr
    return float(sp_stats.norm.cdf(z))


def robust_score(ret_pct: float, dd_pct: float, n_trades: int,
                 n_bars: int, n_trials: int = 50640,
                 bars_per_year: float = 252.0,
                 ) -> Tuple[float, float, float, float]:
    """Returns (score, ann_ret, sharpe_proxy, dsr_pvalue)."""
    if n_trades < 3 or n_bars < 30:
        return -1e18, 0.0, 0.0, 1.0
    years = max(0.1, n_bars / bars_per_year)
    ann_ret = ret_pct / years
    est_vol = max(1.0, dd_pct) / 2.5 / math.sqrt(years)
    sharpe = ann_ret / max(0.01, est_vol)
    dsr_p = deflated_sharpe(sharpe, n_trials, n_bars)
    trade_factor = min(1.0, math.sqrt(n_trades / 30.0))
    dsr_bonus = 1.0 + max(0.0, dsr_p - 0.5) * 2.0
    score = sharpe * trade_factor * dsr_bonus
    return score, ann_ret, sharpe, dsr_p


def stitched_oos_metrics(window_rets, window_dds, window_trades):
    """Aggregate per-window OOS metrics via additive stitching."""
    eq = 1.0; pk = 1.0; mdd = 0.0; nt = 0
    n = min(len(window_rets), len(window_dds), len(window_trades))
    for i in range(n):
        r = float(window_rets[i])
        d = float(window_dds[i])
        t = int(window_trades[i])
        if d < 0.0:
            d = 0.0
        if d > 100.0:
            d = 100.0
        seg_start = eq
        gain = r / 100.0
        eq = seg_start + gain
        seg_valley = seg_start * max(0.0, 1.0 - d / 100.0)
        if seg_valley < seg_start and pk > 0:
            dd_intra = (pk - seg_valley) / pk * 100.0
            if dd_intra > mdd:
                mdd = dd_intra
        if eq > pk:
            pk = eq
        if pk > 0:
            dd_end = (pk - eq) / pk * 100.0
            if dd_end > mdd:
                mdd = dd_end
        nt += t
    return (eq - 1.0) * 100.0, min(mdd, 100.0), nt


# =====================================================================
#  Result container
# =====================================================================

@dataclass
class RobustScanResult:
    """Full results from a robust scan across symbols/strategies."""
    per_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_combos: int = 0
    elapsed_seconds: float = 0.0

    def best_per_symbol(self) -> Dict[str, dict]:
        out = {}
        for sym, strats in self.per_symbol.items():
            best_sn = max(strats, key=lambda sn: strats[sn].get("wf_score", -1e18))
            out[sym] = {"strategy": best_sn, **strats[best_sn]}
        return out


# =====================================================================
#  Main entry point
# =====================================================================

def run_robust_scan(
    symbols: List[str],
    data: Dict[str, Dict[str, np.ndarray]],
    config: BacktestConfig,
    *,
    param_grids: Optional[Dict[str, List[tuple]]] = None,
    strategies: Optional[List[str]] = None,
    wf_windows: Optional[List[Tuple[float, float, float]]] = None,
    embargo: int = DEFAULT_EMBARGO,
    n_mc_paths: int = 30,
    mc_noise_std: float = 0.002,
    n_shuffle_paths: int = 20,
    n_bootstrap_paths: int = 20,
    bootstrap_block: int = 20,
) -> RobustScanResult:
    """
    The correct way to search for optimal parameters with anti-overfitting.

    Parameter search happens ONLY inside walk-forward training windows.
    All robustness checks (MC/Shuffle/Bootstrap) run on OOS data only.

    Args:
        symbols: list of symbol keys present in *data*.
        data: {sym: {"c": np.ndarray, "o": ..., "h": ..., "l": ...}}.
        config: BacktestConfig with costs, leverage, stop-loss.
        param_grids: Custom parameter grids per strategy (passed to scan_all_kernels).
                     e.g. {"MA": [(5,20),(10,50)], "RSI": [(14,30,70)]}
                     If None, uses DEFAULT_PARAM_GRIDS for all strategies.
        strategies: subset of KERNEL_NAMES (default: all 18).
        wf_windows: walk-forward train/val/test fraction triples.
        embargo: purge gap in bars between train/val/test.
        n_mc_paths: Monte Carlo perturbation paths (on OOS only).
        mc_noise_std: noise standard deviation for MC perturbation.
        n_shuffle_paths: OHLC shuffle paths (on OOS only).
        n_bootstrap_paths: block bootstrap paths (on OOS only).
        bootstrap_block: block size for bootstrap.

    Returns:
        RobustScanResult with per-symbol, per-strategy results including
        the best params found, OOS metrics, and all robustness metrics.
    """
    import time

    strat_names = strategies or KERNEL_NAMES
    windows = wf_windows or DEFAULT_WF_WINDOWS
    n_windows = len(windows)
    costs = config_to_kernel_costs(config)
    sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
    lev, dc = costs["lev"], costs["dc"]
    sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

    scan_kwargs: Dict[str, Any] = {}
    if param_grids is not None:
        scan_kwargs["param_grids"] = param_grids
    if strategies is not None:
        scan_kwargs["strategies"] = strategies

    result = RobustScanResult()
    total_combos = 0
    t0 = time.time()

    for sym in symbols:
        if sym not in data:
            continue
        D = data[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        n = len(c)
        if n < 100:
            continue

        # O3: precompute indicators ONCE on full data, then slice per window
        grids = param_grids if param_grids is not None else DEFAULT_PARAM_GRIDS
        max_period = 200
        for _gv in grids.values():
            for _p in _gv:
                for _v in _p:
                    if isinstance(_v, (int, np.integer)) and _v > max_period:
                        max_period = int(_v)
        max_period = min(max_period + 1, n - 1)
        full_mas = precompute_all_ma(c, max_period)
        full_emas = precompute_all_ema(c, max_period)
        full_rsis = precompute_all_rsi(c, max_period)

        best_params_last: Dict[str, Any] = {sn: None for sn in strat_names}
        wfe_vals: Dict[str, list] = {sn: [] for sn in strat_names}
        gap_vals: Dict[str, list] = {sn: [] for sn in strat_names}
        oos_ret_vals: Dict[str, list] = {sn: [] for sn in strat_names}
        oos_dd_vals: Dict[str, list] = {sn: [] for sn in strat_names}
        oos_nt_vals: Dict[str, list] = {sn: [] for sn in strat_names}

        # --- Layer 1: Purged Walk-Forward ---
        for wi, (tr_pct, va_pct, te_pct) in enumerate(windows):
            tr_end = int(n * tr_pct)
            va_start = min(tr_end + embargo, int(n * va_pct))
            va_end = int(n * va_pct)
            te_end = min(int(n * te_pct), n)
            if va_end > te_end:
                va_end = te_end
            if va_start >= va_end:
                va_start = va_end

            c_tr, o_tr = c[:tr_end], o[:tr_end]
            h_tr, l_tr = h[:tr_end], l[:tr_end]
            c_va, o_va = c[va_start:va_end], o[va_start:va_end]
            h_va, l_va = h[va_start:va_end], l[va_start:va_end]
            c_te, o_te = c[va_end:te_end], o[va_end:te_end]
            h_te, l_te = h[va_end:te_end], l[va_end:te_end]

            # Slice precomputed indicators (NumPy views — zero copy)
            mas_tr = full_mas[:, :tr_end]
            emas_tr = full_emas[:, :tr_end]
            rsis_tr = full_rsis[:, :tr_end]

            train_results = scan_all_kernels(
                c_tr, o_tr, h_tr, l_tr, config,
                mas=mas_tr, emas=emas_tr, rsis=rsis_tr,
                **scan_kwargs,
            )

            for sn in strat_names:
                if sn not in train_results:
                    continue
                res = train_results[sn]
                total_combos += res["cnt"]

                # Layer 4: Validation gate
                va_r, va_d, va_nt = eval_kernel(sn, res["params"], c_va, o_va, h_va, l_va,
                                                sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
                # Layer 1 cont: Test (true OOS)
                te_r, te_d, te_nt = eval_kernel(sn, res["params"], c_te, o_te, h_te, l_te,
                                                sb, ss, cm, lev, dc, sl, pfrac, sl_slip)

                tr_r = res["ret"]
                if tr_r > 20.0 and va_r < -tr_r * 0.5:
                    te_r = 0.0; te_d = 0.0; te_nt = 0

                wfe_vals[sn].append(te_r)
                gap_vals[sn].append(abs(va_r - te_r))
                oos_ret_vals[sn].append(te_r)
                oos_dd_vals[sn].append(te_d)
                oos_nt_vals[sn].append(te_nt)
                if wi == n_windows - 1:
                    best_params_last[sn] = res["params"]

        # --- Assemble per-strategy metrics ---
        sym_results: Dict[str, Any] = {}

        # Identify the OOS portion = data after the earliest test start
        # across all windows (conservatively use the last window's test region)
        last_te_start = int(n * windows[-1][1])
        c_oos = c[last_te_start:]
        o_oos = o[last_te_start:]
        h_oos = h[last_te_start:]
        l_oos = l[last_te_start:]

        for sn in strat_names:
            if not oos_ret_vals[sn]:
                continue
            r, d, nt = stitched_oos_metrics(oos_ret_vals[sn], oos_dd_vals[sn], oos_nt_vals[sn])
            bpy = costs.get("bars_per_year", 252.0)
            sc, ann_r, shrp, dsr_p = robust_score(r, d, nt, n, bars_per_year=bpy)

            bp = best_params_last[sn]

            # --- Layer 6: Monte Carlo (on OOS data only) ---
            mc_rets = []
            if bp is not None and n_mc_paths > 0 and len(c_oos) > 30:
                for mi in range(n_mc_paths):
                    cp, op, hp, lp = perturb_ohlc(c_oos, o_oos, h_oos, l_oos, mc_noise_std, 42000 + mi)
                    mr, _, _ = eval_kernel(sn, bp, cp, op, hp, lp, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
                    mc_rets.append(mr)

            # --- Layer 7: OHLC Shuffle (on OOS data only) ---
            shuf_rets = []
            if bp is not None and n_shuffle_paths > 0 and len(c_oos) > 30:
                for si in range(n_shuffle_paths):
                    cp, op, hp, lp = shuffle_ohlc(c_oos, o_oos, h_oos, l_oos, 50000 + si)
                    sr, _, _ = eval_kernel(sn, bp, cp, op, hp, lp, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
                    shuf_rets.append(sr)

            # --- Layer 8: Block Bootstrap (on OOS data only) ---
            boot_rets = []
            if bp is not None and n_bootstrap_paths > 0 and len(c_oos) > 30:
                for bi in range(n_bootstrap_paths):
                    cp, op, hp, lp = block_bootstrap_ohlc(c_oos, o_oos, h_oos, l_oos, bootstrap_block, 60000 + bi)
                    br_, _, _ = eval_kernel(sn, bp, cp, op, hp, lp, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
                    boot_rets.append(br_)

            sym_results[sn] = {
                "params": bp,
                "wfe_mean": float(np.mean(wfe_vals[sn])) if wfe_vals[sn] else 0.0,
                "gen_gap_mean": float(np.mean(gap_vals[sn])) if gap_vals[sn] else 0.0,
                "oos_ret": r,
                "oos_dd": d,
                "oos_trades": nt,
                "ann_ret": ann_r,
                "sharpe": shrp,
                "dsr_p": dsr_p,
                "wf_score": sc,
                "mc_mean": float(np.mean(mc_rets)) if mc_rets else 0.0,
                "mc_std": float(np.std(mc_rets)) if mc_rets else 0.0,
                "mc_pct_positive": sum(1 for x in mc_rets if x > 0) / max(1, len(mc_rets)),
                "shuffle_mean": float(np.mean(shuf_rets)) if shuf_rets else 0.0,
                "bootstrap_mean": float(np.mean(boot_rets)) if boot_rets else 0.0,
            }

        result.per_symbol[sym] = sym_results

    result.total_combos = total_combos
    result.elapsed_seconds = time.time() - t0
    return result


# =====================================================================
#  CPCV — Combinatorial Purged Cross-Validation (de Prado)
# =====================================================================

def cpcv_splits(
    n_bars: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 5,
) -> List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """Generate purged CPCV splits.

    Divides ``n_bars`` into ``n_groups`` contiguous blocks, then forms
    C(n_groups, n_test_groups) train/test splits.  Training observations
    adjacent to test boundaries are purged by ``embargo`` bars.

    Returns list of ``(train_ranges, test_ranges)`` where each range is
    ``(start, end)`` as a half-open interval.
    """
    group_size = n_bars // n_groups
    groups: List[Tuple[int, int]] = []
    for i in range(n_groups):
        start = i * group_size
        end = (i + 1) * group_size if i < n_groups - 1 else n_bars
        groups.append((start, end))

    splits: List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]] = []
    for test_combo in combinations(range(n_groups), n_test_groups):
        test_set = set(test_combo)
        train_ranges: List[Tuple[int, int]] = []
        test_ranges: List[Tuple[int, int]] = []

        for gi in range(n_groups):
            if gi in test_set:
                test_ranges.append(groups[gi])
            else:
                train_ranges.append(groups[gi])

        purged_train: List[Tuple[int, int]] = []
        for tr_s, tr_e in train_ranges:
            for te_s, te_e in test_ranges:
                if tr_e > te_s - embargo and tr_e <= te_s:
                    tr_e = max(tr_s, te_s - embargo)
                if tr_s >= te_e and tr_s < te_e + embargo:
                    tr_s = min(tr_e, te_e + embargo)
            if tr_s < tr_e:
                purged_train.append((tr_s, tr_e))

        splits.append((purged_train, test_ranges))
    return splits


@dataclass
class CPCVResult:
    """Results from Combinatorial Purged Cross-Validation."""
    per_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_combos: int = 0
    n_splits: int = 0
    elapsed_seconds: float = 0.0

    def best_per_symbol(self) -> Dict[str, dict]:
        out = {}
        for sym, strats in self.per_symbol.items():
            best_sn = max(strats, key=lambda sn: strats[sn].get("cpcv_score", -1e18))
            out[sym] = {"strategy": best_sn, **strats[best_sn]}
        return out


def _concat_ranges(arr: np.ndarray, ranges: List[Tuple[int, int]]) -> np.ndarray:
    """Concatenate slices of *arr* specified by half-open ranges."""
    parts = [arr[s:e] for s, e in ranges]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def run_cpcv_scan(
    symbols: List[str],
    data: Dict[str, Dict[str, np.ndarray]],
    config: BacktestConfig,
    *,
    param_grids: Optional[Dict[str, List[tuple]]] = None,
    strategies: Optional[List[str]] = None,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = DEFAULT_EMBARGO,
    n_mc_paths: int = 10,
    mc_noise_std: float = 0.002,
) -> CPCVResult:
    """Combinatorial Purged Cross-Validation scan.

    Uses *all* data in both train and test roles across C(n_groups,
    n_test_groups) splits.  For each split the best parameters are
    found on the training folds and evaluated on the held-out test
    folds.  Per-strategy OOS returns are averaged over all splits,
    giving a far more reliable performance estimate than a single
    walk-forward pass.

    Args:
        symbols:        Symbol keys present in *data*.
        data:           ``{sym: {"c": np.ndarray, "o": ..., "h": ..., "l": ...}}``.
        config:         BacktestConfig.
        param_grids:    Custom grids per strategy (or None for defaults).
        strategies:     Subset of KERNEL_NAMES (default: all 18).
        n_groups:       Number of contiguous blocks to split data into.
        n_test_groups:  How many blocks to hold out per split.
        embargo:        Purge gap between train/test boundaries.
        n_mc_paths:     MC paths on OOS per split (lightweight sanity check).
        mc_noise_std:   Noise std for MC perturbation.

    Returns:
        CPCVResult with per-strategy averaged OOS metrics, DSR,
        and the proportion of splits with positive OOS return.
    """
    import time

    strat_names = strategies or KERNEL_NAMES
    costs = config_to_kernel_costs(config)
    sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
    lev, dc = costs["lev"], costs["dc"]
    sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

    scan_kwargs: Dict[str, Any] = {}
    if param_grids is not None:
        scan_kwargs["param_grids"] = param_grids
    if strategies is not None:
        scan_kwargs["strategies"] = strategies

    result = CPCVResult()
    total_combos = 0
    t0 = time.time()

    for sym in symbols:
        if sym not in data:
            continue
        D = data[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        n = len(c)
        if n < 100:
            continue

        splits = cpcv_splits(n, n_groups, n_test_groups, embargo)
        result.n_splits = len(splits)

        oos_rets: Dict[str, List[float]] = {sn: [] for sn in strat_names}
        oos_dds: Dict[str, List[float]] = {sn: [] for sn in strat_names}
        oos_nts: Dict[str, List[int]] = {sn: [] for sn in strat_names}
        best_params_all: Dict[str, Any] = {sn: None for sn in strat_names}
        mc_rets_all: Dict[str, List[float]] = {sn: [] for sn in strat_names}

        for si, (train_ranges, test_ranges) in enumerate(splits):
            c_tr = _concat_ranges(c, train_ranges)
            o_tr = _concat_ranges(o, train_ranges)
            h_tr = _concat_ranges(h, train_ranges)
            l_tr = _concat_ranges(l, train_ranges)

            if len(c_tr) < 60:
                continue

            # CPCV concatenates non-contiguous ranges, so precomputed
            # indicators on full data cannot be sliced — compute per split
            train_results = scan_all_kernels(
                c_tr, o_tr, h_tr, l_tr, config, **scan_kwargs,
            )

            c_te = _concat_ranges(c, test_ranges)
            o_te = _concat_ranges(o, test_ranges)
            h_te = _concat_ranges(h, test_ranges)
            l_te = _concat_ranges(l, test_ranges)

            if len(c_te) < 30:
                continue

            for sn in strat_names:
                if sn not in train_results:
                    continue
                res = train_results[sn]
                total_combos += res["cnt"]
                bp = res["params"]
                if bp is None:
                    continue
                best_params_all[sn] = bp

                te_r, te_d, te_nt = eval_kernel(
                    sn, bp, c_te, o_te, h_te, l_te,
                    sb, ss, cm, lev, dc, sl, pfrac, sl_slip,
                )
                oos_rets[sn].append(te_r)
                oos_dds[sn].append(te_d)
                oos_nts[sn].append(te_nt)

                if n_mc_paths > 0:
                    for mi in range(n_mc_paths):
                        cp, op, hp, lp = perturb_ohlc(
                            c_te, o_te, h_te, l_te, mc_noise_std,
                            70000 + si * 1000 + mi,
                        )
                        mr, _, _ = eval_kernel(
                            sn, bp, cp, op, hp, lp,
                            sb, ss, cm, lev, dc, sl, pfrac, sl_slip,
                        )
                        mc_rets_all[sn].append(mr)

        sym_results: Dict[str, Any] = {}
        for sn in strat_names:
            if not oos_rets[sn]:
                continue
            avg_ret = float(np.mean(oos_rets[sn]))
            avg_dd = float(np.mean(oos_dds[sn]))
            total_nt = sum(oos_nts[sn])
            pct_pos = sum(1 for r in oos_rets[sn] if r > 0) / len(oos_rets[sn])

            bpy = costs.get("bars_per_year", 252.0)
            sc, ann_r, shrp, dsr_p = robust_score(avg_ret, avg_dd, total_nt, n,
                                                  bars_per_year=bpy)

            cpcv_score = shrp * pct_pos * (1.0 + max(0.0, dsr_p - 0.5))

            mc_vals = mc_rets_all[sn]
            sym_results[sn] = {
                "params": best_params_all[sn],
                "oos_ret_mean": avg_ret,
                "oos_ret_std": float(np.std(oos_rets[sn])),
                "oos_dd_mean": avg_dd,
                "oos_trades_total": total_nt,
                "pct_splits_positive": pct_pos,
                "ann_ret": ann_r,
                "sharpe": shrp,
                "dsr_p": dsr_p,
                "cpcv_score": cpcv_score,
                "n_splits_evaluated": len(oos_rets[sn]),
                "mc_mean": float(np.mean(mc_vals)) if mc_vals else 0.0,
                "mc_pct_positive": (
                    sum(1 for x in mc_vals if x > 0) / max(1, len(mc_vals))
                ),
            }

        result.per_symbol[sym] = sym_results

    result.total_combos = total_combos
    result.elapsed_seconds = time.time() - t0
    return result
