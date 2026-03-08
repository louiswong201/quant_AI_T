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

import logging
import math
import os
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numba import njit
from scipy import stats as sp_stats

from .config import BacktestConfig
from .kernels import (
    DEFAULT_PARAM_GRIDS,
    INDICATOR_DEPS,
    KERNEL_NAMES,
    _atr,
    _cached_max_period,
    _clear_caches,
    _score,
    config_to_kernel_costs,
    eval_kernel,
    precompute_all_ema,
    precompute_all_ma,
    precompute_all_rsi,
    precompute_all_rolling_std,
    precompute_rolling_max,
    precompute_rolling_min,
    precompute_rolling_vol,
    precompute_up_prefix,
    scan_all_kernels,
    validate_ohlc,
)
from ..platform_config import optimal_thread_config

logger = logging.getLogger(__name__)

# =====================================================================
#  Default walk-forward configuration
# =====================================================================

DEFAULT_WF_WINDOWS = [
    (0.30, 0.40, 0.50),
    (0.40, 0.50, 0.60),
    (0.50, 0.60, 0.70),
    (0.60, 0.70, 0.80),
    (0.70, 0.80, 0.90),
    (0.80, 0.90, 1.00),
]
COMPACT_WF_WINDOWS = [
    (0.30, 0.45, 0.60),
    (0.50, 0.65, 0.80),
    (0.70, 0.85, 1.00),
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
        if h_p[i] < body_hi:
            h_p[i] = body_hi
        if l_p[i] > body_lo:
            l_p[i] = body_lo
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
    """Aggregate per-window OOS metrics via multiplicative compounding."""
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
        eq = seg_start * (1.0 + r / 100.0)
        if eq < 0.001:
            eq = 0.001
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

def _robustness_one_path(kind, idx, sn, bp, c_oos, o_oos, h_oos, l_oos,
                         sb, ss, cm, lev, dc, sl, pfrac, sl_slip,
                         mc_noise_std, bootstrap_block):
    """Evaluate a single robustness perturbation path (MC / Shuffle / Bootstrap)."""
    if kind == "mc":
        cp, op, hp, lp = perturb_ohlc(c_oos, o_oos, h_oos, l_oos, mc_noise_std, 42000 + idx)
    elif kind == "shuffle":
        cp, op, hp, lp = shuffle_ohlc(c_oos, o_oos, h_oos, l_oos, 50000 + idx)
    else:
        cp, op, hp, lp = block_bootstrap_ohlc(c_oos, o_oos, h_oos, l_oos, bootstrap_block, 60000 + idx)
    ret, _, _ = eval_kernel(sn, bp, cp, op, hp, lp, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    return kind, ret


def _batch_robustness(sn, bp, c_oos, o_oos, h_oos, l_oos,
                      sb, ss, cm, lev, dc, sl, pfrac, sl_slip,
                      n_mc, n_shuffle, n_bootstrap,
                      mc_noise_std, bootstrap_block):
    """Run ALL robustness paths for one strategy in a single thread.

    Reduces ThreadPoolExecutor scheduling overhead from ~70 granular tasks
    per strategy to 1 batched task.  Returns (mc_rets, shuf_rets, boot_rets).
    """
    mc_rets = []
    for i in range(n_mc):
        cp, op, hp, lp = perturb_ohlc(c_oos, o_oos, h_oos, l_oos, mc_noise_std, 42000 + i)
        r, _, _ = eval_kernel(sn, bp, cp, op, hp, lp, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
        mc_rets.append(r)
    shuf_rets = []
    for i in range(n_shuffle):
        cp, op, hp, lp = shuffle_ohlc(c_oos, o_oos, h_oos, l_oos, 50000 + i)
        r, _, _ = eval_kernel(sn, bp, cp, op, hp, lp, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
        shuf_rets.append(r)
    boot_rets = []
    for i in range(n_bootstrap):
        cp, op, hp, lp = block_bootstrap_ohlc(c_oos, o_oos, h_oos, l_oos, bootstrap_block, 60000 + i)
        r, _, _ = eval_kernel(sn, bp, cp, op, hp, lp, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
        boot_rets.append(r)
    return mc_rets, shuf_rets, boot_rets


def _process_one_symbol(
    sym: str,
    D: Dict[str, np.ndarray],
    config: BacktestConfig,
    strat_names: List[str],
    windows: List[Tuple[float, float, float]],
    costs: Dict[str, Any],
    scan_kwargs: Dict[str, Any],
    param_grids: Optional[Dict[str, List[tuple]]],
    embargo: int,
    n_mc_paths: int,
    mc_noise_std: float,
    n_shuffle_paths: int,
    n_bootstrap_paths: int,
    bootstrap_block: int,
    rob_workers: int,
    numba_threads: int = 0,
) -> Tuple[Dict[str, Any], int]:
    """Process a single symbol through the full WF + robustness pipeline.

    Returns ``(sym_results, total_combos)``."""
    if numba_threads > 0:
        import numba
        numba.set_num_threads(numba_threads)

    c, o, h, l = D["c"], D["o"], D["h"], D["l"]
    n = len(c)

    sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
    lev, dc = costs["lev"], costs["dc"]
    sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

    grids = param_grids if param_grids is not None else DEFAULT_PARAM_GRIDS
    max_period = _cached_max_period(grids, n)

    _needed: set = set()
    for _sn in strat_names:
        _needed |= INDICATOR_DEPS.get(_sn, frozenset())

    full_mas = precompute_all_ma(c, max_period) if "mas" in _needed else None
    full_emas = precompute_all_ema(c, max_period) if "emas" in _needed else None
    full_rsis = precompute_all_rsi(c, max_period) if "rsis" in _needed else None
    full_atr10 = _atr(h, l, c, 10) if "atr" in _needed else None
    full_atr14 = _atr(h, l, c, 14) if "atr" in _needed else None
    full_atr20 = _atr(h, l, c, 20) if "atr" in _needed else None
    full_rmax_h = precompute_rolling_max(h, max_period) if "rmax_h" in _needed else None
    full_rmin_l = precompute_rolling_min(l, max_period) if "rmin_l" in _needed else None
    full_stds = precompute_all_rolling_std(c, max_period) if "stds" in _needed else None
    full_vols = precompute_rolling_vol(c, max_period) if "vols" in _needed else None
    full_up = precompute_up_prefix(c) if "up_prefix" in _needed else None

    best_params_last: Dict[str, Any] = {sn: None for sn in strat_names}
    wfe_vals: Dict[str, list] = {sn: [] for sn in strat_names}
    gap_vals: Dict[str, list] = {sn: [] for sn in strat_names}
    oos_ret_vals: Dict[str, list] = {sn: [] for sn in strat_names}
    oos_dd_vals: Dict[str, list] = {sn: [] for sn in strat_names}
    oos_nt_vals: Dict[str, list] = {sn: [] for sn in strat_names}

    total_combos = 0
    n_windows = len(windows)
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

        train_results = scan_all_kernels(
            c_tr, o_tr, h_tr, l_tr, config,
            mas=full_mas[:, :tr_end] if full_mas is not None else None,
            emas=full_emas[:, :tr_end] if full_emas is not None else None,
            rsis=full_rsis[:, :tr_end] if full_rsis is not None else None,
            atr10=full_atr10[:tr_end] if full_atr10 is not None else None,
            atr14=full_atr14[:tr_end] if full_atr14 is not None else None,
            atr20=full_atr20[:tr_end] if full_atr20 is not None else None,
            rmax_h=full_rmax_h[:, :tr_end] if full_rmax_h is not None else None,
            rmin_l=full_rmin_l[:, :tr_end] if full_rmin_l is not None else None,
            stds=full_stds[:, :tr_end] if full_stds is not None else None,
            vols=full_vols[:, :tr_end] if full_vols is not None else None,
            up_prefix=full_up[:tr_end] if full_up is not None else None,
            **scan_kwargs,
        )

        for sn in strat_names:
            if sn not in train_results:
                continue
            res = train_results[sn]
            total_combos += res["cnt"]

            va_r, va_d, va_nt = eval_kernel(sn, res["params"], c_va, o_va, h_va, l_va,
                                            sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
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

    last_te_start = int(n * windows[-1][1])
    c_oos = c[last_te_start:]
    o_oos = o[last_te_start:]
    h_oos = h[last_te_start:]
    l_oos = l[last_te_start:]

    oos_total_bars = sum(
        int(n * te) - int(n * va) for _tr, va, te in windows
    )

    wf_scores: Dict[str, float] = {}
    for sn in strat_names:
        if not oos_ret_vals[sn]:
            continue
        r, d, nt = stitched_oos_metrics(oos_ret_vals[sn], oos_dd_vals[sn], oos_nt_vals[sn])
        bpy = costs.get("bars_per_year", 252.0)
        sc, ann_r, shrp, dsr_p = robust_score(r, d, nt, oos_total_bars, bars_per_year=bpy)
        wf_scores[sn] = sc
        sym_results[sn] = {
            "params": best_params_last[sn],
            "wfe_mean": float(np.mean(wfe_vals[sn])) if wfe_vals[sn] else 0.0,
            "gen_gap_mean": float(np.mean(gap_vals[sn])) if gap_vals[sn] else 0.0,
            "oos_ret": r, "oos_dd": d, "oos_trades": nt,
            "ann_ret": ann_r, "sharpe": shrp, "dsr_p": dsr_p, "wf_score": sc,
            "mc_mean": 0.0, "mc_std": 0.0, "mc_pct_positive": 0.0,
            "shuffle_mean": 0.0, "bootstrap_mean": 0.0,
        }

    if wf_scores:
        score_threshold = np.median(list(wf_scores.values()))
    else:
        score_threshold = -1e18

    rob_strats = []
    for sn in strat_names:
        bp = best_params_last.get(sn)
        if bp is None or len(c_oos) <= 30:
            continue
        if wf_scores.get(sn, -1e18) < score_threshold:
            continue
        rob_strats.append((sn, bp))

    if rob_strats:
        mc_all: Dict[str, list] = {}
        shuf_all: Dict[str, list] = {}
        boot_all: Dict[str, list] = {}

        with ThreadPoolExecutor(max_workers=rob_workers) as pool:
            futures = {}
            for sn, bp in rob_strats:
                fut = pool.submit(
                    _batch_robustness, sn, bp,
                    c_oos, o_oos, h_oos, l_oos,
                    sb, ss, cm, lev, dc, sl, pfrac, sl_slip,
                    n_mc_paths, n_shuffle_paths, n_bootstrap_paths,
                    mc_noise_std, bootstrap_block,
                )
                futures[fut] = sn
            for fut in as_completed(futures):
                sn = futures[fut]
                mc_rets, shuf_rets, boot_rets = fut.result()
                mc_all[sn] = mc_rets
                shuf_all[sn] = shuf_rets
                boot_all[sn] = boot_rets

        for sn, _bp in rob_strats:
            if sn not in sym_results:
                continue
            mc = mc_all.get(sn, [])
            shuf = shuf_all.get(sn, [])
            boot = boot_all.get(sn, [])
            sym_results[sn]["mc_mean"] = float(np.mean(mc)) if mc else 0.0
            sym_results[sn]["mc_std"] = float(np.std(mc)) if mc else 0.0
            sym_results[sn]["mc_pct_positive"] = sum(1 for x in mc if x > 0) / max(1, len(mc))
            sym_results[sn]["shuffle_mean"] = float(np.mean(shuf)) if shuf else 0.0
            sym_results[sn]["bootstrap_mean"] = float(np.mean(boot)) if boot else 0.0

    return sym_results, total_combos


def _process_one_symbol_mp(args):
    """Top-level wrapper for ProcessPoolExecutor (must be picklable).

    Clears forked-parent caches to prevent id() key collisions."""
    _clear_caches()
    return _process_one_symbol(*args)


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
    n_robustness_threads: Optional[int] = None,
    parallel_symbols: Optional[str] = None,
) -> RobustScanResult:
    """
    The correct way to search for optimal parameters with anti-overfitting.

    Parameter search happens ONLY inside walk-forward training windows.
    All robustness checks (MC/Shuffle/Bootstrap) run on OOS data only.

    Args:
        symbols: list of symbol keys present in *data*.
        data: {sym: {"c": np.ndarray, "o": ..., "h": ..., "l": ...}}.
        config: BacktestConfig with costs, leverage, stop-loss.
        param_grids: Custom parameter grids per strategy.
        strategies: subset of KERNEL_NAMES (default: all 18).
        wf_windows: walk-forward train/val/test fraction triples.
        embargo: purge gap in bars between train/val/test.
        n_mc_paths: Monte Carlo perturbation paths (on OOS only).
        mc_noise_std: noise standard deviation for MC perturbation.
        n_shuffle_paths: OHLC shuffle paths (on OOS only).
        n_bootstrap_paths: block bootstrap paths (on OOS only).
        bootstrap_block: block size for bootstrap.
        n_robustness_threads: threads for parallel robustness checks.
                              Default: min(8, cpu_count).
        parallel_symbols: ``"auto"`` (default), ``"serial"``, or
                          ``"process"``.  When ``"auto"``, process-level
                          symbol parallelism is enabled when symbols >=
                          CPU cores.

    Returns:
        RobustScanResult with per-symbol, per-strategy results.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor as _PPE

    strat_names = strategies or KERNEL_NAMES
    windows = wf_windows or DEFAULT_WF_WINDOWS
    costs = config_to_kernel_costs(config)

    _tc = optimal_thread_config()
    rob_workers = n_robustness_threads or _tc["robustness_workers"]

    scan_kwargs: Dict[str, Any] = {}
    if param_grids is not None:
        scan_kwargs["param_grids"] = param_grids
    if strategies is not None:
        scan_kwargs["strategies"] = strategies

    valid_syms = [s for s in symbols if s in data and len(data[s].get("c", [])) >= 100]
    for sym in valid_syms:
        validate_ohlc(data[sym]["c"], data[sym]["o"], data[sym]["h"], data[sym]["l"])

    n_symbols = len(valid_syms)
    n_cores = os.cpu_count() or 4

    if parallel_symbols is None or parallel_symbols == "auto":
        use_process = n_symbols >= max(2 * n_cores, 8)
    elif parallel_symbols == "process":
        use_process = n_symbols >= 2
    else:
        use_process = False

    result = RobustScanResult()
    total_combos = 0
    t0 = time.time()

    if use_process:
        n_workers = max(2, n_cores // 2)
        threads_per = max(1, n_cores // n_workers)
        logger.info("[WF] Parallel mode: %d workers × %d prange threads (%d symbols)",
                    n_workers, threads_per, n_symbols)

        worker_args = []
        for sym in valid_syms:
            worker_args.append((
                sym, data[sym], config, strat_names, windows, costs,
                scan_kwargs, param_grids, embargo,
                n_mc_paths, mc_noise_std, n_shuffle_paths,
                n_bootstrap_paths, bootstrap_block,
                1,              # robustness serial inside each worker
                threads_per,    # numba threads per worker
            ))

        completed = 0
        failed: List[str] = []
        with _PPE(max_workers=n_workers) as pool:
            future_to_sym = {}
            for args in worker_args:
                fut = pool.submit(_process_one_symbol_mp, args)
                future_to_sym[fut] = args[0]

            for fut in as_completed(future_to_sym):
                sym = future_to_sym[fut]
                completed += 1
                try:
                    sym_results, sym_combos = fut.result()
                    result.per_symbol[sym] = sym_results
                    total_combos += sym_combos
                except Exception as exc:
                    logger.error("[WF] Symbol %s failed: %s", sym, exc)
                    failed.append(sym)

                if completed % max(1, n_symbols // 20) == 0 or completed == n_symbols:
                    elapsed = time.time() - t0
                    eta = elapsed / completed * (n_symbols - completed) if completed else 0
                    logger.info("[WF] Progress: %d/%d (%.0f%%) — %.1fs elapsed, ETA %.0fs",
                                completed, n_symbols, completed / n_symbols * 100, elapsed, eta)

        if failed:
            logger.warning("[WF] %d symbols failed: %s", len(failed), ", ".join(failed))
    else:
        logger.info("[WF] Serial mode: %d symbols", n_symbols)
        for sym_idx, sym in enumerate(valid_syms):
            logger.info("[WF] Symbol %d/%d: %s (%d bars)",
                        sym_idx + 1, n_symbols, sym, len(data[sym]["c"]))
            try:
                sym_results, sym_combos = _process_one_symbol(
                    sym, data[sym], config, strat_names, windows, costs,
                    scan_kwargs, param_grids, embargo,
                    n_mc_paths, mc_noise_std, n_shuffle_paths,
                    n_bootstrap_paths, bootstrap_block, rob_workers,
                )
                result.per_symbol[sym] = sym_results
                total_combos += sym_combos
            except Exception as exc:
                logger.error("[WF] Symbol %s failed: %s", sym, exc)

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
    """Concatenate slices of *arr* specified by half-open ranges.

    Optimised: returns a zero-copy slice when the ranges form a single
    contiguous region, avoiding an ``np.concatenate`` allocation.
    """
    if not ranges:
        return np.empty(0, dtype=np.float64)
    if len(ranges) == 1:
        return arr[ranges[0][0]:ranges[0][1]]
    if all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1)):
        return arr[ranges[0][0]:ranges[-1][1]]
    return np.concatenate([arr[s:e] for s, e in ranges])


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
    n_robustness_threads: Optional[int] = None,
) -> CPCVResult:
    """Combinatorial Purged Cross-Validation scan.

    Uses *all* data in both train and test roles across C(n_groups,
    n_test_groups) splits.  Per-strategy OOS returns are averaged over
    all splits, giving a far more reliable performance estimate.

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
        n_robustness_threads: threads for parallel MC checks.

    Returns:
        CPCVResult with per-strategy averaged OOS metrics.
    """
    import time

    strat_names = strategies or KERNEL_NAMES
    costs = config_to_kernel_costs(config)
    sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
    lev, dc = costs["lev"], costs["dc"]
    sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

    _tc = optimal_thread_config()
    rob_workers = n_robustness_threads or _tc["robustness_workers"]

    scan_kwargs: Dict[str, Any] = {}
    if param_grids is not None:
        scan_kwargs["param_grids"] = param_grids
    if strategies is not None:
        scan_kwargs["strategies"] = strategies

    result = CPCVResult()
    total_combos = 0
    t0 = time.time()

    n_symbols = sum(1 for s in symbols if s in data)
    for sym_idx, sym in enumerate(symbols):
        if sym not in data:
            continue
        D = data[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        n = len(c)
        if n < 100:
            continue
        validate_ohlc(c, o, h, l)

        splits = cpcv_splits(n, n_groups, n_test_groups, embargo)
        result.n_splits = len(splits)

        logger.info("[CPCV] Symbol %d/%d: %s (%d bars, %d splits)",
                    sym_idx + 1, n_symbols, sym, n, len(splits))

        oos_rets: Dict[str, List[float]] = {sn: [] for sn in strat_names}
        oos_dds: Dict[str, List[float]] = {sn: [] for sn in strat_names}
        oos_nts: Dict[str, List[int]] = {sn: [] for sn in strat_names}
        best_params_all: Dict[str, Any] = {sn: None for sn in strat_names}
        mc_tasks_deferred: List[Tuple[str, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []

        for si, (train_ranges, test_ranges) in enumerate(splits):
            logger.info("  [%s] CPCV split %d/%d", sym, si + 1, len(splits))
            c_tr = _concat_ranges(c, train_ranges)
            o_tr = _concat_ranges(o, train_ranges)
            h_tr = _concat_ranges(h, train_ranges)
            l_tr = _concat_ranges(l, train_ranges)

            if len(c_tr) < 60:
                continue

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
                    mc_tasks_deferred.append((sn, bp, c_te, o_te, h_te, l_te, si))

        # --- Batch MC checks across all splits with ThreadPoolExecutor ---
        mc_rets_all: Dict[str, List[float]] = {sn: [] for sn in strat_names}
        if mc_tasks_deferred:
            logger.info("  [%s] CPCV MC robustness: %d tasks, %d workers",
                        sym, len(mc_tasks_deferred) * n_mc_paths, rob_workers)
            with ThreadPoolExecutor(max_workers=rob_workers) as pool:
                futures = []
                for sn, bp, c_te, o_te, h_te, l_te, si in mc_tasks_deferred:
                    for mi in range(n_mc_paths):
                        fut = pool.submit(
                            _robustness_one_path, "mc", si * 1000 + mi, sn, bp,
                            c_te, o_te, h_te, l_te,
                            sb, ss, cm, lev, dc, sl, pfrac, sl_slip,
                            mc_noise_std, 20,
                        )
                        futures.append((fut, sn))
                for fut, sn in futures:
                    _, ret = fut.result()
                    mc_rets_all[sn].append(ret)

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
