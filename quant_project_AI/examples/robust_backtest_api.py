#!/usr/bin/env python3
"""
==========================================================================
  Robust Backtest API  —  Reusable 10-Layer Anti-Overfitting System
==========================================================================
Wraps the full walk_forward_robust_scan.py machinery into a clean,
importable API.  Future backtests can use this with just a few lines:

    from robust_backtest_api import run_robust_pipeline, RobustConfig, quick_scan

    # Full 10-layer scan
    results = run_robust_pipeline(
        symbols=["AAPL", "BTC"],
        strategies=["MA", "MACD", "MomBreak"],
    )

    # Quick single-strategy scan
    result = quick_scan("MomBreak", symbols=["BTC"])

    # Print report
    print(results["report"])
"""
from __future__ import annotations

import os
import sys
import time
import math
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from walk_forward_robust_scan import (
    _ema, _rolling_mean, _rolling_std, _atr, _rsi_wilder,
    _fx, _mtm, _score,
    bt_ma_wf, bt_rsi_wf, bt_macd_wf, bt_drift_wf, bt_ramom_wf,
    bt_turtle_wf, bt_bollinger_wf, bt_keltner_wf, bt_multifactor_wf,
    bt_volregime_wf, bt_connors_wf, bt_mesa_wf, bt_kama_wf,
    bt_donchian_wf, bt_zscore_wf, bt_mombreak_wf, bt_regime_ema_wf,
    bt_tfiltrsi_wf, bt_mombrkplus_wf, bt_dualmom_wf, bt_consensus_wf,
    perturb_ohlc, shuffle_ohlc, block_bootstrap_ohlc,
    precompute_all_ma, precompute_all_ema, precompute_all_rsi,
    scan_all, eval_strategy, eval_strategy_mc,
    STRAT_NAMES, PARAM_TYPES,
)
from scipy import stats as sp_stats


# =====================================================================
#  Configuration
# =====================================================================

@dataclass
class RobustConfig:
    """Configuration for the 10-layer robust backtest pipeline."""
    # Walk-Forward
    wf_windows: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0.35, 0.45, 0.55),
        (0.45, 0.55, 0.65),
        (0.55, 0.65, 0.75),
        (0.65, 0.75, 0.85),
        (0.75, 0.85, 0.95),
        (0.80, 0.90, 1.00),
    ])
    embargo: int = 5
    min_trades: int = 20

    # Cost model
    slippage_buy: float = 1.0005   # 5 bps
    slippage_sell: float = 0.9995
    commission: float = 0.0015     # 15 bps

    # Parameter stability
    perturb_factors: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.1, 1.2])

    # Monte Carlo
    mc_paths: int = 30
    mc_noise_std: float = 0.002

    # OHLC Shuffle
    shuffle_paths: int = 20

    # Block Bootstrap
    bootstrap_paths: int = 20
    bootstrap_block: int = 20

    # Data
    data_dir: str = ""

    # Output
    output_dir: str = ""
    generate_report: bool = True
    generate_csv: bool = True

    def __post_init__(self):
        if not self.data_dir:
            self.data_dir = os.path.join(_HERE, "..", "data")
        if not self.output_dir:
            self.output_dir = os.path.join(_HERE, "..", "results", "robust_scan")

    @property
    def sb(self): return self.slippage_buy
    @property
    def ss(self): return self.slippage_sell
    @property
    def cm(self): return self.commission


# =====================================================================
#  Deflated Sharpe
# =====================================================================

def deflated_sharpe(sharpe_obs, n_trials, n_bars, skew=0.0, kurtosis=3.0):
    if n_bars < 3 or n_trials < 1:
        return 0.0
    e_max_sr = ((1.0 - 0.5772) * (2.0 * math.log(n_trials))**0.5
                + 0.5772 * (2.0 * math.log(n_trials))**-0.5)
    se_sr = ((1.0 - skew * sharpe_obs + (kurtosis - 1.0) / 4.0 * sharpe_obs**2)
             / max(1.0, n_bars - 1.0))**0.5
    if se_sr < 1e-12:
        return 1.0 if sharpe_obs > e_max_sr else 0.0
    z = (sharpe_obs - e_max_sr) / se_sr
    return float(sp_stats.norm.cdf(z))


# =====================================================================
#  JIT Warm-Up
# =====================================================================

_WARMED_UP = False

def _jit_warmup(cfg: RobustConfig):
    global _WARMED_UP
    if _WARMED_UP:
        return
    dc = np.random.rand(200).astype(np.float64) * 100 + 100
    dh = dc + 2; dl = dc - 2; do_ = dc + 0.5
    dm = np.random.rand(200).astype(np.float64) * 100 + 100
    dr = np.random.rand(200).astype(np.float64) * 100
    sb, ss, cm = cfg.sb, cfg.ss, cfg.cm
    bt_ma_wf(dc, do_, dm, dm, sb, ss, cm)
    bt_rsi_wf(dc, do_, dr, 30.0, 70.0, sb, ss, cm)
    bt_macd_wf(dc, do_, dm, dm, 9, sb, ss, cm)
    bt_drift_wf(dc, do_, 20, 0.6, 5, sb, ss, cm)
    bt_ramom_wf(dc, do_, 10, 10, 2.0, 0.5, sb, ss, cm)
    bt_turtle_wf(dc, do_, dh, dl, 10, 5, 14, 2.0, sb, ss, cm)
    bt_bollinger_wf(dc, do_, 20, 2.0, sb, ss, cm)
    bt_keltner_wf(dc, do_, dh, dl, 20, 14, 2.0, sb, ss, cm)
    bt_multifactor_wf(dc, do_, 14, 20, 20, 0.6, 0.35, sb, ss, cm)
    bt_volregime_wf(dc, do_, dh, dl, 14, 0.02, 5, 20, 14, 30, 70, sb, ss, cm)
    bt_connors_wf(dc, do_, 2, 50, 5, 10.0, 90.0, sb, ss, cm)
    bt_mesa_wf(dc, do_, 0.5, 0.05, sb, ss, cm)
    bt_kama_wf(dc, do_, dh, dl, 10, 2, 30, 2.0, 14, sb, ss, cm)
    bt_donchian_wf(dc, do_, dh, dl, 20, 14, 2.0, sb, ss, cm)
    bt_zscore_wf(dc, do_, 20, 2.0, 0.5, 4.0, sb, ss, cm)
    bt_mombreak_wf(dc, do_, dh, dl, 50, 0.02, 14, 2.0, sb, ss, cm)
    bt_regime_ema_wf(dc, do_, dh, dl, 14, 0.02, 5, 20, 50, sb, ss, cm)
    bt_tfiltrsi_wf(dc, do_, dr, dm, 30.0, 70.0, sb, ss, cm)
    bt_mombrkplus_wf(dc, do_, dh, dl, 20, 0.02, 14, 2.0, 3, sb, ss, cm)
    bt_dualmom_wf(dc, do_, 10, 50, sb, ss, cm)
    bt_consensus_wf(dc, do_, dm, dm, dr, 20, 30.0, 70.0, 2, sb, ss, cm)
    perturb_ohlc(dc, do_, dh, dl, 0.002, 42)
    shuffle_ohlc(dc, do_, dh, dl, 42)
    block_bootstrap_ohlc(dc, do_, dh, dl, 20, 42)
    _WARMED_UP = True


# =====================================================================
#  Data Loading
# =====================================================================

def load_data(symbols: List[str], data_dir: str) -> Dict[str, Dict]:
    """Load OHLC data from CSV files.

    Expected format: CSV with columns [date, open, high, low, close].
    """
    datasets = {}
    for sym in symbols:
        fp = os.path.join(data_dir, f"{sym}.csv")
        if not os.path.exists(fp):
            print(f"  WARNING: {fp} not found, skipping {sym}")
            continue
        df = pd.read_csv(fp, parse_dates=["date"])
        datasets[sym] = {
            "c": df["close"].values.astype(np.float64),
            "o": df["open"].values.astype(np.float64),
            "h": df["high"].values.astype(np.float64),
            "l": df["low"].values.astype(np.float64),
            "n": len(df),
        }
    return datasets


def download_data(symbols: List[str], data_dir: str,
                  start: str = "2020-01-01", end: str = "2026-02-01") -> Dict[str, Dict]:
    """Download data from yfinance and save to CSV. Returns loaded datasets."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        return {}

    os.makedirs(data_dir, exist_ok=True)
    yf_map = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}
    datasets = {}

    for sym in symbols:
        ticker = yf_map.get(sym, sym)
        fp = os.path.join(data_dir, f"{sym}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp, parse_dates=["date"])
            if len(df) > 100:
                datasets[sym] = {
                    "c": df["close"].values.astype(np.float64),
                    "o": df["open"].values.astype(np.float64),
                    "h": df["high"].values.astype(np.float64),
                    "l": df["low"].values.astype(np.float64),
                    "n": len(df),
                }
                continue

        print(f"  Downloading {ticker} ...", end=" ", flush=True)
        raw = yf.download(ticker, start=start, end=end, progress=False)
        if raw.empty:
            print("EMPTY")
            continue
        df = raw.reset_index()
        col_map = {}
        for c in df.columns:
            cl = str(c).lower() if isinstance(c, str) else str(c[-1]).lower() if isinstance(c, tuple) else str(c).lower()
            if "date" in cl: col_map[c] = "date"
            elif "open" in cl: col_map[c] = "open"
            elif "high" in cl: col_map[c] = "high"
            elif "low" in cl: col_map[c] = "low"
            elif "close" in cl and "adj" not in cl: col_map[c] = "close"
        df = df.rename(columns=col_map)
        df = df[["date", "open", "high", "low", "close"]].dropna()
        df.to_csv(fp, index=False)
        datasets[sym] = {
            "c": df["close"].values.astype(np.float64),
            "o": df["open"].values.astype(np.float64),
            "h": df["high"].values.astype(np.float64),
            "l": df["low"].values.astype(np.float64),
            "n": len(df),
        }
        print(f"{len(df)} bars")

    return datasets


# =====================================================================
#  Verdict Logic
# =====================================================================

def _verdict(avg_wfe, avg_gen_gap, avg_stability, avg_mc,
             avg_shuffle, avg_bootstrap, avg_dsr, avg_cross):
    checks = 0
    if avg_wfe > 0: checks += 1
    if avg_gen_gap < 5.0: checks += 1
    if avg_stability > 0.5: checks += 1
    if avg_mc > 0.5: checks += 1
    if avg_shuffle > 0.3: checks += 1
    if avg_bootstrap > 0.3: checks += 1
    if avg_dsr > 0.3: checks += 1
    if avg_cross > 0: checks += 1
    if checks >= 7: return "ROBUST"
    if checks >= 5: return "STRONG"
    if checks >= 3: return "MODERATE"
    return "WEAK"


# =====================================================================
#  Core Pipeline
# =====================================================================

def run_robust_pipeline(
    symbols: List[str],
    strategies: Optional[List[str]] = None,
    config: Optional[RobustConfig] = None,
    datasets: Optional[Dict[str, Dict]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full 10-layer anti-overfitting robust scan.

    Args:
        symbols: List of symbol names (e.g., ["AAPL", "BTC"]).
        strategies: Which strategies to test. None = all 21 strategies.
        config: RobustConfig instance. None = default config.
        datasets: Pre-loaded data dict. If None, loads from config.data_dir.
        verbose: Print progress.

    Returns:
        Dict with keys:
            "ranking": List of (rank, strategy, composite_score, verdict)
            "best_params": {symbol: {strategy: params_tuple}}
            "metrics": {strategy: {wfe, gen_gap, stability, mc, shuffle, bootstrap, dsr, cross}}
            "wf_detail": {symbol: {strategy: [window_results]}}
            "report": str (markdown report)
            "csv_path": str (path to summary CSV)
            "elapsed": float (seconds)
            "total_combos": int
    """
    if config is None:
        config = RobustConfig()

    strat_list = strategies if strategies else list(STRAT_NAMES)
    for s in strat_list:
        if s not in STRAT_NAMES:
            raise ValueError(f"Unknown strategy '{s}'. Valid: {STRAT_NAMES}")

    sb, ss, cm = config.sb, config.ss, config.cm

    # ---- JIT warm-up ----
    if verbose:
        print("=" * 80)
        print("  10-Layer Anti-Overfitting Robust Scan")
        print("=" * 80)
        print("\n[0] JIT warm-up ...", end=" ", flush=True)

    t0 = time.time()
    _jit_warmup(config)
    if verbose:
        print(f"done ({time.time()-t0:.1f}s)")

    # ---- Load data ----
    if verbose:
        print("[1] Loading data ...", flush=True)
    if datasets is None:
        datasets = load_data(symbols, config.data_dir)
    valid_symbols = [s for s in symbols if s in datasets]
    if not valid_symbols:
        raise ValueError(f"No data found for any of {symbols} in {config.data_dir}")
    if verbose:
        for sym in valid_symbols:
            print(f"  {sym}: {datasets[sym]['n']} bars")

    n_windows = len(config.wf_windows)
    grand_t0 = time.time()

    # ---- Phase 2: Purged Walk-Forward ----
    if verbose:
        print(f"\n[2] Purged Walk-Forward ({n_windows} windows x {len(valid_symbols)} symbols) ...", flush=True)

    wf = {sym: {sn: [] for sn in strat_list} for sym in valid_symbols}
    best_params = {sym: {sn: None for sn in strat_list} for sym in valid_symbols}
    combo_counts = {sn: 0 for sn in strat_list}
    total_combos = 0

    for sym in valid_symbols:
        D = datasets[sym]
        c, o, h, l, n = D["c"], D["o"], D["h"], D["l"], D["n"]

        for wi, (tr_pct, va_pct, te_pct) in enumerate(config.wf_windows):
            tr_end = int(n * tr_pct)
            va_start = min(tr_end + config.embargo, int(n * va_pct))
            va_end = int(n * va_pct)
            te_end = min(int(n * te_pct), n)
            if va_end > te_end: va_end = te_end
            if va_start >= va_end: va_start = va_end

            c_tr, o_tr = c[:tr_end], o[:tr_end]
            h_tr, l_tr = h[:tr_end], l[:tr_end]
            c_va, o_va = c[va_start:va_end], o[va_start:va_end]
            h_va, l_va = h[va_start:va_end], l[va_start:va_end]
            c_te, o_te = c[va_end:te_end], o[va_end:te_end]
            h_te, l_te = h[va_end:te_end], l[va_end:te_end]

            mas_tr = precompute_all_ma(c_tr, 200)
            emas_tr = precompute_all_ema(c_tr, 200)
            rsis_tr = precompute_all_rsi(c_tr, 200)

            all_results = scan_all(c_tr, o_tr, h_tr, l_tr, mas_tr, emas_tr, rsis_tr, sb, ss, cm)

            mas_va = precompute_all_ma(c_va, 200)
            emas_va = precompute_all_ema(c_va, 200)
            rsis_va = precompute_all_rsi(c_va, 200)
            mas_te = precompute_all_ma(c_te, 200)
            emas_te = precompute_all_ema(c_te, 200)
            rsis_te = precompute_all_rsi(c_te, 200)

            w_combos = 0
            for sn in strat_list:
                res = all_results[sn]
                w_combos += res["cnt"]
                combo_counts[sn] = max(combo_counts[sn], res["cnt"])

                va_ret, va_dd, va_nt = eval_strategy(
                    sn, res["params"], c_va, o_va, h_va, l_va,
                    mas_va, emas_va, rsis_va, sb, ss, cm
                )
                te_ret, te_dd, te_nt = eval_strategy(
                    sn, res["params"], c_te, o_te, h_te, l_te,
                    mas_te, emas_te, rsis_te, sb, ss, cm
                )
                wf[sym][sn].append({
                    "params": res["params"],
                    "train_score": res["score"],
                    "train_ret": res["ret"], "train_dd": res["dd"], "train_nt": res["nt"],
                    "val_ret": va_ret, "val_dd": va_dd, "val_nt": va_nt,
                    "test_ret": te_ret, "test_dd": te_dd, "test_nt": te_nt,
                    "gen_gap": va_ret - te_ret,
                })
                if wi == n_windows - 1:
                    best_params[sym][sn] = res["params"]

            total_combos += w_combos

        if verbose:
            print(f"  {sym}: done", flush=True)

    wfe = {sym: {} for sym in valid_symbols}
    gen_gap = {sym: {} for sym in valid_symbols}
    for sym in valid_symbols:
        for sn in strat_list:
            oos_rets = [w["test_ret"] for w in wf[sym][sn]]
            wfe[sym][sn] = np.mean(oos_rets) if oos_rets else 0.0
            gaps = [abs(w["gen_gap"]) for w in wf[sym][sn]]
            gen_gap[sym][sn] = np.mean(gaps) if gaps else 99.0

    # ---- Phase 3: Parameter Stability ----
    if verbose:
        print(f"[3] Parameter Stability ...", flush=True)

    stability = {sym: {} for sym in valid_symbols}
    for sym in valid_symbols:
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        mas = precompute_all_ma(c, 200)
        emas = precompute_all_ema(c, 200)
        rsis = precompute_all_rsi(c, 200)

        for sn in strat_list:
            params = best_params[sym][sn]
            if params is None:
                stability[sym][sn] = 0.0
                continue
            returns = []
            base_r, _, _ = eval_strategy(sn, params, c, o, h, l, mas, emas, rsis, sb, ss, cm)
            returns.append(base_r)
            ptypes = PARAM_TYPES.get(sn, [])
            for pi in range(len(params)):
                for factor in config.perturb_factors:
                    pv = params[pi] * factor
                    if pi < len(ptypes) and ptypes[pi] == int:
                        pv = max(1, int(round(pv)))
                    new_p = list(params)
                    new_p[pi] = pv
                    r, _, _ = eval_strategy(sn, tuple(new_p), c, o, h, l, mas, emas, rsis, sb, ss, cm)
                    returns.append(r)
            mean_r = np.mean(returns)
            std_r = np.std(returns)
            stab = 1.0 - std_r / abs(mean_r) if abs(mean_r) > 1e-8 else 0.0
            stability[sym][sn] = max(0.0, min(1.0, stab))

    # ---- Phase 4: Monte Carlo ----
    if verbose:
        print(f"[4] Monte Carlo ({config.mc_paths} paths) ...", flush=True)

    mc_results = {sym: {} for sym in valid_symbols}
    for sym in valid_symbols:
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        for sn in strat_list:
            params = best_params[sym][sn]
            if params is None:
                mc_results[sym][sn] = {"profitable": 0.0, "stability": 0.0}
                continue
            mc_rets = []
            for seed in range(config.mc_paths):
                cp, op, hp, lp = perturb_ohlc(c, o, h, l, config.mc_noise_std, seed + 1000)
                r, _, _ = eval_strategy_mc(sn, params, cp, op, hp, lp, sb, ss, cm)
                mc_rets.append(r)
            mc_arr = np.array(mc_rets)
            mc_mean = np.mean(mc_arr)
            mc_std = np.std(mc_arr)
            mc_prof = float(np.sum(mc_arr > 0)) / len(mc_arr)
            mc_stab = max(0.0, min(1.0, 1.0 - mc_std / abs(mc_mean))) if abs(mc_mean) > 1e-8 else 0.0
            mc_results[sym][sn] = {"profitable": mc_prof, "stability": mc_stab}

    # ---- Phase 5: OHLC Shuffle ----
    if verbose:
        print(f"[5] OHLC Shuffle ({config.shuffle_paths} paths) ...", flush=True)

    shuffle_results = {sym: {} for sym in valid_symbols}
    for sym in valid_symbols:
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        for sn in strat_list:
            params = best_params[sym][sn]
            if params is None:
                shuffle_results[sym][sn] = {"profitable": 0.0, "stability": 0.0}
                continue
            sh_rets = []
            for seed in range(config.shuffle_paths):
                cp, op, hp, lp = shuffle_ohlc(c, o, h, l, seed + 5000)
                r, _, _ = eval_strategy_mc(sn, params, cp, op, hp, lp, sb, ss, cm)
                sh_rets.append(r)
            sh_arr = np.array(sh_rets)
            sh_mean = np.mean(sh_arr)
            sh_std = np.std(sh_arr)
            sh_prof = float(np.sum(sh_arr > 0)) / len(sh_arr)
            sh_stab = max(0.0, min(1.0, 1.0 - sh_std / abs(sh_mean))) if abs(sh_mean) > 1e-8 else 0.0
            shuffle_results[sym][sn] = {"profitable": sh_prof, "stability": sh_stab}

    # ---- Phase 6: Block Bootstrap ----
    if verbose:
        print(f"[6] Block Bootstrap ({config.bootstrap_paths} paths) ...", flush=True)

    bootstrap_results = {sym: {} for sym in valid_symbols}
    for sym in valid_symbols:
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        for sn in strat_list:
            params = best_params[sym][sn]
            if params is None:
                bootstrap_results[sym][sn] = {"profitable": 0.0, "stability": 0.0}
                continue
            bs_rets = []
            for seed in range(config.bootstrap_paths):
                cp, op, hp, lp = block_bootstrap_ohlc(c, o, h, l, config.bootstrap_block, seed + 9000)
                r, _, _ = eval_strategy_mc(sn, params, cp, op, hp, lp, sb, ss, cm)
                bs_rets.append(r)
            bs_arr = np.array(bs_rets)
            bs_mean = np.mean(bs_arr)
            bs_std = np.std(bs_arr)
            bs_prof = float(np.sum(bs_arr > 0)) / len(bs_arr)
            bs_stab = max(0.0, min(1.0, 1.0 - bs_std / abs(bs_mean))) if abs(bs_mean) > 1e-8 else 0.0
            bootstrap_results[sym][sn] = {"profitable": bs_prof, "stability": bs_stab}

    # ---- Phase 7: Deflated Sharpe ----
    if verbose:
        print(f"[7] Deflated Sharpe Ratio ...", flush=True)

    dsr_scores = {sym: {} for sym in valid_symbols}
    for sym in valid_symbols:
        n_bars = datasets[sym]["n"]
        for sn in strat_list:
            oos_rets = [w["test_ret"] for w in wf[sym][sn]]
            if not oos_rets or all(r == 0.0 for r in oos_rets):
                dsr_scores[sym][sn] = 0.0
                continue
            ret_arr = np.array(oos_rets) / 100.0
            mu = np.mean(ret_arr); sd = np.std(ret_arr)
            sharpe = mu / sd if sd > 1e-12 else 0.0
            skew = float(sp_stats.skew(ret_arr)) if len(ret_arr) > 2 else 0.0
            kurt = float(sp_stats.kurtosis(ret_arr, fisher=False)) if len(ret_arr) > 3 else 3.0
            n_trials = combo_counts.get(sn, 1000)
            dsr_scores[sym][sn] = deflated_sharpe(sharpe, n_trials, n_bars, skew, kurt)

    # ---- Phase 8: Cross-Asset ----
    if verbose:
        print(f"[8] Cross-Asset Validation ...", flush=True)

    cross = {sn: {} for sn in strat_list}
    precomp_cache = {}
    for sym in valid_symbols:
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        precomp_cache[sym] = {
            "c": c, "o": o, "h": h, "l": l,
            "mas": precompute_all_ma(c, 200),
            "emas": precompute_all_ema(c, 200),
            "rsis": precompute_all_rsi(c, 200),
        }

    for sn in strat_list:
        cross[sn] = {}
        for train_sym in valid_symbols:
            params = best_params[train_sym][sn]
            cross[sn][train_sym] = {}
            for test_sym in valid_symbols:
                if test_sym == train_sym:
                    cross[sn][train_sym][test_sym] = None
                    continue
                pc = precomp_cache[test_sym]
                r, _, _ = eval_strategy(
                    sn, params, pc["c"], pc["o"], pc["h"], pc["l"],
                    pc["mas"], pc["emas"], pc["rsis"], sb, ss, cm
                )
                cross[sn][train_sym][test_sym] = r

    total_elapsed = time.time() - grand_t0

    # ---- Phase 9: Composite Ranking ----
    if verbose:
        print(f"\n[9] 8-dim Composite Ranking ...", flush=True)

    avg_wfe_d = {}; avg_gen_gap_d = {}; avg_stability_d = {}
    avg_mc_d = {}; avg_shuffle_d = {}; avg_bootstrap_d = {}
    avg_dsr_d = {}; avg_cross_d = {}

    for sn in strat_list:
        avg_wfe_d[sn] = np.mean([wfe[sym][sn] for sym in valid_symbols])
        avg_gen_gap_d[sn] = np.mean([gen_gap[sym][sn] for sym in valid_symbols])
        avg_stability_d[sn] = np.mean([stability[sym][sn] for sym in valid_symbols])
        mc_s = [mc_results[sym][sn]["profitable"] * max(0.0, mc_results[sym][sn]["stability"])
                for sym in valid_symbols]
        avg_mc_d[sn] = np.mean(mc_s)
        sh_s = [shuffle_results[sym][sn]["profitable"] * max(0.0, shuffle_results[sym][sn]["stability"])
                for sym in valid_symbols]
        avg_shuffle_d[sn] = np.mean(sh_s)
        bs_s = [bootstrap_results[sym][sn]["profitable"] * max(0.0, bootstrap_results[sym][sn]["stability"])
                for sym in valid_symbols]
        avg_bootstrap_d[sn] = np.mean(bs_s)
        avg_dsr_d[sn] = np.mean([dsr_scores[sym][sn] for sym in valid_symbols])
        cross_rets = []
        for train_sym in valid_symbols:
            for test_sym in valid_symbols:
                if test_sym == train_sym: continue
                v = cross[sn][train_sym][test_sym]
                if v is not None: cross_rets.append(v)
        avg_cross_d[sn] = np.mean(cross_rets) if cross_rets else 0.0

    wfe_ranked = sorted(strat_list, key=lambda s: avg_wfe_d[s], reverse=True)
    gap_ranked = sorted(strat_list, key=lambda s: avg_gen_gap_d[s])
    stab_ranked = sorted(strat_list, key=lambda s: avg_stability_d[s], reverse=True)
    mc_ranked = sorted(strat_list, key=lambda s: avg_mc_d[s], reverse=True)
    shuffle_ranked = sorted(strat_list, key=lambda s: avg_shuffle_d[s], reverse=True)
    bootstrap_ranked = sorted(strat_list, key=lambda s: avg_bootstrap_d[s], reverse=True)
    dsr_ranked = sorted(strat_list, key=lambda s: avg_dsr_d[s], reverse=True)
    cross_ranked = sorted(strat_list, key=lambda s: avg_cross_d[s], reverse=True)

    composite = {}
    for sn in strat_list:
        composite[sn] = (wfe_ranked.index(sn) + gap_ranked.index(sn) +
                         stab_ranked.index(sn) + mc_ranked.index(sn) +
                         shuffle_ranked.index(sn) + bootstrap_ranked.index(sn) +
                         dsr_ranked.index(sn) + cross_ranked.index(sn) + 8)

    final_ranked = sorted(strat_list, key=lambda s: composite[s])

    # Build result
    ranking = []
    for i, sn in enumerate(final_ranked):
        v = _verdict(avg_wfe_d[sn], avg_gen_gap_d[sn], avg_stability_d[sn],
                     avg_mc_d[sn], avg_shuffle_d[sn], avg_bootstrap_d[sn],
                     avg_dsr_d[sn], avg_cross_d[sn])
        ranking.append((i + 1, sn, composite[sn], v))

    metrics = {}
    for sn in strat_list:
        metrics[sn] = {
            "wfe": avg_wfe_d[sn], "gen_gap": avg_gen_gap_d[sn],
            "stability": avg_stability_d[sn], "mc": avg_mc_d[sn],
            "shuffle": avg_shuffle_d[sn], "bootstrap": avg_bootstrap_d[sn],
            "dsr": avg_dsr_d[sn], "cross": avg_cross_d[sn],
            "composite": composite[sn],
        }

    # ---- Phase 10: Report ----
    report_text = ""
    csv_path = ""

    if config.generate_report or config.generate_csv:
        os.makedirs(config.output_dir, exist_ok=True)

    if config.generate_report:
        if verbose:
            print(f"[10] Generating report ...", flush=True)

        L = []
        L.append(f"# 10-Layer Robust Scan Report ({len(strat_list)} Strategies x {len(valid_symbols)} Assets)\n")
        L.append(f"> {total_combos:,} backtests | {total_elapsed:.1f}s | {total_combos/max(1,total_elapsed):,.0f} combos/sec\n")

        L.append("## Composite Ranking\n")
        L.append("| Rank | Strategy | WFE | Gap | Stab | MC | Shuffle | Boot | DSR | Cross | Score | Verdict |")
        L.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")
        for rank, sn, score, verd in ranking:
            m = metrics[sn]
            L.append(f"| {rank} | **{sn}** | {m['wfe']:+.1f}% | {m['gen_gap']:.1f}% | "
                     f"{m['stability']:.3f} | {m['mc']:.3f} | {m['shuffle']:.3f} | "
                     f"{m['bootstrap']:.3f} | {m['dsr']:.3f} | {m['cross']:+.1f}% | {score} | {verd} |")
        L.append("")

        L.append("## Best Params per Symbol\n")
        for sym in valid_symbols:
            L.append(f"### {sym}\n")
            L.append("| Strategy | Params | Train | Val | Test | Verdict |")
            L.append("|:---|:---|:---:|:---:|:---:|:---|")
            for _, sn, _, verd in ranking:
                p = best_params[sym][sn]
                wlast = wf[sym][sn][-1] if wf[sym][sn] else {}
                L.append(f"| {sn} | {p} | {wlast.get('train_ret',0):+.1f}% | "
                         f"{wlast.get('val_ret',0):+.1f}% | {wlast.get('test_ret',0):+.1f}% | {verd} |")
            L.append("")

        report_text = "\n".join(L)
        rpt_path = os.path.join(config.output_dir, "robust_report.md")
        with open(rpt_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        if verbose:
            print(f"  Report: {rpt_path}")

    if config.generate_csv:
        csv_path = os.path.join(config.output_dir, "summary.csv")
        rows = []
        for sym in valid_symbols:
            for sn in strat_list:
                oos_rets = [w["test_ret"] for w in wf[sym][sn]]
                val_rets = [w["val_ret"] for w in wf[sym][sn]]
                row = {
                    "symbol": sym, "strategy": sn,
                    "wfe": round(wfe[sym][sn], 2),
                    "gen_gap": round(gen_gap[sym][sn], 2),
                    "stability": round(stability[sym][sn], 3),
                    "mc_score": round(mc_results[sym][sn]["profitable"] *
                                      max(0, mc_results[sym][sn]["stability"]), 3),
                    "shuffle_score": round(shuffle_results[sym][sn]["profitable"] *
                                          max(0, shuffle_results[sym][sn]["stability"]), 3),
                    "bootstrap_score": round(bootstrap_results[sym][sn]["profitable"] *
                                            max(0, bootstrap_results[sym][sn]["stability"]), 3),
                    "dsr": round(dsr_scores[sym][sn], 3),
                    "composite_rank": composite[sn],
                    "best_params": str(best_params[sym][sn]),
                }
                for wi in range(n_windows):
                    row[f"w{wi+1}_val"] = round(val_rets[wi], 2) if wi < len(val_rets) else 0
                    row[f"w{wi+1}_test"] = round(oos_rets[wi], 2) if wi < len(oos_rets) else 0
                rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        if verbose:
            print(f"  CSV: {csv_path}")

    if verbose:
        print(f"\n  Total: {total_combos:,} backtests in {total_elapsed:.1f}s")
        print("\n  RANKING:")
        for rank, sn, score, verd in ranking:
            print(f"    #{rank} {sn:>14}  score={score}  {verd}")

    return {
        "ranking": ranking,
        "best_params": best_params,
        "metrics": metrics,
        "wf_detail": wf,
        "report": report_text,
        "csv_path": csv_path,
        "elapsed": total_elapsed,
        "total_combos": total_combos,
    }


# =====================================================================
#  Convenience Functions
# =====================================================================

def quick_scan(
    strategy: str,
    symbols: Optional[List[str]] = None,
    config: Optional[RobustConfig] = None,
) -> Dict[str, Any]:
    """Run 10-layer scan for a single strategy. Returns same dict as run_robust_pipeline."""
    if symbols is None:
        symbols = ["AAPL", "GOOGL", "TSLA", "BTC", "SPY"]
    return run_robust_pipeline(symbols=symbols, strategies=[strategy], config=config)


def scan_top_strategies(
    symbols: Optional[List[str]] = None,
    config: Optional[RobustConfig] = None,
) -> Dict[str, Any]:
    """Scan only the top-3 strategies (MomBreak, MACD, MA) from V3 report."""
    if symbols is None:
        symbols = ["AAPL", "GOOGL", "TSLA", "BTC", "ETH", "SOL", "SPY", "AMZN"]
    return run_robust_pipeline(symbols=symbols, strategies=["MomBreak", "MACD", "MA"], config=config)


def compare_configs(
    configs: List[Tuple[str, RobustConfig]],
    symbols: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare multiple RobustConfig configurations side by side.

    Args:
        configs: List of (name, config) tuples.
        symbols: Symbols to test.
        strategies: Strategies to test.

    Returns:
        DataFrame with comparison.
    """
    if symbols is None:
        symbols = ["AAPL", "BTC", "SPY"]
    if strategies is None:
        strategies = ["MA", "MACD", "MomBreak"]

    rows = []
    for name, cfg in configs:
        print(f"\n{'='*40} Config: {name} {'='*40}")
        result = run_robust_pipeline(symbols=symbols, strategies=strategies,
                                     config=cfg, verbose=True)
        for rank, sn, score, verd in result["ranking"]:
            m = result["metrics"][sn]
            rows.append({
                "config": name, "strategy": sn, "rank": rank,
                "wfe": m["wfe"], "gen_gap": m["gen_gap"],
                "stability": m["stability"], "composite": score,
                "verdict": verd,
            })

    return pd.DataFrame(rows)


# =====================================================================
#  CLI
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="10-Layer Robust Backtest")
    parser.add_argument("--symbols", nargs="+",
                        default=["AAPL", "GOOGL", "TSLA", "BTC", "ETH", "SOL", "SPY", "AMZN"])
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Strategies to scan (default: all 21)")
    parser.add_argument("--mc-paths", type=int, default=30)
    parser.add_argument("--shuffle-paths", type=int, default=20)
    parser.add_argument("--bootstrap-paths", type=int, default=20)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-only", action="store_true",
                        help="Scan only top-3 strategies (MomBreak, MACD, MA)")
    args = parser.parse_args()

    cfg = RobustConfig(
        mc_paths=args.mc_paths,
        shuffle_paths=args.shuffle_paths,
        bootstrap_paths=args.bootstrap_paths,
    )
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir

    strats = ["MomBreak", "MACD", "MA"] if args.top_only else args.strategies

    result = run_robust_pipeline(
        symbols=args.symbols,
        strategies=strats,
        config=cfg,
    )
    print(f"\nDone. {result['total_combos']:,} backtests in {result['elapsed']:.1f}s")
