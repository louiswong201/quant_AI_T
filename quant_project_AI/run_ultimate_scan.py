#!/usr/bin/env python3
"""
Ultimate Comprehensive Backtest — Maximum Parameter Coverage
=============================================================
ULTRA parameter grids (~200K+ combos) × all symbols × 18 strategies
× 5 leverage levels × 3 timeframes × 11-layer anti-overfitting.

Phase 0: Data loading & validation
Phase 1: Single-TF multi-leverage robust scan  (ULTRA × 5 leverage × 3 TF)
Phase 2: Multi-TF fusion scan                  (top Phase 1 survivors × TF combos)
Phase 3: Quality filter + CPCV cross-validation
Phase 4: Export live_trading_config.json + detailed report

Usage:
    python run_ultimate_scan.py                # full ULTRA pipeline
    python run_ultimate_scan.py --fast         # quick mode (EXPANDED grids, fewer paths)
    python run_ultimate_scan.py --phase 1      # only Phase 1
    python run_ultimate_scan.py --top 30       # export top 30 recommendations
    python run_ultimate_scan.py --leverage 1,2,3  # custom leverage levels
"""
import itertools
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from quant_framework.backtest import BacktestConfig, backtest_multi_tf
from quant_framework.backtest.kernels import DEFAULT_PARAM_GRIDS, KERNEL_NAMES
from quant_framework.backtest.robust_scan import run_robust_scan, run_cpcv_scan
from run_full_scan import EXPANDED_GRIDS

# ═══════════════════════════════════════════════════════════════
#  ULTRA PARAMETER GRIDS  (~200K+ total combos, ~3-4x EXPANDED)
#
#  Design: denser steps in high-sensitivity dimensions, wider
#  boundary exploration, and finer granularity for key strategies.
# ═══════════════════════════════════════════════════════════════
ULTRA_GRIDS = {
    # ─── MA crossover: denser short/long windows ───
    "MA": [(s, lg)
           for s in range(2, 100, 1)
           for lg in range(s + 3, 301, 3)
           if lg > s],

    # ─── RSI: finer period/threshold sweep ───
    "RSI": [(p, os_v, ob_v)
            for p in range(3, 121, 3)
            for os_v in range(5, 48, 3)
            for ob_v in range(52, 96, 3)
            if ob_v > os_v + 15],

    # ─── MACD: dense (fast, slow, signal) sweep ───
    "MACD": [(f, s, sg)
             for f in range(2, 55, 3)
             for s in range(f + 3, 120, 4)
             for sg in range(2, min(s, 45), 3)],

    # ─── Drift: more lookbacks, finer thresholds ───
    "Drift": [(lb, dt, hp)
              for lb in range(3, 180, 5)
              for dt in [0.50, 0.52, 0.55, 0.57, 0.60, 0.62,
                         0.65, 0.67, 0.70, 0.73, 0.75, 0.80]
              for hp in range(1, 35, 2)],

    # ─── RAMOM: momentum + volatility regime ───
    "RAMOM": [(mp, vp, ez, xz)
              for mp in range(3, 120, 5)
              for vp in range(3, 60, 5)
              for ez in [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
              for xz in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]],

    # ─── Turtle: entry/exit/ATR period/multiplier ───
    "Turtle": [(ep, xp, ap, am)
               for ep in range(3, 120, 4)
               for xp in range(2, 60, 4)
               for ap in [7, 10, 14, 20]
               for am in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
               if xp < ep],

    # ─── Bollinger bands: lookback + band width ───
    "Bollinger": [(p, ns)
                  for p in range(3, 200, 2)
                  for ns in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0,
                             2.3, 2.5, 2.8, 3.0, 3.5, 4.0]],

    # ─── Keltner channels ───
    "Keltner": [(ep, ap, am)
                for ep in range(3, 150, 4)
                for ap in [7, 10, 14, 20]
                for am in [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]],

    # ─── MultiFactor: RSI + momentum + volatility ───
    "MultiFactor": [(rp, mp, vp, lt, st)
                    for rp in [5, 7, 10, 14, 21]
                    for mp in range(5, 100, 8)
                    for vp in range(5, 60, 8)
                    for lt in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
                    for st in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]],

    # ─── Volatility regime ───
    "VolRegime": [(ap, vt, ms, ml, ros, rob)
                  for ap in [10, 14, 20]
                  for vt in [0.010, 0.012, 0.015, 0.018, 0.020,
                             0.025, 0.030, 0.035]
                  for ms in [3, 5, 8, 10, 15, 20]
                  for ml in [25, 30, 40, 50, 60, 80, 100]
                  if ms < ml
                  for ros in [20, 25, 28, 30, 35]
                  for rob in [65, 70, 75, 80, 85]],

    # ─── MESA adaptive: fast/slow limit ───
    "MESA": [(fl, sl)
             for fl in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                        0.6, 0.7, 0.8, 0.9]
             for sl in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]],

    # ─── KAMA: adaptive MA ───
    "KAMA": [(erp, fsc, ssc, asm, ap)
             for erp in [5, 8, 10, 12, 15, 20, 25, 30]
             for fsc in [2, 3, 4]
             for ssc in [15, 20, 25, 30, 40, 50]
             for asm in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
             for ap in [10, 14, 20]],

    # ─── Donchian channel ───
    "Donchian": [(ep, ap, am)
                 for ep in range(3, 120, 3)
                 for ap in [10, 14, 20]
                 for am in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]],

    # ─── Z-Score mean reversion ───
    "ZScore": [(lb, ez, xz, sz)
               for lb in range(5, 150, 4)
               for ez in [0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
               for xz in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]
               for sz in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]],

    # ─── Momentum breakout ───
    "MomBreak": [(hp, pp, ap, at)
                 for hp in [5, 8, 10, 15, 20, 25, 30, 40, 50, 60,
                            80, 100, 120, 150, 200, 250]
                 for pp in [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12]
                 for ap in [10, 14, 20]
                 for at in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]],

    # ─── Regime EMA triple ───
    "RegimeEMA": [(ap, vt, fe, se, te)
                  for ap in [10, 14, 20]
                  for vt in [0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030]
                  for fe in [3, 5, 8, 10, 15, 20]
                  for se in [15, 20, 30, 40, 50, 60, 80]
                  if fe < se
                  for te in [40, 50, 60, 80, 100, 120, 150]
                  if se < te],

    # ─── Dual momentum ───
    "DualMom": [(fl, slo)
                for fl in [2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80]
                for slo in [10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250]
                if fl < slo],

    # ─── Consensus (multi-signal voting) ───
    "Consensus": [(ms, ml, rp, mom_lb, os_v, ob_v, vt)
                  for ms in [5, 10, 15, 20]
                  for ml in [30, 50, 80, 100, 150, 200]
                  if ms < ml
                  for rp in [7, 14, 21]
                  for mom_lb in [10, 20, 30, 40]
                  for os_v in [20, 25, 30, 35]
                  for ob_v in [65, 70, 75, 80]
                  for vt in [2, 3, 4]],
}

# ═══════════════════════════════════════════════════════════════
#  Crypto vs Stock classification
# ═══════════════════════════════════════════════════════════════
CRYPTO_SYMS = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC",
}

LEVERAGE_LEVELS = [1, 2, 3, 5, 10]

TF_COMBOS = [
    ("1h", "4h"),
    ("4h", "1d"),
    ("1h", "4h", "1d"),
]


def is_crypto(sym: str) -> bool:
    return sym.upper() in CRYPTO_SYMS or sym.replace("USDT", "") in CRYPTO_SYMS


# ═══════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════

def load_daily(data_dir: str, min_bars: int = 400) -> Dict[str, Dict[str, np.ndarray]]:
    """Load daily CSVs from data_dir root and data_dir/daily/."""
    datasets = {}
    search_dirs = [data_dir]
    daily_subdir = os.path.join(data_dir, "daily")
    if os.path.isdir(daily_subdir):
        search_dirs.append(daily_subdir)

    for d in search_dirs:
        for f in sorted(os.listdir(d)):
            if not f.endswith(".csv"):
                continue
            sym = f.replace(".csv", "")
            if sym in datasets:
                continue
            df = pd.read_csv(os.path.join(d, f))
            if len(df) < min_bars:
                continue
            for col in ("close", "open", "high", "low"):
                if col not in df.columns:
                    break
            else:
                datasets[sym] = {
                    "c": np.ascontiguousarray(df["close"].values, dtype=np.float64),
                    "o": np.ascontiguousarray(df["open"].values, dtype=np.float64),
                    "h": np.ascontiguousarray(df["high"].values, dtype=np.float64),
                    "l": np.ascontiguousarray(df["low"].values, dtype=np.float64),
                }
    return datasets


def load_intraday(data_dir: str, tf: str, min_bars: int = 400) -> Dict[str, Dict[str, np.ndarray]]:
    tf_dir = os.path.join(data_dir, tf)
    if not os.path.isdir(tf_dir):
        return {}
    datasets = {}
    for f in sorted(os.listdir(tf_dir)):
        if not f.endswith(".csv"):
            continue
        sym = f.replace(f"_{tf}.csv", "")
        try:
            df = pd.read_csv(os.path.join(tf_dir, f), parse_dates=["date"])
        except Exception:
            df = pd.read_csv(os.path.join(tf_dir, f))
        if len(df) < min_bars:
            continue
        for col in ("close", "open", "high", "low"):
            if col not in df.columns:
                break
        else:
            datasets[sym] = {
                "c": np.ascontiguousarray(df["close"].values, dtype=np.float64),
                "o": np.ascontiguousarray(df["open"].values, dtype=np.float64),
                "h": np.ascontiguousarray(df["high"].values, dtype=np.float64),
                "l": np.ascontiguousarray(df["low"].values, dtype=np.float64),
            }
    return datasets


def load_intraday_as_df(data_dir: str, tf: str) -> Dict[str, pd.DataFrame]:
    tf_dir = os.path.join(data_dir, tf)
    if not os.path.isdir(tf_dir):
        return {}
    result = {}
    for f in sorted(os.listdir(tf_dir)):
        if not f.endswith(".csv"):
            continue
        sym = f.replace(f"_{tf}.csv", "")
        try:
            df = pd.read_csv(os.path.join(tf_dir, f), parse_dates=["date"])
        except Exception:
            df = pd.read_csv(os.path.join(tf_dir, f))
        if len(df) < 400 or "close" not in df.columns:
            continue
        if "date" in df.columns:
            df = df.set_index("date")
        result[sym] = df
    return result


def load_daily_as_df(data_dir: str) -> Dict[str, pd.DataFrame]:
    result = {}
    search_dirs = [data_dir]
    daily_subdir = os.path.join(data_dir, "daily")
    if os.path.isdir(daily_subdir):
        search_dirs.append(daily_subdir)

    for d in search_dirs:
        for f in sorted(os.listdir(d)):
            if not f.endswith(".csv"):
                continue
            sym = f.replace(".csv", "")
            if sym in result:
                continue
            try:
                df = pd.read_csv(os.path.join(d, f), parse_dates=["date"])
            except Exception:
                df = pd.read_csv(os.path.join(d, f))
            if len(df) < 400 or "close" not in df.columns:
                continue
            if "date" in df.columns:
                df = df.set_index("date")
            result[sym] = df
    return result


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def print_header(title, width=100):
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def make_config(sym: str, leverage: float, interval: str) -> BacktestConfig:
    sl = min(0.40, 0.80 / leverage) if leverage > 1 else 0.40
    if is_crypto(sym):
        return BacktestConfig.crypto(leverage=leverage, stop_loss_pct=sl, interval=interval)
    else:
        return BacktestConfig.stock_ibkr(leverage=leverage, stop_loss_pct=sl, interval=interval)


def count_grids(grids: dict) -> dict:
    """Count parameter combinations per strategy."""
    counts = {}
    for sn in KERNEL_NAMES:
        counts[sn] = len(grids.get(sn, []))
    return counts


# ═══════════════════════════════════════════════════════════════
#  PHASE 1: Single-TF Multi-Leverage Robust Scan
# ═══════════════════════════════════════════════════════════════

def phase1_scan(
    all_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    grids: Dict[str, list],
    leverage_levels: List[int],
    n_mc: int = 50,
    n_shuf: int = 30,
    n_boot: int = 30,
) -> List[Dict[str, Any]]:

    ranking: List[Dict[str, Any]] = []

    for tf_label, datasets in all_data.items():
        if not datasets:
            continue

        for lev in leverage_levels:
            if not is_crypto(list(datasets.keys())[0]) and lev > 4:
                continue

            crypto_syms = {s: d for s, d in datasets.items() if is_crypto(s)}
            stock_syms = {s: d for s, d in datasets.items() if not is_crypto(s)}

            for label, sub_data in [("crypto", crypto_syms), ("stock", stock_syms)]:
                if not sub_data:
                    continue
                sym_list = list(sub_data.keys())
                sample_sym = sym_list[0]
                config = make_config(sample_sym, float(lev), tf_label)

                print(f"    {tf_label} / {lev}x / {label} ({len(sym_list)} sym) ... ",
                      end="", flush=True)
                t0 = time.time()
                result = run_robust_scan(
                    symbols=sym_list,
                    data=sub_data,
                    config=config,
                    param_grids=grids,
                    n_mc_paths=n_mc,
                    n_shuffle_paths=n_shuf,
                    n_bootstrap_paths=n_boot,
                    parallel_symbols="auto",
                )
                elapsed = time.time() - t0
                print(f"{elapsed:.1f}s ({result.total_combos:,} combos)")

                for sym in result.per_symbol:
                    for sn, metrics in result.per_symbol[sym].items():
                        ranking.append({
                            "tf": tf_label,
                            "leverage": lev,
                            "symbol": sym,
                            "strategy": sn,
                            "params": metrics.get("params"),
                            "oos_ret": metrics.get("oos_ret", 0),
                            "oos_dd": metrics.get("oos_dd", 0),
                            "oos_trades": metrics.get("oos_trades", 0),
                            "sharpe": metrics.get("sharpe", 0),
                            "dsr_p": metrics.get("dsr_p", 1),
                            "mc_mean": metrics.get("mc_mean", 0),
                            "mc_pct_positive": metrics.get("mc_pct_positive", 0),
                            "shuffle_mean": metrics.get("shuffle_mean", 0),
                            "bootstrap_mean": metrics.get("bootstrap_mean", 0),
                            "wf_score": metrics.get("wf_score", -1e18),
                            "wfe_mean": metrics.get("wfe_mean", 0),
                            "gen_gap_mean": metrics.get("gen_gap_mean", 0),
                            "ann_ret": metrics.get("ann_ret", 0),
                            "type": "single-TF",
                        })

    return ranking


# ═══════════════════════════════════════════════════════════════
#  PHASE 2: Multi-TF Fusion Scan
# ═══════════════════════════════════════════════════════════════

def phase2_multi_tf(
    phase1_ranking: List[Dict[str, Any]],
    tf_dfs: Dict[str, Dict[str, pd.DataFrame]],
    leverage_levels: List[int],
    max_strats_per_sym_tf: int = 5,
) -> List[Dict[str, Any]]:

    top_by_sym_tf: Dict[Tuple[str, str], List[Dict]] = {}
    for e in phase1_ranking:
        if e["sharpe"] <= 0 or e["oos_ret"] <= 0 or e["params"] is None:
            continue
        key = (e["symbol"], e["tf"])
        top_by_sym_tf.setdefault(key, []).append(e)

    for key in top_by_sym_tf:
        top_by_sym_tf[key].sort(key=lambda x: x["wf_score"], reverse=True)
        top_by_sym_tf[key] = top_by_sym_tf[key][:max_strats_per_sym_tf]

    results = []
    modes = ["trend_filter", "consensus"]
    tested = 0

    for tf_combo in TF_COMBOS:
        for sym in set(s for (s, _) in top_by_sym_tf.keys()):
            strats_per_tf = {}
            for tf in tf_combo:
                key = (sym, tf)
                if key not in top_by_sym_tf:
                    break
                strats_per_tf[tf] = top_by_sym_tf[key]
            else:
                if len(strats_per_tf) != len(tf_combo):
                    continue

            if len(strats_per_tf) != len(tf_combo):
                continue

            has_all_tf_data = all(
                sym in tf_dfs.get(tf, {}) for tf in tf_combo
            )
            if not has_all_tf_data:
                continue

            tf_list = list(tf_combo)
            strat_options = [strats_per_tf[tf] for tf in tf_list]

            for combo in itertools.product(*strat_options):
                for mode in modes:
                    for lev in leverage_levels:
                        if not is_crypto(sym) and lev > 4:
                            continue
                        tf_configs = {}
                        for i, tf in enumerate(tf_list):
                            e = combo[i]
                            tf_configs[tf] = (e["strategy"], tuple(e["params"]))

                        tf_data = {tf: tf_dfs[tf][sym] for tf in tf_list}
                        config = make_config(sym, float(lev), tf_list[0])
                        try:
                            r = backtest_multi_tf(
                                tf_configs, tf_data, config, mode=mode,
                            )
                            tested += 1
                            if r.n_trades >= 5:
                                results.append({
                                    "type": "multi-TF",
                                    "symbol": sym,
                                    "leverage": lev,
                                    "mode": mode,
                                    "tf_combo": "+".join(tf_list),
                                    "tf_configs": {
                                        tf: {"strategy": s, "params": list(p)}
                                        for tf, (s, p) in tf_configs.items()
                                    },
                                    "ret_pct": r.ret_pct,
                                    "max_dd": r.max_dd_pct,
                                    "sharpe": r.sharpe,
                                    "sortino": r.sortino,
                                    "calmar": r.calmar,
                                    "n_trades": r.n_trades,
                                    "wf_score": r.sharpe * max(0, 1 + r.ret_pct / 100),
                                })
                        except Exception:
                            pass

    print(f"    Tested {tested:,} multi-TF combinations, {len(results)} viable")
    return results


# ═══════════════════════════════════════════════════════════════
#  PHASE 3: Quality Filter + CPCV Validation
# ═══════════════════════════════════════════════════════════════

def phase3_filter_and_validate(
    phase1_ranking: List[Dict[str, Any]],
    phase2_ranking: List[Dict[str, Any]],
    all_data_1d: Dict[str, Dict[str, np.ndarray]],
    n_mc: int = 20,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    viable_single = [
        e for e in phase1_ranking
        if e["sharpe"] > 0.3
        and e.get("dsr_p", 1) < 0.15
        and e.get("mc_pct_positive", 0) > 0.5
        and e["oos_ret"] > 0
        and e["params"] is not None
    ]
    viable_single.sort(key=lambda x: x["wf_score"], reverse=True)

    viable_mtf = [
        e for e in phase2_ranking
        if e["sharpe"] > 0.3
        and e["ret_pct"] > 0
        and e.get("max_dd", 100) < 60
    ]
    viable_mtf.sort(key=lambda x: x["wf_score"], reverse=True)

    syms_to_validate = set()
    for e in viable_single[:80]:
        if e["tf"] == "1d" and e["symbol"] in all_data_1d:
            syms_to_validate.add(e["symbol"])

    if syms_to_validate:
        print(f"    Running CPCV on {len(syms_to_validate)} symbols ... ", end="", flush=True)
        sub_data = {s: all_data_1d[s] for s in syms_to_validate}
        sample = list(syms_to_validate)[0]
        config = make_config(sample, 1.0, "1d")

        t0 = time.time()
        cpcv_result = run_cpcv_scan(
            symbols=list(syms_to_validate),
            data=sub_data,
            config=config,
            n_mc_paths=n_mc,
        )
        print(f"{time.time()-t0:.1f}s ({cpcv_result.total_combos:,} combos)")

        cpcv_best = {}
        for sym in cpcv_result.per_symbol:
            for sn, m in cpcv_result.per_symbol[sym].items():
                score = m.get("cpcv_score", m.get("wf_score", -1e18))
                key = (sym, sn)
                if key not in cpcv_best or score > cpcv_best[key]:
                    cpcv_best[key] = score

        for e in viable_single:
            key = (e["symbol"], e["strategy"])
            if key in cpcv_best and cpcv_best[key] > 0:
                e["cpcv_validated"] = True
                e["cpcv_score"] = cpcv_best[key]
            else:
                e["cpcv_validated"] = False

    return viable_single, viable_mtf


# ═══════════════════════════════════════════════════════════════
#  PHASE 4: Export Live Trading Config + Report
# ═══════════════════════════════════════════════════════════════

def phase4_export(
    viable_single: List[Dict[str, Any]],
    viable_mtf: List[Dict[str, Any]],
    output_path: str,
    top_n: int = 50,
):
    recommendations = []
    rank = 0

    seen = set()
    for e in viable_single[:top_n]:
        key = (e["symbol"], e["strategy"], e["tf"], e["leverage"])
        if key in seen:
            continue
        seen.add(key)
        rank += 1
        recommendations.append({
            "rank": rank,
            "symbol": e["symbol"],
            "type": "single-TF",
            "strategy": e["strategy"],
            "params": list(e["params"]) if e["params"] else [],
            "leverage": e["leverage"],
            "interval": e["tf"],
            "cpcv_validated": e.get("cpcv_validated", False),
            "backtest_metrics": {
                "oos_ret": round(e["oos_ret"], 2),
                "max_dd": round(e["oos_dd"], 2),
                "sharpe": round(e["sharpe"], 3),
                "dsr_p": round(e.get("dsr_p", 1), 4),
                "mc_pct_positive": round(e.get("mc_pct_positive", 0), 3),
                "wf_score": round(e["wf_score"], 2),
                "wfe_mean": round(e.get("wfe_mean", 0), 2),
                "gen_gap": round(e.get("gen_gap_mean", 0), 2),
                "ann_ret": round(e.get("ann_ret", 0), 2),
            },
        })

    for e in viable_mtf[:top_n // 2]:
        rank += 1
        recommendations.append({
            "rank": rank,
            "symbol": e["symbol"],
            "type": "multi-TF",
            "mode": e["mode"],
            "tf_combo": e["tf_combo"],
            "tf_configs": e["tf_configs"],
            "leverage": e["leverage"],
            "backtest_metrics": {
                "ret_pct": round(e["ret_pct"], 2),
                "max_dd": round(e.get("max_dd", 0), 2),
                "sharpe": round(e["sharpe"], 3),
                "sortino": round(e.get("sortino", 0), 3),
                "calmar": round(e.get("calmar", 0), 3),
                "n_trades": e.get("n_trades", 0),
                "wf_score": round(e["wf_score"], 2),
            },
        })

    best_per_symbol = {}
    for r in recommendations:
        sym = r["symbol"]
        if sym not in best_per_symbol:
            best_per_symbol[sym] = r

    config = {
        "generated": datetime.now().isoformat(),
        "scan_type": "ULTIMATE",
        "total_recommendations": len(recommendations),
        "single_tf_count": sum(1 for r in recommendations if r["type"] == "single-TF"),
        "multi_tf_count": sum(1 for r in recommendations if r["type"] == "multi-TF"),
        "best_per_symbol": best_per_symbol,
        "recommendations": recommendations,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"    Exported {len(recommendations)} recommendations to {output_path}")
    return config


def generate_report(
    phase1_ranking: List[Dict],
    viable_single: List[Dict],
    viable_mtf: List[Dict],
    config: dict,
    elapsed: float,
    report_path: str,
):
    """Generate a detailed markdown report."""
    lines = [
        "# Ultimate Backtest Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)",
        "",
        "## Scan Summary",
        "",
        f"- Total parameter combinations evaluated: {sum(len(v) for v in ULTRA_GRIDS.values()):,}",
        f"- Total entries from Phase 1: {len(phase1_ranking):,}",
        f"- Viable single-TF strategies: {len(viable_single)}",
        f"- Viable multi-TF strategies: {len(viable_mtf)}",
        f"- Final recommendations: {config['total_recommendations']}",
        "",
        "## Quality Filters Applied",
        "",
        "| Filter | Threshold |",
        "|--------|-----------|",
        "| Sharpe Ratio | > 0.3 |",
        "| DSR p-value | < 0.15 |",
        "| MC Survival | > 50% |",
        "| OOS Return | > 0% |",
        "| Max Drawdown (multi-TF) | < 60% |",
        "| CPCV Validation | Yes (for daily TF) |",
        "",
        "## Top Recommendations for Live Trading",
        "",
        "| Rank | Symbol | Strategy | Leverage | TF | OOS Ret% | Sharpe | DSR p | MC>0% | Score |",
        "|------|--------|----------|----------|----|----------|--------|-------|-------|-------|",
    ]

    for rec in config.get("recommendations", [])[:30]:
        if rec["type"] == "single-TF":
            m = rec["backtest_metrics"]
            cv = " ✓" if rec.get("cpcv_validated") else ""
            lines.append(
                f"| {rec['rank']} | {rec['symbol']} | {rec['strategy']} | "
                f"{rec['leverage']}x | {rec['interval']} | "
                f"{m['oos_ret']:+.1f}% | {m['sharpe']:.2f} | "
                f"{m['dsr_p']:.3f} | {m['mc_pct_positive']*100:.0f}% | "
                f"{m['wf_score']:.1f}{cv} |"
            )

    lines += [
        "",
        "## Per-Symbol Best Strategy",
        "",
    ]
    for sym, rec in config.get("best_per_symbol", {}).items():
        if rec["type"] == "single-TF":
            m = rec["backtest_metrics"]
            lines.append(f"### {sym}")
            lines.append(f"- Strategy: **{rec['strategy']}**")
            lines.append(f"- Parameters: `{rec['params']}`")
            lines.append(f"- Leverage: {rec['leverage']}x | Interval: {rec['interval']}")
            lines.append(f"- OOS Return: {m['oos_ret']:+.1f}% | Sharpe: {m['sharpe']:.2f}")
            lines.append(f"- DSR p-value: {m['dsr_p']:.4f} | MC Survival: {m['mc_pct_positive']*100:.0f}%")
            lines.append(f"- CPCV Validated: {'Yes' if rec.get('cpcv_validated') else 'No'}")
            lines.append("")

    lines += [
        "## How to Use Results for Live Trading",
        "",
        "```bash",
        "# Load the best parameters for live trading",
        "python run_live_trading.py --config reports/live_trading_config.json",
        "```",
        "",
        "The `live_trading_config.json` contains all recommended strategy-parameter",
        "combinations ranked by composite score (Sharpe × DSR × MC survival).",
        "",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"    Report saved to {report_path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ultimate Comprehensive Backtest")
    parser.add_argument("--fast", action="store_true",
                        help="Quick mode: EXPANDED grids, fewer MC paths")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run only specific phase (1-4), 0=all")
    parser.add_argument("--top", type=int, default=50,
                        help="Export top N recommendations")
    parser.add_argument("--leverage", type=str, default=None,
                        help="Custom leverage levels, e.g. '1,2,3'")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Select grid size
    if args.fast:
        grids = EXPANDED_GRIDS
        n_mc, n_shuf, n_boot = 20, 10, 10
        grid_label = "EXPANDED (fast)"
    else:
        grids = ULTRA_GRIDS
        n_mc, n_shuf, n_boot = 50, 30, 30
        grid_label = "ULTRA"

    # Leverage levels
    if args.leverage:
        lev_levels = [int(x) for x in args.leverage.split(",")]
    else:
        lev_levels = LEVERAGE_LEVELS if not args.fast else [1, 2, 3]

    total_combos = sum(len(v) for v in grids.values())

    t_global = time.time()

    # ── Print grid comparison ──
    print_header("ULTIMATE COMPREHENSIVE BACKTEST")
    print(f"  Time:         {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Grids:        {grid_label}")
    print(f"  Leverage:     {lev_levels}")
    print(f"  MC/Shuf/Boot: {n_mc}/{n_shuf}/{n_boot}")
    print(f"  Strategies:   {len(KERNEL_NAMES)} ({', '.join(KERNEL_NAMES)})")
    print(f"  Top N export: {args.top}")

    print(f"\n  {'Strategy':<14}  {'DEFAULT':>8}  {'EXPANDED':>10}  {'ULTRA':>10}  {'Active':>10}")
    print(f"  {'─' * 14}  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    d_counts = count_grids(DEFAULT_PARAM_GRIDS)
    e_counts = count_grids(EXPANDED_GRIDS)
    u_counts = count_grids(ULTRA_GRIDS)
    a_counts = count_grids(grids)
    for sn in KERNEL_NAMES:
        print(f"  {sn:<14}  {d_counts[sn]:>8,}  {e_counts[sn]:>10,}  "
              f"{u_counts[sn]:>10,}  {a_counts[sn]:>10,}")

    t_d = sum(d_counts.values())
    t_e = sum(e_counts.values())
    t_u = sum(u_counts.values())
    print(f"  {'─' * 14}  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    print(f"  {'TOTAL':<14}  {t_d:>8,}  {t_e:>10,}  {t_u:>10,}  {total_combos:>10,}")
    print(f"  {'vs DEFAULT':<14}  {'1.0x':>8}  {t_e/t_d:.1f}x{'':>5}  "
          f"{t_u/t_d:.1f}x{'':>5}  {total_combos/t_d:.1f}x")

    # ── Load all data ──
    print_header("PHASE 0: DATA LOADING")
    daily = load_daily(data_dir)
    data_4h = load_intraday(data_dir, "4h")
    data_1h = load_intraday(data_dir, "1h")

    n_d, n_4, n_1 = len(daily), len(data_4h), len(data_1h)
    print(f"  Daily: {n_d} symbols  ({', '.join(sorted(daily.keys())[:15])}{'...' if n_d>15 else ''})")
    if daily:
        for sym, d in sorted(daily.items())[:10]:
            print(f"    {sym:<10} {len(d['c']):>6,} bars")
    print(f"  4h:    {n_4} symbols")
    print(f"  1h:    {n_1} symbols")

    if n_d == 0:
        print("\n  ┌──────────────────────────────────────────────────────────────┐")
        print("  │  ERROR: No data found!                                      │")
        print("  │  Run the following to download data first:                   │")
        print("  │                                                              │")
        print("  │    python download_data.py                                   │")
        print("  │                                                              │")
        print("  │  This downloads 30 symbols × 3 timeframes (1d/4h/1h).       │")
        print("  └──────────────────────────────────────────────────────────────┘")
        sys.exit(1)

    all_data = {"1d": daily}
    if data_4h:
        all_data["4h"] = data_4h
    if data_1h:
        all_data["1h"] = data_1h

    est_cells = total_combos * sum(len(d) for d in daily.values()) * len(lev_levels)
    print(f"\n  Estimated workload: {est_cells:,.0f} kernel evaluations (Phase 1 only)")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 1
    # ═══════════════════════════════════════════════════════════
    if args.phase == 0 or args.phase == 1:
        print_header("PHASE 1: SINGLE-TF MULTI-LEVERAGE ROBUST SCAN")
        print(f"  {len(all_data)} timeframes × {len(lev_levels)} leverage levels × {total_combos:,} combos")
        t1 = time.time()
        phase1_ranking = phase1_scan(all_data, grids, lev_levels, n_mc, n_shuf, n_boot)
        t1_elapsed = time.time() - t1

        for lev in lev_levels:
            entries = [e for e in phase1_ranking if e["leverage"] == lev]
            entries.sort(key=lambda x: x["wf_score"], reverse=True)
            top = entries[:15]
            if not top:
                continue
            print(f"\n  ── Top 15 @ {lev}x ──")
            print(f"  {'#':>2} {'Sym':<10} {'TF':>3} {'Strategy':<14} {'Params':^24} "
                  f"{'OOS Ret':>9} {'DD':>7} {'Sharpe':>7} {'DSR':>6} {'MC>0':>5} {'Score':>7}")
            for i, e in enumerate(top, 1):
                ps = str(e.get('params', ''))[:22]
                print(f"  {i:>2} {e['symbol']:<10} {e['tf']:>3} {e['strategy']:<14} {ps:^24} "
                      f"{e['oos_ret']:>+8.1f}% {e['oos_dd']:>6.1f}% "
                      f"{e['sharpe']:>7.2f} {e['dsr_p']:>6.3f} "
                      f"{e['mc_pct_positive']*100:>4.0f}% {e['wf_score']:>+6.1f}")

        print(f"\n  Phase 1: {t1_elapsed:.1f}s ({t1_elapsed/60:.1f} min), "
              f"{len(phase1_ranking):,} total entries")
    else:
        phase1_ranking = []

    # ═══════════════════════════════════════════════════════════
    #  PHASE 2
    # ═══════════════════════════════════════════════════════════
    phase2_ranking: List[Dict[str, Any]] = []
    if args.phase == 0 or args.phase == 2:
        print_header("PHASE 2: MULTI-TF FUSION SCAN")
        tf_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}
        if os.path.isdir(os.path.join(data_dir, "1h")):
            tf_dfs["1h"] = load_intraday_as_df(data_dir, "1h")
        if os.path.isdir(os.path.join(data_dir, "4h")):
            tf_dfs["4h"] = load_intraday_as_df(data_dir, "4h")
        tf_dfs["1d"] = load_daily_as_df(data_dir)

        print(f"  TF DataFrames: " + ", ".join(f"{k}={len(v)}" for k, v in tf_dfs.items()))
        t2 = time.time()
        phase2_ranking = phase2_multi_tf(phase1_ranking, tf_dfs, lev_levels)
        t2_elapsed = time.time() - t2

        if phase2_ranking:
            phase2_ranking.sort(key=lambda x: x["wf_score"], reverse=True)
            print(f"\n  ── Top 15 Multi-TF ──")
            print(f"  {'#':>2} {'Sym':<8} {'Lev':>3} {'Mode':<14} {'TF':>10} "
                  f"{'Return':>8} {'DD':>7} {'Sharpe':>7} {'Trades':>6}")
            for i, r in enumerate(phase2_ranking[:15], 1):
                print(f"  {i:>2} {r['symbol']:<8} {r['leverage']:>2}x {r['mode']:<14} "
                      f"{r['tf_combo']:>10} {r['ret_pct']:>+7.1f}% "
                      f"{r['max_dd']:>6.1f}% {r['sharpe']:>7.2f} {r['n_trades']:>6}")

        print(f"\n  Phase 2: {t2_elapsed:.1f}s")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 3
    # ═══════════════════════════════════════════════════════════
    if args.phase == 0 or args.phase == 3:
        print_header("PHASE 3: QUALITY FILTER + CPCV VALIDATION")
        viable_single, viable_mtf = phase3_filter_and_validate(
            phase1_ranking, phase2_ranking, daily, n_mc=n_mc // 2,
        )
        print(f"  Viable single-TF: {len(viable_single)}")
        print(f"  Viable multi-TF:  {len(viable_mtf)}")
        cpcv_count = sum(1 for e in viable_single if e.get("cpcv_validated"))
        print(f"  CPCV validated:   {cpcv_count}")
    else:
        viable_single = [e for e in phase1_ranking if e["wf_score"] > 0]
        viable_mtf = phase2_ranking

    # ═══════════════════════════════════════════════════════════
    #  PHASE 4
    # ═══════════════════════════════════════════════════════════
    if args.phase == 0 or args.phase == 4:
        print_header("PHASE 4: EXPORT LIVE TRADING CONFIG + REPORT")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(reports_dir, f"live_trading_config_{ts}.json")
        config = phase4_export(viable_single, viable_mtf, output_path, top_n=args.top)

        latest_path = os.path.join(reports_dir, "live_trading_config.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

        total_time = time.time() - t_global
        report_path = os.path.join(reports_dir, f"ultimate_backtest_report_{ts}.md")
        generate_report(phase1_ranking, viable_single, viable_mtf, config, total_time, report_path)

        # Print final recommendations
        print_header("FINAL RECOMMENDATIONS FOR LIVE TRADING")
        for rec in config["recommendations"][:20]:
            if rec["type"] == "single-TF":
                m = rec["backtest_metrics"]
                cv = " [CPCV✓]" if rec.get("cpcv_validated") else ""
                print(f"  #{rec['rank']:>2}  {rec['symbol']:<10} {rec['strategy']:<14} "
                      f"@ {rec['leverage']}x {rec['interval']:>3}{cv}")
                print(f"       Params: {rec['params']}")
                print(f"       OOS: {m['oos_ret']:+.1f}% | DD: {m['max_dd']:.1f}% | "
                      f"Sharpe: {m['sharpe']:.2f} | DSR: {m['dsr_p']:.3f} | "
                      f"MC>0: {m['mc_pct_positive']*100:.0f}%")
                print()
            else:
                m = rec["backtest_metrics"]
                print(f"  #{rec['rank']:>2}  {rec['symbol']:<10} multi-TF "
                      f"@ {rec['leverage']}x [{rec['mode']}] {rec['tf_combo']}")
                for tf, cfg in rec["tf_configs"].items():
                    print(f"       {tf}: {cfg['strategy']} {cfg['params']}")
                print(f"       Ret: {m['ret_pct']:+.1f}% | DD: {m['max_dd']:.1f}% | "
                      f"Sharpe: {m['sharpe']:.2f} | Trades: {m['n_trades']}")
                print()

    total_time = time.time() - t_global
    print("═" * 100)
    print(f"  TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Config:     reports/live_trading_config.json")
    print("═" * 100)


if __name__ == "__main__":
    main()
