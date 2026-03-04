#!/usr/bin/env python3
"""
Production Full Backtest — Live Trading Parameter Search
=========================================================
Phase 0: Data loading (daily / 4h / 1h)
Phase 1: Single-TF multi-leverage robust scan  (EXPANDED_GRIDS × 4 leverage × 3 TF)
Phase 2: Multi-TF fusion scan                  (top Phase 1 survivors × TF combos)
Phase 3: Quality filter + CPCV cross-validation
Phase 4: Export live_trading_config.json

Usage:
    python run_production_scan.py                  # full pipeline
    python run_production_scan.py --phase 1        # only Phase 1
    python run_production_scan.py --fast            # quick mode (DEFAULT grids, fewer paths)
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

from quant_framework.backtest import (
    BacktestConfig,
    backtest_multi_tf,
)
from quant_framework.backtest.kernels import (
    DEFAULT_PARAM_GRIDS,
    KERNEL_NAMES,
)
from quant_framework.backtest.robust_scan import run_robust_scan, run_cpcv_scan

from run_full_scan import EXPANDED_GRIDS

# ═══════════════════════════════════════════════════════════════
#  Crypto vs Stock classification
# ═══════════════════════════════════════════════════════════════

CRYPTO_SYMS = {"BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC"}

LEVERAGE_LEVELS = [1, 2, 3, 5]

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

def load_daily(data_dir: str, min_bars: int = 500) -> Dict[str, Dict[str, np.ndarray]]:
    datasets = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".csv"):
            continue
        sym = f.replace(".csv", "")
        df = pd.read_csv(os.path.join(data_dir, f))
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


def load_intraday(data_dir: str, tf: str, min_bars: int = 500) -> Dict[str, Dict[str, np.ndarray]]:
    tf_dir = os.path.join(data_dir, tf)
    if not os.path.isdir(tf_dir):
        return {}
    datasets = {}
    for f in sorted(os.listdir(tf_dir)):
        if not f.endswith(".csv"):
            continue
        sym = f.replace(f"_{tf}.csv", "")
        df = pd.read_csv(os.path.join(tf_dir, f), parse_dates=["date"])
        if len(df) < min_bars:
            continue
        for col in ("close", "open", "high", "low"):
            if col not in df.columns:
                break
        else:
            ts = df["date"].values.astype("datetime64[s]").astype(np.float64)
            datasets[sym] = {
                "c": np.ascontiguousarray(df["close"].values, dtype=np.float64),
                "o": np.ascontiguousarray(df["open"].values, dtype=np.float64),
                "h": np.ascontiguousarray(df["high"].values, dtype=np.float64),
                "l": np.ascontiguousarray(df["low"].values, dtype=np.float64),
                "timestamps": ts,
            }
    return datasets


def load_intraday_as_df(data_dir: str, tf: str) -> Dict[str, pd.DataFrame]:
    """Load intraday CSVs as DataFrames (needed for backtest_multi_tf)."""
    tf_dir = os.path.join(data_dir, tf)
    if not os.path.isdir(tf_dir):
        return {}
    result = {}
    for f in sorted(os.listdir(tf_dir)):
        if not f.endswith(".csv"):
            continue
        sym = f.replace(f"_{tf}.csv", "")
        df = pd.read_csv(os.path.join(tf_dir, f), parse_dates=["date"])
        if len(df) < 500:
            continue
        for col in ("close", "open", "high", "low"):
            if col not in df.columns:
                break
        else:
            df = df.set_index("date")
            result[sym] = df
    return result


def load_daily_as_df(data_dir: str) -> Dict[str, pd.DataFrame]:
    result = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".csv"):
            continue
        sym = f.replace(".csv", "")
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=["date"])
        if len(df) < 500 or "close" not in df.columns:
            continue
        df = df.set_index("date")
        result[sym] = df
    return result


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def print_header(title, width=90):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def make_config(sym: str, leverage: float, interval: str) -> BacktestConfig:
    sl = min(0.40, 0.80 / leverage) if leverage > 1 else 0.40
    if is_crypto(sym):
        return BacktestConfig.crypto(leverage=leverage, stop_loss_pct=sl, interval=interval)
    else:
        return BacktestConfig.stock_ibkr(leverage=leverage, stop_loss_pct=sl, interval=interval)


# ═══════════════════════════════════════════════════════════════
#  PHASE 1: Single-TF Multi-Leverage Robust Scan
# ═══════════════════════════════════════════════════════════════

def phase1_scan(
    all_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    grids: Dict[str, list],
    leverage_levels: List[int],
    n_mc: int = 30,
    n_shuf: int = 20,
    n_boot: int = 20,
) -> List[Dict[str, Any]]:

    ranking: List[Dict[str, Any]] = []

    for tf_label, datasets in all_data.items():
        if not datasets:
            continue
        syms = list(datasets.keys())

        for lev in leverage_levels:
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
    max_strats_per_sym_tf: int = 3,
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
    n_mc: int = 15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    # Filter single-TF
    viable_single = [
        e for e in phase1_ranking
        if e["sharpe"] > 0.5
        and e.get("dsr_p", 1) < 0.10
        and e.get("mc_pct_positive", 0) > 0.6
        and e["oos_ret"] > 0
        and e["params"] is not None
    ]
    viable_single.sort(key=lambda x: x["wf_score"], reverse=True)

    # Filter multi-TF
    viable_mtf = [
        e for e in phase2_ranking
        if e["sharpe"] > 0.5
        and e["ret_pct"] > 0
        and e.get("max_dd", 100) < 50
    ]
    viable_mtf.sort(key=lambda x: x["wf_score"], reverse=True)

    # CPCV cross-validation on top single-TF candidates (daily data only)
    cpcv_validated: List[Dict[str, Any]] = []
    syms_to_validate = set()
    for e in viable_single[:50]:
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
                cpcv_validated.append(e)
            else:
                e["cpcv_validated"] = False

    return viable_single, viable_mtf


# ═══════════════════════════════════════════════════════════════
#  PHASE 4: Export Live Trading Config
# ═══════════════════════════════════════════════════════════════

def phase4_export(
    viable_single: List[Dict[str, Any]],
    viable_mtf: List[Dict[str, Any]],
    output_path: str,
):
    recommendations = []
    rank = 0

    seen = set()
    for e in viable_single[:30]:
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
            },
        })

    for e in viable_mtf[:20]:
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
                "n_trades": e.get("n_trades", 0),
            },
        })

    config = {
        "generated": datetime.now().isoformat(),
        "total_recommendations": len(recommendations),
        "single_tf_count": sum(1 for r in recommendations if r["type"] == "single-TF"),
        "multi_tf_count": sum(1 for r in recommendations if r["type"] == "multi-TF"),
        "recommendations": recommendations,
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"    Exported {len(recommendations)} recommendations to {output_path}")
    return config


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Quick mode: DEFAULT grids, fewer MC paths")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run only specific phase (1-4)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    grids = DEFAULT_PARAM_GRIDS if args.fast else EXPANDED_GRIDS
    n_mc = 10 if args.fast else 30
    n_shuf = 5 if args.fast else 20
    n_boot = 5 if args.fast else 20
    grid_label = "DEFAULT" if args.fast else "EXPANDED"
    total_combos = sum(len(v) for v in grids.values())

    t_global = time.time()

    print_header("PRODUCTION FULL BACKTEST")
    print(f"  Time:        {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Grids:       {grid_label} ({total_combos:,} combos)")
    print(f"  Leverage:    {LEVERAGE_LEVELS}")
    print(f"  MC/Shuf/Boot: {n_mc}/{n_shuf}/{n_boot}")
    print(f"  Strategies:  {len(KERNEL_NAMES)} ({', '.join(KERNEL_NAMES[:5])}...)")

    # ── Load all data ──
    print_header("PHASE 0: DATA LOADING")
    daily = load_daily(data_dir)
    data_4h = load_daily(os.path.join(data_dir, "4h")) if not os.path.isdir(os.path.join(data_dir, "4h")) else load_intraday(data_dir, "4h")
    data_1h = load_daily(os.path.join(data_dir, "1h")) if not os.path.isdir(os.path.join(data_dir, "1h")) else load_intraday(data_dir, "1h")

    n_d = len(daily)
    n_4 = len(data_4h)
    n_1 = len(data_1h)
    print(f"  Daily: {n_d} symbols  ({', '.join(sorted(daily.keys())[:10])}{'...' if n_d>10 else ''})")
    print(f"  4h:    {n_4} symbols")
    print(f"  1h:    {n_1} symbols")

    if n_d == 0:
        print("\n  ERROR: No daily data found. Run download_data.py first.")
        sys.exit(1)

    all_data = {"1d": daily}
    if data_4h:
        all_data["4h"] = data_4h
    if data_1h:
        all_data["1h"] = data_1h

    # ═══════════════════════════════════════════════════════════
    #  PHASE 1
    # ═══════════════════════════════════════════════════════════
    if args.phase == 0 or args.phase == 1:
        print_header("PHASE 1: SINGLE-TF MULTI-LEVERAGE ROBUST SCAN")
        print(f"  {len(all_data)} timeframes × {len(LEVERAGE_LEVELS)} leverage levels")
        t1 = time.time()
        phase1_ranking = phase1_scan(all_data, grids, LEVERAGE_LEVELS, n_mc, n_shuf, n_boot)
        t1_elapsed = time.time() - t1

        # Print top results per leverage
        for lev in LEVERAGE_LEVELS:
            entries = [e for e in phase1_ranking if e["leverage"] == lev]
            entries.sort(key=lambda x: x["wf_score"], reverse=True)
            top = entries[:10]
            if not top:
                continue
            print(f"\n  ── Top 10 @ {lev}x ──")
            print(f"  {'#':>2} {'Sym':<10} {'TF':>3} {'Strategy':<14} "
                  f"{'OOS Ret':>9} {'DD':>7} {'Sharpe':>7} {'DSR':>6} {'MC>0':>5} {'Score':>7}")
            for i, e in enumerate(top, 1):
                print(f"  {i:>2} {e['symbol']:<10} {e['tf']:>3} {e['strategy']:<14} "
                      f"{e['oos_ret']:>+8.1f}% {e['oos_dd']:>6.1f}% "
                      f"{e['sharpe']:>7.2f} {e['dsr_p']:>6.3f} "
                      f"{e['mc_pct_positive']*100:>4.0f}% {e['wf_score']:>+6.1f}")

        print(f"\n  Phase 1: {t1_elapsed:.1f}s, {len(phase1_ranking):,} total entries")
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
        phase2_ranking = phase2_multi_tf(phase1_ranking, tf_dfs, LEVERAGE_LEVELS)
        t2_elapsed = time.time() - t2

        if phase2_ranking:
            phase2_ranking.sort(key=lambda x: x["wf_score"], reverse=True)
            print(f"\n  ── Top 10 Multi-TF ──")
            print(f"  {'#':>2} {'Sym':<8} {'Lev':>3} {'Mode':<14} {'TF':>10} "
                  f"{'Return':>8} {'DD':>7} {'Sharpe':>7} {'Trades':>6}")
            for i, r in enumerate(phase2_ranking[:10], 1):
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
        print_header("PHASE 4: EXPORT LIVE TRADING CONFIG")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(reports_dir, f"live_trading_config_{ts}.json")
        config = phase4_export(viable_single, viable_mtf, output_path)

        # Also save latest
        latest_path = os.path.join(reports_dir, "live_trading_config.json")
        with open(latest_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        # Print final recommendations
        print_header("FINAL RECOMMENDATIONS FOR LIVE TRADING")
        for rec in config["recommendations"][:15]:
            if rec["type"] == "single-TF":
                m = rec["backtest_metrics"]
                cv = " [CPCV]" if rec.get("cpcv_validated") else ""
                print(f"  #{rec['rank']:>2}  {rec['symbol']:<10} {rec['strategy']:<12} "
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
    print("=" * 90)
    print(f"  TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 90)


if __name__ == "__main__":
    main()
