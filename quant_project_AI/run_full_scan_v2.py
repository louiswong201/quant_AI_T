#!/usr/bin/env python3
"""
Production Full Scan V2 — Leverage + Multi-Timeframe
=====================================================
Phase 1: Single-TF robust scan across multiple leverage levels
Phase 2: Multi-TF fusion scan (1h + 4h) with best strategies
Phase 3: Global ranking & live trading recommendations
"""
import os
import sys
import time
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from quant_framework.backtest import (
    BacktestConfig,
    backtest_multi_tf,
)
from quant_framework.backtest.kernels import (
    KERNEL_NAMES,
    DEFAULT_PARAM_GRIDS,
)
from quant_framework.backtest.robust_scan import run_robust_scan


# ═══════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════

def load_daily(data_dir, min_bars=500):
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
                "c": df["close"].values.astype(np.float64),
                "o": df["open"].values.astype(np.float64),
                "h": df["high"].values.astype(np.float64),
                "l": df["low"].values.astype(np.float64),
            }
    return datasets


def load_intraday(data_dir, tf_label, min_bars=500):
    """Load 1h or 4h data with timestamps for multi-TF."""
    tf_dir = os.path.join(data_dir, tf_label)
    if not os.path.isdir(tf_dir):
        return {}
    datasets = {}
    for f in sorted(os.listdir(tf_dir)):
        if not f.endswith(".csv"):
            continue
        sym = f.replace(f"_{tf_label}.csv", "")
        df = pd.read_csv(os.path.join(tf_dir, f), parse_dates=["date"])
        if len(df) < min_bars:
            continue
        for col in ("close", "open", "high", "low"):
            if col not in df.columns:
                break
        else:
            ts = df["date"].values.astype("datetime64[s]").astype(np.float64)
            datasets[sym] = {
                "c": df["close"].values.astype(np.float64),
                "o": df["open"].values.astype(np.float64),
                "h": df["high"].values.astype(np.float64),
                "l": df["low"].values.astype(np.float64),
                "timestamps": ts,
            }
    return datasets


def fmt(val, width=7, decimals=1, pct=True):
    s = f"{val:+{width}.{decimals}f}"
    return s + "%" if pct else s


def print_header(title, width=90):
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


# ═══════════════════════════════════════════════════════════════
#  Phase 1: Multi-Leverage Robust Scan (daily data)
# ═══════════════════════════════════════════════════════════════

def phase1_leverage_scan(datasets, leverage_levels):
    """Run robust scan at each leverage level."""
    syms = list(datasets.keys())
    all_results = {}

    for lev in leverage_levels:
        is_crypto = any(s in ["BTC", "ETH", "SOL", "BNB", "DOGE", "XRP",
                              "LINK", "AVAX"] for s in syms)
        if is_crypto:
            config = BacktestConfig.crypto(leverage=lev, stop_loss_pct=0.40)
        else:
            config = BacktestConfig.stock(leverage=lev, stop_loss_pct=0.40)

        # For mixed portfolio, use crypto config with conservative costs
        config = BacktestConfig(
            commission_pct_buy=0.0004,
            commission_pct_sell=0.0004,
            slippage_bps_buy=3.0,
            slippage_bps_sell=3.0,
            leverage=lev,
            allow_short=True,
            allow_fractional_shares=True,
            daily_funding_rate=0.0003,
            funding_leverage_scaling=True,
            stop_loss_pct=min(0.40, 0.80 / lev),
            stop_loss_slippage_pct=0.005,
            position_fraction=1.0,
        )

        print(f"\n  Leverage {lev}x  (stop_loss={config.stop_loss_pct:.0%})  ", end="", flush=True)
        t0 = time.time()
        result = run_robust_scan(
            symbols=syms,
            data=datasets,
            config=config,
            n_mc_paths=30,
            n_shuffle_paths=20,
            n_bootstrap_paths=20,
        )
        elapsed = time.time() - t0
        print(f"✓ {elapsed:.1f}s  ({result.total_combos:,} combos)")
        all_results[lev] = result

    return all_results


# ═══════════════════════════════════════════════════════════════
#  Phase 2: Multi-TF Fusion Scan (1h + 4h)
# ═══════════════════════════════════════════════════════════════

def phase2_multi_tf_scan(data_1h, data_4h, leverage_levels):
    """Test multi-TF combinations using best strategies from Phase 1."""
    common_syms = sorted(set(data_1h.keys()) & set(data_4h.keys()))
    if not common_syms:
        print("  No symbols with both 1h and 4h data.")
        return []

    # Strategy combos to test on each timeframe
    tf_strategies = [
        ("MA", [(5, 20), (7, 40), (10, 50), (3, 40), (7, 16)]),
        ("RSI", [(14, 30, 70), (4, 16, 91), (7, 34, 91)]),
        ("MACD", [(12, 26, 9), (29, 93, 3), (7, 27, 9), (7, 11, 6)]),
        ("Bollinger", [(20, 2.0), (5, 1.8), (11, 2.5)]),
        ("RegimeEMA", [(10, 0.01, 3, 15, 80), (10, 0.01, 5, 30, 40)]),
        ("RAMOM", [(5, 45, 3.5, 1.5), (10, 30, 2.5, 0.3), (5, 20, 3.5, 0.8)]),
        ("MomBreak", [(15, 0.08, 10, 2.5), (30, 0.08, 14, 1.0)]),
    ]

    modes = ["trend_filter", "consensus"]
    results = []

    for sym in common_syms:
        d1h = data_1h[sym]
        d4h = data_4h[sym]

        for (s1h_name, s1h_params_list), (s4h_name, s4h_params_list) in \
                itertools.combinations(tf_strategies, 2):
            for p1h in s1h_params_list[:2]:
                for p4h in s4h_params_list[:2]:
                    for mode in modes:
                        for lev in leverage_levels:
                            config = BacktestConfig.crypto(
                                leverage=lev, stop_loss_pct=min(0.40, 0.80 / lev),
                                interval="1h",
                            )
                            try:
                                r = backtest_multi_tf(
                                    {"1h": (s1h_name, p1h), "4h": (s4h_name, p4h)},
                                    {"1h": d1h, "4h": d4h},
                                    config, mode=mode,
                                )
                                results.append({
                                    "symbol": sym,
                                    "mode": mode,
                                    "leverage": lev,
                                    "1h_strategy": s1h_name,
                                    "1h_params": p1h,
                                    "4h_strategy": s4h_name,
                                    "4h_params": p4h,
                                    "ret_pct": r.ret_pct,
                                    "max_dd": r.max_dd_pct,
                                    "sharpe": r.sharpe,
                                    "sortino": r.sortino,
                                    "calmar": r.calmar,
                                    "n_trades": r.n_trades,
                                    "n_bars": len(r.equity),
                                })
                            except Exception:
                                pass
    return results


def main():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    LEVERAGE_LEVELS = [1, 2, 3, 5]

    print_header("FULL SCAN V2 — LEVERAGE + MULTI-TIMEFRAME")
    print(f"  Leverage levels: {LEVERAGE_LEVELS}")
    print(f"  18 strategies × {sum(len(v) for v in DEFAULT_PARAM_GRIDS.values()):,} param combos")

    # ── Load all data ──
    daily = load_daily(data_dir, min_bars=500)
    data_1h = load_intraday(data_dir, "1h", min_bars=500)
    data_4h = load_intraday(data_dir, "4h", min_bars=500)

    print_header("DATA SUMMARY")
    print(f"  Daily:  {len(daily)} symbols  "
          + ", ".join(f"{s}({len(daily[s]['c']):,})" for s in sorted(daily.keys())))
    print(f"  1h:     {len(data_1h)} symbols  "
          + ", ".join(f"{s}({len(data_1h[s]['c']):,})" for s in sorted(data_1h.keys())))
    print(f"  4h:     {len(data_4h)} symbols  "
          + ", ".join(f"{s}({len(data_4h[s]['c']):,})" for s in sorted(data_4h.keys())))

    # ═════════════════════════════════════════════════════
    #  PHASE 1: Multi-Leverage Robust Scan
    # ═════════════════════════════════════════════════════
    print_header("PHASE 1: MULTI-LEVERAGE ROBUST SCAN (daily, 18 strategies)")
    t_start = time.time()
    lev_results = phase1_leverage_scan(daily, LEVERAGE_LEVELS)
    t_phase1 = time.time() - t_start

    # Collect all Phase 1 results
    phase1_ranking = []
    for lev, result in lev_results.items():
        for sym in result.per_symbol:
            for sn, metrics in result.per_symbol[sym].items():
                phase1_ranking.append({
                    "source": "single-TF",
                    "symbol": sym,
                    "leverage": lev,
                    "strategy": sn,
                    "params": metrics.get("params"),
                    "oos_ret": metrics.get("oos_ret", 0),
                    "oos_dd": metrics.get("oos_dd", 0),
                    "sharpe": metrics.get("sharpe", 0),
                    "dsr_p": metrics.get("dsr_p", 1),
                    "mc_positive": metrics.get("mc_pct_positive", 0),
                    "wf_score": metrics.get("wf_score", -1e18),
                })

    # ── Phase 1 results by leverage ──
    for lev in LEVERAGE_LEVELS:
        entries = [e for e in phase1_ranking if e["leverage"] == lev]
        entries.sort(key=lambda x: x["wf_score"], reverse=True)
        top = entries[:10]

        print_header(f"PHASE 1 — TOP 10 at {lev}x LEVERAGE")
        print(f"  {'#':>2} {'Symbol':<8} {'Strategy':<14} {'OOS Ret':>9} "
              f"{'DD':>7} {'Sharpe':>7} {'DSR p':>6} {'MC>0':>5} {'Score':>7}")
        print(f"  {'─'*2} {'─'*8} {'─'*14} {'─'*9} "
              f"{'─'*7} {'─'*7} {'─'*6} {'─'*5} {'─'*7}")
        for i, e in enumerate(top, 1):
            mc = e["mc_positive"]
            print(f"  {i:>2} {e['symbol']:<8} {e['strategy']:<14} "
                  f"{e['oos_ret']:>+8.1f}% {e['oos_dd']:>6.1f}% "
                  f"{e['sharpe']:>7.2f} {e['dsr_p']:>6.3f} "
                  f"{mc*100:>4.0f}% {e['wf_score']:>+6.1f}")

    # ── Cross-leverage comparison ──
    print_header("LEVERAGE COMPARISON — BEST PER SYMBOL")
    print(f"  {'Symbol':<8}", end="")
    for lev in LEVERAGE_LEVELS:
        print(f"  {'──── ' + str(lev) + 'x ────':^28}", end="")
    print()
    print(f"  {'':8}", end="")
    for _ in LEVERAGE_LEVELS:
        print(f"  {'Strategy':<12} {'Ret':>8} {'Sharpe':>7}", end="")
    print()
    print(f"  {'─'*8}", end="")
    for _ in LEVERAGE_LEVELS:
        print(f"  {'─'*12} {'─'*8} {'─'*7}", end="")
    print()

    all_syms_daily = sorted(set(e["symbol"] for e in phase1_ranking))
    for sym in all_syms_daily:
        print(f"  {sym:<8}", end="")
        for lev in LEVERAGE_LEVELS:
            entries = [e for e in phase1_ranking
                       if e["symbol"] == sym and e["leverage"] == lev]
            if entries:
                best = max(entries, key=lambda x: x["wf_score"])
                print(f"  {best['strategy']:<12} {best['oos_ret']:>+7.1f}% "
                      f"{best['sharpe']:>7.2f}", end="")
            else:
                print(f"  {'N/A':<12} {'':>8} {'':>7}", end="")
        print()

    # ═════════════════════════════════════════════════════
    #  PHASE 2: Multi-TF Fusion Scan
    # ═════════════════════════════════════════════════════
    print_header(
        "PHASE 2: MULTI-TF FUSION SCAN (1h + 4h)\n"
        f"  Symbols with 1h+4h data: "
        f"{sorted(set(data_1h.keys()) & set(data_4h.keys()))}"
    )
    t2 = time.time()
    mtf_results = phase2_multi_tf_scan(data_1h, data_4h, LEVERAGE_LEVELS)
    t_phase2 = time.time() - t2
    print(f"\n  ✓ {len(mtf_results):,} multi-TF combos tested in {t_phase2:.1f}s")

    if mtf_results:
        mtf_results.sort(key=lambda x: x["sharpe"], reverse=True)
        top_mtf = [r for r in mtf_results if r["sharpe"] > 0 and r["ret_pct"] > 0][:20]

        print_header("PHASE 2 — TOP 20 MULTI-TF COMBINATIONS")
        print(f"  {'#':>2} {'Sym':<6} {'Lev':>3} {'Mode':<14} "
              f"{'1h':<16} {'4h':<16} "
              f"{'Return':>8} {'DD':>7} {'Sharpe':>7} {'Sortino':>7} {'Trades':>6}")
        print(f"  {'─'*2} {'─'*6} {'─'*3} {'─'*14} "
              f"{'─'*16} {'─'*16} "
              f"{'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*6}")

        for i, r in enumerate(top_mtf, 1):
            s1h = f"{r['1h_strategy']}{str(r['1h_params'])[:8]}"
            s4h = f"{r['4h_strategy']}{str(r['4h_params'])[:8]}"
            print(f"  {i:>2} {r['symbol']:<6} {r['leverage']:>2}x {r['mode']:<14} "
                  f"{s1h:<16} {s4h:<16} "
                  f"{r['ret_pct']:>+7.1f}% {r['max_dd']:>6.1f}% "
                  f"{r['sharpe']:>7.2f} {r['sortino']:>7.2f} {r['n_trades']:>6}")

    # ═════════════════════════════════════════════════════
    #  PHASE 3: Global Ranking & Recommendations
    # ═════════════════════════════════════════════════════

    # Combine Phase 1 + Phase 2 into unified ranking
    global_ranking = []

    for e in phase1_ranking:
        global_ranking.append({
            "type": "single-TF",
            "symbol": e["symbol"],
            "leverage": e["leverage"],
            "config": f"{e['strategy']} {str(e.get('params',''))[:20]}",
            "ret_pct": e["oos_ret"],
            "dd_pct": e["oos_dd"],
            "sharpe": e["sharpe"],
            "dsr_p": e.get("dsr_p", 1),
            "mc_positive": e.get("mc_positive", 0),
            "score": e["wf_score"],
            "detail": e,
        })

    for r in mtf_results:
        score = r["sharpe"] * (1 + r["ret_pct"] / 100)
        global_ranking.append({
            "type": "multi-TF",
            "symbol": r["symbol"],
            "leverage": r["leverage"],
            "config": f"{r['mode']}: 1h={r['1h_strategy']} 4h={r['4h_strategy']}",
            "ret_pct": r["ret_pct"],
            "dd_pct": r["max_dd"],
            "sharpe": r["sharpe"],
            "dsr_p": None,
            "mc_positive": None,
            "score": score,
            "detail": r,
        })

    # Quality filter
    viable_single = [e for e in global_ranking
                     if e["type"] == "single-TF"
                     and e["sharpe"] > 0.5
                     and e.get("dsr_p", 1) < 0.10
                     and (e.get("mc_positive") or 0) > 0.6
                     and e["ret_pct"] > 0]
    viable_single.sort(key=lambda x: x["score"], reverse=True)

    viable_mtf = [e for e in global_ranking
                  if e["type"] == "multi-TF"
                  and e["sharpe"] > 0.5
                  and e["ret_pct"] > 0
                  and e["dd_pct"] < 60]
    viable_mtf.sort(key=lambda x: x["sharpe"], reverse=True)

    print_header("FINAL RECOMMENDATIONS FOR LIVE TRADING")

    if viable_single:
        print(f"\n  ── Single-TF (robust-validated, {len(viable_single)} passed) ──\n")
        for i, e in enumerate(viable_single[:10], 1):
            d = e["detail"]
            print(f"  #{i}  {e['symbol']} / {d['strategy']} @ {e['leverage']}x leverage")
            print(f"      Params: {d.get('params')}")
            print(f"      OOS Return: {e['ret_pct']:+.1f}%  |  DD: {e['dd_pct']:.1f}%  |  "
                  f"Sharpe: {e['sharpe']:.2f}  |  DSR p: {e['dsr_p']:.3f}  |  "
                  f"MC: {(e['mc_positive'] or 0)*100:.0f}%")
            print()

    if viable_mtf:
        print(f"\n  ── Multi-TF Fusion ({len(viable_mtf)} passed) ──\n")
        for i, e in enumerate(viable_mtf[:10], 1):
            d = e["detail"]
            print(f"  #{i}  {e['symbol']} @ {e['leverage']}x  [{d['mode']}]")
            print(f"      1h: {d['1h_strategy']} {d['1h_params']}")
            print(f"      4h: {d['4h_strategy']} {d['4h_params']}")
            print(f"      Return: {e['ret_pct']:+.1f}%  |  DD: {e['dd_pct']:.1f}%  |  "
                  f"Sharpe: {e['sharpe']:.2f}  |  Sortino: {d['sortino']:.2f}")
            print()

    total_time = time.time() - t_start
    total_evals = sum(r.total_combos for r in lev_results.values())
    print("═" * 90)
    print(f"  Total time: {total_time:.1f}s  |  Phase 1: {t_phase1:.1f}s  |  Phase 2: {t_phase2:.1f}s")
    print(f"  Single-TF combos: {total_evals:,}  |  Multi-TF combos: {len(mtf_results):,}")
    print(f"  Viable single-TF: {len(viable_single)}  |  Viable multi-TF: {len(viable_mtf)}")
    print("═" * 90)


if __name__ == "__main__":
    main()
