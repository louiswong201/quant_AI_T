#!/usr/bin/env python3
"""
Production-Grade Full Parameter Scan for Live Trading
=====================================================
Expanded parameter grids (~50K+ combos) × all symbols × all 18 strategies
with full 11-layer anti-overfitting robust scan.

Results rank strategies by OOS Sharpe, DSR, MC survival, and composite score
to select the best candidate for live deployment.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from quant_framework.backtest import BacktestConfig
from quant_framework.backtest.kernels import KERNEL_NAMES
from quant_framework.backtest.robust_scan import run_robust_scan
from quant_framework.features import OfflineMaterializer

# ═══════════════════════════════════════════════════════════════
#  EXPANDED PARAMETER GRIDS  (~50K total combos, ~6x default)
# ═══════════════════════════════════════════════════════════════
EXPANDED_GRIDS = {
    # MA crossover: denser short/long windows
    "MA": [(s, lg)
           for s in range(3, 100, 2)
           for lg in range(s + 5, 251, 4)
           if lg > s],

    # RSI: finer period/threshold sweep
    "RSI": [(p, os_v, ob_v)
            for p in range(4, 100, 3)
            for os_v in range(10, 45, 3)
            for ob_v in range(55, 92, 3)
            if ob_v > os_v + 20],

    # MACD: dense (fast, slow, signal) sweep
    "MACD": [(f, s, sg)
             for f in range(3, 50, 2)
             for s in range(f + 4, 100, 4)
             for sg in range(3, min(s, 40), 3)],

    # Drift: more lookbacks, thresholds, holding periods
    "Drift": [(lb, dt, hp)
              for lb in range(5, 150, 5)
              for dt in [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.68, 0.70, 0.75]
              for hp in range(2, 30, 2)],

    # RAMOM: momentum + volatility regime
    "RAMOM": [(mp, vp, ez, xz)
              for mp in range(5, 100, 5)
              for vp in range(5, 50, 5)
              for ez in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
              for xz in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]],

    # Turtle: entry/exit/ATR period/multiplier
    "Turtle": [(ep, xp, ap, am)
               for ep in range(5, 100, 5)
               for xp in range(3, 50, 4)
               for ap in [10, 14, 20]
               for am in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
               if xp < ep],

    # Bollinger bands: lookback + band width
    "Bollinger": [(p, ns)
                  for p in range(5, 150, 3)
                  for ns in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0, 3.5]],

    # Keltner channels
    "Keltner": [(ep, ap, am)
                for ep in range(5, 120, 5)
                for ap in [10, 14, 20]
                for am in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5]],

    # MultiFactor: RSI + momentum + volatility
    "MultiFactor": [(rp, mp, vp, lt, st)
                    for rp in [5, 7, 10, 14, 21, 28]
                    for mp in range(5, 80, 5)
                    for vp in range(5, 50, 5)
                    for lt in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
                    for st in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]],

    # Volatility regime
    "VolRegime": [(ap, vt, ms, ml, ros, rob)
                  for ap in [10, 14, 20]
                  for vt in [0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.035]
                  for ms in [3, 5, 8, 10, 15, 20]
                  for ml in [25, 30, 40, 50, 60, 80, 100]
                  if ms < ml
                  for ros in [20, 25, 30, 35]
                  for rob in [65, 70, 75, 80]],

    # MESA adaptive: fast/slow limit
    "MESA": [(fl, sl)
             for fl in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
             for sl in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]],

    # KAMA: adaptive MA
    "KAMA": [(erp, fsc, ssc, asm, ap)
             for erp in [5, 8, 10, 12, 15, 20, 25, 30]
             for fsc in [2, 3, 4]
             for ssc in [15, 20, 25, 30, 40, 50]
             for asm in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
             for ap in [10, 14, 20]],

    # Donchian channel
    "Donchian": [(ep, ap, am)
                 for ep in range(5, 100, 5)
                 for ap in [10, 14, 20]
                 for am in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]],

    # Z-Score mean reversion
    "ZScore": [(lb, ez, xz, sz)
               for lb in range(10, 120, 5)
               for ez in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
               for xz in [0.0, 0.3, 0.5, 0.8, 1.0]
               for sz in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]],

    # Momentum breakout
    "MomBreak": [(hp, pp, ap, at)
                 for hp in [10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200]
                 for pp in [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
                 for ap in [10, 14, 20]
                 for at in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]],

    # Regime EMA triple
    "RegimeEMA": [(ap, vt, fe, se, te)
                  for ap in [10, 14, 20]
                  for vt in [0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030]
                  for fe in [3, 5, 8, 10, 15, 20]
                  for se in [15, 20, 30, 40, 50, 60, 80]
                  if fe < se
                  for te in [40, 50, 60, 80, 100, 120, 150]
                  if se < te],

    # Dual momentum
    "DualMom": [(fl, slo)
                for fl in [3, 5, 8, 10, 15, 20, 30, 40, 50, 60]
                for slo in [15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
                if fl < slo],

    # Consensus (multi-signal voting)
    "Consensus": [(ms, ml, rp, mom_lb, os_v, ob_v, vt)
                  for ms in [5, 10, 15, 20]
                  for ml in [30, 50, 80, 100, 150]
                  if ms < ml
                  for rp in [7, 14, 21]
                  for mom_lb in [10, 20, 30, 40]
                  for os_v in [20, 25, 30, 35]
                  for ob_v in [65, 70, 75, 80]
                  for vt in [2, 3]],
}


def load_all_data(data_dir: str, min_bars: int = 500):
    """Load canonical OHLCV arrays via the offline materializer."""
    return OfflineMaterializer(data_dir).load_ohlcv_array_map(interval="1d", min_bars=min_bars)


def print_header(title: str, width: int = 80):
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    # ── Count expanded grid sizes ──
    total_combos = 0
    print_header("EXPANDED PARAMETER GRID SIZES")
    print(f"  {'Strategy':<14}  {'Default':>8}  {'Expanded':>10}  {'Ratio':>6}")
    print(f"  {'─' * 14}  {'─' * 8}  {'─' * 10}  {'─' * 6}")

    from quant_framework.backtest.kernels import DEFAULT_PARAM_GRIDS
    for sn in KERNEL_NAMES:
        n_def = len(DEFAULT_PARAM_GRIDS.get(sn, []))
        n_exp = len(EXPANDED_GRIDS.get(sn, []))
        ratio = n_exp / n_def if n_def > 0 else 0
        total_combos += n_exp
        print(f"  {sn:<14}  {n_def:>8,}  {n_exp:>10,}  {ratio:>5.1f}x")

    print(f"  {'─' * 14}  {'─' * 8}  {'─' * 10}  {'─' * 6}")
    total_def = sum(len(v) for v in DEFAULT_PARAM_GRIDS.values())
    print(f"  {'TOTAL':<14}  {total_def:>8,}  {total_combos:>10,}  "
          f"{total_combos / total_def:.1f}x")

    # ── Load data ──
    datasets = load_all_data(data_dir, min_bars=500)
    if not datasets:
        print("\n  ERROR: No data found. Place CSV files in ./data/")
        return

    syms = list(datasets.keys())
    print_header(f"DATA: {len(syms)} SYMBOLS LOADED")
    for sym in syms:
        n = len(datasets[sym]["c"])
        print(f"  {sym:<8}  {n:>6,} bars")

    # ── Config: realistic trading costs ──
    config = BacktestConfig(
        commission_pct_buy=0.0004,
        commission_pct_sell=0.0004,
        slippage_bps_buy=3.0,
        slippage_bps_sell=3.0,
        leverage=1.0,
        allow_short=True,
        allow_fractional_shares=True,
        daily_funding_rate=0.0003,
        funding_leverage_scaling=True,
        stop_loss_pct=0.40,
        stop_loss_slippage_pct=0.005,
        position_fraction=1.0,
    )

    # ── Run robust scan ──
    print_header(
        f"ROBUST SCAN: {total_combos:,} combos × {len(syms)} symbols × 18 strategies\n"
        f"  Walk-Forward(6 windows) + MC(30) + Shuffle(20) + Bootstrap(20) + DSR"
    )
    print(f"  Starting at {time.strftime('%H:%M:%S')}...")

    t0 = time.time()
    result = run_robust_scan(
        symbols=syms,
        data=datasets,
        config=config,
        param_grids=EXPANDED_GRIDS,
        n_mc_paths=30,
        mc_noise_std=0.002,
        n_shuffle_paths=20,
        n_bootstrap_paths=20,
        bootstrap_block=20,
    )
    elapsed = time.time() - t0

    print(f"\n  ✓ Done in {elapsed:.1f}s  |  "
          f"{result.total_combos:,} combos  |  "
          f"{result.total_combos / max(0.1, elapsed):,.0f} combos/s")

    # ═══════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════

    # ── Per-symbol best strategy ──
    print_header("TOP STRATEGY PER SYMBOL")
    print(f"  {'Symbol':<8} {'Strategy':<14} {'Params':^28} "
          f"{'OOS Ret':>8} {'OOS DD':>7} {'Sharpe':>7} {'DSR p':>6} "
          f"{'MC>0%':>5} {'Score':>7}")
    print(f"  {'─' * 8} {'─' * 14} {'─' * 28} "
          f"{'─' * 8} {'─' * 7} {'─' * 7} {'─' * 6} "
          f"{'─' * 5} {'─' * 7}")

    global_ranking = []

    for sym in sorted(result.per_symbol.keys()):
        strats = result.per_symbol[sym]
        if not strats:
            continue
        best_sn = max(strats, key=lambda s: strats[s].get("wf_score", -1e18))
        b = strats[best_sn]
        ps = str(b.get("params", ""))[:26]
        mc_pct = b.get("mc_pct_positive", 0)
        print(f"  {sym:<8} {best_sn:<14} {ps:^28} "
              f"{b['oos_ret']:>+7.1f}% {b['oos_dd']:>6.1f}% "
              f"{b['sharpe']:>7.2f} {b['dsr_p']:>6.3f} "
              f"{mc_pct * 100:>4.0f}% {b['wf_score']:>+6.1f}")

        for sn, metrics in strats.items():
            global_ranking.append({
                "symbol": sym,
                "strategy": sn,
                "params": metrics.get("params"),
                "oos_ret": metrics.get("oos_ret", 0),
                "oos_dd": metrics.get("oos_dd", 0),
                "sharpe": metrics.get("sharpe", 0),
                "dsr_p": metrics.get("dsr_p", 1),
                "mc_pct_positive": metrics.get("mc_pct_positive", 0),
                "wf_score": metrics.get("wf_score", -1e18),
                "ann_ret": metrics.get("ann_ret", 0),
                "mc_mean": metrics.get("mc_mean", 0),
                "shuffle_mean": metrics.get("shuffle_mean", 0),
                "bootstrap_mean": metrics.get("bootstrap_mean", 0),
            })

    # ── All strategies for each symbol ──
    for sym in sorted(result.per_symbol.keys()):
        strats = result.per_symbol[sym]
        if not strats:
            continue
        sorted_strats = sorted(strats.keys(),
                               key=lambda s: strats[s].get("wf_score", -1e18),
                               reverse=True)

        print_header(f"ALL STRATEGIES — {sym} ({len(datasets[sym]['c']):,} bars)")
        print(f"  {'#':>2} {'Strategy':<14} {'Params':^26} "
              f"{'OOS Ret':>8} {'DD':>6} {'Sharpe':>7} {'DSR':>6} "
              f"{'MC>0':>5} {'WFE':>6} {'Gap':>6} {'Score':>7}")
        print(f"  {'─' * 2} {'─' * 14} {'─' * 26} "
              f"{'─' * 8} {'─' * 6} {'─' * 7} {'─' * 6} "
              f"{'─' * 5} {'─' * 6} {'─' * 6} {'─' * 7}")

        for rank, sn in enumerate(sorted_strats, 1):
            b = strats[sn]
            ps = str(b.get("params", ""))[:24]
            mc_pct = b.get("mc_pct_positive", 0)
            wfe = b.get("wfe_mean", 0)
            gap = b.get("gen_gap_mean", 0)
            print(f"  {rank:>2} {sn:<14} {ps:^26} "
                  f"{b['oos_ret']:>+7.1f}% {b['oos_dd']:>5.1f}% "
                  f"{b['sharpe']:>7.2f} {b['dsr_p']:>6.3f} "
                  f"{mc_pct * 100:>4.0f}% {wfe:>5.1f}% {gap:>5.1f}% "
                  f"{b['wf_score']:>+6.1f}")

    # ── Global top 20 across all symbols ──
    global_ranking.sort(key=lambda x: x["wf_score"], reverse=True)
    top_n = min(20, len(global_ranking))

    print_header(f"GLOBAL TOP {top_n} — BEST CANDIDATES FOR LIVE TRADING")
    print(f"  {'#':>2} {'Symbol':<8} {'Strategy':<14} {'Params':^26} "
          f"{'OOS Ret':>8} {'Sharpe':>7} {'DSR p':>6} {'MC>0':>5} {'Score':>7}")
    print(f"  {'─' * 2} {'─' * 8} {'─' * 14} {'─' * 26} "
          f"{'─' * 8} {'─' * 7} {'─' * 6} {'─' * 5} {'─' * 7}")

    for i, r in enumerate(global_ranking[:top_n], 1):
        ps = str(r["params"])[:24]
        mc_pct = r.get("mc_pct_positive", 0)
        print(f"  {i:>2} {r['symbol']:<8} {r['strategy']:<14} {ps:^26} "
              f"{r['oos_ret']:>+7.1f}% {r['sharpe']:>7.2f} "
              f"{r['dsr_p']:>6.3f} {mc_pct * 100:>4.0f}% "
              f"{r['wf_score']:>+6.1f}")

    # ── Live trading recommendation ──
    viable = [r for r in global_ranking
              if r["sharpe"] > 0.5
              and r["dsr_p"] < 0.10
              and r["mc_pct_positive"] > 0.6
              and r["oos_ret"] > 0]

    print_header("LIVE TRADING RECOMMENDATIONS")
    if viable:
        print(f"  {len(viable)} strategies passed all quality filters:\n"
              f"    • Sharpe > 0.5\n"
              f"    • DSR p-value < 0.10 (statistically significant)\n"
              f"    • MC survival > 60%\n"
              f"    • Positive OOS return\n")

        for i, r in enumerate(viable[:10], 1):
            ps = str(r["params"])
            print(f"  #{i}  {r['symbol']} / {r['strategy']}")
            print(f"      Params: {ps}")
            print(f"      OOS Return: {r['oos_ret']:+.1f}%  |  "
                  f"Sharpe: {r['sharpe']:.2f}  |  "
                  f"DSR p: {r['dsr_p']:.3f}  |  "
                  f"MC Survival: {r['mc_pct_positive'] * 100:.0f}%")
            print(f"      Score: {r['wf_score']:+.1f}")
            print()

        best = viable[0]
        print(f"  ★ RECOMMENDED: {best['symbol']} / {best['strategy']}"
              f" with params {best['params']}")
        print(f"    To start paper trading:")
        print(f"    python examples/paper_trading.py "
              f"--strategy {best['strategy']} "
              f"--params {','.join(str(p) for p in best['params'])}")
    else:
        print("  No strategies passed all quality filters.")
        print("  Consider:")
        print("    • Using longer data history")
        print("    • Adjusting leverage/costs")
        print("    • Relaxing filter thresholds")

    print("\n" + "═" * 80)
    print(f"  Scan completed: {elapsed:.1f}s  |  {result.total_combos:,} total combos evaluated")
    print("═" * 80)


if __name__ == "__main__":
    main()
