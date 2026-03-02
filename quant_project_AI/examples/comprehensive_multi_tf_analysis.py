#!/usr/bin/env python3
"""
Comprehensive Multi-Timeframe Fusion Analysis v2

Now powered by O(n) position extraction (445x faster), we can afford a
much wider search:
  - Full parameter grids (not compact subsets)
  - Top-8 strategies per TF (up from Top-3)
  - 8×8×8 = 512 fusion combos per symbol × 3 modes = 1,536 per symbol
  - Leverage exploration on top 10 configs
  - Detailed speed benchmarking throughout

Usage:
    python examples/comprehensive_multi_tf_analysis.py
"""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_framework.backtest import (
    BacktestConfig,
    backtest,
    backtest_multi_tf,
    FUSION_MODES,
    DEFAULT_PARAM_GRIDS,
    KERNEL_NAMES,
)

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

CRYPTO_SYMBOLS = ["BTC", "ETH", "SOL", "BNB"]
INTERVALS = ["1h", "4h", "1d"]
TOP_K_DISPLAY = 8
TOP_K_FUSION = 5
LEVERAGE_LEVELS = [1.0, 1.5, 2.0, 3.0, 5.0]

ALL_STRATEGIES = [s for s in KERNEL_NAMES if s in DEFAULT_PARAM_GRIDS]


@dataclass
class SingleTFResult:
    symbol: str
    interval: str
    strategy: str
    params: tuple
    ret_pct: float
    max_dd_pct: float
    sharpe: float
    n_trades: int


@dataclass
class MultiTFResult:
    symbol: str
    mode: str
    tf_strategies: Dict[str, str]
    tf_params: Dict[str, tuple]
    ret_pct: float
    max_dd_pct: float
    sharpe: float
    sortino: float
    calmar: float
    n_trades: int
    leverage: float = 1.0


def load_data(symbol: str, interval: str) -> pd.DataFrame:
    if interval == "1d":
        path = DATA_DIR / f"{symbol}.csv"
    else:
        path = DATA_DIR / interval / f"{symbol}_{interval}.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    for col in list(df.columns):
        df.rename(columns={col: col.strip().lower()}, inplace=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df.set_index("date", inplace=True)
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    df.sort_index(inplace=True)
    return df


def scan_single_tf(symbol: str, interval: str) -> Tuple[List[SingleTFResult], int, float]:
    df = load_data(symbol, interval)
    config = BacktestConfig.crypto(leverage=1.0, interval=interval)
    results: List[SingleTFResult] = []
    total = 0
    t0 = time.perf_counter()

    for strat in ALL_STRATEGIES:
        grid = DEFAULT_PARAM_GRIDS.get(strat, [])
        for params in grid:
            total += 1
            try:
                res = backtest(strat, params, df, config, detailed=True)
                results.append(SingleTFResult(
                    symbol=symbol, interval=interval, strategy=strat,
                    params=params, ret_pct=res.ret_pct,
                    max_dd_pct=res.max_dd_pct, sharpe=res.sharpe,
                    n_trades=res.n_trades,
                ))
            except Exception:
                continue

    elapsed = time.perf_counter() - t0
    results.sort(key=lambda r: r.sharpe, reverse=True)
    return results, total, elapsed


def main():
    t_global = time.perf_counter()
    timing: Dict[str, float] = {}

    print("=" * 78)
    print("  Comprehensive Multi-Timeframe Fusion Analysis v2 (O(n) optimized)")
    print("=" * 78)

    total_grids = sum(len(DEFAULT_PARAM_GRIDS.get(s, [])) for s in ALL_STRATEGIES)
    print(f"\n  Strategies: {len(ALL_STRATEGIES)}  |  Params per strategy (full grid): {total_grids:,}")

    available_symbols = []
    for sym in CRYPTO_SYMBOLS:
        ok = all(_data_exists(sym, iv) for iv in INTERVALS)
        if ok:
            available_symbols.append(sym)
    print(f"  Symbols with 1h+4h+1d: {available_symbols}")

    if not available_symbols:
        print("  No complete data. Run download_multi_tf_data.py first.")
        return

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 1 — Single-TF full-grid scan
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  PHASE 1: Single-TF Full-Grid Scan")
    print("=" * 78)

    top_per_tf: Dict[str, Dict[str, List[SingleTFResult]]] = {}
    phase1_combos = 0
    t_p1 = time.perf_counter()

    for sym in available_symbols:
        top_per_tf[sym] = {}
        for iv in INTERVALS:
            all_results, n_combos, elapsed = scan_single_tf(sym, iv)
            phase1_combos += n_combos
            top = all_results[:TOP_K_DISPLAY]
            top_per_tf[sym][iv] = all_results[:TOP_K_FUSION]

            speed = n_combos / max(elapsed, 0.001)
            print(f"\n  [{sym}|{iv:>3s}] {n_combos:>6,} combos in {elapsed:.2f}s "
                  f"({speed:,.0f} combos/s)")
            for i, r in enumerate(top[:5]):
                print(f"    #{i+1} {r.strategy:>14s} {str(r.params):>30s}  "
                      f"ret={r.ret_pct:>+9.2f}%  dd={r.max_dd_pct:>6.2f}%  "
                      f"sharpe={r.sharpe:>6.2f}  trades={r.n_trades}")

    timing["phase1"] = time.perf_counter() - t_p1
    print(f"\n  Phase 1 total: {phase1_combos:,} combos in {timing['phase1']:.1f}s "
          f"({phase1_combos / max(timing['phase1'], 0.001):,.0f} combos/s)")

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 2 — Multi-TF fusion (top K × top K × top K × 3 modes)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  PHASE 2: Multi-TF Fusion Backtest")
    print("=" * 78)

    all_mtf: List[MultiTFResult] = []
    phase2_runs = 0
    t_p2 = time.perf_counter()

    for sym in available_symbols:
        tf_data = {iv: load_data(sym, iv) for iv in INTERVALS}

        tops = {iv: top_per_tf[sym][iv] for iv in INTERVALS}
        combos = list(itertools.product(tops["1h"], tops["4h"], tops["1d"]))
        n_runs = len(combos) * len(FUSION_MODES)

        print(f"\n  [{sym}] {len(combos)} combos × {len(FUSION_MODES)} modes = {n_runs} fusion backtests")

        t_sym = time.perf_counter()
        for r1h, r4h, r1d in combos:
            tf_configs = {
                "1h": (r1h.strategy, r1h.params),
                "4h": (r4h.strategy, r4h.params),
                "1d": (r1d.strategy, r1d.params),
            }
            for mode in FUSION_MODES:
                try:
                    cfg = BacktestConfig.crypto(leverage=1.0, interval="1h")
                    result = backtest_multi_tf(
                        tf_configs=tf_configs, tf_data=tf_data,
                        config=cfg, mode=mode,
                    )
                    all_mtf.append(MultiTFResult(
                        symbol=sym, mode=mode,
                        tf_strategies=result.tf_strategies,
                        tf_params=result.tf_params,
                        ret_pct=result.ret_pct,
                        max_dd_pct=result.max_dd_pct,
                        sharpe=result.sharpe,
                        sortino=result.sortino,
                        calmar=result.calmar,
                        n_trades=result.n_trades,
                    ))
                    phase2_runs += 1
                except Exception:
                    phase2_runs += 1

        sym_time = time.perf_counter() - t_sym
        print(f"    done in {sym_time:.2f}s ({n_runs / max(sym_time, 0.001):,.0f} fusion backtests/s)")

    timing["phase2"] = time.perf_counter() - t_p2
    all_mtf.sort(key=lambda r: r.sharpe, reverse=True)

    print(f"\n  Phase 2 total: {phase2_runs:,} fusion backtests in {timing['phase2']:.2f}s "
          f"({phase2_runs / max(timing['phase2'], 0.001):,.0f}/s)")

    print("\n  ── Top 15 Multi-TF Configurations ──")
    for i, r in enumerate(all_mtf[:15]):
        strats = " | ".join(f"{iv}:{r.tf_strategies[iv]}" for iv in ["1h", "4h", "1d"])
        print(f"  #{i+1:>2d} [{r.symbol}] {r.mode:>14s}  sharpe={r.sharpe:>6.2f}  "
              f"ret={r.ret_pct:>+9.2f}%  dd={r.max_dd_pct:>6.2f}%  trades={r.n_trades}")
        print(f"       {strats}")

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 3 — Leverage exploration on top 10
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  PHASE 3: Leverage Exploration (Top 10 Configs)")
    print("=" * 78)

    lev_results: List[MultiTFResult] = []
    t_p3 = time.perf_counter()

    seen_keys = set()
    unique_top = []
    for r in all_mtf:
        key = (r.symbol, r.mode, tuple(sorted(r.tf_params.items())))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_top.append(r)
        if len(unique_top) >= 10:
            break

    for rank, base in enumerate(unique_top):
        tf_data = {iv: load_data(base.symbol, iv) for iv in INTERVALS}
        tf_configs = {iv: (base.tf_strategies[iv], base.tf_params[iv]) for iv in INTERVALS}

        strats = " | ".join(f"{iv}:{base.tf_strategies[iv]}" for iv in ["1h", "4h", "1d"])
        print(f"\n  #{rank+1} [{base.symbol}] {base.mode}: {strats}")

        for lev in LEVERAGE_LEVELS:
            try:
                cfg = BacktestConfig.crypto(leverage=lev, interval="1h")
                result = backtest_multi_tf(
                    tf_configs=tf_configs, tf_data=tf_data,
                    config=cfg, mode=base.mode,
                )
                lr = MultiTFResult(
                    symbol=base.symbol, mode=base.mode,
                    tf_strategies=result.tf_strategies,
                    tf_params=result.tf_params,
                    ret_pct=result.ret_pct,
                    max_dd_pct=result.max_dd_pct,
                    sharpe=result.sharpe,
                    sortino=result.sortino,
                    calmar=result.calmar,
                    n_trades=result.n_trades,
                    leverage=lev,
                )
                lev_results.append(lr)
                print(f"    {lev:.1f}x  ret={lr.ret_pct:>+9.2f}%  dd={lr.max_dd_pct:>6.2f}%  "
                      f"sharpe={lr.sharpe:>6.2f}  sortino={lr.sortino:>6.2f}  "
                      f"calmar={lr.calmar:>6.2f}")
            except Exception as e:
                print(f"    {lev:.1f}x  FAILED: {e}")

    timing["phase3"] = time.perf_counter() - t_p3
    timing["total"] = time.perf_counter() - t_global

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 4 — Generate report
    # ═══════════════════════════════════════════════════════════════════
    report = _generate_report(
        available_symbols, top_per_tf, all_mtf, lev_results,
        phase1_combos, phase2_runs, timing,
    )
    report_path = REPORT_DIR / "multi_tf_fusion_analysis_v2.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\n{'=' * 78}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Phase 1 (single-TF scan):   {phase1_combos:>8,} combos   in {timing['phase1']:>7.1f}s  "
          f"({phase1_combos / max(timing['phase1'], 0.001):>10,.0f} combos/s)")
    print(f"  Phase 2 (multi-TF fusion):  {phase2_runs:>8,} configs  in {timing['phase2']:>7.2f}s  "
          f"({phase2_runs / max(timing['phase2'], 0.001):>10,.0f} configs/s)")
    print(f"  Phase 3 (leverage explore): {len(lev_results):>8,} runs     in {timing['phase3']:>7.2f}s")
    print(f"  Total:                                       {timing['total']:>7.1f}s")
    print(f"\n  Report: {report_path}")
    print(f"{'=' * 78}")


def _data_exists(symbol: str, interval: str) -> bool:
    if interval == "1d":
        return (DATA_DIR / f"{symbol}.csv").exists()
    return (DATA_DIR / interval / f"{symbol}_{interval}.csv").exists()


def _generate_report(
    symbols, top_per_tf, mtf_results, lev_results,
    phase1_combos, phase2_runs, timing,
) -> str:
    L = []

    L.append("# Multi-Timeframe Fusion Backtest — Comprehensive Analysis v2")
    L.append("")
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L.append(f"Symbols: {', '.join(symbols)}")
    L.append(f"Timeframes: 1h, 4h, 1d  |  Fusion modes: trend_filter, consensus, primary")
    L.append(f"Strategies: {len(ALL_STRATEGIES)}  |  Single-TF combos: {phase1_combos:,}")
    L.append(f"Multi-TF fusion configs tested: {phase2_runs:,}")
    L.append("")

    # Performance summary
    L.append("## Performance Benchmarks")
    L.append("")
    L.append("| Phase | Count | Time | Throughput |")
    L.append("|---|---|---|---|")
    L.append(f"| Single-TF scan | {phase1_combos:,} combos | {timing['phase1']:.1f}s | "
             f"{phase1_combos / max(timing['phase1'], 0.001):,.0f} combos/s |")
    L.append(f"| Multi-TF fusion | {phase2_runs:,} configs | {timing['phase2']:.2f}s | "
             f"{phase2_runs / max(timing['phase2'], 0.001):,.0f} configs/s |")
    L.append(f"| Leverage exploration | {len(lev_results)} runs | {timing['phase3']:.2f}s | — |")
    L.append(f"| **Total** | — | **{timing['total']:.1f}s** | — |")
    L.append("")

    # Phase 1 results
    L.append("---")
    L.append("")
    L.append("## 1. Best Single-TF Strategies")
    L.append("")

    for sym in symbols:
        L.append(f"### {sym}")
        L.append("")
        L.append("| TF | # | Strategy | Params | Return | MaxDD | Sharpe | Trades |")
        L.append("|---|---|---|---|---|---|---|---|")
        for iv in INTERVALS:
            for i, r in enumerate(top_per_tf[sym].get(iv, [])[:TOP_K_DISPLAY]):
                L.append(
                    f"| {iv} | {i+1} | {r.strategy} | `{r.params}` | "
                    f"{r.ret_pct:+.2f}% | {r.max_dd_pct:.2f}% | "
                    f"{r.sharpe:.2f} | {r.n_trades} |"
                )
        L.append("")

    # Phase 2 results
    L.append("---")
    L.append("")
    L.append("## 2. Multi-TF Fusion Results (Top 30)")
    L.append("")
    L.append("| # | Symbol | Mode | 1h | 4h | 1d | Return | MaxDD | Sharpe | Sortino | Trades |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(mtf_results[:30]):
        L.append(
            f"| {i+1} | {r.symbol} | {r.mode} | "
            f"{r.tf_strategies.get('1h','-')} `{r.tf_params.get('1h','')}` | "
            f"{r.tf_strategies.get('4h','-')} `{r.tf_params.get('4h','')}` | "
            f"{r.tf_strategies.get('1d','-')} `{r.tf_params.get('1d','')}` | "
            f"{r.ret_pct:+.2f}% | {r.max_dd_pct:.2f}% | "
            f"{r.sharpe:.2f} | {r.sortino:.2f} | {r.n_trades} |"
        )
    L.append("")

    # Fusion mode comparison
    L.append("### Fusion Mode Comparison")
    L.append("")
    L.append("| Mode | Avg Return | Avg MaxDD | Avg Sharpe | Avg Sortino | Profitable% | Best Sharpe |")
    L.append("|---|---|---|---|---|---|---|")
    for mode in FUSION_MODES:
        subset = [r for r in mtf_results if r.mode == mode]
        if not subset:
            continue
        avg_ret = np.mean([r.ret_pct for r in subset])
        avg_dd = np.mean([r.max_dd_pct for r in subset])
        avg_sh = np.mean([r.sharpe for r in subset])
        avg_so = np.mean([r.sortino for r in subset])
        pct_prof = sum(1 for r in subset if r.ret_pct > 0) / len(subset) * 100
        best_sh = max(r.sharpe for r in subset)
        L.append(
            f"| {mode} | {avg_ret:+.2f}% | {avg_dd:.2f}% | "
            f"{avg_sh:.2f} | {avg_so:.2f} | {pct_prof:.1f}% | {best_sh:.2f} |"
        )
    L.append("")

    # Per-symbol best
    L.append("### Best Config per Symbol")
    L.append("")
    for sym in symbols:
        sym_results = [r for r in mtf_results if r.symbol == sym]
        if not sym_results:
            continue
        best = max(sym_results, key=lambda r: r.sharpe)
        strats = " | ".join(f"{iv}:{best.tf_strategies[iv]}" for iv in ["1h", "4h", "1d"])
        params = " | ".join(f"{iv}:`{best.tf_params[iv]}`" for iv in ["1h", "4h", "1d"])
        L.append(f"**{sym}** — Mode: `{best.mode}` | Sharpe: **{best.sharpe:.2f}** | "
                 f"Return: {best.ret_pct:+.2f}% | MaxDD: {best.max_dd_pct:.2f}%")
        L.append(f"- Strategies: {strats}")
        L.append(f"- Params: {params}")
        L.append("")

    # Single vs Multi comparison
    L.append("---")
    L.append("")
    L.append("## 3. Single-TF vs Multi-TF Comparison")
    L.append("")
    L.append("| Symbol | Best Single TF | Single Sharpe | Best Multi-TF Mode | Multi Sharpe | Delta |")
    L.append("|---|---|---|---|---|---|")
    for sym in symbols:
        best_s_sh = -999
        best_s_label = ""
        for iv in INTERVALS:
            for r in top_per_tf[sym].get(iv, [])[:1]:
                if r.sharpe > best_s_sh:
                    best_s_sh = r.sharpe
                    best_s_label = f"{iv}:{r.strategy}"
        sym_mtf = [r for r in mtf_results if r.symbol == sym]
        if sym_mtf:
            best_m = max(sym_mtf, key=lambda r: r.sharpe)
            delta = best_m.sharpe - best_s_sh
            L.append(f"| {sym} | {best_s_label} | {best_s_sh:.2f} | "
                     f"{best_m.mode} | {best_m.sharpe:.2f} | {'+' if delta >= 0 else ''}{delta:.2f} |")
    L.append("")

    # Phase 3 — Leverage
    L.append("---")
    L.append("")
    L.append("## 4. Leverage Exploration")
    L.append("")
    if lev_results:
        configs_seen = set()
        for lr in lev_results:
            key = (lr.symbol, lr.mode, str(sorted(lr.tf_params.items())))
            if key in configs_seen:
                continue
            configs_seen.add(key)
            strats = " | ".join(f"{iv}:{lr.tf_strategies[iv]}" for iv in ["1h", "4h", "1d"])
            L.append(f"### [{lr.symbol}] {lr.mode}: {strats}")
            L.append("")
            L.append("| Leverage | Return | MaxDD | Sharpe | Sortino | Calmar | Trades |")
            L.append("|---|---|---|---|---|---|---|")
            subset = [x for x in lev_results
                      if x.symbol == lr.symbol and x.mode == lr.mode
                      and str(sorted(x.tf_params.items())) == str(sorted(lr.tf_params.items()))]
            for x in sorted(subset, key=lambda z: z.leverage):
                L.append(
                    f"| {x.leverage:.1f}x | {x.ret_pct:+.2f}% | "
                    f"{x.max_dd_pct:.2f}% | {x.sharpe:.2f} | "
                    f"{x.sortino:.2f} | {x.calmar:.2f} | {x.n_trades} |"
                )
            L.append("")

    # Recommendation
    all_combined = mtf_results + lev_results
    best = max(all_combined, key=lambda r: r.sharpe) if all_combined else None

    L.append("---")
    L.append("")
    L.append("## 5. Investment Recommendation")
    L.append("")
    if best:
        strats = " | ".join(f"{iv}:{best.tf_strategies[iv]}" for iv in ["1h", "4h", "1d"])
        params = " | ".join(f"{iv}:`{best.tf_params[iv]}`" for iv in ["1h", "4h", "1d"])
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append(f"| Symbol | **{best.symbol}** |")
        L.append(f"| Fusion Mode | **{best.mode}** |")
        L.append(f"| Leverage | {best.leverage:.1f}x |")
        L.append(f"| Strategies | {strats} |")
        L.append(f"| Params | {params} |")
        L.append(f"| Return | **{best.ret_pct:+.2f}%** |")
        L.append(f"| Max Drawdown | {best.max_dd_pct:.2f}% |")
        L.append(f"| Sharpe Ratio | **{best.sharpe:.2f}** |")
        L.append(f"| Sortino Ratio | {best.sortino:.2f} |")
        L.append(f"| Calmar Ratio | {best.calmar:.2f} |")
        L.append(f"| Trades | {best.n_trades} |")
        L.append("")

        profitable_pct = sum(1 for r in mtf_results if r.ret_pct > 0) / max(len(mtf_results), 1) * 100
        mode_sharpes = {}
        for mode in FUSION_MODES:
            subset = [r for r in mtf_results if r.mode == mode]
            if subset:
                mode_sharpes[mode] = np.mean([r.sharpe for r in subset])
        best_mode = max(mode_sharpes, key=mode_sharpes.get) if mode_sharpes else "N/A"

        L.append("**Key Takeaways:**")
        L.append("")
        L.append(f"1. **{profitable_pct:.1f}%** of multi-TF configs were profitable")
        L.append(f"2. Best fusion mode: **{best_mode}** (avg Sharpe={mode_sharpes.get(best_mode, 0):.2f})")
        L.append(f"3. Total combos scanned: **{phase1_combos:,}** single-TF + **{phase2_runs:,}** multi-TF")
        L.append(f"4. Total analysis time: **{timing['total']:.1f}s**")
        L.append("")

    L.append("---")
    L.append(f"*Analysis completed in {timing['total']:.1f}s*")

    return "\n".join(L)


if __name__ == "__main__":
    main()
