#!/usr/bin/env python3
"""
Unified Backtest Example
========================
Demonstrates the single-system backtest API that uses Numba kernels
automatically when a strategy has a kernel equivalent.

Usage:
    python examples/unified_backtest.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quant_framework.backtest import (
    BacktestConfig,
    BacktestEngine,
    KERNEL_NAMES,
    run_kernel,
    run_robust_scan,
    scan_all_kernels,
)
from quant_framework.backtest.kernels import KernelResult, config_to_kernel_costs
from quant_framework.data.data_manager import DataManager


def load_ohlc(data_dir: str, symbol: str):
    fp = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, parse_dates=["date"])
    return {
        "c": df["close"].values.astype(np.float64),
        "o": df["open"].values.astype(np.float64),
        "h": df["high"].values.astype(np.float64),
        "l": df["low"].values.astype(np.float64),
        "n": len(df),
    }


def main():
    print("=" * 80)
    print("  Unified Backtest System")
    print("  Numba kernels integrated into BacktestEngine")
    print("=" * 80)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    symbols = ["BTC", "ETH", "AAPL", "TSLA"]
    available = []

    print("\n[1] Loading data ...")
    datasets = {}
    for sym in symbols:
        d = load_ohlc(data_dir, sym)
        if d:
            datasets[sym] = d
            available.append(sym)
            print(f"  {sym}: {d['n']} bars")
        else:
            print(f"  {sym}: MISSING")

    if not datasets:
        print("No data found. Please ensure CSV files exist in data/")
        return

    # ================================================================
    # Demo 1: Single kernel run via framework API
    # ================================================================
    print("\n[2] Single kernel run (MA strategy on first symbol) ...")
    sym = available[0]
    d = datasets[sym]
    config = BacktestConfig.crypto(leverage=1, stop_loss_pct=0.50)

    t0 = time.time()
    result = run_kernel("MA", (10, 50), d["c"], d["o"], d["h"], d["l"], config)
    t1 = time.time()
    print(f"  Strategy: MA(10,50)")
    print(f"  Return: {result.ret_pct:+.2f}%  |  Max DD: {result.max_dd_pct:.2f}%  |  Trades: {result.n_trades}")
    print(f"  Time: {(t1-t0)*1000:.1f}ms")

    # ================================================================
    # Demo 2: Full 18-strategy parameter scan
    # ================================================================
    print(f"\n[3] Full 18-strategy parameter scan on {sym} ...")
    t0 = time.time()
    scan_results = scan_all_kernels(d["c"], d["o"], d["h"], d["l"], config)
    t1 = time.time()
    total_combos = sum(r["cnt"] for r in scan_results.values())
    print(f"  {total_combos:,} combinations scanned in {t1-t0:.1f}s ({total_combos/(t1-t0):,.0f}/s)")
    print()
    print(f"  {'Strategy':<15} {'Best Params':<30} {'Return':>8} {'Max DD':>8} {'Trades':>7} {'Score':>8}")
    print(f"  {'-'*15} {'-'*30} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")
    for sn in sorted(scan_results, key=lambda s: scan_results[s]["score"], reverse=True):
        r = scan_results[sn]
        ps = str(r["params"])[:28]
        print(f"  {sn:<15} {ps:<30} {r['ret']:>+7.1f}% {r['dd']:>7.1f}% {r['nt']:>7} {r['score']:>+7.2f}")

    # ================================================================
    # Demo 3: BacktestEngine with auto kernel detection
    # ================================================================
    print(f"\n[4] BacktestEngine auto kernel detection ...")
    try:
        dm = DataManager(data_dir=data_dir, use_parquet=False)
        engine = BacktestEngine(dm, config)

        from quant_framework.strategy.ma_strategy import MovingAverageStrategy
        from quant_framework.strategy.rsi_strategy import RSIStrategy

        ma_strat = MovingAverageStrategy(short_window=10, long_window=50)
        print(f"  MA strategy kernel_name: {ma_strat.kernel_name}")

        test_sym = available[0]
        df = pd.read_csv(os.path.join(data_dir, f"{test_sym}.csv"), parse_dates=["date"])
        start = df["date"].iloc[0].strftime("%Y-%m-%d")
        end = df["date"].iloc[-1].strftime("%Y-%m-%d")

        t0 = time.time()
        res = engine.run(ma_strat, test_sym, start, end)
        t1 = time.time()

        print(f"  Symbol: {test_sym}")
        print(f"  Kernel mode: {res.get('kernel_mode', False)}")
        print(f"  Return: {res.get('ret_pct', res.get('total_return', 0)):+.2f}%")
        print(f"  Time: {(t1-t0)*1000:.1f}ms")
    except Exception as e:
        print(f"  BacktestEngine demo skipped: {e}")

    # ================================================================
    # Demo 4: Robust scan (10-layer anti-overfitting)
    # ================================================================
    if len(available) >= 2:
        print(f"\n[5] 10-Layer Robust Scan ({available[:2]}) ...")
        scan_data = {}
        for sym in available[:2]:
            scan_data[sym] = {
                "c": datasets[sym]["c"],
                "o": datasets[sym]["o"],
                "h": datasets[sym]["h"],
                "l": datasets[sym]["l"],
            }

        t0 = time.time()
        robust_result = run_robust_scan(
            symbols=available[:2],
            data=scan_data,
            config=config,
            strategies=["MA", "RSI", "MACD", "Bollinger", "DualMom"],
            n_mc_paths=5,
            n_shuffle_paths=3,
            n_bootstrap_paths=3,
        )
        t1 = time.time()

        print(f"  {robust_result.total_combos:,} combos in {t1-t0:.1f}s "
              f"({robust_result.total_combos/max(0.1,t1-t0):,.0f}/s)")

        for sym in robust_result.per_symbol:
            strats = robust_result.per_symbol[sym]
            best_sn = max(strats, key=lambda s: strats[s].get("wf_score", -1e18))
            b = strats[best_sn]
            print(f"  {sym}: Best={best_sn} | OOS Ret={b['oos_ret']:+.1f}% | "
                  f"DD={b['oos_dd']:.1f}% | Sharpe={b['sharpe']:.2f} | "
                  f"DSR p={b['dsr_p']:.3f} | MC Mean={b['mc_mean']:+.1f}%")

    print("\n" + "=" * 80)
    print("  All demos complete. The unified backtest system is working.")
    print("=" * 80)


if __name__ == "__main__":
    main()
