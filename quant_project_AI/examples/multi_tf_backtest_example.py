#!/usr/bin/env python3
"""
Multi-Timeframe Fusion Backtest — demonstrates backtest_multi_tf().

Runs the same fusion modes available in live trading (TrendFilter,
Consensus, Primary) and compares them against single-TF baselines.

Prerequisites:
    python examples/download_multi_tf_data.py   # 1h / 4h data
    data/BTC.csv                                # daily data

Usage:
    python examples/multi_tf_backtest_example.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_framework.backtest import (
    BacktestConfig,
    backtest,
    backtest_multi_tf,
    FUSION_MODES,
)

DATA_DIR = PROJECT_ROOT / "data"


def load_tf_data(symbol: str, interval: str) -> pd.DataFrame:
    """Load OHLCV CSV for a given symbol and interval."""
    if interval == "1d":
        path = DATA_DIR / f"{symbol}.csv"
    else:
        path = DATA_DIR / interval / f"{symbol}_{interval}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    for col in df.columns:
        df.rename(columns={col: col.strip().lower()}, inplace=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df.set_index("date", inplace=True)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    df.sort_index(inplace=True)
    return df


def run_single_tf_baseline(
    symbol: str,
    strategy: str,
    params: tuple,
    interval: str,
    config: BacktestConfig,
) -> dict:
    """Run a single-TF backtest for comparison."""
    df = load_tf_data(symbol, interval)
    result = backtest(strategy, params, df, config, detailed=True)
    return {
        "interval": interval,
        "strategy": strategy,
        "params": params,
        "ret_pct": result.ret_pct,
        "max_dd_pct": result.max_dd_pct,
        "sharpe": result.sharpe,
        "n_trades": result.n_trades,
    }


def main():
    symbol = "BTC"

    tf_strategies = {
        "1h": ("MA", (10, 50)),
        "4h": ("RSI", (14, 30, 70)),
        "1d": ("MACD", (28, 112, 3)),
    }

    config = BacktestConfig.crypto(leverage=2.0, interval="1h")

    print("=" * 70)
    print(f"  Multi-Timeframe Fusion Backtest — {symbol}")
    print("=" * 70)

    # --- Load data --------------------------------------------------------
    tf_data = {}
    for iv in tf_strategies:
        try:
            df = load_tf_data(symbol, iv)
            tf_data[iv] = df
            print(f"  [{iv:>3s}] Loaded {len(df):>6,} bars  "
                  f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
        except FileNotFoundError as e:
            print(f"  [{iv:>3s}] MISSING — {e}")
            return

    # Determine overlapping date range
    start = max(df.index[0] for df in tf_data.values())
    end = min(df.index[-1] for df in tf_data.values())
    print(f"\n  Overlap period: {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}")

    # --- Single-TF baselines ---------------------------------------------
    print("\n" + "-" * 70)
    print("  Single-TF Baselines")
    print("-" * 70)
    baselines = []
    for iv, (strat, params) in tf_strategies.items():
        cfg_iv = BacktestConfig.crypto(leverage=2.0)
        if iv == "1h":
            cfg_iv = BacktestConfig.crypto(leverage=2.0, interval="1h")
        elif iv == "4h":
            cfg_iv = BacktestConfig.crypto(leverage=2.0, interval="4h")
        try:
            bl = run_single_tf_baseline(symbol, strat, params, iv, cfg_iv)
            baselines.append(bl)
            print(f"  {iv:>3s} {strat:>8s}{str(params):>20s}  "
                  f"ret={bl['ret_pct']:>+8.2f}%  dd={bl['max_dd_pct']:>6.2f}%  "
                  f"sharpe={bl['sharpe']:>6.2f}  trades={bl['n_trades']:>4d}")
        except Exception as e:
            print(f"  {iv:>3s} {strat:>8s}  FAILED: {e}")

    # --- Multi-TF fusion backtests ----------------------------------------
    print("\n" + "-" * 70)
    print("  Multi-TF Fusion Backtests")
    print("-" * 70)

    results = []
    for mode in FUSION_MODES:
        t0 = time.perf_counter()
        try:
            result = backtest_multi_tf(
                tf_configs=tf_strategies,
                tf_data=tf_data,
                config=config,
                mode=mode,
            )
            elapsed = time.perf_counter() - t0
            results.append(result)
            print(f"\n  [{mode.upper():>14s}]  "
                  f"ret={result.ret_pct:>+8.2f}%  dd={result.max_dd_pct:>6.2f}%  "
                  f"sharpe={result.sharpe:>6.2f}  sortino={result.sortino:>6.2f}  "
                  f"calmar={result.calmar:>6.2f}  trades={result.n_trades:>4d}  "
                  f"({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"\n  [{mode.upper():>14s}]  FAILED ({elapsed:.2f}s): {e}")
            import traceback; traceback.print_exc()

    # --- Summary ----------------------------------------------------------
    if results:
        print("\n" + "=" * 70)
        print("  Comparison Summary")
        print("=" * 70)
        header = f"  {'Mode':<16s} {'Return':>10s} {'MaxDD':>8s} {'Sharpe':>8s} {'Trades':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for bl in baselines:
            label = f"{bl['interval']}:{bl['strategy']}"
            print(f"  {label:<16s} {bl['ret_pct']:>+9.2f}% {bl['max_dd_pct']:>7.2f}% "
                  f"{bl['sharpe']:>8.2f} {bl['n_trades']:>8d}")

        print("  " + "-" * (len(header) - 2))

        for r in results:
            print(f"  {r.mode:<16s} {r.ret_pct:>+9.2f}% {r.max_dd_pct:>7.2f}% "
                  f"{r.sharpe:>8.2f} {r.n_trades:>8d}")

        best = max(results, key=lambda r: r.sharpe)
        print(f"\n  Best fusion mode: {best.mode} (Sharpe={best.sharpe:.2f})")

    print("\n" + "=" * 70)
    print("  Multi-TF Fusion Backtest Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
