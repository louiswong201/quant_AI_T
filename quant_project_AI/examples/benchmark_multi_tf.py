#!/usr/bin/env python3
"""Benchmark multi-TF backtest performance after O(n) optimization."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_framework.backtest import (
    BacktestConfig, backtest_multi_tf, backtest,
)
from quant_framework.backtest.kernels import (
    config_to_kernel_costs, eval_kernel_position_series, eval_kernel_position_array,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BEFORE_MS = {
    "pos_1h": 158.8, "pos_4h": 45.5, "pos_1d": 9.8,
    "consensus": 214.5, "trend_filter": 217.5, "primary": 209.4,
    "pos_total": 207.6,
}

def load(symbol, interval):
    p = DATA_DIR / f"{symbol}.csv" if interval == "1d" else DATA_DIR / interval / f"{symbol}_{interval}.csv"
    df = pd.read_csv(p)
    for c in list(df.columns): df.rename(columns={c: c.strip().lower()}, inplace=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df.set_index("date", inplace=True)
    for c in ("open","high","low","close"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    return df.sort_index()

def main():
    print("=" * 74)
    print("  Multi-TF Backtest — Before / After O(n) Optimization")
    print("=" * 74)

    tf_configs = {
        "1h": ("RSI", (30, 25, 80)),
        "4h": ("RSI", (10, 20, 70)),
        "1d": ("DualMom", (40, 120)),
    }
    tf_data = {iv: load("BTC", iv) for iv in tf_configs}
    for iv, df in tf_data.items():
        print(f"  {iv}: {len(df):>6,} bars")

    config = BacktestConfig.crypto(leverage=1.0, interval="1h")

    # --- Warmup Numba ---
    backtest_multi_tf(tf_configs, tf_data, config, mode="consensus")

    # --- Position extraction benchmark ---
    print("\n--- Position Extraction (per TF) ---")
    print(f"  {'TF':>3s}  {'Bars':>6s}  {'Before(ms)':>10s}  {'After(ms)':>10s}  {'Speedup':>8s}")
    print(f"  {'-'*3}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")

    total_pos_after = 0
    for iv in ["1h", "4h", "1d"]:
        df = tf_data[iv]
        c = np.ascontiguousarray(df["close"].values, dtype=np.float64)
        o = np.ascontiguousarray(df["open"].values, dtype=np.float64)
        h = np.ascontiguousarray(df["high"].values, dtype=np.float64)
        l = np.ascontiguousarray(df["low"].values, dtype=np.float64)
        strat, params = tf_configs[iv]
        iv_cfg = BacktestConfig.crypto(leverage=1.0, interval=iv)
        iv_costs = config_to_kernel_costs(iv_cfg)

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            pos = eval_kernel_position_series(
                strat, params, c, o, h, l,
                iv_costs["sb"], iv_costs["ss"], iv_costs["cm"], iv_costs["lev"],
                iv_costs["dc"], iv_costs["sl"], iv_costs["pfrac"], iv_costs["sl_slip"],
            )
            times.append(time.perf_counter() - t0)
        med = np.median(times) * 1000
        total_pos_after += med
        before = BEFORE_MS[f"pos_{iv}"]
        speedup = before / max(med, 0.001)
        print(f"  {iv:>3s}  {len(c):>6,}  {before:>10.1f}  {med:>10.3f}  {speedup:>7.0f}x")

    # --- Full backtest_multi_tf ---
    print(f"\n--- Full backtest_multi_tf ---")
    print(f"  {'Mode':>14s}  {'Before(ms)':>10s}  {'After(ms)':>10s}  {'Speedup':>8s}  {'Result':>20s}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*20}")

    for mode in ["consensus", "trend_filter", "primary"]:
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            r = backtest_multi_tf(tf_configs, tf_data, config, mode=mode)
            times.append(time.perf_counter() - t0)
        med = np.median(times) * 1000
        before = BEFORE_MS[mode]
        speedup = before / max(med, 0.001)
        print(f"  {mode:>14s}  {before:>10.1f}  {med:>10.2f}  {speedup:>7.0f}x  "
              f"ret={r.ret_pct:>+.2f}% sh={r.sharpe:.2f}")

    # --- Impact on comprehensive analysis ---
    n_configs = 324
    before_total = BEFORE_MS["consensus"] * n_configs / 1000
    after_est = np.median(times) * 1000 * n_configs / 1000
    print(f"\n--- Impact on Comprehensive Analysis (324 configs × 4 symbols) ---")
    print(f"  Before: {before_total:.0f}s per symbol → {before_total*4:.0f}s total")
    print(f"  After:  {after_est:.1f}s per symbol → {after_est*4:.1f}s total")
    print(f"  Speedup: {before_total*4 / max(after_est*4, 0.01):.0f}x")

    print("\n" + "=" * 74)

if __name__ == "__main__":
    main()
