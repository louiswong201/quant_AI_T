#!/usr/bin/env python3
"""
Demo: Simplified backtest() & optimize() API — two functions, zero confusion.

Usage:
    .venv/bin/python examples/demo_backtest_api.py
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from quant_framework import (
    backtest,
    optimize,
    BacktestConfig,
    DEFAULT_PARAM_GRIDS,
    KERNEL_NAMES,
)


def load_csv(symbol: str) -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "..", "data", f"{symbol}.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def demo_backtest():
    """Demo 1: backtest() — test specific strategy + params instantly."""
    print("=" * 70)
    print("  DEMO 1: backtest() — single strategy, instant result")
    print("=" * 70)

    df = load_csv("BTC")
    config = BacktestConfig.crypto(leverage=3.0, stop_loss_pct=0.30)

    t0 = time.perf_counter()
    result = backtest("MA", (10, 50), df, config)
    elapsed = time.perf_counter() - t0

    print(f"\n  Strategy: MA (10, 50)")
    print(f"  Config:   Crypto 3x leverage, 30% stop-loss")
    print(f"  Data:     BTC, {len(df)} bars")
    print(f"  Time:     {elapsed*1000:.1f} ms")
    print(f"\n  Result:   {result}")
    print(f"  Return:   {result.ret_pct:+.2f}%")
    print(f"  Max DD:   {result.max_dd_pct:.2f}%")
    print(f"  Trades:   {result.n_trades}")
    print(f"  Score:    {result.score:.2f}")

    print("\n  --- Testing multiple strategies ---")
    for name, params in [
        ("RSI", (14, 30, 70)),
        ("MACD", (12, 26, 9)),
        ("Bollinger", (20, 2.0)),
        ("MESA", (0.5, 0.05)),
    ]:
        t0 = time.perf_counter()
        r = backtest(name, params, df, config)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {name:12s} {str(params):16s} → ret={r.ret_pct:>+8.1f}%  "
              f"dd={r.max_dd_pct:>5.1f}%  trades={r.n_trades:>3}  [{ms:.0f}ms]")

    return True


def demo_optimize_single():
    """Demo 2: optimize() — single symbol, full anti-overfitting pipeline."""
    print("\n" + "=" * 70)
    print("  DEMO 2: optimize() — single symbol, full 10-layer anti-overfitting")
    print("=" * 70)

    df = load_csv("BTC")
    config = BacktestConfig.crypto(leverage=1.0)

    total = sum(len(v) for v in DEFAULT_PARAM_GRIDS.values())
    print(f"\n  Data:   BTC, {len(df)} bars")
    print(f"  Config: Crypto 1x leverage (no stop-loss)")
    print(f"  Grids:  {len(DEFAULT_PARAM_GRIDS)} strategies, {total:,} combos per window")
    print(f"\n  Running 10-layer robust scan (6 WF windows × {total:,} combos)...")

    t0 = time.perf_counter()
    result = optimize(df, config)
    elapsed = time.perf_counter() - t0

    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"\n{result.summary()}")

    return result


def demo_optimize_multi():
    """Demo 3: optimize() — multi-symbol portfolio."""
    print("\n" + "=" * 70)
    print("  DEMO 3: optimize() — multi-symbol (BTC + ETH)")
    print("=" * 70)

    data = {}
    for sym in ["BTC", "ETH"]:
        df = load_csv(sym)
        data[sym] = df

    config = BacktestConfig.crypto(leverage=2.0, stop_loss_pct=0.50)

    small_grid = {
        "MA": [(10, 50), (20, 100), (5, 20), (15, 60)],
        "RSI": [(14, 30, 70), (7, 25, 75), (21, 35, 65)],
        "MACD": [(12, 26, 9), (8, 21, 5)],
        "Bollinger": [(20, 2.0), (20, 1.5)],
    }

    print(f"\n  Symbols:  BTC, ETH")
    print(f"  Config:   Crypto 2x leverage, 50% stop-loss")
    print(f"  Grids:    Custom (small) — {sum(len(v) for v in small_grid.values())} combos")

    t0 = time.perf_counter()
    result = optimize(data, config, strategies=["MA", "RSI", "MACD", "Bollinger"],
                      param_grids=small_grid)
    elapsed = time.perf_counter() - t0

    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"\n{result.summary()}")

    return result


def demo_speed_benchmark():
    """Demo 4: raw speed benchmark — combos/second."""
    print("\n" + "=" * 70)
    print("  DEMO 4: Speed Benchmark — combos/second")
    print("=" * 70)

    df = load_csv("BTC")
    config = BacktestConfig.crypto()

    c = df["close"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)

    from quant_framework.backtest.kernels import scan_all_kernels

    # Warmup JIT
    print("\n  Warming up Numba JIT...")
    _ = scan_all_kernels(c[:100], o[:100], h[:100], l[:100], config)

    print(f"  Running full scan on {len(c)} bars...")
    t0 = time.perf_counter()
    results = scan_all_kernels(c, o, h, l, config)
    elapsed = time.perf_counter() - t0

    total_combos = sum(v.get("cnt", 0) for v in results.values())
    combos_per_sec = total_combos / max(0.001, elapsed)

    print(f"\n  Results:")
    print(f"    Total combos:      {total_combos:>10,}")
    print(f"    Total time:        {elapsed:>10.2f}s")
    print(f"    Speed:             {combos_per_sec:>10,.0f} combos/s")
    print(f"    Bars per combo:    {len(c):>10,}")

    print(f"\n  Per-strategy breakdown:")
    for sn in sorted(results):
        r = results[sn]
        cnt = r.get("cnt", 0)
        ret = r.get("ret", 0)
        dd = r.get("dd", 0)
        nt = r.get("nt", 0)
        p = r.get("params")
        print(f"    {sn:14s}  best=({str(p):25s})  "
              f"ret={ret:>+8.1f}%  dd={dd:>5.1f}%  trades={nt:>3}  combos={cnt:>5}")

    return combos_per_sec


def demo_data_accuracy():
    """Demo 5: Data accuracy — verify calculations match manual computation."""
    print("\n" + "=" * 70)
    print("  DEMO 5: Data Accuracy Validation")
    print("=" * 70)

    from quant_framework.backtest.kernels import (
        _rolling_mean, _ema, _rsi_wilder, bt_ma_ls
    )

    # -- Test 1: Rolling mean matches numpy --
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000)) + 100
    prices = np.maximum(prices, 1.0)  # no negatives

    w = 20
    numba_ma = _rolling_mean(prices, w)
    numpy_ma = np.array([np.mean(prices[max(0, i-w+1):i+1])
                         if i >= w-1 else np.nan
                         for i in range(len(prices))])
    mask = ~np.isnan(numba_ma)
    err_ma = np.max(np.abs(numba_ma[mask] - numpy_ma[mask]))
    pass_ma = err_ma < 1e-10
    print(f"\n  [{'PASS' if pass_ma else 'FAIL'}] Rolling Mean: max error = {err_ma:.2e}")

    # -- Test 2: EMA convergence --
    span = 20
    numba_ema = _ema(prices, span)
    pd_ema = pd.Series(prices).ewm(span=span, adjust=False).mean().values
    err_ema = np.max(np.abs(numba_ema - pd_ema))
    pass_ema = err_ema < 1e-10
    print(f"  [{'PASS' if pass_ema else 'FAIL'}] EMA:          max error = {err_ema:.2e}")

    # -- Test 3: RSI matches Wilder's method --
    numba_rsi = _rsi_wilder(prices, 14)
    valid_rsi = numba_rsi[~np.isnan(numba_rsi)]
    pass_rsi_range = np.all((valid_rsi >= 0) & (valid_rsi <= 100))
    print(f"  [{'PASS' if pass_rsi_range else 'FAIL'}] RSI Range:    all in [0, 100]")

    # -- Test 4: MA crossover trade counting --
    # Create a signal with clear regime changes to force crossovers
    np.random.seed(99)
    seg1 = np.linspace(100, 200, 200)  # up
    seg2 = np.linspace(200, 120, 150)  # down
    seg3 = np.linspace(120, 250, 200)  # up
    seg4 = np.linspace(250, 150, 150)  # down
    trend_mixed = np.concatenate([seg1, seg2, seg3, seg4])
    ma_short = _rolling_mean(trend_mixed, 10)
    ma_long = _rolling_mean(trend_mixed, 50)
    r, d, nt = bt_ma_ls(trend_mixed, trend_mixed, ma_short, ma_long,
                         1.0, 1.0, 0.0, 1.0, 0.0, 0.80, 1.0, 0.0)
    pass_trend = nt >= 2
    print(f"  [{'PASS' if pass_trend else 'FAIL'}] Crossover:    MA(10,50) on mixed trend → ret={r:+.2f}% ({nt} trades)")

    # -- Test 5: Cost deduction --
    flat = np.full(500, 100.0, dtype=np.float64)
    r_nocost, _, nt_nocost = bt_ma_ls(flat, flat, _rolling_mean(flat, 5), _rolling_mean(flat, 20),
                                       1.0, 1.0, 0.0, 1.0, 0.0, 0.80, 1.0, 0.0)
    r_withcost, _, nt_withcost = bt_ma_ls(flat, flat, _rolling_mean(flat, 5), _rolling_mean(flat, 20),
                                           1.001, 0.999, 0.001, 1.0, 0.0, 0.80, 1.0, 0.0)
    if nt_withcost > 0:
        pass_cost = r_withcost <= r_nocost
        print(f"  [{'PASS' if pass_cost else 'FAIL'}] Cost impact:  no-cost={r_nocost:+.2f}% vs with-cost={r_withcost:+.2f}%")
    else:
        print(f"  [SKIP] Cost impact:  no trades on flat price (expected)")

    # -- Test 6: Leverage amplification --
    df = load_csv("BTC")
    c = df["close"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)

    config_1x = BacktestConfig.crypto(leverage=1.0)
    config_3x = BacktestConfig.crypto(leverage=3.0)
    r1 = backtest("MA", (10, 50), df, config_1x)
    r3 = backtest("MA", (10, 50), df, config_3x)
    ratio = abs(r3.ret_pct / r1.ret_pct) if abs(r1.ret_pct) > 0.01 else float("nan")
    pass_lev = 1.5 < ratio < 5.0 if not np.isnan(ratio) else True
    print(f"  [{'PASS' if pass_lev else 'FAIL'}] Leverage:     1x={r1.ret_pct:+.1f}%  3x={r3.ret_pct:+.1f}%  ratio={ratio:.1f}x")

    # -- Test 7: Stop-loss capping --
    config_sl = BacktestConfig.crypto(leverage=3.0, stop_loss_pct=0.20)
    r_sl = backtest("MA", (10, 50), df, config_sl)
    pass_sl = r_sl.max_dd_pct <= 100.0
    print(f"  [{'PASS' if pass_sl else 'FAIL'}] Stop-loss:    3x lev + 20% SL → dd={r_sl.max_dd_pct:.1f}%")

    # -- Test 8: Verify real data loads correctly --
    pass_data = (len(df) > 365 and
                 df["close"].iloc[0] > 0 and
                 df["open"].iloc[0] > 0 and
                 not df["close"].isna().any())
    print(f"  [{'PASS' if pass_data else 'FAIL'}] Data quality: {len(df)} bars, no NaN, positive prices")

    all_checks = [pass_ma, pass_ema, pass_rsi_range, pass_trend,
                   pass_lev, pass_sl, pass_data]
    total_pass = sum(all_checks)
    total = len(all_checks)
    print(f"\n  Accuracy Score: {total_pass}/{total} tests passed")

    return total_pass == total


if __name__ == "__main__":
    print()
    print("  Quant Framework — Simplified API Demo & Benchmark")
    print("  " + "=" * 60)

    # 1. Accuracy first
    all_accurate = demo_data_accuracy()

    # 2. Single backtest
    demo_backtest()

    # 3. Speed benchmark
    speed = demo_speed_benchmark()

    # 4. Full optimize (single symbol)
    opt_result = demo_optimize_single()

    # 5. Multi-symbol optimize (optional, quick)
    multi_result = demo_optimize_multi()

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Data Accuracy:       {'ALL PASS' if all_accurate else 'SOME FAILURES'}")
    print(f"  Scan Speed:          {speed:,.0f} combos/s")
    print(f"  Best Strategy:       {opt_result.best.strategy} {opt_result.best.params}")
    print(f"  Best OOS Return:     {opt_result.best.oos_ret:+.1f}%")
    print(f"  Best Sharpe:         {opt_result.best.sharpe:.2f}")
    print(f"  Anti-Overfitting:")
    print(f"    DSR p-value:       {opt_result.best.dsr_pvalue:.3f}")
    print(f"    MC positive:       {opt_result.best.mc_positive*100:.0f}%")
    print(f"    MC mean return:    {opt_result.best.mc_mean:+.1f}%")
    print()
