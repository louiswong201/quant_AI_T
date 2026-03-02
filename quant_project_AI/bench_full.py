"""
Comprehensive Backtest Engine Benchmark
========================================
Tests all components at multiple data sizes, compares against
pre-optimization baselines, and reports detailed speedup ratios.
"""
import time
import sys
import numpy as np

np.random.seed(42)

# ── Pre-optimization baselines (recorded before prange / njit fusion / O3) ──
BASELINE = {
    "scan_serial_8760":       {"time": 0.2762, "throughput": 30_554},
    "scan_threaded_8760":     {"time": 0.2740, "throughput": 30_808},
    "eval_kernel_ms":         0.12,
    "multi_tf_ms":            69.11,
    "robust_scan_3strat":     0.89,
}


def gen_data(n):
    c = 100.0 * np.cumprod(1 + np.random.randn(n) * 0.02)
    o = c * (1 + np.random.randn(n) * 0.005)
    h = np.maximum(c, o) * (1 + np.abs(np.random.randn(n)) * 0.01)
    lo = np.minimum(c, o) * (1 - np.abs(np.random.randn(n)) * 0.01)
    return c, o, h, lo


def fmt_speedup(old, new):
    if new <= 0 or old <= 0:
        return "N/A"
    ratio = old / new
    return f"{ratio:.1f}x"


def run_bench():
    from quant_framework.backtest.config import BacktestConfig
    from quant_framework.backtest.kernels import (
        scan_all_kernels, eval_kernel, config_to_kernel_costs, KERNEL_NAMES,
    )
    from quant_framework.backtest import backtest_multi_tf
    from quant_framework.backtest.robust_scan import run_robust_scan

    config = BacktestConfig.crypto()

    SIZES = [4000, 8760, 20000, 50000]
    RUNS = 3

    print("=" * 72)
    print("            COMPREHENSIVE BACKTEST ENGINE BENCHMARK")
    print("=" * 72)

    # ─────────────────────────────────────────────────
    # Phase 1: JIT warmup
    # ─────────────────────────────────────────────────
    print("\n[Phase 1] JIT warmup (compiling all kernels)...")
    c_w, o_w, h_w, l_w = gen_data(2000)
    t0 = time.perf_counter()
    _ = scan_all_kernels(c_w, o_w, h_w, l_w, config, n_threads=1)
    # warmup multi-TF
    df_w = {"c": c_w, "o": o_w, "h": h_w, "l": l_w,
            "timestamps": np.arange(len(c_w), dtype=np.float64) * 3600}
    n4w = len(c_w) // 4
    df_w4 = {"c": c_w[::4][:n4w], "o": o_w[::4][:n4w], "h": h_w[::4][:n4w],
             "l": l_w[::4][:n4w],
             "timestamps": np.arange(n4w, dtype=np.float64) * 14400}
    backtest_multi_tf(
        {"1h": ("MA", (10, 50)), "4h": ("RSI", (14, 30, 70))},
        {"1h": df_w, "4h": df_w4}, config, mode="trend_filter")
    # warmup robust_scan
    data_w = {"W": {"c": c_w, "o": o_w, "h": h_w, "l": l_w}}
    _ = run_robust_scan(["W"], data_w, config, strategies=["MA"],
                        n_mc_paths=2, n_shuffle_paths=1, n_bootstrap_paths=1)
    t_warmup = time.perf_counter() - t0
    print(f"    Warmup complete: {t_warmup:.2f}s")

    # ─────────────────────────────────────────────────
    # Phase 2: scan_all_kernels — multi-size
    # ─────────────────────────────────────────────────
    print(f"\n[Phase 2] scan_all_kernels — {RUNS} runs per size")
    print("-" * 72)
    print(f"  {'Bars':>8}  {'Best(s)':>10}  {'Combos':>10}  {'Combos/s':>12}  {'vs Baseline':>12}")
    print("-" * 72)

    for sz in SIZES:
        np.random.seed(42)
        c, o, h, lo = gen_data(sz)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            R = scan_all_kernels(c, o, h, lo, config, n_threads=1)
            times.append(time.perf_counter() - t0)
        total = sum(r["cnt"] for r in R.values())
        best = min(times)
        tput = total / best

        if sz == 8760:
            base_t = BASELINE["scan_serial_8760"]["time"]
            sp = fmt_speedup(base_t, best)
        else:
            sp = "-"
        print(f"  {sz:>8,}  {best:>10.4f}  {total:>10,}  {tput:>12,.0f}  {sp:>12}")

    # ─────────────────────────────────────────────────
    # Phase 3: Per-strategy scan breakdown (8760 bars)
    # ─────────────────────────────────────────────────
    print(f"\n[Phase 3] Per-strategy scan time (8760 bars, best of {RUNS})")
    print("-" * 72)
    print(f"  {'Strategy':<14}  {'Best(s)':>10}  {'Combos':>8}  {'Combos/s':>12}")
    print("-" * 72)

    np.random.seed(42)
    c, o, h, lo = gen_data(8760)
    for sn in KERNEL_NAMES:
        times_s = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            R_s = scan_all_kernels(c, o, h, lo, config, strategies=[sn], n_threads=1)
            times_s.append(time.perf_counter() - t0)
        cnt_s = sum(r["cnt"] for r in R_s.values())
        best_s = min(times_s)
        tput_s = cnt_s / best_s if best_s > 0 else 0
        print(f"  {sn:<14}  {best_s:>10.4f}  {cnt_s:>8,}  {tput_s:>12,.0f}")

    # ─────────────────────────────────────────────────
    # Phase 4: eval_kernel throughput
    # ─────────────────────────────────────────────────
    print(f"\n[Phase 4] eval_kernel throughput")
    print("-" * 72)
    costs = config_to_kernel_costs(config)
    sb, ss, cm, lev, dc = costs["sb"], costs["ss"], costs["cm"], costs["lev"], costs["dc"]
    sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

    np.random.seed(42)
    c, o, h, lo = gen_data(8760)

    for n_calls in [100, 500, 1000]:
        t0 = time.perf_counter()
        for _ in range(n_calls):
            eval_kernel("MA", (10, 50), c, o, h, lo, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
        t_e = time.perf_counter() - t0
        ms_per = t_e / n_calls * 1000
        base_ms = BASELINE["eval_kernel_ms"]
        sp = fmt_speedup(base_ms, ms_per)
        print(f"  {n_calls:>5} calls: {t_e:.4f}s ({ms_per:.3f}ms/call)  vs baseline {base_ms}ms → {sp}")

    # ─────────────────────────────────────────────────
    # Phase 5: Multi-TF backtest
    # ─────────────────────────────────────────────────
    print(f"\n[Phase 5] Multi-TF backtest")
    print("-" * 72)

    np.random.seed(42)
    c, o, h, lo = gen_data(8760)
    ts_1h = np.arange(8760, dtype=np.float64) * 3600
    n4 = 8760 // 4
    df_1h = {"c": c, "o": o, "h": h, "l": lo, "timestamps": ts_1h}
    df_4h = {"c": c[::4][:n4], "o": o[::4][:n4], "h": h[::4][:n4],
             "l": lo[::4][:n4], "timestamps": np.arange(n4, dtype=np.float64) * 14400}

    for mode in ["trend_filter", "consensus"]:
        times_m = []
        n_iter = 50
        for _ in range(n_iter):
            t0 = time.perf_counter()
            backtest_multi_tf(
                {"1h": ("MA", (10, 50)), "4h": ("RSI", (14, 30, 70))},
                {"1h": df_1h, "4h": df_4h}, config, mode=mode,
            )
            times_m.append(time.perf_counter() - t0)
        med = sorted(times_m)[n_iter // 2]
        best_m = min(times_m)
        base_ms = BASELINE["multi_tf_ms"]
        sp = fmt_speedup(base_ms, best_m * 1000)
        print(f"  mode={mode:<14}  median={med*1000:.2f}ms  best={best_m*1000:.2f}ms  "
              f"vs baseline {base_ms}ms → {sp}")

    # ─────────────────────────────────────────────────
    # Phase 6: robust_scan — full pipeline
    # ─────────────────────────────────────────────────
    print(f"\n[Phase 6] run_robust_scan (walk-forward + MC + shuffle + bootstrap)")
    print("-" * 72)

    np.random.seed(42)
    c, o, h, lo = gen_data(8760)
    data_r = {"SYM": {"c": c, "o": o, "h": h, "l": lo}}

    for strats, label in [
        (["MA", "RSI", "MACD"], "3 strategies"),
        (["MA", "RSI", "MACD", "Drift", "RAMOM", "Turtle"], "6 strategies"),
        (None, "all 18 strategies"),
    ]:
        times_r = []
        for _ in range(2):
            t0 = time.perf_counter()
            rs = run_robust_scan(
                ["SYM"], data_r, config,
                strategies=strats,
                n_mc_paths=10, n_shuffle_paths=5, n_bootstrap_paths=5,
            )
            times_r.append(time.perf_counter() - t0)
        best_r = min(times_r)
        combos_r = rs.total_combos
        if strats and len(strats) == 3:
            base_r = BASELINE["robust_scan_3strat"]
            sp = fmt_speedup(base_r, best_r)
        else:
            sp = "-"
        print(f"  {label:<18}  best={best_r:.3f}s  combos={combos_r:,}  {sp}")

    # ─────────────────────────────────────────────────
    # Phase 7: Scaling test (data size vs time)
    # ─────────────────────────────────────────────────
    print(f"\n[Phase 7] Scaling: robust_scan with 3 strategies")
    print("-" * 72)
    print(f"  {'Bars':>8}  {'Time(s)':>10}  {'Bars/s':>12}")
    print("-" * 72)

    for sz in [4000, 8760, 20000]:
        np.random.seed(42)
        c_s, o_s, h_s, l_s = gen_data(sz)
        data_s = {"SYM": {"c": c_s, "o": o_s, "h": h_s, "l": l_s}}
        t0 = time.perf_counter()
        run_robust_scan(
            ["SYM"], data_s, config,
            strategies=["MA", "RSI", "MACD"],
            n_mc_paths=10, n_shuffle_paths=5, n_bootstrap_paths=5,
        )
        t_s = time.perf_counter() - t0
        print(f"  {sz:>8,}  {t_s:>10.3f}  {sz / t_s:>12,.0f}")

    # ─────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("                      OPTIMIZATION SUMMARY")
    print("=" * 72)
    print(f"  {'Component':<30}  {'Baseline':>12}  {'Now':>12}  {'Speedup':>10}")
    print("-" * 72)

    # Re-measure key metrics for summary
    np.random.seed(42)
    c, o, h, lo = gen_data(8760)

    # scan
    t_sc = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        scan_all_kernels(c, o, h, lo, config, n_threads=1)
        t_sc.append(time.perf_counter() - t0)
    scan_now = min(t_sc)
    scan_base = BASELINE["scan_serial_8760"]["time"]
    print(f"  {'scan_all (8760, serial)':<30}  {scan_base:>10.4f}s  {scan_now:>10.4f}s  {fmt_speedup(scan_base, scan_now):>10}")

    # multi-TF
    t_mt = []
    for _ in range(50):
        t0 = time.perf_counter()
        backtest_multi_tf(
            {"1h": ("MA", (10, 50)), "4h": ("RSI", (14, 30, 70))},
            {"1h": df_1h, "4h": df_4h}, config, mode="trend_filter",
        )
        t_mt.append(time.perf_counter() - t0)
    mtf_now = min(t_mt) * 1000
    mtf_base = BASELINE["multi_tf_ms"]
    print(f"  {'multi_tf (trend_filter)':<30}  {mtf_base:>10.2f}ms {mtf_now:>10.2f}ms {fmt_speedup(mtf_base, mtf_now):>10}")

    # robust_scan 3-strat
    data_r2 = {"SYM": {"c": c, "o": o, "h": h, "l": lo}}
    t_rs = []
    for _ in range(2):
        t0 = time.perf_counter()
        run_robust_scan(["SYM"], data_r2, config, strategies=["MA", "RSI", "MACD"],
                        n_mc_paths=10, n_shuffle_paths=5, n_bootstrap_paths=5)
        t_rs.append(time.perf_counter() - t0)
    rs_now = min(t_rs)
    rs_base = BASELINE["robust_scan_3strat"]
    print(f"  {'robust_scan (3 strat)':<30}  {rs_base:>10.3f}s  {rs_now:>10.3f}s  {fmt_speedup(rs_base, rs_now):>10}")

    print("-" * 72)
    print("  Baseline = values recorded BEFORE prange/njit-fusion/O3 optimizations")
    print("=" * 72)


if __name__ == "__main__":
    run_bench()
