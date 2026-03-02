"""Benchmark script for backtest engine optimizations."""
import time
import numpy as np

np.random.seed(42)
N = 8760
close = 100.0 * np.cumprod(1 + np.random.randn(N) * 0.02)
open_ = close * (1 + np.random.randn(N) * 0.005)
high = np.maximum(close, open_) * (1 + np.abs(np.random.randn(N)) * 0.01)
low = np.minimum(close, open_) * (1 - np.abs(np.random.randn(N)) * 0.01)

from quant_framework.backtest.config import BacktestConfig
from quant_framework.backtest.kernels import scan_all_kernels, eval_kernel, config_to_kernel_costs

config = BacktestConfig.crypto()

print("=" * 60)
print("Backtest Engine Benchmark")
print(f"Data: {N} bars")
print("=" * 60)

# Warmup JIT
print("\n[1] JIT warmup (first call compiles)...")
t0 = time.perf_counter()
_ = scan_all_kernels(close, open_, high, low, config, n_threads=1)
t_warmup = time.perf_counter() - t0
print(f"    Warmup: {t_warmup:.2f}s")

# Benchmark scan_all_kernels (serial)
print("\n[2] scan_all_kernels (serial, n_threads=1)...")
times = []
for i in range(3):
    t0 = time.perf_counter()
    R = scan_all_kernels(close, open_, high, low, config, n_threads=1)
    times.append(time.perf_counter() - t0)
total_combos = sum(r["cnt"] for r in R.values())
best_t = min(times)
print(f"    Best of 3: {best_t:.4f}s")
print(f"    Total combos: {total_combos:,}")
print(f"    Throughput: {total_combos / best_t:,.0f} combos/s")

# Benchmark scan_all_kernels (threaded)
print("\n[3] scan_all_kernels (threaded, default)...")
times = []
for i in range(3):
    t0 = time.perf_counter()
    R2 = scan_all_kernels(close, open_, high, low, config)
    times.append(time.perf_counter() - t0)
best_t2 = min(times)
print(f"    Best of 3: {best_t2:.4f}s")
print(f"    Throughput: {total_combos / best_t2:,.0f} combos/s")

# Benchmark eval_kernel (used by robust_scan MC/Bootstrap)
print("\n[4] eval_kernel x100 (simulates MC/Bootstrap)...")
costs = config_to_kernel_costs(config)
sb, ss, cm, lev, dc = costs["sb"], costs["ss"], costs["cm"], costs["lev"], costs["dc"]
sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]
best_ma = R.get("MA", {}).get("params", (10, 50))
t0 = time.perf_counter()
for _ in range(100):
    eval_kernel("MA", best_ma, close, open_, high, low, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
t_eval = time.perf_counter() - t0
print(f"    100 calls: {t_eval:.4f}s ({t_eval/100*1000:.2f}ms/call)")

# Benchmark Multi-TF fusion
print("\n[5] Multi-TF backtest...")
from quant_framework.backtest import backtest_multi_tf
df_1h = {"c": close, "o": open_, "h": high, "l": low, "timestamps": np.arange(N, dtype=np.float64) * 3600}
N_4h = N // 4
c4 = close[::4][:N_4h]; o4 = open_[::4][:N_4h]
h4 = high[::4][:N_4h]; l4 = low[::4][:N_4h]
df_4h = {"c": c4, "o": o4, "h": h4, "l": l4, "timestamps": np.arange(N_4h, dtype=np.float64) * 14400}
t0 = time.perf_counter()
for _ in range(10):
    backtest_multi_tf(
        {"1h": ("MA", (10, 50)), "4h": ("RSI", (14, 30, 70))},
        {"1h": df_1h, "4h": df_4h},
        config, mode="trend_filter",
    )
t_mtf = time.perf_counter() - t0
print(f"    10 calls: {t_mtf:.4f}s ({t_mtf/10*1000:.2f}ms/call)")

# Benchmark robust_scan (1 symbol, subset of strategies)
print("\n[6] run_robust_scan (3 strategies, 1 symbol)...")
from quant_framework.backtest.robust_scan import run_robust_scan
data = {"SYM": {"c": close, "o": open_, "h": high, "l": low}}
# warmup run
_ = run_robust_scan(["SYM"], data, config, strategies=["MA"], n_mc_paths=2, n_shuffle_paths=1, n_bootstrap_paths=1)
t0 = time.perf_counter()
rs = run_robust_scan(
    ["SYM"], data, config,
    strategies=["MA", "RSI", "MACD"],
    n_mc_paths=10, n_shuffle_paths=5, n_bootstrap_paths=5,
)
t_robust = time.perf_counter() - t0
print(f"    Time: {t_robust:.2f}s")
print(f"    Combos: {rs.total_combos:,}")

print("\n" + "=" * 60)
print("Benchmark complete.")
print("=" * 60)
