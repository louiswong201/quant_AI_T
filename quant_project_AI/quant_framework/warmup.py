"""
Pre-warm the Numba JIT cache for all 96 kernel functions.

Usage
-----
    python -m quant_framework.warmup

This triggers compilation of every ``@njit`` function in the kernels
module and writes the result to disk so that subsequent imports are
near-instant.  Typical run time: 3-10 minutes (once).
"""

from __future__ import annotations

import sys
import time

from .platform_config import configure

configure()

import numpy as np

from .backtest import BacktestConfig, KERNEL_NAMES, backtest
from .backtest.kernels import (
    config_to_kernel_costs,
    eval_kernel,
    eval_kernel_detailed,
    eval_kernel_position,
    scan_all_kernels,
)

_FIRST_PARAMS = {
    "MA": (10, 50),
    "RSI": (14, 30, 70),
    "MACD": (12, 26, 9),
    "Drift": (20, 0.60, 7),
    "RAMOM": (15, 15, 1.5, 0.5),
    "Turtle": (20, 10, 14, 2.0),
    "Bollinger": (20, 2.0),
    "Keltner": (20, 14, 2.0),
    "MultiFactor": (14, 20, 20, 0.60, 0.30),
    "VolRegime": (14, 0.020, 10, 50, 25, 70),
    "MESA": (0.5, 0.05),
    "KAMA": (10, 2, 30, 2.0, 14),
    "Donchian": (20, 14, 2.0),
    "ZScore": (25, 2.0, 0.5, 3.0),
    "MomBreak": (40, 0.02, 14, 2.0),
    "RegimeEMA": (14, 0.020, 10, 40, 80),
    "DualMom": (10, 40),
    "Consensus": (10, 50, 14, 20, 25, 70, 2),
}


def _make_tiny_data(n: int = 200):
    rng = np.random.RandomState(42)
    rets = rng.normal(0.0003, 0.02, n)
    c = np.ascontiguousarray(np.cumprod(1.0 + rets) * 100.0, dtype=np.float64)
    spread = rng.uniform(0.005, 0.02, n)
    h = np.ascontiguousarray(c * (1.0 + spread), dtype=np.float64)
    l = np.ascontiguousarray(c * (1.0 - spread), dtype=np.float64)
    o = np.ascontiguousarray(c * (1.0 + rng.uniform(-0.01, 0.01, n)), dtype=np.float64)
    return c, o, h, l


def warmup() -> None:
    """Compile and cache every Numba kernel."""
    c, o, h, l = _make_tiny_data()
    config = BacktestConfig.crypto()
    costs = config_to_kernel_costs(config)
    sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
    lev, dc = costs["lev"], costs["dc"]
    sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

    total = len(KERNEL_NAMES)
    t0 = time.perf_counter()

    for i, name in enumerate(KERNEL_NAMES, 1):
        params = _FIRST_PARAMS.get(name)
        if params is None:
            continue
        sys.stdout.write(f"\r  [{i}/{total}] Compiling {name:<14} ...")
        sys.stdout.flush()
        eval_kernel(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
        eval_kernel_detailed(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
        eval_kernel_position(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)

    sys.stdout.write(f"\r  [{total}/{total}] Compiling scan_all_kernels ...\n")
    sys.stdout.flush()
    scan_all_kernels(c, o, h, l, config)

    elapsed = time.perf_counter() - t0
    print(f"\n  Numba cache warmed: {total} strategies compiled in {elapsed:.1f}s")
    print("  Subsequent imports will skip JIT compilation.")


def main() -> None:
    print("=" * 60)
    print("  Quant Framework — Numba JIT Cache Warmup")
    print("=" * 60)
    warmup()
    print("=" * 60)


if __name__ == "__main__":
    main()
