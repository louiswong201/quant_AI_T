"""
Pre-warm the Numba JIT cache for all kernel + robustness functions.

Usage
-----
    python -m quant_framework.warmup          # interactive
    python -m quant_framework.warmup --quiet  # silent (for CI / post-install)

Covers:
  - 96 kernel functions (eval_kernel, eval_kernel_detailed, scan_all_kernels)
  - 3 robustness helpers (perturb_ohlc, shuffle_ohlc, block_bootstrap_ohlc)
  - Mini Walk-Forward run (ensures all type-specialisations are cached)

Also provides ``warmup_if_needed()`` / ``background_warmup()`` for
automatic cache rebuilding on import.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

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

_WARMUP_THREAD: Optional[threading.Thread] = None
_WARMUP_LOCK = threading.Lock()
_WARMUP_ACTIVE = False


def _make_tiny_data(n: int = 200):
    rng = np.random.RandomState(42)
    rets = rng.normal(0.0003, 0.02, n)
    c = np.ascontiguousarray(np.cumprod(1.0 + rets) * 100.0, dtype=np.float64)
    spread = rng.uniform(0.005, 0.02, n)
    h = np.ascontiguousarray(c * (1.0 + spread), dtype=np.float64)
    l = np.ascontiguousarray(c * (1.0 - spread), dtype=np.float64)
    o = np.ascontiguousarray(c * (1.0 + rng.uniform(-0.01, 0.01, n)), dtype=np.float64)
    return c, o, h, l


def _is_cache_complete() -> bool:
    """Check if both kernel AND robustness caches exist on disk."""
    custom = os.environ.get("NUMBA_CACHE_DIR")
    if custom:
        cache_dir = Path(custom)
    else:
        cache_dir = Path(__file__).parent / "backtest" / "__pycache__"

    if not cache_dir.is_dir():
        return False
    has_kernels = any(cache_dir.glob("kernels.*.nbi"))
    has_robust = any(cache_dir.glob("robust_scan.*.nbi"))
    return has_kernels and has_robust


def _full_warmup(verbose: bool = True) -> None:
    """Compile and cache every Numba function (kernels + robustness)."""
    global _WARMUP_ACTIVE
    with _WARMUP_LOCK:
        if _WARMUP_ACTIVE:
            return
        _WARMUP_ACTIVE = True

    try:
        _do_warmup(verbose)
    finally:
        with _WARMUP_LOCK:
            _WARMUP_ACTIVE = False


def _do_warmup(verbose: bool = True) -> None:
    """Internal: actual compilation work (must only run in one thread).

    Cache files are thread-count-independent and will work at full
    parallelism at runtime.
    """
    from .platform_config import configure
    configure()

    from .backtest import BacktestConfig, KERNEL_NAMES
    from .backtest.kernels import (
        config_to_kernel_costs,
        eval_kernel,
        eval_kernel_detailed,
        eval_kernel_position,
        scan_all_kernels,
    )
    from .backtest.robust_scan import (
        perturb_ohlc,
        shuffle_ohlc,
        block_bootstrap_ohlc,
    )

    c, o, h, l = _make_tiny_data()
    config = BacktestConfig.crypto()
    costs = config_to_kernel_costs(config)
    sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
    lev, dc = costs["lev"], costs["dc"]
    sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

    total = len(KERNEL_NAMES)
    t0 = time.perf_counter()

    # Phase 1: kernel layer
    for i, name in enumerate(KERNEL_NAMES, 1):
        params = _FIRST_PARAMS.get(name)
        if params is None:
            continue
        if verbose:
            sys.stdout.write(f"\r  [{i}/{total}] Compiling {name:<14} ...")
            sys.stdout.flush()
        eval_kernel(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
        eval_kernel_detailed(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
        eval_kernel_position(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)

    if verbose:
        sys.stdout.write(f"\r  [{total}/{total}] Compiling scan_all_kernels ...\n")
        sys.stdout.flush()
    scan_all_kernels(c, o, h, l, config)

    # Phase 2: robustness layer
    if verbose:
        sys.stdout.write("  Compiling robustness helpers (perturb/shuffle/bootstrap) ...\n")
        sys.stdout.flush()
    perturb_ohlc(c, o, h, l, 0.002, 42)
    shuffle_ohlc(c, o, h, l, 42)
    block_bootstrap_ohlc(c, o, h, l, 20, 42)

    # Phase 3: mini WF run to warm all eval_kernel type-specialisations
    # inside the robustness dispatch path
    if verbose:
        sys.stdout.write("  Compiling Walk-Forward robustness pipeline ...\n")
        sys.stdout.flush()
    try:
        from .backtest.robust_scan import run_robust_scan
        data_w = {"W": {"c": c, "o": o, "h": h, "l": l}}
        run_robust_scan(
            ["W"], data_w, config, strategies=["MA"],
            n_mc_paths=1, n_shuffle_paths=1, n_bootstrap_paths=1,
        )
    except Exception as exc:
        if verbose:
            sys.stdout.write(f"  (WF warmup skipped: {exc})\n")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\n  Numba cache warmed: {total} strategies + robustness in {elapsed:.1f}s")
        print("  Subsequent runs will skip JIT compilation.")
    else:
        logger.info("Numba cache warmed in %.1fs", elapsed)


def warmup() -> None:
    """Interactive warmup — compiles and caches every Numba function."""
    _full_warmup(verbose=True)


def warmup_if_needed() -> bool:
    """Check cache completeness; start background warmup if incomplete.

    Returns ``True`` if the cache is already warm, ``False`` if a
    background compilation was started.  Non-blocking.
    """
    if _is_cache_complete():
        return True
    if os.environ.get("_QUANT_WARMUP_SUBPROCESS") == "1":
        return False
    background_warmup()
    return False


def background_warmup(verbose: bool = False) -> None:
    """Launch warmup in a background subprocess (non-blocking).

    Uses a subprocess (not a thread) because the Numba workqueue layer
    is not thread-safe and cannot be forced to serial mode once
    initialized in the main process.  The subprocess inherits
    ``NUMBA_NUM_THREADS=1`` to avoid workqueue crashes.

    Safe to call multiple times; only one warmup runs at a time.
    """
    global _WARMUP_THREAD
    with _WARMUP_LOCK:
        if _WARMUP_THREAD is not None and _WARMUP_THREAD.is_alive():
            return
        logger.info("Numba cache incomplete — starting background warmup subprocess")
        _WARMUP_THREAD = threading.Thread(
            target=_background_subprocess_warmup, daemon=True,
            kwargs={"verbose": verbose},
        )
        _WARMUP_THREAD.start()


def _background_subprocess_warmup(verbose: bool = False) -> None:
    """Run warmup in a subprocess with NUMBA_NUM_THREADS=1."""
    import subprocess
    env = os.environ.copy()
    env["NUMBA_NUM_THREADS"] = "1"
    env["_QUANT_WARMUP_SUBPROCESS"] = "1"
    cmd = [sys.executable, "-m", "quant_framework.warmup"]
    if not verbose:
        cmd.append("--quiet")
    try:
        subprocess.run(cmd, env=env, capture_output=not verbose, timeout=600)
    except Exception as exc:
        logger.warning("Background warmup failed: %s", exc)


def wait_for_warmup(timeout: float = 300) -> None:
    """Block until background warmup completes (or *timeout* seconds)."""
    t = _WARMUP_THREAD
    if t is not None and t.is_alive():
        logger.info("Waiting for Numba warmup to complete (timeout=%.0fs)...", timeout)
        t.join(timeout=timeout)


def main() -> None:
    # Force serial Numba execution during warmup to avoid workqueue
    # threading-layer crashes on macOS / systems without TBB.
    os.environ["NUMBA_NUM_THREADS"] = "1"

    print("=" * 60)
    print("  Quant Framework — Numba JIT Cache Warmup")
    print("=" * 60)
    quiet = "--quiet" in sys.argv or "-q" in sys.argv
    _full_warmup(verbose=not quiet)
    print("=" * 60)


if __name__ == "__main__":
    main()
