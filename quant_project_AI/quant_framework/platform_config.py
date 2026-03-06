"""
Cross-platform configuration for optimal Numba and threading behaviour.

Call ``configure()`` as early as possible — before importing any Numba code —
to set environment variables that control JIT compilation, threading layer,
and thread counts.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIGURED = False


def configure() -> None:
    """Set Numba environment variables for the current platform.

    Safe to call multiple times; only the first call has effect.
    Must be called **before** ``import numba``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    system = platform.system()
    n_cpus = os.cpu_count() or 4

    if system == "Windows":
        n_threads = max(1, n_cpus // 2)
        os.environ.setdefault("NUMBA_NUM_THREADS", str(n_threads))

        # TBB is preferred; fall back to workqueue if unavailable
        try:
            import tbb  # noqa: F401
            os.environ.setdefault("NUMBA_THREADING_LAYER", "tbb")
        except ImportError:
            os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
            logger.info("TBB not installed — using workqueue threading layer")

        # Isolate Numba cache from Windows Defender real-time scanning
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            cache_dir = Path(local_app) / "quant_framework" / "numba_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))

        logger.info(
            "Platform: Windows — threads=%d, layer=%s, cache=%s",
            n_threads,
            os.environ.get("NUMBA_THREADING_LAYER", "?"),
            os.environ.get("NUMBA_CACHE_DIR", "default"),
        )
    else:
        os.environ.setdefault("NUMBA_NUM_THREADS", str(n_cpus))


def optimal_thread_config() -> dict:
    """Return a dict with recommended thread counts for each subsystem.

    Keys
    ----
    numba_threads : int
        Threads for Numba ``prange`` loops (already applied via env var).
    robustness_workers : int
        ThreadPoolExecutor workers for MC / shuffle / bootstrap tasks.
    scan_workers : int
        ThreadPoolExecutor workers for ``scan_all_kernels`` (usually 1
        because ``prange`` handles parallelism internally).
    """
    n_cpus = os.cpu_count() or 4
    if platform.system() == "Windows":
        n_physical = max(1, n_cpus // 2)
    else:
        n_physical = n_cpus

    return {
        "numba_threads": n_physical,
        "robustness_workers": max(1, n_physical - 1),
        "scan_workers": 1,
    }


def check_numba_cache(full: bool = True) -> bool:
    """Return True if Numba cache files exist for kernels and robustness.

    Parameters
    ----------
    full : bool
        When *True* (default), require both ``kernels.*.nbi`` and
        ``robust_scan.*.nbi`` to be present.  When *False*, only check
        kernels (legacy behaviour).

    Logs a warning when the cache is missing — the first run will need
    to JIT-compile ~96+ functions which can take several minutes.
    """
    custom = os.environ.get("NUMBA_CACHE_DIR")
    if custom:
        cache_dir = Path(custom)
    else:
        cache_dir = Path(__file__).parent / "backtest" / "__pycache__"

    if not cache_dir.is_dir():
        logger.warning(
            "Numba cache directory not found — first run will JIT-compile "
            "~96+ functions (may take 5-10 min on older CPUs).  "
            "Run  python -m quant_framework.warmup  to pre-compile."
        )
        return False

    has_kernels = any(cache_dir.glob("kernels.*.nbi"))
    has_robust = any(cache_dir.glob("robust_scan.*.nbi"))

    if has_kernels and (has_robust or not full):
        return True

    missing = []
    if not has_kernels:
        missing.append("kernels")
    if full and not has_robust:
        missing.append("robust_scan")

    logger.warning(
        "Numba cache incomplete (missing: %s) — first run will JIT-compile. "
        "Run  python -m quant_framework.warmup  to pre-compile.",
        ", ".join(missing),
    )
    return False
