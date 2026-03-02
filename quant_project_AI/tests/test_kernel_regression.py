"""
Numerical regression & consistency tests for Numba kernels.

Catches silent regressions from optimisation changes by verifying:
  - Reproducibility: same input → identical output across calls.
  - Consistency: eval_kernel() == eval_kernel_detailed() (ret, dd, nt).
  - Cost monotonicity: higher costs → lower (or equal) return.
  - CPCV split geometry: every data point used in train and test.
  - Portfolio API: weight arithmetic correct.
  - DataManager cache: instance-level LRU functions correctly.
"""

import numpy as np
import pytest

from quant_framework.backtest.config import BacktestConfig
from quant_framework.backtest.kernels import (
    KERNEL_NAMES,
    DEFAULT_PARAM_GRIDS,
    config_to_kernel_costs,
    eval_kernel,
    eval_kernel_detailed,
    eval_kernel_position,
    scan_all_kernels,
)
from quant_framework.backtest.robust_scan import cpcv_splits
from quant_framework.backtest import (
    backtest,
    backtest_portfolio,
    optimize,
    BacktestResult,
    PortfolioResult,
)


# ── Frozen reference outputs (seed=12345, n=500, crypto config) ──────

FROZEN_REFS = {
    "MA": (-14.274525, 25.368108, 50),
    "RSI": (4.428861, 12.007105, 40),
    "MACD": (-28.448944, 35.283643, 113),
    "Drift": (-21.276268, 25.510523, 110),
    "RAMOM": (-22.146410, 29.079973, 66),
    "Turtle": (-14.586789, 21.840357, 37),
    "Bollinger": (-10.551377, 18.886282, 57),
    "Keltner": (-35.442559, 41.122848, 50),
    "MultiFactor": (-22.428164, 30.893294, 6),
    "VolRegime": (-12.973031, 19.053744, 15),
    "MESA": (-21.401079, 28.000324, 11),
    "KAMA": (-32.327403, 33.630893, 62),
    "Donchian": (-22.502809, 26.816676, 39),
    "ZScore": (7.069520, 13.268927, 33),
    "MomBreak": (0.000000, 0.000000, 0),
    "RegimeEMA": (6.361727, 17.326001, 18),
    "DualMom": (-29.263234, 31.336029, 61),
    "Consensus": (-22.420405, 29.551237, 10),
}


# ── Deterministic synthetic data ─────────────────────────────────────

def _make_data(n=500, seed=12345):
    """Return deterministic (close, open, high, low) arrays."""
    rng = np.random.RandomState(seed)
    ret = rng.randn(n) * 0.01
    close = 100.0 * np.cumprod(1.0 + ret)
    spread = np.abs(rng.randn(n) * 0.003) * close
    high = close + spread
    low = close - spread
    open_ = close + rng.randn(n) * 0.001 * close
    for arr in (close, open_, high, low):
        arr[:] = np.ascontiguousarray(arr, dtype=np.float64)
    return close, open_, high, low


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ohlc():
    return _make_data()


@pytest.fixture(scope="module")
def config():
    return BacktestConfig.crypto()


@pytest.fixture(scope="module")
def costs(config):
    return config_to_kernel_costs(config)


# ── 1. Kernel reproducibility ───────────────────────────────────────

class TestKernelReproducibility:
    """Same input must produce bit-identical output on repeated calls."""

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_deterministic(self, name, ohlc, costs):
        c, o, h, l = ohlc
        sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
        lev, dc = costs["lev"], costs["dc"]
        sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

        params = DEFAULT_PARAM_GRIDS[name][0]

        r1, d1, t1 = eval_kernel(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
        r2, d2, t2 = eval_kernel(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)

        assert r1 == r2, f"{name}: return not reproducible"
        assert d1 == d2, f"{name}: drawdown not reproducible"
        assert t1 == t2, f"{name}: n_trades not reproducible"


# ── 2. eval_kernel vs eval_kernel_detailed consistency ──────────────

class TestDetailedConsistency:
    """eval_kernel and eval_kernel_detailed must agree on (ret, dd, nt)."""

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_match(self, name, ohlc, costs):
        c, o, h, l = ohlc
        sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
        lev, dc = costs["lev"], costs["dc"]
        sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

        params = DEFAULT_PARAM_GRIDS[name][0]

        r1, d1, t1 = eval_kernel(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
        r2, d2, t2, eq, fpos, _pos_arr = eval_kernel_detailed(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)

        assert abs(r1 - r2) < 0.01, f"{name}: ret mismatch {r1} vs {r2}"
        assert abs(d1 - d2) < 0.01, f"{name}: dd mismatch {d1} vs {d2}"
        assert t1 == t2, f"{name}: trades mismatch {t1} vs {t2}"
        assert eq is not None and len(eq) == len(c)
        assert fpos in (-1, 0, 1), f"{name}: invalid final_position {fpos}"

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_position_matches_detailed(self, name, ohlc, costs):
        """eval_kernel_position must return the same position as eval_kernel_detailed."""
        c, o, h, l = ohlc
        sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
        lev, dc = costs["lev"], costs["dc"]
        sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

        params = DEFAULT_PARAM_GRIDS[name][0]

        _, _, _, _, fpos_d, _ = eval_kernel_detailed(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
        fpos_p = eval_kernel_position(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)

        assert fpos_d == fpos_p, f"{name}: position mismatch {fpos_d} vs {fpos_p}"


# ── 2b. Frozen reference values ──────────────────────────────────────

class TestFrozenReferences:
    """Kernel outputs must match pre-recorded reference values exactly."""

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_frozen(self, name, ohlc, costs):
        c, o, h, l = ohlc
        sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
        lev, dc = costs["lev"], costs["dc"]
        sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

        params = DEFAULT_PARAM_GRIDS[name][0]
        r, d, t = eval_kernel(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)

        ref_r, ref_d, ref_t = FROZEN_REFS[name]
        assert abs(r - ref_r) < 1e-4, f"{name}: ret {r} != frozen {ref_r}"
        assert abs(d - ref_d) < 1e-4, f"{name}: dd {d} != frozen {ref_d}"
        assert t == ref_t, f"{name}: trades {t} != frozen {ref_t}"


# ── 3. Cost monotonicity ────────────────────────────────────────────

class TestCostMonotonicity:
    """Higher costs must produce lower (or equal) net return."""

    def test_higher_commission_lower_return(self, ohlc):
        c, o, h, l = ohlc
        cfg_lo = BacktestConfig(commission_pct_buy=0.0001, commission_pct_sell=0.0001)
        cfg_hi = BacktestConfig(commission_pct_buy=0.01, commission_pct_sell=0.01)

        params = DEFAULT_PARAM_GRIDS["MA"][0]
        co_lo = config_to_kernel_costs(cfg_lo)
        co_hi = config_to_kernel_costs(cfg_hi)

        r_lo, _, _ = eval_kernel("MA", params, c, o, h, l,
                                 co_lo["sb"], co_lo["ss"], co_lo["cm"],
                                 co_lo["lev"], co_lo["dc"],
                                 co_lo["sl"], co_lo["pfrac"], co_lo["sl_slip"])
        r_hi, _, _ = eval_kernel("MA", params, c, o, h, l,
                                 co_hi["sb"], co_hi["ss"], co_hi["cm"],
                                 co_hi["lev"], co_hi["dc"],
                                 co_hi["sl"], co_hi["pfrac"], co_hi["sl_slip"])

        assert r_hi <= r_lo + 0.01, "Higher commission must not improve return"


# ── 4. scan_all_kernels ─────────────────────────────────────────────

class TestScanAllKernels:
    """scan_all_kernels must return results for all 18 strategies."""

    def test_all_strategies_present(self, ohlc, config):
        c, o, h, l = ohlc
        results = scan_all_kernels(c, o, h, l, config)
        for name in KERNEL_NAMES:
            assert name in results, f"{name} missing from scan results"
            assert "params" in results[name]
            assert "ret" in results[name]
            assert "cnt" in results[name]
            assert results[name]["cnt"] > 0


# ── 5. CPCV split geometry ──────────────────────────────────────────

class TestCPCVSplits:
    """CPCV splits must be geometrically correct."""

    def test_all_bars_used_as_test(self):
        n = 600
        splits = cpcv_splits(n, n_groups=6, n_test_groups=2, embargo=0)
        covered = set()
        for _, test_ranges in splits:
            for s, e in test_ranges:
                covered.update(range(s, e))
        assert covered == set(range(n)), "Every bar must appear in at least one test fold"

    def test_n_splits_combinatorial(self):
        from math import comb
        splits = cpcv_splits(600, n_groups=6, n_test_groups=2, embargo=5)
        assert len(splits) == comb(6, 2)

    def test_no_train_test_overlap(self):
        splits = cpcv_splits(600, n_groups=6, n_test_groups=2, embargo=5)
        for train_ranges, test_ranges in splits:
            train_bars = set()
            for s, e in train_ranges:
                train_bars.update(range(s, e))
            test_bars = set()
            for s, e in test_ranges:
                test_bars.update(range(s, e))
            assert train_bars.isdisjoint(test_bars), "Train and test must not overlap"

    def test_embargo_purges_boundary(self):
        splits = cpcv_splits(600, n_groups=6, n_test_groups=2, embargo=10)
        group_size = 100
        for train_ranges, test_ranges in splits:
            for te_s, te_e in test_ranges:
                for tr_s, tr_e in train_ranges:
                    if tr_e <= te_s and tr_e > te_s - group_size:
                        assert tr_e <= te_s - 10, "Embargo not applied before test"


# ── 6. Portfolio backtest API ────────────────────────────────────────

class TestPortfolioBacktest:
    def test_equal_weight_portfolio(self, ohlc):
        c, o, h, l = ohlc
        data = {
            "A": {"c": c, "o": o, "h": h, "l": l},
            "B": {"c": c * 1.01, "o": o * 1.01, "h": h * 1.01, "l": l * 1.01},
        }
        allocations = {"A": ("MA", (10, 50)), "B": ("RSI", (14, 30, 70))}
        result = backtest_portfolio(allocations, data)
        assert isinstance(result, PortfolioResult)
        assert len(result.per_asset) == 2
        assert "A" in result.per_asset
        assert "B" in result.per_asset
        assert result.portfolio_equity is not None
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_custom_weights(self, ohlc):
        c, o, h, l = ohlc
        data = {"X": {"c": c, "o": o, "h": h, "l": l}}
        allocations = {"X": ("MA", (10, 50))}
        result = backtest_portfolio(allocations, data, weights={"X": 1.0})
        assert result.weights["X"] == 1.0


# ── 7. backtest() detailed mode ─────────────────────────────────────

class TestBacktestDetailed:
    def test_equity_curve_length(self, ohlc):
        c, o, h, l = ohlc
        data = {"c": c, "o": o, "h": h, "l": l}
        result = backtest("MA", (10, 50), data, detailed=True)
        assert isinstance(result, BacktestResult)
        assert result.equity is not None
        assert len(result.equity) == len(c)
        assert result.sharpe != 0.0 or result.ret_pct == pytest.approx(0.0, abs=1.0)


# ── 9. KernelAdapter — kernel delegation ────────────────────────────

class TestKernelAdapter:
    def test_signal_generation(self, ohlc):
        import pandas as pd
        from quant_framework.live.kernel_adapter import KernelAdapter

        c, o, h, l = ohlc
        df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
        adapter = KernelAdapter("MA")
        signal = adapter.generate_signal(df, "TEST")
        assert signal is None or signal["action"] in ("buy", "sell")

    def test_unknown_strategy_raises(self):
        from quant_framework.live.kernel_adapter import KernelAdapter
        with pytest.raises(ValueError, match="Unknown"):
            KernelAdapter("NONEXISTENT_STRATEGY")

    def test_position_tracks_correctly(self, ohlc):
        import pandas as pd
        from quant_framework.live.kernel_adapter import KernelAdapter

        c, o, h, l = ohlc
        df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
        adapter = KernelAdapter("MA", params=(5, 20))
        signals = []
        for end in range(30, len(c)):
            sig = adapter.generate_signal(df.iloc[:end + 1], "TEST")
            if sig is not None:
                signals.append(sig["action"])
        for s in signals:
            assert s in ("buy", "sell")


# ── 10. optimize(method="cpcv") ─────────────────────────────────────

class TestOptimizeCPCV:
    def test_cpcv_method(self, ohlc):
        c, o, h, l = ohlc
        data = {"c": c, "o": o, "h": h, "l": l}
        result = optimize(data, strategies=["MA", "RSI"], method="cpcv")
        assert result.best.strategy in ("MA", "RSI")
        assert result.total_combos > 0
        assert result.elapsed_seconds > 0


# ── 8. DataManager cache ────────────────────────────────────────────

class TestDataManagerCache:
    def test_instance_level_cache_independence(self):
        """Two DataManager instances must not share cache state."""
        from collections import OrderedDict
        from quant_framework.data.data_manager import DataManager

        class StubDataset:
            def __init__(self):
                self.call_count = 0
            def load(self, symbol, start, end):
                self.call_count += 1
                return None

        dm1 = DataManager.__new__(DataManager)
        dm1.dataset = StubDataset()
        dm1._cache = None
        dm1._local_cache = OrderedDict()

        dm2 = DataManager.__new__(DataManager)
        dm2.dataset = StubDataset()
        dm2._cache = None
        dm2._local_cache = OrderedDict()

        dm1.load_data("SYM", "2024-01-01", "2024-12-31")
        dm1.load_data("SYM", "2024-01-01", "2024-12-31")

        assert dm1.dataset.call_count == 1, "Second call should hit cache"
        assert dm2.dataset.call_count == 0, "dm2 must not share dm1's cache"

    def test_cache_invalidation_on_save(self):
        from collections import OrderedDict
        import pandas as pd
        from quant_framework.data.data_manager import DataManager

        class StubDataset:
            def __init__(self):
                self.call_count = 0
            def load(self, symbol, start, end):
                self.call_count += 1
                return pd.DataFrame({"close": [1.0]})
            def save(self, symbol, data):
                pass

        dm = DataManager.__new__(DataManager)
        dm.dataset = StubDataset()
        dm._cache = None
        dm._local_cache = OrderedDict()

        dm.load_data("BTC", "2024-01-01", "2024-12-31")
        assert dm.dataset.call_count == 1

        dm.save_data("BTC", pd.DataFrame())
        dm.load_data("BTC", "2024-01-01", "2024-12-31")
        assert dm.dataset.call_count == 2, "Cache must be invalidated after save"
