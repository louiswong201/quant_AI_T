"""Tests for AdaptiveRegimeEnsemble and MicrostructureMomentum strategies."""

import numpy as np
import pandas as pd
import pytest

from quant_framework.strategy.adaptive_regime_ensemble import (
    AdaptiveRegimeEnsemble,
    _atr_kernel,
    _kama_kernel,
    _zscore_kernel,
)
from quant_framework.strategy.microstructure_momentum import (
    MicrostructureMomentum,
    _ofi_roc_kernel,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def large_ohlcv() -> pd.DataFrame:
    """300-bar synthetic OHLCV with trending + mean-reverting segments."""
    np.random.seed(123)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)
    trend = np.linspace(0, 10, n)
    noise = np.cumsum(np.random.randn(n) * 0.3)
    close = 100.0 + trend + noise
    close = np.maximum(close, 1.0)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    low = np.maximum(low, 0.5)
    open_ = close + np.random.randn(n) * 0.2
    open_ = np.maximum(open_, 0.5)
    volume = np.random.randint(5000, 50000, size=n).astype(float)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture()
def arrays_from_df(large_ohlcv):
    """Convert DataFrame to arrays dict for on_bar_fast."""
    return {
        "open": large_ohlcv["open"].values.astype(np.float64),
        "high": large_ohlcv["high"].values.astype(np.float64),
        "low": large_ohlcv["low"].values.astype(np.float64),
        "close": large_ohlcv["close"].values.astype(np.float64),
        "volume": large_ohlcv["volume"].values.astype(np.float64),
        "symbol": "TEST",
    }


# ── Numba kernel tests ──────────────────────────────────────────────

class TestKernels:
    def test_kama_kernel_output_shape(self):
        close = np.random.randn(100).cumsum() + 100
        result = _kama_kernel(close, 10, 2.0 / 3.0, 2.0 / 31.0)
        assert result.shape == (100,)
        assert np.isnan(result[0])
        assert not np.isnan(result[50])

    def test_zscore_kernel_zero_std(self):
        flat = np.full(50, 42.0)
        result = _zscore_kernel(flat, 10)
        assert result[15] == 0.0

    def test_zscore_kernel_positive_deviation(self):
        arr = np.arange(50, dtype=np.float64)
        result = _zscore_kernel(arr, 10)
        assert result[20] > 0

    def test_atr_kernel_output(self):
        high = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.float64)
        low = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.float64)
        close = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.float64)
        result = _atr_kernel(high, low, close, 3)
        assert not np.isnan(result[5])
        assert result[5] > 0

    def test_ofi_roc_kernel(self):
        ofi = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=np.float64)
        result = _ofi_roc_kernel(ofi, 3)
        assert np.isnan(result[0])
        assert not np.isnan(result[5])
        assert result[5] > 0


# ── AdaptiveRegimeEnsemble tests ─────────────────────────────────────

class TestAdaptiveRegimeEnsemble:
    def test_instantiation_defaults(self):
        s = AdaptiveRegimeEnsemble()
        assert s.name == "AdaptiveRegimeEnsemble"
        assert s.entry_threshold == 0.3
        assert s.sigmoid_scale == 5.0

    def test_fast_columns(self):
        s = AdaptiveRegimeEnsemble()
        assert set(s.fast_columns) == {"open", "high", "low", "close", "volume"}

    def test_min_lookback_set(self):
        s = AdaptiveRegimeEnsemble(vol_slow=60, yz_window=20)
        assert s.min_lookback >= 65

    def test_hold_during_warmup(self, arrays_from_df):
        s = AdaptiveRegimeEnsemble()
        sig = s.on_bar_fast(arrays_from_df, 5, pd.Timestamp("2023-01-06"))
        assert sig["action"] == "hold"

    def test_returns_valid_signal_format(self, large_ohlcv):
        s = AdaptiveRegimeEnsemble()
        sig = s.on_bar(large_ohlcv, pd.Timestamp("2024-06-01"))
        assert isinstance(sig, dict)
        assert "action" in sig
        assert sig["action"] in ("buy", "sell", "hold")

    def test_on_bar_fast_produces_signal(self, arrays_from_df):
        s = AdaptiveRegimeEnsemble()
        i = len(arrays_from_df["close"]) - 1
        sig = s.on_bar_fast(arrays_from_df, i, pd.Timestamp("2024-06-01"))
        assert sig is not None
        assert "action" in sig

    def test_buy_signal_has_required_fields(self, arrays_from_df):
        s = AdaptiveRegimeEnsemble(entry_threshold=0.0)
        for i in range(s.min_lookback, len(arrays_from_df["close"])):
            sig = s.on_bar_fast(arrays_from_df, i, pd.Timestamp("2024-01-01"))
            if sig["action"] == "buy":
                assert "symbol" in sig
                assert "shares" in sig
                assert sig["shares"] > 0
                return
        pass

    def test_inverse_vol_sizing(self):
        s = AdaptiveRegimeEnsemble(max_risk_pct=0.02)
        s.portfolio_value = 1_000_000
        low_vol = s._inverse_vol_size(100.0, 0.005)
        high_vol = s._inverse_vol_size(100.0, 0.05)
        assert low_vol > high_vol

    def test_inverse_vol_cap(self):
        s = AdaptiveRegimeEnsemble()
        s.portfolio_value = 100_000
        shares = s._inverse_vol_size(1.0, 0.001)
        assert shares * 1.0 <= 100_000 * 0.95

    def test_on_bar_dict_input(self, large_ohlcv):
        s = AdaptiveRegimeEnsemble()
        sig = s.on_bar({"TEST": large_ohlcv}, pd.Timestamp("2024-06-01"))
        assert isinstance(sig, dict)
        assert "action" in sig

    def test_trailing_stop_triggers_sell(self, arrays_from_df):
        s = AdaptiveRegimeEnsemble(entry_threshold=0.0, atr_stop_mult=0.001)
        sold = False
        for i in range(s.min_lookback, len(arrays_from_df["close"])):
            sig = s.on_bar_fast(arrays_from_df, i, pd.Timestamp("2024-01-01"))
            if sig["action"] == "buy":
                s.positions["TEST"] = sig["shares"]
                s.cash -= sig["shares"] * arrays_from_df["close"][i]
            elif sig["action"] == "sell":
                sold = True
                break
        assert sold or s.positions.get("TEST", 0) == 0


# ── MicrostructureMomentum tests ─────────────────────────────────────

class TestMicrostructureMomentum:
    def test_instantiation_defaults(self):
        s = MicrostructureMomentum()
        assert s.name == "MicrostructureMomentum"
        assert s.ofi_roc_period == 5
        assert s.vpin_buckets == 50

    def test_fast_columns(self):
        s = MicrostructureMomentum()
        assert set(s.fast_columns) == {"open", "high", "low", "close", "volume"}

    def test_min_lookback_set(self):
        s = MicrostructureMomentum(ofi_window=20, ofi_roc_period=5, vov_vol_window=20, vov_vov_window=20)
        assert s.min_lookback >= 45

    def test_hold_during_warmup(self, arrays_from_df):
        s = MicrostructureMomentum()
        sig = s.on_bar_fast(arrays_from_df, 5, pd.Timestamp("2023-01-06"))
        assert sig["action"] == "hold"

    def test_returns_valid_signal(self, large_ohlcv):
        s = MicrostructureMomentum()
        sig = s.on_bar(large_ohlcv, pd.Timestamp("2024-06-01"))
        assert isinstance(sig, dict)
        assert "action" in sig
        assert sig["action"] in ("buy", "sell", "hold")

    def test_on_bar_fast_produces_signal(self, arrays_from_df):
        s = MicrostructureMomentum()
        i = len(arrays_from_df["close"]) - 1
        sig = s.on_bar_fast(arrays_from_df, i, pd.Timestamp("2024-06-01"))
        assert sig is not None
        assert "action" in sig

    def test_vol_gate_blocks_extreme_vol(self, arrays_from_df):
        s = MicrostructureMomentum(vol_min=0.90, vol_max=0.99)
        for i in range(s.min_lookback, len(arrays_from_df["close"])):
            sig = s.on_bar_fast(arrays_from_df, i, pd.Timestamp("2024-01-01"))
            assert sig["action"] != "buy"

    def test_adaptive_stop_distance(self):
        s = MicrostructureMomentum(atr_base_mult=3.0, vov_tightener=2.0)
        wide = s._adaptive_stop_distance(1.0, 0.0)
        tight = s._adaptive_stop_distance(1.0, 0.01)
        assert wide >= tight

    def test_inverse_vol_sizing(self):
        s = MicrostructureMomentum(max_risk_pct=0.02)
        s.portfolio_value = 1_000_000
        low_vol = s._inverse_vol_size(100.0, 0.10)
        high_vol = s._inverse_vol_size(100.0, 0.50)
        assert low_vol > high_vol

    def test_on_bar_dict_input(self, large_ohlcv):
        s = MicrostructureMomentum()
        sig = s.on_bar({"SYM": large_ohlcv}, pd.Timestamp("2024-06-01"))
        assert isinstance(sig, dict)
        assert "action" in sig

    def test_ofi_reversal_exits(self, arrays_from_df):
        s = MicrostructureMomentum(entry_ofi_roc=0.0, vpin_threshold=0.0, vol_min=0.0, vol_max=10.0)
        exited = False
        for i in range(s.min_lookback, len(arrays_from_df["close"])):
            sig = s.on_bar_fast(arrays_from_df, i, pd.Timestamp("2024-01-01"))
            if sig["action"] == "buy":
                s.positions["TEST"] = sig["shares"]
                s.cash -= sig["shares"] * arrays_from_df["close"][i]
            elif sig["action"] == "sell":
                exited = True
                break
        assert exited or s.positions.get("TEST", 0) == 0

    def test_none_on_missing_arrays(self):
        s = MicrostructureMomentum()
        result = s.on_bar_fast({"close": np.ones(10)}, 5, pd.Timestamp("2024-01-01"))
        assert result is None
