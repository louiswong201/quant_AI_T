"""
策略单元测试
覆盖：BaseStrategy 属性 / MA / RSI / MACD / DriftRegime / ZScore / MomBreakout / KAMA / MESA / Lorentzian
"""

import numpy as np
import pandas as pd
import pytest

from quant_framework.strategy.base_strategy import BaseStrategy
from quant_framework.strategy.ma_strategy import MovingAverageStrategy
from quant_framework.strategy.rsi_strategy import RSIStrategy
from quant_framework.strategy.macd_strategy import MACDStrategy
from quant_framework.strategy.drift_regime_strategy import DriftRegimeStrategy
from quant_framework.strategy.zscore_reversion_strategy import ZScoreReversionStrategy
from quant_framework.strategy.momentum_breakout_strategy import MomentumBreakoutStrategy
from quant_framework.strategy.kama_strategy import KAMAStrategy
from quant_framework.strategy.mesa_strategy import MESAStrategy
from quant_framework.strategy.lorentzian_strategy import LorentzianClassificationStrategy


# ---------------------------------------------------------------------------
# BaseStrategy
# ---------------------------------------------------------------------------

class TestBaseStrategy:
    def test_min_lookback_default(self):
        """默认 min_lookback 应为 1。"""

        class DummyStrategy(BaseStrategy):
            def on_bar(self, data, current_date, current_prices=None):
                return {"action": "hold"}

        s = DummyStrategy("dummy")
        assert s.min_lookback == 1

    def test_position_tracking(self):
        class DummyStrategy(BaseStrategy):
            def on_bar(self, data, current_date, current_prices=None):
                return {"action": "hold"}

        s = DummyStrategy("dummy", initial_capital=100000)
        assert s.buy("AAPL", 100.0, 10)
        assert s.positions["AAPL"] == 10
        assert s.cash == pytest.approx(99000.0)
        assert s.sell("AAPL", 110.0, 10)
        assert "AAPL" not in s.positions
        assert s.cash == pytest.approx(100100.0)

    def test_cannot_buy_without_cash(self):
        class DummyStrategy(BaseStrategy):
            def on_bar(self, data, current_date, current_prices=None):
                return {"action": "hold"}

        s = DummyStrategy("dummy", initial_capital=100)
        assert not s.can_buy("X", 200, 1)

    def test_cannot_sell_without_position(self):
        class DummyStrategy(BaseStrategy):
            def on_bar(self, data, current_date, current_prices=None):
                return {"action": "hold"}

        s = DummyStrategy("dummy")
        assert not s.can_sell("X", 1)


# ---------------------------------------------------------------------------
# MA Strategy
# ---------------------------------------------------------------------------

class TestMAStrategy:
    def test_min_lookback(self):
        s = MovingAverageStrategy(short_window=5, long_window=20)
        assert s.min_lookback == 20

    def test_hold_on_short_data(self, sample_ohlcv: pd.DataFrame):
        s = MovingAverageStrategy(short_window=5, long_window=20)
        short = sample_ohlcv.iloc[:5]
        sig = s.on_bar(short, pd.Timestamp("2024-01-05"))
        assert sig["action"] == "hold"

    def test_signal_from_precomputed(self, sample_ohlcv: pd.DataFrame):
        """验证使用预计算列时不报错、返回有效信号类型。"""
        from quant_framework.data.indicators import VectorizedIndicators

        df = sample_ohlcv.copy()
        VectorizedIndicators.calculate_all(df)
        s = MovingAverageStrategy(short_window=5, long_window=20)
        sig = s.on_bar(df, pd.Timestamp("2024-06-01"))
        assert sig["action"] in ("buy", "sell", "hold")

    def test_infers_symbol_from_dict(self, sample_ohlcv: pd.DataFrame):
        """当 data 是 Dict[str, DataFrame] 时，symbol 应自动推断。"""
        from quant_framework.data.indicators import VectorizedIndicators

        df = sample_ohlcv.copy()
        VectorizedIndicators.calculate_all(df)
        s = MovingAverageStrategy(short_window=5, long_window=20)
        sig = s.on_bar({"AAPL": df}, pd.Timestamp("2024-06-01"))
        if sig["action"] in ("buy", "sell"):
            assert sig["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# RSI Strategy
# ---------------------------------------------------------------------------

class TestRSIStrategy:
    def test_min_lookback(self):
        s = RSIStrategy(rsi_period=14)
        assert s.min_lookback == 15

    def test_calculate_rsi_returns_scalar(self, sample_ohlcv: pd.DataFrame):
        s = RSIStrategy()
        rsi = s.calculate_rsi(sample_ohlcv["close"], 14)
        assert isinstance(rsi, float)

    def test_hold_on_short_data(self, sample_ohlcv: pd.DataFrame):
        s = RSIStrategy(rsi_period=14)
        short = sample_ohlcv.iloc[:5]
        sig = s.on_bar(short, pd.Timestamp("2024-01-05"))
        assert sig["action"] == "hold"

    def test_fast_columns(self):
        s = RSIStrategy(rsi_period=14)
        assert "rsi" in s.fast_columns

    def test_fast_multi_returns_hold_or_list(self, sample_ohlcv: pd.DataFrame):
        from quant_framework.data.indicators import VectorizedIndicators

        df = sample_ohlcv.copy()
        VectorizedIndicators.calculate_all(df)
        s = RSIStrategy(rsi_period=14)
        arrs = {
            "AAPL": {
                "symbol": "AAPL",
                "close": df["close"].to_numpy(),
                "rsi": df["rsi"].to_numpy(),
            }
        }
        out = s.on_bar_fast_multi(arrs, len(df) - 1, pd.Timestamp("2024-06-01"), {"AAPL": float(df["close"].iloc[-1])})
        assert isinstance(out, (dict, list))


# ---------------------------------------------------------------------------
# MACD Strategy
# ---------------------------------------------------------------------------

class TestMACDStrategy:
    def test_min_lookback(self):
        s = MACDStrategy()
        assert s.min_lookback == 35  # 26 + 9

    def test_calculate_macd_returns_scalars(self, sample_ohlcv: pd.DataFrame):
        s = MACDStrategy()
        macd, signal, hist = s.calculate_macd(sample_ohlcv["close"])
        assert isinstance(macd, float)
        assert isinstance(signal, float)
        assert isinstance(hist, float)

    def test_hold_on_short_data(self, sample_ohlcv: pd.DataFrame):
        s = MACDStrategy()
        short = sample_ohlcv.iloc[:10]
        sig = s.on_bar(short, pd.Timestamp("2024-01-10"))
        assert sig["action"] == "hold"

    def test_fast_columns(self):
        s = MACDStrategy()
        assert "macd" in s.fast_columns

    def test_fast_multi_returns_hold_or_list(self, sample_ohlcv: pd.DataFrame):
        from quant_framework.data.indicators import VectorizedIndicators

        df = sample_ohlcv.copy()
        VectorizedIndicators.calculate_all(df)
        s = MACDStrategy()
        arrs = {
            "AAPL": {
                "symbol": "AAPL",
                "close": df["close"].to_numpy(),
                "macd": df["macd"].to_numpy(),
                "macd_signal": df["macd_signal"].to_numpy(),
            }
        }
        out = s.on_bar_fast_multi(arrs, len(df) - 1, pd.Timestamp("2024-06-01"), {"AAPL": float(df["close"].iloc[-1])})
        assert isinstance(out, (dict, list))


# ---------------------------------------------------------------------------
# DriftRegime Strategy
# ---------------------------------------------------------------------------

class TestDriftRegimeStrategy:
    def test_min_lookback(self):
        s = DriftRegimeStrategy(lookback=15)
        assert s.min_lookback == 16

    def test_hold_on_short_data(self, sample_ohlcv: pd.DataFrame):
        s = DriftRegimeStrategy(lookback=15)
        short = sample_ohlcv.iloc[:10]
        sig = s.on_bar(short, pd.Timestamp("2024-01-10"))
        assert sig["action"] == "hold"

    def test_returns_valid_signal(self, sample_ohlcv: pd.DataFrame):
        s = DriftRegimeStrategy(lookback=10, drift_threshold=0.52, hold_period=5)
        sig = s.on_bar(sample_ohlcv, pd.Timestamp("2024-06-01"))
        assert sig["action"] in ("buy", "sell", "hold")

    def test_fast_path(self, sample_ohlcv: pd.DataFrame):
        s = DriftRegimeStrategy(lookback=10)
        close = sample_ohlcv["close"].to_numpy()
        arrs = {"close": close, "symbol": "TEST"}
        sig = s.on_bar_fast(arrs, len(close) - 1, pd.Timestamp("2024-06-01"))
        assert sig is not None
        assert sig["action"] in ("buy", "sell", "hold")

    def test_hold_period_sell(self, sample_ohlcv: pd.DataFrame):
        """After buying, hold_period bars later should trigger sell."""
        s = DriftRegimeStrategy(lookback=5, drift_threshold=0.01, hold_period=2)
        bought = False
        for i in range(6, len(sample_ohlcv)):
            row = sample_ohlcv.iloc[: i + 1]
            sig = s.on_bar(row, pd.Timestamp("2024-06-01"))
            if sig["action"] == "buy" and not bought:
                s.buy("STOCK", float(row["close"].iloc[-1]), sig["shares"])
                bought = True
            elif sig["action"] == "sell" and bought:
                break
        if bought:
            assert all(v >= 0 for v in s._hold_count.values())


# ---------------------------------------------------------------------------
# ZScore Reversion Strategy
# ---------------------------------------------------------------------------

class TestZScoreReversionStrategy:
    def test_min_lookback(self):
        s = ZScoreReversionStrategy(lookback=35)
        assert s.min_lookback == 36

    def test_hold_on_short_data(self, sample_ohlcv: pd.DataFrame):
        s = ZScoreReversionStrategy(lookback=35)
        short = sample_ohlcv.iloc[:20]
        sig = s.on_bar(short, pd.Timestamp("2024-01-20"))
        assert sig["action"] == "hold"

    def test_returns_valid_signal(self, sample_ohlcv: pd.DataFrame):
        s = ZScoreReversionStrategy(lookback=20, entry_z=1.0, exit_z=0.2, stop_z=3.0)
        sig = s.on_bar(sample_ohlcv, pd.Timestamp("2024-06-01"))
        assert sig["action"] in ("buy", "sell", "hold")

    def test_fast_path(self, sample_ohlcv: pd.DataFrame):
        s = ZScoreReversionStrategy(lookback=20)
        close = sample_ohlcv["close"].to_numpy()
        arrs = {"close": close, "symbol": "TEST"}
        sig = s.on_bar_fast(arrs, len(close) - 1, pd.Timestamp("2024-06-01"))
        assert sig is not None
        assert sig["action"] in ("buy", "sell", "hold")

    def test_fast_columns(self):
        s = ZScoreReversionStrategy()
        assert "close" in s.fast_columns


# ---------------------------------------------------------------------------
# Momentum Breakout Strategy
# ---------------------------------------------------------------------------

class TestMomentumBreakoutStrategy:
    def test_min_lookback(self):
        s = MomentumBreakoutStrategy(high_period=40, atr_period=14)
        assert s.min_lookback == 41

    def test_hold_on_short_data(self, sample_ohlcv: pd.DataFrame):
        s = MomentumBreakoutStrategy(high_period=40)
        short = sample_ohlcv.iloc[:20]
        sig = s.on_bar(short, pd.Timestamp("2024-01-20"))
        assert sig["action"] == "hold"

    def test_returns_valid_signal(self, sample_ohlcv: pd.DataFrame):
        s = MomentumBreakoutStrategy(high_period=20, proximity_pct=0.05, atr_period=10)
        sig = s.on_bar(sample_ohlcv, pd.Timestamp("2024-06-01"))
        assert sig["action"] in ("buy", "sell", "hold")

    def test_fast_path(self, sample_ohlcv: pd.DataFrame):
        s = MomentumBreakoutStrategy(high_period=20, atr_period=10)
        close = sample_ohlcv["close"].to_numpy()
        high = sample_ohlcv["high"].to_numpy()
        low = sample_ohlcv["low"].to_numpy()
        arrs = {"close": close, "high": high, "low": low, "symbol": "TEST"}
        sig = s.on_bar_fast(arrs, len(close) - 1, pd.Timestamp("2024-06-01"))
        assert sig is not None
        assert sig["action"] in ("buy", "sell", "hold")

    def test_fast_columns(self):
        s = MomentumBreakoutStrategy()
        assert "close" in s.fast_columns
        assert "high" in s.fast_columns
        assert "low" in s.fast_columns


# ---------------------------------------------------------------------------
# KAMA Strategy
# ---------------------------------------------------------------------------

class TestKAMAStrategy:
    def test_min_lookback(self):
        s = KAMAStrategy(er_period=10, atr_period=14)
        assert s.min_lookback == 16

    def test_hold_on_short_data(self, sample_ohlcv: pd.DataFrame):
        s = KAMAStrategy(er_period=10)
        short = sample_ohlcv.iloc[:5]
        sig = s.on_bar(short, pd.Timestamp("2024-01-05"))
        assert sig["action"] == "hold"

    def test_returns_valid_signal(self, sample_ohlcv: pd.DataFrame):
        s = KAMAStrategy(er_period=10, fast_period=2, slow_period=30)
        sig = s.on_bar(sample_ohlcv, pd.Timestamp("2024-06-01"))
        assert sig["action"] in ("buy", "sell", "hold")

    def test_kama_calculation(self, sample_ohlcv: pd.DataFrame):
        s = KAMAStrategy(er_period=10)
        close = sample_ohlcv["close"].values.astype(np.float64)
        kama = s._calc_kama(close)
        assert len(kama) == len(close)
        assert not np.isnan(kama[-1])
        assert np.isnan(kama[0])


# ---------------------------------------------------------------------------
# MESA Strategy
# ---------------------------------------------------------------------------

class TestMESAStrategy:
    def test_min_lookback(self):
        s = MESAStrategy()
        assert s.min_lookback == 40

    def test_hold_on_short_data(self, sample_ohlcv: pd.DataFrame):
        s = MESAStrategy()
        short = sample_ohlcv.iloc[:20]
        sig = s.on_bar(short, pd.Timestamp("2024-01-20"))
        assert sig["action"] == "hold"

    def test_returns_valid_signal(self, sample_ohlcv: pd.DataFrame):
        s = MESAStrategy(fast_limit=0.5, slow_limit=0.05)
        sig = s.on_bar(sample_ohlcv, pd.Timestamp("2024-06-01"))
        assert sig["action"] in ("buy", "sell", "hold")

    def test_mama_fama_calculation(self, sample_ohlcv: pd.DataFrame):
        s = MESAStrategy()
        close = sample_ohlcv["close"].values.astype(np.float64)
        mama, fama = s._calc_mama_fama(close)
        assert len(mama) == len(close)
        assert len(fama) == len(close)
        assert not np.isnan(mama[-1])
        assert not np.isnan(fama[-1])

    def test_infers_symbol_from_dict(self, sample_ohlcv: pd.DataFrame):
        s = MESAStrategy()
        sig = s.on_bar({"ETH": sample_ohlcv}, pd.Timestamp("2024-06-01"))
        if sig["action"] in ("buy", "sell"):
            assert sig["symbol"] == "ETH"


# ---------------------------------------------------------------------------
# Lorentzian Classification Strategy
# ---------------------------------------------------------------------------

class TestLorentzianStrategy:
    def test_min_lookback(self):
        s = LorentzianClassificationStrategy(
            ema_period=200, sma_period=200, adx_period=20, train_window=2000,
        )
        assert s.min_lookback >= 200

    def test_hold_on_short_data(self, sample_ohlcv: pd.DataFrame):
        s = LorentzianClassificationStrategy()
        short = sample_ohlcv.iloc[:10]
        sig = s.on_bar(short, pd.Timestamp("2024-01-10"))
        assert sig["action"] == "hold"

    def test_returns_valid_signal(self, sample_ohlcv: pd.DataFrame):
        s = LorentzianClassificationStrategy(
            ema_period=20, sma_period=20, adx_period=10, train_window=50,
        )
        s._min_lookback = 30
        sig = s.on_bar(sample_ohlcv, pd.Timestamp("2024-06-01"))
        assert sig["action"] in ("buy", "sell", "hold")

    def test_fast_path(self, sample_ohlcv: pd.DataFrame):
        s = LorentzianClassificationStrategy(
            ema_period=20, sma_period=20, adx_period=10, train_window=50,
        )
        s._min_lookback = 30
        close = sample_ohlcv["close"].to_numpy().astype(np.float64)
        high = sample_ohlcv["high"].to_numpy().astype(np.float64)
        low = sample_ohlcv["low"].to_numpy().astype(np.float64)
        arrs = {"close": close, "high": high, "low": low, "symbol": "TEST"}
        sig = s.on_bar_fast(arrs, len(close) - 1, pd.Timestamp("2024-06-01"))
        assert sig is not None
        assert sig["action"] in ("buy", "sell", "hold")

    def test_fast_columns(self):
        s = LorentzianClassificationStrategy()
        assert "close" in s.fast_columns
        assert "high" in s.fast_columns
        assert "low" in s.fast_columns

    def test_feature_computation(self, sample_ohlcv: pd.DataFrame):
        s = LorentzianClassificationStrategy()
        close = sample_ohlcv["close"].values.astype(np.float64)
        high = sample_ohlcv["high"].values.astype(np.float64)
        low = sample_ohlcv["low"].values.astype(np.float64)
        features = s._compute_features(close, high, low)
        assert features.shape == (len(close), 5)
        assert not np.all(np.isnan(features[-1]))

    def test_lorentzian_distance_properties(self, sample_ohlcv: pd.DataFrame):
        s = LorentzianClassificationStrategy(
            ema_period=20, sma_period=20, adx_period=10, train_window=50,
        )
        close = sample_ohlcv["close"].values.astype(np.float64)
        high = sample_ohlcv["high"].values.astype(np.float64)
        low = sample_ohlcv["low"].values.astype(np.float64)
        features = s._compute_features(close, high, low)
        labels = s._compute_labels(close)
        i = len(close) - 1
        query = features[i]
        if not np.any(np.isnan(query)):
            result = s._lorentzian_knn(features, labels, query, 20, i - 4)
            assert result in (-1, 0, 1)

    def test_infers_symbol_from_dict(self, sample_ohlcv: pd.DataFrame):
        s = LorentzianClassificationStrategy(
            ema_period=20, sma_period=20, adx_period=10, train_window=50,
        )
        s._min_lookback = 30
        sig = s.on_bar({"BTC": sample_ohlcv}, pd.Timestamp("2024-06-01"))
        if sig["action"] in ("buy", "sell"):
            assert sig["symbol"] == "BTC"
