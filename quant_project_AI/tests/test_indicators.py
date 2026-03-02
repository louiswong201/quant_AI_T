"""
VectorizedIndicators 单元测试
覆盖：MA / RSI / MACD / Bollinger / ATR / CCI / WillR / Stochastic / ADX / calculate_all
"""

import numpy as np
import pandas as pd
import pytest

from quant_framework.data.indicators import VectorizedIndicators


class TestMA:
    def test_basic_shape(self, close_array: np.ndarray):
        result = VectorizedIndicators.ma(close_array, 5)
        assert result.shape == close_array.shape
        assert np.isnan(result[3])  # window=5: first 4 are NaN
        assert not np.isnan(result[4])

    def test_known_value(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = VectorizedIndicators.ma(arr, 3)
        np.testing.assert_almost_equal(result[2], 2.0)
        np.testing.assert_almost_equal(result[4], 4.0)

    def test_window_larger_than_data(self):
        arr = np.array([1.0, 2.0])
        result = VectorizedIndicators.ma(arr, 5)
        assert np.all(np.isnan(result))


class TestRSI:
    def test_shape(self, close_array: np.ndarray):
        result = VectorizedIndicators.rsi(close_array, 14)
        assert result.shape == close_array.shape

    def test_range(self, close_array: np.ndarray):
        result = VectorizedIndicators.rsi(close_array, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_short_data(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = VectorizedIndicators.rsi(arr, 14)
        assert np.all(np.isnan(result))


class TestMACD:
    def test_shape(self, close_array: np.ndarray):
        macd, signal, hist = VectorizedIndicators.macd(close_array)
        assert macd.shape == close_array.shape
        assert signal.shape == close_array.shape
        assert hist.shape == close_array.shape

    def test_histogram_is_diff(self, close_array: np.ndarray):
        macd, signal, hist = VectorizedIndicators.macd(close_array)
        np.testing.assert_allclose(hist, macd - signal, atol=1e-10)


class TestBollingerBands:
    def test_shape(self, close_array: np.ndarray):
        upper, middle, lower = VectorizedIndicators.bollinger_bands(close_array, 20, 2.0)
        assert upper.shape == close_array.shape
        assert np.all(upper[~np.isnan(upper)] >= middle[~np.isnan(middle)])
        assert np.all(lower[~np.isnan(lower)] <= middle[~np.isnan(middle)])


class TestATR:
    def test_shape(self, sample_ohlcv: pd.DataFrame):
        h = sample_ohlcv["high"].values.astype(np.float64)
        l = sample_ohlcv["low"].values.astype(np.float64)
        c = sample_ohlcv["close"].values.astype(np.float64)
        result = VectorizedIndicators.atr(h, l, c, 14)
        assert result.shape == c.shape
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)


class TestCCI:
    def test_shape(self, sample_ohlcv: pd.DataFrame):
        h = sample_ohlcv["high"].values.astype(np.float64)
        l = sample_ohlcv["low"].values.astype(np.float64)
        c = sample_ohlcv["close"].values.astype(np.float64)
        result = VectorizedIndicators.cci(h, l, c, 20)
        assert result.shape == c.shape


class TestWillR:
    def test_range(self, sample_ohlcv: pd.DataFrame):
        h = sample_ohlcv["high"].values.astype(np.float64)
        l = sample_ohlcv["low"].values.astype(np.float64)
        c = sample_ohlcv["close"].values.astype(np.float64)
        result = VectorizedIndicators.willr(h, l, c, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -100)
        assert np.all(valid <= 0)


class TestStochastic:
    def test_shape(self, sample_ohlcv: pd.DataFrame):
        h = sample_ohlcv["high"].values.astype(np.float64)
        l = sample_ohlcv["low"].values.astype(np.float64)
        c = sample_ohlcv["close"].values.astype(np.float64)
        k, d = VectorizedIndicators.stoch(h, l, c, 14, 3)
        assert k.shape == c.shape
        assert d.shape == c.shape


class TestADX:
    def test_shape(self, sample_ohlcv: pd.DataFrame):
        h = sample_ohlcv["high"].values.astype(np.float64)
        l = sample_ohlcv["low"].values.astype(np.float64)
        c = sample_ohlcv["close"].values.astype(np.float64)
        result = VectorizedIndicators.adx(h, l, c, 14)
        assert result.shape == c.shape


class TestCalculateAll:
    def test_default_indicators(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        result = VectorizedIndicators.calculate_all(df)
        assert "ma5" in result.columns
        assert "ma10" in result.columns
        assert "ma20" in result.columns
        assert "rsi" in result.columns
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns
        assert "bb_upper" in result.columns
        assert "atr" in result.columns
        assert "cci" in result.columns
        assert "willr" in result.columns
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns
        assert "adx" in result.columns

    def test_inplace_modification(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        returned = VectorizedIndicators.calculate_all(df)
        assert returned is df  # same object, in-place

    def test_no_hl_skips_ohlc_indicators(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv[["date", "close"]].copy()
        result = VectorizedIndicators.calculate_all(df)
        assert "atr" not in result.columns
        assert "ma5" in result.columns
