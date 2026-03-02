"""
共享 fixtures：构造用于测试的合成行情数据和基本对象。
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """生成 100 根日 K 线的合成数据（含 open/high/low/close/volume/date）。"""
    np.random.seed(42)
    n = 100
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.15
    volume = np.random.randint(1000, 10000, size=n).astype(float)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture()
def close_array(sample_ohlcv: pd.DataFrame) -> np.ndarray:
    """从 sample_ohlcv 中提取连续 float64 close 数组。"""
    return np.ascontiguousarray(sample_ohlcv["close"].values, dtype=np.float64)


collect_ignore_glob = ["test_rag_full_chain.py"]
