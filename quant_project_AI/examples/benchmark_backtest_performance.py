"""
回测性能基准脚本：
对比 fast 路径（on_bar_fast）与普通路径（on_bar DataFrame 切片）。

运行：
    python examples/benchmark_backtest_performance.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from quant_framework import BacktestConfig, BacktestEngine
from quant_framework.strategy import MovingAverageStrategy


class MockDataManager:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def load_data(self, symbol, start_date, end_date):
        return self.df.copy()

    def calculate_indicators(self, data):
        from quant_framework.data.indicators import VectorizedIndicators

        return VectorizedIndicators.calculate_all(data)


class SlowMAStrategy(MovingAverageStrategy):
    @property
    def fast_columns(self):
        return ()

    def on_bar_fast(self, data_arrays, i, current_date, current_prices=None):
        return None


def make_data(n: int = 200_000) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="T")
    close = 100 + np.cumsum(np.random.randn(n) * 0.02)
    high = close + np.abs(np.random.randn(n) * 0.01)
    low = close - np.abs(np.random.randn(n) * 0.01)
    open_ = close + np.random.randn(n) * 0.005
    volume = np.random.randint(1000, 5000, size=n).astype(float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def bench(strategy, dm, loops: int = 1) -> float:
    engine = BacktestEngine(dm, config=BacktestConfig())
    t0 = time.perf_counter()
    for _ in range(loops):
        engine.run(strategy, "STOCK", "2020-01-01", "2021-01-01")
    return time.perf_counter() - t0


def main() -> None:
    df = make_data()
    dm = MockDataManager(df)

    fast_strategy = MovingAverageStrategy(short_window=5, long_window=20, initial_capital=1_000_000)
    slow_strategy = SlowMAStrategy(short_window=5, long_window=20, initial_capital=1_000_000)

    t_fast = bench(fast_strategy, dm)
    t_slow = bench(slow_strategy, dm)
    speedup = t_slow / t_fast if t_fast > 0 else 0.0

    print("=== Backtest Benchmark ===")
    print(f"bars: {len(df):,}")
    print(f"fast path : {t_fast:.4f}s")
    print(f"slow path : {t_slow:.4f}s")
    print(f"speedup   : {speedup:.2f}x")


if __name__ == "__main__":
    main()

