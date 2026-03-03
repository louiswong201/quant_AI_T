"""
Z-Score 均值回归策略
经典统计套利策略，适用于震荡市场

核心思想：
  价格偏离均值的标准差程度 (z-score) 超过阈值时，预期价格回归均值。
  z < -entry_z → 做多 (超跌反弹)
  z > +entry_z → 做空 (超涨回落)
  |z| < exit_z → 平仓 (回归均值完成)
  |z| > stop_z → 止损 (趋势突破)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_strategy import BaseStrategy


class ZScoreReversionStrategy(BaseStrategy):
    """Z-Score mean reversion strategy (Z-Score 均值回归策略).

    Parameters:
      lookback: Mean/std window (均值/标准差计算窗口，默认 35)
      entry_z:  Entry z threshold (入场 z 阈值，默认 2.5)
      exit_z:   Exit z threshold (出场 z 阈值，默认 0.5，回归到 0.5 标准差内平仓)
      stop_z:   Stop-loss z threshold (止损 z 阈值，默认 3.0，趋势确认止损)
    """

    DEFAULT_PARAM_GRID = [(lb, ez, xz, sz)
                          for lb in range(15, 100, 10)
                          for ez in [1.5, 2.0, 2.5]
                          for xz in [0.0, 0.5] for sz in [3.0, 4.0]]

    def __init__(
        self,
        name: str = "ZScore回归策略",
        initial_capital: float = 1_000_000,
        lookback: int = 35,
        entry_z: float = 2.5,
        exit_z: float = 0.5,
        stop_z: float = 3.0,
    ):
        super().__init__(name, initial_capital)
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self._min_lookback = lookback + 1

    @property
    def kernel_name(self) -> str:
        return "ZScore"

    @property
    def kernel_params(self) -> tuple:
        return (self.lookback, self.entry_z, self.exit_z, self.stop_z)

    @property
    def fast_columns(self) -> Tuple[str, ...]:
        return ("close",)

    def _compute_zscore(self, close_arr: np.ndarray, idx: int) -> float:
        """Compute z-score for current bar (计算当前 bar 的 z-score)."""
        window = close_arr[max(0, idx - self.lookback + 1): idx + 1]
        if len(window) < self.lookback:
            return 0.0
        mean = np.mean(window)
        std = np.std(window)
        if std == 0.0:
            return 0.0
        return (close_arr[idx] - mean) / std

    def on_bar_fast(
        self,
        data_arrays: Dict[str, Any],
        i: int,
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[Union[Dict, List[Dict]]]:
        close = data_arrays.get("close")
        symbol = data_arrays.get("symbol", "STOCK")
        if close is None or i < self.lookback:
            return {"action": "hold"}

        z = self._compute_zscore(close, i)
        current_price = float(close[i])
        holdings = self.positions.get(symbol, 0)

        if holdings > 0:
            if abs(z) < self.exit_z:
                return {"action": "sell", "symbol": symbol, "shares": holdings}
            if z < -self.stop_z:
                return {"action": "sell", "symbol": symbol, "shares": holdings}

        if holdings == 0:
            if z < -self.entry_z:
                shares = self.calculate_position_size(current_price, capital_fraction=0.95)
                if shares > 0 and self.can_buy(symbol, current_price, shares):
                    return {"action": "buy", "symbol": symbol, "shares": shares}

        return {"action": "hold"}

    def on_bar(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Union[Dict, List[Dict]]:
        if isinstance(data, dict):
            symbol = next(iter(data))
            df = data[symbol]
        else:
            df = data
            symbol = df.attrs.get("symbol", "STOCK") if hasattr(df, "attrs") else "STOCK"

        if len(df) < self.lookback + 1:
            return {"action": "hold"}

        close_vals = df["close"].values.astype(np.float64)
        idx = len(close_vals) - 1
        z = self._compute_zscore(close_vals, idx)
        current_price = float(close_vals[-1])
        holdings = self.positions.get(symbol, 0)

        if holdings > 0:
            if abs(z) < self.exit_z:
                return {"action": "sell", "symbol": symbol, "shares": holdings}
            if z < -self.stop_z:
                return {"action": "sell", "symbol": symbol, "shares": holdings}

        if holdings == 0 and z < -self.entry_z:
            shares = self.calculate_position_size(current_price, capital_fraction=0.95)
            if shares > 0 and self.can_buy(symbol, current_price, shares):
                return {"action": "buy", "symbol": symbol, "shares": shares}

        return {"action": "hold"}
