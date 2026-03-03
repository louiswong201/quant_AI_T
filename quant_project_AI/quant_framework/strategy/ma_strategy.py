"""
移动平均线策略示例
当短期均线上穿长期均线时买入，下穿时卖出
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_strategy import BaseStrategy
from ..data.indicators import _rolling_mean_numba


class MovingAverageStrategy(BaseStrategy):
    """移动平均线策略（单标的）。"""

    DEFAULT_PARAM_GRID = [(s, lg) for s in range(5, 100, 3)
                          for lg in range(s + 5, 201, 5)]

    def __init__(self, name: str = "MA策略", initial_capital: float = 1000000,
                 short_window: int = 5, long_window: int = 20):
        super().__init__(name, initial_capital)
        self.short_window = short_window
        self.long_window = long_window
        self._min_lookback = long_window

    @property
    def kernel_name(self) -> str:
        return "MA"

    @property
    def kernel_params(self) -> tuple:
        return (self.short_window, self.long_window)

    @property
    def fast_columns(self) -> Tuple[str, ...]:
        return ("close", f"ma{self.short_window}", f"ma{self.long_window}")

    def on_bar_fast(
        self,
        data_arrays: Dict[str, Any],
        i: int,
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[Union[Dict, List[Dict]]]:
        """
        ndarray 快速路径：避免逐 bar DataFrame 切片，适合大样本回测。
        """
        if i + 1 < self.long_window:
            return {"action": "hold"}
        close = data_arrays.get("close")
        short = data_arrays.get(f"ma{self.short_window}")
        long = data_arrays.get(f"ma{self.long_window}")
        symbol = data_arrays.get("symbol", "STOCK")
        if close is None or short is None or long is None:
            return None
        short_ma = float(short[i])
        long_ma = float(long[i])
        if pd.isna(short_ma) or pd.isna(long_ma):
            return {"action": "hold"}
        prev_short_ma = float(short[i - 1]) if i > 0 else short_ma
        prev_long_ma = float(long[i - 1]) if i > 0 else long_ma
        current_price = float(close[i])
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            shares = self.calculate_position_size(current_price, capital_fraction=0.95)
            if shares > 0 and self.can_buy(symbol, current_price, shares):
                return {"action": "buy", "symbol": symbol, "shares": shares}
        elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
            holdings = self.positions.get(symbol, 0)
            if holdings > 0:
                return {"action": "sell", "symbol": symbol, "shares": holdings}
        return {"action": "hold"}

    def on_bar(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Union[Dict, List[Dict]]:
        """单标的：data 为 DataFrame；多标的时由引擎传入 Dict[symbol, DataFrame]。"""
        if isinstance(data, dict):
            symbol = next(iter(data))
            df = data[symbol]
        else:
            df = data
            symbol = df.attrs.get("symbol", "STOCK") if hasattr(df, "attrs") else "STOCK"
        if len(df) < self.long_window:
            return {"action": "hold"}

        # 极速路径：优先使用引擎预计算列（calculate_indicators 已用 Numba/向量化算好），O(1) 取末尾值
        col_short = f"ma{self.short_window}"
        col_long = f"ma{self.long_window}"
        if col_short in df.columns and col_long in df.columns:
            short_ma = float(df[col_short].iloc[-1])
            long_ma = float(df[col_long].iloc[-1])
            prev_short_ma = float(df[col_short].iloc[-2]) if len(df) > 1 else short_ma
            prev_long_ma = float(df[col_long].iloc[-2]) if len(df) > 1 else long_ma
        else:
            arr = np.ascontiguousarray(df["close"].values, dtype=np.float64)
            short_arr = _rolling_mean_numba(arr, self.short_window)
            long_arr = _rolling_mean_numba(arr, self.long_window)
            short_ma = float(short_arr[-1])
            long_ma = float(long_arr[-1])
            prev_short_ma = float(short_arr[-2]) if len(arr) > 1 else short_ma
            prev_long_ma = float(long_arr[-2]) if len(arr) > 1 else long_ma
        current_price = float(df["close"].iloc[-1])
        
        # 金叉：短期均线上穿长期均线，买入
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            # 计算买入数量（使用全部可用资金）
            shares = self.calculate_position_size(current_price, capital_fraction=0.95)
            if shares > 0 and self.can_buy(symbol, current_price, shares):
                return {
                    'action': 'buy',
                    'symbol': symbol,
                    'shares': shares
                }
        
        # 死叉：短期均线下穿长期均线，卖出
        elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
            # 卖出全部持仓
            holdings = self.positions.get(symbol, 0)
            if holdings > 0:
                return {
                    'action': 'sell',
                    'symbol': symbol,
                    'shares': holdings
                }
        
        return {'action': 'hold'}
