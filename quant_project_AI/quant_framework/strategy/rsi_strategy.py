"""
RSI策略示例
当RSI低于30时买入，高于70时卖出
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from .base_strategy import BaseStrategy
from ..data.indicators import _rsi_numba


class RSIStrategy(BaseStrategy):
    """RSI策略"""

    DEFAULT_PARAM_GRID = [(p, os_v, ob_v) for p in range(5, 101, 5)
                          for os_v in range(15, 40, 5) for ob_v in range(60, 90, 5)]

    def __init__(self, name: str = "RSI策略", initial_capital: float = 1000000,
                 rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        初始化RSI策略
        
        Args:
            name: 策略名称
            initial_capital: 初始资金
            rsi_period: RSI计算周期
            oversold: 超卖阈值
            overbought: 超买阈值
        """
        super().__init__(name, initial_capital)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self._min_lookback = rsi_period + 1

    @property
    def kernel_name(self) -> str:
        return "RSI"

    @property
    def kernel_params(self) -> tuple:
        return (self.rsi_period, self.oversold, self.overbought)

    @property
    def fast_columns(self) -> Tuple[str, ...]:
        return ("close", "rsi")
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Compute RSI scalar via Numba kernel — same speed as backtest engine."""
        arr = np.ascontiguousarray(prices.values, dtype=np.float64)
        rsi_arr = _rsi_numba(arr, period)
        val = float(rsi_arr[-1])
        return val if not np.isnan(val) else 50.0
    
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
        if len(df) < self.rsi_period + 1:
            return {"action": "hold"}
        # 极速路径：优先使用引擎预计算列（Numba/向量化已算好），O(1)
        if "rsi" in df.columns and self.rsi_period == 14:
            rsi = float(df["rsi"].iloc[-1])
        else:
            rsi = self.calculate_rsi(df["close"], self.rsi_period)
        current_price = float(df["close"].iloc[-1])
        
        holdings = self.positions.get(symbol, 0)

        if rsi < self.oversold:
            if holdings == 0:
                shares = self.calculate_position_size(current_price, capital_fraction=self._capital_fraction)
                if shares > 0 and self.can_buy(symbol, current_price, shares):
                    return {"action": "buy", "symbol": symbol, "shares": shares}
            elif holdings < 0:
                return {"action": "buy", "symbol": symbol, "shares": abs(holdings),
                        "flip": True}
        elif rsi > self.overbought:
            if holdings == 0:
                shares = self.calculate_position_size(current_price, capital_fraction=self._capital_fraction)
                if shares > 0 and self.can_sell(symbol, shares):
                    return {"action": "sell", "symbol": symbol, "shares": shares}
            elif holdings > 0:
                return {"action": "sell", "symbol": symbol, "shares": holdings,
                        "flip": True}
        elif holdings > 0 and rsi > 50:
            return {"action": "sell", "symbol": symbol, "shares": holdings}
        elif holdings < 0 and rsi < 50:
            return {"action": "buy", "symbol": symbol, "shares": abs(holdings)}

        return {"action": "hold"}

    def on_bar_fast(
        self,
        data_arrays: Dict[str, Any],
        i: int,
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[Union[Dict, List[Dict]]]:
        if i + 1 < self.rsi_period + 1:
            return {"action": "hold"}
        close = data_arrays.get("close")
        if close is None:
            return None
        symbol = data_arrays.get("symbol", "STOCK")

        rsi_arr = data_arrays.get("rsi")
        if rsi_arr is not None and self.rsi_period == 14:
            rsi = float(rsi_arr[i])
        else:
            rsi = self._rsi_at_index(close, i, self.rsi_period)

        if pd.isna(rsi):
            return {"action": "hold"}

        current_price = float(close[i])
        holdings = self.positions.get(symbol, 0)

        if rsi < self.oversold:
            if holdings == 0:
                shares = self.calculate_position_size(current_price, capital_fraction=self._capital_fraction)
                if shares > 0 and self.can_buy(symbol, current_price, shares):
                    return {"action": "buy", "symbol": symbol, "shares": shares}
            elif holdings < 0:
                return {"action": "buy", "symbol": symbol, "shares": abs(holdings), "flip": True}
        elif rsi > self.overbought:
            if holdings == 0:
                shares = self.calculate_position_size(current_price, capital_fraction=self._capital_fraction)
                if shares > 0 and self.can_sell(symbol, shares):
                    return {"action": "sell", "symbol": symbol, "shares": shares}
            elif holdings > 0:
                return {"action": "sell", "symbol": symbol, "shares": holdings, "flip": True}
        elif holdings > 0 and rsi > 50:
            return {"action": "sell", "symbol": symbol, "shares": holdings}
        elif holdings < 0 and rsi < 50:
            return {"action": "buy", "symbol": symbol, "shares": abs(holdings)}
        return {"action": "hold"}

    @staticmethod
    def _rsi_at_index(close: np.ndarray, i: int, period: int) -> float:
        """Compute RSI at index i using the Numba kernel from indicators.py.

        Runs _rsi_numba on close[:i+1] — after the first JIT compilation (cached),
        this is ~10-50x faster than the pandas rolling approach.
        """
        if i < period:
            return 50.0
        segment = np.ascontiguousarray(close[: i + 1], dtype=np.float64)
        rsi_arr = _rsi_numba(segment, period)
        val = float(rsi_arr[-1])
        return val if not np.isnan(val) else 50.0

    def on_bar_fast_multi(
        self,
        data_arrays_by_symbol: Dict[str, Dict[str, Any]],
        i: int,
        current_date: pd.Timestamp,
        current_prices: Dict[str, float],
    ) -> Optional[Union[Dict, List[Dict]]]:
        signals: List[Dict] = []
        for symbol, arrs in data_arrays_by_symbol.items():
            close = arrs.get("close")
            if close is None or i + 1 < self.rsi_period + 1:
                continue
            rsi_arr = arrs.get("rsi")
            if rsi_arr is not None and self.rsi_period == 14:
                rsi = float(rsi_arr[i])
            else:
                rsi = self._rsi_at_index(close, i, self.rsi_period)
            if pd.isna(rsi):
                continue
            current_price = float(close[i])
            holdings = self.positions.get(symbol, 0)

            if rsi < self.oversold:
                if holdings == 0:
                    shares = self.calculate_position_size(current_price, capital_fraction=self._capital_fraction)
                    if shares > 0 and self.can_buy(symbol, current_price, shares):
                        signals.append({"action": "buy", "symbol": symbol, "shares": shares})
                elif holdings < 0:
                    signals.append({"action": "buy", "symbol": symbol, "shares": abs(holdings), "flip": True})
            elif rsi > self.overbought:
                if holdings == 0:
                    shares = self.calculate_position_size(current_price, capital_fraction=self._capital_fraction)
                    if shares > 0 and self.can_sell(symbol, shares):
                        signals.append({"action": "sell", "symbol": symbol, "shares": shares})
                elif holdings > 0:
                    signals.append({"action": "sell", "symbol": symbol, "shares": holdings, "flip": True})
            elif holdings > 0 and rsi > 50:
                signals.append({"action": "sell", "symbol": symbol, "shares": holdings})
            elif holdings < 0 and rsi < 50:
                signals.append({"action": "buy", "symbol": symbol, "shares": abs(holdings)})
        return signals or [{"action": "hold"}]
