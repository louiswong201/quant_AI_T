"""
动量突破策略 (Momentum Breakout / N-Day High Proximity)

核心思想：
  当价格接近 N 日新高时，说明动量强劲，入场做多。
  结合 ATR 跟踪止损实现动态风控，让利润奔跑同时控制回撤。

适用场景：
  - 强趋势市场（科技股、成长股）
  - 突破型行情
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_strategy import BaseStrategy
from ..data.indicators import _atr_numba


class MomentumBreakoutStrategy(BaseStrategy):
    """Momentum breakout strategy (动量突破策略).

    Parameters:
      high_period:    Lookback window for rolling high (回看最高价窗口，默认 40，也可用 252 = 52 周)
      proximity_pct:  Proximity threshold to high (接近高点的比例阈值，默认 0.03 = 3%)
      atr_period:     ATR period (ATR 计算周期，默认 14)
      atr_trail:      ATR trailing stop multiplier (ATR 止损倍数，默认 2.0)
    """

    DEFAULT_PARAM_GRID = [(hp, pp, ap, at) for hp in [20, 40, 60, 100, 200]
                          for pp in [0.00, 0.02, 0.05, 0.08]
                          for ap in [10, 14, 20]
                          for at in [1.5, 2.0, 2.5, 3.0]]

    def __init__(
        self,
        name: str = "动量突破策略",
        initial_capital: float = 1_000_000,
        high_period: int = 40,
        proximity_pct: float = 0.03,
        atr_period: int = 14,
        atr_trail: float = 2.0,
    ):
        super().__init__(name, initial_capital)
        self.high_period = high_period
        self.proximity_pct = proximity_pct
        self.atr_period = atr_period
        self.atr_trail = atr_trail
        self._min_lookback = max(high_period, atr_period) + 1
        self._trailing_stop: Dict[str, float] = {}

    @property
    def kernel_name(self) -> str:
        return "MomBreak"

    @property
    def kernel_params(self) -> tuple:
        return (self.high_period, self.proximity_pct, self.atr_period, self.atr_trail)

    @property
    def fast_columns(self) -> Tuple[str, ...]:
        return ("close", "high", "low")

    def _calc_atr_scalar(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, i: int) -> float:
        """ATR at bar i via Numba kernel."""
        if i < self.atr_period:
            return float("nan")
        h = np.ascontiguousarray(high[:i + 1], dtype=np.float64)
        l = np.ascontiguousarray(low[:i + 1], dtype=np.float64)
        c = np.ascontiguousarray(close[:i + 1], dtype=np.float64)
        atr_arr = _atr_numba(h, l, c, self.atr_period)
        return float(atr_arr[-1])

    def on_bar_fast(
        self,
        data_arrays: Dict[str, Any],
        i: int,
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[Union[Dict, List[Dict]]]:
        close = data_arrays.get("close")
        high = data_arrays.get("high")
        low = data_arrays.get("low")
        symbol = data_arrays.get("symbol", "STOCK")
        if close is None or high is None or low is None:
            return None
        if i < self._min_lookback:
            return {"action": "hold"}

        current_price = float(close[i])
        holdings = self.positions.get(symbol, 0)

        # 滚动最高价
        start_h = max(0, i - self.high_period + 1)
        roll_high = float(np.max(high[start_h:i + 1]))

        atr = self._calc_atr_scalar(high, low, close, i)
        if np.isnan(atr):
            atr = float(high[i] - low[i])

        # 持仓：更新跟踪止损
        if holdings > 0:
            new_stop = current_price - self.atr_trail * atr
            if new_stop > self._trailing_stop.get(symbol, 0.0):
                self._trailing_stop[symbol] = new_stop
            if current_price < self._trailing_stop.get(symbol, 0.0):
                self._trailing_stop[symbol] = 0.0
                return {"action": "sell", "symbol": symbol, "shares": holdings}

        # 开仓
        if holdings == 0:
            threshold = roll_high * (1.0 - self.proximity_pct)
            if current_price >= threshold:
                shares = self.calculate_position_size(current_price, capital_fraction=0.95)
                if shares > 0 and self.can_buy(symbol, current_price, shares):
                    self._trailing_stop[symbol] = current_price - self.atr_trail * atr
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

        if len(df) < self._min_lookback:
            return {"action": "hold"}

        close_vals = df["close"].values.astype(np.float64)
        high_vals = df["high"].values.astype(np.float64)
        low_vals = df["low"].values.astype(np.float64)
        idx = len(close_vals) - 1
        current_price = float(close_vals[-1])
        holdings = self.positions.get(symbol, 0)

        # 滚动最高价
        start = max(0, idx - self.high_period + 1)
        roll_high = float(np.max(high_vals[start: idx + 1]))

        atr = self._calc_atr_scalar(high_vals, low_vals, close_vals, idx)
        if np.isnan(atr):
            atr = 1.0

        if holdings > 0:
            new_stop = current_price - self.atr_trail * atr
            if new_stop > self._trailing_stop.get(symbol, 0.0):
                self._trailing_stop[symbol] = new_stop
            if current_price < self._trailing_stop.get(symbol, 0.0):
                self._trailing_stop[symbol] = 0.0
                return {"action": "sell", "symbol": symbol, "shares": holdings}

        if holdings == 0:
            threshold = roll_high * (1.0 - self.proximity_pct)
            if current_price >= threshold:
                shares = self.calculate_position_size(current_price, capital_fraction=0.95)
                if shares > 0 and self.can_buy(symbol, current_price, shares):
                    self._trailing_stop[symbol] = current_price - self.atr_trail * atr
                    return {"action": "buy", "symbol": symbol, "shares": shares}

        return {"action": "hold"}
