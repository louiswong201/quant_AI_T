"""
Kaufman KAMA (Adaptive Moving Average) 策略

来源: Perry Kaufman «Smarter Trading» — 效率比驱动的自适应均线

核心思想：
  KAMA 根据效率比 (Efficiency Ratio) 自动在快速与慢速平滑之间切换：
  - 趋势强 (ER → 1) → 快速跟踪，捕捉趋势
  - 震荡 (ER → 0) → 慢速平滑，过滤噪声
  
  结合 ATR 止损实现动态风控。
"""

import numpy as np
import pandas as pd

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def _kama_impl(close: np.ndarray, er_period: int, fast_c: float, slow_c: float) -> np.ndarray:
    n = len(close)
    kama = np.full(n, np.nan, dtype=np.float64)
    if n < er_period:
        return kama
    kama[er_period - 1] = close[er_period - 1]
    for i in range(er_period, n):
        direction = abs(close[i] - close[i - er_period])
        volatility = 0.0
        for j in range(1, er_period + 1):
            volatility += abs(close[i - j + 1] - close[i - j])
        er = direction / volatility if volatility > 0 else 0.0
        sc = (er * (fast_c - slow_c) + slow_c) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    return kama


_kama_numba = njit(cache=True, fastmath=True)(_kama_impl) if NUMBA_AVAILABLE else _kama_impl

from typing import Any, Dict, List, Optional, Tuple, Union

from .base_strategy import BaseStrategy


class KAMAStrategy(BaseStrategy):
    """Kaufman KAMA adaptive moving average strategy (Kaufman KAMA 自适应均线策略).

    Parameters:
      er_period:     Efficiency ratio period (效率比周期，默认 10)
      fast_period:   Fast EMA period (最快 EMA 的周期，默认 2)
      slow_period:   Slow EMA period (最慢 EMA 的周期，默认 30)
      atr_period:    ATR period (ATR 周期，默认 14)
      atr_stop_mult: ATR stop multiplier (ATR 止损倍数，默认 2.0)
    """

    DEFAULT_PARAM_GRID = [(erp, fsc, ssc, asm, ap)
                          for erp in [10, 15, 20] for fsc in [2, 3]
                          for ssc in [20, 30, 50]
                          for asm in [1.5, 2.0, 2.5, 3.0] for ap in [14, 20]]

    def __init__(
        self,
        name: str = "KAMA策略",
        initial_capital: float = 1_000_000,
        er_period: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
    ):
        super().__init__(name, initial_capital)
        self.er_period = er_period
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self._min_lookback = max(er_period, atr_period) + 2
        self._prev_kama = None

    @property
    def kernel_name(self) -> str:
        return "KAMA"

    @property
    def kernel_params(self) -> tuple:
        return (self.er_period, self.fast_period, self.slow_period,
                self.atr_stop_mult, self.atr_period)

    @property
    def fast_columns(self) -> Tuple[str, ...]:
        return ("close", "high", "low")

    def _calc_kama(self, close: np.ndarray) -> np.ndarray:
        """Compute full KAMA series using Numba-compiled kernel (使用 Numba 编译核计算完整 KAMA 序列)."""
        fast_c = 2.0 / (self.fast_period + 1.0)
        slow_c = 2.0 / (self.slow_period + 1.0)
        return _kama_numba(close, self.er_period, fast_c, slow_c)

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

        close = df["close"].values.astype(np.float64)
        kama = self._calc_kama(close)
        idx = len(close) - 1
        k_now = kama[idx]
        k_prev = kama[idx - 1] if idx > 0 else k_now

        if np.isnan(k_now) or np.isnan(k_prev):
            return {"action": "hold"}

        current_price = float(close[-1])
        holdings = self.positions.get(symbol, 0)

        # 持仓：KAMA 掉头平仓
        if holdings > 0 and k_now < k_prev:
            return {"action": "sell", "symbol": symbol, "shares": holdings}

        # 开仓：价格在 KAMA 上方且 KAMA 上行
        if holdings == 0 and current_price > k_now and k_now > k_prev:
            shares = self.calculate_position_size(current_price, capital_fraction=0.95)
            if shares > 0 and self.can_buy(symbol, current_price, shares):
                return {"action": "buy", "symbol": symbol, "shares": shares}

        return {"action": "hold"}
