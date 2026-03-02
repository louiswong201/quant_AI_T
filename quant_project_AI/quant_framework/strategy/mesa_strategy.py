"""
MESA Adaptive (Ehlers MAMA/FAMA) 策略

来源: John Ehlers «Cybernetic Analysis for Stocks and Futures»

核心思想：
  使用 Hilbert 变换实时测量市场主导周期，据此自适应调整均线平滑系数：
  - MAMA (MESA Adaptive Moving Average): 快速响应趋势变化
  - FAMA (Following AMA): 慢速确认均线
  - 交叉信号：MAMA > FAMA 做多，MAMA < FAMA 做空

优势：
  - 零参数调优需求（fast_limit/slow_limit 几乎不需要改动）
  - 自动适应周期变化，减少假突破
"""

import numpy as np
import pandas as pd

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def _mama_fama_impl(close: np.ndarray, fast_limit: float, slow_limit: float):
    n = len(close)
    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    I1 = np.zeros(n, dtype=np.float64)
    Q1 = np.zeros(n, dtype=np.float64)
    jI = np.zeros(n, dtype=np.float64)
    jQ = np.zeros(n, dtype=np.float64)
    I2 = np.zeros(n, dtype=np.float64)
    Q2 = np.zeros(n, dtype=np.float64)
    Re = np.zeros(n, dtype=np.float64)
    Im = np.zeros(n, dtype=np.float64)
    period_arr = np.full(n, 6.0, dtype=np.float64)
    smooth_period = np.full(n, 6.0, dtype=np.float64)
    phase = np.zeros(n, dtype=np.float64)
    mama = close.copy()
    fama = close.copy()

    for i in range(6, n):
        smooth[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        adj = 0.075 * period_arr[i-1] + 0.54

        idx2 = i - 2 if i >= 2 else 0
        idx4 = i - 4 if i >= 4 else 0
        idx6 = i - 6 if i >= 6 else 0

        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[idx2]
                        - 0.5769 * smooth[idx4] - 0.0962 * smooth[idx6]) * adj

        i3 = i - 3 if i >= 3 else 0
        I1[i] = detrender[i3]
        Q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[idx2]
                 - 0.5769 * detrender[idx4] - 0.0962 * detrender[idx6]) * adj

        jI[i] = (0.0962 * I1[i] + 0.5769 * I1[idx2]
                 - 0.5769 * I1[idx4] - 0.0962 * I1[idx6]) * adj
        jQ[i] = (0.0962 * Q1[i] + 0.5769 * Q1[idx2]
                 - 0.5769 * Q1[idx4] - 0.0962 * Q1[idx6]) * adj

        I2[i] = 0.2 * (I1[i] - jQ[i]) + 0.8 * I2[i-1]
        Q2[i] = 0.2 * (Q1[i] + jI[i]) + 0.8 * Q2[i-1]
        Re[i] = 0.2 * (I2[i] * I2[i-1] + Q2[i] * Q2[i-1]) + 0.8 * Re[i-1]
        Im[i] = 0.2 * (I2[i] * Q2[i-1] - Q2[i] * I2[i-1]) + 0.8 * Im[i-1]

        if Im[i] != 0.0 and Re[i] != 0.0:
            period_arr[i] = 2.0 * 3.141592653589793 / np.arctan(Im[i] / Re[i])
        else:
            period_arr[i] = period_arr[i-1]

        lo = 0.67 * period_arr[i-1]
        hi = 1.5 * period_arr[i-1]
        if period_arr[i] < lo:
            period_arr[i] = lo
        if period_arr[i] > hi:
            period_arr[i] = hi
        if period_arr[i] < 6.0:
            period_arr[i] = 6.0
        if period_arr[i] > 50.0:
            period_arr[i] = 50.0

        period_arr[i] = 0.2 * period_arr[i] + 0.8 * period_arr[i-1]
        smooth_period[i] = 0.33 * period_arr[i] + 0.67 * smooth_period[i-1]

        if I1[i] != 0.0:
            phase[i] = np.arctan(Q1[i] / I1[i]) * (180.0 / 3.141592653589793)
        else:
            phase[i] = phase[i-1]

        delta_phase = phase[i-1] - phase[i]
        if delta_phase < 1.0:
            delta_phase = 1.0

        alpha = fast_limit / delta_phase
        if alpha < slow_limit:
            alpha = slow_limit
        if alpha > fast_limit:
            alpha = fast_limit

        mama[i] = alpha * close[i] + (1.0 - alpha) * mama[i-1]
        fama[i] = 0.5 * alpha * mama[i] + (1.0 - 0.5 * alpha) * fama[i-1]

    return mama, fama


_mama_fama_numba = njit(cache=True, fastmath=True)(_mama_fama_impl) if NUMBA_AVAILABLE else _mama_fama_impl

from typing import Any, Dict, List, Optional, Tuple, Union

from .base_strategy import BaseStrategy


class MESAStrategy(BaseStrategy):
    """
    Ehlers MAMA/FAMA 自适应均线交叉策略。

    参数：
      fast_limit: 快速极限 (默认 0.5)
      slow_limit: 慢速极限 (默认 0.05)
    """

    DEFAULT_PARAM_GRID = [(fl, sl) for fl in [0.3, 0.5, 0.7]
                          for sl in [0.02, 0.05, 0.10]]

    def __init__(
        self,
        name: str = "MESA策略",
        initial_capital: float = 1_000_000,
        fast_limit: float = 0.5,
        slow_limit: float = 0.05,
    ):
        super().__init__(name, initial_capital)
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self._min_lookback = 40

    @property
    def kernel_name(self) -> str:
        return "MESA"

    @property
    def kernel_params(self) -> tuple:
        return (self.fast_limit, self.slow_limit)

    @property
    def fast_columns(self) -> Tuple[str, ...]:
        return ("close",)

    def _calc_mama_fama(self, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute MAMA and FAMA using Numba-compiled Hilbert transform kernel (使用 Numba 编译的 Hilbert 变换核计算 MAMA 和 FAMA)."""
        return _mama_fama_numba(close, self.fast_limit, self.slow_limit)

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
        mama, fama = self._calc_mama_fama(close)
        idx = len(close) - 1
        holdings = self.positions.get(symbol, 0)
        current_price = float(close[-1])

        # MAMA 上穿 FAMA → 做多
        if mama[idx] > fama[idx] and mama[idx - 1] <= fama[idx - 1]:
            if holdings == 0:
                shares = self.calculate_position_size(current_price, risk_percent=0.95)
                if shares > 0 and self.can_buy(symbol, current_price, shares):
                    return {"action": "buy", "symbol": symbol, "shares": shares}
        # MAMA 下穿 FAMA → 平仓
        elif mama[idx] < fama[idx] and mama[idx - 1] >= fama[idx - 1]:
            if holdings > 0:
                return {"action": "sell", "symbol": symbol, "shares": holdings}

        return {"action": "hold"}
