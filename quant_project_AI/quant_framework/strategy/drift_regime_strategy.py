"""
Drift Regime Mean Reversion 策略
来源: arxiv 2511.12490 — OOS Sharpe 13 的漂移状态均值回归

核心思想：
  当某标的在过去 N 天中有过高比例的上涨天数（漂移 drift）时，
  预示短期均值回归——做空获利；反之亦然。
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_strategy import BaseStrategy


class DriftRegimeStrategy(BaseStrategy):
    """Drift regime mean reversion strategy (漂移状态均值回归策略).

    Parameters:
      lookback:        Lookback window (观察窗口，默认 15，论文使用 63)
      drift_threshold: Drift threshold (漂移阈值，默认 0.62，论文使用 0.60)
      hold_period:     Force close after hold period (持有周期后强制平仓，默认 27 天)
    """

    DEFAULT_PARAM_GRID = [(lb, dt, hp) for lb in range(10, 120, 10)
                          for dt in [0.55, 0.60, 0.65, 0.70]
                          for hp in range(3, 25, 4)]

    def __init__(
        self,
        name: str = "DriftRegime策略",
        initial_capital: float = 1_000_000,
        lookback: int = 15,
        drift_threshold: float = 0.62,
        hold_period: int = 27,
    ):
        super().__init__(name, initial_capital)
        self.lookback = lookback
        self.drift_threshold = drift_threshold
        self.hold_period = hold_period
        self._min_lookback = lookback + 1
        self._hold_count: Dict[str, int] = {}
        self._entry_bar: Dict[str, int] = {}

    @property
    def kernel_name(self) -> str:
        return "Drift"

    @property
    def kernel_params(self) -> tuple:
        return (self.lookback, self.drift_threshold, self.hold_period)

    @property
    def fast_columns(self) -> Tuple[str, ...]:
        return ("close",)

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

        # 计算上涨天数比例
        window = close[i - self.lookback:i + 1]
        up_days = int(np.sum(np.diff(window) > 0))
        up_ratio = up_days / self.lookback
        current_price = float(close[i])
        holdings = self.positions.get(symbol, 0)

        # 检查持有期平仓
        if holdings != 0 and self._entry_bar.get(symbol, -1) >= 0:
            self._hold_count[symbol] = self._hold_count.get(symbol, 0) + 1
            if self._hold_count.get(symbol, 0) >= self.hold_period:
                self._hold_count[symbol] = 0
                self._entry_bar[symbol] = -1
                if holdings > 0:
                    return {"action": "sell", "symbol": symbol, "shares": holdings}
                # 注意：框架暂不支持空头平仓（buy to cover），用 hold 代替
                return {"action": "hold"}

        # 开仓信号
        if holdings == 0:
            if up_ratio <= (1.0 - self.drift_threshold):
                # 漂移过低 → 做多（反转上涨）
                shares = self.calculate_position_size(current_price, capital_fraction=0.95)
                if shares > 0 and self.can_buy(symbol, current_price, shares):
                    self._entry_bar[symbol] = i
                    self._hold_count[symbol] = 0
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

        close_vals = df["close"].values
        n = len(close_vals)
        window = close_vals[n - self.lookback - 1:n]
        up_days = int(np.sum(np.diff(window) > 0))
        up_ratio = up_days / self.lookback
        current_price = float(close_vals[-1])
        holdings = self.positions.get(symbol, 0)

        # 持有期平仓
        if holdings != 0 and self._entry_bar.get(symbol, -1) >= 0:
            self._hold_count[symbol] = self._hold_count.get(symbol, 0) + 1
            if self._hold_count.get(symbol, 0) >= self.hold_period:
                self._hold_count[symbol] = 0
                self._entry_bar[symbol] = -1
                if holdings > 0:
                    return {"action": "sell", "symbol": symbol, "shares": holdings}
                return {"action": "hold"}

        # 做多信号
        if holdings == 0 and up_ratio <= (1.0 - self.drift_threshold):
            shares = self.calculate_position_size(current_price, capital_fraction=0.95)
            if shares > 0 and self.can_buy(symbol, current_price, shares):
                self._entry_bar[symbol] = n - 1
                self._hold_count[symbol] = 0
                return {"action": "buy", "symbol": symbol, "shares": shares}

        return {"action": "hold"}
