"""
MACD策略示例
当MACD线上穿信号线时买入，下穿时卖出
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from .base_strategy import BaseStrategy
from ..data.indicators import _ema_numba, _macd_numba


class MACDStrategy(BaseStrategy):
    """MACD策略"""

    DEFAULT_PARAM_GRID = [(f, s, sg) for f in range(4, 50, 3)
                          for s in range(f + 4, 120, 5)
                          for sg in range(3, min(s, 50), 4)]

    def __init__(self, name: str = "MACD策略", initial_capital: float = 1000000,
                 fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        初始化MACD策略
        
        Args:
            name: 策略名称
            initial_capital: 初始资金
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
        """
        super().__init__(name, initial_capital)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self._min_lookback = slow_period + signal_period

    @property
    def kernel_name(self) -> str:
        return "MACD"

    @property
    def kernel_params(self) -> tuple:
        return (self.fast_period, self.slow_period, self.signal_period)

    @property
    def fast_columns(self) -> Tuple[str, ...]:
        cols = ("close",)
        if self.fast_period == 12 and self.slow_period == 26 and self.signal_period == 9:
            return cols + ("macd", "macd_signal", "macd_hist")
        return cols
    
    def calculate_macd(self, prices: pd.Series) -> "tuple[float, float, float]":
        """Compute MACD using the Numba kernel from indicators.py."""
        arr = np.ascontiguousarray(prices.values, dtype=np.float64)
        macd_arr, sig_arr, hist_arr = _macd_numba(arr, self.fast_period, self.slow_period, self.signal_period)
        return float(macd_arr[-1]), float(sig_arr[-1]), float(hist_arr[-1])
    
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
        if len(df) < self.slow_period + self.signal_period + 1:
            return {"action": "hold"}
        # 极速路径：优先使用引擎预计算列（Numba/向量化已算好），O(1)
        if all(c in df.columns for c in ("macd", "macd_signal", "macd_hist")) and (
            self.fast_period == 12 and self.slow_period == 26 and self.signal_period == 9
        ):
            macd = float(df["macd"].iloc[-1])
            signal = float(df["macd_signal"].iloc[-1])
            hist = float(df["macd_hist"].iloc[-1])
            prev_macd = float(df["macd"].iloc[-2]) if len(df) > 1 else macd
            prev_signal = float(df["macd_signal"].iloc[-2]) if len(df) > 1 else signal
        else:
            arr = np.ascontiguousarray(df["close"].values, dtype=np.float64)
            macd_line, signal_line, hist_line = _macd_numba(arr, self.fast_period, self.slow_period, self.signal_period)
            macd = float(macd_line[-1])
            signal = float(signal_line[-1])
            hist = float(hist_line[-1])
            prev_macd = float(macd_line[-2]) if len(macd_line) > 1 else macd
            prev_signal = float(signal_line[-2]) if len(signal_line) > 1 else signal
        current_price = float(df["close"].iloc[-1])
        
        # MACD上穿信号线，买入
        if prev_macd <= prev_signal and macd > signal:
            shares = self.calculate_position_size(current_price, capital_fraction=0.95)
            if shares > 0 and self.can_buy(symbol, current_price, shares):
                return {
                    'action': 'buy',
                    'symbol': symbol,
                    'shares': shares
                }
        
        # MACD下穿信号线，卖出
        elif prev_macd >= prev_signal and macd < signal:
            holdings = self.positions.get(symbol, 0)
            if holdings > 0:
                return {
                    'action': 'sell',
                    'symbol': symbol,
                    'shares': holdings
                }
        
        return {'action': 'hold'}

    def on_bar_fast(
        self,
        data_arrays: Dict[str, Any],
        i: int,
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[Union[Dict, List[Dict]]]:
        if i + 1 < self.slow_period + self.signal_period + 1:
            return {"action": "hold"}
        close = data_arrays.get("close")
        macd_arr = data_arrays.get("macd")
        sig_arr = data_arrays.get("macd_signal")
        if close is None or macd_arr is None or sig_arr is None:
            return None
        symbol = data_arrays.get("symbol", "STOCK")
        macd = float(macd_arr[i])
        signal = float(sig_arr[i])
        if pd.isna(macd) or pd.isna(signal):
            return {"action": "hold"}
        prev_macd = float(macd_arr[i - 1]) if i > 0 else macd
        prev_signal = float(sig_arr[i - 1]) if i > 0 else signal
        current_price = float(close[i])
        if prev_macd <= prev_signal and macd > signal:
            shares = self.calculate_position_size(current_price, capital_fraction=0.95)
            if shares > 0 and self.can_buy(symbol, current_price, shares):
                return {"action": "buy", "symbol": symbol, "shares": shares}
        elif prev_macd >= prev_signal and macd < signal:
            holdings = self.positions.get(symbol, 0)
            if holdings > 0:
                return {"action": "sell", "symbol": symbol, "shares": holdings}
        return {"action": "hold"}

    def on_bar_fast_multi(
        self,
        data_arrays_by_symbol: Dict[str, Dict[str, Any]],
        i: int,
        current_date: pd.Timestamp,
        current_prices: Dict[str, float],
    ) -> Optional[Union[Dict, List[Dict]]]:
        signals: List[Dict] = []
        if i + 1 < self.slow_period + self.signal_period + 1:
            return {"action": "hold"}
        for symbol, arrs in data_arrays_by_symbol.items():
            close = arrs.get("close")
            macd_arr = arrs.get("macd")
            sig_arr = arrs.get("macd_signal")
            if close is None or macd_arr is None or sig_arr is None:
                continue
            macd = float(macd_arr[i])
            signal = float(sig_arr[i])
            if pd.isna(macd) or pd.isna(signal):
                continue
            prev_macd = float(macd_arr[i - 1]) if i > 0 else macd
            prev_signal = float(sig_arr[i - 1]) if i > 0 else signal
            current_price = float(close[i])
            if prev_macd <= prev_signal and macd > signal:
                shares = self.calculate_position_size(current_price, capital_fraction=0.95)
                if shares > 0 and self.can_buy(symbol, current_price, shares):
                    signals.append({"action": "buy", "symbol": symbol, "shares": shares})
            elif prev_macd >= prev_signal and macd < signal:
                holdings = self.positions.get(symbol, 0)
                if holdings > 0:
                    signals.append({"action": "sell", "symbol": symbol, "shares": holdings})
        return signals or {"action": "hold"}
