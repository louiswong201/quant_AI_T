"""
策略基类
所有交易策略都应该继承这个基类
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd

if TYPE_CHECKING:
    from ..data.rag_context import RagContextProvider

DEFAULT_CAPITAL_FRACTION = 0.95


class BaseStrategy(ABC):
    """策略基类。支持单标的（data 为 DataFrame）与多标的（data 为 Dict[symbol, DataFrame]）。"""

    def __init__(
        self,
        name: str,
        initial_capital: float = 1000000,
        rag_provider: Optional["RagContextProvider"] = None,
        capital_fraction: float = DEFAULT_CAPITAL_FRACTION,
    ):
        """
        初始化策略

        Args:
            name: 策略名称
            initial_capital: 初始资金
            rag_provider: 可选 RAG 上下文提供者，用于 on_bar 中获取非结构化上下文（新闻/研报等）
            capital_fraction: 仓位占比 (0.0–1.0)，默认 0.95
        """
        self.name = name
        self.initial_capital = initial_capital
        self.positions: Dict[str, Union[int, float]] = {}
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.rag_provider = rag_provider
        self._capital_fraction = capital_fraction

    @property
    def min_lookback(self) -> int:
        """
        策略所需最少历史 bar 数（用于指标 warmup）。
        回测/实盘一致性：实盘拉取的历史 K 线数应 >= min_lookback，否则指标与回测不一致。
        子类在 __init__ 中设置 self._min_lookback 覆盖默认值 1。
        """
        return getattr(self, "_min_lookback", 1)

    @property
    def fast_columns(self) -> Tuple[str, ...]:
        """
        可选：声明 on_bar_fast 需要的列名。
        回测引擎会在单标的场景一次性准备这些 ndarray，减少逐 bar DataFrame 切片开销。
        """
        return ()

    @abstractmethod
    def on_bar(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Union[Dict, List[Dict]]:
        """
        每个 bar（交易日）调用一次，生成交易信号。

        Args:
            data: 单标的时为 DataFrame（历史至当前 bar）；多标的时为 Dict[symbol, DataFrame]
            current_date: 当前日期
            current_prices: 多标的时提供，{symbol: 当前 close}

        Returns:
            单信号 dict 或信号 list。每个信号: action ('buy'|'sell'), symbol, shares;
            可选 order_type ('market'|'limit'|'stop'), limit_price, stop_price

        注意：回测引擎传入的 data 为视图，策略不得修改，否则影响回测正确性。
        """
        pass

    def on_bar_fast(
        self,
        data_arrays: Dict[str, Any],
        i: int,
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[Union[Dict, List[Dict]]]:
        """
        可选快速接口：输入 ndarray 上下文与当前索引 i。
        默认返回 None，表示回退到 on_bar(DataFrame)。
        """
        return None

    def on_bar_fast_multi(
        self,
        data_arrays_by_symbol: Dict[str, Dict[str, Any]],
        i: int,
        current_date: pd.Timestamp,
        current_prices: Dict[str, float],
    ) -> Optional[Union[Dict, List[Dict]]]:
        """
        可选多标的快速接口。
        默认返回 None，表示回退到 on_bar(Dict[symbol, DataFrame])。
        """
        return None
    
    def calculate_position_size(self, price: float, capital_fraction: float = DEFAULT_CAPITAL_FRACTION) -> int:
        """Calculate the number of shares to buy.

        Args:
            price: Current share price.
            capital_fraction: Fraction of portfolio value to allocate
                (0.0–1.0). For example 0.95 means "invest 95% of
                portfolio value". Defaults to DEFAULT_CAPITAL_FRACTION.

        Returns:
            Number of shares (floored to int, minimum 0).
        """
        if price <= 0:
            return 0
        amount = self.portfolio_value * capital_fraction
        shares = int(amount / price)
        return max(0, shares)
    
    def can_buy(self, symbol: str, price: float, shares: int) -> bool:
        """检查是否可以买入"""
        cost = price * shares
        return cost <= self.cash
    
    def can_sell(self, symbol: str, shares: int) -> bool:
        """检查是否可以卖出"""
        pos = self.positions.get(symbol, 0)
        if pos > 0:
            return pos >= shares
        if pos < 0:
            return True  # allow closing/reducing short positions
        return False  # no position
    
    def buy(self, symbol: str, price: float, shares: int) -> bool:
        """
        买入股票
        
        Returns:
            True if successful, False otherwise
        """
        if not self.can_buy(symbol, price, shares):
            return False
        
        cost = price * shares
        self.cash -= cost
        self.positions[symbol] = self.positions.get(symbol, 0) + shares
        return True
    
    def sell(self, symbol: str, price: float, shares: int) -> bool:
        """
        卖出股票
        
        Returns:
            True if successful, False otherwise
        """
        if not self.can_sell(symbol, shares):
            return False
        
        revenue = price * shares
        self.cash += revenue
        self.positions[symbol] = self.positions.get(symbol, 0) - shares
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        return True
    
    def update_portfolio_value(self, current_prices: Dict[str, float]) -> None:
        """更新投资组合价值"""
        positions_value = sum(
            shares * current_prices.get(symbol, 0)
            for symbol, shares in self.positions.items()
        )
        self.portfolio_value = self.cash + positions_value

    def get_rag_context(
        self,
        query: str,
        symbol: Optional[str] = None,
        top_k: int = 5,
        max_chars: int = 4000,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> str:
        """
        获取 RAG 检索上下文（仅当构造时传入 rag_provider 时可用）。
        回测时传入 as_of_date 可保证仅使用该日期前的文档，避免未来信息。
        """
        if self.rag_provider is None:
            return ""
        dt = as_of_date.to_pydatetime() if hasattr(as_of_date, "to_pydatetime") else as_of_date
        return self.rag_provider.get_context(
            query=query,
            symbol=symbol,
            top_k=top_k,
            max_chars=max_chars,
            as_of_date=dt,
        )
