"""
Broker 抽象：实盘/纸交易统一接口

策略产生的 signal（action/symbol/shares/order_type/limit_price/stop_price）
由 Broker 负责执行：回测时由 BacktestEngine 内部模拟；实盘/纸交易时由本模块的实现类发单或记单。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class Broker(ABC):
    """执行层抽象：将策略信号转为委托并查询持仓/资金。"""

    @abstractmethod
    def submit_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        提交一笔委托（或仅记录，纸交易时）。

        Args:
            signal: 策略返回的信号，至少含 action ('buy'|'sell'), symbol, shares；
                    可选 order_type ('market'|'limit'|'stop'), limit_price, stop_price

        Returns:
            至少含 status ('submitted'|'filled'|'rejected'|'cancelled' 等)；
            若已成交可含 fill_price, filled_shares, order_id, message 等
        """
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Union[int, float]]:
        """当前持仓 { symbol: 股数 }。支持分数股。"""
        pass

    @abstractmethod
    def get_cash(self) -> float:
        """当前可用资金（元）。"""
        pass

    def get_account_summary(self) -> Dict[str, Any]:
        """可选：总资产、持仓市值等，便于与策略侧对齐。默认用 get_positions + get_cash 即可。"""
        return {
            "cash": self.get_cash(),
            "positions": self.get_positions(),
        }

    async def submit_order_async(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Async submit. Default: run sync submit_order."""
        return self.submit_order(signal)

    async def cancel_order_async(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError("cancel_order_async")

    async def get_order_status_async(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError("get_order_status_async")

    async def get_open_orders_async(self, symbol: str = "") -> List[Dict[str, Any]]:
        return []

    async def sync_positions(self) -> Dict[str, Union[int, float]]:
        return self.get_positions()

    async def sync_balance(self) -> Dict[str, Any]:
        return {"cash": self.get_cash()}

    def get_available_margin(self) -> float:
        return self.get_cash()

    def get_margin_ratio(self) -> float:
        return 1.0
