from __future__ import annotations

from abc import ABC, abstractmethod


class CostModel(ABC):
    @abstractmethod
    def commission(
        self,
        side: str,
        notional: float,
        is_maker: bool = False,
        shares: float = 0.0,
    ) -> float:
        ...

    @abstractmethod
    def holding_cost(
        self,
        position_notional: float,
        funding_rate: float = 0.0,
        direction: str = "long",
    ) -> float:
        ...


class CryptoFuturesCost(CostModel):
    """Binance VIP0: maker 0.02%, taker 0.04%; funding every 8h."""

    MAKER_RATE = 0.0002
    TAKER_RATE = 0.0004

    def commission(
        self,
        side: str,
        notional: float,
        is_maker: bool = False,
        shares: float = 0.0,
    ) -> float:
        rate = self.MAKER_RATE if is_maker else self.TAKER_RATE
        return notional * rate

    def holding_cost(
        self,
        position_notional: float,
        funding_rate: float = 0.0,
        direction: str = "long",
    ) -> float:
        if direction == "long":
            return position_notional * funding_rate
        return -position_notional * funding_rate


class USEquityCost(CostModel):
    """IBKR Fixed: $0.005/share, min $1; margin interest and borrow cost."""

    PER_SHARE = 0.005
    MIN_COMMISSION = 1.0

    def __init__(
        self,
        margin_interest_rate: float = 0.0583,
        default_borrow_rate: float = 0.05,
    ):
        self._margin_rate = margin_interest_rate
        self._borrow_rate = default_borrow_rate

    def commission(
        self,
        side: str,
        notional: float,
        is_maker: bool = False,
        shares: float = 0.0,
    ) -> float:
        return max(self.MIN_COMMISSION, shares * self.PER_SHARE)

    def holding_cost(
        self,
        position_notional: float,
        funding_rate: float = 0.0,
        direction: str = "long",
        borrowed_amount: float | None = None,
        borrow_rate: float | None = None,
    ) -> float:
        if direction == "long":
            amt = borrowed_amount if borrowed_amount is not None else position_notional * 0.5
            return self.holding_cost_long(amt)
        return self.holding_cost_short(position_notional, borrow_rate)

    def holding_cost_long(
        self, borrowed_amount: float, annual_rate: float | None = None
    ) -> float:
        rate = annual_rate if annual_rate is not None else self._margin_rate
        return borrowed_amount * rate / 365

    def holding_cost_short(
        self, market_value: float, borrow_rate: float | None = None
    ) -> float:
        rate = borrow_rate if borrow_rate is not None else self._borrow_rate
        return market_value * rate / 365

    def settlement_delay_days(self) -> int:
        return 1
