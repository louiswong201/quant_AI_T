"""Symbol specification for order validation and precision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .asset_types import AssetClass


@dataclass
class SymbolSpec:
    symbol: str
    asset_class: AssetClass
    base_asset: str
    quote_asset: str
    tick_size: float
    step_size: float
    min_notional: float
    min_qty: float
    max_qty: float
    max_leverage: float = 125.0

    def round_price(self, price: float) -> float:
        if self.tick_size <= 0:
            return price
        return round(price / self.tick_size) * self.tick_size

    def round_quantity(self, qty: float) -> float:
        if self.step_size <= 0:
            return qty
        return int(qty / self.step_size) * self.step_size

    def validate_order(self, qty: float, price: float) -> Optional[str]:
        if qty < self.min_qty:
            return f"qty {qty} < min_qty {self.min_qty}"
        if qty > self.max_qty:
            return f"qty {qty} > max_qty {self.max_qty}"
        notional = qty * price
        if notional < self.min_notional:
            return f"notional {notional} < min_notional {self.min_notional}"
        return None
