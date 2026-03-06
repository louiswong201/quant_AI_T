"""Emergency position flattening (kill switch)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class KillSwitch:
    """Emergency position flattening."""

    def __init__(
        self,
        broker: Any,
        alert_manager: Optional[Any] = None,
    ) -> None:
        self._broker = broker
        self._alert = alert_manager
        self._triggered = False

    async def flatten_all(self, reason: str) -> None:
        """Cancel all orders, market-close all positions, alert."""
        self._triggered = True
        get_open = getattr(self._broker, "get_open_orders_async", None)
        cancel_order = getattr(self._broker, "cancel_order_async", None)
        submit_order_async = getattr(self._broker, "submit_order_async", None)

        if get_open is not None and cancel_order is not None:
            try:
                open_orders = await get_open("")
                for order in open_orders:
                    oid = order.get("order_id") or order.get("id")
                    if oid is not None:
                        await cancel_order(str(oid))
            except Exception as e:
                logger.error("Kill switch: cancel orders failed: %s", e)

        positions = self._broker.get_positions()
        for symbol, qty in positions.items():
            if abs(float(qty)) <= 1e-10:
                continue
            side = "sell" if float(qty) > 0 else "buy"
            sig: Dict[str, Any] = {
                "action": side,
                "symbol": symbol,
                "shares": abs(float(qty)),
                "reduce_only": True,
            }
            try:
                if submit_order_async is not None:
                    await submit_order_async(sig)
                else:
                    sig["order_type"] = "market"
                    await asyncio.to_thread(self._broker.submit_order, sig)
            except Exception as e:
                logger.error("Kill switch: close %s failed: %s", symbol, e)

        if self._alert is not None:
            send = getattr(self._alert, "send", None)
            if send is not None:
                if asyncio.iscoroutinefunction(send):
                    await send("CRITICAL", "KILL SWITCH", reason)
                elif callable(send):
                    await asyncio.to_thread(send, "CRITICAL", "KILL SWITCH", reason)

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    def reset(self) -> None:
        self._triggered = False
