"""Emergency position flattening (kill switch)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

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

    async def flatten_all(self, reason: str) -> List[Dict[str, Any]]:
        """Cancel all orders, market-close all positions, alert."""
        self._triggered = True
        exec_broker = getattr(self._broker, "_broker", self._broker)
        get_open = getattr(exec_broker, "get_open_orders_async", None)
        cancel_order = getattr(exec_broker, "cancel_order_async", None)
        submit_order_async = getattr(exec_broker, "submit_order_async", None)
        closed: List[Dict[str, Any]] = []

        if get_open is not None and cancel_order is not None:
            try:
                open_orders = await get_open("")
                for order in open_orders:
                    oid = order.get("order_id") or order.get("id")
                    if oid is not None:
                        await cancel_order(str(oid))
            except Exception as e:
                logger.error("Kill switch: cancel orders failed: %s", e)

        positions = exec_broker.get_positions()
        max_retries = 3
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
            last_error: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    if submit_order_async is not None:
                        res = await submit_order_async(sig)
                    else:
                        sig["order_type"] = "market"
                        res = await asyncio.to_thread(exec_broker.submit_order, sig)
                    closed.append({
                        "symbol": symbol,
                        "side": side,
                        "requested_shares": abs(float(qty)),
                        "result": res,
                    })
                    if res.get("status") == "filled":
                        logger.info("Kill switch: closed %s %s @ %.4f shares", symbol, side, abs(float(qty)))
                    else:
                        logger.warning("Kill switch: close %s returned status=%s", symbol, res.get("status", "unknown"))
                    break
                except Exception as e:
                    last_error = e
                    logger.error("Kill switch: close %s failed (attempt %d/%d): %s", symbol, attempt + 1, max_retries, e)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0)
            else:
                logger.error("Kill switch: close %s failed after %d retries: %s", symbol, max_retries, last_error)
                closed.append({
                    "symbol": symbol,
                    "side": side,
                    "requested_shares": abs(float(qty)),
                    "result": {"status": "error", "message": str(last_error) if last_error else "unknown"},
                })

        closed_count = sum(1 for c in closed if c.get("result", {}).get("status") == "filled")
        failed = [c for c in closed if c.get("result", {}).get("status") != "filled"]
        failed_count = len(failed)
        if closed_count > 0:
            logger.info("Kill switch: closed %d position(s): %s", closed_count, [c["symbol"] for c in closed if c.get("result", {}).get("status") == "filled"])
        if failed_count > 0:
            logger.warning("Kill switch: failed to close %d position(s): %s", failed_count, [c["symbol"] for c in failed])

        if self._alert is not None:
            send = getattr(self._alert, "send", None)
            if send is not None:
                if asyncio.iscoroutinefunction(send):
                    await send("CRITICAL", "KILL SWITCH", reason)
                elif callable(send):
                    await asyncio.to_thread(send, "CRITICAL", "KILL SWITCH", reason)

        return closed

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    def reset(self) -> None:
        self._triggered = False
