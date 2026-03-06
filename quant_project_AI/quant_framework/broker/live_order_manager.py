"""Live order lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional

from .base import Broker

logger = logging.getLogger(__name__)


class OrderState(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    ERROR = "error"


class LiveOrderManager:
    def __init__(
        self,
        risk_gate: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    ) -> None:
        self._risk_gate = risk_gate

    async def submit(
        self,
        broker: Broker,
        signal: Dict[str, Any],
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        if self._risk_gate:
            err = self._risk_gate(signal)
            if err:
                return {"status": "rejected", "message": err}
        result = await broker.submit_order_async(signal)
        if result.get("status") in ("filled", "partial"):
            return result
        if result.get("status") == "submitted":
            order_id = result.get("order_id")
            if order_id:
                waited = await self._wait_fill(broker, order_id, signal.get("symbol"), timeout_seconds)
                if waited.get("status") in ("filled", "partial", "cancelled", "expired", "error"):
                    return waited
                await broker.cancel_order_async(order_id, signal.get("symbol"))
                return {"status": "cancelled", "message": "timeout", "order_id": order_id}
        return result

    async def _wait_fill(
        self,
        broker: Broker,
        order_id: str,
        symbol: Optional[str],
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                status = await broker.get_order_status_async(order_id, symbol)
            except NotImplementedError:
                await asyncio.sleep(0.5)
                continue
            except Exception:
                status = {"status": "error"}
            s = str(status.get("status", "")).lower()
            if s in ("filled", "partial", "cancelled", "expired", "error"):
                return status
            await asyncio.sleep(0.5)
        return {"status": "timeout", "order_id": order_id}

    async def cancel(self, broker: Broker, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        result = await broker.cancel_order_async(order_id, symbol)
        if result.get("status") == "cancelled":
            return result
        return result
