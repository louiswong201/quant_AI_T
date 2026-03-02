"""
Live risk gateway with circuit breaker — pre-trade validation and kill switch.

Components:
  - RiskConfig: static risk limits (per-order, per-position)
  - CircuitBreaker: dynamic kill switch triggered by cumulative loss or
    error rate thresholds. Once tripped, ALL orders are rejected until
    manual reset. This prevents runaway losses from bugs, bad data feeds,
    or flash crash scenarios.
  - RiskGate: signal-level pre-trade validation
  - RiskManagedBroker: wraps any Broker with risk checks + latency tracking

Why a circuit breaker:
  In live trading, a strategy bug or data feed corruption can generate
  hundreds of erroneous orders in seconds. Without a kill switch, losses
  compound faster than any human can react. The circuit breaker provides
  an automatic safety net with configurable thresholds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..broker.base import Broker
from .latency import LatencyTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskConfig:
    """Static risk limits for pre-trade validation."""

    max_order_notional: float = 1_000_000.0
    max_order_shares: int = 1_000_000
    max_position_shares: int = 5_000_000
    allow_short: bool = False
    max_daily_loss_pct: float = 0.05
    max_consecutive_errors: int = 5
    cooldown_seconds: float = 60.0


class CircuitBreaker:
    """Dynamic kill switch for live trading safety.

    Tracks cumulative P&L and error counts. Trips when either:
      1. Daily loss exceeds max_daily_loss_pct of initial capital
      2. Consecutive order errors exceed max_consecutive_errors

    Once tripped, all subsequent orders are rejected until reset() is called.
    This is a deliberate manual-reset design — automatic reset would defeat
    the purpose of human oversight during anomalous conditions.
    """

    def __init__(self, config: RiskConfig, initial_capital: float = 1_000_000.0) -> None:
        self._config = config
        self._initial_capital = initial_capital
        self._tripped = False
        self._trip_reason: str = ""
        self._daily_pnl: float = 0.0
        self._consecutive_errors: int = 0
        self._trip_time: float = 0.0
        self._order_count: int = 0
        self._reject_count: int = 0

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    def check(self) -> Optional[str]:
        """Check if circuit breaker should block trading. Returns reason or None."""
        if self._tripped:
            return f"CIRCUIT_BREAKER_TRIPPED: {self._trip_reason}"
        return None

    def record_pnl(self, pnl: float) -> None:
        """Record a P&L change and check loss threshold."""
        self._daily_pnl += pnl
        max_loss = self._initial_capital * self._config.max_daily_loss_pct
        if self._daily_pnl < -max_loss:
            self._trip(
                f"Daily loss ${abs(self._daily_pnl):,.2f} exceeds "
                f"limit ${max_loss:,.2f} ({self._config.max_daily_loss_pct:.1%})"
            )

    def record_error(self) -> None:
        """Record an order error and check consecutive error threshold."""
        self._consecutive_errors += 1
        if self._consecutive_errors >= self._config.max_consecutive_errors:
            self._trip(
                f"{self._consecutive_errors} consecutive errors — "
                "possible data feed or connectivity issue"
            )

    def record_success(self) -> None:
        """Reset consecutive error counter on successful order."""
        self._consecutive_errors = 0
        self._order_count += 1

    def _trip(self, reason: str) -> None:
        self._tripped = True
        self._trip_reason = reason
        self._trip_time = time.time()
        logger.critical("CIRCUIT BREAKER TRIPPED: %s", reason)

    def reset(self) -> None:
        """Manual reset — requires human decision to resume trading."""
        logger.warning("Circuit breaker manually reset")
        self._tripped = False
        self._trip_reason = ""
        self._daily_pnl = 0.0
        self._consecutive_errors = 0

    def reset_daily(self) -> None:
        """Reset daily P&L counter (call at start of each trading day)."""
        self._daily_pnl = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "tripped": self._tripped,
            "trip_reason": self._trip_reason,
            "daily_pnl": self._daily_pnl,
            "consecutive_errors": self._consecutive_errors,
            "total_orders": self._order_count,
            "total_rejects": self._reject_count,
        }


class RiskGate:
    """信号级风险检查。"""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    def validate(
        self,
        signal: Dict[str, Any],
        *,
        cash: float,
        positions: Dict[str, Union[int, float]],
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        action = str(signal.get("action") or "").lower()
        symbol = str(signal.get("symbol") or "")
        shares = float(signal.get("shares") or 0)
        if action not in ("buy", "sell"):
            return "invalid action"
        if not symbol or shares <= 0:
            return "invalid symbol/shares"
        if shares > self.config.max_order_shares:
            return "shares exceed max_order_shares"

        price = float(signal.get("price") or signal.get("fill_price") or 0.0)
        if price <= 0 and current_prices and symbol in current_prices:
            price = float(current_prices[symbol])
        if price > 0:
            notional = price * shares
            if notional > self.config.max_order_notional:
                return "notional exceed max_order_notional"
            if action == "buy" and cash < notional:
                return "insufficient cash"

        pos = int(positions.get(symbol, 0))
        if action == "sell":
            if not self.config.allow_short and shares > pos:
                return "insufficient position"
        else:
            if pos + shares > self.config.max_position_shares:
                return "position exceed max_position_shares"
        return None


class RiskManagedBroker(Broker):
    """Wraps any Broker with risk validation, circuit breaker, and latency tracking.

    Order flow:
      1. Circuit breaker check (if tripped, reject immediately)
      2. RiskGate pre-trade validation (limits, cash, position checks)
      3. Forward to underlying broker
      4. Record latency and update circuit breaker state
    """

    def __init__(
        self,
        broker: Broker,
        risk_gate: Optional[RiskGate] = None,
        latency_tracker: Optional[LatencyTracker] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self._broker = broker
        self._risk_gate = risk_gate or RiskGate()
        self._lat = latency_tracker or LatencyTracker()
        self._cb = circuit_breaker

    def submit_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        if self._cb is not None:
            cb_reason = self._cb.check()
            if cb_reason is not None:
                return {"status": "rejected", "message": cb_reason}

        reason = self._risk_gate.validate(
            signal,
            cash=self._broker.get_cash(),
            positions=self._broker.get_positions(),
        )
        if reason is not None:
            return {"status": "rejected", "message": f"risk_reject: {reason}"}

        t0 = time.perf_counter()
        try:
            res = self._broker.submit_order(signal)
        except Exception as exc:
            if self._cb is not None:
                self._cb.record_error()
            logger.error("Order submission failed: %s", exc)
            return {"status": "error", "message": str(exc)}

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._lat.record_ms(dt_ms)

        if self._cb is not None:
            if isinstance(res, dict) and res.get("status") == "error":
                self._cb.record_error()
            else:
                self._cb.record_success()

        if isinstance(res, dict):
            res.setdefault("latency_ms", dt_ms)
        return res

    def get_positions(self) -> Dict[str, Union[int, float]]:
        return self._broker.get_positions()

    def get_cash(self) -> float:
        return self._broker.get_cash()

    def get_account_summary(self) -> Dict[str, Any]:
        out = dict(self._broker.get_account_summary())
        out["latency"] = self._lat.summary()
        if self._cb is not None:
            out["circuit_breaker"] = self._cb.summary()
        return out

