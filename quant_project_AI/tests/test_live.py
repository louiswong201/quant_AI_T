"""
实盘辅助组件测试：RiskGate / RiskManagedBroker / LatencyTracker
"""

import pytest

from quant_framework.broker.paper import PaperBroker
from quant_framework.live import LatencyTracker, RiskConfig, RiskGate, RiskManagedBroker


class TestRiskGate:
    def test_reject_invalid_action(self):
        gate = RiskGate()
        msg = gate.validate({"action": "hold", "symbol": "AAPL", "shares": 1}, cash=1000, positions={})
        assert msg is not None

    def test_reject_exceed_notional(self):
        gate = RiskGate(RiskConfig(max_order_notional=100.0))
        msg = gate.validate(
            {"action": "buy", "symbol": "AAPL", "shares": 2, "price": 100.0},
            cash=1000,
            positions={},
        )
        assert msg == "notional exceed max_order_notional"

    def test_reject_short_sell_when_not_allowed(self):
        gate = RiskGate(RiskConfig(allow_short=False))
        msg = gate.validate(
            {"action": "sell", "symbol": "AAPL", "shares": 10, "price": 100.0},
            cash=1000,
            positions={"AAPL": 0},
        )
        assert msg == "insufficient position"


class TestRiskManagedBroker:
    def test_risk_reject(self):
        broker = PaperBroker(initial_cash=1000)
        gate = RiskGate(RiskConfig(max_order_notional=100.0))
        rm = RiskManagedBroker(broker=broker, risk_gate=gate)
        res = rm.submit_order({"action": "buy", "symbol": "AAPL", "shares": 10, "price": 100.0})
        assert res["status"] == "rejected"
        assert "risk_reject" in res["message"]

    def test_latency_recorded(self):
        broker = PaperBroker(initial_cash=100000)
        tracker = LatencyTracker()
        rm = RiskManagedBroker(broker=broker, latency_tracker=tracker)
        res = rm.submit_order({"action": "buy", "symbol": "AAPL", "shares": 1, "price": 100.0})
        assert res["status"] == "filled"
        assert "latency_ms" in res
        summary = rm.get_account_summary()["latency"]
        assert summary["count"] >= 1

