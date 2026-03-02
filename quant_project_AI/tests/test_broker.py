"""
Broker / PaperBroker 单元测试
"""

import pytest

from quant_framework.broker.base import Broker
from quant_framework.broker.paper import PaperBroker
from quant_framework.backtest.config import BacktestConfig


class TestPaperBroker:
    def test_buy_and_sell(self):
        broker = PaperBroker(initial_cash=10000)
        result = broker.submit_order({
            "action": "buy", "symbol": "AAPL", "shares": 10, "price": 100,
        })
        assert result["status"] == "filled"
        assert result["filled_shares"] == 10
        assert broker.get_cash() == pytest.approx(9000.0)
        assert broker.get_positions() == {"AAPL": 10}

        result = broker.submit_order({
            "action": "sell", "symbol": "AAPL", "shares": 10, "price": 110,
        })
        assert result["status"] == "filled"
        assert broker.get_cash() == pytest.approx(10100.0)
        assert broker.get_positions() == {}

    def test_reject_invalid_signal(self):
        broker = PaperBroker(initial_cash=10000)
        result = broker.submit_order({"action": "hold", "symbol": "X"})
        assert result["status"] == "rejected"

    def test_reject_insufficient_cash(self):
        broker = PaperBroker(initial_cash=100)
        result = broker.submit_order({
            "action": "buy", "symbol": "AAPL", "shares": 10, "price": 100,
        })
        assert result["status"] == "rejected"
        assert "cash" in result["message"]

    def test_reject_no_price(self):
        broker = PaperBroker(initial_cash=10000)
        result = broker.submit_order({
            "action": "buy", "symbol": "AAPL", "shares": 10,
        })
        assert result["status"] == "rejected"
        assert "price" in result["message"]

    def test_reject_sell_without_position(self):
        broker = PaperBroker(initial_cash=10000)
        result = broker.submit_order({
            "action": "sell", "symbol": "AAPL", "shares": 10, "price": 100,
        })
        assert result["status"] == "rejected"

    def test_partial_sell(self):
        broker = PaperBroker(initial_cash=10000)
        broker.submit_order({"action": "buy", "symbol": "X", "shares": 5, "price": 10})
        result = broker.submit_order({"action": "sell", "symbol": "X", "shares": 10, "price": 20})
        assert result["status"] == "filled"
        assert result["filled_shares"] == 5  # only had 5

    def test_fill_price_callback(self):
        def cb(signal, side):
            return 999.0

        broker = PaperBroker(initial_cash=100000, fill_price_callback=cb)
        result = broker.submit_order({
            "action": "buy", "symbol": "X", "shares": 1,
        })
        assert result["status"] == "filled"
        assert result["fill_price"] == 999.0

    def test_get_orders(self):
        broker = PaperBroker(initial_cash=10000)
        broker.submit_order({"action": "buy", "symbol": "X", "shares": 1, "price": 50})
        orders = broker.get_orders()
        assert len(orders) == 1
        assert orders[0]["status"] == "filled"

    def test_account_summary(self):
        broker = PaperBroker(initial_cash=10000)
        summary = broker.get_account_summary()
        assert summary["cash"] == 10000
        assert summary["positions"] == {}

    def test_from_backtest_config(self):
        cfg = BacktestConfig.conservative()
        broker = PaperBroker.from_backtest_config(cfg, initial_cash=10000)
        result = broker.submit_order(
            {"action": "buy", "symbol": "AAPL", "shares": 10, "price": 100}
        )
        assert result["status"] == "filled"
        assert result["commission"] > 0
