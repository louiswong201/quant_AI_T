"""
Tests for v2.0 new modules: protocols, fill_simulator, order_manager,
portfolio, TCA, bias_detector, circuit_breaker, alpha engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_framework.backtest.protocols import (
    BarData, Fill, Order, OrderSide, OrderStatus, OrderType,
)
from quant_framework.backtest.fill_simulator import CostModelFillSimulator
from quant_framework.backtest.order_manager import DefaultOrderManager
from quant_framework.backtest.portfolio import PortfolioTracker
from quant_framework.backtest.config import BacktestConfig
from quant_framework.backtest.tca import TransactionCostAnalyzer
from quant_framework.backtest.bias_detector import BiasDetector
from quant_framework.live.risk import CircuitBreaker, RiskConfig


def _make_bar_data(n: int = 100, base_price: float = 100.0) -> BarData:
    """Create synthetic BarData for testing."""
    close = np.linspace(base_price, base_price * 1.1, n)
    return BarData(
        open=close * 0.999,
        high=close * 1.005,
        low=close * 0.995,
        close=close,
        volume=np.full(n, 1_000_000.0),
        rolling_vol=np.full(n, 0.02),
    )


def _make_order(
    side: OrderSide = OrderSide.BUY,
    symbol: str = "BTC",
    shares: int = 10,
    order_type: OrderType = OrderType.MARKET,
    order_id: str = "test_1",
) -> Order:
    return Order(
        order_id=order_id,
        side=side,
        symbol=symbol,
        shares=shares,
        order_type=order_type,
        submitted_bar=0,
        submitted_date=pd.Timestamp("2024-01-01"),
    )


# ============================================================================
# Protocol / Value Object Tests
# ============================================================================

class TestProtocols:
    def test_order_defaults(self):
        o = Order(order_id="o1", side=OrderSide.BUY, symbol="BTC", shares=10)
        assert o.order_type == OrderType.MARKET
        assert o.status == OrderStatus.SUBMITTED
        assert o.filled_shares == 0

    def test_order_side_enum(self):
        assert OrderSide.BUY != OrderSide.SELL
        assert OrderSide.CANCEL != OrderSide.BUY

    def test_bar_data_contiguous(self):
        bd = _make_bar_data(50)
        assert bd.close.flags.c_contiguous
        assert len(bd.close) == 50


# ============================================================================
# FillSimulator Tests
# ============================================================================

class TestFillSimulator:
    def test_market_order_fills_at_open(self):
        config = BacktestConfig()
        sim = CostModelFillSimulator(config)
        bar = _make_bar_data()
        order = _make_order()

        sim.reset_bar()
        fills, remaining = sim.try_fill_pending([order], "BTC", bar, 5, pd.Timestamp("2024-01-05"))
        assert len(fills) == 1
        assert fills[0].exec_price > 0
        assert order.status == OrderStatus.FILLED

    def test_limit_buy_fills_when_low_touches(self):
        config = BacktestConfig()
        sim = CostModelFillSimulator(config)
        bar = _make_bar_data()
        limit_price = float(bar.close[5]) * 1.01

        order = Order(
            order_id="lim1", side=OrderSide.BUY, symbol="BTC", shares=10,
            order_type=OrderType.LIMIT, limit_price=limit_price,
            submitted_bar=4,
        )
        sim.reset_bar()
        fills, _ = sim.try_fill_pending([order], "BTC", bar, 5, pd.Timestamp("2024-01-05"))
        assert len(fills) == 1

    def test_limit_buy_misses_when_price_too_low(self):
        config = BacktestConfig()
        sim = CostModelFillSimulator(config)
        bar = _make_bar_data()
        limit_price = float(bar.low[5]) * 0.5

        order = Order(
            order_id="lim2", side=OrderSide.BUY, symbol="BTC", shares=10,
            order_type=OrderType.LIMIT, limit_price=limit_price,
            submitted_bar=4,
        )
        sim.reset_bar()
        fills, remaining = sim.try_fill_pending([order], "BTC", bar, 5, pd.Timestamp("2024-01-05"))
        assert len(fills) == 0
        assert len(remaining) == 1

    def test_execute_market_direct(self):
        config = BacktestConfig()
        sim = CostModelFillSimulator(config)
        bar = _make_bar_data()
        order = _make_order()

        sim.reset_bar()
        fill = sim.execute_market(order, 100.0, bar, 5, pd.Timestamp("2024-01-05"))
        assert fill is not None
        assert fill.exec_shares == 10
        assert fill.commission > 0

    def test_slippage_worsens_buy(self):
        config = BacktestConfig(slippage_bps_buy=50.0)
        sim = CostModelFillSimulator(config)
        bar = _make_bar_data()
        order = _make_order()

        sim.reset_bar()
        fill = sim.execute_market(order, 100.0, bar, 5, pd.Timestamp("2024-01-05"))
        assert fill is not None
        assert fill.exec_price > 100.0


# ============================================================================
# OrderManager Tests
# ============================================================================

class TestOrderManager:
    def test_submit_and_retrieve(self):
        mgr = DefaultOrderManager()
        o = _make_order()
        oid = mgr.submit(o)
        assert oid == o.order_id
        assert len(mgr.pending) == 1
        assert len(mgr.all_orders) == 1

    def test_cancel_order(self):
        mgr = DefaultOrderManager()
        o = _make_order()
        mgr.submit(o)
        assert mgr.cancel(o.order_id)
        assert len(mgr.pending) == 0
        assert o.status == OrderStatus.CANCELLED

    def test_cancel_nonexistent_returns_false(self):
        mgr = DefaultOrderManager()
        assert not mgr.cancel("nonexistent")

    def test_expire_stale(self):
        mgr = DefaultOrderManager()
        o = Order(
            order_id="exp1", side=OrderSide.BUY, symbol="BTC", shares=10,
            submitted_bar=0, tif_bars=3,
        )
        mgr.submit(o)
        expired = mgr.expire_stale(3, pd.Timestamp("2024-01-04"))
        assert len(expired) == 1
        assert expired[0].status == OrderStatus.EXPIRED
        assert len(mgr.pending) == 0

    def test_create_order_shortcut(self):
        mgr = DefaultOrderManager()
        o = mgr.create_order(
            side=OrderSide.SELL, symbol="ETH", shares=5,
            bar_index=2, date=pd.Timestamp("2024-01-03"),
        )
        assert o.status == OrderStatus.SUBMITTED
        assert len(mgr.pending) == 1
        assert o.order_id.startswith("ord_")


# ============================================================================
# PortfolioTracker Tests
# ============================================================================

class TestPortfolioTracker:
    def test_initial_state(self):
        pt = PortfolioTracker(100_000.0, 10)
        assert pt.cash == 100_000.0
        assert pt.portfolio_value == 100_000.0
        assert len(pt.positions) == 0

    def test_apply_buy_fill(self):
        pt = PortfolioTracker(100_000.0, 10)
        order = _make_order(side=OrderSide.BUY, shares=10)
        fill = Fill(
            order=order, exec_price=100.0, exec_shares=10,
            commission=10.0, bar_index=0, date=pd.Timestamp("2024-01-01"),
        )
        pt.apply_fill(fill)
        assert pt.cash == 100_000.0 - (100.0 * 10 + 10.0)
        assert pt.positions.get("BTC") == 10
        assert len(pt.trades) == 1

    def test_apply_sell_fill(self):
        pt = PortfolioTracker(100_000.0, 10)
        buy_order = _make_order(side=OrderSide.BUY, shares=10, order_id="b1")
        buy_fill = Fill(
            order=buy_order, exec_price=100.0, exec_shares=10,
            commission=5.0, bar_index=0, date=pd.Timestamp("2024-01-01"),
        )
        pt.apply_fill(buy_fill)

        sell_order = _make_order(side=OrderSide.SELL, shares=10, order_id="s1")
        sell_fill = Fill(
            order=sell_order, exec_price=110.0, exec_shares=10,
            commission=5.0, bar_index=1, date=pd.Timestamp("2024-01-02"),
        )
        pt.apply_fill(sell_fill)
        assert pt.positions.get("BTC") is None
        assert len(pt.trades) == 2

    def test_mark_to_market(self):
        pt = PortfolioTracker(100_000.0, 10)
        buy_order = _make_order(side=OrderSide.BUY, shares=10, order_id="b1")
        buy_fill = Fill(
            order=buy_order, exec_price=100.0, exec_shares=10,
            commission=0.0, bar_index=0, date=pd.Timestamp("2024-01-01"),
        )
        pt.apply_fill(buy_fill)
        pv = pt.mark_to_market({"BTC": 110.0})
        expected = (100_000.0 - 1000.0) + 10 * 110.0
        assert abs(pv - expected) < 0.01

    def test_record_bar(self):
        pt = PortfolioTracker(100_000.0, 5)
        pt.mark_to_market({})
        pt.record_bar(0, pd.Timestamp("2024-01-01"))
        result = pt.to_results_dict()
        assert result["portfolio_values"][0] == 100_000.0
        assert result["cumulative_returns"][0] == 0.0

    def test_can_afford(self):
        pt = PortfolioTracker(1000.0, 10)
        assert pt.can_afford(100.0, 9, 10.0)
        assert not pt.can_afford(100.0, 10, 50.0)


# ============================================================================
# TCA Tests
# ============================================================================

class TestTCA:
    def test_empty_trades(self):
        report = TransactionCostAnalyzer.analyze(
            pd.DataFrame(), np.array([0.01, -0.005]), 100_000.0
        )
        assert report.total_trades == 0
        assert report.total_cost == 0.0

    def test_basic_cost_decomposition(self):
        trades = pd.DataFrame([
            {"date": "2024-01-01", "action": "buy", "symbol": "BTC",
             "price": 100.0, "shares": 10, "commission": 5.0},
            {"date": "2024-01-05", "action": "sell", "symbol": "BTC",
             "price": 110.0, "shares": 10, "commission": 5.0},
        ])
        config = BacktestConfig(slippage_bps_buy=10.0, slippage_bps_sell=10.0)
        returns = np.array([0.001, 0.002, -0.001, 0.003, 0.005])
        report = TransactionCostAnalyzer.analyze(trades, returns, 100_000.0, config)
        assert report.total_trades == 2
        assert report.total_commission == 10.0
        assert report.total_slippage_est > 0
        assert report.total_cost > 10.0

    def test_markdown_output(self):
        trades = pd.DataFrame([
            {"date": "2024-01-01", "action": "buy", "symbol": "BTC",
             "price": 100.0, "shares": 10, "commission": 5.0},
        ])
        report = TransactionCostAnalyzer.analyze(
            trades, np.array([0.01]), 100_000.0
        )
        md = TransactionCostAnalyzer.to_markdown(report)
        assert "Transaction Cost Analysis" in md
        assert "Total Commission" in md


# ============================================================================
# BiasDetector Tests
# ============================================================================

class TestBiasDetector:
    def test_look_ahead_current_close(self):
        issues = BiasDetector.detect_look_ahead(pd.DataFrame(), "current_close")
        assert any("look-ahead" in i.lower() for i in issues)

    def test_look_ahead_next_open_clean(self):
        issues = BiasDetector.detect_look_ahead(pd.DataFrame(), "next_open")
        assert len(issues) == 0

    def test_survivorship_single_symbol_warning(self):
        issues = BiasDetector.detect_survivorship_bias(
            ["BTC"],
            {"BTC": pd.DataFrame({"date": ["2024-01-01"]})}
        )
        assert any("survivorship" in i.lower() for i in issues)

    def test_data_snooping_too_many_params(self):
        issues = BiasDetector.detect_data_snooping(1000, 50)
        assert any("data-snooped" in i.lower() or "snooping" in i.lower() for i in issues)

    def test_full_audit_passes(self):
        report = BiasDetector.full_audit(
            pd.DataFrame(),
            ["BTC", "ETH", "SOL"],
            {s: pd.DataFrame({"date": ["2024-01-01", "2024-06-01"]}) for s in ["BTC", "ETH", "SOL"]},
            market_fill_mode="next_open",
            n_param_combinations=5,
            n_oos_bars=500,
        )
        assert report.passed


# ============================================================================
# CircuitBreaker Tests
# ============================================================================

class TestCircuitBreaker:
    def test_initial_state_not_tripped(self):
        cb = CircuitBreaker(RiskConfig())
        assert not cb.is_tripped
        assert cb.check() is None

    def test_trips_on_daily_loss(self):
        cb = CircuitBreaker(RiskConfig(max_daily_loss_pct=0.05), initial_capital=100_000.0)
        cb.record_pnl(-6000.0)
        assert cb.is_tripped
        assert "loss" in cb.trip_reason.lower()
        assert cb.check() is not None

    def test_trips_on_consecutive_errors(self):
        cb = CircuitBreaker(RiskConfig(max_consecutive_errors=3))
        cb.record_error()
        cb.record_error()
        assert not cb.is_tripped
        cb.record_error()
        assert cb.is_tripped

    def test_success_resets_error_count(self):
        cb = CircuitBreaker(RiskConfig(max_consecutive_errors=3))
        cb.record_error()
        cb.record_error()
        cb.record_success()
        cb.record_error()
        assert not cb.is_tripped

    def test_manual_reset(self):
        cb = CircuitBreaker(RiskConfig(max_daily_loss_pct=0.01), initial_capital=100_000.0)
        cb.record_pnl(-2000.0)
        assert cb.is_tripped
        cb.reset()
        assert not cb.is_tripped

    def test_summary(self):
        cb = CircuitBreaker(RiskConfig())
        cb.record_success()
        s = cb.summary()
        assert s["tripped"] is False
        assert s["total_orders"] == 1


# ============================================================================
# Alpha Engine Tests
# ============================================================================

class TestAlphaOrderFlow:
    def test_ofi_shape(self):
        from quant_framework.alpha.order_flow import OrderFlowFeatures
        n = 100
        h = np.random.uniform(100, 110, n)
        l = h - np.random.uniform(1, 5, n)
        c = (h + l) / 2
        v = np.random.uniform(1000, 5000, n)
        ofi = OrderFlowFeatures.ofi(h, l, c, v, window=20)
        assert len(ofi) == n
        assert np.isnan(ofi[0])
        assert not np.isnan(ofi[25])

    def test_vpin_shape(self):
        from quant_framework.alpha.order_flow import OrderFlowFeatures
        n = 200
        c = np.cumsum(np.random.randn(n)) + 100
        v = np.random.uniform(1000, 5000, n)
        vpin = OrderFlowFeatures.vpin(c, v, n_buckets=20)
        assert len(vpin) == n

    def test_trade_imbalance_range(self):
        from quant_framework.alpha.order_flow import OrderFlowFeatures
        n = 100
        c = np.cumsum(np.random.randn(n)) + 100
        v = np.abs(np.random.randn(n)) * 1000 + 100
        ti = OrderFlowFeatures.trade_imbalance(c, v, window=20)
        valid = ti[~np.isnan(ti)]
        assert all(-1.0 <= x <= 1.0 for x in valid)


class TestAlphaCrossAsset:
    def test_rolling_beta_shape(self):
        from quant_framework.alpha.cross_asset import CrossAssetFeatures
        n = 200
        c1 = np.cumsum(np.random.randn(n)) + 100
        c2 = np.cumsum(np.random.randn(n)) + 50
        beta = CrossAssetFeatures.rolling_beta(c1, c2, window=30)
        assert len(beta) == n

    def test_rolling_correlation_bounds(self):
        from quant_framework.alpha.cross_asset import CrossAssetFeatures
        n = 200
        c1 = np.cumsum(np.random.randn(n)) + 100
        corr = CrossAssetFeatures.rolling_correlation(c1, c1 * 2 + 5, window=30)
        valid = corr[~np.isnan(corr)]
        assert all(-1.01 <= x <= 1.01 for x in valid)


class TestAlphaVolatility:
    def test_yang_zhang_positive(self):
        from quant_framework.alpha.volatility import VolatilityFeatures
        n = 100
        c = np.cumsum(np.abs(np.random.randn(n))) + 100
        o = c * 0.999
        h = c * 1.01
        l = c * 0.99
        yz = VolatilityFeatures.yang_zhang_vol(o, h, l, c, window=20)
        valid = yz[~np.isnan(yz)]
        assert all(v >= 0 for v in valid)

    def test_vol_ratio_shape(self):
        from quant_framework.alpha.volatility import VolatilityFeatures
        n = 200
        c = np.cumsum(np.random.randn(n)) + 100
        vr = VolatilityFeatures.vol_ratio(c, fast_window=10, slow_window=60)
        assert len(vr) == n


class TestFeatureEvaluator:
    def test_information_coefficient(self):
        from quant_framework.alpha.evaluator import FeatureEvaluator
        n = 200
        np.random.seed(42)
        good_feature = np.random.randn(n)
        fwd_ret = good_feature * 0.5 + np.random.randn(n) * 0.1
        noise = np.random.randn(n)
        result = FeatureEvaluator.information_coefficient(
            {"good": good_feature, "noise": noise},
            fwd_ret, rolling_window=30,
        )
        assert abs(result["good"]["ic_mean"]) > abs(result["noise"]["ic_mean"])

    def test_select_orthogonal(self):
        from quant_framework.alpha.evaluator import FeatureEvaluator
        n = 200
        np.random.seed(42)
        f1 = np.random.randn(n)
        f2 = f1 + np.random.randn(n) * 0.01  # highly correlated with f1
        f3 = np.random.randn(n)
        fwd = f1 * 0.5 + f3 * 0.3 + np.random.randn(n) * 0.1
        selected = FeatureEvaluator.select_orthogonal(
            {"f1": f1, "f2_corr": f2, "f3_indep": f3},
            fwd, max_correlation=0.7, min_icir=0.1,
        )
        assert "f1" in selected or "f2_corr" in selected
        if "f1" in selected:
            assert "f2_corr" not in selected
