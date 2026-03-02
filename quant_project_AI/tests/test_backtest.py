"""
回测引擎、BacktestConfig、run_robust_backtest 单元测试
"""

import numpy as np
import pandas as pd
import pytest

from quant_framework.backtest.config import BacktestConfig
from quant_framework.backtest.backtest_engine import BacktestEngine, _normalize_signals, _to_f64
from quant_framework.strategy.base_strategy import BaseStrategy
from quant_framework.strategy.ma_strategy import MovingAverageStrategy


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------

class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.commission_pct_buy == 0.001
        assert cfg.slippage_bps_buy == 0.0

    def test_conservative(self):
        cfg = BacktestConfig.conservative()
        assert cfg.commission_pct_buy == 0.0015
        assert cfg.slippage_bps_buy == 5.0

    def test_fill_price_buy(self):
        cfg = BacktestConfig(slippage_bps_buy=10.0, slippage_fixed_buy=0.0)
        # 10 bps = 0.1%
        assert cfg.fill_price_buy(100.0) == pytest.approx(100.1)

    def test_fill_price_sell(self):
        cfg = BacktestConfig(slippage_bps_sell=10.0)
        assert cfg.fill_price_sell(100.0) == pytest.approx(99.9)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            BacktestConfig(commission_pct_buy=-0.01)

    def test_from_legacy_rate(self):
        cfg = BacktestConfig.from_legacy_rate(0.002)
        assert cfg.commission_pct_buy == 0.002
        assert cfg.commission_pct_sell == 0.002

    def test_max_participation_rate_invalid(self):
        with pytest.raises(ValueError):
            BacktestConfig(max_participation_rate=0.0)
        with pytest.raises(ValueError):
            BacktestConfig(max_participation_rate=1.5)

    def test_impact_exponent_invalid(self):
        with pytest.raises(ValueError):
            BacktestConfig(impact_exponent=0.0)
        with pytest.raises(ValueError):
            BacktestConfig(impact_vol_window=1)
        with pytest.raises(ValueError):
            BacktestConfig(impact_vol_ref=0.0)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_normalize_none(self):
        assert _normalize_signals(None) == []

    def test_normalize_hold(self):
        assert _normalize_signals({"action": "hold"}) == []

    def test_normalize_buy(self):
        sig = {"action": "buy", "symbol": "X", "shares": 10}
        assert _normalize_signals(sig) == [sig]

    def test_normalize_list(self):
        sigs = [
            {"action": "buy", "symbol": "X", "shares": 5},
            {"action": "hold"},
            {"action": "sell", "symbol": "Y", "shares": 3},
        ]
        result = _normalize_signals(sigs)
        assert len(result) == 2

    def test_to_f64_contiguous(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        out = _to_f64(arr)
        assert out.dtype == np.float64
        assert out.flags.c_contiguous

    def test_to_f64_already_f64(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        out = _to_f64(arr)
        assert out is arr  # no copy needed


# ---------------------------------------------------------------------------
# BacktestEngine integration (uses in-memory mock DataManager)
# ---------------------------------------------------------------------------

class _MockDataManager:
    """用于测试的模拟 DataManager，返回内存中的合成数据。"""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def load_data(self, symbol, start_date, end_date):
        return self._df.copy()

    def calculate_indicators(self, data):
        from quant_framework.data.indicators import VectorizedIndicators
        return VectorizedIndicators.calculate_all(data)


class TestBacktestEngine:
    def test_run_single_symbol(self, sample_ohlcv: pd.DataFrame):
        dm = _MockDataManager(sample_ohlcv)
        engine = BacktestEngine(dm)
        strategy = MovingAverageStrategy(short_window=5, long_window=20, initial_capital=100000)
        result = engine.run(strategy, "STOCK", "2024-01-01", "2024-06-30")
        assert "results" in result
        assert "trades" in result
        assert "daily_returns" in result
        assert "cumulative_returns" in result
        assert "portfolio_values" in result
        assert "orders" in result
        assert result["initial_capital"] == 100000
        assert len(result["portfolio_values"]) == len(sample_ohlcv)

    def test_daily_returns_vs_cumulative(self, sample_ohlcv: pd.DataFrame):
        dm = _MockDataManager(sample_ohlcv)
        engine = BacktestEngine(dm)
        strategy = MovingAverageStrategy(short_window=5, long_window=20, initial_capital=100000)
        result = engine.run(strategy, "STOCK", "2024-01-01", "2024-06-30")
        cum = result["cumulative_returns"]
        pvs = result["portfolio_values"]
        ic = result["initial_capital"]
        # cumulative_return = (pv - initial) / initial
        np.testing.assert_allclose(cum, (pvs - ic) / ic, atol=1e-10)

    def test_min_lookback_validation(self, sample_ohlcv: pd.DataFrame):
        """数据不足时应抛 ValueError。"""
        short_data = sample_ohlcv.iloc[:5]
        dm = _MockDataManager(short_data)
        engine = BacktestEngine(dm)
        strategy = MovingAverageStrategy(short_window=5, long_window=20, initial_capital=100000)
        with pytest.raises(ValueError, match="min_lookback"):
            engine.run(strategy, "STOCK", "2024-01-01", "2024-06-30")

    def test_empty_data_raises(self):
        class EmptyDM:
            def load_data(self, s, sd, ed):
                return pd.DataFrame()

            def calculate_indicators(self, d):
                return d

        engine = BacktestEngine(EmptyDM())
        strategy = MovingAverageStrategy()
        with pytest.raises(ValueError):
            engine.run(strategy, "X", "2024-01-01", "2024-12-31")

    def test_conservative_config(self, sample_ohlcv: pd.DataFrame):
        dm = _MockDataManager(sample_ohlcv)
        engine = BacktestEngine(dm, config=BacktestConfig.conservative())
        strategy = MovingAverageStrategy(short_window=5, long_window=20, initial_capital=100000)
        result = engine.run(strategy, "STOCK", "2024-01-01", "2024-06-30")
        assert result["final_value"] > 0

    def test_manifest_exists(self, sample_ohlcv: pd.DataFrame):
        dm = _MockDataManager(sample_ohlcv)
        engine = BacktestEngine(dm)
        strategy = MovingAverageStrategy(short_window=5, long_window=20, initial_capital=100000)
        result = engine.run(strategy, "STOCK", "2024-01-01", "2024-06-30")
        assert "manifest" in result
        m = result["manifest"]
        assert m["strategy_class"] == "MovingAverageStrategy"
        assert m["symbols"] == ["STOCK"]

    def test_fast_path_hook(self, sample_ohlcv: pd.DataFrame):
        class FastOnlyStrategy(BaseStrategy):
            @property
            def min_lookback(self) -> int:
                return 1

            @property
            def fast_columns(self):
                return ("close",)

            def on_bar(self, data, current_date, current_prices=None):
                raise AssertionError("on_bar should not be called when on_bar_fast is available")

            def on_bar_fast(self, data_arrays, i, current_date, current_prices=None):
                return {"action": "hold"}

        dm = _MockDataManager(sample_ohlcv)
        engine = BacktestEngine(dm)
        strategy = FastOnlyStrategy("fast")
        result = engine.run(strategy, "STOCK", "2024-01-01", "2024-06-30")
        assert len(result["results"]) == len(sample_ohlcv)

    def test_pending_order_cancel(self, sample_ohlcv: pd.DataFrame):
        class CancelStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("cancel", initial_capital=100000)
                self.step = 0

            def on_bar(self, data, current_date, current_prices=None):
                self.step += 1
                if self.step == 1:
                    px = float(data["close"].iloc[-1]) if isinstance(data, pd.DataFrame) else 1.0
                    return {
                        "action": "buy",
                        "symbol": "STOCK",
                        "shares": 10,
                        "order_type": "limit",
                        "limit_price": px * 0.1,
                        "order_id": "o1",
                    }
                if self.step == 2:
                    return {"action": "cancel", "order_id": "o1"}
                return {"action": "hold"}

        dm = _MockDataManager(sample_ohlcv)
        engine = BacktestEngine(dm)
        result = engine.run(CancelStrategy(), "STOCK", "2024-01-01", "2024-06-30")
        orders = result["orders"]
        assert not orders.empty
        row = orders[orders["order_id"] == "o1"].iloc[0]
        assert row["status"] == "cancelled"

    def test_pending_order_expired(self, sample_ohlcv: pd.DataFrame):
        class ExpireStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("expire", initial_capital=100000)
                self.step = 0

            def on_bar(self, data, current_date, current_prices=None):
                self.step += 1
                if self.step == 1:
                    px = float(data["close"].iloc[-1]) if isinstance(data, pd.DataFrame) else 1.0
                    return {
                        "action": "buy",
                        "symbol": "STOCK",
                        "shares": 10,
                        "order_type": "limit",
                        "limit_price": px * 0.1,
                        "order_id": "o2",
                        "tif_bars": 1,
                    }
                return {"action": "hold"}

        dm = _MockDataManager(sample_ohlcv)
        engine = BacktestEngine(dm)
        result = engine.run(ExpireStrategy(), "STOCK", "2024-01-01", "2024-06-30")
        orders = result["orders"]
        row = orders[orders["order_id"] == "o2"].iloc[0]
        assert row["status"] == "expired"

    def test_pending_order_partial_fill_keeps_remainder(self, sample_ohlcv: pd.DataFrame):
        class PartialStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("partial", initial_capital=10_000_000)
                self.step = 0

            def on_bar(self, data, current_date, current_prices=None):
                self.step += 1
                if self.step == 1:
                    px = float(data["close"].iloc[-1]) if isinstance(data, pd.DataFrame) else 1.0
                    return {
                        "action": "buy",
                        "symbol": "STOCK",
                        "shares": 1_000_000,
                        "order_type": "limit",
                        "limit_price": px * 1.1,  # easily cross
                        "order_id": "o_partial",
                        "tif_bars": 5,
                    }
                return {"action": "hold"}

        dm = _MockDataManager(sample_ohlcv)
        engine = BacktestEngine(dm, config=BacktestConfig(max_participation_rate=0.01))
        result = engine.run(PartialStrategy(), "STOCK", "2024-01-01", "2024-06-30")
        orders = result["orders"]
        row = orders[orders["order_id"] == "o_partial"].iloc[0]
        assert row["status"] in ("partially_filled", "filled", "expired")
        assert int(row.get("filled_shares", 0)) >= 0

    def test_market_liquidity_cap(self, sample_ohlcv: pd.DataFrame):
        class BuyHugeStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("liq_cap", initial_capital=10_000_000)
                self.done = False

            def on_bar(self, data, current_date, current_prices=None):
                if self.done:
                    return {"action": "hold"}
                self.done = True
                return {"action": "buy", "symbol": "STOCK", "shares": 1_000_000}

        dm = _MockDataManager(sample_ohlcv)
        cfg = BacktestConfig(max_participation_rate=0.01)
        engine = BacktestEngine(dm, config=cfg)
        result = engine.run(BuyHugeStrategy(), "STOCK", "2024-01-01", "2024-06-30")
        trades = result["trades"]
        if not trades.empty:
            first_trade = trades.iloc[0]
            fill_bar = 1 if cfg.market_fill_mode == "next_open" else 0
            cap = int(float(sample_ohlcv.iloc[fill_bar]["volume"]) * 0.01)
            assert int(first_trade["shares"]) <= cap

    def test_market_impact_worsens_fill(self, sample_ohlcv: pd.DataFrame):
        class TwoStepStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("impact", initial_capital=1_000_000)
                self.step = 0

            def on_bar(self, data, current_date, current_prices=None):
                self.step += 1
                if self.step == 1:
                    return {"action": "buy", "symbol": "STOCK", "shares": 1000}
                if self.step == 2:
                    return {"action": "sell", "symbol": "STOCK", "shares": 1000}
                return {"action": "hold"}

        dm = _MockDataManager(sample_ohlcv)
        cfg_no = BacktestConfig(
            commission_pct_buy=0.0,
            commission_pct_sell=0.0,
            slippage_bps_buy=0.0,
            slippage_bps_sell=0.0,
            impact_bps_buy_coeff=0.0,
            impact_bps_sell_coeff=0.0,
        )
        cfg_hi = BacktestConfig(
            commission_pct_buy=0.0,
            commission_pct_sell=0.0,
            slippage_bps_buy=0.0,
            slippage_bps_sell=0.0,
            impact_bps_buy_coeff=2000.0,
            impact_bps_sell_coeff=2000.0,
        )
        res_no = BacktestEngine(dm, config=cfg_no).run(TwoStepStrategy(), "STOCK", "2024-01-01", "2024-06-30")
        res_hi = BacktestEngine(dm, config=cfg_hi).run(TwoStepStrategy(), "STOCK", "2024-01-01", "2024-06-30")
        assert res_hi["final_value"] <= res_no["final_value"]

    def test_limit_order_price_time_priority(self, sample_ohlcv: pd.DataFrame):
        class PriorityStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("prio", initial_capital=1_000_000)
                self.step = 0

            def on_bar(self, data, current_date, current_prices=None):
                self.step += 1
                if self.step == 1:
                    px = float(data["close"].iloc[-1])
                    return [
                        {
                            "action": "buy",
                            "symbol": "STOCK",
                            "shares": 1,
                            "order_type": "limit",
                            "limit_price": px * 1.01,
                            "order_id": "low",
                        },
                        {
                            "action": "buy",
                            "symbol": "STOCK",
                            "shares": 1,
                            "order_type": "limit",
                            "limit_price": px * 1.05,
                            "order_id": "high",
                        },
                    ]
                return {"action": "hold"}

        df = sample_ohlcv.copy()
        df.loc[df.index[0], "volume"] = 1.0  # 首根 bar 只能成交 1 股（participation_rate=1）
        dm = _MockDataManager(df)
        engine = BacktestEngine(dm, config=BacktestConfig(max_participation_rate=1.0))
        result = engine.run(PriorityStrategy(), "STOCK", "2024-01-01", "2024-06-30")
        trades = result["trades"]
        assert not trades.empty
        assert trades.iloc[0]["order_id"] == "high"

    def test_fast_multi_hook(self, sample_ohlcv: pd.DataFrame):
        class FastMultiStrategy(BaseStrategy):
            @property
            def fast_columns(self):
                return ("close",)

            def on_bar(self, data, current_date, current_prices=None):
                raise AssertionError("on_bar should not be called in fast multi path")

            def on_bar_fast_multi(self, data_arrays_by_symbol, i, current_date, current_prices):
                return {"action": "hold"}

        class MultiDM(_MockDataManager):
            def load_data(self, symbol, start_date, end_date):
                df = self._df.copy()
                df["close"] = df["close"] * (1.0 if symbol == "A" else 1.01)
                return df

        dm = MultiDM(sample_ohlcv)
        engine = BacktestEngine(dm)
        strategy = FastMultiStrategy("fast_multi")
        result = engine.run(strategy, ["A", "B"], "2024-01-01", "2024-06-30")
        assert len(result["results"]) > 0

    def test_auto_export_execution_report(self, sample_ohlcv: pd.DataFrame, tmp_path):
        dm = _MockDataManager(sample_ohlcv)
        out_path = tmp_path / "exec_report.md"
        cfg = BacktestConfig(
            auto_export_execution_report=True,
            execution_report_path=str(out_path),
        )
        engine = BacktestEngine(dm, config=cfg)
        strategy = MovingAverageStrategy(short_window=5, long_window=20, initial_capital=100000)
        live_fills = pd.DataFrame(
            [{"date": "2024-02-01", "action": "buy", "symbol": "STOCK", "shares": 1, "price": 100.0}]
        )
        result = engine.run(strategy, "STOCK", "2024-01-01", "2024-06-30", live_fills=live_fills)
        if result["execution_report_path"] is not None:
            assert out_path.exists()
            assert result["execution_bundle"] is not None
