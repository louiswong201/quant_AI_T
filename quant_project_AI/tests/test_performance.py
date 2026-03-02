"""
PerformanceAnalyzer 单元测试
"""

import numpy as np
import pandas as pd
import pytest

from quant_framework.analysis.performance import PerformanceAnalyzer


class TestPerformanceAnalyzer:
    def test_basic_metrics(self):
        analyzer = PerformanceAnalyzer(risk_free_rate=0.03)
        pv = np.linspace(100000, 120000, 252)  # ~20% return over 252 days
        dr = np.diff(pv) / pv[:-1]
        dr = np.insert(dr, 0, 0.0)
        result = analyzer.analyze(pv, dr, 100000)
        assert result["total_return"] == pytest.approx(0.2, rel=1e-5)
        assert result["trading_days"] == 252
        assert result["final_value"] == pytest.approx(120000)

    def test_empty_portfolio(self):
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(np.array([]), np.array([]), 100000)
        assert result == {}

    def test_max_drawdown(self):
        from quant_framework.analysis.performance import max_drawdown
        pv = np.array([100, 110, 105, 95, 100, 108], dtype=float)
        dd_val, peak_idx, trough_idx, recovery_idx = max_drawdown(pv)
        assert dd_val < 0
        assert dd_val == pytest.approx((95 - 110) / 110)

    def test_analyze_trades_empty(self):
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze_trades(pd.DataFrame())
        assert result["total_trades"] == 0

    def test_analyze_trades_with_data(self):
        analyzer = PerformanceAnalyzer()
        trades = pd.DataFrame([
            {"date": "2024-01-01", "action": "buy", "price": 100, "shares": 10, "symbol": "X"},
            {"date": "2024-01-10", "action": "sell", "price": 110, "shares": 10, "symbol": "X"},
            {"date": "2024-02-01", "action": "buy", "price": 100, "shares": 5, "symbol": "X"},
            {"date": "2024-02-10", "action": "sell", "price": 90, "shares": 5, "symbol": "X"},
        ])
        result = analyzer.analyze_trades(trades)
        assert result["total_trades"] == 2
        assert result["win_rate"] == pytest.approx(0.5)
