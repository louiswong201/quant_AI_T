"""
回测-实盘一致性回放测试
"""

import pandas as pd

from quant_framework.live import (
    analyze_execution_divergence,
    export_execution_diagnostics_bundle,
    export_execution_divergence_report,
)


class TestReplay:
    def test_basic_match(self):
        bt = pd.DataFrame(
            [
                {"date": "2024-01-01 10:00:00", "action": "buy", "symbol": "AAPL", "shares": 10, "price": 100.0},
                {"date": "2024-01-01 10:10:00", "action": "sell", "symbol": "AAPL", "shares": 10, "price": 101.0},
            ]
        )
        lv = pd.DataFrame(
            [
                {"date": "2024-01-01 10:00:30", "action": "buy", "symbol": "AAPL", "shares": 10, "price": 100.1},
                {"date": "2024-01-01 10:10:20", "action": "sell", "symbol": "AAPL", "shares": 10, "price": 100.9},
            ]
        )
        out = analyze_execution_divergence(bt, lv, time_tolerance_seconds=120)
        assert out["matched"] == 2
        assert out["missing_in_live"] == 0

    def test_missing_live(self):
        bt = pd.DataFrame([{"date": "2024-01-01", "action": "buy", "symbol": "AAPL", "shares": 1, "price": 1.0}])
        lv = pd.DataFrame()
        out = analyze_execution_divergence(bt, lv)
        assert out["missing_in_live"] == 1

    def test_export_report(self, tmp_path):
        summary = {
            "matched": 1,
            "backtest_count": 2,
            "live_count": 1,
            "missing_in_live": 1,
            "extra_in_live": 0,
            "mean_delay_seconds": 0.5,
            "p95_delay_seconds": 1.0,
            "mean_price_diff_bps": 3.2,
            "p95_price_diff_bps": 5.6,
            "within_price_tolerance_count": 1,
        }
        out = export_execution_divergence_report(summary, str(tmp_path / "report.md"))
        text = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert out.endswith("report.md")
        assert "Execution Divergence Report" in text

    def test_export_bundle(self, tmp_path):
        bt = pd.DataFrame([{"date": "2024-01-01", "action": "buy", "symbol": "AAPL", "shares": 1, "price": 1.0}])
        lv = pd.DataFrame([{"date": "2024-01-01", "action": "buy", "symbol": "AAPL", "shares": 1, "price": 1.1}])
        summary = analyze_execution_divergence(bt, lv, time_tolerance_seconds=3600)
        out = export_execution_diagnostics_bundle(
            summary=summary,
            backtest_trades=bt,
            live_fills=lv,
            output_dir=str(tmp_path),
        )
        assert (tmp_path / "execution_divergence_report.md").exists()
        assert (tmp_path / "backtest_trades.csv").exists()
        assert (tmp_path / "live_fills.csv").exists()
        assert out["report_path"].endswith("execution_divergence_report.md")

