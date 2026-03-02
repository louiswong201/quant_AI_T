"""
执行偏差诊断示例：
1) 回测时开启 auto_export_execution_report；
2) 传入 live_fills（实盘/纸盘成交）；
3) 自动导出 docs/execution_divergence_report.md 与 CSV 诊断包。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from quant_framework import BacktestConfig, BacktestEngine, DataManager
from quant_framework.strategy import MovingAverageStrategy


def main() -> None:
    dm = DataManager(data_dir="data")
    cfg = BacktestConfig(
        auto_export_execution_report=True,
        execution_report_path="docs/execution_divergence_report.md",
        max_participation_rate=0.2,
        impact_bps_buy_coeff=150.0,
        impact_bps_sell_coeff=120.0,
        impact_exponent=1.2,
        adaptive_impact=True,
    )
    engine = BacktestEngine(dm, config=cfg)
    strategy = MovingAverageStrategy(short_window=5, long_window=20, initial_capital=1_000_000)

    # 示例：真实场景请替换为实盘成交记录
    live_fills = pd.DataFrame(
        [
            {"date": "2024-02-01 10:00:00", "action": "buy", "symbol": "STOCK", "shares": 100, "price": 100.1},
            {"date": "2024-03-01 10:00:00", "action": "sell", "symbol": "STOCK", "shares": 100, "price": 101.2},
        ]
    )

    result = engine.run(
        strategy=strategy,
        symbols="STOCK",
        start_date="2024-01-01",
        end_date="2024-06-30",
        live_fills=live_fills,
    )
    print("final value:", result["final_value"])
    print("execution divergence:", result["execution_divergence"])
    print("report:", result["execution_report_path"])
    print("bundle:", result["execution_bundle"])


if __name__ == "__main__":
    main()

