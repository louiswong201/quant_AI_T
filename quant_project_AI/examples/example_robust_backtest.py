"""
稳健回测示例：多区间 + 多参数，看分布不只看单点

做法：
- 使用 BacktestConfig.conservative() 保守成本；
- 多个不重叠时间窗口、多组策略参数各跑一次；
- 汇总各 (窗口, 参数) 的收益率，输出 min/median/max，避免只报告「最优一组」。

可选：使用框架 run_robust_backtest 一行完成多区间×多参数回测（见下方注释）。
依赖：quant_framework，且 data 目录下有对应标的的历史数据（否则会跳过无数据的窗口）。
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_framework import DataManager, BacktestEngine
from quant_framework.backtest import BacktestConfig, run_robust_backtest
from quant_framework.strategy import MovingAverageStrategy


def run_one(dm, config, symbol: str, start: str, end: str, short: int, long: int):
    """单次回测，返回区间收益率（无法跑时返回 None）。"""
    try:
        strategy = MovingAverageStrategy(
            name="MA",
            initial_capital=1_000_000,
            short_window=short,
            long_window=long,
        )
        engine = BacktestEngine(dm, config=config)
        res = engine.run(strategy, symbol, start, end)
        initial = res["initial_capital"]
        final = res["final_value"]
        if initial and initial > 0:
            return (final - initial) / initial
    except Exception:
        pass
    return None


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    dm = DataManager(data_dir=data_dir)

    # 保守成本：使用框架预设（佣金 0.15%，滑点 5 bps）
    config = BacktestConfig.conservative()

    symbol = "000001.SZ"  # 可改为你有的标的，如 "STOCK"
    # 多区间：不重叠窗口（若某段无数据会跳过）
    windows = [
        ("2018-01-01", "2019-12-31"),
        ("2020-01-01", "2021-12-31"),
        ("2022-01-01", "2023-12-31"),
    ]
    # 多参数：(short_ma, long_ma)
    param_sets = [(5, 20), (10, 30), (20, 60)]

    # (窗口索引, 参数索引) -> 收益率
    table = {}
    for wi, (start, end) in enumerate(windows):
        for pi, (short, long) in enumerate(param_sets):
            ret = run_one(dm, config, symbol, start, end, short, long)
            table[(wi, pi)] = ret

    # 汇总
    print("稳健回测汇总（多区间 + 多参数）")
    print(f"标的: {symbol}, 成本: 佣金 0.15% + 滑点 5 bps")
    print(f"区间: {windows}")
    print(f"参数(short, long): {param_sets}\n")

    # 按参数：在该参数下各区间的收益
    print("按参数组 — 各区间收益率（无数据为 None）:")
    for pi, (short, long) in enumerate(param_sets):
        arr = [table[(wi, pi)] for wi in range(len(windows)) if table.get((wi, pi)) is not None]
        if not arr:
            print(f"  ({short}, {long}): 无有效数据")
            continue
        print(f"  ({short}, {long}): 区间收益 {[round(r, 4) for r in arr]} -> min={min(arr):.4f}, median={np.median(arr):.4f}, max={max(arr):.4f}")

    # 按区间：在该区间下各参数的收益
    print("\n按区间 — 各参数收益率:")
    for wi, (start, end) in enumerate(windows):
        arr = [table.get((wi, pi)) for pi in range(len(param_sets)) if table.get((wi, pi)) is not None]
        if not arr:
            print(f"  {start} ~ {end}: 无有效数据")
            continue
        print(f"  {start} ~ {end}: 参数收益 {[round(r, 4) for r in arr]} -> min={min(arr):.4f}, median={np.median(arr):.4f}, max={max(arr):.4f}")

    print("\n建议：若某参数在多区间下多数为负，或某区间下所有参数都负，则策略在该处不稳，不宜只报告最优一组。详见 docs/ROBUST_BACKTEST.md")


if __name__ == "__main__":
    main()
