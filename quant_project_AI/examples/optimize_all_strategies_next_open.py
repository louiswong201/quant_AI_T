"""
全策略参数优化（排除 Lorentzian）
================================

目标：
1) 在 next-open 成交模式下，对所有内置策略（除 Lorentzian）做参数网格搜索
2) 数值参数按 [x/2, x, 2x] 生成候选
3) 用训练集选参、测试集验证，输出详细报告

数据：BTC-USD (yfinance)
"""

import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_framework.backtest.backtest_engine import BacktestEngine
from quant_framework.backtest.config import BacktestConfig
from quant_framework.data.indicators import VectorizedIndicators
from quant_framework.strategy import (
    DriftRegimeStrategy,
    KAMAStrategy,
    MACDStrategy,
    MESAStrategy,
    MomentumBreakoutStrategy,
    MovingAverageStrategy,
    RSIStrategy,
    ZScoreReversionStrategy,
)


class InMemoryDataManager:
    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()

    def load_data(self, symbol, start_date, end_date):
        d = self._df
        mask = (d["date"] >= pd.Timestamp(start_date)) & (d["date"] <= pd.Timestamp(end_date))
        return d.loc[mask].copy().reset_index(drop=True)

    def calculate_indicators(self, data):
        return VectorizedIndicators.calculate_all(data)


def fetch_real_data(symbol: str = "BTC-USD", start: str = "2021-01-01", end: str = "2026-02-01") -> pd.DataFrame:
    import yfinance as yf

    print(f"下载数据: {symbol} {start} ~ {end}")
    raw = yf.download(symbol, start=start, end=end, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance 下载失败")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.index),
            "open": raw["Open"].astype(float).values,
            "high": raw["High"].astype(float).values,
            "low": raw["Low"].astype(float).values,
            "close": raw["Close"].astype(float).values,
            "volume": raw["Volume"].astype(float).values,
        }
    ).dropna(subset=["close"]).reset_index(drop=True)

    print(f"数据条数: {len(df)} | {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    return df


def max_drawdown(values: np.ndarray) -> float:
    peak = values[0]
    mdd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


def metrics_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    pvs = np.asarray(result["portfolio_values"], dtype=float)
    dr = np.asarray(result["daily_returns"], dtype=float)
    trades = result["trades"]
    initial = float(result["initial_capital"])
    final = float(result["final_value"])

    total_return = (final / initial - 1.0) * 100.0
    excess = dr - (0.04 / 252)
    std = float(np.std(excess))
    sharpe = float(np.mean(excess) / std * np.sqrt(252)) if std > 1e-12 else 0.0
    downside = excess[excess < 0]
    dstd = float(np.std(downside)) if len(downside) > 0 else 0.0
    sortino = float(np.mean(excess) / dstd * np.sqrt(252)) if dstd > 1e-12 else 0.0
    mdd = max_drawdown(pvs) * 100.0

    buy_n = sell_n = wins = 0
    if isinstance(trades, pd.DataFrame) and not trades.empty and "action" in trades.columns:
        buys = trades[trades["action"] == "buy"].reset_index(drop=True)
        sells = trades[trades["action"] == "sell"].reset_index(drop=True)
        buy_n = len(buys)
        sell_n = len(sells)
        pairs = min(buy_n, sell_n)
        for i in range(pairs):
            if float(sells.iloc[i]["price"]) > float(buys.iloc[i]["price"]):
                wins += 1
    pairs = min(buy_n, sell_n)
    win_rate = (wins / pairs * 100.0) if pairs > 0 else 0.0

    return {
        "return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(mdd, 2),
        "trades": int(buy_n + sell_n),
        "round_trips": int(pairs),
        "wins": int(wins),
        "win_rate": round(win_rate, 1),
        "final_value": round(final, 2),
    }


def tri_int(x: int) -> List[int]:
    vals = [max(1, int(round(x / 2))), int(x), max(1, int(round(2 * x)))]
    out = sorted(set(vals))
    return out


def tri_float(x: float, nd: int = 6) -> List[float]:
    vals = [round(float(x) / 2.0, nd), round(float(x), nd), round(float(x) * 2.0, nd)]
    out = sorted(set(vals))
    return out


@dataclass
class StrategySpec:
    name: str
    cls: Any
    defaults: Dict[str, Any]
    types: Dict[str, str]  # int / float / bool
    validator: Callable[[Dict[str, Any]], bool]


SPECS: List[StrategySpec] = [
    StrategySpec(
        name="MovingAverageStrategy",
        cls=MovingAverageStrategy,
        defaults={"short_window": 5, "long_window": 20},
        types={"short_window": "int", "long_window": "int"},
        validator=lambda p: p["short_window"] < p["long_window"],
    ),
    StrategySpec(
        name="RSIStrategy",
        cls=RSIStrategy,
        defaults={"rsi_period": 14, "oversold": 30.0, "overbought": 70.0},
        types={"rsi_period": "int", "oversold": "float", "overbought": "float"},
        validator=lambda p: 0 < p["oversold"] < p["overbought"] < 100,
    ),
    StrategySpec(
        name="MACDStrategy",
        cls=MACDStrategy,
        defaults={"fast_period": 12, "slow_period": 26, "signal_period": 9},
        types={"fast_period": "int", "slow_period": "int", "signal_period": "int"},
        validator=lambda p: p["fast_period"] < p["slow_period"] and p["signal_period"] < p["slow_period"],
    ),
    StrategySpec(
        name="DriftRegimeStrategy",
        cls=DriftRegimeStrategy,
        defaults={"lookback": 15, "drift_threshold": 0.62, "hold_period": 27},
        types={"lookback": "int", "drift_threshold": "float", "hold_period": "int"},
        validator=lambda p: 0 < p["drift_threshold"] < 1 and p["lookback"] >= 2 and p["hold_period"] >= 1,
    ),
    StrategySpec(
        name="ZScoreReversionStrategy",
        cls=ZScoreReversionStrategy,
        defaults={"lookback": 35, "entry_z": 2.5, "exit_z": 0.5, "stop_z": 3.0},
        types={"lookback": "int", "entry_z": "float", "exit_z": "float", "stop_z": "float"},
        validator=lambda p: p["lookback"] >= 5 and 0 < p["exit_z"] < p["entry_z"] < p["stop_z"],
    ),
    StrategySpec(
        name="MomentumBreakoutStrategy",
        cls=MomentumBreakoutStrategy,
        defaults={"high_period": 40, "proximity_pct": 0.03, "atr_period": 14, "atr_trail": 2.0},
        types={"high_period": "int", "proximity_pct": "float", "atr_period": "int", "atr_trail": "float"},
        validator=lambda p: p["high_period"] >= 5 and p["atr_period"] >= 5 and 0 < p["proximity_pct"] < 1 and p["atr_trail"] > 0,
    ),
    StrategySpec(
        name="KAMAStrategy",
        cls=KAMAStrategy,
        defaults={"er_period": 10, "fast_period": 2, "slow_period": 30, "atr_period": 14, "atr_stop_mult": 2.0},
        types={"er_period": "int", "fast_period": "int", "slow_period": "int", "atr_period": "int", "atr_stop_mult": "float"},
        validator=lambda p: p["er_period"] >= 2 and p["fast_period"] < p["slow_period"] and p["atr_period"] >= 5 and p["atr_stop_mult"] > 0,
    ),
    StrategySpec(
        name="MESAStrategy",
        cls=MESAStrategy,
        defaults={"fast_limit": 0.5, "slow_limit": 0.05},
        types={"fast_limit": "float", "slow_limit": "float"},
        validator=lambda p: 0 < p["slow_limit"] < p["fast_limit"] <= 1.0,
    ),
]


def build_grid(spec: StrategySpec) -> List[Dict[str, Any]]:
    keys = list(spec.defaults.keys())
    choices: List[List[Any]] = []
    for k in keys:
        v = spec.defaults[k]
        t = spec.types[k]
        if t == "int":
            choices.append(tri_int(int(v)))
        elif t == "float":
            choices.append(tri_float(float(v), nd=6))
        elif t == "bool":
            choices.append([True, False])
        else:
            raise ValueError(f"Unsupported type {t} for {k}")

    combos: List[Dict[str, Any]] = []
    for vals in itertools.product(*choices):
        p = {k: vals[i] for i, k in enumerate(keys)}
        if spec.validator(p):
            combos.append(p)
    return combos


def run_backtest_for_range(strategy_cls, params: Dict[str, Any], df: pd.DataFrame, start: str, end: str) -> Dict[str, Any]:
    dm = InMemoryDataManager(df)
    engine = BacktestEngine(
        dm,
        config=BacktestConfig(
            market_fill_mode="next_open",
            commission_pct_buy=0.0015,
            commission_pct_sell=0.0015,
            slippage_bps_buy=5.0,
            slippage_bps_sell=5.0,
            max_participation_rate=0.2,
            impact_bps_buy_coeff=5.0,
            impact_bps_sell_coeff=5.0,
            impact_exponent=1.0,
            adaptive_impact=True,
        ),
    )
    strategy = strategy_cls(initial_capital=1_000_000, **params)
    result = engine.run(strategy, "BTC", start, end)
    return metrics_from_result(result)


def select_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 优先 Sharpe，其次收益，再次低回撤
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["train_sharpe"], r["train_return_pct"], -r["train_max_dd_pct"]),
        reverse=True,
    )
    return rows_sorted[0]


def to_md_table(df: pd.DataFrame, cols: List[str], max_rows: int = 10) -> str:
    show = df[cols].head(max_rows).copy()
    show = show.fillna("")
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    body = ""
    for _, r in show.iterrows():
        body += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return header + sep + body


def main():
    t0 = time.time()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "results", "all_strategies_next_open")
    os.makedirs(out_dir, exist_ok=True)

    df = fetch_real_data("BTC-USD", "2021-01-01", "2026-02-01")
    df = df.copy()
    # 与框架符号一致
    symbol = "BTC"

    train_start, train_end = "2021-01-01", "2023-12-31"
    test_start, test_end = "2024-01-01", "2026-01-31"

    rows_summary: List[Dict[str, Any]] = []
    details: Dict[str, pd.DataFrame] = {}

    print("\n开始全策略网格搜索（next-open）")
    for spec in SPECS:
        print(f"\n[{spec.name}]")
        grid = build_grid(spec)
        print(f"参数组合数: {len(grid)}")

        default_params = dict(spec.defaults)
        default_train = run_backtest_for_range(spec.cls, default_params, df, train_start, train_end)
        default_test = run_backtest_for_range(spec.cls, default_params, df, test_start, test_end)

        all_rows = []
        st = time.time()
        for idx, p in enumerate(grid, start=1):
            train_m = run_backtest_for_range(spec.cls, p, df, train_start, train_end)
            row = {
                "strategy": spec.name,
                **p,
                "train_sharpe": train_m["sharpe"],
                "train_return_pct": train_m["return_pct"],
                "train_max_dd_pct": train_m["max_dd_pct"],
                "train_sortino": train_m["sortino"],
                "train_trades": train_m["trades"],
                "train_win_rate": train_m["win_rate"],
            }
            all_rows.append(row)
            if idx % 50 == 0 or idx == len(grid):
                elapsed = time.time() - st
                eta = elapsed / idx * (len(grid) - idx) if idx else 0
                print(f"  进度 {idx}/{len(grid)} | 已用 {elapsed:.1f}s | ETA {eta:.1f}s")

        detail_df = pd.DataFrame(all_rows).sort_values(["train_sharpe", "train_return_pct"], ascending=False)
        details[spec.name] = detail_df
        best = select_best(all_rows)

        best_params = {k: best[k] for k in spec.defaults.keys()}
        best_train = run_backtest_for_range(spec.cls, best_params, df, train_start, train_end)
        best_test = run_backtest_for_range(spec.cls, best_params, df, test_start, test_end)

        rows_summary.append(
            {
                "strategy": spec.name,
                "combos": len(grid),
                "default_train_sharpe": default_train["sharpe"],
                "default_test_sharpe": default_test["sharpe"],
                "best_train_sharpe": best_train["sharpe"],
                "best_test_sharpe": best_test["sharpe"],
                "default_test_return_pct": default_test["return_pct"],
                "best_test_return_pct": best_test["return_pct"],
                "default_test_max_dd_pct": default_test["max_dd_pct"],
                "best_test_max_dd_pct": best_test["max_dd_pct"],
                "default_test_win_rate": default_test["win_rate"],
                "best_test_win_rate": best_test["win_rate"],
                "generalization_ratio": round(
                    (best_test["sharpe"] - default_test["sharpe"]) /
                    (best_train["sharpe"] - default_train["sharpe"])
                    if (best_train["sharpe"] - default_train["sharpe"]) != 0
                    else 0.0,
                    3,
                ),
                "best_params": json.dumps(best_params, ensure_ascii=False),
            }
        )

        detail_path = os.path.join(out_dir, f"{spec.name}_grid_results.csv")
        detail_df.to_csv(detail_path, index=False)
        print(f"  完成: best test sharpe={best_test['sharpe']:.3f}, return={best_test['return_pct']:+.2f}%")

    summary_df = pd.DataFrame(rows_summary)
    summary_df = summary_df.sort_values(["best_test_sharpe", "best_test_return_pct"], ascending=False)
    summary_csv = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # 生成 Markdown 报告
    report_path = os.path.join(root, "docs", "ALL_STRATEGIES_NEXT_OPEN_REPORT.md")
    elapsed_total = time.time() - t0

    lines: List[str] = []
    lines.append("# 全策略 Next-Open 参数优化报告（排除 Lorentzian）\n")
    lines.append("## 结论概览\n")
    lines.append(f"- 数据: `BTC-USD` (yfinance), 2021-01-01 ~ 2026-01-31\n")
    lines.append(f"- 执行模型: `next_open`（信号在 bar[i] close 产生，bar[i+1] open 成交）\n")
    lines.append("- 成本模型: 佣金 0.15% 双边 + 5bps 滑点 + 自适应冲击 + 成交量参与率上限 20%\n")
    lines.append("- 训练/测试: 2021-2023 选参, 2024-2026 样本外验证\n")
    lines.append(f"- 总耗时: {elapsed_total:.1f} 秒\n")
    lines.append("\n## 全策略排名（按测试集 Sharpe）\n")

    rank_cols = [
        "strategy",
        "combos",
        "default_test_sharpe",
        "best_test_sharpe",
        "default_test_return_pct",
        "best_test_return_pct",
        "default_test_max_dd_pct",
        "best_test_max_dd_pct",
        "default_test_win_rate",
        "best_test_win_rate",
        "generalization_ratio",
    ]
    lines.append(to_md_table(summary_df, rank_cols, max_rows=50))

    lines.append("\n## 逐策略最优参数\n")
    for _, row in summary_df.iterrows():
        lines.append(f"### {row['strategy']}\n")
        lines.append(f"- 组合数: {row['combos']}\n")
        lines.append(f"- 默认 -> 最优 (测试 Sharpe): {row['default_test_sharpe']} -> {row['best_test_sharpe']}\n")
        lines.append(f"- 默认 -> 最优 (测试收益): {row['default_test_return_pct']}% -> {row['best_test_return_pct']}%\n")
        lines.append(f"- 默认 -> 最优 (测试回撤): {row['default_test_max_dd_pct']}% -> {row['best_test_max_dd_pct']}%\n")
        lines.append(f"- 默认 -> 最优 (测试胜率): {row['default_test_win_rate']}% -> {row['best_test_win_rate']}%\n")
        lines.append(f"- 泛化保持率: {row['generalization_ratio']}\n")
        lines.append(f"- 最优参数: `{row['best_params']}`\n")

        ddf = details[row["strategy"]]
        top_cols = [c for c in ddf.columns if c.startswith("train_") is False][:]  # keep all params
        param_cols = [c for c in ddf.columns if c not in {
            "strategy", "train_sharpe", "train_return_pct", "train_max_dd_pct", "train_sortino", "train_trades", "train_win_rate"
        }]
        show_cols = param_cols + ["train_sharpe", "train_return_pct", "train_max_dd_pct", "train_win_rate", "train_trades"]
        lines.append("\nTop 5 训练集参数组合:\n")
        lines.append(to_md_table(ddf, show_cols, max_rows=5))

    lines.append("\n## 方法与严谨性说明\n")
    lines.append("1. 市价成交采用 next-open，避免同 bar close 成交带来的前视偏差。\n")
    lines.append("2. 参数空间按 [x/2, x, 2x] 构造，并加入必要物理约束（如 fast < slow, RSI 阈值在 0~100）。\n")
    lines.append("3. 选参仅使用训练集，测试集完全独立；报告中单列泛化保持率用于识别过拟合。\n")
    lines.append("4. 成本模型采用偏保守设定，避免回测结果过于乐观。\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n======================= 完成 =======================")
    print(f"汇总CSV: {summary_csv}")
    print(f"详细CSV目录: {out_dir}")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
