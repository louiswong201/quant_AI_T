#!/usr/bin/env python3
"""
Comprehensive Leverage Exploration Backtest Analysis.

Extends full_strategy_analysis.py with multi-leverage exploration:
  1. Find optimal strategies via Walk-Forward + CPCV (1x baseline)
  2. Run top strategies at leverage levels: 1x, 1.5x, 2x, 3x, 5x
  3. Analyze risk/reward tradeoffs, Kelly criterion, optimal leverage
  4. Generate full report: reports/leverage_analysis_report.md

Usage:
    python examples/leverage_analysis.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_framework.backtest import (
    BacktestConfig,
    BacktestResult,
    OptimizeResult,
    backtest,
    optimize,
    KERNEL_NAMES,
    DEFAULT_PARAM_GRIDS,
)
from quant_framework.analysis.performance import PerformanceAnalyzer

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
CHART_DIR = REPORT_DIR / "leverage_charts"
INITIAL_CAPITAL = 100_000.0
RISK_FREE_RATE = 0.04

CRYPTO_ASSETS = ["BTC", "ETH", "SOL", "BNB"]
STOCK_ASSETS = ["SPY", "QQQ", "NVDA", "META", "MSFT", "AMZN"]
LEVERAGE_LEVELS = [1.0, 1.5, 2.0, 3.0, 5.0]
MIN_BARS_REQUIRED = 500
TOP_N = 5

plt.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})


def load_csv(symbol: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in df.columns:
        df.rename(columns={col: col.strip().lower()}, inplace=True)
    if "close" not in df.columns:
        return None
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_all_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    crypto, stocks = {}, {}
    for sym in CRYPTO_ASSETS:
        df = load_csv(sym)
        if df is not None and len(df) >= MIN_BARS_REQUIRED:
            crypto[sym] = df
    for sym in STOCK_ASSETS:
        df = load_csv(sym)
        if df is not None and len(df) >= MIN_BARS_REQUIRED:
            stocks[sym] = df
    return crypto, stocks


def run_optimization(
    data_map: Dict[str, pd.DataFrame],
    config: BacktestConfig,
    label: str,
) -> Tuple[Optional[OptimizeResult], Optional[OptimizeResult]]:
    if not data_map:
        return None, None
    print(f"\n{'='*60}")
    print(f"  Optimizing [{label}]: {list(data_map.keys())}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    wf_result = optimize(data_map, config, method="wf")
    wf_time = time.perf_counter() - t0
    print(f"  WF: {wf_result.total_combos:,} combos in {wf_time:.1f}s "
          f"({wf_result.total_combos / max(0.1, wf_time):,.0f}/s)")

    t0 = time.perf_counter()
    cpcv_result = optimize(data_map, config, method="cpcv")
    cpcv_time = time.perf_counter() - t0
    print(f"  CPCV: {cpcv_result.total_combos:,} combos in {cpcv_time:.1f}s")

    return wf_result, cpcv_result


def run_detailed_backtest_single(
    strat_name: str,
    params: Any,
    data_map: Dict[str, pd.DataFrame],
    config: BacktestConfig,
) -> Dict[str, Any]:
    per_asset = {}
    combined_equity = []
    for sym, df in data_map.items():
        try:
            res = backtest(strat_name, params, df, config, detailed=True)
            per_asset[sym] = res
            if res.equity is not None:
                combined_equity.append(res.equity)
        except Exception:
            pass

    if not combined_equity:
        return {}

    min_len = min(len(eq) for eq in combined_equity)
    avg_equity = np.mean([eq[:min_len] for eq in combined_equity], axis=0)
    portfolio_values = avg_equity * INITIAL_CAPITAL
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    analyzer = PerformanceAnalyzer(risk_free_rate=RISK_FREE_RATE, periods_per_year=252.0)
    perf = analyzer.analyze(portfolio_values, daily_returns, INITIAL_CAPITAL,
                            n_trials=len(KERNEL_NAMES))

    return {
        "strategy": strat_name,
        "params": params,
        "per_asset": per_asset,
        "portfolio_values": portfolio_values,
        "daily_returns": daily_returns,
        "performance": perf,
    }


def get_top_strategies(wf_result: OptimizeResult, n: int = TOP_N) -> List[Tuple[str, Any]]:
    ranked = sorted(
        wf_result.all_strategies.items(),
        key=lambda kv: kv[1].get("wf_score", -1e18),
        reverse=True,
    )[:n]
    return [(name, meta.get("params")) for name, meta in ranked if meta.get("params") is not None]


def run_leverage_exploration(
    top_strategies: List[Tuple[str, Any]],
    data_map: Dict[str, pd.DataFrame],
    base_config_fn,
    label: str,
) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {}

    for strat_name, params in top_strategies:
        strat_results = []
        for lev in LEVERAGE_LEVELS:
            config = base_config_fn(leverage=lev)
            print(f"    {strat_name} @ {lev:.1f}x leverage ...")
            r = run_detailed_backtest_single(strat_name, params, data_map, config)
            if r:
                r["leverage"] = lev
                strat_results.append(r)
        results[strat_name] = strat_results

    return results


def compute_kelly(daily_returns: np.ndarray) -> float:
    """Half-Kelly fraction."""
    if len(daily_returns) < 30:
        return 0.0
    mu = np.mean(daily_returns)
    var = np.var(daily_returns)
    if var < 1e-12:
        return 0.0
    kelly = mu / var
    return max(0, min(kelly / 2, 20.0))


def plot_leverage_comparison(
    lev_results: Dict[str, List[Dict[str, Any]]],
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, len(lev_results)))

    for idx, (strat, results_list) in enumerate(lev_results.items()):
        if not results_list:
            continue
        levs = [r["leverage"] for r in results_list]
        ann_rets = [r["performance"].get("annual_return", 0) * 100 for r in results_list]
        sharpes = [r["performance"].get("sharpe_ratio", 0) for r in results_list]
        mdds = [abs(r["performance"].get("max_drawdown", 0)) * 100 for r in results_list]
        calmars = [r["performance"].get("calmar_ratio", 0) for r in results_list]

        c = colors[idx]
        axes[0, 0].plot(levs, ann_rets, "o-", color=c, label=strat, linewidth=2, markersize=8)
        axes[0, 1].plot(levs, sharpes, "s-", color=c, label=strat, linewidth=2, markersize=8)
        axes[1, 0].plot(levs, mdds, "^-", color=c, label=strat, linewidth=2, markersize=8)
        axes[1, 1].plot(levs, calmars, "D-", color=c, label=strat, linewidth=2, markersize=8)

    axes[0, 0].set_title("Annual Return vs Leverage", fontweight="bold")
    axes[0, 0].set_ylabel("Annual Return (%)")
    axes[0, 0].set_xlabel("Leverage")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Sharpe Ratio vs Leverage", fontweight="bold")
    axes[0, 1].set_ylabel("Sharpe Ratio")
    axes[0, 1].set_xlabel("Leverage")
    axes[0, 1].axhline(1.0, color="green", linestyle=":", alpha=0.5)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Max Drawdown vs Leverage", fontweight="bold")
    axes[1, 0].set_ylabel("Max Drawdown (%)")
    axes[1, 0].set_xlabel("Leverage")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Calmar Ratio vs Leverage", fontweight="bold")
    axes[1, 1].set_ylabel("Calmar Ratio")
    axes[1, 1].set_xlabel("Leverage")
    axes[1, 1].axhline(1.0, color="green", linestyle=":", alpha=0.5)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {save_path.name}")


def plot_equity_by_leverage(
    results_list: List[Dict[str, Any]],
    strat_name: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(results_list)))

    for i, r in enumerate(results_list):
        pv = r["portfolio_values"]
        lev = r["leverage"]
        sr = r["performance"].get("sharpe_ratio", 0)
        label = f'{lev:.1f}x (Sharpe {sr:.2f})'
        axes[0].plot(pv, label=label, linewidth=1.5, color=cmap[i])

        peak = np.maximum.accumulate(pv)
        dd = (pv - peak) / peak * 100
        axes[1].plot(dd, linewidth=1.0, color=cmap[i], alpha=0.7)

    axes[0].axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title(f"Equity Curves by Leverage — {strat_name}", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Trading Days")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Chart saved: {save_path.name}")


def _sec(n: int) -> str:
    return ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
            "十一", "十二", "十三"][min(n, 13)]


def generate_report(
    crypto_data, stock_data,
    crypto_wf, crypto_cpcv, stock_wf, stock_cpcv,
    crypto_lev_results, stock_lev_results,
    crypto_config, stock_config,
    total_time,
) -> str:
    L = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    L.append("# 全策略回测 + 杠杆率探索分析报告")
    L.append(f"\n> 生成时间: {now} | 总耗时: {total_time:.1f}s")
    L.append(f"> 杠杆探索范围: {', '.join(f'{l:.1f}x' for l in LEVERAGE_LEVELS)}\n")

    # Executive Summary
    L.append("## 一、执行摘要\n")
    L.append(f"本报告对 **{len(crypto_data) + len(stock_data)}** 个资产的 "
             f"**{len(KERNEL_NAMES)}** 种策略进行参数优化后，"
             f"对 Top-{TOP_N} 策略在 **{len(LEVERAGE_LEVELS)} 种杠杆水平**下进行深度回测，"
             "分析杠杆对收益、风险、Sharpe、回撤的影响，推荐最优杠杆配置。\n")

    # Data Overview
    L.append("## 二、数据概览\n")
    L.append("| 资产 | 类型 | 数据天数 | 起始日期 | 结束日期 | 最新价格 |")
    L.append("|:-----|:-----|--------:|:---------|:---------|--------:|")
    for sym, df in {**crypto_data, **stock_data}.items():
        mtype = "加密货币" if sym in crypto_data else "美股"
        start = df["date"].iloc[0].strftime("%Y-%m-%d") if "date" in df.columns else "N/A"
        end = df["date"].iloc[-1].strftime("%Y-%m-%d") if "date" in df.columns else "N/A"
        price = df["close"].iloc[-1]
        L.append(f"| {sym} | {mtype} | {len(df):,} | {start} | {end} | ${price:,.2f} |")
    L.append("")

    # Cost Model
    L.append("## 三、成本模型\n")
    L.append("| 参数 | 加密货币 | 美股 |")
    L.append("|:-----|--------:|-----:|")
    L.append(f"| 手续费 | {crypto_config.commission_pct_buy*100:.2f}% | {stock_config.commission_pct_buy*100:.2f}% |")
    L.append(f"| 滑点 (bps) | {crypto_config.slippage_bps_buy:.0f} | {stock_config.slippage_bps_buy:.0f} |")
    L.append(f"| 资金费率/日 | {crypto_config.daily_funding_rate*100:.2f}% | 0% |")
    L.append(f"| 做空借贷率/年 | {crypto_config.short_borrow_rate_annual*100:.1f}% | {stock_config.short_borrow_rate_annual*100:.1f}% |")
    L.append(f"| 杠杆探索范围 | {', '.join(f'{l:.1f}x' for l in LEVERAGE_LEVELS)} | {', '.join(f'{l:.1f}x' for l in LEVERAGE_LEVELS)} |")
    L.append("")

    sec = 4

    # WF Optimization results (abbreviated)
    for label, wf, data_map in [("加密货币", crypto_wf, crypto_data), ("美股", stock_wf, stock_data)]:
        if wf is None:
            continue
        L.append(f"## {_sec(sec)}、Walk-Forward 优化结果 — {label}\n")
        sec += 1
        L.append(f"> 资产: {', '.join(data_map.keys())} | "
                 f"总组合: {wf.total_combos:,} | "
                 f"速度: {wf.total_combos / max(0.1, wf.elapsed_seconds):,.0f} combos/s\n")
        L.append("| 排名 | 策略 | 参数 | OOS收益 | Sharpe | 最大回撤 | WF得分 |")
        L.append("|-----:|:-----|:-----|--------:|-------:|--------:|-------:|")
        ranked = sorted(wf.all_strategies.items(), key=lambda kv: kv[1].get("wf_score", -1e18), reverse=True)
        for i, (sn, d) in enumerate(ranked, 1):
            ps = str(d.get("params", ""))
            if len(ps) > 28:
                ps = ps[:25] + "..."
            L.append(f"| {i} | **{sn}** | `{ps}` | "
                     f"{d.get('oos_ret', 0):+.1f}% | "
                     f"{d.get('sharpe', 0):.2f} | "
                     f"{d.get('oos_dd', 0):.1f}% | "
                     f"{d.get('wf_score', 0):.1f} |")
        L.append("")

    # Leverage Exploration
    for label, lev_results, data_map in [
        ("加密货币", crypto_lev_results, crypto_data),
        ("美股", stock_lev_results, stock_data),
    ]:
        if not lev_results:
            continue

        L.append(f"## {_sec(sec)}、杠杆率探索 — {label}\n")
        sec += 1

        for strat_name, results_list in lev_results.items():
            if not results_list:
                continue

            L.append(f"### {strat_name}\n")
            L.append("| 杠杆 | 年化收益 | Sharpe | Sortino | Calmar | 最大回撤 | 波动率 | 终值 | VaR 95% | Kelly |")
            L.append("|-----:|--------:|-------:|--------:|-------:|--------:|-------:|-----:|--------:|------:|")

            best_calmar_lev = 0
            best_calmar = -999
            for r in results_list:
                p = r["performance"]
                lev = r["leverage"]
                ann_ret = p.get("annual_return", 0)
                sr = p.get("sharpe_ratio", 0)
                sortino = p.get("sortino_ratio", 0)
                calmar = p.get("calmar_ratio", 0)
                mdd = p.get("max_drawdown", 0)
                vol = p.get("volatility", 0)
                fv = p.get("final_value", 0)
                var95 = p.get("value_at_risk_95", 0)
                kelly = compute_kelly(r["daily_returns"])

                if calmar > best_calmar:
                    best_calmar = calmar
                    best_calmar_lev = lev

                L.append(
                    f"| **{lev:.1f}x** | {ann_ret*100:+.2f}% | {sr:.3f} | {sortino:.3f} | "
                    f"{calmar:.3f} | {mdd*100:.2f}% | {vol*100:.2f}% | "
                    f"${fv:,.0f} | {var95*100:.2f}% | {kelly:.2f} |"
                )
            L.append("")

            # Per-asset breakdown at each leverage
            L.append(f"#### {strat_name} — 各资产各杠杆收益率\n")
            syms = list(data_map.keys())
            header = "| 杠杆 | " + " | ".join(syms) + " |"
            sep = "|-----:|" + "|".join(["------:"] * len(syms)) + "|"
            L.append(header)
            L.append(sep)
            for r in results_list:
                lev = r["leverage"]
                vals = []
                for sym in syms:
                    res = r["per_asset"].get(sym)
                    vals.append(f"{res.ret_pct:+.1f}%" if res else "N/A")
                L.append(f"| **{lev:.1f}x** | " + " | ".join(vals) + " |")
            L.append("")

            # Optimal leverage recommendation
            L.append(f"**{strat_name} 最优杠杆推荐: `{best_calmar_lev:.1f}x`** (Calmar 最优)\n")

    # Comprehensive leverage heatmap table
    L.append(f"## {_sec(sec)}、杠杆率综合对比矩阵\n")
    sec += 1
    L.append("### 加密货币 — Sharpe Ratio 矩阵\n")
    if crypto_lev_results:
        strats = list(crypto_lev_results.keys())
        header = "| 策略 | " + " | ".join(f"{l:.1f}x" for l in LEVERAGE_LEVELS) + " | 最优杠杆 |"
        sep = "|:-----|" + "|".join(["------:"] * len(LEVERAGE_LEVELS)) + "|:---------|"
        L.append(header)
        L.append(sep)
        for sn in strats:
            rlist = crypto_lev_results[sn]
            vals = []
            best_sr = -999
            best_lev = 1.0
            for r in rlist:
                sr = r["performance"].get("sharpe_ratio", 0)
                vals.append(f"{sr:.3f}")
                if sr > best_sr:
                    best_sr = sr
                    best_lev = r["leverage"]
            L.append(f"| **{sn}** | " + " | ".join(vals) + f" | **{best_lev:.1f}x** |")
        L.append("")

    L.append("### 美股 — Sharpe Ratio 矩阵\n")
    if stock_lev_results:
        strats = list(stock_lev_results.keys())
        header = "| 策略 | " + " | ".join(f"{l:.1f}x" for l in LEVERAGE_LEVELS) + " | 最优杠杆 |"
        sep = "|:-----|" + "|".join(["------:"] * len(LEVERAGE_LEVELS)) + "|:---------|"
        L.append(header)
        L.append(sep)
        for sn in strats:
            rlist = stock_lev_results[sn]
            vals = []
            best_sr = -999
            best_lev = 1.0
            for r in rlist:
                sr = r["performance"].get("sharpe_ratio", 0)
                vals.append(f"{sr:.3f}")
                if sr > best_sr:
                    best_sr = sr
                    best_lev = r["leverage"]
            L.append(f"| **{sn}** | " + " | ".join(vals) + f" | **{best_lev:.1f}x** |")
        L.append("")

    # Drawdown matrix
    L.append("### 最大回撤矩阵\n")
    for label, lev_res in [("加密货币", crypto_lev_results), ("美股", stock_lev_results)]:
        if not lev_res:
            continue
        L.append(f"#### {label}\n")
        strats = list(lev_res.keys())
        header = "| 策略 | " + " | ".join(f"{l:.1f}x" for l in LEVERAGE_LEVELS) + " |"
        sep = "|:-----|" + "|".join(["------:"] * len(LEVERAGE_LEVELS)) + "|"
        L.append(header)
        L.append(sep)
        for sn in strats:
            rlist = lev_res[sn]
            vals = [f"{abs(r['performance'].get('max_drawdown', 0))*100:.1f}%" for r in rlist]
            L.append(f"| **{sn}** | " + " | ".join(vals) + " |")
        L.append("")

    # Risk analysis
    L.append(f"## {_sec(sec)}、杠杆风险分析\n")
    sec += 1
    L.append("### 杠杆效应规律总结\n")
    L.append("1. **收益线性放大**: 杠杆 Nx 大致将年化收益放大 N 倍（理想情况）")
    L.append("2. **波动率线性放大**: 波动率也随杠杆线性增长")
    L.append("3. **回撤非线性放大**: 最大回撤的增长速度快于杠杆倍数（凸性效应）")
    L.append("4. **Sharpe 近似不变**: 理论上杠杆不改变 Sharpe（但实际中因成本和非线性效应会下降）")
    L.append("5. **Calmar 递减**: 由于回撤增速 > 收益增速，Calmar 通常随杠杆递减")
    L.append("6. **Kelly 准则**: Half-Kelly 给出理论最优杠杆，超过此值长期回报反而下降\n")

    # Warnings for high leverage
    L.append("### 高杠杆警告\n")
    for label, lev_res in [("加密货币", crypto_lev_results), ("美股", stock_lev_results)]:
        if not lev_res:
            continue
        for sn, rlist in lev_res.items():
            for r in rlist:
                mdd = abs(r["performance"].get("max_drawdown", 0))
                lev = r["leverage"]
                if mdd > 0.50:
                    L.append(f"- ⚠️ **{sn} @ {lev:.1f}x ({label})**: 最大回撤 {mdd*100:.1f}% "
                             f"— 极高风险，实盘中可能触发爆仓")
    L.append("")

    # Investment advice
    L.append(f"## {_sec(sec)}、投资建议 — 杠杆配置\n")
    sec += 1

    for label, lev_res in [("加密货币", crypto_lev_results), ("美股", stock_lev_results)]:
        if not lev_res:
            continue
        L.append(f"### {label}\n")

        for sn, rlist in lev_res.items():
            if not rlist:
                continue
            best_r = max(rlist, key=lambda r: r["performance"].get("calmar_ratio", -999))
            p = best_r["performance"]
            kelly = compute_kelly(best_r["daily_returns"])
            conservative_lev = min(best_r["leverage"], max(1.0, kelly * 0.5))

            L.append(f"**{sn}:**")
            L.append(f"- Calmar 最优杠杆: **{best_r['leverage']:.1f}x** "
                     f"(Calmar={p.get('calmar_ratio', 0):.3f}, "
                     f"年化={p.get('annual_return', 0)*100:+.1f}%, "
                     f"回撤={abs(p.get('max_drawdown', 0))*100:.1f}%)")
            L.append(f"- Half-Kelly 理论杠杆: **{kelly:.2f}x**")

            mdd = abs(p.get("max_drawdown", 0))
            if mdd > 0.40:
                L.append(f"- 建议: 回撤过高，**降至 {min(best_r['leverage'], 1.5):.1f}x** 或更低")
            elif mdd > 0.25:
                L.append(f"- 建议: 回撤中等，**保守用 {min(best_r['leverage'], 2.0):.1f}x**")
            else:
                L.append(f"- 建议: 回撤可控，可使用 **{best_r['leverage']:.1f}x** 杠杆")
            L.append("")

    # General advice
    L.append("### 综合杠杆使用建议\n")
    L.append("1. **新手**: 建议 **1x**（无杠杆），先验证策略有效性")
    L.append("2. **进阶**: **1.5-2x**，在 Sharpe > 1.0 且回撤 < 25% 的策略上适度加杠杆")
    L.append("3. **激进**: **3x** 以上仅适合对策略极有信心且有严格风控的专业交易者")
    L.append("4. **永远不要**: 在 Sharpe < 0.5 的策略上使用杠杆")
    L.append("5. **止损必须**: 使用杠杆时**必须设置止损**（建议 5-10%），否则回撤会迅速致命")
    L.append("6. **资金管理**: 单策略杠杆仓位不超过总资产的 30%\n")

    L.append("### 重要免责声明\n")
    L.append("> 杠杆放大收益的同时也放大亏损。历史回测中的杠杆表现不代表未来实际交易结果。"
             "高杠杆交易可能导致本金快速归零。请在充分理解杠杆风险后谨慎使用。\n")

    # Appendix
    L.append(f"## {_sec(sec)}、技术附录\n")
    L.append(f"- 回测引擎: Numba JIT kernel ({len(KERNEL_NAMES)} strategies)")
    L.append(f"- 优化方法: Walk-Forward + CPCV")
    L.append(f"- 杠杆探索: {', '.join(f'{l:.1f}x' for l in LEVERAGE_LEVELS)}")
    L.append(f"- 无风险利率: {RISK_FREE_RATE*100:.1f}%")
    L.append(f"- 初始资金: ${INITIAL_CAPITAL:,.0f}")
    L.append(f"- 总策略数: {len(KERNEL_NAMES)}")
    L.append(f"- Top-N 策略杠杆探索: {TOP_N}")
    total_combos = sum(len(v) for v in DEFAULT_PARAM_GRIDS.values())
    L.append(f"- 参数网格总组合: {total_combos:,}")
    L.append("")

    return "\n".join(L)


def main():
    t_start = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  全策略回测 + 杠杆率探索分析")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading market data ...")
    crypto_data, stock_data = load_all_data()
    print(f"  Crypto: {list(crypto_data.keys())} ({sum(len(d) for d in crypto_data.values()):,} bars)")
    print(f"  Stocks: {list(stock_data.keys())} ({sum(len(d) for d in stock_data.values()):,} bars)")
    if not crypto_data and not stock_data:
        print("ERROR: No data. Place CSV files in data/.")
        sys.exit(1)

    # Step 2: Optimize at 1x baseline
    print("\n[2/5] Walk-Forward + CPCV optimization (1x baseline) ...")
    crypto_config = BacktestConfig.crypto(leverage=1.0)
    stock_config = BacktestConfig.stock_ibkr(leverage=1.0)
    crypto_wf, crypto_cpcv = run_optimization(crypto_data, crypto_config, "Crypto")
    stock_wf, stock_cpcv = run_optimization(stock_data, stock_config, "US Stocks")

    # Step 3: Leverage exploration
    print(f"\n[3/5] Leverage exploration ({', '.join(f'{l}x' for l in LEVERAGE_LEVELS)}) ...")
    crypto_lev = {}
    stock_lev = {}

    if crypto_wf:
        top_crypto = get_top_strategies(crypto_wf, TOP_N)
        print(f"  Crypto top-{TOP_N}: {[s[0] for s in top_crypto]}")
        crypto_lev = run_leverage_exploration(top_crypto, crypto_data, BacktestConfig.crypto, "Crypto")

    if stock_wf:
        top_stock = get_top_strategies(stock_wf, TOP_N)
        print(f"  Stock top-{TOP_N}: {[s[0] for s in top_stock]}")
        stock_lev = run_leverage_exploration(top_stock, stock_data, BacktestConfig.stock_ibkr, "Stocks")

    # Step 4: Charts
    print("\n[4/5] Generating charts ...")
    if crypto_lev:
        plot_leverage_comparison(crypto_lev, "Leverage Analysis — Crypto", CHART_DIR / "crypto_leverage.png")
        for sn, rlist in crypto_lev.items():
            if rlist:
                plot_equity_by_leverage(rlist, sn, CHART_DIR / f"crypto_{sn.lower()}_equity.png")

    if stock_lev:
        plot_leverage_comparison(stock_lev, "Leverage Analysis — US Stocks", CHART_DIR / "stock_leverage.png")
        for sn, rlist in stock_lev.items():
            if rlist:
                plot_equity_by_leverage(rlist, sn, CHART_DIR / f"stock_{sn.lower()}_equity.png")

    # Step 5: Report
    total_time = time.perf_counter() - t_start
    print(f"\n[5/5] Generating report ...")

    report = generate_report(
        crypto_data, stock_data,
        crypto_wf, crypto_cpcv, stock_wf, stock_cpcv,
        crypto_lev, stock_lev,
        crypto_config, stock_config,
        total_time,
    )
    report_path = REPORT_DIR / "leverage_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("  LEVERAGE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Report: {report_path}")
    print(f"  Charts: {CHART_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
