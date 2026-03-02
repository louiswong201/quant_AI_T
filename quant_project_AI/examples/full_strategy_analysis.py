#!/usr/bin/env python3
"""
Full Strategy Analysis — Production-Grade Multi-Asset Backtest Pipeline.

Goals:
  1. Find optimal parameters across all 18 strategies via Walk-Forward + CPCV
  2. Deep-dive top strategies with PerformanceAnalyzer + equity/drawdown charts
  3. Realistic costs matching live trading (crypto / stock presets)
  4. Generate comprehensive analysis report with investment recommendations

Output:
  reports/full_analysis_report.md   — detailed markdown report
  reports/charts/*.png              — equity curves, drawdowns, heatmaps, ranking
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
import matplotlib.dates as mdates
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
    backtest_portfolio,
    optimize,
    KERNEL_NAMES,
    DEFAULT_PARAM_GRIDS,
)
from quant_framework.analysis.performance import PerformanceAnalyzer

# =====================================================================
#  Configuration
# =====================================================================

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
CHART_DIR = REPORT_DIR / "charts"
INITIAL_CAPITAL = 100_000.0
RISK_FREE_RATE = 0.04

CRYPTO_ASSETS = ["BTC", "ETH", "SOL", "BNB"]
STOCK_ASSETS = ["SPY", "QQQ", "NVDA", "META", "MSFT", "AMZN"]
MIN_BARS_REQUIRED = 500

plt.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})


# =====================================================================
#  Data loading
# =====================================================================

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


# =====================================================================
#  Optimization phase
# =====================================================================

def run_optimization(
    data_map: Dict[str, pd.DataFrame],
    config: BacktestConfig,
    label: str,
) -> Tuple[Optional[OptimizeResult], Optional[OptimizeResult]]:
    """Run both WF and CPCV optimization on a group of assets."""
    if not data_map:
        return None, None

    multi_data = {sym: df for sym, df in data_map.items()}
    print(f"\n{'='*60}")
    print(f"  Optimizing [{label}]: {list(multi_data.keys())}")
    print(f"{'='*60}")

    print(f"  Running Walk-Forward optimization ...")
    t0 = time.perf_counter()
    wf_result = optimize(multi_data, config, method="wf")
    wf_time = time.perf_counter() - t0
    print(f"  WF done: {wf_result.total_combos:,} combos in {wf_time:.1f}s "
          f"({wf_result.total_combos / max(0.1, wf_time):,.0f}/s)")
    print(f"  Best: {wf_result.best}")

    print(f"  Running CPCV validation ...")
    t0 = time.perf_counter()
    cpcv_result = optimize(multi_data, config, method="cpcv")
    cpcv_time = time.perf_counter() - t0
    print(f"  CPCV done: {cpcv_result.total_combos:,} combos in {cpcv_time:.1f}s")
    print(f"  Best: {cpcv_result.best}")

    return wf_result, cpcv_result


# =====================================================================
#  Detailed backtest for top strategies
# =====================================================================

def run_detailed_backtests(
    opt_result: OptimizeResult,
    data_map: Dict[str, pd.DataFrame],
    config: BacktestConfig,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Run detailed backtest for top-N strategies across all assets."""
    ranked = sorted(
        opt_result.all_strategies.items(),
        key=lambda kv: kv[1].get("wf_score", kv[1].get("cpcv_score", -1e18)),
        reverse=True,
    )[:top_n]

    results = []
    for strat_name, meta in ranked:
        params = meta.get("params")
        if params is None:
            continue

        per_asset = {}
        combined_equity = []
        for sym, df in data_map.items():
            try:
                res = backtest(strat_name, params, df, config, detailed=True)
                per_asset[sym] = res
                if res.equity is not None:
                    combined_equity.append(res.equity)
            except Exception as e:
                print(f"    Warning: {strat_name} on {sym} failed: {e}")

        if not combined_equity:
            continue

        min_len = min(len(eq) for eq in combined_equity)
        avg_equity = np.mean(
            [eq[:min_len] for eq in combined_equity], axis=0
        )
        portfolio_values = avg_equity * INITIAL_CAPITAL

        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        daily_returns = daily_returns[np.isfinite(daily_returns)]

        analyzer = PerformanceAnalyzer(
            risk_free_rate=RISK_FREE_RATE, periods_per_year=252.0
        )
        perf = analyzer.analyze(portfolio_values, daily_returns, INITIAL_CAPITAL,
                                n_trials=len(KERNEL_NAMES))

        results.append({
            "strategy": strat_name,
            "params": params,
            "meta": meta,
            "per_asset": per_asset,
            "portfolio_values": portfolio_values,
            "daily_returns": daily_returns,
            "performance": perf,
        })

    return results


# =====================================================================
#  Chart generation
# =====================================================================

def plot_equity_curves(
    detailed_results: List[Dict[str, Any]],
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    for r in detailed_results:
        pv = r["portfolio_values"]
        label = f'{r["strategy"]} ({r["performance"].get("sharpe_ratio", 0):.2f})'
        axes[0].plot(pv, label=label, linewidth=1.5)

    axes[0].axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    axes[0].set_title(title, fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    for r in detailed_results:
        pv = r["portfolio_values"]
        peak = np.maximum.accumulate(pv)
        dd = (pv - peak) / peak * 100
        axes[1].plot(dd, label=r["strategy"], linewidth=1.0, alpha=0.7)

    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Trading Days")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Chart saved: {save_path.name}")


def plot_monthly_returns(
    portfolio_values: np.ndarray,
    dates: Optional[pd.DatetimeIndex],
    strategy_name: str,
    save_path: Path,
) -> None:
    if dates is None or len(dates) < 60:
        return

    n = min(len(portfolio_values), len(dates))
    pv = portfolio_values[:n]
    dt = dates[:n]

    df = pd.DataFrame({"date": dt, "value": pv})
    df.set_index("date", inplace=True)
    monthly = df["value"].resample("ME").last()
    monthly_ret = monthly.pct_change().dropna()

    if monthly_ret.empty:
        return

    pivot = pd.DataFrame({
        "year": monthly_ret.index.year,
        "month": monthly_ret.index.month,
        "return": monthly_ret.values * 100,
    })
    table = pivot.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
    table.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(table.columns)]

    fig, ax = plt.subplots(figsize=(14, max(4, len(table) * 0.6 + 1)))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(table.values, cmap=cmap, aspect="auto", vmin=-15, vmax=15)

    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, fontsize=9)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=9)

    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            val = table.values[i, j]
            if np.isfinite(val):
                color = "white" if abs(val) > 8 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_title(f"Monthly Returns (%) — {strategy_name}", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Return %", shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Chart saved: {save_path.name}")


def plot_strategy_ranking(
    opt_result: OptimizeResult,
    title: str,
    save_path: Path,
) -> None:
    ranked = sorted(
        opt_result.all_strategies.items(),
        key=lambda kv: kv[1].get("wf_score", kv[1].get("cpcv_score", -1e18)),
        reverse=True,
    )

    names = [sn for sn, _ in ranked]
    scores = [d.get("wf_score", d.get("cpcv_score", 0)) for _, d in ranked]
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in scores]

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.35)))
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + (max(scores) - min(scores)) * 0.01 * (1 if score >= 0 else -1),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            va="center", fontsize=7,
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Chart saved: {save_path.name}")


def plot_rolling_sharpe(
    detailed_results: List[Dict[str, Any]],
    window: int = 60,
    save_path: Path = CHART_DIR / "rolling_sharpe.png",
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for r in detailed_results[:3]:
        dr = r["daily_returns"]
        if len(dr) < window + 10:
            continue
        rolling_mean = pd.Series(dr).rolling(window).mean()
        rolling_std = pd.Series(dr).rolling(window).std()
        rs = (rolling_mean / rolling_std * np.sqrt(252)).dropna()
        ax.plot(rs.values, label=r["strategy"], linewidth=1.2)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(1.0, color="green", linestyle=":", alpha=0.5, linewidth=0.5)
    ax.axhline(-1.0, color="red", linestyle=":", alpha=0.5, linewidth=0.5)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Trading Days")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Chart saved: {save_path.name}")


# =====================================================================
#  Report generation
# =====================================================================

def _fmt_pct(v: float, decimals: int = 2) -> str:
    return f"{v:+.{decimals}f}%"


def _fmt_num(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


def generate_report(
    crypto_data: Dict[str, pd.DataFrame],
    stock_data: Dict[str, pd.DataFrame],
    crypto_wf: Optional[OptimizeResult],
    crypto_cpcv: Optional[OptimizeResult],
    stock_wf: Optional[OptimizeResult],
    stock_cpcv: Optional[OptimizeResult],
    crypto_detailed: List[Dict[str, Any]],
    stock_detailed: List[Dict[str, Any]],
    crypto_config: BacktestConfig,
    stock_config: BacktestConfig,
    total_time: float,
) -> str:
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Header ──────────────────────────────────────────────────
    lines.append("# 全策略量化回测分析报告")
    lines.append(f"\n> 生成时间: {now} | 总耗时: {total_time:.1f}s\n")

    # ── Executive Summary ───────────────────────────────────────
    lines.append("## 一、执行摘要\n")
    all_detailed = crypto_detailed + stock_detailed
    if all_detailed:
        best = max(all_detailed, key=lambda r: r["performance"].get("sharpe_ratio", -99))
        bp = best["performance"]
        lines.append(
            f"本报告基于 **{len(crypto_data) + len(stock_data)}** 个资产的真实市场数据，"
            f"对全部 **{len(KERNEL_NAMES)}** 种策略进行参数优化与回测。\n"
        )
        lines.append(
            f"经 Walk-Forward + CPCV 双重验证，**最优策略为 `{best['strategy']}`**，"
            f"参数 `{best['params']}`：\n"
        )
        lines.append(f"- 年化收益率: **{bp.get('annual_return', 0)*100:+.2f}%**")
        lines.append(f"- Sharpe Ratio: **{bp.get('sharpe_ratio', 0):.2f}**")
        lines.append(f"- 最大回撤: **{bp.get('max_drawdown', 0)*100:.2f}%**")
        lines.append(f"- Deflated Sharpe p-value: **{bp.get('deflated_sharpe_pvalue', 1):.4f}**")
        lines.append(f"- 胜率: **{bp.get('win_rate', 0)*100:.1f}%**")
        lines.append(f"- 初始资金 ${INITIAL_CAPITAL:,.0f} → 终值 ${bp.get('final_value', 0):,.0f}\n")

    # ── Data Overview ───────────────────────────────────────────
    lines.append("## 二、数据概览\n")
    lines.append("| 资产 | 类型 | 数据天数 | 起始日期 | 结束日期 | 最新价格 |")
    lines.append("|:-----|:-----|--------:|:---------|:---------|--------:|")
    for sym, df in {**crypto_data, **stock_data}.items():
        mtype = "加密货币" if sym in crypto_data else "美股"
        start = df["date"].iloc[0].strftime("%Y-%m-%d") if "date" in df.columns else "N/A"
        end = df["date"].iloc[-1].strftime("%Y-%m-%d") if "date" in df.columns else "N/A"
        price = df["close"].iloc[-1]
        lines.append(f"| {sym} | {mtype} | {len(df):,} | {start} | {end} | ${price:,.2f} |")
    lines.append("")

    # ── Cost Model ──────────────────────────────────────────────
    lines.append("## 三、成本模型（贴近真实交易）\n")
    lines.append("| 参数 | 加密货币 | 美股 |")
    lines.append("|:-----|--------:|-----:|")
    lines.append(f"| 手续费 | {crypto_config.commission_pct_buy*100:.2f}% | {stock_config.commission_pct_buy*100:.2f}% |")
    lines.append(f"| 滑点 (bps) | {crypto_config.slippage_bps_buy:.0f} | {stock_config.slippage_bps_buy:.0f} |")
    lines.append(f"| 资金费率/日 | {crypto_config.daily_funding_rate*100:.2f}% | 0% |")
    lines.append(f"| 做空借贷率/年 | {crypto_config.short_borrow_rate_annual*100:.1f}% | {stock_config.short_borrow_rate_annual*100:.1f}% |")
    lines.append(f"| 杠杆 | {crypto_config.leverage:.0f}x | {stock_config.leverage:.0f}x |")
    lines.append(f"| 允许做空 | {'是' if crypto_config.allow_short else '否'} | {'是' if stock_config.allow_short else '否'} |")
    lines.append("")

    # ── Optimization Results ────────────────────────────────────
    section_num = 4
    for label, wf, cpcv, data_map in [
        ("加密货币", crypto_wf, crypto_cpcv, crypto_data),
        ("美股", stock_wf, stock_cpcv, stock_data),
    ]:
        if wf is None:
            continue

        lines.append(f"## {_section(section_num)}、Walk-Forward 优化结果 — {label}\n")
        section_num += 1
        lines.append(f"> 资产: {', '.join(data_map.keys())} | "
                     f"总组合数: {wf.total_combos:,} | "
                     f"耗时: {wf.elapsed_seconds:.1f}s | "
                     f"速度: {wf.total_combos / max(0.1, wf.elapsed_seconds):,.0f} combos/s\n")

        lines.append("| 排名 | 策略 | 参数 | OOS收益 | 最大回撤 | Sharpe | DSR p值 | MC正比 | WF得分 |")
        lines.append("|-----:|:-----|:-----|--------:|--------:|-------:|--------:|-------:|-------:|")
        ranked = sorted(
            wf.all_strategies.items(),
            key=lambda kv: kv[1].get("wf_score", -1e18),
            reverse=True,
        )
        for i, (sn, d) in enumerate(ranked, 1):
            wf_s = d.get("wf_score", 0)
            params_str = str(d.get("params", ""))
            if len(params_str) > 25:
                params_str = params_str[:22] + "..."
            lines.append(
                f"| {i} | **{sn}** | `{params_str}` | "
                f"{d.get('oos_ret', 0):+.1f}% | "
                f"{d.get('oos_dd', 0):.1f}% | "
                f"{d.get('sharpe', 0):.2f} | "
                f"{d.get('dsr_p', 0):.3f} | "
                f"{d.get('mc_pct_positive', 0)*100:.0f}% | "
                f"{wf_s:.1f} |"
            )
        lines.append("")

        if cpcv is not None:
            lines.append(f"### CPCV 交叉验证 — {label}\n")
            lines.append("| 排名 | 策略 | OOS均值 | OOS标准差 | 正分割比 | CPCV得分 |")
            lines.append("|-----:|:-----|--------:|----------:|---------:|---------:|")
            ranked_c = sorted(
                cpcv.all_strategies.items(),
                key=lambda kv: kv[1].get("cpcv_score", kv[1].get("wf_score", -1e18)),
                reverse=True,
            )
            for i, (sn, d) in enumerate(ranked_c[:10], 1):
                lines.append(
                    f"| {i} | **{sn}** | "
                    f"{d.get('oos_ret', d.get('oos_ret_mean', 0)):+.1f}% | "
                    f"{d.get('oos_ret_std', d.get('oos_dd', 0)):.1f}% | "
                    f"{d.get('pct_splits_positive', d.get('mc_pct_positive', 0))*100:.0f}% | "
                    f"{d.get('cpcv_score', d.get('wf_score', 0)):.1f} |"
                )
            lines.append("")

    # ── Top Strategy Deep Dive ──────────────────────────────────
    for label, detailed_list, data_map in [
        ("加密货币", crypto_detailed, crypto_data),
        ("美股", stock_detailed, stock_data),
    ]:
        if not detailed_list:
            continue

        lines.append(f"## {_section(section_num)}、Top 策略深度分析 — {label}\n")
        section_num += 1

        for rank, r in enumerate(detailed_list[:3], 1):
            perf = r["performance"]
            lines.append(f"### {rank}. {r['strategy']} `{r['params']}`\n")

            lines.append("| 指标 | 值 |")
            lines.append("|:-----|---:|")
            metrics = [
                ("初始资金", f"${perf.get('initial_capital', INITIAL_CAPITAL):,.0f}"),
                ("终值", f"${perf.get('final_value', 0):,.0f}"),
                ("总收益率", f"{perf.get('total_return', 0)*100:+.2f}%"),
                ("年化收益率", f"{perf.get('annual_return', 0)*100:+.2f}%"),
                ("年化波动率", f"{perf.get('volatility', 0)*100:.2f}%"),
                ("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.3f}"),
                ("Sortino Ratio", f"{perf.get('sortino_ratio', 0):.3f}"),
                ("Calmar Ratio", f"{perf.get('calmar_ratio', 0):.3f}"),
                ("Omega Ratio", f"{perf.get('omega_ratio', 0):.3f}"),
                ("最大回撤", f"{perf.get('max_drawdown', 0)*100:.2f}%"),
                ("最大回撤持续(天)", f"{perf.get('max_drawdown_duration', 0)}"),
                ("VaR (95%)", f"{perf.get('value_at_risk_95', 0)*100:.2f}%"),
                ("CVaR (95%)", f"{perf.get('cvar_95', 0)*100:.2f}%"),
                ("VaR (99%)", f"{perf.get('value_at_risk_99', 0)*100:.2f}%"),
                ("CVaR (99%)", f"{perf.get('cvar_99', 0)*100:.2f}%"),
                ("利润因子", f"{perf.get('profit_factor', 0):.3f}"),
                ("胜率", f"{perf.get('win_rate', 0)*100:.1f}%"),
                ("尾部比率", f"{perf.get('tail_ratio', 0):.3f}"),
                ("偏度", f"{perf.get('skewness', 0):.3f}"),
                ("超额峰度", f"{perf.get('kurtosis_excess', 0):.3f}"),
                ("Deflated Sharpe p值", f"{perf.get('deflated_sharpe_pvalue', 1):.4f}"),
                ("交易天数", f"{perf.get('trading_days', 0):,}"),
            ]
            for name, val in metrics:
                lines.append(f"| {name} | {val} |")

            lines.append("")

            per_asset_lines = ["| 资产 | 收益 | 回撤 | 交易次数 | Sharpe |"]
            per_asset_lines.append("|:-----|-----:|-----:|---------:|-------:|")
            for sym, res in r["per_asset"].items():
                per_asset_lines.append(
                    f"| {sym} | {res.ret_pct:+.2f}% | {res.max_dd_pct:.2f}% | "
                    f"{res.n_trades} | {res.sharpe:.2f} |"
                )
            lines.extend(per_asset_lines)
            lines.append("")

    # ── Risk Assessment ─────────────────────────────────────────
    lines.append(f"## {_section(section_num)}、风险评估\n")
    section_num += 1

    if all_detailed:
        best = max(all_detailed, key=lambda r: r["performance"].get("sharpe_ratio", -99))
        bp = best["performance"]
        dsr_p = bp.get("deflated_sharpe_pvalue", 1.0)
        mdd = bp.get("max_drawdown", 0)

        lines.append("### 过拟合检测\n")
        if dsr_p < 0.05:
            lines.append(f"- Deflated Sharpe p = {dsr_p:.4f} → **统计显著**，"
                         "在多重检验校正后仍具统计意义，策略具有真实 alpha 的可能性高。")
        elif dsr_p < 0.50:
            lines.append(f"- Deflated Sharpe p = {dsr_p:.4f} → **中等可信度**，"
                         "建议增加样本外验证期后再投入实盘。")
        else:
            lines.append(f"- Deflated Sharpe p = {dsr_p:.4f} → **未通过多重检验**，"
                         "策略表现可能受益于参数搜索偏差，实盘需额外谨慎。"
                         "建议缩小参数范围后重新验证，或仅使用 WF/CPCV 的 OOS 指标作为参考。")

        meta = best.get("meta", {})
        mc_pos = meta.get("mc_pct_positive", 0)
        lines.append(f"- Monte Carlo 正收益比: {mc_pos*100:.0f}% "
                     f"{'(稳健)' if mc_pos > 0.6 else '(需注意)'}")
        lines.append(f"- 最大回撤: {mdd*100:.2f}% — "
                     f"{'可控' if abs(mdd) < 0.25 else '偏高，建议降低仓位或增加止损'}")

        lines.append("\n### 尾部风险\n")
        lines.append(f"- 95% VaR: 单日最大可能亏损约 {abs(bp.get('value_at_risk_95', 0))*100:.2f}%")
        lines.append(f"- 99% VaR: 极端情况单日亏损约 {abs(bp.get('value_at_risk_99', 0))*100:.2f}%")
        lines.append(f"- CVaR (95%): 超过 VaR 时平均损失 {abs(bp.get('cvar_95', 0))*100:.2f}%")

        kurt = bp.get("kurtosis_excess", 0)
        if kurt > 3:
            lines.append(f"- 超额峰度 = {kurt:.2f} → **厚尾分布**，极端事件概率高于正态假设")
        lines.append("")

    # ── Investment Advice ───────────────────────────────────────
    lines.append(f"## {_section(section_num)}、投资建议\n")
    section_num += 1

    for label, det_list in [("加密货币", crypto_detailed), ("美股", stock_detailed)]:
        if not det_list:
            continue

        best = max(det_list, key=lambda r: r["performance"].get("sharpe_ratio", -99))
        bp = best["performance"]
        sr = bp.get("sharpe_ratio", 0)
        mdd = abs(bp.get("max_drawdown", 0))
        ann_ret = bp.get("annual_return", 0)
        calmar = bp.get("calmar_ratio", 0)

        lines.append(f"### {label}策略配置建议\n")
        lines.append(f"1. **首选策略**: `{best['strategy']}` 参数 `{best['params']}`")
        lines.append(f"   - 年化 {ann_ret*100:+.1f}% | Sharpe {sr:.2f} | "
                     f"Calmar {calmar:.2f} | 最大回撤 {mdd*100:.1f}%")

        if len(det_list) >= 2:
            second = sorted(det_list,
                            key=lambda r: r["performance"].get("sharpe_ratio", -99),
                            reverse=True)[1]
            sp2 = second["performance"]
            lines.append(f"2. **备选策略**: `{second['strategy']}` 参数 `{second['params']}`")
            lines.append(f"   - 年化 {sp2.get('annual_return', 0)*100:+.1f}% | "
                         f"Sharpe {sp2.get('sharpe_ratio', 0):.2f} | "
                         f"最大回撤 {abs(sp2.get('max_drawdown', 0))*100:.1f}%")

        per_asset_summary = []
        for sym, res in best.get("per_asset", {}).items():
            per_asset_summary.append(f"{sym}({res.ret_pct:+.0f}%)")
        if per_asset_summary:
            lines.append(f"   - 各资产表现: {', '.join(per_asset_summary)}")

        lines.append(f"\n**{label}仓位建议:**\n")
        if mdd > 0.30:
            lines.append("- 回撤偏高，建议**仓位不超过总资金的 50%**，设置 **5-8% 止损**保护")
        elif mdd > 0.15:
            lines.append("- 回撤适中，建议**仓位控制在总资金 60-80%**，设置 **8-12% 止损**")
        else:
            lines.append("- 回撤可控，可**配置 80-100% 仓位**，建议保留 10-20% 现金缓冲")

        if sr > 1.0:
            lines.append(f"- Sharpe > 1.0 表示每单位风险获得超额回报，**风险调整后回报优秀**")
        elif sr > 0.5:
            lines.append(f"- Sharpe {sr:.2f} 属于可接受水平，但需密切关注回撤")
        else:
            lines.append(f"- Sharpe {sr:.2f} 偏低，**建议仅作为组合的一部分**，不宜单独重仓")
        lines.append("")

    lines.append("### 综合风险控制建议\n")
    lines.append("1. **分散化**: 不要仅依赖单一策略/市场，建议同时运行 2-3 种低相关策略")
    lines.append("2. **定期复审**: 每季度重新运行本分析，检查参数是否仍然有效")
    lines.append("3. **渐进投入**: 先用小资金实盘验证 1-2 个月，确认无偏差后再加仓")
    lines.append("4. **极端行情保护**: 在已知重大事件（利率决议、ETF 审批等）前降低敞口")
    lines.append("5. **相关性管理**: 加密货币与美股之间存在一定相关性，需控制总体风险敞口")
    lines.append("6. **止损纪律**: 严格执行预设止损，避免情绪化持仓")

    lines.append("\n### 重要免责声明\n")
    lines.append("> 历史回测结果不代表未来表现。所有收益率和风险指标均基于历史数据，"
                 "实际交易可能因市场条件变化、流动性不足、黑天鹅事件等原因产生显著偏差。"
                 "建议在充分理解风险后，以自身可承受亏损的资金进行投资。\n")

    # ── Appendix ────────────────────────────────────────────────
    lines.append(f"## {_section(section_num)}、技术附录\n")
    lines.append(f"- 回测引擎: Numba JIT kernel ({len(KERNEL_NAMES)} strategies)")
    lines.append(f"- 优化方法: Walk-Forward (6 windows, purged) + CPCV")
    lines.append(f"- 反过拟合层: 11 层 (Monte Carlo, DSR, Block Bootstrap, ...)")
    lines.append(f"- 无风险利率: {RISK_FREE_RATE*100:.1f}%")
    lines.append(f"- 初始资金: ${INITIAL_CAPITAL:,.0f}")
    lines.append(f"- 总策略数: {len(KERNEL_NAMES)}")
    total_combos = sum(len(v) for v in DEFAULT_PARAM_GRIDS.values())
    lines.append(f"- 默认参数网格总组合: {total_combos:,}")
    lines.append("")

    return "\n".join(lines)


def _section(n: int) -> str:
    return ["零","一","二","三","四","五","六","七","八","九","十"][min(n, 10)]


# =====================================================================
#  Main
# =====================================================================

def main():
    t_start = time.perf_counter()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  全策略量化回测分析 — Production Pipeline")
    print("=" * 60)

    # ── Step 1: Load data ───────────────────────────────────────
    print("\n[1/6] Loading market data ...")
    crypto_data, stock_data = load_all_data()
    print(f"  Crypto: {list(crypto_data.keys())} "
          f"({sum(len(d) for d in crypto_data.values()):,} bars)")
    print(f"  Stocks: {list(stock_data.keys())} "
          f"({sum(len(d) for d in stock_data.values()):,} bars)")

    if not crypto_data and not stock_data:
        print("ERROR: No data found. Place CSV files in data/ directory.")
        sys.exit(1)

    # ── Step 2: Configure cost models ───────────────────────────
    print("\n[2/6] Configuring realistic cost models ...")
    crypto_config = BacktestConfig.crypto(leverage=1.0)
    stock_config = BacktestConfig.stock_ibkr(leverage=1.0)
    print(f"  Crypto: commission={crypto_config.commission_pct_buy*100:.2f}%, "
          f"slippage={crypto_config.slippage_bps_buy:.0f}bps, "
          f"funding={crypto_config.daily_funding_rate*100:.2f}%/day")
    print(f"  Stock:  commission={stock_config.commission_pct_buy*100:.2f}%, "
          f"slippage={stock_config.slippage_bps_buy:.0f}bps, "
          f"borrow={stock_config.short_borrow_rate_annual*100:.1f}%/yr")

    # ── Step 3: Optimization ────────────────────────────────────
    print("\n[3/6] Running parameter optimization (WF + CPCV) ...")
    crypto_wf, crypto_cpcv = run_optimization(crypto_data, crypto_config, "Crypto")
    stock_wf, stock_cpcv = run_optimization(stock_data, stock_config, "US Stocks")

    # ── Step 4: Detailed backtests ──────────────────────────────
    print("\n[4/6] Running detailed backtests for top strategies ...")
    crypto_detailed = []
    stock_detailed = []
    if crypto_wf:
        crypto_detailed = run_detailed_backtests(crypto_wf, crypto_data, crypto_config, top_n=5)
        print(f"  Crypto top-5 detailed backtests complete")
    if stock_wf:
        stock_detailed = run_detailed_backtests(stock_wf, stock_data, stock_config, top_n=5)
        print(f"  Stock top-5 detailed backtests complete")

    # ── Step 5: Generate charts ─────────────────────────────────
    print("\n[5/6] Generating charts ...")
    if crypto_detailed:
        plot_equity_curves(
            crypto_detailed,
            "Top Strategy Equity Curves — Crypto",
            CHART_DIR / "crypto_equity.png",
        )
        plot_strategy_ranking(crypto_wf, "Strategy Ranking (WF) — Crypto",
                              CHART_DIR / "crypto_ranking.png")
        best_crypto = crypto_detailed[0]
        first_sym = next(iter(crypto_data))
        if "date" in crypto_data[first_sym].columns:
            dates = pd.DatetimeIndex(crypto_data[first_sym]["date"])
            plot_monthly_returns(
                best_crypto["portfolio_values"], dates,
                best_crypto["strategy"], CHART_DIR / "crypto_monthly.png",
            )

    if stock_detailed:
        plot_equity_curves(
            stock_detailed,
            "Top Strategy Equity Curves — US Stocks",
            CHART_DIR / "stock_equity.png",
        )
        plot_strategy_ranking(stock_wf, "Strategy Ranking (WF) — US Stocks",
                              CHART_DIR / "stock_ranking.png")
        best_stock = stock_detailed[0]
        first_sym = next(iter(stock_data))
        if "date" in stock_data[first_sym].columns:
            dates = pd.DatetimeIndex(stock_data[first_sym]["date"])
            plot_monthly_returns(
                best_stock["portfolio_values"], dates,
                best_stock["strategy"], CHART_DIR / "stock_monthly.png",
            )

    all_detailed = crypto_detailed + stock_detailed
    if len(all_detailed) >= 2:
        plot_rolling_sharpe(all_detailed, save_path=CHART_DIR / "rolling_sharpe.png")

    # ── Step 6: Generate report ─────────────────────────────────
    total_time = time.perf_counter() - t_start
    print(f"\n[6/6] Generating analysis report ...")

    report = generate_report(
        crypto_data, stock_data,
        crypto_wf, crypto_cpcv, stock_wf, stock_cpcv,
        crypto_detailed, stock_detailed,
        crypto_config, stock_config,
        total_time,
    )
    report_path = REPORT_DIR / "full_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    # ── Console summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Report:     {report_path}")
    print(f"  Charts:     {CHART_DIR}/")

    if all_detailed:
        best = max(all_detailed, key=lambda r: r["performance"].get("sharpe_ratio", -99))
        bp = best["performance"]
        print(f"\n  BEST STRATEGY: {best['strategy']} {best['params']}")
        print(f"  Annual Return: {bp.get('annual_return', 0)*100:+.2f}%")
        print(f"  Sharpe Ratio:  {bp.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown:  {bp.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Win Rate:      {bp.get('win_rate', 0)*100:.1f}%")
        print(f"  Final Value:   ${bp.get('final_value', 0):,.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
