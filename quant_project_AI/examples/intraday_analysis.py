#!/usr/bin/env python3
"""
Intraday 5-Minute Backtest Analysis with Leverage Exploration.

Uses the multi-timeframe engine (interval-aware BacktestConfig) to:
  1. Optimize all 18 strategies on 5m data via Walk-Forward + CPCV
  2. Run top strategies at multiple leverage levels
  3. Generate full report: reports/intraday_5m_analysis_report.md

Usage:
    python examples/intraday_analysis.py
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
)
from quant_framework.analysis.performance import PerformanceAnalyzer

DATA_DIR = PROJECT_ROOT / "data" / "5m"
REPORT_DIR = PROJECT_ROOT / "reports"
CHART_DIR = REPORT_DIR / "intraday_charts"
INITIAL_CAPITAL = 100_000.0
RISK_FREE_RATE = 0.04
INTERVAL = "5m"

CRYPTO_ASSETS = ["BTC", "ETH", "SOL", "BNB"]
STOCK_ASSETS = ["SPY", "QQQ", "NVDA", "META", "MSFT", "AMZN"]
LEVERAGE_LEVELS = [1.0, 1.5, 2.0, 3.0, 5.0]
MIN_BARS_REQUIRED = 2000
TOP_N = 5

# 5-minute parameter grids (1-24 hour lookback periods)
PARAM_GRIDS_5M = {
    "MA":          [(f, s) for f in (6, 12, 24, 48) for s in (48, 96, 144, 288) if f < s],
    "RSI":         [(p, ob, os_) for p in (7, 14, 21) for ob in (25, 30) for os_ in (70, 75)],
    "MACD":        [(f, s, sig) for f in (6, 12, 24) for s in (26, 52, 96) for sig in (5, 9) if f < s],
    "Bollinger":   [(p, std) for p in (12, 20, 36, 48) for std in (1.5, 2.0, 2.5)],
    "Keltner":     [(p, atr, m) for p in (20, 40, 80) for atr in (7, 14) for m in (1.0, 1.5, 2.0)],
    "Drift":       [(p, thr, atr) for p in (6, 12, 20) for thr in (0.5, 0.65, 0.8) for atr in (7, 14)],
    "KAMA":        [(er, f, s, m, atr) for er in (10, 20) for f in (2, 3) for s in (15, 30) for m in (1.5, 2.0) for atr in (7, 14)],
    "DualMom":     [(s, l) for s in (12, 24, 48) for l in (48, 96, 192) if s < l],
    "Turtle":      [(e, x, atr, m) for e in (10, 20) for x in (5, 10) for atr in (7, 14) for m in (1.5, 2.5)],
    "Donchian":    [(p, atr, m) for p in (10, 20, 40) for atr in (7, 14) for m in (1.5, 2.0, 2.5)],
    "ZScore":      [(p, en, ex, sl) for p in (24, 48, 96) for en in (2.0, 2.5) for ex in (0.0, 0.5) for sl in (2.5, 3.0)],
    "MomBreak":    [(lb, thr, atr, m) for lb in (12, 24, 48) for thr in (0.04, 0.08) for atr in (7, 14) for m in (1.5, 2.5)],
    "MultiFactor": [(ma, mom, vol, w1, w2) for ma in (7, 14, 21) for mom in (24, 48) for vol in (12, 24) for w1 in (0.45, 0.55) for w2 in (0.25, 0.35)],
    "Consensus":   [(ma_s, ma_l, rsi, bb, rs_lo, rs_hi, thr) for ma_s in (6, 12) for ma_l in (48, 96) for rsi in (14,) for bb in (20,) for rs_lo in (30,) for rs_hi in (70,) for thr in (2, 3)],
    "MESA":        [(fl, sl) for fl in (0.3, 0.5, 0.7) for sl in (0.05, 0.1)],
    "RegimeEMA":   [(atr, thr, fast, mid, slow) for atr in (7, 14) for thr in (0.01, 0.015, 0.02) for fast in (6, 12) for mid in (24,) for slow in (48, 96)],
    "VolRegime":   [(vol_w, vol_thr, ma_s, ma_l, rs_lo, rs_hi) for vol_w in (12, 24) for vol_thr in (0.015, 0.03) for ma_s in (6, 12) for ma_l in (48,) for rs_lo in (30,) for rs_hi in (75,)],
    "RAMOM":       [(lb, vol, up, dn) for lb in (12, 24, 48) for vol in (12, 24) for up in (1.0, 2.0) for dn in (0.5, 1.0)],
}

plt.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})


def load_5m_csv(symbol: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / f"{symbol}_5m.csv"
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


def load_all_data():
    crypto, stocks = {}, {}
    for sym in CRYPTO_ASSETS:
        df = load_5m_csv(sym)
        if df is not None and len(df) >= MIN_BARS_REQUIRED:
            crypto[sym] = df
    for sym in STOCK_ASSETS:
        df = load_5m_csv(sym)
        if df is not None and len(df) >= MIN_BARS_REQUIRED:
            stocks[sym] = df
    return crypto, stocks


def run_optimization(data_map, config, label):
    if not data_map:
        return None, None
    print(f"\n{'='*60}")
    print(f"  Optimizing [{label}] @ {INTERVAL}: {list(data_map.keys())}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    wf_result = optimize(data_map, config, method="wf", param_grids=PARAM_GRIDS_5M)
    wf_time = time.perf_counter() - t0
    print(f"  WF: {wf_result.total_combos:,} combos in {wf_time:.1f}s "
          f"({wf_result.total_combos / max(0.1, wf_time):,.0f}/s)")

    t0 = time.perf_counter()
    cpcv_result = optimize(data_map, config, method="cpcv", param_grids=PARAM_GRIDS_5M)
    cpcv_time = time.perf_counter() - t0
    print(f"  CPCV: {cpcv_result.total_combos:,} combos in {cpcv_time:.1f}s")

    return wf_result, cpcv_result


def run_detailed_single(strat_name, params, data_map, config):
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
    bar_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    bar_returns = bar_returns[np.isfinite(bar_returns)]

    bpy = config.bars_per_year
    analyzer = PerformanceAnalyzer(risk_free_rate=RISK_FREE_RATE, periods_per_year=bpy)
    perf = analyzer.analyze(portfolio_values, bar_returns, INITIAL_CAPITAL,
                            n_trials=len(KERNEL_NAMES))
    return {
        "strategy": strat_name, "params": params,
        "per_asset": per_asset, "portfolio_values": portfolio_values,
        "daily_returns": bar_returns, "performance": perf,
    }


def get_top_strategies(wf_result, n=TOP_N):
    ranked = sorted(
        wf_result.all_strategies.items(),
        key=lambda kv: kv[1].get("wf_score", -1e18), reverse=True,
    )[:n]
    return [(name, meta.get("params")) for name, meta in ranked if meta.get("params") is not None]


def run_leverage_exploration(top_strategies, data_map, config_fn, label):
    results = {}
    for strat_name, params in top_strategies:
        strat_results = []
        for lev in LEVERAGE_LEVELS:
            config = config_fn(leverage=lev, interval=INTERVAL)
            print(f"    {strat_name} @ {lev:.1f}x ...")
            r = run_detailed_single(strat_name, params, data_map, config)
            if r:
                r["leverage"] = lev
                strat_results.append(r)
        results[strat_name] = strat_results
    return results


def compute_kelly(returns):
    if len(returns) < 30:
        return 0.0
    mu = np.mean(returns)
    var = np.var(returns)
    if var < 1e-15:
        return 0.0
    return max(0, min(mu / var / 2, 20.0))


def plot_leverage_comparison(lev_results, title, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, len(lev_results)))
    for idx, (strat, rlist) in enumerate(lev_results.items()):
        if not rlist:
            continue
        levs = [r["leverage"] for r in rlist]
        c = colors[idx]
        axes[0, 0].plot(levs, [r["performance"].get("annual_return", 0) * 100 for r in rlist],
                        "o-", color=c, label=strat, linewidth=2, markersize=8)
        axes[0, 1].plot(levs, [r["performance"].get("sharpe_ratio", 0) for r in rlist],
                        "s-", color=c, label=strat, linewidth=2, markersize=8)
        axes[1, 0].plot(levs, [abs(r["performance"].get("max_drawdown", 0)) * 100 for r in rlist],
                        "^-", color=c, label=strat, linewidth=2, markersize=8)
        axes[1, 1].plot(levs, [r["performance"].get("calmar_ratio", 0) for r in rlist],
                        "D-", color=c, label=strat, linewidth=2, markersize=8)

    for ax, t, y in [(axes[0, 0], "Annual Return vs Leverage", "Annual Return (%)"),
                     (axes[0, 1], "Sharpe Ratio vs Leverage", "Sharpe Ratio"),
                     (axes[1, 0], "Max Drawdown vs Leverage", "Max Drawdown (%)"),
                     (axes[1, 1], "Calmar Ratio vs Leverage", "Calmar Ratio")]:
        ax.set_title(t, fontweight="bold")
        ax.set_ylabel(y)
        ax.set_xlabel("Leverage")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {save_path.name}")


def plot_equity_by_leverage(rlist, strat_name, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(rlist)))
    for i, r in enumerate(rlist):
        pv = r["portfolio_values"]
        sr = r["performance"].get("sharpe_ratio", 0)
        label = f'{r["leverage"]:.1f}x (Sharpe {sr:.2f})'
        axes[0].plot(pv, label=label, linewidth=1.5, color=cmap[i])
        peak = np.maximum.accumulate(pv)
        dd = (pv - peak) / peak * 100
        axes[1].plot(dd, linewidth=1.0, color=cmap[i], alpha=0.7)
    axes[0].axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title(f"Equity Curves by Leverage — {strat_name} (5m)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("5-min Bars")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Chart saved: {save_path.name}")


def _sec(n):
    return ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
            "十一", "十二", "十三"][min(n, 13)]


def generate_report(
    crypto_data, stock_data,
    crypto_wf, crypto_cpcv, stock_wf, stock_cpcv,
    crypto_lev, stock_lev,
    crypto_config, stock_config,
    total_time,
):
    L = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    L.append("# 5 分钟级别全策略回测 + 杠杆率探索分析报告")
    L.append(f"\n> 生成时间: {now} | 总耗时: {total_time:.1f}s")
    L.append(f"> 数据周期: **5 分钟 (5m)** | 杠杆范围: {', '.join(f'{l:.1f}x' for l in LEVERAGE_LEVELS)}\n")

    # Executive summary
    L.append("## 一、执行摘要\n")
    L.append(f"本报告使用 **5 分钟 K 线**数据，对 **{len(crypto_data) + len(stock_data)}** 个资产的 "
             f"**{len(KERNEL_NAMES)}** 种策略进行参数优化，"
             f"并对 Top-{TOP_N} 策略在 **{len(LEVERAGE_LEVELS)} 种杠杆**下进行深度回测。\n")
    L.append("与日线回测的关键区别：")
    L.append("- 参数网格针对 **分钟级别** 重新设计（1-24 小时 lookback）")
    L.append(f"- 加密货币年化因子: **{crypto_config.bars_per_year:,.0f}** bars/year（非 365）")
    L.append(f"- 美股年化因子: **{stock_config.bars_per_year:,.0f}** bars/year（非 252）")
    L.append("- 成本按 **per-bar** 缩放（日资金费率 / 每天 bar 数）\n")

    # Data overview
    L.append("## 二、数据概览\n")
    L.append("| 资产 | 类型 | 5m Bars | 起始时间 | 结束时间 | 最新价格 |")
    L.append("|:-----|:-----|--------:|:---------|:---------|--------:|")
    for sym, df in {**crypto_data, **stock_data}.items():
        mtype = "加密货币" if sym in crypto_data else "美股"
        start = df["date"].iloc[0].strftime("%Y-%m-%d %H:%M") if "date" in df.columns else "N/A"
        end = df["date"].iloc[-1].strftime("%Y-%m-%d %H:%M") if "date" in df.columns else "N/A"
        price = df["close"].iloc[-1]
        L.append(f"| {sym} | {mtype} | {len(df):,} | {start} | {end} | ${price:,.2f} |")
    L.append("")

    # Cost model
    L.append("## 三、成本模型（5 分钟 per-bar 缩放）\n")
    L.append("| 参数 | 加密货币 | 美股 | 说明 |")
    L.append("|:-----|--------:|-----:|:-----|")
    L.append(f"| 手续费 | {crypto_config.commission_pct_buy*100:.2f}% | {stock_config.commission_pct_buy*100:.2f}% | 每笔交易 |")
    L.append(f"| 滑点 (bps) | {crypto_config.slippage_bps_buy:.0f} | {stock_config.slippage_bps_buy:.0f} | 每笔交易 |")
    dc_crypto = crypto_config.daily_funding_rate / crypto_config.bars_per_day
    dc_stock_borrow = stock_config.short_borrow_rate_annual / stock_config.bars_per_year
    L.append(f"| 资金费率/bar | {dc_crypto*10000:.4f} bps | 0 bps | 日费率/{crypto_config.bars_per_day:.0f} |")
    L.append(f"| 做空借贷/bar | 0 bps | {dc_stock_borrow*10000:.4f} bps | 年利率/{stock_config.bars_per_year:.0f} |")
    L.append(f"| bars_per_year | {crypto_config.bars_per_year:,.0f} | {stock_config.bars_per_year:,.0f} | 年化因子 |")
    L.append(f"| bars_per_day | {crypto_config.bars_per_day:.0f} | {stock_config.bars_per_day:.0f} | 成本缩放 |")
    L.append("")

    sec = 4

    # WF results
    for label, wf, cpcv, data_map in [
        ("加密货币", crypto_wf, crypto_cpcv, crypto_data),
        ("美股", stock_wf, stock_cpcv, stock_data),
    ]:
        if wf is None:
            continue
        L.append(f"## {_sec(sec)}、Walk-Forward 优化结果 — {label} (5m)\n")
        sec += 1
        L.append(f"> 资产: {', '.join(data_map.keys())} | "
                 f"总组合: {wf.total_combos:,} | "
                 f"速度: {wf.total_combos / max(0.1, wf.elapsed_seconds):,.0f} combos/s\n")
        L.append("| 排名 | 策略 | 参数 | OOS收益 | Sharpe | 最大回撤 | WF得分 |")
        L.append("|-----:|:-----|:-----|--------:|-------:|--------:|-------:|")
        ranked = sorted(wf.all_strategies.items(),
                        key=lambda kv: kv[1].get("wf_score", -1e18), reverse=True)
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

        if cpcv is not None:
            L.append(f"### CPCV 交叉验证 — {label}\n")
            L.append("| 排名 | 策略 | OOS均值 | OOS标准差 | 正分割比 | CPCV得分 |")
            L.append("|-----:|:-----|--------:|----------:|---------:|---------:|")
            ranked_c = sorted(cpcv.all_strategies.items(),
                              key=lambda kv: kv[1].get("cpcv_score", kv[1].get("wf_score", -1e18)),
                              reverse=True)
            for i, (sn, d) in enumerate(ranked_c[:10], 1):
                L.append(f"| {i} | **{sn}** | "
                         f"{d.get('oos_ret', d.get('oos_ret_mean', 0)):+.1f}% | "
                         f"{d.get('oos_ret_std', d.get('oos_dd', 0)):.1f}% | "
                         f"{d.get('pct_splits_positive', d.get('mc_pct_positive', 0))*100:.0f}% | "
                         f"{d.get('cpcv_score', d.get('wf_score', 0)):.1f} |")
            L.append("")

    # Leverage exploration
    for label, lev_results, data_map in [
        ("加密货币", crypto_lev, crypto_data),
        ("美股", stock_lev, stock_data),
    ]:
        if not lev_results:
            continue
        L.append(f"## {_sec(sec)}、杠杆率探索 — {label} (5m)\n")
        sec += 1

        for sn, rlist in lev_results.items():
            if not rlist:
                continue
            L.append(f"### {sn}\n")
            L.append("| 杠杆 | 年化收益 | Sharpe | Sortino | Calmar | 最大回撤 | 波动率 | 终值 | VaR 95% | Kelly |")
            L.append("|-----:|--------:|-------:|--------:|-------:|--------:|-------:|-----:|--------:|------:|")
            best_calmar_lev, best_calmar = 1.0, -999
            all_negative = all(r["performance"].get("annual_return", 0) < 0 for r in rlist)
            for r in rlist:
                p = r["performance"]
                lev = r["leverage"]
                if all_negative:
                    score = p.get("sharpe_ratio", -999)
                else:
                    score = p.get("calmar_ratio", -999)
                if score > best_calmar:
                    best_calmar = score
                    best_calmar_lev = lev
                kelly = compute_kelly(r["daily_returns"])
                L.append(
                    f"| **{lev:.1f}x** | {p.get('annual_return', 0)*100:+.2f}% | "
                    f"{p.get('sharpe_ratio', 0):.3f} | {p.get('sortino_ratio', 0):.3f} | "
                    f"{p.get('calmar_ratio', 0):.3f} | {p.get('max_drawdown', 0)*100:.2f}% | "
                    f"{p.get('volatility', 0)*100:.2f}% | ${p.get('final_value', 0):,.0f} | "
                    f"{p.get('value_at_risk_95', 0)*100:.2f}% | {kelly:.2f} |"
                )
            L.append("")

            syms = list(data_map.keys())
            L.append(f"#### {sn} — 各资产各杠杆收益率\n")
            header = "| 杠杆 | " + " | ".join(syms) + " |"
            sep = "|-----:|" + "|".join(["------:"] * len(syms)) + "|"
            L.append(header)
            L.append(sep)
            for r in rlist:
                vals = [f"{r['per_asset'][s].ret_pct:+.1f}%" if s in r["per_asset"] else "N/A" for s in syms]
                L.append(f"| **{r['leverage']:.1f}x** | " + " | ".join(vals) + " |")
            L.append("")
            if all_negative:
                L.append(f"**{sn} 最优杠杆推荐: `{best_calmar_lev:.1f}x`** (Sharpe 最优; 注意: 所有杠杆均亏损)\n")
            else:
                L.append(f"**{sn} 最优杠杆推荐: `{best_calmar_lev:.1f}x`** (Calmar 最优)\n")

    # Sharpe matrix
    L.append(f"## {_sec(sec)}、杠杆率综合对比矩阵\n")
    sec += 1
    for label, lev_res in [("加密货币", crypto_lev), ("美股", stock_lev)]:
        if not lev_res:
            continue
        L.append(f"### {label} — Sharpe Ratio 矩阵\n")
        strats = list(lev_res.keys())
        header = "| 策略 | " + " | ".join(f"{l:.1f}x" for l in LEVERAGE_LEVELS) + " | 最优杠杆 |"
        sep = "|:-----|" + "|".join(["------:"] * len(LEVERAGE_LEVELS)) + "|:---------|"
        L.append(header)
        L.append(sep)
        for sn in strats:
            rlist = lev_res[sn]
            vals, best_sr, best_lev = [], -999, 1.0
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
    for label, lev_res in [("加密货币", crypto_lev), ("美股", stock_lev)]:
        if not lev_res:
            continue
        L.append(f"#### {label}\n")
        header = "| 策略 | " + " | ".join(f"{l:.1f}x" for l in LEVERAGE_LEVELS) + " |"
        sep = "|:-----|" + "|".join(["------:"] * len(LEVERAGE_LEVELS)) + "|"
        L.append(header)
        L.append(sep)
        for sn in lev_res:
            vals = [f"{abs(r['performance'].get('max_drawdown', 0))*100:.1f}%" for r in lev_res[sn]]
            L.append(f"| **{sn}** | " + " | ".join(vals) + " |")
        L.append("")

    # Daily vs 5m comparison note
    L.append(f"## {_sec(sec)}、日线 vs 5 分钟对比分析\n")
    sec += 1
    L.append("| 维度 | 日线 (1D) | 5 分钟 (5m) |")
    L.append("|:-----|:---------|:-----------|")
    L.append(f"| bars_per_year (crypto) | 365 | {int(crypto_config.bars_per_year):,} |")
    L.append(f"| bars_per_year (stock) | 252 | {int(stock_config.bars_per_year):,} |")
    L.append("| 参数范围 | MA(5-105), MACD(28,112,3) | MA(6-288), MACD(6-96,5-9) |")
    L.append("| 数据量 (3m crypto) | ~90 bars | ~26,000 bars |")
    L.append("| 交易频率 | 低（数天/笔） | 高（数小时/笔） |")
    L.append("| 成本敏感度 | 低 | 高（频繁交易放大手续费） |")
    L.append("| 信噪比 | 高 | 低（更多市场噪声） |")
    L.append("| 适合策略类型 | 趋势跟踪、动量 | 均值回归、短期突破 |")
    L.append("")

    # Investment advice
    L.append(f"## {_sec(sec)}、投资建议 — 5 分钟策略杠杆配置\n")
    sec += 1
    for label, lev_res in [("加密货币", crypto_lev), ("美股", stock_lev)]:
        if not lev_res:
            continue
        L.append(f"### {label}\n")
        for sn, rlist in lev_res.items():
            if not rlist:
                continue
            all_neg = all(r["performance"].get("annual_return", 0) < 0 for r in rlist)
            if all_neg:
                best_r = max(rlist, key=lambda r: r["performance"].get("sharpe_ratio", -999))
            else:
                best_r = max(rlist, key=lambda r: r["performance"].get("calmar_ratio", -999))
            p = best_r["performance"]
            kelly = compute_kelly(best_r["daily_returns"])
            mdd = abs(p.get("max_drawdown", 0))
            ann_ret = p.get("annual_return", 0)
            L.append(f"**{sn}:**")
            L.append(f"- {'Sharpe' if all_neg else 'Calmar'} 最优: **{best_r['leverage']:.1f}x** "
                     f"(年化={ann_ret*100:+.1f}%, "
                     f"Sharpe={p.get('sharpe_ratio', 0):.3f}, 回撤={mdd*100:.1f}%)")
            L.append(f"- Half-Kelly: **{kelly:.2f}x**")
            if all_neg:
                L.append(f"- 建议: **策略在 5m 周期不盈利，不建议使用任何杠杆**")
            elif mdd > 0.40:
                L.append(f"- 建议: 回撤过高，**降至 {min(best_r['leverage'], 1.5):.1f}x**")
            elif mdd > 0.25:
                L.append(f"- 建议: **保守用 {min(best_r['leverage'], 2.0):.1f}x**")
            else:
                L.append(f"- 建议: 可使用 **{best_r['leverage']:.1f}x**")
            L.append("")

    L.append("### 5 分钟策略特别注意事项\n")
    L.append("1. **交易频率高**: 5 分钟策略交易次数远高于日线，手续费累积影响显著")
    L.append("2. **滑点敏感**: 高频交易中滑点的影响被放大")
    L.append("3. **信噪比低**: 5 分钟数据噪声更大，过拟合风险更高")
    L.append("4. **延迟关键**: 实盘中执行延迟对 5 分钟策略影响远大于日线")
    L.append("5. **夜间风险 (美股)**: 收盘后无交易，隔夜跳空无法对冲\n")

    L.append("### 重要免责声明\n")
    L.append("> 5 分钟级别回测的过拟合风险高于日线。历史回测结果不代表未来表现。"
             "高频策略对执行质量和延迟极为敏感，实盘表现可能显著低于回测。\n")

    # Appendix
    L.append(f"## {_sec(sec)}、技术附录\n")
    L.append(f"- 回测引擎: Numba JIT kernel ({len(KERNEL_NAMES)} strategies)")
    L.append(f"- 数据周期: 5 分钟 (5m)")
    L.append(f"- 优化方法: Walk-Forward + CPCV")
    L.append(f"- 杠杆探索: {', '.join(f'{l:.1f}x' for l in LEVERAGE_LEVELS)}")
    L.append(f"- 无风险利率: {RISK_FREE_RATE*100:.1f}%")
    L.append(f"- 初始资金: ${INITIAL_CAPITAL:,.0f}")
    L.append(f"- 加密货币 bars_per_year: {int(crypto_config.bars_per_year):,}")
    L.append(f"- 美股 bars_per_year: {int(stock_config.bars_per_year):,}")
    total_combos = sum(len(v) for v in PARAM_GRIDS_5M.values())
    L.append(f"- 5m 参数网格总组合: {total_combos:,}")
    L.append("")

    return "\n".join(L)


def main():
    t_start = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  5 分钟级别全策略回测 + 杠杆探索")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading 5m data ...")
    crypto_data, stock_data = load_all_data()
    print(f"  Crypto: {list(crypto_data.keys())} ({sum(len(d) for d in crypto_data.values()):,} bars)")
    print(f"  Stocks: {list(stock_data.keys())} ({sum(len(d) for d in stock_data.values()):,} bars)")
    if not crypto_data and not stock_data:
        print("ERROR: No 5m data. Run download_5m_data.py first.")
        sys.exit(1)

    # Optimize at 1x
    print(f"\n[2/5] WF + CPCV optimization (5m, 1x baseline) ...")
    crypto_config = BacktestConfig.crypto(leverage=1.0, interval=INTERVAL)
    stock_config = BacktestConfig.stock_ibkr(leverage=1.0, interval=INTERVAL)
    print(f"  Crypto bars_per_year={crypto_config.bars_per_year:,.0f}, bars_per_day={crypto_config.bars_per_day:.0f}")
    print(f"  Stock  bars_per_year={stock_config.bars_per_year:,.0f}, bars_per_day={stock_config.bars_per_day:.0f}")
    crypto_wf, crypto_cpcv = run_optimization(crypto_data, crypto_config, "Crypto")
    stock_wf, stock_cpcv = run_optimization(stock_data, stock_config, "US Stocks")

    # Leverage exploration
    print(f"\n[3/5] Leverage exploration ...")
    crypto_lev, stock_lev = {}, {}
    if crypto_wf:
        top_crypto = get_top_strategies(crypto_wf, TOP_N)
        print(f"  Crypto top-{TOP_N}: {[s[0] for s in top_crypto]}")
        crypto_lev = run_leverage_exploration(top_crypto, crypto_data, BacktestConfig.crypto, "Crypto")
    if stock_wf:
        top_stock = get_top_strategies(stock_wf, TOP_N)
        print(f"  Stock top-{TOP_N}: {[s[0] for s in top_stock]}")
        stock_lev = run_leverage_exploration(top_stock, stock_data, BacktestConfig.stock_ibkr, "Stocks")

    # Charts
    print("\n[4/5] Generating charts ...")
    if crypto_lev:
        plot_leverage_comparison(crypto_lev, "Leverage Analysis — Crypto (5m)", CHART_DIR / "crypto_5m_leverage.png")
        for sn, rlist in crypto_lev.items():
            if rlist:
                plot_equity_by_leverage(rlist, sn, CHART_DIR / f"crypto_5m_{sn.lower()}_equity.png")
    if stock_lev:
        plot_leverage_comparison(stock_lev, "Leverage Analysis — US Stocks (5m)", CHART_DIR / "stock_5m_leverage.png")
        for sn, rlist in stock_lev.items():
            if rlist:
                plot_equity_by_leverage(rlist, sn, CHART_DIR / f"stock_5m_{sn.lower()}_equity.png")

    # Report
    total_time = time.perf_counter() - t_start
    print(f"\n[5/5] Generating report ...")
    report = generate_report(
        crypto_data, stock_data,
        crypto_wf, crypto_cpcv, stock_wf, stock_cpcv,
        crypto_lev, stock_lev,
        crypto_config, stock_config,
        total_time,
    )
    report_path = REPORT_DIR / "intraday_5m_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("  5-MINUTE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Report: {report_path}")
    print(f"  Charts: {CHART_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
