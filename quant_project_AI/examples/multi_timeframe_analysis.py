#!/usr/bin/env python3
"""
Multi-Timeframe Comprehensive Backtest + Leverage Exploration.

Runs 18 strategies across 5 timeframes (5m, 15m, 1h, 4h, 1d) on
crypto and US-stock data, then performs leverage exploration on
the top performers at each timeframe.

Outputs: reports/multi_timeframe_analysis_report.md + charts

Usage:
    python examples/multi_timeframe_analysis.py
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
CHART_DIR = REPORT_DIR / "multi_tf_charts"
INITIAL_CAPITAL = 100_000.0
RISK_FREE_RATE = 0.04

CRYPTO_ASSETS = ["BTC", "ETH", "SOL", "BNB"]
STOCK_ASSETS = ["SPY", "QQQ", "NVDA", "META", "MSFT", "AMZN"]
LEVERAGE_LEVELS = [1.0, 1.5, 2.0, 3.0, 5.0]
MIN_BARS = 400
TOP_N = 3

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
TF_LABELS = {"5m": "5 分钟", "15m": "15 分钟", "1h": "1 小时", "4h": "4 小时", "1d": "日线"}

# ---------------------------------------------------------------------------
# Parameter grids scaled to each timeframe's physical lookback
# ---------------------------------------------------------------------------
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

PARAM_GRIDS_15M = {
    "MA":          [(f, s) for f in (4, 8, 16, 32) for s in (32, 64, 96) if f < s],
    "RSI":         [(p, ob, os_) for p in (7, 14, 21) for ob in (25, 30) for os_ in (70, 75)],
    "MACD":        [(f, s, sig) for f in (4, 8, 16) for s in (20, 40, 64) for sig in (5, 9) if f < s],
    "Bollinger":   [(p, std) for p in (8, 16, 32) for std in (1.5, 2.0, 2.5)],
    "Keltner":     [(p, atr, m) for p in (16, 32, 64) for atr in (7, 14) for m in (1.0, 1.5, 2.0)],
    "Drift":       [(p, thr, atr) for p in (4, 8, 16) for thr in (0.5, 0.65, 0.8) for atr in (7, 14)],
    "KAMA":        [(er, f, s, m, atr) for er in (10, 20) for f in (2, 3) for s in (15, 30) for m in (1.5, 2.0) for atr in (7, 14)],
    "DualMom":     [(s, l) for s in (8, 16, 32) for l in (32, 64, 96) if s < l],
    "Turtle":      [(e, x, atr, m) for e in (8, 16) for x in (4, 8) for atr in (7, 14) for m in (1.5, 2.5)],
    "Donchian":    [(p, atr, m) for p in (8, 16, 32) for atr in (7, 14) for m in (1.5, 2.0, 2.5)],
    "ZScore":      [(p, en, ex, sl) for p in (16, 32, 64) for en in (2.0, 2.5) for ex in (0.0, 0.5) for sl in (2.5, 3.0)],
    "MomBreak":    [(lb, thr, atr, m) for lb in (8, 16, 32) for thr in (0.04, 0.08) for atr in (7, 14) for m in (1.5, 2.5)],
    "MultiFactor": [(ma, mom, vol, w1, w2) for ma in (7, 14) for mom in (16, 32) for vol in (8, 16) for w1 in (0.45, 0.55) for w2 in (0.25, 0.35)],
    "Consensus":   [(ma_s, ma_l, rsi, bb, rs_lo, rs_hi, thr) for ma_s in (4, 8) for ma_l in (32, 64) for rsi in (14,) for bb in (16,) for rs_lo in (30,) for rs_hi in (70,) for thr in (2, 3)],
    "MESA":        [(fl, sl) for fl in (0.3, 0.5, 0.7) for sl in (0.05, 0.1)],
    "RegimeEMA":   [(atr, thr, fast, mid, slow) for atr in (7, 14) for thr in (0.01, 0.015) for fast in (4, 8) for mid in (16,) for slow in (32, 64)],
    "VolRegime":   [(vol_w, vol_thr, ma_s, ma_l, rs_lo, rs_hi) for vol_w in (8, 16) for vol_thr in (0.015, 0.03) for ma_s in (4, 8) for ma_l in (32,) for rs_lo in (30,) for rs_hi in (75,)],
    "RAMOM":       [(lb, vol, up, dn) for lb in (8, 16, 32) for vol in (8, 16) for up in (1.0, 2.0) for dn in (0.5, 1.0)],
}

PARAM_GRIDS_1H = {
    "MA":          [(f, s) for f in (3, 6, 12, 24) for s in (12, 24, 48) if f < s],
    "RSI":         [(p, ob, os_) for p in (7, 14, 21) for ob in (25, 30) for os_ in (70, 75)],
    "MACD":        [(f, s, sig) for f in (3, 6, 12) for s in (12, 24, 48) for sig in (5, 9) if f < s],
    "Bollinger":   [(p, std) for p in (6, 12, 20, 36) for std in (1.5, 2.0, 2.5)],
    "Keltner":     [(p, atr, m) for p in (12, 24, 48) for atr in (7, 14) for m in (1.0, 1.5, 2.0)],
    "Drift":       [(p, thr, atr) for p in (3, 6, 12) for thr in (0.5, 0.65, 0.8) for atr in (7, 14)],
    "KAMA":        [(er, f, s, m, atr) for er in (10, 20) for f in (2, 3) for s in (15, 30) for m in (1.5, 2.0) for atr in (7, 14)],
    "DualMom":     [(s, l) for s in (6, 12, 24) for l in (24, 48, 96) if s < l],
    "Turtle":      [(e, x, atr, m) for e in (6, 12) for x in (3, 6) for atr in (7, 14) for m in (1.5, 2.5)],
    "Donchian":    [(p, atr, m) for p in (6, 12, 24) for atr in (7, 14) for m in (1.5, 2.0, 2.5)],
    "ZScore":      [(p, en, ex, sl) for p in (12, 24, 48) for en in (2.0, 2.5) for ex in (0.0, 0.5) for sl in (2.5, 3.0)],
    "MomBreak":    [(lb, thr, atr, m) for lb in (6, 12, 24) for thr in (0.04, 0.08) for atr in (7, 14) for m in (1.5, 2.5)],
    "MultiFactor": [(ma, mom, vol, w1, w2) for ma in (7, 14) for mom in (12, 24) for vol in (6, 12) for w1 in (0.45, 0.55) for w2 in (0.25, 0.35)],
    "Consensus":   [(ma_s, ma_l, rsi, bb, rs_lo, rs_hi, thr) for ma_s in (3, 6) for ma_l in (24, 48) for rsi in (14,) for bb in (12,) for rs_lo in (30,) for rs_hi in (70,) for thr in (2, 3)],
    "MESA":        [(fl, sl) for fl in (0.3, 0.5, 0.7) for sl in (0.05, 0.1)],
    "RegimeEMA":   [(atr, thr, fast, mid, slow) for atr in (7, 14) for thr in (0.01, 0.015) for fast in (3, 6) for mid in (12,) for slow in (24, 48)],
    "VolRegime":   [(vol_w, vol_thr, ma_s, ma_l, rs_lo, rs_hi) for vol_w in (6, 12) for vol_thr in (0.015, 0.03) for ma_s in (3, 6) for ma_l in (24,) for rs_lo in (30,) for rs_hi in (75,)],
    "RAMOM":       [(lb, vol, up, dn) for lb in (6, 12, 24) for vol in (6, 12) for up in (1.0, 2.0) for dn in (0.5, 1.0)],
}

PARAM_GRIDS_4H = {
    "MA":          [(f, s) for f in (3, 6, 12) for s in (12, 24, 48) if f < s],
    "RSI":         [(p, ob, os_) for p in (5, 10, 14) for ob in (25, 30) for os_ in (70, 75)],
    "MACD":        [(f, s, sig) for f in (3, 6, 12) for s in (12, 24, 48) for sig in (3, 5, 9) if f < s],
    "Bollinger":   [(p, std) for p in (5, 10, 20) for std in (1.5, 2.0, 2.5)],
    "Keltner":     [(p, atr, m) for p in (6, 12, 24) for atr in (5, 10) for m in (1.0, 1.5, 2.0)],
    "Drift":       [(p, thr, atr) for p in (3, 6, 10) for thr in (0.5, 0.65, 0.8) for atr in (5, 10)],
    "KAMA":        [(er, f, s, m, atr) for er in (10, 20) for f in (2, 3) for s in (15, 30) for m in (1.5, 2.0) for atr in (5, 10)],
    "DualMom":     [(s, l) for s in (3, 6, 12) for l in (12, 24, 48) if s < l],
    "Turtle":      [(e, x, atr, m) for e in (5, 10) for x in (3, 5) for atr in (5, 10) for m in (1.5, 2.5)],
    "Donchian":    [(p, atr, m) for p in (5, 10, 20) for atr in (5, 10) for m in (1.5, 2.0, 2.5)],
    "ZScore":      [(p, en, ex, sl) for p in (6, 12, 24) for en in (2.0, 2.5) for ex in (0.0, 0.5) for sl in (2.5, 3.0)],
    "MomBreak":    [(lb, thr, atr, m) for lb in (6, 12, 24) for thr in (0.04, 0.08) for atr in (5, 10) for m in (1.5, 2.5)],
    "MultiFactor": [(ma, mom, vol, w1, w2) for ma in (5, 10) for mom in (6, 12) for vol in (6, 12) for w1 in (0.45, 0.55) for w2 in (0.25, 0.35)],
    "Consensus":   [(ma_s, ma_l, rsi, bb, rs_lo, rs_hi, thr) for ma_s in (3, 6) for ma_l in (12, 24) for rsi in (10,) for bb in (10,) for rs_lo in (30,) for rs_hi in (70,) for thr in (2, 3)],
    "MESA":        [(fl, sl) for fl in (0.3, 0.5, 0.7) for sl in (0.05, 0.1)],
    "RegimeEMA":   [(atr, thr, fast, mid, slow) for atr in (5, 10) for thr in (0.01, 0.02) for fast in (3, 6) for mid in (12,) for slow in (24, 48)],
    "VolRegime":   [(vol_w, vol_thr, ma_s, ma_l, rs_lo, rs_hi) for vol_w in (6, 12) for vol_thr in (0.02, 0.04) for ma_s in (3, 6) for ma_l in (12,) for rs_lo in (30,) for rs_hi in (75,)],
    "RAMOM":       [(lb, vol, up, dn) for lb in (6, 12, 24) for vol in (6, 12) for up in (1.0, 2.0) for dn in (0.5, 1.0)],
}

TF_GRIDS = {
    "5m": PARAM_GRIDS_5M,
    "15m": PARAM_GRIDS_15M,
    "1h": PARAM_GRIDS_1H,
    "4h": PARAM_GRIDS_4H,
    "1d": None,  # uses DEFAULT_PARAM_GRIDS
}

plt.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})


# ──────────────────────── helpers ────────────────────────

def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in list(df.columns):
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


def load_tf_data(tf: str):
    crypto, stocks = {}, {}
    if tf == "1d":
        sub = DATA_DIR
        suffix = ".csv"
    else:
        sub = DATA_DIR / tf
        suffix = f"_{tf}.csv"
    for sym in CRYPTO_ASSETS:
        path = sub / f"{sym}{suffix}"
        df = _load_csv(path)
        if df is not None and len(df) >= MIN_BARS:
            crypto[sym] = df
    for sym in STOCK_ASSETS:
        path = sub / f"{sym}{suffix}"
        df = _load_csv(path)
        if df is not None and len(df) >= MIN_BARS:
            stocks[sym] = df
    return crypto, stocks


def run_optimization(data_map, config, label, grids):
    if not data_map:
        return None, None
    print(f"\n  Optimizing [{label}]: {list(data_map.keys())}")
    t0 = time.perf_counter()
    kwargs = {"param_grids": grids} if grids else {}
    wf = optimize(data_map, config, method="wf", **kwargs)
    wf_t = time.perf_counter() - t0
    print(f"    WF: {wf.total_combos:,} combos in {wf_t:.1f}s "
          f"({wf.total_combos / max(0.1, wf_t):,.0f}/s)")
    t0 = time.perf_counter()
    cpcv = optimize(data_map, config, method="cpcv", **kwargs)
    cpcv_t = time.perf_counter() - t0
    print(f"    CPCV: {cpcv.total_combos:,} combos in {cpcv_t:.1f}s")
    return wf, cpcv


def get_top(wf, n=TOP_N):
    ranked = sorted(wf.all_strategies.items(),
                    key=lambda kv: kv[1].get("wf_score", -1e18), reverse=True)[:n]
    return [(name, meta.get("params")) for name, meta in ranked if meta.get("params") is not None]


def run_detailed(strat_name, params, data_map, config):
    per_asset = {}
    combined_eq = []
    for sym, df in data_map.items():
        try:
            res = backtest(strat_name, params, df, config, detailed=True)
            per_asset[sym] = res
            if res.equity is not None:
                combined_eq.append(res.equity)
        except Exception:
            pass
    if not combined_eq:
        return {}
    ml = min(len(e) for e in combined_eq)
    avg_eq = np.mean([e[:ml] for e in combined_eq], axis=0)
    pv = avg_eq * INITIAL_CAPITAL
    rets = np.diff(pv) / pv[:-1]
    rets = rets[np.isfinite(rets)]
    bpy = config.bars_per_year
    analyzer = PerformanceAnalyzer(risk_free_rate=RISK_FREE_RATE, periods_per_year=bpy)
    perf = analyzer.analyze(pv, rets, INITIAL_CAPITAL, n_trials=len(KERNEL_NAMES))
    return {"strategy": strat_name, "params": params, "per_asset": per_asset,
            "portfolio_values": pv, "bar_returns": rets, "performance": perf}


def run_leverage(top_strats, data_map, config_fn, tf):
    results = {}
    for sn, params in top_strats:
        sl = []
        for lev in LEVERAGE_LEVELS:
            cfg = config_fn(leverage=lev, interval=tf)
            r = run_detailed(sn, params, data_map, cfg)
            if r:
                r["leverage"] = lev
                sl.append(r)
        results[sn] = sl
    return results


def compute_kelly(rets):
    if len(rets) < 30:
        return 0.0
    mu, var = np.mean(rets), np.var(rets)
    if var < 1e-15:
        return 0.0
    return max(0, min(mu / var / 2, 20.0))


# ──────────────────────── charts ────────────────────────

def plot_cross_tf_sharpe(all_results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, market in zip(axes, ["crypto", "stock"]):
        tfs, strats_data = [], {}
        for tf in TIMEFRAMES:
            r = all_results.get(tf, {}).get(market)
            if not r or not r.get("wf"):
                continue
            tfs.append(tf)
            for sn, d in r["wf"].all_strategies.items():
                strats_data.setdefault(sn, {})[tf] = d.get("sharpe", d.get("wf_score", 0))
        if not tfs:
            continue
        top_strats = sorted(strats_data.keys(),
                            key=lambda s: max(strats_data[s].values(), default=-999),
                            reverse=True)[:6]
        x = np.arange(len(tfs))
        w = 0.12
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_strats)))
        for i, sn in enumerate(top_strats):
            vals = [strats_data[sn].get(tf, 0) for tf in tfs]
            ax.bar(x + i * w, vals, w, label=sn, color=colors[i])
        ax.set_xticks(x + w * len(top_strats) / 2)
        ax.set_xticklabels(tfs)
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title(f"{'Crypto' if market == 'crypto' else 'US Stocks'} — Sharpe by Timeframe",
                     fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_cost_drag(all_results, save_path):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {"crypto": plt.cm.Blues(0.7), "stock": plt.cm.Oranges(0.7)}
    x_vals, y_vals, labels_done = [], [], set()
    for market in ["crypto", "stock"]:
        for tf in TIMEFRAMES:
            r = all_results.get(tf, {}).get(market)
            if not r or not r.get("wf"):
                continue
            wf = r["wf"]
            for sn, d in wf.all_strategies.items():
                nt = d.get("nt", 0)
                ret = d.get("oos_ret", d.get("ret", 0))
                if nt > 0:
                    label = market if market not in labels_done else None
                    labels_done.add(market)
                    ax.scatter(nt, ret, c=[colors[market]], s=30, alpha=0.5, label=label)
    ax.set_xlabel("Number of Trades")
    ax.set_ylabel("OOS Return (%)")
    ax.set_title("Trade Frequency vs Return Across All Timeframes", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_equity_per_tf(all_results, market, save_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(TIMEFRAMES)))
    for i, tf in enumerate(TIMEFRAMES):
        r = all_results.get(tf, {}).get(market)
        if not r or not r.get("best_detail"):
            continue
        bd = r["best_detail"]
        pv = bd["portfolio_values"]
        sn = bd["strategy"]
        ax.plot(np.linspace(0, 1, len(pv)), pv, label=f"{tf} — {sn}",
                color=cmap[i], linewidth=1.8)
    ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title(f"Best Strategy Equity by Timeframe — {'Crypto' if market == 'crypto' else 'US Stocks'}",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────── report ────────────────────────

def _sec(n):
    return ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
            "十一", "十二", "十三", "十四", "十五"][min(n, 15)]


def generate_report(all_results, total_time):
    L = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    L.append("# 多周期全策略回测 + 杠杆率探索分析报告")
    L.append(f"\n> 生成时间: {now} | 总耗时: {total_time:.1f}s")
    L.append(f"> 周期: **{', '.join(TIMEFRAMES)}** | 杠杆范围: {', '.join(f'{l:.1f}x' for l in LEVERAGE_LEVELS)}\n")

    # ── 1. Executive Summary ──
    sec = 1
    L.append(f"## {_sec(sec)}、执行摘要\n")
    L.append("本报告在 **5 个时间周期**（5m, 15m, 1h, 4h, 1d）上对 **18 种策略**进行全面回测，"
             "覆盖加密货币和美股市场，旨在找到最优周期-策略-杠杆组合。\n")
    L.append("| 周期 | 加密货币最佳 | Sharpe | 美股最佳 | Sharpe |")
    L.append("|:-----|:-----------|-------:|:--------|-------:|")
    for tf in TIMEFRAMES:
        cr = all_results.get(tf, {}).get("crypto", {})
        sr = all_results.get(tf, {}).get("stock", {})
        c_best = cr.get("best_name", "N/A")
        c_sh = cr.get("best_sharpe", 0)
        s_best = sr.get("best_name", "N/A")
        s_sh = sr.get("best_sharpe", 0)
        L.append(f"| **{tf}** | {c_best} | {c_sh:.3f} | {s_best} | {s_sh:.3f} |")
    L.append("")
    sec += 1

    # ── 2. Data Overview ──
    L.append(f"## {_sec(sec)}、数据概览\n")
    L.append("| 周期 | 市场 | 资产数 | 总 bars | 时间跨度 |")
    L.append("|:-----|:-----|-------:|-------:|:---------|")
    for tf in TIMEFRAMES:
        for market in ["crypto", "stock"]:
            r = all_results.get(tf, {}).get(market, {})
            data_map = r.get("data", {})
            if not data_map:
                continue
            total_bars = sum(len(d) for d in data_map.values())
            n_assets = len(data_map)
            first_df = list(data_map.values())[0]
            if "date" in first_df.columns:
                span = f"{first_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {first_df['date'].iloc[-1].strftime('%Y-%m-%d')}"
            else:
                span = f"{len(first_df)} bars"
            label = "加密货币" if market == "crypto" else "美股"
            L.append(f"| **{tf}** | {label} | {n_assets} | {total_bars:,} | {span} |")
    L.append("")
    sec += 1

    # ── 3. Cost model ──
    L.append(f"## {_sec(sec)}、成本模型（per-bar 自适应缩放）\n")
    L.append("| 周期 | bars/year (crypto) | bars/year (stock) | bars/day (crypto) | bars/day (stock) |")
    L.append("|:-----|-------------------:|------------------:|------------------:|-----------------:|")
    for tf in TIMEFRAMES:
        cc = BacktestConfig.crypto(interval=tf)
        sc = BacktestConfig.stock_ibkr(interval=tf)
        L.append(f"| **{tf}** | {int(cc.bars_per_year):,} | {int(sc.bars_per_year):,} | "
                 f"{int(cc.bars_per_day)} | {int(sc.bars_per_day)} |")
    L.append("\n交易成本不变（Crypto: 0.04% 佣金 + 3bps 滑点; 美股: 0.05% + 5bps），"
             "但高频周期累积更多交易次数，总成本拖累更大。\n")
    sec += 1

    # ── 4-8. Per-TF WF + CPCV + Leverage ──
    for tf in TIMEFRAMES:
        L.append(f"## {_sec(sec)}、{TF_LABELS[tf]} ({tf}) 回测结果\n")
        sec += 1

        for market, label in [("crypto", "加密货币"), ("stock", "美股")]:
            r = all_results.get(tf, {}).get(market, {})
            wf = r.get("wf")
            cpcv = r.get("cpcv")
            lev_res = r.get("leverage", {})
            if not wf:
                L.append(f"### {label} — 数据不足，跳过\n")
                continue

            L.append(f"### {label} — Walk-Forward 优化\n")
            L.append(f"> 资产: {', '.join(r.get('data', {}).keys())} | "
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
                sc = d.get("wf_score", 0)
                if abs(sc) > 1e12:
                    sc_str = "N/A"
                else:
                    sc_str = f"{sc:.1f}"
                L.append(f"| {i} | **{sn}** | `{ps}` | "
                         f"{d.get('oos_ret', 0):+.1f}% | "
                         f"{d.get('sharpe', 0):.2f} | "
                         f"{d.get('oos_dd', 0):.1f}% | "
                         f"{sc_str} |")
            L.append("")

            # CPCV
            if cpcv:
                L.append(f"### {label} — CPCV 交叉验证 (Top 10)\n")
                L.append("| 排名 | 策略 | OOS均值 | OOS标准差 | 正分割比 | CPCV得分 |")
                L.append("|-----:|:-----|--------:|----------:|---------:|---------:|")
                ranked_c = sorted(cpcv.all_strategies.items(),
                                  key=lambda kv: kv[1].get("cpcv_score", kv[1].get("wf_score", -1e18)),
                                  reverse=True)
                for i, (sn, d) in enumerate(ranked_c[:10], 1):
                    sc = d.get("cpcv_score", d.get("wf_score", 0))
                    if abs(sc) > 1e12:
                        sc_str = "N/A"
                    else:
                        sc_str = f"{sc:.1f}"
                    L.append(f"| {i} | **{sn}** | "
                             f"{d.get('oos_ret', d.get('oos_ret_mean', 0)):+.1f}% | "
                             f"{d.get('oos_ret_std', d.get('oos_dd', 0)):.1f}% | "
                             f"{d.get('pct_splits_positive', d.get('mc_pct_positive', 0))*100:.0f}% | "
                             f"{sc_str} |")
                L.append("")

            # Leverage
            if lev_res:
                L.append(f"### {label} — 杠杆率探索 (Top-{TOP_N})\n")
                for sn, rlist in lev_res.items():
                    if not rlist:
                        continue
                    L.append(f"#### {sn}\n")
                    L.append("| 杠杆 | 年化收益 | Sharpe | Sortino | Calmar | 最大回撤 | 终值 |")
                    L.append("|-----:|--------:|-------:|--------:|-------:|--------:|-----:|")
                    for rv in rlist:
                        p = rv["performance"]
                        L.append(
                            f"| **{rv['leverage']:.1f}x** | "
                            f"{p.get('annual_return', 0)*100:+.2f}% | "
                            f"{p.get('sharpe_ratio', 0):.3f} | "
                            f"{p.get('sortino_ratio', 0):.3f} | "
                            f"{p.get('calmar_ratio', 0):.3f} | "
                            f"{p.get('max_drawdown', 0)*100:.2f}% | "
                            f"${p.get('final_value', 0):,.0f} |"
                        )
                    L.append("")

    # ── Cross-TF comparison matrix ──
    L.append(f"## {_sec(sec)}、跨周期 Sharpe Ratio 对比矩阵\n")
    sec += 1
    for market, label in [("crypto", "加密货币"), ("stock", "美股")]:
        all_strats = set()
        for tf in TIMEFRAMES:
            wf = all_results.get(tf, {}).get(market, {}).get("wf")
            if wf:
                for sn in wf.all_strategies:
                    all_strats.add(sn)
        if not all_strats:
            continue
        L.append(f"### {label}\n")
        header = "| 策略 | " + " | ".join(TIMEFRAMES) + " | 最优周期 |"
        sep = "|:-----|" + "|".join(["------:"] * len(TIMEFRAMES)) + "|:---------|"
        L.append(header)
        L.append(sep)
        strat_best = []
        for sn in sorted(all_strats):
            vals, best_tf, best_val = [], "N/A", -999
            for tf in TIMEFRAMES:
                wf = all_results.get(tf, {}).get(market, {}).get("wf")
                if wf and sn in wf.all_strategies:
                    v = wf.all_strategies[sn].get("sharpe", 0)
                    vals.append(f"{v:.2f}")
                    if v > best_val:
                        best_val = v
                        best_tf = tf
                else:
                    vals.append("—")
            strat_best.append((sn, best_val))
            L.append(f"| **{sn}** | " + " | ".join(vals) + f" | **{best_tf}** |")
        L.append("")

    # ── Trade frequency analysis ──
    L.append(f"## {_sec(sec)}、交易频率与成本拖累分析\n")
    sec += 1
    L.append("| 周期 | 市场 | 平均交易次数 | 平均 OOS 收益 | 盈利策略数 | 成本影响评估 |")
    L.append("|:-----|:-----|------------:|-------------:|----------:|:-----------|")
    for tf in TIMEFRAMES:
        for market, label in [("crypto", "加密货币"), ("stock", "美股")]:
            wf = all_results.get(tf, {}).get(market, {}).get("wf")
            if not wf:
                continue
            trades = [d.get("oos_trades", d.get("nt", 0)) for d in wf.all_strategies.values()]
            rets = [d.get("oos_ret", 0) for d in wf.all_strategies.values()]
            avg_t = np.mean(trades) if trades else 0
            avg_r = np.mean(rets) if rets else 0
            n_profit = sum(1 for r in rets if r > 0)
            if avg_t > 500:
                impact = "极高 — 手续费严重侵蚀收益"
            elif avg_t > 100:
                impact = "高 — 需选择低频策略"
            elif avg_t > 30:
                impact = "中等 — 交易成本可控"
            else:
                impact = "低 — 成本影响较小"
            L.append(f"| **{tf}** | {label} | {avg_t:.0f} | {avg_r:+.1f}% | {n_profit}/{len(rets)} | {impact} |")
    L.append("")

    # ── Investment advice ──
    L.append(f"## {_sec(sec)}、投资建议\n")
    sec += 1
    L.append("### 最优周期-策略-杠杆推荐\n")
    for market, label in [("crypto", "加密货币"), ("stock", "美股")]:
        L.append(f"#### {label}\n")
        best_overall = {"tf": "1d", "strat": "N/A", "sharpe": -999, "params": None, "lev": 1.0, "ret": 0}
        for tf in TIMEFRAMES:
            r = all_results.get(tf, {}).get(market, {})
            lev_res = r.get("leverage", {})
            for sn, rlist in lev_res.items():
                for rv in rlist:
                    sr = rv["performance"].get("sharpe_ratio", -999)
                    ann_ret = rv["performance"].get("annual_return", 0)
                    if ann_ret > 0 and sr > best_overall["sharpe"]:
                        best_overall = {
                            "tf": tf, "strat": sn, "sharpe": sr,
                            "params": rv["params"],
                            "lev": rv["leverage"],
                            "ret": ann_ret,
                            "dd": rv["performance"].get("max_drawdown", 0),
                        }

        b = best_overall
        L.append(f"- **推荐周期**: `{b['tf']}`")
        L.append(f"- **推荐策略**: `{b['strat']}` {b.get('params', '')}")
        L.append(f"- **推荐杠杆**: `{b['lev']:.1f}x`")
        L.append(f"- **Sharpe**: {b['sharpe']:.3f}")
        L.append(f"- **年化收益**: {b.get('ret', 0)*100:+.1f}%")
        L.append(f"- **最大回撤**: {abs(b.get('dd', 0))*100:.1f}%\n")

    L.append("### 关键结论\n")
    L.append("1. **日线 (1d)** 通常是趋势策略的最佳周期 — 信噪比高、交易成本低")
    L.append("2. **4h** 是日线与高频之间的良好折中 — 足够数据量且成本可控")
    L.append("3. **1h** 策略需要精心挑选参数，避免过度交易")
    L.append("4. **5m/15m** 纯趋势策略不盈利 — 需要均值回归或微观结构策略")
    L.append("5. **杠杆选择**: 高频周期应使用低杠杆（1x-1.5x），日线可适当加杠杆\n")

    L.append("### 重要免责声明\n")
    L.append("> 历史回测不代表未来表现。短周期回测的过拟合风险更高。"
             "实盘交易受延迟、流动性、市场冲击等因素影响，实际表现可能低于回测。\n")

    # Appendix
    L.append(f"## {_sec(sec)}、技术附录\n")
    L.append(f"- 回测引擎: Numba JIT kernel ({len(KERNEL_NAMES)} strategies)")
    L.append(f"- 优化方法: Walk-Forward + Combinatorial Purged Cross-Validation")
    L.append(f"- 时间周期: {', '.join(TIMEFRAMES)}")
    L.append(f"- 杠杆探索: {', '.join(f'{l:.1f}x' for l in LEVERAGE_LEVELS)}")
    L.append(f"- 无风险利率: {RISK_FREE_RATE*100:.1f}%")
    L.append(f"- 初始资金: ${INITIAL_CAPITAL:,.0f}")
    for tf in TIMEFRAMES:
        grids = TF_GRIDS.get(tf) or DEFAULT_PARAM_GRIDS
        total = sum(len(v) for v in grids.values())
        L.append(f"- {tf} 参数网格: {total:,} 组合/策略")
    L.append("")
    return "\n".join(L)


# ──────────────────────── main ────────────────────────

def main():
    t_start = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Multi-Timeframe Comprehensive Backtest")
    print("=" * 60)

    all_results = {}
    step = 1
    total_steps = len(TIMEFRAMES) * 3 + 2  # load + opt + lev per TF + charts + report

    for tf in TIMEFRAMES:
        print(f"\n{'='*60}")
        print(f"  [{step}/{total_steps}] Loading {tf} data ...")
        print(f"{'='*60}")
        crypto, stocks = load_tf_data(tf)
        c_bars = sum(len(d) for d in crypto.values())
        s_bars = sum(len(d) for d in stocks.values())
        print(f"  Crypto: {list(crypto.keys())} ({c_bars:,} bars)")
        print(f"  Stocks: {list(stocks.keys())} ({s_bars:,} bars)")
        step += 1

        grids = TF_GRIDS.get(tf)
        cc = BacktestConfig.crypto(leverage=1.0, interval=tf)
        sc = BacktestConfig.stock_ibkr(leverage=1.0, interval=tf)
        print(f"  Config: crypto bpy={int(cc.bars_per_year):,} bpd={int(cc.bars_per_day)}, "
              f"stock bpy={int(sc.bars_per_year):,} bpd={int(sc.bars_per_day)}")

        # Optimize
        print(f"\n  [{step}/{total_steps}] Optimizing {tf} ...")
        step += 1
        c_wf, c_cpcv = run_optimization(crypto, cc, f"Crypto-{tf}", grids)
        s_wf, s_cpcv = run_optimization(stocks, sc, f"Stock-{tf}", grids)

        # Extract best
        def _best_info(wf):
            if not wf:
                return "N/A", 0
            ranked = sorted(wf.all_strategies.items(),
                            key=lambda kv: kv[1].get("wf_score", -1e18), reverse=True)
            if ranked:
                name, d = ranked[0]
                return name, d.get("sharpe", 0)
            return "N/A", 0

        c_bn, c_bs = _best_info(c_wf)
        s_bn, s_bs = _best_info(s_wf)

        # Leverage
        print(f"\n  [{step}/{total_steps}] Leverage exploration {tf} ...")
        step += 1
        c_lev, s_lev = {}, {}
        c_best_detail, s_best_detail = None, None
        if c_wf:
            top_c = get_top(c_wf, TOP_N)
            print(f"    Crypto top-{TOP_N}: {[s[0] for s in top_c]}")
            c_lev = run_leverage(top_c, crypto, BacktestConfig.crypto, tf)
            if top_c:
                c_best_detail = run_detailed(top_c[0][0], top_c[0][1], crypto, cc)
        if s_wf:
            top_s = get_top(s_wf, TOP_N)
            print(f"    Stock top-{TOP_N}: {[s[0] for s in top_s]}")
            s_lev = run_leverage(top_s, stocks, BacktestConfig.stock_ibkr, tf)
            if top_s:
                s_best_detail = run_detailed(top_s[0][0], top_s[0][1], stocks, sc)

        all_results[tf] = {
            "crypto": {"data": crypto, "wf": c_wf, "cpcv": c_cpcv,
                       "leverage": c_lev, "best_name": c_bn, "best_sharpe": c_bs,
                       "best_detail": c_best_detail},
            "stock": {"data": stocks, "wf": s_wf, "cpcv": s_cpcv,
                      "leverage": s_lev, "best_name": s_bn, "best_sharpe": s_bs,
                      "best_detail": s_best_detail},
        }

    # Charts
    print(f"\n  [{step}/{total_steps}] Generating charts ...")
    step += 1
    plot_cross_tf_sharpe(all_results, CHART_DIR / "cross_tf_sharpe.png")
    for market in ["crypto", "stock"]:
        plot_best_equity_per_tf(all_results, market, CHART_DIR / f"best_equity_{market}.png")
    print("  Charts saved.")

    # Report
    total_time = time.perf_counter() - t_start
    print(f"\n  [{step}/{total_steps}] Generating report ...")
    report = generate_report(all_results, total_time)
    rp = REPORT_DIR / "multi_timeframe_analysis_report.md"
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("  MULTI-TIMEFRAME ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Report: {rp}")
    print(f"  Charts: {CHART_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
