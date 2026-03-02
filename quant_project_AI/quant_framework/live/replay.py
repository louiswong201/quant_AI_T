"""
回测-实盘一致性回放：比较成交方向、数量、价格与时间偏差。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def analyze_execution_divergence(
    backtest_trades: pd.DataFrame,
    live_fills: pd.DataFrame,
    *,
    time_tolerance_seconds: int = 300,
    price_tolerance_bps: float = 20.0,
) -> Dict[str, Any]:
    """
    以 action+symbol+shares 为基础做顺序匹配，评估回测与实盘偏差。
    """
    if backtest_trades is None or backtest_trades.empty:
        return {"matched": 0, "backtest_count": 0, "live_count": 0}
    if live_fills is None or live_fills.empty:
        return {
            "matched": 0,
            "backtest_count": int(len(backtest_trades)),
            "live_count": 0,
            "missing_in_live": int(len(backtest_trades)),
        }
    bt = backtest_trades.copy()
    lv = live_fills.copy()
    bt["date"] = pd.to_datetime(bt["date"])
    lv["date"] = pd.to_datetime(lv["date"])
    bt = bt.sort_values("date").reset_index(drop=True)
    lv = lv.sort_values("date").reset_index(drop=True)

    i, j = 0, 0
    dt_abs = []
    px_bps = []
    matched = 0
    while i < len(bt) and j < len(lv):
        b = bt.iloc[i]
        l = lv.iloc[j]
        same_key = (
            b.get("action") == l.get("action")
            and b.get("symbol") == l.get("symbol")
            and int(b.get("shares", 0)) == int(l.get("shares", 0))
        )
        if not same_key:
            j += 1
            continue
        tdiff = abs((l["date"] - b["date"]).total_seconds())
        if tdiff > time_tolerance_seconds:
            j += 1
            continue
        bp = float(b.get("price", 0.0))
        lp = float(l.get("price", 0.0))
        if bp > 0 and lp > 0:
            bps = abs(lp - bp) / bp * 10000.0
            px_bps.append(bps)
        dt_abs.append(tdiff)
        matched += 1
        i += 1
        j += 1

    mean_delay = float(np.mean(dt_abs)) if dt_abs else 0.0
    p95_delay = float(np.percentile(np.asarray(dt_abs), 95)) if dt_abs else 0.0
    mean_bps = float(np.mean(px_bps)) if px_bps else 0.0
    p95_bps = float(np.percentile(np.asarray(px_bps), 95)) if px_bps else 0.0
    within_price_tol = int(np.sum(np.asarray(px_bps) <= price_tolerance_bps)) if px_bps else 0
    return {
        "matched": matched,
        "backtest_count": int(len(bt)),
        "live_count": int(len(lv)),
        "missing_in_live": int(len(bt) - matched),
        "extra_in_live": int(len(lv) - matched),
        "mean_delay_seconds": mean_delay,
        "p95_delay_seconds": p95_delay,
        "mean_price_diff_bps": mean_bps,
        "p95_price_diff_bps": p95_bps,
        "within_price_tolerance_count": within_price_tol,
    }


def export_execution_divergence_report(
    summary: Dict[str, Any],
    output_path: str,
) -> str:
    """
    将一致性分析结果导出为 Markdown 报告，返回写入文件路径。
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Execution Divergence Report",
        "",
        "## Overview",
        f"- matched: {summary.get('matched', 0)}",
        f"- backtest_count: {summary.get('backtest_count', 0)}",
        f"- live_count: {summary.get('live_count', 0)}",
        f"- missing_in_live: {summary.get('missing_in_live', 0)}",
        f"- extra_in_live: {summary.get('extra_in_live', 0)}",
        "",
        "## Latency",
        f"- mean_delay_seconds: {summary.get('mean_delay_seconds', 0.0):.4f}",
        f"- p95_delay_seconds: {summary.get('p95_delay_seconds', 0.0):.4f}",
        "",
        "## Price Difference",
        f"- mean_price_diff_bps: {summary.get('mean_price_diff_bps', 0.0):.4f}",
        f"- p95_price_diff_bps: {summary.get('p95_price_diff_bps', 0.0):.4f}",
        f"- within_price_tolerance_count: {summary.get('within_price_tolerance_count', 0)}",
        "",
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def export_execution_diagnostics_bundle(
    *,
    summary: Dict[str, Any],
    backtest_trades: pd.DataFrame,
    live_fills: pd.DataFrame,
    output_dir: str,
) -> Dict[str, str]:
    """
    导出执行诊断包：summary markdown + backtest/live 原始 CSV。
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "execution_divergence_report.md"
    bt_path = root / "backtest_trades.csv"
    lv_path = root / "live_fills.csv"
    export_execution_divergence_report(summary, str(report_path))
    (backtest_trades if backtest_trades is not None else pd.DataFrame()).to_csv(bt_path, index=False)
    (live_fills if live_fills is not None else pd.DataFrame()).to_csv(lv_path, index=False)
    return {
        "report_path": str(report_path),
        "backtest_trades_path": str(bt_path),
        "live_fills_path": str(lv_path),
    }

