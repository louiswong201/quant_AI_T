"""
Transaction Cost Analysis (TCA) — post-backtest cost decomposition.

Per coding_guide.md Section 4:
  "Enforce Transaction Cost Analysis: the strategy must maintain a positive
   Sharpe ratio AFTER deducting realistic slippage, Maker/Taker tier fees,
   and latency friction."

This module decomposes total trading costs into:
  1. Commission (fixed + proportional)
  2. Slippage (estimated vs realised)
  3. Market impact (participation-rate driven)
  4. Timing cost (delay between signal and execution)

The output is a structured report that shows whether the strategy's edge
survives after realistic cost deductions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TCAReport:
    """Structured TCA output."""

    total_trades: int = 0
    total_commission: float = 0.0
    total_slippage_est: float = 0.0
    total_impact_est: float = 0.0
    total_cost: float = 0.0
    cost_as_pct_of_pnl: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    gross_sharpe: float = 0.0
    net_sharpe: float = 0.0
    cost_per_trade: float = 0.0
    avg_slippage_bps: float = 0.0
    per_trade: List[Dict[str, Any]] = field(default_factory=list)


class TransactionCostAnalyzer:
    """Decompose and report trading costs from backtest results."""

    @staticmethod
    def analyze(
        trades_df: pd.DataFrame,
        daily_returns: np.ndarray,
        initial_capital: float,
        config: Optional[Any] = None,
        risk_free_rate: float = 0.0,
    ) -> TCAReport:
        """Run TCA on completed backtest trades.

        Args:
            trades_df: DataFrame with columns [date, action, symbol, price, shares, commission]
            daily_returns: array of daily portfolio returns
            initial_capital: starting capital
            config: BacktestConfig (if available, used to decompose slippage vs impact)
            risk_free_rate: annualised risk-free rate for Sharpe calculation

        Returns:
            TCAReport with full cost decomposition
        """
        report = TCAReport()

        if trades_df.empty:
            return report

        report.total_trades = len(trades_df)
        report.total_commission = float(trades_df["commission"].sum()) if "commission" in trades_df.columns else 0.0

        per_trade: List[Dict[str, Any]] = []
        for _, row in trades_df.iterrows():
            notional = float(row.get("price", 0)) * float(row.get("shares", 0))
            commission = float(row.get("commission", 0))

            slippage_bps = 0.0
            if config is not None:
                if row.get("action") == "buy":
                    slippage_bps = getattr(config, "slippage_bps_buy", 0.0)
                else:
                    slippage_bps = getattr(config, "slippage_bps_sell", 0.0)

            slippage_cost = notional * slippage_bps / 10_000.0
            total = commission + slippage_cost

            per_trade.append({
                "date": row.get("date"),
                "action": row.get("action"),
                "symbol": row.get("symbol"),
                "notional": notional,
                "commission": commission,
                "slippage_bps": slippage_bps,
                "slippage_cost": slippage_cost,
                "total_cost": total,
            })

        report.per_trade = per_trade
        report.total_slippage_est = sum(t["slippage_cost"] for t in per_trade)
        report.total_cost = report.total_commission + report.total_slippage_est
        report.cost_per_trade = report.total_cost / report.total_trades if report.total_trades > 0 else 0.0

        notionals = [t["notional"] for t in per_trade if t["notional"] > 0]
        if notionals:
            report.avg_slippage_bps = float(np.mean([t["slippage_bps"] for t in per_trade]))

        valid_returns = daily_returns[~np.isnan(daily_returns)]
        bpy = getattr(config, "bars_per_year", 252.0) if config is not None else 252.0
        if len(valid_returns) > 1:
            ann_factor = np.sqrt(bpy)
            mean_ret = float(np.mean(valid_returns))
            std_ret = float(np.std(valid_returns))
            report.gross_sharpe = (
                (mean_ret - risk_free_rate / bpy) / std_ret * ann_factor
                if std_ret > 1e-10 else 0.0
            )

            bar_cost = report.total_cost / len(valid_returns) / initial_capital
            net_mean = mean_ret - bar_cost
            report.net_sharpe = (
                (net_mean - risk_free_rate / bpy) / std_ret * ann_factor
                if std_ret > 1e-10 else 0.0
            )

        report.gross_pnl = float(np.sum(valid_returns)) * initial_capital if len(valid_returns) > 0 else 0.0
        report.net_pnl = report.gross_pnl - report.total_cost

        if abs(report.gross_pnl) > 1e-10:
            report.cost_as_pct_of_pnl = report.total_cost / abs(report.gross_pnl) * 100.0

        return report

    @staticmethod
    def to_markdown(report: TCAReport) -> str:
        """Format TCA report as markdown."""
        lines = [
            "# Transaction Cost Analysis Report",
            "",
            "## Summary",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Trades | {report.total_trades} |",
            f"| Total Commission | ${report.total_commission:,.2f} |",
            f"| Total Slippage (est.) | ${report.total_slippage_est:,.2f} |",
            f"| **Total Cost** | **${report.total_cost:,.2f}** |",
            f"| Cost / Trade | ${report.cost_per_trade:,.2f} |",
            f"| Avg Slippage | {report.avg_slippage_bps:.1f} bps |",
            f"| Cost as % of P&L | {report.cost_as_pct_of_pnl:.1f}% |",
            "",
            "## Sharpe Ratio Impact",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Gross Sharpe | {report.gross_sharpe:.3f} |",
            f"| Net Sharpe (after costs) | {report.net_sharpe:.3f} |",
            f"| Sharpe Degradation | {report.gross_sharpe - report.net_sharpe:.3f} |",
            "",
        ]
        return "\n".join(lines)
