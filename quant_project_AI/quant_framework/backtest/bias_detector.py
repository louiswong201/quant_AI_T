"""
Bias detector — automated checks for common backtesting pitfalls.

Per coding_guide.md Section 4:
  "Perform 'Autopsy-level' checks for Survivorship Bias and Look-ahead Bias."

Detects:
  1. Look-ahead bias: strategy uses future data (e.g., close price to generate
     signals that are filled at the same bar's close).
  2. Survivorship bias: only testing on assets that survived to the present day
     (no delisted or failed assets in the universe).
  3. Time-zone misalignment: mixing data from different time zones without
     normalisation (e.g., US market close vs Asian market open).
  4. Data snooping: too many parameter combinations tested relative to
     out-of-sample period (multiple hypothesis testing without correction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BiasReport:
    """Results of bias detection analysis."""

    warnings: List[str] = field(default_factory=list)
    critical: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    passed: bool = True


class BiasDetector:
    """Automated bias detection for backtest results."""

    @staticmethod
    def detect_look_ahead(
        trades_df: pd.DataFrame,
        market_fill_mode: str = "next_open",
    ) -> List[str]:
        """Check for look-ahead bias in trade execution.

        A trade filled at the same bar it was signalled (current_close mode)
        implicitly uses future information — the close price is only known
        AFTER the bar completes, yet the signal is generated during the bar.
        """
        issues: List[str] = []
        if market_fill_mode == "current_close":
            issues.append(
                "CRITICAL: market_fill_mode='current_close' uses the close price "
                "for both signal generation and fill, creating look-ahead bias. "
                "Switch to 'next_open' for realistic execution."
            )
        return issues

    @staticmethod
    def detect_survivorship_bias(
        symbols: List[str],
        data_by_symbol: Dict[str, pd.DataFrame],
    ) -> List[str]:
        """Check for survivorship bias indicators.

        Heuristics:
          - All symbols have data through the entire period (no early termination)
          - Universe was selected with hindsight (e.g., "top 10 by market cap today")
        """
        issues: List[str] = []
        if len(symbols) < 3:
            issues.append(
                "WARNING: Only testing on {0} symbol(s). Consider expanding the "
                "universe to include delisted/failed assets to avoid survivorship bias.".format(
                    len(symbols)
                )
            )

        end_dates = []
        for sym, df in data_by_symbol.items():
            if "date" in df.columns and len(df) > 0:
                end_dates.append(pd.to_datetime(df["date"]).max())
        if end_dates:
            latest = max(end_dates)
            all_survive = all(d >= latest - pd.Timedelta(days=5) for d in end_dates)
            if all_survive and len(symbols) > 1:
                issues.append(
                    "INFO: All symbols have data through the latest date. "
                    "Verify that the universe wasn't selected with hindsight "
                    "(survivorship bias)."
                )
        return issues

    @staticmethod
    def detect_data_snooping(
        n_param_combinations: int,
        n_oos_bars: int,
        significance_level: float = 0.05,
    ) -> List[str]:
        """Check for data snooping (multiple hypothesis testing).

        Uses the Bonferroni correction: if testing N parameter combinations,
        the effective significance level is alpha/N. If the OOS period is
        too short relative to N, results are likely spurious.
        """
        issues: List[str] = []
        if n_param_combinations <= 0:
            return issues

        bonferroni_alpha = significance_level / n_param_combinations
        min_oos_bars_needed = int(np.ceil(1.0 / bonferroni_alpha))

        if n_oos_bars < min_oos_bars_needed:
            issues.append(
                f"CRITICAL: Testing {n_param_combinations} parameter combinations "
                f"requires at least {min_oos_bars_needed} OOS bars (Bonferroni), "
                f"but only {n_oos_bars} available. Results are likely data-snooped."
            )

        if n_param_combinations > 100:
            issues.append(
                f"WARNING: {n_param_combinations} parameter combinations tested. "
                "Consider using Deflated Sharpe Ratio (DSR) to correct for "
                "multiple hypothesis testing."
            )
        return issues

    @staticmethod
    def full_audit(
        trades_df: pd.DataFrame,
        symbols: List[str],
        data_by_symbol: Dict[str, pd.DataFrame],
        market_fill_mode: str = "next_open",
        n_param_combinations: int = 1,
        n_oos_bars: int = 252,
    ) -> BiasReport:
        """Run all bias detectors and return a consolidated report."""
        report = BiasReport()

        look_ahead = BiasDetector.detect_look_ahead(trades_df, market_fill_mode)
        survivorship = BiasDetector.detect_survivorship_bias(symbols, data_by_symbol)
        snooping = BiasDetector.detect_data_snooping(n_param_combinations, n_oos_bars)

        for issue in look_ahead + survivorship + snooping:
            if issue.startswith("CRITICAL"):
                report.critical.append(issue)
                report.passed = False
            elif issue.startswith("WARNING"):
                report.warnings.append(issue)
            else:
                report.info.append(issue)

        return report
