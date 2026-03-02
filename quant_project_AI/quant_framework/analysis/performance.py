"""
Production-grade performance analysis — the SINGLE source of truth for all
backtest and live-trading metrics.

All metrics that were previously scattered across examples/ scripts
(deflated_sharpe, max_drawdown, sharpe_ratio, calc_risk_metrics, etc.) are
now consolidated here. External code should import from this module instead
of reimplementing.

Vectorized numpy throughout — no Python for-loops in the hot path.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Core metric functions (stateless, importable individually)
# -----------------------------------------------------------------------

def total_return(equity: np.ndarray) -> float:
    if len(equity) < 2 or equity[0] == 0:
        return 0.0
    return (equity[-1] / equity[0]) - 1.0


def annual_return(equity: np.ndarray, periods_per_year: float = 252.0) -> float:
    n = len(equity)
    if n < 2 or equity[0] <= 0:
        return 0.0
    years = (n - 1) / periods_per_year
    if years <= 0:
        return 0.0
    tr = equity[-1] / equity[0]
    if tr <= 0:
        return -1.0
    return tr ** (1.0 / years) - 1.0


def annual_volatility(
    daily_returns: np.ndarray,
    periods_per_year: float = 252.0,
) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return float(np.std(daily_returns, ddof=1)) * math.sqrt(periods_per_year)


def sharpe_ratio(
    daily_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    vol = annual_volatility(daily_returns, periods_per_year)
    if vol < 1e-12:
        return 0.0
    ann_ret = float(np.mean(daily_returns)) * periods_per_year
    return (ann_ret - risk_free_rate) / vol


def sortino_ratio(
    daily_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Sortino ratio: penalises only downside volatility."""
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2:
        return 0.0
    down_std = float(np.std(downside, ddof=1)) * math.sqrt(periods_per_year)
    if down_std < 1e-12:
        return 0.0
    ann_ret = float(np.mean(daily_returns)) * periods_per_year
    return (ann_ret - risk_free_rate) / down_std


def max_drawdown(equity: np.ndarray) -> Tuple[float, int, int, int]:
    """Returns (max_dd, peak_idx, trough_idx, recovery_idx).

    max_dd is negative (e.g. -0.25 = -25% drawdown).
    """
    if len(equity) < 2:
        return 0.0, 0, 0, 0
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / np.where(running_max > 0, running_max, 1.0)
    trough_idx = int(np.argmin(dd))
    peak_idx = int(np.argmax(equity[:trough_idx + 1])) if trough_idx > 0 else 0
    max_dd_val = float(dd[trough_idx])

    recovery_idx = len(equity) - 1
    for k in range(trough_idx + 1, len(equity)):
        if equity[k] >= equity[peak_idx]:
            recovery_idx = k
            break

    return max_dd_val, peak_idx, trough_idx, recovery_idx


def max_drawdown_duration(equity: np.ndarray) -> int:
    """Longest underwater period (bars)."""
    if len(equity) < 2:
        return 0
    running_max = np.maximum.accumulate(equity)
    underwater = running_max > equity
    longest = 0
    current = 0
    for uw in underwater:
        if uw:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def calmar_ratio(equity: np.ndarray, periods_per_year: float = 252.0) -> float:
    ann_ret = annual_return(equity, periods_per_year)
    dd, *_ = max_drawdown(equity)
    if abs(dd) < 1e-12:
        return 0.0
    return ann_ret / abs(dd)


def omega_ratio(
    daily_returns: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """Omega ratio: probability-weighted gains / losses relative to threshold."""
    excess = daily_returns - threshold
    gains = float(np.sum(excess[excess > 0]))
    losses = float(np.abs(np.sum(excess[excess < 0])))
    if losses < 1e-12:
        return 0.0
    return gains / losses


def value_at_risk(
    daily_returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Historical VaR at given confidence level (returns a negative number)."""
    if len(daily_returns) < 5:
        return 0.0
    return float(np.percentile(daily_returns, (1.0 - confidence) * 100.0))


def conditional_var(
    daily_returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """CVaR (Expected Shortfall) — average loss beyond VaR."""
    var = value_at_risk(daily_returns, confidence)
    tail = daily_returns[daily_returns <= var]
    if len(tail) == 0:
        return var
    return float(np.mean(tail))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis_excess: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio p-value (Bailey & Lopez de Prado, 2014).

    Tests H0: the observed Sharpe is no better than the expected maximum
    Sharpe from `n_trials` independent strategies with zero true Sharpe.
    Returns p-value in [0, 1]; lower = more significant.
    """
    if n_trials < 1 or n_obs < 2:
        return 1.0
    euler_mascheroni = 0.5772156649
    expected_max_sr = (
        (1.0 - euler_mascheroni) * sp_stats.norm.ppf(1.0 - 1.0 / n_trials)
        + euler_mascheroni * sp_stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )

    se_sr = math.sqrt(
        (1.0 + 0.5 * observed_sharpe ** 2
         - skewness * observed_sharpe
         + ((kurtosis_excess) / 4.0) * observed_sharpe ** 2)
        / max(1, n_obs - 1)
    )
    if se_sr < 1e-15:
        return 1.0
    z = (observed_sharpe - expected_max_sr) / se_sr
    return float(1.0 - sp_stats.norm.cdf(z))


def profit_factor(daily_returns: np.ndarray) -> float:
    gains = float(np.sum(daily_returns[daily_returns > 0]))
    losses = float(np.abs(np.sum(daily_returns[daily_returns < 0])))
    if losses < 1e-12:
        return 0.0
    return gains / losses


def win_rate(daily_returns: np.ndarray) -> float:
    if len(daily_returns) == 0:
        return 0.0
    return float(np.sum(daily_returns > 0)) / len(daily_returns)


def tail_ratio(daily_returns: np.ndarray, quantile: float = 0.05) -> float:
    """Right tail / left tail ratio at given quantile."""
    if len(daily_returns) < 20:
        return 0.0
    right = float(np.percentile(daily_returns, 100 * (1 - quantile)))
    left = abs(float(np.percentile(daily_returns, 100 * quantile)))
    if left < 1e-12:
        return 0.0
    return right / left


# -----------------------------------------------------------------------
# Trade-level analysis
# -----------------------------------------------------------------------

def analyze_trades(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """Pair buy/sell trades and compute round-trip statistics."""
    if trades_df.empty or "action" not in trades_df.columns:
        return _empty_trade_stats()

    buys = trades_df[trades_df["action"] == "buy"].reset_index(drop=True)
    sells = trades_df[trades_df["action"] == "sell"].reset_index(drop=True)
    n_pairs = min(len(buys), len(sells))
    if n_pairs == 0:
        return _empty_trade_stats()

    pnl = np.empty(n_pairs, dtype=np.float64)
    for i in range(n_pairs):
        bp = float(buys.iloc[i]["price"])
        sp_val = float(sells.iloc[i]["price"])
        sh = float(min(buys.iloc[i]["shares"], sells.iloc[i]["shares"]))
        pnl[i] = (sp_val - bp) * sh

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    total_profit = float(np.sum(wins)) if len(wins) else 0.0
    total_loss = float(np.abs(np.sum(losses))) if len(losses) else 0.0

    return {
        "total_trades": n_pairs,
        "win_rate": float(len(wins)) / n_pairs if n_pairs else 0.0,
        "avg_win": float(np.mean(wins)) if len(wins) else 0.0,
        "avg_loss": float(np.mean(losses)) if len(losses) else 0.0,
        "profit_factor": total_profit / total_loss if total_loss > 1e-12 else 0.0,
        "total_profit": total_profit,
        "total_loss": total_loss,
        "avg_pnl": float(np.mean(pnl)),
        "pnl_std": float(np.std(pnl, ddof=1)) if n_pairs > 1 else 0.0,
    }


def _empty_trade_stats() -> Dict[str, Any]:
    return {
        "total_trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "profit_factor": 0.0, "total_profit": 0.0, "total_loss": 0.0,
        "avg_pnl": 0.0, "pnl_std": 0.0,
    }


# -----------------------------------------------------------------------
# PerformanceAnalyzer class (backward-compatible + extended)
# -----------------------------------------------------------------------

class PerformanceAnalyzer:
    """Unified performance analyzer.

    Replaces scattered metric code across examples/. All metrics are
    computed via the stateless functions above; this class is a convenience
    wrapper that bundles them.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.03,
        periods_per_year: float = 252.0,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def analyze(
        self,
        portfolio_values: np.ndarray,
        daily_returns: np.ndarray,
        initial_capital: float,
        n_trials: int = 1,
    ) -> Dict[str, Any]:
        """Full performance analysis — returns a flat dict of all metrics."""
        if len(portfolio_values) == 0:
            return {}

        equity = portfolio_values
        rets = daily_returns
        ppy = self.periods_per_year
        rfr = self.risk_free_rate

        sr = sharpe_ratio(rets, rfr, ppy)
        dd_val, dd_peak, dd_trough, dd_recov = max_drawdown(equity)

        skew = float(sp_stats.skew(rets)) if len(rets) > 2 else 0.0
        kurt = float(sp_stats.kurtosis(rets)) if len(rets) > 3 else 0.0

        return {
            "total_return": total_return(equity),
            "annual_return": annual_return(equity, ppy),
            "volatility": annual_volatility(rets, ppy),
            "sharpe_ratio": sr,
            "sortino_ratio": sortino_ratio(rets, rfr, ppy),
            "calmar_ratio": calmar_ratio(equity, ppy),
            "omega_ratio": omega_ratio(rets),
            "max_drawdown": dd_val,
            "max_drawdown_duration": max_drawdown_duration(equity),
            "max_drawdown_peak_idx": dd_peak,
            "max_drawdown_trough_idx": dd_trough,
            "max_drawdown_recovery_idx": dd_recov,
            "value_at_risk_95": value_at_risk(rets, 0.95),
            "cvar_95": conditional_var(rets, 0.95),
            "value_at_risk_99": value_at_risk(rets, 0.99),
            "cvar_99": conditional_var(rets, 0.99),
            "profit_factor": profit_factor(rets),
            "win_rate": win_rate(rets),
            "tail_ratio": tail_ratio(rets),
            "skewness": skew,
            "kurtosis_excess": kurt,
            "deflated_sharpe_pvalue": deflated_sharpe_ratio(
                sr, n_trials, len(rets), skew, kurt,
            ),
            "final_value": float(equity[-1]),
            "initial_capital": initial_capital,
            "trading_days": len(equity),
        }

    def analyze_trades(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        return analyze_trades(trades_df)

    def full_report(
        self,
        portfolio_values: np.ndarray,
        daily_returns: np.ndarray,
        initial_capital: float,
        trades_df: Optional[pd.DataFrame] = None,
        n_trials: int = 1,
    ) -> Dict[str, Any]:
        """Combined performance + trade analysis."""
        perf = self.analyze(portfolio_values, daily_returns, initial_capital, n_trials)
        if trades_df is not None and not trades_df.empty:
            perf["trade_analysis"] = self.analyze_trades(trades_df)
        return perf

    def print_summary(
        self,
        performance: Dict[str, Any],
        trades_analysis: Optional[Dict[str, Any]] = None,
    ) -> None:
        logger.info("\n" + "=" * 60)
        logger.info("Performance Summary")
        logger.info("=" * 60)

        _keys = [
            ("initial_capital", "Initial Capital", "{:,.2f}"),
            ("final_value", "Final Value", "{:,.2f}"),
            ("total_return", "Total Return", "{:.2%}"),
            ("annual_return", "Annual Return", "{:.2%}"),
            ("volatility", "Volatility (ann.)", "{:.2%}"),
            ("sharpe_ratio", "Sharpe Ratio", "{:.3f}"),
            ("sortino_ratio", "Sortino Ratio", "{:.3f}"),
            ("calmar_ratio", "Calmar Ratio", "{:.3f}"),
            ("omega_ratio", "Omega Ratio", "{:.3f}"),
            ("max_drawdown", "Max Drawdown", "{:.2%}"),
            ("max_drawdown_duration", "Max DD Duration (bars)", "{}"),
            ("value_at_risk_95", "VaR (95%)", "{:.4f}"),
            ("cvar_95", "CVaR (95%)", "{:.4f}"),
            ("profit_factor", "Profit Factor", "{:.3f}"),
            ("win_rate", "Win Rate", "{:.2%}"),
            ("deflated_sharpe_pvalue", "Deflated Sharpe p-value", "{:.4f}"),
            ("trading_days", "Trading Days", "{}"),
        ]
        for key, label, fmt in _keys:
            if key in performance:
                logger.info(f"  {label:.<30s} {fmt.format(performance[key])}")

        if trades_analysis:
            logger.info("-" * 40)
            logger.info("  Trade Analysis:")
            for k, v in trades_analysis.items():
                logger.info(f"    {k}: {v}")
        logger.info("=" * 60)
