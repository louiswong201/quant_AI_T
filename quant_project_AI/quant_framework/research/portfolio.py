"""Portfolio Engine — correlation analysis, weight optimization,
and portfolio-level performance metrics.

Runs weekly. Entirely new — the prior system had zero portfolio-level analysis.

4a. Strategy Correlation Matrix   — pairwise return correlations
4b. Portfolio Weight Optimization — inverse-vol + Sharpe tilt + correlation penalty
4c. Portfolio-Level Metrics       — combined Sharpe, diversification ratio, marginal risk
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .database import ResearchDB

_log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  4a  Strategy Correlation Matrix
# ═══════════════════════════════════════════════════════════════

def compute_correlation_matrix(
    db: ResearchDB,
    lookback_days: int = 90,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Build NxN correlation matrix from daily strategy health returns.

    Returns (correlation_df, returns_dict) where returns_dict maps
    "SYMBOL/STRATEGY" → daily return array.
    """
    all_health = db.get_all_latest_health()
    strategy_keys = []
    for h in all_health:
        key = f"{h['symbol']}/{h['strategy']}"
        if key not in strategy_keys:
            strategy_keys.append(key)

    returns_dict: Dict[str, np.ndarray] = {}
    for key in strategy_keys:
        sym, strat = key.split("/", 1)
        trend = db.get_health_trend(sym, strat, days=lookback_days)
        if len(trend) < 5:
            continue
        # Use daily return series from health snapshots
        rets = np.array([h.get("ret_pct", 0) for h in reversed(trend)], dtype=np.float64)
        if len(rets) > 1:
            daily_rets = np.diff(rets) / np.maximum(np.abs(rets[:-1]), 1e-12)
        else:
            daily_rets = rets
        returns_dict[key] = daily_rets

    if len(returns_dict) < 2:
        return pd.DataFrame(), returns_dict

    # Align lengths
    min_len = min(len(v) for v in returns_dict.values())
    if min_len < 3:
        return pd.DataFrame(), returns_dict

    aligned = {k: v[-min_len:] for k, v in returns_dict.items()}
    keys = list(aligned.keys())
    matrix = np.array([aligned[k] for k in keys])

    corr = np.corrcoef(matrix)
    corr_df = pd.DataFrame(corr, index=keys, columns=keys)
    return corr_df, returns_dict


# ═══════════════════════════════════════════════════════════════
#  4b  Portfolio Weight Optimization
# ═══════════════════════════════════════════════════════════════

def optimize_weights(
    returns_dict: Dict[str, np.ndarray],
    corr_df: Optional[pd.DataFrame] = None,
    corr_threshold: float = 0.7,
    corr_penalty: float = 0.5,
    bars_per_year: int = 252,
) -> Dict[str, float]:
    """Inverse-volatility + Sharpe-tilt + correlation-penalty weighting.

    Steps:
    1. Base: w_i = 1 / vol_i (risk parity)
    2. Tilt: w_i *= max(sharpe_i, 0.01) (favour higher Sharpe)
    3. Penalty: for corr(i,j) > threshold, penalise lower-Sharpe by 50%
    4. Normalise to sum = 1.0

    ``bars_per_year`` must match the frequency of return arrays in
    ``returns_dict`` (252 for daily, 1512 for 4h, 6048 for 1h).

    Returns {strategy_key: weight}.
    """
    if not returns_dict:
        return {}

    ann = math.sqrt(bars_per_year)
    sharpe_map: Dict[str, float] = {}
    vol_map: Dict[str, float] = {}

    for key, rets in returns_dict.items():
        if len(rets) < 2:
            sharpe_map[key] = 0.0
            vol_map[key] = 1.0
            continue
        mu = np.mean(rets)
        sig = np.std(rets)
        sharpe_map[key] = float(mu / sig * ann) if sig > 1e-12 else 0.0
        vol_map[key] = max(float(sig * ann), 1e-6)

    # 1. Inverse-vol base
    weights = {k: 1.0 / vol_map[k] for k in returns_dict}

    # 2. Sharpe tilt
    for k in weights:
        weights[k] *= max(sharpe_map.get(k, 0), 0.01)

    # 3. Correlation penalty
    if corr_df is not None and len(corr_df) > 1:
        keys = list(weights.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ki, kj = keys[i], keys[j]
                if ki in corr_df.index and kj in corr_df.columns:
                    c = corr_df.loc[ki, kj]
                    if abs(c) > corr_threshold:
                        worse = ki if sharpe_map.get(ki, 0) < sharpe_map.get(kj, 0) else kj
                        weights[worse] *= corr_penalty

    # 4. Normalise
    total = sum(weights.values())
    if total < 1e-12:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: round(v / total, 4) for k, v in weights.items()}


# ═══════════════════════════════════════════════════════════════
#  4c  Portfolio-Level Metrics
# ═══════════════════════════════════════════════════════════════

def portfolio_metrics(
    returns_dict: Dict[str, np.ndarray],
    weights: Dict[str, float],
    bars_per_year: int = 252,
) -> Dict[str, Any]:
    """Compute portfolio-level performance from weighted strategy returns.

    Returns: portfolio_sharpe, portfolio_sortino, portfolio_max_dd,
    diversification_ratio, marginal_contributions.
    """
    if not returns_dict or not weights:
        return {}

    common_keys = [k for k in weights if k in returns_dict]
    if len(common_keys) < 1:
        return {}

    min_len = min(len(returns_dict[k]) for k in common_keys)
    if min_len < 2:
        return {}

    # Build weighted portfolio return series
    port_rets = np.zeros(min_len, dtype=np.float64)
    component_vols = []
    for k in common_keys:
        w = weights.get(k, 0)
        r = returns_dict[k][-min_len:]
        port_rets += w * r
        component_vols.append(w * float(np.std(r)))

    mu = np.mean(port_rets)
    sig = np.std(port_rets)
    ann = math.sqrt(bars_per_year)

    # Sharpe
    port_sharpe = float(mu / sig * ann) if sig > 1e-12 else 0.0

    # Sortino
    downside = port_rets[port_rets < 0]
    down_std = float(np.std(downside)) if len(downside) > 1 else sig
    port_sortino = float(mu / down_std * ann) if down_std > 1e-12 else 0.0

    # Max drawdown
    equity = np.cumprod(1 + port_rets)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    max_dd = float(np.max(dd))

    # Diversification ratio = weighted avg vol / portfolio vol
    weighted_avg_vol = sum(component_vols)
    port_vol = float(sig * ann)
    div_ratio = weighted_avg_vol / max(port_vol, 1e-12) if port_vol > 1e-12 else 1.0

    # Marginal contribution to risk
    marginal = {}
    if len(common_keys) >= 2:
        for k in common_keys:
            w_orig = weights.get(k, 0)
            if w_orig < 1e-6:
                marginal[k] = 0.0
                continue
            # Remove this strategy, reweight, measure new vol
            sub_rets = np.zeros(min_len, dtype=np.float64)
            sub_total_w = 0
            for k2 in common_keys:
                if k2 != k:
                    sub_rets += weights.get(k2, 0) * returns_dict[k2][-min_len:]
                    sub_total_w += weights.get(k2, 0)
            if sub_total_w > 0:
                sub_rets /= sub_total_w
            sub_vol = float(np.std(sub_rets) * ann)
            marginal[k] = round(port_vol - sub_vol, 4)
    else:
        for k in common_keys:
            marginal[k] = round(port_vol, 4)

    return {
        "portfolio_sharpe": round(port_sharpe, 4),
        "portfolio_sortino": round(port_sortino, 4),
        "portfolio_max_dd": round(max_dd, 4),
        "portfolio_vol": round(port_vol, 4),
        "diversification_ratio": round(div_ratio, 4),
        "n_strategies": len(common_keys),
        "marginal_contributions": marginal,
    }


# ═══════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════

def run_portfolio_analysis(
    db: ResearchDB,
    lookback_days: int = 90,
) -> Dict[str, Any]:
    """Run full portfolio analysis cycle.

    Returns dict with keys: correlation_matrix, weights,
    position_sizes, portfolio_metrics, recommendations.
    """
    corr_df, returns_dict = compute_correlation_matrix(db, lookback_days)

    if not returns_dict:
        _log.warning("Portfolio: insufficient data for analysis (need health history)")
        return {
            "correlation_matrix": None,
            "weights": {},
            "position_sizes": {},
            "portfolio_metrics": {},
            "recommendations": ["Insufficient health history. Run Monitor Engine for a few days first."],
        }

    weights = optimize_weights(returns_dict, corr_df)
    metrics = portfolio_metrics(returns_dict, weights)

    # Generate position size recommendations (assuming 5% total allocation)
    base_pos_pct = 0.05
    position_sizes = {k: round(w * base_pos_pct, 4) for k, w in weights.items()}

    # Generate recommendations
    recs = []
    if metrics.get("diversification_ratio", 1) < 1.2:
        recs.append("Low diversification — consider adding uncorrelated strategies")

    if corr_df is not None and len(corr_df) > 1:
        high_corr_pairs = []
        keys = list(corr_df.index)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                c = corr_df.iloc[i, j]
                if abs(c) > 0.7:
                    high_corr_pairs.append((keys[i], keys[j], round(c, 2)))
        if high_corr_pairs:
            for ki, kj, c in high_corr_pairs[:3]:
                recs.append(f"High correlation ({c}) between {ki} and {kj}")

    # Marginal contribution insights
    mc = metrics.get("marginal_contributions", {})
    if mc:
        best_diversifier = min(mc, key=mc.get)
        recs.append(f"Best diversifier: {best_diversifier} (marginal risk contribution: {mc[best_diversifier]:+.4f})")

    result = {
        "correlation_matrix": corr_df.to_dict() if corr_df is not None and not corr_df.empty else None,
        "weights": weights,
        "position_sizes": position_sizes,
        "portfolio_metrics": metrics,
        "recommendations": recs,
    }

    _log.info(
        "Portfolio: %d strategies | Sharpe=%.2f | DivRatio=%.2f | MaxDD=%.1f%%",
        metrics.get("n_strategies", 0),
        metrics.get("portfolio_sharpe", 0),
        metrics.get("diversification_ratio", 0),
        metrics.get("portfolio_max_dd", 0) * 100,
    )
    return result
