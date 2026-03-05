"""Optimize Engine — Bayesian parameter updates, composite gate scoring,
and champion/challenger protocol.

Runs weekly (or triggered by an ALERT from the Monitor Engine).

3a. Bayesian Parameter Update   — blend historical priors with new evidence
3b. Composite Gate Score        — weighted 6-dimension robustness score
3c. Champion/Challenger Protocol — shadow testing before promotion
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..backtest.config import BacktestConfig
from ..backtest.kernels import (
    DEFAULT_PARAM_GRIDS,
    config_to_kernel_costs,
    eval_kernel_detailed,
    scan_all_kernels,
)
from .database import ResearchDB

_log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  3a  Bayesian Parameter Update
# ═══════════════════════════════════════════════════════════════

def bayesian_param_update(
    param_history: List[Dict[str, Any]],
    new_scan_params: list,
    new_scan_sharpe: float,
    *,
    decay: float = 0.9,
    sharpe_weight_scale: float = 1.0,
) -> list:
    """Blend historical parameter priors with new evidence.

    Uses exponential decay to weight older observations less, with
    Sharpe-weighted tilting toward better-performing parameter sets.

    Integer params are rounded, float params are kept.
    """
    if not param_history:
        return list(new_scan_params)

    n_params = len(new_scan_params)
    weighted_sum = np.zeros(n_params, dtype=np.float64)
    weight_total = 0.0

    for i, entry in enumerate(param_history):
        p = entry.get("params")
        if not p or len(p) != n_params:
            continue
        age_weight = decay ** i
        sharpe = max(entry.get("sharpe", 0), 0) * sharpe_weight_scale
        w = age_weight * (1.0 + sharpe)
        weighted_sum += np.array(p, dtype=np.float64) * w
        weight_total += w

    new_w = 1.0 + new_scan_sharpe * sharpe_weight_scale
    weighted_sum += np.array(new_scan_params, dtype=np.float64) * new_w
    weight_total += new_w

    if weight_total < 1e-12:
        return list(new_scan_params)

    blended = weighted_sum / weight_total

    result = []
    for i, (b, orig) in enumerate(zip(blended, new_scan_params)):
        if isinstance(orig, int) or (isinstance(orig, float) and orig == int(orig)):
            result.append(int(round(b)))
        else:
            result.append(round(float(b), 6))
    return result


# ═══════════════════════════════════════════════════════════════
#  3b  Composite Gate Score
# ═══════════════════════════════════════════════════════════════

GATE_WEIGHTS = {
    "statistical":   0.25,  # Sharpe + DSR
    "walkforward":   0.20,  # Walk-forward consistency
    "neighborhood":  0.20,  # Parameter stability
    "montecarlo":    0.15,  # MC survival rate
    "deflated":      0.10,  # Deflated Sharpe correction
    "tail_risk":     0.10,  # CVaR / worst drawdown
}

def _score_statistical(sharpe: float, dsr_p: float) -> float:
    """Score from Sharpe ratio and Deflated Sharpe p-value."""
    sharpe_score = min(sharpe / 3.0, 1.0) if sharpe > 0 else 0.0
    dsr_score = 1.0 - min(dsr_p, 1.0)
    return 0.6 * sharpe_score + 0.4 * dsr_score


def _score_walkforward(wf_score: float, oos_ret: float) -> float:
    """Score from walk-forward validation."""
    wf_norm = min(max(wf_score / 100, 0), 1.0)
    oos_score = min(max(oos_ret / 0.5, 0), 1.0)
    return 0.7 * wf_norm + 0.3 * oos_score


def _score_montecarlo(mc_pct_positive: float) -> float:
    """Score from Monte Carlo survival rate."""
    return min(mc_pct_positive / 100, 1.0)


def _score_deflated(dsr_p: float) -> float:
    """Additional deflated Sharpe penalty (separate from statistical)."""
    return max(0, 1.0 - 2 * dsr_p)


def _score_tail_risk(max_dd: float, cvar_95: float = 0.0) -> float:
    """Score from tail risk measures (lower DD → higher score)."""
    dd_score = max(0, 1.0 - abs(max_dd) / 0.5)
    cvar_score = max(0, 1.0 - abs(cvar_95) / 0.1) if cvar_95 is not None else 0.5
    return 0.6 * dd_score + 0.4 * cvar_score


def param_neighborhood_stability(
    strategy: str,
    params: list,
    data: Dict[str, np.ndarray],
    config: BacktestConfig,
    perturb_pct: float = 0.10,
    sharpe_drop_threshold: float = 0.30,
    bars_per_year: int = 252,
) -> float:
    """Test parameter sensitivity by perturbing each param by +/-perturb_pct.

    Returns stability score [0, 1] where 1 = all neighbors stable.
    """
    c, o, h, l_arr = data["c"], data["o"], data["h"], data["l"]
    co = config_to_kernel_costs(config)
    ann = math.sqrt(bars_per_year)

    def _sharpe(p):
        try:
            ret, dd, nt, eq, _, _ = eval_kernel_detailed(
                strategy, tuple(p), c, o, h, l_arr,
                co["sb"], co["ss"], co["cm"], co["lev"], co["dc"],
                co["sl"], co["pfrac"], co["sl_slip"],
            )
            if eq is not None and len(eq) > 1:
                rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
                mu, sig = np.mean(rets), np.std(rets)
                return mu / sig * ann if sig > 1e-12 else 0.0
        except Exception:
            pass
        return 0.0

    base_sharpe = _sharpe(params)
    if base_sharpe <= 0:
        return 0.0

    stable_count = 0
    total_tests = 0

    for i, p in enumerate(params):
        if isinstance(p, (int, float)) and p != 0:
            for direction in [1 + perturb_pct, 1 - perturb_pct]:
                test_params = list(params)
                new_val = p * direction
                test_params[i] = int(round(new_val)) if isinstance(p, int) else round(new_val, 6)
                if test_params[i] == params[i]:
                    continue
                total_tests += 1
                neighbor_sharpe = _sharpe(test_params)
                drop = (base_sharpe - neighbor_sharpe) / max(abs(base_sharpe), 1e-12)
                if drop < sharpe_drop_threshold:
                    stable_count += 1

    return stable_count / max(total_tests, 1)


def composite_gate_score(
    metrics: Dict[str, Any],
    neighborhood_score: Optional[float] = None,
) -> float:
    """Compute weighted 6-dimension gate score [0, 1].

    ``metrics`` should contain keys from robust_scan output:
    sharpe, dsr_p, wf_score, oos_ret, mc_pct_positive, oos_dd.
    """
    scores = {
        "statistical": _score_statistical(
            metrics.get("sharpe", 0), metrics.get("dsr_p", 1),
        ),
        "walkforward": _score_walkforward(
            metrics.get("wf_score", 0), metrics.get("oos_ret", 0),
        ),
        "neighborhood": neighborhood_score if neighborhood_score is not None else 0.5,
        "montecarlo": _score_montecarlo(metrics.get("mc_pct_positive", 0)),
        "deflated": _score_deflated(metrics.get("dsr_p", 1)),
        "tail_risk": _score_tail_risk(
            metrics.get("oos_dd", 0), metrics.get("cvar_95", 0),
        ),
    }

    total = sum(GATE_WEIGHTS[k] * scores[k] for k in GATE_WEIGHTS)
    return round(total, 4)


# ═══════════════════════════════════════════════════════════════
#  3c  Champion / Challenger Protocol
# ═══════════════════════════════════════════════════════════════

def register_challenger(
    db: ResearchDB,
    symbol: str,
    strategy: str,
    new_params: list,
    gate_score: float,
    *,
    sharpe: float = 0.0,
    wf_score: float = 0.0,
    leverage: float = 1.0,
    interval: str = "1d",
    reason: str = "optimizer",
):
    """Record a challenger strategy in param_history + strategy_library."""
    db.record_param_update(
        symbol, strategy, new_params,
        sharpe=sharpe, wf_score=wf_score, gate_score=gate_score,
        leverage=leverage, interval=interval,
        source="challenger", reason=reason,
    )
    db.upsert_strategy(
        name=f"{symbol}_{strategy}_challenger",
        kernel_name=strategy,
        status="PAPER",
        source="optimizer",
        symbols=[symbol],
        best_params=new_params,
        gate_score=gate_score,
    )
    _log.info(
        "Challenger registered: %s/%s gate=%.3f params=%s",
        symbol, strategy, gate_score, new_params,
    )


def evaluate_promotion(
    db: ResearchDB,
    symbol: str,
    strategy: str,
    current_gate: float,
    promotion_threshold: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Check if any challenger for this symbol/strategy should be promoted.

    Returns promotion dict or None.
    """
    challengers = db.get_strategies_by_status("PAPER")
    for ch in challengers:
        if ch["kernel_name"] != strategy:
            continue
        if ch.get("symbols") and symbol not in ch["symbols"]:
            continue
        ch_gate = ch.get("gate_score", 0)
        if ch_gate >= current_gate + promotion_threshold:
            return {
                "action": "PROMOTE",
                "symbol": symbol,
                "strategy": strategy,
                "challenger_name": ch["name"],
                "challenger_params": ch.get("best_params"),
                "challenger_gate": ch_gate,
                "champion_gate": current_gate,
                "delta": round(ch_gate - current_gate, 4),
            }
    return None


def promote_challenger(
    db: ResearchDB,
    promotion: Dict[str, Any],
    live_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Promote a challenger to champion: update library status + config version."""
    name = promotion["challenger_name"]
    strategy = promotion["strategy"]
    db.update_strategy_status(name, strategy, "LIVE")

    diff = {
        "symbol": promotion["symbol"],
        "strategy": strategy,
        "old_gate": promotion["champion_gate"],
        "new_gate": promotion["challenger_gate"],
        "new_params": promotion["challenger_params"],
    }
    db.save_config_version(
        live_config,
        summary=f"Promoted {name} (gate {promotion['champion_gate']:.3f} → {promotion['challenger_gate']:.3f})",
        diff=diff,
        n_changes=1,
    )
    _log.info("Promoted %s: gate %.3f → %.3f", name, promotion["champion_gate"], promotion["challenger_gate"])
    return diff


# ═══════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════

def run_optimizer(
    db: ResearchDB,
    live_config: Dict[str, Any],
    tf_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    rescan_results: List[Dict[str, Any]],
    *,
    check_neighborhood: bool = True,
    register_challengers: bool = True,
) -> List[Dict[str, Any]]:
    """Run full optimize cycle for all symbols with new scan results.

    Args:
        db: Research database.
        live_config: Current live config.
        tf_data: Timeframe-keyed data.
        rescan_results: Output from phase2_rescan or run_robust_scan.
        check_neighborhood: Run param stability test (slower).
        register_challengers: Create challenger entries in DB.

    Returns list of optimization results per symbol.
    """
    best_by_sym: Dict[str, Dict] = {}
    for r in sorted(rescan_results, key=lambda x: x.get("wf_score", 0), reverse=True):
        sym = r["symbol"]
        if sym not in best_by_sym and r.get("params") and r.get("sharpe", 0) > 0:
            best_by_sym[sym] = r

    current_by_sym: Dict[str, Dict] = {}
    for rec in live_config.get("recommendations", []):
        if rec["type"] == "single-TF" and rec["symbol"] not in current_by_sym:
            current_by_sym[rec["symbol"]] = rec

    results = []
    for sym, new in best_by_sym.items():
        strategy = new["strategy"]
        params = new["params"]
        leverage = new.get("leverage", 1)
        interval = new.get("interval", "1d")

        # Bayesian blend with history
        history = db.get_param_history(sym, strategy, limit=10)
        blended = bayesian_param_update(
            history, params, new.get("sharpe", 0),
        )

        # Gate score
        neighborhood = None
        ds = tf_data.get(interval, {}).get(sym)
        if check_neighborhood and ds is not None:
            is_c = sym.upper() in {
                "BTC", "ETH", "BNB", "SOL", "XRP", "ADA",
                "DOGE", "AVAX", "DOT", "MATIC",
            }
            sl = min(0.40, 0.80 / leverage) if leverage > 1 else 0.40
            if is_c:
                cfg = BacktestConfig.crypto(leverage=leverage, stop_loss_pct=sl, interval=interval)
            else:
                cfg = BacktestConfig.stock_ibkr(leverage=leverage, stop_loss_pct=sl, interval=interval)
            bpy = {"1d": 252, "4h": 1512, "1h": 6048}.get(interval, 252)
            neighborhood = param_neighborhood_stability(
                strategy, blended, ds, cfg, bars_per_year=bpy,
            )

        gate = composite_gate_score(new, neighborhood)

        # Record in DB
        db.record_param_update(
            sym, strategy, blended,
            sharpe=new.get("sharpe"),
            wf_score=new.get("wf_score"),
            gate_score=gate,
            leverage=leverage,
            interval=interval,
            source="optimizer",
            reason="weekly_rescan",
        )

        # Challenger logic
        cur = current_by_sym.get(sym)
        promotion = None
        if cur and register_challengers and gate >= 0.6:
            cur_metrics = cur.get("backtest_metrics", {})
            cur_gate_approx = composite_gate_score(cur_metrics)
            if gate > cur_gate_approx:
                register_challenger(
                    db, sym, strategy, blended, gate,
                    sharpe=new.get("sharpe", 0),
                    wf_score=new.get("wf_score", 0),
                    leverage=leverage, interval=interval,
                )
                promotion = evaluate_promotion(db, sym, strategy, cur_gate_approx)

        results.append({
            "symbol": sym,
            "strategy": strategy,
            "original_params": params,
            "blended_params": blended,
            "gate_score": gate,
            "neighborhood_score": neighborhood,
            "sharpe": new.get("sharpe", 0),
            "wf_score": new.get("wf_score", 0),
            "promotion": promotion,
        })

    _log.info(
        "Optimizer: %d symbols processed, %d with gate >= 0.6",
        len(results), sum(1 for r in results if r["gate_score"] >= 0.6),
    )
    return results
