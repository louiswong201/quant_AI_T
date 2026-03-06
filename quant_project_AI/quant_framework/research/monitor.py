"""Monitor Engine — daily health trends, regime detection, performance attribution.

Runs daily (~10s) and stores results in the research database.

2a. Multi-Dimension Health  — 5 metrics with 30-day trend comparison
2b. Regime Detection        — soft probability-based regime classification
2c. Performance Attribution — market / strategy / parameter factor decomposition
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..backtest.config import BacktestConfig
from ..backtest.kernels import (
    config_to_kernel_costs,
    eval_kernel_detailed,
    DEFAULT_PARAM_GRIDS,
    scan_all_kernels,
)
from .database import ResearchDB

_log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  2a  Multi-Dimension Health
# ═══════════════════════════════════════════════════════════════

def _rolling_sharpe(equity: np.ndarray, window: int = 30,
                    bars_per_year: int = 252) -> float:
    """Annualised Sharpe from the last *window* bars of the equity curve."""
    if len(equity) < window + 1:
        return 0.0
    tail = equity[-(window + 1):]
    rets = np.diff(tail) / np.maximum(tail[:-1], 1e-12)
    mu = np.mean(rets)
    sig = np.std(rets)
    return float(mu / sig * math.sqrt(bars_per_year)) if sig > 1e-12 else 0.0


def _drawdown_info(equity: np.ndarray) -> Tuple[float, int]:
    """Current drawdown (%) and duration in bars from peak."""
    if len(equity) < 2:
        return 0.0, 0
    peak = np.maximum.accumulate(equity)
    dd_series = (peak - equity) / np.maximum(peak, 1e-12)
    dd_now = float(dd_series[-1])

    # Find the last bar where equity was at its running peak
    last_peak_bar = len(equity) - 1
    for i in range(len(equity) - 1, -1, -1):
        if equity[i] >= peak[i] * (1.0 - 1e-9):
            last_peak_bar = i
            break
    dd_duration = len(equity) - 1 - last_peak_bar
    return dd_now, max(dd_duration, 0)


def _win_rate_ema(equity: np.ndarray, span: int = 20) -> float:
    """Win rate computed via exponential weighting over last *span* returns."""
    if len(equity) < span + 1:
        tail = equity
    else:
        tail = equity[-(span + 1):]
    rets = np.diff(tail)
    if len(rets) == 0:
        return 0.5
    alpha = 2.0 / (len(rets) + 1)
    wr = 0.5
    for r in rets:
        win = 1.0 if r > 0 else 0.0
        wr = alpha * win + (1 - alpha) * wr
    return float(wr)


_BARS_PER_YEAR = {"1d": 252, "4h": 1512, "1h": 6048}
_BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24}


def compute_health_metrics(
    strategy: str,
    params: tuple,
    data: Dict[str, np.ndarray],
    config: BacktestConfig,
    *,
    recent_n: Optional[int] = None,
    sharpe_window: int = 30,
    interval: str = "1d",
) -> Dict[str, Any]:
    """Run strategy and compute 5-dimension health snapshot.

    Returns dict with keys: sharpe_30d, drawdown_pct, dd_duration,
    trade_freq, win_rate, ret_pct, n_trades, status.

    ``interval`` drives the annualisation factor so that Sharpe and trade
    frequency are comparable across timeframes.
    """
    c, o, h, l = data["c"], data["o"], data["h"], data["l"]
    if recent_n and len(c) > recent_n:
        c, o, h, l = c[-recent_n:], o[-recent_n:], h[-recent_n:], l[-recent_n:]

    co = config_to_kernel_costs(config)
    try:
        ret, dd, nt, equity, fpos, _ = eval_kernel_detailed(
            strategy, params, c, o, h, l,
            co["sb"], co["ss"], co["cm"], co["lev"], co["dc"],
            co["sl"], co["pfrac"], co["sl_slip"],
        )
    except Exception as exc:
        _log.warning("Health eval failed for %s: %s", strategy, exc)
        return {"error": str(exc), "status": "ERROR"}

    bars_per_year = _BARS_PER_YEAR.get(interval, 252)
    bars_per_day = _BARS_PER_DAY.get(interval, 1)
    # Scale sharpe window so it always covers ~30 calendar days
    scaled_window = max(sharpe_window * bars_per_day, 10)

    sharpe_30d = _rolling_sharpe(equity, scaled_window, bars_per_year)
    dd_pct, dd_dur = _drawdown_info(equity)
    win_rate = _win_rate_ema(equity, span=max(20 * bars_per_day, 5))
    trade_freq = nt / max(len(c), 1) * bars_per_year

    return {
        "sharpe_30d": round(sharpe_30d, 4),
        "drawdown_pct": round(dd_pct, 4),
        "dd_duration": int(dd_dur),
        "trade_freq": round(trade_freq, 2),
        "win_rate": round(win_rate, 4),
        "ret_pct": round(float(ret), 4),
        "n_trades": int(nt),
    }


def assess_status(
    current: Dict[str, Any],
    history: List[Dict[str, Any]],
    original_sharpe: float = 0.0,
) -> str:
    """Determine HEALTHY / WATCH / ALERT from metrics + trend."""
    if current.get("error"):
        return "ERROR"

    s = current.get("sharpe_30d", 0)

    # Trend: 3+ consecutive declining checks
    if len(history) >= 3:
        recent_sharpes = [h.get("sharpe_30d", 0) for h in history[:3]]
        declining = all(
            recent_sharpes[i] < recent_sharpes[i + 1]
            for i in range(len(recent_sharpes) - 1)
        )
    else:
        declining = False

    # Historical max drawdown
    hist_dds = [h.get("drawdown_pct", 0) for h in history if h.get("drawdown_pct")]
    max_hist_dd = max(hist_dds) if hist_dds else 1.0
    dd_exceeds = current.get("drawdown_pct", 0) > max_hist_dd * 0.8

    # Trade frequency deviation
    hist_freqs = [h.get("trade_freq", 0) for h in history if h.get("trade_freq")]
    if len(hist_freqs) >= 5:
        mu_freq = np.mean(hist_freqs)
        std_freq = np.std(hist_freqs)
        freq_deviation = (
            abs(current.get("trade_freq", mu_freq) - mu_freq) > 2 * std_freq
            if std_freq > 0 else False
        )
    else:
        freq_deviation = False

    n_triggers = sum([declining, dd_exceeds, freq_deviation])
    sharpe_ok = s >= original_sharpe * 0.8 if original_sharpe > 0 else s > 0

    if s <= 0 or n_triggers >= 2:
        return "ALERT"
    if not sharpe_ok or n_triggers >= 1:
        return "WATCH"
    return "HEALTHY"


# ═══════════════════════════════════════════════════════════════
#  2b  Regime Detection
# ═══════════════════════════════════════════════════════════════

def _sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    z = (x - center) / max(scale, 1e-12)
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> float:
    """Simplified ADX (average directional index) for the most recent bar."""
    n = len(c)
    if n < period + 2:
        return 25.0
    plus_dm = np.maximum(np.diff(h[-period - 1:]), 0)
    minus_dm = np.maximum(-np.diff(l[-period - 1:]), 0)
    tr_arr = np.maximum(
        np.abs(np.diff(h[-period - 1:])),
        np.maximum(
            np.abs(h[-period:] - c[-period - 1:-1]),
            np.abs(l[-period:] - c[-period - 1:-1]),
        ),
    )
    tr_sum = np.sum(tr_arr) + 1e-12
    plus_di = np.sum(plus_dm) / tr_sum
    minus_di = np.sum(minus_dm) / tr_sum
    dx = abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-12) * 100
    return float(dx)


def regime_probabilities(
    close: np.ndarray,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    lookback: int = 60,
) -> Dict[str, float]:
    """Soft regime classification via sigmoid scoring.

    Returns {"trending", "mean_reverting", "high_vol", "compression"}
    with values summing to ~1.0.
    """
    n = len(close)
    if n < lookback + 1:
        return {"trending": 0.25, "mean_reverting": 0.25,
                "high_vol": 0.25, "compression": 0.25}

    tail = close[-lookback:]
    rets = np.diff(np.log(np.maximum(tail, 1e-12)))

    # ADX proxy for trend strength
    if high is not None and low is not None:
        adx_val = _adx(high, low, close, period=min(14, lookback // 3))
    else:
        # approximate from close: cumulative absolute returns
        abs_ret = np.abs(rets)
        adx_val = float(np.mean(abs_ret[-14:]) / (np.std(rets) + 1e-12) * 50)

    # ATR ratio = recent vol / historical vol
    recent_vol = float(np.std(rets[-20:])) if len(rets) >= 20 else float(np.std(rets))
    full_vol = float(np.std(rets))
    atr_ratio = recent_vol / max(full_vol, 1e-12)

    # Bollinger bandwidth
    sma = np.mean(tail[-20:])
    std_20 = float(np.std(tail[-20:]))
    bb_width = std_20 / max(sma, 1e-12)

    # Raw scores
    trend_raw = _sigmoid(adx_val, center=25, scale=10)
    mr_raw = _sigmoid(-adx_val, center=-20, scale=10)
    vol_raw = _sigmoid(atr_ratio, center=1.3, scale=0.3)
    comp_raw = _sigmoid(-bb_width, center=-0.02, scale=0.01)

    total = trend_raw + mr_raw + vol_raw + comp_raw + 1e-12
    return {
        "trending": round(trend_raw / total, 4),
        "mean_reverting": round(mr_raw / total, 4),
        "high_vol": round(vol_raw / total, 4),
        "compression": round(comp_raw / total, 4),
    }


# ═══════════════════════════════════════════════════════════════
#  2c  Performance Attribution
# ═══════════════════════════════════════════════════════════════

def _compute_market_return(close: np.ndarray, recent_n: int = 90) -> float:
    """Simple buy-and-hold return over last N bars."""
    if len(close) < recent_n + 1:
        return 0.0
    return float(close[-1] / close[-recent_n] - 1)


def _estimate_beta(strategy_equity: np.ndarray, market_close: np.ndarray) -> float:
    """Estimate beta (strategy vs market) from returns."""
    n = min(len(strategy_equity), len(market_close))
    if n < 10:
        return 1.0
    s_rets = np.diff(strategy_equity[-n:]) / np.maximum(strategy_equity[-n:-1], 1e-12)
    m_rets = np.diff(market_close[-n:]) / np.maximum(market_close[-n:-1], 1e-12)

    cov = np.cov(s_rets, m_rets)
    if cov.shape == (2, 2) and cov[1, 1] > 1e-12:
        return float(cov[0, 1] / cov[1, 1])
    return 1.0


def performance_attribution(
    strategy: str,
    params: tuple,
    data: Dict[str, np.ndarray],
    config: BacktestConfig,
    recent_n: int = 90,
) -> Dict[str, Any]:
    """Decompose recent performance into market / strategy / parameter factors.

    Returns {market_factor, strategy_factor, parameter_factor, explanation}.
    """
    c, o, h, l = data["c"], data["o"], data["h"], data["l"]

    co = config_to_kernel_costs(config)
    try:
        ret, dd, nt, equity, _, _ = eval_kernel_detailed(
            strategy, params, c, o, h, l,
            co["sb"], co["ss"], co["cm"], co["lev"], co["dc"],
            co["sl"], co["pfrac"], co["sl_slip"],
        )
    except Exception:
        return {
            "market_factor": 0, "strategy_factor": 0, "parameter_factor": 0,
            "explanation": "Evaluation failed",
        }

    market_ret = _compute_market_return(c, recent_n)
    beta = _estimate_beta(equity, c)
    market_factor = beta * market_ret

    # Strategy return over recent period
    if len(equity) >= recent_n:
        strategy_ret = float(equity[-1] / max(equity[-recent_n], 1e-12) - 1)
    else:
        strategy_ret = float(ret)
    strategy_factor = strategy_ret - market_factor

    # Parameter factor: quick re-scan a small grid neighborhood and compare
    param_factor = 0.0
    try:
        mini_result = scan_all_kernels(
            c[-recent_n:] if len(c) > recent_n else c,
            o[-recent_n:] if len(o) > recent_n else o,
            h[-recent_n:] if len(h) > recent_n else h,
            l[-recent_n:] if len(l) > recent_n else l,
            config,
            strategies=[strategy],
        )
        if strategy in mini_result:
            best_ret = mini_result[strategy].get("ret", 0)
            if isinstance(best_ret, (int, float)):
                param_factor = best_ret - strategy_ret
    except Exception:
        pass

    parts = []
    if abs(market_factor) > 0.01:
        direction = "up" if market_factor > 0 else "down"
        parts.append(f"market {direction} ({market_factor:+.1%})")
    if abs(strategy_factor) > 0.01:
        quality = "outperforming" if strategy_factor > 0 else "underperforming"
        parts.append(f"strategy {quality} ({strategy_factor:+.1%})")
    if param_factor > 0.02:
        parts.append(f"better params available ({param_factor:+.1%} potential)")

    explanation = "; ".join(parts) if parts else "No significant factors identified"

    return {
        "market_factor": round(market_factor, 4),
        "strategy_factor": round(strategy_factor, 4),
        "parameter_factor": round(param_factor, 4),
        "explanation": explanation,
    }


# ═══════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════

def run_monitor(
    db: ResearchDB,
    live_config: Dict[str, Any],
    tf_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    *,
    recent_days: int = 90,
    attribute: bool = True,
) -> List[Dict[str, Any]]:
    """Run full monitor cycle: health + regime + optional attribution.

    Args:
        db: Research database instance.
        live_config: Parsed live_trading_config.json.
        tf_data: {"1d": {sym: {c,o,h,l}}, "1h": {...}, "4h": {...}}.
        recent_days: Health evaluation window.
        attribute: Whether to run performance attribution (slower).

    Returns list of per-recommendation results.
    """
    results = []
    seen_symbols = set()

    for rec in live_config.get("recommendations", []):
        sym = rec["symbol"]
        if rec["type"] != "single-TF":
            continue

        tf = rec["interval"]
        ds = tf_data.get(tf, {}).get(sym)
        if ds is None:
            continue

        strategy = rec["strategy"]
        params = tuple(rec["params"])
        leverage = rec.get("leverage", 1)
        is_c = sym.upper() in {
            "BTC", "ETH", "BNB", "SOL", "XRP", "ADA",
            "DOGE", "AVAX", "DOT", "MATIC",
        }
        sl = min(0.40, 0.80 / leverage) if leverage > 1 else 0.40
        if is_c:
            config = BacktestConfig.crypto(leverage=leverage, stop_loss_pct=sl, interval=tf)
        else:
            config = BacktestConfig.stock_ibkr(leverage=leverage, stop_loss_pct=sl, interval=tf)

        bars_per_day = {"1d": 1, "4h": 6, "1h": 24}.get(tf, 1)
        recent_n = recent_days * bars_per_day

        # 2a: Health
        metrics = compute_health_metrics(
            strategy, params, ds, config, recent_n=recent_n, interval=tf,
        )
        history = db.get_health_trend(sym, strategy, days=30)
        original_sharpe = rec.get("backtest_metrics", {}).get("sharpe", 0)
        status = assess_status(metrics, history, original_sharpe)
        metrics["status"] = status

        _health_keys = {"sharpe_30d", "drawdown_pct", "dd_duration", "trade_freq",
                        "win_rate", "ret_pct", "n_trades"}
        safe_metrics = {k: v for k, v in metrics.items() if k in _health_keys}
        db.record_health(
            sym, strategy,
            leverage=leverage, interval=tf,
            **safe_metrics,
            status=status,
        )

        # 2b: Regime (once per symbol)
        if sym not in seen_symbols:
            c_arr = ds["c"]
            h_arr = ds.get("h")
            l_arr = ds.get("l")
            regime = regime_probabilities(c_arr, h_arr, l_arr)
            db.record_regime(sym, **regime)
            seen_symbols.add(sym)
        else:
            regime = None

        # 2c: Attribution (on WATCH/ALERT or all if requested)
        attrib = None
        if attribute and status in ("WATCH", "ALERT"):
            attrib = performance_attribution(
                strategy, params, ds, config, recent_n=recent_n,
            )

        entry = {
            "symbol": sym,
            "strategy": strategy,
            "params": list(params),
            "leverage": leverage,
            "interval": tf,
            "type": "single-TF",
            "original_sharpe": original_sharpe,
            "health": metrics,
            "status": status,
            "regime": regime,
            "attribution": attrib,
        }
        results.append(entry)

    _log.info(
        "Monitor: %d strategies evaluated | %s",
        len(results),
        ", ".join(f"{s}:{sum(1 for r in results if r['status']==s)}"
                  for s in ("HEALTHY", "WATCH", "ALERT", "ERROR")
                  if any(r["status"] == s for r in results)),
    )
    return results
