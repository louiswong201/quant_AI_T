"""Discover Engine — internal variant mining, market anomaly scanning,
and external research monitoring.

Runs weekly/monthly to uncover new strategy opportunities.

5a. Internal Variant Mining  — test unexplored parameter/TF combinations
5b. Market Anomaly Scanner   — detect structural changes in price behaviour
5c. External Research Monitor — scan arXiv for new quantitative papers
"""

from __future__ import annotations

import logging
import math
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..backtest.config import BacktestConfig
from ..backtest.kernels import DEFAULT_PARAM_GRIDS, KERNEL_NAMES
from .database import ResearchDB

_log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  5a  Internal Variant Mining
# ═══════════════════════════════════════════════════════════════

def _find_grid_gaps(
    default_grids: Dict[str, list],
    expanded_grids: Optional[Dict[str, list]] = None,
) -> Dict[str, list]:
    """Identify parameter combinations in EXPANDED but not in DEFAULT.

    If expanded_grids is None, generate a small set of interpolated
    combinations from the default grid.
    """
    gap_grids: Dict[str, list] = {}

    if expanded_grids is not None:
        for strat, expanded in expanded_grids.items():
            default = set(map(tuple, default_grids.get(strat, [])))
            gaps = [p for p in expanded if tuple(p) not in default]
            if gaps:
                gap_grids[strat] = gaps[:500]
        return gap_grids

    # Generate interpolated gap combos from default
    for strat, combos in default_grids.items():
        if len(combos) < 10:
            continue
        arr = np.array(combos, dtype=np.float64)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)

        new_combos = []
        rng = np.random.default_rng(42)
        for _ in range(min(100, len(combos) // 2)):
            candidate = means + stds * rng.standard_normal(len(means)) * 0.5
            # Round integers
            rounded = []
            for i, v in enumerate(candidate):
                if all(isinstance(c[i], int) for c in combos[:5]):
                    rounded.append(max(2, int(round(v))))
                else:
                    rounded.append(round(float(v), 4))
            t = tuple(rounded)
            if t not in set(map(tuple, combos)):
                new_combos.append(t)

        if new_combos:
            gap_grids[strat] = new_combos

    return gap_grids


def discover_variants(
    data: Dict[str, Dict[str, np.ndarray]],
    symbols: List[str],
    config_maker,
    *,
    expanded_grids: Optional[Dict[str, list]] = None,
    max_combos_per_strategy: int = 200,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Mine unexplored parameter combinations from grid gaps.

    Args:
        data: {symbol: {c, o, h, l}} arrays.
        symbols: Symbols to test.
        config_maker: Callable(sym, leverage, interval) → BacktestConfig.
        expanded_grids: Full expanded grids (if available).
        **kwargs: leverage and interval overrides (default 1.0 / "1d").
        max_combos_per_strategy: Cap per strategy to limit runtime.

    Returns list of discovery results sorted by Sharpe.
    """
    from ..backtest.robust_scan import run_robust_scan

    gap_grids = _find_grid_gaps(DEFAULT_PARAM_GRIDS, expanded_grids)
    if not gap_grids:
        _log.info("Discover: no grid gaps found")
        return []

    capped = {s: g[:max_combos_per_strategy] for s, g in gap_grids.items()}
    total = sum(len(v) for v in capped.values())
    _log.info("Discover: mining %d gap combos across %d strategies", total, len(capped))

    discoveries = []
    for sym in symbols:
        if sym not in data:
            continue
        config = config_maker(sym, kwargs.get("leverage", 1.0), kwargs.get("interval", "1d"))
        try:
            result = run_robust_scan(
                symbols=[sym],
                data={sym: data[sym]},
                config=config,
                param_grids=capped,
                n_mc_paths=5,
                n_shuffle_paths=3,
                n_bootstrap_paths=3,
                parallel_symbols=None,
            )
            for sn, metrics in result.per_symbol.get(sym, {}).items():
                if metrics.get("sharpe", 0) > 0.5 and metrics.get("params"):
                    discoveries.append({
                        "symbol": sym,
                        "strategy": sn,
                        "params": metrics["params"],
                        "sharpe": metrics.get("sharpe", 0),
                        "wf_score": metrics.get("wf_score", 0),
                        "oos_ret": metrics.get("oos_ret", 0),
                        "source": "variant_mining",
                    })
        except Exception as exc:
            _log.warning("Variant mining failed for %s: %s", sym, exc)

    discoveries.sort(key=lambda x: x["sharpe"], reverse=True)
    return discoveries


# ═══════════════════════════════════════════════════════════════
#  5b  Market Anomaly Scanner
# ═══════════════════════════════════════════════════════════════

def _autocorrelation(rets: np.ndarray, lag: int = 1) -> float:
    """Lag-1 autocorrelation of return series."""
    if len(rets) < lag + 2:
        return 0.0
    r1 = rets[:-lag]
    r2 = rets[lag:]
    mu = np.mean(rets)
    cov = np.mean((r1 - mu) * (r2 - mu))
    var = np.var(rets)
    return float(cov / var) if var > 1e-12 else 0.0


def _rolling_vol(rets: np.ndarray, window: int = 20) -> Tuple[float, float]:
    """Recent vs historical volatility (annualised)."""
    ann = math.sqrt(252)
    if len(rets) < window + 10:
        v = float(np.std(rets) * ann)
        return v, v
    recent = float(np.std(rets[-window:]) * ann)
    historical = float(np.std(rets[:-window]) * ann)
    return recent, historical


def scan_anomalies(
    data: Dict[str, Dict[str, np.ndarray]],
    symbols: Optional[List[str]] = None,
    lookback: int = 252,
    half_lookback: int = 60,
) -> List[Dict[str, Any]]:
    """Flag symbols where statistical behaviour changed significantly.

    Checks:
    1. Autocorrelation structure shift (momentum ↔ mean-reversion)
    2. Volatility regime change (GARCH-like)
    3. Cross-asset correlation drift

    Returns anomalies sorted by severity.
    """
    targets = symbols or list(data.keys())
    anomalies = []

    all_rets = {}
    for sym in targets:
        ds = data.get(sym)
        if ds is None:
            continue
        c = ds["c"]
        if len(c) < lookback:
            continue
        rets = np.diff(np.log(np.maximum(c[-lookback:], 1e-12)))
        all_rets[sym] = rets

    for sym, rets in all_rets.items():
        flags = []
        severity = 0.0

        # 1. Autocorrelation shift
        ac_full = _autocorrelation(rets)
        ac_recent = _autocorrelation(rets[-half_lookback:])
        ac_delta = abs(ac_recent - ac_full)
        if ac_delta > 0.15:
            direction = "momentum→MR" if ac_recent < ac_full else "MR→momentum"
            flags.append(f"AC shift {direction} (Δ={ac_delta:.2f})")
            severity += ac_delta * 2

        # 2. Volatility regime change
        recent_vol, hist_vol = _rolling_vol(rets)
        vol_ratio = recent_vol / max(hist_vol, 1e-12)
        if vol_ratio > 1.5 or vol_ratio < 0.6:
            direction = "expanding" if vol_ratio > 1 else "contracting"
            flags.append(f"Vol {direction} ({vol_ratio:.1f}x)")
            severity += abs(vol_ratio - 1) * 1.5

        # 3. Distribution change (skew/kurtosis)
        full_skew = float(_safe_skew(rets))
        recent_skew = float(_safe_skew(rets[-half_lookback:]))
        skew_delta = abs(recent_skew - full_skew)
        if skew_delta > 0.5:
            flags.append(f"Skew shift ({full_skew:.2f}→{recent_skew:.2f})")
            severity += skew_delta

        if flags:
            anomalies.append({
                "symbol": sym,
                "severity": round(severity, 3),
                "flags": flags,
                "autocorr_full": round(ac_full, 4),
                "autocorr_recent": round(ac_recent, 4),
                "vol_ratio": round(vol_ratio, 3),
                "recommendation": _anomaly_recommendation(flags, sym),
            })

    anomalies.sort(key=lambda x: x["severity"], reverse=True)
    return anomalies


def _safe_skew(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    mu = np.mean(arr)
    sig = np.std(arr)
    if sig < 1e-12:
        return 0.0
    return float(np.mean(((arr - mu) / sig) ** 3))


def _anomaly_recommendation(flags: List[str], sym: str) -> str:
    parts = []
    for f in flags:
        if "AC shift" in f:
            parts.append("re-evaluate strategy type selection")
        if "Vol" in f and "expanding" in f:
            parts.append("consider reducing position size or leverage")
        if "Vol" in f and "contracting" in f:
            parts.append("mean-reversion strategies may outperform")
        if "Skew" in f:
            parts.append("check tail risk and stop-loss adequacy")
    return f"{sym}: " + "; ".join(parts) if parts else f"{sym}: review strategy parameters"


# Cross-asset correlation drift
def scan_cross_asset_correlation(
    data: Dict[str, Dict[str, np.ndarray]],
    lookback: int = 252,
    recent_window: int = 60,
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Detect pairs where correlation changed significantly."""
    symbols = list(data.keys())
    if len(symbols) < 2:
        return []

    rets = {}
    for sym in symbols:
        c = data[sym]["c"]
        if len(c) < lookback:
            continue
        rets[sym] = np.diff(np.log(np.maximum(c[-lookback:], 1e-12)))

    min_len = min(len(v) for v in rets.values()) if rets else 0
    if min_len < recent_window + 10:
        return []

    shifts = []
    sym_list = list(rets.keys())
    for i in range(len(sym_list)):
        for j in range(i + 1, len(sym_list)):
            a, b = rets[sym_list[i]][-min_len:], rets[sym_list[j]][-min_len:]

            full_corr = float(np.corrcoef(a, b)[0, 1])
            recent_corr = float(np.corrcoef(a[-recent_window:], b[-recent_window:])[0, 1])
            delta = abs(recent_corr - full_corr)

            if delta > threshold:
                shifts.append({
                    "pair": f"{sym_list[i]}/{sym_list[j]}",
                    "full_corr": round(full_corr, 3),
                    "recent_corr": round(recent_corr, 3),
                    "delta": round(delta, 3),
                    "direction": "decoupling" if recent_corr < full_corr else "coupling",
                })

    shifts.sort(key=lambda x: x["delta"], reverse=True)
    return shifts


# ═══════════════════════════════════════════════════════════════
#  5c  External Research Monitor
# ═══════════════════════════════════════════════════════════════

ARXIV_KEYWORDS = [
    "trading strategy",
    "momentum trading",
    "mean reversion",
    "alpha generation",
    "machine learning trading",
    "quantitative finance",
]


def _fetch_arxiv(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Fetch recent papers from arXiv API."""
    base_url = "http://export.arxiv.org/api/query"
    encoded = urllib.request.quote(query)
    url = f"{base_url}?search_query=all:{encoded}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml_data = resp.read().decode("utf-8")
    except Exception as exc:
        _log.warning("arXiv fetch failed: %s", exc)
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    papers = []

    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        published = entry.find("atom:published", ns)
        link = entry.find("atom:id", ns)

        papers.append({
            "title": title.text.strip().replace("\n", " ") if title is not None else "",
            "abstract": summary.text.strip()[:500] if summary is not None else "",
            "published": published.text[:10] if published is not None else "",
            "url": link.text.strip() if link is not None else "",
        })

    return papers


def _score_applicability(title: str, abstract: str) -> int:
    """Rate paper applicability 1–5 based on keyword density."""
    text = (title + " " + abstract).lower()
    high_value = ["backtest", "profit", "sharpe", "alpha", "return",
                   "momentum", "mean reversion", "trading signal", "portfolio"]
    medium_value = ["machine learning", "deep learning", "neural",
                    "reinforcement", "prediction", "forecast"]

    score = 1
    for kw in high_value:
        if kw in text:
            score += 1
    for kw in medium_value:
        if kw in text:
            score += 0.5
    return min(int(score), 5)


def _estimate_complexity(abstract: str) -> str:
    """Estimate implementation complexity from abstract."""
    text = abstract.lower()
    complex_indicators = ["transformer", "attention", "reinforcement learning",
                          "neural network", "deep learning", "gan", "lstm"]
    medium_indicators = ["regression", "feature", "ensemble", "random forest",
                         "gradient boosting", "bayesian"]
    simple_indicators = ["moving average", "momentum", "rsi", "macd",
                         "bollinger", "breakout", "trend", "mean reversion"]

    for kw in simple_indicators:
        if kw in text:
            return "LOW"
    for kw in complex_indicators:
        if kw in text:
            return "HIGH"
    for kw in medium_indicators:
        if kw in text:
            return "MEDIUM"
    return "MEDIUM"


def scan_external_research(
    max_results_per_query: int = 5,
    since_days: int = 30,
) -> List[Dict[str, Any]]:
    """Scan arXiv for recent quantitative finance papers.

    Returns list of papers with applicability score and complexity estimate.
    """
    cutoff = (datetime.now() - timedelta(days=since_days)).strftime("%Y-%m-%d")
    all_papers: Dict[str, Dict] = {}

    for kw in ARXIV_KEYWORDS:
        papers = _fetch_arxiv(kw, max_results=max_results_per_query)
        for p in papers:
            if p["published"] >= cutoff and p["url"] not in all_papers:
                score = _score_applicability(p["title"], p["abstract"])
                complexity = _estimate_complexity(p["abstract"])
                all_papers[p["url"]] = {
                    **p,
                    "applicability": score,
                    "complexity": complexity,
                    "keyword": kw,
                }

    results = sorted(all_papers.values(), key=lambda x: x["applicability"], reverse=True)
    _log.info("External research: found %d papers (since %s)", len(results), cutoff)
    return results


# ═══════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════

def run_discover(
    db: ResearchDB,
    data: Dict[str, Dict[str, np.ndarray]],
    symbols: List[str],
    config_maker,
    *,
    do_variants: bool = True,
    do_anomalies: bool = True,
    do_external: bool = False,
    expanded_grids: Optional[Dict[str, list]] = None,
) -> Dict[str, Any]:
    """Run full discovery cycle.

    Args:
        db: Research database.
        data: {symbol: {c, o, h, l}}.
        symbols: Target symbols.
        config_maker: Callable(sym, leverage, interval) → BacktestConfig.
        do_variants: Run internal variant mining (slower).
        do_anomalies: Run market anomaly scanner.
        do_external: Run arXiv paper scan.
        expanded_grids: Optional expanded parameter grids.

    Returns dict with discovery results.
    """
    result: Dict[str, Any] = {
        "variants": [],
        "anomalies": [],
        "correlation_shifts": [],
        "external_papers": [],
    }

    # 5a: Variant mining
    if do_variants:
        _log.info("Discover: starting variant mining for %d symbols", len(symbols))
        result["variants"] = discover_variants(
            data, symbols, config_maker,
            expanded_grids=expanded_grids,
        )
        for v in result["variants"][:5]:
            db.upsert_strategy(
                name=f"{v['symbol']}_{v['strategy']}_variant",
                kernel_name=v["strategy"],
                status="IDEA",
                source="variant_mining",
                symbols=[v["symbol"]],
                best_params=v["params"],
                gate_score=None,
            )

    # 5b: Anomaly scanning
    if do_anomalies:
        _log.info("Discover: scanning anomalies for %d symbols", len(symbols))
        result["anomalies"] = scan_anomalies(data, symbols)
        result["correlation_shifts"] = scan_cross_asset_correlation(data)

    # 5c: External research
    if do_external:
        _log.info("Discover: scanning arXiv for recent papers")
        result["external_papers"] = scan_external_research()
        for p in result["external_papers"]:
            if p["applicability"] >= 3:
                db.upsert_strategy(
                    name=p["title"][:80],
                    kernel_name="external",
                    status="IDEA",
                    source="arxiv",
                    source_url=p["url"],
                    notes=f"Applicability: {p['applicability']}/5, "
                          f"Complexity: {p['complexity']}. "
                          f"{p['abstract'][:200]}",
                )

    _log.info(
        "Discover: %d variants, %d anomalies, %d corr shifts, %d papers",
        len(result["variants"]),
        len(result["anomalies"]),
        len(result["correlation_shifts"]),
        len(result["external_papers"]),
    )
    return result
