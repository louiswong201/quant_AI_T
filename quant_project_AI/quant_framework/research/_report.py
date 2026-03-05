"""V4 Daily Research Report Generator.

Produces structured markdown reports with:
- Health Dashboard (trend sparklines)
- Regime Map
- Performance Attribution
- Action Items with specific recommendations
- Portfolio View (correlation, weights, diversification)
- Discover section (anomalies, variants, papers)
- Changelog (config version diffs)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .database import ResearchDB

# ═══════════════════════════════════════════════════════════════
#  Sparkline helpers
# ═══════════════════════════════════════════════════════════════

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: List[float], width: int = 8) -> str:
    """Text-based sparkline for trend visualisation."""
    if not values or len(values) < 2:
        return "─" * width
    mn, mx = min(values), max(values)
    rng = mx - mn
    if rng < 1e-12:
        return _SPARK_CHARS[3] * min(len(values), width)
    # Resample if needed
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    return "".join(
        _SPARK_CHARS[min(int((v - mn) / rng * 7), 7)]
        for v in sampled
    )


def _trend_arrow(values: List[float]) -> str:
    """Return trend indicator: rising/falling/flat."""
    if len(values) < 2:
        return "─"
    delta = values[-1] - values[0]
    if abs(delta) < 0.01:
        return "→"
    return "↗" if delta > 0 else "↘"


# ═══════════════════════════════════════════════════════════════
#  Section generators
# ═══════════════════════════════════════════════════════════════

def _section_header(lines: list, title: str):
    lines.append(f"\n## {title}\n")


def _section_data_refresh(lines: list, data_updates: Dict[str, int]):
    _section_header(lines, "Data Refresh")
    new_bars = sum(data_updates.values())
    updated = sum(1 for v in data_updates.values() if v > 0)
    lines.append(f"- Updated **{updated}** / {len(data_updates)} symbols, **{new_bars}** new bars total\n")


def _section_health_dashboard(
    lines: list,
    monitor_results: List[Dict[str, Any]],
    db: Optional[ResearchDB],
):
    _section_header(lines, "Health Dashboard")

    n_healthy = sum(1 for r in monitor_results if r["status"] == "HEALTHY")
    n_watch = sum(1 for r in monitor_results if r["status"] == "WATCH")
    n_alert = sum(1 for r in monitor_results if r["status"] == "ALERT")

    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| HEALTHY | {n_healthy} |")
    lines.append(f"| WATCH | {n_watch} |")
    lines.append(f"| ALERT | {n_alert} |")
    lines.append("")

    lines.append("### Per-Strategy Detail\n")
    lines.append(
        "| Symbol | Strategy | Lev | Sharpe(30d) | Trend | DD% | "
        "Win Rate | Trades/yr | Status |"
    )
    lines.append(
        "|--------|----------|-----|-------------|-------|-----|"
        "----------|-----------|--------|"
    )

    for r in sorted(monitor_results, key=lambda x: x.get("original_sharpe", 0), reverse=True):
        h = r.get("health", {})
        s30 = h.get("sharpe_30d", 0)

        # Build trend sparkline from DB history
        trend_str = "─"
        if db:
            history = db.get_health_trend(r["symbol"], r["strategy"], days=14)
            sharpes = [row.get("sharpe_30d", 0) for row in reversed(history)]
            if len(sharpes) >= 2:
                trend_str = _sparkline(sharpes, 8) + " " + _trend_arrow(sharpes)

        lines.append(
            f"| {r['symbol']} | {r['strategy']} | {r['leverage']}x | "
            f"{s30:.2f} | {trend_str} | {h.get('drawdown_pct', 0):.1%} | "
            f"{h.get('win_rate', 0):.0%} | {h.get('trade_freq', 0):.1f} | "
            f"**{r['status']}** |"
        )
    lines.append("")


def _section_regime_map(lines: list, monitor_results: List[Dict[str, Any]]):
    regimes = [r for r in monitor_results if r.get("regime")]
    if not regimes:
        return
    _section_header(lines, "Regime Map")

    lines.append("| Symbol | Trending | Mean Rev | High Vol | Compression | Dominant |")
    lines.append("|--------|----------|----------|----------|-------------|----------|")

    seen = set()
    for r in regimes:
        sym = r["symbol"]
        if sym in seen:
            continue
        seen.add(sym)
        reg = r["regime"]
        dominant = max(reg, key=reg.get)
        lines.append(
            f"| {sym} | {reg.get('trending', 0):.0%} | {reg.get('mean_reverting', 0):.0%} | "
            f"{reg.get('high_vol', 0):.0%} | {reg.get('compression', 0):.0%} | "
            f"**{dominant}** |"
        )
    lines.append("")


def _section_attribution(lines: list, monitor_results: List[Dict[str, Any]]):
    attribs = [r for r in monitor_results if r.get("attribution")]
    if not attribs:
        return
    _section_header(lines, "Performance Attribution")
    lines.append("*Why degraded strategies are underperforming:*\n")

    for r in attribs:
        a = r["attribution"]
        lines.append(
            f"- **{r['symbol']}/{r['strategy']}** ({r['status']}): "
            f"Market={a['market_factor']:+.1%}, "
            f"Strategy={a['strategy_factor']:+.1%}, "
            f"Param={a['parameter_factor']:+.1%} — "
            f"_{a['explanation']}_"
        )
    lines.append("")


def _section_action_items(lines: list, changes: List[Dict[str, Any]]):
    action_items = [c for c in changes if c["priority"] != "NONE"]
    keep_items = [c for c in changes if c["priority"] == "NONE"]

    if action_items:
        _section_header(lines, f"Action Items ({len(action_items)})")
        lines.append("| Priority | Symbol | Action | Detail |")
        lines.append("|----------|--------|--------|--------|")
        for c in action_items:
            lines.append(
                f"| **{c['priority']}** | {c['symbol']} | {c['action']} | {c['detail']} |"
            )
        lines.append("")

    if keep_items:
        _section_header(lines, f"Stable Strategies ({len(keep_items)})")
        lines.append("| Symbol | Strategy | Sharpe | Status |")
        lines.append("|--------|----------|--------|--------|")
        for c in keep_items:
            lines.append(
                f"| {c['symbol']} | {c.get('current_strategy', '-')} | "
                f"{c.get('current_sharpe', 0):.2f} | {c.get('health_status', '-')} |"
            )
        lines.append("")


def _section_optimizer(lines: list, optimize_results: List[Dict[str, Any]]):
    if not optimize_results:
        return
    _section_header(lines, "Optimizer Results")

    lines.append("| Symbol | Strategy | Gate Score | Sharpe | Neighborhood | Promotion |")
    lines.append("|--------|----------|------------|--------|--------------|-----------|")
    for r in sorted(optimize_results, key=lambda x: x["gate_score"], reverse=True):
        ns = f"{r['neighborhood_score']:.2f}" if r.get("neighborhood_score") is not None else "—"
        promo = "PROMOTE" if r.get("promotion") else "—"
        lines.append(
            f"| {r['symbol']} | {r['strategy']} | {r['gate_score']:.3f} | "
            f"{r['sharpe']:.2f} | {ns} | {promo} |"
        )
    lines.append("")


def _section_portfolio(lines: list, portfolio_result: Optional[Dict[str, Any]]):
    if not portfolio_result:
        return
    pm = portfolio_result.get("portfolio_metrics", {})
    if not pm:
        return

    _section_header(lines, "Portfolio View")

    lines.append("### Portfolio Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Portfolio Sharpe | {pm.get('portfolio_sharpe', 0):.3f} |")
    lines.append(f"| Portfolio Sortino | {pm.get('portfolio_sortino', 0):.3f} |")
    lines.append(f"| Portfolio Max DD | {pm.get('portfolio_max_dd', 0):.1%} |")
    lines.append(f"| Portfolio Vol | {pm.get('portfolio_vol', 0):.1%} |")
    lines.append(f"| Diversification Ratio | {pm.get('diversification_ratio', 0):.2f} |")
    lines.append(f"| Active Strategies | {pm.get('n_strategies', 0)} |")
    lines.append("")

    weights = portfolio_result.get("weights", {})
    if weights:
        lines.append("### Recommended Weights\n")
        lines.append("| Strategy | Weight | Position Size |")
        lines.append("|----------|--------|---------------|")
        pos = portfolio_result.get("position_sizes", {})
        for k, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {k} | {w:.1%} | {pos.get(k, 0):.2%} |")
        lines.append("")

    recs = portfolio_result.get("recommendations", [])
    if recs:
        lines.append("### Portfolio Recommendations\n")
        for r in recs:
            lines.append(f"- {r}")
        lines.append("")


def _section_discover(lines: list, discover_result: Optional[Dict[str, Any]]):
    if not discover_result:
        return

    anomalies = discover_result.get("anomalies", [])
    corr_shifts = discover_result.get("correlation_shifts", [])
    variants = discover_result.get("variants", [])
    papers = discover_result.get("external_papers", [])

    if not any([anomalies, corr_shifts, variants, papers]):
        return

    _section_header(lines, "Discovery")

    if anomalies:
        lines.append("### Market Anomalies\n")
        lines.append("| Symbol | Severity | Flags | Recommendation |")
        lines.append("|--------|----------|-------|----------------|")
        for a in anomalies[:10]:
            lines.append(
                f"| {a['symbol']} | {a['severity']:.2f} | "
                f"{', '.join(a['flags'][:2])} | {a['recommendation'][:60]} |"
            )
        lines.append("")

    if corr_shifts:
        lines.append("### Cross-Asset Correlation Shifts\n")
        lines.append("| Pair | Full Corr | Recent Corr | Direction |")
        lines.append("|------|-----------|-------------|-----------|")
        for cs in corr_shifts[:5]:
            lines.append(
                f"| {cs['pair']} | {cs['full_corr']:+.2f} | "
                f"{cs['recent_corr']:+.2f} | {cs['direction']} |"
            )
        lines.append("")

    if variants:
        lines.append("### New Variant Discoveries\n")
        lines.append("| Symbol | Strategy | Sharpe | WF Score |")
        lines.append("|--------|----------|--------|----------|")
        for v in variants[:10]:
            lines.append(
                f"| {v['symbol']} | {v['strategy']} | "
                f"{v['sharpe']:.2f} | {v.get('wf_score', 0):.1f} |"
            )
        lines.append("")

    if papers:
        lines.append("### External Research (arXiv)\n")
        lines.append("| Applicability | Complexity | Title |")
        lines.append("|---------------|------------|-------|")
        for p in papers[:5]:
            title = p['title'][:80]
            url = p.get('url', '')
            lines.append(
                f"| {p['applicability']}/5 | {p['complexity']} | "
                f"[{title}]({url}) |"
            )
        lines.append("")


def _section_changelog(lines: list, db: Optional[ResearchDB]):
    if not db:
        return
    versions = db.get_config_versions(limit=5)
    if not versions:
        return

    _section_header(lines, "Changelog")
    lines.append("*Recent config changes:*\n")

    for v in versions:
        ts = v.get("ts", "?")
        summary = v.get("summary", "—")
        n = v.get("n_changes", 0)
        lines.append(f"- **{ts}** — {summary} ({n} change{'s' if n != 1 else ''})")

        diff = v.get("diff_json")
        if isinstance(diff, dict):
            for k, val in list(diff.items())[:3]:
                if k == "changes" and isinstance(val, list):
                    for ch in val[:3]:
                        if isinstance(ch, dict):
                            lines.append(
                                f"  - {ch.get('symbol', '?')}: "
                                f"{ch.get('old', '?')} → {ch.get('new', '?')}"
                            )
    lines.append("")


# ═══════════════════════════════════════════════════════════════
#  Main report generator
# ═══════════════════════════════════════════════════════════════

def generate_v4_report(
    *,
    monitor_results: List[Dict[str, Any]],
    changes: List[Dict[str, Any]],
    data_updates: Dict[str, int],
    output_path: str,
    db: Optional[ResearchDB] = None,
    portfolio_result: Optional[Dict[str, Any]] = None,
    discover_result: Optional[Dict[str, Any]] = None,
    optimize_results: Optional[List[Dict[str, Any]]] = None,
    mode: str = "daily",
) -> str:
    """Generate the full V4 research report as markdown."""
    now = datetime.now()
    lines = [
        f"# V4 Strategy Research Report",
        f"",
        f"**Date:** {now.strftime('%Y-%m-%d %H:%M')}  ",
        f"**Mode:** {mode.upper()}  ",
        f"**Strategies:** {len(monitor_results)}",
        f"",
    ]

    # Always present
    _section_data_refresh(lines, data_updates)
    _section_health_dashboard(lines, monitor_results, db)
    _section_regime_map(lines, monitor_results)
    _section_attribution(lines, monitor_results)
    _section_action_items(lines, changes)

    # Weekly+ sections
    if optimize_results:
        _section_optimizer(lines, optimize_results)
    if portfolio_result:
        _section_portfolio(lines, portfolio_result)
    if discover_result:
        _section_discover(lines, discover_result)

    # Changelog
    _section_changelog(lines, db)

    # Mode note
    if mode == "daily":
        lines.append(
            "\n*Quick mode — run `--mode weekly` for full re-scan, portfolio, and anomaly analysis.*"
        )

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)

    return report
