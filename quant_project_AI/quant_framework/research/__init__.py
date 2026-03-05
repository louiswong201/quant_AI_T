"""V4 Intelligent Strategy Research System.

Provides four engines backed by a persistent SQLite research database:

- **Monitor Engine** — daily health trends, regime detection, performance attribution
- **Optimize Engine** — Bayesian parameter updates, composite gate scoring, champion/challenger
- **Portfolio Engine** — correlation analysis, weight optimization, portfolio-level metrics
- **Discover Engine** — internal variant mining, market anomaly scanning, external research
"""

from .database import ResearchDB
from .monitor import run_monitor, regime_probabilities, compute_health_metrics
from .optimizer import (
    run_optimizer,
    bayesian_param_update,
    composite_gate_score,
    param_neighborhood_stability,
)
from .portfolio import (
    run_portfolio_analysis,
    compute_correlation_matrix,
    optimize_weights,
    portfolio_metrics,
)
from .discover import (
    run_discover,
    discover_variants,
    scan_anomalies,
    scan_external_research,
)

__all__ = [
    "ResearchDB",
    "run_monitor",
    "regime_probabilities",
    "compute_health_metrics",
    "run_optimizer",
    "bayesian_param_update",
    "composite_gate_score",
    "param_neighborhood_stability",
    "run_portfolio_analysis",
    "compute_correlation_matrix",
    "optimize_weights",
    "portfolio_metrics",
    "run_discover",
    "discover_variants",
    "scan_anomalies",
    "scan_external_research",
]
