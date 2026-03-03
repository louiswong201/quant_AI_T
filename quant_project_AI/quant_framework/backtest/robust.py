"""
Robust backtest: multi-window x multi-parameter grid with unified metrics.

Runs the same strategy across non-overlapping time windows and parameter sets,
then aggregates performance distributions (min/median/max) to reduce overfitting
to a single period or parameter.

Supports two execution paths:
  - Kernel fast-path (~36k runs/s): if strategy has kernel_name, uses Numba kernels
  - Python path (~500 runs/s): for custom/ML strategies without kernel equivalents
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .backtest_engine import BacktestEngine
from .kernels import KERNEL_REGISTRY, config_to_kernel_costs, eval_kernel, eval_kernel_detailed
from ..analysis.performance import PerformanceAnalyzer
from ..strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


def run_robust_backtest(
    engine: BacktestEngine,
    strategy_factory: Callable[..., BaseStrategy],
    symbol: str,
    windows: List[Tuple[str, str]],
    param_sets: List[Any],
    *,
    n_trials: Optional[int] = None,
    risk_free_rate: float = 0.03,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Multi-window x multi-parameter robust backtest with unified metrics.

    Automatically detects kernel-capable strategies and uses the Numba fast
    path for ~70x speedup.

    Args:
        engine: BacktestEngine (bound to DataManager + BacktestConfig).
        strategy_factory: callable(params) -> BaseStrategy.
        symbol: ticker symbol.
        windows: non-overlapping [(start, end), ...].
        param_sets: parameter list -- each passed to strategy_factory(param).
        n_trials: total trials for Deflated Sharpe correction (auto-detected if None).
        risk_free_rate: annual risk-free rate for Sharpe/Sortino.

    Returns:
        results: list of per-run dicts with full metrics.
        summary: aggregated stats (by_param, by_window, table).
    """
    test_strategy = strategy_factory(param_sets[0])
    kernel_name = getattr(test_strategy, "kernel_name", None)

    if kernel_name and kernel_name in KERNEL_REGISTRY:
        return _run_robust_kernel(
            engine, strategy_factory, symbol, windows, param_sets,
            kernel_name, n_trials=n_trials, risk_free_rate=risk_free_rate,
        )

    return _run_robust_python(
        engine, strategy_factory, symbol, windows, param_sets,
        n_trials=n_trials, risk_free_rate=risk_free_rate,
    )


def _run_robust_kernel(
    engine: BacktestEngine,
    strategy_factory: Callable[..., BaseStrategy],
    symbol: str,
    windows: List[Tuple[str, str]],
    param_sets: List[Any],
    kernel_name: str,
    *,
    n_trials: Optional[int] = None,
    risk_free_rate: float = 0.03,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Kernel-accelerated robust backtest (~36k runs/s)."""
    total_trials = n_trials or (len(windows) * len(param_sets))
    costs = config_to_kernel_costs(engine.config)
    sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
    lev, dc = costs["lev"], costs["dc"]
    sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

    results: List[Dict[str, Any]] = []
    by_param: Dict[int, List[float]] = {}
    by_window: Dict[int, List[float]] = {}
    table: Dict[Tuple[int, int], float] = {}

    for wi, (start, end) in enumerate(windows):
        data = engine.data_manager.load_data(symbol, start, end)
        if data is None or data.empty:
            for pi in range(len(param_sets)):
                results.append({
                    "window_idx": wi, "param_idx": pi,
                    "start": start, "end": end,
                    "return_rate": None, "final_value": None,
                    "initial_capital": None, "sharpe_ratio": None,
                    "sortino_ratio": None, "max_drawdown": None,
                    "calmar_ratio": None, "deflated_sharpe_pvalue": None,
                    "total_funding_paid": 0.0, "liquidated": False,
                })
            continue

        from .backtest_engine import _to_f64
        c = _to_f64(data["close"].values)
        o = _to_f64(data["open"].values) if "open" in data.columns else c.copy()
        h = _to_f64(data["high"].values) if "high" in data.columns else c.copy()
        l = _to_f64(data["low"].values) if "low" in data.columns else c.copy()
        n = len(c)

        for pi, params in enumerate(param_sets):
            try:
                strat = strategy_factory(params)
                kernel_params = strat.kernel_params  # type: ignore[attr-defined]
                ic = strat.initial_capital

                ret_pct, max_dd_pct, n_trades, eq_curve, _fpos, _parr = eval_kernel_detailed(
                    kernel_name, kernel_params, c, o, h, l,
                    sb, ss, cm, lev, dc, sl, pfrac, sl_slip,
                )

                final_value = float(eq_curve[-1]) if len(eq_curve) > 0 else ic
                ret_rate = (final_value / ic - 1.0) if ic > 0 else 0.0

                pv = eq_curve * ic if eq_curve[0] == 1.0 else eq_curve
                dr = np.zeros(len(pv), dtype=np.float64)
                if len(pv) > 1:
                    dr[1:] = np.diff(pv) / np.where(pv[:-1] > 0, pv[:-1], 1.0)

                analyzer = PerformanceAnalyzer(risk_free_rate=risk_free_rate)
                perf = analyzer.analyze(pv, dr, ic, n_trials=total_trials)

            except Exception as exc:
                logger.warning(
                    "kernel window=(%s,%s) param_idx=%d failed: %s",
                    start, end, pi, exc,
                )
                perf = {}
                ret_rate = None
                final_value = None
                ic = None

            row = {
                "window_idx": wi,
                "param_idx": pi,
                "start": start,
                "end": end,
                "return_rate": ret_rate,
                "final_value": final_value,
                "initial_capital": ic,
                "sharpe_ratio": perf.get("sharpe_ratio"),
                "sortino_ratio": perf.get("sortino_ratio"),
                "max_drawdown": perf.get("max_drawdown"),
                "calmar_ratio": perf.get("calmar_ratio"),
                "deflated_sharpe_pvalue": perf.get("deflated_sharpe_pvalue"),
                "total_funding_paid": 0.0,
                "liquidated": False,
            }
            results.append(row)

            if ret_rate is not None:
                by_param.setdefault(pi, []).append(ret_rate)
                by_window.setdefault(wi, []).append(ret_rate)
                table[(wi, pi)] = ret_rate

    summary: Dict[str, Any] = {
        "by_param": by_param,
        "by_window": by_window,
        "table": table,
    }
    return results, summary


def _run_robust_python(
    engine: BacktestEngine,
    strategy_factory: Callable[..., BaseStrategy],
    symbol: str,
    windows: List[Tuple[str, str]],
    param_sets: List[Any],
    *,
    n_trials: Optional[int] = None,
    risk_free_rate: float = 0.03,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Original Python-path robust backtest (~500 runs/s)."""
    analyzer = PerformanceAnalyzer(risk_free_rate=risk_free_rate)
    total_trials = n_trials or (len(windows) * len(param_sets))

    results: List[Dict[str, Any]] = []
    by_param: Dict[int, List[float]] = {}
    by_window: Dict[int, List[float]] = {}
    table: Dict[Tuple[int, int], float] = {}

    for wi, (start, end) in enumerate(windows):
        for pi, params in enumerate(param_sets):
            try:
                strategy = strategy_factory(params)
                res = engine.run(strategy, symbol, start, end)
            except Exception as exc:
                logger.warning(
                    "window=(%s,%s) param_idx=%d failed: %s", start, end, pi, exc,
                )
                res = None

            if res and res.get("initial_capital", 0) > 0:
                pv = res.get("portfolio_values", np.array([]))
                dr = res.get("daily_returns", np.array([]))
                ic = res["initial_capital"]

                perf = analyzer.analyze(pv, dr, ic, n_trials=total_trials)
                ret = perf.get("total_return")
            else:
                perf = {}
                ret = None

            row = {
                "window_idx": wi,
                "param_idx": pi,
                "start": start,
                "end": end,
                "return_rate": ret,
                "final_value": res.get("final_value") if res else None,
                "initial_capital": res.get("initial_capital") if res else None,
                "sharpe_ratio": perf.get("sharpe_ratio"),
                "sortino_ratio": perf.get("sortino_ratio"),
                "max_drawdown": perf.get("max_drawdown"),
                "calmar_ratio": perf.get("calmar_ratio"),
                "deflated_sharpe_pvalue": perf.get("deflated_sharpe_pvalue"),
                "total_funding_paid": res.get("total_funding_paid", 0.0) if res else 0.0,
                "liquidated": res.get("liquidated", False) if res else False,
            }
            results.append(row)

            if ret is not None:
                by_param.setdefault(pi, []).append(ret)
                by_window.setdefault(wi, []).append(ret)
                table[(wi, pi)] = ret

    summary: Dict[str, Any] = {
        "by_param": by_param,
        "by_window": by_window,
        "table": table,
    }
    return results, summary
