"""
Backtest module — unified public API for all backtesting needs.

Public API:
    backtest(strategy, params, data, config)        — single strategy → instant result
    optimize(data, config, ...)                     — find best with anti-overfitting
    backtest_portfolio(allocations, data, config)    — multi-asset portfolio backtest
    backtest_multi_tf(tf_configs, tf_data, config)   — multi-timeframe fusion backtest
    run_cpcv_scan(symbols, data, config, ...)        — CPCV anti-overfitting scan
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit

from .config import BacktestConfig
from .kernels import (
    DEFAULT_PARAM_GRIDS,
    KERNEL_NAMES,
    KERNEL_REGISTRY,
    KernelResult,
    DetailedKernelResult,
    _equity_from_fused_positions,
    config_to_kernel_costs,
    eval_kernel,
    eval_kernel_detailed,
    eval_kernel_position,
    eval_kernel_position_array,
    eval_kernel_position_series,
    run_kernel,
    run_kernel_detailed,
    scan_all_kernels,
)
from .robust_scan import CPCVResult, RobustScanResult, run_cpcv_scan, run_robust_scan

# ── Internal imports (not promoted, but importable) ─────────────────
from .backtest_engine import BacktestEngine
from .bias_detector import BiasDetector, BiasReport
from .fill_simulator import CostModelFillSimulator
from .manifest import build_run_manifest
from .order_manager import DefaultOrderManager
from .portfolio import PortfolioTracker
from .protocols import BarData, Fill, Order, OrderSide, OrderStatus, OrderType
from .robust import run_robust_backtest
from .tca import TCAReport, TransactionCostAnalyzer


# =====================================================================
#  Result objects
# =====================================================================

@dataclass
class BacktestResult:
    """Result from a single backtest run with optional equity curve."""
    strategy: str
    params: tuple
    ret_pct: float
    max_dd_pct: float
    n_trades: int
    score: float
    equity: Optional[np.ndarray] = field(default=None, repr=False)
    daily_returns: Optional[np.ndarray] = field(default=None, repr=False)
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0

    def __repr__(self) -> str:
        extra = f" sharpe={self.sharpe:.2f}" if self.equity is not None else ""
        return (f"BacktestResult({self.strategy} {self.params}: "
                f"ret={self.ret_pct:+.2f}% dd={self.max_dd_pct:.2f}% "
                f"trades={self.n_trades} score={self.score:.2f}{extra})")


@dataclass
class BestStrategy:
    """The top-ranked strategy from optimize()."""
    strategy: str
    params: tuple
    oos_ret: float
    max_dd: float
    sharpe: float
    dsr_pvalue: float
    mc_positive: float
    mc_mean: float
    wf_score: float

    def __repr__(self) -> str:
        return (f"BestStrategy({self.strategy} {self.params}: "
                f"OOS={self.oos_ret:+.1f}% sharpe={self.sharpe:.2f} "
                f"DSR_p={self.dsr_pvalue:.3f} MC>{self.mc_positive*100:.0f}%)")


@dataclass
class OptimizeResult:
    """Result from optimize() — best strategy + full ranking with robustness."""
    best: BestStrategy
    all_strategies: Dict[str, Dict[str, Any]]
    symbols: List[str]
    total_combos: int
    elapsed_seconds: float
    per_symbol: Optional[Dict[str, Dict[str, Any]]] = None

    def summary(self) -> str:
        lines = [
            f"OptimizeResult: {self.total_combos:,} combos in {self.elapsed_seconds:.1f}s "
            f"({self.total_combos / max(0.1, self.elapsed_seconds):,.0f}/s)",
            f"",
            f"Best: {self.best.strategy} {self.best.params}",
            f"  OOS Return:  {self.best.oos_ret:+.1f}%",
            f"  Max DD:      {self.best.max_dd:.1f}%",
            f"  Sharpe:      {self.best.sharpe:.2f}",
            f"  DSR p-value: {self.best.dsr_pvalue:.3f}  "
            f"({'robust' if self.best.dsr_pvalue > 0.5 else 'may be overfitted'})",
            f"  MC positive: {self.best.mc_positive*100:.0f}%",
            f"",
            f"All strategies (ranked by WF score):",
        ]
        ranked = sorted(self.all_strategies.items(),
                        key=lambda kv: kv[1].get("wf_score", -1e18), reverse=True)
        for sn, d in ranked:
            wf = d.get("wf_score", 0)
            wf_str = f"{wf:>+6.1f}" if abs(wf) < 1e15 else "    N/A"
            lines.append(
                f"  {sn:<14} OOS={d.get('oos_ret',0):>+7.1f}%  "
                f"sharpe={d.get('sharpe',0):>5.2f}  "
                f"DSR={d.get('dsr_p',0):>5.3f}  "
                f"MC>{d.get('mc_pct_positive',0)*100:>3.0f}%  "
                f"score={wf_str}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"OptimizeResult(best={self.best.strategy}, "
                f"combos={self.total_combos:,}, "
                f"time={self.elapsed_seconds:.1f}s)")


# =====================================================================
#  backtest() — test specific strategy + params
# =====================================================================

def _parse_data(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accept dict or DataFrame, return (c, o, h, l) float64 arrays."""
    if isinstance(data, pd.DataFrame):
        c = data["close"].values.astype(np.float64)
        o = data["open"].values.astype(np.float64) if "open" in data.columns else c.copy()
        h = data["high"].values.astype(np.float64) if "high" in data.columns else c.copy()
        l = data["low"].values.astype(np.float64) if "low" in data.columns else c.copy()
        return c, o, h, l
    if isinstance(data, dict):
        c = np.asarray(data["c"], dtype=np.float64)
        o = np.asarray(data.get("o", c), dtype=np.float64)
        h = np.asarray(data.get("h", c), dtype=np.float64)
        l = np.asarray(data.get("l", c), dtype=np.float64)
        return c, o, h, l
    raise TypeError(f"data must be a dict or DataFrame, got {type(data)}")


def backtest(
    strategy,
    params=None,
    data=None,
    config: Optional[BacktestConfig] = None,
    *,
    detailed: bool = False,
) -> BacktestResult:
    """Test a specific strategy with specific parameters.

    Args:
        strategy: Strategy name (e.g. "MA", "RSI") or strategy object
                  with kernel_name/kernel_params attributes.
        params:   Parameter tuple (e.g. (10, 50)). Required if strategy
                  is a string; auto-extracted if strategy is an object.
        data:     OHLC data as dict {"c":..,"o":..,"h":..,"l":..} or DataFrame.
        config:   BacktestConfig. Defaults to BacktestConfig.crypto().
        detailed: If True, compute equity curve, daily returns, Sharpe,
                  Sortino, and Calmar ratio.

    Returns:
        BacktestResult with ret_pct, max_dd_pct, n_trades, score.
        When detailed=True, also includes equity, daily_returns,
        sharpe, sortino, calmar.

    Examples:
        >>> result = backtest("MA", (10, 50), data)
        >>> result = backtest("RSI", (14, 30, 70), data, detailed=True)
        >>> result.equity   # bar-by-bar equity curve
        >>> result.sharpe   # annualized Sharpe ratio
    """
    if config is None:
        config = BacktestConfig.crypto()

    if isinstance(strategy, str):
        name = strategy
        if params is None:
            raise ValueError("params is required when strategy is a string")
        if not isinstance(params, tuple):
            params = tuple(params)
    else:
        name = getattr(strategy, "kernel_name", None)
        if name is None:
            raise ValueError(
                "strategy object must have a kernel_name attribute, "
                f"got {type(strategy).__name__}"
            )
        params = getattr(strategy, "kernel_params", None)
        if params is None:
            raise ValueError(
                "strategy object must have a kernel_params attribute"
            )
        if data is None:
            data = params
            params = getattr(strategy, "kernel_params", None)

    if name not in KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {', '.join(KERNEL_NAMES)}"
        )

    if data is None:
        raise ValueError("data is required")

    c, o, h, l = _parse_data(data)

    if detailed:
        kr = run_kernel_detailed(name, params, c, o, h, l, config)
        return BacktestResult(
            strategy=name, params=params,
            ret_pct=kr.ret_pct, max_dd_pct=kr.max_dd_pct,
            n_trades=kr.n_trades, score=kr.score,
            equity=kr.equity, daily_returns=kr.daily_returns,
            sharpe=kr.sharpe, sortino=kr.sortino, calmar=kr.calmar,
        )

    kr = run_kernel(name, params, c, o, h, l, config)
    return BacktestResult(
        strategy=name, params=params,
        ret_pct=kr.ret_pct, max_dd_pct=kr.max_dd_pct,
        n_trades=kr.n_trades, score=kr.score,
    )


# =====================================================================
#  optimize() — find best strategy + params with anti-overfitting
# =====================================================================

def optimize(
    data,
    config: Optional[BacktestConfig] = None,
    *,
    strategies: Optional[List[str]] = None,
    param_grids: Optional[Dict[str, List[tuple]]] = None,
    method: str = "wf",
) -> OptimizeResult:
    """Find the best strategy and parameters with anti-overfitting.

    Args:
        data: OHLC data. Three formats accepted:
              - Single: dict {"c":..,"o":..,"h":..,"l":..} or DataFrame
              - Multi-symbol: {"BTC": {"c":..,"o":..}, "ETH": {...}}
        config:      BacktestConfig. Defaults to BacktestConfig.crypto().
        strategies:  Which strategies to test. Default: all 18.
        param_grids: Custom parameter grids per strategy.
        method:      ``"wf"`` — purged walk-forward (10-layer, default).
                     ``"cpcv"`` — Combinatorial Purged Cross-Validation.

    Returns:
        OptimizeResult with best strategy, full ranking, robustness metrics.

    Examples:
        >>> result = optimize(data)
        >>> result = optimize(data, method="cpcv")
    """
    if config is None:
        config = BacktestConfig.crypto()

    multi_data, symbols = _normalize_multi_data(data)

    if method == "cpcv":
        raw = run_cpcv_scan(
            symbols=symbols,
            data=multi_data,
            config=config,
            strategies=strategies,
            param_grids=param_grids,
        )
        merged = _merge_across_symbols_cpcv(raw)
        score_key = "cpcv_score"
    else:
        raw = run_robust_scan(
            symbols=symbols,
            data=multi_data,
            config=config,
            strategies=strategies,
            param_grids=param_grids,
        )
        merged = _merge_across_symbols(raw)
        score_key = "wf_score"

    if not merged:
        raise ValueError("No valid results — check data length and strategy names")

    best_sn = max(merged, key=lambda s: merged[s].get(score_key, -1e18))
    b = merged[best_sn]

    best = BestStrategy(
        strategy=best_sn,
        params=b.get("params"),
        oos_ret=b.get("oos_ret", b.get("oos_ret_mean", 0.0)),
        max_dd=b.get("oos_dd", b.get("oos_dd_mean", 0.0)),
        sharpe=b.get("sharpe", 0.0),
        dsr_pvalue=b.get("dsr_p", 0.0),
        mc_positive=b.get("mc_pct_positive", 0.0),
        mc_mean=b.get("mc_mean", 0.0),
        wf_score=b.get(score_key, 0.0),
    )

    return OptimizeResult(
        best=best,
        all_strategies=merged,
        symbols=symbols,
        total_combos=raw.total_combos,
        elapsed_seconds=raw.elapsed_seconds,
        per_symbol=raw.per_symbol if len(symbols) > 1 else None,
    )


@dataclass
class PortfolioResult:
    """Result from a multi-asset portfolio backtest."""
    per_asset: Dict[str, BacktestResult]
    weights: Dict[str, float]
    portfolio_ret_pct: float
    portfolio_dd_pct: float
    portfolio_sharpe: float
    portfolio_equity: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"PortfolioResult({len(self.per_asset)} assets: "
            f"ret={self.portfolio_ret_pct:+.2f}% dd={self.portfolio_dd_pct:.2f}% "
            f"sharpe={self.portfolio_sharpe:.2f})"
        )


def backtest_portfolio(
    allocations: Dict[str, Tuple[str, tuple]],
    data: Dict[str, Any],
    config: Optional[BacktestConfig] = None,
    *,
    weights: Optional[Dict[str, float]] = None,
) -> PortfolioResult:
    """Multi-asset portfolio backtest with per-asset strategy assignment.

    Args:
        allocations: ``{symbol: (strategy_name, params_tuple)}``.
        data: ``{symbol: ohlc_data}`` where ohlc_data is dict or DataFrame.
        config: BacktestConfig (defaults to crypto).
        weights: ``{symbol: weight}``.  Defaults to equal weight.

    Returns:
        PortfolioResult with per-asset results and combined portfolio metrics.
    """
    if config is None:
        config = BacktestConfig.crypto()

    symbols = list(allocations.keys())
    if weights is None:
        w = 1.0 / len(symbols)
        weights = {s: w for s in symbols}

    per_asset: Dict[str, BacktestResult] = {}
    equity_arrays: List[Tuple[str, np.ndarray]] = []

    for sym in symbols:
        if sym not in data:
            continue
        strat, params = allocations[sym]
        res = backtest(strat, params, data[sym], config, detailed=True)
        per_asset[sym] = res
        if res.equity is not None:
            equity_arrays.append((sym, res.equity))

    if not equity_arrays:
        port_ret = sum(per_asset[s].ret_pct * weights.get(s, 0) for s in per_asset)
        port_dd = max((per_asset[s].max_dd_pct for s in per_asset), default=0.0)
        return PortfolioResult(
            per_asset=per_asset, weights=weights,
            portfolio_ret_pct=port_ret, portfolio_dd_pct=port_dd,
            portfolio_sharpe=0.0,
        )

    min_len = min(len(eq) for _, eq in equity_arrays)
    port_eq = np.zeros(min_len, dtype=np.float64)
    for sym, eq in equity_arrays:
        port_eq += eq[:min_len] * weights.get(sym, 0.0)

    peak = np.maximum.accumulate(port_eq)
    dd = np.where(peak > 0, (peak - port_eq) / peak * 100.0, 0.0)
    port_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    port_ret = float((port_eq[-1] / port_eq[0] - 1.0) * 100.0) if port_eq[0] > 0 else 0.0

    bpy = getattr(config, "bars_per_year", 252.0)
    bar_rets = np.diff(port_eq) / np.maximum(port_eq[:-1], 1e-10)
    if len(bar_rets) > 1:
        mu = np.mean(bar_rets)
        sigma = np.std(bar_rets)
        port_sharpe = float(mu / sigma * np.sqrt(bpy)) if sigma > 0 else 0.0
    else:
        port_sharpe = 0.0

    return PortfolioResult(
        per_asset=per_asset, weights=weights,
        portfolio_ret_pct=port_ret, portfolio_dd_pct=port_dd,
        portfolio_sharpe=port_sharpe, portfolio_equity=port_eq,
    )


# =====================================================================
#  backtest_multi_tf() — multi-timeframe fusion backtest
# =====================================================================

FUSION_MODES = ("trend_filter", "consensus", "primary")

_INTERVAL_RANK = {"1m": 0, "5m": 1, "15m": 2, "1h": 3, "4h": 4, "1d": 5, "1w": 6}


@dataclass
class MultiTFBacktestResult:
    """Result from a multi-timeframe fusion backtest."""
    mode: str
    tf_strategies: Dict[str, str]
    tf_params: Dict[str, tuple]
    ret_pct: float
    max_dd_pct: float
    n_trades: int
    sharpe: float
    sortino: float
    calmar: float
    equity: Optional[np.ndarray] = field(default=None, repr=False)
    daily_returns: Optional[np.ndarray] = field(default=None, repr=False)
    tf_positions: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    fused_positions: Optional[np.ndarray] = field(default=None, repr=False)
    finest_interval: str = ""

    def __repr__(self) -> str:
        strats = " | ".join(f"{iv}:{s}" for iv, s in sorted(
            self.tf_strategies.items(), key=lambda x: _INTERVAL_RANK.get(x[0], 99)))
        return (
            f"MultiTFBacktestResult[{self.mode}]({strats}: "
            f"ret={self.ret_pct:+.2f}% dd={self.max_dd_pct:.2f}% "
            f"sharpe={self.sharpe:.2f} trades={self.n_trades})"
        )


def _extract_timestamps(data) -> np.ndarray:
    """Extract unix timestamps from data (DataFrame or dict)."""
    if isinstance(data, pd.DataFrame):
        if hasattr(data.index, 'to_pydatetime'):
            try:
                ts = data.index.astype(np.int64) // 10**9
                return ts.values.astype(np.float64)
            except Exception:
                pass
        for col in ("timestamp", "date", "datetime", "time"):
            if col in data.columns:
                ts = pd.to_datetime(data[col]).astype(np.int64) // 10**9
                return ts.values.astype(np.float64)
        return np.arange(len(data), dtype=np.float64)
    if isinstance(data, dict):
        if "timestamps" in data:
            return np.asarray(data["timestamps"], dtype=np.float64)
        return np.arange(len(data.get("c", [])), dtype=np.float64)
    raise TypeError(f"Cannot extract timestamps from {type(data)}")


def _forward_fill_to_grid(fine_ts: np.ndarray, coarse_ts: np.ndarray,
                          coarse_pos: np.ndarray) -> np.ndarray:
    """Forward-fill coarse-TF positions onto a finer-TF timestamp grid."""
    if len(coarse_ts) == 0 or len(coarse_pos) == 0:
        return np.zeros(len(fine_ts), dtype=np.int64)
    indices = np.searchsorted(coarse_ts, fine_ts, side="right") - 1
    indices = np.clip(indices, 0, len(coarse_pos) - 1)
    result = coarse_pos[indices]
    mask = fine_ts < coarse_ts[0]
    if np.any(mask):
        result = result.copy()
        result[mask] = 0
    return result.astype(np.int64)


@njit(cache=True, fastmath=True)
def _fuse_trend_filter_njit(trend: np.ndarray, entry: np.ndarray) -> np.ndarray:
    """Highest TF sets trend direction; finest TF provides entry timing."""
    n = len(trend)
    fused = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if trend[i] > 0 and entry[i] > 0:
            fused[i] = 1
        elif trend[i] < 0 and entry[i] < 0:
            fused[i] = -1
    return fused


def _fuse_trend_filter(tf_positions_aligned: Dict[str, np.ndarray],
                       sorted_intervals: List[str]) -> np.ndarray:
    trend = tf_positions_aligned[sorted_intervals[-1]]
    entry = tf_positions_aligned[sorted_intervals[0]]
    return _fuse_trend_filter_njit(trend, entry)


@njit(cache=True, fastmath=True)
def _fuse_consensus_njit(arrays_2d: np.ndarray, threshold: float) -> np.ndarray:
    """Majority vote across all TF positions (Numba-compiled)."""
    k = arrays_2d.shape[0]
    n = arrays_2d.shape[1]
    fused = np.zeros(n, dtype=np.int64)
    for i in range(n):
        longs = 0
        shorts = 0
        for j in range(k):
            if arrays_2d[j, i] > 0:
                longs += 1
            elif arrays_2d[j, i] < 0:
                shorts += 1
        if longs > threshold:
            fused[i] = 1
        elif shorts > threshold:
            fused[i] = -1
    return fused


def _fuse_consensus(tf_positions_aligned: Dict[str, np.ndarray],
                    sorted_intervals: List[str]) -> np.ndarray:
    arrays = [tf_positions_aligned[iv] for iv in sorted_intervals]
    arrays_2d = np.stack(arrays)
    threshold = len(arrays) / 2.0
    return _fuse_consensus_njit(arrays_2d, threshold)


def _fuse_primary(tf_positions_aligned: Dict[str, np.ndarray],
                  primary_interval: str) -> np.ndarray:
    return tf_positions_aligned[primary_interval].copy()


def backtest_multi_tf(
    tf_configs: Dict[str, Tuple[str, tuple]],
    tf_data: Dict[str, Any],
    config: Optional[BacktestConfig] = None,
    *,
    mode: str = "trend_filter",
    primary_interval: Optional[str] = None,
) -> MultiTFBacktestResult:
    """Multi-timeframe fusion backtest — mirrors live MultiTFAdapter logic.

    Runs each kernel independently on its own timeframe data, aligns
    positions onto the finest timeframe's timeline, applies one of the
    three fusion modes (trend_filter / consensus / primary), then
    computes a full equity curve from the fused position series.

    Args:
        tf_configs: ``{interval: (strategy_name, params_tuple)}``.
            Example: ``{"1h": ("MA", (10, 50)), "1d": ("MACD", (28, 112, 3))}``
        tf_data: ``{interval: ohlc_data}`` where ohlc_data is a DataFrame
            (with DatetimeIndex or a 'date'/'timestamp' column) or a dict
            ``{"c": ..., "o": ..., "h": ..., "l": ..., "timestamps": ...}``.
        config: Cost / leverage configuration.  Defaults to crypto.
        mode: Fusion mode — ``"trend_filter"`` | ``"consensus"`` | ``"primary"``.
        primary_interval: Required only for ``"primary"`` mode.

    Returns:
        MultiTFBacktestResult with equity curve, metrics, and per-TF positions.

    Examples:
        >>> result = backtest_multi_tf(
        ...     {"1h": ("MA", (10, 50)), "1d": ("MACD", (28, 112, 3))},
        ...     {"1h": df_1h, "1d": df_1d},
        ...     mode="trend_filter",
        ... )
        >>> print(result)
    """
    if mode not in FUSION_MODES:
        raise ValueError(f"Unknown fusion mode '{mode}'. Choose from {FUSION_MODES}")
    if not tf_configs:
        raise ValueError("tf_configs must contain at least one interval")
    if set(tf_configs.keys()) != set(tf_data.keys()):
        missing = set(tf_configs.keys()) - set(tf_data.keys())
        extra = set(tf_data.keys()) - set(tf_configs.keys())
        raise ValueError(
            f"tf_configs and tf_data keys must match. "
            f"Missing data: {missing}, extra data: {extra}"
        )

    if config is None:
        config = BacktestConfig.crypto()

    for iv, d in tf_data.items():
        n = len(d) if isinstance(d, pd.DataFrame) else len(d.get("c", []))
        if n == 0:
            raise ValueError(f"Empty data for interval '{iv}'")

    sorted_intervals = sorted(
        tf_configs.keys(), key=lambda x: _INTERVAL_RANK.get(x, 99),
    )
    finest = sorted_intervals[0]

    if config.interval != finest:
        config = BacktestConfig.crypto(
            leverage=config.leverage,
            stop_loss_pct=config.stop_loss_pct,
            interval=finest,
        )

    costs = config_to_kernel_costs(config)

    # --- 1. Extract per-TF OHLC, timestamps, and position series ----------
    tf_ohlc: Dict[str, Tuple[np.ndarray, ...]] = {}
    tf_ts: Dict[str, np.ndarray] = {}
    tf_pos_raw: Dict[str, np.ndarray] = {}

    for iv in sorted_intervals:
        strat_name, params = tf_configs[iv]
        data = tf_data[iv]
        c, o, h, l = _parse_data(data)
        ts = _extract_timestamps(data)
        tf_ohlc[iv] = (c, o, h, l)
        tf_ts[iv] = ts

        # Each TF gets its own cost model so that daily funding / borrow
        # rates are scaled correctly for that TF's bar frequency.
        iv_config = BacktestConfig.crypto(
            leverage=config.leverage,
            stop_loss_pct=config.stop_loss_pct,
            interval=iv,
        )
        iv_costs = config_to_kernel_costs(iv_config)

        pos_series = eval_kernel_position_series(
            strat_name, params, c, o, h, l,
            iv_costs["sb"], iv_costs["ss"], iv_costs["cm"], iv_costs["lev"],
            iv_costs["dc"], iv_costs["sl"], iv_costs["pfrac"], iv_costs["sl_slip"],
        )
        tf_pos_raw[iv] = pos_series

    fine_ts = tf_ts[finest]
    fine_c, fine_o, fine_h, fine_l = tf_ohlc[finest]

    # --- 2. Align all TF positions onto the finest TF's grid ---------------
    tf_pos_aligned: Dict[str, np.ndarray] = {}
    for iv in sorted_intervals:
        if iv == finest:
            tf_pos_aligned[iv] = tf_pos_raw[iv]
        else:
            tf_pos_aligned[iv] = _forward_fill_to_grid(
                fine_ts, tf_ts[iv], tf_pos_raw[iv],
            )

    # --- 3. Apply fusion logic ---------------------------------------------
    if mode == "trend_filter":
        fused = _fuse_trend_filter(tf_pos_aligned, sorted_intervals)
    elif mode == "consensus":
        fused = _fuse_consensus(tf_pos_aligned, sorted_intervals)
    else:
        pri = primary_interval or sorted_intervals[-1]
        fused = _fuse_primary(tf_pos_aligned, pri)

    # --- 4. Compute equity from fused positions ----------------------------
    ret_pct, max_dd, n_trades, equity, _ = _equity_from_fused_positions(
        fused, fine_c, fine_o,
        costs["sb"], costs["ss"], costs["cm"], costs["lev"],
        costs["dc"], costs["sl"], costs["pfrac"], costs["sl_slip"],
    )

    bpy = costs.get("bars_per_year", 252.0)
    bar_rets = np.diff(equity) / np.maximum(equity[:-1], 1e-10)
    if len(bar_rets) > 1:
        mu = float(np.mean(bar_rets))
        sigma = float(np.std(bar_rets))
        sharpe = mu / sigma * np.sqrt(bpy) if sigma > 0 else 0.0
        neg = bar_rets[bar_rets < 0]
        ds = float(np.std(neg)) if len(neg) > 0 else 1e-10
        sortino = mu / ds * np.sqrt(bpy) if ds > 0 else 0.0
    else:
        sharpe = sortino = 0.0

    ann_ret = (1.0 + ret_pct / 100.0) ** (bpy / max(len(fine_c), 1)) - 1.0
    calmar = ann_ret / (max_dd / 100.0) if max_dd > 0 else 0.0

    return MultiTFBacktestResult(
        mode=mode,
        tf_strategies={iv: tf_configs[iv][0] for iv in sorted_intervals},
        tf_params={iv: tf_configs[iv][1] for iv in sorted_intervals},
        ret_pct=float(ret_pct),
        max_dd_pct=float(max_dd),
        n_trades=int(n_trades),
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        equity=equity,
        daily_returns=bar_rets,
        tf_positions=tf_pos_aligned,
        fused_positions=fused,
        finest_interval=finest,
    )


def _normalize_multi_data(data):
    """Normalize user data input to {sym: {c,o,h,l}} format."""
    if isinstance(data, pd.DataFrame):
        c, o, h, l = _parse_data(data)
        return {"_": {"c": c, "o": o, "h": h, "l": l}}, ["_"]

    if isinstance(data, dict):
        first_val = next(iter(data.values()))
        if isinstance(first_val, np.ndarray):
            return {"_": data}, ["_"]
        if isinstance(first_val, dict):
            symbols = list(data.keys())
            return data, symbols
        if isinstance(first_val, pd.DataFrame):
            out = {}
            for sym, df in data.items():
                c, o, h, l = _parse_data(df)
                out[sym] = {"c": c, "o": o, "h": h, "l": l}
            return out, list(data.keys())

    raise TypeError(f"data must be a dict or DataFrame, got {type(data)}")


def _merge_across_symbols(raw: RobustScanResult) -> Dict[str, Dict[str, Any]]:
    """If multiple symbols, average metrics across symbols per strategy."""
    all_syms = list(raw.per_symbol.keys())
    if len(all_syms) == 1:
        return raw.per_symbol[all_syms[0]]

    all_strats: set = set()
    for sym_d in raw.per_symbol.values():
        all_strats.update(sym_d.keys())

    merged: Dict[str, Dict[str, Any]] = {}
    for sn in all_strats:
        vals = [raw.per_symbol[sym][sn] for sym in all_syms
                if sn in raw.per_symbol.get(sym, {})]
        if not vals:
            continue
        n = len(vals)
        merged[sn] = {
            "params": vals[0].get("params"),
            "oos_ret": sum(v.get("oos_ret", 0) for v in vals) / n,
            "oos_dd": sum(v.get("oos_dd", 0) for v in vals) / n,
            "sharpe": sum(v.get("sharpe", 0) for v in vals) / n,
            "dsr_p": sum(v.get("dsr_p", 0) for v in vals) / n,
            "mc_pct_positive": sum(v.get("mc_pct_positive", 0) for v in vals) / n,
            "mc_mean": sum(v.get("mc_mean", 0) for v in vals) / n,
            "wf_score": sum(v.get("wf_score", 0) for v in vals) / n,
            "oos_trades": sum(v.get("oos_trades", 0) for v in vals),
            "n_symbols": n,
        }
    return merged


def _merge_across_symbols_cpcv(raw: CPCVResult) -> Dict[str, Dict[str, Any]]:
    """Merge CPCV results across symbols per strategy."""
    all_syms = list(raw.per_symbol.keys())
    if len(all_syms) == 1:
        return raw.per_symbol[all_syms[0]]

    all_strats: set = set()
    for sym_d in raw.per_symbol.values():
        all_strats.update(sym_d.keys())

    merged: Dict[str, Dict[str, Any]] = {}
    _avg_keys = (
        "oos_ret_mean", "oos_ret_std", "oos_dd_mean",
        "pct_splits_positive", "ann_ret", "sharpe", "dsr_p",
        "cpcv_score", "mc_mean", "mc_pct_positive",
    )
    for sn in all_strats:
        vals = [raw.per_symbol[sym][sn] for sym in all_syms
                if sn in raw.per_symbol.get(sym, {})]
        if not vals:
            continue
        n = len(vals)
        entry: Dict[str, Any] = {"params": vals[0].get("params")}
        for k in _avg_keys:
            entry[k] = sum(v.get(k, 0) for v in vals) / n
        entry["oos_trades_total"] = sum(v.get("oos_trades_total", 0) for v in vals)
        entry["n_symbols"] = n
        merged[sn] = entry
    return merged


# =====================================================================
#  __all__ — only promote the essentials
# =====================================================================

__all__ = [
    # User-facing API
    "backtest",
    "optimize",
    "backtest_portfolio",
    "backtest_multi_tf",
    "BacktestConfig",
    "BacktestResult",
    "OptimizeResult",
    "BestStrategy",
    "PortfolioResult",
    "MultiTFBacktestResult",
    "FUSION_MODES",
    # CPCV
    "run_cpcv_scan",
    "CPCVResult",
    # Detailed kernel API
    "DetailedKernelResult",
    "run_kernel_detailed",
    # Reference data
    "DEFAULT_PARAM_GRIDS",
    "KERNEL_NAMES",
]
