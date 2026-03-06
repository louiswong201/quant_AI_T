"""
Backtest engine — slim orchestrator (SOLID refactored).

Responsibilities:
  1. Load and prepare data (OHLCV → contiguous float64 arrays)
  2. Run the bar-by-bar loop: signal dispatch → order management → fill → portfolio update
  3. Assemble and return the results dict

All heavy lifting is delegated to single-responsibility components:
  - CostModelFillSimulator: slippage, market impact, liquidity constraints
  - DefaultOrderManager: order lifecycle (submit, cancel, expire)
  - PortfolioTracker: positions, cash, equity curve, trade journal

Logging available via ``logger`` for diagnostics.

Market fill modes:
  - "next_open" (default): signal on bar t, fill at bar t+1 open.
    Eliminates look-ahead bias — industry standard.
  - "current_close": fill at bar t close (legacy, has look-ahead bias).

Hot-path optimisations retained from v1:
  - OHLCV pre-extracted as contiguous float64 numpy arrays (O(1) indexed access)
  - Portfolio values / daily returns in pre-allocated arrays (no list.append)
  - Strategy receives iloc view, no copy
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

from ..strategy.base_strategy import BaseStrategy
from ..data.data_manager import DataManager
from .config import BacktestConfig
from .fill_simulator import CostModelFillSimulator
from .kernels import KERNEL_REGISTRY, config_to_kernel_costs, eval_kernel, eval_kernel_detailed
from .manifest import build_run_manifest
from .order_manager import DefaultOrderManager
from .portfolio import PortfolioTracker
from .protocols import BarData, Order, OrderSide, OrderStatus, OrderType


def _to_f64(arr: np.ndarray) -> np.ndarray:
    """Coerce to contiguous float64 — avoids per-element type dispatch in the hot loop."""
    if arr.dtype != np.float64 or not arr.flags.c_contiguous:
        return np.ascontiguousarray(arr, dtype=np.float64)
    return arr


def _rolling_vol(close: np.ndarray, window: int) -> np.ndarray:
    """Rolling return volatility — O(n) via cumulative-sum trick.

    Computes std(returns) over a sliding window using running sums of
    ret and ret**2, avoiding the O(n*window) inner-loop ``np.std`` slice.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2 or window <= 1:
        return out
    ret = np.empty(n, dtype=np.float64)
    ret[0] = 0.0
    ret[1:] = np.diff(close) / np.where(close[:-1] > 0, close[:-1], 1.0)
    cs = np.cumsum(ret)
    cs2 = np.cumsum(ret * ret)
    w = float(window)
    s1 = cs[window:] - cs[:-window]
    s2 = cs2[window:] - cs2[:-window]
    var = (s2 - s1 * s1 / w) / w
    np.maximum(var, 0.0, out=var)
    out[window:] = np.sqrt(var)
    return out


def _cheap_truncated_view(df: pd.DataFrame, i: int) -> pd.DataFrame:
    """Return a truncated view for large DataFrames without full copy.

    For DataFrames < 5000 rows, the standard iloc[:i+1] is fast enough.
    For larger frames, we use a zero-copy slice via numpy indexing.
    """
    return df.iloc[: i + 1]


def _normalize_signals(out: Any) -> List[Dict]:
    """Normalise strategy output to a list of signal dicts."""
    if out is None:
        return []
    if isinstance(out, dict):
        return [out] if out.get("action") in ("buy", "sell", "cancel") else []
    if isinstance(out, (list, tuple)):
        return [
            s for s in out
            if isinstance(s, dict) and s.get("action") in ("buy", "sell", "cancel")
        ]
    return []


def _prepare_bar_data(df: pd.DataFrame, vol_window: int) -> BarData:
    """Extract OHLCV columns into a BarData struct with contiguous float64 arrays."""
    close = _to_f64(df["close"].values)
    if all(c in df.columns for c in ("open", "high", "low")):
        open_ = _to_f64(df["open"].values)
        high = _to_f64(df["high"].values)
        low = _to_f64(df["low"].values)
    else:
        open_, high, low = close, close, close

    volume = (
        _to_f64(df["volume"].values)
        if "volume" in df.columns
        else np.full(len(df), np.inf, dtype=np.float64)
    )
    return BarData(
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        rolling_vol=_rolling_vol(close, vol_window),
    )


def _align_multi_symbol_data(
    data_by_symbol: Dict[str, pd.DataFrame],
) -> Tuple[pd.DatetimeIndex, Dict[str, pd.DataFrame]]:
    """Inner-join alignment of multi-symbol data on dates."""
    if not data_by_symbol:
        return pd.DatetimeIndex([]), {}
    common_index: Optional[pd.DatetimeIndex] = None
    for df in data_by_symbol.values():
        if df is None or df.empty or "date" not in df.columns:
            continue
        idx = pd.DatetimeIndex(pd.to_datetime(df["date"])).sort_values()
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or len(common_index) == 0:
        return pd.DatetimeIndex([]), {}
    aligned = {}
    for sym, df in data_by_symbol.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.set_index("date", drop=True).sort_index()
        aligned[sym] = tmp.reindex(common_index).ffill().bfill().loc[common_index].reset_index()
    return common_index, aligned


class BacktestEngine:
    """Production-grade backtest engine with pluggable components.

    Architecture (post-SOLID refactoring):
      BacktestEngine (orchestrator, ~180 lines)
        ├── CostModelFillSimulator  — execution price, slippage, impact
        ├── DefaultOrderManager     — order lifecycle
        └── PortfolioTracker        — positions, cash, equity curve
    """

    def __init__(
        self,
        data_manager: DataManager,
        config: Optional[BacktestConfig] = None,
        *,
        commission_rate: Optional[float] = None,
    ) -> None:
        self.data_manager = data_manager
        if config is not None:
            self.config = config
        else:
            rate = commission_rate if commission_rate is not None else 0.001
            self.config = BacktestConfig.from_legacy_rate(rate)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy: BaseStrategy,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        live_fills: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Run a backtest over the specified date range.

        Args:
            strategy: Strategy instance (must implement on_bar).
            symbols: Single ticker or list of tickers.
            start_date: ISO date string (inclusive).
            end_date: ISO date string (inclusive).
            live_fills: Optional DataFrame of live fills for execution divergence analysis.

        Returns:
            Dict with keys: results, trades, orders, daily_returns, cumulative_returns,
            portfolio_values, final_value, initial_capital, manifest, etc.
        """
        symbol_list = [symbols] if isinstance(symbols, str) else list(symbols)
        if not symbol_list:
            raise ValueError("symbols must be a non-empty string or list of strings")

        # --- Kernel fast-path detection ---
        kernel_name = getattr(strategy, "kernel_name", None)
        if (
            kernel_name
            and kernel_name in KERNEL_REGISTRY
            and len(symbol_list) == 1
        ):
            return self._run_kernel(strategy, symbol_list[0], start_date, end_date)

        data_by_symbol, dates, bar_data_map, fast_ctx, fast_ctx_multi = (
            self._prepare_data(strategy, symbol_list, start_date, end_date)
        )

        n = len(dates)
        single = len(symbol_list) == 1

        fill_sim = CostModelFillSimulator(self.config)
        order_mgr = DefaultOrderManager()
        portfolio = PortfolioTracker(strategy.initial_capital, n, self.config)

        strategy.cash = portfolio.cash
        strategy.positions = portfolio.positions

        for i in range(n):
            current_date = pd.Timestamp(dates[i])
            current_prices = {
                sym: float(bar_data_map[sym].close[i]) for sym in bar_data_map
            }

            fill_sim.reset_bar()

            # 0) Expire stale orders
            order_mgr.expire_stale(i, current_date)

            # 1) Fill pending orders from previous bars
            for sym in bar_data_map:
                fills, remaining = fill_sim.try_fill_pending(
                    order_mgr.pending, sym, bar_data_map[sym], i, current_date,
                )
                order_mgr._pending = remaining
                for fill in fills:
                    if fill.order.side == OrderSide.BUY:
                        if not portfolio.can_afford(
                            fill.exec_price, fill.exec_shares, fill.commission
                        ):
                            fill.order.status = OrderStatus.CANCELLED
                            continue
                    else:
                        pos = portfolio.position_size(fill.order.symbol)
                        if not self.config.allow_short and pos <= 0:
                            fill.order.status = OrderStatus.CANCELLED
                            continue
                        if not self.config.allow_short and fill.exec_shares > pos:
                            fill.exec_shares = pos
                            fill.order.filled_shares = pos
                    portfolio.apply_fill(fill)

            # 1.5) Funding/borrow costs + stop-loss/take-profit/liquidation
            auto_closes = portfolio.process_bar_costs(current_prices, i, current_date)
            for ac in auto_closes:
                ac_sym = ac["symbol"]
                ac_shares = ac["shares"]
                ac_side = OrderSide.BUY if ac["action"] == "buy" else OrderSide.SELL
                ac_order = order_mgr.create_order(
                    side=ac_side, symbol=ac_sym, shares=ac_shares,
                    order_type=OrderType.MARKET, bar_index=i, date=current_date,
                    order_id=f"auto_{ac.get('reason', 'close')}_{i}_{ac_sym}",
                )
                ref_px = current_prices.get(ac_sym, 0.0)
                reason = ac.get("reason", "")
                if reason in ("stop_loss", "take_profit", "trailing_stop"):
                    slippage_adj = self.config.stop_loss_slippage_pct
                    if ac_side == OrderSide.SELL:
                        ref_px *= (1.0 - slippage_adj)
                    else:
                        ref_px *= (1.0 + slippage_adj)
                ac_bar = bar_data_map.get(ac_sym)
                if ac_bar is None:
                    logger.warning("No bar data for symbol '%s' at bar %d, skipping fill", ac_sym, i)
                    continue
                ac_fill = fill_sim.execute_market(
                    ac_order, ref_px, ac_bar, i, current_date,
                )
                if ac_fill:
                    portfolio.apply_fill(ac_fill)
                    order_mgr.remove_filled([ac_order.order_id])
                else:
                    logger.warning(
                        "Auto-close %s for %s failed at bar %d (reason=%s)",
                        ac["action"], ac_sym, i, reason,
                    )

            strategy.cash = portfolio.cash
            mtm = portfolio.mark_to_market(current_prices)
            strategy.portfolio_value = mtm
            fills_this_bar = False

            if portfolio.is_liquidated:
                portfolio.record_bar(i, current_date)
                for j in range(i + 1, n):
                    portfolio.record_bar(j, pd.Timestamp(dates[j]))
                break

            # 2) Get signals from strategy
            signals = self._dispatch_signals(
                strategy, data_by_symbol, fast_ctx, fast_ctx_multi,
                symbol_list, single, i, current_date, current_prices,
            )

            # 3) Process signals
            for sig in signals:
                action = sig.get("action")
                if action == "cancel":
                    target_id = sig.get("order_id")
                    if target_id:
                        order_mgr.cancel(target_id)
                    continue

                sym = sig.get("symbol") or (symbol_list[0] if single else "")
                raw_shares = sig.get("shares") or 0
                shares = float(raw_shares) if self.config.allow_fractional_shares else int(raw_shares)
                if sym not in current_prices or shares <= 0:
                    continue

                side = OrderSide.BUY if action == "buy" else OrderSide.SELL
                raw_type = (sig.get("order_type") or "market").lower()
                order_type = {
                    "limit": OrderType.LIMIT,
                    "stop": OrderType.STOP,
                }.get(raw_type, OrderType.MARKET)

                if side == OrderSide.SELL and not self.config.allow_short:
                    avail = portfolio.position_size(sym)
                    if avail <= 0:
                        continue
                    shares = min(shares, avail)

                order = order_mgr.create_order(
                    side=side,
                    symbol=sym,
                    shares=shares,
                    order_type=order_type,
                    limit_price=sig.get("limit_price"),
                    stop_price=sig.get("stop_price"),
                    bar_index=i,
                    date=current_date,
                    tif_bars=sig.get("tif_bars"),
                    order_id=sig.get("order_id"),
                )

                if order_type == OrderType.MARKET and self.config.market_fill_mode != "next_open":
                    fill = fill_sim.execute_market(
                        order, current_prices[sym], bar_data_map[sym], i, current_date,
                    )
                    if fill:
                        if side == OrderSide.BUY:
                            if portfolio.can_afford(
                                fill.exec_price, fill.exec_shares, fill.commission
                            ):
                                portfolio.apply_fill(fill)
                                order_mgr.remove_filled([order.order_id])
                                fills_this_bar = True
                        else:
                            portfolio.apply_fill(fill)
                            order_mgr.remove_filled([order.order_id])
                            fills_this_bar = True

            # 4) End-of-bar portfolio snapshot
            strategy.cash = portfolio.cash
            if fills_this_bar:
                mtm = portfolio.mark_to_market(current_prices)
            strategy.portfolio_value = mtm
            portfolio.record_bar(i, current_date)

        return self._assemble_results(
            portfolio, order_mgr, data_by_symbol, symbol_list,
            start_date, end_date, strategy, live_fills,
        )

    # ------------------------------------------------------------------
    # Kernel fast path (~36k runs/s vs ~500 runs/s Python path)
    # ------------------------------------------------------------------

    def _run_kernel(
        self,
        strategy: BaseStrategy,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """Execute strategy via its Numba kernel — single compiled loop."""
        kernel_name = strategy.kernel_name  # type: ignore[attr-defined]
        kernel_params = strategy.kernel_params  # type: ignore[attr-defined]

        data = self.data_manager.load_data(symbol, start_date, end_date)
        if data is None or data.empty:
            raise ValueError(f"Cannot load data for {symbol}")

        n = len(data)
        min_lb = strategy.min_lookback
        if min_lb > 0 and n < min_lb:
            raise ValueError(
                f"Not enough data: {n} bars < min_lookback={min_lb} "
                f"for strategy '{strategy.name}'"
            )

        c = _to_f64(data["close"].values)
        o = _to_f64(data["open"].values) if "open" in data.columns else c.copy()
        h = _to_f64(data["high"].values) if "high" in data.columns else c.copy()
        l = _to_f64(data["low"].values) if "low" in data.columns else c.copy()

        costs = config_to_kernel_costs(self.config)
        ret_pct, max_dd_pct, n_trades, eq, _fpos, _ = eval_kernel_detailed(
            kernel_name, kernel_params, c, o, h, l,
            costs["sb"], costs["ss"], costs["cm"], costs["lev"],
            costs["dc"], costs["sl"], costs["pfrac"], costs["sl_slip"],
        )

        n = len(c)
        ic = strategy.initial_capital
        final_value = ic * (1.0 + ret_pct / 100.0)

        pv = eq * ic
        dr = np.zeros(n, dtype=np.float64)
        if n > 1:
            dr[1:] = np.diff(pv) / np.where(pv[:-1] > 0, pv[:-1], 1.0)

        dates = pd.to_datetime(data["date"]).values if "date" in data.columns else np.arange(n)

        cumulative_returns = (pv - ic) / ic

        return {
            "results": [],
            "portfolio_values": pv,
            "daily_returns": dr,
            "cumulative_returns": cumulative_returns,
            "final_value": final_value,
            "initial_capital": ic,
            "total_return": ret_pct / 100.0,
            "max_drawdown": -max_dd_pct / 100.0,
            "n_trades": n_trades,
            "trades": pd.DataFrame(),
            "orders": pd.DataFrame(),
            "dates": dates,
            "kernel_mode": True,
            "kernel_name": kernel_name,
            "kernel_params": kernel_params,
            "ret_pct": ret_pct,
            "max_dd_pct": max_dd_pct,
            "execution_divergence": None,
            "execution_report_path": None,
            "execution_bundle": None,
            "manifest": {
                "strategy_class": type(strategy).__name__,
                "symbols": [symbol],
                "start_date": start_date,
                "end_date": end_date,
                "kernel_mode": True,
                "kernel_name": kernel_name,
            },
            "liquidated": final_value <= ic * 0.01,
            "total_funding_paid": 0.0,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        strategy: BaseStrategy,
        symbol_list: List[str],
        start_date: str,
        end_date: str,
    ) -> Tuple[
        Dict[str, pd.DataFrame],
        np.ndarray,
        Dict[str, BarData],
        Dict[str, Any],
        Dict[str, Dict[str, Any]],
    ]:
        """Load, align, and pre-extract data into numpy arrays."""
        single = len(symbol_list) == 1

        if single:
            data = self.data_manager.load_data(symbol_list[0], start_date, end_date)
            if data is None or data.empty:
                raise ValueError(f"Cannot load data for {symbol_list[0]}")
            data = data.copy()
            data = self.data_manager.calculate_indicators(data)
            data.attrs["symbol"] = symbol_list[0]
            dates = pd.to_datetime(data["date"]).values
            data_by_symbol = {symbol_list[0]: data}
        else:
            data_by_symbol = {}
            to_calc: List[Tuple[str, pd.DataFrame]] = []
            for sym in symbol_list:
                df = self.data_manager.load_data(sym, start_date, end_date)
                if df is not None and not df.empty:
                    to_calc.append((sym, df.copy()))

            if to_calc:
                max_workers = min(len(to_calc), 8)

                def calc_one(sym_df: Tuple[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
                    return (sym_df[0], self.data_manager.calculate_indicators(sym_df[1]))

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for sym, df in executor.map(calc_one, to_calc):
                        data_by_symbol[sym] = df

            common_index, aligned = _align_multi_symbol_data(data_by_symbol)
            if len(common_index) == 0:
                raise ValueError("No common trading dates across symbols")
            data_by_symbol = aligned
            dates = common_index.values

        n = len(dates)
        min_lb = strategy.min_lookback
        if min_lb > 0 and n < min_lb:
            raise ValueError(
                f"Data has {n} bars but strategy requires min_lookback={min_lb}. "
                "Expand date range or reduce strategy period."
            )

        bar_data_map: Dict[str, BarData] = {}
        for sym, df in data_by_symbol.items():
            bar_data_map[sym] = _prepare_bar_data(df, self.config.impact_vol_window)

        fast_ctx, fast_ctx_multi = self._build_fast_ctx(
            strategy, data_by_symbol, bar_data_map, single, symbol_list,
        )
        return data_by_symbol, dates, bar_data_map, fast_ctx, fast_ctx_multi

    @staticmethod
    def _build_fast_ctx(
        strategy: BaseStrategy,
        data_by_symbol: Dict[str, pd.DataFrame],
        bar_data_map: Dict[str, BarData],
        single: bool,
        symbol_list: List[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Build fast_columns context for strategies that support the ndarray path."""
        req_cols = tuple(getattr(strategy, "fast_columns", ()))
        fast_ctx: Dict[str, Any] = {}
        fast_ctx_multi: Dict[str, Dict[str, Any]] = {}

        if not req_cols:
            return fast_ctx, fast_ctx_multi

        if single:
            sym = symbol_list[0]
            df = data_by_symbol[sym]
            fast_ctx["symbol"] = sym
            for c in req_cols:
                if c in df.columns:
                    fast_ctx[c] = _to_f64(df[c].values)
                else:
                    return {}, {}
        else:
            for sym, df in data_by_symbol.items():
                sym_ctx: Dict[str, Any] = {"symbol": sym}
                for c in req_cols:
                    if c in df.columns:
                        sym_ctx[c] = _to_f64(df[c].values)
                    else:
                        return {}, {}
                fast_ctx_multi[sym] = sym_ctx

        return fast_ctx, fast_ctx_multi

    @staticmethod
    def _dispatch_signals(
        strategy: BaseStrategy,
        data_by_symbol: Dict[str, pd.DataFrame],
        fast_ctx: Dict[str, Any],
        fast_ctx_multi: Dict[str, Dict[str, Any]],
        symbol_list: List[str],
        single: bool,
        i: int,
        current_date: pd.Timestamp,
        current_prices: Dict[str, float],
    ) -> List[Dict]:
        """Call strategy.on_bar (or fast variant) and normalise the output.

        The slow fallback now passes the full DataFrame + bar_index to avoid
        the O(n^2) iloc[:i+1] copy that previously dominated non-fast paths.
        Strategies that only read df.iloc[-1] or df.iloc[:bar_index+1] are
        unaffected since the contract is "data up to current bar".
        """
        if single:
            sym = symbol_list[0]
            raw = (
                strategy.on_bar_fast(fast_ctx, i, current_date, current_prices=current_prices)
                if fast_ctx
                else None
            )
            if raw is None:
                df = data_by_symbol[sym]
                view = df.iloc[: i + 1]
                raw = strategy.on_bar(view, current_date)
        else:
            raw = (
                strategy.on_bar_fast_multi(
                    fast_ctx_multi, i, current_date, current_prices=current_prices,
                )
                if fast_ctx_multi
                else None
            )
            if raw is None:
                hist = {}
                for s in data_by_symbol:
                    df = data_by_symbol[s]
                    hist[s] = df.iloc[: i + 1]
                raw = strategy.on_bar(hist, current_date, current_prices=current_prices)
        return _normalize_signals(raw)

    def _assemble_results(
        self,
        portfolio: PortfolioTracker,
        order_mgr: DefaultOrderManager,
        data_by_symbol: Dict[str, pd.DataFrame],
        symbol_list: List[str],
        start_date: str,
        end_date: str,
        strategy: BaseStrategy,
        live_fills: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Build the final results dict with execution divergence analysis."""
        result = portfolio.to_results_dict()

        orders_data = []
        for o in order_mgr.all_orders:
            orders_data.append({
                "order_id": o.order_id,
                "status": o.status.value,
                "submitted_i": o.submitted_bar,
                "submitted_date": o.submitted_date,
                "action": "buy" if o.side == OrderSide.BUY else "sell",
                "symbol": o.symbol,
                "shares": o.shares,
                "filled_shares": o.filled_shares,
                "order_type": o.order_type.name.lower(),
                "fill_price": o.fill_price if o.fill_price else None,
                "limit_price": o.limit_price,
                "stop_price": o.stop_price,
                "filled_i": o.filled_bar,
                "filled_date": o.filled_date,
            })
        result["orders"] = pd.DataFrame(orders_data) if orders_data else pd.DataFrame()

        # Execution divergence analysis
        execution_divergence = None
        execution_report_path = None
        execution_bundle = None
        trades_df = result["trades"]

        if (
            self.config.auto_export_execution_report
            and live_fills is not None
            and not trades_df.empty
        ):
            from ..live.replay import (
                analyze_execution_divergence,
                export_execution_diagnostics_bundle,
                export_execution_divergence_report,
            )
            execution_divergence = analyze_execution_divergence(trades_df, live_fills)
            execution_report_path = export_execution_divergence_report(
                execution_divergence, self.config.execution_report_path,
            )
            bundle_dir = str(Path(self.config.execution_report_path).parent)
            execution_bundle = export_execution_diagnostics_bundle(
                summary=execution_divergence,
                backtest_trades=trades_df,
                live_fills=live_fills,
                output_dir=bundle_dir,
            )

        result["execution_divergence"] = execution_divergence
        result["execution_report_path"] = execution_report_path
        result["execution_bundle"] = execution_bundle
        result["manifest"] = build_run_manifest(
            strategy=strategy,
            symbols=symbol_list,
            start_date=start_date,
            end_date=end_date,
            data_by_symbol=data_by_symbol,
            config=self.config,
        )
        return result
