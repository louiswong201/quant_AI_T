"""Async trading runner — the central event loop for paper/live trading.

Ties together PriceFeed + Strategy/KernelAdapter + Broker + TradeJournal.
Publishes state updates via callbacks for the dashboard.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import platform
import signal
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..backtest.config import BacktestConfig
from ..broker.base import Broker
from .kernel_adapter import KernelAdapter, MultiTFAdapter
from .price_feed import BarEvent, PriceFeedManager, TickEvent
from .trade_journal import TradeJournal

logger = logging.getLogger(__name__)


class TradingRunner:
    """Async event loop that drives paper/live trading.

    On each new bar from PriceFeedManager:
      1. Update the rolling window
      2. Generate signals via KernelAdapter(s) or BaseStrategy
      3. Submit orders via Broker (with risk checks)
      4. Record trades and equity to TradeJournal
      5. Invoke dashboard update callback

    Between bars, real-time tick updates drive:
      - Stoploss / take-profit monitoring (matching kernel behaviour)
      - Live PnL & equity tracking for the dashboard
    """

    def __init__(
        self,
        feed: PriceFeedManager,
        broker: Broker,
        journal: TradeJournal,
        strategies: Dict[str, Union[KernelAdapter, MultiTFAdapter]],
        *,
        bt_config: Optional[BacktestConfig] = None,
        position_size_pct: float = 0.05,
        equity_snapshot_interval: float = 60.0,
        on_update: Optional[Callable[[], Any]] = None,
    ):
        self._feed = feed
        self._broker = broker
        self._journal = journal
        self._strategies = strategies
        self._bt_config = bt_config
        self._pos_size_pct = position_size_pct
        self._eq_interval = equity_snapshot_interval
        self._on_update = on_update
        self._running = False
        self._last_eq_snapshot = 0.0
        self._bar_count = 0
        self._tick_count = 0
        self._entry_prices: Dict[str, float] = {}
        self._live_prices: Dict[str, float] = {}
        self._sl_triggered: Dict[str, bool] = {}
        self._state_cache: Optional[Dict[str, Any]] = None
        self._state_cache_ts: float = 0.0
        self._STATE_CACHE_TTL: float = 1.5

    async def run(self, lookback: int = 200) -> None:
        self._running = True
        loop = asyncio.get_running_loop()
        if platform.system() != "Windows":
            for sig_name in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig_name, self._shutdown)
                except NotImplementedError:
                    pass
        else:
            atexit.register(self._shutdown)

        logger.info("Starting price feeds...")
        await self._feed.start_all(lookback=lookback)

        self._warmup_strategies()

        self._feed.on_bar(self._on_bar)
        self._feed.on_tick(self._on_tick)
        self._take_equity_snapshot()

        sl = self._bt_config.stop_loss_pct if self._bt_config else None
        tp = self._bt_config.take_profit_pct if self._bt_config else None
        logger.info(
            "Trading runner started. Tick monitoring: SL=%s TP=%s. Waiting for bars...",
            f"-{sl*100:.1f}%" if sl else "off",
            f"+{tp*100:.1f}%" if tp else "off",
        )
        await self._feed.run()

    def _warmup_strategies(self) -> None:
        """Run kernels on loaded historical windows to initialise positions
        before the first live bar arrives."""
        for symbol, adapter in self._strategies.items():
            if isinstance(adapter, MultiTFAdapter):
                windows: Dict[str, pd.DataFrame] = {}
                for iv in adapter.intervals:
                    w = self._feed.get_window(symbol, iv)
                    if w is not None and not w.empty:
                        windows[iv] = w
                adapter.warmup(windows, symbol)
            else:
                w = self._feed.get_window(symbol)
                if w is not None and not w.empty:
                    adapter.generate_signal(w, symbol)
                    logger.info(
                        "Warmup [%s] %s position=%+d",
                        symbol, adapter.name, adapter.get_position(),
                    )

    async def _on_bar(self, bar: BarEvent) -> None:
        if not self._running:
            return

        self._bar_count += 1
        symbol = bar.symbol

        adapter = self._strategies.get(symbol)
        if adapter is None:
            return

        if isinstance(adapter, MultiTFAdapter):
            window_df = self._feed.get_window(symbol, bar.interval)
            if window_df.empty:
                return
            arrs = self._feed.get_arrays(symbol, bar.interval)
            sig = adapter.on_bar(window_df, symbol, bar.interval, arrays=arrs)
        else:
            window_df = self._feed.get_window(symbol)
            if window_df.empty:
                return
            arrs = self._feed.get_arrays(symbol)
            sig = adapter.generate_signal(window_df, symbol, arrays=arrs)

        if sig is None:
            now = time.time()
            if now - self._last_eq_snapshot >= self._eq_interval:
                self._take_equity_snapshot()
            return

        shares = self._calc_shares(sig)
        if shares <= 0:
            return

        sig["shares"] = shares
        sig["price"] = bar.close

        self._journal.record_signal(
            symbol=symbol,
            signal_type=sig["action"],
            strategy=sig.get("strategy", ""),
            params={"price": bar.close},
        )

        result = self._broker.submit_order(sig)
        status = result.get("status", "rejected")

        if status == "filled":
            fill_price = result.get("fill_price", bar.close)
            commission = result.get("commission", 0.0)
            filled_shares = result.get("filled_shares", shares)

            pnl = 0.0
            positions = self._broker.get_positions()
            cur_qty = positions.get(symbol, 0)
            if sig["action"] == "buy":
                if symbol in self._entry_prices and self._entry_prices[symbol][1] < 0:
                    entry, _ = self._entry_prices.pop(symbol)
                    pnl = (entry - fill_price) * filled_shares - commission
                    if cur_qty > 0:
                        self._entry_prices[symbol] = (fill_price, cur_qty)
                elif symbol not in self._entry_prices:
                    self._entry_prices[symbol] = (fill_price, cur_qty)
                else:
                    old_entry, old_qty = self._entry_prices[symbol]
                    avg = (old_entry * old_qty + fill_price * filled_shares) / (old_qty + filled_shares)
                    self._entry_prices[symbol] = (avg, cur_qty)
            elif sig["action"] == "sell":
                if symbol in self._entry_prices and self._entry_prices[symbol][1] > 0:
                    entry, _ = self._entry_prices.pop(symbol)
                    pnl = (fill_price - entry) * filled_shares - commission
                    if cur_qty < 0:
                        self._entry_prices[symbol] = (fill_price, cur_qty)
                elif symbol not in self._entry_prices:
                    self._entry_prices[symbol] = (fill_price, cur_qty)
                else:
                    old_entry, old_qty = self._entry_prices[symbol]
                    avg = (old_entry * abs(old_qty) + fill_price * filled_shares) / (abs(old_qty) + filled_shares)
                    self._entry_prices[symbol] = (avg, cur_qty)

            self._record_pnl_to_broker(pnl)

            self._journal.record_trade(
                symbol=symbol,
                side=sig["action"],
                shares=filled_shares,
                price=fill_price,
                commission=commission,
                pnl=pnl,
                strategy=sig.get("strategy", ""),
            )
            logger.info(
                "%s %s %s @ %.4f (%.0f shares, pnl=%.2f)",
                sig["action"].upper(), symbol, sig.get("strategy", ""),
                fill_price, filled_shares, pnl,
            )
        else:
            logger.debug("Order rejected for %s: %s", symbol, result.get("message", ""))

        self._take_equity_snapshot()
        if self._on_update:
            try:
                result = self._on_update()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.debug("Update callback error: %s", e)

    async def _on_tick(self, tick: TickEvent) -> None:
        """Process real-time price updates between bar closes.

        Checks stoploss/take-profit using the bar's running low/high
        (matching the kernel's intra-bar O→H→L→C check) and updates
        live prices for the dashboard.
        """
        if not self._running:
            return

        self._tick_count += 1
        symbol = tick.symbol
        self._live_prices[symbol] = tick.price

        if self._bt_config is None:
            return

        positions = self._broker.get_positions()
        qty = positions.get(symbol, 0)
        if abs(qty) < 1e-12:
            return

        entry_data = self._entry_prices.get(symbol)
        if entry_data is None:
            return
        entry_price = entry_data[0] if isinstance(entry_data, tuple) else entry_data

        if self._sl_triggered.get(symbol):
            return

        sl_pct = self._bt_config.stop_loss_pct
        tp_pct = self._bt_config.take_profit_pct
        action = None
        trigger_price = tick.price
        reason = ""

        if qty > 0:
            if sl_pct and tick.running_low <= entry_price * (1.0 - sl_pct):
                action = "sell"
                trigger_price = entry_price * (1.0 - sl_pct)
                reason = "STOPLOSS"
            elif tp_pct and tick.running_high >= entry_price * (1.0 + tp_pct):
                action = "sell"
                trigger_price = entry_price * (1.0 + tp_pct)
                reason = "TAKE_PROFIT"
        elif qty < 0:
            if sl_pct and tick.running_high >= entry_price * (1.0 + sl_pct):
                action = "buy"
                trigger_price = entry_price * (1.0 + sl_pct)
                reason = "STOPLOSS"
            elif tp_pct and tick.running_low <= entry_price * (1.0 - tp_pct):
                action = "buy"
                trigger_price = entry_price * (1.0 - tp_pct)
                reason = "TAKE_PROFIT"

        if action is None:
            return

        self._sl_triggered[symbol] = True

        order_sig = {
            "action": action,
            "symbol": symbol,
            "shares": abs(qty),
            "price": trigger_price,
            "strategy": f"tick_{reason}",
        }
        result = self._broker.submit_order(order_sig)
        if result.get("status") == "filled":
            fill_price = result.get("fill_price", trigger_price)
            commission = result.get("commission", 0.0)
            filled = result.get("filled_shares", abs(qty))
            pnl = 0.0
            if qty > 0:
                pnl = (fill_price - entry_price) * filled - commission
            else:
                pnl = (entry_price - fill_price) * filled - commission

            self._record_pnl_to_broker(pnl)
            self._entry_prices.pop(symbol, None)
            self._sl_triggered.pop(symbol, None)

            self._journal.record_trade(
                symbol=symbol, side=action, shares=filled,
                price=fill_price, commission=commission, pnl=pnl,
                strategy=f"tick_{reason}",
            )
            logger.info(
                "[TICK %s] %s %s @ %.4f (%.4f shares, pnl=%.2f, entry=%.4f)",
                reason, action.upper(), symbol, fill_price, filled, pnl, entry_price,
            )
            self._take_equity_snapshot()
        else:
            self._sl_triggered.pop(symbol, None)

    def _record_pnl_to_broker(self, pnl: float) -> None:
        """Forward realized PnL to RiskManagedBroker's circuit breaker if available."""
        from .risk import RiskManagedBroker
        broker = self._broker
        if isinstance(broker, RiskManagedBroker) and broker._cb is not None:
            broker._cb.record_pnl(pnl)

    def _calc_shares(self, sig: Dict[str, Any]) -> float:
        price = float(sig.get("price", 0))
        if price <= 0:
            return 0.0
        cash = self._broker.get_cash()
        leverage = self._bt_config.leverage if self._bt_config else 1.0
        notional = cash * self._pos_size_pct * leverage
        raw = notional / price
        allow_frac = getattr(self._broker, "_allow_fractional", False)
        if not allow_frac:
            return float(max(1, int(raw)))
        return max(1e-8, raw)

    def _get_prices(self) -> Dict[str, float]:
        """Merge live tick prices with feed bar prices (tick takes priority)."""
        prices = self._feed.get_latest_prices()
        prices.update(self._live_prices)
        return prices

    def _take_equity_snapshot(self) -> None:
        prices = self._get_prices()
        positions = self._broker.get_positions()
        cash = self._broker.get_cash()
        equity = cash
        for sym, shares in positions.items():
            px = prices.get(sym, 0.0)
            equity += shares * px
        self._journal.record_equity(equity=equity, cash=cash, positions=positions)
        self._last_eq_snapshot = time.time()

    def _shutdown(self) -> None:
        logger.info("Shutdown signal received")
        self._running = False

    async def stop(self) -> None:
        self._running = False
        await self._feed.stop_all()
        self._journal.close()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def bar_count(self) -> int:
        return self._bar_count

    def get_state(self) -> Dict[str, Any]:
        now = time.time()
        if self._state_cache is not None and (now - self._state_cache_ts) < self._STATE_CACHE_TTL:
            return self._state_cache

        try:
            state = self._build_state()
        except Exception as e:
            logger.debug("get_state error: %s", e)
            if self._state_cache is not None:
                return self._state_cache
            state = self._build_state_minimal()

        self._state_cache = state
        self._state_cache_ts = now
        return state

    def _build_multi_tf_info(self) -> Dict[str, Any]:
        """Collect per-symbol multi-TF position states and strategy names."""
        tf_info: Dict[str, Any] = {}
        fusion_mode = ""
        for sym, adapter in self._strategies.items():
            if isinstance(adapter, MultiTFAdapter):
                tf_info[sym] = {
                    "positions": adapter.tf_positions,
                    "strategies": adapter.tf_strategies,
                    "intervals": adapter.intervals,
                    "fused_position": adapter.fused_position,
                }
                fusion_mode = adapter.mode
        return {"tf_positions": tf_info, "fusion_mode": fusion_mode}

    def _build_strategy_performance(
        self,
        initial_cash: float,
        total_return_pct: float,
        max_dd_pct: float,
        trade_stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build strategy-centric rows for leaderboard/scorecard/compare panels."""
        rows: List[Dict[str, Any]] = []

        # Baseline strategy rows from active adapters (always available).
        for sym, adapter in self._strategies.items():
            if isinstance(adapter, MultiTFAdapter):
                rows.append({
                    "strategy": f"Fusion[{adapter.mode}]",
                    "symbol": sym,
                    "kind": "fusion",
                    "return_pct": float(total_return_pct),
                    "max_drawdown_pct": float(max_dd_pct),
                    "win_rate": float(trade_stats.get("win_rate", 0.0)),
                    "profit_factor": float(trade_stats.get("profit_factor", 0.0)),
                    "trades": int(trade_stats.get("total_trades", 0)),
                    "sharpe": 0.0,
                    "calmar": float(total_return_pct / max(max_dd_pct, 1e-9)) if max_dd_pct > 0 else 0.0,
                    "stability": 0.0,
                    "score": 0.0,
                    "last_signal": "",
                    "direction": "MIXED",
                })
                for iv in adapter.intervals:
                    pos = int(adapter.tf_positions.get(iv, 0))
                    rows.append({
                        "strategy": f"{iv}:{adapter.tf_strategies.get(iv, 'Unknown')}",
                        "symbol": sym,
                        "kind": "component",
                        "return_pct": 0.0,
                        "max_drawdown_pct": 0.0,
                        "win_rate": 0.0,
                        "profit_factor": 0.0,
                        "trades": 0,
                        "sharpe": 0.0,
                        "calmar": 0.0,
                        "stability": 0.0,
                        "score": 0.0,
                        "last_signal": "",
                        "direction": "LONG" if pos > 0 else ("SHORT" if pos < 0 else "FLAT"),
                    })
            else:
                pos = int(adapter.get_position())
                rows.append({
                    "strategy": adapter.name,
                    "symbol": sym,
                    "kind": "single",
                    "return_pct": float(total_return_pct),
                    "max_drawdown_pct": float(max_dd_pct),
                    "win_rate": float(trade_stats.get("win_rate", 0.0)),
                    "profit_factor": float(trade_stats.get("profit_factor", 0.0)),
                    "trades": int(trade_stats.get("total_trades", 0)),
                    "sharpe": 0.0,
                    "calmar": float(total_return_pct / max(max_dd_pct, 1e-9)) if max_dd_pct > 0 else 0.0,
                    "stability": 0.0,
                    "score": 0.0,
                    "last_signal": "",
                    "direction": "LONG" if pos > 0 else ("SHORT" if pos < 0 else "FLAT"),
                })

        # Enrich with realized strategy stats from journal.
        strategy_stats = self._journal.get_strategy_trade_stats(limit=3000)
        by_symbol_name: Dict[tuple, Dict[str, Any]] = {}
        for r in rows:
            by_symbol_name[(r["symbol"], r["strategy"])] = r

        for st in strategy_stats:
            symbol = st.get("symbol", "")
            raw_name = st.get("strategy", "")
            # Map runtime strategy labels to UI rows:
            # - MultiTF[...] -> Fusion[mode]
            # - otherwise use raw strategy label.
            mapped = raw_name
            if raw_name.startswith("MultiTF["):
                end = raw_name.find("]")
                mode = raw_name[8:end] if end > 8 else "fusion"
                mapped = f"Fusion[{mode}]"
            key = (symbol, mapped)
            if key not in by_symbol_name:
                by_symbol_name[key] = {
                    "strategy": mapped,
                    "symbol": symbol,
                    "kind": "realized",
                    "direction": "FLAT",
                }

            row = by_symbol_name[key]
            pnl = float(st.get("total_pnl", 0.0))
            ret_pct = (pnl / initial_cash * 100.0) if initial_cash > 0 else 0.0
            dd_abs = float(st.get("max_dd_abs", 0.0))
            dd_pct = (dd_abs / initial_cash * 100.0) if initial_cash > 0 else 0.0
            sharpe = float(st.get("sharpe_proxy", 0.0))
            calmar = float(ret_pct / max(dd_pct, 1e-9)) if dd_pct > 0 else 0.0
            win_rate = float(st.get("win_rate", 0.0))
            pf = float(st.get("profit_factor", 0.0))
            trades = int(st.get("trades", 0))

            # A bounded composite score for quick ranking.
            score = (
                40.0 * np.tanh(sharpe / 3.0)
                + 25.0 * np.tanh(ret_pct / 30.0)
                + 20.0 * np.tanh((win_rate - 50.0) / 20.0)
                + 15.0 * np.tanh((pf - 1.0) / 1.2)
            )
            stability = max(0.0, 100.0 - dd_pct * 4.0)

            row.update({
                "return_pct": ret_pct,
                "max_drawdown_pct": dd_pct,
                "win_rate": win_rate,
                "profit_factor": pf,
                "trades": trades,
                "sharpe": sharpe,
                "calmar": calmar,
                "stability": stability,
                "score": float(score),
                "last_signal": st.get("last_timestamp", ""),
            })

        return list(by_symbol_name.values())

    def _build_state_minimal(self) -> Dict[str, Any]:
        """Fallback state with no DB queries."""
        prices = self._get_prices()
        positions = self._broker.get_positions()
        cash = self._broker.get_cash()
        equity = cash + sum(positions.get(s, 0) * prices.get(s, 0) for s in positions)
        state = {
            "running": self._running, "bar_count": self._bar_count,
            "tick_count": self._tick_count, "cash": cash, "equity": equity,
            "initial_cash": 100_000.0, "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0, "positions": positions,
            "position_details": [], "prices": prices,
            "symbols": self._feed.symbols, "trade_stats": {},
            "sl_tp_levels": {},
            "leverage": self._bt_config.leverage if self._bt_config else 1.0,
            "stop_loss_pct": self._bt_config.stop_loss_pct if self._bt_config else None,
            "take_profit_pct": self._bt_config.take_profit_pct if self._bt_config else None,
            "strategy_name": next(iter(self._strategies.values())).name if self._strategies else "",
            "strategy_performance": [],
        }
        state.update(self._build_multi_tf_info())
        return state

    def _build_state(self) -> Dict[str, Any]:
        prices = self._get_prices()
        positions = self._broker.get_positions()
        cash = self._broker.get_cash()
        equity = cash + sum(positions.get(s, 0) * prices.get(s, 0) for s in positions)

        initial_cash = getattr(self._broker, "_initial_cash_stored", None)
        if initial_cash is None:
            inner = getattr(self._broker, "_broker", None)
            if inner is not None:
                initial_cash = getattr(inner, "_initial_cash_stored", None)
        initial_cash = initial_cash or 100_000.0

        total_return_pct = ((equity - initial_cash) / initial_cash) * 100 if initial_cash else 0.0

        eq_curve = self._journal.get_equity_curve(limit=500)
        max_dd_pct = 0.0
        if not eq_curve.empty and "equity" in eq_curve.columns:
            eq_vals = eq_curve["equity"].values
            peak = eq_vals[0]
            for v in eq_vals:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak * 100 if peak > 0 else 0
                if dd > max_dd_pct:
                    max_dd_pct = dd

        position_details: List[Dict[str, Any]] = []
        for sym, qty in sorted(positions.items()):
            px = prices.get(sym, 0.0)
            mkt_val = qty * px
            entry_data = self._entry_prices.get(sym)
            entry_px = entry_data[0] if isinstance(entry_data, tuple) else (entry_data or 0.0)
            if qty > 0:
                unrealized = (px - entry_px) * qty
            elif qty < 0:
                unrealized = (entry_px - px) * abs(qty)
            else:
                unrealized = 0.0
            pnl_pct = ((px / entry_px - 1) * 100) if entry_px and qty > 0 else (
                ((entry_px / px - 1) * 100) if entry_px and qty < 0 else 0.0
            )
            weight = (mkt_val / equity * 100) if equity else 0.0
            position_details.append({
                "symbol": sym, "qty": qty, "entry_price": entry_px,
                "current_price": px, "market_value": mkt_val,
                "unrealized_pnl": unrealized, "pnl_pct": pnl_pct,
                "weight": weight, "side": "LONG" if qty > 0 else "SHORT",
            })

        trade_stats = self._journal.get_trade_stats()

        sl_pct = self._bt_config.stop_loss_pct if self._bt_config else None
        tp_pct = self._bt_config.take_profit_pct if self._bt_config else None
        sl_tp_levels: Dict[str, Dict[str, Optional[float]]] = {}
        for sym, qty in positions.items():
            entry_data = self._entry_prices.get(sym)
            if entry_data is None or abs(qty) < 1e-12:
                continue
            entry_px = entry_data[0] if isinstance(entry_data, tuple) else entry_data
            levels: Dict[str, Optional[float]] = {"entry": entry_px, "sl": None, "tp": None}
            if qty > 0:
                if sl_pct:
                    levels["sl"] = entry_px * (1.0 - sl_pct)
                if tp_pct:
                    levels["tp"] = entry_px * (1.0 + tp_pct)
            elif qty < 0:
                if sl_pct:
                    levels["sl"] = entry_px * (1.0 + sl_pct)
                if tp_pct:
                    levels["tp"] = entry_px * (1.0 - tp_pct)
            sl_tp_levels[sym] = levels

        state = {
            "running": self._running,
            "bar_count": self._bar_count,
            "tick_count": self._tick_count,
            "cash": cash,
            "equity": equity,
            "initial_cash": initial_cash,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd_pct,
            "positions": positions,
            "position_details": position_details,
            "prices": prices,
            "symbols": self._feed.symbols,
            "trade_stats": trade_stats,
            "sl_tp_levels": sl_tp_levels,
            "leverage": self._bt_config.leverage if self._bt_config else 1.0,
            "stop_loss_pct": self._bt_config.stop_loss_pct if self._bt_config else None,
            "take_profit_pct": self._bt_config.take_profit_pct if self._bt_config else None,
            "strategy_name": next(iter(self._strategies.values())).name if self._strategies else "",
        }
        state.update(self._build_multi_tf_info())
        state["strategy_performance"] = self._build_strategy_performance(
            initial_cash=initial_cash,
            total_return_pct=total_return_pct,
            max_dd_pct=max_dd_pct,
            trade_stats=trade_stats,
        )
        return state
