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
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..backtest.config import BacktestConfig
from ..broker.base import Broker
from ..broker.live_order_manager import LiveOrderManager
from ..features.online_engine import FeatureSnapshot, OnlineFeatureEngine
from .audit import AuditTrail
from .events import InMemoryEventBus
from .kill_switch import KillSwitch
from .kernel_adapter import KernelAdapter, MultiTFAdapter
from .price_feed import BarEvent, PriceFeedManager, TickEvent
from .risk import RiskManagedBroker
from .runtime_slo import LiveRuntimeSLO, RuntimeMetrics
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
        symbol_configs: Optional[Dict[str, BacktestConfig]] = None,
        position_size_pct: float = 0.05,
        symbol_size_overrides: Optional[Dict[str, float]] = None,
        equity_snapshot_interval: float = 60.0,
        on_update: Optional[Callable[[], Any]] = None,
        order_manager: Optional[LiveOrderManager] = None,
        audit_trail: Optional[AuditTrail] = None,
        event_bus: Optional[InMemoryEventBus] = None,
        feature_engine: Optional[OnlineFeatureEngine] = None,
        slo: Optional[LiveRuntimeSLO] = None,
    ):
        self._feed = feed
        self._broker = broker
        self._journal = journal
        self._strategies = strategies
        self._bt_config = bt_config
        self._symbol_configs = symbol_configs or {}
        self._pos_size_pct = position_size_pct
        self._symbol_size_overrides = symbol_size_overrides or {}
        self._eq_interval = equity_snapshot_interval
        self._on_update = on_update
        self._order_manager = order_manager or LiveOrderManager()
        self._audit = audit_trail
        self._event_bus = event_bus
        self._feature_engine = feature_engine or OnlineFeatureEngine()
        self._runtime_metrics = RuntimeMetrics(slo=slo)
        self._running = False
        self._last_eq_snapshot = 0.0
        self._bar_count = 0
        self._tick_count = 0
        self._entry_prices: Dict[str, tuple[float, float]] = {}
        self._live_prices: Dict[str, float] = {}
        self._sl_triggered: Dict[str, bool] = {}
        self._state_cache: Optional[Dict[str, Any]] = None
        self._state_cache_ts: float = 0.0
        self._STATE_CACHE_TTL: float = 1.5
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._kill_switch_active = False
        self._kill_switch_reason = ""
        self._kill_switch_triggered_at = ""
        self._kill_switch_in_progress = False
        self._recent_trades: Deque[Dict[str, Any]] = deque(maxlen=5000)
        self._recent_signals: Deque[Dict[str, Any]] = deque(maxlen=5000)
        self._equity_points: Deque[Dict[str, Any]] = deque(maxlen=5000)
        self._daily_pnl: Dict[str, Dict[str, Any]] = {}
        self._feature_snapshots: Dict[str, Dict[str, Any]] = {}
        self._initial_cash = float(getattr(getattr(self._broker, "_broker", self._broker), "_initial_cash_stored", 100_000.0) or 100_000.0)
        self._last_daily_reset_date: Optional[str] = None
        self._hydrate_read_models()

    async def run(self, lookback: int = 200) -> None:
        self._running = True
        loop = asyncio.get_running_loop()
        self._loop = loop
        if self._event_bus is not None:
            await self._event_bus.start()
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
        self._sync_strategy_positions_to_broker()

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

    def _hydrate_read_models(self) -> None:
        try:
            eq_curve = self._journal.get_equity_curve(limit=2000)
            for _, row in eq_curve.iterrows():
                self._equity_points.append(
                    {
                        "timestamp": pd.Timestamp(row["timestamp"]).isoformat() if "timestamp" in row else "",
                        "equity": float(row.get("equity", 0.0)),
                        "cash": float(row.get("cash", 0.0)),
                        "positions": row.get("positions", "{}"),
                    }
                )
        except Exception:
            pass
        try:
            trades = self._journal.get_trades(limit=2000)
            if not trades.empty:
                for _, row in trades.iloc[::-1].iterrows():
                    self._recent_trades.append(dict(row))
                    day = str(pd.Timestamp(row["timestamp"]).date()) if "timestamp" in row else ""
                    bucket = self._daily_pnl.setdefault(day, {"date": day, "daily_pnl": 0.0, "n_trades": 0})
                    bucket["daily_pnl"] += float(row.get("pnl", 0.0))
                    bucket["n_trades"] += 1
        except Exception:
            pass
        try:
            signals = self._journal.get_signals(limit=2000)
            if not signals.empty:
                for _, row in signals.iloc[::-1].iterrows():
                    self._recent_signals.append(dict(row))
        except Exception:
            pass

    def _publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self._event_bus is not None:
            self._event_bus.publish_nowait(event_type, payload)

    def _record_signal_read_model(
        self,
        *,
        symbol: str,
        signal_type: str,
        strategy: str,
        params: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        self._recent_signals.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "strategy": strategy,
                "signal_type": signal_type,
                "params": params,
            }
        )

    def _record_trade_read_model(
        self,
        *,
        symbol: str,
        side: str,
        shares: float,
        price: float,
        commission: float,
        pnl: float,
        strategy: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        row = {
            "timestamp": ts,
            "symbol": symbol,
            "side": side,
            "shares": shares,
            "price": price,
            "commission": commission,
            "pnl": pnl,
            "strategy": strategy,
            "metadata": metadata or {},
        }
        self._recent_trades.append(row)
        day = str(pd.Timestamp(ts).date())
        bucket = self._daily_pnl.setdefault(day, {"date": day, "daily_pnl": 0.0, "n_trades": 0})
        bucket["daily_pnl"] += float(pnl)
        bucket["n_trades"] += 1

    def _record_equity_read_model(
        self,
        *,
        equity: float,
        cash: float,
        positions: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        self._equity_points.append(
            {
                "timestamp": ts,
                "equity": float(equity),
                "cash": float(cash),
                "positions": dict(positions),
            }
        )

    async def _submit_signal(
        self,
        signal: Dict[str, Any],
        *,
        signal_ts: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        submit_started = time.perf_counter()
        self._publish_event("order.command", {"signal": dict(signal)})
        result = await self._order_manager.submit(self._broker, signal)
        self._runtime_metrics.signal_to_submit.record_ms((time.perf_counter() - submit_started) * 1000.0)
        status = str(result.get("status", "")).lower()
        if status == "rejected":
            self._runtime_metrics.record_order_reject()
        elif status == "error":
            self._runtime_metrics.record_order_error()
        elif status == "timeout":
            self._runtime_metrics.record_order_timeout()
        if self._audit is not None:
            try:
                fill_price = result.get("fill_price")
                signal_price = signal.get("price")
                slippage_bps = None
                if fill_price and signal_price:
                    signal_price = float(signal_price)
                    fill_price = float(fill_price)
                    if signal_price > 0:
                        slippage_bps = (fill_price - signal_price) / signal_price * 10000.0
                self._audit.record_order_lifecycle(
                    internal_id=str(result.get("order_id") or f"{signal.get('symbol','')}-{time.time_ns()}"),
                    exchange_order_id=str(result.get("exchange_order_id", "")),
                    signal_ts=signal_ts or datetime.now(timezone.utc),
                    submit_ts=signal_ts or datetime.now(timezone.utc),
                    ack_ts=datetime.now(timezone.utc),
                    fill_ts=datetime.now(timezone.utc) if status in {"filled", "partial"} else None,
                    fill_price=float(fill_price) if fill_price else None,
                    signal_price=float(signal_price) if signal_price else None,
                    latency_ms=float(result.get("latency_ms", 0.0) or 0.0),
                    slippage_bps=slippage_bps,
                )
            except Exception:
                logger.debug("audit record failure", exc_info=True)
        self._publish_event(f"order.{status or 'unknown'}", {"signal": dict(signal), "result": dict(result)})
        return result

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

    def _maybe_reset_daily_circuit_breaker(self, bar_ts: datetime) -> None:
        today = bar_ts.strftime("%Y-%m-%d")
        if today != self._last_daily_reset_date:
            self._last_daily_reset_date = today
            broker = self._broker
            cb = getattr(broker, "_cb", None)
            if cb is not None:
                cb.reset_daily()
                logger.info("CircuitBreaker daily PnL reset for %s", today)

    async def _on_bar(self, bar: BarEvent) -> None:
        if not self._running:
            return

        self._maybe_reset_daily_circuit_breaker(bar.timestamp)
        self._bar_count += 1
        symbol = bar.symbol
        bar_started = time.perf_counter()

        adapter = self._strategies.get(symbol)
        if adapter is None:
            return

        if isinstance(adapter, MultiTFAdapter):
            window_df = self._feed.get_window(symbol, bar.interval)
            if window_df.empty:
                return
            arrs = self._feed.get_arrays(symbol, bar.interval)
            feature_snap = self._feature_engine.update(symbol, bar.interval or "", arrs, event_time=bar.timestamp)
            self._feature_snapshots[f"{symbol}:{bar.interval or 'default'}"] = {
                "timestamp": feature_snap.timestamp,
                "feature_set_version": feature_snap.feature_set_version,
                "values": dict(feature_snap.values),
            }
            self._runtime_metrics.feature_freshness.record_ms(
                max(0.0, (time.time() - bar.timestamp.timestamp()) * 1000.0)
            )
            self._publish_event("feature.update", {"symbol": symbol, "interval": bar.interval, "snapshot": dict(feature_snap.values)})
            sig = adapter.on_bar(window_df, symbol, bar.interval, arrays=arrs)
        else:
            window_df = self._feed.get_window(symbol)
            if window_df.empty:
                return
            arrs = self._feed.get_arrays(symbol)
            cfg = self._get_config(symbol)
            feature_interval = bar.interval or (cfg.interval if cfg else "")
            feature_snap = self._feature_engine.update(symbol, feature_interval, arrs, event_time=bar.timestamp)
            self._feature_snapshots[f"{symbol}:{bar.interval or 'default'}"] = {
                "timestamp": feature_snap.timestamp,
                "feature_set_version": feature_snap.feature_set_version,
                "values": dict(feature_snap.values),
            }
            self._runtime_metrics.feature_freshness.record_ms(
                max(0.0, (time.time() - bar.timestamp.timestamp()) * 1000.0)
            )
            self._publish_event("feature.update", {"symbol": symbol, "interval": bar.interval, "snapshot": dict(feature_snap.values)})
            sig = adapter.generate_signal(window_df, symbol, arrays=arrs)

        self._runtime_metrics.bar_to_signal.record_ms((time.perf_counter() - bar_started) * 1000.0)

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
        sig["feature_set_version"] = self._feature_engine.feature_set_version

        self._record_signal_read_model(
            symbol=symbol,
            signal_type=sig["action"],
            strategy=sig.get("strategy", ""),
            params={"price": bar.close},
            timestamp=bar.timestamp,
        )
        self._journal.record_signal(
            symbol=symbol,
            signal_type=sig["action"],
            strategy=sig.get("strategy", ""),
            params={"price": bar.close, "feature_set_version": self._feature_engine.feature_set_version},
            timestamp=bar.timestamp,
        )
        self._publish_event("signal.intent", {"signal": dict(sig), "timestamp": bar.timestamp.isoformat()})

        result = await self._submit_signal(sig, signal_ts=bar.timestamp)
        status = result.get("status", "rejected")

        if status == "filled":
            fill_price = result.get("fill_price", bar.close)
            commission = result.get("commission", 0.0)
            filled_shares = result.get("filled_shares", shares)

            pnl = 0.0
            positions = self._broker.get_positions()
            cur_qty = positions.get(symbol, 0)
            prior_entry = self._entry_prices.get(symbol)
            was_realized_close = False
            if sig["action"] == "buy":
                was_realized_close = prior_entry is not None and prior_entry[1] < 0
                if symbol in self._entry_prices and self._entry_prices[symbol][1] < 0:
                    entry, prior_qty = self._entry_prices[symbol]
                    pnl = (entry - fill_price) * filled_shares - commission
                    if cur_qty > 0:
                        self._entry_prices.pop(symbol, None)
                        self._entry_prices[symbol] = (fill_price, cur_qty)
                    elif cur_qty < 0:
                        self._entry_prices[symbol] = (entry, cur_qty)
                    else:
                        self._entry_prices.pop(symbol, None)
                elif symbol not in self._entry_prices:
                    self._entry_prices[symbol] = (fill_price, cur_qty)
                else:
                    old_entry, old_qty = self._entry_prices[symbol]
                    avg = (old_entry * old_qty + fill_price * filled_shares) / (old_qty + filled_shares)
                    self._entry_prices[symbol] = (avg, cur_qty)
            elif sig["action"] == "sell":
                was_realized_close = prior_entry is not None and prior_entry[1] > 0
                if symbol in self._entry_prices and self._entry_prices[symbol][1] > 0:
                    entry, prior_qty = self._entry_prices[symbol]
                    pnl = (fill_price - entry) * filled_shares - commission
                    if cur_qty < 0:
                        self._entry_prices.pop(symbol, None)
                        self._entry_prices[symbol] = (fill_price, cur_qty)
                    elif cur_qty > 0:
                        self._entry_prices[symbol] = (entry, cur_qty)
                    else:
                        self._entry_prices.pop(symbol, None)
                elif symbol not in self._entry_prices:
                    self._entry_prices[symbol] = (fill_price, cur_qty)
                else:
                    old_entry, old_qty = self._entry_prices[symbol]
                    avg = (old_entry * abs(old_qty) + fill_price * filled_shares) / (abs(old_qty) + filled_shares)
                    self._entry_prices[symbol] = (avg, cur_qty)

            self._record_pnl_to_broker(pnl)

            trade_metadata = {"realized": was_realized_close, "position_after": cur_qty}
            self._record_trade_read_model(
                symbol=symbol,
                side=sig["action"],
                shares=filled_shares,
                price=fill_price,
                commission=commission,
                pnl=pnl,
                strategy=sig.get("strategy", ""),
                metadata=trade_metadata,
                timestamp=bar.timestamp,
            )
            self._journal.record_trade(
                symbol=symbol,
                side=sig["action"],
                shares=filled_shares,
                price=fill_price,
                commission=commission,
                pnl=pnl,
                strategy=sig.get("strategy", ""),
                metadata=trade_metadata,
                timestamp=bar.timestamp,
            )
            self._publish_event(
                "order.fill",
                {
                    "symbol": symbol,
                    "side": sig["action"],
                    "shares": filled_shares,
                    "price": fill_price,
                    "commission": commission,
                    "pnl": pnl,
                    "strategy": sig.get("strategy", ""),
                },
            )
            logger.info(
                "%s %s %s @ %.4f (%.0f shares, pnl=%.2f)",
                sig["action"].upper(), symbol, sig.get("strategy", ""),
                fill_price, filled_shares, pnl,
            )
        else:
            logger.debug("Order rejected for %s: %s", symbol, result.get("message", ""))

        self._take_equity_snapshot()
        self._invalidate_state_cache()
        await self._run_update_callback()

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
        self._runtime_metrics.feed_lag.record_ms(max(0.0, (time.time() - tick.timestamp.timestamp()) * 1000.0))

        cfg = self._get_config(symbol)
        if cfg is None:
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

        sl_pct = cfg.stop_loss_pct
        tp_pct = cfg.take_profit_pct
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
            "feature_set_version": self._feature_engine.feature_set_version,
        }
        self._record_signal_read_model(
            symbol=symbol,
            signal_type=action,
            strategy=f"tick_{reason}",
            params={"price": trigger_price, "reason": reason},
            timestamp=tick.timestamp,
        )
        self._journal.record_signal(
            symbol=symbol,
            signal_type=action,
            strategy=f"tick_{reason}",
            params={"price": trigger_price, "reason": reason, "feature_set_version": self._feature_engine.feature_set_version},
            timestamp=tick.timestamp,
        )
        result = await self._submit_signal(order_sig, signal_ts=tick.timestamp)
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

            trade_metadata = {"realized": True, "position_after": 0.0}
            self._record_trade_read_model(
                symbol=symbol,
                side=action,
                shares=filled,
                price=fill_price,
                commission=commission,
                pnl=pnl,
                strategy=f"tick_{reason}",
                metadata=trade_metadata,
                timestamp=tick.timestamp,
            )
            self._journal.record_trade(
                symbol=symbol, side=action, shares=filled,
                price=fill_price, commission=commission, pnl=pnl,
                strategy=f"tick_{reason}",
                metadata=trade_metadata,
                timestamp=tick.timestamp,
            )
            logger.info(
                "[TICK %s] %s %s @ %.4f (%.4f shares, pnl=%.2f, entry=%.4f)",
                reason, action.upper(), symbol, fill_price, filled, pnl, entry_price,
            )
            self._take_equity_snapshot()
            self._invalidate_state_cache()
            await self._run_update_callback()
        else:
            self._sl_triggered.pop(symbol, None)

    def _record_pnl_to_broker(self, pnl: float) -> None:
        """Forward realized PnL to RiskManagedBroker's circuit breaker if available."""
        from .risk import RiskManagedBroker
        broker = self._broker
        if isinstance(broker, RiskManagedBroker) and broker._cb is not None:
            broker._cb.record_pnl(pnl)

    def _get_config(self, symbol: str) -> Optional[BacktestConfig]:
        """Per-symbol config with fallback to the global bt_config."""
        return self._symbol_configs.get(symbol, self._bt_config)

    def _calc_shares(self, sig: Dict[str, Any]) -> float:
        price = float(sig.get("price", 0))
        if price <= 0:
            return 0.0

        symbol = sig.get("symbol", "")
        intent = sig.get("intent", "open")
        positions = self._broker.get_positions()
        held_qty = float(positions.get(symbol, 0))
        allow_frac = getattr(self._broker, "_allow_fractional", False)

        if intent == "close":
            return abs(held_qty) if abs(held_qty) > 1e-12 else 0.0

        cfg = self._get_config(symbol)
        leverage = cfg.leverage if cfg else 1.0
        size_pct = self._symbol_size_overrides.get(symbol, self._pos_size_pct)

        equity = self._broker.get_cash()
        for sym, qty in positions.items():
            px = self._live_prices.get(sym) or self._feed.get_latest_prices().get(sym, 0.0)
            equity += float(qty) * px
        if equity <= 0:
            return 0.0

        new_open_notional = equity * size_pct * leverage
        new_open_shares = new_open_notional / price

        if intent == "reverse":
            total = abs(held_qty) + new_open_shares
        else:
            total = new_open_shares

        if not allow_frac:
            return float(max(1, int(total)))
        return max(1e-8, total)

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
        self._record_equity_read_model(equity=equity, cash=cash, positions=positions)
        self._journal.record_equity(equity=equity, cash=cash, positions=positions)
        self._publish_event("journal.equity_snapshot", {"equity": equity, "cash": cash, "positions": positions})
        self._last_eq_snapshot = time.time()
        self._runtime_metrics.set_queue_depth(int(self._journal.get_write_metrics().get("pending_rows", 0)))

    def _sync_strategy_positions_to_broker(self) -> None:
        """Align adapter runtime states to the recovered broker book."""
        positions = self._broker.get_positions()
        for symbol, adapter in self._strategies.items():
            qty = float(positions.get(symbol, 0.0))
            direction = 1 if qty > 0 else (-1 if qty < 0 else 0)
            if isinstance(adapter, MultiTFAdapter):
                adapter.set_fused_position(direction)
            else:
                adapter.set_position(direction)

    def restore_from_journal(self) -> Dict[str, Any]:
        """Restore broker/account state from the persistent journal."""
        fallback_initial = 100_000.0
        inner = getattr(self._broker, "_broker", None)
        if inner is not None:
            fallback_initial = float(getattr(inner, "_initial_cash_stored", fallback_initial) or fallback_initial)
        else:
            fallback_initial = float(getattr(self._broker, "_initial_cash_stored", fallback_initial) or fallback_initial)

        restored = self._journal.get_latest_account_state(fallback_initial_cash=fallback_initial)
        if not restored.get("has_recovery_data"):
            return restored

        restore_target = inner if inner is not None else self._broker
        restore_fn = getattr(restore_target, "restore_state", None)
        if callable(restore_fn):
            restore_fn(
                cash=float(restored.get("cash", fallback_initial)),
                positions=restored.get("positions", {}),
                entry_prices=restored.get("entry_prices", {}),
                initial_cash=float(restored.get("initial_cash", fallback_initial)),
            )

        self._entry_prices = {
            sym: (float(price), float(restored["positions"][sym]))
            for sym, price in restored.get("entry_prices", {}).items()
            if sym in restored.get("positions", {})
        }
        self._initial_cash = float(restored.get("initial_cash", self._initial_cash))
        self._sync_strategy_positions_to_broker()
        self._invalidate_state_cache()
        logger.info(
            "Recovered account state: cash=%.2f positions=%d snapshot=%s",
            float(restored.get("cash", 0.0)),
            len(restored.get("positions", {})),
            restored.get("snapshot_timestamp", "") or "n/a",
        )
        return restored

    def _shutdown(self) -> None:
        logger.info("Shutdown signal received")
        self._running = False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._feed.stop_all())
                if self._event_bus is not None:
                    loop.create_task(self._event_bus.stop())
        except Exception:
            pass
        try:
            self._journal.close()
        except Exception:
            pass
        try:
            if self._audit is not None:
                self._audit.close()
        except Exception:
            pass

    def _invalidate_state_cache(self) -> None:
        self._state_cache = None
        self._state_cache_ts = 0.0

    async def _run_update_callback(self) -> None:
        if not self._on_update:
            return
        try:
            loop = asyncio.get_event_loop()
            result = self._on_update()
            if asyncio.iscoroutine(result):
                await result
            elif callable(result):
                await loop.run_in_executor(None, result)
        except Exception as e:
            logger.debug("Update callback error: %s", e)

    async def _activate_kill_switch(self, reason: str) -> Dict[str, Any]:
        if self._kill_switch_in_progress:
            return {"status": "busy", "message": "kill switch already running"}
        if self._kill_switch_active:
            return {"status": "already_triggered", "message": self._kill_switch_reason}

        self._kill_switch_in_progress = True
        self._running = False
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        self._kill_switch_triggered_at = datetime.now(timezone.utc).isoformat()
        self._invalidate_state_cache()

        positions_before = {
            sym: qty for sym, qty in self._broker.get_positions().items()
            if abs(float(qty)) > 1e-10
        }

        try:
            closed = await KillSwitch(self._broker).flatten_all(reason)
            for item in closed:
                result = item.get("result", {}) or {}
                if result.get("status") != "filled":
                    continue

                symbol = str(item.get("symbol", ""))
                prior_qty = float(positions_before.get(symbol, 0.0))
                if abs(prior_qty) <= 1e-10:
                    continue

                fill_price = float(result.get("fill_price", 0.0) or 0.0)
                filled_shares = float(result.get("filled_shares", abs(prior_qty)) or abs(prior_qty))
                commission = float(result.get("commission", 0.0) or 0.0)

                entry_data = self._entry_prices.get(symbol)
                entry_price = entry_data[0] if isinstance(entry_data, tuple) else float(entry_data or 0.0)
                pnl = 0.0
                if fill_price > 0 and entry_price > 0:
                    if prior_qty > 0:
                        pnl = (fill_price - entry_price) * filled_shares - commission
                    else:
                        pnl = (entry_price - fill_price) * filled_shares - commission

                self._record_pnl_to_broker(pnl)
                self._journal.record_trade(
                    symbol=symbol,
                    side=str(item.get("side", "")),
                    shares=filled_shares,
                    price=fill_price,
                    commission=commission,
                    pnl=pnl,
                    strategy="kill_switch",
                    metadata={"realized": True, "position_after": 0.0},
                )
                self._entry_prices.pop(symbol, None)
                self._sl_triggered.pop(symbol, None)

            if isinstance(self._broker, RiskManagedBroker) and self._broker._cb is not None:
                self._broker._cb.trip(f"Manual kill switch: {reason}")

            self._take_equity_snapshot()
            self._invalidate_state_cache()
            await self._run_update_callback()
            logger.critical("KILL SWITCH ACTIVATED: %s", reason)
            return {
                "status": "triggered",
                "message": reason,
                "closed_positions": len(closed),
            }
        except Exception as e:
            logger.exception("Kill switch activation failed: %s", e)
            self._kill_switch_reason = f"{reason} (error: {e})"
            self._invalidate_state_cache()
            return {"status": "error", "message": str(e)}
        finally:
            self._kill_switch_in_progress = False

    def activate_kill_switch(self, reason: str = "Manual dashboard activation") -> Dict[str, Any]:
        """Thread-safe kill switch entrypoint for the dashboard/UI layer.

        Uses a short poll loop so the calling thread (Dash callback) is not
        blocked for up to 15 s — it checks every 0.25 s and returns as soon
        as the coroutine finishes.
        """
        if self._loop is not None and self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._activate_kill_switch(reason), self._loop)
            try:
                return fut.result(timeout=15)
            except TimeoutError:
                return {"status": "timeout", "message": "kill switch timed out after 15s"}
        return asyncio.run(self._activate_kill_switch(reason))

    async def stop(self) -> None:
        self._running = False
        await self._feed.stop_all()
        if self._event_bus is not None:
            await self._event_bus.stop()
        self._journal.close()
        if self._audit is not None:
            self._audit.close()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def bar_count(self) -> int:
        return self._bar_count

    def get_equity_curve(self, limit: int = 1000) -> pd.DataFrame:
        rows = list(self._equity_points)[-limit:]
        if not rows:
            return pd.DataFrame(columns=["timestamp", "equity", "cash", "positions"])
        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_recent_trades(self, limit: int = 100) -> pd.DataFrame:
        rows = list(self._recent_trades)[-limit:]
        if not rows:
            return pd.DataFrame(columns=["timestamp", "symbol", "side", "shares", "price", "commission", "pnl", "strategy"])
        df = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
        if "metadata" in df.columns:
            df = df.drop(columns=["metadata"])
        return df

    def get_daily_pnl(self, days: int = 30) -> pd.DataFrame:
        rows = sorted(self._daily_pnl.values(), key=lambda x: x["date"], reverse=True)[:days]
        return pd.DataFrame(rows)

    def _compute_trade_stats(self) -> Dict[str, Any]:
        pnls: List[float] = []
        commissions: List[float] = []
        for row in self._recent_trades:
            metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
            realized = metadata.get("realized", abs(float(row.get("pnl", 0.0) or 0.0)) > 1e-12)
            if realized:
                pnls.append(float(row.get("pnl", 0.0) or 0.0))
                commissions.append(float(row.get("commission", 0.0) or 0.0))
        if not pnls:
            return {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "breakeven_trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "largest_win": 0.0, "largest_loss": 0.0, "profit_factor": 0.0,
                "profit_factor_unbounded": False, "profit_factor_capped": False,
                "total_pnl": 0.0, "total_commission": 0.0, "expectancy": 0.0,
                "win_streak": 0, "loss_streak": 0,
            }
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        breakeven = [p for p in pnls if abs(p) <= 1e-12]
        total_win = sum(wins) if wins else 0.0
        total_loss = abs(sum(losses)) if losses else 0.0
        pf_unbounded = total_loss <= 1e-12 and total_win > 0
        if total_loss > 1e-12:
            pf_raw = total_win / total_loss
            profit_factor = min(pf_raw, 99.99)
            pf_capped = pf_raw > 99.99
        else:
            profit_factor = 99.99 if total_win > 0 else 0.0
            pf_capped = pf_unbounded
        max_w_streak = max_l_streak = cur_w = cur_l = 0
        for p in pnls:
            if p > 0:
                cur_w += 1
                cur_l = 0
                max_w_streak = max(max_w_streak, cur_w)
            elif p < 0:
                cur_l += 1
                cur_w = 0
                max_l_streak = max(max_l_streak, cur_l)
            else:
                cur_w = cur_l = 0
        return {
            "total_trades": len(pnls),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "breakeven_trades": len(breakeven),
            "win_rate": len(wins) / (len(wins) + len(losses)) * 100 if (len(wins) + len(losses)) else 0.0,
            "avg_win": total_win / len(wins) if wins else 0.0,
            "avg_loss": -(total_loss / len(losses)) if losses else 0.0,
            "largest_win": max(wins) if wins else 0.0,
            "largest_loss": min(losses) if losses else 0.0,
            "profit_factor": profit_factor,
            "profit_factor_unbounded": pf_unbounded,
            "profit_factor_capped": pf_capped,
            "total_pnl": float(sum(pnls)),
            "total_commission": float(sum(commissions)),
            "expectancy": float(sum(pnls) / len(pnls)),
            "win_streak": max_w_streak,
            "loss_streak": max_l_streak,
        }

    def _compute_strategy_trade_stats(self) -> List[Dict[str, Any]]:
        if not self._recent_trades:
            return []
        rows = []
        df = pd.DataFrame(list(self._recent_trades))
        if df.empty:
            return rows
        if "metadata" in df.columns:
            df["metadata"] = df["metadata"].apply(lambda x: x if isinstance(x, dict) else {})
            df["realized"] = df["metadata"].apply(lambda x: bool(x.get("realized", False)))
            df = df[df["realized"]]
        if df.empty:
            return rows
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for (symbol, strategy), grp in df.groupby(["symbol", "strategy"], dropna=False):
            pnl_series = grp["pnl"].astype(float)
            wins = pnl_series[pnl_series > 0]
            losses = pnl_series[pnl_series < 0]
            total_win = float(wins.sum()) if not wins.empty else 0.0
            total_loss = abs(float(losses.sum())) if not losses.empty else 0.0
            pf = 99.99 if total_loss <= 1e-12 and total_win > 0 else (total_win / total_loss if total_loss > 1e-12 else 0.0)
            cum = pnl_series.cumsum()
            peak = cum.cummax()
            dd = float((peak - cum).max()) if not cum.empty else 0.0
            std = float(pnl_series.std(ddof=0)) if len(pnl_series) > 1 else 0.0
            expectancy = float(pnl_series.mean()) if len(pnl_series) else 0.0
            sharpe_proxy = expectancy / std if std > 1e-12 else 0.0
            rows.append({
                "symbol": symbol,
                "strategy": strategy,
                "trades": int(len(grp)),
                "win_rate": float((pnl_series > 0).mean() * 100.0) if len(pnl_series) else 0.0,
                "profit_factor": float(min(pf, 99.99)),
                "total_pnl": float(pnl_series.sum()),
                "expectancy": expectancy,
                "sharpe_proxy": sharpe_proxy,
                "max_dd_abs": dd,
                "last_timestamp": grp["timestamp"].max().isoformat() if grp["timestamp"].notna().any() else "",
            })
        return rows

    def get_health(self) -> Dict[str, Any]:
        feed_health = self._safe_feed_health()
        broker_summary = self._broker.get_account_summary()
        positions = self._broker.get_positions()
        metrics_health = self._runtime_metrics.health()
        return {
            "feed_healthy": bool(feed_health.get("feed_healthy", False)),
            "broker_connected": True,
            "open_positions": len([qty for qty in positions.values() if abs(float(qty)) > 1e-12]),
            "margin_ratio": float(getattr(self._broker, "get_margin_ratio", lambda: 1.0)()),
            "kill_switch_active": self._kill_switch_active,
            "queue_depth": int(self._journal.get_write_metrics().get("pending_rows", 0)),
            "feed": feed_health.get("feeds", {}),
            "runtime_slo": metrics_health,
            "broker": broker_summary,
        }

    def get_metrics(self) -> Dict[str, Any]:
        broker_summary = self._broker.get_account_summary()
        return {
            **self._runtime_metrics.summary(),
            "journal": self._journal.get_write_metrics(),
            "feed": self._safe_feed_metrics().get("feed", {}),
            "event_bus": self._event_bus.summary() if self._event_bus is not None else {},
            "feature_set_version": self._feature_engine.feature_set_version,
            "feature_snapshots": self._feature_snapshots,
            "broker_latency": broker_summary.get("latency", {}),
        }

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

    def _safe_feed_health(self) -> Dict[str, Any]:
        getter = getattr(self._feed, "get_health", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                logger.debug("feed health provider failed", exc_info=True)
        return {"feed_healthy": True, "feeds": {}}

    def _safe_feed_metrics(self) -> Dict[str, Any]:
        getter = getattr(self._feed, "get_metrics", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                logger.debug("feed metrics provider failed", exc_info=True)
        return {"feed": {}}

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

        # Enrich with realized strategy stats from the in-memory read model.
        strategy_stats = self._compute_strategy_trade_stats()
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
            "initial_cash": self._initial_cash, "total_return_pct": ((equity - self._initial_cash) / self._initial_cash * 100.0) if self._initial_cash else 0.0,
            "max_drawdown_pct": 0.0, "positions": positions,
            "position_details": [], "prices": prices,
            "symbols": self._feed.symbols, "trade_stats": {},
            "sl_tp_levels": {},
            "leverage": self._bt_config.leverage if self._bt_config else 1.0,
            "stop_loss_pct": self._bt_config.stop_loss_pct if self._bt_config else None,
            "take_profit_pct": self._bt_config.take_profit_pct if self._bt_config else None,
            "strategy_name": next(iter(self._strategies.values())).name if self._strategies else "",
            "strategy_performance": [],
            "kill_switch_active": self._kill_switch_active,
            "kill_switch_reason": self._kill_switch_reason,
            "kill_switch_triggered_at": self._kill_switch_triggered_at,
            "runtime_metrics": self.get_metrics(),
            "feed_health": self._safe_feed_health(),
            "feature_set_version": self._feature_engine.feature_set_version,
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
        initial_cash = float(initial_cash or self._initial_cash or 100_000.0)

        total_return_pct = ((equity - initial_cash) / initial_cash) * 100 if initial_cash else 0.0

        eq_curve = self.get_equity_curve(limit=500)
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

        trade_stats = self._compute_trade_stats()

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
            "kill_switch_active": self._kill_switch_active,
            "kill_switch_reason": self._kill_switch_reason,
            "kill_switch_triggered_at": self._kill_switch_triggered_at,
            "runtime_metrics": self.get_metrics(),
            "feed_health": self._safe_feed_health(),
            "feature_set_version": self._feature_engine.feature_set_version,
        }
        state.update(self._build_multi_tf_info())
        state["strategy_performance"] = self._build_strategy_performance(
            initial_cash=initial_cash,
            total_return_pct=total_return_pct,
            max_dd_pct=max_dd_pct,
            trade_stats=trade_stats,
        )
        return state
