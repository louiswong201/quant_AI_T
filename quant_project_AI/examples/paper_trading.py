#!/usr/bin/env python3
"""Paper trading entry point with live dashboard.

Single-TF mode (backward compatible):
    python examples/paper_trading.py --symbols BTCUSDT --strategy MA --interval 1h

Multi-TF mode (signal fusion):
    python examples/paper_trading.py --symbols BTCUSDT,ETHUSDT \\
        --multi-tf "1h:MA:3,12|4h:MACD:28,112,3|1d:MACD:28,112,3" \\
        --tf-mode trend_filter --leverage 2

Opens a web dashboard at http://localhost:8050 for real-time monitoring.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_framework.broker.paper import PaperBroker
from quant_framework.broker.live_order_manager import LiveOrderManager
from quant_framework.features import OnlineFeatureEngine
from quant_framework.live.audit import AuditTrail
from quant_framework.live.events import InMemoryEventBus
from quant_framework.live.health_server import HealthCheckServer
from quant_framework.live.kernel_adapter import KernelAdapter, MultiTFAdapter
from quant_framework.live.price_feed import PriceFeedManager
from quant_framework.live.risk import CircuitBreaker, RiskConfig, RiskGate, RiskManagedBroker
from quant_framework.live.runtime_slo import LiveRuntimeSLO
from quant_framework.live.trade_journal import TradeJournal
from quant_framework.live.trading_runner import TradingRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_trading")

STRATEGIES = [
    "MA", "RSI", "MACD", "Bollinger", "Keltner", "Drift",
    "KAMA", "DualMom", "Turtle", "Donchian", "ZScore", "MomBreak",
    "MultiFactor", "Consensus", "MESA", "RegimeEMA", "VolRegime", "RAMOM",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper Trading with Live Dashboard")
    p.add_argument("--symbols", "--symbol", type=str, default="BTCUSDT",
                   help="Comma-separated symbol list (e.g. BTCUSDT,ETHUSDT,AAPL)")
    p.add_argument("--strategy", type=str, default="MA", choices=STRATEGIES,
                   help="Strategy name (single-TF mode)")
    p.add_argument("--params", type=str, default="",
                   help="Strategy params as comma-separated values (e.g. 7,50,30,0.55,0.25)")
    p.add_argument("--leverage", type=float, default=1.0,
                   help="Position leverage multiplier")
    p.add_argument("--stop-loss", type=float, default=0.0,
                   help="Stop-loss percentage (e.g. 0.2 for -20%%)")
    p.add_argument("--interval", type=str, default="1m",
                   choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                   help="Bar interval (single-TF mode)")
    p.add_argument("--multi-tf", type=str, default="",
                   help="Multi-TF config: '1h:MA:3,12|4h:MACD:28,112,3|1d:MACD:28,112,3'")
    p.add_argument("--tf-mode", type=str, default="trend_filter",
                   choices=["trend_filter", "consensus", "primary"],
                   help="Multi-TF signal fusion mode")
    p.add_argument("--initial-cash", type=float, default=100000.0,
                   help="Initial cash ($)")
    p.add_argument("--position-size", type=float, default=0.05,
                   help="Position size as fraction of cash (0.05 = 5%%)")
    p.add_argument("--lookback", type=int, default=200,
                   help="Historical bars to load on startup")
    p.add_argument("--poll-seconds", type=float, default=60.0,
                   help="yfinance poll interval (seconds)")
    p.add_argument("--dashboard-port", type=int, default=8050,
                   help="Dashboard web server port")
    p.add_argument("--health-port", type=int, default=8080,
                   help="Health/metrics HTTP port (<=0 disables)")
    p.add_argument("--no-dashboard", action="store_true",
                   help="Run without dashboard")
    p.add_argument("--db-path", type=str, default="paper_trading.db",
                   help="SQLite database path for trade journal")
    p.add_argument("--audit-db-path", type=str, default="audit.db",
                   help="SQLite database path for order/event audit trail")
    p.add_argument("--max-daily-loss", type=float, default=0.05,
                   help="Circuit breaker: max daily loss as fraction (0.03 = 3%%)")
    p.add_argument("--risk-config", type=str, default="",
                   help="Risk overrides as key=value pairs (e.g. max_daily_loss_pct=0.03)")
    return p.parse_args()


def parse_strategy_params(params_str: str):
    """Parse '7,50,30,0.55,0.25' into a numeric tuple."""
    if not params_str.strip():
        return None
    parts = []
    for p in params_str.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            val = int(p)
        except ValueError:
            val = float(p)
        parts.append(val)
    return tuple(parts) if parts else None


def parse_multi_tf(config_str: str):
    """Parse '1h:MA:3,12|4h:MACD:28,112,3' into a list of (interval, strategy, params)."""
    if not config_str.strip():
        return []
    entries = []
    for block in config_str.split("|"):
        block = block.strip()
        if not block:
            continue
        parts = block.split(":", 2)
        if len(parts) < 2:
            raise ValueError(f"Invalid multi-tf block '{block}'. Expected 'interval:strategy[:params]'")
        iv = parts[0].strip()
        strat = parts[1].strip()
        params = parse_strategy_params(parts[2]) if len(parts) == 3 else None
        entries.append((iv, strat, params))
    return entries


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("No symbols specified")
        sys.exit(1)

    multi_tf_entries = parse_multi_tf(args.multi_tf)
    is_multi_tf = len(multi_tf_entries) > 0
    is_crypto = any(s.endswith("USDT") for s in symbols)

    from quant_framework.backtest.config import BacktestConfig

    logger.info("=" * 60)
    logger.info("Paper Trading System")
    logger.info("=" * 60)
    logger.info("Symbols:        %s", ", ".join(symbols))
    if is_multi_tf:
        logger.info("Mode:           Multi-Timeframe (%s)", args.tf_mode)
        for iv, strat, params in multi_tf_entries:
            logger.info("  %s: %s %s", iv, strat, params or "default")
    else:
        logger.info("Mode:           Single-Timeframe")
        logger.info("Strategy:       %s", args.strategy)
        logger.info("Params:         %s", parse_strategy_params(args.params) or "default")
        logger.info("Interval:       %s", args.interval)
    logger.info("Initial Cash:   $%s", f"{args.initial_cash:,.0f}")
    logger.info("Position Size:  %.0f%%", args.position_size * 100)
    logger.info("Leverage:       %.1fx", args.leverage)
    logger.info("Stop Loss:      %s", f"-{args.stop_loss*100:.0f}%" if args.stop_loss else "None")
    logger.info("Dashboard:      %s", f"http://localhost:{args.dashboard_port}" if not args.no_dashboard else "Disabled")
    logger.info("=" * 60)

    sl_pct = args.stop_loss if args.stop_loss > 0 else None

    if is_multi_tf:
        # -- Multi-TF mode --
        intervals = [iv for iv, _, _ in multi_tf_entries]
        highest_iv = intervals[-1]

        if is_crypto:
            bt_config = BacktestConfig.crypto(
                leverage=args.leverage, stop_loss_pct=sl_pct, interval=highest_iv,
            )
        else:
            bt_config = BacktestConfig.stock_ibkr(
                leverage=args.leverage, stop_loss_pct=sl_pct, interval=highest_iv,
            )

        data_dir = Path(__file__).resolve().parent.parent / "data"
        feed = PriceFeedManager(
            interval=intervals[0],
            poll_seconds=args.poll_seconds,
            data_dir=data_dir,
        )
        for sym in symbols:
            feed.add_symbol_multi_tf(sym, intervals)

        strategies = {}
        for sym in symbols:
            adapters = {}
            for iv, strat, params in multi_tf_entries:
                if is_crypto:
                    cfg = BacktestConfig.crypto(
                        leverage=args.leverage, stop_loss_pct=sl_pct, interval=iv,
                    )
                else:
                    cfg = BacktestConfig.stock_ibkr(
                        leverage=args.leverage, stop_loss_pct=sl_pct, interval=iv,
                    )
                adapters[iv] = KernelAdapter(
                    strategy_name=strat, params=params, config=cfg,
                )
            strategies[sym] = MultiTFAdapter(
                adapters=adapters, mode=args.tf_mode,
            )
    else:
        # -- Single-TF mode (backward compatible) --
        strategy_params = parse_strategy_params(args.params)
        if is_crypto:
            bt_config = BacktestConfig.crypto(
                leverage=args.leverage, stop_loss_pct=sl_pct, interval=args.interval,
            )
        else:
            bt_config = BacktestConfig.stock_ibkr(
                leverage=args.leverage, stop_loss_pct=sl_pct, interval=args.interval,
            )

        data_dir = Path(__file__).resolve().parent.parent / "data"
        feed = PriceFeedManager(
            interval=args.interval,
            poll_seconds=args.poll_seconds,
            data_dir=data_dir,
        )
        for sym in symbols:
            feed.add_symbol(sym)

        strategies = {}
        for sym in symbols:
            strategies[sym] = KernelAdapter(
                strategy_name=args.strategy,
                params=strategy_params,
                config=bt_config,
            )

    paper = PaperBroker.from_backtest_config(bt_config, initial_cash=args.initial_cash)
    max_daily_loss = args.max_daily_loss
    if args.risk_config:
        for kv in args.risk_config.split(","):
            kv = kv.strip()
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            if k.strip() == "max_daily_loss_pct":
                max_daily_loss = float(v.strip())
    risk_cfg = RiskConfig(
        allow_short=True,
        max_daily_loss_pct=max_daily_loss,
        max_consecutive_errors=5,
    )
    cb = CircuitBreaker(risk_cfg, initial_capital=args.initial_cash)
    broker = RiskManagedBroker(
        broker=paper,
        risk_gate=RiskGate(risk_cfg),
        circuit_breaker=cb,
    )

    journal = TradeJournal(db_path=args.db_path)
    audit = AuditTrail(db_path=args.audit_db_path)
    event_bus = InMemoryEventBus()
    feature_engine = OnlineFeatureEngine()
    order_manager = LiveOrderManager()

    runner = TradingRunner(
        feed=feed,
        broker=broker,
        journal=journal,
        strategies=strategies,
        bt_config=bt_config,
        position_size_pct=args.position_size,
        order_manager=order_manager,
        audit_trail=audit,
        event_bus=event_bus,
        feature_engine=feature_engine,
        slo=LiveRuntimeSLO(),
    )
    runner.restore_from_journal()

    if not args.no_dashboard:
        try:
            from quant_framework.dashboard.app import create_app, run_dashboard_thread

            app = create_app(
                get_state=runner.get_state,
                get_equity_curve=lambda: runner.get_equity_curve(1000),
                get_trades=lambda limit: runner.get_recent_trades(limit),
                get_window=lambda sym, iv=None: feed.get_window(sym, iv),
                get_daily_pnl=lambda days: runner.get_daily_pnl(days),
                symbols=symbols,
                refresh_ms=2000,
                trigger_kill_switch=runner.activate_kill_switch,
            )

            dash_thread = run_dashboard_thread(
                app, host="0.0.0.0", port=args.dashboard_port
            )
            logger.info("Dashboard started at http://localhost:%d", args.dashboard_port)
        except ImportError as e:
            logger.warning("Dashboard not available: %s", e)
            logger.warning("Install with: pip install dash plotly")

    async def _main() -> None:
        health_server = None
        if args.health_port > 0:
            health_server = HealthCheckServer(
                health_provider=runner.get_health,
                metrics_provider=runner.get_metrics,
            )
            await health_server.start(host="127.0.0.1", port=args.health_port)
            logger.info("Health started at http://127.0.0.1:%d/health", args.health_port)
        try:
            await runner.run(lookback=args.lookback)
        finally:
            if health_server is not None:
                await health_server.stop()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        journal.close()
        audit.close()
        logger.info("Paper trading stopped.")


if __name__ == "__main__":
    main()
