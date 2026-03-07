#!/usr/bin/env python3
"""Multi-symbol live trading — auto-loads the best strategy per symbol
from the production scan results (live_trading_config.json).

Each symbol gets the highest-ranked recommendation (single-TF or multi-TF),
with its own leverage, stop-loss, and strategy parameters.

Usage
-----
    # All symbols from config (paper trading with dashboard):
    python run_live_trading.py

    # Top 10 symbols only:
    python run_live_trading.py --top-n 10

    # Specific symbols:
    python run_live_trading.py --symbols BTC,ETH,PG,JPM

    # Only CPCV-validated strategies:
    python run_live_trading.py --cpcv-only

    # Custom cash and position size:
    python run_live_trading.py --initial-cash 50000 --position-size 0.03
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_framework.backtest.config import BacktestConfig
from quant_framework.broker.paper import PaperBroker
from quant_framework.live.kernel_adapter import KernelAdapter, MultiTFAdapter
from quant_framework.live.price_feed import PriceFeedManager
from quant_framework.live.risk import CircuitBreaker, RiskConfig, RiskGate, RiskManagedBroker
from quant_framework.live.trade_journal import TradeJournal
from quant_framework.live.trading_runner import TradingRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_trading")

# ── Symbol mapping ───────────────────────────────────────────────────

CRYPTO_BASE = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA",
    "DOGE", "AVAX", "DOT", "MATIC", "LINK", "UNI",
}

STOCK_TICKER_MAP = {
    "MA_stock": "MA",
}

_INTERVAL_RANK = {"1m": 0, "5m": 1, "15m": 2, "1h": 3, "4h": 4, "1d": 5, "1w": 6}


def is_crypto(sym: str) -> bool:
    return sym.upper() in CRYPTO_BASE


def to_feed_symbol(config_sym: str) -> str:
    """Config symbol name → price-feed ticker (e.g. BTC → BTCUSDT)."""
    mapped = STOCK_TICKER_MAP.get(config_sym, config_sym)
    if mapped.upper() in CRYPTO_BASE or config_sym.upper() in CRYPTO_BASE:
        return mapped.upper() + "USDT"
    return mapped


# ── Config loading ───────────────────────────────────────────────────

def load_best_per_symbol(
    config_path: str,
    *,
    symbols_filter: Optional[List[str]] = None,
    cpcv_only: bool = False,
    top_n: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return the highest-ranked recommendation for each unique symbol.

    ``symbols_filter`` restricts to the given config symbol names.
    ``cpcv_only`` drops non-CPCV-validated single-TF recommendations.
    ``top_n`` keeps at most N symbols (by rank order of first appearance).
    """
    with open(config_path) as f:
        cfg = json.load(f)

    best: Dict[str, Dict[str, Any]] = {}
    for rec in cfg["recommendations"]:
        sym = rec["symbol"]

        if symbols_filter and sym not in symbols_filter:
            continue
        if cpcv_only and rec["type"] == "single-TF" and not rec.get("cpcv_validated"):
            continue
        if sym in best:
            continue
        if top_n and len(best) >= top_n:
            break

        best[sym] = rec
    return best


# ── Adapter building ─────────────────────────────────────────────────

def _safe_sl(leverage: float) -> float:
    """Conservative stop-loss that scales inversely with leverage."""
    if leverage <= 1:
        return 0.40
    return min(0.40, 0.80 / leverage)


def _make_bt_config(sym: str, leverage: float, interval: str) -> BacktestConfig:
    sl = _safe_sl(leverage)
    if is_crypto(sym):
        return BacktestConfig.crypto(leverage=leverage, stop_loss_pct=sl, interval=interval)
    return BacktestConfig.stock_ibkr(leverage=leverage, stop_loss_pct=sl, interval=interval)


def build_adapter(
    rec: Dict[str, Any],
) -> Tuple[Union[KernelAdapter, MultiTFAdapter], List[str], BacktestConfig]:
    """Create the appropriate adapter from a recommendation dict.

    Returns (adapter, intervals_needed, bt_config).
    """
    sym = rec["symbol"]
    leverage = rec.get("leverage", 1)

    if rec["type"] == "single-TF":
        interval = rec["interval"]
        bt_cfg = _make_bt_config(sym, leverage, interval)
        adapter = KernelAdapter(
            strategy_name=rec["strategy"],
            params=tuple(rec["params"]),
            config=bt_cfg,
        )
        return adapter, [interval], bt_cfg

    # Multi-TF
    tf_configs = rec.get("tf_configs", {})
    mode = rec.get("mode", "consensus")

    adapters: Dict[str, KernelAdapter] = {}
    intervals: List[str] = []
    for iv, tf_cfg in sorted(
        tf_configs.items(), key=lambda x: _INTERVAL_RANK.get(x[0], 99)
    ):
        cfg = _make_bt_config(sym, leverage, iv)
        adapters[iv] = KernelAdapter(
            strategy_name=tf_cfg["strategy"],
            params=tuple(tf_cfg["params"]),
            config=cfg,
        )
        intervals.append(iv)

    adapter = MultiTFAdapter(adapters=adapters, mode=mode)
    highest_iv = intervals[-1] if intervals else "1d"
    bt_cfg = _make_bt_config(sym, leverage, highest_iv)
    return adapter, intervals, bt_cfg


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-symbol live trading from production scan results",
    )
    p.add_argument(
        "--config", type=str,
        default="reports/live_trading_config.json",
        help="Path to live_trading_config.json",
    )
    p.add_argument(
        "--symbols", type=str, default="",
        help="Comma-separated config symbol filter (e.g. BTC,ETH,PG)",
    )
    p.add_argument(
        "--top-n", type=int, default=None,
        help="Only trade the top N symbols by scan rank",
    )
    p.add_argument(
        "--cpcv-only", action="store_true",
        help="Only include CPCV-validated recommendations",
    )
    p.add_argument(
        "--initial-cash", type=float, default=100_000.0,
        help="Initial paper trading cash ($)",
    )
    p.add_argument(
        "--position-size", type=float, default=0.05,
        help="Per-trade position size as fraction of equity (0.05 = 5%%)",
    )
    p.add_argument(
        "--lookback", type=int, default=200,
        help="Historical bars to load on startup",
    )
    p.add_argument(
        "--poll-seconds", type=float, default=60.0,
        help="yfinance poll interval for stock feeds (seconds)",
    )
    p.add_argument(
        "--dashboard-port", type=int, default=8050,
        help="Dashboard web server port",
    )
    p.add_argument(
        "--no-dashboard", action="store_true",
        help="Run without the web dashboard",
    )
    p.add_argument(
        "--db-path", type=str, default="live_trading.db",
        help="SQLite database path for trade journal",
    )
    p.add_argument(
        "--max-daily-loss", type=float, default=0.05,
        help="Circuit breaker: max daily loss fraction (0.05 = 5%%)",
    )
    p.add_argument(
        "--use-portfolio-weights", action="store_true",
        help="Load position sizes from Portfolio Engine (research.db)",
    )
    p.add_argument(
        "--research-db", type=str, default="research.db",
        help="Path to research.db for Portfolio Engine weights",
    )
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    sym_filter = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else None
    )

    best = load_best_per_symbol(
        str(config_path),
        symbols_filter=sym_filter,
        cpcv_only=args.cpcv_only,
        top_n=args.top_n,
    )

    if not best:
        logger.error("No recommendations matched the given filters")
        sys.exit(1)

    # ── Load portfolio weights from research.db if requested ──
    portfolio_weights: Dict[str, float] = {}
    if args.use_portfolio_weights:
        try:
            from quant_framework.research import ResearchDB, run_portfolio_analysis
            rdb = ResearchDB(args.research_db)
            pa = run_portfolio_analysis(rdb, lookback_days=90)
            rdb.close()
            raw_weights = pa.get("position_sizes", {})
            for key, pct in raw_weights.items():
                sym_part = key.split("/")[0]
                portfolio_weights[sym_part] = pct
            if portfolio_weights:
                logger.info("Portfolio weights loaded for %d strategies", len(portfolio_weights))
        except Exception as exc:
            logger.warning("Could not load portfolio weights: %s", exc)

    # ── Build adapters + feed ──
    strategies: Dict[str, Union[KernelAdapter, MultiTFAdapter]] = {}
    symbol_configs: Dict[str, BacktestConfig] = {}

    data_dir = Path(__file__).resolve().parent / "data"
    feed = PriceFeedManager(
        interval="1d",
        poll_seconds=args.poll_seconds,
        data_dir=data_dir,
    )

    logger.info("=" * 70)
    logger.info("  Multi-Symbol Live Trading")
    logger.info("=" * 70)

    for config_sym, rec in best.items():
        adapter, intervals, bt_cfg = build_adapter(rec)
        fsym = to_feed_symbol(config_sym)

        strategies[fsym] = adapter
        symbol_configs[fsym] = bt_cfg

        if is_crypto(config_sym):
            feed.add_symbol_multi_tf(fsym, intervals)
        else:
            feed.add_symbol(fsym)

        lev = rec.get("leverage", 1)
        sl = _safe_sl(lev)
        if rec["type"] == "single-TF":
            strat_desc = f"{rec['strategy']}({rec['params']})"
            iv_desc = rec["interval"]
        else:
            parts = [
                f"{iv}:{tc['strategy']}" for iv, tc in rec.get("tf_configs", {}).items()
            ]
            strat_desc = f"MultiTF[{rec.get('mode','consensus')}]({' | '.join(parts)})"
            iv_desc = rec.get("tf_combo", "+".join(intervals))

        logger.info(
            "  #%-2d  %-10s → %-8s  lev=%.0fx  SL=%.0f%%  %s  [%s]",
            rec["rank"], config_sym, fsym, lev, sl * 100,
            strat_desc, iv_desc,
        )

    logger.info("-" * 70)
    n_crypto = sum(1 for s in best if is_crypto(s))
    n_stock = len(best) - n_crypto
    logger.info(
        "  Loaded %d symbols (%d crypto, %d stock) | Cash=$%s | Pos=%.0f%%",
        len(best), n_crypto, n_stock,
        f"{args.initial_cash:,.0f}", args.position_size * 100,
    )
    logger.info("=" * 70)

    # ── Broker (use crypto preset: low commissions, fractional, short ok) ──
    broker_cfg = BacktestConfig.crypto(leverage=1.0, interval="1d")
    paper = PaperBroker.from_backtest_config(broker_cfg, initial_cash=args.initial_cash)

    risk_cfg = RiskConfig(
        allow_short=True,
        max_daily_loss_pct=args.max_daily_loss,
        max_consecutive_errors=5,
    )
    cb = CircuitBreaker(risk_cfg, initial_capital=args.initial_cash)
    broker = RiskManagedBroker(
        broker=paper,
        risk_gate=RiskGate(risk_cfg),
        circuit_breaker=cb,
    )

    journal = TradeJournal(db_path=args.db_path)

    # Use the first symbol's config as the default (for dashboard metadata)
    first_fsym = next(iter(strategies))
    default_cfg = symbol_configs.get(first_fsym, broker_cfg)

    # Override position_size_pct with portfolio weights if available
    pos_size = args.position_size
    if portfolio_weights:
        first_config_sym = next(iter(best))
        pw = portfolio_weights.get(first_config_sym, args.position_size)
        if pw > 0:
            pos_size = pw
        logger.info(
            "Using portfolio-weighted position sizes (base=%.2f%%, per-symbol overrides available)",
            pos_size * 100,
        )

    runner = TradingRunner(
        feed=feed,
        broker=broker,
        journal=journal,
        strategies=strategies,
        bt_config=default_cfg,
        symbol_configs=symbol_configs,
        position_size_pct=pos_size,
    )
    runner.restore_from_journal()

    # ── Dashboard ──
    if not args.no_dashboard:
        try:
            from quant_framework.dashboard.app import create_app, run_dashboard_thread

            all_fsyms = list(strategies.keys())
            app = create_app(
                get_state=runner.get_state,
                get_equity_curve=lambda: journal.get_equity_curve(1000),
                get_trades=lambda limit: journal.get_trades(limit),
                get_window=lambda sym, iv=None: feed.get_window(sym, iv),
                get_daily_pnl=lambda days: journal.get_daily_pnl(days),
                symbols=all_fsyms,
                refresh_ms=2000,
                trigger_kill_switch=runner.activate_kill_switch,
            )
            run_dashboard_thread(app, host="0.0.0.0", port=args.dashboard_port)
            logger.info("Dashboard → http://localhost:%d", args.dashboard_port)
        except ImportError as e:
            logger.warning("Dashboard unavailable: %s  (pip install dash plotly)", e)

    # ── Run ──
    try:
        asyncio.run(runner.run(lookback=args.lookback))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        journal.close()
        logger.info("Live trading stopped.")


if __name__ == "__main__":
    main()
