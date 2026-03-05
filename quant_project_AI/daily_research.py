#!/usr/bin/env python3
"""
V4 Intelligent Strategy Research Pipeline
==========================================
Unified orchestration of four research engines backed by a persistent
SQLite research database (research.db).

Engines:
  - Monitor   — daily health trends, regime detection, performance attribution
  - Optimize  — Bayesian parameter updates, composite gate, champion/challenger
  - Portfolio  — correlation, weight optimization, portfolio-level metrics
  - Discover  — variant mining, anomaly scanning, external research

Modes:
  daily    — Monitor only (~15s)
  weekly   — Monitor + Optimize + Portfolio + Discover(anomaly)
  monthly  — All engines at full depth (EXPANDED grids + CPCV + variant mining + papers)
  triggered — immediate Optimize for ALERT symbols

Usage:
    python daily_research.py                          # daily: monitor only
    python daily_research.py --mode weekly            # weekly: monitor + optimize + portfolio
    python daily_research.py --mode monthly           # monthly: all engines, deep scan
    python daily_research.py --quick                  # alias for daily
    python daily_research.py --deep                   # alias for monthly
    python daily_research.py --symbols BTC,ETH,PG     # focus on specific symbols
    python daily_research.py --skip-download          # skip data refresh
    python daily_research.py --apply                  # auto-update live config
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed.  pip install yfinance")
    sys.exit(1)

from quant_framework.backtest.config import BacktestConfig
from quant_framework.backtest.kernels import (
    DEFAULT_PARAM_GRIDS,
    KERNEL_NAMES,
    config_to_kernel_costs,
    eval_kernel_detailed,
)
from quant_framework.backtest.robust_scan import run_robust_scan
from quant_framework.research.database import ResearchDB
from quant_framework.research.monitor import run_monitor
from quant_framework.research.optimizer import (
    run_optimizer,
    composite_gate_score,
    promote_challenger,
)
from quant_framework.research.portfolio import run_portfolio_analysis
from quant_framework.research.discover import run_discover

# ═══════════════════════════════════════════════════════════════
#  Symbol Definitions
# ═══════════════════════════════════════════════════════════════

CRYPTO_SYMBOLS = {
    "BTC": "BTC-USD",  "ETH": "ETH-USD",  "BNB": "BNB-USD",
    "SOL": "SOL-USD",  "XRP": "XRP-USD",  "ADA": "ADA-USD",
    "DOGE": "DOGE-USD", "AVAX": "AVAX-USD", "DOT": "DOT-USD",
    "MATIC": "MATIC-USD",
}
STOCK_SYMBOLS = {
    "AAPL": "AAPL", "MSFT": "MSFT", "GOOGL": "GOOGL", "AMZN": "AMZN",
    "NVDA": "NVDA", "META": "META", "TSLA": "TSLA", "BRK-B": "BRK-B",
    "UNH": "UNH",  "JNJ": "JNJ",  "JPM": "JPM",   "V": "V",
    "PG": "PG",    "HD": "HD",    "MA_stock": "MA", "XOM": "XOM",
    "ABBV": "ABBV", "MRK": "MRK", "CVX": "CVX",   "KO": "KO",
}
ALL_SYMBOLS = {**CRYPTO_SYMBOLS, **STOCK_SYMBOLS}
CRYPTO_SET = set(CRYPTO_SYMBOLS.keys())

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def is_crypto(sym: str) -> bool:
    return sym.upper() in CRYPTO_SET


def make_config(sym: str, leverage: float, interval: str) -> BacktestConfig:
    sl = min(0.40, 0.80 / leverage) if leverage > 1 else 0.40
    if is_crypto(sym):
        return BacktestConfig.crypto(leverage=leverage, stop_loss_pct=sl, interval=interval)
    return BacktestConfig.stock_ibkr(leverage=leverage, stop_loss_pct=sl, interval=interval)


def print_header(title, width=80):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index.name = "date"
    df = df.reset_index()
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "open": col_map[c] = "open"
        elif cl == "high": col_map[c] = "high"
        elif cl == "low": col_map[c] = "low"
        elif cl == "close": col_map[c] = "close"
        elif cl == "volume": col_map[c] = "volume"
        elif cl in ("date", "datetime"): col_map[c] = "date"
    df = df.rename(columns=col_map)
    required = ["date", "open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()
    cols = ["date", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    df = df[cols].dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    return df.sort_values("date").reset_index(drop=True)


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h.empty:
        return pd.DataFrame()
    df = df_1h.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    ohlc = df.resample("4h").agg(agg).dropna(subset=["open", "high", "low", "close"])
    return ohlc.reset_index()


# ═══════════════════════════════════════════════════════════════
#  PHASE 0: Incremental Data Refresh
# ═══════════════════════════════════════════════════════════════

def _incremental_download(sym: str, ticker: str, csv_path: str,
                          interval: str, period_fallback: str) -> int:
    existing = None
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        dates = pd.to_datetime(existing["date"], utc=True)
        last_date = dates.max()
        days_since = (datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days
        if days_since <= 0:
            return 0
        period = f"{min(days_since + 5, 30)}d"
    else:
        period = period_fallback

    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval)
            if df is not None and len(df) > 0:
                break
        except Exception:
            if attempt == 2:
                return 0
            time.sleep(0.5)
    else:
        return 0

    df = process_df(df)
    if df.empty:
        return 0

    if existing is not None and not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"], utc=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
        new_bars = len(combined) - len(existing)
    else:
        combined = df
        new_bars = len(df)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    combined.to_csv(csv_path, index=False)
    return max(0, new_bars)


def phase0_refresh(data_dir: str, symbols: List[str]) -> Dict[str, int]:
    updates = {}
    for sym in symbols:
        ticker = ALL_SYMBOLS.get(sym, sym)
        csv_1d = os.path.join(data_dir, f"{sym}.csv")
        n1d = _incremental_download(sym, ticker, csv_1d, "1d", "max")
        csv_1h = os.path.join(data_dir, "1h", f"{sym}_1h.csv")
        n1h = _incremental_download(sym, ticker, csv_1h, "1h", "730d")
        n4h = 0
        if os.path.exists(csv_1h):
            df_1h = pd.read_csv(csv_1h, parse_dates=["date"])
            df_4h = resample_to_4h(df_1h)
            if not df_4h.empty:
                csv_4h = os.path.join(data_dir, "4h", f"{sym}_4h.csv")
                os.makedirs(os.path.dirname(csv_4h), exist_ok=True)
                old_len = 0
                if os.path.exists(csv_4h):
                    old_len = len(pd.read_csv(csv_4h))
                df_4h.to_csv(csv_4h, index=False)
                n4h = max(0, len(df_4h) - old_len)
        total = n1d + n1h + n4h
        updates[sym] = total
        tag = f"+{total}" if total > 0 else "up-to-date"
        print(f"    {sym:<12} {tag}  (1d:+{n1d}  1h:+{n1h}  4h:+{n4h})")
    return updates


# ═══════════════════════════════════════════════════════════════
#  PHASE 1: Monitor Engine (health + regime + attribution)
# ═══════════════════════════════════════════════════════════════

def _load_tf_data(data_dir: str) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Load multi-timeframe data into engine-compatible format."""
    from run_production_scan import load_daily, load_intraday
    daily = load_daily(data_dir, min_bars=50)
    data_1h = load_intraday(data_dir, "1h", min_bars=50) if os.path.isdir(os.path.join(data_dir, "1h")) else {}
    data_4h = load_intraday(data_dir, "4h", min_bars=50) if os.path.isdir(os.path.join(data_dir, "4h")) else {}
    return {"1d": daily, "1h": data_1h, "4h": data_4h}


def phase1_monitor(
    db: ResearchDB,
    live_config: Dict[str, Any],
    tf_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    recent_days: int = 90,
) -> List[Dict[str, Any]]:
    """Run Monitor Engine — health trends, regime, attribution."""
    return run_monitor(
        db, live_config, tf_data,
        recent_days=recent_days,
        attribute=True,
    )


# ═══════════════════════════════════════════════════════════════
#  PHASE 2: Re-Scan (for Optimize Engine input)
# ═══════════════════════════════════════════════════════════════

def phase2_rescan(
    data_dir: str,
    symbols: List[str],
    grids: Dict[str, list],
    leverage_levels: List[int] = [1, 2, 3],
    n_mc: int = 10,
    n_shuf: int = 5,
    n_boot: int = 5,
) -> List[Dict[str, Any]]:
    from run_production_scan import load_daily
    daily = load_daily(data_dir)
    ranking: List[Dict[str, Any]] = []

    crypto_data = {s: d for s, d in daily.items() if is_crypto(s) and (not symbols or s in symbols)}
    stock_data = {s: d for s, d in daily.items() if not is_crypto(s) and (not symbols or s in symbols)}

    for label, sub_data in [("crypto", crypto_data), ("stock", stock_data)]:
        if not sub_data:
            continue
        for lev in leverage_levels:
            sym_list = list(sub_data.keys())
            config = make_config(sym_list[0], float(lev), "1d")
            print(f"    1d / {lev}x / {label} ({len(sym_list)} sym) ... ", end="", flush=True)
            t0 = time.time()
            result = run_robust_scan(
                symbols=sym_list,
                data=sub_data,
                config=config,
                param_grids=grids,
                n_mc_paths=n_mc,
                n_shuffle_paths=n_shuf,
                n_bootstrap_paths=n_boot,
                parallel_symbols="auto",
            )
            elapsed = time.time() - t0
            print(f"{elapsed:.1f}s ({result.total_combos:,} combos)")

            for sym in result.per_symbol:
                for sn, metrics in result.per_symbol[sym].items():
                    ranking.append({
                        "symbol": sym,
                        "strategy": sn,
                        "params": metrics.get("params"),
                        "leverage": lev,
                        "interval": "1d",
                        "sharpe": metrics.get("sharpe", 0),
                        "oos_ret": metrics.get("oos_ret", 0),
                        "oos_dd": metrics.get("oos_dd", 0),
                        "wf_score": metrics.get("wf_score", -1e18),
                        "dsr_p": metrics.get("dsr_p", 1),
                        "mc_pct_positive": metrics.get("mc_pct_positive", 0),
                    })

    return ranking


# ═══════════════════════════════════════════════════════════════
#  PHASE 3: Compare & Rank (legacy compat + new gate scoring)
# ═══════════════════════════════════════════════════════════════

def phase3_compare(
    live_config: Dict[str, Any],
    rescan_ranking: List[Dict[str, Any]],
    monitor_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    current_by_sym: Dict[str, Dict] = {}
    for rec in live_config.get("recommendations", []):
        sym = rec["symbol"]
        if sym not in current_by_sym:
            current_by_sym[sym] = rec

    new_best_by_sym: Dict[str, Dict] = {}
    for e in sorted(rescan_ranking, key=lambda x: x["wf_score"], reverse=True):
        sym = e["symbol"]
        if sym not in new_best_by_sym and e["sharpe"] > 0 and e["params"] is not None:
            new_best_by_sym[sym] = e

    _STATUS_RANK = {"ALERT": 0, "WATCH": 1, "ERROR": 2, "HEALTHY": 3, "UNKNOWN": 4}
    health_by_sym: Dict[str, Dict] = {}
    for h in monitor_results:
        sym = h["symbol"]
        prev = health_by_sym.get(sym)
        if prev is None or _STATUS_RANK.get(h.get("status"), 9) < _STATUS_RANK.get(prev.get("status"), 9):
            health_by_sym[sym] = h

    changes = []
    all_syms = set(list(current_by_sym.keys()) + list(new_best_by_sym.keys()))

    for sym in sorted(all_syms):
        cur = current_by_sym.get(sym)
        new = new_best_by_sym.get(sym)
        health = health_by_sym.get(sym, {})

        if cur is None and new is not None:
            changes.append({
                "symbol": sym, "action": "NEW",
                "detail": f"New discovery: {new['strategy']} @ {new['leverage']}x, Sharpe={new['sharpe']:.2f}",
                "new_strategy": new["strategy"], "new_params": new["params"],
                "new_sharpe": new["sharpe"], "priority": "LOW",
            })
            continue

        if cur is None:
            continue

        cur_sharpe = cur.get("backtest_metrics", {}).get("sharpe", 0)
        health_metrics = health.get("health", {})
        recent_sharpe = health_metrics.get("sharpe_30d", cur_sharpe)
        status = health.get("status", "UNKNOWN")
        new_sharpe = new["sharpe"] if new else 0

        if new and cur["type"] == "single-TF":
            same_strategy = (new["strategy"] == cur["strategy"])
            same_params = (list(new["params"]) == cur.get("params", []))
            sharpe_improved = new_sharpe > cur_sharpe * 1.1

            if not same_strategy or not same_params:
                if sharpe_improved:
                    action = "UPGRADE"
                    priority = "HIGH" if new_sharpe > cur_sharpe * 1.3 else "MEDIUM"
                    detail = (
                        f"{cur['strategy']}({cur.get('params', [])}) -> "
                        f"{new['strategy']}({new['params']})  "
                        f"Sharpe: {cur_sharpe:.2f} -> {new_sharpe:.2f}"
                    )
                else:
                    action = "KEEP"
                    detail = f"Current {cur['strategy']} still optimal (Sharpe {cur_sharpe:.2f})"
                    priority = "NONE"
            else:
                action = "KEEP"
                detail = f"Params unchanged, Sharpe={cur_sharpe:.2f}"
                priority = "NONE"
        else:
            action = "KEEP"
            detail = f"Current {cur.get('strategy', cur['type'])} Sharpe={cur_sharpe:.2f}"
            priority = "NONE"

        if status == "ALERT":
            action = "CHECK"
            priority = "HIGH"
            detail = f"Recent Sharpe={recent_sharpe:.2f} vs original {cur_sharpe:.2f} -- degraded"

        attribution = health.get("attribution")
        if attribution and attribution.get("explanation"):
            detail += f" | Attribution: {attribution['explanation']}"

        changes.append({
            "symbol": sym, "action": action, "detail": detail,
            "current_strategy": cur.get("strategy", cur.get("type")),
            "current_sharpe": cur_sharpe, "recent_sharpe": recent_sharpe,
            "new_strategy": new["strategy"] if new else None,
            "new_sharpe": new_sharpe if new else None,
            "health_status": status, "priority": priority,
            "regime": health.get("regime"),
        })

    changes.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}.get(x["priority"], 4))
    return changes


# ═══════════════════════════════════════════════════════════════
#  Apply updates to live config
# ═══════════════════════════════════════════════════════════════

def apply_updates(
    db: ResearchDB,
    live_config: Dict[str, Any],
    changes: List[Dict[str, Any]],
    output_path: str,
):
    upgrades = [c for c in changes if c["action"] == "UPGRADE" and c.get("new_params")]
    if not upgrades:
        print("    No upgrades to apply.")
        return

    recs = live_config["recommendations"]
    updated = 0
    diffs = []
    for upg in upgrades:
        sym = upg["symbol"]
        for rec in recs:
            if rec["symbol"] == sym and rec["type"] == "single-TF":
                old = f"{rec['strategy']}({rec['params']})"
                old_params = rec["params"]
                rec["strategy"] = upg["new_strategy"]
                rec["params"] = list(upg["new_params"])
                rec["backtest_metrics"]["sharpe"] = upg["new_sharpe"]
                updated += 1
                diffs.append({
                    "symbol": sym,
                    "old": old, "new": f"{upg['new_strategy']}({upg['new_params']})",
                })
                print(f"    {sym}: {old} -> {upg['new_strategy']}({upg['new_params']})")
                break

    if updated:
        live_config["updated"] = datetime.now().isoformat()

        # Version in research DB
        db.save_config_version(
            live_config,
            summary=f"Applied {updated} upgrades",
            diff={"changes": diffs},
            n_changes=updated,
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned = output_path.replace(".json", f"_{ts}.json")
        with open(versioned, "w") as f:
            json.dump(live_config, f, indent=2, default=str)
        with open(output_path, "w") as f:
            json.dump(live_config, f, indent=2, default=str)
        print(f"    Updated {updated} strategies -> {output_path}")
        print(f"    Backup -> {versioned}")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="V4 Intelligent Strategy Research Pipeline")
    p.add_argument("--mode", type=str, default="daily",
                   choices=["daily", "weekly", "monthly", "triggered"],
                   help="Research mode (daily/weekly/monthly/triggered)")
    p.add_argument("--quick", action="store_true",
                   help="Alias for --mode daily (monitor only)")
    p.add_argument("--deep", action="store_true",
                   help="Alias for --mode monthly (all engines, expanded grids)")
    p.add_argument("--symbols", type=str, default="",
                   help="Comma-separated symbol filter (e.g. BTC,ETH,PG)")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip data refresh, use existing data")
    p.add_argument("--apply", action="store_true",
                   help="Auto-update live_trading_config.json if improvements found")
    p.add_argument("--config", type=str,
                   default="reports/live_trading_config.json",
                   help="Path to current live trading config")
    p.add_argument("--recent-days", type=int, default=90,
                   help="Days of recent data for health check (default: 90)")
    p.add_argument("--db", type=str, default="research.db",
                   help="Path to research database (default: research.db)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Resolve mode aliases
    if args.quick:
        args.mode = "daily"
    elif args.deep:
        args.mode = "monthly"

    sym_filter = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    config_path = os.path.join(base_dir, args.config)
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path}")
        print("  Run run_production_scan.py first to generate initial config.")
        sys.exit(1)

    with open(config_path) as f:
        live_config = json.load(f)

    active_syms = list({r["symbol"] for r in live_config["recommendations"]})
    if sym_filter:
        active_syms = [s for s in active_syms if s in sym_filter]

    # Initialise research database
    db_path = os.path.join(base_dir, args.db)
    db = ResearchDB(db_path)

    t_global = time.time()
    mode = args.mode.upper()

    print_header("V4 INTELLIGENT STRATEGY RESEARCH")
    print(f"  Date:     {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Mode:     {mode}")
    print(f"  Symbols:  {len(active_syms)} ({', '.join(active_syms[:8])}{'...' if len(active_syms)>8 else ''})")
    print(f"  Config:   {args.config}")
    print(f"  Database: {args.db}")
    counts = db.table_counts()
    print(f"  DB State: {counts}")

    # ── Phase 0: Data Refresh ──
    data_updates: Dict[str, int] = {}
    if not args.skip_download:
        print_header("PHASE 0: DATA REFRESH")
        data_updates = phase0_refresh(data_dir, active_syms)
        total_new = sum(data_updates.values())
        print(f"\n  Total: {total_new} new bars across {sum(1 for v in data_updates.values() if v > 0)} symbols")
    else:
        print("\n  [Skipping data download]")
        data_updates = {s: 0 for s in active_syms}

    # Load data for engines
    tf_data = _load_tf_data(data_dir)

    # ── Phase 1: Monitor Engine (ALL modes) ──
    print_header("PHASE 1: MONITOR ENGINE")
    t1 = time.time()
    monitor_results = phase1_monitor(
        db, live_config, tf_data, recent_days=args.recent_days,
    )
    t1_elapsed = time.time() - t1

    n_healthy = sum(1 for r in monitor_results if r["status"] == "HEALTHY")
    n_watch = sum(1 for r in monitor_results if r["status"] == "WATCH")
    n_alert = sum(1 for r in monitor_results if r["status"] == "ALERT")

    print(f"\n  {'Symbol':<12} {'Strategy':<14} {'Lev':>3} {'Sharpe30':>8} {'DD%':>6} {'WR':>5} {'Status':>8}")
    print(f"  {'-'*12} {'-'*14} {'-'*3} {'-'*8} {'-'*6} {'-'*5} {'-'*8}")
    for r in sorted(monitor_results, key=lambda x: x.get("original_sharpe", 0), reverse=True):
        h = r.get("health", {})
        print(
            f"  {r['symbol']:<12} {r['strategy']:<14} {r['leverage']:>2}x "
            f"{h.get('sharpe_30d', 0):>8.2f} {h.get('drawdown_pct', 0):>5.1%} "
            f"{h.get('win_rate', 0):>4.0%} {r['status']:>8}"
        )

    print(f"\n  Health: {n_healthy} healthy, {n_watch} watch, {n_alert} alert  ({t1_elapsed:.1f}s)")

    # Check for triggered alerts
    alert_syms = [r["symbol"] for r in monitor_results if r["status"] == "ALERT"]
    if alert_syms and args.mode == "daily":
        print(f"\n  ALERT symbols detected: {alert_syms}")
        print(f"  Triggering immediate Optimize for alert symbols...")
        args.mode = "triggered"

    # ── Phase 2: Re-Scan + Optimize Engine (weekly/monthly/triggered) ──
    rescan_ranking: List[Dict[str, Any]] = []
    optimize_results: List[Dict[str, Any]] = []
    rescan_done = False

    if args.mode in ("weekly", "monthly", "triggered"):
        print_header("PHASE 2: PARAMETER RE-SCAN + OPTIMIZE ENGINE")

        if args.mode == "triggered":
            scan_syms = alert_syms
            grids = DEFAULT_PARAM_GRIDS
            grid_label = "DEFAULT (triggered)"
            levs = [1, 2, 3]
            n_mc, n_shuf, n_boot = 10, 5, 5
        elif args.mode == "monthly":
            scan_syms = active_syms
            try:
                from run_full_scan import EXPANDED_GRIDS
                grids = EXPANDED_GRIDS
                grid_label = "EXPANDED"
                levs = [1, 2, 3, 5]
                n_mc, n_shuf, n_boot = 20, 10, 10
            except ImportError:
                grids = DEFAULT_PARAM_GRIDS
                grid_label = "DEFAULT (EXPANDED unavailable)"
                levs = [1, 2, 3]
                n_mc, n_shuf, n_boot = 10, 5, 5
        else:
            scan_syms = active_syms
            grids = DEFAULT_PARAM_GRIDS
            grid_label = "DEFAULT"
            levs = [1, 2, 3]
            n_mc, n_shuf, n_boot = 10, 5, 5

        total_combos = sum(len(v) for v in grids.values())
        print(f"  Grids: {grid_label} ({total_combos:,} combos)")
        print(f"  Leverage: {levs}")
        print(f"  Symbols: {len(scan_syms)}")

        t2 = time.time()
        rescan_ranking = phase2_rescan(
            data_dir, scan_syms, grids, levs, n_mc, n_shuf, n_boot,
        )
        t2_elapsed = time.time() - t2
        rescan_done = True

        top = sorted(rescan_ranking, key=lambda x: x["wf_score"], reverse=True)[:10]
        if top:
            print(f"\n  -- Top 10 from Re-Scan --")
            print(f"  {'Sym':<12} {'Strategy':<14} {'Lev':>3} {'Sharpe':>7} {'OOS Ret':>9} {'Score':>7}")
            for e in top:
                print(f"  {e['symbol']:<12} {e['strategy']:<14} {e['leverage']:>2}x "
                      f"{e['sharpe']:>7.2f} {e['oos_ret']:>+8.1f}% {e['wf_score']:>+6.1f}")

        print(f"\n  Re-scan: {len(rescan_ranking)} entries in {t2_elapsed:.1f}s")

        # Optimize Engine
        print(f"\n  Running Optimize Engine (Bayesian update + gate scoring)...")
        t_opt = time.time()
        optimize_results = run_optimizer(
            db, live_config, tf_data, rescan_ranking,
            check_neighborhood=(args.mode == "monthly"),
            register_challengers=True,
        )
        t_opt_elapsed = time.time() - t_opt

        if optimize_results:
            print(f"\n  -- Optimizer Results --")
            print(f"  {'Sym':<12} {'Strategy':<14} {'Gate':>6} {'Sharpe':>7} {'Promo':>6}")
            for r in sorted(optimize_results, key=lambda x: x["gate_score"], reverse=True):
                promo = "YES" if r.get("promotion") else "no"
                print(f"  {r['symbol']:<12} {r['strategy']:<14} {r['gate_score']:>6.3f} "
                      f"{r['sharpe']:>7.2f} {promo:>6}")

        print(f"\n  Optimize: {len(optimize_results)} results in {t_opt_elapsed:.1f}s")

    # ── Phase 3: Portfolio Engine (weekly/monthly) ──
    portfolio_result = None
    if args.mode in ("weekly", "monthly"):
        print_header("PHASE 3: PORTFOLIO ENGINE")
        t3 = time.time()
        portfolio_result = run_portfolio_analysis(db, lookback_days=args.recent_days)
        t3_elapsed = time.time() - t3

        pm = portfolio_result.get("portfolio_metrics", {})
        if pm:
            print(f"\n  Portfolio Sharpe:   {pm.get('portfolio_sharpe', 0):.3f}")
            print(f"  Portfolio Max DD:   {pm.get('portfolio_max_dd', 0):.1%}")
            print(f"  Diversification:   {pm.get('diversification_ratio', 0):.2f}")
            print(f"  Strategies:        {pm.get('n_strategies', 0)}")

        weights = portfolio_result.get("weights", {})
        if weights:
            print(f"\n  -- Recommended Weights --")
            for k, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                print(f"    {k:<30} {w:.1%}")

        recs = portfolio_result.get("recommendations", [])
        if recs:
            print(f"\n  Recommendations:")
            for r in recs:
                print(f"    - {r}")

        print(f"\n  Portfolio analysis: {t3_elapsed:.1f}s")

    # ── Phase 4: Discover Engine (weekly: anomaly only; monthly: full) ──
    discover_result = None
    if args.mode in ("weekly", "monthly"):
        print_header("PHASE 4: DISCOVER ENGINE")
        t4 = time.time()

        daily_data = tf_data.get("1d", {})
        expanded = None
        if args.mode == "monthly":
            try:
                from run_full_scan import EXPANDED_GRIDS
                expanded = EXPANDED_GRIDS
            except ImportError:
                pass

        discover_result = run_discover(
            db, daily_data, active_syms, make_config,
            do_variants=(args.mode == "monthly"),
            do_anomalies=True,
            do_external=(args.mode == "monthly"),
            expanded_grids=expanded,
        )
        t4_elapsed = time.time() - t4

        anomalies = discover_result.get("anomalies", [])
        if anomalies:
            print(f"\n  -- Market Anomalies ({len(anomalies)}) --")
            for a in anomalies[:5]:
                print(f"    {a['symbol']:<12} severity={a['severity']:.2f}  {', '.join(a['flags'])}")

        corr_shifts = discover_result.get("correlation_shifts", [])
        if corr_shifts:
            print(f"\n  -- Correlation Shifts ({len(corr_shifts)}) --")
            for cs in corr_shifts[:5]:
                print(f"    {cs['pair']:<20} {cs['full_corr']:>+.2f} -> {cs['recent_corr']:>+.2f} ({cs['direction']})")

        variants = discover_result.get("variants", [])
        if variants:
            print(f"\n  -- New Variants ({len(variants)}) --")
            for v in variants[:5]:
                print(f"    {v['symbol']:<12} {v['strategy']:<14} Sharpe={v['sharpe']:.2f}")

        papers = discover_result.get("external_papers", [])
        if papers:
            print(f"\n  -- External Papers ({len(papers)}) --")
            for p in papers[:3]:
                print(f"    [{p['applicability']}/5] {p['title'][:70]}")

        print(f"\n  Discover: {t4_elapsed:.1f}s")

    # ── Phase 5: Compare & Report ──
    print_header("PHASE 5: COMPARE & REPORT")
    changes = phase3_compare(live_config, rescan_ranking, monitor_results)

    action_items = [c for c in changes if c["priority"] != "NONE"]
    if action_items:
        print(f"\n  {len(action_items)} action items:")
        for c in action_items:
            print(f"    [{c['priority']:>6}] {c['symbol']:<12} {c['action']:<8} {c['detail'][:60]}")
    else:
        print(f"\n  All strategies stable. No action needed.")

    # Generate report
    from quant_framework.research._report import generate_v4_report
    ts = datetime.now().strftime("%Y%m%d")
    report_path = os.path.join(reports_dir, f"daily_research_{ts}.md")
    generate_v4_report(
        monitor_results=monitor_results,
        changes=changes,
        data_updates=data_updates,
        output_path=report_path,
        db=db,
        portfolio_result=portfolio_result,
        discover_result=discover_result,
        optimize_results=optimize_results,
        mode=args.mode,
    )
    print(f"  Report saved: {report_path}")

    # ── Apply updates ──
    if args.apply and rescan_done:
        print_header("APPLYING UPDATES")
        config_out = os.path.join(base_dir, args.config)
        apply_updates(db, live_config, changes, config_out)

    # Save initial config version if none exists
    if db.get_latest_config() is None:
        db.save_config_version(live_config, summary="Initial config snapshot", n_changes=0)

    db.close()

    total_time = time.time() - t_global
    print("\n" + "=" * 80)
    print(f"  V4 RESEARCH COMPLETE [{mode}] -- {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
