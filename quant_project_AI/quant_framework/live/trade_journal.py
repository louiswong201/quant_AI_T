"""SQLite-backed trade and equity journal for paper/live trading.

Zero external dependencies — uses Python's built-in sqlite3 module.
Thread-safe via sqlite3's serialized mode.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    shares      REAL    NOT NULL,
    price       REAL    NOT NULL,
    commission  REAL    DEFAULT 0,
    pnl         REAL    DEFAULT 0,
    strategy    TEXT    DEFAULT '',
    metadata    TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    equity      REAL    NOT NULL,
    cash        REAL    NOT NULL,
    positions   TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    strategy    TEXT    DEFAULT '',
    signal_type TEXT    NOT NULL,
    params      TEXT    DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp);
"""


class TradeJournal:
    """Persistent trade and equity history backed by SQLite.

    Write operations are batched: rows are queued in memory and flushed to
    SQLite every ``flush_interval`` seconds or every ``flush_size`` rows,
    whichever comes first.  This avoids blocking the async event loop with
    per-row commits (the previous bottleneck).  ``record_trade`` still
    returns immediately; the actual SQLite INSERT is deferred.

    Reads always call ``_maybe_flush()`` first so they see the latest data.
    """

    _FLUSH_INTERVAL = 2.0
    _FLUSH_SIZE = 50

    def __init__(self, db_path: str = "paper_trading.db"):
        p = Path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(p), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._trade_buf: List[tuple] = []
        self._equity_buf: List[tuple] = []
        self._signal_buf: List[tuple] = []
        self._last_flush = time.monotonic()

    def _maybe_flush(self) -> None:
        """Flush pending writes if the buffer or timer threshold is reached."""
        total = len(self._trade_buf) + len(self._equity_buf) + len(self._signal_buf)
        elapsed = time.monotonic() - self._last_flush
        if total == 0:
            return
        if total < self._FLUSH_SIZE and elapsed < self._FLUSH_INTERVAL:
            return
        self._flush()

    def _flush(self) -> None:
        """Flush all pending writes to SQLite in a single transaction."""
        with self._lock:
            if self._trade_buf:
                self._conn.executemany(
                    "INSERT INTO trades (timestamp, symbol, side, shares, price, "
                    "commission, pnl, strategy, metadata) VALUES (?,?,?,?,?,?,?,?,?)",
                    self._trade_buf,
                )
                self._trade_buf.clear()
            if self._equity_buf:
                self._conn.executemany(
                    "INSERT INTO equity_snapshots (timestamp, equity, cash, positions) "
                    "VALUES (?,?,?,?)",
                    self._equity_buf,
                )
                self._equity_buf.clear()
            if self._signal_buf:
                self._conn.executemany(
                    "INSERT INTO signals (timestamp, symbol, strategy, signal_type, params) "
                    "VALUES (?,?,?,?,?)",
                    self._signal_buf,
                )
                self._signal_buf.clear()
            self._conn.commit()
        self._last_flush = time.monotonic()

    def record_trade(
        self,
        symbol: str,
        side: str,
        shares: float,
        price: float,
        commission: float = 0.0,
        pnl: float = 0.0,
        strategy: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> int:
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        self._trade_buf.append(
            (ts, symbol, side, shares, price, commission, pnl, strategy,
             json.dumps(metadata or {}))
        )
        self._maybe_flush()
        return 0

    def record_equity(
        self,
        equity: float,
        cash: float,
        positions: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        self._equity_buf.append(
            (ts, equity, cash, json.dumps(positions or {}))
        )
        self._maybe_flush()

    def record_signal(
        self,
        symbol: str,
        signal_type: str,
        strategy: str = "",
        params: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        self._signal_buf.append(
            (ts, symbol, strategy, signal_type, json.dumps(params or {}))
        )
        self._maybe_flush()

    def get_trades(self, limit: int = 100, symbol: Optional[str] = None) -> pd.DataFrame:
        self._flush()
        q = "SELECT timestamp, symbol, side, shares, price, commission, pnl, strategy FROM trades"
        params: list = []
        if symbol:
            q += " WHERE symbol = ?"
            params.append(symbol)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return pd.read_sql_query(q, self._conn, params=params)

    def get_equity_curve(self, limit: int = 1000) -> pd.DataFrame:
        self._flush()
        q = "SELECT timestamp, equity, cash, positions FROM equity_snapshots ORDER BY id DESC LIMIT ?"
        with self._lock:
            df = pd.read_sql_query(q, self._conn, params=[limit])
        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_daily_pnl(self, days: int = 30) -> pd.DataFrame:
        self._flush()
        q = ("SELECT DATE(timestamp) as date, SUM(pnl) as daily_pnl, COUNT(*) as n_trades "
             "FROM trades GROUP BY DATE(timestamp) ORDER BY date DESC LIMIT ?")
        with self._lock:
            return pd.read_sql_query(q, self._conn, params=[days])

    def get_signals(self, limit: int = 100) -> pd.DataFrame:
        self._flush()
        q = "SELECT timestamp, symbol, strategy, signal_type, params FROM signals ORDER BY id DESC LIMIT ?"
        with self._lock:
            return pd.read_sql_query(q, self._conn, params=[limit])

    def get_trade_stats(self) -> Dict[str, Any]:
        """Compute detailed trade statistics for dashboard display."""
        self._flush()
        with self._lock:
            rows = self._conn.execute(
                "SELECT pnl FROM trades WHERE pnl != 0"
            ).fetchall()
            comm_row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(pnl), 0), COALESCE(SUM(commission), 0) FROM trades"
            ).fetchone()
        if not rows:
            return {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "largest_win": 0.0, "largest_loss": 0.0, "profit_factor": 0.0,
                "total_pnl": 0.0, "total_commission": 0.0,
                "expectancy": 0.0, "win_streak": 0, "loss_streak": 0,
            }
        pnls = [r[0] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        max_w_streak, max_l_streak, cur_w, cur_l = 0, 0, 0, 0
        for p in pnls:
            if p > 0:
                cur_w += 1
                cur_l = 0
                max_w_streak = max(max_w_streak, cur_w)
            else:
                cur_l += 1
                cur_w = 0
                max_l_streak = max(max_l_streak, cur_l)

        total_win = sum(wins) if wins else 0.0
        total_loss = abs(sum(losses)) if losses else 0.0
        total_trades, total_pnl, total_commission = comm_row or (0, 0.0, 0.0)

        return {
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(pnls) * 100 if pnls else 0.0,
            "avg_win": total_win / len(wins) if wins else 0.0,
            "avg_loss": -total_loss / len(losses) if losses else 0.0,
            "largest_win": max(wins) if wins else 0.0,
            "largest_loss": min(losses) if losses else 0.0,
            "profit_factor": total_win / total_loss if total_loss > 0 else float("inf"),
            "total_pnl": total_pnl,
            "total_commission": total_commission,
            "expectancy": sum(pnls) / len(pnls) if pnls else 0.0,
            "win_streak": max_w_streak,
            "loss_streak": max_l_streak,
        }

    def get_strategy_trade_stats(self, limit: int = 2000) -> List[Dict[str, Any]]:
        """Aggregate strategy-level performance from recent realized trades."""
        self._flush()
        q = (
            "SELECT timestamp, symbol, strategy, pnl "
            "FROM trades WHERE pnl != 0 AND strategy != '' "
            "ORDER BY id DESC LIMIT ?"
        )
        with self._lock:
            df = pd.read_sql_query(q, self._conn, params=[limit])
        if df.empty:
            return []

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        rows: List[Dict[str, Any]] = []
        for (strategy, symbol), grp in df.groupby(["strategy", "symbol"], dropna=False):
            pnl = grp["pnl"].astype(float)
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            trades = int(len(pnl))
            total_pnl = float(pnl.sum())
            win_rate = float((len(wins) / trades) * 100.0) if trades else 0.0
            total_win = float(wins.sum()) if len(wins) else 0.0
            total_loss_abs = float(abs(losses.sum())) if len(losses) else 0.0
            pf = (total_win / total_loss_abs) if total_loss_abs > 1e-12 else float("inf")
            expectancy = float(total_pnl / trades) if trades else 0.0
            std = float(pnl.std(ddof=0)) if trades > 1 else 0.0
            sharpe_proxy = float((expectancy / std) * (trades ** 0.5)) if std > 1e-12 else 0.0

            grp_sorted = grp.sort_values("timestamp")
            cum = grp_sorted["pnl"].cumsum().astype(float)
            peak = cum.cummax()
            dd = float((peak - cum).max()) if len(cum) else 0.0

            rows.append({
                "strategy": str(strategy),
                "symbol": str(symbol),
                "trades": trades,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "profit_factor": float(pf),
                "expectancy": expectancy,
                "sharpe_proxy": sharpe_proxy,
                "max_dd_abs": dd,
                "last_timestamp": grp_sorted["timestamp"].iloc[-1].isoformat()
                if not grp_sorted.empty else "",
            })
        return rows

    def get_summary(self) -> Dict[str, Any]:
        self._flush()
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(pnl), 0), COALESCE(SUM(commission), 0) FROM trades"
            ).fetchone()
            total_trades, total_pnl, total_commission = row or (0, 0, 0)
            eq_row = self._conn.execute(
                "SELECT equity, cash FROM equity_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
        equity, cash = eq_row if eq_row else (0, 0)
        return {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "total_commission": total_commission,
            "current_equity": equity,
            "current_cash": cash,
        }

    def close(self) -> None:
        self._flush()
        with self._lock:
            self._conn.close()
