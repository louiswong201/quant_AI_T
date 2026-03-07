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

_PF_CAP = 99.99
_POS_EPS = 1e-12

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
                "SELECT pnl, commission, metadata FROM trades WHERE pnl IS NOT NULL"
            ).fetchall()
        if not rows:
            return {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "breakeven_trades": 0,
                "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "largest_win": 0.0, "largest_loss": 0.0, "profit_factor": 0.0,
                "profit_factor_unbounded": False, "profit_factor_capped": False,
                "total_pnl": 0.0, "total_commission": 0.0,
                "expectancy": 0.0, "win_streak": 0, "loss_streak": 0,
            }

        realized_pnls: List[float] = []
        realized_commissions: List[float] = []
        for pnl, commission, metadata_raw in rows:
            metadata = self._parse_metadata(metadata_raw)
            realized = metadata.get("realized")
            if realized is None:
                realized = abs(float(pnl or 0.0)) > _POS_EPS
            if realized:
                realized_pnls.append(float(pnl or 0.0))
                realized_commissions.append(float(commission or 0.0))

        if not realized_pnls:
            return {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "breakeven_trades": 0,
                "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "largest_win": 0.0, "largest_loss": 0.0, "profit_factor": 0.0,
                "profit_factor_unbounded": False, "profit_factor_capped": False,
                "total_pnl": 0.0, "total_commission": 0.0,
                "expectancy": 0.0, "win_streak": 0, "loss_streak": 0,
            }

        pnls = realized_pnls
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        breakeven = [p for p in pnls if abs(p) <= _POS_EPS]

        max_w_streak, max_l_streak, cur_w, cur_l = 0, 0, 0, 0
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
                cur_w = 0
                cur_l = 0

        total_win = sum(wins) if wins else 0.0
        total_loss = abs(sum(losses)) if losses else 0.0
        total_trades = len(pnls)
        total_pnl = float(sum(pnls))
        total_commission = float(sum(realized_commissions))
        pf_unbounded = total_loss <= 1e-12 and total_win > 0
        if total_loss > 1e-12:
            pf_raw = total_win / total_loss
            profit_factor = min(pf_raw, _PF_CAP)
            pf_capped = pf_raw > _PF_CAP
        else:
            profit_factor = _PF_CAP if total_win > 0 else 0.0
            pf_capped = pf_unbounded

        return {
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "breakeven_trades": len(breakeven),
            "win_rate": len(wins) / (len(wins) + len(losses)) * 100 if (len(wins) + len(losses)) else 0.0,
            "avg_win": total_win / len(wins) if wins else 0.0,
            "avg_loss": -total_loss / len(losses) if losses else 0.0,
            "largest_win": max(wins) if wins else 0.0,
            "largest_loss": min(losses) if losses else 0.0,
            "profit_factor": profit_factor,
            "profit_factor_unbounded": pf_unbounded,
            "profit_factor_capped": pf_capped,
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
            "SELECT timestamp, symbol, strategy, pnl, metadata "
            "FROM trades WHERE pnl IS NOT NULL AND strategy != '' "
            "ORDER BY id DESC LIMIT ?"
        )
        with self._lock:
            df = pd.read_sql_query(q, self._conn, params=[limit])
        if df.empty:
            return []

        df["metadata"] = df["metadata"].apply(self._parse_metadata)
        df["realized"] = df.apply(
            lambda row: row["metadata"].get("realized")
            if row["metadata"].get("realized") is not None
            else abs(float(row["pnl"] or 0.0)) > _POS_EPS,
            axis=1,
        )
        df = df[df["realized"]].copy()
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
            if total_loss_abs > 1e-12:
                pf = min(total_win / total_loss_abs, _PF_CAP)
            else:
                pf = _PF_CAP if total_win > 0 else 0.0
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

    @staticmethod
    def _parse_metadata(raw: Any) -> Dict[str, Any]:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _normalize_positions(raw: Any) -> Dict[str, float]:
        if not raw:
            return {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                return {}
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, float] = {}
        for symbol, qty in raw.items():
            try:
                val = float(qty)
            except (TypeError, ValueError):
                continue
            if abs(val) > _POS_EPS:
                out[str(symbol)] = val
        return out

    def get_latest_account_state(self, fallback_initial_cash: float = 100_000.0) -> Dict[str, Any]:
        """Best-effort recovery of broker cash/positions/entry prices from journal."""
        self._flush()
        with self._lock:
            first_snapshot = self._conn.execute(
                "SELECT cash FROM equity_snapshots ORDER BY id ASC LIMIT 1"
            ).fetchone()
            latest_snapshot = self._conn.execute(
                "SELECT timestamp, equity, cash, positions FROM equity_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
            trades = self._conn.execute(
                "SELECT symbol, side, shares, price, commission FROM trades ORDER BY id ASC"
            ).fetchall()

        initial_cash = float(first_snapshot[0]) if first_snapshot and first_snapshot[0] is not None else float(fallback_initial_cash)
        replay_cash = initial_cash
        replay_positions: Dict[str, float] = {}
        replay_entries: Dict[str, float] = {}

        for symbol, side, shares, price, commission in trades:
            sym = str(symbol)
            action = str(side).lower()
            qty = float(shares or 0.0)
            px = float(price or 0.0)
            comm = float(commission or 0.0)
            if not sym or qty <= 0 or action not in {"buy", "sell"}:
                continue

            if action == "buy":
                replay_cash -= px * qty + comm
                old_pos = replay_positions.get(sym, 0.0)
                new_pos = old_pos + qty
                if old_pos < 0 and new_pos > 0:
                    replay_entries[sym] = px
                elif old_pos < 0 and abs(new_pos) <= _POS_EPS:
                    replay_entries.pop(sym, None)
                elif old_pos <= 0 and new_pos > 0:
                    replay_entries[sym] = px
                elif old_pos > 0:
                    old_entry = replay_entries.get(sym, px)
                    replay_entries[sym] = (old_entry * old_pos + px * qty) / new_pos
            else:
                replay_cash += px * qty - comm
                old_pos = replay_positions.get(sym, 0.0)
                new_pos = old_pos - qty
                if old_pos > 0 and new_pos <= 0:
                    replay_entries.pop(sym, None)
                    if new_pos < 0:
                        replay_entries[sym] = px
                elif old_pos <= 0 and new_pos < 0:
                    old_entry = replay_entries.get(sym, px)
                    replay_entries[sym] = ((old_entry * abs(old_pos)) + (px * qty)) / abs(new_pos)

            if abs(new_pos) <= _POS_EPS:
                replay_positions.pop(sym, None)
                replay_entries.pop(sym, None)
            else:
                replay_positions[sym] = new_pos

        if latest_snapshot:
            snapshot_positions = self._normalize_positions(latest_snapshot[3])
            snapshot_cash = float(latest_snapshot[2] or replay_cash)
            positions = snapshot_positions or replay_positions
            cash = snapshot_cash
            timestamp = str(latest_snapshot[0] or "")
            equity = float(latest_snapshot[1] or cash)
        else:
            positions = replay_positions
            cash = replay_cash
            timestamp = ""
            equity = cash

        entry_prices = {
            sym: float(replay_entries[sym])
            for sym, qty in positions.items()
            if sym in replay_entries and abs(qty) > _POS_EPS
        }

        return {
            "initial_cash": initial_cash,
            "cash": cash,
            "equity": equity,
            "positions": positions,
            "entry_prices": entry_prices,
            "snapshot_timestamp": timestamp,
            "has_recovery_data": bool(latest_snapshot or trades),
        }

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
