"""Persistent research database — the state memory for all V4 engines.

Five core tables:
  strategy_health   — daily per-symbol health snapshots (Sharpe trend, DD, win rate, …)
  param_history     — every optimization result with timestamp (drift tracking)
  config_versions   — full JSON snapshot on each config change (rollback support)
  regime_snapshots  — daily per-symbol regime probabilities
  strategy_library  — lifecycle management (IDEA → LIVE → RETIRED)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ResearchDB:
    """Thread-safe SQLite wrapper for the research state layer."""

    def __init__(self, db_path: str = "research.db"):
        self._db_path = str(db_path)
        self._local = threading.local()
        self._init_schema()

    # ── Connection management ────────────────────────────────────

    def _make_conn(self) -> sqlite3.Connection:
        """Create a new connection with pragmas and schema."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        if self._db_path != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        self._create_tables(conn)
        conn.commit()
        return conn

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._make_conn()
            self._local.conn = conn
        return conn

    @contextmanager
    def _cursor(self):
        conn = self._conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    # ── Schema ───────────────────────────────────────────────────

    def _init_schema(self):
        conn = self._make_conn()
        self._local.conn = conn

    def _create_tables(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS strategy_health (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL DEFAULT (datetime('now')),
                symbol      TEXT    NOT NULL,
                strategy    TEXT    NOT NULL,
                leverage    REAL    NOT NULL DEFAULT 1,
                interval    TEXT    NOT NULL DEFAULT '1d',
                sharpe_30d  REAL,
                drawdown_pct REAL,
                dd_duration INTEGER,
                trade_freq  REAL,
                win_rate    REAL,
                ret_pct     REAL,
                n_trades    INTEGER,
                status      TEXT    NOT NULL DEFAULT 'UNKNOWN'
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_health_sym_ts
            ON strategy_health(symbol, strategy, ts)
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS param_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL DEFAULT (datetime('now')),
                symbol      TEXT    NOT NULL,
                strategy    TEXT    NOT NULL,
                params      TEXT    NOT NULL,
                sharpe      REAL,
                wf_score    REAL,
                gate_score  REAL,
                leverage    REAL    NOT NULL DEFAULT 1,
                interval    TEXT    NOT NULL DEFAULT '1d',
                source      TEXT    NOT NULL DEFAULT 'scan',
                reason      TEXT
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_param_sym
            ON param_history(symbol, strategy, ts)
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS config_versions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL DEFAULT (datetime('now')),
                config_json TEXT    NOT NULL,
                diff_json   TEXT,
                summary     TEXT,
                n_changes   INTEGER NOT NULL DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS regime_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL DEFAULT (datetime('now')),
                symbol      TEXT    NOT NULL,
                trending    REAL    NOT NULL DEFAULT 0,
                mean_reverting REAL NOT NULL DEFAULT 0,
                high_vol    REAL    NOT NULL DEFAULT 0,
                compression REAL    NOT NULL DEFAULT 0,
                dominant    TEXT
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_regime_sym_ts
            ON regime_snapshots(symbol, ts)
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS strategy_library (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                kernel_name TEXT    NOT NULL,
                status      TEXT    NOT NULL DEFAULT 'IDEA',
                source      TEXT    NOT NULL DEFAULT 'scan',
                source_url  TEXT,
                created     TEXT    NOT NULL DEFAULT (datetime('now')),
                updated     TEXT    NOT NULL DEFAULT (datetime('now')),
                symbols     TEXT,
                best_params TEXT,
                regime_affinity TEXT,
                half_life_days REAL,
                gate_score  REAL,
                notes       TEXT,
                UNIQUE(name, kernel_name)
            )
        """)

    # ── strategy_health ──────────────────────────────────────────

    def record_health(
        self,
        symbol: str,
        strategy: str,
        *,
        leverage: float = 1.0,
        interval: str = "1d",
        sharpe_30d: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
        dd_duration: Optional[int] = None,
        trade_freq: Optional[float] = None,
        win_rate: Optional[float] = None,
        ret_pct: Optional[float] = None,
        n_trades: Optional[int] = None,
        status: str = "UNKNOWN",
    ):
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO strategy_health
                   (symbol, strategy, leverage, interval,
                    sharpe_30d, drawdown_pct, dd_duration,
                    trade_freq, win_rate, ret_pct, n_trades, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol, strategy, leverage, interval,
                 sharpe_30d, drawdown_pct, dd_duration,
                 trade_freq, win_rate, ret_pct, n_trades, status),
            )

    def get_health_trend(
        self, symbol: str, strategy: str, days: int = 30,
    ) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT * FROM strategy_health
                   WHERE symbol=? AND strategy=?
                   ORDER BY ts DESC LIMIT ?""",
                (symbol, strategy, days),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_latest_health(self, symbol: str, strategy: str) -> Optional[Dict]:
        rows = self.get_health_trend(symbol, strategy, days=1)
        return rows[0] if rows else None

    def get_all_latest_health(self) -> List[Dict[str, Any]]:
        """Latest health row per (symbol, strategy) pair (deduplicated via ROW_NUMBER)."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT * FROM (
                    SELECT h.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY h.symbol, h.strategy
                               ORDER BY h.ts DESC
                           ) as rn
                    FROM strategy_health h
                ) ranked WHERE rn = 1
                ORDER BY symbol, strategy
            """)
            return [dict(r) for r in cur.fetchall()]

    # ── param_history ────────────────────────────────────────────

    def record_param_update(
        self,
        symbol: str,
        strategy: str,
        params: Any,
        *,
        sharpe: Optional[float] = None,
        wf_score: Optional[float] = None,
        gate_score: Optional[float] = None,
        leverage: float = 1.0,
        interval: str = "1d",
        source: str = "scan",
        reason: Optional[str] = None,
    ):
        params_json = json.dumps(params if isinstance(params, list) else list(params))
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO param_history
                   (symbol, strategy, params, sharpe, wf_score, gate_score,
                    leverage, interval, source, reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (symbol, strategy, params_json, sharpe, wf_score, gate_score,
                 leverage, interval, source, reason),
            )

    def get_param_history(
        self, symbol: str, strategy: str, limit: int = 20,
    ) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT * FROM param_history
                   WHERE symbol=? AND strategy=?
                   ORDER BY ts DESC LIMIT ?""",
                (symbol, strategy, limit),
            )
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                if r.get("params"):
                    r["params"] = json.loads(r["params"])
            return rows

    def get_latest_params(self, symbol: str, strategy: str) -> Optional[Dict]:
        rows = self.get_param_history(symbol, strategy, limit=1)
        return rows[0] if rows else None

    # ── config_versions ──────────────────────────────────────────

    def save_config_version(
        self,
        config_dict: Dict[str, Any],
        summary: str = "",
        diff: Optional[Dict] = None,
        n_changes: int = 0,
    ):
        config_json = json.dumps(config_dict, default=str)
        diff_json = json.dumps(diff, default=str) if diff else None
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO config_versions
                   (config_json, diff_json, summary, n_changes)
                   VALUES (?,?,?,?)""",
                (config_json, diff_json, summary, n_changes),
            )

    def get_config_versions(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM config_versions ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                if r.get("config_json"):
                    r["config_json"] = json.loads(r["config_json"])
                if r.get("diff_json"):
                    r["diff_json"] = json.loads(r["diff_json"])
            return rows

    def get_latest_config(self) -> Optional[Dict]:
        rows = self.get_config_versions(limit=1)
        return rows[0] if rows else None

    # ── regime_snapshots ─────────────────────────────────────────

    def record_regime(
        self,
        symbol: str,
        trending: float = 0.0,
        mean_reverting: float = 0.0,
        high_vol: float = 0.0,
        compression: float = 0.0,
    ):
        vals = {"trending": trending, "mean_reverting": mean_reverting,
                "high_vol": high_vol, "compression": compression}
        dominant = max(vals, key=vals.get)
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO regime_snapshots
                   (symbol, trending, mean_reverting, high_vol, compression, dominant)
                   VALUES (?,?,?,?,?,?)""",
                (symbol, trending, mean_reverting, high_vol, compression, dominant),
            )

    def get_regime_history(
        self, symbol: str, days: int = 30,
    ) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT * FROM regime_snapshots
                   WHERE symbol=?
                   ORDER BY ts DESC LIMIT ?""",
                (symbol, days),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_latest_regime(self, symbol: str) -> Optional[Dict]:
        rows = self.get_regime_history(symbol, days=1)
        return rows[0] if rows else None

    # ── strategy_library ─────────────────────────────────────────

    def upsert_strategy(
        self,
        name: str,
        kernel_name: str,
        *,
        status: str = "IDEA",
        source: str = "scan",
        source_url: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        best_params: Optional[Any] = None,
        regime_affinity: Optional[Dict] = None,
        half_life_days: Optional[float] = None,
        gate_score: Optional[float] = None,
        notes: Optional[str] = None,
    ):
        syms_json = json.dumps(symbols) if symbols else None
        params_json = json.dumps(best_params) if best_params else None
        regime_json = json.dumps(regime_affinity) if regime_affinity else None
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO strategy_library
                   (name, kernel_name, status, source, source_url,
                    symbols, best_params, regime_affinity,
                    half_life_days, gate_score, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(name, kernel_name) DO UPDATE SET
                     status=excluded.status,
                     updated=datetime('now'),
                     symbols=COALESCE(excluded.symbols, symbols),
                     best_params=COALESCE(excluded.best_params, best_params),
                     regime_affinity=COALESCE(excluded.regime_affinity, regime_affinity),
                     half_life_days=COALESCE(excluded.half_life_days, half_life_days),
                     gate_score=COALESCE(excluded.gate_score, gate_score),
                     notes=COALESCE(excluded.notes, notes)""",
                (name, kernel_name, status, source, source_url,
                 syms_json, params_json, regime_json,
                 half_life_days, gate_score, notes),
            )

    def get_strategies_by_status(self, status: str) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM strategy_library WHERE status=? ORDER BY updated DESC",
                (status,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                for k in ("symbols", "best_params", "regime_affinity"):
                    if r.get(k):
                        r[k] = json.loads(r[k])
            return rows

    def get_all_strategies(self) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM strategy_library ORDER BY status, updated DESC")
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                for k in ("symbols", "best_params", "regime_affinity"):
                    if r.get(k):
                        r[k] = json.loads(r[k])
            return rows

    def update_strategy_status(self, name: str, kernel_name: str, new_status: str):
        with self._cursor() as cur:
            cur.execute(
                """UPDATE strategy_library
                   SET status=?, updated=datetime('now')
                   WHERE name=? AND kernel_name=?""",
                (new_status, name, kernel_name),
            )

    # ── Utilities ────────────────────────────────────────────────

    def table_counts(self) -> Dict[str, int]:
        tables = ["strategy_health", "param_history", "config_versions",
                   "regime_snapshots", "strategy_library"]
        counts = {}
        with self._cursor() as cur:
            for t in tables:
                cur.execute(f"SELECT COUNT(*) FROM {t}")
                counts[t] = cur.fetchone()[0]
        return counts
