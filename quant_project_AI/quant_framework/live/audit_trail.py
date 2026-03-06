"""SQLite-backed audit trail for order lifecycle and events."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    internal_id      TEXT NOT NULL,
    exchange_order_id TEXT DEFAULT '',
    signal_ts        TEXT,
    submit_ts        TEXT,
    ack_ts           TEXT,
    fill_ts          TEXT,
    fill_price       REAL,
    signal_price     REAL,
    latency_ms       REAL,
    slippage_bps     REAL,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,
    detail      TEXT DEFAULT '{}',
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_orders_internal ON audit_orders(internal_id);
CREATE INDEX IF NOT EXISTS idx_audit_orders_created ON audit_orders(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_events_created ON audit_events(created_at);
"""


def _iso(ts: Optional[Union[datetime, float]]) -> Optional[str]:
    if ts is None:
        return None
    if isinstance(ts, float):
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


class AuditTrail:
    """SQLite-backed audit log for order lifecycle and generic events."""

    def __init__(self, db_path: str = "audit.db") -> None:
        p = Path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(p), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record_order_lifecycle(
        self,
        internal_id: str,
        exchange_order_id: str = "",
        *,
        signal_ts: Optional[Union[datetime, float]] = None,
        submit_ts: Optional[Union[datetime, float]] = None,
        ack_ts: Optional[Union[datetime, float]] = None,
        fill_ts: Optional[Union[datetime, float]] = None,
        fill_price: Optional[float] = None,
        signal_price: Optional[float] = None,
        latency_ms: Optional[float] = None,
        slippage_bps: Optional[float] = None,
    ) -> int:
        created = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO audit_orders
                   (internal_id, exchange_order_id, signal_ts, submit_ts, ack_ts,
                    fill_ts, fill_price, signal_price, latency_ms, slippage_bps, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    internal_id,
                    exchange_order_id,
                    _iso(signal_ts),
                    _iso(submit_ts),
                    _iso(ack_ts),
                    _iso(fill_ts),
                    fill_price,
                    signal_price,
                    latency_ms,
                    slippage_bps,
                    created,
                ),
            )
            self._conn.commit()
            return cur.lastrowid or 0

    def record_event(self, event_type: str, detail: Optional[Dict[str, Any]] = None) -> int:
        created = datetime.now(timezone.utc).isoformat()
        detail_json = json.dumps(detail or {})
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO audit_events (event_type, detail, created_at) VALUES (?,?,?)",
                (event_type, detail_json, created),
            )
            self._conn.commit()
            return cur.lastrowid or 0

    def get_order_audit(
        self,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
    ) -> pd.DataFrame:
        q = "SELECT * FROM audit_orders WHERE 1=1"
        params: List[Any] = []
        if start_date:
            q += " AND DATE(created_at) >= DATE(?)"
            params.append(str(start_date))
        if end_date:
            q += " AND DATE(created_at) <= DATE(?)"
            params.append(str(end_date))
        q += " ORDER BY id ASC"
        with self._lock:
            return pd.read_sql_query(q, self._conn, params=params)

    def generate_daily_audit_report(self, report_date: Optional[Union[date, str]] = None) -> str:
        d = report_date or date.today()
        if isinstance(d, str):
            d = date.fromisoformat(d)
        start = d.isoformat()
        with self._lock:
            orders = pd.read_sql_query(
                "SELECT * FROM audit_orders WHERE DATE(created_at) = DATE(?) ORDER BY id",
                self._conn,
                params=[start],
            )
            events = pd.read_sql_query(
                "SELECT * FROM audit_events WHERE DATE(created_at) = DATE(?) ORDER BY id",
                self._conn,
                params=[start],
            )
        lines = [
            f"# Daily Audit Report — {d}",
            "",
            "## Order Lifecycle",
            "",
        ]
        if orders.empty:
            lines.append("No orders recorded.")
        else:
            lines.append("| ID | Internal | Exchange | Signal TS | Submit TS | Fill TS | Fill Price | Latency ms | Slippage bps |")
            lines.append("|----|----------|----------|-----------|-----------|---------|------------|-------------|---------------|")
            for _, row in orders.iterrows():
                lines.append(
                    f"| {row.get('id','')} | {row.get('internal_id','')} | "
                    f"{row.get('exchange_order_id','')} | {row.get('signal_ts','')} | "
                    f"{row.get('submit_ts','')} | {row.get('fill_ts','')} | "
                    f"{row.get('fill_price','')} | {row.get('latency_ms','')} | "
                    f"{row.get('slippage_bps','')} |"
                )
        lines.extend(["", "## Events", ""])
        if events.empty:
            lines.append("No events recorded.")
        else:
            lines.append("| ID | Type | Detail | Created |")
            lines.append("|----|------|--------|--------|")
            for _, row in events.iterrows():
                lines.append(
                    f"| {row.get('id','')} | {row.get('event_type','')} | "
                    f"{row.get('detail','')} | {row.get('created_at','')} |"
                )
        return "\n".join(lines)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
