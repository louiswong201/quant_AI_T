"""Online feature engine with O(1) incremental indicator updates.

Maintains per-symbol running state (cumulative sums for MA, Wilder
averages for RSI, exponential running values for ATR) so that each
incoming bar requires only constant work — independent of the rolling
window length.

First call (cold start) uses VectorizedIndicators for a full-window
bootstrap; subsequent calls update incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..data.indicators import VectorizedIndicators as _VI
from .registry import FeatureRegistry, default_feature_registry


@dataclass
class FeatureSnapshot:
    symbol: str
    interval: str
    timestamp: str
    feature_set_version: str
    values: Dict[str, float] = field(default_factory=dict)


class _IncrementalState:
    """Per-(symbol, interval) running indicator state."""

    __slots__ = (
        "bar_count", "last_close",
        "ma10_sum", "ma10_buf", "ma10_idx",
        "ma20_sum", "ma20_buf", "ma20_idx",
        "rsi_avg_gain", "rsi_avg_loss",
        "atr_avg", "prev_close",
        "vol_ret_buf", "vol_idx", "vol_count",
    )

    def __init__(self) -> None:
        self.bar_count: int = 0
        self.last_close: float = 0.0

        self.ma10_sum: float = 0.0
        self.ma10_buf: np.ndarray = np.zeros(10, dtype=np.float64)
        self.ma10_idx: int = 0

        self.ma20_sum: float = 0.0
        self.ma20_buf: np.ndarray = np.zeros(20, dtype=np.float64)
        self.ma20_idx: int = 0

        self.rsi_avg_gain: float = 0.0
        self.rsi_avg_loss: float = 0.0

        self.atr_avg: float = 0.0
        self.prev_close: float = 0.0

        self.vol_ret_buf: np.ndarray = np.zeros(20, dtype=np.float64)
        self.vol_idx: int = 0
        self.vol_count: int = 0


class OnlineFeatureEngine:
    """Computes canonical features from live rolling arrays.

    After the initial bootstrap, each ``update()`` call runs in O(1) time
    regardless of the rolling window size.
    """

    def __init__(
        self,
        registry: Optional[FeatureRegistry] = None,
        feature_set_version: str = "core_v1",
    ) -> None:
        self._registry = registry or default_feature_registry()
        self._feature_set_version = feature_set_version
        self._latest: Dict[Tuple[str, str], FeatureSnapshot] = {}
        self._states: Dict[Tuple[str, str], _IncrementalState] = {}

    @property
    def feature_set_version(self) -> str:
        return self._feature_set_version

    def _bootstrap(self, st: _IncrementalState,
                   close: np.ndarray, high: np.ndarray,
                   low: np.ndarray) -> None:
        """Cold-start: initialise running state from full arrays."""
        n = len(close)
        st.bar_count = n
        st.last_close = float(close[-1]) if n else 0.0

        tail10 = close[-10:] if n >= 10 else close
        st.ma10_sum = float(tail10.sum())
        st.ma10_buf[:len(tail10)] = tail10
        st.ma10_idx = len(tail10) % 10

        tail20 = close[-20:] if n >= 20 else close
        st.ma20_sum = float(tail20.sum())
        st.ma20_buf[:len(tail20)] = tail20
        st.ma20_idx = len(tail20) % 20

        if n > 14:
            rsi_full = _VI.rsi(close, 14)
            last_valid = -1
            for i in range(n - 1, -1, -1):
                if not np.isnan(rsi_full[i]):
                    last_valid = i
                    break
            if last_valid >= 0:
                deltas = np.diff(close[:last_valid + 1])
                gains = np.where(deltas > 0, deltas, 0.0)
                losses = np.where(deltas < 0, -deltas, 0.0)
                ag = float(np.mean(gains[:14]))
                al = float(np.mean(losses[:14]))
                for j in range(14, len(deltas)):
                    ag = (ag * 13 + float(gains[j])) / 14
                    al = (al * 13 + float(losses[j])) / 14
                st.rsi_avg_gain = ag
                st.rsi_avg_loss = al

        if n > 14 and len(high) >= n and len(low) >= n:
            atr_full = _VI.atr(high, low, close, 14)
            for i in range(n - 1, -1, -1):
                if not np.isnan(atr_full[i]):
                    st.atr_avg = float(atr_full[i])
                    break
            st.prev_close = float(close[-1])

        vol_n = min(n, 20)
        if vol_n >= 2:
            rets = np.diff(close[-vol_n:]) / np.clip(close[-vol_n:-1], 1e-12, None)
            st.vol_ret_buf[:len(rets)] = rets
            st.vol_idx = len(rets) % 20
            st.vol_count = len(rets)

    def update(
        self,
        symbol: str,
        interval: str,
        arrays: Dict[str, np.ndarray],
        *,
        event_time: Optional[datetime] = None,
    ) -> FeatureSnapshot:
        close = np.asarray(arrays.get("close", np.empty(0)), dtype=np.float64)
        open_ = np.asarray(arrays.get("open", np.empty(0)), dtype=np.float64)
        high = np.asarray(arrays.get("high", np.empty(0)), dtype=np.float64)
        low = np.asarray(arrays.get("low", np.empty(0)), dtype=np.float64)
        volume = np.asarray(arrays.get("volume", np.empty(0)), dtype=np.float64)

        key = (symbol, interval)
        st = self._states.get(key)
        n = len(close)

        if st is None or st.bar_count == 0:
            st = _IncrementalState()
            self._states[key] = st
            if n > 0:
                self._bootstrap(st, close, high, low)
            return self._emit_snapshot_from_arrays(
                symbol, interval, close, open_, high, low, volume, st, event_time
            )

        if n == 0:
            return self._empty_snapshot(symbol, interval, event_time)

        c = float(close[-1])
        prev_c = st.last_close
        st.last_close = c
        st.bar_count += 1

        values: Dict[str, float] = {}
        values["close"] = c
        values["open"] = float(open_[-1]) if len(open_) else c
        values["high"] = float(high[-1]) if len(high) else c
        values["low"] = float(low[-1]) if len(low) else c
        values["volume"] = float(volume[-1]) if len(volume) else 0.0
        values["return_1"] = (c / prev_c - 1.0) if prev_c > 1e-12 else 0.0

        old10 = st.ma10_buf[st.ma10_idx]
        st.ma10_sum += c - old10
        st.ma10_buf[st.ma10_idx] = c
        st.ma10_idx = (st.ma10_idx + 1) % 10
        values["ma_10"] = st.ma10_sum / min(st.bar_count, 10)

        old20 = st.ma20_buf[st.ma20_idx]
        st.ma20_sum += c - old20
        st.ma20_buf[st.ma20_idx] = c
        st.ma20_idx = (st.ma20_idx + 1) % 20
        values["ma_20"] = st.ma20_sum / min(st.bar_count, 20)

        delta = c - prev_c
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        st.rsi_avg_gain = (st.rsi_avg_gain * 13 + gain) / 14
        st.rsi_avg_loss = (st.rsi_avg_loss * 13 + loss) / 14
        if st.rsi_avg_loss < 1e-14:
            values["rsi_14"] = 100.0
        else:
            rs = st.rsi_avg_gain / st.rsi_avg_loss
            values["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

        if prev_c > 1e-12:
            ret = c / prev_c - 1.0
        else:
            ret = 0.0
        st.vol_ret_buf[st.vol_idx] = ret
        st.vol_idx = (st.vol_idx + 1) % 20
        st.vol_count = min(st.vol_count + 1, 20)
        if st.vol_count > 0:
            used = st.vol_ret_buf[:st.vol_count]
            values["volatility_20"] = float(np.std(used))
        else:
            values["volatility_20"] = 0.0

        if len(high) > 0 and len(low) > 0:
            h_val = float(high[-1])
            l_val = float(low[-1])
            tr = max(h_val - l_val, abs(h_val - st.prev_close), abs(l_val - st.prev_close))
            if st.atr_avg > 0:
                st.atr_avg = (st.atr_avg * 13 + tr) / 14
            else:
                st.atr_avg = tr
            values["atr_14"] = st.atr_avg
            st.prev_close = c

        snap = FeatureSnapshot(
            symbol=symbol, interval=interval,
            timestamp=(event_time or datetime.now(timezone.utc)).isoformat(),
            feature_set_version=self._feature_set_version, values=values,
        )
        self._latest[key] = snap
        return snap

    def _emit_snapshot_from_arrays(
        self, symbol: str, interval: str,
        close: np.ndarray, open_: np.ndarray,
        high: np.ndarray, low: np.ndarray, volume: np.ndarray,
        st: _IncrementalState,
        event_time: Optional[datetime],
    ) -> FeatureSnapshot:
        """Build snapshot from full arrays (cold start)."""
        values: Dict[str, float] = {}
        n = len(close)
        if n:
            values["close"] = float(close[-1])
            values["open"] = float(open_[-1]) if len(open_) else values["close"]
            values["high"] = float(high[-1]) if len(high) else values["close"]
            values["low"] = float(low[-1]) if len(low) else values["close"]
            values["volume"] = float(volume[-1]) if len(volume) else 0.0
            values["return_1"] = float(close[-1] / close[-2] - 1.0) if n > 1 and close[-2] else 0.0
            values["ma_10"] = st.ma10_sum / min(n, 10)
            values["ma_20"] = st.ma20_sum / min(n, 20)
            if st.rsi_avg_loss < 1e-14:
                values["rsi_14"] = 100.0 if st.rsi_avg_gain > 0 else 50.0
            else:
                rs = st.rsi_avg_gain / st.rsi_avg_loss
                values["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)
            if st.vol_count > 0:
                values["volatility_20"] = float(np.std(st.vol_ret_buf[:st.vol_count]))
            else:
                values["volatility_20"] = 0.0
            if st.atr_avg > 0:
                values["atr_14"] = st.atr_avg

        snap = FeatureSnapshot(
            symbol=symbol, interval=interval,
            timestamp=(event_time or datetime.now(timezone.utc)).isoformat(),
            feature_set_version=self._feature_set_version, values=values,
        )
        self._latest[(symbol, interval)] = snap
        return snap

    def _empty_snapshot(self, symbol: str, interval: str,
                        event_time: Optional[datetime]) -> FeatureSnapshot:
        snap = FeatureSnapshot(
            symbol=symbol, interval=interval,
            timestamp=(event_time or datetime.now(timezone.utc)).isoformat(),
            feature_set_version=self._feature_set_version, values={},
        )
        self._latest[(symbol, interval)] = snap
        return snap

    def latest(self, symbol: str, interval: str) -> Optional[FeatureSnapshot]:
        return self._latest.get((symbol, interval))

    def latest_all(self) -> Dict[str, Dict[str, Any]]:
        return {
            f"{sym}:{iv}": {
                "timestamp": snap.timestamp,
                "feature_set_version": snap.feature_set_version,
                "values": dict(snap.values),
            }
            for (sym, iv), snap in self._latest.items()
        }
