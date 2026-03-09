"""Online feature engine for incremental live feature snapshots.

All indicator computations delegate to ``VectorizedIndicators`` (Numba-
accelerated) so that online values are numerically identical to offline /
research computations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

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


class OnlineFeatureEngine:
    """Computes canonical features from live rolling arrays."""

    def __init__(
        self,
        registry: Optional[FeatureRegistry] = None,
        feature_set_version: str = "core_v1",
    ) -> None:
        self._registry = registry or default_feature_registry()
        self._feature_set_version = feature_set_version
        self._latest: Dict[tuple[str, str], FeatureSnapshot] = {}

    @property
    def feature_set_version(self) -> str:
        return self._feature_set_version

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

        values: Dict[str, float] = {}
        n = len(close)
        if n:
            values["close"] = float(close[-1])
            values["open"] = float(open_[-1]) if len(open_) else values["close"]
            values["high"] = float(high[-1]) if len(high) else values["close"]
            values["low"] = float(low[-1]) if len(low) else values["close"]
            values["volume"] = float(volume[-1]) if len(volume) else 0.0
            values["return_1"] = float((close[-1] / close[-2] - 1.0)) if n > 1 and close[-2] else 0.0

            ma10 = _VI.ma(close, 10)
            values["ma_10"] = float(ma10[-1]) if not np.isnan(ma10[-1]) else float(np.mean(close))
            ma20 = _VI.ma(close, 20)
            values["ma_20"] = float(ma20[-1]) if not np.isnan(ma20[-1]) else float(np.mean(close))

            rsi = _VI.rsi(close, 14)
            values["rsi_14"] = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0

            if n >= 2:
                sample = close[-20:] if n >= 20 else close
                rets = np.diff(sample) / np.clip(sample[:-1], 1e-12, None)
                values["volatility_20"] = float(np.std(rets)) if len(rets) else 0.0
            else:
                values["volatility_20"] = 0.0

            if len(high) >= n and len(low) >= n:
                atr = _VI.atr(high, low, close, 14)
                values["atr_14"] = float(atr[-1]) if not np.isnan(atr[-1]) else 0.0

        snap = FeatureSnapshot(
            symbol=symbol,
            interval=interval,
            timestamp=(event_time or datetime.now(timezone.utc)).isoformat(),
            feature_set_version=self._feature_set_version,
            values=values,
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
