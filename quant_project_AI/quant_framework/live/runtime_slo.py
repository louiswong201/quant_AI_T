"""Runtime SLOs and latency/correctness metrics for live trading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .latency import LatencyTracker


@dataclass(frozen=True)
class LiveRuntimeSLO:
    """Service-level objectives for the bar-driven live runtime."""

    bar_to_signal_p99_ms: float = 10.0
    signal_to_submit_p99_ms: float = 5.0
    max_feed_lag_ms: float = 5_000.0
    max_feature_freshness_ms: float = 5_000.0


class RuntimeMetrics:
    """Tracks hot-path latency, queue depth, and correctness counters."""

    def __init__(self, slo: LiveRuntimeSLO | None = None) -> None:
        self.slo = slo or LiveRuntimeSLO()
        self.bar_to_signal = LatencyTracker()
        self.signal_to_submit = LatencyTracker()
        self.feed_lag = LatencyTracker()
        self.feature_freshness = LatencyTracker()
        self._order_rejects = 0
        self._order_errors = 0
        self._order_timeouts = 0
        self._queue_depth = 0
        self._out_of_order_events = 0
        self._feed_gaps = 0

    def set_queue_depth(self, depth: int) -> None:
        self._queue_depth = max(0, int(depth))

    def record_order_reject(self) -> None:
        self._order_rejects += 1

    def record_order_error(self) -> None:
        self._order_errors += 1

    def record_order_timeout(self) -> None:
        self._order_timeouts += 1

    def record_out_of_order(self) -> None:
        self._out_of_order_events += 1

    def record_feed_gap(self) -> None:
        self._feed_gaps += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "slo": {
                "bar_to_signal_p99_ms": self.slo.bar_to_signal_p99_ms,
                "signal_to_submit_p99_ms": self.slo.signal_to_submit_p99_ms,
                "max_feed_lag_ms": self.slo.max_feed_lag_ms,
                "max_feature_freshness_ms": self.slo.max_feature_freshness_ms,
            },
            "latency_ms": {
                "bar_to_signal": self.bar_to_signal.summary(),
                "signal_to_submit": self.signal_to_submit.summary(),
                "feed_lag": self.feed_lag.summary(),
                "feature_freshness": self.feature_freshness.summary(),
            },
            "queue_depth": self._queue_depth,
            "order_rejects": self._order_rejects,
            "order_errors": self._order_errors,
            "order_timeouts": self._order_timeouts,
            "out_of_order_events": self._out_of_order_events,
            "feed_gaps": self._feed_gaps,
        }

    def health(self) -> Dict[str, Any]:
        lat = self.summary()["latency_ms"]
        bar_p99 = lat["bar_to_signal"]["p99_ms"]
        submit_p99 = lat["signal_to_submit"]["p99_ms"]
        feed_p99 = lat["feed_lag"]["p99_ms"]
        feature_p99 = lat["feature_freshness"]["p99_ms"]
        return {
            "bar_to_signal_ok": bar_p99 <= self.slo.bar_to_signal_p99_ms,
            "signal_to_submit_ok": submit_p99 <= self.slo.signal_to_submit_p99_ms,
            "feed_lag_ok": feed_p99 <= self.slo.max_feed_lag_ms,
            "feature_freshness_ok": feature_p99 <= self.slo.max_feature_freshness_ms,
            "queue_depth": self._queue_depth,
            "order_errors": self._order_errors,
            "order_timeouts": self._order_timeouts,
            "out_of_order_events": self._out_of_order_events,
            "feed_gaps": self._feed_gaps,
        }
