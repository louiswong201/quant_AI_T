"""
延迟监控：记录关键路径耗时并输出基础统计。
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict

import numpy as np


class LatencyTracker:
    """轻量延迟追踪器（毫秒）。"""

    def __init__(self, max_samples: int = 10000):
        self._samples: Deque[float] = deque(maxlen=max_samples)

    def record_ms(self, value_ms: float) -> None:
        if value_ms >= 0:
            self._samples.append(float(value_ms))

    def summary(self) -> Dict[str, float]:
        if not self._samples:
            return {
                "count": 0.0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "max_ms": 0.0,
            }
        arr = np.asarray(self._samples, dtype=np.float64)
        return {
            "count": float(len(arr)),
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(np.max(arr)),
        }

