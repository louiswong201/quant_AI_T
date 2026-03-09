"""Canonical feature specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: str
    inputs: Tuple[str, ...]
    lookback: int = 1
    frequency: str = "1d"
    warmup_policy: str = "drop_until_ready"
    null_policy: str = "nan"
    alignment_policy: str = "same_bar_close"
    params: Dict[str, Any] = field(default_factory=dict)
