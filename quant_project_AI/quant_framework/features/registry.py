"""Feature registry for offline/online parity."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from .specs import FeatureSpec


class FeatureRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> Optional[FeatureSpec]:
        return self._specs.get(name)

    def all(self) -> Dict[str, FeatureSpec]:
        return dict(self._specs)

    def names(self) -> Iterable[str]:
        return self._specs.keys()


def default_feature_registry() -> FeatureRegistry:
    reg = FeatureRegistry()
    reg.register(FeatureSpec("close", "v1", ("close",)))
    reg.register(FeatureSpec("open", "v1", ("open",)))
    reg.register(FeatureSpec("high", "v1", ("high",)))
    reg.register(FeatureSpec("low", "v1", ("low",)))
    reg.register(FeatureSpec("volume", "v1", ("volume",)))
    reg.register(FeatureSpec("return_1", "v1", ("close",), lookback=2))
    reg.register(FeatureSpec("ma_10", "v1", ("close",), lookback=10, params={"window": 10}))
    reg.register(FeatureSpec("ma_20", "v1", ("close",), lookback=20, params={"window": 20}))
    reg.register(FeatureSpec("rsi_14", "v1", ("close",), lookback=14, params={"window": 14}))
    reg.register(FeatureSpec("volatility_20", "v1", ("close",), lookback=20, params={"window": 20}))
    return reg
