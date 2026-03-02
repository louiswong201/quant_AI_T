"""
Alpha engine — next-generation feature engineering beyond OHLCV.

Modules:
  - order_flow: Microstructure features (OFI, VPIN, Trade Imbalance)
  - cross_asset: Cross-market features (basis, rolling beta, lead-lag)
  - volatility: Volatility surface features (realised vol, vol-of-vol)
  - evaluator: Feature quality assessment (IC/IR, PCA, correlation)

Both the high-level OOP API (OrderFlowFeatures, VolatilityFeatures, ...)
and the low-level Numba kernels (_ofi_numba, _yang_zhang_vol_numba, ...)
are public. Strategies may use either interface.
"""

from .order_flow import (
    OrderFlowFeatures,
    _ofi_numba as ofi_numba,
    _vpin_numba as vpin_numba,
    _trade_imbalance_numba as trade_imbalance_numba,
)
from .cross_asset import CrossAssetFeatures
from .volatility import (
    VolatilityFeatures,
    _yang_zhang_vol_numba as yang_zhang_vol_numba,
    _vol_of_vol_numba as vol_of_vol_numba,
    _vol_ratio_numba as vol_ratio_numba,
)
from .evaluator import FeatureEvaluator

__all__ = [
    "OrderFlowFeatures",
    "CrossAssetFeatures",
    "VolatilityFeatures",
    "FeatureEvaluator",
    "ofi_numba",
    "vpin_numba",
    "trade_imbalance_numba",
    "yang_zhang_vol_numba",
    "vol_of_vol_numba",
    "vol_ratio_numba",
]
