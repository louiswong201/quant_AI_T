"""
Model module — ML model base classes and utilities.

Planned extensions:
  - BasePredictor: abstract interface for signal predictors (sklearn, xgboost, ...)
  - FeaturePipeline: transform raw alpha features into model-ready tensors
  - ModelRegistry: version-tracked model storage (MLflow-compatible)

Currently provides the Protocol contract for future implementations.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class IPredictor(Protocol):
    """Contract for ML-based signal predictors."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return signal scores for each row in features."""
        ...

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train the model on historical features and targets."""
        ...

    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters for reproducibility."""
        ...


__all__ = ["IPredictor"]
