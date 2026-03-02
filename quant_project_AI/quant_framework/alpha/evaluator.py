"""
Feature quality evaluator — IC/IR analysis, PCA, correlation filtering.

Scientific rigour per coding_guide.md Section 4:
  "Use rigorous mathematical logic (PCA, IC/IR analysis, correlation matrices)
   to filter out true orthogonal predictive features."

Methods:
  - Information Coefficient (IC): Spearman rank correlation between each
    feature and forward returns. IC > 0.02 is generally considered meaningful
    in cross-sectional equity factors; for time-series single-asset factors,
    IC > 0.05 is a reasonable threshold.
  - IC Information Ratio (ICIR): IC_mean / IC_std. Measures the consistency
    of the signal. ICIR > 0.5 indicates a reliably predictive feature.
  - PCA orthogonality: Reduces the feature space to identify redundant
    (highly correlated) features that add noise without new information.
  - Correlation matrix: Identifies collinear feature pairs for pruning.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _rank_array(arr: np.ndarray) -> np.ndarray:
    """Rank an array (handling NaN by assigning rank 0)."""
    n = len(arr)
    ranks = np.zeros(n, dtype=np.float64)
    valid = ~np.isnan(arr)
    valid_vals = arr[valid]
    order = np.argsort(valid_vals)
    ranked = np.empty(len(valid_vals), dtype=np.float64)
    for i, idx in enumerate(order):
        ranked[idx] = float(i + 1)
    ranks[valid] = ranked
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between x and y (NaN-safe)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 5:
        return np.nan
    rx = _rank_array(x[mask])
    ry = _rank_array(y[mask])
    n = len(rx)
    mx, my = np.mean(rx), np.mean(ry)
    cov = np.sum((rx - mx) * (ry - my)) / n
    sx = np.sqrt(np.sum((rx - mx) ** 2) / n)
    sy = np.sqrt(np.sum((ry - my) ** 2) / n)
    if sx * sy < 1e-20:
        return 0.0
    return float(cov / (sx * sy))


class FeatureEvaluator:
    """Evaluate predictive power and redundancy of alpha features."""

    @staticmethod
    def information_coefficient(
        features: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        rolling_window: int = 60,
    ) -> Dict[str, Dict[str, float]]:
        """Compute IC and ICIR for each feature.

        Args:
            features: {feature_name: values_array}
            forward_returns: 1-period forward returns
            rolling_window: window for computing rolling IC

        Returns:
            {feature_name: {"ic_mean", "ic_std", "icir", "ic_last"}}
        """
        n = len(forward_returns)
        result: Dict[str, Dict[str, float]] = {}

        for name, values in features.items():
            if len(values) != n:
                continue

            ics: List[float] = []
            for i in range(rolling_window, n):
                ic = _spearman_corr(
                    values[i - rolling_window: i],
                    forward_returns[i - rolling_window: i],
                )
                if not np.isnan(ic):
                    ics.append(ic)

            if len(ics) < 3:
                result[name] = {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan, "ic_last": np.nan}
                continue

            ic_arr = np.array(ics)
            ic_mean = float(np.mean(ic_arr))
            ic_std = float(np.std(ic_arr))
            icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

            result[name] = {
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "ic_last": float(ic_arr[-1]),
            }

        return result

    @staticmethod
    def correlation_matrix(
        features: Dict[str, np.ndarray],
    ) -> Tuple[List[str], np.ndarray]:
        """Compute pairwise Spearman correlation matrix.

        Returns:
            (feature_names, correlation_matrix)
        """
        names = sorted(features.keys())
        k = len(names)
        corr = np.eye(k, dtype=np.float64)

        for i in range(k):
            for j in range(i + 1, k):
                c = _spearman_corr(features[names[i]], features[names[j]])
                corr[i, j] = c
                corr[j, i] = c

        return names, corr

    @staticmethod
    def select_orthogonal(
        features: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        max_correlation: float = 0.7,
        min_icir: float = 0.3,
    ) -> List[str]:
        """Select orthogonal, predictive features.

        Algorithm:
          1. Compute IC/IR for all features.
          2. Sort by ICIR (descending).
          3. Greedily add features, rejecting any that has correlation > threshold
             with an already-selected feature.

        Returns:
            List of selected feature names, ordered by predictive power.
        """
        ic_results = FeatureEvaluator.information_coefficient(features, forward_returns)
        names, corr = FeatureEvaluator.correlation_matrix(features)
        name_to_idx = {n: i for i, n in enumerate(names)}

        candidates = [
            (name, stats["icir"])
            for name, stats in ic_results.items()
            if not np.isnan(stats["icir"]) and stats["icir"] >= min_icir
        ]
        candidates.sort(key=lambda x: -x[1])

        selected: List[str] = []
        selected_indices: List[int] = []

        for name, _ in candidates:
            idx = name_to_idx.get(name)
            if idx is None:
                continue

            too_correlated = False
            for sel_idx in selected_indices:
                if abs(corr[idx, sel_idx]) > max_correlation:
                    too_correlated = True
                    break

            if not too_correlated:
                selected.append(name)
                selected_indices.append(idx)

        return selected
