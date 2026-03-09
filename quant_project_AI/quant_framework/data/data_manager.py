"""Data manager — unified access layer for structured market data.

Responsibilities:
  1. Load OHLCV data for a symbol + date range (with caching).
  2. Calculate technical indicators (delegates to VectorizedIndicators).
  3. Save data (with cache invalidation).
  4. Provide latest price for a symbol.
  5. Expose high-performance Polars lazy / NumPy array paths for
     strategies and the backtest engine.

Caching strategy:
  - If CacheManager is injected: LRU in-memory + disk Parquet cache with
    byte-level eviction and per-symbol invalidation on save.
  - If no CacheManager: process-local functools.lru_cache (128 entries).
    Cleared entirely on save (conservative invalidation).

Per coding_guide.md: the DataManager is the single entry point for
structured data — strategies and the backtest engine never access
Dataset or storage layers directly.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cache_manager import CacheManager, _data_key
from .dataset import Dataset

logger = logging.getLogger(__name__)

_LRU_MAXSIZE = 128


class DataManager:
    """Unified data access with optional multi-tier caching.

    Three access tiers (fastest → most compatible):
      1. ``load_arrays`` — dict of contiguous np.ndarray (Numba-ready)
      2. ``load_lazy``   — Polars LazyFrame (predicate + projection pushdown)
      3. ``load_data``   — pd.DataFrame (classic path, cached)

    Usage::

        dm = DataManager("data")
        df = dm.load_data("BTC", "2020-01-01", "2024-01-01")
        arrays = dm.load_arrays("BTC", "2020-01-01", "2024-01-01")
    """

    def __init__(
        self,
        data_dir: str = "data",
        use_parquet: bool = True,
        cache: Optional[CacheManager] = None,
        fast_io: bool = False,
    ) -> None:
        self.dataset = Dataset(data_dir=data_dir, use_parquet=use_parquet, fast_io=fast_io)
        self._cache = cache
        self._local_cache: OrderedDict[Tuple[str, str, str], Optional[pd.DataFrame]] = OrderedDict()

    # ── Pandas path (backward-compatible) ────────────────────────

    def load_data(
        self,
        symbol: str,
        start_date: str = "",
        end_date: str = "",
    ) -> Optional[pd.DataFrame]:
        """Load OHLCV data for a symbol within a date range.

        Returns a DataFrame with columns [date, open, high, low, close, volume]
        or None if data is unavailable.
        """
        if self._cache is not None:
            key = _data_key(symbol, start_date, end_date)
            data = self._cache.get(key, copy=True)
            if data is not None:
                return data
            data = self.dataset.load(symbol, start_date, end_date)
            if data is not None and not data.empty:
                self._cache.put(key, data)
            return data

        return self._load_data_cached(symbol, start_date, end_date)

    def _load_data_cached(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Instance-level LRU cache — no GC leak from functools.lru_cache."""
        key = (symbol, start_date, end_date)
        if key in self._local_cache:
            self._local_cache.move_to_end(key)
            return self._local_cache[key]
        data = self.dataset.load(symbol, start_date, end_date)
        self._local_cache[key] = data
        if len(self._local_cache) > _LRU_MAXSIZE:
            self._local_cache.popitem(last=False)
        return data

    # ── Polars lazy path (zero-copy) ─────────────────────────────

    def load_lazy(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> Any:
        """Return a Polars LazyFrame with predicate + projection pushdown.

        Requires ``polars`` to be installed.  Callers call ``.collect()``
        when ready to materialise the data.
        """
        return self.dataset.load_lazy(symbol, start_date, end_date, columns)

    # ── NumPy array path (Numba-ready) ───────────────────────────

    def load_arrays(
        self,
        symbol: str,
        start_date: str = "",
        end_date: str = "",
        columns: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Load data as contiguous NumPy arrays — optimal for Numba kernels.

        Returns ``{col_name: np.ndarray}`` with float64 arrays for numeric
        columns.  Uses Polars zero-copy when available.

        Falls back through the cached pandas path when the direct Polars /
        binary route misses, so repeated symbol loads hit the cache.
        """
        try:
            result = self.dataset.load_arrays(symbol, start_date, end_date, columns)
            if result:
                return result
        except Exception:
            pass

        df = self.load_data(symbol, start_date, end_date)
        if df is None or df.empty:
            return {}
        cols = columns or [c for c in df.columns if c != "date"]
        return {
            c: np.ascontiguousarray(df[c].values, dtype=np.float64)
            for c in cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)
        }

    # ── Feature engineering ──────────────────────────────────────

    def load_features(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load data with pre-computed technical indicators."""
        return self.dataset.load_features(symbol, start_date, end_date, features)

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        indicators_config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Calculate technical indicators in-place on the DataFrame."""
        return self.dataset.indicators.calculate_all(data, indicators_config)

    # ── Persistence ──────────────────────────────────────────────

    def save_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Save data and invalidate caches for the symbol."""
        self.dataset.save(symbol, data)
        if self._cache is not None:
            self._cache.remove_keys_with_prefix(f"data|{symbol}|")
        else:
            stale = [k for k in self._local_cache if k[0] == symbol]
            for k in stale:
                del self._local_cache[k]

    # ── Price queries ────────────────────────────────────────────

    def get_latest_price(
        self,
        symbol: str,
        date: Optional[datetime] = None,
    ) -> Optional[float]:
        """Get the most recent close price on or before the given date."""
        if date is None:
            date = datetime.now()

        start_date = (date - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = date.strftime("%Y-%m-%d")

        data = self.load_data(symbol, start_date, end_date)
        if data is None or data.empty:
            return None

        date_ts = pd.Timestamp(date)
        data_dates = pd.to_datetime(data["date"])
        filtered = data[data_dates <= date_ts]
        if filtered.empty:
            return None

        return float(filtered.iloc[-1]["close"])
