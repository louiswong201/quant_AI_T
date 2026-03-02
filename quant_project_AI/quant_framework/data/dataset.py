"""Dataset — unified data access with tiered I/O backends.

数据集：统一数据访问层，支持多级存储后端。

Backend priority (fast_io=True):
  Binary mmap → Arrow IPC → Parquet → File adapter

When Polars is available, a zero-copy ``load_lazy`` method returns a
``pl.LazyFrame`` so callers can compose arbitrary filters and projections
without materializing until needed.  The standard ``load`` path still
returns a ``pd.DataFrame`` for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from .adapters.file_adapter import FileDataAdapter
from .indicators import VectorizedIndicators
from .storage.arrow_ipc_storage import ArrowIpcStorage
from .storage.binary_mmap_storage import BinaryMmapStorage
from .storage.parquet_storage import ParquetStorage

logger = logging.getLogger(__name__)

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False


class Dataset:
    """Unified data access with tiered storage backends.

    数据集：统一数据访问。fast_io=True 时读优先 binary → arrow → parquet；
    写时同步写 binary/arrow。
    """

    def __init__(
        self,
        data_dir: str = "data",
        use_parquet: bool = True,
        fast_io: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.use_parquet = use_parquet
        self.fast_io = bool(fast_io)
        self.adapter = FileDataAdapter(
            data_dir=data_dir,
            preferred_format="parquet" if use_parquet else "csv",
        )
        self.parquet_storage = ParquetStorage(data_dir=data_dir) if use_parquet else None
        self.arrow_storage = ArrowIpcStorage(data_dir=data_dir) if self.fast_io else None
        self.binary_storage = BinaryMmapStorage(data_dir=data_dir) if self.fast_io else None
        self.indicators = VectorizedIndicators()

    # ── Pandas path (backward-compatible) ────────────────────────

    def load(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data, trying fastest backend first.

        加载顺序: fast_io 时 binary → arrow → parquet → adapter。
        """
        data: Optional[pd.DataFrame] = None
        if self.fast_io and self.binary_storage:
            data = self.binary_storage.load(symbol, start_date, end_date, columns=fields)
        if (data is None or data.empty) and self.fast_io and self.arrow_storage:
            data = self.arrow_storage.load(symbol, start_date, end_date, columns=fields)
        if (data is None or data.empty) and self.use_parquet and self.parquet_storage:
            data = self.parquet_storage.load(symbol, start_date, end_date, columns=fields)
        if data is None or data.empty:
            data = self.adapter.load_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                self._backfill_storage(symbol, data)
        if data is None or data.empty:
            return pd.DataFrame()
        if fields:
            available = [f for f in fields if f in data.columns]
            if available:
                data = data[available]
        return data

    # ── Polars lazy path (zero-copy, predicate pushdown) ─────────

    def load_lazy(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> Any:
        """Return a Polars LazyFrame with predicate + projection pushdown.

        This avoids all intermediate pandas copies.  Callers should call
        ``.collect()`` (or ``.collect().to_pandas()`` if they need a
        DataFrame) when they are ready to materialise.

        Falls back to ``load()`` wrapped in ``pl.from_pandas`` when the
        Parquet file is unavailable (e.g. data not yet written).
        """
        if not _POLARS_AVAILABLE:
            raise ImportError("Polars is required for load_lazy — pip install polars")

        if self.use_parquet and self.parquet_storage:
            import os
            fpath = os.path.join(self.data_dir, f"{symbol}.parquet")
            if os.path.exists(fpath):
                lf = pl.scan_parquet(fpath)
                if start_date is not None:
                    lf = lf.filter(pl.col("date") >= pl.lit(pd.Timestamp(start_date)))
                if end_date is not None:
                    lf = lf.filter(pl.col("date") <= pl.lit(pd.Timestamp(end_date)))
                if columns is not None:
                    need = list(dict.fromkeys(["date"] + columns))
                    lf = lf.select([c for c in need if c in lf.collect_schema().names()])
                return lf.sort("date")

        df = self.load(symbol, start_date or "", end_date or "")
        return pl.from_pandas(df).lazy()

    def load_arrays(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
    ) -> dict[str, np.ndarray]:
        """Load data directly as contiguous NumPy arrays — zero-copy via Polars.

        Returns a dict mapping column names to np.ndarray.  Ideal for
        feeding directly into Numba-compiled strategy kernels.
        """
        if _POLARS_AVAILABLE and self.use_parquet and self.parquet_storage:
            import os
            fpath = os.path.join(self.data_dir, f"{symbol}.parquet")
            if os.path.exists(fpath):
                cols = columns or ["date", "open", "high", "low", "close", "volume"]
                lf = pl.scan_parquet(fpath)
                if start_date:
                    lf = lf.filter(pl.col("date") >= pl.lit(pd.Timestamp(start_date)))
                if end_date:
                    lf = lf.filter(pl.col("date") <= pl.lit(pd.Timestamp(end_date)))
                available = lf.collect_schema().names()
                lf = lf.select([c for c in cols if c in available]).sort("date")
                frame = lf.collect()
                return {
                    c: frame.get_column(c).to_numpy(zero_copy_only=False)
                    for c in frame.columns
                }

        df = self.load(symbol, start_date, end_date)
        if df.empty:
            return {}
        return {
            c: np.ascontiguousarray(df[c].values, dtype=np.float64)
            for c in df.columns if c != "date"
        }

    # ── Internal helpers ─────────────────────────────────────────

    def _backfill_storage(self, symbol: str, data: pd.DataFrame) -> None:
        """Write data to all available storage backends for future fast reads."""
        if self.use_parquet and self.parquet_storage:
            self.parquet_storage.save(symbol, data)
        if self.fast_io and self.arrow_storage:
            self.arrow_storage.save(symbol, data)
        if self.fast_io and self.binary_storage:
            self.binary_storage.save(symbol, data)
    
    # ── Feature engineering ─────────────────────────────────────

    def load_features(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load data with pre-computed technical indicators (加载含指标的特征数据)."""
        data = self.load(symbol, start_date, end_date)
        if data.empty:
            return data

        if features is None:
            features_config: dict = {
                "ma": [5, 10, 20],
                "rsi": {"period": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
            }
        else:
            features_config = self._parse_features(features)

        return self.indicators.calculate_all(data, features_config)

    @staticmethod
    def _parse_features(features: List[str]) -> dict:
        """Parse feature list into configuration dict (解析特征列表为配置字典)."""
        config: dict = {}
        for feat in features:
            if feat.startswith("ma"):
                period = int(feat[2:])
                config.setdefault("ma", []).append(period)
            elif feat == "rsi":
                config["rsi"] = {"period": 14}
            elif feat == "macd":
                config["macd"] = {"fast": 12, "slow": 26, "signal": 9}
        return config

    # ── Persistence ──────────────────────────────────────────────

    def save(self, symbol: str, data: pd.DataFrame) -> None:
        """Save data to all available backends (保存数据到所有可用后端)."""
        if self.use_parquet and self.parquet_storage:
            self.parquet_storage.save(symbol, data)
        else:
            self.adapter.save_data(symbol, data)
        if self.fast_io and self.arrow_storage:
            self.arrow_storage.save(symbol, data)
        if self.fast_io and self.binary_storage:
            self.binary_storage.save(symbol, data)

    def get_available_symbols(self) -> List[str]:
        """Return list of available symbols (获取可用的股票代码列表)."""
        return self.adapter.get_available_symbols()

