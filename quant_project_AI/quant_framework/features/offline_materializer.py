"""Offline feature/data materialization utilities.

Uses ``DataManager`` for all I/O so that scan/backtest paths benefit from
the tiered storage hierarchy (mmap → Arrow → Parquet → CSV) and caching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..data.data_manager import DataManager

logger = logging.getLogger(__name__)


class OfflineMaterializer:
    """Canonical offline loader shared by scan/backtest paths.

    Delegates every ``load_*`` call to ``DataManager`` which itself uses
    ``Dataset`` → tiered storage.  This guarantees identical data regardless
    of whether the caller is a live runner, a scan script, or a notebook.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._data_dir = str(data_dir)
        self._dm = DataManager(
            data_dir=self._data_dir,
            use_parquet=True,
            fast_io=True,
        )

    def load_frame(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        min_bars: int = 0,
    ) -> pd.DataFrame:
        df = self._dm.load_data(symbol, "", "")
        if df is None or df.empty:
            df = self._fallback_csv(symbol, interval)
        if df is not None and not df.empty and len(df) >= min_bars:
            return df
        return pd.DataFrame()

    def _fallback_csv(self, symbol: str, interval: str) -> pd.DataFrame:
        """Last-resort CSV load when no tiered storage file is available."""
        path = Path(self._data_dir)
        if interval == "1d":
            candidates = [path / f"{symbol}.csv", path / "daily" / f"{symbol}.csv"]
        else:
            candidates = [path / interval / f"{symbol}_{interval}.csv"]
        for file_path in candidates:
            if file_path.exists():
                try:
                    return pd.read_csv(file_path, parse_dates=["date"])
                except Exception:
                    return pd.read_csv(file_path)
        return pd.DataFrame()

    def load_ohlcv_arrays(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        min_bars: int = 0,
    ) -> Dict[str, np.ndarray]:
        arrays = self._dm.load_arrays(symbol, "", "")
        if arrays and len(next(iter(arrays.values()))) >= min_bars:
            out: Dict[str, np.ndarray] = {}
            _MAP = {"close": "c", "open": "o", "high": "h", "low": "l", "volume": "v"}
            for src, dst in _MAP.items():
                if src in arrays:
                    out[dst] = np.ascontiguousarray(arrays[src], dtype=np.float64)
            if "date" in arrays:
                out["timestamps"] = arrays["date"]
            if all(k in out for k in ("c", "o", "h", "l")):
                return out

        df = self.load_frame(symbol, interval=interval, min_bars=min_bars)
        if df.empty:
            return {}
        required = ("close", "open", "high", "low")
        if any(col not in df.columns for col in required):
            return {}
        out = {
            "c": np.ascontiguousarray(df["close"].values, dtype=np.float64),
            "o": np.ascontiguousarray(df["open"].values, dtype=np.float64),
            "h": np.ascontiguousarray(df["high"].values, dtype=np.float64),
            "l": np.ascontiguousarray(df["low"].values, dtype=np.float64),
        }
        if "volume" in df.columns:
            out["v"] = np.ascontiguousarray(df["volume"].values, dtype=np.float64)
        if "date" in df.columns:
            out["timestamps"] = df["date"].values.astype("datetime64[s]").astype(np.float64)
        return out

    def load_ohlcv_frame_map(
        self,
        *,
        interval: str = "1d",
        min_bars: int = 0,
    ) -> Dict[str, pd.DataFrame]:
        root = Path(self._data_dir)
        if interval == "1d":
            candidates = list(root.glob("*.csv"))
            daily_dir = root / "daily"
            if daily_dir.is_dir():
                candidates += list(daily_dir.glob("*.csv"))
            normalized = {}
            for file_path in candidates:
                symbol = file_path.stem
                normalized.setdefault(symbol, file_path)
        else:
            tf_dir = root / interval
            normalized = {
                p.name.replace(f"_{interval}.csv", ""): p
                for p in tf_dir.glob(f"*_{interval}.csv")
            } if tf_dir.is_dir() else {}
        frames: Dict[str, pd.DataFrame] = {}
        for symbol in sorted(normalized):
            df = self.load_frame(symbol, interval=interval, min_bars=min_bars)
            if not df.empty:
                if "date" in df.columns:
                    df = df.set_index("date")
                frames[symbol] = df
        return frames

    def load_ohlcv_array_map(
        self,
        *,
        interval: str = "1d",
        min_bars: int = 0,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        frames = self.load_ohlcv_frame_map(interval=interval, min_bars=min_bars)
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for symbol, df in frames.items():
            if any(col not in df.columns for col in ("close", "open", "high", "low")):
                continue
            out[symbol] = {
                "c": np.ascontiguousarray(df["close"].values, dtype=np.float64),
                "o": np.ascontiguousarray(df["open"].values, dtype=np.float64),
                "h": np.ascontiguousarray(df["high"].values, dtype=np.float64),
                "l": np.ascontiguousarray(df["low"].values, dtype=np.float64),
            }
            if "volume" in df.columns:
                out[symbol]["v"] = np.ascontiguousarray(df["volume"].values, dtype=np.float64)
        return out
