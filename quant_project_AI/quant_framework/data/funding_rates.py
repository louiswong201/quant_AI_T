"""Binance futures funding rate loader with parquet cache."""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FAPI_BASE = "https://fapi.binance.com"


class FundingRateLoader:
    """Download and cache Binance futures funding rates."""

    def __init__(self, cache_dir: str = "data/funding") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        symbol: str,
        start: Union[pd.Timestamp, str],
        end: Union[pd.Timestamp, str],
    ) -> pd.DataFrame:
        """Download funding rate history. Returns DataFrame with columns timestamp, funding_rate."""
        start_ts = int(pd.Timestamp(start).value // 1_000_000)
        end_ts = int(pd.Timestamp(end).value // 1_000_000)
        sym = symbol.upper()
        if not sym.endswith("USDT"):
            sym = f"{sym}USDT"
        url = f"{FAPI_BASE}/fapi/v1/fundingRate"
        rows: list = []
        current = start_ts
        while current <= end_ts:
            params = {"symbol": sym, "startTime": current, "endTime": end_ts, "limit": 1000}
            try:
                qs = "&".join(f"{k}={v}" for k, v in params.items())
                req = urllib.request.Request(f"{url}?{qs}")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
            except Exception as e:
                logger.warning("Funding rate fetch failed: %s", e)
                break
            if not data:
                break
            for r in data:
                ft = int(r["fundingTime"])
                fr = float(r["fundingRate"])
                rows.append({"timestamp": pd.Timestamp(ft, unit="ms", tz="UTC"), "funding_rate": fr})
            current = int(data[-1]["fundingTime"]) + 1
            if len(data) < 1000:
                break
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def _cache_path(self, symbol: str) -> Path:
        sym = symbol.upper().replace("USDT", "")
        return self._cache_dir / f"{sym}_funding.parquet"

    def download_and_cache(
        self,
        symbol: str,
        start: Union[pd.Timestamp, str],
        end: Union[pd.Timestamp, str],
    ) -> pd.DataFrame:
        df = self.download(symbol, start, end)
        if not df.empty:
            df.to_parquet(self._cache_path(symbol), index=False)
        return df

    def load_cached(self, symbol: str) -> np.ndarray:
        """Load cached funding rates as numpy array (funding_rate values)."""
        p = self._cache_path(symbol)
        if not p.exists():
            return np.array([], dtype=np.float64)
        df = pd.read_parquet(p)
        if "funding_rate" not in df.columns:
            return np.array([], dtype=np.float64)
        return df["funding_rate"].values.astype(np.float64)

    @staticmethod
    def map_to_bars(
        funding_df: pd.DataFrame,
        bar_timestamps: np.ndarray,
        interval: Optional[str] = None,
    ) -> np.ndarray:
        """Map 8h funding rates to bar-level. Each bar gets the latest funding rate <= bar timestamp."""
        if funding_df.empty or bar_timestamps.size == 0:
            return np.full(bar_timestamps.shape[0], np.nan, dtype=np.float64)
        ts_col = "timestamp" if "timestamp" in funding_df.columns else funding_df.columns[0]
        fr_col = "funding_rate" if "funding_rate" in funding_df.columns else funding_df.columns[1]
        funding_ts = pd.to_datetime(funding_df[ts_col]).values.astype("datetime64[ns]")
        funding_vals = funding_df[fr_col].values.astype(np.float64)
        bar_ts = np.atleast_1d(bar_timestamps)
        if bar_ts.dtype.kind in ("O", "U", "M") or str(bar_ts.dtype).startswith("datetime"):
            bar_dt = pd.to_datetime(bar_ts).values.astype("datetime64[ns]")
        else:
            bar_dt = bar_ts.astype("datetime64[ns]")
        idx = np.searchsorted(funding_ts, bar_dt, side="right") - 1
        idx = np.clip(idx, 0, len(funding_vals) - 1)
        result = funding_vals[idx].copy()
        invalid = bar_dt < funding_ts[0]
        result[invalid] = np.nan
        return result.reshape(np.shape(bar_timestamps))
