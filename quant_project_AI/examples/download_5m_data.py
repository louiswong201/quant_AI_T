#!/usr/bin/env python3
"""
Download 5-minute OHLCV data for backtesting.

- Crypto: Binance REST API (3 months, ~26K bars/asset)
- Stocks: yfinance (60 days max for 5m data, ~4.9K bars/asset)

Usage:
    python examples/download_5m_data.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "5m"

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
STOCK_SYMBOLS = ["SPY", "QQQ", "NVDA", "META", "MSFT", "AMZN"]
MONTHS_BACK = 3
BINANCE_LIMIT = 1000


def download_binance_5m(symbol: str, months: int = MONTHS_BACK) -> pd.DataFrame:
    """Download 5m klines from Binance with batching."""
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=months * 30)).timestamp() * 1000)

    all_bars = []
    cursor = start_ms
    while cursor < end_ms:
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval=5m&startTime={cursor}&limit={BINANCE_LIMIT}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        for k in data:
            all_bars.append({
                "date": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        cursor = int(data[-1][0]) + 1
        if len(data) < BINANCE_LIMIT:
            break
        time.sleep(0.15)

    df = pd.DataFrame(all_bars)
    df.drop_duplicates(subset="date", inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def download_yfinance_5m(symbol: str) -> pd.DataFrame:
    """Download 5m bars from yfinance (max ~60 days)."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="60d", interval="5m")
    if df.empty:
        df = yf.download(symbol, period="60d", interval="5m", progress=False)
    if df.empty:
        return df

    df = df.reset_index()
    col_map = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ("datetime", "date", "index"):
            col_map[c] = "date"
        elif cl == "open":
            col_map[c] = "open"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl == "close":
            col_map[c] = "close"
        elif cl == "volume":
            col_map[c] = "volume"
    df.rename(columns=col_map, inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  5-Minute Data Downloader")
    print("=" * 60)

    for sym in CRYPTO_SYMBOLS:
        short = sym.replace("USDT", "")
        out = DATA_DIR / f"{short}_5m.csv"
        print(f"  [{short}] Downloading from Binance ...", end=" ", flush=True)
        try:
            df = download_binance_5m(sym)
            df.to_csv(out, index=False)
            print(f"{len(df):,} bars -> {out.name}")
        except Exception as e:
            print(f"FAILED: {e}")

    for sym in STOCK_SYMBOLS:
        out = DATA_DIR / f"{sym}_5m.csv"
        print(f"  [{sym}] Downloading from yfinance ...", end=" ", flush=True)
        try:
            df = download_yfinance_5m(sym)
            df.to_csv(out, index=False)
            print(f"{len(df):,} bars -> {out.name}")
        except Exception as e:
            print(f"FAILED: {e}")

    print("=" * 60)
    print(f"  Data saved to {DATA_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
