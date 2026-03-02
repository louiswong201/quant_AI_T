#!/usr/bin/env python3
"""
Download multi-timeframe OHLCV data (15m, 1h, 4h) for backtesting.

- Crypto: Binance REST API (variable history per interval)
- Stocks: yfinance (1h up to 730d) + resample for 15m/4h

Usage:
    python examples/download_multi_tf_data.py
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
DATA_DIR = PROJECT_ROOT / "data"

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
STOCK_SYMBOLS = ["SPY", "QQQ", "NVDA", "META", "MSFT", "AMZN"]
BINANCE_LIMIT = 1000

INTERVALS = {
    "15m": {"crypto_months": 6, "stock_method": "resample_5m"},
    "1h":  {"crypto_months": 12, "stock_method": "yfinance"},
    "4h":  {"crypto_months": 24, "stock_method": "resample_1h"},
    "1d":  {"crypto_months": 24, "stock_method": "yfinance_1d"},
}


def download_binance(symbol: str, interval: str, months: int) -> pd.DataFrame:
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=months * 30)).timestamp() * 1000)

    all_bars = []
    cursor = start_ms
    while cursor < end_ms:
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval={interval}&startTime={cursor}&limit={BINANCE_LIMIT}"
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
        time.sleep(0.12)

    df = pd.DataFrame(all_bars)
    if df.empty:
        return df
    df.drop_duplicates(subset="date", inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def download_yfinance_1d(symbol: str, days: int = 730) -> pd.DataFrame:
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{days}d", interval="1d")
    if df.empty:
        df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
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


def download_yfinance_1h(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="730d", interval="1h")
    if df.empty:
        df = yf.download(symbol, period="730d", interval="1h", progress=False)
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


def resample_ohlcv(src_path: Path, factor: int) -> pd.DataFrame:
    if not src_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(src_path)
    for col in df.columns:
        df.rename(columns={col: col.strip().lower()}, inplace=True)
    if "close" not in df.columns:
        return pd.DataFrame()
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["close"], inplace=True)

    n = len(df)
    trim = n - (n % factor)
    df = df.iloc[:trim].copy()

    groups = []
    for i in range(0, trim, factor):
        chunk = df.iloc[i:i + factor]
        row = {
            "date": chunk.iloc[0]["date"] if "date" in chunk.columns else i,
            "open": chunk["open"].iloc[0] if "open" in chunk.columns else chunk["close"].iloc[0],
            "high": chunk["high"].max() if "high" in chunk.columns else chunk["close"].max(),
            "low": chunk["low"].min() if "low" in chunk.columns else chunk["close"].min(),
            "close": chunk["close"].iloc[-1],
        }
        if "volume" in chunk.columns:
            row["volume"] = chunk["volume"].sum()
        groups.append(row)
    return pd.DataFrame(groups)


def main():
    print("=" * 60)
    print("  Multi-Timeframe Data Downloader (15m / 1h / 4h / 1d)")
    print("=" * 60)

    for interval, cfg in INTERVALS.items():
        out_dir = DATA_DIR / interval
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {interval.upper()} ---")

        # Crypto: Binance
        for sym in CRYPTO_SYMBOLS:
            short = sym.replace("USDT", "")
            out = out_dir / f"{short}_{interval}.csv"
            print(f"  [{short}] Binance {interval} ({cfg['crypto_months']}mo) ...", end=" ", flush=True)
            try:
                df = download_binance(sym, interval, cfg["crypto_months"])
                df.to_csv(out, index=False)
                print(f"{len(df):,} bars")
            except Exception as e:
                print(f"FAILED: {e}")

        # Stocks
        method = cfg["stock_method"]
        for sym in STOCK_SYMBOLS:
            out = out_dir / f"{sym}_{interval}.csv"

            if method == "yfinance":
                print(f"  [{sym}] yfinance {interval} ...", end=" ", flush=True)
                try:
                    df = download_yfinance_1h(sym)
                    df.to_csv(out, index=False)
                    print(f"{len(df):,} bars")
                except Exception as e:
                    print(f"FAILED: {e}")

            elif method == "resample_5m":
                src = DATA_DIR / "5m" / f"{sym}_5m.csv"
                print(f"  [{sym}] Resample 5m -> {interval} ...", end=" ", flush=True)
                factor = 3  # 5m * 3 = 15m
                df = resample_ohlcv(src, factor)
                if df.empty:
                    print("NO SOURCE")
                else:
                    df.to_csv(out, index=False)
                    print(f"{len(df):,} bars")

            elif method == "resample_1h":
                src = out_dir.parent / "1h" / f"{sym}_1h.csv"
                print(f"  [{sym}] Resample 1h -> {interval} ...", end=" ", flush=True)
                factor = 4  # 1h * 4 = 4h
                df = resample_ohlcv(src, factor)
                if df.empty:
                    print("NO SOURCE (need 1h data first)")
                else:
                    df.to_csv(out, index=False)
                    print(f"{len(df):,} bars")

            elif method == "yfinance_1d":
                print(f"  [{sym}] yfinance {interval} ...", end=" ", flush=True)
                try:
                    df = download_yfinance_1d(sym, days=730)
                    df.to_csv(out, index=False)
                    print(f"{len(df):,} bars")
                except Exception as e:
                    print(f"FAILED: {e}")

    print("\n" + "=" * 60)
    print("  Multi-timeframe download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
