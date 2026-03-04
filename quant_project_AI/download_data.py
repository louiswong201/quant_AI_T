#!/usr/bin/env python3
"""
Download historical OHLC data for production backtest.

Crypto Top 10 + S&P500 Top 20, across 1d / 4h / 1h timeframes.
Uses yfinance (installed) with ccxt fallback for crypto intraday.

Usage:
    python download_data.py              # download all
    python download_data.py --crypto     # crypto only
    python download_data.py --stock      # stocks only
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed.  pip install yfinance")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
#  Symbol Definitions
# ═══════════════════════════════════════════════════════════════

CRYPTO_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "AVAX": "AVAX-USD",
    "DOT": "DOT-USD",
    "MATIC": "MATIC-USD",
}

STOCK_SYMBOLS = {
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",
    "NVDA": "NVDA",
    "META": "META",
    "TSLA": "TSLA",
    "BRK-B": "BRK-B",
    "UNH": "UNH",
    "JNJ": "JNJ",
    "JPM": "JPM",
    "V": "V",
    "PG": "PG",
    "HD": "HD",
    "MA_stock": "MA",
    "XOM": "XOM",
    "ABBV": "ABBV",
    "MRK": "MRK",
    "CVX": "CVX",
    "KO": "KO",
}

# yfinance interval → (period_for_max_data, min_bars_needed)
TIMEFRAME_CONFIG = {
    "1d": {"period": "max", "min_bars": 500},
    "4h": {"period": "730d", "interval": "1h", "min_bars": 500},
    "1h": {"period": "730d", "min_bars": 500},
}


def download_yf(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Download OHLCV data from yfinance with retry."""
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            if attempt == 2:
                print(f"    WARN: {ticker} {interval} failed after 3 attempts: {e}")
            time.sleep(1)
    return pd.DataFrame()


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and clean data."""
    df = df.copy()
    df.index.name = "date"
    df = df.reset_index()

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("open",):
            col_map[c] = "open"
        elif cl in ("high",):
            col_map[c] = "high"
        elif cl in ("low",):
            col_map[c] = "low"
        elif cl in ("close",):
            col_map[c] = "close"
        elif cl in ("volume",):
            col_map[c] = "volume"
        elif cl in ("date", "datetime"):
            col_map[c] = "date"
    df = df.rename(columns=col_map)

    required = ["date", "open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    df = df[["date", "open", "high", "low", "close"] +
            (["volume"] if "volume" in df.columns else [])]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["open"] > 0) &
            (df["high"] > 0) & (df["low"] > 0)]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h bars to 4h bars."""
    if df_1h.empty:
        return pd.DataFrame()
    df = df_1h.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    ohlc = df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        **({} if "volume" not in df.columns else {"volume": "sum"}),
    }).dropna(subset=["open", "high", "low", "close"])
    ohlc = ohlc.reset_index()
    return ohlc


def download_symbol(sym: str, ticker: str, data_dir: str, is_crypto: bool):
    """Download all timeframes for a single symbol."""
    print(f"  {sym} ({ticker})", end="", flush=True)

    # Daily
    df_1d = download_yf(ticker, "1d", "max")
    df_1d = process_df(df_1d)
    if len(df_1d) >= 500:
        path = os.path.join(data_dir, f"{sym}.csv")
        df_1d.to_csv(path, index=False)
        print(f" | 1d:{len(df_1d):,}", end="", flush=True)
    else:
        print(f" | 1d:SKIP({len(df_1d)})", end="", flush=True)

    # 1h
    df_1h = download_yf(ticker, "1h", "730d")
    df_1h = process_df(df_1h)
    if len(df_1h) >= 500:
        os.makedirs(os.path.join(data_dir, "1h"), exist_ok=True)
        path = os.path.join(data_dir, "1h", f"{sym}_1h.csv")
        df_1h.to_csv(path, index=False)
        print(f" | 1h:{len(df_1h):,}", end="", flush=True)
    else:
        print(f" | 1h:SKIP({len(df_1h)})", end="", flush=True)

    # 4h (resample from 1h)
    df_4h = resample_to_4h(df_1h)
    if len(df_4h) >= 500:
        os.makedirs(os.path.join(data_dir, "4h"), exist_ok=True)
        path = os.path.join(data_dir, "4h", f"{sym}_4h.csv")
        df_4h.to_csv(path, index=False)
        print(f" | 4h:{len(df_4h):,}", end="", flush=True)
    else:
        print(f" | 4h:SKIP({len(df_4h)})", end="", flush=True)

    print()


def main():
    parser = argparse.ArgumentParser(description="Download backtest data")
    parser.add_argument("--crypto", action="store_true", help="Crypto only")
    parser.add_argument("--stock", action="store_true", help="Stocks only")
    args = parser.parse_args()

    do_crypto = args.crypto or (not args.crypto and not args.stock)
    do_stock = args.stock or (not args.crypto and not args.stock)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 70)
    print("  DATA DOWNLOAD — Production Backtest")
    print(f"  Output: {data_dir}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    t0 = time.time()

    if do_crypto:
        print(f"\n  Crypto ({len(CRYPTO_SYMBOLS)} symbols):")
        for sym, ticker in CRYPTO_SYMBOLS.items():
            download_symbol(sym, ticker, data_dir, is_crypto=True)

    if do_stock:
        print(f"\n  Stocks ({len(STOCK_SYMBOLS)} symbols):")
        for sym, ticker in STOCK_SYMBOLS.items():
            download_symbol(sym, ticker, data_dir, is_crypto=False)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")

    # Summary
    n_daily = len([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    n_1h = len(os.listdir(os.path.join(data_dir, "1h"))) if os.path.isdir(os.path.join(data_dir, "1h")) else 0
    n_4h = len(os.listdir(os.path.join(data_dir, "4h"))) if os.path.isdir(os.path.join(data_dir, "4h")) else 0
    print(f"  Files: {n_daily} daily, {n_4h} 4h, {n_1h} 1h")
    print("=" * 70)


if __name__ == "__main__":
    main()
