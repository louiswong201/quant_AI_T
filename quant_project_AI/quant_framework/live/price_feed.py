"""Unified price feed: yfinance polling (stocks) + Binance WebSocket (crypto).

Provides a common async interface for live and paper trading.
Each feed maintains a rolling window of OHLCV bars for strategy lookback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

CRYPTO_SUFFIXES = ("USDT", "BUSD", "BTC", "ETH", "BNB", "USDC")


@dataclass
class BarEvent:
    """Normalised OHLCV bar emitted by any price feed."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str = ""


@dataclass
class TickEvent:
    """Lightweight real-time price update emitted between bar closes.

    ``running_high`` / ``running_low`` track the current bar's extremes
    so far, matching the kernel's intra-bar H/L stoploss check.
    """
    symbol: str
    timestamp: datetime
    price: float
    running_high: float
    running_low: float


class PriceFeed(Protocol):
    """Async price feed interface."""

    @property
    def symbol(self) -> str: ...

    async def start(self, lookback: int = 200) -> None: ...

    async def bars(self) -> AsyncIterator[BarEvent]: ...

    def get_window(self) -> pd.DataFrame: ...

    async def stop(self) -> None: ...


def _is_crypto(symbol: str) -> bool:
    s = symbol.upper()
    return any(s.endswith(sfx) for sfx in CRYPTO_SUFFIXES)


class _RollingWindow:
    """Fixed-size rolling OHLCV window backed by pre-allocated numpy arrays.

    Uses a circular write pointer and avoids any memory allocation in the
    hot path (append).  ``to_arrays`` returns contiguous float64 views
    suitable for direct kernel evaluation without ``np.ascontiguousarray``.
    ``to_dataframe`` is only used for dashboard display, never in the
    signal generation hot path.
    """

    _COLS = ("open", "high", "low", "close", "volume")

    def __init__(self, maxlen: int = 200):
        self._maxlen = maxlen
        self._buf = {c: np.empty(maxlen, dtype=np.float64) for c in self._COLS}
        self._dates: List[Any] = [None] * maxlen
        self._count = 0
        self._write = 0
        self._df_dirty = True
        self._df_cache: Optional[pd.DataFrame] = None
        self._arr_dirty = True
        self._arr_cache: Optional[Dict[str, np.ndarray]] = None

    def append(self, bar: BarEvent) -> None:
        w = self._write
        self._buf["open"][w] = bar.open
        self._buf["high"][w] = bar.high
        self._buf["low"][w] = bar.low
        self._buf["close"][w] = bar.close
        self._buf["volume"][w] = bar.volume
        self._dates[w] = bar.timestamp
        self._write = (w + 1) % self._maxlen
        if self._count < self._maxlen:
            self._count += 1
        self._df_dirty = True
        self._arr_dirty = True

    def _ordered_slice(self, arr: np.ndarray) -> np.ndarray:
        """Return a contiguous copy of the ring buffer in chronological order."""
        n = self._count
        if n == 0:
            return np.empty(0, dtype=np.float64)
        if n < self._maxlen:
            return arr[:n].copy()
        start = self._write
        return np.concatenate((arr[start:], arr[:start]))

    def to_dataframe(self) -> pd.DataFrame:
        if not self._df_dirty and self._df_cache is not None:
            return self._df_cache
        if self._count == 0:
            self._df_cache = pd.DataFrame(columns=["date"] + list(self._COLS))
            self._df_dirty = False
            return self._df_cache
        data: Dict[str, Any] = {}
        for c in self._COLS:
            data[c] = self._ordered_slice(self._buf[c])
        n = self._count
        if n < self._maxlen:
            data["date"] = self._dates[:n]
        else:
            s = self._write
            data["date"] = self._dates[s:] + self._dates[:s]
        self._df_cache = pd.DataFrame(data)
        self._df_dirty = False
        return self._df_cache

    def to_arrays(self) -> Dict[str, np.ndarray]:
        if not self._arr_dirty and self._arr_cache is not None:
            return self._arr_cache
        self._arr_cache = {c: self._ordered_slice(self._buf[c]) for c in self._COLS}
        self._arr_dirty = False
        return self._arr_cache

    def __len__(self) -> int:
        return self._count


class YFinanceFeed:
    """Stock price feed via yfinance polling."""

    TICK_POLL_SECONDS = 15.0

    def __init__(self, symbol: str, interval: str = "1m", poll_seconds: float = 60.0):
        self._symbol = symbol
        self._interval = interval
        self._poll_seconds = poll_seconds
        self._window = _RollingWindow(maxlen=500)
        self._running = False
        self._queue: asyncio.Queue[BarEvent] = asyncio.Queue()
        self._last_ts: Optional[datetime] = None
        self._tick_callbacks: List[Callable[[TickEvent], Any]] = []
        self._tick_task: Optional[asyncio.Task] = None

    @property
    def symbol(self) -> str:
        return self._symbol

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns from yfinance (e.g. ('Close','SPY') → 'Close')."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df

    async def start(self, lookback: int = 200) -> None:
        import yfinance as yf
        period = "5d" if self._interval == "1m" else "60d"
        try:
            ticker = yf.Ticker(self._symbol)
            df = ticker.history(period=period, interval=self._interval)
            if df is None or df.empty:
                df = yf.download(self._symbol, period=period, interval=self._interval,
                                 progress=False, auto_adjust=True)
                if df is not None and not df.empty:
                    df = self._flatten_columns(df)
            if df is not None and not df.empty:
                df = df.tail(lookback).reset_index()
                date_col = "Datetime" if "Datetime" in df.columns else "Date"
                for _, row in df.iterrows():
                    bar = BarEvent(
                        symbol=self._symbol,
                        timestamp=pd.Timestamp(row[date_col]).to_pydatetime(),
                        open=float(row["Open"]), high=float(row["High"]),
                        low=float(row["Low"]), close=float(row["Close"]),
                        volume=float(row.get("Volume", 0)),
                        interval=self._interval,
                    )
                    self._window.append(bar)
                    self._last_ts = bar.timestamp
                logger.info("YFinance %s: loaded %d historical bars", self._symbol, len(self._window))
        except Exception as e:
            logger.warning("YFinance %s history load failed: %s", self._symbol, e)
        self._running = True

    @staticmethod
    def _is_us_market_hours() -> bool:
        """Check if current time falls within US market hours (9:30-16:00 ET)."""
        from datetime import timezone as tz
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            return True
        now_et = datetime.now(ZoneInfo("US/Eastern"))
        if now_et.weekday() >= 5:
            return False
        t = now_et.time()
        from datetime import time as dtime
        return dtime(9, 30) <= t <= dtime(16, 0)

    async def bars(self) -> AsyncIterator[BarEvent]:
        import yfinance as yf
        while self._running:
            await asyncio.sleep(self._poll_seconds)
            if not self._running:
                break

            is_crypto = any(c in self._symbol.upper() for c in ("USD", "BTC", "ETH"))
            if not is_crypto and not self._is_us_market_hours():
                continue

            try:
                ticker = yf.Ticker(self._symbol)
                df = ticker.history(period="1d", interval=self._interval)
                if df is None or df.empty:
                    df = yf.download(self._symbol, period="1d", interval=self._interval,
                                     progress=False, auto_adjust=True)
                    if df is not None and not df.empty:
                        df = self._flatten_columns(df)
                if df is None or df.empty:
                    continue
                df = df.reset_index()
                date_col = "Datetime" if "Datetime" in df.columns else "Date"
                for _, row in df.iterrows():
                    ts = pd.Timestamp(row[date_col]).to_pydatetime()
                    if self._last_ts and ts <= self._last_ts:
                        continue
                    bar = BarEvent(
                        symbol=self._symbol, timestamp=ts,
                        open=float(row["Open"]), high=float(row["High"]),
                        low=float(row["Low"]), close=float(row["Close"]),
                        volume=float(row.get("Volume", 0)),
                        interval=self._interval,
                    )
                    self._window.append(bar)
                    self._last_ts = ts
                    yield bar
            except Exception as e:
                logger.warning("YFinance %s poll error: %s", self._symbol, e)

    async def start_tick_polling(self) -> None:
        """Poll for real-time price updates between bar intervals."""
        import yfinance as yf
        while self._running:
            await asyncio.sleep(self.TICK_POLL_SECONDS)
            if not self._running or not self._tick_callbacks:
                continue
            is_crypto = any(c in self._symbol.upper() for c in ("USD", "BTC", "ETH"))
            if not is_crypto and not self._is_us_market_hours():
                continue
            try:
                ticker = yf.Ticker(self._symbol)
                info = ticker.fast_info
                price = float(info.get("lastPrice", 0) or info.get("last_price", 0))
                if price <= 0:
                    continue
                tick = TickEvent(
                    symbol=self._symbol,
                    timestamp=datetime.now(timezone.utc),
                    price=price,
                    running_high=price,
                    running_low=price,
                )
                for cb in self._tick_callbacks:
                    try:
                        result = cb(tick)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        pass
            except Exception:
                pass

    def get_window(self) -> pd.DataFrame:
        return self._window.to_dataframe()

    def get_arrays(self) -> Dict[str, np.ndarray]:
        return self._window.to_arrays()

    async def stop(self) -> None:
        self._running = False
        if self._tick_task and not self._tick_task.done():
            self._tick_task.cancel()


class BinanceFeed:
    """Crypto price feed via Binance public WebSocket kline stream."""

    WS_BASE = "wss://stream.binance.com:9443/ws"

    def __init__(self, symbol: str, interval: str = "1m"):
        self._symbol = symbol.upper()
        self._interval = interval
        self._window = _RollingWindow(maxlen=500)
        self._running = False
        self._ws = None
        self._tick_callbacks: List[Callable[[TickEvent], Any]] = []

    @property
    def symbol(self) -> str:
        return self._symbol

    async def start(self, lookback: int = 200) -> None:
        import aiohttp
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={self._symbol}&interval={self._interval}&limit={lookback}")
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url) as resp:
                    data = await resp.json()
            for k in data:
                bar = BarEvent(
                    symbol=self._symbol,
                    timestamp=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                    open=float(k[1]), high=float(k[2]),
                    low=float(k[3]), close=float(k[4]),
                    volume=float(k[5]),
                    interval=self._interval,
                )
                self._window.append(bar)
            logger.info("Binance %s: loaded %d historical bars", self._symbol, len(self._window))
        except Exception as e:
            logger.warning("Binance %s history load failed: %s", self._symbol, e)
        self._running = True

    async def _dispatch_tick(self, tick: TickEvent) -> None:
        for cb in self._tick_callbacks:
            try:
                result = cb(tick)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.debug("Tick callback error for %s: %s", tick.symbol, e)

    async def bars(self) -> AsyncIterator[BarEvent]:
        import websockets
        stream = f"{self._symbol.lower()}@kline_{self._interval}"
        url = f"{self.WS_BASE}/{stream}"
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    self._ws = ws
                    logger.info("Binance WS connected: %s", stream)
                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        k = data.get("k", {})
                        if k.get("x", False):
                            bar = BarEvent(
                                symbol=self._symbol,
                                timestamp=datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
                                open=float(k["o"]), high=float(k["h"]),
                                low=float(k["l"]), close=float(k["c"]),
                                volume=float(k["v"]),
                                interval=self._interval,
                            )
                            self._window.append(bar)
                            yield bar
                        elif self._tick_callbacks:
                            tick = TickEvent(
                                symbol=self._symbol,
                                timestamp=datetime.now(timezone.utc),
                                price=float(k["c"]),
                                running_high=float(k["h"]),
                                running_low=float(k["l"]),
                            )
                            await self._dispatch_tick(tick)
            except Exception as e:
                if self._running:
                    backoff = min(2 ** getattr(self, '_reconnect_attempts', 0), 60)
                    self._reconnect_attempts = getattr(self, '_reconnect_attempts', 0) + 1
                    logger.warning(
                        "Binance WS %s error, reconnecting in %ds: %s",
                        self._symbol, backoff, e,
                    )
                    await asyncio.sleep(backoff)
                else:
                    break
            else:
                self._reconnect_attempts = 0

    def get_window(self) -> pd.DataFrame:
        return self._window.to_dataframe()

    def get_arrays(self) -> Dict[str, np.ndarray]:
        return self._window.to_arrays()

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None


_INTERVAL_RANK = {"1m": 0, "5m": 1, "15m": 2, "1h": 3, "4h": 4, "1d": 5, "1w": 6}


def _symbol_to_file_prefix(symbol: str) -> str:
    """BNBUSDT -> BNB, BTCUSDT -> BTC for local file naming."""
    s = symbol.upper()
    for sfx in CRYPTO_SUFFIXES:
        if s.endswith(sfx):
            return s[: -len(sfx)]
    return s


def _load_local_ohlcv(data_dir: Path, symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """Load OHLCV from data/{interval}/{short}_{interval}.csv if exists."""
    prefix = _symbol_to_file_prefix(symbol)
    path = data_dir / interval / f"{prefix}_{interval}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        cols = {str(c).strip().lower(): c for c in df.columns}
        if "date" not in cols and "datetime" in cols:
            df = df.rename(columns={cols["datetime"]: "date"})
        elif "date" not in cols:
            return None
        for c in ("open", "high", "low", "close", "volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return df
    except Exception as e:
        logger.debug("Load local %s failed: %s", path, e)
        return None


class BinanceCombinedFeed:
    """Multi-interval crypto feed using Binance combined-streams WebSocket.

    Subscribes to multiple kline intervals on a single WS connection:
        wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1h/btcusdt@kline_4h

    Each interval maintains its own RollingWindow.  Tick events are derived
    from the finest-granularity stream for real-time SL/TP monitoring.

    If data_dir is set and local files exist (from download_multi_tf_data.py),
    historical bars are pre-loaded from disk before API/WebSocket.
    """

    STREAM_BASE = "wss://stream.binance.com:9443/stream"

    def __init__(self, symbol: str, intervals: List[str], data_dir: Optional[Path] = None):
        self._symbol = symbol.upper()
        self._intervals = sorted(intervals, key=lambda x: _INTERVAL_RANK.get(x, 99))
        self._windows: Dict[str, _RollingWindow] = {
            iv: _RollingWindow(maxlen=500) for iv in self._intervals
        }
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._running = False
        self._ws = None
        self._tick_callbacks: List[Callable[[TickEvent], Any]] = []
        self._bar_callbacks: List[Callable[[BarEvent], Any]] = []
        self._reconnect_attempts = 0

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def intervals(self) -> List[str]:
        return list(self._intervals)

    async def start(self, lookback: int = 200) -> None:
        """Load historical bars: try local data/ first, then Binance REST."""
        import aiohttp
        api_intervals = []
        for iv in self._intervals:
            df = _load_local_ohlcv(self._data_dir, self._symbol, iv)
            if df is not None and not df.empty:
                n = min(len(df), lookback * 2)
                df = df.tail(n)
                for _, row in df.iterrows():
                    bar = BarEvent(
                        symbol=self._symbol,
                        timestamp=pd.Timestamp(row["date"]).to_pydatetime(),
                        open=float(row["open"]), high=float(row["high"]),
                        low=float(row["low"]), close=float(row["close"]),
                        volume=float(row.get("volume", 0)),
                        interval=iv,
                    )
                    self._windows[iv].append(bar)
                logger.info(
                    "BinanceCombined %s@%s: loaded %d bars from local data/",
                    self._symbol, iv, len(self._windows[iv]),
                )
            else:
                api_intervals.append(iv)

        if not api_intervals:
            self._running = True
            return

        async with aiohttp.ClientSession() as sess:
            for iv in api_intervals:
                url = (
                    f"https://api.binance.com/api/v3/klines"
                    f"?symbol={self._symbol}&interval={iv}&limit={lookback}"
                )
                try:
                    async with sess.get(url) as resp:
                        data = await resp.json()
                    for k in data:
                        bar = BarEvent(
                            symbol=self._symbol,
                            timestamp=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                            open=float(k[1]), high=float(k[2]),
                            low=float(k[3]), close=float(k[4]),
                            volume=float(k[5]),
                            interval=iv,
                        )
                        self._windows[iv].append(bar)
                    logger.info(
                        "BinanceCombined %s@%s: loaded %d historical bars (API)",
                        self._symbol, iv, len(self._windows[iv]),
                    )
                except Exception as e:
                    logger.warning("BinanceCombined %s@%s history failed: %s", self._symbol, iv, e)
        self._running = True

    async def _dispatch_tick(self, tick: TickEvent) -> None:
        for cb in self._tick_callbacks:
            try:
                result = cb(tick)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.debug("Tick callback error for %s: %s", tick.symbol, e)

    async def run(self) -> None:
        """Connect to combined-streams WS, route kline messages by interval."""
        import websockets

        streams = "/".join(
            f"{self._symbol.lower()}@kline_{iv}" for iv in self._intervals
        )
        url = f"{self.STREAM_BASE}?streams={streams}"
        finest = self._intervals[0]
        self._running = True

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    self._ws = ws
                    self._reconnect_attempts = 0
                    logger.info("BinanceCombined WS connected: %s [%s]",
                                self._symbol, ",".join(self._intervals))
                    async for msg in ws:
                        if not self._running:
                            break
                        payload = json.loads(msg)
                        stream_name = payload.get("stream", "")
                        data = payload.get("data", {})
                        k = data.get("k", {})
                        if not k:
                            continue

                        iv = k.get("i", "")
                        if iv not in self._windows:
                            continue

                        if k.get("x", False):
                            bar = BarEvent(
                                symbol=self._symbol,
                                timestamp=datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
                                open=float(k["o"]), high=float(k["h"]),
                                low=float(k["l"]), close=float(k["c"]),
                                volume=float(k["v"]),
                                interval=iv,
                            )
                            self._windows[iv].append(bar)
                            for cb in self._bar_callbacks:
                                try:
                                    result = cb(bar)
                                    if asyncio.iscoroutine(result):
                                        await result
                                except Exception as e:
                                    logger.error("Bar callback error %s@%s: %s",
                                                 self._symbol, iv, e)
                        elif iv == finest and self._tick_callbacks:
                            tick = TickEvent(
                                symbol=self._symbol,
                                timestamp=datetime.now(timezone.utc),
                                price=float(k["c"]),
                                running_high=float(k["h"]),
                                running_low=float(k["l"]),
                            )
                            await self._dispatch_tick(tick)
            except Exception as e:
                if self._running:
                    backoff = min(2 ** self._reconnect_attempts, 60)
                    self._reconnect_attempts += 1
                    logger.warning(
                        "BinanceCombined WS %s error, reconnecting in %ds: %s",
                        self._symbol, backoff, e,
                    )
                    await asyncio.sleep(backoff)
                else:
                    break

    def get_window(self, interval: Optional[str] = None) -> pd.DataFrame:
        if interval and interval in self._windows:
            return self._windows[interval].to_dataframe()
        first = self._intervals[0] if self._intervals else None
        return self._windows[first].to_dataframe() if first else pd.DataFrame()

    def get_arrays(self, interval: Optional[str] = None) -> Dict[str, np.ndarray]:
        if interval and interval in self._windows:
            return self._windows[interval].to_arrays()
        first = self._intervals[0] if self._intervals else None
        return self._windows[first].to_arrays() if first else {}

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None


class PriceFeedManager:
    """Manages multiple price feeds, auto-selects source by symbol type.

    Supports both single-interval feeds (backward compatible) and
    multi-interval feeds via ``add_symbol_multi_tf``.

    If data_dir is set, BinanceCombinedFeed will try loading historical bars
    from data/{interval}/{symbol}_{interval}.csv before falling back to API.
    """

    def __init__(
        self,
        interval: str = "1m",
        poll_seconds: float = 60.0,
        data_dir: Optional[Path] = None,
    ):
        self._interval = interval
        self._poll_seconds = poll_seconds
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._feeds: Dict[str, Any] = {}
        self._multi_tf_feeds: Dict[str, BinanceCombinedFeed] = {}
        self._on_bar_callbacks: List[Callable[[BarEvent], Any]] = []
        self._on_tick_callbacks: List[Callable[[TickEvent], Any]] = []
        self._latest_prices: Dict[str, float] = {}

    def add_symbol(self, symbol: str) -> None:
        if symbol in self._feeds or symbol in self._multi_tf_feeds:
            return
        if _is_crypto(symbol):
            self._feeds[symbol] = BinanceFeed(symbol, interval=self._interval)
        else:
            self._feeds[symbol] = YFinanceFeed(symbol, interval=self._interval,
                                                poll_seconds=self._poll_seconds)

    def add_symbol_multi_tf(self, symbol: str, intervals: List[str]) -> None:
        """Register a symbol with multiple timeframes (combined-streams feed)."""
        if symbol in self._multi_tf_feeds or symbol in self._feeds:
            return
        if _is_crypto(symbol):
            self._multi_tf_feeds[symbol] = BinanceCombinedFeed(
                symbol, intervals, data_dir=self._data_dir
            )
        else:
            finest = sorted(intervals, key=lambda x: _INTERVAL_RANK.get(x, 99))[0]
            self._feeds[symbol] = YFinanceFeed(
                symbol, interval=finest, poll_seconds=self._poll_seconds,
            )
            logger.warning(
                "YFinance multi-TF not fully supported; using finest interval %s for %s",
                finest, symbol,
            )

    def on_bar(self, callback: Callable[[BarEvent], Any]) -> None:
        self._on_bar_callbacks.append(callback)

    def on_tick(self, callback: Callable[[TickEvent], Any]) -> None:
        self._on_tick_callbacks.append(callback)

    async def start_all(self, lookback: int = 200) -> None:
        tasks = [f.start(lookback) for f in self._feeds.values()]
        tasks += [f.start(lookback) for f in self._multi_tf_feeds.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    def _update_price_from_tick(self, tick: TickEvent) -> None:
        self._latest_prices[tick.symbol] = tick.price

    async def run(self) -> None:
        all_tick_cbs = [self._update_price_from_tick] + list(self._on_tick_callbacks)

        # -- single-interval feeds (legacy) --
        for feed in self._feeds.values():
            feed._tick_callbacks = all_tick_cbs

        async def _run_feed(feed: Any) -> None:
            async for bar in feed.bars():
                self._latest_prices[bar.symbol] = bar.close
                for cb in self._on_bar_callbacks:
                    try:
                        result = cb(bar)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error("Bar callback error for %s: %s", bar.symbol, e)

        tasks = [asyncio.create_task(_run_feed(f)) for f in self._feeds.values()]

        for feed in self._feeds.values():
            if isinstance(feed, YFinanceFeed) and feed._tick_callbacks:
                feed._tick_task = asyncio.create_task(feed.start_tick_polling())
                tasks.append(feed._tick_task)

        # -- multi-interval feeds --
        async def _dispatch_multi_bar(bar: BarEvent) -> None:
            self._latest_prices[bar.symbol] = bar.close
            for cb in self._on_bar_callbacks:
                try:
                    result = cb(bar)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error("Bar callback error for %s@%s: %s",
                                 bar.symbol, bar.interval, e)

        for mfeed in self._multi_tf_feeds.values():
            mfeed._tick_callbacks = all_tick_cbs
            mfeed._bar_callbacks = [_dispatch_multi_bar]
            tasks.append(asyncio.create_task(mfeed.run()))

        await asyncio.gather(*tasks, return_exceptions=True)

    def get_window(self, symbol: str, interval: Optional[str] = None) -> pd.DataFrame:
        mfeed = self._multi_tf_feeds.get(symbol)
        if mfeed is not None:
            return mfeed.get_window(interval)
        feed = self._feeds.get(symbol)
        return feed.get_window() if feed else pd.DataFrame()

    def get_arrays(self, symbol: str, interval: Optional[str] = None) -> Dict[str, np.ndarray]:
        mfeed = self._multi_tf_feeds.get(symbol)
        if mfeed is not None:
            return mfeed.get_arrays(interval)
        feed = self._feeds.get(symbol)
        return feed.get_arrays() if feed else {}

    def get_latest_prices(self) -> Dict[str, float]:
        if self._latest_prices:
            return dict(self._latest_prices)
        prices = {}
        for sym, feed in self._feeds.items():
            df = feed.get_window()
            if not df.empty:
                prices[sym] = float(df.iloc[-1]["close"])
        for sym, mfeed in self._multi_tf_feeds.items():
            df = mfeed.get_window()
            if not df.empty:
                prices[sym] = float(df.iloc[-1]["close"])
        return prices

    @property
    def symbols(self) -> List[str]:
        return list(self._feeds.keys()) + list(self._multi_tf_feeds.keys())

    def is_multi_tf(self, symbol: str) -> bool:
        return symbol in self._multi_tf_feeds

    def get_intervals(self, symbol: str) -> List[str]:
        mfeed = self._multi_tf_feeds.get(symbol)
        return mfeed.intervals if mfeed else [self._interval]

    async def stop_all(self) -> None:
        tasks = [f.stop() for f in self._feeds.values()]
        tasks += [f.stop() for f in self._multi_tf_feeds.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
