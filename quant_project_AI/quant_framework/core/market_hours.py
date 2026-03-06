from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .asset_types import AssetClass


_US_HOLIDAYS: set[date] = set()


def _build_us_holidays() -> set[date]:
    holidays: set[date] = set()
    for year in range(2020, 2031):
        holidays.add(date(year, 1, 1))
        holidays.add(date(year, 7, 4))
        holidays.add(date(year, 12, 25))
        holidays.add(_nth_weekday(year, 1, 0, 3))
        holidays.add(_nth_weekday(year, 2, 0, 3))
        holidays.add(_easter(year) - timedelta(days=2))
        holidays.add(_nth_weekday(year, 5, 0, -1))
        holidays.add(date(year, 6, 19))
        holidays.add(_nth_weekday(year, 9, 0, 1))
        holidays.add(_nth_weekday(year, 11, 3, 4))
    return holidays


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    if n > 0:
        d += timedelta(weeks=n - 1)
    else:
        d += timedelta(weeks=4 + n)
        while d.month == month:
            d += timedelta(days=1)
        d -= timedelta(days=7)
        while d.weekday() != weekday:
            d -= timedelta(days=1)
    return d


def _easter(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _ensure_holidays() -> None:
    global _US_HOLIDAYS
    if not _US_HOLIDAYS:
        _US_HOLIDAYS = _build_us_holidays()


class MarketCalendar:
    def __init__(self, tz: str = "America/New_York") -> None:
        self._tz = ZoneInfo(tz)

    def is_tradable(self, asset_class: AssetClass, dt: datetime) -> bool:
        if asset_class in (
            AssetClass.CRYPTO_SPOT,
            AssetClass.CRYPTO_PERP,
            AssetClass.CRYPTO_INVERSE,
        ):
            return True
        if asset_class in (AssetClass.US_EQUITY, AssetClass.US_EQUITY_MARGIN):
            return self._is_us_market_open(dt)
        return False

    def _is_us_market_open(self, dt: datetime) -> bool:
        _ensure_holidays()
        et = dt.astimezone(self._tz)
        if et.weekday() >= 5:
            return False
        if et.date() in _US_HOLIDAYS:
            return False
        t = et.time()
        return time(9, 30) <= t <= time(16, 0)

    def next_open(self, asset_class: AssetClass, dt: datetime) -> datetime:
        if asset_class in (
            AssetClass.CRYPTO_SPOT,
            AssetClass.CRYPTO_PERP,
            AssetClass.CRYPTO_INVERSE,
        ):
            return dt
        return self._next_us_open(dt)

    def _next_us_open(self, dt: datetime) -> datetime:
        _ensure_holidays()
        et = dt.astimezone(self._tz)
        d = et.date()
        while True:
            if d.weekday() < 5 and d not in _US_HOLIDAYS:
                open_dt = datetime.combine(d, time(9, 30), tzinfo=self._tz)
                close_dt = datetime.combine(d, time(16, 0), tzinfo=self._tz)
                if et < open_dt:
                    return open_dt
                if et < close_dt:
                    return et
            d += timedelta(days=1)
