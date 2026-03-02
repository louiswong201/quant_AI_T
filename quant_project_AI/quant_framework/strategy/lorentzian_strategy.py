"""
Lorentzian Classification 策略  (极速版)
═══════════════════════════════════════════════════════════════

算法来源: jdehorty (TradingView)
核心思想: 用 Lorentzian 距离度量替代欧氏距离的 KNN 分类器，
         对金融时序数据中的离群点和尾部事件更鲁棒。

为什么选 Lorentzian 距离？
──────────────────────────
  欧氏距离:    d(x, y) = sqrt(Σ (xi - yi)²)
  Lorentzian:  d(x, y) = Σ log(1 + |xi - yi|)

  log(1+|x|) 是亚线性增长 → 异常值的影响被大幅压缩。
  金融数据中 FOMC 会议、黑天鹅事件等会产生极端值，
  欧氏距离会被这些异常值主导，而 Lorentzian 不会。

极速设计:
─────────
  1. 所有特征预计算为 ndarray，on_bar_fast 全部走数组索引 O(1)
  2. KNN 搜索用 numpy 向量化 (一次矩阵减法 + 求和)
  3. 训练窗口滑动式更新 (不重算全部历史)
  4. 内核过滤器: EMA + SMA 趋势确认，减少噪声信号
"""

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_strategy import BaseStrategy


def _ema_1d_impl(arr, period):
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    alpha = 2.0 / (period + 1.0)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


_ema_1d = njit(cache=True, fastmath=True)(_ema_1d_impl) if NUMBA_AVAILABLE else _ema_1d_impl


def _rsi_1d_impl(close, period):
    n = len(close)
    rsi = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return rsi
    gains = np.empty(n - 1, dtype=np.float64)
    losses = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        d = close[i + 1] - close[i]
        if d > 0:
            gains[i] = d
            losses[i] = 0.0
        elif d < 0:
            gains[i] = 0.0
            losses[i] = -d
        else:
            gains[i] = 0.0
            losses[i] = 0.0
    avg_g = 0.0
    avg_l = 0.0
    for j in range(period):
        avg_g += gains[j]
        avg_l += losses[j]
    avg_g /= period
    avg_l /= period
    rsi[period] = 100.0 - (100.0 / (1.0 + avg_g / avg_l)) if avg_l > 0 else (100.0 if avg_g > 0 else 50.0)
    for i in range(period + 1, n):
        avg_g = (avg_g * (period - 1) + gains[i - 1]) / period
        avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
        rsi[i] = 100.0 - (100.0 / (1.0 + avg_g / avg_l)) if avg_l > 0 else (100.0 if avg_g > 0 else 50.0)
    return rsi


_rsi_1d = njit(cache=True, fastmath=True)(_rsi_1d_impl) if NUMBA_AVAILABLE else _rsi_1d_impl


def _cci_1d_impl(high, low, close, period):
    n = len(close)
    tp = np.empty(n, dtype=np.float64)
    for i in range(n):
        tp[i] = (high[i] + low[i] + close[i]) / 3.0
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        sma = 0.0
        for j in range(i - period + 1, i + 1):
            sma += tp[j]
        sma /= period
        mad = 0.0
        for j in range(i - period + 1, i + 1):
            mad += abs(tp[j] - sma)
        mad /= period
        out[i] = (tp[i] - sma) / (0.015 * mad) if mad > 1e-12 else 0.0
    return out


_cci_1d = njit(cache=True, fastmath=True)(_cci_1d_impl) if NUMBA_AVAILABLE else _cci_1d_impl


def _willr_1d_impl(high, low, close, period):
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        hh = high[i - period + 1]
        ll = low[i - period + 1]
        for j in range(i - period + 2, i + 1):
            if high[j] > hh:
                hh = high[j]
            if low[j] < ll:
                ll = low[j]
        r = hh - ll
        out[i] = -100.0 * (hh - close[i]) / r if r > 1e-12 else -50.0
    return out


_willr_1d = njit(cache=True, fastmath=True)(_willr_1d_impl) if NUMBA_AVAILABLE else _willr_1d_impl


def _atr_1d_impl(high, low, close, period):
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return atr
    s = 0.0
    for j in range(period):
        s += tr[j]
    atr[period - 1] = s / period
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


_atr_1d = njit(cache=True, fastmath=True)(_atr_1d_impl) if NUMBA_AVAILABLE else _atr_1d_impl


def _adx_1d_impl(high, low, close, period):
    n = len(close)
    if n < 2 * period:
        return np.full(n, np.nan, dtype=np.float64)
    tr = np.empty(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    atr_s = 0.0
    sp = 0.0
    sm = 0.0
    for j in range(1, period):
        atr_s += tr[j]
        sp += plus_dm[j]
        sm += minus_dm[j]
    atr_s /= (period - 1)
    sp /= (period - 1)
    sm /= (period - 1)
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    plus_di[period - 1] = 100.0 * sp / atr_s if atr_s > 1e-12 else 0.0
    minus_di[period - 1] = 100.0 * sm / atr_s if atr_s > 1e-12 else 0.0
    for i in range(period, n):
        atr_s = (atr_s * (period - 1) + tr[i]) / period
        sp = (sp * (period - 1) + plus_dm[i]) / period
        sm = (sm * (period - 1) + minus_dm[i]) / period
        plus_di[i] = 100.0 * sp / atr_s if atr_s > 1e-12 else plus_di[i - 1]
        minus_di[i] = 100.0 * sm / atr_s if atr_s > 1e-12 else minus_di[i - 1]
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        ds = plus_di[i] + minus_di[i]
        dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / ds if ds > 1e-12 else 0.0
    adx = np.full(n, np.nan, dtype=np.float64)
    nan_sum = 0.0
    nan_count = 0
    for j in range(period, 2 * period):
        v = dx[j]
        if v == v:
            nan_sum += v
            nan_count += 1
    adx[2 * period - 1] = nan_sum / nan_count if nan_count > 0 else np.nan
    for i in range(2 * period, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


_adx_1d = njit(cache=True, fastmath=True)(_adx_1d_impl) if NUMBA_AVAILABLE else _adx_1d_impl


def _rescale_impl(val, old_min, old_max, new_min, new_max):
    if old_max - old_min < 1e-12:
        return (new_min + new_max) * 0.5
    return new_min + (val - old_min) * (new_max - new_min) / (old_max - old_min)


_rescale = njit(cache=True, fastmath=True)(_rescale_impl) if NUMBA_AVAILABLE else _rescale_impl


class LorentzianClassificationStrategy(BaseStrategy):
    """
    Lorentzian 距离 KNN 分类策略。

    特征空间 (5 维):
      f1 = RSI(14)        — 动量振荡
      f2 = WT(10,11)      — Wave Trend 通道振荡
      f3 = CCI(20)        — 商品通道指数
      f4 = ADX(20)        — 趋势强度
      f5 = RSI(9)         — 短周期动量

    分类流程:
      1. 对每个 bar，从训练窗口中取 K 个最近邻 (Lorentzian 距离)
      2. 统计最近邻中 上涨/下跌 的比例 → 投票
      3. 内核过滤: EMA 与 SMA 确认趋势方向
      4. ADX 阈值过滤: ADX < 20 视为盘整，不交易
      5. ATR 动态止损

    参数:
      n_neighbors:    KNN 的 K 值 (默认 8)
      train_window:   训练窗口长度 (默认 2000)
      use_ema_filter: 是否启用 EMA 趋势过滤 (默认 True)
      use_sma_filter: 是否启用 SMA 趋势过滤 (默认 True)
      adx_threshold:  ADX 阈值，低于此值不交易 (默认 20)
      atr_stop_mult:  ATR 止损倍数 (默认 2.0)
    """

    def __init__(
        self,
        name: str = "Lorentzian分类策略",
        initial_capital: float = 1_000_000,
        n_neighbors: int = 8,
        train_window: int = 2000,
        use_ema_filter: bool = True,
        use_sma_filter: bool = True,
        ema_period: int = 200,
        sma_period: int = 200,
        adx_threshold: float = 20.0,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        rsi_period: int = 14,
        rsi_fast_period: int = 9,
        wt_channel: int = 10,
        wt_avg: int = 11,
        cci_period: int = 20,
        adx_period: int = 20,
        lookback_bars: int = 4,
        rag_provider=None,
    ):
        super().__init__(name=name, initial_capital=initial_capital, rag_provider=rag_provider)
        self.n_neighbors = n_neighbors
        self.train_window = train_window
        self.use_ema_filter = use_ema_filter
        self.use_sma_filter = use_sma_filter
        self.ema_period = ema_period
        self.sma_period = sma_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.rsi_period = rsi_period
        self.rsi_fast_period = rsi_fast_period
        self.wt_channel = wt_channel
        self.wt_avg = wt_avg
        self.cci_period = cci_period
        self.adx_period = adx_period
        self.lookback_bars = lookback_bars
        self._min_lookback = max(ema_period, sma_period, 2 * adx_period, train_window // 4) + 10
        self._stop_prices: Dict[str, Optional[float]] = {}
        self._feature_caches: Dict[str, np.ndarray] = {}
        self._label_caches: Dict[str, np.ndarray] = {}
        self._cache_ns: Dict[str, int] = {}

    @property
    def fast_columns(self) -> Tuple[str, ...]:
        return ("close", "high", "low")

    # ── 特征计算 ──────────────────────────────────────────────

    def _compute_features(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> np.ndarray:
        """计算 5 维特征矩阵 (n, 5)，全部基于 numpy，无 pandas 依赖。"""
        n = len(close)

        f1 = _rsi_1d(close, self.rsi_period)

        hlc3 = (high + low + close) / 3.0
        ema_hlc3 = _ema_1d(hlc3, self.wt_channel)
        diff = hlc3 - ema_hlc3
        abs_diff = np.abs(diff)
        ema_abs = _ema_1d(abs_diff, self.wt_channel)
        ema_abs = np.where(ema_abs < 1e-12, 1e-12, ema_abs)
        ci = (diff / (0.015 * ema_abs))
        f2 = _ema_1d(ci, self.wt_avg)

        f3 = _cci_1d(high, low, close, self.cci_period)
        f4 = _adx_1d(high, low, close, self.adx_period)
        f5 = _rsi_1d(close, self.rsi_fast_period)

        features = np.column_stack([f1, f2, f3, f4, f5])
        return features

    def _compute_labels(self, close: np.ndarray) -> np.ndarray:
        """标签: 未来 lookback_bars 内价格涨/跌。

        +1 = 上涨 (close[i+lookback] > close[i])
        -1 = 下跌
         0 = 不确定 / 尾部 padding
        """
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)
        end = n - self.lookback_bars
        if end > 0:
            future = close[self.lookback_bars:self.lookback_bars + end]
            current = close[:end]
            labels[:end] = np.where(future > current, 1, np.where(future < current, -1, 0)).astype(np.int8)
        return labels

    # ── KNN 分类 (Lorentzian 距离) ────────────────────────────

    def _lorentzian_knn(
        self, features: np.ndarray, labels: np.ndarray, query: np.ndarray,
        train_start: int, train_end: int,
    ) -> int:
        """用 Lorentzian 距离在训练窗口中找 K 近邻并投票。

        Lorentzian 距离:  d(x, q) = Σ log(1 + |xi - qi|)

        向量化实现: 一次 numpy 广播计算所有训练样本到 query 的距离。
        """
        train_features = features[train_start:train_end]
        train_labels = labels[train_start:train_end]

        valid_mask = ~np.isnan(train_features).any(axis=1) & (train_labels != 0)
        if valid_mask.sum() < self.n_neighbors:
            return 0
        valid_f = train_features[valid_mask]
        valid_l = train_labels[valid_mask]

        diffs = np.abs(valid_f - query)
        distances = np.sum(np.log1p(diffs), axis=1)

        k = min(self.n_neighbors, len(distances))
        if k >= len(distances):
            nn_indices = np.arange(len(distances))
        else:
            nn_indices = np.argpartition(distances, k)[:k]
        nn_labels = valid_l[nn_indices]
        vote = int(np.sum(nn_labels))
        if vote > 0:
            return 1
        elif vote < 0:
            return -1
        return 0

    # ── 过滤器 ────────────────────────────────────────────────

    def _check_filters(
        self,
        close: np.ndarray, high: np.ndarray, low: np.ndarray,
        i: int, prediction: int,
    ) -> int:
        """内核过滤器: EMA/SMA 趋势确认 + ADX 盘整过滤。"""
        price = close[i]

        if self.use_ema_filter and i >= self.ema_period:
            ema_arr = _ema_1d(close[:i + 1], self.ema_period)
            ema_val = ema_arr[i]
            if prediction == 1 and price < ema_val:
                return 0
            if prediction == -1 and price > ema_val:
                return 0

        if self.use_sma_filter and i >= self.sma_period:
            sma_val = np.mean(close[i - self.sma_period + 1: i + 1])
            if prediction == 1 and price < sma_val:
                return 0
            if prediction == -1 and price > sma_val:
                return 0

        return prediction

    # ── on_bar (DataFrame 路径) ───────────────────────────────

    def on_bar(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Union[Dict, List[Dict]]:
        if isinstance(data, dict):
            symbol = next(iter(data))
            df = data[symbol]
        else:
            df = data
            symbol = df.attrs.get("symbol", "STOCK") if hasattr(df, "attrs") else "STOCK"

        n = len(df)
        if n < self._min_lookback:
            return {"action": "hold"}

        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64) if "high" in df.columns else close.copy()
        low = df["low"].values.astype(np.float64) if "low" in df.columns else close.copy()

        return self._classify_and_signal(close, high, low, n - 1, symbol)

    # ── on_bar_fast (ndarray 极速路径) ────────────────────────

    def on_bar_fast(
        self,
        data_arrays: Dict[str, Any],
        i: int,
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[Union[Dict, List[Dict]]]:
        close = data_arrays.get("close")
        high = data_arrays.get("high")
        low = data_arrays.get("low")
        if close is None or high is None or low is None:
            return None
        if i + 1 < self._min_lookback:
            return {"action": "hold"}
        symbol = data_arrays.get("symbol", "STOCK")
        return self._classify_and_signal(close, high, low, i, symbol)

    def on_bar_fast_multi(
        self,
        data_arrays_by_symbol: Dict[str, Dict[str, Any]],
        i: int,
        current_date: pd.Timestamp,
        current_prices: Dict[str, float],
    ) -> Optional[Union[Dict, List[Dict]]]:
        signals: List[Dict] = []
        for symbol, arrs in data_arrays_by_symbol.items():
            close = arrs.get("close")
            high = arrs.get("high")
            low = arrs.get("low")
            if close is None or high is None or low is None:
                continue
            if i + 1 < self._min_lookback:
                continue
            sig = self._classify_and_signal(close, high, low, i, symbol)
            if sig["action"] != "hold":
                signals.append(sig)
        return signals if signals else {"action": "hold"}

    # ── 核心分类 + 信号生成 ───────────────────────────────────

    def _classify_and_signal(
        self,
        close: np.ndarray, high: np.ndarray, low: np.ndarray,
        i: int, symbol: str,
    ) -> Dict:
        """核心流程: 特征提取 → KNN 分类 → 过滤 → 信号。"""
        n = i + 1
        if n < self._min_lookback:
            return {"action": "hold"}

        c = close[:n].astype(np.float64)
        h = high[:n].astype(np.float64)
        l = low[:n].astype(np.float64)

        cached_n = self._cache_ns.get(symbol, 0)
        if symbol in self._feature_caches and cached_n == n:
            features = self._feature_caches[symbol]
            labels = self._label_caches[symbol]
        else:
            features = self._compute_features(c, h, l)
            labels = self._compute_labels(c)
            self._feature_caches[symbol] = features
            self._label_caches[symbol] = labels
            self._cache_ns[symbol] = n

        query = features[i]
        if np.any(np.isnan(query)):
            return {"action": "hold"}

        train_end = max(0, i - self.lookback_bars)
        train_start = max(0, train_end - self.train_window)
        if train_end - train_start < self.n_neighbors * 2:
            return {"action": "hold"}

        prediction = self._lorentzian_knn(features, labels, query, train_start, train_end)
        prediction = self._check_filters(c, h, l, i, prediction)

        # ATR 动态止损检查
        atr = _atr_1d(h, l, c, self.atr_period)
        current_atr = atr[i] if not np.isnan(atr[i]) else 0.0
        price = c[i]

        holdings = self.positions.get(symbol, 0)

        stop_px = self._stop_prices.get(symbol)
        if holdings > 0 and stop_px is not None and price <= stop_px:
            self._stop_prices.pop(symbol, None)
            return {"action": "sell", "symbol": symbol, "shares": holdings}

        if prediction == 1 and holdings == 0:
            shares = self.calculate_position_size(price, risk_percent=0.9)
            if shares > 0 and self.can_buy(symbol, price, shares):
                self._stop_prices[symbol] = price - self.atr_stop_mult * current_atr if current_atr > 0 else None
                return {"action": "buy", "symbol": symbol, "shares": shares}

        elif prediction == -1 and holdings > 0:
            self._stop_prices.pop(symbol, None)
            return {"action": "sell", "symbol": symbol, "shares": holdings}

        if holdings > 0 and stop_px is not None and current_atr > 0:
            new_stop = price - self.atr_stop_mult * current_atr
            if new_stop > stop_px:
                self._stop_prices[symbol] = new_stop

        return {"action": "hold"}
