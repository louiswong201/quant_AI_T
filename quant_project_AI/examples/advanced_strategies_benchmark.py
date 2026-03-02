#!/usr/bin/env python3
"""
==========================================================================
  7 大顶级量化策略 · 极速 Numba 实现 · 全参数扫描 · 性能基准
==========================================================================

策略来源：
  1. Drift Regime Mean Reversion — arxiv 2511.12490 (OOS Sharpe 13)
  2. Risk-Adjusted Momentum (RAMOM) — SSRN 2457647
  3. Turtle Breakout — 经典海龟系统 + ATR 动态止损
  4. Bollinger Band Mean Reversion — 布林带均值回归
  5. Keltner Channel Breakout — 肯特纳通道突破
  6. Multi-Factor Composite — 动量+价值+波动率因子融合
  7. Volatility Regime Adaptive — 波动率状态自动切换 (趋势/震荡)

所有策略均为 Numba @njit 编译，single-pass O(n)。
"""

import numpy as np
import pandas as pd
import time, sys, os, warnings, itertools
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numba import njit

# =====================================================================
#  Numba helpers
# =====================================================================

@njit(cache=True)
def _rolling_max(arr, window):
    """滚动最高"""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        mx = arr[i]
        for j in range(1, window):
            v = arr[i - j]
            if v > mx:
                mx = v
        out[i] = mx
    return out

@njit(cache=True)
def _rolling_min(arr, window):
    """滚动最低"""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        mn = arr[i]
        for j in range(1, window):
            v = arr[i - j]
            if v < mn:
                mn = v
        out[i] = mn
    return out

@njit(cache=True)
def _rolling_mean(arr, window):
    n = len(arr)
    out = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += arr[i]
        if i >= window:
            s -= arr[i - window]
        if i >= window - 1:
            out[i] = s / window
    return out

@njit(cache=True)
def _rolling_std(arr, window):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        s = 0.0
        s2 = 0.0
        for j in range(window):
            v = arr[i - j]
            s += v
            s2 += v * v
        m = s / window
        out[i] = np.sqrt(s2 / window - m * m)
    return out

@njit(cache=True)
def _ema(arr, span):
    n = len(arr)
    out = np.empty(n)
    k = 2.0 / (span + 1.0)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out

@njit(cache=True)
def _atr(high, low, close, period):
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
    return _rolling_mean(tr, period)

@njit(cache=True)
def _rsi_wilder(close, period):
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    gain = np.empty(n)
    loss = np.empty(n)
    gain[0] = 0.0
    loss[0] = 0.0
    for i in range(1, n):
        d = close[i] - close[i - 1]
        gain[i] = d if d > 0 else 0.0
        loss[i] = -d if d < 0 else 0.0
    ag = 0.0
    al = 0.0
    for i in range(1, period + 1):
        ag += gain[i]
        al += loss[i]
    ag /= period
    al /= period
    if al == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(period + 1, n):
        ag = (ag * (period - 1) + gain[i]) / period
        al = (al * (period - 1) + loss[i]) / period
        if al == 0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + ag / al)
    return out

SLIP_BUY = 1.0 + 5.0 / 10000.0
SLIP_SELL = 1.0 - 5.0 / 10000.0
COMM = 0.0015

# =====================================================================
#  Strategy 1: Drift Regime Mean Reversion
#  来源: arxiv 2511.12490 — OOS Sharpe 13
#  核心: 当股票过去 N 日中涨幅日占比 > threshold (漂移状态) 时
#        做空期回归 (反转); 当跌幅日占比 > threshold 时做多.
# =====================================================================

@njit(cache=True)
def bt_drift_regime(close, lookback, drift_threshold, hold_period):
    """
    Drift Regime Mean Reversion.
    lookback: 观察窗口 (论文用 63)
    drift_threshold: 漂移阈值 (论文用 0.60)
    hold_period: 持有天数后强制平仓
    """
    n = len(close)
    position = 0  # +1 long, -1 short, 0 flat
    entry_price = 0.0
    hold_count = 0
    total_ret = 1.0
    for i in range(lookback, n):
        # 计算过去 lookback 日中上涨日占比
        up_days = 0
        for j in range(1, lookback + 1):
            if close[i - j + 1] > close[i - j]:
                up_days += 1
        up_ratio = up_days / lookback

        if position != 0:
            hold_count += 1
            if hold_count >= hold_period:
                # 平仓
                if position == 1:
                    exit_p = close[i] * SLIP_SELL
                    trade_ret = (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
                else:
                    # short: profit = entry - exit
                    exit_p = close[i] * SLIP_BUY
                    trade_ret = (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0

        if position == 0:
            if up_ratio >= drift_threshold:
                # 漂移过高 → 做空 (反转)
                entry_price = close[i] * SLIP_BUY
                position = -1
                hold_count = 0
            elif up_ratio <= (1.0 - drift_threshold):
                # 漂移过低 → 做多 (反转)
                entry_price = close[i] * SLIP_BUY
                position = 1
                hold_count = 0

    if position == 1:
        exit_p = close[n - 1] * SLIP_SELL
        total_ret *= (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
    elif position == -1:
        exit_p = close[n - 1] * SLIP_BUY
        total_ret *= (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))
    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 2: Risk-Adjusted Momentum (RAMOM)
#  来源: SSRN 2457647
#  核心: 动量信号 / 波动率 → 标准化后交易，波动率越高越谨慎
# =====================================================================

@njit(cache=True)
def bt_ramom(close, mom_period, vol_period, entry_z, exit_z):
    """
    Risk-Adjusted Momentum.
    mom_period: 动量回看窗口
    vol_period: 波动率窗口
    entry_z / exit_z: 标准化动量阈值
    """
    n = len(close)
    position = 0
    entry_price = 0.0
    total_ret = 1.0
    start = max(mom_period, vol_period)
    for i in range(start, n):
        # 动量 = (close[i] / close[i-mom_period]) - 1
        mom = (close[i] / close[i - mom_period]) - 1.0
        # 滚动波动率
        s = 0.0
        s2 = 0.0
        for j in range(vol_period):
            if i - j > 0:
                r = (close[i - j] / close[i - j - 1]) - 1.0
            else:
                r = 0.0
            s += r
            s2 += r * r
        m = s / vol_period
        vol = np.sqrt(s2 / vol_period - m * m)
        if vol < 1e-10:
            vol = 1e-10
        z = mom / vol  # 标准化动量

        if position == 0:
            if z > entry_z:
                entry_price = close[i] * SLIP_BUY
                position = 1
            elif z < -entry_z:
                entry_price = close[i] * SLIP_BUY
                position = -1
        elif position == 1:
            if z < exit_z:
                exit_p = close[i] * SLIP_SELL
                total_ret *= (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
                position = 0
        elif position == -1:
            if z > -exit_z:
                exit_p = close[i] * SLIP_BUY
                total_ret *= (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))
                position = 0

    if position == 1:
        total_ret *= (close[n-1] * SLIP_SELL * (1-COMM)) / (entry_price * (1+COMM))
    elif position == -1:
        total_ret *= (entry_price * (1-COMM)) / (close[n-1] * SLIP_BUY * (1+COMM))
    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 3: Turtle Breakout
#  来源: 经典海龟交易法 (Richard Dennis)
#  核心: N 日最高突破做多, N 日最低突破做空, ATR 止损
# =====================================================================

@njit(cache=True)
def bt_turtle(close, high, low, entry_period, exit_period, atr_period, atr_stop_mult):
    """
    Turtle Breakout.
    entry_period: 入场突破窗口
    exit_period:  出场突破窗口
    atr_period:   ATR 周期
    atr_stop_mult: ATR 止损倍数
    """
    n = len(close)
    position = 0
    entry_price = 0.0
    total_ret = 1.0
    atr_arr = _atr(high, low, close, atr_period)
    start = max(entry_period, exit_period, atr_period)

    for i in range(start, n):
        # 计算入场/出场通道
        entry_high = -1e18
        entry_low = 1e18
        for j in range(1, entry_period + 1):
            if high[i - j] > entry_high:
                entry_high = high[i - j]
            if low[i - j] < entry_low:
                entry_low = low[i - j]

        exit_low = 1e18
        exit_high = -1e18
        for j in range(1, exit_period + 1):
            if low[i - j] < exit_low:
                exit_low = low[i - j]
            if high[i - j] > exit_high:
                exit_high = high[i - j]

        cur_atr = atr_arr[i]
        if cur_atr != cur_atr:
            continue

        if position == 1:
            # ATR 止损 or 出场通道突破
            stop = entry_price - atr_stop_mult * cur_atr
            if close[i] < stop or close[i] < exit_low:
                exit_p = close[i] * SLIP_SELL
                total_ret *= (exit_p * (1 - COMM)) / (entry_price * (1 + COMM))
                position = 0
        elif position == -1:
            stop = entry_price + atr_stop_mult * cur_atr
            if close[i] > stop or close[i] > exit_high:
                exit_p = close[i] * SLIP_BUY
                total_ret *= (entry_price * (1 - COMM)) / (exit_p * (1 + COMM))
                position = 0

        if position == 0:
            if close[i] > entry_high:
                entry_price = close[i] * SLIP_BUY
                position = 1
            elif close[i] < entry_low:
                entry_price = close[i] * SLIP_BUY
                position = -1

    if position == 1:
        total_ret *= (close[n-1] * SLIP_SELL * (1-COMM)) / (entry_price * (1+COMM))
    elif position == -1:
        total_ret *= (entry_price * (1-COMM)) / (close[n-1] * SLIP_BUY * (1+COMM))
    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 4: Bollinger Band Mean Reversion
#  核心: close 跌破下轨做多, 回到中轨平仓; 涨破上轨做空, 回中轨平仓
# =====================================================================

@njit(cache=True)
def bt_bollinger(close, period, num_std):
    n = len(close)
    ma = _rolling_mean(close, period)
    std = _rolling_std(close, period)
    position = 0
    entry_price = 0.0
    total_ret = 1.0
    for i in range(period, n):
        m = ma[i]; sd = std[i]
        if m != m or sd != sd or sd < 1e-10:
            continue
        upper = m + num_std * sd
        lower = m - num_std * sd
        if position == 0:
            if close[i] < lower:
                entry_price = close[i] * SLIP_BUY
                position = 1
            elif close[i] > upper:
                entry_price = close[i] * SLIP_BUY
                position = -1
        elif position == 1:
            if close[i] >= m:
                exit_p = close[i] * SLIP_SELL
                total_ret *= (exit_p * (1-COMM)) / (entry_price * (1+COMM))
                position = 0
        elif position == -1:
            if close[i] <= m:
                exit_p = close[i] * SLIP_BUY
                total_ret *= (entry_price * (1-COMM)) / (exit_p * (1+COMM))
                position = 0
    if position == 1:
        total_ret *= (close[n-1]*SLIP_SELL*(1-COMM)) / (entry_price*(1+COMM))
    elif position == -1:
        total_ret *= (entry_price*(1-COMM)) / (close[n-1]*SLIP_BUY*(1+COMM))
    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 5: Keltner Channel Breakout
#  核心: EMA ± ATR * mult, 突破上轨做多, 回到 EMA 平仓
# =====================================================================

@njit(cache=True)
def bt_keltner(close, high, low, ema_period, atr_period, atr_mult):
    n = len(close)
    ema_arr = _ema(close, ema_period)
    atr_arr = _atr(high, low, close, atr_period)
    position = 0
    entry_price = 0.0
    total_ret = 1.0
    start = max(ema_period, atr_period)
    for i in range(start, n):
        e = ema_arr[i]; a = atr_arr[i]
        if e != e or a != a:
            continue
        upper = e + atr_mult * a
        lower = e - atr_mult * a
        if position == 0:
            if close[i] > upper:
                entry_price = close[i] * SLIP_BUY
                position = 1
            elif close[i] < lower:
                entry_price = close[i] * SLIP_BUY
                position = -1
        elif position == 1:
            if close[i] < e:
                exit_p = close[i] * SLIP_SELL
                total_ret *= (exit_p*(1-COMM)) / (entry_price*(1+COMM))
                position = 0
        elif position == -1:
            if close[i] > e:
                exit_p = close[i] * SLIP_BUY
                total_ret *= (entry_price*(1-COMM)) / (exit_p*(1+COMM))
                position = 0
    if position == 1:
        total_ret *= (close[n-1]*SLIP_SELL*(1-COMM)) / (entry_price*(1+COMM))
    elif position == -1:
        total_ret *= (entry_price*(1-COMM)) / (close[n-1]*SLIP_BUY*(1+COMM))
    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 6: Multi-Factor Composite (Momentum + Value + Volatility)
#  核心: 综合 RSI + 动量 + 波动率三因子打分,
#        得分高做多, 得分低做空
# =====================================================================

@njit(cache=True)
def bt_multifactor(close, rsi_period, mom_period, vol_period, long_thr, short_thr):
    """
    Multi-Factor Composite.
    综合 3 因子:
      - RSI 反转信号 (RSI 越低分越高 → 做多)
      - 动量信号 (正动量分高)
      - 低波动偏好 (波动率越低分越高)
    """
    n = len(close)
    rsi = _rsi_wilder(close, rsi_period)
    position = 0
    entry_price = 0.0
    total_ret = 1.0
    start = max(rsi_period + 1, mom_period, vol_period)
    for i in range(start, n):
        r = rsi[i]
        if r != r:
            continue
        # Factor 1: RSI 反转 (0-100, 低 RSI → 高分, 用 100-RSI 归一化到 0-1)
        rsi_score = (100.0 - r) / 100.0
        # Factor 2: 动量 (close / close[i-mom_period] - 1), 裁剪到 [-0.5, 0.5], 归一化到 0-1
        mom = (close[i] / close[i - mom_period]) - 1.0
        mom_clip = max(-0.5, min(0.5, mom))
        mom_score = (mom_clip + 0.5)  # [0, 1]
        # Factor 3: 低波动偏好
        s2 = 0.0
        for j in range(vol_period):
            if i - j > 0:
                ret = (close[i-j] / close[i-j-1]) - 1.0
            else:
                ret = 0.0
            s2 += ret * ret
        vol = np.sqrt(s2 / vol_period)
        # vol 越低分越高, 大约 vol 在 0-0.05 之间
        vol_score = max(0.0, 1.0 - vol * 20.0)

        composite = (rsi_score + mom_score + vol_score) / 3.0

        if position == 0:
            if composite > long_thr:
                entry_price = close[i] * SLIP_BUY
                position = 1
            elif composite < short_thr:
                entry_price = close[i] * SLIP_BUY
                position = -1
        elif position == 1:
            if composite < 0.5:
                exit_p = close[i] * SLIP_SELL
                total_ret *= (exit_p*(1-COMM)) / (entry_price*(1+COMM))
                position = 0
        elif position == -1:
            if composite > 0.5:
                exit_p = close[i] * SLIP_BUY
                total_ret *= (entry_price*(1-COMM)) / (exit_p*(1+COMM))
                position = 0
    if position == 1:
        total_ret *= (close[n-1]*SLIP_SELL*(1-COMM)) / (entry_price*(1+COMM))
    elif position == -1:
        total_ret *= (entry_price*(1-COMM)) / (close[n-1]*SLIP_BUY*(1+COMM))
    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 7: Volatility Regime Adaptive
#  核心: 用 ATR / close 判断市场状态:
#        高波动 → 均值回归 (RSI)
#        低波动 → 趋势跟踪 (MA 交叉)
# =====================================================================

@njit(cache=True)
def bt_vol_regime(close, high, low, atr_period, vol_threshold,
                  ma_short, ma_long, rsi_period, rsi_os, rsi_ob):
    """
    Volatility Regime Adaptive.
    atr_period: ATR 周期
    vol_threshold: ATR/close 分界线 (高于此用均值回归, 低于此用趋势)
    ma_short/ma_long: 趋势模式参数
    rsi_period/rsi_os/rsi_ob: 均值回归模式参数
    """
    n = len(close)
    atr_arr = _atr(high, low, close, atr_period)
    rsi_arr = _rsi_wilder(close, rsi_period)
    ma_s = _rolling_mean(close, ma_short)
    ma_l = _rolling_mean(close, ma_long)
    position = 0
    entry_price = 0.0
    total_ret = 1.0
    mode = 0  # 0=trend, 1=reversion
    start = max(atr_period, rsi_period + 1, ma_long)

    for i in range(start, n):
        a = atr_arr[i]
        if a != a or close[i] <= 0:
            continue
        norm_vol = a / close[i]
        is_high_vol = 1 if norm_vol > vol_threshold else 0

        if is_high_vol == 1:
            # 均值回归模式 (RSI)
            r = rsi_arr[i]
            if r != r:
                continue
            if position == 0:
                if r < rsi_os:
                    entry_price = close[i] * SLIP_BUY
                    position = 1
                    mode = 1
                elif r > rsi_ob:
                    entry_price = close[i] * SLIP_BUY
                    position = -1
                    mode = 1
            elif position == 1 and mode == 1:
                if r > 50.0:
                    exit_p = close[i] * SLIP_SELL
                    total_ret *= (exit_p*(1-COMM)) / (entry_price*(1+COMM))
                    position = 0
            elif position == -1 and mode == 1:
                if r < 50.0:
                    exit_p = close[i] * SLIP_BUY
                    total_ret *= (entry_price*(1-COMM)) / (exit_p*(1+COMM))
                    position = 0
        else:
            # 趋势模式 (MA 交叉)
            ms = ma_s[i]; ml = ma_l[i]
            ms_p = ma_s[i-1]; ml_p = ma_l[i-1]
            if ms != ms or ml != ml or ms_p != ms_p or ml_p != ml_p:
                continue
            if position == 0:
                if ms_p <= ml_p and ms > ml:
                    entry_price = close[i] * SLIP_BUY
                    position = 1
                    mode = 0
                elif ms_p >= ml_p and ms < ml:
                    entry_price = close[i] * SLIP_BUY
                    position = -1
                    mode = 0
            elif position == 1 and mode == 0:
                if ms < ml:
                    exit_p = close[i] * SLIP_SELL
                    total_ret *= (exit_p*(1-COMM)) / (entry_price*(1+COMM))
                    position = 0
            elif position == -1 and mode == 0:
                if ms > ml:
                    exit_p = close[i] * SLIP_BUY
                    total_ret *= (entry_price*(1-COMM)) / (exit_p*(1+COMM))
                    position = 0

        # 如果模式切换, 强制平仓
        if position != 0:
            if (mode == 0 and is_high_vol == 1) or (mode == 1 and is_high_vol == 0):
                if position == 1:
                    exit_p = close[i] * SLIP_SELL
                    total_ret *= (exit_p*(1-COMM)) / (entry_price*(1+COMM))
                else:
                    exit_p = close[i] * SLIP_BUY
                    total_ret *= (entry_price*(1-COMM)) / (exit_p*(1+COMM))
                position = 0

    if position == 1:
        total_ret *= (close[n-1]*SLIP_SELL*(1-COMM)) / (entry_price*(1+COMM))
    elif position == -1:
        total_ret *= (entry_price*(1-COMM)) / (close[n-1]*SLIP_BUY*(1+COMM))
    return (total_ret - 1.0) * 100.0


# =====================================================================
#  参数空间定义 & 扫描
# =====================================================================

def define_param_grid():
    """返回每个策略的参数网格"""
    grids = {}

    # 1. Drift Regime: lookback × drift_threshold × hold_period
    grids["DriftRegime"] = {
        "func": "bt_drift_regime",
        "params": list(itertools.product(
            range(20, 130, 10),         # lookback: 20,30,...,120
            [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70],  # drift_threshold
            range(3, 25, 3),            # hold_period: 3,6,...,24
        )),
        "labels": ("lookback", "drift_thr", "hold_days"),
    }

    # 2. RAMOM: mom_period × vol_period × entry_z × exit_z
    grids["RAMOM"] = {
        "func": "bt_ramom",
        "params": list(itertools.product(
            range(10, 130, 10),        # mom_period
            range(10, 70, 10),         # vol_period
            [1.0, 1.5, 2.0, 2.5, 3.0],  # entry_z
            [0.0, 0.3, 0.5, 0.8],     # exit_z
        )),
        "labels": ("mom_period", "vol_period", "entry_z", "exit_z"),
    }

    # 3. Turtle: entry × exit × atr_period × atr_stop_mult
    grids["Turtle"] = {
        "func": "bt_turtle",
        "params": list(itertools.product(
            range(10, 80, 5),          # entry_period
            range(5, 40, 5),           # exit_period
            [14, 20],                  # atr_period
            [1.5, 2.0, 2.5, 3.0],     # atr_stop_mult
        )),
        "labels": ("entry_p", "exit_p", "atr_p", "atr_stop"),
    }

    # 4. Bollinger: period × num_std
    grids["Bollinger"] = {
        "func": "bt_bollinger",
        "params": list(itertools.product(
            range(10, 110, 5),         # period
            [1.0, 1.5, 2.0, 2.5, 3.0],  # num_std
        )),
        "labels": ("period", "num_std"),
    }

    # 5. Keltner: ema_period × atr_period × atr_mult
    grids["Keltner"] = {
        "func": "bt_keltner",
        "params": list(itertools.product(
            range(10, 110, 5),         # ema_period
            [10, 14, 20],              # atr_period
            [1.0, 1.5, 2.0, 2.5, 3.0],  # atr_mult
        )),
        "labels": ("ema_p", "atr_p", "atr_mult"),
    }

    # 6. MultiFactor: rsi_period × mom_period × vol_period × long_thr × short_thr
    grids["MultiFactor"] = {
        "func": "bt_multifactor",
        "params": list(itertools.product(
            [7, 14, 21],               # rsi_period
            range(10, 70, 10),         # mom_period
            range(10, 50, 10),         # vol_period
            [0.55, 0.60, 0.65, 0.70],  # long_thr
            [0.25, 0.30, 0.35, 0.40],  # short_thr
        )),
        "labels": ("rsi_p", "mom_p", "vol_p", "long_thr", "short_thr"),
    }

    # 7. VolRegime: atr_period × vol_threshold × ma_short × ma_long × rsi_period × rsi_os × rsi_ob
    grids["VolRegime"] = {
        "func": "bt_vol_regime",
        "params": list(itertools.product(
            [14, 20],                  # atr_period
            [0.015, 0.020, 0.025, 0.030],  # vol_threshold
            [5, 10, 20],               # ma_short
            [20, 40, 60],              # ma_long
            [14],                      # rsi_period
            [25, 30, 35],              # rsi_os
            [65, 70, 75],              # rsi_ob
        )),
        "labels": ("atr_p", "vol_thr", "ma_s", "ma_l", "rsi_p", "rsi_os", "rsi_ob"),
    }

    total = sum(len(g["params"]) for g in grids.values())
    return grids, total


# =====================================================================
#  Main
# =====================================================================

def main():
    symbols = ["AAPL", "GOOGL", "TSLA"]
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    grids, total_per_sym = define_param_grid()
    total_all = total_per_sym * len(symbols)
    print(f"Strategies: {len(grids)}")
    for name, g in grids.items():
        print(f"  {name:15s}: {len(g['params']):>8,} combos")
    print(f"  {'TOTAL/symbol':15s}: {total_per_sym:>8,}")
    print(f"  {'TOTAL (3 sym)':15s}: {total_all:>8,}")

    # --- JIT warm-up ---
    print("\nWarming up Numba JIT (7 strategies)…")
    dc = np.random.rand(200).astype(np.float64) * 100 + 100
    dh = dc + np.random.rand(200) * 2
    dl = dc - np.random.rand(200) * 2
    bt_drift_regime(dc, 20, 0.6, 5)
    bt_ramom(dc, 10, 10, 2.0, 0.5)
    bt_turtle(dc, dh, dl, 10, 5, 14, 2.0)
    bt_bollinger(dc, 20, 2.0)
    bt_keltner(dc, dh, dl, 20, 14, 2.0)
    bt_multifactor(dc, 14, 20, 20, 0.6, 0.35)
    bt_vol_regime(dc, dh, dl, 14, 0.02, 5, 20, 14, 30, 70)
    print("JIT ready.\n")

    # Dispatch table
    func_map = {
        "bt_drift_regime": bt_drift_regime,
        "bt_ramom": bt_ramom,
        "bt_turtle": bt_turtle,
        "bt_bollinger": bt_bollinger,
        "bt_keltner": bt_keltner,
        "bt_multifactor": bt_multifactor,
        "bt_vol_regime": bt_vol_regime,
    }
    # Which strategies need high/low
    needs_hl = {"bt_turtle", "bt_keltner", "bt_vol_regime"}

    all_best = {}
    grand_t0 = time.time()

    for sym in symbols:
        print(f"\n{'=' * 65}")
        print(f"  {sym}")
        print(f"{'=' * 65}")

        df = pd.read_csv(os.path.join(data_dir, f"{sym}.csv"), parse_dates=["date"])
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        n = len(close)
        print(f"  Loaded {n} bars\n")

        sym_results = {}

        for strat_name, grid in grids.items():
            func = func_map[grid["func"]]
            params_list = grid["params"]
            labels = grid["labels"]
            use_hl = grid["func"] in needs_hl

            t0 = time.time()
            best_ret = -1e18
            best_params = None
            count = 0

            for p in params_list:
                if use_hl:
                    if strat_name == "Turtle":
                        ret = func(close, high, low, p[0], p[1], p[2], p[3])
                    elif strat_name == "Keltner":
                        ret = func(close, high, low, p[0], p[1], p[2])
                    elif strat_name == "VolRegime":
                        ret = func(close, high, low, p[0], p[1], p[2], p[3], p[4], p[5], p[6])
                else:
                    if len(p) == 3:
                        ret = func(close, p[0], p[1], p[2])
                    elif len(p) == 4:
                        ret = func(close, p[0], p[1], p[2], p[3])
                    elif len(p) == 5:
                        ret = func(close, p[0], p[1], p[2], p[3], p[4])
                    elif len(p) == 2:
                        ret = func(close, p[0], p[1])
                    else:
                        ret = func(close, *p)

                if ret > best_ret:
                    best_ret = ret
                    best_params = p
                count += 1

            elapsed = time.time() - t0
            per = elapsed / count * 1000 if count else 0
            param_str = ", ".join(f"{l}={v}" for l, v in zip(labels, best_params))
            print(f"  {strat_name:15s}: {count:>8,} combos in {elapsed:6.2f}s "
                  f"({per:.4f}ms/c)  BEST = {best_ret:+8.2f}%  [{param_str}]")

            sym_results[strat_name] = {
                "return": best_ret,
                "params": best_params,
                "labels": labels,
                "combos": count,
                "time": elapsed,
            }

        all_best[sym] = sym_results

    grand_elapsed = time.time() - grand_t0
    total_combos = sum(
        sum(v["combos"] for v in sr.values()) for sr in all_best.values()
    )

    # ==================== SUMMARY ====================
    print(f"\n\n{'=' * 65}")
    print(f"  GRAND SUMMARY")
    print(f"{'=' * 65}")
    print(f"Total combos:  {total_combos:,}")
    print(f"Total time:    {grand_elapsed:.1f}s ({grand_elapsed/60:.1f} min)")
    print(f"Throughput:    {total_combos/grand_elapsed:,.0f} combos/sec")

    # Best strategy per symbol
    print(f"\n--- Best Strategy per Symbol (3-year return) ---")
    print(f"{'Symbol':>6}  {'Strategy':>15}  {'Return':>10}  {'Params'}")
    print(f"{'------':>6}  {'--------':>15}  {'------':>10}  {'------'}")
    for sym, sr in all_best.items():
        sorted_strats = sorted(sr.items(), key=lambda x: x[1]["return"], reverse=True)
        for i, (sname, sdata) in enumerate(sorted_strats):
            param_str = ", ".join(f"{l}={v}" for l, v in zip(sdata["labels"], sdata["params"]))
            marker = " <<<" if i == 0 else ""
            print(f"{sym if i==0 else '':>6}  {sname:>15}  {sdata['return']:+8.2f}%  {param_str}{marker}")

    # Cross-symbol champion
    print(f"\n--- Overall Champion ---")
    best_overall = None
    for sym, sr in all_best.items():
        for sname, sdata in sr.items():
            if best_overall is None or sdata["return"] > best_overall[2]:
                best_overall = (sym, sname, sdata["return"], sdata["params"], sdata["labels"])
    if best_overall:
        sym, sname, ret, params, labels = best_overall
        param_str = ", ".join(f"{l}={v}" for l, v in zip(labels, params))
        print(f"  {sym} / {sname}: {ret:+.2f}% [{param_str}]")

    # Timing breakdown
    print(f"\n--- Timing by Strategy (3 symbols combined) ---")
    strat_times = defaultdict(lambda: {"time": 0.0, "combos": 0})
    for sym, sr in all_best.items():
        for sname, sdata in sr.items():
            strat_times[sname]["time"] += sdata["time"]
            strat_times[sname]["combos"] += sdata["combos"]
    for sname in sorted(strat_times, key=lambda x: strat_times[x]["time"], reverse=True):
        t = strat_times[sname]["time"]
        c = strat_times[sname]["combos"]
        print(f"  {sname:15s}: {t:6.2f}s  {c:>8,} combos  ({t/grand_elapsed*100:5.1f}%)")


if __name__ == "__main__":
    main()
