#!/usr/bin/env python3
"""
==========================================================================
  8 大前沿量化策略 · 极速 Numba 实现 · 全参数扫描
==========================================================================

策略来源：
  8.  Connors RSI-2 Mean Reversion — Larry Connors 经典短期均值回归
  9.  MESA Adaptive (Ehlers MAMA/FAMA) — Hilbert 变换自适应均线交叉
  10. Kaufman AMA (KAMA) Crossover — 效率比自适应均线
  11. Adaptive Donchian + ATR Trailing Stop — 自适应通道突破 + ATR 跟踪止损
  12. Dual Thrust (TB 经典) — 开盘突破动态阈值
  13. Mean Reversion Z-Score — 经典统计套利 z-score 均值回归
  14. Momentum Breakout (52-week high proximity) — 近高点动量突破
  15. Regime-Switching EMA — 波动率分段 + 快慢 EMA 切换

所有策略均为 Numba @njit 编译，single-pass O(n)，零 Python 对象开销。
"""

import numpy as np
import pandas as pd
import time
import sys
import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numba import njit

# =====================================================================
#  全局常量（保守交易成本，与之前基准一致）
# =====================================================================
SLIP_BUY = 1.0005
SLIP_SELL = 0.9995
COMM = 0.0015

# =====================================================================
#  Numba Helpers
# =====================================================================

@njit(cache=True)
def _ema(arr, span):
    """指数移动平均 - O(n) single pass"""
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    k = 2.0 / (span + 1.0)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


@njit(cache=True)
def _rolling_mean(arr, window):
    """滚动均值 - O(n) cumsum 法"""
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
    """滚动标准差 - O(n) running sum-of-squares"""
    n = len(arr)
    out = np.full(n, np.nan)
    s = 0.0
    s2 = 0.0
    for i in range(n):
        s += arr[i]
        s2 += arr[i] * arr[i]
        if i >= window:
            s -= arr[i - window]
            s2 -= arr[i - window] * arr[i - window]
        if i >= window - 1:
            mean = s / window
            var = s2 / window - mean * mean
            out[i] = np.sqrt(max(0.0, var))
    return out


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
def _atr(high, low, close, period):
    """ATR - O(n) Wilder smoothing"""
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
    out = np.full(n, np.nan)
    s = 0.0
    for i in range(period):
        s += tr[i]
    out[period - 1] = s / period
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


@njit(cache=True)
def _rsi_wilder(close, period):
    """Wilder RSI - O(n)"""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    gain_sum = 0.0
    loss_sum = 0.0
    for i in range(1, period + 1):
        d = close[i] - close[i - 1]
        if d > 0:
            gain_sum += d
        else:
            loss_sum -= d
    avg_gain = gain_sum / period
    avg_loss = loss_sum / period
    if avg_loss == 0.0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        d = close[i] - close[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


# =====================================================================
#  Strategy 8: Connors RSI-2 Mean Reversion
# =====================================================================

@njit(cache=True)
def bt_connors_rsi2(close, rsi_period, ma_trend, ma_exit, os_threshold, ob_threshold):
    """
    Larry Connors RSI-2 均值回归策略。
    
    逻辑：
    - 趋势过滤：价格 > MA(ma_trend) 才做多，< MA(ma_trend) 才做空
    - 入场：RSI(rsi_period) < os_threshold 做多 (超卖反弹)
    - 出场：价格穿越 MA(ma_exit) 
    
    参数：
      rsi_period:   RSI 周期 (经典=2)
      ma_trend:     趋势过滤均线 (经典=200)
      ma_exit:      出场均线 (经典=5)
      os_threshold: 超卖阈值 (经典=5 或 10)
      ob_threshold: 超买阈值 (经典=90 或 95)
    """
    n = len(close)
    position = 0  # +1 long, -1 short, 0 flat
    entry_price = 0.0
    total_ret = 1.0

    # 预计算 RSI
    rsi = np.full(n, np.nan)
    if n > rsi_period:
        gain_sum = 0.0
        loss_sum = 0.0
        for i in range(1, rsi_period + 1):
            d = close[i] - close[i - 1]
            if d > 0:
                gain_sum += d
            else:
                loss_sum -= d
        avg_gain = gain_sum / rsi_period
        avg_loss = loss_sum / rsi_period
        if avg_loss == 0.0:
            rsi[rsi_period] = 100.0
        else:
            rsi[rsi_period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        for i in range(rsi_period + 1, n):
            d = close[i] - close[i - 1]
            g = d if d > 0 else 0.0
            l = -d if d < 0 else 0.0
            avg_gain = (avg_gain * (rsi_period - 1) + g) / rsi_period
            avg_loss = (avg_loss * (rsi_period - 1) + l) / rsi_period
            if avg_loss == 0.0:
                rsi[i] = 100.0
            else:
                rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    # 预计算 MA (cumsum 法)
    ma_t = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += close[i]
        if i >= ma_trend:
            s -= close[i - ma_trend]
        if i >= ma_trend - 1:
            ma_t[i] = s / ma_trend

    ma_e = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += close[i]
        if i >= ma_exit:
            s -= close[i - ma_exit]
        if i >= ma_exit - 1:
            ma_e[i] = s / ma_exit

    start = max(rsi_period + 1, max(ma_trend, ma_exit))
    for i in range(start, n):
        r = rsi[i]
        if r != r:
            continue
        mt = ma_t[i]
        me = ma_e[i]
        if mt != mt or me != me:
            continue

        if position == 1:
            # 多头出场：价格穿越 ma_exit
            if close[i] > me:
                exit_p = close[i] * SLIP_SELL
                trade_ret = (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0
        elif position == -1:
            # 空头出场
            if close[i] < me:
                exit_p = close[i] * SLIP_BUY
                trade_ret = (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0

        if position == 0:
            if close[i] > mt and r < os_threshold:
                # 做多
                entry_price = close[i] * SLIP_BUY
                position = 1
            elif close[i] < mt and r > ob_threshold:
                # 做空
                entry_price = close[i] * SLIP_SELL
                position = -1

    # 平仓
    if position == 1:
        exit_p = close[n - 1] * SLIP_SELL
        total_ret *= (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
    elif position == -1:
        exit_p = close[n - 1] * SLIP_BUY
        total_ret *= (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))

    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 9: MESA Adaptive (Ehlers MAMA/FAMA)
# =====================================================================

@njit(cache=True)
def bt_mesa_adaptive(close, fast_limit, slow_limit):
    """
    Ehlers MAMA/FAMA 自适应均线策略。
    
    核心：Hilbert 变换测量主导周期 → 自适应 alpha → MAMA/FAMA 交叉信号
    
    参数：
      fast_limit:  快速极限 (典型 0.5)
      slow_limit:  慢速极限 (典型 0.05)
    """
    n = len(close)
    if n < 40:
        return 0.0

    # Hilbert Transform 变量
    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    I1 = np.zeros(n, dtype=np.float64)
    Q1 = np.zeros(n, dtype=np.float64)
    jI = np.zeros(n, dtype=np.float64)
    jQ = np.zeros(n, dtype=np.float64)
    I2 = np.zeros(n, dtype=np.float64)
    Q2 = np.zeros(n, dtype=np.float64)
    Re = np.zeros(n, dtype=np.float64)
    Im = np.zeros(n, dtype=np.float64)
    period_arr = np.zeros(n, dtype=np.float64)
    smooth_period = np.zeros(n, dtype=np.float64)
    phase = np.zeros(n, dtype=np.float64)
    mama = np.zeros(n, dtype=np.float64)
    fama = np.zeros(n, dtype=np.float64)

    # 初始化
    for i in range(min(6, n)):
        mama[i] = close[i]
        fama[i] = close[i]
        period_arr[i] = 6.0
        smooth_period[i] = 6.0

    for i in range(6, n):
        # Smooth price
        smooth[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0

        # Detrender
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[max(0,i-2)]
                        - 0.5769 * smooth[max(0,i-4)] - 0.0962 * smooth[max(0,i-6)])
        adj = 0.075 * period_arr[i-1] + 0.54
        detrender[i] *= adj

        # InPhase and Quadrature
        I1[i] = detrender[max(0,i-3)]
        Q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[max(0,i-2)]
                 - 0.5769 * detrender[max(0,i-4)] - 0.0962 * detrender[max(0,i-6)])
        Q1[i] *= adj

        # Advance phase
        jI[i] = (0.0962 * I1[i] + 0.5769 * I1[max(0,i-2)]
                 - 0.5769 * I1[max(0,i-4)] - 0.0962 * I1[max(0,i-6)])
        jI[i] *= adj
        jQ[i] = (0.0962 * Q1[i] + 0.5769 * Q1[max(0,i-2)]
                 - 0.5769 * Q1[max(0,i-4)] - 0.0962 * Q1[max(0,i-6)])
        jQ[i] *= adj

        I2[i] = I1[i] - jQ[i]
        Q2[i] = Q1[i] + jI[i]

        # Smooth
        I2[i] = 0.2 * I2[i] + 0.8 * I2[i-1]
        Q2[i] = 0.2 * Q2[i] + 0.8 * Q2[i-1]

        # Period measurement
        Re[i] = I2[i] * I2[i-1] + Q2[i] * Q2[i-1]
        Im[i] = I2[i] * Q2[i-1] - Q2[i] * I2[i-1]
        Re[i] = 0.2 * Re[i] + 0.8 * Re[i-1]
        Im[i] = 0.2 * Im[i] + 0.8 * Im[i-1]

        if Im[i] != 0.0 and Re[i] != 0.0:
            period_arr[i] = 2.0 * np.pi / np.arctan(Im[i] / Re[i])
        else:
            period_arr[i] = period_arr[i-1]

        if period_arr[i] > 1.5 * period_arr[i-1]:
            period_arr[i] = 1.5 * period_arr[i-1]
        if period_arr[i] < 0.67 * period_arr[i-1]:
            period_arr[i] = 0.67 * period_arr[i-1]
        if period_arr[i] < 6.0:
            period_arr[i] = 6.0
        if period_arr[i] > 50.0:
            period_arr[i] = 50.0

        period_arr[i] = 0.2 * period_arr[i] + 0.8 * period_arr[i-1]
        smooth_period[i] = 0.33 * period_arr[i] + 0.67 * smooth_period[i-1]

        # Phase
        if I1[i] != 0.0:
            phase[i] = np.arctan(Q1[i] / I1[i]) * 180.0 / np.pi
        else:
            phase[i] = phase[i-1]

        delta_phase = phase[i-1] - phase[i]
        if delta_phase < 1.0:
            delta_phase = 1.0

        # Adaptive alpha
        alpha = fast_limit / delta_phase
        if alpha < slow_limit:
            alpha = slow_limit
        if alpha > fast_limit:
            alpha = fast_limit

        mama[i] = alpha * close[i] + (1.0 - alpha) * mama[i-1]
        fama[i] = 0.5 * alpha * mama[i] + (1.0 - 0.5 * alpha) * fama[i-1]

    # Trading
    position = 0
    entry_price = 0.0
    total_ret = 1.0

    for i in range(7, n):
        if position == 0:
            if mama[i] > fama[i] and mama[i-1] <= fama[i-1]:
                entry_price = close[i] * SLIP_BUY
                position = 1
            elif mama[i] < fama[i] and mama[i-1] >= fama[i-1]:
                entry_price = close[i] * SLIP_SELL
                position = -1
        elif position == 1:
            if mama[i] < fama[i] and mama[i-1] >= fama[i-1]:
                exit_p = close[i] * SLIP_SELL
                trade_ret = (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
                total_ret *= trade_ret
                # 反手做空
                entry_price = close[i] * SLIP_SELL
                position = -1
        elif position == -1:
            if mama[i] > fama[i] and mama[i-1] <= fama[i-1]:
                exit_p = close[i] * SLIP_BUY
                trade_ret = (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))
                total_ret *= trade_ret
                # 反手做多
                entry_price = close[i] * SLIP_BUY
                position = 1

    if position == 1:
        exit_p = close[n - 1] * SLIP_SELL
        total_ret *= (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
    elif position == -1:
        exit_p = close[n - 1] * SLIP_BUY
        total_ret *= (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))

    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 10: Kaufman AMA (KAMA) Crossover
# =====================================================================

@njit(cache=True)
def bt_kama_crossover(close, er_period, fast_sc, slow_sc, atr_stop_mult, high_arr, low_arr, atr_period):
    """
    Kaufman KAMA 自适应均线 + ATR 止损策略。
    
    核心：效率比 (ER) 驱动自适应平滑 → KAMA 斜率信号 + ATR 动态止损
    
    参数：
      er_period:     效率比周期 (典型 10)
      fast_sc:       最快 EMA 常数的周期 (典型 2)
      slow_sc:       最慢 EMA 常数的周期 (典型 30)
      atr_stop_mult: ATR 止损倍数
      atr_period:    ATR 周期
    """
    n = len(close)
    if n < er_period + 2:
        return 0.0

    fast_c = 2.0 / (fast_sc + 1.0)
    slow_c = 2.0 / (slow_sc + 1.0)

    # 计算 KAMA
    kama = np.full(n, np.nan)
    kama[er_period - 1] = close[er_period - 1]
    for i in range(er_period, n):
        direction = abs(close[i] - close[i - er_period])
        volatility = 0.0
        for j in range(1, er_period + 1):
            volatility += abs(close[i - j + 1] - close[i - j])
        if volatility == 0.0:
            er = 0.0
        else:
            er = direction / volatility
        sc = (er * (fast_c - slow_c) + slow_c) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])

    # ATR
    atr_v = np.full(n, np.nan)
    tr_arr = np.empty(n, dtype=np.float64)
    tr_arr[0] = high_arr[0] - low_arr[0]
    for i in range(1, n):
        hl = high_arr[i] - low_arr[i]
        hc = abs(high_arr[i] - close[i - 1])
        lc = abs(low_arr[i] - close[i - 1])
        tr_arr[i] = max(hl, max(hc, lc))
    s = 0.0
    for i in range(atr_period):
        s += tr_arr[i]
    atr_v[atr_period - 1] = s / atr_period
    for i in range(atr_period, n):
        atr_v[i] = (atr_v[i - 1] * (atr_period - 1) + tr_arr[i]) / atr_period

    # Trading
    position = 0
    entry_price = 0.0
    total_ret = 1.0
    start = max(er_period + 2, atr_period)

    for i in range(start, n):
        k = kama[i]
        k_prev = kama[i - 1]
        a = atr_v[i]
        if k != k or k_prev != k_prev or a != a:
            continue

        if position == 1:
            # ATR 止损
            if close[i] < entry_price - atr_stop_mult * a:
                exit_p = close[i] * SLIP_SELL
                trade_ret = (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0
            elif k < k_prev:  # KAMA 掉头
                exit_p = close[i] * SLIP_SELL
                trade_ret = (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0
        elif position == -1:
            if close[i] > entry_price + atr_stop_mult * a:
                exit_p = close[i] * SLIP_BUY
                trade_ret = (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0
            elif k > k_prev:
                exit_p = close[i] * SLIP_BUY
                trade_ret = (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0

        if position == 0:
            if close[i] > k and k > k_prev:
                entry_price = close[i] * SLIP_BUY
                position = 1
            elif close[i] < k and k < k_prev:
                entry_price = close[i] * SLIP_SELL
                position = -1

    if position == 1:
        exit_p = close[n - 1] * SLIP_SELL
        total_ret *= (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
    elif position == -1:
        exit_p = close[n - 1] * SLIP_BUY
        total_ret *= (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))

    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 11: Adaptive Donchian + ATR Trailing Stop
# =====================================================================

@njit(cache=True)
def bt_donchian_atr(close, high_arr, low_arr, entry_period, atr_period, atr_mult):
    """
    自适应 Donchian 通道突破 + ATR 跟踪止损。
    
    入场：突破 entry_period 高点做多 / 突破低点做空
    出场：ATR 跟踪止损 (动态调整止损线)
    """
    n = len(close)
    if n < entry_period + atr_period:
        return 0.0

    # ATR
    atr_v = np.full(n, np.nan)
    tr_arr = np.empty(n, dtype=np.float64)
    tr_arr[0] = high_arr[0] - low_arr[0]
    for i in range(1, n):
        hl = high_arr[i] - low_arr[i]
        hc = abs(high_arr[i] - close[i - 1])
        lc = abs(low_arr[i] - close[i - 1])
        tr_arr[i] = max(hl, max(hc, lc))
    s = 0.0
    for i in range(atr_period):
        s += tr_arr[i]
    atr_v[atr_period - 1] = s / atr_period
    for i in range(atr_period, n):
        atr_v[i] = (atr_v[i - 1] * (atr_period - 1) + tr_arr[i]) / atr_period

    # Donchian high/low
    don_high = np.full(n, np.nan)
    don_low = np.full(n, np.nan)
    for i in range(entry_period - 1, n):
        mx = high_arr[i]
        mn = low_arr[i]
        for j in range(1, entry_period):
            if high_arr[i - j] > mx:
                mx = high_arr[i - j]
            if low_arr[i - j] < mn:
                mn = low_arr[i - j]
        don_high[i] = mx
        don_low[i] = mn

    position = 0
    entry_price = 0.0
    trailing_stop = 0.0
    total_ret = 1.0
    start = max(entry_period, atr_period)

    for i in range(start, n):
        dh = don_high[i - 1]  # 用前一天的通道
        dl = don_low[i - 1]
        a = atr_v[i]
        if dh != dh or dl != dl or a != a:
            continue

        if position == 1:
            # 更新跟踪止损 (只能上移)
            new_stop = close[i] - atr_mult * a
            if new_stop > trailing_stop:
                trailing_stop = new_stop
            if close[i] < trailing_stop:
                exit_p = close[i] * SLIP_SELL
                trade_ret = (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0
        elif position == -1:
            new_stop = close[i] + atr_mult * a
            if new_stop < trailing_stop:
                trailing_stop = new_stop
            if close[i] > trailing_stop:
                exit_p = close[i] * SLIP_BUY
                trade_ret = (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0

        if position == 0:
            if close[i] > dh:
                entry_price = close[i] * SLIP_BUY
                trailing_stop = close[i] - atr_mult * a
                position = 1
            elif close[i] < dl:
                entry_price = close[i] * SLIP_SELL
                trailing_stop = close[i] + atr_mult * a
                position = -1

    if position == 1:
        exit_p = close[n - 1] * SLIP_SELL
        total_ret *= (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
    elif position == -1:
        exit_p = close[n - 1] * SLIP_BUY
        total_ret *= (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))

    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 12: Dual Thrust (开盘突破策略)
# =====================================================================

@njit(cache=True)
def bt_dual_thrust(close, high_arr, low_arr, open_arr, lookback, k_up, k_down):
    """
    Dual Thrust 开盘区间突破策略 (TB 经典)。
    
    逻辑：
    - 计算前 N 日的 Range = max(HH-LC, HC-LL)
    - 上轨 = Open + k_up * Range
    - 下轨 = Open - k_down * Range
    - 突破上轨做多，突破下轨做空
    - 每日收盘平仓 (日内策略)
    
    参数：
      lookback: 计算范围的回看天数
      k_up:     上轨系数
      k_down:   下轨系数
    """
    n = len(close)
    if n < lookback + 2:
        return 0.0

    total_ret = 1.0

    for i in range(lookback, n):
        # 计算前 N 天的 HH, LL, HC, LC
        hh = high_arr[i - lookback]
        ll = low_arr[i - lookback]
        hc = close[i - lookback]
        lc = close[i - lookback]
        for j in range(i - lookback + 1, i):
            if high_arr[j] > hh:
                hh = high_arr[j]
            if low_arr[j] < ll:
                ll = low_arr[j]
            if close[j] > hc:
                hc = close[j]
            if close[j] < lc:
                lc = close[j]

        range1 = hh - lc
        range2 = hc - ll
        rng = range1 if range1 > range2 else range2

        open_p = open_arr[i] if i < len(open_arr) else close[i - 1]
        upper = open_p + k_up * rng
        lower = open_p - k_down * rng

        # 日内方向判断（用 close 近似）
        if close[i] > upper:
            # 做多
            ep = open_p * SLIP_BUY
            xp = close[i] * SLIP_SELL
            if ep > 0:
                trade_ret = (xp * (1.0 - COMM)) / (ep * (1.0 + COMM))
                total_ret *= trade_ret
        elif close[i] < lower:
            # 做空
            ep = open_p * SLIP_SELL
            xp = close[i] * SLIP_BUY
            if xp > 0:
                trade_ret = (ep * (1.0 - COMM)) / (xp * (1.0 + COMM))
                total_ret *= trade_ret

    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 13: Z-Score Mean Reversion
# =====================================================================

@njit(cache=True)
def bt_zscore_reversion(close, lookback, entry_z, exit_z, stop_z):
    """
    经典 Z-Score 均值回归策略。
    
    逻辑：
    - z = (price - MA) / StdDev
    - z < -entry_z → 做多 (价格偏低)
    - z > +entry_z → 做空 (价格偏高)
    - |z| < exit_z → 平仓 (回归均值)
    - |z| > stop_z → 止损
    
    参数：
      lookback: 均值/标准差计算窗口
      entry_z:  入场 z 阈值
      exit_z:   出场 z 阈值
      stop_z:   止损 z 阈值
    """
    n = len(close)
    if n < lookback + 2:
        return 0.0

    # 预计算 rolling mean 和 std
    r_mean = np.full(n, np.nan)
    r_std = np.full(n, np.nan)
    s = 0.0
    s2 = 0.0
    for i in range(n):
        s += close[i]
        s2 += close[i] * close[i]
        if i >= lookback:
            s -= close[i - lookback]
            s2 -= close[i - lookback] * close[i - lookback]
        if i >= lookback - 1:
            mean = s / lookback
            var = s2 / lookback - mean * mean
            r_mean[i] = mean
            r_std[i] = np.sqrt(max(0.0, var))

    position = 0
    entry_price = 0.0
    total_ret = 1.0

    for i in range(lookback, n):
        m = r_mean[i]
        sd = r_std[i]
        if sd == 0.0 or sd != sd:
            continue
        z = (close[i] - m) / sd

        if position == 1:
            if z > -exit_z or z > stop_z:
                exit_p = close[i] * SLIP_SELL
                trade_ret = (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0
        elif position == -1:
            if z < exit_z or z < -stop_z:
                exit_p = close[i] * SLIP_BUY
                trade_ret = (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0

        if position == 0:
            if z < -entry_z:
                entry_price = close[i] * SLIP_BUY
                position = 1
            elif z > entry_z:
                entry_price = close[i] * SLIP_SELL
                position = -1

    if position == 1:
        exit_p = close[n - 1] * SLIP_SELL
        total_ret *= (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
    elif position == -1:
        exit_p = close[n - 1] * SLIP_BUY
        total_ret *= (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))

    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 14: Momentum Breakout (52-week High Proximity)
# =====================================================================

@njit(cache=True)
def bt_momentum_breakout(close, high_arr, low_arr, high_period, proximity_pct, atr_period, atr_trail):
    """
    动量突破策略：接近 N 日新高时入场 + ATR 跟踪止损。
    
    逻辑：
    - 当价格 >= N日最高 * (1 - proximity_pct) → 做多 (动量确认)
    - ATR trailing stop 出场
    - 不做空（纯动量策略）
    
    参数：
      high_period:    回看最高价窗口 (典型 252 = 52 周)
      proximity_pct:  接近高点的比例阈值 (如 0.02 = 2%)
      atr_period:     ATR 计算周期
      atr_trail:      ATR 止损倍数
    """
    n = len(close)
    if n < max(high_period, atr_period) + 2:
        return 0.0

    # 滚动最高价
    roll_high = np.full(n, np.nan)
    for i in range(high_period - 1, n):
        mx = high_arr[i]
        for j in range(1, high_period):
            if high_arr[i - j] > mx:
                mx = high_arr[i - j]
        roll_high[i] = mx

    # ATR
    atr_v = np.full(n, np.nan)
    tr_arr = np.empty(n, dtype=np.float64)
    tr_arr[0] = high_arr[0] - low_arr[0]
    for i in range(1, n):
        hl = high_arr[i] - low_arr[i]
        hc = abs(high_arr[i] - close[i - 1])
        lc = abs(low_arr[i] - close[i - 1])
        tr_arr[i] = max(hl, max(hc, lc))
    s = 0.0
    for i in range(atr_period):
        s += tr_arr[i]
    atr_v[atr_period - 1] = s / atr_period
    for i in range(atr_period, n):
        atr_v[i] = (atr_v[i - 1] * (atr_period - 1) + tr_arr[i]) / atr_period

    position = 0
    entry_price = 0.0
    trailing_stop = 0.0
    total_ret = 1.0
    start = max(high_period, atr_period)

    for i in range(start, n):
        rh = roll_high[i]
        a = atr_v[i]
        if rh != rh or a != a:
            continue

        if position == 1:
            new_stop = close[i] - atr_trail * a
            if new_stop > trailing_stop:
                trailing_stop = new_stop
            if close[i] < trailing_stop:
                exit_p = close[i] * SLIP_SELL
                trade_ret = (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0

        if position == 0:
            threshold = rh * (1.0 - proximity_pct)
            if close[i] >= threshold:
                entry_price = close[i] * SLIP_BUY
                trailing_stop = close[i] - atr_trail * a
                position = 1

    if position == 1:
        exit_p = close[n - 1] * SLIP_SELL
        total_ret *= (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))

    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Strategy 15: Regime-Switching EMA
# =====================================================================

@njit(cache=True)
def bt_regime_switch_ema(close, high_arr, low_arr, atr_period, vol_threshold,
                         fast_ema_period, slow_ema_period, trend_ema_period):
    """
    波动率分段 + 快慢 EMA 自动切换策略。
    
    逻辑：
    - 计算 ATR / Close 的比率作为波动率代理
    - 高波动 → 趋势跟踪 (快/慢 EMA 交叉)
    - 低波动 → 均值回归 (趋势 EMA 偏离回归)
    
    参数：
      atr_period:      ATR 周期
      vol_threshold:   波动率阈值 (高于=趋势, 低于=回归)
      fast_ema_period: 趋势模式快 EMA
      slow_ema_period: 趋势模式慢 EMA
      trend_ema_period: 回归模式趋势 EMA
    """
    n = len(close)
    if n < max(atr_period, max(slow_ema_period, trend_ema_period)) + 2:
        return 0.0

    # ATR
    atr_v = np.full(n, np.nan)
    tr_arr = np.empty(n, dtype=np.float64)
    tr_arr[0] = high_arr[0] - low_arr[0]
    for i in range(1, n):
        hl = high_arr[i] - low_arr[i]
        hc = abs(high_arr[i] - close[i - 1])
        lc = abs(low_arr[i] - close[i - 1])
        tr_arr[i] = max(hl, max(hc, lc))
    s = 0.0
    for i in range(atr_period):
        s += tr_arr[i]
    atr_v[atr_period - 1] = s / atr_period
    for i in range(atr_period, n):
        atr_v[i] = (atr_v[i - 1] * (atr_period - 1) + tr_arr[i]) / atr_period

    # EMA
    fast_k = 2.0 / (fast_ema_period + 1.0)
    slow_k = 2.0 / (slow_ema_period + 1.0)
    trend_k = 2.0 / (trend_ema_period + 1.0)

    ema_fast = np.empty(n, dtype=np.float64)
    ema_slow = np.empty(n, dtype=np.float64)
    ema_trend = np.empty(n, dtype=np.float64)
    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    ema_trend[0] = close[0]
    for i in range(1, n):
        ema_fast[i] = close[i] * fast_k + ema_fast[i - 1] * (1.0 - fast_k)
        ema_slow[i] = close[i] * slow_k + ema_slow[i - 1] * (1.0 - slow_k)
        ema_trend[i] = close[i] * trend_k + ema_trend[i - 1] * (1.0 - trend_k)

    position = 0
    entry_price = 0.0
    total_ret = 1.0
    start = max(atr_period, max(slow_ema_period, trend_ema_period))

    for i in range(start, n):
        a = atr_v[i]
        if a != a or close[i] <= 0:
            continue
        vol_ratio = a / close[i]
        is_high_vol = vol_ratio > vol_threshold

        if is_high_vol:
            # 趋势模式：快慢 EMA 交叉
            if position == 0:
                if ema_fast[i] > ema_slow[i] and ema_fast[i - 1] <= ema_slow[i - 1]:
                    entry_price = close[i] * SLIP_BUY
                    position = 1
                elif ema_fast[i] < ema_slow[i] and ema_fast[i - 1] >= ema_slow[i - 1]:
                    entry_price = close[i] * SLIP_SELL
                    position = -1
            elif position == 1:
                if ema_fast[i] < ema_slow[i]:
                    exit_p = close[i] * SLIP_SELL
                    trade_ret = (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
                    total_ret *= trade_ret
                    position = 0
            elif position == -1:
                if ema_fast[i] > ema_slow[i]:
                    exit_p = close[i] * SLIP_BUY
                    trade_ret = (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))
                    total_ret *= trade_ret
                    position = 0
        else:
            # 均值回归模式：价格偏离趋势 EMA
            pct_dev = (close[i] - ema_trend[i]) / ema_trend[i]
            if position == 0:
                if pct_dev < -0.02:  # 价格低于趋势 2%
                    entry_price = close[i] * SLIP_BUY
                    position = 1
                elif pct_dev > 0.02:
                    entry_price = close[i] * SLIP_SELL
                    position = -1
            elif position == 1:
                if pct_dev > 0.0:  # 回归到均线
                    exit_p = close[i] * SLIP_SELL
                    trade_ret = (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
                    total_ret *= trade_ret
                    position = 0
            elif position == -1:
                if pct_dev < 0.0:
                    exit_p = close[i] * SLIP_BUY
                    trade_ret = (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))
                    total_ret *= trade_ret
                    position = 0

    if position == 1:
        exit_p = close[n - 1] * SLIP_SELL
        total_ret *= (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
    elif position == -1:
        exit_p = close[n - 1] * SLIP_BUY
        total_ret *= (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))

    return (total_ret - 1.0) * 100.0


# =====================================================================
#  Main: JIT warm-up + 终极 18 策略对决
# =====================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("  18 大策略终极对决 · 极速 Numba · 密集参数扫描")
    print("=" * 75)
    print()

    # JIT warm-up
    print("[1/3] JIT 预热 ...", end=" ", flush=True)
    t0 = time.time()
    dc = np.random.rand(200).astype(np.float64) * 100 + 100
    dh = dc + np.random.rand(200) * 2
    dl = dc - np.random.rand(200) * 2
    do = dc + np.random.rand(200) * 0.5

    bt_connors_rsi2(dc, 2, 50, 5, 10.0, 90.0)
    bt_mesa_adaptive(dc, 0.5, 0.05)
    bt_kama_crossover(dc, 10, 2, 30, 2.0, dh, dl, 14)
    bt_donchian_atr(dc, dh, dl, 20, 14, 2.0)
    bt_dual_thrust(dc, dh, dl, do, 5, 0.5, 0.5)
    bt_zscore_reversion(dc, 20, 2.0, 0.5, 4.0)
    bt_momentum_breakout(dc, dh, dl, 50, 0.02, 14, 2.0)
    bt_regime_switch_ema(dc, dh, dl, 14, 0.02, 5, 20, 50)
    print(f"完成 ({time.time()-t0:.1f}s)")

    # Import the previous 10 strategies
    print("[2/3] 导入前 10 大策略 ...", end=" ", flush=True)
    sys.path.insert(0, "examples")
    try:
        from param_scan_benchmark import (
            _bt_ma, _bt_rsi, _bt_macd,
            precompute_all_ma, precompute_all_ema, precompute_all_rsi,
        )
        from advanced_strategies_benchmark import (
            bt_drift_regime, bt_ramom, bt_turtle, bt_bollinger,
            bt_keltner, bt_multifactor, bt_vol_regime,
        )
        # Warm up old strategies
        dm_arr = np.random.rand(200).astype(np.float64) * 100 + 100
        dr = np.random.rand(200).astype(np.float64) * 100
        _bt_ma(dc, dm_arr, dm_arr, 1.0005, 0.9995, 0.0015)
        _bt_rsi(dc, dr, 30.0, 70.0, 1.0005, 0.9995, 0.0015)
        _bt_macd(dc, dm_arr, dm_arr, 9, 1.0005, 0.9995, 0.0015)
        bt_drift_regime(dc, 20, 0.6, 5)
        bt_ramom(dc, 10, 10, 2.0, 0.5)
        bt_turtle(dc, dh, dl, 10, 5, 14, 2.0)
        bt_bollinger(dc, 20, 2.0)
        bt_keltner(dc, dh, dl, 20, 14, 2.0)
        bt_multifactor(dc, 14, 20, 20, 0.6, 0.35)
        bt_vol_regime(dc, dh, dl, 14, 0.02, 5, 20, 14, 30, 70)
        HAS_OLD = True
        print("完成")
    except ImportError as e:
        HAS_OLD = False
        print(f"跳过 ({e})")

    # Load data
    print("[3/3] 加载数据 + 参数扫描 ...\n")
    symbols = ["AAPL", "GOOGL", "TSLA"]
    grand_t0 = time.time()
    results_all = {}
    total_combos = 0

    for sym in symbols:
        df = pd.read_csv(f"data/{sym}.csv", parse_dates=["date"])
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        open_p = df["open"].values.astype(np.float64) if "open" in df.columns else close.copy()
        n = len(close)
        sym_res = {}

        # ---------- Old 10 strategies ----------
        if HAS_OLD:
            # MA Crossover
            t0 = time.time()
            mas = precompute_all_ma(close, 200)
            best_r, best_p, cnt = -1e18, "", 0
            for s in range(2, 200):
                for l in range(s + 1, 201):
                    r = _bt_ma(close, mas[s], mas[l], SLIP_BUY, SLIP_SELL, COMM)
                    if r > best_r:
                        best_r = r
                        best_p = f"short={s}, long={l}"
                    cnt += 1
            sym_res["MA Crossover"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
            total_combos += cnt

            # RSI
            t0 = time.time()
            rsi_all = precompute_all_rsi(close, 200)
            best_r, best_p, cnt = -1e18, "", 0
            for p in range(2, 201):
                for os_v in range(10, 45, 5):
                    for ob_v in range(55, 95, 5):
                        r = _bt_rsi(close, rsi_all[p], float(os_v), float(ob_v), SLIP_BUY, SLIP_SELL, COMM)
                        if r > best_r:
                            best_r = r
                            best_p = f"period={p}, os={os_v}, ob={ob_v}"
                        cnt += 1
            sym_res["RSI"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
            total_combos += cnt

            # MACD
            t0 = time.time()
            emas = precompute_all_ema(close, 200)
            best_r, best_p, cnt = -1e18, "", 0
            for f in range(2, 100, 2):
                ef = emas[f]
                for s in range(f + 2, 201, 2):
                    es = emas[s]
                    for sg in range(2, min(s, 101), 2):
                        r = _bt_macd(close, ef, es, sg, SLIP_BUY, SLIP_SELL, COMM)
                        if r > best_r:
                            best_r = r
                            best_p = f"fast={f}, slow={s}, sig={sg}"
                        cnt += 1
            sym_res["MACD"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
            total_combos += cnt

            # DriftRegime
            t0 = time.time()
            best_r, best_p, cnt = -1e18, "", 0
            for lb in range(10, 130, 5):
                for dt in [0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72]:
                    for hp in range(3, 30, 2):
                        r = bt_drift_regime(close, lb, dt, hp)
                        if r > best_r:
                            best_r = r
                            best_p = f"lb={lb}, thr={dt}, hold={hp}"
                        cnt += 1
            sym_res["DriftRegime"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
            total_combos += cnt

            # RAMOM
            t0 = time.time()
            best_r, best_p, cnt = -1e18, "", 0
            for mp in range(5, 120, 5):
                for vp in range(5, 60, 5):
                    for ez in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                        for xz in [0.0, 0.2, 0.5, 0.8, 1.0]:
                            r = bt_ramom(close, mp, vp, ez, xz)
                            if r > best_r:
                                best_r = r
                                best_p = f"mom={mp}, vol={vp}, ez={ez}, xz={xz}"
                            cnt += 1
            sym_res["RAMOM"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
            total_combos += cnt

            # Turtle
            t0 = time.time()
            best_r, best_p, cnt = -1e18, "", 0
            for ep in range(5, 80, 3):
                for xp in range(3, 50, 3):
                    for ap in [10, 14, 20]:
                        for am in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                            r = bt_turtle(close, high, low, ep, xp, ap, am)
                            if r > best_r:
                                best_r = r
                                best_p = f"entry={ep}, exit={xp}, atr_p={ap}, stop={am}"
                            cnt += 1
            sym_res["Turtle"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
            total_combos += cnt

            # Bollinger
            t0 = time.time()
            best_r, best_p, cnt = -1e18, "", 0
            for p in range(5, 150, 3):
                for ns in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                    r = bt_bollinger(close, p, ns)
                    if r > best_r:
                        best_r = r
                        best_p = f"period={p}, std={ns}"
                    cnt += 1
            sym_res["Bollinger"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
            total_combos += cnt

            # Keltner
            t0 = time.time()
            best_r, best_p, cnt = -1e18, "", 0
            for ep in range(5, 120, 5):
                for ap in [7, 10, 14, 20, 30]:
                    for am in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                        r = bt_keltner(close, high, low, ep, ap, am)
                        if r > best_r:
                            best_r = r
                            best_p = f"ema={ep}, atr_p={ap}, mult={am}"
                        cnt += 1
            sym_res["Keltner"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
            total_combos += cnt

            # MultiFactor
            t0 = time.time()
            best_r, best_p, cnt = -1e18, "", 0
            for rp in [5, 7, 9, 14, 21, 28]:
                for mp in range(5, 80, 5):
                    for vp in range(5, 50, 5):
                        for lt in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
                            for st in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
                                r = bt_multifactor(close, rp, mp, vp, lt, st)
                                if r > best_r:
                                    best_r = r
                                    best_p = f"rsi={rp}, mom={mp}, vol={vp}, lt={lt}, st={st}"
                                cnt += 1
            sym_res["MultiFactor"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
            total_combos += cnt

            # VolRegime
            t0 = time.time()
            best_r, best_p, cnt = -1e18, "", 0
            for ap in [10, 14, 20]:
                for vt in [0.010, 0.015, 0.020, 0.025, 0.030, 0.035]:
                    for ms in [3, 5, 10, 15, 20]:
                        for ml in [20, 30, 40, 50, 60, 80]:
                            if ms >= ml:
                                continue
                            for ros in [20, 25, 30, 35]:
                                for rob in [65, 70, 75, 80]:
                                    r = bt_vol_regime(close, high, low, ap, vt, ms, ml, 14, ros, rob)
                                    if r > best_r:
                                        best_r = r
                                        best_p = f"atr={ap}, vt={vt}, ms={ms}, ml={ml}, os={ros}, ob={rob}"
                                    cnt += 1
            sym_res["VolRegime"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
            total_combos += cnt

        # ---------- New 8 strategies ----------

        # Connors RSI-2
        t0 = time.time()
        best_r, best_p, cnt = -1e18, "", 0
        for rp in [2, 3, 4, 5, 7, 9]:
            for mat in [50, 100, 150, 200]:
                for mae in [3, 5, 7, 10, 15]:
                    for os_t in [3.0, 5.0, 10.0, 15.0, 20.0]:
                        for ob_t in [80.0, 85.0, 90.0, 95.0]:
                            r = bt_connors_rsi2(close, rp, mat, mae, os_t, ob_t)
                            if r > best_r:
                                best_r = r
                                best_p = f"rsi={rp}, maT={mat}, maE={mae}, os={os_t}, ob={ob_t}"
                            cnt += 1
        sym_res["ConnorsRSI2"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
        total_combos += cnt

        # MESA Adaptive
        t0 = time.time()
        best_r, best_p, cnt = -1e18, "", 0
        for fl in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for sl in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
                r = bt_mesa_adaptive(close, fl, sl)
                if r > best_r:
                    best_r = r
                    best_p = f"fast_lim={fl}, slow_lim={sl}"
                cnt += 1
        sym_res["MESA"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
        total_combos += cnt

        # KAMA Crossover
        t0 = time.time()
        best_r, best_p, cnt = -1e18, "", 0
        for erp in [5, 8, 10, 15, 20, 30]:
            for fsc in [2, 3, 5]:
                for ssc in [20, 30, 40, 50]:
                    for asm in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                        for ap in [10, 14, 20]:
                            r = bt_kama_crossover(close, erp, fsc, ssc, asm, high, low, ap)
                            if r > best_r:
                                best_r = r
                                best_p = f"er={erp}, fast={fsc}, slow={ssc}, atr_m={asm}, atr_p={ap}"
                            cnt += 1
        sym_res["KAMA"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
        total_combos += cnt

        # Donchian ATR
        t0 = time.time()
        best_r, best_p, cnt = -1e18, "", 0
        for ep in range(5, 80, 3):
            for ap in [7, 10, 14, 20]:
                for am in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
                    r = bt_donchian_atr(close, high, low, ep, ap, am)
                    if r > best_r:
                        best_r = r
                        best_p = f"entry={ep}, atr_p={ap}, mult={am}"
                    cnt += 1
        sym_res["DonchianATR"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
        total_combos += cnt

        # Dual Thrust
        t0 = time.time()
        best_r, best_p, cnt = -1e18, "", 0
        for lb in [2, 3, 4, 5, 7, 10, 14, 20]:
            for ku in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
                for kd in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
                    r = bt_dual_thrust(close, high, low, open_p, lb, ku, kd)
                    if r > best_r:
                        best_r = r
                        best_p = f"lb={lb}, k_up={ku}, k_down={kd}"
                    cnt += 1
        sym_res["DualThrust"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
        total_combos += cnt

        # Z-Score Reversion
        t0 = time.time()
        best_r, best_p, cnt = -1e18, "", 0
        for lb in range(10, 120, 5):
            for ez in [1.0, 1.5, 2.0, 2.5, 3.0]:
                for xz in [0.0, 0.25, 0.5, 0.75]:
                    for sz in [3.0, 3.5, 4.0, 5.0]:
                        r = bt_zscore_reversion(close, lb, ez, xz, sz)
                        if r > best_r:
                            best_r = r
                            best_p = f"lb={lb}, entry_z={ez}, exit_z={xz}, stop_z={sz}"
                        cnt += 1
        sym_res["ZScoreRev"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
        total_combos += cnt

        # Momentum Breakout
        t0 = time.time()
        best_r, best_p, cnt = -1e18, "", 0
        for hp in [20, 40, 60, 100, 150, 200, 252]:
            for pp in [0.00, 0.01, 0.02, 0.03, 0.05, 0.08]:
                for ap in [10, 14, 20]:
                    for at in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                        r = bt_momentum_breakout(close, high, low, hp, pp, ap, at)
                        if r > best_r:
                            best_r = r
                            best_p = f"high_p={hp}, prox={pp}, atr_p={ap}, trail={at}"
                        cnt += 1
        sym_res["MomBreakout"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
        total_combos += cnt

        # Regime Switch EMA
        t0 = time.time()
        best_r, best_p, cnt = -1e18, "", 0
        for ap in [10, 14, 20]:
            for vt in [0.010, 0.015, 0.020, 0.025, 0.030]:
                for fe in [3, 5, 8, 10, 15]:
                    for se in [15, 20, 30, 40, 50, 60]:
                        if fe >= se:
                            continue
                        for te in [30, 50, 80, 100]:
                            r = bt_regime_switch_ema(close, high, low, ap, vt, fe, se, te)
                            if r > best_r:
                                best_r = r
                                best_p = f"atr={ap}, vt={vt}, fast={fe}, slow={se}, trend={te}"
                            cnt += 1
        sym_res["RegimeEMA"] = {"return": best_r, "params": best_p, "combos": cnt, "time": time.time() - t0}
        total_combos += cnt

        results_all[sym] = sym_res
        done_combos = sum(v["combos"] for v in sym_res.values())
        print(f"  {sym}: done ({done_combos:,} combos)")

    grand_elapsed = time.time() - grand_t0

    # ============ FINAL REPORT ============
    n_strats = len(results_all[symbols[0]])
    print(f"\n{'=' * 80}")
    print(f"  {n_strats} 大策略终极排名 · 3 年数据 · 密集参数扫描 · 扣除滑点手续费")
    print(f"{'=' * 80}")
    print(f"总计回测次数: {total_combos:,}")
    print(f"总耗时: {grand_elapsed:.1f}s ({grand_elapsed / 60:.1f} min)")
    print(f"吞吐量: {total_combos / grand_elapsed:,.0f} combos/sec\n")

    # Per-symbol ranking
    for sym in symbols:
        sr = results_all[sym]
        ranked = sorted(sr.items(), key=lambda x: x[1]["return"], reverse=True)
        print(f"\n--- {sym} 排名 ---")
        print(f"{'Rank':>4}  {'Strategy':>16}  {'Return':>10}  {'Combos':>10}  {'Time':>7}  Params")
        for i, (sn, sd) in enumerate(ranked):
            print(f"{i + 1:4d}  {sn:>16}  {sd['return']:+8.2f}%  {sd['combos']:>10,}  {sd['time']:5.2f}s  {sd['params']}")

    # Cross-symbol average ranking
    print(f"\n\n{'=' * 80}")
    print(f"  策略平均收益排名 (3 标的平均)")
    print(f"{'=' * 80}")
    avg_ret = {}
    for sn in results_all[symbols[0]]:
        avg_ret[sn] = np.mean([results_all[sym][sn]["return"] for sym in symbols])
    ranked_avg = sorted(avg_ret.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Rank':>4}  {'Strategy':>16}  {'Avg 3yr Return':>15}  AAPL / GOOGL / TSLA")
    for i, (sn, ar) in enumerate(ranked_avg):
        vals = [results_all[sym][sn]["return"] for sym in symbols]
        print(f"{i + 1:4d}  {sn:>16}  {ar:+12.2f}%   {vals[0]:+.1f}% / {vals[1]:+.1f}% / {vals[2]:+.1f}%")

    # Timing breakdown
    print(f"\n--- 耗时分布 ---")
    strat_t = defaultdict(float)
    strat_c = defaultdict(int)
    for sym in symbols:
        for sn, sd in results_all[sym].items():
            strat_t[sn] += sd["time"]
            strat_c[sn] += sd["combos"]
    for sn in sorted(strat_t, key=strat_t.get, reverse=True):
        pct = strat_t[sn] / grand_elapsed * 100
        speed = strat_c[sn] / strat_t[sn] if strat_t[sn] > 0 else 0
        print(f"  {sn:>16}: {strat_t[sn]:6.2f}s  {strat_c[sn]:>10,} combos  ({pct:5.1f}%)  {speed:,.0f}/sec")

    # Champion
    print(f"\n{'=' * 80}")
    print(f"  CHAMPION: {ranked_avg[0][0]} ({ranked_avg[0][1]:+.2f}% avg 3yr return)")
    print(f"  RUNNER-UP: {ranked_avg[1][0]} ({ranked_avg[1][1]:+.2f}% avg)")
    print(f"  3RD PLACE: {ranked_avg[2][0]} ({ranked_avg[2][1]:+.2f}% avg)")
    print(f"{'=' * 80}")
