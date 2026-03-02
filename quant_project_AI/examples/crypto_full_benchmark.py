#!/usr/bin/env python3
"""
==========================================================================
  加密货币 18 大策略全面回测 · 风险指标 · 数据校验 · 币安成本
==========================================================================

标的: BTC-USD, ETH-USD, XRP-USD, SOL-USD
成本: 币安普通用户 Maker/Taker 0.10% (不使用BNB折扣)
滑点: 加密货币按 10bps (0.10%) — 24h 市场流动性略差于美股
数据: yfinance 3 年日线
"""

import numpy as np
import pandas as pd
import time
import sys
import os
import warnings
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Any

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

# =====================================================================
#  币安普通用户成本参数
# =====================================================================
# 币安 VIP0 现货: Maker 0.10%, Taker 0.10%
# 滑点: 加密货币 10bps (波动大, 流动性不如美股)
BINANCE_COMM_PCT = 0.001       # 0.10% 单边手续费
BINANCE_SLIP_BUY = 1.001       # 买入滑点 10bps
BINANCE_SLIP_SELL = 0.999      # 卖出滑点 10bps
# Numba 内核需要的参数
SLIP_BUY = BINANCE_SLIP_BUY
SLIP_SELL = BINANCE_SLIP_SELL
COMM = BINANCE_COMM_PCT

# =====================================================================
#  Import Numba strategies
# =====================================================================
from numba import njit

# --- 经典 3 策略 helpers ---
from param_scan_benchmark import (
    _bt_ma, _bt_rsi, _bt_macd,
    precompute_all_ma, precompute_all_ema, precompute_all_rsi,
)
# --- 高级 7 策略 ---
from advanced_strategies_benchmark import (
    bt_drift_regime, bt_ramom, bt_turtle, bt_bollinger,
    bt_keltner, bt_multifactor, bt_vol_regime,
)
# --- 前沿 8 策略 ---
from cutting_edge_strategies import (
    bt_connors_rsi2, bt_mesa_adaptive, bt_kama_crossover,
    bt_donchian_atr, bt_dual_thrust, bt_zscore_reversion,
    bt_momentum_breakout, bt_regime_switch_ema,
)

# =====================================================================
#  重写 Numba 策略：使用币安成本
# =====================================================================
# 注意: 原始 Numba 函数的 SLIP_BUY/SLIP_SELL/COMM 是硬编码常量
# 对于 classic 3 策略 (_bt_ma, _bt_rsi, _bt_macd), 它们接受参数
# 对于 advanced/cutting_edge 策略, 它们使用模块级常量
# 我们需要在导入后修改 cutting_edge_strategies 和 advanced_strategies_benchmark 的常量
import cutting_edge_strategies
import advanced_strategies_benchmark

# Patch 常量 — 注意: 这些模块的 Numba 函数在编译时已经捕获了常量
# 由于 Numba @njit 已编译, 我们需要重新定义带参数的版本
# 最可靠的方式: 用包装函数并传递正确成本

@njit(cache=True)
def bt_ma_binance(close, ma_short, ma_long):
    return _bt_ma(close, ma_short, ma_long, SLIP_BUY, SLIP_SELL, COMM)

@njit(cache=True)
def bt_rsi_binance(close, rsi_arr, os_threshold, ob_threshold):
    return _bt_rsi(close, rsi_arr, os_threshold, ob_threshold, SLIP_BUY, SLIP_SELL, COMM)

@njit(cache=True)
def bt_macd_binance(close, ema_fast, ema_slow, signal_span):
    return _bt_macd(close, ema_fast, ema_slow, signal_span, SLIP_BUY, SLIP_SELL, COMM)


# For advanced/cutting_edge strategies that use module-level constants,
# we redefine the critical ones with correct Binance costs
@njit(cache=True)
def bt_drift_regime_bn(close, lookback, drift_threshold, hold_period):
    n = len(close)
    position = 0
    entry_price = 0.0
    hold_count = 0
    total_ret = 1.0
    for i in range(lookback, n):
        up_days = 0
        for j in range(1, lookback + 1):
            if close[i - j + 1] > close[i - j]:
                up_days += 1
        up_ratio = up_days / lookback
        if position != 0:
            hold_count += 1
            if hold_count >= hold_period:
                if position == 1:
                    exit_p = close[i] * SLIP_SELL
                    trade_ret = (exit_p * (1.0 - COMM)) / (entry_price * (1.0 + COMM))
                else:
                    exit_p = close[i] * SLIP_BUY
                    trade_ret = (entry_price * (1.0 - COMM)) / (exit_p * (1.0 + COMM))
                total_ret *= trade_ret
                position = 0
        if position == 0:
            if up_ratio >= drift_threshold:
                entry_price = close[i] * SLIP_BUY
                position = -1
                hold_count = 0
            elif up_ratio <= (1.0 - drift_threshold):
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


@njit(cache=True)
def bt_turtle_bn(close, high, low, entry_period, exit_period, atr_period, atr_mult):
    n = len(close)
    atr_v = np.full(n, np.nan)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]; hc = abs(high[i] - close[i-1]); lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
    s = 0.0
    for i in range(atr_period): s += tr[i]
    atr_v[atr_period-1] = s / atr_period
    for i in range(atr_period, n):
        atr_v[i] = (atr_v[i-1]*(atr_period-1) + tr[i]) / atr_period
    position = 0; entry_price = 0.0; stop_price = 0.0; total_ret = 1.0
    start = max(entry_period, max(exit_period, atr_period))
    for i in range(start, n):
        hh = high[i-1]; ll = low[i-1]; hh_exit = high[i-1]; ll_exit = low[i-1]
        for j in range(1, entry_period):
            if high[i-1-j] > hh: hh = high[i-1-j]
            if low[i-1-j] < ll: ll = low[i-1-j]
        for j in range(1, exit_period):
            if high[i-1-j] > hh_exit: hh_exit = high[i-1-j]
            if low[i-1-j] < ll_exit: ll_exit = low[i-1-j]
        a = atr_v[i]
        if a != a: continue
        if position == 1:
            if close[i] < stop_price or close[i] < ll_exit:
                exit_p = close[i] * SLIP_SELL
                total_ret *= (exit_p*(1.0-COMM)) / (entry_price*(1.0+COMM))
                position = 0
        elif position == -1:
            if close[i] > stop_price or close[i] > hh_exit:
                exit_p = close[i] * SLIP_BUY
                total_ret *= (entry_price*(1.0-COMM)) / (exit_p*(1.0+COMM))
                position = 0
        if position == 0:
            if close[i] > hh:
                entry_price = close[i] * SLIP_BUY
                stop_price = close[i] - atr_mult * a
                position = 1
            elif close[i] < ll:
                entry_price = close[i] * SLIP_SELL
                stop_price = close[i] + atr_mult * a
                position = -1
    if position == 1:
        exit_p = close[n-1]*SLIP_SELL; total_ret *= (exit_p*(1.0-COMM)) / (entry_price*(1.0+COMM))
    elif position == -1:
        exit_p = close[n-1]*SLIP_BUY; total_ret *= (entry_price*(1.0-COMM)) / (exit_p*(1.0+COMM))
    return (total_ret - 1.0) * 100.0


@njit(cache=True)
def bt_momentum_breakout_bn(close, high, low, high_period, proximity_pct, atr_period, atr_trail):
    n = len(close)
    if n < max(high_period, atr_period) + 2: return 0.0
    atr_v = np.full(n, np.nan)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i]-low[i]; hc = abs(high[i]-close[i-1]); lc = abs(low[i]-close[i-1])
        tr[i] = max(hl, max(hc, lc))
    s = 0.0
    for i in range(atr_period): s += tr[i]
    atr_v[atr_period-1] = s / atr_period
    for i in range(atr_period, n):
        atr_v[i] = (atr_v[i-1]*(atr_period-1)+tr[i])/atr_period
    roll_high = np.full(n, np.nan)
    for i in range(high_period-1, n):
        mx = high[i]
        for j in range(1, high_period):
            if high[i-j]>mx: mx=high[i-j]
        roll_high[i] = mx
    position = 0; entry_price = 0.0; trailing_stop = 0.0; total_ret = 1.0
    start = max(high_period, atr_period)
    for i in range(start, n):
        rh = roll_high[i]; a = atr_v[i]
        if rh != rh or a != a: continue
        if position == 1:
            ns = close[i] - atr_trail*a
            if ns > trailing_stop: trailing_stop = ns
            if close[i] < trailing_stop:
                exit_p = close[i]*SLIP_SELL
                total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
                position = 0
        if position == 0:
            threshold = rh * (1.0 - proximity_pct)
            if close[i] >= threshold:
                entry_price = close[i]*SLIP_BUY
                trailing_stop = close[i] - atr_trail*a
                position = 1
    if position == 1:
        exit_p = close[n-1]*SLIP_SELL
        total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
    return (total_ret - 1.0) * 100.0


@njit(cache=True)
def bt_zscore_bn(close, lookback, entry_z, exit_z, stop_z):
    n = len(close)
    if n < lookback + 2: return 0.0
    r_mean = np.full(n, np.nan); r_std = np.full(n, np.nan)
    s = 0.0; s2 = 0.0
    for i in range(n):
        s += close[i]; s2 += close[i]*close[i]
        if i >= lookback: s -= close[i-lookback]; s2 -= close[i-lookback]*close[i-lookback]
        if i >= lookback - 1:
            mean = s/lookback; var = s2/lookback - mean*mean
            r_mean[i] = mean; r_std[i] = np.sqrt(max(0.0, var))
    position = 0; entry_price = 0.0; total_ret = 1.0
    for i in range(lookback, n):
        m = r_mean[i]; sd = r_std[i]
        if sd == 0.0 or sd != sd: continue
        z = (close[i] - m) / sd
        if position == 1:
            if z > -exit_z or z > stop_z:
                exit_p = close[i]*SLIP_SELL
                total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
                position = 0
        elif position == -1:
            if z < exit_z or z < -stop_z:
                exit_p = close[i]*SLIP_BUY
                total_ret *= (entry_price*(1.0-COMM))/(exit_p*(1.0+COMM))
                position = 0
        if position == 0:
            if z < -entry_z:
                entry_price = close[i]*SLIP_BUY; position = 1
            elif z > entry_z:
                entry_price = close[i]*SLIP_SELL; position = -1
    if position == 1:
        exit_p = close[n-1]*SLIP_SELL; total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
    elif position == -1:
        exit_p = close[n-1]*SLIP_BUY; total_ret *= (entry_price*(1.0-COMM))/(exit_p*(1.0+COMM))
    return (total_ret - 1.0) * 100.0


# =====================================================================
#  详细回测引擎 (输出逐日净值曲线 + 交易列表)
# =====================================================================

@njit(cache=True)
def detailed_backtest_ma(close, ma_short, ma_long):
    """MA 交叉策略 — 返回逐日净值和交易次数"""
    n = len(close)
    equity = np.ones(n, dtype=np.float64)
    position = 0; entry_price = 0.0; total_ret = 1.0; trade_count = 0
    for i in range(1, n):
        s = ma_short[i]; l = ma_long[i]; sp = ma_short[i-1]; lp = ma_long[i-1]
        if s != s or l != l or sp != sp or lp != lp:
            equity[i] = total_ret; continue
        if sp <= lp and s > l and position == 0:
            entry_price = close[i] * SLIP_BUY; position = 1; trade_count += 1
        elif sp >= lp and s < l and position == 1:
            exit_p = close[i] * SLIP_SELL
            total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
            position = 0; trade_count += 1
        if position == 1:
            equity[i] = total_ret * (close[i]*SLIP_SELL*(1.0-COMM)) / (entry_price*(1.0+COMM))
        else:
            equity[i] = total_ret
    if position == 1:
        exit_p = close[n-1]*SLIP_SELL
        total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
        equity[n-1] = total_ret
    return equity, trade_count


@njit(cache=True)
def detailed_backtest_generic(close, high, low, strategy_id,
                              p1, p2, p3, p4, p5, p6):
    """
    通用详细回测 — 返回逐日净值曲线
    strategy_id: 1=RSI, 2=MACD, 3=DriftRegime, 4=Turtle, 5=MomBreakout, 6=ZScore
    """
    n = len(close)
    equity = np.ones(n, dtype=np.float64)
    position = 0; entry_price = 0.0; total_ret = 1.0; trade_count = 0; hold_count = 0
    trailing_stop = 0.0

    # RSI pre-compute
    rsi = np.full(n, np.nan)
    if strategy_id == 1:
        rsi_period = int(p1)
        if n > rsi_period + 1:
            gs = 0.0; ls = 0.0
            for i in range(1, rsi_period+1):
                d = close[i]-close[i-1]
                if d > 0: gs += d
                else: ls -= d
            ag = gs/rsi_period; al = ls/rsi_period
            rsi[rsi_period] = 100.0 if al == 0.0 else 100.0 - 100.0/(1.0+ag/al)
            for i in range(rsi_period+1, n):
                d = close[i]-close[i-1]
                g = d if d>0 else 0.0; l_v = -d if d<0 else 0.0
                ag = (ag*(rsi_period-1)+g)/rsi_period
                al = (al*(rsi_period-1)+l_v)/rsi_period
                rsi[i] = 100.0 if al == 0.0 else 100.0 - 100.0/(1.0+ag/al)

    # MACD pre-compute
    ema_f = np.empty(n, dtype=np.float64)
    ema_s = np.empty(n, dtype=np.float64)
    macd_line = np.empty(n, dtype=np.float64)
    signal_line = np.empty(n, dtype=np.float64)
    if strategy_id == 2:
        fast_p = int(p1); slow_p = int(p2); sig_p = int(p3)
        kf = 2.0/(fast_p+1.0); ks = 2.0/(slow_p+1.0); ksig = 2.0/(sig_p+1.0)
        ema_f[0] = close[0]; ema_s[0] = close[0]
        for i in range(1, n):
            ema_f[i] = close[i]*kf + ema_f[i-1]*(1.0-kf)
            ema_s[i] = close[i]*ks + ema_s[i-1]*(1.0-ks)
        for i in range(n): macd_line[i] = ema_f[i] - ema_s[i]
        signal_line[0] = macd_line[0]
        for i in range(1, n):
            signal_line[i] = macd_line[i]*ksig + signal_line[i-1]*(1.0-ksig)

    # ATR pre-compute for strategies needing it
    atr_v = np.full(n, np.nan)
    if strategy_id in (4, 5):
        atr_period = int(p3) if strategy_id == 4 else int(p3)
        tr_arr = np.empty(n, dtype=np.float64)
        tr_arr[0] = high[0]-low[0]
        for i in range(1, n):
            hl = high[i]-low[i]; hc = abs(high[i]-close[i-1]); lc = abs(low[i]-close[i-1])
            tr_arr[i] = max(hl, max(hc, lc))
        s_v = 0.0
        for i in range(atr_period): s_v += tr_arr[i]
        atr_v[atr_period-1] = s_v/atr_period
        for i in range(atr_period, n):
            atr_v[i] = (atr_v[i-1]*(atr_period-1)+tr_arr[i])/atr_period

    start = 1
    if strategy_id == 1: start = int(p1) + 1
    elif strategy_id == 2: start = max(int(p1), int(p2)) + 1
    elif strategy_id == 3: start = int(p1)
    elif strategy_id == 4: start = max(int(p1), max(int(p2), int(p3)))
    elif strategy_id == 5: start = max(int(p1), int(p3))
    elif strategy_id == 6: start = int(p1)

    # Z-Score pre-compute
    r_mean = np.full(n, np.nan); r_std = np.full(n, np.nan)
    if strategy_id == 6:
        lb = int(p1)
        s_z = 0.0; s2_z = 0.0
        for i in range(n):
            s_z += close[i]; s2_z += close[i]*close[i]
            if i >= lb: s_z -= close[i-lb]; s2_z -= close[i-lb]*close[i-lb]
            if i >= lb - 1:
                mean = s_z/lb; var = s2_z/lb - mean*mean
                r_mean[i] = mean; r_std[i] = np.sqrt(max(0.0, var))

    for i in range(max(start, 1), n):
        # --- RSI ---
        if strategy_id == 1:
            r = rsi[i]; os_t = p2; ob_t = p3
            if r != r: equity[i] = equity[i-1]; continue
            if position == 0:
                if r < os_t:
                    entry_price = close[i]*SLIP_BUY; position = 1; trade_count += 1
            elif position == 1:
                if r > ob_t:
                    exit_p = close[i]*SLIP_SELL
                    total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
                    position = 0; trade_count += 1
        # --- MACD ---
        elif strategy_id == 2:
            mp = macd_line[i-1]; sp_v = signal_line[i-1]; mc = macd_line[i]; sc = signal_line[i]
            if mp != mp or sp_v != sp_v or mc != mc or sc != sc:
                equity[i] = equity[i-1]; continue
            if mp <= sp_v and mc > sc and position == 0:
                entry_price = close[i]*SLIP_BUY; position = 1; trade_count += 1
            elif mp >= sp_v and mc < sc and position == 1:
                exit_p = close[i]*SLIP_SELL
                total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
                position = 0; trade_count += 1
        # --- DriftRegime ---
        elif strategy_id == 3:
            lookback = int(p1); drift_thr = p2; hold_p = int(p3)
            if i < lookback: equity[i] = equity[i-1]; continue
            up_days = 0
            for j in range(1, lookback+1):
                if close[i-j+1] > close[i-j]: up_days += 1
            up_ratio = up_days / lookback
            if position != 0:
                hold_count += 1
                if hold_count >= hold_p:
                    if position == 1:
                        exit_p = close[i]*SLIP_SELL
                        total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
                    else:
                        exit_p = close[i]*SLIP_BUY
                        total_ret *= (entry_price*(1.0-COMM))/(exit_p*(1.0+COMM))
                    position = 0; trade_count += 1
            if position == 0:
                if up_ratio >= drift_thr:
                    entry_price = close[i]*SLIP_BUY; position = -1; hold_count = 0; trade_count += 1
                elif up_ratio <= (1.0-drift_thr):
                    entry_price = close[i]*SLIP_BUY; position = 1; hold_count = 0; trade_count += 1
        # --- Turtle ---
        elif strategy_id == 4:
            ep = int(p1); xp = int(p2); am = p4
            if i < max(ep, xp): equity[i] = equity[i-1]; continue
            hh = high[i-1]; ll = low[i-1]
            for j in range(1, ep):
                if high[i-1-j]>hh: hh=high[i-1-j]
                if low[i-1-j]<ll: ll=low[i-1-j]
            a = atr_v[i]
            if a != a: equity[i] = equity[i-1]; continue
            if position == 1:
                if close[i] < trailing_stop:
                    exit_p = close[i]*SLIP_SELL
                    total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
                    position = 0; trade_count += 1
            if position == 0:
                if close[i] > hh:
                    entry_price = close[i]*SLIP_BUY; trailing_stop = close[i]-am*a
                    position = 1; trade_count += 1
        # --- MomBreakout ---
        elif strategy_id == 5:
            hp = int(p1); pp = p2; at = p4
            if i < hp: equity[i] = equity[i-1]; continue
            mx = high[i]
            for j in range(1, hp):
                if high[i-j]>mx: mx=high[i-j]
            a = atr_v[i]
            if a != a: equity[i] = equity[i-1]; continue
            if position == 1:
                ns = close[i] - at*a
                if ns > trailing_stop: trailing_stop = ns
                if close[i] < trailing_stop:
                    exit_p = close[i]*SLIP_SELL
                    total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
                    position = 0; trade_count += 1
            if position == 0:
                thr = mx*(1.0-pp)
                if close[i] >= thr:
                    entry_price = close[i]*SLIP_BUY; trailing_stop = close[i]-at*a
                    position = 1; trade_count += 1
        # --- ZScore ---
        elif strategy_id == 6:
            lb = int(p1); ez = p2; xz = p3; sz = p4
            m = r_mean[i]; sd = r_std[i]
            if sd == 0.0 or sd != sd: equity[i] = equity[i-1]; continue
            z = (close[i] - m)/sd
            if position == 1:
                if z > -xz or z > sz:
                    exit_p = close[i]*SLIP_SELL
                    total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
                    position = 0; trade_count += 1
            elif position == -1:
                if z < xz or z < -sz:
                    exit_p = close[i]*SLIP_BUY
                    total_ret *= (entry_price*(1.0-COMM))/(exit_p*(1.0+COMM))
                    position = 0; trade_count += 1
            if position == 0:
                if z < -ez:
                    entry_price = close[i]*SLIP_BUY; position = 1; trade_count += 1
                elif z > ez:
                    entry_price = close[i]*SLIP_SELL; position = -1; trade_count += 1

        # Update equity
        if position == 1:
            equity[i] = total_ret * (close[i]*SLIP_SELL*(1.0-COMM))/(entry_price*(1.0+COMM))
        elif position == -1:
            equity[i] = total_ret * (entry_price*(1.0-COMM))/(close[i]*SLIP_BUY*(1.0+COMM))
        else:
            equity[i] = total_ret

    # Close position
    if position == 1:
        exit_p = close[n-1]*SLIP_SELL
        total_ret *= (exit_p*(1.0-COMM))/(entry_price*(1.0+COMM))
        equity[n-1] = total_ret
    elif position == -1:
        exit_p = close[n-1]*SLIP_BUY
        total_ret *= (entry_price*(1.0-COMM))/(exit_p*(1.0+COMM))
        equity[n-1] = total_ret

    return equity, trade_count


# =====================================================================
#  风险指标计算
# =====================================================================

def calc_risk_metrics(equity: np.ndarray, trading_days: int = 365) -> Dict[str, float]:
    """计算全面的风险指标"""
    daily_returns = np.diff(equity) / equity[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    n = len(daily_returns)
    if n < 2:
        return {k: 0.0 for k in [
            "total_return_pct", "annual_return_pct", "annual_volatility_pct",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown_pct", "max_drawdown_duration_days",
            "var_95_pct", "cvar_95_pct", "win_rate_pct",
            "profit_factor", "avg_daily_return_pct", "skewness", "kurtosis",
            "positive_days_pct", "best_day_pct", "worst_day_pct",
            "longest_win_streak", "longest_loss_streak",
        ]}

    total_ret = (equity[-1] / equity[0] - 1.0) * 100.0
    years = n / trading_days
    annual_ret = ((equity[-1] / equity[0]) ** (1.0 / max(years, 0.01)) - 1.0) * 100.0 if years > 0 else 0.0
    annual_vol = np.std(daily_returns) * np.sqrt(trading_days) * 100.0

    # Sharpe (risk-free = 0 for crypto)
    sharpe = (np.mean(daily_returns) * trading_days) / (np.std(daily_returns) * np.sqrt(trading_days)) if np.std(daily_returns) > 0 else 0.0

    # Sortino (downside deviation)
    neg_returns = daily_returns[daily_returns < 0]
    downside_std = np.sqrt(np.mean(neg_returns ** 2)) if len(neg_returns) > 0 else 1e-10
    sortino = (np.mean(daily_returns) * trading_days) / (downside_std * np.sqrt(trading_days))

    # Max Drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = abs(np.min(drawdowns)) * 100.0

    # Max Drawdown Duration
    dd_dur = 0; max_dd_dur = 0; in_dd = False
    for i in range(len(equity)):
        if equity[i] < running_max[i]:
            dd_dur += 1
            in_dd = True
        else:
            if in_dd:
                if dd_dur > max_dd_dur: max_dd_dur = dd_dur
                dd_dur = 0; in_dd = False
    if dd_dur > max_dd_dur: max_dd_dur = dd_dur

    # Calmar
    calmar = annual_ret / max_dd if max_dd > 0 else 0.0

    # VaR and CVaR
    sorted_returns = np.sort(daily_returns)
    var_idx = max(0, int(n * 0.05) - 1)
    var_95 = abs(sorted_returns[var_idx]) * 100.0
    cvar_95 = abs(np.mean(sorted_returns[:var_idx + 1])) * 100.0 if var_idx > 0 else var_95

    # Win rate
    positive = daily_returns[daily_returns > 0]
    negative = daily_returns[daily_returns < 0]
    win_rate = len(positive) / n * 100.0 if n > 0 else 0.0

    # Profit factor
    gross_profit = np.sum(positive) if len(positive) > 0 else 0.0
    gross_loss = abs(np.sum(negative)) if len(negative) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    # Skewness and Kurtosis
    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns)
    if std_r > 0:
        skew = np.mean(((daily_returns - mean_r) / std_r) ** 3)
        kurt = np.mean(((daily_returns - mean_r) / std_r) ** 4) - 3.0
    else:
        skew = 0.0; kurt = 0.0

    # Streaks
    longest_win = 0; longest_loss = 0; cur_win = 0; cur_loss = 0
    for r in daily_returns:
        if r > 0:
            cur_win += 1; cur_loss = 0
            if cur_win > longest_win: longest_win = cur_win
        elif r < 0:
            cur_loss += 1; cur_win = 0
            if cur_loss > longest_loss: longest_loss = cur_loss
        else:
            cur_win = 0; cur_loss = 0

    return {
        "total_return_pct": round(total_ret, 2),
        "annual_return_pct": round(annual_ret, 2),
        "annual_volatility_pct": round(annual_vol, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "max_drawdown_duration_days": max_dd_dur,
        "var_95_pct": round(var_95, 2),
        "cvar_95_pct": round(cvar_95, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "avg_daily_return_pct": round(np.mean(daily_returns) * 100, 4),
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 3),
        "positive_days_pct": round(len(positive) / max(n, 1) * 100, 2),
        "best_day_pct": round(np.max(daily_returns) * 100, 2),
        "worst_day_pct": round(np.min(daily_returns) * 100, 2),
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
    }


# =====================================================================
#  数据校验
# =====================================================================

def validate_data(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """全面校验数据质量"""
    report = {"symbol": symbol, "issues": [], "passed": True}
    report["rows"] = len(df)
    report["date_range"] = f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}"
    report["trading_days"] = len(df)

    # 检查缺失值
    null_counts = df[["open", "high", "low", "close", "volume"]].isnull().sum()
    if null_counts.sum() > 0:
        report["issues"].append(f"缺失值: {null_counts.to_dict()}")
        report["passed"] = False

    # 检查异常价格
    if (df["close"] <= 0).any():
        report["issues"].append(f"存在非正收盘价: {(df['close'] <= 0).sum()} 行")
        report["passed"] = False

    if (df["high"] < df["low"]).any():
        report["issues"].append(f"high < low 异常: {(df['high'] < df['low']).sum()} 行")
        report["passed"] = False

    # 检查价格跳跃
    pct_change = df["close"].pct_change().abs()
    extreme_jumps = pct_change[pct_change > 0.50]
    if len(extreme_jumps) > 0:
        report["issues"].append(f"单日涨跌幅 > 50%: {len(extreme_jumps)} 次 (加密货币可接受)")

    # 检查日期连续性（加密货币 7x24，但 yfinance 返回日线可能有间隙）
    date_diffs = df["date"].diff().dt.days
    gaps = date_diffs[date_diffs > 3]  # 超过 3 天无数据
    if len(gaps) > 0:
        report["issues"].append(f"日期间隙 > 3 天: {len(gaps)} 处")

    # 检查成交量
    zero_vol = (df["volume"] == 0).sum()
    if zero_vol > 0:
        report["issues"].append(f"零成交量: {zero_vol} 天")

    # 价格统计
    report["price_stats"] = {
        "mean": round(df["close"].mean(), 2),
        "std": round(df["close"].std(), 2),
        "min": round(df["close"].min(), 2),
        "max": round(df["close"].max(), 2),
        "first": round(df["close"].iloc[0], 2),
        "last": round(df["close"].iloc[-1], 2),
        "buy_hold_return_pct": round((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 2),
    }

    # 成交量统计
    report["volume_stats"] = {
        "mean": f"{df['volume'].mean():,.0f}",
        "median": f"{df['volume'].median():,.0f}",
    }

    if len(report["issues"]) == 0:
        report["issues"].append("全部通过 ✓")

    return report


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    import yfinance as yf

    print("=" * 80)
    print("  加密货币 18 大策略全面回测 · 币安成本 · 风险分析 · 数据校验")
    print("=" * 80)
    print(f"\n成本参数:")
    print(f"  手续费: {BINANCE_COMM_PCT*100:.2f}% (币安 VIP0 现货)")
    print(f"  滑点:   {(BINANCE_SLIP_BUY-1)*10000:.0f}bps 买 / {(1-BINANCE_SLIP_SELL)*10000:.0f}bps 卖")
    print(f"  单边总成本: ~{(BINANCE_COMM_PCT + BINANCE_SLIP_BUY - 1)*100:.2f}%\n")

    # ---- Step 1: 下载数据 ----
    symbols_map = {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "XRP": "XRP-USD",
        "SOL": "SOL-USD",
    }
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)

    print("[1/5] 下载 3 年日线数据 ...")
    os.makedirs("data", exist_ok=True)
    data_all = {}
    validation_reports = {}

    for short_name, ticker in symbols_map.items():
        print(f"  下载 {ticker} ...", end=" ", flush=True)
        df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                         end=end_date.strftime("%Y-%m-%d"), progress=False)
        if hasattr(df.columns, 'levels') or isinstance(df.columns[0], tuple):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if "date" not in df.columns and "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(f"data/{short_name}.csv", index=False)
        data_all[short_name] = df
        print(f"{len(df)} 天")

        # 数据校验
        validation_reports[short_name] = validate_data(df, short_name)

    # ---- Step 2: JIT 预热 ----
    print("\n[2/5] JIT 预热 ...", end=" ", flush=True)
    t0 = time.time()
    dc = np.random.rand(300).astype(np.float64) * 100 + 100
    dh = dc + np.random.rand(300) * 2
    dl = dc - np.random.rand(300) * 2
    dm = np.random.rand(300).astype(np.float64) * 100 + 100
    dr = np.random.rand(300).astype(np.float64) * 50 + 25

    bt_ma_binance(dc, dm, dm)
    bt_rsi_binance(dc, dr, 30.0, 70.0)
    bt_macd_binance(dc, dm, dm, 9)
    bt_drift_regime_bn(dc, 20, 0.6, 5)
    bt_turtle_bn(dc, dh, dl, 10, 5, 14, 2.0)
    bt_momentum_breakout_bn(dc, dh, dl, 50, 0.02, 14, 2.0)
    bt_zscore_bn(dc, 20, 2.0, 0.5, 4.0)
    detailed_backtest_ma(dc, dm, dm)
    detailed_backtest_generic(dc, dh, dl, 1, 14.0, 30.0, 70.0, 0.0, 0.0, 0.0)
    detailed_backtest_generic(dc, dh, dl, 2, 12.0, 26.0, 9.0, 0.0, 0.0, 0.0)
    detailed_backtest_generic(dc, dh, dl, 3, 15.0, 0.62, 27.0, 0.0, 0.0, 0.0)
    detailed_backtest_generic(dc, dh, dl, 4, 20.0, 10.0, 14.0, 2.0, 0.0, 0.0)
    detailed_backtest_generic(dc, dh, dl, 5, 40.0, 0.03, 14.0, 2.0, 0.0, 0.0)
    detailed_backtest_generic(dc, dh, dl, 6, 35.0, 2.5, 0.5, 3.0, 0.0, 0.0)
    print(f"完成 ({time.time()-t0:.1f}s)")

    # ---- Step 3: 使用从股票优化出的最优参数 ----
    # 这些参数来自之前 AAPL/GOOGL/TSLA 的密集参数扫描
    # 对加密货币适当调整
    print("\n[3/5] 定义策略与最优参数 ...")

    strategy_configs = [
        # (name, strategy_id, p1-p6 for detailed_backtest_generic, or "ma" special)
        # strategy_id: 0=MA, 1=RSI, 2=MACD, 3=DriftRegime, 4=Turtle, 5=MomBreakout, 6=ZScore
        ("MA(9,21)", "ma", 9, 21, 0, 0, 0, 0),
        ("MA(5,20)", "ma", 5, 20, 0, 0, 0, 0),
        ("MA(50,200)", "ma", 50, 200, 0, 0, 0, 0),
        ("RSI(9,15,80)", 1, 9.0, 15.0, 80.0, 0, 0, 0),
        ("RSI(14,30,70)", 1, 14.0, 30.0, 70.0, 0, 0, 0),
        ("RSI(28,35,75)", 1, 28.0, 35.0, 75.0, 0, 0, 0),
        ("MACD(12,26,9)", 2, 12.0, 26.0, 9.0, 0, 0, 0),
        ("MACD(14,76,50)", 2, 14.0, 76.0, 50.0, 0, 0, 0),
        ("MACD(16,18,16)", 2, 16.0, 18.0, 16.0, 0, 0, 0),
        ("DriftRegime(15,0.62,27)", 3, 15.0, 0.62, 27.0, 0, 0, 0),
        ("DriftRegime(10,0.52,19)", 3, 10.0, 0.52, 19.0, 0, 0, 0),
        ("Turtle(5,30,20,3.5)", 4, 5.0, 30.0, 20.0, 3.5, 0, 0),
        ("Turtle(5,39,10,1.0)", 4, 5.0, 39.0, 10.0, 1.0, 0, 0),
        ("MomBreakout(40,3%,14,2.0)", 5, 40.0, 0.03, 14.0, 2.0, 0, 0),
        ("MomBreakout(20,3%,14,1.5)", 5, 20.0, 0.03, 14.0, 1.5, 0, 0),
        ("ZScore(35,2.5,0.5,3.0)", 6, 35.0, 2.5, 0.5, 3.0, 0, 0),
        ("ZScore(85,2.5,0.0,3.0)", 6, 85.0, 2.5, 0.0, 3.0, 0, 0),
    ]

    print(f"  共 {len(strategy_configs)} 个策略配置\n")

    # ---- Step 4: 回测 ----
    print("[4/5] 执行回测 + 计算风险指标 ...\n")
    grand_t0 = time.time()
    all_results = {}  # {symbol: {strategy_name: {metrics, trade_count, equity}}}

    for sym in data_all:
        df = data_all[sym]
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        n = len(close)

        sym_results = {}
        # Buy & Hold baseline
        bh_equity = close / close[0]
        bh_metrics = calc_risk_metrics(bh_equity, trading_days=365)
        bh_metrics["trade_count"] = 1
        sym_results["Buy&Hold"] = bh_metrics

        for cfg in strategy_configs:
            name = cfg[0]
            sid = cfg[1]

            if sid == "ma":
                short_w = int(cfg[2]); long_w = int(cfg[3])
                mas = precompute_all_ma(close, max(short_w, long_w) + 1)
                if short_w < len(mas) and long_w < len(mas):
                    equity, tc = detailed_backtest_ma(close, mas[short_w], mas[long_w])
                else:
                    equity = np.ones(n); tc = 0
            else:
                equity, tc = detailed_backtest_generic(
                    close, high, low, sid,
                    float(cfg[2]), float(cfg[3]), float(cfg[4]),
                    float(cfg[5]), float(cfg[6]), float(cfg[7])
                )

            metrics = calc_risk_metrics(equity, trading_days=365)
            metrics["trade_count"] = tc
            sym_results[name] = metrics

        all_results[sym] = sym_results
        print(f"  {sym}: {len(sym_results)} 策略完成")

    elapsed = time.time() - grand_t0
    print(f"\n  总耗时: {elapsed:.2f}s")

    # ---- Step 5: 生成报告 ----
    print("\n[5/5] 生成校验与分析报告 ...\n")

    # Print data validation
    print("=" * 80)
    print("  一、数据校验报告")
    print("=" * 80)
    for sym, vr in validation_reports.items():
        status = "✓ PASS" if vr["passed"] else "✗ FAIL"
        print(f"\n  {sym} [{status}]")
        print(f"    日期范围: {vr['date_range']}")
        print(f"    数据量: {vr['rows']} 天")
        ps = vr["price_stats"]
        print(f"    价格: {ps['first']} → {ps['last']} (Buy&Hold {ps['buy_hold_return_pct']:+.2f}%)")
        print(f"    价格区间: {ps['min']} ~ {ps['max']} (均值 {ps['mean']}, 标准差 {ps['std']})")
        print(f"    成交量: 均值 {vr['volume_stats']['mean']}, 中位数 {vr['volume_stats']['median']}")
        for issue in vr["issues"]:
            print(f"    → {issue}")

    # Print per-symbol results
    key_metrics = ["total_return_pct", "annual_return_pct", "sharpe_ratio", "sortino_ratio",
                   "max_drawdown_pct", "calmar_ratio", "var_95_pct", "win_rate_pct",
                   "profit_factor", "trade_count"]
    metric_labels = {
        "total_return_pct": "总收益%",
        "annual_return_pct": "年化%",
        "sharpe_ratio": "Sharpe",
        "sortino_ratio": "Sortino",
        "max_drawdown_pct": "最大回撤%",
        "calmar_ratio": "Calmar",
        "var_95_pct": "VaR95%",
        "win_rate_pct": "胜率%",
        "profit_factor": "盈亏比",
        "trade_count": "交易次数",
    }

    for sym in data_all:
        print(f"\n{'=' * 80}")
        print(f"  二、{sym} 策略表现 (3年日线 · 币安 0.10% 手续费 + 10bps 滑点)")
        print(f"{'=' * 80}")

        sr = all_results[sym]
        # Sort by total return
        ranked = sorted(sr.items(), key=lambda x: x[1]["total_return_pct"], reverse=True)

        header = f"{'Rank':>4} {'Strategy':>28}"
        for m in key_metrics:
            header += f" {metric_labels[m]:>8}"
        print(header)
        print("-" * len(header))

        for i, (sname, metrics) in enumerate(ranked):
            row = f"{i+1:4d} {sname:>28}"
            for m in key_metrics:
                v = metrics[m]
                if isinstance(v, float):
                    row += f" {v:>8.2f}"
                else:
                    row += f" {v:>8}"
            print(row)

    # Cross-symbol average
    print(f"\n{'=' * 80}")
    print(f"  三、策略综合排名 (4 标的平均)")
    print(f"{'=' * 80}")

    strat_names = [cfg[0] for cfg in strategy_configs]
    strat_names.insert(0, "Buy&Hold")
    avg_metrics = {}
    for sn in strat_names:
        vals = {}
        for m in key_metrics:
            if m == "trade_count":
                vals[m] = np.mean([all_results[sym][sn][m] for sym in data_all])
            else:
                vals[m] = np.mean([all_results[sym][sn][m] for sym in data_all])
        avg_metrics[sn] = vals

    ranked_avg = sorted(avg_metrics.items(), key=lambda x: x[1]["sharpe_ratio"], reverse=True)

    header = f"{'Rank':>4} {'Strategy':>28}"
    for m in key_metrics:
        header += f" {metric_labels[m]:>8}"
    print(header)
    print("-" * len(header))
    for i, (sname, metrics) in enumerate(ranked_avg):
        row = f"{i+1:4d} {sname:>28}"
        for m in key_metrics:
            v = metrics[m]
            if isinstance(v, float):
                row += f" {v:>8.2f}"
            else:
                row += f" {v:>8}"
        print(row)

    # Risk summary
    print(f"\n{'=' * 80}")
    print(f"  四、风险指标详细报告 (最优策略 per symbol)")
    print(f"{'=' * 80}")

    for sym in data_all:
        sr = all_results[sym]
        # Best by Sharpe (exclude Buy&Hold)
        strats_only = {k: v for k, v in sr.items() if k != "Buy&Hold"}
        best_sharpe_name = max(strats_only, key=lambda x: strats_only[x]["sharpe_ratio"])
        best = sr[best_sharpe_name]
        print(f"\n  {sym} 最佳策略 (按 Sharpe): {best_sharpe_name}")
        print(f"    总收益:         {best['total_return_pct']:+.2f}%")
        print(f"    年化收益:       {best['annual_return_pct']:+.2f}%")
        print(f"    年化波动率:     {best['annual_volatility_pct']:.2f}%")
        print(f"    Sharpe Ratio:   {best['sharpe_ratio']:.3f}")
        print(f"    Sortino Ratio:  {best['sortino_ratio']:.3f}")
        print(f"    Calmar Ratio:   {best['calmar_ratio']:.3f}")
        print(f"    最大回撤:       {best['max_drawdown_pct']:.2f}%")
        print(f"    最长回撤天数:   {best['max_drawdown_duration_days']} 天")
        print(f"    VaR (95%):      {best['var_95_pct']:.2f}%")
        print(f"    CVaR (95%):     {best['cvar_95_pct']:.2f}%")
        print(f"    胜率:           {best['win_rate_pct']:.2f}%")
        print(f"    盈亏比:         {best['profit_factor']:.3f}")
        print(f"    日均收益:       {best['avg_daily_return_pct']:.4f}%")
        print(f"    偏度 (Skew):    {best['skewness']:.3f}")
        print(f"    峰度 (Kurt):    {best['kurtosis']:.3f}")
        print(f"    最佳单日:       {best['best_day_pct']:+.2f}%")
        print(f"    最差单日:       {best['worst_day_pct']:+.2f}%")
        print(f"    最长连胜:       {best['longest_win_streak']} 天")
        print(f"    最长连亏:       {best['longest_loss_streak']} 天")
        print(f"    交易次数:       {best['trade_count']}")

    # Cost analysis
    print(f"\n{'=' * 80}")
    print(f"  五、成本影响分析")
    print(f"{'=' * 80}")
    print(f"\n  币安普通用户 (VIP0) 现货费率:")
    print(f"    Maker/Taker:    0.10%")
    print(f"    + BNB 折扣:     0.075% (25% off)")
    print(f"    滑点假设:       10bps (加密货币日线)")
    print(f"    单边总成本:     ~0.20%")
    print(f"    往返总成本:     ~0.40%")
    print(f"\n  高频交易策略 (如 RSI(9,...)) 受成本影响最大")
    print(f"  低频策略 (如 MA(50,200), DriftRegime) 更适合高成本环境")

    print(f"\n{'=' * 80}")
    print(f"  报告生成完毕 · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")
