#!/usr/bin/env python3
"""
==========================================================================
  实时信号分析 · 多策略综合研判 · 买入/持有/卖出建议
==========================================================================

基于回测验证表现最优的策略指标，对 BTC/ETH/XRP/SOL 当前状态进行全面分析。
"""

import numpy as np
import pandas as pd
import yfinance as yf
import sys, os, warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# =====================================================================
#  指标计算函数
# =====================================================================

def calc_sma(close, period):
    return pd.Series(close).rolling(period).mean().values

def calc_ema(close, period):
    return pd.Series(close).ewm(span=period, adjust=False).mean().values

def calc_rsi(close, period):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1.0/period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1.0/period, adjust=False).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    return 100.0 - 100.0 / (1.0 + rs)

def calc_macd(close, fast, slow, signal):
    ema_f = calc_ema(close, fast)
    ema_s = calc_ema(close, slow)
    macd_line = ema_f - ema_s
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_atr(high, low, close, period):
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = pd.Series(tr).ewm(alpha=1.0/period, adjust=False).mean().values
    return atr

def calc_bollinger(close, period, num_std):
    sma = calc_sma(close, period)
    std = pd.Series(close).rolling(period).std().values
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower

def calc_kama(close, er_period=10, fast=2, slow=30):
    n = len(close)
    fast_c = 2.0 / (fast + 1.0)
    slow_c = 2.0 / (slow + 1.0)
    kama = np.full(n, np.nan)
    if n < er_period:
        return kama
    kama[er_period - 1] = close[er_period - 1]
    for i in range(er_period, n):
        direction = abs(close[i] - close[i - er_period])
        volatility = sum(abs(close[i-j+1] - close[i-j]) for j in range(1, er_period+1))
        er = direction / volatility if volatility > 0 else 0.0
        sc = (er * (fast_c - slow_c) + slow_c) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    return kama

def calc_zscore(close, lookback):
    sma = calc_sma(close, lookback)
    std = pd.Series(close).rolling(lookback).std().values
    z = np.where(std > 0, (close - sma) / std, 0.0)
    return z

def calc_momentum_proximity(high, period):
    """计算价格距离 N 日最高价的接近度"""
    roll_high = pd.Series(high).rolling(period).max().values
    proximity = np.where(roll_high > 0, high / roll_high, 0.0)
    return proximity, roll_high

def calc_drift_ratio(close, lookback):
    """计算上涨天数比例"""
    n = len(close)
    ratio = np.full(n, np.nan)
    for i in range(lookback, n):
        up = sum(1 for j in range(1, lookback+1) if close[i-j+1] > close[i-j])
        ratio[i] = up / lookback
    return ratio


# =====================================================================
#  信号判断
# =====================================================================

def analyze_symbol(symbol, ticker, days=365):
    """对单个标的进行全面技术分析"""
    df = yf.download(ticker, period=f"{days}d", progress=False)
    if hasattr(df.columns, 'levels') or (len(df.columns) > 0 and isinstance(df.columns[0], tuple)):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    n = len(close)
    last = close[-1]
    prev = close[-2] if n > 1 else last

    result = {
        "symbol": symbol,
        "price": last,
        "date": str(df["date"].iloc[-1].date()),
        "change_1d_pct": round((last / prev - 1) * 100, 2),
    }

    # --- 1. Moving Averages ---
    ma5 = calc_sma(close, 5)
    ma9 = calc_sma(close, 9)
    ma20 = calc_sma(close, 20)
    ma21 = calc_sma(close, 21)
    ma50 = calc_sma(close, 50)
    ma200 = calc_sma(close, 200)

    result["ma"] = {
        "MA5": round(ma5[-1], 2),
        "MA9": round(ma9[-1], 2),
        "MA20": round(ma20[-1], 2),
        "MA50": round(ma50[-1], 2),
        "MA200": round(ma200[-1], 2) if not np.isnan(ma200[-1]) else "N/A",
        "price_vs_MA20": "上方" if last > ma20[-1] else "下方",
        "price_vs_MA50": "上方" if last > ma50[-1] else "下方",
        "price_vs_MA200": ("上方" if last > ma200[-1] else "下方") if not np.isnan(ma200[-1]) else "N/A",
        "MA5_vs_MA20": "金叉" if ma5[-1] > ma20[-1] else "死叉",
        "MA9_vs_MA21": "金叉" if ma9[-1] > ma21[-1] else "死叉",
        "MA50_vs_MA200": ("金叉" if ma50[-1] > ma200[-1] else "死叉") if not np.isnan(ma200[-1]) else "N/A",
    }

    # MA crossover signal
    ma_signal_short = 1 if (ma5[-2] <= ma20[-2] and ma5[-1] > ma20[-1]) else (-1 if (ma5[-2] >= ma20[-2] and ma5[-1] < ma20[-1]) else 0)
    ma_signal_mid = 1 if (ma9[-2] <= ma21[-2] and ma9[-1] > ma21[-1]) else (-1 if (ma9[-2] >= ma21[-2] and ma9[-1] < ma21[-1]) else 0)

    # --- 2. RSI ---
    rsi9 = calc_rsi(close, 9)
    rsi14 = calc_rsi(close, 14)
    rsi28 = calc_rsi(close, 28)

    result["rsi"] = {
        "RSI(9)": round(rsi9[-1], 2),
        "RSI(14)": round(rsi14[-1], 2),
        "RSI(28)": round(rsi28[-1], 2),
        "RSI9_signal": "超卖(<30)" if rsi9[-1] < 30 else ("超买(>70)" if rsi9[-1] > 70 else "中性"),
        "RSI14_signal": "超卖(<30)" if rsi14[-1] < 30 else ("超买(>70)" if rsi14[-1] > 70 else "中性"),
    }

    # Best RSI params from backtest: RSI(9, os=15, ob=80)
    rsi_buy = rsi9[-1] < 15
    rsi_sell = rsi9[-1] > 80

    # --- 3. MACD ---
    macd1, sig1, hist1 = calc_macd(close, 12, 26, 9)
    macd2, sig2, hist2 = calc_macd(close, 14, 76, 50)
    macd3, sig3, hist3 = calc_macd(close, 16, 18, 16)

    result["macd"] = {
        "MACD(12,26,9)": {"macd": round(macd1[-1], 2), "signal": round(sig1[-1], 2), "hist": round(hist1[-1], 2),
                          "cross": "金叉" if (macd1[-2] <= sig1[-2] and macd1[-1] > sig1[-1]) else ("死叉" if (macd1[-2] >= sig1[-2] and macd1[-1] < sig1[-1]) else "无交叉")},
        "MACD(14,76,50)": {"macd": round(macd2[-1], 2), "signal": round(sig2[-1], 2), "hist": round(hist2[-1], 2),
                           "cross": "金叉" if (macd2[-2] <= sig2[-2] and macd2[-1] > sig2[-1]) else ("死叉" if (macd2[-2] >= sig2[-2] and macd2[-1] < sig2[-1]) else "无交叉"),
                           "trend": "多头" if macd2[-1] > sig2[-1] else "空头"},
        "MACD(16,18,16)": {"macd": round(macd3[-1], 2), "signal": round(sig3[-1], 2), "hist": round(hist3[-1], 2),
                           "cross": "金叉" if (macd3[-2] <= sig3[-2] and macd3[-1] > sig3[-1]) else ("死叉" if (macd3[-2] >= sig3[-2] and macd3[-1] < sig3[-1]) else "无交叉"),
                           "trend": "多头" if macd3[-1] > sig3[-1] else "空头"},
    }

    # MACD signals
    macd_buy_1476 = macd2[-2] <= sig2[-2] and macd2[-1] > sig2[-1]
    macd_sell_1476 = macd2[-2] >= sig2[-2] and macd2[-1] < sig2[-1]
    macd_bull_1476 = macd2[-1] > sig2[-1]
    macd_buy_1618 = macd3[-2] <= sig3[-2] and macd3[-1] > sig3[-1]
    macd_sell_1618 = macd3[-2] >= sig3[-2] and macd3[-1] < sig3[-1]
    macd_bull_1618 = macd3[-1] > sig3[-1]

    # --- 4. ATR & Volatility ---
    atr14 = calc_atr(high, low, close, 14)
    atr20 = calc_atr(high, low, close, 20)
    vol_ratio = atr14[-1] / last * 100 if last > 0 else 0

    result["volatility"] = {
        "ATR(14)": round(atr14[-1], 2),
        "ATR(20)": round(atr20[-1], 2),
        "ATR/Price %": round(vol_ratio, 2),
        "regime": "高波动" if vol_ratio > 3.0 else ("中波动" if vol_ratio > 1.5 else "低波动"),
    }

    # --- 5. Bollinger Bands ---
    bb_mid, bb_upper, bb_lower = calc_bollinger(close, 20, 2.0)
    bb_pct = (last - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) * 100 if (bb_upper[-1] - bb_lower[-1]) > 0 else 50

    result["bollinger"] = {
        "upper": round(bb_upper[-1], 2),
        "middle": round(bb_mid[-1], 2),
        "lower": round(bb_lower[-1], 2),
        "percent_b": round(bb_pct, 1),
        "position": "超买(>上轨)" if last > bb_upper[-1] else ("超卖(<下轨)" if last < bb_lower[-1] else "通道内"),
    }

    # --- 6. Z-Score ---
    z35 = calc_zscore(close, 35)
    z85 = calc_zscore(close, 85)

    result["zscore"] = {
        "Z(35)": round(z35[-1], 2),
        "Z(85)": round(z85[-1], 2) if not np.isnan(z85[-1]) else "N/A",
        "signal_35": "超卖" if z35[-1] < -2.5 else ("超买" if z35[-1] > 2.5 else "中性"),
    }

    # --- 7. Momentum Breakout (最优策略) ---
    prox20, rh20 = calc_momentum_proximity(high, 20)
    prox40, rh40 = calc_momentum_proximity(high, 40)

    result["momentum"] = {
        "20日最高价": round(rh20[-1], 2),
        "距20日高点%": round((1 - prox20[-1]) * 100, 2),
        "40日最高价": round(rh40[-1], 2),
        "距40日高点%": round((1 - prox40[-1]) * 100, 2),
        "MomBreakout(20,3%)_signal": "买入" if prox20[-1] >= 0.97 else "等待",
        "MomBreakout(40,3%)_signal": "买入" if prox40[-1] >= 0.97 else "等待",
    }

    # ATR trailing stop (for those in position)
    trail_stop_20 = last - 1.5 * atr14[-1]
    trail_stop_40 = last - 2.0 * atr14[-1]
    result["momentum"]["ATR_trail_stop_1.5"] = round(trail_stop_20, 2)
    result["momentum"]["ATR_trail_stop_2.0"] = round(trail_stop_40, 2)

    # --- 8. Drift Regime ---
    drift15 = calc_drift_ratio(close, 15)
    drift10 = calc_drift_ratio(close, 10)

    result["drift"] = {
        "drift_15d": round(drift15[-1], 2) if not np.isnan(drift15[-1]) else "N/A",
        "drift_10d": round(drift10[-1], 2) if not np.isnan(drift10[-1]) else "N/A",
        "signal_15": ("做空信号" if drift15[-1] >= 0.62 else "做多信号") if (not np.isnan(drift15[-1]) and (drift15[-1] >= 0.62 or drift15[-1] <= 0.38)) else "中性",
    }

    # --- 9. KAMA ---
    kama_arr = calc_kama(close, 10, 2, 30)
    kama_now = kama_arr[-1] if not np.isnan(kama_arr[-1]) else None
    kama_prev = kama_arr[-2] if len(kama_arr) > 1 and not np.isnan(kama_arr[-2]) else None

    result["kama"] = {
        "KAMA": round(kama_now, 2) if kama_now else "N/A",
        "trend": ("上行" if kama_now > kama_prev else "下行") if kama_now and kama_prev else "N/A",
        "price_vs_KAMA": ("上方" if last > kama_now else "下方") if kama_now else "N/A",
    }

    # --- 10. Volume analysis ---
    vol_ma20 = calc_sma(volume, 20)
    vol_ratio_now = volume[-1] / vol_ma20[-1] if vol_ma20[-1] > 0 else 1.0

    result["volume"] = {
        "today_vol": f"{volume[-1]:,.0f}",
        "20d_avg_vol": f"{vol_ma20[-1]:,.0f}",
        "vol_ratio": round(vol_ratio_now, 2),
        "signal": "放量" if vol_ratio_now > 1.5 else ("缩量" if vol_ratio_now < 0.5 else "正常"),
    }

    # --- 11. Recent performance ---
    result["recent"] = {
        "1d_return": round((close[-1]/close[-2]-1)*100, 2) if n > 1 else 0,
        "7d_return": round((close[-1]/close[-8]-1)*100, 2) if n > 7 else 0,
        "30d_return": round((close[-1]/close[-31]-1)*100, 2) if n > 30 else 0,
        "90d_return": round((close[-1]/close[-91]-1)*100, 2) if n > 90 else 0,
    }

    # Max drawdown from ATH
    ath = np.max(close)
    dd_from_ath = (last / ath - 1) * 100
    result["recent"]["drawdown_from_ATH"] = round(dd_from_ath, 2)
    result["recent"]["ATH"] = round(ath, 2)

    # ===================================================================
    #  综合评分 (Composite Signal)
    # ===================================================================
    # 基于回测验证的策略权重
    # MomBreakout: 最高 Sharpe (1.10), 权重 25%
    # MACD(14,76,50): Sharpe 0.97, 权重 20%
    # MACD(16,18,16): Sharpe 0.85, 权重 15%
    # MA crossover: Sharpe 0.74, 权重 15%
    # RSI: Sharpe 0.68, 权重 10%
    # Bollinger/ZScore/KAMA: 辅助, 权重各 5%

    score = 0.0  # -100 到 +100

    # MomBreakout signal (25%)
    if prox20[-1] >= 0.97:
        score += 25  # 接近新高, 强动量
    elif prox20[-1] >= 0.95:
        score += 15
    elif prox20[-1] < 0.85:
        score -= 10  # 远离新高

    # MACD(14,76,50) (20%)
    if macd_buy_1476:
        score += 20
    elif macd_sell_1476:
        score -= 20
    elif macd_bull_1476:
        score += 10
    else:
        score -= 10

    # MACD(16,18,16) (15%)
    if macd_buy_1618:
        score += 15
    elif macd_sell_1618:
        score -= 15
    elif macd_bull_1618:
        score += 7.5
    else:
        score -= 7.5

    # MA crossover (15%)
    if last > ma20[-1] and last > ma50[-1]:
        score += 10
    elif last < ma20[-1] and last < ma50[-1]:
        score -= 10
    if ma_signal_short == 1:
        score += 5
    elif ma_signal_short == -1:
        score -= 5

    # RSI (10%)
    if rsi_buy:
        score += 10
    elif rsi_sell:
        score -= 10
    elif rsi14[-1] < 40:
        score += 3
    elif rsi14[-1] > 60:
        score -= 3

    # Bollinger (5%)
    if last < bb_lower[-1]:
        score += 5  # oversold
    elif last > bb_upper[-1]:
        score -= 5  # overbought

    # Z-Score (5%)
    if z35[-1] < -2.0:
        score += 5
    elif z35[-1] > 2.0:
        score -= 5

    # KAMA trend (5%)
    if kama_now and kama_prev:
        if last > kama_now and kama_now > kama_prev:
            score += 5
        elif last < kama_now and kama_now < kama_prev:
            score -= 5

    # Determine recommendation
    if score >= 30:
        recommendation = "强烈买入"
    elif score >= 15:
        recommendation = "买入"
    elif score >= 5:
        recommendation = "偏多持有"
    elif score >= -5:
        recommendation = "中性观望"
    elif score >= -15:
        recommendation = "偏空减仓"
    elif score >= -30:
        recommendation = "卖出"
    else:
        recommendation = "强烈卖出"

    result["composite"] = {
        "score": round(score, 1),
        "recommendation": recommendation,
        "confidence": "高" if abs(score) >= 30 else ("中" if abs(score) >= 15 else "低"),
    }

    return result


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("  实时信号分析 · 多策略综合研判")
    print(f"  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    symbols = {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "XRP": "XRP-USD",
        "SOL": "SOL-USD",
    }

    all_results = {}
    for sym, ticker in symbols.items():
        print(f"\n正在分析 {sym} ...", flush=True)
        r = analyze_symbol(sym, ticker, days=365)
        all_results[sym] = r

    # ===== Print comprehensive report =====
    print(f"\n\n{'=' * 80}")
    print(f"  综合分析报告 · {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'=' * 80}")

    for sym in ["BTC", "ETH", "XRP", "SOL"]:
        r = all_results[sym]
        print(f"\n{'─' * 80}")
        print(f"  {sym}  当前价格: ${r['price']:,.2f}  (日期: {r['date']})")
        print(f"{'─' * 80}")

        # Recent
        rc = r["recent"]
        print(f"\n  【近期表现】")
        print(f"    1日: {rc['1d_return']:+.2f}%  |  7日: {rc['7d_return']:+.2f}%  |  30日: {rc['30d_return']:+.2f}%  |  90日: {rc['90d_return']:+.2f}%")
        print(f"    历史最高: ${rc['ATH']:,.2f}  |  距ATH: {rc['drawdown_from_ATH']:+.2f}%")

        # Trend
        ma = r["ma"]
        print(f"\n  【趋势分析】")
        print(f"    MA5: ${ma['MA5']:,.2f}  MA20: ${ma['MA20']:,.2f}  MA50: ${ma['MA50']:,.2f}  MA200: {ma['MA200']}")
        print(f"    价格 vs MA20: {ma['price_vs_MA20']}  |  vs MA50: {ma['price_vs_MA50']}  |  vs MA200: {ma['price_vs_MA200']}")
        print(f"    MA5/MA20: {ma['MA5_vs_MA20']}  |  MA9/MA21: {ma['MA9_vs_MA21']}  |  MA50/MA200: {ma['MA50_vs_MA200']}")

        # KAMA
        km = r["kama"]
        print(f"    KAMA: {km['KAMA']}  趋势: {km['trend']}  价格vs KAMA: {km['price_vs_KAMA']}")

        # Momentum
        mm = r["momentum"]
        print(f"\n  【动量分析】(回测最优策略)")
        print(f"    20日最高: ${mm['20日最高价']:,.2f}  距20日高点: {mm['距20日高点%']:.2f}%")
        print(f"    40日最高: ${mm['40日最高价']:,.2f}  距40日高点: {mm['距40日高点%']:.2f}%")
        print(f"    MomBreakout(20,3%): {mm['MomBreakout(20,3%)_signal']}")
        print(f"    MomBreakout(40,3%): {mm['MomBreakout(40,3%)_signal']}")
        print(f"    ATR跟踪止损(1.5x): ${mm['ATR_trail_stop_1.5']:,.2f}  |  (2.0x): ${mm['ATR_trail_stop_2.0']:,.2f}")

        # MACD
        mc = r["macd"]
        print(f"\n  【MACD 分析】")
        for k, v in mc.items():
            parts = f"    {k}: MACD={v['macd']}, Signal={v['signal']}, Hist={v['hist']}"
            if "cross" in v:
                parts += f"  [{v['cross']}]"
            if "trend" in v:
                parts += f"  趋势={v['trend']}"
            print(parts)

        # RSI
        rs = r["rsi"]
        print(f"\n  【RSI 分析】")
        print(f"    RSI(9): {rs['RSI(9)']}  [{rs['RSI9_signal']}]")
        print(f"    RSI(14): {rs['RSI(14)']}  [{rs['RSI14_signal']}]")
        print(f"    RSI(28): {rs['RSI(28)']}")

        # Volatility
        vl = r["volatility"]
        print(f"\n  【波动率】")
        print(f"    ATR(14): ${vl['ATR(14)']:,.2f}  |  ATR/Price: {vl['ATR/Price %']:.2f}%  |  状态: {vl['regime']}")

        # Bollinger
        bb = r["bollinger"]
        print(f"\n  【布林带】")
        print(f"    上轨: ${bb['upper']:,.2f}  |  中轨: ${bb['middle']:,.2f}  |  下轨: ${bb['lower']:,.2f}")
        print(f"    %B: {bb['percent_b']:.1f}%  |  位置: {bb['position']}")

        # Z-Score
        zs = r["zscore"]
        print(f"\n  【Z-Score】")
        print(f"    Z(35): {zs['Z(35)']}  [{zs['signal_35']}]  |  Z(85): {zs['Z(85)']}")

        # Drift
        dr = r["drift"]
        print(f"\n  【漂移状态】")
        print(f"    15日漂移比: {dr['drift_15d']}  |  10日漂移比: {dr['drift_10d']}  |  信号: {dr['signal_15']}")

        # Volume
        vol = r["volume"]
        print(f"\n  【成交量】")
        print(f"    今日: {vol['today_vol']}  |  20日均量: {vol['20d_avg_vol']}  |  比率: {vol['vol_ratio']}x  [{vol['signal']}]")

        # COMPOSITE
        comp = r["composite"]
        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  综合评分: {comp['score']:+.1f} / 100                       │")
        print(f"  │  建议: {comp['recommendation']}  (置信度: {comp['confidence']})          │")
        print(f"  └─────────────────────────────────────────────┘")

    # Summary table
    print(f"\n\n{'=' * 80}")
    print(f"  操作建议汇总")
    print(f"{'=' * 80}")
    print(f"\n  {'标的':>6}  {'价格':>12}  {'评分':>8}  {'建议':>10}  {'置信度':>6}  {'关键信号'}")
    print(f"  {'-'*70}")
    for sym in ["BTC", "ETH", "XRP", "SOL"]:
        r = all_results[sym]
        comp = r["composite"]
        # Key signal
        mm = r["momentum"]
        mc = r["macd"]
        key_sigs = []
        if mm["MomBreakout(20,3%)_signal"] == "买入":
            key_sigs.append("动量突破")
        if mc["MACD(14,76,50)"]["trend"] == "多头":
            key_sigs.append("MACD多头")
        elif mc["MACD(14,76,50)"]["trend"] == "空头":
            key_sigs.append("MACD空头")
        rs = r["rsi"]
        if "超卖" in rs["RSI9_signal"]:
            key_sigs.append("RSI超卖")
        elif "超买" in rs["RSI9_signal"]:
            key_sigs.append("RSI超买")
        ma = r["ma"]
        if ma["MA50_vs_MA200"] == "死叉":
            key_sigs.append("MA死叉")
        elif ma["MA50_vs_MA200"] == "金叉":
            key_sigs.append("MA金叉")

        sig_str = ", ".join(key_sigs) if key_sigs else "无强信号"
        print(f"  {sym:>6}  ${r['price']:>10,.2f}  {comp['score']:>+6.1f}  {comp['recommendation']:>8}  {comp['confidence']:>4}    {sig_str}")

    print(f"\n  分析基于: 最优回测策略权重加权 (MomBreakout 25% + MACD 35% + MA 15% + RSI 10% + 其他 15%)")
    print(f"  风险提示: 技术分析不构成投资建议, 加密货币波动极大, 请控制仓位和止损")
    print(f"\n{'=' * 80}")
