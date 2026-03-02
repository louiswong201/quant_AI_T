#!/usr/bin/env python3
"""
==========================================================================
  Comprehensive 10-Layer Robust Scan  — 21 Strategies × 18 Assets
==========================================================================
Expanded asset universe:
  Crypto  : BTC, ETH, SOL, XRP, DOGE, AVAX, BNB, LINK
  US Stock: AAPL, GOOGL, TSLA, AMZN, NVDA, META, MSFT, MSTR, SPY, QQQ

10-Layer anti-overfitting pipeline (same as V3):
  L1 : Purged Walk-Forward (Train/Val/Test, 6 windows, embargo gap)
  L2 : Multi-Metric Scoring (return / drawdown * trade_count_factor)
  L3 : Minimum Trade Filter (>= 20 trades)
  L4 : Parameter Stability (perturb +/-10%, +/-20%)
  L5 : Cross-Asset Validation
  L6 : Monte Carlo Price Perturbation (30 noisy OHLC paths)
  L7 : OHLC Shuffle (20 paths)
  L8 : Block Bootstrap (20 paths)
  L9 : Deflated Sharpe Ratio
  L10: Composite 8-dim Ranking

Parameter grids are 2-3× denser than V3 for thorough coverage.
Report includes strategy-level deep analysis and application guidance.
"""
import numpy as np
import pandas as pd
import time, sys, os, warnings, math, json
from collections import defaultdict
from scipy import stats as sp_stats
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numba import njit

SB = 1.0005
SS = 0.9995
CM = 0.0015

WF_WINDOWS = [
    (0.35, 0.45, 0.55),
    (0.45, 0.55, 0.65),
    (0.55, 0.65, 0.75),
    (0.65, 0.75, 0.85),
    (0.75, 0.85, 0.95),
    (0.80, 0.90, 1.00),
]
EMBARGO = 5
MIN_TRADES = 20
PERTURB = [0.8, 0.9, 1.1, 1.2]
MC_PATHS = 30
MC_NOISE_STD = 0.002
SHUFFLE_PATHS = 20
BOOTSTRAP_PATHS = 20
BOOTSTRAP_BLOCK = 20

STRAT_NAMES = [
    "MA", "RSI", "MACD", "Drift", "RAMOM", "Turtle", "Bollinger",
    "Keltner", "MultiFactor", "VolRegime", "Connors", "MESA",
    "KAMA", "Donchian", "ZScore", "MomBreak", "RegimeEMA",
    "TFiltRSI", "MomBrkPlus", "DualMom", "Consensus",
]

STRATEGY_DESCRIPTIONS = {
    "MA": {
        "name": "Moving Average Crossover",
        "type": "Trend-Following",
        "logic": "Uses two moving averages (fast/slow). Buy when fast crosses above slow; sell when fast crosses below.",
        "params": "fast_period, slow_period",
        "best_for": "Strong trending markets with low noise",
        "risk": "Whipsaws in sideways markets; lagging entries/exits",
        "frequency": "Low (few trades per year)",
    },
    "RSI": {
        "name": "Relative Strength Index",
        "type": "Mean-Reversion",
        "logic": "Buy when RSI drops below oversold threshold; sell when RSI rises above overbought threshold.",
        "params": "rsi_period, oversold_threshold, overbought_threshold",
        "best_for": "Range-bound markets with clear support/resistance",
        "risk": "Catches falling knives in strong trends; early exits in strong moves",
        "frequency": "Medium",
    },
    "MACD": {
        "name": "MACD Crossover",
        "type": "Trend-Following / Momentum",
        "logic": "Buy when MACD line crosses above signal line; sell on reverse crossover.",
        "params": "fast_ema, slow_ema, signal_period",
        "best_for": "Momentum-driven markets with sustained moves",
        "risk": "Lagging indicator; frequent false signals in choppy markets",
        "frequency": "Medium",
    },
    "Drift": {
        "name": "Drift Regime Detection",
        "type": "Regime / Statistical",
        "logic": "Detects persistent upward drift via consecutive up-days ratio. Enters when drift exceeds threshold.",
        "params": "lookback, drift_threshold, min_up_days",
        "best_for": "Detecting regime changes and persistent momentum",
        "risk": "Slow to detect regime shifts; may hold through corrections",
        "frequency": "Low",
    },
    "RAMOM": {
        "name": "Risk-Adjusted Momentum",
        "type": "Momentum / Risk-Adjusted",
        "logic": "Enters when risk-adjusted momentum (return/volatility) exceeds entry threshold; exits below exit threshold.",
        "params": "lookback, vol_period, entry_threshold, exit_threshold",
        "best_for": "Filtering high-quality momentum signals",
        "risk": "Requires sustained directional moves; poor in choppy markets",
        "frequency": "Medium-Low",
    },
    "Turtle": {
        "name": "Turtle Trading (Donchian Breakout)",
        "type": "Trend-Following / Breakout",
        "logic": "Buy on N-day high breakout; sell on M-day low breakout. ATR-based trailing stop.",
        "params": "entry_period, exit_period, atr_period, atr_multiplier",
        "best_for": "Capturing large trend moves in liquid markets",
        "risk": "High drawdowns; many false breakouts in ranging markets",
        "frequency": "Low",
    },
    "Bollinger": {
        "name": "Bollinger Band Mean-Reversion",
        "type": "Mean-Reversion / Volatility",
        "logic": "Buy when price touches lower Bollinger Band; sell at upper band. Band width adapts to volatility.",
        "params": "period, std_multiplier",
        "best_for": "Range-bound markets with regular volatility patterns",
        "risk": "Fails badly in trending markets; band walks in strong trends",
        "frequency": "Medium-High",
    },
    "Keltner": {
        "name": "Keltner Channel Breakout",
        "type": "Trend-Following / Volatility",
        "logic": "Buy on close above upper Keltner channel; sell below lower channel. ATR-based channel width.",
        "params": "ema_period, atr_period, atr_multiplier",
        "best_for": "Breakout trading with volatility-adjusted thresholds",
        "risk": "False breakouts; requires strong directional conviction",
        "frequency": "Medium-Low",
    },
    "MultiFactor": {
        "name": "Multi-Factor Composite",
        "type": "Multi-Factor / Composite",
        "logic": "Combines RSI, momentum, and MA signals into a weighted composite score. Enters when composite exceeds threshold.",
        "params": "rsi_period, mom_period, ma_period, entry_score, exit_score",
        "best_for": "Diversified signal generation across market conditions",
        "risk": "Parameter sensitivity; composite may dilute strong individual signals",
        "frequency": "Medium",
    },
    "VolRegime": {
        "name": "Volatility Regime Switch",
        "type": "Regime / Adaptive",
        "logic": "Switches between momentum (high-vol) and mean-reversion (low-vol) based on ATR-derived volatility regime.",
        "params": "atr_period, vol_threshold, fast_ma, slow_ma, rsi_oversold, rsi_overbought",
        "best_for": "Markets with clear volatility regime shifts",
        "risk": "Regime detection lag; wrong-footed during transitions",
        "frequency": "Medium",
    },
    "Connors": {
        "name": "Connors RSI",
        "type": "Mean-Reversion / Short-Term",
        "logic": "Short-term RSI with streak component. Enters on extreme RSI readings for quick mean-reversion trades.",
        "params": "rsi_period, streak_period, percentile_period, buy_threshold, sell_threshold",
        "best_for": "Short-term mean-reversion in liquid, stable markets",
        "risk": "High trade frequency leads to high cost drag; fails in trending markets",
        "frequency": "High",
    },
    "MESA": {
        "name": "MESA Adaptive Moving Average",
        "type": "Adaptive / Trend-Following",
        "logic": "Uses Hilbert Transform to extract instantaneous frequency. MAMA/FAMA adaptive crossover system.",
        "params": "fast_limit, slow_limit",
        "best_for": "Markets with varying cycle lengths; adapts to changing momentum",
        "risk": "Complex internals; sensitive to fast/slow limit tuning",
        "frequency": "Medium",
    },
    "KAMA": {
        "name": "Kaufman Adaptive MA",
        "type": "Adaptive / Trend-Following",
        "logic": "Efficiency Ratio-based adaptive MA that speeds up in trends, slows in noise. ATR trailing stop.",
        "params": "er_period, fast_sc, slow_sc, atr_mult, atr_period",
        "best_for": "Markets alternating between trend and consolidation",
        "risk": "Lag in fast-moving markets; ATR stop may be too wide in calm periods",
        "frequency": "Medium-Low",
    },
    "Donchian": {
        "name": "Donchian Channel",
        "type": "Breakout / Trend-Following",
        "logic": "Buy on N-day high breakout; trailing stop based on ATR. Pure price-action breakout system.",
        "params": "channel_period, atr_period, atr_multiplier",
        "best_for": "Strong directional breakouts in trending markets",
        "risk": "False breakouts; significant drawdowns in choppy markets",
        "frequency": "Low",
    },
    "ZScore": {
        "name": "Z-Score Mean Reversion",
        "type": "Mean-Reversion / Statistical",
        "logic": "Enters when z-score (standardized distance from mean) exceeds threshold. Exits at mean or opposite extreme.",
        "params": "lookback, entry_z, exit_z, stop_z",
        "best_for": "Statistical mean-reversion in stable-distribution assets",
        "risk": "Tail risk; z-score assumptions break during regime changes",
        "frequency": "Medium",
    },
    "MomBreak": {
        "name": "Momentum Breakout",
        "type": "Momentum / Breakout",
        "logic": "Enters when price approaches recent high. ATR-based trailing stop for risk management.",
        "params": "high_period, proximity, atr_period, atr_trail",
        "best_for": "Capturing acceleration in momentum after consolidation",
        "risk": "False breakouts; requires strong follow-through",
        "frequency": "Medium-Low",
    },
    "RegimeEMA": {
        "name": "Regime-Aware EMA",
        "type": "Regime / Adaptive",
        "logic": "Fast EMA crossover in high-vol regime; mean-reversion to slow EMA in low-vol regime.",
        "params": "atr_period, vol_threshold, fast_ema, slow_ema, trend_ema",
        "best_for": "Markets with distinct volatility regimes",
        "risk": "Regime misclassification; transition periods",
        "frequency": "Medium",
    },
    "TFiltRSI": {
        "name": "Trend-Filtered RSI",
        "type": "Mean-Reversion + Trend Filter",
        "logic": "RSI mean-reversion entries only when confirmed by trend MA direction. Reduces counter-trend trades.",
        "params": "rsi_period, trend_ma_period, oversold, overbought",
        "best_for": "Mean-reversion with trend confirmation; reduces whipsaws",
        "risk": "Misses pure mean-reversion opportunities; slower entries",
        "frequency": "Medium",
    },
    "MomBrkPlus": {
        "name": "Enhanced Momentum Breakout",
        "type": "Momentum / Breakout / Confirmation",
        "logic": "Momentum breakout with confirmation bars requirement. Reduces false breakouts via multi-bar confirmation.",
        "params": "high_period, proximity, atr_period, atr_trail, confirm_bars",
        "best_for": "High-conviction breakout entries with confirmation",
        "risk": "Late entries due to confirmation delay; may miss fast moves",
        "frequency": "Low-Medium",
    },
    "DualMom": {
        "name": "Dual Momentum",
        "type": "Momentum / Cross-Timeframe",
        "logic": "Enters when both fast and slow momentum are positive (absolute momentum). Exits when either turns negative.",
        "params": "fast_lookback, slow_lookback",
        "best_for": "Confirmed momentum across multiple timeframes",
        "risk": "Slow to exit; misses short-term reversals",
        "frequency": "Low",
    },
    "Consensus": {
        "name": "Multi-Signal Consensus",
        "type": "Ensemble / Voting",
        "logic": "Aggregates MA crossover + RSI + Momentum signals via majority voting. Requires N-of-3 signals to agree.",
        "params": "fast_ma, slow_ma, rsi_period, mom_lookback, oversold, overbought, vote_threshold",
        "best_for": "Robust signal generation by reducing individual indicator noise",
        "risk": "Conservative; may miss single-indicator opportunities; parameter complexity",
        "frequency": "Medium-Low",
    },
}

# Import all numba kernels from the existing V3 scan
# We re-use the same kernels rather than duplicating code
exec_dir = os.path.dirname(os.path.abspath(__file__))
v3_path = os.path.join(exec_dir, "walk_forward_robust_scan.py")

# =====================================================================
#  Import Strategy Kernels from V3
# =====================================================================

import importlib.util

spec = importlib.util.spec_from_file_location("v3_scan", v3_path)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

# Pull all strategy functions from v3
precompute_all_ma = v3.precompute_all_ma
precompute_all_ema = v3.precompute_all_ema
precompute_all_rsi = v3.precompute_all_rsi
eval_strategy = v3.eval_strategy
eval_strategy_mc = v3.eval_strategy_mc
perturb_ohlc = v3.perturb_ohlc
shuffle_ohlc = v3.shuffle_ohlc
block_bootstrap_ohlc = v3.block_bootstrap_ohlc
deflated_sharpe = v3.deflated_sharpe
PARAM_TYPES = v3.PARAM_TYPES
_score = v3._score

# Import individual backtest kernels for the expanded grid scanner
bt_ma_wf = v3.bt_ma_wf
bt_rsi_wf = v3.bt_rsi_wf
bt_macd_wf = v3.bt_macd_wf
bt_drift_wf = v3.bt_drift_wf
bt_ramom_wf = v3.bt_ramom_wf
bt_turtle_wf = v3.bt_turtle_wf
bt_bollinger_wf = v3.bt_bollinger_wf
bt_keltner_wf = v3.bt_keltner_wf
bt_multifactor_wf = v3.bt_multifactor_wf
bt_volregime_wf = v3.bt_volregime_wf
bt_connors_wf = v3.bt_connors_wf
bt_mesa_wf = v3.bt_mesa_wf
bt_kama_wf = v3.bt_kama_wf
bt_donchian_wf = v3.bt_donchian_wf
bt_zscore_wf = v3.bt_zscore_wf
bt_mombreak_wf = v3.bt_mombreak_wf
bt_regime_ema_wf = v3.bt_regime_ema_wf
bt_tfiltrsi_wf = v3.bt_tfiltrsi_wf
bt_mombrkplus_wf = v3.bt_mombrkplus_wf
bt_dualmom_wf = v3.bt_dualmom_wf
bt_consensus_wf = v3.bt_consensus_wf


# =====================================================================
#  Expanded Parameter Grid Scanner (2-3× denser)
# =====================================================================

def scan_all_expanded(c, o, h, l, mas, emas, rsis, sb, ss, cm):
    """Scan all 21 strategies with expanded parameter grids."""
    R = {}

    # 1. MA Crossover — expanded grid
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for s in range(3,100,2):
        for lg in range(s+5, 200, 3):
            if lg > len(c) - 1: continue
            r,d,n = bt_ma_wf(c,o,mas[s],mas[lg],sb,ss,cm)
            sc=_score(r,d,n); cnt+=1
            if sc>bs: bs=sc; bp=(s,lg); br=r; bd=d; bn=n
    R["MA"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 2. RSI — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for p in range(2,51,2):
        for lo in range(15,45,3):
            for hi in range(55,90,3):
                r,d,n = bt_rsi_wf(c,o,rsis[p],float(lo),float(hi),sb,ss,cm)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(p,lo,hi); br=r; bd=d; bn=n
    R["RSI"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 3. MACD — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for f in range(3,30,2):
        for s_ in range(f+5, 60, 3):
            for sg in [3,5,7,9,12,15]:
                r,d,n = bt_macd_wf(c,o,emas[f],emas[s_],sg,sb,ss,cm)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(f,s_,sg); br=r; bd=d; bn=n
    R["MACD"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 4. Drift — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for lb in range(5,100,3):
        for th in [0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75]:
            for ud in range(1,8):
                r,d,n = bt_drift_wf(c,o,lb,th,ud,sb,ss,cm)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(lb,th,ud); br=r; bd=d; bn=n
    R["Drift"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 5. RAMOM — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for lb in range(5,80,3):
        for vp in range(5,40,3):
            for et in [0.5,1.0,1.5,2.0,2.5,3.0,3.5]:
                for xt in [-0.5,0.0,0.25,0.5,1.0]:
                    r,d,n = bt_ramom_wf(c,o,lb,vp,et,xt,sb,ss,cm)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(lb,vp,et,xt); br=r; bd=d; bn=n
    R["RAMOM"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 6. Turtle — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ep_ in [10,15,20,30,40,50,60,80,100]:
        for xp_ in [5,8,10,15,20,30]:
            if xp_ >= ep_: continue
            for ap in [10,14,20,30]:
                for at_ in [1.0,1.5,2.0,2.5,3.0,3.5,4.0]:
                    r,d,n = bt_turtle_wf(c,o,h,l,ep_,xp_,ap,at_,sb,ss,cm)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(ep_,xp_,ap,at_); br=r; bd=d; bn=n
    R["Turtle"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 7. Bollinger — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for p in range(10,80,3):
        for m in [1.0,1.2,1.5,1.8,2.0,2.2,2.5,2.8,3.0]:
            r,d,n = bt_bollinger_wf(c,o,p,m,sb,ss,cm)
            sc=_score(r,d,n); cnt+=1
            if sc>bs: bs=sc; bp=(p,m); br=r; bd=d; bn=n
    R["Bollinger"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 8. Keltner — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ep_ in range(8,60,3):
        for ap in [7,10,14,20,30]:
            for m in [1.0,1.3,1.5,1.8,2.0,2.3,2.5,3.0]:
                r,d,n = bt_keltner_wf(c,o,h,l,ep_,ap,m,sb,ss,cm)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(ep_,ap,m); br=r; bd=d; bn=n
    R["Keltner"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 9. MultiFactor — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for rp in [7,10,14,20,30]:
        for mp in [5,10,15,20,30,40]:
            for ma_p in [10,20,30,50,80]:
                for es in [0.4,0.5,0.6,0.7,0.8]:
                    for xs in [0.2,0.3,0.35,0.4,0.5]:
                        r,d,n = bt_multifactor_wf(c,o,rp,mp,ma_p,es,xs,sb,ss,cm)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(rp,mp,ma_p,es,xs); br=r; bd=d; bn=n
    R["MultiFactor"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 10. VolRegime — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ap in [10,14,20,30]:
        for vt_ in [0.008,0.012,0.015,0.020,0.025,0.030,0.035]:
            for ms_ in [3,5,8,10,15]:
                for ml_ in [15,20,30,40,50,60]:
                    if ms_>=ml_: continue
                    for lo in [20,25,30,35]:
                        for hi in [65,70,75,80]:
                            r,d,n = bt_volregime_wf(c,o,h,l,ap,vt_,ms_,ml_,14,lo,hi,sb,ss,cm)
                            sc=_score(r,d,n); cnt+=1
                            if sc>bs: bs=sc; bp=(ap,vt_,ms_,ml_,lo,hi); br=r; bd=d; bn=n
    R["VolRegime"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 11. Connors — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for rp in [2,3,4,5,7]:
        for sp in [20,30,50,80,100,150]:
            for pp in [2,3,5,8,10]:
                for lo in [3.0,5.0,8.0,10.0,15.0]:
                    for hi in [85.0,90.0,92.0,95.0,97.0]:
                        r,d,n = bt_connors_wf(c,o,rp,sp,pp,lo,hi,sb,ss,cm)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(rp,sp,pp,lo,hi); br=r; bd=d; bn=n
    R["Connors"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 12. MESA — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for fl_ in [0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        for sl_ in [0.01,0.02,0.03,0.05,0.07,0.10]:
            if sl_ >= fl_:
                continue
            r,d,n = bt_mesa_wf(c,o,fl_,sl_,sb,ss,cm)
            sc=_score(r,d,n); cnt+=1
            if sc>bs: bs=sc; bp=(fl_,sl_); br=r; bd=d; bn=n
    R["MESA"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 13. KAMA — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for er in [5,8,10,15,20,30]:
        for fc in [2,3,4,5]:
            for sc_ in [20,30,40,50,60]:
                for am in [1.0,1.5,2.0,2.5,3.0,3.5]:
                    for ap in [10,14,20,30]:
                        r,d,n = bt_kama_wf(c,o,h,l,er,fc,sc_,am,ap,sb,ss,cm)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(er,fc,sc_,am,ap); br=r; bd=d; bn=n
    R["KAMA"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 14. Donchian — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for cp_ in [10,15,20,30,40,50,60,80,100]:
        for ap in [7,10,14,20,30]:
            for am in [1.0,1.5,2.0,2.5,3.0,3.5,4.0]:
                r,d,n = bt_donchian_wf(c,o,h,l,cp_,ap,am,sb,ss,cm)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(cp_,ap,am); br=r; bd=d; bn=n
    R["Donchian"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 15. ZScore — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for lb in range(8,100,4):
        for ez in [0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0]:
            for xz in [0.0,0.2,0.4,0.5,0.8]:
                for sz in [2.5,3.0,3.5,4.0,4.5,5.0]:
                    r,d,n = bt_zscore_wf(c,o,lb,ez,xz,sz,sb,ss,cm)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(lb,ez,xz,sz); br=r; bd=d; bn=n
    R["ZScore"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 16. MomBreak — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for hp_ in [15,20,30,40,50,60,80,100,150,200]:
        for pp in [0.00,0.01,0.02,0.03,0.05,0.08]:
            for ap in [7,10,14,20,30]:
                for at_ in [0.8,1.0,1.5,2.0,2.5,3.0,3.5,4.0]:
                    r,d,n = bt_mombreak_wf(c,o,h,l,hp_,pp,ap,at_,sb,ss,cm)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(hp_,pp,ap,at_); br=r; bd=d; bn=n
    R["MomBreak"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 17. RegimeEMA — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ap in [10,14,20,30]:
        for vt_ in [0.008,0.012,0.015,0.020,0.025,0.030]:
            for fe in [3,5,8,10,12,15]:
                for se in [15,20,25,30,40,50,60]:
                    if fe>=se: continue
                    for te in [30,50,60,80,100,120]:
                        r,d,n = bt_regime_ema_wf(c,o,h,l,ap,vt_,fe,se,te,sb,ss,cm)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(ap,vt_,fe,se,te); br=r; bd=d; bn=n
    R["RegimeEMA"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 18. TrendFilteredRSI — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for rp in range(2,51,2):
        for tp in [15,20,30,40,50,60,80,100,150,200]:
            if tp > min(200, len(c) - 1): continue
            for os_v in [15,20,25,28,30,33,35]:
                for ob_v in [65,67,70,72,75,80,85]:
                    r,d,n = bt_tfiltrsi_wf(c,o,rsis[rp],mas[tp],float(os_v),float(ob_v),sb,ss,cm)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(rp,tp,os_v,ob_v); br=r; bd=d; bn=n
    R["TFiltRSI"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 19. MomBrkPlus — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for hp_ in [15,20,30,40,60,80,100,150,200]:
        for pp in [0.00,0.01,0.02,0.03,0.05,0.08]:
            for ap in [7,10,14,20,30]:
                for at_ in [1.0,1.5,2.0,2.5,3.0,3.5]:
                    for cb in [1,2,3,4,5]:
                        r,d,n = bt_mombrkplus_wf(c,o,h,l,hp_,pp,ap,at_,cb,sb,ss,cm)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(hp_,pp,ap,at_,cb); br=r; bd=d; bn=n
    R["MomBrkPlus"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 20. DualMomentum — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for fl in [3,5,8,10,15,20,25,30,40,50,60]:
        for sl in [15,20,30,40,50,60,80,100,120,150,200,252]:
            if fl >= sl: continue
            r,d,n = bt_dualmom_wf(c,o,fl,sl,sb,ss,cm)
            sc=_score(r,d,n); cnt+=1
            if sc>bs: bs=sc; bp=(fl,sl); br=r; bd=d; bn=n
    R["DualMom"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 21. Consensus — expanded
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ms in [5,8,10,15,20]:
        for ml in [25,30,40,50,60,80,100,150,200]:
            if ms >= ml or ml > min(200, len(c) - 1): continue
            for rp in [5,7,10,14,21]:
                for mom_lb in [5,10,15,20,30,40,60]:
                    for os_v in [20,25,30,35]:
                        for ob_v in [65,70,75,80]:
                            for vt in [2,3]:
                                r,d,n = bt_consensus_wf(c,o,mas[ms],mas[ml],rsis[rp],mom_lb,
                                                         float(os_v),float(ob_v),vt,sb,ss,cm)
                                sc=_score(r,d,n); cnt+=1
                                if sc>bs: bs=sc; bp=(ms,ml,rp,mom_lb,os_v,ob_v,vt); br=r; bd=d; bn=n
    R["Consensus"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    return R


# =====================================================================
#  Main Execution
# =====================================================================

def main():
    print("=" * 90)
    print("  COMPREHENSIVE 10-LAYER ROBUST SCAN")
    print("  21 Strategies × 18 Assets — Expanded Parameter Grids")
    print("=" * 90)

    CRYPTO = ["BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "BNB", "LINK"]
    STOCKS = ["AAPL", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "MSFT", "MSTR", "SPY", "QQQ"]
    symbols = CRYPTO + STOCKS
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    n_windows = len(WF_WINDOWS)

    # ---- Phase 0: JIT warm-up ----
    print(f"\n[0/10] JIT warm-up ...", end=" ", flush=True)
    t0 = time.time()
    dc = np.random.rand(200).astype(np.float64)*100+100
    dh = dc+np.random.rand(200)*2; dl = dc-np.random.rand(200)*2
    do_ = dc+np.random.rand(200)*0.5
    dm = np.random.rand(200).astype(np.float64)*100+100
    dr = np.random.rand(200).astype(np.float64)*100
    bt_ma_wf(dc,do_,dm,dm,SB,SS,CM)
    bt_rsi_wf(dc,do_,dr,30.0,70.0,SB,SS,CM)
    bt_macd_wf(dc,do_,dm,dm,9,SB,SS,CM)
    bt_drift_wf(dc,do_,20,0.6,5,SB,SS,CM)
    bt_ramom_wf(dc,do_,10,10,2.0,0.5,SB,SS,CM)
    bt_turtle_wf(dc,do_,dh,dl,10,5,14,2.0,SB,SS,CM)
    bt_bollinger_wf(dc,do_,20,2.0,SB,SS,CM)
    bt_keltner_wf(dc,do_,dh,dl,20,14,2.0,SB,SS,CM)
    bt_multifactor_wf(dc,do_,14,20,20,0.6,0.35,SB,SS,CM)
    bt_volregime_wf(dc,do_,dh,dl,14,0.02,5,20,14,30,70,SB,SS,CM)
    bt_connors_wf(dc,do_,2,50,5,10.0,90.0,SB,SS,CM)
    bt_mesa_wf(dc,do_,0.5,0.05,SB,SS,CM)
    bt_kama_wf(dc,do_,dh,dl,10,2,30,2.0,14,SB,SS,CM)
    bt_donchian_wf(dc,do_,dh,dl,20,14,2.0,SB,SS,CM)
    bt_zscore_wf(dc,do_,20,2.0,0.5,4.0,SB,SS,CM)
    bt_mombreak_wf(dc,do_,dh,dl,50,0.02,14,2.0,SB,SS,CM)
    bt_regime_ema_wf(dc,do_,dh,dl,14,0.02,5,20,50,SB,SS,CM)
    bt_tfiltrsi_wf(dc,do_,dr,dm,30.0,70.0,SB,SS,CM)
    bt_mombrkplus_wf(dc,do_,dh,dl,20,0.02,14,2.0,3,SB,SS,CM)
    bt_dualmom_wf(dc,do_,10,50,SB,SS,CM)
    bt_consensus_wf(dc,do_,dm,dm,dr,20,30.0,70.0,2,SB,SS,CM)
    perturb_ohlc(dc,do_,dh,dl,0.002,42)
    shuffle_ohlc(dc,do_,dh,dl,42)
    block_bootstrap_ohlc(dc,do_,dh,dl,20,42)
    eval_strategy_mc("MA", (10, 50), dc, do_, dh, dl, SB, SS, CM)
    print(f"done ({time.time()-t0:.1f}s)")

    # ---- Phase 1: Load data ----
    print(f"[1/10] Loading {len(symbols)} assets ...", flush=True)
    datasets = {}
    for sym in symbols:
        fpath = os.path.join(data_dir, f"{sym}.csv")
        if not os.path.exists(fpath):
            print(f"  {sym}: MISSING — skipped")
            continue
        df = pd.read_csv(fpath, parse_dates=["date"])
        c = df["close"].values.astype(np.float64)
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l_ = df["low"].values.astype(np.float64)
        start_d = str(df["date"].iloc[0])[:10]
        end_d = str(df["date"].iloc[-1])[:10]
        datasets[sym] = {"c": c, "o": o, "h": h, "l": l_, "n": len(c),
                         "start": start_d, "end": end_d}
        tag = "CRYPTO" if sym in CRYPTO else "STOCK"
        print(f"  {sym:>5} [{tag:>6}]: {len(c):>5} bars  ({start_d} → {end_d})")

    active_symbols = [s for s in symbols if s in datasets]
    active_crypto = [s for s in CRYPTO if s in datasets]
    active_stocks = [s for s in STOCKS if s in datasets]

    # ---- Phase 2: Purged Walk-Forward Scan (Layers 1+2+3) ----
    print(f"\n[2/10] Purged Walk-Forward Scan ({n_windows} windows × {len(active_symbols)} assets, "
          f"embargo={EMBARGO} bars) ...\n", flush=True)

    wf = {sym: {sn: [] for sn in STRAT_NAMES} for sym in active_symbols}
    best_params = {sym: {sn: None for sn in STRAT_NAMES} for sym in active_symbols}
    combo_counts = {sn: 0 for sn in STRAT_NAMES}
    total_combos = 0
    grand_t0 = time.time()

    for sym in active_symbols:
        D = datasets[sym]
        c, o, h, l_ = D["c"], D["o"], D["h"], D["l"]
        n = D["n"]
        print(f"  {sym} ({n} bars):")

        for wi, (tr_pct, va_pct, te_pct) in enumerate(WF_WINDOWS):
            tr_end = int(n * tr_pct)
            va_start = min(tr_end + EMBARGO, int(n * va_pct))
            va_end = int(n * va_pct)
            te_end = int(n * te_pct)
            if te_end > n: te_end = n
            if va_end > te_end: va_end = te_end
            if va_start >= va_end: va_start = va_end

            c_tr, o_tr = c[:tr_end], o[:tr_end]
            h_tr, l_tr = h[:tr_end], l_[:tr_end]
            c_va, o_va = c[va_start:va_end], o[va_start:va_end]
            h_va, l_va = h[va_start:va_end], l_[va_start:va_end]
            c_te, o_te = c[va_end:te_end], o[va_end:te_end]
            h_te, l_te = h[va_end:te_end], l_[va_end:te_end]

            t_w = time.time()
            mas_tr = precompute_all_ma(c_tr, 200)
            emas_tr = precompute_all_ema(c_tr, 200)
            rsis_tr = precompute_all_rsi(c_tr, 200)

            results = scan_all_expanded(c_tr, o_tr, h_tr, l_tr, mas_tr, emas_tr, rsis_tr, SB, SS, CM)

            mas_va = precompute_all_ma(c_va, 200)
            emas_va = precompute_all_ema(c_va, 200)
            rsis_va = precompute_all_rsi(c_va, 200)
            mas_te = precompute_all_ma(c_te, 200)
            emas_te = precompute_all_ema(c_te, 200)
            rsis_te = precompute_all_rsi(c_te, 200)

            w_combos = 0
            for sn in STRAT_NAMES:
                res = results[sn]
                w_combos += res["cnt"]
                combo_counts[sn] = max(combo_counts[sn], res["cnt"])
                va_ret, va_dd, va_nt = eval_strategy(
                    sn, res["params"], c_va, o_va, h_va, l_va,
                    mas_va, emas_va, rsis_va, SB, SS, CM
                )
                te_ret, te_dd, te_nt = eval_strategy(
                    sn, res["params"], c_te, o_te, h_te, l_te,
                    mas_te, emas_te, rsis_te, SB, SS, CM
                )
                wf[sym][sn].append({
                    "params": res["params"],
                    "train_score": res["score"],
                    "train_ret": res["ret"],
                    "train_dd": res["dd"],
                    "train_nt": res["nt"],
                    "val_ret": va_ret, "val_dd": va_dd, "val_nt": va_nt,
                    "test_ret": te_ret, "test_dd": te_dd, "test_nt": te_nt,
                    "gen_gap": va_ret - te_ret,
                })
                if wi == n_windows - 1:
                    best_params[sym][sn] = res["params"]

            total_combos += w_combos
            elapsed_w = time.time() - t_w
            print(f"    W{wi+1}: train[0:{tr_end}] val[{va_start}:{va_end}] "
                  f"test[{va_end}:{te_end}]  {w_combos:,} combos  {elapsed_w:.1f}s", flush=True)

    wf_elapsed = time.time() - grand_t0
    print(f"\n  Walk-Forward total: {total_combos:,} combos in {wf_elapsed:.1f}s "
          f"({total_combos/wf_elapsed:,.0f}/s)")

    # Compute WFE and gen gap
    wfe = {sym: {} for sym in active_symbols}
    gen_gap = {sym: {} for sym in active_symbols}
    for sym in active_symbols:
        for sn in STRAT_NAMES:
            oos_rets = [w["test_ret"] for w in wf[sym][sn]]
            wfe[sym][sn] = np.mean(oos_rets) if oos_rets else 0.0
            gaps = [abs(w["gen_gap"]) for w in wf[sym][sn]]
            gen_gap[sym][sn] = np.mean(gaps) if gaps else 99.0

    # ---- Phase 3: Parameter Stability (Layer 4) ----
    print(f"\n[3/10] Parameter Stability Analysis ...", flush=True)
    stability = {sym: {} for sym in active_symbols}
    for sym in active_symbols:
        D = datasets[sym]; c, o, h, l_ = D["c"], D["o"], D["h"], D["l"]
        mas = precompute_all_ma(c, 200)
        emas = precompute_all_ema(c, 200)
        rsis = precompute_all_rsi(c, 200)
        for sn in STRAT_NAMES:
            params = best_params[sym][sn]
            if params is None: stability[sym][sn] = 0.0; continue
            returns = []
            base_r, _, _ = eval_strategy(sn, params, c, o, h, l_, mas, emas, rsis, SB, SS, CM)
            returns.append(base_r)
            ptypes = PARAM_TYPES.get(sn, [])
            for pi in range(len(params)):
                for factor in PERTURB:
                    pv = params[pi] * factor
                    if pi < len(ptypes) and ptypes[pi] == int: pv = max(1, int(round(pv)))
                    new_p = list(params); new_p[pi] = pv
                    r, _, _ = eval_strategy(sn, tuple(new_p), c, o, h, l_, mas, emas, rsis, SB, SS, CM)
                    returns.append(r)
            mean_r = np.mean(returns); std_r = np.std(returns)
            stab = 1.0 - std_r / abs(mean_r) if abs(mean_r) > 1e-8 else 0.0
            stability[sym][sn] = max(0.0, min(1.0, stab))
    print("  done.")

    # ---- Phase 4: Monte Carlo Price Perturbation (Layer 6) ----
    print(f"[4/10] Monte Carlo ({MC_PATHS} paths) ...", flush=True)
    mc_results = {sym: {} for sym in active_symbols}
    for sym in active_symbols:
        D = datasets[sym]; c, o, h, l_ = D["c"], D["o"], D["h"], D["l"]
        for sn in STRAT_NAMES:
            params = best_params[sym][sn]
            if params is None:
                mc_results[sym][sn] = {"profitable": 0.0, "stability": 0.0, "mean_ret": 0.0}; continue
            mc_rets = []
            for seed in range(MC_PATHS):
                cp, op, hp, lp = perturb_ohlc(c, o, h, l_, MC_NOISE_STD, seed + 1000)
                r, _, _ = eval_strategy_mc(sn, params, cp, op, hp, lp, SB, SS, CM)
                mc_rets.append(r)
            mc_arr = np.array(mc_rets)
            mc_mean = np.mean(mc_arr); mc_std = np.std(mc_arr)
            mc_prof = float(np.sum(mc_arr > 0)) / len(mc_arr)
            mc_stab = max(0.0, min(1.0, 1.0 - mc_std / abs(mc_mean))) if abs(mc_mean) > 1e-8 else 0.0
            mc_results[sym][sn] = {"profitable": mc_prof, "stability": mc_stab, "mean_ret": mc_mean}
    print("  done.")

    # ---- Phase 5: OHLC Shuffle (Layer 7) ----
    print(f"[5/10] OHLC Shuffle ({SHUFFLE_PATHS} paths) ...", flush=True)
    shuffle_results = {sym: {} for sym in active_symbols}
    for sym in active_symbols:
        D = datasets[sym]; c, o, h, l_ = D["c"], D["o"], D["h"], D["l"]
        for sn in STRAT_NAMES:
            params = best_params[sym][sn]
            if params is None:
                shuffle_results[sym][sn] = {"profitable": 0.0, "stability": 0.0, "mean_ret": 0.0}; continue
            sh_rets = []
            for seed in range(SHUFFLE_PATHS):
                cp, op, hp, lp = shuffle_ohlc(c, o, h, l_, seed + 5000)
                r, _, _ = eval_strategy_mc(sn, params, cp, op, hp, lp, SB, SS, CM)
                sh_rets.append(r)
            sh_arr = np.array(sh_rets); sh_mean = np.mean(sh_arr); sh_std = np.std(sh_arr)
            sh_prof = float(np.sum(sh_arr > 0)) / len(sh_arr)
            sh_stab = max(0.0, min(1.0, 1.0 - sh_std / abs(sh_mean))) if abs(sh_mean) > 1e-8 else 0.0
            shuffle_results[sym][sn] = {"profitable": sh_prof, "stability": sh_stab, "mean_ret": sh_mean}
    print("  done.")

    # ---- Phase 6: Block Bootstrap (Layer 8) ----
    print(f"[6/10] Block Bootstrap ({BOOTSTRAP_PATHS} paths) ...", flush=True)
    bootstrap_results = {sym: {} for sym in active_symbols}
    for sym in active_symbols:
        D = datasets[sym]; c, o, h, l_ = D["c"], D["o"], D["h"], D["l"]
        for sn in STRAT_NAMES:
            params = best_params[sym][sn]
            if params is None:
                bootstrap_results[sym][sn] = {"profitable": 0.0, "stability": 0.0, "mean_ret": 0.0}; continue
            bs_rets = []
            for seed in range(BOOTSTRAP_PATHS):
                cp, op, hp, lp = block_bootstrap_ohlc(c, o, h, l_, BOOTSTRAP_BLOCK, seed + 9000)
                r, _, _ = eval_strategy_mc(sn, params, cp, op, hp, lp, SB, SS, CM)
                bs_rets.append(r)
            bs_arr = np.array(bs_rets); bs_mean = np.mean(bs_arr); bs_std = np.std(bs_arr)
            bs_prof = float(np.sum(bs_arr > 0)) / len(bs_arr)
            bs_stab = max(0.0, min(1.0, 1.0 - bs_std / abs(bs_mean))) if abs(bs_mean) > 1e-8 else 0.0
            bootstrap_results[sym][sn] = {"profitable": bs_prof, "stability": bs_stab, "mean_ret": bs_mean}
    print("  done.")

    # ---- Phase 7: Deflated Sharpe Ratio (Layer 9) ----
    print(f"[7/10] Deflated Sharpe Ratio ...", flush=True)
    dsr_scores = {sym: {} for sym in active_symbols}
    for sym in active_symbols:
        n_bars = datasets[sym]["n"]
        for sn in STRAT_NAMES:
            oos_rets = [w["test_ret"] for w in wf[sym][sn]]
            if not oos_rets or all(r == 0.0 for r in oos_rets):
                dsr_scores[sym][sn] = 0.0; continue
            ret_arr = np.array(oos_rets) / 100.0
            mu = np.mean(ret_arr); sd = np.std(ret_arr)
            sharpe = mu / sd if sd > 1e-12 else 0.0
            skew = float(sp_stats.skew(ret_arr)) if len(ret_arr) > 2 else 0.0
            kurt = float(sp_stats.kurtosis(ret_arr, fisher=False)) if len(ret_arr) > 3 else 3.0
            n_trials = combo_counts.get(sn, 1000)
            dsr_scores[sym][sn] = deflated_sharpe(sharpe, n_trials, n_bars, skew, kurt)
    print("  done.")

    # ---- Phase 8: Cross-Asset Validation (Layer 5) ----
    print(f"[8/10] Cross-Asset Validation ...", flush=True)
    cross = {sn: {} for sn in STRAT_NAMES}
    precomp_cache = {}
    for sym in active_symbols:
        D = datasets[sym]; c, o, h, l_ = D["c"], D["o"], D["h"], D["l"]
        precomp_cache[sym] = {
            "c": c, "o": o, "h": h, "l": l_,
            "mas": precompute_all_ma(c, 200),
            "emas": precompute_all_ema(c, 200),
            "rsis": precompute_all_rsi(c, 200),
        }
    for sn in STRAT_NAMES:
        cross[sn] = {}
        for train_sym in active_symbols:
            params = best_params[train_sym][sn]
            cross[sn][train_sym] = {}
            for test_sym in active_symbols:
                if test_sym == train_sym: cross[sn][train_sym][test_sym] = None; continue
                pc = precomp_cache[test_sym]
                r, _, _ = eval_strategy(sn, params, pc["c"], pc["o"], pc["h"], pc["l"],
                                        pc["mas"], pc["emas"], pc["rsis"], SB, SS, CM)
                cross[sn][train_sym][test_sym] = r
    print("  done.")

    total_elapsed = time.time() - grand_t0

    # ---- Phase 9: 8-dim Composite Ranking ----
    print(f"\n[9/10] 8-dim Composite Ranking ...", flush=True)
    avg_wfe = {}; avg_gen_gap = {}; avg_stability = {}
    avg_mc = {}; avg_shuffle = {}; avg_bootstrap = {}; avg_dsr = {}; avg_cross = {}

    for sn in STRAT_NAMES:
        avg_wfe[sn] = np.mean([wfe[sym][sn] for sym in active_symbols])
        avg_gen_gap[sn] = np.mean([gen_gap[sym][sn] for sym in active_symbols])
        avg_stability[sn] = np.mean([stability[sym][sn] for sym in active_symbols])
        mc_s = [mc_results[sym][sn]["profitable"] * max(0.0, mc_results[sym][sn]["stability"]) for sym in active_symbols]
        avg_mc[sn] = np.mean(mc_s)
        sh_s = [shuffle_results[sym][sn]["profitable"] * max(0.0, shuffle_results[sym][sn]["stability"]) for sym in active_symbols]
        avg_shuffle[sn] = np.mean(sh_s)
        bs_s = [bootstrap_results[sym][sn]["profitable"] * max(0.0, bootstrap_results[sym][sn]["stability"]) for sym in active_symbols]
        avg_bootstrap[sn] = np.mean(bs_s)
        avg_dsr[sn] = np.mean([dsr_scores[sym][sn] for sym in active_symbols])
        cross_rets = []
        for train_sym in active_symbols:
            for test_sym in active_symbols:
                if test_sym == train_sym: continue
                v = cross[sn][train_sym][test_sym]
                if v is not None: cross_rets.append(v)
        avg_cross[sn] = np.mean(cross_rets) if cross_rets else 0.0

    wfe_ranked = sorted(STRAT_NAMES, key=lambda s: avg_wfe[s], reverse=True)
    gap_ranked = sorted(STRAT_NAMES, key=lambda s: avg_gen_gap[s])
    stab_ranked = sorted(STRAT_NAMES, key=lambda s: avg_stability[s], reverse=True)
    mc_ranked = sorted(STRAT_NAMES, key=lambda s: avg_mc[s], reverse=True)
    shuffle_ranked = sorted(STRAT_NAMES, key=lambda s: avg_shuffle[s], reverse=True)
    bootstrap_ranked = sorted(STRAT_NAMES, key=lambda s: avg_bootstrap[s], reverse=True)
    dsr_ranked = sorted(STRAT_NAMES, key=lambda s: avg_dsr[s], reverse=True)
    cross_ranked = sorted(STRAT_NAMES, key=lambda s: avg_cross[s], reverse=True)

    composite = {}
    for sn in STRAT_NAMES:
        composite[sn] = (wfe_ranked.index(sn) + 1 + gap_ranked.index(sn) + 1
                         + stab_ranked.index(sn) + 1 + mc_ranked.index(sn) + 1
                         + shuffle_ranked.index(sn) + 1 + bootstrap_ranked.index(sn) + 1
                         + dsr_ranked.index(sn) + 1 + cross_ranked.index(sn) + 1)

    final_ranked = sorted(STRAT_NAMES, key=lambda s: composite[s])

    def verdict(sn):
        checks = 0
        if avg_wfe[sn] > 0: checks += 1
        if avg_gen_gap[sn] < 5.0: checks += 1
        if avg_stability[sn] > 0.5: checks += 1
        if avg_mc[sn] > 0.5: checks += 1
        if avg_shuffle[sn] > 0.3: checks += 1
        if avg_bootstrap[sn] > 0.3: checks += 1
        if avg_dsr[sn] > 0.3: checks += 1
        if avg_cross[sn] > 0: checks += 1
        if checks >= 7: return "ROBUST"
        if checks >= 5: return "STRONG"
        if checks >= 3: return "MODERATE"
        return "WEAK"

    # Console output
    print(f"\n{'='*130}")
    print(f"  FINAL 8-DIM COMPOSITE RANKING — {total_combos:,} combos in {total_elapsed:.1f}s ({total_combos/total_elapsed:,.0f}/s)")
    print(f"{'='*130}\n")
    hdr = (f"{'Rank':>4}  {'Strategy':>14}  {'WFE':>7}  {'Gap':>6}  {'Stab':>5}  "
           f"{'MC':>5}  {'Shuf':>5}  {'Boot':>5}  {'DSR':>5}  {'Cross':>7}  {'Score':>5}  {'Verdict':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, sn in enumerate(final_ranked):
        w = avg_wfe[sn]; g = avg_gen_gap[sn]; s = avg_stability[sn]
        m = avg_mc[sn]; sh = avg_shuffle[sn]; bs = avg_bootstrap[sn]
        d = avg_dsr[sn]; cr = avg_cross[sn]; cs = composite[sn]
        print(f"{i+1:4d}  {sn:>14}  {w:+6.1f}%  {g:5.1f}%  {s:5.3f}  "
              f"{m:5.3f}  {sh:5.3f}  {bs:5.3f}  {d:5.3f}  {cr:+6.1f}%  {cs:5d}  {verdict(sn):>8}")

    # ---- Phase 10: Generate Comprehensive Report ----
    print(f"\n[10/10] Generating comprehensive report ...", flush=True)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "docs"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "comprehensive_scan"), exist_ok=True)
    rpt = os.path.join(os.path.dirname(__file__), "..", "docs", "COMPREHENSIVE_ROBUST_REPORT.md")

    L = []
    L.append("# Comprehensive 10-Layer Anti-Overfitting Robust Scan Report\n")
    L.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    L.append("---\n")

    # Executive Summary
    L.append("## Executive Summary\n")
    L.append(f"- **Total backtests**: {total_combos:,}")
    L.append(f"- **Total elapsed time**: {total_elapsed:.1f}s")
    L.append(f"- **Throughput**: {total_combos/total_elapsed:,.0f} backtests/sec")
    L.append(f"- **Strategies**: {len(STRAT_NAMES)} (21 distinct algorithms)")
    L.append(f"- **Assets**: {len(active_symbols)} ({len(active_crypto)} crypto + {len(active_stocks)} stocks)")
    L.append(f"- **Walk-Forward Windows**: {n_windows} purged windows (embargo={EMBARGO} bars)")
    L.append(f"- **Anti-Overfitting Layers**: 10 independent robustness checks")
    L.append(f"- **Execution Model**: Next-Open (signal @ close[i] → fill @ open[i+1])")
    L.append(f"- **Cost Model**: 5bps slippage + 15bps commission per trade\n")

    top3 = final_ranked[:3]
    L.append("### Top 3 Strategies (Overall Composite Ranking)\n")
    for i, sn in enumerate(top3):
        desc = STRATEGY_DESCRIPTIONS[sn]
        L.append(f"**#{i+1} {desc['name']} ({sn})** — Composite Score: {composite[sn]}, Verdict: **{verdict(sn)}**")
        L.append(f"- WFE: {avg_wfe[sn]:+.1f}% | Gen Gap: {avg_gen_gap[sn]:.1f}% | "
                 f"Stability: {avg_stability[sn]:.3f} | MC: {avg_mc[sn]:.3f} | DSR: {avg_dsr[sn]:.3f}")
        L.append(f"- Type: {desc['type']} | Best for: {desc['best_for']}\n")

    L.append("---\n")
    L.append("## Data Coverage\n")
    L.append("| Asset | Type | Bars | Date Range |")
    L.append("|:---|:---|:---:|:---|")
    for sym in active_symbols:
        D = datasets[sym]
        tag = "Crypto" if sym in CRYPTO else "Stock"
        L.append(f"| {sym} | {tag} | {D['n']} | {D['start']} → {D['end']} |")
    L.append("")

    # Framework Performance
    L.append("---\n")
    L.append("## Framework Performance\n")
    L.append("| Metric | Value |")
    L.append("|:---|:---|")
    L.append(f"| Walk-Forward scan time | {wf_elapsed:.1f}s |")
    L.append(f"| Total scan time (all 10 layers) | {total_elapsed:.1f}s |")
    L.append(f"| Backtests per second | {total_combos/total_elapsed:,.0f} |")
    L.append(f"| Total parameter combinations | {total_combos:,} |")
    L.append(f"| Avg combos per strategy-asset-window | {total_combos/(len(active_symbols)*len(STRAT_NAMES)*n_windows):,.0f} |")
    L.append(f"| JIT compilation | Numba @njit(cache=True) |")
    L.append(f"| Data loading | pandas → NumPy arrays |")
    L.append(f"| Anti-overfitting layers | 10 independent checks |")
    L.append("")

    # Composite Ranking
    L.append("---\n")
    L.append("## 1. Final 8-Dimensional Composite Ranking\n")
    L.append("Ranks by sum of ranks across 8 dimensions: WFE + GenGap + Stability + MC + OHLC Shuffle + Block Bootstrap + Deflated Sharpe + Cross-Asset.\n")
    L.append("Lower composite score = better. Each dimension rank is 1-21.\n")
    L.append("| Rank | Strategy | WFE | Gap | Stab | MC | Shuffle | Boot | DSR | Cross | Score | Verdict |")
    L.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")
    for i, sn in enumerate(final_ranked):
        w = avg_wfe[sn]; g = avg_gen_gap[sn]; s = avg_stability[sn]
        m = avg_mc[sn]; sh = avg_shuffle[sn]; bs_ = avg_bootstrap[sn]
        d = avg_dsr[sn]; cr = avg_cross[sn]; cs = composite[sn]
        L.append(f"| {i+1} | **{sn}** | {w:+.1f}% | {g:.1f}% | {s:.3f} | {m:.3f} | {sh:.3f} | {bs_:.3f} | {d:.3f} | {cr:+.1f}% | {cs} | {verdict(sn)} |")
    L.append("")

    # WFE Rankings
    L.append("---\n")
    L.append("## 2. Walk-Forward Efficiency (WFE) — Out-of-Sample Returns\n")
    L.append(f"Average out-of-sample (Test) return across {n_windows} purged walk-forward windows.\n")
    L.append("| Rank | Strategy | Avg WFE | " + " | ".join(active_symbols) + " |")
    L.append("|:---:|:---|:---:|" + ":---:|" * len(active_symbols))
    for i, sn in enumerate(wfe_ranked):
        vals = " | ".join([f"{wfe[sym][sn]:+.1f}%" for sym in active_symbols])
        L.append(f"| {i+1} | {sn} | {avg_wfe[sn]:+.1f}% | {vals} |")
    L.append("")

    # Generalization Gap
    L.append("---\n")
    L.append("## 3. Generalization Gap Analysis\n")
    L.append("Gen Gap = |Val_return - Test_return|, averaged across windows. Lower is better.\n")
    L.append("Since Val and Test have equal width, a large gap indicates overfitting.\n")
    L.append("| Rank | Strategy | Avg Gap | " + " | ".join(active_symbols) + " |")
    L.append("|:---:|:---|:---:|" + ":---:|" * len(active_symbols))
    for i, sn in enumerate(gap_ranked):
        vals = " | ".join([f"{gen_gap[sym][sn]:.1f}%" for sym in active_symbols])
        L.append(f"| {i+1} | {sn} | {avg_gen_gap[sn]:.1f}% | {vals} |")
    L.append("")

    # Parameter Stability
    L.append("---\n")
    L.append("## 4. Parameter Stability Analysis\n")
    L.append("Stability = 1 - (std / |mean|) when params perturbed ±10-20%. Higher = more stable.\n")
    L.append("| Rank | Strategy | Avg Stability | Class | " + " | ".join(active_symbols) + " |")
    L.append("|:---:|:---|:---:|:---|" + ":---:|" * len(active_symbols))
    for i, sn in enumerate(stab_ranked):
        vals = " | ".join([f"{stability[sym][sn]:.3f}" for sym in active_symbols])
        avg_s = avg_stability[sn]
        cls = "STABLE" if avg_s > 0.7 else "MODERATE" if avg_s > 0.4 else "FRAGILE"
        L.append(f"| {i+1} | {sn} | {avg_s:.3f} | {cls} | {vals} |")
    L.append("")

    # MC Perturbation
    L.append("---\n")
    L.append("## 5. Monte Carlo Price Perturbation\n")
    L.append(f"{MC_PATHS} OHLC paths with Gaussian noise σ={MC_NOISE_STD*100:.1f}%.\n")
    L.append("| Rank | Strategy | Avg Score | " + " | ".join([f"{s} %Prof" for s in active_symbols]) + " |")
    L.append("|:---:|:---|:---:|" + ":---:|" * len(active_symbols))
    for i, sn in enumerate(mc_ranked):
        profs = " | ".join([f"{mc_results[sym][sn]['profitable']*100:.0f}%" for sym in active_symbols])
        L.append(f"| {i+1} | {sn} | {avg_mc[sn]:.3f} | {profs} |")
    L.append("")

    # OHLC Shuffle
    L.append("---\n")
    L.append("## 6. OHLC Shuffle Perturbation\n")
    L.append(f"{SHUFFLE_PATHS} paths where O/H/L/C are randomly reassigned per bar.\n")
    L.append("| Rank | Strategy | Avg Score | " + " | ".join([f"{s} %Prof" for s in active_symbols]) + " |")
    L.append("|:---:|:---|:---:|" + ":---:|" * len(active_symbols))
    for i, sn in enumerate(shuffle_ranked):
        profs = " | ".join([f"{shuffle_results[sym][sn]['profitable']*100:.0f}%" for sym in active_symbols])
        L.append(f"| {i+1} | {sn} | {avg_shuffle[sn]:.3f} | {profs} |")
    L.append("")

    # Block Bootstrap
    L.append("---\n")
    L.append("## 7. Block Bootstrap Resampling\n")
    L.append(f"{BOOTSTRAP_PATHS} block-bootstrapped paths (block={BOOTSTRAP_BLOCK} bars).\n")
    L.append("| Rank | Strategy | Avg Score | " + " | ".join([f"{s} %Prof" for s in active_symbols]) + " |")
    L.append("|:---:|:---|:---:|" + ":---:|" * len(active_symbols))
    for i, sn in enumerate(bootstrap_ranked):
        profs = " | ".join([f"{bootstrap_results[sym][sn]['profitable']*100:.0f}%" for sym in active_symbols])
        L.append(f"| {i+1} | {sn} | {avg_bootstrap[sn]:.3f} | {profs} |")
    L.append("")

    # DSR
    L.append("---\n")
    L.append("## 8. Deflated Sharpe Ratio\n")
    L.append("Corrects observed Sharpe for multiple hypothesis testing (Bailey & Lopez de Prado, 2014).\n")
    L.append("| Rank | Strategy | Avg DSR | " + " | ".join(active_symbols) + " |")
    L.append("|:---:|:---|:---:|" + ":---:|" * len(active_symbols))
    for i, sn in enumerate(dsr_ranked):
        vals = " | ".join([f"{dsr_scores[sym][sn]:.3f}" for sym in active_symbols])
        L.append(f"| {i+1} | {sn} | {avg_dsr[sn]:.3f} | {vals} |")
    L.append("")

    # Cross-Asset
    L.append("---\n")
    L.append("## 9. Cross-Asset Validation\n")
    L.append("| Rank | Strategy | Avg Cross-Asset Return |")
    L.append("|:---:|:---|:---:|")
    for i, sn in enumerate(cross_ranked):
        L.append(f"| {i+1} | {sn} | {avg_cross[sn]:+.1f}% |")
    L.append("")

    # Top 5 Cross-Asset Matrices
    L.append("### Cross-Asset Matrices (Top 5 Strategies)\n")
    for sn in cross_ranked[:5]:
        L.append(f"#### {sn} (Avg: {avg_cross[sn]:+.1f}%)\n")
        L.append("| Train \\ Test | " + " | ".join(active_symbols) + " |")
        L.append("|:---|" + ":---:|" * len(active_symbols))
        for train_sym in active_symbols:
            vals = []
            for test_sym in active_symbols:
                v = cross[sn][train_sym][test_sym]
                vals.append("---" if v is None else f"{v:+.1f}%")
            L.append(f"| {train_sym} | {' | '.join(vals)} |")
        L.append("")

    # Best Params per Symbol
    L.append("---\n")
    L.append(f"## 10. Best Parameters per Asset (Last Window)\n")
    for sym in active_symbols:
        tag = "CRYPTO" if sym in CRYPTO else "STOCK"
        L.append(f"### {sym} ({tag}, {datasets[sym]['n']} bars)\n")
        L.append("| Strategy | Best Params | Train Ret | Val Ret | Test Ret | Gap | Stability | MC | Verdict |")
        L.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|")
        for sn in final_ranked:
            p = best_params[sym][sn]
            pstr = str(p) if p else "N/A"
            wlast = wf[sym][sn][-1] if wf[sym][sn] else {}
            tr = wlast.get("train_ret", 0.0)
            va = wlast.get("val_ret", 0.0)
            te = wlast.get("test_ret", 0.0)
            gg = abs(wlast.get("gen_gap", 0.0))
            stab = stability[sym][sn]
            mcs = mc_results[sym][sn]["profitable"] * max(0.0, mc_results[sym][sn]["stability"])
            vrd = verdict(sn)
            L.append(f"| {sn} | {pstr} | {tr:+.1f}% | {va:+.1f}% | {te:+.1f}% | {gg:.1f}% | {stab:.3f} | {mcs:.3f} | {vrd} |")
        L.append("")

    # Walk-Forward Detail
    L.append("---\n")
    L.append("## 11. Walk-Forward Detail by Asset\n")
    for sym in active_symbols:
        L.append(f"### {sym}\n")
        wn_hdr = " | ".join([f"W{i+1}_Val" for i in range(n_windows)] +
                             [f"W{i+1}_Test" for i in range(n_windows)])
        L.append(f"| Strategy | {wn_hdr} | Avg WFE | Avg Gap |")
        L.append("|:---|" + ":---:|" * (n_windows * 2 + 2))
        for sn in wfe_ranked:
            wins = wf[sym][sn]
            vals_v = [f"{w['val_ret']:+.1f}%" if i < len(wins) else "N/A" for i, w in enumerate(wins)]
            vals_t = [f"{w['test_ret']:+.1f}%" if i < len(wins) else "N/A" for i, w in enumerate(wins)]
            while len(vals_v) < n_windows: vals_v.append("N/A")
            while len(vals_t) < n_windows: vals_t.append("N/A")
            L.append(f"| {sn} | {' | '.join(vals_v)} | {' | '.join(vals_t)} "
                     f"| {wfe[sym][sn]:+.1f}% | {gen_gap[sym][sn]:.1f}% |")
        L.append("")

    # ========================================
    # Deep Strategy Analysis + Application Guide
    # ========================================
    L.append("---\n")
    L.append("## 12. Deep Strategy Analysis & Application Guide\n")
    L.append("Each strategy is analyzed across all 8 robustness dimensions with specific "
             "application guidance and risk warnings.\n")

    for rank_i, sn in enumerate(final_ranked):
        desc = STRATEGY_DESCRIPTIONS[sn]
        w = avg_wfe[sn]; g = avg_gen_gap[sn]; s = avg_stability[sn]
        m = avg_mc[sn]; sh = avg_shuffle[sn]; bs_ = avg_bootstrap[sn]
        d = avg_dsr[sn]; cr = avg_cross[sn]; cs = composite[sn]
        vrd = verdict(sn)

        L.append(f"### #{rank_i+1}. {desc['name']} ({sn})\n")
        L.append(f"**Type**: {desc['type']}  ")
        L.append(f"**Composite Score**: {cs} | **Verdict**: **{vrd}**  ")
        L.append(f"**Trading Frequency**: {desc['frequency']}\n")

        L.append("#### Strategy Logic\n")
        L.append(f"{desc['logic']}\n")
        L.append(f"**Parameters**: `{desc['params']}`\n")

        L.append("#### Performance Summary\n")
        L.append("| Metric | Value | Interpretation |")
        L.append("|:---|:---:|:---|")
        wfe_interp = "Positive OOS returns" if w > 0 else "Negative OOS returns — exercise caution"
        L.append(f"| Walk-Forward Efficiency | {w:+.1f}% | {wfe_interp} |")
        gap_interp = "Low overfitting risk" if g < 5 else "Moderate gap" if g < 10 else "High overfitting risk"
        L.append(f"| Generalization Gap | {g:.1f}% | {gap_interp} |")
        stab_interp = "Robust to param changes" if s > 0.7 else "Moderate sensitivity" if s > 0.4 else "Fragile — high param sensitivity"
        L.append(f"| Param Stability | {s:.3f} | {stab_interp} |")
        mc_interp = "Robust to price noise" if m > 0.5 else "Moderate resilience" if m > 0.2 else "Sensitive to price perturbation"
        L.append(f"| MC Robustness | {m:.3f} | {mc_interp} |")
        sh_interp = "Genuine price structure edge" if sh > 0.3 else "May rely on OHLC labelling"
        L.append(f"| OHLC Shuffle | {sh:.3f} | {sh_interp} |")
        bs_interp = "Robust to data resampling" if bs_ > 0.3 else "Sequence-dependent"
        L.append(f"| Block Bootstrap | {bs_:.3f} | {bs_interp} |")
        dsr_interp = "Likely genuine edge" if d > 0.5 else "Possible data-mining" if d > 0.2 else "High data-mining risk"
        L.append(f"| Deflated Sharpe | {d:.3f} | {dsr_interp} |")
        cr_interp = "Generalizes across assets" if cr > 0 else "Asset-specific — limited generalization"
        L.append(f"| Cross-Asset | {cr:+.1f}% | {cr_interp} |")
        L.append("")

        # Best assets for this strategy
        sym_wfe = sorted(active_symbols, key=lambda s: wfe[s][sn], reverse=True)
        best_syms = [s for s in sym_wfe[:5] if wfe[s][sn] > 0]
        worst_syms = [s for s in sym_wfe[-3:] if wfe[s][sn] < 0]

        L.append("#### Best & Worst Assets\n")
        if best_syms:
            best_str = ", ".join([f"{s} ({wfe[s][sn]:+.1f}%)" for s in best_syms])
            L.append(f"- **Best**: {best_str}")
        else:
            L.append("- **Best**: No asset showed positive OOS returns")
        if worst_syms:
            worst_str = ", ".join([f"{s} ({wfe[s][sn]:+.1f}%)" for s in worst_syms])
            L.append(f"- **Worst**: {worst_str}")
        L.append("")

        # Optimal parameters for top assets
        L.append("#### Recommended Parameters (Top Assets)\n")
        L.append("| Asset | Optimal Params | Train | Val | Test | Stability |")
        L.append("|:---|:---|:---:|:---:|:---:|:---:|")
        for sym in best_syms[:5]:
            p = best_params[sym][sn]
            wlast = wf[sym][sn][-1] if wf[sym][sn] else {}
            L.append(f"| {sym} | {p} | {wlast.get('train_ret',0):+.1f}% | "
                     f"{wlast.get('val_ret',0):+.1f}% | {wlast.get('test_ret',0):+.1f}% | {stability[sym][sn]:.3f} |")
        L.append("")

        L.append("#### Application Guide\n")
        L.append(f"- **Best suited for**: {desc['best_for']}")
        L.append(f"- **Key risk**: {desc['risk']}")
        if vrd in ("ROBUST", "STRONG"):
            L.append("- **Recommendation**: Suitable for live deployment with proper risk management. "
                     "Use the recommended parameters and monitor for regime changes.")
            L.append("- **Position sizing**: Start with 1-2% risk per trade. Scale up only after confirming "
                     "live performance matches backtest expectations.")
        elif vrd == "MODERATE":
            L.append("- **Recommendation**: Use cautiously. Consider combining with other strategies in an "
                     "ensemble or as a secondary signal. Monitor closely for degradation.")
            L.append("- **Position sizing**: Conservative — 0.5-1% risk per trade maximum.")
        else:
            L.append("- **Recommendation**: NOT recommended for standalone live trading. The strategy shows "
                     "signs of overfitting or instability across the robustness checks.")
            L.append("- **Position sizing**: Paper trading only until further validation.")
        L.append("")

    # Investment Guidance by Asset Class
    L.append("---\n")
    L.append("## 13. Investment Guidance by Asset\n")

    for sym in active_symbols:
        tag = "CRYPTO" if sym in CRYPTO else "STOCK"
        L.append(f"### {sym} ({tag})\n")

        sym_best = sorted(STRAT_NAMES,
                          key=lambda sn: (wfe[sym][sn] > 0,
                                          stability[sym][sn] > 0.4,
                                          mc_results[sym][sn]["profitable"] > 0.5,
                                          wfe[sym][sn]),
                          reverse=True)

        top_strats = [sn for sn in sym_best[:5] if wfe[sym][sn] > 0]
        L.append("| Rank | Strategy | OOS Return | Stability | MC Robust | Params | Verdict |")
        L.append("|:---:|:---|:---:|:---:|:---:|:---|:---|")
        for ri, sn in enumerate(top_strats[:5] if top_strats else sym_best[:3]):
            p = best_params[sym][sn]
            L.append(f"| {ri+1} | {sn} | {wfe[sym][sn]:+.1f}% | {stability[sym][sn]:.3f} | "
                     f"{mc_results[sym][sn]['profitable']*100:.0f}% | {p} | {verdict(sn)} |")
        L.append("")

        if top_strats:
            L.append(f"**Recommended approach for {sym}**: Use **{top_strats[0]}** as the primary strategy. "
                     f"Parameters: `{best_params[sym][top_strats[0]]}`. "
                     f"Out-of-sample return: {wfe[sym][top_strats[0]]:+.1f}%.\n")
            if len(top_strats) >= 2:
                L.append(f"Consider **{top_strats[1]}** as secondary confirmation. "
                         f"Both strategies showing positive OOS returns with acceptable robustness.\n")
        else:
            L.append(f"**Caution for {sym}**: No strategy showed robustly positive OOS returns. "
                     f"Consider staying flat or using trend-following with tight stops.\n")

    # Methodology
    L.append("---\n")
    L.append("## 14. Methodology\n")
    L.append("### 10-Layer Anti-Overfitting Pipeline\n")
    L.append("| Layer | Name | Purpose |")
    L.append("|:---:|:---|:---|")
    L.append("| 1 | Purged Walk-Forward | Train on historical, validate and test on future. Embargo gap prevents leakage. |")
    L.append("| 2 | Multi-Metric Scoring | Composite score: return / max_drawdown * trade_count_factor. |")
    L.append("| 3 | Minimum Trade Filter | Require >= 20 trades for statistical significance. |")
    L.append("| 4 | Parameter Stability | Perturb all params by +/-10% and +/-20%. Stable strategies survive. |")
    L.append("| 5 | Cross-Asset Validation | Test parameters trained on one asset across all other assets. |")
    L.append("| 6 | Monte Carlo Perturbation | Test on 30 noisy OHLC paths (σ=0.2% Gaussian noise). |")
    L.append("| 7 | OHLC Shuffle | Randomly reassign O/H/L/C roles to test price structure dependence. |")
    L.append("| 8 | Block Bootstrap | Resample contiguous blocks (20 bars) with replacement. |")
    L.append("| 9 | Deflated Sharpe Ratio | Corrects for multiple hypothesis testing (Bailey & Lopez de Prado). |")
    L.append("| 10 | Composite 8-dim Ranking | Sum of ranks across all 8 dimensions. Lower = better. |")
    L.append("")

    L.append("### Walk-Forward Windows\n")
    L.append("```")
    for i, (tr, va, te) in enumerate(WF_WINDOWS):
        L.append(f"Window {i+1}: train [0, {tr*100:.0f}%), "
                 f"[embargo {EMBARGO} bars], "
                 f"val [{tr*100:.0f}%+emb, {va*100:.0f}%), "
                 f"test [{va*100:.0f}%, {te*100:.0f}%)")
    L.append("```\n")

    L.append("### Execution Model\n")
    L.append("- Signal generated at **close[i]**, order filled at **open[i+1]** (next-open execution)")
    L.append("- Slippage: 5 basis points (0.05%)")
    L.append("- Commission: 15 basis points (0.15%) per trade")
    L.append("- These costs approximate Binance (crypto) and IBKR (stocks) retail fees\n")

    L.append("### Verdict Criteria\n")
    L.append("A strategy earns one point for each criterion met:\n")
    L.append("1. WFE > 0% (positive out-of-sample returns)")
    L.append("2. Generalization Gap < 5%")
    L.append("3. Parameter Stability > 0.5")
    L.append("4. Monte Carlo Score > 0.5")
    L.append("5. OHLC Shuffle Score > 0.3")
    L.append("6. Block Bootstrap Score > 0.3")
    L.append("7. Deflated Sharpe > 0.3")
    L.append("8. Cross-Asset Return > 0%\n")
    L.append("- **ROBUST**: >= 7/8 | **STRONG**: 5-6/8 | **MODERATE**: 3-4/8 | **WEAK**: < 3/8\n")

    # Risk Disclaimer
    L.append("---\n")
    L.append("## 15. Risk Disclaimer\n")
    L.append("- Past performance does NOT guarantee future results.")
    L.append("- All backtests use historical data and may not reflect live trading conditions.")
    L.append("- Despite 10 layers of anti-overfitting checks, no backtest can fully eliminate data-mining bias.")
    L.append("- Always use proper risk management: position sizing, stop losses, and portfolio diversification.")
    L.append("- Start with paper trading before deploying capital.")
    L.append("- Crypto markets trade 24/7 with higher volatility than equities. Adjust risk accordingly.")
    L.append("- This report is for educational and research purposes only, NOT investment advice.\n")

    with open(rpt, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n  Report: {rpt}")

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "comprehensive_scan", "summary.csv")
    rows = []
    for sym in active_symbols:
        for sn in STRAT_NAMES:
            oos_rets = [w["test_ret"] for w in wf[sym][sn]]
            val_rets = [w["val_ret"] for w in wf[sym][sn]]
            mcs = mc_results[sym][sn]
            shs = shuffle_results[sym][sn]
            bss = bootstrap_results[sym][sn]
            row = {
                "symbol": sym, "strategy": sn,
                "asset_type": "crypto" if sym in CRYPTO else "stock",
                "wfe": round(wfe[sym][sn], 2),
                "gen_gap": round(gen_gap[sym][sn], 2),
                "stability": round(stability[sym][sn], 3),
                "mc_profitable": round(mcs["profitable"], 3),
                "mc_stability": round(mcs["stability"], 3),
                "mc_score": round(mcs["profitable"] * max(0.0, mcs["stability"]), 3),
                "shuffle_profitable": round(shs["profitable"], 3),
                "shuffle_score": round(shs["profitable"] * max(0.0, shs["stability"]), 3),
                "bootstrap_profitable": round(bss["profitable"], 3),
                "bootstrap_score": round(bss["profitable"] * max(0.0, bss["stability"]), 3),
                "dsr": round(dsr_scores[sym][sn], 3),
                "cross_asset_avg": round(avg_cross[sn], 2),
                "composite_rank": composite[sn],
                "verdict": verdict(sn),
                "best_params": str(best_params[sym][sn]),
            }
            for wi in range(n_windows):
                row[f"w{wi+1}_val"] = round(val_rets[wi], 2) if wi < len(val_rets) else 0
                row[f"w{wi+1}_test"] = round(oos_rets[wi], 2) if wi < len(oos_rets) else 0
            rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    # Save JSON with all results
    json_path = os.path.join(os.path.dirname(__file__), "..", "results", "comprehensive_scan", "full_results.json")
    json_data = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "total_combos": total_combos,
            "elapsed_seconds": round(total_elapsed, 1),
            "throughput": round(total_combos / total_elapsed),
            "n_strategies": len(STRAT_NAMES),
            "n_assets": len(active_symbols),
            "n_windows": n_windows,
            "assets_crypto": active_crypto,
            "assets_stocks": active_stocks,
        },
        "composite_ranking": [{"rank": i+1, "strategy": sn, "score": composite[sn],
                               "verdict": verdict(sn),
                               "wfe": round(avg_wfe[sn], 2),
                               "gen_gap": round(avg_gen_gap[sn], 2),
                               "stability": round(avg_stability[sn], 3),
                               "mc": round(avg_mc[sn], 3),
                               "shuffle": round(avg_shuffle[sn], 3),
                               "bootstrap": round(avg_bootstrap[sn], 3),
                               "dsr": round(avg_dsr[sn], 3),
                               "cross": round(avg_cross[sn], 2)}
                              for i, sn in enumerate(final_ranked)],
        "best_params": {sym: {sn: str(best_params[sym][sn]) for sn in STRAT_NAMES} for sym in active_symbols},
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON: {json_path}")

    print(f"\n  Total: {total_combos:,} backtests in {total_elapsed:.1f}s = {total_combos/total_elapsed:,.0f} combos/sec")
    print(f"\n{'='*90}")
    print("  SCAN COMPLETE")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
