#!/usr/bin/env python3
"""
==========================================================================
  Long/Short Leveraged 10-Layer Robust Scan
==========================================================================
Enhancements over walk_forward_robust_scan.py:
  1. ALL strategies are fully bidirectional (long + short)
  2. Leverage support (crypto 1-100x, stocks 1-4x)
  3. Realistic cost model:
     - Binance Futures (crypto): 0.04% taker, 0.03%/day funding rate
     - IBKR (stocks): 0.05% commission, 1-5% annual borrow for shorts
  4. Liquidation modeling (margin call at 80% equity loss)
  5. Multi-layer anti-overfitting: Walk-Forward + Validation Gate + Deflated Sharpe
  6. Institutional metrics: Annualized Return, Sharpe Proxy, DSR p-value

Cost References:
  Binance (standard user, no VIP):
    Maker: 0.02%, Taker: 0.04%
    Funding: ~0.01% per 8h = 0.03%/day (avg, both longs & shorts pay)
  IBKR (standard user, non-tiered):
    Commission: ~0.005/share => ~0.05% per trade
    Short borrow: TSLA ~1% ann, MSTR ~5% ann (hard to borrow)
    Margin interest: ~6.83% ann (up to $100k)
"""
import numpy as np
import pandas as pd
import time, sys, os, warnings, math
from collections import defaultdict

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from numba import njit
from scipy import stats as sp_stats

from walk_forward_robust_scan import (
    _ema, _rolling_mean, _rolling_std, _atr, _rsi_wilder,
    _score,
    perturb_ohlc, shuffle_ohlc, block_bootstrap_ohlc,
    precompute_all_ma, precompute_all_ema, precompute_all_rsi,
    STRAT_NAMES, PARAM_TYPES,
)

# =====================================================================
#  Cost Model Constants
# =====================================================================

# Binance Futures (crypto, standard user)
CRYPTO_CM = 0.0004          # 0.04% taker fee per side
CRYPTO_SB_BASE = 0.0003     # 3 bps base slippage (1x reference)
CRYPTO_FUNDING_DAILY = 0.0003  # 0.03%/day (~0.01% per 8h x 3)

# IBKR (stocks, standard user)
STOCK_CM = 0.0005           # 0.05% commission per side
STOCK_SB_BASE = 0.0005      # 5 bps base slippage (1x reference)
STOCK_BORROW_RATES = {      # annual borrow rates for shorts
    "TSLA": 0.01,           # 1% (easy to borrow)
    "MSTR": 0.05,           # 5% (harder to borrow, BTC proxy)
    "NVDA": 0.005,          # 0.5% (very liquid)
    "META": 0.005,
    "AAPL": 0.003,
    "GOOGL": 0.003,
    "AMZN": 0.003,
    "MSFT": 0.003,
    "SPY": 0.002,
    "QQQ": 0.002,
}
STOCK_MARGIN_RATE = 0.0683  # 6.83% annual margin interest

CRYPTO_LEVERAGE_GRID = [1, 2, 5, 10, 20, 50]
STOCK_LEVERAGE_GRID = [1, 2, 4]

# --- Realistic constraint parameters ---
# (1) Dynamic slippage: slippage = base * sqrt(lev)
#     Models market impact growing with notional size.
# (2) Position sizing cap: fraction of equity deployed per trade.
#     At 1x use 100%; at higher lev, cap to avoid ruin.
POSITION_FRAC = {1: 1.0, 2: 0.90, 4: 0.70, 5: 0.60,
                 10: 0.40, 20: 0.25, 50: 0.15}
# (3) SL slippage: stop-loss orders suffer extra slippage at high lev.
#     Extra SL slip = base_slip * lev * 0.5  (models thin books at SL)
# (4) Dynamic funding: funding = base * (1 + 0.02*lev)
#     Higher lev = more demand = higher funding premium.

STOP_LOSS_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# Walk-forward config
WF_WINDOWS = [
    (0.35, 0.45, 0.55), (0.45, 0.55, 0.65), (0.55, 0.65, 0.75),
    (0.65, 0.75, 0.85), (0.75, 0.85, 0.95), (0.80, 0.90, 1.00),
]
EMBARGO = 5
MC_PATHS = 20
MC_NOISE_STD = 0.002
SHUFFLE_PATHS = 15
BOOTSTRAP_PATHS = 15
BOOTSTRAP_BLOCK = 20
PERTURB = [0.8, 0.9, 1.1, 1.2]

LS_STRAT_NAMES = [
    "MA", "RSI", "MACD", "Drift", "RAMOM", "Turtle", "Bollinger",
    "Keltner", "MultiFactor", "VolRegime", "MESA",
    "KAMA", "Donchian", "ZScore", "MomBreak", "RegimeEMA",
    "DualMom", "Consensus",
]

LS_PARAM_TYPES = {
    "MA": [int, int],
    "RSI": [int, int, int],
    "MACD": [int, int, int],
    "Drift": [int, float, int],
    "RAMOM": [int, int, float, float],
    "Turtle": [int, int, int, float],
    "Bollinger": [int, float],
    "Keltner": [int, int, float],
    "MultiFactor": [int, int, int, float, float],
    "VolRegime": [int, float, int, int, int, int],
    "MESA": [float, float],
    "KAMA": [int, int, int, float, int],
    "Donchian": [int, int, float],
    "ZScore": [int, float, float, float],
    "MomBreak": [int, float, int, float],
    "RegimeEMA": [int, float, int, int, int],
    "DualMom": [int, int],
    "Consensus": [int, int, int, int, int, int, int],
}


# =====================================================================
#  Enhanced Fill with Leverage + Daily Cost + Liquidation
# =====================================================================

@njit(cache=True)
def _deploy(tr, pfrac):
    """Compute deployed capital: min(tr * pfrac, pfrac).
    The second branch caps deployment at the initial-equity fraction,
    preventing infinite compounding. Only kicks in when tr > 1.0.
    """
    d = tr * pfrac
    if tr > 1.0 and d > pfrac:
        d = pfrac
    return d


@njit(cache=True)
def _fx_lev(pend, pos, ep, oi, tr, sb, ss, cm, lev, daily_cost, pfrac):
    """Fill pending order with leverage + realistic constraints.
    pfrac: fraction of INITIAL equity deployed per trade (fixed-lot model).
    When equity tr > 1.0, profits sit in cash reserve and are NOT reinvested.
    sb/ss already include leverage-adjusted slippage.
    daily_cost already includes leverage-adjusted funding.
    Returns (pos, ep, tr, trades_closed, liquidated).
    """
    # Always charge daily funding when holding a position, even on fill days
    if pos != 0:
        deployed = _deploy(tr, pfrac)
        cost = deployed * daily_cost
        tr -= cost
        if tr < 0.01:
            return 0, 0.0, 0.01, 0, 1

    if pend == 0:
        return pos, ep, tr, 0, 0

    tc = 0; liq = 0
    if abs(pend) >= 2:
        deployed = _deploy(tr, pfrac)
        if pos == 1:
            raw = (oi * ss * (1.0 - cm)) / (ep * (1.0 + cm))
            pnl = (raw - 1.0) * lev
            tr += deployed * pnl
            if tr < 0.01: tr = 0.01
            tc = 1
        elif pos == -1:
            raw = (ep * (1.0 - cm)) / (oi * sb * (1.0 + cm))
            pnl = (raw - 1.0) * lev
            tr += deployed * pnl
            if tr < 0.01: tr = 0.01
            tc = 1
        pos = 0
    if pend == 1 or pend == 3:
        ep = oi * sb; pos = 1
    elif pend == -1 or pend == -3:
        ep = oi * ss; pos = -1
    return pos, ep, tr, tc, liq


@njit(cache=True)
def _sl_exit(pos, ep, tr, ci, sb, ss, cm, lev, sl, pfrac, sl_slip):
    """Execute stop-loss with cost-adjusted PnL check + realistic slippage.
    Uses the same cost-adjusted formula as _mtm_lev for consistency.
    sl_slip: extra fractional loss beyond the SL threshold in fast markets.
    Returns (new_pos, new_ep, new_tr, trade_closed).
    """
    if pos == 0 or ep <= 0:
        return pos, ep, tr, 0
    if pos == 1:
        raw = (ci * ss * (1.0 - cm)) / (ep * (1.0 + cm))
        pnl = (raw - 1.0) * lev
    else:
        raw = (ep * (1.0 - cm)) / (ci * sb * (1.0 + cm))
        pnl = (raw - 1.0) * lev
    if pnl >= -sl:
        return pos, ep, tr, 0
    actual_loss = sl + sl_slip
    deployed = _deploy(tr, pfrac)
    tr -= deployed * actual_loss
    if tr < 0.01:
        tr = 0.01
    return 0, 0.0, tr, 1


@njit(cache=True)
def _mtm_lev(pos, tr, ci, ep, sb, ss, cm, lev, sl, pfrac):
    """Mark-to-market equity with leverage + position sizing.
    Uses fixed-lot deployment; caps floating loss at -sl.
    """
    deployed = _deploy(tr, pfrac)
    if pos == 1 and ep > 0:
        raw = (ci * ss * (1 - cm)) / (ep * (1 + cm))
        pnl = (raw - 1.0) * lev
        pnl = max(pnl, -sl)
        return tr + deployed * pnl
    if pos == -1 and ep > 0:
        raw = (ep * (1 - cm)) / (ci * sb * (1 + cm))
        pnl = (raw - 1.0) * lev
        pnl = max(pnl, -sl)
        return tr + deployed * pnl
    return tr


# =====================================================================
#  Bidirectional Kernels with Leverage
#  Each returns (ret_pct, max_dd_pct, n_trades)
# =====================================================================

@njit(cache=True)
def bt_ma_ls(c, o, ma_s, ma_l, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(1, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        s0=ma_s[i-1]; l0=ma_l[i-1]; s1=ma_s[i]; l1=ma_l[i]
        if s0!=s0 or l0!=l0 or s1!=s1 or l1!=l1: pass
        elif s0<=l0 and s1>l1:
            if pos==0: pend=1
            elif pos==-1: pend=3
        elif s0>=l0 and s1<l1:
            if pos==0: pend=-1
            elif pos==1: pend=-3
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_rsi_ls(c, o, rsi, os_v, ob_v, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(1, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        r=rsi[i]
        if r!=r: pass
        elif r<os_v:
            if pos==0: pend=1
            elif pos==-1: pend=3
        elif r>ob_v:
            if pos==0: pend=-1
            elif pos==1: pend=-3
        elif pos==1 and r>50: pend=2
        elif pos==-1 and r<50: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_macd_ls(c, o, ef, es, sig_span, sb, ss, cm, lev, dc, stop=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); ml=np.empty(n); sl=np.empty(n)
    for i in range(n): ml[i]=ef[i]-es[i]
    k=2.0/(sig_span+1.0); sl[0]=ml[0]
    for i in range(1,n): sl[i]=ml[i]*k+sl[i-1]*(1-k)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(1, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,stop,pfrac,sl_slip); nt+=tc2
        mp=ml[i-1]; sp=sl[i-1]; mc=ml[i]; sc=sl[i]
        if mp!=mp or sp!=sp or mc!=mc or sc!=sc: pass
        elif mp<=sp and mc>sc:
            if pos==0: pend=1
            elif pos==-1: pend=3
        elif mp>=sp and mc<sc:
            if pos==0: pend=-1
            elif pos==1: pend=-3
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,stop,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_drift_ls(c, o, lookback, drift_thr, hold_p, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); pos=0; ep=0.0; tr=1.0; pend=0; hc=0; pk=1.0; mdd=0.0; nt=0
    for i in range(lookback, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; hc=0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        if tc2>0: hc=0
        up=0
        for j in range(1, lookback+1):
            if c[i-j+1]>c[i-j]: up+=1
        ratio=up/lookback
        if pos!=0:
            hc+=1
            if hc>=hold_p: pend=2; hc=0
        if pos==0 and pend==0:
            if ratio>=drift_thr: pend=-1; hc=0
            elif ratio<=1.0-drift_thr: pend=1; hc=0
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_ramom_ls(c, o, mom_p, vol_p, ez, xz, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(mom_p, vol_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        mom=(c[i]/c[i-mom_p])-1.0
        s=0.0; s2=0.0
        for j in range(vol_p):
            r=(c[i-j]/c[i-j-1]-1.0) if i-j>0 else 0.0
            s+=r; s2+=r*r
        m=s/vol_p; vol=np.sqrt(max(1e-20, s2/vol_p-m*m))
        z=mom/vol
        if pos==0:
            if z>ez: pend=1
            elif z<-ez: pend=-1
        elif pos==1 and z<xz: pend=2
        elif pos==-1 and z>-xz: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_turtle_ls(c, o, h, l, entry_p, exit_p, atr_p, atr_stop, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); aa=_atr(h,l,c,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(entry_p, exit_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        eh=-1e18; el=1e18
        for j in range(1, entry_p+1):
            if h[i-j]>eh: eh=h[i-j]
            if l[i-j]<el: el=l[i-j]
        xl=1e18; xh=-1e18
        for j in range(1, exit_p+1):
            if l[i-j]<xl: xl=l[i-j]
            if h[i-j]>xh: xh=h[i-j]
        a=aa[i]
        if a!=a: pass
        else:
            if pos==1:
                if c[i]<ep/sb-atr_stop*a or c[i]<xl: pend=2
            elif pos==-1:
                if c[i]>ep/ss+atr_stop*a or c[i]>xh: pend=2
            if pos==0 and pend==0:
                if c[i]>eh: pend=1
                elif c[i]<el: pend=-1
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_bollinger_ls(c, o, period, num_std, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); ma=_rolling_mean(c,period); sd=_rolling_std(c,period)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(period, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        m=ma[i]; s=sd[i]
        if m!=m or s!=s or s<1e-10: pass
        else:
            u=m+num_std*s; lo=m-num_std*s
            if pos==0:
                if c[i]<lo: pend=1
                elif c[i]>u: pend=-1
            elif pos==1 and c[i]>=m: pend=2
            elif pos==-1 and c[i]<=m: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_keltner_ls(c, o, h, l, ema_p, atr_p, atr_m, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); ea=_ema(c,ema_p); aa=_atr(h,l,c,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(ema_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        e=ea[i]; a=aa[i]
        if e!=e or a!=a: pass
        else:
            if pos==0:
                if c[i]>e+atr_m*a: pend=1
                elif c[i]<e-atr_m*a: pend=-1
            elif pos==1 and c[i]<e: pend=2
            elif pos==-1 and c[i]>e: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_multifactor_ls(c, o, rsi_p, mom_p, vol_p, lt, st, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); rsi=_rsi_wilder(c,rsi_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(rsi_p+1, mom_p, vol_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        r=rsi[i]
        if r!=r: pass
        else:
            rs=(100.0-r)/100.0; mom=(c[i]/c[i-mom_p])-1.0; ms=max(-0.5,min(0.5,mom))+0.5
            s2=0.0
            for j in range(vol_p):
                ret=(c[i-j]/c[i-j-1]-1.0) if i-j>0 else 0.0; s2+=ret*ret
            vs=max(0.0,1.0-np.sqrt(s2/vol_p)*20.0); comp=(rs+ms+vs)/3.0
            if pos==0:
                if comp>lt: pend=1
                elif comp<st: pend=-1
            elif pos==1 and comp<0.5: pend=2
            elif pos==-1 and comp>0.5: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_volregime_ls(c, o, h, l, atr_p, vol_thr, ma_s, ma_l, rsi_p, rsi_os, rsi_ob, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); aa=_atr(h,l,c,atr_p); ra=_rsi_wilder(c,rsi_p)
    ms_a=_rolling_mean(c,ma_s); ml_a=_rolling_mean(c,ma_l)
    pos=0; ep=0.0; tr=1.0; pend=0; mode=0; pk=1.0; mdd=0.0; nt=0
    start=max(atr_p,rsi_p+1,ma_l)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        a=aa[i]
        if a!=a or c[i]<=0: pass
        else:
            hv=a/c[i]>vol_thr
            if hv:
                r=ra[i]
                if r!=r: pass
                else:
                    if pos==0:
                        if r<rsi_os: pend=1; mode=1
                        elif r>rsi_ob: pend=-1; mode=1
                    elif pos==1 and mode==1 and r>50: pend=2
                    elif pos==-1 and mode==1 and r<50: pend=2
            else:
                s_=ms_a[i]; l_=ml_a[i]; s0=ms_a[i-1]; l0=ml_a[i-1]
                if s_!=s_ or l_!=l_ or s0!=s0 or l0!=l0: pass
                else:
                    if pos==0:
                        if s0<=l0 and s_>l_: pend=1; mode=0
                        elif s0>=l0 and s_<l_: pend=-1; mode=0
                    elif pos==1 and mode==0 and s_<l_: pend=2
                    elif pos==-1 and mode==0 and s_>l_: pend=2
            if pos!=0 and pend==0:
                if (mode==0 and hv) or (mode==1 and not hv): pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_mesa_ls(c, o, fl, slow_lim, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<40: return 0.0, 0.0, 0
    smooth=np.zeros(n); det=np.zeros(n); I1=np.zeros(n); Q1=np.zeros(n)
    jI=np.zeros(n); jQ=np.zeros(n); I2=np.zeros(n); Q2=np.zeros(n)
    Re_=np.zeros(n); Im_=np.zeros(n); per=np.zeros(n); sp_=np.zeros(n)
    ph=np.zeros(n); mama=np.zeros(n); fama=np.zeros(n)
    for i in range(min(6,n)):
        mama[i]=c[i]; fama[i]=c[i]; per[i]=6.0; sp_[i]=6.0
    for i in range(6, n):
        smooth[i]=(4*c[i]+3*c[i-1]+2*c[i-2]+c[i-3])/10.0
        adj=0.075*per[i-1]+0.54
        det[i]=(0.0962*smooth[i]+0.5769*smooth[max(0,i-2)]-0.5769*smooth[max(0,i-4)]-0.0962*smooth[max(0,i-6)])*adj
        I1[i]=det[max(0,i-3)]
        Q1[i]=(0.0962*det[i]+0.5769*det[max(0,i-2)]-0.5769*det[max(0,i-4)]-0.0962*det[max(0,i-6)])*adj
        jI[i]=(0.0962*I1[i]+0.5769*I1[max(0,i-2)]-0.5769*I1[max(0,i-4)]-0.0962*I1[max(0,i-6)])*adj
        jQ[i]=(0.0962*Q1[i]+0.5769*Q1[max(0,i-2)]-0.5769*Q1[max(0,i-4)]-0.0962*Q1[max(0,i-6)])*adj
        I2[i]=I1[i]-jQ[i]; Q2[i]=Q1[i]+jI[i]
        I2[i]=0.2*I2[i]+0.8*I2[i-1]; Q2[i]=0.2*Q2[i]+0.8*Q2[i-1]
        Re_[i]=I2[i]*I2[i-1]+Q2[i]*Q2[i-1]; Im_[i]=I2[i]*Q2[i-1]-Q2[i]*I2[i-1]
        Re_[i]=0.2*Re_[i]+0.8*Re_[i-1]; Im_[i]=0.2*Im_[i]+0.8*Im_[i-1]
        per[i]=(2*np.pi/np.arctan(Im_[i]/Re_[i])) if (Im_[i]!=0 and Re_[i]!=0) else per[i-1]
        if per[i]>1.5*per[i-1]: per[i]=1.5*per[i-1]
        if per[i]<0.67*per[i-1]: per[i]=0.67*per[i-1]
        if per[i]<6.0: per[i]=6.0
        if per[i]>50.0: per[i]=50.0
        per[i]=0.2*per[i]+0.8*per[i-1]; sp_[i]=0.33*per[i]+0.67*sp_[i-1]
        ph[i]=(np.arctan(Q1[i]/I1[i])*180.0/np.pi) if I1[i]!=0 else ph[i-1]
        dp=ph[i-1]-ph[i]
        if dp<1.0: dp=1.0
        alpha=fl/dp
        if alpha<slow_lim: alpha=slow_lim
        if alpha>fl: alpha=fl
        mama[i]=alpha*c[i]+(1-alpha)*mama[i-1]
        fama[i]=0.5*alpha*mama[i]+(1-0.5*alpha)*fama[i-1]
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(7, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        if pos==0:
            if mama[i]>fama[i] and mama[i-1]<=fama[i-1]: pend=1
            elif mama[i]<fama[i] and mama[i-1]>=fama[i-1]: pend=-1
        elif pos==1 and mama[i]<fama[i] and mama[i-1]>=fama[i-1]: pend=-3
        elif pos==-1 and mama[i]>fama[i] and mama[i-1]<=fama[i-1]: pend=3
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_kama_ls(c, o, h, l, er_p, fast_sc, slow_sc, atr_sm, atr_p, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<er_p+2: return 0.0, 0.0, 0
    fc=2.0/(fast_sc+1.0); sc_v=2.0/(slow_sc+1.0)
    kama=np.full(n,np.nan); kama[er_p-1]=c[er_p-1]
    for i in range(er_p, n):
        d=abs(c[i]-c[i-er_p]); v=0.0
        for j in range(1, er_p+1): v+=abs(c[i-j+1]-c[i-j])
        er=d/v if v>0 else 0.0; sc2=(er*(fc-sc_v)+sc_v)**2
        kama[i]=kama[i-1]+sc2*(c[i]-kama[i-1])
    av=_atr(h,l,c,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(er_p+2, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        k=kama[i]; kp=kama[i-1]; a=av[i]
        if k!=k or kp!=kp or a!=a: pass
        else:
            if pos==1:
                if c[i]<ep/sb-atr_sm*a or k<kp: pend=2
            elif pos==-1:
                if c[i]>ep/ss+atr_sm*a or k>kp: pend=2
            if pos==0 and pend==0:
                if c[i]>k and k>kp: pend=1
                elif c[i]<k and k<kp: pend=-1
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_donchian_ls(c, o, h, l, entry_p, atr_p, atr_m, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<entry_p+atr_p: return 0.0, 0.0, 0
    av=_atr(h,l,c,atr_p)
    dh=np.full(n,np.nan); dl=np.full(n,np.nan)
    for i in range(entry_p-1, n):
        mx=h[i]; mn=l[i]
        for j in range(1, entry_p):
            if h[i-j]>mx: mx=h[i-j]
            if l[i-j]<mn: mn=l[i-j]
        dh[i]=mx; dl[i]=mn
    pos=0; ep=0.0; tr=1.0; pend=0; ts=0.0; pk=1.0; mdd=0.0; nt=0
    start=max(entry_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc
        if pend==1 and av[i]==av[i]: ts=o[i]-atr_m*av[i]
        elif pend==-1 and av[i]==av[i]: ts=o[i]+atr_m*av[i]
        pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        d1=dh[i-1]; d2=dl[i-1]; a=av[i]
        if d1!=d1 or d2!=d2 or a!=a: pass
        else:
            if pos==1:
                ns=c[i]-atr_m*a
                if ns>ts: ts=ns
                if c[i]<ts: pend=2
            elif pos==-1:
                ns=c[i]+atr_m*a
                if ns<ts: ts=ns
                if c[i]>ts: pend=2
            if pos==0 and pend==0:
                if c[i]>d1: pend=1
                elif c[i]<d2: pend=-1
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_zscore_ls(c, o, lookback, ez, xz, sz, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<lookback+2: return 0.0, 0.0, 0
    rm=np.full(n,np.nan); rs=np.full(n,np.nan); s=0.0; s2=0.0
    for i in range(n):
        s+=c[i]; s2+=c[i]*c[i]
        if i>=lookback: s-=c[i-lookback]; s2-=c[i-lookback]*c[i-lookback]
        if i>=lookback-1: m=s/lookback; rm[i]=m; rs[i]=np.sqrt(max(0.0,s2/lookback-m*m))
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(lookback, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        m=rm[i]; sd=rs[i]
        if sd==0 or sd!=sd: pass
        else:
            z=(c[i]-m)/sd
            if pos==1 and (z>-xz or z>sz): pend=2
            elif pos==-1 and (z<xz or z<-sz): pend=2
            if pos==0 and pend==0:
                if z<-ez: pend=1
                elif z>ez: pend=-1
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_mombreak_ls(c, o, h, l, hp, prox, atr_p, atr_t, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """Bidirectional MomBreak: long on high breakout, short on low breakdown."""
    n = len(c)
    if n<max(hp, atr_p)+2: return 0.0, 0.0, 0
    rh=np.full(n,np.nan); rl=np.full(n,np.nan)
    for i in range(hp-1, n):
        mx=h[i]; mn=l[i]
        for j in range(1, hp):
            if h[i-j]>mx: mx=h[i-j]
            if l[i-j]<mn: mn=l[i-j]
        rh[i]=mx; rl[i]=mn
    av=_atr(h,l,c,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; ts_l=0.0; ts_s=1e18; pk=1.0; mdd=0.0; nt=0
    start=max(hp, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc
        if pend==1 and av[i]==av[i]: ts_l=o[i]-atr_t*av[i]
        elif pend==-1 and av[i]==av[i]: ts_s=o[i]+atr_t*av[i]
        pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        hv=rh[i]; lv=rl[i]; a=av[i]
        if hv!=hv or lv!=lv or a!=a: pass
        else:
            if pos==1:
                ns=c[i]-atr_t*a
                if ns>ts_l: ts_l=ns
                if c[i]<ts_l: pend=2
            elif pos==-1:
                ns=c[i]+atr_t*a
                if ns<ts_s: ts_s=ns
                if c[i]>ts_s: pend=2
            if pos==0 and pend==0:
                if c[i]>=hv*(1.0-prox): pend=1
                elif c[i]<=lv*(1.0+prox): pend=-1
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_regime_ema_ls(c, o, h, l, atr_p, vt, fe_p, se_p, te_p, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<max(atr_p, max(se_p, te_p))+2: return 0.0, 0.0, 0
    av=_atr(h,l,c,atr_p)
    fk=2.0/(fe_p+1.0); sk=2.0/(se_p+1.0); tk=2.0/(te_p+1.0)
    ef=np.empty(n); es=np.empty(n); et=np.empty(n)
    ef[0]=c[0]; es[0]=c[0]; et[0]=c[0]
    for i in range(1, n):
        ef[i]=c[i]*fk+ef[i-1]*(1-fk); es[i]=c[i]*sk+es[i-1]*(1-sk); et[i]=c[i]*tk+et[i-1]*(1-tk)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(atr_p, max(se_p, te_p))
    for i in range(start, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        a=av[i]
        if a!=a or c[i]<=0: pass
        else:
            hv=a/c[i]>vt
            if hv:
                if pos==0:
                    if ef[i]>es[i] and ef[i-1]<=es[i-1]: pend=1
                    elif ef[i]<es[i] and ef[i-1]>=es[i-1]: pend=-1
                elif pos==1 and ef[i]<es[i]: pend=2
                elif pos==-1 and ef[i]>es[i]: pend=2
            else:
                pd_=(c[i]-et[i])/et[i]
                if pos==0:
                    if pd_<-0.02: pend=1
                    elif pd_>0.02: pend=-1
                elif pos==1 and pd_>0.0: pend=2
                elif pos==-1 and pd_<0.0: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_dualmom_ls(c, o, fast_lb, slow_lb, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c); lb=max(fast_lb, slow_lb)
    if n<lb+2: return 0.0, 0.0, 0
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(lb, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        fast_ret=(c[i]-c[i-fast_lb])/c[i-fast_lb]
        slow_ret=(c[i]-c[i-slow_lb])/c[i-slow_lb]
        if pos==0:
            if fast_ret>0 and slow_ret>0: pend=1
            elif fast_ret<0 and slow_ret<0: pend=-1
        elif pos==1:
            if fast_ret<0 or slow_ret<0: pend=2
        elif pos==-1:
            if fast_ret>0 or slow_ret>0: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_consensus_ls(c, o, ma_s_arr, ma_l_arr, rsi_arr, mom_lb,
                    rsi_os, rsi_ob, vote_thr, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    n = len(c)
    if n<mom_lb+2: return 0.0, 0.0, 0
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(mom_lb, n):
        pos,ep,tr,tc,liq=_fx_lev(pend,pos,ep,o[i],tr,sb,ss,cm,lev,dc,pfrac); nt+=tc; pend=0
        if liq: pos=0; ep=0.0; continue
        pos,ep,tr,tc2=_sl_exit(pos,ep,tr,c[i],sb,ss,cm,lev,sl,pfrac,sl_slip); nt+=tc2
        v=0; ms=ma_s_arr[i]; ml=ma_l_arr[i]; r=rsi_arr[i]
        if ms!=ms or ml!=ml or r!=r: continue
        if ms>ml: v+=1
        elif ms<ml: v-=1
        if r<rsi_os: v+=1
        elif r>rsi_ob: v-=1
        mom_ret=(c[i]-c[i-mom_lb])/c[i-mom_lb] if c[i-mom_lb]>0 else 0.0
        if mom_ret>0.02: v+=1
        elif mom_ret<-0.02: v-=1
        if pos==0:
            if v>=vote_thr: pend=1
            elif v<=-vote_thr: pend=-1
        elif pos==1 and v<=-1: pend=2
        elif pos==-1 and v>=1: pend=2
        eq=_mtm_lev(pos,tr,c[i],ep,sb,ss,cm,lev,sl,pfrac)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1:
        raw=(c[n-1]*ss*(1-cm))/(ep*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    elif pos==-1:
        raw=(ep*(1-cm))/(c[n-1]*sb*(1+cm)); dep=_deploy(tr,pfrac); tr+=dep*((raw-1.0)*lev); tr=max(0.01,tr); nt+=1
    return (tr-1.0)*100.0, mdd, nt


# =====================================================================
#  Evaluation Dispatch
# =====================================================================

def eval_ls(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    if params is None: return (0.0, 0.0, 0)
    try:
        p = params
        if name=="MA":
            return bt_ma_ls(c,o,_rolling_mean(c,int(p[0])),_rolling_mean(c,int(p[1])),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="RSI":
            return bt_rsi_ls(c,o,_rsi_wilder(c,int(p[0])),float(p[1]),float(p[2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="MACD":
            return bt_macd_ls(c,o,_ema(c,int(p[0])),_ema(c,int(p[1])),int(p[2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="Drift":
            return bt_drift_ls(c,o,int(p[0]),float(p[1]),int(p[2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="RAMOM":
            return bt_ramom_ls(c,o,int(p[0]),int(p[1]),float(p[2]),float(p[3]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="Turtle":
            return bt_turtle_ls(c,o,h,l,int(p[0]),int(p[1]),int(p[2]),float(p[3]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="Bollinger":
            return bt_bollinger_ls(c,o,int(p[0]),float(p[1]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="Keltner":
            return bt_keltner_ls(c,o,h,l,int(p[0]),int(p[1]),float(p[2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="MultiFactor":
            return bt_multifactor_ls(c,o,int(p[0]),int(p[1]),int(p[2]),float(p[3]),float(p[4]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="VolRegime":
            ms_,ml_=int(max(2,p[2])),int(max(5,p[3]))
            if ms_>=ml_: return (0.0,0.0,0)
            return bt_volregime_ls(c,o,h,l,int(p[0]),float(p[1]),ms_,ml_,14,int(p[4]),int(p[5]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="MESA":
            return bt_mesa_ls(c,o,float(p[0]),float(p[1]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="KAMA":
            return bt_kama_ls(c,o,h,l,int(p[0]),int(p[1]),int(p[2]),float(p[3]),int(p[4]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="Donchian":
            return bt_donchian_ls(c,o,h,l,int(p[0]),int(p[1]),float(p[2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="ZScore":
            return bt_zscore_ls(c,o,int(p[0]),float(p[1]),float(p[2]),float(p[3]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="MomBreak":
            return bt_mombreak_ls(c,o,h,l,int(p[0]),float(p[1]),int(p[2]),float(p[3]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="RegimeEMA":
            fe,se=int(max(2,p[2])),int(max(5,p[3]))
            if fe>=se: return (0.0,0.0,0)
            return bt_regime_ema_ls(c,o,h,l,int(p[0]),float(p[1]),fe,se,int(p[4]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="DualMom":
            fl_,sl_=int(max(2,p[0])),int(max(5,p[1]))
            if fl_>=sl_: return (0.0,0.0,0)
            return bt_dualmom_ls(c,o,fl_,sl_,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="Consensus":
            ms_,ml_=int(min(200,max(2,p[0]))),int(min(200,max(5,p[1])))
            if ms_>=ml_: return (0.0,0.0,0)
            rp_=int(min(200,max(2,p[2])))
            return bt_consensus_ls(c,o,_rolling_mean(c,ms_),_rolling_mean(c,ml_),_rsi_wilder(c,rp_),
                                   int(p[3]),float(p[4]),float(p[5]),int(p[6]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
    except Exception:
        pass
    return (0.0, 0.0, 0)


def eval_ls_precomp(name, params, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    if params is None: return (0.0, 0.0, 0)
    try:
        p = params
        if name=="MA":
            s,lg=int(min(199,max(2,p[0]))),int(min(200,max(3,p[1])))
            if s>=lg: return (0.0,0.0,0)
            return bt_ma_ls(c,o,mas[s],mas[lg],sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="RSI":
            pr=int(min(200,max(2,p[0])))
            return bt_rsi_ls(c,o,rsis[pr],float(p[1]),float(p[2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="MACD":
            f,s_,sg=int(min(98,max(2,p[0]))),int(min(200,max(4,p[1]))),int(min(100,max(2,p[2])))
            if f>=s_: return (0.0,0.0,0)
            return bt_macd_ls(c,o,emas[f],emas[s_],sg,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="Drift":
            return bt_drift_ls(c,o,int(max(5,p[0])),float(p[1]),int(max(1,p[2])),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="RAMOM":
            return bt_ramom_ls(c,o,int(max(2,p[0])),int(max(2,p[1])),float(p[2]),float(p[3]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="Turtle":
            return bt_turtle_ls(c,o,h,l,int(max(2,p[0])),int(max(2,p[1])),int(max(5,p[2])),float(p[3]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="Bollinger":
            return bt_bollinger_ls(c,o,int(max(3,p[0])),float(max(0.1,p[1])),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="Keltner":
            return bt_keltner_ls(c,o,h,l,int(max(3,p[0])),int(max(5,p[1])),float(max(0.1,p[2])),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="MultiFactor":
            return bt_multifactor_ls(c,o,int(max(2,p[0])),int(max(2,p[1])),int(max(2,p[2])),float(p[3]),float(p[4]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="VolRegime":
            ms_,ml_=int(max(2,p[2])),int(max(5,p[3]))
            if ms_>=ml_: return (0.0,0.0,0)
            return bt_volregime_ls(c,o,h,l,int(max(5,p[0])),float(p[1]),ms_,ml_,14,int(p[4]),int(p[5]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="MESA":
            return bt_mesa_ls(c,o,float(max(0.01,p[0])),float(max(0.005,p[1])),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="KAMA":
            return bt_kama_ls(c,o,h,l,int(max(2,p[0])),int(max(2,p[1])),int(max(5,p[2])),float(p[3]),int(max(5,p[4])),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="Donchian":
            return bt_donchian_ls(c,o,h,l,int(max(3,p[0])),int(max(5,p[1])),float(max(0.5,p[2])),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="ZScore":
            return bt_zscore_ls(c,o,int(max(5,p[0])),float(p[1]),float(p[2]),float(p[3]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="MomBreak":
            return bt_mombreak_ls(c,o,h,l,int(max(5,p[0])),float(max(0,p[1])),int(max(5,p[2])),float(max(0.5,p[3])),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="RegimeEMA":
            fe,se=int(max(2,p[2])),int(max(5,p[3]))
            if fe>=se: return (0.0,0.0,0)
            return bt_regime_ema_ls(c,o,h,l,int(max(5,p[0])),float(p[1]),fe,se,int(max(10,p[4])),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="DualMom":
            fl_,sl_=int(max(2,p[0])),int(max(5,p[1]))
            if fl_>=sl_: return (0.0,0.0,0)
            return bt_dualmom_ls(c,o,fl_,sl_,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        elif name=="Consensus":
            ms_,ml_=int(min(200,max(2,p[0]))),int(min(200,max(5,p[1])))
            if ms_>=ml_: return (0.0,0.0,0)
            rp_=int(min(200,max(2,p[2])))
            return bt_consensus_ls(c,o,mas[ms_],mas[ml_],rsis[rp_],
                                   int(max(5,p[3])),float(p[4]),float(p[5]),int(p[6]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
    except Exception:
        pass
    return (0.0, 0.0, 0)


# =====================================================================
#  Parameter Grid Scan (at fixed leverage)
# =====================================================================

def scan_all_ls(c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """Scan all 18 L/S strategies with configurable stop-loss."""
    R = {}

    # 1. MA
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for s in range(5,100,3):
        for lg in range(s+5,201,5):
            r,d,n=bt_ma_ls(c,o,mas[s],mas[lg],sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
            sc=_score(r,d,n); cnt+=1
            if sc>bs: bs=sc; bp=(s,lg); br=r; bd=d; bn=n
    R["MA"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 2. RSI
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for p in range(5,101,5):
        for os_v in range(15,40,5):
            for ob_v in range(60,90,5):
                r,d,n=bt_rsi_ls(c,o,rsis[p],float(os_v),float(ob_v),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(p,os_v,ob_v); br=r; bd=d; bn=n
    R["RSI"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 3. MACD
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for f in range(4,50,3):
        for s in range(f+4,120,5):
            for sg in range(3,min(s,50),4):
                r,d,n=bt_macd_ls(c,o,emas[f],emas[s],sg,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(f,s,sg); br=r; bd=d; bn=n
    R["MACD"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 4. Drift
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for lb in range(10,120,10):
        for dt in [0.55,0.60,0.65,0.70]:
            for hp in range(3,25,4):
                r,d,n=bt_drift_ls(c,o,lb,dt,hp,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(lb,dt,hp); br=r; bd=d; bn=n
    R["Drift"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 5. RAMOM
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for mp in range(5,100,10):
        for vp in range(5,50,10):
            for ez in [1.0,1.5,2.0,2.5,3.0]:
                for xz in [0.0,0.5,1.0]:
                    r,d,n=bt_ramom_ls(c,o,mp,vp,ez,xz,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(mp,vp,ez,xz); br=r; bd=d; bn=n
    R["RAMOM"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 6. Turtle
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ep_ in range(10,80,10):
        for xp in range(5,40,7):
            for ap in [10,14,20]:
                for am in [1.5,2.0,2.5,3.0]:
                    r,d,n=bt_turtle_ls(c,o,h,l,ep_,xp,ap,am,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(ep_,xp,ap,am); br=r; bd=d; bn=n
    R["Turtle"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 7. Bollinger
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for p in range(10,120,5):
        for ns in [1.0,1.5,2.0,2.5,3.0]:
            r,d,n=bt_bollinger_ls(c,o,p,ns,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
            sc=_score(r,d,n); cnt+=1
            if sc>bs: bs=sc; bp=(p,ns); br=r; bd=d; bn=n
    R["Bollinger"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 8. Keltner
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ep_ in range(10,100,10):
        for ap in [10,14,20]:
            for am in [1.0,1.5,2.0,2.5,3.0]:
                r,d,n=bt_keltner_ls(c,o,h,l,ep_,ap,am,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(ep_,ap,am); br=r; bd=d; bn=n
    R["Keltner"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 9. MultiFactor
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for rp in [7,14,21]:
        for mp in range(10,60,10):
            for vp in range(10,40,10):
                for lt_ in [0.55,0.60,0.65,0.70]:
                    for st_ in [0.25,0.30,0.35,0.40]:
                        r,d,n=bt_multifactor_ls(c,o,rp,mp,vp,lt_,st_,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(rp,mp,vp,lt_,st_); br=r; bd=d; bn=n
    R["MultiFactor"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 10. VolRegime
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ap in [14,20]:
        for vt_ in [0.015,0.020,0.025,0.030]:
            for ms_ in [5,10,20]:
                for ml_ in [30,50,80]:
                    if ms_>=ml_: continue
                    for ros in [25,30]:
                        for rob in [70,75]:
                            r,d,n=bt_volregime_ls(c,o,h,l,ap,vt_,ms_,ml_,14,ros,rob,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                            sc=_score(r,d,n); cnt+=1
                            if sc>bs: bs=sc; bp=(ap,vt_,ms_,ml_,ros,rob); br=r; bd=d; bn=n
    R["VolRegime"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 11. MESA
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for fl_ in [0.3,0.5,0.7]:
        for sl_ in [0.02,0.05,0.10]:
            r,d,n=bt_mesa_ls(c,o,fl_,sl_,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
            sc=_score(r,d,n); cnt+=1
            if sc>bs: bs=sc; bp=(fl_,sl_); br=r; bd=d; bn=n
    R["MESA"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 12. KAMA
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for erp in [10,15,20]:
        for fsc in [2,3]:
            for ssc in [20,30,50]:
                for asm_ in [1.5,2.0,2.5,3.0]:
                    for ap in [14,20]:
                        r,d,n=bt_kama_ls(c,o,h,l,erp,fsc,ssc,asm_,ap,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(erp,fsc,ssc,asm_,ap); br=r; bd=d; bn=n
    R["KAMA"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 13. Donchian
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ep_ in range(10,80,10):
        for ap in [10,14,20]:
            for am in [1.5,2.0,2.5,3.0]:
                r,d,n=bt_donchian_ls(c,o,h,l,ep_,ap,am,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(ep_,ap,am); br=r; bd=d; bn=n
    R["Donchian"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 14. ZScore
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for lb in range(15,100,10):
        for ez in [1.5,2.0,2.5]:
            for xz in [0.0,0.5]:
                for sz in [3.0,4.0]:
                    r,d,n=bt_zscore_ls(c,o,lb,ez,xz,sz,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(lb,ez,xz,sz); br=r; bd=d; bn=n
    R["ZScore"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 15. MomBreak
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for hp_ in [20,40,60,100,200]:
        for pp in [0.00,0.02,0.05,0.08]:
            for ap in [10,14,20]:
                for at_ in [1.5,2.0,2.5,3.0]:
                    r,d,n=bt_mombreak_ls(c,o,h,l,hp_,pp,ap,at_,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(hp_,pp,ap,at_); br=r; bd=d; bn=n
    R["MomBreak"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 16. RegimeEMA
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ap in [14,20]:
        for vt_ in [0.015,0.020,0.025]:
            for fe in [5,10,15]:
                for se in [20,40,60]:
                    if fe>=se: continue
                    for te in [50,80,100]:
                        r,d,n=bt_regime_ema_ls(c,o,h,l,ap,vt_,fe,se,te,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(ap,vt_,fe,se,te); br=r; bd=d; bn=n
    R["RegimeEMA"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 17. DualMom
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for fl in [5,10,20,40,60]:
        for slo in [20,40,80,120,200]:
            if fl>=slo: continue
            r,d,n=bt_dualmom_ls(c,o,fl,slo,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
            sc=_score(r,d,n); cnt+=1
            if sc>bs: bs=sc; bp=(fl,slo); br=r; bd=d; bn=n
    R["DualMom"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 18. Consensus
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ms in [10,20]:
        for ml in [50,100,150]:
            if ms>=ml or ml>len(c)-1: continue
            for rp in [14,21]:
                for mom_lb in [20,40]:
                    for os_v in [25,30]:
                        for ob_v in [70,75]:
                            for vt in [2,3]:
                                r,d,n=bt_consensus_ls(c,o,mas[ms],mas[ml],rsis[rp],mom_lb,
                                                       float(os_v),float(ob_v),vt,sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
                                sc=_score(r,d,n); cnt+=1
                                if sc>bs: bs=sc; bp=(ms,ml,rp,mom_lb,os_v,ob_v,vt); br=r; bd=d; bn=n
    R["Consensus"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    return R


# =====================================================================
#  Deflated Sharpe
# =====================================================================
def deflated_sharpe(sharpe_obs, n_trials, n_bars, skew=0.0, kurtosis=3.0):
    if n_bars < 3 or n_trials < 1: return 0.0
    e_max_sr = ((1.0-0.5772)*(2.0*math.log(n_trials))**0.5+0.5772*(2.0*math.log(n_trials))**-0.5)
    se_sr = ((1.0-skew*sharpe_obs+(kurtosis-1.0)/4.0*sharpe_obs**2)/max(1.0,n_bars-1.0))**0.5
    if se_sr<1e-12: return 1.0 if sharpe_obs>e_max_sr else 0.0
    z = (sharpe_obs-e_max_sr)/se_sr
    return float(sp_stats.norm.cdf(z))


def robust_score(ret_pct, dd_pct, n_trades, n_bars, n_trials=50640):
    """Institutional-grade scoring combining return, risk, and statistical significance.
    Returns (score, ann_ret, sharpe_proxy, dsr_pvalue).
    """
    if n_trades < 3 or n_bars < 30:
        return -1e18, 0.0, 0.0, 1.0

    years = max(0.1, n_bars / 252.0)
    ann_ret = ret_pct / years

    # Sharpe proxy: assume per-trade returns with avg_ret and estimated vol
    avg_trade_ret = ret_pct / max(1, n_trades)
    # Estimate vol from DD: vol ≈ max_dd / 2.5 (empirical)
    est_vol = max(1.0, dd_pct) / 2.5 / math.sqrt(years)
    sharpe = ann_ret / max(0.01, est_vol)

    # Deflated Sharpe p-value
    dsr_p = deflated_sharpe(sharpe, n_trials, n_bars)

    # Composite: Sharpe * sqrt(n_trades) * dsr_pvalue_bonus
    trade_factor = min(1.0, math.sqrt(n_trades / 30.0))
    dsr_bonus = 1.0 + max(0.0, dsr_p - 0.5) * 2.0  # up to 2x bonus for p > 0.5
    score = sharpe * trade_factor * dsr_bonus

    return score, ann_ret, sharpe, dsr_p


def stitched_oos_metrics(window_rets, window_dds, window_trades):
    """Aggregate per-window OOS metrics via additive stitching.

    Fixed-lot model: each window's return is added to equity (not compounded).
    Intra-segment drawdown uses the window's reported DD applied to the
    segment's starting equity level relative to the global peak.

    Returns:
      (total_oos_return_pct, stitched_max_dd_pct, total_oos_trades)
    """
    eq = 1.0
    pk = 1.0
    mdd = 0.0
    nt = 0
    n = min(len(window_rets), len(window_dds), len(window_trades))

    for i in range(n):
        r = float(window_rets[i])
        d = float(window_dds[i])
        t = int(window_trades[i])

        if d < 0.0: d = 0.0
        if d > 100.0: d = 100.0

        seg_start = eq
        gain = r / 100.0
        eq = seg_start + gain

        # Intra-segment valley: the window started at seg_start and had
        # an internal drawdown of d%. The valley is seg_start * (1 - d/100).
        seg_valley = seg_start * max(0.0, 1.0 - d / 100.0)
        if seg_valley < seg_start and pk > 0:
            dd_intra = (pk - seg_valley) / pk * 100.0
            if dd_intra > mdd:
                mdd = dd_intra

        # Update peak and end-of-segment DD
        if eq > pk:
            pk = eq
        if pk > 0:
            dd_end = (pk - eq) / pk * 100.0
            if dd_end > mdd:
                mdd = dd_end

        nt += t

    return (eq - 1.0) * 100.0, min(mdd, 100.0), nt


# =====================================================================
#  Main
# =====================================================================
def main():
    print("=" * 90)
    print("  Long/Short Leveraged + Stop-Loss Scan")
    print("  18 Strategies x Leverage x 7 Stop-Loss Levels")
    print("  Binance Futures (crypto) + IBKR (stocks)")
    print("=" * 90)

    data_dir = os.path.join(_HERE, "..", "data")
    CRYPTO = ["BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "BNB", "LINK"]
    STOCKS = ["AAPL", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "MSFT", "MSTR", "SPY", "QQQ"]
    symbols = CRYPTO + STOCKS
    crypto_syms = set(CRYPTO)

    # ---- Load data ----
    print(f"\n[1] Loading data ...", flush=True)
    datasets = {}
    for sym in symbols:
        fp = os.path.join(data_dir, f"{sym}.csv")
        if not os.path.exists(fp):
            print(f"  {sym}: MISSING"); continue
        df = pd.read_csv(fp, parse_dates=["date"])
        datasets[sym] = {
            "c": df["close"].values.astype(np.float64),
            "o": df["open"].values.astype(np.float64),
            "h": df["high"].values.astype(np.float64),
            "l": df["low"].values.astype(np.float64),
            "n": len(df),
        }
        print(f"  {sym}: {len(df)} bars")

    # ---- JIT warm-up ----
    print(f"\n[0] JIT warm-up ...", end=" ", flush=True)
    t0 = time.time()
    dc = np.random.rand(300).astype(np.float64)*100+100
    dh=dc+2; dl=dc-2; do_=dc+0.5; dm=_rolling_mean(dc,10); dr=_rsi_wilder(dc,14)
    de=_ema(dc,12)
    _sb=1.0003; _ss=0.9997; _cm=CRYPTO_CM; _dc=0.001; _pf=0.5; _sls=0.01
    bt_ma_ls(dc,do_,dm,dm,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_rsi_ls(dc,do_,dr,30.0,70.0,_sb,_ss,_cm,2.0,_dc,0.5,_pf,_sls)
    bt_macd_ls(dc,do_,de,de,9,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_drift_ls(dc,do_,20,0.6,5,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_ramom_ls(dc,do_,10,10,2.0,0.5,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_turtle_ls(dc,do_,dh,dl,10,5,14,2.0,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_bollinger_ls(dc,do_,20,2.0,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_keltner_ls(dc,do_,dh,dl,20,14,2.0,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_multifactor_ls(dc,do_,14,20,20,0.6,0.35,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_volregime_ls(dc,do_,dh,dl,14,0.02,5,20,14,30,70,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_mesa_ls(dc,do_,0.5,0.05,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_kama_ls(dc,do_,dh,dl,10,2,30,2.0,14,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_donchian_ls(dc,do_,dh,dl,20,14,2.0,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_zscore_ls(dc,do_,20,2.0,0.5,4.0,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_mombreak_ls(dc,do_,dh,dl,20,0.02,14,2.0,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_regime_ema_ls(dc,do_,dh,dl,14,0.02,5,20,50,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_dualmom_ls(dc,do_,10,50,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    bt_consensus_ls(dc,do_,dm,dm,dr,20,30.0,70.0,2,_sb,_ss,_cm,1.0,_dc,0.5,_pf,_sls)
    _fx_lev(0,0,0.0,100.0,1.0,_sb,_ss,_cm,10.0,0.001,0.5)
    _sl_exit(1,100.0*_sb,1.0,100.0,_sb,_ss,_cm,10.0,0.5,0.5,0.01)
    _mtm_lev(1,1.0,100.0,100.0,_sb,_ss,_cm,10.0,0.5,0.5)
    print(f"done ({time.time()-t0:.1f}s)")

    n_windows = len(WF_WINDOWS)
    grand_t0 = time.time()
    total_combos = 0

    # ---- Results storage: keyed by (sym, lev, sl) ----
    all_results = {}

    # ---- Main scan loop: per-symbol, per-leverage, per-stop-loss ----
    print(f"\n[2] Walk-Forward + Leverage + Stop-Loss Scan ...\n", flush=True)

    for sym in sorted(datasets.keys()):
        is_crypto = sym in crypto_syms
        lev_grid = CRYPTO_LEVERAGE_GRID if is_crypto else STOCK_LEVERAGE_GRID

        base_slip = CRYPTO_SB_BASE if is_crypto else STOCK_SB_BASE
        cm = CRYPTO_CM if is_crypto else STOCK_CM

        if is_crypto:
            base_funding = CRYPTO_FUNDING_DAILY
        else:
            borrow_ann = STOCK_BORROW_RATES.get(sym, 0.01)
            base_funding = (borrow_ann + STOCK_MARGIN_RATE) / 365.0

        D = datasets[sym]; c,o,h,l = D["c"],D["o"],D["h"],D["l"]; n = D["n"]
        print(f"  {sym} ({n} bars, {'crypto' if is_crypto else 'stock'}, "
              f"cm={cm*100:.2f}%, base_slip={base_slip*100:.2f}%, "
              f"base_dc={base_funding*100:.4f}%/day):", flush=True)

        for lev in lev_grid:
            # --- Compute realistic per-leverage cost parameters ---
            # Dynamic slippage: scales with sqrt(lev) (market impact model)
            slip = base_slip * math.sqrt(lev)
            sb = 1.0 + slip
            ss = 1.0 - slip
            # Dynamic funding: scales with (1 + 0.02*lev) premium
            daily_cost = base_funding * lev * (1.0 + 0.02 * lev)
            # Position fraction: caps equity deployed per trade
            pfrac = POSITION_FRAC.get(lev, max(0.10, 1.0 / math.sqrt(lev)))
            # SL slippage: extra loss beyond SL threshold in fast markets
            sl_slip = base_slip * lev * 0.5

            for sl_pct in STOP_LOSS_GRID:
                t_run = time.time()
                best_params_last = {sn: None for sn in LS_STRAT_NAMES}
                wfe_vals = {sn: [] for sn in LS_STRAT_NAMES}
                gap_vals = {sn: [] for sn in LS_STRAT_NAMES}
                oos_ret_vals = {sn: [] for sn in LS_STRAT_NAMES}
                oos_dd_vals = {sn: [] for sn in LS_STRAT_NAMES}
                oos_nt_vals = {sn: [] for sn in LS_STRAT_NAMES}
                run_combos = 0

                for wi, (tr_pct, va_pct, te_pct) in enumerate(WF_WINDOWS):
                    tr_end = int(n * tr_pct)
                    va_start = min(tr_end + EMBARGO, int(n * va_pct))
                    va_end = int(n * va_pct)
                    te_end = min(int(n * te_pct), n)
                    if va_end > te_end: va_end = te_end
                    if va_start >= va_end: va_start = va_end

                    c_tr,o_tr = c[:tr_end],o[:tr_end]
                    h_tr,l_tr = h[:tr_end],l[:tr_end]
                    c_va,o_va = c[va_start:va_end],o[va_start:va_end]
                    h_va,l_va = h[va_start:va_end],l[va_start:va_end]
                    c_te,o_te = c[va_end:te_end],o[va_end:te_end]
                    h_te,l_te = h[va_end:te_end],l[va_end:te_end]

                    mas_tr = precompute_all_ma(c_tr,200)
                    emas_tr = precompute_all_ema(c_tr,200)
                    rsis_tr = precompute_all_rsi(c_tr,200)

                    results = scan_all_ls(c_tr,o_tr,h_tr,l_tr,mas_tr,emas_tr,rsis_tr,sb,ss,cm,lev,daily_cost,sl_pct,pfrac,sl_slip)

                    for sn in LS_STRAT_NAMES:
                        res = results[sn]
                        run_combos += res["cnt"]
                        va_r,va_d,va_nt = eval_ls(sn,res["params"],c_va,o_va,h_va,l_va,sb,ss,cm,lev,daily_cost,sl_pct,pfrac,sl_slip)
                        te_r,te_d,te_nt = eval_ls(sn,res["params"],c_te,o_te,h_te,l_te,sb,ss,cm,lev,daily_cost,sl_pct,pfrac,sl_slip)

                        # Validation gate: if train return is massively positive
                        # but validation is deeply negative, disqualify (overfitting)
                        tr_r = res["ret"]
                        if tr_r > 20.0 and va_r < -tr_r * 0.5:
                            te_r = 0.0; te_d = 0.0; te_nt = 0

                        wfe_vals[sn].append(te_r)
                        gap_vals[sn].append(abs(va_r - te_r))
                        oos_ret_vals[sn].append(te_r)
                        oos_dd_vals[sn].append(te_d)
                        oos_nt_vals[sn].append(te_nt)
                        if wi == n_windows - 1:
                            best_params_last[sn] = res["params"]

                total_combos += run_combos

                # Walk-forward stitched OOS return/dd/trades (same metric family as WFE).
                wf_oos = {}
                wf_score = {}
                for sn in LS_STRAT_NAMES:
                    r,d,nt = stitched_oos_metrics(oos_ret_vals[sn], oos_dd_vals[sn], oos_nt_vals[sn])
                    sc, ann_r, shrp, dsr_p = robust_score(r, d, nt, n)
                    wf_oos[sn] = {"ret": r, "dd": d, "nt": nt,
                                  "ann_ret": ann_r, "sharpe": shrp, "dsr_p": dsr_p}
                    wf_score[sn] = sc

                # Keep this only for diagnostics/debugging (not used for ranking/reporting).
                full_ret_last = {}
                for sn in LS_STRAT_NAMES:
                    r,d,nt = eval_ls(sn,best_params_last[sn],c,o,h,l,sb,ss,cm,lev,daily_cost,sl_pct,pfrac,sl_slip)
                    full_ret_last[sn] = {"ret": r, "dd": d, "nt": nt}

                elapsed_run = time.time() - t_run
                all_results[(sym,lev,sl_pct)] = {
                    "best_params": best_params_last,
                    "wfe": {sn: np.mean(wfe_vals[sn]) for sn in LS_STRAT_NAMES},
                    "gen_gap": {sn: np.mean(gap_vals[sn]) for sn in LS_STRAT_NAMES},
                    "wf_oos": wf_oos,
                    "wf_score": wf_score,
                    "full_ret_last": full_ret_last,
                    "combos": run_combos,
                }

                ranked = sorted(LS_STRAT_NAMES,
                                key=lambda s: all_results[(sym,lev,sl_pct)]["wfe"][s], reverse=True)
                top3 = ranked[:3]
                top_info = ", ".join([f"{s}={all_results[(sym,lev,sl_pct)]['wfe'][s]:+.1f}%" for s in top3])
                print(f"    {lev:>3}x SL-{int(sl_pct*100)}%: {run_combos:>7,} combos {elapsed_run:5.1f}s  Top: {top_info}", flush=True)

    total_elapsed = time.time() - grand_t0

    # ====================================================================
    #  Generate Report
    # ====================================================================
    print(f"\n[3] Generating report ...", flush=True)

    rpt_path = os.path.join(_HERE, "..", "docs", "LONG_SHORT_LEVERAGED_REPORT.md")
    os.makedirs(os.path.dirname(rpt_path), exist_ok=True)

    L = []
    L.append("# Long/Short Leveraged + Stop-Loss Backtest Report")
    L.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}  |  **Methodology**: Walk-Forward + Leverage + Stop-Loss Scan\n")
    L.append(f"> **{total_combos:,}** backtests | **{total_elapsed:.1f}s** elapsed | **{total_combos/max(1,total_elapsed):,.0f}**/s")
    L.append(f"> **18** bidirectional strategies (all long + short)")
    L.append(f"> Walk-Forward: {n_windows} purged windows | Embargo: {EMBARGO} bars")
    L.append(f"> Stop-Loss Grid: {[f'-{int(s*100)}%' for s in STOP_LOSS_GRID]}")
    L.append(f"> Leverage Grid: Crypto {CRYPTO_LEVERAGE_GRID} | Stocks {STOCK_LEVERAGE_GRID}")
    L.append(f"> Execution: **Next-Open** | Cost: Binance (crypto) / IBKR (stocks)\n")

    L.append("## Cost Model (Realistic, Leverage-Adjusted)\n")
    L.append("| Parameter | Crypto (Binance) | Stocks (IBKR) |")
    L.append("|:---|:---|:---|")
    L.append(f"| Commission (per side) | {CRYPTO_CM*100:.2f}% (taker) | {STOCK_CM*100:.2f}% |")
    L.append(f"| Base Slippage (1x) | {CRYPTO_SB_BASE*100:.2f}% | {STOCK_SB_BASE*100:.2f}% |")
    L.append(f"| Base Funding/Borrow | {CRYPTO_FUNDING_DAILY*100:.3f}%/day (~{CRYPTO_FUNDING_DAILY*365*100:.1f}%/yr) | variable (1-5%/yr + {STOCK_MARGIN_RATE*100:.1f}% margin) |")
    L.append(f"| Max Leverage | {max(CRYPTO_LEVERAGE_GRID)}x | {max(STOCK_LEVERAGE_GRID)}x |")
    L.append("")
    L.append("### Leverage-Dependent Adjustments\n")
    L.append("| Leverage | Slippage | Daily Cost (Crypto) | Position Fraction | SL Extra Slip |")
    L.append("|---:|---:|---:|---:|---:|")
    for lev_r in CRYPTO_LEVERAGE_GRID:
        s_r = CRYPTO_SB_BASE * math.sqrt(lev_r)
        dc_r = CRYPTO_FUNDING_DAILY * lev_r * (1.0 + 0.02 * lev_r)
        pf_r = POSITION_FRAC.get(lev_r, max(0.10, 1.0 / math.sqrt(lev_r)))
        sls_r = CRYPTO_SB_BASE * lev_r * 0.5
        L.append(f"| {lev_r}x | {s_r*100:.3f}% | {dc_r*100:.4f}% | {pf_r*100:.0f}% | {sls_r*100:.3f}% |")
    L.append("")

    # ---- Best (Strategy x Leverage x Stop-Loss) per Asset ----
    L.append("---\n")
    L.append("## Overall Best (Strategy x Leverage x Stop-Loss) per Asset\n")
    L.append("| Asset | Type | Strategy | Lev | SL | Ann Ret | Comp OOS | Max DD | Trades | Sharpe | DSR p | Score |")
    L.append("|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    best_overall = {}
    for sym in sorted(datasets.keys()):
        is_crypto = sym in crypto_syms
        lev_grid = CRYPTO_LEVERAGE_GRID if is_crypto else STOCK_LEVERAGE_GRID
        best_key = None; best_score = -1e18
        for lev in lev_grid:
            for sl_pct in STOP_LOSS_GRID:
                key = (sym, lev, sl_pct)
                if key not in all_results: continue
                res = all_results[key]
                for sn in LS_STRAT_NAMES:
                    sc = res["wf_score"][sn]
                    if sc > best_score:
                        best_score = sc; best_key = (sym, lev, sl_pct, sn)
        if best_key:
            sym_, lev_, sl_, sn_ = best_key
            res = all_results[(sym_, lev_, sl_)]
            fr = res["wf_oos"][sn_]
            L.append(f"| **{sym}** | {'Crypto' if is_crypto else 'Stock'} | {sn_} | {lev_}x | "
                     f"-{int(sl_*100)}% | {fr['ann_ret']:+.1f}%/yr | {fr['ret']:+.1f}% | {fr['dd']:.1f}% | {fr['nt']} | "
                     f"{fr['sharpe']:.2f} | {fr['dsr_p']:.3f} | {res['wf_score'][sn_]:+.2f} |")
            best_overall[sym] = best_key
    L.append("")

    # ---- Stop-Loss Impact Heat Map ----
    L.append("---\n")
    L.append("## Stop-Loss Impact Analysis\n")
    L.append("Best WFE across all strategies at each (leverage, stop-loss) combination.\n")

    for sym in sorted(datasets.keys()):
        is_crypto = sym in crypto_syms
        lev_grid = CRYPTO_LEVERAGE_GRID if is_crypto else STOCK_LEVERAGE_GRID
        L.append(f"### {sym} ({'Crypto' if is_crypto else 'Stock'})\n")
        sl_headers = [f"SL-{int(s*100)}%" for s in STOP_LOSS_GRID]
        L.append("| Leverage | " + " | ".join(sl_headers) + " | Best SL |")
        L.append("|:---|" + ":---:|" * (len(STOP_LOSS_GRID) + 1))

        for lev in lev_grid:
            vals = []; best_sl = 0.80; best_w = -1e18
            for sl_pct in STOP_LOSS_GRID:
                key = (sym, lev, sl_pct)
                if key in all_results:
                    best_strat_wfe = max(all_results[key]["wfe"][sn] for sn in LS_STRAT_NAMES)
                    vals.append(f"{best_strat_wfe:+.1f}%")
                    if best_strat_wfe > best_w: best_w = best_strat_wfe; best_sl = sl_pct
                else:
                    vals.append("N/A")
            L.append(f"| {lev}x | " + " | ".join(vals) + f" | **-{int(best_sl*100)}%** |")
        L.append("")

    # ---- Top-20 Combinations Overall ----
    L.append("---\n")
    L.append("## Top-20 Strategy-Leverage-StopLoss Combinations (by OOS Score)\n")
    L.append("| Rank | Asset | Strategy | Lev | SL | Score | Ann Ret | Comp OOS | Max DD | Trades | Sharpe | DSR p |")
    L.append("|:---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    all_combos = []
    for (sym, lev, sl_pct), res in all_results.items():
        for sn in LS_STRAT_NAMES:
            fr = res["wf_oos"][sn]
            all_combos.append({
                "sym": sym, "lev": lev, "sl": sl_pct, "sn": sn,
                "score": res["wf_score"][sn],
                "wfe": res["wfe"][sn],
                "full_ret": fr["ret"], "max_dd": fr["dd"], "nt": fr["nt"],
                "ann_ret": fr["ann_ret"], "sharpe": fr["sharpe"], "dsr_p": fr["dsr_p"],
            })
    all_combos.sort(key=lambda x: x["score"], reverse=True)
    for i, c_ in enumerate(all_combos[:20]):
        L.append(f"| {i+1} | {c_['sym']} | {c_['sn']} | {c_['lev']}x | -{int(c_['sl']*100)}% | "
                 f"{c_['score']:+.2f} | {c_['ann_ret']:+.1f}%/yr | {c_['full_ret']:+.1f}% | {c_['max_dd']:.1f}% | {c_['nt']} | "
                 f"{c_['sharpe']:.2f} | {c_['dsr_p']:.3f} |")
    L.append("")

    # ---- Stop-Loss vs No-Stop Comparison ----
    L.append("---\n")
    L.append("## Stop-Loss Effectiveness: Tightest SL (-20%) vs Widest SL (-80%)\n")
    L.append("Shows how tight stop-losses change performance vs. the baseline (widest SL).\n")
    L.append("| Asset | Strategy | Lev | WFE (-20% SL) | WFE (-80% SL) | DD (-20%) | DD (-80%) | SL Benefit? |")
    L.append("|:---|:---|---:|---:|---:|---:|---:|:---|")

    for sym in sorted(datasets.keys()):
        is_crypto = sym in crypto_syms
        lev_grid = CRYPTO_LEVERAGE_GRID if is_crypto else STOCK_LEVERAGE_GRID
        best_lev = lev_grid[-1]
        k_tight = (sym, best_lev, 0.20)
        k_wide = (sym, best_lev, 0.80)
        if k_tight not in all_results or k_wide not in all_results: continue
        for sn in LS_STRAT_NAMES:
            w_t = all_results[k_tight]["wfe"][sn]
            w_w = all_results[k_wide]["wfe"][sn]
            fr_t = all_results[k_tight]["wf_oos"][sn]
            fr_w = all_results[k_wide]["wf_oos"][sn]
            if fr_t["nt"] < 3 and fr_w["nt"] < 3: continue
            benefit = "YES" if (w_t > w_w and fr_t["dd"] < fr_w["dd"]) else "MIXED" if w_t > w_w else "NO"
            L.append(f"| {sym} | {sn} | {best_lev}x | {w_t:+.1f}% | {w_w:+.1f}% | "
                     f"{fr_t['dd']:.1f}% | {fr_w['dd']:.1f}% | {benefit} |")
    L.append("")

    # ---- Leverage Risk Table ----
    L.append("---\n")
    L.append("## Leverage Risk Analysis\n")
    L.append("| Leverage | Daily Cost (Crypto) | Annual Cost | 1% Move P&L | 5% Move P&L |")
    L.append("|---:|---:|---:|---:|---:|")
    for lev in CRYPTO_LEVERAGE_GRID:
        dc_pct = CRYPTO_FUNDING_DAILY * lev * (1.0 + 0.02 * lev) * 100
        ann_pct = dc_pct * 365
        L.append(f"| {lev}x | {dc_pct:.2f}% | {ann_pct:.1f}% | ±{lev*1.0:.0f}% | ±{lev*5.0:.0f}% |")
    L.append("")

    L.append("### Stop-Loss vs Leverage Interaction\n")
    L.append("| Leverage | Effective SL Price Move (-20%) | Effective SL Price Move (-50%) | Effective SL Price Move (-80%) |")
    L.append("|---:|---:|---:|---:|")
    for lev in CRYPTO_LEVERAGE_GRID:
        L.append(f"| {lev}x | {20.0/lev:.1f}% | {50.0/lev:.1f}% | {80.0/lev:.1f}% |")
    L.append("")

    # ---- Investment Recommendations ----
    L.append("---\n")
    L.append("## Investment Recommendations by Asset\n")
    L.append("Based on walk-forward OOS returns across all leverage and stop-loss combinations.\n")

    for sym in sorted(datasets.keys()):
        is_crypto = sym in crypto_syms
        lev_grid = CRYPTO_LEVERAGE_GRID if is_crypto else STOCK_LEVERAGE_GRID
        asset_type = "Crypto" if is_crypto else "Stock"

        L.append(f"### {sym} ({asset_type})\n")

        sym_combos = []
        for lev in lev_grid:
            for sl_pct in STOP_LOSS_GRID:
                key = (sym, lev, sl_pct)
                if key not in all_results: continue
                res = all_results[key]
                for sn in LS_STRAT_NAMES:
                    wfe_ = res["wfe"][sn]
                    fr = res["wf_oos"][sn]
                    sym_combos.append({
                        "sn": sn, "lev": lev, "sl": sl_pct, "wfe": wfe_, "score": res["wf_score"][sn],
                        "full_ret": fr["ret"], "max_dd": fr["dd"], "nt": fr["nt"],
                        "ann_ret": fr["ann_ret"], "sharpe": fr["sharpe"], "dsr_p": fr["dsr_p"],
                        "params": res["best_params"][sn],
                    })

        sym_combos.sort(key=lambda x: x["score"], reverse=True)
        # Deduplicate: keep only one entry per (strategy, params) combo
        seen = set()
        deduped = []
        for c_ in sym_combos:
            if c_["full_ret"] <= 0 or c_["nt"] < 5:
                continue
            key = (c_["sn"], str(c_["params"]))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c_)
        top5 = deduped[:5]

        if top5:
            L.append("| Rank | Strategy | Lev | SL | Score | Ann Ret | Comp OOS | Max DD | Trades | Sharpe | DSR p | Params |")
            L.append("|:---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
            for ri, c_ in enumerate(top5):
                L.append(f"| {ri+1} | {c_['sn']} | {c_['lev']}x | -{int(c_['sl']*100)}% | {c_['score']:+.2f} | "
                         f"{c_['ann_ret']:+.1f}%/yr | {c_['full_ret']:+.1f}% | {c_['max_dd']:.1f}% | {c_['nt']} | "
                         f"{c_['sharpe']:.2f} | {c_['dsr_p']:.3f} | {c_['params']} |")
            L.append("")

            best = top5[0]
            # Risk considers BOTH drawdown AND leverage
            lev_risk = best["lev"]
            if lev_risk >= 20:
                risk_level = "EXTREME"
            elif lev_risk >= 10 or best["max_dd"] >= 50:
                risk_level = "HIGH"
            elif lev_risk >= 5 or best["max_dd"] >= 30:
                risk_level = "MEDIUM"
            elif best["max_dd"] < 15:
                risk_level = "LOW"
            else:
                risk_level = "MEDIUM"
            pos_map = {"LOW": "3-5%", "MEDIUM": "1-3%", "HIGH": "0.5-1%", "EXTREME": "0.25-0.5%"}
            position_size = pos_map[risk_level]

            L.append(f"**Recommendation**: **{best['sn']}** | {best['lev']}x leverage | -{int(best['sl']*100)}% stop-loss")
            L.append(f"- Parameters: `{best['params']}`")
            L.append(f"- Ann Return: {best['ann_ret']:+.1f}%/yr | Sharpe: {best['sharpe']:.2f} | Max DD: {best['max_dd']:.1f}% | Risk: **{risk_level}**")
            if best['dsr_p'] < 0.05:
                L.append(f"- ⚠ **Statistically NOT significant** (Deflated Sharpe p={best['dsr_p']:.3f})")
            L.append(f"- Position size: **{position_size}** of portfolio")
            if best["lev"] > 1:
                sl_price_move = best["sl"] * 100 / best["lev"]
                L.append(f"- At {best['lev']}x, the -{int(best['sl']*100)}% SL triggers on a {sl_price_move:.1f}% price move")
            if best["lev"] > 1 and is_crypto:
                cost_day = CRYPTO_FUNDING_DAILY * best["lev"] * (1.0 + 0.02 * best["lev"]) * 100
                L.append(f"- Daily funding cost: {cost_day:.2f}% ({cost_day*365:.0f}%/yr)")

            if len(top5) >= 2:
                alt = top5[1]
                L.append(f"\n**Alternative**: {alt['sn']} | {alt['lev']}x | -{int(alt['sl']*100)}% SL | "
                         f"OOS: {alt['wfe']:+.1f}%, DD: {alt['max_dd']:.1f}%")
        else:
            L.append(f"**No robustly positive OOS returns for {sym}.** Consider staying flat.\n")
        L.append("")

    # ---- Key Findings ----
    L.append("---\n")
    L.append("## Key Findings\n")
    L.append("1. **Stop-losses significantly impact leveraged performance**: tighter SL (-20%) protects capital at high leverage "
             "but can cause excessive whipsawing at 1x.")
    L.append("2. **Optimal SL depends on leverage**: at 10x+, tight SL (-20% to -30%) is essential; at 1-2x, wider SL (-60% to -80%) "
             "often performs better by letting trends run.")
    L.append("3. **Bidirectional trading (long+short)** significantly improves strategies during bear markets.")
    L.append("4. **Funding/borrow costs compound**: at high leverage, daily costs eat into returns rapidly.")
    L.append("5. **The interaction of (leverage x stop-loss x strategy)** is non-linear — optimal combinations must be searched jointly, "
             "not independently.")
    L.append("6. **Tight SL at high leverage reduces DD** but also caps upside; the sweet spot varies by asset volatility.\n")

    # ---- How to Apply These Strategies ----
    L.append("---\n")
    L.append("## How to Apply These Strategies\n")
    L.append("### Step 1: Choose Your Asset & Strategy")
    L.append("- Use the per-asset recommendations above")
    L.append("- Prefer strategies rated STRONG or ROBUST in the long-only scan")
    L.append("- Start with 1x leverage until you confirm live performance matches backtest\n")
    L.append("### Step 2: Set Up Execution")
    L.append("- **Crypto**: Use Binance Futures for leveraged positions")
    L.append("  - Set leverage via exchange UI/API before opening positions")
    L.append("  - Enable cross-margin mode for better capital efficiency")
    L.append("  - Set stop-loss orders at the ATR-based stop level")
    L.append("- **Stocks**: Use IBKR margin account for shorts and leverage")
    L.append("  - Ensure sufficient margin maintenance (>25% for RegT accounts)")
    L.append("  - Check stock borrow availability before shorting\n")
    L.append("### Step 3: Risk Management")
    L.append("- **Position sizing**: Never risk more than 1-2% of portfolio per trade")
    L.append("- **Leverage limit**: Cap at 5x for crypto, 2x for stocks in production")
    L.append("- **Stop losses**: Always use hard stop losses, especially with leverage")
    L.append("- **Daily monitoring**: Check funding rates (crypto) and margin levels (stocks) daily")
    L.append("- **Drawdown circuit breaker**: If portfolio drops 15%, reduce leverage to 1x\n")
    L.append("### Step 4: Re-calibration")
    L.append("- Re-run parameter optimization monthly (crypto) or quarterly (stocks)")
    L.append("- Monitor regime changes — switch between trend-following and mean-reversion based on volatility")
    L.append("- Track live vs. backtest performance; if divergence > 30%, stop and investigate\n")

    L.append("---\n")
    L.append("## Disclaimers\n")
    L.append("1. Past performance does NOT guarantee future results.")
    L.append("2. Funding rates are averaged; actual rates vary significantly by market conditions.")
    L.append("3. Liquidation model uses 80% threshold; exchanges may liquidate earlier.")
    L.append("4. Short selling carries theoretically unlimited loss potential.")
    L.append("5. Leverage amplifies both gains AND losses — use with extreme caution.")
    L.append("6. This is for educational and research purposes only, NOT investment advice.\n")

    L.append(f"*Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | Long/Short Leveraged Scan*")

    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  Report: {rpt_path}")

    # CSV
    csv_path = os.path.join(_HERE, "..", "results", "long_short_leveraged.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = []
    for (sym, lev, sl_pct), res in all_results.items():
        for sn in LS_STRAT_NAMES:
            fr = res["wf_oos"][sn]
            rows.append({
                "symbol": sym, "leverage": lev, "stop_loss": sl_pct,
                "strategy": sn,
                "wfe": round(res["wfe"][sn], 2),
                "gen_gap": round(res["gen_gap"][sn], 2),
                "comp_oos_return": round(fr["ret"], 2),
                "max_dd": round(fr["dd"], 2),
                "n_trades": fr["nt"],
                "ann_return": round(fr["ann_ret"], 2),
                "sharpe": round(fr["sharpe"], 2),
                "dsr_pvalue": round(fr["dsr_p"], 4),
                "score": round(res["wf_score"][sn], 2),
                "best_params": str(res["best_params"][sn]),
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")
    print(f"\n  Total: {total_combos:,} backtests in {total_elapsed:.1f}s = {total_combos/max(1,total_elapsed):,.0f}/s")
    print("Done.")


if __name__ == "__main__":
    main()
