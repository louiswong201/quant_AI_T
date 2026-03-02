"""
Unified Numba kernel backtest system.

All 18 long/short leveraged strategy kernels live here as first-class
framework components.  Each kernel runs signals + fills + PnL in a single
compiled loop (~36 000 runs/s on a single core).

Public API
----------
KERNEL_REGISTRY : dict[str, Callable]
    Maps strategy name -> Numba kernel function.
run_kernel(name, params, c, o, h, l, config) -> KernelResult
    Run a single kernel with BacktestConfig translation.
scan_all_kernels(c, o, h, l, config) -> dict[str, dict]
    Full 18-strategy parameter grid scan at a fixed config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numba import njit

from .config import BacktestConfig

# =====================================================================
#  Indicator helpers (Numba-compiled)
# =====================================================================

@njit(cache=True)
def _ema(arr, span):
    n = len(arr); out = np.empty(n, dtype=np.float64)
    k = 2.0 / (span + 1.0); out[0] = arr[0]
    for i in range(1, n):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


@njit(cache=True)
def _rolling_mean(arr, w):
    n = len(arr); out = np.full(n, np.nan); s = 0.0
    for i in range(n):
        s += arr[i]
        if i >= w:
            s -= arr[i - w]
        if i >= w - 1:
            out[i] = s / w
    return out


@njit(cache=True)
def _rolling_std(arr, w):
    n = len(arr); out = np.full(n, np.nan); s = 0.0; s2 = 0.0
    for i in range(n):
        s += arr[i]; s2 += arr[i] * arr[i]
        if i >= w:
            s -= arr[i - w]; s2 -= arr[i - w] * arr[i - w]
        if i >= w - 1:
            m = s / w; out[i] = np.sqrt(max(0.0, s2 / w - m * m))
    return out


@njit(cache=True)
def _atr(high, low, close, period):
    n = len(close); tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     max(abs(high[i] - close[i - 1]),
                         abs(low[i] - close[i - 1])))
    out = np.full(n, np.nan); s = 0.0
    for i in range(period):
        s += tr[i]
    out[period - 1] = s / period
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


@njit(cache=True)
def _rsi_wilder(close, period):
    n = len(close); out = np.full(n, np.nan)
    if n < period + 1:
        return out
    gs = 0.0; ls = 0.0
    for i in range(1, period + 1):
        d = close[i] - close[i - 1]
        if d > 0:
            gs += d
        else:
            ls -= d
    ag = gs / period; al = ls / period
    out[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al)
    for i in range(period + 1, n):
        d = close[i] - close[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        ag = (ag * (period - 1) + g) / period
        al = (al * (period - 1) + l) / period
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al)
    return out


@njit(cache=True)
def _score(ret, dd, nt):
    return ret / max(1.0, dd) * min(1.0, nt / 20.0)


# =====================================================================
#  Precomputation — all Numba-compiled for maximum speed
# =====================================================================

def precompute_all_ma(close: np.ndarray, max_w: int = 200) -> np.ndarray:
    n = len(close)
    cs = np.empty(n + 1, dtype=np.float64); cs[0] = 0.0
    np.cumsum(close, out=cs[1:])
    mas = np.full((max_w + 1, n), np.nan, dtype=np.float64)
    for w in range(2, min(max_w + 1, n + 1)):
        mas[w, w - 1:] = (cs[w:] - cs[:n - w + 1]) / w
    return mas


@njit(cache=True)
def precompute_all_ema(close, max_s):
    n = len(close)
    emas = np.full((max_s + 1, n), np.nan, dtype=np.float64)
    for s in range(2, max_s + 1):
        k = 2.0 / (s + 1.0)
        emas[s, 0] = close[0]
        for i in range(1, n):
            emas[s, i] = close[i] * k + emas[s, i - 1] * (1.0 - k)
    return emas


@njit(cache=True)
def precompute_all_rsi(close, max_p):
    n = len(close)
    rsi = np.full((max_p + 1, n), np.nan, dtype=np.float64)
    for p in range(2, max_p + 1):
        if n <= p:
            continue
        gs = 0.0; ls = 0.0
        for i in range(1, p + 1):
            d = close[i] - close[i - 1]
            if d > 0:
                gs += d
            else:
                ls -= d
        ag = gs / p; al = ls / p
        rsi[p, p] = 100.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al)
        for i in range(p + 1, n):
            d = close[i] - close[i - 1]
            g = d if d > 0 else 0.0
            l_ = -d if d < 0 else 0.0
            ag = (ag * (p - 1) + g) / p
            al = (al * (p - 1) + l_) / p
            rsi[p, i] = 100.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al)
    return rsi


# =====================================================================
#  Core fill/exit/MTM helpers with leverage
# =====================================================================

@njit(cache=True)
def _deploy(tr, pfrac):
    d = tr * pfrac
    if tr > 1.0 and d > pfrac:
        d = pfrac
    return d


@njit(cache=True)
def _fx_lev(pend, pos, ep, oi, tr, sb, ss, cm, lev, daily_cost, pfrac):
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
            if tr < 0.01:
                tr = 0.01
            tc = 1
        elif pos == -1:
            raw = (ep * (1.0 - cm)) / (oi * sb * (1.0 + cm))
            pnl = (raw - 1.0) * lev
            tr += deployed * pnl
            if tr < 0.01:
                tr = 0.01
            tc = 1
        pos = 0
    if pend == 1 or pend == 3:
        ep = oi * sb; pos = 1
    elif pend == -1 or pend == -3:
        ep = oi * ss; pos = -1
    return pos, ep, tr, tc, liq


@njit(cache=True)
def _sl_exit(pos, ep, tr, ci, sb, ss, cm, lev, sl, pfrac, sl_slip):
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
#  18 Long/Short Leveraged Strategy Kernels
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
#  Registry
# =====================================================================

KERNEL_REGISTRY: Dict[str, Callable] = {
    "MA": bt_ma_ls,
    "RSI": bt_rsi_ls,
    "MACD": bt_macd_ls,
    "Drift": bt_drift_ls,
    "RAMOM": bt_ramom_ls,
    "Turtle": bt_turtle_ls,
    "Bollinger": bt_bollinger_ls,
    "Keltner": bt_keltner_ls,
    "MultiFactor": bt_multifactor_ls,
    "VolRegime": bt_volregime_ls,
    "MESA": bt_mesa_ls,
    "KAMA": bt_kama_ls,
    "Donchian": bt_donchian_ls,
    "ZScore": bt_zscore_ls,
    "MomBreak": bt_mombreak_ls,
    "RegimeEMA": bt_regime_ema_ls,
    "DualMom": bt_dualmom_ls,
    "Consensus": bt_consensus_ls,
}

KERNEL_NAMES: List[str] = list(KERNEL_REGISTRY.keys())

POSITION_FRAC: Dict[int, float] = {
    1: 1.0, 2: 0.90, 4: 0.70, 5: 0.60,
    10: 0.40, 20: 0.25, 50: 0.15,
}


# =====================================================================
#  DEFAULT_PARAM_GRIDS — user can override any or all of these
#
#  To add a new strategy:
#    1. Write your @njit kernel: bt_newstrat_ls(c, o, ..., sb, ss, cm,
#       lev, dc, sl, pfrac, sl_slip) -> (ret%, maxDD%, nTrades)
#    2. Add it to KERNEL_REGISTRY
#    3. Add its default param grid to DEFAULT_PARAM_GRIDS
#    4. Add a _scan_<name> function below
#    5. Register _scan_<name> in _SCANNERS dict
#    6. Add eval_kernel dispatch case
#    7. (Optional) Write a Python strategy class with kernel_name property
# =====================================================================

DEFAULT_PARAM_GRIDS: Dict[str, List[tuple]] = {
    "MA": [(s, lg) for s in range(5, 100, 3) for lg in range(s + 5, 201, 5)],
    "RSI": [(p, os_v, ob_v) for p in range(5, 101, 5)
            for os_v in range(15, 40, 5) for ob_v in range(60, 90, 5)],
    "MACD": [(f, s, sg) for f in range(4, 50, 3)
             for s in range(f + 4, 120, 5) for sg in range(3, min(s, 50), 4)],
    "Drift": [(lb, dt, hp) for lb in range(10, 120, 10)
              for dt in [0.55, 0.60, 0.65, 0.70] for hp in range(3, 25, 4)],
    "RAMOM": [(mp, vp, ez, xz) for mp in range(5, 100, 10)
              for vp in range(5, 50, 10)
              for ez in [1.0, 1.5, 2.0, 2.5, 3.0] for xz in [0.0, 0.5, 1.0]],
    "Turtle": [(ep, xp, ap, am) for ep in range(10, 80, 10)
               for xp in range(5, 40, 7) for ap in [10, 14, 20]
               for am in [1.5, 2.0, 2.5, 3.0]],
    "Bollinger": [(p, ns) for p in range(10, 120, 5)
                  for ns in [1.0, 1.5, 2.0, 2.5, 3.0]],
    "Keltner": [(ep, ap, am) for ep in range(10, 100, 10)
                for ap in [10, 14, 20] for am in [1.0, 1.5, 2.0, 2.5, 3.0]],
    "MultiFactor": [(rp, mp, vp, lt, st) for rp in [7, 14, 21]
                    for mp in range(10, 60, 10) for vp in range(10, 40, 10)
                    for lt in [0.55, 0.60, 0.65, 0.70]
                    for st in [0.25, 0.30, 0.35, 0.40]],
    "VolRegime": [(ap, vt, ms, ml, ros, rob)
                  for ap in [14, 20] for vt in [0.015, 0.020, 0.025, 0.030]
                  for ms in [5, 10, 20] for ml in [30, 50, 80] if ms < ml
                  for ros in [25, 30] for rob in [70, 75]],
    "MESA": [(fl, sl) for fl in [0.3, 0.5, 0.7] for sl in [0.02, 0.05, 0.10]],
    "KAMA": [(erp, fsc, ssc, asm, ap) for erp in [10, 15, 20] for fsc in [2, 3]
             for ssc in [20, 30, 50] for asm in [1.5, 2.0, 2.5, 3.0] for ap in [14, 20]],
    "Donchian": [(ep, ap, am) for ep in range(10, 80, 10)
                 for ap in [10, 14, 20] for am in [1.5, 2.0, 2.5, 3.0]],
    "ZScore": [(lb, ez, xz, sz) for lb in range(15, 100, 10)
               for ez in [1.5, 2.0, 2.5] for xz in [0.0, 0.5] for sz in [3.0, 4.0]],
    "MomBreak": [(hp, pp, ap, at) for hp in [20, 40, 60, 100, 200]
                 for pp in [0.00, 0.02, 0.05, 0.08]
                 for ap in [10, 14, 20] for at in [1.5, 2.0, 2.5, 3.0]],
    "RegimeEMA": [(ap, vt, fe, se, te) for ap in [14, 20]
                  for vt in [0.015, 0.020, 0.025]
                  for fe in [5, 10, 15] for se in [20, 40, 60] if fe < se
                  for te in [50, 80, 100]],
    "DualMom": [(fl, slo) for fl in [5, 10, 20, 40, 60]
                for slo in [20, 40, 80, 120, 200] if fl < slo],
    "Consensus": [(ms, ml, rp, mom_lb, os_v, ob_v, vt)
                  for ms in [10, 20] for ml in [50, 100, 150] if ms < ml
                  for rp in [14, 21] for mom_lb in [20, 40]
                  for os_v in [25, 30] for ob_v in [70, 75] for vt in [2, 3]],
}


@dataclass
class KernelResult:
    """Result from a single kernel run."""
    ret_pct: float
    max_dd_pct: float
    n_trades: int
    params: Any = None
    score: float = 0.0


def config_to_kernel_costs(config: BacktestConfig) -> Dict[str, float]:
    """Translate BacktestConfig into the flat cost params kernels expect."""
    lev = config.leverage
    base_slip_buy = config.slippage_bps_buy / 10000.0
    base_slip_sell = config.slippage_bps_sell / 10000.0
    slip = max(base_slip_buy, base_slip_sell) * math.sqrt(lev)
    sb = 1.0 + slip
    ss = 1.0 - slip
    cm = max(config.commission_pct_buy, config.commission_pct_sell)

    dc = config.daily_funding_rate
    if config.funding_leverage_scaling and lev > 1.0:
        dc *= lev * (1.0 + 0.02 * lev)

    sl = config.stop_loss_pct if config.stop_loss_pct else 0.80
    pfrac = config.position_fraction
    if pfrac == 1.0 and lev > 1:
        pfrac = POSITION_FRAC.get(int(lev), max(0.10, 1.0 / math.sqrt(lev)))
    sl_slip = max(base_slip_buy, base_slip_sell) * lev * 0.5

    return {
        "sb": sb, "ss": ss, "cm": cm, "lev": lev,
        "dc": dc, "sl": sl, "pfrac": pfrac, "sl_slip": sl_slip,
    }


def run_kernel(
    name: str,
    params: tuple,
    c: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    config: BacktestConfig,
) -> KernelResult:
    """Run a single named kernel with BacktestConfig cost translation."""
    costs = config_to_kernel_costs(config)
    r, d, nt = eval_kernel(
        name, params, c, o, h, l,
        costs["sb"], costs["ss"], costs["cm"], costs["lev"],
        costs["dc"], costs["sl"], costs["pfrac"], costs["sl_slip"],
    )
    sc = _score(r, d, nt)
    return KernelResult(ret_pct=r, max_dd_pct=d, n_trades=nt, params=params, score=sc)


# =====================================================================
#  Single-kernel dispatch (used by BacktestEngine and robust_scan)
# =====================================================================

def eval_kernel(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """Dispatch to the correct kernel with on-the-fly indicator computation."""
    if params is None:
        return (0.0, 0.0, 0)
    p = params
    if name == "MA":
        return bt_ma_ls(c, o, _rolling_mean(c, int(p[0])), _rolling_mean(c, int(p[1])),
                        sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "RSI":
        return bt_rsi_ls(c, o, _rsi_wilder(c, int(p[0])), float(p[1]), float(p[2]),
                         sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "MACD":
        return bt_macd_ls(c, o, _ema(c, int(p[0])), _ema(c, int(p[1])), int(p[2]),
                          sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Drift":
        return bt_drift_ls(c, o, int(p[0]), float(p[1]), int(p[2]),
                           sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "RAMOM":
        return bt_ramom_ls(c, o, int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                           sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Turtle":
        return bt_turtle_ls(c, o, h, l, int(p[0]), int(p[1]), int(p[2]), float(p[3]),
                            sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Bollinger":
        return bt_bollinger_ls(c, o, int(p[0]), float(p[1]),
                               sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Keltner":
        return bt_keltner_ls(c, o, h, l, int(p[0]), int(p[1]), float(p[2]),
                             sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "MultiFactor":
        return bt_multifactor_ls(c, o, int(p[0]), int(p[1]), int(p[2]), float(p[3]), float(p[4]),
                                 sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "VolRegime":
        ms_, ml_ = int(max(2, p[2])), int(max(5, p[3]))
        if ms_ >= ml_:
            return (0.0, 0.0, 0)
        return bt_volregime_ls(c, o, h, l, int(p[0]), float(p[1]), ms_, ml_, 14,
                               int(p[4]), int(p[5]), sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "MESA":
        return bt_mesa_ls(c, o, float(p[0]), float(p[1]),
                          sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "KAMA":
        return bt_kama_ls(c, o, h, l, int(p[0]), int(p[1]), int(p[2]), float(p[3]), int(p[4]),
                          sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Donchian":
        return bt_donchian_ls(c, o, h, l, int(p[0]), int(p[1]), float(p[2]),
                              sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "ZScore":
        return bt_zscore_ls(c, o, int(p[0]), float(p[1]), float(p[2]), float(p[3]),
                            sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "MomBreak":
        return bt_mombreak_ls(c, o, h, l, int(p[0]), float(p[1]), int(p[2]), float(p[3]),
                              sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "RegimeEMA":
        fe, se = int(max(2, p[2])), int(max(5, p[3]))
        if fe >= se:
            return (0.0, 0.0, 0)
        return bt_regime_ema_ls(c, o, h, l, int(max(5, p[0])), float(p[1]), fe, se, int(p[4]),
                                sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "DualMom":
        fl_, sl_ = int(max(2, p[0])), int(max(5, p[1]))
        if fl_ >= sl_:
            return (0.0, 0.0, 0)
        return bt_dualmom_ls(c, o, fl_, sl_,
                             sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    if name == "Consensus":
        ms_, ml_ = int(min(200, max(2, p[0]))), int(min(200, max(5, p[1])))
        if ms_ >= ml_:
            return (0.0, 0.0, 0)
        rp_ = int(min(200, max(2, p[2])))
        return bt_consensus_ls(c, o, _rolling_mean(c, ms_), _rolling_mean(c, ml_),
                               _rsi_wilder(c, rp_), int(p[3]), float(p[4]), float(p[5]),
                               int(p[6]), sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
    return (0.0, 0.0, 0)


# =====================================================================
#  Per-strategy scan functions — direct kernel calls, zero dispatch
#  overhead per param combo.  Each called once per strategy.
# =====================================================================

def _scan_ma(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_ma_ls(c,o,mas[int(p[0])],mas[int(p[1])],sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_rsi(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_rsi_ls(c,o,rsis[int(p[0])],float(p[1]),float(p[2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_macd(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_macd_ls(c,o,emas[int(p[0])],emas[int(p[1])],int(p[2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_drift(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_drift_ls(c,o,int(p[0]),float(p[1]),int(p[2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_ramom(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_ramom_ls(c,o,int(p[0]),int(p[1]),float(p[2]),float(p[3]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_turtle(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_turtle_ls(c,o,h,l,int(p[0]),int(p[1]),int(p[2]),float(p[3]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_bollinger(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_bollinger_ls(c,o,int(p[0]),float(p[1]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_keltner(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_keltner_ls(c,o,h,l,int(p[0]),int(p[1]),float(p[2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_multifactor(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_multifactor_ls(c,o,int(p[0]),int(p[1]),int(p[2]),float(p[3]),float(p[4]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_volregime(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_volregime_ls(c,o,h,l,int(p[0]),float(p[1]),int(p[2]),int(p[3]),14,int(p[4]),int(p[5]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_mesa(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_mesa_ls(c,o,float(p[0]),float(p[1]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_kama(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_kama_ls(c,o,h,l,int(p[0]),int(p[1]),int(p[2]),float(p[3]),int(p[4]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_donchian(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_donchian_ls(c,o,h,l,int(p[0]),int(p[1]),float(p[2]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_zscore(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_zscore_ls(c,o,int(p[0]),float(p[1]),float(p[2]),float(p[3]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_mombreak(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_mombreak_ls(c,o,h,l,int(p[0]),float(p[1]),int(p[2]),float(p[3]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_regime_ema(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_regime_ema_ls(c,o,h,l,int(p[0]),float(p[1]),int(p[2]),int(p[3]),int(p[4]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_dualmom(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        r,d,nt=bt_dualmom_ls(c,o,int(p[0]),int(p[1]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

def _scan_consensus(grid, c, o, h, l, mas, emas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    bs=-1e18; bp=None; br=0.; bd=0.; bn=0; cnt=0
    for p in grid:
        ms_,ml_ = int(p[0]),int(p[1])
        r,d,nt=bt_consensus_ls(c,o,mas[ms_],mas[ml_],rsis[int(p[2])],int(p[3]),float(p[4]),float(p[5]),int(p[6]),sb,ss,cm,lev,dc,sl,pfrac,sl_slip)
        sc=_score(r,d,nt); cnt+=1
        if sc>bs: bs=sc; bp=p; br=r; bd=d; bn=nt
    return dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

_SCANNERS: Dict[str, Callable] = {
    "MA": _scan_ma, "RSI": _scan_rsi, "MACD": _scan_macd,
    "Drift": _scan_drift, "RAMOM": _scan_ramom, "Turtle": _scan_turtle,
    "Bollinger": _scan_bollinger, "Keltner": _scan_keltner,
    "MultiFactor": _scan_multifactor, "VolRegime": _scan_volregime,
    "MESA": _scan_mesa, "KAMA": _scan_kama, "Donchian": _scan_donchian,
    "ZScore": _scan_zscore, "MomBreak": _scan_mombreak,
    "RegimeEMA": _scan_regime_ema, "DualMom": _scan_dualmom,
    "Consensus": _scan_consensus,
}


# =====================================================================
#  Full parameter grid scan — user-configurable
# =====================================================================

def scan_all_kernels(
    c: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray,
    config: BacktestConfig,
    *,
    param_grids: Optional[Dict[str, List[tuple]]] = None,
    strategies: Optional[List[str]] = None,
    mas: Optional[np.ndarray] = None,
    emas: Optional[np.ndarray] = None,
    rsis: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """Scan strategies over parameter grids. Returns best params per strategy.

    Args:
        c, o, h, l: OHLC numpy arrays (float64).
        config: BacktestConfig with costs/leverage/stop-loss.
        param_grids: Custom parameter grids per strategy.
                     e.g. {"MA": [(5,20),(10,50),(20,100)], "RSI": [(14,30,70)]}
                     If None, uses DEFAULT_PARAM_GRIDS.
        strategies: Which strategies to scan. If None, scans all 18.
        mas, emas, rsis: Pre-computed indicator arrays (optional, auto-computed if None).

    Returns:
        Dict mapping strategy name -> {params, score, ret, dd, nt, cnt}.
    """
    costs = config_to_kernel_costs(config)
    sb, ss_, cm = costs["sb"], costs["ss"], costs["cm"]
    lev, dc = costs["lev"], costs["dc"]
    sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

    if mas is None:
        mas = precompute_all_ma(c, 200)
    if emas is None:
        emas = precompute_all_ema(c, 200)
    if rsis is None:
        rsis = precompute_all_rsi(c, 200)

    grids = param_grids if param_grids is not None else DEFAULT_PARAM_GRIDS
    strat_names = strategies or [sn for sn in KERNEL_NAMES if sn in grids]

    R: Dict[str, dict] = {}
    for sn in strat_names:
        grid = grids.get(sn)
        scanner = _SCANNERS.get(sn)
        if not grid or not scanner:
            continue
        R[sn] = scanner(grid, c, o, h, l, mas, emas, rsis,
                        sb, ss_, cm, lev, dc, sl, pfrac, sl_slip)
    return R
