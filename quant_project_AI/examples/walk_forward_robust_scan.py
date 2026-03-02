#!/usr/bin/env python3
"""
==========================================================================
  10-Layer Anti-Overfitting Robust Scan  (V3)
==========================================================================
Layer 1 : Purged Walk-Forward with Train/Val/Test (6 windows, embargo gap)
Layer 2 : Multi-Metric Scoring (return / drawdown * trade_count_factor)
Layer 3 : Minimum Trade Filter (>= 20 trades)
Layer 4 : Parameter Stability (perturb +/-10%, +/-20%)
Layer 5 : Cross-Asset Validation (params from A tested on B,C,D,...)
Layer 6 : Monte Carlo Price Perturbation (30 noisy OHLC paths, sigma=0.2%)
Layer 7 : OHLC Shuffle Perturbation (20 paths, randomly reassign O/H/L/C roles)
Layer 8 : Block Bootstrap Resampling (20 paths, block_size=20 bars)
Layer 9 : Deflated Sharpe Ratio (corrects for multiple hypothesis testing)
Layer 10: Composite 8-dim Ranking (WFE + GenGap + Stability + MC + Shuffle
          + Bootstrap + DSR + Cross-Asset)

All 21 strategy kernels are Numba @njit compiled with enhanced tracking:
  - Mark-to-market equity for max drawdown
  - Round-trip trade counting
  - Composite scoring for parameter selection
"""
import numpy as np
import pandas as pd
import time, sys, os, warnings, math
from collections import defaultdict
from scipy import stats as sp_stats

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

# =====================================================================
#  Numba Helpers
# =====================================================================

@njit(cache=True)
def _ema(arr, span):
    n = len(arr); out = np.empty(n, dtype=np.float64)
    k = 2.0 / (span + 1.0); out[0] = arr[0]
    for i in range(1, n): out[i] = arr[i] * k + out[i-1] * (1.0 - k)
    return out

@njit(cache=True)
def _rolling_mean(arr, w):
    n = len(arr); out = np.full(n, np.nan); s = 0.0
    for i in range(n):
        s += arr[i]
        if i >= w: s -= arr[i - w]
        if i >= w - 1: out[i] = s / w
    return out

@njit(cache=True)
def _rolling_std(arr, w):
    n = len(arr); out = np.full(n, np.nan); s = 0.0; s2 = 0.0
    for i in range(n):
        s += arr[i]; s2 += arr[i] * arr[i]
        if i >= w: s -= arr[i-w]; s2 -= arr[i-w] * arr[i-w]
        if i >= w - 1:
            m = s / w; out[i] = np.sqrt(max(0.0, s2/w - m*m))
    return out

@njit(cache=True)
def _atr(high, low, close, period):
    n = len(close); tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], max(abs(high[i]-close[i-1]), abs(low[i]-close[i-1])))
    out = np.full(n, np.nan); s = 0.0
    for i in range(period): s += tr[i]
    out[period-1] = s / period
    for i in range(period, n): out[i] = (out[i-1]*(period-1)+tr[i]) / period
    return out

@njit(cache=True)
def _rsi_wilder(close, period):
    n = len(close); out = np.full(n, np.nan)
    if n < period + 1: return out
    gs = 0.0; ls = 0.0
    for i in range(1, period+1):
        d = close[i]-close[i-1]
        if d > 0: gs += d
        else: ls -= d
    ag = gs/period; al = ls/period
    out[period] = 100.0 if al == 0 else 100.0 - 100.0/(1+ag/al)
    for i in range(period+1, n):
        d = close[i]-close[i-1]
        g = d if d > 0 else 0.0; l = -d if d < 0 else 0.0
        ag = (ag*(period-1)+g)/period; al = (al*(period-1)+l)/period
        out[i] = 100.0 if al == 0 else 100.0 - 100.0/(1+ag/al)
    return out

# =====================================================================
#  Enhanced Fill + MTM + Score
# =====================================================================

@njit(cache=True)
def _fx(pend, pos, ep, oi, tr, sb, ss, cm):
    """Fill pending order, return (pos, ep, tr, trades_closed)."""
    if pend == 0: return pos, ep, tr, 0
    tc = 0
    if abs(pend) >= 2:
        if pos == 1: tr *= (oi*ss*(1.0-cm))/(ep*(1.0+cm)); tc = 1
        elif pos == -1: tr *= (ep*(1.0-cm))/(oi*sb*(1.0+cm)); tc = 1
        pos = 0
    if pend == 1 or pend == 3: ep = oi*sb; pos = 1
    elif pend == -1 or pend == -3: ep = oi*ss; pos = -1
    return pos, ep, tr, tc

@njit(cache=True)
def _mtm(pos, tr, ci, ep, sb, ss, cm):
    """Mark-to-market equity."""
    if pos == 1 and ep > 0: return tr*(ci*ss*(1-cm))/(ep*(1+cm))
    if pos == -1 and ep > 0: return tr*(ep*(1-cm))/(ci*sb*(1+cm))
    return tr

@njit(cache=True)
def _score(ret, dd, nt):
    """Composite score: return/drawdown * trade_count_factor."""
    return ret / max(1.0, dd) * min(1.0, nt / 20.0)

# =====================================================================
#  Monte Carlo Price Perturbation
# =====================================================================

@njit(cache=True)
def perturb_ohlc(close, open_, high, low, noise_std, seed):
    """Generate perturbed OHLC maintaining validity: low<=min(o,c), high>=max(o,c)."""
    n = len(close)
    np.random.seed(seed)
    c_p = np.empty(n, np.float64)
    o_p = np.empty(n, np.float64)
    h_p = np.empty(n, np.float64)
    l_p = np.empty(n, np.float64)
    for i in range(n):
        nc = 1.0 + np.random.randn() * noise_std
        no = 1.0 + np.random.randn() * noise_std
        c_p[i] = close[i] * nc
        o_p[i] = open_[i] * no
        body_hi = max(c_p[i], o_p[i])
        body_lo = min(c_p[i], o_p[i])
        uw = max(0.0, high[i] - max(close[i], open_[i]))
        lw = max(0.0, min(close[i], open_[i]) - low[i])
        h_p[i] = body_hi + uw * (1.0 + np.random.randn() * noise_std * 0.5)
        l_p[i] = body_lo - lw * (1.0 + np.random.randn() * noise_std * 0.5)
    return c_p, o_p, h_p, l_p


@njit(cache=True)
def shuffle_ohlc(close, open_, high, low, seed):
    """Randomly reassign OHLC roles within each bar.
    For each bar, the 4 prices {O,H,L,C} are randomly permuted, then forced
    into valid OHLC: max->H, min->L, first of remaining two->O, second->C.
    This tests whether the strategy relies on genuine price structure or just
    on the specific labelling of which price is 'close' vs 'open'.
    """
    n = len(close)
    np.random.seed(seed)
    c_p = np.empty(n, np.float64)
    o_p = np.empty(n, np.float64)
    h_p = np.empty(n, np.float64)
    l_p = np.empty(n, np.float64)
    for i in range(n):
        vals = np.empty(4, np.float64)
        vals[0] = open_[i]; vals[1] = high[i]; vals[2] = low[i]; vals[3] = close[i]
        for j in range(3, 0, -1):
            k = int(np.random.random() * (j + 1))
            if k > j: k = j
            vals[j], vals[k] = vals[k], vals[j]
        mx = vals[0]; mn = vals[0]
        mx_i = 0; mn_i = 0
        for j in range(1, 4):
            if vals[j] > mx: mx = vals[j]; mx_i = j
            if vals[j] < mn: mn = vals[j]; mn_i = j
        h_p[i] = mx; l_p[i] = mn
        rem = np.empty(2, np.float64); ri = 0
        for j in range(4):
            if j != mx_i and j != mn_i and ri < 2:
                rem[ri] = vals[j]; ri += 1
        if ri < 2:
            rem[ri] = vals[3]
        o_p[i] = rem[0]; c_p[i] = rem[1]
    return c_p, o_p, h_p, l_p


@njit(cache=True)
def block_bootstrap_ohlc(close, open_, high, low, block_size, seed):
    """Block bootstrap: resample contiguous blocks of bars with replacement.
    Preserves local autocorrelation structure within each block while
    creating a new price path of the same length.
    """
    n = len(close)
    np.random.seed(seed)
    n_blocks = (n + block_size - 1) // block_size
    max_start = n - block_size
    if max_start < 0: max_start = 0
    c_p = np.empty(n, np.float64)
    o_p = np.empty(n, np.float64)
    h_p = np.empty(n, np.float64)
    l_p = np.empty(n, np.float64)
    idx = 0
    scale = 1.0
    last_c = close[0]
    for _ in range(n_blocks):
        start = int(np.random.random() * (max_start + 1))
        if start > max_start: start = max_start
        if start > 0:
            scale = last_c / close[start - 1] if close[start - 1] > 0 else 1.0
        else:
            scale = 1.0
        for j in range(block_size):
            if idx >= n: break
            si = start + j
            if si >= n: si = n - 1
            c_p[idx] = close[si] * scale
            o_p[idx] = open_[si] * scale
            h_p[idx] = high[si] * scale
            l_p[idx] = low[si] * scale
            last_c = c_p[idx]
            idx += 1
    return c_p, o_p, h_p, l_p


def eval_strategy_mc(name, params, c, o, h, l, sb, ss, cm):
    """Evaluate strategy on arbitrary OHLC, computing only needed indicators."""
    if params is None:
        return (0.0, 0.0, 0)
    try:
        p = params
        if name == "MA":
            return bt_ma_wf(c, o, _rolling_mean(c, int(p[0])),
                            _rolling_mean(c, int(p[1])), sb, ss, cm)
        elif name == "RSI":
            return bt_rsi_wf(c, o, _rsi_wilder(c, int(p[0])),
                             float(p[1]), float(p[2]), sb, ss, cm)
        elif name == "MACD":
            return bt_macd_wf(c, o, _ema(c, int(p[0])),
                              _ema(c, int(p[1])), int(p[2]), sb, ss, cm)
        elif name == "Drift":
            return bt_drift_wf(c, o, int(p[0]), float(p[1]),
                               int(p[2]), sb, ss, cm)
        elif name == "RAMOM":
            return bt_ramom_wf(c, o, int(p[0]), int(p[1]),
                               float(p[2]), float(p[3]), sb, ss, cm)
        elif name == "Turtle":
            return bt_turtle_wf(c, o, h, l, int(p[0]), int(p[1]),
                                int(p[2]), float(p[3]), sb, ss, cm)
        elif name == "Bollinger":
            return bt_bollinger_wf(c, o, int(p[0]), float(p[1]), sb, ss, cm)
        elif name == "Keltner":
            return bt_keltner_wf(c, o, h, l, int(p[0]), int(p[1]),
                                 float(p[2]), sb, ss, cm)
        elif name == "MultiFactor":
            return bt_multifactor_wf(c, o, int(p[0]), int(p[1]),
                                     int(p[2]), float(p[3]), float(p[4]),
                                     sb, ss, cm)
        elif name == "VolRegime":
            return bt_volregime_wf(c, o, h, l, int(p[0]), float(p[1]),
                                   int(p[2]), int(p[3]), 14,
                                   int(p[4]), int(p[5]), sb, ss, cm)
        elif name == "Connors":
            return bt_connors_wf(c, o, int(p[0]), int(p[1]), int(p[2]),
                                 float(p[3]), float(p[4]), sb, ss, cm)
        elif name == "MESA":
            return bt_mesa_wf(c, o, float(p[0]), float(p[1]), sb, ss, cm)
        elif name == "KAMA":
            return bt_kama_wf(c, o, h, l, int(p[0]), int(p[1]),
                              int(p[2]), float(p[3]), int(p[4]), sb, ss, cm)
        elif name == "Donchian":
            return bt_donchian_wf(c, o, h, l, int(p[0]), int(p[1]),
                                  float(p[2]), sb, ss, cm)
        elif name == "ZScore":
            return bt_zscore_wf(c, o, int(p[0]), float(p[1]),
                                float(p[2]), float(p[3]), sb, ss, cm)
        elif name == "MomBreak":
            return bt_mombreak_wf(c, o, h, l, int(p[0]), float(p[1]),
                                  int(p[2]), float(p[3]), sb, ss, cm)
        elif name == "RegimeEMA":
            return bt_regime_ema_wf(c, o, h, l, int(p[0]), float(p[1]),
                                    int(p[2]), int(p[3]), int(p[4]),
                                    sb, ss, cm)
        elif name == "TFiltRSI":
            return bt_tfiltrsi_wf(c, o, _rsi_wilder(c, int(p[0])),
                                  _rolling_mean(c, int(p[1])),
                                  float(p[2]), float(p[3]), sb, ss, cm)
        elif name == "MomBrkPlus":
            return bt_mombrkplus_wf(c, o, h, l, int(p[0]), float(p[1]),
                                    int(p[2]), float(p[3]), int(p[4]),
                                    sb, ss, cm)
        elif name == "DualMom":
            return bt_dualmom_wf(c, o, int(p[0]), int(p[1]), sb, ss, cm)
        elif name == "Consensus":
            return bt_consensus_wf(c, o,
                                   _rolling_mean(c, int(p[0])),
                                   _rolling_mean(c, int(p[1])),
                                   _rsi_wilder(c, int(p[2])),
                                   int(p[3]), float(p[4]), float(p[5]),
                                   int(p[6]), sb, ss, cm)
    except Exception:
        pass
    return (0.0, 0.0, 0)

# =====================================================================
#  Precomputation
# =====================================================================

def precompute_all_ma(close, max_w=200):
    n = len(close); cs = np.empty(n+1, dtype=np.float64); cs[0] = 0.0
    np.cumsum(close, out=cs[1:])
    mas = np.full((max_w+1, n), np.nan, dtype=np.float64)
    for w in range(2, min(max_w+1, n+1)):
        mas[w, w-1:] = (cs[w:] - cs[:n-w+1]) / w
    return mas

def precompute_all_ema(close, max_s=200):
    n = len(close); emas = np.full((max_s+1, n), np.nan, dtype=np.float64)
    for s in range(2, max_s+1):
        k = 2.0/(s+1.0); e = np.empty(n, dtype=np.float64); e[0] = close[0]
        for i in range(1, n): e[i] = close[i]*k + e[i-1]*(1.0-k)
        emas[s] = e
    return emas

def precompute_all_rsi(close, max_p=200):
    n = len(close); d = np.diff(close, prepend=close[0])
    g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    rsi = np.full((max_p+1, n), np.nan, dtype=np.float64)
    for p in range(2, max_p+1):
        if n <= p: continue
        ag = np.mean(g[1:p+1]); al = np.mean(l[1:p+1])
        for i in range(p, n):
            if i > p: ag = (ag*(p-1)+g[i])/p; al = (al*(p-1)+l[i])/p
            rsi[p, i] = 100.0 if al == 0 else 100.0 - 100.0/(1+ag/al)
    return rsi

# =====================================================================
#  17 Enhanced Kernels — return (ret_pct, max_dd_pct, n_trades)
# =====================================================================

@njit(cache=True)
def bt_ma_wf(close, open_, ma_s, ma_l, sb, ss, cm):
    n = len(close); pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(1, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        s0=ma_s[i-1]; l0=ma_l[i-1]; s1=ma_s[i]; l1=ma_l[i]
        if s0!=s0 or l0!=l0 or s1!=s1 or l1!=l1: pass
        elif s0<=l0 and s1>l1 and pos==0: pend=1
        elif s0>=l0 and s1<l1 and pos==1: pend=2
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_rsi_wf(close, open_, rsi, os_v, ob_v, sb, ss, cm):
    n = len(close); pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(1, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        r = rsi[i]
        if r!=r: pass
        elif r<os_v and pos==0: pend=1
        elif r>ob_v and pos==1: pend=2
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_macd_wf(close, open_, ef, es, sig_span, sb, ss, cm):
    n = len(close); ml=np.empty(n); sl=np.empty(n)
    for i in range(n): ml[i]=ef[i]-es[i]
    k=2.0/(sig_span+1.0); sl[0]=ml[0]
    for i in range(1,n): sl[i]=ml[i]*k+sl[i-1]*(1-k)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(1, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        mp=ml[i-1]; sp=sl[i-1]; mc=ml[i]; sc=sl[i]
        if mp!=mp or sp!=sp or mc!=mc or sc!=sc: pass
        elif mp<=sp and mc>sc and pos==0: pend=1
        elif mp>=sp and mc<sc and pos==1: pend=2
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_drift_wf(close, open_, lookback, drift_thr, hold_p, sb, ss, cm):
    n = len(close); pos=0; ep=0.0; tr=1.0; pend=0; hc=0; pk=1.0; mdd=0.0; nt=0
    for i in range(lookback, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        up=0
        for j in range(1, lookback+1):
            if close[i-j+1]>close[i-j]: up+=1
        ratio=up/lookback
        if pos!=0:
            hc+=1
            if hc>=hold_p: pend=2; hc=0
        if pos==0 and pend==0:
            if ratio>=drift_thr: pend=-1; hc=0
            elif ratio<=1.0-drift_thr: pend=1; hc=0
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_ramom_wf(close, open_, mom_p, vol_p, ez, xz, sb, ss, cm):
    n = len(close); pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(mom_p, vol_p)
    for i in range(start, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        mom=(close[i]/close[i-mom_p])-1.0
        s=0.0; s2=0.0
        for j in range(vol_p):
            r=(close[i-j]/close[i-j-1]-1.0) if i-j>0 else 0.0
            s+=r; s2+=r*r
        m=s/vol_p; vol=np.sqrt(max(1e-20, s2/vol_p-m*m))
        z=mom/vol
        if pos==0:
            if z>ez: pend=1
            elif z<-ez: pend=-1
        elif pos==1 and z<xz: pend=2
        elif pos==-1 and z>-xz: pend=2
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_turtle_wf(close, open_, high, low, entry_p, exit_p, atr_p, atr_stop, sb, ss, cm):
    n = len(close); aa=_atr(high,low,close,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(entry_p, exit_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        eh=-1e18; el=1e18
        for j in range(1, entry_p+1):
            if high[i-j]>eh: eh=high[i-j]
            if low[i-j]<el: el=low[i-j]
        xl=1e18; xh=-1e18
        for j in range(1, exit_p+1):
            if low[i-j]<xl: xl=low[i-j]
            if high[i-j]>xh: xh=high[i-j]
        a=aa[i]
        if a!=a: pass
        else:
            if pos==1:
                if close[i]<ep/sb-atr_stop*a or close[i]<xl: pend=2
            elif pos==-1:
                if close[i]>ep/ss+atr_stop*a or close[i]>xh: pend=2
            if pos==0 and pend==0:
                if close[i]>eh: pend=1
                elif close[i]<el: pend=-1
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_bollinger_wf(close, open_, period, num_std, sb, ss, cm):
    n = len(close); ma=_rolling_mean(close,period); sd=_rolling_std(close,period)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(period, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        m=ma[i]; s=sd[i]
        if m!=m or s!=s or s<1e-10: pass
        else:
            u=m+num_std*s; lo=m-num_std*s
            if pos==0:
                if close[i]<lo: pend=1
                elif close[i]>u: pend=-1
            elif pos==1 and close[i]>=m: pend=2
            elif pos==-1 and close[i]<=m: pend=2
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_keltner_wf(close, open_, high, low, ema_p, atr_p, atr_m, sb, ss, cm):
    n = len(close); ea=_ema(close,ema_p); aa=_atr(high,low,close,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(ema_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        e=ea[i]; a=aa[i]
        if e!=e or a!=a: pass
        else:
            if pos==0:
                if close[i]>e+atr_m*a: pend=1
                elif close[i]<e-atr_m*a: pend=-1
            elif pos==1 and close[i]<e: pend=2
            elif pos==-1 and close[i]>e: pend=2
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_multifactor_wf(close, open_, rsi_p, mom_p, vol_p, lt, st, sb, ss, cm):
    n = len(close); rsi=_rsi_wilder(close,rsi_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(rsi_p+1, mom_p, vol_p)
    for i in range(start, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        r=rsi[i]
        if r!=r: pass
        else:
            rs=(100.0-r)/100.0
            mom=(close[i]/close[i-mom_p])-1.0
            ms=max(-0.5, min(0.5, mom))+0.5
            s2=0.0
            for j in range(vol_p):
                ret=(close[i-j]/close[i-j-1]-1.0) if i-j>0 else 0.0
                s2+=ret*ret
            vs=max(0.0, 1.0-np.sqrt(s2/vol_p)*20.0)
            comp=(rs+ms+vs)/3.0
            if pos==0:
                if comp>lt: pend=1
                elif comp<st: pend=-1
            elif pos==1 and comp<0.5: pend=2
            elif pos==-1 and comp>0.5: pend=2
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_volregime_wf(close, open_, high, low, atr_p, vol_thr, ma_s, ma_l, rsi_p, rsi_os, rsi_ob, sb, ss, cm):
    n = len(close); aa=_atr(high,low,close,atr_p)
    ra=_rsi_wilder(close,rsi_p); ms_a=_rolling_mean(close,ma_s); ml_a=_rolling_mean(close,ma_l)
    pos=0; ep=0.0; tr=1.0; pend=0; mode=0; pk=1.0; mdd=0.0; nt=0
    start=max(atr_p, rsi_p+1, ma_l)
    for i in range(start, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        a=aa[i]
        if a!=a or close[i]<=0: pass
        else:
            hv=a/close[i]>vol_thr
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
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_connors_wf(close, open_, rsi_p, mat, mae, os_t, ob_t, sb, ss, cm):
    n = len(close)
    rsi=np.full(n, np.nan)
    if n>rsi_p:
        gs=0.0; ls=0.0
        for i in range(1,rsi_p+1):
            d=close[i]-close[i-1]
            if d>0: gs+=d
            else: ls-=d
        ag=gs/rsi_p; al=ls/rsi_p
        rsi[rsi_p]=100.0 if al==0 else 100.0-100.0/(1+ag/al)
        for i in range(rsi_p+1,n):
            d=close[i]-close[i-1]; g=d if d>0 else 0.0; l=-d if d<0 else 0.0
            ag=(ag*(rsi_p-1)+g)/rsi_p; al=(al*(rsi_p-1)+l)/rsi_p
            rsi[i]=100.0 if al==0 else 100.0-100.0/(1+ag/al)
    mt=np.full(n,np.nan); s=0.0
    for i in range(n):
        s+=close[i]
        if i>=mat: s-=close[i-mat]
        if i>=mat-1: mt[i]=s/mat
    me=np.full(n,np.nan); s=0.0
    for i in range(n):
        s+=close[i]
        if i>=mae: s-=close[i-mae]
        if i>=mae-1: me[i]=s/mae
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(rsi_p+1,mat,mae)
    for i in range(start, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        r=rsi[i]; t_=mt[i]; e_=me[i]
        if r!=r or t_!=t_ or e_!=e_: pass
        else:
            if pos==1 and close[i]>e_: pend=2
            elif pos==-1 and close[i]<e_: pend=2
            if pos==0 and pend==0:
                if close[i]>t_ and r<os_t: pend=1
                elif close[i]<t_ and r>ob_t: pend=-1
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_mesa_wf(close, open_, fl, sl, sb, ss, cm):
    n = len(close)
    if n<40: return 0.0, 0.0, 0
    smooth=np.zeros(n); det=np.zeros(n); I1=np.zeros(n); Q1=np.zeros(n)
    jI=np.zeros(n); jQ=np.zeros(n); I2=np.zeros(n); Q2=np.zeros(n)
    Re_=np.zeros(n); Im_=np.zeros(n); per=np.zeros(n); sp_=np.zeros(n)
    ph=np.zeros(n); mama=np.zeros(n); fama=np.zeros(n)
    for i in range(min(6,n)):
        mama[i]=close[i]; fama[i]=close[i]; per[i]=6.0; sp_[i]=6.0
    for i in range(6, n):
        smooth[i]=(4*close[i]+3*close[i-1]+2*close[i-2]+close[i-3])/10.0
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
        if alpha<sl: alpha=sl
        if alpha>fl: alpha=fl
        mama[i]=alpha*close[i]+(1-alpha)*mama[i-1]
        fama[i]=0.5*alpha*mama[i]+(1-0.5*alpha)*fama[i-1]
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(7, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        if pos==0:
            if mama[i]>fama[i] and mama[i-1]<=fama[i-1]: pend=1
            elif mama[i]<fama[i] and mama[i-1]>=fama[i-1]: pend=-1
        elif pos==1 and mama[i]<fama[i] and mama[i-1]>=fama[i-1]: pend=-3
        elif pos==-1 and mama[i]>fama[i] and mama[i-1]<=fama[i-1]: pend=3
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_kama_wf(close, open_, high, low, er_p, fast_sc, slow_sc, atr_sm, atr_p, sb, ss, cm):
    n = len(close)
    if n<er_p+2: return 0.0, 0.0, 0
    fc=2.0/(fast_sc+1.0); sc_v=2.0/(slow_sc+1.0)
    kama=np.full(n,np.nan); kama[er_p-1]=close[er_p-1]
    for i in range(er_p, n):
        d=abs(close[i]-close[i-er_p]); v=0.0
        for j in range(1, er_p+1): v+=abs(close[i-j+1]-close[i-j])
        er=d/v if v>0 else 0.0; sc2=(er*(fc-sc_v)+sc_v)**2
        kama[i]=kama[i-1]+sc2*(close[i]-kama[i-1])
    av=_atr(high,low,close,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(er_p+2, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        k=kama[i]; kp=kama[i-1]; a=av[i]
        if k!=k or kp!=kp or a!=a: pass
        else:
            if pos==1:
                if close[i]<ep/sb-atr_sm*a or k<kp: pend=2
            elif pos==-1:
                if close[i]>ep/ss+atr_sm*a or k>kp: pend=2
            if pos==0 and pend==0:
                if close[i]>k and k>kp: pend=1
                elif close[i]<k and k<kp: pend=-1
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_donchian_wf(close, open_, high, low, entry_p, atr_p, atr_m, sb, ss, cm):
    n = len(close)
    if n<entry_p+atr_p: return 0.0, 0.0, 0
    av=_atr(high,low,close,atr_p)
    dh=np.full(n,np.nan); dl=np.full(n,np.nan)
    for i in range(entry_p-1, n):
        mx=high[i]; mn=low[i]
        for j in range(1, entry_p):
            if high[i-j]>mx: mx=high[i-j]
            if low[i-j]<mn: mn=low[i-j]
        dh[i]=mx; dl[i]=mn
    pos=0; ep=0.0; tr=1.0; pend=0; ts=0.0; pk=1.0; mdd=0.0; nt=0
    start=max(entry_p, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc
        if pend==1 and av[i]==av[i]: ts=open_[i]-atr_m*av[i]
        elif pend==-1 and av[i]==av[i]: ts=open_[i]+atr_m*av[i]
        pend=0
        d1=dh[i-1]; d2=dl[i-1]; a=av[i]
        if d1!=d1 or d2!=d2 or a!=a: pass
        else:
            if pos==1:
                ns=close[i]-atr_m*a
                if ns>ts: ts=ns
                if close[i]<ts: pend=2
            elif pos==-1:
                ns=close[i]+atr_m*a
                if ns<ts: ts=ns
                if close[i]>ts: pend=2
            if pos==0 and pend==0:
                if close[i]>d1: pend=1
                elif close[i]<d2: pend=-1
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_zscore_wf(close, open_, lookback, ez, xz, sz, sb, ss, cm):
    n = len(close)
    if n<lookback+2: return 0.0, 0.0, 0
    rm=np.full(n,np.nan); rs=np.full(n,np.nan); s=0.0; s2=0.0
    for i in range(n):
        s+=close[i]; s2+=close[i]*close[i]
        if i>=lookback: s-=close[i-lookback]; s2-=close[i-lookback]*close[i-lookback]
        if i>=lookback-1: m=s/lookback; rm[i]=m; rs[i]=np.sqrt(max(0.0,s2/lookback-m*m))
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    for i in range(lookback, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        m=rm[i]; sd=rs[i]
        if sd==0 or sd!=sd: pass
        else:
            z=(close[i]-m)/sd
            if pos==1 and (z>-xz or z>sz): pend=2
            elif pos==-1 and (z<xz or z<-sz): pend=2
            if pos==0 and pend==0:
                if z<-ez: pend=1
                elif z>ez: pend=-1
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_mombreak_wf(close, open_, high, low, hp, prox, atr_p, atr_t, sb, ss, cm):
    n = len(close)
    if n<max(hp, atr_p)+2: return 0.0, 0.0, 0
    rh=np.full(n,np.nan)
    for i in range(hp-1, n):
        mx=high[i]
        for j in range(1, hp):
            if high[i-j]>mx: mx=high[i-j]
        rh[i]=mx
    av=_atr(high,low,close,atr_p)
    pos=0; ep=0.0; tr=1.0; pend=0; ts=0.0; pk=1.0; mdd=0.0; nt=0
    start=max(hp, atr_p)
    for i in range(start, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc
        if pend==1 and av[i]==av[i]: ts=open_[i]-atr_t*av[i]
        pend=0
        h=rh[i]; a=av[i]
        if h!=h or a!=a: pass
        else:
            if pos==1:
                ns=close[i]-atr_t*a
                if ns>ts: ts=ns
                if close[i]<ts: pend=2
            if pos==0 and pend==0 and close[i]>=h*(1.0-prox): pend=1
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

@njit(cache=True)
def bt_regime_ema_wf(close, open_, high, low, atr_p, vt, fe_p, se_p, te_p, sb, ss, cm):
    n = len(close)
    if n<max(atr_p, max(se_p, te_p))+2: return 0.0, 0.0, 0
    av=_atr(high,low,close,atr_p)
    fk=2.0/(fe_p+1.0); sk=2.0/(se_p+1.0); tk=2.0/(te_p+1.0)
    ef=np.empty(n); es=np.empty(n); et=np.empty(n)
    ef[0]=close[0]; es[0]=close[0]; et[0]=close[0]
    for i in range(1, n):
        ef[i]=close[i]*fk+ef[i-1]*(1-fk)
        es[i]=close[i]*sk+es[i-1]*(1-sk)
        et[i]=close[i]*tk+et[i-1]*(1-tk)
    pos=0; ep=0.0; tr=1.0; pend=0; pk=1.0; mdd=0.0; nt=0
    start=max(atr_p, max(se_p, te_p))
    for i in range(start, n):
        pos,ep,tr,tc=_fx(pend,pos,ep,open_[i],tr,sb,ss,cm); nt+=tc; pend=0
        a=av[i]
        if a!=a or close[i]<=0: pass
        else:
            hv=a/close[i]>vt
            if hv:
                if pos==0:
                    if ef[i]>es[i] and ef[i-1]<=es[i-1]: pend=1
                    elif ef[i]<es[i] and ef[i-1]>=es[i-1]: pend=-1
                elif pos==1 and ef[i]<es[i]: pend=2
                elif pos==-1 and ef[i]>es[i]: pend=2
            else:
                pd_=(close[i]-et[i])/et[i]
                if pos==0:
                    if pd_<-0.02: pend=1
                    elif pd_>0.02: pend=-1
                elif pos==1 and pd_>0.0: pend=2
                elif pos==-1 and pd_<0.0: pend=2
        eq=_mtm(pos,tr,close[i],ep,sb,ss,cm)
        if eq>pk: pk=eq
        dd_=(pk-eq)/pk*100.0 if pk>0 else 0.0
        if dd_>mdd: mdd=dd_
    if pos==1: tr*=(close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt+=1
    elif pos==-1: tr*=(ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt+=1
    return (tr-1.0)*100.0, mdd, nt

# =====================================================================
#  New Strategy 18: Trend-Filtered RSI
#  Buy when RSI oversold AND price > trend MA; sell when RSI overbought AND price < trend MA.
#  Combines mean-reversion entry with trend confirmation to reduce whipsaws.
# =====================================================================

@njit(cache=True)
def bt_tfiltrsi_wf(close, open_, rsi_arr, trend_ma, os_thr, ob_thr, sb, ss, cm):
    n = len(close)
    if n < 5: return 0.0, 0.0, 0
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    for i in range(1, n):
        pos, ep, tr, tc = _fx(pend, pos, ep, open_[i], tr, sb, ss, cm); nt += tc; pend = 0
        r = rsi_arr[i]; m = trend_ma[i]
        if r != r or m != m: continue
        if pos == 0:
            if r < os_thr and close[i] > m: pend = 1
            elif r > ob_thr and close[i] < m: pend = -1
        elif pos == 1:
            if r > ob_thr or close[i] < m: pend = 2
        elif pos == -1:
            if r < os_thr or close[i] > m: pend = 2
        eq = _mtm(pos, tr, close[i], ep, sb, ss, cm)
        if eq > pk: pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd: mdd = dd_
    if pos == 1: tr *= (close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt += 1
    elif pos == -1: tr *= (ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt += 1
    return (tr - 1.0) * 100.0, mdd, nt

# =====================================================================
#  New Strategy 19: MomBreak Plus
#  Enhanced momentum breakout: requires ATR filter + volume confirmation (via
#  price range as proxy) + higher-high/lower-low confirmation before entering.
# =====================================================================

@njit(cache=True)
def bt_mombrkplus_wf(close, open_, high, low, look, pct_th, atr_p, atr_m, conf_bars, sb, ss, cm):
    n = len(close)
    if n < max(look, atr_p) + conf_bars + 2: return 0.0, 0.0, 0
    av = _atr(high, low, close, atr_p)
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    start = max(look, atr_p) + conf_bars
    for i in range(start, n):
        pos, ep, tr, tc = _fx(pend, pos, ep, open_[i], tr, sb, ss, cm); nt += tc; pend = 0
        a = av[i]
        if a != a or close[i] <= 0: continue
        hh = close[i-look]
        ll = close[i-look]
        for j in range(i-look+1, i+1):
            if high[j] > hh: hh = high[j]
            if low[j] < ll: ll = low[j]
        ret_up = (close[i] - ll) / ll if ll > 0 else 0.0
        ret_dn = (hh - close[i]) / hh if hh > 0 else 0.0
        confirmed_up = True
        confirmed_dn = True
        for cb in range(1, conf_bars + 1):
            if close[i - cb] >= close[i - cb + 1]: confirmed_up = False
            if close[i - cb] <= close[i - cb + 1]: confirmed_dn = False
        if pos == 0:
            if ret_up > pct_th and a > 0 and confirmed_up: pend = 1
            elif ret_dn > pct_th and a > 0 and confirmed_dn: pend = -1
        elif pos == 1:
            if close[i] < ep / (sb * (1 + cm)) - atr_m * a: pend = 2
        elif pos == -1:
            if close[i] > ep * sb * (1 + cm) + atr_m * a: pend = 2
        eq = _mtm(pos, tr, close[i], ep, sb, ss, cm)
        if eq > pk: pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd: mdd = dd_
    if pos == 1: tr *= (close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt += 1
    elif pos == -1: tr *= (ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt += 1
    return (tr - 1.0) * 100.0, mdd, nt

# =====================================================================
#  New Strategy 20: Dual Momentum
#  Absolute + relative momentum: go long when return > 0 over lookback AND
#  return > risk-free (proxy: 0), stay flat otherwise. Exit when abs momentum
#  turns negative.  Classic Antonacci style, adapted for next-open.
# =====================================================================

@njit(cache=True)
def bt_dualmom_wf(close, open_, fast_lb, slow_lb, sb, ss, cm):
    n = len(close)
    lb = max(fast_lb, slow_lb)
    if n < lb + 2: return 0.0, 0.0, 0
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    for i in range(lb, n):
        pos, ep, tr, tc = _fx(pend, pos, ep, open_[i], tr, sb, ss, cm); nt += tc; pend = 0
        fast_ret = (close[i] - close[i - fast_lb]) / close[i - fast_lb]
        slow_ret = (close[i] - close[i - slow_lb]) / close[i - slow_lb]
        if pos == 0:
            if fast_ret > 0 and slow_ret > 0: pend = 1
            elif fast_ret < 0 and slow_ret < 0: pend = -1
        elif pos == 1:
            if fast_ret < 0 or slow_ret < 0: pend = 2
        elif pos == -1:
            if fast_ret > 0 or slow_ret > 0: pend = 2
        eq = _mtm(pos, tr, close[i], ep, sb, ss, cm)
        if eq > pk: pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd: mdd = dd_
    if pos == 1: tr *= (close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt += 1
    elif pos == -1: tr *= (ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt += 1
    return (tr - 1.0) * 100.0, mdd, nt

# =====================================================================
#  New Strategy 21: Consensus (Multi-Signal Voting)
#  Combines MA crossover + RSI + momentum into a voting system. Each
#  indicator casts a vote (+1 long, -1 short, 0 neutral). Enter when
#  vote sum >= threshold.  Designed for stability by requiring agreement.
# =====================================================================

@njit(cache=True)
def bt_consensus_wf(close, open_, ma_s_arr, ma_l_arr, rsi_arr, mom_lb,
                    rsi_os, rsi_ob, vote_thr, sb, ss, cm):
    n = len(close)
    if n < mom_lb + 2: return 0.0, 0.0, 0
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    for i in range(mom_lb, n):
        pos, ep, tr, tc = _fx(pend, pos, ep, open_[i], tr, sb, ss, cm); nt += tc; pend = 0
        v = 0
        ms = ma_s_arr[i]; ml = ma_l_arr[i]; r = rsi_arr[i]
        if ms != ms or ml != ml or r != r: continue
        if ms > ml: v += 1
        elif ms < ml: v -= 1
        if r < rsi_os: v += 1
        elif r > rsi_ob: v -= 1
        mom_ret = (close[i] - close[i - mom_lb]) / close[i - mom_lb] if close[i - mom_lb] > 0 else 0.0
        if mom_ret > 0.02: v += 1
        elif mom_ret < -0.02: v -= 1
        if pos == 0:
            if v >= vote_thr: pend = 1
            elif v <= -vote_thr: pend = -1
        elif pos == 1:
            if v <= -1: pend = 2
        elif pos == -1:
            if v >= 1: pend = 2
        eq = _mtm(pos, tr, close[i], ep, sb, ss, cm)
        if eq > pk: pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd: mdd = dd_
    if pos == 1: tr *= (close[n-1]*ss*(1-cm))/(ep*(1+cm)); nt += 1
    elif pos == -1: tr *= (ep*(1-cm))/(close[n-1]*sb*(1+cm)); nt += 1
    return (tr - 1.0) * 100.0, mdd, nt

# =====================================================================
#  Scan All 21 Strategies
# =====================================================================

def scan_all(c, o, h, l, mas, emas, rsis, sb, ss, cm):
    """Scan all 21 strategies on given data. Returns {name: {params, score, ret, dd, nt, cnt}}."""
    R = {}

    # 1. MA Crossover
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for s in range(2,200):
        for lg in range(s+1,201):
            r,d,n=bt_ma_wf(c,o,mas[s],mas[lg],sb,ss,cm)
            sc=_score(r,d,n); cnt+=1
            if sc>bs: bs=sc; bp=(s,lg); br=r; bd=d; bn=n
    R["MA"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 2. RSI
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for p in range(2,201):
        for os_v in range(10,45,5):
            for ob_v in range(55,95,5):
                r,d,n=bt_rsi_wf(c,o,rsis[p],float(os_v),float(ob_v),sb,ss,cm)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(p,os_v,ob_v); br=r; bd=d; bn=n
    R["RSI"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 3. MACD
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for f in range(2,100,2):
        for s in range(f+2,201,2):
            for sg in range(2,min(s,101),2):
                r,d,n=bt_macd_wf(c,o,emas[f],emas[s],sg,sb,ss,cm)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(f,s,sg); br=r; bd=d; bn=n
    R["MACD"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 4. DriftRegime
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for lb in range(10,130,5):
        for dt in [0.52,0.55,0.58,0.60,0.62,0.65,0.68,0.70,0.72]:
            for hp in range(3,30,2):
                r,d,n=bt_drift_wf(c,o,lb,dt,hp,sb,ss,cm)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(lb,dt,hp); br=r; bd=d; bn=n
    R["Drift"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 5. RAMOM
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for mp in range(5,120,5):
        for vp in range(5,60,5):
            for ez in [0.5,1.0,1.5,2.0,2.5,3.0,3.5]:
                for xz in [0.0,0.2,0.5,0.8,1.0]:
                    r,d,n=bt_ramom_wf(c,o,mp,vp,ez,xz,sb,ss,cm)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(mp,vp,ez,xz); br=r; bd=d; bn=n
    R["RAMOM"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 6. Turtle
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ep in range(5,80,3):
        for xp in range(3,50,3):
            for ap in [10,14,20]:
                for am in [1.0,1.5,2.0,2.5,3.0,3.5]:
                    r,d,n=bt_turtle_wf(c,o,h,l,ep,xp,ap,am,sb,ss,cm)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(ep,xp,ap,am); br=r; bd=d; bn=n
    R["Turtle"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 7. Bollinger
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for p in range(5,150,3):
        for ns in [0.5,1.0,1.5,2.0,2.5,3.0,3.5]:
            r,d,n=bt_bollinger_wf(c,o,p,ns,sb,ss,cm)
            sc=_score(r,d,n); cnt+=1
            if sc>bs: bs=sc; bp=(p,ns); br=r; bd=d; bn=n
    R["Bollinger"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 8. Keltner
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ep in range(5,120,5):
        for ap in [7,10,14,20,30]:
            for am in [0.5,1.0,1.5,2.0,2.5,3.0,3.5]:
                r,d,n=bt_keltner_wf(c,o,h,l,ep,ap,am,sb,ss,cm)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(ep,ap,am); br=r; bd=d; bn=n
    R["Keltner"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 9. MultiFactor
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for rp in [5,7,9,14,21,28]:
        for mp in range(5,80,5):
            for vp in range(5,50,5):
                for lt_ in [0.50,0.55,0.60,0.65,0.70,0.75]:
                    for st_ in [0.20,0.25,0.30,0.35,0.40,0.45]:
                        r,d,n=bt_multifactor_wf(c,o,rp,mp,vp,lt_,st_,sb,ss,cm)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(rp,mp,vp,lt_,st_); br=r; bd=d; bn=n
    R["MultiFactor"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 10. VolRegime
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ap in [10,14,20]:
        for vt_ in [0.010,0.015,0.020,0.025,0.030,0.035]:
            for ms_ in [3,5,10,15,20]:
                for ml_ in [20,30,40,50,60,80]:
                    if ms_>=ml_: continue
                    for ros in [20,25,30,35]:
                        for rob in [65,70,75,80]:
                            r,d,n=bt_volregime_wf(c,o,h,l,ap,vt_,ms_,ml_,14,ros,rob,sb,ss,cm)
                            sc=_score(r,d,n); cnt+=1
                            if sc>bs: bs=sc; bp=(ap,vt_,ms_,ml_,ros,rob); br=r; bd=d; bn=n
    R["VolRegime"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 11. ConnorsRSI2
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for rp in [2,3,4,5,7,9]:
        for mat in [50,100,150,200]:
            for mae in [3,5,7,10,15]:
                for ost in [3.0,5.0,10.0,15.0,20.0]:
                    for obt in [80.0,85.0,90.0,95.0]:
                        r,d,n=bt_connors_wf(c,o,rp,mat,mae,ost,obt,sb,ss,cm)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(rp,mat,mae,ost,obt); br=r; bd=d; bn=n
    R["Connors"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 12. MESA
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for fl_ in [0.3,0.4,0.5,0.6,0.7,0.8]:
        for sl_ in [0.01,0.02,0.03,0.05,0.08,0.10]:
            r,d,n=bt_mesa_wf(c,o,fl_,sl_,sb,ss,cm)
            sc=_score(r,d,n); cnt+=1
            if sc>bs: bs=sc; bp=(fl_,sl_); br=r; bd=d; bn=n
    R["MESA"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 13. KAMA
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for erp in [5,8,10,15,20,30]:
        for fsc in [2,3,5]:
            for ssc in [20,30,40,50]:
                for asm_ in [1.0,1.5,2.0,2.5,3.0,3.5]:
                    for ap in [10,14,20]:
                        r,d,n=bt_kama_wf(c,o,h,l,erp,fsc,ssc,asm_,ap,sb,ss,cm)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(erp,fsc,ssc,asm_,ap); br=r; bd=d; bn=n
    R["KAMA"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 14. DonchianATR
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ep_ in range(5,80,3):
        for ap in [7,10,14,20]:
            for am in [1.0,1.5,2.0,2.5,3.0,3.5,4.0]:
                r,d,n=bt_donchian_wf(c,o,h,l,ep_,ap,am,sb,ss,cm)
                sc=_score(r,d,n); cnt+=1
                if sc>bs: bs=sc; bp=(ep_,ap,am); br=r; bd=d; bn=n
    R["Donchian"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 15. ZScoreRev
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for lb in range(10,120,5):
        for ez in [1.0,1.5,2.0,2.5,3.0]:
            for xz in [0.0,0.25,0.5,0.75]:
                for sz in [3.0,3.5,4.0,5.0]:
                    r,d,n=bt_zscore_wf(c,o,lb,ez,xz,sz,sb,ss,cm)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(lb,ez,xz,sz); br=r; bd=d; bn=n
    R["ZScore"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 16. MomBreakout
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for hp_ in [20,40,60,100,150,200,252]:
        for pp in [0.00,0.01,0.02,0.03,0.05,0.08]:
            for ap in [10,14,20]:
                for at_ in [1.0,1.5,2.0,2.5,3.0,3.5]:
                    r,d,n=bt_mombreak_wf(c,o,h,l,hp_,pp,ap,at_,sb,ss,cm)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(hp_,pp,ap,at_); br=r; bd=d; bn=n
    R["MomBreak"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 17. RegimeEMA
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ap in [10,14,20]:
        for vt_ in [0.010,0.015,0.020,0.025,0.030]:
            for fe in [3,5,8,10,15]:
                for se in [15,20,30,40,50,60]:
                    if fe>=se: continue
                    for te in [30,50,80,100]:
                        r,d,n=bt_regime_ema_wf(c,o,h,l,ap,vt_,fe,se,te,sb,ss,cm)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(ap,vt_,fe,se,te); br=r; bd=d; bn=n
    R["RegimeEMA"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 18. TrendFilteredRSI
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for rp in range(2,51,2):
        for tp in [20,30,50,80,100,150,200]:
            if tp > len(c) - 1: continue
            for os_v in [15,20,25,30,35]:
                for ob_v in [65,70,75,80,85]:
                    r,d,n=bt_tfiltrsi_wf(c,o,rsis[rp],mas[tp],float(os_v),float(ob_v),sb,ss,cm)
                    sc=_score(r,d,n); cnt+=1
                    if sc>bs: bs=sc; bp=(rp,tp,os_v,ob_v); br=r; bd=d; bn=n
    R["TFiltRSI"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 19. MomBreakPlus
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for hp_ in [20,40,60,100,150,200]:
        for pp in [0.01,0.02,0.03,0.05,0.08]:
            for ap in [10,14,20]:
                for at_ in [1.0,1.5,2.0,2.5,3.0]:
                    for cb in [2,3,4,5]:
                        r,d,n=bt_mombrkplus_wf(c,o,h,l,hp_,pp,ap,at_,cb,sb,ss,cm)
                        sc=_score(r,d,n); cnt+=1
                        if sc>bs: bs=sc; bp=(hp_,pp,ap,at_,cb); br=r; bd=d; bn=n
    R["MomBrkPlus"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 20. DualMomentum
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for fl in [5,10,15,20,30,40,50,60]:
        for sl in [20,40,60,80,100,120,150,200,252]:
            if fl >= sl: continue
            r,d,n=bt_dualmom_wf(c,o,fl,sl,sb,ss,cm)
            sc=_score(r,d,n); cnt+=1
            if sc>bs: bs=sc; bp=(fl,sl); br=r; bd=d; bn=n
    R["DualMom"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    # 21. Consensus (Multi-Signal Voting)
    bs=-1e18; bp=None; br=0.0; bd=0.0; bn=0; cnt=0
    for ms in [5,10,15,20]:
        for ml in [30,50,80,100,150,200]:
            if ms >= ml or ml > len(c) - 1: continue
            for rp in [7,14,21]:
                for mom_lb in [10,20,40,60]:
                    for os_v in [25,30,35]:
                        for ob_v in [65,70,75]:
                            for vt in [2,3]:
                                r,d,n=bt_consensus_wf(c,o,mas[ms],mas[ml],rsis[rp],mom_lb,
                                                       float(os_v),float(ob_v),vt,sb,ss,cm)
                                sc=_score(r,d,n); cnt+=1
                                if sc>bs: bs=sc; bp=(ms,ml,rp,mom_lb,os_v,ob_v,vt); br=r; bd=d; bn=n
    R["Consensus"]=dict(params=bp,score=bs,ret=br,dd=bd,nt=bn,cnt=cnt)

    return R

# =====================================================================
#  Evaluate Single Strategy with Specific Params
# =====================================================================

def eval_strategy(name, params, c, o, h, l, mas, emas, rsis, sb, ss, cm):
    """Evaluate strategy with given params. Returns (ret, dd, nt)."""
    if params is None:
        return (0.0, 0.0, 0)
    try:
        p = params
        if name == "MA":
            s, lg = int(min(199,max(2,p[0]))), int(min(200,max(3,p[1])))
            if s >= lg: return (0.0, 0.0, 0)
            return bt_ma_wf(c, o, mas[s], mas[lg], sb, ss, cm)
        elif name == "RSI":
            pr = int(min(200,max(2,p[0])))
            return bt_rsi_wf(c, o, rsis[pr], float(p[1]), float(p[2]), sb, ss, cm)
        elif name == "MACD":
            f,s_,sg = int(min(98,max(2,p[0]))), int(min(200,max(4,p[1]))), int(min(100,max(2,p[2])))
            if f >= s_: return (0.0, 0.0, 0)
            return bt_macd_wf(c, o, emas[f], emas[s_], sg, sb, ss, cm)
        elif name == "Drift":
            return bt_drift_wf(c, o, int(max(5,p[0])), float(p[1]), int(max(1,p[2])), sb, ss, cm)
        elif name == "RAMOM":
            return bt_ramom_wf(c, o, int(max(2,p[0])), int(max(2,p[1])), float(p[2]), float(p[3]), sb, ss, cm)
        elif name == "Turtle":
            return bt_turtle_wf(c, o, h, l, int(max(2,p[0])), int(max(2,p[1])), int(max(5,p[2])), float(p[3]), sb, ss, cm)
        elif name == "Bollinger":
            return bt_bollinger_wf(c, o, int(max(3,p[0])), float(max(0.1,p[1])), sb, ss, cm)
        elif name == "Keltner":
            return bt_keltner_wf(c, o, h, l, int(max(3,p[0])), int(max(5,p[1])), float(max(0.1,p[2])), sb, ss, cm)
        elif name == "MultiFactor":
            return bt_multifactor_wf(c, o, int(max(2,p[0])), int(max(2,p[1])), int(max(2,p[2])), float(p[3]), float(p[4]), sb, ss, cm)
        elif name == "VolRegime":
            ms_,ml_ = int(max(2,p[2])), int(max(5,p[3]))
            if ms_ >= ml_: return (0.0, 0.0, 0)
            return bt_volregime_wf(c, o, h, l, int(max(5,p[0])), float(p[1]), ms_, ml_, 14, int(p[4]), int(p[5]), sb, ss, cm)
        elif name == "Connors":
            return bt_connors_wf(c, o, int(max(2,p[0])), int(max(5,p[1])), int(max(2,p[2])), float(p[3]), float(p[4]), sb, ss, cm)
        elif name == "MESA":
            return bt_mesa_wf(c, o, float(max(0.01,p[0])), float(max(0.005,p[1])), sb, ss, cm)
        elif name == "KAMA":
            return bt_kama_wf(c, o, h, l, int(max(2,p[0])), int(max(2,p[1])), int(max(5,p[2])), float(p[3]), int(max(5,p[4])), sb, ss, cm)
        elif name == "Donchian":
            return bt_donchian_wf(c, o, h, l, int(max(3,p[0])), int(max(5,p[1])), float(max(0.5,p[2])), sb, ss, cm)
        elif name == "ZScore":
            return bt_zscore_wf(c, o, int(max(5,p[0])), float(p[1]), float(p[2]), float(p[3]), sb, ss, cm)
        elif name == "MomBreak":
            return bt_mombreak_wf(c, o, h, l, int(max(5,p[0])), float(max(0,p[1])), int(max(5,p[2])), float(max(0.5,p[3])), sb, ss, cm)
        elif name == "RegimeEMA":
            fe,se = int(max(2,p[2])), int(max(5,p[3]))
            if fe >= se: return (0.0, 0.0, 0)
            return bt_regime_ema_wf(c, o, h, l, int(max(5,p[0])), float(p[1]), fe, se, int(max(10,p[4])), sb, ss, cm)
        elif name == "TFiltRSI":
            rp = int(min(200,max(2,p[0]))); tp = int(min(200,max(2,p[1])))
            return bt_tfiltrsi_wf(c, o, rsis[rp], mas[tp], float(p[2]), float(p[3]), sb, ss, cm)
        elif name == "MomBrkPlus":
            return bt_mombrkplus_wf(c, o, h, l, int(max(5,p[0])), float(max(0,p[1])),
                                    int(max(5,p[2])), float(max(0.5,p[3])), int(max(1,p[4])), sb, ss, cm)
        elif name == "DualMom":
            fl_,sl_ = int(max(2,p[0])), int(max(5,p[1]))
            if fl_ >= sl_: return (0.0, 0.0, 0)
            return bt_dualmom_wf(c, o, fl_, sl_, sb, ss, cm)
        elif name == "Consensus":
            ms_,ml_ = int(min(200,max(2,p[0]))), int(min(200,max(5,p[1])))
            if ms_ >= ml_: return (0.0, 0.0, 0)
            rp_ = int(min(200,max(2,p[2])))
            return bt_consensus_wf(c, o, mas[ms_], mas[ml_], rsis[rp_],
                                   int(max(5,p[3])), float(p[4]), float(p[5]), int(p[6]), sb, ss, cm)
    except Exception:
        pass
    return (0.0, 0.0, 0)

# =====================================================================
#  Param Type Info for Perturbation
# =====================================================================

PARAM_TYPES = {
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
    "Connors": [int, int, int, float, float],
    "MESA": [float, float],
    "KAMA": [int, int, int, float, int],
    "Donchian": [int, int, float],
    "ZScore": [int, float, float, float],
    "MomBreak": [int, float, int, float],
    "RegimeEMA": [int, float, int, int, int],
    "TFiltRSI": [int, int, int, int],
    "MomBrkPlus": [int, float, int, float, int],
    "DualMom": [int, int],
    "Consensus": [int, int, int, int, int, int, int],
}

# =====================================================================
#  Main
# =====================================================================

def deflated_sharpe(sharpe_obs, n_trials, n_bars, skew=0.0, kurtosis=3.0):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
    Tests H0: the best Sharpe among `n_trials` strategies is no better
    than what we'd expect by chance from i.i.d. noise trials.
    Returns a p-value-like score in [0,1]; higher = more likely genuine.
    """
    if n_bars < 3 or n_trials < 1:
        return 0.0
    e_max_sr = ((1.0 - 0.5772) * (2.0 * math.log(n_trials))**0.5
                + 0.5772 * (2.0 * math.log(n_trials))**-0.5)
    se_sr = ((1.0 - skew * sharpe_obs + (kurtosis - 1.0) / 4.0 * sharpe_obs**2)
             / max(1.0, n_bars - 1.0))**0.5
    if se_sr < 1e-12:
        return 1.0 if sharpe_obs > e_max_sr else 0.0
    z = (sharpe_obs - e_max_sr) / se_sr
    return float(sp_stats.norm.cdf(z))


def main():
    print("=" * 80)
    print("  10-Layer Anti-Overfitting Robust Scan  (V3)")
    print("  Purged WF + Scoring + Stability + MC + OHLC Shuffle")
    print("  + Block Bootstrap + Deflated Sharpe + Cross-Asset")
    print("=" * 80)

    symbols = ["AAPL", "GOOGL", "TSLA", "BTC", "ETH", "SOL", "SPY", "AMZN"]
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
    print(f"[1/10] Loading data ...", flush=True)
    datasets = {}
    for sym in symbols:
        df = pd.read_csv(os.path.join(data_dir, f"{sym}.csv"), parse_dates=["date"])
        c = df["close"].values.astype(np.float64)
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        datasets[sym] = {"c": c, "o": o, "h": h, "l": l, "n": len(c)}
        print(f"  {sym}: {len(c)} bars")

    # ---- Phase 2: Purged Walk-Forward Scan (Layers 1+2+3) ----
    print(f"\n[2/10] Purged Walk-Forward Scan ({n_windows} windows x {len(symbols)} symbols, "
          f"embargo={EMBARGO} bars) ...\n", flush=True)

    wf = {sym: {sn: [] for sn in STRAT_NAMES} for sym in symbols}
    best_params = {sym: {sn: None for sn in STRAT_NAMES} for sym in symbols}
    combo_counts = {sn: 0 for sn in STRAT_NAMES}
    total_combos = 0
    grand_t0 = time.time()

    for sym in symbols:
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
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
            h_tr, l_tr = h[:tr_end], l[:tr_end]
            c_va, o_va = c[va_start:va_end], o[va_start:va_end]
            h_va, l_va = h[va_start:va_end], l[va_start:va_end]
            c_te, o_te = c[va_end:te_end], o[va_end:te_end]
            h_te, l_te = h[va_end:te_end], l[va_end:te_end]

            t_w = time.time()
            mas_tr = precompute_all_ma(c_tr, 200)
            emas_tr = precompute_all_ema(c_tr, 200)
            rsis_tr = precompute_all_rsi(c_tr, 200)

            results = scan_all(c_tr, o_tr, h_tr, l_tr, mas_tr, emas_tr, rsis_tr, SB, SS, CM)

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
            emb_s = f" emb={EMBARGO}" if EMBARGO > 0 else ""
            print(f"    W{wi+1}: train[0:{tr_end}] val[{va_start}:{va_end}] "
                  f"test[{va_end}:{te_end}]{emb_s}  {w_combos:,} combos  {elapsed_w:.1f}s", flush=True)

    wf_elapsed = time.time() - grand_t0
    print(f"\n  Walk-Forward total: {total_combos:,} combos in {wf_elapsed:.1f}s "
          f"({total_combos/wf_elapsed:,.0f}/s)")

    wfe = {sym: {} for sym in symbols}
    gen_gap = {sym: {} for sym in symbols}
    for sym in symbols:
        for sn in STRAT_NAMES:
            oos_rets = [w["test_ret"] for w in wf[sym][sn]]
            wfe[sym][sn] = np.mean(oos_rets) if oos_rets else 0.0
            gaps = [abs(w["gen_gap"]) for w in wf[sym][sn]]
            gen_gap[sym][sn] = np.mean(gaps) if gaps else 99.0

    # ---- Phase 3: Parameter Stability (Layer 4) ----
    print(f"\n[3/10] Parameter Stability Analysis ...", flush=True)

    stability = {sym: {} for sym in symbols}
    for sym in symbols:
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        mas = precompute_all_ma(c, 200)
        emas = precompute_all_ema(c, 200)
        rsis = precompute_all_rsi(c, 200)

        for sn in STRAT_NAMES:
            params = best_params[sym][sn]
            if params is None:
                stability[sym][sn] = 0.0
                continue

            returns = []
            base_r, _, _ = eval_strategy(sn, params, c, o, h, l, mas, emas, rsis, SB, SS, CM)
            returns.append(base_r)

            ptypes = PARAM_TYPES.get(sn, [])
            for pi in range(len(params)):
                for factor in PERTURB:
                    pv = params[pi] * factor
                    if pi < len(ptypes) and ptypes[pi] == int:
                        pv = max(1, int(round(pv)))
                    new_p = list(params)
                    new_p[pi] = pv
                    r, _, _ = eval_strategy(sn, tuple(new_p), c, o, h, l, mas, emas, rsis, SB, SS, CM)
                    returns.append(r)

            mean_r = np.mean(returns)
            std_r = np.std(returns)
            stab = 1.0 - std_r / abs(mean_r) if abs(mean_r) > 1e-8 else 0.0
            stability[sym][sn] = max(0.0, min(1.0, stab))

    print("  done.")

    # ---- Phase 4: Monte Carlo Price Perturbation (Layer 6) ----
    print(f"[4/10] Monte Carlo Price Perturbation ({MC_PATHS} paths) ...", flush=True)

    mc_results = {sym: {} for sym in symbols}
    for sym in symbols:
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        for sn in STRAT_NAMES:
            params = best_params[sym][sn]
            if params is None:
                mc_results[sym][sn] = {"profitable": 0.0, "stability": 0.0, "mean_ret": 0.0}
                continue
            mc_rets = []
            for seed in range(MC_PATHS):
                cp, op, hp, lp = perturb_ohlc(c, o, h, l, MC_NOISE_STD, seed + 1000)
                r, _, _ = eval_strategy_mc(sn, params, cp, op, hp, lp, SB, SS, CM)
                mc_rets.append(r)
            mc_arr = np.array(mc_rets)
            mc_mean = np.mean(mc_arr)
            mc_std = np.std(mc_arr)
            mc_prof = float(np.sum(mc_arr > 0)) / len(mc_arr)
            mc_stab = max(0.0, min(1.0, 1.0 - mc_std / abs(mc_mean))) if abs(mc_mean) > 1e-8 else 0.0
            mc_results[sym][sn] = {"profitable": mc_prof, "stability": mc_stab, "mean_ret": mc_mean}

    print("  done.")

    # ---- Phase 5: OHLC Shuffle Perturbation (Layer 7) ----
    print(f"[5/10] OHLC Shuffle Perturbation ({SHUFFLE_PATHS} paths) ...", flush=True)

    shuffle_results = {sym: {} for sym in symbols}
    for sym in symbols:
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        for sn in STRAT_NAMES:
            params = best_params[sym][sn]
            if params is None:
                shuffle_results[sym][sn] = {"profitable": 0.0, "stability": 0.0, "mean_ret": 0.0}
                continue
            sh_rets = []
            for seed in range(SHUFFLE_PATHS):
                cp, op, hp, lp = shuffle_ohlc(c, o, h, l, seed + 5000)
                r, _, _ = eval_strategy_mc(sn, params, cp, op, hp, lp, SB, SS, CM)
                sh_rets.append(r)
            sh_arr = np.array(sh_rets)
            sh_mean = np.mean(sh_arr)
            sh_std = np.std(sh_arr)
            sh_prof = float(np.sum(sh_arr > 0)) / len(sh_arr)
            sh_stab = max(0.0, min(1.0, 1.0 - sh_std / abs(sh_mean))) if abs(sh_mean) > 1e-8 else 0.0
            shuffle_results[sym][sn] = {"profitable": sh_prof, "stability": sh_stab, "mean_ret": sh_mean}

    print("  done.")

    # ---- Phase 6: Block Bootstrap Resampling (Layer 8) ----
    print(f"[6/10] Block Bootstrap ({BOOTSTRAP_PATHS} paths, block={BOOTSTRAP_BLOCK}) ...", flush=True)

    bootstrap_results = {sym: {} for sym in symbols}
    for sym in symbols:
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        for sn in STRAT_NAMES:
            params = best_params[sym][sn]
            if params is None:
                bootstrap_results[sym][sn] = {"profitable": 0.0, "stability": 0.0, "mean_ret": 0.0}
                continue
            bs_rets = []
            for seed in range(BOOTSTRAP_PATHS):
                cp, op, hp, lp = block_bootstrap_ohlc(c, o, h, l, BOOTSTRAP_BLOCK, seed + 9000)
                r, _, _ = eval_strategy_mc(sn, params, cp, op, hp, lp, SB, SS, CM)
                bs_rets.append(r)
            bs_arr = np.array(bs_rets)
            bs_mean = np.mean(bs_arr)
            bs_std = np.std(bs_arr)
            bs_prof = float(np.sum(bs_arr > 0)) / len(bs_arr)
            bs_stab = max(0.0, min(1.0, 1.0 - bs_std / abs(bs_mean))) if abs(bs_mean) > 1e-8 else 0.0
            bootstrap_results[sym][sn] = {"profitable": bs_prof, "stability": bs_stab, "mean_ret": bs_mean}

    print("  done.")

    # ---- Phase 7: Deflated Sharpe Ratio (Layer 9) ----
    print(f"[7/10] Deflated Sharpe Ratio ...", flush=True)

    dsr_scores = {sym: {} for sym in symbols}
    for sym in symbols:
        D = datasets[sym]
        n_bars = D["n"]
        for sn in STRAT_NAMES:
            oos_rets = [w["test_ret"] for w in wf[sym][sn]]
            if not oos_rets or all(r == 0.0 for r in oos_rets):
                dsr_scores[sym][sn] = 0.0
                continue
            ret_arr = np.array(oos_rets) / 100.0
            mu = np.mean(ret_arr)
            sd = np.std(ret_arr)
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
    for sym in symbols:
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        precomp_cache[sym] = {
            "c": c, "o": o, "h": h, "l": l,
            "mas": precompute_all_ma(c, 200),
            "emas": precompute_all_ema(c, 200),
            "rsis": precompute_all_rsi(c, 200),
        }

    for sn in STRAT_NAMES:
        cross[sn] = {}
        for train_sym in symbols:
            params = best_params[train_sym][sn]
            cross[sn][train_sym] = {}
            for test_sym in symbols:
                if test_sym == train_sym:
                    cross[sn][train_sym][test_sym] = None
                    continue
                pc = precomp_cache[test_sym]
                r, _, _ = eval_strategy(
                    sn, params, pc["c"], pc["o"], pc["h"], pc["l"],
                    pc["mas"], pc["emas"], pc["rsis"], SB, SS, CM
                )
                cross[sn][train_sym][test_sym] = r

    print("  done.")

    total_elapsed = time.time() - grand_t0

    # ---- Phase 9: 8-dim Composite Ranking (Layer 10) ----
    print(f"\n[9/10] 8-dim Composite Ranking ...", flush=True)

    avg_wfe = {}; avg_gen_gap = {}; avg_stability = {}
    avg_mc = {}; avg_shuffle = {}; avg_bootstrap = {}; avg_dsr = {}; avg_cross = {}

    for sn in STRAT_NAMES:
        avg_wfe[sn] = np.mean([wfe[sym][sn] for sym in symbols])
        avg_gen_gap[sn] = np.mean([gen_gap[sym][sn] for sym in symbols])
        avg_stability[sn] = np.mean([stability[sym][sn] for sym in symbols])
        mc_s = [mc_results[sym][sn]["profitable"] * max(0.0, mc_results[sym][sn]["stability"]) for sym in symbols]
        avg_mc[sn] = np.mean(mc_s)
        sh_s = [shuffle_results[sym][sn]["profitable"] * max(0.0, shuffle_results[sym][sn]["stability"]) for sym in symbols]
        avg_shuffle[sn] = np.mean(sh_s)
        bs_s = [bootstrap_results[sym][sn]["profitable"] * max(0.0, bootstrap_results[sym][sn]["stability"]) for sym in symbols]
        avg_bootstrap[sn] = np.mean(bs_s)
        avg_dsr[sn] = np.mean([dsr_scores[sym][sn] for sym in symbols])
        cross_rets = []
        for train_sym in symbols:
            for test_sym in symbols:
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
        composite[sn] = (wfe_ranked.index(sn) + 1
                         + gap_ranked.index(sn) + 1
                         + stab_ranked.index(sn) + 1
                         + mc_ranked.index(sn) + 1
                         + shuffle_ranked.index(sn) + 1
                         + bootstrap_ranked.index(sn) + 1
                         + dsr_ranked.index(sn) + 1
                         + cross_ranked.index(sn) + 1)

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

    print(f"\n{'='*120}")
    print(f"  FINAL 8-DIM COMPOSITE RANKING — {total_combos:,} combos in {total_elapsed:.1f}s")
    print(f"{'='*120}\n")
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

    # ---- Phase 10: Generate Report ----
    print(f"\n[10/10] Generating report ...", flush=True)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "docs"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "robust_scan"), exist_ok=True)
    rpt = os.path.join(os.path.dirname(__file__), "..", "docs", "ROBUST_ANTI_OVERFIT_REPORT.md")

    L = []
    L.append("# 10-Layer Anti-Overfitting Robust Scan Report V3 (21 Strategies × 8 Assets)\n")
    L.append(f"> **{total_combos:,}** backtests | **{total_elapsed:.1f}s** | **{total_combos/total_elapsed:,.0f}** combos/sec")
    L.append(f"> Walk-Forward: {n_windows} purged windows (Train/Val/Test, embargo={EMBARGO} bars)")
    L.append(f"> Scoring: return/drawdown * trade_factor | Min trades: {MIN_TRADES}")
    L.append(f"> Stability: perturb ±10-20% | MC: {MC_PATHS} paths, σ={MC_NOISE_STD*100:.1f}%")
    L.append(f"> OHLC Shuffle: {SHUFFLE_PATHS} paths | Block Bootstrap: {BOOTSTRAP_PATHS} paths, block={BOOTSTRAP_BLOCK}")
    L.append(f"> Deflated Sharpe Ratio: corrects for {len(STRAT_NAMES)} × param-grid multiple tests")
    L.append(f"> Cross-asset: {len(symbols)} symbols | Data: {', '.join(symbols)}")
    L.append(f"> Cost: 5bps slippage + 15bps commission")
    L.append(f"> Execution: **Next-Open** (signal @ close[i] → fill @ open[i+1])\n")

    L.append("---\n")
    L.append("## 1. Final Composite Ranking (8-dimensional)\n")
    L.append("Ranks by: WFE + GenGap + Stability + MC + OHLC Shuffle + Block Bootstrap + Deflated Sharpe + Cross-Asset.\n")
    L.append("| Rank | Strategy | WFE | Gap | Stab | MC | Shuffle | Boot | DSR | Cross | Score | Verdict |")
    L.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")
    for i, sn in enumerate(final_ranked):
        w = avg_wfe[sn]; g = avg_gen_gap[sn]; s = avg_stability[sn]
        m = avg_mc[sn]; sh = avg_shuffle[sn]; bs = avg_bootstrap[sn]
        d = avg_dsr[sn]; cr = avg_cross[sn]; cs = composite[sn]
        L.append(f"| {i+1} | **{sn}** | {w:+.1f}% | {g:.1f}% | {s:.3f} | {m:.3f} | {sh:.3f} | {bs:.3f} | {d:.3f} | {cr:+.1f}% | {cs} | {verdict(sn)} |")
    L.append("")

    L.append("---\n")
    L.append("## 2. Walk-Forward Efficiency (WFE) Rankings\n")
    L.append(f"Average out-of-sample (Test) return across {n_windows} anchored walk-forward windows.\n")
    L.append("| Rank | Strategy | Avg WFE | " + " | ".join(symbols) + " |")
    L.append("|:---:|:---|:---:|" + ":---:|" * len(symbols))
    for i, sn in enumerate(wfe_ranked):
        vals = " | ".join([f"{wfe[sym][sn]:+.1f}%" for sym in symbols])
        L.append(f"| {i+1} | {sn} | {avg_wfe[sn]:+.1f}% | {vals} |")
    L.append("")

    L.append("---\n")
    L.append("## 3. Generalization Gap Analysis\n")
    L.append("Gen Gap = |Val_return - Test_return|, averaged across windows. Lower is better.\n")
    L.append("Since Val and Test have equal width and are adjacent, a large gap indicates overfitting.\n")
    L.append("| Rank | Strategy | Avg Gap | " + " | ".join(symbols) + " |")
    L.append("|:---:|:---|:---:|" + ":---:|" * len(symbols))
    for i, sn in enumerate(gap_ranked):
        vals = " | ".join([f"{gen_gap[sym][sn]:.1f}%" for sym in symbols])
        L.append(f"| {i+1} | {sn} | {avg_gen_gap[sn]:.1f}% | {vals} |")
    L.append("")

    L.append("---\n")
    L.append("## 4. Walk-Forward Detail by Symbol\n")
    for sym in symbols:
        L.append(f"### {sym}\n")
        wn_hdr = " | ".join([f"W{i+1} Val" for i in range(n_windows)] +
                             [f"W{i+1} Test" for i in range(n_windows)])
        L.append(f"| Strategy | {wn_hdr} | Avg WFE | Avg Gap |")
        L.append("|:---|" + ":---:|" * (n_windows * 2 + 2))
        for sn in wfe_ranked:
            wins = wf[sym][sn]
            vals_v = [f"{w['val_ret']:+.1f}%" if i < len(wins) else "N/A"
                      for i, w in enumerate(wins)]
            vals_t = [f"{w['test_ret']:+.1f}%" if i < len(wins) else "N/A"
                      for i, w in enumerate(wins)]
            while len(vals_v) < n_windows: vals_v.append("N/A")
            while len(vals_t) < n_windows: vals_t.append("N/A")
            L.append(f"| {sn} | {' | '.join(vals_v)} | {' | '.join(vals_t)} "
                     f"| {wfe[sym][sn]:+.1f}% | {gen_gap[sym][sn]:.1f}% |")
        L.append("")

    L.append("---\n")
    L.append("## 5. Parameter Stability Analysis\n")
    L.append("Stability = 1 - (std / |mean|) of returns when params are perturbed ±10-20%.\n")
    L.append("| Strategy | " + " | ".join(symbols) + " | Average | Class |")
    L.append("|:---|" + ":---:|" * len(symbols) + ":---:|:---|")
    for sn in stab_ranked:
        vals = " | ".join([f"{stability[sym][sn]:.3f}" for sym in symbols])
        avg_s = avg_stability[sn]
        cls = "STABLE" if avg_s > 0.7 else "MODERATE" if avg_s > 0.4 else "FRAGILE"
        L.append(f"| {sn} | {vals} | {avg_s:.3f} | {cls} |")
    L.append("")

    L.append("---\n")
    L.append("## 6. Monte Carlo Price Perturbation\n")
    L.append(f"Each strategy tested on {MC_PATHS} OHLC paths with Gaussian noise σ={MC_NOISE_STD*100:.1f}%.\n")
    L.append("MC Robust = % profitable paths × stability. Higher is better.\n")
    L.append("| Strategy | " + " | ".join([f"{s} %Prof" for s in symbols]) +
             " | " + " | ".join([f"{s} Stab" for s in symbols]) +
             " | Avg Score |")
    L.append("|:---|" + ":---:|" * (len(symbols) * 2 + 1))
    for sn in mc_ranked:
        profs = " | ".join([f"{mc_results[sym][sn]['profitable']*100:.0f}%" for sym in symbols])
        stabs = " | ".join([f"{mc_results[sym][sn]['stability']:.3f}" for sym in symbols])
        L.append(f"| {sn} | {profs} | {stabs} | {avg_mc[sn]:.3f} |")
    L.append("")

    L.append("---\n")
    L.append("## 7. OHLC Shuffle Perturbation\n")
    L.append(f"Each strategy tested on {SHUFFLE_PATHS} paths where O/H/L/C are randomly reassigned per bar.\n")
    L.append("Tests whether strategy relies on genuine price structure vs. specific OHLC labelling.\n")
    L.append("| Strategy | " + " | ".join([f"{s} %Prof" for s in symbols]) + " | Avg Score |")
    L.append("|:---|" + ":---:|" * (len(symbols) + 1))
    for sn in shuffle_ranked:
        profs = " | ".join([f"{shuffle_results[sym][sn]['profitable']*100:.0f}%" for sym in symbols])
        L.append(f"| {sn} | {profs} | {avg_shuffle[sn]:.3f} |")
    L.append("")

    L.append("---\n")
    L.append("## 8. Block Bootstrap Resampling\n")
    L.append(f"Each strategy tested on {BOOTSTRAP_PATHS} block-bootstrapped paths (block={BOOTSTRAP_BLOCK} bars).\n")
    L.append("Preserves local autocorrelation while testing robustness to data ordering.\n")
    L.append("| Strategy | " + " | ".join([f"{s} %Prof" for s in symbols]) + " | Avg Score |")
    L.append("|:---|" + ":---:|" * (len(symbols) + 1))
    for sn in bootstrap_ranked:
        profs = " | ".join([f"{bootstrap_results[sym][sn]['profitable']*100:.0f}%" for sym in symbols])
        L.append(f"| {sn} | {profs} | {avg_bootstrap[sn]:.3f} |")
    L.append("")

    L.append("---\n")
    L.append("## 9. Deflated Sharpe Ratio\n")
    L.append("Corrects the observed Sharpe for multiple hypothesis testing (Bailey & Lopez de Prado, 2014).\n")
    L.append("DSR ∈ [0,1]: probability that observed Sharpe exceeds what we'd expect from noise trials.\n")
    L.append("| Strategy | " + " | ".join(symbols) + " | Average |")
    L.append("|:---|" + ":---:|" * (len(symbols) + 1))
    for sn in dsr_ranked:
        vals = " | ".join([f"{dsr_scores[sym][sn]:.3f}" for sym in symbols])
        L.append(f"| {sn} | {vals} | {avg_dsr[sn]:.3f} |")
    L.append("")

    L.append("---\n")
    L.append("## 10. Cross-Asset Validation Matrix\n")
    L.append("Returns when params trained on row-symbol are tested on column-symbol.\n")
    for sn in cross_ranked[:5]:
        L.append(f"### {sn} (Avg Cross-Asset: {avg_cross[sn]:+.1f}%)\n")
        L.append("| Train \\ Test | " + " | ".join(symbols) + " |")
        L.append("|:---|" + ":---:|" * len(symbols))
        for train_sym in symbols:
            vals = []
            for test_sym in symbols:
                v = cross[sn][train_sym][test_sym]
                vals.append("---" if v is None else f"{v:+.1f}%")
            L.append(f"| {train_sym} | {' | '.join(vals)} |")
        L.append("")

    L.append("---\n")
    L.append("## 11. Methodology\n")
    L.append("### Purged Walk-Forward Windows (Anchored, Train/Val/Test)\n")
    L.append("```")
    for i, (tr, va, te) in enumerate(WF_WINDOWS):
        L.append(f"Window {i+1}: train [0, {tr*100:.0f}%), "
                 f"[embargo {EMBARGO} bars], "
                 f"val [{tr*100:.0f}%+emb, {va*100:.0f}%), "
                 f"test [{va*100:.0f}%, {te*100:.0f}%)")
    L.append("```\n")
    L.append(f"**Embargo**: {EMBARGO}-bar gap between train and val prevents information leakage\n")
    L.append("Val and Test have **equal width** (10% each). This isolates overfitting from regime change.\n")
    L.append("### Monte Carlo Price Perturbation\n")
    L.append("```")
    L.append(f"N paths: {MC_PATHS}, Noise: Gaussian sigma = {MC_NOISE_STD*100:.1f}% per bar")
    L.append("Perturbed: close (signal), open (fill), high/low (channels)")
    L.append("OHLC validity maintained: low <= min(O,C), high >= max(O,C)")
    L.append("```\n")
    L.append("### OHLC Shuffle Perturbation\n")
    L.append("```")
    L.append(f"N paths: {SHUFFLE_PATHS}")
    L.append("Method: randomly permute {O,H,L,C} per bar, then re-assign:")
    L.append("  max -> H, min -> L, remaining two -> O and C")
    L.append("```\n")
    L.append("Tests structural dependence: a strategy that relies on 'close is the last")
    L.append("traded price' should degrade when close becomes a random intra-bar price.\n")
    L.append("### Block Bootstrap Resampling\n")
    L.append("```")
    L.append(f"N paths: {BOOTSTRAP_PATHS}, Block size: {BOOTSTRAP_BLOCK} bars")
    L.append("Method: resample contiguous blocks with replacement, scale to maintain price continuity")
    L.append("```\n")
    L.append("Preserves local autocorrelation (momentum, mean-reversion within blocks)")
    L.append("while destroying long-range dependencies. Strategies that only work on a")
    L.append("specific sequence of events will fail.\n")
    L.append("### Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)\n")
    L.append("```")
    L.append("E[max(SR)] = (1-γ)√(2·ln(N)) + γ/√(2·ln(N))  where N = param grid size")
    L.append("DSR = Φ((SR_obs - E[max(SR)]) / SE(SR))")
    L.append("```\n")
    L.append("Corrects for the fact that scanning many parameter combinations inflates")
    L.append("the best observed Sharpe. A DSR near 0 means the strategy's performance")
    L.append("is no better than the best of N random noise trials.\n")
    L.append("### Composite Ranking (8-dimensional)\n")
    L.append("```")
    L.append("composite = rank(WFE↓) + rank(GenGap↑) + rank(Stability↓) + rank(MC↓)")
    L.append("          + rank(Shuffle↓) + rank(Bootstrap↓) + rank(DSR↓) + rank(Cross↓)")
    L.append("```\n")
    L.append("### Verdicts\n")
    L.append("- **ROBUST**: ≥7/8 criteria (WFE>0, Gap<5%, Stab>0.5, MC>0.5, Shuffle>0.3, Boot>0.3, DSR>0.3, Cross>0)")
    L.append("- **STRONG**: 5-6/8 criteria")
    L.append("- **MODERATE**: 3-4/8 criteria")
    L.append("- **WEAK**: <3/8 criteria\n")

    L.append("---\n")
    L.append(f"## 12. Best Params per Symbol (Window {n_windows})\n")
    for sym in symbols:
        L.append(f"### {sym}\n")
        L.append("| Strategy | Best Params | Train Ret | Val Ret | Test Ret | Gen Gap | Stability | MC Score |")
        L.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
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
            L.append(f"| {sn} | {pstr} | {tr:+.1f}% | {va:+.1f}% | {te:+.1f}% | {gg:.1f}% | {stab:.3f} | {mcs:.3f} |")
        L.append("")

    with open(rpt, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n  Report: {rpt}")

    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "robust_scan", "summary.csv")
    rows = []
    for sym in symbols:
        for sn in STRAT_NAMES:
            oos_rets = [w["test_ret"] for w in wf[sym][sn]]
            val_rets = [w["val_ret"] for w in wf[sym][sn]]
            mcs = mc_results[sym][sn]
            shs = shuffle_results[sym][sn]
            bss = bootstrap_results[sym][sn]
            row = {
                "symbol": sym,
                "strategy": sn,
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
                "best_params": str(best_params[sym][sn]),
            }
            for wi in range(n_windows):
                row[f"w{wi+1}_val"] = round(val_rets[wi], 2) if wi < len(val_rets) else 0
                row[f"w{wi+1}_test"] = round(oos_rets[wi], 2) if wi < len(oos_rets) else 0
            rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")
    print(f"\n  Total: {total_combos:,} backtests in {total_elapsed:.1f}s = {total_combos/total_elapsed:,.0f} combos/sec")


if __name__ == "__main__":
    main()
