#!/usr/bin/env python3
"""
==========================================================================
  Comprehensive Investment Research Report
==========================================================================
For: ETH, BTC, SOL, XRP, TSLA, MSTR
Strategies: Top-3 from 10-Layer V3 (MA, MACD, MomBreak) + all 21 for ranking
Pipeline:
  1. Download / refresh data via yfinance
  2. Run full 10-layer robust scan on target assets
  3. Current market regime & signal analysis
  4. Generate detailed investment research report
"""
import numpy as np
import pandas as pd
import time, sys, os, warnings, math, datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from numba import njit
from scipy import stats as sp_stats

from walk_forward_robust_scan import (
    _ema, _rolling_mean, _rolling_std, _atr, _rsi_wilder,
    _fx, _mtm, _score,
    bt_ma_wf, bt_rsi_wf, bt_macd_wf, bt_drift_wf, bt_ramom_wf,
    bt_turtle_wf, bt_bollinger_wf, bt_keltner_wf, bt_multifactor_wf,
    bt_volregime_wf, bt_connors_wf, bt_mesa_wf, bt_kama_wf,
    bt_donchian_wf, bt_zscore_wf, bt_mombreak_wf, bt_regime_ema_wf,
    bt_tfiltrsi_wf, bt_mombrkplus_wf, bt_dualmom_wf, bt_consensus_wf,
    perturb_ohlc, shuffle_ohlc, block_bootstrap_ohlc,
    precompute_all_ma, precompute_all_ema, precompute_all_rsi,
    scan_all, eval_strategy, eval_strategy_mc,
    STRAT_NAMES, PARAM_TYPES,
    SB, SS, CM,
)

# =====================================================================
#  Constants
# =====================================================================
TARGET_SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "TSLA", "MSTR"]
YF_MAP = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD",
           "TSLA": "TSLA", "MSTR": "MSTR"}
DATA_START = "2020-01-01"
DATA_END = "2026-02-08"

WF_WINDOWS = [
    (0.35, 0.45, 0.55), (0.45, 0.55, 0.65), (0.55, 0.65, 0.75),
    (0.65, 0.75, 0.85), (0.75, 0.85, 0.95), (0.80, 0.90, 1.00),
]
EMBARGO = 5
MC_PATHS = 30
MC_NOISE_STD = 0.002
SHUFFLE_PATHS = 20
BOOTSTRAP_PATHS = 20
BOOTSTRAP_BLOCK = 20
PERTURB = [0.8, 0.9, 1.1, 1.2]

TOP_STRATEGIES = ["MA", "MACD", "MomBreak"]


def deflated_sharpe(sharpe_obs, n_trials, n_bars, skew=0.0, kurtosis=3.0):
    if n_bars < 3 or n_trials < 1: return 0.0
    e_max_sr = ((1.0 - 0.5772) * (2.0 * math.log(n_trials))**0.5
                + 0.5772 * (2.0 * math.log(n_trials))**-0.5)
    se_sr = ((1.0 - skew * sharpe_obs + (kurtosis - 1.0) / 4.0 * sharpe_obs**2)
             / max(1.0, n_bars - 1.0))**0.5
    if se_sr < 1e-12: return 1.0 if sharpe_obs > e_max_sr else 0.0
    z = (sharpe_obs - e_max_sr) / se_sr
    return float(sp_stats.norm.cdf(z))


# =====================================================================
#  Data
# =====================================================================
def download_all(data_dir):
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance"); return {}

    os.makedirs(data_dir, exist_ok=True)
    datasets = {}
    for sym in TARGET_SYMBOLS:
        ticker = YF_MAP[sym]
        fp = os.path.join(data_dir, f"{sym}.csv")
        print(f"  Downloading {ticker} ...", end=" ", flush=True)
        try:
            raw = yf.download(ticker, start=DATA_START, end=DATA_END, progress=False)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        if raw.empty:
            print("EMPTY")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        df = raw.reset_index()
        df.columns = [str(c).strip() for c in df.columns]
        col_lower = {c: c.lower().replace(" ", "") for c in df.columns}
        cols = {}
        for orig, low in col_lower.items():
            if low == "date" or low == "datetime": cols[orig] = "date"
            elif low == "open": cols[orig] = "open"
            elif low == "high": cols[orig] = "high"
            elif low == "low": cols[orig] = "low"
            elif low == "close": cols[orig] = "close"
        df = df.rename(columns=cols)
        needed = ["date", "open", "high", "low", "close"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"MISSING COLUMNS: {missing} (available: {list(df.columns)})")
            continue
        df = df[needed].dropna()
        df.to_csv(fp, index=False)
        datasets[sym] = {
            "c": df["close"].values.astype(np.float64),
            "o": df["open"].values.astype(np.float64),
            "h": df["high"].values.astype(np.float64),
            "l": df["low"].values.astype(np.float64),
            "n": len(df),
            "dates": pd.to_datetime(df["date"]).values,
            "df": df,
        }
        print(f"{len(df)} bars  [{df['date'].iloc[0]} → {df['date'].iloc[-1]}]")
    return datasets


# =====================================================================
#  Current Market Analysis
# =====================================================================
def compute_current_signals(datasets):
    """Compute current technical indicators and signals for each asset."""
    results = {}
    for sym in TARGET_SYMBOLS:
        if sym not in datasets: continue
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        n = len(c)

        rsi_14 = _rsi_wilder(c, 14)
        ma_20 = _rolling_mean(c, 20)
        ma_50 = _rolling_mean(c, 50)
        ma_200 = _rolling_mean(c, 200)
        ema_12 = _ema(c, 12)
        ema_26 = _ema(c, 26)
        macd_line = ema_12 - ema_26
        macd_signal = _ema(macd_line[~np.isnan(macd_line)], 9) if np.sum(~np.isnan(macd_line)) > 9 else np.array([0.0])

        atr_14 = _atr(c, h, l, 14)
        vol_20 = _rolling_std(c, 20)

        # Returns
        ret_1d = (c[-1] / c[-2] - 1) * 100 if n > 1 else 0
        ret_1w = (c[-1] / c[-6] - 1) * 100 if n > 5 else 0
        ret_1m = (c[-1] / c[-22] - 1) * 100 if n > 21 else 0
        ret_3m = (c[-1] / c[-64] - 1) * 100 if n > 63 else 0
        ret_6m = (c[-1] / c[-127] - 1) * 100 if n > 126 else 0
        ret_ytd = (c[-1] / c[-30] - 1) * 100 if n > 29 else 0
        ret_1y = (c[-1] / c[-253] - 1) * 100 if n > 252 else 0

        # Drawdown from ATH
        ath = np.max(c)
        dd_from_ath = (c[-1] / ath - 1) * 100

        # 52-week high/low
        lookback_252 = min(252, n)
        high_52w = np.max(h[-lookback_252:])
        low_52w = np.min(l[-lookback_252:])
        pct_from_52w_high = (c[-1] / high_52w - 1) * 100
        pct_from_52w_low = (c[-1] / low_52w - 1) * 100

        # Volatility
        daily_returns = np.diff(c[-min(60, n):]) / c[-min(60, n):-1]
        ann_vol = np.std(daily_returns) * np.sqrt(365 if sym in ("BTC","ETH","SOL","XRP") else 252) * 100

        # Trend regime
        above_ma200 = c[-1] > ma_200[-1] if not np.isnan(ma_200[-1]) else None
        above_ma50 = c[-1] > ma_50[-1] if not np.isnan(ma_50[-1]) else None
        ma50_above_ma200 = ma_50[-1] > ma_200[-1] if not (np.isnan(ma_50[-1]) or np.isnan(ma_200[-1])) else None

        if above_ma200 and ma50_above_ma200:
            trend = "STRONG UPTREND"
        elif above_ma200:
            trend = "UPTREND"
        elif not above_ma200 and not ma50_above_ma200:
            trend = "STRONG DOWNTREND"
        elif not above_ma200:
            trend = "DOWNTREND"
        else:
            trend = "NEUTRAL"

        # RSI signal
        rsi_val = rsi_14[-1] if not np.isnan(rsi_14[-1]) else 50
        if rsi_val > 70: rsi_signal = "OVERBOUGHT"
        elif rsi_val > 60: rsi_signal = "BULLISH"
        elif rsi_val < 30: rsi_signal = "OVERSOLD"
        elif rsi_val < 40: rsi_signal = "BEARISH"
        else: rsi_signal = "NEUTRAL"

        # MACD signal
        macd_val = macd_line[-1] if not np.isnan(macd_line[-1]) else 0
        macd_sig_val = macd_signal[-1] if len(macd_signal) > 0 else 0
        macd_histogram = macd_val - macd_sig_val
        if macd_val > 0 and macd_histogram > 0: macd_sig = "BULLISH"
        elif macd_val > 0 and macd_histogram < 0: macd_sig = "WEAKENING BULL"
        elif macd_val < 0 and macd_histogram < 0: macd_sig = "BEARISH"
        elif macd_val < 0 and macd_histogram > 0: macd_sig = "RECOVERING"
        else: macd_sig = "NEUTRAL"

        # Momentum breakout check
        high_20 = np.max(h[-min(20, n):])
        high_60 = np.max(h[-min(60, n):])
        near_breakout_20 = c[-1] >= high_20 * 0.98
        near_breakout_60 = c[-1] >= high_60 * 0.98

        results[sym] = {
            "price": c[-1],
            "rsi_14": rsi_val,
            "rsi_signal": rsi_signal,
            "ma_20": ma_20[-1],
            "ma_50": ma_50[-1],
            "ma_200": ma_200[-1],
            "above_ma200": above_ma200,
            "trend": trend,
            "macd_signal": macd_sig,
            "macd_histogram": macd_histogram,
            "atr_14": atr_14[-1] if not np.isnan(atr_14[-1]) else 0,
            "ann_vol": ann_vol,
            "ret_1d": ret_1d, "ret_1w": ret_1w, "ret_1m": ret_1m,
            "ret_3m": ret_3m, "ret_6m": ret_6m, "ret_1y": ret_1y,
            "dd_from_ath": dd_from_ath,
            "high_52w": high_52w, "low_52w": low_52w,
            "pct_from_52w_high": pct_from_52w_high,
            "pct_from_52w_low": pct_from_52w_low,
            "near_breakout_20": near_breakout_20,
            "near_breakout_60": near_breakout_60,
            "n_bars": n,
        }
    return results


# =====================================================================
#  Full 10-Layer Robust Scan
# =====================================================================
def run_full_scan(datasets, strat_list):
    """Run the full 10-layer robust scan."""
    symbols = [s for s in TARGET_SYMBOLS if s in datasets]
    n_windows = len(WF_WINDOWS)

    wf = {sym: {sn: [] for sn in strat_list} for sym in symbols}
    best_params = {sym: {sn: None for sn in strat_list} for sym in symbols}
    combo_counts = {sn: 0 for sn in strat_list}
    total_combos = 0
    grand_t0 = time.time()

    # Phase 2: Purged Walk-Forward
    print(f"\n[2/10] Purged Walk-Forward ({n_windows} windows x {len(symbols)} symbols) ...", flush=True)
    for sym in symbols:
        D = datasets[sym]
        c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        n = D["n"]
        t_sym = time.time()

        for wi, (tr_pct, va_pct, te_pct) in enumerate(WF_WINDOWS):
            tr_end = int(n * tr_pct)
            va_start = min(tr_end + EMBARGO, int(n * va_pct))
            va_end = int(n * va_pct)
            te_end = min(int(n * te_pct), n)
            if va_end > te_end: va_end = te_end
            if va_start >= va_end: va_start = va_end

            c_tr, o_tr = c[:tr_end], o[:tr_end]
            h_tr, l_tr = h[:tr_end], l[:tr_end]
            c_va, o_va = c[va_start:va_end], o[va_start:va_end]
            h_va, l_va = h[va_start:va_end], l[va_start:va_end]
            c_te, o_te = c[va_end:te_end], o[va_end:te_end]
            h_te, l_te = h[va_end:te_end], l[va_end:te_end]

            mas_tr = precompute_all_ma(c_tr, 200)
            emas_tr = precompute_all_ema(c_tr, 200)
            rsis_tr = precompute_all_rsi(c_tr, 200)
            all_results = scan_all(c_tr, o_tr, h_tr, l_tr, mas_tr, emas_tr, rsis_tr, SB, SS, CM)

            mas_va = precompute_all_ma(c_va, 200)
            emas_va = precompute_all_ema(c_va, 200)
            rsis_va = precompute_all_rsi(c_va, 200)
            mas_te = precompute_all_ma(c_te, 200)
            emas_te = precompute_all_ema(c_te, 200)
            rsis_te = precompute_all_rsi(c_te, 200)

            w_combos = 0
            for sn in strat_list:
                res = all_results[sn]
                w_combos += res["cnt"]
                combo_counts[sn] = max(combo_counts[sn], res["cnt"])
                va_ret, va_dd, va_nt = eval_strategy(sn, res["params"], c_va, o_va, h_va, l_va,
                                                      mas_va, emas_va, rsis_va, SB, SS, CM)
                te_ret, te_dd, te_nt = eval_strategy(sn, res["params"], c_te, o_te, h_te, l_te,
                                                      mas_te, emas_te, rsis_te, SB, SS, CM)
                wf[sym][sn].append({
                    "params": res["params"], "train_score": res["score"],
                    "train_ret": res["ret"], "train_dd": res["dd"], "train_nt": res["nt"],
                    "val_ret": va_ret, "val_dd": va_dd, "val_nt": va_nt,
                    "test_ret": te_ret, "test_dd": te_dd, "test_nt": te_nt,
                    "gen_gap": va_ret - te_ret,
                })
                if wi == n_windows - 1:
                    best_params[sym][sn] = res["params"]
            total_combos += w_combos

        print(f"  {sym}: {D['n']} bars, {time.time()-t_sym:.1f}s", flush=True)

    wfe = {sym: {} for sym in symbols}
    gen_gap = {sym: {} for sym in symbols}
    for sym in symbols:
        for sn in strat_list:
            oos_rets = [w["test_ret"] for w in wf[sym][sn]]
            wfe[sym][sn] = np.mean(oos_rets) if oos_rets else 0.0
            gaps = [abs(w["gen_gap"]) for w in wf[sym][sn]]
            gen_gap[sym][sn] = np.mean(gaps) if gaps else 99.0

    # Phase 3: Parameter Stability
    print(f"[3/10] Parameter Stability ...", flush=True)
    stability = {sym: {} for sym in symbols}
    for sym in symbols:
        D = datasets[sym]; c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        mas = precompute_all_ma(c, 200); emas = precompute_all_ema(c, 200); rsis = precompute_all_rsi(c, 200)
        for sn in strat_list:
            params = best_params[sym][sn]
            if params is None: stability[sym][sn] = 0.0; continue
            returns = []
            base_r, _, _ = eval_strategy(sn, params, c, o, h, l, mas, emas, rsis, SB, SS, CM)
            returns.append(base_r)
            ptypes = PARAM_TYPES.get(sn, [])
            for pi in range(len(params)):
                for factor in PERTURB:
                    pv = params[pi] * factor
                    if pi < len(ptypes) and ptypes[pi] == int: pv = max(1, int(round(pv)))
                    new_p = list(params); new_p[pi] = pv
                    r, _, _ = eval_strategy(sn, tuple(new_p), c, o, h, l, mas, emas, rsis, SB, SS, CM)
                    returns.append(r)
            mean_r = np.mean(returns); std_r = np.std(returns)
            stab = 1.0 - std_r / abs(mean_r) if abs(mean_r) > 1e-8 else 0.0
            stability[sym][sn] = max(0.0, min(1.0, stab))
    print("  done.")

    # Phase 4: Monte Carlo
    print(f"[4/10] Monte Carlo ({MC_PATHS} paths) ...", flush=True)
    mc_results = {sym: {} for sym in symbols}
    for sym in symbols:
        D = datasets[sym]; c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        for sn in strat_list:
            params = best_params[sym][sn]
            if params is None: mc_results[sym][sn] = {"profitable": 0.0, "stability": 0.0, "mean_ret": 0.0}; continue
            mc_rets = []
            for seed in range(MC_PATHS):
                cp, op, hp, lp = perturb_ohlc(c, o, h, l, MC_NOISE_STD, seed+1000)
                r, _, _ = eval_strategy_mc(sn, params, cp, op, hp, lp, SB, SS, CM)
                mc_rets.append(r)
            mc_arr = np.array(mc_rets)
            mc_mean = np.mean(mc_arr); mc_std = np.std(mc_arr)
            mc_prof = float(np.sum(mc_arr > 0)) / len(mc_arr)
            mc_stab = max(0.0, min(1.0, 1.0 - mc_std / abs(mc_mean))) if abs(mc_mean) > 1e-8 else 0.0
            mc_results[sym][sn] = {"profitable": mc_prof, "stability": mc_stab, "mean_ret": mc_mean}
    print("  done.")

    # Phase 5: OHLC Shuffle
    print(f"[5/10] OHLC Shuffle ({SHUFFLE_PATHS} paths) ...", flush=True)
    shuffle_results = {sym: {} for sym in symbols}
    for sym in symbols:
        D = datasets[sym]; c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        for sn in strat_list:
            params = best_params[sym][sn]
            if params is None: shuffle_results[sym][sn] = {"profitable": 0.0, "stability": 0.0}; continue
            sh_rets = []
            for seed in range(SHUFFLE_PATHS):
                cp, op, hp, lp = shuffle_ohlc(c, o, h, l, seed+5000)
                r, _, _ = eval_strategy_mc(sn, params, cp, op, hp, lp, SB, SS, CM)
                sh_rets.append(r)
            sh_arr = np.array(sh_rets); sh_mean = np.mean(sh_arr); sh_std = np.std(sh_arr)
            sh_prof = float(np.sum(sh_arr > 0)) / len(sh_arr)
            sh_stab = max(0.0, min(1.0, 1.0 - sh_std / abs(sh_mean))) if abs(sh_mean) > 1e-8 else 0.0
            shuffle_results[sym][sn] = {"profitable": sh_prof, "stability": sh_stab}
    print("  done.")

    # Phase 6: Block Bootstrap
    print(f"[6/10] Block Bootstrap ({BOOTSTRAP_PATHS} paths) ...", flush=True)
    bootstrap_results = {sym: {} for sym in symbols}
    for sym in symbols:
        D = datasets[sym]; c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        for sn in strat_list:
            params = best_params[sym][sn]
            if params is None: bootstrap_results[sym][sn] = {"profitable": 0.0, "stability": 0.0}; continue
            bs_rets = []
            for seed in range(BOOTSTRAP_PATHS):
                cp, op, hp, lp = block_bootstrap_ohlc(c, o, h, l, BOOTSTRAP_BLOCK, seed+9000)
                r, _, _ = eval_strategy_mc(sn, params, cp, op, hp, lp, SB, SS, CM)
                bs_rets.append(r)
            bs_arr = np.array(bs_rets); bs_mean = np.mean(bs_arr); bs_std = np.std(bs_arr)
            bs_prof = float(np.sum(bs_arr > 0)) / len(bs_arr)
            bs_stab = max(0.0, min(1.0, 1.0 - bs_std / abs(bs_mean))) if abs(bs_mean) > 1e-8 else 0.0
            bootstrap_results[sym][sn] = {"profitable": bs_prof, "stability": bs_stab}
    print("  done.")

    # Phase 7: Deflated Sharpe
    print(f"[7/10] Deflated Sharpe Ratio ...", flush=True)
    dsr_scores = {sym: {} for sym in symbols}
    for sym in symbols:
        n_bars = datasets[sym]["n"]
        for sn in strat_list:
            oos_rets = [w["test_ret"] for w in wf[sym][sn]]
            if not oos_rets or all(r == 0.0 for r in oos_rets): dsr_scores[sym][sn] = 0.0; continue
            ret_arr = np.array(oos_rets) / 100.0
            mu = np.mean(ret_arr); sd = np.std(ret_arr)
            sharpe = mu / sd if sd > 1e-12 else 0.0
            skew = float(sp_stats.skew(ret_arr)) if len(ret_arr) > 2 else 0.0
            kurt = float(sp_stats.kurtosis(ret_arr, fisher=False)) if len(ret_arr) > 3 else 3.0
            n_trials = combo_counts.get(sn, 1000)
            dsr_scores[sym][sn] = deflated_sharpe(sharpe, n_trials, n_bars, skew, kurt)
    print("  done.")

    # Phase 8: Cross-Asset
    print(f"[8/10] Cross-Asset Validation ...", flush=True)
    cross = {sn: {} for sn in strat_list}
    precomp_cache = {}
    for sym in symbols:
        D = datasets[sym]; c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        precomp_cache[sym] = {"c": c, "o": o, "h": h, "l": l,
            "mas": precompute_all_ma(c, 200), "emas": precompute_all_ema(c, 200),
            "rsis": precompute_all_rsi(c, 200)}

    for sn in strat_list:
        cross[sn] = {}
        for train_sym in symbols:
            params = best_params[train_sym][sn]; cross[sn][train_sym] = {}
            for test_sym in symbols:
                if test_sym == train_sym: cross[sn][train_sym][test_sym] = None; continue
                pc = precomp_cache[test_sym]
                r, _, _ = eval_strategy(sn, params, pc["c"], pc["o"], pc["h"], pc["l"],
                                        pc["mas"], pc["emas"], pc["rsis"], SB, SS, CM)
                cross[sn][train_sym][test_sym] = r
    print("  done.")

    total_elapsed = time.time() - grand_t0

    # Phase 9: Composite Ranking
    print(f"\n[9/10] 8-dim Composite Ranking ...", flush=True)
    avg = {}
    for key in ["wfe", "gen_gap", "stability", "mc", "shuffle", "bootstrap", "dsr", "cross"]:
        avg[key] = {}
    for sn in strat_list:
        avg["wfe"][sn] = np.mean([wfe[sym][sn] for sym in symbols])
        avg["gen_gap"][sn] = np.mean([gen_gap[sym][sn] for sym in symbols])
        avg["stability"][sn] = np.mean([stability[sym][sn] for sym in symbols])
        avg["mc"][sn] = np.mean([mc_results[sym][sn]["profitable"] * max(0, mc_results[sym][sn]["stability"]) for sym in symbols])
        avg["shuffle"][sn] = np.mean([shuffle_results[sym][sn]["profitable"] * max(0, shuffle_results[sym][sn]["stability"]) for sym in symbols])
        avg["bootstrap"][sn] = np.mean([bootstrap_results[sym][sn]["profitable"] * max(0, bootstrap_results[sym][sn]["stability"]) for sym in symbols])
        avg["dsr"][sn] = np.mean([dsr_scores[sym][sn] for sym in symbols])
        cross_rets = []
        for ts in symbols:
            for tt in symbols:
                if ts == tt: continue
                v = cross[sn][ts][tt]
                if v is not None: cross_rets.append(v)
        avg["cross"][sn] = np.mean(cross_rets) if cross_rets else 0.0

    rankings = {}
    for key, rev in [("wfe", True), ("gen_gap", False), ("stability", True), ("mc", True),
                      ("shuffle", True), ("bootstrap", True), ("dsr", True), ("cross", True)]:
        rankings[key] = sorted(strat_list, key=lambda s: avg[key][s], reverse=rev)

    composite = {}
    for sn in strat_list:
        composite[sn] = sum(rankings[key].index(sn) + 1 for key in rankings)

    final_ranked = sorted(strat_list, key=lambda s: composite[s])

    def verdict(sn):
        checks = 0
        if avg["wfe"][sn] > 0: checks += 1
        if avg["gen_gap"][sn] < 5.0: checks += 1
        if avg["stability"][sn] > 0.5: checks += 1
        if avg["mc"][sn] > 0.5: checks += 1
        if avg["shuffle"][sn] > 0.3: checks += 1
        if avg["bootstrap"][sn] > 0.3: checks += 1
        if avg["dsr"][sn] > 0.3: checks += 1
        if avg["cross"][sn] > 0: checks += 1
        if checks >= 7: return "ROBUST"
        if checks >= 5: return "STRONG"
        if checks >= 3: return "MODERATE"
        return "WEAK"

    print(f"\n{'='*110}")
    print(f"  FINAL RANKING — {total_combos:,} backtests in {total_elapsed:.1f}s")
    print(f"{'='*110}")
    hdr = f"{'Rank':>4}  {'Strategy':>14}  {'WFE':>7}  {'Gap':>6}  {'Stab':>5}  {'MC':>5}  {'Shuf':>5}  {'Boot':>5}  {'DSR':>5}  {'Cross':>7}  {'Score':>5}  {'Verdict':>8}"
    print(hdr); print("-" * len(hdr))
    for i, sn in enumerate(final_ranked):
        w=avg["wfe"][sn]; g=avg["gen_gap"][sn]; s=avg["stability"][sn]
        m=avg["mc"][sn]; sh=avg["shuffle"][sn]; bs=avg["bootstrap"][sn]
        d=avg["dsr"][sn]; cr=avg["cross"][sn]; cs=composite[sn]
        print(f"{i+1:4d}  {sn:>14}  {w:+6.1f}%  {g:5.1f}%  {s:5.3f}  {m:5.3f}  {sh:5.3f}  {bs:5.3f}  {d:5.3f}  {cr:+6.1f}%  {cs:5d}  {verdict(sn):>8}")

    return {
        "wf": wf, "best_params": best_params, "wfe": wfe, "gen_gap": gen_gap,
        "stability": stability, "mc_results": mc_results, "shuffle_results": shuffle_results,
        "bootstrap_results": bootstrap_results, "dsr_scores": dsr_scores, "cross": cross,
        "avg": avg, "composite": composite, "final_ranked": final_ranked, "verdict": verdict,
        "total_combos": total_combos, "total_elapsed": total_elapsed, "combo_counts": combo_counts,
        "symbols": symbols, "strat_list": strat_list,
    }


# =====================================================================
#  Full-Period Backtest (for equity curve analysis)
# =====================================================================
def run_full_period_backtest(datasets, best_params, top_strats):
    """Run best params on full data for equity curve and detailed stats."""
    symbols = [s for s in TARGET_SYMBOLS if s in datasets]
    results = {}
    for sym in symbols:
        D = datasets[sym]; c, o, h, l = D["c"], D["o"], D["h"], D["l"]
        results[sym] = {}
        for sn in top_strats:
            params = best_params[sym].get(sn)
            if params is None: continue
            r, dd, nt = eval_strategy_mc(sn, params, c, o, h, l, SB, SS, CM)

            daily_rets = np.diff(c) / c[:-1]
            ann_factor = 365 if sym in ("BTC","ETH","SOL","XRP") else 252
            n_bars = len(c)
            years = n_bars / ann_factor

            ann_ret = ((1 + r/100) ** (1/years) - 1) * 100 if years > 0 and r > -100 else 0
            sharpe = 0.0
            if len(daily_rets) > 1:
                strategy_daily_ret = r / 100 / n_bars
                strategy_daily_std = np.std(daily_rets)
                if strategy_daily_std > 1e-12:
                    sharpe = strategy_daily_ret / strategy_daily_std * np.sqrt(ann_factor)

            results[sym][sn] = {
                "total_return": r, "max_dd": dd, "n_trades": nt,
                "ann_return": ann_ret, "sharpe": sharpe, "years": years,
            }
    return results


# =====================================================================
#  Report Generation
# =====================================================================
def generate_report(datasets, scan_res, signals, full_bt, report_path):
    symbols = scan_res["symbols"]
    strat_list = scan_res["strat_list"]
    final_ranked = scan_res["final_ranked"]
    best_params = scan_res["best_params"]
    avg = scan_res["avg"]
    verdict_fn = scan_res["verdict"]
    wf = scan_res["wf"]
    wfe = scan_res["wfe"]
    gen_gap = scan_res["gen_gap"]

    L = []
    today = datetime.date.today().strftime("%Y-%m-%d")

    L.append(f"# Investment Research Report")
    L.append(f"## ETH / BTC / SOL / XRP / TSLA / MSTR")
    L.append(f"**Date**: {today}  |  **Methodology**: 10-Layer Anti-Overfitting Robust Backtest System\n")
    L.append(f"> **{scan_res['total_combos']:,}** backtests | **{scan_res['total_elapsed']:.1f}s** elapsed")
    L.append(f"> Walk-Forward: {len(WF_WINDOWS)} purged windows | Embargo: {EMBARGO} bars")
    L.append(f"> MC: {MC_PATHS} paths | OHLC Shuffle: {SHUFFLE_PATHS} paths | Bootstrap: {BOOTSTRAP_PATHS} paths")
    L.append(f"> Cost Model: 5bps slippage + 15bps commission | Execution: Next-Open")
    L.append(f"> **DISCLAIMER**: This is a quantitative research report based on historical data.")
    L.append(f"> Past performance does not guarantee future results. This is NOT investment advice.\n")

    # ==================== Part 1: Market Overview ====================
    L.append("---\n")
    L.append("# Part I: Current Market Overview\n")
    L.append("## 1.1 Price & Return Summary\n")
    L.append("| Asset | Price | 1D | 1W | 1M | 3M | 6M | 1Y | From ATH | 52W Range |")
    L.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|")

    for sym in symbols:
        if sym not in signals: continue
        s = signals[sym]
        range_str = f"${s['low_52w']:,.2f} - ${s['high_52w']:,.2f}"
        if s["price"] < 10:
            range_str = f"${s['low_52w']:.4f} - ${s['high_52w']:.4f}"
            L.append(f"| **{sym}** | ${s['price']:.4f} | {s['ret_1d']:+.1f}% | {s['ret_1w']:+.1f}% | "
                     f"{s['ret_1m']:+.1f}% | {s['ret_3m']:+.1f}% | {s['ret_6m']:+.1f}% | "
                     f"{s['ret_1y']:+.1f}% | {s['dd_from_ath']:+.1f}% | {range_str} |")
        else:
            L.append(f"| **{sym}** | ${s['price']:,.2f} | {s['ret_1d']:+.1f}% | {s['ret_1w']:+.1f}% | "
                     f"{s['ret_1m']:+.1f}% | {s['ret_3m']:+.1f}% | {s['ret_6m']:+.1f}% | "
                     f"{s['ret_1y']:+.1f}% | {s['dd_from_ath']:+.1f}% | {range_str} |")

    L.append("")

    # 1.2 Technical Signals
    L.append("## 1.2 Current Technical Signals\n")
    L.append("| Asset | Trend | RSI(14) | RSI Signal | MACD Signal | Near 20D BO | Near 60D BO | Ann. Vol |")
    L.append("|:---|:---|---:|:---|:---|:---:|:---:|---:|")
    for sym in symbols:
        if sym not in signals: continue
        s = signals[sym]
        bo20 = "YES" if s["near_breakout_20"] else "no"
        bo60 = "YES" if s["near_breakout_60"] else "no"
        L.append(f"| **{sym}** | {s['trend']} | {s['rsi_14']:.1f} | {s['rsi_signal']} | "
                 f"{s['macd_signal']} | {bo20} | {bo60} | {s['ann_vol']:.1f}% |")

    L.append("")

    # 1.3 Moving Average Analysis
    L.append("## 1.3 Moving Average Structure\n")
    L.append("| Asset | Price | MA20 | MA50 | MA200 | Above MA200 | MA50 > MA200 | Structure |")
    L.append("|:---|---:|---:|---:|---:|:---:|:---:|:---|")
    for sym in symbols:
        if sym not in signals: continue
        s = signals[sym]
        above200 = "YES" if s["above_ma200"] else ("NO" if s["above_ma200"] is not None else "N/A")
        golden = "YES" if (s["ma_50"] and s["ma_200"] and not np.isnan(s["ma_50"]) and not np.isnan(s["ma_200"]) and s["ma_50"] > s["ma_200"]) else "NO"
        fmt = ".4f" if s["price"] < 10 else ",.2f"
        L.append(f"| **{sym}** | ${s['price']:{fmt}} | ${s['ma_20']:{fmt}} | "
                 f"${s['ma_50']:{fmt}} | ${s['ma_200']:{fmt}} | {above200} | {golden} | {s['trend']} |")

    L.append("")

    # ==================== Part 2: Strategy Robustness ====================
    L.append("---\n")
    L.append("# Part II: Strategy Robustness Analysis (10-Layer System)\n")

    L.append("## 2.1 Overall Strategy Ranking\n")
    L.append("Strategies ranked across 8 dimensions of robustness. Lower score = better.\n")
    L.append("| Rank | Strategy | WFE | Gap | Stability | MC | Shuffle | Bootstrap | DSR | Cross | Score | Verdict |")
    L.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")
    for i, sn in enumerate(final_ranked):
        a = avg
        L.append(f"| {i+1} | **{sn}** | {a['wfe'][sn]:+.1f}% | {a['gen_gap'][sn]:.1f}% | "
                 f"{a['stability'][sn]:.3f} | {a['mc'][sn]:.3f} | {a['shuffle'][sn]:.3f} | "
                 f"{a['bootstrap'][sn]:.3f} | {a['dsr'][sn]:.3f} | {a['cross'][sn]:+.1f}% | "
                 f"{scan_res['composite'][sn]} | {verdict_fn(sn)} |")
    L.append("")

    # 2.2 Per-Asset WFE
    L.append("## 2.2 Walk-Forward Efficiency by Asset\n")
    L.append("Average out-of-sample (test) return across 6 purged walk-forward windows.\n")
    L.append("| Strategy | " + " | ".join(symbols) + " | Avg |")
    L.append("|:---|" + ":---:|" * (len(symbols) + 1))
    for sn in final_ranked:
        vals = " | ".join([f"{wfe[sym][sn]:+.1f}%" for sym in symbols])
        L.append(f"| {sn} | {vals} | {avg['wfe'][sn]:+.1f}% |")
    L.append("")

    # 2.3 Walk-Forward Detail
    L.append("## 2.3 Walk-Forward Detail (Top-3 Strategies)\n")
    n_windows = len(WF_WINDOWS)
    for sym in symbols:
        L.append(f"### {sym}\n")
        L.append("| Strategy | " +
                 " | ".join([f"W{i+1} Train" for i in range(n_windows)]) + " | " +
                 " | ".join([f"W{i+1} Test" for i in range(n_windows)]) + " | Avg Test | Gap |")
        L.append("|:---|" + ":---:|" * (n_windows * 2 + 2))
        for sn in TOP_STRATEGIES:
            wins = wf[sym][sn]
            tr_vals = " | ".join([f"{w['train_ret']:+.1f}%" for w in wins])
            te_vals = " | ".join([f"{w['test_ret']:+.1f}%" for w in wins])
            L.append(f"| {sn} | {tr_vals} | {te_vals} | {wfe[sym][sn]:+.1f}% | {gen_gap[sym][sn]:.1f}% |")
        L.append("")

    # ==================== Part 3: Best Params & Full Backtest ====================
    L.append("---\n")
    L.append("# Part III: Optimal Parameters & Full-Period Backtest\n")

    L.append("## 3.1 Best Parameters per Asset (Top-3 Strategies)\n")
    for sym in symbols:
        L.append(f"### {sym}\n")
        L.append("| Strategy | Parameters | Full Return | Max DD | Trades | Ann. Return | Sharpe | Verdict |")
        L.append("|:---|:---|---:|---:|---:|---:|---:|:---|")
        for sn in TOP_STRATEGIES:
            p = best_params[sym].get(sn)
            if p is None: continue
            fb = full_bt.get(sym, {}).get(sn, {})
            tr = fb.get("total_return", 0); dd = fb.get("max_dd", 0)
            nt = fb.get("n_trades", 0); ar = fb.get("ann_return", 0)
            sr = fb.get("sharpe", 0)
            verd = verdict_fn(sn)

            param_str = ""
            if sn == "MA":
                param_str = f"Short={p[0]}, Long={p[1]}"
            elif sn == "MACD":
                param_str = f"Fast={p[0]}, Slow={p[1]}, Signal={p[2]}"
            elif sn == "MomBreak":
                param_str = f"Period={p[0]}, Pad={p[1]:.2f}, ATR={p[2]}, Mult={p[3]:.1f}"
            else:
                param_str = str(p)

            L.append(f"| {sn} | {param_str} | {tr:+.1f}% | {dd:.1f}% | {nt} | {ar:+.1f}% | {sr:.2f} | {verd} |")
        L.append("")

    # ==================== Part 4: Per-Asset Deep Dive ====================
    L.append("---\n")
    L.append("# Part IV: Individual Asset Analysis\n")

    for sym in symbols:
        if sym not in signals: continue
        s = signals[sym]
        D = datasets[sym]

        L.append(f"## {sym}\n")

        # Basic info
        is_crypto = sym in ("BTC", "ETH", "SOL", "XRP")
        asset_type = "Cryptocurrency" if is_crypto else "US Equity"
        L.append(f"**Type**: {asset_type}  |  **Data**: {D['n']} bars  |  **Ann. Volatility**: {s['ann_vol']:.1f}%\n")

        # Price overview
        L.append(f"### Price Overview")
        L.append(f"- **Current Price**: ${s['price']:,.2f}" if s['price'] >= 10 else f"- **Current Price**: ${s['price']:.4f}")
        L.append(f"- **52-Week Range**: ${s['low_52w']:,.2f} - ${s['high_52w']:,.2f}" if s['price'] >= 10 else
                 f"- **52-Week Range**: ${s['low_52w']:.4f} - ${s['high_52w']:.4f}")
        L.append(f"- **From ATH**: {s['dd_from_ath']:+.1f}%")
        L.append(f"- **From 52W High**: {s['pct_from_52w_high']:+.1f}%")
        L.append(f"- **From 52W Low**: {s['pct_from_52w_low']:+.1f}%\n")

        # Technical assessment
        L.append(f"### Technical Assessment")
        L.append(f"- **Trend**: {s['trend']}")
        L.append(f"- **RSI(14)**: {s['rsi_14']:.1f} ({s['rsi_signal']})")
        L.append(f"- **MACD**: {s['macd_signal']}")
        L.append(f"- **Near 20-Day Breakout**: {'YES' if s['near_breakout_20'] else 'No'}")
        L.append(f"- **Near 60-Day Breakout**: {'YES' if s['near_breakout_60'] else 'No'}\n")

        # Strategy performance
        L.append(f"### Strategy Backtest Results")
        L.append(f"| Strategy | Params | Total Return | Max DD | Trades | WFE (OOS) | Gen Gap | MC Prof | Stability |")
        L.append(f"|:---|:---|---:|---:|---:|---:|---:|---:|---:|")
        for sn in TOP_STRATEGIES:
            p = best_params[sym].get(sn)
            fb = full_bt.get(sym, {}).get(sn, {})
            tr = fb.get("total_return", 0); dd = fb.get("max_dd", 0); nt = fb.get("n_trades", 0)
            w = wfe[sym][sn]; gg = gen_gap[sym][sn]
            mc_p = scan_res["mc_results"][sym][sn]["profitable"] * 100
            stab = scan_res["stability"][sym][sn]

            if sn == "MA": ps = f"({p[0]},{p[1]})" if p else "N/A"
            elif sn == "MACD": ps = f"({p[0]},{p[1]},{p[2]})" if p else "N/A"
            elif sn == "MomBreak": ps = f"({p[0]},{p[1]:.2f},{p[2]},{p[3]:.1f})" if p else "N/A"
            else: ps = str(p)

            L.append(f"| {sn} | {ps} | {tr:+.1f}% | {dd:.1f}% | {nt} | {w:+.1f}% | {gg:.1f}% | {mc_p:.0f}% | {stab:.3f} |")
        L.append("")

        # Recommendation for this asset
        best_sn = None; best_score = 1e18
        for sn in TOP_STRATEGIES:
            sc = scan_res["composite"].get(sn, 999)
            if sc < best_score and wfe[sym][sn] > 0:
                best_score = sc; best_sn = sn
        if best_sn is None:
            best_sn = final_ranked[0]

        L.append(f"### Recommendation for {sym}")

        overall_score = 0
        if s["trend"] in ("STRONG UPTREND", "UPTREND"): overall_score += 2
        elif s["trend"] in ("STRONG DOWNTREND", "DOWNTREND"): overall_score -= 2
        if s["rsi_signal"] == "OVERSOLD": overall_score += 1
        elif s["rsi_signal"] == "OVERBOUGHT": overall_score -= 1
        if s["macd_signal"] in ("BULLISH",): overall_score += 1
        elif s["macd_signal"] in ("BEARISH",): overall_score -= 1
        if wfe[sym][best_sn] > 5: overall_score += 1
        elif wfe[sym][best_sn] < -5: overall_score -= 1

        if overall_score >= 3: outlook = "BULLISH"
        elif overall_score >= 1: outlook = "CAUTIOUSLY BULLISH"
        elif overall_score <= -3: outlook = "BEARISH"
        elif overall_score <= -1: outlook = "CAUTIOUSLY BEARISH"
        else: outlook = "NEUTRAL"

        L.append(f"- **Technical Outlook**: {outlook} (score: {overall_score}/6)")
        L.append(f"- **Best Strategy**: {best_sn} (Verdict: {verdict_fn(best_sn)})")
        fb = full_bt.get(sym, {}).get(best_sn, {})
        if fb:
            L.append(f"- **Historical Full-Period Return**: {fb.get('total_return',0):+.1f}% ({fb.get('n_trades',0)} trades)")
            L.append(f"- **Max Drawdown**: {fb.get('max_dd',0):.1f}%")
        L.append(f"- **Parameter Re-Optimization**: Every {'2-3 months' if is_crypto else '3-6 months'}")

        if s["rsi_signal"] == "OVERBOUGHT":
            L.append(f"- **WARNING**: RSI overbought ({s['rsi_14']:.0f}), wait for pullback before entry")
        elif s["rsi_signal"] == "OVERSOLD":
            L.append(f"- **OPPORTUNITY**: RSI oversold ({s['rsi_14']:.0f}), potential bounce")
        if s["near_breakout_60"]:
            L.append(f"- **ALERT**: Near 60-day high breakout, momentum strategy may trigger")

        L.append("")

    # ==================== Part 5: Cross-Asset & Portfolio ====================
    L.append("---\n")
    L.append("# Part V: Cross-Asset Validation & Portfolio Considerations\n")

    L.append("## 5.1 Cross-Asset Transferability (Top-3 Strategies)\n")
    L.append("Can parameters optimized on Asset A work on Asset B?\n")
    for sn in TOP_STRATEGIES:
        L.append(f"### {sn}\n")
        L.append("| Train \\ Test | " + " | ".join(symbols) + " |")
        L.append("|:---|" + ":---:|" * len(symbols))
        for ts in symbols:
            vals = []
            for tt in symbols:
                v = scan_res["cross"][sn][ts][tt]
                vals.append("---" if v is None else f"{v:+.1f}%")
            L.append(f"| {ts} | {' | '.join(vals)} |")
        L.append("")

    # 5.2 Risk Assessment
    L.append("## 5.2 Risk Assessment\n")
    L.append("| Asset | Ann. Vol | Max DD (Best Strat) | Risk Level |")
    L.append("|:---|---:|---:|:---|")
    for sym in symbols:
        if sym not in signals: continue
        s = signals[sym]
        fb = full_bt.get(sym, {})
        best_dd = min([v.get("max_dd", 0) for v in fb.values()]) if fb else 0
        risk = "EXTREME" if s["ann_vol"] > 80 else "VERY HIGH" if s["ann_vol"] > 60 else "HIGH" if s["ann_vol"] > 40 else "MODERATE" if s["ann_vol"] > 20 else "LOW"
        L.append(f"| {sym} | {s['ann_vol']:.1f}% | {best_dd:.1f}% | {risk} |")
    L.append("")

    # ==================== Part 6: Conclusions ====================
    L.append("---\n")
    L.append("# Part VI: Conclusions & Key Takeaways\n")

    L.append("## 6.1 Strategy Recommendations\n")
    for i, sn in enumerate(final_ranked[:3]):
        L.append(f"**#{i+1} {sn}** ({verdict_fn(sn)})")
        L.append(f"- WFE: {avg['wfe'][sn]:+.1f}% | Stability: {avg['stability'][sn]:.3f} | MC: {avg['mc'][sn]:.3f}")
        if sn == "MA":
            L.append("- Simple trend-following. Works best on trending assets (BTC, TSLA). May lag in choppy markets.")
        elif sn == "MACD":
            L.append("- Momentum + trend. Most stable parameters across time. Good all-rounder.")
        elif sn == "MomBreak":
            L.append("- Breakout strategy. Captures large moves. Higher risk/reward.")
        L.append("")

    L.append("## 6.2 Asset Attractiveness Ranking\n")
    L.append("Based on combined technical outlook + strategy robustness:\n")

    asset_scores = []
    for sym in symbols:
        if sym not in signals: continue
        s = signals[sym]
        score = 0
        if s["trend"] in ("STRONG UPTREND",): score += 3
        elif s["trend"] in ("UPTREND",): score += 2
        elif s["trend"] in ("DOWNTREND",): score -= 2
        elif s["trend"] in ("STRONG DOWNTREND",): score -= 3
        if s["rsi_signal"] in ("BULLISH", "NEUTRAL"): score += 1
        elif s["rsi_signal"] == "OVERBOUGHT": score -= 1
        best_wfe = max(wfe[sym][sn] for sn in TOP_STRATEGIES)
        if best_wfe > 10: score += 2
        elif best_wfe > 0: score += 1
        elif best_wfe < -10: score -= 2
        asset_scores.append((sym, score, s["trend"], best_wfe))

    asset_scores.sort(key=lambda x: x[1], reverse=True)
    L.append("| Rank | Asset | Score | Trend | Best WFE | Assessment |")
    L.append("|:---:|:---|:---:|:---|:---:|:---|")
    for i, (sym, sc, trend, bwfe) in enumerate(asset_scores):
        assessment = "Strong Buy Signal" if sc >= 4 else "Buy Signal" if sc >= 2 else "Hold" if sc >= 0 else "Caution" if sc >= -2 else "Avoid"
        L.append(f"| {i+1} | **{sym}** | {sc} | {trend} | {bwfe:+.1f}% | {assessment} |")
    L.append("")

    L.append("## 6.3 Important Caveats\n")
    L.append("1. **Past performance ≠ future results**: All backtest returns are historical and subject to overfitting risk, "
             "even with our 10-layer anti-overfitting system.")
    L.append("2. **Parameter decay**: Based on our decay study, parameters should be re-optimized every 2-3 months for crypto, "
             "3-6 months for equities.")
    L.append("3. **Cost model**: All results include 5bps slippage + 15bps commission with Next-Open execution.")
    L.append("4. **Drawdowns are real**: Even robust strategies can experience -20% to -50% drawdowns. "
             "Position sizing and risk management are critical.")
    L.append("5. **Regime changes**: Strategies that work in trending markets may fail in range-bound periods and vice versa.")
    L.append("6. **Diversification**: No single strategy or asset should dominate a portfolio. Use multiple strategies across "
             "uncorrelated assets.")
    L.append("7. **This is NOT investment advice**: This report is for educational and research purposes only.\n")

    L.append("---\n")
    L.append(f"*Report generated: {today} | 10-Layer Anti-Overfitting Robust Backtest System V3*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return "\n".join(L)


# =====================================================================
#  Main
# =====================================================================
def main():
    print("=" * 80)
    print("  Comprehensive Investment Research Report")
    print("  ETH / BTC / SOL / XRP / TSLA / MSTR")
    print("=" * 80)

    data_dir = os.path.join(_HERE, "..", "data")

    # Phase 1: Download data
    print(f"\n[1/10] Downloading / refreshing data ...", flush=True)
    datasets = download_all(data_dir)
    if not datasets:
        print("ERROR: No data downloaded."); return

    # JIT warm-up
    print(f"\n[0/10] JIT warm-up ...", end=" ", flush=True)
    t0 = time.time()
    dc = np.random.rand(300).astype(np.float64)*100+100
    dh = dc+2; dl = dc-2; do_ = dc+0.5
    dm = np.random.rand(300).astype(np.float64)*100+100
    dr = np.random.rand(300).astype(np.float64)*100
    for fn in [bt_ma_wf, bt_rsi_wf, bt_macd_wf, bt_drift_wf, bt_ramom_wf]:
        pass
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
    eval_strategy_mc("MA",(10,50),dc,do_,dh,dl,SB,SS,CM)
    print(f"done ({time.time()-t0:.1f}s)")

    # Phase 2-9: Run full 10-layer scan on all 21 strategies
    scan_res = run_full_scan(datasets, list(STRAT_NAMES))

    # Current market signals
    print(f"\n[EXTRA] Computing current market signals ...", flush=True)
    signals = compute_current_signals(datasets)

    # Full-period backtest
    print(f"[EXTRA] Full-period backtest (top strategies) ...", flush=True)
    full_bt = run_full_period_backtest(datasets, scan_res["best_params"], TOP_STRATEGIES)

    # Generate report
    print(f"\n[10/10] Generating investment research report ...", flush=True)
    report_path = os.path.join(_HERE, "..", "docs", "INVESTMENT_RESEARCH_REPORT.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_text = generate_report(datasets, scan_res, signals, full_bt, report_path)

    # Save CSV
    csv_path = os.path.join(_HERE, "..", "results", "investment_research.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = []
    for sym in scan_res["symbols"]:
        for sn in TOP_STRATEGIES:
            fb = full_bt.get(sym, {}).get(sn, {})
            sig = signals.get(sym, {})
            rows.append({
                "symbol": sym, "strategy": sn,
                "best_params": str(scan_res["best_params"][sym].get(sn)),
                "total_return": fb.get("total_return", 0),
                "max_dd": fb.get("max_dd", 0),
                "n_trades": fb.get("n_trades", 0),
                "wfe": scan_res["wfe"][sym][sn],
                "gen_gap": scan_res["gen_gap"][sym][sn],
                "stability": scan_res["stability"][sym][sn],
                "mc_profitable": scan_res["mc_results"][sym][sn]["profitable"],
                "verdict": scan_res["verdict"](sn),
                "trend": sig.get("trend", ""),
                "rsi_14": sig.get("rsi_14", 0),
                "ann_vol": sig.get("ann_vol", 0),
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n  Report: {report_path}")
    print(f"  CSV: {csv_path}")
    print(f"  Total: {scan_res['total_combos']:,} backtests in {scan_res['total_elapsed']:.1f}s")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
