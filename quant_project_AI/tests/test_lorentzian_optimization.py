"""
Lorentzian Classification 策略 — 真实数据参数优化
═══════════════════════════════════════════════════════════════
数据源: yfinance (BTC-USD 真实日线)
数值参数范围: [x/2, x, 2x]
布尔参数: [True, False]

防过拟合: 训练集优化参数 → 测试集验证
"""

import sys, os, time, itertools, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Any

from quant_framework.strategy.lorentzian_strategy import (
    _rsi_1d, _cci_1d, _willr_1d, _atr_1d, _adx_1d, _ema_1d,
)


# ═══════════════════════════════════════════════════════════════
# 1. 真实数据下载
# ═══════════════════════════════════════════════════════════════

def fetch_real_data(symbol: str = "BTC-USD",
                    start: str = "2021-01-01",
                    end: str = "2026-02-01") -> pd.DataFrame:
    """通过 yfinance 下载真实行情。"""
    import yfinance as yf
    print(f"  正在下载 {symbol} ({start} ~ {end}) ...", flush=True)
    raw = yf.download(symbol, start=start, end=end, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance 下载 {symbol} 失败，请检查网络")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame({
        "date": raw.index,
        "open":  raw["Open"].values,
        "high":  raw["High"].values,
        "low":   raw["Low"].values,
        "close": raw["Close"].values,
        "volume": raw["Volume"].values,
    }).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    print(f"  下载完成: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}, {len(df)} 条", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════
# 2. 向量化回测引擎
# ═══════════════════════════════════════════════════════════════

def vectorized_backtest(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    open_: np.ndarray, params: Dict, commission: float = 0.001,
) -> Dict:
    """全向量化回测 (next-open 成交，无 look-ahead bias)。

    信号在 bar[i] 的 close 时产生 → bar[i+1] 的 open 价成交。
    止损同样在 bar[i+1] 检查: 若 low[i+1] <= stop，以 min(open[i+1], stop) 成交。
    """
    n = len(close)
    p = params

    f1 = _rsi_1d(close, p["rsi_period"])
    hlc3 = (high + low + close) / 3.0
    ema_hlc3 = _ema_1d(hlc3, p["wt_channel"])
    diff = hlc3 - ema_hlc3
    abs_diff = np.abs(diff)
    ema_abs = _ema_1d(abs_diff, p["wt_channel"])
    ema_abs = np.where(ema_abs < 1e-12, 1e-12, ema_abs)
    ci = diff / (0.015 * ema_abs)
    f2 = _ema_1d(ci, p["wt_avg"])
    f3 = _cci_1d(high, low, close, p["cci_period"])
    f4 = _adx_1d(high, low, close, p["adx_period"])
    f5 = _rsi_1d(close, p["rsi_fast_period"])
    features = np.column_stack([f1, f2, f3, f4, f5])

    lb = p["lookback_bars"]
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - lb):
        if close[i + lb] > close[i]:
            labels[i] = 1
        elif close[i + lb] < close[i]:
            labels[i] = -1

    ema_filter = _ema_1d(close, p["ema_period"])
    sma_n = p["sma_period"]
    sma_filter = np.full(n, np.nan)
    if sma_n <= n:
        cs = np.cumsum(close)
        sma_filter[sma_n - 1] = cs[sma_n - 1] / sma_n
        for i in range(sma_n, n):
            sma_filter[i] = (cs[i] - cs[i - sma_n]) / sma_n

    atr = _atr_1d(high, low, close, p["atr_period"])

    min_start = max(p["ema_period"], p["sma_period"], 2 * p["adx_period"], 60)
    k_nn = p["n_neighbors"]
    tw = p["train_window"]
    atr_mult = p["atr_stop_mult"]
    use_ema = p["use_ema_filter"]
    use_sma = p["use_sma_filter"]

    cash = 1_000_000.0
    shares = 0
    stop_price = 0.0
    pending_signal = 0
    trades_buy = []
    trades_sell = []
    portfolio = np.empty(n)

    for i in range(n):
        price = close[i]
        fill_px = open_[i]

        # 执行上一 bar 的挂单 (next-open)
        if pending_signal != 0:
            if pending_signal == 1 and shares == 0:
                affordable = int(cash * 0.95 / (fill_px * (1 + commission))) if fill_px > 0 else 0
                if affordable > 0:
                    cash -= affordable * fill_px * (1 + commission)
                    shares = affordable
                    cur_atr_prev = atr[i - 1] if i > 0 and not np.isnan(atr[i - 1]) else 0
                    stop_price = fill_px - atr_mult * cur_atr_prev if cur_atr_prev > 0 else 0
                    trades_buy.append(fill_px)
            elif pending_signal == -1 and shares > 0:
                cash += shares * fill_px * (1 - commission)
                trades_sell.append(fill_px)
                shares = 0
                stop_price = 0
            pending_signal = 0

        # 止损检查: 用当前 bar 的 low 判断是否触发，以 min(open, stop) 成交
        if shares > 0 and stop_price > 0 and low[i] <= stop_price:
            stop_fill = min(fill_px, stop_price)
            cash += shares * stop_fill * (1 - commission)
            trades_sell.append(stop_fill)
            shares = 0
            stop_price = 0
            portfolio[i] = cash
            continue

        portfolio[i] = cash + shares * price

        if i < min_start or i >= n - 1:
            if shares > 0:
                cur_atr = atr[i] if not np.isnan(atr[i]) else 0
                if cur_atr > 0:
                    new_stop = price - atr_mult * cur_atr
                    if new_stop > stop_price:
                        stop_price = new_stop
            continue

        query = features[i]
        if np.any(np.isnan(query)):
            continue

        train_end = max(0, i - lb)
        train_start = max(0, train_end - tw)
        if train_end - train_start < k_nn * 2:
            continue

        t_f = features[train_start:train_end]
        t_l = labels[train_start:train_end]
        valid = ~np.isnan(t_f).any(axis=1) & (t_l != 0)
        if valid.sum() < k_nn:
            continue
        v_f = t_f[valid]
        v_l = t_l[valid]
        dists = np.sum(np.log1p(np.abs(v_f - query)), axis=1)
        if k_nn < len(dists):
            nn_idx = np.argpartition(dists, k_nn)[:k_nn]
        else:
            nn_idx = np.arange(len(dists))
        vote = int(np.sum(v_l[nn_idx]))
        pred = 1 if vote > 0 else (-1 if vote < 0 else 0)

        if pred != 0:
            if use_ema and not np.isnan(ema_filter[i]):
                if pred == 1 and price < ema_filter[i]:
                    pred = 0
                elif pred == -1 and price > ema_filter[i]:
                    pred = 0
            if pred != 0 and use_sma and not np.isnan(sma_filter[i]):
                if pred == 1 and price < sma_filter[i]:
                    pred = 0
                elif pred == -1 and price > sma_filter[i]:
                    pred = 0

        if pred != 0:
            pending_signal = pred

        cur_atr = atr[i] if not np.isnan(atr[i]) else 0
        if shares > 0 and cur_atr > 0:
            new_stop = price - atr_mult * cur_atr
            if new_stop > stop_price:
                stop_price = new_stop

    final_pv = cash + shares * close[-1]
    portfolio[-1] = final_pv

    total_ret = (final_pv / 1_000_000 - 1) * 100
    dr = np.diff(portfolio) / portfolio[:-1]
    dr = np.where(np.isfinite(dr), dr, 0)
    excess = dr - 0.04 / 252
    std = np.std(excess)
    sharpe = float(np.mean(excess) / std * np.sqrt(252)) if std > 1e-12 else 0.0
    ds = excess[excess < 0]
    ds_std = np.std(ds) if len(ds) > 0 else 0
    sortino = float(np.mean(excess) / ds_std * np.sqrt(252)) if ds_std > 1e-12 else 0.0
    peak = portfolio[0]; max_dd = 0.0
    for v in portfolio:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd
    pairs = min(len(trades_buy), len(trades_sell))
    wins = sum(1 for j in range(pairs) if trades_sell[j] > trades_buy[j])
    win_rate = wins / pairs * 100 if pairs > 0 else 0

    return {
        "total_return_pct": round(total_ret, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_trades": len(trades_buy) + len(trades_sell),
        "round_trips": pairs,
        "wins": wins,
        "win_rate": round(win_rate, 1),
        "final_value": round(final_pv, 2),
    }


# ═══════════════════════════════════════════════════════════════
# 3. 参数定义
# ═══════════════════════════════════════════════════════════════

DEFAULTS = {
    "n_neighbors": 8, "train_window": 200, "ema_period": 50, "sma_period": 50,
    "adx_threshold": 20.0, "atr_period": 14, "atr_stop_mult": 2.0,
    "rsi_period": 14, "rsi_fast_period": 9, "wt_channel": 10, "wt_avg": 11,
    "cci_period": 20, "adx_period": 20, "lookback_bars": 4,
    "use_ema_filter": True, "use_sma_filter": True,
}


def grid_search(close, high, low, open_, grid_keys, grid_vals, base_params, desc):
    """笛卡尔积网格搜索。"""
    combos = list(itertools.product(*[grid_vals[k] for k in grid_keys]))
    n_total = len(combos)
    print(f"  组合数: {n_total}")
    results = []
    t0 = time.perf_counter()
    for idx, combo in enumerate(combos):
        p = dict(base_params)
        for k, v in zip(grid_keys, combo):
            p[k] = v
        try:
            m = vectorized_backtest(close, high, low, open_, p)
            entry = dict(zip(grid_keys, combo))
            entry.update(m)
            results.append(entry)
        except Exception:
            pass
        if (idx + 1) % 200 == 0 or idx == n_total - 1:
            el = time.perf_counter() - t0
            eta = el / (idx + 1) * (n_total - idx - 1)
            print(f"    进度: {idx+1}/{n_total} ({el:.1f}s, ETA {eta:.0f}s)")
    return results


def pick_best(results: List[Dict]) -> Dict:
    """从结果中选最优 (Sharpe 最高)。"""
    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    return df, df.iloc[0].to_dict()


def cast_params(best: Dict, keys: List[str], float_keys=(), bool_keys=()):
    """将 pandas 类型转回 Python 原生类型。"""
    out = {}
    for k in keys:
        v = best.get(k)
        if v is None:
            continue
        if k in bool_keys:
            out[k] = bool(v)
        elif k in float_keys:
            out[k] = float(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = int(round(v))
        else:
            out[k] = int(v)
    return out


def print_top10(df, keys, desc):
    print(f"\n  {desc} Top 10 (by Sharpe):")
    print(f"  {'#':<3} {'Sharpe':>7} {'Ret%':>8} {'DD%':>7} {'Trd':>5} {'Win%':>6}  参数")
    print(f"  {'─' * 85}")
    for r, (_, row) in enumerate(df.head(10).iterrows()):
        ps = ", ".join(f"{k}={row[k]}" for k in keys)
        print(f"  {r+1:<3} {row['sharpe']:>7.3f} {row['total_return_pct']:>+7.2f}% "
              f"{row['max_drawdown_pct']:>6.2f}% {int(row['n_trades']):>5} "
              f"{row['win_rate']:>5.1f}%  {ps}")


# ═══════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("Lorentzian Classification — 真实 BTC-USD 数据参数优化")
    print("数值参数范围: [x/2, x, 2x]   布尔参数: [True, False]")
    print("防过拟合: 训练集 (2021~2023) 选参 → 测试集 (2024~2025) 验证")
    print("成交模式: next-open (信号在 bar[i] close 产生 → bar[i+1] open 成交)")
    print("=" * 80)

    # ── 下载数据 ──
    full_df = fetch_real_data("BTC-USD", start="2021-01-01", end="2026-02-01")

    # ── 训练/测试划分 ──
    split_date = pd.Timestamp("2024-01-01")
    train_df = full_df[full_df["date"] < split_date].reset_index(drop=True)
    test_df = full_df[full_df["date"] >= split_date].reset_index(drop=True)

    print(f"\n  训练集: {train_df['date'].iloc[0].date()} ~ {train_df['date'].iloc[-1].date()}, {len(train_df)} 条")
    print(f"  测试集: {test_df['date'].iloc[0].date()} ~ {test_df['date'].iloc[-1].date()}, {len(test_df)} 条")

    for label, subset in [("训练集", train_df), ("测试集", test_df)]:
        bh = (subset["close"].iloc[-1] / subset["close"].iloc[0] - 1) * 100
        print(f"  {label} Buy&Hold: ${subset['close'].iloc[0]:,.0f} → ${subset['close'].iloc[-1]:,.0f} ({bh:+.1f}%)")

    tr_c = train_df["close"].values.astype(np.float64)
    tr_h = train_df["high"].values.astype(np.float64)
    tr_l = train_df["low"].values.astype(np.float64)
    tr_o = train_df["open"].values.astype(np.float64)

    te_c = test_df["close"].values.astype(np.float64)
    te_h = test_df["high"].values.astype(np.float64)
    te_l = test_df["low"].values.astype(np.float64)
    te_o = test_df["open"].values.astype(np.float64)

    # ── Phase 0: 默认参数在两个集上的表现 ──
    print("\n" + "─" * 80)
    print("Phase 0: 默认参数基准")
    print("─" * 80)
    m0_train = vectorized_backtest(tr_c, tr_h, tr_l, tr_o, DEFAULTS)
    m0_test = vectorized_backtest(te_c, te_h, te_l, te_o, DEFAULTS)
    for tag, m in [("训练集", m0_train), ("测试集", m0_test)]:
        print(f"  [{tag}] 收益: {m['total_return_pct']:+.2f}%  Sharpe: {m['sharpe']:.3f}  "
              f"MaxDD: {m['max_drawdown_pct']:.2f}%  交易: {m['n_trades']}  胜率: {m['win_rate']:.1f}%")

    # ── Phase 1: 核心参数 (在训练集上搜索) ──
    print("\n" + "─" * 80)
    print("Phase 1: 核心参数网格搜索 (训练集)")
    print("─" * 80)

    core_keys = ["n_neighbors", "train_window", "lookback_bars", "atr_stop_mult",
                 "adx_threshold", "use_ema_filter", "use_sma_filter"]
    core_grid = {
        "n_neighbors":    [4, 8, 16],
        "train_window":   [100, 200, 400],
        "lookback_bars":  [2, 4, 8],
        "atr_stop_mult":  [1.0, 2.0, 4.0],
        "adx_threshold":  [10.0, 20.0, 40.0],
        "use_ema_filter": [True, False],
        "use_sma_filter": [True, False],
    }

    results1 = grid_search(tr_c, tr_h, tr_l, tr_o, core_keys, core_grid, DEFAULTS, "Phase 1")
    df1, best1_raw = pick_best(results1)
    best1 = cast_params(best1_raw, core_keys,
                        float_keys=("atr_stop_mult", "adx_threshold"),
                        bool_keys=("use_ema_filter", "use_sma_filter"))
    print_top10(df1, core_keys, "Phase 1")

    # ── Phase 2: 特征参数精调 (在训练集上搜索) ──
    print("\n" + "─" * 80)
    print("Phase 2: 特征参数精调 (训练集, 锁定 Phase 1 最优核心)")
    print("─" * 80)

    feat_keys = ["rsi_period", "rsi_fast_period", "wt_channel", "cci_period",
                 "adx_period", "ema_period", "sma_period"]
    feat_grid = {
        "rsi_period":     [7, 14, 28],
        "rsi_fast_period": [5, 9, 18],
        "wt_channel":     [5, 10, 20],
        "cci_period":     [10, 20, 40],
        "adx_period":     [10, 20, 40],
        "ema_period":     [25, 50, 100],
        "sma_period":     [25, 50, 100],
    }

    base2 = dict(DEFAULTS)
    base2.update(best1)

    results2 = grid_search(tr_c, tr_h, tr_l, tr_o, feat_keys, feat_grid, base2, "Phase 2")
    df2, best2_raw = pick_best(results2)
    best2 = cast_params(best2_raw, feat_keys)
    print_top10(df2, feat_keys, "Phase 2")

    # ── 最优参数组装 ──
    optimal = dict(DEFAULTS)
    optimal.update(best1)
    optimal.update(best2)

    # ── 测试集验证 (样本外) ──
    print("\n" + "=" * 80)
    print("最优参数 & 样本外验证 (测试集: 2024~2025)")
    print("=" * 80)

    print(f"\n  {'参数':<22} {'默认值':>10} {'最优值':>10}")
    print(f"  {'─' * 48}")
    for k in sorted(DEFAULTS.keys()):
        dv = DEFAULTS[k]
        ov = optimal.get(k, dv)
        tag = " *" if ov != dv else ""
        print(f"  {k:<22} {str(dv):>10} {str(ov):>10}{tag}")

    mf_train = vectorized_backtest(tr_c, tr_h, tr_l, tr_o, optimal)
    mf_test = vectorized_backtest(te_c, te_h, te_l, te_o, optimal)

    tr_bh = (tr_c[-1] / tr_c[0] - 1) * 100
    te_bh = (te_c[-1] / te_c[0] - 1) * 100

    header = f"  ┌{'─' * 72}┐"
    sep    = f"  ├{'─' * 72}┤"
    footer = f"  └{'─' * 72}┘"

    print(f"\n{header}")
    print(f"  │ {'':20}{'── 训练集 (2021~23) ──':>24}{'── 测试集 (2024~25) ──':>24}  │")
    print(f"  │ {'指标':<18}{'默认':>12}{'最优':>12}{'默认':>12}{'最优':>12}  │")
    print(sep)
    print(f"  │ {'收益率':<18}{m0_train['total_return_pct']:>+10.2f}%{mf_train['total_return_pct']:>+10.2f}%"
          f"{m0_test['total_return_pct']:>+10.2f}%{mf_test['total_return_pct']:>+10.2f}%  │")
    print(f"  │ {'Sharpe':<18}{m0_train['sharpe']:>11.3f} {mf_train['sharpe']:>11.3f} "
          f"{m0_test['sharpe']:>11.3f} {mf_test['sharpe']:>11.3f}   │")
    print(f"  │ {'Sortino':<18}{m0_train['sortino']:>11.3f} {mf_train['sortino']:>11.3f} "
          f"{m0_test['sortino']:>11.3f} {mf_test['sortino']:>11.3f}   │")
    print(f"  │ {'最大回撤':<18}{m0_train['max_drawdown_pct']:>10.2f}%{mf_train['max_drawdown_pct']:>10.2f}%"
          f"{m0_test['max_drawdown_pct']:>10.2f}%{mf_test['max_drawdown_pct']:>10.2f}%  │")
    print(f"  │ {'交易笔数':<18}{m0_train['n_trades']:>11} {mf_train['n_trades']:>11} "
          f"{m0_test['n_trades']:>11} {mf_test['n_trades']:>11}   │")
    print(f"  │ {'胜/负':<20}{m0_train['wins']:>5}/{m0_train['round_trips']-m0_train['wins']:<5}"
          f"{mf_train['wins']:>5}/{mf_train['round_trips']-mf_train['wins']:<5}"
          f"{m0_test['wins']:>5}/{m0_test['round_trips']-m0_test['wins']:<5}"
          f"{mf_test['wins']:>5}/{mf_test['round_trips']-mf_test['wins']:<5}  │")
    print(f"  │ {'胜率':<20}{m0_train['win_rate']:>10.1f}%{mf_train['win_rate']:>10.1f}%"
          f"{m0_test['win_rate']:>10.1f}%{mf_test['win_rate']:>10.1f}%  │")
    print(f"  │ {'Buy&Hold':<18}{tr_bh:>+10.1f}%{'':>12}{te_bh:>+10.1f}%{'':>12}  │")
    print(footer)

    # ── 过拟合检测 ──
    print("\n  过拟合检测:")
    train_improve = mf_train["sharpe"] - m0_train["sharpe"]
    test_improve = mf_test["sharpe"] - m0_test["sharpe"]
    if train_improve > 0:
        ratio = test_improve / train_improve if train_improve != 0 else 0
    else:
        ratio = 0
    print(f"    训练集 Sharpe 提升: {train_improve:+.3f}")
    print(f"    测试集 Sharpe 提升: {test_improve:+.3f}")
    print(f"    样本外保持率: {ratio:.1%} (>50% 表示参数泛化良好)")

    if mf_test["sharpe"] > m0_test["sharpe"]:
        print("    结论: 最优参数在样本外依然有效，未过拟合")
    elif mf_test["sharpe"] > 0:
        print("    结论: 最优参数在样本外仍盈利，但提升有限，需关注")
    else:
        print("    结论: 最优参数在样本外表现下降，存在过拟合风险")

    print(f"\n  Phase 1 搜索了 {len(results1)} 组, Phase 2 搜索了 {len(results2)} 组")
    print(f"  总搜索: {len(results1) + len(results2)} 组参数组合")
    print("\n" + "=" * 80)
    print("优化完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
