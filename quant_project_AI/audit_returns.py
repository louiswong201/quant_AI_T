#!/usr/bin/env python3
"""
Full Audit of Backtest Returns
==============================
1. Data quality checks
2. Buy-and-hold baseline comparison
3. Independent return recalculation
4. Cost model verification
5. High-return reasonableness analysis
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from quant_framework.backtest import BacktestConfig, backtest
from quant_framework.backtest.kernels import (
    eval_kernel, eval_kernel_detailed, config_to_kernel_costs,
    run_kernel, run_kernel_detailed, _score,
)


def load_csv(path):
    df = pd.read_csv(path)
    return df


def print_header(title, width=80):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    # ═══════════════════════════════════════════════════════════
    #  AUDIT 1: Data Quality
    # ═══════════════════════════════════════════════════════════
    print_header("AUDIT 1: DATA QUALITY")

    symbols_to_check = ["BTC", "SOL", "MSTR", "AVAX", "XRP", "AMZN", "SPY", "NVDA"]
    for sym in symbols_to_check:
        path = os.path.join(data_dir, f"{sym}.csv")
        if not os.path.exists(path):
            continue
        df = load_csv(path)
        c = df["close"].values
        o = df["open"].values
        h = df["high"].values
        lo = df["low"].values

        n = len(df)
        date_start = df["date"].iloc[0]
        date_end = df["date"].iloc[-1]

        # Price range
        p_min, p_max = c.min(), c.max()
        total_return_pct = (c[-1] / c[0] - 1) * 100

        # Daily returns
        daily_rets = np.diff(c) / c[:-1]
        max_daily_gain = daily_rets.max() * 100
        max_daily_loss = daily_rets.min() * 100
        vol_annual = np.std(daily_rets) * np.sqrt(365) * 100

        # Data integrity checks
        gaps = np.where(np.diff(pd.to_datetime(df["date"]).values.astype("int64")) == 0)[0]
        n_gaps = len(gaps)
        ohlc_violations = np.sum((h < lo) | (h < c) | (h < o) | (lo > c) | (lo > o))
        neg_prices = np.sum((c <= 0) | (o <= 0) | (h <= 0) | (lo <= 0))
        nan_count = df[["open", "high", "low", "close"]].isna().sum().sum()

        print(f"\n  {sym}")
        print(f"    Period: {date_start} → {date_end}  ({n} bars)")
        print(f"    Price:  {p_min:.2f} → {p_max:.2f}")
        print(f"    Buy&Hold: {total_return_pct:+.1f}%")
        print(f"    Daily vol (ann): {vol_annual:.1f}%")
        print(f"    Max daily gain: {max_daily_gain:+.1f}%  |  Max daily loss: {max_daily_loss:+.1f}%")
        print(f"    OHLC violations: {ohlc_violations}  |  Neg prices: {neg_prices}  |  "
              f"NaN: {nan_count}  |  Dup dates: {n_gaps}")

        if total_return_pct > 500:
            print(f"    ⚠ Buy&Hold already {total_return_pct:+.0f}% — "
                  f"high strategy returns may be normal for this asset")

    # ═══════════════════════════════════════════════════════════
    #  AUDIT 2: Cost Model Verification
    # ═══════════════════════════════════════════════════════════
    print_header("AUDIT 2: COST MODEL VERIFICATION")

    for lev in [1, 2, 3, 5]:
        config = BacktestConfig(
            commission_pct_buy=0.0004,
            commission_pct_sell=0.0004,
            slippage_bps_buy=3.0,
            slippage_bps_sell=3.0,
            leverage=lev,
            allow_short=True,
            allow_fractional_shares=True,
            daily_funding_rate=0.0003,
            funding_leverage_scaling=True,
            stop_loss_pct=min(0.40, 0.80 / lev),
            position_fraction=1.0,
        )
        costs = config_to_kernel_costs(config)

        slippage_rt = (1 - costs["ss"] / costs["sb"]) * 100
        funding_daily = costs["dc"] * 100
        total_rt_cost = (1 - costs["ss"] / costs["sb"] + 2 * costs["cm"]) * 100

        print(f"\n  Leverage {lev}x:")
        print(f"    Slip buy:  {costs['sb']:.6f}  |  Slip sell: {costs['ss']:.6f}")
        print(f"    Round-trip slippage: {slippage_rt:.3f}%")
        print(f"    Commission: {costs['cm']*100:.3f}% per side")
        print(f"    Total round-trip cost: {total_rt_cost:.3f}%")
        print(f"    Daily funding: {funding_daily:.4f}%")
        print(f"    Annual funding: {funding_daily * 365:.2f}%")
        print(f"    Stop-loss: {costs['sl']*100:.0f}%  |  "
              f"Position frac: {costs['pfrac']:.2f}  |  "
              f"SL slippage: {costs['sl_slip']*100:.2f}%")

    # ═══════════════════════════════════════════════════════════
    #  AUDIT 3: Independent Return Verification
    # ═══════════════════════════════════════════════════════════
    print_header("AUDIT 3: INDEPENDENT RETURN VERIFICATION")
    print("  Compare kernel results with detailed equity curve recalculation")

    test_cases = [
        ("BTC", "MA", (10, 50), 1),
        ("SOL", "MACD", (29, 93, 3), 1),
        ("MSTR", "Turtle", (10, 19, 14, 3.0), 1),
        ("AVAX", "MA", (7, 16), 1),
        ("XRP", "MA", (3, 8), 1),
        ("SOL", "MACD", (29, 93, 3), 5),
        ("MSTR", "Turtle", (10, 19, 14, 3.0), 5),
    ]

    for sym, strat, params, lev in test_cases:
        path = os.path.join(data_dir, f"{sym}.csv")
        if not os.path.exists(path):
            continue
        df = load_csv(path)
        c = df["close"].values.astype(np.float64)
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        lo = df["low"].values.astype(np.float64)

        config = BacktestConfig(
            commission_pct_buy=0.0004, commission_pct_sell=0.0004,
            slippage_bps_buy=3.0, slippage_bps_sell=3.0,
            leverage=lev, allow_short=True, allow_fractional_shares=True,
            daily_funding_rate=0.0003, funding_leverage_scaling=True,
            stop_loss_pct=min(0.40, 0.80 / lev), position_fraction=1.0,
        )
        costs = config_to_kernel_costs(config)
        sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
        dc, sl, pfrac = costs["dc"], costs["sl"], costs["pfrac"]
        sl_slip = costs["sl_slip"]

        # Method 1: eval_kernel (fast)
        r1, d1, nt1 = eval_kernel(strat, params, c, o, h, lo,
                                   sb, ss, cm, lev, dc, sl, pfrac, sl_slip)

        # Method 2: eval_kernel_detailed (with equity curve)
        r2, d2, nt2, eq, _fpos, pos = eval_kernel_detailed(
            strat, params, c, o, h, lo,
            sb, ss, cm, lev, dc, sl, pfrac, sl_slip)

        # Method 3: high-level backtest()
        result = backtest(strat, params,
                          {"c": c, "o": o, "h": h, "l": lo},
                          config, detailed=True)

        # Buy & hold
        bnh = (c[-1] / c[0] - 1) * 100

        # Verify equity curve
        eq_ret = (eq[-1] / eq[0] - 1) * 100 if eq is not None and len(eq) > 0 else None

        # Count position changes from equity
        if pos is not None:
            pos_changes = np.sum(np.diff(pos) != 0)
        else:
            pos_changes = None

        match_12 = abs(r1 - r2) < 0.01
        match_13 = abs(r1 - result.ret_pct) < 0.01
        match_eq = abs(r2 - eq_ret) < 0.01 if eq_ret is not None else None

        status = "OK" if (match_12 and match_13 and match_eq) else "MISMATCH"

        print(f"\n  {sym}/{strat}{params} @ {lev}x  [{status}]")
        print(f"    eval_kernel:          ret={r1:+.1f}%  dd={d1:.1f}%  trades={nt1}")
        print(f"    eval_kernel_detailed: ret={r2:+.1f}%  dd={d2:.1f}%  trades={nt2}")
        print(f"    backtest(detailed):   ret={result.ret_pct:+.1f}%  "
              f"dd={result.max_dd_pct:.1f}%  trades={result.n_trades}")
        if eq_ret is not None:
            print(f"    Equity curve return:  {eq_ret:+.1f}%")
        print(f"    Buy&Hold:            {bnh:+.1f}%")
        print(f"    Strategy/B&H ratio:  {r1/bnh:.2f}x" if bnh != 0 else "    B&H = 0")
        if pos_changes is not None:
            print(f"    Position changes:    {pos_changes}  "
                  f"(avg hold: {len(c)/max(pos_changes,1):.0f} bars)")

    # ═══════════════════════════════════════════════════════════
    #  AUDIT 4: Extreme Return Reasonableness Check
    # ═══════════════════════════════════════════════════════════
    print_header("AUDIT 4: HIGH-RETURN REASONABLENESS")

    extreme_cases = [
        ("SOL", "MACD", (29, 93, 3), 1, 567.6),
        ("MSTR", "Turtle", (10, 19, 14, 3.0), 1, 303.9),
        ("MSTR", "Turtle", (10, 19, 14, 3.0), 5, 1066.6),
        ("XRP", "MA", (3, 8), 1, 341.9),
        ("AVAX", "MA", (7, 16), 1, 361.0),
    ]

    for sym, strat, params, lev, reported_ret in extreme_cases:
        path = os.path.join(data_dir, f"{sym}.csv")
        if not os.path.exists(path):
            continue
        df = load_csv(path)
        c = df["close"].values.astype(np.float64)
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        lo = df["low"].values.astype(np.float64)

        config = BacktestConfig(
            commission_pct_buy=0.0004, commission_pct_sell=0.0004,
            slippage_bps_buy=3.0, slippage_bps_sell=3.0,
            leverage=lev, allow_short=True, allow_fractional_shares=True,
            daily_funding_rate=0.0003, funding_leverage_scaling=True,
            stop_loss_pct=min(0.40, 0.80 / lev), position_fraction=1.0,
        )

        result = backtest(strat, params,
                          {"c": c, "o": o, "h": h, "l": lo},
                          config, detailed=True)

        bnh = (c[-1] / c[0] - 1) * 100
        n_years = len(c) / 365
        ann_strat = (1 + result.ret_pct / 100) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_bnh = (1 + bnh / 100) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Is strategy return within 2x of buy-and-hold? (possible for trend-following)
        ratio = result.ret_pct / bnh if bnh > 0 else float('inf')

        # Check if equity curve has suspicious jumps
        eq = result.equity
        if eq is not None and len(eq) > 1:
            bar_rets = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
            max_bar_gain = bar_rets.max() * 100
            max_bar_loss = bar_rets.min() * 100
            pct_long = np.sum(bar_rets > 0) / len(bar_rets) * 100
        else:
            max_bar_gain = max_bar_loss = pct_long = 0

        print(f"\n  {sym}/{strat}{params} @ {lev}x")
        print(f"    Reported:    {reported_ret:+.1f}%  |  Verified: {result.ret_pct:+.1f}%  "
              f"|  Match: {'YES' if abs(reported_ret - result.ret_pct) < 5 else 'NO'}")
        print(f"    Buy&Hold:    {bnh:+.1f}%  ({n_years:.1f} years)")
        print(f"    Strat/B&H:   {ratio:.2f}x")
        print(f"    Ann return:  Strategy {ann_strat*100:+.1f}%  |  B&H {ann_bnh*100:+.1f}%")
        print(f"    Max bar gain: {max_bar_gain:+.1f}%  |  Max bar loss: {max_bar_loss:+.1f}%")
        print(f"    Win rate:    {pct_long:.0f}%  |  Trades: {result.n_trades}")
        print(f"    Max DD:      {result.max_dd_pct:.1f}%")

        # Verdict
        flags = []
        if ratio > 5:
            flags.append("SUSPICIOUS: >5x buy-and-hold")
        if max_bar_gain > 50:
            flags.append(f"SUSPICIOUS: single bar +{max_bar_gain:.0f}%")
        if result.n_trades < 5:
            flags.append("WARNING: too few trades")
        if result.max_dd_pct > 90:
            flags.append("WARNING: extreme drawdown")
        if ratio <= 3 and bnh > 100:
            flags.append("PLAUSIBLE: strong asset with trend-following alpha")
        if not flags:
            flags.append("PLAUSIBLE: within reasonable bounds")

        for f in flags:
            print(f"    → {f}")

    # ═══════════════════════════════════════════════════════════
    #  AUDIT 5: Robust Scan OOS vs IS gap check
    # ═══════════════════════════════════════════════════════════
    print_header("AUDIT 5: IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print("  Run full-data scan (IS) and compare with OOS from robust_scan")

    from quant_framework.backtest.kernels import scan_all_kernels

    check_syms = ["BTC", "SOL", "MSTR", "SPY"]
    config_1x = BacktestConfig(
        commission_pct_buy=0.0004, commission_pct_sell=0.0004,
        slippage_bps_buy=3.0, slippage_bps_sell=3.0,
        leverage=1, allow_short=True, allow_fractional_shares=True,
        daily_funding_rate=0.0003, funding_leverage_scaling=True,
        stop_loss_pct=0.40, position_fraction=1.0,
    )

    print(f"\n  {'Symbol':<8} {'Strategy':<14} {'IS Ret':>8} {'IS DD':>7} "
          f"{'OOS Ret':>8} {'Shrink':>8}")
    print(f"  {'─'*8} {'─'*14} {'─'*8} {'─'*7} {'─'*8} {'─'*8}")

    from quant_framework.backtest.robust_scan import run_robust_scan

    for sym in check_syms:
        path = os.path.join(data_dir, f"{sym}.csv")
        if not os.path.exists(path):
            continue
        df = load_csv(path)
        c = df["close"].values.astype(np.float64)
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        lo = df["low"].values.astype(np.float64)

        # IS: scan on full data
        is_results = scan_all_kernels(c, o, h, lo, config_1x, n_threads=1)

        # OOS: robust scan
        data_d = {sym: {"c": c, "o": o, "h": h, "l": lo}}
        oos_result = run_robust_scan(
            [sym], data_d, config_1x,
            n_mc_paths=10, n_shuffle_paths=5, n_bootstrap_paths=5,
        )

        for sn in ["MA", "MACD", "RSI", "Turtle", "MESA", "RegimeEMA"]:
            is_r = is_results.get(sn, {})
            is_ret = is_r.get("ret", 0)
            is_dd = is_r.get("dd", 0)

            oos_d = oos_result.per_symbol.get(sym, {}).get(sn, {})
            oos_ret = oos_d.get("oos_ret", 0)

            shrink = ((is_ret - oos_ret) / abs(is_ret) * 100) if is_ret != 0 else 0

            print(f"  {sym:<8} {sn:<14} {is_ret:>+7.1f}% {is_dd:>6.1f}% "
                  f"{oos_ret:>+7.1f}% {shrink:>+7.0f}%")

    # ═══════════════════════════════════════════════════════════
    #  AUDIT 6: Position fraction & leverage sanity
    # ═══════════════════════════════════════════════════════════
    print_header("AUDIT 6: POSITION FRACTION AT LEVERAGE")

    from quant_framework.backtest.kernels import POSITION_FRAC

    print(f"  POSITION_FRAC table: {POSITION_FRAC}")
    print()
    for lev in [1, 2, 3, 5, 10]:
        pf = POSITION_FRAC.get(lev, max(0.10, 1.0 / (lev ** 0.5)))
        print(f"  {lev}x leverage → position_fraction = {pf:.2f}  "
              f"(effective exposure = {lev * pf:.1f}x)")

    print_header("AUDIT COMPLETE")


if __name__ == "__main__":
    main()
