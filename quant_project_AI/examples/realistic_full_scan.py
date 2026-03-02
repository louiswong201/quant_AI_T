#!/usr/bin/env python3
"""
最贴近现实交易的回测 — 参数探索 + 防过拟合 一步到位
===================================================

核心原则:
  参数搜索只在 walk-forward 训练窗口内进行
  所有验证/测试指标都来自优化器从未见过的数据
  MC/Shuffle/Bootstrap 也只在 OOS 数据上运行

用法:
    python examples/realistic_full_scan.py
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quant_framework.backtest import (
    BacktestConfig,
    DEFAULT_PARAM_GRIDS,
    run_robust_scan,
)


def load_all_data(data_dir: str, min_bars: int = 500):
    datasets = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".csv"):
            continue
        sym = f.replace(".csv", "")
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=["date"])
        if len(df) < min_bars:
            continue
        for col in ("close", "open", "high", "low"):
            if col not in df.columns:
                break
        else:
            datasets[sym] = {
                "c": df["close"].values.astype(np.float64),
                "o": df["open"].values.astype(np.float64),
                "h": df["high"].values.astype(np.float64),
                "l": df["low"].values.astype(np.float64),
                "n": len(df),
            }
    return datasets


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    datasets = load_all_data(data_dir, min_bars=500)

    if not datasets:
        print("No data with >= 500 bars found.")
        return

    syms = list(datasets.keys())
    print(f"Loaded {len(syms)} symbols: {', '.join(syms)}")

    # =================================================================
    # Step 1: 真实交易成本
    # =================================================================
    config = BacktestConfig(
        commission_pct_buy=0.0004,
        commission_pct_sell=0.0004,
        slippage_bps_buy=3.0,
        slippage_bps_sell=3.0,
        leverage=1.0,
        allow_short=True,
        allow_fractional_shares=True,
        daily_funding_rate=0.0003,
        funding_leverage_scaling=True,
        stop_loss_pct=0.40,
        stop_loss_slippage_pct=0.005,
        position_fraction=1.0,
    )

    # =================================================================
    # Step 2: 一步到位 — 参数探索 + 防过拟合
    # =================================================================
    # run_robust_scan 内部做的事:
    #   1. 把数据切成6个 train/val/test 窗口
    #   2. 在每个 train 窗口内搜索所有策略的最优参数 (scan_all_kernels)
    #   3. 用 val 窗口做验证门控 (过拟合检测)
    #   4. 用 test 窗口算真正的 OOS 收益
    #   5. 在 OOS 数据上跑 MC/Shuffle/Bootstrap 验证稳健性
    #   6. 用 Deflated Sharpe Ratio 修正多重检验偏差
    #   7. 综合排名

    print(f"\nRunning robust scan on {len(syms)} symbols, 18 strategies...")
    print(f"  (parameter search is inside walk-forward training windows only)\n")

    t0 = time.time()
    result = run_robust_scan(
        symbols=syms,
        data=datasets,
        config=config,
        # param_grids=None 表示用默认参数网格 (DEFAULT_PARAM_GRIDS)
        # 也可以自定义: param_grids={"MA": [(5,20),(10,50)], ...}
        n_mc_paths=20,
        n_shuffle_paths=10,
        n_bootstrap_paths=10,
    )
    elapsed = time.time() - t0

    print(f"Done: {result.total_combos:,} combos in {elapsed:.1f}s "
          f"({result.total_combos/max(0.1,elapsed):,.0f}/s)\n")

    # =================================================================
    # Step 3: 结果
    # =================================================================
    hdr = (f"  {'Symbol':<8} {'Best':<14} {'OOS Ret':>8} {'OOS DD':>8} "
           f"{'Sharpe':>7} {'DSR p':>7} {'MC>0':>5} {'Score':>7}")
    sep = f"  {'—'*8} {'—'*14} {'—'*8} {'—'*8} {'—'*7} {'—'*7} {'—'*5} {'—'*7}"
    print(hdr)
    print(sep)

    for sym in result.per_symbol:
        strats = result.per_symbol[sym]
        if not strats:
            continue
        best_sn = max(strats, key=lambda s: strats[s].get("wf_score", -1e18))
        b = strats[best_sn]
        print(f"  {sym:<8} {best_sn:<14} {b['oos_ret']:>+7.1f}% {b['oos_dd']:>7.1f}% "
              f"{b['sharpe']:>7.2f} {b['dsr_p']:>7.3f} "
              f"{b['mc_pct_positive']*100:>4.0f}% {b['wf_score']:>+6.1f}")

    # All strategies for first symbol
    sym0 = list(result.per_symbol.keys())[0]
    strats0 = result.per_symbol[sym0]
    print(f"\n  All strategies for {sym0}:")
    print(f"  {'Strategy':<14} {'Params':^26} {'OOS Ret':>8} {'Sharpe':>7} {'DSR':>7} {'MC>0':>5} {'Score':>7}")
    print(f"  {'—'*14} {'—'*26} {'—'*8} {'—'*7} {'—'*7} {'—'*5} {'—'*7}")
    for sn in sorted(strats0, key=lambda s: strats0[s].get("wf_score", -1e18), reverse=True):
        b = strats0[sn]
        ps = str(b.get("params", ""))[:24]
        print(f"  {sn:<14} {ps:^26} {b['oos_ret']:>+7.1f}% {b['sharpe']:>7.2f} "
              f"{b['dsr_p']:>7.3f} {b['mc_pct_positive']*100:>4.0f}% {b['wf_score']:>+6.1f}")


if __name__ == "__main__":
    main()
