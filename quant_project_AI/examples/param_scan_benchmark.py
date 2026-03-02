#!/usr/bin/env python3
"""
极速全参数扫描 — 框架计算极限测试
=================================
3 标的 (AAPL/GOOGL/TSLA) × 3 年 (752 bars) × 全参数 0-200
- MA:   ~19,701 combos/symbol
- RSI:  ~11,751 combos/symbol (period × oversold × overbought)
- MACD: ~2,607,099 combos/symbol
- 合计: ~7.9M 回测

使用 Numba JIT 加速内层循环，指标一次性预计算。
"""
import numpy as np
import pandas as pd
import time, sys, os, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if not HAS_NUMBA:
    print("ERROR: numba not available, aborting.")
    sys.exit(1)

# ==================== Numba 加速内核 ====================

@njit(cache=True)
def _bt_ma(close, ma_short, ma_long, slip_buy, slip_sell, comm_pct):
    """MA 交叉回测：返回总收益率 (%)"""
    n = len(close)
    position = 0
    total_ret = 1.0
    entry_price = 0.0
    for i in range(1, n):
        ms_prev = ma_short[i - 1]
        ml_prev = ma_long[i - 1]
        ms = ma_short[i]
        ml = ma_long[i]
        if ms != ms or ml != ml or ms_prev != ms_prev or ml_prev != ml_prev:
            continue
        if ms_prev <= ml_prev and ms > ml and position == 0:
            entry_price = close[i] * slip_buy
            position = 1
        elif ms_prev >= ml_prev and ms < ml and position == 1:
            exit_price = close[i] * slip_sell
            trade_ret = (exit_price * (1.0 - comm_pct)) / (entry_price * (1.0 + comm_pct))
            total_ret *= trade_ret
            position = 0
    if position == 1:
        exit_price = close[n - 1] * slip_sell
        trade_ret = (exit_price * (1.0 - comm_pct)) / (entry_price * (1.0 + comm_pct))
        total_ret *= trade_ret
    return (total_ret - 1.0) * 100.0


@njit(cache=True)
def _bt_rsi(close, rsi, oversold, overbought, slip_buy, slip_sell, comm_pct):
    """RSI 回测：返回总收益率 (%)"""
    n = len(close)
    position = 0
    total_ret = 1.0
    entry_price = 0.0
    for i in range(1, n):
        r = rsi[i]
        if r != r:
            continue
        if r < oversold and position == 0:
            entry_price = close[i] * slip_buy
            position = 1
        elif r > overbought and position == 1:
            exit_price = close[i] * slip_sell
            trade_ret = (exit_price * (1.0 - comm_pct)) / (entry_price * (1.0 + comm_pct))
            total_ret *= trade_ret
            position = 0
    if position == 1:
        exit_price = close[n - 1] * slip_sell
        trade_ret = (exit_price * (1.0 - comm_pct)) / (entry_price * (1.0 + comm_pct))
        total_ret *= trade_ret
    return (total_ret - 1.0) * 100.0


@njit(cache=True)
def _bt_macd(close, ema_fast, ema_slow, signal_span, slip_buy, slip_sell, comm_pct):
    """MACD 交叉回测：返回总收益率 (%)"""
    n = len(close)
    macd_line = np.empty(n, dtype=np.float64)
    signal_line = np.empty(n, dtype=np.float64)
    for i in range(n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    k = 2.0 / (signal_span + 1.0)
    signal_line[0] = macd_line[0]
    for i in range(1, n):
        signal_line[i] = macd_line[i] * k + signal_line[i - 1] * (1.0 - k)
    position = 0
    total_ret = 1.0
    entry_price = 0.0
    for i in range(1, n):
        mp = macd_line[i - 1]
        sp = signal_line[i - 1]
        mc = macd_line[i]
        sc = signal_line[i]
        if mp != mp or sp != sp or mc != mc or sc != sc:
            continue
        if mp <= sp and mc > sc and position == 0:
            entry_price = close[i] * slip_buy
            position = 1
        elif mp >= sp and mc < sc and position == 1:
            exit_price = close[i] * slip_sell
            trade_ret = (exit_price * (1.0 - comm_pct)) / (entry_price * (1.0 + comm_pct))
            total_ret *= trade_ret
            position = 0
    if position == 1:
        exit_price = close[n - 1] * slip_sell
        trade_ret = (exit_price * (1.0 - comm_pct)) / (entry_price * (1.0 + comm_pct))
        total_ret *= trade_ret
    return (total_ret - 1.0) * 100.0


# ==================== 预计算 ====================

def precompute_all_ma(close, max_w=200):
    n = len(close)
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(close, out=cs[1:])
    mas = np.full((max_w + 1, n), np.nan, dtype=np.float64)
    for w in range(2, max_w + 1):
        mas[w, w - 1 :] = (cs[w:] - cs[: n - w + 1]) / w
    return mas


def precompute_all_ema(close, max_span=200):
    n = len(close)
    emas = np.empty((max_span + 1, n), dtype=np.float64)
    emas[:] = np.nan
    for s in range(2, max_span + 1):
        k = 2.0 / (s + 1.0)
        e = np.empty(n, dtype=np.float64)
        e[0] = close[0]
        for i in range(1, n):
            e[i] = close[i] * k + e[i - 1] * (1.0 - k)
        emas[s] = e
    return emas


def precompute_all_rsi(close, max_p=200):
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    rsi_all = np.full((max_p + 1, n), np.nan, dtype=np.float64)
    for p in range(2, max_p + 1):
        if n <= p:
            continue
        avg_g = np.mean(gain[1 : p + 1])
        avg_l = np.mean(loss[1 : p + 1])
        for i in range(p, n):
            if i > p:
                avg_g = (avg_g * (p - 1) + gain[i]) / p
                avg_l = (avg_l * (p - 1) + loss[i]) / p
            if avg_l == 0:
                rsi_all[p, i] = 100.0
            else:
                rsi_all[p, i] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return rsi_all


# ==================== Main ====================

def main():
    SLIP_BUY = 1.0 + 5.0 / 10000.0
    SLIP_SELL = 1.0 - 5.0 / 10000.0
    COMM_PCT = 0.0015
    MAX_W = 200

    symbols = ["AAPL", "GOOGL", "TSLA"]
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # --- JIT warm-up ---
    print("Warming up Numba JIT…")
    dc = np.random.randn(100).astype(np.float64) + 100
    dm = np.random.randn(100).astype(np.float64) + 100
    dr = np.random.rand(100).astype(np.float64) * 100
    _bt_ma(dc, dm, dm, SLIP_BUY, SLIP_SELL, COMM_PCT)
    _bt_rsi(dc, dr, 30.0, 70.0, SLIP_BUY, SLIP_SELL, COMM_PCT)
    _bt_macd(dc, dm, dm, 9, SLIP_BUY, SLIP_SELL, COMM_PCT)
    print("JIT ready.\n")

    all_results = {}
    grand_t0 = time.time()

    for sym in symbols:
        print(f"\n{'=' * 60}")
        print(f"  {sym}")
        print(f"{'=' * 60}")

        # --- Load data ---
        t0 = time.time()
        df = pd.read_csv(os.path.join(data_dir, f"{sym}.csv"), parse_dates=["date"])
        close = df["close"].values.astype(np.float64)
        n = len(close)
        t_load = time.time() - t0
        print(f"[{t_load:.3f}s] Loaded {n} bars")

        # --- Precompute MA ---
        t0 = time.time()
        mas = precompute_all_ma(close, MAX_W)
        t_ma_pre = time.time() - t0
        print(f"[{t_ma_pre:.4f}s] Precomputed {MAX_W - 1} MAs")

        # === MA SCAN ===
        t0 = time.time()
        best_ma_ret = -1e9
        best_ma_params = (0, 0)
        ma_count = 0
        for short_w in range(2, MAX_W):
            ms = mas[short_w]
            for long_w in range(short_w + 1, MAX_W + 1):
                ml = mas[long_w]
                ret = _bt_ma(close, ms, ml, SLIP_BUY, SLIP_SELL, COMM_PCT)
                if ret > best_ma_ret:
                    best_ma_ret = ret
                    best_ma_params = (short_w, long_w)
                ma_count += 1
        t_ma = time.time() - t0
        per_ma = t_ma / ma_count * 1000 if ma_count else 0
        print(f"[{t_ma:.2f}s] MA scan: {ma_count:,} combos @ {per_ma:.4f}ms/combo")
        print(f"  BEST: MA({best_ma_params[0]},{best_ma_params[1]}) = {best_ma_ret:+.2f}%")

        # --- Precompute RSI ---
        t0 = time.time()
        rsi_all = precompute_all_rsi(close, MAX_W)
        t_rsi_pre = time.time() - t0
        print(f"[{t_rsi_pre:.2f}s] Precomputed {MAX_W - 1} RSIs")

        # === RSI SCAN ===
        t0 = time.time()
        best_rsi_ret = -1e9
        best_rsi_params = (0, 0, 0)
        rsi_count = 0
        oversold_range = list(range(10, 45, 5))
        overbought_range = list(range(55, 95, 5))
        for p in range(2, MAX_W + 1):
            ra = rsi_all[p]
            for os_val in oversold_range:
                for ob_val in overbought_range:
                    ret = _bt_rsi(close, ra, float(os_val), float(ob_val), SLIP_BUY, SLIP_SELL, COMM_PCT)
                    if ret > best_rsi_ret:
                        best_rsi_ret = ret
                        best_rsi_params = (p, os_val, ob_val)
                    rsi_count += 1
        t_rsi = time.time() - t0
        per_rsi = t_rsi / rsi_count * 1000 if rsi_count else 0
        print(f"[{t_rsi:.2f}s] RSI scan: {rsi_count:,} combos @ {per_rsi:.4f}ms/combo")
        print(f"  BEST: RSI(period={best_rsi_params[0]}, os={best_rsi_params[1]}, ob={best_rsi_params[2]}) = {best_rsi_ret:+.2f}%")

        # --- Precompute EMA ---
        t0 = time.time()
        emas = precompute_all_ema(close, MAX_W)
        t_ema_pre = time.time() - t0
        print(f"[{t_ema_pre:.2f}s] Precomputed {MAX_W - 1} EMAs")

        # === MACD SCAN ===
        t0 = time.time()
        best_macd_ret = -1e9
        best_macd_params = (0, 0, 0)
        macd_count = 0
        for fast in range(2, MAX_W):
            ef = emas[fast]
            for slow in range(fast + 1, MAX_W + 1):
                es = emas[slow]
                for sig in range(2, min(slow, MAX_W + 1)):
                    ret = _bt_macd(close, ef, es, sig, SLIP_BUY, SLIP_SELL, COMM_PCT)
                    if ret > best_macd_ret:
                        best_macd_ret = ret
                        best_macd_params = (fast, slow, sig)
                    macd_count += 1
        t_macd = time.time() - t0
        per_macd = t_macd / macd_count * 1000 if macd_count else 0
        print(f"[{t_macd:.2f}s] MACD scan: {macd_count:,} combos @ {per_macd:.4f}ms/combo")
        print(f"  BEST: MACD(fast={best_macd_params[0]},slow={best_macd_params[1]},sig={best_macd_params[2]}) = {best_macd_ret:+.2f}%")

        all_results[sym] = {
            "ma": {"params": best_ma_params, "return": best_ma_ret, "combos": ma_count, "time": t_ma},
            "rsi": {"params": best_rsi_params, "return": best_rsi_ret, "combos": rsi_count, "time": t_rsi},
            "macd": {"params": best_macd_params, "return": best_macd_ret, "combos": macd_count, "time": t_macd},
            "precompute_time": t_ma_pre + t_rsi_pre + t_ema_pre,
        }

    grand_elapsed = time.time() - grand_t0

    # ==================== SUMMARY ====================
    print(f"\n\n{'=' * 70}")
    print(f"  GRAND SUMMARY")
    print(f"{'=' * 70}")
    total_combos = sum(
        v["ma"]["combos"] + v["rsi"]["combos"] + v["macd"]["combos"]
        for v in all_results.values()
    )
    print(f"Total combos evaluated: {total_combos:,}")
    print(f"Total wall time:        {grand_elapsed:.1f}s ({grand_elapsed / 60:.1f} min)")
    print(f"Throughput:             {total_combos / grand_elapsed:,.0f} combos/sec")

    print(f"\n--- Timing Breakdown (3 symbols combined) ---")
    total_ma_t = sum(v["ma"]["time"] for v in all_results.values())
    total_rsi_t = sum(v["rsi"]["time"] for v in all_results.values())
    total_macd_t = sum(v["macd"]["time"] for v in all_results.values())
    total_pre_t = sum(v["precompute_time"] for v in all_results.values())
    print(f"  Precompute (MA+RSI+EMA):  {total_pre_t:.2f}s  ({total_pre_t / grand_elapsed * 100:.1f}%)")
    print(f"  MA scan:                  {total_ma_t:.2f}s  ({total_ma_t / grand_elapsed * 100:.1f}%)")
    print(f"  RSI scan:                 {total_rsi_t:.2f}s  ({total_rsi_t / grand_elapsed * 100:.1f}%)")
    print(f"  MACD scan:                {total_macd_t:.2f}s  ({total_macd_t / grand_elapsed * 100:.1f}%)")

    print(f"\n--- Best Parameters by Symbol ---")
    print(f"{'Symbol':>6}  {'Strategy':>15}  {'Params':>30}  {'Return':>10}")
    print(f"{'------':>6}  {'--------':>15}  {'------':>30}  {'------':>10}")
    for sym, res in all_results.items():
        ma_p = res["ma"]["params"]
        print(f"{sym:>6}  {'MA':>15}  {'short=%d, long=%d' % ma_p:>30}  {res['ma']['return']:+.2f}%")
        rsi_p = res["rsi"]["params"]
        print(f"{'':>6}  {'RSI':>15}  {'p=%d, os=%d, ob=%d' % rsi_p:>30}  {res['rsi']['return']:+.2f}%")
        macd_p = res["macd"]["params"]
        print(f"{'':>6}  {'MACD':>15}  {'f=%d, s=%d, sig=%d' % macd_p:>30}  {res['macd']['return']:+.2f}%")

    # Bottleneck analysis
    print(f"\n--- Bottleneck Analysis ---")
    print(f"MACD scan dominates: {total_macd_t / grand_elapsed * 100:.1f}% of total time")
    print(f"  - 2.6M combos/symbol, each requires signal EMA recomputation")
    print(f"  - The Python for-loop over (fast, slow, signal) is the outer bottleneck")
    print(f"  - Numba inner loop ({per_macd:.4f}ms/combo) is already near-optimal for 752 bars")
    print(f"  - Further speedup options:")
    print(f"    1. Precompute MACD lines (ema_fast - ema_slow) for all pairs: eliminates signal EMA from inner loop")
    print(f"    2. Multiprocessing: split fast-slow pairs across CPU cores")
    print(f"    3. Cython/Rust extension: eliminate Python for-loop overhead entirely")
    print(f"    4. GPU (cupy/CUDA): batch all combos in parallel")

    return all_results


if __name__ == "__main__":
    main()
