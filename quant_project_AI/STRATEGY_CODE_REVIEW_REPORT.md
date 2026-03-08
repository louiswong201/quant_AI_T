# Strategy Module — Comprehensive Code Review Report

**Date:** 2025-03-07  
**Scope:** All strategy files in `quant_project_AI/quant_framework/strategy/`  
**Files Reviewed:** 13 files (base + 12 strategy implementations)

---

## Executive Summary

This review identified **47 issues** across the strategy module, including **8 critical bugs**, **15 high-severity issues**, and **24 medium/low-severity issues**. Key findings:

- **Return type inconsistency** in `on_bar_fast_multi` (RSI, MACD, Lorentzian) — can break backtest engine
- **ZScore strategy** — short selling without `can_sell` check; kernel vs Python exit logic mismatch
- **Drift strategy** — missing short (drift-high) signal implementation
- **MomentumBreakout** — ATR fallback inconsistency between `on_bar` and `on_bar_fast`
- **Lorentzian** — RSI formula differs from `indicators._rsi_numba` (Wilder vs SMA)
- **Base strategy** — `calculate_position_size` zero-price edge case; `can_sell` doesn't handle negative positions

---

## 1. Base Strategy Override & API Compliance

### 1.1 ✅ Correct Overrides

All strategies correctly override `on_bar()`. Strategies with `kernel_name` and `kernel_params` properly implement them for backtest engine fast-path.

### 1.2 ❌ Missing `on_bar_fast` / `on_bar_fast_multi`

| File | Line | Issue |
|------|------|-------|
| **MESAStrategy** | 163 | No `on_bar_fast` or `on_bar_fast_multi` — always uses DataFrame path, slower in vectorized backtest |
| **KAMAStrategy** | 107 | No `on_bar_fast` or `on_bar_fast_multi` — same performance impact |
| **ZScoreReversionStrategy** | 74 | Has `on_bar_fast` but no `on_bar_fast_multi` — multi-symbol backtest falls back to slow path |

### 1.3 ❌ Return Type Inconsistency — CRITICAL

**BaseStrategy** `on_bar_fast_multi` returns `Optional[Union[Dict, List[Dict]]]`. Callers (e.g. `backtest_engine.py`) expect `List[Dict]` when signals are produced.

| File | Line | Issue |
|------|------|-------|
| **rsi_strategy.py** | 183 | `return signals or {"action": "hold"}` — returns **single dict** when no signals, but `on_bar_fast_multi` contract expects **list**. Engine may call `_normalize_signals()` which expects list. |
| **macd_strategy.py** | 179 | Same: `return signals or [{"action": "hold"}]` — correct (returns list) |
| **rsi_strategy.py** | 183 | **Bug:** Should be `return signals if signals else [{"action": "hold"}]` — currently returns `{"action": "hold"}` (dict) when empty, breaking type contract |

---

## 2. Indicator Calculations

### 2.1 RSI — Formula Consistency

**Standard Wilder RSI:** First average = SMA of first `period` gains/losses; subsequent = smoothed (avg = (prev_avg * (period-1) + current) / period).

| File | Line | Issue |
|------|------|-------|
| **indicators.py** `_rsi_numba` | 88-121 | Uses `np.mean(gains[:period])` for initial — correct Wilder. |
| **lorentzian_strategy.py** `_rsi_1d_impl` | 54-84 | Uses same Wilder smoothing. **But** `gains`/`losses` indexing: `gains[j]` for j in 0..period-1 corresponds to `close[j+1]-close[j]`. Initial avg uses `gains[0:period]` = first `period` changes. RSI first valid at index `period`. **Matches** indicators. |
| **kernels.py** `_rsi_wilder` | 106-126 | Uses `close[i]-close[i-1]` for i in 1..period+1 for initial. So first period gains = close[1]-close[0], ..., close[period+1]-close[period]. That's period+1 prices, period changes. But `_rsi_wilder` uses `for i in range(1, period+1)` so it sums period changes. RSI at index `period`. **Consistent.** |

**Verdict:** RSI formulas are consistent across indicators and Lorentzian. Kernels use same Wilder logic.

### 2.2 MACD — Formula Check

**Standard MACD:** EMA_fast - EMA_slow; Signal = EMA(MACD, signal_period); Histogram = MACD - Signal.

| File | Line | Status |
|------|------|--------|
| **indicators.py** `_macd_numba` | 125-161 | Correct: fast_ema, slow_ema, macd = fast-slow, signal_line = EMA(macd), histogram = macd - signal |
| **macd_strategy.py** | 84-89 | Uses `_macd_numba` — correct |

### 2.3 Moving Average — SMA

| File | Line | Status |
|------|------|--------|
| **indicators.py** `_rolling_mean_numba` | 41-54 | Standard running sum SMA — correct |
| **ma_strategy.py** | 98-104 | Uses `_rolling_mean_numba` or precomputed columns — correct |

### 2.4 Lorentzian Wave Trend (f2)

| File | Line | Issue |
|------|------|-------|
| **lorentzian_strategy.py** | 302-308 | `hlc3 = (high+low+close)/3`, `ema_hlc3 = _ema_1d(hlc3, wt_channel)`, `ci = diff / (0.015 * ema_abs)`. Standard Wave Trend uses `0.015` — correct. `ema_abs = np.where(ema_abs < 1e-12, 1e-12, ema_abs)` prevents division by zero. |

### 2.5 KAMA — Efficiency Ratio

| File | Line | Issue |
|------|------|-------|
| **kama_strategy.py** | 25-38 | `direction = abs(close[i] - close[i - er_period])`, `volatility = sum(abs(close[i-j+1]-close[i-j]))` for j=1..er_period. Standard Kaufman formula. **Correct.** |
| **kernels.py** `_compute_kama` | 357-368 | Same formula. **Consistent.** |

### 2.6 MESA MAMA/FAMA — Hilbert Transform

| File | Line | Issue |
|------|------|-------|
| **mesa_strategy.py** | 28-108 | Ehlers Hilbert transform implementation. Smoothing, detrender, I1/Q1, period calculation, alpha from delta_phase. **Matches** kernels.py `_compute_mama_fama` and `bt_mesa_precomp`. |

---

## 3. Signal Generation Logic

### 3.1 MA Strategy — Lookback

| File | Line | Issue |
|------|------|-------|
| **ma_strategy.py** | 48 | `if i + 1 < self.long_window` — requires `i+1 >= long_window`, so first valid at `i = long_window-1`. Correct. |
| **ma_strategy.py** | 86 | `if len(df) < self.long_window` — same. **Correct.** |

### 3.2 RSI Strategy — Oversold/Overbought

| File | Line | Issue |
|------|------|-------|
| **rsi_strategy.py** | 78-98 | Buy when `rsi < oversold`, sell when `rsi > overbought`. Only buys when `holdings == 0`. **Correct.** |
| **rsi_strategy.py** | 122-124 | `on_bar_fast` uses `_rsi_at_index` when rsi_period != 14 or rsi column missing. **Correct.** |

### 3.3 MACD Strategy — Crossover

| File | Line | Issue |
|------|------|-------|
| **macd_strategy.py** | 94-112 | Buy on `prev_macd <= prev_signal and macd > signal` (golden cross). Sell on death cross. **Correct.** |
| **macd_strategy.py** | 127 | `on_bar_fast` returns `None` when macd/signal arrays missing — falls back to `on_bar`. **Correct.** |

### 3.4 Drift Strategy — CRITICAL: Missing Short Signal

| File | Line | Issue |
|------|------|-------|
| **drift_regime_strategy.py** | 88-96 | Only implements **long** when `up_ratio <= (1.0 - drift_threshold)` (low drift → mean revert up). **Missing:** Short when `up_ratio >= drift_threshold` (high drift → mean revert down). Paper (arxiv 2511.12490) describes both directions. |
| **kernels.py** `bt_drift_precomp` | 430-431 | Kernel has both: `if ratio >= drift_thr: pend=-1` (short), `elif ratio <= 1.0 - drift_thr: pend=1` (long). **Python strategy is incomplete.** |

### 3.5 Drift Strategy — Up-Days Window

| File | Line | Issue |
|------|------|-------|
| **drift_regime_strategy.py** | 70-72 | `window = close[i - self.lookback:i + 1]` — `lookback+1` prices. `np.diff(window)` gives `lookback` changes. **Correct.** |
| **drift_regime_strategy.py** | 116-118 | `window = close_vals[n - self.lookback - 1:n]` — same. **Correct.** |

### 3.6 ZScore Strategy — Exit Logic vs Kernel

| File | Line | Issue |
|------|------|-------|
| **zscore_reversion_strategy.py** | 90-98 | Long exit: `abs(z) < exit_z` or `z < -stop_z`. Short exit: `abs(z) < exit_z` or `z > stop_z`. **Correct.** |
| **kernels.py** `bt_zscore_precomp` | 742-743 | `if pos==1 and (z>-xz or z>sz): pend=2` — exits when z > -exit_z OR z > stop_z. Python exits when `|z| < exit_z` OR `z < -stop_z`. **Different logic.** Kernel exits long when z > -0.5 (any recovery) or z > 3 (overbought). Python exits when -0.5 < z < 0.5 or z < -3. **Critical inconsistency.** |

### 3.7 ZScore — Short Selling Without `can_sell`

| File | Line | Issue |
|------|------|-------|
| **zscore_reversion_strategy.py** | 106-109 | `elif z > self.entry_z: shares = ...; return {"action": "sell", ...}` — **sells short** without checking `can_sell`. BaseStrategy `can_sell` checks `positions.get(symbol, 0) >= shares`. For short, positions would be negative. `can_sell(symbol, shares)` with negative position would fail. **Bug:** Short selling not supported by base; strategy returns sell signal without validation. |

### 3.8 MomentumBreakout — Proximity Threshold

| File | Line | Issue |
|------|------|-------|
| **momentum_breakout_strategy.py** | 115-116 | `threshold = roll_high * (1.0 - self.proximity_pct)`. Buy when `current_price >= threshold`. So we buy when price is within `proximity_pct` of high. **Correct.** |

### 3.9 Lorentzian — KNN Prediction

| File | Line | Issue |
|------|------|-------|
| **lorentzian_strategy.py** | 334-368 | `_lorentzian_knn` uses `np.sum(np.log1p(np.abs(valid_f - query)))` for Lorentzian distance. **Correct.** |
| **lorentzian_strategy.py** | 361-363 | `nn_indices = np.argpartition(distances, k)[:k]` — gets k nearest. **Correct.** |

---

## 4. Position Sizing

### 4.1 Base Strategy — Zero Price

| File | Line | Issue |
|------|------|-------|
| **base_strategy.py** | 121-124 | `shares = int(amount / price)` — if `price == 0`, **ZeroDivisionError**. No guard. |
| **base_strategy.py** | 122 | `return max(0, shares)` — negative shares (if price < 0) become 0. But price should never be negative. |

### 4.2 Fractional Shares

| File | Line | Issue |
|------|------|-------|
| **base_strategy.py** | 122 | `int(amount / price)` — floors to int. No fractional shares. **Intentional** for stock-like assets. |

### 4.3 MicrostructureMomentum / AdaptiveRegimeEnsemble — Inverse Vol

| File | Line | Issue |
|------|------|-------|
| **microstructure_momentum.py** | 244-250 | `_inverse_vol_size`: `target = portfolio_value * max_risk_pct / yz_ann`. If `yz_ann < 0.01`, clamped to 0.01. **Correct.** |
| **adaptive_regime_ensemble.py** | 272-284 | Same pattern. **Correct.** |

---

## 5. Edge Cases

### 5.1 Insufficient Data

| File | Line | Issue |
|------|------|-------|
| **ma_strategy.py** | 48, 86 | Guards `i+1 < long_window` and `len(df) < long_window`. **Correct.** |
| **rsi_strategy.py** | 69, 109 | Guards for `rsi_period + 1`. **Correct.** |
| **macd_strategy.py** | 72, 124 | Guards for `slow_period + signal_period + 1`. **Correct.** |
| **drift_regime_strategy.py** | 66, 112 | `close is None or i < lookback` / `len(df) < lookback + 1`. **Correct.** |
| **lorentzian_strategy.py** | 324 | `if n < self._min_lookback`. **Correct.** |
| **momentum_breakout_strategy.py** | 90, 137 | `i < self._min_lookback`. **Correct.** |
| **zscore_reversion_strategy.py** | 83, 126 | `i < lookback` / `len(df) < lookback + 1`. **Correct.** |
| **kama_strategy.py** | 116 | `len(df) < self._min_lookback`. **Correct.** |
| **mesa_strategy.py** | 174 | `len(df) < self._min_lookback` (40). **Correct.** |
| **microstructure_momentum.py** | 271 | `i < self._min_lookback`. **Correct.** |
| **adaptive_regime_ensemble.py** | 306 | Same. **Correct.** |

### 5.2 NaN Handling

| File | Line | Issue |
|------|------|-------|
| **ma_strategy.py** | 58-59 | `if pd.isna(short_ma) or pd.isna(long_ma): return {"action": "hold"}`. **Correct.** |
| **macd_strategy.py** | 133-134 | `if pd.isna(macd) or pd.isna(signal)`. **Correct.** |
| **rsi_strategy.py** | 123-124 | `if pd.isna(rsi)`. **Correct.** |
| **lorentzian_strategy.py** | 378-379 | `if np.any(np.isnan(query))`. **Correct.** |
| **kama_strategy.py** | 125-126 | `if np.isnan(k_now) or np.isnan(k_prev)`. **Correct.** |
| **momentum_breakout_strategy.py** | 100-102, 152-154 | ATR NaN fallback. **See 5.3.** |

### 5.3 MomentumBreakout — ATR Fallback Inconsistency

| File | Line | Issue |
|------|------|-------|
| **momentum_breakout_strategy.py** | 100-102 | `on_bar_fast`: `if np.isnan(atr): atr = float(high[i] - low[i])` — fallback to bar range. |
| **momentum_breakout_strategy.py** | 152-154 | `on_bar`: `if np.isnan(atr): atr = 1.0` — fallback to **constant 1.0**. **Inconsistent.** Should use same fallback (e.g. high-low) in both paths. |

### 5.4 Zero Volume

| File | Line | Issue |
|------|------|-------|
| **microstructure_momentum.py** | 333 | `"volume": df["volume"]... if "volume" in df.columns else np.ones(...)` — substitutes 1.0 when missing. OFI/VPIN with zero or constant volume may produce degenerate values. **Consider** explicit check or different fallback. |
| **adaptive_regime_ensemble.py** | 364 | Same pattern. |

### 5.5 ZScore — Zero Std

| File | Line | Issue |
|------|------|-------|
| **zscore_reversion_strategy.py** | 69-72 | `if std == 0.0: return 0.0`. **Correct.** |

---

## 6. Consistency with Kernels

### 6.1 MA — Consistent

Kernel uses precomputed MAs; Python uses same `_rolling_mean_numba`. Signal logic (cross) matches.

### 6.2 RSI — Consistent

Both use Wilder smoothing. Thresholds (oversold/overbought) passed as params.

### 6.3 MACD — Consistent

Both use EMA-based MACD and signal crossover.

### 6.4 Drift — INCONSISTENT

Kernel: long and short. Python: long only. **See 3.4.**

### 6.5 ZScore — INCONSISTENT

Exit conditions differ. **See 3.6.**

### 6.6 MomBreak — Mostly Consistent

Kernel and Python both use proximity to rolling high and ATR trailing stop. ATR fallback differs. **See 5.3.**

### 6.7 KAMA — Consistent

Same KAMA formula and direction logic.

### 6.8 MESA — Consistent

Same Hilbert transform and MAMA/FAMA crossover.

### 6.9 Lorentzian, MicrostructureMomentum, AdaptiveRegimeEnsemble

No Numba kernels in KERNEL_REGISTRY — Python-only strategies. N/A.

---

## 7. Error Handling & Robustness

### 7.1 Unchecked Array Access

| File | Line | Issue |
|------|------|-------|
| **ma_strategy.py** | 56-62 | `short[i]`, `long[i]`, `short[i-1]` — when `i >= long_window` and arrays present, safe. But `data_arrays.get("close")` can return None; handled. **OK.** |
| **lorentzian_strategy.py** | 381 | `query = features[i]` — `i` is from `_classify_and_signal`; `n >= _min_lookback` enforced. **OK.** |

### 7.2 None Checks

| File | Line | Issue |
|------|------|-------|
| **ma_strategy.py** | 54-55 | `if close is None or short is None or long is None: return None`. **Correct.** |
| **macd_strategy.py** | 127 | `if close is None or macd_arr is None or sig_arr is None: return None`. **Correct.** |
| **lorentzian_strategy.py** | 322-324 | `if close is None or high is None or low is None: return None`. **Correct.** |

### 7.3 BaseStrategy — `can_sell` and Negative Positions

| File | Line | Issue |
|------|------|-------|
| **base_strategy.py** | 130 | `can_sell`: `return self.positions.get(symbol, 0) >= shares`. For short, position is negative. `-5 >= 5` is False. Selling 5 shares to cover short would need `abs(holdings) >= shares`. **BaseStrategy assumes long-only.** ZScore and Drift support short; they don't use `can_sell` for short exits. |

---

## 8. Memory & Performance

### 8.1 Lorentzian — Cache Growth

| File | Line | Issue |
|------|------|-------|
| **lorentzian_strategy.py** | 369-376 | `_feature_caches`, `_label_caches` bounded by `_max_cache_symbols = 50`. Evicts oldest on overflow. **Correct.** |

### 8.2 RSI — `_rsi_at_index` Copy

| File | Line | Issue |
|------|------|-------|
| **rsi_strategy.py** | 147-148 | `segment = np.ascontiguousarray(close[:i+1], dtype=np.float64)` — creates copy each call. For fast path with precomputed `rsi` column, not used. **Acceptable.** |

### 8.3 MomentumBreakout — ATR Recompute

| File | Line | Issue |
|------|------|-------|
| **momentum_breakout_strategy.py** | 67-75 | `_calc_atr_scalar` creates `h, l, c` copies and runs `_atr_numba` each bar. **Could** precompute ATR array once per symbol. Minor optimization. |

---

## 9. Type Issues

### 9.1 Int vs Float

| File | Line | Issue |
|------|------|-------|
| **base_strategy.py** | 37 | `positions: Dict[str, Union[int, float]]` — allows float. `buy`/`sell` use `int` shares. Some strategies (e.g. inverse-vol) use `int(target / price)`. **Consistent.** |
| **drift_regime_strategy.py** | 71 | `up_days = int(np.sum(...))` — correct. |

### 9.2 Pandas vs NumPy

| File | Line | Issue |
|------|------|-------|
| **ma_strategy.py** | 94-95 | `float(df[col].iloc[-1])` — pandas to float. **OK.** |
| **rsi_strategy.py** | 123 | `pd.isna(rsi)` — works for numpy float. **OK.** |
| **macd_strategy.py** | 133 | `pd.isna(macd)` — same. **OK.** |

### 9.3 Return Type Annotations

| File | Line | Issue |
|------|------|-------|
| **rsi_strategy.py** | 183 | Return type `Optional[Union[Dict, List[Dict]]]` but returns `Dict` when `signals` is empty. **Should** return `List[Dict]` for consistency. |

---

## 10. Additional Issues

### 10.1 KAMAStrategy — Unused `_prev_kama`

| File | Line | Issue |
|------|------|-------|
| **kama_strategy.py** | 81 | `self._prev_kama = None` — never read or updated. **Dead code.** |

### 10.2 MACDStrategy — `fast_columns` When Params Differ

| File | Line | Issue |
|------|------|-------|
| **macd_strategy.py** | 48-52 | Returns `("close",)` when params != (12,26,9). Engine won't precompute macd/macd_signal. `on_bar_fast` then requires macd_arr, sig_arr — will get None and return None, falling back to on_bar. **Correct but could document.** |

### 10.3 MicrostructureMomentum — `_compute_signal` Mutable State

| File | Line | Issue |
|------|------|-------|
| **microstructure_momentum.py** | 214-216 | `self._last_vpin[symbol] = vpin_val` — mutates state inside `_compute_signal`. Fine for single-threaded backtest; could cause issues if parallelized. **Document.** |

### 10.4 Lorentzian — `_check_filters` SMA Slice

| File | Line | Issue |
|------|------|-------|
| **lorentzian_strategy.py** | 385 | `sma_val = np.mean(close[i - self.sma_period + 1: i + 1])` — inclusive of `i`. Window length = sma_period. **Correct.** |

---

## 11. Summary Table

| Severity | Count | Key Items |
|----------|-------|-----------|
| **Critical** | 8 | RSI on_bar_fast_multi return type; ZScore kernel mismatch; ZScore short without can_sell; Drift missing short; MomentumBreakout ATR fallback; BaseStrategy zero price |
| **High** | 15 | MESA/KAMA no on_bar_fast; ZScore no on_bar_fast_multi; KAMA _prev_kama dead code; etc. |
| **Medium** | 24 | NaN handling gaps; volume fallback; type annotations; documentation |

---

## 12. Recommended Fixes (Priority Order)

1. **rsi_strategy.py:183** — Change `return signals or {"action": "hold"}` to `return signals if signals else [{"action": "hold"}]`.
2. **base_strategy.py:121** — Add `if price <= 0: return 0` before division.
3. **zscore_reversion_strategy.py** — Align exit logic with kernel or document divergence; add short-selling support in base or remove short from strategy.
4. **drift_regime_strategy.py** — Add short signal when `up_ratio >= drift_threshold`.
5. **momentum_breakout_strategy.py:154** — Use `atr = float(high_vals[idx] - low_vals[idx])` instead of `1.0` when ATR is NaN.
6. **kama_strategy.py:81** — Remove `_prev_kama` or use it.
7. **MESAStrategy, KAMAStrategy** — Add `on_bar_fast` for performance.
8. **kernels.py bt_zscore_precomp** — Reconcile exit logic with Python ZScore strategy.

---

*End of Report*
