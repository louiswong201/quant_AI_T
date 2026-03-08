# Strategy Code Review Report

**Scope:** All strategy files in `quant_project_AI/quant_framework/strategy/`  
**Reviewed:** 2025-03-07

---

## 1. Architecture Note: Base Strategy Interface

The base class uses `on_bar`, `on_bar_fast`, and `on_bar_fast_multi` — **not** `generate_signals()` or `calculate_indicators()`. Indicator computation is delegated to the backtest engine / `indicators.py`. All strategies correctly override `on_bar`; several also implement `on_bar_fast` and/or `on_bar_fast_multi`.

---

## 2. Critical Issues

### 2.1 **base_strategy.py** — Zero/negative price in position sizing

| Line | Issue |
|------|-------|
| 121-123 | `calculate_position_size()` does not guard against `price <= 0`. `amount / price` can raise `ZeroDivisionError` or produce `inf`/`nan`. |

**Fix:** Add `if price <= 0: return 0` at the start of `calculate_position_size`.

---

### 2.2 **ma_strategy.py** — Undefined `symbol` in `can_buy` / `can_sell`

| Line | Issue |
|------|-------|
| 65, 66 | `on_bar_fast` uses `symbol = data_arrays.get("symbol", "STOCK")`. If the engine does not set `"symbol"`, `can_buy("STOCK", ...)` is used while positions may be keyed by the real symbol (e.g. `"AAPL"`), causing incorrect checks. |

**Recommendation:** Document that the engine must set `"symbol"` in `data_arrays`, or add a fallback (e.g. from `current_prices` keys).

---

### 2.3 **rsi_strategy.py** — Inconsistency with kernel (long-only vs long/short)

| Line | Issue |
|------|-------|
| 77-98 | Python strategy is **long-only**: buy when RSI &lt; oversold, sell when RSI &gt; overbought. The kernel (`bt_rsi_ls`) is **long/short**: opens short when RSI &gt; overbought, exits long when RSI &gt; 50, exits short when RSI &lt; 50. |

**Impact:** Python strategy and kernel will produce different PnL and positions.

---

### 2.5 **macd_strategy.py** — Same long-only vs long/short mismatch

| Line | Issue |
|------|-------|
| 93-111 | Python strategy only trades long (MACD cross up → buy, cross down → sell). The kernel supports long and short (e.g. `pend=-3` to flip long to short). |

---

### 2.6 **drift_regime_strategy.py** — Long-only vs long/short

| Line | Issue |
|------|-------|
| 89-96, 135-140 | Python strategy only opens **long** when `up_ratio <= 1 - drift_threshold`. The kernel also opens **short** when `ratio >= drift_thr`. |

---

### 2.7 **drift_regime_strategy.py** — Dead code for short close

| Line | Issue |
|------|-------|
| 84-86, 131-133 | Logic for `holdings < 0` (close short) is unreachable because the strategy never opens short. |

---

### 2.8 **momentum_breakout_strategy.py** — Inconsistent ATR fallback

| Line | Issue |
|------|-------|
| 99-100, 151-152 | `on_bar_fast`: when ATR is NaN, uses `high[i] - low[i]`. `on_bar`: uses `atr = 1.0`. Different fallbacks can change behavior between fast and non-fast paths. |

**Recommendation:** Use the same fallback (e.g. `high - low` or a documented constant) in both paths.

---

### 2.9 **momentum_breakout_strategy.py** — Long-only vs long/short

| Line | Issue |
|------|-------|
| 112-119, 162-167 | Python strategy only opens long when price is near the high. The kernel also opens short when price is near the low (`c[i] <= lv * (1 + prox)`). |

---

### 2.10 **zscore_reversion_strategy.py** — Short selling vs base class

| Line | Issue |
|------|-------|
| 106-109, 151-154 | When `holdings == 0` and `z > entry_z`, the strategy returns `{"action": "sell", "shares": shares}` (open short). `base_strategy.can_sell()` requires `positions.get(symbol, 0) >= shares`, so it will reject this. The strategy assumes an engine that supports shorting. |

---

### 2.11 **zscore_reversion_strategy.py** — Z-score std: population vs sample

| Line | Issue |
|------|-------|
| 69-73 | `_compute_zscore` uses `np.std(window)` (default `ddof=1`, sample std). The kernel uses population variance: `sqrt(s2/lookback - m*m)` (ddof=0). Different formulas for small windows. |

**Fix:** Use `np.std(window, ddof=0)` for consistency with the kernel.

---

### 2.12 **zscore_reversion_strategy.py** — Exit/stop logic vs kernel

| Line | Issue |
|------|-------|
| 91-99, 136-144 | Python: long exit when `abs(z) < exit_z` or `z < -stop_z`; short exit when `abs(z) < exit_z` or `z > stop_z`. Kernel: long exit when `z > -xz` or `z > sz`; short exit when `z < xz` or `z < -sz`. Semantics differ (e.g. `z > sz` for long vs `z < -stop_z` for long). |

---

### 2.13 **kama_strategy.py** — No ATR, kernel uses ATR

| Line | Issue |
|------|-------|
| 102-139 | Python strategy uses only KAMA cross (price vs KAMA, KAMA slope). The kernel (`_eq_kama`, `bt_kama_ls`) uses ATR for stops. Python strategy does not implement ATR-based exits. |

---

### 2.14 **mesa_strategy.py** — Index safety for `mama[idx - 1]`

| Line | Issue |
|------|-------|
| 184 | `mama[idx - 1]` is used. With `_min_lookback = 40`, `idx >= 39`, so `idx - 1 >= 38`. Safe. No bug. |

---

### 2.15 **mesa_strategy.py** — MAMA/FAMA params in kernel

| Line | Issue |
|------|-------|
| kernels.py:324-357 | `_compute_mama_fama` in the kernel uses hardcoded `alpha = 0.5/dp` with limits 0.05 and 0.5. It does **not** take `fast_limit` / `slow_limit` from the strategy params. The Python strategy uses `_mama_fama_numba(close, self.fast_limit, self.slow_limit)`. Kernel and Python can diverge for non-default params. |

---

### 2.16 **lorentzian_strategy.py** — Numba decorator when Numba unavailable

| Line | Issue |
|------|-------|
| 52 | `_ema_1d = njit(...)(_ema_1d_impl) if NUMBA_AVAILABLE else _ema_1d_impl` — when Numba is unavailable, `njit` may not be defined if the import failed. The `if NUMBA_AVAILABLE` branch avoids calling `njit`, so this is safe. |

---

### 2.17 **lorentzian_strategy.py** — RSI vs indicators

| Line | Issue |
|------|-------|
| 55-86 | `_rsi_1d_impl` uses a different structure than `indicators._rsi_numba` (e.g. loop vs vectorized). Both use Wilder smoothing. Logic should be equivalent; consider reusing `_rsi_numba` for consistency. |

---

### 2.18 **microstructure_momentum.py** — Volume fallback

| Line | Issue |
|------|-------|
| 331 | When `"volume"` is missing, uses `np.ones(len(df), dtype=np.float64)`. OFI and VPIN assume real volume; constant volume will distort these metrics. |

**Recommendation:** Require volume or return `{"action": "hold"}` when volume is missing.

---

### 2.19 **microstructure_momentum.py** — Exception handling in `_compute_signal`

| Line | Issue |
|------|-------|
| 204-206 | `try/except (ZeroDivisionError, FloatingPointError)` around OFI RoC. Catching `FloatingPointError` can hide serious numerical issues. Consider narrowing the exception handling. |

---

### 2.20 **adaptive_regime_ensemble.py** — `np.nanstd` on slice with NaNs

| Line | Issue |
|------|-------|
| 243 | `ofi_std = np.nanstd(ofi[max(0, i - self.vol_slow):i + 1])` — `np.nanstd` ignores NaNs. If most values are NaN, result can be NaN. The subsequent `ofi_std + 1e-10` avoids division by zero but not downstream use of NaN. |

---

## 3. Position Sizing and Edge Cases

### 3.1 **base_strategy.py**

| Line | Issue |
|------|-------|
| 122 | `shares = int(amount / price)` — for very small `amount` or large `price`, `shares` can be 0. `max(0, shares)` is applied. No fractional shares; this is intentional. |

### 3.2 **zscore_reversion_strategy.py**

| Line | Issue |
|------|-------|
| 107-108 | For opening short, uses `calculate_position_size` (same as long). Short sizing may need different logic (e.g. margin) depending on the engine. |

---

## 4. Edge Cases: Insufficient Data, NaN, Zero Volume

### 4.1 **ma_strategy.py**

| Line | Issue |
|------|-------|
| 48-59 | Checks `i + 1 < self.long_window` and `pd.isna(short_ma) or pd.isna(long_ma)`. Handles insufficient data and NaN. |

### 4.2 **rsi_strategy.py**

| Line | Issue |
|------|-------|
| 70, 120-122 | Checks `len(df) < self.rsi_period + 1` and `pd.isna(rsi)`. Good. |

### 4.3 **macd_strategy.py**

| Line | Issue |
|------|-------|
| 71, 131-132 | Checks `len(df) < self.slow_period + self.signal_period + 1` and `pd.isna(macd) or pd.isna(signal)`. Good. |

### 4.4 **drift_regime_strategy.py**

| Line | Issue |
|------|-------|
| 67, 113 | Checks `close is None` and `i < self.lookback` / `len(df) < self.lookback + 1`. No explicit NaN handling for `close`; `np.diff` and `np.sum` can propagate NaN. |

### 4.5 **zscore_reversion_strategy.py**

| Line | Issue |
|------|-------|
| 71-72 | Returns 0 when `std == 0` to avoid division by zero. Good. |

### 4.6 **momentum_breakout_strategy.py**

| Line | Issue |
|------|-------|
| 99-100 | Handles `np.isnan(atr)` with a fallback. Good. |

### 4.7 **lorentzian_strategy.py**

| Line | Issue |
|------|-------|
| 368-369 | Returns `{"action": "hold"}` when `np.any(np.isnan(query))`. Good. |

### 4.8 **kama_strategy.py**

| Line | Issue |
|------|-------|
| 124-125 | Returns hold when `np.isnan(k_now) or np.isnan(k_prev)`. Good. |

### 4.9 **mesa_strategy.py**

| Line | Issue |
|------|-------|
| 172-173 | Checks `len(df) < self._min_lookback`. No explicit NaN check for `mama`/`fama`. |

---

## 5. Consistency with kernels.py

| Strategy | Python vs Kernel |
|----------|------------------|
| MA | Logic aligned (crossover). Python is long-only; kernel supports long/short. |
| RSI | Logic aligned for long. Python long-only; kernel long/short with RSI 50 exit. |
| MACD | Logic aligned for long. Python long-only; kernel long/short. |
| Drift | Logic aligned for long. Python long-only; kernel long/short. |
| ZScore | Std formula differs (sample vs population). Exit/stop logic differs. Python and kernel both support shorting. |
| MomBreak | Logic aligned for long. Python long-only; kernel long/short. ATR fallback differs. |
| KAMA | Python omits ATR-based exits used in the kernel. |
| MESA | Python uses `fast_limit`/`slow_limit`; kernel `_compute_mama_fama` uses fixed 0.5/0.05. |

---

## 6. Missing Error Handling and Bounds

### 6.1 **ma_strategy.py**

| Line | Issue |
|------|-------|
| 60-61 | `prev_short_ma = float(short[i - 1])` when `i > 0`. For `i == 0` uses `short_ma`. The guard `i + 1 < self.long_window` ensures we do not run with `i < long_window - 1`, so indexing is safe. |

### 6.2 **drift_regime_strategy.py**

| Line | Issue |
|------|-------|
| 71 | `window = close[i - self.lookback:i + 1]` — when `i == self.lookback`, `close[0:lookback+1]` is valid. Good. |

### 6.3 **lorentzian_strategy.py**

| Line | Issue |
|------|-------|
| 382 | `sma_val = np.mean(close[i - self.sma_period + 1: i + 1])` — when `i < sma_period - 1`, slice can be short or empty. The `_min_lookback` and `i + 1 >= _min_lookback` check should prevent this; worth verifying. |

### 6.4 **adaptive_regime_ensemble.py**

| Line | Issue |
|------|-------|
| 283-284 | `_inverse_vol_size` clamps `ann_vol` to 0.01 when too small. Good. |

---

## 7. Memory and Performance

### 7.1 **rsi_strategy.py**

| Line | Issue |
|------|-------|
| 146-148 | `_rsi_at_index` builds `segment = close[:i+1]` and runs `_rsi_numba` on it for every bar when precomputed RSI is not used. O(n) per bar, O(n²) over a full run. |

**Recommendation:** Precompute RSI once when possible, or cache by `(symbol, i)` with a bounded cache.

### 7.2 **lorentzian_strategy.py**

| Line | Issue |
|------|-------|
| 361-366 | `_feature_caches` and `_label_caches` are bounded by `_max_cache_symbols = 50`. Eviction uses `next(iter(...))` (oldest in insertion order). Good. |

### 7.3 **microstructure_momentum.py** / **adaptive_regime_ensemble.py**

| Line | Issue |
|------|-------|
| 167, 168, 214 | `_trailing_stop` and `_last_vpin` grow with the number of symbols. No explicit limit. For many symbols, consider a bounded cache or cleanup. |

---

## 8. Type and API Consistency

### 8.1 **macd_strategy.py**

| Line | Issue |
|------|-------|
| 178 | `on_bar_fast_multi` returns `signals or [{"action": "hold"}]`. Other strategies return `{"action": "hold"}` (single dict). Mixed return types for “hold” may require engine support for both. |

### 8.2 **lorentzian_strategy.py**

| Line | Issue |
|------|-------|
| 337 | `on_bar_fast_multi` returns `signals if signals else [{"action": "hold"}]`. Same pattern as MACD. |

### 8.3 **base_strategy.py**

| Line | Issue |
|------|-------|
| 37 | `positions: Dict[str, Union[int, float]]` — `calculate_position_size` returns `int`; `buy`/`sell` use `int` shares. Float is allowed but not used. |

---

## 9. Strategies Without kernel_name

| Strategy | kernel_name |
|----------|-------------|
| LorentzianClassificationStrategy | None (no kernel) |
| MicrostructureMomentum | None |
| AdaptiveRegimeEnsemble | None |

These are Python-only and do not need to match a kernel.

---

## 10. Summary of Recommended Fixes

| Priority | File | Fix |
|----------|------|-----|
| High | base_strategy.py | Guard `calculate_position_size` against `price <= 0`. |
| High | zscore_reversion_strategy.py | Use `np.std(..., ddof=0)` for kernel consistency. |
| High | momentum_breakout_strategy.py | Unify ATR fallback between `on_bar` and `on_bar_fast`. |
| Medium | rsi_strategy.py | Either add short logic to match the kernel or document long-only design. |
| Medium | macd_strategy.py | Same as RSI. |
| Medium | drift_regime_strategy.py | Same as RSI; remove dead short-close code or add short support. |
| Medium | momentum_breakout_strategy.py | Same as RSI for short support. |
| Medium | zscore_reversion_strategy.py | Align exit/stop logic with kernel or document differences. |
| Medium | kama_strategy.py | Add ATR-based exits to match the kernel, or document simplified behavior. |
| Medium | mesa_strategy.py | Ensure kernel `_compute_mama_fama` uses `fast_limit`/`slow_limit` from params. |
| Low | microstructure_momentum.py | Reject or handle missing volume explicitly. |
| Low | rsi_strategy.py | Optimize or cache `_rsi_at_index` when RSI is not precomputed. |

---

*End of report*
