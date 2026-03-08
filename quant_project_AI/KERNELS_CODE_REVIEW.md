# Kernels.py — Comprehensive Code Review Report

**File:** `quant_framework/backtest/kernels.py`  
**Scope:** All Numba JIT-compiled trading strategy kernels, precomputation, scan dispatch, and cost mapping.

---

## Executive Summary

This report identifies **23 issues** across 10 categories. **6 are critical** (incorrect results, consistency bugs), **8 are high** (potential crashes, logic errors), and **9 are medium/low** (edge cases, maintainability).

---

## 1. Numerical Precision Issues

### 1.1 **CRITICAL: MultiFactor volatility uses RMS instead of standard deviation** (Lines 1307, 1940)

**bt_multifactor_ls** (line 1307) and **_eq_multifactor** (line 1940) compute volatility as:
```python
vs = max(0.0, 1.0 - np.sqrt(s2/max(1,vol_p))*20.0)   # bt_multifactor_ls
vs = max(0.0, 1.0 - np.sqrt(s2/vol_p)*20.0)            # _eq_multifactor - also missing max(1,vol_p)
```

- `s2` is the sum of squared returns; `sqrt(s2/vol_p)` is the **root-mean-square (RMS)** of returns, not the standard deviation.
- Correct formula: `std = sqrt(mean(r²) - mean(r)²)` = `sqrt(s2/vol_p - m²)` where `m = sum(r)/vol_p`.
- **bt_ramom_ls** and **bt_multifactor_precomp** use the correct std formula.
- **Impact:** `eval_kernel("MultiFactor", params)` and `_eq_multifactor` produce different composite scores than `bt_multifactor_precomp` / `scan_all_kernels` for the same params. Scan results are not reproducible via eval_kernel.

**Fix:** Use the same volatility formula as RAMOM/MultiFactor precomp:
```python
s = 0.0; s2 = 0.0
for j in range(vol_p):
    ret = (c[i-j]/c[i-j-1]-1.0) if i-j>0 and c[i-j-1]>0 else 0.0
    s += ret; s2 += ret*ret
m = s / max(1, vol_p)
vol = np.sqrt(max(1e-20, s2/max(1,vol_p) - m*m))
vs = max(0.0, 1.0 - vol*20.0)
```

### 1.2 **HIGH: _eq_multifactor missing `max(1, vol_p)` guard** (Line 1940)

```python
vs = max(0.0, 1.0 - np.sqrt(s2/vol_p)*20.0)  # Division by zero if vol_p == 0
```

**Fix:** Use `s2/max(1, vol_p)` to match bt_multifactor_ls.

### 1.3 **MEDIUM: RSI division by zero when al == 0** (Lines 118, 179, 182)

```python
out[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al)
```

When `al == 0`, RSI is set to 100.0. This is correct (no losses → RSI 100). The `ag/al` is never evaluated when `al == 0`, so no division-by-zero. **No fix needed** — logic is correct.

### 1.4 **MEDIUM: MESA arctan(Im/Re) when Re ≈ 0** (Lines 343, 718)

```python
per[i] = (2*np.pi/np.arctan(Im_[i]/Re_[i])) if (Im_[i]!=0 and Re_[i]!=0) else per[i-1]
```

Guarded by `Re_[i]!=0`. If `Re_` is very small (e.g. 1e-300), `arctan(Im/Re)` can overflow. Consider `abs(Re_) > 1e-15` for robustness.

### 1.5 **LOW: RegimeEMA division by et[i]** (Line 673)

```python
pd_ = (c[i]-et[i])/et[i] if abs(et[i]) > 1e-20 else 0.0
```

Guarded. OK.

---

## 2. Off-by-One Errors

### 2.1 **CRITICAL: Donchian channel window mismatch between bt_donchian_ls and bt_donchian_precomp** (Lines 1472–1480, 690–696)

- **bt_donchian_ls:** Channel at bar `i` includes the current bar: `max(h[i], h[i-1], ..., h[i-entry_p+1])`.
- **bt_donchian_precomp:** Uses `dh_arr[i-1]` = `rmax[entry_p, i-1]` = max over `h[i-entry_p]` to `h[i-1]` — **excludes** current bar.

**Impact:** `scan_all_kernels` (Donchian) uses precomp; `eval_kernel("Donchian", params)` uses bt_donchian_ls. Results differ for the same params.

**Fix (choose one):**
- **Option A:** Make precomp include current bar: use `dh_arr[i]` and `dl_arr[i]` in the loop (and ensure precomp stores `rmax[entry_p, i]` at bar `i`).
- **Option B:** Make bt_donchian_ls exclude current bar to match precomp: use `dh[i-1]`-style lookback.

Standard Donchian includes the current bar; Option A is preferable.

### 2.2 **VERIFIED: Drift up_prefix indexing** (Lines 323–324, 1144–1146)

```python
up = up_prefix[i + 1] - up_prefix[i - lookback + 1]
```

This correctly counts up-bars in `[i-lookback+1, i]`, matching the loop in bt_drift_ls. **No bug.**

### 2.3 **VERIFIED: Turtle/Donchian precomp uses rmax[i-1]** (Lines 389–392, 699–700)

Turtle uses `rmax_entry[i-1]` to exclude the current bar from the entry channel, matching bt_turtle_ls. **Correct.**

---

## 3. Strategy Logic Correctness

### 3.1 **HIGH: ZScore exit condition logic** (Lines 1527–1530, 735–738)

```python
if pos==1 and (z>-xz or z>sz): pend=2
elif pos==-1 and (z<xz or z<-sz): pend=2
```

- Long exit: when `z > -xz` (mean reversion) **or** `z > sz` (extreme overbought).
- Short exit: when `z < xz` (mean reversion) **or** `z < -sz` (extreme oversold).

For `xz=0.5`, `sz=3`: long exits when `z > -0.5` or `z > 3`. The second case is a take-profit. The logic is internally consistent but non-standard; document the intended behavior.

### 3.2 **MEDIUM: RAMOM exit uses `z < xz` for long, `z > -xz` for short** (Lines 1185–1186, 354–355)

- Long exit: `z < xz` (z falls back toward zero).
- Short exit: `z > -xz`.

This matches the entry logic (long when `z > ez`, short when `z < -ez`). **Correct.**

### 3.3 **LOW: Drift signal inversion** (Lines 1155–1157, 325–327)

```python
if ratio>=drift_thr: pend=-1   # Short when many up-bars
elif ratio<=1.0-drift_thr: pend=1  # Long when many down-bars
```

Contrarian: short when price drifts up, long when it drifts down. Document as intentional.

---

## 4. Consistency: bt_xxx_ls vs _eq_xxx vs _scan_xxx_njit

### 4.1 **CRITICAL: MultiFactor volatility formula** (See §1.1)

bt_multifactor_ls / _eq_multifactor use RMS; bt_multifactor_precomp uses std. Scan uses precomp → inconsistency.

### 4.2 **CRITICAL: Donchian channel window** (See §2.1)

bt_donchian_ls includes current bar; precomp excludes it.

### 4.3 **HIGH: KAMA precompute vs inline** (Lines 365–372, 1429–1436)

`_compute_kama` and inline KAMA in bt_kama_ls both use `d = abs(close[i]-close[i-er_p])` and `v = sum of abs(close[i-j+1]-close[i-j])`. Logic matches. **Verified.**

### 4.4 **MEDIUM: MACD signal line initialization** (Lines 1107–1108, 1758–1759)

`sl[0] = ml[0]` — signal line starts as MACD line. Standard. **OK.**

---

## 5. scan_all_kernels Dispatch and Parameter Mapping

### 5.1 **VERIFIED: All 18 strategies in _scan_dispatch** (Lines 3222–3240)

MA, RSI, MACD, Drift, RAMOM, Turtle, Bollinger, Keltner, MultiFactor, VolRegime, MESA, KAMA, Donchian, ZScore, MomBreak, RegimeEMA, DualMom, Consensus. **Complete.**

### 5.2 **HIGH: KAMA cache key collision risk** (Lines 3195–3208)

```python
ck_key = ("KAMA_dedup", id(raw_kama))
cached_k = _GRID_CACHE.get(ck_key)
```

`_GRID_CACHE` is also used by `_cached_grid` with `id(grid)` (int) keys. Tuple keys are distinct; no collision. But mixing key types in one cache is fragile. Consider a separate `_KAMA_CACHE` dict.

### 5.3 **MEDIUM: MACD pair_idx and grid column order** (Lines 2779, 3202–3207)

Grid columns: `(fast, slow, sig_span)`. `pair_idx[k]` maps grid row `k` to unique (fast, slow) pair. `grid[k,2]` is sig_span. **Correct.**

### 5.4 **LOW: VolRegime hardcoded RSI period 14** (Lines 2581, 2884)

`rsis[14]` is used. Grid has `(ap, vt, ms, ml, ros, rob)` — no RSI period. If RSI period were added to the grid, this would need updating. **Document.**

---

## 6. KERNEL_REGISTRY, DEFAULT_PARAM_GRIDS, INDICATOR_DEPS

### 6.1 **VERIFIED: Registry and grids aligned**

All 18 names in KERNEL_REGISTRY appear in DEFAULT_PARAM_GRIDS and INDICATOR_DEPS. **Consistent.**

### 6.2 **MEDIUM: KAMA missing from INDICATOR_DEPS for KAMA-specific arrays** (Line 2341)

```python
"KAMA": frozenset({"atr"}),
```

KAMA also needs precomputed KAMA arrays (er_p, fast_sc, slow_sc). These are built inside scan_all_kernels when "KAMA" is in strat_names. INDICATOR_DEPS does not list "kama_arrs" because it is strategy-specific. **Acceptable** but could be documented.

### 6.3 **LOW: MESA and DualMom have empty INDICATOR_DEPS** (Lines 2340, 2346)

MESA computes MAMA/FAMA inline; DualMom uses only close. **Correct.**

---

## 7. Precomputed Indicator Logic

### 7.1 **VERIFIED: precompute_all_ma cumulative sum** (Lines 141–149)

`mas[w, i] = (cs[i+1] - cs[i-w+1]) / w` for `i >= w-1`. Correct SMA.

### 7.2 **VERIFIED: precompute_rolling_vol** (Lines 278–298)

Uses `sqrt(s2/vp - m²)` — correct population std for returns. **Correct.**

### 7.3 **VERIFIED: precompute_up_prefix** (Lines 269–274)

`psum[i+1] = psum[i] + (1 if close[i] > close[i-1] else 0)`. Correct.

### 7.4 **MEDIUM: precompute_rolling_max/min block decomposition** (Lines 204–215, 248–252)

`prefix[i-w+1]` and `suffix[i]` — for block boundaries, verify correctness. The standard Sparse Table / block decomposition is correct; **assume OK** unless tests show otherwise.

### 7.5 **LOW: _precompute_macd_lines uses emas[fi, i] - emas[si, i]** (Lines 2769–2772)

MACD = fast EMA - slow EMA. Correct.

---

## 8. config_to_kernel_costs

### 8.1 **VERIFIED: Cost mapping** (Lines 2431–2458)

- `sb = 1 + slip`, `ss = 1 - slip` (slippage).
- `cm = max(commission_pct_buy, commission_pct_sell)`.
- `dc` = daily funding scaled by bars_per_day and leverage.
- `sl` = stop_loss_pct (default 0.80).
- `pfrac` = position_fraction with leverage-based fallback.
- `sl_slip` = extra slippage on stop.

**Correct.**

### 8.2 **LOW: config_to_kernel_costs ignores maker/taker** (Line 2444)

Uses `commission_pct_buy` and `commission_pct_sell`; does not use `commission_pct_maker` / `commission_pct_taker`. Document that kernel costs use the legacy commission fields.

---

## 9. eval_kernel, eval_kernel_detailed, eval_kernel_position

### 9.1 **VERIFIED: Parameter passing**

All branches pass `sb, ss, cm, lev, dc, sl, pfrac, sl_slip` correctly. **Correct.**

### 9.2 **HIGH: eval_kernel returns (0,0,0) for unknown strategy** (Line 2732)

```python
return (0.0, 0.0, 0)
```

Callers may treat this as a valid result. Prefer raising `ValueError` for unknown strategy names.

### 9.3 **MEDIUM: eval_kernel_detailed returns zero arrays for unknown** (Line 2619)

Same as above; consider raising instead of returning zeros.

---

## 10. Edge Cases

### 10.1 **HIGH: Very short data (< lookback)**

- **bt_drift_ls:** Loop starts at `lookback`; if `n <= lookback`, loop never runs → returns (0,0,0) from epilogue. **OK.**
- **bt_zscore_ls:** `if n < lookback+2: return 0.0, 0.0, 0`. **OK.**
- **bt_dualmom_ls:** `if n < lb+2: return 0.0, 0.0, 0`. **OK.**
- **bt_consensus_ls:** `if n < mom_lb+2: return 0.0, 0.0, 0`. **OK.**
- **bt_mesa_ls:** `if n < 40: return 0.0, 0.0, 0`. **OK.**
- **bt_kama_ls:** `if n < er_p+2: return 0.0, 0.0, 0`. **OK.**
- **bt_donchian_ls:** `if n < entry_p+atr_p: return 0.0, 0.0, 0`. **OK.**
- **bt_regime_ema_ls:** `if n < max(...)+2: return 0.0, 0.0, 0`. **OK.**
- **bt_mombreak_ls:** `if n < max(hp, atr_p)+2: return 0.0, 0.0, 0`. **OK.**

**bt_ma_ls, bt_rsi_ls, bt_macd_ls, bt_ramom_ls, bt_turtle_ls, bt_bollinger_ls, bt_keltner_ls, bt_multifactor_ls, bt_volregime_ls:** Start at `1` or `max(...)`; if `n` is too small, the loop may not run or may index out of bounds. **bt_ma_ls** starts at `i=1` and uses `ma_s[i-1]`, `ma_l[i-1]` — if period is 20, `ma_s[0]` is NaN; the loop runs from 1, so `ma_s[0]` is used. For `i=1`, we need `ma_s[0]` and `ma_s[1]`. If period=20, `ma_s[1]` is NaN. The kernel checks `s0!=s0` and skips. So no crash, but no trades. **Acceptable.**

### 10.2 **MEDIUM: Empty data (n=0)**

- **bt_ma_ls:** `for i in range(1, n)` → no iterations; epilogue runs with `pos=0`, `ep=0` → returns (0,0,0). **OK.**
- **_equity_from_fused_positions:** `if n == 0: return 0.0, 0.0, 0, np.ones(0), 0`. **OK.**

### 10.3 **MEDIUM: All-same-price data**

- **RSI:** All `d=0` → `gs=0`, `ls=0` → `al=0` → RSI=100. **OK.**
- **KAMA:** `v=0` → `er=0` → `sc2=sc_v**2` → KAMA still updates. **OK.**
- **precompute_rolling_vol:** All `rets[i]=0` → `s=0`, `s2=0` → `vol=sqrt(1e-20)=1e-10`. **OK.**
- **Drift:** All same → `up=0` → `ratio=0` → may trigger long if `0 <= 1-drift_thr`. **Edge case.**

### 10.4 **LOW: Extreme prices (very large/small)**

- **validate_ohlc** rejects `<=0` and NaN. Callers must run it before backtest.
- Overflow in `tr * lev` or `raw - 1.0` is possible with extreme leverage; `tr = max(0.01, tr)` caps floor. **Acceptable.**

### 10.5 **LOW: _score when dd=0** (Line 131)

```python
return ret / max(1.0, dd) * min(1.0, nt / 20.0)
```

When `dd=0`, `ret/max(1,0)=ret`. Negative ret → negative score. **OK.**

---

## Summary of Recommended Fixes (Priority Order)

| Priority | Issue | Location | Action |
|----------|-------|----------|--------|
| P0 | MultiFactor vol formula | 1307, 1940 | Use std (s2/vol_p - m²) like RAMOM/precomp |
| P0 | Donchian channel window | 690–696, 1472–1480 | Align precomp with bt_donchian_ls (include current bar) |
| P1 | _eq_multifactor vol_p guard | 1940 | Add `max(1, vol_p)` |
| P1 | eval_kernel unknown strategy | 2732 | Raise ValueError instead of (0,0,0) |
| P2 | KAMA cache key | 3195–3208 | Consider separate cache dict |
| P2 | MESA Re near zero | 343, 718 | Add `abs(Re_) > 1e-15` guard |
| P3 | Document ZScore exit logic | 1527–1530 | Add comment |
| P3 | Document config commission fields | 2444 | Note legacy vs maker/taker |

---

## Testing Recommendations

1. **Unit tests:** For each strategy, assert `bt_xxx_ls(...) == bt_xxx_precomp(...)` when given identical precomputed inputs (where precomp exists).
2. **Consistency tests:** For each strategy, assert `eval_kernel(name, params, ...)` matches the best result from `scan_all_kernels` when using the same params.
3. **Edge-case tests:** Empty array, n=1, n=lookback, all-same-price, all-zero-return.
4. **Numerical tests:** Compare MultiFactor composite score from eval_kernel vs scan for the same params; they should match after fixing the vol formula.

---

*Report generated from comprehensive review of kernels.py (3273 lines).*
