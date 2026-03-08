# Round 3 Code Review Report — Quant Trading Framework

**Date:** 2025-03-07  
**Scope:** Verification of recent fixes + identification of new issues across 10 files

---

## 1. quant_framework/backtest/kernels.py

### _eq_multifactor fix (lines 1922–1950)
**Status: ✓ CORRECT**

`_eq_multifactor` is aligned with `bt_multifactor_ls` (lines 1291–1321):
- Same `start = max(rsi_p+1, mom_p, vol_p)`
- Same signal logic: `comp > lt` → long, `comp < st` → short
- Same exit logic: `comp < 0.5` for long, `comp > 0.5` for short
- Same volatility: `s2` over `vol_p` bars, `vs = max(0, 1 - sqrt(s2/vol_p)*20)`

### eval_kernel ValueError (line 2732)
**Status: ✓ CORRECT**

`eval_kernel` now raises `ValueError(f"Unknown strategy: {name}")` instead of returning a default tuple, which is appropriate for invalid strategy names.

### New issues

1. **eval_kernel_detailed fallback (lines 2617–2618)**  
   For unknown strategies, `eval_kernel_detailed` returns `(0.0, 0.0, 0, np.ones(...), 0, np.zeros(...))` instead of raising. This is inconsistent with `eval_kernel`, which raises. Callers may not detect invalid strategy names.

2. **bt_multifactor_ls vs _eq_multifactor — close handling**  
   Both use `_close_pos` at the end. `bt_multifactor_ls` uses an explicit close block (lines 1317–1320); `_eq_multifactor` uses `_close_pos` (line 1949). Behavior is equivalent.

3. **scan_all_kernels dispatch**  
   All 18 strategies are present in `_scan_dispatch` (lines 3222–3240). No missing or incorrect mappings.

---

## 2. quant_framework/backtest/robust_scan.py

### perturb_ohlc fix (lines 97–126)
**Status: ✓ CORRECT**

- Uses `body_hi = max(c_p[i], o_p[i])` and `body_lo = min(c_p[i], o_p[i])` so OHLC stays valid.
- Upper/lower wicks scaled separately.
- Post-checks ensure `h_p >= body_hi` and `l_p <= body_lo`.

### deflated_sharpe (lines 107–119)
**Status: ✓ CORRECT — n_bars usage**

- `se_sr` uses `(n_bars - 1)` in the denominator for the standard error of the Sharpe ratio.
- `n_bars` is used correctly for the deflation formula.

### robust_score (lines 122–136)
**Status: ✓ CORRECT**

- `years = max(0.1, n_bars / bars_per_year)` avoids division-by-zero.
- `deflated_sharpe(sharpe, n_trials, n_bars)` is called with the right arguments.

### CPCV purge logic (lines 688–733)
**Status: ✓ CORRECT**

Purge logic in `cpcv_splits` is correct:
- Train block ending in `(te_s - embargo, te_s]` → shrink `tr_e` to `te_s - embargo`.
- Train block starting in `[te_e, te_e + embargo)` → move `tr_s` to `te_e + embargo`.
- `purged_train` is used in splits; `_concat_ranges` receives purged ranges.

### New issues

1. **oos_total_bars in run_robust_scan (lines 405–407)**  
   `oos_total_bars = sum(int(n * te) - int(n * va) for _tr, va, te in windows)` sums test-segment lengths. For `robust_score`, this is a proxy for OOS bars. It does not account for embargo, so the bar count can be slightly high. Low impact.

2. **CPCV robust_score n_bars (line 321)**  
   `robust_score(..., n, bars_per_year=bpy)` uses full series length `n` instead of total OOS bars across splits. Deflated Sharpe is computed on a larger sample than the actual OOS evaluation. Consider using the sum of test-segment lengths.

---

## 3. quant_framework/live/trading_runner.py

### Shutdown fix
**Status: ⚠️ PARTIAL**

`_shutdown` (lines 328–334) sets `_running = False` and closes the journal but does **not** call `self._feed.stop_all()`. The feed may keep running after a signal. `stop()` (lines 416–419) correctly stops the feed. For clean exit on SIGINT/SIGTERM, `_shutdown` should also stop the feed (e.g. schedule `stop()` on the event loop).

### _on_bar (lines 135–247)
**Status: ✓ Logic correct**

- Entry-price tracking and averaging for adds are correct.
- PnL on close: long `(fill_price - entry) * shares`, short `(entry - fill_price) * shares`.
- `was_realized_close` correctly detects closing trades.

### New issues

1. **Entry-price averaging for shorts (lines 215–216)**  
   For short adds, `avg = (old_entry * abs(old_qty) + fill_price * filled_shares) / (abs(old_qty) + filled_shares)`. This is correct for short positions.

2. **can_buy not checked for short signals**  
   Strategy adapters may emit short signals without a `can_buy`-style check. If the broker restricts shorting, this could cause unexpected rejections. Consider a symmetric `can_sell` or similar.

3. **Stop-loss in _on_tick (lines 286–302)**  
   Uses `tick.running_low` / `tick.running_high`. If the feed does not maintain these correctly, SL/TP may be wrong. Depends on `TickEvent` implementation.

---

## 4. quant_framework/broker/binance_futures.py

### Fixes checked
- Order submission, position sync, and WebSocket handling appear consistent with the intended design.

### New issues

1. **sync_balance (lines 261–277)**  
   `cross_wallet` and `unrealized` are added to `total_equity` for **every** asset in the loop. For USDT-M futures, `unrealizedProfit` is typically account-wide (USDT). Adding it per asset can double-count or misattribute. Recommend restricting equity aggregation to the USDT row or confirming Binance’s per-asset semantics.

2. **Order status mapping (lines 184–192)**  
   Only `"filled"` is mapped to `"filled"`; all other statuses become `"submitted"`. `"PARTIALLY_FILLED"`, `"CANCELED"`, `"REJECTED"`, `"EXPIRED"` are not distinguished. Callers cannot tell partial fills or rejections from pending orders.

3. **get_order_status_async symbol fallback (lines 219–221)**  
   If `order_id` is unknown, it falls back to `next(iter(self._symbol_specs))`, which can return the wrong symbol and incorrect order status.

4. **_handle_order_update (lines 293–294)**  
   Empty implementation. WebSocket `ORDER_TRADE_UPDATE` events are not used to update local order/position state. Orders may appear “submitted” until the next REST sync.

---

## 5. quant_framework/broker/execution_algo.py

### Infinite loop fixes
**Status: ✓ CORRECT**

- **LimitChase (line 85):** `max_iterations = min(self._max_retries, 1000)` caps the loop.
- **Iceberg (lines 137–141):** `iterations > max_iterations` breaks the loop.

### Price calculation
**Status: ✓ CORRECT**

- LimitChase: `current_price += delta` for BUY, `current_price -= delta` for SELL.
- TWAP: fixed `qty_per_slice`; no price logic.

### New issues

1. **LimitChase — no fill detection**  
   If the order is `"submitted"` (e.g. limit not filled), the code still sleeps and retries with a new price. There is no check for `"partially_filled"` or similar. Partial fills are not accumulated; each iteration sends `total_qty` again, which can overfill.

2. **Iceberg — no timeout**  
   If the market never fills the display quantity, the loop runs until `max_iterations` (1000) with 1s sleeps (~16 minutes). Consider a wall-clock timeout.

---

## 6. quant_framework/strategy/drift_regime_strategy.py

### Short signal logic
**Status: ✓ CORRECT**

- `on_bar` (lines 150–155): `up_ratio >= self.drift_threshold` → sell (short). Matches `bt_drift_ls` (`ratio >= drift_thr` → `pend = -1`).
- `on_bar_fast` (lines 96–103): Same condition.

### up_ratio calculation
**Status: ✓ CORRECT**

- `on_bar`: `window = close_vals[n - self.lookback - 1:n]`, `up_days = np.sum(np.diff(window) > 0)`, `up_ratio = up_days / self.lookback`.
- Matches kernel: `up` counts `c[i-j+1] > c[i-j]` for `j in range(1, lookback+1)`.

### New issues

1. **Short path skips can_buy (lines 99–103)**  
   Long path uses `can_buy(symbol, current_price, shares)`; short path does not. If shorting is restricted, this can lead to rejected orders. Consider a `can_sell` or equivalent for shorts.

---

## 7. quant_framework/strategy/zscore_reversion_strategy.py

### Stop-loss vs kernel
**Status: ✓ CORRECT**

- **Kernel (bt_zscore_ls, lines 1528–1529):**  
  - Long: `(z > -xz or z > sz)` → exit when mean-reverting (`z > -xz`) or stop (`z > sz`).  
  - Short: `(z < xz or z < -sz)` → exit when mean-reverting (`z < xz`) or stop (`z < -sz`).
- **Strategy (lines 90–99):**  
  - Long: `abs(z) < exit_z` or `z > stop_z`.  
  - Short: `abs(z) < exit_z` or `z < -stop_z`.  
  Semantics match the kernel.

---

## 8. quant_framework/research/portfolio.py

### Cumulative-to-period return conversion (lines 49–58)
**Status: ✓ CORRECT**

- `equity[i] = 1 + ret_pct[i]/100` (cumulative equity).
- `daily_rets = np.diff(equity) / np.maximum(equity[:-1], 1e-12)` → period returns.
- Math: `(equity[i+1] - equity[i]) / equity[i]` is the correct period return.

### Assumption
`ret_pct` in `strategy_health` must be cumulative return from strategy start. If it is a single-period return, this conversion is wrong.

---

## 9. quant_framework/research/optimizer.py

### Champion demotion (lines 306–318)
**Status: ✓ CORRECT**

- `promote_challenger` finds the current champion via `_champion_name_for` and demotes it to RETIRED.
- `update_strategy_status` updates by `(name, kernel_name)`.

### New issues

1. **Multiple LIVE champions**  
   If multiple LIVE strategies exist for the same symbol/kernel, `_champion_name_for` returns the first non-challenger. Only one is demoted; others stay LIVE.

2. **composite_gate_score input (line 314)**  
   `cur_gate_approx = composite_gate_score(cur_metrics)` uses `cur.get("backtest_metrics", {})`. `composite_gate_score` expects keys like `sharpe`, `dsr_p`, `wf_score`, `oos_ret`, etc. If `backtest_metrics` uses different keys (e.g. `sharpe_30d`), the gate score can be wrong. Ensure schema alignment.

3. **evaluate_promotion symbol filtering (lines 269–274)**  
   Filters by `ch["kernel_name"] == strategy` and `symbol in ch_symbols`. Challengers from other symbols are correctly excluded.

---

## 10. quant_framework/research/database.py

### ROW_NUMBER() query (lines 212–224)
**Status: ✓ CORRECT**

```sql
SELECT * FROM (
    SELECT h.*,
           ROW_NUMBER() OVER (
               PARTITION BY h.symbol, h.strategy
               ORDER BY h.ts DESC
           ) as rn
    FROM strategy_health h
) ranked WHERE rn = 1
ORDER BY symbol, strategy
```

- `PARTITION BY symbol, strategy` groups by symbol/strategy.
- `ORDER BY ts DESC` picks the latest row per group.
- `rn = 1` keeps one row per group.  
  SQL is valid for SQLite 3.25+ (window functions).

---

## Summary Table

| File | Fix Status | New Issues |
|------|------------|------------|
| kernels.py | ✓ _eq_multifactor, ✓ eval_kernel | 1 (eval_kernel_detailed fallback) |
| robust_scan.py | ✓ perturb_ohlc, deflated_sharpe, CPCV | 2 (oos_total_bars, CPCV n_bars) |
| trading_runner.py | ⚠️ Shutdown incomplete | 2 (feed stop, can_sell) |
| binance_futures.py | ✓ | 4 (sync_balance, order status, symbol fallback, _handle_order_update) |
| execution_algo.py | ✓ Loop caps | 2 (LimitChase partial fill, Iceberg timeout) |
| drift_regime_strategy.py | ✓ Short signal | 1 (can_sell for short) |
| zscore_reversion_strategy.py | ✓ Stop-loss | 0 |
| portfolio.py | ✓ Conversion | 0 |
| optimizer.py | ✓ Demotion | 2 (multiple champions, gate score schema) |
| database.py | ✓ ROW_NUMBER | 0 |

---

## Recommended Priority Fixes

1. **High:** `_shutdown` should stop the feed for clean exit on signals.
2. **High:** Binance `sync_balance` — fix `total_equity` aggregation for USDT-M.
3. **Medium:** Binance order status mapping — handle PARTIALLY_FILLED, CANCELED, REJECTED, EXPIRED.
4. **Medium:** LimitChase — handle partial fills and avoid overfilling.
5. **Low:** `eval_kernel_detailed` — raise for unknown strategies instead of returning a default.
6. **Low:** Drift/ZScore strategies — add `can_sell` (or equivalent) for short signals.
