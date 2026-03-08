# Comprehensive Test Review & Testing Gaps Report

**Scope:** `quant_project_AI/` — test files, `robust_scan.py`, `optimizer.py`, `database.py`  
**Date:** 2025-03-07

---

## 1. Test Correctness Issues

### 1.1 Tests That Always Pass (Weak Assertions)

| File | Line | Test | Issue |
|------|------|------|-------|
| `tests/test_v52_fixes.py` | 151-153 | `E.7-base-capital-fraction-attr` | `hasattr(BaseStrategy, '_capital_fraction') or True` — the `or True` makes this **always pass** regardless of the attribute. |
| `tests/test_v52_fixes.py` | 256-258 | `B.7-backtest-engine-fallback` | `check("B.7-backtest-engine-fallback", True, ...)` — **always passes**; no actual verification of the fallback logic. |
| `tests/test_v4_research.py` | 111-112 | `DB-008 Trend DESC order` | `trend[0]["sharpe_30d"] <= trend[-1]["sharpe_30d"] or True` — **always passes**; the `or True` defeats the assertion. |
| `tests/test_v4_research.py` | 377-378 | `PF-011 Correlated pair penalised` | `xy_total < w_corr.get("Z", 0) + 0.3 or True` — **always passes**; correlation penalty not actually verified. |
| `tests/test_v4_research.py` | 406-407 | `DSC-010 Vol shift detected` | `len(anom_shift) > 0 or True` — **always passes**; anomaly detection not verified. |

### 1.2 Incorrect or Misleading Assertions

| File | Line | Test | Issue |
|------|------|------|-------|
| `tests/test_backtest.py` | 311 | `test_pending_order_partial_fill_keeps_remainder` | Asserts `row["status"] in ("partially_filled", "filled", "expired")` — too loose; doesn't verify that remainder was kept or that partial fill logic is correct. |
| `tests/test_backtest.py` | 311 | Same | `assert int(row.get("filled_shares", 0)) >= 0` — trivial; any non-negative value passes. |
| `tests/test_v5_full.py` | 328-329 | `expected_lp = 50000 * (1 - 1/lev + 0.004)` | Liquidation formula for long: `LP = Entry * (1 - 1/lev + MM_rate)`. The test uses `0.004` as MM; Binance tier 1 MM is 0.004. But `liquidation_price_long` may use different convention — verify against `CryptoFuturesMargin` implementation. |
| `tests/test_v52_fixes.py` | 352-354 | `ACCURACY-sharpe-manual-vs-engine` | `abs(sharpe_v - manual_sharpe) < 1.0` — tolerance of 1.0 Sharpe is very loose; a 1.0 Sharpe difference is economically significant. |
| `tests/test_v4_research.py` | 428-430 | `MATH-001 Constant returns` | Comment says "std=0 guard" but `expected = 0.001 / 0.0 if False else float("inf")` is dead code; assertion `abs(s) < 1e-6 or s == 0` may not match `_rolling_sharpe` behavior for constant returns. |

### 1.3 Missing Assertions / Incomplete Checks

| File | Line | Test | Issue |
|------|------|------|-------|
| `tests/test_backtest.py` | 291-293 | `test_market_liquidity_cap` | Uses `cfg.market_fill_mode` but doesn't assert which mode is active; `fill_bar` logic may be wrong if mode differs. |
| `tests/test_full_backtest_live.py` | 382-386 | CircuitBreaker test | Uses `assert cb2.check() is None` but `CircuitBreaker` may have `is_tripped` instead of `check()` — verify API. |
| `tests/test_live.py` | 29-31 | `test_reject_short_sell_when_not_allowed` | Passes `positions={"AAPL": 0}` — selling 10 when position is 0 is "insufficient position", but the test doesn't distinguish short-sell vs. over-sell. |

---

## 2. Test Coverage Gaps

### 2.1 Critical Code Paths With NO Tests

| Module | Missing Coverage | Risk |
|--------|------------------|------|
| **robust_scan.py** | `run_robust_scan`, `run_cpcv_scan`, `_process_one_symbol`, `perturb_ohlc`, `shuffle_ohlc`, `block_bootstrap_ohlc`, `deflated_sharpe`, `stitched_oos_metrics`, `robust_score` | High — core anti-overfitting logic untested |
| **optimizer.py** | `run_optimizer`, `param_neighborhood_stability` (only smoke-tested in v4_production), `evaluate_promotion` edge cases, `promote_challenger` | High |
| **database.py** | `upsert_strategy` ON CONFLICT behavior, `get_all_latest_health` with ties, migration/version handling | Medium |
| **BacktestEngine** | Market impact model, participation rate with fractional fills, `live_fills` reconciliation | Medium |
| **KillSwitch** | `flatten_all`, `is_triggered` — only `hasattr` checks in v5_full | High (safety) |
| **BinanceFuturesBroker / IBKRBroker** | No integration or unit tests; only structure checks | High |
| **FundingRateLoader** | `download` (network) not tested; `map_to_bars` only with synthetic data | Medium |
| **TWAP / LimitChase / Iceberg** | Only `hasattr(..., "execute")` — no execution logic tests | Medium |
| **TradeJournal** | `get_equity_curve`, edge cases for `pnl IS NOT NULL` filter | Low |
| **LatencyTracker** | Different modules: `test_live.py` uses `quant_framework.live`, `test_v52_fixes.py` uses `quant_framework.live.latency` — possible duplicate/rename | Low |

### 2.2 robust_scan.py — Zero Direct Tests

- **`cpcv_splits`** is tested in `test_kernel_regression.py` (geometry, embargo).
- **`run_robust_scan`** and **`run_cpcv_scan`** have **no unit or integration tests**.
- **`deflated_sharpe`**, **`robust_score`**, **`stitched_oos_metrics`** — no tests.
- **`perturb_ohlc`**, **`shuffle_ohlc`**, **`block_bootstrap_ohlc`** — no tests.
- **Walk-forward window calculation** (`tr_end`, `va_start`, `va_end`, `te_end`) — no tests.

### 2.3 optimizer.py — Partial Coverage

- `bayesian_param_update`, `composite_gate_score`, `_score_*` functions, `register_challenger`, `evaluate_promotion`, `promote_challenger` are tested in `test_v4_research.py`.
- **`run_optimizer`** — no dedicated test.
- **`param_neighborhood_stability`** — only one smoke test in `test_v4_production.py` (requires real data).

### 2.4 database.py — Partial Coverage

- CRUD for health, param_history, config_versions, regime, strategy_library is tested.
- **No tests for:** schema migrations, `ON CONFLICT` in `upsert_strategy`, connection pool under load, `get_all_latest_health` with multiple rows per (symbol, strategy).

---

## 3. Mock Issues

### 3.1 Over-Mocking / Hiding Real Bugs

| File | Issue |
|------|-------|
| `tests/test_backtest.py` | `_MockDataManager` replaces real `DataManager` — never tests real data loading, caching, or indicator calculation integration. |
| `tests/test_full_backtest_live.py` | Uses `yfinance` with fallback to synthetic data — if download fails, tests run on synthetic data and may miss real-data edge cases. |
| `tests/test_live_trading_async.py` | `SyntheticFeedManager` fully mocks `PriceFeedManager` — no test of real WebSocket/API feed. |
| `tests/test_v4_production.py` | Depends on `run_production_scan.load_daily`, `load_intraday` — if data missing, many tests skip or fail; not isolated. |

### 3.2 Mock Inconsistencies

| File | Line | Issue |
|------|------|-------|
| `tests/test_backtest.py` | 103-114 | `_MockDataManager` has `load_data(symbol, start_date, end_date)` but ignores `start_date`/`end_date` — returns full `_df`. Real `DataManager` would filter by date. |
| `tests/test_live_trading_async.py` | 94-98 | `_make_bar` uses `row.get("date", row.name)` — `sample_ohlcv` from `conftest` uses `date` as column; `generate_data` uses `date` column. Inconsistent index vs column. |

### 3.3 Missing Mocks for External Dependencies

- **CredentialManager.sign_request** — tested with a dummy secret; no test of real signing format.
- **RateLimiter** — only `_current_usage() == 0` checked; no test of actual rate limiting.
- **AuditTrail** — uses real temp file; no mock for DB.

---

## 4. Test Data Issues

### 4.1 Unrealistic or Insufficient Data

| File | Issue |
|------|-------|
| `conftest.py` | `sample_ohlcv` has 100 bars, `np.random.seed(42)` — no gaps, no zeros, no extreme volatility. Strategies may behave differently on real data. |
| `conftest.py` | `sample_ohlcv` uses `dates.strftime("%Y-%m-%d")` — date as string column; some code paths expect `DatetimeIndex`. |
| `tests/test_indicators.py` | `close_array` from `sample_ohlcv` — no test for NaN in input, zero volume, or OHLC violations (high < low). |
| `tests/test_backtest.py` | `test_limit_order_price_time_priority` sets `volume=1.0` on first bar — very artificial; may not generalize. |

### 4.2 Edge Cases Not Covered

- **Empty or single-row DataFrames** — `test_backtest.py` has `test_empty_data_raises` but not for indicators.
- **All-NaN columns** — indicators may not handle gracefully.
- **Negative prices** — not tested (should reject).
- **Zero volume** — participation rate logic may divide by zero.

---

## 5. Flaky Tests

### 5.1 Timing-Dependent

| File | Line | Test | Issue |
|------|------|------|-------|
| `tests/test_v5_full.py` | 443-445 | `PERF-{sz}bars` | `throughput > 50` — depends on CPU load; may fail on slow CI. |
| `tests/test_v5_full.py` | 456-458 | `PERF-scan` | `scan_tput > 100` — same. |
| `tests/test_v5_full.py` | 461-465 | `PERF-margin-model` | `mm_rate > 100_000` — machine-dependent. |
| `tests/test_v52_fixes.py` | 396-398 | `PERF-{sz}bars-throughput` | Same throughput thresholds. |
| `tests/test_v4_production.py` | 169-170 | `Speed < 500ms` | Health eval speed — depends on hardware. |
| `tests/test_v4_production.py` | 254-255 | `Speed < 10ms` | Regime detection speed. |

### 5.2 Network-Dependent

| File | Issue |
|------|-------|
| `tests/test_full_backtest_live.py` | `download_data()` uses `yfinance` — rate limits, network failures cause fallback to synthetic; test outcome changes. |
| `tests/test_v4_production.py` | Requires `run_production_scan.load_daily` / `load_intraday` — depends on local data files. |

### 5.3 Random-State-Dependent

| File | Issue |
|------|-------|
| `conftest.py` | `np.random.seed(42)` in `sample_ohlcv` — global seed; tests that run in different order or call `np.random` elsewhere may get different data. |
| `tests/test_v4_research.py` | Uses `np.random.seed(42)` and `np.random.seed(123)` in different tests — order-dependent if run in isolation. |
| `tests/test_v52_fixes.py` | `make_ohlcv(500, seed=12345)` — deterministic per call, but `test_strategy_fixes` uses `rng_zs = np.random.default_rng(99)` — separate RNG. |

---

## 6. robust_scan.py Deep Review

### 6.1 CPCV Implementation

| Location | Issue |
|----------|-------|
| `cpcv_splits` L530-535 | Purge logic: `if tr_e > te_s - embargo and tr_e <= te_s` → `tr_e = max(tr_s, te_s - embargo)`. The condition `tr_e <= te_s` means train ends at or before test start; the purge shortens train. But `tr_s >= te_e and tr_s < te_e + embargo` shortens the *start* of train. The logic is correct for purging boundaries. |
| `cpcv_splits` L519 | `group_size = n_bars // n_groups` — remainder bars are dropped. For `n_bars=100`, `n_groups=6`, last group gets 16 bars; first 5 get 16 each. Total = 96. **4 bars lost.** |
| `run_cpcv_scan` L674 | `robust_score(avg_ret, avg_dd, total_nt, n, ...)` — uses full `n` (all bars) for DSR, but OOS bars vary per split. Should use total OOS bars across splits for consistency. |
| `run_cpcv_scan` L676 | `cpcv_score = shrp * pct_pos * (1.0 + max(0.0, dsr_p - 0.5))` — formula is ad hoc; no reference to de Prado or standard CPCV scoring. |

### 6.2 Walk-Forward Window Calculation

| Location | Issue |
|----------|-------|
| L384-391 | `tr_end = int(n * tr_pct)`, `va_start = min(tr_end + embargo, int(n * va_pct))`, `va_end = int(n * va_pct)`, `te_end = min(int(n * te_pct), n)`. If `va_end > te_end`, `va_end = te_end`. If `va_start >= va_end`, `va_start = va_end` — validation can become empty. |
| L385 | `va_start = min(tr_end + embargo, int(n * va_pct))` — if `va_pct` is same as `tr_pct`, validation can be empty. For `(0.30, 0.40, 0.50)`, train=30%, val=30–40%, test=40–50%. Correct. |
| L354-355 | `oos_total_bars` uses `int(n * te) - int(n * va)` — this sums test segment lengths across windows. For overlapping windows, this may double-count. Actually it's per-window: `(tr, va, te)` so test is `va_end` to `te_end`. The sum is over windows: each window's test length. Correct. |

### 6.3 Monte Carlo Path Generation

| Location | Issue |
|----------|-------|
| `perturb_ohlc` L99-119 | Uses `np.random.seed(seed)` inside Numba — Numba's `np.random` is separate from NumPy's. Seeds are per-call; reproducibility depends on call order. |
| L117 | `uw = max(0.0, high[i] - max(close[i], open_[i]))` — upper wick. After perturbing close/open, `body_hi`/`body_lo` change; `uw`/`lw` use original OHLC. Perturbed high/low can violate OHLC (e.g. `h_p[i] < c_p[i]`). |
| `shuffle_ohlc` L134-139 | Fisher-Yates shuffle of O,H,L,C — reassigns roles. Output `o_p`, `c_p` from `rem[0]`, `rem[1]` — can produce invalid OHLC (e.g. open > high) if shuffle assigns non-extreme values to high/low. The `mx`/`mn` logic ensures high/low are correct; open/close are the other two. Valid. |

### 6.4 DSR Calculation

| Location | Issue |
|----------|-------|
| `deflated_sharpe` L199-213 | Uses Bailey & López de Prado (2014) formula. `e_max_sr` uses `(1 - 0.5772) * (2*ln(T))^0.5 + 0.5772 * (2*ln(T))^-0.5` — Euler constant 0.5772. `se_sr` uses skew and kurtosis. |
| L206 | `se_sr = ((1 - skew * sharpe_obs + (kurtosis-1)/4 * sharpe_obs^2) / max(1, n_bars-1))^0.5` — with default `skew=0`, `kurtosis=3`, this matches standard formula. |
| L210 | `if se_sr < 1e-12: return 1.0 if sharpe_obs > e_max_sr else 0.0` — when SE is zero, returns 0 or 1. Correct. |

### 6.5 stitched_oos_metrics

| Location | Issue |
|----------|-------|
| L234-261 | Multiplicative compounding: `eq = seg_start * (1 + r/100)`. Clamps `eq` to 0.001 minimum. `seg_valley = seg_start * max(0, 1 - d/100)` — drawdown applied to segment start. |
| L249-251 | `dd_intra` uses `(pk - seg_valley)/pk` — peak is global, valley is segment. For first segment, `pk=1`, `seg_start=1`. Logic is consistent. |

---

## 7. optimizer.py Deep Review

### 7.1 Bayesian Update Correctness

| Location | Issue |
|----------|-------|
| L57-69 | `age_weight = decay ** i` — `i` is index in `param_history`; history is ordered DESC (newest first). So `i=0` is newest, `i=1` is older. Decay gives *less* weight to older. Correct. |
| L61 | `sharpe = max(entry.get("sharpe", 0), 0)` — negative Sharpe gets 0 weight. Intentional. |
| L67 | `new_w = 1.0 + new_scan_sharpe * sharpe_weight_scale` — new params always included with weight 1 + sharpe. |
| L77-81 | Integer rounding: `isinstance(orig, int) or (isinstance(orig, float) and orig == int(orig))` — floats that are whole numbers get rounded to int. May be wrong for params like `0.5`. |

### 7.2 Composite Gate Score Formula

| Location | Issue |
|----------|-------|
| L98-126 | `_score_statistical`: 0.6 * sharpe_score + 0.4 * dsr_score. Sharpe capped at 1.0 for sharpe=3. |
| L106-109 | `_score_walkforward`: `wf_norm = min(max(wf_score/100, 0), 1.0)` — `wf_score` can be negative; `max(..., 0)` handles it. |
| L114 | `_score_montecarlo`: `min(mc_pct_positive/100, 1.0)` — `mc_pct_positive` is 0–100; divide by 100. Correct. |
| L118 | `_score_deflated`: `max(0, 1.0 - 2*dsr_p)` — at dsr_p=0.5, score=0; at dsr_p=0, score=1. |
| L123-125 | `_score_tail_risk`: `cvar_95` default 0; when `cvar_95 is None`, `cvar_score=0.5`. |
| L212 | `total = sum(GATE_WEIGHTS[k] * scores[k] for k in GATE_WEIGHTS)` — weights sum to 1.0. |

### 7.3 Champion/Challenger Protocol

| Location | Issue |
|----------|-------|
| L265-284 | `evaluate_promotion` — gets all PAPER strategies, filters by `kernel_name` and `symbol`. Returns first challenger with `ch_gate >= current_gate + 0.05`. |
| L271 | `challengers = db.get_strategies_by_status("PAPER")` — returns *all* PAPER strategies, not just for this symbol/strategy. Filtering is done in loop. Could be inefficient. |
| L295-296 | `promote_challenger` — updates status to LIVE, saves config version. Does not demote current champion explicitly. |

---

## 8. database.py Deep Review

### 8.1 Schema Integrity

| Table | Issue |
|-------|-------|
| `strategy_health` | No UNIQUE constraint on (symbol, strategy, ts). Multiple inserts per day possible. |
| `param_history` | `params` stored as JSON TEXT. No schema validation. |
| `strategy_library` | `UNIQUE(name, kernel_name)` — upsert uses this. `symbols` and `best_params` stored as JSON. |
| `config_versions` | No retention policy; table can grow unbounded. |

### 8.2 Migration Handling

- **No migration system** — `_create_tables` uses `CREATE TABLE IF NOT EXISTS`. Schema changes require manual ALTER or new DB.
- Adding columns to existing tables would need explicit migration code.

### 8.3 Query Correctness

| Query | Issue |
|-------|-------|
| `get_all_latest_health` L212-222 | Uses `MAX(ts)` per (symbol, strategy). If two rows have same `ts`, both could match. Uses `INNER JOIN` on `h.ts = latest.max_ts` — only one row per group. Correct. |
| `get_health_trend` L196-203 | `ORDER BY ts DESC LIMIT ?` — `days` is limit count, not calendar days. 30 rows may span more or less than 30 days. |
| `get_param_history` L256-265 | `ORDER BY ts DESC` — newest first. `json.loads(r["params"])` — invalid JSON would raise. |

### 8.4 Connection Management

| Location | Issue |
|----------|-------|
| L44-49 | `_conn()` uses `threading.local()` — one connection per thread. No connection pooling. |
| L51-60 | `_cursor()` commits on success, rollback on exception. |
| L36-37 | `:memory:` skips WAL. File DB uses WAL. Correct. |

### 8.5 upsert_strategy ON CONFLICT

| Location | Issue |
|----------|-------|
| L366-378 | `ON CONFLICT(name, kernel_name) DO UPDATE SET` — updates status, updated, and COALESCE for optional fields. `COALESCE(excluded.symbols, symbols)` — if `excluded.symbols` is NULL (INSERT had NULL), keeps old value. But INSERT passes `syms_json` which can be None. So `excluded.symbols` = NULL. Correct to keep old. |

---

## 9. Summary of Recommendations

### High Priority

1. **Remove `or True` from assertions** in test_v52_fixes.py, test_v4_research.py.
2. **Add unit tests for robust_scan.py**: `deflated_sharpe`, `robust_score`, `stitched_oos_metrics`, `perturb_ohlc`, `shuffle_ohlc`, `block_bootstrap_ohlc`.
3. **Add integration test for run_robust_scan and run_cpcv_scan** with small synthetic data.
4. **Fix E.7 test** — actually verify `_capital_fraction` or remove the test.
5. **Fix B.7 test** — verify bar data fallback behavior with a real scenario.

### Medium Priority

6. Add tests for `param_neighborhood_stability` with synthetic data.
7. Add tests for `run_optimizer` with in-memory DB.
8. Relax or parameterize performance thresholds (throughput, latency) for CI.
9. Add tests for OHLC edge cases (NaN, zero volume, invalid OHLC).
10. Fix `conftest` date handling (DatetimeIndex vs string) for consistency.

### Low Priority

11. Add schema migration tests for database.py.
12. Unify LatencyTracker import paths (live vs live.latency).
13. Add tests for KillSwitch `flatten_all` and `is_triggered` with mocked broker.

---

## 10. File-Level Issue Index

| File | Critical | Medium | Low |
|------|----------|--------|-----|
| test_backtest.py | 0 | 2 | 1 |
| test_broker.py | 0 | 0 | 0 |
| test_live.py | 0 | 1 | 0 |
| test_v5_full.py | 1 | 3 | 0 |
| test_v52_fixes.py | 3 | 1 | 0 |
| test_full_backtest_live.py | 0 | 2 | 0 |
| test_live_trading_async.py | 0 | 1 | 0 |
| test_v4_research.py | 4 | 1 | 0 |
| test_v4_production.py | 0 | 2 | 0 |
| test_indicators.py | 0 | 0 | 1 |
| test_strategies.py | 0 | 0 | 0 |
| test_performance.py | 0 | 0 | 0 |
| conftest.py | 0 | 1 | 1 |
| robust_scan.py | 2 | 3 | 0 |
| optimizer.py | 0 | 1 | 0 |
| database.py | 0 | 2 | 2 |
