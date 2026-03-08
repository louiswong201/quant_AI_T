# Comprehensive Test Review Report — quant_project_AI

**Date:** 2025-03-07  
**Scope:** 13 test files + 3 source files (robust_scan.py, optimizer.py, database.py)

---

## Executive Summary

This report identifies **test correctness issues**, **coverage gaps**, **mock problems**, **test data concerns**, **flaky tests**, and **deep reviews** of robust_scan.py, optimizer.py, and database.py. All issues include exact file names, line numbers, and detailed descriptions.

---

## 1. Test Correctness — Assertions Testing Wrong Things / Always Pass

| File | Line(s) | Issue |
|------|---------|-------|
| **test_v52_fixes.py** | 152-154 | **E.7-base-capital-fraction-attr**: `check("E.7-base-capital-fraction-attr", hasattr(BaseStrategy, '_capital_fraction') or True, ...)` — The `or True` makes this test **always pass** regardless of the attribute. |
| **test_v4_research.py** | 111-113 | **DB-008 Trend DESC order**: `check("DB-008 Trend DESC order", trend[0]["sharpe_30d"] <= trend[-1]["sharpe_30d"] or True, ...)` — `or True` makes the assertion vacuous. |
| **test_v4_research.py** | 382-384 | **PF-011 Correlated pair penalised**: `check("PF-011 Correlated pair penalised", xy_total < w_corr.get("Z", 0) + 0.3 or True, ...)` — Always passes. |
| **test_v4_research.py** | 416-418 | **DSC-010 Vol shift detected**: `check("DSC-010 Vol shift detected", len(anom_shift) > 0 or True, ...)` — Always passes. |
| **test_v52_fixes.py** | 256-258 | **B.7-backtest-engine-fallback**: `check("B.7-backtest-engine-fallback", True, ...)` — **Always passes**; no actual verification of the fallback logic. |
| **test_v4_research.py** | 516-518 | **MATH-001 Constant returns**: Comment says "std=0 guard" but the assertion `abs(s) < 1e-6 or s == 0` is loose; constant returns can produce `inf` or `nan` depending on implementation. |
| **test_performance.py** | 33-35 | **max_drawdown**: Asserts `dd_val < 0` but `max_drawdown` typically returns **positive** DD (e.g. 0.136 for 13.6%). The formula `(95 - 110) / 110` is negative; the test may be asserting the wrong sign convention. |
| **test_full_backtest_live.py** | 382-387 | CircuitBreaker assertions use `assert cb2.check() is None` / `is not None` but do not verify the **message** or **reason** for trip. |
| **test_live_trading_async.py** | 339-344 | Pipeline verification table uses `any(r['total_trades'] > 0 for r in ok_results)` — if all strategies produce 0 trades (valid for some data), KernelAdapter and PaperBroker are marked FAIL even when correct. |

---

## 2. Test Coverage Gaps — Critical Code Paths With NO Tests

| Module | Function/Path | Description |
|--------|---------------|-------------|
| **robust_scan.py** | `run_robust_scan` | No unit or integration tests. Only indirect coverage via `test_kernel_regression.py` for CPCV. |
| **robust_scan.py** | `run_cpcv_scan` | No direct tests. |
| **robust_scan.py** | `perturb_ohlc`, `shuffle_ohlc`, `block_bootstrap_ohlc` | No tests for OHLC validity (high ≥ low, open/close within range), deterministic seeds, or edge cases (n=0, n=1). |
| **robust_scan.py** | `deflated_sharpe` | No unit tests for edge cases: `n_bars < 3`, `n_trials < 1`, `se_sr ≈ 0`, or known formula validation. |
| **robust_scan.py** | `robust_score` | No tests for `trade_factor`, `dsr_bonus`, or score formula. |
| **robust_scan.py** | `stitched_oos_metrics` | No tests for multiplicative compounding, drawdown aggregation, or edge cases. |
| **optimizer.py** | `composite_gate_score` with `neighborhood_score=None` | Default path (0.5) tested but not explicitly for `None` vs missing key. |
| **optimizer.py** | `composite_gate_score` with missing metrics | No tests when `dsr_p`, `wf_score`, `mc_pct_positive`, etc. are absent. |
| **database.py** | Schema migrations | No tests for schema changes or migration from older DB versions. |
| **database.py** | Connection failure / timeout | No tests for `busy_timeout`, connection exhaustion, or WAL mode behavior. |
| **database.py** | `get_all_latest_health` with ties | When multiple rows share `MAX(ts)`, INNER JOIN can return multiple rows per (symbol, strategy). No test for dedup. |
| **test_backtest.py** | `run_robust_backtest` | Not tested; only `BacktestEngine.run` is. |
| **test_broker.py** | `Broker` base class | Only `PaperBroker` tested; no tests for abstract `Broker` interface. |
| **test_live.py** | `LatencyTracker` | Only `RiskGate` and `RiskManagedBroker` tested; `LatencyTracker` is used but not directly asserted. |
| **test_indicators.py** | Numba vs NumPy fallback | No tests when Numba is unavailable; fallback paths untested. |
| **test_strategies.py** | `BollingerStrategy`, `DonchianStrategy`, `TurtleStrategy` | Not covered in test_strategies.py. |
| **test_performance.py** | `PerformanceAnalyzer` | Only 4 tests; no tests for Sharpe, Sortino, or other metrics. |

---

## 3. Mock Issues — Over-Mocking / Hiding Real Bugs

| File | Line(s) | Issue |
|------|---------|-------|
| **test_backtest.py** | 103-115 | `_MockDataManager` returns `self._df.copy()` for any symbol/date range — **never validates** that requested dates match. Real `DataManager` may filter by date; mock hides date-boundary bugs. |
| **test_backtest.py** | 393-404 | `MultiDM` overrides `load_data` to return slightly different `close` per symbol but **does not test** that `BacktestEngine` correctly handles multi-symbol alignment (e.g. different lengths). |
| **test_backtest.py** | 407-416 | `auto_export_execution_report` test uses `live_fills` but does not verify that the report **content** matches expected structure; only checks file existence. |
| **test_full_backtest_live.py** | 57-103 | `download_data()` falls back to synthetic data on failure; tests can pass with **no real network** and never exercise yfinance code path. |
| **test_live_trading_async.py** | 33-121 | `SyntheticFeedManager` replays bars but **does not** simulate tick events, reconnects, or WebSocket failures. Real `BinanceFeed` behavior is untested. |
| **test_v4_production.py** | 90-96 | `load_data()` depends on `run_production_scan.load_daily` and `load_intraday` — if those fail or return empty, many tests skip or warn without failing. |
| **test_v4_production.py** | 302-306 | `composite_gate_score(bm)` called with `backtest_metrics` dict that may lack `dsr_p`, `wf_score`, `mc_pct_positive` — defaults used; no test that missing keys produce correct behavior. |

---

## 4. Test Data Issues — Unrealistic / Edge Cases

| File | Line(s) | Issue |
|------|---------|-------|
| **conftest.py** | 10-29 | `sample_ohlcv`: 100 bars, `np.random.seed(42)`. No tests for: **zero volume**, **NaN in OHLC**, **high == low**, **negative prices**, **gaps in dates**, or **single-row** DataFrames. |
| **conftest.py** | 10-29 | `sample_ohlcv` uses `pd.bdate_range` — no weekends. Crypto/24-7 assets need different calendar; not covered. |
| **test_backtest.py** | 381-384 | `test_limit_order_price_time_priority`: Sets `volume=1.0` on first bar to force partial fill. **Brittle** — depends on exact participation logic. |
| **test_v5_full.py** | 38-49 | `make_ohlcv` uses `np.random.default_rng(seed)` — good. But no tests for **crash scenarios** (e.g. -50% drawdown), **flat markets**, or **extreme volatility**. |
| **test_v52_fixes.py** | 89-92 | ZScore test: `close_zs[-1] = 130.0` for "extreme z" — depends on lookback=20 and prior values. **Fragile** if RNG or formula changes. |
| **test_indicators.py** | 43-47 | `test_short_data`: RSI with 3 elements and period=14 — all NaN. No test for **exactly** period+1 elements (first valid RSI). |
| **test_performance.py** | 31-35 | `max_drawdown` test uses 6-element array; no test for **2 elements**, **all equal**, or **monotonically increasing**. |

---

## 5. Flaky Tests — Timing, Network, Random State

| File | Line(s) | Issue |
|------|---------|-------|
| **test_full_backtest_live.py** | 67-92 | `download_data()` uses `yf.download` with retries and `time.sleep` — **network-dependent**; rate limits can cause fallback to synthetic, making results non-deterministic. |
| **test_full_backtest_live.py** | 239-448 | Full pipeline runs on downloaded/synthetic data; **runtime varies** with network and CPU. |
| **test_live_trading_async.py** | 149-209 | `run_live_test` is async; **timing** of bar replay can affect signal generation. No fixed seed for strategy decisions in some paths. |
| **test_v4_production.py** | 169-170 | `elapsed_ms < 500` for health metrics — **machine-dependent**; can fail on slow CI. |
| **test_v4_production.py** | 255-256 | `elapsed_ms < 10` for regime detection — **flaky** on loaded systems. |
| **test_v5_full.py** | 443-446 | `PERF-{sz}bars` requires `throughput > 50` — **hardware-dependent**; may fail on CI or low-end machines. |
| **test_v52_fixes.py** | 352-354 | Same throughput assertion. |
| **test_v4_research.py** | 214-224 | `np.random.seed(42)` used but **order of tests** can affect global state if other tests use `np.random` without reset. |
| **test_v4_research.py** | 286-295 | `regime_probabilities` with `np.random.seed(123)` — seed not reset after; can leak to later tests. |

---

## 6. robust_scan.py Deep Review

### 6.1 CPCV Implementation

| Location | Issue |
|----------|-------|
| **L684-728** (`cpcv_splits`) | **Purge logic bug**: In the loop `for tr_s, tr_e in train_ranges:` and `for te_s, te_e in test_ranges:`, the assignments `tr_e = max(tr_s, te_s - embargo)` and `tr_s = min(tr_e, te_e + embargo)` modify **loop variables** `tr_s`, `tr_e`. These modifications **do not persist** to the next iteration of the outer loop over `train_ranges`, because each iteration gets fresh `tr_s`, `tr_e` from the tuple. The purge is applied per train range but the **updated** `tr_s`/`tr_e` are only used for the `if tr_s < tr_e` check — the logic may be correct for that single range, but the interaction between multiple test ranges purging the same train range is subtle and **not obviously correct**. |
| **L519** | `group_size = n_bars // n_groups` — remainder bars are **dropped**. For `n_bars=100`, `n_groups=6`, `group_size=16`, last group gets indices 80–99 (20 bars). First 5 groups get 16 each. Total = 16*5 + 20 = 100. Actually the loop uses `end = (i+1)*group_size if i < n_groups-1 else n_bars`, so the last group gets the remainder. **4 bars lost** only when `n_bars=100` and `n_groups=6` → 16*6=96, last group `end=n_bars=100` → 4 bars in last group. So 96 bars in first 5 groups, 4 in last. **Inconsistent group sizes.** |
| **L916** (`run_cpcv_scan` / `robust_score`) | `robust_score(avg_ret, avg_dd, total_nt, n, ...)` uses full `n` (all bars) for DSR. CPCV OOS bars **vary per split**; DSR should use **total OOS bars** across splits for consistency with deflated Sharpe theory. |
| **L676** | `cpcv_score = shrp * pct_pos * (1.0 + max(0.0, dsr_p - 0.5))` — **Ad hoc formula**; no reference to de Prado or standard CPCV scoring. |

### 6.2 Walk-Forward Window Calculation

| Location | Issue |
|----------|-------|
| **L283-290** | `tr_end = int(n * tr_pct)`, `va_start = min(tr_end + embargo, int(n * va_pct))`, etc. No tests verify that **param search never sees validation/test data**. Embargo application at boundaries is complex. |
| **DEFAULT_WF_WINDOWS** | 6 windows with 10% train/val/test each. No tests for **minimum bars** per window (e.g. strategy needs 50 bars, window has 30). |
| **L364-369** | `c_tr, o_tr = c[:tr_end], o[:tr_end]` — train uses `[0, tr_end)`. If `va_start` overlaps due to rounding, validation could include train data. `va_start = min(tr_end + embargo, int(n * va_pct))` — when `int(n*va_pct) < tr_end + embargo`, validation starts before embargo. **Potential leak.** |

### 6.3 Monte Carlo Path Generation

| Location | Issue |
|----------|-------|
| **L98-118** (`perturb_ohlc`) | After perturbation, `l_p[i] = body_lo - lw * (1.0 + noise)` can become **> body_hi** if `lw` is large and noise is negative. No assertion that `high >= low` or `open/close` within range. |
| **L121-153** (`shuffle_ohlc`) | Shuffles O,H,L,C roles. Output assigns `h_p = mx`, `l_p = mn` — correct. But `o_p`, `c_p` from `rem` — if `rem` has only 1 element, `rem[ri] = vals[3]` can duplicate. **Edge case** for `ri < 2`. |
| **L156-192** (`block_bootstrap_ohlc`) | `max_start = n - block_size`; when `n < block_size`, `max_start < 0` → set to 0. Loop `for j in range(block_size)` can index `si >= n`; `si = n-1` used. **Scaling** `last_c/close[start-1]` can produce very large/small values if `close[start-1]` is tiny. |
| **Seeds** | `perturb_ohlc` uses `42000+idx`, `shuffle_ohlc` uses `50000+idx`, `block_bootstrap_ohlc` uses `60000+idx`. **No tests** that same seed produces deterministic output. |

### 6.4 DSR Calculation

| Location | Issue |
|----------|-------|
| **L198-212** (`deflated_sharpe`) | Returns `0.0` for `n_bars < 3` or `n_trials < 1`. For `se_sr < 1e-12`, returns `1.0 if sharpe_obs > e_max_sr else 0.0` — **no tests** for this branch. |
| **L204-206** | `se_sr` formula uses `(1 - skew*sharpe_obs + (kurtosis-1)/4 * sharpe_obs**2) / max(1, n_bars-1)` — **no validation** against known reference (e.g. de Prado). |

---

## 7. optimizer.py Deep Review

### 7.1 Bayesian Update Correctness

| Location | Issue |
|----------|-------|
| **L59-66** | `param_history` is iterated in order; `age_weight = decay ** i` — **index `i`** is 0-based. First entry (most recent) gets `decay^0=1`, oldest gets `decay^(n-1)`. If `get_param_history` returns **newest first** (ORDER BY ts DESC), then `i=0` is newest — correct. If oldest first, logic is **inverted**. |
| **L79-82** | Integer params: `orig == int(orig)` for floats that are whole numbers. `round(b)` then `int()` — correct. |

### 7.2 Composite Gate Score Formula

| Location | Issue |
|----------|-------|
| **L98-102** | `_score_statistical`: `sharpe_score = min(sharpe/3, 1)` for sharpe>0 else 0. `dsr_score = 1 - min(dsr_p, 1)`. Combined `0.6*sharpe_score + 0.4*dsr_score`. **Correct.** |
| **L106-109** | `_score_walkforward`: `wf_norm = min(max(wf_score/100, 0), 1)`. For `wf_score > 100`, caps at 1. **Correct.** |
| **L114** | `_score_montecarlo`: `min(mc_pct_positive/100, 1)`. **Correct.** |
| **L118** | `_score_deflated`: `max(0, 1 - 2*dsr_p)`. At dsr_p=0.5, score=0; at dsr_p=0, score=1. **Correct.** |
| **L196-212** | `composite_gate_score`: When `neighborhood_score is None`, uses `0.5`. When metrics lack `dsr_p`, uses `1` (default). **No test** for `metrics = {}` (all defaults). |

### 7.3 Champion/Challenger Protocol

| Location | Issue |
|----------|-------|
| **L266-272** | `evaluate_promotion` iterates **all** PAPER strategies and filters by `kernel_name` and `symbol`. If multiple challengers exist for same symbol/strategy, **first match** is returned. No test for **multiple challengers** (which wins?). |
| **L295-307** | `promote_challenger` updates status to LIVE but does **not** demote or retire the previous champion. **Orphaned LIVE** strategies possible. |
| **L241** | `register_challenger` uses `name=f"{symbol}_{strategy}_challenger"` — **collision** if multiple challengers for same symbol/strategy; `upsert_strategy` will overwrite. |

---

## 8. database.py Deep Review

### 8.1 Schema Integrity

| Location | Issue |
|----------|-------|
| **L76-91** | `strategy_health`: No UNIQUE constraint on (symbol, strategy, ts). **Duplicate** inserts for same symbol/strategy/ts allowed. |
| **L98-111** | `param_history`: `params` stored as JSON TEXT. No validation that it parses. |
| **L144-161** | `strategy_library`: `UNIQUE(name, kernel_name)`. `symbols`, `best_params`, `regime_affinity` stored as JSON. Invalid JSON on read can raise. |
| **L318** | `record_regime`: `dominant = max(vals, key=vals.get)` — if two regimes tie, **arbitrary** choice. |

### 8.2 Migration Handling

| Location | Issue |
|----------|-------|
| **N/A** | **No migration system**. Schema is `CREATE TABLE IF NOT EXISTS` — new columns or tables require manual migration. No version table, no upgrade path. |

### 8.3 Query Correctness

| Location | Issue |
|----------|-------|
| **L213-224** | `get_all_latest_health`: Subquery `MAX(ts)` per (symbol, strategy). If two rows have same `ts`, INNER JOIN can return **multiple rows** per pair. No `DISTINCT` or `LIMIT 1` per group. |
| **L258-260** | `get_param_history`: `ORDER BY ts DESC LIMIT ?`. SQLite `datetime('now')` has second precision; rapid inserts can share same `ts`. Order among ties is **undefined**. |

### 8.4 Connection Management

| Location | Issue |
|----------|-------|
| **L43-48** | `_conn()` uses `threading.local()` — **one connection per thread**. Long-lived threads hold connections. No connection pool or timeout for idle connections. |
| **L34-36** | `:memory:` skips WAL — correct. |
| **L37** | `busy_timeout=5000` — 5 second wait. No test for concurrent write conflicts. |

---

## Recommendations

1. **Remove `or True`** from test_v52_fixes.py (E.7), test_v4_research.py (DB-008, PF-011, DSC-010). Replace with real assertions.
2. **Replace `check(..., True, ...)`** in test_v52_fixes.py B.7 with actual verification of bar_data fallback.
3. **Fix max_drawdown test** in test_performance.py — verify sign convention and expected value.
4. **Add unit tests for robust_scan.py**: `perturb_ohlc` OHLC validity, `deflated_sharpe` edge cases, `robust_score` formula, `cpcv_splits` purge logic.
5. **Add unit tests for optimizer.py**: `composite_gate_score` with empty metrics, `evaluate_promotion` with multiple challengers.
6. **Add database tests**: `get_all_latest_health` with duplicate ts, connection failure, schema migration.
7. **Stabilize flaky tests**: Use `@pytest.mark.slow` for network/perf tests; increase timing tolerances or skip on CI.
8. **Expand conftest.py**: Add fixtures for edge-case data (NaN, zero volume, single row, flat market).
9. **Fix cpcv_splits purge logic** — ensure purge modifications correctly affect purged_train ranges.
10. **Document** DSR and CPCV score formulas with references (e.g. de Prado).

---

## Appendix: Test File Summary

| File | Tests | Key Gaps |
|------|-------|----------|
| test_backtest.py | 25+ | run_robust_backtest, multi-symbol alignment |
| test_broker.py | 10 | Broker base, async methods |
| test_live.py | 5 | LatencyTracker, full RiskGate matrix |
| test_v5_full.py | 50+ | Custom check() not pytest; perf flaky |
| test_v52_fixes.py | 40+ | E.7/B.7 always pass |
| test_full_backtest_live.py | Integration | Network-dependent |
| test_live_trading_async.py | 8 cases | No pytest; async timing |
| test_v4_research.py | 80+ | or True assertions |
| test_v4_production.py | 50+ | Requires live_trading_config.json |
| test_indicators.py | 15 | Numba fallback, edge cases |
| test_strategies.py | 35+ | Bollinger, Donchian, Turtle |
| test_performance.py | 4 | max_drawdown sign, more metrics |
| conftest.py | 2 fixtures | Edge-case data |
