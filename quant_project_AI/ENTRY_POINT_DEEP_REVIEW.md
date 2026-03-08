# Entry Point & Integration — Deep Code Review (New Issues Only)

**Scope:** Entry point scripts and integration points not deeply examined in prior review  
**Date:** 2025-03-07  
**Prerequisite:** See `ENTRY_POINT_CODE_REVIEW.md` for previously identified issues

This report documents **only** issues not already covered in the prior review.

---

## 1. Data Flow & Integration

### 1.1 Multi-TF Stocks: YFinanceFeed Returns Wrong Timeframe Data

| File | Line | Issue |
|------|------|-------|
| `price_feed.py` | 531-539 | When `add_symbol_multi_tf` is called for a **stock** (non-crypto), the code creates a single `YFinanceFeed` with only the finest interval. `YFinanceFeed` has one `_RollingWindow` and `get_window()` returns that single window regardless of requested interval. |
| `trading_runner.py` | 149-158 | `MultiTFAdapter` calls `get_window(symbol, "1h")`, `get_window(symbol, "4h")`, `get_window(symbol, "1d")` for multi-TF strategies. For stocks, all three return the same 1h (or finest) data. The 4h and 1d signals are computed on 1h bars, causing incorrect strategy behavior. |

**Impact:** Multi-TF stock strategies receive wrong timeframe data; signals and risk metrics are invalid.

**Recommendation:** Either (a) resample 1h→4h→1d inside `YFinanceFeed` when multi-TF is requested, (b) use separate feeds per interval for stocks, or (c) document that multi-TF is crypto-only and reject stock multi-TF configs.

---

### 1.2 run_production_scan: data_dir Not Created Before Load

| File | Line | Issue |
|------|------|-------|
| `run_production_scan.py` | 371-372, 386 | `data_dir = os.path.join(base_dir, "data")` is set but `os.makedirs(data_dir, exist_ok=True)` is never called. `load_daily(data_dir)` at line 386 does `os.listdir(data_dir)`, which raises `FileNotFoundError` if `data/` does not exist. |

**Impact:** First-time run without pre-existing `data/` directory crashes.

**Recommendation:** Add `os.makedirs(data_dir, exist_ok=True)` before Phase 0.

---

### 1.3 Phase 2/3/4 Without Phase 1: No Validation

| File | Line | Issue |
|------|------|-------|
| `run_production_scan.py` | 418-424, 441-455 | Running `--phase 2` or `--phase 3` skips Phase 1. `phase1_ranking` is set to `[]`. Phase 2's `top_by_sym_tf` is empty, so no multi-TF results. Phase 3 filters empty lists. Phase 4 exports empty or stale recommendations. No warning is printed. |

**Impact:** User may believe a partial run produced valid output when it did not.

**Recommendation:** Validate phase dependencies (e.g. Phase 2 requires Phase 1) and exit with a clear error if prerequisites are missing.

---

## 2. File I/O & Edge Cases

### 2.1 daily_research: Empty CSV Causes dates.max() Failure

| File | Line | Issue |
|------|------|-------|
| `daily_research.py` | 156-159 | `_incremental_download` reads existing CSV. If the file exists but is empty (0 rows), `existing = pd.read_csv(csv_path)` returns an empty DataFrame. `dates = pd.to_datetime(existing["date"], utc=True)` yields an empty Series. `dates.max()` on an empty datetime Series returns `NaT`. `(datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days` then fails (NaT cannot be used in subtraction). |

**Impact:** Corrupted or empty CSV causes `phase0_refresh` to crash.

**Recommendation:** Check `if existing.empty` before accessing `dates.max()` and treat as “no existing data” (full download).

---

### 2.2 download_data: Summary With No Successful Downloads

| File | Line | Issue |
|------|------|-------|
| `download_data.py` | 216-218 | Summary uses `os.listdir(os.path.join(data_dir, "1h"))` only when `os.path.isdir(...)` is True. If all downloads fail and no symbol writes to `1h/`, the directory may never be created. The `isdir` check prevents a crash, but the summary can be misleading (e.g. “0 daily, 0 4h, 0 1h” when user expected data). |

**Note:** No crash; informational only. Consider creating empty `1h`/`4h` dirs at start for consistent reporting.

---

## 3. Config & Data Structure Assumptions

### 3.1 phase3_compare: Missing "sharpe" Key

| File | Line | Issue |
|------|------|-------|
| `daily_research.py` | 368 | `new_sharpe = new["sharpe"] if new else 0` — if `new` exists but lacks a `"sharpe"` key (e.g. malformed rescan output), this raises `KeyError`. |

**Recommendation:** Use `new.get("sharpe", 0) if new else 0`.

---

### 3.2 apply_updates: rec["params"] May Be None

| File | Line | Issue |
|------|------|-------|
| `daily_research.py` | 339, 354 | `rec["params"] = list(upg["new_params"])` and `old = f"{rec['strategy']}({rec['params']})"`. If `rec["params"]` is `None` (e.g. from a malformed config), `list(None)` raises `TypeError` and `str(None)` in f-string is less informative. |

**Recommendation:** Use `rec.get("params") or []` and validate before `list()`.

---

## 4. Concurrency & Shutdown

### 4.1 Kill Switch Timeout Leaves Partial State

| File | Line | Issue |
|------|------|-------|
| `trading_runner.py` | 416-418 | `activate_kill_switch` uses `fut.result(timeout=15)`. If flattening many positions takes longer, `TimeoutError` is raised. The kill switch may have closed some positions but not all. The dashboard shows an error; the runner’s `_entry_prices`, `_sl_triggered`, and broker state can be inconsistent. |

**Impact:** Partial flatten on timeout; manual reconciliation may be required.

**Recommendation:** Increase timeout for large books, or make it configurable; consider idempotent retry or a “resume flatten” path.

---

### 4.2 Kill Switch During Shutdown Race

| File | Line | Issue |
|------|------|-------|
| `trading_runner.py` | 414-418 | On `KeyboardInterrupt`, the main loop exits. The dashboard (daemon thread) may still be active. If the user clicks the kill switch during shutdown, `self._loop.is_running()` can become `False` mid-call. The code falls back to `asyncio.run(self._activate_kill_switch(reason))`, which creates a new event loop in the dashboard thread. The broker/feed may already be closing. Race between shutdown and kill switch execution. |

**Recommendation:** Guard kill switch when `_running` is False; return a clear “shutdown in progress” response.

---

### 4.3 Dashboard Daemon Thread vs journal.close()

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 406-407 | `finally: journal.close()` runs on main exit. The dashboard runs in a daemon thread and may be in the middle of `get_trades()` or `get_daily_pnl()` when `journal.close()` is called. SQLite connections closed during an active query can raise or leave the connection in a bad state. |

**Impact:** Possible `sqlite3.ProgrammingError` or similar when shutting down during dashboard refresh.

**Recommendation:** Stop the dashboard (or its refresh) before closing the journal; use a shutdown flag or join the dashboard thread with a short timeout.

---

## 5. Price Feed & Data Paths

### 5.1 PriceFeedManager _DEFAULT_DATA_DIR Assumption

| File | Line | Issue |
|------|------|-------|
| `price_feed.py` | 24 | `_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"` assumes `price_feed.py` lives at `quant_project_AI/quant_framework/live/price_feed.py`. If the package is installed or used from another layout, `parent.parent.parent` may not point to the project root. |

**Recommendation:** Prefer an explicit configurable `data_dir` (e.g. from config or env) over a hardcoded relative path.

---

### 5.2 YFinanceFeed: Stocks With 1h Interval

| File | Line | Issue |
|------|------|-------|
| `price_feed.py` | 182-183 | `period = "5d" if self._interval == "1m" else "60d"`. For stocks, yfinance free tier often does not provide 1h data; it may fall back to 1d. The feed reports `interval="1h"` but the actual bars may be daily, causing a mismatch between expected and actual bar frequency. |

**Recommendation:** Document yfinance limitations for stock intraday; consider validation or fallback to 1d when 1h is unavailable.

---

## 6. quant_framework/data Module Disconnect

### 6.1 Entry Points Do Not Use quant_framework.data

| File | Issue |
|------|-------|
| `quant_framework/data/__init__.py` | Exports `DataManager`, `Dataset`, `CacheManager`, etc. |
| `download_data.py`, `run_production_scan.py` | Use custom CSV loading (`load_daily`, `load_intraday`, `process_df`). |
| `quant_framework/backtest` | Uses `DataManager` for some backtest paths. |

**Impact:** Two data-loading paths with potentially different schemas, encodings, and directory layouts. Using both for the same assets can cause format mismatches.

**Recommendation:** Either unify on `DataManager`/dataset for all entry points or clearly separate “research/backtest” vs “live/production” data and document the split.

---

## 7. Portfolio Weights Integration (Extension of Prior Finding)

### 7.1 Key Format Mismatch

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 275-277 | `raw_weights = pa.get("position_sizes", {})` — keys are `"SYMBOL/STRATEGY"` (e.g. `"BTC/MA"`). The code maps `sym_part = key.split("/")[0]`, so `portfolio_weights["BTC"] = pct`. If multiple strategies exist for the same symbol (e.g. `"BTC/MA"` and `"BTC/RSI"`), the last one overwrites. |
| `run_live_trading.py` | 364-368 | Only `first_config_sym`’s weight is used for the global `position_size_pct`. Per-symbol weights are never passed to `TradingRunner`. |

**Impact:** Multi-strategy-per-symbol portfolios get a single weight; per-symbol sizing is not applied.

---

## 8. Summary of New Issues

| # | Severity | File(s) | Description |
|---|----------|---------|-------------|
| 1 | **Critical** | price_feed.py, trading_runner.py | Multi-TF stocks receive wrong timeframe data |
| 2 | High | run_production_scan.py | data_dir not created before load |
| 3 | High | run_production_scan.py | Phase 2/3/4 without Phase 1 produces empty output, no warning |
| 4 | High | daily_research.py | Empty CSV causes dates.max() failure |
| 5 | Medium | trading_runner.py | Kill switch timeout leaves partial state |
| 6 | Medium | trading_runner.py | Kill switch during shutdown race |
| 7 | Medium | run_live_trading.py | Dashboard/journal close race on exit |
| 8 | Medium | daily_research.py | phase3_compare KeyError on missing "sharpe" |
| 9 | Medium | daily_research.py | apply_updates TypeError when params is None |
| 10 | Low | price_feed.py | _DEFAULT_DATA_DIR assumes fixed project layout |
| 11 | Low | price_feed.py | YFinance stock 1h may fall back to 1d silently |
| 12 | Low | quant_framework/data | Disconnect between DataManager and entry-point CSV loading |

---

*End of deep review. See ENTRY_POINT_CODE_REVIEW.md for the full prior issue set.*
