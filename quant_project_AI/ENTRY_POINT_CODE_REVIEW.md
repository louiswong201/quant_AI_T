# Entry Point & Integration Code Review Report

**Scope:** `run_live_trading.py`, `run_production_scan.py`, `daily_research.py`, `download_data.py`, `trading_runner.py`, `app.py` (dashboard), `quant_framework/data/__init__.py`, `price_feed.py`  
**Date:** 2025-03-07  
**Focus:** Data flow, CLI, file I/O, network, memory, signals, logging, dashboard integration, config validation, recovery, timezone, encoding

---

## 1. Data Flow Issues

### 1.1 Symbol/Interval Mismatch Between Download and Live Trading

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 288-291 | `PriceFeedManager` is created with `interval="1d"` hardcoded. Config recommendations can specify `1h`, `4h`, or `1d`. Multi-TF strategies add multiple intervals via `add_symbol_multi_tf`, but single-TF strategies with `interval="1h"` still get a feed that polls at 1d. |
| `price_feed.py` | 181-182 | `YFinanceFeed` uses `period="5d"` for 1m, `"60d"` for others. Live trading uses `1d` interval; the feed may not align with strategy interval. |

**Impact:** Single-TF strategies with 1h or 4h intervals may receive daily bars only, causing signal misalignment.

### 1.2 Data Format Inconsistency: Daily vs Intraday

| File | Line | Issue |
|------|------|-------|
| `run_production_scan.py` | 64-83, 86-110 | `load_daily` returns `{sym: {c, o, h, l}}` (no timestamps). `load_intraday` returns `{sym: {c, o, h, l, timestamps}}`. Daily data has no timezone; intraday uses `parse_dates=["date"]` and `astype("datetime64[s]")`. |
| `daily_research.py` | 139-146 | `resample_to_4h` uses `pd.to_datetime(df["date"], utc=True)`; `download_data.py` line 134 uses `pd.to_datetime(df["date"])` without `utc=True`. Inconsistent timezone handling across scripts. |

### 1.3 run_full_scan Import Dependency

| File | Line | Issue |
|------|------|-------|
| `run_production_scan.py` | 39 | `from run_full_scan import EXPANDED_GRIDS` — hard dependency. If `run_full_scan.py` is missing or renamed, production scan fails at import. |
| `daily_research.py` | 454-461, 531-536 | `try: from run_full_scan import EXPANDED_GRIDS` with fallback to DEFAULT. But `run_production_scan` has no fallback; it will crash. |

---

## 2. CLI Argument Handling

### 2.1 Missing Validation

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 205-206 | `--position-size` accepts any float; values like `0` or `1.5` (150%) are not validated. |
| `run_live_trading.py` | 209-210 | `--lookback` accepts any int; negative or very large values (e.g. 100000) could cause memory issues. |
| `run_live_trading.py` | 226-228 | `--max-daily-loss` accepts any float; `0` or `1.0` (100%) are not rejected. |
| `download_data.py` | 186-187 | `do_crypto` and `do_stock` both True when neither flag is set (`not args.crypto and not args.stock` → both download). Intentional but confusing. |
| `daily_research.py` | 377-378 | `--recent-days` accepts any int; negative or zero could cause errors in monitor/portfolio. |

### 2.2 Conflicting Options

| File | Line | Issue |
|------|------|-------|
| `daily_research.py` | 368-372 | `--quick` and `--deep` override `--mode`. If user passes `--mode weekly --quick`, mode becomes `daily`. No warning. |
| `run_production_scan.py` | 366-367 | `--phase 0` runs all phases; `--phase 1` skips Phase 0 (data loading). Phase 0 is always run first in main(); `phase1_ranking` is empty if `args.phase == 1` and Phase 0 wasn't run — but Phase 0 is always run. Actually Phase 0 runs before the phase check. Re-check: Phase 0 runs unconditionally. Phase 1/2/3/4 are gated. OK. |

### 2.3 Wrong Defaults

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 223 | `--db-path` default `"live_trading.db"` is relative. If run from a different cwd, DB is created in that directory. Should use `Path(__file__).parent / "live_trading.db"` or document cwd requirement. |
| `run_live_trading.py` | 234 | `--research-db` default `"research.db"` — same relative path issue. |

---

## 3. File I/O Issues

### 3.1 Hardcoded Paths

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 183-184, 246 | `--config` default `"reports/live_trading_config.json"` is relative to cwd, not script dir. |
| `daily_research.py` | 396-397 | `--config` default `"reports/live_trading_config.json"` — same. |
| `run_production_scan.py` | 371-372 | `data_dir = os.path.join(base_dir, "data")` — correct. But `reports_dir` is used; config path in run_live_trading is not joined with base_dir when loading. |
| `run_live_trading.py` | 246 | `config_path = Path(args.config)` — used as-is. If `args.config` is `"reports/live_trading_config.json"`, it's relative to cwd. |

### 3.2 Missing Directory Creation

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 286 | `data_dir = Path(__file__).resolve().parent / "data"` — directory is not created. `PriceFeedManager` may write to it; if it doesn't exist, could fail. |
| `download_data.py` | 193 | `os.makedirs(data_dir, exist_ok=True)` — correct. Subdirs `1h`, `4h` created in `download_symbol`. |

### 3.3 File Locking / Concurrent Writes

| File | Line | Issue |
|------|------|-------|
| `daily_research.py` | 358-361 | `apply_updates` writes `versioned` and `output_path` sequentially. If another process (e.g. run_production_scan) writes to the same file concurrently, corruption possible. No file locking. |
| `run_production_scan.py` | 447-451 | Writes `live_trading_config_{ts}.json` and `live_trading_config.json`. Same risk if daily_research runs `--apply` simultaneously. |

### 3.4 CSV Encoding

| File | Line | Issue |
|------|------|-------|
| `daily_research.py` | 196 | `combined.to_csv(csv_path, index=False)` — no explicit `encoding`. On Windows with non-ASCII symbol names (e.g. future symbols), could fail. |
| `download_data.py` | 152, 163, 173 | Same — no `encoding="utf-8"` in `to_csv`. |
| `run_production_scan.py` | 70, 95 | `pd.read_csv(...)` — no encoding. Usually fine for ASCII symbols. |

---

## 4. Network Error Handling

### 4.1 yfinance Failures

| File | Line | Issue |
|------|------|-------|
| `daily_research.py` | 166-176 | `_incremental_download`: 3 attempts, 0.5s sleep. On final failure, returns 0 silently. No logging of the exception. |
| `download_data.py` | 77-87 | `download_yf`: 3 attempts, 1s sleep. Prints `WARN` on final failure. Returns empty DataFrame. |
| `price_feed.py` | 183-206 | `YFinanceFeed.start`: No retry. Single failure leaves window empty. `logger.warning` only. |
| `price_feed.py` | 234-244 | `YFinanceFeed.bars`: On empty df, `continue` silently. No backoff on repeated failures. |

### 4.2 No Timeouts

| File | Line | Issue |
|------|------|-------|
| `daily_research.py` | 168-170 | `t.history(period=period, interval=interval)` — no timeout. yfinance can hang on network issues. |
| `download_data.py` | 80-81 | Same. |
| `price_feed.py` | 235-236 | `ticker.history(period="1d", interval=self._interval)` — no timeout. |

### 4.3 Binance WebSocket (price_feed.py)

| File | Line | Issue |
|------|------|-------|
| `price_feed.py` | (BinanceFeed) | WebSocket reconnection logic should be verified. If connection drops, does it reconnect? Need to read full implementation. |

---

## 5. Memory Management

### 5.1 Loading All Data Into Memory

| File | Line | Issue |
|------|------|-------|
| `run_production_scan.py` | 386-389 | `load_daily`, `load_intraday` load all symbols into memory. With 30+ symbols × 3 timeframes × ~500 bars × 5 floats, manageable but not streaming. |
| `daily_research.py` | 234-236 | `_load_tf_data` loads 1d, 1h, 4h for all active symbols. Same pattern. |
| `trading_runner.py` | 541-548 | `_build_state` calls `get_equity_curve(limit=500)` and iterates. `get_strategy_trade_stats(limit=3000)` — 3000 rows loaded per refresh. With 2s dashboard refresh, acceptable. |

### 5.2 No Cleanup of Large Objects

| File | Line | Issue |
|------|------|-------|
| `daily_research.py` | 430-470 | `phase2_rescan` builds `rescan_ranking` with thousands of entries. `phase3_compare` and `phase1_monitor` all hold references. After `phase5`, `tf_data`, `rescan_ranking`, `monitor_results` are not explicitly cleared. Python GC will reclaim, but peak memory can be high for monthly runs. |

---

## 6. Signal Handling

### 6.1 Graceful Shutdown

| File | Line | Issue |
|------|------|-------|
| `trading_runner.py` | 89-96 | `add_signal_handler(SIGINT, SIGTERM)` on non-Windows. `atexit.register(self._shutdown)` on Windows. `_shutdown` sets `_running = False`. |
| `trading_runner.py` | 331-333 | `_shutdown` does not call `feed.stop_all()` or `journal.close()`. The event loop continues until `_feed.run()` exits. If the feed blocks, shutdown may be delayed. |
| `run_live_trading.py` | 403-407 | `except KeyboardInterrupt` logs and exits. `finally: journal.close()`. But `runner.run()` is async; `runner.stop()` is never called. The feed may keep running until the process exits. |

### 6.2 Windows Compatibility

| File | Line | Issue |
|------|------|-------|
| `trading_runner.py` | 89-96 | `loop.add_signal_handler` can raise `NotImplementedError` on Windows. Caught and ignored; falls back to `atexit`. But `atexit` runs on normal exit; Ctrl+C may not trigger it reliably on Windows. |

---

## 7. Logging

### 7.1 Missing Logging

| File | Line | Issue |
|------|------|-------|
| `daily_research.py` | 174-176 | `_incremental_download` catches `Exception` and returns 0. No `logger.debug` or `logger.warning` with the exception. |
| `run_production_scan.py` | 324-325 | `except Exception: pass` in `phase2_multi_tf` — silently skips failed multi-TF backtests. No logging. |
| `download_data.py` | 84-86 | Logs `WARN` on final failure but not the exception message in some code paths. |

### 7.2 Sensitive Data in Logs

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 322-326 | Logs `rec["rank"]`, `config_sym`, `lev`, `sl`, `strat_desc`. No API keys. OK. |
| `trading_runner.py` | 234-237 | Logs `sig["action"]`, `symbol`, `fill_price`, `filled_shares`, `pnl`. No credentials. OK. |

### 7.3 Excessive Logging

| File | Line | Issue |
|------|------|-------|
| `trading_runner.py` | 261-262 | `logger.debug` for rejected orders — OK. |
| `price_feed.py` | 204 | `logger.info` for each symbol's historical load — with 20 symbols, 20 lines. Consider `logger.debug` or single summary. |

---

## 8. Dashboard–Runner Integration

### 8.1 Data Staleness

| File | Line | Issue |
|------|------|-------|
| `app.py` | 462 | `dcc.Interval(id="interval", interval=refresh_ms, n_intervals=0)` — default 2000ms. |
| `trading_runner.py` | 424-431 | `get_state` uses `_STATE_CACHE_TTL = 1.5` seconds. Dashboard polls every 2s. Cache TTL is shorter than poll interval, so each poll gets fresh data. OK. |
| `trading_runner.py` | 421-422 | If `_build_state` raises, falls back to `_build_state_minimal()`. Stale cache is not returned on error; minimal state is. Good. |

### 8.2 get_window Signature Mismatch

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 391 | `get_window=lambda sym, iv=None: feed.get_window(sym, iv)` |
| `app.py` | 831-833 | `try: df = get_window(symbol, iv)` then `except TypeError: df = get_window(symbol)`. The dashboard handles both 1-arg and 2-arg. If `feed.get_window` only accepts 1 arg for single-TF symbols, the TypeError catch is correct. Need to verify `PriceFeedManager.get_window` signature. |

### 8.3 Kill Switch Thread Safety

| File | Line | Issue |
|------|------|-------|
| `trading_runner.py` | 414-418 | `activate_kill_switch` uses `asyncio.run_coroutine_threadsafe` when loop is running. Dashboard runs in a separate thread. The call is thread-safe. |
| `trading_runner.py` | 416 | `fut.result(timeout=15)` — if kill switch takes longer (e.g. many positions to flatten), could timeout. |

---

## 9. Config Validation

### 9.1 live_trading_config.json

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 95-96, 99-111 | `load_best_per_symbol` reads JSON. No validation of `recommendations` structure. If a rec is missing `symbol`, `type`, `strategy`, `params`, `interval`, `leverage`, `build_adapter` will raise KeyError. |
| `run_live_trading.py` | 141-149 | `build_adapter` assumes `rec["type"] in ("single-TF", "multi-TF")`. Unknown type raises KeyError on `rec["interval"]`. |
| `run_live_trading.py` | 145-146 | `rec["params"]` passed to `tuple()`. If `params` is not a list (e.g. null or string), `tuple(rec["params"])` may produce wrong result. |
| `daily_research.py` | 325-327 | `phase3_compare` assumes `cur["type"] == "single-TF"` for param comparison. Multi-TF recs skip the `same_strategy`/`same_params` check. |

### 9.2 No Schema Validation

| File | Line | Issue |
|------|------|-------|
| All | — | No jsonschema or pydantic validation of `live_trading_config.json`. Malformed config causes runtime errors. |

---

## 10. Recovery

### 10.1 Process Restart

| File | Line | Issue |
|------|------|-------|
| `trading_runner.py` | 396-332 | `restore_from_journal` loads `get_latest_account_state` and restores broker/positions/entry_prices. Good. |
| `trading_runner.py` | 401-405 | `restore_target = inner if inner is not None else self._broker`. Uses `restore_state` if available. |
| `run_live_trading.py` | 376 | `runner.restore_from_journal()` called before `runner.run()`. Correct order. |

### 10.2 State Restoration Gaps

| File | Line | Issue |
|------|------|-------|
| `trading_runner.py` | 318-323 | `_entry_prices` restored from `entry_prices` in journal. `_sl_triggered` is not persisted. After restart, SL/TP state is lost; could re-trigger on first tick if price crosses. |
| `trading_runner.py` | 81-82 | `_sl_triggered: Dict[str, bool]` — not saved to journal. |

### 10.3 Journal Path

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 223, 353 | `--db-path` default `"live_trading.db"` — relative. If user runs from different directory after restart, a new DB is created; prior state is lost. |

---

## 11. Timezone Issues

### 11.1 Inconsistent Timezone Handling

| File | Line | Issue |
|------|------|-------|
| `daily_research.py` | 159 | `last_date.to_pydatetime().replace(tzinfo=None)` — strips timezone for `days_since` calc. Compares naive to `datetime.now()` (local). If server is UTC and data is ET, off by hours. |
| `daily_research.py` | 139, 186-187 | `resample_to_4h` and concat use `utc=True`. `_incremental_download` line 159 uses naive. |
| `download_data.py` | 134 | `pd.to_datetime(df["date"])` — no timezone. yfinance often returns UTC for crypto, local for stocks. |
| `price_feed.py` | 196 | `pd.Timestamp(row[date_col]).to_pydatetime()` — no timezone conversion. |
| `run_production_scan.py` | 102 | `ts = df["date"].values.astype("datetime64[s]").astype(np.float64)` — timezone info lost. |

### 11.2 YFinanceFeed Market Hours

| File | Line | Issue |
|------|------|-------|
| `price_feed.py` | 211-222 | `_is_us_market_hours` uses `ZoneInfo("US/Eastern")`. Requires Python 3.9+ or `backports.zoneinfo`. `except Exception: return True` — if ZoneInfo fails, assumes market open. Fallback may be wrong on non-US systems. |

---

## 12. Unicode/Encoding

### 12.1 Symbol Names

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 72-77 | `to_feed_symbol` and `STOCK_TICKER_MAP` use ASCII. `MA_stock` → `MA`. No non-ASCII symbols in current sets. |
| `run_production_scan.py` | 69 | `sym = f.replace(".csv", "")` — filename used as symbol. If filename has non-ASCII (e.g. from manual rename), could cause issues. |
| `download_data.py` | 150 | `path = os.path.join(data_dir, f"{sym}.csv")` — sym from dict keys, all ASCII. |

### 12.2 File Paths

| File | Line | Issue |
|------|------|-------|
| All | — | No handling of paths with non-ASCII characters (e.g. `C:\Users\用户名\...`). `os.path.join` and `Path` generally handle UTF-8 on modern Python, but CSV read/write should use `encoding="utf-8"` explicitly. |

---

## 13. Additional Critical Issues

### 13.1 run_production_scan Phase 0 Data Loading Bug

| File | Line | Issue |
|------|------|-------|
| `run_production_scan.py` | 386-389 | `data_4h = load_daily(os.path.join(data_dir, "4h")) if not os.path.isdir(...) else load_intraday(...)`. Logic is inverted: `load_daily` expects a directory of CSV files. `os.path.join(data_dir, "4h")` is the 4h subdir. So if 4h dir exists, use `load_intraday`. If not, use `load_daily` on "4h" path — but that path would be a dir name "4h" under data_dir, so `load_daily` would look for CSVs in `data/4h/`. Actually `load_daily` does `os.listdir(data_dir)` and joins `os.path.join(data_dir, f)`. So `load_daily(data/4h)` would load from data/4h/*.csv. The condition `not os.path.isdir(os.path.join(data_dir, "4h"))` — if 4h does NOT exist, we call `load_daily(data_dir)` for data_4h? No: `load_daily(os.path.join(data_dir, "4h"))`. So we're passing `data/4h` to load_daily. If `data/4h` doesn't exist, `load_daily(".../data/4h")` would list a non-existent dir — `os.listdir` raises FileNotFoundError. So the logic is: if 4h dir exists, load_intraday. If not, load_daily from 4h — but 4h doesn't exist, so we'd get an error. This is a bug: when 4h doesn't exist, we shouldn't call load_daily on it. |

Re-reading: `data_4h = load_daily(os.path.join(data_dir, "4h")) if not os.path.isdir(os.path.join(data_dir, "4h")) else load_intraday(data_dir, "4h")`

So: if 4h is NOT a directory → load_daily(data/4h). That would try to list data/4h which doesn't exist. Bug.
If 4h IS a directory → load_intraday. Correct.

The fix should be: if 4h exists, load_intraday. Else, data_4h = {}.

### 13.2 Portfolio Weights Application Bug

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 363-368 | When `portfolio_weights` is used, only `first_config_sym`'s weight is applied to `pos_size`. All strategies get the same position size. The comment says "per-symbol overrides available" but the code only uses the first symbol's weight for the runner's global `position_size_pct`. Per-symbol weights are never passed to the runner. |

### 13.3 Recommendation Rank Missing

| File | Line | Issue |
|------|------|-------|
| `run_live_trading.py` | 323 | Logs `rec["rank"]` but `load_best_per_symbol` does not set `rank` on the dict. It takes the first matching rec per symbol. The `rank` key may not exist in the config; it's set in `phase4_export` of run_production_scan. So it should exist. But if config is hand-edited and rank is removed, KeyError. |

### 13.4 Dashboard get_daily_pnl(1) IndexError Risk

| File | Line | Issue |
|------|------|-------|
| `app.py` | 707-708 | `pnl_df = get_daily_pnl(1)` then `daily = float(pnl_df.iloc[0]["daily_pnl"])`. If `get_daily_pnl(1)` returns columns but 0 rows (e.g. no trades yet), `iloc[0]` raises IndexError. The check `if not pnl_df.empty` is done, so empty returns 0.0. But what if it has columns and 0 rows? `not pnl_df.empty` is False, so we'd take the else branch... No, we'd go to `daily = float(pnl_df.iloc[0]["daily_pnl"])` — we only skip that if `not pnl_df.empty` is False. So we only access iloc[0] when not empty. Good. |

Actually the code is:
```python
pnl_df = get_daily_pnl(1)
daily = float(pnl_df.iloc[0]["daily_pnl"]) if not pnl_df.empty else 0.0
```
So when empty, we use 0.0. Safe.

### 13.5 Triggered Mode With Empty alert_syms

| File | Line | Issue |
|------|------|-------|
| `daily_research.py` | 447-450 | When `args.mode == "daily"` and `alert_syms` is non-empty, mode switches to "triggered". |
| `daily_research.py` | 458-459 | In triggered mode, `scan_syms = alert_syms`. If for some reason `alert_syms` is empty (e.g. logic error), `phase2_rescan` gets `scan_syms=[]`. `run_robust_scan` with empty symbols — need to check. `phase2_rescan` filters `crypto_data` and `stock_data` by `symbols`; if symbols is empty, both filters yield empty, and the loops `for label, sub_data in [...]` skip. So we get `ranking=[]`. No crash, but rescan does nothing. |

---

## 14. Summary of Recommendations

### High Priority

1. **run_production_scan.py L386-389:** Fix phase 0 data loading — when 4h/1h dirs don't exist, use `{}` instead of calling `load_daily` on non-existent path.
2. **run_live_trading.py:** Validate `--position-size`, `--lookback`, `--max-daily-loss`; use absolute paths for config and db defaults.
3. **Config validation:** Add validation for `live_trading_config.json` structure before use (required keys, types).
4. **daily_research.py:** Add `encoding="utf-8"` to CSV read/write; fix timezone consistency in `_incremental_download`.

### Medium Priority

5. **price_feed.py:** Add timeout to yfinance calls; add retry with backoff for `YFinanceFeed.start`.
6. **trading_runner.py:** Persist `_sl_triggered` or document that SL state is lost on restart.
7. **run_live_trading.py:** Fix portfolio weights — pass per-symbol position sizes to the runner or document limitation.
8. **Signal handling:** Ensure `runner.stop()` is called on KeyboardInterrupt before exit.

### Low Priority

9. **run_production_scan.py L324:** Log failed multi-TF backtests instead of silent `pass`.
10. **daily_research.py L174:** Log yfinance exceptions in `_incremental_download`.
11. **File locking:** Consider advisory locking when writing live_trading_config.json from multiple scripts.

---

*End of report*
