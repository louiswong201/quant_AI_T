# Round 3: Integration & Data Flow Review

**Scope:** End-to-end data flow across download → load → backtest → live trading.

---

## 1. Format Mismatches

### 1.1 CSV Column Naming — **CONSISTENT**
- **download_data.py** (L91–125): Standardizes to `date`, `open`, `high`, `low`, `close`, `volume`.
- **run_production_scan.py** (L73–82, L98–108): Expects `close`, `open`, `high`, `low`; `load_intraday` also expects `date`.
- **daily_research.py** (L112–134): Same `process_df` logic.
- **price_feed.py** (L336–347): `_load_local_ohlcv` accepts `date` or `datetime` (renames to `date`).
- **Verdict:** Column names are aligned.

### 1.2 Volume Column — **MINOR INCONSISTENCY**
- **download_data.py** (L119): Includes `volume` only if present; saves optional.
- **FileDataAdapter** (L77): Requires `volume` in `required_columns`; fails if missing.
- **run_production_scan.load_daily** (L78–81): Does not require `volume`; only uses OHLC.
- **Impact:** DataManager/Dataset loading of CSVs without volume will fail; production scan does not.

---

## 2. Path Inconsistencies

### 2.1 Data Directory Layout — **CRITICAL**

| Script | Daily Path | Intraday Path |
|--------|------------|---------------|
| **download_data.py** | `data/{sym}.csv` | `data/1h/{sym}_1h.csv`, `data/4h/{sym}_4h.csv` |
| **run_production_scan** | `data/*.csv` (flat) | `data/1h/*_1h.csv`, `data/4h/*_4h.csv` |
| **daily_research** | Same | Same |
| **price_feed._load_local_ohlcv** | `data/{interval}/{prefix}_{interval}.csv` | Same |

**Bug:** For `interval="1d"`, `_load_local_ohlcv` looks for `data/1d/BTC_1d.csv`, but **download_data.py** writes daily to `data/BTC.csv`. Daily data is never loaded from disk by `BinanceCombinedFeed`; it always falls back to Binance API.

- **File:** `quant_framework/live/price_feed.py` L331–352
- **Fix:** Add a branch for `interval == "1d"` to check `data_dir / f"{prefix}.csv"` (flat layout).

### 2.2 Config Path Resolution
- **run_production_scan** (L371–372): `data_dir = os.path.join(base_dir, "data")`; `reports_dir` for config.
- **run_live_trading** (L182, L245): Default `--config reports/live_trading_config.json`; resolved via `Path(args.config)` without `base_dir` prepended when checking existence.
- **daily_research** (L399): `config_path = os.path.join(base_dir, args.config)`.
- **Issue:** `run_live_trading` uses `Path(args.config)`; if run from a different cwd, `reports/live_trading_config.json` may not resolve to the project root. `run_production_scan` and `daily_research` use `base_dir` explicitly.
- **File:** `run_live_trading.py` L245–248
- **Fix:** Resolve config path relative to script directory: `config_path = Path(__file__).resolve().parent / args.config` when `args.config` is relative.

---

## 3. Timezone Handling

### 3.1 Inconsistent UTC Usage
- **download_data.py** (L133): `resample_to_4h` uses `pd.to_datetime(df["date"])` — no `utc=True`.
- **daily_research.py** (L140, L165, L192–193): `pd.to_datetime(..., utc=True)` in `resample_to_4h` and incremental refresh.
- **price_feed.py** (L337, L375, L448): Binance timestamps use `timezone.utc`; `_load_local_ohlcv` uses `pd.to_datetime(df["date"])` without `utc=True`.
- **Impact:** Mixed naive/aware datetimes can cause resampling and merge issues.
- **Files:** `download_data.py` L133; `daily_research.py` L140; `quant_framework/live/price_feed.py` L448

### 3.2 run_production_scan load_daily
- **run_production_scan.py** (L71): `load_daily` does not use `parse_dates`; date column remains string. Only OHLC arrays are used, so no direct bug, but inconsistent with `load_intraday` (L95) which uses `parse_dates=["date"]`.

---

## 4. Symbol Naming

### 4.1 Conversion Points
- **download_data.py**: Uses `CRYPTO_SYMBOLS` (e.g. `BTC` → `BTC-USD`) and `STOCK_SYMBOLS` (e.g. `MA_stock` → `MA`).
- **run_production_scan**: Uses file basenames as symbols (`BTC`, `MA_stock` from `BTC.csv`, `MA_stock.csv`).
- **run_live_trading** (L72–77): `to_feed_symbol()` maps config symbols to feed tickers: `BTC` → `BTCUSDT`, `MA_stock` → `MA`.
- **price_feed._symbol_to_file_prefix** (L322–328): `BTCUSDT` → `BTC` for local file lookup.
- **Verdict:** Conversions are consistent; `CRYPTO_BASE` in run_live_trading (L56–59) includes `LINK`, `UNI` not in download_data’s `CRYPTO_SYMBOLS`.

---

## 5. Missing Data Handling

### 5.1 run_production_scan — **CRITICAL BUG**
```python
# L513-514
data_4h = load_daily(os.path.join(data_dir, "4h")) if not os.path.isdir(os.path.join(data_dir, "4h")) else load_intraday(data_dir, "4h")
data_1h = load_daily(os.path.join(data_dir, "1h")) if not os.path.isdir(os.path.join(data_dir, "1h")) else load_intraday(data_dir, "1h")
```
**Logic is inverted:**
- When `data/4h` does **not** exist: calls `load_daily("data/4h")` → `os.listdir("data/4h")` → **FileNotFoundError**.
- When `data/4h` exists: correctly calls `load_intraday`.

**Fix:**
```python
data_4h = load_intraday(data_dir, "4h") if os.path.isdir(os.path.join(data_dir, "4h")) else {}
data_1h = load_intraday(data_dir, "1h") if os.path.isdir(os.path.join(data_dir, "1h")) else {}
```

- **File:** `run_production_scan.py` L513–514

### 5.2 Other Missing-Data Behavior
- **load_daily** / **load_intraday**: Skip symbols with `< min_bars`; no explicit error.
- **price_feed._load_local_ohlcv**: Returns `None` on failure; feed falls back to API.
- **BinanceCombinedFeed.start**: Logs "history load failed" but continues; empty windows may cause strategy issues.

---

## 6. Config Format (live_trading_config.json)

### 6.1 Writer vs Readers
- **phase4_export** (run_production_scan L403–356): Writes `recommendations` with `symbol`, `type`, `strategy`, `params`, `leverage`, `interval`, `tf_configs` (multi-TF), `backtest_metrics`, etc.
- **load_best_per_symbol** (run_live_trading L82–111): Expects `cfg["recommendations"]`, `rec["symbol"]`, `rec["type"]`, `rec["interval"]`, `rec["params"]`, `rec["tf_configs"]`.
- **daily_research** (L419–421): Expects `live_config["recommendations"]`, `r["symbol"]`.
- **Verdict:** Structure is consistent.

### 6.2 Multi-TF vs Single-TF
- Single-TF: `interval`, `params`, `strategy`.
- Multi-TF: `tf_configs`, `tf_combo`, `mode`.
- `build_adapter` handles both; no format mismatch.

---

## 7. bars_per_year Consistency

### 7.1 BacktestConfig
- **config.py** (L15–28): `_BARS_PER_YEAR_CRYPTO` and `_BARS_PER_YEAR_STOCK` with 1m, 5m, 15m, 1h, 4h, 1d.
- Crypto 1d: 365; 4h: 2190; 1h: 8760.
- Stock 1d: 252; 4h: 441; 1h: 1764.

### 7.2 Monitor Engine — **BUG**
- **monitor.py** (L80–81): `_BARS_PER_YEAR = {"1d": 252, "4h": 1512, "1h": 6048}` — stock-style values only.
- 1512 = 252×6, 6048 = 252×24 (trading hours).
- **Impact:** Monitor treats all symbols as stocks. Crypto strategies get incorrect Sharpe and trade-frequency annualization.
- **File:** `quant_framework/research/monitor.py` L80–81, L117
- **Fix:** Pass asset class (crypto vs stock) into `compute_health_metrics` and select `_BARS_PER_YEAR` accordingly, or use `BacktestConfig.bars_per_year` when available.

---

## 8. requirements.txt

### 8.1 Imported vs Listed
- **yfinance**: Used by download_data, daily_research, price_feed — listed.
- **aiohttp**: Used by price_feed (Binance REST) — listed.
- **websockets**: Used by price_feed (Binance WS) — listed.
- **dash, plotly, dash-bootstrap-components**: Dashboard — listed.
- **polars**: Optional in Dataset/DataManager — commented out.
- **sqlalchemy**: Optional — commented out.

### 8.2 Potential Gaps
- **pandas**, **numpy**, **scipy**, **scikit-learn**, **numba**, **pyarrow**: All used and listed.
- No version conflicts identified.

---

## 9. Error Messages

### 9.1 Weak or Missing Messages
- **run_production_scan** (L513–514): No message before `FileNotFoundError` when 4h/1h dirs missing.
- **load_daily** (L74–76): Silently skips symbols missing OHLC columns (no log).
- **price_feed._load_local_ohlcv** (L350): `logger.debug` only; failures are quiet.
- **BinanceCombinedFeed** (L346): "history load failed" is a warning; empty window is not clearly surfaced.

### 9.2 Suggestions
- Add explicit check: "4h/1h data directories not found; run download_data.py first."
- Log when symbols are skipped due to missing columns.
- Use `logger.info` or `logger.warning` when local load fails and API fallback is used.

---

## 10. Logging Consistency

### 10.1 Logger Names
- **run_live_trading**: `logger = logging.getLogger("live_trading")`.
- **price_feed**: `logger = logging.getLogger(__name__)`.
- **trading_runner**: `logger = logging.getLogger(__name__)`.
- **kernel_adapter**: `logger = logging.getLogger(__name__)`.

### 10.2 Levels
- Mix of `info`, `warning`, `debug`, `error`; generally appropriate.
- **run_production_scan** and **daily_research**: Use `print()` instead of logging — inconsistent with live trading.

---

## 11. Stock Single-TF Interval Mismatch — **CRITICAL**

### 11.1 Problem
- **run_live_trading** (L304–308): For crypto, uses `add_symbol_multi_tf(fsym, intervals)`. For stocks, uses `add_symbol(fsym)`.
- `add_symbol` uses `PriceFeedManager._interval`, which is hardcoded to `"1d"` (L288).
- **Impact:** Stock strategies optimized on 1h or 4h receive 1d data in live trading. Signals will not match backtest.

### 11.2 Fix
- For single-TF stocks, pass the strategy’s interval into the feed. For example:
  - Add `add_symbol(symbol, interval=...)` and use it when registering stocks, or
  - Use `add_symbol_multi_tf(symbol, [interval])` for stocks as well, so the correct interval is used.

- **File:** `run_live_trading.py` L304–308, L288

---

## 12. Data Loading: quant_framework/data vs Entry Points

### 12.1 DataManager / Dataset
- **data/__init__.py**: Exposes `DataManager`, `Dataset`, `CacheManager`, `FundingRateLoader`, `RagContextProvider`.
- **FileDataAdapter**: Expects `data/{symbol}.csv` or `data/{symbol}.parquet` (flat layout).
- **Dataset.load**: Tries binary → arrow → parquet → file adapter; no support for `data/1h/` or `data/4h/`.

### 12.2 Entry Points
- **run_production_scan**, **daily_research**: Use custom `load_daily`, `load_intraday` from `run_production_scan` — not DataManager.
- **Verdict:** Two parallel data-loading paths. DataManager is not used by the main production pipeline.

---

## Summary: Priority Fixes

| Priority | File | Line(s) | Issue |
|----------|------|---------|-------|
| P0 | run_production_scan.py | 513–514 | Inverted logic: FileNotFoundError when 4h/1h dirs missing |
| P0 | run_live_trading.py | 304–308 | Stocks always get 1d interval; 1h/4h strategies broken |
| P1 | price_feed.py | 331–352 | Daily local path: add support for `data/{sym}.csv` |
| P1 | run_live_trading.py | 245–248 | Resolve config path relative to script directory |
| P1 | monitor.py | 80–81, 117 | bars_per_year assumes stocks; crypto Sharpe wrong |
| P2 | download_data.py | 133 | Use `utc=True` in resample_to_4h for consistency |
| P2 | daily_research.py | 140 | Already uses utc; align download_data |
| P2 | price_feed.py | 448 | Use `utc=True` in _load_local_ohlcv |
| P3 | FileDataAdapter | 77 | Make volume optional or add fallback |
| P3 | run_production_scan | 513+ | Add clear error when data dirs missing |
