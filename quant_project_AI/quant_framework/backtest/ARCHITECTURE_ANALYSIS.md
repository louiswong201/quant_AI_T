# Backtest Engine: Deep Architectural Analysis

This document provides a comprehensive analysis of the backtest engine at `quant_project_AI/quant_framework/backtest/`, covering architecture, execution flow, redundancy, and optimization opportunities ranked by impact.

---

## 1. backtest_engine.py

### How BacktestEngine Works

**Dual execution paths:**

1. **Kernel fast-path** (~36k runs/s): When `strategy.kernel_name` exists, is in `KERNEL_REGISTRY`, and `len(symbols) == 1`, the engine bypasses the Python event loop and calls `_run_kernel()`, which invokes `eval_kernel_detailed()` — a single Numba-compiled loop.

2. **Python event-driven path** (~500 runs/s): For multi-symbol runs or custom strategies without a kernel, the engine uses:
   - `CostModelFillSimulator` — execution prices, slippage, market impact
   - `DefaultOrderManager` — order lifecycle
   - `PortfolioTracker` — positions, cash, equity curve

**Execution flow (Python path, per bar):**

```
for i in range(n):
  0) fill_sim.reset_bar()
  1) order_mgr.expire_stale(i, current_date)
  2) Fill pending orders (try_fill_pending per symbol) → portfolio.apply_fill
  3) process_bar_costs (funding, stop-loss, take-profit, liquidation) → auto_closes
  4) Execute auto-closes (stop-loss/take-profit/liquidation)
  5) strategy.cash, strategy.portfolio_value = ...
  6) _dispatch_signals (strategy.on_bar or on_bar_fast)
  7) Process signals → create_order, optionally execute_market (current_close mode)
  8) portfolio.record_bar(i)
```

### Hot Paths

- **Lines 224–364**: The main `for i in range(n)` loop — O(n) bars.
- **Lines 235–256**: `try_fill_pending` per symbol, then `apply_fill` per fill.
- **Lines 298–301**: `_dispatch_signals` — strategy callback.
- **Lines 419–427**: `_prepare_bar_data` — `_rolling_vol(close, vol_window)` has a Python for-loop.

### Memory Allocation

- Pre-allocated: `portfolio_values`, `daily_returns`, `cumulative_returns` in `PortfolioTracker`.
- Per-bar: `current_prices = {sym: float(bar_data_map[sym].close[i]) for sym in bar_data_map}` — new dict each bar.
- `_bar_results` in PortfolioTracker: `list.append` each bar — grows dynamically.

### Numba Candidates

- `_rolling_vol` (lines 56–71): Python `for i in range(window, n)` with `np.std(ret[i-window+1:i+1])` — O(n×window). Can be replaced with a vectorized or Numba rolling std.

---

## 2. fill_simulator.py

### How CostModelFillSimulator Works

- **Responsibilities**: Match pending orders to bar OHLCV, compute execution price (slippage + market impact), enforce liquidity limits.
- **Cost model**: Uses `config.fill_price_buy/sell()` for slippage, then `_compute_market_impact()` (Almgren-Chriss: `impact_bps = coeff * participation^exponent`).
- **Order types**: Market (immediate at ref price), Limit (range + probabilistic fill), Stop (trigger-based).

### Duplication with kernels.py?

**No.** CostModelFillSimulator is used only on the Python event-driven path. The kernel path uses its own inline cost model (`sb`, `ss`, `cm`, `lev`, `dc`, `sl`, `pfrac`, `sl_slip`) in `_fx_lev`, `_sl_exit`, `_mtm_lev`. The two are complementary:

- Kernels: single-symbol, pre-compiled, minimal costs (slippage/commission/leverage).
- FillSimulator: multi-symbol, limit/stop, Almgren-Chriss market impact, participation caps.

### Optimizations

1. **`try_fill_pending`**: Builds `symbol_orders` and `other_orders` via list comprehensions; sorts with `_pending_priority_key`. For typical k < 10 orders, overhead is acceptable.

2. **`_match_price` / `_limit_fill_probability`**: Pure floating-point logic; Numba would help only if matching hundreds of orders per bar.

3. **`_used_volume` dict**: `reset_bar()` clears every bar; `_record_usage` mutates. Fine for current design.

---

## 3. order_manager.py

### How DefaultOrderManager Works

- **State**: `_pending` (list), `_all_orders` (list), `_index` (order_id → index).
- **Flow**: `create_order` → `submit` → `_pending.append`, `_index[oid]=len(_all_orders)`, `_all_orders.append`.
- **cancel**: Rebuilds `_pending` via list comprehension — O(k).
- **expire_stale**: Rebuilds `_pending` — O(k).
- **remove_filled**: `[o for o in _pending if o.order_id not in id_set]` — O(k).

### Hot Paths

- `expire_stale` and `remove_filled` run every bar.
- For k &lt; 100 orders typical in backtests, impact is small.

### Data Structure Choices

- **Mutable Order dataclass** is intentional (avoids copying for small pending queues).
- **Dict for order_id lookup** gives O(1) access; suitable for backtest scale.

### Optimizations

1. **In-place filtering**: Avoid rebuilding `_pending` in `expire_stale` and `remove_filled`; use a second pass to compact in-place or a double-ended structure.
2. **Batch remove_filled**: Engine calls `remove_filled([order.order_id])` one-by-one for auto-closes; could batch.

---

## 4. portfolio.py

### How PortfolioTracker Works

- Tracks: `_cash`, `_positions`, `_entry_prices`, `_high_water`, `_portfolio_values`, `_cumulative_returns`, `_daily_returns`, `_trades`, `_bar_results`.
- **apply_fill**: Updates positions/cash, computes VWAP for adds, records trade dict.
- **mark_to_market**: `sum(qty * px)` over positions.
- **process_bar_costs**: Per position: funding, borrow, trailing stop, stop-loss, take-profit, liquidation.

### Redundant Computation

1. **`mark_to_market(current_prices)`** called twice per bar (lines 290 and 363 in backtest_engine) — once before signals, once after. Could cache or call once at end-of-bar.

2. **`record_bar`** appends to `_bar_results` a dict containing `positions`, `date`, `portfolio_value`, `cash`, `daily_return`, `cumulative_return` — full copy of positions each bar. For long backtests this grows memory usage.

### Hot Paths

- `process_bar_costs`: Iterates all positions; dict lookups for `entry`, `high_water`.
- `apply_fill`: Dict get/set and VWAP math.
- `record_bar`: One append + array writes.

### Optimizations

1. **Pre-allocate `_bar_results`**: Replace `list.append` with pre-sized structure or only store indices/cash/pv if full history is not required.
2. **Lazy `_bar_results`**: Store only when explicitly requested (e.g., for reporting), not every bar.
3. **Reduce mark_to_market calls**: Compute once per bar and reuse.

---

## 5. protocols.py

### Interfaces Defined

- **Value objects**: `OrderSide`, `OrderType`, `OrderStatus`, `Order`, `Fill`, `BarData`.
- **Protocols**: `IFillSimulator`, `IOrderManager`, `IPortfolioTracker`.

`BarData` holds contiguous float64 arrays (`open`, `high`, `low`, `close`, `volume`, `rolling_vol`) to avoid DataFrame access in the loop.

### Data Structure Choices

- **Dataclasses** for Order/Fill: good for clarity; slight overhead vs namedtuple.
- **BarData with numpy arrays**: Correct choice for hot-path indexing.

---

## 6. bias_detector.py

### What BiasDetector Does

- **detect_look_ahead**: Flags `market_fill_mode='current_close'` as look-ahead bias.
- **detect_survivorship_bias**: Heuristics on symbol count, end dates.
- **detect_data_snooping**: Bonferroni correction for N param combos vs OOS bars.
- **full_audit**: Runs all checks and returns `BiasReport`.

### Is It Slow?

**No.** All methods run once post-backtest. Survivorship iterates symbols and DataFrames; data-snooping is O(1). Not on the hot path.

---

## 7. tca.py

### What TransactionCostAnalyzer Does

- **analyze**: Iterates `trades_df.itertuples()`, computes commission + slippage cost per trade, then Sharpe (gross vs net).
- **to_markdown**: Formats report.

### Hot Path

- `trades_df.itertuples()` — O(num_trades). Runs once after backtest.

### Memory

- Builds `per_trade` list of dicts; acceptable for typical trade counts.

---

## 8. manifest.py

### What build_run_manifest Does

- Builds a manifest dict: strategy class, params, symbols, dates, cost config, `bars_by_symbol`, strategy source file SHA256.
- Uses `_file_sha256` for strategy source digest — reads file in 1MB chunks.

### Performance

- Runs once at end of backtest. File hashing cost is negligible relative to backtest time.

---

## 9. robust.py vs robust_scan.py

### run_robust_backtest (robust.py)

- **Purpose**: Run one strategy across multiple time windows and parameter sets.
- **Paths**: Kernel-capable → `_run_robust_kernel`; otherwise → `_run_robust_python` (calls `engine.run`).
- **Output**: List of per-run results + summary (by_param, by_window, table).

### run_robust_scan (robust_scan.py)

- **Purpose**: Parameter search with anti-overfitting (11-layer framework).
- **Flow**: Purged walk-forward → parameter scan in train windows only → validation gate → cross-window consistency → MC/Shuffle/Bootstrap on OOS → deflated Sharpe → composite ranking.
- **Uses**: `scan_all_kernels`, `eval_kernel`, Numba `perturb_ohlc`, `shuffle_ohlc`, `block_bootstrap_ohlc`.

### Relation

- **robust.py**: Executor for “run strategy X over windows × params.”
- **robust_scan.py**: Optimizer that finds best params and scores robustness.
- They share kernels and `config_to_kernel_costs` but serve different roles.

---

## 10. config.py

### BacktestConfig Structure

Frozen dataclass with:

- Fill mode, commission, slippage, liquidity/impact
- Leverage, shorting, stop-loss/take-profit/trailing
- Funding, borrow, margin, liquidation
- Bar interval, `bars_per_year`, `bars_per_day`
- Execution reporting options

### Factory Methods

- `from_legacy_rate(commission_rate)`
- `conservative()` — higher commissions/slippage
- `crypto()` — Binance-style
- `stock_ibkr()` — IBKR-style

### Cost Helpers

- `effective_slippage_bps(side)` — sqrt(leverage)-scaled
- `daily_funding_cost()`
- `borrow_rate_for(symbol)`
- `fill_price_buy/sell(ref_price)`
- `commission_buy/sell(notional)`

---

## Optimization Opportunities (Ranked by Impact)

### 1. **Vectorize / Numba _rolling_vol (backtest_engine.py)** — HIGH

- Current: Python `for i in range(window, n)` with `np.std(ret[i-window+1:i+1])` — O(n×window).
- Fix: Use `pd.Series(ret).rolling(window).std()` or a Numba rolling std.
- Impact: Reduces CPU in `_prepare_bar_data`, which runs once per symbol at startup and can dominate for large datasets.

### 2. **Avoid Redundant mark_to_market Calls (backtest_engine.py)** — MEDIUM–HIGH

- Called twice per bar (before and after processing). Cache the result and reuse.
- Impact: Saves O(positions) per bar across the full run.

### 3. **Pre-allocate / Lazy _bar_results (portfolio.py)** — MEDIUM

- `record_bar` appends a full dict (including `positions`) every bar.
- Options: Pre-allocate arrays for scalar fields only, or record only when needed.
- Impact: Lower memory and fewer allocations for long backtests.

### 4. **Numba Rolling Std in _rolling_vol** — MEDIUM

- Even if not fully vectorized, a Numba `@njit` rolling std would remove Python loop overhead.
- Impact: Noticeable for strategies with large `impact_vol_window`.

### 5. **Batch remove_filled and In-Place Pending Updates (order_manager.py)** — LOW–MEDIUM

- Engine often calls `remove_filled` with single IDs. Batching reduces list rebuilds.
- In-place compaction for `expire_stale` and `remove_filled` avoids full list copies.
- Impact: Relevant when many orders per bar.

### 6. **Optimize current_prices Dict (backtest_engine.py)** — LOW

- `current_prices = {sym: float(bar_data_map[sym].close[i]) for sym in bar_data_map}` creates a new dict each bar.
- Could reuse a dict and update in-place if keys are stable.
- Impact: Minor; dict creation is cheap for small symbol counts.

### 7. **Fill Simulator: Numba for _compute_market_impact** — LOW

- Logic is O(1) per fill. Extracting to Numba would give minimal gain.
- Impact: &lt;1% for typical fill rates.

### 8. **TCA: Vectorize over trades** — LOW

- `itertuples` is already efficient. Full vectorization possible but complexity high.
- Impact: TCA runs once; unlikely to affect backtest runtime.

---

## Summary Table

| File            | Hot Path                           | Memory Pattern                  | Numba Candidate      | Redundancy                            |
|-----------------|------------------------------------|---------------------------------|----------------------|---------------------------------------|
| backtest_engine | Bar loop, _dispatch_signals        | Pre-alloc arrays, dict/bar      | _rolling_vol         | mark_to_market 2×/bar                 |
| fill_simulator  | try_fill_pending                   | Dict _used_volume               | Low value            | None vs kernels                       |
| order_manager   | expire_stale, remove_filled        | List rebuild                    | No                   | None                                  |
| portfolio       | process_bar_costs, record_bar      | list.append _bar_results       | No                   | mark_to_market caller redundancy      |
| protocols       | N/A (interfaces)                   | BarData arrays                  | N/A                  | None                                  |
| bias_detector  | Post-run only                      | Minimal                         | No                   | None                                  |
| tca             | Post-run only                      | per_trade list                 | No                   | None                                  |
| manifest        | Post-run only                      | SHA256 read                     | No                   | None                                  |
| robust          | Window × param loop                | Per-run dicts                   | Uses kernels         | None                                  |
| robust_scan     | scan_all_kernels, perturbations    | Precomputed indicators          | Already Numba        | None                                  |
| config          | One-time setup                     | Frozen dataclass                | N/A                  | None                                  |
| kernels         | Numba kernels                      | Pre-alloc in kernels            | Already @njit        | Cost model vs fill_sim (different)    |
