# Live Trading Components — Code Review Report

**Scope:** Reliability, data integrity, correctness for live trading.  
**Files reviewed:** trading_runner, price_feed, trade_journal, kill_switch, alerts, audit, health_server, binance_futures, ibkr_broker, paper broker, execution_algo, rate_limiter.

---

## 1. trading_runner.py

| # | Severity | Category | Line(s) | Description | Proposed Fix |
|---|----------|----------|---------|------------|--------------|
| 1 | **CRITICAL** | DATA INTEGRITY | 194–217 | **Entry price bug on partial close:** When partially closing a long or short, the remaining position is assigned `fill_price` as its entry instead of the original entry. This corrupts unrealized P&L and SL/TP levels. | After `entry, old_qty = self._entry_prices.pop(symbol)`, distinguish partial close vs flip: if `(cur_qty > 0 and old_qty > 0) or (cur_qty < 0 and old_qty < 0)` use `(entry, cur_qty)`; else use `(fill_price, cur_qty)`. |
| 2 | **HIGH** | RELIABILITY | 333–336 | **Shutdown does not await feed stop:** `_shutdown` creates a task for `_feed.stop_all()` but does not await it. Process may exit before feeds and WebSockets are closed. | Use `asyncio.create_task` and store it, or call `loop.run_until_complete(self._feed.stop_all())` if loop is not running. Prefer an explicit shutdown sequence that awaits cleanup. |
| 3 | **HIGH** | RELIABILITY | 334 | **Deprecated `asyncio.get_event_loop()`:** Deprecated in Python 3.10+ and can return a stopped loop. | Use `asyncio.get_running_loop()` when in async context, or handle the case when no loop is running. |
| 4 | **MEDIUM** | RELIABILITY | 91, 96 | **Signal handler / atexit mismatch:** On Windows, `atexit.register(self._shutdown)` runs at interpreter exit; `_shutdown` creates an async task that may not complete. | Ensure shutdown is invoked from the main loop (e.g. via a shutdown event) and that `atexit` only triggers a synchronous flag that the loop checks. |
| 5 | **MEDIUM** | LOGIC | 367 | **`_allow_fractional` not unwrapped:** When broker is `RiskManagedBroker(PaperBroker(...))`, `getattr(self._broker, "_allow_fractional", False)` returns False because the wrapper has no such attribute. | Unwrap: `inner = getattr(self._broker, "_broker", self._broker)` then `getattr(inner, "_allow_fractional", False)`. |
| 6 | **MEDIUM** | EDGE CASE | 414–418 | **Kill switch timeout:** `fut.result(timeout=15)` may raise `TimeoutError` if closing many positions takes longer. | Increase timeout or make it configurable; handle `TimeoutError` and surface status to the caller. |
| 7 | **LOW** | EDGE CASE | - | **No duplicate bar handling:** If the feed emits the same bar twice (e.g. API overlap), the runner could double-trade. | Add a `_last_bar_key` (symbol + interval + timestamp) and skip processing if the same bar was already handled. |

---

## 2. price_feed.py

| # | Severity | Category | Line(s) | Description | Proposed Fix |
|---|----------|----------|---------|------------|--------------|
| 8 | **HIGH** | DATA INTEGRITY | 195, 246, 378 | **Timezone-naive timestamps:** YFinance and local CSV may return naive datetimes. `BarEvent.timestamp` should be UTC for consistency. | Normalize: `ts = pd.Timestamp(...).tz_localize(None).tz_localize("UTC")` or `ts.tz_convert("UTC")` if already timezone-aware. |
| 9 | **HIGH** | DATA INTEGRITY | 279–285 | **YFinance tick lacks running high/low:** `TickEvent` uses `running_high=price, running_low=price`. SL/TP checks rely on bar extremes; with only last price, intra-bar stops can be missed. | Document limitation or implement bar tracking (e.g. from last closed bar + current bar’s high/low if available). |
| 10 | **MEDIUM** | RELIABILITY | 363–365 | **BinanceFeed `stop()` races with iterator:** Setting `_running=False` and `await self._ws.close()` may not immediately unblock `async for msg in ws` if no message is pending. | Use a shared `asyncio.Event` for shutdown; have the iterator check it, or ensure `ws.close()` causes the iterator to exit promptly. |
| 11 | **MEDIUM** | RELIABILITY | 559–562 | **Failed feeds not restarted:** `PriceFeedManager.run()` uses `asyncio.gather(..., return_exceptions=True)`. A failed feed task ends; there is no restart. | Add per-feed error handling and restart logic (with backoff) for critical feeds. |
| 12 | **LOW** | RELIABILITY | 392 | **Redundant `getattr`:** `getattr(self, '_reconnect_attempts', 0)` is used when `_reconnect_attempts` is initialized in `__init__`. | Use `self._reconnect_attempts` directly. |

---

## 3. trade_journal.py

| # | Severity | Category | Line(s) | Description | Proposed Fix |
|---|----------|----------|---------|------------|--------------|
| 13 | **MEDIUM** | RELIABILITY | 104–127 | **No exception handling in `_flush_unlocked`:** If `executemany` or `commit` fails, the connection can be left inconsistent and buffers may be lost. | Wrap in try/except; on failure, `conn.rollback()` and re-raise or log; consider retry for transient errors. |
| 14 | **MEDIUM** | RELIABILITY | 393–396 | **`close()` can be called multiple times:** Second call to `close()` will operate on a closed connection. | Add `if self._conn is None: return` and set `self._conn = None` after closing. |
| 15 | **LOW** | EDGE CASE | 305–376 | **`get_latest_account_state` replays all trades:** For long-running journals, full replay can be slow. | Add optional `limit` or use incremental replay from last known snapshot. |

---

## 4. kill_switch.py

| # | Severity | Category | Line(s) | Description | Proposed Fix |
|---|----------|----------|---------|------------|--------------|
| 16 | **HIGH** | LOGIC | 43–44 | **Positions not synced before flatten:** `get_positions()` may be stale (e.g. Binance positions changed elsewhere). | Call `await exec_broker.sync_positions()` before iterating over positions. |
| 17 | **MEDIUM** | EDGE CASE | 35–41 | **`get_open` / `cancel_order` may not exist:** For brokers without these, order cancellation is skipped. | Document which brokers support cancellation; consider a no-op implementation for brokers that don’t. |

---

## 5. alerts.py

| # | Severity | Category | Line(s) | Description | Proposed Fix |
|---|----------|----------|---------|------------|--------------|
| 18 | **MEDIUM** | RELIABILITY | 78–90 | **`close()` uses deprecated `get_event_loop()` and may not await:** `loop.run_until_complete(self._session.close())` can fail or block incorrectly in async contexts. | Add an `async def aclose()` and document that callers should await it during shutdown. |
| 19 | **LOW** | RELIABILITY | 84–86 | **Fire-and-forget `create_task` for session close:** When loop is running, `create_task(self._session.close())` is not awaited; session may not close before exit. | Await the close task or use a shutdown hook that ensures the session is closed. |

---

## 6. audit.py (AuditTrail)

| # | Severity | Category | Line(s) | Description | Proposed Fix |
|---|----------|----------|---------|------------|--------------|
| 20 | **MEDIUM** | LOGIC | - | **AuditTrail not integrated:** No calls to `record_order_lifecycle` or `record_event` from trading_runner or brokers. Orders are not audited. | Wire AuditTrail into the order flow (e.g. in RiskManagedBroker or trading_runner) and record lifecycle events. |
| 21 | **LOW** | RELIABILITY | 181–184 | **`close()` on already-closed connection:** Same pattern as trade_journal. | Add a guard and set `self._conn = None` after close. |

---

## 7. health_server.py

| # | Severity | Category | Line(s) | Description | Proposed Fix |
|---|----------|----------|---------|------------|--------------|
| 22 | **LOW** | RELIABILITY | 55–59 | **Blocking health/metrics providers:** If providers block (e.g. DB or network), the HTTP handler blocks. | Run providers in `asyncio.to_thread()` or ensure they are non-blocking. |

---

## 8. binance_futures.py

| # | Severity | Category | Line(s) | Description | Proposed Fix |
|---|----------|----------|---------|------------|--------------|
| 23 | **HIGH** | RELIABILITY | 318–321 | **Keepalive task leak on reconnect:** Each `_user_data_stream` reconnect creates a new `_keepalive_listen_key` task; previous ones are not cancelled. | Cancel the previous keepalive task before creating a new one, or use a single long-lived keepalive task. |
| 24 | **MEDIUM** | LOGIC | 186–194 | **`submit_order` uses `get_event_loop()`:** Can fail when no loop exists or when called from an async context. | Prefer `asyncio.get_running_loop()` and `run_coroutine_threadsafe` when in async context; avoid creating new loops. |
| 25 | **MEDIUM** | LOGIC | 184–194 | **Limit orders reported as "submitted":** For limit orders, Binance may return `NEW`; code returns `"submitted"`. Trading_runner treats non-`"filled"` as rejected and does not record the trade. | Either document that only market orders are supported for live fills, or add async fill handling (WebSocket / polling) for limit orders. |
| 26 | **LOW** | RELIABILITY | 294–296 | **`_handle_order_update` is empty:** Order status and partial fills are not reflected in broker state. | Implement order status tracking and partial fill handling if needed for live P&L. |

---

## 9. ibkr_broker.py

| # | Severity | Category | Line(s) | Description | Proposed Fix |
|---|----------|----------|---------|------------|--------------|
| 27 | **MEDIUM** | RELIABILITY | 97–98 | **`submit_order` uses `get_event_loop()`:** Can raise `RuntimeError` when no loop exists. | Use `asyncio.get_running_loop()` or `asyncio.run()` with proper error handling. |
| 28 | **LOW** | EDGE CASE | 104–107 | **`get_open_orders_async` not overridden:** Base returns `[]`. Kill switch will not cancel IBKR orders. | Implement `get_open_orders_async` using `ib.openTrades()` if IBKR supports it. |

---

## 10. paper.py (PaperBroker)

| # | Severity | Category | Line(s) | Description | Proposed Fix |
|---|----------|----------|---------|------------|--------------|
| 29 | **MEDIUM** | LOGIC | - | **`reduce_only` ignored:** Kill switch sends `reduce_only=True`; PaperBroker does not enforce it. A bug could open a new position instead of only reducing. | Add a check: if `reduce_only` and position would flip sign, reject or cap shares to current position. |
| 30 | **LOW** | LOGIC | - | **No partial fill simulation:** All orders fill completely at the given price. | Optionally add configurable partial fill simulation for more realistic paper trading. |

---

## 11. execution_algo.py

| # | Severity | Category | Line(s) | Description | Proposed Fix |
|---|----------|----------|---------|------------|--------------|
| 31 | **MEDIUM** | EDGE CASE | 51–56 | **TWAP stops on first error/reject:** A single slice failure returns immediately; remaining slices are not attempted. | Document behavior or add retry/continue logic for transient failures. |
| 32 | **LOW** | EDGE CASE | 96–106 | **LimitChase `submit_fn` must be async:** `await submit_fn(signal)` assumes an awaitable; sync `submit_order` would raise. | Document that `submit_fn` must be async (e.g. `submit_order_async`). |

---

## 12. rate_limiter.py

| # | Severity | Category | Line(s) | Description | Proposed Fix |
|---|----------|----------|---------|------------|--------------|
| 33 | **LOW** | RELIABILITY | 34–36 | **Busy-wait in `acquire`:** `while not self.check(weight): await asyncio.sleep(0.05)` can spin for a long time under heavy load. | Consider a wait queue or event-based wake when capacity frees up. |

---

## Summary by Severity

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| HIGH | 6 |
| MEDIUM | 15 |
| LOW | 11 |

## Recommended Priority Order

1. **CRITICAL:** Fix entry price bug on partial close (trading_runner).
2. **HIGH:** Sync positions before kill switch; fix YFinance tick running high/low; ensure UTC timestamps; fix Binance keepalive leak; fix shutdown sequence.
3. **MEDIUM:** Wire AuditTrail; fix `_allow_fractional` unwrapping; add flush error handling; improve alert session cleanup; handle Binance limit order fills.
4. **LOW:** Duplicate bar handling; connection close guards; performance improvements.
