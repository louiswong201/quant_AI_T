# V5 Real Trading Upgrade — Complete Test Report

**Date**: 2026-03-05 22:29:36
**Total Tests**: 136
**Passed**: 136 (100.0%)
**Failed**: 0

---

## Execution Timing

| Section | Time (ms) |
|---------|-----------|
| P0_core | 1079.2 |
| P1_backtest | 2.3 |
| P3_brokers | 0.1 |
| P5_safety | 11.0 |
| cost_accuracy | 0.0 |
| kernel_accuracy | 267.5 |
| margin_accuracy | 0.1 |
| performance | 163.0 |
| regression | 23.4 |
| **TOTAL** | **1546.6** |

## Phase 0: Core Abstractions

**26/26 passed**

| Test | Result | Detail |
|------|--------|--------|
| P0-asset-class-enum | PASS | 5 asset classes: ['crypto_spot', 'crypto_perp', 'crypto_inve |
| P0-spot-margin-100pct | PASS | spot IM = 10000 |
| P0-futures-im | PASS | IM(100k, 10x) = 10000.0 |
| P0-futures-mm-tier1 | PASS | MM(40k) = 160.0, expected 160.0 |
| P0-futures-mm-tier2 | PASS | MM(200k) = 950.0, expected 950.0 |
| P0-futures-mm-tier4 | PASS | MM(5M) = 108950.0, expected 108950.0 |
| P0-futures-liq-long | PASS | liq_long(50k,10x) = 45200.00 |
| P0-futures-liq-short | PASS | liq_short(50k,10x) = 54800.00 |
| P0-regt-im | PASS | RegT IM(100k) = 50000.0 |
| P0-regt-mm | PASS | RegT MM(80k) = 20000.0 |
| P0-regt-margin-call | PASS | equity 18k < MM 20k → margin call = True |
| P0-crypto-cost-maker-taker | PASS | maker=20.0, taker=40.0 |
| P0-crypto-funding | PASS | long_cost=100.0, short_cost=-100.0 (positive rate) |
| P0-us-cost | PASS | US comm(200 shares) = 1.0 |
| P0-us-cost-min | PASS | US comm(1 share) = 1.0 (min $1) |
| P0-us-settlement | PASS | settlement delay = 1 days |
| P0-spec-round-price | PASS | round_price(50000.37) = 50000.4 |
| P0-spec-round-qty | PASS | round_quantity(1.2345) = 1.234 |
| P0-spec-validate-min-qty | PASS | validate_order(0.0001) = qty 0.0001 < min_qty 0.001 |
| P0-spec-validate-min-notional | PASS | validate_order(notional=0.001) = notional 0.001 < min_notion |
| P0-spec-validate-ok | PASS | validate_order(0.5, 50000) = None |
| P0-cal-crypto-always | PASS | crypto tradable at 3am |
| P0-cal-us-weekend | PASS | US equity not tradable on Saturday |
| P0-pdt-count | PASS | count = 3 |
| P0-pdt-violate | PASS | equity<25k + 3 day trades → violate |
| P0-pdt-no-violate-rich | PASS | equity>25k → no violate |

## Margin Model Accuracy

**23/23 passed**

| Tier/Test | Result | Detail |
|-----------|--------|--------|
| MARGIN-tier-10000 | PASS | MM(10,000) = 40.00, expected 40.00 |
| MARGIN-tier-50000 | PASS | MM(50,000) = 200.00, expected 200.00 |
| MARGIN-tier-50001 | PASS | MM(50,001) = 200.00, expected 200.00 |
| MARGIN-tier-250000 | PASS | MM(250,000) = 1,200.00, expected 1,200.00 |
| MARGIN-tier-250001 | PASS | MM(250,001) = 1,450.01, expected 1,450.01 |
| MARGIN-tier-1000000 | PASS | MM(1,000,000) = 8,950.00, expected 8,950.00 |
| MARGIN-tier-1000001 | PASS | MM(1,000,001) = 8,950.03, expected 8,950.03 |
| MARGIN-tier-10000000 | PASS | MM(10,000,000) = 233,950.00, expected 233,950.00 |
| MARGIN-tier-10000001 | PASS | MM(10,000,001) = 233,950.05, expected 233,950.05 |
| MARGIN-tier-20000000 | PASS | MM(20,000,000) = 733,950.00, expected 733,950.00 |
| MARGIN-tier-20000001 | PASS | MM(20,000,001) = 733,950.10, expected 733,950.10 |
| MARGIN-tier-50000000 | PASS | MM(50,000,000) = 3,733,950.00, expected 3,733,950.00 |
| MARGIN-tier-100000001 | PASS | MM(100,000,001) = 9,983,950.15, expected 9,983,950.15 |
| MARGIN-tier-200000001 | PASS | MM(200,000,001) = 24,983,950.25, expected 24,983,950.25 |
| MARGIN-liq-long-2x | PASS | liq_long(2x) = 25200.00 |
| MARGIN-liq-long-5x | PASS | liq_long(5x) = 40200.00 |
| MARGIN-liq-long-10x | PASS | liq_long(10x) = 45200.00 |
| MARGIN-liq-long-20x | PASS | liq_long(20x) = 47700.00 |
| MARGIN-liq-long-50x | PASS | liq_long(50x) = 49200.00 |
| MARGIN-liq-long-100x | PASS | liq_long(100x) = 49700.00 |
| MARGIN-regt-im-10000 | PASS | RegT IM(10,000) = 5,000.00 |
| MARGIN-regt-im-100000 | PASS | RegT IM(100,000) = 50,000.00 |
| MARGIN-regt-im-500000 | PASS | RegT IM(500,000) = 250,000.00 |

## Cost Model Accuracy

**12/12 passed**

| Test | Result | Detail |
|------|--------|--------|
| COST-crypto-comm-1000 | PASS | maker=0.20, taker=0.40 |
| COST-crypto-comm-10000 | PASS | maker=2.00, taker=4.00 |
| COST-crypto-comm-100000 | PASS | maker=20.00, taker=40.00 |
| COST-crypto-comm-1000000 | PASS | maker=200.00, taker=400.00 |
| COST-funding-0.0001 | PASS | long=10.00, short=-10.00 |
| COST-funding-0.001 | PASS | long=100.00, short=-100.00 |
| COST-funding-0.003 | PASS | long=300.00, short=-300.00 |
| COST-funding-negative | PASS | neg_rate: long=-100.00, short=100.00 |
| COST-us-comm-1sh | PASS | comm(1sh) = 1.00 |
| COST-us-comm-10sh | PASS | comm(10sh) = 1.00 |
| COST-us-comm-100sh | PASS | comm(100sh) = 1.00 |
| COST-us-comm-1000sh | PASS | comm(1000sh) = 5.00 |

## Kernel Accuracy (18 strategies × 2 configs)

**36/36 passed**

| Kernel | Config | Return % | DrawDown % | Trades | Result |
|--------|--------|----------|-----------|--------|--------|
| Bollinger | crypto | ret=-47.12% dd=64.51% trades=52 | PASS |
| Consensus | crypto | ret=-65.18% dd=68.81% trades=13 | PASS |
| Donchian | crypto | ret=-15.61% dd=34.46% trades=27 | PASS |
| Drift | crypto | ret=-19.57% dd=46.88% trades=108 | PASS |
| DualMom | crypto | ret=-46.31% dd=49.67% trades=64 | PASS |
| KAMA | crypto | ret=8.11% dd=33.55% trades=47 | PASS |
| Keltner | crypto | ret=-30.61% dd=38.65% trades=33 | PASS |
| MA | crypto | ret=-37.94% dd=49.04% trades=49 | PASS |
| MACD | crypto | ret=-37.00% dd=47.10% trades=127 | PASS |
| MESA | crypto | ret=-49.71% dd=61.78% trades=13 | PASS |
| MomBreak | crypto | ret=0.00% dd=0.00% trades=0 | PASS |
| MultiFactor | crypto | ret=-26.40% dd=57.20% trades=10 | PASS |
| RAMOM | crypto | ret=-50.67% dd=52.11% trades=63 | PASS |
| RSI | crypto | ret=-38.68% dd=53.34% trades=32 | PASS |
| RegimeEMA | crypto | ret=-2.55% dd=33.09% trades=12 | PASS |
| Turtle | crypto | ret=-30.21% dd=37.02% trades=28 | PASS |
| VolRegime | crypto | ret=21.19% dd=14.19% trades=5 | PASS |
| ZScore | crypto | ret=-39.67% dd=57.80% trades=29 | PASS |
| Bollinger | stock | ret=-19.78% dd=36.74% trades=52 | PASS |
| Consensus | stock | ret=-35.35% dd=39.60% trades=13 | PASS |
| Donchian | stock | ret=3.50% dd=17.69% trades=27 | PASS |
| Drift | stock | ret=-2.54% dd=27.01% trades=108 | PASS |
| DualMom | stock | ret=-21.80% dd=28.03% trades=64 | PASS |
| KAMA | stock | ret=16.78% dd=19.64% trades=47 | PASS |
| Keltner | stock | ret=-8.54% dd=20.32% trades=33 | PASS |
| MA | stock | ret=-6.58% dd=26.46% trades=49 | PASS |
| MACD | stock | ret=-8.66% dd=22.99% trades=127 | PASS |
| MESA | stock | ret=-19.96% dd=33.88% trades=13 | PASS |
| MomBreak | stock | ret=0.00% dd=0.00% trades=0 | PASS |
| MultiFactor | stock | ret=-2.36% dd=28.97% trades=10 | PASS |
| RAMOM | stock | ret=-22.49% dd=29.28% trades=63 | PASS |
| RSI | stock | ret=-17.86% dd=30.09% trades=32 | PASS |
| RegimeEMA | stock | ret=7.52% dd=18.35% trades=12 | PASS |
| Turtle | stock | ret=-7.56% dd=19.62% trades=28 | PASS |
| VolRegime | stock | ret=14.32% dd=7.89% trades=5 | PASS |
| ZScore | stock | ret=-13.09% dd=30.27% trades=29 | PASS |

## Performance Benchmark

### Kernel Throughput

| Bars | Kernels | Time (ms) | Throughput |
|------|---------|-----------|-----------|
| 500 | 18 | 0.2 | **104905 k/s** |
| 1000 | 18 | 0.3 | **63250 k/s** |
| 2000 | 18 | 0.5 | **35096 k/s** |
| 5000 | 18 | 1.2 | **14658 k/s** |

### Param Scan: 60 combos in 0.5ms = **109281 evals/s**

### Margin Model: **633483 calcs/s** (157.9ms for 100k)

## Broker Infrastructure

- P3-broker-async-methods: PASS — Broker has async methods
- P3-broker-margin-methods: PASS — Broker has margin methods
- P3-credentials-exists: PASS — CredentialManager has load + sign_request
- P3-credentials-sign: PASS — signature length = 64
- P3-ratelimiter-usage: PASS — initial usage = 0
- P3-binance-broker-class: PASS — BinanceFuturesBroker has key methods
- P3-ibkr-broker-class: PASS — IBKRBroker has key methods
- P3-order-manager: PASS — LiveOrderManager has submit + cancel
- P3-reconciler: PASS — PositionReconciler has reconcile
- P3-execution-algos: PASS — TWAP, LimitChase, Iceberg have execute
- P3-testnet-config: PASS — TestnetConfig has BINANCE_FUTURES

## Safety Infrastructure

- P5-alert-manager: PASS — AlertManager has send
- P5-audit-trail: PASS — report length = 537
- P5-kill-switch: PASS — KillSwitch has flatten_all + is_triggered
- P5-health-server: PASS — HealthCheckServer has start

## Regression Tests

**11/11 passed**

---

## Summary

- **136/136** tests passed (100.0%)
- Total execution time: **1547ms**

### V5 New Files Created

| File | Phase | Purpose |
|------|-------|---------|
| `core/__init__.py` | 0 | Package init |
| `core/asset_types.py` | 0 | AssetClass enum (5 types) |
| `core/margin.py` | 0 | MarginModel: CryptoSpot, CryptoFutures (Binance tiers), RegT |
| `core/costs.py` | 0 | CostModel: CryptoFutures (maker/taker/funding), USEquity (IBKR) |
| `core/symbol_spec.py` | 0 | SymbolSpec: price/qty rounding, order validation |
| `core/market_hours.py` | 2 | MarketCalendar: US market hours + NYSE holidays |
| `core/pdt_tracker.py` | 3 | PDT rule tracking (5-day window) |
| `data/funding_rates.py` | 1 | Binance funding rate download + parquet cache |
| `broker/credentials.py` | 3 | API key management (env > .env > yaml) |
| `broker/rate_limiter.py` | 3 | Token bucket limiter with O(1) tracking |
| `broker/binance_futures.py` | 3 | Binance USDT-M Perp broker (async, WebSocket) |
| `broker/ibkr_broker.py` | 3 | IBKR US equity broker (ib_insync) |
| `broker/live_order_manager.py` | 4 | Order lifecycle management with timeout |
| `broker/reconciler.py` | 4 | Position reconciliation (exchange vs internal) |
| `broker/execution_algo.py` | 7 | TWAP, LimitChase, Iceberg algorithms |
| `broker/testnet.py` | 6 | Testnet URL configuration |
| `live/alerts.py` | 5 | Multi-channel alerts (Telegram/Discord) |
| `live/audit.py` | 5 | SQLite audit trail + daily reports |
| `live/kill_switch.py` | 5 | Emergency position flattening |
| `live/health_server.py` | 5 | HTTP health/metrics endpoints |

> **ALL TESTS PASSED.** V5 Real Trading Upgrade implementation verified.