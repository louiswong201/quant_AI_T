# Live Trading Integration Test Report

**Date**: 2026-03-08 23:04:34
**Total Time**: 13.5s
**Tests**: 8 passed, 0 failed

---

## Test Configuration

| # | Symbol | Strategy | Params | Leverage |
|---|--------|----------|--------|----------|
| 1 | BTC | RSI | (5, 15, 75) | 2.0x |
| 2 | ETH | Drift | (10, 0.55, 11) | 2.0x |
| 3 | AAPL | Donchian | (10, 10, 3.0) | 1.0x |
| 4 | MSFT | Bollinger | (10, 2.0) | 1.0x |
| 5 | BTC | MA | (10, 30) | 1.0x |
| 6 | ETH | ZScore | (20, 2.0, 3.0, 0.5) | 3.0x |
| 7 | AAPL | MESA | (10, 0.5, 0.05) | 2.0x |
| 8 | MSFT | Drift | (10, 0.55, 11) | 1.0x |

## Results

| Symbol | Strategy | Lev | Trades | PnL | Return | Bars/s | CB Tripped |
|--------|----------|-----|--------|-----|--------|--------|------------|
| BTC | RSI | 2x | 3 | $+76,270.59 | +2.1% | 106 | No |
| ETH | Drift | 2x | 13 | $-6,511.70 | -30.5% | 231 | No |
| AAPL | Donchian | 1x | 5 | $-6,045.54 | -1.0% | 176 | No |
| MSFT | Bollinger | 1x | 6 | $+4,915.37 | +0.7% | 297 | No |
| BTC | MA | 1x | 7 | $-2,081.68 | -0.3% | 357 | No |
| ETH | ZScore | 3x | 7 | $+1,276,965.84 | +81.8% | 270 | No |
| AAPL | MESA | 2x | 46 | $+5,734.77 | +3.8% | 114 | No |
| MSFT | Drift | 1x | 13 | $+6,733.24 | +0.8% | 1023 | No |

## Pipeline Verification

| Component | Status |
|-----------|--------|
| SyntheticFeedManager (BarEvent emission) | PASS |
| KernelAdapter (signal generation) | PASS |
| PaperBroker (order execution) | PASS |
| RiskManagedBroker (risk checks) | PASS |
| TradeJournal (persistence) | PASS |
| CircuitBreaker (safety) | PASS |
| TradingRunner (async event loop) | PASS |

---

> Async live trading integration test complete.