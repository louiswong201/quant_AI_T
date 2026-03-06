# Live Trading Integration Test Report

**Date**: 2026-03-05 23:32:43
**Total Time**: 0.6s
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
| BTC | RSI | 2x | 11 | $+35,940.42 | +2.1% | 1869 | No |
| ETH | Drift | 2x | 25 | $+3,653.70 | -24.8% | 5168 | No |
| AAPL | Donchian | 1x | 6 | $+5,071.41 | -0.2% | 5143 | No |
| MSFT | Bollinger | 1x | 15 | $+7,991.22 | +0.7% | 5225 | No |
| BTC | MA | 1x | 13 | $-4,337.12 | +1.2% | 4991 | No |
| ETH | ZScore | 3x | 26 | $+605,422.81 | +81.8% | 5365 | No |
| AAPL | MESA | 2x | 91 | $+13,009.51 | +6.9% | 4547 | No |
| MSFT | Drift | 1x | 24 | $+7,051.04 | +0.8% | 5863 | No |

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