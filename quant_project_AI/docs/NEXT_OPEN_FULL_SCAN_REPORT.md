# 17 Strategy Next-Open Full Scan Report

> 1,040,788 backtests | 4.5s | 230,397 combos/sec
> Data: AAPL, GOOGL, TSLA, BTC | Cost: 5bps slippage + 15bps commission
> Execution: **Next-Open** (signal @ close[i] → fill @ open[i+1])
> Split: 60% train / 40% test

---

## 1. Performance Diagnosis: Why Was the Previous Scan Slow?

| Item | BacktestEngine (Previous) | Numba @njit (This Scan) |
|:---|:---|:---|
| Execution | Python per-bar loop + DataFrame.iloc | Compiled machine code |
| Indicators | Recalculated per combo via `calculate_all()` | Precomputed once, O(1) lookup |
| KAMA Speed | ~5.7s per combo (243 combos = 1,392s) | ~0.005ms per combo |
| Total Throughput | ~0.2 combos/sec | **230,397 combos/sec** |
| Speedup | — | **~1,151,984x faster** |

Root cause: `BacktestEngine.run()` iterates each bar in Python, calling
`DataFrame.iloc[]` for every price lookup. `VectorizedIndicators.calculate_all()`
recomputes ALL indicators (including KAMA's O(n*period) Python loop) for every
parameter combination. Numba compiles the entire backtest+indicator pipeline to
native machine code, eliminating all Python overhead.

---

## 2. Overall Ranking (Average Full-Period Return)

| Rank | Strategy | Avg Return | Avg Train | Avg Test | Overfit Risk |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | **RSI** | +342.9% | +201.7% | +32.8% | HIGH |
| 2 | **MACD** | +158.5% | +129.6% | +4.0% | HIGH |
| 3 | **MA Crossover** | +123.9% | +148.9% | +6.8% | HIGH |
| 4 | **DriftRegime** | +120.6% | +251.5% | -29.8% | HIGH |
| 5 | **RAMOM** | +116.9% | +91.1% | +11.7% | HIGH |
| 6 | **MomBreakout** | +95.4% | +95.8% | +1.5% | HIGH |
| 7 | **MESA** | +91.9% | +81.5% | +4.5% | HIGH |
| 8 | **MultiFactor** | +83.0% | +124.9% | -5.6% | HIGH |
| 9 | **Turtle** | +81.6% | +70.1% | +14.6% | HIGH |
| 10 | **VolRegime** | +80.8% | +111.9% | -8.8% | HIGH |
| 11 | **DonchianATR** | +77.9% | +53.0% | +9.2% | HIGH |
| 12 | **Keltner** | +71.3% | +50.3% | +8.8% | HIGH |
| 13 | **RegimeEMA** | +59.4% | +82.9% | -11.1% | HIGH |
| 14 | **KAMA** | +59.2% | +40.2% | +8.9% | HIGH |
| 15 | **ConnorsRSI2** | +34.4% | +41.4% | +0.5% | HIGH |
| 16 | **ZScoreRev** | +14.3% | +32.8% | -13.0% | HIGH |
| 17 | **Bollinger** | +14.1% | +33.2% | -4.1% | HIGH |

---

## 3. Per-Symbol Best Parameters

### AAPL

| Rank | Strategy | Full Return | Train | Test | Best Params |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | RSI | +101.5% | +80.5% | +9.1% | period=7, os=30, ob=85 |
| 2 | MACD | +73.7% | +60.4% | -1.0% | fast=24, slow=48, sig=36 |
| 3 | MomBreakout | +62.9% | +44.8% | +6.4% | high_p=40, prox=0.03, atr_p=14, trail=1.5 |
| 4 | MultiFactor | +57.4% | +73.4% | -12.7% | rsi=9, mom=5, vol=5, lt=0.55, st=0.35 |
| 5 | DriftRegime | +38.5% | +49.5% | -21.2% | lb=105, thr=0.52, hold=25 |
| 6 | MA Crossover | +33.7% | +66.1% | -8.0% | short=139, long=178 |
| 7 | Bollinger | +31.9% | +20.7% | +13.9% | period=8, std=2.0 |
| 8 | DonchianATR | +30.8% | +12.5% | +9.9% | entry=5, atr_p=10, mult=4.0 |
| 9 | RAMOM | +28.3% | +31.8% | +3.1% | mom=40, vol=30, ez=3.0, xz=0.8 |
| 10 | Turtle | +23.4% | +31.8% | +26.2% | entry=5, exit=30, atr_p=14, stop=3.5 |
| 11 | KAMA | +17.8% | +26.6% | +11.3% | er=20, fast=5, slow=30, atr_m=1.5, atr_p=10 |
| 12 | ZScoreRev | +15.3% | +30.1% | -10.7% | lb=110, entry_z=1.0, exit_z=0.75, stop_z=3.0 |
| 13 | ConnorsRSI2 | +12.6% | +22.1% | -1.4% | rsi=4, maT=200, maE=7, os=20.0, ob=80.0 |
| 14 | Keltner | +1.5% | +24.7% | -18.5% | ema=20, atr_p=14, mult=2.5 |
| 15 | VolRegime | -2.8% | +24.8% | -16.2% | atr=10, vt=0.025, ms=15, ml=20, os=20, ob=70 |
| 16 | RegimeEMA | -11.4% | +18.1% | -8.2% | atr=10, vt=0.02, fast=15, slow=60, trend=30 |
| 17 | MESA | -24.5% | +32.2% | -46.0% | fast_lim=0.7, slow_lim=0.08 |

### GOOGL

| Rank | Strategy | Full Return | Train | Test | Best Params |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | RSI | +279.2% | +120.1% | +57.3% | period=9, os=35, ob=85 |
| 2 | MESA | +181.2% | +23.9% | +108.6% | fast_lim=0.3, slow_lim=0.03 |
| 3 | MA Crossover | +181.0% | +59.1% | +70.8% | short=10, long=12 |
| 4 | MultiFactor | +131.4% | +82.3% | +19.7% | rsi=14, mom=15, vol=10, lt=0.55, st=0.2 |
| 5 | DonchianATR | +129.1% | +38.9% | +46.7% | entry=8, atr_p=14, mult=2.5 |
| 6 | MomBreakout | +119.9% | +48.4% | +38.6% | high_p=20, prox=0.02, atr_p=10, trail=2.0 |
| 7 | KAMA | +112.4% | +37.9% | +46.2% | er=30, fast=5, slow=50, atr_m=1.0, atr_p=10 |
| 8 | Turtle | +112.1% | +32.7% | +31.8% | entry=5, exit=18, atr_p=10, stop=1.5 |
| 9 | RAMOM | +94.2% | +24.9% | +44.5% | mom=85, vol=5, ez=2.0, xz=0.2 |
| 10 | Keltner | +61.9% | +7.8% | +33.4% | ema=30, atr_p=30, mult=2.0 |
| 11 | DriftRegime | +54.9% | +123.4% | -41.4% | lb=10, thr=0.52, hold=13 |
| 12 | MACD | +36.0% | +45.6% | -6.6% | fast=4, slow=28, sig=10 |
| 13 | VolRegime | +28.5% | +50.7% | -8.9% | atr=14, vt=0.025, ms=20, ml=30, os=35, ob=80 |
| 14 | ConnorsRSI2 | +12.8% | +12.1% | -1.5% | rsi=3, maT=200, maE=5, os=15.0, ob=80.0 |
| 15 | Bollinger | -9.0% | +42.4% | -33.5% | period=107, std=1.0 |
| 16 | ZScoreRev | -22.7% | +47.0% | -45.4% | lb=65, entry_z=1.5, exit_z=0.25, stop_z=3.0 |
| 17 | RegimeEMA | -38.0% | +23.2% | -41.9% | atr=20, vt=0.03, fast=3, slow=60, trend=30 |

### TSLA

| Rank | Strategy | Full Return | Train | Test | Best Params |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | RSI | +746.5% | +232.7% | +84.5% | period=28, os=35, ob=75 |
| 2 | MACD | +286.2% | +179.0% | +40.6% | fast=12, slow=14, sig=12 |
| 3 | DriftRegime | +147.7% | +554.3% | -56.4% | lb=20, thr=0.55, hold=21 |
| 4 | MA Crossover | +122.5% | +188.5% | +2.7% | short=54, long=65 |
| 5 | DonchianATR | +113.5% | +50.4% | +13.7% | entry=38, atr_p=7, mult=2.5 |
| 6 | MomBreakout | +92.3% | +89.8% | -10.4% | high_p=60, prox=0.02, atr_p=10, trail=3.5 |
| 7 | Keltner | +90.7% | +39.6% | +20.8% | ema=20, atr_p=10, mult=2.0 |
| 8 | MultiFactor | +80.2% | +91.8% | +24.4% | rsi=14, mom=5, vol=5, lt=0.5, st=0.3 |
| 9 | Turtle | +75.7% | +84.7% | -1.3% | entry=20, exit=33, atr_p=10, stop=1.0 |
| 10 | RAMOM | +74.4% | +122.4% | -29.3% | mom=65, vol=5, ez=3.5, xz=0.8 |
| 11 | RegimeEMA | +63.3% | +71.4% | -7.7% | atr=10, vt=0.01, fast=5, slow=20, trend=50 |
| 12 | ZScoreRev | +46.8% | +42.9% | -1.8% | lb=90, entry_z=2.5, exit_z=0.0, stop_z=3.0 |
| 13 | ConnorsRSI2 | +42.4% | +85.5% | -9.8% | rsi=2, maT=100, maE=3, os=20.0, ob=80.0 |
| 14 | KAMA | +36.9% | +37.9% | -28.3% | er=15, fast=2, slow=20, atr_m=1.5, atr_p=10 |
| 15 | VolRegime | +34.7% | +71.1% | -6.1% | atr=10, vt=0.01, ms=3, ml=80, os=30, ob=75 |
| 16 | Bollinger | +32.3% | +65.7% | +6.1% | period=47, std=0.5 |
| 17 | MESA | +16.0% | +83.6% | -47.9% | fast_lim=0.7, slow_lim=0.1 |

### BTC

| Rank | Strategy | Full Return | Train | Test | Best Params |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | RAMOM | +270.7% | +185.1% | +28.4% | mom=30, vol=20, ez=3.5, xz=0.8 |
| 2 | VolRegime | +262.5% | +300.9% | -4.2% | atr=20, vt=0.035, ms=5, ml=40, os=35, ob=75 |
| 3 | RSI | +244.2% | +373.6% | -19.7% | period=21, os=30, ob=85 |
| 4 | DriftRegime | +241.3% | +279.0% | -0.3% | lb=10, thr=0.52, hold=19 |
| 5 | MACD | +237.9% | +233.5% | -17.0% | fast=10, slow=50, sig=20 |
| 6 | RegimeEMA | +223.7% | +219.0% | +13.2% | atr=10, vt=0.02, fast=8, slow=20, trend=100 |
| 7 | MESA | +194.7% | +186.5% | +3.1% | fast_lim=0.7, slow_lim=0.08 |
| 8 | MA Crossover | +158.3% | +282.1% | -38.5% | short=43, long=46 |
| 9 | Keltner | +131.2% | +128.9% | -0.5% | ema=25, atr_p=7, mult=2.5 |
| 10 | Turtle | +115.2% | +131.1% | +1.7% | entry=14, exit=24, atr_p=10, stop=1.0 |
| 11 | MomBreakout | +106.4% | +200.0% | -28.6% | high_p=20, prox=0.08, atr_p=20, trail=2.5 |
| 12 | ConnorsRSI2 | +69.9% | +46.1% | +14.6% | rsi=3, maT=50, maE=10, os=20.0, ob=80.0 |
| 13 | KAMA | +69.8% | +58.4% | +6.3% | er=5, fast=5, slow=50, atr_m=1.0, atr_p=10 |
| 14 | MultiFactor | +62.9% | +252.3% | -53.9% | rsi=21, mom=75, vol=40, lt=0.5, st=0.4 |
| 15 | DonchianATR | +38.1% | +110.0% | -33.6% | entry=14, atr_p=7, mult=2.5 |
| 16 | ZScoreRev | +17.6% | +11.0% | +6.0% | lb=20, entry_z=3.0, exit_z=0.5, stop_z=3.0 |
| 17 | Bollinger | +1.1% | +4.0% | -2.7% | period=8, std=2.5 |

---

## 4. Overfitting Analysis

Strategies where test return is negative or <30% of train return are HIGH risk:

| Strategy | Avg Train | Avg Test | Test/Train Ratio | Verdict |
|:---|:---:|:---:|:---:|:---|
| MA Crossover | +148.9% | +6.8% | 0.05 | OVERFITTING |
| RSI | +201.7% | +32.8% | 0.16 | OVERFITTING |
| MACD | +129.6% | +4.0% | 0.03 | OVERFITTING |
| DriftRegime | +251.5% | -29.8% | -0.12 | OVERFITTING |
| RAMOM | +91.1% | +11.7% | 0.13 | OVERFITTING |
| Turtle | +70.1% | +14.6% | 0.21 | MODERATE |
| Bollinger | +33.2% | -4.1% | -0.12 | OVERFITTING |
| Keltner | +50.3% | +8.8% | 0.18 | OVERFITTING |
| MultiFactor | +124.9% | -5.6% | -0.05 | OVERFITTING |
| VolRegime | +111.9% | -8.8% | -0.08 | OVERFITTING |
| ConnorsRSI2 | +41.4% | +0.5% | 0.01 | OVERFITTING |
| MESA | +81.5% | +4.5% | 0.05 | OVERFITTING |
| KAMA | +40.2% | +8.9% | 0.22 | MODERATE |
| DonchianATR | +53.0% | +9.2% | 0.17 | OVERFITTING |
| ZScoreRev | +32.8% | -13.0% | -0.40 | OVERFITTING |
| MomBreakout | +95.8% | +1.5% | 0.02 | OVERFITTING |
| RegimeEMA | +82.9% | -11.1% | -0.13 | OVERFITTING |

---

## 5. Performance Benchmark

| Strategy | Total Time | Combos (4 sym) | Speed (combos/sec) |
|:---|:---:|:---:|:---:|
| MACD | 1.08s | 651,700 | 601,285 |
| MultiFactor | 1.14s | 116,640 | 102,351 |
| MA Crossover | 0.07s | 78,804 | 1,177,584 |
| RSI | 0.04s | 44,576 | 1,189,392 |
| RAMOM | 0.35s | 35,420 | 100,851 |
| VolRegime | 0.22s | 33,408 | 151,670 |
| Turtle | 0.59s | 28,800 | 48,670 |
| DriftRegime | 0.19s | 12,096 | 64,569 |
| ConnorsRSI2 | 0.04s | 9,600 | 260,008 |
| ZScoreRev | 0.01s | 7,040 | 656,657 |
| RegimeEMA | 0.03s | 6,960 | 252,411 |
| KAMA | 0.04s | 5,184 | 145,028 |
| Keltner | 0.01s | 3,220 | 274,533 |
| MomBreakout | 0.05s | 3,024 | 63,581 |
| DonchianATR | 0.03s | 2,800 | 81,193 |
| Bollinger | 0.00s | 1,372 | 454,154 |
| MESA | 0.00s | 144 | 45,029 |

**Total: 1,040,788 backtests in 4.5s = 230,397 combos/sec**

---

## 6. Methodology Notes

1. **Next-Open Execution**: All entries/exits use the NEXT bar's open price.
   This eliminates look-ahead bias inherent in close-price execution.
2. **Cost Model**: 5bps slippage (buy 1.0005x, sell 0.9995x) + 15bps commission
3. **Train/Test Split**: First 60% of bars for parameter selection, last 40% for validation
4. **Long+Short**: Most strategies trade both long and short (except MA, RSI, MomBreakout)
5. **No Position Sizing**: All trades are 100% of capital (full in/full out)
6. **Numba JIT**: All 17 kernels compiled to machine code with `@njit(cache=True)`
