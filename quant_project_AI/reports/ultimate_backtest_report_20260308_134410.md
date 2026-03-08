# Ultimate Backtest Report

Generated: 2026-03-08 13:44:10
Total runtime: 2194.8s (36.6 min)

## Scan Summary

- Total parameter combinations evaluated: 132,442
- Total entries from Phase 1: 4,968
- Viable single-TF strategies: 332
- Viable multi-TF strategies: 24212
- Final recommendations: 75

## Quality Filters Applied

| Filter | Threshold |
|--------|-----------|
| Sharpe Ratio | > 0.3 |
| DSR p-value | < 0.15 |
| MC Survival | > 50% |
| OOS Return | > 0% |
| Max Drawdown (multi-TF) | < 60% |
| CPCV Validation | Yes (for daily TF) |

## Top Recommendations for Live Trading

| Rank | Symbol | Strategy | Leverage | TF | OOS Ret% | Sharpe | DSR p | MC>0% | Score |
|------|--------|----------|----------|----|----------|--------|-------|-------|-------|
| 1 | NVDA | MultiFactor | 1x | 1h | +36.6% | 2.05 | 0.073 | 98% | 2.0 |
| 2 | KO | Consensus | 1x | 1d | +178.5% | 2.04 | 0.001 | 100% | 2.0 ✓ |
| 3 | DOGE | RAMOM | 1x | 4h | +38.7% | 2.04 | 0.057 | 68% | 2.0 |
| 4 | JNJ | RAMOM | 3x | 4h | +29.6% | 2.02 | 0.082 | 100% | 2.0 ✓ |
| 5 | GOOGL | KAMA | 2x | 4h | +45.4% | 2.01 | 0.063 | 96% | 2.0 |
| 6 | UNH | Bollinger | 3x | 4h | +57.1% | 1.99 | 0.029 | 100% | 2.0 |
| 7 | MRK | DualMom | 3x | 1d | +380.6% | 1.97 | 0.000 | 100% | 2.0 ✓ |
| 8 | BNB | MomBreak | 1x | 4h | +33.8% | 1.96 | 0.000 | 100% | 2.0 ✓ |
| 9 | DOGE | Keltner | 1x | 1h | +33.4% | 1.93 | 0.000 | 76% | 1.9 |
| 10 | GOOGL | Drift | 2x | 4h | +51.4% | 1.92 | 0.001 | 66% | 1.9 |
| 11 | JNJ | RAMOM | 1x | 4h | +16.4% | 1.92 | 0.001 | 100% | 1.9 ✓ |
| 12 | JNJ | DualMom | 1x | 1h | +14.3% | 1.90 | 0.000 | 100% | 1.9 ✓ |
| 13 | PG | MESA | 1x | 4h | +17.0% | 2.03 | 0.118 | 100% | 1.9 ✓ |
| 14 | AAPL | MACD | 3x | 1h | +31.1% | 1.89 | 0.000 | 100% | 1.9 |
| 15 | AVAX | KAMA | 1x | 4h | +44.8% | 1.88 | 0.000 | 100% | 1.9 |
| 16 | BTC | KAMA | 2x | 1d | +171.6% | 1.88 | 0.000 | 72% | 1.9 ✓ |
| 17 | BTC | MA | 2x | 1d | +140.2% | 1.87 | 0.000 | 100% | 1.9 ✓ |
| 18 | JPM | VolRegime | 2x | 1d | +308.7% | 1.87 | 0.000 | 100% | 1.9 ✓ |
| 19 | JNJ | VolRegime | 1x | 1d | +173.4% | 1.86 | 0.000 | 100% | 1.9 ✓ |
| 20 | META | RAMOM | 2x | 4h | +36.6% | 1.85 | 0.000 | 66% | 1.9 |
| 21 | HD | MACD | 1x | 4h | +24.7% | 1.84 | 0.000 | 96% | 1.8 |
| 22 | ABBV | MomBreak | 2x | 1d | +156.3% | 1.84 | 0.000 | 100% | 1.8 ✓ |
| 23 | ETH | Donchian | 2x | 1h | +36.5% | 1.83 | 0.000 | 66% | 1.8 |
| 24 | MRK | DualMom | 2x | 1d | +357.4% | 1.83 | 0.000 | 100% | 1.8 ✓ |
| 25 | DOT | Bollinger | 3x | 1d | +109.5% | 1.81 | 0.000 | 100% | 1.8 ✓ |
| 26 | ABBV | MultiFactor | 3x | 1d | +147.9% | 1.81 | 0.000 | 100% | 1.8 ✓ |
| 27 | GOOGL | KAMA | 3x | 4h | +40.3% | 1.80 | 0.000 | 96% | 1.8 |
| 28 | AVAX | Donchian | 2x | 1h | +56.3% | 1.80 | 0.000 | 98% | 1.8 |
| 29 | PG | VolRegime | 3x | 1h | +15.0% | 1.84 | 0.000 | 100% | 1.8 ✓ |
| 30 | JNJ | ZScore | 3x | 1d | +207.1% | 1.78 | 0.000 | 100% | 1.8 ✓ |

## Per-Symbol Best Strategy

### NVDA
- Strategy: **MultiFactor**
- Parameters: `[10, 13, 29, 0.6, 0.4]`
- Leverage: 1x | Interval: 1h
- OOS Return: +36.6% | Sharpe: 2.05
- DSR p-value: 0.0728 | MC Survival: 98%
- CPCV Validated: No

### KO
- Strategy: **Consensus**
- Parameters: `[20, 200, 7, 40, 35, 80, 3]`
- Leverage: 1x | Interval: 1d
- OOS Return: +178.5% | Sharpe: 2.04
- DSR p-value: 0.0013 | MC Survival: 100%
- CPCV Validated: Yes

### DOGE
- Strategy: **RAMOM**
- Parameters: `[23, 3, 3.0, 1.0]`
- Leverage: 1x | Interval: 4h
- OOS Return: +38.7% | Sharpe: 2.04
- DSR p-value: 0.0574 | MC Survival: 68%
- CPCV Validated: No

### JNJ
- Strategy: **RAMOM**
- Parameters: `[58, 33, 3.5, 0.0]`
- Leverage: 3x | Interval: 4h
- OOS Return: +29.6% | Sharpe: 2.02
- DSR p-value: 0.0816 | MC Survival: 100%
- CPCV Validated: Yes

### GOOGL
- Strategy: **KAMA**
- Parameters: `[25, 4, 50, 2.5, 10]`
- Leverage: 2x | Interval: 4h
- OOS Return: +45.4% | Sharpe: 2.01
- DSR p-value: 0.0630 | MC Survival: 96%
- CPCV Validated: No

### UNH
- Strategy: **Bollinger**
- Parameters: `[63, 2.0]`
- Leverage: 3x | Interval: 4h
- OOS Return: +57.1% | Sharpe: 1.99
- DSR p-value: 0.0295 | MC Survival: 100%
- CPCV Validated: No

### MRK
- Strategy: **DualMom**
- Parameters: `[80, 250]`
- Leverage: 3x | Interval: 1d
- OOS Return: +380.6% | Sharpe: 1.97
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: Yes

### BNB
- Strategy: **MomBreak**
- Parameters: `[10, 0.05, 20, 4.0]`
- Leverage: 1x | Interval: 4h
- OOS Return: +33.8% | Sharpe: 1.96
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: Yes

### PG
- Strategy: **MESA**
- Parameters: `[0.9, 0.1]`
- Leverage: 1x | Interval: 4h
- OOS Return: +17.0% | Sharpe: 2.03
- DSR p-value: 0.1180 | MC Survival: 100%
- CPCV Validated: Yes

### AAPL
- Strategy: **MACD**
- Parameters: `[29, 60, 41]`
- Leverage: 3x | Interval: 1h
- OOS Return: +31.1% | Sharpe: 1.89
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: No

### AVAX
- Strategy: **KAMA**
- Parameters: `[8, 2, 20, 1.5, 20]`
- Leverage: 1x | Interval: 4h
- OOS Return: +44.8% | Sharpe: 1.88
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: No

### BTC
- Strategy: **KAMA**
- Parameters: `[12, 3, 40, 1.0, 14]`
- Leverage: 2x | Interval: 1d
- OOS Return: +171.6% | Sharpe: 1.88
- DSR p-value: 0.0000 | MC Survival: 72%
- CPCV Validated: Yes

### JPM
- Strategy: **VolRegime**
- Parameters: `[14, 0.035, 20, 100, 35, 75]`
- Leverage: 2x | Interval: 1d
- OOS Return: +308.7% | Sharpe: 1.87
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: Yes

### META
- Strategy: **RAMOM**
- Parameters: `[23, 48, 1.0, 0.3]`
- Leverage: 2x | Interval: 4h
- OOS Return: +36.6% | Sharpe: 1.85
- DSR p-value: 0.0000 | MC Survival: 66%
- CPCV Validated: No

### HD
- Strategy: **MACD**
- Parameters: `[14, 57, 17]`
- Leverage: 1x | Interval: 4h
- OOS Return: +24.7% | Sharpe: 1.84
- DSR p-value: 0.0000 | MC Survival: 96%
- CPCV Validated: No

### ABBV
- Strategy: **MomBreak**
- Parameters: `[5, 0.12, 20, 4.0]`
- Leverage: 2x | Interval: 1d
- OOS Return: +156.3% | Sharpe: 1.84
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: Yes

### ETH
- Strategy: **Donchian**
- Parameters: `[117, 14, 1.5]`
- Leverage: 2x | Interval: 1h
- OOS Return: +36.5% | Sharpe: 1.83
- DSR p-value: 0.0000 | MC Survival: 66%
- CPCV Validated: No

### DOT
- Strategy: **Bollinger**
- Parameters: `[11, 0.5]`
- Leverage: 3x | Interval: 1d
- OOS Return: +109.5% | Sharpe: 1.81
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: Yes

### SOL
- Strategy: **KAMA**
- Parameters: `[10, 3, 25, 1.0, 20]`
- Leverage: 1x | Interval: 1d
- OOS Return: +82.3% | Sharpe: 1.76
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: Yes

### CVX
- Strategy: **MA**
- Parameters: `[69, 285]`
- Leverage: 1x | Interval: 1d
- OOS Return: +268.1% | Sharpe: 1.72
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: Yes

### TSLA
- Strategy: **VolRegime**
- Parameters: `[10, 0.02, 20, 100, 35, 65]`
- Leverage: 3x | Interval: 1h
- OOS Return: +54.8% | Sharpe: 1.72
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: No

### XRP
- Strategy: **MomBreak**
- Parameters: `[120, 0.01, 20, 3.0]`
- Leverage: 1x | Interval: 1h
- OOS Return: +25.6% | Sharpe: 1.69
- DSR p-value: 0.0000 | MC Survival: 98%
- CPCV Validated: Yes

### MSFT
- Strategy: **RSI**
- Parameters: `[54, 47, 76]`
- Leverage: 1x | Interval: 1d
- OOS Return: +113.6% | Sharpe: 1.64
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: Yes

### ADA
- Strategy: **Donchian**
- Parameters: `[99, 20, 3.5]`
- Leverage: 1x | Interval: 1h
- OOS Return: +25.0% | Sharpe: 1.63
- DSR p-value: 0.0000 | MC Survival: 92%
- CPCV Validated: No

### BRK-B
- Strategy: **MomBreak**
- Parameters: `[5, 0.12, 14, 2.5]`
- Leverage: 3x | Interval: 1d
- OOS Return: +177.8% | Sharpe: 1.59
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: Yes

### QQQ
- Strategy: **MultiFactor**
- Parameters: `[5, 21, 37, 0.5, 0.45]`
- Leverage: 3x | Interval: 1h
- OOS Return: +28.1% | Sharpe: 1.62
- DSR p-value: 0.0000 | MC Survival: 100%
- CPCV Validated: No

## How to Use Results for Live Trading

```bash
# Load the best parameters for live trading
python run_live_trading.py --config reports/live_trading_config.json
```

The `live_trading_config.json` contains all recommended strategy-parameter
combinations ranked by composite score (Sharpe × DSR × MC survival).
