# Data Directory

This folder stores OHLCV data for backtesting and paper trading.

## Structure

```
data/
├── 1m/   # 1-minute bars
├── 5m/   # 5-minute bars
├── 15m/  # 15-minute bars
├── 1h/   # 1-hour bars
├── 4h/   # 4-hour bars
└── 1d/   # 1-day bars
```

Each subdirectory contains CSV files named `{SYMBOL}_{interval}.csv`, e.g.:
- `BTC_1h.csv`, `ETH_1h.csv` (crypto)
- `AAPL_1d.csv`, `SPY_1d.csv` (stocks)

## Download Data

### Multi-timeframe (crypto + stocks)

```bash
python examples/download_multi_tf_data.py
```

Downloads 1h, 4h, 1d data for BTC, ETH, SOL, BNB (crypto) and SPY, QQQ, AAPL, etc. (stocks).

### 5-minute intraday

```bash
python examples/download_5m_data.py
```

## CSV Format

| Column | Type |
|--------|------|
| date   | datetime (ISO or parseable) |
| open   | float |
| high   | float |
| low    | float |
| close  | float |
| volume | float |
