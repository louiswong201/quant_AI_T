#!/usr/bin/env python3
"""Generate static HTML preview of the redesigned dashboard charts.

Run: python examples/dashboard_preview.py
Opens: reports/dashboard_preview.html
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from quant_framework.dashboard.charts import (
    build_equity_chart,
    build_candlestick_chart,
    build_positions_table,
    build_pnl_bar_chart,
    build_trade_log_table,
    build_stats_panel,
    _C,
)

np.random.seed(42)
N = 200
dates = pd.date_range("2025-01-01", periods=N, freq="h")

price = 100.0
prices = [price]
for _ in range(N - 1):
    price *= np.exp(np.random.normal(0.0002, 0.008))
    prices.append(price)
prices = np.array(prices)

ohlcv = pd.DataFrame({
    "date": dates,
    "open": prices * (1 + np.random.uniform(-0.003, 0.003, N)),
    "high": prices * (1 + np.random.uniform(0.001, 0.012, N)),
    "low":  prices * (1 - np.random.uniform(0.001, 0.012, N)),
    "close": prices,
    "volume": np.random.randint(1000, 50000, N),
})

eq_start = 100000
returns = np.random.normal(0.0003, 0.004, N)
equity = eq_start * np.cumprod(1 + returns)
cash = equity * np.random.uniform(0.3, 0.5, N)

trade_times = np.random.choice(dates, 20, replace=False)
trade_times.sort()
trades_df = pd.DataFrame({
    "timestamp": trade_times,
    "symbol": ["BTCUSDT"] * 20,
    "side": np.random.choice(["buy", "sell"], 20),
    "shares": np.random.uniform(0.01, 0.5, 20).round(4),
    "price": np.random.uniform(95, 110, 20).round(2),
    "pnl": np.random.normal(50, 200, 20).round(2),
    "strategy": ["MA"] * 20,
})

pnl_dates = pd.date_range("2025-01-01", periods=30, freq="D")
pnl_values = np.random.normal(100, 500, 30).tolist()

position_details = [
    {"symbol": "BTCUSDT", "qty": 0.15, "entry_price": 97500.0, "current_price": 101200.0,
     "market_value": 15180.0, "unrealized_pnl": 555.0, "pnl_pct": 3.79, "weight": 14.2, "side": "LONG"},
    {"symbol": "ETHUSDT", "qty": -2.0, "entry_price": 3800.0, "current_price": 3750.0,
     "market_value": -7500.0, "unrealized_pnl": 100.0, "pnl_pct": 1.32, "weight": -7.0, "side": "SHORT"},
    {"symbol": "AAPL", "qty": 50, "entry_price": 185.0, "current_price": 182.5,
     "market_value": 9125.0, "unrealized_pnl": -125.0, "pnl_pct": -1.35, "weight": 8.5, "side": "LONG"},
]

stats = {
    "win_rate": 58.3, "profit_factor": 1.72, "expectancy": 42.5,
    "avg_win": 185.0, "avg_loss": -107.5, "largest_win": 890.0,
    "largest_loss": -520.0, "win_streak": 6, "loss_streak": 3,
    "total_trades": 48, "winning_trades": 28, "losing_trades": 20,
    "total_pnl": 2040.0, "total_commission": 96.0,
}

sl_tp = {"entry": 97500.0, "sl": 78000.0, "tp": 117000.0}

fig_eq = build_equity_chart(dates.tolist(), equity.tolist(), cash.tolist())
fig_candle = build_candlestick_chart(ohlcv, trades_df=trades_df, title="BTCUSDT Price", sl_tp_levels=sl_tp)
fig_pos = build_positions_table(
    {"BTCUSDT": 0.15, "ETHUSDT": -2.0, "AAPL": 50},
    {"BTCUSDT": 101200, "ETHUSDT": 3750, "AAPL": 182.5},
    total_equity=107000, position_details=position_details,
)
fig_pnl = build_pnl_bar_chart(pnl_dates.strftime("%Y-%m-%d").tolist(), pnl_values)
fig_trades = build_trade_log_table(trades_df)
fig_stats = build_stats_panel(stats)

BG = _C["bg"]
CARD = _C["card"]
BORDER = _C["border"]
TEXT = _C["text"]
MUTED = _C["muted"]
GREEN = _C["green"]
BLUE = _C["blue"]
AMBER = _C["amber"]
RED = _C["red"]
CYAN = _C["cyan"]

out_dir = Path(__file__).resolve().parent.parent / "reports"
out_dir.mkdir(exist_ok=True)

html_parts = []
html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Terminal Preview</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: {BG}; font-family: Inter, -apple-system, sans-serif; color: {TEXT}; }}
.container {{ max-width: 1600px; margin: 0 auto; padding: 20px 24px; }}
.header {{
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 16px; border-bottom: 1px solid {BORDER}; margin-bottom: 20px;
}}
.header-left {{ display: flex; align-items: center; gap: 12px; }}
.logo {{
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, {BLUE}, {CYAN});
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; font-weight: 700; color: #fff;
}}
.header h3 {{ font-size: 1.1rem; letter-spacing: 1px; margin: 0; }}
.header .sub {{ font-size: 0.72rem; color: {MUTED}; }}
.status {{ display: flex; align-items: center; gap: 6px; font-size: 0.8rem; font-weight: 600; }}
.dot {{
    width: 8px; height: 8px; border-radius: 50%; background: {GREEN};
    animation: pulse 2s infinite;
}}
@keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }} }}
.metrics {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin-bottom: 20px;
}}
.metric-card {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 14px 18px; transition: all 0.2s ease;
}}
.metric-card:hover {{ border-color: {BLUE}; box-shadow: 0 0 20px rgba(59,130,246,0.08); }}
.metric-label {{
    font-size: 0.7rem; font-weight: 500; color: {MUTED};
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
}}
.metric-value {{ font-size: 1.25rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
.row {{ display: grid; gap: 16px; margin-bottom: 16px; }}
.row-60-40 {{ grid-template-columns: 3fr 2fr; }}
.row-40-60 {{ grid-template-columns: 2fr 3fr; }}
.chart-card {{
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 12px; overflow: hidden;
}}
.footer {{
    text-align: center; padding: 12px; color: {MUTED}; font-size: 0.7rem;
    border-top: 1px solid {BORDER};
}}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <div class="header-left">
    <div class="logo">Q</div>
    <div>
      <h3>TRADING TERMINAL</h3>
      <span class="sub">Paper Trading • Dashboard Preview</span>
    </div>
  </div>
  <div class="status">
    <span class="dot"></span>
    <span style="color:{GREEN}">LIVE</span>
  </div>
</div>

<div class="metrics">
  <div class="metric-card"><div class="metric-label">Cash</div><div class="metric-value" style="color:{AMBER}">$43,285</div></div>
  <div class="metric-card"><div class="metric-label">Equity</div><div class="metric-value" style="color:{BLUE}">$107,842</div></div>
  <div class="metric-card"><div class="metric-label">Total Return</div><div class="metric-value" style="color:{GREEN}">+7.84%</div></div>
  <div class="metric-card"><div class="metric-label">Daily P&L</div><div class="metric-value" style="color:{GREEN}">+$1,250</div></div>
  <div class="metric-card"><div class="metric-label">Max Drawdown</div><div class="metric-value" style="color:{AMBER}">-3.21%</div></div>
  <div class="metric-card"><div class="metric-label">Win Rate</div><div class="metric-value" style="color:{GREEN}">58.3%</div></div>
  <div class="metric-card"><div class="metric-label">Profit Factor</div><div class="metric-value" style="color:{GREEN}">1.72x</div></div>
  <div class="metric-card"><div class="metric-label">Trades</div><div class="metric-value" style="color:{CYAN}">48</div></div>
</div>
""")

for label, fig, cols in [
    ("Row 1", [fig_eq, fig_pos], "row-60-40"),
    ("Row 2", [fig_candle, fig_stats], "row-60-40"),
    ("Row 3", [fig_pnl, fig_trades], "row-40-60"),
]:
    html_parts.append(f'<div class="row {cols}">')
    for f in fig:
        inner = f.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
        html_parts.append(f'<div class="chart-card">{inner}</div>')
    html_parts.append("</div>")

html_parts.append("""
<div class="footer">Quant Framework • Paper Trading Dashboard • Preview Mode</div>
</div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</body>
</html>""")

out_path = out_dir / "dashboard_preview.html"
out_path.write_text("\n".join(html_parts), encoding="utf-8")
print(f"Dashboard preview saved to: {out_path}")
print(f"Open in browser: file://{out_path}")
