"""
Download 1y market data for AAPL/GOOGL/TSLA and run strategy backtests.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from statistics import mean, median

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_framework import BacktestConfig, BacktestEngine, DataManager
from quant_framework.analysis.performance import PerformanceAnalyzer
from quant_framework.strategy.ma_strategy import MovingAverageStrategy
from quant_framework.strategy.rsi_strategy import RSIStrategy
from quant_framework.strategy.macd_strategy import MACDStrategy


def fetch_symbol(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"no data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    need = ["date", "open", "high", "low", "close", "volume"]
    out = df[need].copy()
    out["date"] = pd.to_datetime(out["date"])
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna().sort_values("date").reset_index(drop=True)
    return out


def main() -> None:
    end_d = date.today()
    start_d = end_d - timedelta(days=365)
    start = start_d.isoformat()
    end = end_d.isoformat()

    symbols = ["AAPL", "GOOGL", "TSLA"]
    data_manager = DataManager(data_dir="data", use_parquet=False, fast_io=False)
    analyzer = PerformanceAnalyzer(risk_free_rate=0.03)
    engine = BacktestEngine(data_manager, config=BacktestConfig.conservative())

    print(f"Downloading data from {start} to {end} ...")
    for s in symbols:
        data = fetch_symbol(s, start, end)
        data_manager.save_data(s, data)
        print(f"  {s}: {len(data)} bars saved")

    strategy_builders = {
        "MA(5,20)": lambda: MovingAverageStrategy(short_window=5, long_window=20, initial_capital=1_000_000),
        "RSI(14)": lambda: RSIStrategy(rsi_period=14, oversold=30, overbought=70, initial_capital=1_000_000),
        "MACD(12,26,9)": lambda: MACDStrategy(fast_period=12, slow_period=26, signal_period=9, initial_capital=1_000_000),
    }

    rows = []
    for sym in symbols:
        for strat_name, builder in strategy_builders.items():
            strat = builder()
            result = engine.run(strat, sym, start, end)
            perf = analyzer.analyze(result["portfolio_values"], result["daily_returns"], result["initial_capital"])
            trades = analyzer.analyze_trades(result["trades"])
            rows.append(
                {
                    "symbol": sym,
                    "strategy": strat_name,
                    "total_return_pct": perf["total_return"] * 100.0,
                    "annual_return_pct": perf["annual_return"] * 100.0,
                    "max_drawdown_pct": perf["max_drawdown"] * 100.0,
                    "sharpe": perf["sharpe_ratio"],
                    "final_value": perf["final_value"],
                    "trades": trades["total_trades"],
                    "win_rate_pct": trades["win_rate"] * 100.0 if trades["total_trades"] > 0 else 0.0,
                }
            )

    result_df = pd.DataFrame(rows).sort_values(["symbol", "total_return_pct"], ascending=[True, False])
    out_csv = "output/real_market_backtest_results.csv"
    os.makedirs("output", exist_ok=True)
    result_df.to_csv(out_csv, index=False)

    print("\n=== Backtest Results (AAPL/GOOGL/TSLA, last 1y) ===")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\nSaved detailed table: {out_csv}")

    framework_summary = {
        "avg_total_return_pct": mean(result_df["total_return_pct"]),
        "median_total_return_pct": median(result_df["total_return_pct"]),
        "avg_max_drawdown_pct": mean(result_df["max_drawdown_pct"]),
        "avg_sharpe": mean(result_df["sharpe"]),
        "best_case_total_return_pct": float(result_df["total_return_pct"].max()),
        "worst_case_total_return_pct": float(result_df["total_return_pct"].min()),
    }
    print("\n=== Framework Performance Summary ===")
    for k, v in framework_summary.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()

