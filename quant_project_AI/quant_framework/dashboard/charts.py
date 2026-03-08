"""Plotly chart builders for the live trading dashboard.

Professional dark-theme design inspired by Bloomberg Terminal / TradingView.
Pure functions — no Dash dependency.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design Tokens ─────────────────────────────────────────────────────

_C = {
    "bg":         "#060912",
    "card":       "#0c1220",
    "surface":    "#121a2b",
    "border":     "#243041",
    "text":       "#f8fbff",
    "muted":      "#91a3bb",
    "green":      "#22c55e",
    "green_dim":  "rgba(34,197,94,0.12)",
    "red":        "#f87171",
    "red_dim":    "rgba(248,113,113,0.12)",
    "blue":       "#4f8cff",
    "blue_dim":   "rgba(79,140,255,0.12)",
    "amber":      "#fbbf24",
    "purple":     "#9b8afb",
    "cyan":       "#22d3ee",
    "grid":       "rgba(36,48,65,0.65)",
    "candle_up":  "#22c55e",
    "candle_dn":  "#f87171",
}

_FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
_PF_CAP = 99.99


def _safe_pf(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return _PF_CAP
    if out < 0:
        return 0.0
    return min(out, _PF_CAP)

_LAYOUT_BASE = dict(
    font=dict(family=_FONT, color=_C["text"], size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=_C["card"],
    margin=dict(l=65, r=30, t=60, b=45),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor=_C["surface"], 
        font_color=_C["text"],
        bordercolor=_C["blue"], 
        font_size=11,
    ),
    legend=dict(
        orientation="h", 
        yanchor="top", 
        y=1.08, 
        xanchor="left", 
        x=0,
        bgcolor="rgba(0,0,0,0)", 
        font=dict(size=10, color=_C["muted"]),
        itemsizing="constant",
        tracegroupgap=10,
    ),
    xaxis=dict(
        gridcolor=_C["grid"], 
        zerolinecolor=_C["grid"],
        tickfont=dict(size=10, color=_C["muted"]),
        showgrid=True,
        gridwidth=0.5,
    ),
    yaxis=dict(
        gridcolor=_C["grid"], 
        zerolinecolor=_C["grid"],
        tickfont=dict(size=10, color=_C["muted"]),
        showgrid=True,
        gridwidth=0.5,
    ),
)


def _apply_base(fig: go.Figure, **overrides: Any) -> go.Figure:
    layout = {**_LAYOUT_BASE, **overrides}
    fig.update_layout(**layout)
    return fig


# ── Equity Curve ──────────────────────────────────────────────────────

def build_equity_chart(
    timestamps: List[Any],
    equity_values: List[float],
    cash_values: Optional[List[float]] = None,
) -> go.Figure:
    """Equity + cash + drawdown overlay."""
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.72, 0.28],
        shared_xaxes=True, vertical_spacing=0.04,
    )

    fig.add_trace(go.Scatter(
        x=timestamps, y=equity_values, name="Equity", mode="lines",
        line=dict(color=_C["blue"], width=2.5),
        fill="tozeroy", fillcolor=_C["blue_dim"],
    ), row=1, col=1)

    if cash_values:
        fig.add_trace(go.Scatter(
            x=timestamps, y=cash_values, name="Cash", mode="lines",
            line=dict(color=_C["amber"], width=1.5, dash="dot"),
        ), row=1, col=1)

    if len(equity_values) > 1:
        eq = np.array(equity_values, dtype=np.float64)
        peak = np.maximum.accumulate(eq)
        dd_pct = np.where(peak > 0, (peak - eq) / peak * 100, 0)
        fig.add_trace(go.Scatter(
            x=timestamps, y=-dd_pct, name="Drawdown",
            mode="lines", line=dict(color=_C["red"], width=1.2),
            fill="tozeroy", fillcolor=_C["red_dim"],
        ), row=2, col=1)

    _apply_base(fig, height=370,
                title=dict(text="EQUITY CURVE", font=dict(size=14, color=_C["muted"])))
    fig.update_yaxes(title_text="Value ($)", row=1, col=1,
                     gridcolor=_C["grid"], tickfont=dict(size=10, color=_C["muted"]))
    fig.update_yaxes(title_text="DD %", row=2, col=1,
                     gridcolor=_C["grid"], tickfont=dict(size=10, color=_C["muted"]))
    fig.update_xaxes(gridcolor=_C["grid"], tickfont=dict(size=10, color=_C["muted"]))
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
    return fig


# ── Candlestick ───────────────────────────────────────────────────────

def build_candlestick_chart(
    ohlcv_df: pd.DataFrame,
    trades_df: Optional[pd.DataFrame] = None,
    title: str = "Price",
    sl_tp_levels: Optional[Dict[str, Optional[float]]] = None,
    show_ma: bool = True,
) -> go.Figure:
    """Candlestick + volume + buy/sell markers + SL/TP lines + MA overlays."""
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.78, 0.22],
        shared_xaxes=True, vertical_spacing=0.04,
    )

    if ohlcv_df.empty:
        _apply_base(fig, height=420,
                    title=dict(text=title, font=dict(size=13, color=_C["muted"])))
        return fig

    # Normalize x-axis time and de-duplicate rows for stable rendering.
    ohlcv_df = ohlcv_df.copy()
    date_col = "date" if "date" in ohlcv_df.columns else ohlcv_df.columns[0]
    ohlcv_df[date_col] = pd.to_datetime(ohlcv_df[date_col], errors="coerce", utc=True)
    ohlcv_df = ohlcv_df.dropna(subset=[date_col, "open", "high", "low", "close"])
    ohlcv_df = ohlcv_df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    
    if ohlcv_df.empty:
        _apply_base(fig, height=420,
                    title=dict(text=title, font=dict(size=13, color=_C["muted"])))
        return fig

    # Create integer index for x-axis to avoid gaps
    ohlcv_df = ohlcv_df.reset_index(drop=True)
    x_data = list(range(len(ohlcv_df)))
    
    # Store original dates for hover text
    date_strings = ohlcv_df[date_col].dt.strftime("%Y-%m-%d %H:%M").tolist()

    fig.add_trace(go.Candlestick(
        x=x_data,
        open=ohlcv_df["open"], 
        high=ohlcv_df["high"],
        low=ohlcv_df["low"], 
        close=ohlcv_df["close"],
        name="OHLC",
        increasing=dict(line=dict(color=_C["candle_up"], width=1.2), fillcolor=_C["candle_up"]),
        decreasing=dict(line=dict(color=_C["candle_dn"], width=1.2), fillcolor=_C["candle_dn"]),
        showlegend=True,
        hovertext=date_strings,
        hoverinfo="text+y",
    ), row=1, col=1)

    if show_ma and "close" in ohlcv_df.columns and len(ohlcv_df) >= 20:
        close = ohlcv_df["close"].ffill()
        ma20 = close.rolling(20, min_periods=20).mean()
        ma50 = close.rolling(50, min_periods=50).mean()
        fig.add_trace(go.Scatter(
            x=x_data, y=ma20, name="MA20",
            mode="lines", line=dict(color=_C["amber"], width=2, dash="dot"),
            opacity=0.8,
            showlegend=True,
            hoverinfo="skip",
        ), row=1, col=1)
        if ma50.notna().any():
            fig.add_trace(go.Scatter(
                x=x_data, y=ma50, name="MA50",
                mode="lines", line=dict(color=_C["purple"], width=2, dash="dot"),
                opacity=0.8,
                showlegend=True,
                hoverinfo="skip",
            ), row=1, col=1)

    if sl_tp_levels:
        entry = sl_tp_levels.get("entry")
        sl = sl_tp_levels.get("sl")
        tp = sl_tp_levels.get("tp")
        if entry:
            fig.add_hline(
                y=entry, line=dict(color=_C["blue"], width=1.5, dash="dash"),
                annotation_text=f"Entry ${entry:,.2f}",
                annotation=dict(font_color=_C["blue"], font_size=10, bgcolor=_C["card"]),
                row=1, col=1,
            )
        if sl:
            fig.add_hline(
                y=sl, line=dict(color=_C["red"], width=1.5, dash="dash"),
                annotation_text=f"SL ${sl:,.2f}",
                annotation=dict(font_color=_C["red"], font_size=10, bgcolor=_C["card"]),
                row=1, col=1,
            )
        if tp:
            fig.add_hline(
                y=tp, line=dict(color=_C["green"], width=1.5, dash="dash"),
                annotation_text=f"TP ${tp:,.2f}",
                annotation=dict(font_color=_C["green"], font_size=10, bgcolor=_C["card"]),
                row=1, col=1,
            )

    if "volume" in ohlcv_df.columns:
        colors = [
            _C["candle_up"] if c >= o else _C["candle_dn"]
            for o, c in zip(ohlcv_df["open"], ohlcv_df["close"])
        ]
        fig.add_trace(go.Bar(
            x=x_data, 
            y=ohlcv_df["volume"],
            name="Volume", 
            marker_color=colors, 
            opacity=0.4,
            showlegend=False,
            hoverinfo="skip",
        ), row=2, col=1)

    if trades_df is not None and not trades_df.empty:
        ts_col = "timestamp" if "timestamp" in trades_df.columns else trades_df.columns[0]
        trades_df = trades_df.copy()
        trades_df[ts_col] = pd.to_datetime(trades_df[ts_col], errors="coerce", utc=True)
        trades_df = trades_df.dropna(subset=[ts_col])

        if "side" not in trades_df.columns:
            trades_df["side"] = ""
        trades_df["side"] = trades_df["side"].astype(str).str.lower()

        trade_indices = []
        for trade_time in trades_df[ts_col]:
            idx = (ohlcv_df[date_col] - trade_time).abs().idxmin()
            trade_indices.append(idx)

        trades_df["x_idx"] = trade_indices

        buys = trades_df[trades_df["side"] == "buy"]
        sells = trades_df[trades_df["side"] == "sell"]
        
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["x_idx"], y=buys["price"],
                mode="markers", name="BUY",
                marker=dict(
                    symbol="triangle-up", size=14,
                    color=_C["green"], 
                    line=dict(width=2, color=_C["card"]),
                ),
                showlegend=True,
            ), row=1, col=1)
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["x_idx"], y=sells["price"],
                mode="markers", name="SELL",
                marker=dict(
                    symbol="triangle-down", size=14,
                    color=_C["red"], 
                    line=dict(width=2, color=_C["card"]),
                ),
                showlegend=True,
            ), row=1, col=1)

    # Create custom tick labels showing dates at regular intervals
    n_ticks = min(12, len(ohlcv_df))
    tick_indices = np.linspace(0, len(ohlcv_df) - 1, n_ticks, dtype=int)
    tick_labels = [ohlcv_df[date_col].iloc[i].strftime("%m/%d %H:%M") for i in tick_indices]
    
    _apply_base(fig, height=420,
                title=dict(text=title.upper(), font=dict(size=13, color=_C["muted"], weight=600)))
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            showgrid=True,
            gridcolor=_C["grid"],
            tickmode="array",
            tickvals=tick_indices.tolist(),
            ticktext=tick_labels,
            tickangle=-45,
        ),
        xaxis2=dict(
            showgrid=True,
            gridcolor=_C["grid"],
            tickmode="array",
            tickvals=tick_indices.tolist(),
            ticktext=tick_labels,
            tickangle=-45,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9, color=_C["muted"]),
            itemsizing="constant",
            tracegroupgap=8,
        ),
    )
    fig.update_yaxes(
        title_text="Price", 
        row=1, col=1,
        gridcolor=_C["grid"], 
        tickfont=dict(size=10, color=_C["muted"]),
        side="right",
    )
    fig.update_yaxes(
        title_text="Vol", 
        row=2, col=1,
        gridcolor=_C["grid"], 
        tickfont=dict(size=10, color=_C["muted"]),
        side="right",
    )
    fig.update_xaxes(
        gridcolor=_C["grid"], 
        tickfont=dict(size=9, color=_C["muted"])
    )
    return fig


# ── Positions Table ───────────────────────────────────────────────────

def build_positions_table(
    positions: Dict[str, Union[int, float]],
    prices: Dict[str, float],
    total_equity: float = 0.0,
    position_details: Optional[List[Dict[str, Any]]] = None,
) -> go.Figure:
    """Rich positions table with entry price, unrealized PnL, PnL%."""
    cols = ["Symbol", "Side", "Qty", "Entry", "Price", "Unr. PnL", "PnL %", "Weight"]
    header_colors = _C["surface"]

    if position_details and len(position_details) > 0:
        syms = [d["symbol"] for d in position_details]
        sides = [d["side"] for d in position_details]
        qtys = [f'{abs(d["qty"]):.0f}' for d in position_details]
        entries = [f'${d["entry_price"]:,.2f}' if d["entry_price"] else "—" for d in position_details]
        cur_px = [f'${d["current_price"]:,.2f}' for d in position_details]
        pnls = [f'${d["unrealized_pnl"]:+,.2f}' for d in position_details]
        pnl_pcts = [f'{d["pnl_pct"]:+.2f}%' for d in position_details]
        weights = [f'{d["weight"]:.1f}%' for d in position_details]

        row_colors = []
        for d in position_details:
            if d["unrealized_pnl"] > 0:
                row_colors.append(_C["green_dim"])
            elif d["unrealized_pnl"] < 0:
                row_colors.append(_C["red_dim"])
            else:
                row_colors.append(_C["card"])

        pnl_font_colors = [
            _C["green"] if d["unrealized_pnl"] > 0 else
            (_C["red"] if d["unrealized_pnl"] < 0 else _C["muted"])
            for d in position_details
        ]

        side_colors = [_C["green"] if d["side"] == "LONG" else _C["red"] for d in position_details]

        cell_font_colors = [
            [_C["text"]] * len(syms),
            side_colors,
            [_C["text"]] * len(syms),
            [_C["muted"]] * len(syms),
            [_C["text"]] * len(syms),
            pnl_font_colors,
            pnl_font_colors,
            [_C["muted"]] * len(syms),
        ]
    elif not positions:
        syms = ["—"]
        sides = qtys = entries = cur_px = pnls = pnl_pcts = weights = ["—"]
        row_colors = [_C["card"]]
        cell_font_colors = [[_C["muted"]]] * 8
    else:
        syms, sides, qtys, entries, cur_px, pnls, pnl_pcts, weights = (
            [], [], [], [], [], [], [], []
        )
        row_colors = []
        cell_font_colors = [[] for _ in range(8)]
        for sym, sh in sorted(positions.items()):
            px = prices.get(sym, 0.0)
            val = sh * px
            w = (val / total_equity * 100) if total_equity > 0 else 0
            syms.append(sym)
            sides.append("LONG" if sh > 0 else "SHORT")
            qtys.append(f"{abs(sh):.0f}")
            entries.append("—")
            cur_px.append(f"${px:,.2f}")
            pnls.append("—")
            pnl_pcts.append("—")
            weights.append(f"{w:.1f}%")
            row_colors.append(_C["card"])
            for i in range(8):
                cell_font_colors[i].append(_C["text"])

    fig = go.Figure(data=[go.Table(
        columnwidth=[80, 60, 55, 80, 80, 90, 65, 60],
        header=dict(
            values=cols,
            fill_color=header_colors,
            font=dict(color=_C["muted"], size=11, family=_FONT),
            align="center", line=dict(color=_C["border"], width=1),
            height=32,
        ),
        cells=dict(
            values=[syms, sides, qtys, entries, cur_px, pnls, pnl_pcts, weights],
            fill_color=[row_colors],
            font=dict(color=cell_font_colors, size=12, family=_FONT),
            align="center", line=dict(color=_C["border"], width=0.5),
            height=30,
        ),
    )])
    fig.update_layout(
        title=dict(text="POSITIONS", font=dict(size=14, color=_C["muted"], family=_FONT)),
        height=280, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=5),
    )
    return fig


# ── Daily P&L Bar Chart ──────────────────────────────────────────────

def build_pnl_bar_chart(
    dates: List[Any],
    pnl_values: List[float],
) -> go.Figure:
    """Bar chart of daily P&L with cumulative line overlay."""
    colors = [_C["green"] if v >= 0 else _C["red"] for v in pnl_values]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=dates, y=pnl_values,
        marker_color=colors, name="Daily P&L",
        opacity=0.85,
    ), secondary_y=False)

    if len(pnl_values) > 1:
        cum = list(np.cumsum(pnl_values))
        fig.add_trace(go.Scatter(
            x=dates, y=cum, name="Cumulative",
            mode="lines", line=dict(color=_C["cyan"], width=2),
        ), secondary_y=True)

    _apply_base(fig, height=280,
                title=dict(text="DAILY P&L", font=dict(size=14, color=_C["muted"])))
    fig.update_yaxes(title_text="Daily ($)", secondary_y=False,
                     gridcolor=_C["grid"], tickfont=dict(size=10, color=_C["muted"]))
    fig.update_yaxes(title_text="Cumulative ($)", secondary_y=True,
                     gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10, color=_C["cyan"]))
    return fig


# ── Trade Log Table ───────────────────────────────────────────────────

def build_trade_log_table(trades_df: pd.DataFrame) -> go.Figure:
    """Color-coded trade log."""
    cols = ["Time", "Symbol", "Side", "Shares", "Price", "PnL", "Strategy"]

    if trades_df.empty:
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=cols, fill_color=_C["surface"],
                font=dict(color=_C["muted"], size=11, family=_FONT),
                align="center", line=dict(color=_C["border"], width=1), height=32,
            ),
            cells=dict(
                values=[["—"]] * len(cols), fill_color=_C["card"],
                font=dict(color=_C["muted"], size=12, family=_FONT),
                align="center", height=30,
            ),
        )])
        fig.update_layout(
            title=dict(text="TRADE LOG", font=dict(size=14, color=_C["muted"], family=_FONT)),
            height=340, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=5),
        )
        return fig

    df = trades_df.head(50).copy()
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]

    if "side" not in df.columns:
        df["side"] = ""
    if "pnl" not in df.columns:
        df["pnl"] = 0.0

    sides = df["side"].astype(str).str.lower().tolist()
    pnl_vals = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0).tolist()

    row_colors = []
    side_font = []
    pnl_font = []
    for s, p in zip(sides, pnl_vals):
        if s == "buy":
            row_colors.append("rgba(0,212,170,0.06)")
            side_font.append(_C["green"])
        else:
            row_colors.append("rgba(255,71,87,0.06)")
            side_font.append(_C["red"])
        if p > 0:
            pnl_font.append(_C["green"])
        elif p < 0:
            pnl_font.append(_C["red"])
        else:
            pnl_font.append(_C["muted"])

    n = len(df)
    cell_font_colors = [
        [_C["muted"]] * n,
        [_C["text"]] * n,
        side_font,
        [_C["text"]] * n,
        [_C["text"]] * n,
        pnl_font,
        [_C["muted"]] * n,
    ]

    time_strs = df[ts_col].astype(str).tolist()
    short_times = []
    for t in time_strs:
        if len(t) > 19:
            t = t[:19]
        short_times.append(t.replace("T", " "))

    fig = go.Figure(data=[go.Table(
        columnwidth=[130, 65, 50, 55, 75, 80, 90],
        header=dict(
            values=cols, fill_color=_C["surface"],
            font=dict(color=_C["muted"], size=11, family=_FONT),
            align="center", line=dict(color=_C["border"], width=1), height=32,
        ),
        cells=dict(
            values=[
                short_times,
                df.get("symbol", pd.Series(dtype=str)).tolist(),
                [s.upper() for s in sides],
                df.get("shares", pd.Series(dtype=float)).tolist(),
                [f"${p:,.2f}" for p in df.get("price", pd.Series(dtype=float))],
                [f"${p:+,.2f}" for p in pnl_vals],
                df.get("strategy", pd.Series(dtype=str)).tolist(),
            ],
            fill_color=[row_colors],
            font=dict(color=cell_font_colors, size=11, family=_FONT),
            align="center", line=dict(color=_C["border"], width=0.5), height=28,
        ),
    )])
    fig.update_layout(
        title=dict(text="TRADE LOG", font=dict(size=14, color=_C["muted"], family=_FONT)),
        height=340, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=5),
    )
    return fig


# ── Trade Stats Panel ─────────────────────────────────────────────────

def build_multi_tf_panel(tf_positions: Dict[str, Any]) -> go.Figure:
    """Per-symbol, per-interval position states with fused consensus result."""
    if not tf_positions:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="MULTI-TF SIGNALS", font=dict(size=14, color=_C["muted"], family=_FONT)),
            height=200, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=5),
            annotations=[dict(
                text="Single-timeframe mode", showarrow=False,
                font=dict(size=14, color=_C["muted"]),
                xref="paper", yref="paper", x=0.5, y=0.5,
            )],
        )
        return fig

    all_ivs: List[str] = []
    for info in tf_positions.values():
        for iv in info.get("intervals", []):
            if iv not in all_ivs:
                all_ivs.append(iv)

    symbols = list(tf_positions.keys())
    cols = ["Symbol"] + [f"{iv}" for iv in all_ivs] + ["FUSED"]
    col_widths = [80] + [90] * len(all_ivs) + [100]

    sym_vals: List[str] = []
    strat_vals: List[List[str]] = [[] for _ in all_ivs]
    cell_colors: List[List[str]] = [[] for _ in all_ivs]
    fused_vals: List[str] = []
    fused_colors: List[str] = []

    for sym in symbols:
        info = tf_positions[sym]
        positions = info.get("positions", {})
        strategies = info.get("strategies", {})
        fused = info.get("fused_position", 0)
        sym_vals.append(sym)

        agree_count = 0
        total = len(all_ivs)
        for i, iv in enumerate(all_ivs):
            pos = positions.get(iv, 0)
            strat = strategies.get(iv, "—")
            if pos > 0:
                label = f"LONG ({strat})"
                color = _C["green"]
                agree_count += 1
            elif pos < 0:
                label = f"SHORT ({strat})"
                color = _C["red"]
                agree_count += 1
            else:
                label = f"FLAT ({strat})"
                color = _C["muted"]
            strat_vals[i].append(label)
            cell_colors[i].append(color)

        if fused > 0:
            fused_vals.append(f"LONG ({agree_count}/{total})")
            fused_colors.append(_C["green"])
        elif fused < 0:
            fused_vals.append(f"SHORT ({agree_count}/{total})")
            fused_colors.append(_C["red"])
        else:
            longs = sum(1 for iv in all_ivs if positions.get(iv, 0) > 0)
            shorts = sum(1 for iv in all_ivs if positions.get(iv, 0) < 0)
            flats = total - longs - shorts
            detail = f"L{longs}/S{shorts}/F{flats}"
            fused_vals.append(f"NO SIGNAL ({detail})")
            fused_colors.append("#ff9800")

    n = len(symbols)
    all_font_colors = [[_C["text"]] * n] + cell_colors + [fused_colors]
    all_cell_fill = [[_C["card"]] * n] + [[_C["card"]] * n] * len(all_ivs)
    fused_bg = []
    for fc in fused_colors:
        if fc == _C["green"]:
            fused_bg.append("#0d2818")
        elif fc == _C["red"]:
            fused_bg.append("#2d0a0a")
        else:
            fused_bg.append("#2a1f00")
    all_cell_fill.append(fused_bg)

    fig = go.Figure(data=[go.Table(
        columnwidth=col_widths,
        header=dict(
            values=cols, fill_color=_C["surface"],
            font=dict(color=_C["muted"], size=11, family=_FONT),
            align="center", line=dict(color=_C["border"], width=1), height=30,
        ),
        cells=dict(
            values=[sym_vals] + strat_vals + [fused_vals],
            fill_color=all_cell_fill,
            font=dict(color=all_font_colors, size=12, family=_FONT),
            align="center", line=dict(color=_C["border"], width=0.5), height=32,
        ),
    )])
    fig.update_layout(
        title=dict(
            text="MULTI-TF SIGNAL FUSION  •  Consensus requires ≥2/3 timeframes to agree",
            font=dict(size=14, color=_C["muted"], family=_FONT),
        ),
        height=200, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=5),
    )
    return fig


def build_stats_panel(stats: Dict[str, Any]) -> go.Figure:
    """Key trade statistics displayed as a compact table."""
    labels = [
        "Win Rate", "Profit Factor", "Expectancy",
        "Avg Win", "Avg Loss", "Largest Win",
        "Largest Loss", "Win Streak", "Loss Streak",
    ]
    values = [
        f'{stats.get("win_rate", 0):.1f}%',
        f'{_safe_pf(stats.get("profit_factor", 0)):.2f}',
        f'${stats.get("expectancy", 0):+,.2f}',
        f'${stats.get("avg_win", 0):,.2f}',
        f'${stats.get("avg_loss", 0):,.2f}',
        f'${stats.get("largest_win", 0):,.2f}',
        f'${stats.get("largest_loss", 0):,.2f}',
        str(stats.get("win_streak", 0)),
        str(stats.get("loss_streak", 0)),
    ]
    val_colors = [
        _C["green"] if stats.get("win_rate", 0) >= 50 else _C["red"],
        _C["green"] if _safe_pf(stats.get("profit_factor", 0)) >= 1 else _C["red"],
        _C["green"] if stats.get("expectancy", 0) >= 0 else _C["red"],
        _C["green"], _C["red"], _C["green"],
        _C["red"], _C["green"], _C["red"],
    ]

    fig = go.Figure(data=[go.Table(
        columnwidth=[120, 100],
        header=dict(
            values=["Metric", "Value"], fill_color=_C["surface"],
            font=dict(color=_C["muted"], size=11, family=_FONT),
            align=["left", "right"], line=dict(color=_C["border"], width=1), height=30,
        ),
        cells=dict(
            values=[labels, values],
            fill_color=_C["card"],
            font=dict(
                color=[[_C["text"]] * len(labels), val_colors],
                size=12, family=_FONT,
            ),
            align=["left", "right"],
            line=dict(color=_C["border"], width=0.5), height=27,
        ),
    )])
    fig.update_layout(
        title=dict(text="TRADE STATISTICS", font=dict(size=14, color=_C["muted"], family=_FONT)),
        height=310, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=5),
    )
    return fig


def build_strategy_leaderboard(
    rows: List[Dict[str, Any]],
    rank_by: str = "score",
) -> go.Figure:
    """Sortable strategy leaderboard table."""
    if not rows:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="STRATEGY LEADERBOARD", font=dict(size=14, color=_C["muted"], family=_FONT)),
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=5),
            annotations=[dict(
                text="No strategy rows yet",
                showarrow=False,
                font=dict(size=13, color=_C["muted"]),
                xref="paper", yref="paper", x=0.5, y=0.5,
            )],
        )
        return fig

    rank_key_map = {
        "score": "score",
        "sharpe": "sharpe",
        "return": "return_pct",
        "calmar": "calmar",
        "stability": "stability",
    }
    rk = rank_key_map.get(rank_by, "score")
    sorted_rows = sorted(rows, key=lambda r: float(r.get(rk, 0.0)), reverse=True)[:12]

    rank = list(range(1, len(sorted_rows) + 1))
    strategy = [str(r.get("strategy", "")) for r in sorted_rows]
    symbol = [str(r.get("symbol", "")) for r in sorted_rows]
    direction = [str(r.get("direction", "FLAT")) for r in sorted_rows]
    ret = [f'{float(r.get("return_pct", 0.0)):+.2f}%' for r in sorted_rows]
    sharpe = [f'{float(r.get("sharpe", 0.0)):.2f}' for r in sorted_rows]
    dd = [f'{float(r.get("max_drawdown_pct", 0.0)):.2f}%' for r in sorted_rows]
    wr = [f'{float(r.get("win_rate", 0.0)):.1f}%' for r in sorted_rows]
    pf = [f'{_safe_pf(r.get("profit_factor", 0.0)):.2f}' for r in sorted_rows]
    score = [f'{float(r.get("score", 0.0)):.1f}' for r in sorted_rows]
    trades = [str(int(r.get("trades", 0))) for r in sorted_rows]

    dir_colors = []
    for d in direction:
        if d == "LONG":
            dir_colors.append(_C["green"])
        elif d == "SHORT":
            dir_colors.append(_C["red"])
        else:
            dir_colors.append(_C["muted"])
    score_colors = [_C["green"] if float(r.get("score", 0.0)) >= 0 else _C["red"] for r in sorted_rows]

    n = len(sorted_rows)
    font_colors = [
        [_C["text"]] * n, [_C["text"]] * n, [_C["muted"]] * n,
        dir_colors, [_C["text"]] * n, [_C["text"]] * n, [_C["text"]] * n,
        [_C["text"]] * n, [_C["text"]] * n, score_colors, [_C["muted"]] * n,
    ]

    fig = go.Figure(data=[go.Table(
        columnwidth=[35, 130, 60, 55, 65, 55, 60, 60, 55, 55, 45],
        header=dict(
            values=["#", "Strategy", "Symbol", "Dir", "Return", "Sharpe", "MaxDD", "Win", "PF", "Score", "N"],
            fill_color=_C["surface"],
            font=dict(color=_C["muted"], size=11, family=_FONT),
            align="center", line=dict(color=_C["border"], width=1), height=30,
        ),
        cells=dict(
            values=[rank, strategy, symbol, direction, ret, sharpe, dd, wr, pf, score, trades],
            fill_color=_C["card"],
            font=dict(color=font_colors, size=11, family=_FONT),
            align="center", line=dict(color=_C["border"], width=0.5), height=27,
        ),
    )])
    fig.update_layout(
        title=dict(text=f"STRATEGY LEADERBOARD • RANK BY {rank_by.upper()}",
                   font=dict(size=14, color=_C["muted"], family=_FONT)),
        height=330, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=42, b=5),
    )
    return fig


def build_strategy_scorecard(row: Optional[Dict[str, Any]]) -> go.Figure:
    """Single-strategy scorecard with composite and key diagnostics."""
    if not row:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="STRATEGY SCORECARD", font=dict(size=14, color=_C["muted"], family=_FONT)),
            height=330, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=42, b=5),
            annotations=[dict(
                text="Select a strategy",
                showarrow=False,
                font=dict(size=13, color=_C["muted"]),
                xref="paper", yref="paper", x=0.5, y=0.5,
            )],
        )
        return fig

    strategy = str(row.get("strategy", "Unknown"))
    symbol = str(row.get("symbol", ""))
    score = float(row.get("score", 0.0))
    ret = float(row.get("return_pct", 0.0))
    sharpe = float(row.get("sharpe", 0.0))
    dd = float(row.get("max_drawdown_pct", 0.0))
    wr = float(row.get("win_rate", 0.0))
    pf = _safe_pf(row.get("profit_factor", 0.0))
    stability = float(row.get("stability", 0.0))

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "font": {"size": 30, "color": _C["text"]}},
        gauge={
            "axis": {"range": [-100, 100], "tickcolor": _C["muted"]},
            "bar": {"color": _C["blue"]},
            "bgcolor": _C["card"],
            "bordercolor": _C["border"],
            "steps": [
                {"range": [-100, -20], "color": _C["red_dim"]},
                {"range": [-20, 20], "color": "rgba(139,149,165,0.15)"},
                {"range": [20, 100], "color": _C["green_dim"]},
            ],
        },
        domain={"x": [0.0, 0.45], "y": [0.08, 0.95]},
        title={"text": "Composite Score", "font": {"size": 12, "color": _C["muted"]}},
    ))

    labels = ["Return", "Sharpe", "MaxDD", "WinRate", "PF", "Stability"]
    values = [ret, sharpe, -dd, wr / 20.0, pf * 10.0, stability / 2.0]
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=[_C["blue"], _C["cyan"], _C["red"], _C["green"], _C["amber"], _C["purple"]],
        text=[
            f"{ret:+.2f}%",
            f"{sharpe:.2f}",
            f"{dd:.2f}%",
            f"{wr:.1f}%",
            f"{pf:.2f}",
            f"{stability:.1f}",
        ],
        textposition="outside",
        hoverinfo="skip",
        xaxis="x2",
        yaxis="y2",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=f"STRATEGY SCORECARD • {strategy} ({symbol})",
            font=dict(size=14, color=_C["muted"], family=_FONT),
        ),
        height=330,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_C["card"],
        margin=dict(l=10, r=10, t=42, b=10),
        xaxis2=dict(domain=[0.53, 0.98], showgrid=False, zeroline=False, showticklabels=False),
        yaxis2=dict(domain=[0.12, 0.94], showgrid=False, tickfont=dict(color=_C["muted"], size=11)),
        annotations=[
            dict(
                x=0.98, y=0.03, xref="paper", yref="paper",
                text=f"Trades: {int(row.get('trades', 0))} • Dir: {row.get('direction', 'FLAT')}",
                showarrow=False, font=dict(size=10, color=_C["muted"]),
            ),
        ],
    )
    return fig


def build_strategy_compare(
    rows: List[Dict[str, Any]],
    metric: str = "risk_return",
) -> go.Figure:
    """Compare top strategies across a chosen dimension."""
    if not rows:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="STRATEGY COMPARE", font=dict(size=14, color=_C["muted"], family=_FONT)),
            height=360, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=50, r=20, t=42, b=35),
        )
        return fig

    top = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)[:6]
    x = [f'{r.get("strategy","")} ({r.get("symbol","")})' for r in top]

    fig = go.Figure()
    if metric == "risk_return":
        fig.add_trace(go.Bar(name="Return %", x=x, y=[float(r.get("return_pct", 0.0)) for r in top],
                             marker_color=_C["green"]))
        fig.add_trace(go.Bar(name="MaxDD %", x=x, y=[-float(r.get("max_drawdown_pct", 0.0)) for r in top],
                             marker_color=_C["red"]))
    elif metric == "quality":
        fig.add_trace(go.Bar(name="Sharpe", x=x, y=[float(r.get("sharpe", 0.0)) for r in top],
                             marker_color=_C["cyan"]))
        fig.add_trace(go.Bar(name="Profit Factor", x=x, y=[_safe_pf(r.get("profit_factor", 0.0)) for r in top],
                             marker_color=_C["amber"]))
        fig.add_trace(go.Bar(name="Win Rate / 10", x=x, y=[float(r.get("win_rate", 0.0)) / 10.0 for r in top],
                             marker_color=_C["blue"]))
    else:
        fig.add_trace(go.Bar(name="Composite Score", x=x, y=[float(r.get("score", 0.0)) for r in top],
                             marker_color=_C["purple"]))
        fig.add_trace(go.Bar(name="Stability", x=x, y=[float(r.get("stability", 0.0)) for r in top],
                             marker_color=_C["blue"]))

    _apply_base(fig, height=360,
                title=dict(text=f"STRATEGY COMPARE • {metric.upper()}",
                           font=dict(size=14, color=_C["muted"])))
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-20)
    return fig
