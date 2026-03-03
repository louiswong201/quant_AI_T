"""Dash web application for live paper-trading monitoring.

Professional terminal-style dashboard focused on strategy clarity:
  - Strategy leaderboard
  - Strategy scorecard
  - Strategy comparison panel
  - Market context (candles/positions/equity/pnl/log)
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List

import pandas as pd

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from .charts import (
    build_candlestick_chart,
    build_equity_chart,
    build_multi_tf_panel,
    build_pnl_bar_chart,
    build_positions_table,
    build_stats_panel,
    build_strategy_compare,
    build_strategy_leaderboard,
    build_strategy_scorecard,
    build_trade_log_table,
)

_BG = "#08090e"
_CARD = "#0f1117"
_SURFACE = "#161b26"
_BORDER = "#1e2433"
_TEXT = "#f8fafc"
_MUTED = "#94a3b8"
_GREEN = "#10b981"
_RED = "#ef4444"
_BLUE = "#3b82f6"
_AMBER = "#f59e0b"
_PURPLE = "#8b5cf6"
_CYAN = "#06b6d4"

_FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
_HEAD_SCRIPTS = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
]

_CUSTOM_CSS = f"""
body {{
    background-color: {_BG} !important;
    font-family: {_FONT} !important;
    color: {_TEXT};
    margin: 0;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

/* Shell Card */
.shell-card {{
    background: {_CARD};
    border: 1px solid {_BORDER};
    border-radius: 12px;
    padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}}

/* Metric Cards */
.metric-card {{
    background: linear-gradient(135deg, {_CARD} 0%, {_SURFACE} 100%);
    border: 1px solid {_BORDER};
    border-radius: 12px;
    padding: 14px 16px;
    min-width: 130px;
    transition: all 0.2s ease;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}}
.metric-card:hover {{
    border-color: {_BLUE};
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
    transform: translateY(-2px);
}}
.metric-label {{
    font-size: 0.7rem;
    font-weight: 600;
    color: {_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}}
.metric-value {{
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0;
    font-variant-numeric: tabular-nums;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}}

/* Chart Cards */
.chart-card {{
    background: {_CARD};
    border: 1px solid {_BORDER};
    border-radius: 12px;
    padding: 14px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}}

/* Dropdown Styles - CRITICAL FIX */
.dash-dropdown {{
    font-family: {_FONT} !important;
}}
.dash-dropdown .Select-control {{
    background-color: {_SURFACE} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 8px !important;
    min-height: 38px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}}
.dash-dropdown .Select-control:hover {{
    border-color: {_BLUE} !important;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
}}
.dash-dropdown .Select-value-label {{
    color: {_TEXT} !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
}}
.dash-dropdown .Select-placeholder {{
    color: {_MUTED} !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
}}
.dash-dropdown .Select-input {{
    color: {_TEXT} !important;
    height: auto !important;
}}
.dash-dropdown .Select-input > input {{
    color: {_TEXT} !important;
    background: transparent !important;
    padding: 8px 0 !important;
}}
.dash-dropdown .Select-arrow-zone {{
    padding-right: 10px !important;
}}
.dash-dropdown .Select-arrow {{
    border-color: {_MUTED} transparent transparent !important;
    border-width: 6px 6px 3px !important;
}}
.dash-dropdown.is-open .Select-arrow {{
    border-color: transparent transparent {_MUTED} !important;
    border-width: 3px 6px 6px !important;
}}
.dash-dropdown .Select-menu-outer {{
    background-color: {_CARD} !important;
    border: 1px solid {_BLUE} !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
    margin-top: 4px !important;
    z-index: 9999 !important;
}}
.dash-dropdown .Select-menu {{
    max-height: 320px !important;
    overflow-y: auto !important;
}}
.dash-dropdown .Select-option {{
    background-color: transparent !important;
    color: {_TEXT} !important;
    padding: 12px 14px !important;
    font-size: 0.85rem !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}}
.dash-dropdown .Select-option:hover,
.dash-dropdown .Select-option.is-focused {{
    background-color: {_SURFACE} !important;
    color: {_TEXT} !important;
}}
.dash-dropdown .Select-option.is-selected {{
    background-color: {_BLUE}30 !important;
    color: {_TEXT} !important;
    font-weight: 600 !important;
}}
.dash-dropdown .Select-option.is-selected:hover {{
    background-color: {_BLUE}40 !important;
}}
/* Fix for multi-value (if used) */
.dash-dropdown .Select-value {{
    background-color: {_BLUE}20 !important;
    border: 1px solid {_BLUE}40 !important;
    color: {_TEXT} !important;
}}
.dash-dropdown .Select-value-icon {{
    border-right-color: {_BLUE}40 !important;
}}
.dash-dropdown .Select-value-icon:hover {{
    background-color: {_RED}20 !important;
    color: {_RED} !important;
}}

/* Status Indicator */
.status-dot {{
    display: inline-block;
    width: 9px; 
    height: 9px;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s infinite;
    box-shadow: 0 0 8px currentColor;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.6; transform: scale(0.95); }}
}}

/* Scrollbar Styling */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}
::-webkit-scrollbar-track {{
    background: {_CARD};
}}
::-webkit-scrollbar-thumb {{
    background: {_BORDER};
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {_MUTED};
}}
"""


def _metric_card(label: str, value_id: str) -> html.Div:
    return html.Div(
        className="metric-card",
        children=[
            html.Div(label, className="metric-label"),
            html.Div(id=value_id, className="metric-value", style={"color": _TEXT}),
        ],
    )


def _rank_rows(rows: List[Dict[str, Any]], rank_by: str) -> List[Dict[str, Any]]:
    key_map = {
        "score": "score",
        "sharpe": "sharpe",
        "return": "return_pct",
        "calmar": "calmar",
        "stability": "stability",
    }
    key = key_map.get(rank_by, "score")
    return sorted(rows, key=lambda r: float(r.get(key, 0.0)), reverse=True)


def create_app(
    get_state: Callable[[], Dict[str, Any]],
    get_equity_curve: Callable[[], pd.DataFrame],
    get_trades: Callable[[int], pd.DataFrame],
    get_window: Callable,
    get_daily_pnl: Callable[[int], pd.DataFrame],
    symbols: List[str],
    refresh_ms: int = 2000,
) -> "dash.Dash":
    if not DASH_AVAILABLE:
        raise ImportError("Dashboard requires: pip install dash plotly dash-bootstrap-components")

    app = dash.Dash(
        __name__,
        external_stylesheets=_HEAD_SCRIPTS,
        title="Trading Terminal",
        update_title=None,
        suppress_callback_exceptions=True,
    )

    symbol_options = [{"label": s, "value": s} for s in symbols]
    default_symbol = symbols[0] if symbols else ""

    app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>""" + _CUSTOM_CSS + """</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>"""

    app.layout = html.Div(
        style={"maxWidth": "1680px", "margin": "0 auto", "padding": "14px 20px", "backgroundColor": _BG},
        children=[
            dcc.Interval(id="interval", interval=refresh_ms, n_intervals=0),
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "marginBottom": "16px",
                    "paddingBottom": "14px",
                    "borderBottom": f"1px solid {_BORDER}",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "14px"},
                        children=[
                            html.Div(
                                style={
                                    "width": "40px",
                                    "height": "40px",
                                    "borderRadius": "10px",
                                    "background": f"linear-gradient(135deg, {_BLUE}, {_CYAN})",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "fontSize": "19px",
                                    "fontWeight": "700",
                                    "color": "#fff",
                                    "boxShadow": f"0 4px 12px {_BLUE}40",
                                },
                                children="Q",
                            ),
                            html.Div(
                                children=[
                                    html.H3(
                                        "TRADING TERMINAL PRO",
                                        style={
                                            "margin": "0",
                                            "fontSize": "1.1rem",
                                            "fontWeight": "700",
                                            "color": _TEXT,
                                            "letterSpacing": "0.8px",
                                            "textShadow": "0 1px 2px rgba(0,0,0,0.3)",
                                        },
                                    ),
                                    html.Span(
                                        id="header-subtitle", 
                                        style={
                                            "fontSize": "0.75rem", 
                                            "color": _MUTED,
                                            "fontWeight": "500",
                                        }
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "16px"},
                        children=[
                            html.Div(
                                id="status-indicator", 
                                style={
                                    "display": "flex", 
                                    "alignItems": "center", 
                                    "fontSize": "0.82rem",
                                    "fontWeight": "600",
                                    "padding": "6px 12px",
                                    "background": f"{_SURFACE}",
                                    "border": f"1px solid {_BORDER}",
                                    "borderRadius": "8px",
                                }
                            ),
                            html.Div(
                                style={"width": "200px"},
                                children=[
                                    dcc.Dropdown(
                                        id="symbol-select",
                                        options=symbol_options,
                                        value=default_symbol,
                                        clearable=False,
                                        className="dash-dropdown",
                                        style={"fontSize": "0.85rem"},
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(id="config-bar", style={"display": "flex", "gap": "8px", "marginBottom": "12px", "flexWrap": "wrap"}),
            html.Div(
                id="metrics-row",
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(130px, 1fr))", "gap": "10px", "marginBottom": "12px"},
                children=[
                    _metric_card("Cash", "m-cash"),
                    _metric_card("Equity", "m-equity"),
                    _metric_card("Total Return", "m-return"),
                    _metric_card("Daily P&L", "m-daily-pnl"),
                    _metric_card("Max Drawdown", "m-drawdown"),
                    _metric_card("Win Rate", "m-winrate"),
                    _metric_card("Profit Factor", "m-pf"),
                    _metric_card("Trades", "m-trades"),
                ],
            ),
            html.Div(
                className="shell-card",
                style={
                    "marginBottom": "14px", 
                    "display": "grid", 
                    "gridTemplateColumns": "200px 280px 240px 1fr", 
                    "gap": "12px",
                    "alignItems": "center",
                },
                children=[
                    dcc.Dropdown(
                        id="rank-basis",
                        options=[
                            {"label": "🏆 Rank: Score", "value": "score"},
                            {"label": "📊 Rank: Sharpe", "value": "sharpe"},
                            {"label": "💰 Rank: Return", "value": "return"},
                            {"label": "📈 Rank: Calmar", "value": "calmar"},
                            {"label": "⚖️ Rank: Stability", "value": "stability"},
                        ],
                        value="score",
                        clearable=False,
                        className="dash-dropdown",
                    ),
                    dcc.Dropdown(
                        id="scorecard-strategy", 
                        options=[], 
                        value="", 
                        clearable=False, 
                        className="dash-dropdown",
                        placeholder="Select Strategy..."
                    ),
                    dcc.Dropdown(
                        id="compare-metric",
                        options=[
                            {"label": "📉 Compare: Risk/Return", "value": "risk_return"},
                            {"label": "✨ Compare: Quality", "value": "quality"},
                            {"label": "🎯 Compare: Composite", "value": "composite"},
                        ],
                        value="risk_return",
                        clearable=False,
                        className="dash-dropdown",
                    ),
                    html.Div(
                        id="footer-ticks", 
                        style={
                            "display": "flex", 
                            "alignItems": "center", 
                            "justifyContent": "flex-end", 
                            "color": _MUTED, 
                            "fontSize": "0.75rem",
                            "fontWeight": "500",
                        }
                    ),
                ],
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "2fr 1.25fr", "gap": "12px", "marginBottom": "12px"},
                children=[
                    html.Div(className="chart-card", children=[dcc.Graph(id="strategy-leaderboard", config={"displayModeBar": False}, style={"height": "340px"})]),
                    html.Div(className="chart-card", children=[dcc.Graph(id="strategy-scorecard", config={"displayModeBar": False}, style={"height": "340px"})]),
                ],
            ),
            html.Div(className="chart-card", style={"marginBottom": "12px"}, children=[dcc.Graph(id="strategy-compare", config={"displayModeBar": False}, style={"height": "360px"})]),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "3fr 2fr", "gap": "12px", "marginBottom": "12px"},
                children=[
                    html.Div(
                        className="chart-card",
                        children=[
                            html.Div(
                                style={"display": "flex", "justifyContent": "flex-end", "padding": "2px 8px 0 0"},
                                children=[
                                    html.Div(
                                        style={"width": "100px"},
                                        children=[
                                            dcc.Dropdown(
                                                id="interval-select",
                                                options=[{"label": "Auto", "value": "auto"}],
                                                value="auto",
                                                clearable=False,
                                                className="dash-dropdown",
                                                style={"fontSize": "0.78rem"},
                                            )
                                        ],
                                    )
                                ],
                            ),
                            dcc.Graph(id="candle-chart", config={"displayModeBar": False}, style={"height": "410px"}),
                        ],
                    ),
                    html.Div(className="chart-card", children=[dcc.Graph(id="positions-table", config={"displayModeBar": False}, style={"height": "410px"})]),
                ],
            ),
            html.Div(
                id="multi-tf-row",
                style={"marginBottom": "12px"},
                children=[html.Div(className="chart-card", children=[dcc.Graph(id="multi-tf-panel", config={"displayModeBar": False}, style={"height": "200px"})])],
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "2fr 1.5fr 1.5fr", "gap": "12px", "marginBottom": "12px"},
                children=[
                    html.Div(className="chart-card", children=[dcc.Graph(id="equity-chart", config={"displayModeBar": False}, style={"height": "340px"})]),
                    html.Div(className="chart-card", children=[dcc.Graph(id="stats-panel", config={"displayModeBar": False}, style={"height": "340px"})]),
                    html.Div(className="chart-card", children=[dcc.Graph(id="pnl-chart", config={"displayModeBar": False}, style={"height": "340px"})]),
                ],
            ),
            html.Div(className="chart-card", children=[dcc.Graph(id="trade-log", config={"displayModeBar": False}, style={"height": "340px"})]),
        ],
    )

    def _badge(label: str, value: str, color: str) -> html.Span:
        return html.Span(
            f"{label}: {value}",
            style={
                "display": "inline-block",
                "padding": "4px 11px",
                "borderRadius": "6px",
                "fontSize": "0.74rem",
                "fontWeight": "600",
                "backgroundColor": f"{color}18",
                "color": color,
                "border": f"1px solid {color}40",
            },
        )

    @app.callback(
        [
            Output("m-cash", "children"),
            Output("m-cash", "style"),
            Output("m-equity", "children"),
            Output("m-equity", "style"),
            Output("m-return", "children"),
            Output("m-return", "style"),
            Output("m-daily-pnl", "children"),
            Output("m-daily-pnl", "style"),
            Output("m-drawdown", "children"),
            Output("m-drawdown", "style"),
            Output("m-winrate", "children"),
            Output("m-winrate", "style"),
            Output("m-pf", "children"),
            Output("m-pf", "style"),
            Output("m-trades", "children"),
            Output("m-trades", "style"),
            Output("status-indicator", "children"),
            Output("footer-ticks", "children"),
            Output("header-subtitle", "children"),
            Output("config-bar", "children"),
        ],
        Input("interval", "n_intervals"),
    )
    def update_metrics(_n: int):
        state = get_state()
        cash = float(state.get("cash", 0))
        equity = float(state.get("equity", 0))
        ret = float(state.get("total_return_pct", 0))
        dd = float(state.get("max_drawdown_pct", 0))
        running = bool(state.get("running", False))
        ticks = int(state.get("tick_count", 0))
        bars = int(state.get("bar_count", 0))
        stats = state.get("trade_stats", {})
        leverage = float(state.get("leverage", 1.0))
        sl_pct = state.get("stop_loss_pct")
        tp_pct = state.get("take_profit_pct")
        strategy_name = state.get("strategy_name", "")

        pnl_df = get_daily_pnl(1)
        daily = float(pnl_df.iloc[0]["daily_pnl"]) if not pnl_df.empty else 0.0
        wr = float(stats.get("win_rate", 0))
        pf = float(stats.get("profit_factor", 0))
        total_t = int(stats.get("total_trades", 0))

        def _pnl_style(val: float) -> dict:
            color = _GREEN if val > 0 else (_RED if val < 0 else _TEXT)
            return {"color": color, "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}

        cash_s = {"color": _AMBER, "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}
        eq_s = {"color": _BLUE, "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}
        dd_s = {"color": _RED if dd > 5 else (_AMBER if dd > 2 else _GREEN), "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}
        wr_s = {"color": _GREEN if wr >= 50 else _RED, "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}
        pf_s = {"color": _GREEN if pf >= 1 else _RED, "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}
        tr_s = {"color": _CYAN, "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}

        dot_color = _GREEN if running else _RED
        status_children = [
            html.Span(className="status-dot", style={"backgroundColor": dot_color}),
            html.Span("LIVE" if running else "STOPPED", style={"color": dot_color, "fontWeight": "600"}),
        ]
        subtitle = f"Paper Trading • {strategy_name}" if strategy_name else "Paper Trading • Strategy Terminal"
        fusion_mode = state.get("fusion_mode", "")
        badges = [_badge("Leverage", f"{leverage:.0f}x", _RED if leverage > 1 else _MUTED)]
        if fusion_mode:
            badges.append(_badge("Fusion", fusion_mode.replace("_", " ").title(), _PURPLE))
        if strategy_name:
            badges.append(_badge("Strategy", strategy_name, _BLUE))
        if sl_pct:
            badges.append(_badge("Stop Loss", f"-{float(sl_pct)*100:.1f}%", _RED))
        if tp_pct:
            badges.append(_badge("Take Profit", f"+{float(tp_pct)*100:.1f}%", _GREEN))
        sym_list = ", ".join(state.get("symbols", []))
        if sym_list:
            badges.append(_badge("Symbols", sym_list, _CYAN))
        init_cash = float(state.get("initial_cash", 0))
        if init_cash:
            badges.append(_badge("Initial", f"${init_cash:,.0f}", _AMBER))

        footer_text = f"Bars: {bars} • Ticks: {ticks:,}"
        return (
            f"${cash:,.0f}",
            cash_s,
            f"${equity:,.0f}",
            eq_s,
            f"{ret:+.2f}%",
            _pnl_style(ret),
            f"${daily:+,.0f}",
            _pnl_style(daily),
            f"-{dd:.2f}%",
            dd_s,
            f"{wr:.1f}%",
            wr_s,
            f"{pf:.2f}x",
            pf_s,
            str(total_t),
            tr_s,
            status_children,
            footer_text,
            subtitle,
            badges,
        )

    @app.callback(
        [
            Output("strategy-leaderboard", "figure"),
            Output("strategy-scorecard", "figure"),
            Output("strategy-compare", "figure"),
            Output("scorecard-strategy", "options"),
            Output("scorecard-strategy", "value"),
        ],
        [
            Input("interval", "n_intervals"),
            Input("rank-basis", "value"),
            Input("scorecard-strategy", "value"),
            Input("compare-metric", "value"),
            Input("symbol-select", "value"),
        ],
    )
    def update_strategy_panels(_n: int, rank_basis: str, selected_strategy: str, compare_metric: str, symbol: str):
        state = get_state()
        rows = state.get("strategy_performance", []) or []
        if symbol:
            filtered = [r for r in rows if str(r.get("symbol", "")) == symbol]
            rows = filtered or rows
        ranked = _rank_rows(rows, rank_basis)
        options = [{"label": f"{r.get('strategy','')} ({r.get('symbol','')})", "value": f"{r.get('strategy','')}|{r.get('symbol','')}"} for r in ranked]

        selected_row = ranked[0] if ranked else None
        if selected_strategy:
            for r in ranked:
                key = f"{r.get('strategy','')}|{r.get('symbol','')}"
                if key == selected_strategy:
                    selected_row = r
                    break

        selected_value = f"{selected_row.get('strategy','')}|{selected_row.get('symbol','')}" if selected_row else ""
        return (
            build_strategy_leaderboard(ranked, rank_by=rank_basis),
            build_strategy_scorecard(selected_row),
            build_strategy_compare(ranked, metric=compare_metric),
            options,
            selected_value,
        )

    @app.callback(Output("equity-chart", "figure"), Input("interval", "n_intervals"))
    def update_equity(_n: int):
        df = get_equity_curve()
        if df.empty:
            return build_equity_chart([], [])
        return build_equity_chart(
            timestamps=df["timestamp"].tolist(),
            equity_values=df["equity"].tolist(),
            cash_values=df["cash"].tolist() if "cash" in df.columns else None,
        )

    @app.callback(Output("positions-table", "figure"), Input("interval", "n_intervals"))
    def update_positions(_n: int):
        state = get_state()
        return build_positions_table(
            positions=state.get("positions", {}),
            prices=state.get("prices", {}),
            total_equity=state.get("equity", 0),
            position_details=state.get("position_details"),
        )

    @app.callback(
        [Output("multi-tf-panel", "figure"), Output("multi-tf-row", "style"), Output("interval-select", "options")],
        Input("interval", "n_intervals"),
    )
    def update_multi_tf(_n: int):
        state = get_state()
        tf_pos = state.get("tf_positions", {})
        fig = build_multi_tf_panel(tf_pos)
        show = {"marginBottom": "12px"} if tf_pos else {"display": "none"}
        iv_opts = [{"label": "Auto", "value": "auto"}]
        for info in tf_pos.values():
            for iv in info.get("intervals", []):
                opt = {"label": iv, "value": iv}
                if opt not in iv_opts:
                    iv_opts.append(opt)
        return fig, show, iv_opts

    @app.callback(
        Output("candle-chart", "figure"),
        [Input("interval", "n_intervals"), Input("symbol-select", "value"), Input("interval-select", "value")],
    )
    def update_candles(_n: int, symbol: str, sel_iv: str):
        if not symbol:
            return build_candlestick_chart(pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]))
        iv = None if sel_iv == "auto" else sel_iv
        try:
            df = get_window(symbol, iv)
        except TypeError:
            df = get_window(symbol)

        bars = 0
        span = "no data"
        if isinstance(df, pd.DataFrame) and not df.empty:
            bars = len(df)
            dcol = "date" if "date" in df.columns else df.columns[0]
            dt = pd.to_datetime(df[dcol], errors="coerce", utc=True).dropna()
            if not dt.empty:
                span = f"{dt.iloc[0]} -> {dt.iloc[-1]}"

        trades = get_trades(80)
        sym_trades = trades[trades["symbol"] == symbol] if (not trades.empty and "symbol" in trades.columns) else pd.DataFrame()
        state = get_state()
        levels = state.get("sl_tp_levels", {}).get(symbol)
        title_suffix = f" ({sel_iv})" if sel_iv and sel_iv != "auto" else ""
        return build_candlestick_chart(
            df,
            trades_df=sym_trades,
            title=f"{symbol} Price{title_suffix} | bars={bars} | {span}",
            sl_tp_levels=levels,
        )

    @app.callback(Output("stats-panel", "figure"), Input("interval", "n_intervals"))
    def update_stats(_n: int):
        state = get_state()
        return build_stats_panel(state.get("trade_stats", {}))

    @app.callback(Output("trade-log", "figure"), Input("interval", "n_intervals"))
    def update_trade_log(_n: int):
        return build_trade_log_table(get_trades(60))

    @app.callback(Output("pnl-chart", "figure"), Input("interval", "n_intervals"))
    def update_pnl(_n: int):
        df = get_daily_pnl(30)
        if df.empty:
            return build_pnl_bar_chart([], [])
        return build_pnl_bar_chart(dates=df["date"].tolist(), pnl_values=df["daily_pnl"].tolist())

    return app


def run_dashboard_thread(
    app: "dash.Dash",
    host: str = "0.0.0.0",
    port: int = 8050,
    debug: bool = False,
) -> threading.Thread:
    """Launch the Dash app in a daemon thread (non-blocking)."""
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=debug, use_reloader=False),
        daemon=True,
    )
    t.start()
    return t
