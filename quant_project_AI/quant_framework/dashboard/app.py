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

_BG = "#060912"
_CARD = "#0c1220"
_SURFACE = "#121a2b"
_BORDER = "#243041"
_TEXT = "#f8fbff"
_MUTED = "#91a3bb"
_GREEN = "#22c55e"
_RED = "#f87171"
_BLUE = "#4f8cff"
_AMBER = "#fbbf24"
_PURPLE = "#9b8afb"
_CYAN = "#22d3ee"

_FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
_HEAD_SCRIPTS = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
]

_CUSTOM_CSS = f"""
body {{
    background:
        radial-gradient(circle at top left, rgba(79, 140, 255, 0.16), transparent 24%),
        radial-gradient(circle at top right, rgba(34, 211, 238, 0.10), transparent 20%),
        linear-gradient(180deg, #070b14 0%, {_BG} 38%, #04070d 100%) !important;
    font-family: {_FONT} !important;
    color: {_TEXT};
    margin: 0;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

.app-shell {{
    max-width: 1700px;
    margin: 0 auto;
    padding: 18px 24px 28px;
}}

.topbar {{
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 14px;
}}

.brand-panel {{
    display: flex;
    align-items: center;
    gap: 14px;
    min-width: 360px;
    padding: 14px 16px;
    background: linear-gradient(135deg, rgba(12, 18, 32, 0.96) 0%, rgba(18, 26, 43, 0.92) 100%);
    border: 1px solid rgba(79, 140, 255, 0.18);
    border-radius: 16px;
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.34);
}}

.brand-lockup {{
    display: flex;
    align-items: center;
    gap: 14px;
}}

.brand-mark {{
    width: 42px;
    height: 42px;
    border-radius: 13px;
    background: linear-gradient(135deg, {_BLUE}, {_CYAN});
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: 800;
    color: #ffffff;
    box-shadow: 0 10px 28px rgba(79, 140, 255, 0.34);
}}

.brand-copy {{
    display: flex;
    flex-direction: column;
    gap: 3px;
}}

.brand-eyebrow {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: {_CYAN};
}}

.brand-title {{
    margin: 0;
    font-size: 1.12rem;
    font-weight: 700;
    color: {_TEXT};
    letter-spacing: 0.85px;
}}

.brand-subtitle {{
    font-size: 0.76rem;
    color: {_MUTED};
    font-weight: 500;
}}

.topbar-actions {{
    display: flex;
    align-items: stretch;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: flex-end;
}}

.topbar-panel {{
    min-width: 180px;
    padding: 10px 12px;
    border-radius: 14px;
    border: 1px solid {_BORDER};
    background: linear-gradient(180deg, rgba(12, 18, 32, 0.92), rgba(10, 15, 27, 0.98));
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
}}

.topbar-panel.danger {{
    min-width: 260px;
    border-color: rgba(248, 113, 113, 0.34);
    background: linear-gradient(180deg, rgba(36, 14, 20, 0.95), rgba(18, 9, 15, 0.98));
}}

.topbar-label {{
    margin-bottom: 8px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    color: {_MUTED};
}}

.kill-switch-btn {{
    width: 100%;
    min-height: 44px;
    border: 1px solid rgba(248, 113, 113, 0.5);
    border-radius: 12px;
    background: linear-gradient(180deg, rgba(248, 113, 113, 0.18), rgba(120, 22, 33, 0.88));
    color: {_TEXT};
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    box-shadow: 0 10px 24px rgba(120, 22, 33, 0.28);
}}

.kill-switch-btn:hover {{
    transform: translateY(-1px);
    border-color: {_RED};
    box-shadow: 0 14px 30px rgba(248, 113, 113, 0.18);
}}

.kill-switch-btn:disabled {{
    cursor: not-allowed;
    opacity: 0.72;
    transform: none;
}}

.kill-switch-status {{
    margin-top: 8px;
    font-size: 0.73rem;
    line-height: 1.45;
    color: {_MUTED};
}}

.shell-card {{
    background: linear-gradient(180deg, rgba(12, 18, 32, 0.96) 0%, rgba(10, 15, 27, 0.98) 100%);
    border: 1px solid {_BORDER};
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.34);
}}

.toolbar-card {{
    margin-bottom: 14px;
}}

.toolbar-grid {{
    display: grid;
    grid-template-columns: repeat(12, minmax(0, 1fr));
    gap: 12px;
    align-items: end;
}}

.toolbar-field {{
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
}}

.toolbar-field.compact {{
    grid-column: span 2;
}}

.toolbar-field.standard {{
    grid-column: span 3;
}}

.toolbar-field.wide {{
    grid-column: span 4;
}}

.toolbar-field.fill {{
    grid-column: span 3;
}}

.toolbar-label {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    color: {_MUTED};
}}

.toolbar-hint {{
    display: flex;
    align-items: center;
    justify-content: flex-end;
    min-height: 46px;
    padding: 0 14px;
    border-radius: 12px;
    border: 1px solid {_BORDER};
    background: rgba(18, 26, 43, 0.72);
    color: {_MUTED};
    font-size: 0.76rem;
    font-weight: 600;
    white-space: nowrap;
}}

.config-bar {{
    display: flex;
    gap: 8px;
    margin-bottom: 14px;
    flex-wrap: wrap;
}}

.section-grid {{
    display: grid;
    gap: 12px;
    margin-bottom: 12px;
}}

.section-grid.main {{
    grid-template-columns: 2fr 1.25fr;
}}

.section-grid.market {{
    grid-template-columns: 3fr 2fr;
}}

.section-grid.triple {{
    grid-template-columns: 2fr 1.5fr 1.5fr;
}}

.chart-header-row {{
    display: flex;
    justify-content: flex-end;
    padding: 2px 8px 0 0;
}}

.metric-card {{
    position: relative;
    background: linear-gradient(180deg, rgba(18, 26, 43, 0.96) 0%, rgba(12, 18, 32, 0.98) 100%);
    border: 1px solid rgba(36, 48, 65, 0.92);
    border-radius: 14px;
    padding: 15px 16px;
    min-width: 130px;
    transition: all 0.2s ease;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.24);
}}
.metric-card::before {{
    content: "";
    position: absolute;
    left: 16px;
    right: 16px;
    top: 0;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(79, 140, 255, 0), rgba(79, 140, 255, 0.9), rgba(34, 211, 238, 0));
}}
.metric-card:hover {{
    border-color: {_BLUE};
    box-shadow: 0 16px 32px rgba(79, 140, 255, 0.12);
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

.chart-card {{
    background: linear-gradient(180deg, rgba(12, 18, 32, 0.98) 0%, rgba(8, 12, 22, 0.98) 100%);
    border: 1px solid {_BORDER};
    border-radius: 16px;
    padding: 14px;
    overflow: hidden;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.34);
}}
.chart-card:hover {{
    border-color: rgba(79, 140, 255, 0.5);
}}
.chart-card .js-plotly-plot .plotly .main-svg {{
    border-radius: 12px;
}}

.dash-dropdown {{
    font-family: {_FONT} !important;
}}
.dash-dropdown .Select {{
    color: {_TEXT} !important;
}}
.dash-dropdown .Select-control {{
    background-color: {_SURFACE} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 12px !important;
    min-height: 44px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02), 0 8px 18px rgba(0, 0, 0, 0.22) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease !important;
}}
.dash-dropdown .Select-control:hover {{
    border-color: {_BLUE} !important;
    box-shadow: 0 0 0 1px rgba(79, 140, 255, 0.25), 0 12px 24px rgba(79, 140, 255, 0.16) !important;
}}
.dash-dropdown.is-focused > .Select-control,
.dash-dropdown.is-open > .Select-control,
.dash-dropdown .is-focused:not(.is-open) > .Select-control {{
    border-color: {_CYAN} !important;
    box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.45), 0 0 0 4px rgba(34, 211, 238, 0.12) !important;
}}
.dash-dropdown .Select-placeholder,
.dash-dropdown .Select--single > .Select-control .Select-value,
.dash-dropdown .has-value.Select--single > .Select-control .Select-value,
.dash-dropdown .Select-value-label,
.dash-dropdown .Select-value {{
    color: {_TEXT} !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    line-height: 42px !important;
    opacity: 1 !important;
}}
.dash-dropdown .Select-placeholder {{
    color: {_MUTED} !important;
}}
.dash-dropdown .Select-placeholder,
.dash-dropdown .Select--single > .Select-control .Select-value {{
    padding-left: 14px !important;
    padding-right: 38px !important;
}}
.dash-dropdown .Select-input {{
    color: {_TEXT} !important;
    height: auto !important;
    padding-left: 14px !important;
}}
.dash-dropdown .Select-input > input {{
    color: {_TEXT} !important;
    background: transparent !important;
    padding: 10px 0 !important;
}}
.dash-dropdown .Select-clear-zone,
.dash-dropdown .Select-arrow-zone {{
    display: flex !important;
    align-items: center !important;
}}
.dash-dropdown .Select-arrow-zone {{
    padding-right: 12px !important;
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
    background: linear-gradient(180deg, rgba(18, 26, 43, 0.98), rgba(8, 12, 22, 0.98)) !important;
    border: 1px solid rgba(79, 140, 255, 0.65) !important;
    border-radius: 12px !important;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.48) !important;
    margin-top: 6px !important;
    z-index: 9999 !important;
    overflow: hidden !important;
}}
.dash-dropdown .Select-menu {{
    max-height: 320px !important;
    overflow-y: auto !important;
    background: transparent !important;
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
.dash-dropdown .VirtualizedSelectOption {{
    background: transparent !important;
    color: {_TEXT} !important;
    padding: 12px 14px !important;
    font-size: 0.85rem !important;
}}
.dash-dropdown .VirtualizedSelectFocusedOption {{
    background: {_SURFACE} !important;
    color: {_TEXT} !important;
}}
.dash-dropdown .VirtualizedSelectSelectedOption {{
    background: {_BLUE}30 !important;
    color: {_TEXT} !important;
    font-weight: 600 !important;
}}
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

@media (max-width: 1280px) {{
    .toolbar-field.compact,
    .toolbar-field.standard,
    .toolbar-field.wide,
    .toolbar-field.fill {{
        grid-column: span 6;
    }}
    .section-grid.main,
    .section-grid.market,
    .section-grid.triple {{
        grid-template-columns: 1fr;
    }}
}}

@media (max-width: 820px) {{
    .app-shell {{
        padding: 14px;
    }}
    .topbar {{
        flex-direction: column;
    }}
    .brand-panel,
    .topbar-panel {{
        min-width: 0;
    }}
    .toolbar-grid {{
        grid-template-columns: 1fr;
    }}
    .toolbar-field.compact,
    .toolbar-field.standard,
    .toolbar-field.wide,
    .toolbar-field.fill {{
        grid-column: span 1;
    }}
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


def _dropdown_field(label: str, control: html.Div, class_name: str = "standard") -> html.Div:
    return html.Div(
        className=f"toolbar-field {class_name}",
        children=[
            html.Div(label, className="toolbar-label"),
            control,
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
    trigger_kill_switch: Callable[[str], Dict[str, Any]] | None = None,
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
        className="app-shell",
        children=[
            dcc.Interval(id="interval", interval=refresh_ms, n_intervals=0),
            html.Div(
                className="topbar",
                children=[
                    html.Div(
                        className="brand-panel",
                        children=[
                            html.Div(
                                className="brand-lockup",
                                children=[
                                    html.Div("Q", className="brand-mark"),
                                    html.Div(
                                        className="brand-copy",
                                        children=[
                                            html.Div("Quant AI Monitor", className="brand-eyebrow"),
                                            html.H3("TRADING TERMINAL PRO", className="brand-title"),
                                            html.Span(id="header-subtitle", className="brand-subtitle"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="topbar-actions",
                        children=[
                            html.Div(
                                className="topbar-panel danger",
                                children=[
                                    html.Div("Emergency Control", className="topbar-label"),
                                    html.Button(
                                        "Activate Kill Switch",
                                        id="kill-switch-btn",
                                        n_clicks=0,
                                        className="kill-switch-btn",
                                    ),
                                    html.Div(
                                        id="kill-switch-status",
                                        className="kill-switch-status",
                                        children="One-click flatten and halt trading.",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="topbar-panel",
                                children=[
                                    html.Div("Engine Status", className="topbar-label"),
                                    html.Div(
                                        id="status-indicator",
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "fontSize": "0.82rem",
                                            "fontWeight": "600",
                                            "padding": "6px 0",
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                className="topbar-panel",
                                children=[
                                    html.Div("Active Symbol", className="topbar-label"),
                                    dcc.Dropdown(
                                        id="symbol-select",
                                        options=symbol_options,
                                        value=default_symbol,
                                        clearable=False,
                                        className="dash-dropdown",
                                        style={"fontSize": "0.85rem"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="shell-card toolbar-card",
                children=[
                    html.Div(
                        className="toolbar-grid",
                        children=[
                            _dropdown_field(
                                "Leaderboard Rank",
                                dcc.Dropdown(
                                    id="rank-basis",
                                    options=[
                                        {"label": "Rank by Score", "value": "score"},
                                        {"label": "Rank by Sharpe", "value": "sharpe"},
                                        {"label": "Rank by Return", "value": "return"},
                                        {"label": "Rank by Calmar", "value": "calmar"},
                                        {"label": "Rank by Stability", "value": "stability"},
                                    ],
                                    value="score",
                                    clearable=False,
                                    className="dash-dropdown",
                                ),
                                "compact",
                            ),
                            _dropdown_field(
                                "Scorecard Focus",
                                dcc.Dropdown(
                                    id="scorecard-strategy",
                                    options=[],
                                    value="",
                                    clearable=False,
                                    className="dash-dropdown",
                                    placeholder="Select strategy and symbol",
                                ),
                                "wide",
                            ),
                            _dropdown_field(
                                "Compare View",
                                dcc.Dropdown(
                                    id="compare-metric",
                                    options=[
                                        {"label": "Risk / Return", "value": "risk_return"},
                                        {"label": "Quality", "value": "quality"},
                                        {"label": "Composite", "value": "composite"},
                                    ],
                                    value="risk_return",
                                    clearable=False,
                                    className="dash-dropdown",
                                ),
                                "compact",
                            ),
                            _dropdown_field(
                                "Feed Snapshot",
                                html.Div(id="footer-ticks", className="toolbar-hint"),
                                "fill",
                            ),
                        ],
                    )
                ],
            ),
            html.Div(id="config-bar", className="config-bar"),
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
                className="section-grid main",
                children=[
                    html.Div(className="chart-card", children=[dcc.Graph(id="strategy-leaderboard", config={"displayModeBar": False}, style={"height": "340px"})]),
                    html.Div(className="chart-card", children=[dcc.Graph(id="strategy-scorecard", config={"displayModeBar": False}, style={"height": "340px"})]),
                ],
            ),
            html.Div(className="chart-card", style={"marginBottom": "12px"}, children=[dcc.Graph(id="strategy-compare", config={"displayModeBar": False}, style={"height": "360px"})]),
            html.Div(
                className="section-grid market",
                children=[
                    html.Div(
                        className="chart-card",
                        children=[
                            html.Div(
                                className="chart-header-row",
                                children=[
                                    html.Div(
                                        style={"width": "116px"},
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
                className="section-grid triple",
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
            Output("kill-switch-status", "children"),
            Output("kill-switch-status", "style"),
            Output("kill-switch-btn", "children"),
            Output("kill-switch-btn", "disabled"),
        ],
        [Input("interval", "n_intervals"), Input("kill-switch-btn", "n_clicks")],
    )
    def sync_kill_switch(_n: int, n_clicks: int):
        error_message = ""
        ctx = getattr(dash, "callback_context", None)
        triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx and ctx.triggered else ""

        if triggered == "kill-switch-btn" and n_clicks:
            if trigger_kill_switch is None:
                error_message = "Kill switch is unavailable in this session."
            else:
                try:
                    result = trigger_kill_switch("Manual dashboard activation")
                    if result.get("status") == "error":
                        error_message = f"Activation failed: {result.get('message', 'unknown error')}"
                except Exception as e:
                    error_message = f"Activation failed: {e}"

        state = get_state()
        active = bool(state.get("kill_switch_active", False))
        reason = str(state.get("kill_switch_reason", "") or "")
        if active:
            return (
                f"Kill switch active. Trading halted. {reason}".strip(),
                {"color": _RED, "fontWeight": "700"},
                "Kill Switch Engaged",
                True,
            )
        if error_message:
            return (
                error_message,
                {"color": _RED, "fontWeight": "600"},
                "Retry Kill Switch",
                False,
            )
        if trigger_kill_switch is None:
            return (
                "Kill switch is unavailable for this dashboard session.",
                {"color": _MUTED},
                "Kill Switch Unavailable",
                True,
            )
        return (
            "One-click flatten and halt trading immediately.",
            {"color": _MUTED},
            "Activate Kill Switch",
            False,
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
        kill_switch_active = bool(state.get("kill_switch_active", False))
        kill_switch_reason = str(state.get("kill_switch_reason", "") or "")

        def _pnl_style(val: float) -> dict:
            color = _GREEN if val > 0 else (_RED if val < 0 else _TEXT)
            return {"color": color, "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}

        cash_s = {"color": _AMBER, "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}
        eq_s = {"color": _BLUE, "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}
        dd_s = {"color": _RED if dd > 5 else (_AMBER if dd > 2 else _GREEN), "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}
        wr_s = {"color": _GREEN if wr >= 50 else _RED, "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}
        pf_s = {"color": _GREEN if pf >= 1 else _RED, "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}
        tr_s = {"color": _CYAN, "fontSize": "1.18rem", "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}

        dot_color = _RED if kill_switch_active else (_GREEN if running else _AMBER)
        status_children = [
            html.Span(className="status-dot", style={"backgroundColor": dot_color}),
            html.Span(
                "KILL SWITCH" if kill_switch_active else ("LIVE" if running else "HALTED"),
                style={"color": dot_color, "fontWeight": "600"},
            ),
        ]
        subtitle = f"Paper Trading • {strategy_name}" if strategy_name else "Paper Trading • Strategy Terminal"
        fusion_mode = state.get("fusion_mode", "")
        badges = [_badge("Leverage", f"{leverage:.0f}x", _RED if leverage > 1 else _MUTED)]
        if kill_switch_active:
            badges.insert(0, _badge("Kill Switch", "ACTIVE", _RED))
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
        if kill_switch_active and kill_switch_reason:
            badges.append(_badge("Reason", kill_switch_reason[:36], _RED))

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
