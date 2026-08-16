"""Climate & hydrology presentation layer for Economic Data.

This module renders only persisted GloFAS monthly indicators plus the small
PEGELONLINE Kaub live cache.  Long GloFAS downloads are intentionally handled
by :mod:`economic_data.climate_data` from the terminal, not inside Dash.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from .climate_data import (
    GLOFAS_CLIMATOLOGY_END,
    GLOFAS_CLIMATOLOGY_START,
    GLOFAS_SOURCE_VERSION,
    SERIES_LABELS,
    SERIES_ORDER,
    dashboard_snapshot,
)


TOKENS = {
    "ink": "#111827",
    "muted": "#667085",
    "hairline": "#E4E7EC",
    "panel": "#FFFFFF",
    "background": "#F5F7FA",
    "blue": "#1F4E79",
    "teal": "#0B7A5C",
    "red": "#B42318",
    "amber": "#A66A00",
}

PANEL_STYLE = {
    "backgroundColor": "#FFFFFF",
    "border": "1px solid #E4E7EC",
    "borderRadius": "12px",
    "padding": "18px",
    "marginBottom": "16px",
}
SECONDARY_BUTTON_STYLE = {
    "backgroundColor": "#FFFFFF",
    "color": TOKENS["blue"],
    "border": f"1px solid {TOKENS['blue']}",
    "borderRadius": "6px",
    "padding": "9px 15px",
    "fontWeight": 700,
    "cursor": "pointer",
}
CONTROL_LABEL_STYLE = {
    "display": "block",
    "fontWeight": 650,
    "fontSize": "11px",
    "color": TOKENS["muted"],
    "marginBottom": "7px",
    "textTransform": "uppercase",
    "letterSpacing": ".04em",
}


def _panel_heading(eyebrow: str, title: str, subtitle: str):
    return html.Div(
        [
            html.Div(
                eyebrow,
                style={
                    "fontSize": "9px", "fontWeight": 750,
                    "letterSpacing": ".07em", "textTransform": "uppercase",
                    "color": TOKENS["muted"], "marginBottom": "4px",
                },
            ),
            html.H3(title, style={"margin": "0 0 4px 0", "fontSize": "18px", "fontWeight": 650}),
            html.P(
                subtitle,
                style={"margin": 0, "fontSize": "11px", "color": TOKENS["muted"], "lineHeight": 1.5},
            ),
        ],
        style={"marginBottom": "14px"},
    )


def _empty_figure(message: str, *, height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=message, showarrow=False,
        font={"size": 12, "color": TOKENS["muted"]},
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 30, "r": 20, "t": 25, "b": 30},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _frame(payload: Mapping[str, Any] | None) -> pd.DataFrame:
    if not payload or not payload.get("monthly_json"):
        return pd.DataFrame()
    try:
        frame = pd.read_json(StringIO(payload["monthly_json"]), orient="split")
    except Exception:
        return pd.DataFrame()
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in ("stress_score", "raw_value"):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["date"]) if "date" in frame else pd.DataFrame()


def _fmt_sigma(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(x) else f"{x:+.2f}σ"


def _fmt_raw(value: Any, unit: str | None) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    unit = str(unit or "")
    if unit == "m³/s":
        return f"{x:,.0f} m³/s"
    if unit.startswith("%"):
        return f"{x:.1f}%"
    return f"{x:.2f} {unit}".strip()


def _stress_colour(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return TOKENS["muted"]
    if not np.isfinite(x):
        return TOKENS["muted"]
    if x >= 1.5:
        return TOKENS["red"]
    if x >= 0.75:
        return TOKENS["amber"]
    if x <= -0.75:
        return TOKENS["teal"]
    return TOKENS["ink"]


def _kpi_card(title: str, value: str, subtitle: str, *, colour: str | None = None):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "9px", "fontWeight": 750, "color": TOKENS["muted"], "textTransform": "uppercase", "letterSpacing": ".05em"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": 750, "marginTop": "3px", "color": colour or TOKENS["ink"], "fontVariantNumeric": "tabular-nums"}),
            html.Div(subtitle, style={"fontSize": "9px", "color": TOKENS["muted"], "marginTop": "3px", "lineHeight": 1.35}),
        ],
        style={
            "border": f"1px solid {TOKENS['hairline']}",
            "borderRadius": "8px", "padding": "11px 12px", "background": "#FBFCFE",
        },
    )


def climate_section():
    """Return the additive Climate & Hydrology section for Economic Data."""
    options = [{"label": SERIES_LABELS[name], "value": name} for name in SERIES_ORDER]
    return html.Div(
        [
            dcc.Store(id="economic-climate-store", storage_type="memory"),
            _panel_heading(
                "Climate & hydrology",
                "River stress monitor",
                "GloFAS v5.0 modelled discharge is converted to monthly, seasonally-standardised stress indicators. PEGELONLINE Kaub is kept separate as the observed live Rhine gauge.",
            ),
            html.Div(
                [
                    html.Button(
                        "Refresh live Kaub / reload cache",
                        id="economic-climate-refresh",
                        n_clicks=0,
                        style=SECONDARY_BUTTON_STYLE,
                    ),
                    html.Div(
                        id="economic-climate-status",
                        style={"fontSize": "10px", "color": TOKENS["muted"], "lineHeight": 1.45},
                    ),
                ],
                style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap", "marginBottom": "12px"},
            ),
            html.Div(
                id="economic-climate-kpis",
                style={"display": "grid", "gridTemplateColumns": "repeat(4,minmax(160px,1fr))", "gap": "10px", "marginBottom": "16px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Climate stress heatmap", style={"fontSize": "13px", "fontWeight": 700, "marginBottom": "2px"}),
                            html.Div(
                                f"Positive values = drier / lower-flow stress. Seasonal z-scores use the {GLOFAS_CLIMATOLOGY_START}–{GLOFAS_CLIMATOLOGY_END} climatology.",
                                style={"fontSize": "10px", "color": TOKENS["muted"]},
                            ),
                        ]
                    ),
                    dcc.Graph(
                        id="economic-climate-heatmap",
                        figure=_empty_figure("Run the one-time GloFAS build to populate climate history.", height=360),
                        config={"displaylogo": False, "scrollZoom": True},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Detailed series", style=CONTROL_LABEL_STYLE),
                            dcc.Dropdown(
                                id="economic-climate-series",
                                options=options,
                                value=SERIES_ORDER[0],
                                clearable=False,
                                style={"minWidth": "260px"},
                            ),
                            dcc.Graph(
                                id="economic-climate-detail",
                                figure=_empty_figure("No processed GloFAS history available."),
                                config={"displaylogo": False, "scrollZoom": True},
                            ),
                        ],
                        style={"minWidth": 0},
                    ),
                    html.Div(
                        [
                            html.Div("Kaub · observed Rhine gauge", style={"fontSize": "13px", "fontWeight": 700, "marginBottom": "3px"}),
                            html.Div(
                                "PEGELONLINE WSV raw water level: last 31 days plus the published WV forecast when available. This observed gauge is not spliced into GloFAS.",
                                style={"fontSize": "10px", "color": TOKENS["muted"], "lineHeight": 1.45, "marginBottom": "4px"},
                            ),
                            dcc.Graph(
                                id="economic-climate-kaub",
                                figure=_empty_figure("Kaub live data are unavailable."),
                                config={"displaylogo": False, "scrollZoom": True},
                            ),
                        ],
                        style={"minWidth": 0},
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"},
            ),
            html.Div(
                [
                    html.Strong("Model-point method: "),
                    "v1 selects the highest long-run median-discharge GloFAS cell inside a small local bbox around Kaub / the Vienna Danube reach, persists that cell, and reuses it on refresh. The GloFAS point is a model river cell, not an observed gauge. Upstream-area validation can be added without changing the dashboard data contract.",
                ],
                style={"fontSize": "9px", "color": TOKENS["muted"], "lineHeight": 1.45, "marginTop": "8px"},
            ),
        ],
        style=PANEL_STYLE,
    )


def _latest(frame: pd.DataFrame, name: str) -> pd.Series | None:
    block = frame.loc[frame.get("series", pd.Series(dtype=str)).astype(str).eq(name)].dropna(subset=["stress_score"]).sort_values("date")
    return None if block.empty else block.iloc[-1]


def render_kpis(payload: Mapping[str, Any] | None):
    frame = _frame(payload)
    cards = []
    for name in ("rhine_discharge_stress", "rhine_low_flow_intensity", "danube_discharge_stress"):
        row = _latest(frame, name)
        if row is None:
            cards.append(_kpi_card(SERIES_LABELS[name], "—", "GloFAS cache not built"))
        else:
            score = float(row["stress_score"])
            date = pd.Timestamp(row["date"]).strftime("%b %Y")
            cards.append(
                _kpi_card(
                    SERIES_LABELS[name],
                    _fmt_sigma(score),
                    f"{date} · {_fmt_raw(row.get('raw_value'), row.get('raw_unit'))}",
                    colour=_stress_colour(score),
                )
            )

    kaub = dict((payload or {}).get("kaub") or {})
    current = dict(kaub.get("current") or {})
    value = current.get("value")
    timestamp = current.get("timestamp")
    status = str(kaub.get("status") or "unavailable").replace("_", " ")
    try:
        current_text = f"{float(value):.0f} cm"
    except (TypeError, ValueError):
        current_text = "—"
    try:
        stamp = pd.Timestamp(timestamp).strftime("%d %b %H:%M")
    except Exception:
        stamp = "no timestamp"
    cards.append(_kpi_card("Kaub · observed water level", current_text, f"{stamp} · {status}", colour=TOKENS["blue"]))
    return cards


def climate_heatmap_figure(payload: Mapping[str, Any] | None) -> go.Figure:
    frame = _frame(payload)
    if frame.empty:
        return _empty_figure("No processed GloFAS history. Run climate_data.py build.", height=360)
    use = frame.loc[frame["series"].astype(str).isin(SERIES_ORDER)].copy()
    if use.empty:
        return _empty_figure("Climate cache has no recognised stress series.", height=360)
    # Keep a readable macro window while retaining zoom/pan for older history.
    cutoff = use["date"].max() - pd.DateOffset(years=12)
    use = use.loc[use["date"] >= cutoff]
    pivot = use.pivot_table(index="series", columns="date", values="stress_score", aggfunc="last")
    pivot = pivot.reindex([name for name in SERIES_ORDER if name in pivot.index])
    labels = [SERIES_LABELS.get(name, name) for name in pivot.index]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=pd.DatetimeIndex(pivot.columns),
            y=labels,
            zmin=-3,
            zmax=3,
            zmid=0,
            colorscale="RdBu_r",
            colorbar={"title": "stress σ", "thickness": 12},
            hovertemplate="%{y}<br>%{x|%b %Y}<br>stress %{z:+.2f}σ<extra></extra>",
            connectgaps=False,
        )
    )
    fig.update_layout(
        template="plotly_white", height=360,
        margin={"l": 165, "r": 30, "t": 18, "b": 35},
        dragmode="pan", hovermode="closest",
        xaxis_title=None, yaxis_title=None,
    )
    fig.update_xaxes(showgrid=False)
    return fig


def climate_detail_figure(payload: Mapping[str, Any] | None, series: str) -> go.Figure:
    frame = _frame(payload)
    block = frame.loc[frame.get("series", pd.Series(dtype=str)).astype(str).eq(str(series))].sort_values("date") if not frame.empty else pd.DataFrame()
    if block.empty:
        return _empty_figure("No history is available for this climate series.")
    label = SERIES_LABELS.get(str(series), str(series))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=block["date"], y=block["stress_score"], mode="lines",
            name="Climate stress", line={"width": 1.8, "color": TOKENS["blue"]},
            customdata=np.column_stack([block["raw_value"].to_numpy(), block["raw_unit"].astype(str).to_numpy()]),
            hovertemplate="%{x|%b %Y}<br>stress %{y:+.2f}σ<br>raw %{customdata[0]:.2f} %{customdata[1]}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    fig.add_hrect(y0=1.5, y1=4.0, fillcolor="rgba(180,35,24,.05)", line_width=0)
    fig.update_layout(
        template="plotly_white", height=390, title={"text": label, "x": .01, "font": {"size": 14}},
        margin={"l": 50, "r": 20, "t": 44, "b": 35},
        yaxis_title="stress σ", dragmode="pan", hovermode="x unified",
    )
    return fig


def kaub_figure(payload: Mapping[str, Any] | None) -> go.Figure:
    kaub = dict((payload or {}).get("kaub") or {})
    history = pd.DataFrame(kaub.get("history") or [])
    forecast = pd.DataFrame(kaub.get("forecast") or [])
    if history.empty and forecast.empty:
        return _empty_figure("PEGELONLINE Kaub data are unavailable.")
    fig = go.Figure()
    if not history.empty and {"timestamp", "value"}.issubset(history.columns):
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce", utc=True)
        history["value"] = pd.to_numeric(history["value"], errors="coerce")
        history = history.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=history["timestamp"], y=history["value"], mode="lines",
                name="Observed", line={"width": 2.0, "color": TOKENS["ink"]},
                hovertemplate="%{x|%d %b %H:%M}<br>%{y:.0f} cm<extra>Observed</extra>",
            )
        )
    if not forecast.empty and {"timestamp", "value"}.issubset(forecast.columns):
        forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], errors="coerce", utc=True)
        forecast["value"] = pd.to_numeric(forecast["value"], errors="coerce")
        forecast = forecast.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=forecast["timestamp"], y=forecast["value"], mode="lines",
                name="WV forecast / estimate", line={"width": 1.7, "dash": "dash", "color": TOKENS["blue"]},
                hovertemplate="%{x|%d %b %H:%M}<br>%{y:.0f} cm<extra>Forecast</extra>",
            )
        )
    fig.update_layout(
        template="plotly_white", height=390,
        margin={"l": 50, "r": 20, "t": 30, "b": 35},
        yaxis_title="cm", dragmode="pan", hovermode="x unified",
        legend={"orientation": "h", "y": 1.06, "x": 1, "xanchor": "right"},
    )
    return fig


def climate_status_text(payload: Mapping[str, Any] | None) -> str:
    payload = dict(payload or {})
    meta = dict(payload.get("metadata") or {})
    credentials = dict(payload.get("credentials") or {})
    frame = _frame(payload)
    parts = []
    if not frame.empty:
        latest = frame["date"].max()
        parts.append(f"{GLOFAS_SOURCE_VERSION} cache through {pd.Timestamp(latest).strftime('%b %Y')}")
    else:
        parts.append("GloFAS history not built")
    kaub = dict(payload.get("kaub") or {})
    parts.append(f"Kaub: {str(kaub.get('status') or 'unavailable').replace('_', ' ')}")
    if not credentials.get("configured") or not credentials.get("url_is_ewds"):
        parts.append("EWDS credentials not ready")
    if meta.get("built_at_utc"):
        try:
            parts.append("built " + pd.Timestamp(meta["built_at_utc"]).strftime("%d %b %Y %H:%M UTC"))
        except Exception:
            pass
    return " · ".join(parts)


def register_climate_callbacks(
    app,
    *,
    project_root: str | Path,
    registry_store_id: str = "registry-store",
):
    project_root = Path(project_root)

    @app.callback(
        Output("economic-climate-store", "data"),
        Input(registry_store_id, "data"),
        Input("economic-climate-refresh", "n_clicks"),
        Input("url", "pathname"),
        State("economic-climate-store", "data"),
        prevent_initial_call=False,
    )
    def refresh_climate(registry_snapshot, _n_clicks, pathname, current):
        if (pathname or "") != "/economic-data":
            raise PreventUpdate
        snapshot_id = str((registry_snapshot or {}).get("snapshot_id") or "")
        manual = ctx.triggered_id == "economic-climate-refresh"
        if (
            not manual
            and isinstance(current, dict)
            and current.get("dashboard_registry_snapshot_id") == snapshot_id
        ):
            raise PreventUpdate
        payload = dashboard_snapshot(project_root, refresh_live=manual or not current)
        payload["dashboard_registry_snapshot_id"] = snapshot_id
        return payload

    @app.callback(
        Output("economic-climate-status", "children"),
        Output("economic-climate-kpis", "children"),
        Output("economic-climate-heatmap", "figure"),
        Output("economic-climate-kaub", "figure"),
        Input("economic-climate-store", "data"),
    )
    def render_climate(payload):
        return (
            climate_status_text(payload),
            render_kpis(payload),
            climate_heatmap_figure(payload),
            kaub_figure(payload),
        )

    @app.callback(
        Output("economic-climate-detail", "figure"),
        Input("economic-climate-store", "data"),
        Input("economic-climate-series", "value"),
    )
    def render_detail(payload, series):
        return climate_detail_figure(payload, series or SERIES_ORDER[0])


__all__ = [
    "climate_section",
    "register_climate_callbacks",
    "climate_heatmap_figure",
    "climate_detail_figure",
    "kaub_figure",
]
