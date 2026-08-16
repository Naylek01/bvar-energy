"""Headline HICP pages and pure figure builders for Inflation Dashboard slice 3.

The module consumes only the compact Headline ``display_v1.parquet`` frame held
in Dash memory. It never reopens posterior ``*.npz`` files and never performs
chain-linking in a callback.

Design contract
---------------
Overview
    Official Headline YoY history + posterior forecast fan; h=1/h=3/h=12 KPIs;
    exact additive posterior-mean contribution stack.

Contributions
    Energy/Food/NEIG/Services contributions in percentage points. Stacking uses
    posterior means, not medians, because expectation preserves the draw-wise
    additivity identity exactly.

Components
    Comparative YoY paths for the four endogenous HICP blocks with an explicit
    drill-through from Energy to the existing HICP-Energy aggregate dashboard.

Forecast
    Reuses the existing global Forecast page; no duplicate callbacks here.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from energy_bvar_theme import TOKENS, SERIES, apply_theme, graph_config

HEADLINE_SERIES = "hicp_total"
COMPONENTS = (
    "hicp_energy",
    "hicp_food",
    "hicp_neig",
    "hicp_services",
)
LABELS = {
    "hicp_total": "Headline HICP",
    "hicp_energy": "Energy",
    "hicp_food": "Food",
    "hicp_neig": "NEIG",
    "hicp_services": "Services",
}
COMPONENT_COLOURS = dict(zip(COMPONENTS, SERIES[1:5]))


def _empty_figure(message: str, *, height: int = 430) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.52,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        font={"size": 13, "color": TOKENS["muted"]},
    )
    fig.update_layout(
        template="spx_light",
        height=height,
        margin={"l": 36, "r": 18, "t": 34, "b": 28},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _rows(
    frame: pd.DataFrame,
    *,
    record_type: str,
    metric: str,
    series: str | None = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    mask = (
        frame["record_type"].astype(str).eq(record_type)
        & frame["metric"].astype(str).eq(metric)
    )
    if series is not None:
        mask &= frame["series"].astype(str).eq(series)
    return frame.loc[mask].sort_values("date").copy()


def _future_rows(
    frame: pd.DataFrame,
    metric: str,
    series: str,
    *,
    record_type: str = "fan",
) -> pd.DataFrame:
    rows = _rows(
        frame,
        record_type=record_type,
        metric=metric,
        series=series,
    )
    if rows.empty:
        return rows
    future = rows.loc[rows["is_future"].fillna(False).astype(bool)].copy()
    if future.empty:
        future = rows.loc[rows["segment"].astype(str).eq("forecast")].copy()
    return future.sort_values("date")


def _origin_date(frame: pd.DataFrame) -> pd.Timestamp | None:
    future = _future_rows(frame, "yoy", HEADLINE_SERIES)
    if future.empty:
        return None
    return pd.Timestamp(future["date"].iloc[0])


def _fan_traces(
    rows: pd.DataFrame,
    *,
    name: str,
    colour: str,
    fan_mode: str,
) -> list[go.Scatter]:
    if rows.empty:
        return []
    dates = pd.to_datetime(rows["date"])
    traces: list[go.Scatter] = []

    if fan_mode in {"90", "both"}:
        traces.extend(
            [
                go.Scatter(
                    x=dates,
                    y=rows["q95"],
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                go.Scatter(
                    x=dates,
                    y=rows["q05"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(0,159,227,0.13)",
                    hoverinfo="skip",
                    name="90%",
                    showlegend=True,
                ),
            ]
        )
    if fan_mode in {"68", "both"}:
        traces.extend(
            [
                go.Scatter(
                    x=dates,
                    y=rows["q84"],
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                go.Scatter(
                    x=dates,
                    y=rows["q16"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(0,159,227,0.26)",
                    hoverinfo="skip",
                    name="68%",
                    showlegend=True,
                ),
            ]
        )

    traces.append(
        go.Scatter(
            x=dates,
            y=rows["q50"],
            mode="lines",
            line={"color": colour, "width": 2.2},
            name=name,
        )
    )
    return traces


def headline_overview_kpis(frame: pd.DataFrame) -> dict[str, object]:
    """Return official latest YoY and h=1/h=3/h=12 posterior medians."""
    history = _rows(
        frame,
        record_type="history",
        metric="yoy",
        series=HEADLINE_SERIES,
    )
    future = _future_rows(frame, "yoy", HEADLINE_SERIES)

    out: dict[str, object] = {
        "observed": np.nan,
        "observed_date": None,
        "h1": np.nan,
        "h1_date": None,
        "h3": np.nan,
        "h3_date": None,
        "h12": np.nan,
        "h12_date": None,
        "horizon": int(len(future)),
    }
    if not history.empty:
        row = history.iloc[-1]
        out["observed"] = float(row["value"])
        out["observed_date"] = pd.Timestamp(row["date"])
    if not future.empty:
        for horizon, key in ((1, "h1"), (3, "h3"), (12, "h12")):
            index = min(horizon, len(future)) - 1
            row = future.iloc[index]
            out[key] = float(row["q50"])
            out[f"{key}_date"] = pd.Timestamp(row["date"])
    return out


def headline_overview_figure(
    frame: pd.DataFrame,
    *,
    fan_mode: str = "both",
    uirevision: str = "headline-overview",
) -> go.Figure:
    history = _rows(
        frame,
        record_type="history",
        metric="yoy",
        series=HEADLINE_SERIES,
    )
    fan = _rows(
        frame,
        record_type="fan",
        metric="yoy",
        series=HEADLINE_SERIES,
    )
    if history.empty and fan.empty:
        return _empty_figure("Select a saved Headline run.")

    fig = go.Figure()
    if not history.empty:
        history = history.iloc[-120:].copy()
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["value"],
                mode="lines",
                line={"color": TOKENS["ink_soft"], "width": 1.6},
                name="Official Headline HICP",
            )
        )
    for trace in _fan_traces(
        fan,
        name="Posterior median",
        colour=TOKENS["cyan"],
        fan_mode=fan_mode,
    ):
        fig.add_trace(trace)

    origin = _origin_date(frame)
    if origin is not None:
        fig.add_vline(
            x=origin,
            line_width=1,
            line_dash="dash",
            line_color=TOKENS["muted"],
        )

    apply_theme(
        fig,
        uirevision=uirevision,
        height=460,
        y_title="% y/y",
    )
    fig.update_layout(title="Headline HICP inflation")
    return fig


def contribution_mean_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Future posterior-mean contribution table with an exact additivity audit."""
    blocks = []
    for component in COMPONENTS:
        rows = _future_rows(
            frame,
            "yoy_contribution",
            component,
            record_type="contribution",
        )
        if rows.empty:
            continue
        block = rows[["date", "value"]].rename(columns={"value": component})
        blocks.append(block.set_index("date"))

    if len(blocks) != len(COMPONENTS):
        return pd.DataFrame()

    table = pd.concat(blocks, axis=1).sort_index()
    headline = _future_rows(frame, "yoy", HEADLINE_SERIES)
    if headline.empty:
        return pd.DataFrame()
    headline = headline.set_index("date")["value"].rename("headline_mean")
    table = table.join(headline, how="inner")
    table["component_sum"] = table[list(COMPONENTS)].sum(axis=1)
    table["additivity_error"] = table["component_sum"] - table["headline_mean"]
    table.index = pd.DatetimeIndex(table.index, name="date")
    return table


def headline_contribution_stack_figure(
    frame: pd.DataFrame,
    *,
    uirevision: str = "headline-contributions-stack",
) -> go.Figure:
    table = contribution_mean_table(frame)
    if table.empty:
        return _empty_figure("Contribution paths are not available for this run.")

    fig = go.Figure()
    for component in COMPONENTS:
        fig.add_trace(
            go.Bar(
                x=table.index,
                y=table[component],
                name=LABELS[component],
                marker_color=COMPONENT_COLOURS[component],
            )
        )
    fig.add_trace(
        go.Scatter(
            x=table.index,
            y=table["headline_mean"],
            mode="lines+markers",
            line={"color": TOKENS["ink"], "width": 2.0},
            marker={"size": 5},
            name="Headline posterior mean",
        )
    )
    apply_theme(
        fig,
        uirevision=uirevision,
        height=430,
        y_title="percentage points",
    )
    fig.update_layout(
        barmode="relative",
        title="Contributions to Headline inflation — posterior means",
    )
    return fig


def headline_component_contribution_figure(
    frame: pd.DataFrame,
    component: str,
    *,
    fan_mode: str = "68",
    uirevision: str = "headline-contribution-detail",
) -> go.Figure:
    if component not in COMPONENTS:
        return _empty_figure("Select a Headline component.")
    rows = _rows(
        frame,
        record_type="contribution",
        metric="yoy_contribution",
        series=component,
    )
    if rows.empty:
        return _empty_figure("Contribution uncertainty is not available.")

    fig = go.Figure()
    for trace in _fan_traces(
        rows,
        name=LABELS[component],
        colour=COMPONENT_COLOURS[component],
        fan_mode=fan_mode,
    ):
        fig.add_trace(trace)
    fig.add_hline(
        y=0,
        line_width=1,
        line_color=TOKENS["hairline"],
    )
    origin = _origin_date(frame)
    if origin is not None:
        fig.add_vline(
            x=origin,
            line_width=1,
            line_dash="dash",
            line_color=TOKENS["muted"],
        )
    apply_theme(
        fig,
        uirevision=f"{uirevision}:{component}",
        height=400,
        y_title="percentage points",
    )
    fig.update_layout(title=f"{LABELS[component]} contribution uncertainty")
    return fig


def contribution_table_records(frame: pd.DataFrame) -> list[dict]:
    table = contribution_mean_table(frame)
    if table.empty:
        return []
    out = table.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    rename = {
        "hicp_energy": "energy",
        "hicp_food": "food",
        "hicp_neig": "neig",
        "hicp_services": "services",
    }
    out = out.rename(columns=rename)
    for column in (
        "energy",
        "food",
        "neig",
        "services",
        "component_sum",
        "headline_mean",
        "additivity_error",
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce").round(6)
    return out.to_dict("records")


def headline_components_figure(
    frame: pd.DataFrame,
    *,
    uirevision: str = "headline-components",
) -> go.Figure:
    fig = go.Figure()
    any_rows = False

    for component in COMPONENTS:
        history = _rows(
            frame,
            record_type="history",
            metric="yoy",
            series=component,
        )
        future = _future_rows(frame, "yoy", component)
        colour = COMPONENT_COLOURS[component]

        if not history.empty:
            history = history.iloc[-120:].copy()
            any_rows = True
            fig.add_trace(
                go.Scatter(
                    x=history["date"],
                    y=history["value"],
                    mode="lines",
                    line={"color": colour, "width": 1.2},
                    opacity=0.55,
                    name=f"{LABELS[component]} observed",
                    legendgroup=component,
                    showlegend=False,
                )
            )
        if not future.empty:
            any_rows = True
            fig.add_trace(
                go.Scatter(
                    x=future["date"],
                    y=future["q50"],
                    mode="lines+markers",
                    line={"color": colour, "width": 2.0},
                    marker={"size": 4},
                    name=LABELS[component],
                    legendgroup=component,
                )
            )

    if not any_rows:
        return _empty_figure("Component paths are not available.")

    origin = _origin_date(frame)
    if origin is not None:
        fig.add_vline(
            x=origin,
            line_width=1,
            line_dash="dash",
            line_color=TOKENS["muted"],
        )
    apply_theme(
        fig,
        uirevision=uirevision,
        height=460,
        y_title="% y/y",
    )
    fig.update_layout(title="Headline components — YoY inflation")
    return fig


def component_terminal_table(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for component in COMPONENTS:
        future = _future_rows(frame, "yoy", component)
        if future.empty:
            continue
        first = future.iloc[0]
        terminal = future.iloc[-1]
        rows.append(
            {
                "component": LABELS[component],
                "first_date": pd.Timestamp(first["date"]).strftime("%Y-%m-%d"),
                "first_median": round(float(first["q50"]), 4),
                "terminal_date": pd.Timestamp(terminal["date"]).strftime("%Y-%m-%d"),
                "terminal_median": round(float(terminal["q50"]), 4),
            }
        )
    return rows


def _stat_card(title: str, value_id: str, subtitle_id: str):
    from dash import html
    return html.Div(
        [
            html.Div(title, className="stat-title"),
            html.Div("—", id=value_id, className="stat-value"),
            html.Div("", id=subtitle_id, className="stat-subtitle"),
        ],
        className="stat-card",
    )


def headline_overview_page():
    from dash import dcc, html
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Headline HICP", className="page-title"),
                            html.P(
                                "Official history and the locked joint BVAR(12) bottom-up forecast. Publication horizon is hard-capped at 12 months.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    dcc.RadioItems(
                        id="headline-overview-fan",
                        options=[
                            {"label": "68%", "value": "68"},
                            {"label": "90%", "value": "90"},
                            {"label": "Both", "value": "both"},
                        ],
                        value="both",
                        inline=True,
                        className="fan-radio",
                    ),
                ],
                className="page-heading-row",
            ),
            html.Div(
                [
                    _stat_card("Latest observed", "headline-kpi-observed", "headline-kpi-observed-date"),
                    _stat_card("1 month", "headline-kpi-h1", "headline-kpi-h1-date"),
                    _stat_card("3 months", "headline-kpi-h3", "headline-kpi-h3-date"),
                    _stat_card("12 months", "headline-kpi-h12", "headline-kpi-h12-date"),
                ],
                className="stats-grid",
            ),
            html.Div(
                dcc.Loading(
                    dcc.Graph(
                        id="headline-overview-graph",
                        config=graph_config("headline_hicp_forecast"),
                    ),
                    type="circle",
                ),
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("What drives the forecast", className="panel-title"),
                            html.P(
                                "Posterior-mean Energy, Food, NEIG and Services contributions. Means are used so the stack remains exactly additive to the posterior-mean Headline path.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dcc.Graph(
                        id="headline-overview-contributions",
                        config=graph_config("headline_contributions"),
                    ),
                ],
                className="panel chart-panel",
            ),
        ],
        className="page-body",
    )


def headline_contributions_page():
    from dash import dash_table, dcc, html
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Headline contributions", className="page-title"),
                            html.P(
                                "Exact draw-wise component attribution. Posterior means are stacked for additivity; component uncertainty remains available separately.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Component", className="control-label"),
                                    dcc.Dropdown(
                                        id="headline-contrib-series",
                                        options=[
                                            {"label": LABELS[x], "value": x}
                                            for x in COMPONENTS
                                        ],
                                        value="hicp_energy",
                                        clearable=False,
                                        className="compact-dropdown wide-control",
                                    ),
                                ],
                                className="control-block wide-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Fan", className="control-label"),
                                    dcc.RadioItems(
                                        id="headline-contrib-fan",
                                        options=[
                                            {"label": "68%", "value": "68"},
                                            {"label": "90%", "value": "90"},
                                            {"label": "Both", "value": "both"},
                                        ],
                                        value="68",
                                        inline=True,
                                        className="fan-radio",
                                    ),
                                ],
                                className="control-block",
                            ),
                        ],
                        className="chart-controls",
                    ),
                ],
                className="page-heading-row",
            ),
            html.Div(
                dcc.Graph(
                    id="headline-contrib-stack",
                    config=graph_config("headline_contribution_stack"),
                ),
                className="panel chart-panel",
            ),
            html.Div(
                dcc.Graph(
                    id="headline-contrib-detail",
                    config=graph_config("headline_contribution_detail"),
                ),
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Additive forecast table", className="panel-title"),
                            html.P(
                                "All entries are posterior means in percentage points. Component sum minus Headline should be numerical zero.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
                        id="headline-contrib-table",
                        columns=[
                            {"name": "Date", "id": "date"},
                            {"name": "Energy", "id": "energy", "type": "numeric"},
                            {"name": "Food", "id": "food", "type": "numeric"},
                            {"name": "NEIG", "id": "neig", "type": "numeric"},
                            {"name": "Services", "id": "services", "type": "numeric"},
                            {"name": "Component sum", "id": "component_sum", "type": "numeric"},
                            {"name": "Headline mean", "id": "headline_mean", "type": "numeric"},
                            {"name": "Error", "id": "additivity_error", "type": "numeric"},
                        ],
                        data=[],
                        page_size=12,
                        sort_action="native",
                        style_as_list_view=True,
                        style_cell={"fontFamily": "Inter, Segoe UI, sans-serif"},
                    ),
                ],
                className="panel table-panel",
            ),
        ],
        className="page-body",
    )


def headline_components_page():
    from dash import dash_table, dcc, html
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Headline components", className="page-title"),
                            html.P(
                                "Joint Energy, Food, NEIG and Services dynamics from the same posterior forecast.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    dcc.Link(
                        "Open detailed Energy dashboard →",
                        href="/aggregate",
                        className="refresh-button headline-energy-link",
                    ),
                ],
                className="page-heading-row",
            ),
            html.Div(
                dcc.Graph(
                    id="headline-components-graph",
                    config=graph_config("headline_components"),
                ),
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Component forecast endpoints", className="panel-title"),
                            html.P(
                                "First and terminal posterior-median YoY rates. Energy drill-through opens the existing seven-model HICP Energy aggregate.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
                        id="headline-components-table",
                        columns=[
                            {"name": "Component", "id": "component"},
                            {"name": "First date", "id": "first_date"},
                            {"name": "First median", "id": "first_median", "type": "numeric"},
                            {"name": "Terminal date", "id": "terminal_date"},
                            {"name": "Terminal median", "id": "terminal_median", "type": "numeric"},
                        ],
                        data=[],
                        page_size=8,
                        sort_action="native",
                        style_as_list_view=True,
                        style_cell={"fontFamily": "Inter, Segoe UI, sans-serif"},
                    ),
                ],
                className="panel table-panel",
            ),
        ],
        className="page-body",
    )



# ---------------------------------------------------------------------------
# Slice 4 — Structural analysis and diagnostics
# ---------------------------------------------------------------------------

STATE_LABELS = {
    "state__hicp_energy": "Energy",
    "state__hicp_food": "Food",
    "state__hicp_neig": "NEIG",
    "state__hicp_services": "Services",
}
STATE_COMPONENTS = tuple(STATE_LABELS)


def _structural_rows(
    frame: pd.DataFrame,
    *,
    record_type: str,
    metric: str | None = None,
    response: str | None = None,
    shock: str | None = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    mask = frame["record_type"].astype(str).eq(record_type)
    if metric is not None:
        mask &= frame["metric"].astype(str).eq(metric)
    if response is not None:
        mask &= frame["response"].astype(str).eq(response)
    if shock is not None:
        mask &= frame["shock"].astype(str).eq(shock)
    return frame.loc[mask].copy()


def headline_irf_figure(
    frame: pd.DataFrame,
    *,
    response: str = "state__hicp_services",
    shock: str = "state__hicp_energy",
    metric: str = "cumulative_log_change",
    fan_mode: str = "both",
    uirevision: str = "headline-structural-irf",
) -> go.Figure:
    rows = _structural_rows(
        frame,
        record_type="structural_irf",
        metric=metric,
        response=response,
        shock=shock,
    ).sort_values("horizon")
    if rows.empty:
        return _empty_figure("Structural IRFs have not been materialised for this run.")

    x = pd.to_numeric(rows["horizon"], errors="coerce")
    fig = go.Figure()
    if fan_mode in {"90", "both"}:
        fig.add_trace(go.Scatter(x=x, y=rows["q95"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=rows["q05"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(0,159,227,0.13)", name="90%"))
    if fan_mode in {"68", "both"}:
        fig.add_trace(go.Scatter(x=x, y=rows["q84"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=rows["q16"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(0,159,227,0.26)", name="68%"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=rows["q50"],
            mode="lines+markers",
            line={"color": TOKENS["cyan"], "width": 2.1},
            marker={"size": 4},
            name="Posterior median",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    y_title = "100 × cumulative Δlog" if metric == "cumulative_log_change" else "100 × Δlog points"
    apply_theme(fig, uirevision=uirevision, height=430, y_title=y_title)
    fig.update_layout(
        title=f"IRF — {STATE_LABELS.get(shock, shock)} shock → {STATE_LABELS.get(response, response)}",
        xaxis_title="Months after shock",
    )
    return fig


def headline_fevd_figure(
    frame: pd.DataFrame,
    *,
    response: str = "state__hicp_services",
    uirevision: str = "headline-structural-fevd",
) -> go.Figure:
    rows = _structural_rows(
        frame,
        record_type="structural_fevd",
        metric="share",
        response=response,
    )
    if rows.empty:
        return _empty_figure("FEVD has not been materialised for this run.")
    fig = go.Figure()
    for i, shock in enumerate(STATE_COMPONENTS):
        block = rows.loc[rows["shock"].astype(str).eq(shock)].sort_values("horizon")
        if block.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=block["horizon"],
                y=block["q50"],
                mode="lines",
                line={"width": 0.8, "color": SERIES[i + 1]},
                stackgroup="fevd",
                name=STATE_LABELS[shock],
            )
        )
    apply_theme(fig, uirevision=uirevision, height=420, y_title="share")
    fig.update_layout(
        title=f"Forecast error variance decomposition — posterior median — {STATE_LABELS.get(response, response)}",
        xaxis_title="Forecast horizon (months)",
        yaxis={"range": [0, 1], "tickformat": ".0%"},
    )
    return fig


def headline_hd_figure(
    frame: pd.DataFrame,
    *,
    response: str = "state__hicp_services",
    uirevision: str = "headline-structural-hd",
) -> go.Figure:
    contrib = _structural_rows(
        frame,
        record_type="structural_hd",
        metric="shock_contribution_12m_log",
        response=response,
    )
    observed = _structural_rows(
        frame,
        record_type="structural_hd",
        metric="observed_12m_log",
        response=response,
    ).sort_values("date")
    base = _structural_rows(
        frame,
        record_type="structural_hd",
        metric="base_12m_log",
        response=response,
    ).sort_values("date")
    if contrib.empty or observed.empty:
        return _empty_figure("Historical decomposition has not been materialised for this run.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=observed["date"],
            y=observed["value"],
            mode="lines",
            line={"color": TOKENS["ink"], "width": 2.0},
            name="Observed 12m log inflation",
        )
    )
    if not base.empty:
        fig.add_trace(
            go.Scatter(
                x=base["date"],
                y=base["q50"],
                mode="lines",
                line={"color": TOKENS["muted"], "dash": "dash", "width": 1.4},
                name="Base: constant + seasonals + initial dynamics",
            )
        )
    for i, shock in enumerate(STATE_COMPONENTS):
        block = contrib.loc[contrib["shock"].astype(str).eq(shock)].sort_values("date")
        if block.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=block["date"],
                y=block["q50"],
                mode="lines",
                line={"color": SERIES[i + 1], "width": 1.5},
                name=f"{STATE_LABELS[shock]} shock",
            )
        )
    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    apply_theme(
        fig,
        uirevision=uirevision,
        height=455,
        y_title="12m contribution (100 × log points)",
    )
    fig.update_layout(
        title=f"Recursive historical decomposition — {STATE_LABELS.get(response, response)}",
    )
    return fig


def root_diagnostic_records(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    roots = frame.loc[frame["record_type"].astype(str).eq("root")].copy()
    if roots.empty:
        return []
    order = {name: i for i, name in enumerate(("zero", "12m", "6m", "4m", "3m", "2.4m", "2m"))}
    roots["_order"] = roots["series"].map(order).fillna(999)
    roots = roots.sort_values("_order")
    out = []
    for _, row in roots.iterrows():
        out.append(
            {
                "frequency": str(row["series"]),
                "q05": _round_or_none(row.get("q05"), 3),
                "median": _round_or_none(row.get("q50"), 3),
                "q95": _round_or_none(row.get("q95"), 3),
                "p90": _round_or_none(row.get("prob_90"), 3),
                "p95": _round_or_none(row.get("prob_95"), 3),
                "p97": _round_or_none(row.get("prob_97"), 3),
            }
        )
    return out


def _round_or_none(value, digits: int = 4):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(number) else round(number, digits)


def mcmc_diagnostic_records(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    rows = frame.loc[
        frame["record_type"].astype(str).eq("diagnostic")
        & frame["metric"].astype(str).eq("mcmc")
    ].copy()
    out = []
    for _, row in rows.iterrows():
        out.append(
            {
                "parameter": str(row.get("key") or row.get("series") or ""),
                "mean": _round_or_none(row.get("posterior_mean"), 5),
                "sd": _round_or_none(row.get("posterior_sd"), 5),
                "ess": _round_or_none(row.get("ess"), 1),
                "mcse_sd": _round_or_none(row.get("mcse_over_sd"), 4),
            }
        )
    return out


def validation_diagnostic_records(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    rows = frame.loc[
        frame["record_type"].astype(str).eq("diagnostic")
        & frame["metric"].astype(str).eq("reconstruction")
    ].copy()
    return [
        {"metric": str(row.get("key") or row.get("series") or ""), "value": _round_or_none(row.get("value"), 8)}
        for _, row in rows.iterrows()
    ]


def stability_diagnostic_kpis(frame: pd.DataFrame) -> dict[str, object]:
    if frame is None or frame.empty:
        return {}
    rows = frame.loc[
        frame["record_type"].astype(str).eq("diagnostic")
        & frame["metric"].astype(str).eq("stability")
    ]
    values = {}
    for _, row in rows.iterrows():
        key = str(row.get("key") or row.get("series") or "")
        values[key] = _round_or_none(row.get("value"), 5)
    meta = frame.loc[
        frame["record_type"].astype(str).eq("structural_meta")
    ]
    for _, row in meta.iterrows():
        key = str(row.get("key") or "")
        if key in {"hd_max_reconstruction_error", "fevd_max_share_sum_error", "posterior_draws_used"}:
            values[key] = _round_or_none(row.get("value"), 10)
    return values


def headline_structural_page():
    from dash import dcc, html

    options = [{"label": STATE_LABELS[x], "value": x} for x in STATE_COMPONENTS]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Structural analysis", className="page-title"),
                            html.P(
                                "Recursive identification on the locked four-variable BVAR. IRFs and FEVD use the regular SV state at the final estimation date; horizon is capped at 12 months.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span("Recursive", className="headline-lock-badge"),
                            html.Span("Energy → Food → NEIG → Services", className="headline-order-badge"),
                            html.Span("H ≤ 12m", className="headline-lock-badge"),
                        ],
                        className="headline-badge-row",
                    ),
                ],
                className="page-heading-row",
            ),
            html.Div(
                [
                    html.Div([html.Label("Shock", className="control-label"), dcc.Dropdown(id="headline-structural-shock", options=options, value="state__hicp_energy", clearable=False, className="compact-dropdown wide-control")], className="control-block wide-control"),
                    html.Div([html.Label("Response", className="control-label"), dcc.Dropdown(id="headline-structural-response", options=options, value="state__hicp_services", clearable=False, className="compact-dropdown wide-control")], className="control-block wide-control"),
                    html.Div([html.Label("IRF metric", className="control-label"), dcc.Dropdown(id="headline-structural-irf-metric", options=[{"label": "Cumulative log-price effect", "value": "cumulative_log_change"}, {"label": "Monthly log-change effect", "value": "monthly_log_change"}], value="cumulative_log_change", clearable=False, className="compact-dropdown wide-control")], className="control-block wide-control"),
                    html.Div([html.Label("Fan", className="control-label"), dcc.RadioItems(id="headline-structural-fan", options=[{"label": "68%", "value": "68"}, {"label": "90%", "value": "90"}, {"label": "Both", "value": "both"}], value="both", inline=True, className="fan-radio")], className="control-block"),
                ],
                className="chart-controls",
            ),
            html.Div(dcc.Graph(id="headline-structural-irf", config=graph_config("headline_irf")), className="panel chart-panel"),
            html.Div(dcc.Graph(id="headline-structural-fevd", config=graph_config("headline_fevd")), className="panel chart-panel"),
            html.Div(
                [
                    html.Div([html.H3("Historical decomposition", className="panel-title"), html.P("Posterior-median 12-month log-inflation contributions, matching the locked notebook. Regular and transitory outlier-amplification pieces are recombined by structural shock.", className="panel-subtitle")], className="panel-heading"),
                    dcc.Graph(id="headline-structural-hd", config=graph_config("headline_hd")),
                ],
                className="panel chart-panel",
            ),
        ],
        className="page-body",
    )



def _diag_stat_card(title: str, value_id: str, subtitle: str):
    from dash import html
    return html.Div(
        [
            html.Div(title, className="stat-title"),
            html.Div("—", id=value_id, className="stat-value"),
            html.Div(subtitle, className="stat-subtitle"),
        ],
        className="stat-card",
    )

def headline_diagnostics_page():
    from dash import dash_table, html

    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.H2("Headline diagnostics", className="page-title"), html.P("Saved-run MCMC precision, stability, accounting reconstruction and posterior companion-root diagnostics for the locked production specification.", className="page-subtitle")]),
                    html.Div([html.Span("LOCKED", className="headline-lock-badge"), html.Span("BVAR(12) + 11 month dummies", className="headline-order-badge"), html.Span("Eurostat FOOD", className="headline-order-badge")], className="headline-badge-row"),
                ],
                className="page-heading-row",
            ),
            html.Div(
                [
                    _diag_stat_card("Median spectral radius", "headline-diag-radius", "retained posterior"),
                    _diag_stat_card("B instability rejection", "headline-diag-rejection", "stability-truncated B draws"),
                    _diag_stat_card("HD reconstruction error", "headline-diag-hd-error", "draw-by-draw maximum"),
                    _diag_stat_card("FEVD sum error", "headline-diag-fevd-error", "share-sum identity"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    html.Div([html.H3("Posterior roots by frequency", className="panel-title"), html.P("Stable near-roots at 6m and 3m are monitored rather than mechanically removed. The publication horizon remains capped at 12 months.", className="panel-subtitle")], className="panel-heading"),
                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", id="headline-root-table", columns=[{"name": "Frequency", "id": "frequency"}, {"name": "q05", "id": "q05", "type": "numeric"}, {"name": "Median", "id": "median", "type": "numeric"}, {"name": "q95", "id": "q95", "type": "numeric"}, {"name": "P(|λ|>.90)", "id": "p90", "type": "numeric"}, {"name": "P(|λ|>.95)", "id": "p95", "type": "numeric"}, {"name": "P(|λ|>.97)", "id": "p97", "type": "numeric"}], data=[], style_as_list_view=True, style_cell={"fontFamily": "Inter, Segoe UI, sans-serif"}, style_data_conditional=[{"if": {"filter_query": '{frequency} = "6m" || {frequency} = "3m"'}, "fontWeight": "700"}]),
                ],
                className="panel table-panel",
            ),
            html.Div(
                [
                    html.Div([html.H3("MCMC precision", className="panel-title"), html.P("Single-chain ESS and Monte Carlo standard error diagnostics for key SV/outlier/VAR parameters. No synthetic R-hat is reported.", className="panel-subtitle")], className="panel-heading"),
                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", id="headline-mcmc-table", columns=[{"name": "Parameter", "id": "parameter"}, {"name": "Posterior mean", "id": "mean", "type": "numeric"}, {"name": "Posterior sd", "id": "sd", "type": "numeric"}, {"name": "ESS", "id": "ess", "type": "numeric"}, {"name": "MCSE / sd", "id": "mcse_sd", "type": "numeric"}], data=[], page_size=14, sort_action="native", style_as_list_view=True, style_cell={"fontFamily": "Inter, Segoe UI, sans-serif"}),
                ],
                className="panel table-panel",
            ),
            html.Div(
                [
                    html.Div([html.H3("Model-selection record", className="panel-title"), html.P("Frozen evidence behind the locked production specification.", className="panel-subtitle")], className="panel-heading"),
                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
                        columns=[
                            {"name": "Prior-root control", "id": "metric"},
                            {"name": "Value", "id": "value"},
                        ],
                        data=[
                            {"metric": "Median spectral radius", "value": "0.821"},
                            {"metric": "P(radius > 0.95)", "value": "0.002"},
                            {"metric": "P(radius > 0.97)", "value": "0.000"},
                            {"metric": "6m P(|λ| > 0.97)", "value": "0.000"},
                            {"metric": "3m P(|λ| > 0.97)", "value": "0.000"},
                        ],
                        style_as_list_view=True,
                        style_cell={"fontFamily": "Inter, Segoe UI, sans-serif"},
                    ),
                    html.Div(
                        [
                            html.Strong("Decision: "),
                            html.Span("The targeted 2015/2020 seasonal-break candidate was rejected because the 6m/3m roots were essentially unchanged. BVAR(6) and additional seasonal differencing were not promoted. The 6m/3m posterior persistence is monitored as data-driven dynamics."),
                        ],
                        className="headline-method-note",
                    ),
                ],
                className="panel table-panel",
            ),
            html.Div(
                [
                    html.Div([html.H3("Accounting reconstruction", className="panel-title"), html.P("Headline history reconstructed from Energy/Food/NEIG/Services and annual Eurostat weights. These are data/aggregation diagnostics, not sampler diagnostics.", className="panel-subtitle")], className="panel-heading"),
                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", id="headline-validation-table", columns=[{"name": "Metric", "id": "metric"}, {"name": "Value", "id": "value", "type": "numeric"}], data=[], page_size=12, style_as_list_view=True, style_cell={"fontFamily": "Inter, Segoe UI, sans-serif"}),
                    html.Div(
                        [
                            html.Strong("Specification record: "),
                            html.Span("11 fixed monthly dummies retained; 2015/2020 seasonal-break interactions were tested but not promoted. Prior-root control did not generate the observed 6m/3m persistence, so those modes are documented as posterior dynamics rather than prior artefacts."),
                        ],
                        className="headline-method-note",
                    ),
                ],
                className="panel table-panel",
            ),
        ],
        className="page-body",
    )

__all__ = [
    "HEADLINE_SERIES",
    "COMPONENTS",
    "LABELS",
    "STATE_COMPONENTS",
    "STATE_LABELS",
    "headline_overview_kpis",
    "headline_overview_figure",
    "contribution_mean_table",
    "headline_contribution_stack_figure",
    "headline_component_contribution_figure",
    "contribution_table_records",
    "headline_components_figure",
    "component_terminal_table",
    "headline_irf_figure",
    "headline_fevd_figure",
    "headline_hd_figure",
    "root_diagnostic_records",
    "mcmc_diagnostic_records",
    "validation_diagnostic_records",
    "stability_diagnostic_kpis",
    "headline_overview_page",
    "headline_contributions_page",
    "headline_components_page",
    "headline_structural_page",
    "headline_diagnostics_page",
]
