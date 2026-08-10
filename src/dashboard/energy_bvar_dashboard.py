"""Dash front-end for the ECB-style Energy BVAR suite.

Dashboard foundation
---------------------
* persistent URL routing;
* central context store: vintage / model_id / run_id / forecast_name / draw_mode;
* SQLite-backed selectors with promoted-run awareness;
* exactly one ``display_v1.parquet`` read when the selected run changes;
* Forecast page driven exclusively by the in-memory display store;
* 68%, 90%, or combined posterior fans with ``uirevision``;
* observed / nowcast / forecast segmentation and a compact values table;
* Diskcache background manager initialised for the estimation/structural pages
  added in the next dashboard slices.

The econometric code remains in ``src/model``.  This module only reads registry
and display artefacts and never re-estimates a model from a plotting callback.
"""

from __future__ import annotations

import json
import os
from io import StringIO
import sys
from pathlib import Path
from typing import Any

import diskcache
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import (
    ALL,
    Dash,
    DiskcacheManager,
    Input,
    Output,
    State,
    callback,
    ctx,
    dash_table,
    dcc,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate


# ---------------------------------------------------------------------------
# Project / model-library discovery
# ---------------------------------------------------------------------------


def _find_project_root(start: Path) -> Path:
    explicit = os.getenv("ENERGY_BVAR_PROJECT_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "model").is_dir() and (candidate / "data").exists():
            return candidate
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "model").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).parent)
MODEL_DIR = PROJECT_ROOT / "src" / "model"
if not MODEL_DIR.is_dir():
    # Also supports placing this file directly beside the model modules while
    # the dashboard structure is being introduced.
    MODEL_DIR = PROJECT_ROOT
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from energy_bvar_dashboard_aggregate import (  # noqa: E402
    aggregate_contribution_figure,
    aggregate_diagnostics_table,
    aggregate_fan_figure,
    aggregate_kpis,
    aggregate_metric_options,
    aggregate_live_scenario_payload,
    aggregate_live_scenario_figure,
    aggregate_live_impact_figure,
    aggregate_live_contribution_impact_figure,
    aggregate_live_scenario_kpis,
    empty_aggregate_figure,
)
from energy_bvar_dashboard_scenarios import (  # noqa: E402
    clear_scenario_set,
    empty_scenario_figure,
    remove_scenario_component,
    scenario_impact_figure,
    scenario_kpis,
    scenario_main_figure,
    scenario_payload,
    scenario_set_components,
    scenario_set_payload,
    scenario_set_signature,
    scenario_set_summary,
    scenario_set_to_tax_scenarios,
    scenario_tax_figure,
    upsert_scenario_component,
)
from energy_bvar_component_hicp import (  # noqa: E402
    TAX_SCENARIO_MODEL_IDS,
    build_saved_component_tax_scenario,
    component_tax_scenario_contract,
)
from energy_bvar_aggregate_pipeline import run_aggregate  # noqa: E402
from energy_bvar_display import (  # noqa: E402
    DISPLAY_FILENAME,
    build_aggregate_display,
    build_component_display,
    display_metadata,
    load_display_artifact,
)
from energy_bvar_pipeline import model_spec  # noqa: E402
from energy_bvar_registry import (  # noqa: E402
    default_registry_path,
    init_registry,
    list_aggregates,
    list_forecasts,
    list_runs,
    scan_results,
)

RESULTS_ROOT = Path(
    os.getenv("ENERGY_BVAR_RESULTS_ROOT", str(PROJECT_ROOT / "results"))
).expanduser().resolve()
REGISTRY_PATH = Path(
    os.getenv(
        "ENERGY_BVAR_REGISTRY",
        str(default_registry_path(results_root=RESULTS_ROOT)),
    )
).expanduser().resolve()
AUTO_BUILD_DISPLAY = os.getenv("ENERGY_BVAR_AUTO_BUILD_DISPLAY", "1") not in {
    "0",
    "false",
    "False",
}

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
init_registry(REGISTRY_PATH)

CACHE_DIR = Path(
    os.getenv("ENERGY_BVAR_DASH_CACHE", str(RESULTS_ROOT / ".dash_cache"))
).expanduser().resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_diskcache = diskcache.Cache(str(CACHE_DIR))
background_callback_manager = DiskcacheManager(_diskcache)


# ---------------------------------------------------------------------------
# Small data helpers
# ---------------------------------------------------------------------------


def _scan_registry() -> dict[str, Any]:
    try:
        report = scan_results(results_root=RESULTS_ROOT, registry_path=REGISTRY_PATH)
        return {
            "ok": True,
            "revision": pd.Timestamp.utcnow().isoformat(),
            "message": (
                f"{report.component_runs_seen} runs · {report.forecasts_seen} forecasts · "
                f"{report.aggregates_seen} aggregates"
            ),
            "unexpected": list(report.unexpected_directories),
        }
    except Exception as exc:  # surfaced in the UI, not swallowed
        return {
            "ok": False,
            "revision": pd.Timestamp.utcnow().isoformat(),
            "message": f"Registry scan failed: {exc}",
            "unexpected": [],
        }


def _json_frame(frame: pd.DataFrame) -> str:
    return frame.to_json(orient="split", date_format="iso", double_precision=15)


def _frame_from_store(store: dict | None) -> pd.DataFrame:
    if not store or not store.get("frame_json"):
        return pd.DataFrame()
    frame = pd.read_json(StringIO(store["frame_json"]), orient="split")
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _short_run(run_id: str) -> str:
    run_id = str(run_id)
    return run_id if len(run_id) <= 12 else run_id[:12]


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if not np.isfinite(value):
        return "—"
    absolute = abs(value)
    if absolute != 0 and (absolute < 0.01 or absolute >= 10_000):
        return f"{value:.2e}"
    return f"{value:,.2f}"


def _forecast_rows(frame: pd.DataFrame, metric: str, series: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = (
        (frame["record_type"] == "fan")
        & (frame["metric"] == metric)
        & (frame["series"] == series)
    )
    return frame.loc[mask].sort_values("date").copy()


def _history_rows(frame: pd.DataFrame, metric: str, series: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = (
        (frame["record_type"] == "history")
        & (frame["metric"] == metric)
        & (frame["series"] == series)
    )
    return frame.loc[mask].sort_values("date").copy()


def _rgba(hex_color: str, opacity: float) -> str:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{opacity})"


def _add_band(
    fig: go.Figure,
    fan: pd.DataFrame,
    lower: str,
    upper: str,
    name: str,
    *,
    opacity: float,
    color: str = "#2563eb",
) -> None:
    if fan.empty:
        return
    x = pd.concat([fan["date"], fan["date"].iloc[::-1]], ignore_index=True)
    y = pd.concat([fan[upper], fan[lower].iloc[::-1]], ignore_index=True)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="toself",
            fillcolor=_rgba(color, opacity),
            line={"width": 0},
            hoverinfo="skip",
            name=name,
            legendgroup="fan",
        )
    )


def _anchor_fan_line(block: pd.DataFrame, date, value) -> pd.DataFrame:
    if block.empty or date is None or pd.isna(date) or value is None or pd.isna(value):
        return block
    anchor = {column: np.nan for column in block.columns}
    anchor["date"] = pd.Timestamp(date)
    for column in ("q05", "q16", "q50", "q84", "q95"):
        if column in block.columns:
            anchor[column] = float(value)
    return pd.concat([pd.DataFrame([anchor]), block], ignore_index=True).sort_values("date")

def forecast_figure(
    frame: pd.DataFrame,
    *,
    metric: str,
    series: str,
    fan_mode: str,
    context: dict,
) -> go.Figure:
    """Continuous observed -> nowcast -> forecast chart with distinct nowcast styling."""
    history = _history_rows(frame, metric, series)
    fan = _forecast_rows(frame, metric, series)
    nowcast = fan.loc[fan["segment"].astype(str) == "nowcast"].copy() if not fan.empty else fan
    forecast = fan.loc[fan["segment"].astype(str) == "forecast"].copy() if not fan.empty else fan
    fig = go.Figure()

    now_color = "#A66A00"
    fc_color = "#2563eb"
    if fan_mode in {"90", "both"}:
        _add_band(fig, nowcast, "q05", "q95", "Nowcast 90% interval", opacity=0.09, color=now_color)
        _add_band(fig, forecast, "q05", "q95", "Forecast 90% interval", opacity=0.11, color=fc_color)
    if fan_mode in {"68", "both"}:
        _add_band(fig, nowcast, "q16", "q84", "Nowcast 68% interval", opacity=0.18, color=now_color)
        _add_band(fig, forecast, "q16", "q84", "Forecast 68% interval", opacity=0.22, color=fc_color)

    if not history.empty:
        fig.add_trace(
            go.Scatter(
                x=history["date"], y=history["value"], mode="lines", name="Observed",
                line={"width": 2.1, "color": "#0f172a"},
                hovertemplate="%{x|%Y-%m-%d}<br>Observed: %{y:.3f}<extra></extra>",
            )
        )

    anchor_date = None
    anchor_value = None
    if not history.empty:
        anchor_date = pd.Timestamp(history["date"].iloc[-1])
        anchor_value = float(history["value"].iloc[-1])

    now_line = _anchor_fan_line(nowcast, anchor_date, anchor_value)
    if not now_line.empty:
        fig.add_trace(
            go.Scatter(
                x=now_line["date"], y=now_line["q50"], mode="lines", name="Nowcast median",
                line={"width": 2.4, "color": now_color},
                hovertemplate="%{x|%Y-%m-%d}<br>Nowcast: %{y:.3f}<extra></extra>",
            )
        )
        anchor_date = pd.Timestamp(now_line["date"].iloc[-1])
        anchor_value = float(now_line["q50"].iloc[-1])

    fc_line = _anchor_fan_line(forecast, anchor_date, anchor_value)
    if not fc_line.empty:
        fig.add_trace(
            go.Scatter(
                x=fc_line["date"], y=fc_line["q50"], mode="lines", name="Forecast median",
                line={"width": 2.4, "dash": "dash", "color": fc_color},
                hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.3f}<extra></extra>",
            )
        )

    if not history.empty and not nowcast.empty:
        last_obs = pd.Timestamp(history["date"].iloc[-1])
        fig.add_vline(x=last_obs, line_width=1, line_dash="dash", line_color="#94a3b8")
    if not forecast.empty:
        first_future = pd.Timestamp(forecast["date"].min())
        fig.add_vline(x=first_future, line_width=1, line_dash="dot", line_color="#94a3b8")
        fig.add_annotation(
            x=first_future, y=1, yref="paper", text="Forecast", showarrow=False,
            xanchor="left", yanchor="bottom", font={"size": 11, "color": "#64748b"},
        )

    label = series.replace("_", " ").title()
    unit = ""
    selected = frame.loc[
        (frame["series"] == series)
        & (frame["metric"].astype(str) == str(metric))
        & frame["unit"].notna(),
        "unit",
    ]
    if len(selected):
        unit = str(selected.iloc[0])

    fig.update_layout(
        template="plotly_white",
        margin={"l": 54, "r": 24, "t": 54, "b": 42},
        height=520,
        title={"text": label, "x": 0.01, "xanchor": "left", "font": {"size": 18, "color": "#111827"}},
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#374151", "size": 12},
        xaxis_title=None, yaxis_title=unit, hovermode="x unified", dragmode="pan",
        hoverlabel={"bgcolor": "white", "bordercolor": "#e5e7eb", "font": {"color": "#111827"}},
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right", "font": {"size": 11}},
        uirevision=f"{context.get('model_id')}::{series}::{metric}",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor="#e5e7eb", tickfont={"color": "#6b7280"})
    fig.update_yaxes(gridcolor="#eef0f3", zerolinecolor="#d1d5db", tickfont={"color": "#6b7280"})
    return fig

def _empty_forecast_figure(message: str = "Select a run to display its forecast") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.54,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        align="center",
        font={"size": 14, "color": "#6b7280"},
    )
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin={"l": 40, "r": 20, "t": 30, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


# ---------------------------------------------------------------------------
# UI pieces
# ---------------------------------------------------------------------------


def _selector(label: str, component_id: str, placeholder: str) -> html.Div:
    return html.Div(
        [
            html.Label(label, className="selector-label"),
            dcc.Dropdown(
                id=component_id,
                placeholder=placeholder,
                clearable=False,
                className="selector-dropdown",
            ),
        ],
        className="selector-block",
    )


def _nav_link(label: str, href: str, icon: str) -> dcc.Link:
    return dcc.Link(
        [html.Span(icon, className="nav-icon"), html.Span(label)],
        href=href,
        className="nav-link",
    )


def _stat_card(title: str, value_id: str, subtitle_id: str | None = None) -> html.Div:
    children = [
        html.Div(title, className="stat-title"),
        html.Div("—", id=value_id, className="stat-value"),
    ]
    if subtitle_id:
        children.append(html.Div("", id=subtitle_id, className="stat-subtitle"))
    return html.Div(children, className="stat-card")


_GRAPH_CONFIG: dict = {
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def forecast_page() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Forecast", className="page-title"),
                            html.P(
                                "Observed data, ragged-edge nowcast and posterior predictive forecast from the selected saved run.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Metric", className="control-label"),
                                    dcc.Dropdown(
                                        id="forecast-metric",
                                        clearable=False,
                                        className="compact-dropdown",
                                    ),
                                ],
                                className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Label("Series", className="control-label"),
                                    dcc.Dropdown(
                                        id="forecast-series",
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
                                        id="forecast-fan",
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
                [
                    _stat_card("Latest observed", "stat-observed", "stat-observed-date"),
                    _stat_card("First future median", "stat-first", "stat-first-date"),
                    _stat_card("Terminal median", "stat-terminal", "stat-terminal-date"),
                    _stat_card("Forecast horizon", "stat-horizon", "stat-horizon-unit"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    dcc.Loading(
                        dcc.Graph(
                            id="forecast-graph",
                            config={
                                "displaylogo": False,
                                "scrollZoom": True,
                                "doubleClick": "reset",
                                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
                            },
                        ),
                        type="circle",
                    )
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Path values", className="panel-title"),
                            html.P(
                                "Quantiles are read from the compact display artefact; changing the fan does not reopen posterior draw files.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dash_table.DataTable(
                        id="forecast-values-table",
                        columns=[
                            {"name": "Date", "id": "date"},
                            {"name": "Segment", "id": "segment"},
                            {"name": "q05", "id": "q05", "type": "numeric"},
                            {"name": "q16", "id": "q16", "type": "numeric"},
                            {"name": "Median", "id": "q50", "type": "numeric"},
                            {"name": "q84", "id": "q84", "type": "numeric"},
                            {"name": "q95", "id": "q95", "type": "numeric"},
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



def aggregate_page() -> html.Div:
    """HICP Energy aggregate: the only place year-on-year HICP inflation lives.

    Component runs for gas, electricity and the fuels forecast a pre-tax price,
    so their own year-on-year rate is not HICP inflation. The tax bridge, the
    weekly-to-monthly conversion and the HICP rebasing are applied only in the
    aggregation layer, and this page reads its saved output.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Aggregate", className="page-title"),
                            html.P(
                                "Draw-wise chain-linked HICP Energy, six-component "
                                "contributions and aggregation diagnostics.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            _selector("Aggregate run", "agg-select", "Select an aggregate"),
                            _selector("Metric", "agg-metric", "Metric"),
                            html.Div(
                                [
                                    html.Span("Fan", className="control-label"),
                                    dcc.RadioItems(
                                        id="agg-fan",
                                        options=[
                                            {"label": "68%", "value": "68"},
                                            {"label": "90%", "value": "90"},
                                            {"label": "Both", "value": "both"},
                                        ],
                                        value="68",
                                        className="fan-radio",
                                        inline=True,
                                    ),
                                ],
                                className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Span("Model fit overlays", className="control-label"),
                                    dcc.Checklist(
                                        id="agg-historical-overlays",
                                        options=[
                                            {
                                                "label": "Show six-model reconstruction",
                                                "value": "components",
                                            },
                                            {
                                                "label": "Show BVAR model reconstruction",
                                                "value": "bvar_reconstruction",
                                            },
                                        ],
                                        value=[],
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
            html.Div(id="agg-banner"),
            html.Div(
                [
                    _stat_card("Latest observed", "agg-observed", "agg-observed-date"),
                    _stat_card("First future median", "agg-first", "agg-first-date"),
                    _stat_card("Terminal median", "agg-terminal", "agg-terminal-date"),
                    _stat_card("Posterior draws", "agg-draws", "agg-draws-unit"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [dcc.Loading(dcc.Graph(id="agg-graph", config=_GRAPH_CONFIG), type="circle")],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Active component tax scenarios", className="panel-title"),
                                    html.P(
                                        "The active VAT/excise scenario set is propagated jointly, draw-by-draw, through the selected HICP Energy aggregate. No BVAR is re-estimated.",
                                        className="panel-subtitle",
                                    ),
                                ],
                                className="panel-heading",
                            ),
                            html.Div(id="agg-scenario-banner", className="selection-banner"),
                            html.Div(
                                [
                                    _stat_card("Baseline terminal YoY", "agg-scen-baseline", "agg-scen-date"),
                                    _stat_card("Scenario terminal YoY", "agg-scen-scenario", "agg-scen-date-2"),
                                    _stat_card("Terminal impact", "agg-scen-impact", "agg-scen-interval"),
                                    _stat_card("Paired aggregate draws", "agg-scen-draws", "agg-scen-draws-note"),
                                ],
                                className="stats-grid",
                            ),
                            dcc.Loading(dcc.Graph(id="agg-scenario-graph", config=_GRAPH_CONFIG), type="circle"),
                            html.Div(
                                [
                                    html.Div(
                                        [dcc.Loading(dcc.Graph(id="agg-scenario-impact", config=_GRAPH_CONFIG), type="circle")],
                                        className="panel chart-panel",
                                    ),
                                    html.Div(
                                        [dcc.Loading(dcc.Graph(id="agg-scenario-contrib-impact", config=_GRAPH_CONFIG), type="circle")],
                                        className="panel chart-panel",
                                    ),
                                ],
                                style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "16px"},
                            ),
                        ]
                    )
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [dcc.Loading(dcc.Graph(id="agg-contrib", config=_GRAPH_CONFIG), type="circle")],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H3("Aggregation provenance", className="panel-title")],
                        className="panel-heading",
                    ),
                    dash_table.DataTable(
                        id="agg-table",
                        columns=[{"name": "Item", "id": "Item"}, {"name": "Value", "id": "Value"}],
                        data=[],
                        page_size=12,
                        style_as_list_view=True,
                        style_cell={
                            "fontFamily": "Inter, Segoe UI, sans-serif",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                    ),
                ],
                className="panel table-panel",
            ),
        ],
        className="page-body",
    )



def scenario_page() -> html.Div:
    """Multi-component ex-post VAT/excise scenario builder."""
    controls = html.Div([
        html.Div([html.Label("Component", className="control-label"), dcc.Dropdown(id="scenario-component-select", options=[], value=None, clearable=False, className="compact-dropdown")], className="control-block"),
        html.Div([html.Label("Scenario start", className="control-label"), dcc.DatePickerSingle(id="scenario-start-date", display_format="YYYY-MM-DD", first_day_of_week=1, clearable=False)], className="control-block"),
        html.Div([html.Label("VAT change (pp)", className="control-label"), dcc.Input(id="scenario-vat-delta", type="number", value=0.0, step=0.1, debounce=0.4, className="compact-dropdown")], className="control-block"),
        html.Div([html.Label(id="scenario-excise-label", children="Excise change", className="control-label"), dcc.Input(id="scenario-excise-delta", type="number", value=0.0, step=0.1, debounce=0.4, className="compact-dropdown")], className="control-block"),
        html.Div([html.Label("Fan", className="control-label"), dcc.RadioItems(id="scenario-fan", options=[{"label":"68%","value":"68"},{"label":"90%","value":"90"},{"label":"Both","value":"both"}], value="68", inline=True, className="fan-radio")], className="control-block"),
        html.Button("Add / update", id="scenario-apply", n_clicks=0, className="refresh-button"),
        html.Button("Remove", id="scenario-remove", n_clicks=0, className="refresh-button"),
        html.Button("Reset all", id="scenario-reset-all", n_clicks=0, className="refresh-button"),
    ], className="chart-controls")
    return html.Div([
        html.Div([html.Div([html.H2("Scenarios", className="page-title"), html.P("Build several VAT/excise shocks on different energy components. Saved BVAR draws remain fixed; the combined set is propagated draw-by-draw to HICP Energy.", className="page-subtitle")]), controls], className="page-heading-row"),
        html.Div(id="scenario-support-note", className="selection-banner"), html.Div(id="scenario-banner"),
        html.Div([html.Div([html.Div("Active scenario set", className="eyebrow"), html.Div(id="scenario-set-summary")], className="panel", style={"padding":"16px 18px","marginBottom":"16px"})]),
        html.Div([_stat_card("Baseline terminal YoY","scenario-baseline-terminal","scenario-terminal-date"), _stat_card("Scenario terminal YoY","scenario-scenario-terminal","scenario-terminal-date-2"), _stat_card("Terminal tax impact","scenario-impact-terminal","scenario-impact-interval"), _stat_card("Paired posterior draws","scenario-draws","scenario-draws-note")], className="stats-grid"),
        html.Div([dcc.Loading(dcc.Graph(id="scenario-main-graph", config=_GRAPH_CONFIG), type="circle")], className="panel chart-panel"),
        html.Div([html.Div([dcc.Loading(dcc.Graph(id="scenario-level-impact", config=_GRAPH_CONFIG), type="circle")], className="panel chart-panel"), html.Div([dcc.Loading(dcc.Graph(id="scenario-yoy-impact", config=_GRAPH_CONFIG), type="circle")], className="panel chart-panel")], style={"display":"grid","gridTemplateColumns":"repeat(2, minmax(0, 1fr))","gap":"16px"}),
        html.Div([dcc.Loading(dcc.Graph(id="scenario-tax-graph", config=_GRAPH_CONFIG), type="circle")], className="panel chart-panel"),
    ], className="page-body")

def placeholder_page(title: str, description: str, next_slice: str) -> html.Div:
    return html.Div(
        [
            html.H2(title, className="page-title"),
            html.P(description, className="page-subtitle"),
            html.Div(
                [
                    html.Div("Wired shell", className="eyebrow"),
                    html.H3(next_slice, className="placeholder-title"),
                    html.P(
                        "The route is reserved and shares the same global context. Its econometric callback will be connected without changing the navigation or selection contract.",
                        className="placeholder-text",
                    ),
                ],
                className="panel placeholder-panel",
            ),
        ],
        className="page-body",
    )


def sidebar() -> html.Aside:
    return html.Aside(
        [
            html.Div(
                [
                    html.Div(className="brand-mark", title="BVAR"),
                    html.Div("Energy Inflation", className="brand-title"),
                ],
                className="brand-row",
            ),
            html.Nav(
                [
                    _nav_link("Forecast", "/forecast", "↗"),
                    _nav_link("Aggregate", "/aggregate", "Σ"),
                    _nav_link("Scenarios", "/scenarios", "△"),
                    _nav_link("Structural", "/structural", "ψ"),
                    _nav_link("Estimation", "/estimation", "⚙"),
                ],
                className="nav-stack",
            ),
            html.Div(
                [
                    html.Div("Storage", className="sidebar-meta-label"),
                    html.Div(str(RESULTS_ROOT), className="sidebar-path"),
                ],
                className="sidebar-footer",
            ),
        ],
        className="sidebar",
    )


def topbar() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    _selector("Vintage", "vintage-select", "Select vintage"),
                    _selector("Model", "model-select", "Select model"),
                    _selector("Forecast", "forecast-select", "Forecast contract"),
                    _selector("Run", "run-select", "Select run"),
                ],
                className="selectors-grid",
            ),
            html.Div(
                [
                    html.Button("Refresh", id="refresh-registry", n_clicks=0, className="refresh-button"),
                    html.Div(id="registry-status", className="registry-status"),
                ],
                className="refresh-area",
            ),
        ],
        className="topbar",
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Dash(
    __name__,
    assets_folder=str(Path(__file__).parent / "assets"),
    suppress_callback_exceptions=True,
    background_callback_manager=background_callback_manager,
    title="Energy BVAR Dashboard",
    update_title="Updating…",
)
server = app.server

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="registry-store", data=_scan_registry()),
        dcc.Store(id="ctx-store", storage_type="session"),
        dcc.Store(id="data-store", storage_type="memory"),
        dcc.Store(id="agg-store", storage_type="memory"),
        dcc.Store(id="scenario-store", storage_type="memory"),
        dcc.Store(id="agg-scenario-store", storage_type="memory"),
        sidebar(),
        html.Main(
            [
                topbar(),
                html.Div(id="selection-banner", className="selection-banner"),
                html.Div(
                    [
                        html.Div(forecast_page(), id="page-forecast"),
                        html.Div(
                            aggregate_page(),
                            id="page-aggregate",
                            style={"display": "none"},
                        ),
                        html.Div(
                            scenario_page(),
                            id="page-scenarios",
                            style={"display": "none"},
                        ),
                        html.Div(
                            placeholder_page(
                                "Structural analysis",
                                "Impulse responses, FEVD and historical decomposition using saved Gibbs draws.",
                                "Next: recursive/sign identification, reference date and shock-unit controls.",
                            ),
                            id="page-structural",
                            style={"display": "none"},
                        ),
                        html.Div(
                            placeholder_page(
                                "Estimation",
                                "Declarative hyperparameters and background estimation through the shared pipeline.",
                                "Next: Diskcache background callback with progress, cancel and global estimation lock.",
                            ),
                            id="page-estimation",
                            style={"display": "none"},
                        ),
                    ],
                    className="page-container",
                ),
            ],
            className="main-shell",
        ),
    ],
    className="app-shell",
)


# ---------------------------------------------------------------------------
# Registry and global context callbacks
# ---------------------------------------------------------------------------


@callback(Output("registry-store", "data"), Input("refresh-registry", "n_clicks"), prevent_initial_call=True)
def refresh_registry(_: int) -> dict:
    return _scan_registry()


@callback(Output("registry-status", "children"), Input("registry-store", "data"))
def registry_status(store: dict | None):
    if not store:
        return "Registry unavailable"
    cls = "status-dot ok" if store.get("ok") else "status-dot error"
    unexpected = store.get("unexpected") or []
    children = [html.Span(className=cls), html.Span(str(store.get("message", "")))]
    if unexpected:
        children.append(
            html.Span(
                f" · {len(unexpected)} ignored folder{'s' if len(unexpected) != 1 else ''}",
                title="\n".join(map(str, unexpected)),
            )
        )
    return html.Div(children)


@callback(
    Output("vintage-select", "options"),
    Output("vintage-select", "value"),
    Input("registry-store", "data"),
    State("vintage-select", "value"),
)
def vintage_options(_: dict | None, current: str | None):
    forecasts = list_forecasts(REGISTRY_PATH, valid_only=True, present_only=True)
    if forecasts.empty:
        return [], None
    vintages = sorted(forecasts["vintage"].astype(str).unique(), reverse=True)
    options = [{"label": value, "value": value} for value in vintages]
    return options, current if current in vintages else vintages[0]


@callback(
    Output("model-select", "options"),
    Output("model-select", "value"),
    Input("vintage-select", "value"),
    Input("registry-store", "data"),
    State("model-select", "value"),
)
def model_options(vintage: str | None, _: dict | None, current: str | None):
    if not vintage:
        return [], None
    forecasts = list_forecasts(
        REGISTRY_PATH, vintage=vintage, valid_only=True, present_only=True
    )
    if forecasts.empty:
        return [], None
    models = sorted(forecasts["model_id"].astype(str).unique())
    options = []
    for model_id in models:
        try:
            label = model_spec(model_id).label
        except Exception:
            label = model_id.replace("_", " ").title()
        options.append({"label": label, "value": model_id})
    values = [item["value"] for item in options]
    return options, current if current in values else values[0]


@callback(
    Output("forecast-select", "options"),
    Output("forecast-select", "value"),
    Input("vintage-select", "value"),
    Input("model-select", "value"),
    Input("registry-store", "data"),
    State("forecast-select", "value"),
)
def forecast_options(
    vintage: str | None,
    model_id: str | None,
    _: dict | None,
    current: str | None,
):
    if not vintage or not model_id:
        return [], None
    frame = list_forecasts(
        REGISTRY_PATH,
        model_id=model_id,
        vintage=vintage,
        valid_only=True,
        present_only=True,
    )
    names = sorted(frame["forecast_name"].astype(str).unique()) if not frame.empty else []
    options = [{"label": name.replace("_", " ").title(), "value": name} for name in names]
    preferred = "unconditional" if "unconditional" in names else (names[0] if names else None)
    return options, current if current in names else preferred


@callback(
    Output("run-select", "options"),
    Output("run-select", "value"),
    Input("vintage-select", "value"),
    Input("model-select", "value"),
    Input("forecast-select", "value"),
    Input("registry-store", "data"),
    State("run-select", "value"),
)
def run_options(
    vintage: str | None,
    model_id: str | None,
    forecast_name: str | None,
    _: dict | None,
    current: str | None,
):
    if not all([vintage, model_id, forecast_name]):
        return [], None
    forecasts = list_forecasts(
        REGISTRY_PATH,
        model_id=model_id,
        vintage=vintage,
        forecast_name=forecast_name,
        valid_only=True,
        present_only=True,
    )
    runs = list_runs(
        REGISTRY_PATH,
        model_id=model_id,
        vintage=vintage,
        status="complete",
        present_only=True,
    )
    if forecasts.empty or runs.empty:
        return [], None
    merged = forecasts.merge(
        runs[["run_id", "promoted", "created_at_utc", "missing_data_method", "code_version"]],
        on="run_id",
        how="inner",
    )
    if merged.empty:
        return [], None
    merged = merged.sort_values(["promoted", "created_at_utc", "run_id"], ascending=[False, False, True])
    options = []
    for _, row in merged.iterrows():
        tags = []
        if bool(row.get("promoted")):
            tags.append("PROMOTED")
        if not bool(row.get("has_display")):
            tags.append("display pending")
        detail = " · ".join(tags)
        label = f"{_short_run(row['run_id'])}{' · ' + detail if detail else ''}"
        options.append({"label": label, "value": str(row["run_id"])})
    values = [item["value"] for item in options]
    promoted = merged.loc[merged["promoted"] == 1, "run_id"].astype(str).tolist()
    default = promoted[0] if promoted else values[0]
    return options, current if current in values else default


@callback(
    Output("ctx-store", "data"),
    Input("vintage-select", "value"),
    Input("model-select", "value"),
    Input("run-select", "value"),
    Input("forecast-select", "value"),
    State("ctx-store", "data"),
)
def update_context(
    vintage: str | None,
    model_id: str | None,
    run_id: str | None,
    forecast_name: str | None,
    previous: dict | None,
):
    previous = dict(previous or {})
    previous.update(
        {
            "vintage": vintage,
            "model_id": model_id,
            "run_id": run_id,
            "forecast_name": forecast_name,
            "draw_mode": previous.get("draw_mode", "posterior"),
        }
    )
    return previous


@callback(
    Output("data-store", "data"),
    Output("selection-banner", "children"),
    Input("ctx-store", "data"),
    Input("registry-store", "data"),
)
def load_selected_display(context: dict | None, _: dict | None):
    if not context or not all(
        context.get(key) for key in ("vintage", "model_id", "run_id", "forecast_name")
    ):
        return None, html.Div("Select a saved run to load its display artefact.")

    forecasts = list_forecasts(
        REGISTRY_PATH,
        model_id=context["model_id"],
        vintage=context["vintage"],
        forecast_name=context["forecast_name"],
        valid_only=True,
        present_only=True,
    )
    selected = forecasts.loc[forecasts["run_id"].astype(str) == str(context["run_id"])]
    if selected.empty:
        return None, html.Div("Selected forecast store is no longer present in the registry.")

    row = selected.iloc[0]
    forecast_dir = Path(str(row["directory"]))
    display_path = forecast_dir / DISPLAY_FILENAME
    materialised = False
    try:
        if not display_path.is_file():
            if not AUTO_BUILD_DISPLAY:
                raise FileNotFoundError(
                    f"{display_path} is missing and ENERGY_BVAR_AUTO_BUILD_DISPLAY=0."
                )
            run_dir = forecast_dir.parent.parent
            build_component_display(
                run_dir,
                project_root=PROJECT_ROOT,
                forecast_name=context["forecast_name"],
            )
            materialised = True
        frame = load_display_artifact(display_path)
        # display_v1 files created before the component-HICP contract can be
        # perfectly valid for the raw BVAR forecast while still lacking HICP
        # Level / YoY.  Rebuild them once from the saved forecast; the builder
        # materialises hicp_draws.npz without re-estimating the BVAR.
        fan_mask = frame["record_type"].astype(str) == "fan"
        metrics_present = set(frame.loc[fan_mask, "metric"].dropna().astype(str))
        if AUTO_BUILD_DISPLAY and not {"hicp_level", "yoy"}.issubset(metrics_present):
            run_dir = forecast_dir.parent.parent
            build_component_display(
                run_dir,
                project_root=PROJECT_ROOT,
                forecast_name=context["forecast_name"],
                overwrite=True,
            )
            materialised = True
            frame = load_display_artifact(display_path)

        meta = display_metadata(frame)
        fan_rows = int((frame["record_type"].astype(str) == "fan").sum())
        if fan_rows == 0:
            raise ValueError(
                f"{DISPLAY_FILENAME} loaded successfully but contains no forecast fan rows."
            )
        final_metrics = set(
            frame.loc[frame["record_type"].astype(str) == "fan", "metric"]
            .dropna().astype(str)
        )
        if not {"hicp_level", "yoy"}.issubset(final_metrics):
            raise ValueError(
                "Component HICP post-processing is incomplete: expected both "
                "'hicp_level' and 'yoy' in display_v1."
            )
    except Exception as exc:
        return None, html.Div(
            [html.Strong("Display load failed: "), html.Span(str(exc))],
            className="banner-error",
        )

    label = meta.get("model_label") or context["model_id"]
    method = meta.get("missing_data_method") or "—"
    approximation_used = meta.get("missing_data_approximation_used")
    if str(method).lower() == "dk":
        missing_note = "exact augmentation"
    elif str(method).lower() == "linear":
        if str(approximation_used).lower() in {"true", "1"}:
            missing_note = "interpolation used"
        elif str(approximation_used).lower() in {"false", "0"}:
            missing_note = "interpolation not used"
        else:
            missing_note = "linear method"
    else:
        missing_note = ""
    built_note = " · display/HICP materialised now" if materialised else ""
    banner = html.Div(
        [
            html.Strong(str(label)),
            html.Span(f" · vintage {context['vintage']}"),
            html.Span(f" · run {_short_run(context['run_id'])}"),
            html.Span(f" · missing data: {method}{' (' + missing_note + ')' if missing_note else ''}"),
            html.Span(" · component HICP: ready"),
            html.Span(built_note),
        ]
    )
    return {
        "frame_json": _json_frame(frame),
        "meta": meta,
        "context": context,
        "display_path": str(display_path),
    }, banner


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@callback(
    Output("page-forecast", "style"),
    Output("page-aggregate", "style"),
    Output("page-scenarios", "style"),
    Output("page-structural", "style"),
    Output("page-estimation", "style"),
    Input("url", "pathname"),
)
def route(pathname: str | None):
    """Toggle pre-mounted pages instead of injecting callback targets dynamically.

    Keeping the Forecast controls in the initial DOM is important: ``data-store``
    can be populated during app bootstrap, and the metric/series callbacks must
    already have mounted Output components when that happens.
    """
    pathname = pathname or "/forecast"
    route_name = {
        "/": "forecast",
        "/forecast": "forecast",
        "/aggregate": "aggregate",
        "/scenarios": "scenarios",
        "/structural": "structural",
        "/estimation": "estimation",
    }.get(pathname, "forecast")
    visible = {"display": "block"}
    hidden = {"display": "none"}
    names = ("forecast", "aggregate", "scenarios", "structural", "estimation")
    return tuple(visible if name == route_name else hidden for name in names)


# ---------------------------------------------------------------------------
# Forecast-page callbacks: all downstream of data-store, no filesystem I/O
# ---------------------------------------------------------------------------


_METRIC_LABELS = {
    "level": "Level",
    "hicp_level": "HICP level",
    "absolute_change": "Absolute change",
    "yoy": "Year-on-year",
}


@callback(
    Output("forecast-metric", "options"),
    Output("forecast-metric", "value"),
    Input("data-store", "data"),
    State("forecast-metric", "value"),
)
def metric_options(store: dict | None, current: str | None):
    frame = _frame_from_store(store)
    if frame.empty:
        return [], None
    fan = frame.loc[frame["record_type"] == "fan"]
    metrics = [str(value) for value in fan["metric"].dropna().unique()]
    preferred_order = ["yoy", "hicp_level", "level", "absolute_change"]
    metrics = [m for m in preferred_order if m in metrics] + [m for m in metrics if m not in preferred_order]
    options = [{"label": _METRIC_LABELS.get(m, m.replace("_", " ").title()), "value": m} for m in metrics]
    default = "yoy" if "yoy" in metrics else ("hicp_level" if "hicp_level" in metrics else ("level" if "level" in metrics else (metrics[0] if metrics else None)))
    return options, current if current in metrics else default


@callback(
    Output("forecast-series", "options"),
    Output("forecast-series", "value"),
    Input("data-store", "data"),
    Input("forecast-metric", "value"),
    State("forecast-series", "value"),
)
def series_options(store: dict | None, metric: str | None, current: str | None):
    frame = _frame_from_store(store)
    if frame.empty or not metric:
        return [], None
    fan = frame.loc[(frame["record_type"] == "fan") & (frame["metric"] == metric)]
    if fan.empty:
        return [], None
    pairs = fan[["series", "label"]].dropna(subset=["series"]).drop_duplicates("series")
    options = [
        {
            "label": str(row["label"]) if pd.notna(row["label"]) else str(row["series"]),
            "value": str(row["series"]),
        }
        for _, row in pairs.iterrows()
    ]
    values = [item["value"] for item in options]
    target = None
    if store and store.get("meta"):
        target = store["meta"].get("target")
    default = target if target in values else (values[0] if values else None)
    return options, current if current in values else default


@callback(
    Output("forecast-graph", "figure"),
    Output("forecast-values-table", "data"),
    Output("stat-observed", "children"),
    Output("stat-observed-date", "children"),
    Output("stat-first", "children"),
    Output("stat-first-date", "children"),
    Output("stat-terminal", "children"),
    Output("stat-terminal-date", "children"),
    Output("stat-horizon", "children"),
    Output("stat-horizon-unit", "children"),
    Input("data-store", "data"),
    Input("forecast-metric", "value"),
    Input("forecast-series", "value"),
    Input("forecast-fan", "value"),
)
def update_forecast_view(
    store: dict | None,
    metric: str | None,
    series: str | None,
    fan_mode: str,
):
    frame = _frame_from_store(store)
    if frame.empty or not metric or not series:
        empty = _empty_forecast_figure(
            "Forecast display is not available yet. Check the run banner above."
        )
        return empty, [], "—", "", "—", "", "—", "", "—", ""

    context = dict((store or {}).get("context") or {})
    fig = forecast_figure(frame, metric=metric, series=series, fan_mode=fan_mode, context=context)
    history = _history_rows(frame, metric, series)
    fan = _forecast_rows(frame, metric, series)
    future = fan.loc[fan["segment"] == "forecast"].copy()

    table = fan[["date", "segment", "q05", "q16", "q50", "q84", "q95"]].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    for column in ("q05", "q16", "q50", "q84", "q95"):
        table[column] = table[column].round(4)
    table_data = table.to_dict("records")

    observed_value = observed_date = first_value = first_date = terminal_value = terminal_date = "—"
    if not history.empty:
        last = history.iloc[-1]
        observed_value = _format_number(last["value"])
        observed_date = pd.Timestamp(last["date"]).strftime("%Y-%m-%d")
    if not future.empty:
        first = future.iloc[0]
        last = future.iloc[-1]
        first_value = _format_number(first["q50"])
        first_date = pd.Timestamp(first["date"]).strftime("%Y-%m-%d")
        terminal_value = _format_number(last["q50"])
        terminal_date = pd.Timestamp(last["date"]).strftime("%Y-%m-%d")

    horizon = int(len(future))
    meta = (store or {}).get("meta", {})
    if metric in {"hicp_level", "yoy"}:
        frequency = str(meta.get("hicp_frequency", "monthly")).lower()
    else:
        frequency = str(meta.get("frequency", "period")).lower()
    unit = "months" if frequency == "monthly" else "weeks" if frequency == "weekly" else "periods"
    return (
        fig,
        table_data,
        observed_value,
        observed_date,
        first_value,
        first_date,
        terminal_value,
        terminal_date,
        str(horizon),
        unit,
    )



# ---------------------------------------------------------------------------
# Tax-scenario page callbacks
# ---------------------------------------------------------------------------


def _scenario_component_forecast_row(vintage: str | None, model_id: str | None, forecast_name: str | None):
    if not vintage or not model_id or not forecast_name:
        return None
    forecasts = list_forecasts(REGISTRY_PATH, model_id=str(model_id), vintage=str(vintage), forecast_name=str(forecast_name), valid_only=True, present_only=True)
    if forecasts.empty:
        return None
    runs = list_runs(REGISTRY_PATH, model_id=str(model_id), vintage=str(vintage), status="complete", present_only=True)
    promoted = set(runs.loc[runs["promoted"].fillna(0).astype(int)==1,"run_id"].astype(str)) if not runs.empty else set()
    promoted_rows = forecasts.loc[forecasts["run_id"].astype(str).isin(promoted)]
    if len(promoted_rows)==1: return promoted_rows.iloc[0]
    if len(forecasts)==1: return forecasts.iloc[0]
    return None


@callback(Output("scenario-component-select","options"), Output("scenario-component-select","value"), Input("vintage-select","value"), Input("forecast-select","value"), Input("registry-store","data"), State("scenario-component-select","value"))
def scenario_component_options(vintage, forecast_name, _, current):
    if not vintage or not forecast_name: return [], None
    available=[]; options=[]
    for model_id in TAX_SCENARIO_MODEL_IDS:
        if _scenario_component_forecast_row(vintage, model_id, forecast_name) is not None:
            available.append(model_id); options.append({"label":model_spec(model_id).label,"value":model_id})
    return options, (current if current in available else (available[0] if available else None))


@callback(Output("scenario-start-date","date"), Output("scenario-start-date","min_date_allowed"), Output("scenario-start-date","max_date_allowed"), Output("scenario-vat-delta","value"), Output("scenario-excise-delta","value"), Output("scenario-excise-label","children"), Output("scenario-support-note","children"), Output("scenario-start-date","disabled"), Output("scenario-vat-delta","disabled"), Output("scenario-excise-delta","disabled"), Input("scenario-component-select","value"), Input("vintage-select","value"), Input("forecast-select","value"), Input("registry-store","data"), Input("scenario-store","data"))
def scenario_control_defaults(model_id, vintage, forecast_name, _, scenario_store):
    if not model_id or not vintage or not forecast_name:
        return None,None,None,0.0,0.0,"Excise change","No scenario-capable component forecast is available.",True,True,True
    row=_scenario_component_forecast_row(vintage,model_id,forecast_name)
    if row is None:
        return None,None,None,0.0,0.0,"Excise change",f"{model_spec(model_id).label}: no unique saved forecast. Promote one run if several coexist.",True,True,True
    try:
        from energy_bvar_io import load_energy_bvar_forecast
        forecast=load_energy_bvar_forecast(Path(str(row["directory"])))
        dataset=PROJECT_ROOT/"data"/"processed"/str(vintage)/model_spec(model_id).dataset_file
        contract=component_tax_scenario_contract(model_id,forecast,dataset_path=dataset)
    except Exception as exc:
        return None,None,None,0.0,0.0,"Excise change",f"Scenario controls unavailable: {exc}",True,True,True
    existing=scenario_set_payload(scenario_store,model_id); em=dict((existing or {}).get("meta",{}) or {})
    unit=str(contract.get("excise_unit","source unit"))
    min_start=pd.Timestamp(contract["min_start"])
    max_start=pd.Timestamp(contract["max_start"])
    start=pd.Timestamp(em.get("scenario_start") or contract["default_start"])
    if start < min_start or start > max_start:
        start=pd.Timestamp(contract["default_start"])
    freq_label="monthly periods" if str(contract.get("frequency"))=="monthly" else "weekly periods (Monday)"
    note=(
        f"{model_spec(model_id).label} · run {str(row['run_id'])[:12]} · "
        f"available scenario window {min_start.date().isoformat()} — {max_start.date().isoformat()} "
        f"({freq_label}) · baseline at first future period: VAT "
        f"{contract['baseline_vat_at_start']:.2f}% · excise "
        f"{contract['baseline_excise_at_start']:.3f} {unit}. "
        "Changes are additive to the baseline tax path."
    )
    return start.date(),min_start.date(),max_start.date(),float(em.get("vat_delta_pp",0.0) or 0.0),float(em.get("excise_delta",0.0) or 0.0),f"Excise change ({unit})",note,False,False,False


@callback(Output("scenario-store","data"), Output("scenario-banner","children"), Input("scenario-apply","n_clicks"), Input("scenario-remove","n_clicks"), Input("scenario-reset-all","n_clicks"), State("vintage-select","value"), State("forecast-select","value"), State("scenario-component-select","value"), State("scenario-start-date","date"), State("scenario-vat-delta","value"), State("scenario-excise-delta","value"), State("scenario-store","data"), prevent_initial_call=True)
def mutate_scenario_set(_apply,_remove,_reset,vintage,forecast_name,model_id,start_date,vat_delta,excise_delta,current_store):
    trigger=ctx.triggered_id
    if trigger=="scenario-reset-all": return clear_scenario_set(vintage=vintage,forecast_name=forecast_name), html.Div("All component tax scenarios cleared.",className="selection-banner")
    if not model_id: return no_update, html.Div("Select a component.",className="banner-error")
    if trigger=="scenario-remove": return remove_scenario_component(current_store,model_id), html.Div(f"{model_spec(model_id).label}: scenario removed.",className="selection-banner")
    if trigger!="scenario-apply": raise PreventUpdate
    if not vintage or not forecast_name or start_date is None: return no_update, html.Div("Vintage, forecast and start date are required.",className="banner-error")
    row=_scenario_component_forecast_row(vintage,model_id,forecast_name)
    if row is None: return no_update, html.Div(f"{model_spec(model_id).label}: no unique promoted/saved run is available.",className="banner-error")
    vat_delta=float(vat_delta or 0.0); excise_delta=float(excise_delta or 0.0)
    if abs(vat_delta)<1e-15 and abs(excise_delta)<1e-15:
        return remove_scenario_component(current_store,model_id), html.Div(f"{model_spec(model_id).label}: zero changes, component removed from the active set.",className="selection-banner")
    try:
        from energy_bvar_io import load_energy_bvar_forecast
        forecast_dir=Path(str(row["directory"])); run_dir=forecast_dir.parent.parent
        forecast=load_energy_bvar_forecast(forecast_dir)
        dataset=PROJECT_ROOT/"data"/"processed"/str(vintage)/model_spec(model_id).dataset_file
        contract=component_tax_scenario_contract(model_id,forecast,dataset_path=dataset)
        frequency=str(contract.get("frequency","monthly"))
        requested_start=pd.Timestamp(start_date)
        if frequency=="weekly":
            effective_start=requested_start.to_period("W-SUN").start_time
        else:
            effective_start=requested_start.to_period("M").to_timestamp(how="start")
        min_start=pd.Timestamp(contract["min_start"])
        max_start=pd.Timestamp(contract["max_start"])
        if effective_start < min_start or effective_start > max_start:
            return no_update, html.Div(
                f"Scenario start must lie inside the selected forecast horizon: "
                f"{min_start.date().isoformat()} — {max_start.date().isoformat()}.",
                className="banner-error",
            )
        result=build_saved_component_tax_scenario(run_dir,project_root=PROJECT_ROOT,forecast_name=str(forecast_name),start_date=effective_start,vat_delta_pp=vat_delta,excise_delta=excise_delta,max_draws=500)
        payload=scenario_payload(result); payload.setdefault("meta",{}).update({"vintage":str(vintage),"run_id":str(row["run_id"]),"forecast_name":str(forecast_name)})
        updated=upsert_scenario_component(current_store,payload,vintage=str(vintage),forecast_name=str(forecast_name))
    except Exception as exc:
        return no_update, html.Div([html.Strong("Scenario calculation failed: "),html.Span(str(exc))],className="banner-error")
    meta=payload["meta"]; count=len(scenario_set_components(updated))
    return updated, html.Div([html.Strong(str(result["hicp_label"])),html.Span(f" · VAT {vat_delta:+.2f} pp"),html.Span(f" · excise {excise_delta:+.3f} {meta['excise_unit']}"),html.Span(f" · {meta['n_draws_effective']} paired draws"),html.Span(f" · {count} active component scenario"+("s" if count!=1 else ""))],className="selection-banner")


@callback(
    Output("scenario-set-summary","children"),
    Input("scenario-store","data"),
    Input("scenario-component-select","value"),
)
def render_scenario_set_summary(store, selected_model_id):
    rows=scenario_set_summary(store)
    if not rows:
        return html.Div(
            "No active tax scenario. Choose a component, set VAT and/or excise, then Add / update.",
            className="placeholder-text",
        )
    cards=[]
    for row in rows:
        model_id=str(row["model_id"])
        selected=(str(selected_model_id)==model_id)
        start=pd.to_datetime(row.get("scenario_start"),errors="coerce")
        start_label="—" if pd.isna(start) else pd.Timestamp(start).date().isoformat()
        cards.append(
            html.Button(
                [
                    html.Div(
                        [
                            html.Strong(row["label"]),
                            html.Span("Selected", style={
                                "display":"inline-block" if selected else "none",
                                "marginLeft":"8px",
                                "fontSize":"10px",
                                "fontWeight":"700",
                                "textTransform":"uppercase",
                                "letterSpacing":"0.08em",
                                "color":"#2563eb",
                            }),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(f"start {start_label}"),
                            html.Span(f" · VAT {row['vat_delta_pp']:+.2f} pp"),
                            html.Span(f" · excise {row['excise_delta']:+.3f} {row['excise_unit']}"),
                        ],
                        style={"marginTop":"3px","fontSize":"12px","color":"#64748b"},
                    ),
                ],
                id={"type":"scenario-card","model_id":model_id},
                n_clicks=0,
                title=f"Select {row['label']} to edit it and inspect its component charts",
                style={
                    "display":"block",
                    "width":"100%",
                    "textAlign":"left",
                    "background":"#eff6ff" if selected else "#ffffff",
                    "border":"1px solid #2563eb" if selected else "1px solid #e2e8f0",
                    "borderRadius":"10px",
                    "padding":"10px 12px",
                    "marginTop":"7px",
                    "cursor":"pointer",
                    "color":"#0f172a",
                    "boxShadow":"0 1px 2px rgba(15,23,42,0.04)",
                },
            )
        )
    return cards


@callback(
    Output("scenario-component-select","value", allow_duplicate=True),
    Input({"type":"scenario-card","model_id":ALL},"n_clicks"),
    prevent_initial_call=True,
)
def select_scenario_card(clicks):
    # Dynamic-card creation can trigger the pattern callback with all zeros;
    # only an actual click should change the editor selection.
    if not clicks or not any(int(value or 0) > 0 for value in clicks):
        raise PreventUpdate
    triggered=ctx.triggered_id
    if not isinstance(triggered,dict) or triggered.get("type")!="scenario-card":
        raise PreventUpdate
    model_id=triggered.get("model_id")
    if not model_id:
        raise PreventUpdate
    return str(model_id)


@callback(Output("scenario-baseline-terminal","children"),Output("scenario-terminal-date","children"),Output("scenario-scenario-terminal","children"),Output("scenario-terminal-date-2","children"),Output("scenario-impact-terminal","children"),Output("scenario-impact-interval","children"),Output("scenario-draws","children"),Output("scenario-draws-note","children"),Input("scenario-store","data"),Input("scenario-component-select","value"))
def scenario_stat_cards(store,model_id):
    values=scenario_kpis(scenario_set_payload(store,model_id))
    if values.get("terminal_date") is None: return "—","","—","","—","","—",""
    date=pd.Timestamp(values["terminal_date"]).date().isoformat(); low=values.get("impact_low"); high=values.get("impact_high"); interval="" if low is None or high is None else f"68% [{low:+.2f}, {high:+.2f}] pp"; draws=values.get("n_draws")
    return f"{values['baseline_terminal']:.2f}%",date,f"{values['scenario_terminal']:.2f}%",date,f"{values['impact_terminal']:+.2f} pp",interval,("—" if draws is None else f"{int(draws):,}"),"matched baseline/scenario"


@callback(Output("scenario-main-graph","figure"),Output("scenario-level-impact","figure"),Output("scenario-yoy-impact","figure"),Output("scenario-tax-graph","figure"),Input("scenario-store","data"),Input("scenario-component-select","value"),Input("scenario-fan","value"))
def scenario_figures(store,model_id,fan_mode):
    payload=scenario_set_payload(store,model_id)
    if not payload:
        empty=empty_scenario_figure("Add a tax scenario for the selected component"); return empty,empty,empty,empty
    meta=dict(payload.get("meta",{}) or {}); revision=f"{meta.get('run_id','run')}::{meta.get('forecast_name','forecast')}::{model_id}::scenario"; mode=fan_mode or "68"
    return scenario_main_figure(payload,fan_mode=mode,uirevision=revision+"::main"),scenario_impact_figure(payload,metric="level",fan_mode=mode,uirevision=revision+"::level"),scenario_impact_figure(payload,metric="yoy",fan_mode=mode,uirevision=revision+"::yoy"),scenario_tax_figure(payload,uirevision=revision+"::tax")

def _selected_aggregate_row(vintage: str | None, aggregate_run_id: str | None):
    if not vintage or not aggregate_run_id:
        return None
    frame = list_aggregates(REGISTRY_PATH, vintage=str(vintage), present_only=True)
    if frame.empty:
        return None
    selected = frame.loc[frame["aggregate_run_id"].astype(str) == str(aggregate_run_id)]
    return None if selected.empty else selected.iloc[0]


def _aggregate_metadata(directory: Path) -> dict:
    path = Path(directory) / "metadata.json"
    if not path.is_file():
        fallback = Path(directory) / "aggregate_config.json"
        path = fallback if fallback.is_file() else path
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _aggregate_forecast_origin(directory: Path) -> pd.Timestamp | None:
    """Latest first-future month across component stores = aggregate forecast origin."""
    metadata = _aggregate_metadata(directory)
    stores = dict(metadata.get("component_forecast_stores", {}) or {})
    starts: list[pd.Timestamp] = []
    for raw in stores.values():
        path = Path(str(raw)) / "forecast_metadata.json"
        if not path.is_file():
            continue
        try:
            fm = json.loads(path.read_text(encoding="utf-8"))
            future = list(fm.get("future_dates", []) or [])
            if not future:
                continue
            first = pd.Timestamp(future[0]).to_period("M").to_timestamp(how="start")
            starts.append(first)
        except Exception:
            continue
    return max(starts) if starts else None


def _run_ids_from_aggregate_metadata(metadata: dict) -> dict[str, str]:
    """Extract the exact component run IDs used by a saved aggregate."""
    out: dict[str, str] = {}
    for key, raw in dict(metadata.get("component_forecast_stores", {}) or {}).items():
        path = Path(str(raw))
        try:
            run_id = path.parents[1].name  # <run>/forecasts/<forecast_name>
        except IndexError:
            continue
        if run_id:
            out[str(key)] = str(run_id)
    return out


def _tax_scenarios_from_payload(payload: dict | None) -> dict[str, dict]:
    return scenario_set_to_tax_scenarios(payload)


# ---------------------------------------------------------------------------
# Aggregate page callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("agg-select", "options"),
    Output("agg-select", "value"),
    Input("vintage-select", "value"),
    Input("registry-store", "data"),
    State("agg-select", "value"),
)
def aggregate_options(vintage: str | None, _: dict | None, current: str | None):
    """Aggregate runs for the selected vintage.

    Failed stores are listed rather than filtered out. A dropdown that silently
    shows nothing gives no way to tell "none saved" from "saved but rejected by
    the scan", and the registry already holds the reason.
    """
    if not vintage:
        return [], None
    frame = list_aggregates(REGISTRY_PATH, present_only=True)
    if frame.empty:
        return [], None
    frame = frame.loc[frame["vintage"].astype(str) == str(vintage)]
    if frame.empty:
        return [], None
    frame = frame.sort_values(
        ["status", "promoted", "created_at_utc", "aggregate_run_id"],
        ascending=[True, False, False, True],
    )
    options = []
    for _index, row in frame.iterrows():
        tags = []
        if str(row.get("status")) != "complete":
            tags.append(str(row.get("status", "unusable")).upper())
        if bool(row.get("promoted")):
            tags.append("PROMOTED")
        if not bool(row.get("has_display")):
            tags.append("display pending")
        if row.get("scenario_active"):
            tags.append("tax scenario")
        suffix = " · " + " · ".join(tags) if tags else ""
        options.append(
            {
                "label": f"{_short_run(row['aggregate_run_id'])}{suffix}",
                "value": str(row["aggregate_run_id"]),
            }
        )
    values = [item["value"] for item in options]
    usable = frame.loc[frame["status"].astype(str) == "complete", "aggregate_run_id"]
    default = str(usable.iloc[0]) if not usable.empty else values[0]
    return options, current if current in values else default


def _no_aggregate_banner(vintage: str | None) -> html.Div:
    """Say what the registry actually holds instead of showing a blank page."""
    frame = list_aggregates(REGISTRY_PATH, present_only=True)
    if frame.empty:
        return html.Div(
            [
                html.Strong("No aggregate saved. "),
                html.Span(
                    "Nothing was found under results/hicp_energy_aggregate/. Run "
                    "notebook 10 or run_aggregate, then press Refresh."
                ),
            ],
            className="selection-banner",
        )
    here = frame.loc[frame["vintage"].astype(str) == str(vintage)]
    if here.empty:
        others = ", ".join(sorted(frame["vintage"].astype(str).unique()))
        return html.Div(
            [
                html.Strong(f"No aggregate for vintage {vintage}. "),
                html.Span(f"The registry holds aggregates for: {others}."),
            ],
            className="selection-banner",
        )
    failed = here.loc[here["status"].astype(str) != "complete"]
    if not failed.empty:
        reason = str(failed["error"].dropna().iloc[0]) if failed["error"].notna().any() else "no reason recorded"
        return html.Div(
            [
                html.Strong(f"{len(failed)} aggregate store(s) rejected by the scan: "),
                html.Span(reason),
            ],
            className="banner-error",
        )
    return html.Div("Select an aggregate run.", className="selection-banner")


@callback(
    Output("agg-store", "data"),
    Output("agg-banner", "children"),
    Input("agg-select", "value"),
    Input("vintage-select", "value"),
    Input("registry-store", "data"),
)
def load_aggregate_display(aggregate_run_id: str | None, vintage: str | None, _: dict | None):
    """Read one aggregate display artefact, materialising it if needed."""
    if not vintage:
        return None, html.Div("Select a vintage.", className="selection-banner")
    if not aggregate_run_id:
        return None, _no_aggregate_banner(vintage)

    frame_rows = list_aggregates(REGISTRY_PATH, vintage=vintage, present_only=True)
    selected = frame_rows.loc[
        frame_rows["aggregate_run_id"].astype(str) == str(aggregate_run_id)
    ]
    if selected.empty:
        return None, html.Div(
            "The selected aggregate is no longer present in the registry.",
            className="banner-error",
        )
    row = selected.iloc[0]
    if str(row.get("status")) != "complete":
        return None, html.Div(
            [
                html.Strong("This aggregate store was rejected by the scan: "),
                html.Span(str(row.get("error") or "no reason recorded")),
            ],
            className="banner-error",
        )

    directory = Path(str(row["directory"]))
    display_path = directory / DISPLAY_FILENAME
    materialised = False
    try:
        if not display_path.is_file():
            if not AUTO_BUILD_DISPLAY:
                raise FileNotFoundError(
                    f"{display_path} is missing and ENERGY_BVAR_AUTO_BUILD_DISPLAY=0."
                )
            build_aggregate_display(directory, project_root=PROJECT_ROOT)
            materialised = True
        frame = load_display_artifact(display_path)
        if not (frame["record_type"].astype(str) == "fan").any():
            raise ValueError(
                f"{DISPLAY_FILENAME} loaded but carries no aggregate fan rows."
            )

        # Upgrade display artefacts created before the true posterior BVAR-fitted
        # contract.  A negative availability result is persisted in metadata so
        # legacy runs without posterior B draws are not retried on every load.
        current_meta = display_metadata(frame)
        needs_fitted_upgrade = "bvar_fitted_available" not in current_meta
        if current_meta.get("bvar_fitted_available") is False:
            # Displays built by the first fitted-reconstruction patch can carry
            # a false-negative caused only by the petrol/diesel metadata-key
            # mismatch (aggregate keys ``petrol``/``diesel`` versus canonical
            # run ids ``car_fuels_petrol``/``car_fuels_diesel``). Rebuild those
            # artefacts once so the backward-compatible resolver can retry.
            reason = str(current_meta.get("bvar_fitted_reason") or "")
            resolver_contract_upgraded = (
                "Aggregate metadata does not record forecast stores" in reason
            )

            # A same-run re-estimation with the updated IO layer can add the
            # compact fit cache later without changing the deterministic run_id.
            # Detect that transition so an old "unavailable" display is rebuilt.
            try:
                aggregate_meta = json.loads(
                    (directory / "metadata.json").read_text(encoding="utf-8")
                )
                stores = dict(aggregate_meta.get("component_forecast_stores", {}))
                fit_ready = bool(stores) and all(
                    (Path(str(path)).parent.parent / "fit_draws.npz").is_file()
                    or (Path(str(path)).parent.parent / "draws.npz").is_file()
                    for path in stores.values()
                )
            except Exception:
                fit_ready = False
            needs_fitted_upgrade = resolver_contract_upgraded or fit_ready
        if needs_fitted_upgrade and AUTO_BUILD_DISPLAY:
            build_aggregate_display(
                directory, project_root=PROJECT_ROOT, overwrite=True
            )
            frame = load_display_artifact(display_path)
            materialised = True
    except Exception as exc:
        return None, html.Div(
            [html.Strong("Aggregate load failed: "), html.Span(str(exc))],
            className="banner-error",
        )

    meta = display_metadata(frame)
    origin = _aggregate_forecast_origin(directory)
    if origin is not None:
        meta["forecast_origin"] = pd.Timestamp(origin).isoformat()
    if not meta.get("last_observed_hicp_energy"):
        history = frame.loc[
            (frame["record_type"].astype(str) == "history")
            & (frame["metric"].astype(str) == "yoy")
            & (frame["series"].astype(str) == "hicp_energy")
        ].dropna(subset=["value"])
        if not history.empty:
            meta["last_observed_hicp_energy"] = pd.Timestamp(history["date"].max()).isoformat()
    banner = html.Div(
        [
            html.Strong("HICP Energy"),
            html.Span(f" · vintage {vintage}"),
            html.Span(f" · aggregate {_short_run(aggregate_run_id)}"),
            html.Span(f" · weekly tax: {meta.get('weekly_tax_mode_effective', '—')}"),
            html.Span(" · display_v1 materialised now" if materialised else ""),
        ],
        className="selection-banner",
    )
    return {"frame_json": _json_frame(frame), "meta": meta}, banner


@callback(
    Output("agg-metric", "options"),
    Output("agg-metric", "value"),
    Input("agg-store", "data"),
    State("agg-metric", "value"),
)
def aggregate_metric_dropdown(store: dict | None, current: str | None):
    frame = _frame_from_store(store)
    options, default = aggregate_metric_options(frame)
    values = [item["value"] for item in options]
    return options, current if current in values else default


@callback(
    Output("agg-observed", "children"),
    Output("agg-observed-date", "children"),
    Output("agg-first", "children"),
    Output("agg-first-date", "children"),
    Output("agg-terminal", "children"),
    Output("agg-terminal-date", "children"),
    Output("agg-draws", "children"),
    Output("agg-draws-unit", "children"),
    Input("agg-store", "data"),
    Input("agg-metric", "value"),
)
def aggregate_stat_cards(store: dict | None, metric: str | None):
    frame = _frame_from_store(store)
    if frame.empty or not metric:
        return ("—", "", "—", "", "—", "", "—", "")
    values = aggregate_kpis(frame, metric)

    def stamp(value):
        return "" if value is None else pd.Timestamp(value).date().isoformat()

    draws = values.get("n_draws")
    return (
        _format_number(values.get("latest_observed")),
        stamp(values.get("latest_observed_date")),
        _format_number(values.get("first_future")),
        stamp(values.get("first_future_date")),
        _format_number(values.get("terminal")),
        stamp(values.get("terminal_date")),
        "—" if draws is None else f"{draws:,}",
        "effective" if draws is not None else "",
    )


@callback(
    Output("agg-graph", "figure"),
    Input("agg-store", "data"),
    Input("agg-metric", "value"),
    Input("agg-fan", "value"),
    Input("agg-historical-overlays", "value"),
)
def aggregate_graph(
    store: dict | None,
    metric: str | None,
    fan_mode: str | None,
    historical_overlays: list[str] | None,
):
    frame = _frame_from_store(store)
    if frame.empty:
        return empty_aggregate_figure("Select a saved aggregate run")
    if not metric:
        return empty_aggregate_figure("Select a metric")
    meta = (store or {}).get("meta", {})
    revision = f"{meta.get('aggregate_run_id', 'agg')}::{metric}"
    overlays = set(historical_overlays or [])
    return aggregate_fan_figure(
        frame, metric=metric, fan_mode=fan_mode or "68", uirevision=revision,
        forecast_origin=meta.get("forecast_origin"),
        last_observed=meta.get("last_observed_hicp_energy"),
        show_six_model_components="components" in overlays,
        show_bvar_reconstruction="bvar_reconstruction" in overlays,
    )


@callback(
    Output("agg-contrib", "figure"),
    Input("agg-store", "data"),
)
def aggregate_contributions(store: dict | None):
    frame = _frame_from_store(store)
    if frame.empty:
        return empty_aggregate_figure("Select a saved aggregate run")
    meta = (store or {}).get("meta", {})
    revision = f"{meta.get('aggregate_run_id', 'agg')}::contrib"
    return aggregate_contribution_figure(
        frame, basis="baseline", uirevision=revision,
        forecast_origin=meta.get("forecast_origin"),
        last_observed=meta.get("last_observed_hicp_energy"),
    )


@callback(
    Output("agg-scenario-store", "data"),
    Output("agg-scenario-banner", "children"),
    Input("url", "pathname"),
    Input("agg-select", "value"),
    Input("vintage-select", "value"),
    Input("scenario-store", "data"),
    Input("agg-store", "data"),
)
def compute_live_aggregate_tax_scenario(pathname,aggregate_run_id,vintage,scenario_store,aggregate_store):
    if (pathname or "/forecast")!="/aggregate": raise PreventUpdate
    if not aggregate_run_id or not vintage: return None,"Select an aggregate run."
    summaries=scenario_set_summary(scenario_store)
    if not summaries: return None,"No active component tax scenario. Configure one or more components in Scenarios."
    set_vintage=None if not scenario_store else scenario_store.get("vintage")
    if set_vintage is None:
        fp=scenario_set_payload(scenario_store,summaries[0]["model_id"]); set_vintage=dict((fp or {}).get("meta",{}) or {}).get("vintage")
    if set_vintage is not None and str(set_vintage)!=str(vintage): return None,f"The active scenario set belongs to vintage {set_vintage}; selected aggregate is {vintage}."
    tax_scenarios=_tax_scenarios_from_payload(scenario_store)
    if not tax_scenarios: return None,"The active scenario set contains no non-zero VAT or excise change."
    row=_selected_aggregate_row(vintage,aggregate_run_id)
    if row is None: return None,"The selected aggregate is unavailable."
    directory=Path(str(row["directory"])); metadata=_aggregate_metadata(directory); run_ids=_run_ids_from_aggregate_metadata(metadata)
    if not run_ids: return None,"This aggregate does not record its component forecast stores, so an exact live propagation cannot be reproduced."
    n_requested=int(metadata.get("n_aggregate_draws_requested") or 500); n_draws=min(500,max(1,n_requested)); pairing_seed=int(metadata.get("pairing_seed") or 2026); forecast_name=str(metadata.get("forecast_name") or "unconditional")
    set_forecast=None if not scenario_store else scenario_store.get("forecast_name")
    if set_forecast and str(set_forecast)!=forecast_name: return None,f"The active scenario set was built on {set_forecast!r}, while this aggregate uses {forecast_name!r}."
    signature=scenario_set_signature(scenario_store); starts={str(x["label"]):x.get("scenario_start") for x in summaries if x.get("scenario_start") is not None}; parsed=[pd.Timestamp(x) for x in starts.values()]
    combined_meta={"scenario_start":min(parsed).isoformat() if parsed else None,"scenario_starts":starts,"scenario_components":[x["model_id"] for x in summaries],"scenario_count":len(summaries),"scenario_signature":signature}
    cache_key="agg-tax-scenario-v2::"+json.dumps({"aggregate_run_id":str(aggregate_run_id),"vintage":str(vintage),"scenario_set":signature,"forecast_name":forecast_name,"n_draws":n_draws,"pairing_seed":pairing_seed},sort_keys=True,default=str)
    cached=_diskcache.get(cache_key)
    if isinstance(cached,dict): payload=cached
    else:
        try:
            outcome=run_aggregate(str(vintage),project_root=PROJECT_ROOT,results_root=RESULTS_ROOT,forecast_name=forecast_name,run_ids=run_ids,n_aggregate_draws=n_draws,pairing_seed=pairing_seed,tax_scenarios=tax_scenarios,weekly_tax_mode="strict",persist=False)
            payload=aggregate_live_scenario_payload(outcome,combined_meta); payload.setdefault("meta",{}).update({"aggregate_run_id":str(aggregate_run_id),"aggregate_forecast_name":forecast_name}); _diskcache.set(cache_key,payload,expire=3600)
        except Exception as exc:
            return None,html.Div([html.Strong("HICP Energy scenario propagation failed: "),html.Span(str(exc))],className="banner-error")
    parts=[html.Strong(f"Combined scenario · {len(summaries)} component"+("s" if len(summaries)!=1 else ""))]
    for item in summaries: parts.append(html.Span(f" · {item['label']}: VAT {item['vat_delta_pp']:+.2f} pp, excise {item['excise_delta']:+.3f} {item['excise_unit']}"))
    parts.append(html.Span(f" · propagated with {payload.get('meta',{}).get('n_draws','—')} paired aggregate draws"))
    return payload,html.Div(parts,className="selection-banner")


@callback(
    Output("agg-scen-baseline", "children"),
    Output("agg-scen-date", "children"),
    Output("agg-scen-scenario", "children"),
    Output("agg-scen-date-2", "children"),
    Output("agg-scen-impact", "children"),
    Output("agg-scen-interval", "children"),
    Output("agg-scen-draws", "children"),
    Output("agg-scen-draws-note", "children"),
    Input("agg-scenario-store", "data"),
)
def aggregate_scenario_cards(store: dict | None):
    values = aggregate_live_scenario_kpis(store)
    if values.get("date") is None:
        return "—", "", "—", "", "—", "", "—", ""
    date = pd.Timestamp(values["date"]).date().isoformat()
    return (
        f"{values['baseline']:.2f}%", date,
        f"{values['scenario']:.2f}%", date,
        f"{values['impact']:+.2f} pp",
        f"68% [{values['low']:+.2f}, {values['high']:+.2f}] pp",
        f"{int(values['n_draws']):,}" if values.get("n_draws") is not None else "—",
        "matched baseline/scenario",
    )


@callback(
    Output("agg-scenario-graph", "figure"),
    Output("agg-scenario-impact", "figure"),
    Output("agg-scenario-contrib-impact", "figure"),
    Input("agg-scenario-store", "data"),
    Input("agg-store", "data"),
    Input("agg-fan", "value"),
)
def aggregate_scenario_figures(
    scenario_store: dict | None, aggregate_store: dict | None, fan_mode: str | None
):
    if not scenario_store:
        empty = empty_aggregate_figure("Configure a component tax scenario in Scenarios")
        return empty, empty, empty
    frame = _frame_from_store(aggregate_store)
    meta = (aggregate_store or {}).get("meta", {})
    revision = f"{meta.get('aggregate_run_id', 'agg')}::live-tax"
    return (
        aggregate_live_scenario_figure(
            scenario_store, frame, fan_mode=fan_mode or "68",
            forecast_origin=meta.get("forecast_origin"),
            uirevision=revision + "::main",
        ),
        aggregate_live_impact_figure(
            scenario_store, fan_mode=fan_mode or "68", uirevision=revision + "::impact"
        ),
        aggregate_live_contribution_impact_figure(
            scenario_store, uirevision=revision + "::contrib"
        ),
    )


@callback(
    Output("agg-table", "data"),
    Input("agg-store", "data"),
)
def aggregate_table(store: dict | None):
    frame = _frame_from_store(store)
    if frame.empty:
        return []
    return aggregate_diagnostics_table(frame).to_dict("records")


if __name__ == "__main__":
    host = os.getenv("ENERGY_BVAR_DASH_HOST", "127.0.0.1")
    port = int(os.getenv("ENERGY_BVAR_DASH_PORT", "8050"))
    debug = os.getenv("ENERGY_BVAR_DASH_DEBUG", "0") in {"1", "true", "True"}
    app.run(host=host, port=port, debug=debug)
