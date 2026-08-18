"""Standalone Core-HICP dashboard pages.

Core is a derived view of the locked Headline joint BVAR.  The Headline BVAR is
never re-estimated here: NEIG + Services are aggregated draw-by-draw with the
maintained Core chain-linking utilities.  Official Core history is Eurostat
TOT_X_NRG_FOOD, currently loaded through ``headline_core``'s Python API/cache.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from dashboard_snapshot_cache import get_or_build as snapshot_get_or_build
from inflation_table_contract import (
    FORECAST_COLUMNS,
    PAIRED_EFFECT_COLUMNS,
    forecast_summary_records,
    paired_blocks_records,
    readable_table,
)
from energy_bvar_theme import (
    TOKENS,
    INFLATION_COLORS,
    apply_inflation_figure_style,
    fan_traces,
    graph_config,
)
from headline_bvar_conditional import (
    HeadlineConditionalError,
    load_saved_headline_posterior,
    manual_condition_levels,
    run_saved_headline_conditional,
)
from headline_bvar_pipeline import NATIVE_VARIABLES
from headline_bvar_display import materialize_headline_core
from headline_core import aggregate_core_draws, load_or_fetch_official_core
from headline_dashboard_slice5 import (
    CORE_LONG_LABEL,
    CORE_SERIES,
    _core_overview_values,
    _core_validation_summary,
    _frame_from_store,
    core_contribution_decomposition_figure,
    headline_forecast_figure,
)

HORIZONS = (3, 6, 12)
CORE_CONDITION_VARIABLES = ("hicp_neig", "hicp_services")
CORE_LABELS = {
    "hicp_neig": "NEIG",
    "hicp_services": "Services",
}


def _stat_card(title: str, value_id: str, note_id: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="stat-title"),
            html.Div("—", id=value_id, className="stat-value"),
            html.Div("", id=note_id, className="stat-subtitle"),
        ],
        className="stat-card",
    )


def core_forecast_page() -> html.Div:
    controls = html.Div(
        [
            html.Div(
                [
                    html.Label("Display horizon", className="selector-label"),
                    dcc.Dropdown(
                        id="core-forecast-horizon",
                        options=[{"label": f"{h} months", "value": h} for h in HORIZONS],
                        value=3,
                        clearable=False,
                    ),
                ],
                className="selector-block",
            ),
            html.Div(
                [
                    html.Label("Fan", className="selector-label"),
                    dcc.Dropdown(
                        id="core-forecast-fan",
                        options=[
                            {"label": "68%", "value": "68"},
                            {"label": "90%", "value": "90"},
                            {"label": "68% + 90%", "value": "both"},
                        ],
                        value="68",
                        clearable=False,
                    ),
                ],
                className="selector-block",
            ),
        ],
        className="chart-controls",
    )
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Core forecast & contributions", className="page-title"),
                            html.P(
                                f"{CORE_LONG_LABEL}. Official history is Eurostat "
                                "TOT_X_NRG_FOOD; model paths are NEIG + Services "
                                "aggregated draw-by-draw. No Headline BVAR re-estimation.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    controls,
                ],
                className="page-heading-row",
            ),
            html.Div(id="core-forecast-validation", className="selection-banner"),
            html.Div(
                [
                    _stat_card("Latest official Core YoY", "core-kpi-latest", "core-kpi-latest-date"),
                    _stat_card("+1m posterior mean", "core-kpi-1m", "core-kpi-1m-date"),
                    _stat_card("+3m posterior mean", "core-kpi-3m", "core-kpi-3m-date"),
                    _stat_card("+12m posterior mean", "core-kpi-12m", "core-kpi-12m-date"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    html.Div([html.Div("Key results", className="eyebrow"), html.H3("Readable Core outlook", className="panel-title"), html.P("Posterior mean first; M+1 / M+2 / M+3 prioritised, with M+6 / M+12 only when selected.", className="panel-subtitle")], className="panel-heading"),
                    readable_table("core-forecast-summary-table", FORECAST_COLUMNS, page_size=7),
                ], className="panel table-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Forecast", className="eyebrow"),
                            html.H3("Observed Core · baseline posterior", className="panel-title"),
                            html.P(
                                "The central path is the posterior mean; interval bands are posterior quantiles.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dcc.Loading(
                        dcc.Graph(id="core-forecast-graph", config=graph_config("core_forecast")),
                        type="circle",
                    ),
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Exact additive decomposition", className="eyebrow"),
                            html.H3("NEIG + Services contributions to Core YoY", className="panel-title"),
                            html.P(
                                "Historical contributions add exactly to official Core YoY; model contributions are aggregated draw-by-draw before posterior means are taken.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="core-contributions-graph",
                            config=graph_config("core_contributions"),
                        ),
                        type="circle",
                    ),
                ],
                className="panel chart-panel",
            ),
        ],
        className="page-body headline-page",
    )


def core_scenarios_page() -> html.Div:
    return html.Div(
        [
            dcc.Store(id="core-scenario-store", storage_type="session"),
            html.Div(
                [
                    html.H2("Core conditional forecast", className="page-title"),
                    html.P(
                        "Condition NEIG and/or Services in the saved Headline BVAR. Core is then re-aggregated draw-by-draw. The computational horizon remains 12 months.",
                        className="page-subtitle",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Condition horizon", className="selector-label"),
                            dcc.Dropdown(
                                id="core-scenario-horizon",
                                options=[{"label": f"{h} months", "value": h} for h in HORIZONS],
                                value=3,
                                clearable=False,
                            ),
                        ],
                        className="selector-block",
                    ),
                    html.Div(
                        [
                            html.Label("Input metric", className="selector-label"),
                            dcc.Dropdown(
                                id="core-scenario-metric",
                                options=[
                                    {"label": "Year-on-year (%)", "value": "yoy"},
                                    {"label": "HICP level", "value": "level"},
                                ],
                                value="yoy",
                                clearable=False,
                            ),
                        ],
                        className="selector-block",
                    ),
                ],
                className="chart-controls",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("NEIG path", className="selector-label"),
                            dcc.Textarea(
                                id="core-scenario-neig-values",
                                placeholder="Enter H values separated by commas. Leave blank if unconstrained.",
                                className="scenario-textarea",
                            ),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.Label("Services path", className="selector-label"),
                            dcc.Textarea(
                                id="core-scenario-services-values",
                                placeholder="Enter H values separated by commas. Leave blank if unconstrained.",
                                className="scenario-textarea",
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="two-column-grid",
            ),
            html.Div(
                [
                    html.Button("Run conditional forecast", id="core-scenario-run", n_clicks=0, className="refresh-button"),
                    html.Button("Clear", id="core-scenario-clear", n_clicks=0, className="refresh-button"),
                ],
                style={"display": "flex", "gap": "10px", "marginTop": "12px"},
            ),
            html.Div(id="core-scenario-status", className="selection-banner"),
            html.Div(
                [
                    html.Div([html.Div("Key scenario results", className="eyebrow"), html.H3("Core conditional effect by horizon", className="panel-title"), html.P("Baseline, conditional path and paired effect from the same frozen Core scenario draws.", className="panel-subtitle")], className="panel-heading"),
                    readable_table("core-scenario-summary-table", PAIRED_EFFECT_COLUMNS, page_size=7),
                ], className="panel table-panel",
            ),
            html.Div(
                [
                    html.Div("Baseline vs conditional", className="eyebrow"),
                    html.H3("Core HICP — year-on-year", className="panel-title"),
                    dcc.Loading(
                        dcc.Graph(id="core-scenario-main", config=graph_config("core_scenario_main")),
                        type="circle",
                    ),
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div("Conditional impact", className="eyebrow"),
                    html.H3("Conditional − baseline", className="panel-title"),
                    dcc.Loading(
                        dcc.Graph(id="core-scenario-impact", config=graph_config("core_scenario_impact")),
                        type="circle",
                    ),
                ],
                className="panel chart-panel",
            ),
        ],
        className="page-body headline-page",
    )


def _run_directory(results_root: Path, store: dict | None) -> Path:
    meta = dict((store or {}).get("meta") or {})
    context = dict((store or {}).get("context") or {})
    vintage = str(meta.get("vintage") or context.get("vintage") or "")
    run_id = str(meta.get("run_id") or meta.get("headline_run_id") or context.get("run_id") or "")
    model_id = str(meta.get("model_id") or context.get("model_id") or "headline_joint")
    if model_id != "headline_joint" or not vintage or not run_id:
        raise HeadlineConditionalError("Select a complete Headline/Core run first.")
    path = results_root / "headline_joint" / vintage / run_id
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _qsummary(paths: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(paths, dtype=float)
    q = np.nanquantile(arr, [0.05, 0.16, 0.50, 0.84, 0.95], axis=0)
    return {
        "mean": np.nanmean(arr, axis=0).tolist(),
        "q05": q[0].tolist(),
        "q16": q[1].tolist(),
        "q50": q[2].tolist(),
        "q84": q[3].tolist(),
        "q95": q[4].tolist(),
    }


def _core_scenario_payload(run_dir: Path, headline_payload: dict) -> dict:
    scenario_dir = Path(str((headline_payload.get("meta") or {}).get("directory") or ""))
    draws_path = scenario_dir / "scenario_draws.npz"
    if not draws_path.is_file():
        raise FileNotFoundError(draws_path)

    native_history = pd.read_csv(
        run_dir / "headline_native_history.csv", index_col=0, parse_dates=True
    ).sort_index()
    native_history.index = pd.DatetimeIndex(native_history.index, name="date")
    weights = pd.read_csv(run_dir / "headline_weights_annual.csv", index_col=0)
    official_core, _ = load_or_fetch_official_core(
        run_dir,
        native_history=native_history,
        observed_end=pd.Timestamp(native_history.index.max()),
    )

    with np.load(draws_path, allow_pickle=False) as archive:
        dates = pd.DatetimeIndex(pd.to_datetime(archive["future_dates"]), name="date")
        baseline_native = np.asarray(archive["baseline_native_level_paths"], dtype=float)
        conditional_native = np.asarray(archive["conditional_native_level_paths"], dtype=float)

    baseline = aggregate_core_draws(
        baseline_native,
        dates,
        native_variables=NATIVE_VARIABLES,
        native_history=native_history,
        annual_weights=weights,
        official_core=official_core,
    )
    conditional = aggregate_core_draws(
        conditional_native,
        dates,
        native_variables=NATIVE_VARIABLES,
        native_history=native_history,
        annual_weights=weights,
        official_core=official_core,
    )
    base_level = np.asarray(baseline["core_level_paths"], dtype=float)
    cond_level = np.asarray(conditional["core_level_paths"], dtype=float)
    base_yoy = np.asarray(baseline["core_yoy_paths"], dtype=float)
    cond_yoy = np.asarray(conditional["core_yoy_paths"], dtype=float)

    return {
        "dates": [pd.Timestamp(x).isoformat() for x in dates],
        "meta": dict(headline_payload.get("meta") or {}),
        "baseline": {"level": _qsummary(base_level), "yoy": _qsummary(base_yoy)},
        "conditional": {"level": _qsummary(cond_level), "yoy": _qsummary(cond_yoy)},
        "impact": {
            "level": _qsummary(cond_level - base_level),
            "yoy": _qsummary(cond_yoy - base_yoy),
        },
        "core_level_additivity_error": max(
            float(baseline["core_drawwise_level_additivity_max_abs_error"]),
            float(conditional["core_drawwise_level_additivity_max_abs_error"]),
        ),
        "core_yoy_additivity_error": max(
            float(baseline["core_yoy_contribution_additivity_max_abs_error"]),
            float(conditional["core_yoy_contribution_additivity_max_abs_error"]),
        ),
    }


# GRAPH_EXPORT_READABILITY_G5_CORE_V1
def _empty(message: str, height: int = 440) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper", text=message,
        showarrow=False, font=dict(size=13, color=TOKENS["muted"]),
    )
    apply_inflation_figure_style(fig, uirevision="core-empty", height=height)
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _add_fan(fig: go.Figure, dates, block: dict, *, name: str, color: str) -> None:
    quantiles = {k: np.asarray(block.get(k, []), dtype=float) for k in ("q05", "q16", "q50", "q84", "q95")}
    mean = np.asarray(block.get("mean", []), dtype=float)
    if len(mean) != len(dates) or len(quantiles["q50"]) != len(dates):
        return
    for trace in fan_traces(
        dates, quantiles, bands=("68",), color=color, name=name, show_median=False
    ):
        fig.add_trace(trace)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=mean,
            mode="lines",
            name=f"{name} · posterior mean",
            line=dict(color=color, width=2.2),
        )
    )


def core_scenario_main_figure(payload: dict | None, headline_store: dict | None) -> go.Figure:
    if not payload:
        return _empty("Run a Core conditional forecast.", 500)
    dates = pd.DatetimeIndex(pd.to_datetime(payload.get("dates") or []))
    fig = go.Figure()

    frame = _frame_from_store(headline_store)
    if not frame.empty:
        hist = frame.loc[
            frame["record_type"].astype(str).eq("history")
            & frame["series"].astype(str).eq(CORE_SERIES)
            & frame["metric"].astype(str).eq("yoy")
        ].sort_values("date")
        if not hist.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist["date"], y=hist["value"], mode="lines",
                    name="Observed Core", line=dict(width=2, color=INFLATION_COLORS["observed"]),
                )
            )

    _add_fan(fig, dates, (payload.get("baseline") or {}).get("yoy") or {}, name="Baseline", color=INFLATION_COLORS["baseline"])
    _add_fan(fig, dates, (payload.get("conditional") or {}).get("yoy") or {}, name="Conditional", color=INFLATION_COLORS["conditional"])
    apply_inflation_figure_style(
        fig,
        uirevision=f"core-scenario::{(payload.get('meta') or {}).get('scenario_id')}",
        height=500,
        y_title="% y/y",
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def core_scenario_impact_figure(payload: dict | None) -> go.Figure:
    if not payload:
        return _empty("Run a Core conditional forecast.")
    dates = pd.DatetimeIndex(pd.to_datetime(payload.get("dates") or []))
    block = (payload.get("impact") or {}).get("yoy") or {}
    mean = np.asarray(block.get("mean", []), dtype=float)
    if len(mean) != len(dates):
        return _empty("Core conditional impact is unavailable.")
    fig = go.Figure()
    quantiles = {k: np.asarray(block.get(k, []), dtype=float) for k in ("q05", "q16", "q50", "q84", "q95")}
    for trace in fan_traces(
        dates, quantiles, bands=("68",), color=INFLATION_COLORS["impact"], name="Impact", show_median=False
    ):
        fig.add_trace(trace)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=mean,
            mode="lines+markers",
            name="Posterior mean impact",
            line=dict(width=2, color=INFLATION_COLORS["impact"]),
            marker=dict(size=5),
        )
    )
    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    apply_inflation_figure_style(
        fig,
        uirevision=f"core-impact::{(payload.get('meta') or {}).get('scenario_id')}",
        height=430,
        y_title="percentage points",
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def register_core_dashboard_callbacks(
    app,
    *,
    results_root,
    project_root=None,
    store_id="data-store",
):
    results_root = Path(results_root).resolve()

    @app.callback(
        Output("core-kpi-latest", "children"),
        Output("core-kpi-latest-date", "children"),
        Output("core-kpi-1m", "children"),
        Output("core-kpi-1m-date", "children"),
        Output("core-kpi-3m", "children"),
        Output("core-kpi-3m-date", "children"),
        Output("core-kpi-12m", "children"),
        Output("core-kpi-12m-date", "children"),
        Output("core-forecast-validation", "children"),
        Output("core-forecast-graph", "figure"),
        Output("core-contributions-graph", "figure"),
        Output("core-forecast-summary-table", "data"),
        Input(store_id, "data"),
        Input("core-forecast-horizon", "value"),
        Input("core-forecast-fan", "value"),
    )
    def core_forecast(store, horizon, fan):
        frame = _frame_from_store(store)
        h = int(horizon or 3)
        if not frame.empty:
            core_ready = (
                frame["series"].astype(str).eq(CORE_SERIES)
                & frame["record_type"].astype(str).isin(["history", "fan"])
            ).any()
            if not core_ready:
                try:
                    run_dir = _run_directory(results_root, store)
                    forecast_name = str(
                        (store or {}).get("context", {}).get("forecast_name")
                        or "unconditional"
                    )

                    def _build_core_frame():
                        display_path = materialize_headline_core(
                            run_dir,
                            forecast_name=forecast_name,
                        )
                        cached_frame = pd.read_parquet(display_path)
                        cached_frame["date"] = pd.to_datetime(
                            cached_frame["date"],
                            errors="coerce",
                        )
                        return cached_frame

                    frame = snapshot_get_or_build(
                        "headline_core_display",
                        (str(run_dir.resolve()), forecast_name),
                        _build_core_frame,
                    )
                except Exception:
                    # Validation/status rows below will surface the exact Core
                    # availability error without invalidating the Headline run.
                    pass
        if frame.empty:
            empty = _empty("Select a Core/Headline run.")
            return (
                "—", "", "—", "", "—", "", "—", "",
                html.Div("Select a Core/Headline run."), empty, empty, [],
            )
        k = _core_overview_values(frame)
        fig = headline_forecast_figure(
            frame,
            series=CORE_SERIES,
            metric="yoy",
            horizon=h,
            fan_mode=fan or "68",
            statistic="mean",
            history_months=None,
            include_fitted=False,
        )
        return (
            k[0][0], k[0][1], k[1][0], k[1][1], k[2][0], k[2][1], k[3][0], k[3][1],
            _core_validation_summary(frame),
            fig,
            core_contribution_decomposition_figure(frame, h),
            forecast_summary_records(
                frame, series=CORE_SERIES, metric="yoy", max_months=h
            ),
        )

    @app.callback(
        Output("core-scenario-store", "data"),
        Output("core-scenario-status", "children"),
        Input("core-scenario-run", "n_clicks"),
        Input("core-scenario-clear", "n_clicks"),
        State(store_id, "data"),
        State("core-scenario-horizon", "value"),
        State("core-scenario-metric", "value"),
        State("core-scenario-neig-values", "value"),
        State("core-scenario-services-values", "value"),
        prevent_initial_call=True,
    )
    def run_or_clear(run_clicks, clear_clicks, headline_store, horizon, metric, neig_text, services_text):
        from dash import ctx

        if ctx.triggered_id == "core-scenario-clear":
            return None, "Core conditional forecast cleared."
        if ctx.triggered_id != "core-scenario-run" or not run_clicks:
            raise PreventUpdate
        try:
            H = int(horizon or 3)
            run_dir = _run_directory(results_root, headline_store)
            posterior = load_saved_headline_posterior(run_dir, project_root=project_root)
            conditions = {}
            raw_inputs = {
                "hicp_neig": str(neig_text or "").strip(),
                "hicp_services": str(services_text or "").strip(),
            }
            for variable, text in raw_inputs.items():
                if not text:
                    continue
                _, values = manual_condition_levels(
                    posterior,
                    variable=variable,
                    metric=metric or "yoy",
                    values=text,
                    H=H,
                )
                conditions[variable] = values
            if not conditions:
                raise HeadlineConditionalError(
                    "Enter a NEIG path, a Services path, or both."
                )
            lineage = {
                "source_type": "core_manual",
                "condition_metric": str(metric or "yoy"),
                "core_condition_variables": sorted(conditions),
                "core_derived_from": list(CORE_CONDITION_VARIABLES),
                "core_aggregation": "draw_wise_neig_services",
            }
            headline_payload = run_saved_headline_conditional(
                run_dir,
                native_level_conditions=conditions,
                H=H,
                lineage=lineage,
                project_root=project_root,
                persist=True,
            )
            payload = _core_scenario_payload(run_dir, headline_payload)
            m = payload["meta"]
            return payload, html.Div(
                [
                    html.Strong("Core conditional complete"),
                    html.Span(f" · vintage {m['headline_vintage']}"),
                    html.Span(f" · {m['n_draws']} paired draws"),
                    html.Span(f" · condition {m['condition_horizon']}m / compute {m['computational_horizon']}m"),
                    html.Span(f" · lineage {m['lineage_verification_status']}"),
                    html.Span(" · BVAR re-estimation: NO"),
                ]
            )
        except Exception as exc:
            return None, html.Div(
                [html.Strong("Core conditional failed: "), html.Span(str(exc))],
                className="banner-error",
            )

    @app.callback(
        Output("core-scenario-main", "figure"),
        Output("core-scenario-impact", "figure"),
        Output("core-scenario-summary-table", "data"),
        Input("core-scenario-store", "data"),
        Input(store_id, "data"),
        Input("core-scenario-horizon", "value"),
    )
    def scenario_figures(payload, headline_store, horizon):
        h = int(horizon or 3)
        table = paired_blocks_records(
            (payload or {}).get("dates") or [],
            ((payload or {}).get("baseline") or {}).get("yoy") or {},
            ((payload or {}).get("conditional") or {}).get("yoy") or {},
            ((payload or {}).get("impact") or {}).get("yoy") or {},
            max_months=h,
        )
        return (
            core_scenario_main_figure(payload, headline_store),
            core_scenario_impact_figure(payload),
            table,
        )


__all__ = [
    "core_forecast_page",
    "core_scenarios_page",
    "core_scenario_main_figure",
    "core_scenario_impact_figure",
    "register_core_dashboard_callbacks",
]
