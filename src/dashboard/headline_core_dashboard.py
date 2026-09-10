"""Standalone Core-HICP dashboard pages.

Core is a derived view of the locked Headline joint BVAR.  The Headline BVAR is
never re-estimated here: NEIG + Services are aggregated draw-by-draw with the
maintained Core chain-linking utilities.  Official Core history is Eurostat
TOT_X_NRG_FOOD, currently loaded through ``headline_core``'s Python API/cache.
"""
from __future__ import annotations
CONDITIONAL_WINDOWS_HARMONIZED_V1_CORE_DASH = True

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



# CORE_CONDITIONAL_HARMONISATION_V1
def _core_persisted_component_yoy_q50(
    posterior,
    *,
    variable: str,
    condition_start: int,
    condition_end: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    import json

    if variable not in CORE_CONDITION_VARIABLES:
        raise HeadlineConditionalError(
            f"Core conditioning variable must be one of "
            f"{CORE_CONDITION_VARIABLES}; received {variable!r}."
        )

    start = int(condition_start)
    end = int(condition_end)
    if start < 1 or end < start or end > 12:
        raise HeadlineConditionalError(
            "Persisted-baseline window must satisfy "
            f"1 <= start <= end <= 12; received M+{start}..M+{end}."
        )

    forecast_directory = (
        posterior.run_directory / "forecasts" / "unconditional"
    )
    draws_path = forecast_directory / "headline_draws.npz"
    metadata_path = forecast_directory / "headline_metadata.json"

    if not draws_path.is_file():
        raise HeadlineConditionalError(
            "Persisted unconditional Headline draws are missing: "
            f"{draws_path}"
        )
    if not metadata_path.is_file():
        raise HeadlineConditionalError(
            "Persisted unconditional Headline metadata are missing: "
            f"{metadata_path}"
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HeadlineConditionalError(
            "Persisted unconditional Headline metadata are unreadable."
        ) from exc
    if not isinstance(metadata, dict):
        raise HeadlineConditionalError(
            "Persisted unconditional Headline metadata have invalid structure."
        )

    path_values = metadata.get("path_dates")
    future_values = metadata.get("future_dates")
    if path_values is None or future_values is None:
        raise HeadlineConditionalError(
            "Persisted unconditional Headline calendar is incomplete."
        )

    path_dates = (
        pd.DatetimeIndex(pd.to_datetime(path_values))
        .to_period("M")
        .to_timestamp(how="start")
    )
    future_dates = (
        pd.DatetimeIndex(pd.to_datetime(future_values))
        .to_period("M")
        .to_timestamp(how="start")
    )
    tail = int(metadata.get("tail_length") or 0)

    with np.load(draws_path, allow_pickle=False) as archive:
        if "component_yoy_paths" not in archive.files:
            raise HeadlineConditionalError(
                "Persisted unconditional Headline draws do not contain "
                "component_yoy_paths."
            )
        component_yoy_paths = np.asarray(
            archive["component_yoy_paths"],
            dtype=float,
        ).copy()

    native_variables = list(NATIVE_VARIABLES)
    if component_yoy_paths.ndim != 3:
        raise HeadlineConditionalError(
            "Persisted component_yoy_paths must be three-dimensional."
        )
    if component_yoy_paths.shape[2] != len(native_variables):
        raise HeadlineConditionalError(
            "Persisted component_yoy_paths component dimension changed: "
            f"{component_yoy_paths.shape[2]} != {len(native_variables)}."
        )
    if len(path_dates) != component_yoy_paths.shape[1]:
        raise HeadlineConditionalError(
            "Persisted Headline path_dates length differs from "
            "component_yoy_paths."
        )
    if tail < 0 or tail > component_yoy_paths.shape[1]:
        raise HeadlineConditionalError(
            f"Persisted Headline tail_length is invalid: {tail}."
        )

    future_draws = component_yoy_paths[:, tail:, :]
    if len(future_dates) != future_draws.shape[1]:
        raise HeadlineConditionalError(
            "Persisted Headline future_dates length differs from "
            "the post-tail component forecast."
        )
    if not path_dates[tail:].equals(future_dates):
        raise HeadlineConditionalError(
            "Persisted Headline path_dates[tail:] does not equal future_dates."
        )
    if end > len(future_dates):
        raise HeadlineConditionalError(
            f"Persisted Headline baseline has only {len(future_dates)} "
            f"future months; requested M+{end}."
        )

    variable_index = native_variables.index(variable)
    selected_draws = future_draws[:, start - 1:end, variable_index]
    if selected_draws.shape[1] != end - start + 1:
        raise HeadlineConditionalError(
            "Persisted Core component q50 window length is inconsistent."
        )

    q50 = np.nanquantile(selected_draws, 0.50, axis=0)
    if (
        q50.ndim != 1
        or len(q50) != end - start + 1
        or not np.isfinite(q50).all()
    ):
        raise HeadlineConditionalError(
            "Persisted unconditional Core component YoY q50 is invalid."
        )

    return (
        pd.DatetimeIndex(future_dates[start - 1:end], name="date"),
        np.asarray(q50, dtype=float),
    )


def _core_path_editor(prefix: str, label: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(label.upper() + " PATH", className="eyebrow"),
                    html.H3(
                        f"Conditioning path — {label}",
                        className="panel-title",
                    ),
                    html.P(
                        "Leave every active cell blank to leave this component "
                        "unconstrained. Otherwise enter one value per active month.",
                        className="panel-subtitle",
                    ),
                ],
                className="panel-heading",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Set all", className="selector-label"),
                            html.Div(
                                [
                                    dcc.Input(
                                        id=f"core-{prefix}-fill-all-value",
                                        type="number",
                                        step="any",
                                        placeholder="Value",
                                        style={"width": "100%"},
                                    ),
                                    html.Button(
                                        "Apply",
                                        id=f"core-{prefix}-fill-all",
                                        n_clicks=0,
                                        className="refresh-button",
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns":
                                        "minmax(120px, 1fr) auto",
                                    "gap": "8px",
                                    "alignItems": "center",
                                },
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Linear path",
                                className="selector-label",
                            ),
                            html.Div(
                                [
                                    dcc.Input(
                                        id=f"core-{prefix}-linear-start",
                                        type="number",
                                        step="any",
                                        placeholder="Start",
                                        style={"width": "100%"},
                                    ),
                                    dcc.Input(
                                        id=f"core-{prefix}-linear-end",
                                        type="number",
                                        step="any",
                                        placeholder="End",
                                        style={"width": "100%"},
                                    ),
                                    html.Button(
                                        "Fill",
                                        id=f"core-{prefix}-fill-linear",
                                        n_clicks=0,
                                        className="refresh-button",
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns":
                                        "minmax(100px, 1fr) "
                                        "minmax(100px, 1fr) auto",
                                    "gap": "8px",
                                    "alignItems": "center",
                                },
                            ),
                        ]
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns":
                        "repeat(auto-fit, minmax(280px, 1fr))",
                    "gap": "12px",
                    "marginBottom": "14px",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                f"M+{month}",
                                className="selector-label",
                            ),
                            dcc.Input(
                                id=f"core-{prefix}-value-m{month}",
                                type="number",
                                step="any",
                                style={"width": "100%"},
                            ),
                        ],
                        id=f"core-{prefix}-value-wrap-m{month}",
                        style={
                            "display": "block" if month <= 3 else "none"
                        },
                    )
                    for month in range(1, 13)
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns":
                        "repeat(auto-fit, minmax(100px, 1fr))",
                    "gap": "10px",
                },
            ),
            dcc.Textarea(
                id=f"core-scenario-{prefix}-values",
                style={"display": "none"},
            ),
        ],
        className="panel",
    )


def _core_summary_slice(block: dict | None, n: int) -> dict:
    out = {}
    for key, values in dict(block or {}).items():
        if isinstance(values, (list, tuple, np.ndarray)):
            out[key] = list(values)[:n]
        else:
            out[key] = values
    return out


def core_conditioning_assumption_figure(
    payload: dict | None,
    variable: str,
    horizon: int = 6,
) -> go.Figure:
    label = {
        "hicp_neig": "NEIG",
        "hicp_services": "Services",
    }.get(variable, variable)

    if not payload:
        return _empty(f"Run a Core conditional scenario for {label}.", 490)

    entry = dict(
        ((payload or {}).get("conditioning_assumptions") or {}).get(variable)
        or {}
    )
    if not entry:
        return _empty(f"{label} is not conditioned in this scenario.", 490)

    all_dates = pd.DatetimeIndex(
        pd.to_datetime((payload or {}).get("dates") or [])
    )
    n = min(max(int(horizon or 6), 1), len(all_dates))
    dates = all_dates[:n]
    if n < 1:
        return _empty("Core conditional scenario has no future dates.", 490)

    baseline = dict(entry.get("baseline_fan") or {})
    q16 = np.asarray(baseline.get("q16") or [], dtype=float)[:n]
    q50 = np.asarray(baseline.get("q50") or [], dtype=float)[:n]
    q84 = np.asarray(baseline.get("q84") or [], dtype=float)[:n]
    if (
        len(q16) != n
        or len(q50) != n
        or len(q84) != n
        or not np.isfinite(q16).all()
        or not np.isfinite(q50).all()
        or not np.isfinite(q84).all()
    ):
        return _empty(
            f"Unconditional {label} YoY fan is unavailable.",
            490,
        )

    observed_dates = pd.DatetimeIndex(
        pd.to_datetime(entry.get("observed_dates") or [])
    )
    observed_values = np.asarray(
        entry.get("observed_values") or [],
        dtype=float,
    )
    if len(observed_dates) != len(observed_values):
        return _empty(
            f"Observed {label} YoY history is misaligned.",
            490,
        )

    fig = go.Figure()

    if len(observed_dates):
        finite = np.isfinite(observed_values)
        observed_dates = observed_dates[finite]
        observed_values = observed_values[finite]
        if len(observed_dates):
            fig.add_trace(
                go.Scatter(
                    x=observed_dates,
                    y=observed_values,
                    mode="lines",
                    name="Observed",
                    line={"color": "#111827", "width": 1.8},
                )
            )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=q84,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            legendgroup="baseline",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=q16,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(100,116,139,0.20)",
            name="Unconditional forecast 68%",
            hoverinfo="skip",
            legendgroup="baseline",
        )
    )

    median_dates = dates
    median_values = q50
    if len(observed_dates) and len(dates):
        last_observed_month = (
            pd.Timestamp(observed_dates[-1])
            .to_period("M")
            .to_timestamp(how="start")
        )
        first_forecast_month = (
            pd.Timestamp(dates[0])
            .to_period("M")
            .to_timestamp(how="start")
        )
        if (
            first_forecast_month
            == last_observed_month + pd.offsets.MonthBegin(1)
        ):
            median_dates = pd.DatetimeIndex(
                [last_observed_month, *list(dates)]
            )
            median_values = np.r_[
                float(observed_values[-1]),
                q50,
            ]

    fig.add_trace(
        go.Scatter(
            x=median_dates,
            y=median_values,
            mode="lines",
            name="Unconditional forecast median",
            line={"color": "#64748b", "width": 2},
        )
    )

    start = int(entry.get("condition_start_offset") or 1)
    end = int(entry.get("condition_end_offset") or start)
    imposed = np.asarray(entry.get("imposed_values") or [], dtype=float)
    first = max(start - 1, 0)
    last = min(end, n)
    if first < last and len(imposed):
        imposed_dates = dates[first:last]
        visible_count = len(imposed_dates)
        imposed_visible = imposed[:visible_count]
        if (
            len(imposed_visible) == visible_count
            and np.isfinite(imposed_visible).all()
        ):
            fig.add_trace(
                go.Scatter(
                    x=imposed_dates,
                    y=imposed_visible,
                    mode="lines+markers",
                    name="Imposed future path",
                    line={
                        "color": "#009FE3",
                        "width": 2.6,
                        "dash": "dash",
                    },
                    marker={"size": 5},
                )
            )

    fig.update_layout(
        template="plotly_white",
        margin={"l": 64, "r": 26, "t": 66, "b": 54},
        height=490,
        title={
            "text": f"Conditioning assumption — {label}",
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 18, "color": "#111827"},
        },
        font={
            "family": "Inter, Segoe UI, sans-serif",
            "color": "#374151",
            "size": 12,
        },
        yaxis_title="% y/y",
        hovermode="x unified",
        dragmode="pan",
        legend={
            "orientation": "h",
            "y": 1.08,
            "x": 1,
            "xanchor": "right",
            "font": {"size": 11},
        },
        uirevision=(
            "core-conditioning-assumption::"
            f"{(payload.get('meta') or {}).get('scenario_id')}::"
            f"{variable}::{n}"
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, linecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#eef0f3", zerolinecolor="#d1d5db")
    return fig


def core_scenarios_page() -> html.Div:
    return html.Div([dcc.Store(id='core-scenario-store', storage_type='session'), dcc.Store(id='core-condition-set-store', storage_type='memory'), dcc.Store(id='core-condition-selected-store', storage_type='memory'), html.Div([html.H2('Core conditional forecast', className='page-title'), html.P('Condition NEIG and/or Services in the saved Headline BVAR. Core is re-aggregated draw-by-draw. The computational horizon remains 12 months.', className='page-subtitle')]), html.Div([html.Div([html.Label('Condition start', className='selector-label'), dcc.Dropdown(id='core-scenario-start', options=[{'label': f'M+{h}', 'value': h} for h in range(1, 13)], value=1, clearable=False)], className='selector-block'), html.Div([html.Label('Condition end', className='selector-label'), dcc.Dropdown(id='core-scenario-horizon', options=[{'label': f'M+{h}', 'value': h} for h in range(1, 13)], value=3, clearable=False)], className='selector-block'), html.Div([html.Label('Display horizon', className='selector-label'), dcc.Dropdown(id='core-scenario-display-horizon', options=[{'label': f'{h} months', 'value': h} for h in range(1, 13)], value=6, clearable=False)], className='selector-block'), html.Div([html.Label('Conditioning mode', className='selector-label'), dcc.Dropdown(id='core-scenario-metric', options=[{'label': 'YoY target (%)', 'value': 'yoy'}, {'label': 'Δ vs unconditional (pp)', 'value': 'delta_pp'}], value='yoy', clearable=False)], className='selector-block')], className='chart-controls'), html.Div([_core_path_editor('neig', 'NEIG'), _core_path_editor('services', 'Services')], className='two-column-grid'), html.Div([html.Div('ACTIVE SCENARIOS', className='eyebrow'), html.H3('Calculated Core conditionals', className='panel-title'), html.P('Add / update computes the marginal scenario immediately. Click NEIG or Services for the marginal effect, or Joint effect for the simultaneous Core scenario.', className='panel-subtitle'), html.Button('Add / update conditional', id='core-condition-add', n_clicks=0, className='refresh-button'), html.Div([html.Div([html.Button(id='core-card-select-hicp-neig', n_clicks=0, style={'width': '100%', 'textAlign': 'left'}), html.Button('×', id='core-card-remove-hicp-neig', n_clicks=0, title='Remove NEIG scenario', style={'border': 'none', 'background': 'transparent', 'cursor': 'pointer', 'fontSize': '18px'})], id='core-card-wrap-hicp-neig', style={'display': 'none'}), html.Div([html.Button(id='core-card-select-hicp-services', n_clicks=0, style={'width': '100%', 'textAlign': 'left'}), html.Button('×', id='core-card-remove-hicp-services', n_clicks=0, title='Remove Services scenario', style={'border': 'none', 'background': 'transparent', 'cursor': 'pointer', 'fontSize': '18px'})], id='core-card-wrap-hicp-services', style={'display': 'none'}), html.Div([html.Button(id='core-card-select-joint', n_clicks=0, style={'width': '100%', 'textAlign': 'left'})], id='core-card-wrap-joint', style={'display': 'none'})], id='core-computed-scenario-cards', style={'marginTop': '12px'}), html.Button('Reset all', id='core-condition-reset', n_clicks=0, style={'border': 'none', 'background': 'transparent', 'textDecoration': 'underline', 'cursor': 'pointer', 'padding': '8px 2px', 'fontSize': '13px'})], className='panel'), html.Div([html.Progress(id='core-scenario-progress', value=0, max=100, className='estimation-progress'), html.Div([html.Div('Idle', id='core-scenario-progress-phase', className='estimation-phase'), html.Div('Run a conditional forecast to see progress.', id='core-scenario-progress-detail', className='estimation-progress-detail')], className='estimation-progress-text')], className='estimation-progress-wrap'), html.Div(id='core-scenario-status', className='selection-banner'), html.Div([html.Div([html.Div('Conditioning assumptions', className='eyebrow'), html.H3('NEIG and Services · YoY', className='panel-title'), html.P('Observed history, unconditional forecast and exact imposed paths for each constrained Core component.', className='panel-subtitle')], className='panel-heading'), html.Div([dcc.Loading(dcc.Graph(id='core-neig-conditioning-assumption', config=graph_config('core_neig_conditioning_assumption')), type='circle'), dcc.Loading(dcc.Graph(id='core-services-conditioning-assumption', config=graph_config('core_services_conditioning_assumption')), type='circle')], className='two-column-grid')], className='panel chart-panel', id='core-results-conditioning', style={'display': 'none'}), html.Div([html.Div([html.Div('Key scenario results', className='eyebrow'), html.H3('Core conditional effect by horizon', className='panel-title'), html.P('Baseline, conditional path and paired effect from the same frozen Core scenario draws.', className='panel-subtitle')], className='panel-heading'), readable_table('core-scenario-summary-table', PAIRED_EFFECT_COLUMNS, page_size=12)], className='panel table-panel', id='core-results-summary', style={'display': 'none'}), html.Div([html.Div('Baseline vs conditional', className='eyebrow'), html.H3('Core HICP — year-on-year', className='panel-title'), dcc.Loading(dcc.Graph(id='core-scenario-main', config=graph_config('core_scenario_main')), type='circle')], className='panel chart-panel', id='core-results-main', style={'display': 'none'}), html.Div([html.Div('Conditional impact', className='eyebrow'), html.H3('Conditional − baseline', className='panel-title'), dcc.Loading(dcc.Graph(id='core-scenario-impact', config=graph_config('core_scenario_impact')), type='circle')], className='panel chart-panel', id='core-results-impact', style={'display': 'none'})], className='page-body headline-page')


def _run_directory(
    results_root: Path,
    display_store: dict | None,
    production_store: dict | None,
    headline_run_resolver,
) -> Path:
    """Resolve the production Headline/Core run from durable production state."""
    # DURABLE_HEADLINE_RUN_IDENTITY_V1
    meta = dict((display_store or {}).get("meta") or {})
    context = dict((display_store or {}).get("context") or {})
    display_vintage = str(meta.get("vintage") or context.get("vintage") or "")
    production_vintage = str((production_store or {}).get("vintage") or "")

    if not production_vintage:
        raise HeadlineConditionalError(
            "No production vintage is selected. "
            "Select a production vintage before loading Core results."
        )

    if display_vintage and display_vintage != production_vintage:
        raise HeadlineConditionalError(
            "Display/production vintage mismatch: "
            f"display={display_vintage!r} production={production_vintage!r}. "
            "The display store is not used to choose the Headline/Core posterior; "
            "align the displayed context with the production vintage first."
        )

    if headline_run_resolver is None:
        raise HeadlineConditionalError(
            "Headline/Core production-run resolver is unavailable."
        )

    state = dict(headline_run_resolver(production_vintage) or {})
    status = str(state.get("status") or "UNKNOWN")
    note = str(state.get("note") or "")
    candidates = [
        str(run_id)
        for run_id in (state.get("complete_run_ids") or [])
        if str(run_id)
    ]

    if status != "COMPLETE":
        candidate_text = ", ".join(candidates) if candidates else "none"
        action = (
            " Promote exactly one complete Headline run."
            if status == "AMBIGUOUS"
            else ""
        )
        raise HeadlineConditionalError(
            "Headline/Core production run is not uniquely usable: "
            f"vintage={production_vintage!r} status={status!r} "
            f"note={note!r} complete_run_ids=[{candidate_text}]."
            f"{action}"
        )

    run_id = str(state.get("run_id") or "")
    directory = state.get("directory")
    if not run_id or directory is None:
        raise HeadlineConditionalError(
            "Headline/Core resolver returned COMPLETE without both run_id and directory: "
            f"vintage={production_vintage!r} run_id={run_id!r} "
            f"directory={directory!r}."
        )

    expected = (
        Path(results_root).resolve()
        / "headline_joint"
        / production_vintage
        / run_id
    ).resolve()
    resolved = Path(directory).resolve()
    if resolved != expected:
        raise HeadlineConditionalError(
            "Headline/Core resolver directory disagrees with the dashboard results root: "
            f"resolved={resolved} expected={expected}."
        )
    if not expected.is_dir():
        raise FileNotFoundError(expected)
    return expected


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
            line=dict(color=color, width=2.2, dash=("dash" if color == INFLATION_COLORS["conditional"] else "solid")),
        )
    )



def core_scenario_main_figure(
    payload: dict | None,
    headline_store: dict | None,
    horizon: int = 12,
) -> go.Figure:
    if not payload:
        return _empty("Run a Core conditional forecast.", 500)

    all_dates = pd.DatetimeIndex(pd.to_datetime(payload.get("dates") or []))
    n = min(max(int(horizon or 12), 1), len(all_dates))
    dates = all_dates[:n]
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
                    x=hist["date"],
                    y=hist["value"],
                    mode="lines",
                    name="Observed Core",
                    line=dict(
                        width=2,
                        color=INFLATION_COLORS["observed"],
                    ),
                )
            )

    _add_fan(
        fig,
        dates,
        _core_summary_slice(
            (payload.get("baseline") or {}).get("yoy") or {},
            n,
        ),
        name="Baseline",
        color=INFLATION_COLORS["baseline"],
    )
    _add_fan(
        fig,
        dates,
        _core_summary_slice(
            (payload.get("conditional") or {}).get("yoy") or {},
            n,
        ),
        name="Conditional",
        color=INFLATION_COLORS["conditional"],
    )

    apply_inflation_figure_style(
        fig,
        uirevision=(
            f"core-scenario::"
            f"{(payload.get('meta') or {}).get('scenario_id')}::{n}"
        ),
        height=500,
        y_title="% y/y",
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig



def core_scenario_impact_figure(
    payload: dict | None,
    horizon: int = 12,
) -> go.Figure:
    if not payload:
        return _empty("Run a Core conditional forecast.")

    all_dates = pd.DatetimeIndex(pd.to_datetime(payload.get("dates") or []))
    n = min(max(int(horizon or 12), 1), len(all_dates))
    dates = all_dates[:n]
    block = _core_summary_slice(
        (payload.get("impact") or {}).get("yoy") or {},
        n,
    )
    mean = np.asarray(block.get("mean", []), dtype=float)
    if len(mean) != len(dates):
        return _empty("Core conditional impact is unavailable.")

    fig = go.Figure()
    quantiles = {
        key: np.asarray(block.get(key, []), dtype=float)
        for key in ("q05", "q16", "q50", "q84", "q95")
    }
    for trace in fan_traces(
        dates,
        quantiles,
        bands=("68",),
        color=INFLATION_COLORS["impact"],
        name="Impact",
        show_median=False,
    ):
        fig.add_trace(trace)

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=mean,
            mode="lines+markers",
            name="Posterior mean impact",
            line=dict(
                width=2,
                color=INFLATION_COLORS["impact"],
            ),
            marker=dict(size=5),
        )
    )
    fig.add_hline(
        y=0,
        line_width=1,
        line_color=TOKENS["hairline"],
    )

    apply_inflation_figure_style(
        fig,
        uirevision=(
            f"core-impact::"
            f"{(payload.get('meta') or {}).get('scenario_id')}::{n}"
        ),
        height=430,
        y_title="percentage points",
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def register_core_dashboard_callbacks(app, *, results_root, project_root=None, store_id='data-store', production_vintage_store_id='production-vintage-store', headline_run_resolver=None):
    results_root = Path(results_root).resolve()
    display_store_id = store_id
    if headline_run_resolver is None:
        raise RuntimeError('register_core_dashboard_callbacks requires headline_run_resolver.')

    @app.callback(Output('core-kpi-latest', 'children'), Output('core-kpi-latest-date', 'children'), Output('core-kpi-1m', 'children'), Output('core-kpi-1m-date', 'children'), Output('core-kpi-3m', 'children'), Output('core-kpi-3m-date', 'children'), Output('core-kpi-12m', 'children'), Output('core-kpi-12m-date', 'children'), Output('core-forecast-validation', 'children'), Output('core-forecast-graph', 'figure'), Output('core-contributions-graph', 'figure'), Output('core-forecast-summary-table', 'data'), Input(display_store_id, 'data'), Input(production_vintage_store_id, 'data'), Input('core-forecast-horizon', 'value'), Input('core-forecast-fan', 'value'), Input('url', 'pathname'))
    def core_forecast(display_store, production_store, horizon, fan, pathname):
        if (pathname or "") not in {"/forecast/core", "/core", "/core/forecast"}:
            raise PreventUpdate
        h = int(horizon or 3)
        try:
            run_dir = _run_directory(results_root, display_store, production_store, headline_run_resolver)
        except Exception as exc:
            message = f'Core forecast unavailable: {exc}'
            empty = _empty(message)
            return ('—', '', '—', '', '—', '', '—', '', html.Div(message, className='banner-error'), empty, empty, [])
        frame = _frame_from_store(display_store)
        if not frame.empty:
            core_ready = (frame['series'].astype(str).eq(CORE_SERIES) & frame['record_type'].astype(str).isin(['history', 'fan'])).any()
            if not core_ready:
                try:
                    forecast_name = str((display_store or {}).get('context', {}).get('forecast_name') or 'unconditional')

                    def _build_core_frame():
                        display_path = materialize_headline_core(run_dir, forecast_name=forecast_name)
                        cached_frame = pd.read_parquet(display_path)
                        cached_frame['date'] = pd.to_datetime(cached_frame['date'], errors='coerce')
                        return cached_frame
                    frame = snapshot_get_or_build('headline_core_display', (str(run_dir.resolve()), forecast_name), _build_core_frame)
                except Exception:
                    pass
        if frame.empty:
            empty = _empty('Select a Core/Headline run.')
            return ('—', '', '—', '', '—', '', '—', '', html.Div('Select a Core/Headline run.'), empty, empty, [])
        k = _core_overview_values(frame)
        fig = headline_forecast_figure(frame, series=CORE_SERIES, metric='yoy', horizon=h, fan_mode=fan or '68', statistic='mean', history_months=None, include_fitted=False)
        return (k[0][0], k[0][1], k[1][0], k[1][1], k[2][0], k[2][1], k[3][0], k[3][1], _core_validation_summary(frame), fig, core_contribution_decomposition_figure(frame, h), forecast_summary_records(frame, series=CORE_SERIES, metric='yoy', max_months=h))

    @app.callback(*[Output(f'core-neig-value-wrap-m{month}', 'style') for month in range(1, 13)], *[Output(f'core-services-value-wrap-m{month}', 'style') for month in range(1, 13)], Output('core-scenario-neig-values', 'value'), Output('core-scenario-services-values', 'value'), Input('core-scenario-start', 'value'), Input('core-scenario-horizon', 'value'), *[Input(f'core-neig-value-m{month}', 'value') for month in range(1, 13)], *[Input(f'core-services-value-m{month}', 'value') for month in range(1, 13)])
    def core_manual_paths_sync(condition_start, condition_end, *cell_values):
        start = min(max(int(condition_start or 1), 1), 12)
        end = min(max(int(condition_end or start), start), 12)
        if len(cell_values) != 24:
            raise PreventUpdate
        neig_values = list(cell_values[:12])
        services_values = list(cell_values[12:])
        styles = [{'display': 'block'} if start <= month <= end else {'display': 'none'} for month in range(1, 13)]

        def _active_text(values):
            active = values[start - 1:end]
            if all((value is None or value == '' for value in active)):
                return ''
            pieces = ['' if value is None or value == '' else f'{float(value):.15g}' for value in active]
            return ', '.join(pieces)
        return (*styles, *styles, _active_text(neig_values), _active_text(services_values))

    @app.callback(*[Output(f'core-neig-value-m{month}', 'value') for month in range(1, 13)], *[Output(f'core-services-value-m{month}', 'value') for month in range(1, 13)], Input('core-neig-fill-all', 'n_clicks'), Input('core-neig-fill-linear', 'n_clicks'), Input('core-services-fill-all', 'n_clicks'), Input('core-services-fill-linear', 'n_clicks'), State('core-scenario-start', 'value'), State('core-scenario-horizon', 'value'), State('core-neig-fill-all-value', 'value'), State('core-neig-linear-start', 'value'), State('core-neig-linear-end', 'value'), State('core-services-fill-all-value', 'value'), State('core-services-linear-start', 'value'), State('core-services-linear-end', 'value'), *[State(f'core-neig-value-m{month}', 'value') for month in range(1, 13)], *[State(f'core-services-value-m{month}', 'value') for month in range(1, 13)], prevent_initial_call=True)
    def core_manual_paths_fill(_neig_fill_clicks, _neig_linear_clicks, _services_fill_clicks, _services_linear_clicks, condition_start, condition_end, neig_fill_value, neig_linear_start, neig_linear_end, services_fill_value, services_linear_start, services_linear_end, *current_values):
        from dash import ctx
        start = min(max(int(condition_start or 1), 1), 12)
        end = min(max(int(condition_end or start), start), 12)
        if len(current_values) != 24:
            raise PreventUpdate
        neig = list(current_values[:12])
        services = list(current_values[12:])

        def _fill_all(values, value):
            if value is None:
                raise PreventUpdate
            for index in range(start - 1, end):
                values[index] = float(value)

        def _fill_linear(values, first_value, last_value):
            if first_value is None or last_value is None:
                raise PreventUpdate
            first = float(first_value)
            last = float(last_value)
            count = end - start + 1
            if count == 1:
                path = [first]
            else:
                path = [first + (last - first) * step / (count - 1) for step in range(count)]
            values[start - 1:end] = path
        if ctx.triggered_id == 'core-neig-fill-all':
            _fill_all(neig, neig_fill_value)
        elif ctx.triggered_id == 'core-neig-fill-linear':
            _fill_linear(neig, neig_linear_start, neig_linear_end)
        elif ctx.triggered_id == 'core-services-fill-all':
            _fill_all(services, services_fill_value)
        elif ctx.triggered_id == 'core-services-fill-linear':
            _fill_linear(services, services_linear_start, services_linear_end)
        else:
            raise PreventUpdate
        return tuple(neig + services)

    def _core_compute_recipe_payload(run_dir, posterior, recipes, *, application_kind):
        recipes = [dict(recipe) for recipe in recipes]
        if not recipes:
            raise HeadlineConditionalError('No Core condition recipe supplied.')
        windows = {(int(recipe.get('condition_start') or 1), int(recipe.get('condition_end') or 1)) for recipe in recipes}
        if len(windows) != 1:
            readable = ', '.join((f'{('NEIG' if recipe.get('variable') == 'hicp_neig' else 'Services')}: M+{int(recipe.get('condition_start') or 1)}..M+{int(recipe.get('condition_end') or 1)}' for recipe in recipes))
            raise HeadlineConditionalError(f'Joint Core conditions must share one conditioning window. Active windows: {readable}.')
        condition_start, condition_end = next(iter(windows))
        H = condition_end - condition_start + 1
        conditions = {}
        assumption_inputs = {}
        for recipe in recipes:
            variable = str(recipe.get('variable') or '')
            if variable not in CORE_CONDITION_VARIABLES:
                raise HeadlineConditionalError(f'Unknown Core component {variable!r}.')
            input_mode = str(recipe.get('input_mode') or 'yoy')
            if input_mode not in {'yoy', 'delta_pp'}:
                raise HeadlineConditionalError(f'{variable}: unknown conditioning mode {input_mode!r}.')
            raw_values = np.asarray(recipe.get('values') or [], dtype=float)
            if raw_values.ndim != 1 or len(raw_values) != H or (not np.isfinite(raw_values).all()):
                raise HeadlineConditionalError(f'{variable}: expected exactly {H} finite stored values.')
            component_history = posterior.inputs.native_levels[variable].astype(float).copy()
            component_history.index = pd.DatetimeIndex(pd.to_datetime(component_history.index)).to_period('M').to_timestamp(how='start')
            component_history = component_history.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
            observed_yoy = 100.0 * (component_history / component_history.shift(12) - 1.0)
            observed_yoy = observed_yoy.replace([np.inf, -np.inf], np.nan).dropna()
            if input_mode == 'yoy':
                target_yoy = raw_values.copy()
                reference_dates = None
                reference_q50 = None
            else:
                reference_dates, reference_q50 = _core_persisted_component_yoy_q50(posterior, variable=variable, condition_start=condition_start, condition_end=condition_end)
                target_yoy = np.asarray(reference_q50, dtype=float) + raw_values
            text_values = ', '.join((f'{float(value):.15g}' for value in target_yoy))
            condition_dates, levels = manual_condition_levels(posterior, variable=variable, metric='yoy', values=text_values, H=H, condition_start=condition_start)
            if reference_dates is not None and (not pd.DatetimeIndex(condition_dates).equals(pd.DatetimeIndex(reference_dates))):
                raise HeadlineConditionalError(f'{variable}: delta-vs-unconditional calendar mismatch.')
            conditions[variable] = levels
            assumption_inputs[variable] = {'variable': variable, 'metric': 'yoy', 'input_mode': input_mode, 'input_mode_label': 'YoY target (%)' if input_mode == 'yoy' else 'Δ vs unconditional (pp)', 'input_values': [float(value) for value in raw_values], 'imposed_values': [float(value) for value in target_yoy], 'unconditional_reference_q50': None if reference_q50 is None else [float(value) for value in np.asarray(reference_q50, dtype=float)], 'observed_dates': [pd.Timestamp(date).isoformat() for date in observed_yoy.index], 'observed_values': [float(value) for value in observed_yoy.to_numpy(dtype=float)], 'condition_dates': [pd.Timestamp(date).isoformat() for date in condition_dates], 'condition_start_offset': condition_start, 'condition_end_offset': condition_end}
        joint = len(conditions) > 1
        lineage = {'source_type': 'core_manual_joint' if joint else 'core_manual', 'condition_metric': 'yoy', 'condition_input_modes': {variable: details['input_mode'] for variable, details in assumption_inputs.items()}, 'core_condition_variables': sorted(conditions), 'core_derived_from': list(CORE_CONDITION_VARIABLES), 'core_aggregation': 'draw_wise_neig_services', 'core_condition_input_values': {variable: details['input_values'] for variable, details in assumption_inputs.items()}, 'core_condition_target_yoy_values': {variable: details['imposed_values'] for variable, details in assumption_inputs.items()}, 'core_unconditional_reference_q50': {variable: details['unconditional_reference_q50'] for variable, details in assumption_inputs.items()}, 'condition_application_mode': application_kind, 'condition_start_offset': condition_start, 'condition_end_offset': condition_end}
        headline_payload = run_saved_headline_conditional(run_dir, native_level_conditions=conditions, H=H, lineage=lineage, project_root=project_root, persist=True, condition_start=condition_start)
        payload = dict(_core_scenario_payload(run_dir, headline_payload))
        assumptions = {}
        for variable, details in assumption_inputs.items():
            fan_block = ((headline_payload.get('fans') or {}).get(variable) or {}).get('yoy') or {}
            assumptions[variable] = {**details, 'baseline_fan': dict(fan_block.get('baseline') or {})}
        payload['conditioning_assumptions'] = assumptions
        payload['dashboard_condition_set'] = {'contract': 'core-computed-scenario-set-v1', 'application_mode': application_kind, 'condition_variables': sorted(conditions)}
        return payload

    def _core_recompute_joint(run_dir, posterior, scenarios):
        active = [name for name in CORE_CONDITION_VARIABLES if name in scenarios]
        if len(active) < 2:
            return None
        recipes = [dict(scenarios[name]['recipe']) for name in active]
        payload = _core_compute_recipe_payload(run_dir, posterior, recipes, application_kind='joint')
        return {'variables': active, 'payload': payload}

    @app.callback(Output('core-condition-set-store', 'data'), Output('core-condition-selected-store', 'data'), Output('core-scenario-status', 'children'), Input('core-condition-add', 'n_clicks'), Input('core-condition-reset', 'n_clicks'), Input('core-card-remove-hicp-neig', 'n_clicks'), Input('core-card-remove-hicp-services', 'n_clicks'), State('core-condition-set-store', 'data'), State('core-condition-selected-store', 'data'), State(display_store_id, 'data'), State(production_vintage_store_id, 'data'), State('core-scenario-start', 'value'), State('core-scenario-horizon', 'value'), State('core-scenario-metric', 'value'), State('core-scenario-neig-values', 'value'), State('core-scenario-services-values', 'value'), background=True, running=[(Output('core-condition-add', 'disabled'), True, False), (Output('core-condition-reset', 'disabled'), True, False)], progress=[Output('core-scenario-progress', 'value'), Output('core-scenario-progress-phase', 'children'), Output('core-scenario-progress-detail', 'children')], progress_default=(0, 'Idle', 'Add / update a conditional to compute it.'), prevent_initial_call=True)
    def mutate_core_computed_scenario_set(set_progress, _add_clicks, _reset_clicks, _remove_neig, _remove_services, store, selected, display_store, production_store, condition_start, condition_end, metric, neig_text, services_text):
        from dash import ctx
        current = dict(store or {})
        scenarios = dict(current.get('scenarios') or {})
        trigger = ctx.triggered_id
        if trigger == 'core-condition-reset':
            set_progress((0, 'Idle', 'All Core conditionals reset.'))
            return ({'contract': 'core-computed-scenario-set-v1', 'scenarios': {}, 'joint': None}, None, 'All Core conditionals reset.')
        remove_map = {'core-card-remove-hicp-neig': 'hicp_neig', 'core-card-remove-hicp-services': 'hicp_services'}
        if trigger in remove_map:
            key = remove_map[trigger]
            if key not in scenarios:
                raise PreventUpdate
            scenarios.pop(key, None)
            remaining = [name for name in CORE_CONDITION_VARIABLES if name in scenarios]
            new_selected = str(selected or '')
            if new_selected == key or new_selected == '__joint__':
                new_selected = remaining[0] if remaining else None
            joint = None
            if len(remaining) >= 2:
                run_dir = _run_directory(results_root, display_store, production_store, headline_run_resolver)
                posterior = load_saved_headline_posterior(run_dir, project_root=project_root)
                joint = _core_recompute_joint(run_dir, posterior, scenarios)
            set_progress((100, 'Scenario set updated', f'{len(remaining)} active Core scenario(s).'))
            return ({'contract': 'core-computed-scenario-set-v1', 'scenarios': scenarios, 'joint': joint}, new_selected, 'Core conditional removed.')
        if trigger != 'core-condition-add':
            raise PreventUpdate
        start = int(condition_start or 1)
        end = int(condition_end or start)
        if start < 1 or end < start or end > 12:
            raise HeadlineConditionalError(f'Condition window must satisfy 1 <= start <= end <= 12; received M+{start}..M+{end}.')
        H = end - start + 1
        input_mode = str(metric or 'yoy')
        if input_mode not in {'yoy', 'delta_pp'}:
            raise HeadlineConditionalError(f'Unknown Core conditioning mode {input_mode!r}.')
        raw_inputs = {'hicp_neig': str(neig_text or '').strip(), 'hicp_services': str(services_text or '').strip()}
        recipes = []
        for variable, text in raw_inputs.items():
            if not text:
                continue
            pieces = [piece.strip() for piece in text.split(',')]
            if len(pieces) != H or any((piece == '' for piece in pieces)):
                raise HeadlineConditionalError(f'{variable}: expected exactly {H} values for M+{start}..M+{end}.')
            values = np.asarray([float(piece) for piece in pieces], dtype=float)
            if not np.isfinite(values).all():
                raise HeadlineConditionalError(f'{variable}: active values must be finite.')
            recipes.append({'variable': variable, 'label': 'NEIG' if variable == 'hicp_neig' else 'Services', 'condition_start': start, 'condition_end': end, 'input_mode': input_mode, 'values': [float(value) for value in values]})
        if not recipes:
            raise HeadlineConditionalError('Enter a NEIG path, a Services path, or both before Add / update.')
        set_progress((20, 'Loading saved posterior', 'Resolving the durable production Headline run.'))
        run_dir = _run_directory(results_root, display_store, production_store, headline_run_resolver)
        posterior = load_saved_headline_posterior(run_dir, project_root=project_root)
        for index, recipe in enumerate(recipes, 1):
            variable = recipe['variable']
            label = recipe['label']
            set_progress((30 + int(30 * index / max(len(recipes), 1)), 'Computing marginal effect', f'Running {label} alone.'))
            marginal = _core_compute_recipe_payload(run_dir, posterior, [recipe], application_kind='marginal')
            scenarios[variable] = {'recipe': recipe, 'payload': marginal}
        set_progress((78, 'Updating joint effect', 'Recomputing NEIG + Services simultaneously when both are active.'))
        joint = _core_recompute_joint(run_dir, posterior, scenarios)
        selected_next = recipes[-1]['variable']
        set_progress((100, 'Conditional scenario active', f'{len(scenarios)} calculated Core scenario(s) active.'))
        return ({'contract': 'core-computed-scenario-set-v1', 'scenarios': scenarios, 'joint': joint}, selected_next, html.Div([html.Strong('Conditional added / updated'), html.Span(' · ' + ' + '.join((recipe['label'] for recipe in recipes)) + ' marginal calculated'), html.Span(' · joint recalculated' if joint is not None else ' · joint available when NEIG and Services are both active'), html.Span(' · BVAR re-estimation: NO')]))

    @app.callback(Output('core-card-select-hicp-neig', 'children'), Output('core-card-wrap-hicp-neig', 'style'), Output('core-card-select-hicp-services', 'children'), Output('core-card-wrap-hicp-services', 'style'), Output('core-card-select-joint', 'children'), Output('core-card-wrap-joint', 'style'), Input('core-condition-set-store', 'data'), Input('core-condition-selected-store', 'data'))
    def render_core_computed_scenario_cards(store, selected):
        current = dict(store or {})
        scenarios = dict(current.get('scenarios') or {})
        selected = str(selected or '')
        visible_base = {'display': 'grid', 'gridTemplateColumns': '1fr auto', 'gap': '6px', 'alignItems': 'center', 'marginTop': '7px', 'borderRadius': '10px', 'padding': '4px'}
        hidden = {'display': 'none'}
        out = []
        for variable in ('hicp_neig', 'hicp_services'):
            row = scenarios.get(variable)
            if not row:
                out.extend(['', hidden])
                continue
            recipe = dict(row.get('recipe') or {})
            label = 'NEIG' if variable == 'hicp_neig' else 'Services'
            mode = 'YoY target' if recipe.get('input_mode') == 'yoy' else 'Δ vs unconditional'
            children = [html.Div([html.Strong(label.upper()), html.Span('Selected', style={'display': 'inline-block' if selected == variable else 'none', 'marginLeft': '8px', 'fontSize': '10px', 'fontWeight': '700', 'textTransform': 'uppercase', 'color': '#2563eb'})]), html.Div(f'M+{recipe.get('condition_start')}..M+{recipe.get('condition_end')} · {mode} · marginal Core effect calculated', style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '3px'})]
            style = dict(visible_base)
            style.update({'background': '#eff6ff' if selected == variable else '#ffffff', 'border': '1px solid #2563eb' if selected == variable else '1px solid #e2e8f0'})
            out.extend([children, style])
        joint = current.get('joint')
        if joint and joint.get('payload'):
            joint_children = [html.Div([html.Strong('JOINT EFFECT'), html.Span('Selected', style={'display': 'inline-block' if selected == '__joint__' else 'none', 'marginLeft': '8px', 'fontSize': '10px', 'fontWeight': '700', 'textTransform': 'uppercase', 'color': '#2563eb'})]), html.Div('NEIG + Services · simultaneous conditional calculation', style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '3px'})]
            joint_style = dict(visible_base)
            joint_style.update({'gridTemplateColumns': '1fr', 'background': '#eff6ff' if selected == '__joint__' else '#ffffff', 'border': '1px solid #2563eb' if selected == '__joint__' else '1px solid #e2e8f0'})
        else:
            joint_children = ''
            joint_style = hidden
        return (*out, joint_children, joint_style)

    @app.callback(Output('core-condition-selected-store', 'data', allow_duplicate=True), Input('core-card-select-hicp-neig', 'n_clicks'), Input('core-card-select-hicp-services', 'n_clicks'), Input('core-card-select-joint', 'n_clicks'), prevent_initial_call=True)
    def select_core_computed_scenario(_neig, _services, _joint):
        from dash import ctx
        mapping = {'core-card-select-hicp-neig': 'hicp_neig', 'core-card-select-hicp-services': 'hicp_services', 'core-card-select-joint': '__joint__'}
        selected = mapping.get(ctx.triggered_id)
        if not selected:
            raise PreventUpdate
        return selected

    @app.callback(Output('core-scenario-store', 'data'), Input('core-condition-set-store', 'data'), Input('core-condition-selected-store', 'data'))
    def project_core_computed_scenario(store, selected):
        current = dict(store or {})
        scenarios = dict(current.get('scenarios') or {})
        selected = str(selected or '')
        if selected == '__joint__':
            joint = dict(current.get('joint') or {})
            payload = joint.get('payload')
            if payload:
                return payload
        if selected in scenarios:
            return (scenarios[selected] or {}).get('payload')
        for variable in CORE_CONDITION_VARIABLES:
            if variable in scenarios:
                return (scenarios[variable] or {}).get('payload')
        return None

    @app.callback(Output('core-neig-conditioning-assumption', 'figure'), Output('core-services-conditioning-assumption', 'figure'), Output('core-scenario-main', 'figure'), Output('core-scenario-impact', 'figure'), Output('core-scenario-summary-table', 'data'), Output('core-results-conditioning', 'style'), Output('core-results-summary', 'style'), Output('core-results-main', 'style'), Output('core-results-impact', 'style'), Input('core-scenario-store', 'data'), Input(display_store_id, 'data'), Input('core-scenario-display-horizon', 'value'), Input('url', 'pathname'))
    def scenario_figures(payload, display_store, display_horizon, pathname):
        if (pathname or "") not in {"/scenarios/core", "/core/scenarios"}:
            raise PreventUpdate
        h = min(max(int(display_horizon or 6), 1), 12)
        table = paired_blocks_records((payload or {}).get('dates') or [], ((payload or {}).get('baseline') or {}).get('yoy') or {}, ((payload or {}).get('conditional') or {}).get('yoy') or {}, ((payload or {}).get('impact') or {}).get('yoy') or {}, max_months=h)
        result_style = {} if bool(payload) else {'display': 'none'}
        return (core_conditioning_assumption_figure(payload, 'hicp_neig', h), core_conditioning_assumption_figure(payload, 'hicp_services', h), core_scenario_main_figure(payload, display_store, h), core_scenario_impact_figure(payload, h), table, result_style, result_style, result_style, result_style)


__all__ = [
    "core_forecast_page",
    "core_scenarios_page",
    "core_scenario_main_figure",
    "core_scenario_impact_figure",
    "register_core_dashboard_callbacks",
]
