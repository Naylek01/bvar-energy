"""Headline Dashboard — Slice 6.

Conditional forecasts and Energy → Headline scenario bridge.

Design contract
---------------
This page intentionally reuses the Energy dashboard visual grammar:
``page-heading-row`` / ``chart-controls`` / ``stats-grid`` / ``panel-heading`` /
``chart-panel`` / ``dcc.Loading`` plus the exact shared Plotly theme and fan
helpers from ``energy_bvar_theme``.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from energy_bvar_theme import (
    TOKENS,
    INFLATION_COLORS,
    apply_inflation_figure_style,
    fan_traces,
    graph_config,
)
from inflation_chart_contract import short_horizon_labels, apply_short_horizon_axis
from energy_bvar_dashboard_conditional import (
    compute_conditional_scenario,
)
from energy_bvar_dashboard_scenarios import (
    scenario_set_summary,
    scenario_set_to_tax_scenarios,
)
from inflation_table_contract import (
    PAIRED_EFFECT_COLUMNS,
    paired_blocks_records,
    readable_table,
)
from headline_bvar_conditional import (
    CONDITION_STATISTICS,
    CONDITIONAL_CONTRACT_VERSION,
    COMPUTATIONAL_HORIZON,
    ENERGY_BRIDGE_CONTRACT_VERSION,
    HeadlineConditionalError,
    load_saved_headline_posterior,
    manual_condition_levels,
    run_saved_headline_conditional,
    run_saved_headline_energy_marginal,
)

# HEADLINE_CONSUMES_PREBUILT_JOINT_ENERGY_V1
# HEADLINE_JOINT_ENERGY_PACKAGE_V1
# HEADLINE_ENERGY_LEGACY_BRIDGE_AUTO_REFRESH_V4
HORIZONS = (3, 6, 12)
COMPONENTS = ("hicp_energy", "hicp_food", "hicp_neig", "hicp_services")
SERIES_LABELS = {
    "hicp_total": "Headline HICP",
    "hicp_energy": "Energy",
    "hicp_food": "Food",
    "hicp_neig": "NEIG",
    "hicp_services": "Services",
}


# GRAPH_EXPORT_READABILITY_G5_HEADLINE_SCENARIOS_V1
def _layout(fig: go.Figure, revision: str, y_title: str | None, height: int = 440) -> go.Figure:
    apply_inflation_figure_style(
        fig,
        uirevision=revision,
        height=height,
        y_title=y_title,
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _empty(message: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=message, showarrow=False,
        font=dict(size=13, color=TOKENS["muted"]),
    )
    _layout(fig, "h6-empty", None, height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _q(block: dict | None, key: str = "mean") -> np.ndarray:
    if not block:
        return np.asarray([], dtype=float)
    return np.asarray(block.get(key, []), dtype=float)


def _fan(
    fig: go.Figure,
    dates,
    block: dict,
    *,
    name: str,
    color: str,
    bands=("68",),
    anchor_value: float | None = None,
) -> str | None:
    """Plot one posterior fan on deterministic +1m/+2m/... categories."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    n = len(idx)
    if n < 1:
        return f"{name}: no display dates"
    labels = short_horizon_labels(idx)
    quantiles = {
        key: np.asarray(block.get(key, []), dtype=float)[:n]
        for key in ("q05", "q16", "q50", "q84", "q95")
        if key in block
    }
    mean = _q(block, "mean")[:n]
    q50 = quantiles.get("q50", np.asarray([], dtype=float))
    if not quantiles:
        return f"{name}: no posterior quantiles"
    if len(q50) != n or len(mean) != n:
        return f"{name}: posterior summary length does not match display horizon"
    if not np.isfinite(mean).all():
        return f"{name}: posterior mean contains non-finite values"
    required = {"q16", "q50", "q84"}
    if "90" in bands:
        required |= {"q05", "q95"}
    missing = sorted(required.difference(quantiles))
    if missing:
        return f"{name}: missing quantiles {missing}"
    for key in required:
        values = np.asarray(quantiles[key], dtype=float)
        if len(values) != n or not np.isfinite(values).all():
            return f"{name}: invalid {key} values"

    for trace in fan_traces(
        labels,
        quantiles,
        bands=bands,
        color=color,
        name=name,
        show_median=False,
    ):
        fig.add_trace(trace)

    line_x = list(labels)
    line_y = mean.tolist()
    custom = [pd.Timestamp(x).strftime("%Y-%m") for x in idx]
    if anchor_value is not None and np.isfinite(float(anchor_value)):
        line_x = ["Actual"] + line_x
        line_y = [float(anchor_value)] + line_y
        custom = ["Latest observed"] + custom
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            customdata=custom,
            mode="lines+markers",
            name=f"{name} · posterior mean",
            line=dict(color=color, width=2.2),
            marker=dict(size=5),
            hovertemplate="%{x} · %{customdata}<br>%{y:.2f}%<extra>" + name + "</extra>",
        )
    )
    return None

def _scenario_dates(payload: dict | None, horizon: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime((payload or {}).get("dates") or []))[: int(horizon)]


def _calendar_hover(dates: pd.DatetimeIndex) -> list[str]:
    return [pd.Timestamp(x).strftime("%Y-%m") for x in dates]


def _impact_semantics(payload: dict | None) -> dict[str, str]:
    """Render A/B semantics from structured lineage, with legacy fallback only."""
    meta = dict((payload or {}).get("meta") or {})
    definition = str(meta.get("impact_definition") or "")
    baseline_label = str(meta.get("impact_baseline_label") or "")
    scenario_label = str(meta.get("impact_scenario_label") or "")
    interpretation = str(meta.get("impact_interpretation") or "")

    if definition and baseline_label and scenario_label:
        return {
            "definition": definition,
            "reference": f"A · {baseline_label}",
            "scenario": f"B · {scenario_label}",
            "impact": "Impact · B − A",
            "denominator": baseline_label,
            "draws": f"A · {baseline_label} · B · {scenario_label}",
            "interpretation": interpretation,
        }

    # Legacy persisted scenarios predate the structured semantic lineage.
    reference = str(meta.get("impact_reference") or "")
    source_type = str(meta.get("source_type") or "")
    if reference == "energy_baseline_conditioned":
        baseline_label = "Energy baseline-conditioned · legacy metadata"
        scenario_label = "Energy scenario-conditioned · legacy metadata"
        definition = "legacy_energy_scenario_minus_energy_baseline"
    elif source_type in {"energy_scenario", "joint_energy_scenario"}:
        baseline_label = "Headline unconditional · legacy definition"
        scenario_label = "Energy-conditioned · legacy definition"
        definition = "legacy_energy_minus_unconditional"
    elif source_type == "manual" or reference == "unconditional":
        baseline_label = "Headline unconditional · legacy metadata"
        scenario_label = "Manual conditional · legacy metadata"
        definition = "legacy_manual_minus_unconditional"
    else:
        baseline_label = "Headline unconditional · legacy metadata"
        scenario_label = "Conditional · legacy metadata"
        definition = "legacy_conditional_minus_unconditional"
    return {
        "definition": definition,
        "reference": f"A · {baseline_label}",
        "scenario": f"B · {scenario_label}",
        "impact": "Impact · B − A",
        "denominator": baseline_label,
        "draws": f"A · {baseline_label} · B · {scenario_label}",
        "interpretation": interpretation,
    }


def _horizon_semantics(payload: dict | None) -> str:
    meta = dict((payload or {}).get("meta") or {})
    conditioned = meta.get(
        "conditioned_horizon_months",
        meta.get("condition_horizon"),
    )
    computational = meta.get("computational_horizon")
    propagated = meta.get("free_propagation_horizon_months")
    try:
        conditioned_i = int(conditioned)
        computational_i = int(computational)
        propagated_i = (
            int(propagated)
            if propagated is not None
            else max(0, computational_i - conditioned_i)
        )
    except (TypeError, ValueError):
        return ""
    return (
        f"conditioned {conditioned_i}m · freely propagated "
        f"{propagated_i}m inside {computational_i}m DK horizon"
    )


def _energy_source_contract(payload: dict | None) -> tuple[str, str]:
    """Resolve the saved source aggregate and forecast contract without fallback IDs."""
    item = dict(payload or {})
    meta = dict(item.get("meta") or {})
    bridge = dict(item.get("headline_bridge") or {})
    source_aggregate_run_id = str(meta.get("aggregate_run_id") or "")
    forecast_name = str(bridge.get("forecast_name") or "")
    if not source_aggregate_run_id:
        raise HeadlineConditionalError(
            "Energy scenario payload is missing meta.aggregate_run_id "
            "(saved source aggregate)."
        )
    if not forecast_name:
        raise HeadlineConditionalError(
            "Energy scenario bridge is missing forecast_name."
        )
    return source_aggregate_run_id, forecast_name


def _symmetric_zero_axis(fig: go.Figure, *arrays, minimum: float = 0.01) -> None:
    finite = []
    for array in arrays:
        values = np.asarray(array, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if len(values):
            finite.append(values)
    maximum = max((float(np.max(np.abs(x))) for x in finite), default=0.0)
    half = max(float(minimum), 1.15 * maximum)
    fig.update_yaxes(range=[-half, half], zeroline=False)


def _observed_anchor(payload: dict | None) -> tuple[str | None, float | None]:
    meta = dict((payload or {}).get("meta") or {})
    raw_value = meta.get("headline_observed_anchor_yoy")
    raw_date = meta.get("headline_observed_anchor_date")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite(value):
        return None, None
    return (None if raw_date is None else str(raw_date)), value


def headline_scenario_main_figure(payload: dict | None, horizon: int = 3) -> go.Figure:
    if not payload:
        return _empty("Run a Headline conditional scenario.", 500)
    dates = _scenario_dates(payload, horizon)
    block = (((payload.get("fans") or {}).get("hicp_total") or {}).get("yoy") or {})
    base = block.get("baseline") or {}
    cond = block.get("conditional") or {}
    if not len(dates):
        return _empty("Conditional scenario has no future dates.", 500)

    anchor_date, anchor_value = _observed_anchor(payload)
    fig = go.Figure()
    if anchor_value is not None:
        fig.add_trace(go.Scatter(
            x=["Actual"], y=[anchor_value], mode="markers",
            name="Latest observed", marker=dict(size=8, color=INFLATION_COLORS["observed"]),
            customdata=[anchor_date or "Latest observed"],
            hovertemplate="Actual · %{customdata}<br>%{y:.2f}%<extra>Latest observed</extra>",
        ))
    semantics = _impact_semantics(payload)
    errors = [
        _fan(
            fig,
            dates,
            base,
            name=semantics["reference"],
            color=INFLATION_COLORS["baseline"],
            bands=("68",),
            anchor_value=anchor_value,
        ),
        _fan(
            fig,
            dates,
            cond,
            name=semantics["scenario"],
            color=INFLATION_COLORS["conditional"],
            bands=("68",),
            anchor_value=anchor_value,
        ),
    ]
    errors = [error for error in errors if error]
    if errors:
        return _empty("Baseline vs conditional display unavailable: " + " | ".join(errors), 500)
    labels = short_horizon_labels(dates)
    apply_short_horizon_axis(fig, (["Actual"] if anchor_value is not None else []) + labels)
    return _layout(
        fig,
        f"h6-main::{payload.get('meta', {}).get('scenario_id')}::{horizon}",
        "% y/y",
        500,
    )

def headline_scenario_impact_figure(payload: dict | None, horizon: int = 3) -> go.Figure:
    if not payload:
        return _empty("Run a Headline conditional scenario.")
    dates = _scenario_dates(payload, horizon)
    block = payload.get("headline_yoy_impact") or {}
    n = len(dates)
    if n < 1:
        return _empty("Headline impact is unavailable: no display dates.")
    q16 = _q(block, "q16")[:n]
    q50 = _q(block, "q50")[:n]
    mean = _q(block, "mean")[:n]
    q84 = _q(block, "q84")[:n]
    if any(len(x) != n for x in (q16, q50, mean, q84)):
        return _empty("Headline impact is unavailable: posterior summary length mismatch.")
    if not all(np.isfinite(x).all() for x in (q16, q50, mean, q84)):
        return _empty("Headline impact is unavailable: posterior summaries contain non-finite values.")
    labels = short_horizon_labels(dates)
    semantics = _impact_semantics(payload)
    horizon_note = _horizon_semantics(payload)
    fig = go.Figure()
    for trace in fan_traces(
        labels, {"q16": q16, "q50": q50, "q84": q84}, bands=("68",),
        color=INFLATION_COLORS["impact"], name=semantics["impact"], show_median=False,
    ):
        fig.add_trace(trace)
    fig.add_trace(go.Scatter(
        x=labels, y=mean, customdata=_calendar_hover(dates), mode="lines+markers",
        name="Posterior mean impact",
        line=dict(color=INFLATION_COLORS["impact"], width=2),
        marker=dict(size=5),
        hovertemplate="%{x} · %{customdata}<br>%{y:+.3f} pp<extra>Posterior mean impact</extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    apply_short_horizon_axis(fig, labels)
    _symmetric_zero_axis(fig, q16, q84, mean)
    _layout(
        fig,
        f"h6-impact::{payload.get('meta', {}).get('scenario_id')}::{horizon}",
        "percentage points",
        420,
    )
    note = (
        f"{semantics['impact']} · {semantics['reference']} · {semantics['scenario']}"
    )
    if horizon_note:
        note += " · " + horizon_note
    if semantics.get("interpretation"):
        note += " · " + semantics["interpretation"]
    if str((payload.get("meta") or {}).get("impact_reference") or "") == "energy_baseline_conditioned":
        note += "; pre-scenario dispersion can arise from joint DK conditioning"
    fig.add_annotation(
        x=0,
        y=1.12,
        xref="paper",
        yref="paper",
        text=note,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        align="left",
        font=dict(size=10, color=TOKENS["muted"]),
    )
    return fig

def component_transmission_figure(payload: dict | None, horizon: int = 3) -> go.Figure:
    if not payload:
        return _empty("Run a Headline conditional scenario.")
    dates = _scenario_dates(payload, horizon)
    labels = short_horizon_labels(dates) if len(dates) else []
    blocks = payload.get("component_yoy_impact") or {}
    fig = go.Figure()
    plotted = []
    for name in COMPONENTS:
        values = _q(blocks.get(name) or {})[: len(dates)]
        if len(values) != len(dates) or not np.isfinite(values).all():
            continue
        plotted.append(values)
        fig.add_trace(go.Scatter(
            x=labels, y=values, customdata=_calendar_hover(dates), mode="lines+markers",
            name=SERIES_LABELS[name], line=dict(color=INFLATION_COLORS[name], width=2), marker=dict(size=5),
            hovertemplate="%{x} · %{customdata}<br>%{y:+.3f} pp<extra>" + SERIES_LABELS[name] + "</extra>",
        ))
    if not plotted:
        return _empty("Component transmission is unavailable.")
    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    apply_short_horizon_axis(fig, labels)
    _symmetric_zero_axis(fig, *plotted)
    semantics = _impact_semantics(payload)
    return _layout(
        fig,
        f"h6-components::{payload.get('meta', {}).get('scenario_id')}::{horizon}",
        f"pp vs {semantics['denominator']}",
        430,
    )

def contribution_impact_figure(payload: dict | None, horizon: int = 3) -> go.Figure:
    if not payload:
        return _empty("Run a Headline conditional scenario.")
    dates = _scenario_dates(payload, horizon)
    blocks = payload.get("contribution_yoy_impact") or {}
    labels = short_horizon_labels(dates) if len(dates) else []
    fig = go.Figure()
    plotted = []
    for name in COMPONENTS:
        values = np.asarray((blocks.get(name) or {}).get("mean", []), dtype=float)[: len(dates)]
        if len(values) != len(dates) or not np.isfinite(values).all():
            continue
        plotted.append(values)
        fig.add_trace(go.Bar(
            x=labels, y=values, customdata=_calendar_hover(dates), name=SERIES_LABELS[name],
            marker_color=INFLATION_COLORS[name],
            hovertemplate="%{x} · %{customdata}<br>%{y:+.3f} pp<extra>" + SERIES_LABELS[name] + "</extra>",
        ))
    if not plotted:
        return _empty("Contribution impact is unavailable.")
    matrix = np.vstack(plotted)
    positive = np.maximum(matrix, 0.0).sum(axis=0)
    negative = np.minimum(matrix, 0.0).sum(axis=0)
    fig.update_layout(barmode="relative")
    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    apply_short_horizon_axis(fig, labels)
    _symmetric_zero_axis(fig, positive, negative)
    semantics = _impact_semantics(payload)
    return _layout(
        fig,
        f"h6-contrib::{payload.get('meta', {}).get('scenario_id')}::{horizon}",
        f"pp contribution impact vs {semantics['denominator']} · posterior mean",
        430,
    )

def overlay_headline_conditional_forecast(
    fig: go.Figure,
    payload: dict | None,
    *,
    series: str,
    metric: str,
    horizon: int,
) -> go.Figure:
    """Overlay conditional posterior mean on Slice-5 Forecast without replacing baseline."""
    if not payload or str(payload.get("contract")) != CONDITIONAL_CONTRACT_VERSION:
        return fig
    dates = pd.DatetimeIndex(pd.to_datetime(payload.get("dates") or []))[: int(horizon)]
    block = (((payload.get("fans") or {}).get(series) or {}).get(metric) or {}).get("conditional") or {}
    q50 = _q(block)[: len(dates)]
    if len(dates) and len(q50) == len(dates):
        fig.add_trace(go.Scatter(
            x=dates,
            y=q50,
            mode="lines+markers",
            name="Conditional · posterior mean",
            line=dict(color=INFLATION_COLORS["conditional"], width=2.3, dash="dash"),
            marker=dict(size=5),
        ))
        fig.update_layout(
            uirevision=f"{fig.layout.uirevision}::conditional::{payload.get('meta', {}).get('scenario_id')}"
        )
    return fig


def _stat_card(title: str, value_id: str, note_id: str | None = None) -> html.Div:
    children = [
        html.Div(title, className="stat-title"),
        html.Div("—", id=value_id, className="stat-value"),
    ]
    if note_id:
        children.append(html.Div("", id=note_id, className="stat-subtitle"))
    return html.Div(children, className="stat-card")


def _panel_heading(eyebrow: str, title: str, subtitle: str | None = None) -> html.Div:
    children = [html.Div(eyebrow, className="eyebrow"), html.H3(title, className="panel-title")]
    if subtitle:
        children.append(html.P(subtitle, className="panel-subtitle"))
    return html.Div(children, className="panel-heading")


def headline_scenarios_page() -> html.Div:
    controls = html.Div(
        [
            html.Div(
                [
                    html.Label("Source", className="selector-label"),
                    dcc.Dropdown(
                        id="h6-source",
                        options=[
                            {"label": "Energy scenario → Headline", "value": "energy"},
                            {"label": "Manual conditional path", "value": "manual"},
                        ],
                        value="energy",
                        clearable=False,
                    ),
                ],
                className="selector-block",
            ),
            html.Div(
                [
                    html.Label("Condition horizon", className="selector-label"),
                    dcc.Dropdown(
                        id="h6-condition-horizon",
                        options=[{"label": f"{x} months", "value": x} for x in HORIZONS],
                        value=3,
                        clearable=False,
                    ),
                ],
                className="selector-block",
            ),
            html.Div(
                [
                    html.Label("Display horizon", className="selector-label"),
                    dcc.Dropdown(
                        id="h6-display-horizon",
                        options=[{"label": f"{x} months", "value": x} for x in HORIZONS],
                        value=6,
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
            dcc.Store(id="h6-headline-scenario-store", storage_type="session"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Headline scenarios", className="page-title"),
                            html.P(
                                "Apply a live Energy scenario to the saved Headline BVAR, or enter a manual component path. Conditioning and display horizons are independent.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    controls,
                ],
                className="page-heading-row",
            ),
            html.Div(id="h6-status", className="selection-banner"),
            html.Div(
                [
                    _stat_card("+1m impact", "h6-kpi-1", "h6-kpi-1-date"),
                    _stat_card("+2m impact", "h6-kpi-2", "h6-kpi-2-date"),
                    _stat_card("+3m impact", "h6-kpi-3", "h6-kpi-3-date"),
                    _stat_card("Paired draws", "h6-kpi-draws", "h6-kpi-draws-note"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    _panel_heading(
                        "Key scenario results",
                        "Headline conditional effect by display horizon",
                        "The condition can end before the display horizon; later months then evolve endogenously inside the locked 12-month BVAR forecast.",
                    ),
                    readable_table("h6-summary-table", PAIRED_EFFECT_COLUMNS, page_size=12),
                ],
                className="panel table-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            _panel_heading(
                                "Energy → Headline bridge",
                                "Active Energy scenario",
                                "Energy scenarios are detected automatically. Application to Headline remains an explicit Run action, with a hard same-vintage guard.",
                            ),
                            html.Label("Energy application", className="selector-label"),
                            dcc.RadioItems(
                                id="h6-energy-application-mode",
                                options=[
                                    {
                                        "label": "Selected scenario only",
                                        "value": "selected",
                                    },
                                    {
                                        "label": "All active scenarios jointly",
                                        "value": "joint",
                                    },
                                ],
                                value="selected",
                                inline=True,
                                className="h6-inline-radio",
                            ),
                            html.Div(
                                id="h6-joint-scenario-summary",
                                className="placeholder-text",
                            ),
                            html.Label(
                                "Selected Energy scenario · individual mode",
                                className="selector-label",
                            ),
                            dcc.Dropdown(
                                id="h6-energy-scenario-select",
                                options=[],
                                value=None,
                                placeholder="No compatible active Energy scenario",
                                clearable=False,
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Energy conditioning path", className="selector-label"),
                                            dcc.Dropdown(
                                                id="h6-energy-path-kind",
                                                options=[
                                                    {"label": "Posterior mean", "value": "mean"},
                                                    {"label": "Pointwise percentile", "value": "percentile"},
                                                ],
                                                value="mean",
                                                clearable=False,
                                            ),
                                        ],
                                        className="selector-block",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Percentile (P01–P99)", className="selector-label"),
                                            dcc.Input(
                                                id="h6-energy-percentile",
                                                type="number",
                                                min=1,
                                                max=99,
                                                step=1,
                                                value=50,
                                                debounce=True,
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        className="selector-block",
                                    ),
                                ],
                                className="selectors-grid",
                            ),
                            html.P(
                                "The conditioning statistic is selected after the Energy scenario package is constructed. In joint mode, all active Energy scenarios are first applied simultaneously and HICP Energy is aggregated draw by draw; only then is the posterior mean or requested pointwise percentile P01–P99 transferred to Headline. A percentile path need not correspond to one single Energy draw through time.",
                                className="placeholder-text",
                            ),
                            html.Div(
                                "Checking active Energy scenarios…",
                                id="h6-bridge-readiness",
                                className="selection-banner",
                            ),
                            html.Div(id="h6-bridge-details", className="placeholder-text"),
                            dcc.Link("Open Energy scenarios →", href="/scenarios", className="refresh-button"),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            _panel_heading(
                                "Manual conditioning",
                                "Alternative component path",
                                "Enter exactly the number of values in the Condition horizon. The BVAR is still solved over 12 months; later months remain latent and respond endogenously.",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Variable", className="selector-label"),
                                            dcc.Dropdown(
                                                id="h6-manual-variable",
                                                options=[{"label": SERIES_LABELS[x], "value": x} for x in COMPONENTS],
                                                value="hicp_energy",
                                                clearable=False,
                                            ),
                                        ],
                                        className="selector-block",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Metric", className="selector-label"),
                                            dcc.Dropdown(
                                                id="h6-manual-metric",
                                                options=[
                                                    {"label": "HICP level", "value": "level"},
                                                    {"label": "Year-on-year", "value": "yoy"},
                                                ],
                                                value="yoy",
                                                clearable=False,
                                            ),
                                        ],
                                        className="selector-block",
                                    ),
                                ],
                                className="selectors-grid",
                            ),
                            html.Label("Future values", className="selector-label"),
                            dcc.Textarea(
                                id="h6-manual-values",
                                placeholder="Example for a 3-month condition: 2.5, 2.2, 2.0",
                                className="h6-path-input",
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="two-column-grid",
            ),
            html.Div(
                [
                    html.Button("Run conditional forecast", id="h6-run", n_clicks=0, className="refresh-button"),
                    html.Button("Clear", id="h6-clear", n_clicks=0, className="refresh-button h6-secondary-button"),
                    html.Span(
                        "Energy source runtime on the 20260816 reference run: "
                        "~24 s with cached Energy-baseline A; ~46 s on a cold A+B run.",
                        className="placeholder-text",
                    ),
                ],
                className="h6-actions",
            ),
            html.Div(
                [
                    _panel_heading("Baseline vs conditional", "Headline HICP — year-on-year", "Latest observed Headline is shown as the common anchor; future months use +1m / +2m / … labels."),
                    dcc.Loading(dcc.Graph(id="h6-main", config=graph_config("headline_scenario_main")), type="circle"),
                    html.P(
                        "The Headline conditional fan is conditional on the selected deterministic Energy path. Energy-path uncertainty is not integrated out; a narrower band must not be interpreted as a reduction in economic uncertainty.",
                        className="placeholder-text",
                    ),
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            _panel_heading(
                                "Headline impact",
                                "Scenario effect — exact denominator shown inside chart",
                            ),
                            dcc.Loading(dcc.Graph(id="h6-impact", config=graph_config("headline_scenario_impact")), type="circle"),
                        ],
                        className="panel chart-panel",
                    ),
                    html.Div(
                        [
                            _panel_heading("Transmission", "Component response"),
                            dcc.Loading(dcc.Graph(id="h6-components", config=graph_config("headline_scenario_components")), type="circle"),
                        ],
                        className="panel chart-panel",
                    ),
                ],
                className="two-column-grid",
            ),
            html.Div(
                [
                    _panel_heading(
                        "Exact additive decomposition",
                        "Change in component contributions",
                        "Posterior means are shown. Impact axes are symmetric around zero to preserve sign and scale.",
                    ),
                    dcc.Loading(dcc.Graph(id="h6-contributions", config=graph_config("headline_scenario_contributions")), type="circle"),
                ],
                className="panel chart-panel",
            ),
        ],
        className="page-body headline-page",
    )

def _headline_identity(store: dict | None) -> dict[str, str]:
    meta = dict((store or {}).get("meta") or {})
    context = dict((store or {}).get("context") or {})
    return {
        "model_id": str(meta.get("model_id") or context.get("model_id") or ""),
        "vintage": str(meta.get("vintage") or context.get("vintage") or ""),
        "run_id": str(
            meta.get("run_id")
            or meta.get("headline_run_id")
            or context.get("run_id")
            or ""
        ),
    }



def _energy_statistic(path_kind: str | None, percentile) -> str:
    if str(path_kind or "mean") == "mean":
        return "mean"
    try:
        value = int(percentile)
    except (TypeError, ValueError) as exc:
        raise HeadlineConditionalError("Energy percentile must be an integer from 1 to 99.") from exc
    if value < 1 or value > 99:
        raise HeadlineConditionalError("Energy percentile must lie in P01..P99.")
    return f"p{value:02d}"


def _conditional_payloads(store: dict | None) -> list[tuple[str, dict]]:
    rows = []
    for model_id, item in dict((store or {}).get("items") or {}).items():
        payload = dict((item or {}).get("payload") or {})
        if payload:
            rows.append((str(model_id), payload))
    return rows


def _conditional_refresh_spec(
    payload: dict | None,
    headline_vintage: str | None,
) -> tuple[dict | None, str | None]:
    """Return the saved-scenario recipe needed to rebuild a stale bridge.

    Legacy conditional-store payloads retain the exact aggregate/model/path
    recipe. That is enough to replay the persisted Energy posterior and
    materialise the current Headline bridge without running Gibbs again.
    """
    meta = dict((payload or {}).get("meta") or {})
    vintage = str(meta.get("vintage") or "")
    if not vintage:
        return None, "Energy scenario vintage is missing"
    if headline_vintage and vintage != str(headline_vintage):
        return None, f"vintage {vintage} ≠ Headline {headline_vintage}"

    required = {
        "aggregate_run_id": meta.get("aggregate_run_id"),
        "model_id": meta.get("model_id"),
        "condition_variable": meta.get("condition_variable"),
        "path_mode": meta.get("path_mode"),
        "path_value": meta.get("path_value"),
    }
    missing = [key for key, value in required.items() if value in (None, "")]
    if missing:
        return None, "legacy scenario metadata missing " + ", ".join(missing)
    try:
        path_value = float(required["path_value"])
    except (TypeError, ValueError):
        return None, "legacy scenario path_value is invalid"
    if not np.isfinite(path_value):
        return None, "legacy scenario path_value is non-finite"

    return {
        "vintage": vintage,
        "aggregate_run_id": str(required["aggregate_run_id"]),
        "model_id": str(required["model_id"]),
        "condition_variable": str(required["condition_variable"]),
        "path_mode": str(required["path_mode"]),
        "path_value": path_value,
    }, None


def _candidate_record(
    candidate_id: str,
    label: str,
    payload: dict,
    headline_vintage: str,
    *,
    allow_conditional_refresh: bool = False,
) -> dict:
    bridge = dict((payload or {}).get("headline_bridge") or {})
    bridge_vintage = str(bridge.get("vintage") or "")
    contract_ok = (
        str(bridge.get("contract") or "") == ENERGY_BRIDGE_CONTRACT_VERSION
    )
    active = bool(bridge.get("scenario_active", False))
    same = bool(headline_vintage) and bridge_vintage == str(headline_vintage)

    refresh_spec = None
    refresh_reason = None
    if allow_conditional_refresh and (not contract_ok or not bridge):
        refresh_spec, refresh_reason = _conditional_refresh_spec(
            payload,
            headline_vintage,
        )

    refreshable = refresh_spec is not None
    enabled = (contract_ok and active and same) or refreshable
    reason = None
    refresh_on_apply = False

    if refreshable:
        refresh_on_apply = True
    elif not bridge:
        reason = (
            refresh_reason
            or "Headline bridge cannot be reconstructed from this scenario"
        )
    elif not contract_ok:
        reason = (
            refresh_reason
            or "stale bridge cannot be reconstructed automatically"
        )
    elif not active:
        reason = "scenario is not active"
    elif headline_vintage and not same:
        reason = (
            f"vintage {bridge_vintage or 'missing'} ≠ "
            f"Headline {headline_vintage}"
        )

    if enabled and refresh_on_apply:
        shown = label + " · bridge refreshes automatically on Apply"
    else:
        shown = label + ("" if enabled else f" · BLOCKED: {reason}")

    return {
        "id": candidate_id,
        "label": shown,
        "payload": payload,
        "bridge": bridge,
        "enabled": enabled,
        "reason": reason,
        "refresh_on_apply": refresh_on_apply,
        "refresh_spec": refresh_spec,
    }


def _energy_scenario_candidates(
    conditional_store: dict | None,
    aggregate_store: dict | None,
    headline_store: dict | None,
) -> list[dict]:
    headline_vintage = _headline_identity(headline_store).get("vintage", "")
    out = []

    for model_id, payload in _conditional_payloads(conditional_store):
        meta = dict(payload.get("meta") or {})
        bridge = dict(payload.get("headline_bridge") or {})
        label = str(
            bridge.get("source_label")
            or (
                f"Conditional · {meta.get('model_label') or model_id} · "
                f"{meta.get('condition_description') or meta.get('condition_variable') or ''}"
            )
        ).strip(" ·")
        out.append(
            _candidate_record(
                f"conditional::{model_id}",
                label,
                payload,
                headline_vintage,
                allow_conditional_refresh=True,
            )
        )

    aggregate_bridge = dict(
        (aggregate_store or {}).get("headline_bridge") or {}
    )
    if aggregate_store and (
        aggregate_bridge or bool((aggregate_store or {}).get("meta"))
    ):
        components = list(
            aggregate_bridge.get("scenario_components") or []
        )
        component_text = (
            ", ".join(str(x).replace("_", " ") for x in components)
            or "active tax set"
        )
        label = str(
            aggregate_bridge.get("source_label")
            or f"Tax scenario set · {component_text}"
        )
        out.append(
            _candidate_record(
                "tax::aggregate",
                label,
                dict(aggregate_store or {}),
                headline_vintage,
            )
        )
    return out


def _selected_energy_scenario_record(
    conditional_store: dict | None,
    aggregate_store: dict | None,
    headline_store: dict | None,
    selected_id: str | None,
) -> dict | None:
    for row in _energy_scenario_candidates(
        conditional_store,
        aggregate_store,
        headline_store,
    ):
        if str(row["id"]) == str(selected_id) and row["enabled"]:
            return dict(row)
    return None


def _selected_energy_scenario_payload(
    conditional_store: dict | None,
    aggregate_store: dict | None,
    headline_store: dict | None,
    selected_id: str | None,
) -> dict | None:
    row = _selected_energy_scenario_record(
        conditional_store,
        aggregate_store,
        headline_store,
        selected_id,
    )
    return None if row is None else dict(row["payload"])


def _ensure_current_energy_bridge(
    row: dict,
    *,
    results_root: Path,
    project_root: str | Path | None,
) -> tuple[dict, bool]:
    """Materialise a missing/stale conditional bridge from saved results.

    Returns ``(payload, rebuilt)``. ``compute_conditional_scenario`` reloads
    and replays persisted posterior draws; it does not run the Gibbs sampler.
    """
    payload = dict(row.get("payload") or {})
    bridge = dict(payload.get("headline_bridge") or {})
    if (
        str(bridge.get("contract") or "")
        == ENERGY_BRIDGE_CONTRACT_VERSION
    ):
        return payload, False

    if not bool(row.get("refresh_on_apply")):
        raise HeadlineConditionalError(
            str(
                row.get("reason")
                or "Selected Energy bridge is not usable."
            )
        )

    spec = dict(row.get("refresh_spec") or {})
    if not spec:
        raise HeadlineConditionalError(
            "Selected legacy Energy scenario has no reconstructible "
            "bridge recipe."
        )

    aggregate_dir = (
        Path(results_root)
        / "hicp_energy_aggregate"
        / str(spec["vintage"])
        / str(spec["aggregate_run_id"])
    )
    if not aggregate_dir.is_dir():
        raise HeadlineConditionalError(
            "Cannot rebuild the Headline bridge because the saved Energy "
            f"aggregate is missing: {aggregate_dir}"
        )

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(results_root).resolve().parent
    )
    fresh = compute_conditional_scenario(
        aggregate_dir,
        project_root=root,
        model_id=str(spec["model_id"]),
        condition_variable=str(spec["condition_variable"]),
        path_mode=str(spec["path_mode"]),
        path_value=float(spec["path_value"]),
    )

    new_bridge = dict((fresh or {}).get("headline_bridge") or {})
    if (
        str(new_bridge.get("contract") or "")
        != ENERGY_BRIDGE_CONTRACT_VERSION
    ):
        raise HeadlineConditionalError(
            "Automatic Energy bridge refresh completed without the "
            f"current {ENERGY_BRIDGE_CONTRACT_VERSION} contract."
        )
    return dict(fresh), True


def _bridge_readiness_state(
    headline_store: dict | None,
    energy_payload: dict | None,
    *,
    source: str | None,
    horizon: int | None,
    statistic: str | None,
) -> dict[str, object]:
    """Cheap UI precheck; model-owned hard guards still run on explicit Run."""
    if str(source or "energy") != "energy":
        return {"ready": True, "level": "info", "title": "Manual conditioning mode", "detail": "The Energy → Headline bridge is not required for this run."}
    headline = _headline_identity(headline_store)
    if headline["model_id"] != "headline_joint" or not headline["vintage"] or not headline["run_id"]:
        return {"ready": False, "level": "wait", "title": "Waiting for a Headline run", "detail": "Select a complete Headline HICP run first."}
    if not energy_payload:
        return {"ready": False, "level": "wait", "title": "Select an active Energy scenario", "detail": "Active Energy scenarios are listed above. If a conditional scenario says BLOCKED/stale, update it once on the Energy Scenarios page."}
    bridge = dict((energy_payload or {}).get("headline_bridge") or {})
    if str(bridge.get("contract") or "") != ENERGY_BRIDGE_CONTRACT_VERSION:
        return {"ready": False, "level": "error", "title": "Stale Energy bridge contract", "detail": f"Expected {ENERGY_BRIDGE_CONTRACT_VERSION}. Recompute the selected Energy scenario."}
    if not bool(bridge.get("scenario_active", False)):
        return {"ready": False, "level": "wait", "title": "Energy scenario is inactive", "detail": "Activate/recompute the scenario before applying it to Headline."}
    energy_vintage = str(bridge.get("vintage") or "")
    if energy_vintage != headline["vintage"]:
        return {"ready": False, "level": "error", "title": "Vintage mismatch", "detail": f"Headline={headline['vintage']} · Energy={energy_vintage or 'missing'}. Cross-vintage conditioning is blocked."}
    stat = str(statistic or "mean")
    values = dict(bridge.get("scenario_level") or {}).get(stat)
    dates = list(bridge.get("dates") or [])
    if values is None:
        return {"ready": False, "level": "error", "title": f"Energy path {stat.upper()} is missing", "detail": "Recompute the Energy scenario with the current bridge contract."}
    try:
        path = np.asarray(values, dtype=float)
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
    except Exception as exc:
        return {"ready": False, "level": "error", "title": "Energy bridge path is unreadable", "detail": str(exc)}
    if path.ndim != 1 or len(path) != len(idx) or len(path) < 2 or not np.isfinite(path).all():
        return {"ready": False, "level": "error", "title": "Energy bridge path is incomplete", "detail": f"dates={len(idx)} · values={len(path)}. Aligned finite monthly levels are required."}
    monthly = idx.to_period("M")
    if monthly.has_duplicates or not monthly.is_monotonic_increasing:
        return {"ready": False, "level": "error", "title": "Energy bridge calendar is invalid", "detail": "Recompute the Energy scenario."}
    components = list(bridge.get("scenario_components") or [])
    aggregate_run_id = str(bridge.get("aggregate_run_id") or "")
    n_draws = bridge.get("n_draws")
    source_label = str(bridge.get("source_label") or bridge.get("source_kind") or "Energy scenario")
    component_text = ", ".join(map(str, components)) if components else "aggregate scenario"
    draw_text = "—" if n_draws is None else f"{int(n_draws):,}"
    path_text = "posterior mean" if stat == "mean" else f"pointwise {stat.upper()}"
    H = int(horizon or 3)
    return {
        "ready": True,
        "level": "ready",
        "title": f"Bridge ready · vintage {headline['vintage']}",
        "detail": (
            f"{source_label} · Energy aggregate {aggregate_run_id[:12] or '—'} · {component_text} · "
            f"{draw_text} Energy draws · {path_text} · condition H={H}m. "
            "The percentile is computed directly from Energy draws month by month; same-vintage, calendar coverage, ratio diagnostics and baseline identity are revalidated on Run."
        ),
    }

def _joint_energy_recipe(
    conditional_store: dict | None,
    tax_store: dict | None,
    aggregate_store: dict | None,
    headline_store: dict | None,
) -> dict[str, object]:
    """Resolve all active Energy scenarios into one same-vintage package."""
    headline_vintage = _headline_identity(
        headline_store
    ).get("vintage", "")
    if not headline_vintage:
        raise HeadlineConditionalError(
            "No Headline vintage is loaded."
        )

    conditional_specs = []
    labels = []
    aggregate_ids = set()
    for model_id, payload in _conditional_payloads(
        conditional_store
    ):
        spec, reason = _conditional_refresh_spec(
            payload,
            headline_vintage,
        )
        if spec is None:
            raise HeadlineConditionalError(
                f"Conditional {model_id}: {reason or 'scenario recipe is unavailable'}."
            )
        conditional_specs.append(spec)
        aggregate_ids.add(str(spec["aggregate_run_id"]))
        meta = dict((payload or {}).get("meta") or {})
        labels.append(
            "Conditional · "
            + str(meta.get("model_label") or model_id)
            + " · "
            + str(
                meta.get("condition_description")
                or meta.get("condition_variable")
                or ""
            )
        )

    tax_scenarios = scenario_set_to_tax_scenarios(
        tax_store
    )
    tax_rows = scenario_set_summary(tax_store)
    if tax_scenarios:
        tax_vintage = str(
            (tax_store or {}).get("vintage") or ""
        )
        if (
            tax_vintage
            and tax_vintage != str(headline_vintage)
        ):
            raise HeadlineConditionalError(
                f"Tax scenario vintage {tax_vintage} ≠ Headline {headline_vintage}."
            )
        aggregate_meta = dict(
            (aggregate_store or {}).get("meta") or {}
        )
        tax_aggregate_id = str(
            aggregate_meta.get("aggregate_run_id") or ""
        )
        if tax_aggregate_id:
            aggregate_ids.add(tax_aggregate_id)
        elif not aggregate_ids:
            raise HeadlineConditionalError(
                "Active tax scenarios require the live HICP Energy aggregate "
                "for the same vintage."
            )
        for row in tax_rows:
            vat = float(row.get("vat_delta_pp", 0.0) or 0.0)
            exc = float(row.get("excise_delta", 0.0) or 0.0)
            if abs(vat) < 1e-15 and abs(exc) < 1e-15:
                continue
            labels.append(
                "Tax · "
                + str(row.get("label") or row.get("model_id"))
                + f" · VAT {vat:+.2f} pp"
                + (
                    ""
                    if abs(exc) < 1e-15
                    else (
                        f", excise {exc:+.3f} "
                        + str(row.get("excise_unit") or "")
                    )
                )
            )

    count = len(conditional_specs) + len(tax_scenarios)
    if count < 1:
        raise HeadlineConditionalError(
            "No active Energy scenario is available for joint application."
        )
    if len(aggregate_ids) != 1:
        raise HeadlineConditionalError(
            "All active Energy scenarios must refer to one common HICP Energy "
            "aggregate run. Active aggregate ids: "
            + ", ".join(sorted(aggregate_ids))
        )
    aggregate_run_id = next(iter(aggregate_ids))
    return {
        "vintage": str(headline_vintage),
        "aggregate_run_id": aggregate_run_id,
        "conditional_specs": conditional_specs,
        "tax_scenarios": tax_scenarios,
        "labels": labels,
        "scenario_count": int(count),
    }


def _joint_summary_children(recipe: dict[str, object] | None):
    if not recipe:
        return html.Span(
            "Joint mode uses every compatible active Energy scenario."
        )
    labels = list(recipe.get("labels") or [])
    return html.Div(
        [
            html.Strong(
                f"{int(recipe.get('scenario_count') or 0)} active Energy scenario"
                + (
                    "s"
                    if int(recipe.get("scenario_count") or 0) != 1
                    else ""
                )
                + " in joint package"
            ),
            html.Ul(
                [html.Li(str(label)) for label in labels],
                style={
                    "margin": "6px 0 0 18px",
                    "padding": "0",
                },
            ),
            html.Div(
                "No marginal Headline effects are added. The component scenarios "
                "are combined first and HICP Energy is chain-linked once, draw by draw.",
                style={"marginTop": "6px"},
            ),
        ]
    )


def _bridge_readiness_for_record(
    headline_store: dict | None,
    row: dict | None,
    *,
    source: str | None,
    horizon: int | None,
    statistic: str | None,
) -> dict[str, object]:
    if str(source or "energy") != "energy":
        return _bridge_readiness_state(
            headline_store,
            None,
            source=source,
            horizon=horizon,
            statistic=statistic,
        )

    if row and bool(row.get("refresh_on_apply")):
        headline = _headline_identity(headline_store)
        spec = dict(row.get("refresh_spec") or {})
        H = int(horizon or 3)
        stat = str(statistic or "mean")
        path_text = (
            "posterior mean"
            if stat == "mean"
            else f"pointwise {stat.upper()}"
        )
        base_label = str(
            row.get("label") or "Energy conditional scenario"
        ).split(" · bridge refreshes")[0]
        return {
            "ready": True,
            "level": "ready",
            "title": (
                "Legacy Energy scenario ready · vintage "
                f"{headline.get('vintage') or '—'}"
            ),
            "detail": (
                f"{base_label} · Energy aggregate "
                f"{str(spec.get('aggregate_run_id') or '')[:12] or '—'} · "
                f"{path_text} · condition H={H}m. The current Headline "
                "bridge will be rebuilt automatically from the saved Energy "
                "posterior when you click Run; Gibbs is not re-estimated."
            ),
        }

    payload = None if row is None else dict(row.get("payload") or {})
    return _bridge_readiness_state(
        headline_store,
        payload,
        source=source,
        horizon=horizon,
        statistic=statistic,
    )


def _bridge_status_children(state: dict[str, object]):
    title = str(state.get("title") or "")
    detail = str(state.get("detail") or "")
    level = str(state.get("level") or "info")
    prefix = {
        "ready": "READY",
        "wait": "WAITING",
        "error": "BLOCKED",
        "info": "INFO",
    }.get(level, "INFO")
    return html.Div(
        [
            html.Strong(f"{prefix} · {title}"),
            html.Span(f" · {detail}" if detail else ""),
        ]
    )


def _run_directory(results_root: Path, store: dict | None) -> Path:
    meta = dict((store or {}).get("meta") or {})
    context = dict((store or {}).get("context") or {})
    vintage = str(meta.get("vintage") or context.get("vintage") or "")
    run_id = str(meta.get("run_id") or meta.get("headline_run_id") or context.get("run_id") or "")
    model_id = str(meta.get("model_id") or context.get("model_id") or "headline_joint")
    if model_id != "headline_joint" or not vintage or not run_id:
        raise HeadlineConditionalError("Select a complete Headline HICP run first.")
    path = results_root / "headline_joint" / vintage / run_id
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _impact_kpis(payload: dict | None) -> tuple[str, str, str, str, str, str, str, str]:
    if not payload:
        return "—", "", "—", "", "—", "", "—", ""
    mean = _q(payload.get("headline_yoy_impact") or {}, "mean")
    dates = pd.DatetimeIndex(pd.to_datetime(payload.get("dates") or []))
    semantics = _impact_semantics(payload)

    def item(i: int):
        if len(mean) <= i or not np.isfinite(mean[i]):
            return "—", ""
        date = "" if len(dates) <= i else pd.Timestamp(dates[i]).strftime("%Y-%m")
        note = f"{date} · posterior mean · vs {semantics['denominator']}"
        return f"{mean[i]:+.2f} pp", note

    a, ad = item(0)
    b, bd = item(1)
    c, cd = item(2)
    meta = payload.get("meta") or {}
    draws = meta.get("n_draws")
    return (
        a,
        ad,
        b,
        bd,
        c,
        cd,
        ("—" if draws is None else f"{int(draws):,}"),
        semantics["draws"],
    )


def register_headline_slice6_callbacks(
    app,
    *,
    results_root,
    project_root=None,
    store_id="data-store",
    energy_scenario_store_id="agg-scenario-store",
    energy_conditional_store_id="conditional-store",
    energy_tax_store_id="scenario-store",
    energy_joint_store_id="joint-energy-scenario-store",
):
    results_root = Path(results_root).resolve()

    @app.callback(
        Output("h6-energy-scenario-select", "options"),
        Output("h6-energy-scenario-select", "value"),
        Input(energy_conditional_store_id, "data"),
        Input(energy_scenario_store_id, "data"),
        Input(store_id, "data"),
        State("h6-energy-scenario-select", "value"),
    )
    def energy_scenario_options(conditional_store, aggregate_store, headline_store, current):
        rows = _energy_scenario_candidates(conditional_store, aggregate_store, headline_store)
        options = [
            {"label": row["label"], "value": row["id"], "disabled": not row["enabled"]}
            for row in rows
        ]
        enabled = [row["id"] for row in rows if row["enabled"]]
        value = current if current in enabled else (enabled[0] if enabled else None)
        return options, value

    @app.callback(
        Output("h6-energy-scenario-select", "disabled"),
        Output("h6-joint-scenario-summary", "children"),
        Input("h6-energy-application-mode", "value"),
        Input(energy_conditional_store_id, "data"),
        Input(energy_tax_store_id, "data"),
        Input(energy_scenario_store_id, "data"),
        Input(energy_joint_store_id, "data"),
        Input(store_id, "data"),
    )
    def energy_application_mode_ui(
        mode,
        conditional_store,
        tax_store,
        aggregate_store,
        joint_store,
        headline_store,
    ):
        joint = str(mode or "selected") == "joint"
        if not joint:
            return False, html.Span(
                "Individual mode applies only the scenario selected below."
            )
        payload = dict(joint_store or {})
        meta = dict(payload.get("meta") or {})
        headline_vintage = str(
            _headline_identity(headline_store).get("vintage") or ""
        )
        joint_vintage = str(meta.get("vintage") or "")
        if (
            payload.get("ok")
            and headline_vintage
            and joint_vintage == headline_vintage
        ):
            labels = list(
                meta.get("dashboard_active_labels")
                or []
            )
            return True, html.Div(
                [
                    html.Strong(
                        f"Joint Energy scenario ready · "
                        f"{int(meta.get('scenario_count', 0))} assumptions · "
                        f"vintage {joint_vintage}"
                    ),
                    html.Ul(
                        [
                            html.Li(
                                f"{item.get('kind', 'Scenario')} · "
                                f"{item.get('component', '')} · "
                                f"{item.get('detail', '')}"
                            )
                            for item in labels
                        ],
                        style={
                            "margin": "6px 0 0 18px",
                            "padding": "0",
                        },
                    ),
                    html.Div(
                        "This is the joint HICP Energy distribution already built "
                        "in Energy → Scenarios; Headline will only select its mean "
                        "or requested Pxx conditioning path.",
                        style={"marginTop": "6px"},
                    ),
                ]
            )
        return True, html.Div(
            [
                html.Strong("Joint Energy scenario not built for this vintage. "),
                dcc.Link(
                    "Open Energy scenarios →",
                    href="/scenarios",
                ),
                html.Span(
                    " Build / refresh the Joint Energy scenario there first."
                ),
            ],
            style={"color": "#991b1b"},
        )

    @app.callback(
        Output("h6-energy-percentile", "disabled"),
        Input("h6-energy-path-kind", "value"),
    )
    def percentile_enabled(path_kind):
        return str(path_kind or "mean") == "mean"

    @app.callback(
        Output("h6-bridge-readiness", "children"),
        Output("h6-bridge-readiness", "className"),
        Output("h6-bridge-details", "children"),
        Output("h6-run", "disabled"),
        Input(store_id, "data"),
        Input(energy_conditional_store_id, "data"),
        Input(energy_tax_store_id, "data"),
        Input(energy_scenario_store_id, "data"),
        Input(energy_joint_store_id, "data"),
        Input("h6-energy-scenario-select", "value"),
        Input("h6-energy-application-mode", "value"),
        Input("h6-source", "value"),
        Input("h6-condition-horizon", "value"),
        Input("h6-energy-path-kind", "value"),
        Input("h6-energy-percentile", "value"),
    )
    def bridge_readiness(
        headline_store,
        conditional_store,
        tax_store,
        aggregate_store,
        joint_store,
        selected_id,
        application_mode,
        source,
        H,
        path_kind,
        percentile,
    ):
        try:
            statistic = _energy_statistic(path_kind, percentile)
        except Exception as exc:
            state = {
                "ready": False,
                "level": "error",
                "title": "Invalid Energy percentile",
                "detail": str(exc),
            }
        else:
            if (
                str(source or "energy") == "energy"
                and str(application_mode or "selected") == "joint"
            ):
                payload = dict(joint_store or {})
                meta = dict(payload.get("meta") or {})
                if not payload.get("ok"):
                    state = {
                        "ready": False,
                        "level": "error",
                        "title": "Joint Energy scenario not built",
                        "detail": (
                            "Open Energy → Scenarios and build / refresh the "
                            "Joint Energy scenario first."
                        ),
                    }
                else:
                    state = _bridge_readiness_state(
                        headline_store,
                        payload,
                        source=source,
                        horizon=H,
                        statistic=statistic,
                    )
                    if state.get("ready"):
                        state["title"] = (
                            "JOINT READY · "
                            f"{int(meta.get('scenario_count', 0))} assumptions · "
                            f"vintage {meta.get('vintage', '—')}"
                        )
                        state["detail"] = (
                            str(state.get("detail") or "")
                            + " · Source: pre-built Joint Energy scenario from "
                            "Energy → Scenarios; Headline does not rebuild Energy."
                        )
            else:
                row = _selected_energy_scenario_record(
                    conditional_store,
                    aggregate_store,
                    headline_store,
                    selected_id,
                )
                state = _bridge_readiness_for_record(
                    headline_store,
                    row,
                    source=source,
                    horizon=H,
                    statistic=statistic,
                )
        level = str(state.get("level") or "info")
        css = "banner-error" if level == "error" else "selection-banner"
        detail = (
            ""
            if level in {"error", "wait"}
            else (
                "Detection is automatic; the selected or joint Energy package "
                "is applied to Headline only when you click Run conditional forecast."
            )
        )
        return _bridge_status_children(state), css, detail, not bool(state.get("ready", False))

    @app.callback(
        Output("h6-headline-scenario-store", "data"),
        Output("h6-status", "children"),
        Input("h6-run", "n_clicks"),
        Input("h6-clear", "n_clicks"),
        State(store_id, "data"),
        State(energy_conditional_store_id, "data"),
        State(energy_tax_store_id, "data"),
        State(energy_scenario_store_id, "data"),
        State(energy_joint_store_id, "data"),
        State("h6-energy-scenario-select", "value"),
        State("h6-energy-application-mode", "value"),
        State("h6-source", "value"),
        State("h6-condition-horizon", "value"),
        State("h6-energy-path-kind", "value"),
        State("h6-energy-percentile", "value"),
        State("h6-manual-variable", "value"),
        State("h6-manual-metric", "value"),
        State("h6-manual-values", "value"),
        prevent_initial_call=True,
    )
    def run_or_clear(
        run_clicks,
        clear_clicks,
        headline_store,
        conditional_store,
        tax_store,
        aggregate_store,
        joint_store,
        selected_id,
        application_mode,
        source,
        H,
        path_kind,
        percentile,
        manual_variable,
        manual_metric,
        manual_values,
    ):
        from dash import ctx
        if ctx.triggered_id == "h6-clear":
            return None, "Conditional scenario cleared."
        if ctx.triggered_id != "h6-run" or not run_clicks:
            raise PreventUpdate
        try:
            H = int(H or 3)
            run_dir = _run_directory(results_root, headline_store)
            posterior = load_saved_headline_posterior(run_dir, project_root=project_root)
            if source == "energy":
                statistic = _energy_statistic(
                    path_kind,
                    percentile,
                )
                if str(application_mode or "selected") == "joint":
                    energy_payload = dict(joint_store or {})
                    if not energy_payload.get("ok"):
                        raise HeadlineConditionalError(
                            "Build / refresh the Joint Energy scenario in "
                            "Energy → Scenarios before applying it to Headline."
                        )
                    joint_meta = dict(energy_payload.get("meta") or {})
                    lineage = {
                        "source_type": "joint_energy_scenario",
                        "energy_application_mode": "joint",
                        "energy_joint_scenario_count": int(
                            joint_meta.get("scenario_count", 0)
                        ),
                        "energy_joint_scenario_labels": list(
                            joint_meta.get("dashboard_active_labels") or []
                        ),
                        "energy_joint_contract": energy_payload.get(
                            "contract_version"
                        ),
                        "energy_marginal_effects_summed": False,
                        "energy_joint_source": (
                            "prebuilt_energy_scenarios_workspace"
                        ),
                        "energy_headline_bridge_rebuilt_on_apply": False,
                        "condition_metric": "level",
                        "condition_variable": "hicp_energy",
                    }
                else:
                    energy_row = _selected_energy_scenario_record(
                        conditional_store,
                        aggregate_store,
                        headline_store,
                        selected_id,
                    )
                    if energy_row is None:
                        raise HeadlineConditionalError(
                            "Select a compatible active Energy scenario."
                        )
                    energy_payload, bridge_rebuilt = _ensure_current_energy_bridge(
                        energy_row,
                        results_root=results_root,
                        project_root=project_root,
                    )
                    lineage = {
                        "source_type": "energy_scenario",
                        "energy_application_mode": "selected",
                        "energy_scenario_selection_id": str(selected_id),
                        "energy_headline_bridge_rebuilt_on_apply": bool(
                            bridge_rebuilt
                        ),
                        "condition_metric": "level",
                        "condition_variable": "hicp_energy",
                    }
                source_aggregate_run_id, energy_forecast_name = (
                    _energy_source_contract(energy_payload)
                )
                lineage.update(
                    {
                        "impact_definition": "energy_scenario_minus_energy_baseline",
                        "impact_baseline_label": "Energy baseline-conditioned",
                        "impact_scenario_label": "Energy scenario-conditioned",
                        "impact_interpretation": "conditional-forecast effect",
                    }
                )
            else:
                _, values = manual_condition_levels(
                    posterior,
                    variable=manual_variable or "hicp_energy",
                    metric=manual_metric or "yoy",
                    values=manual_values or "",
                    H=H,
                )
                conditions = {manual_variable or "hicp_energy": values}
                lineage = {
                    "source_type": "manual",
                    "condition_variable": manual_variable or "hicp_energy",
                    "condition_metric": manual_metric or "yoy",
                    "condition_input_values": str(manual_values or ""),
                    "energy_path_uncertainty_propagated": False,
                    "impact_reference": "unconditional",
                    "impact_definition": "manual_minus_unconditional",
                    "impact_baseline_label": "Headline unconditional",
                    "impact_scenario_label": "Manual conditional",
                    "impact_interpretation": "conditional-forecast effect",
                    "conditioned_horizon_months": H,
                    "free_propagation_horizon_months": max(0, COMPUTATIONAL_HORIZON - H),
                }

            official = pd.Series(posterior.inputs.official_total).astype(float).sort_index()
            observed_yoy = 100.0 * (official / official.shift(12) - 1.0)
            observed_yoy = observed_yoy.replace([np.inf, -np.inf], np.nan).dropna()
            if not observed_yoy.empty:
                lineage.update({
                    "headline_observed_anchor_date": pd.Timestamp(observed_yoy.index[-1]).isoformat(),
                    "headline_observed_anchor_yoy": float(observed_yoy.iloc[-1]),
                })

            if source == "energy":
                payload = run_saved_headline_energy_marginal(
                    run_dir,
                    energy_store=energy_payload,
                    source_aggregate_run_id=source_aggregate_run_id,
                    forecast_name=energy_forecast_name,
                    H=H,
                    statistic=statistic,
                    lineage=lineage,
                    project_root=project_root,
                    persist=True,
                )
            else:
                payload = run_saved_headline_conditional(
                    run_dir,
                    native_level_conditions=conditions,
                    H=H,
                    lineage=lineage,
                    project_root=project_root,
                    persist=True,
                )
            m = payload["meta"]
            semantics = _impact_semantics(payload)
            path_note = (
                f" · {m.get('condition_statistic_label', m.get('condition_statistic', ''))}"
                if m.get("source_type") in {"energy_scenario", "joint_energy_scenario"}
                else ""
            )
            path_note += (
                f" · {semantics['impact']} · "
                f"{semantics['reference']} · {semantics['scenario']}"
            )
            horizon_note = _horizon_semantics(payload)
            return payload, html.Div([
                html.Strong("Conditional complete"),
                html.Span(f" · {m['source_type']}"),
                html.Span(path_note),
                html.Span(f" · vintage {m['headline_vintage']}"),
                html.Span(f" · {m['n_draws']} paired Headline draws"),
                html.Span(f" · {horizon_note}" if horizon_note else ""),
                html.Span(f" · lineage {m['lineage_verification_status']}"),
                html.Span(f" · run {str(m['headline_run_id'])[:12]}"),
                html.Span(" · BVAR re-estimation: NO"),
            ])
        except Exception as exc:
            return None, html.Div([html.Strong("Conditional failed: "), html.Span(str(exc))], className="banner-error")

    @app.callback(
        Output("h6-kpi-1", "children"),
        Output("h6-kpi-1-date", "children"),
        Output("h6-kpi-2", "children"),
        Output("h6-kpi-2-date", "children"),
        Output("h6-kpi-3", "children"),
        Output("h6-kpi-3-date", "children"),
        Output("h6-kpi-draws", "children"),
        Output("h6-kpi-draws-note", "children"),
        Output("h6-main", "figure"),
        Output("h6-impact", "figure"),
        Output("h6-components", "figure"),
        Output("h6-contributions", "figure"),
        Output("h6-summary-table", "data"),
        Input("h6-headline-scenario-store", "data"),
        Input("h6-display-horizon", "value"),
    )
    def figures(payload, display_H):
        h = int(display_H or 6)
        k = _impact_kpis(payload)
        block = (((payload or {}).get("fans") or {}).get("hicp_total") or {}).get("yoy") or {}
        table = paired_blocks_records(
            (payload or {}).get("dates") or [],
            block.get("baseline") or {},
            block.get("conditional") or {},
            (payload or {}).get("headline_yoy_impact") or {},
            max_months=h,
        )
        return (
            *k,
            headline_scenario_main_figure(payload, h),
            headline_scenario_impact_figure(payload, h),
            component_transmission_figure(payload, h),
            contribution_impact_figure(payload, h),
            table,
        )

__all__ = [
    "headline_scenarios_page",
    "register_headline_slice6_callbacks",
    "overlay_headline_conditional_forecast",
    "headline_scenario_main_figure",
    "headline_scenario_impact_figure",
    "component_transmission_figure",
    "contribution_impact_figure",
]
