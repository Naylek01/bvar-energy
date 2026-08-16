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
from inflation_table_contract import (
    PAIRED_EFFECT_COLUMNS,
    paired_blocks_records,
    readable_table,
)
from headline_bvar_conditional import (
    CONDITION_STATISTICS,
    CONDITIONAL_CONTRACT_VERSION,
    ENERGY_BRIDGE_CONTRACT_VERSION,
    HeadlineConditionalError,
    energy_bridge_condition,
    future_dates_for_saved,
    load_saved_headline_posterior,
    manual_condition_levels,
    run_saved_headline_conditional,
)

HORIZONS = (3, 6, 12)
COMPONENTS = ("hicp_energy", "hicp_food", "hicp_neig", "hicp_services")
SERIES_LABELS = {
    "hicp_total": "Headline HICP",
    "hicp_energy": "Energy",
    "hicp_food": "Food",
    "hicp_neig": "NEIG",
    "hicp_services": "Services",
}


def _layout(fig: go.Figure, revision: str, y_title: str | None, height: int = 440) -> go.Figure:
    return apply_inflation_figure_style(
        fig,
        uirevision=revision,
        height=height,
        y_title=y_title,
    )


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


def _fan(fig: go.Figure, dates, block: dict, *, name: str, color: str, bands=("68",)) -> str | None:
    """Plot one posterior fan, returning an explicit diagnostic on failure.

    Conditional computation is always 12 months. The UI horizon (3/6/12)
    is only a display slice, so posterior summaries are sliced before
    validating them against ``dates``.
    """
    n = len(dates)
    if n < 1:
        return f"{name}: no display dates"

    quantiles = {
        key: np.asarray(block.get(key, []), dtype=float)[:n]
        for key in ("q05", "q16", "q50", "q84", "q95")
        if key in block
    }
    mean = _q(block, "mean")[:n]
    q50 = quantiles.get("q50", np.asarray([], dtype=float))

    if not quantiles:
        return f"{name}: no posterior quantiles"
    if len(q50) != n:
        return f"{name}: q50 length={len(q50)} but display dates={n}"
    if len(mean) != n:
        return f"{name}: mean length={len(mean)} but display dates={n}"
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
        if len(values) != n:
            return f"{name}: {key} length={len(values)} but display dates={n}"
        if not np.isfinite(values).all():
            return f"{name}: {key} contains non-finite values"

    for trace in fan_traces(
        dates,
        quantiles,
        bands=bands,
        color=color,
        name=name,
        show_median=False,
    ):
        fig.add_trace(trace)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=mean,
            mode="lines",
            name=f"{name} · posterior mean",
            line=dict(color=color, width=2.2),
            hovertemplate="%{y:.2f}<extra>Posterior mean</extra>",
        )
    )
    return None


def headline_scenario_main_figure(payload: dict | None, horizon: int = 3) -> go.Figure:
    if not payload:
        return _empty("Run a Headline conditional scenario.", 500)
    dates = pd.DatetimeIndex(pd.to_datetime(payload.get("dates") or []))[: int(horizon)]
    block = (((payload.get("fans") or {}).get("hicp_total") or {}).get("yoy") or {})
    base = block.get("baseline") or {}
    cond = block.get("conditional") or {}
    if not len(dates):
        return _empty("Conditional scenario has no future dates.", 500)

    fig = go.Figure()
    errors = [
        _fan(fig, dates, base, name="Baseline", color=INFLATION_COLORS["baseline"], bands=("68",)),
        _fan(fig, dates, cond, name="Conditional", color=INFLATION_COLORS["conditional"], bands=("68",)),
    ]
    errors = [error for error in errors if error]
    if errors:
        return _empty(
            "Baseline vs conditional display unavailable: " + " | ".join(errors),
            500,
        )
    return _layout(
        fig,
        f"h6-main::{payload.get('meta', {}).get('scenario_id')}::{horizon}",
        "% y/y",
        500,
    )


def headline_scenario_impact_figure(payload: dict | None, horizon: int = 3) -> go.Figure:
    if not payload:
        return _empty("Run a Headline conditional scenario.")
    dates = pd.DatetimeIndex(pd.to_datetime(payload.get("dates") or []))[: int(horizon)]
    block = payload.get("headline_yoy_impact") or {}
    n = len(dates)
    if n < 1:
        return _empty("Headline impact is unavailable: no display dates.")

    q16 = _q(block, "q16")[:n]
    q50 = _q(block, "q50")[:n]
    mean = _q(block, "mean")[:n]
    q84 = _q(block, "q84")[:n]

    lengths = {
        "q16": len(q16),
        "q50": len(q50),
        "mean": len(mean),
        "q84": len(q84),
    }
    bad_lengths = {key: value for key, value in lengths.items() if value != n}
    if bad_lengths:
        return _empty(
            f"Headline impact is unavailable: display dates={n}, lengths={bad_lengths}."
        )
    if not all(np.isfinite(values).all() for values in (q16, q50, mean, q84)):
        return _empty("Headline impact is unavailable: posterior summaries contain non-finite values.")

    fig = go.Figure()
    for trace in fan_traces(
        dates,
        {"q16": q16, "q50": q50, "q84": q84},
        bands=("68",),
        color=INFLATION_COLORS["impact"],
        name="Impact",
        show_median=False,
    ):
        fig.add_trace(trace)
    fig.add_trace(go.Scatter(
        x=dates, y=mean, mode="lines+markers", name="Posterior mean impact",
        line=dict(color=INFLATION_COLORS["impact"], width=2), marker=dict(size=5),
    ))
    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    return _layout(
        fig,
        f"h6-impact::{payload.get('meta', {}).get('scenario_id')}::{horizon}",
        "percentage points",
        420,
    )


def component_transmission_figure(payload: dict | None, horizon: int = 3) -> go.Figure:
    if not payload:
        return _empty("Run a Headline conditional scenario.")
    dates = pd.DatetimeIndex(pd.to_datetime(payload.get("dates") or []))[: int(horizon)]
    blocks = payload.get("component_yoy_impact") or {}
    fig = go.Figure()
    found = False
    for j, name in enumerate(COMPONENTS):
        values = _q(blocks.get(name) or {})[: len(dates)]
        if len(values) != len(dates):
            continue
        found = True
        fig.add_trace(go.Scatter(
            x=dates, y=values, mode="lines+markers",
            name=SERIES_LABELS[name], line=dict(color=INFLATION_COLORS[name], width=2), marker=dict(size=5),
        ))
    if not found:
        return _empty("Component transmission is unavailable.")
    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    return _layout(
        fig,
        f"h6-components::{payload.get('meta', {}).get('scenario_id')}::{horizon}",
        "pp vs baseline",
        430,
    )


def contribution_impact_figure(payload: dict | None, horizon: int = 3) -> go.Figure:
    if not payload:
        return _empty("Run a Headline conditional scenario.")
    dates = pd.DatetimeIndex(pd.to_datetime(payload.get("dates") or []))[: int(horizon)]
    blocks = payload.get("contribution_yoy_impact") or {}
    labels = short_horizon_labels(dates)
    fig = go.Figure()
    found = False
    for j, name in enumerate(COMPONENTS):
        values = np.asarray((blocks.get(name) or {}).get("mean", []), dtype=float)[: len(dates)]
        if len(values) != len(dates):
            continue
        found = True
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            name=SERIES_LABELS[name],
            marker_color=INFLATION_COLORS[name],
            hovertemplate="%{y:+.2f} pp<extra>" + SERIES_LABELS[name] + "</extra>",
        ))
    if not found:
        return _empty("Contribution impact is unavailable.")
    fig.update_layout(barmode="relative")
    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    apply_short_horizon_axis(fig, labels)
    return _layout(
        fig,
        f"h6-contrib::{payload.get('meta', {}).get('scenario_id')}::{horizon}",
        "pp contribution impact · posterior mean",
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
                    html.Label("Horizon", className="selector-label"),
                    dcc.Dropdown(
                        id="h6-horizon",
                        options=[{"label": f"{x} months", "value": x} for x in HORIZONS],
                        value=3,
                        clearable=False,
                    ),
                ],
                className="selector-block",
            ),
            html.Div(
                [
                    html.Label("Energy path", className="selector-label"),
                    dcc.Dropdown(
                        id="h6-energy-stat",
                        options=[
                            {
                                "label": ("Posterior mean" if x == "mean" else x.upper()),
                                "value": x,
                            }
                            for x in CONDITION_STATISTICS
                        ],
                        value="mean",
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
                                "Condition the saved Headline BVAR directly, or propagate the live HICP Energy scenario into Headline.",
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
                    _panel_heading("Key scenario results", "Headline conditional effect by horizon", "Posterior means and paired uncertainty. M+1 / M+2 / M+3 are prioritised; M+6 / M+12 appear only when the selected condition horizon reaches them."),
                    readable_table("h6-summary-table", PAIRED_EFFECT_COLUMNS, page_size=7),
                ],
                className="panel table-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            _panel_heading(
                                "Energy → Headline bridge",
                                "Linked scenario",
                                "Hard same-vintage guard. Energy growth is transferred only when the Energy scenario and Headline posterior use the same processed vintage.",
                            ),
                            html.P(
                                "The selected Energy posterior summary (mean / q16 / q50 / q84) is converted to month-to-month HICP Energy growth and re-anchored on the latest observed Headline Energy index. Raw Energy index levels are never imposed on Headline.",
                                className="placeholder-text",
                            ),
                            html.Div(
                                "Checking the active Energy scenario…",
                                id="h6-bridge-readiness",
                                className="selection-banner",
                            ),
                            html.Div(
                                id="h6-bridge-details",
                                className="placeholder-text",
                            ),
                            dcc.Link("Open Energy scenarios →", href="/scenarios", className="refresh-button"),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            _panel_heading(
                                "Manual conditioning",
                                "Alternative component path",
                                "Enter exactly H future values. The BVAR is still solved over 12 months; months after H remain latent and respond endogenously.",
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
                                placeholder="Example for 3 months: 2.5, 2.2, 2.0",
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
                ],
                className="h6-actions",
            ),
            html.Div(
                [
                    _panel_heading("Baseline vs conditional", "Headline HICP — year-on-year"),
                    dcc.Loading(dcc.Graph(id="h6-main", config=graph_config("headline_scenario_main")), type="circle"),
                    html.P(
                        "For linked Energy scenarios, the conditioning path is fixed at the selected Energy posterior summary (mean / q16 / q50 / q84). Energy-path uncertainty itself is not propagated into the Headline conditional fan, so a narrower conditional band must not be interpreted as an economic reduction in uncertainty.",
                        className="placeholder-text",
                    ),
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            _panel_heading("Headline impact", "Conditional − baseline"),
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
                        "Posterior means are shown; quantile bands elsewhere remain quantile intervals.",
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


def _bridge_readiness_state(
    headline_store: dict | None,
    energy_store: dict | None,
    *,
    source: str | None,
    horizon: int | None,
    statistic: str | None,
) -> dict[str, object]:
    """Cheap UI precheck; the hard bridge guards remain model-owned."""
    if str(source or "energy") != "energy":
        return {
            "ready": True,
            "level": "info",
            "title": "Manual conditioning mode",
            "detail": "The Energy → Headline bridge is not required for this run.",
        }

    headline = _headline_identity(headline_store)
    if (
        headline["model_id"] != "headline_joint"
        or not headline["vintage"]
        or not headline["run_id"]
    ):
        return {
            "ready": False,
            "level": "wait",
            "title": "Waiting for a Headline run",
            "detail": "Select a complete Headline HICP saved run before using the linked Energy scenario.",
        }

    if not energy_store:
        return {
            "ready": False,
            "level": "wait",
            "title": "No active Energy scenario",
            "detail": (
                "Open Energy scenarios, configure and propagate a scenario, then return here. "
                "The live scenario stays in dashboard memory while the app remains open."
            ),
        }

    bridge = dict((energy_store or {}).get("headline_bridge") or {})
    if not bridge:
        return {
            "ready": False,
            "level": "error",
            "title": "Energy scenario has no Headline bridge",
            "detail": (
                "The live Energy payload predates the current Headline bridge contract. "
                "Recompute the Energy scenario with the current dashboard."
            ),
        }

    if str(bridge.get("contract") or "") != ENERGY_BRIDGE_CONTRACT_VERSION:
        return {
            "ready": False,
            "level": "error",
            "title": "Stale Energy bridge contract",
            "detail": (
                f"Expected {ENERGY_BRIDGE_CONTRACT_VERSION}; "
                f"received {bridge.get('contract') or 'missing'}. Recompute the Energy scenario."
            ),
        }

    if not bool(bridge.get("scenario_active", False)):
        return {
            "ready": False,
            "level": "wait",
            "title": "Energy aggregate has no active scenario",
            "detail": "Configure at least one Energy scenario component and propagate it before running Headline.",
        }

    energy_vintage = str(bridge.get("vintage") or "")
    if energy_vintage != headline["vintage"]:
        return {
            "ready": False,
            "level": "error",
            "title": "Vintage mismatch",
            "detail": (
                f"Headline={headline['vintage']} · Energy={energy_vintage or 'missing'}. "
                "The bridge is intentionally blocked until both use the same processed vintage."
            ),
        }

    stat = str(statistic or "mean")
    values = ((bridge.get("scenario_level") or {}).get(stat))
    dates = list(bridge.get("dates") or [])
    if values is None:
        return {
            "ready": False,
            "level": "error",
            "title": f"Energy path {stat} is missing",
            "detail": "Recompute the Energy scenario with the current bridge contract.",
        }

    try:
        path = np.asarray(values, dtype=float)
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
    except Exception as exc:
        return {
            "ready": False,
            "level": "error",
            "title": "Energy bridge path is unreadable",
            "detail": str(exc),
        }

    if path.ndim != 1 or len(path) != len(idx) or len(path) < 2:
        return {
            "ready": False,
            "level": "error",
            "title": "Energy bridge path is incomplete",
            "detail": (
                f"dates={len(idx)} · {stat} values={len(path)}. "
                "At least two aligned monthly levels are required to form month-to-month growth."
            ),
        }
    if not np.isfinite(path).all():
        return {
            "ready": False,
            "level": "error",
            "title": "Energy bridge contains non-finite levels",
            "detail": f"The selected {stat} path contains NaN/inf.",
        }

    monthly = idx.to_period("M")
    if monthly.has_duplicates:
        return {
            "ready": False,
            "level": "error",
            "title": "Energy bridge contains duplicate months",
            "detail": "Recompute the Energy scenario before conditioning Headline.",
        }
    if not monthly.is_monotonic_increasing:
        return {
            "ready": False,
            "level": "error",
            "title": "Energy bridge calendar is not ordered",
            "detail": "Recompute the Energy scenario before conditioning Headline.",
        }

    components = list(bridge.get("scenario_components") or [])
    aggregate_run_id = str(bridge.get("aggregate_run_id") or "")
    n_draws = bridge.get("n_draws")
    H = int(horizon or 3)

    component_text = ", ".join(map(str, components)) if components else "active aggregate scenario"
    draw_text = "—" if n_draws is None else f"{int(n_draws):,}"
    return {
        "ready": True,
        "level": "ready",
        "title": f"Bridge ready · vintage {headline['vintage']}",
        "detail": (
            f"Energy aggregate {aggregate_run_id[:12] or '—'} · {component_text} · "
            f"{draw_text} Energy aggregate draws · path={stat} · condition H={H}m. "
            "On Run, exact target-calendar coverage, the historical ratio diagnostic, "
            "same-vintage guard and bit-identical Headline baseline are revalidated."
        ),
    }


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
    q50 = _q(payload.get("headline_yoy_impact") or {})
    dates = pd.DatetimeIndex(pd.to_datetime(payload.get("dates") or []))

    def item(i: int):
        if len(q50) <= i or not np.isfinite(q50[i]):
            return "—", ""
        date = "" if len(dates) <= i else pd.Timestamp(dates[i]).strftime("%Y-%m")
        return f"{q50[i]:+.2f} pp", date

    a, ad = item(0)
    b, bd = item(1)
    c, cd = item(2)
    meta = payload.get("meta") or {}
    draws = meta.get("n_draws")
    return a, ad, b, bd, c, cd, ("—" if draws is None else f"{int(draws):,}"), "paired baseline/conditional"


def register_headline_slice6_callbacks(
    app,
    *,
    results_root,
    project_root=None,
    store_id="data-store",
    energy_scenario_store_id="agg-scenario-store",
):
    results_root = Path(results_root).resolve()

    @app.callback(
        Output("h6-bridge-readiness", "children"),
        Output("h6-bridge-readiness", "className"),
        Output("h6-bridge-details", "children"),
        Output("h6-run", "disabled"),
        Input(store_id, "data"),
        Input(energy_scenario_store_id, "data"),
        Input("h6-source", "value"),
        Input("h6-horizon", "value"),
        Input("h6-energy-stat", "value"),
    )
    def bridge_readiness(headline_store, energy_store, source, H, energy_stat):
        state = _bridge_readiness_state(
            headline_store,
            energy_store,
            source=source,
            horizon=H,
            statistic=energy_stat,
        )
        level = str(state.get("level") or "info")
        css = "banner-error" if level == "error" else "selection-banner"
        detail = (
            ""
            if level in {"error", "wait"}
            else "The linked scenario uses the active in-memory Energy aggregate scenario; it never invents or auto-selects a tax scenario."
        )
        return (
            _bridge_status_children(state),
            css,
            detail,
            not bool(state.get("ready", False)),
        )

    @app.callback(
        Output("h6-headline-scenario-store", "data"),
        Output("h6-status", "children"),
        Input("h6-run", "n_clicks"),
        Input("h6-clear", "n_clicks"),
        State(store_id, "data"),
        State(energy_scenario_store_id, "data"),
        State("h6-source", "value"),
        State("h6-horizon", "value"),
        State("h6-energy-stat", "value"),
        State("h6-manual-variable", "value"),
        State("h6-manual-metric", "value"),
        State("h6-manual-values", "value"),
        prevent_initial_call=True,
    )
    def run_or_clear(
        run_clicks, clear_clicks, headline_store, energy_store, source, H,
        energy_stat, manual_variable, manual_metric, manual_values,
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
            target = future_dates_for_saved(posterior, H)

            if source == "energy":
                values, lineage = energy_bridge_condition(
                    energy_store or {},
                    posterior=posterior,
                    target_dates=target,
                    statistic=energy_stat or "mean",
                    project_root=project_root,
                )
                conditions = {"hicp_energy": values}
                lineage.update({"condition_metric": "level", "condition_variable": "hicp_energy"})
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
                }

            payload = run_saved_headline_conditional(
                run_dir,
                native_level_conditions=conditions,
                H=H,
                lineage=lineage,
                project_root=project_root,
                persist=True,
            )
            m = payload["meta"]
            return payload, html.Div(
                [
                    html.Strong("Conditional complete"),
                    html.Span(f" · {m['source_type']}"),
                    html.Span(f" · vintage {m['headline_vintage']}"),
                    html.Span(f" · {m['n_draws']} paired draws"),
                    html.Span(f" · condition {m['condition_horizon']}m / compute {m['computational_horizon']}m"),
                    html.Span(f" · lineage {m['lineage_verification_status']}"),
                    html.Span(f" · run {str(m['headline_run_id'])[:12]}"),
                    html.Span(" · BVAR re-estimation: NO"),
                ]
            )
        except Exception as exc:
            return None, html.Div(
                [html.Strong("Conditional failed: "), html.Span(str(exc))],
                className="banner-error",
            )

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
        Input("h6-horizon", "value"),
    )
    def figures(payload, H):
        h = int(H or 3)
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
