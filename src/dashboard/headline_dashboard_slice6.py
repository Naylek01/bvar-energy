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
CONDITIONAL_WINDOWS_HARMONIZED_V1_HEADLINE_DASH = True

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
            line=dict(color=color, width=2.2, dash=("dash" if color == INFLATION_COLORS["conditional"] else "solid")),
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
    # Explicit economic-object labels; never expose letter shorthand.
    meta = dict((payload or {}).get("meta") or {})
    definition = str(meta.get("impact_definition") or "")
    baseline_label = str(meta.get("impact_baseline_label") or "")
    scenario_label = str(meta.get("impact_scenario_label") or "")
    interpretation = str(meta.get("impact_interpretation") or "")

    if definition and baseline_label and scenario_label:
        return {
            "definition": definition,
            "reference": baseline_label,
            "scenario": scenario_label,
            "impact": f"{scenario_label} − {baseline_label}",
            "denominator": baseline_label,
            "draws": f"{baseline_label} vs {scenario_label}",
            "interpretation": interpretation,
        }

    reference = str(meta.get("impact_reference") or "")
    source_type = str(meta.get("source_type") or "")
    if source_type == "manual" or reference == "unconditional":
        baseline_label = "Headline unconditional"
        scenario_label = "Manual conditional"
        definition = "manual_minus_unconditional"
    elif reference == "energy_baseline_conditioned":
        # Legacy display support only. Lot C3 no longer creates Energy scenarios here.
        baseline_label = "Headline conditioned on Energy baseline"
        scenario_label = "Headline conditioned on Energy scenario"
        definition = "legacy_energy_scenario_minus_energy_baseline"
    elif source_type in {"energy_scenario", "joint_energy_scenario"}:
        baseline_label = "Headline unconditional"
        scenario_label = "Headline conditioned on Energy scenario"
        definition = "legacy_energy_minus_unconditional"
    else:
        baseline_label = "Headline unconditional"
        scenario_label = "Conditional"
        definition = "conditional_minus_unconditional"

    return {
        "definition": definition,
        "reference": baseline_label,
        "scenario": scenario_label,
        "impact": f"{scenario_label} − {baseline_label}",
        "denominator": baseline_label,
        "draws": f"{baseline_label} vs {scenario_label}",
        "interpretation": interpretation,
    }


def _horizon_semantics(payload: dict | None) -> str:
    meta = dict((payload or {}).get('meta') or {})
    start = meta.get('condition_start_offset', meta.get('condition_start', 1))
    end = meta.get('condition_end_offset', meta.get('condition_end'))
    conditioned = meta.get('conditioned_horizon_months', meta.get('condition_horizon'))
    computational = meta.get('computational_horizon')
    try:
        start_i = int(start or 1)
        conditioned_i = int(conditioned)
        computational_i = int(computational)
        end_i = int(end) if end is not None else start_i + conditioned_i - 1
        unconstrained_i = max(0, computational_i - conditioned_i)
    except (TypeError, ValueError):
        return ''
    return f'hard-conditioned M+{start_i}→M+{end_i} ({conditioned_i}m) · {unconstrained_i}m unconstrained but jointly conditioned inside {computational_i}m DK horizon'


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


# HEADLINE_SCENARIOS_TWO_VISIBLE_GRAPHS_V1_1
# HEADLINE_SCENARIOS_AB_CHAINLINK_NO_INSET_V1

# HEADLINE_CONDITIONING_ASSUMPTION_GRAPH_V1

# HEADLINE_MANUAL_CONDITIONING_MODES_V1
def _headline_persisted_component_yoy_q50(
    posterior,
    *,
    variable: str,
    condition_start: int,
    condition_end: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    import json

    if variable not in COMPONENTS:
        raise HeadlineConditionalError(
            f"Unknown Headline component {variable!r}."
        )

    start = int(condition_start)
    end = int(condition_end)
    if start < 1 or end < start or end > COMPUTATIONAL_HORIZON:
        raise HeadlineConditionalError(
            "Persisted-baseline window must satisfy "
            f"1 <= start <= end <= {COMPUTATIONAL_HORIZON}; "
            f"received M+{start}..M+{end}."
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

    if component_yoy_paths.ndim != 3:
        raise HeadlineConditionalError(
            "Persisted component_yoy_paths must be three-dimensional."
        )
    if component_yoy_paths.shape[2] != len(COMPONENTS):
        raise HeadlineConditionalError(
            "Persisted component_yoy_paths component dimension changed: "
            f"{component_yoy_paths.shape[2]} != {len(COMPONENTS)}."
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

    variable_index = COMPONENTS.index(variable)
    selected_draws = future_draws[
        :,
        start - 1:end,
        variable_index,
    ]
    if selected_draws.shape[1] != end - start + 1:
        raise HeadlineConditionalError(
            "Persisted Headline q50 window length is inconsistent."
        )

    q50 = np.nanquantile(selected_draws, 0.50, axis=0)
    if (
        q50.ndim != 1
        or len(q50) != end - start + 1
        or not np.isfinite(q50).all()
    ):
        raise HeadlineConditionalError(
            "Persisted unconditional component YoY q50 is invalid."
        )

    return (
        pd.DatetimeIndex(
            future_dates[start - 1:end],
            name="date",
        ),
        np.asarray(q50, dtype=float),
    )


# HEADLINE_CONDITIONING_ASSUMPTION_ENERGY_CLONE_V2
def headline_conditioning_assumption_figure(
    payload: dict | None,
    horizon: int = 6,
) -> go.Figure:
    if not payload:
        return _empty("Run a manual Headline conditional scenario.", 490)

    display = dict(
        (payload or {}).get("dashboard_conditioning_assumption") or {}
    )
    variable = str(display.get("variable") or "")
    metric = str(display.get("metric") or "")
    if variable not in COMPONENTS or metric != "yoy":
        return _empty("Conditioning assumption metadata is unavailable.", 490)

    all_dates = pd.DatetimeIndex(
        pd.to_datetime((payload or {}).get("dates") or [])
    )
    n = min(max(int(horizon or 6), 1), len(all_dates))
    dates = all_dates[:n]
    if n < 1:
        return _empty("Conditional scenario has no future dates.", 490)

    fan_block = (
        (((payload or {}).get("fans") or {}).get(variable) or {}).get("yoy")
        or {}
    )
    baseline = dict(fan_block.get("baseline") or {})

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
        return _empty("Unconditional component YoY fan is unavailable.", 490)

    observed_dates = pd.DatetimeIndex(
        pd.to_datetime(display.get("observed_dates") or [])
    )
    observed_values = np.asarray(
        display.get("observed_values") or [],
        dtype=float,
    )
    if len(observed_dates) != len(observed_values):
        return _empty("Observed component YoY history is misaligned.", 490)

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
    # HEADLINE_FULL_HISTORY_SAFE_JOIN_V1
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

    start = int(display.get("condition_start_offset") or 1)
    end = int(display.get("condition_end_offset") or start)
    imposed = np.asarray(display.get("imposed_values") or [], dtype=float)
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
                    line={"color": "#009FE3", "width": 2.6, "dash": "dash"},
                    marker={"size": 5},
                )
            )

    fig.update_layout(
        template="plotly_white",
        margin={"l": 64, "r": 26, "t": 66, "b": 54},
        height=490,
        title={
            "text": (
                "Conditioning assumption — "
                + SERIES_LABELS.get(variable, variable)
            ),
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
            "h6-conditioning-assumption::"
            f"{(payload.get('meta') or {}).get('scenario_id')}::{variable}::{n}"
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, linecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#eef0f3", zerolinecolor="#d1d5db")
    return fig


def headline_scenario_main_figure(payload: dict | None, horizon: int = 3) -> go.Figure:
    if not payload:
        return _empty("Run a manual Headline conditional scenario.", 500)
    dates = _scenario_dates(payload, horizon)
    block = (((payload.get("fans") or {}).get("hicp_total") or {}).get("yoy") or {})
    base = block.get("baseline") or {}
    cond = block.get("conditional") or {}
    if not len(dates):
        return _empty("Conditional scenario has no future dates.", 500)

    semantics = _impact_semantics(payload)
    baseline_label = str(semantics.get("reference") or "Headline unconditional")
    conditional_label = str(semantics.get("scenario") or "Manual conditional")

    anchor_date, anchor_value = _observed_anchor(payload)
    fig = go.Figure()
    if anchor_value is not None:
        fig.add_trace(go.Scatter(
            x=["Actual"], y=[anchor_value], mode="markers", name="Latest observed",
            marker=dict(size=8, color=INFLATION_COLORS["observed"]),
            customdata=[anchor_date or "Latest observed"],
            hovertemplate="Actual · %{customdata}<br>%{y:.2f}%<extra>Latest observed</extra>",
        ))

    errors = [
        _fan(fig, dates, base, name=baseline_label,
             color=INFLATION_COLORS["baseline"], bands=("68",), anchor_value=anchor_value),
        _fan(fig, dates, cond, name=conditional_label,
             color=INFLATION_COLORS["conditional"], bands=("68",), anchor_value=anchor_value),
    ]
    errors = [error for error in errors if error]
    if errors:
        return _empty("Baseline vs manual conditional display unavailable: " + " | ".join(errors), 500)

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
        return _empty("Run a manual Headline conditional scenario.")
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
        name="Posterior mean impact", line=dict(color=INFLATION_COLORS["impact"], width=2),
        marker=dict(size=5),
        hovertemplate="%{x} · %{customdata}<br>%{y:+.3f} pp<extra>Posterior mean impact</extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    apply_short_horizon_axis(fig, labels)
    _symmetric_zero_axis(fig, q16, q84, mean)
    _layout(fig, f"h6-impact::{payload.get('meta', {}).get('scenario_id')}::{horizon}",
            "percentage points", 420)

    note = (
        f"{semantics['impact']} · baseline: {semantics['reference']} · "
        f"conditional: {semantics['scenario']}"
    )
    if horizon_note:
        note += " · " + horizon_note
    if semantics.get("interpretation"):
        note += " · " + semantics["interpretation"]
    fig.add_annotation(
        x=0, y=1.12, xref="paper", yref="paper", text=note, showarrow=False,
        xanchor="left", yanchor="bottom", align="left",
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
        values = np.asarray(
            (blocks.get(name) or {}).get("mean", []),
            dtype=float,
        )[: len(dates)]
        if len(values) != len(dates) or not np.isfinite(values).all():
            continue
        plotted.append(values)
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            customdata=_calendar_hover(dates),
            name=SERIES_LABELS[name],
            marker_color=INFLATION_COLORS[name],
            hovertemplate=(
                "%{x} · %{customdata}<br>%{y:+.3f} pp<extra>"
                + SERIES_LABELS[name]
                + "</extra>"
            ),
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
    return _layout(
        fig,
        f"h6-contrib::{payload.get('meta', {}).get('scenario_id')}::{horizon}",
        "pp",
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


# HEADLINE_SCENARIOS_MANUAL_ONLY_LOTC3_V1
HEADLINE_SCENARIO_PATH_EDITOR_V1 = True

def headline_scenarios_page() -> html.Div:
    controls = html.Div([html.Div([html.Label('Condition start', className='selector-label'), dcc.Dropdown(id='h6-condition-start', options=[{'label': f'M+{x}', 'value': x} for x in range(1, 13)], value=1, clearable=False)], className='selector-block'), html.Div([html.Label('Condition end', className='selector-label'), dcc.Dropdown(id='h6-condition-horizon', options=[{'label': f'M+{x}', 'value': x} for x in range(1, 13)], value=3, clearable=False)], className='selector-block'), html.Div([html.Label('Display horizon', className='selector-label'), dcc.Dropdown(id='h6-display-horizon', options=[{'label': f'{x} months', 'value': x} for x in HORIZONS], value=6, clearable=False)], className='selector-block')], className='chart-controls')
    return html.Div([dcc.Store(id='h6-headline-scenario-store', storage_type='session'), dcc.Store(id='h6-condition-set-store', storage_type='memory'), dcc.Store(id='h6-condition-selected-store', storage_type='memory'), html.Div([html.Div([html.H2('Headline scenarios', className='page-title'), html.P('Conditional paths for the saved Headline BVAR.', className='page-subtitle')]), controls], className='page-heading-row'), html.Div(id='h6-status', className='selection-banner'), html.Div([_stat_card('+1m impact', 'h6-kpi-1', 'h6-kpi-1-date'), _stat_card('+2m impact', 'h6-kpi-2', 'h6-kpi-2-date'), _stat_card('+3m impact', 'h6-kpi-3', 'h6-kpi-3-date'), _stat_card('Paired draws', 'h6-kpi-draws', 'h6-kpi-draws-note')], className='stats-grid', id='h6-results-kpis', style={'display': 'none'}), html.Div([_panel_heading('Conditional path', 'Headline component path', 'Set the selected Headline component over the conditioning window.'), html.Div([html.Div([html.Label('Variable', className='selector-label'), dcc.Dropdown(id='h6-manual-variable', options=[{'label': SERIES_LABELS[x], 'value': x} for x in COMPONENTS], value='hicp_energy', clearable=False)], className='selector-block'), html.Div([html.Label('Conditioning mode', className='selector-label'), dcc.Dropdown(id='h6-manual-metric', options=[{'label': 'YoY target (%)', 'value': 'yoy'}, {'label': 'Δ vs unconditional (pp)', 'value': 'delta_pp'}], value='yoy', clearable=False)], className='selector-block')], className='selectors-grid'), html.Label('Conditioned values', className='selector-label'), html.Div([html.Div([html.Label('Set all', className='selector-label'), html.Div([dcc.Input(id='h6-manual-fill-all-value', type='number', step='any', placeholder='Value', style={'width': '100%'}), html.Button('Apply', id='h6-manual-fill-all', n_clicks=0, className='refresh-button h6-secondary-button')], style={'display': 'grid', 'gridTemplateColumns': 'minmax(120px, 1fr) auto', 'gap': '8px', 'alignItems': 'center'})]), html.Div([html.Label('Linear path', className='selector-label'), html.Div([dcc.Input(id='h6-manual-linear-start', type='number', step='any', placeholder='Start', style={'width': '100%'}), dcc.Input(id='h6-manual-linear-end', type='number', step='any', placeholder='End', style={'width': '100%'}), html.Button('Fill', id='h6-manual-fill-linear', n_clicks=0, className='refresh-button h6-secondary-button')], style={'display': 'grid', 'gridTemplateColumns': 'minmax(100px, 1fr) minmax(100px, 1fr) auto', 'gap': '8px', 'alignItems': 'center'})])], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(280px, 1fr))', 'gap': '12px', 'marginBottom': '14px'}), html.Div([html.Div([html.Label('M+1', className='selector-label'), dcc.Input(id='h6-manual-value-m1', type='number', step='any', style={'width': '100%'})], id='h6-manual-value-wrap-m1', style={'display': 'block'}), html.Div([html.Label('M+2', className='selector-label'), dcc.Input(id='h6-manual-value-m2', type='number', step='any', style={'width': '100%'})], id='h6-manual-value-wrap-m2', style={'display': 'block'}), html.Div([html.Label('M+3', className='selector-label'), dcc.Input(id='h6-manual-value-m3', type='number', step='any', style={'width': '100%'})], id='h6-manual-value-wrap-m3', style={'display': 'block'}), html.Div([html.Label('M+4', className='selector-label'), dcc.Input(id='h6-manual-value-m4', type='number', step='any', style={'width': '100%'})], id='h6-manual-value-wrap-m4', style={'display': 'none'}), html.Div([html.Label('M+5', className='selector-label'), dcc.Input(id='h6-manual-value-m5', type='number', step='any', style={'width': '100%'})], id='h6-manual-value-wrap-m5', style={'display': 'none'}), html.Div([html.Label('M+6', className='selector-label'), dcc.Input(id='h6-manual-value-m6', type='number', step='any', style={'width': '100%'})], id='h6-manual-value-wrap-m6', style={'display': 'none'}), html.Div([html.Label('M+7', className='selector-label'), dcc.Input(id='h6-manual-value-m7', type='number', step='any', style={'width': '100%'})], id='h6-manual-value-wrap-m7', style={'display': 'none'}), html.Div([html.Label('M+8', className='selector-label'), dcc.Input(id='h6-manual-value-m8', type='number', step='any', style={'width': '100%'})], id='h6-manual-value-wrap-m8', style={'display': 'none'}), html.Div([html.Label('M+9', className='selector-label'), dcc.Input(id='h6-manual-value-m9', type='number', step='any', style={'width': '100%'})], id='h6-manual-value-wrap-m9', style={'display': 'none'}), html.Div([html.Label('M+10', className='selector-label'), dcc.Input(id='h6-manual-value-m10', type='number', step='any', style={'width': '100%'})], id='h6-manual-value-wrap-m10', style={'display': 'none'}), html.Div([html.Label('M+11', className='selector-label'), dcc.Input(id='h6-manual-value-m11', type='number', step='any', style={'width': '100%'})], id='h6-manual-value-wrap-m11', style={'display': 'none'}), html.Div([html.Label('M+12', className='selector-label'), dcc.Input(id='h6-manual-value-m12', type='number', step='any', style={'width': '100%'})], id='h6-manual-value-wrap-m12', style={'display': 'none'})], id='h6-manual-path-grid', style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(100px, 1fr))', 'gap': '10px'}), dcc.Textarea(id='h6-manual-values', style={'display': 'none'})], className='panel'), html.Div([_panel_heading('Active scenarios', 'Calculated Headline conditionals', 'Add / update computes the marginal scenario immediately. Click a calculated scenario to inspect it; click Joint effect to inspect all active conditions simultaneously.'), html.Button('Add / update conditional', id='h6-condition-add', n_clicks=0, className='refresh-button'), html.Div([html.Div([html.Button(id='h6-card-select-hicp-energy', n_clicks=0, style={'width': '100%', 'textAlign': 'left'}), html.Button('×', id='h6-card-remove-hicp-energy', n_clicks=0, title='Remove Energy scenario', style={'border': 'none', 'background': 'transparent', 'cursor': 'pointer', 'fontSize': '18px'})], id='h6-card-wrap-hicp-energy', style={'display': 'none'}), html.Div([html.Button(id='h6-card-select-hicp-food', n_clicks=0, style={'width': '100%', 'textAlign': 'left'}), html.Button('×', id='h6-card-remove-hicp-food', n_clicks=0, title='Remove Food scenario', style={'border': 'none', 'background': 'transparent', 'cursor': 'pointer', 'fontSize': '18px'})], id='h6-card-wrap-hicp-food', style={'display': 'none'}), html.Div([html.Button(id='h6-card-select-hicp-neig', n_clicks=0, style={'width': '100%', 'textAlign': 'left'}), html.Button('×', id='h6-card-remove-hicp-neig', n_clicks=0, title='Remove NEIG scenario', style={'border': 'none', 'background': 'transparent', 'cursor': 'pointer', 'fontSize': '18px'})], id='h6-card-wrap-hicp-neig', style={'display': 'none'}), html.Div([html.Button(id='h6-card-select-hicp-services', n_clicks=0, style={'width': '100%', 'textAlign': 'left'}), html.Button('×', id='h6-card-remove-hicp-services', n_clicks=0, title='Remove Services scenario', style={'border': 'none', 'background': 'transparent', 'cursor': 'pointer', 'fontSize': '18px'})], id='h6-card-wrap-hicp-services', style={'display': 'none'}), html.Div([html.Button(id='h6-card-select-joint', n_clicks=0, style={'width': '100%', 'textAlign': 'left'})], id='h6-card-wrap-joint', style={'display': 'none'})], id='h6-computed-scenario-cards', style={'marginTop': '12px'}), html.Button('Reset all', id='h6-condition-reset', n_clicks=0, style={'border': 'none', 'background': 'transparent', 'textDecoration': 'underline', 'cursor': 'pointer', 'padding': '8px 2px', 'fontSize': '13px'})], className='panel'), html.Div([html.Progress(id='h6-progress', value=0, max=100, className='estimation-progress'), html.Div([html.Div('Idle', id='h6-progress-phase', className='estimation-phase'), html.Div('Run a conditional forecast to see progress.', id='h6-progress-detail', className='estimation-progress-detail')], className='estimation-progress-text')], className='estimation-progress-wrap'), html.Div([_panel_heading('Conditioning assumption', 'Selected HICP component · YoY', 'Observed history, unconditional forecast, conditional forecast and exact imposed path.'), dcc.Loading(dcc.Graph(id='h6-conditioning-assumption', config=graph_config('headline_conditioning_assumption')), type='circle')], className='panel chart-panel', id='h6-results-conditioning', style={'display': 'none'}), html.Div([_panel_heading('Key scenario results', 'Headline conditional effect by display horizon', 'Later months evolve endogenously after the conditioning window.'), readable_table('h6-summary-table', PAIRED_EFFECT_COLUMNS, page_size=12)], className='panel table-panel', id='h6-results-summary', style={'display': 'none'}), html.Div([_panel_heading('Headline paths', 'Headline HICP — baseline vs conditional path', None), dcc.Loading(dcc.Graph(id='h6-main', config=graph_config('headline_scenario_main')), type='circle')], className='panel chart-panel', id='h6-results-main', style={'display': 'none'}), html.Div([_panel_heading('Exact additive decomposition', 'Change in component contributions · conditional minus baseline', None), dcc.Loading(dcc.Graph(id='h6-contributions', config=graph_config('headline_scenario_contributions')), type='circle')], className='panel chart-panel', id='h6-results-contributions', style={'display': 'none'}), html.Div([dcc.Graph(id='h6-impact', config=graph_config('headline_scenario_impact')), dcc.Graph(id='h6-components', config=graph_config('headline_scenario_components'))], style={'display': 'none'})], className='page-body headline-page')

def _display_identity(store: dict | None) -> dict[str, str]:
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
    headline_vintage = _display_identity(headline_store).get("vintage", "")
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
    headline = _display_identity(headline_store)
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
    headline_vintage = _display_identity(
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
        headline = _display_identity(headline_store)
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


def _run_directory(
    results_root: Path,
    display_store: dict | None,
    production_store: dict | None,
    headline_run_resolver,
) -> Path:
    """Resolve the production Headline run from durable production state."""
    # DURABLE_HEADLINE_RUN_IDENTITY_V1
    display = _display_identity(display_store)
    display_vintage = str(display.get("vintage") or "")
    production_vintage = str((production_store or {}).get("vintage") or "")

    if not production_vintage:
        raise HeadlineConditionalError(
            "No production vintage is selected. "
            "Select a production vintage before running Headline scenarios."
        )

    if display_vintage and display_vintage != production_vintage:
        raise HeadlineConditionalError(
            "Display/production vintage mismatch: "
            f"display={display_vintage!r} production={production_vintage!r}. "
            "The display store is not used to choose the Headline posterior; "
            "align the displayed context with the production vintage first."
        )

    if headline_run_resolver is None:
        raise HeadlineConditionalError(
            "Headline production-run resolver is unavailable."
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
            "Headline production run is not uniquely usable: "
            f"vintage={production_vintage!r} status={status!r} "
            f"note={note!r} complete_run_ids=[{candidate_text}]."
            f"{action}"
        )

    run_id = str(state.get("run_id") or "")
    directory = state.get("directory")
    if not run_id or directory is None:
        raise HeadlineConditionalError(
            "Headline resolver returned COMPLETE without both run_id and directory: "
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
            "Headline resolver directory disagrees with the dashboard results root: "
            f"resolved={resolved} expected={expected}."
        )
    if not expected.is_dir():
        raise FileNotFoundError(expected)
    return expected


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


def register_headline_slice6_callbacks(app, *, results_root, project_root=None, store_id='data-store', production_vintage_store_id='production-vintage-store', headline_run_resolver=None, energy_scenario_store_id='agg-scenario-store', energy_conditional_store_id='conditional-store', energy_tax_store_id='scenario-store', energy_joint_store_id='joint-energy-scenario-store'):
    results_root = Path(results_root).resolve()
    display_store_id = store_id
    if headline_run_resolver is None:
        raise RuntimeError('register_headline_slice6_callbacks requires headline_run_resolver.')
    _ = (energy_scenario_store_id, energy_conditional_store_id, energy_tax_store_id, energy_joint_store_id)

    @app.callback(Output('h6-manual-value-wrap-m1', 'style'), Output('h6-manual-value-wrap-m2', 'style'), Output('h6-manual-value-wrap-m3', 'style'), Output('h6-manual-value-wrap-m4', 'style'), Output('h6-manual-value-wrap-m5', 'style'), Output('h6-manual-value-wrap-m6', 'style'), Output('h6-manual-value-wrap-m7', 'style'), Output('h6-manual-value-wrap-m8', 'style'), Output('h6-manual-value-wrap-m9', 'style'), Output('h6-manual-value-wrap-m10', 'style'), Output('h6-manual-value-wrap-m11', 'style'), Output('h6-manual-value-wrap-m12', 'style'), Output('h6-manual-values', 'value'), Input('h6-condition-start', 'value'), Input('h6-condition-horizon', 'value'), Input('h6-manual-value-m1', 'value'), Input('h6-manual-value-m2', 'value'), Input('h6-manual-value-m3', 'value'), Input('h6-manual-value-m4', 'value'), Input('h6-manual-value-m5', 'value'), Input('h6-manual-value-m6', 'value'), Input('h6-manual-value-m7', 'value'), Input('h6-manual-value-m8', 'value'), Input('h6-manual-value-m9', 'value'), Input('h6-manual-value-m10', 'value'), Input('h6-manual-value-m11', 'value'), Input('h6-manual-value-m12', 'value'))
    def headline_manual_path_sync(condition_start, condition_end, *values):
        start = int(condition_start or 1)
        end = int(condition_end or start)
        start = min(max(start, 1), 12)
        end = min(max(end, start), 12)
        styles = [{'display': 'block'} if start <= i <= end else {'display': 'none'} for i in range(1, 13)]
        active = list(values)[start - 1:end]
        pieces = []
        for value in active:
            if value is None or value == '':
                pieces.append('')
            else:
                pieces.append(f'{float(value):.15g}')
        return (*styles, ', '.join(pieces))

    @app.callback(Output('h6-manual-value-m1', 'value'), Output('h6-manual-value-m2', 'value'), Output('h6-manual-value-m3', 'value'), Output('h6-manual-value-m4', 'value'), Output('h6-manual-value-m5', 'value'), Output('h6-manual-value-m6', 'value'), Output('h6-manual-value-m7', 'value'), Output('h6-manual-value-m8', 'value'), Output('h6-manual-value-m9', 'value'), Output('h6-manual-value-m10', 'value'), Output('h6-manual-value-m11', 'value'), Output('h6-manual-value-m12', 'value'), Input('h6-manual-fill-all', 'n_clicks'), Input('h6-manual-fill-linear', 'n_clicks'), State('h6-condition-start', 'value'), State('h6-condition-horizon', 'value'), State('h6-manual-fill-all-value', 'value'), State('h6-manual-linear-start', 'value'), State('h6-manual-linear-end', 'value'), State('h6-manual-value-m1', 'value'), State('h6-manual-value-m2', 'value'), State('h6-manual-value-m3', 'value'), State('h6-manual-value-m4', 'value'), State('h6-manual-value-m5', 'value'), State('h6-manual-value-m6', 'value'), State('h6-manual-value-m7', 'value'), State('h6-manual-value-m8', 'value'), State('h6-manual-value-m9', 'value'), State('h6-manual-value-m10', 'value'), State('h6-manual-value-m11', 'value'), State('h6-manual-value-m12', 'value'), prevent_initial_call=True)
    def headline_manual_path_fill(_set_all_clicks, _linear_clicks, condition_start, condition_end, set_all_value, linear_start, linear_end, *current_values):
        from dash import ctx
        start = int(condition_start or 1)
        end = int(condition_end or start)
        start = min(max(start, 1), 12)
        end = min(max(end, start), 12)
        values = list(current_values)
        if len(values) != 12:
            raise PreventUpdate
        if ctx.triggered_id == 'h6-manual-fill-all':
            if set_all_value is None:
                raise PreventUpdate
            fill_value = float(set_all_value)
            for idx in range(start - 1, end):
                values[idx] = fill_value
        elif ctx.triggered_id == 'h6-manual-fill-linear':
            if linear_start is None or linear_end is None:
                raise PreventUpdate
            first = float(linear_start)
            last = float(linear_end)
            count = end - start + 1
            if count == 1:
                path = [first]
            else:
                path = [first + (last - first) * step / (count - 1) for step in range(count)]
            values[start - 1:end] = path
        else:
            raise PreventUpdate
        return tuple(values)

    def _headline_compute_recipe_payload(run_dir, posterior, recipes, *, application_kind):
        recipes = [dict(recipe) for recipe in recipes]
        if not recipes:
            raise HeadlineConditionalError('No Headline condition recipe supplied.')
        windows = {(int(recipe.get('condition_start') or 1), int(recipe.get('condition_end') or 1)) for recipe in recipes}
        if len(windows) != 1:
            readable = ', '.join((f'{recipe.get('label', recipe.get('variable'))}: M+{int(recipe.get('condition_start') or 1)}..M+{int(recipe.get('condition_end') or 1)}' for recipe in recipes))
            raise HeadlineConditionalError(f'Joint Headline conditions must share one conditioning window. Active windows: {readable}.')
        condition_start, condition_end = next(iter(windows))
        H = condition_end - condition_start + 1
        conditions = {}
        assumptions = {}
        for recipe in recipes:
            variable = str(recipe.get('variable') or '')
            if variable not in COMPONENTS:
                raise HeadlineConditionalError(f'Unknown Headline component {variable!r}.')
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
                reference_dates, reference_q50 = _headline_persisted_component_yoy_q50(posterior, variable=variable, condition_start=condition_start, condition_end=condition_end)
                target_yoy = np.asarray(reference_q50, dtype=float) + raw_values
            text_values = ', '.join((f'{float(value):.15g}' for value in target_yoy))
            condition_dates, levels = manual_condition_levels(posterior, variable=variable, metric='yoy', values=text_values, H=H, condition_start=condition_start)
            if reference_dates is not None and (not pd.DatetimeIndex(condition_dates).equals(pd.DatetimeIndex(reference_dates))):
                raise HeadlineConditionalError(f'{variable}: delta-vs-unconditional calendar mismatch.')
            conditions[variable] = levels
            assumptions[variable] = {'contract': 'headline-conditioning-assumption-display-v1', 'variable': variable, 'metric': 'yoy', 'input_mode': input_mode, 'input_mode_label': 'YoY target (%)' if input_mode == 'yoy' else 'Δ vs unconditional (pp)', 'input_values': [float(value) for value in raw_values], 'imposed_values': [float(value) for value in target_yoy], 'unconditional_reference_q50': None if reference_q50 is None else [float(value) for value in np.asarray(reference_q50, dtype=float)], 'observed_dates': [pd.Timestamp(date).isoformat() for date in observed_yoy.index], 'observed_values': [float(value) for value in observed_yoy.to_numpy(dtype=float)], 'condition_start_offset': condition_start, 'condition_end_offset': condition_end}
        variables = [str(recipe['variable']) for recipe in recipes]
        joint = len(variables) > 1
        lineage = {'source_type': 'manual_joint' if joint else 'manual', 'condition_variables': list(variables), 'condition_metric': 'yoy', 'condition_input_modes': {variable: assumptions[variable]['input_mode'] for variable in variables}, 'condition_input_values': {variable: assumptions[variable]['input_values'] for variable in variables}, 'condition_target_yoy_values': {variable: assumptions[variable]['imposed_values'] for variable in variables}, 'unconditional_reference_q50': {variable: assumptions[variable]['unconditional_reference_q50'] for variable in variables}, 'condition_application_mode': application_kind, 'energy_path_uncertainty_propagated': False, 'impact_reference': 'unconditional', 'impact_definition': 'joint_manual_minus_unconditional' if joint else 'manual_minus_unconditional', 'impact_baseline_label': 'Headline unconditional', 'impact_scenario_label': 'Joint conditional' if joint else 'Manual conditional', 'impact_interpretation': 'conditional-forecast effect', 'conditioned_horizon_months': H, 'free_propagation_horizon_months': max(0, COMPUTATIONAL_HORIZON - H), 'condition_start_offset': condition_start, 'condition_end_offset': condition_end, 'free_before_condition_months': condition_start - 1, 'free_after_condition_months': COMPUTATIONAL_HORIZON - condition_end}
        official = pd.Series(posterior.inputs.official_total).astype(float).sort_index()
        official_yoy = (100.0 * (official / official.shift(12) - 1.0)).replace([np.inf, -np.inf], np.nan).dropna()
        if not official_yoy.empty:
            lineage.update({'headline_observed_anchor_date': pd.Timestamp(official_yoy.index[-1]).isoformat(), 'headline_observed_anchor_yoy': float(official_yoy.iloc[-1])})
        payload = run_saved_headline_conditional(run_dir, native_level_conditions=conditions, H=H, lineage=lineage, project_root=project_root, persist=True, condition_start=condition_start)
        payload = dict(payload)
        payload['dashboard_conditioning_assumptions'] = assumptions
        payload['dashboard_conditioning_assumption'] = assumptions[variables[0]]
        payload['dashboard_condition_set'] = {'contract': 'headline-computed-scenario-set-v1', 'application_mode': application_kind, 'condition_variables': list(variables)}
        return payload

    def _headline_recompute_joint(run_dir, posterior, scenarios):
        active = [name for name in COMPONENTS if name in scenarios]
        if len(active) < 2:
            return None
        recipes = [dict(scenarios[name]['recipe']) for name in active]
        payload = _headline_compute_recipe_payload(run_dir, posterior, recipes, application_kind='joint')
        return {'variables': active, 'payload': payload}

    @app.callback(Output('h6-condition-set-store', 'data'), Output('h6-condition-selected-store', 'data'), Output('h6-status', 'children'), Input('h6-condition-add', 'n_clicks'), Input('h6-condition-reset', 'n_clicks'), Input('h6-card-remove-hicp-energy', 'n_clicks'), Input('h6-card-remove-hicp-food', 'n_clicks'), Input('h6-card-remove-hicp-neig', 'n_clicks'), Input('h6-card-remove-hicp-services', 'n_clicks'), State('h6-condition-set-store', 'data'), State('h6-condition-selected-store', 'data'), State(display_store_id, 'data'), State(production_vintage_store_id, 'data'), State('h6-condition-start', 'value'), State('h6-condition-horizon', 'value'), State('h6-manual-variable', 'value'), State('h6-manual-metric', 'value'), State('h6-manual-value-m1', 'value'), State('h6-manual-value-m2', 'value'), State('h6-manual-value-m3', 'value'), State('h6-manual-value-m4', 'value'), State('h6-manual-value-m5', 'value'), State('h6-manual-value-m6', 'value'), State('h6-manual-value-m7', 'value'), State('h6-manual-value-m8', 'value'), State('h6-manual-value-m9', 'value'), State('h6-manual-value-m10', 'value'), State('h6-manual-value-m11', 'value'), State('h6-manual-value-m12', 'value'), background=True, running=[(Output('h6-condition-add', 'disabled'), True, False), (Output('h6-condition-reset', 'disabled'), True, False)], progress=[Output('h6-progress', 'value'), Output('h6-progress-phase', 'children'), Output('h6-progress-detail', 'children')], progress_default=(0, 'Idle', 'Add / update a conditional to compute it.'), prevent_initial_call=True)
    def mutate_headline_computed_scenario_set(set_progress, _add_clicks, _reset_clicks, _remove_energy, _remove_food, _remove_neig, _remove_services, store, selected, display_store, production_store, condition_start, condition_end, variable, metric, *cell_values):
        from dash import ctx
        current = dict(store or {})
        scenarios = dict(current.get('scenarios') or {})
        trigger = ctx.triggered_id
        if trigger == 'h6-condition-reset':
            set_progress((0, 'Idle', 'All Headline conditionals reset.'))
            return ({'contract': 'headline-computed-scenario-set-v1', 'scenarios': {}, 'joint': None}, None, 'All Headline conditionals reset.')
        remove_map = {'h6-card-remove-hicp-energy': 'hicp_energy', 'h6-card-remove-hicp-food': 'hicp_food', 'h6-card-remove-hicp-neig': 'hicp_neig', 'h6-card-remove-hicp-services': 'hicp_services'}
        if trigger in remove_map:
            key = remove_map[trigger]
            if key not in scenarios:
                raise PreventUpdate
            scenarios.pop(key, None)
            remaining = [name for name in COMPONENTS if name in scenarios]
            new_selected = str(selected or '')
            if new_selected == key or new_selected == '__joint__':
                new_selected = remaining[0] if remaining else None
            joint = None
            if len(remaining) >= 2:
                set_progress((35, 'Updating joint effect', 'Recomputing remaining active Headline conditions jointly.'))
                run_dir = _run_directory(results_root, display_store, production_store, headline_run_resolver)
                posterior = load_saved_headline_posterior(run_dir, project_root=project_root)
                joint = _headline_recompute_joint(run_dir, posterior, scenarios)
            set_progress((100, 'Scenario set updated', f'{len(remaining)} active Headline scenario(s).'))
            return ({'contract': 'headline-computed-scenario-set-v1', 'scenarios': scenarios, 'joint': joint}, new_selected, 'Headline conditional removed.')
        if trigger != 'h6-condition-add':
            raise PreventUpdate
        start = int(condition_start or 1)
        end = int(condition_end or start)
        if start < 1 or end < start or end > COMPUTATIONAL_HORIZON:
            raise HeadlineConditionalError(f'Condition window must satisfy 1 <= start <= end <= {COMPUTATIONAL_HORIZON}; received M+{start}..M+{end}.')
        if len(cell_values) != 12:
            raise HeadlineConditionalError(f'Headline editor expected 12 cells; got {len(cell_values)}.')
        active = list(cell_values)[start - 1:end]
        missing = [start + index for index, value in enumerate(active) if value is None or value == '']
        if missing:
            raise HeadlineConditionalError('Enter every active month; missing ' + ', '.join((f'M+{month}' for month in missing)) + '.')
        values = np.asarray(active, dtype=float)
        if not np.isfinite(values).all():
            raise HeadlineConditionalError('Headline condition values must be finite.')
        variable = str(variable or 'hicp_energy')
        if variable not in COMPONENTS:
            raise HeadlineConditionalError(f'Unknown Headline component {variable!r}.')
        input_mode = str(metric or 'yoy')
        if input_mode not in {'yoy', 'delta_pp'}:
            raise HeadlineConditionalError(f'Unknown Headline conditioning mode {input_mode!r}.')
        recipe = {'variable': variable, 'label': SERIES_LABELS.get(variable, variable), 'condition_start': start, 'condition_end': end, 'input_mode': input_mode, 'values': [float(value) for value in values]}
        set_progress((20, 'Loading saved posterior', 'Resolving the durable production Headline run.'))
        run_dir = _run_directory(results_root, display_store, production_store, headline_run_resolver)
        posterior = load_saved_headline_posterior(run_dir, project_root=project_root)
        set_progress((45, 'Computing marginal effect', f'Running {SERIES_LABELS.get(variable, variable)} alone.'))
        marginal = _headline_compute_recipe_payload(run_dir, posterior, [recipe], application_kind='marginal')
        scenarios[variable] = {'recipe': recipe, 'payload': marginal}
        set_progress((75, 'Updating joint effect', 'Recomputing all active Headline conditions simultaneously.'))
        joint = _headline_recompute_joint(run_dir, posterior, scenarios)
        set_progress((100, 'Conditional scenario active', f'{len(scenarios)} calculated Headline scenario(s) active.'))
        return ({'contract': 'headline-computed-scenario-set-v1', 'scenarios': scenarios, 'joint': joint}, variable, html.Div([html.Strong('Conditional added / updated'), html.Span(f' · {SERIES_LABELS.get(variable, variable)} marginal calculated'), html.Span(f' · joint recalculated' if joint is not None else ' · joint available after a second active condition'), html.Span(' · BVAR re-estimation: NO')]))

    @app.callback(Output('h6-card-select-hicp-energy', 'children'), Output('h6-card-wrap-hicp-energy', 'style'), Output('h6-card-select-hicp-food', 'children'), Output('h6-card-wrap-hicp-food', 'style'), Output('h6-card-select-hicp-neig', 'children'), Output('h6-card-wrap-hicp-neig', 'style'), Output('h6-card-select-hicp-services', 'children'), Output('h6-card-wrap-hicp-services', 'style'), Output('h6-card-select-joint', 'children'), Output('h6-card-wrap-joint', 'style'), Input('h6-condition-set-store', 'data'), Input('h6-condition-selected-store', 'data'))
    def render_headline_computed_scenario_cards(store, selected):
        current = dict(store or {})
        scenarios = dict(current.get('scenarios') or {})
        selected = str(selected or '')
        visible_base = {'display': 'grid', 'gridTemplateColumns': '1fr auto', 'gap': '6px', 'alignItems': 'center', 'marginTop': '7px', 'borderRadius': '10px', 'padding': '4px'}
        hidden = {'display': 'none'}
        out = []
        for variable in ('hicp_energy', 'hicp_food', 'hicp_neig', 'hicp_services'):
            row = scenarios.get(variable)
            if not row:
                out.extend(['', hidden])
                continue
            recipe = dict(row.get('recipe') or {})
            label = SERIES_LABELS.get(variable, variable)
            mode = 'YoY target' if recipe.get('input_mode') == 'yoy' else 'Δ vs unconditional'
            children = [html.Div([html.Strong(str(label).upper()), html.Span('Selected', style={'display': 'inline-block' if selected == variable else 'none', 'marginLeft': '8px', 'fontSize': '10px', 'fontWeight': '700', 'textTransform': 'uppercase', 'color': '#2563eb'})]), html.Div(f'M+{recipe.get('condition_start')}..M+{recipe.get('condition_end')} · {mode} · marginal effect calculated', style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '3px'})]
            style = dict(visible_base)
            style.update({'background': '#eff6ff' if selected == variable else '#ffffff', 'border': '1px solid #2563eb' if selected == variable else '1px solid #e2e8f0'})
            out.extend([children, style])
        joint = current.get('joint')
        if joint and joint.get('payload'):
            labels = [SERIES_LABELS.get(variable, variable) for variable in joint.get('variables') or []]
            joint_children = [html.Div([html.Strong('JOINT EFFECT'), html.Span('Selected', style={'display': 'inline-block' if selected == '__joint__' else 'none', 'marginLeft': '8px', 'fontSize': '10px', 'fontWeight': '700', 'textTransform': 'uppercase', 'color': '#2563eb'})]), html.Div(' + '.join(labels) + ' · simultaneous conditional calculation', style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '3px'})]
            joint_style = dict(visible_base)
            joint_style.update({'gridTemplateColumns': '1fr', 'background': '#eff6ff' if selected == '__joint__' else '#ffffff', 'border': '1px solid #2563eb' if selected == '__joint__' else '1px solid #e2e8f0'})
        else:
            joint_children = ''
            joint_style = hidden
        return (*out, joint_children, joint_style)

    @app.callback(Output('h6-condition-selected-store', 'data', allow_duplicate=True), Input('h6-card-select-hicp-energy', 'n_clicks'), Input('h6-card-select-hicp-food', 'n_clicks'), Input('h6-card-select-hicp-neig', 'n_clicks'), Input('h6-card-select-hicp-services', 'n_clicks'), Input('h6-card-select-joint', 'n_clicks'), prevent_initial_call=True)
    def select_headline_computed_scenario(_energy, _food, _neig, _services, _joint):
        from dash import ctx
        mapping = {'h6-card-select-hicp-energy': 'hicp_energy', 'h6-card-select-hicp-food': 'hicp_food', 'h6-card-select-hicp-neig': 'hicp_neig', 'h6-card-select-hicp-services': 'hicp_services', 'h6-card-select-joint': '__joint__'}
        selected = mapping.get(ctx.triggered_id)
        if not selected:
            raise PreventUpdate
        return selected

    @app.callback(Output('h6-headline-scenario-store', 'data'), Input('h6-condition-set-store', 'data'), Input('h6-condition-selected-store', 'data'))
    def project_headline_computed_scenario(store, selected):
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
        for variable in COMPONENTS:
            if variable in scenarios:
                return (scenarios[variable] or {}).get('payload')
        return None

    @app.callback(Output('h6-kpi-1', 'children'), Output('h6-kpi-1-date', 'children'), Output('h6-kpi-2', 'children'), Output('h6-kpi-2-date', 'children'), Output('h6-kpi-3', 'children'), Output('h6-kpi-3-date', 'children'), Output('h6-kpi-draws', 'children'), Output('h6-kpi-draws-note', 'children'), Output('h6-conditioning-assumption', 'figure'), Output('h6-main', 'figure'), Output('h6-impact', 'figure'), Output('h6-components', 'figure'), Output('h6-contributions', 'figure'), Output('h6-summary-table', 'data'), Output('h6-results-kpis', 'style'), Output('h6-results-conditioning', 'style'), Output('h6-results-summary', 'style'), Output('h6-results-main', 'style'), Output('h6-results-contributions', 'style'), Input('h6-headline-scenario-store', 'data'), Input('h6-display-horizon', 'value'), Input('url', 'pathname'))
    def figures(payload, display_H, pathname):
        if (pathname or "") not in {"/scenarios/headline", "/headline/scenarios"}:
            raise PreventUpdate
        h = int(display_H or 6)
        k = _impact_kpis(payload)
        block = (((payload or {}).get('fans') or {}).get('hicp_total') or {}).get('yoy') or {}
        table = paired_blocks_records((payload or {}).get('dates') or [], block.get('baseline') or {}, block.get('conditional') or {}, (payload or {}).get('headline_yoy_impact') or {}, max_months=h)

        def _safe(fn, *args, label):
            try:
                return fn(*args)
            except Exception as exc:
                message = f'{label} unavailable: {type(exc).__name__}: {exc}'
                print(f'[Headline Scenarios] {message}', flush=True)
                return _empty(message)
        result_style = {} if bool(payload) else {'display': 'none'}
        return (*k, _safe(headline_conditioning_assumption_figure, payload, h, label='Conditioning assumption'), _safe(headline_scenario_main_figure, payload, h, label='Headline baseline vs manual conditional'), _safe(headline_scenario_impact_figure, payload, h, label='Manual conditional Headline impact'), _safe(component_transmission_figure, payload, h, label='Component response'), _safe(contribution_impact_figure, payload, h, label='Exact additive decomposition'), table, result_style, result_style, result_style, result_style, result_style)

__all__ = [
    "headline_scenarios_page",
    "register_headline_slice6_callbacks",
    "overlay_headline_conditional_forecast",
    "headline_scenario_main_figure",
    "headline_scenario_impact_figure",
    "component_transmission_figure",
    "contribution_impact_figure",
]
