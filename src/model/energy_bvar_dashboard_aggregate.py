"""Aggregate page for the energy BVAR dashboard.

Kept in its own module so the pure builders can be tested without importing
Dash. The page reads ``display_v1.parquet`` produced by
:func:`energy_bvar_display.build_aggregate_display`. Forecast aggregation remains
the responsibility of Notebook 10 / ``run_aggregate``; an optional historical
posterior BVAR-fitted overlay is materialised separately with the same
chain-linking contract and cached beside the aggregate run.

What the aggregate exposes that a component run cannot
------------------------------------------------------
The component BVARs for gas, electricity, petrol, diesel and liquid fuels
forecast a **pre-tax price**, so their year-on-year rate is not HICP inflation.
Only the aggregate store has been through the tax bridge, the weekly-to-monthly
conversion and the HICP rebasing, which is why ``metric="yoy"`` on
``hicp_energy`` lives here and not on the Forecast page.
"""

from __future__ import annotations

from io import StringIO
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

__all__ = [
    "AGGREGATE_METRIC_LABELS",
    "COMPONENT_LABELS",
    "aggregate_metric_options",
    "aggregate_kpis",
    "aggregate_fan_figure",
    "aggregate_contribution_figure",
    "aggregate_diagnostics_table",
    "aggregate_live_scenario_payload",
    "aggregate_live_scenario_figure",
    "aggregate_live_impact_figure",
    "aggregate_live_contribution_impact_figure",
    "aggregate_live_scenario_kpis",
]


AGGREGATE_METRIC_LABELS: dict[str, str] = {
    "yoy": "HICP Energy inflation (% y/y)",
    "level": "HICP Energy index",
    "yoy_impact": "Tax-scenario impact (pp)",
}

COMPONENT_LABELS: dict[str, str] = {
    "car_fuels": "Car fuels",
    "liquid_fuels": "Liquid fuels",
    "gas": "Gas",
    "electricity": "Electricity",
    "heat_energy": "Heat energy",
    "solid_fuels": "Solid fuels",
}

# One hue per component, stable across charts so a colour always means the same
# component. Cyan is reserved for the aggregate itself.
COMPONENT_COLOURS: dict[str, str] = {
    "car_fuels": "#004B99",
    "liquid_fuels": "#0079AE",
    "gas": "#A66A00",
    "electricity": "#0B7A5C",
    "heat_energy": "#6E4B9E",
    "solid_fuels": "#B3341F",
}

_AGG_SERIES = "hicp_energy"
_ACCENT = "#009FE3"
_INK = "#111827"
_MUTED = "#6b7280"
_GRID = "#eef0f3"
_NOWCAST = "#A66A00"
_SCENARIO = "#0B7A5C"


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


def _rows(frame: pd.DataFrame, record_type: str, metric: str, series: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = (frame["record_type"].astype(str) == record_type) & (
        frame["metric"].astype(str) == metric
    )
    if series is not None:
        mask &= frame["series"].astype(str) == series
    return frame.loc[mask].sort_values("date")


def aggregate_metric_options(frame: pd.DataFrame) -> tuple[list[dict], str | None]:
    """Metric dropdown contents, YoY first because it is the headline number."""
    if frame.empty:
        return [], None
    fan = frame.loc[frame["record_type"].astype(str) == "fan"]
    present = [str(value) for value in fan["metric"].dropna().unique()]
    order = ["yoy", "level", "yoy_impact"]
    metrics = [m for m in order if m in present] + [m for m in present if m not in order]
    options = [
        {"label": AGGREGATE_METRIC_LABELS.get(m, m.replace("_", " ").capitalize()), "value": m}
        for m in metrics
    ]
    default = "yoy" if "yoy" in metrics else (metrics[0] if metrics else None)
    return options, default


def aggregate_kpis(frame: pd.DataFrame, metric: str = "yoy") -> dict[str, Any]:
    """Headline numbers for the four cards.

    ``latest_observed`` comes from the published history, the two medians from
    the predictive fan. Values are returned unformatted so the caller decides
    the presentation.
    """
    out: dict[str, Any] = {
        "latest_observed": None,
        "latest_observed_date": None,
        "first_future": None,
        "first_future_date": None,
        "terminal": None,
        "terminal_date": None,
        "n_draws": None,
    }
    if frame.empty:
        return out

    history = _rows(frame, "history", metric, _AGG_SERIES).dropna(subset=["value"])
    if not history.empty:
        out["latest_observed"] = float(history["value"].iloc[-1])
        out["latest_observed_date"] = pd.Timestamp(history["date"].iloc[-1])

    fan = _rows(frame, "fan", metric, _AGG_SERIES)
    future = fan.loc[fan["is_future"].fillna(False).astype(bool)]
    if future.empty:
        future = fan
    if not future.empty:
        out["first_future"] = float(future["q50"].iloc[0])
        out["first_future_date"] = pd.Timestamp(future["date"].iloc[0])
        out["terminal"] = float(future["q50"].iloc[-1])
        out["terminal_date"] = pd.Timestamp(future["date"].iloc[-1])

    meta = frame.loc[frame["record_type"].astype(str) == "meta"]
    draws = meta.loc[meta["key"].astype(str) == "n_aggregate_draws_effective", "text_value"]
    if not draws.empty:
        try:
            out["n_draws"] = int(float(draws.iloc[0]))
        except (TypeError, ValueError):
            out["n_draws"] = None
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _rgba(hex_color: str, opacity: float) -> str:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{opacity})"


def _band(
    fig: go.Figure, fan: pd.DataFrame, lower: str, upper: str, name: str,
    opacity: float, *, color: str = _ACCENT,
) -> None:
    if fan.empty:
        return
    x = pd.concat([fan["date"], fan["date"].iloc[::-1]], ignore_index=True)
    y = pd.concat([fan[upper], fan[lower].iloc[::-1]], ignore_index=True)
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", fill="toself", fillcolor=_rgba(color, opacity),
            line={"width": 0}, hoverinfo="skip", name=name, legendgroup="fan",
        )
    )


def _append_anchor(block: pd.DataFrame, *, date, value) -> pd.DataFrame:
    if block.empty or date is None or pd.isna(date) or value is None or pd.isna(value):
        return block
    anchor = {column: np.nan for column in block.columns}
    anchor["date"] = pd.Timestamp(date)
    for column in ("q05", "q16", "q50", "q84", "q95"):
        if column in block.columns:
            anchor[column] = float(value)
    return pd.concat([pd.DataFrame([anchor]), block], ignore_index=True).sort_values("date")


def _layout(fig: go.Figure, *, title: str, unit: str | None, uirevision: str, height: int = 500) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        margin={"l": 54, "r": 24, "t": 54, "b": 42},
        height=height,
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 18, "color": _INK}},
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#374151", "size": 12},
        xaxis_title=None,
        yaxis_title=unit,
        hovermode="x unified",
        dragmode="pan",
        hoverlabel={"bgcolor": "white", "bordercolor": "#e5e7eb", "font": {"color": _INK}},
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right", "font": {"size": 11}},
        uirevision=uirevision,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor="#e5e7eb", tickfont={"color": _MUTED})
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor="#d1d5db", tickfont={"color": _MUTED})
    return fig


def empty_aggregate_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
        font={"size": 13, "color": _MUTED},
    )
    fig.update_layout(
        template="plotly_white", height=420, margin={"l": 54, "r": 24, "t": 54, "b": 42},
        xaxis={"visible": False}, yaxis={"visible": False},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def aggregate_fan_figure(
    frame: pd.DataFrame,
    *,
    metric: str = "yoy",
    fan_mode: str = "both",
    uirevision: str = "aggregate",
    forecast_origin=None,
    last_observed=None,
    show_six_model_components: bool = False,
    show_bvar_reconstruction: bool = False,
) -> go.Figure:
    """Observed -> nowcast -> forecast fan for the saved baseline aggregate."""
    if frame.empty or not metric:
        return empty_aggregate_figure("No aggregate display artefact loaded")

    history = _rows(frame, "history", metric, _AGG_SERIES).dropna(subset=["value"])
    fan = _rows(frame, "fan", metric, _AGG_SERIES)
    if "basis" in fan.columns:
        baseline = fan.loc[fan["basis"].astype(str) == "baseline"].copy()
        if not baseline.empty:
            fan = baseline
    fan = fan.sort_values("date")
    if fan.empty and history.empty:
        return empty_aggregate_figure(f"The aggregate store carries no '{metric}' rows")

    if last_observed is None or pd.isna(pd.to_datetime(last_observed, errors="coerce")):
        last_observed = history["date"].max() if not history.empty else pd.NaT
    else:
        last_observed = pd.Timestamp(last_observed)
    origin = pd.to_datetime(forecast_origin, errors="coerce")
    if pd.isna(origin):
        future_flag = fan.loc[fan["is_future"].fillna(False).astype(bool)]
        origin = future_flag["date"].min() if not future_flag.empty else pd.NaT

    if not pd.isna(last_observed):
        nowcast = fan.loc[(fan["date"] > last_observed) & ((fan["date"] < origin) if not pd.isna(origin) else True)].copy()
    else:
        nowcast = fan.iloc[0:0].copy()
    forecast = fan.loc[fan["date"] >= origin].copy() if not pd.isna(origin) else fan.copy()

    fig = go.Figure()
    if fan_mode in {"90", "both"}:
        _band(fig, nowcast, "q05", "q95", "Nowcast 90% interval", 0.10, color=_NOWCAST)
        _band(fig, forecast, "q05", "q95", "Forecast 90% interval", 0.12, color=_ACCENT)
    if fan_mode in {"68", "both"}:
        _band(fig, nowcast, "q16", "q84", "Nowcast 68% interval", 0.20, color=_NOWCAST)
        _band(fig, forecast, "q16", "q84", "Forecast 68% interval", 0.25, color=_ACCENT)

    if not history.empty:
        hist = history.loc[history["date"] <= last_observed] if not pd.isna(last_observed) else history
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["value"], mode="lines", name="Observed",
            line={"color": _INK, "width": 1.8},
            hovertemplate="%{y:.2f}<extra>Observed</extra>",
        ))

    # Optional posterior in-sample BVAR fitted overlays. The first control
    # shows the six fitted HICP component blocks separately; the second shows
    # their draw-wise chain-linked Laspeyres HICP Energy reconstruction.
    if metric in {"level", "yoy"} and show_six_model_components:
        component_rows = frame.loc[
            (frame["record_type"].astype(str) == "bvar_fitted_component")
            & (frame["metric"].astype(str) == str(metric))
        ].copy()
        for name in COMPONENT_LABELS:
            block = component_rows.loc[
                component_rows["series"].astype(str) == name
            ].sort_values("date")
            if block.empty:
                continue
            label = COMPONENT_LABELS.get(name, name.replace("_", " ").title())
            fig.add_trace(
                go.Scatter(
                    x=block["date"],
                    y=block["q50"],
                    mode="lines",
                    name=f"{label} BVAR fit",
                    line={
                        "color": COMPONENT_COLOURS.get(name, _MUTED),
                        "width": 1.25,
                    },
                    opacity=0.82,
                    hovertemplate="%{y:.2f}<extra>" + label + " BVAR fit</extra>",
                    legendgroup="six-model-components",
                )
            )

    if metric in {"level", "yoy"} and show_bvar_reconstruction:
        reconstructed = frame.loc[
            (frame["record_type"].astype(str) == "bvar_fitted_aggregate")
            & (frame["metric"].astype(str) == str(metric))
        ].sort_values("date")
        if not reconstructed.empty:
            fig.add_trace(
                go.Scatter(
                    x=reconstructed["date"],
                    y=reconstructed["q50"],
                    mode="lines",
                    name="BVAR model reconstruction",
                    line={"color": "#4b5563", "width": 2.0, "dash": "dot"},
                    hovertemplate=(
                        "%{y:.2f}<extra>Posterior BVAR fit + Laspeyres</extra>"
                    ),
                    legendgroup="bvar-reconstruction",
                )
            )
        else:
            reason_rows = frame.loc[
                (frame["record_type"].astype(str) == "meta")
                & (frame["key"].astype(str) == "bvar_fitted_reason")
            ]
            reason = (
                str(reason_rows["text_value"].iloc[-1])
                if not reason_rows.empty and pd.notna(reason_rows["text_value"].iloc[-1])
                else "Posterior fitted cache unavailable for one or more component runs."
            )
            fig.add_annotation(
                x=0.01, y=0.99, xref="paper", yref="paper",
                text="BVAR fitted reconstruction unavailable: " + reason,
                showarrow=False, align="left", xanchor="left", yanchor="top",
                font={"size": 11, "color": _MUTED},
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#e5e7eb", borderwidth=1, borderpad=6,
            )

    anchor_date = None
    anchor_value = None
    if not history.empty and not pd.isna(last_observed):
        hist_anchor = history.loc[history["date"] <= last_observed]
        if not hist_anchor.empty:
            anchor_date = hist_anchor["date"].iloc[-1]
            anchor_value = hist_anchor["value"].iloc[-1]
    now_line = _append_anchor(nowcast, date=anchor_date, value=anchor_value)
    if not now_line.empty:
        fig.add_trace(go.Scatter(
            x=now_line["date"], y=now_line["q50"], mode="lines", name="Nowcast median",
            line={"color": _NOWCAST, "width": 2.2},
            hovertemplate="%{y:.2f}<extra>Nowcast</extra>",
        ))
        anchor_date = now_line["date"].iloc[-1]
        anchor_value = now_line["q50"].iloc[-1]
    fc_line = _append_anchor(forecast, date=anchor_date, value=anchor_value)
    if not fc_line.empty:
        fig.add_trace(go.Scatter(
            x=fc_line["date"], y=fc_line["q50"], mode="lines", name="Forecast median",
            line={"color": _ACCENT, "width": 2.4, "dash": "dash"},
            hovertemplate="%{y:.2f}<extra>Forecast</extra>",
        ))

    if not pd.isna(last_observed) and not nowcast.empty:
        fig.add_vline(x=last_observed, line={"color": _MUTED, "width": 1, "dash": "dash"})
    if not pd.isna(origin):
        fig.add_vline(
            x=origin, line={"color": _MUTED, "width": 1, "dash": "dot"},
            annotation_text="Forecast", annotation_position="top left",
            annotation={"font": {"size": 10, "color": _MUTED}},
        )

    unit = None
    if not fan.empty and fan["unit"].notna().any():
        unit = str(fan["unit"].dropna().iloc[0])
    elif not history.empty and history["unit"].notna().any():
        unit = str(history["unit"].dropna().iloc[0])
    if metric == "yoy":
        fig.add_hline(y=0.0, line={"color": "#d1d5db", "width": 1})

    return _layout(
        fig, title=AGGREGATE_METRIC_LABELS.get(metric, metric), unit=unit,
        uirevision=uirevision,
    )

def aggregate_contribution_figure(
    frame: pd.DataFrame,
    *,
    basis: str = "baseline",
    uirevision: str = "aggregate-contrib",
    forecast_origin=None,
    last_observed=None,
) -> go.Figure:
    """Median component contributions to aggregate YoY, as stacked bars.

    Contributions are summarised per draw and then plotted, so the stack does
    not add exactly to the median aggregate rate: the median of a sum is not the
    sum of medians. The aggregate median is overlaid as a line so the size of
    that gap is visible rather than hidden.
    """
    contrib = frame.loc[
        (frame["record_type"].astype(str) == "contribution")
        & (frame["metric"].astype(str) == "contribution_yoy")
        & (frame["basis"].astype(str) == basis)
    ]
    if contrib.empty:
        return empty_aggregate_figure("The aggregate store carries no contribution rows")

    fig = go.Figure()
    order = [name for name in COMPONENT_LABELS if name in set(contrib["series"].astype(str))]
    order += [
        name for name in sorted(set(contrib["series"].astype(str))) if name not in order
    ]
    for name in order:
        block = contrib.loc[contrib["series"].astype(str) == name].sort_values("date")
        fig.add_trace(
            go.Bar(
                x=block["date"], y=block["q50"],
                name=COMPONENT_LABELS.get(name, name.replace("_", " ").capitalize()),
                marker_color=COMPONENT_COLOURS.get(name, _MUTED),
                hovertemplate="%{y:+.2f} pp<extra>" + COMPONENT_LABELS.get(name, name) + "</extra>",
            )
        )

    fan = _rows(frame, "fan", "yoy", _AGG_SERIES)
    if "basis" in fan.columns:
        chosen = fan.loc[fan["basis"].astype(str) == basis]
        if not chosen.empty:
            fan = chosen
    if not fan.empty:
        fig.add_trace(
            go.Scatter(
                x=fan["date"], y=fan["q50"], mode="lines", name="HICP Energy (median)",
                line={"color": _INK, "width": 2},
                hovertemplate="%{y:.2f}%<extra>Aggregate median</extra>",
            )
        )

    origin = pd.to_datetime(forecast_origin, errors="coerce")
    last_obs = pd.to_datetime(last_observed, errors="coerce")
    if not pd.isna(last_obs):
        fig.add_vline(x=last_obs, line={"color": _MUTED, "width": 1, "dash": "dash"})
    if not pd.isna(origin):
        fig.add_vline(x=origin, line={"color": _MUTED, "width": 1, "dash": "dot"})
    fig.update_layout(barmode="relative")
    fig.add_hline(y=0.0, line={"color": "#d1d5db", "width": 1})
    return _layout(
        fig,
        title="Component contributions to HICP Energy inflation",
        unit="percentage points",
        uirevision=uirevision,
        height=440,
    )


def _quantile_frame(paths: np.ndarray, dates, *, name: str) -> pd.DataFrame:
    values = np.asarray(paths, dtype=float)
    dates = pd.DatetimeIndex(dates, name="date")
    if values.ndim != 2 or values.shape[1] != len(dates):
        raise ValueError(f"{name}: expected draw x date paths, got {values.shape}.")
    q = np.nanquantile(values, (0.05, 0.16, 0.50, 0.84, 0.95), axis=0).T
    out = pd.DataFrame(q, index=dates, columns=("q05", "q16", "q50", "q84", "q95")).reset_index()
    out["name"] = name
    return out


def _json_frame(frame: pd.DataFrame) -> str:
    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.to_json(orient="split", date_format="iso", double_precision=15)


def _read_json_frame(payload: Mapping | None, key: str) -> pd.DataFrame:
    if not payload or not payload.get(key):
        return pd.DataFrame()
    out = pd.read_json(StringIO(payload[key]), orient="split")
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def aggregate_live_scenario_payload(outcome, scenario_meta: Mapping | None = None) -> dict[str, Any]:
    """Compact browser payload for a live component-tax shock propagated to HICP Energy."""
    dates = pd.DatetimeIndex(outcome.dates, name="date")
    baseline = np.asarray(outcome.baseline_yoy["yoy_paths"], dtype=float)
    scenario = np.asarray(outcome.scenario_yoy["yoy_paths"], dtype=float)
    impact = scenario - baseline
    fans = pd.concat([
        _quantile_frame(baseline, dates, name="baseline_yoy"),
        _quantile_frame(scenario, dates, name="scenario_yoy"),
        _quantile_frame(impact, dates, name="impact_yoy_pp"),
    ], ignore_index=True)

    components = [str(x) for x in outcome.baseline_yoy.get("components", [])]
    base_c = np.asarray(outcome.baseline_yoy["contribution_paths"], dtype=float)
    scen_c = np.asarray(outcome.scenario_yoy["contribution_paths"], dtype=float)
    delta_c = scen_c - base_c
    rows = []
    for j, component in enumerate(components):
        q = np.nanquantile(delta_c[:, :, j], (0.05, 0.16, 0.50, 0.84, 0.95), axis=0).T
        block = pd.DataFrame(q, index=dates, columns=("q05", "q16", "q50", "q84", "q95")).reset_index()
        block["component"] = component
        rows.append(block)
    contrib = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    meta = dict(scenario_meta or {})
    meta.update({
        "vintage": str(outcome.vintage),
        "forecast_name": str(outcome.forecast_name),
        "n_draws": int(outcome.n_aggregate_draws_effective),
        "scenario_active": bool(outcome.scenario_active),
        "max_additivity_error": float(outcome.max_additivity_error),
    })
    return {
        "fans_json": _json_frame(fans),
        "contrib_json": _json_frame(contrib),
        "meta": meta,
    }


def _live_slice(payload: Mapping | None, name: str) -> pd.DataFrame:
    fans = _read_json_frame(payload, "fans_json")
    if fans.empty:
        return fans
    return fans.loc[fans["name"].astype(str) == str(name)].sort_values("date")


def _scenario_start_items(payload: Mapping | None) -> list[tuple[str, pd.Timestamp]]:
    meta = dict((payload or {}).get("meta", {}) or {})
    raw = dict(meta.get("scenario_starts", {}) or {})
    items: list[tuple[str, pd.Timestamp]] = []
    if raw:
        grouped: dict[pd.Timestamp, list[str]] = {}
        for label, value in raw.items():
            stamp = pd.to_datetime(value, errors="coerce")
            if pd.isna(stamp):
                continue
            grouped.setdefault(pd.Timestamp(stamp), []).append(str(label))
        for stamp, labels in sorted(grouped.items(), key=lambda item: item[0]):
            items.append((" + ".join(labels), stamp))
        return items
    start = pd.to_datetime(meta.get("scenario_start"), errors="coerce")
    if not pd.isna(start):
        items.append(("Scenario", pd.Timestamp(start)))
    return items


def _add_scenario_start_lines(fig: go.Figure, payload: Mapping | None, *, annotate: bool = False) -> None:
    for i, (label, stamp) in enumerate(_scenario_start_items(payload)):
        fig.add_shape(
            type="line", x0=stamp, x1=stamp, y0=0, y1=1,
            xref="x", yref="paper",
            line={"color": _SCENARIO, "width": 1, "dash": "dot"},
        )
        if annotate:
            fig.add_annotation(
                x=stamp, y=1, xref="x", yref="paper",
                text=label, showarrow=False, xanchor="left", yanchor="bottom",
                font={"size": 9, "color": _SCENARIO}, yshift=3 + 12 * i,
            )


def aggregate_live_scenario_figure(
    payload: Mapping | None,
    history: pd.DataFrame,
    *, fan_mode: str = "68", forecast_origin=None, uirevision: str = "aggregate-live-scenario",
) -> go.Figure:
    baseline = _live_slice(payload, "baseline_yoy")
    scenario = _live_slice(payload, "scenario_yoy")
    if baseline.empty or scenario.empty:
        return empty_aggregate_figure("No active component tax scenario")
    hist = history.copy() if history is not None else pd.DataFrame()
    if not hist.empty:
        hist = hist.loc[(hist["record_type"].astype(str) == "history") & (hist["metric"].astype(str) == "yoy") & (hist["series"].astype(str) == _AGG_SERIES)].dropna(subset=["value"]).sort_values("date")
    last_obs = hist["date"].max() if not hist.empty else pd.NaT
    origin = pd.to_datetime(forecast_origin, errors="coerce")
    nowcast = baseline.loc[(baseline["date"] > last_obs) & (baseline["date"] < origin)].copy() if not pd.isna(last_obs) and not pd.isna(origin) else baseline.iloc[0:0].copy()
    base_fc = baseline.loc[baseline["date"] >= origin].copy() if not pd.isna(origin) else baseline.copy()
    scen_fc = scenario.loc[scenario["date"] >= origin].copy() if not pd.isna(origin) else scenario.copy()

    fig = go.Figure()
    if fan_mode in {"90", "both"}:
        _band(fig, nowcast, "q05", "q95", "Nowcast 90% interval", 0.09, color=_NOWCAST)
        _band(fig, base_fc, "q05", "q95", "Baseline 90% interval", 0.08, color=_ACCENT)
        _band(fig, scen_fc, "q05", "q95", "Scenario 90% interval", 0.08, color=_SCENARIO)
    if fan_mode in {"68", "both"}:
        _band(fig, nowcast, "q16", "q84", "Nowcast 68% interval", 0.18, color=_NOWCAST)
        _band(fig, base_fc, "q16", "q84", "Baseline 68% interval", 0.17, color=_ACCENT)
        _band(fig, scen_fc, "q16", "q84", "Scenario 68% interval", 0.17, color=_SCENARIO)
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["value"], mode="lines", name="Observed HICP Energy", line={"color": _INK, "width": 1.8}))
    anchor_date = hist["date"].iloc[-1] if not hist.empty else None
    anchor_value = hist["value"].iloc[-1] if not hist.empty else None
    now_line = _append_anchor(nowcast, date=anchor_date, value=anchor_value)
    if not now_line.empty:
        fig.add_trace(go.Scatter(x=now_line["date"], y=now_line["q50"], mode="lines", name="Nowcast median", line={"color": _NOWCAST, "width": 2.2}))
        anchor_date, anchor_value = now_line["date"].iloc[-1], now_line["q50"].iloc[-1]
    base_line = _append_anchor(base_fc, date=anchor_date, value=anchor_value)
    scen_line = _append_anchor(scen_fc, date=anchor_date, value=anchor_value)
    fig.add_trace(go.Scatter(x=base_line["date"], y=base_line["q50"], mode="lines", name="Baseline forecast", line={"color": _ACCENT, "width": 2.2}))
    fig.add_trace(go.Scatter(x=scen_line["date"], y=scen_line["q50"], mode="lines", name="Tax-scenario forecast", line={"color": _SCENARIO, "width": 2.2, "dash": "dash"}))
    if not pd.isna(last_obs) and not nowcast.empty:
        fig.add_vline(x=last_obs, line={"color": _MUTED, "width": 1, "dash": "dash"})
    if not pd.isna(origin):
        fig.add_vline(x=origin, line={"color": _MUTED, "width": 1, "dash": "dot"})
    _add_scenario_start_lines(fig, payload, annotate=True)
    fig.add_hline(y=0.0, line={"color": "#d1d5db", "width": 1})
    return _layout(fig, title="HICP Energy: baseline vs combined tax scenario", unit="% y/y", uirevision=uirevision)


def aggregate_live_impact_figure(payload: Mapping | None, *, fan_mode: str = "68", uirevision: str = "aggregate-live-impact") -> go.Figure:
    block = _live_slice(payload, "impact_yoy_pp")
    if block.empty:
        return empty_aggregate_figure("No active component tax scenario")
    fig = go.Figure()
    if fan_mode in {"90", "both"}:
        _band(fig, block, "q05", "q95", "90% posterior interval", 0.10, color=_SCENARIO)
    if fan_mode in {"68", "both"}:
        _band(fig, block, "q16", "q84", "68% posterior interval", 0.20, color=_SCENARIO)
    fig.add_trace(go.Scatter(x=block["date"], y=block["q50"], mode="lines", name="Median impact", line={"color": _SCENARIO, "width": 2.3}))
    fig.add_hline(y=0.0, line={"color": "#d1d5db", "width": 1})
    _add_scenario_start_lines(fig, payload)
    return _layout(fig, title="Combined tax-scenario impact on HICP Energy inflation", unit="percentage points", uirevision=uirevision, height=390)


def aggregate_live_contribution_impact_figure(payload: Mapping | None, *, uirevision: str = "aggregate-live-contrib") -> go.Figure:
    frame = _read_json_frame(payload, "contrib_json")
    if frame.empty:
        return empty_aggregate_figure("No contribution impact is available")
    fig = go.Figure()
    for component in COMPONENT_LABELS:
        block = frame.loc[frame["component"].astype(str) == component].sort_values("date")
        if block.empty:
            continue
        fig.add_trace(go.Bar(
            x=block["date"], y=block["q50"],
            name=COMPONENT_LABELS.get(component, component),
            marker_color=COMPONENT_COLOURS.get(component, _MUTED),
            hovertemplate="%{y:+.3f} pp<extra>" + COMPONENT_LABELS.get(component, component) + "</extra>",
        ))
    fig.update_layout(barmode="relative")
    fig.add_hline(y=0.0, line={"color": "#d1d5db", "width": 1})
    _add_scenario_start_lines(fig, payload)
    return _layout(fig, title="Change in component contributions caused by the combined tax scenario", unit="percentage points", uirevision=uirevision, height=420)


def aggregate_live_scenario_kpis(payload: Mapping | None) -> dict[str, Any]:
    baseline = _live_slice(payload, "baseline_yoy")
    scenario = _live_slice(payload, "scenario_yoy")
    impact = _live_slice(payload, "impact_yoy_pp")
    if baseline.empty or scenario.empty or impact.empty:
        return {"baseline": None, "scenario": None, "impact": None, "low": None, "high": None, "date": None, "n_draws": None}
    return {
        "baseline": float(baseline["q50"].iloc[-1]),
        "scenario": float(scenario["q50"].iloc[-1]),
        "impact": float(impact["q50"].iloc[-1]),
        "low": float(impact["q16"].iloc[-1]),
        "high": float(impact["q84"].iloc[-1]),
        "date": pd.Timestamp(impact["date"].iloc[-1]),
        "n_draws": (payload or {}).get("meta", {}).get("n_draws"),
    }


def aggregate_diagnostics_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Diagnostics and provenance rows, ready for a DataTable."""
    if frame.empty:
        return pd.DataFrame(columns=["Item", "Value"])
    rows: list[dict[str, Any]] = []

    diagnostics = frame.loc[frame["record_type"].astype(str) == "diagnostic"]
    for _, row in diagnostics.iterrows():
        label = str(row.get("metric", "")).replace("_", " ").strip().capitalize()
        value = row.get("value")
        rows.append(
            {"Item": label, "Value": "—" if pd.isna(value) else f"{float(value):,.6f}"}
        )

    keep = {
        "aggregate_run_id": "Aggregate run",
        "n_aggregate_draws_effective": "Effective draws",
        "weekly_tax_mode_effective": "Weekly tax mode",
        "forecast_name": "Forecast contract",
        "predictive_distribution_interpretation": "Fan interpretation",
        "cross_model_dependence": "Cross-model dependence",
        "other_transport_fuels_forecast": "Other transport fuels",
        "weight_policy": "Weight policy",
    }
    meta = frame.loc[frame["record_type"].astype(str) == "meta"]
    for key, label in keep.items():
        block = meta.loc[meta["key"].astype(str) == key, "text_value"]
        if not block.empty and pd.notna(block.iloc[0]):
            rows.append({"Item": label, "Value": str(block.iloc[0])})
    return pd.DataFrame(rows, columns=["Item", "Value"])
