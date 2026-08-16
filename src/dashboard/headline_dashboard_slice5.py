"""Headline Dashboard — Slice 5.

Additive UI/callback module for the unified Inflation Dashboard.
"""
from __future__ import annotations

import inspect
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from dashboard_snapshot_cache import frame_from_store as snapshot_frame_from_store, get_or_build as snapshot_get_or_build
from energy_bvar_model import BVARSVOPriorConfig, SamplerConfig
from energy_bvar_theme import (
    TOKENS,
    INFLATION_COLORS,
    apply_inflation_figure_style,
    fan_traces,
    forecast_origin_marker,
    graph_config,
)
from inflation_chart_contract import (
    statistic_column,
    short_horizon_labels,
    apply_short_horizon_axis,
    displayed_mean_additivity_error,
)
from inflation_table_contract import (
    FORECAST_COLUMNS,
    forecast_summary_records,
    readable_table,
)
from headline_bvar_display import build_headline_display
from headline_bvar_fitted import materialize_headline_fitted
from headline_dashboard_slice6 import overlay_headline_conditional_forecast
from headline_bvar_pipeline import (
    MAX_PUBLISHED_HORIZON_MONTHS,
    model_contract,
    production_prior_config,
    production_sampler_config,
    run_headline,
)

HORIZON_OPTIONS = (3, 6, 12)
DEFAULT_DISPLAY_HORIZON = 3
SERIES_ORDER = (
    "hicp_total",
    "hicp_energy",
    "hicp_food",
    "hicp_neig",
    "hicp_services",
)
SERIES_LABELS = {
    "hicp_total": "Headline HICP",
    "hicp_core": "Core HICP",
    "hicp_energy": "Energy",
    "hicp_food": "Food",
    "hicp_neig": "NEIG",
    "hicp_services": "Services",
}
COMPONENTS = ("hicp_energy","hicp_food","hicp_neig","hicp_services")
CORE_SERIES = "hicp_core"
CORE_COMPONENTS = ("hicp_neig", "hicp_services")
CORE_LONG_LABEL = "Core HICP excl. energy, food, alcohol & tobacco"


def _frame_from_store(store):
    return snapshot_frame_from_store(store)


def _select(frame, *, record_type, series=None, metric=None):
    if frame.empty or "record_type" not in frame:
        return pd.DataFrame()
    mask = frame["record_type"].astype(str).eq(record_type)
    if series is not None and "series" in frame:
        mask &= frame["series"].astype(str).eq(series)
    if metric is not None and "metric" in frame:
        mask &= frame["metric"].astype(str).eq(metric)
    out = frame.loc[mask].copy()
    return out.sort_values("date") if "date" in out else out


def _future(frame):
    if frame.empty:
        return frame
    if "is_future" in frame:
        flag = frame["is_future"].fillna(False).astype(bool)
        return frame.loc[flag].copy()
    return frame.loc[frame.get("segment", pd.Series(index=frame.index)).astype(str).eq("forecast")].copy()


def _nowcast(frame):
    """Return ragged-edge model paths that are neither observed nor future.

    ``display_v1`` marks these rows explicitly as ``segment='nowcast'`` and
    ``is_future=False``.  They must not be silently discarded by forecast plots.
    """
    if frame.empty:
        return frame
    if "segment" not in frame:
        return frame.iloc[0:0].copy()
    return frame.loc[frame["segment"].astype(str).eq("nowcast")].copy()


def _fmt(value, suffix=""):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(x) else f"{x:.2f}{suffix}"


def _figure_base(fig, *, uirevision, y_title=None, height=430):
    """Shared Headline chart geometry; panel headings own the visible title."""
    return apply_inflation_figure_style(
        fig,
        uirevision=uirevision,
        height=height,
        y_title=y_title,
    )


def _empty(message, height=430):
    fig = go.Figure()
    fig.add_annotation(x=.5,y=.5,xref="paper",yref="paper",text=message,showarrow=False,
                       font=dict(size=13,color=TOKENS["muted"]))
    _figure_base(fig, uirevision="h5-empty", height=height)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig


def _add_fan(
    fig,
    fan,
    mode="68",
    name="Baseline",
    *,
    statistic="mean",
    color=None,
):
    """Add quantile bands plus an explicitly labelled central statistic."""
    if fan.empty:
        return
    bands = ("90", "68") if mode == "both" else (str(mode),)
    quantiles = {
        key: fan[key].to_numpy(dtype=float)
        for key in ("q05", "q16", "q50", "q84", "q95")
        if key in fan
    }
    colour = color or INFLATION_COLORS["baseline"]
    for trace in fan_traces(
        fan["date"],
        quantiles,
        bands=bands,
        color=colour,
        name=name,
        show_median=False,
    ):
        fig.add_trace(trace)

    col = statistic_column(fan, statistic)
    label = "Posterior mean" if statistic == "mean" else "Posterior median"
    fig.add_trace(
        go.Scatter(
            x=fan["date"],
            y=fan[col],
            mode="lines",
            name=f"{name} · {label.lower()}",
            line=dict(color=colour, width=2.2),
            hovertemplate=f"%{{y:.2f}}<extra>{label}</extra>",
        )
    )


def headline_forecast_figure(
    frame,
    *,
    series,
    metric,
    horizon,
    fan_mode,
    statistic="mean",
    history_months=None,
    include_fitted=True,
):
    """Analytical forecast figure with posterior mean as the central path."""
    hist = _select(frame, record_type="history", series=series, metric=metric)
    fitted = _select(frame, record_type="fitted", series=series, metric=metric)
    all_fan = _select(frame, record_type="fan", series=series, metric=metric)
    nowcast = _nowcast(all_fan)
    fan = _future(all_fan).head(int(horizon))
    if hist.empty and nowcast.empty and fan.empty:
        return _empty("No Headline forecast display is available.")

    fig = go.Figure()
    if not hist.empty:
        h = (
            hist
            if history_months is None
            else hist.tail(int(history_months))
        )
        fig.add_trace(
            go.Scatter(
                x=h["date"],
                y=h["value"],
                mode="lines",
                name="Observed",
                line=dict(width=2, color=INFLATION_COLORS["observed"]),
            )
        )

    path_dates = (
        pd.DatetimeIndex(all_fan["date"].dropna())
        if not all_fan.empty
        else pd.DatetimeIndex([])
    )
    origin = (
        path_dates.min() - pd.DateOffset(months=1)
        if len(path_dates)
        else (hist["date"].max() if not hist.empty else None)
    )

    if include_fitted and not fitted.empty:
        # Show every saved fitted observation. The fitted path can naturally
        # start later than observed history because of lag initialisation, but
        # there is no dashboard-imposed historical cutoff.
        f = fitted.sort_values("date").copy()
        fitted_col = statistic_column(f, "mean")
        fig.add_trace(
            go.Scatter(
                x=f["date"],
                y=f[fitted_col],
                mode="lines",
                name="BVAR fitted · posterior mean (1-step)",
                line=dict(width=1.5, dash="dot", color=INFLATION_COLORS["fitted"]),
            )
        )

    if not nowcast.empty:
        nowcast_col = statistic_column(nowcast, statistic)
        nowcast_label = "mean" if statistic == "mean" else "median"
        fig.add_trace(
            go.Scatter(
                x=nowcast["date"],
                y=nowcast[nowcast_col],
                mode="lines+markers",
                name=f"Nowcast · posterior {nowcast_label}",
                line=dict(width=1.8, dash="dash", color=INFLATION_COLORS["nowcast"]),
                marker=dict(size=5),
            )
        )

    _add_fan(
        fig,
        fan,
        fan_mode,
        name="Baseline",
        statistic=statistic,
        color=INFLATION_COLORS["baseline"],
    )
    if origin is not None:
        forecast_origin_marker(fig, pd.Timestamp(origin), label="Forecast origin")

    unit = None
    source = hist if not hist.empty else (nowcast if not nowcast.empty else fan)
    if not source.empty and "unit" in source and source["unit"].notna().any():
        unit = str(source["unit"].dropna().iloc[0])
    return _figure_base(
        fig,
        uirevision=f"h5::{series}::{metric}::{horizon}::{statistic}",
        y_title=unit,
        height=500,
    )


def headline_overview_figure(frame, horizon=3):
    """Executive view: recent history + posterior mean + 68% band, no fitted."""
    return headline_forecast_figure(
        frame,
        series="hicp_total",
        metric="yoy",
        horizon=int(horizon),
        fan_mode="68",
        statistic="mean",
        history_months=None,
        include_fitted=False,
    )


def _headline_mean_by_dates(frame, dates):
    fan = _select(frame, record_type="fan", series="hicp_total", metric="yoy")
    if fan.empty:
        return pd.Series(dtype=float)
    fan = fan.loc[fan["date"].isin(pd.DatetimeIndex(dates))].sort_values("date")
    col = statistic_column(fan, "mean")
    return pd.Series(
        fan[col].to_numpy(dtype=float),
        index=pd.DatetimeIndex(fan["date"]),
    )


def displayed_contribution_additivity_error(frame, horizon=3):
    return displayed_mean_additivity_error(
        frame,
        components=COMPONENTS,
        headline_series="hicp_total",
        contribution_metric="yoy_contribution",
        headline_metric="yoy",
        horizon=int(horizon),
    )


def _displayed_model_contribution_additivity_error(frame, dates):
    """Mean additivity on the exact nowcast+forecast dates shown in the chart."""
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        return np.nan

    rows = _select(
        frame,
        record_type="contribution",
        metric="yoy_contribution",
    )
    rows = rows.loc[
        rows["series"].isin(COMPONENTS)
        & rows["date"].isin(dates)
    ].copy()
    if rows.empty:
        return np.nan

    col = statistic_column(rows, "mean")
    pivot = (
        rows.pivot_table(
            index="date",
            columns="series",
            values=col,
            aggfunc="first",
        )
        .reindex(columns=list(COMPONENTS))
        .dropna(how="any")
    )
    if pivot.empty:
        return np.nan

    headline = _headline_mean_by_dates(frame, pivot.index).reindex(pivot.index)
    comparable = headline.notna()
    if not comparable.any():
        return np.nan

    return float(
        np.max(
            np.abs(
                pivot.loc[comparable].sum(axis=1).to_numpy(dtype=float)
                - headline.loc[comparable].to_numpy(dtype=float)
            )
        )
    )


def contribution_snapshot_figure(frame, horizon=3):
    rows = _future(
        _select(frame, record_type="contribution", metric="yoy_contribution")
    )
    rows = rows.loc[rows["series"].isin(COMPONENTS)] if not rows.empty else rows
    dates = sorted(pd.DatetimeIndex(rows["date"].dropna().unique())) if not rows.empty else []
    if not dates:
        return _empty("Contribution paths are not available.", 330)
    target = dates[min(int(horizon), len(dates)) - 1]
    snap = rows.loc[rows["date"] == target]
    col = statistic_column(snap, "mean")

    labels = []
    values = []
    colours = []
    for j, name in enumerate(COMPONENTS):
        r = snap.loc[snap["series"] == name]
        if r.empty:
            continue
        labels.append(SERIES_LABELS[name])
        values.append(float(r.iloc[0][col]))
        colours.append(INFLATION_COLORS[name])

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colours,
            text=[f"{v:+.2f} pp" for v in values],
            textposition="outside",
            hovertemplate="%{x:+.2f} pp<extra>%{y}</extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=TOKENS["hairline"])
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)))
    return _figure_base(
        fig,
        uirevision=f"h5-snap::{target}",
        y_title=None,
        height=330,
    )


def components_figure(frame, horizon=3):
    fig = go.Figure()
    found = False
    for j, name in enumerate(COMPONENTS):
        hist = _select(frame, record_type="history", series=name, metric="yoy")
        fan = _future(_select(frame, record_type="fan", series=name, metric="yoy")).head(int(horizon))
        if hist.empty and fan.empty:
            continue
        found = True
        if not hist.empty:
            h = hist.sort_values("date")
            fig.add_trace(
                go.Scatter(
                    x=h["date"],
                    y=h["value"],
                    mode="lines",
                    name=f"{SERIES_LABELS[name]} observed",
                    legendgroup=name,
                    showlegend=False,
                    line=dict(width=1, color=INFLATION_COLORS[name]),
                    opacity=0.45,
                )
            )
        if not fan.empty:
            col = statistic_column(fan, "mean")
            fig.add_trace(
                go.Scatter(
                    x=fan["date"],
                    y=fan[col],
                    mode="lines+markers",
                    name=f"{SERIES_LABELS[name]} · mean",
                    legendgroup=name,
                    line=dict(color=INFLATION_COLORS[name], width=2),
                )
            )
    return (
        _figure_base(
            fig,
            uirevision=f"h5-components::{horizon}",
            y_title="% y/y",
            height=450,
        )
        if found
        else _empty("Component paths are not available.")
    )


def historical_contribution_additivity_error(frame):
    rows = _select(
        frame,
        record_type="contribution_history",
        metric="yoy_contribution",
    )
    rows = rows.loc[rows["series"].isin(COMPONENTS)] if not rows.empty else rows
    headline = _select(
        frame,
        record_type="history",
        series="hicp_total",
        metric="yoy",
    )
    if rows.empty or headline.empty:
        return np.nan

    pivot = (
        rows.pivot_table(
            index="date",
            columns="series",
            values="value",
            aggfunc="first",
        )
        .reindex(columns=list(COMPONENTS))
        .dropna(how="any")
    )
    if pivot.empty:
        return np.nan

    target = (
        headline.set_index("date")["value"]
        .reindex(pivot.index)
        .astype(float)
    )
    comparable = target.notna()
    if not comparable.any():
        return np.nan
    return float(
        np.max(
            np.abs(
                pivot.loc[comparable].sum(axis=1).to_numpy(dtype=float)
                - target.loc[comparable].to_numpy(dtype=float)
            )
        )
    )


def contribution_decomposition_figure(
    frame,
    horizon=3,
    *,
    display_mode="bars",
    show_latest=False,
):
    """Observed history + saved model-path contribution decomposition.

    History uses exact contributions to official Headline year-on-year
    inflation. The model path uses saved posterior-mean contribution rows:
    all available nowcast months plus the requested number of future months.
    """
    historical = _select(
        frame,
        record_type="contribution_history",
        metric="yoy_contribution",
    )
    historical = (
        historical.loc[historical["series"].isin(COMPONENTS)].copy()
        if not historical.empty
        else historical
    )

    all_model = _select(
        frame,
        record_type="contribution",
        metric="yoy_contribution",
    )
    all_model = (
        all_model.loc[all_model["series"].isin(COMPONENTS)].copy()
        if not all_model.empty
        else all_model
    )
    nowcast = _nowcast(all_model)
    future = _future(all_model)

    future_dates = (
        sorted(pd.DatetimeIndex(future["date"].dropna().unique()))[: int(horizon)]
        if not future.empty
        else []
    )
    if not future.empty:
        future = future.loc[future["date"].isin(future_dates)].copy()

    model = pd.concat([nowcast, future], ignore_index=True, sort=False)
    if not model.empty:
        model = model.sort_values(["date", "series"]).reset_index(drop=True)
    model_dates = (
        sorted(pd.DatetimeIndex(model["date"].dropna().unique()))
        if not model.empty
        else []
    )

    # One owner per date: exact history before the saved model path, then
    # nowcast + requested forecast months from the posterior display rows.
    if model_dates and not historical.empty:
        model_start = pd.Timestamp(model_dates[0])
        historical = historical.loc[historical["date"] < model_start].copy()

    if historical.empty and model.empty:
        return _empty("Exact Headline contribution decomposition is not available.")

    hist_error = historical_contribution_additivity_error(frame)
    if not historical.empty and (
        not np.isfinite(hist_error) or hist_error > 1e-10
    ):
        return _empty(
            "Historical contribution additivity guard failed "
            f"(max error {hist_error:.3e})."
        )

    if model_dates:
        model_error = _displayed_model_contribution_additivity_error(
            frame,
            model_dates,
        )
        if not np.isfinite(model_error) or model_error > 1e-10:
            return _empty(
                "Model-path contribution additivity guard failed "
                f"(max error {model_error:.3e})."
            )

    mode = str(display_mode or "bars").lower()
    if mode not in {"bars", "lines"}:
        mode = "bars"

    fig = go.Figure()
    model_col = statistic_column(model, "mean") if not model.empty else None

    for name in COMPONENTS:
        hist_block = (
            historical.loc[historical["series"] == name]
            .sort_values("date")
        )
        model_block = (
            model.loc[model["series"] == name]
            .sort_values("date")
        )

        x_values = []
        y_values = []
        if not hist_block.empty:
            x_values.extend(pd.DatetimeIndex(hist_block["date"]).tolist())
            y_values.extend(
                pd.to_numeric(hist_block["value"], errors="coerce")
                .to_numpy(dtype=float)
                .tolist()
            )
        if not model_block.empty:
            x_values.extend(pd.DatetimeIndex(model_block["date"]).tolist())
            y_values.extend(
                pd.to_numeric(model_block[model_col], errors="coerce")
                .to_numpy(dtype=float)
                .tolist()
            )

        if not x_values:
            continue

        if mode == "bars":
            text_values = [""] * len(y_values)
            if show_latest and y_values:
                text_values[-1] = f"{y_values[-1]:+.2f}"
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=y_values,
                    name=SERIES_LABELS[name],
                    marker_color=INFLATION_COLORS[name],
                    text=text_values if show_latest else None,
                    textposition="auto" if show_latest else None,
                    hovertemplate=(
                        "%{x|%b %Y}<br>%{y:+.2f} pp"
                        "<extra>" + SERIES_LABELS[name] + "</extra>"
                    ),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=SERIES_LABELS[name],
                    line=dict(
                        color=INFLATION_COLORS[name],
                        width=2,
                    ),
                    hovertemplate=(
                        "%{x|%b %Y}<br>%{y:+.2f} pp"
                        "<extra>" + SERIES_LABELS[name] + "</extra>"
                    ),
                )
            )
            if show_latest and y_values:
                fig.add_trace(
                    go.Scatter(
                        x=[x_values[-1]],
                        y=[y_values[-1]],
                        mode="markers+text",
                        marker=dict(
                            color=INFLATION_COLORS[name],
                            size=6,
                        ),
                        text=[f"{SERIES_LABELS[name]} {y_values[-1]:+.2f}"],
                        textposition="middle right",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    headline_history = _select(
        frame,
        record_type="history",
        series="hicp_total",
        metric="yoy",
    )
    if not historical.empty and not headline_history.empty:
        first_hist = pd.Timestamp(historical["date"].min())
        headline_history = headline_history.loc[
            headline_history["date"] >= first_hist
        ].copy()
    if model_dates and not headline_history.empty:
        headline_history = headline_history.loc[
            headline_history["date"] < pd.Timestamp(model_dates[0])
        ].copy()

    if not headline_history.empty:
        fig.add_trace(
            go.Scatter(
                x=headline_history["date"],
                y=headline_history["value"],
                mode="lines",
                name="Headline HICP",
                line=dict(
                    color=INFLATION_COLORS["headline"],
                    width=2,
                ),
                hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra>Headline HICP</extra>",
            )
        )

    headline_model = _headline_mean_by_dates(frame, model_dates)
    if len(headline_model):
        fig.add_trace(
            go.Scatter(
                x=headline_model.index,
                y=headline_model.to_numpy(dtype=float),
                mode="lines",
                name="Headline model path · posterior mean",
                line=dict(
                    color=INFLATION_COLORS["headline"],
                    width=2,
                    dash="dash",
                ),
                showlegend=False,
                hovertemplate=(
                    "%{x|%b %Y}<br>%{y:.2f}%"
                    "<extra>Headline model-path mean</extra>"
                ),
            )
        )

    if mode == "bars":
        fig.update_layout(barmode="relative")

    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])

    if future_dates:
        forecast_origin_marker(
            fig,
            pd.Timestamp(future_dates[0]),
            label="Forecast origin",
        )

    if show_latest and mode == "lines":
        fig.update_layout(margin=dict(r=145))

    return _figure_base(
        fig,
        uirevision=(
            f"h5-contrib-history-model::{horizon}::{mode}::{bool(show_latest)}"
        ),
        y_title="percentage points",
        height=520,
    )



def _core_mean_by_dates(frame, dates):
    fan = _select(frame, record_type="fan", series=CORE_SERIES, metric="yoy")
    if fan.empty:
        return pd.Series(dtype=float)
    fan = fan.loc[fan["date"].isin(pd.DatetimeIndex(dates))].sort_values("date")
    if fan.empty:
        return pd.Series(dtype=float)
    col = statistic_column(fan, "mean")
    return pd.Series(
        fan[col].to_numpy(dtype=float),
        index=pd.DatetimeIndex(fan["date"]),
    )


def _core_materialization_error(frame):
    rows = _select(frame, record_type="core_status", series=CORE_SERIES)
    if rows.empty or "text_value" not in rows:
        return None
    values = rows["text_value"].dropna().astype(str)
    return None if values.empty else values.iloc[-1]


def _core_overview_values(frame):
    hist = _select(
        frame,
        record_type="history",
        series=CORE_SERIES,
        metric="yoy",
    )
    future = _future(
        _select(frame, record_type="fan", series=CORE_SERIES, metric="yoy")
    )
    out = []
    if hist.empty:
        out.append(("—", "Official Core unavailable"))
    else:
        row = hist.iloc[-1]
        out.append(
            (
                _fmt(row["value"], "%"),
                pd.Timestamp(row["date"]).strftime("%Y-%m"),
            )
        )

    col = statistic_column(future, "mean") if not future.empty else "value"
    for horizon in (1, 3, 12):
        if len(future) < horizon:
            out.append(("—", f"{horizon}m unavailable"))
        else:
            row = future.iloc[horizon - 1]
            out.append(
                (
                    _fmt(row[col], "%"),
                    pd.Timestamp(row["date"]).strftime("%Y-%m"),
                )
            )
    return out


def _core_validation_summary(frame):
    error = _core_materialization_error(frame)
    if error:
        return html.Div(
            [html.Strong("Core unavailable: "), html.Span(error)],
            className="banner-error",
        )

    rows = _select(
        frame,
        record_type="core_diagnostic",
        series=CORE_SERIES,
    )
    if rows.empty:
        return html.Div(
            "Core display has not been materialised for this saved run.",
            className="selection-banner",
        )

    by_key = {
        str(row.get("key")): row
        for _, row in rows.iterrows()
        if pd.notna(row.get("key"))
    }

    def number(key):
        row = by_key.get(key)
        if row is None:
            return np.nan
        return pd.to_numeric(
            pd.Series([row.get("extra_value")]),
            errors="coerce",
        ).iloc[0]

    mae = number("core_reconstruction_yoy_mae_pp")
    maximum = number("core_reconstruction_yoy_max_abs_error_pp")
    level_add = number("core_drawwise_level_additivity_max_abs_error")
    yoy_add = number("core_yoy_contribution_additivity_max_abs_error")

    bits = [
        html.Strong("Official Core validation · Eurostat TOT_X_NRG_FOOD"),
    ]
    if np.isfinite(mae):
        bits.append(html.Span(f" · YoY MAE {mae:.4f} pp"))
    if np.isfinite(maximum):
        bits.append(html.Span(f" · max {maximum:.4f} pp"))
    if np.isfinite(level_add):
        bits.append(html.Span(f" · level additivity {level_add:.2e}"))
    if np.isfinite(yoy_add):
        bits.append(html.Span(f" · YoY additivity {yoy_add:.2e} pp"))
    return html.Div(bits, className="selection-banner")


def headline_core_comparison_figure(frame, horizon=3):
    """Official Headline/Core history plus posterior-mean model paths."""
    headline_hist = _select(
        frame, record_type="history", series="hicp_total", metric="yoy"
    )
    core_hist = _select(
        frame, record_type="history", series=CORE_SERIES, metric="yoy"
    )
    headline_all = _select(
        frame, record_type="fan", series="hicp_total", metric="yoy"
    )
    core_all = _select(
        frame, record_type="fan", series=CORE_SERIES, metric="yoy"
    )
    if core_hist.empty and core_all.empty:
        error = _core_materialization_error(frame)
        return _empty(
            "Core is unavailable." if not error else f"Core unavailable: {error}",
            450,
        )

    core_nowcast = _nowcast(core_all)
    core_future = _future(core_all).head(int(horizon))
    model_dates = pd.DatetimeIndex(
        pd.concat([core_nowcast, core_future], ignore_index=True)["date"].dropna()
    )
    if len(model_dates):
        model_start = model_dates.min()
        headline_hist = headline_hist.loc[headline_hist["date"] < model_start]
        core_hist = core_hist.loc[core_hist["date"] < model_start]

    fig = go.Figure()
    if not headline_hist.empty:
        fig.add_trace(
            go.Scatter(
                x=headline_hist["date"],
                y=headline_hist["value"],
                mode="lines",
                name="Headline observed",
                line=dict(color=INFLATION_COLORS["headline"], width=1.8),
            )
        )
    if not core_hist.empty:
        fig.add_trace(
            go.Scatter(
                x=core_hist["date"],
                y=core_hist["value"],
                mode="lines",
                name="Core observed",
                line=dict(color=TOKENS["navy_deep"], width=1.8),
            )
        )

    if len(model_dates):
        headline_mean = _headline_mean_by_dates(frame, model_dates)
        core_mean = _core_mean_by_dates(frame, model_dates)
        if len(headline_mean):
            fig.add_trace(
                go.Scatter(
                    x=headline_mean.index,
                    y=headline_mean.to_numpy(dtype=float),
                    mode="lines+markers",
                    name="Headline · posterior mean",
                    line=dict(
                        color=INFLATION_COLORS["headline"],
                        width=2.2,
                        dash="dash",
                    ),
                    marker=dict(size=4),
                )
            )
        if len(core_mean):
            fig.add_trace(
                go.Scatter(
                    x=core_mean.index,
                    y=core_mean.to_numpy(dtype=float),
                    mode="lines+markers",
                    name="Core · posterior mean",
                    line=dict(
                        color=TOKENS["navy_deep"],
                        width=2.2,
                        dash="dash",
                    ),
                    marker=dict(size=4),
                )
            )
        future_dates = pd.DatetimeIndex(core_future["date"].dropna())
        if len(future_dates):
            forecast_origin_marker(
                fig,
                pd.Timestamp(future_dates.min()),
                label="Forecast origin",
            )

    return _figure_base(
        fig,
        uirevision=f"h5-headline-core::{int(horizon)}",
        y_title="% y/y",
        height=450,
    )


def core_historical_contribution_additivity_error(frame):
    rows = _select(
        frame,
        record_type="core_contribution_history",
        metric="core_yoy_contribution",
    )
    rows = (
        rows.loc[rows["series"].isin(CORE_COMPONENTS)].copy()
        if not rows.empty
        else rows
    )
    core = _select(
        frame,
        record_type="history",
        series=CORE_SERIES,
        metric="yoy",
    )
    if rows.empty or core.empty:
        return np.nan
    pivot = (
        rows.pivot_table(
            index="date",
            columns="series",
            values="value",
            aggfunc="first",
        )
        .reindex(columns=list(CORE_COMPONENTS))
        .dropna(how="any")
    )
    if pivot.empty:
        return np.nan
    target = core.set_index("date")["value"].reindex(pivot.index)
    comparable = target.notna()
    if not comparable.any():
        return np.nan
    return float(
        np.max(
            np.abs(
                pivot.loc[comparable].sum(axis=1).to_numpy(dtype=float)
                - target.loc[comparable].to_numpy(dtype=float)
            )
        )
    )


def _core_model_contribution_additivity_error(frame, dates):
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        return np.nan
    rows = _select(
        frame,
        record_type="core_contribution",
        metric="core_yoy_contribution",
    )
    rows = rows.loc[
        rows["series"].isin(CORE_COMPONENTS)
        & rows["date"].isin(dates)
    ].copy()
    if rows.empty:
        return np.nan
    col = statistic_column(rows, "mean")
    pivot = (
        rows.pivot_table(
            index="date",
            columns="series",
            values=col,
            aggfunc="first",
        )
        .reindex(columns=list(CORE_COMPONENTS))
        .dropna(how="any")
    )
    if pivot.empty:
        return np.nan
    core = _core_mean_by_dates(frame, pivot.index).reindex(pivot.index)
    comparable = core.notna()
    if not comparable.any():
        return np.nan
    return float(
        np.max(
            np.abs(
                pivot.loc[comparable].sum(axis=1).to_numpy(dtype=float)
                - core.loc[comparable].to_numpy(dtype=float)
            )
        )
    )


def core_contribution_decomposition_figure(frame, horizon=3):
    """Exact NEIG + Services contributions to official/model Core YoY."""
    historical = _select(
        frame,
        record_type="core_contribution_history",
        metric="core_yoy_contribution",
    )
    historical = (
        historical.loc[historical["series"].isin(CORE_COMPONENTS)].copy()
        if not historical.empty
        else historical
    )

    all_model = _select(
        frame,
        record_type="core_contribution",
        metric="core_yoy_contribution",
    )
    all_model = (
        all_model.loc[all_model["series"].isin(CORE_COMPONENTS)].copy()
        if not all_model.empty
        else all_model
    )
    nowcast = _nowcast(all_model)
    future = _future(all_model)
    future_dates = (
        sorted(pd.DatetimeIndex(future["date"].dropna().unique()))[: int(horizon)]
        if not future.empty
        else []
    )
    if not future.empty:
        future = future.loc[future["date"].isin(future_dates)].copy()
    model = pd.concat([nowcast, future], ignore_index=True, sort=False)
    model_dates = (
        sorted(pd.DatetimeIndex(model["date"].dropna().unique()))
        if not model.empty
        else []
    )

    if model_dates and not historical.empty:
        historical = historical.loc[
            historical["date"] < pd.Timestamp(model_dates[0])
        ].copy()

    if historical.empty and model.empty:
        error = _core_materialization_error(frame)
        return _empty(
            "Exact Core contribution decomposition is unavailable."
            if not error
            else f"Core unavailable: {error}",
            500,
        )

    hist_error = core_historical_contribution_additivity_error(frame)
    if not historical.empty and (
        not np.isfinite(hist_error) or hist_error > 1e-10
    ):
        return _empty(
            "Historical Core contribution additivity guard failed "
            f"(max error {hist_error:.3e}).",
            500,
        )

    if model_dates:
        model_error = _core_model_contribution_additivity_error(
            frame, model_dates
        )
        if not np.isfinite(model_error) or model_error > 1e-10:
            return _empty(
                "Model Core contribution additivity guard failed "
                f"(max error {model_error:.3e}).",
                500,
            )

    fig = go.Figure()
    model_col = statistic_column(model, "mean") if not model.empty else None
    for name in CORE_COMPONENTS:
        hist_block = historical.loc[
            historical["series"] == name
        ].sort_values("date")
        model_block = model.loc[model["series"] == name].sort_values("date")

        x_values = []
        y_values = []
        if not hist_block.empty:
            x_values.extend(pd.DatetimeIndex(hist_block["date"]).tolist())
            y_values.extend(
                pd.to_numeric(
                    hist_block["value"], errors="coerce"
                ).to_numpy(dtype=float).tolist()
            )
        if not model_block.empty:
            x_values.extend(pd.DatetimeIndex(model_block["date"]).tolist())
            y_values.extend(
                pd.to_numeric(
                    model_block[model_col], errors="coerce"
                ).to_numpy(dtype=float).tolist()
            )
        if not x_values:
            continue
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=y_values,
                name=SERIES_LABELS[name],
                marker_color=INFLATION_COLORS[name],
                hovertemplate=(
                    "%{x|%b %Y}<br>%{y:+.2f} pp"
                    "<extra>" + SERIES_LABELS[name] + "</extra>"
                ),
            )
        )

    core_history = _select(
        frame,
        record_type="history",
        series=CORE_SERIES,
        metric="yoy",
    )
    if not historical.empty and not core_history.empty:
        core_history = core_history.loc[
            core_history["date"] >= pd.Timestamp(historical["date"].min())
        ].copy()
    if model_dates and not core_history.empty:
        core_history = core_history.loc[
            core_history["date"] < pd.Timestamp(model_dates[0])
        ].copy()
    if not core_history.empty:
        fig.add_trace(
            go.Scatter(
                x=core_history["date"],
                y=core_history["value"],
                mode="lines",
                name="Core HICP",
                line=dict(color=TOKENS["navy_deep"], width=2),
                hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra>Core HICP</extra>",
            )
        )

    core_model = _core_mean_by_dates(frame, model_dates)
    if len(core_model):
        fig.add_trace(
            go.Scatter(
                x=core_model.index,
                y=core_model.to_numpy(dtype=float),
                mode="lines",
                name="Core model path · posterior mean",
                line=dict(
                    color=TOKENS["navy_deep"],
                    width=2,
                    dash="dash",
                ),
                showlegend=False,
            )
        )

    fig.update_layout(barmode="relative")
    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    if future_dates:
        forecast_origin_marker(
            fig,
            pd.Timestamp(future_dates[0]),
            label="Forecast origin",
        )
    return _figure_base(
        fig,
        uirevision=f"h5-core-contrib::{int(horizon)}",
        y_title="percentage points",
        height=500,
    )


# Backward-compatible builder name used by older tests/scripts.
def contribution_stack_figure(frame, horizon=3):
    return contribution_decomposition_figure(
        frame,
        horizon=horizon,
        display_mode="bars",
        show_latest=False,
    )


def _header(title,subtitle):
    return [html.H2(title,className="page-title"),html.P(subtitle,className="page-subtitle")]


def _stat(title,value_id,date_id):
    return html.Div([html.Div(title,className="stat-title"),html.Div("—",id=value_id,className="stat-value"),html.Div("",id=date_id,className="stat-subtitle")],className="stat-card")


def headline_forecast_v2_page():
    controls = html.Div([
        html.Div([
            html.Label("Series", className="selector-label"),
            dcc.Dropdown(
                id="h5-forecast-series",
                options=[{"label": SERIES_LABELS[x], "value": x} for x in SERIES_ORDER],
                value="hicp_total",
                clearable=False,
            ),
        ], className="selector-block"),
        html.Div([
            html.Label("Metric", className="selector-label"),
            dcc.Dropdown(
                id="h5-forecast-metric",
                options=[
                    {"label": "Year-on-year", "value": "yoy"},
                    {"label": "HICP level", "value": "level"},
                ],
                value="yoy",
                clearable=False,
            ),
        ], className="selector-block"),
        html.Div([
            html.Label("Display horizon", className="selector-label"),
            dcc.Dropdown(
                id="h5-forecast-horizon",
                options=[{"label": f"{h} months", "value": h} for h in HORIZON_OPTIONS],
                value=DEFAULT_DISPLAY_HORIZON,
                clearable=False,
            ),
        ], className="selector-block"),
        html.Div([
            html.Label("Fan", className="selector-label"),
            dcc.Dropdown(
                id="h5-forecast-fan",
                options=[
                    {"label": "68%", "value": "68"},
                    {"label": "90%", "value": "90"},
                    {"label": "68% + 90%", "value": "both"},
                ],
                value="68",
                clearable=False,
            ),
        ], className="selector-block"),
        html.Div([
            html.Label("Energy drill-down", className="selector-label"),
            dcc.Link(
                "Open Energy aggregate →",
                href="/aggregate",
                className="refresh-button",
            ),
        ], className="selector-block"),
    ], className="selectors-grid")

    return html.Div([
        *_header(
            "Headline forecast & contributions",
            "One analytical workspace for the Headline path, component inflation "
            "and exact additive contributions.",
        ),
        controls,
        html.Div([
            html.Div([html.Div("Key results", className="eyebrow"), html.H3("Readable Headline outlook", className="panel-title"), html.P("M+1 / M+2 / M+3 first; M+6 / M+12 appear only when selected in Display horizon.", className="panel-subtitle")], className="panel-heading"),
            readable_table("h5-forecast-summary-table", FORECAST_COLUMNS, page_size=7),
        ], className="panel table-panel"),
        html.Div([
            html.Div([
                html.Div("Forecast", className="eyebrow"),
                html.H3(
                    "Observed · fitted · baseline · conditional",
                    className="panel-title",
                ),
                html.P(
                    "The conditional line appears after running a Headline scenario.",
                    className="panel-subtitle",
                ),
            ], className="panel-heading"),
            dcc.Loading(
                dcc.Graph(
                    id="h5-forecast-graph",
                    config=graph_config("headline_forecast"),
                ),
                type="circle",
            ),
        ], className="panel chart-panel"),
        html.Div([
            html.Div([
                html.Div("Components", className="eyebrow"),
                html.H3(
                    "Energy · Food · NEIG · Services",
                    className="panel-title",
                ),
                html.P(
                    "Component paths use the same display horizon selected above.",
                    className="panel-subtitle",
                ),
            ], className="panel-heading"),
            dcc.Loading(
                dcc.Graph(
                    id="h5-components-graph",
                    config=graph_config("headline_components"),
                ),
                type="circle",
            ),
        ], className="panel chart-panel"),
        html.Div([
            html.Div([
                html.Div([
                    html.Div("Exact additive decomposition", className="eyebrow"),
                    html.H3(
                        "Headline HICP contribution decomposition",
                        className="panel-title",
                    ),
                    html.P(
                        "Observed history uses exact chain-linked component shares "
                        "rescaled to official Headline HICP; the model path uses saved "
                        "draw-wise posterior contribution means.",
                        className="panel-subtitle",
                    ),
                ]),
                html.Div([
                    html.Div([
                        html.Span("DISPLAY", className="control-label"),
                        dcc.RadioItems(
                            id="h5-contrib-display",
                            options=[
                                {"label": "Stacked bars", "value": "bars"},
                                {"label": "Lines", "value": "lines"},
                            ],
                            value="bars",
                            inline=True,
                            className="fan-radio",
                        ),
                    ], className="control-block"),
                    html.Div([
                        html.Span("LABELS", className="control-label"),
                        dcc.Checklist(
                            id="h5-contrib-labels",
                            options=[
                                {"label": "Show latest values", "value": "latest"},
                            ],
                            value=[],
                            className="fan-radio",
                        ),
                    ], className="control-block"),
                ], className="chart-controls"),
            ], className="page-heading-row"),
            dcc.Loading(
                dcc.Graph(
                    id="h5-contributions-graph",
                    config=graph_config("headline_contributions"),
                ),
                type="circle",
            ),
        ], className="panel chart-panel"),
    ], className="page-body headline-page")


def _num(cid,value,step,minv=None):
    return dcc.Input(id=cid,type="number",value=value,step=step,min=minv,debounce=True,className="estimation-number-input")


def headline_estimation_v2_page():
    p=production_prior_config(); s=production_sampler_config(); c=model_contract()
    return html.Div([*_header("Headline estimation","Locked architecture with explicit prior/sampler overrides and background execution."),
        html.Div([
            html.Div([html.Div("Production lock",className="eyebrow"),html.H3("Headline joint BVAR",className="panel-title"),html.P(f"Dense BVAR({c['lag_order']}) · {c['seasonal_dummies']} monthly dummies · December reference · Eurostat FOOD · horizon ≤ {c['publication_horizon_max_months']}m",className="placeholder-text"),html.P("Numeric overrides create a different run_id; they do not change the locked model architecture.",className="stat-subtitle")],className="panel"),
            html.Div([html.Div("Run diagnostics",className="eyebrow"),html.Div(id="h5-estimation-diagnostics",children="Select a Headline run.")],className="panel")
        ],className="two-column-grid"),
        html.Div([html.Div("Prior",className="eyebrow"),html.H3("Minnesota + outlier prior",className="panel-title"),html.Div([
            html.Div([html.Label("λ1"),_num("h5-lambda1",p.lambda1,.01,.001)]),html.Div([html.Label("λ2"),_num("h5-lambda2",p.lambda2,.01,.001)]),html.Div([html.Label("λ3"),_num("h5-lambda3",p.lambda3,.05,.001)]),html.Div([html.Label("λ4"),_num("h5-lambda4",p.lambda4,.5,.001)]),html.Div([html.Label("Mean outlier interval (months)"),_num("h5-outlier-months",1/p.outlier_mean_frequency,1,2)])],className="selectors-grid")],className="panel"),
        html.Div([html.Div("Sampler",className="eyebrow"),html.H3("Gibbs configuration",className="panel-title"),html.Div([
            html.Div([html.Label("Reps"),_num("h5-reps",s.reps,500,2)]),html.Div([html.Label("Burn"),_num("h5-burn",s.burn,500,0)]),html.Div([html.Label("Thin"),_num("h5-thin",s.thin,1,1)]),html.Div([html.Label("Seed"),_num("h5-seed",s.seed,1,0)]),html.Div([dcc.Checklist(id="h5-promote-after",options=[{"label":"Promote after successful run","value":"promote"}],value=["promote"])])],className="selectors-grid"),
            html.Div([html.Button("Run Headline BVAR",id="h5-estimate-run",n_clicks=0,className="refresh-button"),html.Button("Cancel",id="h5-estimate-cancel",n_clicks=0,disabled=True,className="refresh-button")],style={"display":"flex","gap":"10px","marginTop":"16px"}),
            html.Progress(id="h5-estimate-progress",value=0,max=100,style={"width":"100%","marginTop":"16px"}),html.Div("Ready",id="h5-estimate-progress-text",className="stat-subtitle"),dcc.Store(id="h5-estimate-result"),html.Div(id="h5-estimate-result-banner",className="selection-banner")
        ],className="panel")
    ],className="page-body headline-page")


def _overview_values(frame):
    hist=_select(frame,record_type="history",series="hicp_total",metric="yoy")
    fut=_future(_select(frame,record_type="fan",series="hicp_total",metric="yoy")); out=[]
    if hist.empty: out.append(("—",""))
    else:
        r=hist.iloc[-1]; out.append((_fmt(r["value"],"%"),pd.Timestamp(r["date"]).strftime("%Y-%m")))
    col=statistic_column(fut, "mean") if not fut.empty else "value"
    for h in (1,2,3):
        if len(fut)<h: out.append(("—",""))
        else:
            r=fut.iloc[h-1]; out.append((_fmt(r[col],"%"),pd.Timestamp(r["date"]).strftime("%Y-%m")))
    return out


def _diagnostic_summary(results_root,store):
    meta=dict((store or {}).get("meta") or {})
    run_id=meta.get("run_id") or meta.get("headline_run_id")
    vintage=meta.get("vintage")
    if not run_id or not vintage:
        return html.P("Select a Headline run.",className="placeholder-text")
    run=Path(results_root)/"headline_joint"/str(vintage)/str(run_id)

    def _build():
        pieces=[]
        path=run/"diagnostics.parquet"; csv=run/"diagnostics.csv"
        try: d=pd.read_parquet(path) if path.is_file() else pd.read_csv(csv)
        except Exception: d=pd.DataFrame()
        if not d.empty and "parameter" in d:
            r=d.loc[d["parameter"].astype(str).str.lower().eq("spectral radius")]
            if not r.empty:
                for col in ("posterior_mean","value"):
                    if col in r and pd.notna(r.iloc[0][col]):
                        pieces.append(f"spectral radius {float(r.iloc[0][col]):.3f}")
                        break
        if not d.empty and "ESS" in d:
            ess=pd.to_numeric(d["ESS"],errors="coerce").dropna()
            if not ess.empty: pieces.append(f"min ESS {ess.min():.0f}")
        display=run/"forecasts"/"unconditional"/"display_v1.parquet"
        if display.is_file():
            try:
                rec=pd.read_parquet(display,columns=["record_type"])["record_type"].astype(str)
                pieces.append("fitted ready" if rec.eq("fitted").any() else "fitted missing")
            except Exception: pass
        pieces.append("structural ready" if (run/"headline_structural_v1.parquet").is_file() else "structural not materialised")
        return html.P(" · ".join(pieces) if pieces else "Saved diagnostics unavailable.",className="placeholder-text")

    return snapshot_get_or_build(
        "headline_diagnostics",
        str(run.resolve()),
        _build,
    )


def _promote_if_supported(results_root,registry_path,vintage,run_id):
    try: import inflation_bvar_registry as reg
    except Exception: return False,"Unified registry unavailable; run saved but not promoted."
    scan=getattr(reg,"scan_results",None)
    if callable(scan):
        try:
            sig=inspect.signature(scan); kw={}
            if "results_root" in sig.parameters: kw["results_root"]=results_root
            if "registry_path" in sig.parameters and registry_path is not None: kw["registry_path"]=registry_path
            scan(**kw)
        except Exception: pass
    for name in ("promote_headline_run","promote_run"):
        fn=getattr(reg,name,None)
        if not callable(fn): continue
        sig=inspect.signature(fn); kw={}; p=sig.parameters
        if "registry_path" in p and registry_path is not None: kw["registry_path"]=registry_path
        if "results_root" in p: kw["results_root"]=results_root
        if "model_id" in p: kw["model_id"]="headline_joint"
        if "domain" in p: kw["domain"]="headline"
        if "suite" in p: kw["suite"]="headline"
        if "vintage" in p: kw["vintage"]=str(vintage)
        if "run_id" in p: kw["run_id"]=str(run_id)
        if "required_forecast" in p: kw["required_forecast"]="unconditional"
        if "require_draws" in p: kw["require_draws"]=True
        try: fn(**kw); return True,"Promoted in unified registry."
        except TypeError: continue
        except Exception as exc: return False,f"Run saved; promotion failed: {exc}"
    return False,"Run saved; no Headline-capable promotion function found."


def register_headline_slice5_callbacks(app, *, results_root, registry_path=None, background_manager=None, store_id="data-store", vintage_selector_id="vintage-select"):
    results_root=Path(results_root).resolve(); registry_path=None if registry_path is None else Path(registry_path).resolve()

    @app.callback(Output("h5-forecast-graph","figure"),Output("h5-forecast-summary-table","data"),Input(store_id,"data"),Input("h5-forecast-series","value"),Input("h5-forecast-metric","value"),Input("h5-forecast-horizon","value"),Input("h5-forecast-fan","value"),Input("h6-headline-scenario-store","data"))
    def forecast(store,series,metric,horizon,fan,conditional_store):
        f=_frame_from_store(store)
        if f.empty:
            return _empty("Select a Headline run."), []
        series=series or "hicp_total"; metric=metric or "yoy"; horizon=int(horizon or 3)
        fig=headline_forecast_figure(f,series=series,metric=metric,horizon=horizon,fan_mode=fan or "68",statistic="mean")
        fig=overlay_headline_conditional_forecast(fig,conditional_store,series=series,metric=metric,horizon=horizon)
        return fig, forecast_summary_records(
            f, series=series, metric=metric, max_months=horizon
        )

    @app.callback(
        Output("h5-components-graph","figure"),
        Input(store_id,"data"),
        Input("h5-forecast-horizon","value"),
    )
    def components(store,horizon):
        f=_frame_from_store(store); h=int(horizon or 3)
        return _empty("Select a Headline run.") if f.empty else components_figure(f,h)

    @app.callback(
        Output("h5-contributions-graph","figure"),
        Input(store_id,"data"),
        Input("h5-forecast-horizon","value"),
        Input("h5-contrib-display","value"),
        Input("h5-contrib-labels","value"),
    )
    def contributions(store,horizon,display_mode,labels):
        f=_frame_from_store(store); h=int(horizon or 3)
        if f.empty:
            return _empty("Select a Headline run.")
        return contribution_decomposition_figure(
            f,
            horizon=h,
            display_mode=display_mode or "bars",
            show_latest="latest" in (labels or []),
        )

    @app.callback(Output("h5-estimation-diagnostics","children"),Input(store_id,"data"),Input("h5-estimate-result","data"))
    def diagnostics(store,_): return _diagnostic_summary(results_root,store)

    @app.callback(Output("h5-estimate-result-banner","children"),Input("h5-estimate-result","data"))
    def banner(r):
        if not r: return ""
        return html.Div([html.Strong("Complete" if r.get("ok") else "Failed"),html.Span(f" · {r.get('message','')}"),html.Span(f" · run {str(r.get('run_id',''))[:12]}" if r.get("run_id") else "")])

    states=[State(vintage_selector_id,"value"),State("h5-lambda1","value"),State("h5-lambda2","value"),State("h5-lambda3","value"),State("h5-lambda4","value"),State("h5-outlier-months","value"),State("h5-reps","value"),State("h5-burn","value"),State("h5-thin","value"),State("h5-seed","value"),State("h5-promote-after","value")]
    if background_manager is None:
        @app.callback(Output("h5-estimate-result","data"),Input("h5-estimate-run","n_clicks"),*states,prevent_initial_call=True)
        def no_manager(n,*_):
            if not n: raise PreventUpdate
            return {"ok":False,"message":"Headline estimation requires the shell background callback manager."}
        return

    @app.callback(Output("h5-estimate-result","data"),Input("h5-estimate-run","n_clicks"),*states,background=True,manager=background_manager,
                  progress=[Output("h5-estimate-progress","value"),Output("h5-estimate-progress-text","children")],progress_default=[0,"Ready"],
                  cancel=[Input("h5-estimate-cancel","n_clicks")],running=[(Output("h5-estimate-run","disabled"),True,False),(Output("h5-estimate-cancel","disabled"),False,True)],prevent_initial_call=True)
    def estimate(set_progress,n,vintage,l1,l2,l3,l4,outlier_months,reps,burn,thin,seed,promote):
        if not n: raise PreventUpdate
        started=time.monotonic()
        try:
            if vintage is None: raise ValueError("Select a Headline vintage.")
            outlier_months=float(outlier_months)
            if outlier_months<=1: raise ValueError("Mean outlier interval must exceed one month.")
            p0=production_prior_config(); s0=production_sampler_config()
            prior=BVARSVOPriorConfig(lambda1=float(l1),lambda2=float(l2),lambda3=float(l3),lambda4=float(l4),a_prior_var=p0.a_prior_var,phi_prior_mean=p0.phi_prior_mean,phi_prior_df=p0.phi_prior_df,h0_var=p0.h0_var,outlier_mean_frequency=1/outlier_months,outlier_prior_observations=p0.outlier_prior_observations,outlier_grid_min=p0.outlier_grid_min,outlier_grid_max=p0.outlier_grid_max,outlier_grid_step=p0.outlier_grid_step,ksc_offset_scale=p0.ksc_offset_scale,ksc_offset_floor=p0.ksc_offset_floor)
            sampler=SamplerConfig(reps=int(reps),burn=int(burn),thin=int(thin),seed=int(seed),max_stability_tries=s0.max_stability_tries,progress_every=s0.progress_every,sv_sampler=s0.sv_sampler,dk_projection_mode=s0.dk_projection_mode,dk_level_relative_gate=s0.dk_level_relative_gate,dk_difference_relative_gate=s0.dk_difference_relative_gate,dk_catastrophic_level_relative_gate=s0.dk_catastrophic_level_relative_gate,dk_catastrophic_difference_relative_gate=s0.dk_catastrophic_difference_relative_gate)
            prior.validate(); sampler.validate(); set_progress((5,"Preparing locked Headline BVAR")); set_progress((15,"Sampling BVAR — this stage may take time"))
            outcome=run_headline(str(vintage),results_root=results_root,prior_config=prior,sampler_config=sampler,H=MAX_PUBLISHED_HORIZON_MONTHS,n_forecast_draws=1000,simulate_future_outliers=True,forecast_seed=2026,persist=True)
            set_progress((88,"Materialising display")); build_headline_display(outcome.run_directory,forecast_name="unconditional",overwrite=True)
            set_progress((93,"Materialising one-step fitted paths")); materialize_headline_fitted(outcome.run_directory,forecast_name="unconditional")
            promoted=False; msg="Promotion not requested."
            if "promote" in (promote or []): set_progress((97,"Registering / promoting")); promoted,msg=_promote_if_supported(results_root,registry_path,outcome.vintage,outcome.run_id)
            set_progress((100,"Complete")); return {"ok":True,"message":f"Estimation completed. {msg}","vintage":outcome.vintage,"run_id":outcome.run_id,"directory":str(outcome.run_directory),"posterior_draws":outcome.n_posterior_draws,"forecast_draws":outcome.n_forecast_draws,"promoted":promoted,"elapsed_seconds":time.monotonic()-started}
        except Exception as exc:
            set_progress((0,f"Failed: {exc}")); return {"ok":False,"message":str(exc),"vintage":None if vintage is None else str(vintage),"elapsed_seconds":time.monotonic()-started}


__all__=["HORIZON_OPTIONS","DEFAULT_DISPLAY_HORIZON","headline_forecast_v2_page","headline_estimation_v2_page","headline_forecast_figure","headline_overview_figure","displayed_contribution_additivity_error","historical_contribution_additivity_error","contribution_snapshot_figure","components_figure","contribution_decomposition_figure","contribution_stack_figure","headline_core_comparison_figure","core_historical_contribution_additivity_error","core_contribution_decomposition_figure","register_headline_slice5_callbacks"]
