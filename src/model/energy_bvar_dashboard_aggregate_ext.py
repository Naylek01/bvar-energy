"""Step-2 presentation helpers for the HICP Energy aggregate dashboard.

This module is deliberately additive: it does not alter model estimation,
posterior draws, tax bridges, or the canonical aggregate pipeline.  It turns
already-saved aggregate results plus the canonical historical accounting
reconstruction into dashboard-ready diagnostics.

The historical contribution identity is delegated to
``energy_bvar_aggregate.yoy_contributions_from_terms``.  Forecast contribution
medians are read from ``display_v1.parquet``.  No chain-linking or posterior
aggregation is reimplemented here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

COMPONENT_ORDER: tuple[str, ...] = (
    "car_fuels",
    "liquid_fuels",
    "gas",
    "electricity",
    "heat_energy",
    "solid_fuels",
)

COMPONENT_LABELS: dict[str, str] = {
    "car_fuels": "Car fuels",
    "liquid_fuels": "Liquid fuels",
    "gas": "Gas",
    "electricity": "Electricity",
    "heat_energy": "Heat energy",
    "solid_fuels": "Solid fuels",
}

COMPONENT_COLOURS: dict[str, str] = {
    "car_fuels": "#004B99",
    "liquid_fuels": "#0079AE",
    "gas": "#A66A00",
    "electricity": "#0B7A5C",
    "heat_energy": "#6E4B9E",
    "solid_fuels": "#B3341F",
}

RUN_ID_LABELS: dict[str, str] = {
    "gas": "Gas run",
    "electricity": "Electricity run",
    "heat_energy": "Heat energy run",
    "solid_fuels": "Solid fuels run",
    "petrol": "Petrol run",
    "car_fuels_petrol": "Petrol run",
    "diesel": "Diesel run",
    "car_fuels_diesel": "Diesel run",
    "liquid_fuels": "Liquid fuels run",
}

PROVENANCE_LABELS: dict[str, str] = {
    "aggregate_run_id": "Aggregate run",
    "n_aggregate_draws_effective": "Effective aggregate draws",
    "n_aggregate_draws_requested": "Requested aggregate draws",
    "weekly_tax_mode_effective": "Weekly fuel tax treatment",
    "forecast_name": "Forecast contract",
    "pairing_seed": "Cross-model pairing seed",
    "weight_policy": "Annual HICP weight policy",
    "predictive_distribution_interpretation": "Predictive distribution",
    "cross_model_dependence": "Cross-model dependence",
    "other_transport_fuels_forecast": "Other transport fuels forecast rule",
    "bvar_fitted_cache_version": "BVAR fitted cache",
    "bvar_fitted_draws": "BVAR fitted aggregate draws",
}

DIAGNOSTIC_LABELS: dict[str, str] = {
    "weekly_rejection_rate": "Weekly admissibility rejection rate",
    "tax_round_trip_max_error": "Tax round-trip max error",
    "historical_reconstruction_max_error": "Historical reconstruction max error",
    "six_model_history_max_error": "Six-component cumulative reconstruction max error",
    "six_model_history_annual_reanchored_max_error": "Six-component annual re-anchored max error",
    "drawwise_contribution_additivity_error": "Draw-wise contribution additivity max error",
}


def build_historical_contribution_frame(
    project_root: str | Path,
    vintage: str,
) -> pd.DataFrame:
    """Return exact historical six-component contributions in long form.

    This is an *accounting* decomposition of observed HICP Energy, not a BVAR
    fitted object.  The canonical aggregate module supplies both the historical
    Laspeyres terms and the exact additive YoY contribution formula.
    """
    # Lazy import keeps this presentation module independently testable.
    from energy_bvar_aggregate import (  # type: ignore
        historical_model_component_reconstruction,
        yoy_contributions_from_terms,
    )

    root = Path(project_root).resolve()
    processed_dir = root / "data" / "processed" / str(vintage)
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"Processed vintage not found: {processed_dir}")

    model_history = historical_model_component_reconstruction(processed_dir)
    energy = model_history["energy"]
    contributions = yoy_contributions_from_terms(energy["index"], energy["terms"])

    missing = [name for name in COMPONENT_ORDER if name not in contributions.columns]
    if missing:
        raise ValueError(
            "Historical contribution reconstruction is missing component columns "
            f"{missing}."
        )

    rows: list[dict[str, Any]] = []
    for date, row in contributions.iterrows():
        aggregate_yoy = row.get("aggregate_yoy", np.nan)
        additivity_error = row.get("additivity_error", np.nan)
        for name in COMPONENT_ORDER:
            value = row.get(name, np.nan)
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "series": name,
                    "label": COMPONENT_LABELS[name],
                    "value": float(value),
                    "aggregate_yoy": (
                        float(aggregate_yoy) if np.isfinite(aggregate_yoy) else np.nan
                    ),
                    "additivity_error": (
                        float(additivity_error)
                        if np.isfinite(additivity_error)
                        else np.nan
                    ),
                    "basis": "historical_accounting",
                    "segment": "historical",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Historical contribution reconstruction produced no finite rows.")
    out = out.sort_values(["date", "series"]).reset_index(drop=True)

    errors = out["additivity_error"].dropna().abs()
    if len(errors) and float(errors.max()) >= 1e-10:
        raise ValueError(
            "Historical HICP Energy contributions are not additive to numerical "
            f"precision; max error={float(errors.max()):.3e}."
        )
    return out


def _forecast_contribution_rows(frame: pd.DataFrame, basis: str = "baseline") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    needed = {"record_type", "metric", "basis", "series", "date", "q50"}
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.loc[
        frame["record_type"].astype(str).eq("contribution")
        & frame["metric"].astype(str).eq("contribution_yoy")
        & frame["basis"].astype(str).eq(str(basis))
        & frame["series"].astype(str).isin(COMPONENT_ORDER)
    ].copy()
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["q50"] = pd.to_numeric(out["q50"], errors="coerce")
    return out.dropna(subset=["date", "q50"]).sort_values(["date", "series"])


def _aggregate_fan_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    needed = {"record_type", "metric", "series", "date", "q50"}
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.loc[
        frame["record_type"].astype(str).eq("fan")
        & frame["metric"].astype(str).eq("yoy")
        & frame["series"].astype(str).eq("hicp_energy")
    ].copy()
    if out.empty:
        return out
    if "basis" in out.columns and out["basis"].notna().any():
        baseline = out.loc[out["basis"].astype(str).eq("baseline")]
        if not baseline.empty:
            out = baseline
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["q50"] = pd.to_numeric(out["q50"], errors="coerce")
    return out.dropna(subset=["date", "q50"]).sort_values("date")


def _history_before_model_path(
    historical: pd.DataFrame,
    model_rows: pd.DataFrame,
) -> pd.DataFrame:
    if historical.empty or model_rows.empty:
        return historical.copy()
    first_model_date = pd.Timestamp(model_rows["date"].min())
    return historical.loc[pd.to_datetime(historical["date"]) < first_model_date].copy()


def _end_text(values: pd.Series, enabled: bool) -> list[str]:
    text = [""] * len(values)
    if enabled and len(values):
        final = pd.to_numeric(values, errors="coerce").iloc[-1]
        if np.isfinite(final):
            text[-1] = f"{float(final):+.2f}"
    return text


def _add_forecast_origin(fig: go.Figure, forecast_origin: Any) -> None:
    if forecast_origin is None or pd.isna(forecast_origin):
        return
    try:
        date = pd.Timestamp(forecast_origin)
    except Exception:
        return
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    # Avoid Plotly's add_vline(annotation=...) datetime arithmetic, which is
    # incompatible with newer pandas Timestamp semantics.
    fig.add_shape(
        type="line",
        x0=date,
        x1=date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "#9CA3AF", "width": 1, "dash": "dot"},
    )
    fig.add_annotation(
        x=date,
        y=0.98,
        xref="x",
        yref="paper",
        text="Forecast origin",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"size": 10, "color": "#6B7280"},
        bgcolor="rgba(255,255,255,0.72)",
    )


def tidy_aggregate_figure(
    fig: go.Figure,
    *,
    title: str | None = None,
    y_title: str | None = None,
    height: int | None = None,
    uirevision: str | None = None,
) -> go.Figure:
    """Reserve explicit title/legend space for Aggregate-page figures."""
    update: dict[str, Any] = {
        "margin": {"l": 58, "r": 28, "t": 82, "b": 96},
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "hovermode": "x unified",
        "dragmode": "pan",
    }
    if title is not None:
        update["title"] = {
            "text": title,
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 17, "color": "#111827"},
        }
    if y_title is not None:
        update["yaxis_title"] = y_title
    if height is not None:
        update["height"] = int(height)
    if uirevision is not None:
        update["uirevision"] = str(uirevision)
    fig.update_layout(**update)
    return fig


def _visible_x_window(relayout_data: Mapping[str, Any] | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Extract Plotly's current x-window from relayoutData.

    Double-click/autoscale returns ``xaxis.autorange=True`` and therefore maps
    back to ``None`` (full history).
    """
    data = dict(relayout_data or {})
    if bool(data.get("xaxis.autorange")):
        return None

    left = data.get("xaxis.range[0]")
    right = data.get("xaxis.range[1]")
    if left is None or right is None:
        raw = data.get("xaxis.range")
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            left, right = raw
    if left is None or right is None:
        return None

    try:
        a = pd.Timestamp(left)
        b = pd.Timestamp(right)
    except Exception:
        return None
    if a.tzinfo is not None:
        a = a.tz_localize(None)
    if b.tzinfo is not None:
        b = b.tz_localize(None)
    return (min(a, b), max(a, b))


def adapt_yaxis_to_visible_window(
    fig: go.Figure,
    relayout_data: Mapping[str, Any] | None,
    *,
    include_zero: bool = False,
    padding_fraction: float = 0.08,
    minimum_padding: float = 0.5,
) -> go.Figure:
    """Fit Y to the currently visible X range without clipping extrema.

    Plotly normally keeps the vertical range established when the full figure
    was first built.  That is awkward here because the six historical fitted
    series have very large 2021-23 spikes: after an x-zoom, the local series can
    hit the top/bottom of the old frame.  This helper scans every visible trace
    point inside the current x-window and applies a padded Y range.

    On full-history/reset it also scans all traces, so fitted overlays added
    after the base fan figure cannot remain outside the original Y range.
    """
    window = _visible_x_window(relayout_data)
    values: list[float] = []

    for trace in fig.data:
        if getattr(trace, "visible", True) == "legendonly":
            continue
        xs = getattr(trace, "x", None)
        ys = getattr(trace, "y", None)
        if xs is None or ys is None:
            continue
        try:
            x = pd.to_datetime(pd.Series(list(xs)), errors="coerce")
            y = pd.to_numeric(pd.Series(list(ys)), errors="coerce")
        except Exception:
            continue
        valid = x.notna() & y.notna()
        if window is not None:
            valid &= (x >= window[0]) & (x <= window[1])
        if valid.any():
            values.extend(y.loc[valid].astype(float).tolist())

    if include_zero:
        values.append(0.0)

    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        fig.update_yaxes(autorange=True)
    else:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        span = hi - lo
        pad = max(
            float(minimum_padding),
            float(padding_fraction) * (span if span > 0 else max(abs(lo), abs(hi), 1.0)),
        )
        fig.update_yaxes(range=[lo - pad, hi + pad], autorange=False)

    # Make the callback-returned figure explicitly retain the x-window that
    # triggered it.  On double-click/autoscale, let Plotly restore full range.
    if window is None:
        if bool(dict(relayout_data or {}).get("xaxis.autorange")):
            fig.update_xaxes(autorange=True)
    else:
        fig.update_xaxes(range=[window[0], window[1]], autorange=False)
    return fig


def aggregate_contribution_timeline_figure(
    display_frame: pd.DataFrame,
    historical_frame: pd.DataFrame,
    *,
    display_mode: str = "bars",
    show_labels: bool = False,
    forecast_origin: Any = None,
    uirevision: str = "aggregate-contribution-timeline",
) -> go.Figure:
    """Historical exact contributions + saved posterior median model path."""
    mode = str(display_mode or "bars").lower()
    if mode not in {"bars", "lines"}:
        raise ValueError("display_mode must be 'bars' or 'lines'.")

    forecast = _forecast_contribution_rows(display_frame, basis="baseline")
    history = historical_frame.copy()
    if not history.empty:
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        history["value"] = pd.to_numeric(history["value"], errors="coerce")
        history = history.dropna(subset=["date", "value"])
    history = _history_before_model_path(history, forecast)

    if history.empty and forecast.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No historical or forecast contribution rows are available.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return tidy_aggregate_figure(fig, title="HICP Energy contribution decomposition")

    fig = go.Figure()
    for name in COMPONENT_ORDER:
        h = history.loc[history["series"].astype(str).eq(name)].sort_values("date")
        f = forecast.loc[forecast["series"].astype(str).eq(name)].sort_values("date")
        x = pd.concat([h.get("date", pd.Series(dtype="datetime64[ns]")), f.get("date", pd.Series(dtype="datetime64[ns]"))], ignore_index=True)
        y = pd.concat([h.get("value", pd.Series(dtype=float)), f.get("q50", pd.Series(dtype=float))], ignore_index=True)
        valid = pd.DataFrame({"date": x, "value": pd.to_numeric(y, errors="coerce")}).dropna()
        if valid.empty:
            continue
        label = COMPONENT_LABELS[name]
        if mode == "bars":
            text = _end_text(valid["value"], show_labels)
            fig.add_trace(
                go.Bar(
                    x=valid["date"],
                    y=valid["value"],
                    name=label,
                    marker_color=COMPONENT_COLOURS[name],
                    text=text,
                    textposition="outside" if show_labels else None,
                    cliponaxis=False,
                    hovertemplate=f"%{{y:+.2f}} pp<extra>{label}</extra>",
                )
            )
        else:
            text = _end_text(valid["value"], show_labels)
            fig.add_trace(
                go.Scatter(
                    x=valid["date"],
                    y=valid["value"],
                    mode="lines+text" if show_labels else "lines",
                    text=text if show_labels else None,
                    textposition="middle right",
                    name=label,
                    line={"width": 1.7, "color": COMPONENT_COLOURS[name]},
                    hovertemplate=f"%{{y:+.2f}} pp<extra>{label}</extra>",
                )
            )

    # Aggregate observed YoY history + aggregate posterior median path.
    aggregate_hist = pd.DataFrame()
    if not history.empty:
        aggregate_hist = (
            history[["date", "aggregate_yoy"]]
            .dropna()
            .drop_duplicates("date")
            .sort_values("date")
        )
    fan = _aggregate_fan_rows(display_frame)
    if not aggregate_hist.empty or not fan.empty:
        agg_x = pd.concat(
            [aggregate_hist.get("date", pd.Series(dtype="datetime64[ns]")), fan.get("date", pd.Series(dtype="datetime64[ns]"))],
            ignore_index=True,
        )
        agg_y = pd.concat(
            [aggregate_hist.get("aggregate_yoy", pd.Series(dtype=float)), fan.get("q50", pd.Series(dtype=float))],
            ignore_index=True,
        )
        agg = pd.DataFrame({"date": agg_x, "value": pd.to_numeric(agg_y, errors="coerce")}).dropna()
        agg = agg.drop_duplicates("date", keep="last").sort_values("date")
        if not agg.empty:
            text = _end_text(agg["value"], show_labels)
            fig.add_trace(
                go.Scatter(
                    x=agg["date"],
                    y=agg["value"],
                    mode="lines+text" if show_labels else "lines",
                    text=text if show_labels else None,
                    textposition="top right",
                    name="HICP Energy",
                    line={"color": "#111827", "width": 2.4, "dash": "dash"},
                    hovertemplate="%{y:.2f}%<extra>HICP Energy</extra>",
                )
            )

    fig.add_hline(y=0.0, line={"color": "#D1D5DB", "width": 1})
    _add_forecast_origin(fig, forecast_origin)
    if mode == "bars":
        fig.update_layout(barmode="relative")
    return tidy_aggregate_figure(
        fig,
        title="HICP Energy contribution decomposition — history + model path",
        y_title="percentage points",
        height=540,
        uirevision=uirevision,
    )


def aggregate_notebook_median_contribution_figure(
    display_frame: pd.DataFrame,
    *,
    show_labels: bool = False,
    forecast_origin: Any = None,
    uirevision: str = "aggregate-notebook-median-contrib",
) -> go.Figure:
    """Reproduce Notebook 10's forecast median-contribution line chart."""
    forecast = _forecast_contribution_rows(display_frame, basis="baseline")
    if forecast.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No saved baseline contribution paths are available.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return tidy_aggregate_figure(
            fig, title="Median component contributions to HICP Energy inflation"
        )

    fig = go.Figure()
    for name in COMPONENT_ORDER:
        block = forecast.loc[forecast["series"].astype(str).eq(name)].sort_values("date")
        if block.empty:
            continue
        label = COMPONENT_LABELS[name]
        text = _end_text(block["q50"], show_labels)
        fig.add_trace(
            go.Scatter(
                x=block["date"],
                y=block["q50"],
                mode="lines+markers+text" if show_labels else "lines+markers",
                text=text if show_labels else None,
                textposition="middle right",
                name=label,
                line={"width": 1.7, "color": COMPONENT_COLOURS[name]},
                marker={"size": 5},
                hovertemplate=f"%{{y:+.2f}} pp<extra>{label}</extra>",
            )
        )

    fan = _aggregate_fan_rows(display_frame)
    if not fan.empty:
        text = _end_text(fan["q50"], show_labels)
        fig.add_trace(
            go.Scatter(
                x=fan["date"],
                y=fan["q50"],
                mode="lines+markers+text" if show_labels else "lines+markers",
                text=text if show_labels else None,
                textposition="top right",
                name="HICP Energy YoY median",
                line={"color": "#111827", "width": 2.4, "dash": "dash"},
                marker={"size": 5},
                hovertemplate="%{y:.2f}%<extra>HICP Energy YoY median</extra>",
            )
        )

    fig.add_hline(y=0.0, line={"color": "#D1D5DB", "width": 1})
    _add_forecast_origin(fig, forecast_origin)
    return tidy_aggregate_figure(
        fig,
        title="Median component contributions to HICP Energy inflation",
        y_title="percentage points",
        height=500,
        uirevision=uirevision,
    )


def _short_run_id(value: Any, n: int = 12) -> str:
    text = str(value or "")
    return text if len(text) <= n else text[:n] + "…"


def aggregate_provenance_table_v2(
    display_frame: pd.DataFrame,
    meta: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Readable build details while preserving exact machine identifiers."""
    meta = dict(meta or {})
    rows: list[dict[str, str]] = []

    # Put the seven source model runs first: these are usually what a user
    # wants when asking "which results am I looking at?".
    run_ids = dict(meta.get("component_run_ids", {}) or {})
    canonical_key_order = (
        "gas",
        "electricity",
        "heat_energy",
        "solid_fuels",
        "petrol",
        "diesel",
        "liquid_fuels",
    )
    alias = {
        "petrol": ("petrol", "car_fuels_petrol"),
        "diesel": ("diesel", "car_fuels_diesel"),
    }
    for key in canonical_key_order:
        candidates = alias.get(key, (key,))
        full = next(
            (run_ids.get(candidate) for candidate in candidates if run_ids.get(candidate)),
            None,
        )
        if not full:
            continue
        label = RUN_ID_LABELS.get(key, key.replace("_", " ").title() + " run")
        rows.append(
            {
                "Section": "Source model runs",
                "Item": label.replace(" run", ""),
                "Value": _short_run_id(full),
                "Detail": str(full),
            }
        )

    # Then show the actual construction choices of the selected aggregate.
    construction_keys = (
        "aggregate_run_id",
        "forecast_name",
        "n_aggregate_draws_effective",
        "n_aggregate_draws_requested",
        "pairing_seed",
        "cross_model_dependence",
        "weekly_tax_mode_effective",
        "weight_policy",
        "other_transport_fuels_forecast",
        "predictive_distribution_interpretation",
        "bvar_fitted_cache_version",
        "bvar_fitted_draws",
    )
    for key in construction_keys:
        value = meta.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        rows.append(
            {
                "Section": "Aggregate construction",
                "Item": PROVENANCE_LABELS.get(key, key.replace("_", " ").title()),
                "Value": str(value),
                "Detail": "",
            }
        )

    # Validation metrics stay visually separate from source/provenance fields.
    if (
        not display_frame.empty
        and {"record_type", "metric", "series", "value"}.issubset(display_frame.columns)
    ):
        diagnostics = display_frame.loc[
            display_frame["record_type"].astype(str).eq("diagnostic")
        ]
        for _, row in diagnostics.iterrows():
            metric = str(row.get("metric", ""))
            series = str(row.get("series", ""))
            label = DIAGNOSTIC_LABELS.get(
                metric, metric.replace("_", " ").strip().capitalize()
            )
            if series and series not in {"hicp_energy", "nan"}:
                label += " — " + COMPONENT_LABELS.get(
                    series, series.replace("_", " ").title()
                )
            value = row.get("value")
            if pd.notna(value):
                rows.append(
                    {
                        "Section": "Validation checks",
                        "Item": label,
                        "Value": f"{float(value):,.6g}",
                        "Detail": "",
                    }
                )

    return pd.DataFrame(
        rows,
        columns=["Section", "Item", "Value", "Detail"],
    )


__all__ = [
    "COMPONENT_ORDER",
    "COMPONENT_LABELS",
    "build_historical_contribution_frame",
    "aggregate_contribution_timeline_figure",
    "aggregate_provenance_table_v2",
    "adapt_yaxis_to_visible_window",
    "tidy_aggregate_figure",
]
