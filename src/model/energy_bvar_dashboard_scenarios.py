"""Pure display helpers for interactive component tax scenarios.

The econometric/tax transformation lives in ``energy_bvar_component_hicp``.
This module only reduces paired baseline/scenario draws to dashboard-sized
quantiles and renders Plotly figures.  It never re-estimates a BVAR.
"""

from __future__ import annotations

from io import StringIO
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from energy_bvar_theme import TOKENS, apply_theme
except Exception:  # pragma: no cover - safe fallback during standalone testing
    TOKENS = {
        "ink": "#040506",
        "ink_soft": "#3A3D40",
        "muted": "#6B6E72",
        "cyan": "#009FE3",
        "cyan_deep": "#0079AE",
        "positive": "#0B7A5C",
        "warning": "#A66A00",
        "hairline": "#D2D0D1",
        "hairline_soft": "#DEDCDD",
        "surface": "#F7F6F7",
    }

    def apply_theme(fig, **kwargs):
        if kwargs.get("height"):
            fig.update_layout(height=kwargs["height"])
        if kwargs.get("y_title"):
            fig.update_yaxes(title_text=kwargs["y_title"])
        if kwargs.get("uirevision"):
            fig.update_layout(uirevision=kwargs["uirevision"])
        fig.update_layout(template="plotly_white", dragmode="pan")
        return fig


# TAX_SCENARIO_EXPLICIT_HICP_LABEL_V1
_QUANTILES = (0.05, 0.16, 0.50, 0.84, 0.95)
_QCOLS = ("q05", "q16", "q50", "q84", "q95")
_BASE = TOKENS["cyan_deep"]
_SCEN = TOKENS["positive"]
_INK = TOKENS["ink"]
_MUTED = TOKENS["muted"]
_NOW = TOKENS.get("warning", "#A66A00")

SCENARIO_DISPLAY_LAYOUT_VERSION = "energy-tax-scenario-layout-v2"


def _quantile_frame(paths: np.ndarray, dates, *, name: str) -> pd.DataFrame:
    values = np.asarray(paths, dtype=float)
    dates = pd.DatetimeIndex(dates, name="date")
    if values.ndim != 2 or values.shape[1] != len(dates):
        raise ValueError(f"{name}: expected draw x date array, got {values.shape}.")
    q = np.nanquantile(values, _QUANTILES, axis=0).T
    frame = pd.DataFrame(q, index=dates, columns=_QCOLS).reset_index()
    frame["name"] = name
    return frame


def _json(frame: pd.DataFrame) -> str:
    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.to_json(orient="split", date_format="iso", double_precision=15)


def _read(payload: Mapping | None, key: str) -> pd.DataFrame:
    if not payload or not payload.get(key):
        return pd.DataFrame()
    frame = pd.read_json(StringIO(payload[key]), orient="split")
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def scenario_payload(result: Mapping) -> dict[str, Any]:
    """Reduce a paired scenario object to browser-safe quantiles/tax tables."""
    dates = pd.DatetimeIndex(result["path_dates"], name="date")
    actual = pd.Series(result["actual_hicp"], dtype=float).dropna().sort_index()
    actual.index = pd.DatetimeIndex(pd.to_datetime(actual.index), name="date")
    actual_yoy = 100.0 * (actual / actual.shift(12) - 1.0)
    history = pd.DataFrame({"date": actual_yoy.index, "value": actual_yoy.to_numpy()}).dropna()

    fan_blocks = []
    for name, key in (
        ("baseline_yoy", "baseline_yoy_paths"),
        ("scenario_yoy", "scenario_yoy_paths"),
        ("baseline_level", "baseline_level_paths"),
        ("scenario_level", "scenario_level_paths"),
        ("level_impact_pct", "level_impact_pct_paths"),
        ("yoy_impact_pp", "yoy_impact_pp_paths"),
    ):
        fan_blocks.append(_quantile_frame(result[key], dates, name=name))
    fans = pd.concat(fan_blocks, ignore_index=True)

    tax_path = pd.DataFrame(result.get("tax_path", pd.DataFrame())).copy()
    if not tax_path.empty:
        tax_path.index = pd.DatetimeIndex(pd.to_datetime(tax_path.index), name="date")
        tax_path = tax_path.reset_index()
    tax_history = pd.DataFrame(result.get("tax_history", pd.DataFrame())).copy()
    if not tax_history.empty:
        tax_history.index = pd.DatetimeIndex(pd.to_datetime(tax_history.index), name="date")
        tax_history = tax_history.reset_index()
    raw_tax = pd.DataFrame(result.get("raw_tax_publications", pd.DataFrame())).copy()
    if not raw_tax.empty:
        raw_tax.index = pd.DatetimeIndex(pd.to_datetime(raw_tax.index), name="date")
        raw_tax = raw_tax.reset_index()

    future_dates = pd.DatetimeIndex(result.get("future_dates", []), name="date")
    last_observed = actual.dropna().index.max() if not actual.dropna().empty else pd.NaT
    forecast_origin = future_dates.min() if len(future_dates) else pd.NaT
    meta = {
        "model_id": str(result["model_id"]),
        "hicp_series": str(result["hicp_series"]),
        "hicp_label": str(result["hicp_label"]),
        "scenario_start": pd.Timestamp(result["scenario_start"]).isoformat(),
        "forecast_origin": None if pd.isna(forecast_origin) else pd.Timestamp(forecast_origin).isoformat(),
        "last_observed": None if pd.isna(last_observed) else pd.Timestamp(last_observed).isoformat(),
        "vat_delta_pp": float(result.get("vat_delta_pp", 0.0)),
        "excise_delta": float(result.get("excise_delta", 0.0)),
        "excise_unit": str(result.get("excise_unit", "source unit")),
        "n_draws_effective": int(result.get("n_draws_effective", 0)),
        "rejection_rate": float(result.get("rejection_rate", 0.0)),
        "distribution_interpretation": str(result.get("distribution_interpretation", "")),
        "source_frequency": str(result.get("source_frequency", "")),
    }
    return {
        "history_json": _json(history),
        "fans_json": _json(fans),
        "tax_path_json": _json(tax_path),
        "tax_history_json": _json(tax_history),
        "raw_tax_json": _json(raw_tax),
        "meta": meta,
    }


def scenario_frames(payload: Mapping | None) -> dict[str, pd.DataFrame]:
    return {
        "history": _read(payload, "history_json"),
        "fans": _read(payload, "fans_json"),
        "tax_path": _read(payload, "tax_path_json"),
        "tax_history": _read(payload, "tax_history_json"),
        "raw_tax": _read(payload, "raw_tax_json"),
    }


def _slice(fans: pd.DataFrame, name: str) -> pd.DataFrame:
    if fans.empty:
        return fans
    return fans.loc[fans["name"].astype(str) == str(name)].sort_values("date")


def _rgba(hex_color: str, opacity: float) -> str:
    value = hex_color.lstrip("#")
    rgb = tuple(int(value[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{opacity})"


def _band(fig: go.Figure, block: pd.DataFrame, lower: str, upper: str, *, color: str, name: str, opacity: float) -> None:
    if block.empty:
        return
    x = pd.concat([block["date"], block["date"].iloc[::-1]], ignore_index=True)
    y = pd.concat([block[upper], block[lower].iloc[::-1]], ignore_index=True)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={"width": 0},
            fill="toself",
            fillcolor=_rgba(color, opacity),
            hoverinfo="skip",
            name=name,
        )
    )


def empty_scenario_figure(message: str = "Configure a tax scenario") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper", text=message,
        showarrow=False, font={"size": 13, "color": _MUTED},
    )
    fig.update_layout(
        template="plotly_white", height=440,
        xaxis={"visible": False}, yaxis={"visible": False},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _append_anchor(block: pd.DataFrame, *, date, value) -> pd.DataFrame:
    """Prepend a single anchor point so adjacent line segments meet cleanly."""
    if block.empty or date is None or pd.isna(date) or value is None or pd.isna(value):
        return block
    anchor = {column: np.nan for column in block.columns}
    anchor["date"] = pd.Timestamp(date)
    for column in ("q05", "q16", "q50", "q84", "q95"):
        anchor[column] = float(value)
    anchor["name"] = block["name"].iloc[0] if "name" in block and len(block) else "anchor"
    return pd.concat([pd.DataFrame([anchor]), block], ignore_index=True).sort_values("date")



def _finalise_scenario_layout(
    fig: go.Figure,
    *,
    title: str,
    y_title: str | None,
    height: int,
    uirevision: str,
    bottom_margin: int,
) -> go.Figure:
    """Apply one non-overlapping title/legend contract to VAT/excise charts.

    The figure title owns the top margin.  The legend is deliberately placed
    below the plotting area so it can wrap without colliding with either the
    title or Plotly's modebar.  This is presentation-only: traces, values,
    intervals, dates and econometric objects are unchanged.
    """
    apply_theme(
        fig,
        uirevision=uirevision,
        height=height,
        y_title=y_title,
    )
    fig.update_layout(
        title={
            "text": str(title),
            "x": 0.01,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
            "font": {"size": 17, "color": _INK},
        },
        margin={
            "l": 68,
            "r": 28,
            "t": 74,
            "b": int(bottom_margin),
        },
        legend={
            "orientation": "h",
            "x": 0.0,
            "xanchor": "left",
            "y": -0.16,
            "yanchor": "top",
            "font": {"size": 10},
            "bgcolor": "rgba(255,255,255,0)",
            "borderwidth": 0,
            "traceorder": "normal",
        },
        hovermode="x unified",
        dragmode="pan",
    )
    return fig


def _selected_hicp_label(payload: Mapping | None) -> str:
    meta = dict((payload or {}).get("meta") or {})
    label = str(meta.get("hicp_label") or "").strip()
    if label:
        return label
    series = str(meta.get("hicp_series") or "").strip()
    if series:
        return series.replace("_", " ").title()
    model_id = str(meta.get("model_id") or "").strip()
    if model_id:
        return "HICP " + model_id.replace("_", " ").title()
    return "Selected HICP component"


def scenario_main_figure(payload: Mapping | None, *, fan_mode: str = "68", uirevision: str = "tax-scenario") -> go.Figure:
    """Observed -> nowcast -> forecast, with baseline/scenario split only in forecast."""
    frames = scenario_frames(payload)
    fans = frames["fans"]
    history = frames["history"]
    baseline = _slice(fans, "baseline_yoy")
    scenario = _slice(fans, "scenario_yoy")
    if baseline.empty or scenario.empty:
        return empty_scenario_figure("No paired tax-scenario paths are available")

    meta = (payload or {}).get("meta", {})
    hicp_label = _selected_hicp_label(payload)
    last_observed = pd.to_datetime(meta.get("last_observed"), errors="coerce")
    forecast_origin = pd.to_datetime(meta.get("forecast_origin"), errors="coerce")
    scenario_start = pd.to_datetime(meta.get("scenario_start"), errors="coerce")
    if pd.isna(last_observed) and not history.empty:
        last_observed = pd.Timestamp(history["date"].max())
    if pd.isna(forecast_origin):
        forecast_origin = scenario_start if not pd.isna(scenario_start) else pd.Timestamp(baseline["date"].min())

    nowcast = baseline.loc[
        (baseline["date"] > last_observed) & (baseline["date"] < forecast_origin)
    ].copy() if not pd.isna(last_observed) else baseline.iloc[0:0].copy()
    baseline_fc = baseline.loc[baseline["date"] >= forecast_origin].copy()
    scenario_fc = scenario.loc[scenario["date"] >= forecast_origin].copy()

    fig = go.Figure()
    if not history.empty:
        hist = history.loc[history["date"] <= last_observed] if not pd.isna(last_observed) else history
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["value"], mode="lines", name=f"Observed {hicp_label} inflation",
            line={"color": _INK, "width": 1.8},
            hovertemplate="%{x|%Y-%m}<br>%{y:.2f}%<extra>Observed</extra>",
        ))

    # Nowcast is a distinct object: it is predictive but lies before the true
    # forecast origin.  Showing it separately removes the apparent history/forecast gap.
    if fan_mode in {"90", "both"}:
        _band(fig, nowcast, "q05", "q95", color=_NOW, name="Nowcast 90% interval", opacity=0.09)
        _band(fig, baseline_fc, "q05", "q95", color=_BASE, name="Baseline forecast 90% interval", opacity=0.08)
        _band(fig, scenario_fc, "q05", "q95", color=_SCEN, name="Scenario forecast 90% interval", opacity=0.08)
    if fan_mode in {"68", "both"}:
        _band(fig, nowcast, "q16", "q84", color=_NOW, name="Nowcast 68% interval", opacity=0.18)
        _band(fig, baseline_fc, "q16", "q84", color=_BASE, name="Baseline forecast 68% interval", opacity=0.16)
        _band(fig, scenario_fc, "q16", "q84", color=_SCEN, name="Scenario forecast 68% interval", opacity=0.16)

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
            line={"color": _NOW, "width": 2.2},
            hovertemplate="%{x|%Y-%m}<br>%{y:.2f}%<extra>Nowcast</extra>",
        ))
        anchor_date = now_line["date"].iloc[-1]
        anchor_value = now_line["q50"].iloc[-1]

    baseline_line = _append_anchor(baseline_fc, date=anchor_date, value=anchor_value)
    scenario_line = _append_anchor(scenario_fc, date=anchor_date, value=anchor_value)
    if not baseline_line.empty:
        fig.add_trace(go.Scatter(
            x=baseline_line["date"], y=baseline_line["q50"], mode="lines", name="Baseline forecast median",
            line={"color": _BASE, "width": 2.2},
            hovertemplate="%{x|%Y-%m}<br>%{y:.2f}%<extra>Baseline</extra>",
        ))
    if not scenario_line.empty:
        fig.add_trace(go.Scatter(
            x=scenario_line["date"], y=scenario_line["q50"], mode="lines", name="Tax-scenario forecast median",
            line={"color": _SCEN, "width": 2.2, "dash": "dash"},
            hovertemplate="%{x|%Y-%m}<br>%{y:.2f}%<extra>Scenario</extra>",
        ))

    if not pd.isna(last_observed) and not nowcast.empty:
        fig.add_vline(x=last_observed, line={"color": _MUTED, "width": 1, "dash": "dash"})
    if not pd.isna(forecast_origin):
        fig.add_vline(x=forecast_origin, line={"color": _MUTED, "width": 1, "dash": "dot"})
    if not pd.isna(scenario_start) and (pd.isna(forecast_origin) or scenario_start != forecast_origin):
        fig.add_vline(x=scenario_start, line={"color": _SCEN, "width": 1, "dash": "dot"})

    return _finalise_scenario_layout(
        fig,
        title=f"{hicp_label} — inflation: baseline vs tax scenario",
        y_title="% y/y",
        height=520,
        uirevision=uirevision,
        bottom_margin=150,
    )

def scenario_impact_figure(payload: Mapping | None, *, metric: str, fan_mode: str = "68", uirevision: str = "tax-impact") -> go.Figure:
    frames = scenario_frames(payload)
    fans = frames["fans"]
    name = "level_impact_pct" if metric == "level" else "yoy_impact_pp"
    block = _slice(fans, name)
    if block.empty:
        return empty_scenario_figure("No paired impact paths are available")
    fig = go.Figure()
    if fan_mode in {"90", "both"}:
        _band(fig, block, "q05", "q95", color=_SCEN, name="90% posterior interval", opacity=0.10)
    if fan_mode in {"68", "both"}:
        _band(fig, block, "q16", "q84", color=_SCEN, name="68% posterior interval", opacity=0.20)
    fig.add_trace(go.Scatter(
        x=block["date"], y=block["q50"], mode="lines", name="Median effect",
        line={"color": _SCEN, "width": 2.3},
        hovertemplate="%{x|%Y-%m}<br>%{y:+.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.0, line={"color": TOKENS["hairline"], "width": 1})
    meta = (payload or {}).get("meta", {})
    hicp_label = _selected_hicp_label(payload)
    origin = pd.to_datetime(meta.get("forecast_origin"), errors="coerce")
    start = pd.to_datetime(meta.get("scenario_start"), errors="coerce")
    if not pd.isna(origin):
        fig.add_vline(x=origin, line={"color": _MUTED, "width": 1, "dash": "dot"})
    if not pd.isna(start) and (pd.isna(origin) or start != origin):
        fig.add_vline(x=start, line={"color": _SCEN, "width": 1, "dash": "dot"})
    if metric == "level":
        title, unit = f"{hicp_label} — price-level impact", "Scenario − baseline (%)"
    else:
        title, unit = f"{hicp_label} — year-on-year inflation impact", "Scenario − baseline (pp)"
    return _finalise_scenario_layout(
        fig,
        title=title,
        y_title=unit,
        height=390,
        uirevision=uirevision,
        bottom_margin=100,
    )


def scenario_tax_figure(payload: Mapping | None, *, uirevision: str = "tax-path") -> go.Figure:
    frames = scenario_frames(payload)
    tax = frames["tax_path"]
    history = frames["tax_history"]
    raw = frames["raw_tax"]
    if tax.empty:
        return empty_scenario_figure("No tax path is available")
    meta = (payload or {}).get("meta", {})
    hicp_label = _selected_hicp_label(payload)
    excise_unit = str(meta.get("excise_unit", "source unit"))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        subplot_titles=("VAT", "Excise"))

    if not history.empty and "applied_vat_percent" in history:
        fig.add_trace(go.Scatter(
            x=history["date"], y=history["applied_vat_percent"], mode="lines",
            name="VAT applied historically", line={"color": _MUTED, "width": 1.7, "shape": "hv"},
        ), row=1, col=1)
    if not raw.empty and "vat_percent" in raw:
        fig.add_trace(go.Scatter(
            x=raw["date"], y=raw["vat_percent"], mode="markers", name="VAT publications",
            marker={"color": _INK, "size": 5},
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=tax["date"], y=tax["baseline_vat_percent"], mode="lines", name="VAT baseline",
        line={"color": _BASE, "width": 2, "dash": "dash", "shape": "hv"},
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=tax["date"], y=tax["scenario_vat_percent"], mode="lines", name="VAT scenario",
        line={"color": _SCEN, "width": 2, "shape": "hv"},
    ), row=1, col=1)

    if not history.empty and "applied_excise" in history:
        fig.add_trace(go.Scatter(
            x=history["date"], y=history["applied_excise"], mode="lines",
            name="Excise applied historically", line={"color": _MUTED, "width": 1.7, "shape": "hv"},
        ), row=2, col=1)
    if not raw.empty and "excise" in raw:
        fig.add_trace(go.Scatter(
            x=raw["date"], y=raw["excise"], mode="markers", name="Excise publications",
            marker={"color": _INK, "size": 5},
        ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tax["date"], y=tax["baseline_excise"], mode="lines", name="Excise baseline",
        line={"color": _BASE, "width": 2, "dash": "dash", "shape": "hv"},
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tax["date"], y=tax["scenario_excise"], mode="lines", name="Excise scenario",
        line={"color": _SCEN, "width": 2, "shape": "hv"},
    ), row=2, col=1)

    origin = pd.to_datetime(meta.get("forecast_origin"), errors="coerce")
    start = pd.to_datetime(meta.get("scenario_start"), errors="coerce")
    if not pd.isna(origin):
        fig.add_vline(x=origin, line={"color": _MUTED, "width": 1, "dash": "dash"}, row="all", col=1)
    if not pd.isna(start):
        fig.add_vline(x=start, line={"color": _SCEN, "width": 1, "dash": "dot"}, row="all", col=1)
    fig.update_yaxes(title_text="VAT (%)", row=1, col=1)
    fig.update_yaxes(title_text=f"Excise ({excise_unit})", row=2, col=1)
    return _finalise_scenario_layout(
        fig,
        title=f"{hicp_label} — VAT and excise assumptions",
        y_title=None,
        height=660,
        uirevision=uirevision,
        bottom_margin=150,
    )


def scenario_kpis(payload: Mapping | None) -> dict[str, Any]:
    frames = scenario_frames(payload)
    fans = frames["fans"]
    baseline = _slice(fans, "baseline_yoy")
    scenario = _slice(fans, "scenario_yoy")
    impact = _slice(fans, "yoy_impact_pp")
    out = {
        "baseline_terminal": None,
        "scenario_terminal": None,
        "impact_terminal": None,
        "impact_low": None,
        "impact_high": None,
        "terminal_date": None,
        "n_draws": (payload or {}).get("meta", {}).get("n_draws_effective"),
    }
    if not baseline.empty and not scenario.empty and not impact.empty:
        out.update({
            "baseline_terminal": float(baseline["q50"].iloc[-1]),
            "scenario_terminal": float(scenario["q50"].iloc[-1]),
            "impact_terminal": float(impact["q50"].iloc[-1]),
            "impact_low": float(impact["q16"].iloc[-1]),
            "impact_high": float(impact["q84"].iloc[-1]),
            "terminal_date": pd.Timestamp(impact["date"].iloc[-1]),
        })
    return out


# ---------------------------------------------------------------------------
# Multi-component scenario-set helpers
# ---------------------------------------------------------------------------

SCENARIO_SET_SCHEMA_VERSION = "2.0"
_SCENARIO_AGGREGATE_KEYS = {
    "gas": "gas",
    "electricity": "electricity",
    "car_fuels_petrol": "petrol",
    "car_fuels_diesel": "diesel",
    "liquid_fuels": "liquid_fuels",
}


def _normalise_scenario_set(store: Mapping | None) -> dict[str, Any]:
    """Return the v2 scenario-set shape, accepting a legacy single payload."""
    if not store:
        return {"schema_version": SCENARIO_SET_SCHEMA_VERSION, "vintage": None, "forecast_name": None, "components": {}}
    if "components" in store:
        out = dict(store)
        out["schema_version"] = SCENARIO_SET_SCHEMA_VERSION
        out["components"] = dict(out.get("components", {}) or {})
        return out
    meta = dict(store.get("meta", {}) or {})
    model_id = str(meta.get("model_id", ""))
    return {
        "schema_version": SCENARIO_SET_SCHEMA_VERSION,
        "vintage": meta.get("vintage"),
        "forecast_name": meta.get("forecast_name"),
        "components": {model_id: dict(store)} if model_id else {},
    }


def scenario_set_components(store: Mapping | None) -> dict[str, dict]:
    return dict(_normalise_scenario_set(store).get("components", {}) or {})


def scenario_set_payload(store: Mapping | None, model_id: str | None = None) -> dict | None:
    components = scenario_set_components(store)
    if not components:
        return None
    if model_id is not None and str(model_id) in components:
        return components[str(model_id)]
    return next(iter(components.values()))


def upsert_scenario_component(store: Mapping | None, payload: Mapping, *, vintage: str | None = None, forecast_name: str | None = None) -> dict[str, Any]:
    out = _normalise_scenario_set(store)
    meta = dict(payload.get("meta", {}) or {})
    model_id = str(meta.get("model_id", ""))
    if not model_id:
        raise ValueError("Scenario payload is missing meta.model_id.")
    target_vintage = str(vintage or meta.get("vintage") or out.get("vintage") or "") or None
    target_forecast = str(forecast_name or meta.get("forecast_name") or out.get("forecast_name") or "") or None
    if out.get("vintage") not in (None, target_vintage) or out.get("forecast_name") not in (None, target_forecast):
        out["components"] = {}
    out["vintage"] = target_vintage
    out["forecast_name"] = target_forecast
    out["components"][model_id] = dict(payload)
    return out


def remove_scenario_component(store: Mapping | None, model_id: str) -> dict[str, Any]:
    out = _normalise_scenario_set(store)
    out["components"].pop(str(model_id), None)
    return out


def clear_scenario_set(*, vintage: str | None = None, forecast_name: str | None = None) -> dict[str, Any]:
    return {"schema_version": SCENARIO_SET_SCHEMA_VERSION, "vintage": None if vintage is None else str(vintage), "forecast_name": None if forecast_name is None else str(forecast_name), "components": {}}


def scenario_set_signature(store: Mapping | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, payload in sorted(scenario_set_components(store).items()):
        meta = dict(payload.get("meta", {}) or {})
        rows.append({
            "model_id": str(model_id), "run_id": meta.get("run_id"), "scenario_start": meta.get("scenario_start"),
            "vat_delta_pp": float(meta.get("vat_delta_pp", 0.0) or 0.0),
            "excise_delta": float(meta.get("excise_delta", 0.0) or 0.0), "excise_unit": str(meta.get("excise_unit", "")),
        })
    return rows


def scenario_set_summary(store: Mapping | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, payload in sorted(scenario_set_components(store).items()):
        meta = dict(payload.get("meta", {}) or {})
        rows.append({
            "model_id": str(model_id), "label": str(meta.get("hicp_label") or model_id.replace("_", " ").title()),
            "scenario_start": meta.get("scenario_start"), "vat_delta_pp": float(meta.get("vat_delta_pp", 0.0) or 0.0),
            "excise_delta": float(meta.get("excise_delta", 0.0) or 0.0), "excise_unit": str(meta.get("excise_unit", "")),
            "run_id": meta.get("run_id"), "n_draws_effective": meta.get("n_draws_effective"),
        })
    return rows


def scenario_set_to_tax_scenarios(store: Mapping | None) -> dict[str, dict]:
    """Translate all active component payloads to run_aggregate's tax contract."""
    out: dict[str, dict] = {}
    for model_id, payload in scenario_set_components(store).items():
        meta = dict(payload.get("meta", {}) or {})
        vat_delta = float(meta.get("vat_delta_pp", 0.0) or 0.0)
        excise_delta = float(meta.get("excise_delta", 0.0) or 0.0)
        if abs(vat_delta) < 1e-15 and abs(excise_delta) < 1e-15:
            continue
        key = _SCENARIO_AGGREGATE_KEYS.get(str(model_id))
        if key is None:
            continue
        tax = scenario_frames(payload)["tax_path"]
        if tax.empty:
            continue
        tax = tax.set_index("date").sort_index()
        start = pd.Timestamp(meta["scenario_start"])
        active = tax.loc[tax.index >= start]
        if active.empty:
            continue
        scenario: dict[str, Any] = {"start_date": start}
        if abs(vat_delta) >= 1e-15:
            scenario["vat_percent"] = active["scenario_vat_percent"].astype(float).copy()
        if abs(excise_delta) >= 1e-15:
            scenario["excise"] = active["scenario_excise"].astype(float).copy()
            scenario["excise_unit"] = str(meta.get("excise_unit", "source unit"))
        out[key] = scenario
    return out


__all__ = [
    "scenario_payload",
    "scenario_frames",
    "scenario_main_figure",
    "scenario_impact_figure",
    "scenario_tax_figure",
    "scenario_kpis",
    "empty_scenario_figure",
    "SCENARIO_SET_SCHEMA_VERSION",
    "scenario_set_components",
    "scenario_set_payload",
    "upsert_scenario_component",
    "remove_scenario_component",
    "clear_scenario_set",
    "scenario_set_signature",
    "scenario_set_summary",
    "scenario_set_to_tax_scenarios",
]
