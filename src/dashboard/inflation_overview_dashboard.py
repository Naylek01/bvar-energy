"""Cross-domain Inflation Dashboard overview.

The Overview is intentionally a display layer.  It resolves the production
vintage, reads saved Headline/Energy artefacts, materialises display-only fitted
caches when needed, and never launches a BVAR estimation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from dashboard_snapshot_cache import (
    get as snapshot_get,
    get_or_build as snapshot_get_or_build,
)
from energy_bvar_dashboard_aggregate_ext import (
    COMPONENT_LABELS as ENERGY_COMPONENT_LABELS,
    COMPONENT_ORDER as ENERGY_COMPONENT_ORDER,
    build_historical_contribution_frame,
)
from energy_bvar_aggregate import (
    historical_model_component_reconstruction,
    load_aggregation_inputs,
    model_energy_weights,
    yoy_contributions_from_terms,
)
from energy_bvar_fitted import load_energy_aggregate_fitted
from energy_bvar_theme import (
    TOKENS,
    INFLATION_COLORS,
    apply_inflation_figure_style,
    graph_config,
)
from headline_bvar_display import (
    DISPLAY_FILENAME as HEADLINE_DISPLAY_FILENAME,
    build_headline_display,
    materialize_headline_core,
)
from headline_bvar_fitted import materialize_headline_fitted
from inflation_bvar_display import (
    DISPLAY_FILENAME as ENERGY_DISPLAY_FILENAME,
    build_aggregate_display,
    load_display_artifact,
)
from inflation_bvar_registry import list_aggregates, list_forecasts, list_runs

_OVERVIEW_GRAPH_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset+autosize",
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

OVERVIEW_OVERLAYS = (
    "seven_energy_bvars",
    "energy_aggregate_fitted",
    "headline_fitted",
    "core_aggregate_history",
)
SOURCE_BVAR_LABELS = {
    "gas": "Gas BVAR fitted",
    "electricity": "Electricity BVAR fitted",
    "heat_energy": "Heat energy BVAR fitted",
    "solid_fuels": "Solid fuels BVAR fitted",
    "petrol": "Petrol BVAR fitted",
    "diesel": "Diesel BVAR fitted",
    "liquid_fuels": "Liquid fuels BVAR fitted",
}
HEADLINE_COMPONENT_LABELS = {
    "hicp_energy": "Energy",
    "hicp_food": "Food",
    "hicp_neig": "NEIG",
    "hicp_services": "Services",
}
CORE_COMPONENT_LABELS = {
    "hicp_neig": "NEIG",
    "hicp_services": "Services",
}


def overview_page() -> html.Div:
    return html.Div(
        [
            dcc.Store(id="overview-data-store", storage_type="memory"),
            dcc.Download(id="overview-download-results-file"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Inflation overview", className="page-title"),
                            html.P(
                                "Official Headline, HICP Energy and Core inflation on one chart, with optional saved-model fitted histories.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div("Production vintage", className="eyebrow"),
                            html.Div(id="overview-vintage-label", className="stat-value"),
                            dcc.Link("Change in Data →", href="/data", className="refresh-button"),
                            html.Button(
                                "Download results (.xlsx)",
                                id="overview-download-results",
                                n_clicks=0,
                                disabled=True,
                                className="refresh-button",
                            ),
                            html.Div(
                                "Persisted display artifacts only · observed history + stored nowcast/forecast · no recomputation.",
                                className="control-help",
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="page-heading-row",
            ),
            html.Div(id="overview-status", className="selection-banner"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Inflation outlook", className="eyebrow"),
                            html.H3(
                                "Latest observation · nowcast · three-month outlook",
                                className="panel-title",
                            ),
                            html.P(
                                "All rows use one common monthly display calendar. Predictive cells show posterior mean, median and q05–q95. Δ columns show the change in posterior mean versus the immediately preceding displayed period, in percentage points; positive changes are green and negative changes red.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dcc.Loading(
                        html.Div(id="overview-outlook-table"),
                        type="circle",
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("OVERLAYS", className="control-label"),
                            dcc.Checklist(
                                id="overview-overlays",
                                options=[
                                    {"label": "7 Energy BVAR fitted", "value": "seven_energy_bvars"},
                                    {"label": "Energy aggregate fitted", "value": "energy_aggregate_fitted"},
                                    {"label": "Headline BVAR fitted", "value": "headline_fitted"},
                                    {"label": "Core aggregate full history", "value": "core_aggregate_history"},
                                ],
                                value=[],
                                className="fan-radio",
                            ),
                        ],
                        className="control-block",
                    ),
                    html.Div(
                        [
                            html.Label("Forecast horizon", className="selector-label"),
                            dcc.Dropdown(
                                id="overview-forecast-horizon",
                                options=[
                                    {"label": "3 months", "value": 3},
                                    {"label": "6 months", "value": 6},
                                    {"label": "12 months", "value": 12},
                                ],
                                value=12,
                                clearable=False,
                            ),
                        ],
                        className="selector-block",
                    ),
                    html.Div(
                        [
                            html.Label("Fan", className="selector-label"),
                            dcc.Dropdown(
                                id="overview-forecast-fan",
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
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Observed inflation", className="eyebrow"),
                            html.H3("Headline · Energy · Core", className="panel-title"),
                            html.P(
                                "Observed lines show the full official history; saved nowcast and forecast continue directly on the same chart. Optional fitted curves are posterior means.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dcc.Loading(
                        dcc.Graph(id="overview-main-graph", config=_OVERVIEW_GRAPH_CONFIG),
                        type="circle",
                    ),
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Historical contribution decomposition", className="eyebrow"),
                                    html.H3("Exact observed accounting contributions", className="panel-title"),
                                    html.P(
                                        "Exact observed contributions continue directly into the saved posterior-mean nowcast and forecast contributions; the vertical marker is the forecast origin.",
                                        className="panel-subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Domain", className="selector-label"),
                                            dcc.Dropdown(
                                                id="overview-decomp-domain",
                                                options=[
                                                    {"label": "Headline", "value": "headline"},
                                                    {"label": "Energy", "value": "energy"},
                                                    {"label": "Core", "value": "core"},
                                                ],
                                                value="headline",
                                                clearable=False,
                                            ),
                                        ],
                                        className="selector-block",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Display", className="selector-label"),
                                            dcc.RadioItems(
                                                id="overview-decomp-display",
                                                options=[
                                                    {"label": "Stacked bars", "value": "bars"},
                                                    {"label": "Lines", "value": "lines"},
                                                ],
                                                value="bars",
                                                inline=True,
                                                className="fan-radio",
                                            ),
                                        ],
                                        className="selector-block",
                                    ),
                                ],
                                className="chart-controls",
                            ),
                        ],
                        className="page-heading-row",
                    ),
                    dcc.Loading(
                        dcc.Graph(id="overview-decomp-graph", config=_OVERVIEW_GRAPH_CONFIG),
                        type="circle",
                    ),
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "Model & aggregation specifications",
                                className="eyebrow",
                            ),
                            html.H3(
                                "What sits behind the Overview",
                                className="panel-title",
                            ),
                            html.P(
                                "Exact saved-run BVAR contracts followed by the two derived inflation aggregates.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(id="overview-model-spec-table"),
                ],
                className="panel",
            ),
        ],
        className="page-body",
    )


def _snapshot_registry_table(name: str) -> pd.DataFrame:
    frame = snapshot_get("registry_table", str(name))
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _latest_headline_run(results_root: Path, registry_path, vintage: str) -> Path:
    runs = _snapshot_registry_table("runs")
    if not runs.empty:
        runs = runs.loc[
            runs["model_id"].astype(str).eq("headline_joint")
            & runs["vintage"].astype(str).eq(str(vintage))
            & runs["status"].astype(str).eq("complete")
        ].copy()
    forecasts = _snapshot_registry_table("forecasts")
    if not forecasts.empty:
        forecasts = forecasts.loc[
            forecasts["model_id"].astype(str).eq("headline_joint")
            & forecasts["vintage"].astype(str).eq(str(vintage))
            & forecasts["forecast_name"].astype(str).eq("unconditional")
        ].copy()
    if runs.empty or forecasts.empty:
        raise FileNotFoundError(f"No complete Headline unconditional run for vintage {vintage}.")
    valid = set(forecasts["run_id"].astype(str))
    runs = runs.loc[runs["run_id"].astype(str).isin(valid)].copy()
    if runs.empty:
        raise FileNotFoundError(f"No Headline run with a valid unconditional forecast for vintage {vintage}.")
    for col in ("promoted",):
        if col not in runs:
            runs[col] = False
    if "created_at_utc" not in runs:
        runs["created_at_utc"] = ""
    runs = runs.sort_values(["promoted", "created_at_utc", "run_id"], ascending=[False, False, True])
    run_id = str(runs.iloc[0]["run_id"])
    run_dir = results_root / "headline_joint" / str(vintage) / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    return run_dir


def _latest_energy_aggregate(registry_path, vintage: str) -> Path:
    rows = _snapshot_registry_table("aggregates")
    if not rows.empty:
        rows = rows.loc[
            rows["vintage"].astype(str).eq(str(vintage))
        ].copy()
    if rows.empty:
        raise FileNotFoundError(f"No HICP Energy aggregate for vintage {vintage}.")
    rows = rows.loc[rows["status"].astype(str).eq("complete")].copy()
    if rows.empty:
        raise FileNotFoundError(f"No complete HICP Energy aggregate for vintage {vintage}.")
    if "promoted" not in rows:
        rows["promoted"] = False
    if "created_at_utc" not in rows:
        rows["created_at_utc"] = ""
    rows = rows.sort_values(["promoted", "created_at_utc", "aggregate_run_id"], ascending=[False, False, True])
    directory = Path(str(rows.iloc[0]["directory"]))
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return directory


def _headline_display(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "forecasts" / "unconditional" / HEADLINE_DISPLAY_FILENAME
    if not path.is_file():
        build_headline_display(run_dir, forecast_name="unconditional", overwrite=False)
    frame = pd.read_parquet(path)
    record = frame["record_type"].astype(str)
    if not record.eq("fitted").any():
        materialize_headline_fitted(run_dir, forecast_name="unconditional")
        frame = pd.read_parquet(path)
        record = frame["record_type"].astype(str)
    core_ready = (
        frame["series"].astype(str).eq("hicp_core")
        & record.isin(["history", "fan"])
    ).any()
    if not core_ready:
        materialize_headline_core(run_dir, forecast_name="unconditional")
        frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _energy_history(project_root: Path, vintage: str) -> pd.DataFrame:
    path = project_root / "data" / "processed" / str(vintage) / "hicp_indices_monthly.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if "hicp_energy" not in frame:
        raise KeyError(f"{path} has no hicp_energy column.")
    level = pd.to_numeric(frame["hicp_energy"], errors="coerce")
    yoy = 100.0 * (level / level.shift(12) - 1.0)
    out = yoy.dropna().rename("value").reset_index()
    out["series"] = "hicp_energy"
    return out[["date", "series", "value"]]


def _simple_rows(frame: pd.DataFrame, *, record_type: str, series: str, metric: str, value_col: str = "value") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "series", "value"])
    mask = (
        frame["record_type"].astype(str).eq(record_type)
        & frame["series"].astype(str).eq(series)
        & frame["metric"].astype(str).eq(metric)
    )
    out = frame.loc[mask].copy().sort_values("date")
    if out.empty:
        return pd.DataFrame(columns=["date", "series", "value"])
    col = value_col
    if col not in out or not pd.to_numeric(out[col], errors="coerce").notna().any():
        for candidate in ("posterior_mean", "value", "q50"):
            if candidate in out and pd.to_numeric(out[candidate], errors="coerce").notna().any():
                col = candidate
                break
    result = out[["date"]].copy()
    result["series"] = series
    result["value"] = pd.to_numeric(out[col], errors="coerce").to_numpy()
    return result.dropna(subset=["date", "value"])



def _predictive_rows(
    frame: pd.DataFrame,
    *,
    record_type: str,
    metric: str,
    series: str | None = None,
    basis: str | None = None,
) -> pd.DataFrame:
    """Preserve saved predictive-regime metadata and posterior summaries."""
    keep = [
        "date", "series", "segment", "is_future",
        "value", "q05", "q16", "q50", "q84", "q95",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=keep)

    mask = (
        frame["record_type"].astype(str).eq(str(record_type))
        & frame["metric"].astype(str).eq(str(metric))
    )
    if series is not None:
        mask &= frame["series"].astype(str).eq(str(series))
    if basis is not None and "basis" in frame.columns:
        selected = frame["basis"].astype(str).eq(str(basis))
        if (mask & selected).any():
            mask &= selected

    out = frame.loc[mask].copy().sort_values("date")
    if out.empty:
        return pd.DataFrame(columns=keep)

    if (
        "value" not in out.columns
        or not pd.to_numeric(out["value"], errors="coerce").notna().any()
    ):
        for candidate in ("posterior_mean", "q50"):
            if (
                candidate in out.columns
                and pd.to_numeric(out[candidate], errors="coerce").notna().any()
            ):
                out["value"] = pd.to_numeric(out[candidate], errors="coerce")
                break

    for column in keep:
        if column not in out.columns:
            if column == "segment":
                out[column] = ""
            elif column == "is_future":
                out[column] = False
            else:
                out[column] = np.nan

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["segment"] = out["segment"].fillna("").astype(str).str.lower()
    out["is_future"] = out["is_future"].fillna(False).astype(bool)
    for column in ("value", "q05", "q16", "q50", "q84", "q95"):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    return (
        out[keep]
        .dropna(subset=["date", "value"])
        .sort_values(["date", "series"])
        .reset_index(drop=True)
    )


def _split_predictive_dates(
    rows: pd.DataFrame,
    *,
    last_observed: pd.Timestamp | None,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | None]:
    """Saved nowcast + true forecast, with H applied to forecast dates only."""
    if rows is None or rows.empty:
        empty = (
            pd.DataFrame(columns=["date", "series", "segment", "is_future", "value"])
            if rows is None
            else rows.iloc[0:0].copy()
        )
        return empty, empty.copy(), None

    work = rows.sort_values(["date", "series"]).copy()
    segment = work["segment"].fillna("").astype(str).str.lower()
    future_flag = work["is_future"].fillna(False).astype(bool)

    if future_flag.any():
        forecast = work.loc[future_flag].copy()
    else:
        forecast = work.loc[segment.eq("forecast")].copy()

    forecast_origin = (
        pd.Timestamp(forecast["date"].min())
        if not forecast.empty
        else None
    )

    nowcast = work.loc[segment.eq("nowcast") & ~future_flag].copy()
    if (
        nowcast.empty
        and last_observed is not None
        and forecast_origin is not None
    ):
        nowcast = work.loc[
            (work["date"] > pd.Timestamp(last_observed))
            & (work["date"] < forecast_origin)
            & ~future_flag
        ].copy()

    if last_observed is not None and not nowcast.empty:
        nowcast = nowcast.loc[
            nowcast["date"] > pd.Timestamp(last_observed)
        ].copy()

    if not forecast.empty:
        future_dates = sorted(
            pd.DatetimeIndex(forecast["date"].dropna().unique())
        )[: max(1, int(horizon))]
        forecast = forecast.loc[forecast["date"].isin(future_dates)].copy()

    return (
        nowcast.sort_values(["date", "series"]),
        forecast.sort_values(["date", "series"]),
        forecast_origin,
    )


def _rgba(hex_color: str, alpha: float) -> str:
    value = str(hex_color).lstrip("#")
    if len(value) != 6:
        return f"rgba(37,99,235,{float(alpha):.3f})"
    r, g, b = (int(value[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{float(alpha):.3f})"


def _fan_band(
    fig: go.Figure,
    rows: pd.DataFrame,
    lower: str,
    upper: str,
    *,
    name: str,
    color: str,
    alpha: float,
) -> None:
    if rows is None or rows.empty:
        return
    block = rows.dropna(subset=["date", lower, upper]).sort_values("date")
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
            fillcolor=_rgba(color, alpha),
            name=name,
            hoverinfo="skip",
            legendgroup="overview-predictive-fan",
        )
    )


def _anchor_path(
    rows: pd.DataFrame,
    anchor_date,
    anchor_value,
) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame() if rows is None else rows
    block = rows.sort_values("date").copy()
    if anchor_date is None or anchor_value is None or pd.isna(anchor_value):
        return block
    anchor = {column: np.nan for column in block.columns}
    anchor["date"] = pd.Timestamp(anchor_date)
    anchor["series"] = block["series"].iloc[0] if "series" in block else ""
    anchor["segment"] = "anchor"
    anchor["is_future"] = False
    for column in ("value", "q05", "q16", "q50", "q84", "q95"):
        if column in block.columns:
            anchor[column] = float(anchor_value)
    return pd.concat([pd.DataFrame([anchor]), block], ignore_index=True)


def _contribution_rows(frame: pd.DataFrame, *, record_type: str, metric: str, series: Iterable[str]) -> pd.DataFrame:
    wanted = set(map(str, series))
    mask = (
        frame["record_type"].astype(str).eq(record_type)
        & frame["metric"].astype(str).eq(metric)
        & frame["series"].astype(str).isin(wanted)
    )
    out = frame.loc[mask, ["date", "series", "value"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["date", "value"]).sort_values(["date", "series"])


def _core_reconstruction(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "headline_core_reconstructed_history.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["date", "series", "value"])
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if frame.empty:
        return pd.DataFrame(columns=["date", "series", "value"])
    level = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    yoy = 100.0 * (level / level.shift(12) - 1.0)
    out = yoy.dropna().rename("value").reset_index()
    out["series"] = "hicp_core_reconstructed"
    return out[["date", "series", "value"]]


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    out = frame.copy()
    if "date" in out:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").map(
            lambda x: None if pd.isna(x) else pd.Timestamp(x).isoformat()
        )
    return out.to_dict("records")


def _records_frame(records: list[dict] | None) -> pd.DataFrame:
    frame = pd.DataFrame(records or [])
    if not frame.empty and "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


_OVERVIEW_COMPONENT_RUN_IDS: dict[str, str] = {
    "gas": "gas",
    "electricity": "electricity",
    "heat_energy": "heat_energy",
    "solid_fuels": "solid_fuels",
    "petrol": "car_fuels_petrol",
    "diesel": "car_fuels_diesel",
    "liquid_fuels": "liquid_fuels",
}

_OVERVIEW_MODEL_LABELS: dict[str, str] = {
    "headline_joint": "Headline joint",
    "gas": "Gas",
    "electricity": "Electricity",
    "heat_energy": "Heat energy",
    "solid_fuels": "Solid fuels",
    "car_fuels_petrol": "Petrol",
    "car_fuels_diesel": "Diesel",
    "liquid_fuels": "Liquid fuels",
}

_HEADLINE_VARIABLE_LABELS: dict[str, str] = {
    "hicp_energy": "Energy",
    "hicp_food": "Food",
    "hicp_neig": "NEIG",
    "hicp_services": "Services",
}


def _read_json_object(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return value


def _saved_frequency(meta: dict) -> str:
    raw = str(meta.get("frequency") or "").strip()
    if not raw:
        return "—"
    label = raw.replace("_", " ").title()
    calendar = str(meta.get("calendar_rule") or "").strip()
    if raw.lower() == "weekly" and calendar:
        return f"{label} · {calendar}"
    return label


def _saved_ordering(meta: dict, *, headline: bool = False) -> str:
    if headline:
        values = meta.get("native_variables")
        if not isinstance(values, (list, tuple)) or not values:
            values = meta.get("variables")
    else:
        values = meta.get("variables")
    if not isinstance(values, (list, tuple)) or not values:
        return "metadata unavailable"
    names = [str(value) for value in values]
    if headline:
        names = [_HEADLINE_VARIABLE_LABELS.get(name, name) for name in names]
    return " → ".join(names)


def _saved_lag(meta: dict) -> str:
    try:
        return f"p = {int(meta.get('p'))}"
    except (TypeError, ValueError):
        return "—"


def _saved_model_method(meta: dict, *, headline: bool = False) -> str:
    model_class = str(meta.get("model_class") or "").strip()
    if model_class:
        return model_class
    if headline:
        exog = meta.get("exog_names")
        if isinstance(exog, (list, tuple)) and exog:
            return "Joint BVAR · saved seasonal/exogenous design"
        return "Joint BVAR · saved posterior fitted path"
    return "Saved BVAR · posterior fitted path"


def _component_run_directories_for_overview(
    aggregate_directory: Path,
) -> dict[str, Path]:
    """Resolve exact component runs recorded by the displayed Energy aggregate."""
    meta_path = aggregate_directory / "metadata.json"
    if not meta_path.is_file():
        fallback = aggregate_directory / "aggregate_config.json"
        meta_path = fallback if fallback.is_file() else meta_path
    aggregate_meta = _read_json_object(meta_path)

    stores = dict(aggregate_meta.get("component_forecast_stores", {}) or {})
    if not stores:
        raise FileNotFoundError(
            "Energy aggregate metadata does not record component_forecast_stores."
        )

    vintage = str(
        aggregate_meta.get("vintage") or aggregate_directory.parent.name
    )
    results_root = aggregate_directory.parents[2]

    out: dict[str, Path] = {}
    for aggregate_key, canonical_model_id in _OVERVIEW_COMPONENT_RUN_IDS.items():
        raw = stores.get(aggregate_key)
        if raw is None:
            raw = stores.get(canonical_model_id)
        if raw is None:
            continue

        store_path = Path(str(raw))
        run_id = ""
        try:
            run_id = store_path.parents[1].name
        except IndexError:
            pass

        candidate = (
            store_path.parents[1]
            if len(store_path.parents) >= 2
            else None
        )
        if (candidate is None or not candidate.is_dir()) and run_id:
            candidate = results_root / canonical_model_id / vintage / run_id
        if candidate is not None and (candidate / "metadata.json").is_file():
            out[canonical_model_id] = candidate
    return out


def _overview_model_spec_rows(
    headline_run: Path,
    aggregate_directory: Path | None,
) -> list[dict]:
    """Eight saved BVAR rows plus HICP Energy and Core derived aggregates."""
    rows: list[dict] = []

    try:
        meta = _read_json_object(headline_run / "metadata.json")
        rows.append(
            {
                "object": "Headline joint",
                "nature": "BVAR",
                "frequency": _saved_frequency(meta),
                "ordering": _saved_ordering(meta, headline=True),
                "lag": _saved_lag(meta),
                "method": _saved_model_method(meta, headline=True),
                "derived": False,
            }
        )
    except Exception as exc:
        rows.append(
            {
                "object": "Headline joint",
                "nature": "BVAR",
                "frequency": "—",
                "ordering": "metadata unavailable",
                "lag": "—",
                "method": f"Saved-run metadata unavailable: {exc}",
                "derived": False,
            }
        )

    component_dirs: dict[str, Path] = {}
    if aggregate_directory is not None:
        try:
            component_dirs = _component_run_directories_for_overview(
                aggregate_directory
            )
        except Exception:
            component_dirs = {}

    for model_id in (
        "gas",
        "electricity",
        "heat_energy",
        "solid_fuels",
        "car_fuels_petrol",
        "car_fuels_diesel",
        "liquid_fuels",
    ):
        run_dir = component_dirs.get(model_id)
        if run_dir is None:
            rows.append(
                {
                    "object": _OVERVIEW_MODEL_LABELS[model_id],
                    "nature": "BVAR",
                    "frequency": "—",
                    "ordering": "metadata unavailable",
                    "lag": "—",
                    "method": "Exact aggregate-source run metadata unavailable",
                    "derived": False,
                }
            )
            continue
        try:
            meta = _read_json_object(run_dir / "metadata.json")
            rows.append(
                {
                    "object": _OVERVIEW_MODEL_LABELS[model_id],
                    "nature": "BVAR",
                    "frequency": _saved_frequency(meta),
                    "ordering": _saved_ordering(meta),
                    "lag": _saved_lag(meta),
                    "method": _saved_model_method(meta),
                    "derived": False,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "object": _OVERVIEW_MODEL_LABELS[model_id],
                    "nature": "BVAR",
                    "frequency": "—",
                    "ordering": "metadata unavailable",
                    "lag": "—",
                    "method": f"Saved-run metadata unavailable: {exc}",
                    "derived": False,
                }
            )

    rows.extend(
        [
            {
                "object": "HICP Energy",
                "nature": "AGGREGATE",
                "frequency": "Monthly",
                "ordering": (
                    "Petrol + Diesel → Car fuels; then + Liquid fuels + Gas "
                    "+ Electricity + Heat energy + Solid fuels"
                ),
                "lag": "—",
                "method": (
                    "Draw-wise Laspeyres · annual HICP weights · "
                    "previous-December chain-link. Weekly fuel paths are "
                    "converted/rebased to monthly HICP before aggregation."
                ),
                "derived": True,
            },
            {
                "object": "Core HICP",
                "nature": "AGGREGATE",
                "frequency": "Monthly",
                "ordering": "NEIG + Services",
                "lag": "—",
                "method": (
                    "Draw-wise native-level aggregation · annual Core weights "
                    "renormalised inside the two-component basket · "
                    "previous-December chain-link. Official TOT_X_NRG_FOOD "
                    "is the validation series."
                ),
                "derived": True,
            },
        ]
    )
    return rows


def overview_model_spec_table(bundle: dict | None):
    rows = [] if not bundle else list(bundle.get("model_spec_rows") or [])
    if not rows:
        return html.Div(
            "Model specification metadata is unavailable for this Overview.",
            className="selection-banner",
        )

    header = {
        "textAlign": "left",
        "padding": "8px 10px",
        "fontSize": "10px",
        "fontWeight": 700,
        "letterSpacing": "0.055em",
        "textTransform": "uppercase",
        "color": TOKENS["muted"],
        "background": "rgba(17, 24, 39, 0.025)",
        "borderBottom": f"1px solid {TOKENS['hairline']}",
        "whiteSpace": "nowrap",
    }
    cell = {
        "padding": "8px 10px",
        "fontSize": "11px",
        "lineHeight": 1.35,
        "verticalAlign": "top",
        "borderBottom": f"1px solid {TOKENS['hairline']}",
    }
    badge = {
        "display": "inline-block",
        "padding": "2px 7px",
        "borderRadius": "999px",
        "fontSize": "9px",
        "fontWeight": 750,
        "letterSpacing": "0.035em",
        "whiteSpace": "nowrap",
        "background": "rgba(17, 24, 39, 0.055)",
        "color": TOKENS["ink"],
    }
    aggregate_badge = {
        **badge,
        "background": "rgba(11, 122, 92, 0.10)",
        "color": "#0B7A5C",
    }

    body = []
    first_aggregate_seen = False
    for row in rows:
        derived = bool(row.get("derived"))
        row_style = (
            {"background": "rgba(11, 122, 92, 0.025)"}
            if derived
            else {}
        )
        if derived and not first_aggregate_seen:
            row_style["borderTop"] = f"2px solid {TOKENS['hairline']}"
            first_aggregate_seen = True

        body.append(
            html.Tr(
                [
                    html.Td(
                        html.Div(
                            str(row.get("object") or "—"),
                            style={"fontWeight": 650},
                        ),
                        style=cell,
                    ),
                    html.Td(
                        html.Span(
                            str(row.get("nature") or "—"),
                            style=aggregate_badge if derived else badge,
                        ),
                        style=cell,
                    ),
                    html.Td(
                        str(row.get("frequency") or "—"),
                        style={**cell, "whiteSpace": "nowrap"},
                    ),
                    html.Td(
                        html.Code(
                            str(row.get("ordering") or "—"),
                            style={
                                "fontFamily": (
                                    "ui-monospace, SFMono-Regular, Menlo, "
                                    "Consolas, monospace"
                                ),
                                "fontSize": "10px",
                                "whiteSpace": "normal",
                                "wordBreak": "break-word",
                                "background": "transparent",
                                "padding": 0,
                            },
                        ),
                        style={**cell, "minWidth": "320px"},
                    ),
                    html.Td(
                        html.Span(str(row.get("lag") or "—"), style=badge),
                        style={**cell, "whiteSpace": "nowrap"},
                    ),
                    html.Td(
                        str(row.get("method") or "—"),
                        style={**cell, "minWidth": "290px"},
                    ),
                ],
                style=row_style,
            )
        )

    return html.Div(
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Object", style=header),
                            html.Th("Nature", style=header),
                            html.Th("Frequency", style=header),
                            html.Th("Variables / inputs", style=header),
                            html.Th("Lag", style=header),
                            html.Th("Method / construction", style=header),
                        ]
                    )
                ),
                html.Tbody(body),
            ],
            style={
                "width": "100%",
                "borderCollapse": "separate",
                "borderSpacing": 0,
            },
        ),
        style={
            "overflowX": "auto",
            "border": f"1px solid {TOKENS['hairline']}",
            "borderRadius": "12px",
            "background": "#FFFFFF",
        },
    )



_OUTLOOK_HEADLINE_SERIES = (
    "hicp_energy",
    "hicp_food",
    "hicp_neig",
    "hicp_services",
)
_OUTLOOK_ENERGY_COMPONENTS = (
    "car_fuels",
    "liquid_fuels",
    "gas",
    "electricity",
    "heat_energy",
    "solid_fuels",
)
_OUTLOOK_ENERGY_MODEL_CHILDREN = (
    "car_fuels_petrol",
    "car_fuels_diesel",
)
_OUTLOOK_GRID = (
    "minmax(205px,1.45fr) 104px 104px "
    "minmax(118px,1fr) "
    "repeat(4,minmax(118px,1fr) 76px)"
)


def _multi_history_rows(
    frame: pd.DataFrame,
    *,
    series: Iterable[str],
    metric: str = "yoy",
) -> pd.DataFrame:
    wanted = set(map(str, series))
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "series", "value"])
    mask = (
        frame["record_type"].astype(str).eq("history")
        & frame["metric"].astype(str).eq(str(metric))
        & frame["series"].astype(str).isin(wanted)
    )
    out = frame.loc[mask, ["date", "series", "value"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["date", "value"]).sort_values(["series", "date"])


def _summary_rows_from_cube(
    paths: np.ndarray,
    dates: Iterable[pd.Timestamp],
    names: Iterable[str],
) -> pd.DataFrame:
    """Summarise already-saved draw paths; never forecast or re-estimate."""
    values = np.asarray(paths, dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)), name="date")
    names = [str(x) for x in names]
    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3:
        raise ValueError(f"Outlook paths must be draw x date x series; got {values.shape}.")
    if values.shape[1] != len(dates) or values.shape[2] != len(names):
        raise ValueError(
            "Outlook path dimensions do not match saved date/name metadata: "
            f"paths={values.shape}, dates={len(dates)}, names={len(names)}."
        )
    q = np.nanquantile(values, (0.05, 0.50, 0.95), axis=0)
    mean = np.nanmean(values, axis=0)
    rows: list[dict] = []
    for j, name in enumerate(names):
        for t, date in enumerate(dates):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "series": name,
                    "value": float(mean[t, j]),
                    "q05": float(q[0, t, j]),
                    "q50": float(q[1, t, j]),
                    "q95": float(q[2, t, j]),
                }
            )
    return pd.DataFrame(rows)


def _energy_outlook_predictive(aggregate_dir: Path) -> pd.DataFrame:
    """Read compact Energy component outlook directly from the saved aggregate."""
    aggregate_dir = Path(aggregate_dir)
    meta_path = aggregate_dir / "metadata.json"
    draws_path = aggregate_dir / "aggregate_draws.npz"
    if not meta_path.is_file() or not draws_path.is_file():
        return pd.DataFrame(columns=["date", "series", "value", "q05", "q50", "q95"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(draws_path, allow_pickle=False) as archive:
        if "aggregate_dates" in archive.files:
            dates = pd.DatetimeIndex(pd.to_datetime(archive["aggregate_dates"]), name="date")
        else:
            dates = pd.DatetimeIndex(pd.to_datetime(meta.get("path_dates", [])), name="date")
        blocks: list[pd.DataFrame] = []
        if "component_yoy_paths" in archive.files:
            names = list(meta.get("component_index_names") or meta.get("components") or [])
            values = np.asarray(archive["component_yoy_paths"], dtype=float)
            if names and values.shape[-1] == len(names):
                blocks.append(_summary_rows_from_cube(values, dates, names))
        if "model_hicp_yoy_paths" in archive.files:
            names = list(meta.get("model_hicp_names") or [])
            values = np.asarray(archive["model_hicp_yoy_paths"], dtype=float)
            if names and values.shape[-1] == len(names):
                model = _summary_rows_from_cube(values, dates, names)
                model = model.loc[
                    model["series"].astype(str).isin(_OUTLOOK_ENERGY_MODEL_CHILDREN)
                ].copy()
                if not model.empty:
                    blocks.append(model)
    if not blocks:
        return pd.DataFrame(columns=["date", "series", "value", "q05", "q50", "q95"])
    return pd.concat(blocks, ignore_index=True, sort=False).sort_values(["series", "date"])


def _energy_history_and_contributions(
    project_root: Path,
    vintage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One canonical historical reconstruction feeds both table and HD chart."""
    processed = Path(project_root) / "data" / "processed" / str(vintage)
    reconstruction = historical_model_component_reconstruction(processed)
    component_history = reconstruction["component_history"].copy()
    indices = reconstruction["inputs"]["indices"]

    rows: list[pd.DataFrame] = []
    for name in _OUTLOOK_ENERGY_COMPONENTS:
        if name not in component_history:
            continue
        level = pd.to_numeric(component_history[name], errors="coerce")
        yoy = 100.0 * (level / level.shift(12) - 1.0)
        block = yoy.dropna().rename("value").reset_index()
        block["series"] = name
        rows.append(block[["date", "series", "value"]])

    for name, column in (
        ("car_fuels_petrol", "hicp_petrol"),
        ("car_fuels_diesel", "hicp_diesel"),
    ):
        if column not in indices:
            continue
        level = pd.to_numeric(indices[column], errors="coerce")
        yoy = 100.0 * (level / level.shift(12) - 1.0)
        block = yoy.dropna().rename("value").reset_index()
        block["series"] = name
        rows.append(block[["date", "series", "value"]])

    observed = (
        pd.concat(rows, ignore_index=True, sort=False)
        if rows
        else pd.DataFrame(columns=["date", "series", "value"])
    )

    exact = yoy_contributions_from_terms(
        reconstruction["energy"]["index"],
        reconstruction["energy"]["terms"],
    )
    contribution_rows: list[dict] = []
    for date, row in exact.iterrows():
        for name in ENERGY_COMPONENT_ORDER:
            value = row.get(name, np.nan)
            if not np.isfinite(value):
                continue
            contribution_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "series": str(name),
                    "value": float(value),
                    "aggregate_yoy": float(row.get("aggregate_yoy", np.nan)),
                }
            )
    contributions = pd.DataFrame(contribution_rows)
    return observed, contributions, reconstruction["inputs"]["weights"]


def _headline_weight_payload(headline_run: Path) -> dict[str, dict[str, float]]:
    path = Path(headline_run) / "headline_weights_annual.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if "year" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "year"})
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame.dropna(subset=["year"])
    out: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        year = str(int(row["year"]))
        values = {}
        for name in _OUTLOOK_HEADLINE_SERIES:
            value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
            if pd.notna(value):
                values[name] = float(value) / 10.0  # per thousand -> percent of Headline
        out[year] = values
    return out


def _energy_weight_payload(weights: pd.DataFrame) -> dict[str, dict[str, float]]:
    if weights is None or weights.empty:
        return {}
    model = model_energy_weights(weights)
    out: dict[str, dict[str, float]] = {}
    for year in weights.index:
        values: dict[str, float] = {}
        for name in ("hicp_energy", *_OUTLOOK_ENERGY_COMPONENTS):
            source = model if name in model.columns else weights
            if name in source.columns and pd.notna(source.loc[year, name]):
                values[name] = float(source.loc[year, name]) / 10.0
        for name, column in (
            ("car_fuels_petrol", "hicp_petrol"),
            ("car_fuels_diesel", "hicp_diesel"),
        ):
            if column in weights.columns and pd.notna(weights.loc[year, column]):
                values[name] = float(weights.loc[year, column]) / 10.0
        out[str(int(year))] = values
    return out


def _resolved_weight_row(payload: dict | None, year: int) -> dict[str, float]:
    source = dict(payload or {})
    years = sorted(int(k) for k in source if str(k).isdigit())
    if not years:
        return {}
    chosen = int(year) if str(int(year)) in source else max((y for y in years if y <= int(year)), default=years[-1])
    return dict(source.get(str(chosen)) or {})


def _common_predictive_dates(bundle: dict | None, horizon: int) -> pd.DatetimeIndex:
    """One monthly calendar for Headline, Energy and Core Overview tails."""
    if not bundle:
        return pd.DatetimeIndex([])
    date_sets: list[set[pd.Timestamp]] = []
    for key in ("headline_predictive", "energy_predictive", "core_predictive"):
        frame = _records_frame(bundle.get(key))
        if frame.empty or "date" not in frame:
            continue
        dates = set(pd.DatetimeIndex(frame["date"].dropna().unique()))
        if dates:
            date_sets.append(dates)
    if not date_sets:
        return pd.DatetimeIndex([])
    common = set.intersection(*date_sets)
    if not common:
        return pd.DatetimeIndex([])
    dates = pd.DatetimeIndex(sorted(common), name="date")
    return dates[: max(1, int(horizon))]


def _fmt_outlook_number(value) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(x) else f"{x:.2f}"


def _outlook_latest_cell(frame: pd.DataFrame, series: str) -> html.Div:
    if frame is None or frame.empty or not {"date", "series", "value"}.issubset(frame.columns):
        return html.Div("—", style={"color": TOKENS["muted"]})
    block = frame.loc[frame["series"].astype(str).eq(str(series))].dropna(subset=["date", "value"]).sort_values("date")
    if block.empty:
        return html.Div("—", style={"color": TOKENS["muted"]})
    row = block.iloc[-1]
    return html.Div(
        [
            html.Div(_fmt_outlook_number(row["value"]), style={"fontWeight": 700, "fontVariantNumeric": "tabular-nums"}),
            html.Div(pd.Timestamp(row["date"]).strftime("%b %Y") + " · OBS", style={"fontSize": "9px", "color": TOKENS["muted"], "marginTop": "2px"}),
        ]
    )


def _outlook_predictive_cell(frame: pd.DataFrame, series: str, date) -> html.Div:
    if date is None or frame is None or frame.empty or "series" not in frame.columns:
        return html.Div("—", style={"color": TOKENS["muted"]})
    block = frame.loc[
        frame["series"].astype(str).eq(str(series))
        & pd.to_datetime(frame["date"], errors="coerce").eq(pd.Timestamp(date))
    ]
    if block.empty:
        return html.Div("—", style={"color": TOKENS["muted"]})
    row = block.iloc[0]
    return html.Div(
        [
            html.Div(_fmt_outlook_number(row.get("value")), style={"fontWeight": 700, "fontVariantNumeric": "tabular-nums"}),
            html.Div("med " + _fmt_outlook_number(row.get("q50")), style={"fontSize": "9px", "color": TOKENS["muted"], "marginTop": "2px", "fontVariantNumeric": "tabular-nums"}),
            html.Div(
                _fmt_outlook_number(row.get("q05")) + " – " + _fmt_outlook_number(row.get("q95")),
                style={"fontSize": "9px", "color": TOKENS["muted"], "fontVariantNumeric": "tabular-nums"},
            ),
        ]
    )


def _outlook_latest_value(frame: pd.DataFrame, series: str) -> float:
    if (
        frame is None
        or frame.empty
        or not {"date", "series", "value"}.issubset(frame.columns)
    ):
        return float("nan")
    block = (
        frame.loc[frame["series"].astype(str).eq(str(series))]
        .dropna(subset=["date", "value"])
        .sort_values("date")
    )
    if block.empty:
        return float("nan")
    try:
        value = float(block.iloc[-1]["value"])
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _outlook_predictive_mean(
    frame: pd.DataFrame,
    series: str,
    date,
) -> float:
    if (
        date is None
        or frame is None
        or frame.empty
        or not {"date", "series", "value"}.issubset(frame.columns)
    ):
        return float("nan")
    block = frame.loc[
        frame["series"].astype(str).eq(str(series))
        & pd.to_datetime(frame["date"], errors="coerce").eq(pd.Timestamp(date))
    ]
    if block.empty:
        return float("nan")
    try:
        value = float(block.iloc[0]["value"])
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _outlook_variation_cell(current: float, previous: float) -> html.Div:
    if not np.isfinite(current) or not np.isfinite(previous):
        return html.Div(
            "—",
            style={
                "color": TOKENS["muted"],
                "fontVariantNumeric": "tabular-nums",
                "textAlign": "right",
            },
        )

    delta = float(current - previous)
    tolerance = 5e-5
    if delta > tolerance:
        color = "#0B7A5C"
        background = "rgba(11, 122, 92, 0.075)"
        text = f"+{delta:.2f}"
    elif delta < -tolerance:
        color = "#B42318"
        background = "rgba(180, 35, 24, 0.070)"
        text = f"{delta:.2f}"
    else:
        color = TOKENS["muted"]
        background = "rgba(17, 24, 39, 0.035)"
        text = "0.00"

    return html.Div(
        [
            html.Div(
                text,
                style={
                    "fontWeight": 750,
                    "fontVariantNumeric": "tabular-nums",
                },
            ),
            html.Div(
                "pp",
                style={
                    "fontSize": "8px",
                    "fontWeight": 600,
                    "marginTop": "1px",
                    "opacity": 0.82,
                },
            ),
        ],
        style={
            "display": "inline-flex",
            "flexDirection": "column",
            "alignItems": "flex-end",
            "minWidth": "50px",
            "padding": "3px 6px",
            "borderRadius": "8px",
            "color": color,
            "background": background,
            "textAlign": "right",
        },
    )


def _weight_span(
    payload: dict | None,
    key: str | None,
    dates: Iterable[pd.Timestamp],
    *,
    numerator_keys: tuple[str, ...] | None = None,
    denominator_key: str | None = None,
    constant: float | None = None,
) -> html.Div:
    if constant is not None:
        return html.Div(f"{float(constant):.1f}%", style={"fontVariantNumeric": "tabular-nums"})
    years = sorted({pd.Timestamp(x).year for x in dates if x is not None and not pd.isna(x)})
    if not years:
        return html.Div("—", style={"color": TOKENS["muted"]})
    values: list[tuple[int, float]] = []
    for year in years:
        row = _resolved_weight_row(payload, year)
        if numerator_keys:
            num = sum(float(row.get(k, np.nan)) for k in numerator_keys)
            if not np.isfinite(num):
                continue
        elif key is not None and key in row:
            num = float(row[key])
        else:
            continue
        if denominator_key is not None:
            den = float(row.get(denominator_key, np.nan))
            if not np.isfinite(den) or den <= 0:
                continue
            num = 100.0 * num / den
        values.append((year, num))
    if not values:
        return html.Div("—", style={"color": TOKENS["muted"]})
    unique = []
    for item in values:
        if not unique or abs(unique[-1][1] - item[1]) > 1e-12:
            unique.append(item)
    if len(unique) == 1:
        return html.Div(f"{unique[0][1]:.1f}%", style={"fontVariantNumeric": "tabular-nums"})
    return html.Div(
        [html.Div(f"{year}: {value:.1f}%") for year, value in unique],
        style={"fontSize": "10px", "lineHeight": 1.25, "fontVariantNumeric": "tabular-nums"},
    )


def _outlook_label(label: str, note: str | None = None, *, level: int = 0, strong: bool = False) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontWeight": 700 if strong else 560}),
            html.Div(note, style={"fontSize": "9px", "color": TOKENS["muted"], "marginTop": "2px"}) if note else None,
        ],
        style={"paddingLeft": f"{14 * int(level)}px"},
    )


def _outlook_row(
    label: str,
    *,
    observed: pd.DataFrame,
    predictive: pd.DataFrame,
    series: str,
    dates: list[pd.Timestamp | None],
    headline_weight,
    within_weight,
    level: int = 0,
    strong: bool = False,
    note: str | None = None,
    top: bool = False,
) -> html.Div:
    cells = [
        _outlook_label(label, note, level=level, strong=strong),
        headline_weight,
        within_weight,
        _outlook_latest_cell(observed, series),
    ]

    previous_mean = _outlook_latest_value(observed, series)
    for date in dates:
        current_mean = _outlook_predictive_mean(
            predictive,
            series,
            date,
        )
        cells.append(
            _outlook_predictive_cell(
                predictive,
                series,
                date,
            )
        )
        cells.append(
            _outlook_variation_cell(
                current_mean,
                previous_mean,
            )
        )
        if np.isfinite(current_mean):
            previous_mean = current_mean

    return html.Div(
        cells,
        style={
            "display": "grid",
            "gridTemplateColumns": _OUTLOOK_GRID,
            "alignItems": "stretch",
            "minWidth": "1390px",
            "borderTop": ("2px solid " + TOKENS["hairline"]) if top else ("1px solid " + TOKENS["hairline"]),
            "background": "rgba(17,24,39,0.018)" if top else "transparent",
        },
        className="overview-outlook-row",
    )


def _outlook_cell_wrap(component) -> html.Div:
    return html.Div(component, style={"padding": "9px 8px", "minWidth": 0})


def _wrap_outlook_row(row: html.Div) -> html.Div:
    # Apply compact cell padding without changing semantic values.
    row.children = [_outlook_cell_wrap(child) for child in row.children]
    return row


def _outlook_header_cell(title: str, date=None) -> html.Div:
    children = [
        html.Div(
            title,
            style={
                "fontWeight": 700,
                "fontSize": "10px",
                "letterSpacing": ".04em",
            },
        )
    ]
    if date is not None:
        if isinstance(date, str) and date == "vs prior":
            subtitle = date
        else:
            subtitle = pd.Timestamp(date).strftime("%b %Y")
        children.append(
            html.Div(
                subtitle,
                style={
                    "fontSize": "9px",
                    "fontWeight": 500,
                    "marginTop": "2px",
                },
            )
        )
    return html.Div(
        children,
        style={
            "padding": "8px",
            "color": TOKENS["muted"],
        },
    )


def overview_outlook_table(bundle: dict | None) -> html.Div:
    if not bundle:
        return html.Div("Select a production vintage in Data.", className="placeholder-text")

    common = list(_common_predictive_dates(bundle, 4))
    dates: list[pd.Timestamp | None] = common + [None] * max(0, 4 - len(common))
    dates = dates[:4]
    weight_dates = [x for x in dates if x is not None]

    headline_obs = _records_frame(bundle.get("headline_observed"))
    headline_pred = _records_frame(bundle.get("headline_predictive"))
    headline_component_obs = _records_frame(bundle.get("headline_component_observed"))
    headline_component_pred = _records_frame(bundle.get("headline_component_predictive"))
    energy_obs = _records_frame(bundle.get("energy_observed"))
    energy_pred = _records_frame(bundle.get("energy_predictive"))
    energy_component_obs = _records_frame(bundle.get("energy_component_observed"))
    energy_component_pred = _records_frame(bundle.get("energy_component_predictive"))
    core_obs = _records_frame(bundle.get("core_observed"))
    core_pred = _records_frame(bundle.get("core_predictive"))
    headline_weights = dict(bundle.get("headline_weights") or {})
    energy_weights = dict(bundle.get("energy_weights") or {})

    header_dates = dates
    header = html.Div(
        [
            _outlook_header_cell("SERIES"),
            _outlook_header_cell("HEADLINE WT"),
            _outlook_header_cell("WITHIN GROUP"),
            _outlook_header_cell("LATEST"),
            _outlook_header_cell("NOWCAST", header_dates[0]),
            _outlook_header_cell("Δ", "vs prior"),
            _outlook_header_cell("M+1", header_dates[1]),
            _outlook_header_cell("Δ", "vs prior"),
            _outlook_header_cell("M+2", header_dates[2]),
            _outlook_header_cell("Δ", "vs prior"),
            _outlook_header_cell("M+3", header_dates[3]),
            _outlook_header_cell("Δ", "vs prior"),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": _OUTLOOK_GRID,
            "minWidth": "1390px",
            "background": "rgba(17,24,39,0.035)",
            "borderBottom": "1px solid " + TOKENS["hairline"],
        },
    )

    h_headline = _wrap_outlook_row(_outlook_row(
        "Headline HICP",
        observed=headline_obs,
        predictive=headline_pred,
        series="hicp_total",
        dates=dates,
        headline_weight=_weight_span(None, None, weight_dates, constant=100.0),
        within_weight=html.Div("—", style={"color": TOKENS["muted"]}),
        strong=True,
        note="Official total; BVAR components aggregated draw by draw",
        top=True,
    ))
    headline_children = []
    for series, label in (
        ("hicp_energy", "Energy"),
        ("hicp_food", "Food"),
        ("hicp_neig", "NEIG"),
        ("hicp_services", "Services"),
    ):
        headline_children.append(_wrap_outlook_row(_outlook_row(
            label,
            observed=headline_component_obs,
            predictive=headline_component_pred,
            series=series,
            dates=dates,
            headline_weight=_weight_span(headline_weights, series, weight_dates),
            within_weight=html.Div("—", style={"color": TOKENS["muted"]}),
            level=1,
        )))

    h_energy = _wrap_outlook_row(_outlook_row(
        "HICP Energy",
        observed=energy_obs,
        predictive=energy_pred,
        series="hicp_energy",
        dates=dates,
        headline_weight=_weight_span(headline_weights, "hicp_energy", weight_dates),
        within_weight=_weight_span(None, None, weight_dates, constant=100.0),
        strong=True,
        note="Six-component chain-linked Laspeyres aggregate",
        top=True,
    ))

    energy_children = []
    car_row = _wrap_outlook_row(_outlook_row(
        "Car fuels",
        observed=energy_component_obs,
        predictive=energy_component_pred,
        series="car_fuels",
        dates=dates,
        headline_weight=_weight_span(energy_weights, "car_fuels", weight_dates),
        within_weight=_weight_span(energy_weights, "car_fuels", weight_dates, denominator_key="hicp_energy"),
        level=1,
        note="Petrol + Diesel + Other transport fuels",
    ))
    car_children = []
    for series, label in (
        ("car_fuels_petrol", "Petrol"),
        ("car_fuels_diesel", "Diesel"),
    ):
        car_children.append(_wrap_outlook_row(_outlook_row(
            label,
            observed=energy_component_obs,
            predictive=energy_component_pred,
            series=series,
            dates=dates,
            headline_weight=_weight_span(energy_weights, series, weight_dates),
            within_weight=_weight_span(energy_weights, series, weight_dates, denominator_key="car_fuels"),
            level=2,
            note="Model-specific HICP path",
        )))
    energy_children.append(
        html.Details(
            [html.Summary(car_row, style={"cursor": "pointer", "listStyle": "revert"}), *car_children],
            open=False,
        )
    )
    for series, label in (
        ("liquid_fuels", "Liquid fuels"),
        ("gas", "Gas"),
        ("electricity", "Electricity"),
        ("heat_energy", "Heat energy"),
        ("solid_fuels", "Solid fuels"),
    ):
        energy_children.append(_wrap_outlook_row(_outlook_row(
            label,
            observed=energy_component_obs,
            predictive=energy_component_pred,
            series=series,
            dates=dates,
            headline_weight=_weight_span(energy_weights, series, weight_dates),
            within_weight=_weight_span(energy_weights, series, weight_dates, denominator_key="hicp_energy"),
            level=1,
        )))

    h_core = _wrap_outlook_row(_outlook_row(
        "Core HICP",
        observed=core_obs,
        predictive=core_pred,
        series="hicp_core",
        dates=dates,
        headline_weight=_weight_span(headline_weights, None, weight_dates, numerator_keys=("hicp_neig", "hicp_services")),
        within_weight=_weight_span(None, None, weight_dates, constant=100.0),
        strong=True,
        note="NEIG + Services; weights renormalised inside Core",
        top=True,
    ))
    core_children = []
    for series, label in (("hicp_neig", "NEIG"), ("hicp_services", "Services")):
        core_children.append(_wrap_outlook_row(_outlook_row(
            label,
            observed=headline_component_obs,
            predictive=headline_component_pred,
            series=series,
            dates=dates,
            headline_weight=_weight_span(headline_weights, series, weight_dates),
            within_weight=_weight_span(
                headline_weights,
                series,
                weight_dates,
                denominator_key=None,
            ),
            level=1,
        )))

    # Replace the Core child within-group cells with the exact NEIG/(NEIG+Services)
    # and Services/(NEIG+Services) shares; doing this explicitly avoids inventing
    # a synthetic weight series in the persisted contract.
    for idx, series in enumerate(("hicp_neig", "hicp_services")):
        shares = []
        years = sorted({pd.Timestamp(x).year for x in weight_dates})
        for year in years:
            row = _resolved_weight_row(headline_weights, year)
            den = float(row.get("hicp_neig", np.nan)) + float(row.get("hicp_services", np.nan))
            num = float(row.get(series, np.nan))
            if np.isfinite(num) and np.isfinite(den) and den > 0:
                shares.append((year, 100.0 * num / den))
        within = (
            html.Div("—", style={"color": TOKENS["muted"]})
            if not shares
            else html.Div(
                f"{shares[0][1]:.1f}%" if len({round(v, 10) for _, v in shares}) == 1 else [html.Div(f"{y}: {v:.1f}%") for y, v in shares],
                style={"fontVariantNumeric": "tabular-nums", "fontSize": "10px" if len(shares) > 1 else None},
            )
        )
        # children are wrapped row divs; second data cell after label/headline wt.
        core_children[idx].children[2] = _outlook_cell_wrap(within)

    empty_note = None
    if len(common) < 4:
        empty_note = html.Div(
            f"Only {len(common)} common predictive month(s) are available across Headline, Energy and Core; unavailable M+ cells are left blank rather than extrapolated.",
            style={"fontSize": "10px", "color": TOKENS["muted"], "padding": "8px 10px"},
        )

    return html.Div(
        [
            html.Div(
                [
                    header,
                    html.Details(
                        [html.Summary(h_headline, style={"cursor": "pointer", "listStyle": "revert"}), *headline_children],
                        open=False,
                    ),
                    html.Details(
                        [html.Summary(h_energy, style={"cursor": "pointer", "listStyle": "revert"}), *energy_children],
                        open=False,
                    ),
                    html.Details(
                        [html.Summary(h_core, style={"cursor": "pointer", "listStyle": "revert"}), *core_children],
                        open=False,
                    ),
                    empty_note,
                ],
                style={"minWidth": "1390px"},
            )
        ],
        style={
            "overflowX": "auto",
            "border": "1px solid " + TOKENS["hairline"],
            "borderRadius": "12px",
            "background": "#FFFFFF",
        },
    )

def _load_bundle(results_root: Path, registry_path, project_root: Path, vintage: str) -> tuple[dict, list[str]]:
    messages: list[str] = []
    headline_run = _latest_headline_run(results_root, registry_path, vintage)
    headline = _headline_display(headline_run)

    headline_observed = _simple_rows(headline, record_type="history", series="hicp_total", metric="yoy")
    core_observed = _simple_rows(headline, record_type="history", series="hicp_core", metric="yoy")
    headline_component_observed = _multi_history_rows(
        headline,
        series=_OUTLOOK_HEADLINE_SERIES,
        metric="yoy",
    )
    headline_fitted = _simple_rows(headline, record_type="fitted", series="hicp_total", metric="yoy", value_col="posterior_mean")
    headline_predictive = _predictive_rows(
        headline,
        record_type="fan",
        metric="yoy",
        series="hicp_total",
        basis="baseline",
    )
    core_predictive = _predictive_rows(
        headline,
        record_type="fan",
        metric="yoy",
        series="hicp_core",
        basis="baseline",
    )
    headline_component_predictive = _predictive_rows(
        headline,
        record_type="fan",
        metric="yoy",
        basis="baseline",
    )
    headline_component_predictive = headline_component_predictive.loc[
        headline_component_predictive["series"].astype(str).isin(_OUTLOOK_HEADLINE_SERIES)
    ].copy()
    core_reconstructed = _core_reconstruction(headline_run)
    headline_contrib = _contribution_rows(
        headline,
        record_type="contribution_history",
        metric="yoy_contribution",
        series=HEADLINE_COMPONENT_LABELS,
    )
    headline_contrib_model = _predictive_rows(
        headline,
        record_type="contribution",
        metric="yoy_contribution",
        basis="baseline",
    )
    headline_contrib_model = headline_contrib_model.loc[
        headline_contrib_model["series"].astype(str).isin(HEADLINE_COMPONENT_LABELS)
    ].copy()
    core_contrib = _contribution_rows(
        headline,
        record_type="core_contribution_history",
        metric="core_yoy_contribution",
        series=CORE_COMPONENT_LABELS,
    )
    core_contrib_model = _predictive_rows(
        headline,
        record_type="core_contribution",
        metric="core_yoy_contribution",
        basis="baseline",
    )
    core_contrib_model = core_contrib_model.loc[
        core_contrib_model["series"].astype(str).isin(CORE_COMPONENT_LABELS)
    ].copy()

    headline_weights = _headline_weight_payload(headline_run)
    energy_observed = _energy_history(project_root, vintage)
    energy_predictive = pd.DataFrame()
    energy_component_predictive = pd.DataFrame()
    energy_component_observed = pd.DataFrame(columns=["date", "series", "value"])
    energy_weights = {}
    energy_contrib_model = pd.DataFrame()
    energy_source_fitted = pd.DataFrame()
    energy_aggregate_fitted = pd.DataFrame()
    energy_contrib = pd.DataFrame()
    aggregate_dir = None

    try:
        aggregate_dir = _latest_energy_aggregate(registry_path, vintage)
        display_path = aggregate_dir / ENERGY_DISPLAY_FILENAME
        if not display_path.is_file():
            build_aggregate_display(aggregate_dir, project_root=project_root)
        aggregate_display = load_display_artifact(display_path)
        energy_predictive = _predictive_rows(
            aggregate_display,
            record_type="fan",
            metric="yoy",
            series="hicp_energy",
            basis="baseline",
        )
        energy_component_predictive = _energy_outlook_predictive(aggregate_dir)
        energy_contrib_model = _predictive_rows(
            aggregate_display,
            record_type="contribution",
            metric="contribution_yoy",
            basis="baseline",
        )
        fitted, fitted_meta, _ = load_energy_aggregate_fitted(
            aggregate_dir,
            project_root=project_root,
            max_draws=300,
            pairing_seed=2026,
            start_date=None,
            persist_cache=True,
        )
        source = fitted.loc[
            fitted["basis"].astype(str).eq("source_bvar_fitted")
            & fitted["metric"].astype(str).eq("yoy")
        ].copy()
        if not source.empty:
            source = source[["date", "series", "posterior_mean"]].rename(columns={"posterior_mean": "value"})
            energy_source_fitted = source.dropna(subset=["value"])
        agg = fitted.loc[
            fitted["basis"].astype(str).eq("aggregate_bvar_fitted")
            & fitted["series"].astype(str).eq("hicp_energy")
            & fitted["metric"].astype(str).eq("yoy")
        ].copy()
        if not agg.empty:
            agg = agg[["date", "series", "posterior_mean"]].rename(columns={"posterior_mean": "value"})
            energy_aggregate_fitted = agg.dropna(subset=["value"])
        messages.append(f"Energy fitted cache {fitted_meta.get('cache_version', 'loaded')}")
    except Exception as exc:
        messages.append(f"Energy fitted unavailable: {exc}")

    try:
        energy_component_observed, energy_contrib, raw_energy_weights = _energy_history_and_contributions(
            project_root,
            vintage,
        )
        energy_weights = _energy_weight_payload(raw_energy_weights)
    except Exception as exc:
        messages.append(f"Energy historical accounting/outlook weights unavailable: {exc}")
        # Weight-only fallback is cheap and independent of the historical reconstruction.
        try:
            raw_energy_weights = load_aggregation_inputs(
                project_root / "data" / "processed" / str(vintage)
            )["weights"]
            energy_weights = _energy_weight_payload(raw_energy_weights)
        except Exception:
            pass

    bundle = {
        "vintage": str(vintage),
        "headline_run_id": headline_run.name,
        "energy_aggregate_run_id": None if aggregate_dir is None else aggregate_dir.name,
        "headline_observed": _records(headline_observed),
        "energy_observed": _records(energy_observed),
        "core_observed": _records(core_observed),
        "headline_component_observed": _records(headline_component_observed),
        "energy_component_observed": _records(energy_component_observed),
        "headline_fitted": _records(headline_fitted),
        "core_reconstructed": _records(core_reconstructed),
        "energy_source_fitted": _records(energy_source_fitted),
        "energy_aggregate_fitted": _records(energy_aggregate_fitted),
        "headline_predictive": _records(headline_predictive),
        "energy_predictive": _records(energy_predictive),
        "core_predictive": _records(core_predictive),
        "headline_component_predictive": _records(headline_component_predictive),
        "energy_component_predictive": _records(energy_component_predictive),
        "headline_weights": headline_weights,
        "energy_weights": energy_weights,
        "headline_contributions": _records(headline_contrib),
        "headline_contributions_model": _records(headline_contrib_model),
        "core_contributions": _records(core_contrib),
        "core_contributions_model": _records(core_contrib_model),
        "energy_contributions": _records(energy_contrib),
        "energy_contributions_model": _records(energy_contrib_model),
        "model_spec_rows": _overview_model_spec_rows(
            headline_run,
            aggregate_dir,
        ),
    }
    return bundle, messages

# GRAPH_EXPORT_READABILITY_G5_OVERVIEW_V1
def _empty(message: str, height: int = 500) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=message, showarrow=False, font=dict(size=13, color=TOKENS["muted"]))
    apply_inflation_figure_style(fig, uirevision="overview-empty", height=height)
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def overview_main_figure(
    bundle: dict | None,
    overlays: list[str] | None,
    *,
    forecast_horizon: int = 12,
    fan_mode: str = "68",
) -> go.Figure:
    """Full observed history with fitted overlays and predictive tail in-place."""
    if not bundle:
        return _empty("Select a production vintage in Data.")

    overlays = set(overlays or [])
    forecast_horizon = int(forecast_horizon or 12)
    fan_mode = str(fan_mode or "68")
    fig = go.Figure()

    specs = {
        "headline": (
            "headline_observed",
            "headline_predictive",
            "Headline",
            INFLATION_COLORS["observed"],
        ),
        "energy": (
            "energy_observed",
            "energy_predictive",
            "Energy",
            INFLATION_COLORS["hicp_energy"],
        ),
        "core": (
            "core_observed",
            "core_predictive",
            "Core",
            INFLATION_COLORS["hicp_neig"],
        ),
    }

    observed_cache: dict[str, pd.DataFrame] = {}
    regimes = {}
    common_predictive_dates = _common_predictive_dates(bundle, forecast_horizon)

    # Full official histories.
    for key, (observed_key, predictive_key, label, color) in specs.items():
        observed = _records_frame(bundle.get(observed_key)).sort_values("date")
        predictive = _records_frame(bundle.get(predictive_key))
        observed_cache[key] = observed

        if not observed.empty:
            fig.add_trace(
                go.Scatter(
                    x=observed["date"],
                    y=observed["value"],
                    mode="lines",
                    name=f"{label} observed",
                    line={"width": 2, "color": color},
                )
            )
            last_obs = pd.Timestamp(observed["date"].max())
        else:
            last_obs = None

        # Segment from the saved contract first, then clip every domain to the
        # same monthly Overview calendar. This is display-only: no path is
        # extended or recomputed.
        available_h = max(1, int(predictive["date"].nunique())) if not predictive.empty else 1
        nowcast, forecast, origin = _split_predictive_dates(
            predictive,
            last_observed=last_obs,
            horizon=available_h,
        )
        if len(common_predictive_dates):
            nowcast = nowcast.loc[nowcast["date"].isin(common_predictive_dates)].copy()
            forecast = forecast.loc[forecast["date"].isin(common_predictive_dates)].copy()
        regimes[key] = (nowcast, forecast, origin)

    # Existing fitted overlays are preserved exactly.
    if "seven_energy_bvars" in overlays:
        source = _records_frame(bundle.get("energy_source_fitted"))
        for series, label in SOURCE_BVAR_LABELS.items():
            block = (
                source.loc[
                    source.get("series", pd.Series(dtype=str))
                    .astype(str)
                    .eq(series)
                ].sort_values("date")
                if not source.empty
                else source
            )
            if block.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=block["date"],
                    y=block["value"],
                    mode="lines",
                    name=label,
                    line={"width": 1.2, "dash": "dot"},
                )
            )

    if "energy_aggregate_fitted" in overlays:
        block = _records_frame(bundle.get("energy_aggregate_fitted"))
        if not block.empty:
            fig.add_trace(
                go.Scatter(
                    x=block["date"],
                    y=block["value"],
                    mode="lines",
                    name="Energy aggregate fitted · posterior mean",
                    line={
                        "width": 2,
                        "dash": "dash",
                        "color": INFLATION_COLORS["hicp_energy"],
                    },
                )
            )

    if "headline_fitted" in overlays:
        block = _records_frame(bundle.get("headline_fitted"))
        if not block.empty:
            fig.add_trace(
                go.Scatter(
                    x=block["date"],
                    y=block["value"],
                    mode="lines",
                    name="Headline BVAR fitted · posterior mean",
                    line={
                        "width": 2,
                        "dash": "dash",
                        "color": INFLATION_COLORS["fitted"],
                    },
                )
            )

    if "core_aggregate_history" in overlays:
        block = _records_frame(bundle.get("core_reconstructed"))
        if not block.empty:
            fig.add_trace(
                go.Scatter(
                    x=block["date"],
                    y=block["value"],
                    mode="lines",
                    name="Core aggregate full history",
                    line={
                        "width": 1.8,
                        "dash": "dot",
                        "color": INFLATION_COLORS["hicp_neig"],
                    },
                )
            )

    # Posterior fans for all three aggregates, each from its own saved draws.
    for key, (_observed_key, _predictive_key, label, color) in specs.items():
        nowcast, forecast, _origin = regimes[key]
        if fan_mode in {"90", "both"}:
            _fan_band(
                fig, nowcast, "q05", "q95",
                name=f"{label} nowcast 90%",
                color=color, alpha=0.055,
            )
            _fan_band(
                fig, forecast, "q05", "q95",
                name=f"{label} forecast 90%",
                color=color, alpha=0.075,
            )
        if fan_mode in {"68", "both"}:
            _fan_band(
                fig, nowcast, "q16", "q84",
                name=f"{label} nowcast 68%",
                color=color, alpha=0.105,
            )
            _fan_band(
                fig, forecast, "q16", "q84",
                name=f"{label} forecast 68%",
                color=color, alpha=0.145,
            )

    for key, (_observed_key, _predictive_key, label, color) in specs.items():
        observed = observed_cache[key]
        nowcast, forecast, _origin = regimes[key]

        anchor_date = (
            pd.Timestamp(observed["date"].iloc[-1])
            if not observed.empty
            else None
        )
        anchor_value = (
            float(observed["value"].iloc[-1])
            if not observed.empty
            else None
        )

        nowcast_line = _anchor_path(nowcast, anchor_date, anchor_value)
        if not nowcast_line.empty:
            fig.add_trace(
                go.Scatter(
                    x=nowcast_line["date"],
                    y=nowcast_line["value"],
                    mode="lines+markers",
                    name=f"{label} nowcast · posterior mean",
                    line={"width": 2.2, "color": color},
                    marker={"size": 4},
                )
            )
            forecast_anchor_date = pd.Timestamp(nowcast["date"].iloc[-1])
            forecast_anchor_value = float(nowcast["value"].iloc[-1])
        else:
            forecast_anchor_date = anchor_date
            forecast_anchor_value = anchor_value

        forecast_line = _anchor_path(
            forecast, forecast_anchor_date, forecast_anchor_value
        )
        if not forecast_line.empty:
            fig.add_trace(
                go.Scatter(
                    x=forecast_line["date"],
                    y=forecast_line["value"],
                    mode="lines+markers",
                    name=f"{label} forecast · posterior mean",
                    line={"width": 2.2, "dash": "dash", "color": color},
                    marker={"size": 4},
                )
            )

    # Forecast origins are model-owned. If they coincide, show one marker;
    # otherwise label each distinct domain origin.
    grouped_origins: dict[pd.Timestamp, list[str]] = {}
    for key, (_observed_key, _predictive_key, label, _color) in specs.items():
        origin = regimes[key][2]
        if origin is not None:
            grouped_origins.setdefault(pd.Timestamp(origin), []).append(label)

    for origin, labels_at_origin in sorted(grouped_origins.items()):
        fig.add_shape(
            type="line",
            x0=origin,
            x1=origin,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line={"color": TOKENS["muted"], "width": 1, "dash": "dot"},
        )
        text = (
            "Forecast origin"
            if len(grouped_origins) == 1
            else " / ".join(labels_at_origin) + " forecast"
        )
        fig.add_annotation(
            x=origin,
            y=0.995,
            xref="x",
            yref="paper",
            text=text,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 10, "color": TOKENS["muted"]},
            bgcolor="rgba(255,255,255,0.72)",
        )

    fig.add_hline(y=0.0, line_width=1, line_color=TOKENS["hairline"])
    fig.update_layout(dragmode="pan", hovermode="x unified")
    apply_inflation_figure_style(
        fig,
        uirevision=(
            f"overview::{bundle.get('vintage')}::"
            f"{','.join(sorted(overlays))}::all-fans::"
            f"{forecast_horizon}::{fan_mode}"
        ),
        height=560,
        y_title="% y/y",
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _safe_series_block(frame: pd.DataFrame, series: str) -> pd.DataFrame:
    if (
        frame is None
        or frame.empty
        or not {"date", "series", "value"}.issubset(frame.columns)
    ):
        return pd.DataFrame(columns=["date", "series", "value"])
    out = frame.loc[
        frame["series"].astype(str).eq(str(series)),
        ["date", "series", "value"],
    ].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["date", "value"]).sort_values("date")


def _contribution_figure(
    historical: pd.DataFrame,
    model_rows: pd.DataFrame,
    observed_total: pd.DataFrame,
    predictive_total: pd.DataFrame,
    labels: dict[str, str],
    *,
    mode: str,
    title_key: str,
    horizon: int,
) -> go.Figure:
    """Historical accounting continued by saved posterior-mean contributions."""
    mode = str(mode or "bars")
    horizon = int(horizon or 12)

    historical = (
        historical.copy()
        if historical is not None
        else pd.DataFrame(columns=["date", "series", "value"])
    )
    model_rows = (
        model_rows.copy()
        if model_rows is not None
        else pd.DataFrame(columns=["date", "series", "value"])
    )
    observed_total = (
        observed_total.copy()
        if observed_total is not None
        else pd.DataFrame(columns=["date", "series", "value"])
    )
    predictive_total = (
        predictive_total.copy()
        if predictive_total is not None
        else pd.DataFrame(columns=["date", "series", "value"])
    )

    for frame in (historical, model_rows, observed_total, predictive_total):
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if "value" in frame.columns:
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    last_observed = (
        pd.Timestamp(observed_total["date"].dropna().max())
        if (
            not observed_total.empty
            and "date" in observed_total.columns
            and observed_total["date"].notna().any()
        )
        else None
    )

    total_nowcast, total_forecast, origin = _split_predictive_dates(
        predictive_total,
        last_observed=last_observed,
        horizon=horizon,
    )
    model_nowcast, model_forecast, _ = _split_predictive_dates(
        model_rows,
        last_observed=last_observed,
        horizon=horizon,
    )
    model = pd.concat([model_nowcast, model_forecast], ignore_index=True, sort=False)

    if (
        not model.empty
        and {"date", "series"}.issubset(model.columns)
    ):
        model = model.sort_values(["date", "series"]).reset_index(drop=True)
        model_start = pd.Timestamp(model["date"].dropna().min())
        if not historical.empty and "date" in historical.columns:
            historical = historical.loc[
                pd.to_datetime(historical["date"], errors="coerce") < model_start
            ].copy()

    if historical.empty and model.empty:
        return _empty(
            "Historical / predictive contribution decomposition is unavailable.",
            500,
        )

    fig = go.Figure()
    trace_count = 0

    for series, label in labels.items():
        h = _safe_series_block(historical, series)
        m = _safe_series_block(model, series)
        combined = pd.concat(
            [h[["date", "value"]], m[["date", "value"]]],
            ignore_index=True,
        ).dropna(subset=["date", "value"]).sort_values("date")

        if combined.empty:
            continue
        trace_count += 1

        if mode == "lines":
            fig.add_trace(
                go.Scatter(
                    x=combined["date"],
                    y=combined["value"],
                    mode="lines",
                    name=label,
                    line={"width": 1.7},
                    hovertemplate=f"%{{y:+.2f}} pp<extra>{label}</extra>",
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=combined["date"],
                    y=combined["value"],
                    name=label,
                    hovertemplate=f"%{{y:+.2f}} pp<extra>{label}</extra>",
                )
            )

    if trace_count == 0:
        return _empty("No contribution rows matched the selected domain.", 500)

    # Total observed path.
    if (
        not observed_total.empty
        and {"date", "value"}.issubset(observed_total.columns)
    ):
        total_hist = observed_total.dropna(subset=["date", "value"]).sort_values("date")
        if (
            not model.empty
            and "date" in model.columns
            and model["date"].notna().any()
        ):
            total_hist = total_hist.loc[
                total_hist["date"] < pd.Timestamp(model["date"].dropna().min())
            ]
        if not total_hist.empty:
            fig.add_trace(
                go.Scatter(
                    x=total_hist["date"],
                    y=total_hist["value"],
                    mode="lines",
                    name="Observed total",
                    line={"width": 2.4, "color": TOKENS["ink"]},
                )
            )

    total_model = pd.concat(
        [total_nowcast, total_forecast],
        ignore_index=True,
        sort=False,
    )
    if (
        not total_model.empty
        and {"date", "value"}.issubset(total_model.columns)
    ):
        total_model = total_model.dropna(subset=["date", "value"]).sort_values("date")
        if not total_model.empty:
            fig.add_trace(
                go.Scatter(
                    x=total_model["date"],
                    y=total_model["value"],
                    mode="lines+markers",
                    name="Model total · posterior mean",
                    line={"width": 2.2, "dash": "dash", "color": TOKENS["ink"]},
                    marker={"size": 4},
                )
            )

    if mode != "lines":
        fig.update_layout(barmode="relative")

    if origin is not None:
        origin = pd.Timestamp(origin)
        fig.add_shape(
            type="line",
            x0=origin,
            x1=origin,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line={"color": TOKENS["muted"], "width": 1, "dash": "dot"},
        )
        fig.add_annotation(
            x=origin,
            y=0.98,
            xref="x",
            yref="paper",
            text="Forecast origin",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={"size": 10, "color": TOKENS["muted"]},
            bgcolor="rgba(255,255,255,0.72)",
        )

    fig.add_hline(y=0, line_width=1, line_color=TOKENS["hairline"])
    fig.update_layout(
        dragmode="pan",
        hovermode="x unified",
        uirevision=f"overview-decomp::{title_key}::{mode}::{horizon}",
    )
    apply_inflation_figure_style(
        fig,
        uirevision=f"overview-decomp::{title_key}::{mode}::{horizon}",
        height=500,
        y_title="percentage points",
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def overview_decomposition_figure(
    bundle: dict | None,
    domain: str,
    mode: str,
    *,
    horizon: int = 12,
) -> go.Figure:
    if not bundle:
        return _empty("Select a production vintage in Data.", 480)

    domain = str(domain or "headline")
    mode = str(mode or "bars")
    horizon = int(horizon or 12)

    if domain == "energy":
        historical = _records_frame(bundle.get("energy_contributions"))
        model = _records_frame(bundle.get("energy_contributions_model"))
        observed = _records_frame(bundle.get("energy_observed"))
        predictive = _records_frame(bundle.get("energy_predictive"))
        return _contribution_figure(
            historical,
            model,
            observed,
            predictive,
            ENERGY_COMPONENT_LABELS,
            mode=mode,
            title_key="energy",
            horizon=horizon,
        )

    if domain == "core":
        historical = _records_frame(bundle.get("core_contributions"))
        model = _records_frame(bundle.get("core_contributions_model"))
        observed = _records_frame(bundle.get("core_observed"))
        predictive = _records_frame(bundle.get("core_predictive"))
        return _contribution_figure(
            historical,
            model,
            observed,
            predictive,
            CORE_COMPONENT_LABELS,
            mode=mode,
            title_key="core",
            horizon=horizon,
        )

    historical = _records_frame(bundle.get("headline_contributions"))
    model = _records_frame(bundle.get("headline_contributions_model"))
    observed = _records_frame(bundle.get("headline_observed"))
    predictive = _records_frame(bundle.get("headline_predictive"))
    return _contribution_figure(
        historical,
        model,
        observed,
        predictive,
        HEADLINE_COMPONENT_LABELS,
        mode=mode,
        title_key="headline",
        horizon=horizon,
    )


# OVERVIEW_RESULTS_XLSX_V1
_OVERVIEW_EXPORT_HEADERS = [
    "date",
    "vintage",
    "run_id",
    "series",
    "unit",
    "record_type",
    "observed",
    "mean",
    "median",
    "q05",
    "q16",
    "q84",
    "q95",
]


def _overview_export_scalar(value):
    import pandas as pd

    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


def _overview_export_rows_from_frame(
    frame,
    *,
    domain: str,
    model: str,
    run_id: str,
    vintage: str,
    source_artifact: str,
    max_future_periods: int = 12,
):
    """Extract one canonical observed/predictive metric per persisted series.

    Results omit source/domain/model/forecast/label/metric presentation columns.
    To keep rows unambiguous after removing ``metric``, HICP series prefer YoY
    and other observables prefer their LEVEL metric.
    """
    import pandas as pd

    required = {
        "record_type", "metric", "series", "date", "value",
        "q05", "q16", "q50", "q84", "q95",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Persisted display {source_artifact} lacks required columns: {missing}"
        )

    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.loc[work["date"].notna()].copy()
    work["metric"] = work["metric"].fillna("").astype(str)
    work["series"] = work["series"].fillna("").astype(str)

    metric_order = ("level", "yoy", "hicp_level", "absolute_change")
    keep_blocks = []
    for series_name, block in work.groupby("series", dropna=False, sort=False):
        available = set(block["metric"].dropna().astype(str))
        preferred = "yoy" if str(series_name).lower().startswith("hicp_") else "level"
        chosen = preferred if preferred in available else next(
            (name for name in metric_order if name in available),
            None,
        )
        if chosen is not None:
            keep_blocks.append(block.loc[block["metric"].astype(str).eq(chosen)].copy())
    work = (
        pd.concat(keep_blocks, ignore_index=False)
        if keep_blocks
        else work.iloc[0:0].copy()
    )

    record = work["record_type"].fillna("").astype(str).str.lower()
    history = work.loc[record.eq("history")].copy()
    fan = work.loc[record.eq("fan")].copy()

    if not fan.empty and "basis" in fan.columns:
        basis = fan["basis"].fillna("").astype(str).str.lower()
        if basis.eq("baseline").any():
            fan = fan.loc[basis.eq("baseline")].copy()

    if not fan.empty:
        if "segment" not in fan.columns:
            fan = fan.iloc[0:0].copy()
        else:
            segment = fan["segment"].fillna("").astype(str).str.lower()
            fan = fan.loc[segment.isin({"nowcast", "forecast"})].copy()

    capped = []
    if not fan.empty:
        for _, block in fan.groupby(["series", "metric"], dropna=False, sort=False):
            dates = sorted(pd.DatetimeIndex(block["date"].dropna().unique()))
            keep_dates = set(dates[: max(1, int(max_future_periods))])
            capped.append(block.loc[block["date"].isin(keep_dates)].copy())
    fan = pd.concat(capped, ignore_index=False) if capped else fan.iloc[0:0].copy()

    selected = pd.concat([history, fan], ignore_index=False)
    if selected.empty:
        return []

    rows = []
    for _, row in selected.iterrows():
        raw_series = str(row.get("series") or "")
        rec = str(row.get("record_type") or "").lower()
        segment = str(row.get("segment") or "").lower()
        output_type = "observed" if rec == "history" else segment
        is_history = output_type == "observed"
        values = {
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "vintage": str(vintage),
            "run_id": str(run_id),
            "series": raw_series,
            "unit": _overview_export_scalar(row.get("unit", "")),
            "record_type": output_type,
            "observed": _overview_export_scalar(row.get("value")) if is_history else "",
            "mean": "" if is_history else _overview_export_scalar(row.get("value")),
            "median": "" if is_history else _overview_export_scalar(row.get("q50")),
            "q05": "" if is_history else _overview_export_scalar(row.get("q05")),
            "q16": "" if is_history else _overview_export_scalar(row.get("q16")),
            "q84": "" if is_history else _overview_export_scalar(row.get("q84")),
            "q95": "" if is_history else _overview_export_scalar(row.get("q95")),
        }
        rows.append([values[name] for name in _OVERVIEW_EXPORT_HEADERS])
    return rows


def _overview_results_export_payload(
    bundle,
    *,
    vintage: str,
    results_root,
    registry_path,
):
    """Build the Overview workbook payload from persisted display artifacts only."""
    import json
    import pandas as pd
    from inflation_bvar_registry import promoted_run_ids

    item = dict(bundle or {})
    vintage = str(vintage or "").strip()
    headline_run_id = str(item.get("headline_run_id") or "").strip()
    aggregate_run_id = str(item.get("energy_aggregate_run_id") or "").strip()
    if not vintage or not headline_run_id or not aggregate_run_id:
        raise ValueError(
            "Overview export requires a production vintage, Headline run and Energy aggregate run."
        )

    root = Path(results_root).resolve()
    forecast_name = "unconditional"
    component_run_ids = promoted_run_ids(
        registry_path,
        vintage,
        forecast_name=forecast_name,
        require_all=True,
        require_draws=False,
    )
    if len(component_run_ids) != 7:
        raise ValueError(
            f"Expected 7 promoted Energy component runs for {vintage}, found {len(component_run_ids)}."
        )

    aggregate_dir = root / "hicp_energy_aggregate" / vintage / aggregate_run_id
    aggregate_display = aggregate_dir / "display_v1.parquet"
    aggregate_metadata = aggregate_dir / "metadata.json"
    if not aggregate_metadata.is_file():
        raise FileNotFoundError(aggregate_metadata)
    aggregate_meta = json.loads(aggregate_metadata.read_text(encoding="utf-8"))
    aggregate_meta_text = json.dumps(aggregate_meta, sort_keys=True, default=str)
    lineage_missing = [
        f"{model_id}:{run_id}"
        for model_id, run_id in component_run_ids.items()
        if str(run_id) not in aggregate_meta_text
    ]
    if lineage_missing:
        raise ValueError(
            "Current promoted Energy runs do not match the selected Overview aggregate lineage: "
            + ", ".join(lineage_missing)
        )

    headline_display = (
        root
        / "headline_joint"
        / vintage
        / headline_run_id
        / "forecasts"
        / forecast_name
        / "display_v1.parquet"
    )

    sources = []
    for model_id, run_id in sorted(component_run_ids.items()):
        path = (
            root
            / model_id
            / vintage
            / str(run_id)
            / "forecasts"
            / forecast_name
            / "display_v1.parquet"
        )
        sources.append(("Energy model", model_id, str(run_id), path))
    sources.extend(
        [
            ("Energy aggregate", "hicp_energy_aggregate", aggregate_run_id, aggregate_display),
            ("Headline", "headline_joint", headline_run_id, headline_display),
        ]
    )

    all_rows = []
    source_notes = []
    core_rows = 0
    series_pos = _OVERVIEW_EXPORT_HEADERS.index("series")
    for domain, model, run_id, path in sources:
        if not path.is_file():
            raise FileNotFoundError(
                f"Persisted display required for Overview export is missing: {path}. "
                "The export refuses to materialize/recompute it."
            )
        frame = pd.read_parquet(path)
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        rows = _overview_export_rows_from_frame(
            frame,
            domain=domain,
            model=model,
            run_id=run_id,
            vintage=vintage,
            source_artifact=rel,
            max_future_periods=12,
        )
        all_rows.extend(rows)
        core_rows += sum(
            1 for row in rows
            if str(row[series_pos]) == "hicp_core"
        )
        source_notes.append(f"{model}: {rel}")

    if not all_rows:
        raise ValueError(
            "Persisted Overview artifacts produced no observed/nowcast/forecast rows."
        )

    date_pos = _OVERVIEW_EXPORT_HEADERS.index("date")
    run_pos = _OVERVIEW_EXPORT_HEADERS.index("run_id")
    record_pos = _OVERVIEW_EXPORT_HEADERS.index("record_type")
    all_rows.sort(
        key=lambda row: (
            str(row[run_pos]),
            str(row[series_pos]),
            str(row[record_pos]),
        )
    )
    all_rows.sort(key=lambda row: str(row[date_pos]), reverse=True)

    metadata_rows = [
        ["Forecast name", forecast_name],
        ["Headline run ID", headline_run_id],
        ["Energy aggregate run ID", aggregate_run_id],
    ]
    for model_id, run_id in sorted(component_run_ids.items()):
        metadata_rows.append([f"Energy run · {model_id}", str(run_id)])
    metadata_rows.extend(
        [
            ["Source contract", "Persisted display_v1.parquet artifacts only"],
            ["Future export cap", "First 12 stored future periods per exported series"],
            ["Metric policy", "HICP series: YoY preferred; other observables: level preferred"],
            ["Row ordering", "Most recent date first"],
            ["Posterior mapping", "mean=value; median=q50; q05/q16/q84/q95 copied when stored"],
            ["Observed mapping", "history.value -> observed"],
            ["Core persisted rows", int(core_rows)],
            [
                "Core policy",
                (
                    "Included from persisted Headline display"
                    if core_rows
                    else "Unavailable in persisted Headline display; not materialized/recomputed"
                ),
            ],
            ["Aggregate metadata", str(aggregate_metadata.relative_to(root))],
            ["BVAR re-estimated", "No"],
            ["Model/result recomputation", "No"],
        ]
    )
    for note in source_notes:
        metadata_rows.append(["Source artifact", note])

    return {
        "vintage": vintage,
        "table_id": "overview-results",
        "title": "ECB VAR Overview results",
        "sheet_name": "Results",
        "filename": f"ECB_VAR_results_vintage_{vintage}.xlsx",
        "metadata_rows": metadata_rows,
        "table": {
            "headers": list(_OVERVIEW_EXPORT_HEADERS),
            "rows": all_rows,
        },
    }

def register_overview_callbacks(
    app,
    *,
    results_root,
    registry_path,
    project_root,
    production_vintage_store_id="production-vintage-store",
    registry_store_id="registry-store",
):
    from dash import State
    from dash.exceptions import PreventUpdate

    results_root = Path(results_root).resolve()
    project_root = Path(project_root).resolve()

    @app.callback(
        Output("overview-data-store", "data"),
        Output("overview-status", "children"),
        Output("overview-vintage-label", "children"),
        Input(production_vintage_store_id, "data"),
        Input(registry_store_id, "data"),
    )
    def load_overview(production_store, _registry):
        vintage = str((production_store or {}).get("vintage") or "")
        if not vintage:
            return None, html.Div("Select a production vintage in Data."), "—"
        try:
            snapshot_id = str(
                (_registry or {}).get("snapshot_id")
                or (_registry or {}).get("revision")
                or "bootstrap"
            )
            bundle, messages = snapshot_get_or_build(
                "overview_bundle",
                (snapshot_id, vintage),
                lambda: _load_bundle(
                    results_root,
                    registry_path,
                    project_root,
                    vintage,
                ),
            )
            status = [
                html.Strong("Overview ready"),
                html.Span(f" · vintage {vintage}"),
                html.Span(f" · Headline run {str(bundle.get('headline_run_id'))[:12]}"),
            ]
            if bundle.get("energy_aggregate_run_id"):
                status.append(html.Span(f" · Energy aggregate {str(bundle.get('energy_aggregate_run_id'))[:12]}"))
            if messages:
                status.append(html.Span(" · " + " · ".join(messages)))
            return bundle, html.Div(status), vintage
        except Exception as exc:
            return None, html.Div([html.Strong("Overview load failed: "), html.Span(str(exc))], className="banner-error"), vintage

    @app.callback(
        Output("overview-outlook-table", "children"),
        Input("overview-data-store", "data"),
    )
    def outlook_table(bundle):
        return overview_outlook_table(bundle)

    @app.callback(
        Output("overview-main-graph", "figure"),
        Input("overview-data-store", "data"),
        Input("overview-overlays", "value"),
        Input("overview-forecast-horizon", "value"),
        Input("overview-forecast-fan", "value"),
    )
    def main_figure(bundle, overlays, horizon, fan):
        return overview_main_figure(
            bundle,
            overlays,
            forecast_horizon=horizon,
            fan_mode=fan,
        )

    @app.callback(
        Output("overview-decomp-graph", "figure"),
        Input("overview-data-store", "data"),
        Input("overview-decomp-domain", "value"),
        Input("overview-decomp-display", "value"),
        Input("overview-forecast-horizon", "value"),
    )
    def decomp_figure(bundle, domain, mode, horizon):
        return overview_decomposition_figure(
            bundle,
            domain,
            mode,
            horizon=horizon,
        )

    @app.callback(
        Output("overview-model-spec-table", "children"),
        Input("overview-data-store", "data"),
    )
    def model_spec_table(bundle):
        return overview_model_spec_table(bundle)

    @app.callback(
        Output("overview-download-results", "disabled"),
        Input("overview-data-store", "data"),
    )
    def download_ready(bundle):
        item = dict(bundle or {})
        return not bool(
            item.get("headline_run_id")
            and item.get("energy_aggregate_run_id")
        )

    @app.callback(
        Output("overview-download-results-file", "data"),
        Input("overview-download-results", "n_clicks"),
        State("overview-data-store", "data"),
        State(production_vintage_store_id, "data"),
        prevent_initial_call=True,
    )
    def download_results(n_clicks, bundle, production_store):
        if not n_clicks:
            raise PreventUpdate
        vintage = str((production_store or {}).get("vintage") or "").strip()
        if not vintage:
            raise PreventUpdate
        payload = _overview_results_export_payload(
            bundle,
            vintage=vintage,
            results_root=results_root,
            registry_path=registry_path,
        )
        from dashboard_xlsx_export import build_export_xlsx

        content, filename = build_export_xlsx(payload)

        def _writer(buffer):
            buffer.write(content)

        return dcc.send_bytes(_writer, filename)


__all__ = [
    "overview_page",
    "overview_outlook_table",
    "overview_main_figure",
    "overview_decomposition_figure",
    "overview_model_spec_table",
    "register_overview_callbacks",
]
