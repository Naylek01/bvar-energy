"""Interactive recursive structural analysis for the locked Headline-HICP BVAR.

This dashboard backend mirrors the Energy Structural UX while preserving the
Headline model's own state-space units and production contract.

Econometric contract
--------------------
* saved Headline posterior only; no BVAR re-estimation;
* recursive / Cholesky ordering: Energy -> Food -> NEIG -> Services;
* IRF / FEVD use the regular structural-SV state at a selected reference date;
* outlier multipliers are excluded from IRF / FEVD;
* FEVD always uses one-standard-deviation structural shocks;
* recursive historical decomposition is reference-date invariant because each
  historical observation uses its realised SV/outlier state;
* Headline state variables are log HICP indices.  "Level impact" in the UI is
  expressed in 100 x log points and converted to raw log units before calling
  the generic BVAR engine;
* HD is reported as 12-month log-inflation contributions (100 x log points).
  Posterior means are used so the additive reconstruction identity is preserved.

The existing materialised Headline structural sidecar/display is not modified.
This module reads the persisted Gibbs posterior through the verified Headline
saved-run loader and computes the interactive view on demand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from energy_bvar_model import (
    forecast_error_variance_decomposition,
    historical_decomposition,
    impulse_responses,
)
from energy_bvar_theme import graph_config
from inflation_table_contract import (
    HD_COLUMNS,
    IRF_COLUMNS,
    readable_table,
    structural_fevd_table,
    structural_hd_records,
    structural_irf_records,
    fmt,
    interval,
)
from headline_bvar_conditional import (
    HeadlineConditionalError,
    load_saved_headline_posterior,
)
from headline_joint_bvar import STATE_VARIABLES


HEADLINE_STRUCTURAL_CONTRACT_VERSION = "headline-structural-interactive-v1"
DEFAULT_STRUCTURAL_DRAWS = 300
MAX_STRUCTURAL_DRAWS = 1000
DEFAULT_HORIZON = 12
MAX_HORIZON = 12
DEFAULT_SHOCK_UNIT = "structural_std"
DEFAULT_SHOCK_SIZE = 1.0

STATE_LABELS = {
    "state__hicp_energy": "Energy",
    "state__hicp_food": "Food",
    "state__hicp_neig": "NEIG",
    "state__hicp_services": "Services",
}
STATE_COMPONENTS = tuple(STATE_VARIABLES)
_STRUCTURAL_COLOURS = ("#2563EB", "#10B981", "#F59E0B", "#EF4444")


class HeadlineStructuralDashboardError(RuntimeError):
    """Raised when a saved Headline run cannot satisfy the interactive contract."""


# GRAPH_EXPORT_READABILITY_G1_HEADLINE_STRUCTURAL_V1
def _label(name: str) -> str:
    return STATE_LABELS.get(str(name), str(name).replace("state__hicp_", "").replace("_", " ").title())


def _draw_subset_indices(n_draws: int, requested: int) -> np.ndarray:
    n = int(n_draws)
    m = min(n, max(1, int(requested)))
    if n < 1:
        raise HeadlineStructuralDashboardError("The saved Headline posterior contains no draws.")
    if m == n:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, m, dtype=int))


def resolve_headline_run_directory(
    results_root: str | Path,
    *,
    vintage: str,
    run_id: str,
) -> Path:
    directory = Path(results_root) / "headline_joint" / str(vintage) / str(run_id)
    if not directory.is_dir():
        raise HeadlineStructuralDashboardError(f"Saved Headline run directory not found: {directory}")
    if not (directory / "metadata.json").is_file():
        raise HeadlineStructuralDashboardError(f"metadata.json missing in {directory}")
    if not (directory / "draws.npz").is_file():
        raise HeadlineStructuralDashboardError(
            "Headline Structural requires persisted posterior draws; "
            f"draws.npz is missing in {directory}."
        )
    return directory


def load_headline_structural_result(
    run_directory: str | Path,
    *,
    project_root: str | Path,
    posterior_draws: int = DEFAULT_STRUCTURAL_DRAWS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exact saved Headline posterior and retain a deterministic draw subset."""
    posterior = load_saved_headline_posterior(
        run_directory,
        project_root=project_root,
    )
    result = dict(posterior.result)
    variables = list(result.get("variables") or STATE_VARIABLES)
    if variables != list(STATE_VARIABLES):
        raise HeadlineStructuralDashboardError(
            "Headline state ordering changed: "
            f"expected {list(STATE_VARIABLES)}, found {variables}."
        )

    required = ("B", "A", "log_variance", "outlier_scales")
    missing = [name for name in required if name not in result]
    if missing:
        raise HeadlineStructuralDashboardError(
            "Saved Headline posterior is incomplete for interactive Structural: "
            + ", ".join(missing)
        )

    n_draws = int(np.asarray(result["B"]).shape[0])
    indices = _draw_subset_indices(
        n_draws,
        min(int(posterior_draws), MAX_STRUCTURAL_DRAWS),
    )

    selected: dict[str, Any] = {}
    for name, value in result.items():
        if name in {"metadata", "prep", "prior", "variables", "p", "frequency", "missing_data_method", "n_draws"}:
            selected[name] = value
            continue
        try:
            array = np.asarray(value)
        except Exception:
            selected[name] = value
            continue
        if array.ndim >= 1 and array.shape[0] == n_draws:
            selected[name] = array[indices]
        else:
            selected[name] = value

    selected["variables"] = variables
    selected["n_draws"] = int(len(indices))
    selected["p"] = int(result["p"])
    selected["frequency"] = "monthly"

    dates = pd.DatetimeIndex(selected["prep"]["dates"])
    if len(dates) < 1:
        raise HeadlineStructuralDashboardError("Headline regression calendar is empty.")

    info = {
        "contract_version": HEADLINE_STRUCTURAL_CONTRACT_VERSION,
        "model_id": "headline_joint",
        "model_label": "Headline HICP joint BVAR",
        "vintage": posterior.vintage,
        "run_id": posterior.run_id,
        "frequency": "monthly",
        "p": int(selected["p"]),
        "variables": variables,
        "labels": {name: _label(name) for name in variables},
        "units": {name: "100 × log points" for name in variables},
        "available_posterior_draws": n_draws,
        "selected_posterior_draws": int(len(indices)),
        "selected_draw_indices": indices.astype(int).tolist(),
        "reference_dates": [pd.Timestamp(x).isoformat() for x in dates],
        "default_reference_date": pd.Timestamp(dates[-1]).isoformat(),
        "recursive_ordering": variables,
        "requires_data_augmentation": bool(
            selected["prep"].get("requires_data_augmentation", False)
        ),
        "missing_data_method": str(
            selected.get("missing_data_method")
            or posterior.metadata.get("missing_data_method")
            or "dk"
        ),
    }
    return selected, info


def headline_structural_run_contract(
    run_directory: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    _, info = load_headline_structural_result(
        run_directory,
        project_root=project_root,
        posterior_draws=1,
    )
    return info


def _quantile_summary(array: np.ndarray, axis: int = 0) -> dict[str, np.ndarray]:
    q = np.quantile(
        np.asarray(array, dtype=float),
        [0.05, 0.16, 0.50, 0.84, 0.95],
        axis=axis,
    )
    return {
        "q05": q[0],
        "q16": q[1],
        "q50": q[2],
        "q84": q[3],
        "q95": q[4],
    }


def _volatility_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Posterior persistent/total structural innovation scales in 100 x log points."""
    variables = list(result["variables"])
    dates = pd.DatetimeIndex(result["prep"]["dates"])
    log_h = np.asarray(result["log_variance"], dtype=float)
    if log_h.ndim != 3:
        raise HeadlineStructuralDashboardError(
            f"Unexpected Headline log_variance shape {log_h.shape}."
        )
    persistent = 100.0 * np.exp(0.5 * np.clip(log_h[:, 1:, :], -745.0, 700.0))
    scales = np.asarray(result["outlier_scales"], dtype=float)
    if scales.shape != persistent.shape:
        raise HeadlineStructuralDashboardError(
            "Headline outlier_scales and SV paths have incompatible shapes: "
            f"{scales.shape} vs {persistent.shape}."
        )
    total = persistent * scales

    if "posterior_outlier_probability_draws" in result:
        outlier_probability = np.asarray(
            result["posterior_outlier_probability_draws"], dtype=float
        ).mean(axis=0)
    elif "outlier_indicators" in result:
        outlier_probability = np.asarray(
            result["outlier_indicators"], dtype=float
        ).mean(axis=0)
    else:
        outlier_probability = (scales > 1.0 + 1e-12).mean(axis=0)

    persistent_q = np.quantile(persistent, [0.16, 0.50, 0.84], axis=0)
    total_q = np.quantile(total, [0.16, 0.50, 0.84], axis=0)
    rows: list[dict[str, Any]] = []
    for t, date in enumerate(dates):
        for j, variable in enumerate(variables):
            rows.append(
                {
                    "date": pd.Timestamp(date).isoformat(),
                    "variable": variable,
                    "persistent_q16": float(persistent_q[0, t, j]),
                    "persistent_q50": float(persistent_q[1, t, j]),
                    "persistent_q84": float(persistent_q[2, t, j]),
                    "total_q16": float(total_q[0, t, j]),
                    "total_q50": float(total_q[1, t, j]),
                    "total_q84": float(total_q[2, t, j]),
                    "outlier_probability": float(outlier_probability[t, j]),
                }
            )
    return rows


def _reference_state_frame(payload: Mapping[str, Any] | None) -> pd.DataFrame:
    if not payload or not payload.get("ok"):
        return pd.DataFrame()
    frame = pd.DataFrame(payload.get("volatility", []))
    needed = {"date", "variable", "persistent_q50"}
    if frame.empty or not needed.issubset(frame.columns):
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["persistent_q50"] = pd.to_numeric(
        frame["persistent_q50"], errors="coerce"
    )
    frame = frame.dropna(subset=["date", "persistent_q50"])
    frame = frame.loc[frame["persistent_q50"] > 0.0]
    if frame.empty:
        return pd.DataFrame()

    wide = frame.pivot_table(
        index="date",
        columns="variable",
        values="persistent_q50",
        aggfunc="last",
    ).sort_index()
    ordered = [name for name in payload.get("variables", []) if name in wide.columns]
    if not ordered:
        return pd.DataFrame()
    wide = wide.loc[:, ordered]
    med = wide.median(axis=0, skipna=True).replace(0.0, np.nan)
    relative_sd = wide.divide(med, axis=1)
    relative_variance = relative_sd.pow(2.0)
    positive = relative_variance.where(relative_variance > 0.0)
    out = relative_variance.copy()
    out["joint_stress"] = np.exp(np.log(positive).mean(axis=1, skipna=True))
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


def reference_regime_date(
    payload: Mapping[str, Any] | None,
    regime: str = "latest",
) -> str | None:
    state = _reference_state_frame(payload)
    if state.empty:
        return None
    joint = pd.to_numeric(state["joint_stress"], errors="coerce").dropna()
    if joint.empty:
        return None

    key = str(regime or "latest").strip().lower()
    if key == "latest":
        chosen = joint.index[-1]
    elif key in {"p10", "p50", "p90"}:
        q = {"p10": 0.10, "p50": 0.50, "p90": 0.90}[key]
        target = float(np.nanquantile(joint.to_numpy(dtype=float), q))
        distance = np.abs(
            np.log(joint.to_numpy(dtype=float)) - np.log(target)
        )
        chosen = joint.index[int(np.nanargmin(distance))]
    elif key in {"peak_2022", "peak2022"}:
        block = joint.loc[joint.index.year == 2022]
        chosen = block.idxmax() if not block.empty else joint.idxmax()
    elif key in {"peak", "sample_peak"}:
        chosen = joint.idxmax()
    else:
        raise HeadlineStructuralDashboardError(
            f"Unknown Headline reference-volatility regime {regime!r}."
        )
    return pd.Timestamp(chosen).isoformat()


def reference_volatility_snapshot(
    payload: Mapping[str, Any] | None,
    reference_date=None,
) -> dict[str, Any]:
    if not payload or not payload.get("ok"):
        return {"ok": False, "cards": []}
    frame = pd.DataFrame(payload.get("volatility", []))
    state = _reference_state_frame(payload)
    if frame.empty or state.empty:
        return {"ok": False, "cards": []}

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    ref = pd.Timestamp(
        reference_date
        if reference_date is not None
        else payload.get("reference_date", state.index[-1])
    )
    if ref.tzinfo is not None:
        ref = ref.tz_localize(None)
    available = pd.DatetimeIndex(state.index)
    if ref not in available:
        pos = int(np.argmin(np.abs((available - ref).asi8)))
        ref = pd.Timestamp(available[pos])

    cards: list[dict[str, Any]] = []
    labels = dict(payload.get("labels", {}) or {})
    units = dict(payload.get("units", {}) or {})
    for variable in payload.get("variables", []):
        history = frame.loc[
            frame["variable"].astype(str).eq(str(variable))
        ].sort_values("date")
        row = history.loc[history["date"].eq(ref)]
        if row.empty or variable not in state.columns:
            continue
        row = row.iloc[-1]
        rel_var = float(state.loc[ref, variable])
        rel_sd = float(np.sqrt(rel_var)) if rel_var >= 0.0 else np.nan
        series_state = pd.to_numeric(state[variable], errors="coerce").dropna()
        percentile = (
            100.0 * float((series_state <= rel_var).mean())
            if len(series_state)
            else np.nan
        )
        cards.append(
            {
                "variable": str(variable),
                "label": labels.get(variable, _label(variable)),
                "unit": units.get(variable, "100 × log points"),
                "persistent_q16": float(row.get("persistent_q16", np.nan)),
                "persistent_q50": float(row.get("persistent_q50", np.nan)),
                "persistent_q84": float(row.get("persistent_q84", np.nan)),
                "relative_sd": rel_sd,
                "relative_variance": rel_var,
                "percentile": percentile,
                "outlier_probability": float(
                    row.get("outlier_probability", np.nan)
                ),
            }
        )

    joint = pd.to_numeric(state["joint_stress"], errors="coerce").dropna()
    joint_value = float(state.loc[ref, "joint_stress"])
    joint_percentile = (
        100.0 * float((joint <= joint_value).mean())
        if len(joint)
        else np.nan
    )
    return {
        "ok": True,
        "reference_date": ref.isoformat(),
        "joint_stress": joint_value,
        "joint_percentile": joint_percentile,
        "cards": cards,
    }


def _empty_figure(message: str, *, height: int = 430) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.52,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        font={"size": 13, "color": "#64748B"},
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 48, "r": 24, "t": 42, "b": 48},
        xaxis={"visible": False},
        yaxis={"visible": False},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def volatility_sparkline_figure(
    payload: Mapping[str, Any] | None,
    *,
    variable: str,
    reference_date=None,
) -> go.Figure:
    state = _reference_state_frame(payload)
    if state.empty or variable not in state.columns:
        return _empty_figure("No SV history", height=105)
    rel_sd = np.sqrt(
        pd.to_numeric(state[variable], errors="coerce").clip(lower=0.0)
    )
    ref = pd.Timestamp(
        reference_date
        if reference_date is not None
        else (payload or {}).get("reference_date", state.index[-1])
    )
    if ref.tzinfo is not None:
        ref = ref.tz_localize(None)
    if ref not in state.index:
        ref = pd.Timestamp(state.index[-1])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=state.index,
            y=rel_sd,
            mode="lines",
            line={"width": 1.8, "color": "#2563EB"},
            hovertemplate="%{x|%Y-%m}<br>√λ / own median=%{y:.2f}×<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(y=1.0, line_width=1, line_color="#E5E7EB")
    fig.add_shape(
        type="line",
        x0=ref.isoformat(),
        x1=ref.isoformat(),
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"width": 1.2, "dash": "dot", "color": "#F97316"},
    )
    value = float(rel_sd.loc[ref]) if pd.notna(rel_sd.loc[ref]) else np.nan
    if np.isfinite(value):
        fig.add_trace(
            go.Scatter(
                x=[ref],
                y=[value],
                mode="markers",
                marker={
                    "size": 7,
                    "color": "#F97316",
                    "line": {"width": 1, "color": "white"},
                },
                hovertemplate="Reference<br>√λ / own median=%{y:.2f}×<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=105,
        margin={"l": 4, "r": 4, "t": 2, "b": 2},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x",
        dragmode=False,
        xaxis={"visible": False, "fixedrange": True},
        yaxis={"visible": False, "fixedrange": True, "rangemode": "tozero"},
        uirevision=f"{(payload or {}).get('run_id')}::headline-sv-card::{variable}",
    )
    return fig


def relative_volatility_state_figure(
    payload: Mapping[str, Any] | None,
    *,
    reference_date=None,
) -> go.Figure:
    snap = reference_volatility_snapshot(payload, reference_date)
    if not snap.get("ok") or not snap.get("cards"):
        return _empty_figure("No joint SV state is available.", height=250)
    cards = list(snap["cards"])
    labels = [card["label"] for card in cards]
    values = [float(card["relative_variance"]) for card in cards]
    colours = [
        _STRUCTURAL_COLOURS[i % len(_STRUCTURAL_COLOURS)]
        for i in range(len(cards))
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colours},
            text=[f"{value:.2f}×" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>λ / own-sample median=%{x:.2f}×<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_vline(
        x=1.0,
        line_width=1.2,
        line_dash="dash",
        line_color="#94A3B8",
    )
    xmax = max(max(values) * 1.18, 1.35)
    fig.update_layout(
        template="plotly_white",
        height=max(220, 58 * len(cards) + 58),
        margin={"l": 10, "r": 54, "t": 18, "b": 42},
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis={
            "title": "Structural variance relative to its own sample median (×)",
            "range": [0.0, xmax],
            "gridcolor": "#EEF2F7",
            "zeroline": False,
        },
        yaxis={"autorange": "reversed", "showgrid": False},
        hovermode="closest",
        uirevision=(
            f"{(payload or {}).get('run_id')}::headline-sv-relative::"
            f"{snap.get('reference_date')}"
        ),
    )
    return fig


def _irf_rows(irf: Mapping[str, Any]) -> list[dict[str, Any]]:
    variables = list(irf["variables"])
    shocks = list(irf["shock_names"])
    horizons = np.asarray(irf["horizons"], dtype=int)
    rows: list[dict[str, Any]] = []
    # Engine results are in log units. Dashboard uses 100 x log points.
    for metric, key in (
        ("change", "change_irfs"),
        ("cumulative", "cumulative_level_irfs"),
    ):
        summary = _quantile_summary(
            100.0 * np.asarray(irf[key], dtype=float),
            axis=0,
        )
        for hpos, horizon in enumerate(horizons):
            for i, response in enumerate(variables):
                for j, shock in enumerate(shocks):
                    rows.append(
                        {
                            "metric": metric,
                            "horizon": int(horizon),
                            "response": response,
                            "shock": shock,
                            **{
                                name: float(values[hpos, i, j])
                                for name, values in summary.items()
                            },
                        }
                    )
    return rows


def _fevd_rows(fevd: Mapping[str, Any]) -> list[dict[str, Any]]:
    variables = list(fevd["variables"])
    shocks = list(fevd["shock_names"])
    horizons = np.asarray(fevd["horizons"], dtype=int)
    summary = _quantile_summary(
        np.asarray(fevd["fevd_draws"], dtype=float),
        axis=0,
    )
    rows: list[dict[str, Any]] = []
    for hpos, horizon in enumerate(horizons):
        for i, response in enumerate(variables):
            for j, shock in enumerate(shocks):
                rows.append(
                    {
                        "horizon": int(horizon),
                        "response": response,
                        "shock": shock,
                        **{
                            name: float(100.0 * values[hpos, i, j])
                            for name, values in summary.items()
                        },
                    }
                )
    return rows


def _rolling_sum_12(array: np.ndarray) -> np.ndarray:
    """Rolling 12-period sum along axis=1, preserving all other dimensions."""
    arr = np.asarray(array, dtype=float)
    if arr.ndim < 2:
        raise ValueError("Expected an array with a time axis at position 1.")
    out = np.full_like(arr, np.nan, dtype=float)
    if arr.shape[1] < 12:
        return out
    csum = np.cumsum(arr, axis=1)
    out[:, 11:] = csum[:, 11:]
    if arr.shape[1] > 12:
        out[:, 12:] -= csum[:, :-12]
    return out


def _hd_rows(
    hd: Mapping[str, Any],
    *,
    requires_data_augmentation: bool,
) -> tuple[list[dict[str, Any]], float, str]:
    """Posterior-mean 12m log-inflation HD in 100 x log points."""
    variables = list(hd["variables"])
    components = list(hd["component_names"])
    dates = pd.DatetimeIndex(hd["dates"])

    contribution_draws = 100.0 * _rolling_sum_12(
        np.asarray(hd["contributions"], dtype=float)
    )
    base_draws = 100.0 * _rolling_sum_12(
        np.asarray(hd["base"], dtype=float)
    )
    reconstructed_draws = 100.0 * _rolling_sum_12(
        np.asarray(hd["reconstructed"], dtype=float)
    )

    contributions = np.nanmean(contribution_draws, axis=0)
    base = np.nanmean(base_draws, axis=0)
    reconstructed_mean = np.nanmean(reconstructed_draws, axis=0)

    if requires_data_augmentation:
        display_observed = reconstructed_mean
        observed_label = "Posterior-mean completed 12m log inflation"
    else:
        observed_draw = np.asarray(hd["observed"], dtype=float)[None, ...]
        display_observed = 100.0 * _rolling_sum_12(observed_draw)[0]
        observed_label = "Observed 12m log inflation"

    additive = base + contributions.sum(axis=2)
    valid = np.isfinite(additive) & np.isfinite(display_observed)
    mean_error = (
        float(np.max(np.abs(additive[valid] - display_observed[valid])))
        if valid.any()
        else float("nan")
    )

    rows: list[dict[str, Any]] = []
    for t, date in enumerate(dates):
        if t < 11:
            continue
        for i, response in enumerate(variables):
            if not np.isfinite(display_observed[t, i]):
                continue
            row = {
                "date": pd.Timestamp(date).isoformat(),
                "response": response,
                "observed": float(display_observed[t, i]),
                "base": float(base[t, i]),
                "reconstructed": float(additive[t, i]),
            }
            for j, component in enumerate(components):
                row[component] = float(contributions[t, i, j])
            rows.append(row)
    return rows, mean_error, observed_label


def compute_headline_structural(
    run_directory: str | Path,
    *,
    project_root: str | Path,
    reference_date=None,
    horizon: int = DEFAULT_HORIZON,
    posterior_draws: int = DEFAULT_STRUCTURAL_DRAWS,
    shock_unit: str = DEFAULT_SHOCK_UNIT,
    shock_size: float = DEFAULT_SHOCK_SIZE,
    split_outlier_amplification: bool = True,
) -> dict[str, Any]:
    """Compute interactive recursive IRF/FEVD/HD from the saved Headline posterior."""
    horizon = int(horizon)
    if horizon < 1 or horizon > MAX_HORIZON:
        raise HeadlineStructuralDashboardError(
            f"Headline structural horizon must lie in [1, {MAX_HORIZON}], got {horizon}."
        )
    shock_unit = str(shock_unit).strip().lower()
    if shock_unit not in {"structural_std", "level"}:
        raise HeadlineStructuralDashboardError(
            "shock_unit must be 'structural_std' or 'level'."
        )
    shock_size = float(shock_size)
    if not np.isfinite(shock_size) or shock_size <= 0:
        raise HeadlineStructuralDashboardError(
            f"shock_size must be finite and positive, got {shock_size}."
        )

    result, info = load_headline_structural_result(
        run_directory,
        project_root=project_root,
        posterior_draws=posterior_draws,
    )
    reference = (
        info["default_reference_date"]
        if reference_date is None
        else pd.Timestamp(reference_date).isoformat()
    )

    # UI level-impact units are 100 x log points. The generic engine's state
    # variable is raw log(HICP), so convert only the IRF normalisation target.
    engine_shock_size = shock_size / 100.0 if shock_unit == "level" else shock_size

    irf = impulse_responses(
        result,
        identification="recursive",
        reference_date=reference,
        horizon=horizon,
        shock_unit=shock_unit,
        shock_size=engine_shock_size,
        include_outlier_scale=False,
    )
    fevd = forecast_error_variance_decomposition(
        result,
        identification="recursive",
        reference_date=reference,
        horizon=horizon,
        include_outlier_scale=False,
    )
    hd = historical_decomposition(
        result,
        identification="recursive",
        reference_date=reference,
        split_outlier_amplification=bool(split_outlier_amplification),
    )

    requires_augmentation = bool(info["requires_data_augmentation"])
    hd_rows, mean_hd_error, hd_observed_label = _hd_rows(
        hd,
        requires_data_augmentation=requires_augmentation,
    )

    return {
        "ok": True,
        "contract_version": HEADLINE_STRUCTURAL_CONTRACT_VERSION,
        "identification": "recursive",
        "model_id": "headline_joint",
        "model_label": info["model_label"],
        "vintage": info["vintage"],
        "run_id": info["run_id"],
        "frequency": "monthly",
        "p": info["p"],
        "variables": info["variables"],
        "labels": info["labels"],
        "units": info["units"],
        "recursive_ordering": info["recursive_ordering"],
        "reference_date": pd.Timestamp(irf["reference_date"]).isoformat(),
        "horizon": horizon,
        "posterior_draws": int(info["selected_posterior_draws"]),
        "available_posterior_draws": int(info["available_posterior_draws"]),
        "selected_draw_indices": info["selected_draw_indices"],
        "missing_data_method": info["missing_data_method"],
        "shock_unit": shock_unit,
        "shock_size": shock_size,
        "engine_shock_size": float(engine_shock_size),
        "reference_dependence": {
            "irf_structural_std": True,
            "irf_level": False,
            "fevd": True,
            "hd_recursive": False,
            "outlier_scale_in_irf_fevd": False,
        },
        "split_outlier_amplification": bool(split_outlier_amplification),
        "requires_data_augmentation": requires_augmentation,
        "hd_observed_label": hd_observed_label,
        "volatility": _volatility_rows(result),
        "irf": _irf_rows(irf),
        "fevd": _fevd_rows(fevd),
        "hd": hd_rows,
        "hd_component_names": list(hd["component_names"]),
        "diagnostics": {
            "fevd_max_share_sum_error": float(fevd["max_share_sum_error"]),
            "hd_max_reconstruction_error_drawwise": float(
                hd["max_reconstruction_error"]
            ),
            "hd_posterior_mean_12m_reconstruction_error": float(mean_hd_error),
            "recursive_acceptance_rate": 1.0,
        },
    }


def _layout(
    fig: go.Figure,
    *,
    y_title: str | None,
    height: int,
    uirevision: str,
) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 68, "r": 28, "t": 34, "b": 74},
        hovermode="x unified",
        dragmode="pan",
        font={
            "family": "Inter, Segoe UI, sans-serif",
            "color": "#334155",
            "size": 12,
        },
        legend={
            "orientation": "h",
            "x": 0.0,
            "xanchor": "left",
            "y": 1.08,
            "yanchor": "bottom",
            "font": {"size": 10},
        },
        xaxis={"showgrid": False, "linecolor": "#E2E8F0"},
        yaxis={
            "title": y_title,
            "gridcolor": "#EEF2F7",
            "zerolinecolor": "#CBD5E1",
            "zerolinewidth": 1,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel={"bgcolor": "white", "bordercolor": "#E2E8F0"},
        uirevision=uirevision,
    )
    return fig


def irf_figure(
    payload: Mapping[str, Any] | None,
    *,
    response: str | None,
    shock: str | None,
    metric: str = "cumulative",
    fan_mode: str = "68",
) -> go.Figure:
    if not payload:
        return _empty_figure("Loading Headline structural analysis…", height=470)
    if not payload.get("ok"):
        return _empty_figure(
            "Headline Structural failed: " + str(payload.get("error", "unknown error")),
            height=470,
        )
    if response not in payload["variables"] or shock not in payload["variables"]:
        return _empty_figure("Select a valid shock and response.", height=470)

    frame = pd.DataFrame(payload["irf"])
    part = frame.loc[
        (frame["response"] == response)
        & (frame["shock"] == shock)
        & (frame["metric"] == metric)
    ].sort_values("horizon")
    if part.empty:
        return _empty_figure("No IRF summary is available.", height=470)

    fig = go.Figure()
    if fan_mode in {"90", "both"}:
        fig.add_trace(go.Scatter(
            x=part["horizon"], y=part["q95"], mode="lines",
            line={"width": 0}, hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=part["horizon"], y=part["q05"], mode="lines",
            line={"width": 0}, fill="tonexty",
            fillcolor="rgba(37,99,235,0.10)", name="90% interval",
            hoverinfo="skip",
        ))
    if fan_mode in {"68", "both"}:
        fig.add_trace(go.Scatter(
            x=part["horizon"], y=part["q84"], mode="lines",
            line={"width": 0}, hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=part["horizon"], y=part["q16"], mode="lines",
            line={"width": 0}, fill="tonexty",
            fillcolor="rgba(37,99,235,0.22)", name="68% interval",
            hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=part["horizon"], y=part["q50"], mode="lines+markers",
        line={"width": 2.5, "color": "#2563EB"},
        marker={"size": 4, "color": "#2563EB"},
        name="Posterior median",
        hovertemplate="h=%{x}<br>%{y:.4g}<extra>Posterior median</extra>",
    ))
    fig.add_hline(y=0.0, line_width=1.1, line_color="#94A3B8")
    fig.update_xaxes(title="Horizon (months)")
    y_title = "100 × Δlog points" if metric == "change" else "100 × cumulative Δlog"
    labels = dict(payload.get("labels", {}) or {})
    result = _layout(
        fig,
        y_title=y_title,
        height=470,
        uirevision=(
            f"{payload.get('run_id')}::headline-irf::{response}::{shock}::{metric}::"
            f"{payload.get('shock_unit')}::{payload.get('shock_size')}"
        ),
    )
    result.update_layout(
        title=(
            "Component IRF — "
            f"{labels.get(shock, _label(shock))} shock → "
            f"{labels.get(response, _label(response))} · {y_title}"
        )
    )
    result.update_layout(
        title_x=0.0,
        title_xanchor="left",
        legend={
            "orientation": "h",
            "x": 0.0,
            "xanchor": "left",
            "y": -0.24,
            "yanchor": "top",
        },
        margin={"b": 110},
    )
    return result


def fevd_figure(
    payload: Mapping[str, Any] | None,
    *,
    response: str | None,
) -> go.Figure:
    if not payload:
        return _empty_figure("Loading Headline structural analysis…", height=420)
    if not payload.get("ok"):
        return _empty_figure(
            "Headline Structural failed: " + str(payload.get("error", "unknown error")),
            height=420,
        )
    if response not in payload["variables"]:
        return _empty_figure("Select a valid response variable.", height=420)

    frame = pd.DataFrame(payload["fevd"])
    part = frame.loc[frame["response"] == response].copy()
    if part.empty:
        return _empty_figure("No FEVD summary is available.", height=420)

    # Display-only focus. The full FEVD remains in the table.
    cross = part.loc[part["shock"] != response].copy()
    if cross.empty:
        return _empty_figure("No cross-shock FEVD shares are available.", height=420)

    labels = dict(payload.get("labels", {}) or {})
    variables = list(payload.get("variables") or [])

    fig = go.Figure()
    for j, shock in enumerate(variables):
        if shock == response:
            continue
        block = cross.loc[cross["shock"] == shock].sort_values("horizon")
        if block.empty:
            continue
        label = labels.get(shock, _label(shock))
        fig.add_trace(go.Bar(
            x=block["horizon"],
            y=block["q50"],
            name=label,
            marker={"color": _STRUCTURAL_COLOURS[j % len(_STRUCTURAL_COLOURS)]},
            hovertemplate="h=%{x}<br>%{y:.2f}%<extra>" + label + "</extra>",
        ))

    cross_sum = (
        cross.groupby("horizon", as_index=False)["q50"]
        .sum()["q50"]
        .to_numpy(dtype=float)
    )
    finite = cross_sum[np.isfinite(cross_sum)]
    upper = float(np.max(finite)) if finite.size else 1.0
    upper = max(1.0, upper * 1.15)

    fig.update_layout(barmode="stack", bargap=0.18)
    fig.update_yaxes(range=[0.0, upper], ticksuffix="%")
    fig.update_xaxes(title="Forecast horizon (months)")

    response_label = labels.get(response, _label(response))
    result = _layout(
        fig,
        y_title="Cross-shock forecast-error variance share",
        height=420,
        uirevision=(
            f"{payload.get('run_id')}::headline-fevd-cross::{response}::"
            f"{payload.get('reference_date')}"
        ),
    )
    result.update_layout(
        title=(
            f"Cross-shock FEVD — {response_label} · "
            "own shock excluded from chart"
        ),
        title_x=0.0,
        title_xanchor="left",
        legend={
            "orientation": "h",
            "x": 0.0,
            "xanchor": "left",
            "y": -0.24,
            "yanchor": "top",
        },
        margin={"b": 110},
    )
    return result


def _visible_x_window(relayout_data):
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
        a, b = pd.Timestamp(left), pd.Timestamp(right)
    except Exception:
        return None
    return min(a, b), max(a, b)


def _adaptive_hd_yaxis(
    fig: go.Figure,
    relayout_data: Mapping[str, Any] | None,
) -> go.Figure:
    window = _visible_x_window(relayout_data)
    stacked: dict[pd.Timestamp, list[float]] = {}
    lines: list[float] = []
    for trace in fig.data:
        xs = getattr(trace, "x", None)
        ys = getattr(trace, "y", None)
        if xs is None or ys is None:
            continue
        x = pd.to_datetime(pd.Series(list(xs)), errors="coerce")
        y = pd.to_numeric(pd.Series(list(ys)), errors="coerce")
        valid = x.notna() & y.notna()
        if window is not None:
            valid &= (x >= window[0]) & (x <= window[1])
        for stamp, value in zip(x.loc[valid], y.loc[valid].astype(float)):
            if not np.isfinite(value):
                continue
            if getattr(trace, "type", "") == "bar":
                key = pd.Timestamp(stamp)
                stacked.setdefault(key, [0.0, 0.0])
                if value >= 0:
                    stacked[key][0] += float(value)
                else:
                    stacked[key][1] += float(value)
            else:
                lines.append(float(value))
    values = list(lines)
    for positive, negative in stacked.values():
        values.extend([positive, negative])
    values.append(0.0)
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        span = hi - lo
        pad = 0.08 * (span if span > 0 else max(abs(lo), abs(hi), 1.0))
        fig.update_yaxes(range=[lo - pad, hi + pad], autorange=False)
    if window is not None:
        fig.update_xaxes(range=[window[0], window[1]], autorange=False)
    elif bool(dict(relayout_data or {}).get("xaxis.autorange")):
        fig.update_xaxes(autorange=True)
    return fig


def historical_decomposition_figure(
    payload: Mapping[str, Any] | None,
    *,
    response: str | None,
    last_obs: int | None = None,
    relayout_data: Mapping[str, Any] | None = None,
) -> go.Figure:
    if not payload:
        return _empty_figure("Loading Headline structural analysis…", height=560)
    if not payload.get("ok"):
        return _empty_figure(
            "Headline Structural failed: " + str(payload.get("error", "unknown error")),
            height=560,
        )
    if response not in payload["variables"]:
        return _empty_figure("Select a valid response variable.", height=560)

    frame = pd.DataFrame(payload["hd"])
    part = frame.loc[frame["response"] == response].copy()
    part["date"] = pd.to_datetime(part["date"])
    part = part.sort_values("date")
    if last_obs is not None and int(last_obs) > 0:
        part = part.tail(int(last_obs))
    if part.empty:
        return _empty_figure("No Headline historical decomposition is available.", height=560)

    variables = list(payload.get("variables", []))
    labels = dict(payload.get("labels", {}) or {})
    fig = go.Figure()
    for component in payload["hd_component_names"]:
        if component not in part.columns:
            continue
        base_name = str(component).split(":", 1)[0].strip()
        try:
            colour_index = variables.index(base_name)
        except ValueError:
            colour_index = 0
        colour = _STRUCTURAL_COLOURS[
            colour_index % len(_STRUCTURAL_COLOURS)
        ]
        is_outlier = "outlier amplification" in str(component).lower()
        label = str(component)
        for var in variables:
            if label.startswith(var):
                label = label.replace(var, labels.get(var, _label(var)), 1)
                break
        fig.add_trace(go.Bar(
            x=part["date"],
            y=part[component],
            name=label.replace("_", " ").title(),
            marker={"color": colour},
            opacity=0.36 if is_outlier else 0.88,
            hovertemplate="%{y:.3f}<extra>" + label.replace("_", " ").title() + "</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=part["date"],
        y=part["observed"],
        mode="lines",
        line={"width": 2.4, "color": "#0F172A"},
        name=str(payload.get("hd_observed_label", "Observed 12m log inflation")),
    ))
    fig.add_trace(go.Scatter(
        x=part["date"],
        y=part["base"],
        mode="lines",
        line={"width": 1.5, "dash": "dash", "color": "#64748B"},
        name="Base / initial conditions",
    ))
    fig.update_layout(barmode="relative", bargap=0.0)
    fig.add_hline(y=0.0, line_width=1.1, line_color="#94A3B8")
    fig = _layout(
        fig,
        y_title="12m contribution (100 × log points)",
        height=560,
        uirevision=(
            f"{payload.get('run_id')}::headline-hd::{response}::{last_obs}"
        ),
    )
    return _adaptive_hd_yaxis(fig, relayout_data)


def _stat_card(title: str, value_id: str, subtitle_id: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="stat-title"),
            html.Div("—", id=value_id, className="stat-value"),
            html.Div("", id=subtitle_id, className="stat-subtitle"),
        ],
        className="stat-card",
    )


# HEADLINE_STRUCTURAL_PRESENTATION_V1
# HEADLINE_STRUCTURAL_LABELS_LOTA_V1
# HEADLINE_STRUCTURAL_LAYOUT_LOTB_V1
# HEADLINE_STRUCTURAL_FOLLOWUP_V1_2
def headline_structural_page() -> html.Div:
    """Energy-style Structural workspace for the Headline joint BVAR."""
    badge_style = {
        "display": "inline-flex",
        "alignItems": "center",
        "gap": "6px",
        "padding": "6px 10px",
        "borderRadius": "999px",
        "border": "1px solid #DCE3EC",
        "background": "#F8FAFC",
        "fontSize": "12px",
        "fontWeight": 600,
        "color": "#334155",
    }
    return html.Div(
        [
            dcc.Store(id="headline-structural-live-store", storage_type="memory"),
            dcc.Store(id="headline-total-structural-store", storage_type="memory"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Structural analysis", className="page-title"),
                            html.P(
                                "Recursive structural analysis of the locked four-variable Headline BVAR. "
                                "The selected reference date sets the joint regular SV state for 1σ IRFs and FEVD; "
                                "recursive historical decomposition uses the realised historical volatility path.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span("Recursive / Cholesky", style=badge_style),
                            html.Span(
                                "Energy → Food → NEIG → Services",
                                style=badge_style,
                            ),
                            html.Span("H ≤ 12m", style=badge_style),
                        ],
                        style={
                            "display": "flex",
                            "gap": "8px",
                            "flexWrap": "wrap",
                            "justifyContent": "flex-end",
                        },
                    ),
                ],
                className="page-heading-row",
            ),
            html.Div(id="headline-structural-run-banner", className="selection-banner"),

            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Horizon", className="control-label"),
                                    dcc.Input(
                                        id="headline-structural-horizon",
                                        type="number",
                                        min=1,
                                        max=MAX_HORIZON,
                                        step=1,
                                        value=6,
                                        className="est-profile-name-input",
                                    ),
                                ],
                                className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Label("Posterior draws", className="control-label"),
                                    dcc.Input(
                                        id="headline-structural-draws",
                                        type="number",
                                        min=1,
                                        max=MAX_STRUCTURAL_DRAWS,
                                        step=1,
                                        value=DEFAULT_STRUCTURAL_DRAWS,
                                        className="est-profile-name-input",
                                    ),
                                ],
                                className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "IRF shock definition",
                                        className="control-label",
                                    ),
                                    dcc.RadioItems(
                                        id="headline-structural-shock-unit",
                                        options=[
                                            {
                                                "label": "Standard deviation",
                                                "value": "structural_std",
                                            },
                                            {
                                                "label": "Level impact",
                                                "value": "level",
                                            },
                                        ],
                                        value=DEFAULT_SHOCK_UNIT,
                                        inline=True,
                                        className="fan-radio",
                                    ),
                                ],
                                className="control-block wide-control",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Magnitude",
                                        id="headline-structural-shock-size-label",
                                        className="control-label",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="headline-structural-shock-size",
                                                type="number",
                                                min=1e-6,
                                                step=0.1,
                                                value=DEFAULT_SHOCK_SIZE,
                                                className="est-profile-name-input",
                                                style={
                                                    "minWidth": "110px",
                                                    "width": "100%",
                                                },
                                            ),
                                            html.Span(
                                                "σ",
                                                id="headline-structural-shock-size-unit",
                                                style={
                                                    "fontSize": "12px",
                                                    "fontWeight": 600,
                                                    "whiteSpace": "nowrap",
                                                    "color": "#6B6E72",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "8px",
                                        },
                                    ),
                                ],
                                className="control-block",
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": (
                                "minmax(110px,.45fr) minmax(130px,.5fr) "
                                "minmax(250px,1fr) minmax(180px,.75fr)"
                            ),
                            "gap": "14px",
                            "alignItems": "end",
                        },
                    ),
                    html.Div(
                        id="headline-structural-shock-interpretation",
                        className="selection-banner",
                        style={"marginTop": "12px", "marginBottom": "2px"},
                    ),
                    html.Div(
                        [
                            dcc.Checklist(
                                id="headline-structural-hd-options",
                                options=[
                                    {
                                        "label": (
                                            "Split realised outlier amplification "
                                            "in historical decomposition"
                                        ),
                                        "value": "split_outliers",
                                    }
                                ],
                                value=["split_outliers"],
                                className="estimation-checklist",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Refresh structural analysis",
                                        id="headline-structural-run",
                                        n_clicks=0,
                                        className=(
                                            "refresh-button "
                                            "estimation-run-all-button"
                                        ),
                                    ),
                                    html.Button(
                                        "Cancel",
                                        id="headline-structural-cancel",
                                        n_clicks=0,
                                        disabled=True,
                                        className="estimation-cancel-button",
                                    ),
                                ],
                                className="estimation-actions",
                            ),
                        ],
                        className="estimation-run-row",
                    ),
                    html.Div(
                        [
                            html.Progress(
                                id="headline-structural-progress",
                                value=0,
                                max=100,
                                className="estimation-progress",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Idle",
                                        id="headline-structural-phase",
                                        className="estimation-phase",
                                    ),
                                    html.Div(
                                        "Opening Headline Structural automatically computes the default analysis from the selected saved posterior.",
                                        id="headline-structural-progress-detail",
                                        className="estimation-progress-detail",
                                    ),
                                ],
                                className="estimation-progress-text",
                            ),
                        ],
                        className="estimation-progress-wrap",
                    ),
                ],
                className="panel",
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        "Reference structural-volatility state",
                                        className="panel-title",
                                    ),
                                    html.P(
                                        "There is no single BVAR volatility parameter. Each structural shock has its own λⱼ,ₜ. "
                                        "A reference date selects all four simultaneously. √λ is shown in 100 × log points; "
                                        "the relative-state diagnostics are scale free.",
                                        className="panel-subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Exact reference date",
                                        className="control-label",
                                    ),
                                    dcc.Dropdown(
                                        id="headline-structural-reference-date",
                                        options=[],
                                        value=None,
                                        clearable=False,
                                        className="compact-dropdown wide-control",
                                    ),
                                ],
                                className="control-block wide-control",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(
                        [
                            html.Div("Regime", className="control-label"),
                            dcc.RadioItems(
                                id="headline-structural-reference-regime",
                                options=[
                                    {"label": "Latest", "value": "latest"},
                                    {"label": "Calm · P10", "value": "p10"},
                                    {"label": "Median · P50", "value": "p50"},
                                    {"label": "Stressed · P90", "value": "p90"},
                                    {"label": "Peak 2022", "value": "peak_2022"},
                                ],
                                value="latest",
                                inline=True,
                                className="fan-radio",
                            ),
                        ],
                        style={"padding": "0 2px 12px 2px"},
                    ),
                    html.Div(
                        id="headline-structural-reference-status",
                        style={"marginBottom": "12px"},
                    ),
                    html.Div(
                        id="headline-structural-volatility-cards",
                        style={
                            "display": "grid",
                            "gridTemplateColumns": (
                                "repeat(auto-fit, minmax(260px, 1fr))"
                            ),
                            "gap": "12px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "Relative structural-variance state",
                                        style={
                                            "fontWeight": 700,
                                            "fontSize": "13px",
                                        },
                                    ),
                                    html.Div(
                                        "λⱼ,ₜ divided by that shock's own sample-median λⱼ. "
                                        "This is a dimensionless state diagnostic, not a FEVD share.",
                                        style={
                                            "fontSize": "11px",
                                            "color": "#64748B",
                                            "marginTop": "2px",
                                        },
                                    ),
                                ],
                                style={"padding": "14px 14px 0 14px"},
                            ),
                            dcc.Graph(
                                id="headline-structural-volatility-relative-graph",
                                config={
                                    "displayModeBar": False,
                                    "responsive": True,
                                },
                            ),
                        ],
                        style={
                            "marginTop": "14px",
                            "border": "1px solid #E2E8F0",
                            "borderRadius": "12px",
                            "background": "#FBFCFE",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                "What the selected date changes",
                                style={
                                    "fontWeight": 700,
                                    "fontSize": "13px",
                                    "marginBottom": "8px",
                                },
                            ),
                            html.Div(
                                id="headline-structural-reference-effects",
                                style={
                                    "display": "flex",
                                    "gap": "8px",
                                    "flexWrap": "wrap",
                                },
                            ),
                        ],
                        style={"marginTop": "14px"},
                    ),
                ],
                className="panel chart-panel",
            ),

            html.Div(
                [
                    _stat_card(
                        "Identification",
                        "headline-structural-stat-identification",
                        "headline-structural-stat-ordering",
                    ),
                    _stat_card(
                        "Computed reference",
                        "headline-structural-stat-reference",
                        "headline-structural-stat-frequency",
                    ),
                    _stat_card(
                        "Posterior draws",
                        "headline-structural-stat-draws",
                        "headline-structural-stat-draws-total",
                    ),
                    _stat_card(
                        "HD reconstruction",
                        "headline-structural-stat-hd-error",
                        "headline-structural-stat-hd-error-note",
                    ),
                    _stat_card(
                        "FEVD sum error",
                        "headline-structural-stat-fevd-error",
                        "headline-structural-stat-fevd-error-note",
                    ),
                    _stat_card(
                        "IRF shock scale",
                        "headline-structural-stat-shock-scale",
                        "headline-structural-stat-shock-scale-note",
                    ),
                ],
                className="stats-grid",
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H3(
                                                "Impulse responses",
                                                className="panel-title",
                                            ),
                                            html.P(
                                                'Structural timing: the shock hits at the reference date; h=0 is the contemporaneous impact.',
                                                className="panel-subtitle",
                                            ),
                                            html.P(
                                                "A 1σ IRF uses the shocked equation's √λ at the computed reference date. "
                                                "Level-impact IRFs are normalised in 100 × log points and are exactly invariant "
                                                "to the selected SV date.",
                                                className="panel-subtitle",
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Shock",
                                                        className="control-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="headline-structural-shock",
                                                        options=[],
                                                        clearable=False,
                                                        className=(
                                                            "compact-dropdown wide-control"
                                                        ),
                                                    ),
                                                ],
                                                className="control-block wide-control",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Response",
                                                        className="control-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="headline-structural-response",
                                                        options=[],
                                                        clearable=False,
                                                        className=(
                                                            "compact-dropdown wide-control"
                                                        ),
                                                    ),
                                                ],
                                                className="control-block wide-control",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "IRF object",
                                                        className="control-label",
                                                    ),
                                                    dcc.RadioItems(
                                                        id="headline-structural-irf-metric",
                                                        options=[
                                                            {
                                                                "label": "Cumulative level",
                                                                "value": "cumulative",
                                                            },
                                                            {
                                                                "label": "Period change",
                                                                "value": "change",
                                                            },
                                                        ],
                                                        value="cumulative",
                                                        inline=True,
                                                        className="fan-radio",
                                                    ),
                                                ],
                                                className="control-block",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Fan",
                                                        className="control-label",
                                                    ),
                                                    dcc.RadioItems(
                                                        id="headline-structural-irf-fan",
                                                        options=[
                                                            {
                                                                "label": "68%",
                                                                "value": "68",
                                                            },
                                                            {
                                                                "label": "90%",
                                                                "value": "90",
                                                            },
                                                            {
                                                                "label": "Both",
                                                                "value": "both",
                                                            },
                                                        ],
                                                        value="68",
                                                        inline=True,
                                                        className="fan-radio",
                                                    ),
                                                ],
                                                className="control-block",
                                            ),
                                        ],
                                        className="chart-controls",
                                    ),
                                ],
                                className="panel-heading",
                            ),
                            readable_table("headline-structural-irf-table", IRF_COLUMNS, page_size=7),
                            dcc.Loading(
                                dcc.Graph(
                                    id="headline-structural-irf",
                                    config=graph_config("headline_irf_interactive"),
                                ),
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
                                    html.Div(
                                        [
                                            html.H3(
                                                "Headline Total response",
                                                className="panel-title",
                                            ),
                                            html.P(
                                                'Structural timing: the shock hits at the reference date; h=0 is the contemporaneous impact.',
                                                className="panel-subtitle",
                                            ),
                                            html.P(
                                                "Exact nonlinear re-aggregation of the four component structural responses "
                                                "through the production Headline chain-link engine. Response is always "
                                                "Headline HICP Total; no aggregate FEVD or historical decomposition is constructed.",
                                                className="panel-subtitle",
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Shock", className="control-label"),
                                                    html.P("Shared with component IRF", className="panel-subtitle"),
                                                ],
                                                className="control-block",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Metric", className="control-label"),
                                                    dcc.RadioItems(
                                                        id="headline-total-structural-metric",
                                                        options=[
                                                            {
                                                                "label": "Level response [%]",
                                                                "value": "level_response_pct",
                                                            },
                                                            {
                                                                "label": "YoY response [pp]",
                                                                "value": "yoy_response_pp",
                                                            },
                                                        ],
                                                        value="level_response_pct",
                                                        inline=True,
                                                        className="fan-radio",
                                                    ),
                                                ],
                                                className="control-block wide-control",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Posterior band", className="control-label"),
                                                    dcc.RadioItems(
                                                        id="headline-total-structural-fan",
                                                        options=[
                                                            {"label": "68%", "value": "68"},
                                                            {"label": "Mean only", "value": "mean"},
                                                        ],
                                                        value="68",
                                                        inline=True,
                                                        className="fan-radio",
                                                    ),
                                                ],
                                                className="control-block",
                                            ),
                                        ],
                                        className="chart-controls",
                                    ),
                                ],
                                className="panel-heading",
                            ),
                            html.Div(
                                id="headline-total-structural-status",
                                className="selection-banner",
                            ),
                            html.P(
                                id="headline-total-structural-effective-parameters",
                                className="panel-subtitle",
                            ),
                            html.P(
                                id="headline-total-structural-weights",
                                className="panel-subtitle",
                            ),
                            readable_table("headline-total-structural-table", [{'name': 'Horizon', 'id': 'horizon'}, {'name': 'Model horizon', 'id': 'model_h'}, {'name': 'Posterior mean', 'id': 'mean'}, {'name': '68% interval', 'id': 'interval68'}], page_size=13),
                            html.P(
                                id="headline-total-structural-yoy-base-note",
                                className="panel-subtitle",
                                style={"marginTop": "6px"},
                            ),
                            dcc.Loading(
                                dcc.Graph(
                                    id="headline-total-structural-irf",
                                    config=graph_config("headline_total_structural_irf"),
                                ),
                                type="circle",
                            ),
                        ],
                        className="panel chart-panel",
                    ),
                            html.Div(
                                [
                                html.Div(
                                    [
                                        html.H3(
                                            "Forecast error variance decomposition",
                                            className="panel-title",
                                        ),
                                        html.P(
                                            "The table reports the full posterior-median FEVD. "
                                            "The chart isolates cross-shock transmission and excludes "
                                            "the response variable\'s own shock.",
                                            className="panel-subtitle",
                                        ),
                                    ],
                                    className="panel-heading",
                                ),
                                readable_table("headline-structural-fevd-table", [{"name":"Shock","id":"shock"}], page_size=12),
                                dcc.Loading(
                                    dcc.Graph(
                                        id="headline-structural-fevd",
                                        config=graph_config("headline_fevd_interactive"),
                                    ),
                                    type="circle",
                                ),
                                ],
                                className="panel chart-panel structural-fevd-panel",
                            ),
                        ],
                        className="structural-total-fevd-group",
                    ),
                ],
                className="structural-comparison-grid",
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        "Historical decomposition",
                                        className="panel-title",
                                    ),
                                    html.P(
                                        "Posterior-mean 12-month log-inflation contributions preserve exact additivity. "
                                        "Under recursive identification the selected reference date and IRF magnitude do not "
                                        "change the HD; each historical observation uses its realised SV/outlier state.",
                                        className="panel-subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "History window",
                                        className="control-label",
                                    ),
                                    dcc.Dropdown(
                                        id="headline-structural-hd-window",
                                        options=[
                                            {
                                                "label": "Last 5 years",
                                                "value": 60,
                                            },
                                            {
                                                "label": "Last 10 years",
                                                "value": 120,
                                            },
                                            {
                                                "label": "Last 13 years",
                                                "value": 156,
                                            },
                                            {
                                                "label": "Full sample",
                                                "value": -1,
                                            },
                                        ],
                                        value=-1,
                                        clearable=False,
                                        className="compact-dropdown",
                                    ),
                                ],
                                className="control-block",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    readable_table("headline-structural-hd-table", HD_COLUMNS, page_size=20),
                    dcc.Loading(
                        dcc.Graph(
                            id="headline-structural-hd",
                            config=graph_config("headline_hd_interactive"),
                        ),
                        type="circle",
                    ),
                ],
                className="panel chart-panel",
            ),
        ],
        className="page-body",
    )


def _selected_run_directory(
    results_root: Path,
    store: Mapping[str, Any] | None,
) -> Path:
    context = dict((store or {}).get("context") or {})
    if str(context.get("model_id") or "") != "headline_joint":
        raise HeadlineStructuralDashboardError(
            "Select a Headline HICP saved run before opening Structural."
        )
    vintage = str(context.get("vintage") or "")
    run_id = str(context.get("run_id") or "")
    if not vintage or not run_id:
        raise HeadlineStructuralDashboardError(
            "Headline Structural requires vintage and run_id."
        )
    return resolve_headline_run_directory(
        results_root,
        vintage=vintage,
        run_id=run_id,
    )


def register_headline_structural_callbacks(
    app,
    *,
    results_root: str | Path,
    project_root: str | Path,
    store_id: str = "data-store",
) -> None:
    """Register Headline Structural callbacks on the unified Dash app."""
    results_root = Path(results_root).resolve()
    project_root = Path(project_root).resolve()

    @app.callback(
        Output("headline-structural-run-banner", "children"),
        Output("headline-structural-reference-date", "options"),
        Output("headline-structural-shock", "options"),
        Output("headline-structural-shock", "value"),
        Output("headline-structural-response", "options"),
        Output("headline-structural-response", "value"),
        Input(store_id, "data"),
        Input("url", "pathname"),
        State("headline-structural-shock", "value"),
        State("headline-structural-response", "value"),
    )
    def controls(store, pathname, current_shock, current_response):
        if (pathname or "") not in {"/structural/headline", "/headline/structural"}:
            raise PreventUpdate
        try:
            directory = _selected_run_directory(results_root, store)
            contract = headline_structural_run_contract(
                directory,
                project_root=project_root,
            )
            variables = list(contract["variables"])
            labels = dict(contract["labels"])
            dates = pd.to_datetime(contract["reference_dates"])
            date_options = [
                {
                    "label": pd.Timestamp(date).strftime("%Y-%m"),
                    "value": pd.Timestamp(date).isoformat(),
                }
                for date in dates
            ]
            variable_options = [
                {
                    "label": labels.get(name, _label(name)),
                    "value": name,
                }
                for name in variables
            ]
            banner = html.Div(
                [
                    html.Strong("Headline HICP joint BVAR"),
                    html.Span(f" · vintage {contract['vintage']}"),
                    html.Span(
                        f" · run {str(contract['run_id'])[:12]}"
                    ),
                    html.Span(
                        f" · {contract['available_posterior_draws']:,} posterior draws"
                    ),
                    html.Span(
                        " · recursive ordering: "
                        + " → ".join(
                            labels.get(name, _label(name))
                            for name in variables
                        )
                    ),
                    html.Span(
                        f" · missing data: {str(contract['missing_data_method']).upper()}"
                    ),
                ]
            )
            return (
                banner,
                date_options,
                variable_options,
                current_shock if current_shock in variables else variables[0],
                variable_options,
                current_response if current_response in variables else variables[-1],
            )
        except Exception as exc:
            return (
                html.Div(
                    [
                        html.Strong("Headline Structural unavailable: "),
                        html.Span(str(exc)),
                    ],
                    className="estimation-error-text",
                ),
                [],
                [],
                None,
                [],
                None,
            )

    @app.callback(
        Output("headline-structural-reference-date", "value"),
        Input("headline-structural-reference-regime", "value"),
        Input(store_id, "data"),
        State("headline-structural-live-store", "data"),
        State("headline-structural-reference-date", "options"),
        State("headline-structural-reference-date", "value"),
        prevent_initial_call=False,
    )
    def reference_regime(regime, store, structural_store, options, current):
        values = [
            item.get("value")
            for item in (options or [])
            if item.get("value")
        ]
        if not values:
            return None
        trigger = ctx.triggered_id
        if trigger == store_id:
            return values[-1]
        if trigger == "headline-structural-reference-regime":
            if str(regime or "latest") == "latest":
                return values[-1]
            same_run = bool(
                structural_store
                and structural_store.get("ok")
                and store
                and str(structural_store.get("run_id"))
                == str(((store.get("context") or {}).get("run_id")))
            )
            if not same_run:
                return values[-1]
            chosen = reference_regime_date(
                structural_store,
                str(regime),
            )
            return chosen if chosen in set(values) else values[-1]
        return current if current in set(values) else values[-1]

    @app.callback(
        Output("headline-structural-shock-size-label", "children"),
        Output("headline-structural-shock-size-unit", "children"),
        Output("headline-structural-shock-interpretation", "children"),
        Input("headline-structural-shock-unit", "value"),
        Input("headline-structural-shock-size", "value"),
        Input("headline-structural-shock", "value"),
    )
    def shock_definition(shock_unit, shock_size, shock):
        mode = str(shock_unit or DEFAULT_SHOCK_UNIT)
        try:
            size = float(shock_size)
        except (TypeError, ValueError):
            size = DEFAULT_SHOCK_SIZE
        shock_name = _label(str(shock or "selected shock"))
        if mode == "level":
            pct = 100.0 * (np.exp(size / 100.0) - 1.0)
            return (
                "Impact size",
                "100 × log points",
                html.Div(
                    [
                        html.Strong("Level-normalised IRF · "),
                        html.Span(
                            f"{shock_name} moves by +{size:g} in 100×log(HICP) on impact "
                            f"(≈ {pct:.3f}% in the HICP index). This magnitude applies to the IRF only. "
                            "FEVD remains defined from 1σ structural shocks; recursive HD is unchanged."
                        ),
                    ]
                ),
            )
        return (
            "Shock magnitude",
            "σ",
            html.Div(
                [
                    html.Strong("Structural-SD IRF · "),
                    html.Span(
                        f"{size:g}σ at the selected joint SV reference state. "
                        "The IRF scale depends on the shocked equation's λ at that date. "
                        "FEVD always uses 1σ shocks; recursive HD is unchanged."
                    ),
                ]
            ),
        )

    @app.callback(
        output=Output("headline-structural-live-store", "data"),
        inputs=[
            Input("headline-structural-run", "n_clicks"),
            Input("url", "pathname"),
            Input(store_id, "data"),
            Input("headline-structural-reference-date", "value"),
            Input("headline-structural-horizon", "value"),
            Input("headline-structural-draws", "value"),
            Input("headline-structural-shock-unit", "value"),
            Input("headline-structural-shock-size", "value"),
            Input("headline-structural-hd-options", "value"),
        ],
        state=[
            State("headline-structural-live-store", "data"),
        ],
        background=True,
        running=[
            (
                Output("headline-structural-run", "disabled"),
                True,
                False,
            ),
            (
                Output("headline-structural-cancel", "disabled"),
                False,
                True,
            ),
        ],
        cancel=[Input("headline-structural-cancel", "n_clicks")],
        progress=[
            Output("headline-structural-progress", "value"),
            Output("headline-structural-phase", "children"),
            Output("headline-structural-progress-detail", "children"),
        ],
        progress_default=(
            0,
            "Idle",
            "Opening Headline Structural automatically computes the default analysis once; unchanged revisits are no-ops.",
        ),
        prevent_initial_call=False,
    )
    def compute(
        set_progress,
        n_clicks,
        pathname,
        store,
        reference_date,
        horizon,
        posterior_draws,
        shock_unit,
        shock_size,
        hd_options,
        current_live_store,
    ):
        if (pathname or "") not in {"/structural/headline", "/headline/structural"}:
            raise PreventUpdate

        context = dict((store or {}).get("context") or {})
        if str(context.get("model_id") or "") != "headline_joint":
            raise PreventUpdate
        try:
            size = float(shock_size or DEFAULT_SHOCK_SIZE)
        except (TypeError, ValueError):
            size = float(DEFAULT_SHOCK_SIZE)
        request_signature = {
            "vintage": str(context.get("vintage") or ""),
            "run_id": str(context.get("run_id") or ""),
            "reference_date": None if reference_date is None else pd.Timestamp(reference_date).isoformat(),
            "horizon": int(horizon or DEFAULT_HORIZON),
            "posterior_draws": int(posterior_draws or DEFAULT_STRUCTURAL_DRAWS),
            "shock_unit": str(shock_unit or DEFAULT_SHOCK_UNIT),
            "shock_size": size,
            "split_outlier_amplification": "split_outliers" in set(hd_options or []),
        }
        manual_refresh = ctx.triggered_id == "headline-structural-run"
        if (
            not manual_refresh
            and isinstance(current_live_store, dict)
            and current_live_store.get("ok")
            and current_live_store.get("dashboard_request_signature") == request_signature
        ):
            # Returning to the page with the same saved run and controls must
            # not recompute or rewrite the structural object.
            raise PreventUpdate

        try:
            directory = _selected_run_directory(results_root, store)
            set_progress(
                (
                    10,
                    "Loading posterior",
                    "Reading the persisted Headline posterior and verified lineage.",
                )
            )
            set_progress(
                (
                    30,
                    "Recursive identification",
                    "Computing regular structural impact matrices, IRFs, FEVD and 12m HD.",
                )
            )
            payload = compute_headline_structural(
                directory,
                project_root=project_root,
                reference_date=reference_date,
                horizon=request_signature["horizon"],
                posterior_draws=request_signature["posterior_draws"],
                shock_unit=request_signature["shock_unit"],
                shock_size=request_signature["shock_size"],
                split_outlier_amplification=request_signature["split_outlier_amplification"],
            )
            payload["dashboard_request_signature"] = request_signature
            set_progress(
                (
                    100,
                    "Structural analysis ready",
                    "The selected structural state is frozen until one of its parameters changes.",
                )
            )
            return payload
        except Exception as exc:
            set_progress(
                (
                    0,
                    "Structural analysis failed",
                    str(exc).splitlines()[0],
                )
            )
            return {
                "ok": False,
                "dashboard_request_signature": request_signature,
                "error": f"{type(exc).__name__}: {exc}",
            }


    @app.callback(
        Output("headline-structural-volatility-cards", "children"),
        Output(
            "headline-structural-volatility-relative-graph",
            "figure",
        ),
        Output("headline-structural-reference-effects", "children"),
        Output("headline-structural-reference-status", "children"),
        Input("headline-structural-live-store", "data"),
        Input("headline-structural-reference-date", "value"),
        Input("headline-structural-shock-unit", "value"),
    )
    def volatility_state(structural_store, reference_date, shock_unit):
        if not structural_store or not structural_store.get("ok"):
            return (
                [],
                relative_volatility_state_figure(
                    structural_store,
                    reference_date=reference_date,
                ),
                [],
                "",
            )

        snapshot = reference_volatility_snapshot(
            structural_store,
            reference_date,
        )
        cards = []
        for item in snapshot.get("cards", []):
            sv = item.get("persistent_q50")
            ratio = item.get("relative_sd")
            percentile = item.get("percentile")
            outlier_p = item.get("outlier_probability")
            cards.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    item["label"],
                                    style={
                                        "fontWeight": 700,
                                        "fontSize": "13px",
                                        "color": "#0F172A",
                                    },
                                ),
                                html.Div(
                                    (
                                        "—"
                                        if not np.isfinite(percentile)
                                        else f"P{int(round(percentile))}"
                                    ),
                                    style={
                                        "fontSize": "11px",
                                        "fontWeight": 700,
                                        "color": "#64748B",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                            },
                        ),
                        dcc.Graph(
                            figure=volatility_sparkline_figure(
                                structural_store,
                                variable=item["variable"],
                                reference_date=reference_date,
                            ),
                            config={
                                "displayModeBar": False,
                                "responsive": True,
                            },
                            style={"height": "105px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "√λ ",
                                            style={"color": "#64748B"},
                                        ),
                                        html.Strong(
                                            "—"
                                            if not np.isfinite(sv)
                                            else f"{float(sv):.3f}"
                                        ),
                                        html.Span(" · 100×log pts"),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Strong(
                                            "—"
                                            if not np.isfinite(ratio)
                                            else f"×{float(ratio):.2f}"
                                        ),
                                        html.Span(
                                            " own median",
                                            style={"color": "#64748B"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    (
                                        ""
                                        if not np.isfinite(outlier_p)
                                        else f"P(outlier) {float(outlier_p):.0%}"
                                    ),
                                    style={
                                        "color": "#94A3B8",
                                        "fontSize": "10px",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "gap": "10px",
                                "fontSize": "11px",
                                "alignItems": "baseline",
                            },
                        ),
                    ],
                    style={
                        "border": "1px solid #E2E8F0",
                        "borderRadius": "12px",
                        "background": "#FFFFFF",
                        "padding": "12px 14px 10px 14px",
                        "boxShadow": "0 1px 2px rgba(15,23,42,.04)",
                    },
                )
            )

        chip = {
            "display": "inline-flex",
            "alignItems": "center",
            "padding": "7px 10px",
            "borderRadius": "999px",
            "fontSize": "11px",
            "fontWeight": 700,
            "border": "1px solid #DCE3EC",
            "background": "#F8FAFC",
            "color": "#334155",
        }
        green = {
            **chip,
            "color": "#166534",
            "background": "#F0FDF4",
            "borderColor": "#BBF7D0",
        }
        irf_text = (
            "IRF 1σ · date-dependent"
            if str(shock_unit or DEFAULT_SHOCK_UNIT)
            == "structural_std"
            else "IRF level-impact · invariant"
        )
        effects = [
            html.Span("✓ " + irf_text, style=green),
            html.Span("✓ FEVD · date-dependent", style=green),
            html.Span("— Recursive HD · invariant", style=chip),
            html.Span(
                "— Outlier multiplier · excluded from IRF / FEVD",
                style=chip,
            ),
        ]

        selected = (
            pd.Timestamp(reference_date)
            if reference_date
            else None
        )
        computed = (
            pd.Timestamp(structural_store.get("reference_date"))
            if structural_store.get("reference_date")
            else None
        )
        if (
            selected is not None
            and computed is not None
            and selected != computed
        ):
            status = html.Div(
                [
                    html.Strong(
                        f"Selected {selected.strftime('%Y-%m')} · "
                    ),
                    html.Span(
                        f"charts are still computed at {computed.strftime('%Y-%m')}. "
                        "Changing the structural reference state automatically refreshes the frozen analysis once."
                    ),
                ],
                style={
                    "padding": "9px 11px",
                    "borderRadius": "9px",
                    "border": "1px solid #FCD34D",
                    "background": "#FFFBEB",
                    "color": "#92400E",
                    "fontSize": "11px",
                },
            )
        else:
            ref_text = (
                computed.strftime("%Y-%m")
                if computed is not None
                else "—"
            )
            joint_p = snapshot.get("joint_percentile")
            status = html.Div(
                [
                    html.Strong(f"Computed state · {ref_text}"),
                    html.Span(
                        ""
                        if joint_p is None or not np.isfinite(joint_p)
                        else f" · joint stress P{int(round(joint_p))}"
                    ),
                ],
                style={
                    "padding": "9px 11px",
                    "borderRadius": "9px",
                    "border": "1px solid #BBF7D0",
                    "background": "#F0FDF4",
                    "color": "#166534",
                    "fontSize": "11px",
                },
            )
        return (
            cards,
            relative_volatility_state_figure(
                structural_store,
                reference_date=reference_date,
            ),
            effects,
            status,
        )

    @app.callback(
        Output("headline-structural-stat-identification", "children"),
        Output("headline-structural-stat-ordering", "children"),
        Output("headline-structural-stat-reference", "children"),
        Output("headline-structural-stat-frequency", "children"),
        Output("headline-structural-stat-draws", "children"),
        Output("headline-structural-stat-draws-total", "children"),
        Output("headline-structural-stat-hd-error", "children"),
        Output("headline-structural-stat-hd-error-note", "children"),
        Output("headline-structural-stat-fevd-error", "children"),
        Output("headline-structural-stat-fevd-error-note", "children"),
        Output("headline-structural-stat-shock-scale", "children"),
        Output("headline-structural-stat-shock-scale-note", "children"),
        Input("headline-structural-live-store", "data"),
    )
    def stats(structural_store):
        if not structural_store or not structural_store.get("ok"):
            return (
                "—", "", "—", "", "—", "",
                "—", "", "—", "", "—", "",
            )
        diag = dict(structural_store.get("diagnostics", {}) or {})
        labels = dict(structural_store.get("labels", {}) or {})
        ordering = " → ".join(
            labels.get(name, _label(name))
            for name in structural_store.get("recursive_ordering", [])
        )
        ref = pd.Timestamp(
            structural_store["reference_date"]
        ).strftime("%Y-%m")
        hd_error = diag.get(
            "hd_posterior_mean_12m_reconstruction_error"
        )
        fevd_error = diag.get("fevd_max_share_sum_error")
        shock_size = float(structural_store.get("shock_size", 1.0))
        level = str(structural_store.get("shock_unit")) == "level"
        return (
            "Recursive",
            ordering,
            ref,
            "Monthly",
            f"{int(structural_store.get('posterior_draws', 0)):,}",
            (
                f"of {int(structural_store.get('available_posterior_draws', 0)):,} "
                "saved draws"
            ),
            "—" if hd_error is None else f"{float(hd_error):.3e}",
            "posterior-mean 12m additive error",
            "—" if fevd_error is None else f"{float(fevd_error):.3e}",
            "max draw-wise |sum FEVD shares − 1|",
            (
                f"{shock_size:g} × 100-log pts"
                if level
                else f"{shock_size:g}σ"
            ),
            (
                "IRF impact-normalised; FEVD remains 1σ"
                if level
                else "IRF in structural standard deviations; FEVD remains 1σ"
            ),
        )

    @app.callback(
        Output("headline-structural-irf", "figure"),
        Output("headline-structural-irf-table", "data"),
        Input("headline-structural-live-store", "data"),
        Input("headline-structural-response", "value"),
        Input("headline-structural-shock", "value"),
        Input("headline-structural-irf-metric", "value"),
        Input("headline-structural-irf-fan", "value"),
    )
    def irf_graph(structural_store, response, shock, metric, fan_mode):
        resolved_metric = metric or "cumulative"
        return (
            irf_figure(
                structural_store, response=response, shock=shock,
                metric=resolved_metric, fan_mode=fan_mode or "68",
            ),
            structural_irf_records(
                structural_store, response=response, shock=shock,
                metric=resolved_metric,
            ),
        )

    @app.callback(
        Output("headline-structural-fevd", "figure"),
        Output("headline-structural-fevd-table", "data"),
        Output("headline-structural-fevd-table", "columns"),
        Input("headline-structural-live-store", "data"),
        Input("headline-structural-response", "value"),
    )
    def fevd_graph(structural_store, response):
        rows, columns = structural_fevd_table(structural_store, response=response)
        labels = (
            dict(structural_store.get("labels", {}) or {})
            if isinstance(structural_store, dict) else {}
        )
        variables = (
            list(structural_store.get("variables") or [])
            if isinstance(structural_store, dict) else []
        )
        response_label = labels.get(response, _label(response))

        display_label_map = {}
        for variable in variables:
            pretty = labels.get(variable, _label(variable))
            display_label_map[str(variable)] = pretty
            display_label_map[_label(variable)] = pretty
            display_label_map[str(variable).replace("_", " ").title()] = pretty

        cleaned_rows = []
        for row in rows or []:
            item = dict(row)
            raw = str(item.get("shock") or "")
            if raw in display_label_map:
                item["shock"] = display_label_map[raw]
            cleaned_rows.append(item)

        if columns:
            columns = [dict(column) for column in columns]
            columns[0]["name"] = f"Shock · FEVD of {response_label}"

        return (
            fevd_figure(structural_store, response=response),
            cleaned_rows,
            columns,
        )

    @app.callback(
        Output("headline-structural-hd", "figure"),
        Output("headline-structural-hd-table", "data"),
        Input("headline-structural-live-store", "data"),
        Input("headline-structural-response", "value"),
        Input("headline-structural-hd-window", "value"),
        Input("headline-structural-hd", "relayoutData"),
    )
    def hd_graph(structural_store, response, window, relayout_data):
        last_obs = (
            None
            if window is None or int(window) < 0
            else int(window)
        )
        return (
            historical_decomposition_figure(
                structural_store, response=response, last_obs=last_obs,
                relayout_data=relayout_data,
            ),
            structural_hd_records(structural_store, response=response),
        )
    # LOT3_BLOCK4_HEADLINE_TOTAL_STRUCTURAL_UI_V1
    def _headline_total_ui_contract(
        directory,
        *,
        horizon,
        posterior_draws,
        shock_unit,
        shock_size,
        reference_date,
    ):
        contract = headline_structural_run_contract(
            directory,
            project_root=project_root,
        )
        endpoint = (
            pd.Timestamp(contract["default_reference_date"])
            .to_period("M")
            .to_timestamp(how="start")
        )

        reasons = []
        try:
            resolved_horizon = int(horizon)
        except (TypeError, ValueError):
            resolved_horizon = None
            reasons.append("Horizon must be an integer in [1, 12].")
        if resolved_horizon is not None and not (1 <= resolved_horizon <= 12):
            reasons.append(
                f"Horizon {resolved_horizon}m is unsupported; "
                "headline-total-structural-irf-v1.1 requires H in [1, 12]."
            )

        try:
            resolved_draws = int(posterior_draws)
        except (TypeError, ValueError):
            resolved_draws = None
            reasons.append("Posterior draws must be an integer in [1, 1000].")
        if resolved_draws is not None and not (1 <= resolved_draws <= 1000):
            reasons.append(
                f"Posterior draws {resolved_draws} are outside the validated [1, 1000] range."
            )

        resolved_unit = str(shock_unit or "").strip().lower()
        if resolved_unit != "structural_std":
            reasons.append(
                "Shock definition is unsupported; Headline Total v1 requires "
                "Standard deviation (structural_std)."
            )

        try:
            resolved_size = float(shock_size)
        except (TypeError, ValueError):
            resolved_size = None
            reasons.append("Shock size must equal 1 structural standard deviation.")
        if resolved_size is not None and resolved_size != 1.0:
            reasons.append(
                f"Shock size {resolved_size:g} is unsupported; Headline Total v1 requires 1σ."
            )

        if reference_date in (None, ""):
            resolved_reference = endpoint
        else:
            resolved_reference = (
                pd.Timestamp(reference_date)
                .to_period("M")
                .to_timestamp(how="start")
            )
        if resolved_reference != endpoint:
            reasons.append(
                "Historical reference dates are unsupported by "
                "headline-total-structural-irf-v1.1; select the current structural endpoint "
                f"{endpoint.strftime('%Y-%m')}."
            )

        return {
            "compatible": not reasons,
            "reasons": reasons,
            "run_directory": str(Path(directory).resolve()),
            "vintage": str(contract.get("vintage") or ""),
            "run_id": str(contract.get("run_id") or ""),
            "horizon": resolved_horizon,
            "posterior_draws": resolved_draws,
            "shock_unit": resolved_unit,
            "shock_size": resolved_size,
            "reference_date": resolved_reference.date().isoformat(),
            "endpoint": endpoint.date().isoformat(),
            "contract": "headline-total-structural-irf-v1.1",
        }

    def _headline_total_weights_note(total_store):
        if not total_store or not total_store.get("ok"):
            return ""
        meta = dict(total_store.get("meta") or {})
        years = list(meta.get("headline_weight_year_used") or [])
        carried = list(meta.get("headline_weight_carried_forward") or [])
        dates = list(total_store.get("dates") or [])
        if not years or len(years) != len(carried) or len(years) != len(dates):
            return "Aggregation weights: exact production chain-link diagnostics unavailable."
        first_carried = next((i for i, flag in enumerate(carried) if bool(flag)), None)
        if first_carried is None:
            unique = sorted({int(year) for year in years})
            return (
                "Aggregation weights: published annual weights used throughout "
                + ", ".join(str(year) for year in unique)
                + "."
            )
        source_year = int(years[first_carried])
        carry_date = pd.Timestamp(dates[first_carried]).strftime("%b %Y")
        return (
            f"Aggregation weights: {source_year} published weights are carried forward "
            f"from {carry_date} where future annual weights are unavailable. "
            "This assumption is part of the exact nonlinear Headline response."
        )

    def _headline_total_effective_text(total_store, display_horizon=None):
        try:
            display_h = int(display_horizon)
        except (TypeError, ValueError):
            display_h = 12
        display_h = max(1, min(12, display_h))
        if not total_store or not total_store.get("ok"):
            return (
                "Headline Total computation contract: H=12 stored once; "
                f"displayed through H={display_h} · Recursive/Cholesky · "
                "1 structural std · current endpoint · future innovations=0 · "
                "future outliers=0."
            )
        meta = dict(total_store.get("meta") or {})
        stored_h = int(meta.get("horizon", 12))
        return (
            "Headline Total computation contract: "
            f"stored H={stored_h} · displayed H={display_h} · Recursive/Cholesky · "
            f"{float(meta.get('shock_size', 1.0)):g} structural std · "
            f"endpoint {str(meta.get('reference_date') or '—')[:7]} · "
            f"{int(meta.get('n_draws', 0)):,} posterior draws · "
            "future innovations=0 · future outliers=0. "
            "The band reflects saved posterior-draw uncertainty only."
        )

    def _headline_total_records(total_store, *, shock, metric, horizon):
        if not total_store or not total_store.get("ok"):
            return []
        shock_names = list(total_store.get("shock_names") or [])
        resolved_shock = shock if shock in shock_names else (shock_names[0] if shock_names else None)
        resolved_metric = metric if metric in {"level_response_pct", "yoy_response_pp"} else "level_response_pct"
        if resolved_shock is None:
            return []

        try:
            display_h = int(horizon)
        except (TypeError, ValueError):
            display_h = 12
        display_h = max(1, min(12, display_h))

        stats = dict(total_store.get("shocks") or {}).get(resolved_shock, {}).get(resolved_metric, {})
        mean = np.asarray(stats.get("mean") or [], dtype=float)
        q16 = np.asarray(stats.get("q16") or [], dtype=float)
        q84 = np.asarray(stats.get("q84") or [], dtype=float)
        available = min(len(mean), len(q16), len(q84), display_h + 1)
        rows = []
        for h in range(available):
            rows.append({
                "horizon": "Impact" if h == 0 else f"M+{h}",
                "model_h": f"h={h}m",
                "mean": fmt(mean[h], digits=4),
                "interval68": interval(q16[h], q84[h], digits=4),
            })
        return rows


    def _headline_total_figure(total_store, *, shock, metric, fan_mode, horizon):
        import plotly.graph_objects as go

        if not total_store or not total_store.get("ok"):
            return _empty_figure("Headline Total response")
        shock_names = list(total_store.get("shock_names") or [])
        resolved_shock = shock if shock in shock_names else (shock_names[0] if shock_names else None)
        resolved_metric = metric if metric in {"level_response_pct", "yoy_response_pp"} else "level_response_pct"
        if resolved_shock is None:
            return _empty_figure("Headline Total response")

        try:
            display_h = int(horizon)
        except (TypeError, ValueError):
            display_h = 12
        display_h = max(1, min(12, display_h))

        stats = dict(total_store.get("shocks") or {}).get(resolved_shock, {}).get(resolved_metric, {})
        dates = pd.to_datetime(total_store.get("dates") or [])[: display_h + 1]
        mean = np.asarray(stats.get("mean") or [], dtype=float)[: display_h + 1]
        q16 = np.asarray(stats.get("q16") or [], dtype=float)[: display_h + 1]
        q84 = np.asarray(stats.get("q84") or [], dtype=float)[: display_h + 1]
        if len(dates) == 0 or len(mean) != len(dates):
            return _empty_figure("Headline Total response")

        horizons = np.arange(len(mean), dtype=int)
        date_labels = np.asarray(
            [pd.Timestamp(value).strftime("%Y-%m") for value in dates],
            dtype=object,
        )

        labels = {
            "hicp_energy": "Energy", "hicp_food": "Food",
            "hicp_neig": "NEIG", "hicp_services": "Services",
        }
        unit = "%" if resolved_metric == "level_response_pct" else "pp"
        metric_label = "Level response [%]" if resolved_metric == "level_response_pct" else "YoY response [pp]"

        meta = dict(total_store.get("meta") or {})
        reference_value = meta.get("reference_date")
        if reference_value is None and len(dates):
            reference_value = dates[0]
        try:
            reference_label = pd.Timestamp(reference_value).strftime("%Y-%m")
        except Exception:
            reference_label = str(reference_value or "—")[:7]

        fig = go.Figure()
        if str(fan_mode or "68") == "68" and len(q16) == len(horizons) and len(q84) == len(horizons):
            fig.add_trace(go.Scatter(
                x=horizons, y=q84, mode="lines", line={"width": 0},
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=horizons, y=q16, mode="lines", line={"width": 0},
                fill="tonexty", fillcolor="rgba(0,159,227,0.16)",
                name="68% posterior interval", hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=horizons, y=mean, mode="lines+markers",
            customdata=date_labels,
            line={"color": "#009FE3", "width": 2.5},
            marker={"size": 4, "color": "#009FE3"},
            name="Posterior mean",
            hovertemplate=(
                "h=%{x} · %{customdata}<br>%{y:+.4f} "
                + unit
                + "<extra></extra>"
            ),
        ))
        fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#6B6E72")

        if (
            resolved_metric == "yoy_response_pp"
            and display_h >= 12
            and len(mean) > 12
        ):
            fig.add_vline(
                x=12,
                line_width=1,
                line_dash="dot",
                line_color="#A66A00",
            )

        fig.update_layout(
            title=(
                f"Headline Total — {labels.get(resolved_shock, resolved_shock)} shock · "
                f"{metric_label}"
                f"<br><sup>Structural reference {reference_label}; "
                "h=0 is the contemporaneous impact</sup>"
            ),
            paper_bgcolor="white", plot_bgcolor="white",
            margin={"l": 58, "r": 24, "t": 72, "b": 48},
            legend={"orientation": "h", "y": 1.08, "x": 0},
            hovermode="x unified",
        )
        fig.update_yaxes(title=unit, showgrid=True, gridcolor="#DEDCDD", zeroline=False)
        fig.update_xaxes(
            title="Horizon (months)",
            showgrid=False,
            tickmode="linear",
            tick0=0,
            dtick=1,
        )
        fig.update_layout(
            title_x=0.0,
            title_xanchor="left",
            legend={
                "orientation": "h",
                "x": 0.0,
                "xanchor": "left",
                "y": -0.24,
                "yanchor": "top",
            },
            margin={"b": 110},
        )
        return fig

    @app.callback(
        Output("headline-total-structural-store", "data"),
        Input("headline-structural-run", "n_clicks"),
        Input("url", "pathname"),
        Input(store_id, "data"),
        State("headline-structural-horizon", "value"),
        State("headline-structural-draws", "value"),
        State("headline-structural-shock-unit", "value"),
        State("headline-structural-shock-size", "value"),
        State("headline-structural-reference-date", "value"),
        prevent_initial_call=False,
    )
    def headline_total_compute(
        n_clicks,
        pathname,
        store,
        horizon,
        posterior_draws,
        shock_unit,
        shock_size,
        reference_date,
    ):
        import time

        if (pathname or "") not in {"/structural/headline", "/headline/structural"}:
            raise PreventUpdate
        try:
            directory = _selected_run_directory(results_root, store)
            ui = _headline_total_ui_contract(
                directory,
                horizon=horizon,
                posterior_draws=posterior_draws,
                shock_unit=shock_unit,
                shock_size=shock_size,
                reference_date=reference_date,
            )
            if not ui["compatible"]:
                return {
                    "ok": False, "status": "unsupported",
                    "contract": "headline-total-structural-irf-v1.1",
                    "reasons": list(ui["reasons"]), "ui": ui,
                }

            from headline_total_structural import run_saved_headline_total_structural_irf

            started = time.perf_counter()
            payload = run_saved_headline_total_structural_irf(
                directory,
                requested_draws=int(ui["posterior_draws"]),
                horizon=12,
                identification="recursive",
                reference_date=ui["reference_date"],
                shock_unit="structural_std",
                shock_size=1.0,
                include_outlier_scale=False,
                seed=42,
                project_root=project_root,
            )
            payload["ok"] = True
            payload["status"] = "fresh"
            payload["ui"] = ui
            payload["compute_seconds"] = float(time.perf_counter() - started)
            return payload
        except Exception as exc:
            return {
                "ok": False, "status": "error",
                "contract": "headline-total-structural-irf-v1.1",
                "error": f"{type(exc).__name__}: {exc}",
            }

    @app.callback(
        Output("headline-total-structural-yoy-base-note", "children"),
        Input("headline-total-structural-metric", "value"),
    )
    def headline_total_yoy_base_note(metric):
        if metric != "yoy_response_pp":
            return ""
        return (
            "YoY metric — at h=12, the 12-month comparison base is h=0, "
            "which is itself affected by the shock. The h=12 YoY response therefore "
            "measures the net 12-month effect relative to an already-shocked base "
            "and is not directly comparable with h<12."
        )
    @app.callback(
        Output("headline-total-structural-status", "children"),
        Output("headline-total-structural-effective-parameters", "children"),
        Output("headline-total-structural-weights", "children"),
        Output("headline-total-structural-table", "data"),
        Output("headline-total-structural-irf", "figure"),
        Input("headline-total-structural-store", "data"),
        Input("headline-structural-shock", "value"),
        Input("headline-total-structural-metric", "value"),
        Input("headline-total-structural-fan", "value"),
        Input(store_id, "data"),
        Input("headline-structural-horizon", "value"),
        Input("headline-structural-draws", "value"),
        Input("headline-structural-shock-unit", "value"),
        Input("headline-structural-shock-size", "value"),
        Input("headline-structural-reference-date", "value"),
    )
    def headline_total_render(
        total_store,
        shock,
        metric,
        fan_mode,
        store,
        horizon,
        posterior_draws,
        shock_unit,
        shock_size,
        reference_date,
    ):
        try:
            display_h = int(horizon)
        except (TypeError, ValueError):
            display_h = 12
        display_h = max(1, min(12, display_h))

        try:
            directory = _selected_run_directory(results_root, store)
            current = _headline_total_ui_contract(
                directory,
                horizon=display_h,
                posterior_draws=posterior_draws,
                shock_unit=shock_unit,
                shock_size=shock_size,
                reference_date=reference_date,
            )
        except Exception as exc:
            return (
                html.Div([html.Strong("Headline Total unavailable: "), html.Span(str(exc))],
                         className="estimation-error-text"),
                _headline_total_effective_text(None, display_h), "", [],
                _empty_figure("Headline Total response"),
            )

        if not current["compatible"]:
            reason = " ".join(current["reasons"])
            return (
                html.Div([
                    html.Strong("Parameters not supported by headline-total-structural-irf-v1.1. "),
                    html.Span(reason),
                ], className="estimation-error-text"),
                _headline_total_effective_text(None, display_h), "", [],
                _empty_figure("Headline Total response · unsupported parameters"),
            )

        if not total_store:
            return (
                html.Div([
                    html.Strong("Headline Total not calculated. "),
                    html.Span("Open or refresh Structural analysis with the supported Headline Total parameters."),
                ]),
                _headline_total_effective_text(None, display_h), "", [],
                _empty_figure("Headline Total response"),
            )

        if not total_store.get("ok"):
            status = str(total_store.get("status") or "error")
            detail = (
                " ".join(total_store.get("reasons") or [])
                if status == "unsupported"
                else str(total_store.get("error") or "Unknown Headline Total error.")
            )
            return (
                html.Div([html.Strong(f"Headline Total {status}: "), html.Span(detail)],
                         className="estimation-error-text"),
                _headline_total_effective_text(None, display_h), "", [],
                _empty_figure("Headline Total response"),
            )

        stored_ui = dict(total_store.get("ui") or {})
        compare_keys = (
            "run_directory", "posterior_draws",
            "shock_unit", "shock_size", "reference_date",
        )
        stale = any(stored_ui.get(key) != current.get(key) for key in compare_keys)
        if stale:
            return (
                html.Div([
                    html.Strong("Headline Total result is stale. "),
                    html.Span(
                        "The selected run, draws, shock definition, magnitude, or reference state "
                        "changed after calculation; refresh Structural analysis."
                    ),
                ], className="estimation-error-text"),
                _headline_total_effective_text(total_store, display_h),
                _headline_total_weights_note(total_store), [],
                _empty_figure("Headline Total response · stale"),
            )

        elapsed = float(total_store.get("compute_seconds", 0.0) or 0.0)
        status = html.Div([
            html.Strong("Headline Total response ready"),
            html.Span(
                f" · exact nonlinear re-aggregation · {elapsed:.2f}s"
                if elapsed > 0 else " · exact nonlinear re-aggregation"
            ),
            html.Span(" · no aggregate FEVD/HD"),
        ])
        return (
            status,
            _headline_total_effective_text(total_store, display_h),
            _headline_total_weights_note(total_store),
            _headline_total_records(total_store, shock=shock, metric=metric, horizon=display_h),
            _headline_total_figure(
                total_store, shock=shock, metric=metric,
                fan_mode=fan_mode, horizon=display_h,
            ),
        )


__all__ = [
    "HEADLINE_STRUCTURAL_CONTRACT_VERSION",
    "DEFAULT_STRUCTURAL_DRAWS",
    "MAX_STRUCTURAL_DRAWS",
    "DEFAULT_HORIZON",
    "MAX_HORIZON",
    "DEFAULT_SHOCK_UNIT",
    "DEFAULT_SHOCK_SIZE",
    "STATE_LABELS",
    "STATE_COMPONENTS",
    "HeadlineStructuralDashboardError",
    "resolve_headline_run_directory",
    "headline_structural_run_contract",
    "load_headline_structural_result",
    "compute_headline_structural",
    "reference_regime_date",
    "reference_volatility_snapshot",
    "volatility_sparkline_figure",
    "relative_volatility_state_figure",
    "irf_figure",
    "fevd_figure",
    "historical_decomposition_figure",
    "headline_structural_page",
    "register_headline_structural_callbacks",
]
