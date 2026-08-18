"""Recursive structural-analysis backend for the Energy BVAR dashboard.

V1 deliberately implements one identification scheme only:
    recursive / Cholesky

It consumes persisted Gibbs draws and the exact model input contract.  It never
re-estimates the BVAR.

The module returns compact posterior summaries for Dash:
- IRFs: q05/q16/q50/q84/q95 for monthly/weekly changes and cumulative levels;
- FEVD: posterior quantiles of horizon-specific variance shares;
- historical decomposition: posterior means, because the mean preserves the
  exact additive reconstruction identity while component-wise medians do not.

Sign restrictions and conditional commodity-path scenarios are intentionally
out of scope for this V1.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from energy_bvar_model import (
    forecast_error_variance_decomposition,
    historical_decomposition,
    impulse_responses,
    prepare_bvar_panel,
)
try:
    from energy_bvar_model import hash_model_data as _hash_model_data
except ImportError:  # legacy engine
    _hash_model_data = None

from energy_bvar_pipeline import build_panel, model_contract, model_spec


STRUCTURAL_CONTRACT_VERSION = "energy-structural-recursive-v1"
STRUCTURAL_INVARIANT_CACHE_VERSION = "energy-structural-invariant-v1"
STRUCTURAL_LAZY_HD_CONTRACT_VERSION = "energy-structural-lazy-hd-v1"
DEFAULT_STRUCTURAL_DRAWS = 300
DEFAULT_HORIZON = 24
MAX_STRUCTURAL_DRAWS = 1000
MAX_HORIZON = 104
DEFAULT_SHOCK_UNIT = "structural_std"
DEFAULT_SHOCK_SIZE = 1.0


# Stable structural palette.  A colour always denotes the same shock order
# within the selected BVAR.  The reference-state diagnostics use the same hues.
_STRUCTURAL_COLOURS = (
    "#2563EB",
    "#10B981",
    "#F59E0B",
    "#EF4444",
    "#8B5CF6",
    "#0EA5E9",
)


class StructuralDashboardError(RuntimeError):
    """Raised when a saved run cannot satisfy the structural contract."""


# GRAPH_EXPORT_READABILITY_G2_ENERGY_STRUCTURAL_V1
def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StructuralDashboardError(f"Missing metadata: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StructuralDashboardError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StructuralDashboardError(f"{path} does not contain a JSON object.")
    return payload


def resolve_run_directory(
    results_root: str | Path,
    *,
    model_id: str,
    vintage: str,
    run_id: str,
) -> Path:
    directory = Path(results_root) / str(model_id) / str(vintage) / str(run_id)
    if not directory.is_dir():
        raise StructuralDashboardError(f"Saved run directory not found: {directory}")
    if not (directory / "metadata.json").is_file():
        raise StructuralDashboardError(f"metadata.json missing in {directory}")
    if not (directory / "draws.npz").is_file():
        raise StructuralDashboardError(
            "Structural analysis requires persisted posterior draws. "
            f"draws.npz is missing in {directory}."
        )
    return directory


def _draw_subset_indices(n_draws: int, requested: int) -> np.ndarray:
    n = int(n_draws)
    m = min(n, max(1, int(requested)))
    if n < 1:
        raise StructuralDashboardError("The saved posterior contains no draws.")
    if m == n:
        return np.arange(n, dtype=int)
    # Same reproducible spread-across-chain convention as the notebooks.
    return np.unique(np.linspace(0, n - 1, m, dtype=int))


def _prepare_saved_run(panel, metadata: Mapping[str, Any]) -> tuple[dict, str]:
    """Recreate the exact estimation design represented by a persisted run."""
    variables = list(metadata.get("variables") or panel.variables)
    missing = [name for name in variables if name not in panel.levels.columns]
    if missing:
        raise StructuralDashboardError(
            f"{metadata.get('model_id')}: current processed panel is missing {missing}."
        )

    levels = panel.levels.loc[:, variables].copy().astype(float)
    exog_names = list(metadata.get("exog_names") or [])
    exog = panel.exog
    if exog_names:
        if exog is None:
            raise StructuralDashboardError(
                f"{metadata.get('model_id')}: saved run expects exogenous columns {exog_names}."
            )
        missing_exog = [name for name in exog_names if name not in exog.columns]
        if missing_exog:
            raise StructuralDashboardError(
                f"{metadata.get('model_id')}: current deterministic block is missing "
                f"{missing_exog}."
            )
        exog = exog.loc[:, exog_names]
    elif exog is not None and exog.shape[1]:
        # Never let a later spec change inject deterministic terms into an old run.
        exog = None

    expected_hash = metadata.get("source_data_hash") or metadata.get("data_hash")
    if expected_hash and _hash_model_data is not None:
        actual_hash = _hash_model_data(levels, exog)
        if str(actual_hash) != str(expected_hash):
            raise StructuralDashboardError(
                f"{metadata.get('model_id')}: processed data no longer match the "
                "source-data hash of the saved posterior run."
            )

    missing_method = str(metadata.get("missing_data_method") or "dk").lower()
    params = inspect.signature(prepare_bvar_panel).parameters
    kwargs: dict[str, Any] = {
        "p": int(metadata.get("p") or panel.p),
        "variables": variables,
    }
    if "frequency" in params:
        kwargs["frequency"] = str(metadata.get("frequency") or panel.frequency)
    if "exog" in params:
        kwargs["exog"] = exog

    effective_levels = levels
    if "missing_data_method" in params:
        kwargs["missing_data_method"] = missing_method
    elif missing_method == "linear":
        effective_levels = levels.interpolate(method="time", limit_area="inside")
    elif missing_method not in {"dk", "durbin_koopman", "durbin-koopman", ""}:
        raise StructuralDashboardError(
            f"Unsupported saved missing_data_method={missing_method!r}."
        )

    prep = prepare_bvar_panel(effective_levels, **kwargs)

    expected_effective = metadata.get("effective_estimation_data_hash")
    if expected_effective and missing_method == "linear" and _hash_model_data is not None:
        actual_effective = _hash_model_data(
            prep["levels"][variables],
            prep.get("exog"),
        )
        if str(actual_effective) != str(expected_effective):
            raise StructuralDashboardError(
                f"{metadata.get('model_id')}: recreated linear estimation panel "
                "does not match the saved effective-data hash."
            )

    return prep, missing_method


def load_structural_result(
    run_directory: str | Path,
    *,
    project_root: str | Path,
    posterior_draws: int = DEFAULT_STRUCTURAL_DRAWS,
) -> tuple[dict, dict[str, Any]]:
    """Load and reconstruct a structural-ready result from a saved run."""
    directory = Path(run_directory)
    metadata = _read_json(directory / "metadata.json")
    model_id = str(metadata.get("model_id") or "")
    vintage = str(metadata.get("vintage") or "")
    run_id = str(metadata.get("run_id") or directory.name)
    if not model_id or not vintage:
        raise StructuralDashboardError(
            f"Saved run metadata in {directory} lacks model_id/vintage."
        )

    panel = build_panel(model_id, vintage, project_root=project_root)
    prep, missing_method = _prepare_saved_run(panel, metadata)

    with np.load(directory / "draws.npz", allow_pickle=False) as archive:
        draws = {name: archive[name] for name in archive.files}

    required = {"B", "A", "log_variance", "outlier_scales"}
    missing_arrays = sorted(required.difference(draws))
    if missing_arrays:
        raise StructuralDashboardError(
            f"{model_id}: draws.npz is missing structural arrays {missing_arrays}."
        )

    B = np.asarray(draws["B"])
    if B.ndim != 3 or len(B) < 1:
        raise StructuralDashboardError(
            f"{model_id}: unexpected coefficient draw shape {B.shape}."
        )
    indices = _draw_subset_indices(len(B), min(int(posterior_draws), MAX_STRUCTURAL_DRAWS))

    selected: dict[str, Any] = {}
    for name, value in draws.items():
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == len(B):
            selected[name] = array[indices]
        else:
            selected[name] = array

    variables = list(metadata.get("variables") or panel.variables)
    result = {
        **selected,
        "metadata": metadata,
        "variables": variables,
        "p": int(metadata.get("p") or panel.p),
        "n_draws": int(len(indices)),
        "prep": prep,
        "frequency": str(metadata.get("frequency") or panel.frequency),
    }

    contract = model_contract(model_id, vintage, project_root=project_root)
    info = {
        "contract_version": STRUCTURAL_CONTRACT_VERSION,
        "model_id": model_id,
        "model_label": str(getattr(panel.spec, "label", model_id)),
        "vintage": vintage,
        "run_id": run_id,
        "frequency": str(panel.frequency),
        "p": int(result["p"]),
        "variables": variables,
        "target": str(panel.target),
        "units": dict(contract.get("units", panel.units)),
        "exog_names": list(metadata.get("exog_names") or []),
        "missing_data_method": missing_method,
        "available_posterior_draws": int(len(B)),
        "selected_posterior_draws": int(len(indices)),
        "selected_draw_indices": indices.astype(int).tolist(),
        "reference_dates": [
            pd.Timestamp(date).isoformat()
            for date in pd.DatetimeIndex(prep["dates"])
        ],
        "default_reference_date": pd.Timestamp(prep["dates"][-1]).isoformat(),
        "recursive_ordering": variables,
        "requires_data_augmentation": bool(
            prep.get("requires_data_augmentation", False)
        ),
    }
    return result, info


def structural_run_contract(
    run_directory: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Cheap-enough run contract used to populate Structural controls."""
    _, info = load_structural_result(
        run_directory,
        project_root=project_root,
        posterior_draws=1,
    )
    return info


def _quantile_summary(array: np.ndarray, axis: int = 0) -> dict[str, np.ndarray]:
    q = np.quantile(np.asarray(array, dtype=float), [0.05, 0.16, 0.50, 0.84, 0.95], axis=axis)
    return {
        "q05": q[0],
        "q16": q[1],
        "q50": q[2],
        "q84": q[3],
        "q95": q[4],
    }



def _volatility_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Posterior volatility/outlier history for reference-date selection.

    ``persistent_*`` summarises sqrt(lambda_t), i.e. the regular structural
    standard deviation. ``total_*`` summarises o_t * sqrt(lambda_t), so realised
    outlier amplification appears as transient spikes.

    The outlier probability is the posterior mean conditional outlier
    probability when that array is persisted; otherwise it falls back to the
    posterior frequency of the sampled outlier indicator.
    """
    variables = list(result["variables"])
    dates = pd.DatetimeIndex(result["prep"]["dates"])

    log_h = np.asarray(result["log_variance"], dtype=float)
    if log_h.ndim != 3:
        raise StructuralDashboardError(
            f"Unexpected log_variance shape {log_h.shape}."
        )
    persistent = np.exp(
        0.5 * np.clip(log_h[:, 1:, :], -745.0, 700.0)
    )
    scales = np.asarray(result["outlier_scales"], dtype=float)
    if scales.shape != persistent.shape:
        raise StructuralDashboardError(
            "outlier_scales and stochastic-volatility paths have incompatible "
            f"shapes: {scales.shape} vs {persistent.shape}."
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

    persistent_q = np.quantile(
        persistent, [0.16, 0.50, 0.84], axis=0
    )
    total_q = np.quantile(
        total, [0.16, 0.50, 0.84], axis=0
    )

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
                    "outlier_probability": float(
                        outlier_probability[t, j]
                    ),
                }
            )
    return rows


def _shock_scale_rows(irf: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Summarise the draw-specific multiplier used for each structural shock."""
    factors = np.asarray(irf.get("scale_factors"), dtype=float)
    shocks = list(irf.get("shock_names") or [])
    if factors.ndim != 2 or factors.shape[1] != len(shocks):
        return []
    q = np.quantile(factors, [0.05, 0.16, 0.50, 0.84, 0.95], axis=0)
    rows = []
    for j, shock in enumerate(shocks):
        rows.append(
            {
                "shock": shock,
                "q05": float(q[0, j]),
                "q16": float(q[1, j]),
                "q50": float(q[2, j]),
                "q84": float(q[3, j]),
                "q95": float(q[4, j]),
            }
        )
    return rows



def _reference_state_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Dimensionless joint SV state used only for reference-date selection.

    The structural model has one regular stochastic variance ``lambda_j,t`` per
    structural shock.  Raw variances are not comparable across variables because
    the variables have different units.  For display and preset selection we
    therefore normalise each structural standard deviation by *its own* sample
    median, square the ratio back to variance units, and form a geometric mean
    across shocks.  This diagnostic is scale-free; the FEVD itself is still
    computed from the full impact dynamics, not from these bars.
    """
    if not payload or not payload.get("ok"):
        return pd.DataFrame()
    frame = pd.DataFrame(payload.get("volatility", []))
    needed = {"date", "variable", "persistent_q50"}
    if frame.empty or not needed.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["persistent_q50"] = pd.to_numeric(
        frame["persistent_q50"], errors="coerce"
    )
    frame = frame.dropna(subset=["date", "persistent_q50"])
    frame = frame.loc[frame["persistent_q50"] > 0.0]
    if frame.empty:
        return pd.DataFrame()

    wide = frame.pivot_table(
        index="date", columns="variable", values="persistent_q50", aggfunc="last"
    ).sort_index()
    ordered = [name for name in payload.get("variables", []) if name in wide.columns]
    if not ordered:
        return pd.DataFrame()
    wide = wide.loc[:, ordered]
    med = wide.median(axis=0, skipna=True).replace(0.0, np.nan)
    relative_sd = wide.divide(med, axis=1)
    relative_variance = relative_sd.pow(2.0)
    positive = relative_variance.where(relative_variance > 0.0)
    joint = np.exp(np.log(positive).mean(axis=1, skipna=True))

    out = relative_variance.copy()
    out["joint_stress"] = joint
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


def reference_regime_date(
    payload: Mapping[str, Any] | None,
    regime: str = "latest",
) -> str | None:
    """Resolve a named reference-volatility regime to an observed model date."""
    state = _reference_state_frame(payload or {})
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
        distance = np.abs(np.log(joint.to_numpy(dtype=float)) - np.log(target))
        chosen = joint.index[int(np.nanargmin(distance))]
    elif key in {"peak_2022", "peak2022"}:
        block = joint.loc[joint.index.year == 2022]
        chosen = block.idxmax() if not block.empty else joint.idxmax()
    elif key in {"peak", "sample_peak"}:
        chosen = joint.idxmax()
    else:
        raise StructuralDashboardError(
            f"Unknown reference-volatility regime {regime!r}."
        )
    return pd.Timestamp(chosen).isoformat()


def reference_volatility_snapshot(
    payload: Mapping[str, Any] | None,
    reference_date=None,
) -> dict[str, Any]:
    """Summarise the *joint* structural-SV vector at one reference date."""
    if not payload or not payload.get("ok"):
        return {"ok": False, "cards": []}
    frame = pd.DataFrame(payload.get("volatility", []))
    state = _reference_state_frame(payload)
    if frame.empty or state.empty:
        return {"ok": False, "cards": []}

    frame = frame.copy()
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
        # Calendar controls should make this rare; nearest keeps the helper safe.
        pos = int(np.argmin(np.abs((available - ref).asi8)))
        ref = pd.Timestamp(available[pos])

    cards: list[dict[str, Any]] = []
    for variable in payload.get("variables", []):
        history = frame.loc[frame["variable"].astype(str).eq(str(variable))].copy()
        history = history.sort_values("date")
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
                "label": str(variable).replace("_", " ").title(),
                "unit": str((payload.get("units") or {}).get(variable, "")),
                "persistent_q16": float(row.get("persistent_q16", np.nan)),
                "persistent_q50": float(row.get("persistent_q50", np.nan)),
                "persistent_q84": float(row.get("persistent_q84", np.nan)),
                "relative_sd": rel_sd,
                "relative_variance": rel_var,
                "percentile": percentile,
                "outlier_probability": float(row.get("outlier_probability", np.nan)),
            }
        )

    joint = pd.to_numeric(state["joint_stress"], errors="coerce").dropna()
    joint_value = float(state.loc[ref, "joint_stress"])
    joint_percentile = 100.0 * float((joint <= joint_value).mean()) if len(joint) else np.nan
    return {
        "ok": True,
        "reference_date": ref.isoformat(),
        "joint_stress": joint_value,
        "joint_percentile": joint_percentile,
        "cards": cards,
    }


def volatility_sparkline_figure(
    payload: Mapping[str, Any] | None,
    *,
    variable: str,
    reference_date=None,
) -> go.Figure:
    """Compact, scale-free SV history for one variable card."""
    state = _reference_state_frame(payload or {})
    if state.empty or variable not in state.columns:
        return _empty_figure("No SV history", height=105)
    series = pd.to_numeric(state[variable], errors="coerce")
    rel_sd = np.sqrt(series.clip(lower=0.0))
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
            hovertemplate="%{x|%Y-%m-%d}<br>√λ / own median=%{y:.2f}×<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(y=1.0, line_width=1, line_color="#E5E7EB")
    fig.add_shape(
        type="line",
        x0=ref.isoformat(), x1=ref.isoformat(), y0=0, y1=1,
        xref="x", yref="paper",
        line={"width": 1.2, "dash": "dot", "color": "#F97316"},
    )
    value = float(rel_sd.loc[ref]) if pd.notna(rel_sd.loc[ref]) else np.nan
    if np.isfinite(value):
        fig.add_trace(
            go.Scatter(
                x=[ref], y=[value], mode="markers",
                marker={"size": 7, "color": "#F97316", "line": {"width": 1, "color": "white"}},
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
        uirevision=f"{(payload or {}).get('run_id')}::sv-card::{variable}",
    )
    return fig


def relative_volatility_state_figure(
    payload: Mapping[str, Any] | None,
    *,
    reference_date=None,
) -> go.Figure:
    """Dimensionless structural-variance multipliers at the selected date."""
    snap = reference_volatility_snapshot(payload, reference_date)
    if not snap.get("ok") or not snap.get("cards"):
        return _empty_figure("No joint SV state is available.", height=250)
    cards = list(snap["cards"])
    labels = [card["label"] for card in cards]
    values = [float(card["relative_variance"]) for card in cards]
    colours = [_STRUCTURAL_COLOURS[i % len(_STRUCTURAL_COLOURS)] for i in range(len(cards))]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colours},
            text=[f"{value:.2f}×" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>λ / own-sample median=%{x:.2f}×<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=1.0, line_width=1.2, line_dash="dash", line_color="#94A3B8")
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
        uirevision=f"{(payload or {}).get('run_id')}::sv-relative::{snap.get('reference_date')}",
    )
    return fig


def _visible_x_window(
    relayout_data: Mapping[str, Any] | None,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Extract a Plotly x-axis window from relayoutData."""
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
    return min(a, b), max(a, b)


def _adaptive_line_yaxis(
    fig: go.Figure,
    relayout_data: Mapping[str, Any] | None,
    *,
    log_axis: bool = False,
    padding_fraction: float = 0.08,
) -> go.Figure:
    """Fit the y-axis to visible line/scatter values after an x zoom/pan."""
    window = _visible_x_window(relayout_data)
    values: list[float] = []

    for trace in fig.data:
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
        vals = y.loc[valid].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if log_axis:
            vals = vals[vals > 0]
        values.extend(vals.tolist())

    if not values:
        fig.update_yaxes(autorange=True)
    else:
        lo = float(np.min(values))
        hi = float(np.max(values))
        if log_axis:
            lo_log = np.log10(max(lo, np.finfo(float).tiny))
            hi_log = np.log10(max(hi, np.finfo(float).tiny))
            span = max(hi_log - lo_log, 0.25)
            pad = padding_fraction * span
            fig.update_yaxes(
                type="log",
                range=[lo_log - pad, hi_log + pad],
                autorange=False,
            )
        else:
            span = hi - lo
            pad = max(
                1e-12,
                padding_fraction
                * (span if span > 0 else max(abs(lo), abs(hi), 1.0)),
            )
            fig.update_yaxes(
                range=[lo - pad, hi + pad],
                autorange=False,
            )

    if window is None:
        if bool(dict(relayout_data or {}).get("xaxis.autorange")):
            fig.update_xaxes(autorange=True)
    else:
        fig.update_xaxes(
            range=[window[0], window[1]],
            autorange=False,
        )
    return fig


def _adaptive_hd_yaxis(
    fig: go.Figure,
    relayout_data: Mapping[str, Any] | None,
    *,
    padding_fraction: float = 0.08,
) -> go.Figure:
    """Fit HD y-axis to visible stacked-bar totals plus line extrema.

    A generic trace-wise autoscale is insufficient for relative stacked bars:
    several positive contributions can stack above every individual trace.
    """
    window = _visible_x_window(relayout_data)
    stacked: dict[pd.Timestamp, list[float]] = {}
    line_values: list[float] = []

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

        trace_type = getattr(trace, "type", "")
        for stamp, value in zip(
            x.loc[valid],
            y.loc[valid].astype(float),
        ):
            if not np.isfinite(value):
                continue
            if trace_type == "bar":
                key = pd.Timestamp(stamp)
                if key not in stacked:
                    stacked[key] = [0.0, 0.0]
                if value >= 0:
                    stacked[key][0] += float(value)
                else:
                    stacked[key][1] += float(value)
            else:
                line_values.append(float(value))

    values = list(line_values)
    for positive, negative in stacked.values():
        values.extend([positive, negative])
    values.append(0.0)

    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size:
        lo = float(finite.min())
        hi = float(finite.max())
        span = hi - lo
        pad = max(
            1e-12,
            padding_fraction
            * (span if span > 0 else max(abs(lo), abs(hi), 1.0)),
        )
        fig.update_yaxes(
            range=[lo - pad, hi + pad],
            autorange=False,
        )
    else:
        fig.update_yaxes(autorange=True)

    if window is None:
        if bool(dict(relayout_data or {}).get("xaxis.autorange")):
            fig.update_xaxes(autorange=True)
    else:
        fig.update_xaxes(
            range=[window[0], window[1]],
            autorange=False,
        )
    return fig


def _irf_rows(irf: Mapping[str, Any]) -> list[dict[str, Any]]:
    variables = list(irf["variables"])
    shocks = list(irf["shock_names"])
    horizons = np.asarray(irf["horizons"], dtype=int)
    rows: list[dict[str, Any]] = []
    for metric, key in (
        ("change", "change_irfs"),
        ("cumulative", "cumulative_level_irfs"),
    ):
        summary = _quantile_summary(np.asarray(irf[key], dtype=float), axis=0)
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
    summary = _quantile_summary(np.asarray(fevd["fevd_draws"], dtype=float), axis=0)
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


def _hd_rows(
    hd: Mapping[str, Any],
    *,
    requires_data_augmentation: bool,
) -> tuple[list[dict[str, Any]], float, str]:
    """Posterior-mean HD rows with an additive display contract.

    For balanced/legacy-linear runs the historical ``Y`` is common to every
    posterior draw, so the observed line is the actual model-space change.

    Under exact Durbin--Koopman augmentation, completed missing observations are
    draw-specific. ``historical_decomposition`` therefore reconstructs a
    different completed ``Y`` for each posterior draw.  Its legacy ``observed``
    field is only the final loop draw and must *not* be compared with posterior
    mean contributions.  In that case the dashboard displays the posterior mean
    completed change, which is exactly equal to posterior-mean base plus
    posterior-mean structural contributions.
    """
    variables = list(hd["variables"])
    components = list(hd["component_names"])
    dates = pd.DatetimeIndex(hd["dates"])

    contribution_draws = np.asarray(hd["contributions"], dtype=float)
    base_draws = np.asarray(hd["base"], dtype=float)
    reconstructed_draws = np.asarray(hd["reconstructed"], dtype=float)

    contributions = contribution_draws.mean(axis=0)
    base = base_draws.mean(axis=0)
    reconstructed_mean = reconstructed_draws.mean(axis=0)

    if requires_data_augmentation:
        display_observed = reconstructed_mean
        observed_label = "Posterior-mean completed change"
    else:
        display_observed = np.asarray(hd["observed"], dtype=float)
        observed_label = "Observed change"

    additive_reconstruction = base + contributions.sum(axis=2)
    mean_error = float(
        np.max(np.abs(additive_reconstruction - display_observed))
    )

    rows: list[dict[str, Any]] = []
    for t, date in enumerate(dates):
        for i, response in enumerate(variables):
            row = {
                "date": pd.Timestamp(date).isoformat(),
                "response": response,
                "observed": float(display_observed[t, i]),
                "base": float(base[t, i]),
                "reconstructed": float(additive_reconstruction[t, i]),
            }
            for j, component in enumerate(components):
                row[component] = float(contributions[t, i, j])
            rows.append(row)

    return rows, mean_error, observed_label


def structural_invariant_cache_from_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the reference-invariant Structural block from a valid payload."""
    if not isinstance(payload, Mapping) or not bool(payload.get("ok")):
        raise StructuralDashboardError(
            "Cannot build a Structural invariant cache from an unsuccessful payload."
        )

    required = (
        "model_id",
        "vintage",
        "run_id",
        "posterior_draws",
        "selected_draw_indices",
        "split_outlier_amplification",
        "requires_data_augmentation",
        "hd_observed_label",
        "volatility",
        "hd",
        "hd_component_names",
        "diagnostics",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise StructuralDashboardError(
            f"Structural payload is missing invariant-cache fields {missing}."
        )

    diagnostics = dict(payload.get("diagnostics") or {})
    return {
        "cache_version": STRUCTURAL_INVARIANT_CACHE_VERSION,
        "source_contract_version": str(payload.get("contract_version") or ""),
        "model_id": str(payload["model_id"]),
        "vintage": str(payload["vintage"]),
        "run_id": str(payload["run_id"]),
        "posterior_draws": int(payload["posterior_draws"]),
        "selected_draw_indices": [
            int(value) for value in payload.get("selected_draw_indices", [])
        ],
        "split_outlier_amplification": bool(
            payload["split_outlier_amplification"]
        ),
        "requires_data_augmentation": bool(
            payload["requires_data_augmentation"]
        ),
        "hd_observed_label": str(payload["hd_observed_label"]),
        "volatility": list(payload.get("volatility") or []),
        "hd": list(payload.get("hd") or []),
        "hd_component_names": list(payload.get("hd_component_names") or []),
        "diagnostics": {
            "hd_max_reconstruction_error_drawwise": float(
                diagnostics["hd_max_reconstruction_error_drawwise"]
            ),
            "hd_posterior_mean_reconstruction_error": float(
                diagnostics["hd_posterior_mean_reconstruction_error"]
            ),
        },
    }


def _validated_structural_invariant_cache(
    cache: Mapping[str, Any] | None,
    *,
    info: Mapping[str, Any],
    split_outlier_amplification: bool,
) -> dict[str, Any] | None:
    """Validate that the invariant cache belongs to the exact saved run/subset."""
    if cache is None:
        return None
    if not isinstance(cache, Mapping):
        raise StructuralDashboardError("Structural invariant cache is not a mapping.")
    if str(cache.get("cache_version") or "") != STRUCTURAL_INVARIANT_CACHE_VERSION:
        raise StructuralDashboardError(
            "Structural invariant cache has an incompatible version."
        )
    if str(cache.get("source_contract_version") or "") != STRUCTURAL_CONTRACT_VERSION:
        raise StructuralDashboardError(
            "Structural invariant cache was produced by another Structural contract."
        )

    for key in ("model_id", "vintage", "run_id"):
        actual = str(cache.get(key) or "")
        expected = str(info.get(key) or "")
        if actual != expected:
            raise StructuralDashboardError(
                f"Structural invariant cache {key} mismatch: {actual!r} != {expected!r}."
            )

    if int(cache.get("posterior_draws", -1)) != int(
        info["selected_posterior_draws"]
    ):
        raise StructuralDashboardError(
            "Structural invariant cache posterior-draw count mismatch."
        )

    cached_indices = [
        int(value) for value in cache.get("selected_draw_indices", [])
    ]
    expected_indices = [
        int(value) for value in info.get("selected_draw_indices", [])
    ]
    if cached_indices != expected_indices:
        raise StructuralDashboardError(
            "Structural invariant cache draw indices mismatch."
        )

    if bool(cache.get("split_outlier_amplification")) != bool(
        split_outlier_amplification
    ):
        raise StructuralDashboardError(
            "Structural invariant cache uses another outlier-splitting convention."
        )

    if bool(cache.get("requires_data_augmentation")) != bool(
        info.get("requires_data_augmentation", False)
    ):
        raise StructuralDashboardError(
            "Structural invariant cache data-augmentation contract mismatch."
        )

    diagnostics = dict(cache.get("diagnostics") or {})
    for key in (
        "hd_max_reconstruction_error_drawwise",
        "hd_posterior_mean_reconstruction_error",
    ):
        if key not in diagnostics or not np.isfinite(float(diagnostics[key])):
            raise StructuralDashboardError(
                f"Structural invariant cache lacks finite diagnostic {key!r}."
            )

    for key in ("volatility", "hd", "hd_component_names", "hd_observed_label"):
        if key not in cache:
            raise StructuralDashboardError(
                f"Structural invariant cache lacks {key!r}."
            )

    return {
        **dict(cache),
        "volatility": list(cache.get("volatility") or []),
        "hd": list(cache.get("hd") or []),
        "hd_component_names": list(cache.get("hd_component_names") or []),
        "diagnostics": diagnostics,
    }


def _structural_identity_payload(
    info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "ok": True,
        "contract_version": STRUCTURAL_CONTRACT_VERSION,
        "model_id": info["model_id"],
        "model_label": info["model_label"],
        "vintage": info["vintage"],
        "run_id": info["run_id"],
        "frequency": info["frequency"],
        "p": info["p"],
        "variables": info["variables"],
        "target": info["target"],
        "units": info["units"],
        "recursive_ordering": info["recursive_ordering"],
        "posterior_draws": int(info["selected_posterior_draws"]),
        "available_posterior_draws": int(info["available_posterior_draws"]),
        "selected_draw_indices": info["selected_draw_indices"],
        "missing_data_method": info["missing_data_method"],
        "exog_names": info["exog_names"],
        "requires_data_augmentation": bool(
            info.get("requires_data_augmentation", False)
        ),
    }


def compute_structural_volatility_v1(
    run_directory: str | Path,
    *,
    project_root: str | Path,
    posterior_draws: int = DEFAULT_STRUCTURAL_DRAWS,
) -> dict[str, Any]:
    """Load the saved posterior and materialise only SV/outlier history.

    This is deliberately independent of IRF, FEVD and HD so the reference-state
    cards can become usable before the rest of Structural is ready.
    """
    result, info = load_structural_result(
        run_directory,
        project_root=project_root,
        posterior_draws=posterior_draws,
    )
    payload = {
        **_structural_identity_payload(info),
        "payload_kind": "volatility",
        "reference_date": info["default_reference_date"],
        "volatility": _volatility_rows(result),
        "reference_dependence": {
            "volatility_history": False,
            "irf_structural_std": True,
            "irf_level": False,
            "fevd": True,
            "hd_recursive": False,
        },
    }
    return payload


def compute_structural_irf_fevd_v1(
    run_directory: str | Path,
    *,
    project_root: str | Path,
    reference_date=None,
    horizon: int = DEFAULT_HORIZON,
    posterior_draws: int = DEFAULT_STRUCTURAL_DRAWS,
    shock_unit: str = DEFAULT_SHOCK_UNIT,
    shock_size: float = DEFAULT_SHOCK_SIZE,
) -> dict[str, Any]:
    """Compute only the reference-dependent IRF/FEVD block."""
    horizon = int(horizon)
    if horizon < 1 or horizon > MAX_HORIZON:
        raise StructuralDashboardError(
            f"horizon must lie in [1, {MAX_HORIZON}], got {horizon}."
        )
    shock_unit = str(shock_unit).strip().lower()
    if shock_unit not in {"structural_std", "level"}:
        raise StructuralDashboardError(
            "shock_unit must be 'structural_std' or 'level'."
        )
    shock_size = float(shock_size)
    if not np.isfinite(shock_size) or shock_size <= 0:
        raise StructuralDashboardError(
            f"shock_size must be a finite positive number, got {shock_size}."
        )

    result, info = load_structural_result(
        run_directory,
        project_root=project_root,
        posterior_draws=posterior_draws,
    )
    reference = (
        info["default_reference_date"]
        if reference_date is None
        else pd.Timestamp(reference_date).isoformat()
    )
    irf = impulse_responses(
        result,
        identification="recursive",
        reference_date=reference,
        horizon=horizon,
        shock_unit=shock_unit,
        shock_size=shock_size,
        include_outlier_scale=False,
    )
    fevd = forecast_error_variance_decomposition(
        result,
        identification="recursive",
        reference_date=reference,
        horizon=horizon,
        include_outlier_scale=False,
    )
    return {
        **_structural_identity_payload(info),
        "payload_kind": "irf_fevd",
        "identification": "recursive",
        "reference_date": pd.Timestamp(irf["reference_date"]).isoformat(),
        "horizon": horizon,
        "shock_unit": shock_unit,
        "shock_size": shock_size,
        "shock_scale_factors": _shock_scale_rows(irf),
        "irf": _irf_rows(irf),
        "fevd": _fevd_rows(fevd),
        "reference_dependence": {
            "irf_structural_std": True,
            "irf_level": False,
            "fevd": True,
            "hd_recursive": False,
            "outlier_scale_in_irf_fevd": False,
        },
        "diagnostics": {
            "fevd_max_share_sum_error": float(fevd["max_share_sum_error"]),
            "recursive_acceptance_rate": 1.0,
        },
    }


def compute_structural_hd_v1(
    run_directory: str | Path,
    *,
    project_root: str | Path,
    posterior_draws: int = DEFAULT_STRUCTURAL_DRAWS,
    split_outlier_amplification: bool = True,
) -> dict[str, Any]:
    """Compute only recursive historical decomposition.

    The cache identity intentionally excludes reference date, IRF horizon and
    shock scaling because recursive HD is invariant to all three.
    """
    result, info = load_structural_result(
        run_directory,
        project_root=project_root,
        posterior_draws=posterior_draws,
    )
    reference = info["default_reference_date"]
    hd = historical_decomposition(
        result,
        identification="recursive",
        reference_date=reference,
        split_outlier_amplification=bool(split_outlier_amplification),
    )
    hd_rows, mean_hd_error, hd_observed_label = _hd_rows(
        hd,
        requires_data_augmentation=bool(
            info.get("requires_data_augmentation", False)
        ),
    )
    return {
        **_structural_identity_payload(info),
        "payload_kind": "historical_decomposition",
        "lazy_hd_contract_version": STRUCTURAL_LAZY_HD_CONTRACT_VERSION,
        "identification": "recursive",
        "split_outlier_amplification": bool(split_outlier_amplification),
        "hd_observed_label": hd_observed_label,
        "hd": hd_rows,
        "hd_component_names": list(hd["component_names"]),
        "diagnostics": {
            "hd_max_reconstruction_error_drawwise": float(
                hd["max_reconstruction_error"]
            ),
            "hd_posterior_mean_reconstruction_error": float(mean_hd_error),
        },
        "reference_dependence": {
            "hd_recursive": False,
        },
    }


def compute_structural_v1(
    run_directory: str | Path,
    *,
    project_root: str | Path,
    reference_date=None,
    horizon: int = DEFAULT_HORIZON,
    posterior_draws: int = DEFAULT_STRUCTURAL_DRAWS,
    shock_unit: str = DEFAULT_SHOCK_UNIT,
    shock_size: float = DEFAULT_SHOCK_SIZE,
    split_outlier_amplification: bool = True,
    invariant_cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute recursive IRF, FEVD and HD from persisted Gibbs draws."""
    horizon = int(horizon)
    if horizon < 1 or horizon > MAX_HORIZON:
        raise StructuralDashboardError(
            f"horizon must lie in [1, {MAX_HORIZON}], got {horizon}."
        )
    shock_unit = str(shock_unit).strip().lower()
    if shock_unit not in {"structural_std", "level"}:
        raise StructuralDashboardError(
            "shock_unit must be 'structural_std' or 'level'."
        )
    shock_size = float(shock_size)
    if not np.isfinite(shock_size) or shock_size <= 0:
        raise StructuralDashboardError(
            f"shock_size must be a finite positive number, got {shock_size}."
        )

    result, info = load_structural_result(
        run_directory,
        project_root=project_root,
        posterior_draws=posterior_draws,
    )
    reference = (
        info["default_reference_date"]
        if reference_date is None
        else pd.Timestamp(reference_date).isoformat()
    )

    # Regular one-standard-deviation shocks.  Outlier multipliers are excluded
    # by construction for IRFs/FEVD; HD separately reconstructs their realised
    # historical amplification when requested.
    irf = impulse_responses(
        result,
        identification="recursive",
        reference_date=reference,
        horizon=horizon,
        shock_unit=shock_unit,
        shock_size=shock_size,
        include_outlier_scale=False,
    )
    fevd = forecast_error_variance_decomposition(
        result,
        identification="recursive",
        reference_date=reference,
        horizon=horizon,
        include_outlier_scale=False,
    )
    requires_augmentation = bool(
        info.get("requires_data_augmentation", False)
    )
    cached_invariant = _validated_structural_invariant_cache(
        invariant_cache,
        info=info,
        split_outlier_amplification=bool(split_outlier_amplification),
    )

    if cached_invariant is None:
        hd = historical_decomposition(
            result,
            identification="recursive",
            reference_date=reference,
            split_outlier_amplification=bool(split_outlier_amplification),
        )
        hd_rows, mean_hd_error, hd_observed_label = _hd_rows(
            hd,
            requires_data_augmentation=requires_augmentation,
        )
        volatility_rows = _volatility_rows(result)
        hd_component_names = list(hd["component_names"])
        hd_max_reconstruction_error = float(hd["max_reconstruction_error"])
        invariant_reused = False
    else:
        hd_rows = list(cached_invariant["hd"])
        hd_observed_label = str(cached_invariant["hd_observed_label"])
        volatility_rows = list(cached_invariant["volatility"])
        hd_component_names = list(cached_invariant["hd_component_names"])
        invariant_diagnostics = dict(cached_invariant["diagnostics"])
        hd_max_reconstruction_error = float(
            invariant_diagnostics["hd_max_reconstruction_error_drawwise"]
        )
        mean_hd_error = float(
            invariant_diagnostics["hd_posterior_mean_reconstruction_error"]
        )
        invariant_reused = True

    payload = {
        "ok": True,
        "contract_version": STRUCTURAL_CONTRACT_VERSION,
        "identification": "recursive",
        "model_id": info["model_id"],
        "model_label": info["model_label"],
        "vintage": info["vintage"],
        "run_id": info["run_id"],
        "frequency": info["frequency"],
        "p": info["p"],
        "variables": info["variables"],
        "target": info["target"],
        "units": info["units"],
        "recursive_ordering": info["recursive_ordering"],
        "reference_date": pd.Timestamp(irf["reference_date"]).isoformat(),
        "horizon": horizon,
        "posterior_draws": int(info["selected_posterior_draws"]),
        "available_posterior_draws": int(info["available_posterior_draws"]),
        "selected_draw_indices": info["selected_draw_indices"],
        "missing_data_method": info["missing_data_method"],
        "exog_names": info["exog_names"],
        "shock_unit": shock_unit,
        "shock_size": shock_size,
        "shock_scale_factors": _shock_scale_rows(irf),
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
        "volatility": volatility_rows,
        "irf": _irf_rows(irf),
        "fevd": _fevd_rows(fevd),
        "hd": hd_rows,
        "hd_component_names": hd_component_names,
        "cache_usage": {
            "reference_invariant_reused": bool(invariant_reused),
        },
        "diagnostics": {
            "fevd_max_share_sum_error": float(fevd["max_share_sum_error"]),
            "hd_max_reconstruction_error_drawwise": hd_max_reconstruction_error,
            "hd_posterior_mean_reconstruction_error": float(mean_hd_error),
            "recursive_acceptance_rate": 1.0,
        },
    }
    return payload


def _empty_figure(message: str, *, height: int = 500) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.52, xref="paper", yref="paper",
        text=message, showarrow=False,
        font={"size": 14, "color": "#6b7280"},
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 48, "r": 24, "t": 56, "b": 48},
        xaxis={"visible": False},
        yaxis={"visible": False},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig



def _layout(
    fig: go.Figure,
    *,
    title: str | None = None,
    y_title: str | None = None,
    height: int = 500,
    uirevision: str,
) -> go.Figure:
    """Modern structural-chart geometry; the Dash card owns the visible heading."""
    fig.update_layout(
        template="plotly_white",
        title=(None if not title else {"text": title, "x": 0.01, "xanchor": "left"}),
        height=height,
        margin={"l": 68, "r": 28, "t": 34 if not title else 64, "b": 74},
        hovermode="x unified",
        dragmode="pan",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#334155", "size": 12},
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
        return _empty_figure("Loading structural analysis…")
    if not payload.get("ok"):
        return _empty_figure(
            "Structural analysis failed: " + str(payload.get("error", "unknown error"))
        )
    if response not in payload["variables"] or shock not in payload["variables"]:
        return _empty_figure("Select a valid shock and response.")

    frame = pd.DataFrame(payload["irf"])
    part = frame.loc[
        (frame["response"] == response)
        & (frame["shock"] == shock)
        & (frame["metric"] == metric)
    ].sort_values("horizon")
    if part.empty:
        return _empty_figure("No IRF summary is available for this selection.")

    fig = go.Figure()
    if fan_mode in {"90", "both"}:
        fig.add_trace(go.Scatter(
            x=part["horizon"], y=part["q95"],
            mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=part["horizon"], y=part["q05"],
            mode="lines", line={"width": 0}, fill="tonexty",
            fillcolor="rgba(37,99,235,0.10)", name="90% interval", hoverinfo="skip",
        ))
    if fan_mode in {"68", "both"}:
        fig.add_trace(go.Scatter(
            x=part["horizon"], y=part["q84"],
            mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=part["horizon"], y=part["q16"],
            mode="lines", line={"width": 0}, fill="tonexty",
            fillcolor="rgba(37,99,235,0.22)", name="68% interval", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=part["horizon"], y=part["q50"],
        mode="lines+markers",
        line={"width": 2.5, "color": "#2563EB"},
        marker={"size": 4, "color": "#2563EB"},
        name="Posterior median",
        hovertemplate="h=%{x}<br>%{y:.4g}<extra>Posterior median</extra>",
    ))
    fig.add_hline(y=0.0, line_width=1.1, line_color="#94A3B8")
    frequency = str(payload.get("frequency", "period"))
    x_title = "Weeks" if frequency == "weekly" else "Months" if frequency == "monthly" else "Periods"
    fig.update_xaxes(title=f"Horizon ({x_title.lower()})", dtick=None)
    unit = str(payload.get("units", {}).get(response, ""))
    y_title = unit if metric == "change" else f"Cumulative {unit}".strip()
    return _layout(
        fig,
        y_title=y_title,
        height=470,
        uirevision=(
            f"{payload.get('run_id')}::irf::{response}::{shock}::{metric}::"
            f"{payload.get('shock_unit')}::{payload.get('shock_size')}"
        ),
    )



def fevd_figure(
    payload: Mapping[str, Any] | None,
    *,
    response: str | None,
) -> go.Figure:
    """Posterior-median FEVD as 100% stacked bars rather than an area wall."""
    if not payload:
        return _empty_figure("Loading structural analysis…", height=420)
    if not payload.get("ok"):
        return _empty_figure(
            "Structural analysis failed: " + str(payload.get("error", "unknown error")),
            height=420,
        )
    if response not in payload["variables"]:
        return _empty_figure("Select a valid response variable.", height=420)

    frame = pd.DataFrame(payload["fevd"])
    part = frame.loc[frame["response"] == response].copy()
    if part.empty:
        return _empty_figure("No FEVD summary is available for this response.", height=420)

    fig = go.Figure()
    for j, shock in enumerate(payload["variables"]):
        block = part.loc[part["shock"] == shock].sort_values("horizon")
        if block.empty:
            continue
        label = shock.replace("_", " ").title()
        fig.add_trace(go.Bar(
            x=block["horizon"],
            y=block["q50"],
            name=label,
            marker={"color": _STRUCTURAL_COLOURS[j % len(_STRUCTURAL_COLOURS)]},
            hovertemplate="h=%{x}<br>%{y:.1f}%<extra>" + label + "</extra>",
        ))
    fig.update_layout(barmode="stack", bargap=0.18)
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    frequency = str(payload.get("frequency", "period"))
    unit = "weeks" if frequency == "weekly" else "months" if frequency == "monthly" else "periods"
    fig.update_xaxes(title=f"Forecast horizon ({unit})")
    return _layout(
        fig,
        y_title="Forecast-error variance share",
        height=420,
        uirevision=f"{payload.get('run_id')}::fevd::{response}::{payload.get('reference_date')}",
    )



def historical_decomposition_figure(
    payload: Mapping[str, Any] | None,
    *,
    response: str | None,
    last_obs: int | None = 156,
    relayout_data: Mapping[str, Any] | None = None,
) -> go.Figure:
    if not payload:
        return _empty_figure("Loading structural analysis…", height=560)
    if not payload.get("ok"):
        return _empty_figure(
            "Structural analysis failed: " + str(payload.get("error", "unknown error")),
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
        return _empty_figure("No historical decomposition is available.", height=560)

    variables = list(payload.get("variables", []))
    fig = go.Figure()
    for component in payload["hd_component_names"]:
        if component not in part.columns:
            continue
        base_name = str(component).split(":", 1)[0].strip()
        try:
            colour_index = variables.index(base_name)
        except ValueError:
            colour_index = 0
        colour = _STRUCTURAL_COLOURS[colour_index % len(_STRUCTURAL_COLOURS)]
        is_outlier = "outlier amplification" in str(component).lower()
        marker_colour = colour
        opacity = 0.36 if is_outlier else 0.88
        label = str(component).replace("_", " ").title()
        fig.add_trace(go.Bar(
            x=part["date"],
            y=part[component],
            name=label,
            marker={"color": marker_colour},
            opacity=opacity,
            hovertemplate="%{y:.4g}<extra>" + label + "</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=part["date"],
        y=part["observed"],
        mode="lines",
        line={"width": 2.4, "color": "#0F172A"},
        name=str(payload.get("hd_observed_label", "Observed change")),
        hovertemplate=(
            "%{y:.4g}<extra>"
            + str(payload.get("hd_observed_label", "Observed change"))
            + "</extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=part["date"],
        y=part["base"],
        mode="lines",
        line={"width": 1.5, "dash": "dash", "color": "#64748B"},
        name="Base / initial conditions",
        hovertemplate="%{y:.4g}<extra>Base / initial conditions</extra>",
    ))
    fig.update_layout(barmode="relative", bargap=0.0)
    fig.add_hline(y=0.0, line_width=1.1, line_color="#94A3B8")
    unit = str(payload.get("units", {}).get(response, ""))
    fig = _layout(
        fig,
        y_title=f"Change ({unit})" if unit else "Change",
        height=560,
        uirevision=f"{payload.get('run_id')}::hd::{response}::{last_obs}",
    )
    return _adaptive_hd_yaxis(fig, relayout_data)


def volatility_history_figure(
    payload: Mapping[str, Any] | None,
    *,
    variable: str | None,
    reference_date=None,
    relayout_data: Mapping[str, Any] | None = None,
) -> go.Figure:
    """SV/outlier history used to choose a reference volatility state."""
    if not payload:
        return _empty_figure("Loading volatility history…", height=470)
    if not payload.get("ok"):
        return _empty_figure(
            "Structural analysis failed: " + str(payload.get("error", "unknown error")),
            height=470,
        )
    if variable not in payload.get("variables", []):
        return _empty_figure("Select a valid volatility series.", height=470)

    frame = pd.DataFrame(payload.get("volatility", []))
    if frame.empty:
        return _empty_figure("No saved volatility history is available.", height=470)

    part = frame.loc[frame["variable"] == variable].copy()
    part["date"] = pd.to_datetime(part["date"])
    part = part.sort_values("date")
    if part.empty:
        return _empty_figure("No volatility history for this variable.", height=470)

    fig = go.Figure()

    # 68% posterior band for persistent SV sqrt(lambda_t).
    fig.add_trace(
        go.Scatter(
            x=part["date"],
            y=part["persistent_q84"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=part["date"],
            y=part["persistent_q16"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(37,99,235,0.13)",
            name="Persistent SV 68% interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=part["date"],
            y=part["persistent_q50"],
            mode="lines",
            line={"width": 2.3, "color": "#2563eb"},
            name="Persistent SV median √λ",
            customdata=np.column_stack(
                [
                    part["persistent_q16"],
                    part["persistent_q84"],
                    part["outlier_probability"],
                ]
            ),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "persistent median=%{y:.4g}<br>"
                "68% interval=[%{customdata[0]:.4g}, %{customdata[1]:.4g}]<br>"
                "P(outlier)=%{customdata[2]:.1%}"
                "<extra>Persistent SV</extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=part["date"],
            y=part["total_q50"],
            mode="lines",
            line={"width": 1.4, "color": "#64748b"},
            name="Total scale median o√λ",
            customdata=part[["outlier_probability"]].to_numpy(),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "total scale=%{y:.4g}<br>"
                "P(outlier)=%{customdata[0]:.1%}"
                "<extra>Total scale</extra>"
            ),
        )
    )

    flagged = part.loc[part["outlier_probability"] > 0.50]
    if not flagged.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged["date"],
                y=flagged["total_q50"],
                mode="markers",
                marker={"size": 8, "color": "#dc4c64"},
                name="Outlier periods P>0.5",
                customdata=flagged[["outlier_probability"]].to_numpy(),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    "total scale=%{y:.4g}<br>"
                    "P(outlier)=%{customdata[0]:.1%}"
                    "<extra>Outlier period</extra>"
                ),
            )
        )

    if reference_date is not None:
        ref = pd.Timestamp(reference_date)
        ref_x = ref.isoformat()
        # Avoid Plotly's legacy add_vline annotation arithmetic on pandas
        # Timestamp objects; add the axis-spanning shape and annotation
        # explicitly instead.
        fig.add_shape(
            type="line",
            x0=ref_x,
            x1=ref_x,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line={"width": 1.5, "dash": "dash", "color": "#7c3aed"},
        )
        fig.add_annotation(
            x=ref_x,
            y=0.98,
            xref="x",
            yref="paper",
            text=f"Reference · {ref.strftime('%Y-%m-%d')}",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={"size": 10, "color": "#7c3aed"},
            bgcolor="rgba(255,255,255,0.72)",
        )

    unit = str(payload.get("units", {}).get(variable, ""))
    fig = _layout(
        fig,
        title=(
            "Reference volatility history — "
            + variable.replace("_", " ").title()
        ),
        y_title=(
            f"Structural innovation scale ({unit})"
            if unit
            else "Structural innovation scale"
        ),
        height=500,
        uirevision=f"{payload.get('run_id')}::volatility::{variable}",
    )
    fig.update_yaxes(type="log")
    fig.update_layout(
        hovermode="x unified",
        margin={"l": 68, "r": 28, "t": 80, "b": 96},
    )
    return _adaptive_line_yaxis(
        fig,
        relayout_data,
        log_axis=True,
        padding_fraction=0.08,
    )



__all__ = [
    "STRUCTURAL_CONTRACT_VERSION",
    "STRUCTURAL_INVARIANT_CACHE_VERSION",
    "STRUCTURAL_LAZY_HD_CONTRACT_VERSION",
    "DEFAULT_STRUCTURAL_DRAWS",
    "DEFAULT_HORIZON",
    "MAX_STRUCTURAL_DRAWS",
    "MAX_HORIZON",
    "DEFAULT_SHOCK_UNIT",
    "DEFAULT_SHOCK_SIZE",
    "StructuralDashboardError",
    "resolve_run_directory",
    "structural_run_contract",
    "load_structural_result",
    "compute_structural_v1",
    "compute_structural_volatility_v1",
    "compute_structural_irf_fevd_v1",
    "compute_structural_hd_v1",
    "structural_invariant_cache_from_payload",
    "irf_figure",
    "fevd_figure",
    "historical_decomposition_figure",
    "volatility_history_figure",
    "reference_regime_date",
    "reference_volatility_snapshot",
    "volatility_sparkline_figure",
    "relative_volatility_state_figure",
]
