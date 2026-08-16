"""One-step posterior fitted paths for the locked Headline BVAR.

Post-run materialiser only: it never re-estimates the BVAR.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from headline_bvar_pipeline import BASELINE_P, NATIVE_VARIABLES, STATE_VARIABLES, build_inputs, make_joint_seasonals
from headline_joint_bvar import aggregate_headline_draws, native_to_state_levels

QUANTILES = (0.05, 0.16, 0.50, 0.84, 0.95)
QCOLS = ("q05", "q16", "q50", "q84", "q95")
FITTED_CONTRACT = "one_step_conditional_drawwise_v1"


class HeadlineFittedError(RuntimeError):
    pass


def _normalise_B(B: np.ndarray, *, n: int, k_expected: int) -> np.ndarray:
    arr = np.asarray(B, dtype=float)
    if arr.ndim == 3:
        if arr.shape[1:] != (k_expected, n):
            raise HeadlineFittedError(
                f"Saved B has shape {arr.shape}; expected (draws, {k_expected}, {n})."
            )
        return arr
    if arr.ndim == 2 and arr.shape[1] == k_expected * n:
        return arr.reshape(arr.shape[0], k_expected, n, order="F")
    raise HeadlineFittedError(f"Unsupported saved B shape {arr.shape}.")


def build_fitted_design(native_levels: pd.DataFrame, *, p: int = BASELINE_P):
    """Return transformed changes, exact production X, and aligned fitted dates."""
    p = int(p)
    if p != BASELINE_P:
        raise HeadlineFittedError(f"Production fitted materialisation is locked to p={BASELINE_P}; got {p}.")
    native = native_levels[NATIVE_VARIABLES].copy().astype(float).sort_index()
    state = native_to_state_levels(native)
    changes = state.diff()
    complete = changes.notna().all(axis=1)
    if not complete.any():
        raise HeadlineFittedError("No complete transformed Headline observations.")
    first, last = complete[complete].index[[0, -1]]
    inside = changes.loc[first:last]
    if inside.isna().any().any():
        bad = inside.index[inside.isna().any(axis=1)][:10]
        raise HeadlineFittedError(
            "Interior missing transformed observations prevent exact fitted reconstruction: "
            + ", ".join(pd.DatetimeIndex(bad).strftime("%Y-%m"))
        )
    changes = inside
    if len(changes) <= p:
        raise HeadlineFittedError("Not enough observations after differencing and lags.")

    arr = changes.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(changes.index[p:], name="date")
    lag_blocks = [arr[p-lag:len(arr)-lag] for lag in range(1, p+1)]
    seasonals = make_joint_seasonals(changes.index).reindex(dates)
    if seasonals.isna().any().any() or seasonals.shape[1] != 11:
        raise HeadlineFittedError("Production seasonal design is incomplete.")
    X = np.column_stack([np.ones(len(dates)), *lag_blocks, seasonals.to_numpy(dtype=float)])
    cols = ["constant"]
    for lag in range(1, p+1):
        cols.extend([f"L{lag}.{name}" for name in STATE_VARIABLES])
    cols.extend(seasonals.columns.tolist())
    return changes, pd.DataFrame(X, index=dates, columns=cols), dates


def one_step_fitted_component_paths(native_levels: pd.DataFrame, B_draws: np.ndarray, *, p: int = BASELINE_P):
    """Compute draw-wise one-step fitted native component HICP levels."""
    _, X, dates = build_fitted_design(native_levels, p=p)
    n = len(NATIVE_VARIABLES)
    B = _normalise_B(B_draws, n=n, k_expected=X.shape[1])
    fitted_changes = np.einsum("tk,skn->stn", X.to_numpy(dtype=float), B)

    native = native_levels[NATIVE_VARIABLES].copy().sort_index()
    native.index = pd.DatetimeIndex(native.index).to_period("M").to_timestamp(how="start")
    previous_dates = dates - pd.offsets.MonthBegin(1)
    prev = native.reindex(previous_dates).to_numpy(dtype=float)
    if not np.isfinite(prev).all() or np.any(prev <= 0):
        raise HeadlineFittedError("Observed previous-month component levels are incomplete/non-positive.")
    fitted_native = prev[None, :, :] * np.exp(fitted_changes)
    if not np.isfinite(fitted_native).all() or np.any(fitted_native <= 0):
        raise HeadlineFittedError("Fitted native paths contain invalid values.")

    yoy = np.full_like(fitted_native, np.nan)
    for t, date in enumerate(dates):
        lag_date = pd.Timestamp(date) - pd.DateOffset(months=12)
        if lag_date not in native.index:
            continue
        denom = native.loc[lag_date, NATIVE_VARIABLES].to_numpy(dtype=float)
        if np.isfinite(denom).all() and np.all(denom > 0):
            yoy[:, t, :] = 100.0 * (fitted_native[:, t, :] / denom[None, :] - 1.0)

    return {
        "native_variables": list(NATIVE_VARIABLES),
        "state_variables": list(STATE_VARIABLES),
        "path_dates": dates,
        "tail_length": len(dates),
        "future_dates": pd.DatetimeIndex([], name="date"),
        "native_level_paths": fitted_native,
        "component_yoy_paths": yoy,
        "fitted_state_change_paths": fitted_changes,
        "fitted_contract": FITTED_CONTRACT,
    }


def fitted_headline_paths(component_fitted: Mapping, *, native_levels, weights, official_total):
    return aggregate_headline_draws(
        component_fitted,
        native_levels=native_levels,
        weights=weights,
        official_total=official_total,
        future_weight_policy="carry_forward_latest",
    )


def _display_rows(paths, dates, *, series, label, metric, unit, vintage, run_id, forecast_name):
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(dates):
        raise HeadlineFittedError(f"{series}/{metric}: fitted path dimensions changed.")
    q = np.nanquantile(arr, QUANTILES, axis=0).T
    out = pd.DataFrame({
        "record_type": "fitted", "scope": "headline", "model_id": "headline_joint",
        "vintage": str(vintage), "run_id": str(run_id), "forecast_name": str(forecast_name),
        "basis": "fitted", "metric": metric, "series": series, "label": label, "unit": unit,
        "date": dates, "segment": "fitted", "is_future": False, "value": np.nanmean(arr, axis=0),
    })
    for j, col in enumerate(QCOLS):
        out[col] = q[:, j]
    return out


def build_fitted_display_rows(*, component_fitted, aggregate_fitted, vintage, run_id, forecast_name="unconditional"):
    dates = pd.DatetimeIndex(component_fitted["path_dates"], name="date")
    native = np.asarray(component_fitted["native_level_paths"], dtype=float)
    component_yoy = np.asarray(component_fitted["component_yoy_paths"], dtype=float)
    headline_level = np.asarray(aggregate_fitted["headline_level_paths"], dtype=float)
    headline_yoy = np.asarray(aggregate_fitted["headline_yoy_paths"], dtype=float)
    labels = {"hicp_total":"Headline HICP","hicp_energy":"Energy","hicp_food":"Food","hicp_neig":"NEIG","hicp_services":"Services"}
    frames = [
        _display_rows(headline_level, dates, series="hicp_total", label=labels["hicp_total"], metric="level", unit="HICP index", vintage=vintage, run_id=run_id, forecast_name=forecast_name),
        _display_rows(headline_yoy, dates, series="hicp_total", label=labels["hicp_total"], metric="yoy", unit="% y/y", vintage=vintage, run_id=run_id, forecast_name=forecast_name),
    ]
    for j, name in enumerate(NATIVE_VARIABLES):
        frames += [
            _display_rows(native[:,:,j], dates, series=name, label=labels[name], metric="level", unit="HICP index", vintage=vintage, run_id=run_id, forecast_name=forecast_name),
            _display_rows(component_yoy[:,:,j], dates, series=name, label=labels[name], metric="yoy", unit="% y/y", vintage=vintage, run_id=run_id, forecast_name=forecast_name),
        ]
    frames.append(pd.DataFrame([{
        "record_type":"metadata","scope":"headline","model_id":"headline_joint",
        "vintage":str(vintage),"run_id":str(run_id),"forecast_name":str(forecast_name),
        "key":"fitted_contract","text_value":FITTED_CONTRACT,
    }]))
    return pd.concat(frames, ignore_index=True, sort=False)


def _write_atomic(frame: pd.DataFrame, path: Path):
    temp = path.with_name(path.name + ".slice5.tmp")
    if temp.exists():
        temp.unlink()
    frame.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, path)


def materialize_headline_fitted(run_directory: str | Path, *, project_root: str | Path | None = None, forecast_name="unconditional"):
    """Append fitted rows to the CURRENT display artifact while preserving all other rows."""
    run_dir = Path(run_directory).resolve()
    metadata_path, draws_path = run_dir / "metadata.json", run_dir / "draws.npz"
    display_path = run_dir / "forecasts" / forecast_name / "display_v1.parquet"
    for path in (metadata_path, draws_path, display_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_id") != "headline_joint":
        raise HeadlineFittedError("Selected run is not headline_joint.")
    vintage, run_id = str(metadata["vintage"]), str(metadata["run_id"])
    if run_dir.name != run_id:
        raise HeadlineFittedError("Run directory and metadata run_id disagree.")
    with np.load(draws_path, allow_pickle=False) as z:
        if "B" not in z.files:
            raise HeadlineFittedError("draws.npz does not contain B.")
        B = np.asarray(z["B"], dtype=float)

    native_path = run_dir / "headline_native_history.csv"
    weights_path = run_dir / "headline_weights_annual.csv"
    official_path = run_dir / "headline_official_total.csv"
    if all(p.is_file() for p in (native_path, weights_path, official_path)):
        native = pd.read_csv(native_path, index_col=0, parse_dates=True).sort_index()
        native.index = pd.DatetimeIndex(native.index, name="date")
        weights = pd.read_csv(weights_path, index_col=0)
        weights.index = pd.to_numeric(weights.index, errors="raise").astype(int)
        official = pd.read_csv(official_path, index_col=0, parse_dates=True).iloc[:,0].sort_index()
        official.index = pd.DatetimeIndex(official.index, name="date")
    else:
        inputs = build_inputs(vintage, project_root=project_root)
        native, weights, official = inputs.native_levels, inputs.weights, inputs.official_total

    component = one_step_fitted_component_paths(native, B)
    aggregate = fitted_headline_paths(component, native_levels=native, weights=weights, official_total=official)
    fitted = build_fitted_display_rows(component_fitted=component, aggregate_fitted=aggregate, vintage=vintage, run_id=run_id, forecast_name=forecast_name)

    current = pd.read_parquet(display_path)
    if "record_type" in current:
        current = current.loc[current["record_type"].astype(str) != "fitted"].copy()
    if {"record_type","key"}.issubset(current.columns):
        current = current.loc[~((current["record_type"].astype(str)=="metadata") & (current["key"].astype(str)=="fitted_contract"))].copy()
    columns = list(current.columns)
    for c in fitted.columns:
        if c not in columns:
            columns.append(c)
    for c in columns:
        if c not in current: current[c] = np.nan
        if c not in fitted: fitted[c] = np.nan
    combined = pd.concat([current[columns], fitted[columns]], ignore_index=True, sort=False)
    _write_atomic(combined, display_path)
    return {
        "run_directory": run_dir, "display_path": display_path, "vintage": vintage, "run_id": run_id,
        "n_B_draws": int(B.shape[0]), "n_fitted_dates": len(component["path_dates"]),
        "fitted_start": pd.Timestamp(component["path_dates"][0]), "fitted_end": pd.Timestamp(component["path_dates"][-1]),
        "fitted_contract": FITTED_CONTRACT, "bvar_reestimated": False,
    }


__all__ = ["FITTED_CONTRACT","HeadlineFittedError","build_fitted_design","one_step_fitted_component_paths","fitted_headline_paths","build_fitted_display_rows","materialize_headline_fitted"]
