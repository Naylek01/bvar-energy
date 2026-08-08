"""Serialization and cache helpers for energy BVAR results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from energy_bvar_model import (
    coefficient_posterior_table,
    mcmc_diagnostics,
    stability_diagnostics,
)

_HEAVY_ARRAYS = (
    "B",
    "A",
    "log_variance",
    "phi",
    "outlier_scales",
    "outlier_indicators",
    "outlier_probabilities",
    "posterior_outlier_probability_draws",
    "spectral_radius",
    "log_likelihood",
    "completed_differences_draws",
    "missing_level_draws",
    "last_companion_state_draws",
)


def _write_table(frame: pd.DataFrame, base: Path) -> Path:
    try:
        path = base.with_suffix(".parquet")
        frame.to_parquet(path)
        return path
    except (ImportError, ModuleNotFoundError):
        path = base.with_suffix(".csv")
        frame.to_csv(path)
        return path


def model_summary_frame(result: Mapping) -> pd.DataFrame:
    coefficients = coefficient_posterior_table(result).reset_index().rename(columns={"index": "parameter"})
    coefficients.insert(0, "block", "coefficient")

    phi = np.asarray(result["phi"], dtype=float)
    p_out = np.asarray(result["outlier_probabilities"], dtype=float)
    rows = []
    for j, name in enumerate(result["variables"]):
        for block, draws in (("phi", phi[:, j]), ("outlier_probability", p_out[:, j])):
            q = np.percentile(draws, [5, 16, 50, 84, 95])
            rows.append({
                "block": block,
                "parameter": name,
                "posterior_mean": float(np.mean(draws)),
                "posterior_median": float(q[2]),
                "posterior_sd": float(np.std(draws, ddof=1)),
                "q05": float(q[0]),
                "q16": float(q[1]),
                "q84": float(q[3]),
                "q95": float(q[4]),
                "ESS": np.nan,
            })
    extra = pd.DataFrame(rows)
    return pd.concat([coefficients, extra], ignore_index=True, sort=False)


def diagnostics_frame(result: Mapping) -> pd.DataFrame:
    mcmc = mcmc_diagnostics(result).reset_index().rename(columns={"index": "parameter"})
    mcmc.insert(0, "diagnostic_group", "mcmc")
    stability = stability_diagnostics(result).rename("value").reset_index().rename(columns={"index": "parameter"})
    stability.insert(0, "diagnostic_group", "stability")
    return pd.concat([mcmc, stability], ignore_index=True, sort=False)


def save_energy_bvar_result(result: Mapping, output_root: str | Path) -> dict[str, Path]:
    metadata = dict(result.get("metadata", {}))
    required = {"model_id", "vintage", "run_id"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"Result metadata is missing {sorted(missing)}. Use run_energy_bvar().")

    directory = (
        Path(output_root)
        / str(metadata["model_id"])
        / str(metadata["vintage"])
        / str(metadata["run_id"])
    )
    directory.mkdir(parents=True, exist_ok=True)

    metadata_path = directory / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    summary_path = _write_table(model_summary_frame(result), directory / "summary")
    diagnostics_path = _write_table(diagnostics_frame(result), directory / "diagnostics")
    draws_path = directory / "draws.npz"
    np.savez_compressed(
        draws_path,
        **{name: np.asarray(result[name]) for name in _HEAVY_ARRAYS if name in result},
    )
    return {
        "directory": directory,
        "metadata": metadata_path,
        "summary": summary_path,
        "diagnostics": diagnostics_path,
        "draws": draws_path,
    }


def load_energy_bvar_draws(run_directory: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    directory = Path(run_directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with np.load(directory / "draws.npz", allow_pickle=False) as archive:
        draws = {name: archive[name] for name in archive.files}
    return metadata, draws


# -----------------------------------------------------------------------------
# Forecast serialization
# -----------------------------------------------------------------------------

_FORECAST_ARRAY_KEYS = (
    "diff_paths",
    "level_paths",
    "future_diff_paths",
    "future_level_paths",
    "future_log_variance",
    "future_outlier_scales",
    "draw_indices",
)

_FORECAST_METADATA_KEYS = (
    "variables",
    "tail_length",
    "frequency",
    "calendar_rule",
    "period_name",
    "H",
    "simulate_future_outliers",
    "balanced_end",
    "last_calendar_date",
    "missing_data_method",
    "missing_treatment_exact",
    "missing_data_approximation_used",
)


def _forecast_metadata(forecast: Mapping, forecast_name: str) -> dict:
    """Return the small JSON-safe contract needed to reload one forecast."""
    required = {
        "level_paths",
        "path_dates",
        "future_dates",
        "tail_length",
        "variables",
        "frequency",
    }
    missing = required.difference(forecast)
    if missing:
        raise ValueError(
            f"Forecast {forecast_name!r} is missing required keys {sorted(missing)}."
        )

    level_paths = np.asarray(forecast["level_paths"], dtype=float)
    path_dates = pd.DatetimeIndex(forecast["path_dates"])
    future_dates = pd.DatetimeIndex(forecast["future_dates"])
    variables = list(forecast["variables"])

    if level_paths.ndim != 3:
        raise ValueError(
            "forecast['level_paths'] must have shape "
            "(n_draws, n_path_dates, n_variables)."
        )
    if level_paths.shape[1] != len(path_dates):
        raise ValueError("Forecast level_paths and path_dates are inconsistent.")
    if level_paths.shape[2] != len(variables):
        raise ValueError("Forecast level_paths and variables are inconsistent.")
    tail_length = int(forecast["tail_length"])
    if tail_length < 0 or tail_length > len(path_dates):
        raise ValueError("Forecast tail_length is invalid.")
    if not path_dates[tail_length:].equals(future_dates):
        raise ValueError(
            "Forecast future_dates must equal path_dates[tail_length:]."
        )

    metadata = {
        "forecast_schema_version": "1.0",
        "forecast_name": str(forecast_name),
        "n_draws": int(level_paths.shape[0]),
        "n_path_dates": int(level_paths.shape[1]),
        "n_variables": int(level_paths.shape[2]),
        "path_dates": [stamp.isoformat() for stamp in path_dates],
        "future_dates": [stamp.isoformat() for stamp in future_dates],
        "tail_dates": [
            stamp.isoformat()
            for stamp in pd.DatetimeIndex(
                forecast.get("tail_dates", path_dates[:tail_length])
            )
        ],
    }
    for key in _FORECAST_METADATA_KEYS:
        if key not in forecast:
            continue
        value = forecast[key]
        if isinstance(value, (pd.Timestamp,)):
            value = value.isoformat()
        elif isinstance(value, pd.DatetimeIndex):
            value = [stamp.isoformat() for stamp in value]
        elif isinstance(value, np.generic):
            value = value.item()
        elif key == "variables":
            value = list(value)
        metadata[key] = value

    # Conditions are part of the generation contract but are not duplicated
    # as columns in the draw store. Keep only compact, JSON-safe paths here.
    conditions = forecast.get("level_conditions", {})
    if conditions:
        condition_meta = {}
        for name, values in dict(conditions).items():
            arr = np.asarray(values, dtype=float).reshape(-1)
            if len(arr) != len(future_dates):
                raise ValueError(
                    f"Condition {name!r} has length {len(arr)}; "
                    f"expected {len(future_dates)}."
                )
            condition_meta[str(name)] = arr.tolist()
        metadata["level_conditions"] = condition_meta

    return metadata


def save_energy_bvar_forecast(
    forecast: Mapping,
    run_directory: str | Path,
    forecast_name: str = "unconditional",
) -> dict[str, Path]:
    """Persist posterior predictive paths without rerunning the Gibbs sampler.

    The forecast is stored below ``<run_directory>/forecasts/<forecast_name>/``.
    Only generation-level predictive objects are serialized here. Tax scenarios
    are intentionally excluded: taxes are applied ex post from the processed
    vintage, so the same pre-tax forecast store can be reused by the aggregate
    notebook and later by the dashboard.
    """
    directory = Path(run_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Run directory not found: {directory}")
    if not (directory / "metadata.json").is_file():
        raise FileNotFoundError(
            f"Run metadata not found in {directory}; save the BVAR result first."
        )

    name = str(forecast_name).strip()
    if not name:
        raise ValueError("forecast_name cannot be empty.")
    if any(piece in name for piece in ("/", "\\", "..")):
        raise ValueError("forecast_name must be a simple directory name.")

    metadata = _forecast_metadata(forecast, name)
    target = directory / "forecasts" / name
    target.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for key in _FORECAST_ARRAY_KEYS:
        if key in forecast:
            arrays[key] = np.asarray(forecast[key])

    # ``level_paths`` is required and therefore must be present.
    if "level_paths" not in arrays:
        raise RuntimeError("Internal forecast serialization error: level_paths missing.")

    arrays_path = target / "forecast_draws.npz"
    np.savez_compressed(arrays_path, **arrays)

    metadata_path = target / "forecast_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    # Deterministic/exogenous paths are small tables and are easier to audit
    # as CSV than as opaque arrays.
    exog_paths = {}
    for key in ("exog_path", "future_exog"):
        value = forecast.get(key)
        if value is None:
            continue
        frame = pd.DataFrame(value).copy()
        if isinstance(value, pd.DataFrame):
            frame.index = value.index
            frame.columns = value.columns
        if not isinstance(frame.index, pd.DatetimeIndex):
            expected_dates = (
                pd.DatetimeIndex(forecast["path_dates"])
                if key == "exog_path"
                else pd.DatetimeIndex(forecast["future_dates"])
            )
            if len(frame) != len(expected_dates):
                raise ValueError(
                    f"{key} has {len(frame)} rows; expected {len(expected_dates)}."
                )
            frame.index = expected_dates
        frame.index.name = "date"
        path = target / f"{key}.csv"
        frame.to_csv(path, date_format="%Y-%m-%d")
        exog_paths[key] = path

    return {
        "directory": target,
        "metadata": metadata_path,
        "draws": arrays_path,
        **exog_paths,
    }



def save_energy_bvar_forecast_for_result(
    forecast: Mapping,
    result: Mapping,
    output_root: str | Path,
    forecast_name: str = "unconditional",
) -> dict[str, Path]:
    """Save a forecast without requiring the heavy MCMC result cache.

    The fitted result's cache-safe metadata determines the same
    ``model_id / vintage / run_id`` directory used by ``save_energy_bvar_result``.
    Only ``metadata.json`` plus the forecast store are written. This is the
    lightweight path used by notebooks 03--09 when Notebook 10 needs predictive
    draws but the full Gibbs arrays do not need to be persisted.
    """
    metadata = dict(result.get("metadata", {}))
    required = {"model_id", "vintage", "run_id"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(
            f"Result metadata is missing {sorted(missing)}. Use run_energy_bvar()."
        )

    directory = (
        Path(output_root)
        / str(metadata["model_id"])
        / str(metadata["vintage"])
        / str(metadata["run_id"])
    )
    directory.mkdir(parents=True, exist_ok=True)

    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in required:
            if str(existing.get(key)) != str(metadata[key]):
                raise ValueError(
                    f"Existing run metadata disagrees on {key!r}: "
                    f"{existing.get(key)!r} != {metadata[key]!r}."
                )
    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )

    return save_energy_bvar_forecast(
        forecast,
        directory,
        forecast_name=forecast_name,
    )



def load_energy_bvar_forecast(
    forecast_directory: str | Path,
) -> dict:
    """Reload a forecast saved by :func:`save_energy_bvar_forecast`."""
    directory = Path(forecast_directory)
    metadata_path = directory / "forecast_metadata.json"
    arrays_path = directory / "forecast_draws.npz"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not arrays_path.is_file():
        raise FileNotFoundError(arrays_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        forecast = {name: archive[name] for name in archive.files}

    forecast.update(metadata)
    forecast["path_dates"] = pd.DatetimeIndex(
        pd.to_datetime(metadata["path_dates"]), name="date"
    )
    forecast["future_dates"] = pd.DatetimeIndex(
        pd.to_datetime(metadata["future_dates"]), name="date"
    )
    forecast["tail_dates"] = pd.DatetimeIndex(
        pd.to_datetime(metadata.get("tail_dates", [])), name="date"
    )
    for key in ("balanced_end", "last_calendar_date"):
        if metadata.get(key) is not None:
            forecast[key] = pd.Timestamp(metadata[key])

    # Recreate compact condition arrays for callers that need provenance.
    if "level_conditions" in metadata:
        forecast["level_conditions"] = {
            name: np.asarray(values, dtype=float)
            for name, values in metadata["level_conditions"].items()
        }

    for key in ("exog_path", "future_exog"):
        path = directory / f"{key}.csv"
        if path.is_file():
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            frame.index = pd.DatetimeIndex(frame.index, name="date")
            forecast[key] = frame

    # Hard reload checks prevent a corrupt cache from silently entering the
    # aggregate forecast.
    level_paths = np.asarray(forecast["level_paths"], dtype=float)
    if level_paths.ndim != 3:
        raise ValueError("Reloaded forecast level_paths is not three-dimensional.")
    if level_paths.shape[1] != len(forecast["path_dates"]):
        raise ValueError("Reloaded forecast path date count is inconsistent.")
    if level_paths.shape[2] != len(forecast["variables"]):
        raise ValueError("Reloaded forecast variable count is inconsistent.")
    tail_length = int(forecast["tail_length"])
    if not forecast["path_dates"][tail_length:].equals(forecast["future_dates"]):
        raise ValueError("Reloaded forecast future-date contract is inconsistent.")

    return forecast


__all__ = [
    "model_summary_frame",
    "diagnostics_frame",
    "save_energy_bvar_result",
    "load_energy_bvar_draws",
    "save_energy_bvar_forecast",
    "save_energy_bvar_forecast_for_result",
    "load_energy_bvar_forecast",
]
