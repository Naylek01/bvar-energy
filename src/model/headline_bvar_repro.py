"""Exact saved-baseline reconstruction for Headline Delivery 1.

No Gibbs sampling and no scenario are run here. The purpose is to establish a
bit-identical regression harness before any dashboard runtime is removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy_bvar_io import load_energy_bvar_draws
from energy_bvar_model import make_bvar_svo_prior, prepare_bvar_panel
from headline_joint_bvar import (
    STATE_VARIABLES,
    aggregate_headline_draws,
    native_to_state_levels,
)
from headline_bvar_integrity import (
    current_headline_source_contract,
    load_integrity_contract,
    validate_run_directory,
)
from headline_bvar_pipeline import (
    BASELINE_P,
    SEASONAL_EXOG_PRIOR_SCALE,
    build_inputs,
    exact_headline_yoy_contribution_paths,
    forecast_locked_headline,
    make_joint_seasonals,
    production_prior_config,
)


class HeadlineReproError(RuntimeError):
    pass


_REQUIRED_POSTERIOR_ARRAYS = (
    "B",
    "A",
    "log_variance",
    "phi",
    "outlier_probabilities",
)


def _max_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(a[finite] - b[finite])))


def _comparison(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    return {
        "name": name,
        "actual_shape": list(actual_arr.shape),
        "expected_shape": list(expected_arr.shape),
        "bit_identical": bool(
            actual_arr.shape == expected_arr.shape
            and np.array_equal(actual_arr, expected_arr, equal_nan=True)
        ),
        "max_abs_error": _max_abs_error(actual_arr, expected_arr),
    }


def load_saved_headline_result_for_repro(
    run_directory: str | Path,
    *,
    project_root: str | Path,
):
    """Rebuild the DK prep with the true current prepare_bvar_panel API."""
    run_dir = Path(run_directory)
    metadata = validate_run_directory(run_dir)

    unsupported = metadata.get("missing_data_method")
    if unsupported not in (None, "", "dk"):
        raise HeadlineReproError(
            "Headline Delivery 1 supports only the production DK contract; "
            f"missing_data_method={unsupported!r} is not supported."
        )

    inputs = build_inputs(str(metadata["vintage"]), project_root=project_root)
    state = native_to_state_levels(inputs.native_levels)
    seasonals = make_joint_seasonals(state.index)
    prep = prepare_bvar_panel(
        state,
        p=BASELINE_P,
        variables=STATE_VARIABLES,
        exog=seasonals,
        frequency="monthly",
    )

    saved_metadata, arrays = load_energy_bvar_draws(run_dir)
    for key in _REQUIRED_POSTERIOR_ARRAYS:
        if key not in arrays:
            raise HeadlineReproError(
                f"Saved posterior is missing required array {key!r}."
            )

    # Forecast code only needs the prior's outlier grid, but rebuilding the
    # validated production prior keeps the saved-result object honest.
    prior = make_bvar_svo_prior(
        prep,
        production_prior_config(),
        exog_prior_scale=SEASONAL_EXOG_PRIOR_SCALE,
        active_lags=None,
    )
    result = {
        "metadata": saved_metadata,
        "prep": prep,
        "prior": prior,
        "variables": list(STATE_VARIABLES),
        "p": BASELINE_P,
        "n_draws": int(np.asarray(arrays["B"]).shape[0]),
        **arrays,
    }
    return result, inputs


def validate_repro_lineage(
    run_directory: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    contract = load_integrity_contract(run_directory)
    current = current_headline_source_contract(
        contract["vintage"], project_root=project_root
    )
    expected_hash = contract.get("source_data_hash")
    if expected_hash is None:
        raise HeadlineReproError(
            "No source_data_hash is available. Backfill data_integrity.json "
            "or use a new lineage-aware run."
        )
    if str(expected_hash) != str(current["source_data_hash"]):
        raise HeadlineReproError(
            "Current processed Headline inputs do not match the saved run lineage."
        )
    for key in ("n_forecast_draws", "forecast_seed", "simulate_future_outliers"):
        if contract.get(key) is None:
            raise HeadlineReproError(
                f"Forecast reconstruction field {key!r} is unavailable. "
                "A4 cannot claim bit reproducibility for this run."
            )
    return contract


def reproduce_saved_headline_baseline(
    run_directory: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    run_dir = Path(run_directory)
    contract = validate_repro_lineage(run_dir, project_root=project_root)
    result, inputs = load_saved_headline_result_for_repro(
        run_dir, project_root=project_root
    )

    if contract.get("exog_names") is not None:
        if list(result["prep"].get("exog_names", [])) != list(contract["exog_names"]):
            raise HeadlineReproError("Reconstructed exog_names differ from lineage.")
    for key in ("balanced_end", "last_calendar_date"):
        expected = contract.get(key)
        if expected is not None and pd.Timestamp(result["prep"][key]) != pd.Timestamp(expected):
            raise HeadlineReproError(
                f"Reconstructed {key} differs from persisted lineage."
            )

    forecast = forecast_locked_headline(
        result,
        inputs=inputs,
        H=12,
        n_draws=int(contract["n_forecast_draws"]),
        simulate_future_outliers=bool(contract["simulate_future_outliers"]),
        seed=int(contract["forecast_seed"]),
    )
    aggregate = aggregate_headline_draws(
        forecast,
        native_levels=inputs.native_levels,
        weights=inputs.weights,
        official_total=inputs.official_total,
    )
    contributions = exact_headline_yoy_contribution_paths(aggregate, inputs=inputs)

    forecast_dir = run_dir / "forecasts" / "unconditional"
    headline_path = forecast_dir / "headline_draws.npz"
    if not headline_path.is_file():
        raise FileNotFoundError(headline_path)
    with np.load(headline_path, allow_pickle=False) as archive:
        saved_headline = {name: archive[name] for name in archive.files}

    comparisons: list[dict[str, Any]] = []
    for key, actual in (
        ("native_level_paths", forecast["native_level_paths"]),
        ("component_yoy_paths", forecast["component_yoy_paths"]),
        ("headline_level_paths", aggregate["headline_level_paths"]),
        ("headline_yoy_paths", aggregate["headline_yoy_paths"]),
        ("headline_yoy_contribution_paths", contributions),
    ):
        if key not in saved_headline:
            raise HeadlineReproError(f"headline_draws.npz is missing {key!r}.")
        comparisons.append(_comparison(key, actual, saved_headline[key]))

    state_path = forecast_dir / "forecast_draws.npz"
    if state_path.is_file():
        with np.load(state_path, allow_pickle=False) as archive:
            if "level_paths" in archive.files:
                comparisons.append(
                    _comparison("state_level_paths", forecast["level_paths"], archive["level_paths"])
                )
            if "draw_indices" in archive.files:
                comparisons.append(
                    _comparison("draw_indices", forecast["draw_indices"], archive["draw_indices"])
                )

    passed = all(row["bit_identical"] for row in comparisons)
    report = {
        "run_id": contract["run_id"],
        "vintage": contract["vintage"],
        "verification_status": contract["verification_status"],
        "forecast_contract": {
            "n_forecast_draws": int(contract["n_forecast_draws"]),
            "forecast_seed": int(contract["forecast_seed"]),
            "simulate_future_outliers": bool(contract["simulate_future_outliers"]),
            "provenance": contract["provenance"],
        },
        "comparisons": comparisons,
        "bit_identical": bool(passed),
    }
    if not passed:
        failures = [row for row in comparisons if not row["bit_identical"]]
        raise HeadlineReproError(
            "Saved baseline is not bit-identical: "
            + ", ".join(
                f"{row['name']} max_abs_error={row['max_abs_error']:.3e}"
                for row in failures
            )
        )
    return report


def save_repro_report(report: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return path


__all__ = [
    "HeadlineReproError",
    "load_saved_headline_result_for_repro",
    "validate_repro_lineage",
    "reproduce_saved_headline_baseline",
    "save_repro_report",
]
