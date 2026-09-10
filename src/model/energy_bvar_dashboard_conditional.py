"""Conditional observable-path scenarios for the Energy BVAR dashboard.

This module implements the dashboard counterpart of the conditional forecasts
already used in the component notebooks:

    "What if an observable endogenous commodity / upstream-price path were X?"

It does NOT re-estimate a BVAR.  It reloads the persisted Gibbs posterior,
replays the saved unconditional forecast with the original forecast seed, then
runs a second forecast with a future LEVEL path imposed through
``forecast_bvar_sv_outlier(..., level_conditions=...)``.

The two forecasts are paired:
- same posterior draw indices;
- same future stochastic-volatility innovations;
- same future outlier draws;
- same simulation-smoother random stream.

The module then maps the conditional component forecast through the production
HICP adapters and propagates it to the saved HICP Energy aggregate while
preserving the aggregate store's original cross-model pairing.  All Laspeyres,
chain-linking and contribution mathematics are delegated to
``energy_bvar_aggregate``.

V1 intentionally supports ONE conditioned component/variable at a time and a
permanent flat future level path.  Existing VAT/excise scenarios remain a
separate dashboard block.
"""

from __future__ import annotations
CONDITIONAL_WINDOWS_HARMONIZED_V1_ENERGY_ENGINE = True

import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from energy_bvar_model import (
    BVARSVOPriorConfig,
    forecast_bvar_sv_outlier,
    prepare_bvar_panel,
)
try:
    from energy_bvar_model import hash_model_data as _hash_model_data
except ImportError:  # legacy engine
    _hash_model_data = None

from energy_bvar_io import load_energy_bvar_forecast
from energy_bvar_pipeline import (
    CANONICAL_MODEL_IDS,
    build_panel,
    model_contract,
    model_spec,
)

from energy_bvar_aggregate import (
    MODEL_AGGREGATE_COMPONENTS,
    aggregate_component_draw_paths_laspeyres,
    aggregate_draw_paths_laspeyres,
    carry_last_index_path,
    draw_yoy_and_contributions,
    filter_positive_price_draws,
    historical_model_component_reconstruction,
    load_aggregation_inputs,
    rebase_price_paths_to_hicp_index,
    summarise_draw_paths,
    weekly_paths_with_history_to_monthly_mean,
)
from energy_bvar_gas import load_gas_tax_context, forecast_to_hicp_gas
from energy_bvar_electricity import (
    load_electricity_tax_context,
    forecast_to_hicp_electricity,
)
from energy_bvar_monthly_hicp import forecast_hicp_component
from energy_bvar_weekly_fuels import (
    load_weekly_tax_context,
    reattribute_weekly_taxes,
)
from headline_bvar_conditional import energy_level_paths_headline_bridge


# JOINT_ENERGY_TO_HEADLINE_CONTRACT_V1
CONDITIONAL_CONTRACT_VERSION = 'energy-conditional-observable-v5'
CONDITIONAL_MONTHLY_AGGREGATE_ADMISSIBILITY_V1 = True
BASELINE_REPLAY_TOLERANCE = 1e-8
AGGREGATE_REPLAY_TOLERANCE = 1e-8
PAIRING_TOLERANCE = 1e-12

WEEKLY_MODEL_IDS = {
    "car_fuels_petrol": "petrol",
    "car_fuels_diesel": "diesel",
    "liquid_fuels": "liquid_fuels",
}
WEEKLY_HICP_COLUMNS = {
    "petrol": "hicp_petrol",
    "diesel": "hicp_diesel",
    "liquid_fuels": "hicp_liquid_fuels",
}
TRANSPORT_COMPONENTS = (
    "hicp_diesel",
    "hicp_petrol",
    "hicp_other_transport_fuels",
)
TRANSPORT_ANCHOR = pd.Timestamp("2016-12-01")

AGG_COMPONENT_BY_MODEL = {
    "gas": "gas",
    "electricity": "electricity",
    "heat_energy": "heat_energy",
    "solid_fuels": "solid_fuels",
    "liquid_fuels": "liquid_fuels",
    "car_fuels_petrol": "car_fuels",
    "car_fuels_diesel": "car_fuels",
}
FINAL_PAIR_KEY = {
    "gas": "final_gas_pair_indices",
    "electricity": "final_electricity_pair_indices",
    "heat_energy": "final_heat_energy_pair_indices",
    "solid_fuels": "final_solid_fuels_pair_indices",
    "liquid_fuels": "final_liquid_fuels_pair_indices",
    "car_fuels_petrol": "final_car_fuels_pair_indices",
    "car_fuels_diesel": "final_car_fuels_pair_indices",
}
RETAINED_KEY = {
    "petrol": "petrol_retained_original_draw_indices",
    "diesel": "diesel_retained_original_draw_indices",
    "liquid_fuels": "liquid_fuels_retained_original_draw_indices",
}
TRANSPORT_PAIR_KEY = {
    "petrol": "transport_petrol_pair_indices",
    "diesel": "transport_diesel_pair_indices",
}

_INK = "#111827"
_MUTED = "#6b7280"
_GRID = "#eef0f3"
_BASE = "#64748b"
_SCEN = "#009FE3"
_IMPACT = "#7c3aed"


class ConditionalScenarioError(RuntimeError):
    """Raised when an exact paired conditional scenario cannot be reproduced."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConditionalScenarioError(f"Missing JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConditionalScenarioError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConditionalScenarioError(f"{path} does not contain a JSON object.")
    return payload


def _forecast_directory_for_model(
    aggregate_metadata: Mapping[str, Any],
    model_id: str,
) -> Path:
    """Resolve one component forecast on the current clone."""
    from inflation_path_portability import resolve_forecast_store_reference

    stores = dict(aggregate_metadata.get("component_forecast_stores", {}) or {})
    spec = model_spec(model_id)
    candidates = [str(getattr(spec, "aggregate_key", "")), str(model_id)]
    candidates += [str(x) for x in getattr(spec, "aliases", ())]
    failures: list[str] = []

    for key in candidates:
        if not key or key not in stores:
            continue
        try:
            return resolve_forecast_store_reference(stores[key], must_exist=True)
        except Exception as exc:
            failures.append(f"{key}: {exc}")

    for raw in stores.values():
        try:
            path = resolve_forecast_store_reference(raw, must_exist=True)
        except Exception:
            continue
        meta_path = path.parent.parent / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            meta = _read_json(meta_path)
        except Exception:
            continue
        if str(meta.get("model_id")) == str(model_id):
            return path

    detail = (" · " + " | ".join(failures)) if failures else ""
    raise ConditionalScenarioError(
        f"The aggregate does not contain a resolvable forecast store for "
        f"{model_id!r} on the current clone.{detail}"
    )


def _aggregate_metadata(aggregate_directory: str | Path) -> dict[str, Any]:
    directory = Path(aggregate_directory)
    path = directory / "metadata.json"
    if not path.is_file():
        path = directory / "aggregate_config.json"
    metadata = _read_json(path)
    metadata.setdefault("aggregate_run_id", directory.name)
    return metadata


def _aggregate_arrays(aggregate_directory: str | Path) -> dict[str, np.ndarray]:
    path = Path(aggregate_directory) / "aggregate_draws.npz"
    if not path.is_file():
        raise ConditionalScenarioError(f"Missing aggregate draw store: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _aggregate_dates(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> pd.DatetimeIndex:
    if "aggregate_dates" in arrays:
        return pd.DatetimeIndex(
            pd.to_datetime(np.asarray(arrays["aggregate_dates"])),
            name="date",
        )
    values = list(metadata.get("path_dates", []) or [])
    if not values:
        raise ConditionalScenarioError(
            "Aggregate store does not record aggregate_dates/path_dates."
        )
    return pd.DatetimeIndex(pd.to_datetime(values), name="date")


def _aggregate_forecast_origin(
    aggregate_metadata: Mapping[str, Any],
) -> pd.Timestamp | None:
    """Latest first-future month across portable component-store references."""
    from inflation_path_portability import resolve_forecast_store_reference

    starts: list[pd.Timestamp] = []
    for raw in dict(
        aggregate_metadata.get("component_forecast_stores", {}) or {}
    ).values():
        try:
            directory = resolve_forecast_store_reference(raw, must_exist=True)
        except Exception:
            continue
        path = directory / "forecast_metadata.json"
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
            future = list(payload.get("future_dates", []) or [])
            if not future:
                continue
            starts.append(
                pd.Timestamp(future[0]).to_period("M").to_timestamp(how="start")
            )
        except Exception:
            continue
    return max(starts) if starts else None


def _aggregate_component_names(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> list[str]:
    names = list(metadata.get("component_index_names", []) or [])
    if not names:
        names = list(metadata.get("components", []) or [])
    paths = np.asarray(arrays.get("component_index_paths"))
    if paths.ndim != 3:
        raise ConditionalScenarioError(
            "Aggregate store must contain component_index_paths with shape "
            "(draws, dates, six components). Rebuild the aggregate with the "
            "current pipeline."
        )
    if len(names) != paths.shape[2]:
        # Current production order is canonical; use it only when dimensions
        # prove that the fallback is safe.
        canonical = list(MODEL_AGGREGATE_COMPONENTS)
        if len(canonical) == paths.shape[2]:
            names = canonical
        else:
            raise ConditionalScenarioError(
                "Aggregate component names are absent/inconsistent."
            )
    return [str(x) for x in names]


def conditional_aggregate_contract(
    aggregate_directory: str | Path,
) -> dict[str, Any]:
    """Cheap contract used to populate the conditional-scenario controls."""
    directory = Path(aggregate_directory)
    metadata = _aggregate_metadata(directory)
    arrays = _aggregate_arrays(directory)

    required = {
        "component_index_paths",
        "baseline_level_paths",
        "baseline_yoy_paths",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ConditionalScenarioError(
            "This aggregate predates conditional-scenario provenance. Missing "
            f"{missing}. Rebuild the aggregate with the current pipeline."
        )

    # Pairing provenance is required for exact component substitution.
    model_ids: list[str] = []
    for model_id in CANONICAL_MODEL_IDS:
        try:
            _forecast_directory_for_model(metadata, model_id)
        except Exception:
            continue
        pair_key = FINAL_PAIR_KEY.get(model_id)
        if pair_key and pair_key in arrays:
            if model_id in WEEKLY_MODEL_IDS:
                short = WEEKLY_MODEL_IDS[model_id]
                if RETAINED_KEY[short] not in arrays:
                    continue
                if short in {"petrol", "diesel"}:
                    if (
                        TRANSPORT_PAIR_KEY[short] not in arrays
                        or "final_car_fuels_pair_indices" not in arrays
                    ):
                        continue
            model_ids.append(model_id)

    return {
        "contract_version": CONDITIONAL_CONTRACT_VERSION,
        "aggregate_run_id": str(
            metadata.get("aggregate_run_id") or directory.name
        ),
        "vintage": str(metadata.get("vintage") or ""),
        "forecast_name": str(metadata.get("forecast_name") or "unconditional"),
        "weekly_tax_mode_effective": str(
            metadata.get("weekly_tax_mode_effective") or "unknown"
        ),
        "model_ids": model_ids,
        "component_names": _aggregate_component_names(metadata, arrays),
        "n_aggregate_draws": int(
            np.asarray(arrays["component_index_paths"]).shape[0]
        ),
        "dates": [
            pd.Timestamp(x).isoformat()
            for x in _aggregate_dates(metadata, arrays)
        ],
    }


def conditional_component_contract(
    aggregate_directory: str | Path,
    *,
    model_id: str,
    project_root: str | Path,
) -> dict[str, Any]:
    """Control metadata for one model inside one exact saved aggregate."""
    aggregate_directory = Path(aggregate_directory)
    aggregate_meta = _aggregate_metadata(aggregate_directory)
    forecast_dir = _forecast_directory_for_model(aggregate_meta, model_id)
    forecast = load_energy_bvar_forecast(forecast_dir)
    run_dir = forecast_dir.parent.parent
    run_meta = _read_json(run_dir / "metadata.json")

    panel = build_panel(
        model_id,
        str(run_meta.get("vintage") or aggregate_meta.get("vintage")),
        project_root=project_root,
    )
    variables = list(forecast.get("variables") or run_meta.get("variables") or panel.variables)
    target = str(panel.target)
    if target not in variables:
        raise ConditionalScenarioError(
            f"{model_id}: target {target!r} is absent from saved forecast variables."
        )
    condition_variables = [name for name in variables if name != target]
    if not condition_variables:
        raise ConditionalScenarioError(
            f"{model_id}: no non-target endogenous variable can be conditioned."
        )

    try:
        units = dict(model_contract(
            model_id,
            str(run_meta.get("vintage") or aggregate_meta.get("vintage")),
            project_root=project_root,
        ).get("units", {}))
    except Exception:
        units = dict(panel.units)

    latest: dict[str, dict[str, Any]] = {}
    for variable in condition_variables:
        series = panel.levels[variable].astype(float).dropna()
        if series.empty:
            continue
        latest[variable] = {
            "value": float(series.iloc[-1]),
            "date": pd.Timestamp(series.index[-1]).isoformat(),
            "unit": str(units.get(variable, "")),
        }

    return {
        "contract_version": CONDITIONAL_CONTRACT_VERSION,
        "aggregate_run_id": str(
            aggregate_meta.get("aggregate_run_id") or aggregate_directory.name
        ),
        "vintage": str(run_meta.get("vintage") or aggregate_meta.get("vintage") or ""),
        "forecast_name": str(
            aggregate_meta.get("forecast_name")
            or forecast.get("forecast_name")
            or "unconditional"
        ),
        "model_id": str(model_id),
        "model_label": str(getattr(panel.spec, "label", model_id)),
        "run_id": str(run_meta.get("run_id") or run_dir.name),
        "forecast_directory": str(forecast_dir),
        "run_directory": str(run_dir),
        "frequency": str(forecast.get("frequency") or panel.frequency),
        "H": int(forecast.get("H") or len(forecast["future_dates"])),
        "future_dates": [
            pd.Timestamp(x).isoformat()
            for x in pd.DatetimeIndex(forecast["future_dates"])
        ],
        "n_forecast_draws": int(np.asarray(forecast["level_paths"]).shape[0]),
        "variables": variables,
        "target": target,
        "target_unit": str(units.get(target, "")),
        "condition_variables": condition_variables,
        "units": units,
        "latest_observed": latest,
        "affected_hicp_component": AGG_COMPONENT_BY_MODEL[model_id],
    }


def _prepare_saved_run(
    panel,
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Recreate the exact estimation panel represented by persisted draws."""
    variables = list(metadata.get("variables") or panel.variables)
    missing = [name for name in variables if name not in panel.levels.columns]
    if missing:
        raise ConditionalScenarioError(
            f"{metadata.get('model_id')}: current processed panel is missing {missing}."
        )

    levels = panel.levels.loc[:, variables].copy().astype(float)
    exog_names = list(metadata.get("exog_names") or [])
    exog = panel.exog
    if exog_names:
        if exog is None:
            raise ConditionalScenarioError(
                f"{metadata.get('model_id')}: saved run expects deterministic "
                f"columns {exog_names}."
            )
        missing_exog = [name for name in exog_names if name not in exog.columns]
        if missing_exog:
            raise ConditionalScenarioError(
                f"{metadata.get('model_id')}: deterministic block is missing "
                f"{missing_exog}."
            )
        exog = exog.loc[:, exog_names]
    elif exog is not None and exog.shape[1]:
        exog = None

    expected_hash = metadata.get("source_data_hash") or metadata.get("data_hash")
    if expected_hash and _hash_model_data is not None:
        actual_hash = _hash_model_data(levels, exog)
        if str(actual_hash) != str(expected_hash):
            raise ConditionalScenarioError(
                f"{metadata.get('model_id')}: current processed data no longer "
                "match the saved run's source-data hash."
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
        raise ConditionalScenarioError(
            f"Unsupported missing_data_method={missing_method!r}."
        )

    prep = prepare_bvar_panel(effective_levels, **kwargs)
    return prep, missing_method


def _default_outlier_grid() -> np.ndarray:
    cfg = BVARSVOPriorConfig()
    start = float(getattr(cfg, "outlier_grid_min", 2.0))
    stop = float(getattr(cfg, "outlier_grid_max", 20.0))
    step = float(getattr(cfg, "outlier_grid_step", 1.0))
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def _recover_outlier_grid(
    metadata: Mapping[str, Any],
    draws: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, str]:
    """Recover future-outlier support with a guarded legacy fallback."""
    if "outlier_support" in draws:
        grid = np.asarray(draws["outlier_support"], dtype=float).reshape(-1)
        if len(grid):
            return grid, "persisted draw store"

    prior_meta = metadata.get("prior_config")
    if isinstance(prior_meta, Mapping):
        try:
            start = float(prior_meta["outlier_grid_min"])
            stop = float(prior_meta["outlier_grid_max"])
            step = float(prior_meta["outlier_grid_step"])
            grid = np.arange(start, stop + 0.5 * step, step, dtype=float)
            if len(grid):
                return grid, "persisted prior metadata"
        except Exception:
            pass

    grid = _default_outlier_grid()
    # Current legacy production runs used the canonical integer 2..20 support.
    # Refuse the fallback if persisted sampled scales prove otherwise.
    if "outlier_scales" in draws:
        observed = np.unique(np.asarray(draws["outlier_scales"], dtype=float))
        observed = observed[np.isfinite(observed) & (observed > 1.0 + 1e-12)]
        for value in observed:
            if not np.any(np.isclose(grid, value, rtol=0.0, atol=1e-12)):
                raise ConditionalScenarioError(
                    "The saved run does not persist its outlier support and its "
                    f"sampled scale {value:g} is incompatible with the canonical "
                    "fallback. Re-save/re-estimate with current IO provenance."
                )
    return grid, "canonical 2..20 legacy fallback"


def _load_full_saved_result(
    run_directory: str | Path,
    *,
    project_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = Path(run_directory)
    metadata = _read_json(directory / "metadata.json")
    model_id = str(metadata.get("model_id") or "")
    vintage = str(metadata.get("vintage") or "")
    if not model_id or not vintage:
        raise ConditionalScenarioError(
            f"Saved run metadata in {directory} lacks model_id/vintage."
        )

    panel = build_panel(model_id, vintage, project_root=project_root)
    prep, missing_method = _prepare_saved_run(panel, metadata)

    draws_path = directory / "draws.npz"
    if not draws_path.is_file():
        raise ConditionalScenarioError(
            f"Conditional forecasts require persisted posterior draws: {draws_path}"
        )
    with np.load(draws_path, allow_pickle=False) as archive:
        draws = {name: archive[name] for name in archive.files}

    required = {
        "B",
        "A",
        "log_variance",
        "phi",
        "outlier_probabilities",
    }
    missing = sorted(required.difference(draws))
    if missing:
        raise ConditionalScenarioError(
            f"{model_id}: draws.npz is missing forecast arrays {missing}."
        )

    grid, grid_source = _recover_outlier_grid(metadata, draws)
    result: dict[str, Any] = {
        **draws,
        "metadata": metadata,
        "variables": list(metadata.get("variables") or panel.variables),
        "p": int(metadata.get("p") or panel.p),
        "frequency": str(metadata.get("frequency") or panel.frequency),
        "prep": prep,
        "prior": {"outlier_grid": grid},
        "n_draws": int(np.asarray(draws["B"]).shape[0]),
    }
    info = {
        "model_id": model_id,
        "vintage": vintage,
        "run_id": str(metadata.get("run_id") or directory.name),
        "missing_data_method": missing_method,
        "outlier_support_source": grid_source,
        "n_posterior_draws": int(result["n_draws"]),
        "panel": panel,
    }
    return result, info


def _max_abs_difference(left, right) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    valid = np.isfinite(a) & np.isfinite(b)
    mismatch_nan = np.isnan(a) ^ np.isnan(b)
    if mismatch_nan.any():
        return float("inf")
    if not valid.any():
        return 0.0
    return float(np.max(np.abs(a[valid] - b[valid])))


# ENERGY_CONDITIONAL_VECTOR_PATH_E1_V1
def conditional_path_signature_fields(
    *,
    path_value: Any = None,
    path_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    if path_values is None:
        try:
            value = float(path_value)
        except (TypeError, ValueError) as exc:
            raise ConditionalScenarioError("Scenario path_value is invalid.") from exc
        if not np.isfinite(value):
            raise ConditionalScenarioError("Scenario path_value must be finite.")
        return {"path_value": value}

    try:
        values = [float(x) for x in list(path_values)]
    except (TypeError, ValueError) as exc:
        raise ConditionalScenarioError(
            "Scenario path_values must be a finite numeric sequence."
        ) from exc
    if not values:
        raise ConditionalScenarioError("Scenario path_values cannot be empty.")
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise ConditionalScenarioError("Scenario path_values must all be finite.")

    first = float(values[0])
    if all(float(x) == first for x in values[1:]):
        return {"path_value": first}

    return {
        "path_signature_version": 2,
        "path_values": values,
    }


def _conditional_signature_item(
    model_id: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_id": str(model_id),
        "condition_variable": str(spec["condition_variable"]),
        "path_mode": str(spec["path_mode"]),
        **conditional_path_signature_fields(
            path_value=spec.get("path_value"),
            path_values=spec.get("path_values"),
        ),
        "condition_start": int(spec.get("condition_start") or 1),
        "condition_end": (
            None
            if spec.get("condition_end") is None
            else int(spec.get("condition_end"))
        ),
        "condition_mask_hash": spec.get("condition_mask_hash"),
    }


def _conditional_path_description(
    *,
    path_mode: str,
    path_fields: Mapping[str, Any],
    condition_target_level: float,
    unit: str,
) -> str:
    mode = str(path_mode).strip().lower()
    if "path_value" in path_fields:
        value = float(path_fields["path_value"])
        if mode == "percent":
            return f"{value:+.2f}% vs last observed"
        return f"{float(condition_target_level):.6g} {unit}".strip()

    values = [float(x) for x in list(path_fields.get("path_values") or [])]
    payload = json.dumps(
        dict(path_fields),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    token = __import__("hashlib").sha256(
        payload.encode("utf-8")
    ).hexdigest()[:10]
    if mode == "percent":
        return (
            f"custom % path {values[0]:+.2f}%→{values[-1]:+.2f}% "
            f"vs last observed · profile {token}"
        )
    return (
        f"custom level path {values[0]:.6g}→{values[-1]:.6g} {unit}".strip()
        + f" · profile {token}"
    )

def _build_condition_path(
    result: Mapping[str, Any],
    forecast: Mapping[str, Any],
    *,
    variable: str,
    path_mode: str,
    path_value: float | None = None,
    path_values: Sequence[float] | None = None,
    condition_start: int = 1,
    condition_end: int | None = None,
) -> tuple[np.ndarray, float, pd.Timestamp]:
    variables = list(result["variables"])
    if variable not in variables:
        raise ConditionalScenarioError(
            f"Condition variable {variable!r} is absent from {variables}."
        )

    levels_original = result["prep"].get(
        "levels_original", result["prep"]["levels"]
    )
    history = pd.DataFrame(levels_original)[variable].astype(float).dropna()
    if history.empty:
        raise ConditionalScenarioError(
            f"No observed history is available for {variable!r}."
        )
    last_value = float(history.iloc[-1])
    last_date = pd.Timestamp(history.index[-1])

    mode = str(path_mode).strip().lower()
    if mode not in {"percent", "level"}:
        raise ConditionalScenarioError("path_mode must be 'percent' or 'level'.")

    H = int(forecast.get("H") or len(forecast["future_dates"]))
    start = int(condition_start or 1)
    end = H if condition_end is None else int(condition_end)
    if start < 1 or end < start or end > H:
        raise ConditionalScenarioError(
            f"Condition window must satisfy 1 <= start <= end <= {H}; "
            f"received start={start}, end={end}."
        )

    path_fields = conditional_path_signature_fields(
        path_value=path_value,
        path_values=path_values,
    )
    path = np.full(H, np.nan, dtype=float)
    n_window = end - start + 1

    if "path_value" in path_fields:
        value = float(path_fields["path_value"])
        if mode == "percent":
            target_level = last_value * (1.0 + value / 100.0)
        else:
            target_level = value
        if not np.isfinite(target_level):
            raise ConditionalScenarioError("Imposed future level is non-finite.")
        path[start - 1:end] = target_level
        return path, last_value, last_date

    values = np.asarray(path_fields["path_values"], dtype=float)
    if values.ndim != 1 or len(values) != n_window:
        raise ConditionalScenarioError(
            "Scenario path_values length must equal the inclusive condition "
            f"window ({n_window}); received {len(values)}."
        )
    if mode == "percent":
        target_levels = last_value * (1.0 + values / 100.0)
    else:
        target_levels = values
    if not np.all(np.isfinite(target_levels)):
        raise ConditionalScenarioError(
            "Imposed future levels contain a non-finite value."
        )
    path[start - 1:end] = target_levels
    return path, last_value, last_date


def _paired_forecasts(result: Mapping[str, Any], saved_forecast: Mapping[str, Any], *, condition_variable: str, condition_path: np.ndarray, forecast_seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    """Replay saved baseline and conditional path with identical randomness."""
    n_saved = int(np.asarray(saved_forecast['level_paths']).shape[0])
    H = int(saved_forecast.get('H') or len(saved_forecast['future_dates']))
    simulate_outliers = bool(saved_forecast.get('simulate_future_outliers', True))
    future_exog = saved_forecast.get('future_exog')
    exog_names = list(saved_forecast.get('exog_names') or [])
    if exog_names and future_exog is None:
        raise ConditionalScenarioError('Saved forecast expects future deterministic regressors but does not persist future_exog.csv. Re-save the forecast with current IO.')
    kwargs = {'H': H, 'future_exog': future_exog, 'n_draws': n_saved, 'simulate_future_outliers': simulate_outliers, 'seed': int(forecast_seed)}
    baseline = forecast_bvar_sv_outlier(result, **kwargs)
    saved_indices = np.asarray(saved_forecast.get('draw_indices'), dtype=int)
    replay_indices = np.asarray(baseline.get('draw_indices'), dtype=int)
    if saved_indices.shape != replay_indices.shape or not np.array_equal(saved_indices, replay_indices):
        raise ConditionalScenarioError('The canonical forecast seed does not reproduce the saved posterior draw selection. Exact paired conditional forecasting is refused rather than comparing independent Monte Carlo samples.')
    replay_error = _max_abs_difference(baseline['level_paths'], saved_forecast['level_paths'])
    if replay_error > BASELINE_REPLAY_TOLERANCE:
        raise ConditionalScenarioError(f'Saved unconditional forecast could not be reproduced exactly from persisted draws (max |level error|={replay_error:.3e}). This usually means the forecast-generation code/seed changed.')
    conditional = forecast_bvar_sv_outlier(result, level_conditions={condition_variable: condition_path}, **kwargs, allow_partial_level_conditions=True)
    if not np.array_equal(np.asarray(baseline['draw_indices'], dtype=int), np.asarray(conditional['draw_indices'], dtype=int)):
        raise ConditionalScenarioError('Baseline and conditional posterior draw selections differ.')
    if not pd.DatetimeIndex(baseline['path_dates']).equals(pd.DatetimeIndex(conditional['path_dates'])):
        raise ConditionalScenarioError('Baseline and conditional forecast calendars differ.')
    h_error = _max_abs_difference(baseline['future_log_variance'], conditional['future_log_variance'])
    o_error = _max_abs_difference(baseline['future_outlier_scales'], conditional['future_outlier_scales'])
    if h_error > PAIRING_TOLERANCE or o_error > PAIRING_TOLERANCE:
        raise ConditionalScenarioError('Future SV/outlier draws are not paired across baseline and conditional forecasts.')
    j = list(conditional['variables']).index(condition_variable)
    finite_condition_mask = np.isfinite(condition_path)
    if not finite_condition_mask.any():
        raise ConditionalScenarioError('The conditional path contains no finite imposed period.')
    exact_condition_error = _max_abs_difference(np.asarray(conditional['future_level_paths'])[:, finite_condition_mask, j], np.repeat(np.asarray(condition_path, dtype=float)[finite_condition_mask][None, :], len(conditional['draw_indices']), axis=0))
    if exact_condition_error > 1e-08:
        raise ConditionalScenarioError(f'The imposed future level condition was not satisfied numerically: max error={exact_condition_error:.3e}.')
    return (baseline, conditional, {'baseline_replay_max_abs_error': replay_error, 'future_sv_pairing_max_abs_error': h_error, 'future_outlier_pairing_max_abs_error': o_error, 'condition_exact_max_abs_error': exact_condition_error})


def _monthly_mean_series(series: pd.Series) -> pd.Series:
    s = series.astype(float).dropna().sort_index()
    periods = pd.DatetimeIndex(s.index).to_period("M")
    out = s.groupby(periods).mean()
    out.index = out.index.to_timestamp(how="start")
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out.sort_index()


def _tax_array(payload: Mapping[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in payload:
            return np.asarray(payload[name], dtype=float)
    raise ConditionalScenarioError(
        f"Weekly tax adapter returned none of the expected keys {names}."
    )


def _weekly_pair_hicp(
    baseline_forecast: Mapping[str, Any],
    scenario_forecast: Mapping[str, Any],
    *,
    model_id: str,
    processed_dir: Path,
    indices: pd.DataFrame,
    tax_scenario: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    short = WEEKLY_MODEL_IDS[model_id]
    dataset = processed_dir / model_spec(model_id).dataset_file
    context = load_weekly_tax_context(dataset, model=short)

    baseline_taxed = reattribute_weekly_taxes(baseline_forecast, context)
    scenario_taxed = reattribute_weekly_taxes(
        scenario_forecast,
        context,
        tax_scenario=tax_scenario,
    )

    baseline_pre = _tax_array(
        baseline_taxed, "pre_tax_level_paths", "pre_tax_paths"
    )
    scenario_pre = _tax_array(
        scenario_taxed, "pre_tax_level_paths", "pre_tax_paths"
    )
    baseline_after = _tax_array(
        baseline_taxed, "post_tax_level_paths", "after_tax_paths"
    )
    scenario_after = _tax_array(
        scenario_taxed, "post_tax_level_paths", "after_tax_paths"
    )
    path_dates = pd.DatetimeIndex(
        baseline_taxed.get("path_dates", baseline_forecast["path_dates"]),
        name="date",
    )
    scenario_dates = pd.DatetimeIndex(
        scenario_taxed.get("path_dates", scenario_forecast["path_dates"]),
        name="date",
    )
    if not path_dates.equals(scenario_dates):
        raise ConditionalScenarioError(
            f"{short}: baseline/conditional weekly calendars differ."
        )

    baseline_monthly, monthly_dates, base_cov = (
        weekly_paths_with_history_to_monthly_mean(
            baseline_after,
            path_dates,
            context["data"]["after_tax"],
            return_coverage=True,
        )
    )
    scenario_monthly, scenario_monthly_dates, scen_cov = (
        weekly_paths_with_history_to_monthly_mean(
            scenario_after,
            scenario_dates,
            context["data"]["after_tax"],
            return_coverage=True,
        )
    )
    if not monthly_dates.equals(scenario_monthly_dates):
        raise ConditionalScenarioError(
            f"{short}: baseline/conditional monthly calendars differ."
        )
    if not base_cov.equals(scen_cov):
        raise ConditionalScenarioError(
            f"{short}: baseline/conditional monthly coverage differs."
        )

    filtered = filter_positive_price_draws(
        baseline_monthly,
        scenario_monthly,
        admissibility_paths={
            "baseline_pre_tax_weekly": baseline_pre,
            "scenario_pre_tax_weekly": scenario_pre,
            "baseline_after_tax_weekly": baseline_after,
            "scenario_after_tax_weekly": scenario_after,
        },
        max_rejection_rate=None,
        label=f"{short}-conditional",
    )

    historical_monthly_price = _monthly_mean_series(context["data"]["after_tax"])
    hicp_history = indices[WEEKLY_HICP_COLUMNS[short]].astype(float)

    base_rebased = rebase_price_paths_to_hicp_index(
        np.asarray(filtered["baseline_paths"], dtype=float),
        monthly_dates,
        historical_monthly_price,
        hicp_history,
    )
    scen_rebased = rebase_price_paths_to_hicp_index(
        np.asarray(filtered["scenario_paths"], dtype=float),
        monthly_dates,
        historical_monthly_price,
        hicp_history,
    )
    if not base_rebased["dates"].equals(scen_rebased["dates"]):
        raise ConditionalScenarioError(
            f"{short}: HICP proxy calendars differ after rebasing."
        )

    return {
        "short_model": short,
        "dates": pd.DatetimeIndex(base_rebased["dates"], name="date"),
        "baseline": np.asarray(base_rebased["index_paths"], dtype=float),
        "scenario": np.asarray(scen_rebased["index_paths"], dtype=float),
        "retained_original_draw_indices": np.asarray(
            filtered["draw_indices"], dtype=int
        ),
        "filter_diagnostics": dict(filtered.get("diagnostics", {})),
        "context": context,
    }


def _monthly_pair_hicp(
    baseline_forecast: Mapping[str, Any],
    scenario_forecast: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    model_id: str,
    processed_dir: Path,
    tax_scenario: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = processed_dir / model_spec(model_id).dataset_file
    if model_id == "gas":
        context = load_gas_tax_context(dataset)
        base = forecast_to_hicp_gas(baseline_forecast, result, context)
        scen = forecast_to_hicp_gas(
            scenario_forecast,
            result,
            context,
            tax_scenario=tax_scenario,
        )
    elif model_id == "electricity":
        context = load_electricity_tax_context(dataset)
        base = forecast_to_hicp_electricity(
            baseline_forecast, result, context
        )
        scen = forecast_to_hicp_electricity(
            scenario_forecast,
            result,
            context,
            tax_scenario=tax_scenario,
        )
    elif model_id in {"heat_energy", "solid_fuels"}:
        if tax_scenario:
            raise ConditionalScenarioError(
                f"{model_id}: no tax scenario is defined for this direct-HICP model."
            )
        base = forecast_hicp_component(
            baseline_forecast, result, component=model_id
        )
        scen = forecast_hicp_component(
            scenario_forecast, result, component=model_id
        )
    else:
        raise ConditionalScenarioError(
            f"No monthly HICP adapter exists for {model_id!r}."
        )

    dates = pd.DatetimeIndex(base["path_dates"], name="date")
    if not dates.equals(pd.DatetimeIndex(scen["path_dates"], name="date")):
        raise ConditionalScenarioError(
            f"{model_id}: baseline/conditional HICP calendars differ."
        )
    return {
        "dates": dates,
        "baseline": np.asarray(base["hicp_level_paths"], dtype=float),
        "scenario": np.asarray(scen["hicp_level_paths"], dtype=float),
        "retained_original_draw_indices": np.arange(
            np.asarray(base["hicp_level_paths"]).shape[0], dtype=int
        ),
        "filter_diagnostics": {
            "total_draws": int(np.asarray(base["hicp_level_paths"]).shape[0]),
            "retained_draws": int(np.asarray(base["hicp_level_paths"]).shape[0]),
            "rejected_draws": 0,
            "rejection_rate": 0.0,
        },
    }


def _align_paths(
    paths: np.ndarray,
    source_dates: Sequence[pd.Timestamp],
    target_dates: Sequence[pd.Timestamp],
) -> np.ndarray:
    values = np.asarray(paths, dtype=float)
    source = pd.DatetimeIndex(source_dates)
    target = pd.DatetimeIndex(target_dates)
    positions = source.get_indexer(target)
    if (positions < 0).any():
        missing = target[positions < 0]
        raise ConditionalScenarioError(
            "Component HICP path does not cover aggregate dates: "
            + ", ".join(pd.Timestamp(x).date().isoformat() for x in missing[:5])
        )
    return values[:, positions]


def _common_monthly_dates(
    *date_sets: Sequence[pd.Timestamp],
) -> pd.DatetimeIndex:
    starts = [pd.DatetimeIndex(x).min() for x in date_sets]
    ends = [pd.DatetimeIndex(x).max() for x in date_sets]
    start = max(starts)
    end = min(ends)
    if start > end:
        raise ConditionalScenarioError("No common monthly scenario window exists.")
    return pd.date_range(start, end, freq="MS", name="date")


def _position_lookup(indices: Sequence[int]) -> dict[int, int]:
    return {int(original): int(pos) for pos, original in enumerate(indices)}


def _component_yoy_paths(
    history: pd.Series,
    paths: np.ndarray,
    dates: Sequence[pd.Timestamp],
) -> np.ndarray:
    history = history.astype(float).sort_index()
    dates = pd.DatetimeIndex(dates, name="date")
    values = np.asarray(paths, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    date_lookup = {pd.Timestamp(date): i for i, date in enumerate(dates)}
    for t, date in enumerate(dates):
        lag = pd.Timestamp(date) - pd.DateOffset(months=12)
        if lag in date_lookup:
            lag_value = values[:, date_lookup[lag]]
        elif lag in history.index and np.isfinite(float(history.loc[lag])):
            lag_value = np.full(values.shape[0], float(history.loc[lag]))
        else:
            continue
        out[:, t] = 100.0 * (values[:, t] / lag_value - 1.0)
    return out


def _saved_weekly_baseline_hicp(
    *,
    aggregate_metadata: Mapping[str, Any],
    model_id: str,
    processed_dir: Path,
    indices: pd.DataFrame,
) -> dict[str, Any]:
    forecast_dir = _forecast_directory_for_model(aggregate_metadata, model_id)
    forecast = load_energy_bvar_forecast(forecast_dir)
    return _weekly_pair_hicp(
        forecast,
        forecast,
        model_id=model_id,
        processed_dir=processed_dir,
        indices=indices,
    )


def _conditioned_component_for_aggregate(
    *,
    model_id: str,
    pair_hicp: Mapping[str, Any],
    aggregate_metadata: Mapping[str, Any],
    aggregate_arrays: Mapping[str, np.ndarray],
    aggregate_dates: pd.DatetimeIndex,
    component_names: list[str],
    processed_dir: Path,
    model_history: Mapping[str, Any],
    indices: pd.DataFrame,
    weights: pd.DataFrame,
) -> dict[str, Any]:
    """Map conditional component draws onto the exact saved aggregate pairing."""
    stored_components = np.asarray(
        aggregate_arrays["component_index_paths"], dtype=float
    )
    n_agg = stored_components.shape[0]
    final_pair = np.asarray(
        aggregate_arrays[FINAL_PAIR_KEY[model_id]], dtype=int
    )
    if len(final_pair) != n_agg:
        raise ConditionalScenarioError(
            f"{FINAL_PAIR_KEY[model_id]} length does not match aggregate draws."
        )

    affected = AGG_COMPONENT_BY_MODEL[model_id]
    try:
        component_col = component_names.index(affected)
    except ValueError as exc:
        raise ConditionalScenarioError(
            f"Aggregate component {affected!r} is absent from {component_names}."
        ) from exc

    if model_id not in WEEKLY_MODEL_IDS:
        base_all = _align_paths(
            pair_hicp["baseline"],
            pair_hicp["dates"],
            aggregate_dates,
        )
        scen_all = _align_paths(
            pair_hicp["scenario"],
            pair_hicp["dates"],
            aggregate_dates,
        )
        if final_pair.max(initial=-1) >= len(base_all):
            raise ConditionalScenarioError(
                f"{model_id}: saved aggregate pairing exceeds the available "
                "monthly HICP draw pool."
            )

        baseline_component = base_all[final_pair]
        scenario_component = scen_all[final_pair]

        audit = _max_abs_difference(
            baseline_component,
            stored_components[:, :, component_col],
        )
        if audit > AGGREGATE_REPLAY_TOLERANCE:
            raise ConditionalScenarioError(
                f"{model_id}: mapped monthly baseline does not reproduce the "
                f"saved aggregate component (max error={audit:.3e})."
            )

        # Persisted baseline validity is a hard production contract. Do not
        # silently turn a latent baseline defect into scenario admissibility.
        if (
            np.any(~np.isfinite(baseline_component))
            or np.any(baseline_component <= 0)
        ):
            raise ConditionalScenarioError(
                f"{model_id}: mapped persisted baseline HICP component contains "
                "non-finite or non-positive values; refusing scenario filtering."
            )

        # NONFINITE has a different causal meaning from a finite non-positive
        # economic path. Preserve it as a hard diagnostic failure.
        if np.any(~np.isfinite(scenario_component)):
            raise ConditionalScenarioError(
                f"{model_id}: mapped conditional HICP component contains "
                "non-finite values; refusing admissibility filtering."
            )

        # Apply the existing complete-draw admissibility helper only AFTER the
        # monthly posterior rows have been mapped into saved aggregate-row space.
        # This preserves the saved cross-model pairing and rejects no individual
        # cell. Since baseline validity was asserted above, any rejection here is
        # caused by a finite non-positive scenario HICP path.
        filtered = filter_positive_price_draws(
            baseline_component,
            scenario_component,
            max_rejection_rate=None,
            label=f"{model_id}-conditional-aggregate-hicp",
        )
        keep = np.asarray(filtered["draw_indices"], dtype=int)
        valid_mask = np.asarray(filtered["valid_mask"], dtype=bool)
        diagnostics = dict(filtered.get("diagnostics", {}))
        diagnostics.update(
            {
                "model_id": str(model_id),
                "scope": "mapped_aggregate_component_hicp",
                "aggregate_draws_before_filter": int(n_agg),
                "aggregate_draws_after_filter": int(len(keep)),
                "rejected_aggregate_rows_0based": [
                    int(x)
                    for x in np.flatnonzero(~valid_mask).tolist()
                ],
                "baseline_validity_policy": (
                    "hard fail if mapped persisted baseline is non-finite "
                    "or non-positive"
                ),
                "scenario_nonfinite_policy": "hard fail before admissibility filter",
                "scenario_nonpositive_policy": (
                    "reject complete mapped aggregate row from both baseline "
                    "and scenario comparison"
                ),
                "distribution_interpretation": (
                    "posterior predictive conditional on the mapped conditional "
                    "HICP component path being finite and strictly positive; "
                    "persisted baseline HICP validity remains a hard contract"
                ),
            }
        )

        return {
            "keep": keep,
            "affected_component": affected,
            "baseline_component": stored_components[
                keep, :, component_col
            ],
            "scenario_component": np.asarray(
                filtered["scenario_paths"],
                dtype=float,
            ),
            "component_replay_max_abs_error": audit,
            "filter_diagnostics": diagnostics,
        }

    short = WEEKLY_MODEL_IDS[model_id]
    saved_retained = np.asarray(
        aggregate_arrays[RETAINED_KEY[short]], dtype=int
    )
    joint_retained = np.asarray(
        pair_hicp["retained_original_draw_indices"], dtype=int
    )
    joint_lookup = _position_lookup(joint_retained)

    if short == "liquid_fuels":
        if final_pair.max(initial=-1) >= len(saved_retained):
            raise ConditionalScenarioError(
                "Saved liquid-fuels final pairing points outside retained pool."
            )
        original = saved_retained[final_pair]
        keep_mask = np.array(
            [int(x) in joint_lookup for x in original], dtype=bool
        )
        keep = np.flatnonzero(keep_mask)
        if len(keep) < 1:
            raise ConditionalScenarioError(
                "Conditional liquid-fuels path leaves no aggregate draw admissible."
            )
        source_rows = np.array(
            [joint_lookup[int(x)] for x in original[keep]], dtype=int
        )
        base_all = _align_paths(
            pair_hicp["baseline"], pair_hicp["dates"], aggregate_dates
        )
        scen_all = _align_paths(
            pair_hicp["scenario"], pair_hicp["dates"], aggregate_dates
        )
        replay_component = base_all[source_rows]
        scenario_component = scen_all[source_rows]
        audit = _max_abs_difference(
            replay_component,
            stored_components[keep, :, component_col],
        )
        if audit > AGGREGATE_REPLAY_TOLERANCE:
            raise ConditionalScenarioError(
                "Liquid-fuels baseline mapping does not reproduce the saved "
                f"aggregate component (max error={audit:.3e})."
            )
        return {
            "keep": keep,
            "affected_component": affected,
            "baseline_component": stored_components[keep, :, component_col],
            "scenario_component": scenario_component,
            "component_replay_max_abs_error": audit,
        }

    # Petrol/diesel: rebuild the affected final Car-fuels component using the
    # aggregate store's exact transport pairing and the unchanged sibling path.
    sibling_short = "diesel" if short == "petrol" else "petrol"
    sibling_model_id = (
        "car_fuels_diesel"
        if sibling_short == "diesel"
        else "car_fuels_petrol"
    )
    sibling = _saved_weekly_baseline_hicp(
        aggregate_metadata=aggregate_metadata,
        model_id=sibling_model_id,
        processed_dir=processed_dir,
        indices=indices,
    )

    saved_sibling_retained = np.asarray(
        aggregate_arrays[RETAINED_KEY[sibling_short]], dtype=int
    )
    if not np.array_equal(
        np.asarray(sibling["retained_original_draw_indices"], dtype=int),
        saved_sibling_retained,
    ):
        raise ConditionalScenarioError(
            f"{sibling_short}: replayed admissibility mask does not match the "
            "saved aggregate provenance."
        )

    final_car = np.asarray(
        aggregate_arrays["final_car_fuels_pair_indices"], dtype=int
    )
    transport_condition_pair = np.asarray(
        aggregate_arrays[TRANSPORT_PAIR_KEY[short]], dtype=int
    )
    transport_sibling_pair = np.asarray(
        aggregate_arrays[TRANSPORT_PAIR_KEY[sibling_short]], dtype=int
    )
    if final_car.max(initial=-1) >= len(transport_condition_pair):
        raise ConditionalScenarioError(
            "Saved final Car-fuels pairing points outside transport pool."
        )

    cond_pool_pos = transport_condition_pair[final_car]
    sibling_pool_pos = transport_sibling_pair[final_car]
    if cond_pool_pos.max(initial=-1) >= len(saved_retained):
        raise ConditionalScenarioError(
            f"{short}: transport pairing points outside retained pool."
        )
    if sibling_pool_pos.max(initial=-1) >= len(saved_sibling_retained):
        raise ConditionalScenarioError(
            f"{sibling_short}: transport pairing points outside retained pool."
        )

    cond_original = saved_retained[cond_pool_pos]
    sibling_original = saved_sibling_retained[sibling_pool_pos]
    sibling_lookup = _position_lookup(
        sibling["retained_original_draw_indices"]
    )

    keep_mask = np.array(
        [
            int(c) in joint_lookup and int(s) in sibling_lookup
            for c, s in zip(cond_original, sibling_original)
        ],
        dtype=bool,
    )
    keep = np.flatnonzero(keep_mask)
    if len(keep) < 1:
        raise ConditionalScenarioError(
            f"Conditional {short} path leaves no Car-fuels aggregate draw admissible."
        )

    cond_rows = np.array(
        [joint_lookup[int(x)] for x in cond_original[keep]], dtype=int
    )
    sibling_rows = np.array(
        [sibling_lookup[int(x)] for x in sibling_original[keep]], dtype=int
    )

    transport_dates = _common_monthly_dates(
        pair_hicp["dates"], sibling["dates"]
    )
    cond_base = _align_paths(
        pair_hicp["baseline"], pair_hicp["dates"], transport_dates
    )[cond_rows]
    cond_scen = _align_paths(
        pair_hicp["scenario"], pair_hicp["dates"], transport_dates
    )[cond_rows]
    sibling_base = _align_paths(
        sibling["baseline"], sibling["dates"], transport_dates
    )[sibling_rows]

    other = carry_last_index_path(
        indices["hicp_other_transport_fuels"],
        transport_dates,
        n_draws=len(keep),
    )
    if short == "petrol":
        base_transport = {
            "hicp_petrol": cond_base,
            "hicp_diesel": sibling_base,
            "hicp_other_transport_fuels": other,
        }
        scen_transport = {
            "hicp_petrol": cond_scen,
            "hicp_diesel": sibling_base,
            "hicp_other_transport_fuels": other,
        }
    else:
        base_transport = {
            "hicp_petrol": sibling_base,
            "hicp_diesel": cond_base,
            "hicp_other_transport_fuels": other,
        }
        scen_transport = {
            "hicp_petrol": sibling_base,
            "hicp_diesel": cond_scen,
            "hicp_other_transport_fuels": other,
        }

    transport_history_index = pd.concat(
        [
            pd.Series(
                [100.0],
                index=pd.DatetimeIndex([TRANSPORT_ANCHOR], name="date"),
            ),
            pd.Series(model_history["transport_energy"]["index"]),
        ]
    ).sort_index()
    transport_history_index.name = "car_fuels"

    transport_history = indices[list(TRANSPORT_COMPONENTS)]
    transport_weights = weights[list(TRANSPORT_COMPONENTS)]
    base_car = aggregate_component_draw_paths_laspeyres(
        transport_history,
        base_transport,
        transport_dates,
        transport_weights,
        transport_history_index,
        components=TRANSPORT_COMPONENTS,
    )
    scen_car = aggregate_component_draw_paths_laspeyres(
        transport_history,
        scen_transport,
        transport_dates,
        transport_weights,
        transport_history_index,
        components=TRANSPORT_COMPONENTS,
    )
    replay_component = _align_paths(
        base_car["level_paths"], transport_dates, aggregate_dates
    )
    scenario_component = _align_paths(
        scen_car["level_paths"], transport_dates, aggregate_dates
    )
    audit = _max_abs_difference(
        replay_component,
        stored_components[keep, :, component_col],
    )
    if audit > AGGREGATE_REPLAY_TOLERANCE:
        raise ConditionalScenarioError(
            f"Car-fuels baseline reconstruction for {short} does not reproduce "
            f"the saved aggregate component (max error={audit:.3e})."
        )
    return {
        "keep": keep,
        "affected_component": affected,
        "baseline_component": stored_components[keep, :, component_col],
        "scenario_component": scenario_component,
        "component_replay_max_abs_error": audit,
    }


def _fan_rows(
    paths: np.ndarray,
    dates: Sequence[pd.Timestamp],
) -> list[dict[str, Any]]:
    values = np.asarray(paths, dtype=float)
    frame = summarise_draw_paths(values, dates).reset_index()
    means = np.nanmean(values, axis=0)
    rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(frame.iterrows()):
        rows.append(
            {
                "date": pd.Timestamp(row["date"]).isoformat(),
                "mean": float(means[i]),
                "q05": float(row["q05"]),
                "q16": float(row["q16"]),
                "q50": float(row["q50"]),
                "q84": float(row["q84"]),
                "q95": float(row["q95"]),
            }
        )
    return rows


def _history_rows(
    series: pd.Series,
    *,
    max_points: int = 180,
) -> list[dict[str, Any]]:
    s = series.astype(float).dropna().sort_index().iloc[-int(max_points):]
    return [
        {"date": pd.Timestamp(date).isoformat(), "value": float(value)}
        for date, value in s.items()
    ]


def _impact_terminal(paths: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(paths, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        return {"q16": None, "q50": None, "q84": None}
    terminal = values[:, -1]
    terminal = terminal[np.isfinite(terminal)]
    if len(terminal) == 0:
        return {"q16": None, "q50": None, "q84": None}
    q = np.quantile(terminal, [0.16, 0.50, 0.84])
    return {"q16": float(q[0]), "q50": float(q[1]), "q84": float(q[2])}



def _nowcast_slice(
    forecast: Mapping[str, Any],
    *,
    variable: str,
    levels_original: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return only genuinely missing ragged-edge dates for one variable."""
    variables = list(forecast["variables"])
    if variable not in variables:
        raise ConditionalScenarioError(
            f"{variable!r} is absent from forecast variables."
        )
    j = variables.index(variable)

    path_dates = pd.DatetimeIndex(forecast["path_dates"], name="date")
    tail_dates = pd.DatetimeIndex(
        forecast.get("tail_dates", []),
        name="date",
    )
    tail_length = int(forecast.get("tail_length", len(tail_dates)) or 0)
    if tail_length <= 0 or len(tail_dates) == 0:
        return pd.DatetimeIndex([], name="date"), np.empty(
            (len(forecast["level_paths"]), 0), dtype=float
        )

    tail_dates = path_dates[:tail_length]
    observed = (
        pd.DataFrame(levels_original)[variable]
        .astype(float)
        .reindex(tail_dates)
    )
    missing_mask = observed.isna().to_numpy(dtype=bool)
    if not missing_mask.any():
        return pd.DatetimeIndex([], name="date"), np.empty(
            (len(forecast["level_paths"]), 0), dtype=float
        )

    paths = np.asarray(forecast["level_paths"], dtype=float)[:, :tail_length, j]
    return tail_dates[missing_mask], paths[:, missing_mask]


def _nowcast_fan_rows(
    forecast: Mapping[str, Any],
    *,
    variable: str,
    levels_original: pd.DataFrame,
) -> list[dict[str, Any]]:
    dates, paths = _nowcast_slice(
        forecast,
        variable=variable,
        levels_original=levels_original,
    )
    if len(dates) == 0:
        return []
    return _fan_rows(paths, dates)



JOINT_ENERGY_SCENARIO_CONTRACT_VERSION = "energy-joint-scenario-v1"

_TAX_KEY_TO_MODEL_ID = {
    "gas": "gas",
    "electricity": "electricity",
    "petrol": "car_fuels_petrol",
    "diesel": "car_fuels_diesel",
    "liquid_fuels": "liquid_fuels",
    "car_fuels_petrol": "car_fuels_petrol",
    "car_fuels_diesel": "car_fuels_diesel",
}


def _normalise_joint_conditional_specs(
    conditional_specs: Sequence[Mapping[str, Any]] | None,
    *,
    vintage: str,
    aggregate_run_id: str,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in list(conditional_specs or []):
        spec = dict(raw or {})
        model_id = str(spec.get("model_id") or "")
        if not model_id:
            raise ConditionalScenarioError(
                "Joint Energy scenario contains a conditional item without model_id."
            )
        if model_id in out:
            raise ConditionalScenarioError(
                f"Joint Energy scenario contains more than one conditional item for {model_id}."
            )
        spec_vintage = str(spec.get("vintage") or vintage)
        if spec_vintage != str(vintage):
            raise ConditionalScenarioError(
                f"{model_id}: scenario vintage {spec_vintage} != aggregate vintage {vintage}."
            )
        spec_aggregate = str(spec.get("aggregate_run_id") or aggregate_run_id)
        if spec_aggregate != str(aggregate_run_id):
            raise ConditionalScenarioError(
                f"{model_id}: scenario aggregate {spec_aggregate} != selected aggregate {aggregate_run_id}."
            )

        condition_variable = str(spec.get("condition_variable") or "")
        path_mode = str(spec.get("path_mode") or "")
        if not condition_variable or not path_mode:
            raise ConditionalScenarioError(
                f"{model_id}: joint conditional recipe is incomplete."
            )

        path_fields = conditional_path_signature_fields(
            path_value=spec.get("path_value"),
            path_values=spec.get("path_values"),
        )
        normalised = {
            **spec,
            "model_id": model_id,
            "vintage": str(vintage),
            "aggregate_run_id": str(aggregate_run_id),
            "condition_variable": condition_variable,
            "path_mode": path_mode,
        }
        normalised.pop("path_value", None)
        normalised.pop("path_values", None)
        normalised.pop("path_signature_version", None)
        normalised.update(path_fields)
        out[model_id] = normalised
    return out


def _normalise_joint_tax_scenarios(
    tax_scenarios: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw_key, raw in dict(tax_scenarios or {}).items():
        model_id = _TAX_KEY_TO_MODEL_ID.get(str(raw_key))
        if model_id is None:
            raise ConditionalScenarioError(
                f"Unsupported Energy tax scenario key {raw_key!r} in joint scenario."
            )
        if model_id in out:
            raise ConditionalScenarioError(
                f"Joint Energy tax scenario duplicates model {model_id}."
            )
        scenario = dict(raw or {})
        if scenario:
            out[model_id] = scenario
    return out


def _joint_pair_hicp_for_model(*, aggregate_metadata: Mapping[str, Any], aggregate_arrays: Mapping[str, np.ndarray], aggregate_dates: pd.DatetimeIndex, component_names: list[str], processed_dir: Path, model_history: Mapping[str, Any], indices: pd.DataFrame, weights: pd.DataFrame, project_root: str | Path, model_id: str, conditional_spec: Mapping[str, Any] | None, tax_scenario: Mapping[str, Any] | None) -> dict[str, Any]:
    forecast_dir = _forecast_directory_for_model(aggregate_metadata, model_id)
    saved_forecast = load_energy_bvar_forecast(forecast_dir)
    run_dir = forecast_dir.parent.parent
    result, result_info = _load_full_saved_result(run_dir, project_root=project_root)
    panel = result_info['panel']
    target = str(panel.target)
    pairing_audit: dict[str, Any] = {}
    condition_meta: dict[str, Any] = {}

    if conditional_spec:
        condition_variable = str(conditional_spec['condition_variable'])
        if condition_variable == target:
            raise ConditionalScenarioError(
                f'{model_id}: the joint scenario cannot condition the model target itself.'
            )
        path_fields = conditional_path_signature_fields(
            path_value=conditional_spec.get('path_value'),
            path_values=conditional_spec.get('path_values'),
        )
        condition_path, last_condition, last_condition_date = _build_condition_path(
            result,
            saved_forecast,
            variable=condition_variable,
            path_mode=str(conditional_spec['path_mode']),
            path_value=path_fields.get('path_value'),
            path_values=path_fields.get('path_values'),
            condition_start=int(conditional_spec.get('condition_start') or 1),
            condition_end=conditional_spec.get('condition_end'),
        )
        forecast_seed = int(getattr(model_spec(model_id), 'forecast_seed', 123))
        baseline, scenario, pairing_audit = _paired_forecasts(
            result,
            saved_forecast,
            condition_variable=condition_variable,
            condition_path=condition_path,
            forecast_seed=forecast_seed,
        )
        condition_meta = {
            'condition_variable': condition_variable,
            'path_mode': str(conditional_spec['path_mode']),
            **path_fields,
            'last_condition_observed': float(last_condition),
            'last_condition_observed_date': pd.Timestamp(last_condition_date).isoformat(),
            'scenario_start': pd.Timestamp(pd.DatetimeIndex(baseline['future_dates'])[0]).isoformat(),
            'scenario_end': pd.Timestamp(pd.DatetimeIndex(baseline['future_dates'])[-1]).isoformat(),
        }
    else:
        baseline = saved_forecast
        scenario = saved_forecast

    if model_id in WEEKLY_MODEL_IDS:
        pair_hicp = _weekly_pair_hicp(
            baseline,
            scenario,
            model_id=model_id,
            processed_dir=processed_dir,
            indices=indices,
            tax_scenario=tax_scenario,
        )
    else:
        pair_hicp = _monthly_pair_hicp(
            baseline,
            scenario,
            result=result,
            model_id=model_id,
            processed_dir=processed_dir,
            tax_scenario=tax_scenario,
        )

    mapped = _conditioned_component_for_aggregate(
        model_id=model_id,
        pair_hicp=pair_hicp,
        aggregate_metadata=aggregate_metadata,
        aggregate_arrays=aggregate_arrays,
        aggregate_dates=aggregate_dates,
        component_names=component_names,
        processed_dir=processed_dir,
        model_history=model_history,
        indices=indices,
        weights=weights,
    )
    return {
        'model_id': model_id,
        'result_info': result_info,
        'pair_hicp': pair_hicp,
        'mapped': mapped,
        'condition_meta': condition_meta,
        'pairing_audit': pairing_audit,
        'tax_scenario_active': bool(tax_scenario),
    }


def _joint_car_fuels_for_aggregate(
    *,
    petrol_pair: Mapping[str, Any],
    diesel_pair: Mapping[str, Any],
    aggregate_arrays: Mapping[str, np.ndarray],
    aggregate_dates: pd.DatetimeIndex,
    component_names: list[str],
    model_history: Mapping[str, Any],
    indices: pd.DataFrame,
    weights: pd.DataFrame,
) -> dict[str, Any]:
    """Rebuild Car fuels when petrol and/or diesel change simultaneously.

    This is the exact two-leg analogue of the single-model mapping above. It
    uses the saved transport and final aggregate pairing arrays, intersects
    admissible original petrol/diesel draws, then chain-links the three
    transport HICP subcomponents draw by draw.
    """
    stored_components = np.asarray(
        aggregate_arrays["component_index_paths"], dtype=float
    )
    try:
        component_col = component_names.index("car_fuels")
    except ValueError as exc:
        raise ConditionalScenarioError(
            "Saved aggregate does not contain the car_fuels component."
        ) from exc

    saved_petrol = np.asarray(
        aggregate_arrays[RETAINED_KEY["petrol"]], dtype=int
    )
    saved_diesel = np.asarray(
        aggregate_arrays[RETAINED_KEY["diesel"]], dtype=int
    )
    petrol_lookup = _position_lookup(
        petrol_pair["retained_original_draw_indices"]
    )
    diesel_lookup = _position_lookup(
        diesel_pair["retained_original_draw_indices"]
    )

    final_car = np.asarray(
        aggregate_arrays["final_car_fuels_pair_indices"], dtype=int
    )
    petrol_transport = np.asarray(
        aggregate_arrays[TRANSPORT_PAIR_KEY["petrol"]], dtype=int
    )
    diesel_transport = np.asarray(
        aggregate_arrays[TRANSPORT_PAIR_KEY["diesel"]], dtype=int
    )
    if final_car.max(initial=-1) >= len(petrol_transport):
        raise ConditionalScenarioError(
            "Saved final Car-fuels pairing points outside petrol transport pool."
        )
    if final_car.max(initial=-1) >= len(diesel_transport):
        raise ConditionalScenarioError(
            "Saved final Car-fuels pairing points outside diesel transport pool."
        )

    petrol_pool = petrol_transport[final_car]
    diesel_pool = diesel_transport[final_car]
    if petrol_pool.max(initial=-1) >= len(saved_petrol):
        raise ConditionalScenarioError(
            "Saved petrol transport pairing points outside retained pool."
        )
    if diesel_pool.max(initial=-1) >= len(saved_diesel):
        raise ConditionalScenarioError(
            "Saved diesel transport pairing points outside retained pool."
        )

    petrol_original = saved_petrol[petrol_pool]
    diesel_original = saved_diesel[diesel_pool]
    keep_mask = np.array(
        [
            int(p) in petrol_lookup and int(d) in diesel_lookup
            for p, d in zip(petrol_original, diesel_original)
        ],
        dtype=bool,
    )
    keep = np.flatnonzero(keep_mask)
    if len(keep) < 1:
        raise ConditionalScenarioError(
            "Joint petrol/diesel scenario leaves no admissible Car-fuels aggregate draw."
        )

    petrol_rows = np.array(
        [petrol_lookup[int(x)] for x in petrol_original[keep]],
        dtype=int,
    )
    diesel_rows = np.array(
        [diesel_lookup[int(x)] for x in diesel_original[keep]],
        dtype=int,
    )
    transport_dates = _common_monthly_dates(
        petrol_pair["dates"],
        diesel_pair["dates"],
    )

    petrol_base = _align_paths(
        petrol_pair["baseline"],
        petrol_pair["dates"],
        transport_dates,
    )[petrol_rows]
    petrol_scenario = _align_paths(
        petrol_pair["scenario"],
        petrol_pair["dates"],
        transport_dates,
    )[petrol_rows]
    diesel_base = _align_paths(
        diesel_pair["baseline"],
        diesel_pair["dates"],
        transport_dates,
    )[diesel_rows]
    diesel_scenario = _align_paths(
        diesel_pair["scenario"],
        diesel_pair["dates"],
        transport_dates,
    )[diesel_rows]

    other = carry_last_index_path(
        indices["hicp_other_transport_fuels"],
        transport_dates,
        n_draws=len(keep),
    )
    base_transport = {
        "hicp_petrol": petrol_base,
        "hicp_diesel": diesel_base,
        "hicp_other_transport_fuels": other,
    }
    scenario_transport = {
        "hicp_petrol": petrol_scenario,
        "hicp_diesel": diesel_scenario,
        "hicp_other_transport_fuels": other,
    }

    transport_history_index = pd.concat(
        [
            pd.Series(
                [100.0],
                index=pd.DatetimeIndex([TRANSPORT_ANCHOR], name="date"),
            ),
            pd.Series(model_history["transport_energy"]["index"]),
        ]
    ).sort_index()
    transport_history_index.name = "car_fuels"
    transport_history = indices[list(TRANSPORT_COMPONENTS)]
    transport_weights = weights[list(TRANSPORT_COMPONENTS)]

    baseline_car = aggregate_component_draw_paths_laspeyres(
        transport_history,
        base_transport,
        transport_dates,
        transport_weights,
        transport_history_index,
        components=TRANSPORT_COMPONENTS,
    )
    scenario_car = aggregate_component_draw_paths_laspeyres(
        transport_history,
        scenario_transport,
        transport_dates,
        transport_weights,
        transport_history_index,
        components=TRANSPORT_COMPONENTS,
    )
    baseline_component = _align_paths(
        baseline_car["level_paths"],
        transport_dates,
        aggregate_dates,
    )
    scenario_component = _align_paths(
        scenario_car["level_paths"],
        transport_dates,
        aggregate_dates,
    )
    replay_error = _max_abs_difference(
        baseline_component,
        stored_components[keep, :, component_col],
    )
    if replay_error > AGGREGATE_REPLAY_TOLERANCE:
        raise ConditionalScenarioError(
            "Joint Car-fuels baseline reconstruction does not reproduce the "
            f"saved aggregate component (max error={replay_error:.3e})."
        )
    return {
        "keep": keep,
        "affected_component": "car_fuels",
        "baseline_component": stored_components[keep, :, component_col],
        "scenario_component": scenario_component,
        "component_replay_max_abs_error": replay_error,
    }


def _common_mapping_keep(
    mappings: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    common: set[int] | None = None
    for mapping in mappings:
        values = set(
            int(x)
            for x in np.asarray(mapping["keep"], dtype=int).tolist()
        )
        common = values if common is None else common.intersection(values)
    out = np.array(sorted(common or []), dtype=int)
    if len(out) < 1:
        raise ConditionalScenarioError(
            "The active Energy scenarios have no common admissible aggregate draw."
        )
    return out


def _mapping_rows(
    mapping: Mapping[str, Any],
    common_keep: np.ndarray,
) -> np.ndarray:
    lookup = {
        int(aggregate_pos): int(row)
        for row, aggregate_pos in enumerate(
            np.asarray(mapping["keep"], dtype=int)
        )
    }
    try:
        return np.array(
            [lookup[int(pos)] for pos in common_keep],
            dtype=int,
        )
    except KeyError as exc:
        raise ConditionalScenarioError(
            "Internal joint-scenario draw alignment failed."
        ) from exc


def compute_joint_energy_scenario(aggregate_directory: str | Path, *, project_root: str | Path, conditional_specs: Sequence[Mapping[str, Any]] | None=None, tax_scenarios: Mapping[str, Mapping[str, Any]] | None=None) -> dict[str, Any]:
    """Apply all compatible Energy scenarios simultaneously, draw by draw.

    Conditional observable paths and VAT/excise changes are combined at the
    component HICP layer first. Petrol and diesel are recombined jointly into
    Car fuels. The six final component paths are then aggregated once through
    the production Laspeyres chain-linker. Marginal Headline effects are never
    added together.
    """
    aggregate_directory = Path(aggregate_directory)
    aggregate_meta = _aggregate_metadata(aggregate_directory)
    arrays = _aggregate_arrays(aggregate_directory)
    aggregate_dates = _aggregate_dates(aggregate_meta, arrays)
    component_names = _aggregate_component_names(aggregate_meta, arrays)
    vintage = str(aggregate_meta.get('vintage') or aggregate_directory.parent.name)
    aggregate_run_id = str(aggregate_meta.get('aggregate_run_id') or aggregate_directory.name)
    forecast_name = str(aggregate_meta.get('forecast_name') or 'unconditional')
    conditional_by_model = _normalise_joint_conditional_specs(conditional_specs, vintage=vintage, aggregate_run_id=aggregate_run_id)
    for raw_conditional_spec in conditional_specs or []:
        raw_model_id = str(raw_conditional_spec.get('model_id') or '')
        if raw_model_id in conditional_by_model:
            conditional_by_model[raw_model_id]['condition_start'] = int(raw_conditional_spec.get('condition_start') or 1)
            raw_condition_end = raw_conditional_spec.get('condition_end')
            conditional_by_model[raw_model_id]['condition_end'] = None if raw_condition_end is None else int(raw_condition_end)
    tax_by_model = _normalise_joint_tax_scenarios(tax_scenarios)
    active_models = sorted(set(conditional_by_model).union(tax_by_model))
    if not active_models:
        raise ConditionalScenarioError('No active Energy scenario is available for joint application.')
    processed_dir = Path(project_root) / 'data' / 'processed' / vintage
    model_history = historical_model_component_reconstruction(processed_dir)
    inputs = load_aggregation_inputs(processed_dir)
    indices = inputs['indices']
    weights = inputs['weights']
    per_model: dict[str, dict[str, Any]] = {}
    for model_id in active_models:
        if model_id in {'car_fuels_petrol', 'car_fuels_diesel'}:
            continue
        per_model[model_id] = _joint_pair_hicp_for_model(aggregate_metadata=aggregate_meta, aggregate_arrays=arrays, aggregate_dates=aggregate_dates, component_names=component_names, processed_dir=processed_dir, model_history=model_history, indices=indices, weights=weights, project_root=project_root, model_id=model_id, conditional_spec=conditional_by_model.get(model_id), tax_scenario=tax_by_model.get(model_id))
    mappings: list[dict[str, Any]] = [dict(item['mapped']) for item in per_model.values()]
    car_active = bool(set(active_models).intersection({'car_fuels_petrol', 'car_fuels_diesel'}))
    if car_active:
        car_pairs: dict[str, dict[str, Any]] = {}
        for model_id in ('car_fuels_petrol', 'car_fuels_diesel'):
            item = _joint_pair_hicp_for_model(aggregate_metadata=aggregate_meta, aggregate_arrays=arrays, aggregate_dates=aggregate_dates, component_names=component_names, processed_dir=processed_dir, model_history=model_history, indices=indices, weights=weights, project_root=project_root, model_id=model_id, conditional_spec=conditional_by_model.get(model_id), tax_scenario=tax_by_model.get(model_id))
            car_pairs[model_id] = item
        car_mapping = _joint_car_fuels_for_aggregate(petrol_pair=car_pairs['car_fuels_petrol']['pair_hicp'], diesel_pair=car_pairs['car_fuels_diesel']['pair_hicp'], aggregate_arrays=arrays, aggregate_dates=aggregate_dates, component_names=component_names, model_history=model_history, indices=indices, weights=weights)
        mappings.append(car_mapping)
        per_model.update(car_pairs)
    affected_components = [str(mapping['affected_component']) for mapping in mappings]
    if len(affected_components) != len(set(affected_components)):
        raise ConditionalScenarioError('Joint Energy scenario produced duplicate final aggregate components; this would double-apply a scenario.')
    common_keep = _common_mapping_keep(mappings)
    stored_component_paths = np.asarray(arrays['component_index_paths'], dtype=float)
    baseline_components = {name: stored_component_paths[common_keep, :, j] for j, name in enumerate(component_names)}
    scenario_components = dict(baseline_components)
    for mapping in mappings:
        rows = _mapping_rows(mapping, common_keep)
        scenario_components[str(mapping['affected_component'])] = np.asarray(mapping['scenario_component'], dtype=float)[rows]
    baseline_energy = aggregate_draw_paths_laspeyres(model_history['component_history'], baseline_components, aggregate_dates, weights, indices['hicp_energy'])
    aggregate_history_yoy = 100.0 * (indices['hicp_energy'].astype(float) / indices['hicp_energy'].astype(float).shift(12) - 1.0)
    aggregate_forecast_origin = _aggregate_forecast_origin(aggregate_meta)
    if aggregate_forecast_origin is None:
        aggregate_forecast_origin = pd.Timestamp(aggregate_dates[0]).to_period('M').to_timestamp(how='start')
    scenario_energy = aggregate_draw_paths_laspeyres(model_history['component_history'], scenario_components, aggregate_dates, weights, indices['hicp_energy'])
    baseline_yoy = draw_yoy_and_contributions(baseline_energy, model_history['energy'])
    scenario_yoy = draw_yoy_and_contributions(scenario_energy, model_history['energy'])
    baseline_level_replay = _max_abs_difference(baseline_energy['level_paths'], np.asarray(arrays['baseline_level_paths'], dtype=float)[common_keep])
    baseline_yoy_replay = _max_abs_difference(baseline_yoy['yoy_paths'], np.asarray(arrays['baseline_yoy_paths'], dtype=float)[common_keep])
    if baseline_level_replay > AGGREGATE_REPLAY_TOLERANCE:
        raise ConditionalScenarioError(f'Joint scenario baseline does not reproduce saved HICP Energy levels (max error={baseline_level_replay:.3e}).')
    if baseline_yoy_replay > AGGREGATE_REPLAY_TOLERANCE:
        raise ConditionalScenarioError(f'Joint scenario baseline does not reproduce saved HICP Energy YoY (max error={baseline_yoy_replay:.3e}).')
    additivity = np.asarray(scenario_yoy['additivity_error'], dtype=float)
    finite = np.isfinite(additivity)
    max_additivity = float(np.max(np.abs(additivity[finite]))) if finite.any() else 0.0
    if max_additivity > 1e-10:
        raise ConditionalScenarioError(f'Joint Energy scenario contributions are not additive to numerical precision (max error={max_additivity:.3e}).')
    conditional_signature = [
        _conditional_signature_item(model_id, spec)
        for model_id, spec in sorted(conditional_by_model.items())
    ]
    tax_signature = []
    scenario_starts: dict[str, str] = {}
    for model_id, scenario in sorted(tax_by_model.items()):
        start = scenario.get('start_date')
        if start is not None:
            scenario_starts[f"Tax · {model_id.replace('_', ' ')}"] = pd.Timestamp(start).isoformat()
        tax_signature.append({'model_id': model_id, 'start_date': None if start is None else pd.Timestamp(start).isoformat(), 'vat_changed': 'vat_percent' in scenario, 'excise_changed': 'excise' in scenario})
    for model_id, item in sorted(per_model.items()):
        start = (item.get('condition_meta', {}) or {}).get('scenario_start')
        if start:
            scenario_starts['Conditional · ' + model_id.replace('_', ' ')] = str(start)
    source_labels = []
    for item in conditional_signature:
        source_labels.append('Conditional · ' + str(item['model_id']).replace('_', ' '))
    for item in tax_signature:
        source_labels.append('Tax · ' + str(item['model_id']).replace('_', ' '))
    headline_bridge = energy_level_paths_headline_bridge(dates=aggregate_dates, baseline_level_paths=np.asarray(baseline_energy['level_paths'], dtype=float), scenario_level_paths=np.asarray(scenario_energy['level_paths'], dtype=float), vintage=vintage, aggregate_run_id=aggregate_run_id, forecast_name=forecast_name, scenario_active=True, lineage={'source_kind': 'joint_energy_scenario', 'source_label': 'Joint Energy · ' + ' + '.join(source_labels), 'scenario_signature': {'conditional': conditional_signature, 'tax': tax_signature}, 'scenario_components': active_models, 'scenario_starts': scenario_starts, 'joint_contract': JOINT_ENERGY_SCENARIO_CONTRACT_VERSION, 'cross_model_dependence': aggregate_meta.get('cross_model_dependence', 'independent draw pairing')})
    aggregate_impact_yoy = np.asarray(scenario_yoy['yoy_paths'], dtype=float) - np.asarray(baseline_yoy['yoy_paths'], dtype=float)
    contribution_names = list(scenario_yoy.get('components') or baseline_yoy.get('components') or component_names)
    baseline_contributions = np.asarray(baseline_yoy['contribution_paths'], dtype=float)
    scenario_contributions = np.asarray(scenario_yoy['contribution_paths'], dtype=float)
    if baseline_contributions.shape != scenario_contributions.shape:
        raise ConditionalScenarioError('Joint scenario baseline/scenario contribution shapes differ.')
    if baseline_contributions.ndim != 3 or baseline_contributions.shape[2] != len(contribution_names):
        raise ConditionalScenarioError('Joint scenario contribution array is incompatible with component names.')
    contribution_impact = scenario_contributions - baseline_contributions
    contribution_payload = {str(name): _fan_rows(contribution_impact[:, :, j], aggregate_dates) for j, name in enumerate(contribution_names)}
    return {'ok': True, 'contract_version': JOINT_ENERGY_SCENARIO_CONTRACT_VERSION, 'meta': {'vintage': vintage, 'aggregate_run_id': aggregate_run_id, 'forecast_name': forecast_name, 'scenario_count': int(len(conditional_signature) + len(tax_signature)), 'conditional_count': int(len(conditional_signature)), 'tax_count': int(len(tax_signature)), 'active_models': active_models, 'affected_components': affected_components, 'aggregate_forecast_origin': pd.Timestamp(aggregate_forecast_origin).isoformat(), 'n_aggregate_draws_original': int(stored_component_paths.shape[0]), 'n_aggregate_draws_paired': int(len(common_keep)), 'component_price_admissibility': {str(mapping['affected_component']): dict(mapping.get('filter_diagnostics', {})) for mapping in mappings if mapping.get('filter_diagnostics')}, 'baseline_level_replay_max_abs_error': float(baseline_level_replay), 'baseline_yoy_replay_max_abs_error': float(baseline_yoy_replay), 'maximum_drawwise_contribution_additivity_error': float(max_additivity), 'conditional_signature': conditional_signature, 'tax_signature': tax_signature, 'scenario_starts': scenario_starts, 'joint_effect_interpretation': 'all active Energy scenarios are applied to component HICP paths first and HICP Energy is then aggregated once draw by draw', 'marginal_effects_summed': False}, 'headline_bridge': headline_bridge, 'aggregate_history_yoy': _history_rows(aggregate_history_yoy), 'aggregate_baseline_level': _fan_rows(baseline_energy['level_paths'], aggregate_dates), 'aggregate_scenario_level': _fan_rows(scenario_energy['level_paths'], aggregate_dates), 'aggregate_baseline_yoy': _fan_rows(baseline_yoy['yoy_paths'], aggregate_dates), 'aggregate_scenario_yoy': _fan_rows(scenario_yoy['yoy_paths'], aggregate_dates), 'aggregate_impact_yoy': _fan_rows(aggregate_impact_yoy, aggregate_dates), 'aggregate_contribution_impact_yoy': contribution_payload}



def joint_energy_contribution_impact_figure(
    payload: Mapping[str, Any] | None,
    *,
    uirevision: str = "joint-energy-contribution-impact",
) -> go.Figure:
    """Posterior-median component contribution impact for a joint Energy package."""
    blocks = dict(
        (payload or {}).get(
            "aggregate_contribution_impact_yoy"
        )
        or {}
    )
    if not blocks:
        return empty_conditional_figure(
            "Build the joint Energy scenario to display contribution impacts."
        )

    labels = {
        "car_fuels": "Car fuels",
        "liquid_fuels": "Liquid fuels",
        "gas": "Gas",
        "electricity": "Electricity",
        "heat_energy": "Heat energy",
        "solid_fuels": "Solid fuels",
    }
    fig = go.Figure()
    found = False
    for name, rows in blocks.items():
        frame = pd.DataFrame(list(rows or []))
        if frame.empty or "q50" not in frame:
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        values = pd.to_numeric(
            frame["q50"],
            errors="coerce",
        )
        if not values.notna().any():
            continue
        found = True
        fig.add_trace(
            go.Bar(
                x=frame["date"],
                y=values,
                name=labels.get(
                    str(name),
                    str(name).replace("_", " ").title(),
                ),
                hovertemplate=(
                    "%{x|%Y-%m}<br>%{y:+.3f} pp"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    if not found:
        return empty_conditional_figure(
            "Joint Energy contribution impacts are unavailable."
        )

    fig.update_layout(barmode="relative")
    fig.add_hline(
        y=0.0,
        line={"color": "#d1d5db", "width": 1},
    )
    return _layout(
        fig,
        title=(
            "HICP Energy — component contribution impact "
            "of the joint scenario"
        ),
        y_title="percentage points",
        uirevision=uirevision,
        height=430,
    )

def compute_conditional_scenario(aggregate_directory: str | Path, *, project_root: str | Path, model_id: str, condition_variable: str, path_mode: str='percent', path_value: float | None=10.0, path_values: Sequence[float] | None=None, condition_start: int=1, condition_end: int | None=None) -> dict[str, Any]:
    """Compute one exact paired conditional path and its HICP Energy effect."""
    aggregate_directory = Path(aggregate_directory)
    aggregate_meta = _aggregate_metadata(aggregate_directory)
    arrays = _aggregate_arrays(aggregate_directory)
    aggregate_dates = _aggregate_dates(aggregate_meta, arrays)
    component_names = _aggregate_component_names(aggregate_meta, arrays)
    weekly_tax_mode = str(aggregate_meta.get('weekly_tax_mode_effective') or '')
    if model_id in WEEKLY_MODEL_IDS and weekly_tax_mode not in {'strict', 'official_wob_tax_reattribution'}:
        raise ConditionalScenarioError('Weekly conditional propagation V1 requires the production WOB tax bridge. Select/rebuild a strict aggregate.')
    forecast_dir = _forecast_directory_for_model(aggregate_meta, model_id)
    saved_forecast = load_energy_bvar_forecast(forecast_dir)
    run_dir = forecast_dir.parent.parent
    result, result_info = _load_full_saved_result(run_dir, project_root=project_root)
    panel = result_info['panel']
    target = str(panel.target)
    if condition_variable == target:
        raise ConditionalScenarioError('V1 only conditions non-target observable/upstream variables. Conditioning the consumer-price target itself is intentionally hidden.')
    condition_path, last_condition, last_condition_date = _build_condition_path(result, saved_forecast, variable=condition_variable, path_mode=path_mode, path_value=path_value, path_values=path_values, condition_start=condition_start, condition_end=condition_end)

    path_signature_fields = conditional_path_signature_fields(path_value=path_value, path_values=path_values)
    forecast_seed = int(getattr(model_spec(model_id), 'forecast_seed', 123))
    baseline, conditional, pairing_audit = _paired_forecasts(result, saved_forecast, condition_variable=condition_variable, condition_path=condition_path, forecast_seed=forecast_seed)
    processed_dir = Path(project_root) / 'data' / 'processed' / str(result_info['vintage'])
    model_history = historical_model_component_reconstruction(processed_dir)
    inputs = load_aggregation_inputs(processed_dir)
    indices = inputs['indices']
    weights = inputs['weights']
    if model_id in WEEKLY_MODEL_IDS:
        pair_hicp = _weekly_pair_hicp(baseline, conditional, model_id=model_id, processed_dir=processed_dir, indices=indices)
    else:
        pair_hicp = _monthly_pair_hicp(baseline, conditional, result=result, model_id=model_id, processed_dir=processed_dir)
    mapped = _conditioned_component_for_aggregate(model_id=model_id, pair_hicp=pair_hicp, aggregate_metadata=aggregate_meta, aggregate_arrays=arrays, aggregate_dates=aggregate_dates, component_names=component_names, processed_dir=processed_dir, model_history=model_history, indices=indices, weights=weights)
    keep = np.asarray(mapped['keep'], dtype=int)
    if len(keep) < 1:
        raise ConditionalScenarioError('No paired aggregate draw remains.')
    stored_component_paths = np.asarray(arrays['component_index_paths'], dtype=float)[keep]
    baseline_components = {name: stored_component_paths[:, :, j] for j, name in enumerate(component_names)}
    scenario_components = dict(baseline_components)
    scenario_components[mapped['affected_component']] = np.asarray(mapped['scenario_component'], dtype=float)
    baseline_energy = aggregate_draw_paths_laspeyres(model_history['component_history'], baseline_components, aggregate_dates, weights, indices['hicp_energy'])
    scenario_energy = aggregate_draw_paths_laspeyres(model_history['component_history'], scenario_components, aggregate_dates, weights, indices['hicp_energy'])
    baseline_energy_yoy = draw_yoy_and_contributions(baseline_energy, model_history['energy'])
    scenario_energy_yoy = draw_yoy_and_contributions(scenario_energy, model_history['energy'])
    saved_baseline_level = np.asarray(arrays['baseline_level_paths'], dtype=float)[keep]
    saved_baseline_yoy = np.asarray(arrays['baseline_yoy_paths'], dtype=float)[keep]
    aggregate_level_replay_error = _max_abs_difference(baseline_energy['level_paths'], saved_baseline_level)
    aggregate_yoy_replay_error = _max_abs_difference(baseline_energy_yoy['yoy_paths'], saved_baseline_yoy)
    if aggregate_level_replay_error > AGGREGATE_REPLAY_TOLERANCE:
        raise ConditionalScenarioError(f'Saved component paths do not reproduce the baseline HICP Energy aggregate (max level error={aggregate_level_replay_error:.3e}).')
    if aggregate_yoy_replay_error > AGGREGATE_REPLAY_TOLERANCE:
        raise ConditionalScenarioError(f'Saved component paths do not reproduce the baseline HICP Energy YoY path (max error={aggregate_yoy_replay_error:.3e}).')
    additivity = np.asarray(scenario_energy_yoy['additivity_error'], dtype=float)
    finite_add = np.isfinite(additivity)
    max_additivity = float(np.max(np.abs(additivity[finite_add]))) if finite_add.any() else 0.0
    if max_additivity > 1e-10:
        raise ConditionalScenarioError(f'Conditional aggregate contributions are not exactly additive: {max_additivity:.3e}.')
    affected = str(mapped['affected_component'])
    base_component = np.asarray(mapped['baseline_component'], dtype=float)
    scen_component = np.asarray(mapped['scenario_component'], dtype=float)
    component_history = model_history['component_history'][affected]
    base_component_yoy = _component_yoy_paths(component_history, base_component, aggregate_dates)
    scen_component_yoy = _component_yoy_paths(component_history, scen_component, aggregate_dates)
    component_impact_yoy = scen_component_yoy - base_component_yoy
    aggregate_impact_yoy = np.asarray(scenario_energy_yoy['yoy_paths'], dtype=float) - np.asarray(baseline_energy_yoy['yoy_paths'], dtype=float)
    variables = list(baseline['variables'])
    target_j = variables.index(target)
    condition_j = variables.index(condition_variable)
    future_dates = pd.DatetimeIndex(baseline['future_dates'], name='date')
    baseline_condition = np.asarray(baseline['future_level_paths'], dtype=float)[:, :, condition_j]
    baseline_target = np.asarray(baseline['future_level_paths'], dtype=float)[:, :, target_j]
    scenario_target = np.asarray(conditional['future_level_paths'], dtype=float)[:, :, target_j]
    target_impact = scenario_target - baseline_target
    levels_original = result['prep'].get('levels_original', result['prep']['levels'])
    levels_original = pd.DataFrame(levels_original)
    condition_history = levels_original[condition_variable]
    target_history = levels_original[target]
    condition_baseline_nowcast = _nowcast_fan_rows(baseline, variable=condition_variable, levels_original=levels_original)
    condition_scenario_nowcast = _nowcast_fan_rows(conditional, variable=condition_variable, levels_original=levels_original)
    target_baseline_nowcast = _nowcast_fan_rows(baseline, variable=target, levels_original=levels_original)
    target_scenario_nowcast = _nowcast_fan_rows(conditional, variable=target, levels_original=levels_original)
    try:
        units = dict(model_contract(model_id, str(result_info['vintage']), project_root=project_root).get('units', {}))
    except Exception:
        units = dict(panel.units)
    component_terminal = _impact_terminal(component_impact_yoy)
    aggregate_terminal = _impact_terminal(aggregate_impact_yoy)
    target_terminal = _impact_terminal(target_impact)
    component_history_yoy = 100.0 * (model_history['component_history'][affected] / model_history['component_history'][affected].shift(12) - 1.0)
    aggregate_history_yoy = 100.0 * (indices['hicp_energy'].astype(float) / indices['hicp_energy'].astype(float).shift(12) - 1.0)
    aggregate_forecast_origin = _aggregate_forecast_origin(aggregate_meta)
    if aggregate_forecast_origin is None:
        aggregate_forecast_origin = pd.Timestamp(future_dates[0]).to_period('M').to_timestamp(how='start')
    finite_condition_positions = np.flatnonzero(np.isfinite(condition_path))
    if len(finite_condition_positions) < 1:
        raise ConditionalScenarioError('The conditional path contains no finite imposed period.')
    condition_start_index = int(finite_condition_positions[0])
    condition_end_index = int(finite_condition_positions[-1])
    condition_start_date = pd.Timestamp(future_dates[condition_start_index])
    condition_end_date = pd.Timestamp(future_dates[condition_end_index])
    condition_mask = np.isfinite(condition_path)
    condition_mask_hash = __import__('hashlib').sha256(condition_mask.astype(np.uint8).tobytes()).hexdigest()
    period_prefix = 'W' if str(baseline['frequency']).lower() == 'weekly' else 'M'
    condition_target_level = float(condition_path[condition_end_index])
    condition_description = _conditional_path_description(path_mode=path_mode, path_fields=path_signature_fields, condition_target_level=condition_target_level, unit=str(units.get(condition_variable, '')))
    condition_description = f'{condition_description} · {period_prefix}+{condition_start_index + 1}→{period_prefix}+{condition_end_index + 1}'
    aggregate_run_id = str(aggregate_meta.get('aggregate_run_id') or aggregate_directory.name)
    forecast_name = str(aggregate_meta.get('forecast_name') or saved_forecast.get('forecast_name') or 'unconditional')
    headline_bridge = energy_level_paths_headline_bridge(dates=aggregate_dates, baseline_level_paths=np.asarray(baseline_energy['level_paths'], dtype=float), scenario_level_paths=np.asarray(scenario_energy['level_paths'], dtype=float), vintage=str(result_info['vintage']), aggregate_run_id=aggregate_run_id, forecast_name=forecast_name, scenario_active=True, lineage={'source_kind': 'conditional_observable', 'source_label': f"Conditional · {getattr(panel.spec, 'label', model_id)} · {condition_variable.replace('_', ' ')} {condition_description}", 'scenario_signature': {'model_id': str(model_id), 'condition_variable': str(condition_variable), 'path_mode': str(path_mode), **path_signature_fields, 'condition_start': condition_start_index + 1, 'condition_end': condition_end_index + 1, 'condition_mask_hash': condition_mask_hash}, 'scenario_components': [str(model_id)], 'scenario_starts': {str(getattr(panel.spec, 'label', model_id)): condition_start_date.isoformat()}, 'model_id': str(model_id), 'model_label': str(getattr(panel.spec, 'label', model_id)), 'condition_variable': str(condition_variable), 'condition_description': condition_description})
    return {'ok': True, 'contract_version': CONDITIONAL_CONTRACT_VERSION, 'meta': {'aggregate_run_id': aggregate_run_id, 'vintage': str(result_info['vintage']), 'forecast_name': forecast_name, 'model_id': str(model_id), 'model_label': str(getattr(panel.spec, 'label', model_id)), 'run_id': str(result_info['run_id']), 'frequency': str(baseline['frequency']), 'target': target, 'target_unit': str(units.get(target, '')), 'condition_variable': str(condition_variable), 'condition_unit': str(units.get(condition_variable, '')), 'path_mode': str(path_mode), **path_signature_fields, 'condition_level': condition_target_level, 'condition_description': condition_description, 'last_condition_observed': last_condition, 'last_condition_observed_date': last_condition_date.isoformat(), 'scenario_start': condition_start_date.isoformat(), 'scenario_end': condition_end_date.isoformat(), 'aggregate_forecast_origin': pd.Timestamp(aggregate_forecast_origin).isoformat(), 'affected_hicp_component': affected, 'n_forecast_draws': int(len(baseline['draw_indices'])), 'n_aggregate_draws_original': int(np.asarray(arrays['component_index_paths']).shape[0]), 'n_aggregate_draws_paired': int(len(keep)), 'forecast_seed': forecast_seed, 'outlier_support_source': str(result_info['outlier_support_source']), 'baseline_replay_max_abs_error': float(pairing_audit['baseline_replay_max_abs_error']), 'condition_exact_max_abs_error': float(pairing_audit['condition_exact_max_abs_error']), 'future_sv_pairing_max_abs_error': float(pairing_audit['future_sv_pairing_max_abs_error']), 'future_outlier_pairing_max_abs_error': float(pairing_audit['future_outlier_pairing_max_abs_error']), 'component_replay_max_abs_error': float(mapped['component_replay_max_abs_error']), 'aggregate_level_replay_max_abs_error': float(aggregate_level_replay_error), 'aggregate_yoy_replay_max_abs_error': float(aggregate_yoy_replay_error), 'maximum_drawwise_contribution_additivity_error': float(max_additivity), 'weekly_joint_filter': dict(pair_hicp.get('filter_diagnostics', {})), 'component_price_admissibility': dict(mapped.get('filter_diagnostics', pair_hicp.get('filter_diagnostics', {}))), 'target_terminal_impact': target_terminal, 'component_terminal_yoy_impact': component_terminal, 'aggregate_terminal_yoy_impact': aggregate_terminal, 'condition_start': condition_start_index + 1, 'condition_end': condition_end_index + 1, 'condition_mask_hash': condition_mask_hash, 'condition_periods': condition_end_index - condition_start_index + 1}, 'headline_bridge': headline_bridge, 'condition_history': _history_rows(condition_history), 'condition_baseline_nowcast': condition_baseline_nowcast, 'condition_scenario_nowcast': condition_scenario_nowcast, 'condition_baseline': _fan_rows(baseline_condition, future_dates), 'condition_path': [{'date': pd.Timestamp(date).isoformat(), 'value': float(value)} for date, value in zip(future_dates, condition_path) if np.isfinite(value)], 'target_history': _history_rows(target_history), 'target_baseline_nowcast': target_baseline_nowcast, 'target_scenario_nowcast': target_scenario_nowcast, 'target_baseline': _fan_rows(baseline_target, future_dates), 'target_scenario': _fan_rows(scenario_target, future_dates), 'target_impact': _fan_rows(target_impact, future_dates), 'component_history_yoy': _history_rows(component_history_yoy), 'component_baseline_yoy': _fan_rows(base_component_yoy, aggregate_dates), 'component_scenario_yoy': _fan_rows(scen_component_yoy, aggregate_dates), 'component_impact_yoy': _fan_rows(component_impact_yoy, aggregate_dates), 'aggregate_history_yoy': _history_rows(aggregate_history_yoy), 'aggregate_baseline_level': _fan_rows(baseline_energy['level_paths'], aggregate_dates), 'aggregate_scenario_level': _fan_rows(scenario_energy['level_paths'], aggregate_dates), 'aggregate_baseline_yoy': _fan_rows(baseline_energy_yoy['yoy_paths'], aggregate_dates), 'aggregate_scenario_yoy': _fan_rows(scenario_energy_yoy['yoy_paths'], aggregate_dates), 'aggregate_impact_yoy': _fan_rows(aggregate_impact_yoy, aggregate_dates)}



def empty_conditional_set(
    *,
    vintage: str | None = None,
    aggregate_run_id: str | None = None,
) -> dict[str, Any]:
    return {
        "version": "conditional-scenario-set-v2",
        "vintage": None if vintage is None else str(vintage),
        "aggregate_run_id": (
            None if aggregate_run_id is None else str(aggregate_run_id)
        ),
        "items": {},
    }


def conditional_set_payload(
    store: Mapping[str, Any] | None,
    model_id: str | None,
) -> dict[str, Any] | None:
    if not store or not model_id:
        return None
    item = dict((store.get("items") or {}).get(str(model_id), {}) or {})
    payload = item.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else None


def conditional_set_components(
    store: Mapping[str, Any] | None,
) -> list[str]:
    if not store:
        return []
    return sorted(str(key) for key in dict(store.get("items") or {}))


def conditional_set_summary(
    store: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not store:
        return rows
    for model_id in conditional_set_components(store):
        payload = conditional_set_payload(store, model_id)
        if not payload:
            continue
        meta = dict(payload.get("meta", {}) or {})
        impact = dict(meta.get("aggregate_terminal_yoy_impact", {}) or {})
        rows.append(
            {
                "model_id": model_id,
                "label": str(meta.get("model_label") or model_id),
                "condition_variable": str(
                    meta.get("condition_variable") or ""
                ),
                "condition_description": str(
                    meta.get("condition_description") or ""
                ),
                "affected_hicp_component": str(
                    meta.get("affected_hicp_component") or ""
                ),
                "aggregate_terminal_impact": impact.get("q50"),
                "scenario_start": meta.get("scenario_start"),
                "scenario_end": meta.get("scenario_end"),
                "aggregate_run_id": meta.get("aggregate_run_id"),
            }
        )
    return rows


def conditional_set_signature(
    store: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    signature = []
    for row in conditional_set_summary(store):
        signature.append(
            {
                "model_id": row["model_id"],
                "condition_variable": row["condition_variable"],
                "condition_description": row["condition_description"],
                "aggregate_run_id": row["aggregate_run_id"],
            }
        )
    return signature


def upsert_conditional_component(
    store: Mapping[str, Any] | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    meta = dict(payload.get("meta", {}) or {})
    model_id = str(meta.get("model_id") or "")
    vintage = str(meta.get("vintage") or "")
    aggregate_run_id = str(meta.get("aggregate_run_id") or "")
    if not model_id or not vintage or not aggregate_run_id:
        raise ConditionalScenarioError(
            "Conditional payload lacks model_id/vintage/aggregate_run_id."
        )

    current = dict(store or {})
    if (
        str(current.get("vintage") or "") != vintage
        or str(current.get("aggregate_run_id") or "") != aggregate_run_id
    ):
        current = empty_conditional_set(
            vintage=vintage,
            aggregate_run_id=aggregate_run_id,
        )
    else:
        current = {
            **current,
            "items": dict(current.get("items") or {}),
        }

    current["items"][model_id] = {
        "payload": dict(payload),
    }
    return current


def remove_conditional_component(
    store: Mapping[str, Any] | None,
    model_id: str | None,
) -> dict[str, Any]:
    current = dict(store or {})
    current["items"] = dict(current.get("items") or {})
    if model_id:
        current["items"].pop(str(model_id), None)
    return current


def clear_conditional_set(
    *,
    vintage: str | None = None,
    aggregate_run_id: str | None = None,
) -> dict[str, Any]:
    return empty_conditional_set(
        vintage=vintage,
        aggregate_run_id=aggregate_run_id,
    )


def scenario_marginal_impact_figure(
    conditional_store: Mapping[str, Any] | None,
    tax_records: Sequence[Mapping[str, Any]] | None = None,
    *,
    fan_mode: str = "68",
    uirevision: str = "all-scenario-marginals",
) -> go.Figure:
    """Compare all active scenario *marginal* HICP Energy impacts.

    Conditional impacts come directly from their exact paired component
    scenarios. Tax impacts are supplied as marginal aggregate records computed
    by the dashboard with one tax scenario at a time.

    This chart deliberately does not add marginal effects together.
    """
    fig = go.Figure()
    count = 0

    for row in conditional_set_summary(conditional_store):
        payload = conditional_set_payload(
            conditional_store, row["model_id"]
        )
        frame = _frame(payload, "aggregate_impact_yoy")
        if frame.empty:
            continue
        label = (
            "Conditional · "
            + row["label"]
            + " · "
            + row["condition_variable"].replace("_", " ")
        )
        if fan_mode in {"90", "both"}:
            _band(
                fig, frame, "q05", "q95", f"{label} 90%",
                color=_SCEN, alpha=0.06,
                legendgroup=f"conditional-{row['model_id']}",
            )
        if fan_mode in {"68", "both"}:
            _band(
                fig, frame, "q16", "q84", f"{label} 68%",
                color=_SCEN, alpha=0.10,
                legendgroup=f"conditional-{row['model_id']}",
            )
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["q50"],
                mode="lines",
                name=label,
                line={"width": 2.2},
                hovertemplate="%{y:+.3f} pp<extra>" + label + "</extra>",
                legendgroup=f"conditional-{row['model_id']}",
            )
        )
        count += 1

    for record in list(tax_records or []):
        frame = pd.DataFrame(list(record.get("impact", []) or []))
        if frame.empty:
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        label = str(record.get("label") or "Tax scenario")
        if fan_mode in {"90", "both"} and {"q05", "q95"}.issubset(frame.columns):
            _band(
                fig, frame, "q05", "q95", f"{label} 90%",
                color="#f59e0b", alpha=0.05,
                legendgroup=f"tax-{record.get('model_id','tax')}",
            )
        if fan_mode in {"68", "both"} and {"q16", "q84"}.issubset(frame.columns):
            _band(
                fig, frame, "q16", "q84", f"{label} 68%",
                color="#f59e0b", alpha=0.10,
                legendgroup=f"tax-{record.get('model_id','tax')}",
            )
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["q50"],
                mode="lines",
                name=label,
                line={"width": 2.2, "dash": "dot"},
                hovertemplate="%{y:+.3f} pp<extra>" + label + "</extra>",
                legendgroup=f"tax-{record.get('model_id','tax')}",
            )
        )
        count += 1

    if count == 0:
        return empty_conditional_figure(
            "No active conditional or tax scenario is available."
        )

    fig.add_hline(y=0.0, line={"color": "#d1d5db", "width": 1})
    return _layout(
        fig,
        title="Marginal impact of all active scenarios on HICP Energy",
        y_title="percentage points",
        uirevision=uirevision,
        height=500,
    )


def conditional_kpis(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not payload or not payload.get("ok"):
        return {
            "condition_level": None,
            "component_impact": None,
            "aggregate_impact": None,
            "n_draws": None,
        }
    meta = dict(payload.get("meta", {}) or {})
    comp = dict(meta.get("component_terminal_yoy_impact", {}) or {})
    agg = dict(meta.get("aggregate_terminal_yoy_impact", {}) or {})
    return {
        "condition_level": meta.get("condition_level"),
        "condition_unit": meta.get("condition_unit"),
        "condition_description": meta.get("condition_description"),
        "component_impact": comp.get("q50"),
        "component_low": comp.get("q16"),
        "component_high": comp.get("q84"),
        "aggregate_impact": agg.get("q50"),
        "aggregate_low": agg.get("q16"),
        "aggregate_high": agg.get("q84"),
        "terminal_date": meta.get("scenario_end"),
        "n_draws": meta.get("n_aggregate_draws_paired"),
        "n_draws_original": meta.get("n_aggregate_draws_original"),
        "affected_component": meta.get("affected_hicp_component"),
    }


def _frame(payload: Mapping[str, Any] | None, key: str) -> pd.DataFrame:
    frame = pd.DataFrame(list((payload or {}).get(key, []) or []))
    if not frame.empty and "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _band(
    fig: go.Figure,
    frame: pd.DataFrame,
    lower: str,
    upper: str,
    name: str,
    *,
    color: str,
    alpha: float,
    legendgroup: str,
) -> None:
    if frame.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame[upper],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=legendgroup,
        )
    )
    # Convert #RRGGBB to rgba for fill if possible.
    if color.startswith("#") and len(color) == 7:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill = f"rgba({r},{g},{b},{alpha})"
    else:
        fill = f"rgba(100,116,139,{alpha})"
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame[lower],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=fill,
            name=name,
            hoverinfo="skip",
            legendgroup=legendgroup,
        )
    )


# GRAPH_EXPORT_READABILITY_G4_ENERGY_CONDITIONAL_V1
def _layout(
    fig: go.Figure,
    *,
    title: str,
    y_title: str,
    uirevision: str,
    height: int = 500,
) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        margin={"l": 64, "r": 26, "t": 66, "b": 54},
        height=height,
        title={
            "text": title,
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 18, "color": _INK},
        },
        font={
            "family": "Inter, Segoe UI, sans-serif",
            "color": "#374151",
            "size": 12,
        },
        yaxis_title=y_title,
        hovermode="x unified",
        dragmode="pan",
        legend={
            "orientation": "h",
            "y": 1.08,
            "x": 1,
            "xanchor": "right",
            "font": {"size": 11},
        },
        uirevision=uirevision,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, linecolor="#e5e7eb")
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor="#d1d5db")
    return fig


def empty_conditional_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font={"size": 13, "color": _MUTED},
    )
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin={"l": 54, "r": 24, "t": 54, "b": 42},
        xaxis={"visible": False},
        yaxis={"visible": False},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _fan_mode_bands(
    fig: go.Figure,
    frame: pd.DataFrame,
    *,
    fan_mode: str,
    color: str,
    prefix: str,
    legendgroup: str,
) -> None:
    if fan_mode in {"90", "both"}:
        _band(
            fig,
            frame,
            "q05",
            "q95",
            f"{prefix} 90%",
            color=color,
            alpha=0.10,
            legendgroup=legendgroup,
        )
    if fan_mode in {"68", "both"}:
        _band(
            fig,
            frame,
            "q16",
            "q84",
            f"{prefix} 68%",
            color=color,
            alpha=0.20,
            legendgroup=legendgroup,
        )


def conditional_path_figure(
    payload: Mapping[str, Any] | None,
    *,
    fan_mode: str = "68",
    uirevision: str = "conditional-path",
) -> go.Figure:
    if not payload or not payload.get("ok"):
        return empty_conditional_figure(
            "Compute a conditional observable path to display the assumption."
        )
    meta = dict(payload.get("meta", {}) or {})
    history = _frame(payload, "condition_history")
    base_now = _frame(payload, "condition_baseline_nowcast")
    scen_now = _frame(payload, "condition_scenario_nowcast")
    baseline = _frame(payload, "condition_baseline")
    imposed = _frame(payload, "condition_path")
    fig = go.Figure()

    if not history.empty:
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["value"],
                mode="lines",
                name="Observed",
                line={"color": _INK, "width": 1.8},
            )
        )

    if not base_now.empty:
        _fan_mode_bands(
            fig, base_now, fan_mode=fan_mode,
            color="#d4a24c", prefix="Unconditional nowcast",
            legendgroup="nowcast-base",
        )
        # ENERGY_CONDITIONAL_SAFE_JOIN_V1_2
        base_now_dates = base_now["date"]
        base_now_values = base_now["q50"]
        if not history.empty:
            last_observed_date = pd.Timestamp(history["date"].iloc[-1])
            first_model_date = pd.Timestamp(base_now["date"].iloc[0])
            frequency = str(meta.get("frequency") or "").strip().lower()

            if frequency == "weekly":
                contiguous = (
                    first_model_date - last_observed_date
                    == pd.Timedelta(days=7)
                )
            else:
                last_observed_month = (
                    last_observed_date
                    .to_period("M")
                    .to_timestamp(how="start")
                )
                first_model_month = (
                    first_model_date
                    .to_period("M")
                    .to_timestamp(how="start")
                )
                contiguous = (
                    first_model_month
                    == last_observed_month + pd.offsets.MonthBegin(1)
                )

            if contiguous:
                base_now_dates = pd.DatetimeIndex(
                    [last_observed_date, *list(base_now["date"])]
                )
                base_now_values = np.r_[
                    float(history["value"].iloc[-1]),
                    base_now["q50"].to_numpy(dtype=float),
                ]

        fig.add_trace(
            go.Scatter(
                x=base_now_dates,
                y=base_now_values,
                mode="lines+markers",
                name="Unconditional nowcast median",
                line={"color": "#a66b00", "width": 2.2},
                marker={"size": 5},
            )
        )
    if not scen_now.empty:
        # Only draw a separate conditional nowcast when conditioning changes it.
        merged = base_now[["date", "q50"]].merge(
            scen_now[["date", "q50"]],
            on="date",
            suffixes=("_base", "_scen"),
            how="inner",
        )
        materially_diff = (
            not merged.empty
            and np.nanmax(
                np.abs(
                    merged["q50_scen"].to_numpy(dtype=float)
                    - merged["q50_base"].to_numpy(dtype=float)
                )
            ) > 1e-10
        )
        if materially_diff:
            fig.add_trace(
                go.Scatter(
                    x=scen_now["date"], y=scen_now["q50"],
                    mode="lines+markers",
                    name="Conditional nowcast median",
                    line={"color": _SCEN, "width": 2.0, "dash": "dash"},
                    marker={"size": 5},
                )
            )

    _fan_mode_bands(
        fig,
        baseline,
        fan_mode=fan_mode,
        color=_BASE,
        prefix="Unconditional forecast",
        legendgroup="baseline",
    )
    if not baseline.empty:
        baseline_dates = baseline["date"]
        baseline_values = baseline["q50"]

        # If a real nowcast exists, it is the first model segment and owns
        # the observed/model join. Never bridge directly across that nowcast.
        if base_now.empty and not history.empty:
            last_observed_date = pd.Timestamp(history["date"].iloc[-1])
            first_model_date = pd.Timestamp(baseline["date"].iloc[0])
            frequency = str(meta.get("frequency") or "").strip().lower()

            if frequency == "weekly":
                contiguous = (
                    first_model_date - last_observed_date
                    == pd.Timedelta(days=7)
                )
            else:
                last_observed_month = (
                    last_observed_date
                    .to_period("M")
                    .to_timestamp(how="start")
                )
                first_model_month = (
                    first_model_date
                    .to_period("M")
                    .to_timestamp(how="start")
                )
                contiguous = (
                    first_model_month
                    == last_observed_month + pd.offsets.MonthBegin(1)
                )

            if contiguous:
                baseline_dates = pd.DatetimeIndex(
                    [last_observed_date, *list(baseline["date"])]
                )
                baseline_values = np.r_[
                    float(history["value"].iloc[-1]),
                    baseline["q50"].to_numpy(dtype=float),
                ]

        fig.add_trace(
            go.Scatter(
                x=baseline_dates,
                y=baseline_values,
                mode="lines",
                name="Unconditional forecast median",
                line={"color": _BASE, "width": 2},
            )
        )
    if not imposed.empty:
        fig.add_trace(
            go.Scatter(
                x=imposed["date"],
                y=imposed["value"],
                mode="lines+markers",
                name="Imposed future path",
                line={"color": _SCEN, "width": 2.6, "dash": "dash"},
                marker={"size": 5},
            )
        )
    unit = str(meta.get("condition_unit") or "")
    return _layout(
        fig,
        title=(
            "Conditioning assumption — "
            + str(meta.get("condition_variable", "")).replace("_", " ").title()
        ),
        y_title=unit or "Level",
        uirevision=uirevision,
        height=490,
    )


def conditional_target_figure(
    payload: Mapping[str, Any] | None,
    *,
    fan_mode: str = "68",
    uirevision: str = "conditional-target",
) -> go.Figure:
    if not payload or not payload.get("ok"):
        return empty_conditional_figure(
            "Compute a conditional scenario to display the model target."
        )
    meta = dict(payload.get("meta", {}) or {})
    history = _frame(payload, "target_history")
    base_now = _frame(payload, "target_baseline_nowcast")
    scen_now = _frame(payload, "target_scenario_nowcast")
    baseline = _frame(payload, "target_baseline")
    scenario = _frame(payload, "target_scenario")
    fig = go.Figure()

    if not history.empty:
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["value"],
                mode="lines",
                name="Observed target",
                line={"color": _INK, "width": 1.7},
            )
        )

    if not base_now.empty:
        _fan_mode_bands(
            fig, base_now, fan_mode=fan_mode,
            color="#d4a24c", prefix="Unconditional nowcast",
            legendgroup="nowcast-base",
        )
        fig.add_trace(
            go.Scatter(
                x=base_now["date"], y=base_now["q50"],
                mode="lines+markers",
                name="Unconditional nowcast median",
                line={"color": "#a66b00", "width": 2.2},
                marker={"size": 5},
            )
        )

    if not scen_now.empty:
        merged = base_now[["date", "q50"]].merge(
            scen_now[["date", "q50"]],
            on="date",
            suffixes=("_base", "_scen"),
            how="inner",
        )
        materially_diff = (
            not merged.empty
            and np.nanmax(
                np.abs(
                    merged["q50_scen"].to_numpy(dtype=float)
                    - merged["q50_base"].to_numpy(dtype=float)
                )
            ) > 1e-10
        )
        if materially_diff:
            fig.add_trace(
                go.Scatter(
                    x=scen_now["date"], y=scen_now["q50"],
                    mode="lines+markers",
                    name="Conditional nowcast median",
                    line={"color": _SCEN, "width": 2.0, "dash": "dash"},
                    marker={"size": 5},
                )
            )

    _fan_mode_bands(
        fig, baseline, fan_mode=fan_mode, color=_BASE,
        prefix="Unconditional forecast", legendgroup="baseline",
    )
    _fan_mode_bands(
        fig, scenario, fan_mode=fan_mode, color=_SCEN,
        prefix="Conditional forecast", legendgroup="scenario",
    )
    if not baseline.empty:
        fig.add_trace(
            go.Scatter(
                x=baseline["date"], y=baseline["q50"],
                mode="lines", name="Unconditional forecast median",
                line={"color": _BASE, "width": 2},
            )
        )
    if not scenario.empty:
        fig.add_trace(
            go.Scatter(
                x=scenario["date"], y=scenario["q50"],
                mode="lines", name="Conditional forecast median",
                line={"color": _SCEN, "width": 2.4, "dash": "dash"},
            )
        )
    return _layout(
        fig,
        title=(
            "BVAR target — "
            + str(meta.get("target", "")).replace("_", " ").title()
        ),
        y_title=str(meta.get("target_unit") or "Level"),
        uirevision=uirevision,
        height=490,
    )


def conditional_component_figure(
    payload: Mapping[str, Any] | None,
    *,
    fan_mode: str = "68",
    uirevision: str = "conditional-component",
) -> go.Figure:
    if not payload or not payload.get("ok"):
        return empty_conditional_figure(
            "Compute a conditional scenario to display the HICP component."
        )
    meta = dict(payload.get("meta", {}) or {})
    history = _frame(payload, "component_history_yoy")
    baseline = _frame(payload, "component_baseline_yoy")
    scenario = _frame(payload, "component_scenario_yoy")
    origin = pd.Timestamp(meta.get("aggregate_forecast_origin"))

    base_now = baseline.loc[baseline["date"] < origin].copy()
    scen_now = scenario.loc[scenario["date"] < origin].copy()
    base_future = baseline.loc[baseline["date"] >= origin].copy()
    scen_future = scenario.loc[scenario["date"] >= origin].copy()

    fig = go.Figure()
    if not history.empty:
        hist = history.loc[history["date"] < origin].copy()
        if not hist.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist["date"], y=hist["value"],
                    mode="lines", name="Observed",
                    line={"color": _INK, "width": 1.8},
                )
            )

    if not base_now.empty:
        _fan_mode_bands(
            fig, base_now, fan_mode=fan_mode,
            color="#d4a24c", prefix="Baseline nowcast",
            legendgroup="nowcast-base",
        )
        fig.add_trace(
            go.Scatter(
                x=base_now["date"], y=base_now["q50"],
                mode="lines+markers", name="Baseline nowcast median",
                line={"color": "#a66b00", "width": 2.2},
                marker={"size": 5},
            )
        )
    if not scen_now.empty:
        merged = base_now[["date", "q50"]].merge(
            scen_now[["date", "q50"]], on="date",
            suffixes=("_base", "_scen"), how="inner",
        )
        if (
            not merged.empty
            and np.nanmax(
                np.abs(
                    merged["q50_scen"].to_numpy(dtype=float)
                    - merged["q50_base"].to_numpy(dtype=float)
                )
            ) > 1e-10
        ):
            fig.add_trace(
                go.Scatter(
                    x=scen_now["date"], y=scen_now["q50"],
                    mode="lines+markers", name="Conditional nowcast median",
                    line={"color": _SCEN, "width": 2.0, "dash": "dash"},
                    marker={"size": 5},
                )
            )

    _fan_mode_bands(
        fig, base_future, fan_mode=fan_mode, color=_BASE,
        prefix="Baseline forecast", legendgroup="baseline",
    )
    _fan_mode_bands(
        fig, scen_future, fan_mode=fan_mode, color=_SCEN,
        prefix="Conditional forecast", legendgroup="scenario",
    )
    if not base_future.empty:
        fig.add_trace(
            go.Scatter(
                x=base_future["date"], y=base_future["q50"],
                mode="lines+markers", name="Baseline forecast median",
                line={"color": _BASE, "width": 2},
                marker={"size": 4},
            )
        )
    if not scen_future.empty:
        fig.add_trace(
            go.Scatter(
                x=scen_future["date"], y=scen_future["q50"],
                mode="lines+markers", name="Conditional forecast median",
                line={"color": _SCEN, "width": 2.4, "dash": "dash"},
                marker={"size": 4},
            )
        )
    fig.add_vline(
        x=pd.Timestamp(origin).isoformat(),
        line_width=1,
        line_dash="dash",
        line_color="#94a3b8",
    )
    label = str(meta.get("affected_hicp_component", "")).replace(
        "_", " "
    ).title()
    return _layout(
        fig,
        title=f"{label} HICP inflation — observed, nowcast and scenario",
        y_title="% y/y",
        uirevision=uirevision,
        height=470,
    )


def conditional_aggregate_figure(
    payload: Mapping[str, Any] | None,
    *,
    fan_mode: str = "68",
    uirevision: str = "conditional-aggregate",
) -> go.Figure:
    if not payload or not payload.get("ok"):
        return empty_conditional_figure(
            "Compute a conditional scenario to display HICP Energy."
        )
    meta = dict(payload.get("meta", {}) or {})
    history = _frame(payload, "aggregate_history_yoy")
    baseline = _frame(payload, "aggregate_baseline_yoy")
    scenario = _frame(payload, "aggregate_scenario_yoy")
    origin = pd.Timestamp(meta.get("aggregate_forecast_origin"))

    base_now = baseline.loc[baseline["date"] < origin].copy()
    scen_now = scenario.loc[scenario["date"] < origin].copy()
    base_future = baseline.loc[baseline["date"] >= origin].copy()
    scen_future = scenario.loc[scenario["date"] >= origin].copy()

    fig = go.Figure()
    if not history.empty:
        hist = history.loc[history["date"] < origin].copy()
        if not hist.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist["date"], y=hist["value"],
                    mode="lines", name="Observed HICP Energy",
                    line={"color": _INK, "width": 1.8},
                )
            )
    if not base_now.empty:
        _fan_mode_bands(
            fig, base_now, fan_mode=fan_mode,
            color="#d4a24c", prefix="Baseline nowcast",
            legendgroup="nowcast-base",
        )
        fig.add_trace(
            go.Scatter(
                x=base_now["date"], y=base_now["q50"],
                mode="lines+markers", name="Baseline nowcast median",
                line={"color": "#a66b00", "width": 2.2},
                marker={"size": 5},
            )
        )
    if not scen_now.empty:
        merged = base_now[["date", "q50"]].merge(
            scen_now[["date", "q50"]], on="date",
            suffixes=("_base", "_scen"), how="inner",
        )
        if (
            not merged.empty
            and np.nanmax(
                np.abs(
                    merged["q50_scen"].to_numpy(dtype=float)
                    - merged["q50_base"].to_numpy(dtype=float)
                )
            ) > 1e-10
        ):
            fig.add_trace(
                go.Scatter(
                    x=scen_now["date"], y=scen_now["q50"],
                    mode="lines+markers", name="Conditional nowcast median",
                    line={"color": _SCEN, "width": 2.0, "dash": "dash"},
                    marker={"size": 5},
                )
            )
    _fan_mode_bands(
        fig, base_future, fan_mode=fan_mode, color=_BASE,
        prefix="Baseline forecast", legendgroup="baseline",
    )
    _fan_mode_bands(
        fig, scen_future, fan_mode=fan_mode, color=_SCEN,
        prefix="Conditional forecast", legendgroup="scenario",
    )
    if not base_future.empty:
        fig.add_trace(
            go.Scatter(
                x=base_future["date"], y=base_future["q50"],
                mode="lines+markers", name="Baseline HICP Energy",
                line={"color": _BASE, "width": 2},
                marker={"size": 4},
            )
        )
    if not scen_future.empty:
        fig.add_trace(
            go.Scatter(
                x=scen_future["date"], y=scen_future["q50"],
                mode="lines+markers", name="Conditional HICP Energy",
                line={"color": _SCEN, "width": 2.5, "dash": "dash"},
                marker={"size": 4},
            )
        )
    fig.add_vline(
        x=pd.Timestamp(origin).isoformat(),
        line_width=1,
        line_dash="dash",
        line_color="#94a3b8",
    )
    return _layout(
        fig,
        title="HICP Energy inflation — observed, nowcast and conditional path",
        y_title="% y/y",
        uirevision=uirevision,
        height=480,
    )


def conditional_aggregate_impact_figure(
    payload: Mapping[str, Any] | None,
    *,
    fan_mode: str = "68",
    uirevision: str = "conditional-aggregate-impact",
) -> go.Figure:
    if not payload or not payload.get("ok"):
        return empty_conditional_figure(
            "Compute a conditional scenario to display aggregate impact."
        )
    impact = _frame(payload, "aggregate_impact_yoy")
    fig = go.Figure()
    _fan_mode_bands(
        fig, impact, fan_mode=fan_mode, color=_IMPACT,
        prefix="Impact", legendgroup="impact",
    )
    if not impact.empty:
        fig.add_trace(
            go.Scatter(
                x=impact["date"], y=impact["q50"],
                mode="lines+markers", name="Median impact",
                line={"color": _IMPACT, "width": 2.5},
                marker={"size": 5},
                hovertemplate="%{y:+.3f} pp<extra>HICP Energy impact</extra>",
            )
        )
    fig.add_hline(y=0.0, line={"color": "#d1d5db", "width": 1})
    return _layout(
        fig,
        title="HICP Energy impact of the conditional path",
        y_title="percentage points",
        uirevision=uirevision,
        height=430,
    )


__all__ = [
    "CONDITIONAL_CONTRACT_VERSION",
    "ConditionalScenarioError",
    "conditional_aggregate_contract",
    "conditional_component_contract",
    "compute_conditional_scenario",
    "compute_joint_energy_scenario",
    "joint_energy_contribution_impact_figure",
    "empty_conditional_set",
    "conditional_set_payload",
    "conditional_set_components",
    "conditional_set_summary",
    "conditional_set_signature",
    "upsert_conditional_component",
    "remove_conditional_component",
    "clear_conditional_set",
    "scenario_marginal_impact_figure",
    "conditional_kpis",
    "conditional_path_figure",
    "conditional_target_figure",
    "conditional_component_figure",
    "conditional_aggregate_figure",
    "conditional_aggregate_impact_figure",
    "empty_conditional_figure",
]
