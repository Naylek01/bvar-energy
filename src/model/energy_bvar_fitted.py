"""Posterior in-sample fitted HICP paths for the Energy BVAR suite.

This module is deliberately separate from the dashboard.  It turns retained
posterior coefficient draws into one-step in-sample conditional-mean paths,
then sends those paths through exactly the same component HICP bridges used by
the forecast layer.  Finally it combines the seven BVARs into the six HICP
Energy blocks and applies the same annual-weight / December chain-linked
Laspeyres aggregation used by the production aggregate.

Definition of "fitted"
-----------------------
For each retained posterior draw d and regression date t,

    fitted_delta_y[d,t] = X[d,t] @ B[d]

where X contains the realised (or, under DK, draw-specific completed) lagged
states and deterministic regressors used by the likelihood.  The fitted level
is the one-step conditional mean

    fitted_level[d,t] = conditioning_level[d,t-1] + fitted_delta_y[d,t].

Innovations are therefore set to zero; this is an in-sample posterior fitted
value, not a recursively simulated counterfactual forecast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


FITTED_SCHEMA_VERSION = "1.0"
FITTED_COMPONENT_FILENAME = "fitted_hicp_draws.npz"
FITTED_COMPONENT_METADATA = "fitted_hicp_metadata.json"
FITTED_AGGREGATE_FILENAME = "bvar_fitted_v1.npz"
FITTED_AGGREGATE_METADATA = "bvar_fitted_v1_metadata.json"

MODEL_IDS: tuple[str, ...] = (
    "gas",
    "electricity",
    "heat_energy",
    "solid_fuels",
    "car_fuels_petrol",
    "car_fuels_diesel",
    "liquid_fuels",
)
SIX_COMPONENTS: tuple[str, ...] = (
    "car_fuels",
    "liquid_fuels",
    "gas",
    "electricity",
    "heat_energy",
    "solid_fuels",
)
TRANSPORT_COMPONENTS: tuple[str, ...] = (
    "hicp_diesel",
    "hicp_petrol",
    "hicp_other_transport_fuels",
)
MIN_AGGREGATION_DATE = pd.Timestamp("2017-01-01")


class FittedReconstructionError(RuntimeError):
    """Base error for fitted reconstruction failures."""


class FittedUnavailableError(FittedReconstructionError):
    """Raised when the saved run does not contain posterior fit information."""


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_directory_from_forecast_store(path_value: object) -> Path:
    path = Path(str(path_value))
    # .../<run_id>/forecasts/<forecast_name>
    if path.parent.name != "forecasts":
        raise FittedReconstructionError(
            f"Forecast store does not follow <run>/forecasts/<name>: {path}"
        )
    return path.parent.parent


def _run_id_from_forecast_store(path_value: object) -> str | None:
    """Extract the run id from ``.../<run_id>/forecasts/<forecast_name>``.

    Paths saved by older notebook aggregates may be absolute paths from another
    machine/location.  The run id is still useful because the current project
    tree can then be reconstructed from ``project_root/results``.
    """
    text = str(path_value or "").replace("\\", "/").rstrip("/")
    parts = [piece for piece in text.split("/") if piece]
    if len(parts) >= 3 and parts[-2] == "forecasts":
        return str(parts[-3])
    return None


def _metadata_run_id(metadata: Mapping, keys: Sequence[str]) -> str | None:
    """Read an exact component run id from legacy/new aggregate metadata."""
    for field in ("component_run_ids", "run_ids", "component_runs"):
        mapping = metadata.get(field)
        if not isinstance(mapping, Mapping):
            continue
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, Mapping):
                value = value.get("run_id") or value.get("id")
            if value not in (None, ""):
                return str(value)
    return None


def _resolve_component_forecast_store(
    metadata: Mapping,
    model_id: str,
    *,
    project_root: str | Path,
    vintage: str,
    forecast_name: str,
) -> Path:
    """Resolve one component forecast store across all aggregate schemas.

    Production aggregate metadata uses the *aggregate key* for weekly petrol
    and diesel (``petrol`` / ``diesel``), while the BVAR run directories use
    canonical model ids (``car_fuels_petrol`` / ``car_fuels_diesel``). Older
    stores may also contain absolute paths that no longer exist after moving the
    project.  Resolution therefore follows a deterministic hierarchy:

    1. canonical model-id key;
    2. model ``aggregate_key`` / ``spec_key`` aliases;
    3. exact run id recorded elsewhere in aggregate metadata;
    4. rebuild a stale stored absolute path under the current ``results`` root.

    It never selects "latest" or a promoted run, because that could silently
    change the provenance of a historical aggregate.
    """
    from energy_bvar_pipeline import model_spec

    spec = model_spec(model_id)
    keys: list[str] = []
    for value in (
        str(model_id),
        str(getattr(spec, "aggregate_key", "") or ""),
        str(getattr(spec, "spec_key", "") or ""),
    ):
        if value and value not in keys:
            keys.append(value)

    stores = metadata.get("component_forecast_stores", {})
    stores = dict(stores) if isinstance(stores, Mapping) else {}
    raw_store = next((stores[key] for key in keys if key in stores), None)

    current_results = Path(project_root) / "results"
    canonical_store: Path | None = None
    if raw_store not in (None, ""):
        raw_path = Path(str(raw_store))
        if raw_path.is_dir():
            return raw_path
        run_id = _run_id_from_forecast_store(raw_store)
        if run_id:
            canonical_store = (
                current_results
                / str(model_id)
                / str(vintage)
                / str(run_id)
                / "forecasts"
                / str(forecast_name)
            )
            if canonical_store.is_dir():
                return canonical_store

    run_id = _metadata_run_id(metadata, keys)
    if run_id:
        candidate = (
            current_results
            / str(model_id)
            / str(vintage)
            / str(run_id)
            / "forecasts"
            / str(forecast_name)
        )
        if candidate.is_dir():
            return candidate
        canonical_store = candidate

    aliases = ", ".join(repr(key) for key in keys)
    detail = (
        f" Recorded path: {raw_store}." if raw_store not in (None, "") else ""
    )
    if canonical_store is not None:
        detail += f" Reconstructed current path: {canonical_store}."
    raise FittedUnavailableError(
        f"Aggregate provenance cannot resolve the {model_id!r} forecast store "
        f"using keys [{aliases}].{detail}"
    )


def _fit_arrays(run_directory: Path) -> tuple[dict, dict[str, np.ndarray], str]:
    """Load compact fit cache, falling back to the full Gibbs cache."""
    metadata_path = run_directory / "metadata.json"
    if not metadata_path.is_file():
        raise FittedUnavailableError(f"Run metadata missing: {metadata_path}")
    metadata = _json_load(metadata_path)

    fit_path = run_directory / "fit_draws.npz"
    full_path = run_directory / "draws.npz"
    source = None
    path = None
    if fit_path.is_file():
        path = fit_path
        source = "fit_draws.npz"
    elif full_path.is_file():
        path = full_path
        source = "draws.npz"
    else:
        raise FittedUnavailableError(
            f"{metadata.get('model_id', run_directory.parent.parent.name)} run "
            f"{metadata.get('run_id', run_directory.name)[:12]} was saved without "
            "fit_draws.npz or draws.npz. The posterior coefficient draws no longer "
            "exist on disk; the true BVAR in-sample fit cannot be reconstructed "
            "without re-running that model once."
        )

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if "B" not in arrays:
        raise FittedUnavailableError(f"{path} does not contain posterior B draws.")
    return metadata, arrays, source


def _select_draw_indices(n_total: int, max_draws: int, seed: int) -> np.ndarray:
    n = min(int(max_draws), int(n_total))
    if n < 1:
        raise FittedReconstructionError("No posterior draws are available for fitted paths.")
    if n == n_total:
        return np.arange(n_total, dtype=int)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_total, size=n, replace=False)).astype(int)


def _previous_levels_balanced(prep: Mapping, dates: pd.DatetimeIndex) -> np.ndarray:
    levels = pd.DataFrame(prep["levels"]).astype(float).sort_index()
    positions = levels.index.get_indexer(dates)
    if (positions <= 0).any():
        raise FittedReconstructionError(
            "Could not recover the one-period conditioning levels for fitted values."
        )
    previous = levels.iloc[positions - 1].to_numpy(dtype=float)
    if previous.shape != (len(dates), int(prep["n"])):
        raise FittedReconstructionError("Conditioning-level shape mismatch.")
    return previous


def _fitted_model_level_paths(
    run_directory: str | Path,
    *,
    project_root: str | Path,
    max_draws: int = 500,
    seed: int = 2026,
) -> dict:
    """Posterior one-step fitted level paths for one saved BVAR run."""
    from energy_bvar_model import _prepare_var_regression, prepare_bvar_panel
    from energy_bvar_pipeline import build_panel

    run_directory = Path(run_directory)
    metadata, arrays, cache_source = _fit_arrays(run_directory)
    model_id = str(metadata["model_id"])
    vintage = str(metadata["vintage"])
    p = int(metadata["p"])
    missing_method = str(metadata.get("missing_data_method", "dk"))

    panel = build_panel(model_id, vintage, project_root=project_root)
    variables = [str(v) for v in metadata.get("variables", panel.variables)]
    if variables != list(panel.variables):
        raise FittedReconstructionError(
            f"{model_id}: stored variables {variables} differ from current panel "
            f"contract {list(panel.variables)}."
        )

    prep = prepare_bvar_panel(
        panel.levels,
        p=p,
        variables=variables,
        exog=panel.exog,
        frequency=panel.frequency,
        missing_data_method=missing_method,
    )
    B_all = np.asarray(arrays["B"], dtype=float)
    if B_all.ndim != 3:
        raise FittedReconstructionError(f"{model_id}: B must be 3-D, got {B_all.shape}.")
    indices = _select_draw_indices(B_all.shape[0], max_draws=max_draws, seed=seed)
    B = B_all[indices]
    n_draws = len(indices)
    n = len(variables)

    if not prep.get("requires_data_augmentation", False):
        X = np.asarray(prep["X"], dtype=float)
        dates = pd.DatetimeIndex(prep["dates"], name="date")
        if B.shape[1:] != (X.shape[1], n):
            raise FittedReconstructionError(
                f"{model_id}: coefficient shape {B.shape[1:]} does not match "
                f"design {(X.shape[1], n)}."
            )
        fitted_diff = np.einsum("tk,dkj->dtj", X, B, optimize=True)
        previous = _previous_levels_balanced(prep, dates)
        fitted_levels = fitted_diff + previous[None, :, :]
    else:
        if "completed_differences_draws" not in arrays:
            raise FittedUnavailableError(
                f"{model_id}: this run used exact DK interior-data augmentation, "
                "but the saved fit cache has no completed_differences_draws. "
                "Re-run the model once with the updated energy_bvar_io.py."
            )
        completed_all = np.asarray(arrays["completed_differences_draws"], dtype=float)
        if completed_all.shape[0] != B_all.shape[0]:
            raise FittedReconstructionError(
                f"{model_id}: B and completed-difference draw counts differ."
            )
        completed = completed_all[indices]
        dates = pd.DatetimeIndex(prep["dates"], name="date")
        fitted_levels = np.empty((n_draws, len(dates), n), dtype=float)
        warmup_index = prep["warmup_differences"].index
        full_index = warmup_index.append(dates)
        for out_i in range(n_draws):
            complete = pd.DataFrame(
                completed[out_i], index=full_index, columns=variables
            )
            _, X, regression_dates = _prepare_var_regression(
                complete, p, prep.get("estimation_exog")
            )
            regression_dates = pd.DatetimeIndex(regression_dates, name="date")
            if not regression_dates.equals(dates):
                raise FittedReconstructionError(
                    f"{model_id}: draw-specific design calendar drifted from prep dates."
                )
            fitted_diff = X @ B[out_i]
            # completed[p:] are the draw-specific realised/latent differences on
            # the regression dates. They provide the t-1 conditioning level.
            estimation_diff = complete.iloc[p:].to_numpy(dtype=float)
            anchor = np.asarray(prep["augmentation_anchor_level"], dtype=float)
            previous = np.empty((len(dates), n), dtype=float)
            previous[0] = anchor
            if len(dates) > 1:
                previous[1:] = anchor[None, :] + np.cumsum(
                    estimation_diff[:-1], axis=0
                )
            fitted_levels[out_i] = previous + fitted_diff

    if not np.isfinite(fitted_levels).all():
        raise FittedReconstructionError(f"{model_id}: non-finite fitted levels were produced.")

    forecast = {
        "variables": variables,
        "level_paths": fitted_levels,
        "path_dates": dates,
        "future_dates": pd.DatetimeIndex([], name="date"),
        "tail_length": len(dates),
        "frequency": panel.frequency,
        "last_calendar_date": dates[-1],
        "balanced_end": prep["balanced_end"],
        "missing_data_method": prep.get("missing_data_method", missing_method),
        "missing_treatment_exact": prep.get("missing_treatment_exact", True),
        "missing_data_approximation_used": prep.get(
            "missing_data_approximation_used", False
        ),
    }
    result = {
        "prep": prep,
        "metadata": metadata,
        "variables": variables,
        "p": p,
        "frequency": panel.frequency,
    }
    return {
        "model_id": model_id,
        "vintage": vintage,
        "run_id": str(metadata["run_id"]),
        "draw_indices": indices,
        "cache_source": cache_source,
        "forecast": forecast,
        "result": result,
        "dataset_path": Path(panel.dataset_path),
    }


def _valid_positive_hicp(obj: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    level = np.asarray(obj["hicp_level_paths"], dtype=float)
    yoy = np.asarray(obj["hicp_yoy_paths"], dtype=float)
    draw_indices = np.asarray(
        obj.get("draw_indices", np.arange(level.shape[0])), dtype=int
    )
    valid = np.isfinite(level).all(axis=1) & (level > 0).all(axis=1)
    if not valid.any():
        raise FittedReconstructionError(
            f"{obj.get('model_id')}: no finite positive fitted HICP draws remain."
        )
    return level[valid], yoy[valid], draw_indices[valid]


def build_component_fitted_hicp(
    run_directory: str | Path,
    *,
    project_root: str | Path,
    forecast_name: str = "unconditional",
    max_draws: int = 500,
    seed: int = 2026,
) -> dict:
    """Create posterior fitted monthly HICP paths for one component run."""
    from energy_bvar_component_hicp import build_component_hicp_forecast

    raw = _fitted_model_level_paths(
        run_directory,
        project_root=project_root,
        max_draws=max_draws,
        seed=seed,
    )
    hicp = build_component_hicp_forecast(
        raw["model_id"],
        raw["forecast"],
        result=raw["result"],
        dataset_path=raw["dataset_path"],
        project_root=project_root,
        forecast_name=forecast_name,
    )
    level, yoy, retained = _valid_positive_hicp(hicp)
    out = dict(hicp)
    out["hicp_level_paths"] = level
    out["hicp_yoy_paths"] = yoy
    # build_component_hicp_forecast may itself reject weekly paths. Its
    # draw_indices are positions in the pseudo-forecast draw pool, so map them
    # back to the original posterior B draws retained from the run.
    if len(retained) and int(np.max(retained)) < len(raw["draw_indices"]):
        out["posterior_draw_indices"] = raw["draw_indices"][retained]
    else:
        out["posterior_draw_indices"] = retained
    out["fitted_definition"] = "one_step_posterior_conditional_mean_XB"
    out["fit_cache_source"] = raw["cache_source"]
    return out


def _save_component_fitted(obj: Mapping, forecast_directory: Path) -> None:
    forecast_directory.mkdir(parents=True, exist_ok=True)
    level = np.asarray(obj["hicp_level_paths"], dtype=float)
    yoy = np.asarray(obj["hicp_yoy_paths"], dtype=float)
    dates = pd.DatetimeIndex(obj["path_dates"], name="date")
    np.savez_compressed(
        forecast_directory / FITTED_COMPONENT_FILENAME,
        hicp_level_paths=level,
        hicp_yoy_paths=yoy,
        path_dates=dates.astype("datetime64[ns]").to_numpy(),
        posterior_draw_indices=np.asarray(
            obj.get("posterior_draw_indices", np.arange(level.shape[0])), dtype=int
        ),
    )
    metadata = {
        "fitted_schema_version": FITTED_SCHEMA_VERSION,
        "model_id": str(obj.get("model_id")),
        "vintage": str(obj.get("vintage")),
        "run_id": str(obj.get("run_id")),
        "forecast_name": str(obj.get("forecast_name", "unconditional")),
        "hicp_series": str(obj.get("hicp_series")),
        "hicp_label": str(obj.get("hicp_label")),
        "n_draws": int(level.shape[0]),
        "path_start": None if len(dates) == 0 else dates[0].isoformat(),
        "path_end": None if len(dates) == 0 else dates[-1].isoformat(),
        "fitted_definition": str(obj.get("fitted_definition")),
        "fit_cache_source": str(obj.get("fit_cache_source")),
        "transformation_method": str(obj.get("transformation_method")),
    }
    (forecast_directory / FITTED_COMPONENT_METADATA).write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )


def materialize_component_fitted_hicp(
    run_directory: str | Path,
    *,
    project_root: str | Path,
    forecast_name: str = "unconditional",
    max_draws: int = 500,
    seed: int = 2026,
    overwrite: bool = False,
) -> dict:
    run_directory = Path(run_directory)
    target = run_directory / "forecasts" / str(forecast_name)
    meta_path = target / FITTED_COMPONENT_METADATA
    draws_path = target / FITTED_COMPONENT_FILENAME
    if meta_path.is_file() and draws_path.is_file() and not overwrite:
        metadata = _json_load(meta_path)
        with np.load(draws_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        return {
            **metadata,
            "hicp_level_paths": arrays["hicp_level_paths"],
            "hicp_yoy_paths": arrays["hicp_yoy_paths"],
            "path_dates": pd.DatetimeIndex(pd.to_datetime(arrays["path_dates"]), name="date"),
            "posterior_draw_indices": arrays.get("posterior_draw_indices"),
        }

    obj = build_component_fitted_hicp(
        run_directory,
        project_root=project_root,
        forecast_name=forecast_name,
        max_draws=max_draws,
        seed=seed,
    )
    _save_component_fitted(obj, target)
    return obj


def _intersection_calendar(objects: Sequence[Mapping], *, start: pd.Timestamp) -> pd.DatetimeIndex:
    common: pd.DatetimeIndex | None = None
    for obj in objects:
        dates = pd.DatetimeIndex(obj["path_dates"], name="date")
        dates = dates[dates >= start]
        common = dates if common is None else common.intersection(dates)
    if common is None or len(common) == 0:
        raise FittedReconstructionError("No common monthly fitted calendar exists.")
    common = common.sort_values()
    expected = pd.date_range(common[0], common[-1], freq="MS", name="date")
    common = common.intersection(expected)
    if not common.equals(expected):
        raise FittedReconstructionError("Common fitted calendar contains monthly gaps.")
    return common


def _align(obj: Mapping, dates: pd.DatetimeIndex, key: str = "hicp_level_paths") -> np.ndarray:
    source_dates = pd.DatetimeIndex(obj["path_dates"], name="date")
    pos = source_dates.get_indexer(dates)
    if (pos < 0).any():
        raise FittedReconstructionError("A fitted component is missing common aggregation dates.")
    return np.asarray(obj[key], dtype=float)[:, pos]


def _yoy_paths(paths: np.ndarray, dates: pd.DatetimeIndex, history: pd.Series) -> np.ndarray:
    values = np.asarray(paths, dtype=float)
    history = history.astype(float).dropna().sort_index()
    full = pd.date_range(
        min(history.index.min(), dates.min()),
        max(history.index.max(), dates.max()),
        freq="MS",
    )
    out = np.full_like(values, np.nan, dtype=float)
    for d in range(values.shape[0]):
        s = history.reindex(full)
        s.loc[dates] = values[d]
        out[d] = (100.0 * (s / s.shift(12) - 1.0)).reindex(dates).to_numpy(dtype=float)
    return out


def build_aggregate_bvar_fitted(
    aggregate_directory: str | Path,
    *,
    project_root: str | Path,
    max_draws: int = 500,
    seed: int = 2026,
) -> dict:
    """Build six fitted component paths and their fitted HICP Energy aggregate."""
    from energy_bvar_aggregate import (
        aggregate_component_draw_paths_laspeyres,
        historical_model_component_reconstruction,
        independent_draw_pairing,
        load_aggregation_inputs,
    )

    aggregate_directory = Path(aggregate_directory)
    metadata = _json_load(aggregate_directory / "metadata.json")
    vintage = str(metadata["vintage"])
    forecast_name = str(metadata.get("forecast_name", "unconditional"))
    # Resolve stores through the pipeline's canonical/aggregate-key contract.
    # In particular, notebook/production aggregate metadata stores the weekly
    # car-fuel paths under ``petrol`` and ``diesel``, whereas their BVAR model
    # ids on disk are ``car_fuels_petrol`` and ``car_fuels_diesel``.
    stores = {
        name: _resolve_component_forecast_store(
            metadata,
            name,
            project_root=project_root,
            vintage=vintage,
            forecast_name=forecast_name,
        )
        for name in MODEL_IDS
    }

    fitted = {}
    cache_sources = {}
    for i, name in enumerate(MODEL_IDS):
        run_dir = _run_directory_from_forecast_store(stores[name])
        obj = materialize_component_fitted_hicp(
            run_dir,
            project_root=project_root,
            forecast_name=forecast_name,
            max_draws=max_draws,
            seed=seed + 17 * i,
        )
        fitted[name] = obj
        cache_sources[name] = obj.get("fit_cache_source")

    processed = Path(project_root) / "data" / "processed" / vintage
    inputs = load_aggregation_inputs(processed)
    indices = inputs["indices"]
    weights = inputs["weights"]
    model_history = historical_model_component_reconstruction(processed)

    # Petrol + diesel -> fitted car-fuels HICP.  The small unmodelled residual
    # "other transport fuels" remains observed historically, because the paper
    # suite has no BVAR for it.
    transport_dates = _intersection_calendar(
        [fitted["car_fuels_petrol"], fitted["car_fuels_diesel"]],
        start=MIN_AGGREGATION_DATE,
    )
    petrol = _align(fitted["car_fuels_petrol"], transport_dates)
    diesel = _align(fitted["car_fuels_diesel"], transport_dates)
    n_transport = min(max_draws, petrol.shape[0], diesel.shape[0])
    other_series = indices["hicp_other_transport_fuels"].reindex(transport_dates)
    if other_series.isna().any():
        raise FittedReconstructionError(
            "Observed other-transport-fuels HICP is incomplete on the fitted calendar."
        )
    other = np.repeat(other_series.to_numpy(dtype=float)[None, :], n_transport, axis=0)
    transport_paired, _ = independent_draw_pairing(
        {
            "hicp_petrol": petrol,
            "hicp_diesel": diesel,
            "hicp_other_transport_fuels": other,
        },
        n_draws=n_transport,
        seed=seed + 1001,
    )
    transport_history_index = pd.concat(
        [
            pd.Series(
                [100.0],
                index=pd.DatetimeIndex([pd.Timestamp("2016-12-01")], name="date"),
            ),
            model_history["transport_energy"]["index"],
        ]
    ).sort_index()
    transport_component_history = indices[list(TRANSPORT_COMPONENTS)]
    transport_weights = weights[list(TRANSPORT_COMPONENTS)]
    car = aggregate_component_draw_paths_laspeyres(
        transport_component_history,
        transport_paired,
        transport_dates,
        transport_weights,
        transport_history_index,
        components=TRANSPORT_COMPONENTS,
    )
    car_obj = {"path_dates": transport_dates, "hicp_level_paths": car["level_paths"]}

    six_objects = {
        "car_fuels": car_obj,
        "liquid_fuels": fitted["liquid_fuels"],
        "gas": fitted["gas"],
        "electricity": fitted["electricity"],
        "heat_energy": fitted["heat_energy"],
        "solid_fuels": fitted["solid_fuels"],
    }
    dates = _intersection_calendar(list(six_objects.values()), start=MIN_AGGREGATION_DATE)
    unpaired = {name: _align(obj, dates) for name, obj in six_objects.items()}
    n_effective = min(int(max_draws), *(arr.shape[0] for arr in unpaired.values()))
    paired, pairing_indices = independent_draw_pairing(
        unpaired, n_draws=n_effective, seed=seed + 2001
    )

    component_history = model_history["component_history"]
    energy = aggregate_component_draw_paths_laspeyres(
        component_history,
        paired,
        dates,
        model_history["model_weights"],
        indices["hicp_energy"],
        components=SIX_COMPONENTS,
    )
    component_level_paths = np.stack([paired[name] for name in SIX_COMPONENTS], axis=-1)
    component_yoy_paths = np.stack(
        [_yoy_paths(paired[name], dates, component_history[name]) for name in SIX_COMPONENTS],
        axis=-1,
    )
    energy_level_paths = np.asarray(energy["level_paths"], dtype=float)
    energy_yoy_paths = _yoy_paths(energy_level_paths, dates, indices["hicp_energy"])

    return {
        "fitted_schema_version": FITTED_SCHEMA_VERSION,
        "vintage": vintage,
        "aggregate_run_id": str(metadata.get("aggregate_run_id", aggregate_directory.name)),
        "forecast_name": forecast_name,
        "dates": dates,
        "components": list(SIX_COMPONENTS),
        "component_level_paths": component_level_paths,
        "component_yoy_paths": component_yoy_paths,
        "energy_level_paths": energy_level_paths,
        "energy_yoy_paths": energy_yoy_paths,
        "n_draws": int(n_effective),
        "pairing_seed": int(seed),
        "pairing_indices": pairing_indices,
        "fit_cache_sources": cache_sources,
        "fitted_definition": "one_step_posterior_conditional_mean_XB_then_HICP_bridge_then_Laspeyres",
    }


def materialize_aggregate_bvar_fitted(
    aggregate_directory: str | Path,
    *,
    project_root: str | Path,
    max_draws: int = 500,
    seed: int = 2026,
    overwrite: bool = False,
) -> dict:
    """Materialise/reload a compact fitted aggregate cache beside the aggregate run."""
    directory = Path(aggregate_directory)
    arrays_path = directory / FITTED_AGGREGATE_FILENAME
    metadata_path = directory / FITTED_AGGREGATE_METADATA
    if arrays_path.is_file() and metadata_path.is_file() and not overwrite:
        metadata = _json_load(metadata_path)
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        return {
            **metadata,
            "dates": pd.DatetimeIndex(pd.to_datetime(arrays["dates"]), name="date"),
            "components": [str(x) for x in metadata["components"]],
            "component_level_paths": arrays["component_level_paths"],
            "component_yoy_paths": arrays["component_yoy_paths"],
            "energy_level_paths": arrays["energy_level_paths"],
            "energy_yoy_paths": arrays["energy_yoy_paths"],
        }

    obj = build_aggregate_bvar_fitted(
        directory,
        project_root=project_root,
        max_draws=max_draws,
        seed=seed,
    )
    dates = pd.DatetimeIndex(obj["dates"], name="date")
    np.savez_compressed(
        arrays_path,
        dates=dates.astype("datetime64[ns]").to_numpy(),
        component_level_paths=np.asarray(obj["component_level_paths"], dtype=float),
        component_yoy_paths=np.asarray(obj["component_yoy_paths"], dtype=float),
        energy_level_paths=np.asarray(obj["energy_level_paths"], dtype=float),
        energy_yoy_paths=np.asarray(obj["energy_yoy_paths"], dtype=float),
    )
    meta = {
        "fitted_schema_version": FITTED_SCHEMA_VERSION,
        "vintage": obj["vintage"],
        "aggregate_run_id": obj["aggregate_run_id"],
        "forecast_name": obj["forecast_name"],
        "components": list(obj["components"]),
        "n_draws": int(obj["n_draws"]),
        "pairing_seed": int(obj["pairing_seed"]),
        "fitted_definition": obj["fitted_definition"],
        "fit_cache_sources": obj["fit_cache_sources"],
        "path_start": dates[0].isoformat(),
        "path_end": dates[-1].isoformat(),
    }
    metadata_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return obj


__all__ = [
    "FITTED_SCHEMA_VERSION",
    "FittedReconstructionError",
    "FittedUnavailableError",
    "build_component_fitted_hicp",
    "materialize_component_fitted_hicp",
    "build_aggregate_bvar_fitted",
    "materialize_aggregate_bvar_fitted",
]
