"""Production pipeline for the locked joint Headline-HICP BVAR.

Contract
--------
- model_id: ``headline_joint``;
- dense BVAR(12), no production lag selector;
- 11 deterministic monthly dummies, December omitted;
- Minnesota prior + stochastic volatility + outliers + DK engine;
- monthly Food index is already supplied by the processed builder from
  Eurostat ``FOOD``;
- annual Food weight is ``FOOD_NP + FOOD_P`` upstream;
- forecast/publication horizon is hard-capped at 12 months;
- Headline is reconstructed draw-by-draw from Energy/Food/NEIG/Services.

This module intentionally reuses the validated Energy engine and I/O layout.
It does not change the seven Energy models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from energy_bvar_io import save_energy_bvar_forecast, save_energy_bvar_result
from energy_bvar_model import (
    BVARSVOPriorConfig,
    SamplerConfig,
    outlier_mean_frequency_from_years,
    forecast_bvar_sv_outlier,
    monthly_seasonal_dummies,
    run_energy_bvar,
)
from headline_joint_bvar import (
    MODEL_ID,
    NATIVE_VARIABLES,
    STATE_VARIABLES,
    STATE_VARIABLE_MAP,
    aggregate_headline_draws,
    headline_reconstruction_diagnostics,
    load_headline_joint_panel,
    load_headline_joint_weights,
    load_official_headline_total,
    native_to_state_levels,
    reconstruct_headline_history,
    resolve_headline_joint_dataset,
    resolve_headline_joint_weights,
    resolve_headline_official_indices,
    state_paths_to_native,
)

PIPELINE_VERSION = "2026-08-17-frequency-aware-a-prior-v3"
# HEADLINE DELIVERY 1 — LINEAGE HARNESS
MODEL_STATUS = "locked"
BASELINE_P = 12
MAX_PUBLISHED_HORIZON_MONTHS = 12
SEASONAL_REFERENCE_MONTH = 12
SEASONAL_PREFIX = "month"
SEASONAL_EXOG_PRIOR_SCALE = 10.0
FOOD_SOURCE = {
    "provider": "Eurostat",
    "dataset": "prc_hicp_minr",
    "key": "M.I25.FOOD.EA",
    "code": "FOOD",
    "unit": "I25",
    "geo": "EA",
}
FORECAST_NAME = "unconditional"


def production_prior_config() -> BVARSVOPriorConfig:
    """Exact prior contract used by the locked production notebook."""
    return BVARSVOPriorConfig(
        lambda1=0.20,
        lambda2=0.50,
        lambda3=1.00,
        lambda4=10.0,
        a_prior_var=10.0,
        phi_prior_mean=0.02,
        phi_prior_df=10.0,
        h0_var=4.0,
        outlier_mean_frequency=outlier_mean_frequency_from_years(4.0, "monthly"),
        outlier_prior_observations=120.0,
        outlier_grid_min=2.0,
        outlier_grid_max=20.0,
        outlier_grid_step=1.0,
        ksc_offset_scale=1e-6,
    )


def production_sampler_config() -> SamplerConfig:
    """Exact sampler contract used by the locked production notebook."""
    return SamplerConfig(
        reps=6_000,
        burn=3_000,
        thin=1,
        seed=42,
        max_stability_tries=1_000,
        progress_every=500,
    )


class HeadlinePipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class HeadlineInputs:
    vintage: str
    dataset_path: Path
    weights_path: Path
    official_indices_path: Path
    native_levels: pd.DataFrame
    weights: pd.DataFrame
    official_total: pd.Series

    @property
    def sample_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.native_levels.index.min())

    @property
    def sample_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.native_levels.index.max())


@dataclass(frozen=True)
class HeadlineRunOutcome:
    vintage: str
    run_id: str
    run_directory: Path | None
    forecast_directory: Path | None
    horizon: int
    n_posterior_draws: int
    n_forecast_draws: int
    inputs: HeadlineInputs
    result: Mapping
    component_forecast: Mapping
    aggregate_forecast: Mapping


def find_project_root(start: str | Path | None = None) -> Path:
    start_path = Path(start or __file__).resolve()
    if start_path.is_file():
        start_path = start_path.parent
    for candidate in (start_path, *start_path.parents):
        if (candidate / "src" / "model").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate project root containing src/model and data.")


def model_contract() -> dict[str, object]:
    return {
        "suite": "headline",
        "model_id": MODEL_ID,
        "status": MODEL_STATUS,
        "frequency": "monthly",
        "lag_order": BASELINE_P,
        "seasonal_dummies": 11,
        "seasonal_reference_month": SEASONAL_REFERENCE_MONTH,
        "seasonal_exog_prior_scale": SEASONAL_EXOG_PRIOR_SCALE,
        "food_source": dict(FOOD_SOURCE),
        "food_weight_source": "Eurostat FOOD_NP + FOOD_P annual item weights",
        "publication_horizon_max_months": MAX_PUBLISHED_HORIZON_MONTHS,
        "dashboard_publish_beyond_h12": False,
        "pipeline_version": PIPELINE_VERSION,
        "sampler": {
            "reps": 6000,
            "burn": 3000,
            "thin": 1,
            "seed": 42,
            "max_stability_tries": 1000,
        },
        "prior": {
            "lambda1": 0.20,
            "lambda2": 0.50,
            "lambda3": 1.00,
            "lambda4": 10.0,
            "a_prior_var": 10.0,
            "a_prior_scale_normalized": True,
            "phi_prior_mean": 0.02,
            "phi_prior_df": 10.0,
            "h0_var": 4.0,
            "outlier_mean_frequency": outlier_mean_frequency_from_years(4.0, "monthly"),
            "outlier_mean_interval_years": 4.0,
            "outlier_mean_interval_periods": 48.0,
            "outlier_prior_observations": 120.0,
            "outlier_grid_min": 2.0,
            "outlier_grid_max": 20.0,
            "outlier_grid_step": 1.0,
            "ksc_offset_scale": 1e-6,
        },
    }


def _processed_root(project_root: str | Path | None = None) -> Path:
    root = find_project_root(project_root) if project_root is not None else find_project_root()
    return root / "data" / "processed"


def available_vintages(project_root: str | Path | None = None) -> list[str]:
    processed = _processed_root(project_root)
    required = (
        "headline_joint_monthly.csv",
        "headline_joint_weights_annual.csv",
        "headline_hicp_indices_monthly.csv",
    )
    vintages = []
    if not processed.is_dir():
        return vintages
    for directory in sorted(p for p in processed.iterdir() if p.is_dir()):
        if all((directory / name).is_file() for name in required):
            vintages.append(directory.name)
    return vintages


def resolve_vintage(
    vintage: str | None = None,
    *,
    project_root: str | Path | None = None,
) -> str:
    vintages = available_vintages(project_root)
    if not vintages:
        raise FileNotFoundError("No complete processed Headline vintage was found.")
    if vintage is None:
        return str(vintages[-1])
    vintage = str(vintage)
    if vintage not in vintages:
        raise FileNotFoundError(
            f"Headline vintage {vintage!r} is incomplete or absent. "
            f"Available complete vintages: {vintages}."
        )
    return vintage


def build_inputs(
    vintage: str | None = None,
    *,
    project_root: str | Path | None = None,
) -> HeadlineInputs:
    root = find_project_root(project_root) if project_root is not None else find_project_root()
    vintage = resolve_vintage(vintage, project_root=root)

    dataset_path = resolve_headline_joint_dataset(root, vintage=vintage)
    weights_path = resolve_headline_joint_weights(root, vintage=vintage)
    official_path = resolve_headline_official_indices(root, vintage=vintage)

    native_levels = load_headline_joint_panel(dataset_path)
    weights = load_headline_joint_weights(weights_path)
    official_total = load_official_headline_total(official_path)

    return HeadlineInputs(
        vintage=vintage,
        dataset_path=dataset_path,
        weights_path=weights_path,
        official_indices_path=official_path,
        native_levels=native_levels,
        weights=weights,
        official_total=official_total,
    )


def make_joint_seasonals(index: Sequence[pd.Timestamp]) -> pd.DataFrame:
    """Eleven monthly dummies; December is the omitted reference category."""
    out = monthly_seasonal_dummies(
        index,
        reference_month=SEASONAL_REFERENCE_MONTH,
        prefix=SEASONAL_PREFIX,
    ).astype(float)
    if out.shape[1] != 11:
        raise RuntimeError(f"Expected 11 monthly dummies, found {out.shape[1]}.")
    december = pd.DatetimeIndex(out.index).month == SEASONAL_REFERENCE_MONTH
    if len(out) and not out.loc[december].sum(axis=1).eq(0.0).all():
        raise RuntimeError("December must be the omitted seasonal category.")
    return out


def _native_conditions_to_state(
    conditions: Mapping[str, float | Sequence[float]] | None,
    *,
    allow_partial_level_conditions: bool = False,
) -> dict[str, float | np.ndarray] | None:
    if conditions is None:
        return None
    output: dict[str, float | np.ndarray] = {}
    for native_name, values in conditions.items():
        if native_name not in STATE_VARIABLE_MAP:
            raise KeyError(
                f"Unknown Headline condition {native_name!r}; choose from {NATIVE_VARIABLES}."
            )
        arr = np.asarray(values, dtype=float)
        if allow_partial_level_conditions:
            if np.isinf(arr).any():
                raise ValueError(f"{native_name}: partial HICP conditions cannot contain +/-inf.")
            finite = np.isfinite(arr)
            if not finite.any():
                raise ValueError(f"{native_name}: partial condition must constrain at least one month.")
            if np.any(arr[finite] <= 0):
                raise ValueError(f"{native_name}: finite conditioned HICP levels must be positive.")
            transformed = np.full(arr.shape, np.nan, dtype=float)
            transformed[finite] = np.log(arr[finite])
        else:
            if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
                raise ValueError(
                    f"{native_name}: conditioned HICP levels must be finite and positive."
                )
            transformed = np.log(arr)
        output[STATE_VARIABLE_MAP[native_name]] = (
            float(transformed) if transformed.ndim == 0 else transformed
        )
    return output


def _component_yoy_paths(
    native_paths: np.ndarray,
    path_dates: pd.DatetimeIndex,
    native_history: pd.DataFrame,
) -> np.ndarray:
    draws, _, _ = native_paths.shape
    output = np.full_like(native_paths, np.nan, dtype=float)
    pos = {pd.Timestamp(date): i for i, date in enumerate(path_dates)}
    history = native_history[NATIVE_VARIABLES].copy().sort_index()
    history.index = pd.DatetimeIndex(history.index).to_period("M").to_timestamp(how="start")

    for t, date in enumerate(path_dates):
        lag_date = pd.Timestamp(date) - pd.DateOffset(months=12)
        for j, name in enumerate(NATIVE_VARIABLES):
            if lag_date in pos:
                denominator = native_paths[:, pos[lag_date], j]
            else:
                if lag_date not in history.index or pd.isna(history.loc[lag_date, name]):
                    continue
                denominator = np.full(draws, float(history.loc[lag_date, name]), dtype=float)
            output[:, t, j] = 100.0 * (native_paths[:, t, j] / denominator - 1.0)
    return output


def _validate_horizon(H: int) -> int:
    H = int(H)
    if H < 1:
        raise ValueError("Forecast horizon must be positive.")
    if H > MAX_PUBLISHED_HORIZON_MONTHS:
        raise ValueError(
            f"Headline publication horizon is locked at <= "
            f"{MAX_PUBLISHED_HORIZON_MONTHS} months; received H={H}."
        )
    return H


def estimate_locked_headline(
    inputs: HeadlineInputs,
    *,
    prior_config: BVARSVOPriorConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    code_version: str = PIPELINE_VERSION,
) -> dict:
    """Estimate the only production specification: BVAR(12) + 11 month dummies."""
    state = native_to_state_levels(inputs.native_levels)
    seasonals = make_joint_seasonals(state.index)
    prior_config = production_prior_config() if prior_config is None else prior_config
    sampler_config = production_sampler_config() if sampler_config is None else sampler_config

    result = run_energy_bvar(
        levels=state,
        model_id=MODEL_ID,
        vintage=inputs.vintage,
        p=BASELINE_P,
        variables=STATE_VARIABLES,
        frequency="monthly",
        exog=seasonals,
        exog_prior_scale=SEASONAL_EXOG_PRIOR_SCALE,
        prior_config=prior_config,
        sampler_config=sampler_config,
        code_version=code_version,
    )
    result["headline_joint_context"] = {
        "native_variables": list(NATIVE_VARIABLES),
        "state_variables": list(STATE_VARIABLES),
        "state_variable_map": dict(STATE_VARIABLE_MAP),
        "native_levels": inputs.native_levels[NATIVE_VARIABLES].copy(),
        "seasonal_reference_month": SEASONAL_REFERENCE_MONTH,
    }
    result["metadata"].update(
        {
            "suite": "headline",
            "component": MODEL_ID,
            "model_class": "dense BVAR(12) + 11 monthly seasonal dummies",
            "native_variables": list(NATIVE_VARIABLES),
            "state_variables": list(STATE_VARIABLES),
            "state_variable_map": dict(STATE_VARIABLE_MAP),
            "transform_map": {name: "logdiff" for name in NATIVE_VARIABLES},
                "seasonal_dummies": True,
            "seasonal_reference_month": SEASONAL_REFERENCE_MONTH,
            "seasonal_exog_names": list(seasonals.columns),
            "hicp_food_source": dict(FOOD_SOURCE),
            "model_lock": model_contract(),
            # Delivery 1: estimation-time lineage. data_hash is already the
            # engine hash of exact state levels + deterministic exog.
            "source_data_hash": str(result["metadata"].get("data_hash", "")),
            "exog_names": list(result["prep"].get("exog_names", [])),
            "balanced_end": pd.Timestamp(result["prep"]["balanced_end"]).isoformat(),
            "last_calendar_date": pd.Timestamp(
                result["prep"]["last_calendar_date"]
            ).isoformat(),
        }
    )
    return result


def forecast_locked_headline(
    result: Mapping,
    *,
    inputs: HeadlineInputs,
    H: int = MAX_PUBLISHED_HORIZON_MONTHS,
    n_draws: int | None = None,
    native_level_conditions: Mapping[str, float | Sequence[float]] | None = None,
    allow_partial_level_conditions: bool = False,
    simulate_future_outliers: bool = True,
    seed: int = 2026,
) -> dict[str, object]:
    H = _validate_horizon(H)
    last_date = pd.Timestamp(result["prep"]["last_calendar_date"])
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        periods=H,
        freq="MS",
        name="date",
    )
    future_exog = make_joint_seasonals(future_dates)
    fitted_names = list(result["prep"].get("exog_names", []))
    if list(future_exog.columns) != fitted_names:
        raise ValueError(
            "Future monthly-dummy columns do not match the fitted contract: "
            f"future={list(future_exog.columns)}, fitted={fitted_names}."
        )

    state_forecast = forecast_bvar_sv_outlier(
        result,
        H=H,
        level_conditions=_native_conditions_to_state(
            native_level_conditions,
            allow_partial_level_conditions=bool(allow_partial_level_conditions),
        ),
        future_exog=future_exog,
        n_draws=n_draws,
        simulate_future_outliers=bool(simulate_future_outliers),
        seed=int(seed),
        allow_partial_level_conditions=bool(allow_partial_level_conditions),
    )
    if list(state_forecast["variables"]) != STATE_VARIABLES:
        raise ValueError(
            f"State-variable ordering changed: expected {STATE_VARIABLES}, "
            f"found {state_forecast['variables']}."
        )

    native_paths = state_paths_to_native(state_forecast["level_paths"])
    path_dates = pd.DatetimeIndex(state_forecast["path_dates"], name="date")
    component_yoy = _component_yoy_paths(
        native_paths,
        path_dates,
        inputs.native_levels,
    )
    tail = int(state_forecast["tail_length"])

    output = dict(state_forecast)
    output.update(
        {
            "component": MODEL_ID,
            "component_label": "Headline components joint BVAR",
            "native_variables": list(NATIVE_VARIABLES),
            "state_variables": list(STATE_VARIABLES),
            "native_level_paths": native_paths,
            "future_native_level_paths": native_paths[:, tail:],
            "component_yoy_paths": component_yoy,
            "future_component_yoy_paths": component_yoy[:, tail:],
            "native_level_conditions": (
                None if native_level_conditions is None
                else dict(native_level_conditions)
            ),
            "future_seasonal_exog": future_exog,
            "forecast_contract_version": "headline_joint_locked_h12_v1",
            "model_lock": model_contract(),
        }
    )
    return output


def reconstruct_headline_component_history(
    native_levels: pd.DataFrame,
    weights: pd.DataFrame,
    official_total: pd.Series,
) -> pd.DataFrame:
    """Historical index-level component contributions used by exact YoY attribution."""
    native = native_levels[NATIVE_VARIABLES].copy().sort_index()
    native.index = pd.DatetimeIndex(native.index).to_period("M").to_timestamp(how="start")
    weights = weights[NATIVE_VARIABLES].copy().sort_index()
    total = official_total.copy().sort_index()
    total.index = pd.DatetimeIndex(total.index).to_period("M").to_timestamp(how="start")

    reconstructed = reconstruct_headline_history(native, weights, total)
    if reconstructed.empty:
        raise ValueError("Historical Headline reconstruction is empty.")

    first_year = int(reconstructed.index.min().year)
    last_date = reconstructed.index.max()
    out = pd.DataFrame(index=reconstructed.index, columns=NATIVE_VARIABLES, dtype=float)
    anchor = None

    for year in range(first_year, int(last_date.year) + 1):
        if year not in weights.index or weights.loc[year, NATIVE_VARIABLES].isna().any():
            raise ValueError(f"No complete historical Headline weight vector for {year}.")
        w = weights.loc[year, NATIVE_VARIABLES].astype(float) / 1000.0
        dec_prev = pd.Timestamp(year - 1, 12, 1)
        if dec_prev not in native.index or native.loc[dec_prev, NATIVE_VARIABLES].isna().any():
            raise ValueError(f"Missing component December anchor at {dec_prev.date()}.")

        if anchor is None:
            if dec_prev not in total.index or pd.isna(total.loc[dec_prev]):
                raise ValueError(f"Missing official Headline anchor at {dec_prev.date()}.")
            anchor = float(total.loc[dec_prev])

        dec_component = native.loc[dec_prev, NATIVE_VARIABLES].astype(float)
        dates = out.index[out.index.year == year]
        for date in dates:
            ratios = native.loc[date, NATIVE_VARIABLES].astype(float) / dec_component
            out.loc[date, NATIVE_VARIABLES] = anchor * w * ratios

        december = pd.Timestamp(year, 12, 1)
        if december in reconstructed.index:
            anchor = float(reconstructed.loc[december])

    add_err = (out.sum(axis=1) - reconstructed).abs().max()
    if not np.isfinite(add_err) or add_err > 1e-10:
        raise RuntimeError(f"Historical component additivity failed: {add_err:.3e}.")
    return out


def exact_headline_yoy_contribution_paths(
    aggregate_forecast: Mapping,
    *,
    inputs: HeadlineInputs,
) -> np.ndarray:
    """Exact draw-wise pp contributions to Headline YoY.

    For a lag date inside the posterior path, both current and lag contributions
    are draw-specific. For a lag date before the posterior path, historical
    reconstructed component shares are rescaled to the *official* total level,
    so the four contributions sum exactly to the Headline YoY definition used
    by ``aggregate_headline_draws``.
    """
    current = np.asarray(
        aggregate_forecast["headline_component_index_contributions"],
        dtype=float,
    )
    headline = np.asarray(aggregate_forecast["headline_level_paths"], dtype=float)
    headline_yoy = np.asarray(aggregate_forecast["headline_yoy_paths"], dtype=float)
    dates = pd.DatetimeIndex(aggregate_forecast["path_dates"], name="date")
    if current.ndim != 3 or current.shape[2] != len(NATIVE_VARIABLES):
        raise ValueError("Headline component contribution path dimensions are invalid.")

    hist_components = reconstruct_headline_component_history(
        inputs.native_levels,
        inputs.weights,
        inputs.official_total,
    )
    hist_reconstructed = hist_components.sum(axis=1)
    official = inputs.official_total.copy()
    official.index = pd.DatetimeIndex(official.index).to_period("M").to_timestamp(how="start")

    pos = {pd.Timestamp(date): i for i, date in enumerate(dates)}
    out = np.full_like(current, np.nan, dtype=float)

    for t, date in enumerate(dates):
        lag_date = pd.Timestamp(date) - pd.DateOffset(months=12)

        if lag_date in pos:
            lag_pos = pos[lag_date]
            lag_components = current[:, lag_pos, :]
            denominator = headline[:, lag_pos]
        else:
            if (
                lag_date not in hist_components.index
                or lag_date not in official.index
                or pd.isna(official.loc[lag_date])
            ):
                continue
            hist_total = float(hist_reconstructed.loc[lag_date])
            if not np.isfinite(hist_total) or hist_total <= 0:
                continue
            official_total = float(official.loc[lag_date])
            shares = hist_components.loc[lag_date, NATIVE_VARIABLES].to_numpy(dtype=float) / hist_total
            lag_components = np.broadcast_to(
                official_total * shares,
                (current.shape[0], len(NATIVE_VARIABLES)),
            )
            denominator = np.full(current.shape[0], official_total, dtype=float)

        out[:, t, :] = 100.0 * (
            current[:, t, :] - lag_components
        ) / denominator[:, None]

    summed = np.nansum(out, axis=2)
    contribution_complete = np.isfinite(out).all(axis=2)
    comparable = np.isfinite(headline_yoy) & contribution_complete
    if comparable.any():
        err = np.nanmax(np.abs(summed[comparable] - headline_yoy[comparable]))
        if not np.isfinite(err) or err > 1e-9:
            raise RuntimeError(
                f"Headline YoY contribution additivity failed: {err:.3e} pp."
            )
    return out


def _save_headline_sidecars(
    *,
    outcome_result: Mapping,
    component_forecast: Mapping,
    aggregate_forecast: Mapping,
    yoy_contributions: np.ndarray,
    inputs: HeadlineInputs,
    run_directory: Path,
    forecast_directory: Path,
) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    forecast_directory.mkdir(parents=True, exist_ok=True)

    inputs.native_levels.to_csv(
        run_directory / "headline_native_history.csv",
        date_format="%Y-%m-%d",
    )
    inputs.weights.to_csv(
        run_directory / "headline_weights_annual.csv",
        index_label="year",
    )
    inputs.official_total.to_frame().to_csv(
        run_directory / "headline_official_total.csv",
        date_format="%Y-%m-%d",
    )

    reconstructed = reconstruct_headline_history(
        inputs.native_levels,
        inputs.weights,
        inputs.official_total,
    )
    recon_diag = headline_reconstruction_diagnostics(
        reconstructed,
        inputs.official_total,
    )
    reconstructed.to_frame().to_csv(
        run_directory / "headline_reconstructed_history.csv",
        date_format="%Y-%m-%d",
    )
    recon_diag.to_frame("value").to_csv(
        run_directory / "headline_reconstruction_diagnostics.csv",
        index_label="metric",
    )

    np.savez_compressed(
        forecast_directory / "headline_draws.npz",
        native_level_paths=np.asarray(component_forecast["native_level_paths"], dtype=float),
        component_yoy_paths=np.asarray(component_forecast["component_yoy_paths"], dtype=float),
        headline_level_paths=np.asarray(aggregate_forecast["headline_level_paths"], dtype=float),
        headline_yoy_paths=np.asarray(aggregate_forecast["headline_yoy_paths"], dtype=float),
        headline_component_index_contributions=np.asarray(
            aggregate_forecast["headline_component_index_contributions"], dtype=float
        ),
        headline_yoy_contribution_paths=np.asarray(yoy_contributions, dtype=float),
        headline_weight_year_used=np.asarray(
            aggregate_forecast["headline_weight_year_used"], dtype=int
        ),
        headline_weight_carried_forward=np.asarray(
            aggregate_forecast["headline_weight_carried_forward"], dtype=bool
        ),
    )
    contribution_sum = np.nansum(yoy_contributions, axis=2)
    headline_yoy = np.asarray(
        aggregate_forecast["headline_yoy_paths"],
        dtype=float,
    )
    contribution_complete = np.isfinite(yoy_contributions).all(axis=2)
    comparable = contribution_complete & np.isfinite(headline_yoy)
    yoy_additivity_error = (
        float(np.nanmax(np.abs(contribution_sum[comparable] - headline_yoy[comparable])))
        if comparable.any()
        else float("nan")
    )

    meta = {
        "headline_schema_version": "1.0",
        "model_id": MODEL_ID,
        "vintage": inputs.vintage,
        "run_id": str(outcome_result["metadata"]["run_id"]),
        "forecast_name": FORECAST_NAME,
        "path_dates": [
            pd.Timestamp(x).isoformat()
            for x in pd.DatetimeIndex(aggregate_forecast["path_dates"])
        ],
        "future_dates": [
            pd.Timestamp(x).isoformat()
            for x in pd.DatetimeIndex(aggregate_forecast["future_dates"])
        ],
        "tail_length": int(aggregate_forecast["tail_length"]),
        "native_variables": list(NATIVE_VARIABLES),
        "aggregate_series": "hicp_total",
        "aggregation_method": aggregate_forecast["headline_aggregation_method"],
        "future_weight_policy": aggregate_forecast["future_weight_policy"],
        "source_data_hash": str(
            outcome_result["metadata"].get("source_data_hash", "")
        ),
        "exog_names": list(outcome_result["metadata"].get("exog_names", [])),
        "balanced_end": outcome_result["metadata"].get("balanced_end"),
        "last_calendar_date": outcome_result["metadata"].get("last_calendar_date"),
        "n_forecast_draws": int(outcome_result["metadata"]["n_forecast_draws"]),
        "forecast_seed": int(outcome_result["metadata"]["forecast_seed"]),
        "simulate_future_outliers": bool(
            outcome_result["metadata"]["simulate_future_outliers"]
        ),
        "draw_wise_level_additivity_max_abs_error": float(
            aggregate_forecast["headline_drawwise_additivity_max_abs_error"]
        ),
        "yoy_contribution_additivity_max_abs_error": yoy_additivity_error,
        "headline_display_schema_version": "headline-1.0",
        "model_lock": model_contract(),
    }
    (forecast_directory / "headline_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )


def run_headline(
    vintage: str | None = None,
    *,
    project_root: str | Path | None = None,
    results_root: str | Path | None = None,
    prior_config: BVARSVOPriorConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    H: int = MAX_PUBLISHED_HORIZON_MONTHS,
    n_forecast_draws: int | None = 1000,
    simulate_future_outliers: bool = True,
    forecast_seed: int = 2026,
    persist: bool = True,
) -> HeadlineRunOutcome:
    """Estimate, forecast, aggregate and optionally persist the locked model."""
    H = _validate_horizon(H)
    root = find_project_root(project_root) if project_root is not None else find_project_root()
    inputs = build_inputs(vintage, project_root=root)

    result = estimate_locked_headline(
        inputs,
        prior_config=prior_config,
        sampler_config=sampler_config,
    )
    available = int(result["n_draws"])
    requested = available if n_forecast_draws is None else int(n_forecast_draws)
    n_forecast = min(available, requested)
    if n_forecast < 1:
        raise ValueError("n_forecast_draws must be positive.")

    # Exact unconditional forecast reconstruction contract, persisted before
    # save_energy_bvar_result() writes metadata.json.
    result["metadata"].update(
        {
            "n_forecast_draws": int(n_forecast),
            "forecast_seed": int(forecast_seed),
            "simulate_future_outliers": bool(simulate_future_outliers),
        }
    )

    component_forecast = forecast_locked_headline(
        result,
        inputs=inputs,
        H=H,
        n_draws=n_forecast,
        simulate_future_outliers=simulate_future_outliers,
        seed=forecast_seed,
    )
    aggregate_forecast = aggregate_headline_draws(
        component_forecast,
        native_levels=inputs.native_levels,
        weights=inputs.weights,
        official_total=inputs.official_total,
    )
    yoy_contrib = exact_headline_yoy_contribution_paths(
        aggregate_forecast,
        inputs=inputs,
    )
    aggregate_forecast = dict(aggregate_forecast)
    aggregate_forecast["headline_yoy_contribution_paths"] = yoy_contrib

    run_directory = None
    forecast_directory = None
    if persist:
        target_root = (
            Path(results_root)
            if results_root is not None
            else root / "results"
        )
        saved = save_energy_bvar_result(result, target_root)
        run_directory = Path(saved["directory"])
        saved_forecast = save_energy_bvar_forecast(
            component_forecast,
            run_directory,
            forecast_name=FORECAST_NAME,
        )
        forecast_directory = Path(saved_forecast["directory"])
        _save_headline_sidecars(
            outcome_result=result,
            component_forecast=component_forecast,
            aggregate_forecast=aggregate_forecast,
            yoy_contributions=yoy_contrib,
            inputs=inputs,
            run_directory=run_directory,
            forecast_directory=forecast_directory,
        )

    return HeadlineRunOutcome(
        vintage=inputs.vintage,
        run_id=str(result["metadata"]["run_id"]),
        run_directory=run_directory,
        forecast_directory=forecast_directory,
        horizon=H,
        n_posterior_draws=int(result["n_draws"]),
        n_forecast_draws=n_forecast,
        inputs=inputs,
        result=result,
        component_forecast=component_forecast,
        aggregate_forecast=aggregate_forecast,
    )


__all__ = [
    "PIPELINE_VERSION",
    "MODEL_STATUS",
    "BASELINE_P",
    "MAX_PUBLISHED_HORIZON_MONTHS",
    "SEASONAL_REFERENCE_MONTH",
    "SEASONAL_EXOG_PRIOR_SCALE",
    "FOOD_SOURCE",
    "production_prior_config",
    "production_sampler_config",
    "HeadlineInputs",
    "HeadlineRunOutcome",
    "model_contract",
    "available_vintages",
    "resolve_vintage",
    "build_inputs",
    "make_joint_seasonals",
    "estimate_locked_headline",
    "forecast_locked_headline",
    "reconstruct_headline_component_history",
    "exact_headline_yoy_contribution_paths",
    "run_headline",
]
