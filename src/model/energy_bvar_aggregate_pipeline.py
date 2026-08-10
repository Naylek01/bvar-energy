"""Orchestration layer for the HICP Energy aggregate.

This module lifts the sequence of notebook 10 (FINAL_v2) into a callable so the
notebook and the dashboard share one implementation. It re-estimates nothing and
re-implements no chain-linking: every econometric and Laspeyres operation is
delegated to :mod:`energy_bvar_aggregate` and the component tax modules.

Reference run this mirrors (vintage 20260808, strict weekly tax mode)::

    six-model cumulative max error              0.006756
    six-model annual re-anchored max error      0.008318
    max draw-wise contribution additivity error 0.0
    effective aggregate draws                   476
    weekly tax mode effective                   official_wob_tax_reattribution

:func:`assert_reference_audit` checks a result against those numbers.

Notes on two conventions that are easy to get wrong
---------------------------------------------------
*Draw pooling.* The admissibility filter retains different numbers of draws per
component. ``n_draws`` is therefore set to the minimum retained pool so that
:func:`independent_draw_pairing` permutes every component rather than
bootstrapping the short ones, which would inject asymmetric Monte Carlo noise.

*Fan interpretation.* Rejecting non-positive price paths truncates the lower
tail, so the reported bands are predictive intervals **conditional on
admissibility**. The per-model rejection rate is carried through to the result
so a dashboard can label them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from energy_bvar_aggregate import (
    AGGREGATE_MODULE_VERSION,
    aggregate_component_draw_paths_laspeyres,
    aggregate_draw_paths_laspeyres,
    assert_reconstruction_suite,
    carry_last_index_path,
    draw_yoy_and_contributions,
    filter_positive_price_draws,
    historical_model_component_reconstruction,
    historical_reconstruction_suite,
    independent_draw_pairing,
    load_aggregation_inputs,
    price_proxy_diagnostics,
    price_proxy_diagnostic_table,
    rebase_price_paths_to_hicp_index,
    summarise_draw_paths,
    weekly_paths_with_history_to_monthly_mean,
)
from energy_bvar_electricity import (
    expand_semester_series as expand_electricity_semester_series,
    load_electricity_tax_context,
    reattribute_electricity_taxes,
    validate_electricity_tax_round_trip,
)
from energy_bvar_gas import (
    expand_semester_series as expand_gas_semester_series,
    load_gas_tax_context,
    reattribute_gas_taxes,
    validate_tax_round_trip as validate_gas_tax_round_trip,
)
from energy_bvar_io import load_energy_bvar_forecast
from energy_bvar_weekly_fuels import load_weekly_tax_context, reattribute_weekly_taxes

from energy_bvar_pipeline import (
    CANONICAL_MODEL_IDS,
    PipelineError,
    find_project_root,
    model_spec,
    resolve_common_vintage,
    resolve_forecast_stores,
)

__all__ = [
    "AggregateError",
    "AggregateOutcome",
    "TRANSPORT_COMPONENTS",
    "MODEL_COMPONENTS",
    "WEEKLY_MODELS",
    "resolve_aggregate_vintage",
    "run_aggregate",
    "assert_reference_audit",
    "REFERENCE_AUDIT",
    "AGGREGATE_PIPELINE_VERSION",
    "MODEL_HICP_NAMES",
    "ensure_dashboard_aggregate",
]


class AggregateError(PipelineError):
    """Raised when the aggregation contract cannot be satisfied."""


# ---------------------------------------------------------------------------
# Component contracts (mirroring notebook 10)
# ---------------------------------------------------------------------------

WEEKLY_MODELS: tuple[str, ...] = ("petrol", "diesel", "liquid_fuels")

WEEKLY_TARGETS: dict[str, str] = {
    "petrol": "wob_petrol_pre_tax",
    "diesel": "wob_diesel_pre_tax",
    "liquid_fuels": "wob_heating_oil_pre_tax",
}

WEEKLY_HICP_COLUMNS: dict[str, str] = {
    "petrol": "hicp_petrol",
    "diesel": "hicp_diesel",
    "liquid_fuels": "hicp_liquid_fuels",
}

MONTHLY_DIRECT: dict[str, str] = {
    "heat_energy": "hicp_heat_energy",
    "solid_fuels": "hicp_solid_fuels",
}

TRANSPORT_COMPONENTS: tuple[str, ...] = (
    "hicp_diesel",
    "hicp_petrol",
    "hicp_other_transport_fuels",
)

MODEL_COMPONENTS: tuple[str, ...] = (
    "car_fuels",
    "liquid_fuels",
    "gas",
    "electricity",
    "heat_energy",
    "solid_fuels",
)

TRANSPORT_ANCHOR = pd.Timestamp("2016-12-01")

AGGREGATE_PIPELINE_VERSION = "2026-08-09-seven-model-hicp-auto-v2"

MODEL_HICP_NAMES: tuple[str, ...] = (
    "car_fuels_petrol",
    "car_fuels_diesel",
    "liquid_fuels",
    "gas",
    "electricity",
    "heat_energy",
    "solid_fuels",
)


REFERENCE_AUDIT: dict[str, float] = {
    "six_model_cumulative_max_error": 0.006756,
    "six_model_annual_reanchored_max_error": 0.008318,
    "max_additivity_error": 0.0,
    "n_aggregate_draws_effective": 476,
}


# ---------------------------------------------------------------------------
# Small helpers lifted out of the notebook
# ---------------------------------------------------------------------------


def _monthly_mean_series(series: pd.Series) -> pd.Series:
    series = series.astype(float).dropna().sort_index()
    grouped = series.groupby(series.index.to_period("M")).mean()
    grouped.index = pd.DatetimeIndex(
        grouped.index.to_timestamp(how="start"), name="date"
    )
    return grouped


def _common_monthly_dates(objects: Sequence[Mapping]) -> pd.DatetimeIndex:
    """Intersection of component calendars, refusing interior gaps."""
    common: pd.DatetimeIndex | None = None
    for obj in objects:
        dates = pd.DatetimeIndex(obj["dates"], name="date")
        common = dates if common is None else common.intersection(dates)
    if common is None or len(common) == 0:
        raise AggregateError("No common monthly aggregation dates.")
    common = pd.DatetimeIndex(common.sort_values(), name="date")
    expected = pd.date_range(common[0], common[-1], freq="MS", name="date")
    if not common.equals(expected):
        raise AggregateError("The common monthly path contains an interior gap.")
    return common


def _align_paths(obj: Mapping, key: str, dates: pd.DatetimeIndex) -> np.ndarray:
    source_dates = pd.DatetimeIndex(obj["dates"], name="date")
    positions = source_dates.get_indexer(dates)
    if (positions < 0).any():
        raise AggregateError("Requested dates are absent from a component path.")
    return np.asarray(obj[key], dtype=float)[:, positions]


def _scenario_values(value, dates: pd.DatetimeIndex, label: str) -> np.ndarray:
    if isinstance(value, pd.Series):
        series = value.astype(float).copy()
        series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp(
            how="start"
        )
        if series.index.has_duplicates:
            raise AggregateError(f"{label}: duplicate scenario months.")
        array = series.reindex(dates).to_numpy(dtype=float)
    else:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.repeat(float(array), len(dates))
    if array.ndim != 1 or len(array) != len(dates):
        raise AggregateError(
            f"{label}: expected scalar, dated Series or length {len(dates)}."
        )
    if not np.all(np.isfinite(array)):
        raise AggregateError(f"{label}: scenario contains non-finite values.")
    return array


def _apply_monthly_scenario(
    baseline_vat: pd.Series,
    baseline_excise: pd.Series,
    scenario: Mapping | None,
    *,
    excise_unit: str,
) -> tuple[pd.Series, pd.Series]:
    vat = baseline_vat.astype(float).copy()
    excise = baseline_excise.astype(float).copy()
    if scenario is None:
        return vat, excise

    scenario = dict(scenario)
    if "start_date" not in scenario:
        raise AggregateError("Monthly tax scenario requires 'start_date'.")
    if "vat_percent" not in scenario and "excise" not in scenario:
        raise AggregateError("Monthly tax scenario must change VAT and/or excise.")

    start = pd.Timestamp(scenario["start_date"]).to_period("M").to_timestamp(
        how="start"
    )
    active = vat.index[vat.index >= start]
    if len(active) == 0:
        raise AggregateError(
            f"Tax scenario starts after the forecast path: {start.date()}."
        )

    if "vat_percent" in scenario:
        values = _scenario_values(scenario["vat_percent"], active, "vat_percent")
        if np.any((values < 0) | (values > 100)):
            raise AggregateError(
                "VAT must be supplied in percentage points, e.g. 21 for 21%."
            )
        vat.loc[active] = values

    if "excise" in scenario:
        supplied = scenario.get("excise_unit")
        if supplied is None:
            raise AggregateError(f"Excise scenario requires excise_unit={excise_unit!r}.")
        norm = lambda x: "".join(str(x).lower().split())
        if norm(supplied) != norm(excise_unit):
            raise AggregateError(
                f"Excise unit mismatch: expected {excise_unit!r}, got {supplied!r}."
            )
        excise.loc[active] = _scenario_values(scenario["excise"], active, "excise")

    return vat, excise


def _monthly_pretax_forecast_to_hicp(
    forecast: Mapping,
    *,
    target_variable: str,
    tax_context: Mapping,
    expand_function,
    reattribute_function,
    tax_scenario: Mapping | None,
) -> dict:
    dates = pd.DatetimeIndex(forecast["path_dates"], name="date")
    variables = list(forecast["variables"])
    if target_variable not in variables:
        raise AggregateError(f"{target_variable!r} absent from {variables}.")
    j = variables.index(target_variable)
    pre_tax = np.asarray(forecast["level_paths"], dtype=float)[:, :, j]

    baseline_vat = expand_function(tax_context["vat_percent"], dates)
    baseline_excise = expand_function(tax_context["excise"], dates)
    scenario_vat, scenario_excise = _apply_monthly_scenario(
        baseline_vat,
        baseline_excise,
        tax_scenario,
        excise_unit=str(tax_context["excise_unit"]),
    )
    gamma = float(tax_context["gamma"])
    baseline = reattribute_function(
        pre_tax,
        gamma,
        baseline_vat.to_numpy(dtype=float)[None, :],
        baseline_excise.to_numpy(dtype=float)[None, :],
    )
    scenario = reattribute_function(
        pre_tax,
        gamma,
        scenario_vat.to_numpy(dtype=float)[None, :],
        scenario_excise.to_numpy(dtype=float)[None, :],
    )
    return {
        "dates": dates,
        "baseline": np.asarray(baseline, dtype=float),
        "scenario": np.asarray(scenario, dtype=float),
    }


def _direct_hicp_paths(forecast: Mapping, target: str) -> dict:
    variables = list(forecast["variables"])
    if target not in variables:
        raise AggregateError(f"{target!r} absent from {variables}.")
    j = variables.index(target)
    paths = np.asarray(forecast["level_paths"], dtype=float)[:, :, j]
    return {
        "dates": pd.DatetimeIndex(forecast["path_dates"], name="date"),
        "baseline": paths,
        "scenario": paths,
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregateOutcome:
    """Everything the dashboard needs from one aggregation run."""

    vintage: str
    forecast_name: str
    aggregate_run_id: str
    directory: Path | None
    dates: pd.DatetimeIndex
    weekly_tax_mode_effective: str
    n_aggregate_draws_requested: int
    n_aggregate_draws_effective: int
    component_pool_sizes: dict[str, int]
    weekly_rejection_rates: dict[str, float]
    baseline: Mapping
    scenario: Mapping
    baseline_yoy: Mapping
    scenario_yoy: Mapping
    historical_validation: Mapping
    model_history: Mapping
    proxy_diagnostics: Mapping
    max_additivity_error: float
    config: Mapping
    scenario_active: bool

    def audit(self) -> pd.Series:
        """The same audit table notebook 10 prints at the end."""
        return pd.Series(
            {
                "processed vintage": self.vintage,
                "forecast contract": self.forecast_name,
                "all annual re-anchored historical hard gates": bool(
                    self.historical_validation["summary"][
                        "annual_reanchored_pass"
                    ].all()
                ),
                "all cumulative reconstruction watch limits": bool(
                    self.historical_validation["summary"]["cumulative_watch_pass"].all()
                ),
                "six-model annual re-anchored hard gate": bool(
                    self.model_history["summary"]["annual_reanchored_pass"].iloc[0]
                ),
                "six-model cumulative watch": bool(
                    self.model_history["summary"]["cumulative_watch_pass"].iloc[0]
                ),
                "six-model cumulative max error": float(
                    self.model_history["summary"]["max_abs_error"].iloc[0]
                ),
                "six-model annual re-anchored max error": float(
                    self.model_history["summary"][
                        "annual_reanchored_max_abs_error"
                    ].iloc[0]
                ),
                "maximum draw-wise contribution additivity error": (
                    self.max_additivity_error
                ),
                "weekly tax mode effective": self.weekly_tax_mode_effective,
                "tax scenario active": self.scenario_active,
                "aggregate path start": self.dates.min(),
                "aggregate path end": self.dates.max(),
                "aggregate posterior draws": self.n_aggregate_draws_effective,
                "aggregate output directory": (
                    "not saved" if self.directory is None else str(self.directory)
                ),
            }
        )

    def fan(self, kind: str = "level", basis: str = "baseline") -> pd.DataFrame:
        """Quantile summary computed draw-wise, never from component medians."""
        source = {
            ("level", "baseline"): (self.baseline, "level_paths"),
            ("level", "scenario"): (self.scenario, "level_paths"),
            ("yoy", "baseline"): (self.baseline_yoy, "yoy_paths"),
            ("yoy", "scenario"): (self.scenario_yoy, "yoy_paths"),
        }
        try:
            store, key = source[(kind, basis)]
        except KeyError as exc:
            raise AggregateError(
                f"Unknown fan request kind={kind!r} basis={basis!r}."
            ) from exc
        return summarise_draw_paths(np.asarray(store[key], dtype=float), self.dates)


# ---------------------------------------------------------------------------
# Aggregate vintage contract
# ---------------------------------------------------------------------------


_AGGREGATION_SIDECARS = (
    "manifest.json",
    "hicp_indices_monthly.csv",
    "hicp_weights_annual.csv",
    "hicp_series_metadata.csv",
    "hicp_flags.csv",
    "hicp_weight_identity_diagnostics.csv",
)


def _processed_vintage_missing(path: Path) -> list[str]:
    """Files required before a processed vintage can enter Notebook-10 logic."""
    required = {
        *_AGGREGATION_SIDECARS,
        *(model_spec(name).dataset_file for name in CANONICAL_MODEL_IDS),
    }
    return sorted(name for name in required if not (path / name).is_file())


def resolve_aggregate_vintage(
    vintage: str | None = None,
    *,
    project_root: Path | str | None = None,
    results_root: Path | str | None = None,
    forecast_name: str = "unconditional",
    run_ids: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Path]]:
    """Resolve the processed vintage and all seven forecast stores together.

    With ``vintage=None`` this follows Notebook 10 FINAL_v2: scan processed
    vintages from newest to oldest and select the latest one that has all model
    datasets, all HICP aggregation sidecars and all seven saved forecast stores.

    Unlike the notebook's old modification-time rule, an ambiguous component
    run is never guessed. If several runs for the same model/vintage carry the
    requested forecast, the caller must pass ``run_ids`` (normally the promoted
    run IDs from the dashboard registry).
    """
    root = find_project_root(project_root)
    results = root / "results" if results_root is None else Path(results_root)
    processed_root = root / "data" / "processed"
    if not processed_root.is_dir():
        raise AggregateError(f"Processed root does not exist: {processed_root}")

    def resolve_one(candidate: str) -> tuple[str, dict[str, Path]]:
        candidate = resolve_common_vintage(
            candidate, models=CANONICAL_MODEL_IDS, project_root=root
        )
        missing = _processed_vintage_missing(processed_root / candidate)
        if missing:
            raise AggregateError(
                f"Vintage {candidate!r} is missing aggregate input(s): {missing}."
            )
        stores = resolve_forecast_stores(
            candidate,
            results_root=results,
            forecast_name=forecast_name,
            run_ids=run_ids,
        )
        return candidate, stores

    if vintage is not None:
        return resolve_one(str(vintage))

    candidates = sorted(
        path.name for path in processed_root.iterdir() if path.is_dir()
    )
    if not candidates:
        raise AggregateError(f"No processed vintage exists below {processed_root}.")

    diagnostics: list[str] = []
    for candidate in reversed(candidates):
        path = processed_root / candidate
        missing = _processed_vintage_missing(path)
        if missing:
            diagnostics.append(f"{candidate}: missing {missing}")
            continue
        try:
            # Dataset completeness is checked explicitly so error messages remain
            # consistent with the component pipeline.
            resolve_common_vintage(
                candidate, models=CANONICAL_MODEL_IDS, project_root=root
            )
            stores = resolve_forecast_stores(
                candidate,
                results_root=results,
                forecast_name=forecast_name,
                run_ids=run_ids,
            )
        except PipelineError as exc:
            message = str(exc)
            # Multiple qualifying runs are not "missing forecasts". Falling
            # back to an older vintage would hide a registry-selection problem.
            if "Pass run_id explicitly" in message or "runs carry a" in message:
                raise AggregateError(message) from exc
            diagnostics.append(f"{candidate}: {message}")
            continue
        return candidate, stores

    detail = "\n  - ".join(diagnostics[-8:])
    raise AggregateError(
        "No processed vintage has the complete HICP aggregation contract and "
        f"all seven {forecast_name!r} forecast stores. Checked:\n  - {detail}"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------



def _component_yoy_paths(
    history: pd.Series,
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    """Year-on-year rates for one component's HICP index paths.

    The twelve-month lag can fall either inside the published history (short
    horizons) or inside the forecast window itself (horizons beyond a year), so
    history and path are concatenated on one monthly calendar before shifting.
    Rates are derived from the index, never averaged from component rates.
    """
    paths = np.asarray(paths, dtype=float)
    observed = pd.Series(history, dtype=float).dropna().sort_index()
    if observed.empty:
        return np.full(paths.shape, np.nan, dtype=float)

    full = pd.date_range(
        min(observed.index.min(), dates.min()), dates.max(), freq="MS", name="date"
    )
    base = observed.reindex(full).to_numpy(dtype=float)
    positions = full.get_indexer(dates)
    if (positions < 0).any():
        raise AggregateError(
            "Component forecast dates are absent from the monthly calendar."
        )

    combined = np.repeat(base[None, :], paths.shape[0], axis=0)
    combined[:, positions] = paths

    lagged = np.full_like(combined, np.nan)
    lagged[:, 12:] = combined[:, :-12]
    with np.errstate(divide="ignore", invalid="ignore"):
        yoy = 100.0 * (combined / lagged - 1.0)
    return yoy[:, positions]


def run_aggregate(
    vintage: str | None = None,
    *,
    project_root: Path | str | None = None,
    results_root: Path | str | None = None,
    forecast_name: str = "unconditional",
    run_ids: Mapping[str, str] | None = None,
    n_aggregate_draws: int = 500,
    pairing_seed: int = 2026,
    tax_scenarios: Mapping[str, Mapping | None] | None = None,
    weekly_tax_mode: str = "strict",
    allow_weak_pretax_proxy_fallback: bool = False,
    max_weekly_rejection_rate: float | None = None,
    persist: bool = True,
    overwrite: bool = False,
) -> AggregateOutcome:
    """Aggregate the seven component forecasts into HICP Energy.

    Mirrors notebook 10 (FINAL_v2) step for step. It validates the historical
    aggregation identities first, so a broken vintage fails before any
    predictive path is touched.
    """
    if weekly_tax_mode not in {"strict", "pre_tax_proxy_fallback"}:
        raise AggregateError(
            "weekly_tax_mode must be 'strict' or 'pre_tax_proxy_fallback'."
        )
    if allow_weak_pretax_proxy_fallback:
        raise AggregateError(
            "allow_weak_pretax_proxy_fallback is notebook-only and cannot be "
            "enabled in the production aggregate pipeline."
        )
    if weekly_tax_mode != "strict":
        raise AggregateError(
            "run_aggregate is the production aggregation path and requires "
            "weekly_tax_mode='strict'. The pre-tax WOB proxy remains an "
            "exploratory notebook-only fallback."
        )

    root = find_project_root(project_root)
    if results_root is None:
        results_root = root / "results"
    results_root = Path(results_root)

    # -- 1. vintage, stores, processed directory ---------------------------
    vintage, stores = resolve_aggregate_vintage(
        vintage,
        project_root=root,
        results_root=results_root,
        forecast_name=forecast_name,
        run_ids=run_ids,
    )
    processed_dir = root / "data" / "processed" / vintage
    forecasts = {name: load_energy_bvar_forecast(path) for name, path in stores.items()}

    tax_scenarios = dict(tax_scenarios or {})
    allowed_scenarios = {*WEEKLY_MODELS, "gas", "electricity"}
    unknown_scenarios = sorted(set(tax_scenarios).difference(allowed_scenarios))
    if unknown_scenarios:
        raise AggregateError(
            f"Unknown tax-scenario component(s): {unknown_scenarios}. "
            f"Allowed: {sorted(allowed_scenarios)}."
        )
    scenario_active = any(value is not None for value in tax_scenarios.values())

    # -- 2. historical aggregation gate ------------------------------------
    inputs = load_aggregation_inputs(processed_dir)
    indices = inputs["indices"]
    weights = inputs["weights"]

    historical_validation = historical_reconstruction_suite(processed_dir)
    model_history = historical_model_component_reconstruction(processed_dir)
    assert_reconstruction_suite(historical_validation)
    assert_reconstruction_suite(model_history)

    # -- 3. gas and electricity: exact tax re-attribution ------------------
    gas_dataset = processed_dir / model_spec("gas").dataset_file
    electricity_dataset = processed_dir / model_spec("electricity").dataset_file
    gas_context = load_gas_tax_context(gas_dataset)
    electricity_context = load_electricity_tax_context(electricity_dataset)
    round_trips = {
        "gas": validate_gas_tax_round_trip(gas_context),
        "electricity": validate_electricity_tax_round_trip(electricity_context),
    }

    gas_hicp = _monthly_pretax_forecast_to_hicp(
        forecasts["gas"],
        target_variable="gas_pre_tax",
        tax_context=gas_context,
        expand_function=expand_gas_semester_series,
        reattribute_function=reattribute_gas_taxes,
        tax_scenario=tax_scenarios.get("gas"),
    )
    electricity_hicp = _monthly_pretax_forecast_to_hicp(
        forecasts["electricity"],
        target_variable="electricity_pre_tax",
        tax_context=electricity_context,
        expand_function=expand_electricity_semester_series,
        reattribute_function=reattribute_electricity_taxes,
        tax_scenario=tax_scenarios.get("electricity"),
    )

    # -- 4. heat energy and solid fuels are already HICP indices -----------
    monthly_direct = {
        name: _direct_hicp_paths(forecasts[name], target)
        for name, target in MONTHLY_DIRECT.items()
    }

    # -- 5. weekly fuels ---------------------------------------------------
    weekly_contexts: dict[str, Mapping] = {}
    weekly_errors: dict[str, str] = {}
    for model in WEEKLY_MODELS:
        dataset = processed_dir / model_spec(model).dataset_file
        try:
            weekly_contexts[model] = load_weekly_tax_context(dataset, model)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            weekly_errors[model] = str(exc)

    if weekly_errors:
        raise AggregateError(
            "Strict mode requires the separate WOB VAT and IDT series for "
            "petrol, diesel and liquid fuels. Source contract incomplete:\n  - "
            + "\n  - ".join(f"{k}: {v}" for k, v in weekly_errors.items())
        )
    weekly_tax_mode_effective = "official_wob_tax_reattribution"

    weekly_hicp_paths: dict[str, dict] = {}
    weekly_rejection: dict[str, float] = {}
    weekly_filter_diagnostics: dict[str, dict] = {}
    proxy_diags: dict[str, object] = {}

    for model in WEEKLY_MODELS:
        forecast = forecasts[model]
        context = weekly_contexts[model]
        baseline_taxed = reattribute_weekly_taxes(forecast, context)
        scenario_taxed = reattribute_weekly_taxes(
            forecast, context, tax_scenario=tax_scenarios.get(model)
        )

        # The BVAR forecasts the PRE-TAX price: it must be admissible on its
        # own. Positive taxes must not rescue a negative pre-tax draw.
        admissibility = {
            "pre_tax_weekly": np.asarray(
                baseline_taxed["pre_tax_level_paths"], dtype=float
            ),
            "baseline_after_tax_weekly": np.asarray(
                baseline_taxed["post_tax_level_paths"], dtype=float
            ),
            "scenario_after_tax_weekly": np.asarray(
                scenario_taxed["post_tax_level_paths"], dtype=float
            ),
        }

        baseline_monthly, monthly_dates, baseline_coverage = (
            weekly_paths_with_history_to_monthly_mean(
                baseline_taxed["post_tax_level_paths"],
                baseline_taxed["path_dates"],
                context["data"]["after_tax"],
                return_coverage=True,
            )
        )
        scenario_monthly, scenario_dates, scenario_coverage = (
            weekly_paths_with_history_to_monthly_mean(
                scenario_taxed["post_tax_level_paths"],
                scenario_taxed["path_dates"],
                context["data"]["after_tax"],
                return_coverage=True,
            )
        )
        if not monthly_dates.equals(scenario_dates):
            raise AggregateError(f"{model}: baseline/scenario monthly calendars differ.")
        if not baseline_coverage.equals(scenario_coverage):
            raise AggregateError(f"{model}: baseline/scenario monthly coverage differs.")

        historical_price = context["data"]["after_tax"]
        historical_monthly_price = _monthly_mean_series(historical_price)
        hicp_history = indices[WEEKLY_HICP_COLUMNS[model]]
        proxy_diags[model] = price_proxy_diagnostics(
            historical_monthly_price,
            hicp_history,
            label=f"{model}: after-tax WOB consumer price vs HICP",
        )

        price_filter = filter_positive_price_draws(
            baseline_monthly,
            scenario_monthly,
            admissibility_paths=admissibility,
            max_rejection_rate=max_weekly_rejection_rate,
            label=model,
        )
        baseline_monthly = price_filter["baseline_paths"]
        scenario_monthly = price_filter["scenario_paths"]
        weekly_filter_diagnostics[model] = price_filter["diagnostics"]
        weekly_rejection[model] = float(
            price_filter["diagnostics"]["rejection_rate"]
        )

        baseline_rebased = rebase_price_paths_to_hicp_index(
            baseline_monthly, monthly_dates, historical_monthly_price, hicp_history
        )
        scenario_rebased = rebase_price_paths_to_hicp_index(
            scenario_monthly, scenario_dates, historical_monthly_price, hicp_history
        )
        if not baseline_rebased["dates"].equals(scenario_rebased["dates"]):
            raise AggregateError(f"{model}: baseline/scenario HICP-proxy calendars differ.")
        if baseline_rebased["anchor_date"] != scenario_rebased["anchor_date"]:
            raise AggregateError(f"{model}: baseline/scenario HICP-proxy anchors differ.")

        weekly_hicp_paths[model] = {
            "dates": baseline_rebased["dates"],
            "baseline": baseline_rebased["index_paths"],
            "scenario": scenario_rebased["index_paths"],
            "anchor_date": baseline_rebased["anchor_date"],
            "anchor_price": baseline_rebased.get("anchor_price"),
            "anchor_hicp": baseline_rebased.get("anchor_hicp"),
            "method": baseline_rebased["method"],
            "retained_original_draw_indices": np.asarray(
                price_filter["draw_indices"], dtype=int
            ),
        }

    # -- 6. car-fuels sub-aggregate (first stage) --------------------------
    transport_dates = _common_monthly_dates(
        [weekly_hicp_paths["petrol"], weekly_hicp_paths["diesel"]]
    )
    petrol_base = _align_paths(weekly_hicp_paths["petrol"], "baseline", transport_dates)
    diesel_base = _align_paths(weekly_hicp_paths["diesel"], "baseline", transport_dates)
    petrol_scen = _align_paths(weekly_hicp_paths["petrol"], "scenario", transport_dates)
    diesel_scen = _align_paths(weekly_hicp_paths["diesel"], "scenario", transport_dates)

    transport_pool_sizes = {
        "petrol": int(petrol_base.shape[0]),
        "diesel": int(diesel_base.shape[0]),
    }
    n_transport = min(
        int(n_aggregate_draws), *transport_pool_sizes.values()
    )
    if n_transport < 1:
        raise AggregateError("No admissible petrol/diesel draws remain for car fuels.")

    other_transport = carry_last_index_path(
        indices["hicp_other_transport_fuels"], transport_dates, n_draws=n_transport
    )
    transport_paired, transport_indices = independent_draw_pairing(
        {
            "hicp_petrol": petrol_base,
            "hicp_diesel": diesel_base,
            "hicp_other_transport_fuels": other_transport,
        },
        n_draws=n_transport,
        seed=pairing_seed,
    )
    transport_scenario_paired = {
        "hicp_petrol": petrol_scen[transport_indices["hicp_petrol"]],
        "hicp_diesel": diesel_scen[transport_indices["hicp_diesel"]],
        "hicp_other_transport_fuels": other_transport[
            transport_indices["hicp_other_transport_fuels"]
        ],
    }

    transport_history_index = pd.concat(
        [
            pd.Series([100.0], index=pd.DatetimeIndex([TRANSPORT_ANCHOR], name="date")),
            model_history["transport_energy"]["index"],
        ]
    ).sort_index()
    transport_history_index.name = "car_fuels"
    transport_component_history = indices[list(TRANSPORT_COMPONENTS)]
    transport_weights = weights[list(TRANSPORT_COMPONENTS)]

    car_fuels_baseline = aggregate_component_draw_paths_laspeyres(
        transport_component_history,
        transport_paired,
        transport_dates,
        transport_weights,
        transport_history_index,
        components=TRANSPORT_COMPONENTS,
    )
    car_fuels_scenario = aggregate_component_draw_paths_laspeyres(
        transport_component_history,
        transport_scenario_paired,
        transport_dates,
        transport_weights,
        transport_history_index,
        components=TRANSPORT_COMPONENTS,
    )

    # -- 7. six-component aggregation (second stage) -----------------------
    component_objects = {
        "car_fuels": {
            "dates": transport_dates,
            "baseline": car_fuels_baseline["level_paths"],
            "scenario": car_fuels_scenario["level_paths"],
        },
        "liquid_fuels": weekly_hicp_paths["liquid_fuels"],
        "gas": gas_hicp,
        "electricity": electricity_hicp,
        "heat_energy": monthly_direct["heat_energy"],
        "solid_fuels": monthly_direct["solid_fuels"],
    }
    aggregate_dates = _common_monthly_dates(list(component_objects.values()))
    baseline_unpaired = {
        name: _align_paths(obj, "baseline", aggregate_dates)
        for name, obj in component_objects.items()
    }
    scenario_unpaired = {
        name: _align_paths(obj, "scenario", aggregate_dates)
        for name, obj in component_objects.items()
    }
    final_component_pool_sizes = {
        name: int(paths.shape[0]) for name, paths in baseline_unpaired.items()
    }
    n_effective = min(
        int(n_aggregate_draws), *final_component_pool_sizes.values()
    )
    if n_effective < 1:
        raise AggregateError("No common admissible draw pool remains for HICP Energy.")

    baseline_components, final_indices = independent_draw_pairing(
        baseline_unpaired, n_draws=n_effective, seed=pairing_seed + 1
    )
    scenario_components = {
        name: scenario_unpaired[name][final_indices[name]]
        for name in baseline_components
    }

    # Per-component HICP index and YoY paths. run_aggregate is the only place
    # where the tax bridge, the weekly-to-monthly conversion and the HICP
    # rebasing have all been applied, so these cannot be reconstructed from a
    # component forecast store alone. Persisting them is what lets the
    # dashboard show HICP inflation for gas, electricity and the fuels, whose
    # BVARs forecast a pre-tax price rather than an index.
    component_names = list(baseline_components)
    component_history_frame = model_history["component_history"]
    component_index_paths = np.stack(
        [baseline_components[name] for name in component_names], axis=-1
    )
    component_scenario_index_paths = np.stack(
        [scenario_components[name] for name in component_names], axis=-1
    )
    component_yoy_paths = np.stack(
        [
            _component_yoy_paths(
                component_history_frame[name], baseline_components[name], aggregate_dates
            )
            for name in component_names
        ],
        axis=-1,
    )
    component_scenario_yoy_paths = np.stack(
        [
            _component_yoy_paths(
                component_history_frame[name], scenario_components[name], aggregate_dates
            )
            for name in component_names
        ],
        axis=-1,
    )

    # Seven model-specific HICP paths for the Forecast page.  The six-component
    # arrays above intentionally contain the final car-fuels aggregate.  For the
    # model selector, however, petrol and diesel must remain individually
    # available.  Trace the final car-fuels draw selection back through the
    # first-stage transport pairing so every model-specific path is aligned to
    # the same n_effective draws and monthly aggregate calendar.
    transport_positions = transport_dates.get_indexer(aggregate_dates)
    if (transport_positions < 0).any():
        raise AggregateError(
            "Aggregate dates are not contained in the petrol/diesel monthly calendar."
        )
    final_car_indices = np.asarray(final_indices["car_fuels"], dtype=int)

    model_hicp_names = list(MODEL_HICP_NAMES)
    model_hicp_baseline = {
        "car_fuels_petrol": np.asarray(
            transport_paired["hicp_petrol"], dtype=float
        )[final_car_indices][:, transport_positions],
        "car_fuels_diesel": np.asarray(
            transport_paired["hicp_diesel"], dtype=float
        )[final_car_indices][:, transport_positions],
        "liquid_fuels": baseline_components["liquid_fuels"],
        "gas": baseline_components["gas"],
        "electricity": baseline_components["electricity"],
        "heat_energy": baseline_components["heat_energy"],
        "solid_fuels": baseline_components["solid_fuels"],
    }
    model_hicp_scenario = {
        "car_fuels_petrol": np.asarray(
            transport_scenario_paired["hicp_petrol"], dtype=float
        )[final_car_indices][:, transport_positions],
        "car_fuels_diesel": np.asarray(
            transport_scenario_paired["hicp_diesel"], dtype=float
        )[final_car_indices][:, transport_positions],
        "liquid_fuels": scenario_components["liquid_fuels"],
        "gas": scenario_components["gas"],
        "electricity": scenario_components["electricity"],
        "heat_energy": scenario_components["heat_energy"],
        "solid_fuels": scenario_components["solid_fuels"],
    }
    model_hicp_history = {
        "car_fuels_petrol": indices["hicp_petrol"],
        "car_fuels_diesel": indices["hicp_diesel"],
        "liquid_fuels": component_history_frame["liquid_fuels"],
        "gas": component_history_frame["gas"],
        "electricity": component_history_frame["electricity"],
        "heat_energy": component_history_frame["heat_energy"],
        "solid_fuels": component_history_frame["solid_fuels"],
    }
    model_hicp_index_paths = np.stack(
        [model_hicp_baseline[name] for name in model_hicp_names], axis=-1
    )
    model_hicp_scenario_index_paths = np.stack(
        [model_hicp_scenario[name] for name in model_hicp_names], axis=-1
    )
    model_hicp_yoy_paths = np.stack(
        [
            _component_yoy_paths(
                model_hicp_history[name], model_hicp_baseline[name], aggregate_dates
            )
            for name in model_hicp_names
        ],
        axis=-1,
    )
    model_hicp_scenario_yoy_paths = np.stack(
        [
            _component_yoy_paths(
                model_hicp_history[name], model_hicp_scenario[name], aggregate_dates
            )
            for name in model_hicp_names
        ],
        axis=-1,
    )

    energy_baseline = aggregate_draw_paths_laspeyres(
        model_history["component_history"],
        baseline_components,
        aggregate_dates,
        weights,
        indices["hicp_energy"],
    )
    energy_scenario = aggregate_draw_paths_laspeyres(
        model_history["component_history"],
        scenario_components,
        aggregate_dates,
        weights,
        indices["hicp_energy"],
    )
    energy_baseline_yoy = draw_yoy_and_contributions(
        energy_baseline, model_history["energy"]
    )
    energy_scenario_yoy = draw_yoy_and_contributions(
        energy_scenario, model_history["energy"]
    )

    # -- 8. draw-wise additivity of contributions --------------------------
    contributions = np.asarray(
        energy_baseline_yoy["contribution_paths"], dtype=float
    )
    yoy = np.asarray(energy_baseline_yoy["yoy_paths"], dtype=float)
    additivity_error = contributions.sum(axis=-1) - yoy if contributions.ndim == 3 else None
    if additivity_error is None:
        max_additivity_error = float("nan")
    else:
        finite = np.isfinite(additivity_error)
        max_additivity_error = (
            float(np.max(np.abs(additivity_error[finite]))) if finite.any() else 0.0
        )
    if np.isfinite(max_additivity_error) and max_additivity_error >= 1e-10:
        raise AggregateError(
            "Draw-wise contributions do not add to aggregate YoY inflation; "
            f"maximum error {max_additivity_error:.3e}."
        )

    scenario_impact_paths = (
        np.asarray(energy_scenario_yoy["yoy_paths"], dtype=float)
        - np.asarray(energy_baseline_yoy["yoy_paths"], dtype=float)
    )

    # -- 9. persist --------------------------------------------------------
    config = {
        "vintage": str(vintage),
        "forecast_name": str(forecast_name),
        "aggregate_module_version": AGGREGATE_MODULE_VERSION,
        "aggregate_pipeline_version": AGGREGATE_PIPELINE_VERSION,
        "n_aggregate_draws_requested": int(n_aggregate_draws),
        "n_aggregate_draws_effective": int(n_effective),
        "transport_draws_effective": int(n_transport),
        "pairing_seed": int(pairing_seed),
        "weekly_tax_mode_requested": weekly_tax_mode,
        "weekly_tax_mode_effective": weekly_tax_mode_effective,
        "tax_scenarios": {
            k: (None if v is None else dict(v)) for k, v in tax_scenarios.items()
        },
        "component_forecast_stores": {k: str(v) for k, v in stores.items()},
        "wob_hicp_proxy_diagnostics": {
            name: diagnostic.as_dict() for name, diagnostic in proxy_diags.items()
        },
        "weekly_price_admissibility": weekly_filter_diagnostics,
        "weekly_rejection_rates": weekly_rejection,
        "final_component_pool_sizes": final_component_pool_sizes,
        "transport_pool_sizes": transport_pool_sizes,
        "predictive_distribution_interpretation": (
            "posterior predictive conditional on finite, strictly positive weekly "
            "pre-tax and after-tax consumer-price paths for the weekly fuel models"
        ),
        "weekly_price_to_hicp_method": (
            "proportional WOB price-relative rebasing to latest common HICP month"
        ),
        "weight_policy": energy_baseline.get("future_weight_policy"),
        "cross_model_dependence": (
            "independent random pairing of marginal component predictive paths; "
            "this is not a joint posterior across the seven separately "
            "estimated BVARs"
        ),
        "other_transport_fuels_forecast": (
            "latest published HICP level held constant over the forecast horizon"
        ),
    }
    aggregate_run_id = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:20]

    directory: Path | None = None
    if persist:
        directory = (
            results_root / "hicp_energy_aggregate" / str(vintage) / aggregate_run_id
        )
        if directory.exists() and not overwrite:
            raise AggregateError(
                f"Aggregate store already exists at {directory}. "
                "Pass overwrite=True to replace it."
            )
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "aggregate_draws.npz",
            baseline_level_paths=np.asarray(energy_baseline["level_paths"], dtype=float),
            baseline_yoy_paths=np.asarray(energy_baseline_yoy["yoy_paths"], dtype=float),
            baseline_contribution_paths=contributions,
            scenario_level_paths=np.asarray(energy_scenario["level_paths"], dtype=float),
            scenario_yoy_paths=np.asarray(energy_scenario_yoy["yoy_paths"], dtype=float),
            scenario_contribution_paths=np.asarray(
                energy_scenario_yoy["contribution_paths"], dtype=float
            ),
            scenario_impact_yoy_paths=np.asarray(scenario_impact_paths, dtype=float),
            aggregate_dates=aggregate_dates.astype("datetime64[ns]").to_numpy(),
            component_index_paths=component_index_paths,
            component_scenario_index_paths=component_scenario_index_paths,
            component_yoy_paths=component_yoy_paths,
            component_scenario_yoy_paths=component_scenario_yoy_paths,
            model_hicp_index_paths=model_hicp_index_paths,
            model_hicp_scenario_index_paths=model_hicp_scenario_index_paths,
            model_hicp_yoy_paths=model_hicp_yoy_paths,
            model_hicp_scenario_yoy_paths=model_hicp_scenario_yoy_paths,
            petrol_retained_original_draw_indices=np.asarray(
                weekly_hicp_paths["petrol"]["retained_original_draw_indices"], dtype=int
            ),
            diesel_retained_original_draw_indices=np.asarray(
                weekly_hicp_paths["diesel"]["retained_original_draw_indices"], dtype=int
            ),
            liquid_fuels_retained_original_draw_indices=np.asarray(
                weekly_hicp_paths["liquid_fuels"]["retained_original_draw_indices"], dtype=int
            ),
            transport_petrol_pair_indices=np.asarray(
                transport_indices["hicp_petrol"], dtype=int
            ),
            transport_diesel_pair_indices=np.asarray(
                transport_indices["hicp_diesel"], dtype=int
            ),
            final_car_fuels_pair_indices=np.asarray(
                final_indices["car_fuels"], dtype=int
            ),
            final_liquid_fuels_pair_indices=np.asarray(
                final_indices["liquid_fuels"], dtype=int
            ),
            final_gas_pair_indices=np.asarray(final_indices["gas"], dtype=int),
            final_electricity_pair_indices=np.asarray(
                final_indices["electricity"], dtype=int
            ),
            final_heat_energy_pair_indices=np.asarray(
                final_indices["heat_energy"], dtype=int
            ),
            final_solid_fuels_pair_indices=np.asarray(
                final_indices["solid_fuels"], dtype=int
            ),
        )
        metadata = {
            **config,
            "aggregate_run_id": aggregate_run_id,
            "path_dates": [date.isoformat() for date in aggregate_dates],
            "components": list(energy_baseline["components"]),
            "component_index_names": component_names,
            "model_hicp_names": model_hicp_names,
            "weight_year_used": {
                pd.Timestamp(date).isoformat(): int(year)
                for date, year in energy_baseline["weight_year_used"].items()
            },
            "historical_validation_max_errors": {
                name: float(value)
                for name, value in historical_validation["summary"][
                    "max_abs_error"
                ].items()
            },
            "six_model_history_max_error": float(
                model_history["summary"]["max_abs_error"].iloc[0]
            ),
            "six_model_history_annual_reanchored_max_error": float(
                model_history["summary"][
                    "annual_reanchored_max_abs_error"
                ].iloc[0]
            ),
            "maximum_drawwise_contribution_additivity_error": (
                max_additivity_error
            ),
            "scenario_active": bool(scenario_active),
            "tax_round_trip_max_errors": {
                name: float(series["maximum_absolute_error"])
                for name, series in round_trips.items()
            },
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        # Keep the compact config file for backward compatibility with the first
        # dashboard prototype; metadata.json is the authoritative store.
        (directory / "aggregate_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

        summarise_draw_paths(
            energy_baseline["level_paths"], aggregate_dates
        ).to_csv(directory / "baseline_level_fan.csv")
        summarise_draw_paths(
            energy_baseline_yoy["yoy_paths"], aggregate_dates
        ).to_csv(directory / "baseline_yoy_fan.csv")
        summarise_draw_paths(
            energy_scenario["level_paths"], aggregate_dates
        ).to_csv(directory / "scenario_level_fan.csv")
        summarise_draw_paths(
            energy_scenario_yoy["yoy_paths"], aggregate_dates
        ).to_csv(directory / "scenario_yoy_fan.csv")
        summarise_draw_paths(
            scenario_impact_paths, aggregate_dates
        ).to_csv(directory / "scenario_impact_yoy_fan.csv")
        pd.DataFrame(
            np.nanmedian(contributions, axis=0),
            index=aggregate_dates,
            columns=energy_baseline_yoy["components"],
        ).to_csv(directory / "median_component_contributions.csv")
        price_proxy_diagnostic_table(proxy_diags).to_csv(
            directory / "weekly_wob_hicp_proxy_diagnostics.csv"
        )
        pd.DataFrame(weekly_filter_diagnostics).T.to_csv(
            directory / "weekly_price_admissibility_diagnostics.csv"
        )
        historical_validation["summary"].to_csv(
            directory / "historical_reconstruction_summary.csv"
        )

    return AggregateOutcome(
        vintage=str(vintage),
        forecast_name=str(forecast_name),
        aggregate_run_id=aggregate_run_id,
        directory=directory,
        dates=aggregate_dates,
        weekly_tax_mode_effective=weekly_tax_mode_effective,
        n_aggregate_draws_requested=int(n_aggregate_draws),
        n_aggregate_draws_effective=int(n_effective),
        component_pool_sizes=final_component_pool_sizes,
        weekly_rejection_rates=weekly_rejection,
        baseline=energy_baseline,
        scenario=energy_scenario,
        baseline_yoy=energy_baseline_yoy,
        scenario_yoy=energy_scenario_yoy,
        historical_validation=historical_validation,
        model_history=model_history,
        proxy_diagnostics=proxy_diags,
        max_additivity_error=max_additivity_error,
        config=config,
        scenario_active=scenario_active,
    )


def _run_id_from_forecast_store(path_value: object) -> str | None:
    """Extract <run_id> from .../<run_id>/forecasts/<forecast_name>."""
    text = str(path_value or "").replace("\\", "/").rstrip("/")
    parts = [piece for piece in text.split("/") if piece]
    if len(parts) >= 3 and parts[-2] == "forecasts":
        return parts[-3]
    return None


def _aggregate_metadata_matches(
    metadata: Mapping,
    *,
    forecast_name: str,
    run_ids: Mapping[str, str],
) -> bool:
    """Return True only for a store produced by the current seven-HICP contract."""
    if str(metadata.get("aggregate_pipeline_version", "")) != AGGREGATE_PIPELINE_VERSION:
        return False
    if str(metadata.get("forecast_name", "unconditional")) != str(forecast_name):
        return False
    names = tuple(str(name) for name in metadata.get("model_hicp_names", ()))
    if names != MODEL_HICP_NAMES:
        return False
    if bool(metadata.get("scenario_active", False)):
        return False

    stores = dict(metadata.get("component_forecast_stores", {}) or {})
    for model_id in CANONICAL_MODEL_IDS:
        spec = model_spec(model_id)
        stored_run_id = _run_id_from_forecast_store(stores.get(spec.aggregate_key))
        if stored_run_id != str(run_ids.get(model_id, "")):
            return False
    return True


def _authoritative_run_ids_for_vintage(
    registry_path: Path,
    vintage: str,
    *,
    forecast_name: str,
) -> dict[str, str]:
    """Use promoted runs; auto-promote only when exactly one usable run exists."""
    from energy_bvar_registry import list_forecasts, promote_run, promoted_run_ids

    chosen = promoted_run_ids(
        registry_path,
        vintage,
        forecast_name=forecast_name,
        require_all=False,
        require_draws=False,
    )
    for model_id in CANONICAL_MODEL_IDS:
        if model_id in chosen:
            continue
        rows = list_forecasts(
            registry_path,
            model_id=model_id,
            vintage=vintage,
            forecast_name=forecast_name,
            valid_only=True,
            present_only=True,
        )
        run_values = sorted(set(rows["run_id"].astype(str))) if not rows.empty else []
        if len(run_values) == 0:
            raise AggregateError(
                f"{model_id}/{vintage}: no valid {forecast_name!r} forecast store."
            )
        if len(run_values) > 1:
            raise AggregateError(
                f"{model_id}/{vintage}: {len(run_values)} valid runs exist but none is "
                "promoted. Promote one run before automatic aggregation; refusing "
                "to choose by timestamp."
            )
        run_id = run_values[0]
        promote_run(
            registry_path,
            model_id,
            vintage,
            run_id,
            required_forecast=forecast_name,
            require_draws=False,
        )
        chosen[model_id] = run_id
    return chosen


def ensure_dashboard_aggregate(
    vintage: str | None = None,
    *,
    project_root: Path | str | None = None,
    results_root: Path | str | None = None,
    registry_path: Path | str | None = None,
    forecast_name: str = "unconditional",
    n_aggregate_draws: int = 500,
    pairing_seed: int = 2026,
) -> dict:
    """Ensure the neutral seven-model HICP bridge exists for the dashboard.

    No BVAR is estimated here. Existing component forecast stores are reused.
    The operation is idempotent: a compatible aggregate generated from the
    current authoritative component runs is reused; otherwise a new immutable
    aggregate store is created and promoted.
    """
    from energy_bvar_registry import (
        default_registry_path,
        list_aggregates,
        list_forecasts,
        promote_aggregate,
        scan_results,
    )

    root = find_project_root(project_root)
    results = root / "results" if results_root is None else Path(results_root)
    registry = (
        default_registry_path(results_root=results)
        if registry_path is None
        else Path(registry_path)
    )
    scan_results(results_root=results, registry_path=registry)

    if vintage is None:
        forecasts = list_forecasts(
            registry,
            forecast_name=forecast_name,
            valid_only=True,
            present_only=True,
        )
        if forecasts.empty:
            raise AggregateError(
                f"No valid {forecast_name!r} component forecasts are registered."
            )
        candidates = sorted(forecasts["vintage"].astype(str).unique(), reverse=True)
        resolved_vintage = None
        run_ids = None
        diagnostics: list[str] = []
        for candidate in candidates:
            present_models = set(
                forecasts.loc[
                    forecasts["vintage"].astype(str) == candidate, "model_id"
                ].astype(str)
            )
            missing_models = sorted(set(CANONICAL_MODEL_IDS).difference(present_models))
            if missing_models:
                diagnostics.append(f"{candidate}: missing forecasts for {missing_models}")
                continue
            missing_inputs = _processed_vintage_missing(
                root / "data" / "processed" / candidate
            )
            if missing_inputs:
                diagnostics.append(f"{candidate}: missing aggregate inputs {missing_inputs}")
                continue
            try:
                resolve_common_vintage(
                    candidate, models=CANONICAL_MODEL_IDS, project_root=root
                )
                candidate_run_ids = _authoritative_run_ids_for_vintage(
                    registry, candidate, forecast_name=forecast_name
                )
            except PipelineError as exc:
                if "none is promoted" in str(exc):
                    raise AggregateError(str(exc)) from exc
                diagnostics.append(f"{candidate}: {exc}")
                continue
            resolved_vintage = candidate
            run_ids = candidate_run_ids
            break
        if resolved_vintage is None or run_ids is None:
            detail = "\n  - ".join(diagnostics[-8:])
            raise AggregateError(
                "No vintage can support the automatic seven-model HICP bridge."
                + (f" Checked:\n  - {detail}" if detail else "")
            )
        vintage = resolved_vintage
    else:
        vintage = str(vintage)
        resolve_common_vintage(vintage, models=CANONICAL_MODEL_IDS, project_root=root)
        missing_inputs = _processed_vintage_missing(
            root / "data" / "processed" / vintage
        )
        if missing_inputs:
            raise AggregateError(
                f"Vintage {vintage!r} is missing aggregate input(s): {missing_inputs}."
            )
        run_ids = _authoritative_run_ids_for_vintage(
            registry, vintage, forecast_name=forecast_name
        )

    aggregates = list_aggregates(
        registry,
        vintage=vintage,
        forecast_name=forecast_name,
        status="complete",
        present_only=True,
    )
    for _, row in aggregates.iterrows():
        directory = Path(str(row["directory"]))
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _aggregate_metadata_matches(
            metadata, forecast_name=forecast_name, run_ids=run_ids
        ):
            aggregate_run_id = str(metadata.get("aggregate_run_id", directory.name))
            promote_aggregate(registry, vintage, aggregate_run_id)
            return {
                "vintage": vintage,
                "aggregate_run_id": aggregate_run_id,
                "directory": directory,
                "run_ids": dict(run_ids),
                "created": False,
                "pipeline_version": AGGREGATE_PIPELINE_VERSION,
            }

    outcome = run_aggregate(
        vintage=vintage,
        project_root=root,
        results_root=results,
        forecast_name=forecast_name,
        run_ids=run_ids,
        n_aggregate_draws=n_aggregate_draws,
        pairing_seed=pairing_seed,
        tax_scenarios=None,
        weekly_tax_mode="strict",
        persist=True,
        overwrite=False,
    )
    if outcome.directory is None:
        raise AggregateError("Automatic aggregate run completed without a persisted store.")

    scan_results(results_root=results, registry_path=registry)
    promote_aggregate(registry, vintage, outcome.aggregate_run_id)
    return {
        "vintage": vintage,
        "aggregate_run_id": outcome.aggregate_run_id,
        "directory": outcome.directory,
        "run_ids": dict(run_ids),
        "created": True,
        "pipeline_version": AGGREGATE_PIPELINE_VERSION,
    }


def assert_reference_audit(
    outcome: AggregateOutcome,
    *,
    reference: Mapping[str, float] = REFERENCE_AUDIT,
    rtol: float = 1e-3,
) -> pd.DataFrame:
    """Compare a run against the executed notebook 10 v2 reference figures.

    A mismatch is not automatically a bug -- a new vintage legitimately moves
    these numbers. It is a signal that something changed, and the table says
    what.
    """
    summary = outcome.model_history["summary"]
    observed = {
        "six_model_cumulative_max_error": float(summary["max_abs_error"].iloc[0]),
        "six_model_annual_reanchored_max_error": float(
            summary["annual_reanchored_max_abs_error"].iloc[0]
        ),
        "max_additivity_error": float(outcome.max_additivity_error),
        "n_aggregate_draws_effective": float(outcome.n_aggregate_draws_effective),
    }
    rows = []
    for key, expected in reference.items():
        got = observed[key]
        if expected == 0.0:
            ok = abs(got) <= 1e-10
        else:
            ok = abs(got - expected) <= rtol * abs(expected)
        rows.append(
            {"metric": key, "reference": expected, "observed": got, "match": bool(ok)}
        )
    return pd.DataFrame(rows).set_index("metric")
